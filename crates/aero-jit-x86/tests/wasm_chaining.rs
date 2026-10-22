//! Blocks of one module chaining to each other without returning to the host.
//!
//! The host pays a `TypedFunc::call` per chain step — trampoline, trap-handler
//! thread state, and the lookup between blocks — and a block retires about four
//! guest instructions, so that cost is divided by almost nothing. When the successor
//! is a block of the *same* module the dispatcher can call it directly.
//!
//! Two things have to be pinned. The module has to *validate*: an invalid one is not
//! a loud failure, it is a module that never loads and a tier that silently keeps
//! doing what it did before. And the chaining has to actually happen — which is what
//! a lookup that always declines proves, since without in-wasm chaining exactly one
//! block could run.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test wasm_chaining -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const A: u64 = 0x100;
const B: u64 = 0x200;
const MODULE_ID: u32 = 7;

/// `add eax, 1` then `jmp` to `target`, using a rel32 so the two can be far apart.
fn write_block(backend: &mut WasmtimeBackend<CpuState>, at: u64, target: u64) {
    let mut code = vec![0x83u8, 0xc0, 0x01, 0xe9];
    let next = at + 8;
    code.extend_from_slice(&((target as i64 - next as i64) as i32).to_le_bytes());
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(at + i as u64, *b);
    }
}

fn options() -> Tier1WasmOptions {
    Tier1WasmOptions {
        inline_tlb: true,
        dispatcher: true,
        chain_module_id: Some(MODULE_ID),
        ..Default::default()
    }
}

/// Two blocks that jump to each other, compiled into one module.
fn two_chained_blocks(backend: &mut WasmtimeBackend<CpuState>) -> Vec<u32> {
    write_block(backend, A, B);
    write_block(backend, B, A);
    let irs: Vec<_> = [A, B]
        .iter()
        .map(|&at| translate_block(&discover_block(&*backend, at, BlockLimits::default())))
        .collect();
    let refs: Vec<_> = irs.iter().collect();
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&refs, options());
    backend.add_dispatched_blocks(&wasm, 2)
}

#[test]
fn a_module_chains_between_its_own_blocks() {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = two_chained_blocks(&mut backend);

    backend.chain_map_insert(A, MODULE_ID, 0);
    backend.chain_map_insert(B, MODULE_ID, 1);
    backend.set_chain_limits(u64::MAX, 50);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = A;

    // The lookup declines every step, so anything past the first block was chained
    // inside wasm rather than dispatched by the host.
    let outcome = backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);
    assert_eq!(
        outcome.blocks, 50,
        "the dispatcher should have chained to its own block cap"
    );
    assert_eq!(
        outcome.retired, 100,
        "each block retires two guest instructions"
    );
    assert!(!outcome.exit_to_interpreter);
}

#[test]
fn the_block_cap_and_the_budget_both_stop_it() {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = two_chained_blocks(&mut backend);
    backend.chain_map_insert(A, MODULE_ID, 0);
    backend.chain_map_insert(B, MODULE_ID, 1);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = A;
    // Budget first: 7 instructions allows three two-instruction blocks and stops.
    backend.set_chain_limits(7, 1000);
    let outcome = backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);
    assert_eq!(outcome.blocks, 4, "stops once retired reaches the budget");
    assert_eq!(outcome.retired, 8);
}

#[test]
fn an_entry_for_another_module_is_not_followed() {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = two_chained_blocks(&mut backend);
    // Right RIP, wrong module: the index means nothing here and must be ignored.
    backend.chain_map_insert(A, MODULE_ID + 1, 0);
    backend.chain_map_insert(B, MODULE_ID + 1, 1);
    backend.set_chain_limits(u64::MAX, 50);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = A;
    let outcome = backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);
    assert_eq!(outcome.blocks, 1, "no chaining across modules");
    assert_eq!(outcome.retired, 2);
}

#[test]
fn a_forgotten_entry_stops_the_chain() {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = two_chained_blocks(&mut backend);
    backend.chain_map_insert(A, MODULE_ID, 0);
    backend.chain_map_insert(B, MODULE_ID, 1);
    backend.chain_map_forget(B);
    backend.set_chain_limits(u64::MAX, 50);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = A;
    let outcome = backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);
    assert_eq!(
        outcome.blocks, 1,
        "A ran, and its successor is no longer in the map"
    );
}

//! What one link of a block chain costs.
//!
//! The chain loop lives on the host: each block is a `TypedFunc::call` into the
//! engine, and between calls the host looks the next block up by RIP. Blocks retire
//! about four guest instructions, so whatever a link costs is divided by four before
//! it reaches the per-instruction number — a 40ns link is 10ns per guest instruction,
//! and compiled code currently costs 38ns per guest instruction in total.
//!
//! This measures the link itself, separately from the block body, so the decision to
//! move dispatch inside wasm rests on a number rather than an intuition.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test chain_step_cost -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x100;
const ITERS: usize = 200_000;

/// `add eax, 1` then a jump back to the top.
///
/// Deliberately not `ret`: a return reads the stack, and this measures the *link*, not
/// a memory access. With the inline TLB compiled in, that read also has to resolve
/// against a real mapping or the block traps.
fn tiny_block(backend: &mut WasmtimeBackend<CpuState>) -> u32 {
    for (i, b) in [0x83u8, 0xc0, 0x01, 0xeb, 0xfb].iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let block = discover_block(&*backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions {
            inline_tlb: true,
            ..Default::default()
        },
    );
    backend.add_compiled_block(&wasm)
}

#[test]
fn a_chain_link_costs() {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = tiny_block(&mut backend);

    // `ret` reads the stack, and with the inline TLB compiled in that probe needs a
    // translator to fill from — without one every access misses and the block bails,
    // so the chain stops after one link and the assertion below fails. Identity-map
    // guest pages as RAM.
    backend.set_translator(Some(Box::new(|vaddr: u64, _access: u32| {
        Some((
            vaddr & !0xfff,
            aero_jit_x86::TLB_FLAG_READ
                | aero_jit_x86::TLB_FLAG_WRITE
                | aero_jit_x86::TLB_FLAG_EXEC
                | aero_jit_x86::TLB_FLAG_IS_RAM,
        ))
    })));

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.gpr[aero_types::Gpr::Rsp.as_u8() as usize] = 0x8000;

    // Warm.
    let mut lookup = |_rip: u64, _retired: u64| Some(idx);
    backend.execute_chain(idx, &mut cpu, &mut lookup, 1000);

    // One long chain: the loop runs `ITERS` links but crosses into the engine's
    // `execute_chain` only once, so the per-link cost is the call plus the lookup.
    let start = Instant::now();
    let mut lookup = |_rip: u64, _retired: u64| Some(idx);
    let outcome = backend.execute_chain(idx, &mut cpu, &mut lookup, ITERS as u32);
    let chained = start.elapsed();
    assert_eq!(
        outcome.blocks as usize, ITERS,
        "the chain must run to the block cap"
    );

    // The same work with one `execute_chain` per block: this is what the tier pays
    // when a block's successor is not compiled, and it includes the state sync.
    let start = Instant::now();
    for _ in 0..ITERS {
        let mut lookup = |_rip: u64, _retired: u64| None;
        backend.execute_chain(idx, &mut cpu, &mut lookup, 1);
    }
    let unchained = start.elapsed();

    let per_link = chained.as_secs_f64() * 1e9 / ITERS as f64;
    let per_entry = unchained.as_secs_f64() * 1e9 / ITERS as f64;
    println!("\n  chained link:  {per_link:6.1} ns  (call + host lookup)");
    println!("  chain entry:   {per_entry:6.1} ns  (call + lookup + full state sync)");
    println!("  sync overhead: {:6.1} ns\n", per_entry - per_link);
    println!(
        "  a block retires ~4 guest instructions, so a link is {:.1} ns per guest instruction",
        per_link / 4.0
    );
    println!("  compiled code currently costs 38.3 ns per guest instruction total\n");
}

//! What one chain step taken *inside* wasm costs, in host instructions.
//!
//! Whole-machine windows cannot measure this. A small change to the tier shifts
//! timing, the guest schedules differently, and the window then measures a different
//! mix of guest code — the same dispatcher edit read as 314.5 in one window and 112.2
//! in another while the in-wasm chain rate and the promotion count were unchanged.
//! Anything smaller than a few percent has to be measured somewhere the guest cannot
//! move.
//!
//! Here the guest is two blocks jumping to each other, the chain length is set by
//! `AERO_LINKS`, and differencing two lengths cancels everything fixed:
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test wasm_chain_step_insns -- --nocapture
//! BIN=$(ls -t target/release/deps/wasm_chain_step_insns-* | grep -v '\.d$' | head -1)
//! for n in 2000000 10000000; do
//!   perf stat -e instructions:u env AERO_LINKS=$n $BIN --nocapture
//! done
//! ```
//!
//! The difference divided by the extra links is the dispatcher's loop plus the block
//! body — and the same block body appears in `chain_link_insns`, so subtracting the
//! two isolates what chaining removed.
//!
//! Measured 2026-08-29, and the engine is the whole story:
//!
//! | | host instructions per step |
//! | --- | --- |
//! | in-wasm chain step, Cranelift (`AERO_OPT=1`) | **136** |
//! | host round trip (`chain_link_insns`) | 420 |
//! | in-wasm chain step, Winch | **526** |
//!
//! So chaining is a 3x saving *only* in a promoted module, and on Winch a chain step
//! costs more than the host round trip it replaces. That is the explanation for
//! admission-time chaining measuring 319.6: it was never really about thrashing the
//! chain map, it was that the steps it added were the expensive kind. Run this without
//! `AERO_OPT` and it measures an engine the machine never chains on.

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const A: u64 = 0x100;
const B: u64 = 0x200;
const MODULE_ID: u32 = 7;

/// `add eax, 1` then `jmp` to `target`. No memory access, so what is measured is the
/// step and not a TLB miss into a host helper.
fn write_block(backend: &mut WasmtimeBackend<CpuState>, at: u64, target: u64) {
    let mut code = vec![0x83u8, 0xc0, 0x01, 0xe9];
    let next = at + 8;
    code.extend_from_slice(&((target as i64 - next as i64) as i32).to_le_bytes());
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(at + i as u64, *b);
    }
}

#[test]
fn an_in_wasm_chain_step_costs_host_instructions() {
    let links: u32 = std::env::var("AERO_LINKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    write_block(&mut backend, A, B);
    write_block(&mut backend, B, A);
    let irs: Vec<_> = [A, B]
        .iter()
        .map(|&at| translate_block(&discover_block(&backend, at, BlockLimits::default())))
        .collect();
    let refs: Vec<_> = irs.iter().collect();
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(
        &refs,
        Tier1WasmOptions {
            inline_tlb: true,
            dispatcher: true,
            chain_module_id: Some(MODULE_ID),
            ..Default::default()
        },
    );
    let idx = backend.add_dispatched_blocks(&wasm, 2);
    // `AERO_OPT=1` promotes the pair to Cranelift first, which is the only engine that
    // actually chains in the machine: the chain map is promoted-only. Measuring the
    // Winch dispatcher instead answers a question nobody asks — and answers it
    // misleadingly, because on Winch a chain step costs *more* than a host round trip.
    if std::env::var_os("AERO_OPT").is_some() {
        assert!(
            backend.promote_blocks(&idx, &wasm) == 2,
            "the Cranelift engine must be available (unset AERO_JIT_NO_TIERUP)"
        );
    }
    backend.chain_map_insert(A, MODULE_ID, 0);
    backend.chain_map_insert(B, MODULE_ID, 1);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = A;

    // Warm, so first-call lazy work is outside the measured span.
    backend.set_chain_limits(u64::MAX, 1000);
    backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);

    // One host call; every step after the first is taken by the dispatcher.
    backend.set_chain_limits(u64::MAX, links);
    let outcome = backend.execute_chain(idx[0], &mut cpu, &mut |_rip, _retired| None, 64);
    assert_eq!(
        outcome.blocks, links,
        "the dispatcher must chain to its own block cap"
    );
    println!("  ran {links} in-wasm steps");
}

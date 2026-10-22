//! What one link of a block chain costs, in host instructions.
//!
//! `chain_step_cost` measures the same link in nanoseconds, which is unusable on a
//! loaded machine. Host instructions retired are immune to load, so this runs a
//! chain of `AERO_LINKS` links and lets the caller difference two lengths:
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test chain_link_insns -- --nocapture
//! for n in 2000000 6000000; do
//!   perf stat -e instructions:u env AERO_LINKS=$n \
//!     ./target/release/deps/chain_link_insns-* --nocapture
//! done
//! ```
//!
//! The difference divided by the extra links is the cost of one host-side chain
//! step: the `TypedFunc::call` into the engine plus the host lookup between blocks.
//! That is precisely what moving dispatch inside wasm would remove, so it bounds
//! the prize before any of it is built.

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x100;

/// `add eax, 1` then a jump back to the top.
///
/// Deliberately *not* `ret`: a return reads the stack, and in a bare backend with
/// no TLB warmed that access leaves wasm for a host helper — which is itself an
/// engine crossing, and would be counted as part of the link. A self-jump touches
/// no memory, so what is left is the link alone.
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
fn a_chain_link_costs_host_instructions() {
    let links: u32 = std::env::var("AERO_LINKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    let idx = tiny_block(&mut backend);

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.gpr[aero_types::Gpr::Rsp.as_u8() as usize] = 0x8000;

    // Warm, so first-call lazy work is not in the measured span.
    backend.execute_chain(idx, &mut cpu, &mut |_rip, _retired| Some(idx), 1000);

    // One long chain: `execute_chain` is entered once, so the state sync is paid
    // once and everything else scales with the number of links.
    let outcome = backend.execute_chain(idx, &mut cpu, &mut |_rip, _retired| Some(idx), links);
    assert_eq!(outcome.blocks, links, "the chain must run to the block cap");
    println!("  ran {links} links");
}

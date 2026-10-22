//! What a block boundary costs, measured by making blocks longer.
//!
//! Compiled blocks here average 4.2 guest instructions, and every boundary reloads
//! the guest registers from `CpuState`, spills what it wrote, and returns through the
//! host to be dispatched again. Divided by four instructions, that is paid often.
//!
//! Superblock formation is the standard answer, and it is a large change, so it is
//! worth knowing what it can buy before building it. Running the *same* instruction
//! mix in blocks of 1, 2, 4 ... 64 separates the two costs: the slope is what an
//! instruction costs to execute, and the intercept is what the boundary costs.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --features tier1-inline-tlb \
//!   --test block_length_cost -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};
use aero_types::Gpr;

const ENTRY: u64 = 0x100;
const STACK: u64 = 0x8000;

/// `n` copies of the body, then `jmp +0` so the block ends on a real control transfer.
fn build(body: &[u8], n: usize) -> Vec<u8> {
    let mut code = Vec::new();
    for _ in 0..n {
        code.extend_from_slice(body);
    }
    code.extend_from_slice(&[0xeb, 0x00]);
    code
}

fn measure(name: &str, body: &[u8], iters: usize) {
    println!("\n  {name}");
    println!("       n   ns/block   ns/guest-insn");
    let mut first: Option<(f64, f64)> = None;
    for n in [1usize, 2, 4, 8, 16, 32, 64] {
        let code = build(body, n);
        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(32, 0x9_0000);
        for (i, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + i as u64, *b);
        }
        backend.write(STACK, aero_jit_x86::Width::W64, 0);

        let block = discover_block(&backend, ENTRY, BlockLimits { max_insts: n + 1, max_bytes: 4096, ..Default::default() });
        assert_eq!(block.insts.len(), n + 1, "expected {n} bodies plus the jump");
        let mut ir = translate_block(&block);
        let _ = aero_jit_x86::tier1::eliminate_dead_flags(&mut ir);
        let wasm = Tier1WasmCodegen::new().compile_block_with_options(
            &ir,
            Tier1WasmOptions {
                inline_tlb: true,
                ..Default::default()
            },
        );
        let idx = backend.add_compiled_block(&wasm);

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.rip = ENTRY;
        cpu.gpr[Gpr::Rsp.as_u8() as usize] = STACK;
        cpu.gpr[Gpr::Rsi.as_u8() as usize] = 0x4000;

        // `execute` snapshots guest RAM for rollback and is not the path the tier
        // uses; measuring it measures a half-megabyte memcpy. `execute_chain` with a
        // lookup that declines is one block, entered the way the tier enters one.
        for _ in 0..1000 {
            cpu.rip = ENTRY;
            let mut lookup = |_rip: u64, _retired: u64| None;
            backend.execute_chain(idx, &mut cpu, &mut lookup, 1);
        }
        let start = Instant::now();
        for _ in 0..iters {
            cpu.rip = ENTRY;
            let mut lookup = |_rip: u64, _retired: u64| None;
            backend.execute_chain(idx, &mut cpu, &mut lookup, 1);
        }
        let elapsed = start.elapsed();

        let per_block = elapsed.as_secs_f64() * 1e9 / iters as f64;
        // The trailing jump retires too, but it is the block's terminator rather than
        // work the guest asked for; count only the bodies.
        let per_insn = per_block / n as f64;
        println!("    {n:>4}   {per_block:8.1}   {per_insn:10.1}");
        if n == 1 {
            first = Some((per_block, per_insn));
        }
        if n == 64 {
            let (b1, _) = first.unwrap();
            // Two points, two unknowns: cost = boundary + n * per_insn.
            let slope = (per_block - b1) / 63.0;
            let intercept = b1 - slope;
            println!(
                "    -> {intercept:.1} ns per block boundary, {slope:.1} ns per guest instruction"
            );
            println!(
                "    -> at the 4.2-instruction blocks this workload really has, the boundary is {:.0}% of the cost",
                100.0 * intercept / (intercept + 4.2 * slope)
            );
        }
    }
}

#[test]
fn what_a_block_boundary_costs() {
    // `add eax, 1` — register-only, one flag-setting ALU op.
    measure("add eax, 1 (register only)", &[0x83, 0xc0, 0x01], 200_000);
    // `mov eax, [rsi]` — a load through the inline TLB.
    measure("mov eax, [rsi] (load)", &[0x8b, 0x06], 200_000);
    // `add eax, [rsi]` — load plus flags.
    measure("add eax, [rsi] (load + flags)", &[0x03, 0x06], 200_000);
}

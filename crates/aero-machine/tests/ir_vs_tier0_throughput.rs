//! Throughput of the Tier-1 IR interpreter against Tier-0, on identical guest code.
//!
//! This exists to answer one question before any more work goes into widening JIT
//! coverage: *what is the ceiling?* Coverage only matters in proportion to how much
//! faster the covered path is, so the speedup measured here bounds what perfect
//! coverage could ever buy.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p aero-machine --test ir_vs_tier0_throughput -- --nocapture
//! ```
//!
//! It is a measurement, not a regression gate: the assertions only check that both
//! paths ran and agreed on the architectural result, never on a wall-clock number.

use std::time::Instant;

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::run_batch_cpu_core_with_assists;
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode, CpuState};
use aero_cpu_core::CpuCore;
use aero_machine::jit::{run_ir_chain, IrBlockCache};

/// A loop whose body is a plain ALU/branch mix, the shape Tier-1 handles best.
///
/// ```text
/// 0x1000: add eax, ecx
/// 0x1002: xor ebx, eax
/// 0x1004: add ecx, 1
/// 0x1007: sub edx, 1
/// 0x100a: jne 0x1000
/// ```
const ALU_LOOP: &[u8] = &[
    0x01, 0xc8, // add eax, ecx
    0x31, 0xc3, // xor ebx, eax
    0x83, 0xc1, 0x01, // add ecx, 1
    0x83, 0xea, 0x01, // sub edx, 1
    0x75, 0xf4, // jne -12
];

const ENTRY: u64 = 0x1000;
/// Loop trip count. Large enough that setup cost is noise, small enough to stay quick.
const ITERATIONS: u64 = 2_000_000;
const INSTS_PER_ITER: u64 = 5;

fn seed_cpu_state() -> CpuState {
    let mut state = CpuState::new(CpuMode::Long);
    state.set_rip(ENTRY);
    state.write_gpr64(gpr::RAX, 0);
    state.write_gpr64(gpr::RBX, 0);
    state.write_gpr64(gpr::RCX, 1);
    state.write_gpr64(gpr::RDX, ITERATIONS);
    state
}

fn bus_with_loop() -> FlatTestBus {
    let mut bus = FlatTestBus::new(0x4000);
    bus.load(ENTRY, ALU_LOOP);
    bus
}

/// Architectural result of the loop, used to prove both paths did the same work.
fn result_of(state: &CpuState) -> (u64, u64, u64) {
    (
        state.read_gpr64(gpr::RAX),
        state.read_gpr64(gpr::RBX),
        state.read_gpr64(gpr::RCX),
    )
}

#[test]
fn ir_interpreter_vs_tier0_throughput() {
    let total_insts = ITERATIONS * INSTS_PER_ITER;

    // ── Tier-0 ───────────────────────────────────────────────────────────────
    // `exit_on_branch: false` matches what `Machine::run_slice` uses; leaving it
    // true would return to the caller on every iteration and measure call overhead
    // rather than interpretation.
    let mut bus = bus_with_loop();
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state = seed_cpu_state();
    let cfg = Tier0Config {
        features: Default::default(),
        exit_on_branch: false,
    };
    let mut assists = AssistContext::default();

    let t0_start = Instant::now();
    let batch =
        run_batch_cpu_core_with_assists(&cfg, &mut assists, &mut cpu, &mut bus, total_insts + 16);
    let t0_elapsed = t0_start.elapsed();
    let t0_executed = batch.executed;
    let t0_result = result_of(&cpu.state);

    // ── Tier-1 IR interpreter ────────────────────────────────────────────────
    let mut bus = bus_with_loop();
    let mut cache = IrBlockCache::new();
    assert!(
        cache.compile(ENTRY, 64, &mut bus),
        "the ALU loop must be a cacheable IR block"
    );
    let mut state = seed_cpu_state();

    // `run_ir_chain` stops after `MAX_CHAIN_BLOCKS`, so re-enter until the loop is
    // done. Re-entry is part of the cost the machine would pay, so it stays inside
    // the timed region.
    let ir_start = Instant::now();
    let mut ir_executed = 0u64;
    let mut ir_blocks = 0u64;
    let mut ir_entries = 0u64;
    while ir_executed < total_insts {
        let Some((_next_rip, insts, blocks)) =
            run_ir_chain(&cache, &mut state, &mut bus, total_insts - ir_executed)
        else {
            break;
        };
        ir_executed += insts;
        ir_blocks += blocks;
        ir_entries += 1;
    }
    let ir_elapsed = ir_start.elapsed();
    let ir_result = result_of(&state);
    assert!(ir_entries > 0, "the compiled loop must execute from the cache");

    // ── Report ───────────────────────────────────────────────────────────────
    let t0_ns = t0_elapsed.as_secs_f64() * 1e9 / t0_executed as f64;
    let ir_ns = ir_elapsed.as_secs_f64() * 1e9 / ir_executed as f64;

    println!("\n  guest instructions: tier0={t0_executed} ir={ir_executed} (ir blocks={ir_blocks}, chain entries={ir_entries})");
    println!("  tier0: {t0_ns:6.2} ns/guest insn  ({:.1} MIPS)", 1000.0 / t0_ns);
    println!("  ir:    {ir_ns:6.2} ns/guest insn  ({:.1} MIPS)", 1000.0 / ir_ns);
    println!("  speedup: {:.2}x\n", t0_ns / ir_ns);

    // Both paths must have run the whole loop and produced identical architectural
    // state. Without this the timing above could be comparing different work.
    assert_eq!(
        t0_executed, total_insts,
        "Tier-0 must retire every instruction in the loop"
    );
    assert_eq!(
        ir_executed, total_insts,
        "the IR chain must retire every instruction in the loop"
    );
    assert_eq!(
        t0_result, ir_result,
        "Tier-0 and the IR interpreter must agree on (rax, rbx, rcx)"
    );
}

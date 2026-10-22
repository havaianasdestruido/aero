//! WinSetup apply-scan loop (WIM expand, live RIP `0x7fefc0c1272`):
//! `inc rdx; inc dword [r9+0x2ed8]; cmp byte [rdx], 0xe8; jnz`.
//!
//! The fused Tier-0 batch path must retire the same architectural state as
//! stepping one instruction at a time through `run_batch_cpu_core_with_assists`
//! (the function `Machine::run_slice` actually calls). A batch that stops
//! early, skips the counter, or misses the `0xe8` is the expand-wall defect.

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode};
use aero_cpu_core::CpuCore;

const LOOP: [u8; 15] = [
    0x48, 0xff, 0xc2, 0x41, 0xff, 0x81, 0xd8, 0x2e, 0x00, 0x00, 0x80, 0x3a, 0xe8, 0x75, 0xf1,
];
const LOOP_RIP: u64 = 0x1272;
const COUNTER_BASE: u64 = 0x1000;
const STREAM: u64 = 0x2000;

fn load_scan_fixture(bus: &mut FlatTestBus) {
    bus.load(LOOP_RIP, &LOOP);
    bus.load(LOOP_RIP + LOOP.len() as u64, &[0xF4]); // HLT after fallthrough
    let mut stream = [0u8; 16];
    stream[8] = 0xe8;
    bus.load(STREAM, &stream);
}

fn fresh_cpu() -> CpuCore {
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.set_rip(LOOP_RIP);
    cpu.state.write_gpr64(gpr::RDX, STREAM - 1);
    cpu.state.write_gpr64(gpr::R9, COUNTER_BASE);
    cpu
}

fn run(
    cpu: &mut CpuCore,
    bus: &mut FlatTestBus,
    max: u64,
) -> aero_cpu_core::interp::tier0::exec::BatchResult {
    let mut ctx = AssistContext::default();
    let mut cfg = Tier0Config::from_cpuid(&ctx.features);
    cfg.exit_on_branch = false;
    run_batch_cpu_core_with_assists(&cfg, &mut ctx, cpu, bus, max)
}

fn counter(bus: &FlatTestBus) -> u32 {
    u32::from_le_bytes(bus.slice(COUNTER_BASE + 0x2ed8, 4).try_into().unwrap())
}

#[test]
fn winsetup_e8_scan_batch_matches_single_step_oracle() {
    let mut stepped_bus = FlatTestBus::new(0x4000);
    load_scan_fixture(&mut stepped_bus);
    let mut stepped = fresh_cpu();
    let mut step_executed = 0u64;
    for _ in 0..64 {
        let res = run(&mut stepped, &mut stepped_bus, 1);
        step_executed += res.executed;
        if matches!(res.exit, BatchExit::Halted) {
            break;
        }
    }

    let mut batched_bus = FlatTestBus::new(0x4000);
    load_scan_fixture(&mut batched_bus);
    let mut batched = fresh_cpu();
    let batched_res = run(&mut batched, &mut batched_bus, 10_000);

    assert_eq!(
        batched.state.read_gpr64(gpr::RDX),
        STREAM + 8,
        "rdx must land on the 0xe8"
    );
    assert_eq!(
        counter(&batched_bus),
        9,
        "byte counter must count 8 zeros + the 0xe8"
    );
    assert_eq!(
        batched.state.rip(),
        LOOP_RIP + LOOP.len() as u64 + 1,
        "HLT after the scan must retire"
    );
    assert_eq!(
        batched.state.read_gpr64(gpr::RDX),
        stepped.state.read_gpr64(gpr::RDX)
    );
    assert_eq!(counter(&batched_bus), counter(&stepped_bus));
    assert_eq!(batched.state.rip(), stepped.state.rip());
    assert_eq!(batched.state.rflags(), stepped.state.rflags());
    assert_eq!(batched_res.executed, step_executed);
    assert!(
        batched_res.executed >= 36,
        "9 iterations of a 4-instruction loop, got {}",
        batched_res.executed
    );
}

#[test]
fn winsetup_e8_scan_budget_stops_on_a_loop_head() {
    let mut bus = FlatTestBus::new(0x4000);
    load_scan_fixture(&mut bus);
    let mut cpu = fresh_cpu();
    let res = run(&mut cpu, &mut bus, 8);
    assert_eq!(res.executed, 8, "two complete iterations");
    assert_eq!(cpu.state.rip(), LOOP_RIP);
    assert_eq!(cpu.state.read_gpr64(gpr::RDX), STREAM + 1);
    assert_eq!(counter(&bus), 2);
}

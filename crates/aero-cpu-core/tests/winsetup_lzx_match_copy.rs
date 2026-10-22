//! WinSetup LZX match-copy loops sampled on the expand grind:
//! - verbatim `0x7fefc0c3ac4`: `mov al,[rcx]; dec r10d; inc esi; mov [r14+r15],al; inc r14; inc rcx; test r10d; jg`
//! - aligned `0x7fefc0c4038`: `mov rdx,[r9]; dec r10d; inc ebx; mov cl,[r8+rdx]; inc r8d; mov [rdx+r12],cl; inc r12; test r10d; jg`
//!
//! The fused Tier-0 path must match single-step through
//! `run_batch_cpu_core_with_assists`.

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode};
use aero_cpu_core::CpuCore;

const VERBATIM: [u8; 22] = [
    0x8a, 0x01, 0x41, 0xff, 0xca, 0xff, 0xc6, 0x43, 0x88, 0x04, 0x3e, 0x49, 0xff, 0xc6, 0x48, 0xff,
    0xc1, 0x45, 0x85, 0xd2, 0x7f, 0xea,
];
const ALIGNED: [u8; 25] = [
    0x49, 0x8b, 0x11, 0x41, 0xff, 0xca, 0xff, 0xc3, 0x41, 0x8a, 0x0c, 0x10, 0x41, 0xff, 0xc0, 0x42,
    0x88, 0x0c, 0x22, 0x49, 0xff, 0xc4, 0x45, 0x85, 0xd2, // 7f xx patched below
];

const CODE: u64 = 0x1000;
const WINDOW: u64 = 0x2000;
const OUT: u64 = 0x3000;

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

fn oracle_vs_batch(mut setup: impl FnMut(&mut CpuCore, &mut FlatTestBus)) {
    let mut stepped_bus = FlatTestBus::new(0x5000);
    let mut stepped = CpuCore::new(CpuMode::Long);
    setup(&mut stepped, &mut stepped_bus);
    let mut step_executed = 0u64;
    for _ in 0..512 {
        let res = run(&mut stepped, &mut stepped_bus, 1);
        step_executed += res.executed;
        if matches!(res.exit, BatchExit::Halted) {
            break;
        }
    }

    let mut batched_bus = FlatTestBus::new(0x5000);
    let mut batched = CpuCore::new(CpuMode::Long);
    setup(&mut batched, &mut batched_bus);
    let batched_res = run(&mut batched, &mut batched_bus, 10_000);

    assert_eq!(batched.state.rip(), stepped.state.rip(), "rip");
    assert_eq!(
        batched.state.rflags() & 0x8d5,
        stepped.state.rflags() & 0x8d5,
        "status flags"
    );
    for r in 0..16 {
        assert_eq!(
            batched.state.read_gpr64(r),
            stepped.state.read_gpr64(r),
            "gpr {r}"
        );
    }
    assert_eq!(batched_bus.slice(0, 0x5000), stepped_bus.slice(0, 0x5000));
    assert_eq!(batched_res.executed, step_executed);
}

#[test]
fn verbatim_match_copy_batch_matches_single_step() {
    oracle_vs_batch(|cpu, bus| {
        bus.load(CODE, &VERBATIM);
        bus.load(CODE + VERBATIM.len() as u64, &[0xF4]);
        let mut src = [0u8; 16];
        for (i, b) in src.iter_mut().enumerate() {
            *b = 0x40 + i as u8;
        }
        bus.load(WINDOW, &src);
        cpu.state.set_rip(CODE);
        cpu.state.write_gpr64(gpr::RCX, WINDOW);
        cpu.state.write_gpr64(gpr::R10, 16);
        cpu.state.write_gpr64(gpr::RSI, 0);
        cpu.state.write_gpr64(gpr::R14, OUT);
        cpu.state.write_gpr64(gpr::R15, 0);
        cpu.state.write_gpr64(gpr::RAX, 0);
    });
}

#[test]
fn aligned_match_copy_batch_matches_single_step() {
    oracle_vs_batch(|cpu, bus| {
        let mut code = ALIGNED.to_vec();
        // jg back to head: 9-insn body is 25 bytes + 2-byte jg = 27, disp = -27 = 0xe5
        code.extend_from_slice(&[0x7f, 0xe5]);
        bus.load(CODE, &code);
        bus.load(CODE + code.len() as u64, &[0xF4]);
        let mut src = [0u8; 16];
        for (i, b) in src.iter_mut().enumerate() {
            *b = 0x80 + i as u8;
        }
        bus.load(WINDOW, &src);
        bus.load(0x1f00, &WINDOW.to_le_bytes()); // [r9] = window
        cpu.state.set_rip(CODE);
        cpu.state.write_gpr64(gpr::R9, 0x1f00);
        cpu.state.write_gpr64(gpr::R10, 16);
        cpu.state.write_gpr64(gpr::RBX, 0);
        cpu.state.write_gpr64(gpr::R8, 0);
        cpu.state.write_gpr64(gpr::R12, 0); // dest offset into window — write at WINDOW+0
                                            // Put dest in a separate region by using a copy dest via r12 offset.
                                            // Store is [rdx+r12]; rdx is WINDOW, so write at WINDOW+r12.
                                            // Use r12 = OUT-WINDOW so dest is OUT.
        cpu.state.write_gpr64(gpr::R12, OUT.wrapping_sub(WINDOW));
        cpu.state.write_gpr64(gpr::RCX, 0);
        cpu.state.write_gpr64(gpr::RDX, 0);
    });
}

#[test]
fn verbatim_match_copy_budget_stops_on_loop_head() {
    let mut bus = FlatTestBus::new(0x5000);
    bus.load(CODE, &VERBATIM);
    bus.load(CODE + VERBATIM.len() as u64, &[0xF4]);
    bus.load(WINDOW, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.set_rip(CODE);
    cpu.state.write_gpr64(gpr::RCX, WINDOW);
    cpu.state.write_gpr64(gpr::R10, 8);
    cpu.state.write_gpr64(gpr::R14, OUT);
    let res = run(&mut cpu, &mut bus, 16);
    assert_eq!(res.executed, 16, "two complete 8-inst iterations");
    assert_eq!(cpu.state.rip(), CODE);
    assert_eq!(cpu.state.read_gpr64(gpr::RCX), WINDOW + 2);
    assert_eq!(bus.slice(OUT, 2), &[1, 2]);
}

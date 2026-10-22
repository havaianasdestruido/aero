//! Win7 SP1 x64 `KiSwapContext` waits for `KTHREAD.Running` (`+0x49`) with
//! `pause; cmp byte [rsi+0x49], 0; je; jmp`. On a uniprocessor the flag can
//! stick at 1 and the loop never exits (live RIP `0xfffff800026eb41e`).

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::{CpuBus, FlatTestBus};
use aero_cpu_core::state::{gpr, CpuMode, FLAG_ZF, RFLAGS_IF};
use aero_cpu_core::CpuCore;
use aero_cpu_core::Exception;

/// Live guest encoding from ntos `0xfffff800026eb41e`.
const SPIN: [u8; 14] = [
    0xF3, 0x90, // pause
    0x80, 0x7E, 0x49, 0x00, // cmp byte [rsi+0x49], 0
    0x0F, 0x84, 0xE1, 0xFC, 0xFF, 0xFF, // je exit
    0xEB, 0xF2, // jmp pause
];

const CODE: u64 = 0x1000;
const KTHREAD: u64 = 0x4000;
const GS_BASE: u64 = 0xffff_8000_0000_5000;
const PRCB_CURRENT: u64 = 0x5188; // GS_BASE+0x188 mapped into the low 64 KiB

/// Identity map plus kernel-canonical GS so `gs:[0x188]` (CurrentThread) is
/// readable on a 64 KiB `FlatTestBus`.
struct KernelGsBus {
    inner: FlatTestBus,
}

impl KernelGsBus {
    fn new() -> Self {
        Self {
            inner: FlatTestBus::new(0x10000),
        }
    }

    fn map(addr: u64) -> u64 {
        if addr >= 0xffff_8000_0000_0000 {
            addr & 0xffff
        } else {
            addr
        }
    }

    fn load(&mut self, addr: u64, data: &[u8]) {
        self.inner.load(Self::map(addr), data);
    }
}

impl CpuBus for KernelGsBus {
    fn read_u8(&mut self, vaddr: u64) -> Result<u8, Exception> {
        self.inner.read_u8(Self::map(vaddr))
    }
    fn read_u16(&mut self, vaddr: u64) -> Result<u16, Exception> {
        self.inner.read_u16(Self::map(vaddr))
    }
    fn read_u32(&mut self, vaddr: u64) -> Result<u32, Exception> {
        self.inner.read_u32(Self::map(vaddr))
    }
    fn read_u64(&mut self, vaddr: u64) -> Result<u64, Exception> {
        self.inner.read_u64(Self::map(vaddr))
    }
    fn read_u128(&mut self, vaddr: u64) -> Result<u128, Exception> {
        self.inner.read_u128(Self::map(vaddr))
    }
    fn write_u8(&mut self, vaddr: u64, val: u8) -> Result<(), Exception> {
        self.inner.write_u8(Self::map(vaddr), val)
    }
    fn write_u16(&mut self, vaddr: u64, val: u16) -> Result<(), Exception> {
        self.inner.write_u16(Self::map(vaddr), val)
    }
    fn write_u32(&mut self, vaddr: u64, val: u32) -> Result<(), Exception> {
        self.inner.write_u32(Self::map(vaddr), val)
    }
    fn write_u64(&mut self, vaddr: u64, val: u64) -> Result<(), Exception> {
        self.inner.write_u64(Self::map(vaddr), val)
    }
    fn write_u128(&mut self, vaddr: u64, val: u128) -> Result<(), Exception> {
        self.inner.write_u128(Self::map(vaddr), val)
    }
    fn fetch(&mut self, vaddr: u64, max_len: usize) -> Result<[u8; 15], Exception> {
        self.inner.fetch(Self::map(vaddr), max_len)
    }
    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception> {
        self.inner.io_read(port, size)
    }
    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception> {
        self.inner.io_write(port, size, val)
    }
}

fn load_spin(bus: &mut impl LoadSpin, running: u8) {
    bus.load_spin(CODE, &SPIN);
    bus.load_spin(CODE + SPIN.len() as u64, &[0xF4]);
    // je rel32 from CODE+6 is CODE+12-0x31F = 0xCED.
    bus.load_spin(0xCED, &[0xF4]);
    bus.load_spin(KTHREAD + 0x49, &[running]);
}

trait LoadSpin {
    fn load_spin(&mut self, addr: u64, data: &[u8]);
}

impl LoadSpin for FlatTestBus {
    fn load_spin(&mut self, addr: u64, data: &[u8]) {
        self.load(addr, data);
    }
}

impl LoadSpin for KernelGsBus {
    fn load_spin(&mut self, addr: u64, data: &[u8]) {
        self.load(addr, data);
    }
}

fn long_cpu() -> CpuCore {
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.set_rip(CODE);
    cpu.state.write_gpr64(gpr::RSI, KTHREAD);
    cpu.state.write_gpr64(gpr::RSP, 0x8000);
    cpu
}

fn run<B: CpuBus>(
    cpu: &mut CpuCore,
    bus: &mut B,
    max: u64,
) -> aero_cpu_core::interp::tier0::exec::BatchResult {
    let mut ctx = AssistContext::default();
    let mut cfg = Tier0Config::from_cpuid(&ctx.features);
    cfg.exit_on_branch = false;
    run_batch_cpu_core_with_assists(&cfg, &mut ctx, cpu, bus, max)
}

#[test]
fn win7_kthread_running_spin_clears_stale_flag_and_exits() {
    let mut bus = FlatTestBus::new(0x10000);
    load_spin(&mut bus, 1);

    let mut cpu = long_cpu();
    let res = run(&mut cpu, &mut bus, 32);
    assert!(res.executed > 0, "fuse must retire at least the pause");
    assert_eq!(
        bus.read_u8(KTHREAD + 0x49).unwrap(),
        0,
        "stale KTHREAD.Running must be cleared"
    );
    assert!(
        matches!(res.exit, BatchExit::Halted) || cpu.state.rip() == 0xCED,
        "cmp/je must take the Running==0 exit, not the pause spin (rip={:#x} exit={:?})",
        cpu.state.rip(),
        res.exit
    );
}

#[test]
fn win7_kthread_running_spin_does_not_clear_current_thread() {
    // Clearing Running on Current (the UP hog) used to self-reschedule sppsvc
    // 316 instead of yielding to RIT 628.
    let mut bus = KernelGsBus::new();
    load_spin(&mut bus, 1);
    bus.load(PRCB_CURRENT, &KTHREAD.to_le_bytes());

    let mut cpu = long_cpu();
    cpu.state.msr.gs_base = GS_BASE;
    let res = run(&mut cpu, &mut bus, 64);
    assert_eq!(res.executed, 64, "Current hog must keep retiring the spin");
    assert_eq!(
        bus.read_u8(KTHREAD + 0x49).unwrap(),
        1,
        "Current KTHREAD.Running must stay set"
    );
    assert_eq!(cpu.state.rip(), CODE, "RIP must remain at the pause head");
    assert!(
        !cpu.state.get_flag(FLAG_ZF),
        "cmp of Running=1 must leave ZF clear so je is not taken"
    );
}

#[test]
fn win7_kthread_running_spin_current_batch_matches_single_step() {
    let mut stepped_bus = KernelGsBus::new();
    load_spin(&mut stepped_bus, 1);
    stepped_bus.load(PRCB_CURRENT, &KTHREAD.to_le_bytes());
    let mut stepped = long_cpu();
    stepped.state.msr.gs_base = GS_BASE;
    let mut step_executed = 0u64;
    for _ in 0..64 {
        let res = run(&mut stepped, &mut stepped_bus, 1);
        step_executed += res.executed;
    }

    let mut batched_bus = KernelGsBus::new();
    load_spin(&mut batched_bus, 1);
    batched_bus.load(PRCB_CURRENT, &KTHREAD.to_le_bytes());
    let mut batched = long_cpu();
    batched.state.msr.gs_base = GS_BASE;
    let batched_res = run(&mut batched, &mut batched_bus, 64);

    assert_eq!(step_executed, 64);
    assert_eq!(batched_res.executed, 64);
    assert_eq!(stepped.state.rip(), batched.state.rip());
    assert_eq!(stepped.state.rflags(), batched.state.rflags());
    assert_eq!(
        stepped_bus.read_u8(KTHREAD + 0x49).unwrap(),
        batched_bus.read_u8(KTHREAD + 0x49).unwrap()
    );
    assert_eq!(batched.state.rip(), CODE);
}

#[test]
fn win7_kthread_running_spin_batches_with_queued_undelivered_irq() {
    // IF=0 leaves the vector queued. Refusing the helper then (the SHA-512
    // prefix bug, same shape) would send the hog back through per-insn
    // dispatch at CR8=2.
    let mut bus = KernelGsBus::new();
    load_spin(&mut bus, 1);
    bus.load(PRCB_CURRENT, &KTHREAD.to_le_bytes());

    let mut cpu = long_cpu();
    cpu.state.msr.gs_base = GS_BASE;
    cpu.state.set_flag(RFLAGS_IF, false);
    cpu.pending.inject_external_interrupt(0xD1);

    let res = run(&mut cpu, &mut bus, 64);
    assert_eq!(res.executed, 64);
    assert_eq!(cpu.state.rip(), CODE);
    assert_eq!(bus.read_u8(KTHREAD + 0x49).unwrap(), 1);
    assert_eq!(
        cpu.pending.external_interrupts().len(),
        1,
        "undelivered IRQ must still be queued after the batch"
    );
}

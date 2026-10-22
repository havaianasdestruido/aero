//! ntoskrnl PAGELK 3-byte name-hash insert (RVA 0x2b5ba0, live
//! `0xfffff8000d4cdba0`). The fused helper must leave the same table,
//! callee-saved GPRs, RSP, RIP and RAX as single-stepping the real bytes.

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode};
use aero_cpu_core::CpuCore;

const HASH_FN: [u8; 365] = [
    0x48, 0x89, 0x5c, 0x24, 0x08, 0x48, 0x89, 0x6c, 0x24, 0x10, 0x48, 0x89, 0x74, 0x24, 0x18, 0x48,
    0x89, 0x7c, 0x24, 0x20, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x44, 0x0f, 0xb6, 0x39,
    0x4c, 0x8b, 0x32, 0x44, 0x0f, 0xb6, 0x59, 0x02, 0x48, 0x8b, 0x5a, 0x08, 0x44, 0x8b, 0x4a, 0x10,
    0x48, 0x8b, 0xfa, 0x0f, 0xb6, 0x51, 0x01, 0x41, 0x8b, 0xc7, 0x4c, 0x8b, 0xc1, 0xc1, 0xe0, 0x04,
    0x33, 0xc9, 0x33, 0xc2, 0x44, 0x8d, 0x51, 0x03, 0xc1, 0xe0, 0x04, 0x41, 0x33, 0xc3, 0x69, 0xc0,
    0x5f, 0x9e, 0xff, 0xff, 0xc1, 0xf8, 0x04, 0x25, 0xff, 0x0f, 0x00, 0x00, 0x44, 0x8b, 0xe0, 0x4d,
    0x03, 0xe4, 0x4c, 0x8d, 0x68, 0x02, 0x4a, 0x8b, 0x6c, 0xe7, 0x28, 0x4d, 0x03, 0xed, 0x4a, 0x8b,
    0x34, 0xef, 0x49, 0x3b, 0xf6, 0x73, 0x31, 0x49, 0x3b, 0xee, 0x73, 0x7f, 0x4a, 0x89, 0x74, 0xe7,
    0x28, 0x4e, 0x89, 0x04, 0xef, 0x8b, 0xc1, 0x48, 0x89, 0x77, 0x18, 0x48, 0x8b, 0x5c, 0x24, 0x28,
    0x48, 0x8b, 0x6c, 0x24, 0x30, 0x48, 0x8b, 0x74, 0x24, 0x38, 0x48, 0x8b, 0x7c, 0x24, 0x40, 0x41,
    0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0xc3, 0x49, 0x3b, 0xf0, 0x73, 0xca, 0x44, 0x38, 0x3e,
    0x75, 0xc5, 0x38, 0x56, 0x01, 0x75, 0xc0, 0x44, 0x38, 0x5e, 0x02, 0x75, 0xba, 0x41, 0x8b, 0xca,
    0x44, 0x3b, 0xc9, 0x76, 0xb2, 0x4c, 0x8b, 0xde, 0x49, 0x8d, 0x50, 0x03, 0x4d, 0x2b, 0xd8, 0x8b,
    0xc1, 0x49, 0x03, 0xc0, 0x48, 0x3b, 0xc3, 0x73, 0x13, 0x41, 0x0f, 0xb6, 0x04, 0x13, 0x38, 0x02,
    0x75, 0x0a, 0xff, 0xc1, 0x48, 0xff, 0xc2, 0x41, 0x3b, 0xc9, 0x72, 0xe3, 0x41, 0x0f, 0xb6, 0x50,
    0x01, 0x45, 0x0f, 0xb6, 0x58, 0x02, 0xe9, 0x7c, 0xff, 0xff, 0xff, 0x49, 0x3b, 0xe8, 0x0f, 0x83,
    0x78, 0xff, 0xff, 0xff, 0x44, 0x38, 0x7d, 0x00, 0x0f, 0x85, 0x6e, 0xff, 0xff, 0xff, 0x38, 0x55,
    0x01, 0x0f, 0x85, 0x65, 0xff, 0xff, 0xff, 0x44, 0x38, 0x5d, 0x02, 0x0f, 0x85, 0x5b, 0xff, 0xff,
    0xff, 0x45, 0x3b, 0xca, 0x76, 0x29, 0x4c, 0x8b, 0xdd, 0x49, 0x8d, 0x50, 0x03, 0x4d, 0x2b, 0xd8,
    0x41, 0x8b, 0xc2, 0x49, 0x03, 0xc0, 0x48, 0x3b, 0xc3, 0x73, 0x14, 0x41, 0x0f, 0xb6, 0x04, 0x13,
    0x38, 0x02, 0x75, 0x0b, 0x41, 0xff, 0xc2, 0x48, 0xff, 0xc2, 0x45, 0x3b, 0xd1, 0x72, 0xe1, 0x41,
    0x3b, 0xca, 0x0f, 0x83, 0x24, 0xff, 0xff, 0xff, 0x4a, 0x89, 0x74, 0xe7, 0x28, 0x4e, 0x89, 0x04,
    0xef, 0x41, 0x8b, 0xc2, 0x48, 0x89, 0x6f, 0x18, 0xe9, 0x1e, 0xff, 0xff, 0xff,
];

const FN: u64 = 0x1000;
const RET: u64 = 0x1f00;
const STACK_TOP: u64 = 0x8000;
const KEY: u64 = 0x9000;
const TABLE: u64 = 0x10000;
const MEM: usize = 0x30000;

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

fn hash3(b0: u8, b1: u8, b2: u8) -> u32 {
    let mut eax = b0 as u32;
    eax <<= 4;
    eax ^= b1 as u32;
    eax <<= 4;
    eax ^= b2 as u32;
    eax = eax.wrapping_mul(0xffff9e5f);
    eax = ((eax as i32) >> 4) as u32;
    eax & 0xfff
}

fn load_fixture(bus: &mut FlatTestBus, key: &[u8], lo: u64, hi: u64, len: u32) {
    bus.load(FN, &HASH_FN);
    bus.load(RET, &[0xF4]);
    bus.load(KEY, key);
    bus.load(STACK_TOP - 8, &RET.to_le_bytes());
    bus.load(TABLE, &lo.to_le_bytes());
    bus.load(TABLE + 8, &hi.to_le_bytes());
    bus.load(TABLE + 0x10, &len.to_le_bytes());
}

fn fresh_cpu() -> CpuCore {
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.set_rip(FN);
    cpu.state.write_gpr64(gpr::RCX, KEY);
    cpu.state.write_gpr64(gpr::RDX, TABLE);
    cpu.state.write_gpr64(gpr::RSP, STACK_TOP - 8);
    cpu.state.write_gpr64(gpr::RBX, 0x1111);
    cpu.state.write_gpr64(gpr::RBP, 0x2222);
    cpu.state.write_gpr64(gpr::RSI, 0x3333);
    cpu.state.write_gpr64(gpr::RDI, 0x4444);
    cpu.state.write_gpr64(gpr::R12, 0x5555);
    cpu.state.write_gpr64(gpr::R13, 0x6666);
    cpu.state.write_gpr64(gpr::R14, 0x7777);
    cpu.state.write_gpr64(gpr::R15, 0x8888);
    cpu
}

fn step_all(cpu: &mut CpuCore, bus: &mut FlatTestBus) -> u64 {
    let mut n = 0;
    for _ in 0..50_000 {
        let res = run(cpu, bus, 1);
        n += res.executed;
        if matches!(res.exit, BatchExit::Halted) || cpu.state.rip() == RET {
            // one more if we landed on HLT via RET
            if cpu.state.rip() == RET && !matches!(res.exit, BatchExit::Halted) {
                let res = run(cpu, bus, 1);
                n += res.executed;
            }
            break;
        }
    }
    n
}

fn assert_same(
    stepped: &CpuCore,
    stepped_bus: &FlatTestBus,
    batched: &CpuCore,
    batched_bus: &FlatTestBus,
) {
    assert_eq!(batched.state.rip(), stepped.state.rip(), "rip");
    assert_eq!(
        batched.state.read_gpr64(gpr::RSP),
        stepped.state.read_gpr64(gpr::RSP),
        "rsp"
    );
    assert_eq!(
        batched.state.read_gpr64(gpr::RAX),
        stepped.state.read_gpr64(gpr::RAX),
        "rax"
    );
    for &r in &[
        gpr::RBX,
        gpr::RBP,
        gpr::RSI,
        gpr::RDI,
        gpr::R12,
        gpr::R13,
        gpr::R14,
        gpr::R15,
    ] {
        assert_eq!(
            batched.state.read_gpr64(r),
            stepped.state.read_gpr64(r),
            "callee-saved {r}"
        );
    }
    assert_eq!(
        batched_bus.slice(TABLE, 0x12000),
        stepped_bus.slice(TABLE, 0x12000)
    );
}

fn run_case(
    key: &[u8],
    lo: u64,
    hi: u64,
    len: u32,
    seed_slot: Option<(u64, u64)>,
    extra: &[(u64, &[u8])],
) {
    let mut stepped_bus = FlatTestBus::new(MEM);
    load_fixture(&mut stepped_bus, key, lo, hi, len);
    if let Some((a, b)) = seed_slot {
        let h = hash3(key[0], key[1], key[2]) as u64;
        stepped_bus.load(TABLE + h * 16 + 0x20, &a.to_le_bytes());
        stepped_bus.load(TABLE + h * 16 + 0x28, &b.to_le_bytes());
    }
    for (addr, bytes) in extra {
        stepped_bus.load(*addr, bytes);
    }
    let mut stepped = fresh_cpu();
    step_all(&mut stepped, &mut stepped_bus);
    assert_eq!(stepped.state.rip(), RET + 1, "stepped must RET to HLT");

    let mut batched_bus = FlatTestBus::new(MEM);
    load_fixture(&mut batched_bus, key, lo, hi, len);
    if let Some((a, b)) = seed_slot {
        let h = hash3(key[0], key[1], key[2]) as u64;
        batched_bus.load(TABLE + h * 16 + 0x20, &a.to_le_bytes());
        batched_bus.load(TABLE + h * 16 + 0x28, &b.to_le_bytes());
    }
    for (addr, bytes) in extra {
        batched_bus.load(*addr, bytes);
    }
    let mut batched = fresh_cpu();
    let res = run(&mut batched, &mut batched_bus, 10_000);
    if batched.state.rip() == RET {
        let _ = run(&mut batched, &mut batched_bus, 1);
    }
    assert!(
        res.executed >= 40,
        "fuse should retire the whole function, got {}",
        res.executed
    );
    assert_same(&stepped, &stepped_bus, &batched, &batched_bus);
}

#[test]
fn empty_bucket_inserts_key() {
    // lo = MAX so 0-filled slots are below the bound → insert path 0x7c.
    run_case(b"abcXXXX", u64::MAX, 0xFFFF_FFFF_FFFF_FFF0, 3, None, &[]);
}

#[test]
fn empty_bucket_longer_name() {
    run_case(
        b"NtCreateFile\0",
        u64::MAX,
        0xFFFF_FFFF_FFFF_FFF0,
        12,
        None,
        &[],
    );
}

#[test]
fn prefix_miss_existing_slot() {
    run_case(
        b"xyz_____",
        u64::MAX,
        0xFFFF_FFFF_FFFF_FFF0,
        8,
        Some((0, 0)),
        &[],
    );
}

#[test]
fn prefix_hit_then_strcmp() {
    // Slot holds a pointer to a same-prefix string so the walk takes the
    // 3-byte hit + remaining-byte compare before inserting.
    const OTHER: u64 = 0x9100;
    run_case(
        b"abcdefg\0",
        0,
        OTHER + 0x100,
        7,
        Some((OTHER, OTHER)),
        &[(OTHER, b"abcdefg\0")],
    );
}

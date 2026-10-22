//! WinSetup LZX Huffman walk sampled on expand (`encode_uncompressed_block`
//! @ +0x1b384e): `neg r8d; movsxd rax,r8d; test ebx,ecx; je even;
//! movsx r8d,word [r9+rax*4+odd]; jmp join; even: movsx …+even;
//! shr ecx,1; test r8d; js`.

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode};
use aero_cpu_core::CpuCore;

const WALK: [u8; 37] = [
    0x41, 0xf7, 0xd8, // neg r8d
    0x49, 0x63, 0xc0, // movsxd rax, r8d
    0x85, 0xcb, // test ebx, ecx
    0x74, 0x0b, // je even
    0x45, 0x0f, 0xbf, 0x84, 0x81, 0x42, 0x0e, 0x00, 0x00, // movsx r8d, [r9+rax*4+0xe42]
    0xeb, 0x09, // jmp join
    0x45, 0x0f, 0xbf, 0x84, 0x81, 0x40, 0x0e, 0x00, 0x00, // movsx r8d, [r9+rax*4+0xe40]
    0xd1, 0xe9, // shr ecx, 1
    0x45, 0x85, 0xc0, // test r8d, r8d
    0x78, 0xdb, // js head
];

const CODE: u64 = 0x1000;
const TABLE: u64 = 0x2000;

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

fn load_table(bus: &mut FlatTestBus) {
    // After neg(-1)=1, even child at [r9+1*4+0xe40] = -2
    // After neg(-2)=2, even child at [r9+2*4+0xe40] = 7 (leaf)
    // Odd children unused (ebx=0).
    let mut table = vec![0u8; 0x1000];
    table[0xe40 + 4..0xe40 + 6].copy_from_slice(&(-2i16).to_le_bytes());
    table[0xe40 + 8..0xe40 + 10].copy_from_slice(&7i16.to_le_bytes());
    bus.load(TABLE, &table);
}

fn setup(cpu: &mut CpuCore, bus: &mut FlatTestBus) {
    bus.load(CODE, &WALK);
    bus.load(CODE + WALK.len() as u64, &[0xF4]);
    load_table(bus);
    cpu.state.set_rip(CODE);
    cpu.state.write_gpr64(gpr::R8, (-1i32) as u32 as u64);
    cpu.state.write_gpr64(gpr::RBX, 0);
    cpu.state.write_gpr64(gpr::RCX, 0x0020_0000);
    cpu.state.write_gpr64(gpr::R9, TABLE);
    cpu.state.write_gpr64(gpr::RAX, 0);
}

#[test]
fn huffman_walk_batch_matches_single_step() {
    let mut stepped_bus = FlatTestBus::new(0x4000);
    let mut stepped = CpuCore::new(CpuMode::Long);
    setup(&mut stepped, &mut stepped_bus);
    let mut step_n = 0u64;
    for _ in 0..256 {
        let res = run(&mut stepped, &mut stepped_bus, 1);
        step_n += res.executed;
        if matches!(res.exit, BatchExit::Halted) {
            break;
        }
    }

    let mut batched_bus = FlatTestBus::new(0x4000);
    let mut batched = CpuCore::new(CpuMode::Long);
    setup(&mut batched, &mut batched_bus);
    let batched_res = run(&mut batched, &mut batched_bus, 10_000);

    assert_eq!(batched.state.rip(), stepped.state.rip(), "rip");
    assert_eq!(
        batched.state.read_gpr64(gpr::R8) as i32,
        7,
        "walk must land on the leaf"
    );
    assert_eq!(
        batched.state.read_gpr64(gpr::R8),
        stepped.state.read_gpr64(gpr::R8)
    );
    assert_eq!(
        batched.state.read_gpr64(gpr::RCX),
        stepped.state.read_gpr64(gpr::RCX)
    );
    assert_eq!(
        batched.state.read_gpr64(gpr::RAX),
        stepped.state.read_gpr64(gpr::RAX)
    );
    assert_eq!(
        batched.state.rflags() & 0x8d5,
        stepped.state.rflags() & 0x8d5
    );
    assert_eq!(batched_res.executed, step_n);
    assert!(
        batched_res.executed >= 16,
        "two walk steps, got {}",
        batched_res.executed
    );
}

#[test]
fn huffman_walk_odd_bit_uses_odd_disp() {
    let mut bus = FlatTestBus::new(0x4000);
    let mut cpu = CpuCore::new(CpuMode::Long);
    bus.load(CODE, &WALK);
    bus.load(CODE + WALK.len() as u64, &[0xF4]);
    let mut table = vec![0u8; 0x1000];
    // neg(-1)=1, ebx has the 0x200000 bit → odd disp 0xe42
    table[0xe42 + 4..0xe42 + 6].copy_from_slice(&11i16.to_le_bytes());
    bus.load(TABLE, &table);
    cpu.state.set_rip(CODE);
    cpu.state.write_gpr64(gpr::R8, (-1i32) as u32 as u64);
    cpu.state.write_gpr64(gpr::RBX, 0x0020_0000);
    cpu.state.write_gpr64(gpr::RCX, 0x0020_0000);
    cpu.state.write_gpr64(gpr::R9, TABLE);
    let _ = run(&mut cpu, &mut bus, 10_000);
    assert_eq!(cpu.state.read_gpr64(gpr::R8) as i32, 11);
    assert_eq!(cpu.state.rip(), CODE + WALK.len() as u64 + 1);
}

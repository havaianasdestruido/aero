//! `BSF`/`BSR` compiled must do exactly what Tier-0 does, including where the manual
//! says "undefined".
//!
//! Two places this can be subtly wrong, and neither is loud:
//!
//! * **A zero source.** Tier-0 (`exec_bsf_bsr`) sets `ZF` and returns *without writing
//!   the destination*. Writing the old value back instead is not the same thing at a
//!   32-bit operand size, because a 32-bit write zeroes the upper half of the
//!   register and not writing does not. A block that gets this wrong corrupts the top
//!   half of a register only when the source happens to be zero.
//! * **The other flags.** `CF`, `OF`, `SF`, `AF` and `PF` are architecturally
//!   undefined here, which means Tier-1 has to leave them exactly as Tier-0 leaves
//!   them — untouched — rather than as any particular value.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test bitscan_equivalence
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::compiled_backend::CompiledBackend;
use aero_jit_x86::tier1::Tier1WasmOptions;
use aero_jit_x86::{discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen};
use aero_types::Gpr;

const ENTRY: u64 = 0x1000;

/// The flags `BSF`/`BSR` leave architecturally undefined, which here means untouched.
const UNDEFINED_FLAGS: [u64; 5] = [
    aero_cpu_core::state::FLAG_CF,
    aero_cpu_core::state::FLAG_OF,
    aero_cpu_core::state::FLAG_SF,
    aero_cpu_core::state::FLAG_PF,
    aero_cpu_core::state::FLAG_AF,
];

fn options() -> Tier1WasmOptions {
    Tier1WasmOptions {
        inline_tlb: true,
        single_fast_path: true,
        shared_mem_helpers: true,
        dispatcher: true,
        ..Default::default()
    }
}

/// Run `code` compiled, from `start`, and return the resulting state.
fn run_compiled(code: &[u8], start: &CpuState) -> CpuState {
    aero_x86::tier1::force_hint_ops(true);
    aero_x86::tier1::force_bitscan(true);
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(64, 0x20_0000);
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    assert!(
        !ir.insts.is_empty(),
        "the block must translate; if BSF was refused this test proves nothing"
    );
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&[&ir], options());
    let idx = backend.add_dispatched_blocks(&wasm, 1)[0];

    let mut cpu = start.clone();
    cpu.rip = ENTRY;
    CompiledBackend::run_chain(&mut backend, idx, &mut cpu, &mut |_rip, _retired| None, 1);
    cpu
}

fn start_state() -> CpuState {
    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu
}

/// `bsf edi, ebx` then `jmp $` so the block ends without touching memory.
fn bsf_edi_ebx() -> Vec<u8> {
    vec![0x0f, 0xbc, 0xfb, 0xeb, 0xfe]
}

/// `bsr edi, ebx` then `jmp $`.
fn bsr_edi_ebx() -> Vec<u8> {
    vec![0x0f, 0xbd, 0xfb, 0xeb, 0xfe]
}

#[test]
fn bsf_finds_the_lowest_set_bit() {
    let mut start = start_state();
    start.gpr[Gpr::Rbx.as_u8() as usize] = 0b1011_0000;
    let out = run_compiled(&bsf_edi_ebx(), &start);
    assert_eq!(out.gpr[Gpr::Rdi.as_u8() as usize], 4);
    assert!(!out.get_flag(aero_cpu_core::state::FLAG_ZF));
}

#[test]
fn bsr_finds_the_highest_set_bit() {
    let mut start = start_state();
    start.gpr[Gpr::Rbx.as_u8() as usize] = 0b1011_0000;
    let out = run_compiled(&bsr_edi_ebx(), &start);
    assert_eq!(out.gpr[Gpr::Rdi.as_u8() as usize], 7);
    assert!(!out.get_flag(aero_cpu_core::state::FLAG_ZF));
}

/// The case the whole design of the translation turns on.
#[test]
fn a_zero_source_leaves_the_whole_destination_register_alone() {
    let mut start = start_state();
    start.gpr[Gpr::Rbx.as_u8() as usize] = 0;
    // Upper half non-zero: a 32-bit *write* would clear it, and Tier-0 does not write.
    start.gpr[Gpr::Rdi.as_u8() as usize] = 0xdead_beef_0000_002a;
    let out = run_compiled(&bsf_edi_ebx(), &start);
    assert_eq!(
        out.gpr[Gpr::Rdi.as_u8() as usize],
        0xdead_beef_0000_002a,
        "a zero source must not write the destination at all"
    );
    assert!(out.get_flag(aero_cpu_core::state::FLAG_ZF));
}

#[test]
fn a_zero_source_sets_zf_for_bsr_too() {
    let mut start = start_state();
    start.gpr[Gpr::Rbx.as_u8() as usize] = 0;
    start.gpr[Gpr::Rdi.as_u8() as usize] = 0x1234_5678_9abc_def0;
    let out = run_compiled(&bsr_edi_ebx(), &start);
    assert_eq!(out.gpr[Gpr::Rdi.as_u8() as usize], 0x1234_5678_9abc_def0);
    assert!(out.get_flag(aero_cpu_core::state::FLAG_ZF));
}

/// A non-zero source at 32 bits *does* clear the upper half, because that is what a
/// 32-bit register write does.
#[test]
fn a_non_zero_source_writes_at_the_operand_width() {
    let mut start = start_state();
    start.gpr[Gpr::Rbx.as_u8() as usize] = 1 << 20;
    start.gpr[Gpr::Rdi.as_u8() as usize] = 0xdead_beef_ffff_ffff;
    let out = run_compiled(&bsf_edi_ebx(), &start);
    assert_eq!(out.gpr[Gpr::Rdi.as_u8() as usize], 20);
}

/// The flags `BSF` does not define must come out as Tier-0 leaves them: untouched.
#[test]
fn the_undefined_flags_are_left_alone() {
    for src in [0u64, 0b1000] {
        let mut start = start_state();
        start.gpr[Gpr::Rbx.as_u8() as usize] = src;
        for f in UNDEFINED_FLAGS {
            start.set_flag(f, true);
        }
        let out = run_compiled(&bsf_edi_ebx(), &start);
        for f in UNDEFINED_FLAGS {
            assert!(
                out.get_flag(f),
                "flag {f:#x} must be left as it was for src={src:#x}"
            );
        }
    }
}

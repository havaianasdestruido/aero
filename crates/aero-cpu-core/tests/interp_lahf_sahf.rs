//! LAHF/SAHF must transfer CF (and the other status flags) through AH.
//!
//! Windows 7's MBR does `int 13; lahf; add sp,10h; sahf` so the EDD read's
//! carry survives the DAP teardown. Missing those opcodes parked RIP on `0x9F`
//! right after a successful PUSHAD.

use aero_cpu_core::interp::tier0::exec::run_batch;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{CpuMode, CpuState, FLAG_CF, FLAG_ZF};
use aero_x86::Register;

#[test]
fn lahf_sahf_preserve_cf_across_add_sp() {
    // stc; lahf; add sp, 10h; sahf; hlt
    let code = [0xF9, 0x9F, 0x83, 0xC4, 0x10, 0x9E, 0xF4];
    let mut bus = FlatTestBus::new(0x2000);
    bus.load(0, &code);
    let mut state = CpuState::new(CpuMode::Bit16);
    state.set_rip(0);
    state.write_reg(Register::SS, 0);
    state.write_reg(Register::SP, 0x7C00);
    state.set_flag(FLAG_CF, false);
    state.set_flag(FLAG_ZF, true);

    assert_eq!(run_batch(&mut state, &mut bus, 1).executed, 1); // stc
    assert_ne!(state.rflags() & FLAG_CF, 0);

    assert_eq!(run_batch(&mut state, &mut bus, 1).executed, 1); // lahf
    assert_eq!(state.read_reg(Register::AH) & 0x01, 1);
    assert_eq!(
        state.read_reg(Register::AH) & 0x02,
        2,
        "bit 1 is reserved 1"
    );
    assert_eq!(state.read_reg(Register::AH) & 0x40, 0x40);

    assert_eq!(run_batch(&mut state, &mut bus, 1).executed, 1); // add sp,10h
    assert_eq!(state.read_reg(Register::SP), 0x7C10);
    assert_eq!(state.rflags() & FLAG_CF, 0, "ADD clears CF");

    assert_eq!(run_batch(&mut state, &mut bus, 1).executed, 1); // sahf
    assert_ne!(
        state.rflags() & FLAG_CF,
        0,
        "SAHF must restore INT13-style CF"
    );
    assert_ne!(state.rflags() & FLAG_ZF, 0);
}

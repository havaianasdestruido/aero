//! Handing a real `CpuState` to hardware.
//!
//! Running code written *for* KVM proves the ioctls work; running the guest's own
//! architectural state proves the backend could replace the compiled tier. Between those
//! two is the state conversion, and it is the part with silent failure modes:
//!
//! * `CpuState::gpr` is in architectural encoding order (RAX, RCX, RDX, RBX, RSP, RBP,
//!   RSI, RDI) while `struct kvm_regs` is RAX, RBX, RCX, RDX, RSI, RDI, RSP, RBP. A
//!   straight copy swaps four registers and the guest keeps running, wrongly.
//! * Segments carry hidden base/limit/access that need not match the descriptor table.
//! * A null-but-loaded segment must arrive as *unusable*, not merely not-present.
//!
//! So: round-trip a populated state, then actually enter 64-bit long mode with page
//! tables built here and check the guest computed what it should.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_cpu_state -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_cpu_core::state::{gpr, CpuMode, CpuState, Segment};
use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};

const RAM: usize = 8 << 20;

/// Where the page tables and code go. Anything page-aligned and clear of each other.
const PML4: u64 = 0x1000;
const PDPT: u64 = 0x2000;
const PD: u64 = 0x3000;
const CODE: u64 = 0x4000;

fn flat_code_segment() -> Segment {
    Segment {
        selector: 0x08,
        _pad0: 0,
        _pad1: 0,
        base: 0,
        limit: 0xFFFF_FFFF,
        // type=execute/read (0xB), S, DPL 0, present, L (64-bit), G.
        access: 0xB | (1 << 4) | (1 << 7) | (1 << 9) | (1 << 11),
    }
}

fn flat_data_segment() -> Segment {
    Segment {
        selector: 0x10,
        _pad0: 0,
        _pad1: 0,
        base: 0,
        limit: 0xFFFF_FFFF,
        // type=read/write (0x3), S, DPL 0, present, D/B, G.
        access: 0x3 | (1 << 4) | (1 << 7) | (1 << 10) | (1 << 11),
    }
}

/// A state that exercises every field the conversion touches.
fn populated_state() -> CpuState {
    let mut cpu = CpuState::new(CpuMode::Long);
    for (i, r) in cpu.gpr.iter_mut().enumerate() {
        // Distinct per register, so a swapped pair is visible rather than plausible.
        *r = 0x1000_0000_0000_0000 | ((i as u64 + 1) * 0x0101_0101);
    }
    cpu.rip = CODE;
    cpu.rflags = 0x2;
    cpu.segments.cs = flat_code_segment();
    cpu.segments.ds = flat_data_segment();
    cpu.segments.es = flat_data_segment();
    cpu.segments.fs = flat_data_segment();
    cpu.segments.gs = flat_data_segment();
    cpu.segments.ss = flat_data_segment();
    cpu.tables.gdtr.base = 0x8000;
    cpu.tables.gdtr.limit = 0x27;
    cpu.tables.idtr.base = 0x9000;
    cpu.tables.idtr.limit = 0xFFF;
    // CR0: PE | MP | ET | NE | WP | PG
    cpu.control.cr0 = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 16) | (1 << 31);
    cpu.control.cr3 = PML4;
    cpu.control.cr4 = 1 << 5; // PAE
    cpu.msr.efer = (1 << 8) | (1 << 10); // LME | LMA
    // The MSRs a booted guest depends on. Distinct values so a mis-mapped index shows up
    // as the wrong field rather than a plausible number.
    cpu.msr.star = 0x0023_0010_0000_0000;
    cpu.msr.lstar = 0xFFFF_F800_0100_0000;
    cpu.msr.cstar = 0xFFFF_F800_0200_0000;
    cpu.msr.fmask = 0x0000_0000_0004_7700;
    cpu.msr.fs_base = 0xFFFF_F880_0300_0000;
    cpu.msr.gs_base = 0xFFFF_F800_0400_0000;
    cpu.msr.kernel_gs_base = 0x0000_07FF_FFFD_E000;
    cpu.msr.sysenter_cs = 0x8;
    cpu.msr.sysenter_esp = 0xFFFF_F800_0500_0000;
    cpu.msr.sysenter_eip = 0xFFFF_F800_0600_0000;
    cpu
}

#[test]
fn a_cpu_state_survives_a_round_trip_through_the_vcpu() {
    let mut ram = GuestRam::new(RAM);
    // Safety: `ram` outlives `vm` and its mapping does not move.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };

    let before = populated_state();
    vm.load_cpu_state(&before).expect("load");
    let mut after = CpuState::new(CpuMode::Long);
    vm.store_cpu_state(&mut after).expect("store");

    for i in 0..16 {
        assert_eq!(
            before.gpr[i], after.gpr[i],
            "GPR {i} did not survive: {:#x} in, {:#x} out — check the two register orders",
            before.gpr[i], after.gpr[i]
        );
    }
    assert_eq!(before.rip, after.rip, "RIP");
    assert_eq!(before.control.cr0, after.control.cr0, "CR0");
    assert_eq!(before.control.cr3, after.control.cr3, "CR3");
    assert_eq!(before.control.cr4, after.control.cr4, "CR4");
    assert_eq!(before.msr.efer, after.msr.efer, "EFER");
    // EFER travels in `kvm_sregs` and the rest through `KVM_SET_MSRS`; a guest restored
    // without these lands in the wrong syscall entry point on its first `syscall`.
    for (name, a, b) in [
        ("STAR", before.msr.star, after.msr.star),
        ("LSTAR", before.msr.lstar, after.msr.lstar),
        ("CSTAR", before.msr.cstar, after.msr.cstar),
        ("FMASK", before.msr.fmask, after.msr.fmask),
        ("FS.base", before.msr.fs_base, after.msr.fs_base),
        ("GS.base", before.msr.gs_base, after.msr.gs_base),
        ("KernelGSbase", before.msr.kernel_gs_base, after.msr.kernel_gs_base),
        ("SYSENTER_CS", before.msr.sysenter_cs, after.msr.sysenter_cs),
        ("SYSENTER_ESP", before.msr.sysenter_esp, after.msr.sysenter_esp),
        ("SYSENTER_EIP", before.msr.sysenter_eip, after.msr.sysenter_eip),
    ] {
        assert_eq!(a, b, "{name} did not survive: {a:#x} in, {b:#x} out");
    }
    assert_eq!(before.tables.gdtr.base, after.tables.gdtr.base, "GDTR base");
    assert_eq!(before.tables.idtr.limit, after.tables.idtr.limit, "IDTR limit");

    for (name, a, b) in [
        ("cs", &before.segments.cs, &after.segments.cs),
        ("ds", &before.segments.ds, &after.segments.ds),
        ("ss", &before.segments.ss, &after.segments.ss),
    ] {
        assert_eq!(a.base, b.base, "{name} base");
        assert_eq!(a.limit, b.limit, "{name} limit");
        assert_eq!(a.selector, b.selector, "{name} selector");
        assert_eq!(
            a.access, b.access,
            "{name} access rights: {:#x} in, {:#x} out",
            a.access, b.access
        );
    }
}

#[test]
fn a_long_mode_guest_runs_from_a_loaded_cpu_state() {
    let mut ram = GuestRam::new(RAM);
    {
        let m = ram.as_mut_slice();
        // Identity-map the first gigabyte with 2MiB pages: PML4 -> PDPT -> PD.
        let put = |m: &mut [u8], at: u64, v: u64| {
            m[at as usize..at as usize + 8].copy_from_slice(&v.to_le_bytes());
        };
        // present | writable
        put(m, PML4, PDPT | 0b11);
        put(m, PDPT, PD | 0b11);
        for i in 0..512u64 {
            // present | writable | page-size (2MiB)
            put(m, PD + i * 8, (i << 21) | 0b1000_0011);
        }
        // 64-bit: add rax, rbx ; hlt
        let code: [u8; 4] = [0x48, 0x01, 0xd8, 0xf4];
        m[CODE as usize..CODE as usize + code.len()].copy_from_slice(&code);
    }

    // Safety: as above.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };

    let mut cpu = populated_state();
    cpu.gpr[gpr::RAX] = 0x1111_2222_3333_4444;
    cpu.gpr[gpr::RBX] = 0x0000_1111_1111_1111;
    vm.load_cpu_state(&cpu).expect("load");

    match vm.run().expect("KVM_RUN") {
        KvmExit::Halt => {}
        other => panic!("expected the guest to halt, got {other:?}"),
    }

    let mut out = CpuState::new(CpuMode::Long);
    vm.store_cpu_state(&mut out).expect("store");
    assert_eq!(
        out.gpr[gpr::RAX],
        0x1111_2222_3333_4444u64.wrapping_add(0x0000_1111_1111_1111),
        "the guest's `add rax, rbx` did not produce what it should — \
         which means RAX or RBX did not arrive where the guest expected"
    );
    println!(
        "  long-mode guest ran from a loaded CpuState: rax={:#x}",
        out.gpr[gpr::RAX]
    );
}

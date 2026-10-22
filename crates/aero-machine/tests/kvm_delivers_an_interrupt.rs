//! Waking a halted guest with an interrupt.
//!
//! Aero's PIC and IOAPIC already decide *which* vector to raise — they do it for the
//! interpreter and the compiled tier. What a KVM backend needs from the hypervisor is only
//! the timing: an interrupt cannot be injected while the guest has them masked or is
//! part-way through an instruction. `request_interrupt_window` turns that from "retry until
//! it takes" into a single exit, and `KVM_INTERRUPT` hands the vector over.
//!
//! The guest here halts with interrupts enabled, so it makes no further progress on its
//! own. If the handler runs, the vector was delivered; if the test hangs or the flag stays
//! clear, it was not. That is the whole property, and it is the last mechanism a `Machine`
//! execution mode needs before it is a matter of wiring.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_delivers_an_interrupt -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};

const RAM: usize = 1 << 20;
/// The vector Aero's PIC would raise for the timer on a stock PC.
const VECTOR: u8 = 0x20;
/// Where the handler records that it ran.
const FLAG: usize = 0x9000;

#[test]
fn a_halted_guest_wakes_on_an_injected_vector() {
    let mut ram = GuestRam::new(RAM);
    {
        let m = ram.as_mut_slice();

        // Real-mode IVT entry for VECTOR: offset then segment, pointing at the handler.
        let handler: u16 = 0x0400;
        m[VECTOR as usize * 4..VECTOR as usize * 4 + 2].copy_from_slice(&handler.to_le_bytes());
        m[VECTOR as usize * 4 + 2..VECTOR as usize * 4 + 4].copy_from_slice(&0u16.to_le_bytes());

        // Entry: enable interrupts and halt. Without the interrupt this guest is done.
        //   sti ; hlt ; jmp $
        let code: Vec<u8> = vec![0xfb, 0xf4, 0xeb, 0xfe];
        m[..code.len()].copy_from_slice(&code);

        // Handler at 0x0400: record a byte, then halt again so the test can observe it.
        //   mov bx, 0x9000 ; mov byte [bx], 0x5a ; hlt ; iret
        let h: Vec<u8> = vec![
            0xbb, 0x00, 0x90, // mov bx, 0x9000
            0xc6, 0x07, 0x5a, // mov byte [bx], 0x5a
            0xf4, // hlt
            0xcf, // iret
        ];
        m[handler as usize..handler as usize + h.len()].copy_from_slice(&h);
        m[FLAG] = 0;
    }

    // Safety: `ram` outlives `vm` and its mapping does not move.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    vm.set_real_mode(0, 0).expect("real mode");

    // Run to the `hlt`. The guest is now idle with interrupts enabled — exactly the state
    // the machine's halt path leaves a vCPU in while it waits for a device.
    match vm.run().expect("KVM_RUN") {
        KvmExit::Halt => {}
        other => panic!("expected the guest to halt first, got {other:?}"),
    }
    assert!(
        vm.interrupts_enabled(),
        "the guest halted with interrupts disabled, so no vector could ever wake it"
    );

    // Ask for the first moment a vector can go in, rather than retrying blindly.
    vm.request_interrupt_window(true);
    let exit = vm.run().expect("KVM_RUN");
    assert!(
        matches!(exit, KvmExit::IrqWindowOpen),
        "expected an interrupt window, got {exit:?}"
    );
    assert!(vm.ready_for_interrupt(), "the window was not actually open");

    assert!(
        vm.inject_interrupt(VECTOR).expect("inject"),
        "KVM refused the vector even with the window open"
    );
    vm.request_interrupt_window(false);

    // The handler halts once it has recorded the byte.
    match vm.run().expect("KVM_RUN") {
        KvmExit::Halt => {}
        other => panic!("expected the handler to halt, got {other:?}"),
    }

    let m = ram.as_mut_slice();
    assert_eq!(
        m[FLAG], 0x5a,
        "the interrupt handler did not run: the halted guest was never woken"
    );
    println!("  injected vector {VECTOR:#x}; the halted guest woke and ran its handler");
}

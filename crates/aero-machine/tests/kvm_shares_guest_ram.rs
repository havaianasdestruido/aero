//! KVM and the host addressing the same guest RAM.
//!
//! The machine's RAM is not a slice it can lend out: under `AERO_JIT_RAM=1` it lives inside
//! the JIT backend's linear memory and is reached through a raw pointer, so the interpreter
//! and compiled code can both touch it. That memory is an `mmap`, which is already
//! page-aligned and contiguous — exactly what `KVM_SET_USER_MEMORY_REGION` requires — so it
//! can be given to KVM directly rather than copied.
//!
//! This checks the property that makes an execution mode possible at all: **one memory,
//! two viewers**. A device doing DMA, a snapshot being restored and the vCPU must all see
//! the same bytes, in both directions and without a flush.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_shares_guest_ram -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};

const RAM: usize = 1 << 20;
/// Where the host plants a value for the guest, and where the guest writes one back.
const HOST_TO_GUEST: usize = 0x8000;
const GUEST_TO_HOST: usize = 0x8100;

#[test]
fn the_guest_and_the_host_see_one_memory() {
    let mut ram = GuestRam::new(RAM);
    let base = ram.as_mut_slice().as_mut_ptr();
    let len = ram.len();

    {
        let m = ram.as_mut_slice();
        // 16-bit real mode:
        //   mov bx, 0x8000 ; mov al, [bx]      ; read what the host planted
        //   inc al
        //   mov bx, 0x8100 ; mov [bx], al      ; write it back for the host to find
        //   hlt
        let code: Vec<u8> = vec![
            0xbb, 0x00, 0x80, // mov bx, 0x8000
            0x8a, 0x07, // mov al, [bx]
            0xfe, 0xc0, // inc al
            0xbb, 0x00, 0x81, // mov bx, 0x8100
            0x88, 0x07, // mov [bx], al
            0xf4, // hlt
        ];
        m[..code.len()].copy_from_slice(&code);
        // Planted before the VM exists, and never copied anywhere.
        m[HOST_TO_GUEST] = 0x41;
        m[GUEST_TO_HOST] = 0x00;
    }

    // Safety: `ram` outlives `vm`, its mapping is page-aligned and does not move.
    let mut vm = match unsafe { KvmVm::new_from_raw(base, len, 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    vm.set_real_mode(0, 0).expect("real mode");

    match vm.run().expect("KVM_RUN") {
        KvmExit::Halt => {}
        other => panic!("expected the guest to halt, got {other:?}"),
    }

    // Read through the *host's* view, with no synchronisation of any kind.
    let m = ram.as_mut_slice();
    assert_eq!(
        m[GUEST_TO_HOST], 0x42,
        "the guest read {:#x} from the host's byte and wrote back {:#x}; \
         one memory would give 0x42",
        m[HOST_TO_GUEST], m[GUEST_TO_HOST]
    );
    println!(
        "  host wrote {:#04x} at {HOST_TO_GUEST:#x}; guest read it, incremented, \
         and the host sees {:#04x} at {GUEST_TO_HOST:#x}",
        0x41, m[GUEST_TO_HOST]
    );
}

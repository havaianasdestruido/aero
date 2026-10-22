//! A KVM guest driving a real device model through the real I/O bus.
//!
//! The previous step routed exits to closures, which proves the plumbing but not that the
//! device models can serve it. This one registers an actual `DebugCon` on an actual
//! `IoPortBus` and has a guest running under hardware virtualisation write to it — the
//! same `IoPortBus::read`/`write` calls the interpreter and the compiled tier make.
//!
//! That is the whole of what "route MMIO and PIO to the existing device models" means for
//! port I/O: `KVM_EXIT_IO` carries port, direction, size and count, and the bus takes
//! exactly those. What remains after this is breadth — every device, and the MMIO side —
//! rather than a question of shape.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_drives_a_real_device -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use std::cell::RefCell;
use std::rc::Rc;

use aero_devices::debugcon::{DebugCon, DEBUGCON_PORT};
use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};
use aero_platform::io::IoPortBus;

const RAM: usize = 1 << 20;

#[test]
fn a_kvm_guest_writes_through_the_real_io_bus_to_a_real_device() {
    // The device and the bus the rest of the machine uses, not a stand-in.
    let log: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
    let mut io = IoPortBus::new();
    io.register(DEBUGCON_PORT, Box::new(DebugCon::new(log.clone())));

    // 16-bit real mode: write "KVM" a byte at a time to the debug console, then halt.
    //   mov al,'K' ; out 0xE9,al ; mov al,'V' ; out 0xE9,al ; mov al,'M' ; out 0xE9,al ; hlt
    let code: Vec<u8> = vec![
        0xb0, b'K', 0xe6, 0xe9, // mov al,'K' ; out 0xe9,al
        0xb0, b'V', 0xe6, 0xe9, // mov al,'V' ; out 0xe9,al
        0xb0, b'M', 0xe6, 0xe9, // mov al,'M' ; out 0xe9,al
        0xf4, // hlt
    ];

    let mut ram = GuestRam::new(RAM);
    ram.as_mut_slice()[..code.len()].copy_from_slice(&code);

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

    let exit = {
        // The bridge a `Machine` execution mode would use: an exit becomes the same bus
        // call every other tier makes. `size` is in bytes and the bus takes it directly.
        let mut pio = |port: u16, write: bool, data: &mut [u8]| {
            let size = data.len() as u8;
            if write {
                let mut v = [0u8; 4];
                v[..data.len().min(4)].copy_from_slice(&data[..data.len().min(4)]);
                io.write(port, size, u32::from_le_bytes(v));
            } else {
                let v = io.read(port, size).to_le_bytes();
                let n = data.len().min(4);
                data[..n].copy_from_slice(&v[..n]);
            }
        };
        let mut mmio = |_a: u64, _w: bool, _d: &mut [u8]| {};
        vm.run_routed(64, &mut pio, &mut mmio).expect("run")
    };

    assert_eq!(exit, KvmExit::Halt, "the guest should have halted");
    assert_eq!(
        log.borrow().as_slice(),
        b"KVM",
        "the device did not receive what the guest wrote — the bytes it saw were {:?}",
        log.borrow()
    );
    println!(
        "  a KVM guest wrote {:?} to a real DebugCon through the real IoPortBus",
        String::from_utf8_lossy(&log.borrow())
    );
}

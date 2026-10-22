//! A KVM guest's MMIO reaching a real device through the real memory bus.
//!
//! Port I/O was the easy half — `KVM_EXIT_IO` hands over port, direction, size and count,
//! and `IoPortBus` takes exactly those. MMIO is how every PCI BAR is reached, and it goes
//! through `MemoryBus::read_physical`/`write_physical`, which is also what the interpreter
//! and the compiled tier call.
//!
//! So this maps a real `MmioHandler` on a real `MemoryBus`, and has a guest running under
//! hardware virtualisation read and write it. The handler is a counter rather than a
//! stub: a read must return what the *device* decides, not what happened to be in the
//! `kvm_run` buffer, and the only way to see the difference is for the device to answer
//! something the guest could not have written.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_mmio_reaches_a_device -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use std::cell::RefCell;
use std::rc::Rc;

use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};
use aero_platform::address_filter::AddressFilter;
use aero_platform::chipset::ChipsetState;
use aero_platform::memory::MemoryBus;
use memory::bus::MmioHandler;

const RAM: usize = 1 << 20;
/// Just past the mapped RAM, so the access leaves the memory region and exits.
const MMIO_BASE: u64 = 0x10_0000;

/// A device that answers reads with a value the guest never wrote, and records writes.
struct Counter {
    reads: Rc<RefCell<Vec<u64>>>,
    writes: Rc<RefCell<Vec<(u64, u64)>>>,
}

impl MmioHandler for Counter {
    fn read(&mut self, offset: u64, _size: usize) -> u64 {
        self.reads.borrow_mut().push(offset);
        // Distinctive: no instruction in the guest produces this by accident.
        0xA5
    }

    fn write(&mut self, offset: u64, _size: usize, value: u64) {
        self.writes.borrow_mut().push((offset, value));
    }
}

#[test]
fn a_kvm_guest_reads_and_writes_a_device_over_mmio() {
    let reads: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));
    let writes: Rc<RefCell<Vec<(u64, u64)>>> = Rc::new(RefCell::new(Vec::new()));

    // The bus the rest of the machine uses.
    // A20 enabled, so a 20-bit-wrapping address is not folded back below 1MiB.
    let chipset = ChipsetState::new(true);
    let mut bus = MemoryBus::new(AddressFilter::new(chipset.a20()), RAM);
    bus.map_mmio(
        MMIO_BASE,
        0x1000,
        Box::new(Counter {
            reads: reads.clone(),
            writes: writes.clone(),
        }),
    )
    .expect("map mmio");

    // 16-bit real mode, reaching 0x100000 through ES:BX = 0xFFFF:0x0010.
    //   mov ax,0xffff ; mov es,ax ; mov bx,0x10
    //   mov al, es:[bx]     ; MMIO read  -> device answers 0xA5
    //   inc al              ; 0xA6, so a stale buffer cannot fake it
    //   mov es:[bx], al     ; MMIO write -> device records 0xA6
    //   hlt
    let code: Vec<u8> = vec![
        0xb8, 0xff, 0xff, // mov ax, 0xffff
        0x8e, 0xc0, // mov es, ax
        0xbb, 0x10, 0x00, // mov bx, 0x10
        0x26, 0x8a, 0x07, // mov al, es:[bx]
        0xfe, 0xc0, // inc al
        0x26, 0x88, 0x07, // mov es:[bx], al
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
        // The bridge a `Machine` execution mode would use for MMIO: the exit's address and
        // length become the same physical access every other tier makes.
        let mut pio = |_p: u16, _w: bool, _d: &mut [u8]| {};
        let mut mmio = |addr: u64, write: bool, data: &mut [u8]| {
            if write {
                bus.write_physical(addr, data);
            } else {
                bus.read_physical(addr, data);
            }
        };
        vm.run_routed(64, &mut pio, &mut mmio).expect("run")
    };

    assert_eq!(exit, KvmExit::Halt, "the guest should have halted");
    assert_eq!(
        reads.borrow().as_slice(),
        &[0],
        "the guest's MMIO read did not reach the device"
    );
    assert_eq!(
        writes.borrow().as_slice(),
        &[(0, 0xA6)],
        "the guest should have written back the device's own answer plus one; \
         getting something else means the read's value never reached the guest"
    );
    println!("  MMIO: device answered 0xA5, guest wrote back 0xA6 through MemoryBus");
}

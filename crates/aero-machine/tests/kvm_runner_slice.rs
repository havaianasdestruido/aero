//! One slice of guest execution, driven the way the machine's outer loop drives a tier.
//!
//! The pieces underneath are each tested on their own — state, memory, port I/O, MMIO,
//! interrupts. This is the composition, and composing them is where the ordering mistakes
//! live: a state written back before the last exit is serviced, a vector injected after
//! entry rather than before, a halt reported without `CpuState::halted` set.
//!
//! The guest here does all four things in one slice: reads a port from a real device,
//! writes an MMIO register on a real bus, halts, and then — on a second slice — takes a
//! vector the "platform" decided. That is the shape `Machine::run_slice` would call.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_runner_slice -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use std::cell::RefCell;
use std::rc::Rc;

use aero_cpu_core::state::{gpr, CpuMode, CpuState};
use aero_machine::kvm::{GuestRam, KvmError, KvmRunner, KvmSliceExit, KvmVm};
use aero_platform::address_filter::AddressFilter;
use aero_platform::chipset::ChipsetState;
use aero_platform::io::{IoPortBus, PortIoDevice};
use aero_platform::memory::MemoryBus;
use memory::bus::MmioHandler;

const RAM: usize = 1 << 20;
const MMIO_BASE: u64 = 0x10_0000;
const VECTOR: u8 = 0x20;
const FLAG: usize = 0x9000;

/// A port device that answers with a fixed byte, like a status register.
struct Status(u8, Rc<RefCell<u32>>);
impl PortIoDevice for Status {
    fn read(&mut self, _port: u16, _size: u8) -> u32 {
        *self.1.borrow_mut() += 1;
        u32::from(self.0)
    }
    fn write(&mut self, _port: u16, _size: u8, _value: u32) {}
}

struct Sink(Rc<RefCell<Vec<(u64, u64)>>>);
impl MmioHandler for Sink {
    fn read(&mut self, _offset: u64, _size: usize) -> u64 {
        0
    }
    fn write(&mut self, offset: u64, _size: usize, value: u64) {
        self.0.borrow_mut().push((offset, value));
    }
}

#[test]
fn one_slice_services_ports_mmio_and_a_halt_then_takes_a_vector() {
    let writes: Rc<RefCell<Vec<(u64, u64)>>> = Rc::new(RefCell::new(Vec::new()));

    let mut io = IoPortBus::new();
    let port_reads: Rc<RefCell<u32>> = Rc::new(RefCell::new(0));
    io.register(0x64, Box::new(Status(0x3c, port_reads.clone())));

    // Sanity: the bus must answer before blaming the exit path.
    assert_eq!(io.read(0x64, 1), 0x3c, "the device is not on the bus at all");

    let chipset = ChipsetState::new(true);
    let mut mem = MemoryBus::new(AddressFilter::new(chipset.a20()), RAM);
    mem.map_mmio(MMIO_BASE, 0x1000, Box::new(Sink(writes.clone())))
        .expect("map mmio");

    let mut ram = GuestRam::new(RAM);
    {
        let m = ram.as_mut_slice();
        // IVT entry for VECTOR -> handler at 0x0500.
        let handler: u16 = 0x0500;
        m[VECTOR as usize * 4..VECTOR as usize * 4 + 2].copy_from_slice(&handler.to_le_bytes());
        m[VECTOR as usize * 4 + 2..VECTOR as usize * 4 + 4].copy_from_slice(&0u16.to_le_bytes());

        //   in al, 0x64          ; real device -> 0x3c
        //   mov dl, al           ; keep it: the segment load below clobbers AX, and
        //                        ; `mov ax, 0xffff` takes AL with it
        //   mov ax, 0xffff ; mov es, ax ; mov bx, 0x10
        //   mov es:[bx], dl      ; MMIO write of 0x3c through the real bus
        //   sti ; hlt            ; idle, waiting for the platform
        //   jmp $
        let code: Vec<u8> = vec![
            0xe4, 0x64, // in al, 0x64
            0x88, 0xc2, // mov dl, al
            0xb8, 0xff, 0xff, // mov ax, 0xffff
            0x8e, 0xc0, // mov es, ax
            0xbb, 0x10, 0x00, // mov bx, 0x10
            0x26, 0x88, 0x17, // mov es:[bx], dl
            0xfb, 0xf4, // sti ; hlt
            0xeb, 0xfe, // jmp $
        ];
        m[..code.len()].copy_from_slice(&code);

        // Handler: record, halt so the slice returns.
        let h: Vec<u8> = vec![
            0xbb, 0x00, 0x90, // mov bx, 0x9000
            0xc6, 0x07, 0x5a, // mov byte [bx], 0x5a
            0xf4, // hlt
            0xcf, // iret
        ];
        m[handler as usize..handler as usize + h.len()].copy_from_slice(&h);
    }

    // Safety: `ram` outlives the VM and its mapping does not move.
    let vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    let mut runner = KvmRunner::new(vm);

    // A complete real-mode state, loaded into the vCPU by `run_slice`. "Complete" is the
    // operative word and it is easy to get wrong, because `CpuState::new` leaves whole
    // groups at zero and a vCPU does not tolerate that the way an interpreter does:
    //
    //  * segment descriptors of all zeros are *not present*, and the guest faults on its
    //    first access;
    //  * `IDTR.limit = 0` lets the guest run perfectly and then triple-fault the instant an
    //    interrupt has to be delivered, because the vector's entry lies outside the table.
    //    In real mode that table is the IVT, and it still goes through IDTR.
    //
    // A restored snapshot carries the guest's own values for all of this. A test has to
    // spell them out, and getting the second one wrong costs an afternoon.
    let mut cpu = CpuState::new(CpuMode::Real);
    cpu.rip = 0;
    cpu.rflags = 0x2;
    const CODE_AR: u32 = 0xB | (1 << 4) | (1 << 7);
    const DATA_AR: u32 = 0x3 | (1 << 4) | (1 << 7);
    for (seg, ar) in [
        (&mut cpu.segments.cs, CODE_AR),
        (&mut cpu.segments.ds, DATA_AR),
        (&mut cpu.segments.es, DATA_AR),
        (&mut cpu.segments.fs, DATA_AR),
        (&mut cpu.segments.gs, DATA_AR),
        (&mut cpu.segments.ss, DATA_AR),
    ] {
        seg.selector = 0;
        seg.base = 0;
        seg.limit = 0xFFFF;
        seg.access = ar;
    }
    cpu.control.cr0 = 0x6000_0010;
    cpu.tables.tr.limit = 0xFFFF;
    cpu.tables.tr.access = 0xB | (1 << 7);
    cpu.tables.ldtr.limit = 0xFFFF;
    cpu.tables.ldtr.access = 0x2 | (1 << 7);
    // The real-mode IVT: 256 four-byte entries at physical zero.
    cpu.tables.idtr.base = 0;
    cpu.tables.idtr.limit = 0x3FF;
    let exit = runner
        .run_slice(&mut cpu, &mut io, &mut mem, None, 64)
        .expect("slice");
    assert_eq!(exit, KvmSliceExit::Halted, "the guest should have idled");
    assert!(cpu.halted, "a halt must be visible in the CpuState the caller sees");
    assert!(
        *port_reads.borrow() > 0,
        "the port device was never read: the IO exit did not reach the handler"
    );
    assert_eq!(
        writes.borrow().as_slice(),
        &[(0, 0x3c)],
        "the port read's value should have reached the MMIO device — \
         getting something else means a value was dropped between the two exits"
    );

    // Slice two: the platform has decided a vector. The guest is halted with IF set, so it
    // must wake and run its handler.
    // `resume_slice`: the vCPU already holds the state, so there is nothing to write back
    // in. `run_slice` would work too — the state round-trips faithfully — but re-loading
    // what the vCPU already has is wasted ioctls.
    cpu.halted = false;
    let exit = runner
        .resume_slice(&mut cpu, &mut io, &mut mem, Some(VECTOR), 64)
        .expect("slice");
    println!(
        "  slice 2 -> {exit:?} at rip={:#x}, sp={:#x} (three words pushed by the vector)",
        cpu.rip,
        cpu.gpr[gpr::RSP]
    );
    assert_eq!(exit, KvmSliceExit::Halted, "the handler should have halted");
    assert_eq!(
        ram.as_mut_slice()[FLAG],
        0x5a,
        "the vector was not delivered across the slice boundary"
    );

    println!(
        "  slice 1: in 0x64 -> 0x3c -> MMIO write, then halt; \
         slice 2: vector {VECTOR:#x} delivered and handled"
    );
}

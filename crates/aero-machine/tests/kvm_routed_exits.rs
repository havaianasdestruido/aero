//! Port I/O and MMIO exits reaching a device model.
//!
//! This is what turns the KVM backend from something that runs code into something that
//! could be a `Machine` execution mode: the device models already answer by port and by
//! address, so the backend's whole job at an exit is to make one of those calls and put
//! the answer where the guest will look.
//!
//! Two details that are easy to get wrong and quiet when wrong:
//!
//! * A **read** must have its answer written into the `kvm_run` buffer before the next
//!   entry, or the guest reads whatever was there — stale data rather than an error.
//! * `count` on a port exit may be more than one, for a string instruction. Each
//!   iteration is a separate access, so a handler that looks at the buffer once handles
//!   the first and silently drops the rest.
//!
//! Skips where `/dev/kvm` is absent.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_routed_exits -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};

const RAM: usize = 1 << 20;
/// Outside the single memory region, so an access here exits as MMIO.
const MMIO_ADDR: u64 = 0xF000_0000;

#[test]
fn port_io_and_mmio_exits_reach_their_handlers() {
    // 16-bit real mode:
    //   in  al, 0x71        ; a device read
    //   out 0x70, al        ; hand the same byte back
    //   mov ax, 0xF000
    //   mov es, ax
    //   mov bx, 0
    //   mov al, es:[bx]     ; MMIO read at 0xF0000000? no — real mode cannot reach it,
    //                       ; so the MMIO half is driven from long mode below.
    //   hlt
    let code: Vec<u8> = vec![
        0xe4, 0x71, // in al, 0x71
        0xe6, 0x70, // out 0x70, al
        0xf4, // hlt
    ];

    let mut ram = GuestRam::new(RAM);
    ram.as_mut_slice()[..code.len()].copy_from_slice(&code);

    // Safety: `ram` outlives `vm` and does not move.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    vm.set_real_mode(0, 0).expect("real mode");

    let mut reads: Vec<u16> = Vec::new();
    let mut writes: Vec<(u16, u8)> = Vec::new();
    let mut mmio_hits: Vec<u64> = Vec::new();

    let exit = {
        let mut pio = |port: u16, write: bool, data: &mut [u8]| {
            if write {
                writes.push((port, data[0]));
            } else {
                reads.push(port);
                // The answer has to land here, before the guest resumes.
                data[0] = 0x5a;
            }
        };
        let mut mmio = |addr: u64, _write: bool, data: &mut [u8]| {
            mmio_hits.push(addr);
            data.fill(0);
        };
        vm.run_routed(64, &mut pio, &mut mmio).expect("run")
    };

    assert_eq!(exit, KvmExit::Halt, "the guest should have halted");
    assert_eq!(reads, vec![0x71], "the `in` did not reach the handler");
    assert_eq!(
        writes,
        vec![(0x70, 0x5a)],
        "the `out` should carry back exactly what the read handler supplied — \
         a different value means the read's answer never reached the guest"
    );
    assert!(mmio_hits.is_empty(), "no MMIO was expected here");
    println!("  routed: in 0x71 -> 0x5a, out 0x70 <- 0x5a");
}

#[test]
fn an_access_outside_guest_ram_exits_as_mmio() {
    // 16-bit real mode reaching just past the mapped region: RAM is 1MiB, so a read at
    // 0x0F_FFF0 + segment reaches past it once ES is set high enough. Simpler: point ES
    // at the top of the 20-bit space and read there.
    //   mov ax, 0xFFFF ; mov es, ax ; mov bx, 0x10 ; mov al, es:[bx] ; hlt
    // ES:BX = 0xFFFF0 + 0x10 = 0x100000, exactly one byte past 1MiB of RAM.
    let code: Vec<u8> = vec![
        0xb8, 0xff, 0xff, // mov ax, 0xffff
        0x8e, 0xc0, // mov es, ax
        0xbb, 0x10, 0x00, // mov bx, 0x10
        0x26, 0x8a, 0x07, // mov al, es:[bx]
        0xf4, // hlt
    ];

    let mut ram = GuestRam::new(RAM);
    ram.as_mut_slice()[..code.len()].copy_from_slice(&code);

    // Safety: as above.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    vm.set_real_mode(0, 0).expect("real mode");

    let mut mmio_hits: Vec<(u64, bool)> = Vec::new();
    let exit = {
        let mut pio = |_p: u16, _w: bool, _d: &mut [u8]| {};
        let mut mmio = |addr: u64, write: bool, data: &mut [u8]| {
            mmio_hits.push((addr, write));
            data.fill(0xa5);
        };
        vm.run_routed(64, &mut pio, &mut mmio).expect("run")
    };

    assert_eq!(exit, KvmExit::Halt, "the guest should have halted");
    assert_eq!(
        mmio_hits,
        vec![(0x10_0000, false)],
        "an access one byte past the mapped region must arrive as an MMIO read at that \
         address — this is how every device BAR will be reached"
    );
    println!("  routed: MMIO read at {:#x}", mmio_hits[0].0);
    let _ = MMIO_ADDR;
}

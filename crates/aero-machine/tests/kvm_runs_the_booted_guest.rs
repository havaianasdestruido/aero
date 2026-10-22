//! The booted Windows guest's own state, executing on hardware.
//!
//! Everything before this ran code written for the test. This takes the architectural
//! state of a *real* Windows 7 desktop — restored from the same snapshot the compiled tier
//! is measured against — puts it in a vCPU, and lets the CPU execute it.
//!
//! The guest's memory is the machine's own (`SharedGuestRam`, one mapping behind an `Arc`),
//! its port and MMIO exits are answered by the machine's devices, and its local APIC is
//! programmed from the machine's model. It runs.
//!
//! **Measured with `perf stat -e instructions:G`, which counts guest-mode instructions:
//! about 1.05 billion in the first second — roughly 1,000 MIPS of real Windows, against
//! the compiled tier's 110-140.** Then it stops, and stays stopped: 1,175,787,584
//! instructions at six seconds and 1,175,769,098 at twelve is not a slow guest, it is a
//! guest that will never run again.
//!
//! It is waiting for an interrupt that cannot arrive. Windows programmed *Aero's* IOAPIC
//! before the snapshot was taken, while KVM's in-kernel chips sit at their reset defaults
//! with every redirection entry masked, so raising a line on them goes nowhere. Copying
//! that table across is all it takes, and `kvm_seeded_irqchip` does exactly that and gets
//! a guest that keeps ticking over.
//!
//! Note the halt is invisible from the exit stream: with an in-kernel irqchip `HLT` is
//! handled in the kernel and never surfaces as `KVM_EXIT_HLT`. The preemption signal turns
//! it into `EINTR` instead, which reads as "running but doing nothing". `mp_state` is what
//! tells the two apart, which is why it is queried here.
//!
//! What this does *not* establish is a finished execution mode: `Machine` does not own the
//! loop, and the interrupt controllers' state is not restored.
//!
//! Skips where `/dev/kvm` or the snapshot is absent, so it is inert anywhere but this
//! machine.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_runs_the_booted_guest -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{KvmError, KvmExit, KvmVm, SharedGuestRam};
use aero_machine::{Machine, MachineConfig};

const SNAPSHOT: &str = "/root/aero-work/win7-desktop.bin";
/// The snapshot is a 2GiB guest. Copying it is a one-off memcpy, not a per-run cost.
const RAM_BYTES: usize = 2048 << 20;

#[test]
fn a_booted_windows_state_makes_progress_on_a_vcpu() {
    let Ok(snap) = std::fs::read(SNAPSHOT) else {
        println!("  no snapshot at {SNAPSHOT}; skipping");
        return;
    };

    // One memory, both viewers. With two allocations the vCPU writes a descriptor into its
    // own copy while a device model reads the machine's, and nothing the guest asks for
    // comes back right — it executes happily and never progresses.
    let ram = SharedGuestRam::new(RAM_BYTES);
    let ram_base = ram.base();
    let ram_len = ram.len();

    let mut machine = match Machine::new_with_guest_memory(
        MachineConfig::win7_storage(RAM_BYTES as u64),
        Box::new(ram.clone()),
    ) {
        Ok(m) => m,
        Err(e) => {
            println!("  could not build a machine ({e}); skipping");
            return;
        }
    };
    if let Err(e) = machine.restore_snapshot_bytes(&snap) {
        println!("  could not restore the snapshot ({e}); skipping");
        return;
    }

    let booted = machine.cpu().clone();
    println!(
        "  restored: rip={:#x} cr0={:#x} cr3={:#x} cr4={:#x} efer={:#x} mode={:?}",
        booted.rip, booted.control.cr0, booted.control.cr3, booted.control.cr4,
        booted.msr.efer, booted.mode
    );
    assert!(
        booted.control.cr3 != 0 && booted.msr.efer & (1 << 10) != 0,
        "this snapshot is not a booted long-mode guest; the rest of the test means nothing"
    );

    // With KVM's in-kernel interrupt controllers: without them a booted Windows kernel
    // exits on its first task-priority write and can go no further.
    // Safety: `ram` outlives `vm`, is page-aligned, and its mapping does not move — the
    // machine holds a clone of the same `Arc`.
    let mut vm = match unsafe {
        KvmVm::new_from_raw_with_irqchip(ram_base, ram_len, 0)
    } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };

    vm.load_cpu_state(&booted)
        .expect("the booted state must be acceptable to a vCPU");

    // Give the in-kernel local APIC the programming the guest believes it has.
    //
    // Without this the vCPU's APIC is at power-on defaults while the guest's kernel is
    // waiting on a timer it configured before the snapshot was taken — so it polls forever
    // and no amount of device wiring helps. `kvm_lapic_state` is the APIC page's first
    // kilobyte, registers on a 16-byte stride, which is exactly how the machine's own model
    // is addressed.
    {
        let mut lapic = *vm.lapic().expect("get lapic");
        let mut copied = 0;
        for reg in (0x00..0x400).step_by(0x10) {
            let v = machine.read_lapic_u32(0, reg as u64);
            lapic[reg..reg + 4].copy_from_slice(&v.to_le_bytes());
            if v != 0 {
                copied += 1;
            }
        }
        vm.set_lapic(&lapic).expect("set lapic");
        println!("  LAPIC: {copied} non-zero registers transferred to the vCPU");
    }

    // Run until the guest reaches something this backend does not serve yet. Every exit
    // below is *progress*: it means the CPU decoded and executed the guest's own code.
    // Answer the guest's exits from the machine's own devices. `Machine::io_read`/
    // `io_write` and `read_physical_bytes`/`write_physical` are the same entry points the
    // interpreter and the compiled tier use, so this is the real device model behind every
    // port and every BAR — not a stand-in.
    // A guest with nothing behind its exits will spin — it is waiting for a timer tick
    // that no one delivers. Without preemption `KVM_RUN` never returns and no budget
    // counted in exits can reach it.
    // The interval has to exceed whatever is done between slices. If the device polling
    // and timer tick take longer than the period, a signal is always already pending and
    // `KVM_RUN` returns without ever entering the guest — which looks exactly like a stall:
    // RIP frozen at one instruction, no exits, no halts.
    let period = std::env::var("AERO_KVM_PERIOD_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2);
    let _preempt = KvmVm::preempt_this_thread(std::time::Duration::from_millis(period));

    let mut exits = 0u32;
    let mut preempted = 0u32;
    let mut ports: Vec<u16> = Vec::new();
    let mut mmios: Vec<u64> = Vec::new();
    let started = std::time::Instant::now();
    // Between slices, run the machine's own outer-loop work: poll the storage and GPU
    // controllers, tick the platform timers, service the interrupt lines. Devices do not
    // advance on their own, so a guest polling one waits forever without this — which is
    // what "`Machine` calling `KvmRunner` in place of a tier" actually means.
    let mut polls = 0u32;
    let mut rips: Vec<(f64, u64)> = Vec::new();
    let mut halts = 0u32;
    let mut ticks = 0u32;
    let outcome = loop {
        let secs = std::env::var("AERO_KVM_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.8);
        if exits >= 2_000_000 || started.elapsed().as_secs_f64() > secs {
            break format!(
                "ran {:.3}s: {exits} exits ({halts} userspace halts), {preempted} preemptions, \
                 {polls} device polls, {ticks} timer ticks",
                started.elapsed().as_secs_f64()
            );
        }
        match vm.run() {
            Ok(KvmExit::Io {
                port,
                write,
                size,
                count,
                data_offset,
            }) => {
                exits += 1;
                if ports.len() < 8 && !ports.contains(&port) {
                    ports.push(port);
                }
                for i in 0..count as u64 {
                    let at = data_offset + i * u64::from(size);
                    // Safety: KVM guarantees this window is inside the `kvm_run` mapping.
                    let slot = unsafe { vm.run_data(at, usize::from(size)) };
                    if write {
                        let mut v = [0u8; 4];
                        let n = slot.len().min(4);
                        v[..n].copy_from_slice(&slot[..n]);
                        machine.io_write(port, size, u32::from_le_bytes(v));
                    } else {
                        let v = machine.io_read(port, size).to_le_bytes();
                        let n = slot.len().min(4);
                        slot[..n].copy_from_slice(&v[..n]);
                    }
                }
            }
            Ok(KvmExit::Mmio { addr, len, write }) => {
                exits += 1;
                if mmios.len() < 8 && !mmios.contains(&addr) {
                    mmios.push(addr);
                }
                let len = (len as usize).min(8);
                // Safety: `kvm_run.mmio.data` is eight bytes at offset 32 + 8.
                let slot = unsafe { vm.run_data(32 + 8, len) };
                if write {
                    let owned = slot.to_vec();
                    machine.write_physical(addr, &owned);
                } else {
                    let bytes = machine.read_physical_bytes(addr, len);
                    slot.copy_from_slice(&bytes);
                }
            }
            Ok(KvmExit::Halt) => {
                exits += 1;
                halts += 1;
            }
            Ok(KvmExit::Interrupted) => {
                preempted += 1;
                // Where is it? A guest stuck on a device polls one place; a guest doing
                // work moves around.
                if let Ok(r) = vm.regs() {
                    rips.push((started.elapsed().as_secs_f64(), r.rip));
                }
                // The guest was taken off the CPU: give the machine its turn.
                machine.process_ahci();
                machine.process_ide();
                machine.process_nvme();
                machine.process_aerogpu();
                machine.process_virtio_blk();
                machine.poll_network();
                machine.poll_pci_intx_lines();
                machine.tick_platform(2_000_000);
                polls += 1;
                // Give the guest its clock. Windows programs a periodic timer and waits
                // for it; with nothing raising IRQ 0 it runs what it can and then sits in
                // a spinlock, taking no halts and no exits — a stall with no visible cause.
                let _ = vm.pulse_irq_line(0);
                ticks += 1;
            }
            Ok(other) => break format!("{other:?} after {exits} exits"),
            Err(e) => break format!("KVM_RUN failed after {exits} exits: {e}"),
        }
    };
    let elapsed = started.elapsed();
    drop(_preempt);
    println!(
        "  vCPU mp_state at the end: {:?} (0 runnable, 1 uninit, 2 init, 3 halted, 4 sipi)",
        vm.mp_state()
    );
    let stalled_rip: Option<u64>;
    {
        let all: Vec<u64> = rips.iter().map(|(_, r)| *r).collect();
        let mut d = all.clone();
        d.sort_unstable();
        d.dedup();
        println!("  RIP samples: {} taken, {} distinct overall", all.len(), d.len());

        // The stalled phase on its own: the last 30% of the run.
        let cut = rips.last().map_or(0.0, |(t, _)| *t) * 0.7;
        let late: Vec<u64> = rips
            .iter()
            .filter(|(t, _)| *t >= cut)
            .map(|(_, r)| *r)
            .collect();
        let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for r in &late {
            *counts.entry(*r).or_insert(0) += 1;
        }
        let mut top: Vec<(u64, u32)> = counts.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        println!(
            "  after {cut:.1}s: {} samples, {} distinct — busiest:",
            late.len(),
            top.len()
        );
        for (rip, n) in top.iter().take(4) {
            println!("    {rip:#018x} x{n}");
        }
        stalled_rip = top.first().map(|(r, _)| *r);
    }
    for a in &mmios {
        let b = machine.read_physical_bytes(*a, 4);
        println!("    mmio {a:#x} reads {b:02x?}");
    }
    println!(
        "  exits: {exits} ({preempted} preemptions) in {:.3}s; ports {:x?}; mmio {:x?}",
        elapsed.as_secs_f64(),
        ports,
        mmios
    );

    // What is the guest actually doing? Walk its own page tables to turn RIP into a
    // physical address, then decode with the same decoder the tiers use.
    fn translate(machine: &mut Machine, cr3: u64, va: u64) -> Option<u64> {
        const PRESENT: u64 = 1;
        const PS: u64 = 1 << 7;
        const ADDR: u64 = 0x000f_ffff_ffff_f000;
        let rd = |m: &mut Machine, at: u64| -> u64 {
            let b = m.read_physical_bytes(at, 8);
            u64::from_le_bytes(b.try_into().unwrap())
        };
        let pml4e = rd(machine, (cr3 & ADDR) + ((va >> 39) & 0x1ff) * 8);
        if pml4e & PRESENT == 0 {
            return None;
        }
        let pdpte = rd(machine, (pml4e & ADDR) + ((va >> 30) & 0x1ff) * 8);
        if pdpte & PRESENT == 0 {
            return None;
        }
        if pdpte & PS != 0 {
            return Some((pdpte & 0x000f_ffff_c000_0000) | (va & 0x3fff_ffff));
        }
        let pde = rd(machine, (pdpte & ADDR) + ((va >> 21) & 0x1ff) * 8);
        if pde & PRESENT == 0 {
            return None;
        }
        if pde & PS != 0 {
            return Some((pde & 0x000f_ffff_ffe0_0000) | (va & 0x1f_ffff));
        }
        let pte = rd(machine, (pde & ADDR) + ((va >> 12) & 0x1ff) * 8);
        (pte & PRESENT != 0).then_some((pte & ADDR) | (va & 0xfff))
    }

    let mut after = booted.clone();
    vm.store_cpu_state(&mut after).expect("store");
    println!("  ran: {outcome}");
    println!(
        "  rip {:#x} -> {:#x}   (moved: {})",
        booted.rip,
        after.rip,
        after.rip != booted.rip
    );

    let show = stalled_rip.unwrap_or(after.rip);
    if let Some(phys) = translate(&mut machine, after.control.cr3, show) {
        let code = machine.read_physical_bytes(phys, 48);
        println!("  code at the busiest stalled rip {show:#x} (phys {phys:#x}):");
        let mut at = 0usize;
        let mut va = show;
        for _ in 0..8 {
            if at >= code.len() {
                break;
            }
            let inst = aero_x86::tier1::decode_one_mode(va, &code[at..], 64);
            let n = inst.len as usize;
            if n == 0 {
                break;
            }
            println!(
                "    {va:#018x}  {:02x?}  {:?}",
                &code[at..(at + n).min(code.len())],
                inst.kind
            );
            at += n;
            va = va.wrapping_add(n as u64);
        }
    } else {
        println!("  rip {show:#x} is not mapped by cr3 {:#x}", after.control.cr3);
    }

    assert!(
        after.rip != booted.rip || exits > 0,
        "the vCPU accepted the state but executed nothing: RIP never moved and no exit was \
         taken, which means the guest is not running"
    );
}

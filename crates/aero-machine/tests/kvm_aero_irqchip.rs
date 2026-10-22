//! The booted guest running on hardware with *Aero's* interrupt controllers.
//!
//! The companion test (`kvm_runs_the_booted_guest`) gives KVM its in-kernel chips and the
//! guest reaches `HLT` and stays there: Windows programmed Aero's IOAPIC before the
//! snapshot was taken, so the kernel's chips sit at reset defaults and the timer the guest
//! is waiting on is routed by a controller that is not the one being asked.
//!
//! Translating one controller's state into the other's is the obvious fix and the wrong
//! one — it has to be redone for every device that can raise a line, and it leaves two
//! models of the same hardware that have to agree. The cheaper architecture is to not
//! create the in-kernel chips at all. Then:
//!
//! * the guest's local-APIC page is backed by nothing, so every access to it leaves the
//!   vCPU as an MMIO exit and is answered from `Machine`'s own LAPIC model;
//! * `HLT` becomes a userspace exit instead of a kernel-side idle, so the idle guest is
//!   visible and can be held there rather than resumed past the instruction it used to
//!   sleep on;
//! * and the vector to deliver comes from the same PIC/IOAPIC/LAPIC path the interpreter
//!   and the compiled tier use, via [`Machine::peek_external_interrupt_vector`] and
//!   [`Machine::take_external_interrupt_vector`], through KVM's interrupt-window handshake.
//!
//! What that buys is a guest that keeps running: the point of the test is the interval
//! *after* the first idle, which is where the reset-defaults in-kernel version stops dead.
//!
//! What it costs is the reason `kvm_seeded_irqchip` exists. Every `CR8` write is a vmexit
//! when the APIC is in userspace, and a Windows kernel changes IRQL constantly: of 348,648
//! exits in four seconds, **344,399 were TPR exits** — 99%, none of which this loop has any
//! work to do for. Differenced under `perf stat -e instructions:G`, the sustained rate is
//! 85.7 MIPS from 1s to 5s and 55.5 MIPS from 5s to 9s, against 2,480 MIPS for the seeded
//! in-kernel path.
//!
//! The vector histogram this prints is what made the seeded version possible: the
//! controllers pick `0xd1` 69 times out of 111, which is how its IOAPIC pin was identified
//! rather than guessed at.
//!
//! `AERO_KVM_SECS` sets how long to run. Two runs at different budgets, differenced under
//! `perf stat -e instructions:G`, give a sustained guest-instruction rate with the restore
//! and warm-up cancelled out — the same differencing the compiled tier is measured with.
//!
//! Skips where `/dev/kvm` or the snapshot is absent, so it is inert anywhere but this
//! machine.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_aero_irqchip -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{KvmError, KvmExit, KvmVm, SharedGuestRam};
use aero_machine::{Machine, MachineConfig};
use std::time::{Duration, Instant};

const SNAPSHOT: &str = "/root/aero-work/win7-desktop.bin";
const RAM_BYTES: usize = 2048 << 20;
/// The local APIC's architectural page. With no in-kernel APIC, this is plain MMIO.
const LAPIC_BASE: u64 = 0xFEE0_0000;

/// Read guest code at a virtual address by walking the guest's own page tables.
///
/// `Machine` has no public virtual read — the tiers translate through the MMU they own —
/// and the vCPU's translation is not reachable from here either, so the walk is done by
/// hand against physical memory. Long mode, 4-level, honouring 1GiB and 2MiB leaves.
/// Returns empty on any unmapped level rather than guessing.
fn read_guest_code(machine: &mut Machine, cr3: u64, va: u64, len: usize) -> Vec<u8> {
    let mut table = cr3 & 0x000f_ffff_ffff_f000;
    for level in (0..4).rev() {
        let idx = (va >> (12 + 9 * level)) & 0x1ff;
        let e = machine.read_physical_bytes(table + idx * 8, 8);
        let e = u64::from_le_bytes(e.try_into().unwrap_or([0; 8]));
        if e & 1 == 0 {
            return Vec::new();
        }
        // A PS leaf below the PTE level ends the walk with a large page.
        if level > 0 && e & (1 << 7) != 0 {
            let mask = (1u64 << (12 + 9 * level)) - 1;
            let phys = (e & 0x000f_ffff_ffff_f000 & !mask) | (va & mask);
            return machine.read_physical_bytes(phys, len);
        }
        table = e & 0x000f_ffff_ffff_f000;
    }
    machine.read_physical_bytes(table | (va & 0xfff), len)
}

/// One turn of the machine's outer loop: nothing here advances on its own.
fn service(machine: &mut Machine) {
    machine.process_ahci();
    machine.process_ide();
    machine.process_nvme();
    machine.process_aerogpu();
    machine.process_virtio_blk();
    machine.poll_network();
    machine.poll_pci_intx_lines();
    machine.tick_platform(2_000_000);
}

#[test]
fn the_guest_keeps_running_on_aeros_own_interrupt_controllers() {
    let Ok(snap) = std::fs::read(SNAPSHOT) else {
        println!("  no snapshot at {SNAPSHOT}; skipping");
        return;
    };

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
    assert!(
        booted.control.cr3 != 0 && booted.msr.efer & (1 << 10) != 0,
        "this snapshot is not a booted long-mode guest; the rest of the test means nothing"
    );

    // No `KVM_CREATE_IRQCHIP`. Safety: `ram` outlives `vm`, is page-aligned, and its
    // mapping does not move — the machine holds a clone of the same `Arc`.
    let mut vm = match unsafe { KvmVm::new_from_raw(ram_base, ram_len, 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };
    vm.load_cpu_state(&booted)
        .expect("the booted state must be acceptable to a vCPU");

    let secs = std::env::var("AERO_KVM_SECS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.0);
    // A guest that spins rather than halting still has to be taken off the CPU.
    let period = std::env::var("AERO_KVM_PERIOD_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2);
    let _preempt = KvmVm::preempt_this_thread(Duration::from_millis(period));

    let mut exits = 0u64;
    let mut halts = 0u64;
    let mut injected = 0u64;
    let mut windows = 0u64;
    let mut lost = 0u64;
    let mut lapic_reads = 0u64;
    let mut lapic_writes = 0u64;
    let mut tpr_exits = 0u64;
    let mut want_window = false;
    let mut cr8 = booted.control.cr8;
    // Where the guest is over time, sampled coarsely. A guest making progress moves; one
    // wedged on a device does not. Bucketed by half-second so the tail is comparable to
    // the head rather than drowned by it.
    let mut buckets: Vec<(u64, u64, u64, std::collections::HashSet<u64>)> = Vec::new();
    // Exact RIPs, not just pages: a guest that never halts is spinning, and the only way
    // to say what it is waiting for is to read the instructions it keeps re-executing.
    let mut hot: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    // Which vectors the guest is actually being given. This is the authoritative answer to
    // "what is Windows waiting for" — the controllers chose these, nothing here guessed.
    let mut vectors: std::collections::HashMap<u8, u64> = std::collections::HashMap::new();
    let started = Instant::now();
    let bucket_of = |t: f64| (t * 2.0) as usize;

    let outcome = loop {
        let now = started.elapsed().as_secs_f64();
        if now > secs {
            break format!("ran {now:.3}s to budget");
        }
        let b = bucket_of(now);
        while buckets.len() <= b {
            buckets.push((0, 0, 0, std::collections::HashSet::new()));
        }

        // Offer a vector if the guest can take one; otherwise ask to be told when it can.
        // The peek matters: acknowledging a vector we cannot deliver would clear the
        // controller's latch and lose the interrupt.
        if machine.peek_external_interrupt_vector(0, cr8).is_some() {
            if vm.ready_for_interrupt() && vm.interrupts_enabled() {
                if let Some(v) = machine.take_external_interrupt_vector(0, true, cr8) {
                    match vm.inject_interrupt(v) {
                        Ok(true) => {
                            injected += 1;
                            buckets[b].2 += 1;
                            *vectors.entry(v).or_insert(0u64) += 1;
                        }
                        // The vector was acknowledged before the injection was attempted,
                        // so a refusal here loses it outright. Counted separately: a run
                        // where this is not zero is delivering fewer interrupts than the
                        // controllers think it is.
                        _ => lost += 1,
                    }
                }
                want_window = false;
            } else if !want_window {
                want_window = true;
                windows += 1;
            }
        } else if want_window {
            want_window = false;
        }
        vm.request_interrupt_window(want_window);

        match vm.run() {
            Ok(KvmExit::Io {
                port,
                write,
                size,
                count,
                data_offset,
            }) => {
                exits += 1;
                buckets[b].0 += 1;
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
                buckets[b].0 += 1;
                let len = (len as usize).min(8);
                // Safety: `kvm_run.mmio.data` is eight bytes at offset 32 + 8.
                let slot = unsafe { vm.run_data(32 + 8, len) };
                if addr & !0xFFF == LAPIC_BASE {
                    // The guest's own APIC, answered from the model the machine has been
                    // keeping all along — which is the whole point of not creating the
                    // in-kernel one.
                    let off = addr & 0xFFF;
                    if write {
                        let mut v = [0u8; 4];
                        let n = len.min(4);
                        v[..n].copy_from_slice(&slot[..n]);
                        machine.write_lapic_u32(0, off, u32::from_le_bytes(v));
                        lapic_writes += 1;
                    } else {
                        let v = machine.read_lapic_u32(0, off).to_le_bytes();
                        let n = len.min(4);
                        slot[..n].copy_from_slice(&v[..n]);
                        lapic_reads += 1;
                    }
                } else if write {
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
                buckets[b].1 += 1;
                // The guest asked to sleep until an interrupt. Re-entering now would
                // resume it past the `HLT` without one, which is how an idle guest turns
                // into a busy loop that looks like progress. Hold it here and run the
                // machine until a vector actually exists.
                loop {
                    service(&mut machine);
                    if machine.peek_external_interrupt_vector(0, cr8).is_some() {
                        break;
                    }
                    if started.elapsed().as_secs_f64() > secs {
                        break;
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
            Ok(KvmExit::IrqWindowOpen) => {
                exits += 1;
                want_window = false;
            }
            Ok(KvmExit::SetTpr) => {
                exits += 1;
                tpr_exits += 1;
            }
            Ok(KvmExit::Interrupted) => {
                if let Ok(r) = vm.regs() {
                    buckets[b].3.insert(r.rip & !0xFFF);
                    *hot.entry(r.rip).or_insert(0u64) += 1;
                }
                service(&mut machine);
            }
            Ok(other) => break format!("{other:?} after {exits} exits"),
            Err(e) => break format!("KVM_RUN failed after {exits} exits: {e}"),
        }
        // CR8 is the guest's IRQL and gates which vectors the LAPIC will hand over, so it
        // has to be re-read rather than remembered.
        if let Ok(s) = vm.sregs() {
            cr8 = s.cr8;
        }
    };

    println!("  {outcome}");
    println!(
        "  {exits} exits, {halts} halts, {injected} interrupts injected ({lost} lost), \
         {windows} window requests, LAPIC {lapic_reads}r/{lapic_writes}w, {tpr_exits} TPR \
         exits"
    );
    for (i, (e, h, inj, rips)) in buckets.iter().enumerate() {
        println!(
            "    t={:.1}-{:.1}s: {e} exits, {h} halts, {inj} injected, {} distinct code pages",
            i as f64 / 2.0,
            (i + 1) as f64 / 2.0,
            rips.len()
        );
    }
    {
        let mut vs: Vec<(u8, u64)> = vectors.into_iter().collect();
        vs.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        println!("  vectors delivered: {vs:02x?}");
    }
    {
        let mut top: Vec<(u64, u64)> = hot.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        let cr3 = vm.sregs().map(|s| s.cr3).unwrap_or(0);
        for (rip, n) in top.iter().take(6) {
            let bytes = read_guest_code(&mut machine, cr3, *rip, 24);
            let hex = bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join("");
            println!("    hot rip {rip:#018x} x{n}: {hex}");
        }
    }
    if let Ok(mp) = vm.mp_state() {
        println!("  mp_state = {mp} (0 = runnable, 3 = halted)");
    }

    assert!(
        injected > 0,
        "no interrupt was ever delivered: the guest cannot be kept alive without one"
    );
    assert_eq!(
        lost, 0,
        "{lost} vectors were acknowledged and then refused by the vCPU, so the controllers \
         and the guest disagree about what has been delivered"
    );
    // The claim under test is the *tail*, not the head: an in-kernel-irqchip run reaches
    // its first idle and never leaves it, so a late bucket doing work is the whole result.
    let late: u64 = buckets
        .iter()
        .skip(buckets.len().saturating_sub(2))
        .map(|(e, _, inj, _)| e + inj)
        .sum();
    assert!(
        late > 0,
        "the guest stopped making progress before the run ended: {outcome}"
    );
}

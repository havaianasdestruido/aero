//! The booted guest on hardware with KVM's in-kernel chips *seeded from Aero's*.
//!
//! Three ways to give a restored guest its interrupts, and this is the third:
//!
//! 1. **In-kernel chips at reset defaults** (`kvm_runs_the_booted_guest`). Fast, and then
//!    permanently stuck: every redirection entry is masked, so the timer the guest is
//!    waiting for is routed by nobody.
//! 2. **No in-kernel chips at all** (`kvm_aero_irqchip`). Correct — Aero's own controllers
//!    pick the vectors and the guest keeps running — but a userspace APIC costs a vmexit on
//!    every `CR8` write, and a Windows kernel changes IRQL constantly: 89,000 TPR exits a
//!    second, and the sustained rate falls to 55-86 MIPS.
//! 3. **In-kernel chips, seeded.** The guest programmed its IOAPIC before the snapshot was
//!    taken, and that programming is still sitting in Aero's model. Copying it into KVM's
//!    chip is a few hundred bytes, done once, and it buys the whole in-kernel path: no TPR
//!    exits, no APIC MMIO exits, `HLT` handled without leaving the kernel.
//!
//! The seeding, the exit loop and the clock all live in [`aero_machine::kvm_mode::KvmMode`];
//! this test drives that mode rather than restating it, so what is measured here is what a
//! caller would actually get. The redirection table is read back through the IOAPIC's own
//! index/data window at `0xFEC00000` — the same registers the guest wrote — rather than
//! through a private accessor, so what gets copied is by construction what the guest asked
//! for. It says the
//! clock is on pin 2 at vector `0xd1` in logical destination mode, which the seeded
//! `LDR`/`DFR` (flat, logical id 1) resolve to this one vCPU. Vector `0xd1` is also what
//! run 2's controllers pick on their own, which is how the pin was confirmed rather than
//! guessed.
//!
//! **What seeding buys, by `perf stat -e instructions:G` at two run budgets:**
//!
//! | guest instructions | at 6s | at 12s | over that window |
//! |---|---|---|---|
//! | reset defaults | 1,175,787,584 | 1,175,769,098 | **0** |
//! | seeded | 1,190,356,959 | 1,206,044,182 | **15,687,223** |
//!
//! Zero is the point. The unseeded guest is not slow, it is unreachable: nothing can wake
//! it again, ever. The seeded one ticks over at about 2.6 MIPS — 64 clock interrupts a
//! second at roughly 27,000 instructions of ISR and DPC each, which is what an idle Windows
//! desktop costs.
//!
//! **The rate while it has work to do**, same instrument, differenced across budgets:
//!
//! | window | guest instructions | rate |
//! |---|---|---|
//! | 0 → 0.3s | 744M | **2,480 MIPS** |
//! | 0.3 → 0.6s | 165M | 550 MIPS |
//! | 0.6 → 1.0s | 159M | 397 MIPS |
//! | 1.0 → 3.0s | 117M | 58 MIPS |
//!
//! The taper is not the backend slowing down. The guest gets through about 1.19 billion
//! instructions of real Windows — the work a resumed desktop has queued up — and then runs
//! out of things to do. The compiled tier needs roughly nine seconds for that same 1.19
//! billion.
//!
//! Which makes the end state the delicate part of the test, because "idle" and "wedged"
//! both read as `mp_state = HALTED`, and both leave the guest at the same `rip` with the
//! same flags. What separates them is whether a delivered vector is ever *taken*. Sampling
//! `IRR` once cannot tell: a tick that arrived a microsecond ago is sitting there
//! legitimately, and at a fast enough clock that is most of the time — which is exactly the
//! false reading that first made the tick rate look like it mattered here. It does not.
//! So the test stops the clock at the end and watches `IRR` **drain**, and separately
//! requires `IF` set, so a park at the `cli; hlt` where Windows leaves a CPU it has given
//! up on cannot pass.
//!
//! `AERO_KVM_HZ` sets the clock rate, default 64. The guest tolerates 500 as well.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_seeded_irqchip -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::SharedGuestRam;
use aero_machine::kvm_mode::{KvmMode, SliceOutcome};
use aero_machine::{Machine, MachineConfig};
use std::time::{Duration, Instant};

const SNAPSHOT: &str = "/root/aero-work/win7-desktop.bin";
const RAM_BYTES: usize = 2048 << 20;
const IOAPIC_BASE: u64 = 0xFEC0_0000;
const PINS: usize = 24;

/// Read one 32-bit IOAPIC register through the guest-visible index/data window.
fn ioapic_reg(machine: &mut Machine, index: u8) -> u32 {
    machine.write_physical(IOAPIC_BASE, &[index, 0, 0, 0]);
    let d = machine.read_physical_bytes(IOAPIC_BASE + 0x10, 4);
    u32::from_le_bytes(d.try_into().unwrap_or([0; 4]))
}

#[test]
fn seeding_the_in_kernel_chips_keeps_the_guest_running() {
    let Ok(snap) = std::fs::read(SNAPSHOT) else {
        println!("  no snapshot at {SNAPSHOT}; skipping");
        return;
    };

    // One memory, both viewers. Two allocations means the vCPU writes a page table into
    // its own copy while a device model reads the machine's, and the guest executes
    // happily and never progresses.
    let ram = SharedGuestRam::new(RAM_BYTES);
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

    // What the guest actually programmed, read the same way the mode reads it. A masked
    // entry has bit 16 set; a live one carries its vector in bits 0..8.
    let mut live = Vec::new();
    for pin in 0..PINS {
        let lo = ioapic_reg(&mut machine, 0x10 + 2 * pin as u8);
        let hi = ioapic_reg(&mut machine, 0x11 + 2 * pin as u8);
        let e = u64::from(lo) | (u64::from(hi) << 32);
        if e != 0 && e & (1 << 16) == 0 {
            println!(
                "  IOAPIC pin {pin:2}: vector {:#04x} dest {:#04x} {} ({e:#018x})",
                e & 0xff,
                (e >> 56) & 0xff,
                if e & (1 << 15) != 0 { "level" } else { "edge " },
            );
            live.push(pin);
        }
    }
    assert!(
        !live.is_empty(),
        "the guest's IOAPIC has no unmasked entry, so there is nothing to seed and the \
         snapshot cannot be what it claims"
    );

    // Safety: `ram` is the very allocation the machine was built on — the same `Arc` — and
    // it outlives the mode. This is also the thread that will run the slices.
    let mut mode = match unsafe { KvmMode::new(&mut machine, &ram) } {
        Ok(m) => m,
        Err(e) => {
            println!("  no usable vCPU ({e}); skipping");
            return;
        }
    };
    if let Ok(hz) = std::env::var("AERO_KVM_HZ") {
        if let Ok(hz) = hz.parse::<f64>() {
            mode.set_clock_hz(hz);
        }
    }
    println!("  clock on IOAPIC pin {}", mode.timer_pin());

    let secs = std::env::var("AERO_KVM_SECS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.0);
    let started = Instant::now();
    let mut total = SliceOutcome::default();
    // Half-second slices, so the tail of the run is reported separately from the head:
    // this guest does its work in the first second and idles afterwards, and collapsing
    // the two would hide exactly the thing being tested.
    while started.elapsed().as_secs_f64() < secs {
        let out = mode.run_slice(&mut machine, Duration::from_millis(500));
        println!(
            "    t={:.1}s: {} exits, {} preemptions, {} ticks{}",
            started.elapsed().as_secs_f64(),
            out.exits,
            out.preemptions,
            out.ticks,
            out.stopped.as_deref().unwrap_or("")
        );
        total.exits += out.exits;
        total.ticks += out.ticks;
        total.preemptions += out.preemptions;
        if let Some(stop) = out.stopped {
            total.stopped = Some(stop);
            break;
        }
    }
    println!(
        "  {} exits, {} preemptions, {} ticks over {:.2}s",
        total.exits,
        total.preemptions,
        total.ticks,
        started.elapsed().as_secs_f64()
    );
    assert!(
        total.stopped.is_none(),
        "the guest reached an exit the mode does not serve: {:?}",
        total.stopped
    );

    // "Halted" is where a healthy idle guest and a dead one both end up, at the same `rip`
    // with the same flags, so the assertions below are about *how* it got there.
    assert!(total.ticks > 0, "the clock never ticked; nothing was asked of the guest");
    assert!(
        total.exits > 100,
        "the guest never reached its devices, so it did not run"
    );

    // Sampling `IRR` once proves nothing: a tick that arrived a microsecond ago is sitting
    // there legitimately, and the faster the clock the likelier that is — which is the
    // false reading that first made the tick rate look like it mattered here. Stop the
    // clock instead and require the register to *drain*.
    mode.set_clock_hz(0.0);
    let deadline = Instant::now() + Duration::from_millis(200);
    let left = loop {
        let lapic = mode.vm().lapic().expect("read the local APIC back");
        let reg = |o: usize| u32::from_le_bytes(lapic[o..o + 4].try_into().unwrap());
        let pending: Vec<usize> = (0..8).filter(|i| reg(0x200 + i * 0x10) != 0).collect();
        if pending.is_empty() {
            break None;
        }
        if Instant::now() >= deadline {
            break Some(pending);
        }
        mode.run_slice(&mut machine, Duration::from_millis(10));
    };
    println!("  IRR after the clock stopped: {left:?}");
    assert!(
        left.is_none(),
        "vectors stayed undelivered in IRR words {left:?} after 200ms with the clock \
         stopped: the guest is being sent interrupts it never takes"
    );

    let regs = mode.vm().regs().expect("read the guest's registers");
    println!(
        "  rip={:#x} rflags={:#x} mp_state={}",
        regs.rip,
        regs.rflags,
        mode.vm().mp_state().unwrap_or(u32::MAX)
    );
    assert!(
        regs.rflags & (1 << 9) != 0,
        "the guest is halted with interrupts disabled at rip {:#x} — parked at a `cli; hlt`, \
         not idling",
        regs.rip
    );
}

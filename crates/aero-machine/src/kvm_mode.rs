//! Running the machine's guest on a vCPU.
//!
//! [`crate::kvm`] is the ioctl layer: it can create a VM, load architectural state, and
//! turn one `KVM_RUN` into a typed exit. This is the execution mode built on top — it owns
//! the loop, answers the guest's exits from the machine's own device models, and keeps the
//! clock running, which is what the interpreter and the compiled tier's drivers do for
//! their own engines.
//!
//! Two things have to be right before a restored guest will run at all, and neither is
//! obvious from the ioctl layer:
//!
//! * **The interrupt controllers have to be seeded.** KVM's in-kernel chips come up at
//!   power-on defaults with every IOAPIC redirection entry masked. A guest restored from a
//!   snapshot programmed its controllers long before that, so raising a line on KVM's chips
//!   does nothing whatsoever, and the guest waits for a timer tick that can never arrive.
//!   Measured on the Windows 7 snapshot: 1,175,787,584 guest instructions at six seconds
//!   and 1,175,769,098 at twelve — not a slow guest, a stopped one. Seeded, the same window
//!   moves 15,687,223 instructions, which is an idle desktop servicing its clock.
//! * **The vCPU has to be preemptible.** A guest that spins never leaves `KVM_RUN`, and no
//!   budget expressed in exits can reach it. [`crate::kvm::KvmVm::preempt_this_thread`]
//!   arranges a per-thread signal, which is why [`KvmMode::new`] must be called on the same
//!   thread that will call [`KvmMode::run_slice`].
//!
//! The redirection table is read back through the IOAPIC's own index/data window rather
//! than a private accessor, so what gets copied is by construction what the guest wrote.

use crate::kvm::{KvmError, KvmExit, KvmVm, Preemption, SharedGuestRam};
use crate::Machine;
use std::time::{Duration, Instant};

/// The IOAPIC's architectural MMIO window.
const IOAPIC_BASE: u64 = 0xFEC0_0000;
/// `chip_id` for the IOAPIC in `struct kvm_irqchip`.
const CHIP_IOAPIC: u32 = 2;
/// `struct kvm_ioapic_state`: `base_address`, `ioregsel`, `id`, `irr`, pad, then the table.
const REDIRTBL_OFF: usize = 24;
const PINS: usize = 24;
/// What a Windows kernel's clock is built for. Faster is tolerated but pointless.
const DEFAULT_HZ: f64 = 64.0;

/// How often the vCPU is taken off the CPU so the machine can have its turn.
///
/// This has to exceed the work done between slices. If servicing the devices takes longer
/// than the interval, a signal is always already pending and `KVM_RUN` returns without ever
/// entering the guest — which looks exactly like a stall: `rip` frozen on one instruction,
/// no exits, no halts.
const PREEMPT_INTERVAL: Duration = Duration::from_millis(2);

/// What one call to [`KvmMode::run_slice`] did.
#[derive(Debug, Default, Clone)]
pub struct SliceOutcome {
    /// Exits answered from the machine's devices.
    pub exits: u64,
    /// Clock interrupts raised.
    pub ticks: u64,
    /// Times the vCPU was taken off the CPU so the machine could have its turn.
    pub preemptions: u64,
    /// Set when the loop stopped early because the guest reached something this mode does
    /// not serve. `None` means the budget simply ran out.
    pub stopped: Option<String>,
}

/// A guest executing on hardware, driven from a [`Machine`].
pub struct KvmMode {
    vm: KvmVm,
    timer_pin: u32,
    tick_period: Duration,
    last_tick: Instant,
    _preempt: Preemption,
}

/// Read one 32-bit IOAPIC register through the guest-visible index/data window.
fn ioapic_reg(machine: &mut Machine, index: u8) -> u32 {
    machine.write_physical(IOAPIC_BASE, &[index, 0, 0, 0]);
    let d = machine.read_physical_bytes(IOAPIC_BASE + 0x10, 4);
    u32::from_le_bytes(d.try_into().unwrap_or([0; 4]))
}

impl KvmMode {
    /// Put the machine's current CPU state on a vCPU and prepare to run it.
    ///
    /// Must be called on the thread that will drive [`Self::run_slice`], because the
    /// preemption signal is delivered to a specific thread.
    ///
    /// # Safety
    ///
    /// `ram` must be the very guest memory this `Machine` was built on — the same
    /// allocation behind the same `Arc`, not a copy. Two allocations means the vCPU writes
    /// a page table into one while a device model reads the other, and the guest executes
    /// happily and never progresses. `ram` must also outlive the returned `KvmMode`.
    pub unsafe fn new(machine: &mut Machine, ram: &SharedGuestRam) -> Result<Self, KvmError> {
        // Safety: forwarded to the caller's obligation above.
        let mut vm = unsafe { KvmVm::new_from_raw_with_irqchip(ram.base(), ram.len(), 0)? };
        vm.load_cpu_state(machine.cpu())?;

        // The vCPU's local APIC, from the machine's model: 1KiB of register file on a
        // 16-byte stride, which is how the machine's own model is addressed too. This
        // carries the `LDR`/`DFR` that logical-destination redirection entries resolve
        // against, so it has to happen before the IOAPIC is seeded to be of any use.
        let mut lapic = *vm.lapic()?;
        for reg in (0x00..0x400).step_by(0x10) {
            let v = machine.read_lapic_u32(0, reg as u64);
            lapic[reg..reg + 4].copy_from_slice(&v.to_le_bytes());
        }
        vm.set_lapic(&lapic)?;

        // The IOAPIC's redirection table. Everything else in `kvm_ioapic_state` — base
        // address, id, the pending-interrupt bitmap — is left as KVM initialised it:
        // copying a stale `irr` would assert lines the guest has already serviced.
        let mut rte = [0u64; PINS];
        for (pin, slot) in rte.iter_mut().enumerate() {
            let lo = ioapic_reg(machine, 0x10 + 2 * pin as u8);
            let hi = ioapic_reg(machine, 0x11 + 2 * pin as u8);
            *slot = u64::from(lo) | (u64::from(hi) << 32);
        }
        let mut chip = *vm.irqchip(CHIP_IOAPIC)?;
        for (pin, e) in rte.iter().enumerate() {
            let at = REDIRTBL_OFF + pin * 8;
            chip[at..at + 8].copy_from_slice(&e.to_le_bytes());
        }
        vm.set_irqchip(CHIP_IOAPIC, &chip)?;

        // Which pin carries the clock. IRQ 0 is the PIT and ACPI usually overrides it onto
        // GSI 2, so prefer the override and fall back to whatever the guest left unmasked.
        let live: Vec<usize> = (0..PINS)
            .filter(|&p| rte[p] != 0 && rte[p] & (1 << 16) == 0)
            .collect();
        let timer_pin = if live.contains(&2) {
            2
        } else if live.contains(&0) {
            0
        } else {
            live.first().copied().unwrap_or(0)
        } as u32;

        Ok(Self {
            vm,
            timer_pin,
            tick_period: Duration::from_secs_f64(1.0 / DEFAULT_HZ),
            last_tick: Instant::now(),
            _preempt: KvmVm::preempt_this_thread(PREEMPT_INTERVAL),
        })
    }

    /// The IOAPIC pin this mode raises for the guest's clock.
    #[must_use]
    pub fn timer_pin(&self) -> u32 {
        self.timer_pin
    }

    /// Set the clock rate in hertz. The default is 64. Zero stops the clock.
    ///
    /// Stopping it is how a caller tells a guest that is idle from one that is ignoring its
    /// interrupts: with nothing new arriving, a healthy guest drains its local APIC's
    /// request register and a wedged one does not.
    pub fn set_clock_hz(&mut self, hz: f64) {
        self.tick_period = if hz > 0.0 {
            Duration::from_secs_f64(1.0 / hz)
        } else {
            Duration::MAX
        };
    }

    /// The underlying VM, for state a driver needs to inspect.
    #[must_use]
    pub fn vm(&self) -> &KvmVm {
        &self.vm
    }

    /// The underlying VM, mutably.
    pub fn vm_mut(&mut self) -> &mut KvmVm {
        &mut self.vm
    }

    /// Run the guest for up to `budget`, answering its exits from `machine`.
    ///
    /// Returns early if the guest reaches an exit this mode does not serve; the vCPU is
    /// unharmed either way and the next call resumes it.
    pub fn run_slice(&mut self, machine: &mut Machine, budget: Duration) -> SliceOutcome {
        let started = Instant::now();
        let mut out = SliceOutcome::default();
        while started.elapsed() < budget {
            match self.vm.run() {
                Ok(KvmExit::Io {
                    port,
                    write,
                    size,
                    count,
                    data_offset,
                }) => {
                    out.exits += 1;
                    for i in 0..u64::from(count) {
                        let at = data_offset + i * u64::from(size);
                        // Safety: KVM guarantees this window is inside the `kvm_run`
                        // mapping.
                        let slot = unsafe { self.vm.run_data(at, usize::from(size)) };
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
                    out.exits += 1;
                    let len = (len as usize).min(8);
                    // Safety: `kvm_run.mmio.data` is eight bytes at offset 32 + 8.
                    let slot = unsafe { self.vm.run_data(32 + 8, len) };
                    if write {
                        let owned = slot.to_vec();
                        machine.write_physical(addr, &owned);
                    } else {
                        let bytes = machine.read_physical_bytes(addr, len);
                        slot.copy_from_slice(&bytes);
                    }
                }
                Ok(KvmExit::Halt) => {
                    // With in-kernel chips this is rare: the kernel handles `HLT` itself
                    // and the guest reappears here only via preemption. When it does
                    // surface, the guest is idle and the machine should have its turn.
                    out.exits += 1;
                    self.serve(machine, &mut out);
                }
                Ok(KvmExit::Interrupted) => {
                    out.preemptions += 1;
                    self.serve(machine, &mut out);
                }
                Ok(other) => {
                    out.stopped = Some(format!("{other:?}"));
                    break;
                }
                Err(e) => {
                    out.stopped = Some(format!("KVM_RUN failed: {e}"));
                    break;
                }
            }
        }
        out
    }

    /// The machine's turn: nothing in a device model advances on its own, and the guest's
    /// clock is raised here rather than by any device, because the tick has to be paced
    /// against wall time and not against how often the vCPU happens to be taken away.
    fn serve(&mut self, machine: &mut Machine, out: &mut SliceOutcome) {
        machine.process_ahci();
        machine.process_ide();
        machine.process_nvme();
        machine.process_aerogpu();
        machine.process_virtio_blk();
        machine.poll_network();
        machine.poll_pci_intx_lines();
        machine.tick_platform(2_000_000);
        if self.tick_period != Duration::MAX && self.last_tick.elapsed() >= self.tick_period {
            self.last_tick = Instant::now();
            let _ = self.vm.pulse_irq_line(self.timer_pin);
            out.ticks += 1;
        }
    }
}

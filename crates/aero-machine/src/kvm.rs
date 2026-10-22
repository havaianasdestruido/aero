//! Running guest code on the CPU instead of emulating it.
//!
//! The compiled tier reaches ~50-64 host instructions per guest instruction, which is
//! roughly 110-140 MIPS on this machine. Every remaining item found by profiling and
//! disassembly is worth a few percent, and the structure says why: a block entry costs
//! far more than the four to seven guest instructions it covers, and that is inherent to
//! translating into WebAssembly and calling into an engine.
//!
//! Hardware virtualisation does not have that structure. Measured on this machine with a
//! standalone probe, KVM runs **3,693 MIPS** — not a tuning gap, a different mechanism.
//!
//! This is the beginning of a native backend built on it. It is deliberately narrow: open
//! `/dev/kvm`, give the VM a slice of host memory as guest RAM, create a vCPU, run it, and
//! route the exits that matter. What it is *not* yet is a `Machine` execution mode — that
//! needs the booted snapshot's architectural state loaded into the vCPU, the platform's
//! interrupt controllers wired to `KVM_INTERRUPT`, and every MMIO exit routed to the device
//! models. Those are the next steps and they are large; this is the part that has to work
//! first, and it can be measured on its own.
//!
//! Linux only, and only where `/dev/kvm` is present and permits access. The browser build
//! never sees this file.

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd, RawFd};

// From `linux/kvm.h` on this machine, read out of the header rather than transcribed from
// documentation: the encodings carry struct sizes, so a wrong one is a silently wrong
// ioctl rather than a compile error.
const KVM_GET_API_VERSION: u64 = 0xae00;
const KVM_CREATE_VM: u64 = 0xae01;
const KVM_GET_VCPU_MMAP_SIZE: u64 = 0xae04;
const KVM_CREATE_VCPU: u64 = 0xae41;
const KVM_SET_USER_MEMORY_REGION: u64 = 0x4020_ae46;
const KVM_RUN: u64 = 0xae80;
const KVM_GET_REGS: u64 = 0x8090_ae81;
const KVM_SET_REGS: u64 = 0x4090_ae82;
const KVM_GET_SREGS: u64 = 0x8138_ae83;
const KVM_SET_SREGS: u64 = 0x4138_ae84;
const KVM_GET_MSRS: u64 = 0xc008_ae88;
const KVM_SET_MSRS: u64 = 0x4008_ae89;
const KVM_INTERRUPT: u64 = 0x4004_ae86;
const KVM_CREATE_IRQCHIP: u64 = 0xae60;
const KVM_IRQ_LINE: u64 = 0x4008_ae61;
const KVM_GET_MP_STATE: u64 = 0x8004_ae98;
const KVM_GET_IRQCHIP: u64 = 0xc208_ae62;
const KVM_SET_IRQCHIP: u64 = 0x8208_ae63;
const KVM_GET_LAPIC: u64 = 0x8400_ae8e;
const KVM_SET_LAPIC: u64 = 0x4400_ae8f;

/// The MSRs a booted guest's state depends on. `EFER` is not here: it travels in
/// `kvm_sregs`, and setting it through both paths is how you get a VM that enters long
/// mode in one place and not the other.
const MSR_INDICES: [(u32, MsrField); 12] = [
    (0xC000_0081, MsrField::Star),
    (0xC000_0082, MsrField::Lstar),
    (0xC000_0083, MsrField::Cstar),
    (0xC000_0084, MsrField::Fmask),
    (0xC000_0100, MsrField::FsBase),
    (0xC000_0101, MsrField::GsBase),
    (0xC000_0102, MsrField::KernelGsBase),
    (0x0000_0174, MsrField::SysenterCs),
    (0x0000_0175, MsrField::SysenterEsp),
    (0x0000_0176, MsrField::SysenterEip),
    (0x0000_0010, MsrField::Tsc),
    (0xC000_0103, MsrField::TscAux),
];

/// Which `MsrState` field an index corresponds to. An enum rather than offsets so the
/// mapping is checked by the compiler.
#[derive(Clone, Copy)]
enum MsrField {
    Star,
    Lstar,
    Cstar,
    Fmask,
    FsBase,
    GsBase,
    KernelGsBase,
    SysenterCs,
    SysenterEsp,
    SysenterEip,
    Tsc,
    TscAux,
}

impl MsrField {
    fn get(self, m: &aero_cpu_core::state::MsrState) -> u64 {
        match self {
            Self::Star => m.star,
            Self::Lstar => m.lstar,
            Self::Cstar => m.cstar,
            Self::Fmask => m.fmask,
            Self::FsBase => m.fs_base,
            Self::GsBase => m.gs_base,
            Self::KernelGsBase => m.kernel_gs_base,
            Self::SysenterCs => m.sysenter_cs,
            Self::SysenterEsp => m.sysenter_esp,
            Self::SysenterEip => m.sysenter_eip,
            Self::Tsc => m.tsc,
            Self::TscAux => u64::from(m.tsc_aux),
        }
    }

    fn set(self, m: &mut aero_cpu_core::state::MsrState, v: u64) {
        match self {
            Self::Star => m.star = v,
            Self::Lstar => m.lstar = v,
            Self::Cstar => m.cstar = v,
            Self::Fmask => m.fmask = v,
            Self::FsBase => m.fs_base = v,
            Self::GsBase => m.gs_base = v,
            Self::KernelGsBase => m.kernel_gs_base = v,
            Self::SysenterCs => m.sysenter_cs = v,
            Self::SysenterEsp => m.sysenter_esp = v,
            Self::SysenterEip => m.sysenter_eip = v,
            Self::Tsc => m.tsc = v,
            Self::TscAux => m.tsc_aux = v as u32,
        }
    }
}

/// Why the vCPU stopped. Only the exits this backend can act on are named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvmExit {
    /// `HLT` with interrupts enabled: the guest is idle until something wakes it.
    Halt,
    /// Port I/O. `data` is the value for a write, or where a read's value must go.
    Io {
        port: u16,
        write: bool,
        size: u8,
        count: u32,
        data_offset: u64,
    },
    /// An access outside every registered memory region.
    Mmio { addr: u64, len: u32, write: bool },
    /// The guest can accept an interrupt now. Only produced after
    /// [`KvmVm::request_interrupt_window`].
    IrqWindowOpen,
    /// The guest lowered its task-priority register and KVM wants the userspace APIC
    /// updated. Reached immediately by a booted Windows guest, which is why it is named:
    /// treating it as unknown reads as "the guest did not run".
    SetTpr,
    /// `KVM_RUN` returned because a signal arrived. The guest is unharmed and can be
    /// re-entered; this is how a spinning guest is taken away from the CPU.
    Interrupted,
    /// The guest triple-faulted or asked to power down.
    Shutdown,
    /// The vCPU could not be entered; `hardware_entry_failure_reason` is the detail.
    FailedEntry,
    /// Something this backend does not model yet.
    Other(u32),
}

/// Stops the preemption helper when dropped.
pub struct Preemption {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Preemption {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The general-purpose register file, in `struct kvm_regs` order.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct KvmRegs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rflags: u64,
}

/// One segment descriptor as KVM presents it.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct KvmSegment {
    pub base: u64,
    pub limit: u32,
    pub selector: u16,
    pub type_: u8,
    pub present: u8,
    pub dpl: u8,
    pub db: u8,
    pub s: u8,
    pub l: u8,
    pub g: u8,
    pub avl: u8,
    pub unusable: u8,
    pub padding: u8,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct KvmDtable {
    base: u64,
    limit: u16,
    padding: [u16; 3],
}

/// `struct kvm_sregs`. The layout is checked against the header's offsets in tests.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct KvmSregs {
    pub cs: KvmSegment,
    pub ds: KvmSegment,
    pub es: KvmSegment,
    pub fs: KvmSegment,
    pub gs: KvmSegment,
    pub ss: KvmSegment,
    pub tr: KvmSegment,
    pub ldt: KvmSegment,
    gdt: KvmDtable,
    idt: KvmDtable,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub apic_base: u64,
    interrupt_bitmap: [u64; 4],
}

#[repr(C)]
struct KvmUserspaceMemoryRegion {
    slot: u32,
    flags: u32,
    guest_phys_addr: u64,
    memory_size: u64,
    userspace_addr: u64,
}

/// Page-aligned guest RAM.
///
/// `KVM_SET_USER_MEMORY_REGION` rejects a userspace address that is not page-aligned, and
/// a `Vec<u8>` is aligned to one byte — so handing it a `Vec` fails with `EINVAL` and no
/// further explanation. Owning the allocation here means callers cannot get that wrong,
/// and it is also what the eventual `Machine` integration needs: the region has to stay
/// put for the VM's whole life, because KVM holds host addresses rather than copying.
pub struct GuestRam {
    base: *mut u8,
    len: usize,
}

// Safety: a private anonymous mapping owned by this struct for its whole life. `Sync` as
// well as `Send` because `SharedGuestRam` puts one behind an `Arc`; the aliasing that
// implies is the whole point of that type and is documented there.
unsafe impl Send for GuestRam {}
unsafe impl Sync for GuestRam {}

impl Drop for GuestRam {
    fn drop(&mut self) {
        // Safety: `base`/`len` are what `mmap` returned.
        unsafe {
            libc::munmap(self.base.cast(), self.len);
        }
    }
}

impl GuestRam {
    /// Reserve `len` bytes, rounded up to a page.
    ///
    /// # Panics
    /// If the mapping cannot be made.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let page = 4096;
        let len = len.div_ceil(page) * page;
        // Safety: a fresh anonymous mapping at no fixed address.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(
            base != libc::MAP_FAILED,
            "could not reserve {len} bytes of guest RAM: {}",
            std::io::Error::last_os_error()
        );
        Self {
            base: base.cast(),
            len,
        }
    }

    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: the mapping is `len` readable, writable bytes owned by this struct.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Guest RAM the machine and a vCPU both address, without a copy.
///
/// [`GuestRam`] owns its mapping; this shares one. That is what closes the last gap
/// between "KVM runs the guest's code" and "KVM runs the guest": with two allocations the
/// vCPU writes a descriptor into its own copy, a device model reads the machine's, and
/// nothing the guest asks for ever comes back right. The symptom is a guest that executes
/// happily and never progresses.
///
/// Hand one clone to [`Machine::new_with_guest_memory`](crate::Machine::new_with_guest_memory)
/// and the pointer from [`Self::base`] to [`KvmVm::new_from_raw`], and both are looking at
/// the same pages.
#[derive(Clone)]
pub struct SharedGuestRam {
    map: std::sync::Arc<GuestRam>,
}

// Safety: the mapping is owned for the life of the `Arc`, and the machine drives one vCPU
// from one thread — the same discipline the rest of the machine's memory already assumes.
unsafe impl Send for SharedGuestRam {}
unsafe impl Sync for SharedGuestRam {}

impl SharedGuestRam {
    /// Reserve `len` bytes, rounded up to a page.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            map: std::sync::Arc::new(GuestRam::new(len)),
        }
    }

    /// The mapping's base, for [`KvmVm::new_from_raw`].
    #[must_use]
    pub fn base(&self) -> *mut u8 {
        self.map.base
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.len == 0
    }

    #[inline]
    fn slice(&self) -> &[u8] {
        // Safety: the mapping is `len` readable bytes owned by the `Arc`.
        unsafe { std::slice::from_raw_parts(self.map.base, self.map.len) }
    }

    /// # Safety
    /// The mapping is deliberately aliased — that is what this type is for — so the caller
    /// must not hold two mutable views at once. In practice the machine drives one vCPU
    /// from one thread, which is the same discipline the rest of its memory assumes.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self) -> &mut [u8] {
        // Safety: as `slice`, plus the caller's obligation above.
        unsafe { std::slice::from_raw_parts_mut(self.map.base, self.map.len) }
    }
}

impl memory::GuestMemory for SharedGuestRam {
    fn size(&self) -> u64 {
        self.map.len as u64
    }

    fn read_into(&self, paddr: u64, dst: &mut [u8]) -> memory::phys::GuestMemoryResult<()> {
        let end = (paddr as usize)
            .checked_add(dst.len())
            .filter(|e| *e <= self.map.len)
            .ok_or(memory::phys::GuestMemoryError::OutOfRange {
                paddr,
                len: dst.len(),
                size: self.map.len as u64,
            })?;
        dst.copy_from_slice(&self.slice()[paddr as usize..end]);
        Ok(())
    }

    fn write_from(&mut self, paddr: u64, src: &[u8]) -> memory::phys::GuestMemoryResult<()> {
        let end = (paddr as usize)
            .checked_add(src.len())
            .filter(|e| *e <= self.map.len)
            .ok_or(memory::phys::GuestMemoryError::OutOfRange {
                paddr,
                len: src.len(),
                size: self.map.len as u64,
            })?;
        // Safety: `&mut self`, so this is the only view being written through.
        let dst = unsafe { self.slice_mut() };
        dst[paddr as usize..end].copy_from_slice(src);
        Ok(())
    }

    fn get_slice(&self, paddr: u64, len: usize) -> Option<&[u8]> {
        let at = paddr as usize;
        let end = at.checked_add(len)?;
        (end <= self.map.len).then(|| &self.slice()[at..end])
    }

    fn get_slice_mut(&mut self, paddr: u64, len: usize) -> Option<&mut [u8]> {
        let at = paddr as usize;
        let end = at.checked_add(len)?;
        if end > self.map.len {
            return None;
        }
        // Safety: `&mut self`, so this is the only view being handed out.
        let all = unsafe { self.slice_mut() };
        Some(&mut all[at..end])
    }
}

/// A VM with one vCPU and one memory region.
pub struct KvmVm {
    _kvm: OwnedFd,
    vm: OwnedFd,
    vcpu: OwnedFd,
    run: *mut u8,
    run_size: usize,
}

// Safety: the `kvm_run` mapping belongs to this VM's vCPU and is only touched through
// `&mut self`. The machine drives one vCPU from one thread.
unsafe impl Send for KvmVm {}

impl Drop for KvmVm {
    fn drop(&mut self) {
        if !self.run.is_null() {
            // Safety: `run`/`run_size` are what `mmap` returned for this vCPU.
            unsafe {
                libc::munmap(self.run.cast(), self.run_size);
            }
        }
    }
}

/// Why a VM could not be created. Absence of `/dev/kvm` is expected, not exceptional:
/// the caller falls back to the compiled tier.
#[derive(Debug)]
pub enum KvmError {
    /// No `/dev/kvm`, or no permission to open it.
    Unavailable(std::io::Error),
    /// The kernel's KVM API is not the one this code was written against.
    ApiVersion(i32),
    /// An ioctl failed; the name says which.
    Ioctl(&'static str, std::io::Error),
}

impl std::fmt::Display for KvmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(e) => write!(f, "/dev/kvm is not available: {e}"),
            Self::ApiVersion(v) => write!(f, "unexpected KVM API version {v}, expected 12"),
            Self::Ioctl(name, e) => write!(f, "{name} failed: {e}"),
        }
    }
}

fn ioctl(fd: RawFd, req: u64, arg: u64, name: &'static str) -> Result<i32, KvmError> {
    // Safety: every call site passes a request this file took from `linux/kvm.h` together
    // with the argument that request's encoding demands.
    let rc = unsafe { libc::ioctl(fd, req as libc::c_ulong, arg) };
    if rc < 0 {
        return Err(KvmError::Ioctl(name, std::io::Error::last_os_error()));
    }
    Ok(rc)
}

impl KvmVm {
    /// Create a VM whose guest physical memory is `ram`, mapped at `guest_phys`.
    ///
    /// `ram` must stay alive and unmoved for the VM's lifetime — KVM holds the host
    /// addresses, not a copy.
    ///
    /// # Safety
    /// The caller guarantees `ram` outlives the returned VM and is not reallocated. It
    /// remains readable and writable by the guest for that whole time, so nothing else may
    /// assume exclusive access to it.
    pub unsafe fn new(ram: &mut [u8], guest_phys: u64) -> Result<Self, KvmError> {
        // Safety: forwarded to the caller's obligation.
        unsafe { Self::new_inner(ram, guest_phys, false) }
    }

    /// As [`Self::new`], with KVM's in-kernel interrupt controllers.
    ///
    /// `KVM_CREATE_IRQCHIP` gives the VM a PIC, an IOAPIC and a per-vCPU local APIC that
    /// the kernel emulates. That is how a production VMM runs, and it is what a guest
    /// expects to find: without it, a booted Windows kernel exits with
    /// `KVM_EXIT_SET_TPR` on its first task-priority write and cannot get further,
    /// because it is asking a userspace APIC that is not there.
    ///
    /// It has to be created **before the vCPU**, which is why it is a constructor rather
    /// than a method: the local APIC is part of the vCPU, so the order is not negotiable.
    ///
    /// The trade is that Aero's own PIC and IOAPIC models are then bypassed for interrupt
    /// *routing* — the kernel decides vectors. Wiring Aero's models back in means the
    /// split-irqchip arrangement instead, which this does not do yet.
    ///
    /// # Safety
    /// As [`Self::new`].
    pub unsafe fn new_with_irqchip(ram: &mut [u8], guest_phys: u64) -> Result<Self, KvmError> {
        // Safety: forwarded to the caller's obligation.
        unsafe { Self::new_inner(ram, guest_phys, true) }
    }

    unsafe fn new_inner(
        ram: &mut [u8],
        guest_phys: u64,
        irqchip: bool,
    ) -> Result<Self, KvmError> {
        let kvm = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm")
            .map_err(KvmError::Unavailable)?;
        // Safety: `into` gives up ownership of a valid descriptor.
        let kvm: OwnedFd = kvm.into();

        let version = ioctl(kvm.as_raw_fd(), KVM_GET_API_VERSION, 0, "KVM_GET_API_VERSION")?;
        if version != 12 {
            return Err(KvmError::ApiVersion(version));
        }

        let vmfd = ioctl(kvm.as_raw_fd(), KVM_CREATE_VM, 0, "KVM_CREATE_VM")?;
        // Safety: a fresh descriptor from `KVM_CREATE_VM`, owned by nothing else.
        let vm = unsafe { OwnedFd::from_raw_fd(vmfd) };

        let region = KvmUserspaceMemoryRegion {
            slot: 0,
            flags: 0,
            guest_phys_addr: guest_phys,
            memory_size: ram.len() as u64,
            userspace_addr: ram.as_mut_ptr() as u64,
        };
        ioctl(
            vm.as_raw_fd(),
            KVM_SET_USER_MEMORY_REGION,
            std::ptr::addr_of!(region) as u64,
            "KVM_SET_USER_MEMORY_REGION",
        )?;

        // Before the vCPU: the local APIC belongs to it.
        if irqchip {
            ioctl(vm.as_raw_fd(), KVM_CREATE_IRQCHIP, 0, "KVM_CREATE_IRQCHIP")?;
        }

        let vcpufd = ioctl(vm.as_raw_fd(), KVM_CREATE_VCPU, 0, "KVM_CREATE_VCPU")?;
        // Safety: as for `vm`.
        let vcpu = unsafe { OwnedFd::from_raw_fd(vcpufd) };

        let run_size = ioctl(
            kvm.as_raw_fd(),
            KVM_GET_VCPU_MMAP_SIZE,
            0,
            "KVM_GET_VCPU_MMAP_SIZE",
        )? as usize;

        // Safety: `run_size` is what KVM said to map, for this vCPU's descriptor.
        let run = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                run_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                vcpu.as_raw_fd(),
                0,
            )
        };
        if run == libc::MAP_FAILED {
            return Err(KvmError::Ioctl(
                "mmap(kvm_run)",
                std::io::Error::last_os_error(),
            ));
        }

        Ok(Self {
            _kvm: kvm,
            vm,
            vcpu,
            run: run.cast(),
            run_size,
        })
    }

    /// Create a VM over guest RAM the caller already owns.
    ///
    /// [`Self::new`] takes a slice, which is what a test wants. The machine's RAM is not a
    /// slice it can lend out: under `AERO_JIT_RAM=1` it lives inside the JIT backend's
    /// linear memory, reached through a raw pointer (`WasmGuestRam`) precisely so the
    /// interpreter and compiled code can both touch it. That memory is an `mmap`, so it is
    /// already page-aligned and contiguous — the two things
    /// `KVM_SET_USER_MEMORY_REGION` insists on — and it can be handed to KVM directly
    /// rather than copied.
    ///
    /// Mapping it means the guest's RAM is the *same pages* the rest of the machine reads
    /// and writes. That is the point: a device doing DMA, a snapshot being restored, and
    /// the vCPU all see one memory.
    ///
    /// # Safety
    /// `base` must be page-aligned, `len` a multiple of the page size, and the region must
    /// stay mapped and unmoved for the VM's lifetime. The guest may read and write all of
    /// it at any time, so nothing else may assume exclusive access.
    pub unsafe fn new_from_raw(
        base: *mut u8,
        len: usize,
        guest_phys: u64,
    ) -> Result<Self, KvmError> {
        // Safety: forwarded to the caller's obligation.
        unsafe { Self::new_from_raw_maybe_irqchip(base, len, guest_phys, false) }
    }

    /// As [`Self::new_from_raw`], with KVM's in-kernel interrupt controllers.
    ///
    /// # Safety
    /// As [`Self::new_from_raw`].
    pub unsafe fn new_from_raw_with_irqchip(
        base: *mut u8,
        len: usize,
        guest_phys: u64,
    ) -> Result<Self, KvmError> {
        // Safety: forwarded to the caller's obligation.
        unsafe { Self::new_from_raw_maybe_irqchip(base, len, guest_phys, true) }
    }

    unsafe fn new_from_raw_maybe_irqchip(
        base: *mut u8,
        len: usize,
        guest_phys: u64,
        irqchip: bool,
    ) -> Result<Self, KvmError> {
        assert!(
            (base as usize).is_multiple_of(4096),
            "guest RAM must be page-aligned for KVM; got {base:p}"
        );
        assert!(
            len.is_multiple_of(4096),
            "guest RAM length must be a multiple of the page size; got {len}"
        );
        // Safety: the caller's obligation, forwarded — construction only needs the slice to
        // build the memory region, and never retains it.
        let slice = unsafe { std::slice::from_raw_parts_mut(base, len) };
        // Safety: as above.
        unsafe { Self::new_inner(slice, guest_phys, irqchip) }
    }

    pub fn regs(&self) -> Result<KvmRegs, KvmError> {
        let mut regs = KvmRegs::default();
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_GET_REGS,
            std::ptr::addr_of_mut!(regs) as u64,
            "KVM_GET_REGS",
        )?;
        Ok(regs)
    }

    pub fn set_regs(&mut self, regs: &KvmRegs) -> Result<(), KvmError> {
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_SET_REGS,
            std::ptr::addr_of!(*regs) as u64,
            "KVM_SET_REGS",
        )?;
        Ok(())
    }

    pub fn sregs(&self) -> Result<KvmSregs, KvmError> {
        let mut sregs = KvmSregs::default();
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_GET_SREGS,
            std::ptr::addr_of_mut!(sregs) as u64,
            "KVM_GET_SREGS",
        )?;
        Ok(sregs)
    }

    pub fn set_sregs(&mut self, sregs: &KvmSregs) -> Result<(), KvmError> {
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_SET_SREGS,
            std::ptr::addr_of!(*sregs) as u64,
            "KVM_SET_SREGS",
        )?;
        Ok(())
    }

    /// Run until the vCPU exits.
    pub fn run(&mut self) -> Result<KvmExit, KvmError> {
        // `EINTR` is not a failure: it is the only way to take a guest off the CPU that is
        // not asking for anything. A guest spinning on a timer that never ticks — which is
        // exactly what a booted Windows does with no devices behind its exits — never
        // leaves `KVM_RUN` otherwise, and no budget expressed in exits can bound it.
        // The caller arranges the signal; this only has to not treat it as an error.
        match ioctl(self.vcpu.as_raw_fd(), KVM_RUN, 0, "KVM_RUN") {
            Ok(_) => {}
            Err(KvmError::Ioctl(_, e)) if e.raw_os_error() == Some(libc::EINTR) => {
                return Ok(KvmExit::Interrupted);
            }
            Err(e) => return Err(e),
        }
        // Safety: `run` maps at least `run_size` bytes, and these offsets come from the
        // header (`exit_reason` at 8, the exit union at 32).
        let reason = unsafe { std::ptr::read_volatile(self.run.add(8).cast::<u32>()) };
        Ok(match reason {
            2 => {
                // Safety: as above; `struct kvm_run.io` is direction/size/port/count/offset.
                let io = unsafe { self.run.add(32) };
                unsafe {
                    KvmExit::Io {
                        write: std::ptr::read_volatile(io) == 1,
                        size: std::ptr::read_volatile(io.add(1)),
                        port: std::ptr::read_volatile(io.add(2).cast::<u16>()),
                        count: std::ptr::read_volatile(io.add(4).cast::<u32>()),
                        data_offset: std::ptr::read_volatile(io.add(8).cast::<u64>()),
                    }
                }
            }
            5 => KvmExit::Halt,
            7 => KvmExit::IrqWindowOpen,
            11 => KvmExit::SetTpr,
            6 => {
                // Safety: as above; `struct kvm_run.mmio` is addr/data[8]/len/is_write.
                let m = unsafe { self.run.add(32) };
                unsafe {
                    KvmExit::Mmio {
                        addr: std::ptr::read_volatile(m.cast::<u64>()),
                        len: std::ptr::read_volatile(m.add(16).cast::<u32>()),
                        write: std::ptr::read_volatile(m.add(20)) == 1,
                    }
                }
            }
            10 => KvmExit::Interrupted,
            8 => KvmExit::Shutdown,
            9 => KvmExit::FailedEntry,
            other => KvmExit::Other(other),
        })
    }

    /// The `kvm_run` data area, for reading or writing an I/O or MMIO payload.
    ///
    /// # Safety
    /// `offset .. offset + len` must be inside the `kvm_run` mapping, which is what the
    /// exit's own `data_offset`/`len` describe.
    pub unsafe fn run_data(&mut self, offset: u64, len: usize) -> &mut [u8] {
        // Safety: forwarded to the caller's obligation.
        unsafe { std::slice::from_raw_parts_mut(self.run.add(offset as usize), len) }
    }


    /// Load a `CpuState` into the vCPU.
    ///
    /// This is the piece everything else depends on: until the booted guest's
    /// architectural state can be handed to hardware, KVM can only run code written for
    /// it. Three things here are easy to get wrong and silent when wrong:
    ///
    /// * **The register orders differ.** `CpuState::gpr` is in architectural encoding
    ///   order (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8..R15); `struct kvm_regs` is
    ///   RAX, RBX, RCX, RDX, RSI, RDI, RSP, RBP, R8... Copying straight across swaps
    ///   RCX with RBX and RSI/RDI with RSP/RBP.
    /// * **Segments carry hidden state.** A selector alone is not enough: the base, limit
    ///   and access rights are what the CPU actually uses, and after a `mov ds, ax` they
    ///   need not agree with the descriptor table any more. `CpuState::Segment` already
    ///   keeps them, and its `access` field is deliberately the VMX AR-byte layout, which
    ///   is what KVM wants field by field.
    /// * **`unusable` is not "absent".** A null but loaded segment has to arrive as
    ///   unusable, or the guest faults on the first access through it.
    ///
    /// **The state has to be complete, and `CpuState::new` is not.** This writes what it
    /// is given, faithfully — the round trip is checked in `tests/kvm_cpu_state.rs` — and a
    /// vCPU does not tolerate a half-filled state the way an interpreter does. Two groups
    /// are zero by default and both are load-bearing:
    ///
    /// * **Segment descriptors of all zeros are *not present*.** The guest faults on its
    ///   first access through one.
    /// * **`IDTR.limit = 0` means no interrupt can ever be delivered.** The guest runs
    ///   perfectly and then triple-faults the instant a vector arrives, coming back at the
    ///   reset vector `0xffff0000:0xfff0` — because the vector's entry lies outside the
    ///   table. In real mode that table is the IVT, and it still goes through IDTR.
    ///
    /// The second cost an afternoon: it looks like a bug in this conversion, and it is not.
    /// Bisected by loading the vCPU's *own* state back through here (delivers), then each
    /// group of a hand-built one in turn — segments and control registers deliver, the
    /// descriptor tables do not. A restored snapshot carries real values throughout;
    /// anything constructing a state by hand has to fill them.
    pub fn load_cpu_state(&mut self, cpu: &aero_cpu_core::state::CpuState) -> Result<(), KvmError> {
        use aero_cpu_core::state::gpr;

        let g = &cpu.gpr;
        let regs = KvmRegs {
            rax: g[gpr::RAX],
            rbx: g[gpr::RBX],
            rcx: g[gpr::RCX],
            rdx: g[gpr::RDX],
            rsi: g[gpr::RSI],
            rdi: g[gpr::RDI],
            rsp: g[gpr::RSP],
            rbp: g[gpr::RBP],
            r8: g[8],
            r9: g[9],
            r10: g[10],
            r11: g[11],
            r12: g[12],
            r13: g[13],
            r14: g[14],
            r15: g[15],
            rip: cpu.rip,
            // Bit 1 is reserved and must be set, whatever the guest left there.
            rflags: cpu.rflags | 0x2,
        };
        self.set_regs(&regs)?;

        let mut sregs = self.sregs()?;
        sregs.cs = seg_to_kvm(&cpu.segments.cs);
        sregs.ds = seg_to_kvm(&cpu.segments.ds);
        sregs.es = seg_to_kvm(&cpu.segments.es);
        sregs.fs = seg_to_kvm(&cpu.segments.fs);
        sregs.gs = seg_to_kvm(&cpu.segments.gs);
        sregs.ss = seg_to_kvm(&cpu.segments.ss);
        sregs.tr = seg_to_kvm(&cpu.tables.tr);
        sregs.ldt = seg_to_kvm(&cpu.tables.ldtr);
        sregs.gdt = KvmDtable {
            base: cpu.tables.gdtr.base,
            limit: cpu.tables.gdtr.limit,
            padding: [0; 3],
        };
        sregs.idt = KvmDtable {
            base: cpu.tables.idtr.base,
            limit: cpu.tables.idtr.limit,
            padding: [0; 3],
        };
        sregs.cr0 = cpu.control.cr0;
        sregs.cr2 = cpu.control.cr2;
        sregs.cr3 = cpu.control.cr3;
        sregs.cr4 = cpu.control.cr4;
        sregs.cr8 = cpu.control.cr8;
        sregs.efer = cpu.msr.efer;
        self.set_sregs(&sregs)?;
        self.set_msrs(&cpu.msr)?;
        Ok(())
    }

    /// Read the vCPU back into a `CpuState`.
    ///
    /// The inverse of [`Self::load_cpu_state`], and the reason the pair is testable: a
    /// state that survives a round trip is one the conversion has not quietly dropped.
    pub fn store_cpu_state(
        &self,
        cpu: &mut aero_cpu_core::state::CpuState,
    ) -> Result<(), KvmError> {
        use aero_cpu_core::state::gpr;

        let r = self.regs()?;
        cpu.gpr[gpr::RAX] = r.rax;
        cpu.gpr[gpr::RBX] = r.rbx;
        cpu.gpr[gpr::RCX] = r.rcx;
        cpu.gpr[gpr::RDX] = r.rdx;
        cpu.gpr[gpr::RSI] = r.rsi;
        cpu.gpr[gpr::RDI] = r.rdi;
        cpu.gpr[gpr::RSP] = r.rsp;
        cpu.gpr[gpr::RBP] = r.rbp;
        cpu.gpr[8] = r.r8;
        cpu.gpr[9] = r.r9;
        cpu.gpr[10] = r.r10;
        cpu.gpr[11] = r.r11;
        cpu.gpr[12] = r.r12;
        cpu.gpr[13] = r.r13;
        cpu.gpr[14] = r.r14;
        cpu.gpr[15] = r.r15;
        cpu.rip = r.rip;
        cpu.set_rflags(r.rflags);

        let s = self.sregs()?;
        cpu.segments.cs = kvm_to_seg(&s.cs);
        cpu.segments.ds = kvm_to_seg(&s.ds);
        cpu.segments.es = kvm_to_seg(&s.es);
        cpu.segments.fs = kvm_to_seg(&s.fs);
        cpu.segments.gs = kvm_to_seg(&s.gs);
        cpu.segments.ss = kvm_to_seg(&s.ss);
        cpu.tables.tr = kvm_to_seg(&s.tr);
        cpu.tables.ldtr = kvm_to_seg(&s.ldt);
        cpu.tables.gdtr.base = s.gdt.base;
        cpu.tables.gdtr.limit = s.gdt.limit;
        cpu.tables.idtr.base = s.idt.base;
        cpu.tables.idtr.limit = s.idt.limit;
        cpu.control.cr0 = s.cr0;
        cpu.control.cr2 = s.cr2;
        cpu.control.cr3 = s.cr3;
        cpu.control.cr4 = s.cr4;
        cpu.control.cr8 = s.cr8;
        cpu.msr.efer = s.efer;
        self.get_msrs(&mut cpu.msr)?;
        // `EFER` travels in `kvm_sregs`; `get_msrs` must not be allowed to overwrite it
        // with whatever the MSR path reports.
        cpu.msr.efer = s.efer;
        Ok(())
    }

    /// Write the MSRs a booted guest depends on.
    ///
    /// `struct kvm_msrs` is a count followed by a flexible array of 16-byte entries, so
    /// the buffer is built by hand: 8 bytes of header (`nmsrs` then padding) and then
    /// `(index, data)` pairs.
    fn set_msrs(&mut self, msrs: &aero_cpu_core::state::MsrState) -> Result<(), KvmError> {
        let mut buf = vec![0u8; 8 + MSR_INDICES.len() * 16];
        buf[0..4].copy_from_slice(&(MSR_INDICES.len() as u32).to_le_bytes());
        for (i, (index, field)) in MSR_INDICES.iter().enumerate() {
            let at = 8 + i * 16;
            buf[at..at + 4].copy_from_slice(&index.to_le_bytes());
            buf[at + 8..at + 16].copy_from_slice(&field.get(msrs).to_le_bytes());
        }
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_SET_MSRS,
            buf.as_mut_ptr() as u64,
            "KVM_SET_MSRS",
        )?;
        Ok(())
    }

    /// Read those MSRs back.
    fn get_msrs(&self, msrs: &mut aero_cpu_core::state::MsrState) -> Result<(), KvmError> {
        let mut buf = vec![0u8; 8 + MSR_INDICES.len() * 16];
        buf[0..4].copy_from_slice(&(MSR_INDICES.len() as u32).to_le_bytes());
        for (i, (index, _)) in MSR_INDICES.iter().enumerate() {
            let at = 8 + i * 16;
            buf[at..at + 4].copy_from_slice(&index.to_le_bytes());
        }
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_GET_MSRS,
            buf.as_mut_ptr() as u64,
            "KVM_GET_MSRS",
        )?;
        for (i, (_, field)) in MSR_INDICES.iter().enumerate() {
            let at = 8 + i * 16;
            let mut v = [0u8; 8];
            v.copy_from_slice(&buf[at + 8..at + 16]);
            field.set(msrs, u64::from_le_bytes(v));
        }
        Ok(())
    }

    /// The vCPU's multiprocessing state: 0 runnable, 1 uninitialised, 2 INIT received,
    /// 3 halted, 4 SIPI received.
    ///
    /// Worth having when a guest appears to stop: "runnable but making no progress" and
    /// "halted" and "waiting for a startup IPI" look identical from outside and mean
    /// entirely different things.
    pub fn mp_state(&self) -> Result<u32, KvmError> {
        let mut state: u32 = 0;
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_GET_MP_STATE,
            std::ptr::addr_of_mut!(state) as u64,
            "KVM_GET_MP_STATE",
        )?;
        Ok(state)
    }

    /// Raise or lower an ISA interrupt line into the in-kernel controllers.
    ///
    /// The counterpart to [`Self::inject_interrupt`] for a VM built with
    /// [`Self::new_with_irqchip`]: with the PIC and IOAPIC in the kernel it is a *line*
    /// that gets asserted, not a vector — the chips decide the vector, exactly as the
    /// hardware does, and exactly as Aero's own models do for the other tiers.
    ///
    /// This is how a guest gets its clock. Windows programs a periodic timer and then
    /// waits for it; with nothing raising IRQ 0 the guest runs whatever it can and then
    /// sits in a spinlock forever, taking no halts and no exits, which reads as a stall
    /// with no obvious cause.
    ///
    /// Requires the in-kernel irqchip; without it KVM has nothing to route.
    pub fn set_irq_line(&mut self, irq: u32, level: bool) -> Result<(), KvmError> {
        #[repr(C)]
        struct IrqLevel {
            irq: u32,
            level: u32,
        }
        let arg = IrqLevel {
            irq,
            level: u32::from(level),
        };
        ioctl(
            self.vm.as_raw_fd(),
            KVM_IRQ_LINE,
            std::ptr::addr_of!(arg) as u64,
            "KVM_IRQ_LINE",
        )?;
        Ok(())
    }

    /// Pulse a line: assert then deassert, which is what an edge-triggered source does.
    pub fn pulse_irq_line(&mut self, irq: u32) -> Result<(), KvmError> {
        self.set_irq_line(irq, true)?;
        self.set_irq_line(irq, false)
    }

    /// Deliver an external interrupt vector on the next entry.
    ///
    /// The PIC and IOAPIC still decide *which* vector; this is only the handoff. KVM
    /// refuses if the guest cannot accept one yet, which is not an error — the caller
    /// tries again after the next exit.
    pub fn inject_interrupt(&mut self, vector: u8) -> Result<bool, KvmError> {
        let v = u32::from(vector);
        match ioctl(
            self.vcpu.as_raw_fd(),
            KVM_INTERRUPT,
            std::ptr::addr_of!(v) as u64,
            "KVM_INTERRUPT",
        ) {
            Ok(_) => Ok(true),
            // EEXIST: one is already pending. Not a failure, just not now.
            Err(KvmError::Ioctl(_, e)) if e.raw_os_error() == Some(libc::EEXIST) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Take the guest off the CPU periodically, so one that asks for nothing can still be
    /// bounded.
    ///
    /// A guest spinning on a timer that never ticks — which is what a booted Windows does
    /// with nothing behind its exits — never leaves `KVM_RUN`, and no budget counted in
    /// exits can reach it. The remedy is a signal: with no `SA_RESTART` the in-flight
    /// ioctl fails with `EINTR`, which [`Self::run`] reports as [`KvmExit::Interrupted`].
    ///
    /// **It has to be aimed at the thread inside `KVM_RUN`.** `setitimer` raises the signal
    /// process-wide and any thread may take it; under a test harness that is reliably the
    /// wrong one, and the guest runs on undisturbed. So this records the calling thread and
    /// has a helper `pthread_kill` that thread specifically.
    ///
    /// Returns a guard: dropping it stops the helper.
    ///
    /// # Panics
    /// If the signal handler cannot be installed.
    #[must_use]
    pub fn preempt_this_thread(interval: std::time::Duration) -> Preemption {
        extern "C" fn on_alarm(_sig: libc::c_int) {}

        // Safety: a valid, empty `sigaction` for SIGALRM, deliberately without
        // `SA_RESTART` so an interrupted `ioctl` returns rather than resuming.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_alarm as usize;
            sa.sa_flags = 0;
            libc::sigemptyset(std::ptr::addr_of_mut!(sa.sa_mask));
            assert!(
                libc::sigaction(libc::SIGALRM, &sa, std::ptr::null_mut()) == 0,
                "could not install the preemption handler: {}",
                std::io::Error::last_os_error()
            );
        }

        // Safety: reading this thread's own handle.
        let target = unsafe { libc::pthread_self() } as usize;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::spawn(move || {
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(interval);
                // Safety: `target` is a live `pthread_t` — the guard joins this thread
                // before the target can go away.
                unsafe {
                    libc::pthread_kill(target as libc::pthread_t, libc::SIGALRM);
                }
            }
        });

        Preemption {
            stop,
            handle: Some(handle),
        }
    }

    /// Ask KVM to exit as soon as the guest could take an external interrupt.
    ///
    /// Aero decides *which* vector — its PIC and IOAPIC already do that for the
    /// interpreter and the compiled tier — so what the backend needs from KVM is only the
    /// timing: an interrupt cannot be injected while the guest has them masked or is
    /// mid-instruction. Setting this makes KVM return [`KvmExit::IrqWindowOpen`] at the
    /// first moment it *can* be injected, which turns "poll until it works" into one exit.
    pub fn request_interrupt_window(&mut self, want: bool) {
        // Safety: `request_interrupt_window` is the first byte of `kvm_run`, and no block
        // is running while `&mut self` is held.
        unsafe {
            std::ptr::write_volatile(self.run, u8::from(want));
        }
    }

    /// Whether the guest could accept an external interrupt at the last exit.
    #[must_use]
    pub fn ready_for_interrupt(&self) -> bool {
        // Safety: `ready_for_interrupt_injection` is at offset 12 of `kvm_run`.
        unsafe { std::ptr::read_volatile(self.run.add(12)) != 0 }
    }

    /// The guest's `IF` as of the last exit.
    ///
    /// Worth having separately: a guest sitting in `cli; hlt` is *halted* but cannot take
    /// an interrupt, and waking it would be wrong.
    #[must_use]
    pub fn interrupts_enabled(&self) -> bool {
        // Safety: `if_flag` is at offset 13 of `kvm_run`.
        unsafe { std::ptr::read_volatile(self.run.add(13)) != 0 }
    }

    /// The vCPU's local APIC state, as `KVM_GET_LAPIC` returns it.
    ///
    /// 1KiB of register file. The machine's own LAPIC model owns the architectural state;
    /// this is the copy the vCPU runs against, and restoring a snapshot means writing it.
    pub fn lapic(&self) -> Result<Box<[u8; 1024]>, KvmError> {
        let mut buf = Box::new([0u8; 1024]);
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_GET_LAPIC,
            buf.as_mut_ptr() as u64,
            "KVM_GET_LAPIC",
        )?;
        Ok(buf)
    }

    /// Write the vCPU's local APIC state.
    pub fn set_lapic(&mut self, state: &[u8; 1024]) -> Result<(), KvmError> {
        ioctl(
            self.vcpu.as_raw_fd(),
            KVM_SET_LAPIC,
            state.as_ptr() as u64,
            "KVM_SET_LAPIC",
        )?;
        Ok(())
    }

    /// One of the VM's in-kernel interrupt chips, as `KVM_GET_IRQCHIP` returns it.
    ///
    /// `chip_id` is 0 for the master PIC, 1 for the slave, 2 for the IOAPIC. The buffer is
    /// the `chip` union of `struct kvm_irqchip`: 512 bytes, of which the IOAPIC uses 216
    /// (`base_address`, `ioregsel`, `id`, `irr`, padding, then 24 64-bit redirection
    /// entries at offset 24) and each PIC 16.
    ///
    /// This exists to be *written*. A VM created with in-kernel chips starts them at their
    /// power-on defaults, every redirection entry masked — which is right for a VM that
    /// boots, and wrong for one restored into the middle of a running OS that programmed
    /// its controllers long ago. Raising a line on a chip in that state does nothing at
    /// all, silently, and the guest waits forever for a timer tick.
    pub fn irqchip(&self, chip_id: u32) -> Result<Box<[u8; 512]>, KvmError> {
        let mut buf = [0u8; 520];
        buf[0..4].copy_from_slice(&chip_id.to_le_bytes());
        ioctl(
            self.vm.as_raw_fd(),
            KVM_GET_IRQCHIP,
            buf.as_mut_ptr() as u64,
            "KVM_GET_IRQCHIP",
        )?;
        let mut chip = Box::new([0u8; 512]);
        chip.copy_from_slice(&buf[8..520]);
        Ok(chip)
    }

    /// Write one of the VM's in-kernel interrupt chips. See [`Self::irqchip`].
    pub fn set_irqchip(&mut self, chip_id: u32, chip: &[u8; 512]) -> Result<(), KvmError> {
        let mut buf = [0u8; 520];
        buf[0..4].copy_from_slice(&chip_id.to_le_bytes());
        buf[8..520].copy_from_slice(chip);
        ioctl(
            self.vm.as_raw_fd(),
            KVM_SET_IRQCHIP,
            buf.as_ptr() as u64,
            "KVM_SET_IRQCHIP",
        )?;
        Ok(())
    }

    /// Run until the guest halts, an unhandled exit occurs, or `max_exits` is reached,
    /// routing port I/O and MMIO to the caller.
    ///
    /// This is the shape a `Machine` execution mode needs: the device models already
    /// answer reads and writes by address and port, so the backend's job is to turn a
    /// vCPU exit into one of those calls and put the answer where the guest will find it.
    ///
    /// * **Port I/O** carries its payload in the `kvm_run` mapping at `data_offset`, and
    ///   `count` may be more than one for a string instruction — each iteration is a
    ///   separate access of `size` bytes, which is why this loops rather than passing the
    ///   whole buffer along.
    /// * **MMIO** carries up to eight bytes inline in `kvm_run.mmio.data`.
    ///
    /// A read must write its answer back into that buffer *before the next entry*, which
    /// is the one ordering mistake that produces a guest reading stale data rather than an
    /// error.
    pub fn run_routed(
        &mut self,
        max_exits: u64,
        pio: &mut dyn FnMut(u16, bool, &mut [u8]),
        mmio: &mut dyn FnMut(u64, bool, &mut [u8]),
    ) -> Result<KvmExit, KvmError> {
        for _ in 0..max_exits {
            match self.run()? {
                KvmExit::Io {
                    port,
                    write,
                    size,
                    count,
                    data_offset,
                } => {
                    let size = usize::from(size);
                    for i in 0..count as usize {
                        let at = data_offset + (i * size) as u64;
                        // Safety: KVM guarantees `data_offset .. + size * count` is inside
                        // the `kvm_run` mapping for this exit.
                        let slot = unsafe { self.run_data(at, size) };
                        pio(port, write, slot);
                    }
                }
                KvmExit::Mmio { addr, len, write } => {
                    let len = len as usize;
                    // Safety: `kvm_run.mmio.data` is 8 bytes at offset 32 + 8, and KVM
                    // will not report a `len` beyond it.
                    let slot = unsafe { self.run_data(32 + 8, len.min(8)) };
                    mmio(addr, write, slot);
                }
                other => return Ok(other),
            }
        }
        Ok(KvmExit::Other(u32::MAX))
    }

    /// Put the vCPU in 16-bit real mode at `cs:ip`, which is where a fresh VM starts
    /// anyway — but stated explicitly so a test does not depend on the default.
    pub fn set_real_mode(&mut self, cs_base: u64, ip: u64) -> Result<(), KvmError> {
        let mut sregs = self.sregs()?;
        sregs.cs.base = cs_base;
        sregs.cs.selector = (cs_base >> 4) as u16;
        self.set_sregs(&sregs)?;
        let mut regs = self.regs()?;
        regs.rip = ip;
        // Bit 1 is reserved and must be set; nothing else is needed to start.
        regs.rflags = 0x2;
        self.set_regs(&regs)?;
        Ok(())
    }
}

/// `CpuState`'s segment to KVM's.
///
/// `Segment::access` is already the VMX AR-byte layout — type in 0..4, S at 4, DPL at
/// 5..7, P at 7, AVL at 8, L at 9, D/B at 10, G at 11 — which is exactly the set of
/// fields `struct kvm_segment` splits out, so this is a decode rather than a translation.
fn seg_to_kvm(seg: &aero_cpu_core::state::Segment) -> KvmSegment {
    use aero_cpu_core::state::SEG_ACCESS_UNUSABLE;
    KvmSegment {
        base: seg.base,
        limit: seg.limit,
        selector: seg.selector,
        type_: (seg.access & 0xF) as u8,
        s: u8::from(seg.access & (1 << 4) != 0),
        dpl: ((seg.access >> 5) & 0b11) as u8,
        present: u8::from(seg.access & (1 << 7) != 0),
        avl: u8::from(seg.access & (1 << 8) != 0),
        l: u8::from(seg.access & (1 << 9) != 0),
        db: u8::from(seg.access & (1 << 10) != 0),
        g: u8::from(seg.access & (1 << 11) != 0),
        unusable: u8::from(seg.access & SEG_ACCESS_UNUSABLE != 0),
        padding: 0,
    }
}

/// The inverse of [`seg_to_kvm`].
fn kvm_to_seg(seg: &KvmSegment) -> aero_cpu_core::state::Segment {
    use aero_cpu_core::state::SEG_ACCESS_UNUSABLE;
    let mut access = u32::from(seg.type_) & 0xF;
    access |= u32::from(seg.s) << 4;
    access |= (u32::from(seg.dpl) & 0b11) << 5;
    access |= u32::from(seg.present) << 7;
    access |= u32::from(seg.avl) << 8;
    access |= u32::from(seg.l) << 9;
    access |= u32::from(seg.db) << 10;
    access |= u32::from(seg.g) << 11;
    if seg.unusable != 0 {
        access |= SEG_ACCESS_UNUSABLE;
    }
    aero_cpu_core::state::Segment {
        selector: seg.selector,
        _pad0: 0,
        _pad1: 0,
        base: seg.base,
        limit: seg.limit,
        access,
    }
}

/// Why a slice of guest execution ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvmSliceExit {
    /// The guest halted. It will make no further progress until a vector is delivered.
    Halted,
    /// The slice's exit budget ran out with the guest still running.
    BudgetExhausted,
    /// The guest triple-faulted or asked to power down.
    Shutdown,
    /// An exit this backend does not model. The caller decides what to do.
    Unhandled(KvmExit),
}

/// A vCPU driven the way `Machine`'s outer loop drives the other tiers.
///
/// The pieces beneath this — state, memory, exits, interrupts — are each testable on their
/// own, and are. This is the composition: the loop that decides an interrupt, enters the
/// guest, services what comes back, and hands control out again with the CPU state
/// updated. It takes the machine's own `IoPortBus` and `MemoryBus` because those are what
/// the devices are already behind; nothing here knows what a device is.
///
/// What it is not is `Machine` itself. The machine's loop also polls storage, ticks
/// platform timers and services the interrupt controllers, and those stay where they are —
/// this replaces only the part that executes guest instructions.
pub struct KvmRunner {
    vm: KvmVm,
}

impl KvmRunner {
    /// Wrap a VM whose memory is already registered.
    #[must_use]
    pub fn new(vm: KvmVm) -> Self {
        Self { vm }
    }

    /// The VM underneath, for state the runner does not manage (the LAPIC, for one).
    pub fn vm_mut(&mut self) -> &mut KvmVm {
        &mut self.vm
    }

    /// Run guest instructions until the guest halts, the budget runs out, or something
    /// happens that this does not model.
    ///
    /// `pending_vector` is what the platform's interrupt controllers decided, if anything.
    /// It is offered rather than forced: the guest may have interrupts masked, in which
    /// case KVM says so and the vector stays with the caller to offer again.
    ///
    /// `cpu` is loaded before the first entry and written back before returning, so the
    /// caller sees the guest's state exactly as the other tiers leave it.
    pub fn run_slice(
        &mut self,
        cpu: &mut aero_cpu_core::state::CpuState,
        io: &mut aero_platform::io::IoPortBus,
        mem: &mut aero_platform::memory::MemoryBus,
        pending_vector: Option<u8>,
        max_exits: u64,
    ) -> Result<KvmSliceExit, KvmError> {
        self.run_slice_inner(cpu, io, mem, pending_vector, max_exits, true)
    }

    /// As [`Self::run_slice`], continuing from the state already in the vCPU.
    ///
    /// When KVM owns the CPU, the vCPU *is* the architectural state — writing it back in
    /// before every entry is not just wasted ioctls, it is wrong. A vCPU halted at a `HLT`
    /// carries activity state that `KVM_SET_REGS`/`KVM_SET_SREGS` do not describe, and
    /// re-loading a state captured from a *running* guest and entering again triple-faults
    /// it. Measured: the same guest woken by an injected vector runs its handler when
    /// resumed and shuts down when reloaded, with every architectural field this backend
    /// knows about identical either way.
    ///
    /// So the rule is the one the other tiers already follow: sync state when a *different*
    /// executor has touched it, and otherwise leave it where it lives. `run_slice` is for
    /// entering from a state Tier-0 or the compiled tier produced; this is for continuing.
    ///
    /// The caller still gets `cpu` written back, because the machine's outer loop reads it
    /// between slices.
    pub fn resume_slice(
        &mut self,
        cpu: &mut aero_cpu_core::state::CpuState,
        io: &mut aero_platform::io::IoPortBus,
        mem: &mut aero_platform::memory::MemoryBus,
        pending_vector: Option<u8>,
        max_exits: u64,
    ) -> Result<KvmSliceExit, KvmError> {
        self.run_slice_inner(cpu, io, mem, pending_vector, max_exits, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_slice_inner(
        &mut self,
        cpu: &mut aero_cpu_core::state::CpuState,
        io: &mut aero_platform::io::IoPortBus,
        mem: &mut aero_platform::memory::MemoryBus,
        pending_vector: Option<u8>,
        max_exits: u64,
        load: bool,
    ) -> Result<KvmSliceExit, KvmError> {
        if load {
            self.vm.load_cpu_state(cpu)?;
        }

        // The vector goes in through KVM's handshake, not by asking at an arbitrary
        // moment. `KVM_INTERRUPT` will accept one whenever nothing else is queued, but
        // accepting it is not the same as the guest being able to *take* it — inject while
        // an interrupt shadow is in force, or between the parts of an instruction, and the
        // delivery faults; a real-mode guest then triple-faults and comes back at
        // `0xffff0000:0xfff0`, which is what this looked like before it was fixed.
        //
        // So: ask for a window, and inject when KVM says the guest can take one. That is
        // one extra exit and it is the documented order.
        let mut vector = pending_vector;
        if vector.is_some() {
            self.vm.request_interrupt_window(true);
        }

        let mut budget = max_exits;
        let outcome = loop {
            let out = {
                let mut pio = |port: u16, write: bool, data: &mut [u8]| {
                    let size = data.len() as u8;
                    if write {
                        let mut v = [0u8; 4];
                        let n = data.len().min(4);
                        v[..n].copy_from_slice(&data[..n]);
                        io.write(port, size, u32::from_le_bytes(v));
                    } else {
                        let v = io.read(port, size).to_le_bytes();
                        let n = data.len().min(4);
                        data[..n].copy_from_slice(&v[..n]);
                    }
                };
                let mut mmio = |addr: u64, write: bool, data: &mut [u8]| {
                    if write {
                        mem.write_physical(addr, data);
                    } else {
                        mem.read_physical(addr, data);
                    }
                };
                self.vm.run_routed(budget, &mut pio, &mut mmio)?
            };

            // The window opened: hand the vector over and keep going, so a slice that was
            // asked to deliver an interrupt actually delivers it.
            if let (KvmExit::IrqWindowOpen, Some(v)) = (out, vector) {
                self.vm.inject_interrupt(v)?;
                self.vm.request_interrupt_window(false);
                vector = None;
                budget = budget.saturating_sub(1);
                if budget > 0 {
                    continue;
                }
            }
            break out;
        };

        self.vm.store_cpu_state(cpu)?;

        Ok(match outcome {
            KvmExit::Halt => {
                cpu.halted = true;
                KvmSliceExit::Halted
            }
            KvmExit::IrqWindowOpen => KvmSliceExit::BudgetExhausted,
            KvmExit::Shutdown => KvmSliceExit::Shutdown,
            KvmExit::Other(n) if n == u32::MAX => KvmSliceExit::BudgetExhausted,
            other => KvmSliceExit::Unhandled(other),
        })
    }
}

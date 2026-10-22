use crate::exception::Exception;
use crate::mem::CpuBus;
use aero_mmu::{AccessType, MemoryBus, Mmu, TranslateFault};
use core::fmt;

/// Port I/O backend for [`PagingBus`].
///
/// Tier-0 implements IN/OUT/INS*/OUTS* via [`CpuBus::io_read`] and
/// [`CpuBus::io_write`]. [`PagingBus`] is primarily responsible for translating
/// linear memory accesses, but it also implements [`CpuBus`] and therefore needs
/// a way to forward port I/O to an embedding-specific device model.
pub trait IoBus {
    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception>;
    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception>;
}

impl<T: IoBus + ?Sized> IoBus for &mut T {
    #[inline]
    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception> {
        <T as IoBus>::io_read(&mut **self, port, size)
    }

    #[inline]
    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception> {
        <T as IoBus>::io_write(&mut **self, port, size, val)
    }
}

impl<T: IoBus + ?Sized> IoBus for Box<T> {
    #[inline]
    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception> {
        <T as IoBus>::io_read(&mut **self, port, size)
    }

    #[inline]
    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception> {
        <T as IoBus>::io_write(&mut **self, port, size, val)
    }
}

/// Default port I/O backend used by [`PagingBus::new`].
///
/// This preserves the historical behaviour of `PagingBus` where port I/O was
/// stubbed out (reads return 0 and writes are ignored).
#[derive(Clone, Copy, Default)]
pub struct NoIo;

impl IoBus for NoIo {
    #[inline]
    fn io_read(&mut self, _port: u16, _size: u32) -> Result<u64, Exception> {
        Ok(0)
    }

    #[inline]
    fn io_write(&mut self, _port: u16, _size: u32, _val: u64) -> Result<(), Exception> {
        Ok(())
    }
}

/// A paging-aware [`CpuBus`] implementation backed by [`aero_mmu::Mmu`].
///
/// The tier-0 interpreter passes *linear* addresses to [`CpuBus`] methods. This
/// adapter translates them to physical addresses via `aero-mmu` before accessing
/// the underlying physical bus `B`.
pub struct PagingBus<B, IO = NoIo> {
    /// Boxed, not inline.
    ///
    /// The MMU carries a full software TLB, which makes the struct around it several
    /// kilobytes wide. `Machine` builds one of these on every outer-loop batch boundary
    /// and swaps its durable MMU in, so an inline TLB turned each boundary into seven
    /// multi-kilobyte `memmove`s — the construction, the swap's three copies, and the
    /// moves into the wrapping bus. Measured at 66KB per boundary, 2.5GB across a 120M
    /// instruction window, ~3.4 host instructions per guest instruction. Behind a box,
    /// all of that becomes pointer-sized, at the cost of one indirection per MMU access.
    mmu: Box<Mmu>,
    phys: B,
    io: IO,
    cpl: u8,
    /// Virtual and physical base of the page the last instruction fetch resolved.
    ///
    /// Straight-line execution fetches from the same page for many consecutive
    /// instructions, and re-deriving that translation is a large share of the
    /// interpreter's per-instruction cost. Architecturally this is the same
    /// guarantee a TLB gives: a translation stays usable until something that
    /// requires a flush happens, so it is dropped whenever the MMU's control
    /// registers change, on `INVLPG`, on a `CR3` write, and on a privilege
    /// change (which can change the page's accessibility).
    fetch_page: Option<(u64, u64)>,
    write_chunks: Vec<(u64, usize, usize)>,
    scratch: Vec<u8>,
}

const PAGE_SIZE: u64 = 4096;

impl<B> PagingBus<B, NoIo> {
    pub fn new(phys: B) -> PagingBus<B, NoIo> {
        PagingBus::new_with_io(phys, NoIo)
    }
}

impl<B, IO> PagingBus<B, IO> {
    pub fn new_with_io(phys: B, io: IO) -> PagingBus<B, IO> {
        Self::with_mmu_and_io(Mmu::new(), phys, io)
    }

    /// Construct a paging bus around an existing [`Mmu`].
    ///
    /// Prefer this over [`PagingBus::new_with_io`] when the caller already owns a durable
    /// MMU (e.g. `Machine`'s TLB across batches): `new_with_io` allocates and zeroes a full
    /// software TLB that is immediately discarded via swap.
    pub fn with_mmu_and_io(mmu: Mmu, phys: B, io: IO) -> PagingBus<B, IO> {
        Self::with_boxed_mmu_and_io(Box::new(mmu), phys, io)
    }

    /// As [`PagingBus::with_mmu_and_io`], but takes the MMU already boxed.
    ///
    /// This is the one that costs nothing to call in a loop: handing over an owned `Mmu`
    /// moves the whole software TLB, and a caller that rebuilds the bus every batch pays
    /// that move every time.
    pub fn with_boxed_mmu_and_io(mmu: Box<Mmu>, phys: B, io: IO) -> PagingBus<B, IO> {
        Self {
            mmu,
            phys,
            io,
            cpl: 0,
            fetch_page: None,
            write_chunks: Vec::new(),
            scratch: Vec::with_capacity(PAGE_SIZE as usize),
        }
    }

    #[inline]
    pub fn mmu(&self) -> &Mmu {
        &self.mmu
    }

    /// Hands out the MMU for direct mutation, so the memoised fetch translation
    /// has to be dropped: the caller may swap the whole MMU or flush its TLB.
    #[inline]
    pub fn mmu_mut(&mut self) -> &mut Mmu {
        self.fetch_page = None;
        &mut self.mmu
    }

    /// The MMU's box itself, so a caller can swap in another without moving either TLB.
    ///
    /// Same invalidation obligation as [`PagingBus::mmu_mut`], and more of it: the
    /// memoised fetch translation belongs to whichever MMU was here before.
    #[inline]
    pub fn boxed_mmu_mut(&mut self) -> &mut Box<Mmu> {
        self.fetch_page = None;
        &mut self.mmu
    }

    /// Consume the bus and return the MMU (preserving TLB state) plus the physical backend.
    #[inline]
    pub fn into_mmu_and_phys(self) -> (Mmu, B) {
        (*self.mmu, self.phys)
    }

    /// As [`PagingBus::into_mmu_and_phys`], without unboxing the MMU.
    #[inline]
    pub fn into_boxed_mmu_and_phys(self) -> (Box<Mmu>, B) {
        (self.mmu, self.phys)
    }

    #[inline]
    pub fn into_inner(self) -> B {
        self.phys
    }

    #[inline]
    pub fn inner(&self) -> &B {
        &self.phys
    }

    #[inline]
    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.phys
    }

    #[inline]
    pub fn io(&self) -> &IO {
        &self.io
    }

    #[inline]
    pub fn io_mut(&mut self) -> &mut IO {
        &mut self.io
    }

    #[inline]
    fn translate(&mut self, vaddr: u64, access: AccessType) -> Result<u64, Exception>
    where
        B: MemoryBus,
    {
        match self.mmu.translate(&mut self.phys, vaddr, access, self.cpl) {
            Ok(paddr) => Ok(paddr),
            Err(TranslateFault::PageFault(pf)) => Err(Exception::PageFault {
                addr: pf.addr,
                error_code: pf.error_code,
            }),
            Err(TranslateFault::NonCanonical(_addr)) => Err(Exception::gp0()),
        }
    }

    /// Translate a range without performing any guest-visible side effects.
    ///
    /// Returns `Ok(true)` when the entire range can be accessed, and `Ok(false)` when translation
    /// fails with a page fault (callers should fall back to scalar accesses to preserve correct
    /// architectural partial-progress semantics).
    ///
    /// Non-canonical linear addresses return `Ok(false)` so callers can fall back to scalar
    /// accesses, which preserve correct architectural partial-progress semantics (e.g. REP string
    /// ops that cross the canonical boundary in long mode).
    fn preflight_range_probe(
        &mut self,
        vaddr: u64,
        len: usize,
        access: AccessType,
    ) -> Result<bool, Exception>
    where
        B: MemoryBus,
    {
        if len == 0 {
            return Ok(true);
        }

        let mut offset = 0usize;
        while offset < len {
            // Avoid panicking on overflow in debug builds; treat wrap as non-contiguous so callers
            // can fall back to scalar accesses (which handle architectural wrapping semantics when
            // needed).
            let addr = match vaddr.checked_add(offset as u64) {
                Some(v) => v,
                None => return Ok(false),
            };
            match self
                .mmu
                .translate_probe(&mut self.phys, addr, access, self.cpl)
            {
                Ok(_paddr) => {}
                Err(TranslateFault::PageFault(_pf)) => return Ok(false),
                Err(TranslateFault::NonCanonical(_addr)) => return Ok(false),
            }

            let page_off = (addr & (PAGE_SIZE - 1)) as usize;
            let page_rem = (PAGE_SIZE as usize) - page_off;
            let chunk_len = page_rem.min(len - offset);
            offset += chunk_len;
        }

        Ok(true)
    }

    #[inline]
    fn read_u8_access(&mut self, vaddr: u64, access: AccessType) -> Result<u8, Exception>
    where
        B: MemoryBus,
    {
        let paddr = self.translate(vaddr, access)?;
        Ok(self.phys.read_u8(paddr))
    }

    #[inline]
    fn read_u16_access(&mut self, vaddr: u64, access: AccessType) -> Result<u16, Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 2 {
            let paddr = self.translate(vaddr, access)?;
            return Ok(self.phys.read_u16(paddr));
        }

        let mut buf = [0u8; 2];
        self.read_bytes_access(vaddr, &mut buf, access)?;
        Ok(u16::from_le_bytes(buf))
    }

    #[inline]
    fn read_u32_access(&mut self, vaddr: u64, access: AccessType) -> Result<u32, Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 4 {
            let paddr = self.translate(vaddr, access)?;
            return Ok(self.phys.read_u32(paddr));
        }

        let mut buf = [0u8; 4];
        self.read_bytes_access(vaddr, &mut buf, access)?;
        Ok(u32::from_le_bytes(buf))
    }

    #[inline]
    fn read_u64_access(&mut self, vaddr: u64, access: AccessType) -> Result<u64, Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 8 {
            let paddr = self.translate(vaddr, access)?;
            return Ok(self.phys.read_u64(paddr));
        }

        let mut buf = [0u8; 8];
        self.read_bytes_access(vaddr, &mut buf, access)?;
        Ok(u64::from_le_bytes(buf))
    }

    #[inline]
    fn write_u8_access(
        &mut self,
        vaddr: u64,
        access: AccessType,
        value: u8,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        let paddr = self.translate(vaddr, access)?;
        self.phys.write_u8(paddr, value);
        Ok(())
    }

    #[inline]
    fn write_u16_access(
        &mut self,
        vaddr: u64,
        access: AccessType,
        value: u16,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 2 {
            let paddr = self.translate(vaddr, access)?;
            self.phys.write_u16(paddr, value);
            return Ok(());
        }

        self.write_bytes_access(vaddr, &value.to_le_bytes(), access)
    }

    #[inline]
    fn write_u32_access(
        &mut self,
        vaddr: u64,
        access: AccessType,
        value: u32,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 4 {
            let paddr = self.translate(vaddr, access)?;
            self.phys.write_u32(paddr, value);
            return Ok(());
        }

        self.write_bytes_access(vaddr, &value.to_le_bytes(), access)
    }

    #[inline]
    fn write_u64_access(
        &mut self,
        vaddr: u64,
        access: AccessType,
        value: u64,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        let page_off = vaddr & (PAGE_SIZE - 1);
        if page_off <= PAGE_SIZE - 8 {
            let paddr = self.translate(vaddr, access)?;
            self.phys.write_u64(paddr, value);
            return Ok(());
        }

        self.write_bytes_access(vaddr, &value.to_le_bytes(), access)
    }

    fn read_bytes_access(
        &mut self,
        vaddr: u64,
        dst: &mut [u8],
        access: AccessType,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        if dst.is_empty() {
            return Ok(());
        }

        let mut offset = 0usize;
        while offset < dst.len() {
            let addr = vaddr.wrapping_add(offset as u64);
            let paddr = self.translate(addr, access)?;

            let page_off = (addr & (PAGE_SIZE - 1)) as usize;
            let page_rem = (PAGE_SIZE as usize) - page_off;
            let chunk_len = page_rem.min(dst.len() - offset);

            self.phys
                .read_bytes(paddr, &mut dst[offset..offset + chunk_len]);

            offset += chunk_len;
        }

        Ok(())
    }

    fn write_bytes_access(
        &mut self,
        vaddr: u64,
        src: &[u8],
        access: AccessType,
    ) -> Result<(), Exception>
    where
        B: MemoryBus,
    {
        if src.is_empty() {
            return Ok(());
        }

        self.write_chunks.clear();
        let mut offset = 0usize;
        while offset < src.len() {
            let addr = vaddr.wrapping_add(offset as u64);
            let paddr = self.translate(addr, access)?;

            let page_off = (addr & (PAGE_SIZE - 1)) as usize;
            let page_rem = (PAGE_SIZE as usize) - page_off;
            let chunk_len = page_rem.min(src.len() - offset);

            self.write_chunks.push((paddr, chunk_len, offset));
            offset += chunk_len;
        }

        let phys = &mut self.phys;
        for (paddr, len, src_off) in self.write_chunks.iter().copied() {
            phys.write_bytes(paddr, &src[src_off..src_off + len]);
        }

        Ok(())
    }
}

/// Adapter that performs reads with write-intent access checks.
///
/// Tier-0 uses [`CpuBus::atomic_rmw`] to model `LOCK`ed RMW instructions. Even if
/// the computed new value equals the old one, real hardware still performs a
/// write-intent access (and thus may set accessed/dirty bits and fault on
/// read-only pages). By performing reads with [`AccessType::Write`], we ensure
/// those permission checks happen.
struct WriteIntent<'a, B, IO> {
    bus: &'a mut PagingBus<B, IO>,
}

impl<B: MemoryBus, IO: IoBus> CpuBus for WriteIntent<'_, B, IO> {
    fn sync(&mut self, state: &crate::state::CpuState) {
        self.bus.sync(state);
    }

    fn invlpg(&mut self, vaddr: u64) {
        self.bus.invlpg(vaddr);
    }

    fn jit_translate(&mut self, vaddr: u64, write: bool) -> Option<(u64, bool)> {
        self.bus.jit_translate(vaddr, write)
    }

    fn jit_tlb_generation(&mut self) -> u64 {
        self.bus.jit_tlb_generation()
    }

    fn write_cr3(&mut self, value: u64) {
        self.bus.write_cr3(value);
    }

    fn read_u8(&mut self, vaddr: u64) -> Result<u8, Exception> {
        self.bus.read_u8_access(vaddr, AccessType::Write)
    }

    fn read_u16(&mut self, vaddr: u64) -> Result<u16, Exception> {
        self.bus.read_u16_access(vaddr, AccessType::Write)
    }

    fn read_u32(&mut self, vaddr: u64) -> Result<u32, Exception> {
        self.bus.read_u32_access(vaddr, AccessType::Write)
    }

    fn read_u64(&mut self, vaddr: u64) -> Result<u64, Exception> {
        self.bus.read_u64_access(vaddr, AccessType::Write)
    }

    fn read_u128(&mut self, vaddr: u64) -> Result<u128, Exception> {
        let mut buf = [0u8; 16];
        self.bus
            .read_bytes_access(vaddr, &mut buf, AccessType::Write)?;
        Ok(u128::from_le_bytes(buf))
    }

    fn write_u8(&mut self, vaddr: u64, val: u8) -> Result<(), Exception> {
        self.bus.write_u8_access(vaddr, AccessType::Write, val)
    }

    fn write_u16(&mut self, vaddr: u64, val: u16) -> Result<(), Exception> {
        self.bus.write_u16_access(vaddr, AccessType::Write, val)
    }

    fn write_u32(&mut self, vaddr: u64, val: u32) -> Result<(), Exception> {
        self.bus.write_u32_access(vaddr, AccessType::Write, val)
    }

    fn write_u64(&mut self, vaddr: u64, val: u64) -> Result<(), Exception> {
        self.bus.write_u64_access(vaddr, AccessType::Write, val)
    }

    fn write_u128(&mut self, vaddr: u64, val: u128) -> Result<(), Exception> {
        self.bus
            .write_bytes_access(vaddr, &val.to_le_bytes(), AccessType::Write)
    }

    fn read_bytes(&mut self, vaddr: u64, dst: &mut [u8]) -> Result<(), Exception> {
        self.bus.read_bytes_access(vaddr, dst, AccessType::Write)
    }

    fn write_bytes(&mut self, vaddr: u64, src: &[u8]) -> Result<(), Exception> {
        self.bus.write_bytes_access(vaddr, src, AccessType::Write)
    }

    fn fetch(&mut self, vaddr: u64, max_len: usize) -> Result<[u8; 15], Exception> {
        self.bus.fetch(vaddr, max_len)
    }

    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception> {
        self.bus.io_read(port, size)
    }

    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception> {
        self.bus.io_write(port, size, val)
    }
}

impl<B, IO> CpuBus for PagingBus<B, IO>
where
    B: MemoryBus,
    IO: IoBus,
{
    fn sync(&mut self, state: &crate::state::CpuState) {
        let mmu_changed = state.sync_mmu(&mut self.mmu);
        let cpl = state.cpl();
        if mmu_changed || cpl != self.cpl {
            self.fetch_page = None;
        }
        self.cpl = cpl;
    }

    fn invlpg(&mut self, vaddr: u64) {
        self.fetch_page = None;
        self.mmu.invlpg(vaddr);
    }

    fn jit_tlb_generation(&mut self) -> u64 {
        self.mmu.tlb_generation()
    }

    fn jit_translate(&mut self, vaddr: u64, write: bool) -> Option<(u64, bool)> {
        let access = if write {
            AccessType::Write
        } else {
            AccessType::Read
        };
        // A fault here is not ours to raise: report no mapping and let the
        // interpreter re-execute the instruction so it raises the fault with the
        // right error code and CR2.
        let paddr = self.translate(vaddr, access).ok()?;
        let page = paddr & !(PAGE_SIZE - 1);
        let is_ram = self.phys.is_ram(page);
        Some((page, is_ram))
    }

    fn write_cr3(&mut self, value: u64) {
        if std::env::var_os("AERO_LOG_CR3_WRITE").is_some() {
            eprintln!(
                "AERO_LOG_CR3_WRITE: old={:#x} new={value:#x} cr4={:#x}",
                self.mmu.cr3(),
                self.mmu.cr4()
            );
        }
        self.fetch_page = None;
        self.mmu.set_cr3(value);
    }

    fn read_u8(&mut self, vaddr: u64) -> Result<u8, Exception> {
        self.read_u8_access(vaddr, AccessType::Read)
    }

    fn read_u16(&mut self, vaddr: u64) -> Result<u16, Exception> {
        self.read_u16_access(vaddr, AccessType::Read)
    }

    fn read_u32(&mut self, vaddr: u64) -> Result<u32, Exception> {
        self.read_u32_access(vaddr, AccessType::Read)
    }

    fn read_u64(&mut self, vaddr: u64) -> Result<u64, Exception> {
        self.read_u64_access(vaddr, AccessType::Read)
    }

    fn read_u128(&mut self, vaddr: u64) -> Result<u128, Exception> {
        let mut buf = [0u8; 16];
        self.read_bytes_access(vaddr, &mut buf, AccessType::Read)?;
        Ok(u128::from_le_bytes(buf))
    }

    fn write_u8(&mut self, vaddr: u64, val: u8) -> Result<(), Exception> {
        self.write_u8_access(vaddr, AccessType::Write, val)
    }

    fn write_u16(&mut self, vaddr: u64, val: u16) -> Result<(), Exception> {
        self.write_u16_access(vaddr, AccessType::Write, val)
    }

    fn write_u32(&mut self, vaddr: u64, val: u32) -> Result<(), Exception> {
        self.write_u32_access(vaddr, AccessType::Write, val)
    }

    fn write_u64(&mut self, vaddr: u64, val: u64) -> Result<(), Exception> {
        self.write_u64_access(vaddr, AccessType::Write, val)
    }

    fn write_u128(&mut self, vaddr: u64, val: u128) -> Result<(), Exception> {
        self.write_bytes_access(vaddr, &val.to_le_bytes(), AccessType::Write)
    }

    fn read_bytes(&mut self, vaddr: u64, dst: &mut [u8]) -> Result<(), Exception> {
        self.read_bytes_access(vaddr, dst, AccessType::Read)
    }

    fn write_bytes(&mut self, vaddr: u64, src: &[u8]) -> Result<(), Exception> {
        self.write_bytes_access(vaddr, src, AccessType::Write)
    }

    fn preflight_write_bytes(&mut self, vaddr: u64, len: usize) -> Result<(), Exception> {
        // Translate the full range with write intent, but do not touch the target bytes.
        // This allows higher-level helpers (e.g. wrapped multi-byte writes) to remain
        // atomic w.r.t #PF even when the access must be split into multiple segments.
        if len == 0 {
            return Ok(());
        }

        let mut offset = 0usize;
        while offset < len {
            let addr = vaddr.wrapping_add(offset as u64);
            let _paddr = self.translate(addr, AccessType::Write)?;

            let page_off = (addr & (PAGE_SIZE - 1)) as usize;
            let page_rem = (PAGE_SIZE as usize) - page_off;
            let chunk_len = page_rem.min(len - offset);

            offset += chunk_len;
        }

        Ok(())
    }

    fn supports_bulk_copy(&self) -> bool {
        true
    }

    fn bulk_copy(&mut self, dst: u64, src: u64, len: usize) -> Result<bool, Exception> {
        if len == 0 || dst == src {
            return Ok(true);
        }

        // Treat address arithmetic overflow as a reason to decline the bulk fast path so callers
        // can fall back to scalar accesses (which preserve correct architectural wrap/partial
        // progress semantics). Do not surface `MemoryFault` here: it's an internal error class and
        // is not appropriate for guest-visible behavior.
        let Ok(len_u64) = u64::try_from(len) else {
            return Ok(false);
        };
        if src.checked_add(len_u64).is_none() || dst.checked_add(len_u64).is_none() {
            return Ok(false);
        }

        if !self.preflight_range_probe(src, len, AccessType::Read)? {
            return Ok(false);
        }
        if !self.preflight_range_probe(dst, len, AccessType::Write)? {
            return Ok(false);
        }

        // Memmove semantics: choose copy direction based on overlap.
        let src_end = src + len_u64;
        let dst_end = dst + len_u64;
        let overlap = src < dst_end && dst < src_end;
        let copy_backward = overlap && dst > src;

        if copy_backward {
            let mut remaining = len;
            while remaining > 0 {
                let idx = remaining - 1;
                let src_addr = src + idx as u64;
                let dst_addr = dst + idx as u64;

                let src_page_off = (src_addr & (PAGE_SIZE - 1)) as usize;
                let dst_page_off = (dst_addr & (PAGE_SIZE - 1)) as usize;
                let chunk_len = (src_page_off + 1).min(dst_page_off + 1).min(remaining);
                let chunk_start = remaining - chunk_len;

                let src_chunk_addr = src + chunk_start as u64;
                let dst_chunk_addr = dst + chunk_start as u64;
                let src_paddr = self.translate(src_chunk_addr, AccessType::Read)?;
                let dst_paddr = self.translate(dst_chunk_addr, AccessType::Write)?;

                self.scratch.resize(chunk_len, 0);
                self.phys
                    .read_bytes(src_paddr, &mut self.scratch[..chunk_len]);
                self.phys.write_bytes(dst_paddr, &self.scratch[..chunk_len]);

                remaining -= chunk_len;
            }
        } else {
            let mut offset = 0usize;
            while offset < len {
                let src_addr = src + offset as u64;
                let dst_addr = dst + offset as u64;
                let src_paddr = self.translate(src_addr, AccessType::Read)?;
                let dst_paddr = self.translate(dst_addr, AccessType::Write)?;

                let src_page_off = (src_addr & (PAGE_SIZE - 1)) as usize;
                let dst_page_off = (dst_addr & (PAGE_SIZE - 1)) as usize;
                let src_page_rem = (PAGE_SIZE as usize) - src_page_off;
                let dst_page_rem = (PAGE_SIZE as usize) - dst_page_off;
                let chunk_len = src_page_rem.min(dst_page_rem).min(len - offset);

                self.scratch.resize(chunk_len, 0);
                self.phys
                    .read_bytes(src_paddr, &mut self.scratch[..chunk_len]);
                self.phys.write_bytes(dst_paddr, &self.scratch[..chunk_len]);

                offset += chunk_len;
            }
        }

        Ok(true)
    }

    fn supports_bulk_set(&self) -> bool {
        true
    }

    fn bulk_set(&mut self, dst: u64, pattern: &[u8], repeat: usize) -> Result<bool, Exception> {
        if repeat == 0 || pattern.is_empty() {
            return Ok(true);
        }

        // Like `bulk_copy`, decline on overflow so callers can fall back to scalar accesses.
        let Some(total) = pattern.len().checked_mul(repeat) else {
            return Ok(false);
        };
        let Ok(total_u64) = u64::try_from(total) else {
            return Ok(false);
        };
        if dst.checked_add(total_u64).is_none() {
            return Ok(false);
        }

        if !self.preflight_range_probe(dst, total, AccessType::Write)? {
            return Ok(false);
        }

        let pat_len = pattern.len();
        let mut offset = 0usize;
        while offset < total {
            let addr = dst + offset as u64;
            let paddr = self.translate(addr, AccessType::Write)?;

            let page_off = (addr & (PAGE_SIZE - 1)) as usize;
            let page_rem = (PAGE_SIZE as usize) - page_off;
            let chunk_len = page_rem.min(total - offset);

            self.scratch.resize(chunk_len, 0);

            if pat_len == 1 {
                self.scratch[..chunk_len].fill(pattern[0]);
            } else {
                let mut filled = 0usize;
                let mut pat_off = offset % pat_len;
                while filled < chunk_len {
                    let run = (pat_len - pat_off).min(chunk_len - filled);
                    self.scratch[filled..filled + run]
                        .copy_from_slice(&pattern[pat_off..pat_off + run]);
                    filled += run;
                    pat_off = 0;
                }
            }

            self.phys.write_bytes(paddr, &self.scratch[..chunk_len]);

            offset += chunk_len;
        }

        Ok(true)
    }
    fn atomic_rmw<T, R>(&mut self, vaddr: u64, f: impl FnOnce(T) -> (T, R)) -> Result<R, Exception>
    where
        T: crate::mem::CpuBusValue,
        Self: Sized,
    {
        // Perform the read with write-intent translation so that permission
        // checks and accessed/dirty bit updates match real RMW semantics.
        let old = {
            let mut intent = WriteIntent { bus: self };
            T::read_from(&mut intent, vaddr)?
        };
        let (new, ret) = f(old);
        if new != old {
            let mut intent = WriteIntent { bus: self };
            T::write_to(&mut intent, vaddr, new)?;
        }
        Ok(ret)
    }

    fn fetch(&mut self, vaddr: u64, max_len: usize) -> Result<[u8; 15], Exception> {
        let mut buf = [0u8; 15];
        let len = max_len.min(15);
        let page_off = vaddr & (PAGE_SIZE - 1);

        // A window that stays inside one page can reuse the previous fetch's
        // translation. Anything else (a page-crossing window, or the first fetch
        // after a flush) goes the long way and re-arms the memo.
        if len != 0 && page_off + len as u64 <= PAGE_SIZE {
            let vpage = vaddr - page_off;
            if let Some((memo_vpage, memo_ppage)) = self.fetch_page {
                if memo_vpage == vpage {
                    // A full window is the overwhelmingly common case, and asking
                    // for a constant 15 bytes lets the backend copy them inline
                    // instead of calling `memcpy` once per instruction.
                    if len == 15 {
                        return Ok(self.phys.read_insn_window(memo_ppage + page_off));
                    }
                    self.phys.read_bytes(memo_ppage + page_off, &mut buf[..len]);
                    return Ok(buf);
                }
            }
            let paddr = self.translate(vaddr, AccessType::Execute)?;
            self.fetch_page = Some((vpage, paddr - page_off));
            if len == 15 {
                return Ok(self.phys.read_insn_window(paddr));
            }
            self.phys.read_bytes(paddr, &mut buf[..len]);
            return Ok(buf);
        }

        self.read_bytes_access(vaddr, &mut buf[..len], AccessType::Execute)?;
        Ok(buf)
    }

    fn io_read(&mut self, port: u16, size: u32) -> Result<u64, Exception> {
        self.io.io_read(port, size)
    }

    fn io_write(&mut self, port: u16, size: u32, val: u64) -> Result<(), Exception> {
        self.io.io_write(port, size, val)
    }
}

impl<B: fmt::Debug, IO> fmt::Debug for PagingBus<B, IO> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PagingBus")
            .field("mmu", &self.mmu)
            .field("cpl", &self.cpl)
            .field("phys", &self.phys)
            // Avoid requiring `IO: Debug` (real backends often aren't).
            .field("io", &core::any::type_name::<IO>())
            .finish()
    }
}

#[cfg(test)]
mod fetch_memo_tests {
    use super::*;
    use crate::state::{CpuMode, CpuState};

    const PTE_P: u64 = 1 << 0;
    const PTE_RW: u64 = 1 << 1;
    const PTE_US: u64 = 1 << 2;
    const CR0_PG: u64 = 1 << 31;
    const CR4_PAE: u64 = 1 << 5;
    const EFER_LME: u64 = 1 << 8;
    const EFER_LMA: u64 = 1 << 10;

    const PML4: u64 = 0x1000;
    const PDPT: u64 = 0x2000;
    const PD: u64 = 0x3000;
    const PT: u64 = 0x4000;
    /// The page the guest executes from, and the two physical frames it is
    /// pointed at in turn.
    const CODE_VA: u64 = 0x10_0000;
    const FRAME_A: u64 = 0x8000;
    const FRAME_B: u64 = 0x9000;

    struct FlatMem(Vec<u8>);

    impl FlatMem {
        fn put_u64(&mut self, paddr: u64, value: u64) {
            let off = paddr as usize;
            self.0[off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    impl MemoryBus for FlatMem {
        fn read_u8(&mut self, paddr: u64) -> u8 {
            self.0[paddr as usize]
        }

        fn read_u16(&mut self, paddr: u64) -> u16 {
            let o = paddr as usize;
            u16::from_le_bytes(self.0[o..o + 2].try_into().unwrap())
        }

        fn read_u32(&mut self, paddr: u64) -> u32 {
            let o = paddr as usize;
            u32::from_le_bytes(self.0[o..o + 4].try_into().unwrap())
        }

        fn read_u64(&mut self, paddr: u64) -> u64 {
            let o = paddr as usize;
            u64::from_le_bytes(self.0[o..o + 8].try_into().unwrap())
        }

        fn write_u8(&mut self, paddr: u64, value: u8) {
            self.0[paddr as usize] = value;
        }

        fn write_u16(&mut self, paddr: u64, value: u16) {
            let o = paddr as usize;
            self.0[o..o + 2].copy_from_slice(&value.to_le_bytes());
        }

        fn write_u32(&mut self, paddr: u64, value: u32) {
            let o = paddr as usize;
            self.0[o..o + 4].copy_from_slice(&value.to_le_bytes());
        }

        fn write_u64(&mut self, paddr: u64, value: u64) {
            let o = paddr as usize;
            self.0[o..o + 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Long-mode paging with `CODE_VA` mapped to `FRAME_A`, whose first byte is
    /// `0xAA`, while `FRAME_B` holds `0xBB`. Remapping the leaf PTE between
    /// fetches is what makes a stale memo observable.
    fn bus_with_code_page() -> (PagingBus<FlatMem, NoIo>, CpuState) {
        let mut mem = FlatMem(vec![0u8; 0x20000]);
        mem.put_u64(PML4, PDPT | PTE_P | PTE_RW | PTE_US);
        mem.put_u64(PDPT, PD | PTE_P | PTE_RW | PTE_US);
        mem.put_u64(PD, PT | PTE_P | PTE_RW | PTE_US);
        let pte = PT + ((CODE_VA >> 12) & 0x1ff) * 8;
        mem.put_u64(pte, FRAME_A | PTE_P | PTE_RW | PTE_US);
        mem.0[FRAME_A as usize] = 0xAA;
        mem.0[FRAME_B as usize] = 0xBB;

        let mut state = CpuState::new(CpuMode::Long);
        state.control.cr0 |= CR0_PG;
        state.control.cr3 = PML4;
        state.control.cr4 |= CR4_PAE;
        state.msr.efer |= EFER_LME | EFER_LMA;

        let mut bus = PagingBus::new(mem);
        bus.sync(&state);
        (bus, state)
    }

    fn remap_code_page_to(bus: &mut PagingBus<FlatMem, NoIo>, frame: u64) {
        let pte = PT + ((CODE_VA >> 12) & 0x1ff) * 8;
        bus.phys.put_u64(pte, frame | PTE_P | PTE_RW | PTE_US);
    }

    #[test]
    fn repeated_fetch_in_one_page_reuses_the_memoised_translation() {
        let (mut bus, _state) = bus_with_code_page();
        assert_eq!(bus.fetch(CODE_VA, 1).expect("first fetch")[0], 0xAA);
        assert_eq!(bus.fetch_page, Some((CODE_VA, FRAME_A)));
        // A second fetch at a different offset in the same page must still be
        // served, and must not re-arm the memo with a different page.
        assert_eq!(bus.fetch(CODE_VA + 8, 1).expect("second fetch")[0], 0x00);
        assert_eq!(bus.fetch_page, Some((CODE_VA, FRAME_A)));
    }

    #[test]
    fn invlpg_drops_the_memoised_fetch_translation() {
        let (mut bus, _state) = bus_with_code_page();
        assert_eq!(bus.fetch(CODE_VA, 1).expect("prime")[0], 0xAA);

        remap_code_page_to(&mut bus, FRAME_B);
        bus.invlpg(CODE_VA);

        assert_eq!(
            bus.fetch(CODE_VA, 1).expect("after invlpg")[0],
            0xBB,
            "INVLPG must make the new mapping visible to instruction fetch"
        );
    }

    #[test]
    fn cr3_write_drops_the_memoised_fetch_translation() {
        let (mut bus, _state) = bus_with_code_page();
        assert_eq!(bus.fetch(CODE_VA, 1).expect("prime")[0], 0xAA);

        remap_code_page_to(&mut bus, FRAME_B);
        // Reloading CR3 with the same value is still an architectural flush.
        bus.write_cr3(PML4);

        assert_eq!(bus.fetch(CODE_VA, 1).expect("after cr3")[0], 0xBB);
    }

    #[test]
    fn sync_after_a_control_register_change_drops_the_memoised_translation() {
        let (mut bus, mut state) = bus_with_code_page();
        assert_eq!(bus.fetch(CODE_VA, 1).expect("prime")[0], 0xAA);

        remap_code_page_to(&mut bus, FRAME_B);
        // A guest `MOV CR3` reaches the bus as a changed control register at the
        // next instruction boundary rather than through `write_cr3`.
        state.control.cr3 = PML4;
        state.control.cr4 |= 1 << 7; // CR4.PGE
        bus.sync(&state);

        assert_eq!(bus.fetch(CODE_VA, 1).expect("after sync")[0], 0xBB);
    }

    #[test]
    fn handing_out_the_mmu_drops_the_memoised_translation() {
        let (mut bus, _state) = bus_with_code_page();
        assert_eq!(bus.fetch(CODE_VA, 1).expect("prime")[0], 0xAA);

        remap_code_page_to(&mut bus, FRAME_B);
        // `Machine` swaps its durable MMU in through this accessor every batch.
        bus.mmu_mut().invlpg(CODE_VA);

        assert_eq!(bus.fetch(CODE_VA, 1).expect("after mmu_mut")[0], 0xBB);
    }

    #[test]
    fn a_page_crossing_fetch_window_does_not_use_the_memo() {
        let (mut bus, _state) = bus_with_code_page();
        // Map the page after the code page so a 15-byte window can span both.
        let next_pte = PT + (((CODE_VA + 0x1000) >> 12) & 0x1ff) * 8;
        bus.phys
            .put_u64(next_pte, FRAME_B | PTE_P | PTE_RW | PTE_US);

        let near_end = CODE_VA + 0xffc;
        let window = bus.fetch(near_end, 15).expect("crossing fetch");
        assert_eq!(
            window[4], 0xBB,
            "bytes past the page boundary come from the next frame"
        );
    }
}

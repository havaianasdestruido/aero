//! Guest RAM that lives inside the JIT's WASM linear memory.
//!
//! Compiled Tier-1 blocks address guest RAM as offsets into the linear memory they
//! import. When the machine's RAM is a separate allocation, every guest load and
//! store in compiled code has to call out to a host helper to reach it — and since
//! roughly a fifth of the machine's host instructions are already memory and MMU
//! work, a compiled tier that keeps that call is bounded by Amdahl at about 2x no
//! matter how good the code generation is.
//!
//! Making the machine's RAM *be* that linear memory removes the call: with the
//! inline TLB enabled a guest load becomes a bounds-checked WASM load, which the
//! browser and Wasmtime both compile to a native load against a fixed base.
//!
//! The region is laid out with the CPU state at the bottom and RAM above it, so RAM
//! can exceed the 2GiB that a signed `i32` ABI pointer would otherwise cap it at.
//! `JitContext::ram_base` already exists to carry that offset to compiled code.

use std::cell::RefCell;
use std::rc::Rc;

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use memory::phys::{GuestMemory, GuestMemoryError, GuestMemoryResult};

/// Bytes reserved below guest RAM for `CpuState` and the JIT context.
///
/// One WASM page. `CpuState` plus the JIT context header and inline TLB fit inside
/// it, and keeping RAM page-aligned means a guest physical page maps to a linear
/// memory page.
pub const RAM_BASE: u64 = 0x1_0000;

/// Guest RAM backed by a `WasmtimeBackend`'s linear memory.
///
/// Holds the backend alive because the pointer below aliases its memory. Access
/// goes through the raw pointer rather than the `RefCell`, so the interpreter —
/// which reaches guest RAM on most instructions — pays nothing for the sharing.
pub struct WasmGuestRam {
    /// Kept only so the linear memory outlives `base`.
    _backend: Rc<RefCell<WasmtimeBackend<CpuState>>>,
    base: *mut u8,
    len: u64,
}

impl WasmGuestRam {
    /// Claim `len` bytes at [`RAM_BASE`] in `backend`'s linear memory as guest RAM.
    ///
    /// Returns `None` if the backend's memory is too small to hold both the state
    /// region and that much RAM, or if its guest RAM has already been claimed.
    ///
    /// Safe because the backend hands out exactly one such view: compiled blocks
    /// writing the same bytes is the design, but a second *host-side* view would be
    /// two `&mut` to one region.
    pub fn new(backend: Rc<RefCell<WasmtimeBackend<CpuState>>>, len: u64) -> Option<Self> {
        let (ptr, total) = backend.borrow_mut().claim_guest_ram()?;
        let need = RAM_BASE.checked_add(len)?;
        if need > total as u64 {
            return None;
        }
        Some(Self {
            _backend: backend,
            base: ptr.wrapping_add(RAM_BASE as usize),
            len,
        })
    }

    #[inline]
    fn range(&self, paddr: u64, len: usize) -> GuestMemoryResult<usize> {
        let oob = || GuestMemoryError::OutOfRange {
            paddr,
            len,
            size: self.len,
        };
        let end = paddr.checked_add(len as u64).ok_or_else(oob)?;
        if end > self.len {
            return Err(oob());
        }
        usize::try_from(paddr).map_err(|_| oob())
    }
}

impl GuestMemory for WasmGuestRam {
    fn size(&self) -> u64 {
        self.len
    }

    fn read_into(&self, paddr: u64, dst: &mut [u8]) -> GuestMemoryResult<()> {
        let off = self.range(paddr, dst.len())?;
        // Safety: `range` bounded the access, and `base` is valid for `len` bytes
        // for as long as `_backend` is alive.
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(off), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn write_from(&mut self, paddr: u64, src: &[u8]) -> GuestMemoryResult<()> {
        let off = self.range(paddr, src.len())?;
        // Safety: as above.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.base.add(off), src.len());
        }
        Ok(())
    }

    fn get_slice(&self, paddr: u64, len: usize) -> Option<&[u8]> {
        let off = self.range(paddr, len).ok()?;
        // Safety: as above.
        Some(unsafe { std::slice::from_raw_parts(self.base.add(off), len) })
    }

    fn get_slice_mut(&mut self, paddr: u64, len: usize) -> Option<&mut [u8]> {
        let off = self.range(paddr, len).ok()?;
        // Safety: as above.
        Some(unsafe { std::slice::from_raw_parts_mut(self.base.add(off), len) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(pages: u32) -> Rc<RefCell<WasmtimeBackend<CpuState>>> {
        Rc::new(RefCell::new(WasmtimeBackend::new_with_memory_pages(
            pages, 0x100,
        )))
    }

    #[test]
    fn writes_land_in_the_backends_linear_memory() {
        let b = backend(4);
        let mut ram = WasmGuestRam::new(b.clone(), 0x1_0000).expect("ram fits");

        ram.write_from(0x40, &[0xde, 0xad, 0xbe, 0xef]).unwrap();

        // Read it back the way compiled code would: straight out of linear memory
        // at RAM_BASE + paddr.
        let (ptr, _) = unsafe { b.borrow_mut().linear_memory_ptr() };
        let seen = unsafe { std::slice::from_raw_parts(ptr.add(RAM_BASE as usize + 0x40), 4) };
        assert_eq!(seen, &[0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn reads_see_what_linear_memory_holds() {
        let b = backend(4);
        let ram = WasmGuestRam::new(b.clone(), 0x1_0000).expect("ram fits");

        // Write as compiled code would, read through the guest-memory interface.
        {
            let (ptr, _) = unsafe { b.borrow_mut().linear_memory_ptr() };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    [1u8, 2, 3, 4].as_ptr(),
                    ptr.add(RAM_BASE as usize + 0x80),
                    4,
                );
            }
        }

        let mut buf = [0u8; 4];
        ram.read_into(0x80, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
        assert_eq!(ram.get_slice(0x80, 4).unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn accesses_past_the_end_are_rejected_not_wrapped() {
        let b = backend(4);
        let ram = WasmGuestRam::new(b, 0x1_0000).expect("ram fits");

        let mut buf = [0u8; 8];
        assert!(ram.read_into(0x1_0000 - 4, &mut buf).is_err());
        assert!(ram.read_into(u64::MAX - 2, &mut buf).is_err());
        assert!(ram.get_slice(0x1_0000, 1).is_none());
        // The last valid byte is still reachable.
        assert!(ram.get_slice(0x1_0000 - 1, 1).is_some());
    }

    #[test]
    fn ram_larger_than_the_backing_memory_is_refused() {
        let b = backend(2);
        assert!(WasmGuestRam::new(b, 0x10_0000).is_none());
    }
}

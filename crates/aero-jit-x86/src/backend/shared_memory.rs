//! One guest memory, mapped into more than one Wasmtime engine.
//!
//! Wasmtime picks its compiler per [`Engine`](wasmtime::Engine), and a `Store` can
//! only import a memory its own engine created. That is a problem for a tier-up:
//! Winch admits a block for ~144us and Cranelift produces code a quarter cheaper to
//! run, and the useful arrangement is both at once — Winch for everything, Cranelift
//! for the few hundred blocks that account for most of the running.
//!
//! The way through is [`wasmtime::MemoryCreator`]: each engine still creates its own
//! `Memory` object, but both are handed a view of the *same* pages, so guest RAM,
//! `CpuState` and the JIT context are shared and a chain can cross between them
//! without copying anything.
//!
//! The mapping is made here rather than left to Wasmtime because of the guard
//! region. Winch requires signals-based traps — it refuses to start without them —
//! which means the bytes after the memory have to be unmapped so an out-of-range
//! access faults rather than landing in whatever follows. A plain allocation cannot
//! offer that.

use std::sync::Arc;

use wasmtime::{LinearMemory, MemoryCreator, MemoryType};

/// The mapped region: `len` accessible bytes followed by `guard` unmapped ones.
#[derive(Debug)]
struct Mapping {
    base: *mut u8,
    len: usize,
    guard: usize,
}

// Safety: the pointer is a private mapping owned by this struct for its whole life.
// Wasmtime requires `MemoryCreator: Send + Sync`; the pages are shared deliberately,
// and every writer of guest memory runs under the machine's single CPU thread.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        // Safety: `base`/`len`/`guard` are what `mmap` returned and was given.
        unsafe {
            libc::munmap(self.base.cast(), self.len + self.guard);
        }
    }
}

/// A guest memory that several engines can map.
///
/// Clone it and hand a clone to each engine's
/// [`Config::with_host_memory`](wasmtime::Config::with_host_memory).
#[derive(Clone, Debug)]
pub struct SharedGuestMemory {
    map: Arc<Mapping>,
}

impl SharedGuestMemory {
    /// Map `bytes` accessible bytes plus `guard` unmapped bytes after them.
    ///
    /// # Panics
    /// If the mapping cannot be made.
    #[must_use]
    pub fn new(bytes: usize, guard: usize) -> Self {
        let total = bytes
            .checked_add(guard)
            .expect("guest memory size overflows");
        // Reserve the whole span unreadable, then open up the part the guest may
        // touch. What is left is the guard: an access past the end faults instead of
        // reaching whatever the allocator put there.
        //
        // Safety: a fresh anonymous mapping with no fixed address.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(
            base != libc::MAP_FAILED,
            "could not reserve {total} bytes for guest memory: {}",
            std::io::Error::last_os_error()
        );
        // Safety: `base..base + bytes` is inside the reservation just made.
        let rc = unsafe { libc::mprotect(base, bytes, libc::PROT_READ | libc::PROT_WRITE) };
        assert!(
            rc == 0,
            "could not make {bytes} bytes of guest memory writable: {}",
            std::io::Error::last_os_error()
        );
        // Guest RAM is gigabytes and every compiled memory access goes through it,
        // so how the host maps it shows up twice: once as page faults while it is
        // touched for the first time, and again as host TLB pressure for the whole
        // run. Ask for huge pages — 2MiB instead of 4KiB is 512 times fewer of both.
        // Advisory: if transparent huge pages are off, this does nothing and the
        // mapping still works.
        //
        // Safety: the range is the mapping just made.
        if std::env::var_os("AERO_JIT_NO_HUGEPAGES").is_none() {
            let _ = unsafe { libc::madvise(base, total, libc::MADV_HUGEPAGE) };
        }
        Self {
            map: Arc::new(Mapping {
                base: base.cast::<u8>(),
                len: bytes,
                guard,
            }),
        }
    }

    /// Base of the accessible region.
    #[must_use]
    pub fn base(&self) -> *mut u8 {
        self.map.base
    }

    /// How many bytes the guest may touch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.len == 0
    }
}

/// A view Wasmtime holds. Several of these alias one [`SharedGuestMemory`], which is
/// the entire point; none of them owns the pages.
struct SharedView {
    map: Arc<Mapping>,
    len: usize,
}

// Safety: as for `Mapping`.
unsafe impl Send for SharedView {}
unsafe impl Sync for SharedView {}

// Safety: `as_ptr` returns the base of a mapping that outlives this view — the
// `Arc` holds it — and never moves, because the memory is created with min == max
// and so is never grown.
unsafe impl LinearMemory for SharedView {
    fn byte_size(&self) -> usize {
        self.len
    }

    fn byte_capacity(&self) -> usize {
        self.map.len
    }

    fn grow_to(&mut self, new_size: usize) -> wasmtime::Result<()> {
        if new_size > self.map.len {
            anyhow_bail(new_size, self.map.len)
        } else {
            // Growth within the reservation needs nothing: the pages are already
            // there and the base does not move.
            self.len = new_size;
            Ok(())
        }
    }

    fn as_ptr(&self) -> *mut u8 {
        self.map.base
    }
}

fn anyhow_bail(want: usize, have: usize) -> wasmtime::Result<()> {
    Err(wasmtime::Error::msg(format!(
        "guest memory cannot grow to {want} bytes; the mapping is {have}"
    )))
}

// Safety: every view hands back the same mapping, deliberately, so that two engines
// see one guest memory. Nothing else may write those pages while WASM is running,
// which holds because the machine drives its CPUs from a single thread.
unsafe impl MemoryCreator for SharedGuestMemory {
    fn new_memory(
        &self,
        _ty: MemoryType,
        minimum: usize,
        _maximum: Option<usize>,
        _reserved_size_in_bytes: Option<usize>,
        _guard_size_in_bytes: usize,
    ) -> Result<Box<dyn LinearMemory>, String> {
        if minimum > self.map.len {
            return Err(format!(
                "asked for {minimum} bytes of guest memory; the mapping is {}",
                self.map.len
            ));
        }
        Ok(Box::new(SharedView {
            map: self.map.clone(),
            len: self.map.len,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_views_of_one_mapping_see_each_others_writes() {
        let mem = SharedGuestMemory::new(64 * 1024, 4096);
        let a = mem
            .new_memory(MemoryType::new(1, Some(1)), 64 * 1024, None, None, 0)
            .expect("first view");
        let b = mem
            .new_memory(MemoryType::new(1, Some(1)), 64 * 1024, None, None, 0)
            .expect("second view");
        // Safety: both views are the same mapping, which is what is being asserted.
        unsafe {
            *a.as_ptr().add(1234) = 0xab;
            assert_eq!(*b.as_ptr().add(1234), 0xab);
            *b.as_ptr().add(4321) = 0xcd;
            assert_eq!(*a.as_ptr().add(4321), 0xcd);
        }
        assert_eq!(a.byte_size(), 64 * 1024);
    }

    #[test]
    fn the_mapping_is_zeroed_to_begin_with() {
        let mem = SharedGuestMemory::new(64 * 1024, 4096);
        // Safety: inside the accessible region.
        unsafe {
            for off in [0usize, 1, 4095, 4096, 65535] {
                assert_eq!(*mem.base().add(off), 0, "byte {off} was not zero");
            }
        }
    }
}

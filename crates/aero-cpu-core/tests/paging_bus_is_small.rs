//! `PagingBus` must stay small enough to move freely.
//!
//! `Machine` builds one of these on every outer-loop batch boundary and hands it its
//! durable MMU. An `Mmu` is ~16.5KB, almost all of it software TLB, so a `PagingBus`
//! that stores one inline makes every boundary a multi-kilobyte `memmove` — and the
//! placeholder MMU such a design needs to swap against costs six more, because
//! `Mmu::new()` builds its TLB arrays from a template.
//!
//! Measured before the MMU was boxed: 66KB copied per boundary, 2.5GB across a 120M
//! guest-instruction window, ~3.4 host instructions per guest instruction. Nothing
//! about that is visible from the source, which is why it stood for so long — hence
//! this test rather than a comment.
//!
//! ```text
//! cargo test --release -p aero-cpu-core --test paging_bus_is_small
//! ```

use aero_cpu_core::PagingBus;

/// Generous: the point is to catch an inline TLB, not to pin the exact layout.
const MAX_BYTES: usize = 512;

#[test]
fn paging_bus_does_not_store_the_mmu_inline() {
    let bytes = std::mem::size_of::<PagingBus<()>>();
    assert!(
        bytes <= MAX_BYTES,
        "PagingBus is {bytes} bytes (limit {MAX_BYTES}). An Mmu is \
         {} bytes; if it has been un-boxed, every batch boundary now copies it.",
        std::mem::size_of::<aero_mmu::Mmu>()
    );
}

/// The premise of the test above: the MMU really is far too big to move per batch.
#[test]
fn an_mmu_is_much_larger_than_the_bus_that_uses_it() {
    assert!(
        std::mem::size_of::<aero_mmu::Mmu>() > 4 * MAX_BYTES,
        "if the MMU ever became small, boxing it would stop being worth the indirection"
    );
}

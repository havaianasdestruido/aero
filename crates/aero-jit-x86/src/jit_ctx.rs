//! JIT-side context shared with generated WASM blocks/traces.
//!
//! Tier-1 blocks receive a separate `jit_ctx_ptr` parameter pointing at a [`JitContext`] instance
//! stored in linear memory. Tier-2 traces currently only receive a `cpu_ptr` pointing at the
//! architectural CPU state, so Tier-2 metadata is stored at a fixed offset relative to `cpu_ptr`.

use crate::{abi, JIT_TLB_ENTRIES, JIT_TLB_ENTRY_SIZE};

/// Header for the Tier-1 JIT context.
///
/// The context is stored in linear memory at `jit_ctx_ptr` (WASM i32 byte offset). All integer
/// fields are little-endian.
///
/// Layout (bytes):
/// - `ram_base` (`u64`): base offset of guest RAM within linear memory.
/// - `tlb_salt` (`u64`): tag salt used by the direct-mapped JIT TLB.
/// - `retired` (`u64`): guest instructions the block ran before it left.
/// - `chain_retired` (`u64`): guest instructions retired by chain steps taken inside
///   wasm, accumulated by the dispatcher across its internal loop.
/// - `chain_budget` (`u64`): the dispatcher stops looping once `chain_retired`
///   reaches this, so the machine's instruction budget still bounds a chain that
///   never returns to the host between blocks.
/// - `chain_blocks` (`u32`): how many blocks the dispatcher ran internally.
/// - `chain_max_blocks` (`u32`): and the cap on that.
/// - `tlb[]` (`JIT_TLB_ENTRIES` × 16 bytes): `{ tag: u64, data: u64 }`
/// - `chain_map[]` (`JIT_CHAIN_MAP_ENTRIES` × 16 bytes):
///   `{ rip: u64, module_id: u32, index: u32 }`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct JitContext {
    pub ram_base: u64,
    pub tlb_salt: u64,
    /// Written by the block, read by the runtime; see [`Self::RETIRED_OFFSET`].
    pub retired: u64,
    /// Written by the dispatcher, read by the runtime; see [`Self::CHAIN_RETIRED_OFFSET`].
    pub chain_retired: u64,
    /// Written by the runtime, read by the dispatcher.
    pub chain_budget: u64,
    /// Written by the dispatcher, read by the runtime.
    pub chain_blocks: u32,
    /// Written by the runtime, read by the dispatcher.
    pub chain_max_blocks: u32,
}

/// Entries in the rip-to-block map the dispatcher chains through.
///
/// Direct-mapped and deliberately small: it is consulted once per chain step from
/// inside wasm, so it wants to stay in cache, and a miss is not a fault — it simply
/// returns to the host, which does the authoritative lookup.
///
/// **2048, and this is not a dial.** Swept with the promoted-only default, which uses
/// ~240 entries whatever the table size and so should barely care: 512 gives 120.4,
/// 1024 gives 258.3, 2048 gives **102.5**, 8192 gives 628.2, and 32768 outgrows the
/// region reserved for the context in linear memory and traps out of bounds.
///
/// Jagged in both directions, which is the same signature every size-like parameter in
/// this tier has: the context lives in guest linear memory, so changing its size moves
/// what follows it, and the guest ends up on a different path. The 8192 figure is at
/// least consistent across four windows (628–638) and its cost is host-side rather than
/// in generated code, but 512 and 1024 being *also* worse, and unequally, says most of
/// what is being measured here is layout and not capacity.
///
/// The practical conclusions are the two that matter: the map cannot usefully be grown,
/// so admission-time chaining cannot be de-thrashed that way; and this constant should
/// be left alone. See `wiki/areas/performance.md`.
pub const JIT_CHAIN_MAP_ENTRIES: usize = 2048;

/// `{ rip: u64, module_id: u32, index: u32 }`.
///
/// It was widened to 32 bytes once, to carry an execution count so the dispatcher
/// could report hot blocks itself and admission-time chaining would stop hiding them
/// from promotion. That worked — 32.2% of block entries chained against 20.1%, and
/// promotions rose from 237 to 265 — and the run still collapsed to 498.3 against
/// 128.9, because putting all 36,832 admitted blocks through a 2,048-entry
/// direct-mapped map thrashes it, and the promoted entries that matter are evicted by
/// cold ones. The wider entry cost 1.6% on its own even with the feature off.
pub const JIT_CHAIN_MAP_ENTRY_SIZE: u32 = 16;

impl JitContext {
    pub const RAM_BASE_OFFSET: u32 = 0;
    pub const TLB_SALT_OFFSET: u32 = Self::RAM_BASE_OFFSET + 8;

    /// How many guest instructions the block retired before leaving.
    ///
    /// A block used to retire a fixed count — its own length — so the runtime could
    /// assume it. A block with side exits cannot: which exit was taken decides how
    /// much of it ran, and the machine's timing follows that number. The block writes
    /// it at whichever exit it leaves through.
    pub const RETIRED_OFFSET: u32 = Self::TLB_SALT_OFFSET + 8;

    /// Guest instructions retired by chain steps the dispatcher took without
    /// returning to the host.
    ///
    /// The block itself still writes [`Self::RETIRED_OFFSET`] with its own count and
    /// knows nothing about chaining; the dispatcher adds that into this after each
    /// call. Keeping the two separate is what lets in-wasm chaining be added without
    /// changing a single line of block codegen.
    pub const CHAIN_RETIRED_OFFSET: u32 = Self::RETIRED_OFFSET + 8;
    pub const CHAIN_BUDGET_OFFSET: u32 = Self::CHAIN_RETIRED_OFFSET + 8;
    pub const CHAIN_BLOCKS_OFFSET: u32 = Self::CHAIN_BUDGET_OFFSET + 8;
    pub const CHAIN_MAX_BLOCKS_OFFSET: u32 = Self::CHAIN_BLOCKS_OFFSET + 4;
    pub const TLB_OFFSET: u32 = Self::CHAIN_MAX_BLOCKS_OFFSET + 4;

    /// Size of the fixed header in bytes.
    pub const BYTE_SIZE: usize = Self::TLB_OFFSET as usize;

    pub const TLB_BYTES: usize = JIT_TLB_ENTRIES * (JIT_TLB_ENTRY_SIZE as usize);

    /// The chain map sits after the TLB, so adding it leaves both the header offsets
    /// the blocks were compiled against and the TLB indexing untouched.
    pub const CHAIN_MAP_OFFSET: u32 = Self::TLB_OFFSET + (Self::TLB_BYTES as u32);
    pub const CHAIN_MAP_BYTES: usize =
        JIT_CHAIN_MAP_ENTRIES * (JIT_CHAIN_MAP_ENTRY_SIZE as usize);
    pub const TOTAL_BYTE_SIZE: usize = Self::BYTE_SIZE + Self::TLB_BYTES + Self::CHAIN_MAP_BYTES;

    /// Slot a RIP hashes to in the chain map.
    ///
    /// The low four bits of a RIP carry almost no information — blocks start at
    /// instruction boundaries — so they are folded in from higher up rather than used
    /// directly. The dispatcher emits exactly this, so the two must change together.
    #[must_use]
    pub const fn chain_map_slot(rip: u64) -> usize {
        ((rip ^ (rip >> 4) ^ (rip >> 13)) as usize) & (JIT_CHAIN_MAP_ENTRIES - 1)
    }

    /// Writes just the header fields (`ram_base`, `tlb_salt`) into `mem[base..]`.
    ///
    /// The TLB array is left untouched (typically zeroed on allocation).
    #[track_caller]
    pub fn write_header_to_mem(&self, mem: &mut [u8], base: usize) {
        let end = match base.checked_add(Self::BYTE_SIZE) {
            Some(end) => end,
            None => panic!(
                "JitContext write out of bounds: base={base} size={} mem_len={} (overflow)",
                Self::BYTE_SIZE,
                mem.len()
            ),
        };
        assert!(
            end <= mem.len(),
            "JitContext write out of bounds: base={base} size={} mem_len={}",
            Self::BYTE_SIZE,
            mem.len()
        );

        let ram_base_off = match base.checked_add(Self::RAM_BASE_OFFSET as usize) {
            Some(off) => off,
            None => panic!(
                "JitContext ram_base offset overflow: base={base} off={} mem_len={}",
                Self::RAM_BASE_OFFSET,
                mem.len()
            ),
        };
        mem[ram_base_off..ram_base_off + 8].copy_from_slice(&self.ram_base.to_le_bytes());

        let tlb_salt_off = match base.checked_add(Self::TLB_SALT_OFFSET as usize) {
            Some(off) => off,
            None => panic!(
                "JitContext tlb_salt offset overflow: base={base} off={} mem_len={}",
                Self::TLB_SALT_OFFSET,
                mem.len()
            ),
        };
        mem[tlb_salt_off..tlb_salt_off + 8].copy_from_slice(&self.tlb_salt.to_le_bytes());
    }
}

const _: () = {
    use core::mem::{offset_of, size_of};

    assert!(JitContext::TOTAL_BYTE_SIZE <= u32::MAX as usize);
    assert!(offset_of!(JitContext, ram_base) == JitContext::RAM_BASE_OFFSET as usize);
    assert!(offset_of!(JitContext, tlb_salt) == JitContext::TLB_SALT_OFFSET as usize);
    assert!(size_of::<JitContext>() == JitContext::BYTE_SIZE);
    assert!(JitContext::TLB_OFFSET as usize == JitContext::BYTE_SIZE);
    assert!(JitContext::TLB_BYTES == JIT_TLB_ENTRIES * (JIT_TLB_ENTRY_SIZE as usize));
    assert!(
        JitContext::TOTAL_BYTE_SIZE
            == JitContext::BYTE_SIZE + JitContext::TLB_BYTES + JitContext::CHAIN_MAP_BYTES
    );
    assert!(JIT_CHAIN_MAP_ENTRY_SIZE.is_power_of_two());
    assert!(JitContext::CHAIN_MAP_OFFSET as usize == JitContext::BYTE_SIZE + JitContext::TLB_BYTES);
    // The dispatcher masks a hash rather than dividing, so the table has to be a
    // power of two, and it indexes with a shift by 4.
    assert!(JIT_CHAIN_MAP_ENTRIES.is_power_of_two());
    assert!(offset_of!(JitContext, chain_retired) == JitContext::CHAIN_RETIRED_OFFSET as usize);
    assert!(offset_of!(JitContext, chain_budget) == JitContext::CHAIN_BUDGET_OFFSET as usize);
    assert!(offset_of!(JitContext, chain_blocks) == JitContext::CHAIN_BLOCKS_OFFSET as usize);
    assert!(
        offset_of!(JitContext, chain_max_blocks) == JitContext::CHAIN_MAX_BLOCKS_OFFSET as usize
    );
};

/// Offset (relative to `cpu_ptr`) of the Tier-2 context region.
pub const TIER2_CTX_OFFSET: u32 = abi::CPU_STATE_SIZE + (JitContext::TOTAL_BYTE_SIZE as u32);

/// Offset of the Tier-2 trace exit reason (`u32`).
pub const TRACE_EXIT_REASON_OFFSET: u32 = TIER2_CTX_OFFSET;

/// The trace exited normally (no special handling required).
pub const TRACE_EXIT_REASON_NONE: u32 = 0;

/// The trace exited because a code page version guard failed.
///
/// Runtimes are expected to invalidate the cached Tier-2 trace and resume execution in the
/// interpreter/Tier-1 at the returned RIP.
pub const TRACE_EXIT_REASON_CODE_INVALIDATION: u32 = 1;

/// Offset of a pointer (`u32`, byte offset) to the page-version table.
///
/// The table is a dense `u32[]` indexed by 4KiB physical page number (`paddr >> 12`), and entries
/// are treated as modulo-2^32 counters (wrapping on overflow).
pub const CODE_VERSION_TABLE_PTR_OFFSET: u32 = TIER2_CTX_OFFSET + 4;

/// Offset of the length (`u32`, number of `u32` entries) of the page-version table.
pub const CODE_VERSION_TABLE_LEN_OFFSET: u32 = TIER2_CTX_OFFSET + 8;

/// Total size (in bytes) of the Tier-2 context region.
pub const TIER2_CTX_SIZE: u32 = 12;

/// Backwards-compatible alias for [`TIER2_CTX_SIZE`].
pub const JIT_CTX_SIZE: u32 = TIER2_CTX_SIZE;

const _: () = {
    // Keep the Tier-2 context layout stable and internally consistent. The JS host and tiered VM
    // rely on these offsets when allocating and inspecting the shared JIT ABI buffer.
    assert!(TIER2_CTX_OFFSET == abi::CPU_STATE_SIZE + (JitContext::TOTAL_BYTE_SIZE as u32));
    assert!(TRACE_EXIT_REASON_OFFSET == TIER2_CTX_OFFSET);
    assert!(CODE_VERSION_TABLE_PTR_OFFSET == TIER2_CTX_OFFSET + 4);
    assert!(CODE_VERSION_TABLE_LEN_OFFSET == TIER2_CTX_OFFSET + 8);
    assert!(TIER2_CTX_SIZE == 12);
    assert!(JIT_CTX_SIZE == TIER2_CTX_SIZE);
    assert!(TIER2_CTX_OFFSET.is_multiple_of(4));
    assert!(TIER2_CTX_SIZE.is_multiple_of(4));
};

#[cfg(test)]
mod tests {
    use memoffset::offset_of;

    use super::JitContext;
    use crate::test_util::capture_panic_location;

    #[test]
    fn jit_context_layout_matches_constants() {
        assert_eq!(
            offset_of!(JitContext, ram_base) as u32,
            JitContext::RAM_BASE_OFFSET
        );
        assert_eq!(
            offset_of!(JitContext, tlb_salt) as u32,
            JitContext::TLB_SALT_OFFSET
        );

        assert_eq!(core::mem::size_of::<JitContext>(), JitContext::BYTE_SIZE);
        assert_eq!(JitContext::TLB_OFFSET as usize, JitContext::BYTE_SIZE);

        assert_eq!(
            JitContext::TLB_BYTES,
            crate::JIT_TLB_ENTRIES * (crate::JIT_TLB_ENTRY_SIZE as usize)
        );
        assert_eq!(
            JitContext::TOTAL_BYTE_SIZE,
            JitContext::BYTE_SIZE + JitContext::TLB_BYTES + JitContext::CHAIN_MAP_BYTES
        );
        assert_eq!(
            JitContext::CHAIN_MAP_OFFSET as usize,
            JitContext::BYTE_SIZE + JitContext::TLB_BYTES
        );
        assert_eq!(
            offset_of!(JitContext, chain_retired) as u32,
            JitContext::CHAIN_RETIRED_OFFSET
        );
        assert_eq!(
            offset_of!(JitContext, chain_budget) as u32,
            JitContext::CHAIN_BUDGET_OFFSET
        );
        assert_eq!(
            offset_of!(JitContext, chain_blocks) as u32,
            JitContext::CHAIN_BLOCKS_OFFSET
        );
        assert_eq!(
            offset_of!(JitContext, chain_max_blocks) as u32,
            JitContext::CHAIN_MAX_BLOCKS_OFFSET
        );
        // The dispatcher masks rather than divides, and scales the slot by a shift.
        assert!(super::JIT_CHAIN_MAP_ENTRIES.is_power_of_two());
        assert!(super::JIT_CHAIN_MAP_ENTRY_SIZE.is_power_of_two());
    }

    #[test]
    fn jit_context_write_header_writes_little_endian_fields() {
        let ctx = JitContext {
            ram_base: 0x1122_3344_5566_7788,
            tlb_salt: 0x99aa_bbcc_ddee_ff00,
            // Not part of the header: the block writes it, the runtime reads it.
            retired: 0,
            chain_retired: 0,
            chain_budget: 0,
            chain_blocks: 0,
            chain_max_blocks: 0,
        };

        let mut mem = [0u8; 64];
        let base = 7usize;
        ctx.write_header_to_mem(&mut mem, base);

        let ram_base_off = base + (JitContext::RAM_BASE_OFFSET as usize);
        assert_eq!(
            &mem[ram_base_off..ram_base_off + 8],
            &ctx.ram_base.to_le_bytes()
        );

        let salt_off = base + (JitContext::TLB_SALT_OFFSET as usize);
        assert_eq!(&mem[salt_off..salt_off + 8], &ctx.tlb_salt.to_le_bytes());
    }

    #[test]
    fn jit_context_write_header_panics_at_call_site_on_oob() {
        let ctx = JitContext::default();
        let mut mem = [0u8; 0];

        let expected_file = file!();
        let expected_line = line!() + 2;
        let (file, line) = capture_panic_location(|| {
            ctx.write_header_to_mem(&mut mem, 0);
        });
        assert_eq!(file, expected_file);
        assert_eq!(line, expected_line);
    }

    #[test]
    fn jit_context_write_header_panics_at_call_site_on_overflow() {
        let ctx = JitContext::default();
        let mut mem = [0u8; 0];

        let expected_file = file!();
        let expected_line = line!() + 2;
        let (file, line) = capture_panic_location(|| {
            ctx.write_header_to_mem(&mut mem, usize::MAX);
        });
        assert_eq!(file, expected_file);
        assert_eq!(line, expected_line);
    }
}

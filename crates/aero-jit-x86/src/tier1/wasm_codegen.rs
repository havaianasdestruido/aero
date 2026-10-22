use aero_types::{Cond, Flag, FlagSet, Gpr, Width};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemArg, MemoryType, Module, TypeSection, ValType,
};

use super::ir::{BinOp, GuestReg, IrBlock, IrInst, IrTerminator, ValueId};
use crate::abi;
use crate::abi::{MMU_ACCESS_READ, MMU_ACCESS_WRITE};
use crate::jit_ctx::{self, JitContext};
use crate::wasm::inline_tlb_codegen::{self, InlineTlbLocals};

use crate::wasm::abi::{
    IMPORT_JIT_EXIT, IMPORT_JIT_EXIT_MMIO, IMPORT_MEMORY, IMPORT_MEM_READ_U16, IMPORT_MEM_READ_U32,
    IMPORT_ASSIST, IMPORT_RDTSC,
    IMPORT_MEM_READ_U64, IMPORT_MEM_READ_U8, IMPORT_MEM_WRITE_U16, IMPORT_MEM_WRITE_U32,
    IMPORT_MEM_WRITE_U64, IMPORT_MEM_WRITE_U8, IMPORT_MMU_TRANSLATE, IMPORT_MODULE,
    JIT_EXIT_SENTINEL_I64, WASM32_MAX_PAGES,
};

/// WASM export name for Tier-1 blocks.
pub const EXPORT_BLOCK_FN: &str = crate::wasm::abi::EXPORT_BLOCK_FN;

/// Backwards-compatible alias for [`EXPORT_BLOCK_FN`].
pub const EXPORT_TIER1_BLOCK_FN: &str = EXPORT_BLOCK_FN;

/// Name of the single exported entry point when [`Tier1WasmOptions::dispatcher`] is set.
///
/// Exporting each block separately costs ~80us *per block* in `Module::new`, because
/// a function that can be named from outside — exported, or merely present in a
/// table — needs a host-callable trampoline compiled for it. That was the largest
/// part of admitting a block. A function only ever reached by a direct `call` needs
/// none, so the module exports one dispatcher that `br_table`s to a direct call per
/// block: 64 blocks a module fell from 91.6us to 18.6us each, and 256 to 13.9us.
pub const EXPORT_TIER1_DISPATCH_FN: &str = "dispatch";

#[derive(Debug, Clone, Copy)]
pub struct Tier1WasmOptions {
    /// Enable the inline direct-mapped JIT TLB + direct guest RAM fast-path for same-page loads.
    ///
    /// Note: this option is ignored unless the crate feature `tier1-inline-tlb` is enabled.
    pub inline_tlb: bool,

    /// When [`Self::inline_tlb`] is enabled, allow stores to use the same inline-TLB fast-path as
    /// loads.
    ///
    /// When disabled, Tier-1 stores always go through the imported slow helpers (`env.mem_write_*`)
    /// so the host runtime can observe writes (MMIO classification, self-modifying code
    /// invalidation via `jit.on_guest_write(..)`, etc).
    pub inline_tlb_stores: bool,

    /// When [`Self::inline_tlb`] is enabled, control how same-page non-RAM translations are
    /// handled.
    ///
    /// - When `true` (default), Tier-1 treats these accesses as MMIO and calls the imported
    ///   `env.jit_exit_mmio` helper, causing the backend to roll back guest state and resume via
    ///   the interpreter.
    /// - When `false`, Tier-1 falls back to the imported slow mem helpers (`env.mem_read_*` /
    ///   `env.mem_write_*`) and continues executing the block.
    pub inline_tlb_mmio_exit: bool,

    /// When [`Self::inline_tlb`] is enabled, allow cross-page (4KiB boundary-crossing) loads/stores
    /// to use an inline-TLB RAM fast-path instead of immediately falling back to the imported slow
    /// helpers (`env.mem_read_*` / `env.mem_write_*`).
    ///
    /// When enabled and an access crosses a page boundary, the code generator emits a split access
    /// that translates both pages and performs direct guest-RAM loads/stores for each page.
    ///
    /// Note: this option is ignored unless the crate feature `tier1-inline-tlb` is enabled.
    pub inline_tlb_cross_page_fastpath: bool,

    /// Emit one inline fast path per memory access, and one call for everything else.
    ///
    /// The default emission inlines a path per case — cross-page, non-RAM, MMIO exit,
    /// TLB hit — nested three deep with the same-page sequence duplicated inside two
    /// of them. One `mov eax,[rsi]` costs about 128 wasm operations that way, against
    /// five to ten host instructions for the equivalent soft-MMU hit in a mature
    /// emulator. The slow helper already handles misses, page crossings and device
    /// memory correctly, so the fast path can reduce to a single predicate — TLB hit,
    /// backed by RAM, inside one page — with everything else a call.
    pub single_fast_path: bool,
    /// Emit the load fast path once per module and call it, instead of inlining it
    /// at every access. See `inline_tlb_codegen::emit_shared_load_helper`.
    pub shared_mem_helpers: bool,
    /// Export one `dispatch(cpu_ptr, jit_ctx_ptr, index)` entry point that calls the
    /// blocks directly, instead of exporting every block. See
    /// [`EXPORT_TIER1_DISPATCH_FN`].
    pub dispatcher: bool,
    /// Emit a dispatcher that chains between this module's own blocks without
    /// returning to the host, and stamp it with this module's id.
    ///
    /// A chain step costs the host a `TypedFunc::call` into the engine — the
    /// trampoline, the trap-handler thread state, and the lookup between blocks —
    /// and a block retires about four guest instructions, so that cost is divided by
    /// almost nothing. When the successor is a block of the *same* module the
    /// dispatcher can simply call it, and the whole crossing disappears. The module
    /// id is compared against the chain map's entry so a stale RIP can never be
    /// dispatched into the wrong module's function index.
    ///
    /// Requires [`Tier1WasmOptions::dispatcher`].
    pub chain_module_id: Option<u32>,

    /// Whether the imported `env.memory` is expected to be a shared memory (i.e. created with
    /// `WebAssembly.Memory({ shared: true, ... })`).
    ///
    /// Note: shared memories require a declared maximum page count.
    pub memory_shared: bool,

    /// Minimum size (in 64KiB pages) of the imported `env.memory`.
    pub memory_min_pages: u32,

    /// Maximum size (in 64KiB pages) of the imported `env.memory`.
    ///
    /// When [`Tier1WasmOptions::memory_shared`] is `true` and this is unset, the code generator
    /// defaults to 65536 pages (4GiB) so the module can accept any smaller shared memory.
    pub memory_max_pages: Option<u32>,
}

impl Default for Tier1WasmOptions {
    fn default() -> Self {
        Self {
            inline_tlb: false,
            // Preserve historical behaviour: when callers enable `inline_tlb`, stores take the
            // fast-path by default unless explicitly disabled.
            inline_tlb_stores: true,
            // Preserve existing behaviour by default: MMIO/non-RAM access forces a runtime exit.
            inline_tlb_mmio_exit: true,
            // Preserve historical behaviour: cross-page accesses always used the slow helpers.
            inline_tlb_cross_page_fastpath: false,
            single_fast_path: false,
            shared_mem_helpers: false,
            dispatcher: false,
            chain_module_id: None,
            // Preserve existing behaviour by default: import an unshared memory with min=1 and no
            // maximum.
            memory_shared: false,
            memory_min_pages: 1,
            memory_max_pages: None,
        }
    }
}

impl Tier1WasmOptions {
    #[track_caller]
    fn validate_memory_import(self) {
        let effective_max_pages = if self.memory_shared {
            Some(self.memory_max_pages.unwrap_or(WASM32_MAX_PAGES))
        } else {
            self.memory_max_pages
        };

        assert!(
            self.memory_min_pages <= WASM32_MAX_PAGES,
            "invalid env.memory import type: min_pages ({}) exceeds wasm32 max pages ({})",
            self.memory_min_pages,
            WASM32_MAX_PAGES
        );
        if let Some(max) = effective_max_pages {
            assert!(
                max <= WASM32_MAX_PAGES,
                "invalid env.memory import type: max_pages ({}) exceeds wasm32 max pages ({})",
                max,
                WASM32_MAX_PAGES
            );
            assert!(
                self.memory_min_pages <= max,
                "invalid env.memory import type: min_pages ({}) > max_pages ({})",
                self.memory_min_pages,
                max
            );
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ImportedFuncs {
    mem_read_u8: Option<u32>,
    mem_read_u16: Option<u32>,
    mem_read_u32: Option<u32>,
    mem_read_u64: Option<u32>,
    mem_write_u8: Option<u32>,
    mem_write_u16: Option<u32>,
    mem_write_u32: Option<u32>,
    mem_write_u64: Option<u32>,
    mmu_translate: Option<u32>,
    jit_exit_mmio: Option<u32>,
    jit_exit: Option<u32>,
    rdtsc: Option<u32>,
    assist: Option<u32>,
    count: u32,
}

/// Indices of the per-module shared load helpers, by width, where emitted.
#[derive(Debug, Clone, Copy, Default)]
struct SharedLoadHelpers {
    load: [Option<u32>; 4],
}

impl SharedLoadHelpers {
    fn get(self, width: Width) -> Option<u32> {
        self.load[match width {
            Width::W8 => 0,
            Width::W16 => 1,
            Width::W32 => 2,
            Width::W64 => 3,
        }]
    }
}

/// What a compiled module needs from the host that runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tier1ModuleNeeds {
    /// The module can call `env.mem_read_*` / `env.mem_write_*`, which address the
    /// imported linear memory with an untranslated guest address.
    pub slow_mem: bool,
}

pub struct Tier1WasmCodegen;

#[derive(Debug, Clone, Copy, Default)]
struct BlockStateUsage {
    /// Whether a GPR's *initial* value must be loaded from `CpuState` at block entry.
    ///
    /// This is required when the block reads the GPR, or when the block performs a partial write
    /// (8/16/high8) that needs the previous 64-bit value.
    gpr_used: [bool; 16],
    /// Whether a GPR may be written by the block and therefore must be spilled back to `CpuState`
    /// at block exit.
    gpr_written: [bool; 16],
    /// Whether the block's RIP local must be initialized from `CpuState.rip` at block entry.
    ///
    /// Tier-1 blocks always write back `next_rip` at exit, but loading the *current* RIP is only
    /// needed when the IR reads it directly or when Tier-1 needs to pass it to runtime exit
    /// helpers (MMIO exits / `CallHelper` bailouts).
    rip_used: bool,
    /// Whether the block reads and/or writes RFLAGS.
    uses_rflags: bool,
    /// Whether the block writes RFLAGS and therefore must spill `CpuState.rflags` at exit.
    rflags_written: bool,
    /// First guest instruction from which the block could leave part-way, if any.
    ///
    /// Before it, nothing can bail, so the running retired count is unobservable and
    /// need not be maintained — which matters, because maintaining it at every
    /// instruction measured at about 5% of the machine.
    first_exiting_guest_inst: Option<u64>,
    /// Whether the block reads a flag it did not produce itself.
    ///
    /// Flags are handed between blocks as the operation that produced them rather
    /// than as computed bits, so `CpuState.rflags` alone is not enough for a block
    /// that reads one on arrival: the previous block may have left the work owed.
    /// Such a block settles the incoming descriptor once, on entry.
    reads_incoming_flags: bool,
}

fn analyze_state_usage(block: &IrBlock, options: Tier1WasmOptions) -> BlockStateUsage {
    let mut usage = BlockStateUsage::default();
    // Set once the block has produced flags of its own; a read after that point is
    // satisfied from what it produced, not from what it was handed.
    let mut produced_flags = false;
    let mut initialized = [false; 16];
    let mut first_write_idx: [Option<usize>; 16] = [None; 16];
    let mut first_rflags_write_idx: Option<usize> = None;
    let mut rip_initialized = false;

    let mut earliest_may_exit: Option<usize> = None;
    // Guest instructions are delimited by the RIP write `translate` emits at the
    // start of each.
    let mut guest_idx = 0u64;

    for (i, inst) in block.insts.iter().enumerate() {
        if matches!(
            inst,
            IrInst::WriteReg {
                reg: GuestReg::Rip,
                ..
            }
        ) {
            guest_idx += 1;
        }
        // A side exit leaves part-way through, exactly as a bail-out does: the
        // running retired count has to be maintained from there on, and a register
        // written after it has to be loaded so the spill does not invent a zero.
        if matches!(inst, IrInst::SideExit { .. }) {
            usage.first_exiting_guest_inst = usage
                .first_exiting_guest_inst
                .or(Some(guest_idx.saturating_sub(1)));
            earliest_may_exit = earliest_may_exit.or(Some(i));
        }
        // An assist may report a fault, or land somewhere the rest of the block was
        // not decoded for; either way the block leaves from there.
        if matches!(inst, IrInst::Assist { .. }) {
            usage.first_exiting_guest_inst = usage
                .first_exiting_guest_inst
                .or(Some(guest_idx.saturating_sub(1)));
            earliest_may_exit = earliest_may_exit.or(Some(i));
            // The host may write any register, so every one of them is both read and
            // written from here on.
            for k in 0..16 {
                usage.gpr_used[k] = true;
                usage.gpr_written[k] = true;
            }
            usage.uses_rflags = true;
            usage.rflags_written = true;
            usage.rip_used = true;
        }
        // Conservative: helper calls always bail out.
        if matches!(inst, IrInst::CallHelper { .. }) {
            usage.first_exiting_guest_inst = usage
                .first_exiting_guest_inst
                .or(Some(guest_idx.saturating_sub(1)));
            earliest_may_exit = earliest_may_exit.or(Some(i));
        } else if options.inline_tlb && options.inline_tlb_mmio_exit {
            // Inline-TLB fast paths can exit early on MMIO when configured to do so.
            let is_mmio_exit_point = match inst {
                IrInst::Load { .. } => true,
                IrInst::Store { .. } => options.inline_tlb_stores,
                _ => false,
            };
            if is_mmio_exit_point {
                earliest_may_exit = earliest_may_exit.or(Some(i));
                usage.first_exiting_guest_inst = usage
                    .first_exiting_guest_inst
                    .or(Some(guest_idx.saturating_sub(1)));
            }
        }

        match inst {
            IrInst::ReadReg { reg, .. } => match *reg {
                GuestReg::Gpr { reg, .. } => {
                    let idx = reg.as_u8() as usize;
                    if !initialized[idx] {
                        usage.gpr_used[idx] = true;
                        initialized[idx] = true;
                    }
                }
                GuestReg::Flag(_) => usage.uses_rflags = true,
                // Loaded straight from `CpuState` at the point of use; no local, so
                // nothing to mark as used.
                GuestReg::SegBase(_) => {}
                GuestReg::Rip => {
                    if !rip_initialized {
                        usage.rip_used = true;
                        rip_initialized = true;
                    }
                }
            },
            IrInst::WriteReg { reg, .. } => match *reg {
                GuestReg::SegBase(_) => unreachable!("segment bases are read-only in Tier-1"),
                GuestReg::Gpr { reg, width, high8 } => {
                    let idx = reg.as_u8() as usize;
                    usage.gpr_written[idx] = true;
                    first_write_idx[idx].get_or_insert(i);

                    let is_partial = matches!(width, Width::W8 | Width::W16) || high8;
                    if is_partial && !initialized[idx] {
                        // Partial writes need the previous 64-bit value.
                        usage.gpr_used[idx] = true;
                        initialized[idx] = true;
                    }

                    // Any write (full or partial) initializes the local for subsequent accesses.
                    initialized[idx] = true;
                }
                GuestReg::Flag(_) => {
                    usage.uses_rflags = true;
                    usage.rflags_written = true;
                    first_rflags_write_idx = first_rflags_write_idx.or(Some(i));
                    // A partial write leaves the other five as they arrived.
                    if !produced_flags {
                        usage.reads_incoming_flags = true;
                    }
                }
                GuestReg::Rip => rip_initialized = true,
            },
            IrInst::BinOp { op, flags, .. } => {
                // `adc`/`sbb` consume the incoming carry before producing anything.
                if matches!(op, BinOp::Adc | BinOp::Sbb) && !produced_flags {
                    usage.uses_rflags = true;
                    usage.reads_incoming_flags = true;
                }
                if !flags.is_empty() {
                    usage.uses_rflags = true;
                    usage.rflags_written = true;
                    first_rflags_write_idx = first_rflags_write_idx.or(Some(i));
                    if *flags == FlagSet::ALU {
                        produced_flags = true;
                    } else if !produced_flags {
                        // A partial write keeps the rest of what arrived.
                        usage.reads_incoming_flags = true;
                    }
                }
            }
            IrInst::CmpFlags { flags, .. } | IrInst::TestFlags { flags, .. } => {
                if !flags.is_empty() {
                    // A deferred ALU flag set never reaches the RFLAGS local. Marking
                    // RFLAGS used here is still right — the epilogue needs to know
                    // flags were touched — but it must not by itself pull the
                    // prologue's *load* in; see `reads_incoming_flags` at the load.
                    //
                    // This once measured as nothing (102.5 / 86.7 / 99.0 / 90.2 against
                    // 102.5 / 86.7 / 99.1 / 90.2) and was written up as "a dead load
                    // does not pay because Cranelift deletes it". That reasoning was
                    // wrong: a wasm load can fault into the guard region, so it is a
                    // trapping operation Cranelift has to keep, and the disassembly
                    // shows it surviving. It measured as nothing because a block was
                    // ~64 native instructions then.
                    usage.uses_rflags = true;
                    usage.rflags_written = true;
                    first_rflags_write_idx = first_rflags_write_idx.or(Some(i));
                    if *flags == FlagSet::ALU {
                        produced_flags = true;
                    } else if !produced_flags {
                        usage.reads_incoming_flags = true;
                    }
                }
            }
            IrInst::EvalCond { cond, .. } => {
                if !cond.uses_flags().is_empty() {
                    usage.uses_rflags = true;
                    if !produced_flags {
                        usage.reads_incoming_flags = true;
                    }
                }
            }
            IrInst::CallHelper { .. } => {
                // Helper-call bailouts pass the current RIP to the host.
                if !rip_initialized {
                    usage.rip_used = true;
                }
                // Tier-1 treats helper calls as a runtime exit; the remainder of the IR block is
                // unreachable (both in the IR interpreter and in Tier-1 WASM codegen).
                break;
            }
            IrInst::Load { .. } => {
                // Inline-TLB loads may take an MMIO exit and must pass the current RIP.
                if options.inline_tlb && options.inline_tlb_mmio_exit && !rip_initialized {
                    usage.rip_used = true;
                    rip_initialized = true;
                }
            }
            IrInst::Store { .. } => {
                // Inline-TLB stores may take an MMIO exit (when store fast-path is enabled) and
                // must pass the current RIP.
                if options.inline_tlb
                    && options.inline_tlb_mmio_exit
                    && options.inline_tlb_stores
                    && !rip_initialized
                {
                    usage.rip_used = true;
                    rip_initialized = true;
                }
            }
            _ => {}
        }
    }

    // If the block can exit early (MMIO fast path or CallHelper bailout) before a GPR's first
    // write, we still need to load that GPR if we're going to spill it at function exit; otherwise
    // the epilogue store would clobber the register with the local's default value (0).
    if let Some(exit_idx) = earliest_may_exit {
        for gpr in all_gprs() {
            let idx = gpr.as_u8() as usize;
            if !usage.gpr_written[idx] || usage.gpr_used[idx] {
                continue;
            }
            if let Some(first_write) = first_write_idx[idx] {
                if exit_idx < first_write {
                    usage.gpr_used[idx] = true;
                }
            }
        }
    }


    // A block that can leave before its end must load every register it will spill.
    //
    // "Load what you read, spill what you write" is only sound if the whole block
    // runs. A register written near the end has no load — nothing read it — so its
    // local is zero until that write happens, and an exit before then spills the zero
    // over the guest's value. Side exits make this constant; a mid-block bail-out
    // always could, which is why this is a fix and not a new requirement.
    if let Some(exit_at) = earliest_may_exit {
        // Only a register whose first write comes *after* the earliest exit needs
        // it. One written before then already holds a real value by the time any
        // exit can happen, so spilling its local is correct — and loading it would
        // be pure cost, which matters most for the long blocks that write the most.
        for i in 0..16 {
            if first_write_idx[i].is_some_and(|w| w > exit_at) {
                usage.gpr_used[i] = true;
            }
        }
        if first_rflags_write_idx.is_some_and(|w| w > exit_at) {
            usage.uses_rflags = true;
        }
    }

    // A block that settles an incoming record writes the flags into the RFLAGS local
    // and consumes the record, so the local has to reach `CpuState`. A block that
    // only reads leaves the record alone and needs no spill — which is exactly the
    // case worth being cheap.
    if usage.reads_incoming_flags && usage.rflags_written {
        usage.rflags_written = true;
    }

    usage
}

impl Tier1WasmCodegen {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    #[track_caller]
    pub fn compile_block(&self, block: &IrBlock) -> Vec<u8> {
        self.compile_block_with_options(block, Tier1WasmOptions::default())
    }

    #[must_use]
    #[track_caller]
    pub fn compile_block_with_options(
        &self,
        block: &IrBlock,
        options: Tier1WasmOptions,
    ) -> Vec<u8> {
        self.compile_blocks_with_options(&[block], options)
    }

    /// Compile several blocks into a single module.
    ///
    /// Admitting a block costs about 577us, essentially all of it `Module::new`,
    /// and that is a cost per *module* rather than per instruction compiled — a
    /// 663-byte module costs as much as a large one. One module per block therefore
    /// makes a block worth roughly 6,000 interpreted guest instructions to admit,
    /// which a four-instruction block only repays after ~1,700 executions. Sharing
    /// a module across many blocks divides that fixed cost between them.
    ///
    /// The exports are `block` for the first and `block1`, `block2`, ... for the
    /// rest, so a single-block module is byte-identical to what
    /// [`Self::compile_block_with_options`] produced before this existed.
    ///
    /// Import requirements are unioned across the blocks: they share one import
    /// section, so the call indices baked into each body must agree with it.
    #[must_use]
    #[track_caller]
    pub fn compile_blocks_with_options(
        &self,
        blocks: &[&IrBlock],
        options: Tier1WasmOptions,
    ) -> Vec<u8> {
        self.compile_blocks_reporting(blocks, options).0
    }

    /// As [`Self::compile_blocks_with_options`], but also reporting what the module
    /// needs from its host.
    ///
    /// The slow memory helpers (`env.mem_read_*` / `env.mem_write_*`) take a guest
    /// address and index the imported linear memory with it directly — an identity
    /// mapping. That is right for an embedder that does not page and wrong for one
    /// that does, and it fails loudly only when the address happens to be out of
    /// range: a kernel address panics, a low one silently reads the wrong memory.
    /// An embedder with real page tables therefore has to know whether a module can
    /// reach that path, which is what this reports.
    #[must_use]
    #[track_caller]
    pub fn compile_blocks_reporting(
        &self,
        blocks: &[&IrBlock],
        options: Tier1WasmOptions,
    ) -> (Vec<u8>, Tier1ModuleNeeds) {
        assert!(!blocks.is_empty(), "a module needs at least one block");
        let mut options = options;
        #[cfg(not(feature = "tier1-inline-tlb"))]
        {
            options.inline_tlb = false;
        }

        let mut uses_load_u8 = false;
        let mut uses_load_u16 = false;
        let mut uses_load_u32 = false;
        let mut uses_load_u64 = false;
        let mut uses_store_u8 = false;
        let mut uses_store_u16 = false;
        let mut uses_store_u32 = false;
        let mut uses_store_u64 = false;
        let mut needs_jit_exit = false;
        let mut uses_rdtsc = false;
        let mut uses_assist = false;
        let mut uses_inline_tlb = false;
        for block in blocks.iter().copied() {
        for inst in &block.insts {
            match inst {
                IrInst::ReadTsc { .. } => {
                    uses_rdtsc = true;
                }
                IrInst::Assist { .. } => {
                    uses_assist = true;
                }
                IrInst::Load { width, .. } => {
                    uses_inline_tlb = true;
                    match *width {
                        Width::W8 => uses_load_u8 = true,
                        Width::W16 => uses_load_u16 = true,
                        Width::W32 => uses_load_u32 = true,
                        Width::W64 => uses_load_u64 = true,
                    }
                }
                IrInst::Store { width, .. } => {
                    if options.inline_tlb_stores {
                        uses_inline_tlb = true;
                    }
                    match *width {
                        Width::W8 => uses_store_u8 = true,
                        Width::W16 => uses_store_u16 = true,
                        Width::W32 => uses_store_u32 = true,
                        Width::W64 => uses_store_u64 = true,
                    }
                }
                IrInst::CallHelper { .. } => {
                    needs_jit_exit = true;
                    // Tier-1 treats helper calls as an unconditional runtime exit; the remainder
                    // of the IR block is unreachable (both in the IR interpreter and in Tier-1
                    // WASM codegen).
                    break;
                }
                _ => {}
            }
        }
        }

        // Enabling the inline-TLB fast-path only matters if the block contains a memory operation
        // eligible for it.
        if options.inline_tlb && !uses_inline_tlb {
            options.inline_tlb = false;
        }

        let mut may_cross_page_load_u16 = false;
        let mut may_cross_page_load_u32 = false;
        let mut may_cross_page_load_u64 = false;
        let mut may_cross_page_store_u16 = false;
        let mut may_cross_page_store_u32 = false;
        let mut may_cross_page_store_u64 = false;

        // When MMIO exits are enabled and cross-page fast-paths are disabled, cross-page accesses
        // fall back to the slow helpers. We can elide these helper imports if we can prove an
        // access will never cross a page boundary, which requires a small constant-propagation
        // pass over the IR. Avoid paying that cost for blocks that never perform a multi-byte
        // memory access via the inline-TLB fast-path.
        let needs_cross_page_analysis = uses_load_u16
            || uses_load_u32
            || uses_load_u64
            || (options.inline_tlb_stores && (uses_store_u16 || uses_store_u32 || uses_store_u64));
        // With one fast path per access the helper is the fallback for *every* case,
        // not only a page crossing, so it can never be elided.
        if options.inline_tlb
            && options.inline_tlb_mmio_exit
            && !options.inline_tlb_cross_page_fastpath
            && !options.single_fast_path
            && needs_cross_page_analysis
        {
            for block in blocks.iter().copied() {
            let mut const_values: Vec<Option<u64>> = vec![None; block.value_types.len()];
            for inst in &block.insts {
                match inst {
                    IrInst::Const { dst, value, width } => {
                        const_values[dst.0 as usize] = Some(width.truncate(*value));
                    }
                    IrInst::Trunc { dst, src, width } => {
                        if let Some(v) = const_values[src.0 as usize] {
                            const_values[dst.0 as usize] = Some(width.truncate(v));
                        }
                    }
                    IrInst::BinOp {
                        dst,
                        op,
                        lhs,
                        rhs,
                        width,
                        ..
                    } => {
                        let (Some(lhs), Some(rhs)) =
                            (const_values[lhs.0 as usize], const_values[rhs.0 as usize])
                        else {
                            continue;
                        };

                        let mask = width.mask();
                        let trunc = |v: u64| v & mask;
                        let shift_mask: u64 = if *width == Width::W64 { 63 } else { 31 };

                        let res = match *op {
                            BinOp::Add => lhs.wrapping_add(rhs),
                            BinOp::Adc | BinOp::Sbb => continue,
                            BinOp::Sub => lhs.wrapping_sub(rhs),
                            BinOp::And => lhs & rhs,
                            BinOp::Or => lhs | rhs,
                            BinOp::Xor => lhs ^ rhs,
                            BinOp::Shl => {
                                let amt = (rhs & shift_mask) as u32;
                                lhs.wrapping_shl(amt)
                            }
                            BinOp::Shr => {
                                let amt = (rhs & shift_mask) as u32;
                                lhs.wrapping_shr(amt)
                            }
                            BinOp::Sar => {
                                let lhs_trunc = lhs & mask;
                                let sign_bit = 1u64 << (width.bits() - 1);
                                let lhs_sext = if (lhs_trunc & sign_bit) != 0 {
                                    lhs_trunc | !mask
                                } else {
                                    lhs_trunc
                                };
                                let amt = (rhs & shift_mask) as u32;
                                (lhs_sext as i64).wrapping_shr(amt) as u64
                            }
                            BinOp::Rol => {
                                let bits = width.bits();
                                let amt = ((rhs & shift_mask) as u32) % bits;
                                let lhs_trunc = lhs & mask;
                                if amt == 0 {
                                    lhs_trunc
                                } else {
                                    ((lhs_trunc << amt) | (lhs_trunc >> (bits - amt))) & mask
                                }
                            }
                            BinOp::Ror => {
                                let bits = width.bits();
                                let amt = ((rhs & shift_mask) as u32) % bits;
                                let lhs_trunc = lhs & mask;
                                if amt == 0 {
                                    lhs_trunc
                                } else {
                                    ((lhs_trunc >> amt) | (lhs_trunc << (bits - amt))) & mask
                                }
                            }
                            BinOp::Bswap => match *width {
                                Width::W64 => lhs.swap_bytes(),
                                Width::W32 => (lhs as u32).swap_bytes() as u64,
                                Width::W16 => (lhs as u16).swap_bytes() as u64,
                                Width::W8 => lhs & 0xff,
                            },
                            BinOp::Mul => lhs.wrapping_mul(rhs),
                        };
                        const_values[dst.0 as usize] = Some(trunc(res));
                    }
                    IrInst::CallHelper { .. } => break,
                    _ => {}
                }
            }

            let may_cross_page = |addr: ValueId, size_bytes: u32| -> bool {
                if size_bytes <= 1 {
                    return false;
                }
                let cross_limit =
                    crate::PAGE_OFFSET_MASK.saturating_sub(size_bytes.saturating_sub(1) as u64);
                let addr = const_values
                    .get(addr.0 as usize)
                    .copied()
                    .flatten()
                    .unwrap_or(u64::MAX);
                (addr & crate::PAGE_OFFSET_MASK) > cross_limit
            };

            for inst in &block.insts {
                match *inst {
                    IrInst::Load { addr, width, .. } => match width {
                        Width::W8 => {}
                        Width::W16 => may_cross_page_load_u16 |= may_cross_page(addr, 2),
                        Width::W32 => may_cross_page_load_u32 |= may_cross_page(addr, 4),
                        Width::W64 => may_cross_page_load_u64 |= may_cross_page(addr, 8),
                    },
                    IrInst::Store { addr, width, .. } => match width {
                        Width::W8 => {}
                        Width::W16 => may_cross_page_store_u16 |= may_cross_page(addr, 2),
                        Width::W32 => may_cross_page_store_u32 |= may_cross_page(addr, 4),
                        Width::W64 => may_cross_page_store_u64 |= may_cross_page(addr, 8),
                    },
                    IrInst::CallHelper { .. } => break,
                    _ => {}
                }
            }
            }
        }

        // Decide which slow-path helper imports are actually required by this block under the
        // selected options.
        //
        // When the inline-TLB fast-path is enabled, some slow helpers become unused:
        // - if `inline_tlb_mmio_exit` is enabled, non-RAM translations trigger an MMIO exit rather
        //   than falling back to the slow helpers.
        // - if `inline_tlb_cross_page_fastpath` is enabled, boundary-crossing loads/stores are
        //   handled inline for widths that can cross pages (>= 16-bit).
        // With one fast path per access, the helper is the fallback for every case, so
        // every width that is loaded or stored needs its import.
        let single = options.single_fast_path && options.inline_tlb;
        let needs_mem_read_u8 =
            uses_load_u8 && (single || !options.inline_tlb || !options.inline_tlb_mmio_exit);
        let needs_mem_read_u16 = uses_load_u16
            && (single
                || !options.inline_tlb
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_load_u16));
        let needs_mem_read_u32 = uses_load_u32
            && (single
                || !options.inline_tlb
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_load_u32));
        let needs_mem_read_u64 = uses_load_u64
            && (single
                || !options.inline_tlb
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_load_u64));

        let single_store = single && store_fastpath_enabled();
        let needs_mem_write_u8 = uses_store_u8
            && (single_store
                || !options.inline_tlb
                || !options.inline_tlb_stores
                || !options.inline_tlb_mmio_exit);
        let needs_mem_write_u16 = uses_store_u16
            && (single_store
                || !options.inline_tlb
                || !options.inline_tlb_stores
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_store_u16));
        let needs_mem_write_u32 = uses_store_u32
            && (single_store
                || !options.inline_tlb
                || !options.inline_tlb_stores
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_store_u32));
        let needs_mem_write_u64 = uses_store_u64
            && (single_store
                || !options.inline_tlb
                || !options.inline_tlb_stores
                || !options.inline_tlb_mmio_exit
                || (!options.inline_tlb_cross_page_fastpath && may_cross_page_store_u64));

        // A shared load helper only pays for itself if the module actually loads,
        // and it needs the slow-read import for its own miss path.
        let shared_ok = options.shared_mem_helpers && options.single_fast_path && options.inline_tlb;
        let shared_widths = [
            shared_ok && uses_load_u8,
            shared_ok && uses_load_u16,
            shared_ok && uses_load_u32,
            shared_ok && uses_load_u64,
        ];
        let any_shared = shared_widths.iter().any(|&b| b);

        let mut module = Module::new();

        let mut types = TypeSection::new();
        // Keep the type section minimal: only emit helper import types that are actually used by
        // this block.
        //
        // Note: some helpers share identical signatures (e.g. `mem_read_u8/u16/u32` all return
        // i32). Reuse types to avoid emitting duplicates.
        let needs_mem_read_i32 = needs_mem_read_u8 || needs_mem_read_u16 || needs_mem_read_u32;
        let needs_mem_write_i32 = needs_mem_write_u8 || needs_mem_write_u16 || needs_mem_write_u32;
        let needs_i32_i64_to_i64 = needs_mem_read_u64 || needs_jit_exit || uses_assist;

        let ty_rdtsc = uses_rdtsc.then(|| {
            let ty = types.len();
            types.ty().function([ValType::I32], [ValType::I64]);
            ty
        });
        let ty_dispatch = options.dispatcher.then(|| {
            let ty = types.len();
            types
                .ty()
                .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I64]);
            ty
        });
        let ty_shared_load = any_shared.then(|| {
            let ty = types.len();
            types.ty().function(
                [
                    ValType::I32,
                    ValType::I32,
                    ValType::I64,
                    ValType::I64,
                    ValType::I64,
                ],
                [ValType::I64],
            );
            ty
        });
        let ty_i32_i64_to_i32 = needs_mem_read_i32.then(|| {
            let ty = types.len();
            types
                .ty()
                .function([ValType::I32, ValType::I64], [ValType::I32]);
            ty
        });
        let ty_i32_i64_to_i64 = needs_i32_i64_to_i64.then(|| {
            let ty = types.len();
            types
                .ty()
                .function([ValType::I32, ValType::I64], [ValType::I64]);
            ty
        });
        let ty_i32_i64_i32_to_empty = needs_mem_write_i32.then(|| {
            let ty = types.len();
            types
                .ty()
                .function([ValType::I32, ValType::I64, ValType::I32], []);
            ty
        });
        let ty_i32_i64_i64_to_empty = needs_mem_write_u64.then(|| {
            let ty = types.len();
            types
                .ty()
                .function([ValType::I32, ValType::I64, ValType::I64], []);
            ty
        });
        let ty_mmu_translate = if options.inline_tlb {
            let ty = types.len();
            types.ty().function(
                [ValType::I32, ValType::I32, ValType::I64, ValType::I32],
                [ValType::I64],
            );
            Some(ty)
        } else {
            None
        };
        let ty_jit_exit_mmio = if options.inline_tlb && options.inline_tlb_mmio_exit {
            let ty = types.len();
            types.ty().function(
                [
                    ValType::I32,
                    ValType::I64,
                    ValType::I32,
                    ValType::I32,
                    ValType::I64,
                    ValType::I64,
                ],
                [ValType::I64],
            );
            Some(ty)
        } else {
            None
        };
        let ty_jit_exit = ty_i32_i64_to_i64;
        let ty_block = types.len();
        types
            .ty()
            .function([ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);

        options.validate_memory_import();
        let mut imports = ImportSection::new();
        let memory_max_pages: Option<u64> = if options.memory_shared {
            // Shared memories require an explicit maximum. Default to 4GiB (the maximum size of a
            // wasm32 memory) so we can link against any smaller shared memory.
            Some(u64::from(
                options.memory_max_pages.unwrap_or(WASM32_MAX_PAGES),
            ))
        } else {
            options.memory_max_pages.map(u64::from)
        };
        imports.import(
            IMPORT_MODULE,
            IMPORT_MEMORY,
            MemoryType {
                minimum: u64::from(options.memory_min_pages),
                maximum: memory_max_pages,
                memory64: false,
                shared: options.memory_shared,
                page_size_log2: None,
            },
        );

        let func_base = 0u32;
        let mut next_func = func_base;
        let imported = ImportedFuncs {
            mem_read_u8: needs_mem_read_u8.then(|| next(&mut next_func)),
            mem_read_u16: needs_mem_read_u16.then(|| next(&mut next_func)),
            mem_read_u32: needs_mem_read_u32.then(|| next(&mut next_func)),
            mem_read_u64: needs_mem_read_u64.then(|| next(&mut next_func)),
            mem_write_u8: needs_mem_write_u8.then(|| next(&mut next_func)),
            mem_write_u16: needs_mem_write_u16.then(|| next(&mut next_func)),
            mem_write_u32: needs_mem_write_u32.then(|| next(&mut next_func)),
            mem_write_u64: needs_mem_write_u64.then(|| next(&mut next_func)),
            mmu_translate: options.inline_tlb.then(|| next(&mut next_func)),
            jit_exit_mmio: (options.inline_tlb && options.inline_tlb_mmio_exit)
                .then(|| next(&mut next_func)),
            jit_exit: needs_jit_exit.then(|| next(&mut next_func)),
            rdtsc: uses_rdtsc.then(|| next(&mut next_func)),
            assist: uses_assist.then(|| next(&mut next_func)),
            count: next_func - func_base,
        };

        if needs_mem_read_u8 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U8,
                EntityType::Function(ty_i32_i64_to_i32.expect("type for mem_read_u8")),
            );
        }
        if needs_mem_read_u16 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U16,
                EntityType::Function(ty_i32_i64_to_i32.expect("type for mem_read_u16")),
            );
        }
        if needs_mem_read_u32 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U32,
                EntityType::Function(ty_i32_i64_to_i32.expect("type for mem_read_u32")),
            );
        }
        if needs_mem_read_u64 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U64,
                EntityType::Function(ty_i32_i64_to_i64.expect("type for mem_read_u64")),
            );
        }
        if needs_mem_write_u8 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U8,
                EntityType::Function(ty_i32_i64_i32_to_empty.expect("type for mem_write_u8")),
            );
        }
        if needs_mem_write_u16 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U16,
                EntityType::Function(ty_i32_i64_i32_to_empty.expect("type for mem_write_u16")),
            );
        }
        if needs_mem_write_u32 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U32,
                EntityType::Function(ty_i32_i64_i32_to_empty.expect("type for mem_write_u32")),
            );
        }
        if needs_mem_write_u64 {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U64,
                EntityType::Function(ty_i32_i64_i64_to_empty.expect("type for mem_write_u64")),
            );
        }
        if options.inline_tlb {
            imports.import(
                IMPORT_MODULE,
                IMPORT_MMU_TRANSLATE,
                EntityType::Function(ty_mmu_translate.expect("type for mmu_translate")),
            );
        }
        if options.inline_tlb && options.inline_tlb_mmio_exit {
            imports.import(
                IMPORT_MODULE,
                IMPORT_JIT_EXIT_MMIO,
                EntityType::Function(ty_jit_exit_mmio.expect("type for jit_exit_mmio")),
            );
        }
        if needs_jit_exit {
            imports.import(
                IMPORT_MODULE,
                IMPORT_JIT_EXIT,
                EntityType::Function(ty_jit_exit.expect("type for jit_exit")),
            );
        }
        if uses_rdtsc {
            imports.import(
                IMPORT_MODULE,
                IMPORT_RDTSC,
                EntityType::Function(ty_rdtsc.expect("type for rdtsc")),
            );
        }
        if uses_assist {
            imports.import(
                IMPORT_MODULE,
                IMPORT_ASSIST,
                EntityType::Function(ty_i32_i64_to_i64.expect("type for assist")),
            );
        }
        module.section(&imports);

        let mut funcs = FunctionSection::new();
        for _ in blocks {
            funcs.function(ty_block);
        }
        // Helpers come after the blocks so the exported block indices are unchanged.
        let shared_base = imported.count + blocks.len() as u32;
        let mut shared_loads = SharedLoadHelpers::default();
        let mut next_shared = shared_base;
        for (i, &wanted) in shared_widths.iter().enumerate() {
            if wanted {
                funcs.function(ty_shared_load.expect("type for shared load helper"));
                shared_loads.load[i] = Some(next_shared);
                next_shared += 1;
            }
        }
        if options.dispatcher {
            funcs.function(ty_dispatch.expect("type for dispatcher"));
            // The chaining entry point is a second function of the same type that
            // calls the first in a loop, so the `br_table` stays exactly as it was.
            if options.chain_module_id.is_some() {
                funcs.function(ty_dispatch.expect("type for dispatcher"));
            }
        }
        module.section(&funcs);

        let mut exports = ExportSection::new();
        if options.dispatcher {
            exports.export(
                EXPORT_TIER1_DISPATCH_FN,
                ExportKind::Func,
                if options.chain_module_id.is_some() {
                    next_shared + 1
                } else {
                    next_shared
                },
            );
        }
        for (i, _) in blocks.iter().enumerate() {
            if options.dispatcher {
                break;
            }
            // The first keeps the historical name so a single-block module is
            // byte-identical to what this produced before it could hold several.
            let name = if i == 0 {
                EXPORT_BLOCK_FN.to_string()
            } else {
                format!("{EXPORT_BLOCK_FN}{i}")
            };
            exports.export(&name, ExportKind::Func, imported.count + i as u32);
        }
        module.section(&exports);

        let mut code = CodeSection::new();
        for block in blocks.iter().copied() {
            let func = Self::emit_block_function(
                block,
                options,
                imported,
                shared_loads,
                uses_store_u8,
                uses_store_u16,
                uses_store_u32,
                uses_store_u64,
                needs_jit_exit,
            );
            code.function(&func);
        }
        for (i, &wanted) in shared_widths.iter().enumerate() {
            if !wanted {
                continue;
            }
            let size_bytes = 1u32 << i;
            let slow_read = match size_bytes {
                1 => imported.mem_read_u8,
                2 => imported.mem_read_u16,
                4 => imported.mem_read_u32,
                _ => imported.mem_read_u64,
            }
            .expect("slow read import missing for shared load helper");
            // Three scratch locals beyond the five parameters.
            let mut helper = Function::new(vec![(3, ValType::I64)]);
            inline_tlb_codegen::emit_shared_load_helper(
                &mut helper,
                imported.mmu_translate.expect("mmu_translate import missing"),
                slow_read,
                size_bytes,
            );
            code.function(&helper);
        }
        if options.dispatcher {
            code.function(&Self::emit_dispatcher(imported.count, blocks.len() as u32));
            if let Some(module_id) = options.chain_module_id {
                code.function(&Self::emit_chaining_dispatcher(
                    next_shared,
                    module_id,
                    imported.count,
                    blocks.len() as u32,
                ));
            }
        }
        module.section(&code);

        let needs = Tier1ModuleNeeds {
            slow_mem: needs_mem_read_u8
                || needs_mem_read_u16
                || needs_mem_read_u32
                || needs_mem_read_u64
                || needs_mem_write_u8
                || needs_mem_write_u16
                || needs_mem_write_u32
                || needs_mem_write_u64,
        };
        (module.finish(), needs)
    }

    /// The module's single exported entry point.
    ///
    /// `dispatch(cpu_ptr, jit_ctx_ptr, index)` branches to a direct call of block
    /// `index`. Direct calls are what keeps the blocks off the trampoline path — see
    /// [`EXPORT_TIER1_DISPATCH_FN`]. An out-of-range index returns the exit sentinel
    /// rather than trapping, so a stale table index is a bail rather than a crash.
    fn emit_dispatcher(first_block_fn: u32, count: u32) -> Function {
        let mut f = Function::new(vec![]);
        // One label per block, plus one for the out-of-range default.
        for _ in 0..=count {
            f.instruction(&Instruction::Block(BlockType::Empty));
        }
        f.instruction(&Instruction::LocalGet(2));
        let targets: Vec<u32> = (0..count).collect();
        f.instruction(&Instruction::BrTable(targets.into(), count));
        for i in 0..count {
            // Branching to label `i` lands here, just past that block's `end`.
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::LocalGet(1));
            f.instruction(&Instruction::Call(first_block_fn + i));
            f.instruction(&Instruction::Return);
        }
        f.instruction(&Instruction::End);
        f.instruction(&Instruction::I64Const(crate::wasm::JIT_EXIT_SENTINEL_I64));
        f.instruction(&Instruction::End);
        f
    }

    /// The chaining entry point: run blocks of this module back to back.
    ///
    /// Calls [`Self::emit_dispatcher`] in a loop, and between calls does what the
    /// host chain loop would have done — accumulate the retired count, honour the
    /// instruction budget and the block cap, and look the next RIP up. The lookup is
    /// a direct-mapped probe of the chain map in the JIT context; a miss, or a hit
    /// belonging to another module, returns to the host, which holds the real block
    /// map and can dispatch anywhere.
    ///
    /// Nothing here is required for correctness: returning immediately after the
    /// first block is exactly the old behaviour, which is why the map is allowed to
    /// be lossy. What it removes is the engine crossing per block — the trampoline
    /// and the trap-handler thread state — which a four-instruction block cannot
    /// amortise.
    fn emit_chaining_dispatcher(
        dispatch_one: u32,
        module_id: u32,
        first_block_fn: u32,
        count: u32,
    ) -> Function {
        const CPU: u32 = 0;
        const CTX: u32 = 1;
        const IDX: u32 = 2;
        const RET: u32 = 3;
        // Loop-carried accumulators and the two limits they are tested against. These
        // lived in the context and were read-modify-written every step: two stores and
        // six loads a block, for values only this function touches until it returns.
        //
        // **Kept for being strictly less work, not for a measured win.** 49.50 and 49.51
        // against a 49.81-49.83 baseline in the 140-200M window — reproducible to 0.01 —
        // but 58.90 against 58.86 deep, which is nothing, and deep is the instrument that
        // decides. The browser is inside its noise band too (429.8, band 424.8-448.6).
        //
        // It also settles an anomaly worth not re-chasing: adding *one* store per block
        // elsewhere measured 31% worse, which suggested stores here were ruinous. Removing
        // two and four loads buys 0.6%, so they are not. Whatever that 31% was, it was
        // specific to that change and not a property of memory traffic on this path.
        const ACC: u32 = 4;
        const BUD: u32 = 5;
        const ADDR: u32 = 6;
        const CNT: u32 = 7;
        const MAXB: u32 = 8;
        // Depth 0 is the loop, depth 1 the block that wraps it: `br 0` goes round
        // again, `br_if 1` leaves with the last block's return value in `RET`.
        const AGAIN: u32 = 0;
        const OUT: u32 = 1;

        // Two wasm calls a step — this, then the block — and inlining the `br_table`
        // here to make it one measured **300 host instructions a block against 282**.
        // Removing a call made it slower, presumably because the loop body grows and
        // is scheduled worse. That is the fourth restructuring of emitted wasm in this
        // area to come out neutral or negative; Cranelift is already doing better with
        // this than hand-rearranging it does.
        let mut f = Function::new(vec![(3, ValType::I64), (3, ValType::I32)]);
        // Pull the counters and their limits into locals once. The host zeroes the two
        // counters before every call and reads them after, so starting from what is in
        // memory and writing back on the way out is exactly the old contract.
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I64Load(memarg(JitContext::CHAIN_RETIRED_OFFSET, 3)));
        f.instruction(&Instruction::LocalSet(ACC));
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I64Load(memarg(JitContext::CHAIN_BUDGET_OFFSET, 3)));
        f.instruction(&Instruction::LocalSet(BUD));
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I32Load(memarg(JitContext::CHAIN_BLOCKS_OFFSET, 2)));
        f.instruction(&Instruction::LocalSet(CNT));
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I32Load(memarg(JitContext::CHAIN_MAX_BLOCKS_OFFSET, 2)));
        f.instruction(&Instruction::LocalSet(MAXB));
        f.instruction(&Instruction::Block(BlockType::Empty));
        f.instruction(&Instruction::Loop(BlockType::Empty));

        if inline_dispatch_enabled() {
            // The same `br_table` `emit_dispatcher` builds, but written into the loop
            // so a chain step is one wasm call rather than two. The intermediate
            // function is not free: it carries its own frame, its own stack-limit
            // check and a call/return, which is 28 native instructions of a ~94
            // instruction step.
            //
            // Each arm leaves the block's result on the stack and branches to a
            // result-typed block wrapping the whole `br_table` nest. After the `i`-th
            // `End` there are `count - i` blocks still open inside it, which is the
            // branch depth.
            f.instruction(&Instruction::Block(BlockType::Result(ValType::I64)));
            for _ in 0..=count {
                f.instruction(&Instruction::Block(BlockType::Empty));
            }
            f.instruction(&Instruction::LocalGet(IDX));
            let targets: Vec<u32> = (0..count).collect();
            f.instruction(&Instruction::BrTable(targets.into(), count));
            for i in 0..count {
                f.instruction(&Instruction::End);
                f.instruction(&Instruction::LocalGet(CPU));
                f.instruction(&Instruction::LocalGet(CTX));
                f.instruction(&Instruction::Call(first_block_fn + i));
                f.instruction(&Instruction::Br(count - i));
            }
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::I64Const(crate::wasm::JIT_EXIT_SENTINEL_I64));
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::LocalSet(RET));
        } else {
            f.instruction(&Instruction::LocalGet(CPU));
            f.instruction(&Instruction::LocalGet(CTX));
            f.instruction(&Instruction::LocalGet(IDX));
            f.instruction(&Instruction::Call(dispatch_one));
            f.instruction(&Instruction::LocalSet(RET));
        }

        // The block wrote its own retired count, as it always has. Fold it in here so
        // block codegen needs no knowledge of chaining — into a local, not back into the
        // context.
        f.instruction(&Instruction::LocalGet(ACC));
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I64Load(memarg(JitContext::RETIRED_OFFSET, 3)));
        f.instruction(&Instruction::I64Add);
        f.instruction(&Instruction::LocalSet(ACC));

        f.instruction(&Instruction::LocalGet(CNT));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(CNT));

        // A bail, a fault or an exit to the interpreter: the host has to see it.
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::I64Const(crate::wasm::JIT_EXIT_SENTINEL_I64));
        f.instruction(&Instruction::I64Eq);
        f.instruction(&Instruction::BrIf(OUT));

        f.instruction(&Instruction::LocalGet(ACC));
        f.instruction(&Instruction::LocalGet(BUD));
        f.instruction(&Instruction::I64GeU);
        f.instruction(&Instruction::BrIf(OUT));

        f.instruction(&Instruction::LocalGet(CNT));
        f.instruction(&Instruction::LocalGet(MAXB));
        f.instruction(&Instruction::I32GeU);
        f.instruction(&Instruction::BrIf(OUT));

        // addr = ctx + chain_map + slot(rip) * 16, with the same hash the host uses.
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::I64Const(4));
        f.instruction(&Instruction::I64ShrU);
        f.instruction(&Instruction::I64Xor);
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::I64Const(13));
        f.instruction(&Instruction::I64ShrU);
        f.instruction(&Instruction::I64Xor);
        f.instruction(&Instruction::I32WrapI64);
        f.instruction(&Instruction::I32Const(
            (jit_ctx::JIT_CHAIN_MAP_ENTRIES - 1) as i32,
        ));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::I32Const(
            jit_ctx::JIT_CHAIN_MAP_ENTRY_SIZE.trailing_zeros() as i32,
        ));
        f.instruction(&Instruction::I32Shl);
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(ADDR));

        f.instruction(&Instruction::LocalGet(ADDR));
        f.instruction(&Instruction::I64Load(memarg(JitContext::CHAIN_MAP_OFFSET, 3)));
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::I64Ne);
        f.instruction(&Instruction::BrIf(OUT));

        // Same RIP, different module: its function index means nothing here.
        f.instruction(&Instruction::LocalGet(ADDR));
        f.instruction(&Instruction::I32Load(memarg(JitContext::CHAIN_MAP_OFFSET + 8, 2)));
        f.instruction(&Instruction::I32Const(module_id as i32));
        f.instruction(&Instruction::I32Ne);
        f.instruction(&Instruction::BrIf(OUT));

        f.instruction(&Instruction::LocalGet(ADDR));
        f.instruction(&Instruction::I32Load(memarg(JitContext::CHAIN_MAP_OFFSET + 12, 2)));
        f.instruction(&Instruction::LocalSet(IDX));
        f.instruction(&Instruction::Br(AGAIN));

        f.instruction(&Instruction::End); // $again
        f.instruction(&Instruction::End); // $out
        // Publish what the loop accumulated. Every path out of the loop lands here, so
        // the host sees the same two values it always did.
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::LocalGet(ACC));
        f.instruction(&Instruction::I64Store(memarg(JitContext::CHAIN_RETIRED_OFFSET, 3)));
        f.instruction(&Instruction::LocalGet(CTX));
        f.instruction(&Instruction::LocalGet(CNT));
        f.instruction(&Instruction::I32Store(memarg(JitContext::CHAIN_BLOCKS_OFFSET, 2)));
        f.instruction(&Instruction::LocalGet(RET));
        f.instruction(&Instruction::End);
        f
    }

    /// Emit one block's function body.
    ///
    /// Split out so a module can carry more than one. Admitting a block costs
    /// ~577us, essentially all of it `Module::new`, and that cost is per *module*
    /// rather than per instruction compiled — so the way to make a compiled tier
    /// pay for itself is to stop emitting one module per block.
    ///
    /// `imported` must describe the enclosing module's imports, so when several
    /// blocks share a module the caller unions their requirements first: the call
    /// indices baked into these bodies have to match that module's import section.
    fn emit_block_function(
        block: &IrBlock,
        options: Tier1WasmOptions,
        imported: ImportedFuncs,
        shared_loads: SharedLoadHelpers,
        uses_store_u8: bool,
        uses_store_u16: bool,
        uses_store_u32: bool,
        uses_store_u64: bool,
        needs_jit_exit: bool,
    ) -> Function {
        let layout = LocalsLayout::new(block.value_types.len() as u32);
        let state_usage = analyze_state_usage(block, options);

        let mut func = Function::new(vec![(layout.total_i64_locals(), ValType::I64)]);

        // Load architectural state into locals.
        for gpr in all_gprs() {
            if state_usage.gpr_used[gpr.as_u8() as usize] {
                func.instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
                func.instruction(&Instruction::I64Load(memarg(
                    abi::CPU_GPR_OFF[gpr.as_u8() as usize],
                    3,
                )));
                func.instruction(&Instruction::LocalSet(layout.gpr_local(gpr)));
            }
        }
        if state_usage.rip_used {
            func.instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
            func.instruction(&Instruction::I64Load(memarg(abi::CPU_RIP_OFF, 3)));
            func.instruction(&Instruction::LocalSet(layout.rip_local()));
        }

        // Loaded whenever the block touches flags at all, including when every write is
        // deferred into the lazy descriptor and nothing reads the local.
        //
        // That load *is* dead in such a block, and Cranelift cannot remove it — a wasm
        // load may fault into the guard region, so it is a trapping operation. The
        // disassembly shows it surviving, one instruction in a twenty-instruction block.
        //
        // Narrowing the condition to `reads_incoming_flags` — a partial flag write with
        // nothing produced yet, `adc`/`sbb` consuming carry, a condition evaluated before
        // any flags are produced — looks exactly right and is **wrong**: the oracle
        // mismatches at 100K instructions and the machine falls to 1243.2 host
        // instructions per guest instruction. Some path reads the local without any of
        // those three marking it, and a wasm local starts at zero, so it reads flags of
        // zero. Whatever that path is, it has to be found before this load can go.
        //
        // Worth noting how it failed: `AERO_CT_VERIFY` reported **0** mismatches over 3M
        // while the machine-level oracle caught it on the first checkpoint. The per-block
        // verifier only checks replayable blocks, so it is not a substitute for the
        // oracle.
        if state_usage.uses_rflags {
            func.instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
            func.instruction(&Instruction::I64Load(memarg(abi::CPU_RFLAGS_OFF, 3)));
            func.instruction(&Instruction::LocalSet(layout.rflags_local()));
        }

        if options.inline_tlb {
            let has_store_mem = options.inline_tlb_stores
                && (uses_store_u8 || uses_store_u16 || uses_store_u32 || uses_store_u64);
            if has_store_mem {
                // Cache the code-version table pointer and length in locals so the RAM write
                // fast-path can bump code page versions without repeated loads from `cpu_ptr`.
                func.instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
                func.instruction(&Instruction::I32Load(memarg(
                    jit_ctx::CODE_VERSION_TABLE_PTR_OFFSET,
                    2,
                )));
                func.instruction(&Instruction::I64ExtendI32U);
                func.instruction(&Instruction::LocalSet(
                    layout.code_version_table_ptr_local(),
                ));

                func.instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
                func.instruction(&Instruction::I32Load(memarg(
                    jit_ctx::CODE_VERSION_TABLE_LEN_OFFSET,
                    2,
                )));
                func.instruction(&Instruction::I64ExtendI32U);
                func.instruction(&Instruction::LocalSet(
                    layout.code_version_table_len_local(),
                ));
            }

            // Load JIT metadata (guest RAM base and TLB salt).
            func.instruction(&Instruction::LocalGet(layout.jit_ctx_ptr_local()));
            func.instruction(&Instruction::I64Load(memarg(
                JitContext::RAM_BASE_OFFSET,
                3,
            )));
            func.instruction(&Instruction::LocalSet(layout.ram_base_local()));

            func.instruction(&Instruction::LocalGet(layout.jit_ctx_ptr_local()));
            func.instruction(&Instruction::I64Load(memarg(
                JitContext::TLB_SALT_OFFSET,
                3,
            )));
            func.instruction(&Instruction::LocalSet(layout.tlb_salt_local()));

        }

        // Structured single-exit block so we can `br` out of the block on MMIO exits.
        func.instruction(&Instruction::Block(BlockType::Empty));

        let mut emitter = Emitter {
            func: &mut func,
            imported,
            shared_loads,
            layout,
            options,
            depth: 0,
            pending: None,
            incoming_live: false,
            descriptor_touched: false,
            rflags_written: false,
            guest_insts_started: 0,
            track_retired_from: state_usage.first_exiting_guest_inst,
        };

        // A block that reads a flag it did not produce needs what the previous block
        // left owed. It is loaded, not settled: a block that only *reads* flags takes
        // the one it wants straight out of the record and hands the record on, which
        // is the whole point — `je`/`jne` need ZF, and ZF is "the result was zero"
        // whatever produced it.
        if state_usage.reads_incoming_flags {
            emitter.emit_load_incoming_descriptor();
            emitter.incoming_live = true;
            emitter.descriptor_touched = true;
        }

        for inst in &block.insts {
            emitter.emit_inst(inst);
            if matches!(inst, IrInst::CallHelper { .. }) {
                break;
            }
        }
        // Reaching here means every instruction ran. This is inside the exit block on
        // purpose: a bail-out branches past it, so it keeps the prefix count the
        // bailing instruction left behind rather than claiming the block's length.
        if !needs_jit_exit {
            let total = emitter.guest_insts_started;
            emitter.func.instruction(&Instruction::I64Const(total as i64));
            emitter
                .func
                .instruction(&Instruction::LocalSet(layout.retired_local()));
        }

        // If the block contains a `CallHelper`, Tier-1 treats it as an unconditional runtime exit
        // (see `Emitter::emit_inst`). In that case the IR terminator is unreachable and can be
        // omitted to keep the emitted WASM smaller.
        if !needs_jit_exit {
            emitter.emit_terminator(&block.terminator);
        }

        emitter.func.instruction(&Instruction::End); // end exit block

        // Spill guest state back to linear memory.
        for gpr in all_gprs() {
            if state_usage.gpr_written[gpr.as_u8() as usize] {
                emitter
                    .func
                    .instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
                emitter
                    .func
                    .instruction(&Instruction::LocalGet(layout.gpr_local(gpr)));
                emitter.func.instruction(&Instruction::I64Store(memarg(
                    abi::CPU_GPR_OFF[gpr.as_u8() as usize],
                    3,
                )));
            }
        }

        // `state_usage.rflags_written` is a static over-approximation: it is set for any
        // instruction that writes flags, including one whose flags are *deferred* into
        // the descriptor and never materialised into the RFLAGS local at all. Such a
        // block leaves RFLAGS exactly as it found it, so reading it back, setting a bit
        // that is architecturally already set, and storing it is a round trip to
        // nowhere. `emitter.rflags_written` is the fact rather than the estimate.
        if state_usage.rflags_written && (emitter.rflags_written || !rflags_store_trimmed()) {
            // Ensure the architectural "always-1" RFLAGS bit stays set.
            //
            // Note: We update the `rflags` local first and then emit the final store using the
            // simple `local.get cpu_ptr; local.get rflags; i64.store` pattern. Several WASM IR
            // inspection tests rely on that adjacency to detect spills.
            emitter
                .func
                .instruction(&Instruction::LocalGet(layout.rflags_local()));
            emitter
                .func
                .instruction(&Instruction::I64Const(abi::RFLAGS_RESERVED1 as i64));
            emitter.func.instruction(&Instruction::I64Or);
            emitter
                .func
                .instruction(&Instruction::LocalSet(layout.rflags_local()));

            emitter
                .func
                .instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
            emitter
                .func
                .instruction(&Instruction::LocalGet(layout.rflags_local()));
            emitter
                .func
                .instruction(&Instruction::I64Store(memarg(abi::CPU_RFLAGS_OFF, 3)));
        }

        // Record how many guest instructions ran. Constant today, because a block
        // has one exit and therefore one answer; a block with side exits sets it at
        // whichever exit it leaves through, and the runtime reads it either way.
        emitter
            .func
            .instruction(&Instruction::LocalGet(layout.jit_ctx_ptr_local()));
        emitter
            .func
            .instruction(&Instruction::LocalGet(layout.retired_local()));
        emitter
            .func
            .instruction(&Instruction::I64Store(memarg(JitContext::RETIRED_OFFSET, 3)));

        // `AERO_JIT_LAZY_FLAGS_LOCAL=1` settles the descriptor here instead of
        // handing it on, which keeps deferral inside the block and takes the
        // cross-block half out of the picture — the way to tell which half of the
        // change a divergence belongs to.
        if std::env::var_os("AERO_JIT_LAZY_FLAGS_LOCAL").is_some() {
            emitter.emit_materialize_descriptor_locals();
        }

        // Hand back the operation that produced the flags rather than the flags.
        //
        // Tier-0 reads every flag through `get_flag`/`rflags()`, both of which
        // materialise a pending descriptor, so this is what lets a flag nobody looks
        // at go uncomputed by either tier. The header is written on *every* exit,
        // with `op = None` when nothing is pending, so a descriptor left by an
        // earlier block can never be mistaken for this one's.
        emitter.emit_spill_pending_descriptor();

        // The block returns where the guest goes next, and that return value is what
        // the chain loop uses to find the next block and what it writes into
        // `CpuState.rip` when the chain ends. So this store is dead for a block that
        // always runs to its end.
        //
        // It is *not* dead for one that can leave part-way: a bail hands back the
        // exit sentinel rather than a RIP, and the interpreter has to resume at the
        // instruction that faulted, which only this store records.
        // A block whose *terminator* is `ExitToInterpreter` also needs it: it runs to
        // its end, so it has no early exit, but it returns the sentinel rather than a
        // RIP, and the sentinel is exactly what the caller cannot take a RIP from.
        let returns_a_rip = !matches!(block.terminator, IrTerminator::ExitToInterpreter { .. });
        if state_usage.first_exiting_guest_inst.is_some() || !returns_a_rip || !rip_store_trimmed()
        {
            emitter
                .func
                .instruction(&Instruction::LocalGet(layout.cpu_ptr_local()));
            emitter
                .func
                .instruction(&Instruction::LocalGet(layout.next_rip_local()));
            emitter
                .func
                .instruction(&Instruction::I64Store(memarg(abi::CPU_RIP_OFF, 3)));
        }

        emitter
            .func
            .instruction(&Instruction::LocalGet(layout.scratch_local()));
        emitter.func.instruction(&Instruction::Return);
        emitter.func.instruction(&Instruction::End);

        func
    }
}

impl Default for Tier1WasmCodegen {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
struct LocalsLayout {
    values: u32,
}

impl LocalsLayout {
    fn new(values: u32) -> Self {
        Self { values }
    }

    fn cpu_ptr_local(self) -> u32 {
        0
    }

    fn jit_ctx_ptr_local(self) -> u32 {
        1
    }

    fn gpr_local(self, reg: Gpr) -> u32 {
        2 + reg.as_u8() as u32
    }

    fn rip_local(self) -> u32 {
        2 + 16
    }

    fn rflags_local(self) -> u32 {
        self.rip_local() + 1
    }

    fn next_rip_local(self) -> u32 {
        self.rflags_local() + 1
    }

    fn code_version_table_ptr_local(self) -> u32 {
        self.next_rip_local() + 1
    }

    fn code_version_table_len_local(self) -> u32 {
        self.code_version_table_ptr_local() + 1
    }

    fn ram_base_local(self) -> u32 {
        self.code_version_table_len_local() + 1
    }

    fn tlb_salt_local(self) -> u32 {
        self.ram_base_local() + 1
    }

    fn scratch_vaddr_local(self) -> u32 {
        self.tlb_salt_local() + 1
    }

    fn scratch_vpn_local(self) -> u32 {
        self.scratch_vaddr_local() + 1
    }

    fn scratch_tlb_data_local(self) -> u32 {
        self.scratch_vpn_local() + 1
    }

    fn scratch_local(self) -> u32 {
        self.scratch_tlb_data_local() + 1
    }

    /// The TLB entry address for the access being emitted.
    ///
    /// Held so the tag load and the data load do not each rebuild it.
    fn tlb_entry_local(self) -> u32 {
        self.scratch_local() + 1
    }

    /// Guest instructions retired so far.
    ///
    /// Kept in a local rather than derived at the exit, for the same reason the
    /// lazy-flag record is: a block can leave from more than one place, and the
    /// shared epilogue cannot know which.
    fn retired_local(self) -> u32 {
        self.tlb_entry_local() + 1
    }

    /// `op | size << 8 | carry_in << 16` for the deferred operation, or zero.
    ///
    /// The descriptor lives in locals rather than being reconstructed at the exit
    /// from what the code generator last saw, because a block can leave through a
    /// bail-out in the middle: the epilogue is shared, so a compile-time answer would
    /// describe arithmetic that path never ran. Locals start at zero, which is
    /// exactly "nothing pending".
    fn lazy_header_local(self) -> u32 {
        self.retired_local() + 1
    }

    /// Result of the deferred operation.
    ///
    /// It cannot simply reference wherever the result already is: `cmp` and `test`
    /// compute theirs into the shared scratch local, which the inline TLB and the
    /// shift helpers reuse long before the block ends.
    fn lazy_result_local(self) -> u32 {
        self.lazy_header_local() + 1
    }

    fn lazy_op1_local(self) -> u32 {
        self.lazy_result_local() + 1
    }

    fn lazy_op2_local(self) -> u32 {
        self.lazy_op1_local() + 1
    }

    /// Holds the carry that a deferred `adc`/`sbb` consumed.
    ///
    /// Its flags are reconstructed later, possibly after CF has been overwritten, so
    /// the value it was given has to be kept rather than re-read.
    fn lazy_carry_local(self) -> u32 {
        self.lazy_op2_local() + 1
    }

    fn value_base(self) -> u32 {
        self.lazy_carry_local() + 1
    }

    fn value_local(self, ValueId(id): ValueId) -> u32 {
        self.value_base() + id
    }

    fn total_i64_locals(self) -> u32 {
        // gpr[16] + rip + rflags + next_rip + code_version_table ptr/len + ram_base + tlb_salt +
        // scratch locals (vaddr, vpn, tlb_data, scratch, tlb_entry) + retired + the
        // lazy-flag descriptor (header, result, op1, op2, carry) + values
        16 + 1 + 1 + 1 + 2 + 1 + 1 + 5 + 1 + 5 + self.values
    }
}

/// The arithmetic that produced the flags a block has not computed yet.
///
/// x86 defines six flags on most ALU instructions and real code reads one or two of
/// them, usually the next instruction. Computing all six eagerly is the single
/// largest cost in compiled code, so instead the operation is *remembered* — which
/// operands, which width, which operation — and a flag is built only where it is
/// actually read. What is left pending when the block ends is written to
/// `CpuState.lazy_flags`, which Tier-0 already knows how to materialise on demand, so
/// flags nobody reads are never computed by either tier.
///
/// The operand fields are wasm locals. Values live in locals for the whole function,
/// so a pending record stays valid however far the read is from the operation.
#[derive(Clone, Copy)]
struct PendingFlags {
    op: LazyOp,
    width: Width,
}

/// The operations whose flags can be deferred, matching `LazyFlagOp` in `CpuState`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LazyOp {
    Add,
    Adc,
    Sub,
    Sbb,
    Logic,
}

impl LazyOp {
    /// The discriminant `aero_cpu_core::state::LazyFlagOp` uses.
    fn abi_code(self) -> u64 {
        match self {
            LazyOp::Add => 1,
            LazyOp::Adc => 2,
            LazyOp::Sub => 3,
            LazyOp::Sbb => 4,
            LazyOp::Logic => 5,
        }
    }

    fn is_add_like(self) -> bool {
        matches!(self, LazyOp::Add | LazyOp::Adc)
    }
}

/// Whether stores take the single fast path too. Off, and measured.
///
/// The same change is a win for loads and a loss for stores. Over a whole session,
/// host instructions per guest instruction: nested paths 674.7, loads on the single
/// fast path 632.5, loads *and* stores 763.3. The load half is kept and the store
/// half is not; `AERO_JIT_STORE_FP=1` turns it on for anyone who wants to chase the
/// difference, which is most likely that a non-RAM store now calls the helper where
/// it used to leave the block, and that every storing block must import the helper.
/// Skip the lazy-flag descriptor's three operand stores in a block that never touched
/// it — one that neither loaded what the previous block owed nor deferred an operation
/// of its own, and so is storing three zeros nobody reads.
///
/// The header store stays: it may have to clear a record left owed by an earlier block,
/// and a stale one would be read as work still due.
///
/// Worth 12 host instructions of the 384 a one-instruction load block costs
/// (`tests/compiled_code_cost.rs`), and end to end 104.9 / 88.9 / 101.5 / 92.3 against
/// 105.2 / 89.0 / 102.4 / 92.5 — better in all four windows, and small because most
/// blocks *do* touch the descriptor: they end in a compare and a branch.
/// `AERO_JIT_NO_TRIM_SPILL=1` goes back.
fn descriptor_spill_trimmed() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_NO_TRIM_SPILL").is_none());
    }
    #[cfg(target_arch = "wasm32")]
    {
        true
    }
}

/// Skip the epilogue's `CpuState.rip` store in a block that can never leave part-way.
///
/// The block already *returns* where the guest goes next: the chain loop uses that to
/// find the next block, and writes it into `CpuState.rip` when the chain ends. So for a
/// block that always runs to its end the store is dead. A block that can bail keeps it,
/// because a bail returns the exit sentinel instead of a RIP and the interpreter has to
/// resume at the instruction that faulted.
///
/// **104.4 / 88.3 / 100.9 / 91.8 against 104.9 / 88.8 / 101.5 / 92.3** — better in all
/// four windows by 0.5–0.6%, unusually consistent for this workload because removing a
/// store does not move any block boundary, and boundaries are what move the guest onto
/// a different path. `AERO_JIT_NO_TRIM_RIP=1` goes back.
fn rip_store_trimmed() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_NO_TRIM_RIP").is_none());
    }
    #[cfg(target_arch = "wasm32")]
    {
        true
    }
}

/// Skip the epilogue's RFLAGS store when nothing actually wrote the local.
///
/// `state_usage.rflags_written` is set for any instruction that writes flags — including
/// one whose flags are *deferred* into the lazy descriptor and never materialised into
/// the RFLAGS local. Such a block leaves RFLAGS exactly as it found it, so loading it,
/// setting a bit that is architecturally already set, and storing it back is a round
/// trip to nowhere. Deferred flags are the common case, so this is most ALU blocks.
///
/// **102.5 / 90.2 / 86.8 / 99.1 against 104.3 / 91.8 / 88.4 / 100.9** — better in all
/// four windows by 1.7–1.8%, the largest of the three epilogue trims and, like them,
/// consistent because it moves no block boundary.
/// `AERO_JIT_NO_TRIM_RFLAGS=1` goes back.
fn rflags_store_trimmed() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_NO_TRIM_RFLAGS").is_none());
    }
    #[cfg(target_arch = "wasm32")]
    {
        true
    }
}

/// Inline the `br_table` into the chaining dispatcher (`AERO_JIT_TWO_CALL_DISPATCH=1`
/// restores the old shape).
///
/// A chain step used to be *two* wasm calls: the chaining loop called a `br_table`
/// function, which called the block. Disassembly put 28 native instructions in that
/// middle function — its own frame, its own stack-limit check, the jump table and a
/// call/return — out of a ~94 instruction step.
///
/// This was measured at **300 against 282** and rejected once. That was before the
/// linear-memory guard covered the whole `i32` range, when every context access carried
/// a three-instruction bounds check and the middle call was a small share of a much
/// bigger step. Re-run at the new baseline it is a consistent win: **49.4/50.7,
/// 49.1/50.4, 64.1/65.6, 63.5/65.0, 62.6/64.1** across five paired deep windows, about
/// 2.4%. (A sixth read 43.6 against 63.9; 300M-400M swings and one window is not a
/// result.)
fn inline_dispatch_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_TWO_CALL_DISPATCH").is_none());
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Use `& 0xffff_ffff` for 32-bit truncation instead of wrap/extend (`AERO_JIT_MASK32=1`).
fn mask32_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_MASK32").is_some());
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

fn store_fastpath_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_JIT_STORE_FP").is_some());
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

struct Emitter<'a> {
    func: &'a mut Function,
    imported: ImportedFuncs,
    shared_loads: SharedLoadHelpers,
    layout: LocalsLayout,
    options: Tier1WasmOptions,
    /// Current nesting depth *inside* the single-exit `block`.
    depth: u32,
    /// Flags produced but not yet computed. See [`PendingFlags`].
    pending: Option<PendingFlags>,
    /// The descriptor locals hold work handed in by the previous block, not yet
    /// settled. A block that only reads flags leaves it that way and passes it on.
    incoming_live: bool,
    /// Whether the descriptor's *operand* locals were ever given a value — by loading
    /// what the previous block owed, or by deferring an operation here.
    descriptor_touched: bool,
    /// Whether anything actually put a value into the RFLAGS local. A block that only
    /// *defers* its flags leaves RFLAGS exactly as it found it.
    rflags_written: bool,
    /// Guest instructions begun so far. `translate` writes RIP exactly once at the
    /// start of each, which is what makes that a reliable boundary marker.
    guest_insts_started: u64,
    /// From which guest instruction the running retired count has to be kept.
    /// `None` means the block cannot leave part-way, so it never does.
    track_retired_from: Option<u64>,
}

impl Emitter<'_> {
    fn inline_tlb_locals(&self) -> InlineTlbLocals {
        InlineTlbLocals {
            cpu_ptr: self.layout.cpu_ptr_local(),
            jit_ctx_ptr: self.layout.jit_ctx_ptr_local(),
            ram_base: self.layout.ram_base_local(),
            tlb_salt: self.layout.tlb_salt_local(),
            scratch_vaddr: self.layout.scratch_vaddr_local(),
            scratch_vpn: self.layout.scratch_vpn_local(),
            scratch_tlb_data: self.layout.scratch_tlb_data_local(),

            code_version_table_ptr: Some(self.layout.code_version_table_ptr_local()),
            code_version_table_len: Some(self.layout.code_version_table_len_local()),
        }
    }

    fn emit_inst(&mut self, inst: &IrInst) {
        match inst {
            IrInst::Const { dst, value, width } => {
                let v = width.truncate(*value) as i64;
                self.func.instruction(&Instruction::I64Const(v));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::ReadReg { dst, reg } => {
                match *reg {
                    GuestReg::Rip => {
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.rip_local()));
                    }
                    GuestReg::Gpr { reg, width, high8 } => {
                        self.emit_read_gpr_part(reg, width, high8);
                    }
                    GuestReg::Flag(flag) => {
                        self.emit_read_flag(flag);
                        self.func.instruction(&Instruction::I64ExtendI32U);
                    }
                    // Read from `CpuState` at the point of use rather than cached in
                    // a local at block entry. It only changes through instructions
                    // Tier-1 refuses to translate, so a local would also be correct —
                    // but one load where a `gs:` operand appears is cheaper than one
                    // in every block that happens to contain one.
                    GuestReg::SegBase(seg) => {
                        let off = match seg {
                            crate::tier1::ir::SegBase::Fs => abi::CPU_FS_BASE_OFF,
                            crate::tier1::ir::SegBase::Gs => abi::CPU_GS_BASE_OFF,
                        };
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
                        self.func.instruction(&Instruction::I64Load(memarg(off, 3)));
                    }
                }
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::SideExit { cond, target } => {
                // The taken path leaves; the fall-through carries on in this same
                // function with the guest registers still in locals, which is the
                // whole point of continuing past a branch.
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*cond)));
                self.func.instruction(&Instruction::I32WrapI64);
                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                {
                    // The branch itself has retired by now: its RIP write already
                    // counted it.
                    let retired = self.guest_insts_started;
                    self.func.instruction(&Instruction::I64Const(retired as i64));
                    self.func
                        .instruction(&Instruction::LocalSet(self.layout.retired_local()));

                    // Two locals, both needed: `next_rip` is spilled to
                    // `CpuState.rip`, and the *return value* is taken from the scratch
                    // local — which the inline TLB also uses, so it holds junk unless
                    // set here. A side exit is ordinary control flow, so it returns
                    // the target rather than the bail-out sentinel.
                    self.func
                        .instruction(&Instruction::I64Const(*target as i64));
                    self.func
                        .instruction(&Instruction::LocalSet(self.layout.next_rip_local()));
                    self.func
                        .instruction(&Instruction::I64Const(*target as i64));
                    self.func
                        .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
                    // Out to the block's single exit, where state is spilled.
                    self.func.instruction(&Instruction::Br(self.depth));
                }
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }
            IrInst::WriteReg { reg, src } => match *reg {
                GuestReg::SegBase(_) => unreachable!("segment bases are read-only in Tier-1"),
                GuestReg::Rip => {
                    // A new guest instruction starts here, so everything before it
                    // retired. A block that bails part-way reports that prefix rather
                    // than claiming its whole length — but only instructions that can
                    // actually bail need the count kept up to date, and most cannot.
                    let done = self.guest_insts_started;
                    self.guest_insts_started += 1;
                    if self.track_retired_from.is_some_and(|first| done >= first) {
                        self.func.instruction(&Instruction::I64Const(done as i64));
                        self.func
                            .instruction(&Instruction::LocalSet(self.layout.retired_local()));
                    }
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*src)));
                    self.func
                        .instruction(&Instruction::LocalSet(self.layout.rip_local()));
                }
                GuestReg::Gpr { reg, width, high8 } => {
                    self.emit_write_gpr_part(reg, width, high8, *src);
                }
                GuestReg::Flag(flag) => {
                    self.emit_write_flag(flag, *src);
                }
            },
            IrInst::Trunc { dst, src, width } | IrInst::Zext { dst, src, width } => {
                // Locals are i64 and a narrow value is held with its high bits
                // clear, so widening and narrowing are the same masking operation.
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*src)));
                self.emit_trunc(*width);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::Load { dst, addr, width } => {
                if self.options.inline_tlb && self.options.single_fast_path {
                    self.emit_load_single_fast_path(*dst, *addr, *width);
                    return;
                }
                if !self.options.inline_tlb {
                    // Baseline mode: always go through the imported slow helpers.
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*addr)));
                    match *width {
                        Width::W8 => {
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_read_u8
                                    .expect("mem_read_u8 import missing"),
                            ));
                            self.func.instruction(&Instruction::I64ExtendI32U);
                        }
                        Width::W16 => {
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_read_u16
                                    .expect("mem_read_u16 import missing"),
                            ));
                            self.func.instruction(&Instruction::I64ExtendI32U);
                        }
                        Width::W32 => {
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_read_u32
                                    .expect("mem_read_u32 import missing"),
                            ));
                            self.func.instruction(&Instruction::I64ExtendI32U);
                        }
                        Width::W64 => {
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_read_u64
                                    .expect("mem_read_u64 import missing"),
                            ));
                        }
                    }
                    self.emit_trunc(*width);
                    self.func
                        .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
                    return;
                }

                let dst = *dst;
                let addr = *addr;
                let width = *width;

                // Save vaddr into a scratch local (used by both slow/fast paths).
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.scratch_vaddr_local()));

                let (size_bytes, slow_read) = match width {
                    Width::W8 => (1u32, self.imported.mem_read_u8),
                    Width::W16 => (2u32, self.imported.mem_read_u16),
                    Width::W32 => (4u32, self.imported.mem_read_u32),
                    Width::W64 => (8u32, self.imported.mem_read_u64),
                };

                let emit_same_page_load = |this: &mut Self| {
                    // Fast path: inline TLB probe + direct RAM load.
                    this.emit_translate_and_cache(MMU_ACCESS_READ, crate::TLB_FLAG_READ);

                    if this.options.inline_tlb_mmio_exit {
                        this.emit_mmio_exit(size_bytes, 0, None);

                        this.emit_compute_ram_addr();
                        match width {
                            Width::W8 => {
                                this.func.instruction(&Instruction::I64Load8U(memarg(0, 0)))
                            }
                            Width::W16 => this
                                .func
                                .instruction(&Instruction::I64Load16U(memarg(0, 1))),
                            Width::W32 => this
                                .func
                                .instruction(&Instruction::I64Load32U(memarg(0, 2))),
                            Width::W64 => {
                                this.func.instruction(&Instruction::I64Load(memarg(0, 3)))
                            }
                        };
                        this.emit_trunc(width);
                        this.func
                            .instruction(&Instruction::LocalSet(this.layout.value_local(dst)));
                    } else {
                        // If the translation isn't backed by RAM (MMIO/ROM/unmapped), fall back to
                        // the slow helper and keep executing this block.
                        this.func.instruction(&Instruction::LocalGet(
                            this.layout.scratch_tlb_data_local(),
                        ));
                        this.func
                            .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
                        this.func.instruction(&Instruction::I64And);
                        this.func.instruction(&Instruction::I64Eqz);

                        this.func.instruction(&Instruction::If(BlockType::Empty));
                        this.depth += 1;
                        {
                            // Slow path.
                            let slow_read = slow_read.expect("memory read helper import missing");
                            this.emit_slow_mem_read(dst, width, slow_read);
                        }
                        this.func.instruction(&Instruction::Else);
                        {
                            // RAM fast-path.
                            this.emit_compute_ram_addr();
                            match width {
                                Width::W8 => {
                                    this.func.instruction(&Instruction::I64Load8U(memarg(0, 0)))
                                }
                                Width::W16 => this
                                    .func
                                    .instruction(&Instruction::I64Load16U(memarg(0, 1))),
                                Width::W32 => this
                                    .func
                                    .instruction(&Instruction::I64Load32U(memarg(0, 2))),
                                Width::W64 => {
                                    this.func.instruction(&Instruction::I64Load(memarg(0, 3)))
                                }
                            };
                            this.emit_trunc(width);
                            this.func
                                .instruction(&Instruction::LocalSet(this.layout.value_local(dst)));
                        }
                        this.func.instruction(&Instruction::End);
                        this.depth -= 1;
                    }
                };

                if size_bytes == 1 {
                    // An 8-bit load can never cross a page boundary, so we can skip the cross-page
                    // check and its associated slow-path fallback entirely.
                    emit_same_page_load(self);
                    return;
                }

                if self.options.inline_tlb_mmio_exit
                    && !self.options.inline_tlb_cross_page_fastpath
                    && slow_read.is_none()
                {
                    // When MMIO exits are enabled and the cross-page fast-path is disabled, the
                    // code generator normally falls back to the slow helper on boundary-crossing
                    // accesses. If we didn't import the slow helper, the import selection logic
                    // has proven that cross-page loads of this width cannot occur, so we can skip
                    // the cross-page check entirely.
                    emit_same_page_load(self);
                    return;
                }

                // Cross-page accesses default to the slow helper for correctness unless the
                // optional split-access fast-path is enabled.
                let cross_limit =
                    crate::PAGE_OFFSET_MASK.saturating_sub(size_bytes.saturating_sub(1) as u64);
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
                self.func
                    .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
                self.func.instruction(&Instruction::I64And);
                self.func
                    .instruction(&Instruction::I64Const(cross_limit as i64));
                self.func.instruction(&Instruction::I64GtU);

                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                {
                    if self.options.inline_tlb_cross_page_fastpath
                        && matches!(width, Width::W16 | Width::W32 | Width::W64)
                    {
                        // Cross-page fast-path: translate both pages, ensure both are RAM, then do
                        // two direct same-page loads and combine the result (little-endian).

                        // shift_bytes = (vaddr & 0xFFF) - (PAGE_SIZE - size_bytes)
                        let page_tail = crate::PAGE_SIZE.saturating_sub(size_bytes as u64);
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
                        self.func
                            .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
                        self.func.instruction(&Instruction::I64And);
                        self.func
                            .instruction(&Instruction::I64Const(page_tail as i64));
                        self.func.instruction(&Instruction::I64Sub);
                        self.func
                            .instruction(&Instruction::LocalSet(self.layout.scratch_local()));

                        let emit_split_load = |this: &mut Self| {
                            // part0 = loadN(vaddr - shift_bytes) >> (shift_bytes * 8)
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.value_local(addr)));
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::LocalSet(
                                this.layout.scratch_vaddr_local(),
                            ));

                            this.emit_translate_and_cache(MMU_ACCESS_READ, crate::TLB_FLAG_READ);
                            this.emit_compute_ram_addr();
                            match width {
                                Width::W16 => this
                                    .func
                                    .instruction(&Instruction::I64Load16U(memarg(0, 1))),
                                Width::W32 => this
                                    .func
                                    .instruction(&Instruction::I64Load32U(memarg(0, 2))),
                                Width::W64 => {
                                    this.func.instruction(&Instruction::I64Load(memarg(0, 3)))
                                }
                                _ => unreachable!(),
                            };
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Const(8));
                            this.func.instruction(&Instruction::I64Mul);
                            this.func.instruction(&Instruction::I64ShrU);
                            // Stash part0 in `dst`'s local as a temporary.
                            this.func
                                .instruction(&Instruction::LocalSet(this.layout.value_local(dst)));

                            // part1 = (loadN(vaddr1) & ((1 << (shift_bytes * 8)) - 1))
                            //           << ((size_bytes * 8) - (shift_bytes * 8))
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.value_local(addr)));
                            this.func
                                .instruction(&Instruction::I64Const(size_bytes as i64));
                            this.func.instruction(&Instruction::I64Add);
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::LocalSet(
                                this.layout.scratch_vaddr_local(),
                            ));

                            this.emit_translate_and_cache(MMU_ACCESS_READ, crate::TLB_FLAG_READ);
                            this.emit_compute_ram_addr();
                            match width {
                                Width::W16 => this
                                    .func
                                    .instruction(&Instruction::I64Load16U(memarg(0, 1))),
                                Width::W32 => this
                                    .func
                                    .instruction(&Instruction::I64Load32U(memarg(0, 2))),
                                Width::W64 => {
                                    this.func.instruction(&Instruction::I64Load(memarg(0, 3)))
                                }
                                _ => unreachable!(),
                            };

                            // mask = (1 << (shift_bytes*8)) - 1
                            this.func.instruction(&Instruction::I64Const(1));
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Const(8));
                            this.func.instruction(&Instruction::I64Mul);
                            this.func.instruction(&Instruction::I64Shl);
                            this.func.instruction(&Instruction::I64Const(1));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::I64And);

                            // << ((size_bytes*8) - (shift_bytes*8))
                            this.func
                                .instruction(&Instruction::I64Const((size_bytes * 8) as i64));
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Const(8));
                            this.func.instruction(&Instruction::I64Mul);
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::I64Shl);

                            // part0 | part1
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.value_local(dst)));
                            this.func.instruction(&Instruction::I64Or);
                            this.emit_trunc(width);
                            this.func
                                .instruction(&Instruction::LocalSet(this.layout.value_local(dst)));
                        };

                        // Page 0: translate.
                        self.emit_translate_and_cache(MMU_ACCESS_READ, crate::TLB_FLAG_READ);

                        if self.options.inline_tlb_mmio_exit {
                            self.emit_mmio_exit(size_bytes, 0, None);

                            // Page 1: translate and ensure RAM. Use the original vaddr for the MMIO
                            // exit payload (even though we probe the second page).
                            //
                            // vaddr1 = vaddr + size_bytes - shift_bytes
                            self.func
                                .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
                            self.func
                                .instruction(&Instruction::I64Const(size_bytes as i64));
                            self.func.instruction(&Instruction::I64Add);
                            self.func
                                .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
                            self.func.instruction(&Instruction::I64Sub);
                            self.func.instruction(&Instruction::LocalSet(
                                self.layout.scratch_vaddr_local(),
                            ));

                            self.emit_translate_and_cache(MMU_ACCESS_READ, crate::TLB_FLAG_READ);

                            self.func
                                .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
                            self.func.instruction(&Instruction::LocalSet(
                                self.layout.scratch_vaddr_local(),
                            ));
                            self.emit_mmio_exit(size_bytes, 0, None);

                            emit_split_load(self);
                        } else {
                            // If either page isn't backed by RAM, fall back to the slow helper and
                            // keep executing this block.
                            self.func.instruction(&Instruction::LocalGet(
                                self.layout.scratch_tlb_data_local(),
                            ));
                            self.func
                                .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
                            self.func.instruction(&Instruction::I64And);
                            self.func.instruction(&Instruction::I64Eqz);

                            self.func.instruction(&Instruction::If(BlockType::Empty));
                            self.depth += 1;
                            {
                                // Slow path.
                                let slow_read =
                                    slow_read.expect("memory read helper import missing");
                                self.emit_slow_mem_read(dst, width, slow_read);
                            }
                            self.func.instruction(&Instruction::Else);
                            {
                                // Translate page 1.
                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.value_local(addr),
                                ));
                                self.func
                                    .instruction(&Instruction::I64Const(size_bytes as i64));
                                self.func.instruction(&Instruction::I64Add);
                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.scratch_local(),
                                ));
                                self.func.instruction(&Instruction::I64Sub);
                                self.func.instruction(&Instruction::LocalSet(
                                    self.layout.scratch_vaddr_local(),
                                ));

                                self.emit_translate_and_cache(
                                    MMU_ACCESS_READ,
                                    crate::TLB_FLAG_READ,
                                );

                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.scratch_tlb_data_local(),
                                ));
                                self.func.instruction(&Instruction::I64Const(
                                    crate::TLB_FLAG_IS_RAM as i64,
                                ));
                                self.func.instruction(&Instruction::I64And);
                                self.func.instruction(&Instruction::I64Eqz);

                                self.func.instruction(&Instruction::If(BlockType::Empty));
                                self.depth += 1;
                                {
                                    // Slow path (restore original vaddr first).
                                    self.func.instruction(&Instruction::LocalGet(
                                        self.layout.value_local(addr),
                                    ));
                                    self.func.instruction(&Instruction::LocalSet(
                                        self.layout.scratch_vaddr_local(),
                                    ));

                                    let slow_read =
                                        slow_read.expect("memory read helper import missing");
                                    self.emit_slow_mem_read(dst, width, slow_read);
                                }
                                self.func.instruction(&Instruction::Else);
                                {
                                    emit_split_load(self);
                                }
                                self.func.instruction(&Instruction::End);
                                self.depth -= 1;
                            }
                            self.func.instruction(&Instruction::End);
                            self.depth -= 1;
                        }
                    } else {
                        // Slow path.
                        let slow_read = slow_read.expect("memory read helper import missing");
                        self.emit_slow_mem_read(dst, width, slow_read);
                    }
                }
                self.func.instruction(&Instruction::Else);
                {
                    emit_same_page_load(self);
                }
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }
            IrInst::Store { addr, src, width } => {
                if !self.options.inline_tlb || !self.options.inline_tlb_stores {
                    // Slow path: always go through the imported helpers.
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*addr)));
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*src)));
                    match *width {
                        Width::W8 => {
                            self.emit_trunc(Width::W8);
                            self.func.instruction(&Instruction::I32WrapI64);
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_write_u8
                                    .expect("mem_write_u8 import missing"),
                            ));
                        }
                        Width::W16 => {
                            self.emit_trunc(Width::W16);
                            self.func.instruction(&Instruction::I32WrapI64);
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_write_u16
                                    .expect("mem_write_u16 import missing"),
                            ));
                        }
                        Width::W32 => {
                            self.emit_trunc(Width::W32);
                            self.func.instruction(&Instruction::I32WrapI64);
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_write_u32
                                    .expect("mem_write_u32 import missing"),
                            ));
                        }
                        Width::W64 => {
                            self.func.instruction(&Instruction::Call(
                                self.imported
                                    .mem_write_u64
                                    .expect("mem_write_u64 import missing"),
                            ));
                        }
                    }
                    return;
                }

                if self.options.single_fast_path && store_fastpath_enabled() {
                    self.emit_store_single_fast_path(*src, *addr, *width);
                    return;
                }

                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*addr)));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.scratch_vaddr_local()));

                let (size_bytes, slow_write) = match *width {
                    Width::W8 => (1u32, self.imported.mem_write_u8),
                    Width::W16 => (2u32, self.imported.mem_write_u16),
                    Width::W32 => (4u32, self.imported.mem_write_u32),
                    Width::W64 => (8u32, self.imported.mem_write_u64),
                };

                let emit_same_page_store = |this: &mut Self| {
                    // Fast path: inline TLB probe + direct RAM store.
                    this.emit_translate_and_cache(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);

                    if this.options.inline_tlb_mmio_exit {
                        this.emit_mmio_exit(size_bytes, 1, Some(*src));

                        this.emit_compute_ram_addr();
                        this.func
                            .instruction(&Instruction::LocalGet(this.layout.value_local(*src)));
                        match *width {
                            Width::W8 => {
                                this.func.instruction(&Instruction::I64Store8(memarg(0, 0)))
                            }
                            Width::W16 => this
                                .func
                                .instruction(&Instruction::I64Store16(memarg(0, 1))),
                            Width::W32 => this
                                .func
                                .instruction(&Instruction::I64Store32(memarg(0, 2))),
                            Width::W64 => {
                                this.func.instruction(&Instruction::I64Store(memarg(0, 3)))
                            }
                        };

                        // Self-modifying code invalidation: bump the version entry for the written
                        // physical page. We conservatively bump for all RAM writes.
                        this.emit_bump_code_version_fastpath();
                    } else {
                        // If the translation isn't backed by RAM (MMIO/ROM/unmapped), fall back to
                        // the slow helper and keep executing this block.
                        this.func.instruction(&Instruction::LocalGet(
                            this.layout.scratch_tlb_data_local(),
                        ));
                        this.func
                            .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
                        this.func.instruction(&Instruction::I64And);
                        this.func.instruction(&Instruction::I64Eqz);

                        this.func.instruction(&Instruction::If(BlockType::Empty));
                        this.depth += 1;
                        {
                            // Slow path.
                            let slow_write =
                                slow_write.expect("memory write helper import missing");
                            this.emit_slow_mem_write(*src, *width, slow_write);
                        }
                        this.func.instruction(&Instruction::Else);
                        {
                            // RAM fast-path.
                            this.emit_compute_ram_addr();
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.value_local(*src)));
                            match *width {
                                Width::W8 => {
                                    this.func.instruction(&Instruction::I64Store8(memarg(0, 0)))
                                }
                                Width::W16 => this
                                    .func
                                    .instruction(&Instruction::I64Store16(memarg(0, 1))),
                                Width::W32 => this
                                    .func
                                    .instruction(&Instruction::I64Store32(memarg(0, 2))),
                                Width::W64 => {
                                    this.func.instruction(&Instruction::I64Store(memarg(0, 3)))
                                }
                            };

                            // Self-modifying code invalidation: bump the version entry for the
                            // written physical page. We conservatively bump for all RAM writes.
                            this.emit_bump_code_version_fastpath();
                        }
                        this.func.instruction(&Instruction::End);
                        this.depth -= 1;
                    }
                };

                if size_bytes == 1 {
                    // An 8-bit store can never cross a page boundary, so we can skip the cross-page
                    // check and its associated slow-path fallback entirely.
                    emit_same_page_store(self);
                    return;
                }

                if self.options.inline_tlb_mmio_exit
                    && !self.options.inline_tlb_cross_page_fastpath
                    && slow_write.is_none()
                {
                    // See the analogous load case above: if we didn't import the slow helper, this
                    // store is guaranteed not to cross a page boundary.
                    emit_same_page_store(self);
                    return;
                }

                let cross_limit =
                    crate::PAGE_OFFSET_MASK.saturating_sub(size_bytes.saturating_sub(1) as u64);
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
                self.func
                    .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
                self.func.instruction(&Instruction::I64And);
                self.func
                    .instruction(&Instruction::I64Const(cross_limit as i64));
                self.func.instruction(&Instruction::I64GtU);

                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                {
                    if self.options.inline_tlb_cross_page_fastpath
                        && matches!(*width, Width::W16 | Width::W32 | Width::W64)
                    {
                        // Cross-page fast-path: translate both pages, ensure both are RAM, then
                        // split the store into per-page writes (little-endian).

                        // shift_bytes = (vaddr & 0xFFF) - (PAGE_SIZE - size_bytes)
                        let page_tail = crate::PAGE_SIZE.saturating_sub(size_bytes as u64);
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
                        self.func
                            .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
                        self.func.instruction(&Instruction::I64And);
                        self.func
                            .instruction(&Instruction::I64Const(page_tail as i64));
                        self.func.instruction(&Instruction::I64Sub);
                        self.func
                            .instruction(&Instruction::LocalSet(self.layout.scratch_local()));

                        let emit_split_store_and_bump = |this: &mut Self| {
                            // After confirming both pages are RAM, emit a per-chunk split store. The
                            // additional inline-TLB probes in these chunks should hit the already
                            // cached entries in the common case, while keeping the codegen simple.

                            let addr_local = this.layout.value_local(*addr);
                            let src_local = this.layout.value_local(*src);
                            let shift_local = this.layout.scratch_local();
                            let scratch_vaddr_local = this.layout.scratch_vaddr_local();

                            let emit_store_page0_chunk =
                                |this: &mut Self, mem_off: u32, nbytes: u32, shift_bits: u32| {
                                    // scratch_vaddr = vaddr + mem_off
                                    this.func.instruction(&Instruction::LocalGet(addr_local));
                                    if mem_off != 0 {
                                        this.func
                                            .instruction(&Instruction::I64Const(mem_off as i64));
                                        this.func.instruction(&Instruction::I64Add);
                                    }
                                    this.func
                                        .instruction(&Instruction::LocalSet(scratch_vaddr_local));

                                    this.emit_translate_and_cache(
                                        MMU_ACCESS_WRITE,
                                        crate::TLB_FLAG_WRITE,
                                    );
                                    this.emit_compute_ram_addr();
                                    this.func.instruction(&Instruction::LocalGet(src_local));
                                    if shift_bits != 0 {
                                        this.func
                                            .instruction(&Instruction::I64Const(shift_bits as i64));
                                        this.func.instruction(&Instruction::I64ShrU);
                                    }
                                    match nbytes {
                                        1 => this
                                            .func
                                            .instruction(&Instruction::I64Store8(memarg(0, 0))),
                                        2 => this
                                            .func
                                            .instruction(&Instruction::I64Store16(memarg(0, 1))),
                                        4 => this
                                            .func
                                            .instruction(&Instruction::I64Store32(memarg(0, 2))),
                                        _ => unreachable!("invalid store chunk size: {nbytes}"),
                                    };
                                };

                            let emit_store_page1_chunk =
                                |this: &mut Self, mem_off: u32, nbytes: u32, shift_bits: u32| {
                                    // scratch_vaddr = vaddr + size_bytes - shift_bytes + mem_off
                                    this.func.instruction(&Instruction::LocalGet(addr_local));
                                    this.func
                                        .instruction(&Instruction::I64Const(size_bytes as i64));
                                    this.func.instruction(&Instruction::I64Add);
                                    this.func.instruction(&Instruction::LocalGet(shift_local));
                                    this.func.instruction(&Instruction::I64Sub);
                                    if mem_off != 0 {
                                        this.func
                                            .instruction(&Instruction::I64Const(mem_off as i64));
                                        this.func.instruction(&Instruction::I64Add);
                                    }
                                    this.func
                                        .instruction(&Instruction::LocalSet(scratch_vaddr_local));

                                    this.emit_translate_and_cache(
                                        MMU_ACCESS_WRITE,
                                        crate::TLB_FLAG_WRITE,
                                    );
                                    this.emit_compute_ram_addr();
                                    this.func.instruction(&Instruction::LocalGet(src_local));
                                    if shift_bits != 0 {
                                        this.func
                                            .instruction(&Instruction::I64Const(shift_bits as i64));
                                        this.func.instruction(&Instruction::I64ShrU);
                                    }
                                    match nbytes {
                                        1 => this
                                            .func
                                            .instruction(&Instruction::I64Store8(memarg(0, 0))),
                                        2 => this
                                            .func
                                            .instruction(&Instruction::I64Store16(memarg(0, 1))),
                                        4 => this
                                            .func
                                            .instruction(&Instruction::I64Store32(memarg(0, 2))),
                                        _ => unreachable!("invalid store chunk size: {nbytes}"),
                                    };
                                };

                            // Dispatch based on `shift_bytes` (bytes written to page 1). Note that in
                            // the cross-page case `shift_bytes` is always in the range [1, size_bytes).
                            match *width {
                                Width::W16 => {
                                    emit_store_page0_chunk(this, 0, 1, 0);
                                    emit_store_page1_chunk(this, 0, 1, 8);
                                }
                                Width::W32 => {
                                    this.func.instruction(&Instruction::LocalGet(shift_local));
                                    this.func.instruction(&Instruction::I64Const(1));
                                    this.func.instruction(&Instruction::I64Eq);
                                    this.func.instruction(&Instruction::If(BlockType::Empty));
                                    {
                                        // n1=3, n2=1
                                        emit_store_page0_chunk(this, 0, 2, 0);
                                        emit_store_page0_chunk(this, 2, 1, 16);
                                        emit_store_page1_chunk(this, 0, 1, 24);
                                    }
                                    this.func.instruction(&Instruction::Else);
                                    {
                                        this.func.instruction(&Instruction::LocalGet(shift_local));
                                        this.func.instruction(&Instruction::I64Const(2));
                                        this.func.instruction(&Instruction::I64Eq);
                                        this.func.instruction(&Instruction::If(BlockType::Empty));
                                        {
                                            // n1=2, n2=2
                                            emit_store_page0_chunk(this, 0, 2, 0);
                                            emit_store_page1_chunk(this, 0, 2, 16);
                                        }
                                        this.func.instruction(&Instruction::Else);
                                        {
                                            // n1=1, n2=3
                                            emit_store_page0_chunk(this, 0, 1, 0);
                                            emit_store_page1_chunk(this, 0, 2, 8);
                                            emit_store_page1_chunk(this, 2, 1, 24);
                                        }
                                        this.func.instruction(&Instruction::End);
                                    }
                                    this.func.instruction(&Instruction::End);
                                }
                                Width::W64 => {
                                    this.func.instruction(&Instruction::LocalGet(shift_local));
                                    this.func.instruction(&Instruction::I64Const(1));
                                    this.func.instruction(&Instruction::I64Eq);
                                    this.func.instruction(&Instruction::If(BlockType::Empty));
                                    {
                                        // n1=7, n2=1
                                        emit_store_page0_chunk(this, 0, 4, 0);
                                        emit_store_page0_chunk(this, 4, 2, 32);
                                        emit_store_page0_chunk(this, 6, 1, 48);
                                        emit_store_page1_chunk(this, 0, 1, 56);
                                    }
                                    this.func.instruction(&Instruction::Else);
                                    {
                                        this.func.instruction(&Instruction::LocalGet(shift_local));
                                        this.func.instruction(&Instruction::I64Const(2));
                                        this.func.instruction(&Instruction::I64Eq);
                                        this.func.instruction(&Instruction::If(BlockType::Empty));
                                        {
                                            // n1=6, n2=2
                                            emit_store_page0_chunk(this, 0, 4, 0);
                                            emit_store_page0_chunk(this, 4, 2, 32);
                                            emit_store_page1_chunk(this, 0, 2, 48);
                                        }
                                        this.func.instruction(&Instruction::Else);
                                        {
                                            this.func
                                                .instruction(&Instruction::LocalGet(shift_local));
                                            this.func.instruction(&Instruction::I64Const(3));
                                            this.func.instruction(&Instruction::I64Eq);
                                            this.func
                                                .instruction(&Instruction::If(BlockType::Empty));
                                            {
                                                // n1=5, n2=3
                                                emit_store_page0_chunk(this, 0, 4, 0);
                                                emit_store_page0_chunk(this, 4, 1, 32);
                                                emit_store_page1_chunk(this, 0, 2, 40);
                                                emit_store_page1_chunk(this, 2, 1, 56);
                                            }
                                            this.func.instruction(&Instruction::Else);
                                            {
                                                this.func.instruction(&Instruction::LocalGet(
                                                    shift_local,
                                                ));
                                                this.func.instruction(&Instruction::I64Const(4));
                                                this.func.instruction(&Instruction::I64Eq);
                                                this.func.instruction(&Instruction::If(
                                                    BlockType::Empty,
                                                ));
                                                {
                                                    // n1=4, n2=4
                                                    emit_store_page0_chunk(this, 0, 4, 0);
                                                    emit_store_page1_chunk(this, 0, 4, 32);
                                                }
                                                this.func.instruction(&Instruction::Else);
                                                {
                                                    this.func.instruction(&Instruction::LocalGet(
                                                        shift_local,
                                                    ));
                                                    this.func
                                                        .instruction(&Instruction::I64Const(5));
                                                    this.func.instruction(&Instruction::I64Eq);
                                                    this.func.instruction(&Instruction::If(
                                                        BlockType::Empty,
                                                    ));
                                                    {
                                                        // n1=3, n2=5
                                                        emit_store_page0_chunk(this, 0, 2, 0);
                                                        emit_store_page0_chunk(this, 2, 1, 16);
                                                        emit_store_page1_chunk(this, 0, 4, 24);
                                                        emit_store_page1_chunk(this, 4, 1, 56);
                                                    }
                                                    this.func.instruction(&Instruction::Else);
                                                    {
                                                        this.func.instruction(
                                                            &Instruction::LocalGet(shift_local),
                                                        );
                                                        this.func
                                                            .instruction(&Instruction::I64Const(6));
                                                        this.func.instruction(&Instruction::I64Eq);
                                                        this.func.instruction(&Instruction::If(
                                                            BlockType::Empty,
                                                        ));
                                                        {
                                                            // n1=2, n2=6
                                                            emit_store_page0_chunk(this, 0, 2, 0);
                                                            emit_store_page1_chunk(this, 0, 4, 16);
                                                            emit_store_page1_chunk(this, 4, 2, 48);
                                                        }
                                                        this.func.instruction(&Instruction::Else);
                                                        {
                                                            // n1=1, n2=7
                                                            emit_store_page0_chunk(this, 0, 1, 0);
                                                            emit_store_page1_chunk(this, 0, 4, 8);
                                                            emit_store_page1_chunk(this, 4, 2, 40);
                                                            emit_store_page1_chunk(this, 6, 1, 56);
                                                        }
                                                        this.func.instruction(&Instruction::End);
                                                    }
                                                    this.func.instruction(&Instruction::End);
                                                }
                                                this.func.instruction(&Instruction::End);
                                            }
                                            this.func.instruction(&Instruction::End);
                                        }
                                        this.func.instruction(&Instruction::End);
                                    }
                                    this.func.instruction(&Instruction::End);
                                }
                                _ => unreachable!(),
                            }

                            // Self-modifying code invalidation: bump the version entry for both written
                            // physical pages.
                            //
                            // Re-translate each page to get the correct physical base for the bump.
                            this.func.instruction(&Instruction::LocalGet(
                                this.layout.value_local(*addr),
                            ));
                            this.func.instruction(&Instruction::LocalSet(
                                this.layout.scratch_vaddr_local(),
                            ));
                            this.emit_translate_and_cache(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);
                            this.emit_bump_code_version_fastpath();

                            this.func.instruction(&Instruction::LocalGet(
                                this.layout.value_local(*addr),
                            ));
                            this.func
                                .instruction(&Instruction::I64Const(size_bytes as i64));
                            this.func.instruction(&Instruction::I64Add);
                            this.func
                                .instruction(&Instruction::LocalGet(this.layout.scratch_local()));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::LocalSet(
                                this.layout.scratch_vaddr_local(),
                            ));
                            this.emit_translate_and_cache(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);
                            this.emit_bump_code_version_fastpath();
                        };

                        // Page 0: translate.
                        self.emit_translate_and_cache(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);

                        if self.options.inline_tlb_mmio_exit {
                            self.emit_mmio_exit(size_bytes, 1, Some(*src));

                            // Translate page 1 (but use original vaddr in the exit payload).
                            self.func.instruction(&Instruction::LocalGet(
                                self.layout.value_local(*addr),
                            ));
                            self.func
                                .instruction(&Instruction::I64Const(size_bytes as i64));
                            self.func.instruction(&Instruction::I64Add);
                            self.func
                                .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
                            self.func.instruction(&Instruction::I64Sub);
                            self.func.instruction(&Instruction::LocalSet(
                                self.layout.scratch_vaddr_local(),
                            ));

                            self.emit_translate_and_cache(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);

                            self.func.instruction(&Instruction::LocalGet(
                                self.layout.value_local(*addr),
                            ));
                            self.func.instruction(&Instruction::LocalSet(
                                self.layout.scratch_vaddr_local(),
                            ));
                            self.emit_mmio_exit(size_bytes, 1, Some(*src));

                            emit_split_store_and_bump(self);
                        } else {
                            // If either page isn't backed by RAM, fall back to the slow helper and
                            // keep executing this block.
                            self.func.instruction(&Instruction::LocalGet(
                                self.layout.scratch_tlb_data_local(),
                            ));
                            self.func
                                .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
                            self.func.instruction(&Instruction::I64And);
                            self.func.instruction(&Instruction::I64Eqz);

                            self.func.instruction(&Instruction::If(BlockType::Empty));
                            self.depth += 1;
                            {
                                // Slow path.
                                let slow_write =
                                    slow_write.expect("memory write helper import missing");
                                self.emit_slow_mem_write(*src, *width, slow_write);
                            }
                            self.func.instruction(&Instruction::Else);
                            {
                                // Translate page 1.
                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.value_local(*addr),
                                ));
                                self.func
                                    .instruction(&Instruction::I64Const(size_bytes as i64));
                                self.func.instruction(&Instruction::I64Add);
                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.scratch_local(),
                                ));
                                self.func.instruction(&Instruction::I64Sub);
                                self.func.instruction(&Instruction::LocalSet(
                                    self.layout.scratch_vaddr_local(),
                                ));

                                self.emit_translate_and_cache(
                                    MMU_ACCESS_WRITE,
                                    crate::TLB_FLAG_WRITE,
                                );

                                self.func.instruction(&Instruction::LocalGet(
                                    self.layout.scratch_tlb_data_local(),
                                ));
                                self.func.instruction(&Instruction::I64Const(
                                    crate::TLB_FLAG_IS_RAM as i64,
                                ));
                                self.func.instruction(&Instruction::I64And);
                                self.func.instruction(&Instruction::I64Eqz);

                                self.func.instruction(&Instruction::If(BlockType::Empty));
                                self.depth += 1;
                                {
                                    // Slow path (restore original vaddr first).
                                    self.func.instruction(&Instruction::LocalGet(
                                        self.layout.value_local(*addr),
                                    ));
                                    self.func.instruction(&Instruction::LocalSet(
                                        self.layout.scratch_vaddr_local(),
                                    ));

                                    let slow_write =
                                        slow_write.expect("memory write helper import missing");
                                    self.emit_slow_mem_write(*src, *width, slow_write);
                                }
                                self.func.instruction(&Instruction::Else);
                                {
                                    emit_split_store_and_bump(self);
                                }
                                self.func.instruction(&Instruction::End);
                                self.depth -= 1;
                            }
                            self.func.instruction(&Instruction::End);
                            self.depth -= 1;
                        }
                    } else {
                        // Slow path.
                        let slow_write = slow_write.expect("memory write helper import missing");
                        self.emit_slow_mem_write(*src, *width, slow_write);
                    }
                }
                self.func.instruction(&Instruction::Else);
                {
                    emit_same_page_store(self);
                }
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }
            IrInst::BinOp {
                dst,
                op,
                lhs,
                rhs,
                width,
                flags,
            } => {
                match *op {
                    BinOp::Sar => {
                        self.emit_sar(*width, *lhs, *rhs);
                    }
                    _ => {
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.value_local(*lhs)));
                        self.func
                            .instruction(&Instruction::LocalGet(self.layout.value_local(*rhs)));
                        match *op {
                            BinOp::Add => {
                                self.func.instruction(&Instruction::I64Add);
                            }
                            BinOp::Adc => {
                                self.func.instruction(&Instruction::I64Add);
                                self.emit_read_flag(Flag::Cf);
                                self.func.instruction(&Instruction::I64ExtendI32U);
                                // Keep it: CF may be overwritten long before this
                                // operation's own flags are asked for.
                                self.func.instruction(&Instruction::LocalTee(
                                    self.layout.lazy_carry_local(),
                                ));
                                self.func.instruction(&Instruction::I64Add);
                            }
                            BinOp::Sub => {
                                self.func.instruction(&Instruction::I64Sub);
                            }
                            BinOp::Sbb => {
                                self.func.instruction(&Instruction::I64Sub);
                                self.emit_read_flag(Flag::Cf);
                                self.func.instruction(&Instruction::I64ExtendI32U);
                                // See `Adc`.
                                self.func.instruction(&Instruction::LocalTee(
                                    self.layout.lazy_carry_local(),
                                ));
                                self.func.instruction(&Instruction::I64Sub);
                            }
                            BinOp::And => {
                                self.func.instruction(&Instruction::I64And);
                            }
                            BinOp::Or => {
                                self.func.instruction(&Instruction::I64Or);
                            }
                            BinOp::Xor => {
                                self.func.instruction(&Instruction::I64Xor);
                            }
                            BinOp::Shl => {
                                self.emit_shift_mask(*width);
                                self.func.instruction(&Instruction::I64Shl);
                            }
                            BinOp::Shr => {
                                self.emit_shift_mask(*width);
                                self.func.instruction(&Instruction::I64ShrU);
                            }
                            BinOp::Sar => unreachable!(),
                            BinOp::Rol | BinOp::Ror => {
                                self.emit_rotate(*op, *width);
                            }
                            BinOp::Bswap => {
                                // lhs is already on the stack; drop rhs and swap bytes.
                                self.func.instruction(&Instruction::Drop);
                                self.emit_bswap(*width);
                            }
                            BinOp::Mul => {
                                self.func.instruction(&Instruction::I64Mul);
                            }
                        };
                        self.emit_trunc(*width);
                    }
                }

                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));

                if !flags.is_empty() {
                    match *op {
                        BinOp::Add | BinOp::Adc => self.emit_add_flags(
                            *width,
                            *flags,
                            *lhs,
                            *rhs,
                            *dst,
                            matches!(*op, BinOp::Adc),
                        ),
                        BinOp::Sub | BinOp::Sbb => self.emit_sub_flags(
                            *width,
                            *flags,
                            *lhs,
                            *rhs,
                            *dst,
                            matches!(*op, BinOp::Sbb),
                        ),
                        BinOp::And | BinOp::Or | BinOp::Xor => {
                            self.emit_logic_flags(*width, *flags, *dst)
                        }
                        BinOp::Shl | BinOp::Shr | BinOp::Sar => {
                            self.emit_shift_flags(*op, *width, *flags, *lhs, *rhs, *dst)
                        }
                        BinOp::Rol | BinOp::Ror => {
                            self.emit_rotate_flags(*op, *width, *flags, *rhs, *dst)
                        }
                        BinOp::Mul => self.emit_mul_flags(*width, *flags, *lhs, *rhs, *dst),
                        BinOp::Bswap => {
                            // BSWAP updates no flags, and the IR interpreter agrees.
                        }
                    }
                }
            }
            IrInst::CmpFlags {
                lhs,
                rhs,
                width,
                flags,
            } => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*lhs)));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*rhs)));
                self.func.instruction(&Instruction::I64Sub);
                self.emit_trunc(*width);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
                self.emit_sub_flags_with_res(
                    *width,
                    *flags,
                    *lhs,
                    *rhs,
                    self.layout.scratch_local(),
                );
            }
            IrInst::TestFlags {
                lhs,
                rhs,
                width,
                flags,
            } => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*lhs)));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*rhs)));
                self.func.instruction(&Instruction::I64And);
                self.emit_trunc(*width);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
                self.emit_logic_flags_with_res(*width, *flags, self.layout.scratch_local());
            }
            IrInst::EvalCond { dst, cond } => {
                self.emit_eval_cond(*cond);
                self.func.instruction(&Instruction::I64ExtendI32U);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::Select {
                dst,
                cond,
                if_true,
                if_false,
                width,
            } => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*if_true)));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*if_false)));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(*cond)));
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
                self.func.instruction(&Instruction::Select);
                self.emit_trunc(*width);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::BitScan {
                dst,
                src,
                width,
                forward,
            } => {
                // `i64.ctz` / `i64.clz` on the zero-extended value. That is right for a
                // 32-bit operand as well as a 64-bit one, and `63 - clz` is exactly
                // what Tier-0's `exec_bsf_bsr` computes, on the same u64.
                if *forward {
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*src)));
                    self.func.instruction(&Instruction::I64Ctz);
                } else {
                    self.func.instruction(&Instruction::I64Const(63));
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.value_local(*src)));
                    self.func.instruction(&Instruction::I64Clz);
                    self.func.instruction(&Instruction::I64Sub);
                }
                self.emit_trunc(*width);
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::Assist { rip, next_rip } => {
                self.emit_assist(*rip, *next_rip);
            }
            IrInst::ReadTsc { dst } => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
                self.func.instruction(&Instruction::Call(
                    self.imported.rdtsc.expect("rdtsc import missing"),
                ));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.value_local(*dst)));
            }
            IrInst::CallHelper { .. } => {
                // Tier-1 blocks currently do not have deopt metadata for resuming mid-block. Treat
                // helper calls as a runtime bailout and request a one-shot interpreter fallback.
                //
                // Higher-level compilation helpers may reject helper calls up-front, but keep the
                // codegen itself defensive so direct users of `Tier1WasmCodegen` don't hit a hard
                // panic at compile time.
                //
                // Mark this as a runtime exit so the backend can roll back state if needed.
                // `kind` is currently unused by the host stub helpers, so use 0.
                self.func.instruction(&Instruction::I32Const(0));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.rip_local()));
                self.func.instruction(&Instruction::Call(
                    self.imported.jit_exit.expect("jit_exit import missing"),
                ));
                // `jit_exit` returns the RIP to resume at while we use the sentinel return value to
                // request an interpreter step.
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.next_rip_local()));
                self.func
                    .instruction(&Instruction::I64Const(JIT_EXIT_SENTINEL_I64));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
                self.func.instruction(&Instruction::Br(self.depth));
            }
        }
    }

    fn emit_terminator(&mut self, term: &IrTerminator) {
        match *term {
            IrTerminator::Jump { target } => {
                self.func.instruction(&Instruction::I64Const(target as i64));
            }
            IrTerminator::CondJump {
                cond,
                target,
                fallthrough,
            } => {
                self.func.instruction(&Instruction::I64Const(target as i64));
                self.func
                    .instruction(&Instruction::I64Const(fallthrough as i64));
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(cond)));
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
                self.func.instruction(&Instruction::Select);
            }
            IrTerminator::IndirectJump { target } => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(target)));
            }
            IrTerminator::ExitToInterpreter { next_rip } => {
                self.func
                    .instruction(&Instruction::I64Const(next_rip as i64));
            }
        }
        self.func
            .instruction(&Instruction::LocalSet(self.layout.next_rip_local()));

        // Encode `exit_to_interpreter` in the return value while still updating `CpuState.rip`
        // in linear memory:
        // - normal control flow returns the computed `next_rip`
        // - `ExitToInterpreter` returns `JIT_EXIT_SENTINEL_I64` and the runtime reads the real
        //   `next_rip` from `CpuState.rip`
        match *term {
            IrTerminator::ExitToInterpreter { .. } => {
                self.func
                    .instruction(&Instruction::I64Const(JIT_EXIT_SENTINEL_I64));
            }
            _ => {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.next_rip_local()));
            }
        }
        self.func
            .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
    }

    /// See `inline_tlb_codegen::emit_probe_and_fill`.
    fn emit_probe_and_fill(&mut self, access_code: i32, required_flag: u64) {
        debug_assert!(self.options.inline_tlb);
        let locals = self.inline_tlb_locals();
        let mmu_translate = self
            .imported
            .mmu_translate
            .expect("mmu_translate import missing");
        inline_tlb_codegen::emit_probe_and_fill(
            self.func,
            &mut self.depth,
            locals,
            mmu_translate,
            access_code,
            required_flag,
            self.layout.tlb_entry_local(),
        );
    }

    fn emit_translate_and_cache(&mut self, access_code: i32, required_flag: u64) {
        debug_assert!(self.options.inline_tlb);
        let locals = self.inline_tlb_locals();
        let mmu_translate = self
            .imported
            .mmu_translate
            .expect("mmu_translate import missing");
        inline_tlb_codegen::emit_translate_and_cache(
            self.func,
            &mut self.depth,
            locals,
            mmu_translate,
            access_code,
            required_flag,
        );
    }

    fn emit_mmio_exit(&mut self, size_bytes: u32, is_write: i32, value: Option<ValueId>) {
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_tlb_data_local()));
        self.func
            .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Eqz);

        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
            self.func
                .instruction(&Instruction::I32Const(size_bytes as i32));
            self.func.instruction(&Instruction::I32Const(is_write));
            if let Some(value) = value {
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.value_local(value)));
            } else {
                self.func.instruction(&Instruction::I64Const(0));
            }
            self.func
                .instruction(&Instruction::LocalGet(self.layout.rip_local()));
            self.func.instruction(&Instruction::Call(
                self.imported
                    .jit_exit_mmio
                    .expect("jit_exit_mmio import missing"),
            ));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.next_rip_local()));
            // MMIO exits are runtime exits; return the sentinel and let the runtime read the
            // concrete `next_rip` from `CpuState.rip`.
            self.func
                .instruction(&Instruction::I64Const(JIT_EXIT_SENTINEL_I64));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.scratch_local()));
            self.func.instruction(&Instruction::Br(self.depth));
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }

    /// One inline fast path for a store; see [`Self::emit_load_single_fast_path`].
    fn emit_store_single_fast_path(&mut self, src: ValueId, addr: ValueId, width: Width) {
        let (size_bytes, slow_write) = match width {
            Width::W8 => (1u32, self.imported.mem_write_u8),
            Width::W16 => (2u32, self.imported.mem_write_u16),
            Width::W32 => (4u32, self.imported.mem_write_u32),
            Width::W64 => (8u32, self.imported.mem_write_u64),
        };
        let slow_write = slow_write.expect("slow write helper missing for single-fast-path store");

        self.func
            .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
        self.func
            .instruction(&Instruction::LocalSet(self.layout.scratch_vaddr_local()));

        self.emit_probe_and_fill(MMU_ACCESS_WRITE, crate::TLB_FLAG_WRITE);

        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_tlb_data_local()));
        self.func
            .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::I64Ne);

        if size_bytes > 1 {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
            self.func
                .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
            self.func.instruction(&Instruction::I64And);
            self.func.instruction(&Instruction::I64Const(
                (crate::PAGE_OFFSET_MASK - (size_bytes as u64 - 1)) as i64,
            ));
            self.func.instruction(&Instruction::I64LeU);
            self.func.instruction(&Instruction::I32And);
        }

        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            self.emit_compute_ram_addr();
            self.func
                .instruction(&Instruction::LocalGet(self.layout.value_local(src)));
            match width {
                Width::W8 => self.func.instruction(&Instruction::I64Store8(memarg(0, 0))),
                Width::W16 => self.func.instruction(&Instruction::I64Store16(memarg(0, 1))),
                Width::W32 => self.func.instruction(&Instruction::I64Store32(memarg(0, 2))),
                Width::W64 => self.func.instruction(&Instruction::I64Store(memarg(0, 3))),
            };
            // A guest write to a page holding translated code has to invalidate it.
            self.emit_bump_code_version_fastpath();
        }
        self.func.instruction(&Instruction::Else);
        {
            self.emit_slow_mem_write(src, width, slow_write);
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }

    /// One inline fast path for a load; everything else is a call.
    ///
    /// The predicate is "the TLB knows this page, it is backed by RAM, and the access
    /// stays inside it". Anything else — a miss the probe could not fill, a device
    /// register, an access straddling a page — goes to the imported helper, which
    /// already handles all three and reaches non-RAM through the embedder's bus. That
    /// replaces three nested branches per access, with the same-page sequence
    /// duplicated inside two of them.
    fn emit_load_single_fast_path(&mut self, dst: ValueId, addr: ValueId, width: Width) {
        if let Some(helper) = self.shared_loads.get(width) {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.jit_ctx_ptr_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.ram_base_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.tlb_salt_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
            self.func.instruction(&Instruction::Call(helper));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.value_local(dst)));
            return;
        }
        let (size_bytes, slow_read) = match width {
            Width::W8 => (1u32, self.imported.mem_read_u8),
            Width::W16 => (2u32, self.imported.mem_read_u16),
            Width::W32 => (4u32, self.imported.mem_read_u32),
            Width::W64 => (8u32, self.imported.mem_read_u64),
        };
        let slow_read = slow_read.expect("slow read helper missing for single-fast-path load");

        self.func
            .instruction(&Instruction::LocalGet(self.layout.value_local(addr)));
        self.func
            .instruction(&Instruction::LocalSet(self.layout.scratch_vaddr_local()));

        // Probe, and fill the entry on a miss so the next access to this page hits.
        // The permission bit joins the predicate below rather than getting its own
        // re-translate, which a hit never reaches.
        self.emit_probe_and_fill(MMU_ACCESS_READ, crate::TLB_FLAG_READ);

        // backed by RAM ...
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_tlb_data_local()));
        self.func
            .instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::I64Ne);

        // ... and inside one page. An access of one byte never crosses.
        if size_bytes > 1 {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
            self.func
                .instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
            self.func.instruction(&Instruction::I64And);
            self.func.instruction(&Instruction::I64Const(
                (crate::PAGE_OFFSET_MASK - (size_bytes as u64 - 1)) as i64,
            ));
            self.func.instruction(&Instruction::I64LeU);
            self.func.instruction(&Instruction::I32And);
        }

        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            self.emit_compute_ram_addr();
            match width {
                Width::W8 => self.func.instruction(&Instruction::I64Load8U(memarg(0, 0))),
                Width::W16 => self.func.instruction(&Instruction::I64Load16U(memarg(0, 1))),
                Width::W32 => self.func.instruction(&Instruction::I64Load32U(memarg(0, 2))),
                Width::W64 => self.func.instruction(&Instruction::I64Load(memarg(0, 3))),
            };
            self.emit_trunc(width);
            self.func
                .instruction(&Instruction::LocalSet(self.layout.value_local(dst)));
        }
        self.func.instruction(&Instruction::Else);
        {
            self.emit_slow_mem_read(dst, width, slow_read);
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }

    fn emit_slow_mem_read(&mut self, dst: ValueId, width: Width, slow_read: u32) {
        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
        self.func.instruction(&Instruction::Call(slow_read));
        if width != Width::W64 {
            self.func.instruction(&Instruction::I64ExtendI32U);
        }
        self.emit_trunc(width);
        self.func
            .instruction(&Instruction::LocalSet(self.layout.value_local(dst)));
    }

    fn emit_slow_mem_write(&mut self, src: ValueId, width: Width, slow_write: u32) {
        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_vaddr_local()));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.value_local(src)));
        if width != Width::W64 {
            self.emit_trunc(width);
            self.func.instruction(&Instruction::I32WrapI64);
        }
        self.func.instruction(&Instruction::Call(slow_write));
    }

    /// Computes the linear-memory address for the current `{vaddr, tlb_data}` pair and leaves it
    /// on the stack as an `i32` suitable for a WASM `load/store`.
    fn emit_compute_ram_addr(&mut self) {
        let locals = self.inline_tlb_locals();
        inline_tlb_codegen::emit_compute_ram_addr(self.func, locals);
    }

    /// Bumps the page-version entry for the current RAM write (inline fast-path stores only).
    ///
    /// The runtime may choose to only bump for pages marked as executable/code, but for initial
    /// correctness we bump for all writes that hit RAM.
    fn emit_bump_code_version_fastpath(&mut self) {
        let locals = self.inline_tlb_locals();
        inline_tlb_codegen::emit_bump_code_version_fastpath(
            self.func,
            locals,
            self.options.memory_shared,
        );
    }

    /// Hand one guest instruction to the host and carry on with what it left behind.
    ///
    /// Everything the block holds in locals has to be in `CpuState` before the call —
    /// the host works from linear memory — and everything has to come back after,
    /// because an assisted instruction may write any register. Sixteen registers and
    /// the flags each way is not cheap, but an assist happens a few times in a
    /// million guest instructions, and the alternative is ending the block.
    fn emit_assist(&mut self, rip: u64, next_rip: u64) {
        // Flags first: the record has to be settled before the host sees them, and
        // whatever the host leaves is what the rest of the block reads.
        self.materialize_pending();
        self.func.instruction(&Instruction::I64Const(rip as i64));
        self.func
            .instruction(&Instruction::LocalSet(self.layout.rip_local()));
        self.emit_spill_all_state();

        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func.instruction(&Instruction::I64Const(rip as i64));
        self.func.instruction(&Instruction::Call(
            self.imported.assist.expect("assist import missing"),
        ));
        // Keep the answer: it is both the test and, if the test fails, the value the
        // block leaves with.
        self.func
            .instruction(&Instruction::LocalTee(self.layout.scratch_local()));

        // Did the guest end up where the rest of this block assumes?
        let started = self.guest_insts_started;
        self.func.instruction(&Instruction::I64Const(next_rip as i64));
        self.func.instruction(&Instruction::I64Ne);
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            // No. Two different noes: the sentinel means the instruction did not
            // happen — a fault, a halt — and the block has retired only what came
            // before it. Any other answer means it happened and went somewhere else,
            // which retires it. The machine's timing follows this count, so the
            // difference of one matters.
            self.func
                .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
            self.func
                .instruction(&Instruction::I64Const(crate::wasm::JIT_EXIT_SENTINEL_I64));
            self.func.instruction(&Instruction::I64Eq);
            self.func.instruction(&Instruction::If(BlockType::Empty));
            self.depth += 1;
            {
                self.func.instruction(&Instruction::I64Const(started as i64));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.retired_local()));
            }
            self.func.instruction(&Instruction::Else);
            {
                self.func
                    .instruction(&Instruction::I64Const(started as i64 + 1));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.retired_local()));
                // RIP is where the host says, so the spill leaves the right value.
                self.func
                    .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
                self.func
                    .instruction(&Instruction::LocalSet(self.layout.rip_local()));
            }
            self.func.instruction(&Instruction::End);
            self.depth -= 1;

            // The block's single exit takes its answer from `scratch_local`, which
            // already holds it, and `next_rip_local` alongside — and branches with
            // *nothing* on the stack. Leaving a value there instead produces WASM that
            // does not validate, which shows up as almost no blocks being admitted
            // rather than as an error.
            self.func
                .instruction(&Instruction::LocalGet(self.layout.rip_local()));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.next_rip_local()));
            self.func.instruction(&Instruction::Br(self.depth));
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;

        // Yes: pick the state back up and keep going.
        self.emit_reload_all_state();
        self.guest_insts_started += 1;
    }

    /// Write every guest register and the flags to `CpuState`.
    fn emit_spill_all_state(&mut self) {
        for gpr in all_gprs() {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
            self.func
                .instruction(&Instruction::LocalGet(self.layout.gpr_local(gpr)));
            self.func.instruction(&Instruction::I64Store(memarg(
                abi::CPU_GPR_OFF[gpr.as_u8() as usize],
                3,
            )));
        }
        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func
            .instruction(&Instruction::I64Store(memarg(abi::CPU_RFLAGS_OFF, 3)));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func
            .instruction(&Instruction::LocalGet(self.layout.rip_local()));
        self.func
            .instruction(&Instruction::I64Store(memarg(abi::CPU_RIP_OFF, 3)));
    }

    /// Read every guest register and the flags back from `CpuState`.
    fn emit_reload_all_state(&mut self) {
        for gpr in all_gprs() {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
            self.func.instruction(&Instruction::I64Load(memarg(
                abi::CPU_GPR_OFF[gpr.as_u8() as usize],
                3,
            )));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.gpr_local(gpr)));
        }
        self.func
            .instruction(&Instruction::LocalGet(self.layout.cpu_ptr_local()));
        self.func
            .instruction(&Instruction::I64Load(memarg(abi::CPU_RFLAGS_OFF, 3)));
        self.func
            .instruction(&Instruction::LocalSet(self.layout.rflags_local()));

        // And the TLB salt. An assisted instruction can write CR3, CR0, CR4 or EFER,
        // and then every inline-TLB entry this block filled describes the old
        // translation. The host bumps the salt when that happens, which makes every
        // existing tag mismatch — the same thing a flush does — but only if the block
        // stops using the salt it loaded at entry.
        if self.options.inline_tlb {
            self.func
                .instruction(&Instruction::LocalGet(self.layout.jit_ctx_ptr_local()));
            self.func.instruction(&Instruction::I64Load(memarg(
                crate::jit_ctx::JitContext::TLB_SALT_OFFSET,
                3,
            )));
            self.func
                .instruction(&Instruction::LocalSet(self.layout.tlb_salt_local()));
        }
    }

    fn emit_trunc(&mut self, width: Width) {
        if width == Width::W64 {
            return;
        }
        self.func
            .instruction(&Instruction::I64Const(width.mask() as i64));
        self.func.instruction(&Instruction::I64And);
    }

    fn emit_bswap_step(&mut self, mask: u64, shift: i64) {
        let tmp = self.layout.scratch_local();
        self.func.instruction(&Instruction::LocalTee(tmp));
        self.func.instruction(&Instruction::I64Const(mask as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Const(shift));
        self.func.instruction(&Instruction::I64Shl);
        self.func.instruction(&Instruction::LocalGet(tmp));
        self.func.instruction(&Instruction::I64Const(shift));
        self.func.instruction(&Instruction::I64ShrU);
        self.func.instruction(&Instruction::I64Const(mask as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Or);
    }

    fn emit_bswap(&mut self, width: Width) {
        match width {
            Width::W8 => {}
            Width::W16 => self.emit_bswap_step(0x00ff, 8),
            Width::W32 => {
                self.emit_bswap_step(0x0000_ffff, 16);
                self.emit_bswap_step(0x00ff_00ff, 8);
            }
            Width::W64 => {
                self.emit_bswap_step(0x0000_0000_ffff_ffff, 32);
                self.emit_bswap_step(0x0000_ffff_0000_ffff, 16);
                self.emit_bswap_step(0x00ff_00ff_00ff_00ff, 8);
            }
        }
    }

    fn emit_shift_mask(&mut self, width: Width) {
        // x86 shifts mask the count to 5 bits for 8/16/32-bit operands and 6 bits for 64-bit.
        // Note: this differs from WASM's built-in masking (which always uses 6 bits for i64
        // shifts), so we must apply the x86 mask explicitly for narrow widths.
        let mask = if width == Width::W64 { 63 } else { 31 };
        self.func.instruction(&Instruction::I64Const(mask));
        self.func.instruction(&Instruction::I64And);
    }

    /// Rotate within `width`. `i64.rotr` of a zero-extended 32-bit value is not
    /// `ror r32`: zeros rotate into the low half (`0x12345678 ror 8` becomes
    /// `0x00123456` instead of `0x78123456`). That is the SHA-256 leaf.
    fn emit_rotate(&mut self, op: BinOp, width: Width) {
        // stack: val, count
        match width {
            Width::W64 => {
                self.emit_shift_mask(width);
                if matches!(op, BinOp::Rol) {
                    self.func.instruction(&Instruction::I64Rotl);
                } else {
                    self.func.instruction(&Instruction::I64Rotr);
                }
            }
            Width::W32 => {
                self.emit_shift_mask(width);
                let count = self.layout.scratch_local();
                self.func.instruction(&Instruction::LocalSet(count));
                self.func.instruction(&Instruction::I32WrapI64);
                self.func.instruction(&Instruction::LocalGet(count));
                self.func.instruction(&Instruction::I32WrapI64);
                if matches!(op, BinOp::Rol) {
                    self.func.instruction(&Instruction::I32Rotl);
                } else {
                    self.func.instruction(&Instruction::I32Rotr);
                }
                self.func.instruction(&Instruction::I64ExtendI32U);
            }
            Width::W16 | Width::W8 => {
                let bits = width.bits() as i64;
                self.emit_shift_mask(width);
                self.func.instruction(&Instruction::I64Const(bits - 1));
                self.func.instruction(&Instruction::I64And);
                let count = self.layout.scratch_local();
                let val = self.layout.scratch_vaddr_local();
                self.func.instruction(&Instruction::LocalSet(count));
                self.emit_trunc(width);
                self.func.instruction(&Instruction::LocalSet(val));
                self.func.instruction(&Instruction::LocalGet(val));
                self.func.instruction(&Instruction::LocalGet(count));
                let (first, second) = match op {
                    BinOp::Rol => (Instruction::I64Shl, Instruction::I64ShrU),
                    _ => (Instruction::I64ShrU, Instruction::I64Shl),
                };
                self.func.instruction(&first);
                self.func.instruction(&Instruction::LocalGet(val));
                self.func.instruction(&Instruction::I64Const(bits));
                self.func.instruction(&Instruction::LocalGet(count));
                self.func.instruction(&Instruction::I64Sub);
                self.func.instruction(&second);
                self.func.instruction(&Instruction::I64Or);
                self.emit_trunc(width);
            }
        }
    }

    /// Zero-extend the low 32 bits of the value on the stack.
    ///
    /// `i32.wrap_i64` + `i64.extend_i32_u` rather than `& 0xffff_ffff`. The two mean the
    /// same thing, and Cranelift compiles them very differently: the mask needs the
    /// 64-bit constant `0xffff_ffff`, which it materialises out of the constant pool —
    /// `and 0x78(%rip), %r11`, a memory operand — while the wrap/extend pair is a 32-bit
    /// move, which x86-64 zero-extends for free and which folds away entirely when the
    /// producer already wrote a 32-bit result.
    ///
    /// 32-bit operands are most of what a Windows guest executes, so this is on the
    /// hottest path there is: a four-instruction `add eax` block was emitting five of
    /// these. `AERO_JIT_MASK32=1` restores the mask for comparison.
    fn emit_truncate32(&mut self) {
        if mask32_enabled() {
            self.func
                .instruction(&Instruction::I64Const(0xffff_ffffu64 as i64));
            self.func.instruction(&Instruction::I64And);
        } else {
            self.func.instruction(&Instruction::I32WrapI64);
            self.func.instruction(&Instruction::I64ExtendI32U);
        }
    }

    fn emit_read_gpr_part(&mut self, reg: Gpr, width: Width, high8: bool) {
        self.func
            .instruction(&Instruction::LocalGet(self.layout.gpr_local(reg)));
        match (width, high8) {
            (Width::W64, false) => {}
            (Width::W32, false) => {
                self.emit_truncate32();
            }
            (Width::W16, false) => {
                self.func.instruction(&Instruction::I64Const(0xffff));
                self.func.instruction(&Instruction::I64And);
            }
            (Width::W8, false) => {
                self.func.instruction(&Instruction::I64Const(0xff));
                self.func.instruction(&Instruction::I64And);
            }
            (Width::W8, true) => {
                self.func.instruction(&Instruction::I64Const(8));
                self.func.instruction(&Instruction::I64ShrU);
                self.func.instruction(&Instruction::I64Const(0xff));
                self.func.instruction(&Instruction::I64And);
            }
            _ => unreachable!("invalid gpr part access: {width:?} high8={high8}"),
        }
    }

    fn emit_write_gpr_part(&mut self, reg: Gpr, width: Width, high8: bool, src: ValueId) {
        let dst_local = self.layout.gpr_local(reg);
        let src_local = self.layout.value_local(src);
        match (width, high8) {
            (Width::W64, false) => {
                self.func.instruction(&Instruction::LocalGet(src_local));
                self.func.instruction(&Instruction::LocalSet(dst_local));
            }
            (Width::W32, false) => {
                self.func.instruction(&Instruction::LocalGet(src_local));
                self.emit_truncate32();
                self.func.instruction(&Instruction::LocalSet(dst_local));
            }
            (Width::W16, false) => {
                self.func.instruction(&Instruction::LocalGet(dst_local));
                self.func.instruction(&Instruction::I64Const(!0xffffi64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::LocalGet(src_local));
                self.func.instruction(&Instruction::I64Const(0xffff));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Or);
                self.func.instruction(&Instruction::LocalSet(dst_local));
            }
            (Width::W8, false) => {
                self.func.instruction(&Instruction::LocalGet(dst_local));
                self.func.instruction(&Instruction::I64Const(!0xffi64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::LocalGet(src_local));
                self.func.instruction(&Instruction::I64Const(0xff));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Or);
                self.func.instruction(&Instruction::LocalSet(dst_local));
            }
            (Width::W8, true) => {
                self.func.instruction(&Instruction::LocalGet(dst_local));
                self.func.instruction(&Instruction::I64Const(!0xff00i64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::LocalGet(src_local));
                self.func.instruction(&Instruction::I64Const(0xff));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(8));
                self.func.instruction(&Instruction::I64Shl);
                self.func.instruction(&Instruction::I64Or);
                self.func.instruction(&Instruction::LocalSet(dst_local));
            }
            _ => unreachable!("invalid gpr part write: {width:?} high8={high8}"),
        }
    }

    fn emit_read_flag(&mut self, flag: Flag) {
        // Build just this flag from the operation that produced it, rather than
        // having computed all six when that operation ran.
        if let Some(p) = self.pending {
            self.emit_pending_flag_value(p, flag);
            return;
        }
        if self.incoming_live {
            // Either the record owes this flag, or nothing is owed and RFLAGS has it.
            self.emit_descriptor_flag_value(flag);
            self.emit_rflags_bit(flag);
            self.func
                .instruction(&Instruction::LocalGet(self.layout.lazy_header_local()));
            self.func.instruction(&Instruction::I64Eqz);
            self.func.instruction(&Instruction::I32Eqz);
            self.func.instruction(&Instruction::Select);
            return;
        }
        self.emit_rflags_bit(flag);
    }

    /// Push `flag`'s bit from the RFLAGS local.
    fn emit_rflags_bit(&mut self, flag: Flag) {
        let bit = 1u64 << flag.rflags_bit();
        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func.instruction(&Instruction::I64Const(bit as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::I64Ne);
    }

    fn emit_write_flag(&mut self, flag: Flag, src: ValueId) {
        self.materialize_pending();
        let bit = 1u64 << flag.rflags_bit();
        let clear_mask = !(bit as i64);
        let bit_mask = bit as i64;
        let src_local = self.layout.value_local(src);

        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func.instruction(&Instruction::I64Const(bit_mask));
        self.func.instruction(&Instruction::I64Or);

        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func.instruction(&Instruction::I64Const(clear_mask));
        self.func.instruction(&Instruction::I64And);

        self.func.instruction(&Instruction::LocalGet(src_local));
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::I64Ne);

        self.func.instruction(&Instruction::Select);
        self.func
            .instruction(&Instruction::LocalSet(self.layout.rflags_local()));
        self.rflags_written = true;
    }

    fn emit_set_flag(&mut self, flag: Flag, emit_value: impl FnOnce(&mut Self)) {
        let bit = 1u64 << flag.rflags_bit();
        let set_mask = bit as i64;
        let clear_mask = !(bit as i64);

        // r_set
        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func.instruction(&Instruction::I64Const(set_mask));
        self.func.instruction(&Instruction::I64Or);

        // r_clear
        self.func
            .instruction(&Instruction::LocalGet(self.layout.rflags_local()));
        self.func.instruction(&Instruction::I64Const(clear_mask));
        self.func.instruction(&Instruction::I64And);

        // cond_i32
        emit_value(self);

        // select + store
        self.func.instruction(&Instruction::Select);
        self.func
            .instruction(&Instruction::LocalSet(self.layout.rflags_local()));
        self.rflags_written = true;
    }

    fn emit_set_flag_const(&mut self, flag: Flag, value: bool) {
        self.emit_set_flag(flag, |this| {
            this.func
                .instruction(&Instruction::I32Const(if value { 1 } else { 0 }));
        });
    }

    fn emit_add_flags(
        &mut self,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res: ValueId,
        carry_in: bool,
    ) {
        self.emit_addsub_flags(width, flags, lhs, rhs, self.layout.value_local(res), true, carry_in);
    }

    fn emit_sub_flags(
        &mut self,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res: ValueId,
        carry_in: bool,
    ) {
        self.emit_addsub_flags(width, flags, lhs, rhs, self.layout.value_local(res), false, carry_in);
    }

    fn emit_sub_flags_with_res(
        &mut self,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res_local: u32,
    ) {
        self.emit_addsub_flags(width, flags, lhs, rhs, res_local, false, false);
    }

    /// Fold an incoming lazy-flag descriptor into the RFLAGS local, on entry.
    ///
    /// The previous block may have handed over the operation that produced the flags
    /// instead of the flags themselves, so a block that reads one on arrival has to
    /// finish that work. Emitted only for blocks that actually do — most read only
    /// flags they produced themselves, and pay nothing.
    ///
    /// No branch on the operation is needed: which formula applies reduces to two
    /// booleans, "is this add-shaped" and "is this a logic op", so the whole thing is
    /// straight-line arithmetic.
    fn emit_load_incoming_descriptor(&mut self) {
        let cpu = self.layout.cpu_ptr_local();
        let header = self.layout.lazy_header_local();
        for (local, off) in [
            (header, abi::CPU_LAZY_FLAGS_OFF),
            (self.layout.lazy_result_local(), abi::CPU_LAZY_RESULT_OFF),
            (self.layout.lazy_op1_local(), abi::CPU_LAZY_OP1_OFF),
            (self.layout.lazy_op2_local(), abi::CPU_LAZY_OP2_OFF),
        ] {
            self.func.instruction(&Instruction::LocalGet(cpu));
            self.func.instruction(&Instruction::I64Load(memarg(off, 3)));
            self.func.instruction(&Instruction::LocalSet(local));
        }
    }

    /// The arithmetic half of [`Self::emit_materialize_incoming_flags`], over the
    /// descriptor locals as they stand.
    /// Push the value of `flag` implied by the descriptor locals, as an i32 boolean.
    ///
    /// The operation is only known at run time here, but which formula applies
    /// reduces to two booleans — "add-shaped" and "not a logic op" — so this stays
    /// straight-line arithmetic rather than a branch over five cases. ZF, SF and PF
    /// do not depend on the operation at all, which is why a block that only tests
    /// `je`/`jne` gets away with almost nothing.
    fn emit_descriptor_flag_value(&mut self, flag: Flag) {
        let header = self.layout.lazy_header_local();
        let (a, b, r) = (
            self.layout.lazy_op1_local(),
            self.layout.lazy_op2_local(),
            self.layout.lazy_result_local(),
        );

        let op_byte = |this: &mut Self| {
            this.func.instruction(&Instruction::LocalGet(header));
            this.func.instruction(&Instruction::I64Const(0xff));
            this.func.instruction(&Instruction::I64And);
        };
        // `select` yields its *first* pushed value when the condition is true, and
        // the guards below read `[value, 0, cond]` — so the condition is "not logic".
        let is_not_logic = |this: &mut Self| {
            op_byte(this);
            this.func
                .instruction(&Instruction::I64Const(LazyOp::Logic.abi_code() as i64));
            this.func.instruction(&Instruction::I64Ne);
        };
        let is_add = |this: &mut Self| {
            // Add = 1, Adc = 2, so "at most Adc" is the add-shaped test.
            op_byte(this);
            this.func
                .instruction(&Instruction::I64Const(LazyOp::Adc.abi_code() as i64));
            this.func.instruction(&Instruction::I64LeU);
        };
        // sign bit for this operand size: 1 << (size * 8 - 1)
        let push_sign = |this: &mut Self| {
            this.func.instruction(&Instruction::I64Const(1));
            this.func.instruction(&Instruction::LocalGet(header));
            this.func.instruction(&Instruction::I64Const(8));
            this.func.instruction(&Instruction::I64ShrU);
            this.func.instruction(&Instruction::I64Const(0xff));
            this.func.instruction(&Instruction::I64And);
            this.func.instruction(&Instruction::I64Const(8));
            this.func.instruction(&Instruction::I64Mul);
            this.func.instruction(&Instruction::I64Const(1));
            this.func.instruction(&Instruction::I64Sub);
            this.func.instruction(&Instruction::I64Shl);
        };
        let push_carry = |this: &mut Self| {
            this.func.instruction(&Instruction::LocalGet(header));
            this.func.instruction(&Instruction::I64Const(16));
            this.func.instruction(&Instruction::I64ShrU);
            this.func.instruction(&Instruction::I64Const(1));
            this.func.instruction(&Instruction::I64And);
        };

        match flag {
            Flag::Zf => {
                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I64Eqz);
            }
            Flag::Sf => {
                self.func.instruction(&Instruction::LocalGet(r));
                push_sign(self);
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
            }
            Flag::Pf => self.emit_parity_even_i32(r),
            Flag::Af => {
                self.func.instruction(&Instruction::LocalGet(a));
                self.func.instruction(&Instruction::LocalGet(b));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::I64Const(0x10));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
                self.func.instruction(&Instruction::I32Const(0));
                is_not_logic(self);
                self.func.instruction(&Instruction::Select);
            }
            Flag::Cf => {
                let cf = |x: u32, y: u32, this: &mut Self| {
                    this.func.instruction(&Instruction::LocalGet(x));
                    this.func.instruction(&Instruction::LocalGet(y));
                    this.func.instruction(&Instruction::I64LtU);
                    this.func.instruction(&Instruction::LocalGet(x));
                    this.func.instruction(&Instruction::LocalGet(y));
                    this.func.instruction(&Instruction::I64Eq);
                    push_carry(this);
                    this.func.instruction(&Instruction::I64Const(0));
                    this.func.instruction(&Instruction::I64Ne);
                    this.func.instruction(&Instruction::I32And);
                    this.func.instruction(&Instruction::I32Or);
                };
                cf(r, a, self);
                cf(a, b, self);
                is_add(self);
                self.func.instruction(&Instruction::Select);
                self.func.instruction(&Instruction::I32Const(0));
                is_not_logic(self);
                self.func.instruction(&Instruction::Select);
            }
            Flag::Of => {
                self.func.instruction(&Instruction::LocalGet(a));
                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::LocalGet(b));
                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::I64And);
                push_sign(self);
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);

                self.func.instruction(&Instruction::LocalGet(a));
                self.func.instruction(&Instruction::LocalGet(b));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::LocalGet(a));
                self.func.instruction(&Instruction::LocalGet(r));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::I64And);
                push_sign(self);
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);

                is_add(self);
                self.func.instruction(&Instruction::Select);
                self.func.instruction(&Instruction::I32Const(0));
                is_not_logic(self);
                self.func.instruction(&Instruction::Select);
            }
        }
    }

    /// Fold the descriptor locals into the RFLAGS local and mark them consumed.
    fn emit_materialize_descriptor_locals(&mut self) {
        let header = self.layout.lazy_header_local();
        self.func.instruction(&Instruction::LocalGet(header));
        self.func.instruction(&Instruction::I64Eqz);
        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        for flag in [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of] {
            self.emit_set_flag(flag, |this| this.emit_descriptor_flag_value(flag));
        }
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::LocalSet(header));
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }


    /// Write `CpuState.lazy_flags` from the descriptor locals.
    ///
    /// Unconditional, and reading locals rather than compile-time state, so that a
    /// block leaving through a bail-out in the middle hands back what actually ran.
    /// A zero header means nothing is owed, which is also what the locals hold if no
    /// operation was ever deferred.
    fn emit_spill_pending_descriptor(&mut self) {
        let cpu = self.layout.cpu_ptr_local();
        // The *header* is always written: it may have to clear a record the previous
        // block left owed, and a stale one would be read as work that is no longer due.
        // The three operands are only meaningful when the header says something is
        // owed, so a block that never loaded a record and never deferred an operation
        // is storing three zeros nobody reads.
        let skip_operands = !self.descriptor_touched && descriptor_spill_trimmed();
        for (off, local) in [
            (abi::CPU_LAZY_FLAGS_OFF, self.layout.lazy_header_local()),
            (abi::CPU_LAZY_RESULT_OFF, self.layout.lazy_result_local()),
            (abi::CPU_LAZY_OP1_OFF, self.layout.lazy_op1_local()),
            (abi::CPU_LAZY_OP2_OFF, self.layout.lazy_op2_local()),
        ] {
            if skip_operands && off != abi::CPU_LAZY_FLAGS_OFF {
                continue;
            }
            self.func.instruction(&Instruction::LocalGet(cpu));
            self.func.instruction(&Instruction::LocalGet(local));
            self.func.instruction(&Instruction::I64Store(memarg(off, 3)));
        }
    }

    /// Push the value of `flag` for an add/sub-shaped operation, as an i32 boolean.
    ///
    /// These are the same expressions the eager path used; naming them once is what
    /// lets a deferred operation reconstruct just the flag that is read instead of
    /// all six.
    fn emit_addsub_flag_value(
        &mut self,
        flag: Flag,
        width: Width,
        is_add: bool,
        carry_in: bool,
        lhs_local: u32,
        rhs_local: u32,
        res_local: u32,
    ) {
        let sign_bit = 1u64 << (width.bits() - 1);
        match flag {
            Flag::Cf => {
                // `add`/`sub` carry out iff the result wrapped. `adc`/`sbb` have one
                // more case: when the carry *in* is what pushes it over, the
                // comparison comes out equal rather than less-than.
                let (a, b) = if is_add {
                    (res_local, lhs_local)
                } else {
                    (lhs_local, rhs_local)
                };
                self.func.instruction(&Instruction::LocalGet(a));
                self.func.instruction(&Instruction::LocalGet(b));
                self.func.instruction(&Instruction::I64LtU);
                if carry_in {
                    self.func.instruction(&Instruction::LocalGet(a));
                    self.func.instruction(&Instruction::LocalGet(b));
                    self.func.instruction(&Instruction::I64Eq);
                    self.func
                        .instruction(&Instruction::LocalGet(self.layout.lazy_carry_local()));
                    self.func.instruction(&Instruction::I64Const(0));
                    self.func.instruction(&Instruction::I64Ne);
                    self.func.instruction(&Instruction::I32And);
                    self.func.instruction(&Instruction::I32Or);
                }
            }
            Flag::Zf => {
                self.func.instruction(&Instruction::LocalGet(res_local));
                self.func.instruction(&Instruction::I64Eqz);
            }
            Flag::Sf => {
                self.func.instruction(&Instruction::LocalGet(res_local));
                self.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
            }
            Flag::Pf => self.emit_parity_even_i32(res_local),
            Flag::Af => {
                self.func.instruction(&Instruction::LocalGet(lhs_local));
                self.func.instruction(&Instruction::LocalGet(rhs_local));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::LocalGet(res_local));
                self.func.instruction(&Instruction::I64Xor);
                self.func.instruction(&Instruction::I64Const(0x10));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
            }
            Flag::Of => {
                if is_add {
                    self.func.instruction(&Instruction::LocalGet(lhs_local));
                    self.func.instruction(&Instruction::LocalGet(res_local));
                    self.func.instruction(&Instruction::I64Xor);
                    self.func.instruction(&Instruction::LocalGet(rhs_local));
                    self.func.instruction(&Instruction::LocalGet(res_local));
                    self.func.instruction(&Instruction::I64Xor);
                    self.func.instruction(&Instruction::I64And);
                } else {
                    self.func.instruction(&Instruction::LocalGet(lhs_local));
                    self.func.instruction(&Instruction::LocalGet(rhs_local));
                    self.func.instruction(&Instruction::I64Xor);
                    self.func.instruction(&Instruction::LocalGet(lhs_local));
                    self.func.instruction(&Instruction::LocalGet(res_local));
                    self.func.instruction(&Instruction::I64Xor);
                    self.func.instruction(&Instruction::I64And);
                }
                self.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
            }
        }
    }

    /// Push the value of `flag` for a logic operation. CF, OF and AF are zero there.
    fn emit_logic_flag_value(&mut self, flag: Flag, width: Width, res_local: u32) {
        let sign_bit = 1u64 << (width.bits() - 1);
        match flag {
            Flag::Cf | Flag::Of | Flag::Af => {
                self.func.instruction(&Instruction::I32Const(0));
            }
            Flag::Zf => {
                self.func.instruction(&Instruction::LocalGet(res_local));
                self.func.instruction(&Instruction::I64Eqz);
            }
            Flag::Sf => {
                self.func.instruction(&Instruction::LocalGet(res_local));
                self.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                self.func.instruction(&Instruction::I64And);
                self.func.instruction(&Instruction::I64Const(0));
                self.func.instruction(&Instruction::I64Ne);
            }
            Flag::Pf => self.emit_parity_even_i32(res_local),
        }
    }

    /// Push the value of `flag` from whatever operation is pending.
    /// Whether flags may be deferred. `AERO_JIT_NO_LAZY_FLAGS=1` forces the eager
    /// path, which is how a divergence gets attributed to deferral or ruled out.
    fn lazy_flags_enabled() -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var_os("AERO_JIT_NO_LAZY_FLAGS").is_none())
        }
        #[cfg(target_arch = "wasm32")]
        {
            true
        }
    }

    fn emit_pending_flag_value(&mut self, p: PendingFlags, flag: Flag) {
        let (op1, op2, res) = (
            self.layout.lazy_op1_local(),
            self.layout.lazy_op2_local(),
            self.layout.lazy_result_local(),
        );
        if p.op == LazyOp::Logic {
            self.emit_logic_flag_value(flag, p.width, res);
        } else {
            let carry = matches!(p.op, LazyOp::Adc | LazyOp::Sbb);
            self.emit_addsub_flag_value(flag, p.width, p.op.is_add_like(), carry, op1, op2, res);
        }
    }

    /// Record `op` as the operation whose flags are owed, in the descriptor locals.
    fn emit_defer_flags(&mut self, op: LazyOp, width: Width, lhs: u32, rhs: u32, res: u32) {
        for (dst, src) in [
            (self.layout.lazy_op1_local(), lhs),
            (self.layout.lazy_op2_local(), rhs),
            (self.layout.lazy_result_local(), res),
        ] {
            self.func.instruction(&Instruction::LocalGet(src));
            self.func.instruction(&Instruction::LocalSet(dst));
        }

        let header = op.abi_code() | (((width.bits() / 8) as u64) << 8);
        self.func.instruction(&Instruction::I64Const(header as i64));
        if matches!(op, LazyOp::Adc | LazyOp::Sbb) {
            // The carry is a runtime value; the rest of the header is constant.
            self.func
                .instruction(&Instruction::LocalGet(self.layout.lazy_carry_local()));
            self.func.instruction(&Instruction::I64Const(1));
            self.func.instruction(&Instruction::I64And);
            self.func.instruction(&Instruction::I64Const(16));
            self.func.instruction(&Instruction::I64Shl);
            self.func.instruction(&Instruction::I64Or);
        }
        self.func
            .instruction(&Instruction::LocalSet(self.layout.lazy_header_local()));
        self.pending = Some(PendingFlags { op, width });
        self.descriptor_touched = true;
        // The record just written replaces anything handed in.
        self.incoming_live = false;
    }

    /// Compute every pending flag into the RFLAGS local and forget the operation.
    ///
    /// Needed wherever something wants RFLAGS as a whole rather than one flag: an
    /// instruction that writes only part of it, a helper call, or a bail-out.
    fn materialize_pending(&mut self) {
        let Some(p) = self.pending.take() else {
            if self.incoming_live {
                self.emit_materialize_descriptor_locals();
                self.incoming_live = false;
            }
            return;
        };
        for flag in [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of] {
            self.emit_set_flag(flag, |this| this.emit_pending_flag_value(p, flag));
        }
        // The flags are in RFLAGS now, so the exit must not also hand back a
        // descriptor claiming they are still owed.
        self.func.instruction(&Instruction::I64Const(0));
        self.func
            .instruction(&Instruction::LocalSet(self.layout.lazy_header_local()));
    }

    fn emit_logic_flags(&mut self, width: Width, flags: FlagSet, res: ValueId) {
        self.emit_logic_flags_with_res(width, flags, self.layout.value_local(res));
    }

    fn emit_logic_flags_with_res(&mut self, width: Width, flags: FlagSet, res_local: u32) {
        if flags == FlagSet::ALU && Emitter::lazy_flags_enabled() {
            self.emit_defer_flags(LazyOp::Logic, width, res_local, res_local, res_local);
            return;
        }
        self.materialize_pending();
        let sign_bit = 1u64 << (width.bits() - 1);

        if flags.contains(FlagSet::CF) {
            self.emit_set_flag_const(Flag::Cf, false);
        }
        if flags.contains(FlagSet::OF) {
            self.emit_set_flag_const(Flag::Of, false);
        }
        if flags.contains(FlagSet::AF) {
            // Tier-1 interpreter forces AF=false for logic ops.
            self.emit_set_flag_const(Flag::Af, false);
        }

        if flags.contains(FlagSet::ZF) {
            self.emit_set_flag(Flag::Zf, |this| {
                this.func.instruction(&Instruction::LocalGet(res_local));
                this.func.instruction(&Instruction::I64Eqz);
            });
        }
        if flags.contains(FlagSet::SF) {
            self.emit_set_flag(Flag::Sf, |this| {
                this.func.instruction(&Instruction::LocalGet(res_local));
                this.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                this.func.instruction(&Instruction::I64And);
                this.func.instruction(&Instruction::I64Const(0));
                this.func.instruction(&Instruction::I64Ne);
            });
        }
        if flags.contains(FlagSet::PF) {
            self.emit_set_flag(Flag::Pf, |this| {
                this.emit_parity_even_i32(res_local);
            });
        }
    }

    /// CF and OF for the two-operand `imul`, plus the flags it defines as zero.
    ///
    /// Both are set when the signed product does not fit in the destination, which
    /// is the test the IR interpreter makes: sign-extend both operands, multiply
    /// exactly, and compare against the sign-extended truncated result. For widths
    /// up to 32 that product fits in an `i64`. At 64 it does not, so the high half
    /// is built from 32-bit limbs — everything is recomputed from the value locals
    /// rather than held, because the layout has a single scratch local.
    ///
    /// `imul` leaves ZF, SF, PF and AF architecturally undefined, and Tier-0 leaves
    /// them holding whatever they held. So does this: a guest can set a flag in one
    /// tier and read it in another, and it has to look like one machine.
    fn emit_mul_flags(
        &mut self,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res: ValueId,
    ) {
        // Writes only part of RFLAGS, so whatever is pending has to be settled
        // first or the flags it does not write would be lost.
        self.materialize_pending();
        if !flags.contains(FlagSet::CF) && !flags.contains(FlagSet::OF) {
            return;
        }

        let lhs_local = self.layout.value_local(lhs);
        let rhs_local = self.layout.value_local(rhs);
        let res_local = self.layout.value_local(res);

        // Pushes 1 if the signed product overflowed the destination width.
        let overflow = |this: &mut Self| {
            let sext = |this: &mut Self, local: u32| {
                this.func.instruction(&Instruction::LocalGet(local));
                match width {
                    Width::W8 => this.func.instruction(&Instruction::I64Extend8S),
                    Width::W16 => this.func.instruction(&Instruction::I64Extend16S),
                    Width::W32 => this.func.instruction(&Instruction::I64Extend32S),
                    Width::W64 => this.func.instruction(&Instruction::Nop),
                };
            };
            if width != Width::W64 {
                // The exact product fits in an i64 for every narrower width.
                sext(this, lhs_local);
                sext(this, rhs_local);
                this.func.instruction(&Instruction::I64Mul);
                sext(this, res_local);
                this.func.instruction(&Instruction::I64Ne);
                return;
            }

            // High 64 bits of the signed 128-bit product, from 32-bit limbs:
            //   hi_u = a1*b1 + (a0*b1 >> 32) + (a1*b0 >> 32)
            //        + ((a0*b0 >> 32) + (a0*b1 & m) + (a1*b0 & m)) >> 32
            //   hi_s = hi_u - (a >> 63 & b) - (b >> 63 & a)
            const M: i64 = 0xffff_ffff;
            let lo = |this: &mut Self, local: u32| {
                this.func.instruction(&Instruction::LocalGet(local));
                this.func.instruction(&Instruction::I64Const(M));
                this.func.instruction(&Instruction::I64And);
            };
            let hi = |this: &mut Self, local: u32| {
                this.func.instruction(&Instruction::LocalGet(local));
                this.func.instruction(&Instruction::I64Const(32));
                this.func.instruction(&Instruction::I64ShrU);
            };
            let mul = |this: &mut Self, f: &dyn Fn(&mut Self, u32), a: u32, g: &dyn Fn(&mut Self, u32), b: u32| {
                f(this, a);
                g(this, b);
                this.func.instruction(&Instruction::I64Mul);
            };
            let shr32 = |this: &mut Self| {
                this.func.instruction(&Instruction::I64Const(32));
                this.func.instruction(&Instruction::I64ShrU);
            };
            let mask32 = |this: &mut Self| {
                this.func.instruction(&Instruction::I64Const(M));
                this.func.instruction(&Instruction::I64And);
            };

            // a1*b1
            mul(this, &hi, lhs_local, &hi, rhs_local);
            // + (a0*b1 >> 32)
            mul(this, &lo, lhs_local, &hi, rhs_local);
            shr32(this);
            this.func.instruction(&Instruction::I64Add);
            // + (a1*b0 >> 32)
            mul(this, &hi, lhs_local, &lo, rhs_local);
            shr32(this);
            this.func.instruction(&Instruction::I64Add);
            // + (mid >> 32)
            mul(this, &lo, lhs_local, &lo, rhs_local);
            shr32(this);
            mul(this, &lo, lhs_local, &hi, rhs_local);
            mask32(this);
            this.func.instruction(&Instruction::I64Add);
            mul(this, &hi, lhs_local, &lo, rhs_local);
            mask32(this);
            this.func.instruction(&Instruction::I64Add);
            shr32(this);
            this.func.instruction(&Instruction::I64Add);
            // signed correction
            this.func.instruction(&Instruction::LocalGet(lhs_local));
            this.func.instruction(&Instruction::I64Const(63));
            this.func.instruction(&Instruction::I64ShrS);
            this.func.instruction(&Instruction::LocalGet(rhs_local));
            this.func.instruction(&Instruction::I64And);
            this.func.instruction(&Instruction::I64Sub);
            this.func.instruction(&Instruction::LocalGet(rhs_local));
            this.func.instruction(&Instruction::I64Const(63));
            this.func.instruction(&Instruction::I64ShrS);
            this.func.instruction(&Instruction::LocalGet(lhs_local));
            this.func.instruction(&Instruction::I64And);
            this.func.instruction(&Instruction::I64Sub);
            // != sign extension of the low half
            this.func.instruction(&Instruction::LocalGet(res_local));
            this.func.instruction(&Instruction::I64Const(63));
            this.func.instruction(&Instruction::I64ShrS);
            this.func.instruction(&Instruction::I64Ne);
        };

        if flags.contains(FlagSet::CF) {
            self.emit_set_flag(Flag::Cf, overflow);
        }
        if flags.contains(FlagSet::OF) {
            self.emit_set_flag(Flag::Of, overflow);
        }
    }

    /// CF and OF for `rol`/`ror`.
    ///
    /// These were not emitted at all, on the grounds that rotates only touch CF and
    /// OF and nothing downstream looked at them. Two things make that untenable: the
    /// IR interpreter *does* compute them, so the two disagreed on every rotate, and
    /// once dead-flag elimination can drop an earlier write because a rotate is
    /// declared to overwrite it, the flag is left holding a value from before the
    /// instruction that really did set it.
    ///
    /// The rules, matching `tier1::ir::interp`: with a rotate count of zero nothing
    /// changes at all; otherwise CF is the bit rotated out — the result's low bit for
    /// `rol`, its high bit for `ror` — and OF is defined only for a count of one.
    fn emit_rotate_flags(
        &mut self,
        op: BinOp,
        width: Width,
        flags: FlagSet,
        rhs: ValueId,
        res: ValueId,
    ) {
        // Writes only part of RFLAGS, so whatever is pending has to be settled
        // first or the flags it does not write would be lost.
        self.materialize_pending();
        debug_assert!(matches!(op, BinOp::Rol | BinOp::Ror));
        if !flags.contains(FlagSet::CF) && !flags.contains(FlagSet::OF) {
            return;
        }

        let bits = width.bits();
        let sign_bit = 1u64 << (bits - 1);
        let rhs_local = self.layout.value_local(rhs);
        let res_local = self.layout.value_local(res);
        let amt_local = self.layout.scratch_local();

        // amt = (rhs & shift_mask) % bits, the number of positions actually rotated.
        self.func.instruction(&Instruction::LocalGet(rhs_local));
        self.emit_shift_mask(width);
        self.func.instruction(&Instruction::I64Const(bits as i64));
        self.func.instruction(&Instruction::I64RemU);
        self.func.instruction(&Instruction::LocalTee(amt_local));

        // if amt != 0 { ... }
        self.func.instruction(&Instruction::I64Eqz);
        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            if flags.contains(FlagSet::CF) {
                let bit = if matches!(op, BinOp::Rol) { 1 } else { sign_bit };
                self.emit_set_flag(Flag::Cf, |this| {
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Const(bit as i64));
                    this.func.instruction(&Instruction::I64And);
                    this.func.instruction(&Instruction::I64Const(0));
                    this.func.instruction(&Instruction::I64Ne);
                });
            }

            if flags.contains(FlagSet::OF) {
                // OF is architecturally defined only for a rotate of one.
                self.func.instruction(&Instruction::LocalGet(amt_local));
                self.func.instruction(&Instruction::I64Const(1));
                self.func.instruction(&Instruction::I64Eq);
                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                let other = if matches!(op, BinOp::Rol) {
                    1
                } else {
                    sign_bit >> 1
                };
                self.emit_set_flag(Flag::Of, |this| {
                    // msb(res) xor (rol ? lsb(res) : msb-1(res))
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func
                        .instruction(&Instruction::I64Const(sign_bit as i64));
                    this.func.instruction(&Instruction::I64And);
                    this.func.instruction(&Instruction::I64Const(0));
                    this.func.instruction(&Instruction::I64Ne);
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Const(other as i64));
                    this.func.instruction(&Instruction::I64And);
                    this.func.instruction(&Instruction::I64Const(0));
                    this.func.instruction(&Instruction::I64Ne);
                    this.func.instruction(&Instruction::I32Xor);
                });
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }

    fn emit_shift_flags(
        &mut self,
        op: BinOp,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res: ValueId,
    ) {
        // Writes only part of RFLAGS, so whatever is pending has to be settled
        // first or the flags it does not write would be lost.
        self.materialize_pending();
        debug_assert!(matches!(op, BinOp::Shl | BinOp::Shr | BinOp::Sar));

        // x86 shifts do not update any flags when the (masked) shift count is 0.
        //
        // AF is architecturally undefined for shifts, so its value is a choice rather
        // than a requirement — but it has to be the *same* choice Tier-0 makes, since
        // the guest can set flags in one tier and read them in the other. Tier-0
        // clears it, so a requested AF is written as zero below.
        let sign_bit = 1u64 << (width.bits() - 1);
        let lhs_local = self.layout.value_local(lhs);
        let rhs_local = self.layout.value_local(rhs);
        let res_local = self.layout.value_local(res);
        let amt_local = self.layout.scratch_local();

        // amt = rhs & shift_mask
        self.func.instruction(&Instruction::LocalGet(rhs_local));
        self.emit_shift_mask(width);
        self.func.instruction(&Instruction::LocalTee(amt_local));

        // if amt != 0 { ... }
        self.func.instruction(&Instruction::I64Eqz);
        self.func.instruction(&Instruction::I32Eqz);
        self.func.instruction(&Instruction::If(BlockType::Empty));
        self.depth += 1;
        {
            if flags.contains(FlagSet::AF) {
                self.emit_set_flag_const(Flag::Af, false);
            }
            if flags.contains(FlagSet::ZF) {
                self.emit_set_flag(Flag::Zf, |this| {
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Eqz);
                });
            }
            if flags.contains(FlagSet::SF) {
                self.emit_set_flag(Flag::Sf, |this| {
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func
                        .instruction(&Instruction::I64Const(sign_bit as i64));
                    this.func.instruction(&Instruction::I64And);
                    this.func.instruction(&Instruction::I64Const(0));
                    this.func.instruction(&Instruction::I64Ne);
                });
            }
            if flags.contains(FlagSet::PF) {
                self.emit_set_flag(Flag::Pf, |this| {
                    this.emit_parity_even_i32(res_local);
                });
            }

            // CF is only defined for shift counts in the range [1, width.bits()]. For larger
            // counts (possible for 8/16-bit operands due to x86's 5-bit masking), CF is undefined;
            // conservatively leave it unchanged.
            if flags.contains(FlagSet::CF) {
                self.func.instruction(&Instruction::LocalGet(amt_local));
                self.func
                    .instruction(&Instruction::I64Const(width.bits() as i64));
                self.func.instruction(&Instruction::I64LeU);
                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                {
                    self.emit_set_flag(Flag::Cf, |this| match op {
                        BinOp::Shl => {
                            // (lhs >> (bits - amt)) & 1
                            this.func.instruction(&Instruction::LocalGet(lhs_local));
                            this.func
                                .instruction(&Instruction::I64Const(width.bits() as i64));
                            this.func.instruction(&Instruction::LocalGet(amt_local));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::I64ShrU);
                            this.func.instruction(&Instruction::I64Const(1));
                            this.func.instruction(&Instruction::I64And);
                            this.func.instruction(&Instruction::I64Const(0));
                            this.func.instruction(&Instruction::I64Ne);
                        }
                        BinOp::Shr | BinOp::Sar => {
                            // (lhs >> (amt - 1)) & 1
                            this.func.instruction(&Instruction::LocalGet(lhs_local));
                            this.func.instruction(&Instruction::LocalGet(amt_local));
                            this.func.instruction(&Instruction::I64Const(1));
                            this.func.instruction(&Instruction::I64Sub);
                            this.func.instruction(&Instruction::I64ShrU);
                            this.func.instruction(&Instruction::I64Const(1));
                            this.func.instruction(&Instruction::I64And);
                            this.func.instruction(&Instruction::I64Const(0));
                            this.func.instruction(&Instruction::I64Ne);
                        }
                        _ => unreachable!(),
                    });
                }
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }

            // OF is only defined for a shift count of 1. For counts > 1, OF is undefined;
            // conservatively leave it unchanged.
            if flags.contains(FlagSet::OF) {
                self.func.instruction(&Instruction::LocalGet(amt_local));
                self.func.instruction(&Instruction::I64Const(1));
                self.func.instruction(&Instruction::I64Eq);
                self.func.instruction(&Instruction::If(BlockType::Empty));
                self.depth += 1;
                {
                    match op {
                        BinOp::Shl => self.emit_set_flag(Flag::Of, |this| {
                            // (lhs ^ res) has the sign bit set if MSB(result) != MSB(lhs).
                            this.func.instruction(&Instruction::LocalGet(lhs_local));
                            this.func.instruction(&Instruction::LocalGet(res_local));
                            this.func.instruction(&Instruction::I64Xor);
                            this.func
                                .instruction(&Instruction::I64Const(sign_bit as i64));
                            this.func.instruction(&Instruction::I64And);
                            this.func.instruction(&Instruction::I64Const(0));
                            this.func.instruction(&Instruction::I64Ne);
                        }),
                        BinOp::Shr => self.emit_set_flag(Flag::Of, |this| {
                            // OF = old MSB.
                            this.func.instruction(&Instruction::LocalGet(lhs_local));
                            this.func
                                .instruction(&Instruction::I64Const(sign_bit as i64));
                            this.func.instruction(&Instruction::I64And);
                            this.func.instruction(&Instruction::I64Const(0));
                            this.func.instruction(&Instruction::I64Ne);
                        }),
                        BinOp::Sar => self.emit_set_flag_const(Flag::Of, false),
                        _ => unreachable!(),
                    }
                }
                self.func.instruction(&Instruction::End);
                self.depth -= 1;
            }
        }
        self.func.instruction(&Instruction::End);
        self.depth -= 1;
    }

    fn emit_addsub_flags(
        &mut self,
        width: Width,
        flags: FlagSet,
        lhs: ValueId,
        rhs: ValueId,
        res_local: u32,
        is_add: bool,
        carry_in: bool,
    ) {
        // Defer, unless the instruction declares fewer than all six flags. A partial
        // declaration means the ones left out keep their old values, and a pending
        // record cannot express "these six, except two" — so those stay eager.
        if flags == FlagSet::ALU && Emitter::lazy_flags_enabled() {
            let op = match (is_add, carry_in) {
                (true, false) => LazyOp::Add,
                (true, true) => LazyOp::Adc,
                (false, false) => LazyOp::Sub,
                (false, true) => LazyOp::Sbb,
            };
            let (lhs, rhs) = (self.layout.value_local(lhs), self.layout.value_local(rhs));
            self.emit_defer_flags(op, width, lhs, rhs, res_local);
            return;
        }
        // Anything computed here replaces what was pending for those flags, and the
        // rest of the pending set still has to land somewhere.
        self.materialize_pending();
        let sign_bit = 1u64 << (width.bits() - 1);

        let lhs_local = self.layout.value_local(lhs);
        let rhs_local = self.layout.value_local(rhs);

        if flags.contains(FlagSet::CF) {
            self.emit_set_flag(Flag::Cf, |this| {
                // `add`/`sub` carry out iff the result wrapped, which the unsigned
                // comparison below detects. `adc`/`sbb` have one more case: when the
                // carry *in* is what pushes the result over the edge, the comparison
                // comes out equal rather than less-than. `adc rax, -1` with CF set,
                // and `sbb eax, eax` with CF set, both land exactly there — and the
                // second is how compilers turn a comparison into 0/-1, so it is
                // everywhere in real code.
                //
                // The old CF is still readable here: this runs before the store that
                // overwrites it.
                let (a, b) = if is_add {
                    (res_local, lhs_local)
                } else {
                    (lhs_local, rhs_local)
                };
                this.func.instruction(&Instruction::LocalGet(a));
                this.func.instruction(&Instruction::LocalGet(b));
                this.func.instruction(&Instruction::I64LtU);
                if carry_in {
                    this.func.instruction(&Instruction::LocalGet(a));
                    this.func.instruction(&Instruction::LocalGet(b));
                    this.func.instruction(&Instruction::I64Eq);
                    this.emit_read_flag(Flag::Cf);
                    this.func.instruction(&Instruction::I32And);
                    this.func.instruction(&Instruction::I32Or);
                }
            });
        }

        if flags.contains(FlagSet::ZF) {
            self.emit_set_flag(Flag::Zf, |this| {
                this.func.instruction(&Instruction::LocalGet(res_local));
                this.func.instruction(&Instruction::I64Eqz);
            });
        }

        if flags.contains(FlagSet::SF) {
            self.emit_set_flag(Flag::Sf, |this| {
                this.func.instruction(&Instruction::LocalGet(res_local));
                this.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                this.func.instruction(&Instruction::I64And);
                this.func.instruction(&Instruction::I64Const(0));
                this.func.instruction(&Instruction::I64Ne);
            });
        }

        if flags.contains(FlagSet::PF) {
            self.emit_set_flag(Flag::Pf, |this| {
                this.emit_parity_even_i32(res_local);
            });
        }

        if flags.contains(FlagSet::AF) {
            self.emit_set_flag(Flag::Af, |this| {
                this.func.instruction(&Instruction::LocalGet(lhs_local));
                this.func.instruction(&Instruction::LocalGet(rhs_local));
                this.func.instruction(&Instruction::I64Xor);
                this.func.instruction(&Instruction::LocalGet(res_local));
                this.func.instruction(&Instruction::I64Xor);
                this.func.instruction(&Instruction::I64Const(0x10));
                this.func.instruction(&Instruction::I64And);
                this.func.instruction(&Instruction::I64Const(0));
                this.func.instruction(&Instruction::I64Ne);
            });
        }

        if flags.contains(FlagSet::OF) {
            self.emit_set_flag(Flag::Of, |this| {
                if is_add {
                    this.func.instruction(&Instruction::LocalGet(lhs_local));
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Xor);
                    this.func.instruction(&Instruction::LocalGet(rhs_local));
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Xor);
                    this.func.instruction(&Instruction::I64And);
                } else {
                    this.func.instruction(&Instruction::LocalGet(lhs_local));
                    this.func.instruction(&Instruction::LocalGet(rhs_local));
                    this.func.instruction(&Instruction::I64Xor);
                    this.func.instruction(&Instruction::LocalGet(lhs_local));
                    this.func.instruction(&Instruction::LocalGet(res_local));
                    this.func.instruction(&Instruction::I64Xor);
                    this.func.instruction(&Instruction::I64And);
                }
                this.func
                    .instruction(&Instruction::I64Const(sign_bit as i64));
                this.func.instruction(&Instruction::I64And);
                this.func.instruction(&Instruction::I64Const(0));
                this.func.instruction(&Instruction::I64Ne);
            });
        }
    }

    fn emit_parity_even_i32(&mut self, res_local: u32) {
        self.func.instruction(&Instruction::LocalGet(res_local));
        self.func.instruction(&Instruction::I64Const(0xff));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I32WrapI64);
        self.func.instruction(&Instruction::I32Popcnt);
        self.func.instruction(&Instruction::I32Const(1));
        self.func.instruction(&Instruction::I32And);
        self.func.instruction(&Instruction::I32Eqz);
    }

    fn emit_eval_cond(&mut self, cond: Cond) {
        let read = |this: &mut Self, flag: Flag| {
            this.emit_read_flag(flag);
        };
        match cond {
            Cond::O => read(self, Flag::Of),
            Cond::No => {
                read(self, Flag::Of);
                self.func.instruction(&Instruction::I32Eqz);
            }
            Cond::B => read(self, Flag::Cf),
            Cond::Ae => {
                read(self, Flag::Cf);
                self.func.instruction(&Instruction::I32Eqz);
            }
            Cond::E => read(self, Flag::Zf),
            Cond::Ne => {
                read(self, Flag::Zf);
                self.func.instruction(&Instruction::I32Eqz);
            }
            Cond::Be => {
                read(self, Flag::Cf);
                read(self, Flag::Zf);
                self.func.instruction(&Instruction::I32Or);
            }
            Cond::A => {
                read(self, Flag::Cf);
                self.func.instruction(&Instruction::I32Eqz);
                read(self, Flag::Zf);
                self.func.instruction(&Instruction::I32Eqz);
                self.func.instruction(&Instruction::I32And);
            }
            Cond::S => read(self, Flag::Sf),
            Cond::Ns => {
                read(self, Flag::Sf);
                self.func.instruction(&Instruction::I32Eqz);
            }
            Cond::P => read(self, Flag::Pf),
            Cond::Np => {
                read(self, Flag::Pf);
                self.func.instruction(&Instruction::I32Eqz);
            }
            Cond::L => {
                read(self, Flag::Sf);
                read(self, Flag::Of);
                self.func.instruction(&Instruction::I32Xor);
            }
            Cond::Ge => {
                read(self, Flag::Sf);
                read(self, Flag::Of);
                self.func.instruction(&Instruction::I32Eq);
            }
            Cond::Le => {
                read(self, Flag::Zf);
                read(self, Flag::Sf);
                read(self, Flag::Of);
                self.func.instruction(&Instruction::I32Xor);
                self.func.instruction(&Instruction::I32Or);
            }
            Cond::G => {
                read(self, Flag::Zf);
                self.func.instruction(&Instruction::I32Eqz);
                read(self, Flag::Sf);
                read(self, Flag::Of);
                self.func.instruction(&Instruction::I32Eq);
                self.func.instruction(&Instruction::I32And);
            }
        }
    }

    fn emit_sar(&mut self, width: Width, lhs: ValueId, rhs: ValueId) {
        let mask = width.mask();
        let sign_bit = 1u64 << (width.bits() - 1);

        // scratch = lhs truncated
        self.func
            .instruction(&Instruction::LocalGet(self.layout.value_local(lhs)));
        self.emit_trunc(width);
        self.func
            .instruction(&Instruction::LocalSet(self.layout.scratch_local()));

        // if_sign = scratch | !mask
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
        self.func
            .instruction(&Instruction::I64Const(!(mask as i64)));
        self.func.instruction(&Instruction::I64Or);

        // else = scratch
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_local()));

        // sign_cond = (scratch & sign_bit) != 0
        self.func
            .instruction(&Instruction::LocalGet(self.layout.scratch_local()));
        self.func
            .instruction(&Instruction::I64Const(sign_bit as i64));
        self.func.instruction(&Instruction::I64And);
        self.func.instruction(&Instruction::I64Const(0));
        self.func.instruction(&Instruction::I64Ne);

        // select sign-extended value
        self.func.instruction(&Instruction::Select);

        // shift amount
        self.func
            .instruction(&Instruction::LocalGet(self.layout.value_local(rhs)));
        self.emit_shift_mask(width);
        self.func.instruction(&Instruction::I64ShrS);
        self.emit_trunc(width);
    }
}

fn all_gprs() -> [Gpr; 16] {
    [
        Gpr::Rax,
        Gpr::Rcx,
        Gpr::Rdx,
        Gpr::Rbx,
        Gpr::Rsp,
        Gpr::Rbp,
        Gpr::Rsi,
        Gpr::Rdi,
        Gpr::R8,
        Gpr::R9,
        Gpr::R10,
        Gpr::R11,
        Gpr::R12,
        Gpr::R13,
        Gpr::R14,
        Gpr::R15,
    ]
}

fn next(idx: &mut u32) -> u32 {
    let cur = *idx;
    *idx += 1;
    cur
}

fn memarg(offset: u32, align: u32) -> MemArg {
    MemArg {
        offset: offset as u64,
        align,
        memory_index: 0,
    }
}

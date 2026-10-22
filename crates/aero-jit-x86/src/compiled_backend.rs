//! The seam between the compiled tier and whatever engine runs its blocks.
//!
//! The tier is not Wasmtime-specific and must not be: in the browser the WASM
//! engine *is* the browser, reached through JS rather than a host crate. Everything
//! else the tier needs already works there — guest RAM is the linear memory by
//! construction, and the Tier-0 seam and the address translator are plain Rust — so
//! this trait is the whole of what separates a native tier from a browser one.
//!
//! Compiled for every target on purpose, including `wasm32`, where the Wasmtime
//! backend beneath it is not.

use aero_cpu_core::state::CpuState;

/// Translates a guest virtual page on an inline-TLB miss.
///
/// Called with the faulting virtual address and the access type
/// (`abi::MMU_ACCESS_READ` / `MMU_ACCESS_WRITE`). Returns the guest *physical* page
/// base and the TLB flags to cache for it — read/write/exec for what the mapping
/// permits, and `TLB_FLAG_IS_RAM` when compiled code may address the page directly
/// rather than taking an MMIO exit.
///
/// `None` means the access faults: the block exits to the interpreter, which
/// re-executes the instruction and raises the fault architecturally.
pub type GuestTranslator = Box<dyn FnMut(u64, u32) -> Option<(u64, u64)>>;

/// Performs one guest memory access through the embedder's own bus.
///
/// The slow-path helpers translate an address and then read or write the engine's
/// linear memory at the result, which is right for RAM and wrong for everything
/// else: a device register has no backing there, so the access reads the wrong bytes
/// and the device never sees it. An embedder that has a real bus installs this, and
/// the helpers hand the access over instead of resolving it themselves.
///
/// `(addr, size_in_bytes, value)` — `Some(value)` writes, `None` reads. Returns the
/// value read (zero for a write), or `None` if the access faulted.
pub type GuestAccessor = Box<dyn FnMut(u64, u32, Option<u64>) -> Option<u64>>;

/// Reads the machine's timestamp counter, and leaves the architectural copy behind
/// exactly as the interpreter's `RDTSC` assist does.
///
/// Compiled code cannot reach the machine's `TimeSource`, and `rdtsc` is the largest
/// single coverage wall in a desktop session — so it asks the host.
pub type GuestClock = Box<dyn FnMut() -> u64>;

/// The part of the guest a compiled block maintains in linear memory.
///
/// Everything else about the CPU — control registers, MSRs, segment bases, the
/// paging mode — is only ever on the host side, so an assist has to be handed *these*
/// and left to find the rest where it already is. Getting that wrong is not subtle
/// but it is quiet: an assist given a blank `CpuState` reads CR3 as zero and writes
/// MSRs into something that is thrown away, and the guest stops about sixty thousand
/// instructions later with no other sign.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssistRegs {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    /// The FS and GS bases. A block reads these straight out of linear memory for
    /// `fs:`/`gs:`-relative addressing, and an assisted `swapgs` or `wrmsr` changes
    /// them — so they have to come back with the registers. Windows follows `swapgs`
    /// with `mov gs:[...], rsp` immediately, so getting this wrong writes the kernel
    /// stack pointer through a user-mode base.
    pub fs_base: u64,
    pub gs_base: u64,
    /// Set when the instruction changed how addresses translate — CR3, CR0, CR4,
    /// EFER, an `invlpg`. The inline TLB the block is using describes the old
    /// mapping and has to stop matching.
    pub flush_tlb: bool,
}

/// Performs one guest instruction on the host, given what the block was holding, and
/// reports where the guest went.
///
/// `None` means it could not be completed — a fault, a halt — and the block leaves
/// for the interpreter with linear memory untouched from before the instruction.
pub type GuestAssist = Box<dyn FnMut(&mut AssistRegs) -> Option<u64>>;

/// An engine that can hold compiled Tier-1 blocks and run chains of them.
/// What a chain of compiled blocks did.
///
/// `retired` is what the blocks reported, not what their lengths imply: a block with
/// side exits runs a different amount depending on which exit it leaves through, and
/// the machine's timing follows the real number.
#[derive(Debug, Clone, Copy)]
pub struct ChainOutcome {
    pub next_rip: u64,
    pub retired: u64,
    pub blocks: u32,
    pub exit_to_interpreter: bool,
}

pub trait CompiledBackend {
    /// Register a compiled module, returning the index used to run it.
    fn add_block(&mut self, wasm: &[u8]) -> u32;

    /// Add a module holding `count` blocks, returning one table index per block in
    /// the order they were compiled.
    ///
    /// Admission is dominated by per-module work, not by code size, so putting
    /// several blocks in one module is what makes a large block cache affordable.
    fn add_blocks(&mut self, wasm: &[u8], count: usize) -> Vec<u32>;

    /// Add a module whose blocks are reached through one exported dispatcher rather
    /// than being exported individually. Backends that cannot tell the difference
    /// may treat it as [`CompiledBackend::add_blocks`].
    fn add_dispatched_blocks(&mut self, wasm: &[u8], count: usize) -> Vec<u32> {
        self.add_blocks(wasm, count)
    }

    /// Whether this backend can recompile a block with a better code generator.
    fn can_optimize(&self) -> bool {
        false
    }

    /// Recompile an already-admitted block, keeping its table index. Returns whether
    /// it happened.
    fn promote_block(&mut self, table_index: u32, wasm: &[u8]) -> bool {
        let _ = (table_index, wasm);
        false
    }

    /// Recompile several blocks from one module, in the order `table_indices` gives.
    /// Returns how many were replaced.
    fn promote_blocks(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        let _ = (table_indices, wasm);
        0
    }

    /// As [`CompiledBackend::promote_blocks`], but replaces already-promoted blocks
    /// too, so the whole promoted set can be gathered into one module.
    fn promote_blocks_replacing(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        let _ = (table_indices, wasm);
        0
    }

    /// Whether this backend compiles modules off the caller's thread.
    ///
    /// Admitting a block is ~417,000 host instructions, 94% of it inside the engine's
    /// own compiler, and today the guest waits for all of it. Those instructions do
    /// not go away by moving to another core — the host-instruction metric cannot see
    /// the difference — but throughput is the point, and a machine with spare cores
    /// should not stop the guest to run a compiler.
    fn compiles_async(&self) -> bool {
        false
    }

    /// Hand a module to a compile worker, returning the id its blocks will arrive
    /// under. `None` means the caller must compile inline after all.
    fn submit_dispatched_blocks(&mut self, wasm: Vec<u8>, count: usize) -> Option<u64> {
        let _ = (wasm, count);
        None
    }

    /// Instantiate submission `id`, waiting for the worker if it has not finished.
    ///
    /// The caller decides *when* in guest time to install a module, rather than
    /// installing it whenever a thread finishes — otherwise the guest's path depends
    /// on host scheduling and the machine stops being reproducible.
    fn take_dispatched_blocks(&mut self, id: u64) -> Option<Vec<u32>> {
        let _ = id;
        None
    }

    /// Instantiate every module the workers have finished. Non-blocking.
    fn poll_dispatched_blocks(&mut self) -> Vec<(u64, Vec<u32>)> {
        Vec::new()
    }

    /// Whether this backend can save and reload compiled modules.
    fn can_cache_modules(&self) -> bool {
        false
    }

    /// Register a module's blocks and return its serialized form for a cache.
    fn add_dispatched_blocks_serialized(
        &mut self,
        wasm: &[u8],
        count: usize,
    ) -> (Vec<u32>, Option<Vec<u8>>) {
        (self.add_dispatched_blocks(wasm, count), None)
    }

    /// Register the blocks of a previously serialized module, or `None` if the bytes
    /// cannot be loaded and the caller should compile instead.
    ///
    /// # Safety
    /// `bytes` must come from this backend's own cache; engines do not fully validate
    /// precompiled input.
    unsafe fn add_dispatched_precompiled(
        &mut self,
        bytes: &[u8],
        count: usize,
    ) -> Option<Vec<u32>> {
        let _ = (bytes, count);
        None
    }

    /// Promote a module's blocks and return its serialized form for a cache.
    ///
    /// Promotion is the expensive compile, and caching it is what lets the promotion
    /// threshold be chosen on execution cost rather than on Cranelift's price.
    fn promote_blocks_serialized(
        &mut self,
        table_indices: &[u32],
        wasm: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        (self.promote_blocks(table_indices, wasm), None)
    }

    /// Promote from a previously serialized module, or `None` to compile instead.
    ///
    /// # Safety
    /// As [`CompiledBackend::add_dispatched_precompiled`], and additionally the bytes
    /// must have come from the *optimizing* engine.
    unsafe fn promote_blocks_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        let _ = (table_indices, bytes);
        None
    }

    /// Consolidate a module's blocks and return its serialized form for a cache.
    fn promote_blocks_replacing_serialized(
        &mut self,
        table_indices: &[u32],
        wasm: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        (self.promote_blocks_replacing(table_indices, wasm), None)
    }

    /// Consolidate from a previously serialized module, or `None` to compile instead.
    ///
    /// # Safety
    /// As [`CompiledBackend::promote_blocks_precompiled`].
    unsafe fn promote_blocks_replacing_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        let _ = (table_indices, bytes);
        None
    }

    /// Modules still being compiled.
    fn compiles_in_flight(&self) -> usize {
        0
    }

    /// Whether this backend's dispatchers can chain between blocks of one module
    /// without returning here between them.
    fn can_chain_in_wasm(&self) -> bool {
        false
    }

    /// Record that `rip` is block `index` of module `module_id`, so a dispatcher of
    /// that module can chain to it directly.
    ///
    /// Lossy by design: the map is direct-mapped and a miss simply comes back here.
    fn chain_map_insert(&mut self, rip: u64, module_id: u32, index: u32) {
        let _ = (rip, module_id, index);
    }

    /// Drop any chain-map entry for `rip`. Must be called whenever a RIP stops
    /// meaning the block it was entered for — re-admission, eviction — because a
    /// stale entry would dispatch into the *old* block's function index.
    fn chain_map_forget(&mut self, rip: u64) {
        let _ = rip;
    }

    /// Drop every chain-map entry.
    fn chain_map_clear(&mut self) {}

    /// Chain steps taken inside wasm rather than by returning here.
    fn wasm_chain_steps(&self) -> u64 {
        0
    }

    /// Bound what a dispatcher may run without returning here: a retired-instruction
    /// budget and a block count.
    fn set_chain_limits(&mut self, retired_budget: u64, max_blocks: u32) {
        let _ = (retired_budget, max_blocks);
    }

    /// Set where guest physical address 0 lands in the engine's linear memory.
    fn set_ram_base(&mut self, ram_base: u64);

    /// Drop every cached inline-TLB translation.
    ///
    /// Callers poll `CpuBus::jit_tlb_generation` and call this only when it moves,
    /// rather than flushing per chain.
    fn flush_tlb(&mut self);

    /// Install the translator used on a TLB miss, returning the previous one.
    fn set_translator(&mut self, translator: Option<GuestTranslator>) -> Option<GuestTranslator>;

    /// Guest instructions the most recently run block reported retiring.
    ///
    /// A block used to retire a fixed count — its own length — and the runtime could
    /// assume it. One with side exits cannot, so it reports what actually ran.
    fn last_retired(&self) -> u64;

    /// Install the accessor the slow-path helpers use for guest memory.
    ///
    /// Without one they fall back to resolving addresses inside the engine's linear
    /// memory, which only models RAM.
    fn set_accessor(&mut self, accessor: Option<GuestAccessor>) -> Option<GuestAccessor>;

    /// Install the clock compiled `rdtsc` reads. Without one, a block that reads the
    /// counter must not be admitted.
    fn set_clock(&mut self, clock: Option<GuestClock>) -> Option<GuestClock> {
        let _ = clock;
        None
    }

    /// Install what performs an instruction Tier-1 cannot translate. Without one, a
    /// block containing an assist must not be admitted.
    fn set_assist(&mut self, assist: Option<GuestAssist>) -> Option<GuestAssist> {
        let _ = assist;
        None
    }

    /// Whether an assist is installed.
    fn can_assist(&self) -> bool {
        false
    }

    /// Run a chain of blocks, marshalling CPU state once for the whole chain.
    ///
    /// Marshalling per *block* costs about 293ns against blocks of three to five
    /// instructions, which is why chaining rather than one call per block is part of
    /// the contract rather than an optimisation left to the implementation.
    ///
    /// `lookup` maps a completed block's exit RIP to the next block's index and is
    /// called once per block that completes — never for one that exits to the
    /// interpreter, which is what lets a caller count retirement exactly.
    ///
    /// Returns `(next_rip, blocks_run, exited_to_interpreter)`.
    fn run_chain(
        &mut self,
        entry_index: u32,
        cpu: &mut CpuState,
        lookup: &mut dyn FnMut(u64, u64) -> Option<u32>,
        max_blocks: u32,
    ) -> ChainOutcome;
}

/// A backend that holds no blocks and runs nothing.
///
/// The run loop has to name a tier type on every target, including the ones where no
/// engine is wired up yet. Naming this one keeps those paths type-checking while the
/// `Option` that would reach them stays `None`, which is cheaper than threading `cfg`
/// through an if-else chain and cannot drift out of step with it.
#[derive(Default)]
pub struct NullCompiledBackend;

impl CompiledBackend for NullCompiledBackend {
    fn add_block(&mut self, _wasm: &[u8]) -> u32 {
        unreachable!("NullCompiledBackend never admits a block")
    }

    fn add_blocks(&mut self, _wasm: &[u8], _count: usize) -> Vec<u32> {
        unreachable!("NullCompiledBackend never admits a block")
    }

    fn last_retired(&self) -> u64 {
        0
    }

    fn set_ram_base(&mut self, _ram_base: u64) {}

    fn flush_tlb(&mut self) {}

    fn set_translator(&mut self, _translator: Option<GuestTranslator>) -> Option<GuestTranslator> {
        None
    }

    fn set_clock(&mut self, _clock: Option<GuestClock>) -> Option<GuestClock> {
        None
    }

    fn set_accessor(&mut self, _accessor: Option<GuestAccessor>) -> Option<GuestAccessor> {
        None
    }

    fn run_chain(
        &mut self,
        _entry_index: u32,
        cpu: &mut CpuState,
        _lookup: &mut dyn FnMut(u64, u64) -> Option<u32>,
        _max_blocks: u32,
    ) -> ChainOutcome {
        ChainOutcome {
            next_rip: cpu.rip,
            retired: 0,
            blocks: 0,
            exit_to_interpreter: true,
        }
    }
}

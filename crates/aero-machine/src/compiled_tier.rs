            // Admission costs ~258us once blocks are batched into shared modules,
            // and it is paid once while coverage pays for as long as the machine
            // runs — so the threshold's real job is deciding how *soon* the tier
            // reaches its plateau, not how much it compiles in the end.
            //
            // Measured over the desktop checkpoint, host instructions per guest
            // instruction: at 12 the warm-up window costs 941 and the steady window
            // 638; at 4 they are 630 and 686. Warm-up improves monotonically as this
            // falls, while the steady-state differences sit inside the run-to-run
            // spread, so the low value wins on any session long enough to matter and
            // loses nothing measurable on one that is not.
            //! Compiled Tier-1 execution for the machine.
//!
//! Hot blocks are translated to WASM once and then run as compiled code, chained
//! so one host call executes many of them. The pieces this rests on are measured:
//! compiled code costs about 7ns per guest instruction against the interpreter's
//! ~95ns, but entering and leaving a block costs ~293ns, so chaining is what makes
//! it worth anything at the three-to-five instruction blocks real x86 produces.
//!
//! Guest RAM must already live in the backend's linear memory (see
//! [`crate::jit_ram`]) — that is what lets a compiled load be a load rather than a
//! call back out to the host.
//!
//! Correctness rests on three things:
//!
//! - **Translation comes from the machine's own MMU.** The backend's built-in
//!   behaviour identity-maps, which is wrong for a paging guest.
//! - **The TLB is flushed before every chain.** Tier-1 never compiles `mov cr3` or
//!   `INVLPG` — they are not in its IR and become exits — so page tables cannot
//!   change *inside* a chain. A plain store to a PTE without an `INVLPG` leaves a
//!   stale TLB, which is what real hardware does too.
//! - **Nothing is entered with an event pending.** A chain runs many blocks without
//!   consulting the interrupt controller, so it must not start while delivery is
//!   owed.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use aero_cpu_core::interp::tier0::exec::Tier0BranchSeam;
use aero_cpu_core::mem::CpuBus;
#[cfg(not(target_arch = "wasm32"))]
use aero_cpu_core::state::CpuState;
#[cfg(not(target_arch = "wasm32"))]
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::compiled_backend::CompiledBackend;
use aero_jit_x86::tier1::{BlockLimits, Tier1WasmOptions};
use aero_jit_x86::{
    discover_block_mode, translate_block, Tier1Bus, Tier1WasmCodegen, TLB_FLAG_EXEC,
    TLB_FLAG_IS_RAM, TLB_FLAG_READ, TLB_FLAG_WRITE,
};

use crate::rip_map::{RipMap, RipSet};

#[cfg(not(target_arch = "wasm32"))]
pub type SharedBackend = Rc<RefCell<WasmtimeBackend<CpuState>>>;

/// The tier as the native machine uses it.
#[cfg(not(target_arch = "wasm32"))]
pub type NativeCompiledTier = CompiledTier<WasmtimeBackend<CpuState>>;

/// Longest chain a single entry may run, matching the IR chain's bound.
/// Bytes reserved below guest RAM in the backend's memory for `CpuState` and the
/// JIT context. One wasm page; see `jit_ram` for the native side of this layout.
pub const RAM_BASE: u64 = 0x1_0000;

const MAX_CHAIN_BLOCKS: u32 = 4096;

/// How many chains a partly-filled promotion queue may wait before it is compiled
/// anyway.
///
/// Waiting is what makes a promoted module big enough for its blocks to chain to
/// each other; waiting forever would strand the tail of the hot set on the baseline
/// compiler. This is generous — the queue fills in far fewer chains than this while
/// blocks are still crossing the threshold — so in practice it only fires at the end.
const PROMOTE_STALL_CHAINS: u32 = 20_000;

/// Don't gather the promoted set into one module until it is worth gathering.
const CONSOLIDATE_MIN: usize = 48;

/// And gather it again once it has grown by this much since the last time.
///
/// Every gather recompiles the whole set, so this trades compile time for how much
/// of the hot path can chain without leaving wasm. The set here reaches ~240 blocks,
/// so this is a handful of gathers over a session.
const CONSOLIDATE_STEP: usize = 48;

/// How many chain-step lookups the cache holds. 4096 entries is 128KiB, which stays
/// in L2 while covering the working set of blocks a chain actually walks.
const CHAIN_CACHE_ENTRIES: usize = 4096;



#[derive(Clone, Copy)]
struct ChainCacheEntry {
    /// The block's entry RIP, or `u64::MAX` for an empty slot — no real block entry
    /// can be that, and it keeps the hit test to a single compare.
    rip: u64,
    table_index: u32,
    guest_insts: u32,
    exec_count: u32,
}

/// Whether to keep a per-RIP histogram of refused blocks (`AERO_CT_BAILS`).
///
/// Cached, because the caller is hot: this sits on the path taken by every *refused*
/// offer, which is 588,757 of them across a 200M-instruction run. Reading the
/// environment there means `getenv` walking `environ` with `strncmp` each time, and it
/// showed up in the steady-state profile at 0.6 host instructions per guest
/// instruction under `__strncmp_avx2` — for a diagnostic that is off.
fn bail_histogram_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AERO_CT_BAILS").is_some())
}

impl ChainCacheEntry {
    const EMPTY: Self = Self {
        rip: u64::MAX,
        table_index: 0,
        guest_insts: 0,
        exec_count: 0,
    };

    /// Index for `rip`. Block entries are instruction-aligned but not to any fixed
    /// stride, so the low bits are the ones that vary.
    #[inline]
    fn slot(rip: u64) -> usize {
        (rip as usize >> 2) & (CHAIN_CACHE_ENTRIES - 1)
    }
}

/// A block's compiled form and how many guest instructions it retires.
#[derive(Clone, Copy)]
struct Compiled {
    table_index: u32,
    /// Which admission batch — and therefore which WASM module — this block is in.
    ///
    /// A module's dispatcher can call its own blocks directly, so a chain step whose
    /// successor shares a module could be taken inside WASM instead of returning to
    /// the host. Whether that is worth building depends on how often it happens.
    module_id: u32,
    guest_insts: u64,
    /// How the block ends, so executions can be weighted by terminator. Counted only
    /// under `AERO_CT_EDGES=1`; a block that ends at an *unconditional* jump could
    /// keep going, and whether that is worth building depends on how much of the
    /// running such blocks carry — not on how many of them are admitted.
    ends_in_jmp: bool,
    /// Where the block's conditional terminator falls through to, if it has one.
    ///
    /// Enough to tell a taken edge from a not-taken one, which is what decides
    /// whether fusing along the fall-through follows the hot path or leaves it.
    fallthrough_rip: Option<u64>,
    /// Whether the block can be run twice with the same effect, which is what lets
    /// the verifier execute it under both engines and compare.
    ///
    /// Plain stores qualify: writing the same bytes twice leaves the same memory. A
    /// read-modify-write does not — `add [mem], eax` applied twice adds twice — and
    /// neither does anything that can reach the slow memory helpers, because that is
    /// how a block gets to a device register, where a second read or write is not
    /// free of consequence.
    replayable: bool,
}

/// Where the translator finds the bus for the chain currently running.
///
/// The translator is installed once, not per chain: building a boxed closure for
/// every chain entry is a heap allocation on a path taken millions of times. What
/// actually changes between chains is the bus, so only that is swapped — through a
/// cell the installed closure reads.
type BusHook = Rc<Cell<Option<(*mut (), fn(*mut (), u64, bool) -> Option<(u64, bool)>)>>>;

/// Where the slow-path memory helpers find the bus for the chain currently running.
///
/// Same arrangement as [`BusHook`], for the same reason: the closure is installed
/// once and only the bus it points at changes.
type AccessHook = Rc<Cell<Option<(*mut (), fn(*mut (), u64, u32, Option<u64>) -> Option<u64>)>>>;

/// Points at the running chain's `TimeSource`, a field of `CpuCore` disjoint from
/// the `CpuState` the backend borrows. Set for the duration of a chain, cleared
/// after, like the bus hooks.
type ClockHook = Rc<Cell<Option<*mut aero_cpu_core::time::TimeSource>>>;

/// What performing one guest instruction on the host needs: the interpreter's assist
/// context, the machine's clock, and the bus. All three are borrowed by the batch for
/// the duration of a chain, and handed over the same way the bus already is.
type AssistHook = Rc<
    Cell<
        Option<(
            *mut aero_cpu_core::assist::AssistContext,
            *mut aero_cpu_core::time::TimeSource,
            *mut (),
            *mut aero_cpu_core::state::CpuState,
            fn(
                *mut aero_cpu_core::assist::AssistContext,
                *mut aero_cpu_core::time::TimeSource,
                *mut (),
                *mut aero_cpu_core::state::CpuState,
                &mut aero_jit_x86::compiled_backend::AssistRegs,
            ) -> Option<u64>,
        )>,
    >,
>;

/// A block translated and waiting to be compiled with others into one module.
struct PendingBlock {
    rip: u64,
    ir: aero_jit_x86::tier1::ir::IrBlock,
    guest_insts: u64,
    ends_in_jmp: bool,
    fallthrough_rip: Option<u64>,
    replayable: bool,
}

pub struct CompiledTier<E: CompiledBackend> {
    backend: Rc<RefCell<E>>,
    pending: Vec<PendingBlock>,
    pending_set: RipSet,
    batch_size: usize,
    bus_hook: BusHook,
    access_hook: AccessHook,
    clock_hook: ClockHook,
    assist_hook: AssistHook,
    /// What compiled `RDTSC` last read, so the architectural copy can be settled
    /// after the chain rather than written through a second pointer while the
    /// backend holds `&mut CpuState`.
    last_tsc: Rc<Cell<Option<u64>>>,
    /// Whether an instruction Tier-1 cannot translate is handed to the host from
    /// inside the block, instead of ending it.
    allow_assists: bool,
    /// Whether `RDTSC` is translated rather than ending a block.
    compile_rdtsc: bool,
    blocks: RipMap<Compiled>,
    hotness: RipMap<u32>,
    /// Entries that failed to compile. Remembered so a hot block that cannot be
    /// translated is not rediscovered and re-attempted on every hit.
    rejected: RipSet,
    hot_threshold: u32,
    chain_miss_threshold: u32,
    max_blocks: usize,
    /// Longest chain a single seam entry may run.
    ///
    /// A long chain does not consult the interrupt controller, so shortening it
    /// brings interrupt delivery back towards where the interpreter would put it.
    /// `AERO_CT_MAX_CHAIN=1` makes the tier run one block per entry, which is how a
    /// divergence gets classified: if it survives that, it is a translation bug
    /// rather than a timing difference.
    max_chain_blocks: u32,
    /// Drop cached translations on every chain entry (`AERO_CT_ALWAYS_FLUSH_TLB=1`).
    ///
    /// Correct but slower. If a divergence disappears under this, the generation
    /// counter is missing an invalidation the guest is relying on.
    always_flush_tlb: bool,
    /// Decode past conditional branches into superblocks (`AERO_CT_NO_SUPERBLOCKS=1` disables).
    ///
    /// On by default since 2026-08-29, which reverses an earlier measurement. Nearly all
    /// compiled-tier time is per-block *entry* cost — ~270 host instructions for a
    /// Cranelift block against 2.25 per guest ALU op — so the thing that pays is fewer,
    /// longer blocks. Superblocks take block entries from 43.7M to 26.7M over a 200M
    /// instruction run, guest instructions per block from 4.38 to 6.80, and the in-wasm
    /// chain rate from 56.9% to 67.9%.
    ///
    /// They cost coverage (95.5% -> 90.9%): a longer span is likelier to contain an
    /// instruction Tier-1 refuses, and refusing it refuses the whole block. That is why
    /// warm-up is *worse* — 485.8 against 350.3 — while every window after it is better:
    /// 84.3/86.8 steady, 71.1/81.2 mid, 67.8/78.7 late, 90.8/93.8 at 200-300M,
    /// 75.2/82.6 at 250-350M, 81.0/92.5 at 500-600M. Six windows, mean ~9%.
    ///
    /// The crossover is around 1.4G guest instructions, which is seconds of real use, so
    /// this is right for a machine that runs and wrong for a machine that only boots.
    superblocks: bool,
    /// Decode through a direct `call` into the callee (`AERO_CT_FOLLOW_CALLS=1`).
    follow_calls: bool,
    /// Perform non-RAM accesses through the bus instead of bailing
    /// (`AERO_CT_MMIO_VIA_BUS=1`).
    mmio_via_bus: bool,
    /// One inline fast path per memory access. On by default;
    /// `AERO_CT_NESTED_MEM_PATHS=1` restores the per-case inline paths.
    single_fast_path: bool,
    /// Keep page-crossing accesses on the inline path (`AERO_CT_CROSS_PAGE=1`).
    /// Off by default: it costs more to compile than it saves to run.
    cross_page_fastpath: bool,
    /// Longest block Tier-1 will decode (`AERO_CT_MAX_BLOCK_INSTS`).
    max_block_insts: usize,
    /// Where each block went, when `AERO_CT_EDGES=1`.
    ///
    /// Fusing blocks only pays if one successor dominates: a trace that follows a
    /// branch taken half the time exits half the time and is merely bigger. This
    /// counts the real edges so that is a measurement rather than an assumption.
    edges: std::collections::HashMap<(u64, u64), u64>,
    record_edges: bool,
    /// Check every store-free block against Tier-0 (`AERO_CT_VERIFY=1`).
    verify: bool,
    verify_failures: u64,
    rejected_slow_mem: u64,
    verify_reported: std::collections::HashSet<u64>,
    executed_blocks: u64,
    executed_guest_instructions: u64,
    compiled_blocks: u64,
    /// Wall time inside compiled chains, and inside admission.
    ///
    /// Divergence makes whole-run comparisons between configurations unsound — the
    /// two end up executing different guest code — so cost is measured per path
    /// *within* one run instead, which nothing about timing can distort.
    run_nanos: u128,
    compile_nanos: u128,
    /// Flag writes removed as dead across all admitted blocks.
    flag_writes_dropped: u64,
    /// MMU generation the inline TLB was last flushed for.
    last_tlb_generation: Option<u64>,
    /// Why admission was refused, so it is clear where coverage is lost.
    rejected_empty: u64,
    rejected_bails: u64,
    /// Mnemonics that ended a block in a bail, when `AERO_CT_BAILS` is set.
    bail_mnemonics: std::collections::HashMap<String, u64>,
    rejected_cap: u64,
    /// Chain steps that stopped because the successor was not compiled.
    chain_misses: u64,
    /// Chain steps taken, and how many stayed inside one module.
    chain_steps: u64,
    chain_steps_same_module: u64,
    next_module_id: u32,
    /// Executions per block, and which have already been recompiled.
    ///
    /// Execution is concentrated enough to make a second code generator worth its
    /// compile time: the busiest 1% of blocks account for 82% of all block
    /// executions, and the busiest 5% for 92%. So the ones that cross
    /// `promote_threshold` are handed to Cranelift and everything else stays with
    /// Winch, which admits about seven times faster.
    count_execs: bool,
    promote_threshold: u64,
    /// Total WASM emitted, admissions and promotions together.
    ///
    /// Worth knowing because it is the compiled tier's footprint in the host's
    /// caches, and that is a cost the host-instruction metric cannot see at all.
    wasm_bytes: u64,
    /// How many hot blocks to recompile in one Cranelift module.
    promote_batch: usize,
    promoted: RipSet,
    promotions: u64,
    promote_nanos: u128,
    /// How many resume points were armed, and how often one was reached.
    resume_sites: u64,
    resume_hits: u64,
    resume_enabled: bool,
    /// Whether the tier steps over the instructions that stop a chain. See
    /// [`CompiledTier::run`].
    step_over_enabled: bool,
    /// Guest instructions the tier executed under Tier-0 to get past a refusal.
    stepped_over: u64,
    /// Instructions performed by the host from inside a compiled block.
    assists: Rc<Cell<u64>>,
    blocks_with_assists: u64,
    /// How many it will step in one go before leaving the rest to the interpreter.
    step_over_max: u32,
    /// RIPs that crossed the threshold and are waiting to be recompiled. Promotion
    /// cannot happen inside the chain that noticed, because the backend is borrowed
    /// for its duration.
    promote_queue: Vec<u64>,
    /// Where compiled modules are kept between runs, if anywhere.
    ///
    /// Compiling a block is ~390,000 host instructions; loading one that was compiled
    /// before is under 5,000. The guest image and the checkpoint are the same every
    /// run here, so the same blocks are compiled from scratch every time for no
    /// reason. Off unless `AERO_CT_CACHE_DIR` names a directory, because it writes
    /// files, and because loading a precompiled module is only safe for data this
    /// program wrote itself.
    module_cache: Option<std::path::PathBuf>,
    /// How long to wait for a promotion batch to fill; see `PROMOTE_STALL_CHAINS`.
    promote_stall_chains: u32,
    promote_cache_hits: u64,
    promote_cache_misses: u64,
    cache_hits: u64,
    cache_misses: u64,
    /// Whether modules are compiled on worker threads instead of inline.
    async_compile: bool,
    /// How far ahead in guest instructions a module submitted now is installed.
    ///
    /// Long enough that the compile has finished by then — so the guest is not
    /// stopped waiting for it — and short enough that the blocks are not interpreted
    /// for meaningfully longer than they would have been.
    async_admit_delay: u64,
    /// Offers made, which is the clock deadlines are expressed in.
    admission_clock: u64,
    /// The earliest clock value at which anything in flight is due, so the offer
    /// path can skip the whole check with one comparison.
    next_admission_due: u64,
    /// Batches handed to the workers and not yet registered, by submission id, each
    /// with the guest-instruction count it is to be installed at.
    in_flight: std::collections::HashMap<u64, (u32, Option<u32>, Vec<PendingBlock>, u64)>,
    /// How admitted blocks end: jmp, jcc, call, ret, exit-to-interpreter, limit.
    end_kinds: [u64; 6],
    /// Chain steps into a block that ends at an unconditional jump. Counted only under
    /// `AERO_CT_EDGES=1`.
    jmp_block_steps: u64,
    /// Chains entered, and how many of those had too little budget left to chain
    /// inside wasm at all.
    chains: u64,
    chains_no_budget: u64,
    /// Whether the promoted set is periodically gathered into one module.
    consolidate: bool,
    /// Every promoted RIP, in the order it was promoted.
    promoted_order: Vec<u64>,
    /// How many had been promoted when the set was last gathered into one module.
    consolidated_len: usize,
    consolidations: u64,
    /// Whether admission modules chain too, not just promoted ones. Off, and measured
    /// again on the current baseline rather than inherited:
    ///
    /// | configuration | promoted | host insns/guest insn |
    /// |---|---|---|
    /// | off (shipped) | 270 | **49.83** |
    /// | on, `chain_sample_period` 64 | 368 | 65.93 |
    /// | on, `chain_sample_period` 8 | — | 68.98 |
    /// | on, no sampling | 216 | 76.13 |
    ///
    /// `flush_pending` records the original reason: chaining hides a block from
    /// promotion. `chain_sample_period` fixes exactly that — promotions go from 216 to
    /// 368, more than the shipped 270 — and the configuration is still 32% slower, so
    /// hiding promotions was never the whole cost. Whatever the rest of it is, it is
    /// larger than the block entries chaining saves across the admitted set.
    ///
    /// Worth noting for anyone re-testing this: blocks-per-chain is *anti*-correlated
    /// with the metric here. The no-sampling row has the healthiest-looking chains of
    /// the three (34.54 blocks against the shipped 34.21) and the worst result, while
    /// the sampled row looks wrecked at 22.23 and does better. Reading the derived
    /// counters instead of the metric points the wrong way twice.
    chain_admission: bool,
    /// Run one chain in every `n` unchained, so promotion can still see block
    /// executions that chaining would otherwise hide. Zero disables sampling.
    ///
    /// Chaining inside wasm is what makes block entry cheap, but a block entered by a
    /// module's own dispatcher never reaches the host's chain-step closure, and that
    /// closure is what counts executions and queues a block for Cranelift. Chaining
    /// admission modules therefore used to cost more than it saved: the hottest blocks
    /// became invisible to promotion precisely because they were hot.
    ///
    /// Sampling breaks that trade. Every `n`-th chain runs with the in-wasm chain bound
    /// to a single block, so every step goes back through the closure and is counted;
    /// the counts from those chains are weighted by `n` to stand for the ones that were
    /// not observed. Promotion sees a scaled picture at a `1/n` accounting cost, and the
    /// other `n-1` chains keep the whole benefit of chaining.
    ///
    /// It does what it claims — promotions rise from 216 to 368, past the 270 that
    /// chaining was costing — and it does not rescue `chain_admission`, which loses
    /// regardless. Finer is worse (`8` measures 68.98 against `64`'s 65.93): the extra
    /// single-block chains cost more than the resolution buys. Only reachable with
    /// `chain_admission` on, so this is inert in the shipped configuration.
    chain_sample_period: u32,
    /// Chains seen while the promotion queue sat below a full group.
    promote_stall: u32,
    /// Whether the backend's dispatchers can chain between blocks of one module.
    ///
    /// Only *promoted* blocks are entered into the chain map, and deliberately: a
    /// block reached inside wasm never reaches the host's chain-step closure, so it
    /// never accumulates the execution count that promotion is triggered by. Putting
    /// blocks in the map before they are promoted would starve tier-up of exactly
    /// the blocks worth promoting.
    chain_in_wasm: bool,
    /// The RIP the interpreter should offer the tier at, outside a taken branch.
    ///
    /// Set when a chain stops at a block the tier refused: it names the instruction
    /// *after* the one that could not be translated, so the interpreter hands back
    /// as soon as it has stepped over it rather than running on to the next branch.
    /// Shared with the seam, which reads it once per interpreted instruction.
    resume_at: Rc<Cell<u64>>,
    /// Direct-mapped cache of chain-step lookups.
    ///
    /// The step's map lookup is the busiest host code left in the profile — once per
    /// chain step, 43 million times a session. Hot blocks stay resident here, so the
    /// hash and probe are skipped for nearly all of them.
    ///
    /// The execution count lives in the entry too, which is what keeps a hit from
    /// having to touch the map at all. A block evicted before it is promoted loses
    /// its count — that is the right trade: a block cold enough to be evicted from a
    /// cache this size is not one worth recompiling.
    chain_cache: Box<[ChainCacheEntry]>,
    /// The IR each block was compiled from, kept so it can be recompiled later.
    ///
    /// Re-deriving it at promotion time is not an option: the entry is a *linear*
    /// address, Windows changes CR3 constantly, and the same address in another
    /// process holds different code. Keeping the bytes and re-decoding them is
    /// closer but still not the same block — a block that ended at the decoder's
    /// instruction limit re-decodes into a shorter one when the saved bytes run out.
    /// Keeping the IR is exact, and skips the decode as well. Only kept when there
    /// is a code generator to promote into.
    block_ir: std::collections::HashMap<u64, aero_jit_x86::tier1::ir::IrBlock>,
    /// How often the Tier-0 seam offered a branch target, and why it was declined.
    ///
    /// Coverage plateaus near 60%; this says which of the three reasons accounts for
    /// the other 40%, so effort goes where the instructions actually are.
    offered: u64,
    declined_event: u64,
    declined_uncompiled: u64,
    /// Of those, offers whose RIP is already known to be uncompilable. This splits
    /// the coverage wall in two: code Tier-1 *cannot* take (a missing opcode, and no
    /// admission policy will ever help) from code it simply has not taken yet.
    declined_rejected: u64,
    /// Offers per rejected entry, so the missing opcodes can be ranked by how often
    /// the guest actually arrives at them rather than by how many distinct sites
    /// exist. Only filled when `AERO_CT_BAILS` is set.
    rejected_hits: std::collections::HashMap<u64, u64>,
    /// Bytes of the instruction that stopped translation, kept per entry RIP so the
    /// report can name the opcode without a bus to read from.
    rejected_bytes: std::collections::HashMap<u64, String>,
    /// Chain steps where a block's successor is the block itself.
    ///
    /// A self-looping block re-enters through the host and reloads its registers
    /// from `CpuState` every iteration. If most steps look like this, emitting the
    /// loop inside the compiled function would remove both costs.
    self_loop_steps: u64,
}

impl<E: CompiledBackend> CompiledTier<E> {
    /// Build a tier over `backend`, with guest RAM at [`RAM_BASE`] in its memory.
    pub fn new(backend: Rc<RefCell<E>>) -> Self {
        Self::new_with_ram_base(backend, RAM_BASE)
    }

    /// Turn in-wasm chaining off for this tier.
    ///
    /// Exists so the browser can A/B it. Natively `AERO_CT_NO_WASM_CHAIN` does the job,
    /// but a wasm32 build has no environment to read, so without a setter the only way
    /// to compare is to rebuild — and comparing two builds is how an A/B ends up
    /// measuring something other than the knob.
    pub fn disable_wasm_chaining(&mut self) {
        self.chain_in_wasm = false;
        self.chain_admission = false;
    }

    /// Build a tier whose guest RAM starts at `ram_base` in the backend's memory.
    ///
    /// The browser puts it somewhere else: there, guest RAM *is* the module's linear
    /// memory by construction rather than a region carved out of the engine's.
    pub fn new_with_ram_base(backend: Rc<RefCell<E>>, ram_base: u64) -> Self {
        let can_optimize = CompiledBackend::can_optimize(&*backend.borrow());
        let can_chain = CompiledBackend::can_chain_in_wasm(&*backend.borrow());
        let compiles_async = CompiledBackend::compiles_async(&*backend.borrow());
        let can_cache = CompiledBackend::can_cache_modules(&*backend.borrow());
        CompiledBackend::set_ram_base(&mut *backend.borrow_mut(), ram_base);

        let bus_hook: BusHook = Rc::new(Cell::new(None));
        let hook = bus_hook.clone();
        backend
            .borrow_mut()
            .set_translator(Some(Box::new(move |vaddr, access| {
                let (ctx, thunk) = hook.get()?;
                let write = access == aero_jit_x86::abi::MMU_ACCESS_WRITE as u32;
                let (page, is_ram) = thunk(ctx, vaddr, write)?;
                let mut flags = TLB_FLAG_EXEC;
                // A write translation grants read as well: x86 has no write-only
                // page, so anything this translator just let a write through is by
                // construction readable. Without that, a page written and then read
                // misses on the read and re-translates — and writing then reading is
                // what a stack and every buffer does.
                //
                // The converse is *not* safe, for two reasons. A read says nothing
                // about whether the page is writable, so a write to a read-only page
                // has to miss in order to fault. And the MMU sets the accessed and
                // dirty bits as it translates — granting write here on the strength of
                // the page tables alone would let the first write through the inline
                // fast path without ever setting `D`, and the guest OS believes that
                // bit. The asymmetry is deliberate; don't "finish the job".
                flags |= if write {
                    TLB_FLAG_WRITE | TLB_FLAG_READ
                } else {
                    TLB_FLAG_READ
                };
                if is_ram {
                    flags |= TLB_FLAG_IS_RAM;
                }
                Some((page, flags))
            })));

        // Slow-path guest accesses go to the machine's own bus, so a device register
        // is read and written as one, with its side effects, rather than being
        // resolved against linear memory that does not back it.
        let access_hook: AccessHook = Rc::new(Cell::new(None));
        let hook = access_hook.clone();
        backend
            .borrow_mut()
            .set_accessor(Some(Box::new(move |addr, size, value| {
                let (ctx, thunk) = hook.get()?;
                thunk(ctx, addr, size, value)
            })));

        // Compiled `RDTSC` reads the machine's clock through here, and leaves the
        // architectural copy behind as the interpreter's assist does.
        let clock_hook: ClockHook = Rc::new(Cell::new(None));
        let last_tsc: Rc<Cell<Option<u64>>> = Rc::new(Cell::new(None));
        let hook = clock_hook.clone();
        let seen = last_tsc.clone();
        // One guest instruction, performed by Tier-0 against the state the block just
        // wrote to linear memory. What makes it worth the round trip is that the block
        // then *carries on*: the instructions Tier-1 cannot translate come in short
        // runs, so a block that ends at one lands on another.
        let assist_hook: AssistHook = Rc::new(Cell::new(None));
        let a_hook = assist_hook.clone();
        let assists: Rc<Cell<u64>> = Rc::new(Cell::new(0));
        let n_assists = assists.clone();
        backend
            .borrow_mut()
            .set_assist(Some(Box::new(move |regs| {
                n_assists.set(n_assists.get() + 1);
                let (ctx, time, bus, state, thunk) = a_hook.get()?;
                thunk(ctx, time, bus, state, regs)
            })));

        // Asked after the hook is installed, not before: it reports whether the
        // backend accepted one.
        let can_assist = CompiledBackend::can_assist(&*backend.borrow());

        backend.borrow_mut().set_clock(Some(Box::new(move || {
            let Some(time) = hook.get() else {
                return 0;
            };
            // Safety: the pointer is `&mut cpu.time`, a field disjoint from the
            // `&mut cpu.state` the backend holds; it is set for the duration of the
            // chain and cleared after, and nothing else touches it meanwhile.
            let tsc = unsafe { &mut *time }.read_tsc();
            seen.set(Some(tsc));
            tsc
        })));

        Self {
            backend,
            clock_hook,
            assist_hook,
            last_tsc,
            // `AERO_RDTSC_QUANTUM` puts tight-poll detection in the interpreter's
            // assist, which cannot see reads a compiled block makes — so that mode
            // keeps `RDTSC` in the interpreter, where its bookkeeping works.
            // Off, and not because it does not work — it does, and it is the only
            // mechanism that takes `bails_to_interp` to zero. It is incomplete: an
            // assisted instruction may write CR3, CR0, CR4 or EFER, and the block
            // then carries on against an inline TLB filled under the old translation.
            // The guest gets a few hundred thousand instructions further and resets.
            // Finishing it means deciding what invalidating that TLB means from
            // inside a block, which is a design question and not a patch.
            // `AERO_CT_ASSISTS=1` to work on it.
            allow_assists: can_assist && std::env::var_os("AERO_CT_ASSISTS").is_some(),
            compile_rdtsc: std::env::var_os("AERO_CT_NO_RDTSC").is_none()
                && std::env::var("AERO_RDTSC_QUANTUM")
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .is_none_or(|v| v == 0),
            pending: Vec::new(),
            pending_set: RipSet::default(),
            batch_size: env_usize("AERO_CT_BATCH", 16),
            bus_hook,
            access_hook,
            blocks: RipMap::default(),
            hotness: RipMap::default(),
            rejected: RipSet::default(),
            // Admission costs ~600us and compiled code saves ~110ns per guest
            // instruction, so a block must be entered a few hundred times to repay
            // being compiled. Swept against the desktop checkpoint over 120M
            // instructions: 12 was the best of 4/12/24/48/96. Below it, admission
            // of blocks that never run hot outweighs what they save; above it,
            // coverage falls away faster than admission cost does.
            hot_threshold: env_u32("AERO_CT_HOT_THRESHOLD", 4),
            chain_miss_threshold: env_u32("AERO_CT_CHAIN_THRESHOLD", 4),
            // 65536, not 32768: the desktop working set is ~51,600 blocks, and a
            // cache that holds it is worth 13.5% — 519.0 against 599.9 host
            // instructions per guest instruction, interleaved and reproducible.
            // Larger changes nothing, because nothing more asks to be compiled.
            max_blocks: env_usize("AERO_CT_MAX_BLOCKS", 65536),
            executed_blocks: 0,
            executed_guest_instructions: 0,
            compiled_blocks: 0,
            run_nanos: 0,
            compile_nanos: 0,
            flag_writes_dropped: 0,
            last_tlb_generation: None,
            rejected_empty: 0,
            rejected_bails: 0,
            bail_mnemonics: std::collections::HashMap::new(),
            rejected_cap: 0,
            chain_misses: 0,
            chain_steps: 0,
            chain_steps_same_module: 0,
            next_module_id: 0,
            // Counting is needed either to report concentration or to act on it.
            count_execs: std::env::var_os("AERO_CT_HOTNESS").is_some()
                || can_optimize,
            // 2,000 executions: enough that a block is plainly hot rather than
            // merely warm, low enough that the ones carrying the run reach it early.
            promote_threshold: u64::from(env_u32("AERO_CT_PROMOTE", 2000)),
            // One a module unless the dispatcher can chain: see `drain_promotions`.
            promote_batch: env_usize(
                "AERO_CT_PROMOTE_BATCH",
                if std::env::var_os("AERO_CT_NO_WASM_CHAIN").is_none() && can_optimize {
                    // Big enough to hold the set that matters: the busiest ~190
                    // blocks are 94% of chain steps, and they cross the promotion
                    // threshold earliest, so they land in the first module together.
                    256
                } else {
                    1
                },
            ),
            wasm_bytes: 0,
            promoted: RipSet::default(),
            promotions: 0,
            promote_nanos: 0,
            promote_queue: Vec::new(),
            // Off by default. It cuts the work on the CPU thread by 17% in steady
            // state and 32% over a whole session — real throughput on a machine with
            // spare cores — but the machine stops being reproducible run to run, and
            // every oracle and measurement here depends on that. `AERO_CT_ASYNC=1`.
            promote_stall_chains: env_u32("AERO_CT_PROMOTE_STALL", PROMOTE_STALL_CHAINS),
            promote_cache_hits: 0,
            promote_cache_misses: 0,
            module_cache: std::env::var_os("AERO_CT_CACHE_DIR")
                .map(std::path::PathBuf::from)
                .filter(|dir| can_cache && std::fs::create_dir_all(dir).is_ok()),
            cache_hits: 0,
            cache_misses: 0,
            async_compile: compiles_async && std::env::var_os("AERO_CT_ASYNC").is_some(),
            async_admit_delay: u64::from(env_u32("AERO_CT_ADMIT_DELAY", 64)),
            admission_clock: 0,
            next_admission_due: u64::MAX,
            in_flight: std::collections::HashMap::new(),
            end_kinds: [0; 6],
            jmp_block_steps: 0,
            chains: 0,
            chains_no_budget: 0,
            consolidate: std::env::var_os("AERO_CT_CONSOLIDATE").is_some(),
            promoted_order: Vec::new(),
            consolidated_len: 0,
            consolidations: 0,
            promote_stall: 0,
            // A backend that cannot promote has nothing to hide a block *from*, so the
            // objection recorded above does not apply to it and admission modules are
            // the only thing it could ever chain. That is the browser: no Cranelift, so
            // without this every block execution crosses the JS boundary.
            chain_admission: can_chain
                && (!can_optimize || std::env::var_os("AERO_CT_CHAIN_ADMISSION").is_some()),
            // Read directly rather than through `env_u32`, which filters out `0` and
            // hands back the default — so `AERO_CT_CHAIN_SAMPLE=0` would silently mean
            // 64 and the "no sampling" control would be a second copy of the sampled
            // run. Zero has to mean zero for the A/B to exist at all.
            chain_sample_period: std::env::var("AERO_CT_CHAIN_SAMPLE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),
            // Not gated on `can_optimize`: chaining is a property of the backend's
            // dispatcher, not of whether it has a second compiler. Requiring both kept
            // the browser on the unchained path, which is where its throughput sat.
            chain_in_wasm: can_chain && std::env::var_os("AERO_CT_NO_WASM_CHAIN").is_none(),
            block_ir: std::collections::HashMap::new(),
            chain_cache: vec![ChainCacheEntry::EMPTY; CHAIN_CACHE_ENTRIES].into_boxed_slice(),
            resume_at: Rc::new(Cell::new(u64::MAX)),
            resume_hits: 0,
            resume_sites: 0,
            // Off, and measured: 184.8 against 182.2. Handing back right after the
            // refused instruction works — but there is no compiled block at that
            // point either, so the seam declines and the interpreter carries on
            // regardless, and the extra offers are pure cost. Making it pay would
            // mean admitting a block there, which is the same trade that has lost
            // every time it has been measured.
            resume_enabled: std::env::var_os("AERO_CT_RESUME").is_some(),
            step_over_enabled: std::env::var_os("AERO_CT_NO_STEP_OVER").is_none(),
            stepped_over: 0,
            assists,
            blocks_with_assists: 0,
            step_over_max: env_u32("AERO_CT_STEP_OVER_MAX", 1),
            max_chain_blocks: env_u32("AERO_CT_MAX_CHAIN", MAX_CHAIN_BLOCKS),
            always_flush_tlb: std::env::var_os("AERO_CT_ALWAYS_FLUSH_TLB").is_some(),
            cross_page_fastpath: std::env::var_os("AERO_CT_CROSS_PAGE").is_some(),
            superblocks: std::env::var_os("AERO_CT_NO_SUPERBLOCKS").is_none(),
            follow_calls: std::env::var_os("AERO_CT_FOLLOW_CALLS").is_some(),
            mmio_via_bus: std::env::var_os("AERO_CT_MMIO_VIA_BUS").is_some(),
            single_fast_path: std::env::var_os("AERO_CT_NESTED_MEM_PATHS").is_none(),
            max_block_insts: env_usize(
                "AERO_CT_MAX_BLOCK_INSTS",
                BlockLimits::default().max_insts,
            ),
            edges: std::collections::HashMap::new(),
            record_edges: std::env::var_os("AERO_CT_EDGES").is_some(),
            verify: std::env::var_os("AERO_CT_VERIFY").is_some(),
            verify_failures: 0,
            rejected_slow_mem: 0,
            verify_reported: std::collections::HashSet::new(),
            self_loop_steps: 0,
            offered: 0,
            declined_event: 0,
            declined_uncompiled: 0,
            declined_rejected: 0,
            rejected_hits: std::collections::HashMap::new(),
            rejected_bytes: std::collections::HashMap::new(),
        }
    }

    /// `(empty, bails-to-interpreter, at-cap, chain-misses)`.
    pub fn rejection_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.rejected_empty,
            self.rejected_bails,
            self.rejected_cap,
            self.chain_misses,
        )
    }

    /// How many store-free blocks disagreed with Tier-0 under `AERO_CT_VERIFY`.
    pub fn verify_failures(&self) -> u64 {
        self.verify_failures
    }

    /// How concentrated the control-flow edges are.
    ///
    /// Returns `(transitions, to_the_top_successor, blocks_with_edges)`. The ratio of
    /// the first two is what a trace following the hottest edge would get right.
    pub fn edge_concentration(&self) -> (u64, u64, usize) {
        let mut per_block: std::collections::HashMap<u64, Vec<u64>> =
            std::collections::HashMap::new();
        for ((from, _to), n) in &self.edges {
            per_block.entry(*from).or_default().push(*n);
        }
        let mut total = 0u64;
        let mut top = 0u64;
        for counts in per_block.values() {
            total += counts.iter().sum::<u64>();
            top += counts.iter().copied().max().unwrap_or(0);
        }
        (total, top, per_block.len())
    }

    /// How often the hottest edge out of a block goes backwards.
    ///
    /// Returns `(hot_edges, of_those_backward)`. A backward edge is a loop's
    /// back-edge, and "backward branches are taken" is the classic static prediction
    /// — if the hot edge is usually backward, a trace can follow it without any
    /// profile at all.
    /// Whether the hottest edge out of a conditionally-ending block is its
    /// fall-through.
    ///
    /// Returns `(hot_edges_from_conditional_blocks, of_those_fallthrough)`. This is
    /// the number that decides trace policy: fusing along the fall-through only
    /// follows the hot path to the extent this is high.
    pub fn hot_edge_is_fallthrough(&self) -> (u64, u64) {
        let mut best: std::collections::HashMap<u64, (u64, u64)> =
            std::collections::HashMap::new();
        for ((from, to), n) in &self.edges {
            let slot = best.entry(*from).or_insert((0, 0));
            if *n > slot.0 {
                *slot = (*n, *to);
            }
        }
        let mut hot = 0u64;
        let mut fallthrough = 0u64;
        for (from, (n, to)) in &best {
            let Some(ft) = self.blocks.get(from).and_then(|c| c.fallthrough_rip) else {
                continue;
            };
            hot += n;
            if *to == ft {
                fallthrough += n;
            }
        }
        (hot, fallthrough)
    }

    pub fn hot_edge_direction(&self) -> (u64, u64) {
        let mut best: std::collections::HashMap<u64, (u64, u64)> =
            std::collections::HashMap::new();
        for ((from, to), n) in &self.edges {
            let slot = best.entry(*from).or_insert((0, 0));
            if *n > slot.0 {
                *slot = (*n, *to);
            }
        }
        let mut hot = 0u64;
        let mut backward = 0u64;
        for (from, (n, to)) in &best {
            hot += n;
            if to <= from {
                backward += n;
            }
        }
        (hot, backward)
    }

    /// Blocks refused because they could reach the slow memory helpers.
    pub fn rejected_slow_mem(&self) -> u64 {
        self.rejected_slow_mem
    }

    pub fn self_loop_steps(&self) -> u64 {
        self.self_loop_steps
    }

    /// `(offered, declined-for-a-pending-event, declined-because-uncompiled)`.
    pub fn seam_counts(&self) -> (u64, u64, u64) {
        (self.offered, self.declined_event, self.declined_uncompiled)
    }

    /// Offers that landed on a RIP Tier-1 has already refused, and the busiest such
    /// entries with the bytes of the instruction that stopped translation.
    /// How concentrated block execution is: the share of all executions covered by
    /// the busiest 1%, 5%, 10% and 25% of blocks.
    pub fn execution_concentration(&self) -> Option<(u64, usize, Vec<(usize, f64)>)> {
        if self.blocks.is_empty() {
            return None;
        }
        // Read from the chain cache, which is where the counts live. It holds the
        // blocks a chain actually walks, which is exactly the population this is
        // asking about.
        let mut counts: Vec<u64> = self
            .chain_cache
            .iter()
            .filter(|e| e.rip != u64::MAX && e.exec_count > 0)
            .map(|e| u64::from(e.exec_count))
            .collect();
        if counts.is_empty() {
            return None;
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = counts.iter().sum();
        let n = counts.len();
        let shares = [1usize, 5, 10, 25, 50]
            .into_iter()
            .map(|pct| {
                let take = (n * pct / 100).max(1);
                let covered: u64 = counts[..take.min(n)].iter().sum();
                (take, 100.0 * covered as f64 / total as f64)
            })
            .collect();
        Some((total, n, shares))
    }

    /// Total WASM emitted, in bytes.
    pub fn wasm_bytes(&self) -> u64 {
        self.wasm_bytes
    }

    /// Blocks recompiled by the better code generator, and what that cost.
    pub fn promotion_counts(&self) -> (u64, f64) {
        let us = self.promote_nanos as f64 / 1000.0;
        (
            self.promotions,
            if self.promotions == 0 {
                0.0
            } else {
                us / self.promotions as f64
            },
        )
    }

    /// The shared cell the seam reads to know where to offer outside a branch.
    pub fn resume_cell(&self) -> Rc<Cell<u64>> {
        self.resume_at.clone()
    }

    /// Whether a resume point is currently armed and about to fire.
    pub fn resume_armed(&self) -> bool {
        self.resume_at.get() > 0
    }

    pub fn note_resume_hit(&mut self) {
        self.resume_hits += 1;
    }

    /// Guest instructions the tier stepped over under Tier-0 to get past a block it
    /// had refused.
    /// Instructions performed by the host from inside a compiled block.
    pub fn assists(&self) -> u64 {
        self.assists.get()
    }

    /// Blocks admitted that contain a host-performed instruction.
    pub fn blocks_with_assists(&self) -> u64 {
        self.blocks_with_assists
    }

    pub fn stepped_over(&self) -> u64 {
        self.stepped_over
    }

    /// Resume points armed, and how many were reached.
    pub fn resume_counts(&self) -> (u64, u64) {
        (self.resume_sites, self.resume_hits)
    }

    /// Chain steps taken, and how many stayed inside one WASM module.
    pub fn chain_locality(&self) -> (u64, u64) {
        (self.chain_steps, self.chain_steps_same_module)
    }

    pub fn rejected_offer_counts(&self) -> (u64, Vec<(String, u64)>) {
        let mut top: Vec<_> = self.rejected_hits.iter().map(|(&r, &n)| (r, n)).collect();
        top.sort_by_key(|&(rip, n)| (std::cmp::Reverse(n), rip));
        top.truncate(16);
        let named = top
            .into_iter()
            .map(|(rip, n)| {
                let bytes = self
                    .rejected_bytes
                    .get(&rip)
                    .cloned()
                    .unwrap_or_else(|| format!("entry {rip:#x}"));
                (bytes, n)
            })
            .collect();
        (self.declined_rejected, named)
    }

    pub fn note_offered(&mut self) {
        self.offered += 1;
    }

    pub fn note_declined_event(&mut self) {
        self.declined_event += 1;
    }

    pub fn note_declined_uncompiled(&mut self) {
        self.declined_uncompiled += 1;
    }

    /// Bail-causing mnemonics, most frequent first.
    pub fn bail_mnemonics(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self
            .bail_mnemonics
            .iter()
            .map(|(k, n)| (k.clone(), *n))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    pub fn flag_writes_dropped(&self) -> u64 {
        self.flag_writes_dropped
    }

    pub fn compiled_blocks(&self) -> u64 {
        self.compiled_blocks
    }

    /// Chain steps the dispatcher took inside wasm, without a crossing per block.
    #[must_use]
    pub fn wasm_chain_steps(&self) -> u64 {
        CompiledBackend::wasm_chain_steps(&*self.backend.borrow())
    }

    /// Nanoseconds per guest instruction executed as compiled code, excluding the
    /// cost of getting it compiled.
    pub fn ns_per_executed_insn(&self) -> f64 {
        if self.executed_guest_instructions == 0 {
            return 0.0;
        }
        self.run_nanos as f64 / self.executed_guest_instructions as f64
    }

    /// Microseconds spent admitting one block, averaged.
    pub fn us_per_compile(&self) -> f64 {
        if self.compiled_blocks == 0 {
            return 0.0;
        }
        self.compile_nanos as f64 / self.compiled_blocks as f64 / 1000.0
    }

    pub fn executed_blocks(&self) -> u64 {
        self.executed_blocks
    }

    pub fn executed_guest_instructions(&self) -> u64 {
        self.executed_guest_instructions
    }

    /// Count a hit at `rip`, returning whether it is now worth compiling.
    pub fn record_hit(&mut self, rip: u64) -> bool {
        self.record_hit_with(rip, self.hot_threshold)
    }

    /// Count a hit at a RIP a chain ran off the end of the cache into.
    pub fn record_chain_miss(&mut self, rip: u64) -> bool {
        self.record_hit_with(rip, self.chain_miss_threshold)
    }

    fn record_hit_with(&mut self, rip: u64, threshold: u32) -> bool {
        if self.rejected.contains(&rip) {
            self.declined_rejected += 1;
            if bail_histogram_enabled() {
                *self.rejected_hits.entry(rip).or_insert(0) += 1;
            }
            return false;
        }
        if self.blocks.contains_key(&rip)
            || self.pending_set.contains(&rip)
        {
            return false;
        }
        // Once the cache is full nothing more can be admitted, and the seam offers a
        // RIP on every taken branch — 1.7M admission attempts per 80M instructions
        // were being made and refused one call later. Stop counting hits instead.
        if self.blocks.len() >= self.max_blocks {
            self.rejected_cap += 1;
            return false;
        }
        let n = self.hotness.entry(rip).or_insert(0);
        *n += 1;
        *n >= threshold
    }

    /// Translate the block at `rip` to WASM and register it.
    /// Ask the interpreter to hand back as soon as it has stepped over the
    /// instruction the tier could not translate.
    ///
    /// Off by default and measured: 184.8 against 182.2. It works — the interpreter
    /// otherwise carries on to the next taken branch, about thirteen instructions —
    /// but there is no compiled block at the resume point either, so the seam
    /// declines and the interpreter carries on regardless. Making it pay would mean
    /// admitting a block there, which is the trade that has lost every time it has
    /// been measured. `AERO_CT_RESUME=1`.
    fn arm_resume(&mut self, rip: u64) {
        if !self.resume_enabled || !self.rejected.contains(&rip) {
            self.resume_at.set(0);
            return;
        }
        // Two, not one: the interpreter decrements at the top of each iteration,
        // *before* running that instruction. Two means "let the one you are about to
        // run happen, then hand back" — and the one it is about to run is exactly
        // the one Tier-1 refused.
        self.resume_at.set(2);
        self.resume_sites += 1;
    }

    /// Recompile the blocks that have proved hot with the better code generator.
    ///
    /// Called after a chain rather than inside one: the backend is borrowed for the
    /// chain's duration, and the block that crossed the threshold is running.
    pub fn drain_promotions(&mut self) {
        if self.promote_queue.is_empty() {
            return;
        }
        // How many go into one module. When the dispatcher can chain, this is the
        // whole point of promoting in batches: a step between two blocks of the same
        // module is a direct call, and a step out of it is a return to the host, a
        // lookup and a fresh crossing into the engine. Execution is extremely
        // concentrated — the busiest 187 blocks are 94.4% of chain steps — so a
        // module big enough to hold that set captures nearly every step, while a
        // module of four captures almost none.
        let group = self.promote_batch.max(1);
        // Wait for a full group before compiling, so the module is actually big.
        // Without a floor here the queue is drained a few at a time and `chunks`
        // never sees more than that many, which is how in-wasm chaining can be built,
        // be correct, and change nothing at all.
        if self.chain_in_wasm && self.promote_queue.len() < group {
            // Not forever, though: a queue that never fills would leave its blocks
            // on the baseline compiler for the rest of the run.
            self.promote_stall += 1;
            if self.promote_stall < self.promote_stall_chains {
                return;
            }
        }
        self.promote_stall = 0;
        // Hottest first, so the set that carries the run lands in **one** module.
        //
        // A block joins the queue at the instant its count reaches the threshold, so
        // the queue is in arrival order and every entry looked equally hot when it
        // joined. By the time a group has gathered they no longer are, and the
        // difference matters more than anything else here: a step between two blocks
        // of one module costs 136 host instructions, a step out of it 420.
        //
        // Arrival order is why the promotion threshold *appeared* to be unlowerable. At
        // 2000 the ~190 blocks that carry the run cross earliest and land together by
        // accident; any lower threshold mixed them with colder blocks that crossed at
        // the same time and split them across modules, so 500 and 100 measured 169.2
        // and 102.2 against 86.8, and the in-wasm rate fell 56.9% -> 48.0% -> 43.0%.
        //
        // This fixes that: ordering by what the blocks have actually executed keeps the
        // rate at 67.4% for a threshold of 500 instead of 48.0%, with 864 blocks
        // promoted rather than 270.
        //
        // **And lowering the threshold still does not pay** — 79.4/67.1, 79.8/80.0,
        // 81.8/81.7, 72.5/67.7 across four paired deep windows. So fragmentation was
        // never the binding constraint. Execution is concentrated enough that the
        // busiest ~190 blocks are 94% of chain steps, and the 594 extra promotions
        // reach almost nothing while adding emitted code. The hot set is already
        // promoted; there is nothing below it worth having.
        //
        // Kept anyway, because it is neutral at the default (67.9% either way) and it
        // replaces an accident with a rule — the shipping threshold worked because
        // arrival order happened to coincide with hotness order at 2000.
        //
        // The count comes from the chain cache, which is direct-mapped, so a block
        // evicted since it joined reads zero and sorts last. That is the right answer
        // anyway: a block that has not been through the cache lately is not the one
        // carrying the run.
        if self.chain_in_wasm && self.promote_queue.len() > 1 {
            let Self {
                promote_queue,
                chain_cache,
                ..
            } = self;
            promote_queue.sort_by_key(|rip| {
                let e = &chain_cache[ChainCacheEntry::slot(*rip)];
                std::cmp::Reverse(if e.rip == *rip { e.exec_count } else { 0 })
            });
        }
        // Bounded per chain so a burst of newly-hot blocks does not stall the guest
        // behind a run of millisecond compilations.
        let take = self.promote_queue.len().min(group.max(4));
        let batch: Vec<u64> = self.promote_queue.drain(..take).collect();
        let mut irs = Vec::with_capacity(batch.len());
        let mut targets = Vec::with_capacity(batch.len());
        for rip in batch {
            if self.promoted.contains(&rip) {
                continue;
            }
            self.promoted.insert(rip);
            self.promoted_order.push(rip);
            let Some(entry) = self.blocks.get(&rip).copied() else {
                continue;
            };
            let Some(ir) = self.block_ir.get(&rip) else {
                continue;
            };
            irs.push(ir.clone());
            targets.push(entry.table_index);
        }
        if irs.is_empty() {
            return;
        }
        // How many share one Cranelift module. Batching halves the per-block compile
        // cost — 1,482us to 730.4us — because most of Cranelift's time is per module,
        // and it still measures *worse*: 184.8 against 182.1 in steady state at 16 a
        // module, 183.5 at 4. A block waits for the rest of its batch before it gets
        // the better code, and that delay costs more than the compilation saves. One
        // a module is the default for that reason; the batching path is kept because
        // it is what in-WASM chaining between hot blocks would need.
        let started = timing_enabled().then(std::time::Instant::now);
        let mut promoted = 0usize;
        for (chunk_ir, chunk_idx) in irs.chunks(group).zip(targets.chunks(group)) {
            let refs: Vec<&aero_jit_x86::tier1::ir::IrBlock> = chunk_ir.iter().collect();
            let module_id = self.next_module_id;
            self.next_module_id += 1;
            let mut options = self.codegen_options();
            if self.chain_in_wasm {
                options.chain_module_id = Some(module_id);
            }
            let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&refs, options);
            self.wasm_bytes += wasm.len() as u64;
            promoted += self.promote_cached(chunk_idx, &wasm);
            if !self.chain_in_wasm {
                continue;
            }
            // Now that these share a module, tell the dispatcher where they are: a
            // step from one of them to another is a direct call rather than a
            // return to the host, a lookup, and a fresh crossing into the engine.
            let mut backend = self.backend.borrow_mut();
            for (i, block_ir) in chunk_ir.iter().enumerate() {
                let entry_rip = block_ir.entry_rip;
                // Index within the module, which is the position `promote_blocks`
                // assigned — it walks the same slice in the same order.
                CompiledBackend::chain_map_insert(
                    &mut *backend,
                    entry_rip,
                    module_id,
                    u32::try_from(i).unwrap_or(u32::MAX),
                );
            }
        }
        if let Some(started) = started {
            self.promote_nanos += started.elapsed().as_nanos();
        }
        self.promotions += promoted as u64;
        self.consolidate_promoted();
    }

    /// Recompile every promoted block into a single module, so they chain to each
    /// other inside wasm instead of returning to the host between blocks.
    ///
    /// **Off, and measured: 239.6 against 132.8 in steady state.** The idea was that
    /// blocks cross the promotion threshold a few at a time, so promoting them as
    /// they qualify scatters the hot set across many small modules, and two blocks
    /// chain only when they share one. Gathering them does work — but it moves the
    /// in-wasm share from 20.1% of block entries to 20.3%, and costs a recompile of
    /// the whole set each time (2041us per promoted block against 708us).
    ///
    /// The premise was wrong, and the number that shows it is
    /// `AERO_CT_PROMOTED_EDGES`: only **36.3% of chain steps land on a promoted block
    /// and 28.7% run promoted-to-promoted**, so 20.3% is already about seven tenths
    /// of everything this map can reach. The concentration figure that suggested
    /// otherwise — "the busiest 188 blocks cover 89%" — is read from the 4096-entry
    /// chain cache, so it describes the resident set at the end of the run and not
    /// the session. Execution here is much flatter than that made it look.
    ///
    /// Kept because it is what a workload with a genuinely small hot set would want.
    fn consolidate_promoted(&mut self) {
        if !self.chain_in_wasm || !self.consolidate {
            return;
        }
        let n = self.promoted_order.len();
        if n < CONSOLIDATE_MIN || n - self.consolidated_len < CONSOLIDATE_STEP {
            return;
        }
        let mut irs = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        let mut rips = Vec::with_capacity(n);
        for &rip in &self.promoted_order {
            // A promoted block keeps its IR from admission; re-deriving it by
            // decoding at the entry again is wrong, because CR3 may have changed and
            // the same linear address can belong to another process by now.
            let (Some(entry), Some(ir)) = (self.blocks.get(&rip), self.block_ir.get(&rip)) else {
                continue;
            };
            irs.push(ir.clone());
            targets.push(entry.table_index);
            rips.push(rip);
        }
        if irs.len() < CONSOLIDATE_MIN {
            return;
        }
        let module_id = self.next_module_id;
        self.next_module_id += 1;
        let mut options = self.codegen_options();
        options.chain_module_id = Some(module_id);
        let started = timing_enabled().then(std::time::Instant::now);
        let refs: Vec<&aero_jit_x86::tier1::ir::IrBlock> = irs.iter().collect();
        let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&refs, options);
        self.wasm_bytes += wasm.len() as u64;
        let done = self.consolidate_cached(&targets, &wasm);
        if let Some(started) = started {
            self.promote_nanos += started.elapsed().as_nanos();
        }
        if done == 0 {
            return;
        }
        let mut backend = self.backend.borrow_mut();
        for (i, &rip) in rips.iter().enumerate() {
            CompiledBackend::chain_map_insert(
                &mut *backend,
                rip,
                module_id,
                u32::try_from(i).unwrap_or(u32::MAX),
            );
        }
        drop(backend);
        self.consolidated_len = n;
        self.consolidations += 1;
    }

    /// Of the recorded chain steps: how much edge weight lands on a promoted block,
    /// and how much runs promoted-to-promoted.
    ///
    /// The second is the ceiling for chaining inside wasm with the map populated at
    /// promotion, before module membership is taken into account. Requires
    /// `AERO_CT_EDGES=1`.
    #[must_use]
    pub fn promoted_edge_share(&self) -> Option<(u64, u64, u64)> {
        if self.edges.is_empty() {
            return None;
        }
        let mut total = 0u64;
        let mut to_promoted = 0u64;
        let mut both = 0u64;
        for (&(from, to), &n) in &self.edges {
            total += n;
            if self.promoted.contains(&to) {
                to_promoted += n;
                if self.promoted.contains(&from) {
                    both += n;
                }
            }
        }
        Some((both, to_promoted, total))
    }

    /// How admitted blocks end, in the order jmp, jcc, call, ret, exit, limit.
    #[must_use]
    pub fn end_kinds(&self) -> [u64; 6] {
        self.end_kinds
    }

    /// Chain steps into a block ending at an unconditional jump (needs `AERO_CT_EDGES`).
    #[must_use]
    pub fn jmp_block_steps(&self) -> u64 {
        self.jmp_block_steps
    }

    /// Chain steps recorded (needs `AERO_CT_EDGES`).
    #[must_use]
    pub fn recorded_chain_steps(&self) -> u64 {
        self.chain_steps
    }

    /// Chains entered.
    #[must_use]
    pub fn chains(&self) -> u64 {
        self.chains
    }

    /// Chains that had too little of the slice left to chain inside wasm.
    #[must_use]
    pub fn chains_no_budget(&self) -> u64 {
        self.chains_no_budget
    }

    /// How many times the promoted set has been gathered into one module.
    #[must_use]
    pub fn consolidations(&self) -> u64 {
        self.consolidations
    }

    pub fn compile<B: CpuBus>(&mut self, rip: u64, bitness: u32, bus: &mut B) -> bool {
        // Before anything else: a module a worker finished is of no use until it is
        // instantiated, and this is the one path that runs whether or not there is a
        // block to enter a chain with. Polling only after a chain deadlocks at zero
        // blocks — nothing is registered, so no chain runs, so nothing is registered.
        // Advanced here as well as in `run`: neither site alone is enough. `run` is
        // only reached once there is a block to enter a chain at, and offers go quiet
        // once coverage is high — so the clock has to be driven by both.
        self.admission_clock += 1;
        self.poll_admissions();
        if self.blocks.len() >= self.max_blocks {
            self.rejected_cap += 1;
            return false;
        }

        // `Tier1Bus` reads code bytes with `&self` while `CpuBus` needs `&mut`, so
        // the adapter carries a pointer. It lives only for this call and the bus
        // outlives it.
        struct CodeReader<B: CpuBus> {
            ptr: *mut B,
        }
        impl<B: CpuBus> Tier1Bus for CodeReader<B> {
            fn read_u8(&self, addr: u64) -> u8 {
                // Safety: `ptr` comes from the `&mut B` this call borrows, and no
                // other reference to it is live inside `discover_block_mode`.
                unsafe { (*self.ptr).read_u8(addr) }.unwrap_or(0xff)
            }
            fn write_u8(&mut self, _addr: u64, _value: u8) {}
        }

        let reader = CodeReader { ptr: bus as *mut B };
        // A compiled block runs to its end without consulting the interrupt
        // controller, while Tier-0 checks between instructions. Shortening blocks
        // therefore brings interrupt delivery closer to where the interpreter puts
        // it, which is how a timing difference is told apart from a translation bug.
        let block = discover_block_mode(
            &reader,
            rip,
            BlockLimits {
                max_insts: self.max_block_insts,
                // Superblocks: keep decoding past a conditional branch so the guest
                // registers stay in wasm locals across it. A block boundary costs
                // ~50ns against ~7ns for a guest instruction, and blocks here average
                // 4.2 instructions because they stop at the first branch.
                follow_conditionals: self.superblocks,
                follow_calls: self.follow_calls,
                allow_rdtsc: self.compile_rdtsc,
                allow_assists: self.allow_assists,
                ..BlockLimits::default()
            },
            bitness,
        );
        // How blocks end, weighted by how often they are wanted. A block that ends at
        // an *unconditional* jump could keep going — the target is known and always
        // taken, so following it needs no side exit and no speculation — and every
        // block boundary costs ~283 host instructions. Whether that is worth building
        // depends entirely on how many blocks end that way.
        {
            let slot = match block.end_kind {
                aero_jit_x86::tier1::BlockEndKind::Jmp => 0,
                aero_jit_x86::tier1::BlockEndKind::Jcc => 1,
                aero_jit_x86::tier1::BlockEndKind::Call => 2,
                aero_jit_x86::tier1::BlockEndKind::Ret => 3,
                aero_jit_x86::tier1::BlockEndKind::ExitToInterpreter { .. } => 4,
                aero_jit_x86::tier1::BlockEndKind::Limit { .. } => 5,
            };
            self.end_kinds[slot] += 1;
        }
        // A block that bails to the interpreter retires an unknown prefix of its
        // instructions — and one that bails at its own entry retires none at all,
        // leaving RIP where it was. Admitting those spins the machine: it "executes"
        // a block forever without making progress. Only take blocks that end in a
        // real control transfer, so an admitted block always retires its full count.
        if block.insts.is_empty() {
            self.rejected_empty += 1;
            self.rejected.insert(rip);
            return false;
        }
        // Compiling only the prefix of a block that bails was tried and measured:
        // it added 67 blocks for 0.3 points of coverage and made the compiled path
        // slower (38.8ns to 40.9ns). The blocks that decline the seam are not these
        // — they are a long tail of cold code, and no admission policy makes cold
        // code worth a ~600us compile.
        if let aero_jit_x86::BlockEndKind::ExitToInterpreter { next_rip } = block.end_kind {
            self.rejected_bails += 1;
            self.rejected.insert(rip);
            // A block Tier-1 cannot finish is a wall every chain reaching it stops
            // at, so which instruction caused it is worth knowing: a handful of
            // missing opcodes in hot paths cost far more coverage than their number
            // suggests.
            if bail_histogram_enabled() {
                // Raw bytes rather than a mnemonic: naming it would mean depending
                // on the decoder from here, and the prefix identifies the opcode
                // well enough to look up.
                let mut bytes = [0u8; 6];
                for (i, b) in bytes.iter_mut().enumerate() {
                    *b = bus.read_u8(next_rip.wrapping_add(i as u64)).unwrap_or(0xff);
                }
                let key = bytes
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");

                self.rejected_bytes.insert(rip, key.clone());
                *self.bail_mnemonics.entry(key).or_insert(0) += 1;
            }
            return false;
        }
        let guest_insts = block.insts.len() as u64;
        self.finish_compile(rip, block, guest_insts, bus)
    }

    fn finish_compile<B: CpuBus>(
        &mut self,
        rip: u64,
        block: aero_jit_x86::tier1::BasicBlock,
        guest_insts: u64,
        bus: &mut B,
    ) -> bool {
        let ends_in_jmp = matches!(block.end_kind, aero_jit_x86::tier1::BlockEndKind::Jmp);
        // `AERO_CT_TRACE_ADMIT=1` prints each admitted block in admission order, so a
        // divergence bisected to the Nth admitted block can be turned into an actual
        // instruction sequence to look at.
        if std::env::var_os("AERO_CT_TRACE_ADMIT").is_some() {
            let bytes: Vec<String> = block
                .insts
                .iter()
                .map(|i| {
                    let b: Vec<String> = (0..i.len as u64)
                        .map(|k| {
                            format!("{:02x}", bus.read_u8(i.rip.wrapping_add(k)).unwrap_or(0xff))
                        })
                        .collect();
                    format!("{:#x}:{}", i.rip, b.join(""))
                })
                .collect();
            eprintln!(
                "AERO_CT_ADMIT #{} rip={rip:#x} insts={} end={:?} [{}]",
                self.compiled_blocks,
                block.insts.len(),
                block.end_kind,
                bytes.join(" ")
            );
        }

        let mut ir = translate_block(&block);
        let fallthrough_rip = match ir.terminator {
            aero_jit_x86::tier1::ir::IrTerminator::CondJump { fallthrough, .. } => {
                Some(fallthrough)
            }
            _ => None,
        };
        // Most arithmetic in real code sets six flags nobody reads. Dropping the
        // writes a later write shadows is free correctness-wise and is a large part
        // of what compiled code spends its time on.
        let dropped = aero_jit_x86::tier1::eliminate_dead_flags(&mut ir);
        self.flag_writes_dropped += u64::from(dropped);
        // Kept for a possible promotion later — exactly this IR, so the recompiled
        // block is the same block.
        if self.count_execs {
            self.block_ir.insert(rip, ir.clone());
        }
        // A read-modify-write cannot be replayed: `sub [mem], 1` applied twice
        // subtracts two. Matching the load's address value against the store's was
        // tried and is not enough — `sub dword ptr [rip+0xfac1a], 1` builds the
        // address twice and gets two value ids for it, so the check passed and the
        // verifier reported its own double-subtraction as a mismatch.
        //
        // So the rule is the blunt one: a block that both loads and stores is not
        // replayed. Blocks that only load — the large majority, and where the
        // arithmetic bugs live — still are. A helper call is opaque; assume the worst.
        use aero_jit_x86::tier1::ir::IrInst;
        let mut has_load = false;
        let mut has_store = false;
        let mut opaque = false;
        let mut has_assist = false;
        for inst in &ir.insts {
            match inst {
                IrInst::Load { .. } => has_load = true,
                IrInst::Store { .. } => has_store = true,
                // Neither a timestamp read nor a host-performed instruction can be
                // replayed: the second execution does not give the same answer.
                IrInst::ReadTsc { .. } | IrInst::CallHelper { .. } => opaque = true,
                IrInst::Assist { .. } => {
                    opaque = true;
                    has_assist = true;
                }
                _ => {}
            }
        }
        // `AERO_CT_VERIFY_ALL=1` replays load-and-store blocks too. Their writes get
        // applied twice, so a read-modify-write reports a mismatch that is the
        // verifier's own doing — but a superblock nearly always stores, and without
        // this the interesting ones go unchecked.
        if has_assist {
            self.blocks_with_assists += 1;
        }
        let verify_all = std::env::var_os("AERO_CT_VERIFY_ALL").is_some();
        let rmw = opaque || (has_load && has_store && !verify_all);

        // Refusing on `slow_mem` needs the answer per block, and the answer is only
        // available after compiling, so that mode compiles one at a time.
        if refuse_slow_mem() {
            let (wasm, needs) = Tier1WasmCodegen::new()
                .compile_blocks_reporting(&[&ir], self.codegen_options());
            if needs.slow_mem {
                self.rejected_slow_mem += 1;
                self.rejected.insert(rip);
                return false;
            }
            let started = timing_enabled().then(std::time::Instant::now);
            // A one-block module still carries the dispatcher, so it is admitted the
            // same way the batches are.
            let table_index = if self.codegen_options().dispatcher {
                CompiledBackend::add_dispatched_blocks(&mut *self.backend.borrow_mut(), &wasm, 1)[0]
            } else {
                CompiledBackend::add_block(&mut *self.backend.borrow_mut(), &wasm)
            };
            if let Some(started) = started {
                self.compile_nanos += started.elapsed().as_nanos();
            }
            self.chain_cache[ChainCacheEntry::slot(rip)] = ChainCacheEntry::EMPTY;
            // A RIP that used to be a promoted block must not still be reachable
            // through the chain map: the entry names the old block's index.
            if self.chain_in_wasm {
                CompiledBackend::chain_map_forget(&mut *self.backend.borrow_mut(), rip);
            }
            self.blocks.insert(
                rip,
                Compiled {
                    table_index,
                    module_id: self.next_module_id,
                    guest_insts,
                    ends_in_jmp,
                    fallthrough_rip,
                    replayable: !rmw,
                },
            );
            self.compiled_blocks += 1;
            self.next_module_id += 1;
            return true;
        }

        // Otherwise queue the block and compile a batch of them into one module.
        // Admission is dominated by per-module work rather than by how much code is
        // in the module — 888us for one block on its own against 290us each when
        // sixteen share a module — and with a cache big enough to hold this
        // workload's 32,014 blocks that difference is tens of seconds.
        self.pending.push(PendingBlock {
            rip,
            ir,
            guest_insts,
            ends_in_jmp,
            fallthrough_rip,
            replayable: !rmw,
        });
        self.pending_set.insert(rip);
        if self.pending.len() >= self.batch_size {
            self.flush_pending();
        }
        // Not runnable yet. The caller falls back to the interpreter for this entry
        // and finds the block compiled the next time through.
        false
    }

    /// Options the code generator is driven with. One place, so the batched and
    /// unbatched paths cannot drift apart.
    fn codegen_options(&self) -> Tier1WasmOptions {
        Tier1WasmOptions {
            inline_tlb: true,
            // The load fast path emitted once per module and called, rather than
            // inlined at every access: a load falls from 218 emitted bytes to 21,
            // and admission from 246us to 222us a block.
            //
            // **Off on native now, and it used to be on.** The helper is a wasm call
            // per memory access, and a memory access is the most expensive thing a
            // guest instruction can be: measured in `tests/compiled_code_cost.rs`,
            // where the guest cannot diverge, a load costs **90.0 host instructions
            // through the shared helper against 43.6 inlined** — against 2.0 for an
            // ALU instruction.
            //
            // The reason it was on is real but has expired. Inlining emits ten times
            // the bytes, so it costs more to compile, and end to end that used to
            // cancel the win: without a module cache it is 129.1 against 132.8 in
            // steady state and 101.1 against 105.4 later, but 359.8 against 352.0
            // over a whole session, where the extra compilation is being paid for.
            // With `AERO_CT_CACHE_DIR` the compile is paid once ever, and inlining
            // wins there too — 236.1 against 239.3.
            //
            // **The browser used to keep the helper, and that has expired too.** The
            // reasons given were: no module cache there, module compilation on the main
            // thread, and a tenth of the emitted bytes being worth more than steady
            // state. The second is not true — Aero runs its CPU in a worker, which is
            // exactly why synchronous `WebAssembly.Module` is allowed at all (see
            // `browser_backend::instantiate`) — and the third is outweighed once a module
            // lives long enough for the engine to optimise it, which in a real session it
            // does.
            //
            // Measured in Chromium, one binary, one program, chaining identical:
            // a load-heavy loop goes **137.2 -> 195.3 MIPS (+42%)**, and a mixed block of
            // four ALU ops, two loads and a store — 37.5% memory, near real x86 —
            // **209.9 -> 250.7 (+19.4%)**. An ALU-only loop is unchanged at ~446, as it
            // should be. Well outside this benchmark's ~6% run-to-run band.
            //
            // Not measured: cold-start compile cost. Inlining emits ~10x the bytes for a
            // load, and a page load pays that once before any of the above applies.
            //
            // `AERO_CT_SHARED_MEM=1` restores the helper on either target for comparison.
            shared_mem_helpers: std::env::var_os("AERO_CT_SHARED_MEM").is_some(),
            // One exported entry point per module instead of one per block. A block
            // that can be named from outside needs a host-callable trampoline
            // compiled for it — ~80us, which was most of what admitting a block
            // cost. Reached by a direct call from a `br_table`, it needs none.
            dispatcher: std::env::var_os("AERO_CT_NO_DISPATCH").is_none(),
            // Off, and worth saying why, because it was on for a while.
            //
            // An access that cannot be proved to stay inside one page falls back to
            // the imported slow helpers. That used to mark the block as needing slow
            // memory, and such blocks were refused — 12,724 of them against 8,192
            // compiled — so splitting the access to keep it inline was the only way
            // to get coverage. Now that a non-RAM slow access is handed to the
            // machine's bus, those blocks are admitted and the split buys nothing.
            //
            // It is not free: the split emits a second translate-and-access path for
            // every wide load and store, and that code has to be generated, validated
            // and compiled. Turning it off cut admission from 612us to 249us a block
            // and compiled code from 108ns to 78ns per guest instruction, at
            // identical coverage.
            inline_tlb_cross_page_fastpath: self.cross_page_fastpath,
            // When false, a non-RAM access goes to the slow helpers — which hand it
            // to the machine's own bus — instead of bailing out of the block.
            // Bailing leaves the faulting instruction half-done, and `execute_chain`
            // has no rollback, so the interpreter redoes an instruction whose earlier
            // effects are already committed. Short blocks hit that rarely enough to
            // pass; superblocks hit it constantly.
            inline_tlb_mmio_exit: !self.mmio_via_bus,
            // One inline fast path per access, everything else a call.
            single_fast_path: self.single_fast_path,
            ..Default::default()
        }
    }

    /// Compile whatever is queued now rather than waiting for a full batch.
    ///
    /// Admission is batched, so a caller that wants one specific block runnable —
    /// a test, or a browser harness with nothing else to fill the batch — has to be
    /// able to say so.
    pub fn flush_for_test(&mut self) {
        self.flush_pending();
        // Admission may be happening on a worker thread; a caller that asked for a
        // specific block wants it runnable when this returns, not eventually.
        while !self.in_flight.is_empty() {
            self.drain_in_flight_now();
        }
    }

    /// Compile every queued block into one module and register them all.
    fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let refs: Vec<&aero_jit_x86::tier1::ir::IrBlock> = pending.iter().map(|p| &p.ir).collect();
        // The id has to be known before the module is built: the dispatcher compares
        // it against the chain map so a stale RIP can never be dispatched into
        // another module's function index.
        let module_id = self.next_module_id;
        self.next_module_id += 1;
        let mut options = self.codegen_options();
        // Blocks are queued in the order the guest first reached them, so the ones
        // sharing a batch are very often each other's successors — and chaining here
        // does raise the in-wasm rate, from 20.1% of block entries to 26.5%.
        //
        // It is off anyway, and measured: 319.6 against 132.8 in steady state. A
        // block entered inside wasm never reaches the host's chain-step closure, and
        // that closure is what counts executions and queues a block for Cranelift.
        // Chaining to a block therefore *hides* it from promotion, and the blocks
        // reached this way are the hot ones — so the set that most needs the better
        // code generator is exactly the set that stops being promoted (237 blocks
        // promoted before, 215 after, and the losses concentrated at the top).
        // Cranelift over Winch is worth 77 host instructions per guest instruction
        // here, which is far more than chaining saves.
        //
        // Promoted blocks do not have this problem: they have already been promoted,
        // so there is nothing left to hide them from. That is why the chain map is
        // populated at promotion and not at admission.
        if self.chain_admission && pending.len() > 1 {
            options.chain_module_id = Some(module_id);
        }
        let (wasm, _needs) = Tier1WasmCodegen::new().compile_blocks_reporting(&refs, options);

        let started = timing_enabled().then(std::time::Instant::now);
        self.wasm_bytes += wasm.len() as u64;
        // Hand the module to a worker when there is one. The blocks stay in
        // `pending_set` until it comes back, so nothing re-queues them, and the guest
        // keeps running in the meantime — which is the whole point, because
        // `Module::new` is ~390,000 host instructions and the guest has been waiting
        // through every one of them.
        let wasm = if self.async_compile && self.codegen_options().dispatcher {
            let submitted = CompiledBackend::submit_dispatched_blocks(
                &mut *self.backend.borrow_mut(),
                wasm,
                pending.len(),
            );
            match submitted {
                Some(id) => {
                    if let Some(started) = started {
                        self.compile_nanos += started.elapsed().as_nanos();
                    }
                    // Installed at a fixed point in *guest* time, not whenever a
                    // thread finishes: see `take_dispatched_blocks`.
                    let due = self.admission_clock + self.async_admit_delay;
                    self.next_admission_due = self.next_admission_due.min(due);
                    self.in_flight
                        .insert(id, (module_id, options.chain_module_id, pending, due));
                    return;
                }
                None => {
                    // The workers are gone. Compile here from now on, and rebuild the
                    // bytes that were handed over.
                    self.async_compile = false;
                    let refs: Vec<&aero_jit_x86::tier1::ir::IrBlock> =
                        pending.iter().map(|p| &p.ir).collect();
                    Tier1WasmCodegen::new()
                        .compile_blocks_reporting(&refs, options)
                        .0
                }
            }
        } else {
            wasm
        };
        let indices = if self.codegen_options().dispatcher {
            self.add_dispatched_cached(&wasm, pending.len())
        } else {
            CompiledBackend::add_blocks(&mut *self.backend.borrow_mut(), &wasm, pending.len())
        };
        if let Some(started) = started {
            self.compile_nanos += started.elapsed().as_nanos();
        }

        self.register_admitted(module_id, options.chain_module_id, pending, indices);
    }

    /// Admit a module, reusing a compiled copy from the cache when there is one.
    ///
    /// Keyed by the hash of the generated wasm, so a key can only match bytes that
    /// would have produced exactly this module. A miss, a corrupt file or an
    /// incompatible engine all end the same way: compile it.
    fn add_dispatched_cached(&mut self, wasm: &[u8], count: usize) -> Vec<u32> {
        let Some(dir) = self.module_cache.clone() else {
            return CompiledBackend::add_dispatched_blocks(
                &mut *self.backend.borrow_mut(),
                wasm,
                count,
            );
        };
        let key = blake3::hash(wasm).to_hex();
        let path = dir.join(format!("{key}.cwasm"));
        if let Ok(bytes) = std::fs::read(&path) {
            // Safety: this file was written by `add_dispatched_blocks_serialized`
            // below, in a directory the operator named for this purpose. A failure to
            // load is handled, not trusted.
            let loaded =
                unsafe { CompiledBackend::add_dispatched_precompiled(
                    &mut *self.backend.borrow_mut(),
                    &bytes,
                    count,
                ) };
            if let Some(indices) = loaded {
                self.cache_hits += 1;
                return indices;
            }
        }
        self.cache_misses += 1;
        let (indices, serialized) = CompiledBackend::add_dispatched_blocks_serialized(
            &mut *self.backend.borrow_mut(),
            wasm,
            count,
        );
        if let Some(bytes) = serialized {
            // Write beside the target and rename, so a run that dies half way through
            // cannot leave a truncated module for the next one to load.
            let tmp = dir.join(format!("{key}.tmp{}", std::process::id()));
            if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        indices
    }

    /// Promote a batch, loading the Cranelift module from the cache when it is there.
    ///
    /// Keyed separately from the admission cache (`.opt.cwasm`, not `.cwasm`) because
    /// the two engines produce different code for the same wasm, and loading one into
    /// the other is exactly the kind of thing `Module::deserialize` will not catch.
    fn promote_cached(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        let Some(dir) = self.module_cache.clone() else {
            return CompiledBackend::promote_blocks(
                &mut *self.backend.borrow_mut(),
                table_indices,
                wasm,
            );
        };
        let key = blake3::hash(wasm).to_hex();
        let path = dir.join(format!("{key}.opt.cwasm"));
        if let Ok(bytes) = std::fs::read(&path) {
            // Safety: written by `promote_blocks_serialized` below, into a directory
            // the operator named for this purpose; a failure to load is handled.
            let loaded = unsafe {
                CompiledBackend::promote_blocks_precompiled(
                    &mut *self.backend.borrow_mut(),
                    table_indices,
                    &bytes,
                )
            };
            if let Some(n) = loaded {
                self.promote_cache_hits += 1;
                return n;
            }
        }
        self.promote_cache_misses += 1;
        let (n, serialized) = CompiledBackend::promote_blocks_serialized(
            &mut *self.backend.borrow_mut(),
            table_indices,
            wasm,
        );
        if let Some(bytes) = serialized {
            // Write beside the target and rename, as the admission cache does.
            let tmp = dir.join(format!("{key}.opt.tmp{}", std::process::id()));
            if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        n
    }

    /// Consolidate a batch, loading the Cranelift module from the cache when it is there.
    ///
    /// Shares the promotion cache's key space and its counters — consolidation is the
    /// same expensive compile, just over the whole promoted set at once, and it is the
    /// one that repeats as the set grows.
    fn consolidate_cached(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        let Some(dir) = self.module_cache.clone() else {
            return CompiledBackend::promote_blocks_replacing(
                &mut *self.backend.borrow_mut(),
                table_indices,
                wasm,
            );
        };
        let key = blake3::hash(wasm).to_hex();
        let path = dir.join(format!("{key}.opt.cwasm"));
        if let Ok(bytes) = std::fs::read(&path) {
            // Safety: as `promote_cached`.
            let loaded = unsafe {
                CompiledBackend::promote_blocks_replacing_precompiled(
                    &mut *self.backend.borrow_mut(),
                    table_indices,
                    &bytes,
                )
            };
            if let Some(n) = loaded {
                self.promote_cache_hits += 1;
                return n;
            }
        }
        self.promote_cache_misses += 1;
        let (n, serialized) = CompiledBackend::promote_blocks_replacing_serialized(
            &mut *self.backend.borrow_mut(),
            table_indices,
            wasm,
        );
        if let Some(bytes) = serialized {
            let tmp = dir.join(format!("{key}.opt.tmp{}", std::process::id()));
            if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        n
    }

    /// Promoted modules loaded from the cache, and promoted modules compiled.
    #[must_use]
    pub fn promote_cache_stats(&self) -> (u64, u64) {
        (self.promote_cache_hits, self.promote_cache_misses)
    }

    /// Modules loaded from the cache, and modules that had to be compiled.
    #[must_use]
    pub fn module_cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    /// Register blocks whose module is now instantiated.
    fn register_admitted(
        &mut self,
        module_id: u32,
        chain_module_id: Option<u32>,
        pending: Vec<PendingBlock>,
        indices: Vec<u32>,
    ) {
        for (i, (p, table_index)) in pending.into_iter().zip(indices).enumerate() {
            self.pending_set.remove(&p.rip);
            // A RIP admitted again would otherwise be reached through the old
            // block's table index.
            self.chain_cache[ChainCacheEntry::slot(p.rip)] = ChainCacheEntry::EMPTY;
            if self.chain_in_wasm {
                let mut backend = self.backend.borrow_mut();
                if chain_module_id.is_some() {
                    // Point at this block rather than merely forgetting the old one.
                    CompiledBackend::chain_map_insert(
                        &mut *backend,
                        p.rip,
                        module_id,
                        u32::try_from(i).unwrap_or(u32::MAX),
                    );
                } else {
                    CompiledBackend::chain_map_forget(&mut *backend, p.rip);
                }
            }
            self.blocks.insert(
                p.rip,
                Compiled {
                    table_index,
                    module_id,
                    guest_insts: p.guest_insts,
                    ends_in_jmp: p.ends_in_jmp,
                    fallthrough_rip: p.fallthrough_rip,
                    replayable: p.replayable,
                },
            );
            self.compiled_blocks += 1;
        }
        // Only this batch's RIPs leave the set — they were removed one by one above.
        // Clearing it wholesale was safe while admission was synchronous and there
        // was only ever one batch; with modules in flight it would let a RIP that is
        // already being compiled be queued a second time.
    }

    /// Register any modules the compile workers have finished.
    ///
    /// Called often and cheap when there is nothing ready: a non-blocking receive.
    fn poll_admissions(&mut self) {
        // This is on the offer path, which runs millions of times a session, so the
        // common case has to be a single comparison. Scanning the in-flight map on
        // every call — even just to find what is due — cost 54% of the whole run.
        if !self.async_compile || self.in_flight.is_empty() {
            return;
        }
        // Install whatever is finished, in submission order so the result does not
        // depend on how the map iterates.
        //
        // This is where the determinism goes. Installing a module the moment a thread
        // finishes it makes *when* a block becomes runnable depend on host
        // scheduling, and the guest's path follows that — three runs of the same 60M
        // workload differed by 2.3%, against five significant figures of agreement
        // when compilation is inline. A deterministic variant was built, installing
        // each module at a fixed point in guest time instead: it measured 269.6
        // against 132.7, because the deadline almost always arrives before the
        // compile has finished, so the guest waits anyway *and* the blocks land later
        // than they would have. That is why this is opt-in rather than the default.
        let mut ready = CompiledBackend::poll_dispatched_blocks(&mut *self.backend.borrow_mut());
        ready.sort_by_key(|(id, _)| *id);
        for (id, indices) in ready {
            let Some((module_id, chain_module_id, pending, _)) = self.in_flight.remove(&id) else {
                continue;
            };
            self.register_admitted(module_id, chain_module_id, pending, indices);
        }
        self.next_admission_due = 0;
    }

    /// Install a module without waiting for its deadline. For tests only.
    fn drain_in_flight_now(&mut self) {
        let mut ids: Vec<u64> = self.in_flight.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            if let Some((_, _, _, at)) = self.in_flight.get_mut(&id) {
                *at = 0;
            }
        }
        self.next_admission_due = 0;
        self.poll_admissions();
    }

    pub fn has_block(&self, rip: u64) -> bool {
        self.blocks.contains_key(&rip)
    }

    /// Run a chain of compiled blocks starting at the CPU's current RIP.
    ///
    /// Returns the guest instructions retired, having advanced RIP, the time source
    /// and retirement for them. Returns 0 if there is nothing compiled to run.
    /// Run one block compiled, then run the same instructions under Tier-0 from the
    /// same state, and report any architectural difference.
    ///
    /// This is how a compiled tier is kept honest. Bisecting a whole-machine
    /// divergence down to one block costs a dozen multi-minute runs and finds one
    /// bug; this finds every mis-translated block in a single run, and names it.
    ///
    /// Only store-free blocks are checked, because verification re-executes the
    /// block and a second set of stores would corrupt the guest. Tier-0's result is
    /// the one kept, so the machine stays on the interpreter's path while checking.
    fn verify_block<B: CpuBus>(
        &mut self,
        cpu: &mut aero_cpu_core::CpuCore,
        bus: &mut B,
        assist: &mut aero_cpu_core::assist::AssistContext,
        entry: Compiled,
        rip: u64,
    ) {
        let before = cpu.state.clone();

        // The compiled block runs first now, because how much of it runs is no longer
        // known in advance: a block with side exits retires a different amount
        // depending on which exit it takes, and Tier-0 has to be stepped that many
        // times to be comparing the same work. Its writes come first and Tier-0's
        // repeat them, which is why only `replayable` blocks are checked.
        let retired = {
            let mut backend = self.backend.borrow_mut();
            let blocks = &self.blocks;
            // Bound the *in-wasm* chain to a single block as well.
            //
            // `run_chain`'s `max_blocks` limits the host loop; the module's own
            // dispatcher chains inside wasm up to `chain_max_blocks` in the JIT
            // context, which this path never set — so it inherited whatever the last
            // real chain left there and ran hundreds of blocks. A verifier meant to
            // compare *one* block was reporting `retired=1302` for a nine-instruction
            // block and then replaying 1302 instructions under Tier-0, which is the
            // whole-machine comparison, and that legitimately diverges on interrupt
            // timing. Every mismatch `AERO_X86_FOLLOW_JMP=1` was blamed for came from
            // here.
            if self.chain_in_wasm {
                CompiledBackend::set_chain_limits(&mut *backend, u64::MAX, 1);
            }
            let outcome = CompiledBackend::run_chain(
                &mut *backend,
                entry.table_index,
                &mut cpu.state,
                &mut |rip, _retired| blocks.get(&rip).map(|c| c.table_index),
                1,
            );
            outcome.retired
        };
        let mut after_compiled = cpu.state.clone();
        after_compiled.commit_lazy_flags();

        cpu.state = before.clone();
        for _ in 0..retired {
            // `exec::step` *reports* an assist rather than performing one, so a block
            // holding an instruction Tier-0 reaches through its assist path — a fence,
            // for one — would leave RIP where it was and be reported as a mismatch
            // that is the verifier's own. This performs it.
            if aero_cpu_core::assist::assist_one_instruction(assist, &mut cpu.time, &mut cpu.state, bus)
                .is_none()
            {
                // Tier-0 faulted where the compiled block did not, or stopped. The
                // states are not comparable; leave the interpreter's outcome in place.
                return;
            }
        }
        let mut after_tier0 = cpu.state.clone();
        after_tier0.commit_lazy_flags();
        cpu.state = after_tier0.clone();

        // RF and TF are delivery/debug state Tier-1 does not model; the six status
        // flags and the architectural registers are what must agree.
        const STATUS: u64 = 0x8d5;
        let same = after_compiled.rip == after_tier0.rip
            && after_compiled.gpr == after_tier0.gpr
            && (after_compiled.rflags & STATUS) == (after_tier0.rflags & STATUS);
        if same {
            return;
        }

        self.verify_failures += 1;
        if !self.verify_reported.insert(rip) || self.verify_reported.len() > 40 {
            return;
        }
        let bytes: Vec<String> = (0..24)
            .map(|k| format!("{:02x}", bus.read_u8(rip.wrapping_add(k)).unwrap_or(0xff)))
            .collect();
        eprintln!(
            "AERO_CT_MISMATCH rip={rip:#x} insts={} retired={retired} bytes={}",
            entry.guest_insts,
            bytes.join("")
        );
        if after_compiled.rip != after_tier0.rip {
            eprintln!(
                "    rip:    tier1={:#x} tier0={:#x}",
                after_compiled.rip, after_tier0.rip
            );
        }
        for (i, name) in [
            "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
            "r12", "r13", "r14", "r15",
        ]
        .iter()
        .enumerate()
        {
            if after_compiled.gpr[i] != after_tier0.gpr[i] {
                eprintln!(
                    "    {name}: tier1={:#x} tier0={:#x} (in={:#x})",
                    after_compiled.gpr[i], after_tier0.gpr[i], before.gpr[i]
                );
            }
        }
        let d = (after_compiled.rflags ^ after_tier0.rflags) & STATUS;
        if d != 0 {
            eprintln!(
                "    flags:  tier1={:#x} tier0={:#x} differ={:#x}",
                after_compiled.rflags & STATUS,
                after_tier0.rflags & STATUS,
                d
            );
        }
    }

    /// Run compiled blocks, stepping over the instructions that stop them.
    ///
    /// A chain ends where the tier has no block, and for a block it *refused* that
    /// is an instruction Tier-1 cannot translate. Handing back to the interpreter
    /// there costs about thirteen instructions before the seam is offered again, and
    /// a point of coverage is worth ~18 host instructions per guest instruction — so
    /// those thirteen are most of what the compiled tier still spends.
    ///
    /// Instead the tier steps that one instruction itself, through Tier-0, and
    /// carries on. What makes this work where offering the seam after an assist did
    /// not is that the tier is still in control afterwards: it admits a block at
    /// wherever the guest landed, rather than hoping the interpreter reaches a branch
    /// target that already has one.
    pub fn run<B: CpuBus>(
        &mut self,
        cpu: &mut aero_cpu_core::CpuCore,
        bus: &mut B,
        assist: &mut aero_cpu_core::assist::AssistContext,
        budget: u64,
    ) -> u64 {
        // The clock deadlines are expressed in. It has to advance whether or not
        // anything is compiled *and* whether or not new blocks are being offered —
        // two earlier choices failed on exactly that. Guest instructions executed by
        // the tier stay at zero until something is installed, so nothing ever
        // becomes due; offers go quiet once coverage is high, which stalls the clock
        // precisely when modules are waiting for it. This runs once per batch of
        // guest instructions, always.
        self.admission_clock += 1;
        self.poll_admissions();
        // Hand the assist its borrows for the whole of this call. A block performs
        // an instruction the host owns by calling out through here, and it needs the
        // interpreter's context, the machine's clock, the bus and the CPU state that
        // holds everything a block does not — CR3, the MSRs, the segment bases.
        //
        // Safety: all four are borrows the batch gave this call and holds for its
        // duration. `cpu.state` is the one the backend also has, and the backend does
        // not read it while WASM is running; only the registers the assist writes
        // back through linear memory survive the chain's own sync.
        self.assist_hook.set(Some((
            assist as *mut aero_cpu_core::assist::AssistContext,
            &mut cpu.time as *mut aero_cpu_core::time::TimeSource,
            bus as *mut B as *mut (),
            &mut cpu.state as *mut aero_cpu_core::state::CpuState,
            assist_thunk::<B>
                as fn(_, _, _, _, &mut aero_jit_x86::compiled_backend::AssistRegs) -> Option<u64>,
        )));
        let out = self.run_stepping(cpu, bus, assist, budget);
        self.assist_hook.set(None);
        out
    }

    fn run_stepping<B: CpuBus>(
        &mut self,
        cpu: &mut aero_cpu_core::CpuCore,
        bus: &mut B,
        assist: &mut aero_cpu_core::assist::AssistContext,
        budget: u64,
    ) -> u64 {
        let mut total = 0u64;
        loop {
            let ran = self.run_chain_once(cpu, bus, assist, budget.saturating_sub(total));
            total = total.saturating_add(ran);
            if !self.step_over_enabled || total >= budget {
                return total;
            }
            // A chain ends where the tier has no block. If that is somewhere it
            // *refused*, step over the instruction and carry on — and keep stepping
            // if what follows is refused too. The instructions Tier-1 cannot
            // translate cluster: kernel code runs `cli`, `out`, `mov cr8` and locked
            // compare-exchanges in short sequences, so stepping one and giving up on
            // the next recovers almost nothing (it measured ~1 instruction per step).
            if !self.rejected.contains(&cpu.state.rip()) {
                return total;
            }
            // Bounded so a long stretch of untranslatable code cannot turn this into
            // an open-ended interpreter loop inside the tier; past the bound the
            // interpreter is the right place for it.
            let step_over_max = self.step_over_max;
            let mut guard = 0u32;
            loop {
                let rip = cpu.state.rip();
                if self.blocks.contains_key(&rip) {
                    break;
                }
                // Try to give the landing point a block before deciding to give up:
                // that is the whole return on having stepped at all.
                if !self.rejected.contains(&rip) {
                    let bitness = cpu.state.bitness();
                    if self.compile(rip, bitness, bus) {
                        break;
                    }
                }
                if guard >= step_over_max || total >= budget {
                    return total;
                }
                match self.step_one(cpu, bus, assist) {
                    Some(n) => {
                        total = total.saturating_add(n);
                        self.stepped_over = self.stepped_over.saturating_add(n);
                        guard += 1;
                    }
                    None => return total,
                }
            }
        }
    }

    /// Execute exactly one guest instruction under Tier-0, returning how many it
    /// retired. `None` means it did not retire one — an exception, a halt, anything
    /// the interpreter has to be left to handle.
    fn step_one<B: CpuBus>(
        &mut self,
        cpu: &mut aero_cpu_core::CpuCore,
        bus: &mut B,
        assist: &mut aero_cpu_core::assist::AssistContext,
    ) -> Option<u64> {
        use aero_cpu_core::interp::tier0::exec::{run_batch_cpu_core_with_assists, BatchExit};
        use aero_cpu_core::interp::tier0::Tier0Config;
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            return None;
        }
        let cfg = Tier0Config::from_cpuid(&assist.features);
        let before = cpu.state.rip();
        let result = run_batch_cpu_core_with_assists(&cfg, assist, cpu, bus, 1);
        // Anything other than plain progress goes back to the interpreter: an
        // exception has to be delivered where Tier-0 would deliver it, and a halt or
        // a reset is not this loop's business.
        if result.executed != 1 || cpu.state.rip() == before {
            return None;
        }
        if !matches!(result.exit, BatchExit::Completed | BatchExit::Branch) {
            return None;
        }
        Some(result.executed)
    }

    fn run_chain_once<B: CpuBus>(
        &mut self,
        cpu: &mut aero_cpu_core::CpuCore,
        bus: &mut B,
        assist: &mut aero_cpu_core::assist::AssistContext,
        budget: u64,
    ) -> u64 {
        let rip = cpu.state.rip();
        let Some(entry) = self.blocks.get(&rip).copied() else {
            return 0;
        };
        // A block that cannot fit in what the slice has left would retire past it.
        // Let the interpreter have those instructions instead.
        if entry.guest_insts > budget {
            return 0;
        }

        if self.verify && entry.replayable {
            // The bus has to have seen the current paging registers before either
            // engine runs, exactly as a normal chain entry would arrange.
            CpuBus::sync(bus, &cpu.state);
            self.bus_hook.set(Some((
                bus as *mut B as *mut (),
                verify_thunk::<B> as fn(*mut (), u64, bool) -> Option<(u64, bool)>,
            )));
            CompiledBackend::flush_tlb(&mut *self.backend.borrow_mut());
            self.last_tlb_generation = None;
            self.verify_block(cpu, bus, assist, entry, rip);
            self.executed_blocks += 1;
            self.executed_guest_instructions += entry.guest_insts;
            return entry.guest_insts;
        }

        // Reading the clock twice per chain entry is itself measurable — it showed
        // up as ~1.3% of the run in `[vdso]` — so the per-path cost accounting is
        // opt-in rather than always on.
        let started = timing_enabled().then(std::time::Instant::now);

        // The chain's loads and stores resolve against this bus, so it has to have
        // observed the current paging registers and CPL first.
        CpuBus::sync(bus, &cpu.state);

        // Drop cached translations only when the MMU says something invalidated
        // them, rather than on every chain. Flushing unconditionally is correct but
        // costs ~11% of the compiled path at real coverage — every chain re-fills
        // the inline TLB through host calls. The generation counter changes on
        // `INVLPG`, a CR3 write and any paging-mode change, which is exactly the
        // set of events that invalidates the MMU's own TLB.
        let generation = bus.jit_tlb_generation();
        if self.always_flush_tlb || self.last_tlb_generation != Some(generation) {
            CompiledBackend::flush_tlb(&mut *self.backend.borrow_mut());
            self.last_tlb_generation = Some(generation);
        }

        // Point the installed translator at this chain's bus. Erasing the type is
        // what lets the translator itself be `'static` and installed once.
        fn translate_thunk<B: CpuBus>(ctx: *mut (), vaddr: u64, write: bool) -> Option<(u64, bool)> {
            // Safety: `ctx` is the `&mut B` borrowed by `run`, and nothing else
            // touches that bus while WASM is running.
            let bus = unsafe { &mut *(ctx as *mut B) };
            bus.jit_translate(vaddr, write)
        }
        self.bus_hook.set(Some((
            bus as *mut B as *mut (),
            translate_thunk::<B> as fn(*mut (), u64, bool) -> Option<(u64, bool)>,
        )));
        self.access_hook.set(Some((
            bus as *mut B as *mut (),
            access_thunk::<B> as fn(*mut (), u64, u32, Option<u64>) -> Option<u64>,
        )));
        self.last_tsc.set(None);
        self.clock_hook
            .set(Some(&mut cpu.time as *mut aero_cpu_core::time::TimeSource));

        let mut backend = self.backend.borrow_mut();

        // Bound the chain by what the slice has left as well as by the chain cap,
        // or a single entry could run far past `max_insts`.
        let by_budget = budget / entry.guest_insts.max(1);
        let max_blocks = u32::try_from(by_budget.max(1))
            .unwrap_or(MAX_CHAIN_BLOCKS)
            .min(self.max_chain_blocks);

        // What a chaining dispatcher may run without returning here. It tests the
        // budget *after* a block, so it is given room for one fewer than the longest
        // block this tier admits: the machine's instruction accounting has to land
        // exactly on `max_insts` for the interpreter oracles to hold at a count.
        // Is this the chain that pays for promotion's visibility? Bounding the in-wasm
        // chain to one block sends every step back through the host closure below, which
        // is the only place a block's executions are counted. See `chain_sample_period`.
        let accounting_chain = self.chain_admission
            && self.chain_sample_period > 0
            && self.chains.is_multiple_of(u64::from(self.chain_sample_period));
        if self.chain_in_wasm {
            self.chains += 1;
            if budget <= self.max_block_insts as u64 {
                self.chains_no_budget += 1;
            }
            CompiledBackend::set_chain_limits(
                &mut *backend,
                budget.saturating_sub(self.max_block_insts as u64),
                if accounting_chain { 1 } else { max_blocks },
            );
        }
        // What one observed execution stands for. Outside a sampled chain, steps taken
        // inside wasm are invisible here, so counting the few that still surface would
        // over-weight exactly the blocks that fail to chain — the ones promotion least
        // needs help finding. Counting only the sample keeps the picture unbiased.
        let exec_weight: u32 = if !self.chain_admission || self.chain_sample_period == 0 {
            // No sampling: count every step the host still sees, which is the behaviour
            // that predates sampling and the control this is measured against.
            1
        } else if accounting_chain {
            self.chain_sample_period
        } else {
            0
        };

        // Count retirement exactly rather than estimating it. `execute_chain` calls
        // the lookup after each block that *completed*, so accumulating there gives
        // the real total: a block that bailed to the interpreter never reaches the
        // lookup and is therefore never counted. Estimating instead — the entry
        // block's size times the number of blocks — silently mis-reports how much
        // guest work happened, which shows up as a throughput number that is not
        // measuring what it claims.
        let blocks = &self.blocks;
        let edges = &mut self.edges;
        let chain_cache = &mut self.chain_cache;
        let record_edges = self.record_edges;
        let entry_module = entry.module_id;
        let count_execs = self.count_execs;
        let promote_threshold = self.promote_threshold;
        let promote_queue = &mut self.promote_queue;
        let mut last_entry = rip;
        let mut self_loops = 0u64;
        // A chain step whose successor shares a module could be taken inside WASM by
        // that module's dispatcher, which calls its own blocks directly. Count how
        // often that is true before building it.
        let mut steps = 0u64;
        let mut steps_same_module = 0u64;
        let mut jmp_block_steps = 0u64;
        let mut last_module = entry_module;
        let outcome = CompiledBackend::run_chain(
            &mut *backend,
            entry.table_index,
            &mut cpu.state,
            // `retired` is what the blocks reported running, not what their lengths
            // imply — the two differ once a block has more than one exit.
            // This runs once per chain step — 42M times over a 200M-instruction
            // session — and is now the busiest host code in the profile, so what is
            // in it matters. Folding the execution counter into the block entry, so
            // one lookup serves the whole step instead of two, was worth 18.5% on
            // its own. The reporting counters below are behind `record_edges` for
            // the same reason: a branch and two increments per step is not free at
            // this rate.
            &mut |rip, retired| {
                if retired >= budget {
                    return None;
                }
                let slot = ChainCacheEntry::slot(rip);
                // Safety: `slot` is masked to the cache length.
                let e = unsafe { chain_cache.get_unchecked_mut(slot) };
                if e.rip != rip {
                    // Miss: consult the map and take the slot. A block is never
                    // removed once admitted, so an entry can only be wrong by being
                    // for a different RIP, which the compare above catches.
                    let c = blocks.get(&rip)?;
                    *e = ChainCacheEntry {
                        rip,
                        table_index: c.table_index,
                        guest_insts: c.guest_insts as u32,
                        exec_count: 0,
                    };
                }
                // Only take the next block if it can fit in what is left. Its length
                // is an upper bound on what it will retire, so this is conservative
                // in the direction that matters: the chain never runs past the slice
                // the machine allotted it, and the machine steps devices between
                // slices.
                if retired + u64::from(e.guest_insts) > budget {
                    return None;
                }
                if count_execs && exec_weight > 0 {
                    // Crossing the threshold, not landing on it: a weighted increment
                    // steps over any single value, so an equality test would queue
                    // nothing at all once sampling is on.
                    let before = u64::from(e.exec_count);
                    e.exec_count = e.exec_count.saturating_add(exec_weight);
                    if before < promote_threshold && u64::from(e.exec_count) >= promote_threshold
                    {
                        promote_queue.push(rip);
                    }
                }
                if record_edges {
                    *edges.entry((last_entry, rip)).or_insert(0) += 1;
                    if rip == last_entry {
                        self_loops += 1;
                    }
                    steps += 1;
                    if let Some(c) = blocks.get(&rip) {
                        if c.ends_in_jmp {
                            jmp_block_steps += 1;
                        }
                        if c.module_id == last_module {
                            steps_same_module += 1;
                        }
                        last_module = c.module_id;
                    }
                }
                last_entry = rip;
                Some(e.table_index)
            },
            max_blocks,
        );
        drop(backend);
        self.chain_steps += steps;
        self.chain_steps_same_module += steps_same_module;
        self.jmp_block_steps += jmp_block_steps;
        // The backend is free again, so anything that crossed the promotion
        // threshold during that chain can be recompiled now — and any module a
        // worker finished while the guest was running can be instantiated.
        self.poll_admissions();
        self.drain_promotions();
        self.arm_resume(cpu.state.rip());
        // The bus borrow ends with this call; leaving a stale pointer behind would
        // let a later translation dereference it.
        self.bus_hook.set(None);
        self.clock_hook.set(None);
        self.assist_hook.set(None);
        // Settle the architectural copy of the counter now the backend has given
        // `CpuState` back. Tier-0's assist leaves it behind too, and a later `rdmsr`
        // of the TSC reads it.
        if let Some(tsc) = self.last_tsc.take() {
            cpu.state.msr.tsc = tsc;
        }

        let (next_rip, ran, exited, retired) = (
            outcome.next_rip,
            outcome.blocks,
            outcome.exit_to_interpreter,
            outcome.retired,
        );
        if ran == 0 {
            return 0;
        }
        self.self_loop_steps += self_loops;
        // A block that bailed part-way used to be discarded whole: its retirement was
        // unknowable, so the safe thing was to pretend it had not run and let the
        // interpreter redo it. That is no longer true — the block reports the prefix
        // it got through — and discarding it is now wrong, because the register
        // writes of that prefix are already committed and only the count would be
        // thrown away. Longer blocks bail more often, which is why superblocks turned
        // this from a rare drift into an immediate one.
        if retired == 0 {
            return 0;
        }
        let guest_insts = retired;
        let completed = ran;

        cpu.state.set_rip(next_rip);
        cpu.time.advance_cycles(guest_insts);
        cpu.state.msr.tsc = cpu.time.read_tsc();
        cpu.pending.retire_instructions(guest_insts);

        self.executed_blocks = self.executed_blocks.saturating_add(u64::from(completed));
        self.executed_guest_instructions = self
            .executed_guest_instructions
            .saturating_add(guest_insts);
        if let Some(started) = started {
            self.run_nanos += started.elapsed().as_nanos();
        }

        // Grow along the chain. Where it stopped is the successor it was missing,
        // and having reached it by executing compiled blocks is far stronger
        // evidence than the batch-boundary sample, which lands on whatever RIP the
        // instruction budget happened to expire at.
        if !exited && !self.blocks.contains_key(&next_rip) {
            self.chain_misses += 1;
        }
        if !exited && !self.blocks.contains_key(&next_rip) && self.record_chain_miss(next_rip) {
            let bitness = cpu.state.bitness();
            let _ = self.compile(next_rip, bitness, bus);
        }

        guest_insts
    }
}

/// Whether Tier-0 offers its block boundaries to the compiled tier.
///
/// On by default; `AERO_CT_SEAM=0` falls back to entering only at outer-loop
/// boundaries, which is what the tier did before the seam existed.
pub fn seam_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AERO_CT_SEAM").as_deref() != Ok("0"))
}

/// `AERO_CT_TIMING=1`: account per-path cost, at the price of two clock reads per
/// chain entry.
#[cfg(not(target_arch = "wasm32"))]
fn timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AERO_CT_TIMING").is_some())
}

/// There is no `Instant` on `wasm32-unknown-unknown`, and this is a native
/// diagnostic rather than something the browser tier needs.
#[cfg(target_arch = "wasm32")]
fn timing_enabled() -> bool {
    false
}

/// Refuse blocks that can reach the slow memory helpers. Off by default.
///
/// Those helpers used to translate a guest address and then read or write the WASM
/// linear memory at the result, which is right for RAM and wrong for everything else:
/// a device register has no backing there, so the access read the wrong bytes and the
/// device never saw it. They now hand a non-RAM access to the embedder's bus instead,
/// at its natural width, which is the same bus Tier-0 uses.
///
/// Refusing them was the correct answer while that was broken, and it cost 12,724
/// blocks against 8,192 compiled. `AERO_CT_NO_SLOW_MEM=1` still refuses them, which
/// is how the two are compared.
fn refuse_slow_mem() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AERO_CT_NO_SLOW_MEM").is_some())
}

/// Build a Tier-0 seam that runs compiled chains from every taken branch.
///
/// Entering only at outer-loop boundaries caps coverage at what one arbitrary RIP
/// per batch can reach — about 1.5% here. The covered path is 3.6x cheaper than
/// interpreting, so reaching more of the workload is worth real time, which it was
/// not for the IR tier.
///
/// `probe` exists only to fix `B` for the callback: the seam carries the bus
/// type-erased, and this is where the concrete type is known.
/// Perform one guest access on `ctx`, which is the chain's `&mut B`.
fn access_thunk<B: CpuBus>(ctx: *mut (), addr: u64, size: u32, value: Option<u64>) -> Option<u64> {
    // Safety: `ctx` is the `&mut B` borrowed by the caller for this access, and
    // nothing else touches that bus while WASM is running.
    let bus = unsafe { &mut *(ctx as *mut B) };
    match (size, value) {
        (1, None) => bus.read_u8(addr).ok().map(u64::from),
        (2, None) => bus.read_u16(addr).ok().map(u64::from),
        (4, None) => bus.read_u32(addr).ok().map(u64::from),
        (8, None) => bus.read_u64(addr).ok(),
        (1, Some(v)) => bus.write_u8(addr, v as u8).ok().map(|()| 0),
        (2, Some(v)) => bus.write_u16(addr, v as u16).ok().map(|()| 0),
        (4, Some(v)) => bus.write_u32(addr, v as u32).ok().map(|()| 0),
        (8, Some(v)) => bus.write_u64(addr, v).ok().map(|()| 0),
        _ => None,
    }
}

/// Translator thunk for the verifier, which runs outside a chain.
fn verify_thunk<B: CpuBus>(ctx: *mut (), vaddr: u64, write: bool) -> Option<(u64, bool)> {
    // Safety: `ctx` is the `&mut B` the caller borrowed for this call.
    let bus = unsafe { &mut *(ctx as *mut B) };
    bus.jit_translate(vaddr, write)
}

pub fn seam_for<B: CpuBus, E: CompiledBackend>(
    probe: &B,
    tier: &mut CompiledTier<E>,
) -> Tier0BranchSeam {
    let _ = probe;
    fn run<B: CpuBus, E: CompiledBackend>(
        ctx: *mut (),
        cpu: &mut aero_cpu_core::CpuCore,
        bus: *mut (),
        assist: *mut aero_cpu_core::assist::AssistContext,
        budget: u64,
    ) -> u64 {
        // Safety: `ctx` and `bus` were built from the live `&mut CompiledTier` and
        // the batch's own `&mut B`; the batch hands its borrow over for exactly the
        // duration of this call and touches neither meanwhile.
        let tier = unsafe { &mut *(ctx as *mut CompiledTier<E>) };
        let bus = unsafe { &mut *(bus as *mut B) };

        // A chain runs many blocks without consulting the interrupt controller, so
        // it must not start while delivery is owed; Tier-0's loop head is where
        // that happens.
        tier.note_offered();
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            tier.note_declined_event();
            return 0;
        }
        // The seam is offered at every Tier-0 loop head, so a RIP that reaches here
        // is a branch target the guest is executing *now* — a far better admission
        // signal than the one-per-batch sample the machine loop takes, which is why
        // coverage sat at 59% while the seam declined ~4M offers for want of a
        // compiled block. Count the hit here and compile when it crosses.
        if tier.resume_armed() {
            tier.note_resume_hit();
        }
        let rip = cpu.state.rip();
        if !tier.has_block(rip) {
            tier.note_declined_uncompiled();
            if !(tier.record_chain_miss(rip) && tier.compile(rip, cpu.state.bitness(), bus)) {
                return 0;
            }
            // It is compiled now and the guest is at its entry, so run it rather
            // than paying another trip round the interpreter to come back here.
        }
        // Safety: the batch hands over its `&mut AssistContext` for the duration of
        // this call, exactly as it does the bus.
        let assist = unsafe { &mut *assist };
        tier.run(cpu, bus, assist, budget)
    }
    let resume = tier.resume_cell();
    Tier0BranchSeam::new(
        tier as *mut CompiledTier<E> as *mut (),
        run::<B, E>,
        resume,
    )
}

/// Perform one guest instruction against the state a block handed over.
///
/// Tier-0's assist path is the authority on what these instructions do, so it does
/// the work; this only supplies it with a `CpuState` and reports where the guest
/// ended up. `None` means it did not complete and the interpreter has to own it.
fn assist_thunk<B: CpuBus>(
    ctx: *mut aero_cpu_core::assist::AssistContext,
    time: *mut aero_cpu_core::time::TimeSource,
    bus: *mut (),
    state: *mut aero_cpu_core::state::CpuState,
    regs: &mut aero_jit_x86::compiled_backend::AssistRegs,
) -> Option<u64> {
    // Safety: all four are borrows the chain was handed and holds for its duration.
    // `cpu.state` is the one the backend also holds — but it does not read it while
    // WASM is running, and only the registers written back below survive the chain's
    // own sync, which is exactly the ones being overlaid here.
    let ctx = unsafe { &mut *ctx };
    let time = unsafe { &mut *time };
    let bus = unsafe { &mut *(bus as *mut B) };
    let state = unsafe { &mut *state };

    // The block has the registers; the machine has everything else — CR3, the MSRs,
    // the segment bases, the paging mode. Overlay one on the other rather than
    // handing the assist a `CpuState` that only has half of it.
    for i in 0..16 {
        state.gpr[i] = regs.gpr[i];
    }
    state.rip = regs.rip;
    state.set_rflags(regs.rflags);

    // `AERO_CT_ASSIST_HIST=1`: what is actually being handed over, weighted by how
    // often it runs. This is the list worth translating, and it is not the same as
    // the coverage wall, which names only the *first* instruction a block could not
    // take.
    if assist_histogram_enabled() {
        let rip = state.rip();
        let mut b = [0u8; 4];
        for (i, x) in b.iter_mut().enumerate() {
            *x = bus.read_u8(rip.wrapping_add(i as u64)).unwrap_or(0xff);
        }
        ASSIST_HIST.with(|h| *h.borrow_mut().entry(b).or_insert(0u64) += 1);
    }
    let before = bus.jit_tlb_generation();
    let next = aero_cpu_core::assist::assist_one_instruction(ctx, time, state, bus)?;
    regs.flush_tlb = bus.jit_tlb_generation() != before;

    for i in 0..16 {
        regs.gpr[i] = state.gpr[i];
    }
    regs.rip = state.rip;
    regs.rflags = state.rflags_snapshot();
    regs.fs_base = state.msr.fs_base;
    regs.gs_base = state.msr.gs_base;
    Some(next)
}

fn assist_histogram_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AERO_CT_ASSIST_HIST").is_some())
}

thread_local! {
    static ASSIST_HIST: RefCell<std::collections::HashMap<[u8; 4], u64>> =
        RefCell::new(std::collections::HashMap::new());
}

/// The instructions handed to the host, busiest first.
pub fn assist_histogram() -> Vec<(String, u64)> {
    ASSIST_HIST.with(|h| {
        let mut v: Vec<_> = h
            .borrow()
            .iter()
            .map(|(b, n)| {
                (
                    b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" "),
                    *n,
                )
            })
            .collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        v.truncate(24);
        v
    })
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

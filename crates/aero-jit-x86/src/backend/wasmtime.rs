use std::cell::RefCell;
use std::rc::Rc;
use std::marker::PhantomData;

use aero_cpu_core::jit::runtime::{
    CodeVersionTable, JitBackend, JitBlockExit, DEFAULT_CODE_VERSION_MAX_PAGES,
};
use aero_cpu_core::state::CpuState as CoreCpuState;
use wasmtime::{Caller, Config, Engine, Linker, Memory, Module, Store, TypedFunc};

use super::Tier1Cpu;
use crate::compiled_backend::{GuestAccessor, GuestTranslator};
use crate::abi;
use crate::jit_ctx::{self, JitContext};
use crate::tier1::{EXPORT_TIER1_BLOCK_FN, EXPORT_TIER1_DISPATCH_FN};

/// How to reach one compiled block.
///
/// A block exported under its own name is called directly. A block behind its
/// module's dispatcher is called by index — which is what keeps `Module::new` from
/// compiling a trampoline per block. See [`EXPORT_TIER1_DISPATCH_FN`].
#[derive(Clone)]
enum BlockRef {
    Exported(TypedFunc<(i32, i32), i64>),
    Dispatched {
        dispatch: TypedFunc<(i32, i32, i32), i64>,
        index: i32,
    },
    /// Recompiled by Cranelift, and therefore living in the other store.
    Optimized {
        dispatch: TypedFunc<(i32, i32, i32), i64>,
        index: i32,
    },
}

impl BlockRef {
    /// Which store this block runs in. The chain loop needs to know because the exit
    /// flags a block raises belong to its own store.
    fn is_optimized(&self) -> bool {
        matches!(self, BlockRef::Optimized { .. })
    }

}

impl BlockRef {
    /// Takes `&self`: cloning a `TypedFunc` per block shows up as
    /// `RegisteredType::clone` and `malloc` in a profile, on a path taken once for
    /// every four guest instructions.
    fn call(&self, store: impl wasmtime::AsContextMut, cpu_ptr: i32, jit_ctx_ptr: i32) -> i64 {
        match self {
            BlockRef::Exported(f) => f.call(store, (cpu_ptr, jit_ctx_ptr)),
            BlockRef::Dispatched { dispatch, index }
            | BlockRef::Optimized { dispatch, index } => {
                dispatch.call(store, (cpu_ptr, jit_ctx_ptr, *index))
            }
        }
        .expect("wasm tier1 block trapped")
    }
}
use crate::tier1_pipeline::Tier1WasmRegistry;
use crate::wasm::{
    IMPORT_JIT_EXIT, IMPORT_JIT_EXIT_MMIO, IMPORT_MEMORY, IMPORT_MEM_READ_U16, IMPORT_MEM_READ_U32,
    IMPORT_MEM_READ_U64, IMPORT_MEM_READ_U8, IMPORT_MEM_WRITE_U16, IMPORT_MEM_WRITE_U32,
    IMPORT_MEM_WRITE_U64, IMPORT_MEM_WRITE_U8, IMPORT_MMU_TRANSLATE, IMPORT_MODULE,
    IMPORT_ASSIST, IMPORT_PAGE_FAULT, IMPORT_RDTSC, JIT_EXIT_SENTINEL_I64,
};
use crate::Tier1Bus;

/// Optional callback invoked after a Tier-1 block *commits* guest memory writes performed via the
/// imported slow-path helpers (`env.mem_write_*`).
///
/// This can be used by embedders/tests to forward guest writes into a host-side tracker such as
/// [`aero_cpu_core::jit::runtime::JitRuntime::on_guest_write`] (self-modifying code coherence).
///
/// Writes from blocks that exit via MMIO/runtime bailout and are rolled back by the backend are
/// not reported.
///
/// Note: this callback is only invoked for the imported helper path. It does **not** cover the
/// inline-TLB direct-store fast path.
pub type WriteObserver = Box<dyn FnMut(u64, usize)>;


#[derive(Debug, Default, Clone, Copy)]
struct HostExitState {
    mmio_exit: bool,
    jit_exit: bool,
    page_fault: bool,
}

#[derive(Debug, Clone, Copy)]
struct WasmtimeCodeVersionTable {
    ptr: *mut u32,
    len: usize,
}

// Safety: `WasmtimeCodeVersionTable` is a plain pointer+length pair. It does not provide any
// implicit sharing; it is only safe to *move* between threads if the embedder ensures the backing
// linear memory remains valid and is not concurrently accessed.
unsafe impl Send for WasmtimeCodeVersionTable {}

impl CodeVersionTable for WasmtimeCodeVersionTable {
    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, idx: usize) -> u32 {
        if idx >= self.len {
            return 0;
        }
        // Safety: `idx < len` and the backend guarantees `ptr` points to `len` `u32` entries.
        unsafe { self.ptr.add(idx).read_unaligned() }
    }

    fn set(&self, idx: usize, value: u32) {
        if idx >= self.len {
            return;
        }
        // Safety: `idx < len` and the backend guarantees `ptr` points to `len` `u32` entries.
        unsafe {
            self.ptr.add(idx).write_unaligned(value);
        }
    }

    fn bump_range(&self, start: usize, end: usize) {
        if self.len == 0 || start > end {
            return;
        }
        let end = end.min(self.len - 1);
        for idx in start..=end {
            // Safety: `idx < len` and the backend guarantees `ptr` points to `len` `u32` entries.
            unsafe {
                let entry = self.ptr.add(idx);
                let cur = entry.read_unaligned();
                entry.write_unaligned(cur.wrapping_add(1));
            }
        }
    }

    fn reset(&self) {
        if self.len == 0 {
            return;
        }
        for idx in 0..self.len {
            // Safety: `idx < len` and the backend guarantees `ptr` points to `len` `u32` entries.
            unsafe {
                self.ptr.add(idx).write_unaligned(0);
            }
        }
    }

    fn table_ptr_len(&self) -> (*mut u32, usize) {
        if self.len == 0 {
            (core::ptr::null_mut(), 0)
        } else {
            (self.ptr, self.len)
        }
    }
}

impl HostExitState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn should_rollback(self) -> bool {
        self.mmio_exit || self.jit_exit || self.page_fault
    }
}

/// What the imported helpers reach for, shared between the stores.
///
/// The backend runs two of them — one per compiler — over the same guest memory, and
/// a chain crosses freely between them, so a helper called from either has to find
/// the same bus, the same clock and the same translation. Only the exit flags and
/// the write log stay per-store: those belong to the block that just ran, and the
/// chain loop knows which store that was.
#[derive(Default)]
struct HostHooks {
    write_observer: Option<WriteObserver>,
    /// Installed by embedders that page; absent means identity mapping.
    translator: Option<GuestTranslator>,
    /// Installed by embedders with a real bus; absent means slow-path accesses are
    /// resolved inside linear memory, which models RAM only.
    accessor: Option<GuestAccessor>,
    /// Installed by embedders with a real machine; absent means a block that reads
    /// the timestamp counter must not have been admitted.
    clock: Option<crate::compiled_backend::GuestClock>,
    /// Performs an instruction Tier-1 could not translate. See `IMPORT_ASSIST`.
    assist: Option<crate::compiled_backend::GuestAssist>,
    /// Linear-memory offset that guest physical address 0 maps to. Mirrors the
    /// backend field so the imported helpers, which are defined once at
    /// construction, can see later changes.
    ram_base: u64,
}

struct HostState {
    /// Raised instance/table ceilings; see the store's limiter.
    limits: wasmtime::StoreLimits,
    exit: HostExitState,
    pending_writes: Vec<(u64, usize)>,
    hooks: Rc<RefCell<HostHooks>>,
}

/// Ask Wasmtime to write `/tmp/perf-<pid>.map` so `perf` can name generated code.
///
/// Diagnostic only, and off unless `AERO_JIT_PERFMAP` is set. Without it every
/// compiled block is an anonymous executable mapping, so a profile can say how much
/// time is *not* in the emulator but cannot say which block it is in — and telling
/// those apart is the difference between "the JIT is the cost" and "this one block
/// is the cost".
fn maybe_enable_perf_map(config: &mut Config) {
    // JitDump carries the generated machine code as well as its symbols, so
    // `perf inject --jit` can hand `perf annotate` something to disassemble — which
    // is the only way to see what compiled code spends its instructions *on* rather
    // than merely how many. PerfMap gives names only.
    //
    // Setting a profiler that this build of Wasmtime does not carry makes
    // `Engine::new` fail, and a diagnostic switch must never be able to stop the
    // machine — so jitdump is requested only when the feature is compiled in.
    if cfg!(feature = "wasmtime-jitdump") && std::env::var_os("AERO_JIT_JITDUMP").is_some() {
        let _ = config.profiler(wasmtime::ProfilingStrategy::JitDump);
    } else if std::env::var_os("AERO_JIT_PERFMAP").is_some() {
        let _ = config.profiler(wasmtime::ProfilingStrategy::PerfMap);
    }
}

impl Default for HostState {
    fn default() -> Self {
        Self {
            limits: wasmtime::StoreLimitsBuilder::new()
                .instances(1 << 20)
                .tables(1 << 20)
                .memories(1 << 20)
                .build(),
            exit: HostExitState::default(),
            pending_writes: Vec::new(),
            hooks: Rc::new(RefCell::new(HostHooks::default())),
        }
    }
}

/// Instantiate a module whose only content is the guest memory, and return it.
///
/// See the call site: a memory has to be *defined by a module* for Wasmtime to route
/// its allocation through the host memory creator.
fn instantiate_anchor_memory(
    engine: &Engine,
    store: &mut Store<HostState>,
    memory_pages: u32,
) -> Memory {
    use wasm_encoder::{ExportKind, ExportSection, MemorySection, MemoryType as EncMemoryType};

    let mut module = wasm_encoder::Module::new();
    let mut mems = MemorySection::new();
    mems.memory(EncMemoryType {
        minimum: u64::from(memory_pages),
        maximum: Some(u64::from(memory_pages)),
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&mems);
    let mut exports = ExportSection::new();
    exports.export(IMPORT_MEMORY, ExportKind::Memory, 0);
    module.section(&exports);

    let module = Module::new(engine, &module.finish()).expect("compile anchor memory module");
    let instance = wasmtime::Instance::new(&mut *store, &module, &[])
        .expect("instantiate anchor memory module");
    instance
        .get_memory(&mut *store, IMPORT_MEMORY)
        .expect("anchor module exports its memory")
}

/// The Cranelift half of the backend.
/// Worker threads compiling modules, and the bookkeeping to match results to
/// submissions. See `start_compile_workers`.
#[cfg(not(target_arch = "wasm32"))]
struct CompileWorkers {
    jobs: std::sync::mpsc::Sender<(u64, Vec<u8>)>,
    done: std::sync::mpsc::Receiver<(u64, Module)>,
    /// How many blocks each outstanding submission holds.
    counts: std::collections::HashMap<u64, usize>,
    /// Modules a worker has finished that the caller has not asked for yet.
    completed: std::collections::HashMap<u64, Module>,
    next_id: u64,
    inflight: usize,
}

struct OptimizingEngine {
    #[allow(dead_code)]
    engine: Engine,
    store: Store<HostState>,
    linker: Linker<HostState>,
}

/// Read the architectural state a block wrote to linear memory.
///
/// Only the fields a block maintains: the assisted instruction sees the guest as the
/// block left it, and everything else about the machine lives on the host side.
fn read_cpu_state(
    mem: &Memory,
    store: &Caller<'_, HostState>,
    cpu_ptr: i32,
    regs: &mut crate::compiled_backend::AssistRegs,
) {
    let data = mem.data(store);
    let base = cpu_ptr as usize;
    let load = |off: u32| -> u64 {
        let at = base + off as usize;
        u64::from_le_bytes(data[at..at + 8].try_into().expect("8 bytes"))
    };
    for i in 0..16 {
        regs.gpr[i] = load(abi::CPU_GPR_OFF[i]);
    }
    regs.rip = load(abi::CPU_RIP_OFF);
    regs.rflags = load(abi::CPU_RFLAGS_OFF);
}

/// Write back what the assisted instruction left.
fn write_cpu_state(
    mem: &Memory,
    store: &mut Caller<'_, HostState>,
    cpu_ptr: i32,
    regs: &crate::compiled_backend::AssistRegs,
) {
    let data = mem.data_mut(store);
    let base = cpu_ptr as usize;
    let mut store_at = |off: u32, v: u64| {
        let at = base + off as usize;
        data[at..at + 8].copy_from_slice(&v.to_le_bytes());
    };
    for i in 0..16 {
        store_at(abi::CPU_GPR_OFF[i], regs.gpr[i]);
    }
    store_at(abi::CPU_RIP_OFF, regs.rip);
    store_at(abi::CPU_RFLAGS_OFF, regs.rflags);
    store_at(abi::CPU_FS_BASE_OFF, regs.fs_base);
    store_at(abi::CPU_GS_BASE_OFF, regs.gs_base);
    // Nothing is owed: the flags just written are materialised.
    store_at(abi::CPU_LAZY_FLAGS_OFF, 0);
}

/// Whether the Cranelift half of the backend is built at all.
///
/// It costs a second engine, store and set of imports, so it is only made when
/// something is going to promote blocks into it.
fn optimizing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AERO_JIT_NO_TIERUP").is_none())
}

/// How often the inline TLB missed and had to ask the host to translate.
///
/// Diagnostic: the miss path costs ~10 host instructions per guest instruction across
/// `jit_translate`, `Mmu::translate` and `TlbSet::lookup`, which is only worth
/// attacking if misses are common. One relaxed increment on a path that already calls
/// out of wasm is not measurable.
pub static JIT_TLB_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How often the whole inline TLB was invalidated by bumping the salt.
///
/// Diagnostic. A 16x larger TLB removes only 2% of misses, so they are not capacity
/// misses; this is the other candidate.
pub static JIT_TLB_FLUSHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Unmapped bytes after guest memory, so an out-of-range access from compiled code
/// faults instead of reaching whatever follows. Wasmtime elides bounds checks up to
/// this distance, so it has to match what the engine is told.
/// Unmapped bytes after guest memory, and the reason bounds checks disappear.
///
/// Wasmtime must bounds-check a linear-memory access unless it can prove the address
/// cannot escape the mapping. The block ABI passes `cpu_ptr`/`jit_ctx_ptr` as runtime
/// `i32`s, so every `CpuState` and JIT-context field access is a *dynamic* index — and
/// with a 32MiB guard an `i32` can obviously escape it, so each one compiled to
/// `lea` + `cmp` + `cmov`-to-null + the access.
///
/// Reserving the whole `i32` range as guard removes the proof obligation: no `i32`
/// index can leave the mapping, so no check is emitted. It costs address space and
/// nothing else — the region is `PROT_NONE` with `MAP_NORESERVE`.
///
/// `AERO_JIT_SMALL_GUARD=1` restores the 32MiB guard, which is what the A/B was run
/// against.
fn guard_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var_os("AERO_JIT_SMALL_GUARD").is_some() {
            32 << 20
        } else {
            (4usize << 30) + (32 << 20)
        }
    })
}

/// Reference `wasmtime`-powered backend that can execute Tier-1 compiled blocks.
///
/// ## Tier-1 ABI contract (`export block(cpu_ptr: i32, jit_ctx_ptr: i32) -> i64`)
///
/// The compiled block receives a pointer (`cpu_ptr`) into the shared linear memory where an
/// [`aero_cpu_core::state::CpuState`] is stored.
///
/// The block also receives a pointer (`jit_ctx_ptr`) to a JIT-only context region (`crate::jit_ctx`)
/// stored separately from the architectural CPU state. This allows the Tier-1 code generator to
/// use an optional inline direct-mapped TLB + direct-RAM fast-path without polluting
/// `aero_cpu_core::state::CpuState`.
///
/// The block mutates CPU state in-place and returns an `i64`:
///
/// - `ret != JIT_EXIT_SENTINEL_I64`: `ret` is the next RIP; execution can continue in the JIT.
/// - `ret == JIT_EXIT_SENTINEL_I64`: the block requests a one-shot exit to the interpreter.
///   The concrete `next_rip` is read from `CpuState.rip` after the block has updated it.
///
/// This mirrors the existing sentinel-based contract used by the older baseline WASM codegen.
pub struct WasmtimeBackend<Cpu> {
    engine: Engine,
    store: Store<HostState>,
    linker: Linker<HostState>,
    memory: Memory,
    /// A second engine, compiling with Cranelift instead of Winch, over the same
    /// guest memory.
    ///
    /// Winch admits a block for ~144us and Cranelift for something closer to a
    /// millisecond, but Cranelift's code is a quarter cheaper to run — 197.6 against
    /// 264.5 host instructions per guest instruction. Neither alone is the right
    /// answer: compiling everything with Cranelift costs more in warm-up than it
    /// ever returns, and compiling nothing with it leaves that quarter on the table.
    /// Execution is concentrated enough to have both — the busiest 1% of blocks
    /// account for 82% of all block executions — so the few hundred that matter are
    /// recompiled here and everything else stays with Winch.
    ///
    /// It has to be a second `Engine` because Wasmtime chooses its compiler per
    /// engine, and therefore a second `Store`; they share guest memory because both
    /// engines map the same pages. See `backend::shared_memory`.
    opt: Option<OptimizingEngine>,
    cpu_ptr: i32,
    jit_ctx_ptr: i32,
    /// Pointer/length of the dense `u32` code-version table in linear memory.
    ///
    /// `*_ptr` is a wasm32 byte offset; `*_host_ptr` is the corresponding host pointer into
    /// Wasmtime's mapped linear memory.
    ///
    /// Pointer stability: this backend configures a fixed-size memory (max = min) and uses
    /// `Config::memory_reservation` to ensure the linear memory base pointer does not move.
    code_version_table_ptr: u32,
    code_version_table_len: u32,
    code_version_table_host_ptr: *mut u32,
    blocks: Vec<BlockRef>,
    /// Base of the linear memory, cached.
    ///
    /// Entering a chain used to write `CpuState` field by field through
    /// `Memory::write` — twenty-one bounds-checked calls, each re-borrowing the store
    /// — and leaving one allocated a `Vec` the size of `CpuState` and copied it back.
    /// Measured at ~184ns a chain entry against ~7ns for a guest instruction, on
    /// blocks that retire four. The memory is created with `min == max` and a
    /// matching `Config::memory_reservation`, so its base never moves and the state
    /// can be copied directly.
    mem_base: *mut u8,
    /// Chain steps taken inside wasm, i.e. blocks entered without a crossing.
    wasm_chain_steps: u64,
    /// Threads compiling modules off the CPU thread; `None` disables it.
    #[cfg(not(target_arch = "wasm32"))]
    workers: Option<CompileWorkers>,
    /// Whether a guest-RAM view over this backend's linear memory has been handed
    /// out. Exactly one may exist: two would alias the same bytes as `&mut`.
    guest_ram_claimed: bool,
    /// Linear-memory offset that guest physical address 0 maps to.
    ///
    /// Compiled blocks form an address as `ram_base + phys`, so an embedder that
    /// keeps guest RAM above a state region at the bottom of linear memory sets this
    /// to where RAM starts.
    ram_base: u64,
    /// Salt that tags TLB entries. Bumping it invalidates every cached translation
    /// at once, which is what a CR3 write or a paging-mode change needs.
    tlb_salt: u64,
    _phantom: PhantomData<Cpu>,
}

impl<Cpu> Default for WasmtimeBackend<Cpu> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Cpu> WasmtimeBackend<Cpu> {
    /// Default location for the `CpuState` structure within linear memory.
    ///
    /// Page 0 is used as guest RAM; page 1 begins at 0x1_0000.
    pub const DEFAULT_CPU_PTR: i32 = 0x1_0000;

    /// Create a backend with a fixed 128KiB linear memory (two WASM pages).
    #[must_use]
    #[track_caller]
    pub fn new() -> Self {
        Self::new_with_memory_pages(2, Self::DEFAULT_CPU_PTR)
    }

    /// Create a backend with a configurable memory size and `cpu_ptr` base.
    #[must_use]
    #[track_caller]
    pub fn new_with_memory_pages(memory_pages: u32, cpu_ptr: i32) -> Self {
        // Explicitly enable WebAssembly SIMD so the same backend can execute future tier-1 blocks
        // that make use of SIMD ops.
        //
        // (Wasmtime's default feature set has historically changed over time; keeping this
        // explicit makes the backend's capabilities stable across upgrades.)
        let mut config = Config::new();
        config.wasm_simd(true);
        // Admitting a block costs far more than running one. `Module::new` is
        // ~1090us per block under Cranelift and ~577us under Winch — for a 663-byte
        // module — while instantiating it is 0.3us. See
        // `tests/compile_cost.rs`; the cost is per *module*, not per instruction
        // compiled, and every block is its own module today.
        //
        // Winch is Wasmtime's baseline compiler: worse code, far cheaper to
        // produce. For a tier whose cost is dominated by compilation rather than
        // execution that is the right side of the trade.
        if std::env::var("AERO_JIT_COMPILER").as_deref() != Ok("cranelift") {
            let _ = config.strategy(wasmtime::Strategy::Winch);
        }
        maybe_enable_perf_map(&mut config);
        config.memory_reservation(u64::from(memory_pages) * 65_536);
        // Explicit bounds checks are not an option here even if they were wanted:
        // Winch refuses to start without signals-based traps. Hence the real guard
        // region in `backend::shared_memory`.
        // Guest memory mapped here rather than by Wasmtime, so a second engine can
        // be given the same pages. Winch needs signals-based traps, so the mapping
        // carries a real unmapped guard region. See `backend::shared_memory`.
        let bytes = (memory_pages as usize) * 65_536;
        config.memory_guard_size(guard_bytes() as u64);
        let host_memory = crate::backend::shared_memory::SharedGuestMemory::new(bytes, guard_bytes());
        config.with_host_memory(std::sync::Arc::new(host_memory.clone()));
        let engine = Engine::new(&config).expect("create wasmtime engine");
        #[cfg(not(target_arch = "wasm32"))]
        let workers = Self::start_compile_workers(&engine);
        let mut store = Store::new(&engine, HostState::default());
        // Every admitted block is its own module and therefore its own instance, and
        // Wasmtime allows 10,000 of those per store by default — which is why the
        // block cache capped out at 8,192 and then refused 8.2M admissions over an
        // 80M-instruction run. Raise the ceiling so the cache size is a policy choice
        // rather than an engine limit. Batching blocks into shared modules is the
        // real fix and would cut admission cost about threefold as well.
        store.limiter(|state| &mut state.limits);
        let mut linker = Linker::new(&engine);

        // A single memory, imported by all generated blocks — but *created* by a
        // module rather than by `Memory::new`.
        //
        // That distinction is the whole reason a tier-up is possible.
        // `Memory::new` builds its own instance with a default on-demand allocator
        // whose memory creator is `None`, so it ignores `Config::with_host_memory`
        // and allocates its own pages. A memory a module defines goes through the
        // configured creator, which is what lets the second engine be handed the
        // same pages. Getting this wrong does not fail loudly: the Cranelift half
        // simply operates on a private copy of guest memory, and the machine resets
        // a million instructions later.
        let memory = instantiate_anchor_memory(&engine, &mut store, memory_pages);
        linker
            .define(&mut store, IMPORT_MODULE, IMPORT_MEMORY, memory)
            .expect("define env.memory import");

        define_mem_helpers(&mut linker, memory);
        define_stub_helpers(&mut linker, memory);

        // The Cranelift half, over the same pages. Off unless asked for: it is only
        // worth its compile time where a block runs often enough to repay it, and
        // that decision belongs to whoever is admitting blocks.
        let opt = optimizing_enabled().then(|| {
            let mut opt_config = Config::new();
            opt_config.wasm_simd(true);
            // Diagnostic: run the second store on Winch too, so a fault can be told
            // apart — the two-store plumbing, or the code Cranelift produces.
            if std::env::var_os("AERO_JIT_TIERUP_WINCH").is_some() {
                let _ = opt_config.strategy(wasmtime::Strategy::Winch);
            }
            maybe_enable_perf_map(&mut opt_config);
            // How hard Cranelift works on a promoted block. Promotion is bounded by
            // compile time rather than by how good the code is — a block costs
            // ~630us to promote and there are tens of thousands of them — so a level
            // that compiles faster buys *coverage*, which is worth more than the last
            // few percent of code quality on any one block.
            let _ = opt_config.cranelift_opt_level(
                match std::env::var("AERO_JIT_CRANELIFT_OPT").as_deref() {
                    Ok("none") => wasmtime::OptLevel::None,
                    Ok("size") => wasmtime::OptLevel::SpeedAndSize,
                    _ => wasmtime::OptLevel::Speed,
                },
            );
            opt_config.memory_reservation(u64::from(memory_pages) * 65_536);
            opt_config.memory_guard_size(guard_bytes() as u64);
            opt_config.with_host_memory(std::sync::Arc::new(host_memory.clone()));
            let engine = Engine::new(&opt_config).expect("create cranelift engine");
            let mut opt_store = Store::new(&engine, HostState::default());
            // One set of hooks, so a helper called from either half finds the same
            // bus and the same clock.
            opt_store.data_mut().hooks = store.data().hooks.clone();
            opt_store.limiter(|state| &mut state.limits);
            let mut opt_linker = Linker::new(&engine);
            let opt_memory = instantiate_anchor_memory(&engine, &mut opt_store, memory_pages);
            opt_linker
                .define(&mut opt_store, IMPORT_MODULE, IMPORT_MEMORY, opt_memory)
                .expect("define env.memory import");
            define_mem_helpers(&mut opt_linker, opt_memory);
            define_stub_helpers(&mut opt_linker, opt_memory);
            // The whole arrangement rests on the two memories being the same pages.
            // If Wasmtime ever stops consulting the host creator — a different
            // allocator, a different memory layout — this is where that shows up,
            // rather than as a guest that resets a million instructions later.

            // The whole arrangement rests on the two memories being the same pages.
            // If Wasmtime ever stops consulting the host creator — a different
            // allocator, a different memory layout — this is where that shows up,
            // rather than as a guest that resets a million instructions later.
            // `tests/cranelift_tierup.rs` covers the same property end to end.
            assert_eq!(
                memory.data_ptr(&store),
                opt_memory.data_ptr(&opt_store),
                "the two engines did not map the same guest memory"
            );
            OptimizingEngine {
                engine,
                store: opt_store,
                linker: opt_linker,
            }
        });

        // Verify the configured CpuState window fits within the linear memory.
        let byte_len = (memory_pages as usize)
            .checked_mul(65_536)
            .expect("memory_pages overflow");
        assert!(cpu_ptr >= 0, "cpu_ptr must be non-negative (got {cpu_ptr})");
        let jit_ctx_ptr = cpu_ptr
            .checked_add(abi::CPU_STATE_SIZE as i32)
            .expect("jit_ctx_ptr overflow");
        let cpu_end = (cpu_ptr as usize)
            .checked_add(abi::CPU_STATE_SIZE as usize)
            .expect("cpu_ptr overflow");
        let ctx_end = (jit_ctx_ptr as usize)
            .checked_add(JitContext::TOTAL_BYTE_SIZE)
            .expect("jit_ctx_ptr overflow");
        let end = cpu_end.max(ctx_end);
        assert!(
            end <= byte_len,
            "cpu_ptr (0x{cpu_ptr:x}) + cpu/jit_ctx regions (end=0x{end:x}) must fit in linear memory ({} bytes)",
            byte_len
        );
        let end = (cpu_ptr as usize)
            .checked_add((jit_ctx::TIER2_CTX_OFFSET + jit_ctx::JIT_CTX_SIZE) as usize)
            .expect("cpu_ptr overflow");
        assert!(
            end <= byte_len,
            "cpu_ptr (0x{cpu_ptr:x}) + Tier-2 context ({}) must fit in linear memory ({} bytes)",
            jit_ctx::TIER2_CTX_OFFSET + jit_ctx::JIT_CTX_SIZE,
            byte_len
        );

        // Allocate a dense `u32` code-version table in linear memory so Tier-1 inline-TLB stores
        // (and Tier-2 inline code-version guards) can bump/read page versions without importing a
        // host callback.
        //
        // Versions are treated as modulo-2^32 counters (`u32::MAX + 1 == 0`), matching the
        // wrapping semantics of WASM `i32.add` / `i32.atomic.rmw.add` and JS `Atomics.add`.
        //
        // The table is indexed by 4KiB physical page number: `paddr >> PAGE_SHIFT`.
        //
        // Layout:
        // - Tier-2 context: `cpu_ptr + jit_ctx::TIER2_CTX_OFFSET` (size `jit_ctx::TIER2_CTX_SIZE`)
        // - Version table: immediately after the Tier-2 context, aligned to 4 bytes.
        //
        // If the configured linear memory isn't large enough to hold the table, we disable it by
        // writing `LEN = 0`. Generated WASM checks for this and skips version bumps/reads.
        let (code_version_table_ptr, code_version_table_len) = {
            let guest_ram_bytes = u64::try_from(cpu_ptr).expect("cpu_ptr must be non-negative");
            let guest_ram_pages = guest_ram_bytes
                .checked_add(crate::PAGE_SIZE - 1)
                .expect("cpu_ptr overflow")
                / crate::PAGE_SIZE;

            // Keep the table within the runtime's default maximum tracked pages. (The reference
            // backend's `cpu_ptr` is an `i32`, so this is mostly defensive.)
            let requested_pages = guest_ram_pages.min(DEFAULT_CODE_VERSION_MAX_PAGES as u64);
            let requested_len: u32 = u32::try_from(requested_pages).unwrap_or(u32::MAX);

            let reserved_end = (cpu_ptr as usize)
                .checked_add((jit_ctx::TIER2_CTX_OFFSET + jit_ctx::TIER2_CTX_SIZE) as usize)
                .expect("cpu_ptr overflow");
            let table_base = (reserved_end + 3) & !3usize; // 4-byte alignment

            let table_bytes = (requested_len as usize).checked_mul(4);
            let fits =
                table_bytes.is_some_and(|bytes| table_base.saturating_add(bytes) <= byte_len);

            if requested_len == 0 || !fits {
                if requested_len != 0 {
                    // Worth saying out loud. Disabling the table is silent otherwise:
                    // generated code simply stops bumping and checking page versions,
                    // and the only symptom is stale compiled code surviving a write.
                    // It is also a moving target — the table sits *after* the JIT
                    // context, so enlarging the inline TLB or the chain map can push
                    // it over the edge and turn version tracking off as a side effect.
                    eprintln!(
                        "AERO_JIT_WARN: code-version table disabled — {} entries ({} bytes) \
                         will not fit after the JIT context at 0x{table_base:x} in {byte_len} \
                         bytes of linear memory",
                        requested_len,
                        (requested_len as usize) * 4,
                    );
                }
                (0u32, 0u32)
            } else {
                let ptr: u32 = table_base.try_into().expect("table base fits in wasm32");
                (ptr, requested_len)
            }
        };

        // Write `ptr`/`len` into the Tier-2 context ABI slots and zero-initialize the table.
        {
            let cpu_base = cpu_ptr as usize;
            let ptr_off = cpu_base + jit_ctx::CODE_VERSION_TABLE_PTR_OFFSET as usize;
            let len_off = cpu_base + jit_ctx::CODE_VERSION_TABLE_LEN_OFFSET as usize;

            let mem = memory.data_mut(&mut store);
            mem[ptr_off..ptr_off + 4].copy_from_slice(&code_version_table_ptr.to_le_bytes());
            mem[len_off..len_off + 4].copy_from_slice(&code_version_table_len.to_le_bytes());

            if code_version_table_len != 0 {
                let table_base = code_version_table_ptr as usize;
                let table_bytes = (code_version_table_len as usize)
                    .checked_mul(4)
                    .expect("code version table byte size overflow");
                mem[table_base..table_base + table_bytes].fill(0);
            }
        }

        let code_version_table_host_ptr = if code_version_table_len == 0 {
            core::ptr::null_mut()
        } else {
            // Safety: `code_version_table_ptr` is computed to be in-bounds and 4-byte aligned.
            unsafe {
                memory
                    .data_mut(&mut store)
                    .as_mut_ptr()
                    .add(code_version_table_ptr as usize) as *mut u32
            }
        };

        // Safety: the memory is created with `min == max` and a matching
        // `Config::memory_reservation`, so this base is stable for the backend's
        // lifetime. See `mem_base`.
        let mem_base = memory.data_mut(&mut store).as_mut_ptr();

        Self {
            engine,
            store,
            linker,
            memory,
            mem_base,
            wasm_chain_steps: 0,
            #[cfg(not(target_arch = "wasm32"))]
            workers,
            cpu_ptr,
            jit_ctx_ptr,
            code_version_table_ptr,
            code_version_table_len,
            code_version_table_host_ptr,
            blocks: Vec::new(),
            opt,
            guest_ram_claimed: false,
            ram_base: 0,
            tlb_salt: 0x1234_5678_9abc_def0,
            _phantom: PhantomData,
        }
    }

    /// Installs or clears the optional write observer invoked by the imported `env.mem_write_*`
    /// helpers.
    /// Host pointer to the base of the backend's linear memory, and its length.
    ///
    /// This is what lets guest RAM *be* the linear memory the compiled code
    /// addresses. With the inline TLB enabled a guest load becomes a bounds-checked
    /// load inside WASM instead of a call out to a host helper, which is the
    /// difference between a compiled tier that is bounded by Amdahl at about 2x and
    /// one that is not.
    ///
    /// # Safety
    ///
    /// The returned pointer aliases the linear memory, so the caller takes on the
    /// invariants Wasmtime would otherwise enforce:
    ///
    /// - It stays valid only while this backend is alive. The memory is created
    ///   with `min == max` and a matching `Config::memory_reservation`, so it is
    ///   never relocated by a `memory.grow`.
    /// - Writes through it are visible to compiled blocks immediately and vice
    ///   versa, so the caller must not hold a `&mut` into the region across a call
    ///   into WASM.
    pub unsafe fn linear_memory_ptr(&mut self) -> (*mut u8, usize) {
        let data = self.memory.data_mut(&mut self.store);
        (data.as_mut_ptr(), data.len())
    }

    /// Take the one permitted guest-RAM view of this backend's linear memory.
    ///
    /// Returns `None` if one has already been taken. Exclusivity is what makes a
    /// safe wrapper possible: the returned pointer aliases memory that compiled
    /// blocks also write, which is the entire point, but two host-side views of it
    /// would be two `&mut` to the same bytes.
    pub fn claim_guest_ram(&mut self) -> Option<(*mut u8, usize)> {
        if self.guest_ram_claimed {
            return None;
        }
        self.guest_ram_claimed = true;
        // Safety: exclusivity is established above; the memory is fixed-size, so
        // the pointer is stable for this backend's lifetime.
        Some(unsafe { self.linear_memory_ptr() })
    }

    /// Set where guest physical address 0 lands in linear memory.
    pub fn set_ram_base(&mut self, ram_base: u64) {
        self.ram_base = ram_base;
        self.store.data().hooks.borrow_mut().ram_base = ram_base;
    }

    /// Invalidate every cached inline-TLB translation.
    ///
    /// Entries are tagged `(vpn ^ salt) | 1`, so changing the salt makes all of them
    /// miss. This is what a CR3 write, an `INVLPG`, or a paging-mode change needs:
    /// missing one leaves compiled code using a stale physical page, which corrupts
    /// the guest silently.
    pub fn flush_tlb(&mut self) {
        JIT_TLB_FLUSHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.tlb_salt = self.tlb_salt.wrapping_mul(6364136223846793005).wrapping_add(1);
    }

    /// Install the translator used on inline-TLB misses.
    ///
    /// Returns the previous one, so a caller can install for the duration of a run
    /// and put back what was there.
    /// Install the bus accessor used by the slow-path memory helpers.
    pub fn set_accessor(&mut self, accessor: Option<GuestAccessor>) -> Option<GuestAccessor> {
        std::mem::replace(&mut self.store.data().hooks.borrow_mut().accessor, accessor)
    }

    pub fn set_clock(
        &mut self,
        clock: Option<crate::compiled_backend::GuestClock>,
    ) -> Option<crate::compiled_backend::GuestClock> {
        std::mem::replace(&mut self.store.data().hooks.borrow_mut().clock, clock)
    }

    pub fn set_assist(
        &mut self,
        assist: Option<crate::compiled_backend::GuestAssist>,
    ) -> Option<crate::compiled_backend::GuestAssist> {
        std::mem::replace(&mut self.store.data().hooks.borrow_mut().assist, assist)
    }

    #[must_use]
    pub fn has_assist(&self) -> bool {
        self.store.data().hooks.borrow().assist.is_some()
    }

    pub fn set_translator(&mut self, translator: Option<GuestTranslator>) -> Option<GuestTranslator> {
        std::mem::replace(&mut self.store.data().hooks.borrow_mut().translator, translator)
    }

    pub fn set_write_observer(&mut self, observer: Option<WriteObserver>) {
        self.store.data().hooks.borrow_mut().write_observer = observer;
    }

    /// Instantiate a Tier-1 block WASM module and append it to the internal table.
    ///
    /// Returns the table index used by `JitRuntime` and [`Self::execute`].
    #[track_caller]
    /// Add a module holding `count` block functions, returning their table indices.
    ///
    /// The code generator names the first export `EXPORT_TIER1_BLOCK_FN` and the rest
    /// with an index suffix, so a one-block module is byte-identical to what a single
    /// compile produced.
    pub fn add_compiled_blocks(&mut self, wasm_bytes: &[u8], count: usize) -> Vec<u32> {
        let module = Module::new(&self.engine, wasm_bytes).expect("compile wasm module");
        let instance = self
            .linker
            .instantiate(&mut self.store, &module)
            .expect("instantiate wasm module");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let name = if i == 0 {
                EXPORT_TIER1_BLOCK_FN.to_string()
            } else {
                format!("{EXPORT_TIER1_BLOCK_FN}{i}")
            };
            let func = instance
                .get_typed_func::<(i32, i32), i64>(&mut self.store, &name)
                .expect("get exported tier1 block function");
            out.push(self.blocks.len() as u32);
            self.blocks.push(BlockRef::Exported(func));
        }
        out
    }

    /// Whether this backend has a Cranelift half to promote blocks into.
    #[must_use]
    pub fn can_optimize(&self) -> bool {
        self.opt.is_some()
    }

    /// How many chain steps were taken inside wasm rather than through the host.
    #[must_use]
    pub fn wasm_chain_steps(&self) -> u64 {
        self.wasm_chain_steps
    }

    /// Base of the chain map inside the shared linear memory.
    ///
    /// Safety: the caller must not let a block run while it holds this, and must stay
    /// inside `JIT_CHAIN_MAP_ENTRIES` entries of 16 bytes. The region was
    /// bounds-checked against the memory at construction.
    unsafe fn chain_map_base(&self) -> *mut u8 {
        // Safety: as documented above.
        unsafe {
            self.mem_base
                .add(self.jit_ctx_ptr as usize + JitContext::CHAIN_MAP_OFFSET as usize)
        }
    }

    /// Tell the dispatcher that `rip` is block `index` of module `module_id`.
    ///
    /// Direct-mapped, so this evicts whatever shared the slot. That is not a
    /// correctness question — a miss just returns to the host, which holds the real
    /// map — but a *stale* hit would be, which is why [`Self::chain_map_forget`]
    /// exists and has to be called whenever a RIP stops meaning this block.
    pub fn chain_map_insert(&mut self, rip: u64, module_id: u32, index: u32) {
        let slot = JitContext::chain_map_slot(rip);
        let stride = jit_ctx::JIT_CHAIN_MAP_ENTRY_SIZE as usize;
        // Safety: `slot` is masked into range; no block is running.
        unsafe {
            let e = self.chain_map_base().add(slot * stride);
            // The RIP goes in last: a torn entry that is read by the dispatcher
            // between these writes must not match, and the RIP is what it matches on.
            std::ptr::write_unaligned(e.add(8) as *mut u32, module_id);
            std::ptr::write_unaligned(e.add(12) as *mut u32, index);
            std::ptr::write_unaligned(e as *mut u64, rip);
        }
    }

    /// Drop any chain-map entry for `rip`.
    ///
    /// Called when a RIP is re-admitted or evicted: the entry would otherwise send a
    /// chain into the function index the *old* block had.
    pub fn chain_map_forget(&mut self, rip: u64) {
        let slot = JitContext::chain_map_slot(rip);
        let stride = jit_ctx::JIT_CHAIN_MAP_ENTRY_SIZE as usize;
        // Safety: as above.
        unsafe {
            let e = self.chain_map_base().add(slot * stride);
            if std::ptr::read_unaligned(e as *const u64) == rip {
                // Zero the RIP only: `0` is not a legal block entry, so the entry can
                // never match again.
                std::ptr::write_unaligned(e as *mut u64, 0);
            }
        }
    }

    /// Drop every chain-map entry.
    pub fn chain_map_clear(&mut self) {
        // Safety: as above; the whole region belongs to the map.
        unsafe {
            std::ptr::write_bytes(self.chain_map_base(), 0, JitContext::CHAIN_MAP_BYTES);
        }
    }

    /// Bound what the dispatcher may do without returning to the host.
    ///
    /// `retired_budget` has to leave room for one more block than the caller can
    /// afford: the loop checks *after* running a block, so it can overshoot by the
    /// length of whatever it last ran. The caller subtracts the longest block it
    /// admits, which makes the overshoot impossible rather than unlikely — the
    /// machine's instruction accounting has to be exact for the interpreter oracles
    /// to hold at a given count.
    pub fn set_chain_limits(&mut self, retired_budget: u64, max_blocks: u32) {
        // Safety: header fields, bounds-checked at construction; nothing is running.
        unsafe {
            let ctx = self.mem_base.add(self.jit_ctx_ptr as usize);
            std::ptr::write_unaligned(
                ctx.add(JitContext::CHAIN_BUDGET_OFFSET as usize) as *mut u64,
                retired_budget,
            );
            std::ptr::write_unaligned(
                ctx.add(JitContext::CHAIN_MAX_BLOCKS_OFFSET as usize) as *mut u32,
                max_blocks,
            );
        }
    }

    /// Recompile one already-admitted block with Cranelift and run it from there.
    ///
    /// `table_index` keeps its identity: every chain that reaches this block now goes
    /// to the better code, and nothing that refers to it has to be found and updated.
    /// The Winch copy is left where it is — reclaiming it would mean unloading a
    /// module, which Wasmtime does per-store, and it costs only address space.
    #[track_caller]
    pub fn promote_blocks(&mut self, table_indices: &[u32], wasm_bytes: &[u8]) -> usize {
        let Some(opt) = self.opt.as_mut() else {
            return 0;
        };
        let Ok(module) = Module::new(&opt.engine, wasm_bytes) else {
            return 0;
        };
        let Ok(instance) = opt.linker.instantiate(&mut opt.store, &module) else {
            return 0;
        };
        let Ok(dispatch) = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut opt.store, EXPORT_TIER1_DISPATCH_FN)
        else {
            return 0;
        };
        let mut done = 0;
        for (i, &table_index) in table_indices.iter().enumerate() {
            let Some(slot) = self.blocks.get_mut(table_index as usize) else {
                continue;
            };
            if slot.is_optimized() {
                continue;
            }
            *slot = BlockRef::Optimized {
                dispatch: dispatch.clone(),
                index: i as i32,
            };
            done += 1;
        }
        done
    }

    /// As [`Self::promote_blocks`], but replaces blocks that are *already* promoted.
    ///
    /// Used to gather the whole promoted set into one module. Blocks cross the
    /// promotion threshold a few at a time, so promoting them as they qualify
    /// scatters them across many small modules — and two blocks chain inside wasm
    /// only when they share one. Recompiling the set together is what makes the hot
    /// path a sequence of direct calls instead of a return to the host per block.
    ///
    /// The modules being replaced are left loaded; reclaiming one means unloading it
    /// from the store, and it costs only address space.
    #[track_caller]
    /// As [`Self::promote_blocks`], also returning the compiled module's bytes.
    ///
    /// Promotion is the expensive half of the tier: Cranelift costs far more per block
    /// than Winch, and unlike admission it was never cached — so every run paid for
    /// every promotion again. That is why lowering the promotion threshold measured as
    /// a large loss even "with a warm cache": the cache did not cover the thing being
    /// increased.
    pub fn promote_blocks_serialized(
        &mut self,
        table_indices: &[u32],
        wasm_bytes: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        let Some(opt) = self.opt.as_ref() else {
            return (0, None);
        };
        let Ok(module) = Module::new(&opt.engine, wasm_bytes) else {
            return (0, None);
        };
        let serialized = module.serialize().ok();
        (self.promote_module(table_indices, &module), serialized)
    }

    /// Promote using a module compiled earlier by [`Self::promote_blocks_serialized`].
    ///
    /// Returns `None` if the bytes cannot be loaded, so the caller compiles instead.
    ///
    /// # Safety
    /// As [`Self::add_dispatched_precompiled`]: the bytes must come from a compatible
    /// engine, and Wasmtime does not fully validate them. Note this is the *optimizing*
    /// engine — a module serialized by the admission engine must never be loaded here,
    /// which is why the cache key carries which engine produced it.
    pub unsafe fn promote_blocks_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        let opt = self.opt.as_ref()?;
        // Safety: forwarded to the caller's obligation, documented above.
        let module = unsafe { Module::deserialize(&opt.engine, bytes) }.ok()?;
        Some(self.promote_module(table_indices, &module))
    }

    pub fn promote_blocks_replacing(&mut self, table_indices: &[u32], wasm_bytes: &[u8]) -> usize {
        let Some(opt) = self.opt.as_ref() else {
            return 0;
        };
        let Ok(module) = Module::new(&opt.engine, wasm_bytes) else {
            return 0;
        };
        self.promote_module_replacing(table_indices, &module)
    }

    /// As [`Self::promote_blocks_replacing`], also returning the module's bytes.
    pub fn promote_blocks_replacing_serialized(
        &mut self,
        table_indices: &[u32],
        wasm_bytes: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        let Some(opt) = self.opt.as_ref() else {
            return (0, None);
        };
        let Ok(module) = Module::new(&opt.engine, wasm_bytes) else {
            return (0, None);
        };
        let serialized = module.serialize().ok();
        (self.promote_module_replacing(table_indices, &module), serialized)
    }

    /// Consolidate from a module compiled earlier, or `None` to compile instead.
    ///
    /// # Safety
    /// As [`Self::promote_blocks_precompiled`].
    pub unsafe fn promote_blocks_replacing_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        let opt = self.opt.as_ref()?;
        // Safety: forwarded to the caller's obligation.
        let module = unsafe { Module::deserialize(&opt.engine, bytes) }.ok()?;
        Some(self.promote_module_replacing(table_indices, &module))
    }

    /// Point the given table slots at a module's blocks, replacing whatever is there.
    ///
    /// Unlike [`Self::promote_module`] this overwrites slots that are already
    /// optimized, which is what consolidation needs: the whole promoted set moves into
    /// one module so steps between its blocks stay inside wasm.
    fn promote_module_replacing(&mut self, table_indices: &[u32], module: &Module) -> usize {
        let Some(opt) = self.opt.as_mut() else {
            return 0;
        };
        let Ok(instance) = opt.linker.instantiate(&mut opt.store, module) else {
            return 0;
        };
        let Ok(dispatch) = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut opt.store, EXPORT_TIER1_DISPATCH_FN)
        else {
            return 0;
        };
        let mut done = 0;
        for (i, &table_index) in table_indices.iter().enumerate() {
            let Some(slot) = self.blocks.get_mut(table_index as usize) else {
                continue;
            };
            *slot = BlockRef::Optimized {
                dispatch: dispatch.clone(),
                index: i as i32,
            };
            done += 1;
        }
        done
    }

    #[track_caller]
    pub fn promote_block(&mut self, table_index: u32, wasm_bytes: &[u8]) -> bool {
        let Some(opt) = self.opt.as_mut() else {
            return false;
        };
        let Some(slot) = self.blocks.get_mut(table_index as usize) else {
            return false;
        };
        if slot.is_optimized() {
            return false;
        }
        let Ok(module) = Module::new(&opt.engine, wasm_bytes) else {
            return false;
        };
        let Ok(instance) = opt.linker.instantiate(&mut opt.store, &module) else {
            return false;
        };
        let Ok(dispatch) = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut opt.store, EXPORT_TIER1_DISPATCH_FN)
        else {
            return false;
        };
        *slot = BlockRef::Optimized {
            dispatch,
            index: 0,
        };
        true
    }

    /// Register a module's blocks and hand back its serialized form, so a caller can
    /// keep it and skip the compile next time.
    ///
    /// Compiling is ~390,000 host instructions a block; loading a serialized module
    /// is under 5,000. For a machine that boots the same guest image repeatedly that
    /// turns admission from a per-run cost into a first-run one.
    #[track_caller]
    pub fn add_dispatched_blocks_serialized(
        &mut self,
        wasm_bytes: &[u8],
        count: usize,
    ) -> (Vec<u32>, Option<Vec<u8>>) {
        let module = Module::new(&self.engine, wasm_bytes).expect("compile wasm module");
        let serialized = module.serialize().ok();
        (self.add_dispatched_module(&module, count), serialized)
    }

    /// Register the blocks of a module that was compiled earlier and serialized.
    ///
    /// Returns `None` if the bytes cannot be loaded — a different Wasmtime, a
    /// different target, a truncated file — and the caller compiles instead.
    ///
    /// # Safety
    /// `bytes` must be the output of [`Self::add_dispatched_blocks_serialized`] from a
    /// compatible engine. Wasmtime checks a version header but does not fully
    /// validate the body, so this must never be pointed at data another program could
    /// have written.
    #[track_caller]
    pub unsafe fn add_dispatched_precompiled(
        &mut self,
        bytes: &[u8],
        count: usize,
    ) -> Option<Vec<u32>> {
        // Safety: forwarded to the caller's obligation, documented above.
        let module = unsafe { Module::deserialize(&self.engine, bytes) }.ok()?;
        Some(self.add_dispatched_module(&module, count))
    }

    /// Compile a module, serialize it, and deserialize it again — reporting what the
    /// deserialize alone costs.
    ///
    /// If loading a precompiled module is much cheaper than compiling one, a
    /// persistent cache keyed by the generated wasm turns admission from a
    /// per-run cost into a first-run cost, which matters for a machine that boots the
    /// same guest image over and over.
    ///
    /// # Panics
    /// If the module cannot be compiled or serialized.
    #[track_caller]
    pub fn serialize_module(&mut self, wasm_bytes: &[u8]) -> Vec<u8> {
        Module::new(&self.engine, wasm_bytes)
            .expect("compile wasm module")
            .serialize()
            .expect("serialize wasm module")
    }

    /// Load a module produced by [`Self::serialize_module`].
    ///
    /// # Safety
    /// The bytes must have come from `serialize_module` on a compatible engine;
    /// Wasmtime does not fully validate them.
    #[track_caller]
    pub unsafe fn deserialize_module(&mut self, bytes: &[u8]) -> Option<Module> {
        // Safety: forwarded to the caller's obligation.
        unsafe { Module::deserialize(&self.engine, bytes) }.ok()
    }

    /// Compile a module and throw it away, without instantiating it.
    ///
    /// Only for attributing admission cost: `Module::new` is validation plus code
    /// generation, and instantiation is separate. Knowing which of the two dominates
    /// decides whether admission is something this project can do anything about.
    #[track_caller]
    pub fn compile_module_only(&mut self, wasm_bytes: &[u8]) {
        let _ = Module::new(&self.engine, wasm_bytes).expect("compile wasm module");
    }

    /// Compiles modules on worker threads so the CPU thread does not wait for them.
    ///
    /// Admitting a block costs ~417,000 host instructions and 94% of that is
    /// `Module::new`. Those instructions do not disappear by moving — the metric this
    /// project measures with, host instructions per guest instruction, is blind to the
    /// difference — but the goal is throughput, and on a machine with cores to spare
    /// the guest should not be stopped while a compiler runs.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_compile_workers(engine: &Engine) -> Option<CompileWorkers> {
        // Only when someone has asked for off-thread compilation. Every backend would
        // otherwise spawn threads it never uses — and a test suite builds a great many
        // backends, so that is both wasteful and a way to make tests flaky.
        if std::env::var_os("AERO_CT_ASYNC").is_none() {
            return None;
        }
        let threads = match std::env::var("AERO_JIT_COMPILE_THREADS").ok().as_deref() {
            Some("0") => return None,
            Some(n) => n.parse().unwrap_or(0),
            None => std::thread::available_parallelism()
                .map(|n| (n.get() / 2).clamp(1, 4))
                .unwrap_or(1),
        };
        if threads == 0 {
            return None;
        }
        let (job_tx, job_rx) = std::sync::mpsc::channel::<(u64, Vec<u8>)>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<(u64, Module)>();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        for _ in 0..threads {
            let engine = engine.clone();
            let job_rx = job_rx.clone();
            let done_tx = done_tx.clone();
            std::thread::Builder::new()
                .name("aero-jit-compile".into())
                .spawn(move || loop {
                    // The lock is held only to take a job, never across a compile.
                    let job = match job_rx.lock() {
                        Ok(rx) => rx.recv(),
                        Err(_) => return,
                    };
                    let Ok((id, wasm)) = job else { return };
                    let Ok(module) = Module::new(&engine, &wasm) else {
                        continue;
                    };
                    if done_tx.send((id, module)).is_err() {
                        return;
                    }
                })
                .ok()?;
        }
        Some(CompileWorkers {
            jobs: job_tx,
            done: done_rx,
            counts: std::collections::HashMap::new(),
            completed: std::collections::HashMap::new(),
            next_id: 0,
            inflight: 0,
        })
    }

    /// Hand a module to a worker. Returns the id its blocks will arrive under, or
    /// `None` if there are no workers and the caller must compile inline.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn submit_dispatched_blocks(&mut self, wasm: Vec<u8>, count: usize) -> Option<u64> {
        let w = self.workers.as_mut()?;
        let id = w.next_id;
        w.next_id += 1;
        if w.jobs.send((id, wasm)).is_err() {
            self.workers = None;
            return None;
        }
        w.counts.insert(id, count);
        w.inflight += 1;
        Some(id)
    }

    /// Move anything the workers have finished into the completed set.
    #[cfg(not(target_arch = "wasm32"))]
    fn drain_finished(&mut self) {
        if let Some(w) = self.workers.as_mut() {
            while let Ok((id, module)) = w.done.try_recv() {
                w.inflight -= 1;
                w.completed.insert(id, module);
            }
        }
    }

    /// Instantiate every module the workers have finished. Non-blocking.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_dispatched_blocks(&mut self) -> Vec<(u64, Vec<u32>)> {
        self.drain_finished();
        let ready: Vec<(u64, Module, usize)> = {
            let Some(w) = self.workers.as_mut() else {
                return Vec::new();
            };
            let ids: Vec<u64> = w.completed.keys().copied().collect();
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                let Some(module) = w.completed.remove(&id) else {
                    continue;
                };
                match w.counts.remove(&id) {
                    Some(count) => out.push((id, module, count)),
                    None => continue,
                }
            }
            out
        };
        ready
            .into_iter()
            .map(|(id, module, count)| (id, self.add_dispatched_module(&module, count)))
            .collect()
    }

    /// Instantiate submission `id`, waiting for the worker if it is not done yet.
    ///
    /// Waiting is what keeps the machine deterministic. Registering a module the
    /// instant a thread happens to finish it makes *when* a block becomes runnable
    /// depend on host scheduling, and the guest's path depends on that — measured,
    /// three runs of the same 60M-instruction workload differed by 2.3%, against five
    /// significant figures of agreement when compilation is inline. The caller
    /// instead names a fixed point in guest time to install it at; by then the
    /// compile has almost always finished, so this rarely blocks, and when it does the
    /// answer is still the same one every run.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn take_dispatched_blocks(&mut self, id: u64) -> Option<Vec<u32>> {
        self.drain_finished();
        let module = loop {
            let w = self.workers.as_mut()?;
            if let Some(module) = w.completed.remove(&id) {
                break module;
            }
            match w.done.recv() {
                Ok((got, module)) => {
                    w.inflight -= 1;
                    if got == id {
                        break module;
                    }
                    w.completed.insert(got, module);
                }
                Err(_) => return None,
            }
        };
        let count = self.workers.as_mut()?.counts.remove(&id)?;
        Some(self.add_dispatched_module(&module, count))
    }

    /// Whether compile workers were started.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn has_compile_workers(&self) -> bool {
        self.workers.is_some()
    }

    /// How many modules are still being compiled.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn compiles_in_flight(&self) -> usize {
        self.workers.as_ref().map_or(0, |w| w.inflight)
    }

    /// A handle to the baseline engine, for compiling modules off the CPU thread.
    ///
    /// `Engine` is `Send + Sync` and compiles concurrently, and `Module` is `Send`, so
    /// `Module::new` — which is 94% of what admitting a block costs — can happen on a
    /// worker while the guest keeps running. Instantiating the result is 2,303 host
    /// instructions and stays here, because `Store` is not `Send`.
    #[must_use]
    pub fn engine_handle(&self) -> Engine {
        self.engine.clone()
    }

    /// A handle to the Cranelift engine, when there is one.
    #[must_use]
    pub fn optimizing_engine_handle(&self) -> Option<Engine> {
        self.opt.as_ref().map(|opt| opt.engine.clone())
    }

    /// Register an already-compiled module's blocks. See [`Self::engine_handle`].
    ///
    /// # Panics
    /// If the module was compiled by a different engine, or does not export the
    /// dispatcher.
    #[track_caller]
    pub fn add_dispatched_module(&mut self, module: &Module, count: usize) -> Vec<u32> {
        let instance = self
            .linker
            .instantiate(&mut self.store, module)
            .expect("instantiate wasm module");
        let dispatch = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut self.store, EXPORT_TIER1_DISPATCH_FN)
            .expect("get exported tier1 dispatcher");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(self.blocks.len() as u32);
            self.blocks.push(BlockRef::Dispatched {
                dispatch: dispatch.clone(),
                index: i as i32,
            });
        }
        out
    }

    /// As [`Self::promote_blocks`], but for a module compiled elsewhere.
    #[track_caller]
    pub fn promote_module(&mut self, table_indices: &[u32], module: &Module) -> usize {
        let Some(opt) = self.opt.as_mut() else {
            return 0;
        };
        let Ok(instance) = opt.linker.instantiate(&mut opt.store, module) else {
            return 0;
        };
        let Ok(dispatch) = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut opt.store, EXPORT_TIER1_DISPATCH_FN)
        else {
            return 0;
        };
        let mut done = 0;
        for (i, &table_index) in table_indices.iter().enumerate() {
            let Some(slot) = self.blocks.get_mut(table_index as usize) else {
                continue;
            };
            if slot.is_optimized() {
                continue;
            }
            *slot = BlockRef::Optimized {
                dispatch: dispatch.clone(),
                index: i as i32,
            };
            done += 1;
        }
        done
    }

    /// Add a module whose blocks are reached through its single exported dispatcher.
    #[track_caller]
    pub fn add_dispatched_blocks(&mut self, wasm_bytes: &[u8], count: usize) -> Vec<u32> {
        let module = Module::new(&self.engine, wasm_bytes).expect("compile wasm module");
        let instance = self
            .linker
            .instantiate(&mut self.store, &module)
            .expect("instantiate wasm module");
        let dispatch = instance
            .get_typed_func::<(i32, i32, i32), i64>(&mut self.store, EXPORT_TIER1_DISPATCH_FN)
            .expect("get exported tier1 dispatcher");
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(self.blocks.len() as u32);
            self.blocks.push(BlockRef::Dispatched {
                dispatch: dispatch.clone(),
                index: i as i32,
            });
        }
        out
    }

    pub fn add_compiled_block(&mut self, wasm_bytes: &[u8]) -> u32 {
        let module = Module::new(&self.engine, wasm_bytes).expect("compile wasm module");
        let instance = self
            .linker
            .instantiate(&mut self.store, &module)
            .expect("instantiate wasm module");
        let func = instance
            .get_typed_func::<(i32, i32), i64>(&mut self.store, EXPORT_TIER1_BLOCK_FN)
            .expect("get exported tier1 block function");
        let idx = self.blocks.len() as u32;
        self.blocks.push(BlockRef::Exported(func));
        idx
    }

    /// Address of `CpuState` field `off` in the linear memory.
    ///
    /// Safety: the base never moves (see `mem_base`), and `cpu_ptr + CPU_STATE_SIZE`
    /// was bounds-checked against the memory at construction.
    #[inline]
    fn cpu_field(&self, off: u32) -> *mut u64 {
        // Safety: as above.
        unsafe { self.mem_base.add(self.cpu_ptr as usize + off as usize) as *mut u64 }
    }

    #[inline]
    fn poke(&self, off: u32, value: u64) {
        // Safety: `cpu_field` is in bounds, and no compiled block is running.
        unsafe { std::ptr::write_unaligned(self.cpu_field(off), value) }
    }

    /// Guest instructions the block just run reported retiring.
    ///
    /// Safety: `jit_ctx_ptr + JitContext::TOTAL_BYTE_SIZE` was bounds-checked at
    /// construction, and no block is running.
    #[inline]
    pub fn last_retired(&self) -> u64 {
        unsafe {
            std::ptr::read_unaligned(
                self.mem_base
                    .add(self.jit_ctx_ptr as usize + JitContext::RETIRED_OFFSET as usize)
                    as *const u64,
            )
        }
    }

    #[inline]
    fn peek(&self, off: u32) -> u64 {
        // Safety: as `poke`.
        unsafe { std::ptr::read_unaligned(self.cpu_field(off)) }
    }

    fn sync_cpu_to_wasm(&mut self, cpu: &CoreCpuState) {
        for i in 0..16 {
            self.poke(abi::CPU_GPR_OFF[i], cpu.gpr[i]);
        }
        self.poke(abi::CPU_RIP_OFF, cpu.rip);
        self.poke(abi::CPU_RFLAGS_OFF, cpu.rflags_snapshot());

        // Nothing is owed on entry: the RFLAGS just written is a fully materialised
        // snapshot. Leaving the descriptor region alone would hand the block whatever
        // an *earlier* block left there, and a block that reads a flag on arrival
        // would settle that stale record over the correct flags.
        self.poke(abi::CPU_LAZY_FLAGS_OFF, 0);

        // `fs:`/`gs:`-relative addressing reads these straight out of `CpuState`, so
        // they have to be here before a block runs. Only this direction matters:
        // compiled code never writes a segment base, because the instructions that
        // change one are not translated.
        self.poke(abi::CPU_FS_BASE_OFF, cpu.msr.fs_base);
        self.poke(abi::CPU_GS_BASE_OFF, cpu.msr.gs_base);

        // The JIT context header goes with it: compiled code reads guest RAM's base
        // and the TLB salt from here on entry, and a block that finds a zero base
        // addresses guest memory from the wrong place entirely.
        //
        // Safety: `jit_ctx_ptr + JitContext::TOTAL_BYTE_SIZE` was bounds-checked
        // against the linear memory at construction, and no block is running.
        unsafe {
            let ctx = self.mem_base.add(self.jit_ctx_ptr as usize);
            std::ptr::write_unaligned(
                ctx.add(JitContext::RAM_BASE_OFFSET as usize) as *mut u64,
                self.ram_base,
            );
            std::ptr::write_unaligned(
                ctx.add(JitContext::TLB_SALT_OFFSET as usize) as *mut u64,
                self.tlb_salt,
            );
        }
    }

    fn sync_cpu_from_wasm(&mut self, cpu: &mut CoreCpuState) {
        for i in 0..16 {
            cpu.gpr[i] = self.peek(abi::CPU_GPR_OFF[i]);
        }
        cpu.rip = self.peek(abi::CPU_RIP_OFF);
        cpu.set_rflags(self.peek(abi::CPU_RFLAGS_OFF));

        // Compiled code hands back the operation that produced the flags rather than
        // the flags themselves, so the descriptor has to come with it. `set_rflags`
        // above clears any pending state, which is why this follows it.
        let header = self.peek(abi::CPU_LAZY_FLAGS_OFF);
        if let Some(op) = lazy_op_from_abi(header as u8) {
            cpu.lazy_flags = aero_cpu_core::state::LazyFlags {
                op,
                size: lazy_size_from_abi((header >> 8) as u8),
                carry_in: ((header >> 16) & 1) as u8,
                _pad0: [0; 5],
                result: self.peek(abi::CPU_LAZY_RESULT_OFF),
                op1: self.peek(abi::CPU_LAZY_OP1_OFF),
                op2: self.peek(abi::CPU_LAZY_OP2_OFF),
            };
        }
    }
}

impl<Cpu> Tier1Bus for WasmtimeBackend<Cpu> {
    fn read_u8(&self, addr: u64) -> u8 {
        self.memory.data(&self.store)[addr as usize]
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        self.memory.data_mut(&mut self.store)[addr as usize] = value;
    }
}

impl<Cpu: Tier1Cpu> WasmtimeBackend<Cpu> {
    /// Clear the flags a block raises on its way out, in **both** stores.
    ///
    /// A chain crosses between them freely, so a flag left behind by a promoted
    /// block in one would be read as this chain's bail in the other.
    fn reset_exit_state(&mut self) {
        self.store.data_mut().exit.reset();
        self.store.data_mut().pending_writes.clear();
        if let Some(opt) = self.opt.as_mut() {
            opt.store.data_mut().exit.reset();
            opt.store.data_mut().pending_writes.clear();
        }
    }

    /// Run a chain of compiled blocks, marshalling the CPU state once for the whole
    /// chain rather than once per block.
    ///
    /// [`JitBackend::execute`] copies sixteen GPRs plus RIP and RFLAGS into linear
    /// memory, calls the block, and copies them all back — about 293ns of fixed
    /// cost, measured, against basic blocks that are three to five instructions
    /// long. Compiled code itself runs at roughly 7ns per guest instruction, so
    /// that round trip is an order of magnitude more expensive than the work it
    /// wraps, and it is why compiled execution measures only ~1.5x faster than the
    /// interpreter at realistic block lengths.
    ///
    /// The architectural state lives in linear memory for the duration of the
    /// chain, so successive blocks read each other's results directly. Only the
    /// next entry point comes back to the host, to be resolved by `lookup`.
    ///
    /// Returns `(next_rip, blocks_executed, exit_to_interpreter)`.
    ///
    /// Rollback is the caller's responsibility: on `exit_to_interpreter` the guest
    /// state is whatever the chain left behind, exactly as if the interpreter had
    /// run those blocks. Blocks that request a host exit mid-chain are therefore
    /// only safe here if their partial effects are ones the interpreter can resume
    /// from — the same condition the IR chain in `aero-machine` works under.
    pub fn execute_chain(
        &mut self,
        entry_index: u32,
        cpu: &mut Cpu,
        lookup: &mut dyn FnMut(u64, u64) -> Option<u32>,
        max_blocks: u32,
    ) -> crate::compiled_backend::ChainOutcome {
        self.reset_exit_state();
        self.sync_cpu_to_wasm(cpu.tier1_state());

        // Captured before the loop: the body destructures `self` to borrow the
        // function table and the store as disjoint fields, which rules out calling a
        // method on `self` inside it.
        let retired_slot = {
            // Safety: bounds-checked at construction; see `last_retired`.
            unsafe {
                self.mem_base
                    .add(self.jit_ctx_ptr as usize + JitContext::RETIRED_OFFSET as usize)
                    as *const u64
            }
        };
        // Where a chaining dispatcher accumulates what it ran without coming back.
        // Safety: as for `retired_slot`.
        let (chain_retired_slot, chain_blocks_slot) = unsafe {
            let ctx = self.mem_base.add(self.jit_ctx_ptr as usize);
            (
                ctx.add(JitContext::CHAIN_RETIRED_OFFSET as usize) as *mut u64,
                ctx.add(JitContext::CHAIN_BLOCKS_OFFSET as usize) as *mut u32,
            )
        };
        // Safety: as above; `cpu_ptr + CPU_STATE_SIZE` was bounds-checked too.
        let rip_slot =
            unsafe { self.mem_base.add(self.cpu_ptr as usize + abi::CPU_RIP_OFF as usize) };

        let mut index = entry_index;
        let mut executed = 0u32;
        let mut retired = 0u64;
        let mut next_rip = cpu.tier1_state().rip;
        let mut exit_to_interpreter = false;

        while executed < max_blocks {
            // Borrow the function table and the store as *disjoint fields* rather
            // than cloning the `TypedFunc` out. Cloning it per block showed up as
            // `RegisteredType::clone` and `malloc` in a profile — a refcount and an
            // allocation on a path taken once per block.
            let Self {
                blocks, store, opt, ..
            } = self;
            let Some(func) = blocks.get(index as usize) else {
                break;
            };
            // A promoted block lives in the Cranelift store, and so do the exit flags
            // it raises. Both stores see the same guest memory, so the chain crosses
            // without any state moving; only which store to ask changes.
            let store: &mut Store<HostState> = if func.is_optimized() {
                match opt.as_mut() {
                    Some(opt) => &mut opt.store,
                    None => break,
                }
            } else {
                store
            };
            // Skipping this for modules that cannot chain — carrying a `chains` flag
            // on the block reference to know which those are — measured 129.2 against
            // 128.8 and 101.2 against 100.9. The branch costs what the two stores and
            // the read cost, and a bool through five trait methods is not free to read
            // either.
            // Safety: no block is running, and both slots are in bounds.
            unsafe {
                std::ptr::write_unaligned(chain_retired_slot, 0);
                std::ptr::write_unaligned(chain_blocks_slot, 0);
            }
            let ret = func.call(&mut *store, self.cpu_ptr, self.jit_ctx_ptr);
            // Safety: as above.
            let inner = unsafe { std::ptr::read_unaligned(chain_blocks_slot) };
            if inner == 0 {
                // A module whose dispatcher does not chain — or no dispatcher at all.
                // One block ran and wrote its own count, as it always has.
                executed += 1;
                // Safety: as above.
                retired += unsafe { std::ptr::read_unaligned(retired_slot) };
            } else {
                // The dispatcher ran `inner` blocks back to back and folded each
                // one's count in as it went. Adding `retired` as well would count the
                // last block twice.
                executed += inner;
                self.wasm_chain_steps += u64::from(inner - 1);
                // Safety: as above.
                retired += unsafe { std::ptr::read_unaligned(chain_retired_slot) };
            }

            if ret == JIT_EXIT_SENTINEL_I64 {
                // The sentinel means two different things and they must not be
                // confused. A block that bailed part-way — MMIO, a page fault, a
                // runtime exit — leaves a flag set, retired an unknown prefix of its
                // instructions, and has to hand control back untallied.
                //
                // A block whose *terminator* is `ExitToInterpreter` sets no flag: it
                // ran every one of its instructions and RIP is the next one. Those
                // are the blocks that ended at the decoder's instruction limit rather
                // than at a branch. Treating them as bails leaves their instructions
                // retired by the guest but uncounted by the tier, and the machine's
                // timing follows that count — which is why shortening blocks to two
                // or four instructions, where nearly every block ends at the limit,
                // made the tier disagree with the interpreter.
                if store.data().exit.should_rollback() {
                    exit_to_interpreter = true;
                    break;
                }
                // Read through the mapping rather than through a `Memory` handle:
                // the block may have run in the other store, and a handle belongs to
                // the store that made it. Both stores map the same pages, so the
                // base pointer is the one thing that is true for either.
                //
                // Safety: the base does not move, and the field is in bounds.
                next_rip = unsafe {
                    std::ptr::read_unaligned(
                        rip_slot as *const u64,
                    )
                };
            } else {
                next_rip = ret as u64;
            }

            match lookup(next_rip, retired) {
                Some(next) => index = next,
                None => break,
            }
        }

        self.sync_cpu_from_wasm(cpu.tier1_state_mut());
        if exit_to_interpreter {
            next_rip = cpu.tier1_state().rip;
        } else {
            // The last block's return value is where the guest goes next, and it is
            // authoritative — a block that runs to its end need not also store it, and
            // when `AERO_JIT_TRIM_RIP` is on it does not. A bail is the other case, and
            // it is handled above from what the block wrote.
            cpu.tier1_state_mut().rip = next_rip;
        }
        crate::compiled_backend::ChainOutcome {
            next_rip,
            retired,
            blocks: executed,
            exit_to_interpreter,
        }
    }
}

impl<Cpu> JitBackend for WasmtimeBackend<Cpu>
where
    Cpu: Tier1Cpu,
{
    type Cpu = Cpu;

    fn execute(&mut self, table_index: u32, cpu: &mut Self::Cpu) -> JitBlockExit {
        let func = match self.blocks.get(table_index as usize).cloned() {
            Some(func) => func,
            None => panic!(
                "invalid JIT table index {table_index} (len={})",
                self.blocks.len()
            ),
        };

        // Snapshot state so we can roll back side effects if the block performs an MMIO/runtime
        // exit. Tier-1 blocks do not currently provide deopt metadata for resuming mid-block, so
        // the conservative fallback is to roll back and let the interpreter re-execute.
        let pre_state = cpu.tier1_state().clone();
        let ram_len = usize::try_from(self.cpu_ptr).expect("cpu_ptr must be non-negative");
        let pre_ram = self.memory.data(&self.store)[..ram_len].to_vec();
        let pre_code_versions: Option<(usize, Vec<u8>)> = if self.code_version_table_len == 0 {
            None
        } else {
            let mem = self.memory.data(&self.store);
            let table_ptr = self.code_version_table_ptr as usize;
            let table_len = self.code_version_table_len as usize;
            let table_bytes = table_len
                .checked_mul(4)
                .expect("code version table byte size overflow");
            Some((table_ptr, mem[table_ptr..table_ptr + table_bytes].to_vec()))
        };

        self.reset_exit_state();
        self.sync_cpu_to_wasm(cpu.tier1_state());

        // A promoted block runs in the other store, and raises its flags there.
        let (ret, host_exit) = if func.is_optimized() {
            let opt = self.opt.as_mut().expect("optimized block without its engine");
            let ret = func.call(&mut opt.store, self.cpu_ptr, self.jit_ctx_ptr);
            (ret, opt.store.data().exit)
        } else {
            let ret = func.call(&mut self.store, self.cpu_ptr, self.jit_ctx_ptr);
            (ret, self.store.data().exit)
        };

        let exit_to_interpreter = ret == JIT_EXIT_SENTINEL_I64;
        if exit_to_interpreter && host_exit.should_rollback() {
            // Restore guest RAM and CPU state snapshot.
            self.memory
                .write(&mut self.store, 0, &pre_ram)
                .expect("restore guest RAM");
            if let Some((table_ptr, table_snapshot)) = pre_code_versions {
                self.memory
                    .write(&mut self.store, table_ptr, &table_snapshot)
                    .expect("restore code version table");
            }
            *cpu.tier1_state_mut() = pre_state.clone();

            return JitBlockExit {
                next_rip: pre_state.rip,
                exit_to_interpreter: true,
                committed: false,
            };
        }

        self.sync_cpu_from_wasm(cpu.tier1_state_mut());
        if !exit_to_interpreter {
            // As in `execute_chain`: the block's return value is where the guest goes
            // next, and it is authoritative. A block that always runs to its end no
            // longer stores RIP itself, so the caller has to take it from here.
            cpu.tier1_state_mut().rip = ret as u64;
        }

        // Deliver write-observer notifications for committed slow-path stores.
        //
        // The backend may roll back guest state on MMIO/runtime exits (committed=false); in that
        // case any imported `env.mem_write_*` side effects must *not* be forwarded to external
        // observers (e.g. a host-side page-version tracker).
        let (mut observer, pending_writes) = {
            let host = self.store.data_mut();
            let pending = std::mem::take(&mut host.pending_writes);
            (host.hooks.borrow_mut().write_observer.take(), pending)
        };
        if let Some(obs) = observer.as_mut() {
            for (paddr, len) in pending_writes {
                obs(paddr, len);
            }
        }
        self.store.data().hooks.borrow_mut().write_observer = observer;

        let next_rip = if exit_to_interpreter {
            cpu.tier1_state().rip
        } else {
            ret as u64
        };

        JitBlockExit {
            next_rip,
            exit_to_interpreter,
            committed: true,
        }
    }

    fn code_version_table(&mut self) -> Option<Box<dyn CodeVersionTable>> {
        // This backend always configures the `(ptr,len)` slots in the Tier-2 context ABI. When the
        // table does not fit, `len` is 0 and JIT code treats the table as disabled.
        Some(Box::new(WasmtimeCodeVersionTable {
            ptr: self.code_version_table_host_ptr,
            len: self.code_version_table_len as usize,
        }))
    }
}

impl<Cpu> Tier1WasmRegistry for WasmtimeBackend<Cpu> {
    fn register_tier1_block(&mut self, wasm: Vec<u8>, _exit_to_interpreter: bool) -> u32 {
        self.add_compiled_block(&wasm)
    }
}

/// Linear-memory offset for one guest byte, honouring an installed translator.
///
/// The slow helpers exist for accesses the inline TLB declines — notably ones that
/// cross a page boundary — so translation is per byte rather than per access: the
/// two halves of a crossing access live on different pages, and with a translator
/// installed those pages need not be adjacent.
///
/// `None` means the access faults; the caller records a page-fault exit and the
/// block bails to the interpreter, which re-raises it architecturally.
/// Whether `addr` translates to a page the engine's linear memory actually backs.
///
/// `None` means the translation failed. `Some(false)` means it succeeded and named
/// something that is not RAM — a device register — which linear memory does not hold
/// and which must be reached through the embedder's bus instead.
fn translates_to_ram(caller: &mut Caller<'_, HostState>, addr: u64, write: bool) -> Option<bool> {
    let Some(mut translate) = caller.data().hooks.borrow_mut().translator.take() else {
        // No translator: the embedder does not page, so everything is linear memory.
        return Some(true);
    };
    let out = translate(addr, if write { 1 } else { 0 });
    caller.data().hooks.borrow_mut().translator = Some(translate);
    let (_page, flags) = out?;
    Some(flags & crate::TLB_FLAG_IS_RAM != 0)
}

/// Decode the `op` byte of a lazy-flag descriptor. `None` means nothing is pending.
fn lazy_op_from_abi(code: u8) -> Option<aero_cpu_core::state::LazyFlagOp> {
    use aero_cpu_core::state::LazyFlagOp;
    match code {
        1 => Some(LazyFlagOp::Add),
        2 => Some(LazyFlagOp::Adc),
        3 => Some(LazyFlagOp::Sub),
        4 => Some(LazyFlagOp::Sbb),
        5 => Some(LazyFlagOp::Logic),
        _ => None,
    }
}

/// Decode the `size` byte, which counts operand bytes.
fn lazy_size_from_abi(code: u8) -> aero_cpu_core::state::OperandSize {
    use aero_cpu_core::state::OperandSize;
    match code {
        1 => OperandSize::Byte,
        2 => OperandSize::Word,
        4 => OperandSize::Dword,
        _ => OperandSize::Qword,
    }
}

fn linear_offset(caller: &mut Caller<'_, HostState>, addr: u64, write: bool) -> Option<usize> {
    let Some(mut translate) = caller.data().hooks.borrow_mut().translator.take() else {
        // No translator: the embedder does not page, so a guest address *is* a
        // linear-memory offset.
        return usize::try_from(addr).ok();
    };
    let out = translate(addr, if write { 1 } else { 0 });
    caller.data().hooks.borrow_mut().translator = Some(translate);

    let (page, _flags) = out?;
    let ram_base = caller.data().hooks.borrow().ram_base;
    usize::try_from(ram_base.checked_add(page)?.checked_add(addr & 0xfff)?).ok()
}

/// Read `n` bytes of guest memory.
///
/// An access that translates to something other than RAM goes to the embedder's bus
/// at its natural width. That is what makes a device register behave like one:
/// resolving it inside linear memory is not a transaction at all, and splitting it
/// into `n` one-byte transactions is the wrong one. RAM keeps the direct path, which
/// is also where guest writes get their code-version bookkeeping.
fn guest_read(caller: &mut Caller<'_, HostState>, mem: &Memory, addr: u64, n: usize) -> u64 {
    // Bound first: the borrow would otherwise still be held when
    // `translates_to_ram` takes a mutable one.
    let has_accessor = caller.data().hooks.borrow().accessor.is_some();
    if has_accessor && translates_to_ram(caller, addr, false) == Some(false) {
        let mut access = caller.data().hooks.borrow_mut().accessor.take().expect("checked above");
        let out = access(addr, n as u32, None);
        caller.data().hooks.borrow_mut().accessor = Some(access);
        return match out {
            Some(v) => v,
            None => {
                caller.data_mut().exit.page_fault = true;
                0
            }
        };
    }
    let mut v = 0u64;
    for i in 0..n {
        let Some(off) = linear_offset(caller, addr.wrapping_add(i as u64), false) else {
            caller.data_mut().exit.page_fault = true;
            return 0;
        };
        let Some(byte) = mem.data(&*caller).get(off).copied() else {
            caller.data_mut().exit.page_fault = true;
            return 0;
        };
        v |= u64::from(byte) << (i * 8);
    }
    v
}

/// Write `n` bytes of guest memory. See [`guest_read`] for why the accessor wins.
fn guest_write(caller: &mut Caller<'_, HostState>, mem: &Memory, addr: u64, n: usize, value: u64) {
    // Bound first: the borrow would otherwise still be held when
    // `translates_to_ram` takes a mutable one.
    let has_accessor = caller.data().hooks.borrow().accessor.is_some();
    if has_accessor && translates_to_ram(caller, addr, true) == Some(false) {
        let mut access = caller.data().hooks.borrow_mut().accessor.take().expect("checked above");
        let out = access(addr, n as u32, Some(value));
        caller.data().hooks.borrow_mut().accessor = Some(access);
        if out.is_none() {
            caller.data_mut().exit.page_fault = true;
        }
        return;
    }
    for i in 0..n {
        let Some(off) = linear_offset(caller, addr.wrapping_add(i as u64), true) else {
            caller.data_mut().exit.page_fault = true;
            return;
        };
        let byte = (value >> (i * 8)) as u8;
        match mem.data_mut(&mut *caller).get_mut(off) {
            Some(slot) => *slot = byte,
            None => {
                caller.data_mut().exit.page_fault = true;
                return;
            }
        }
    }
}

fn define_mem_helpers(linker: &mut Linker<HostState>, memory: Memory) {
    fn read_u32_le(mem: &[u8], off: usize) -> u32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&mem[off..off + 4]);
        u32::from_le_bytes(buf)
    }

    fn write_u32_le(mem: &mut [u8], off: usize, value: u32) {
        mem[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn bump_code_versions(mem: &mut [u8], cpu_ptr: i32, paddr: u64, len: usize) {
        if len == 0 {
            return;
        }

        let cpu_base = cpu_ptr as usize;
        let table_len = read_u32_le(
            mem,
            cpu_base + jit_ctx::CODE_VERSION_TABLE_LEN_OFFSET as usize,
        ) as u64;
        if table_len == 0 {
            return;
        }

        let table_ptr = read_u32_le(
            mem,
            cpu_base + jit_ctx::CODE_VERSION_TABLE_PTR_OFFSET as usize,
        ) as u64;

        let start_page = paddr >> crate::PAGE_SHIFT;
        let end = paddr.saturating_add(len as u64 - 1);
        let end_page = end >> crate::PAGE_SHIFT;

        for page in start_page..=end_page {
            if page >= table_len {
                continue;
            }
            let entry_off = table_ptr + page * 4;
            let entry_off = usize::try_from(entry_off).unwrap_or(usize::MAX);
            if entry_off.saturating_add(4) > mem.len() {
                continue;
            }
            let cur = read_u32_le(mem, entry_off);
            write_u32_le(mem, entry_off, cur.wrapping_add(1));
        }
    }

    // Reads.
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U8,
                move |mut caller: Caller<'_, HostState>, _cpu_ptr: i32, addr: i64| -> i32 {
                    guest_read(&mut caller, &mem, addr as u64, 1) as i32
                },
            )
            .expect("define mem_read_u8");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U16,
                move |mut caller: Caller<'_, HostState>, _cpu_ptr: i32, addr: i64| -> i32 {
                    guest_read(&mut caller, &mem, addr as u64, 2) as i32
                },
            )
            .expect("define mem_read_u16");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U32,
                move |mut caller: Caller<'_, HostState>, _cpu_ptr: i32, addr: i64| -> i32 {
                    guest_read(&mut caller, &mem, addr as u64, 4) as i32
                },
            )
            .expect("define mem_read_u32");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_READ_U64,
                move |mut caller: Caller<'_, HostState>, _cpu_ptr: i32, addr: i64| -> i64 {
                    guest_read(&mut caller, &mem, addr as u64, 8) as i64
                },
            )
            .expect("define mem_read_u64");
    }

    // Writes.
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U8,
                move |mut caller: Caller<'_, HostState>, cpu_ptr: i32, addr: i64, value: i32| {
                    let addr_u = addr as u64;
                    guest_write(&mut caller, &mem, addr_u, 1, value as u64);
                    {
                        let mem_mut = mem.data_mut(&mut caller);
                        bump_code_versions(mem_mut, cpu_ptr, addr_u, 1);
                    }

                    let host = caller.data_mut();
                    if host.hooks.borrow_mut().write_observer.is_some() {
                        host.pending_writes.push((addr_u, 1));
                    }
                },
            )
            .expect("define mem_write_u8");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U16,
                move |mut caller: Caller<'_, HostState>, cpu_ptr: i32, addr: i64, value: i32| {
                    let addr_u = addr as u64;
                    guest_write(&mut caller, &mem, addr_u, 2, value as u64);
                    {
                        let mem_mut = mem.data_mut(&mut caller);
                        bump_code_versions(mem_mut, cpu_ptr, addr_u, 2);
                    }
                    let host = caller.data_mut();
                    if host.hooks.borrow_mut().write_observer.is_some() {
                        host.pending_writes.push((addr_u, 2));
                    }
                },
            )
            .expect("define mem_write_u16");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U32,
                move |mut caller: Caller<'_, HostState>, cpu_ptr: i32, addr: i64, value: i32| {
                    let addr_u = addr as u64;
                    guest_write(&mut caller, &mem, addr_u, 4, value as u64);
                    {
                        let mem_mut = mem.data_mut(&mut caller);
                        bump_code_versions(mem_mut, cpu_ptr, addr_u, 4);
                    }
                    let host = caller.data_mut();
                    if host.hooks.borrow_mut().write_observer.is_some() {
                        host.pending_writes.push((addr_u, 4));
                    }
                },
            )
            .expect("define mem_write_u32");
    }
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MEM_WRITE_U64,
                move |mut caller: Caller<'_, HostState>, cpu_ptr: i32, addr: i64, value: i64| {
                    let addr_u = addr as u64;
                    guest_write(&mut caller, &mem, addr_u, 8, value as u64);
                    {
                        let mem_mut = mem.data_mut(&mut caller);
                        bump_code_versions(mem_mut, cpu_ptr, addr_u, 8);
                    }
                    let host = caller.data_mut();
                    if host.hooks.borrow_mut().write_observer.is_some() {
                        host.pending_writes.push((addr_u, 8));
                    }
                },
            )
            .expect("define mem_write_u64");
    }
}

fn define_stub_helpers(linker: &mut Linker<HostState>, memory: Memory) {
    // Present for ABI completeness. When called, treat these as runtime exits and roll back state.
    linker
        .func_wrap(
            IMPORT_MODULE,
            IMPORT_PAGE_FAULT,
            |mut caller: Caller<'_, HostState>, _cpu_ptr: i32, _addr: i64| -> i64 {
                caller.data_mut().exit.page_fault = true;
                JIT_EXIT_SENTINEL_I64
            },
        )
        .expect("define page_fault");

    // Minimal inline-TLB translation helper: identity map addresses that fall within the guest RAM
    // window (0..cpu_ptr) and classify anything else as MMIO.
    {
        let mem = memory;
        linker
            .func_wrap(
                IMPORT_MODULE,
                IMPORT_MMU_TRANSLATE,
                move |mut caller: Caller<'_, HostState>,
                      cpu_ptr: i32,
                      jit_ctx_ptr: i32,
                      vaddr: i64,
                      _access: i32|
                      -> i64 {
                    JIT_TLB_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let vaddr_u = vaddr as u64;
                    let vpn = vaddr_u >> crate::PAGE_SHIFT;
                    let idx = vpn & crate::JIT_TLB_INDEX_MASK;

                    let tlb_salt = {
                        let addr = jit_ctx_ptr as usize + (JitContext::TLB_SALT_OFFSET as usize);
                        let bytes: [u8; 8] = mem.data(&caller)[addr..addr + 8].try_into().unwrap();
                        u64::from_le_bytes(bytes)
                    };

                    // tag = (vpn ^ salt) | 1, keep tag 0 reserved for invalidation.
                    let tag = (vpn ^ tlb_salt) | 1;

                    // Bound first: a `borrow_mut()` in a match scrutinee lives for
                    // the whole match, and the arms below borrow again.
                    let taken = caller.data().hooks.borrow_mut().translator.take();
                    let translated = match taken {
                        Some(mut translate) => {
                            let out = translate(vaddr_u, _access as u32);
                            caller.data().hooks.borrow_mut().translator = Some(translate);
                            Some(out)
                        }
                        None => None,
                    };

                    let (phys_base, flags) = match translated {
                        // The embedder pages: it decides the physical page and what
                        // the mapping permits.
                        Some(Some((phys, flags))) => (phys & crate::PAGE_BASE_MASK, flags),
                        // The embedder pages and this access faults. Leave the TLB
                        // entry invalid and report no permissions, so the block takes
                        // its slow path and exits to the interpreter rather than
                        // caching a mapping that does not exist.
                        Some(None) => {
                            caller.data_mut().exit.page_fault = true;
                            return 0;
                        }
                        // No embedder translator: identity map guest RAM.
                        None => {
                            let is_ram = vaddr_u < cpu_ptr as u64;
                            (
                                vaddr_u & crate::PAGE_BASE_MASK,
                                crate::TLB_FLAG_READ
                                    | crate::TLB_FLAG_WRITE
                                    | crate::TLB_FLAG_EXEC
                                    | if is_ram { crate::TLB_FLAG_IS_RAM } else { 0 },
                            )
                        }
                    };
                    let data = phys_base | flags;

                    let entry_addr = jit_ctx_ptr as usize
                        + (JitContext::TLB_OFFSET as usize)
                        + (idx as usize) * (crate::JIT_TLB_ENTRY_SIZE as usize);
                    let mem_mut = mem.data_mut(&mut caller);
                    mem_mut[entry_addr..entry_addr + 8].copy_from_slice(&tag.to_le_bytes());
                    mem_mut[entry_addr + 8..entry_addr + 16].copy_from_slice(&data.to_le_bytes());

                    data as i64
                },
            )
            .expect("define mmu_translate");
    }

    linker
        .func_wrap(
            IMPORT_MODULE,
            IMPORT_JIT_EXIT_MMIO,
            |mut caller: Caller<'_, HostState>,
             _cpu_ptr: i32,
             _vaddr: i64,
             _size: i32,
             _is_write: i32,
             _value: i64,
             rip: i64|
             -> i64 {
                caller.data_mut().exit.mmio_exit = true;
                // Return the RIP the block should resume at after the runtime has handled the
                // MMIO access. The Tier-1 code generator returns the sentinel separately.
                rip
            },
        )
        .expect("define jit_exit_mmio");

    // One guest instruction Tier-1 could not translate, performed here.
    //
    // The block has just written every register, the flags and RIP to `CpuState` in
    // linear memory, so that is where the architectural state is; this reads it into
    // a scratch `CpuState`, hands that to whatever the embedder installed, and writes
    // back what comes out. The block reloads from there and carries on.
    linker
        .func_wrap(
            IMPORT_MODULE,
            IMPORT_ASSIST,
            // `memory` is captured, not looked up: a block module *imports* the
            // memory, so asking the calling instance for an export of that name finds
            // nothing — and the closure then reports every assist as a fault, which
            // reads as correct behaviour with the assist silently never happening.
            move |mut caller: Caller<'_, HostState>, cpu_ptr: i32, rip: i64| -> i64 {
                let taken = caller.data().hooks.borrow_mut().assist.take();
                let Some(mut assist) = taken else {
                    return JIT_EXIT_SENTINEL_I64;
                };
                let mut regs = crate::compiled_backend::AssistRegs::default();
                read_cpu_state(&memory, &caller, cpu_ptr, &mut regs);
                regs.rip = rip as u64;
                let out = assist(&mut regs);
                if out.is_some() {
                    write_cpu_state(&memory, &mut caller, cpu_ptr, &regs);
                    if regs.flush_tlb {
                        // A different salt makes every tag in the table mismatch, so
                        // the entries filled under the old translation are simply
                        // never found again. The block reloads the salt right after
                        // this returns; the next chain entry re-flushes anyway,
                        // because the MMU generation it compares has moved.
                        let ctx = cpu_ptr + abi::CPU_STATE_SIZE as i32;
                        let data = memory.data_mut(&mut caller);
                        let at = ctx as usize + JitContext::TLB_SALT_OFFSET as usize;
                        let old = u64::from_le_bytes(data[at..at + 8].try_into().expect("8 bytes"));
                        data[at..at + 8]
                            .copy_from_slice(&old.wrapping_add(0x9E37_79B9_7F4A_7C15).to_le_bytes());
                    }
                }
                caller.data().hooks.borrow_mut().assist = Some(assist);
                match out {
                    Some(next) => next as i64,
                    None => JIT_EXIT_SENTINEL_I64,
                }
            },
        )
        .expect("define assist");

    linker
        .func_wrap(
            IMPORT_MODULE,
            IMPORT_RDTSC,
            |caller: Caller<'_, HostState>, _cpu_ptr: i32| -> i64 {
                // No clock installed means no block reading one should exist; a zero
                // is a visible wrong answer rather than a silent plausible one.
                // Taken out and put back, so the closure is free to reach anything
                // else behind the same hooks while it runs.
                let taken = caller.data().hooks.borrow_mut().clock.take();
                match taken {
                    Some(mut clock) => {
                        let tsc = clock() as i64;
                        caller.data().hooks.borrow_mut().clock = Some(clock);
                        tsc
                    }
                    None => 0,
                }
            },
        )
        .expect("define rdtsc");

    linker
        .func_wrap(
            IMPORT_MODULE,
            IMPORT_JIT_EXIT,
            |mut caller: Caller<'_, HostState>, _kind: i32, rip: i64| -> i64 {
                caller.data_mut().exit.jit_exit = true;
                // Like `jit_exit_mmio`, return the RIP to resume at while the caller uses the
                // sentinel return value to request an interpreter step.
                rip
            },
        )
        .expect("define jit_exit");
}

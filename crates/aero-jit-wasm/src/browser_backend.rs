//! A [`CompiledBackend`] over the browser's own WebAssembly engine.
//!
//! Natively the compiled tier drives Wasmtime. In the browser the engine *is* the
//! browser, reached through `WebAssembly.Module` and `WebAssembly.Instance` rather
//! than a host crate, and this is the whole of the difference: everything above it —
//! block discovery, translation, dead-flag elimination, the hotness policy, chain
//! execution — is the same code.
//!
//! The layout is simpler here than natively, because there is only one linear memory.
//! `CpuState`, the JIT context and guest RAM are ordinary allocations in this module's
//! memory, and compiled blocks import that same memory, so a guest load compiles to a
//! wasm load against a pointer the emulator already holds. Nothing is copied in or out
//! around a chain.

use alloc_shim::*;

use aero_cpu_core::state::CpuState;
use aero_jit_x86::compiled_backend::{CompiledBackend, GuestAccessor, GuestTranslator};
use aero_jit_x86::jit_ctx::JitContext;
use aero_jit_x86::wasm::{
    IMPORT_CODE_PAGE_VERSION, IMPORT_JIT_EXIT, IMPORT_JIT_EXIT_MMIO, IMPORT_MEMORY, IMPORT_MEM_READ_U16, IMPORT_MEM_READ_U32, IMPORT_MEM_READ_U64,
    IMPORT_MEM_READ_U8, IMPORT_MEM_WRITE_U16, IMPORT_MEM_WRITE_U32, IMPORT_MEM_WRITE_U64,
    IMPORT_MEM_WRITE_U8, IMPORT_MMU_TRANSLATE, IMPORT_MODULE, IMPORT_PAGE_FAULT, IMPORT_RDTSC,
    JIT_EXIT_SENTINEL_I64,
};
use aero_jit_x86::tier1::EXPORT_TIER1_DISPATCH_FN;
use js_sys::{Function, Object, Reflect, Uint8Array, WebAssembly};

/// How to reach one compiled block.
///
/// A module exports one dispatcher rather than one function per block: a WASM
/// function that can be named from outside needs a host-callable trampoline built
/// for it, and that dominated compile time. The browser pays the same cost, on the
/// worker thread, so it takes the same route.
#[derive(Clone)]
struct BlockRef {
    dispatch: Function,
    index: JsValue,
}
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

mod alloc_shim {
    pub use std::cell::RefCell;
    pub use std::rc::Rc;
}

/// State the imported helpers reach, shared with the backend that installed them.
///
/// The helpers are defined once at construction and live as long as the backend, so
/// what changes between chains has to be reachable through a cell rather than
/// captured by value.
#[derive(Default)]
struct Helpers {
    translator: Option<GuestTranslator>,
    accessor: Option<GuestAccessor>,
    /// Set when a helper could not complete an access.
    faulted: bool,
}

/// A compiled-block backend backed by the browser's WebAssembly engine.
pub struct BrowserBackend {
    /// `{ env: { memory, ...helpers } }`, built once and reused for every block.
    imports: Object,
    blocks: Vec<BlockRef>,
    helpers: Rc<RefCell<Helpers>>,
    /// Backing store for the JIT context header and its inline TLB.
    jit_ctx: Vec<u8>,
    /// Blocks a chaining dispatcher ran without returning here. Zero means chaining
    /// never engaged, which is indistinguishable from "chaining did not help" in a
    /// timing result — so it is counted rather than inferred.
    wasm_chain_steps: u64,
    /// `CpuState` followed by the Tier-2 context, as the ABI lays it out.
    ///
    /// Compiled code reads the code-version table pointer and length from the twelve
    /// bytes *after* `CpuState`. Handing it a bare `&mut CpuState` leaves those
    /// reading whatever the allocator put there: a nonzero length sends the
    /// version bump on every RAM store to a wild address, which corrupts the heap
    /// rather than failing. Keeping the region here — allocated once, zeroed once —
    /// is what makes a store safe.
    cpu_region: Vec<u128>,
    ram_base: u64,
    tlb_salt: u64,
    /// Kept alive for as long as the imports object references them.
    _closures: Vec<Closure<dyn FnMut(i32, i64) -> i64>>,
    _closures_t: Vec<Closure<dyn FnMut(i32) -> i64>>,
    _closures_w: Vec<Closure<dyn FnMut(i32, i64, i64)>>,
    _translate: Option<Closure<dyn FnMut(i32, i32, i64, i32) -> i64>>,
    _exit_mmio: Option<Closure<dyn FnMut(i32, i64, i32, i32, i64, i64) -> i64>>,
}

fn tlb_salt_seed() -> u64 {
    // Any odd value works; the tag is `(vpn ^ salt) | 1`.
    0x9E37_79B9_7F4A_7C15
}

impl BrowserBackend {
    /// Build a backend whose blocks import this module's linear memory.
    ///
    /// Fails only if the runtime does not expose `WebAssembly`, which no browser that
    /// can run this code does.
    pub fn new() -> Result<Self, JsValue> {
        let helpers = Rc::new(RefCell::new(Helpers::default()));
        let mut jit_ctx = vec![0u8; JitContext::TOTAL_BYTE_SIZE];
        let cpu_region_bytes =
            (aero_jit_x86::jit_ctx::TIER2_CTX_OFFSET + aero_jit_x86::jit_ctx::TIER2_CTX_SIZE)
                as usize;
        // `u128` elements give the 16-byte alignment `CpuState` asks for.
        let cpu_region = vec![0u128; cpu_region_bytes.div_ceil(16)];
        let ctx = JitContext {
            ram_base: 0,
            tlb_salt: tlb_salt_seed(),
            retired: 0,
            ..Default::default()
        };
        ctx.write_header_to_mem(&mut jit_ctx, 0);

        let env = Object::new();
        Reflect::set(&env, &JsValue::from_str(IMPORT_MEMORY), &wasm_bindgen::memory())?;

        let mut closures: Vec<Closure<dyn FnMut(i32, i64) -> i64>> = Vec::new();
        let mut closures_w: Vec<Closure<dyn FnMut(i32, i64, i64)>> = Vec::new();
        let mut closures_t: Vec<Closure<dyn FnMut(i32) -> i64>> = Vec::new();

        // Reads. The inline TLB handles RAM itself, so these are the miss and
        // non-RAM paths.
        for (name, width) in [
            (IMPORT_MEM_READ_U8, 1u32),
            (IMPORT_MEM_READ_U16, 2),
            (IMPORT_MEM_READ_U32, 4),
            (IMPORT_MEM_READ_U64, 8),
        ] {
            let h = helpers.clone();
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32, addr: i64| -> i64 {
                guest_read(&h, addr as u64, width) as i64
            }) as Box<dyn FnMut(i32, i64) -> i64>);
            Reflect::set(&env, &JsValue::from_str(name), c.as_ref())?;
            closures.push(c);
        }

        for (name, width) in [
            (IMPORT_MEM_WRITE_U8, 1u32),
            (IMPORT_MEM_WRITE_U16, 2),
            (IMPORT_MEM_WRITE_U32, 4),
            (IMPORT_MEM_WRITE_U64, 8),
        ] {
            let h = helpers.clone();
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32, addr: i64, value: i64| {
                guest_write(&h, addr as u64, width, value as u64);
            }) as Box<dyn FnMut(i32, i64, i64)>);
            Reflect::set(&env, &JsValue::from_str(name), c.as_ref())?;
            closures_w.push(c);
        }

        // A block that faults hands control back; the tier re-runs the instruction
        // under the interpreter, which raises the fault properly.
        {
            let h = helpers.clone();
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32, _addr: i64| -> i64 {
                h.borrow_mut().faulted = true;
                JIT_EXIT_SENTINEL_I64
            }) as Box<dyn FnMut(i32, i64) -> i64>);
            Reflect::set(&env, &JsValue::from_str(IMPORT_PAGE_FAULT), c.as_ref())?;
            closures.push(c);
        }

        // A translated access that lands on something other than RAM. Natively the
        // backend rolls guest state back and lets the interpreter redo the access;
        // here the block has written `CpuState.rip` for the instruction it is on, so
        // reporting a fault and stopping the chain has the same effect.
        let exit_mmio = {
            let h = helpers.clone();
            Closure::wrap(Box::new(
                move |_cpu: i32, _addr: i64, _a: i32, _b: i32, _c: i64, _d: i64| -> i64 {
                    h.borrow_mut().faulted = true;
                    JIT_EXIT_SENTINEL_I64
                },
            ) as Box<dyn FnMut(i32, i64, i32, i32, i64, i64) -> i64>)
        };
        Reflect::set(
            &env,
            &JsValue::from_str(IMPORT_JIT_EXIT_MMIO),
            exit_mmio.as_ref(),
        )?;

        // An explicit bailout: the block asked to stop. `CpuState.rip` is already the
        // instruction to resume at, so this is a clean exit rather than a fault.
        {
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32, _addr: i64| -> i64 {
                JIT_EXIT_SENTINEL_I64
            }) as Box<dyn FnMut(i32, i64) -> i64>);
            Reflect::set(&env, &JsValue::from_str(IMPORT_JIT_EXIT), c.as_ref())?;
            closures.push(c);
        }

        // Compiled `RDTSC`. The browser backend has no machine `TimeSource` behind
        // it, so it reports the clock this harness does have: a monotonic count of
        // reads. Enough to keep a block that reads the counter runnable and
        // non-decreasing, which is what guest spin loops require.
        {
            let ticks = Rc::new(std::cell::Cell::new(0i64));
            let t = ticks.clone();
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32| -> i64 {
                t.set(t.get().wrapping_add(1));
                t.get()
            }) as Box<dyn FnMut(i32) -> i64>);
            Reflect::set(&env, &JsValue::from_str(IMPORT_RDTSC), c.as_ref())?;
            closures_t.push(c);
        }

        // Self-modifying-code guard. Nothing invalidates blocks in this harness yet,
        // so a constant version means the guard always passes.
        {
            let c = Closure::wrap(Box::new(move |_cpu_ptr: i32, _page: i64| -> i64 { 0 })
                as Box<dyn FnMut(i32, i64) -> i64>);
            Reflect::set(&env, &JsValue::from_str(IMPORT_CODE_PAGE_VERSION), c.as_ref())?;
            closures.push(c);
        }

        // The inline TLB's miss path: translate, fill the entry, return the packed
        // `{page | flags}` word the fast path expects.
        let translate = {
            let h = helpers.clone();
            Closure::wrap(Box::new(
                move |_cpu_ptr: i32, jit_ctx_ptr: i32, vaddr: i64, access: i32| -> i64 {
                    translate_and_fill(&h, jit_ctx_ptr, vaddr as u64, access)
                },
            ) as Box<dyn FnMut(i32, i32, i64, i32) -> i64>)
        };
        Reflect::set(&env, &JsValue::from_str(IMPORT_MMU_TRANSLATE), translate.as_ref())?;

        let imports = Object::new();
        Reflect::set(&imports, &JsValue::from_str(IMPORT_MODULE), &env)?;

        Ok(Self {
            imports,
            blocks: Vec::new(),
            helpers,
            jit_ctx,
            wasm_chain_steps: 0,
            cpu_region,
            ram_base: 0,
            tlb_salt: tlb_salt_seed(),
            _closures: closures,
            _closures_t: closures_t,
            _closures_w: closures_w,
            _translate: Some(translate),
            _exit_mmio: Some(exit_mmio),
        })
    }

    /// Address of the JIT context in this module's linear memory.
    pub fn jit_ctx_ptr(&self) -> i32 {
        self.jit_ctx.as_ptr() as i32
    }

    /// Whether a helper reported a fault during the last chain.
    pub fn take_fault(&mut self) -> bool {
        let mut h = self.helpers.borrow_mut();
        core::mem::replace(&mut h.faulted, false)
    }

    fn write_ctx_header(&mut self) {
        let ctx = JitContext {
            ram_base: self.ram_base,
            tlb_salt: self.tlb_salt,
            retired: 0,
            ..Default::default()
        };
        ctx.write_header_to_mem(&mut self.jit_ctx, 0);
    }

    /// Steps taken inside wasm by a chaining dispatcher, over this backend's life.
    pub fn wasm_chain_steps(&self) -> u64 {
        self.wasm_chain_steps
    }

    /// Blocks a chaining dispatcher ran without returning, and what they retired.
    fn chain_blocks(&self) -> u32 {
        let at = JitContext::CHAIN_BLOCKS_OFFSET as usize;
        u32::from_le_bytes(self.jit_ctx[at..at + 4].try_into().unwrap_or([0; 4]))
    }

    fn chain_retired(&self) -> u64 {
        let at = JitContext::CHAIN_RETIRED_OFFSET as usize;
        u64::from_le_bytes(self.jit_ctx[at..at + 8].try_into().unwrap_or([0; 8]))
    }

    /// Zero both before entering, so what comes back belongs to this call alone.
    fn reset_chain_counters(&mut self) {
        let r = JitContext::CHAIN_RETIRED_OFFSET as usize;
        self.jit_ctx[r..r + 8].fill(0);
        let b = JitContext::CHAIN_BLOCKS_OFFSET as usize;
        self.jit_ctx[b..b + 4].fill(0);
    }

    fn instantiate(&self, wasm: &[u8], count: usize) -> Result<Vec<BlockRef>, JsValue> {
        // `new WebAssembly.Module` compiles synchronously, which the main thread
        // refuses for anything over 4KiB. Aero runs its CPU in a worker, where it is
        // allowed and is what lets a block be admitted without an await in the
        // middle of the run loop.
        let bytes = Uint8Array::from(wasm);
        let module = WebAssembly::Module::new(&bytes.into())?;
        let instance = WebAssembly::Instance::new(&module, &self.imports)?;
        let exports = instance.exports();
        let dispatch =
            Reflect::get(&exports, &JsValue::from_str(EXPORT_TIER1_DISPATCH_FN))?
                .dyn_into::<Function>()?;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(BlockRef {
                dispatch: dispatch.clone(),
                index: JsValue::from_f64(i as f64),
            });
        }
        Ok(out)
    }
}

/// Read `width` bytes at `addr`, through the accessor when there is one.
fn guest_read(h: &Rc<RefCell<Helpers>>, addr: u64, width: u32) -> u64 {
    let taken = h.borrow_mut().accessor.take();
    let Some(mut access) = taken else {
        return 0;
    };
    let out = access(addr, width, None);
    let mut b = h.borrow_mut();
    b.accessor = Some(access);
    match out {
        Some(v) => v,
        None => {
            b.faulted = true;
            0
        }
    }
}

fn guest_write(h: &Rc<RefCell<Helpers>>, addr: u64, width: u32, value: u64) {
    let taken = h.borrow_mut().accessor.take();
    let Some(mut access) = taken else {
        return;
    };
    let out = access(addr, width, Some(value));
    let mut b = h.borrow_mut();
    b.accessor = Some(access);
    if out.is_none() {
        b.faulted = true;
    }
}

/// Translate `vaddr`, fill the inline TLB entry, and return the packed word.
///
/// The fast path recomputes the same index and tag, so filling the entry here is what
/// makes the next access to the page skip this call entirely.
fn translate_and_fill(h: &Rc<RefCell<Helpers>>, jit_ctx_ptr: i32, vaddr: u64, access: i32) -> i64 {
    let taken = h.borrow_mut().translator.take();
    let Some(mut translate) = taken else {
        return 0;
    };
    let out = translate(vaddr, access as u32);
    let mut b = h.borrow_mut();
    b.translator = Some(translate);
    let Some((page, flags)) = out else {
        b.faulted = true;
        return 0;
    };
    drop(b);

    // Safety: `jit_ctx_ptr` is the address of a live `Vec<u8>` of at least
    // `JitContext::TOTAL_BYTE_SIZE` bytes, owned by the backend that installed this
    // helper, and the only writer while a block is running.
    unsafe {
        let base = jit_ctx_ptr as usize as *mut u8;
        let salt = core::ptr::read_unaligned(
            base.add(JitContext::TLB_SALT_OFFSET as usize) as *const u64
        );
        let vpn = vaddr >> 12;
        let index = (vpn & aero_jit_x86::JIT_TLB_INDEX_MASK) as usize;
        let entry = base
            .add(JitContext::TLB_OFFSET as usize)
            .add(index * aero_jit_x86::JIT_TLB_ENTRY_SIZE as usize);
        core::ptr::write_unaligned(entry as *mut u64, (vpn ^ salt) | 1);
        core::ptr::write_unaligned(entry.add(8) as *mut u64, page | flags);
    }
    (page | flags) as i64
}

impl CompiledBackend for BrowserBackend {
    fn add_block(&mut self, wasm: &[u8]) -> u32 {
        self.add_blocks(wasm, 1)[0]
    }

    fn add_blocks(&mut self, wasm: &[u8], count: usize) -> Vec<u32> {
        let funcs = self
            .instantiate(wasm, count)
            .expect("instantiate a compiled Tier-1 module");
        let mut out = Vec::with_capacity(count);
        for f in funcs {
            out.push(self.blocks.len() as u32);
            self.blocks.push(f);
        }
        out
    }

    /// The browser can chain inside wasm, and this is the one place it matters most.
    ///
    /// Nothing is ever promoted here — there is no Cranelift in a browser — so without
    /// this every block execution crosses the JS boundary, converts an `i64` result
    /// through a `BigInt`, and runs a lookup closure. Measured on a five-instruction
    /// loop, that is what separates the browser's 20.2 MIPS from the native tier's 86.0
    /// chained; native's *unchained* path measures 27.4, which is where the browser was
    /// sitting. The dispatcher that removes it is already emitted by the shared codegen
    /// and exported under the same name, so all this backend has to do is maintain the
    /// chain map and the limits it reads.
    fn can_chain_in_wasm(&self) -> bool {
        true
    }

    fn chain_map_insert(&mut self, rip: u64, module_id: u32, index: u32) {
        let slot = JitContext::chain_map_slot(rip);
        let stride = aero_jit_x86::jit_ctx::JIT_CHAIN_MAP_ENTRY_SIZE as usize;
        let at = JitContext::CHAIN_MAP_OFFSET as usize + slot * stride;
        // The RIP goes in last: the dispatcher matches on it, so a torn entry read
        // between these writes must fail to match rather than dispatch on half of one.
        self.jit_ctx[at + 8..at + 12].copy_from_slice(&module_id.to_le_bytes());
        self.jit_ctx[at + 12..at + 16].copy_from_slice(&index.to_le_bytes());
        self.jit_ctx[at..at + 8].copy_from_slice(&rip.to_le_bytes());
    }

    fn chain_map_forget(&mut self, rip: u64) {
        let slot = JitContext::chain_map_slot(rip);
        let stride = aero_jit_x86::jit_ctx::JIT_CHAIN_MAP_ENTRY_SIZE as usize;
        let at = JitContext::CHAIN_MAP_OFFSET as usize + slot * stride;
        // Only if it is still this RIP's: the slot is direct-mapped and may since have
        // been taken by another block, which must not be evicted by this call.
        if self.jit_ctx[at..at + 8] == rip.to_le_bytes() {
            self.jit_ctx[at..at + 8].fill(0);
        }
    }

    fn set_chain_limits(&mut self, retired_budget: u64, max_blocks: u32) {
        let b = JitContext::CHAIN_BUDGET_OFFSET as usize;
        self.jit_ctx[b..b + 8].copy_from_slice(&retired_budget.to_le_bytes());
        let m = JitContext::CHAIN_MAX_BLOCKS_OFFSET as usize;
        self.jit_ctx[m..m + 4].copy_from_slice(&max_blocks.to_le_bytes());
    }

    fn last_retired(&self) -> u64 {
        // Safety: `jit_ctx` is a live `Vec<u8>` of at least `TOTAL_BYTE_SIZE` bytes.
        unsafe {
            core::ptr::read_unaligned(
                self.jit_ctx
                    .as_ptr()
                    .add(JitContext::RETIRED_OFFSET as usize) as *const u64,
            )
        }
    }

    fn set_ram_base(&mut self, ram_base: u64) {
        self.ram_base = ram_base;
        self.write_ctx_header();
    }

    fn flush_tlb(&mut self) {
        // Change the salt rather than clearing 4KiB of entries: every cached tag
        // stops matching at once.
        self.tlb_salt = self.tlb_salt.wrapping_add(2);
        self.write_ctx_header();
    }

    fn set_translator(&mut self, translator: Option<GuestTranslator>) -> Option<GuestTranslator> {
        core::mem::replace(&mut self.helpers.borrow_mut().translator, translator)
    }

    fn set_accessor(&mut self, accessor: Option<GuestAccessor>) -> Option<GuestAccessor> {
        core::mem::replace(&mut self.helpers.borrow_mut().accessor, accessor)
    }

    fn run_chain(
        &mut self,
        entry_index: u32,
        cpu: &mut CpuState,
        lookup: &mut dyn FnMut(u64, u64) -> Option<u32>,
        max_blocks: u32,
    ) -> aero_jit_x86::compiled_backend::ChainOutcome {
        // Copy `CpuState` into the ABI's region, run, and copy it back — the same
        // shape the native backend uses. The Tier-2 context that follows stays zero,
        // which is what tells compiled stores there is no code-version table.
        let region = self.cpu_region.as_mut_ptr() as *mut u8;
        // Safety: `cpu_region` is at least `CPU_STATE_SIZE` bytes and 16-byte
        // aligned, and `CpuState` is plain data with no interior pointers.
        unsafe {
            core::ptr::copy_nonoverlapping(
                cpu as *const CpuState as *const u8,
                region,
                core::mem::size_of::<CpuState>(),
            );
        }
        let cpu_ptr = JsValue::from(region as i32);
        let ctx_ptr = JsValue::from(self.jit_ctx_ptr());

        let mut index = entry_index;
        let mut executed = 0u32;
        let mut retired = 0u64;
        let mut next_rip = cpu.rip;
        let mut exit_to_interpreter = false;

        while executed < max_blocks {
            // Cloned rather than borrowed: resetting the chain counters needs `&mut
            // self` while a borrow of `self.blocks` is live. `BlockRef` is two
            // `JsValue`s, so this is a pair of refcount bumps — and with a chaining
            // dispatcher this loop runs once per *chain* rather than once per block,
            // so it is paid once for the thousands of blocks the dispatcher then runs.
            let Some(func) = self.blocks.get(index as usize).cloned() else {
                break;
            };
            self.reset_chain_counters();
            let ret = match func
                .dispatch
                .call3(&JsValue::NULL, &cpu_ptr, &ctx_ptr, &func.index)
            {
                Ok(v) => v,
                Err(_) => {
                    exit_to_interpreter = true;
                    break;
                }
            };
            // A chaining dispatcher runs several blocks before it comes back and folds
            // each one's count into the chain totals as it goes; a plain one runs a
            // single block that wrote its own. Reading both and picking by the block
            // count is how the native backend tells them apart, and adding `retired` in
            // the chained case would count the last block twice.
            let inner = self.chain_blocks();
            if inner == 0 {
                executed += 1;
                retired += self.last_retired();
            } else {
                executed += inner;
                self.wasm_chain_steps += u64::from(inner - 1);
                retired += self.chain_retired();
            }

            // An `i64` result crosses into JS as a BigInt.
            let ret = js_sys::BigInt::try_from(ret)
                .ok()
                .and_then(|b| i64::try_from(b).ok())
                .unwrap_or(JIT_EXIT_SENTINEL_I64);

            if ret == JIT_EXIT_SENTINEL_I64 {
                // A helper that faulted means the block stopped part-way and its
                // instructions must not be counted. A terminator exit means it ran
                // everything and `CpuState.rip` is the next instruction, so the chain
                // can be counted and continued through it.
                if self.take_fault() {
                    exit_to_interpreter = true;
                    break;
                }
                // The block wrote the next RIP into the region, not into `cpu`.
                // Safety: `region` holds a live `CpuState` image.
                next_rip = unsafe {
                    core::ptr::read_unaligned(
                        region.add(aero_jit_x86::abi::CPU_RIP_OFF as usize) as *const u64,
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

        // Safety: as above, in the other direction.
        unsafe {
            core::ptr::copy_nonoverlapping(
                region as *const u8,
                cpu as *mut CpuState as *mut u8,
                core::mem::size_of::<CpuState>(),
            );
        }

        if exit_to_interpreter {
            next_rip = cpu.rip;
        }
        aero_jit_x86::compiled_backend::ChainOutcome {
            next_rip,
            retired,
            blocks: executed,
            exit_to_interpreter,
        }
    }
}

/// `CpuState` and guest RAM both live in this module's memory, and `abi` fixes where
/// the fields sit inside `CpuState`. Assert the pieces line up rather than trusting it.
const _: () = {
    assert!(aero_jit_x86::abi::CPU_STATE_SIZE as usize >= core::mem::size_of::<CpuState>());
};

/// Run the compiled tier in a loop, for measuring throughput rather than correctness.
///
/// [`tier1_run_block`] rebuilds everything on each call — a megabyte of guest RAM, a
/// backend, the tier, and the block itself — and one call retires at most a chain's
/// worth of instructions. Timing it therefore measures setup, not execution: at 8,192
/// instructions against a few milliseconds it reports single-digit MIPS no matter how
/// fast the engine is. This sets all of that up once and then runs `iters` chains, so
/// the fixed cost falls away and what is left is the tier executing guest code.
///
/// The guest program must have its loop head *at* `entry_rip`, or the chain leaves the
/// one compiled block after a single pass. Returns the total retired across all chains
/// so the caller can check the work actually happened rather than trusting the rate.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn tier1_bench_block(
    code: &[u8],
    entry_rip: u32,
    rax_in: f64,
    iters: u32,
    chain: bool,
) -> Result<JsValue, JsValue> {
    use aero_cpu_core::mem::FlatTestBus;
    use aero_cpu_core::state::CpuMode;
    use aero_cpu_core::CpuCore;
    use aero_machine::compiled_tier::CompiledTier;
    use aero_types::Gpr;

    install_panic_reporter();

    const RAM_BYTES: usize = 1 << 20;
    let entry = u64::from(entry_rip);
    if code.is_empty() || entry as usize + code.len() > RAM_BYTES {
        return Err(js_sys::Error::new("code does not fit in the test RAM").into());
    }

    let mut bus = FlatTestBus::new(RAM_BYTES);
    bus.load(entry, code);
    let ram_base = bus.base_ptr() as u64;
    let backend = Rc::new(RefCell::new(BrowserBackend::new()?));
    let backend_for_stats = backend.clone();
    let mut tier = CompiledTier::new_with_ram_base(backend, ram_base);
    if !chain {
        tier.disable_wasm_chaining();
    }
    // Deliberately *not* admitting the entry block here. Doing so flushes it on its own,
    // and then the two-pass warm-up below has only the remaining blocks left to queue —
    // which can leave a single block pending, below the two a chaining dispatcher needs.
    // The whole loop has to go in as one batch.

    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.rip = entry;
    cpu.state.gpr[Gpr::Rax.as_u8() as usize] = rax_in as u64;
    cpu.state.gpr[Gpr::Rsp.as_u8() as usize] = (RAM_BYTES as u64) - 0x100;
    let mut assist = aero_cpu_core::assist::AssistContext::default();

    // Admit the whole loop, as *one module*, before timing anything.
    //
    // Two passes, and the second is the point. Only the entry block is compiled above,
    // and this harness has no interpreter seam to admit the rest as the guest reaches
    // them, so a loop spanning several blocks runs the first one, stops at a RIP nothing
    // is compiled for, and returns — which reads as a complete run rather than a failure.
    //
    // Compiling at each stop as it happens fixes that but admits one block per module,
    // and a chaining dispatcher is only emitted for a module holding more than one. The
    // result is a run that reports plausible throughput while never chaining once. So the
    // first pass only *discovers* which RIPs the loop visits, using a throwaway tier, and
    // the second admits them together so they share a module and can chain.
    let mut rips: Vec<u64> = Vec::new();
    {
        let scout_backend = Rc::new(RefCell::new(BrowserBackend::new()?));
        let mut scout = CompiledTier::new_with_ram_base(scout_backend, ram_base);
        let mut c = CpuCore::new(CpuMode::Long);
        c.state.rip = entry;
        c.state.gpr[Gpr::Rax.as_u8() as usize] = rax_in as u64;
        c.state.gpr[Gpr::Rsp.as_u8() as usize] = (RAM_BYTES as u64) - 0x100;
        let mut a = aero_cpu_core::assist::AssistContext::default();
        for _ in 0..64 {
            let at = c.state.rip;
            if rips.contains(&at) {
                break;
            }
            rips.push(at);
            if !scout.compile(at, 64, &mut bus) {
                scout.flush_for_test();
            }
            if scout.run(&mut c, &mut bus, &mut a, 1_000_000) == 0 {
                break;
            }
        }
    }
    for &rip in &rips {
        tier.compile(rip, 64, &mut bus);
    }
    tier.flush_for_test();
    let compiled = tier.compiled_blocks();
    // Put the counter back: the warm-up spent some of it.
    cpu.state.gpr[Gpr::Rax.as_u8() as usize] = rax_in as u64;

    let mut retired: u64 = 0;
    for _ in 0..iters {
        retired += tier.run(&mut cpu, &mut bus, &mut assist, 1_000_000) as u64;
    }

    let out = Object::new();
    Reflect::set(&out, &JsValue::from_str("retired"), &JsValue::from_f64(retired as f64))?;
    Reflect::set(
        &out,
        &JsValue::from_str("compiledBlocks"),
        &JsValue::from_f64(compiled as f64),
    )?;
    Reflect::set(&out, &JsValue::from_str("iters"), &JsValue::from_f64(f64::from(iters)))?;
    Reflect::set(&out, &JsValue::from_str("chain"), &JsValue::from_bool(chain))?;
    Reflect::set(
        &out,
        &JsValue::from_str("wasmChainSteps"),
        &JsValue::from_f64(backend_for_stats.borrow().wasm_chain_steps() as f64),
    )?;
    Ok(out.into())
}

/// Run a block of guest code through the compiled tier, in the browser.
///
/// This is the thing the tier exists to do, reduced to something a test can assert
/// on: hand it x86 bytes and a starting RAX, and it discovers the block, translates
/// it, eliminates dead flag writes, compiles it to wasm, hands that to the browser's
/// own engine, and runs it. The value coming back is what the guest left in RAX.
///
/// `expected_blocks` is what the caller believes should have been compiled; the
/// result reports the real count, so a test can tell "the tier ran it" apart from
/// "the interpreter ran it and the tier did nothing".
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn tier1_run_block(code: &[u8], entry_rip: u32, rax_in: f64) -> Result<JsValue, JsValue> {
    use aero_cpu_core::mem::FlatTestBus;
    use aero_cpu_core::state::CpuMode;
    use aero_cpu_core::CpuCore;
    use aero_machine::compiled_tier::CompiledTier;
    use aero_types::Gpr;

    // Without a hook a Rust panic reaches JS as a bare `unreachable`, which says
    // nothing about what went wrong inside the tier.
    install_panic_reporter();

    const RAM_BYTES: usize = 1 << 20;
    let entry = u64::from(entry_rip);
    if code.is_empty() || entry as usize + code.len() > RAM_BYTES {
        return Err(js_sys::Error::new("code does not fit in the test RAM").into());
    }

    let mut bus = FlatTestBus::new(RAM_BYTES);
    bus.load(entry, code);
    // Guest RAM is this module's memory, so `ram_base` is simply where that buffer
    // happens to live. Natively it is a region carved out of the engine's memory
    // instead; nothing above this line knows the difference.
    let ram_base = bus.base_ptr() as u64;

    let backend = Rc::new(RefCell::new(BrowserBackend::new()?));
    let mut tier = CompiledTier::new_with_ram_base(backend, ram_base);

    if !tier.compile(entry, 64, &mut bus) {
        // Admission is batched, so the first call queues the block; the tier picks it
        // up on the next hit. Ask again rather than reaching into its internals.
        tier.flush_for_test();
    }
    let compiled = tier.compiled_blocks();

    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state.rip = entry;
    cpu.state.gpr[Gpr::Rax.as_u8() as usize] = rax_in as u64;
    // A stack inside the test RAM, well clear of the code.
    cpu.state.gpr[Gpr::Rsp.as_u8() as usize] = (RAM_BYTES as u64) - 0x100;

    // The tier can step an instruction through Tier-0 to get past one Tier-1 refused,
    // which needs the interpreter's assist context. This harness has no machine to
    // supply one, so it uses a default — the smoke block never reaches an assist.
    let mut assist = aero_cpu_core::assist::AssistContext::default();
    let retired = tier.run(&mut cpu, &mut bus, &mut assist, 1_000_000);

    let out = Object::new();
    Reflect::set(&out, &JsValue::from_str("rax"), &JsValue::from_f64(
        cpu.state.gpr[Gpr::Rax.as_u8() as usize] as f64,
    ))?;
    Reflect::set(&out, &JsValue::from_str("rcx"), &JsValue::from_f64(
        cpu.state.gpr[Gpr::Rcx.as_u8() as usize] as f64,
    ))?;
    Reflect::set(&out, &JsValue::from_str("rip"), &JsValue::from_f64(cpu.state.rip as f64))?;
    Reflect::set(&out, &JsValue::from_str("retired"), &JsValue::from_f64(retired as f64))?;
    Reflect::set(
        &out,
        &JsValue::from_str("compiledBlocks"),
        &JsValue::from_f64(compiled as f64),
    )?;
    Ok(out.into())
}

/// Route Rust panics to `console.error` so a browser run says what failed.
#[cfg(target_arch = "wasm32")]
fn install_panic_reporter() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("aero tier panic: {info}");
            web_log(&msg);
        }));
    });
}

#[cfg(target_arch = "wasm32")]
fn web_log(msg: &str) {
    // `console.error` without pulling in `web-sys` for one call.
    let console = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("console"))
        .unwrap_or(JsValue::UNDEFINED);
    if let Ok(f) = js_sys::Reflect::get(&console, &JsValue::from_str("error")) {
        if let Ok(f) = f.dyn_into::<Function>() {
            let _ = f.call1(&console, &JsValue::from_str(msg));
        }
    }
}

/// Compile a block through the tier without running it.
///
/// Splits "the browser instantiated the module" from "the browser ran it", so a
/// failure says which.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn tier1_compile_only(code: &[u8], entry_rip: u32) -> Result<JsValue, JsValue> {
    use aero_cpu_core::mem::FlatTestBus;
    use aero_machine::compiled_tier::CompiledTier;

    install_panic_reporter();
    const RAM_BYTES: usize = 1 << 20;
    let entry = u64::from(entry_rip);
    let mut bus = FlatTestBus::new(RAM_BYTES);
    bus.load(entry, code);
    let ram_base = bus.base_ptr() as u64;

    let backend = Rc::new(RefCell::new(BrowserBackend::new()?));
    let mut tier = CompiledTier::new_with_ram_base(backend, ram_base);
    tier.compile(entry, 64, &mut bus);
    tier.flush_for_test();

    let out = Object::new();
    Reflect::set(
        &out,
        &JsValue::from_str("compiledBlocks"),
        &JsValue::from_f64(tier.compiled_blocks() as f64),
    )?;
    Reflect::set(
        &out,
        &JsValue::from_str("hasBlock"),
        &JsValue::from_bool(tier.has_block(entry)),
    )?;
    Reflect::set(
        &out,
        &JsValue::from_str("ramBase"),
        &JsValue::from_f64(ram_base as f64),
    )?;
    Ok(out.into())
}

//! IR block cache for the machine's execute loop.
//!
//! Hot guest blocks are discovered once, translated to `aero-jit-x86`'s IR, and cached by entry
//! address, so a loop body is decoded once rather than on every iteration. Execution stays in the
//! IR interpreter: this path needs no WASM engine, which is what lets it run everywhere the
//! machine does.
//!
//! Blocks are admitted only after they have been entered [`IrBlockCache::hot_threshold`] times,
//! because translating a block that runs once costs more than interpreting it.

use std::cell::Cell;

use aero_cpu_core::mem::CpuBus;
use aero_cpu_core::state::CpuState;
use aero_jit_x86::tier1::ir::interp::{
    execute_block_cpu_state, execute_block_cpu_with_scratch, ExecResult, TestCpu,
};
use aero_jit_x86::tier1::ir::{IrBlock, IrInst};
use aero_jit_x86::tier1::{discover_block, translate_block, BlockLimits};
use aero_jit_x86::Tier1Bus as JitBus;
use aero_jit_x86::Width;

// ── IR interpreter fast-path (no WASM engine required) ───────────────────────

/// Read a tuning knob from the environment, falling back to `default`.
///
/// These exist so the tiering thresholds can be swept against a real workload
/// without a rebuild; the defaults are what ships.
#[cfg(not(target_arch = "wasm32"))]
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

#[cfg(target_arch = "wasm32")]
fn env_u32(_name: &str, default: u32) -> u32 {
    default
}

#[cfg(not(target_arch = "wasm32"))]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

#[cfg(target_arch = "wasm32")]
fn env_usize(_name: &str, default: usize) -> usize {
    default
}

pub use crate::rip_map::{RipBuildHasher, RipHasher, RipMap, RipSet};

/// Longest chain of cached blocks a single entry may run.
///
/// Read once: this sits on the chain's hot path and must not touch the environment
/// per call.
fn max_chain_blocks() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *N.get_or_init(|| u64::from(env_u32("AERO_JIT_MAX_CHAIN", 4096)))
    }
    #[cfg(target_arch = "wasm32")]
    {
        4096
    }
}

/// IR block cache entry: stores a pre-decoded + translated IR block ready for
/// execution by the IR interpreter.
pub struct IrBlockCache {
    /// Maps entry RIP → compiled IR block.
    blocks: RipMap<IrBlock>,
    /// Number of guest x86 instructions represented by each compiled block.
    ///
    /// `IrBlock::insts.len()` is the number of lowered IR operations, not the
    /// architectural retired-instruction count.
    guest_instruction_counts: RipMap<u64>,
    /// Hotness counters per RIP.
    hotness: RipMap<u32>,
    /// RIPs currently being compiled (prevent recompilation).
    compiling: RipSet,
    /// RIPs that failed admission (helpers / empty blocks). Remember
    /// them so a hot rejected loop does not rediscover and translate
    /// every `hot_threshold` hits — that tax collapsed a live WMI
    /// slice from ~4.5 to ~1.2 Minst/s.
    rejected: RipSet,
    /// Block discovery limits.
    limits: BlockLimits,
    /// Hotness threshold: a block must be hit this many times before compilation.
    hot_threshold: u32,
    /// Threshold for a RIP found by a chain running off the end of the cache.
    chain_miss_threshold: u32,
    /// Maximum number of cached blocks (LRU eviction when exceeded).
    max_blocks: usize,
    /// Cap on live hotness counters before the whole map is dropped.
    max_hotness_entries: usize,
    /// Cached blocks that touch no guest memory.
    ///
    /// These are the ones a compiled tier can run without any address-translation
    /// machinery: no inline TLB, no MMIO exit, nothing to roll back. Counted to size
    /// that opportunity before building for it.
    memory_free_blocks: u64,
    /// Guest instructions executed from memory-free cached blocks.
    memory_free_insts: u64,
    /// Diagnostic counters for proving that the cached path actually ran.
    executed_blocks: u64,
    executed_guest_instructions: u64,
}

impl Default for IrBlockCache {
    fn default() -> Self {
        Self::new()
    }
}

impl IrBlockCache {
    pub fn new() -> Self {
        Self {
            blocks: RipMap::default(),
            guest_instruction_counts: RipMap::default(),
            hotness: RipMap::default(),
            compiling: RipSet::default(),
            rejected: RipSet::default(),
            limits: BlockLimits::default(),
            hot_threshold: env_u32("AERO_JIT_HOT_THRESHOLD", 50),
            chain_miss_threshold: env_u32("AERO_JIT_CHAIN_THRESHOLD", 2),
            max_blocks: env_usize("AERO_JIT_MAX_BLOCKS", 4096),
            max_hotness_entries: env_usize("AERO_JIT_MAX_HOTNESS", 65_536),
            memory_free_blocks: 0,
            memory_free_insts: 0,
            executed_blocks: 0,
            executed_guest_instructions: 0,
        }
    }

    /// Record a hit at `rip` and return `true` if the block should be compiled.
    pub fn record_hit(&mut self, rip: u64) -> bool {
        self.record_hit_with_threshold(rip, self.hot_threshold)
    }

    /// Record a hit at a RIP reached by a chain running off the end of the cache.
    ///
    /// A much stronger signal than the batch-boundary sample [`record_hit`] gets:
    /// we only arrived here by executing cached blocks, so `rip` is on a path
    /// already known to be hot, and compiling it extends that chain rather than
    /// starting a new one. Hence the lower threshold.
    ///
    /// [`record_hit`]: Self::record_hit
    pub fn record_chain_miss(&mut self, rip: u64) -> bool {
        self.record_hit_with_threshold(rip, self.chain_miss_threshold)
    }

    fn record_hit_with_threshold(&mut self, rip: u64, threshold: u32) -> bool {
        if self.rejected.contains(&rip) {
            return false;
        }
        // Checked before the bump so a cached block cannot accumulate hotness it
        // can never spend.
        if self.blocks.contains_key(&rip) || self.compiling.contains(&rip) {
            return false;
        }
        // A long-running guest walks through far more distinct entries than it ever
        // runs hot, and counters for the cold ones are pure growth. Drop them
        // wholesale when there are too many: anything genuinely hot re-arms within
        // a few hits, and the alternative is a map that only grows.
        if self.hotness.len() >= self.max_hotness_entries {
            self.hotness.clear();
        }
        let count = self.hotness.entry(rip).or_insert(0);
        *count += 1;
        *count >= threshold
    }

    /// Check if a compiled block exists for `rip`.
    pub fn get(&self, rip: u64) -> Option<&IrBlock> {
        self.blocks.get(&rip)
    }

    /// Compile and cache a block. Returns true on success.
    pub fn compile<B: CpuBus>(&mut self, rip: u64, bitness: u32, bus: &mut B) -> bool {
        if self.blocks.len() >= self.max_blocks {
            // Evict a random block (simple LRU approximation).
            if let Some(&evict_rip) = self.blocks.keys().next() {
                self.blocks.remove(&evict_rip);
                self.guest_instruction_counts.remove(&evict_rip);
                self.hotness.remove(&evict_rip);
            }
        }

        self.compiling.insert(rip);
        let result = self.compile_inner(rip, bitness, bus);
        self.compiling.remove(&rip);

        if let Some((ir, guest_instruction_count)) = result {
            if !ir.insts.iter().any(|i| {
                matches!(i, IrInst::Load { .. }) || matches!(i, IrInst::Store { .. })
            }) {
                self.memory_free_blocks += 1;
                self.memory_free_insts += guest_instruction_count;
            }
            self.blocks.insert(rip, ir);
            self.guest_instruction_counts
                .insert(rip, guest_instruction_count);
            true
        } else {
            self.rejected.insert(rip);
            self.hotness.remove(&rip);
            false
        }
    }

    fn compile_inner<B: CpuBus>(
        &self,
        rip: u64,
        bitness: u32,
        bus: &mut B,
    ) -> Option<(IrBlock, u64)> {
        // Create a Tier1Bus adapter to read code bytes for block discovery.
        // Uses a raw pointer because Tier1Bus::read_u8 takes &self while CpuBus::read_u8 takes &mut self.
        struct ReadAdapter<B: CpuBus> {
            ptr: *mut B,
        }
        impl<B: CpuBus> JitBus for ReadAdapter<B> {
            fn read_u8(&self, addr: u64) -> u8 {
                unsafe { (*self.ptr).read_u8(addr) }.unwrap_or(0xFF)
            }
            fn write_u8(&mut self, _: u64, _: u8) {}
        }

        let adapter = ReadAdapter { ptr: bus as *mut B };
        let block = match bitness {
            64 => discover_block(&adapter, rip, self.limits),
            32 | 16 => aero_jit_x86::discover_block_mode(&adapter, rip, self.limits, bitness),
            _ => return None,
        };

        // Check that the block didn't hit a decode error or limit.
        if block.insts.is_empty() {
            return None;
        }

        let guest_instruction_count = block.insts.len() as u64;
        let ir = translate_block(&block);
        if !admit_ir_block(rip, &ir) {
            return None;
        }
        Some((ir, guest_instruction_count))
    }

    /// Architectural x86 instructions represented by the cached block at `rip`.
    pub fn guest_instruction_count(&self, rip: u64) -> Option<u64> {
        self.guest_instruction_counts.get(&rip).copied()
    }

    pub fn record_execution(&mut self, guest_instruction_count: u64) {
        self.record_chain_execution(1, guest_instruction_count);
    }

    pub fn record_chain_execution(&mut self, blocks: u64, guest_instruction_count: u64) {
        self.executed_blocks = self.executed_blocks.saturating_add(blocks);
        self.executed_guest_instructions = self
            .executed_guest_instructions
            .saturating_add(guest_instruction_count);
    }

    pub fn memory_free_blocks(&self) -> u64 {
        self.memory_free_blocks
    }

    pub fn memory_free_insts(&self) -> u64 {
        self.memory_free_insts
    }

    pub fn executed_blocks(&self) -> u64 {
        self.executed_blocks
    }

    pub fn executed_guest_instructions(&self) -> u64 {
        self.executed_guest_instructions
    }

    /// Invalidate a cached block (e.g., due to self-modifying code).
    pub fn invalidate(&mut self, rip: u64) {
        self.blocks.remove(&rip);
        self.guest_instruction_counts.remove(&rip);
    }

    /// Clear all cached blocks.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.guest_instruction_counts.clear();
        self.hotness.clear();
        self.rejected.clear();
        self.compiling.clear();
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Admit a translated IR block for the live `--jit` path.
///
/// After VDS is RUNNING the hot CIM walk is `repdrvfs` / FastProx (user)
/// plus ntoskrnl's pool-tag / name hash at `nt!+0x2b5c00` (kernel). Those
/// loads and stores are RAM. A later #PF restores CPU and retries the
/// block head; plain `Store` IR is a register-to-memory MOV of the same
/// SSA value, so re-issuing several stores is idempotent. Helpers stay
/// fail-closed: MMIO and string/SSE assists must not replay.
fn admit_ir_block(_rip: u64, ir: &IrBlock) -> bool {
    !ir.insts
        .iter()
        .any(|inst| matches!(inst, IrInst::CallHelper { .. }))
}

/// Execute a cached IR block. Returns `Some(next_rip)` on success, or `None`
/// if the block should fall back to the interpreter.
pub fn run_ir_block<B: CpuBus>(
    block: &IrBlock,
    cpu_state: &mut CpuState,
    bus: &mut B,
) -> Option<u64> {
    struct BusAdapter<B: CpuBus> {
        ptr: *mut B,
        faulted: Cell<bool>,
    }
    impl<B: CpuBus> JitBus for BusAdapter<B> {
        fn read_u8(&self, addr: u64) -> u8 {
            self.read(addr, Width::W8) as u8
        }
        fn read(&self, addr: u64, width: Width) -> u64 {
            if self.faulted.get() {
                return 0;
            }
            let result = unsafe {
                match width {
                    Width::W8 => (*self.ptr).read_u8(addr).map(u64::from),
                    Width::W16 => (*self.ptr).read_u16(addr).map(u64::from),
                    Width::W32 => (*self.ptr).read_u32(addr).map(u64::from),
                    Width::W64 => (*self.ptr).read_u64(addr),
                }
            };
            match result {
                Ok(value) => value,
                Err(_) => {
                    self.faulted.set(true);
                    0
                }
            }
        }
        fn write_u8(&mut self, addr: u64, val: u8) {
            self.write(addr, Width::W8, val as u64);
        }
        fn write(&mut self, addr: u64, width: Width, value: u64) {
            if self.faulted.get() {
                return;
            }
            let len = match width {
                Width::W8 => 1,
                Width::W16 => 2,
                Width::W32 => 4,
                Width::W64 => 8,
            };
            let result = unsafe {
                (*self.ptr)
                    .preflight_write_bytes(addr, len)
                    .and_then(|()| match width {
                        Width::W8 => (*self.ptr).write_u8(addr, value as u8),
                        Width::W16 => (*self.ptr).write_u16(addr, value as u16),
                        Width::W32 => (*self.ptr).write_u32(addr, value as u32),
                        Width::W64 => (*self.ptr).write_u64(addr, value),
                    })
            };
            if result.is_err() {
                self.faulted.set(true);
            }
        }
    }

    let cpu_before = cpu_state.clone();
    let mut adapter = BusAdapter {
        ptr: bus as *mut B,
        faulted: Cell::new(false),
    };
    let result = execute_block_cpu_state(block, cpu_state, &mut adapter);
    if adapter.faulted.get() {
        // At most one trailing store, preflighted before commit. Restoring
        // CPU state undoes register updates so Tier-0 can retry the head.
        *cpu_state = cpu_before;
        return None;
    }
    match result {
        ExecResult::Continue => {
            // The IR interpreter updated cpu_state.rip; read it back.
            Some(cpu_state.rip())
        }
        ExecResult::ExitToInterpreter { next_rip } => {
            // Block hit an unsupported instruction or MMIO; fall back.
            cpu_state.set_rip(next_rip);
            None
        }
    }
}

/// Run cached IR blocks in a chain until the next RIP is uncached, a fault
/// occurs, or `max_guest_insts` would be exceeded.
///
/// The wmiprvse merge loop is three 2–5 instruction blocks split by `Jcc`.
/// Copying [`CpuState`] through [`TestCpu`] on every block made that loop
/// *slower* than Tier-0. One conversion plus a RIP lookup per block is the
/// path that actually keeps the loop in the cache.
///
/// Returns `(next_rip, guest_instructions, blocks_executed)`.
pub fn run_ir_chain<B: CpuBus>(
    cache: &IrBlockCache,
    cpu_state: &mut CpuState,
    bus: &mut B,
    max_guest_insts: u64,
) -> Option<(u64, u64, u64)> {
    struct BusAdapter<B: CpuBus> {
        ptr: *mut B,
        faulted: Cell<bool>,
    }
    impl<B: CpuBus> JitBus for BusAdapter<B> {
        fn read_u8(&self, addr: u64) -> u8 {
            self.read(addr, Width::W8) as u8
        }
        fn read(&self, addr: u64, width: Width) -> u64 {
            if self.faulted.get() {
                return 0;
            }
            let result = unsafe {
                match width {
                    Width::W8 => (*self.ptr).read_u8(addr).map(u64::from),
                    Width::W16 => (*self.ptr).read_u16(addr).map(u64::from),
                    Width::W32 => (*self.ptr).read_u32(addr).map(u64::from),
                    Width::W64 => (*self.ptr).read_u64(addr),
                }
            };
            match result {
                Ok(value) => value,
                Err(_) => {
                    self.faulted.set(true);
                    0
                }
            }
        }
        fn write_u8(&mut self, addr: u64, val: u8) {
            self.write(addr, Width::W8, val as u64);
        }
        fn write(&mut self, addr: u64, width: Width, value: u64) {
            if self.faulted.get() {
                return;
            }
            let len = match width {
                Width::W8 => 1,
                Width::W16 => 2,
                Width::W32 => 4,
                Width::W64 => 8,
            };
            let result = unsafe {
                (*self.ptr)
                    .preflight_write_bytes(addr, len)
                    .and_then(|()| match width {
                        Width::W8 => (*self.ptr).write_u8(addr, value as u8),
                        Width::W16 => (*self.ptr).write_u16(addr, value as u16),
                        Width::W32 => (*self.ptr).write_u32(addr, value as u32),
                        Width::W64 => (*self.ptr).write_u64(addr, value),
                    })
            };
            if result.is_err() {
                self.faulted.set(true);
            }
        }
    }

    let first_rip = cpu_state.rip();
    let first_count = cache.guest_instruction_count(first_rip)?;
    if cache.get(first_rip).is_none() || first_count > max_guest_insts {
        return None;
    }

    let cpu_before = cpu_state.clone();
    let mut test_cpu = TestCpu::from_cpu_state(cpu_state);
    let mut adapter = BusAdapter {
        ptr: bus as *mut B,
        faulted: Cell::new(false),
    };
    let mut guest_insts = 0u64;
    let mut blocks = 0u64;
    let max_chain_blocks = max_chain_blocks();
    // One buffer for the whole chain. Sizing it per block inside the loop is an
    // allocator round-trip per block, which is most of what the chain is trying to
    // save over interpreting the same instructions.
    let mut temps: Vec<u64> = Vec::new();

    loop {
        let rip = test_cpu.rip;
        let Some(block) = cache.get(rip) else {
            break;
        };
        let Some(count) = cache.guest_instruction_count(rip) else {
            break;
        };
        if guest_insts.saturating_add(count) > max_guest_insts || blocks >= max_chain_blocks {
            break;
        }
        let before = test_cpu;
        match execute_block_cpu_with_scratch(block, &mut test_cpu, &mut adapter, &mut temps) {
            ExecResult::Continue => {
                if adapter.faulted.get() {
                    test_cpu = before;
                    break;
                }
                guest_insts += count;
                blocks += 1;
            }
            ExecResult::ExitToInterpreter { .. } => {
                test_cpu = before;
                break;
            }
        }
    }

    if blocks == 0 {
        *cpu_state = cpu_before;
        return None;
    }
    test_cpu.write_to_cpu_state(cpu_state);
    Some((cpu_state.rip(), guest_insts, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aero_cpu_core::mem::FlatTestBus;
    use aero_cpu_core::state::{gpr, CpuMode};

    struct CodeBus(Vec<u8>);

    impl JitBus for CodeBus {
        fn read_u8(&self, addr: u64) -> u8 {
            self.0.get(addr as usize).copied().unwrap_or(0xcc)
        }

        fn write_u8(&mut self, _addr: u64, _value: u8) {}
    }

    #[test]
    fn read_fault_rolls_cpu_state_back_for_tier0_retry() {
        // mov eax,[edi]; add eax,1; jmp short 0
        let code = CodeBus(vec![0x8b, 0x07, 0x83, 0xc0, 0x01, 0xeb, 0xf9]);
        let decoded = aero_jit_x86::discover_block_mode(&code, 0, BlockLimits::default(), 32);
        let block = translate_block(&decoded);

        let mut cpu = CpuState::new(CpuMode::Protected);
        cpu.set_rip(0);
        cpu.write_gpr32(gpr::RAX, 0xfeed_beef);
        cpu.write_gpr32(gpr::RDI, 0x1000);
        let before_gpr = cpu.gpr;
        let before_rip = cpu.rip();
        let before_rflags = cpu.rflags_snapshot();

        let mut bus = FlatTestBus::new(16);
        assert_eq!(run_ir_block(&block, &mut cpu, &mut bus), None);
        assert_eq!(cpu.gpr, before_gpr);
        assert_eq!(cpu.rip(), before_rip);
        assert_eq!(cpu.rflags_snapshot(), before_rflags);
    }

    #[test]
    fn second_load_fault_discards_the_first_load_register_write() {
        // mov eax,[edi]; add eax,[edi+4]; jmp short 0
        let code = CodeBus(vec![0x8b, 0x07, 0x03, 0x47, 0x04, 0xeb, 0xf9]);
        let decoded = aero_jit_x86::discover_block_mode(&code, 0, BlockLimits::default(), 32);
        let block = translate_block(&decoded);
        assert!(
            block
                .insts
                .iter()
                .filter(|inst| matches!(inst, IrInst::Load { .. }))
                .count()
                >= 2,
            "expected two IR loads, got {:?}",
            block.insts
        );

        let mut cpu = CpuState::new(CpuMode::Protected);
        cpu.set_rip(0);
        cpu.write_gpr32(gpr::RAX, 0x1111_1111);
        cpu.write_gpr32(gpr::RDI, 0);
        let before_gpr = cpu.gpr;
        let before_rip = cpu.rip();

        let mut bus = FlatTestBus::new(4);
        bus.load(0, &[0x02, 0x00, 0x00, 0x00]);
        assert_eq!(run_ir_block(&block, &mut cpu, &mut bus), None);
        assert_eq!(
            cpu.gpr, before_gpr,
            "a faulting second load must not keep the first load's EAX"
        );
        assert_eq!(cpu.rip(), before_rip);
    }

    fn two_store_ir() -> IrBlock {
        IrBlock {
            entry_rip: 0,
            insts: vec![
                IrInst::Store {
                    addr: aero_jit_x86::tier1::ir::ValueId(0),
                    src: aero_jit_x86::tier1::ir::ValueId(1),
                    width: Width::W32,
                },
                IrInst::Store {
                    addr: aero_jit_x86::tier1::ir::ValueId(2),
                    src: aero_jit_x86::tier1::ir::ValueId(3),
                    width: Width::W32,
                },
            ],
            terminator: aero_jit_x86::tier1::ir::IrTerminator::Jump { target: 0 },
            value_types: vec![Width::W64; 4],
        }
    }

    #[test]
    fn user_mode_two_store_block_is_cached_and_writes_both() {
        // mov [edi],eax; mov [edi+4],eax; ret — two RAM MOVs, user RIP.
        let code = vec![0x89, 0x07, 0x89, 0x47, 0x04, 0xc3];
        let mut bus = FlatTestBus::new(32);
        bus.load(0, &code);

        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0, 32, &mut bus),
            "user-mode multi-store WMI/repdrvfs loops must be cacheable"
        );

        let mut cpu = CpuState::new(CpuMode::Protected);
        cpu.set_rip(0);
        cpu.write_gpr32(gpr::RAX, 0xaabb_ccdd);
        cpu.write_gpr32(gpr::RDI, 8);
        assert!(run_ir_block(cache.get(0).expect("cached"), &mut cpu, &mut bus).is_some());
        assert_eq!(u32::from_le_bytes(bus.slice(8, 4).try_into().unwrap()), 0xaabb_ccdd);
        assert_eq!(u32::from_le_bytes(bus.slice(12, 4).try_into().unwrap()), 0xaabb_ccdd);
    }

    #[test]
    fn winsetup_e8_scan_loop_is_cached() {
        // Exact WinSetup.dll apply scan sampled on vds-copy-93b
        // (RIP 0x7fefc0c1272, ~77% of expand samples):
        //   inc rdx; inc dword [r9+0x2ed8]; cmp byte [rdx], 0xe8; jnz
        // Without 0x80 /7 the CMP is Invalid and this loop never caches.
        let code = [
            0x48, 0xff, 0xc2, 0x41, 0xff, 0x81, 0xd8, 0x2e, 0x00, 0x00, 0x80, 0x3a, 0xe8, 0x75,
            0xf1,
        ];
        let mut bus = FlatTestBus::new(0x4000);
        bus.load(0x1272, &code);
        let mut stream = vec![0u8; 16];
        stream[8] = 0xe8;
        bus.load(0x2000, &stream);
        // [r9+0x2ed8] lands at 0x3ed8 and is already zero. Do not
        // `load(0x1000, 0x3000 zeros)` here: that wipes the code at
        // 0x1272 and the stream at 0x2000, so compile succeeds on a
        // page of `add [rax], al` and run_ir_chain exits immediately.

        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0x1272, 64, &mut bus),
            "WinSetup E8 scan must be a cacheable IR block"
        );
        {
            let ir = cache.get(0x1272).expect("cached");
            assert!(
                matches!(
                    ir.terminator,
                    aero_jit_x86::tier1::ir::IrTerminator::CondJump { target: 0x1272, .. }
                ),
                "E8 scan must end in jnz back to the inc, got {}",
                ir.to_text()
            );
        }

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0x1272);
        cpu.write_gpr64(gpr::RDX, 0x1fff);
        cpu.write_gpr64(gpr::R9, 0x1000);

        let (next, insts, blocks) = run_ir_chain(&cache, &mut cpu, &mut bus, 10_000)
            .expect("E8 scan must execute from the cache");
        assert!(
            insts >= 9 * 4,
            "9 iterations (8 zeros + the 0xe8) of a 4-instruction loop, got {insts}"
        );
        assert!(blocks >= 9, "one cached block per iteration, got {blocks}");
        assert_eq!(cpu.read_gpr64(gpr::RDX), 0x2008, "rdx must land on the 0xe8");
        let count = u32::from_le_bytes(bus.slice(0x1000 + 0x2ed8, 4).try_into().unwrap());
        assert_eq!(count, 9, "byte counter at [r9+0x2ed8] must count the scan");
        assert_eq!(next, 0x1272 + 15, "taken-until-E8 then fall through, next={next:#x}");
    }

    #[test]
    fn winsetup_bitstream_sub_shl_test_jg_is_cached() {
        // WinSetup.dll LZX refill sampled on the expand grind
        // (RIP 0x7fefc0c3e9c):
        //   sub dil, cl; shl r11d, cl; test dil, dil; jg
        // Without 8-bit ALU / D3-CL the block never caches.
        let code = [0x40, 0x2a, 0xf9, 0x41, 0xd3, 0xe3, 0x40, 0x84, 0xff, 0x7f, 0x20];
        let mut bus = FlatTestBus::new(0x100);
        bus.load(0, &code);

        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0, 64, &mut bus),
            "WinSetup bitstream test/jg must be a cacheable IR block"
        );
        {
            let ir = cache.get(0).expect("cached");
            assert!(
                matches!(
                    ir.terminator,
                    aero_jit_x86::tier1::ir::IrTerminator::CondJump { fallthrough: 0x0b, .. }
                ),
                "bitstream block must end in jg, got {}",
                ir.to_text()
            );
        }

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0);
        cpu.write_gpr64(gpr::RDI, 0x05);
        cpu.write_gpr64(gpr::RCX, 0x02);
        cpu.write_gpr64(gpr::R11, 0x01);

        let (next, insts, blocks) = run_ir_chain(&cache, &mut cpu, &mut bus, 10_000)
            .expect("bitstream block must execute from the cache");
        assert_eq!(insts, 4);
        assert_eq!(blocks, 1);
        assert_eq!(cpu.read_gpr64(gpr::RDI) & 0xff, 0x03, "dil = 5-2");
        assert_eq!(cpu.read_gpr32(gpr::R11), 0x04, "r11d = 1<<2");
        assert_eq!(next, 0x2b, "jg taken when dil>0, next={next:#x}");
    }

    #[test]
    fn ntoskrnl_cim_hash_imul_sar_and_is_cached() {
        // Exact mix from vds-copy-7b RIP 0xfffff8000d4cdc00:
        //   imul eax, eax, 0xffff9e5f; sar eax, 4; and eax, 0xfff
        let code = [
            0x69, 0xc0, 0x5f, 0x9e, 0xff, 0xff, 0xc1, 0xf8, 0x04, 0x25, 0xff, 0x0f, 0x00, 0x00,
        ];
        let mut bus = FlatTestBus::new(0x2000);
        bus.load(0, &code);
        // 0x00 is now ADD r/m8, r8. Cap the block so discovery does not
        // walk the zero page after the AND immediate.
        bus.load(14, &[0xEB, 0xFE]);
        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0, 64, &mut bus),
            "ntoskrnl CIM hash IMUL must be a cacheable IR block"
        );

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0);
        cpu.write_gpr64(gpr::RAX, 0x41);
        let expected = {
            let mut eax = 0x41u32;
            eax = eax.wrapping_mul(0xffff_9e5f);
            eax = ((eax as i32) >> 4) as u32;
            eax & 0xfff
        };
        let _ = run_ir_block(cache.get(0).expect("cached"), &mut cpu, &mut bus);
        assert_eq!(cpu.rip(), 14, "hash mix must retire to the jmp after AND");
        assert_eq!(cpu.read_gpr32(gpr::RAX), expected);
    }

    #[test]
    fn helper_free_kernel_store_block_is_admitted() {
        let ir = two_store_ir();
        assert!(
            admit_ir_block(0xffff_f800_0000_1000, &ir),
            "ntoskrnl CIM/hash RAM stores are the live WMI wall and must cache"
        );
        assert!(
            admit_ir_block(0x7fef_b190_0000, &ir),
            "user-mode RAM stores stay cacheable"
        );
    }

    #[test]
    fn helper_blocks_stay_rejected() {
        let ir = IrBlock {
            entry_rip: 0,
            insts: vec![IrInst::CallHelper {
                helper: "rep_movs",
                args: vec![],
                ret: None,
            }],
            terminator: aero_jit_x86::tier1::ir::IrTerminator::Jump { target: 0 },
            value_types: vec![Width::W64],
        };
        assert!(
            !admit_ir_block(0x7fef_b190_0000, &ir),
            "helpers stay fail-closed so MMIO/SSE assists cannot replay"
        );
    }

    #[test]
    fn repdrvfs_utf16_store_loop_is_cached() {
        // Exact repdrvfs word-store loop sampled on vds-copy-3.8b at
        // 0x7fefb1918e0: mov [rcx],r11w; add rcx,2; dec rdx; jnz
        let code = [
            0x66, 0x44, 0x89, 0x19, 0x48, 0x83, 0xc1, 0x02, 0x48, 0x83, 0xea, 0x01, 0x75, 0xf2,
        ];
        let mut bus = FlatTestBus::new(0x2000);
        bus.load(0x8e0, &code);
        bus.load(0x1000, &[0u8; 0x40]);

        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0x8e0, 64, &mut bus),
            "repdrvfs UTF-16 store loop must be cacheable"
        );

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0x8e0);
        cpu.write_gpr64(gpr::RCX, 0x1000);
        cpu.write_gpr64(gpr::RDX, 4);
        cpu.write_gpr64(gpr::R11, 0x0041);

        let (_next, insts, blocks) = run_ir_chain(&cache, &mut cpu, &mut bus, 10_000)
            .expect("UTF-16 store loop must execute from the cache");
        assert!(insts >= 16, "4 iterations of a 4-instruction loop, got {insts}");
        assert!(blocks >= 4, "one cached block per iteration, got {blocks}");
        assert_eq!(cpu.read_gpr64(gpr::RDX), 0);
        assert_eq!(cpu.read_gpr64(gpr::RCX), 0x1000 + 8);
        let mut got = [0u8; 8];
        bus.read_bytes(0x1000, &mut got).unwrap();
        assert_eq!(&got, &[0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0x41, 0x00]);
    }

    #[test]
    fn wmiprvse_merge_loop_runs_as_a_cached_chain() {
        // Exact bytes of the wmiprvse compare/store loop at 0x7feface062b
        // (see plus6730m DUMP_LINEAR). Three Jcc-split blocks; the back-edge
        // is `cmp r10d,r9d; jb 0x62b`.
        let code = [
            0x44, 0x3b, 0x19, 0x75, 0x07, 0x8b, 0x02, 0x89, 0x07, 0x44, 0x3b, 0x19, 0x72, 0x10,
            0x41, 0xff, 0xc2, 0x48, 0x83, 0xc1, 0x08, 0x48, 0x83, 0xc2, 0x08, 0x45, 0x3b, 0xd1,
            0x72, 0xe2,
        ];
        let mut bus = FlatTestBus::new(0x2000);
        bus.load(0x62b, &code);
        bus.load(0x1000, &[0u8; 0x200]);
        bus.load(0x1200, &[0u8; 0x200]);
        bus.load(0x1400, &[0u8; 0x40]);

        let mut cache = IrBlockCache::new();
        for rip in [0x62b, 0x630, 0x637, 0x639] {
            assert!(
                cache.compile(rip, 64, &mut bus),
                "loop entry {rip:#x} must be cacheable"
            );
        }

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0x62b);
        cpu.write_gpr64(gpr::R11, 0xffff_ffff);
        cpu.write_gpr64(gpr::R9, 16);
        cpu.write_gpr64(gpr::R10, 0);
        cpu.write_gpr64(gpr::RCX, 0x1000);
        cpu.write_gpr64(gpr::RDX, 0x1200);
        cpu.write_gpr64(gpr::RDI, 0x1400);

        let (next, insts, blocks) = run_ir_chain(&cache, &mut cpu, &mut bus, 10_000)
            .expect("chained loop must execute");
        assert!(
            insts >= 16 * 8,
            "16 iterations of an 8-instruction loop, got {insts}"
        );
        assert!(
            blocks >= 16 * 3,
            "at least 3 blocks per iteration, got {blocks}"
        );
        assert_eq!(cpu.read_gpr32(gpr::R10), 16);
        assert_eq!(cpu.read_gpr64(gpr::RCX), 0x1000 + 16 * 8);
        assert_ne!(next, 0x62b, "the loop should exit via the taken-fallthrough of the last jb");
    }

    #[test]
    fn cached_test_eax_eax_jz_takes_when_eax_is_zero() {
        // Exact wbemcomn bit-extractor prefix at 0x7fefb2ccd14:
        //   mov eax,[rcx+4]; mov rdx,rcx; test eax,eax; jz +0x14
        //   dec eax; mov [rcx+4],eax
        // If JZ is wrong, remaining underflows to 0xffffffff and Setup
        // waits on WMI for hours.
        let code = [
            0x8b, 0x41, 0x04, 0x48, 0x8b, 0xd1, 0x85, 0xc0, 0x74, 0x14, 0xff, 0xc8, 0x89, 0x41,
            0x04,
        ];
        let mut bus = FlatTestBus::new(0x2000);
        bus.load(0x100, &code);
        let obj = [0u8; 32];
        bus.load(0x400, &obj);

        let mut cache = IrBlockCache::new();
        assert!(
            cache.compile(0x100, 64, &mut bus),
            "test/jz prefix must be cacheable"
        );

        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.set_rip(0x100);
        cpu.write_gpr64(gpr::RCX, 0x400);
        cpu.write_gpr64(gpr::RAX, 0xdead);

        let (next, _, _) =
            run_ir_chain(&cache, &mut cpu, &mut bus, 16).expect("block must run");
        assert_eq!(
            next, 0x11e,
            "JZ must be taken when [rcx+4]==0, next={next:#x} eax={:#x}",
            cpu.read_gpr32(gpr::RAX)
        );
        // remaining must stay 0, not underflow
        let mut got = [0u8; 8];
        bus.read_bytes(0x400, &mut got).unwrap();
        assert_eq!(
            u32::from_le_bytes(got[4..8].try_into().unwrap()),
            0,
            "JZ miss would store dec(0)=0xffffffff into remaining"
        );
    }

    #[test]
    fn ir_cache_cryptsp_sha256_compress_abc_matches_fips() {
        // The live `--jit` path is this IR cache, not Wasmtime. cryptsp's
        // SHA-256 round loop (0xe2c0) is user-mode, load-only, and hot —
        // it is compiled. Interpreter compress already matches FIPS; this
        // is the path VDS actually hashes rsaenh.dll on.
        const FN: &[u8] = include_bytes!("../../aero-cpu-core/tests/data/cryptsp_sha256_compress.bin");
        const K_LE: &[u8] = &aero_cpu_core::sha2_constants::SHA256_K_LE;
        const CODE: u64 = 0xe210;
        const K_ADDR: u64 = 0x1590;
        const HLT: u64 = 0x1f00;
        const STACK: u64 = 0x8000;
        const STATE: u64 = 0x2000;
        const BLOCK: u64 = 0x3000;

        let mut bus = FlatTestBus::new(0x1_0000);
        bus.load(HLT, &[0xF4]);
        bus.load(CODE, FN);
        bus.load(K_ADDR, K_LE);
        let iv = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];
        for (i, word) in iv.iter().enumerate() {
            bus.write_u32(STATE + (i as u64) * 4, *word).unwrap();
        }
        let mut block = [0u8; 64];
        block[0] = b'a';
        block[1] = b'b';
        block[2] = b'c';
        block[3] = 0x80;
        block[63] = 24;
        bus.load(BLOCK, &block);
        bus.write_u64(STACK - 8, HLT).unwrap();

        let mut cache = IrBlockCache::new();
        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.write_gpr64(gpr::RSP, STACK - 8);
        cpu.write_gpr64(gpr::RCX, STATE);
        cpu.write_gpr64(gpr::RDX, BLOCK);
        cpu.set_rip(CODE);
        cpu.halted = false;

        let mut steps = 0u64;
        let mut compiled = 0u32;
        let mut tried = std::collections::HashSet::new();
        while steps < 200_000 && !cpu.halted {
            let rip = cpu.rip();
            if cache.get(rip).is_none() && tried.insert(rip) && cache.compile(rip, 64, &mut bus) {
                compiled += 1;
            }
            if cache.get(rip).is_some() {
                match run_ir_block(cache.get(rip).unwrap(), &mut cpu, &mut bus) {
                    Some(_) => {
                        steps += cache.guest_instruction_count(rip).unwrap_or(1);
                        continue;
                    }
                    None => {}
                }
            }
            let res = aero_cpu_core::interp::tier0::exec::run_batch(&mut cpu, &mut bus, 1);
            steps += res.executed;
            match res.exit {
                aero_cpu_core::interp::tier0::exec::BatchExit::Halted => break,
                aero_cpu_core::interp::tier0::exec::BatchExit::Completed
                | aero_cpu_core::interp::tier0::exec::BatchExit::Branch => {}
                other => panic!("compress {other:?} rip={rip:#x}"),
            }
        }
        assert!(cpu.halted, "compress did not halt after {steps} inst");
        assert!(
            compiled > 0,
            "expected the SHA-256 round loop to be IR-cached"
        );
        let mut got = [0u32; 8];
        for (i, word) in got.iter_mut().enumerate() {
            *word = bus.read_u32(STATE + (i as u64) * 4).unwrap();
        }
        assert_eq!(
            got,
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c,
                0xb410ff61, 0xf20015ad
            ],
            "IR-cache SHA-256 compress of padded \"abc\" (compiled={compiled})"
        );
    }

    #[test]
    fn ir_cache_cryptsp_pe_hasher_matches_authenticode_sha256_if_fixture_present() {
        // 0xa298 is what CheckSignatureInFile uses. Compile every admissible
        // block the hasher hits (the live machine does this after ~50 entries)
        // and require the digest still match the host Authenticode SHA-256.
        // The mapped `cryptsp.dll` image is a copy of a Microsoft binary and is
        // deliberately not committed; supply one locally via `AERO_CRYPTSP_IMAGE`
        // or at `crates/aero-cpu-core/tests/data/cryptsp_mapped.bin`. Without it
        // this test skips. See that directory's README.
        let image = {
            match std::env::var("AERO_CRYPTSP_IMAGE") {
                Ok(path) => std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("AERO_CRYPTSP_IMAGE={path}: {e}")),
                Err(_) => {
                    let path = concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../aero-cpu-core/tests/data/cryptsp_mapped.bin"
                    );
                    match std::fs::read(path) {
                        Ok(bytes) => bytes,
                        Err(_) => {
                            eprintln!("SKIP: no cryptsp image at {path} (set AERO_CRYPTSP_IMAGE)");
                            return;
                        }
                    }
                }
            }
        };
        const HLT: u64 = 0x1_f000;
        const STUB_MEMCPY: u64 = 0x1_9000;
        const STUB_MEMSET: u64 = 0x1_9080;
        const HASH_OBJ: u64 = 0x1_e000;
        const DIGEST: u64 = 0x1_e800;
        const STACK: u64 = 0x1_f800;
        const FILE: u64 = 0x2_0000;
        const PE_HASH: u64 = 0xa298;
        const COOKIE_CHECK: u64 = 0xaa90;

        let memcpy = [
            0x56u8, 0x57, 0x48, 0x89, 0xcf, 0x48, 0x89, 0xd6, 0x4c, 0x89, 0xc1, 0x48, 0x89, 0xf8,
            0xf3, 0xa4, 0x5f, 0x5e, 0xc3,
        ];
        let memset = [
            0x57u8, 0x48, 0x89, 0xcf, 0x48, 0x89, 0xd0, 0x4c, 0x89, 0xc1, 0xf3, 0xaa, 0x5f, 0xc3,
        ];

        let mut file = vec![0u8; 0x80];
        file[0] = b'M';
        file[1] = b'Z';
        file[0x3c] = 0x80;
        let mut pe = vec![0u8; 0x108];
        pe[0..4].copy_from_slice(b"PE\0\0");
        pe[4..6].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[6..8].copy_from_slice(&1u16.to_le_bytes());
        pe[20..22].copy_from_slice(&0x00f0u16.to_le_bytes());
        pe[22..24].copy_from_slice(&0x0022u16.to_le_bytes());
        pe[24..26].copy_from_slice(&0x020bu16.to_le_bytes());
        pe[26] = 8;
        pe[24 + 32..24 + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[24 + 36..24 + 40].copy_from_slice(&0x200u32.to_le_bytes());
        pe[24 + 56..24 + 60].copy_from_slice(&0x2000u32.to_le_bytes());
        pe[24 + 60..24 + 64].copy_from_slice(&0x200u32.to_le_bytes());
        pe[24 + 64..24 + 68].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        pe[24 + 108..24 + 112].copy_from_slice(&16u32.to_le_bytes());
        let cert_off = 0x400u32;
        let cert_sz = 0x80u32;
        let cert_dir = 24 + 112 + 32;
        pe[cert_dir..cert_dir + 4].copy_from_slice(&cert_off.to_le_bytes());
        pe[cert_dir + 4..cert_dir + 8].copy_from_slice(&cert_sz.to_le_bytes());
        let mut sect = vec![0u8; 40];
        sect[0..5].copy_from_slice(b".text");
        sect[8..12].copy_from_slice(&0x200u32.to_le_bytes());
        sect[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
        sect[16..20].copy_from_slice(&0x200u32.to_le_bytes());
        sect[20..24].copy_from_slice(&0x200u32.to_le_bytes());
        sect[36..40].copy_from_slice(&0x20u32.to_le_bytes());
        file.extend_from_slice(&pe);
        file.extend_from_slice(&sect);
        file.resize(0x200, 0);
        file.extend((0..0x200).map(|i| i as u8));
        file.extend(std::iter::repeat(0x30u8).take(0x80));
        file[cert_off as usize] = 0x80;
        file[cert_off as usize + 5] = 0x02;
        file[cert_off as usize + 6] = 0x02;

        let checksum = 0x80 + 24 + 64;
        let cdir = 0x80 + cert_dir;
        let mut stream = Vec::new();
        stream.extend_from_slice(&file[..checksum]);
        stream.extend_from_slice(&file[checksum + 4..cdir]);
        stream.extend_from_slice(&file[cdir + 8..cert_off as usize]);
        stream.extend_from_slice(&file[cert_off as usize + cert_sz as usize..]);
        let mut bus = FlatTestBus::new(0x8_0000);
        bus.load(0, &image);
        bus.load(HLT, &[0xF4]);
        bus.load(STUB_MEMCPY, &memcpy);
        bus.load(STUB_MEMSET, &memset);
        bus.write_u64(0x1268, STUB_MEMCPY).unwrap();
        bus.write_u64(0x1270, STUB_MEMSET).unwrap();
        bus.write_u64(0x12b0, STUB_MEMCPY).unwrap();
        bus.load(COOKIE_CHECK, &[0xC3]);
        bus.load(FILE, &file);
        bus.write_u32(HASH_OBJ, 0x800c).unwrap();
        for off in 0u64..0x48 {
            bus.write_u64(STACK + off, 0).unwrap();
        }
        bus.write_u64(STACK - 8 + 0x38, DIGEST).unwrap();
        bus.write_u64(STACK - 8, HLT).unwrap();

        let mut cache = IrBlockCache::new();
        let mut cpu = CpuState::new(CpuMode::Long);
        cpu.write_gpr64(gpr::RSP, STACK - 8);
        cpu.write_gpr64(gpr::RCX, HASH_OBJ);
        cpu.write_gpr64(gpr::RDX, FILE);
        cpu.write_gpr64(gpr::R8, file.len() as u64);
        cpu.write_gpr64(gpr::R9, file.len() as u64);
        cpu.set_rip(PE_HASH);
        cpu.halted = false;

        let mut tried = std::collections::HashSet::new();
        let mut steps = 0u64;
        while steps < 2_000_000 && !cpu.halted {
            let rip = cpu.rip();
            if cache.get(rip).is_none() && tried.insert(rip) {
                let _ = cache.compile(rip, 64, &mut bus);
            }
            if let Some(block) = cache.get(rip) {
                if let Some(_) = run_ir_block(block, &mut cpu, &mut bus) {
                    steps += cache.guest_instruction_count(rip).unwrap_or(1);
                    continue;
                }
            }
            let res = aero_cpu_core::interp::tier0::exec::run_batch(&mut cpu, &mut bus, 1);
            steps += res.executed;
            match res.exit {
                aero_cpu_core::interp::tier0::exec::BatchExit::Halted => break,
                aero_cpu_core::interp::tier0::exec::BatchExit::Completed
                | aero_cpu_core::interp::tier0::exec::BatchExit::Branch => {}
                other => panic!("hasher {other:?} rip={rip:#x}"),
            }
        }
        assert!(cpu.halted, "0xa298 did not halt after {steps}");
        assert_eq!(cpu.read_gpr32(gpr::RAX), 0, "0xa298 returned {:#x}", cpu.read_gpr64(gpr::RAX));
        let mut got = [0u8; 32];
        bus.read_bytes(DIGEST, &mut got).unwrap();
        let expect = host_sha256_for_jit_test(&stream);
        assert_eq!(got, expect, "IR-cache 0xa298 Authenticode SHA-256");
    }

    fn host_sha256_for_jit_test(data: &[u8]) -> [u8; 32] {
        fn rotr(x: u32, n: u32) -> u32 {
            x.rotate_right(n)
        }
        fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
            // The published FIPS round constants, from the one place they are defined
            // and re-derived: a fourth hand-copy is a fourth chance for a typo.
            const K: [u32; 64] = aero_cpu_core::sha2_constants::SHA256_K;
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = s1.wrapping_add(w[i - 7]).wrapping_add(s0).wrapping_add(w[i - 16]);
            }
            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            );
            for i in 0..64 {
                let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
                let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
            state[4] = state[4].wrapping_add(e);
            state[5] = state[5].wrapping_add(f);
            state[6] = state[6].wrapping_add(g);
            state[7] = state[7].wrapping_add(h);
        }
        let mut state = [
            0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
            0x1f83d9ab, 0x5be0cd19,
        ];
        let mut i = 0;
        while i + 64 <= data.len() {
            compress(&mut state, data[i..i + 64].try_into().unwrap());
            i += 64;
        }
        let mut block = [0u8; 64];
        let rem = data.len() - i;
        block[..rem].copy_from_slice(&data[i..]);
        block[rem] = 0x80;
        let bit_len = (data.len() as u64) * 8;
        if rem >= 56 {
            compress(&mut state, &block);
            block = [0u8; 64];
        }
        block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        compress(&mut state, &block);
        let mut out = [0u8; 32];
        for (i, word) in state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

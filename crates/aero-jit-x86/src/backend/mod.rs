//! Native (non-wasm32) runtime backend for executing dynamically-generated WASM blocks.
//!
//! This module provides:
//! - A reference `wasmtime`-powered Tier-1 executor (`WasmtimeBackend`)
//! - A shareable wrapper (`WasmBackend`) suitable for driving
//!   `aero_cpu_core::jit::runtime::JitRuntime` from a compile worker
//! - Convenience wrappers for Tier-1 compilation (`compile_and_install`)
//!
//! The canonical Tier-1 compilation pipeline + compile request queue live in
//! [`crate::tier1::pipeline`]. The helpers in this module are thin wrappers
//! around that API for consumers that already depend on `WasmBackend`.

use std::cell::RefCell;
use std::rc::Rc;

use aero_cpu_core::jit::cache::CompiledBlockHandle;
use aero_cpu_core::jit::runtime::{CodeVersionTable, CompileRequestSink, JitBackend, JitRuntime};
use aero_cpu_core::state::CpuState as CoreCpuState;

use crate::tier1::Tier1WasmOptions;
use crate::tier1_pipeline::{Tier1Compiler, Tier1WasmRegistry};
use crate::Tier1Bus;

/// Minimal interface a host CPU type must expose to execute Tier-1 WASM blocks.
///
/// The Tier-1 WASM ABI uses the in-memory layout of [`aero_cpu_core::state::CpuState`] (plus
/// optional JIT-only context data appended after the struct). The backend copies the architectural
/// subset used by Tier-1 (GPRs/RIP/RFLAGS) into the shared `WebAssembly.Memory`, calls the compiled
/// block, and then copies the updated values back into the host CPU value.
pub trait Tier1Cpu {
    fn tier1_state(&self) -> &CoreCpuState;
    fn tier1_state_mut(&mut self) -> &mut CoreCpuState;
}

impl Tier1Cpu for CoreCpuState {
    fn tier1_state(&self) -> &CoreCpuState {
        self
    }

    fn tier1_state_mut(&mut self) -> &mut CoreCpuState {
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod shared_memory;
mod wasmtime;

pub use crate::compiled_backend::{CompiledBackend, GuestAccessor, GuestTranslator};
pub use wasmtime::{WasmtimeBackend, WriteObserver, JIT_TLB_FLUSHES, JIT_TLB_MISSES};

/// A cloneable handle around [`WasmtimeBackend`] so compilation workers can add table entries while
/// the [`JitRuntime`] owns a copy of the backend.
///
/// `WasmBackend` also implements [`Tier1Bus`], allowing the Tier-1 compiler pipeline to read guest
/// code bytes directly from the backend's shared linear memory.
pub struct WasmBackend<Cpu>(Rc<RefCell<WasmtimeBackend<Cpu>>>);

impl<Cpu> Clone for WasmBackend<Cpu> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<Cpu> Default for WasmBackend<Cpu> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Cpu> WasmBackend<Cpu> {
    #[must_use]
    #[track_caller]
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(WasmtimeBackend::new())))
    }

    #[must_use]
    #[track_caller]
    pub fn with_memory_pages(memory_pages: u32, cpu_ptr: i32) -> Self {
        Self(Rc::new(RefCell::new(
            WasmtimeBackend::new_with_memory_pages(memory_pages, cpu_ptr),
        )))
    }

    #[track_caller]
    pub fn add_compiled_block(&mut self, wasm_bytes: &[u8]) -> u32 {
        self.0.borrow_mut().add_compiled_block(wasm_bytes)
    }

    /// Installs or clears the optional guest-write observer invoked by the imported `env.mem_write_*`
    /// helpers.
    pub fn set_write_observer(&mut self, observer: Option<WriteObserver>) {
        self.0.borrow_mut().set_write_observer(observer);
    }
}

impl<Cpu> Tier1WasmRegistry for WasmBackend<Cpu> {
    fn register_tier1_block(&mut self, wasm: Vec<u8>, _exit_to_interpreter: bool) -> u32 {
        self.add_compiled_block(&wasm)
    }
}

impl<Cpu> Tier1Bus for WasmBackend<Cpu> {
    fn read_u8(&self, addr: u64) -> u8 {
        self.0.borrow().read_u8(addr)
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        self.0.borrow_mut().write_u8(addr, value);
    }
}

impl<Cpu> JitBackend for WasmBackend<Cpu>
where
    Cpu: Tier1Cpu,
{
    type Cpu = Cpu;

    fn execute(
        &mut self,
        table_index: u32,
        cpu: &mut Self::Cpu,
    ) -> aero_cpu_core::jit::runtime::JitBlockExit {
        self.0.borrow_mut().execute(table_index, cpu)
    }

    fn code_version_table(&mut self) -> Option<Box<dyn CodeVersionTable>> {
        self.0.borrow_mut().code_version_table()
    }
}

/// Compile the Tier-1 block starting at `entry_rip`, instantiate it in `backend`, and return a
/// cache-installable [`CompiledBlockHandle`].
///
/// The returned handle is *not* installed into `runtime`; callers should pass it to
/// [`JitRuntime::install_handle`].
#[track_caller]
pub fn compile_and_install<Cpu, C>(
    backend: &mut WasmBackend<Cpu>,
    runtime: &JitRuntime<WasmBackend<Cpu>, C>,
    entry_rip: u64,
    bitness: u32,
) -> CompiledBlockHandle
where
    Cpu: Tier1Cpu,
    C: CompileRequestSink,
{
    compile_and_install_with_options(
        backend,
        runtime,
        entry_rip,
        bitness,
        Tier1WasmOptions::default(),
    )
}

/// Same as [`compile_and_install`], but allows selecting Tier-1 WASM codegen options (e.g. enabling
/// the inline-TLB fast-path).
#[track_caller]
pub fn compile_and_install_with_options<Cpu, C>(
    backend: &mut WasmBackend<Cpu>,
    runtime: &JitRuntime<WasmBackend<Cpu>, C>,
    entry_rip: u64,
    bitness: u32,
    options: Tier1WasmOptions,
) -> CompiledBlockHandle
where
    Cpu: Tier1Cpu,
    C: CompileRequestSink,
{
    Tier1Compiler::new(backend.clone(), backend.clone())
        .with_wasm_options(options)
        .compile_handle(runtime, entry_rip, bitness)
        .expect("Tier-1 compilation failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::capture_panic_location;

    #[test]
    fn wasm_backend_with_memory_pages_panics_at_call_site_on_negative_cpu_ptr() {
        let expected_file = file!();
        let expected_line = line!() + 2;
        let (file, line) = capture_panic_location(|| {
            let _backend = WasmBackend::<()>::with_memory_pages(1, -1);
        });
        assert_eq!(file, expected_file);
        assert_eq!(line, expected_line);
    }
}

impl CompiledBackend for WasmtimeBackend<aero_cpu_core::state::CpuState> {
    fn add_block(&mut self, wasm: &[u8]) -> u32 {
        self.add_compiled_block(wasm)
    }

    fn add_blocks(&mut self, wasm: &[u8], count: usize) -> Vec<u32> {
        self.add_compiled_blocks(wasm, count)
    }

    fn add_dispatched_blocks(&mut self, wasm: &[u8], count: usize) -> Vec<u32> {
        WasmtimeBackend::add_dispatched_blocks(self, wasm, count)
    }

    fn can_optimize(&self) -> bool {
        WasmtimeBackend::can_optimize(self)
    }

    fn promote_block(&mut self, table_index: u32, wasm: &[u8]) -> bool {
        WasmtimeBackend::promote_block(self, table_index, wasm)
    }

    fn promote_blocks(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        WasmtimeBackend::promote_blocks(self, table_indices, wasm)
    }

    fn promote_blocks_replacing(&mut self, table_indices: &[u32], wasm: &[u8]) -> usize {
        WasmtimeBackend::promote_blocks_replacing(self, table_indices, wasm)
    }

    fn compiles_async(&self) -> bool {
        WasmtimeBackend::compiles_in_flight(self) < usize::MAX
            && WasmtimeBackend::has_compile_workers(self)
    }

    fn submit_dispatched_blocks(&mut self, wasm: Vec<u8>, count: usize) -> Option<u64> {
        WasmtimeBackend::submit_dispatched_blocks(self, wasm, count)
    }

    fn take_dispatched_blocks(&mut self, id: u64) -> Option<Vec<u32>> {
        WasmtimeBackend::take_dispatched_blocks(self, id)
    }

    fn poll_dispatched_blocks(&mut self) -> Vec<(u64, Vec<u32>)> {
        WasmtimeBackend::poll_dispatched_blocks(self)
    }

    fn can_cache_modules(&self) -> bool {
        true
    }

    fn add_dispatched_blocks_serialized(
        &mut self,
        wasm: &[u8],
        count: usize,
    ) -> (Vec<u32>, Option<Vec<u8>>) {
        WasmtimeBackend::add_dispatched_blocks_serialized(self, wasm, count)
    }

    unsafe fn add_dispatched_precompiled(
        &mut self,
        bytes: &[u8],
        count: usize,
    ) -> Option<Vec<u32>> {
        // Safety: forwarded from the trait's contract.
        unsafe { WasmtimeBackend::add_dispatched_precompiled(self, bytes, count) }
    }

    fn compiles_in_flight(&self) -> usize {
        WasmtimeBackend::compiles_in_flight(self)
    }

    fn promote_blocks_serialized(
        &mut self,
        table_indices: &[u32],
        wasm: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        WasmtimeBackend::promote_blocks_serialized(self, table_indices, wasm)
    }

    unsafe fn promote_blocks_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        // Safety: forwarded from the trait's contract.
        unsafe { WasmtimeBackend::promote_blocks_precompiled(self, table_indices, bytes) }
    }

    fn promote_blocks_replacing_serialized(
        &mut self,
        table_indices: &[u32],
        wasm: &[u8],
    ) -> (usize, Option<Vec<u8>>) {
        WasmtimeBackend::promote_blocks_replacing_serialized(self, table_indices, wasm)
    }

    unsafe fn promote_blocks_replacing_precompiled(
        &mut self,
        table_indices: &[u32],
        bytes: &[u8],
    ) -> Option<usize> {
        // Safety: forwarded from the trait's contract.
        unsafe {
            WasmtimeBackend::promote_blocks_replacing_precompiled(self, table_indices, bytes)
        }
    }

    fn can_chain_in_wasm(&self) -> bool {
        true
    }

    fn chain_map_insert(&mut self, rip: u64, module_id: u32, index: u32) {
        WasmtimeBackend::chain_map_insert(self, rip, module_id, index);
    }

    fn chain_map_forget(&mut self, rip: u64) {
        WasmtimeBackend::chain_map_forget(self, rip);
    }

    fn chain_map_clear(&mut self) {
        WasmtimeBackend::chain_map_clear(self);
    }

    fn wasm_chain_steps(&self) -> u64 {
        WasmtimeBackend::wasm_chain_steps(self)
    }

    fn set_chain_limits(&mut self, retired_budget: u64, max_blocks: u32) {
        WasmtimeBackend::set_chain_limits(self, retired_budget, max_blocks);
    }

    fn set_clock(
        &mut self,
        clock: Option<crate::compiled_backend::GuestClock>,
    ) -> Option<crate::compiled_backend::GuestClock> {
        WasmtimeBackend::set_clock(self, clock)
    }

    fn set_assist(
        &mut self,
        assist: Option<crate::compiled_backend::GuestAssist>,
    ) -> Option<crate::compiled_backend::GuestAssist> {
        WasmtimeBackend::set_assist(self, assist)
    }

    fn can_assist(&self) -> bool {
        WasmtimeBackend::has_assist(self)
    }

    fn set_ram_base(&mut self, ram_base: u64) {
        WasmtimeBackend::set_ram_base(self, ram_base);
    }

    fn flush_tlb(&mut self) {
        WasmtimeBackend::flush_tlb(self);
    }

    fn set_translator(&mut self, translator: Option<GuestTranslator>) -> Option<GuestTranslator> {
        WasmtimeBackend::set_translator(self, translator)
    }

    fn set_accessor(&mut self, accessor: Option<GuestAccessor>) -> Option<GuestAccessor> {
        WasmtimeBackend::set_accessor(self, accessor)
    }

    fn last_retired(&self) -> u64 {
        WasmtimeBackend::last_retired(self)
    }

    fn run_chain(
        &mut self,
        entry_index: u32,
        cpu: &mut aero_cpu_core::state::CpuState,
        lookup: &mut dyn FnMut(u64, u64) -> Option<u32>,
        max_blocks: u32,
    ) -> crate::compiled_backend::ChainOutcome {
        WasmtimeBackend::execute_chain(self, entry_index, cpu, lookup, max_blocks)
    }
}

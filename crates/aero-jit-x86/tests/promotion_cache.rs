//! A promotion loaded from the cache must be the promotion it replaced.
//!
//! Promotion is the expensive compile — Cranelift, not Winch — and until this cache
//! existed it was the one thing the module cache did *not* cover, so every run paid
//! for every promotion again. That mattered beyond warm-up: it meant the promotion
//! threshold had only ever been tuned against Cranelift's price rather than against
//! what promoted code is worth, and the sweep that rejected lower thresholds was run
//! with a cache that did not cover the thing being increased.
//!
//! Two properties are worth pinning:
//!
//! * **A deserialized promotion computes what a compiled one does.** The alternative
//!   is a machine that is subtly wrong only when a cache directory happens to be warm,
//!   which is the worst shape a bug can take.
//! * **The two engines' modules are keyed apart.** Both engines compile the *same*
//!   wasm bytes, so a cache keyed on those bytes alone would let a Winch module be
//!   loaded as a promotion. `Module::deserialize` checks a version header, not which
//!   engine produced the body — so this is enforced by the key, and the key is in
//!   `CompiledTier::promote_cached` (`.opt.cwasm` against `.cwasm`).
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test promotion_cache
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{gpr, CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::compiled_backend::CompiledBackend;
use aero_jit_x86::tier1::Tier1WasmOptions;
use aero_jit_x86::{discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen};
use aero_types::Width;

const ENTRY: u64 = 0x1000;

fn tier_options() -> Tier1WasmOptions {
    Tier1WasmOptions {
        inline_tlb: true,
        single_fast_path: true,
        shared_mem_helpers: true,
        dispatcher: true,
        ..Default::default()
    }
}

/// `add rax, rbx ; xor rcx, rdx ; sub rax, rcx ; ret`
const CODE: [u8; 10] = [
    0x48, 0x01, 0xd8, 0x48, 0x31, 0xd1, 0x48, 0x29, 0xc8, 0xc3,
];

fn start_state() -> CpuState {
    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.write_gpr64(gpr::RAX, 0x1234_5678_9abc_def0);
    cpu.write_gpr64(gpr::RBX, 0x0fed_cba9_8765_4321);
    cpu.write_gpr64(gpr::RCX, 0xaaaa_5555_aaaa_5555);
    cpu.write_gpr64(gpr::RDX, 0x5555_aaaa_5555_aaaa);
    cpu.write_gpr64(gpr::RSP, 0x8000);
    cpu
}

/// Build a backend with the block admitted, returning it and the block's wasm.
fn admitted() -> Option<(WasmtimeBackend<CpuState>, Vec<u8>, u32)> {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(64, 0x20000);
    if !backend.can_optimize() {
        return None;
    }
    for (i, b) in CODE.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let ir = translate_block(&discover_block(&backend, ENTRY, BlockLimits::default()));
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&[&ir], tier_options());
    let idx = backend.add_dispatched_blocks(&wasm, 1)[0];
    Some((backend, wasm, idx))
}

fn run(backend: &mut WasmtimeBackend<CpuState>, idx: u32) -> CpuState {
    let mut cpu = start_state();
    backend.write(0x8000, Width::W64, 0x2000);
    let outcome =
        CompiledBackend::run_chain(backend, idx, &mut cpu, &mut |_rip, _retired| None, 1);
    assert!(outcome.retired > 0, "the block retired nothing");
    cpu
}

#[test]
fn a_cached_promotion_computes_what_a_compiled_one_does() {
    let Some((mut a, wasm, idx_a)) = admitted() else {
        return; // Cranelift half unavailable; nothing to compare.
    };
    let (n, serialized) = a.promote_blocks_serialized(&[idx_a], &wasm);
    assert_eq!(n, 1, "the block was not promoted");
    let bytes = serialized.expect("a promoted module must serialize, or nothing can cache it");
    let compiled = run(&mut a, idx_a);

    // A second backend that never compiles the promotion, only loads it.
    let Some((mut b, _wasm, idx_b)) = admitted() else {
        return;
    };
    // Safety: `bytes` came from `promote_blocks_serialized` on an engine configured
    // identically, in this same process.
    let loaded = unsafe { b.promote_blocks_precompiled(&[idx_b], &bytes) };
    assert_eq!(
        loaded,
        Some(1),
        "the serialized promotion must load; if it cannot, the cache is silently dead"
    );
    let cached = run(&mut b, idx_b);

    assert_eq!(compiled.rip, cached.rip, "RIP differs");
    for i in 0..16 {
        assert_eq!(
            compiled.gpr[i], cached.gpr[i],
            "GPR {i} differs: {:#x} compiled, {:#x} cached",
            compiled.gpr[i], cached.gpr[i]
        );
    }
    assert_eq!(compiled.rflags(), cached.rflags(), "RFLAGS differ");
}

/// A promotion must not be satisfied by the *admission* engine's module for the same
/// wasm. The two are keyed apart on disk; this shows why that is load-bearing.
#[test]
fn an_admission_module_is_not_a_promotion() {
    let Some((mut a, wasm, _idx)) = admitted() else {
        return;
    };
    let winch = a.serialize_module(&wasm);
    let promoted = a.promote_blocks_serialized(&[], &wasm).1;
    let Some(cranelift) = promoted else {
        return;
    };
    assert_ne!(
        winch, cranelift,
        "the two engines produced byte-identical modules, so the cache key would not \
         need to separate them — check that tier-up is actually using Cranelift"
    );
}

//! What admitting a block costs, in host instructions.
//!
//! `compile_cost` times admission with `Instant`, which on a loaded machine measures
//! contention as much as work — and the conclusion "Wasmtime compilation is what
//! rations the tier" rests entirely on that number being real. Host instructions
//! retired are immune to load, so this settles it.
//!
//! `AERO_BLOCKS` blocks are admitted, batched `AERO_BATCH` to a module, and two
//! counts are differenced so engine construction and warm-up cancel:
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test admission_insns -- --nocapture
//! BIN=$(ls -t target/release/deps/admission_insns-* | grep -v '\.d$' | head -1)
//! for n in 200 1000; do perf stat -e instructions:u env AERO_BLOCKS=$n $BIN --nocapture; done
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::tier1::ir::IrBlock;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x100;

/// Distinct blocks so nothing is deduplicated: each is `add eax, N` then `ret`.
///
/// One backend for all of them. Building a `WasmtimeBackend` per block — an engine, a
/// store and an anchor module each time — costs far more than the thing being
/// measured, and charging that to "codegen" made our own emitter look like 54% of
/// admission when it is a small fraction of it.
fn ir_for(backend: &mut WasmtimeBackend<CpuState>, n: usize) -> IrBlock {
    let code = [0x83, 0xc0, (n % 200) as u8 + 1, 0xc3];
    for (j, b) in code.iter().enumerate() {
        backend.write_u8(ENTRY + j as u64, *b);
    }
    translate_block(&discover_block(&*backend, ENTRY, BlockLimits::default()))
}

#[test]
fn admitting_blocks_costs_host_instructions() {
    let blocks: usize = std::env::var("AERO_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let batch: usize = std::env::var("AERO_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let options = Tier1WasmOptions {
        inline_tlb: true,
        dispatcher: true,
        ..Default::default()
    };
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
    let irs: Vec<IrBlock> = (0..blocks).map(|i| ir_for(&mut backend, i)).collect();

    // `AERO_STAGE` stops the pipeline early so differencing two block counts at each
    // stage attributes the cost: `codegen` is our own wasm emission, `module` adds
    // Wasmtime's validation and compilation, and the default adds instantiation.
    let stage = std::env::var("AERO_STAGE").unwrap_or_default();
    let mut admitted = 0usize;
    let mut bytes = 0usize;
    if stage == "irs" {
        println!("  {blocks} blocks decoded and translated only");
        return;
    }
    for chunk in irs.chunks(batch) {
        let refs: Vec<&IrBlock> = chunk.iter().collect();
        let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&refs, options);
        bytes += wasm.len();
        if stage == "codegen" {
            continue;
        }
        if stage == "module" {
            backend.compile_module_only(&wasm);
            continue;
        }
        if stage == "deserialize" {
            // Pay the compile once outside the differenced region would be better
            // still; here both budgets pay it, so the *difference* is dominated by
            // the deserialize when compile is hoisted. Instead compare against the
            // "module" stage: this does compile + serialize + deserialize, so
            // subtracting "module" leaves serialize + deserialize.
            let bytes = backend.serialize_module(&wasm);
            // Safety: produced by `serialize_module` on this very engine.
            let _ = unsafe { backend.deserialize_module(&bytes) };
            continue;
        }
        admitted += backend.add_dispatched_blocks(&wasm, chunk.len()).len();
    }
    println!("  {blocks} blocks, {batch} to a module, stage={stage:?}, {bytes} wasm bytes, {admitted} admitted");
}

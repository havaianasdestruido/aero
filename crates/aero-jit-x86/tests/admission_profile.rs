//! Admission cost with nothing else in the process.
//!
//! `compile_cost` builds its inputs by constructing a backend per block, so a
//! profile of that binary is mostly setup. This compiles one module repeatedly and
//! nothing else, so `perf` on it points at whatever admission actually spends its
//! time in.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --features tier1-inline-tlb \
//!   --test admission_profile -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};
use wasmtime::{Config, Engine, Module, Strategy};

const ENTRY: u64 = 0x100;

fn one_block_wasm() -> Vec<u8> {
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
    for (i, b) in [0x83u8, 0xc0, 0x01, 0xc3].iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions {
            inline_tlb: true,
            ..Default::default()
        },
    )
}

fn time_strategy(name: &str, strategy: Strategy, wasm: &[u8], n: usize) {
    let mut config = Config::new();
    config.wasm_simd(true);
    config.strategy(strategy);
    let engine = Engine::new(&config).expect("engine");
    let _ = Module::new(&engine, wasm).expect("warmup");

    let start = Instant::now();
    for _ in 0..n {
        let _ = Module::new(&engine, wasm).expect("module");
    }
    let elapsed = start.elapsed();
    println!(
        "  {name:>10}: {:7.1} us per module ({} bytes)",
        elapsed.as_secs_f64() * 1e6 / n as f64,
        wasm.len()
    );
}

#[test]
fn admission_by_strategy() {
    let wasm = one_block_wasm();
    let n = std::env::var("AERO_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    println!();
    time_strategy("winch", Strategy::Winch, &wasm, n);
    time_strategy("cranelift", Strategy::Cranelift, &wasm, n);
    println!();
}

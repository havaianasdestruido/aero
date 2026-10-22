//! What a block does around its body.
//!
//! Compiled code costs ~275 host instructions per block execution plus ~2.0 per ALU
//! guest instruction (`compiled_code_cost`), and blocks here average 4.28 guest
//! instructions — so almost all of the `[JIT]` bucket is that fixed term. This prints
//! the generated wasm so the fixed part can be read rather than guessed at.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test block_prologue -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x1000;

#[test]
fn print_a_small_blocks_wasm() {
    let n: usize = std::env::var("AERO_BODY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let kind = std::env::var("AERO_KIND").unwrap_or_else(|_| "alu".into());
    let mut code = Vec::new();
    for i in 0..n {
        if kind == "load" {
            code.extend_from_slice(&[0x8b, 0x03]); // mov eax, [rbx]
        } else {
            code.extend_from_slice(&[0x83, 0xc0, (i % 7 + 1) as u8]);
        }
    }
    let next = ENTRY + code.len() as u64 + 5;
    code.push(0xe9);
    code.extend_from_slice(&((ENTRY as i64 - next as i64) as i32).to_le_bytes());

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(96, 0x40_0000);
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let ir = translate_block(&discover_block(
        &backend,
        ENTRY,
        BlockLimits {
            max_insts: n + 2,
            max_bytes: 4096,
            ..Default::default()
        },
    ));
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(
        &[&ir],
        // `inline_tlb` alone is not enough to see the inline probe: `shared_mem_helpers`
        // defaults on and routes every access through an imported host function instead,
        // which is *not* what native compiles today. Set `AERO_INLINE_MEM=1` to print the
        // configuration the machine actually runs.
        Tier1WasmOptions {
            inline_tlb: true,
            dispatcher: true,
            shared_mem_helpers: std::env::var_os("AERO_INLINE_MEM").is_none(),
            single_fast_path: true,
            inline_tlb_mmio_exit: true,
            ..Default::default()
        },
    );
    let text = wasmprinter::print_bytes(&wasm).expect("print wasm");
    println!("=== {n} guest instructions, {} wasm bytes ===", wasm.len());
    println!("{text}");
}

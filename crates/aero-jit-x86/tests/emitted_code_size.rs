//! How much WASM one guest instruction turns into.
//!
//! Compiled code costs about 197 host instructions per guest instruction here, of
//! which roughly 150 per *block* is the host-to-wasm call; the rest is the body. A
//! TCG-class core runs 10-20. Winch, a baseline compiler, turns each wasm instruction
//! into a few host instructions with memory traffic either side, so the body cost is
//! roughly "wasm instructions emitted" times a small constant — which makes the count
//! the thing to know before trying to reduce it.
//!
//! This counts what the code generator emits for real guest blocks, and separates the
//! per-instruction bookkeeping from the work.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --features tier1-inline-tlb \
//!   --test emitted_code_size -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x1000;

struct Bus {
    mem: Vec<u8>,
}

impl Tier1Bus for Bus {
    fn read_u8(&self, addr: u64) -> u8 {
        self.mem.get(addr as usize).copied().unwrap_or(0)
    }
    fn write_u8(&mut self, addr: u64, value: u8) {
        if let Some(b) = self.mem.get_mut(addr as usize) {
            *b = value;
        }
    }
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Count instructions in the one function a single-block module carries.
fn count_wasm_instructions(wasm: &[u8]) -> (usize, usize) {
    let mut ops = 0usize;
    let mut body_bytes = 0usize;
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        if let Ok(wasmparser::Payload::CodeSectionEntry(body)) = payload {
            body_bytes += body.range().len();
            if let Ok(reader) = body.get_operators_reader() {
                for op in reader.into_iter() {
                    if op.is_ok() {
                        ops += 1;
                    }
                }
            }
        }
    }
    (ops, body_bytes)
}

fn measure(name: &str, hex: &str) {
    let code = unhex(hex);
    let mut mem = vec![0u8; 0x4000];
    mem[ENTRY as usize..ENTRY as usize + code.len()].copy_from_slice(&code);
    let bus = Bus { mem };

    let block = discover_block(&bus, ENTRY, BlockLimits::default());
    let guest = block.insts.len();
    let mut ir = translate_block(&block);
    let ir_before = ir.insts.len();
    let dropped = aero_jit_x86::tier1::eliminate_dead_flags(&mut ir);

    let mut out = Vec::new();
    for single in [false, true] {
        let opts = Tier1WasmOptions {
            inline_tlb: true,
            single_fast_path: single,
            ..Default::default()
        };
        let wasm = Tier1WasmCodegen::new().compile_block_with_options(&ir, opts);
        out.push(count_wasm_instructions(&wasm));
    }
    let (ops, _) = out[0];
    let (ops1, _) = out[1];
    let _ = (ir_before, dropped);
    println!(
        "  {name:<22} {guest:>3} guest | nested paths {ops:>5} ops ({:>3.0}/insn) | one fast path {ops1:>5} ops ({:>3.0}/insn) | {:.2}x",
        ops as f64 / guest as f64,
        ops1 as f64 / guest as f64,
        ops as f64 / ops1.max(1) as f64
    );
}

/// Which operations a single guest load turns into, by frequency.
#[test]
fn where_a_guest_load_goes() {
    let code = unhex("8b06eb00");
    let mut mem = vec![0u8; 0x4000];
    mem[ENTRY as usize..ENTRY as usize + code.len()].copy_from_slice(&code);
    let bus = Bus { mem };
    let block = discover_block(&bus, ENTRY, BlockLimits::default());
    let mut ir = translate_block(&block);
    let _ = aero_jit_x86::tier1::eliminate_dead_flags(&mut ir);
    let wasm = Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions { inline_tlb: true, ..Default::default() },
    );

    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm) {
        if let Ok(wasmparser::Payload::CodeSectionEntry(body)) = payload {
            if let Ok(reader) = body.get_operators_reader() {
                for op in reader.into_iter().flatten() {
                    let name = format!("{op:?}");
                    let name = name.split(&[' ', '{'][..]).next().unwrap_or("?").to_string();
                    *counts.entry(name).or_insert(0) += 1;
                }
            }
        }
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let total: usize = v.iter().map(|(_, n)| n).sum();
    println!("\n  one `mov eax,[rsi]` plus `jmp +0` = {total} wasm ops:");
    for (name, n) in v.iter().take(14) {
        println!("    {n:>4}  {name}");
    }
    println!();
}

#[test]
fn what_one_guest_instruction_costs_in_wasm() {
    println!();
    // Register-only arithmetic: the cheapest thing there is.
    measure("add eax,1 x4", "83c00183c00183c00183c001eb00");
    // A load through the inline TLB.
    measure("mov eax,[rsi] x4", "8b068b068b068b06eb00");
    // A store, which also bumps the code-version table.
    measure("mov [rsi],eax x4", "8906890689068906eb00");
    // The shape real code has: compare and branch.
    measure("cmp/jne", "4839c175f0");
    // A real captured block: SHA-256 message schedule, mostly register work.
    measure(
        "sha256 schedule",
        "83e00f458bcd448b4484104183e10f418d45fe83e00f8b4c84108bc18bd1c1e90ac1ca13c1c81133d0418bc033d1",
    );
    println!();
}

//! What compiled Tier-1 code actually costs, against both interpreters.
//!
//! This is the measurement the tiering question turns on. Widening JIT coverage is
//! only worth anything in proportion to how much cheaper the covered path is, and
//! the answer has already been counter-intuitive once: the Tier-1 *IR interpreter*
//! measures 3.9x faster than Tier-0 here and is roughly cost-neutral inside the
//! real machine, because on a flat bus Tier-0's decode and dispatch dominate while
//! in the machine memory access dominates both.
//!
//! So this file measures the third engine — x86 compiled to WASM and run by
//! Wasmtime — on the same guest loop as the other two. Run it with:
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test tier1_compiled_throughput -- --nocapture
//! ```
//!
//! It is a measurement, not a regression gate: the assertions check only that each
//! engine ran the loop and that they agree on the architectural result.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use aero_cpu_core::assist::AssistContext;
use aero_cpu_core::interp::tier0::exec::run_batch_cpu_core_with_assists;
use aero_cpu_core::interp::tier0::Tier0Config;
use aero_cpu_core::jit::runtime::JitBackend;
use aero_cpu_core::mem::FlatTestBus;
use aero_cpu_core::state::{gpr, CpuMode, CpuState};
use aero_cpu_core::CpuCore;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen};

/// A plain ALU/branch loop — the shape Tier-1 handles best.
///
/// ```text
/// add eax, ecx
/// xor ebx, eax
/// add ecx, 1
/// sub edx, 1
/// jne <entry>
/// ```
/// Build a loop whose body is `body_adds` extra `add eax, ecx` instructions ahead
/// of the four-instruction tail, so block length can be varied while the work per
/// instruction stays identical.
///
/// Varying it separates the two costs that matter: the per-*instruction* cost of
/// the compiled code, and the per-*block* cost of entering and leaving it.
/// Address of the scratch word the memory-touching variant loads and stores.
const DATA: u64 = 0x180;

fn alu_loop(body_adds: usize) -> (Vec<u8>, u64) {
    let mut code = Vec::new();
    let mem = std::env::var("AERO_BENCH_MEM").map(|v| v == "1").unwrap_or(false);
    for _ in 0..body_adds {
        if mem {
            // mov eax, [rsi] ; add eax, ecx ; mov [rdi], eax
            //
            // With `inline_tlb` off — the codegen default — each of these becomes a
            // call from WASM out to an imported host helper, where the interpreter
            // does a direct Rust call. That difference is the point of the variant.
            code.extend_from_slice(&[0x8b, 0x06]);
            code.extend_from_slice(&[0x01, 0xc8]);
            code.extend_from_slice(&[0x89, 0x07]);
        } else {
            code.extend_from_slice(&[0x01, 0xc8]); // add eax, ecx
        }
    }
    code.extend_from_slice(&[0x31, 0xc3]); // xor ebx, eax
    code.extend_from_slice(&[0x83, 0xc1, 0x01]); // add ecx, 1
    code.extend_from_slice(&[0x83, 0xea, 0x01]); // sub edx, 1
    // jne back to entry. rel8 only reaches -128, which a long body overruns — and
    // an out-of-range rel8 silently branches somewhere else rather than failing, so
    // pick the encoding by distance.
    let rel8_target = -((code.len() as i64) + 2);
    if rel8_target >= -128 {
        code.extend_from_slice(&[0x75, rel8_target as i8 as u8]);
    } else {
        let rel32 = -((code.len() as i64) + 6) as i32;
        code.extend_from_slice(&[0x0f, 0x85]);
        code.extend_from_slice(&rel32.to_le_bytes());
    }
    let insts = if mem {
        body_adds as u64 * 3 + 4
    } else {
        body_adds as u64 + 4
    };
    (code, insts)
}

/// Kept low so the backend's per-block guest-RAM snapshot stays small: it copies
/// everything below `CPU_PTR` on every block execution, which would otherwise be
/// the only thing this benchmark measures.
const ENTRY: u64 = 0x100;
/// Overridable so the snapshot's share of the per-block cost can be measured: the
/// backend copies everything below `CPU_PTR` on every block execution, so varying
/// it and re-timing separates the fixed per-block cost from the per-byte one.
fn cpu_ptr() -> i32 {
    std::env::var("AERO_BENCH_CPU_PTR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x400)
}
const MEMORY_PAGES: u32 = 8;

const ITERATIONS: u64 = 200_000;

fn body_adds() -> usize {
    std::env::var("AERO_BENCH_BODY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

fn seed(state: &mut CpuState) {
    state.rip = ENTRY;
    state.gpr[gpr::RAX as usize] = 0;
    state.gpr[gpr::RBX as usize] = 0;
    state.gpr[gpr::RCX as usize] = 1;
    state.gpr[gpr::RDX as usize] = ITERATIONS;
    state.gpr[gpr::RSI as usize] = DATA;
    state.gpr[gpr::RDI as usize] = DATA + 8;
}

fn result_of(state: &CpuState) -> (u64, u64, u64) {
    (
        state.gpr[gpr::RAX as usize],
        state.gpr[gpr::RBX as usize],
        state.gpr[gpr::RCX as usize],
    )
}

#[test]
fn compiled_tier1_vs_interpreters() {
    let (code, insts_per_iter) = alu_loop(body_adds());
    let total = ITERATIONS * insts_per_iter;

    // ── Tier-0 ───────────────────────────────────────────────────────────────
    // `exit_on_branch: false` matches `Machine::run_slice`; leaving it true would
    // measure call overhead rather than interpretation.
    let mut bus = FlatTestBus::new(0x4000);
    bus.load(ENTRY, &code);
    let mut cpu = CpuCore::new(CpuMode::Long);
    cpu.state = CpuState::new(CpuMode::Long);
    seed(&mut cpu.state);
    let cfg = Tier0Config {
        features: Default::default(),
        exit_on_branch: false,
    };
    let mut assists = AssistContext::default();

    let t0_start = Instant::now();
    let batch = run_batch_cpu_core_with_assists(&cfg, &mut assists, &mut cpu, &mut bus, total + 16);
    let t0_ns = t0_start.elapsed().as_secs_f64() * 1e9 / batch.executed as f64;
    let t0_result = result_of(&cpu.state);

    // ── Tier-1, compiled to WASM and run by Wasmtime ─────────────────────────
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(MEMORY_PAGES, cpu_ptr());
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }

    let limits = BlockLimits::default();
    assert!(
        insts_per_iter <= limits.max_insts as u64,
        "AERO_BENCH_BODY makes a {insts_per_iter}-instruction loop, but block \
         discovery stops at {} instructions. A truncated block does not end in the \
         loop branch, so it would execute once and measure nothing.",
        limits.max_insts
    );
    let block = discover_block(&backend, ENTRY, limits);
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_block(&ir);
    let table_index = backend.add_compiled_block(&wasm);

    let mut jit_cpu = CpuState::new(CpuMode::Long);
    seed(&mut jit_cpu);

    // One `execute` per loop iteration: a basic block ends at the branch, so the
    // backend's per-block costs — the RAM snapshot, and copying the architectural
    // registers in and out of linear memory — are paid every five instructions.
    // That is the real shape of short-block execution, so it stays in the timing.
    let jit_start = Instant::now();
    let mut jit_insts = 0u64;
    let mut blocks = 0u64;
    while jit_insts < total {
        let exit = backend.execute(table_index, &mut jit_cpu);
        blocks += 1;
        if exit.exit_to_interpreter {
            break;
        }
        jit_insts += insts_per_iter;
        if exit.next_rip != ENTRY {
            break;
        }
    }
    let jit_ns = jit_start.elapsed().as_secs_f64() * 1e9 / jit_insts.max(1) as f64;
    let jit_result = result_of(&jit_cpu);

    // ── Same compiled block, chained ─────────────────────────────────────────
    // One host call runs the whole loop: the architectural state stays in linear
    // memory and the blocks branch into each other, so the per-block round trip is
    // paid once instead of once per iteration.
    let mut chain_cpu = CpuState::new(CpuMode::Long);
    seed(&mut chain_cpu);
    let chain_start = Instant::now();
    let mut chain_insts = 0u64;
    let mut chain_blocks = 0u64;
    let mut chain_calls = 0u64;
    while chain_insts < total {
        // Bound the chain the way `aero-machine`'s IR chain is bounded, so this is
        // not measuring one unrealistically long host call.
        const MAX_CHAIN_BLOCKS: u64 = 4096;
        let budget = ((total - chain_insts) / insts_per_iter)
            .min(MAX_CHAIN_BLOCKS)
            .min(u32::MAX as u64) as u32;
        let outcome =
            backend.execute_chain(table_index, &mut chain_cpu, &mut |rip, _retired| {
                if rip == ENTRY {
                    Some(table_index)
                } else {
                    None
                }
            }, budget.max(1));
        let ran = outcome.blocks;
        if ran == 0 {
            break;
        }
        chain_calls += 1;
        chain_blocks += u64::from(ran);
        chain_insts += u64::from(ran) * insts_per_iter;
        if outcome.exit_to_interpreter {
            break;
        }
    }
    let chain_ns = chain_start.elapsed().as_secs_f64() * 1e9 / chain_insts.max(1) as f64;
    let chain_result = result_of(&chain_cpu);

    println!("\n  cpu_ptr={:#x}  insts/block={insts_per_iter}", cpu_ptr());
    println!("  guest instructions: tier0={} compiled={jit_insts} ({blocks} blocks)", batch.executed);
    println!("  tier0:    {t0_ns:6.2} ns/guest insn  ({:.1} MIPS)", 1000.0 / t0_ns);
    println!("  compiled: {jit_ns:6.2} ns/guest insn  ({:.1} MIPS)", 1000.0 / jit_ns);
    println!(
        "  chained:  {chain_ns:6.2} ns/guest insn  ({:.1} MIPS)  [{chain_blocks} blocks in {chain_calls} host calls]",
        1000.0 / chain_ns
    );
    println!("  speedup vs tier0: per-block {:.2}x   chained {:.2}x\n", t0_ns / jit_ns, t0_ns / chain_ns);

    assert_eq!(
        batch.executed, total,
        "Tier-0 must retire every instruction in the loop"
    );
    assert_eq!(
        jit_insts, total,
        "the compiled block must run every iteration of the loop"
    );
    assert_eq!(
        t0_result, jit_result,
        "Tier-0 and compiled Tier-1 must agree on (rax, rbx, rcx)"
    );
    assert_eq!(
        chain_insts, total,
        "the chained path must run every iteration of the loop"
    );
    assert_eq!(
        t0_result, chain_result,
        "chained execution must agree with Tier-0 on (rax, rbx, rcx)"
    );
}

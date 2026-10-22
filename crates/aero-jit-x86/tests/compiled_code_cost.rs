//! What compiled code costs per guest instruction, and what it spends it on.
//!
//! The whole-machine profile says the `[JIT]` bucket is ~72 host instructions per
//! guest instruction — the largest single term left — but says nothing about what is
//! in it. Whole-machine A/B cannot answer that either: any codegen change moves where
//! block boundaries fall, the guest takes a different path, and effects under a few
//! percent are unmeasurable (see `wiki/areas/performance.md`).
//!
//! So measure it where the guest cannot move. A block of `AERO_BODY` identical guest
//! instructions is compiled and run as a chain of `AERO_LINKS` blocks; differencing
//! two link counts cancels setup, and differencing two *body* lengths separates the
//! per-instruction cost from the per-block overhead.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test compiled_code_cost -- --nocapture
//! BIN=$(ls -t target/release/deps/compiled_code_cost-* | grep -v '\.d$' | head -1)
//! for b in 4 32; do for n in 200000 1000000; do
//!   perf stat -e instructions:u env AERO_BODY=$b AERO_LINKS=$n AERO_KIND=alu $BIN --nocapture
//! done; done
//! ```
//!
//! `AERO_KIND` picks what the body is made of — `alu` for register arithmetic, `load`
//! for guest memory reads — so the memory path can be priced against it. `AERO_OPT=1`
//! promotes the block to Cranelift first, which is what the hot blocks in a real run
//! are compiled with.

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x1000;
/// Where the `load` body reads from: inside guest RAM and page-aligned, so every
/// access hits the inline TLB rather than exercising a miss.
const DATA: u64 = 0x2000;

fn body_bytes(kind: &str, n: usize) -> Vec<u8> {
    let mut code = Vec::new();
    for i in 0..n {
        match kind {
            // mov eax, [rbx] — one guest memory read, the inline-TLB fast path.
            "load" => code.extend_from_slice(&[0x8b, 0x03]),
            // add eax, imm8 — register arithmetic and the flags that go with it.
            _ => code.extend_from_slice(&[0x83, 0xc0, (i % 7 + 1) as u8]),
        }
    }
    // jmp back to the top: a taken branch, no memory, so the chain continues.
    let next = ENTRY + code.len() as u64 + 5;
    code.push(0xe9);
    code.extend_from_slice(&((ENTRY as i64 - next as i64) as i32).to_le_bytes());
    code
}

#[test]
fn compiled_code_costs_host_instructions_per_guest_instruction() {
    let body: usize = std::env::var("AERO_BODY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let links: u32 = std::env::var("AERO_LINKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let kind = std::env::var("AERO_KIND").unwrap_or_else(|_| "alu".into());

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(96, 0x40_0000);
    for (i, b) in body_bytes(&kind, body).iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }

    let limits = BlockLimits {
        max_insts: body + 2,
        max_bytes: 4096,
        ..Default::default()
    };
    let ir = translate_block(&discover_block(&backend, ENTRY, limits));
    // `AERO_CHAIN=1` lets the block reach itself through the module's own dispatcher
    // instead of returning to the host between iterations, which is what separates the
    // per-block overhead into the part that is the engine crossing and the part that
    // is the block's own prologue and epilogue.
    let chained = std::env::var_os("AERO_CHAIN").is_some();
    // Match what the machine actually compiles with (`CompiledTier::codegen_options`),
    // or this prices a code path the real run never takes: the shared load helper is on
    // there, and the *inlined* probe rebuilds the TLB entry address where the shared one
    // computes it once.
    let options = Tier1WasmOptions {
        inline_tlb: true,
        dispatcher: true,
        shared_mem_helpers: std::env::var_os("AERO_INLINE_MEM").is_none(),
        single_fast_path: true,
        inline_tlb_mmio_exit: true,
        chain_module_id: chained.then_some(1),
        ..Default::default()
    };
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&[&ir], options);
    let idx = backend.add_dispatched_blocks(&wasm, 1)[0];
    if chained {
        backend.chain_map_insert(ENTRY, 1, 0);
        backend.set_chain_limits(u64::MAX, links);
    }
    if std::env::var_os("AERO_OPT").is_some() {
        assert!(
            backend.promote_blocks(&[idx], &wasm) == 1,
            "the Cranelift engine must be available (unset AERO_JIT_NO_TIERUP)"
        );
    }

    // Without a translator every inline-TLB probe misses and leaves wasm for a helper
    // that cannot resolve the address — which is not the memory path a real run takes.
    // Identity-map guest pages as RAM so a `load` body measures a TLB *hit*, which is
    // what nearly every access in a real run is.
    static TRANSLATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    backend.set_translator(Some(Box::new(|vaddr: u64, _access: u32| {
        TRANSLATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some((
            vaddr & !0xfff,
            aero_jit_x86::TLB_FLAG_READ
                | aero_jit_x86::TLB_FLAG_WRITE
                | aero_jit_x86::TLB_FLAG_EXEC
                | aero_jit_x86::TLB_FLAG_IS_RAM,
        ))
    })));

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.gpr[aero_types::Gpr::Rbx.as_u8() as usize] = DATA;

    backend.execute_chain(idx, &mut cpu, &mut |_rip, _retired| Some(idx), 1000);
    if chained {
        backend.set_chain_limits(u64::MAX, links);
    }
    // When chaining, the host lookup declines: everything past the first block is
    // taken inside wasm.
    let outcome = if chained {
        backend.execute_chain(idx, &mut cpu, &mut |_rip, _retired| None, links)
    } else {
        backend.execute_chain(idx, &mut cpu, &mut |_rip, _retired| Some(idx), links)
    };
    assert_eq!(outcome.blocks, links, "the chain must run to the block cap");
    println!(
        "  kind={kind} body={body} links={links} retired={} translates={}",
        outcome.retired,
        TRANSLATES.load(std::sync::atomic::Ordering::Relaxed)
    );
}

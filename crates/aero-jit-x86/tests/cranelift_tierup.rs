//! A promoted block must compute exactly what the block it replaced did.
//!
//! The tier-up runs two Wasmtime engines — Winch for admission, Cranelift for the few
//! hundred blocks that carry most of the running — over one guest memory, mapped into
//! both. Nothing about that arrangement fails loudly when it is wrong: the earlier
//! version of it silently gave the second engine a private copy of guest RAM, and the
//! only symptom was a machine that reset a million instructions later.
//!
//! So the property is worth pinning down directly: run a block, promote it, run it
//! again from the same starting state, and require the same answer.

#![cfg(not(target_arch = "wasm32"))]

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::tier1::Tier1WasmOptions;
use aero_jit_x86::{discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen};
use aero_types::Width;

/// The options the compiled tier actually uses, so the test exercises the shape that
/// ships rather than a simpler one.
fn tier_options() -> Tier1WasmOptions {
    Tier1WasmOptions {
        inline_tlb: true,
        single_fast_path: true,
        shared_mem_helpers: true,
        dispatcher: true,
        ..Default::default()
    }
}

fn run_once(backend: &mut WasmtimeBackend<CpuState>, table_index: u32, start: &CpuState) -> CpuState {
    use aero_jit_x86::compiled_backend::CompiledBackend;
    let mut cpu = start.clone();
    // A chain of one: enough to exercise entry, the block, and the exit sync.
    let outcome = CompiledBackend::run_chain(
        backend,
        table_index,
        &mut cpu,
        &mut |_rip, _retired| None,
        1,
    );
    assert!(outcome.retired > 0, "the block retired nothing");
    cpu
}

#[test]
fn a_promoted_block_computes_what_the_original_did() {
    // add rax, rbx ; xor rcx, rdx ; sub rax, rcx ; ret
    let code = [
        0x48, 0x01, 0xd8, // add rax, rbx
        0x48, 0x31, 0xd1, // xor rcx, rdx
        0x48, 0x29, 0xc8, // sub rax, rcx
        0xc3, // ret
    ];
    let entry = 0x1000u64;

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(64, 0x20000);
    if !backend.can_optimize() {
        // The Cranelift half is off; nothing to compare against.
        return;
    }
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(entry + i as u64, *b);
    }

    let block = discover_block(&backend, entry, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&[&ir], tier_options());
    let table_index = backend.add_dispatched_blocks(&wasm, 1)[0];

    let mut start = CpuState::new(aero_cpu_core::state::CpuMode::Long);
    start.rip = entry;
    start.write_gpr64(aero_cpu_core::state::gpr::RAX, 0x1234_5678_9abc_def0);
    start.write_gpr64(aero_cpu_core::state::gpr::RBX, 0x0fed_cba9_8765_4321);
    start.write_gpr64(aero_cpu_core::state::gpr::RCX, 0xaaaa_5555_aaaa_5555);
    start.write_gpr64(aero_cpu_core::state::gpr::RDX, 0x5555_aaaa_5555_aaaa);
    // Somewhere for RET to return to.
    start.write_gpr64(aero_cpu_core::state::gpr::RSP, 0x8000);
    backend.write(0x8000, Width::W64, 0x2000);

    let before = run_once(&mut backend, table_index, &start);

    // Promote it and run the same input again. The table index is deliberately
    // unchanged: everything that referred to the block still does.
    assert!(
        backend.promote_blocks(&[table_index], &wasm) == 1,
        "the block was not promoted"
    );
    backend.write(0x8000, Width::W64, 0x2000);
    let after = run_once(&mut backend, table_index, &start);

    assert_eq!(before.rip, after.rip, "RIP differs after promotion");
    for i in 0..16 {
        assert_eq!(
            before.gpr[i], after.gpr[i],
            "GPR {i} differs after promotion: {:#x} then {:#x}",
            before.gpr[i], after.gpr[i]
        );
    }
    assert_eq!(
        before.rflags(),
        after.rflags(),
        "RFLAGS differ after promotion"
    );
}

/// The same, for a block that reads and writes guest memory — the case where the two
/// engines having separate copies of it would show.
#[test]
fn a_promoted_block_sees_the_same_guest_memory() {
    // mov rax, [rsi] ; add rax, 1 ; mov [rsi], rax ; ret
    let code = [
        0x48, 0x8b, 0x06, // mov rax, [rsi]
        0x48, 0x83, 0xc0, 0x01, // add rax, 1
        0x48, 0x89, 0x06, // mov [rsi], rax
        0xc3, // ret
    ];
    let entry = 0x1000u64;

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(64, 0x20000);
    if !backend.can_optimize() {
        return;
    }
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(entry + i as u64, *b);
    }

    let block = discover_block(&backend, entry, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_blocks_with_options(&[&ir], tier_options());
    let table_index = backend.add_dispatched_blocks(&wasm, 1)[0];

    let mut start = CpuState::new(aero_cpu_core::state::CpuMode::Long);
    start.rip = entry;
    start.write_gpr64(aero_cpu_core::state::gpr::RSI, 0x9000);
    start.write_gpr64(aero_cpu_core::state::gpr::RSP, 0x8000);

    backend.write(0x8000, Width::W64, 0x2000);
    backend.write(0x9000, Width::W64, 41);
    let before = run_once(&mut backend, table_index, &start);
    let mem_before = backend.read(0x9000, Width::W64);

    assert!(backend.promote_blocks(&[table_index], &wasm) == 1);
    backend.write(0x8000, Width::W64, 0x2000);
    backend.write(0x9000, Width::W64, 41);
    let after = run_once(&mut backend, table_index, &start);
    let mem_after = backend.read(0x9000, Width::W64);

    assert_eq!(
        before.gpr[0], after.gpr[0],
        "RAX differs after promotion: the promoted block read different memory"
    );
    assert_eq!(
        mem_before, mem_after,
        "guest memory differs after promotion: the two engines are not sharing it"
    );
    assert_eq!(mem_after, 42, "the block did not write what it should have");
}

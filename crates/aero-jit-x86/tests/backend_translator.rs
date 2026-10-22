//! The inline TLB, driven by an embedder-supplied translator.
//!
//! Compiled blocks reach guest RAM through a direct load into linear memory, which
//! is what makes a compiled tier worth more than an interpreter. Getting there
//! needs the guest's *own* page tables: the backend's built-in behaviour identity
//! maps, which is only correct for a guest that is not paging, and Windows pages
//! from very early on.
//!
//! These cover the seam that lets a real MMU supply the mapping instead — that a
//! non-identity translation is actually followed, that the result is cached, and
//! that flushing the salt makes it be asked again.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use aero_cpu_core::jit::runtime::JitBackend;
use aero_cpu_core::state::{CpuMode, CpuState};
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
    TLB_FLAG_EXEC, TLB_FLAG_IS_RAM, TLB_FLAG_READ, TLB_FLAG_WRITE,
};
use aero_types::Gpr;

/// `mov eax, [rsi]` then `ret`. The load is the whole point.
const LOAD_AND_RET: &[u8] = &[0x8b, 0x06, 0xc3];

const ENTRY: u64 = 0x100;
/// The virtual page the guest reads from.
const VIRT: u64 = 0x4_0000;
/// The physical page it is actually mapped to — deliberately not `VIRT`.
const PHYS: u64 = 0x2_0000;
const RET_ADDR: u64 = 0x9000;
const STACK: u64 = 0x8000;

fn backend_with_block(inline_tlb: bool) -> (WasmtimeBackend<CpuState>, u32) {
    // 16 pages: enough for the code, the stack, and both the virtual and physical
    // pages used below.
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    for (i, b) in LOAD_AND_RET.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    // Return address for RET.
    backend.write(STACK, aero_jit_x86::Width::W64, RET_ADDR);
    // The value lives at the *physical* page. If translation is ignored, the block
    // reads the virtual page instead, which holds a different value.
    backend.write(PHYS, aero_jit_x86::Width::W32, 0xcafe_f00d);
    backend.write(VIRT, aero_jit_x86::Width::W32, 0xdead_dead);

    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions {
            inline_tlb,
            ..Default::default()
        },
    );
    let idx = backend.add_compiled_block(&wasm);
    (backend, idx)
}

fn cpu_reading_virt() -> CpuState {
    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.gpr[Gpr::Rsi.as_u8() as usize] = VIRT;
    cpu.gpr[Gpr::Rsp.as_u8() as usize] = STACK;
    cpu
}

#[test]
fn a_non_identity_mapping_is_followed() {
    let (mut backend, idx) = backend_with_block(true);

    let calls = Rc::new(RefCell::new(Vec::<u64>::new()));
    let seen = calls.clone();
    backend.set_translator(Some(Box::new(move |vaddr, _access| {
        seen.borrow_mut().push(vaddr);
        // Map the one page the guest reads; identity-map everything else so the
        // block's own fetch and stack still work.
        let base = if vaddr & !0xfff == VIRT { PHYS } else { vaddr & !0xfff };
        Some((
            base,
            TLB_FLAG_READ | TLB_FLAG_WRITE | TLB_FLAG_EXEC | TLB_FLAG_IS_RAM,
        ))
    })));

    let mut cpu = cpu_reading_virt();
    backend.execute(idx, &mut cpu);

    assert_eq!(
        cpu.gpr[Gpr::Rax.as_u8() as usize] & 0xffff_ffff,
        0xcafe_f00d,
        "the load must follow the translation to {PHYS:#x}, not read {VIRT:#x} directly"
    );
    assert!(
        calls.borrow().iter().any(|v| v & !0xfff == VIRT),
        "the translator must have been asked about the page being read; saw {:?}",
        calls.borrow()
    );
}

#[test]
fn a_translation_is_cached_and_flushing_asks_again() {
    let (mut backend, idx) = backend_with_block(true);

    let count = Rc::new(RefCell::new(0usize));
    let n = count.clone();
    backend.set_translator(Some(Box::new(move |vaddr, _access| {
        if vaddr & !0xfff == VIRT {
            *n.borrow_mut() += 1;
        }
        let base = if vaddr & !0xfff == VIRT { PHYS } else { vaddr & !0xfff };
        Some((
            base,
            TLB_FLAG_READ | TLB_FLAG_WRITE | TLB_FLAG_EXEC | TLB_FLAG_IS_RAM,
        ))
    })));

    let mut cpu = cpu_reading_virt();
    backend.execute(idx, &mut cpu);
    let after_first = *count.borrow();
    assert_eq!(after_first, 1, "the first access must miss and translate");

    let mut cpu = cpu_reading_virt();
    backend.execute(idx, &mut cpu);
    assert_eq!(
        *count.borrow(),
        after_first,
        "the second access must hit the cached entry, not translate again"
    );

    // A CR3 write or INVLPG has to make compiled code stop trusting what it cached.
    backend.flush_tlb();
    let mut cpu = cpu_reading_virt();
    backend.execute(idx, &mut cpu);
    assert_eq!(
        *count.borrow(),
        after_first + 1,
        "flushing the salt must force a re-translation"
    );
}

#[test]
fn without_a_translator_the_backend_still_identity_maps() {
    // The existing behaviour has to survive: embedders that do not page rely on it,
    // and every other test in this crate runs that way.
    let (mut backend, idx) = backend_with_block(true);
    let mut cpu = cpu_reading_virt();
    backend.execute(idx, &mut cpu);
    assert_eq!(
        cpu.gpr[Gpr::Rax.as_u8() as usize] & 0xffff_ffff,
        0xdead_dead,
        "with no translator installed the virtual address is the physical one"
    );
}

/// A load that straddles a page boundary, where the two pages are not adjacent.
///
/// `inline_tlb_cross_page_fastpath` keeps such an access on the inline path by
/// splitting it and translating each page. That is only correct if the second page
/// really is translated: guest pages either side of a boundary are adjacent in
/// virtual space and almost never in physical space, so an implementation that
/// assumes contiguity reads the right first bytes and the wrong rest — and passes any
/// test whose two pages happen to be neighbours.
#[test]
fn a_cross_page_load_translates_both_pages() {
    // `mov eax, [rsi]` then `ret`, with RSI two bytes below a page boundary.
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(16, 0x8_0000);
    for (i, b) in LOAD_AND_RET.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    backend.write(STACK, aero_jit_x86::Width::W64, RET_ADDR);

    // Virtual 0x5000 and 0x6000 map to physical 0x2000 and 0x1000 — deliberately in
    // the opposite order, so a contiguity assumption cannot accidentally be right.
    const V_LO: u64 = 0x5000;
    const V_HI: u64 = 0x6000;
    const P_LO: u64 = 0x2000;
    const P_HI: u64 = 0x1000;
    // Low half of the dword lives at the end of the low physical page...
    backend.write(P_LO + 0xffe, aero_jit_x86::Width::W16, 0xbeef);
    // ...and the high half at the start of the other one.
    backend.write(P_HI, aero_jit_x86::Width::W16, 0xdead);
    // Poison the page that follows P_LO physically, which is what a contiguous read
    // would pick up.
    backend.write(P_LO + 0x1000, aero_jit_x86::Width::W16, 0x5555);

    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions {
            inline_tlb: true,
            inline_tlb_cross_page_fastpath: true,
            ..Default::default()
        },
    );
    let idx = backend.add_compiled_block(&wasm);

    backend.set_translator(Some(Box::new(move |vaddr, _access| {
        let page = match vaddr & !0xfff {
            V_LO => P_LO,
            V_HI => P_HI,
            other => other,
        };
        Some((
            page,
            TLB_FLAG_READ | TLB_FLAG_WRITE | TLB_FLAG_EXEC | TLB_FLAG_IS_RAM,
        ))
    })));

    let mut cpu = CpuState::new(CpuMode::Long);
    cpu.rip = ENTRY;
    cpu.gpr[Gpr::Rsi.as_u8() as usize] = V_LO + 0xffe;
    cpu.gpr[Gpr::Rsp.as_u8() as usize] = STACK;
    backend.execute(idx, &mut cpu);

    assert_eq!(
        cpu.gpr[Gpr::Rax.as_u8() as usize] & 0xffff_ffff,
        0xdead_beef,
        "the high half must come from the page {V_HI:#x} translates to, not from \
         whatever physically follows {P_LO:#x}"
    );
}

//! Shared instruction emission helpers for the inline JIT TLB fast-path.
//!
//! Tier-1 and Tier-2 WASM codegens both emit identical inline-TLB probes and related helpers
//! (direct-mapped TLB tag check, RAM address computation, and code-version bumping). Keep the
//! sequences here to avoid semantic drift between tiers.

use wasm_encoder::{BlockType, Function, Instruction, MemArg, ValType};

use crate::jit_ctx::JitContext;

#[derive(Debug, Clone, Copy)]
pub(crate) struct InlineTlbLocals {
    pub(crate) cpu_ptr: u32,
    pub(crate) jit_ctx_ptr: u32,
    pub(crate) ram_base: u32,
    pub(crate) tlb_salt: u32,
    pub(crate) scratch_vaddr: u32,
    pub(crate) scratch_vpn: u32,
    pub(crate) scratch_tlb_data: u32,
    pub(crate) code_version_table_ptr: Option<u32>,
    pub(crate) code_version_table_len: Option<u32>,
}

/// Probe the TLB, filling the entry on a miss or a permission mismatch.
///
/// The entry address is computed once and kept, rather than rebuilt for the tag load
/// and again for the data load.
///
/// The permission re-check has to stay here and cannot be folded into a caller's
/// fast-path predicate. An entry caches only the permission actually exercised, so a
/// page first touched by a read carries no write bit — and a predicate that merely
/// *fails* on that never refills the entry, sending every later store to the slow
/// helper for as long as the page stays cached. Measured: doing that cost 50%.
pub(crate) fn emit_probe_and_fill(
    f: &mut Function,
    depth: &mut u32,
    locals: InlineTlbLocals,
    mmu_translate_fn: u32,
    access_code: i32,
    required_flag: u64,
    entry_addr_local: u32,
) {
    // vpn = vaddr >> 12
    f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
    f.instruction(&Instruction::I64Const(crate::PAGE_SHIFT as i64));
    f.instruction(&Instruction::I64ShrU);
    f.instruction(&Instruction::LocalSet(locals.scratch_vpn));

    // The entry address, once, and kept as an i32 — the width it is used at.
    emit_tlb_entry_addr(f, locals);
    f.instruction(&Instruction::I64ExtendI32U);
    f.instruction(&Instruction::LocalSet(entry_addr_local));

    f.instruction(&Instruction::LocalGet(entry_addr_local));
    f.instruction(&Instruction::I32WrapI64);
    f.instruction(&Instruction::I64Load(tlb_entry_memarg(0))); // tag
    f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
    f.instruction(&Instruction::LocalGet(locals.tlb_salt));
    f.instruction(&Instruction::I64Xor);
    f.instruction(&Instruction::I64Const(1));
    f.instruction(&Instruction::I64Or);
    f.instruction(&Instruction::I64Eq);

    f.instruction(&Instruction::If(BlockType::Empty));
    *depth += 1;
    {
        f.instruction(&Instruction::LocalGet(entry_addr_local));
        f.instruction(&Instruction::I32WrapI64);
        f.instruction(&Instruction::I64Load(tlb_entry_memarg(8))); // data
        f.instruction(&Instruction::LocalSet(locals.scratch_tlb_data));
    }
    f.instruction(&Instruction::Else);
    {
        emit_mmu_translate(f, locals, mmu_translate_fn, access_code);
    }
    f.instruction(&Instruction::End);
    *depth -= 1;


    // Cached with the wrong permission: translate again so the entry is refilled with
    // this access's rights, rather than failing every time from now on.
    f.instruction(&Instruction::LocalGet(locals.scratch_tlb_data));
    f.instruction(&Instruction::I64Const(required_flag as i64));
    f.instruction(&Instruction::I64And);
    f.instruction(&Instruction::I64Eqz);
    f.instruction(&Instruction::If(BlockType::Empty));
    *depth += 1;
    {
        emit_mmu_translate(f, locals, mmu_translate_fn, access_code);
    }
    f.instruction(&Instruction::End);
    *depth -= 1;
}

pub(crate) fn emit_translate_and_cache(
    f: &mut Function,
    depth: &mut u32,
    locals: InlineTlbLocals,
    mmu_translate_fn: u32,
    access_code: i32,
    required_flag: u64,
) {
    // vpn = vaddr >> 12
    f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
    f.instruction(&Instruction::I64Const(crate::PAGE_SHIFT as i64));
    f.instruction(&Instruction::I64ShrU);
    f.instruction(&Instruction::LocalSet(locals.scratch_vpn));

    // Two branches here, not one. Folding the permission test into the tag test —
    // reading `data` speculatively so both are in hand at once — was written and
    // measured at **43.7 host instructions a load against 43.7**, so it bought a
    // reader the obligation to check that a speculative read of another page's entry
    // is safe, in exchange for nothing.
    //
    // The same goes for the arithmetic: making this sequence's index math i32 instead
    // of i64-then-wrap, and its multiply a shift, moved it by 0.1. Cranelift already
    // does that. What is left in a guest memory access — 43.7 against 2.0 for an ALU
    // instruction — is not something rearranging this emission reaches.
    emit_tlb_entry_addr(f, locals);
    f.instruction(&Instruction::I64Load(tlb_entry_memarg(0))); // tag
    f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
    f.instruction(&Instruction::LocalGet(locals.tlb_salt));
    f.instruction(&Instruction::I64Xor);
    // expect_tag = (vpn ^ salt) | 1; keep 0 reserved for invalidation.
    f.instruction(&Instruction::I64Const(1));
    f.instruction(&Instruction::I64Or);
    f.instruction(&Instruction::I64Eq);

    f.instruction(&Instruction::If(BlockType::Empty));
    *depth += 1;
    {
        // Hit: load `data` from the entry.
        emit_tlb_entry_addr(f, locals);
        f.instruction(&Instruction::I64Load(tlb_entry_memarg(8))); // data
        f.instruction(&Instruction::LocalSet(locals.scratch_tlb_data));
    }
    f.instruction(&Instruction::Else);
    {
        // Miss: call the translation helper (expected to fill the entry).
        emit_mmu_translate(f, locals, mmu_translate_fn, access_code);
    }
    f.instruction(&Instruction::End);
    *depth -= 1;

    // Permission check: if the cached entry doesn't permit this access, go slow-path.
    f.instruction(&Instruction::LocalGet(locals.scratch_tlb_data));
    f.instruction(&Instruction::I64Const(required_flag as i64));
    f.instruction(&Instruction::I64And);
    f.instruction(&Instruction::I64Eqz);

    f.instruction(&Instruction::If(BlockType::Empty));
    *depth += 1;
    {
        emit_mmu_translate(f, locals, mmu_translate_fn, access_code);
    }
    f.instruction(&Instruction::End);
    *depth -= 1;
}

/// Body of a shared per-module load helper.
///
/// Signature: `(cpu_ptr i32, jit_ctx_ptr i32, ram_base i64, tlb_salt i64, vaddr i64) -> i64`.
///
/// Every memory access used to inline this whole sequence. That is ~248 bytes of
/// WASM per access, and a block's admission cost tracks its emitted bytes almost
/// exactly (~390ns per byte through Winch) — so with `ret` alone making a load of
/// every block, the inline form set the price of admitting anything. Emitting it
/// once per module and calling it costs a handful of bytes per site instead.
pub(crate) fn emit_shared_load_helper(
    f: &mut Function,
    mmu_translate_fn: u32,
    slow_read_fn: u32,
    size_bytes: u32,
) {
    // Params 0..4, then three scratch locals.
    let locals = InlineTlbLocals {
        cpu_ptr: 0,
        jit_ctx_ptr: 1,
        ram_base: 2,
        tlb_salt: 3,
        scratch_vaddr: 4,
        scratch_vpn: 5,
        scratch_tlb_data: 6,
        code_version_table_ptr: None,
        code_version_table_len: None,
    };
    let entry_addr_local = 7;
    let mut depth = 0u32;

    emit_probe_and_fill(
        f,
        &mut depth,
        locals,
        mmu_translate_fn,
        crate::abi::MMU_ACCESS_READ,
        crate::TLB_FLAG_READ,
        entry_addr_local,
    );

    // Backed by RAM ...
    f.instruction(&Instruction::LocalGet(locals.scratch_tlb_data));
    f.instruction(&Instruction::I64Const(crate::TLB_FLAG_IS_RAM as i64));
    f.instruction(&Instruction::I64And);
    f.instruction(&Instruction::I64Const(0));
    f.instruction(&Instruction::I64Ne);

    // ... and inside one page. An access of one byte never crosses.
    if size_bytes > 1 {
        f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
        f.instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
        f.instruction(&Instruction::I64And);
        f.instruction(&Instruction::I64Const(
            (crate::PAGE_OFFSET_MASK - (size_bytes as u64 - 1)) as i64,
        ));
        f.instruction(&Instruction::I64LeU);
        f.instruction(&Instruction::I32And);
    }

    f.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    {
        emit_compute_ram_addr(f, locals);
        match size_bytes {
            1 => f.instruction(&Instruction::I64Load8U(memarg(0, 0))),
            2 => f.instruction(&Instruction::I64Load16U(memarg(0, 1))),
            4 => f.instruction(&Instruction::I64Load32U(memarg(0, 2))),
            _ => f.instruction(&Instruction::I64Load(memarg(0, 3))),
        };
    }
    f.instruction(&Instruction::Else);
    {
        f.instruction(&Instruction::LocalGet(locals.cpu_ptr));
        f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
        f.instruction(&Instruction::Call(slow_read_fn));
        if size_bytes != 8 {
            // The narrow helpers return i32.
            f.instruction(&Instruction::I64ExtendI32U);
            let mask = (1u64 << (size_bytes * 8)) - 1;
            f.instruction(&Instruction::I64Const(mask as i64));
            f.instruction(&Instruction::I64And);
        }
    }
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End); // end function
}

pub(crate) fn emit_compute_ram_addr(f: &mut Function, locals: InlineTlbLocals) {
    // paddr = (phys_base & !0xFFF) | (vaddr & 0xFFF)
    f.instruction(&Instruction::LocalGet(locals.scratch_tlb_data));
    f.instruction(&Instruction::I64Const(crate::PAGE_BASE_MASK as i64));
    f.instruction(&Instruction::I64And);

    f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
    f.instruction(&Instruction::I64Const(crate::PAGE_OFFSET_MASK as i64));
    f.instruction(&Instruction::I64And);
    f.instruction(&Instruction::I64Or);

    // Q35 high-memory remap: translate physical addresses in the high-RAM region back into the
    // contiguous RAM backing store used by the wasm runtime.
    //
    // If paddr >= 4GiB:
    //   paddr = LOW_RAM_END + (paddr - 4GiB)
    const HIGH_RAM_BASE: i64 = 0x1_0000_0000;
    const LOW_RAM_END: i64 = aero_pc_constants::PCIE_ECAM_BASE as i64;
    f.instruction(&Instruction::LocalTee(locals.scratch_vpn));
    f.instruction(&Instruction::I64Const(HIGH_RAM_BASE));
    f.instruction(&Instruction::I64GeU);
    f.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    {
        f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
        f.instruction(&Instruction::I64Const(HIGH_RAM_BASE));
        f.instruction(&Instruction::I64Sub);
        f.instruction(&Instruction::I64Const(LOW_RAM_END));
        f.instruction(&Instruction::I64Add);
    }
    f.instruction(&Instruction::Else);
    {
        f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
    }
    f.instruction(&Instruction::End);

    // wasm_addr = ram_base + paddr
    f.instruction(&Instruction::LocalGet(locals.ram_base));
    f.instruction(&Instruction::I64Add);
    f.instruction(&Instruction::I32WrapI64);
}

pub(crate) fn emit_bump_code_version_fastpath(
    f: &mut Function,
    locals: InlineTlbLocals,
    memory_shared: bool,
) {
    let code_version_table_ptr = locals
        .code_version_table_ptr
        .expect("code version table locals disabled");
    let code_version_table_len = locals
        .code_version_table_len
        .expect("code version table locals disabled");

    // If the runtime hasn't configured a version table, skip.
    f.instruction(&Instruction::LocalGet(code_version_table_len));
    f.instruction(&Instruction::I64Eqz);
    f.instruction(&Instruction::If(BlockType::Empty));
    f.instruction(&Instruction::Else);
    {
        // Compute the physical page number for this store.
        f.instruction(&Instruction::LocalGet(locals.scratch_tlb_data));
        f.instruction(&Instruction::I64Const(crate::PAGE_BASE_MASK as i64));
        f.instruction(&Instruction::I64And);
        f.instruction(&Instruction::I64Const(crate::PAGE_SHIFT as i64));
        f.instruction(&Instruction::I64ShrU); // -> page (i64)
        f.instruction(&Instruction::LocalTee(locals.scratch_vpn));

        // Bounds check: page < table_len.
        f.instruction(&Instruction::LocalGet(code_version_table_len));
        f.instruction(&Instruction::I64LtU);

        f.instruction(&Instruction::If(BlockType::Empty));
        {
            // addr = table_ptr + page * 4
            f.instruction(&Instruction::LocalGet(code_version_table_ptr));
            f.instruction(&Instruction::LocalGet(locals.scratch_vpn));

            f.instruction(&Instruction::I64Const(4));
            f.instruction(&Instruction::I64Mul);
            f.instruction(&Instruction::I64Add);
            f.instruction(&Instruction::LocalSet(locals.scratch_vpn));

            // table[page] += 1
            //
            // This is wrapping arithmetic: both `i32.add` and `i32.atomic.rmw.add` wrap on
            // overflow (`0xffff_ffff + 1 == 0`), matching the runtime and JS `Atomics.add`
            // semantics.
            //
            // Note: when the module imports a shared memory, we must use atomic operations for
            // correctness under concurrency. Non-atomic `i32.load`/`i32.store` sequences can lose
            // increments when multiple agents bump the same entry concurrently.
            f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
            f.instruction(&Instruction::I32WrapI64);
            if memory_shared {
                f.instruction(&Instruction::I32Const(1));
                f.instruction(&Instruction::I32AtomicRmwAdd(memarg(0, 2)));
                // `i32.atomic.rmw.add` returns the old value; we only care that the increment
                // happens.
                f.instruction(&Instruction::Drop);
            } else {
                f.instruction(&Instruction::I32Load(memarg(0, 2)));
                f.instruction(&Instruction::I32Const(1));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::I64ExtendI32U);
                f.instruction(&Instruction::LocalSet(locals.scratch_vaddr));

                f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
                f.instruction(&Instruction::I32WrapI64);
                f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
                f.instruction(&Instruction::I32WrapI64);
                f.instruction(&Instruction::I32Store(memarg(0, 2)));
            }
        }
        f.instruction(&Instruction::End);
    }
    f.instruction(&Instruction::End);
}

/// `jit_ctx_ptr + ((vpn & mask) << log2(ENTRY_SIZE))`, as an i32 address.
///
/// `JitContext::TLB_OFFSET` is deliberately *not* added here: it is a constant, so it
/// belongs in the load's `memarg` offset — see [`tlb_entry_memarg`]. And the index
/// arithmetic is done in i32 rather than i64-then-wrap, which is sound because the
/// mask keeps only low bits: masking after `i32.wrap_i64` gives the same answer as
/// masking the i64 and wrapping afterwards.
///
/// Eleven i64 operations became eight i32 ones. A guest memory access is the most
/// expensive thing a block does — 43.6 host instructions against 2.0 for an ALU
/// instruction — so what this sequence costs is worth caring about.
fn emit_tlb_entry_addr(f: &mut Function, locals: InlineTlbLocals) {
    const _: () = assert!(crate::JIT_TLB_ENTRY_SIZE.is_power_of_two());
    f.instruction(&Instruction::LocalGet(locals.scratch_vpn));
    f.instruction(&Instruction::I32WrapI64);
    f.instruction(&Instruction::I32Const(crate::JIT_TLB_INDEX_MASK as i32));
    f.instruction(&Instruction::I32And);
    f.instruction(&Instruction::I32Const(
        crate::JIT_TLB_ENTRY_SIZE.trailing_zeros() as i32,
    ));
    f.instruction(&Instruction::I32Shl);
    f.instruction(&Instruction::LocalGet(locals.jit_ctx_ptr));
    f.instruction(&Instruction::I32Add);
}

/// Memarg for a field of the TLB entry whose address `emit_tlb_entry_addr` left on the
/// stack. `field` is 0 for the tag and 8 for the data.
fn tlb_entry_memarg(field: u32) -> MemArg {
    memarg(JitContext::TLB_OFFSET + field, 3)
}

fn emit_mmu_translate(
    f: &mut Function,
    locals: InlineTlbLocals,
    mmu_translate_fn: u32,
    access_code: i32,
) {
    f.instruction(&Instruction::LocalGet(locals.cpu_ptr));
    f.instruction(&Instruction::LocalGet(locals.jit_ctx_ptr));
    f.instruction(&Instruction::LocalGet(locals.scratch_vaddr));
    f.instruction(&Instruction::I32Const(access_code));
    f.instruction(&Instruction::Call(mmu_translate_fn));
    f.instruction(&Instruction::LocalSet(locals.scratch_tlb_data));
}

fn memarg(offset: u32, align: u32) -> MemArg {
    MemArg {
        offset: offset as u64,
        align,
        memory_index: 0,
    }
}

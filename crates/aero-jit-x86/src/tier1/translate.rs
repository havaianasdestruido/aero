use aero_types::Cond;
use aero_types::Flag;
use aero_x86::tier1::BitOp;
use super::block::{BasicBlock, BlockEndKind};
use super::ir::{BinOp, GuestReg, IrBlock, IrBuilder, IrTerminator, SegBase, ValueId};
use aero_types::{FlagSet, Gpr, Width};
use aero_x86::tier1::{
    Address, AluOp, DecodedInst, InstKind, Operand, Reg, SegmentPrefix, ShiftOp,
};

fn gpr_reg(reg: Reg) -> GuestReg {
    GuestReg::Gpr {
        reg: reg.gpr,
        width: reg.width,
        high8: reg.high8,
    }
}

fn mask_addr(b: &mut IrBuilder, addr: ValueId, addr_mask: Option<ValueId>) -> ValueId {
    if let Some(mask) = addr_mask {
        b.binop(BinOp::And, Width::W64, addr, mask, FlagSet::EMPTY)
    } else {
        addr
    }
}

fn emit_address(
    b: &mut IrBuilder,
    inst: &DecodedInst,
    addr: &Address,
    addr_mask: Option<ValueId>,
) -> ValueId {
    let next_rip = inst.next_rip();
    let mut acc = if addr.rip_relative {
        b.const_int(Width::W64, next_rip)
    } else if let Some(base) = addr.base {
        b.read_reg(GuestReg::Gpr {
            reg: base,
            width: Width::W64,
            high8: false,
        })
    } else {
        b.const_int(Width::W64, 0)
    };

    if let Some(index) = addr.index {
        let idx = b.read_reg(GuestReg::Gpr {
            reg: index,
            width: Width::W64,
            high8: false,
        });
        let scaled = match addr.scale {
            1 => idx,
            2 => {
                let sh = b.const_int(Width::W64, 1);
                b.binop(BinOp::Shl, Width::W64, idx, sh, FlagSet::EMPTY)
            }
            4 => {
                let sh = b.const_int(Width::W64, 2);
                b.binop(BinOp::Shl, Width::W64, idx, sh, FlagSet::EMPTY)
            }
            8 => {
                let sh = b.const_int(Width::W64, 3);
                b.binop(BinOp::Shl, Width::W64, idx, sh, FlagSet::EMPTY)
            }
            _ => panic!("invalid scale {}", addr.scale),
        };
        acc = b.binop(BinOp::Add, Width::W64, acc, scaled, FlagSet::EMPTY);
    }

    if addr.disp != 0 {
        let disp = b.const_int(Width::W64, addr.disp as i64 as u64);
        acc = b.binop(BinOp::Add, Width::W64, acc, disp, FlagSet::EMPTY);
    }

    // A segment override adds the segment's base. In long mode only FS and GS have
    // one, and the decoder only reports those. The base is added *after* the
    // address-size mask, because it is not part of the effective address the mask
    // applies to — `gs:[0x88]` in 64-bit mode reaches gs_base+0x88, not a wrapped sum.
    let acc = mask_addr(b, acc, addr_mask);
    match addr.segment {
        None => acc,
        Some(seg) => {
            let base = b.read_reg(GuestReg::SegBase(match seg {
                SegmentPrefix::Fs => SegBase::Fs,
                SegmentPrefix::Gs => SegBase::Gs,
            }));
            b.binop(BinOp::Add, Width::W64, acc, base, FlagSet::EMPTY)
        }
    }
}

fn emit_read_operand(
    b: &mut IrBuilder,
    inst: &DecodedInst,
    op: &Operand,
    width: Width,
    addr_mask: Option<ValueId>,
) -> ValueId {
    match op {
        Operand::Imm(v) => b.const_int(width, *v),
        Operand::Reg(r) => {
            debug_assert_eq!(r.width, width);
            b.read_reg(gpr_reg(*r))
        }
        Operand::Mem(addr) => {
            let a = emit_address(b, inst, addr, addr_mask);
            b.load(width, a)
        }
    }
}

fn emit_write_operand(
    b: &mut IrBuilder,
    inst: &DecodedInst,
    op: &Operand,
    width: Width,
    value: ValueId,
    addr_mask: Option<ValueId>,
) {
    match op {
        Operand::Imm(_) => panic!("cannot write to immediate"),
        Operand::Reg(r) => {
            debug_assert_eq!(r.width, width);
            b.write_reg(gpr_reg(*r), value)
        }
        Operand::Mem(addr) => {
            let a = emit_address(b, inst, addr, addr_mask);
            b.store(width, a, value);
        }
    }
}

fn to_binop(op: AluOp) -> BinOp {
    match op {
        AluOp::Add => BinOp::Add,
        AluOp::Adc => BinOp::Adc,
        AluOp::Sub => BinOp::Sub,
        AluOp::Sbb => BinOp::Sbb,
        AluOp::And => BinOp::And,
        AluOp::Or => BinOp::Or,
        AluOp::Xor => BinOp::Xor,
        AluOp::Shl => BinOp::Shl,
        AluOp::Shr => BinOp::Shr,
        AluOp::Sar => BinOp::Sar,
        AluOp::Imul => BinOp::Mul,
    }
}

/// Flags an ALU operation writes.
///
/// Everything here writes the usual six except `imul`, which writes CF and OF and
/// leaves SF, ZF, AF and PF alone — undefined architecturally, and untouched by
/// Tier-0, which is the choice that counts because a guest can set a flag in one
/// tier and read it in another. Declaring more than an instruction writes is not
/// harmless: dead-flag elimination reads these sets as what gets overwritten, so an
/// over-broad claim deletes the write that really produced the flag.
fn alu_flag_set(op: AluOp) -> FlagSet {
    match op {
        AluOp::Imul => FlagSet::CF.union(FlagSet::OF),
        _ => FlagSet::ALU,
    }
}

/// Flags a shift or rotate writes.
///
/// AF is architecturally undefined after `SHL`/`SHR`/`SAR`, so any value is legal
/// in isolation — but Tier-0 clears it, and a tiered CPU has to agree with itself:
/// a guest can set flags in one tier and read them in another, and on real hardware
/// that is one machine. Tier-0's choice is the reference, so shifts declare AF and
/// write zero. Rotates leave AF alone on both sides, so their set is unchanged.
fn shift_flag_set(op: ShiftOp) -> FlagSet {
    match op {
        ShiftOp::Shl | ShiftOp::Shr | ShiftOp::Sar => FlagSet::ALU,
        // A rotate touches CF and OF and nothing else — ZF, SF, PF and AF keep the
        // values they had. Declaring more than that is not merely imprecise: dead
        // flag elimination reads these sets as what the instruction overwrites, so
        // claiming a rotate writes ZF lets it shadow the write that really produced
        // ZF, and neither the code generator nor the IR interpreter puts anything
        // back. Same shape of bug as an `sbb` whose carry went missing.
        ShiftOp::Rol | ShiftOp::Ror => FlagSet::CF.union(FlagSet::OF),
    }
}

fn to_shift_binop(op: ShiftOp) -> BinOp {
    match op {
        ShiftOp::Rol => BinOp::Rol,
        ShiftOp::Ror => BinOp::Ror,
        ShiftOp::Shl => BinOp::Shl,
        ShiftOp::Shr => BinOp::Shr,
        ShiftOp::Sar => BinOp::Sar,
    }
}

#[must_use]
#[track_caller]
pub fn translate_block(block: &BasicBlock) -> IrBlock {
    let mut b = IrBuilder::new(block.entry_rip);
    let stack_width = match block.bitness {
        16 => Width::W16,
        32 => Width::W32,
        64 => Width::W64,
        other => panic!("unsupported Tier-1 bitness {other}"),
    };
    let ip_mask: u64 = match block.bitness {
        16 => 0xffff,
        32 => 0xffff_ffff,
        64 => u64::MAX,
        other => panic!("unsupported Tier-1 bitness {other}"),
    };
    let addr_mask = if ip_mask != u64::MAX {
        Some(b.const_int(Width::W64, ip_mask))
    } else {
        None
    };

    // Default terminator: if we run off the end, keep executing in interpreter.
    let mut terminator = match block.end_kind {
        BlockEndKind::Limit { next_rip } | BlockEndKind::ExitToInterpreter { next_rip } => {
            IrTerminator::ExitToInterpreter {
                next_rip: next_rip & ip_mask,
            }
        }
        _ => IrTerminator::ExitToInterpreter {
            next_rip: block.entry_rip & ip_mask,
        },
    };

    let last_idx = block.insts.len().saturating_sub(1);
    for (idx, inst) in block.insts.iter().enumerate() {
        let is_last = idx == last_idx;
        // Keep `CpuState.rip` up to date within the block so runtime exits (MMIO/helper) can resume
        // precisely at the faulting x86 instruction rather than conservatively restarting at the
        // block entry RIP.
        //
        // `InstKind::Invalid` is not executed (Tier-1 side-exits to the interpreter at `inst.rip`),
        // so we skip emitting the per-instruction RIP update in that case.
        if !matches!(&inst.kind, InstKind::Invalid) {
            let rip_v = b.const_int(Width::W64, inst.rip & ip_mask);
            b.write_reg(GuestReg::Rip, rip_v);
        }

        match &inst.kind {
            InstKind::Nop => {}
            InstKind::Mov { dst, src, width } => {
                let v = emit_read_operand(&mut b, inst, src, *width, addr_mask);
                emit_write_operand(&mut b, inst, dst, *width, v, addr_mask);
            }
            InstKind::MovZx {
                dst,
                src,
                dst_width,
                src_width,
            } => {
                // IR values carry a width and `WriteReg` insists the value match
                // the register, so the narrow read needs converting rather than
                // just being handed over. `Trunc` to a *wider* width is that
                // conversion: the value already has its high bits clear, so
                // re-typing it at the destination width is the zero extension.
                let v = emit_read_operand(&mut b, inst, src, *src_width, addr_mask);
                let widened = b.zext(*dst_width, v);
                emit_write_operand(&mut b, inst, dst, *dst_width, widened, addr_mask);
            }
            InstKind::BitTest {
                op,
                dst,
                bit,
                width,
            } => {
                // CF takes the bit, and nothing else is written: Tier-0 leaves the
                // flags Intel calls undefined alone, so this must too.
                let v = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let shift = b.const_int(*width, u64::from(*bit));
                let shifted = b.binop(BinOp::Shr, *width, v, shift, FlagSet::EMPTY);
                let one = b.const_int(*width, 1);
                let cf = b.binop(BinOp::And, *width, shifted, one, FlagSet::EMPTY);
                // A flag is an i8 in the IR's type system.
                let cf = b.trunc(Width::W8, cf);
                b.write_reg(GuestReg::Flag(Flag::Cf), cf);

                let mask = b.const_int(*width, 1u64 << bit);
                let new = match op {
                    BitOp::Test => None,
                    BitOp::Set => Some(b.binop(BinOp::Or, *width, v, mask, FlagSet::EMPTY)),
                    BitOp::Reset => {
                        let inv = b.const_int(*width, !(1u64 << bit));
                        Some(b.binop(BinOp::And, *width, v, inv, FlagSet::EMPTY))
                    }
                    BitOp::Complement => {
                        Some(b.binop(BinOp::Xor, *width, v, mask, FlagSet::EMPTY))
                    }
                };
                if let Some(new) = new {
                    emit_write_operand(&mut b, inst, dst, *width, new, addr_mask);
                }
            }
            InstKind::Xadd { dst, src, width } => {
                // `dst` is read once: `xadd [mem], eax` must not evaluate the address
                // twice, and the old value is what the register receives.
                let old_dst = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let old_src = b.read_reg(gpr_reg(*src));
                let sum = b.binop(BinOp::Add, *width, old_dst, old_src, FlagSet::ALU);
                emit_write_operand(&mut b, inst, dst, *width, sum, addr_mask);
                b.write_reg(gpr_reg(*src), old_dst);
            }
            InstKind::Cmpxchg { dst, src, width } => {
                // Flags are CMP's — accumulator minus destination — whichever way the
                // comparison goes, and they are set before anything is stored.
                let acc = Reg {
                    gpr: Gpr::Rax,
                    width: *width,
                    high8: false,
                };
                let old_dst = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let old_acc = b.read_reg(gpr_reg(acc));
                b.cmp_flags(*width, old_acc, old_dst, FlagSet::ALU);
                let eq = b.eval_cond(Cond::E);

                // Equal: the destination takes the source. Not equal: it keeps what
                // it had. Written unconditionally either way, because a conditional
                // store would need control flow the IR does not have here — and
                // storing the same bytes back is what the architecture does anyway.
                let src_v = b.read_reg(gpr_reg(*src));
                let stored = b.select(*width, eq, src_v, old_dst);
                emit_write_operand(&mut b, inst, dst, *width, stored, addr_mask);

                // The accumulator takes the destination only when they differed.
                let acc_new = b.select(*width, eq, old_acc, old_dst);
                b.write_reg(gpr_reg(acc), acc_new);
            }
            InstKind::BitTestReg {
                op,
                dst,
                index,
                width,
            } => {
                // The offset is signed and can reach outside the operand, which for
                // a memory destination moves the bit base. `bits` is a power of two,
                // so Intel's euclidean divide and remainder are an arithmetic shift
                // and a mask — which is what Tier-0 computes too.
                let bits = width.bits();
                let log2_bits = bits.trailing_zeros();
                let raw = b.read_reg(gpr_reg(*index));
                // The register reads at its own width; the arithmetic below is 64-bit.
                let raw = b.zext(Width::W64, raw);
                // Sign-extend the offset to 64 bits: shift it up and back down.
                let sh = b.const_int(Width::W64, u64::from(64 - bits));
                let up = b.binop(BinOp::Shl, Width::W64, raw, sh, FlagSet::EMPTY);
                let off = b.binop(BinOp::Sar, Width::W64, up, sh, FlagSet::EMPTY);

                let bit_mask = b.const_int(Width::W64, u64::from(bits - 1));
                let bit = b.binop(BinOp::And, Width::W64, off, bit_mask, FlagSet::EMPTY);

                // Reading and writing the operand: a register destination ignores
                // the word offset entirely, a memory one is displaced by it.
                let (read_val, mem_addr) = match dst {
                    Operand::Reg(r) => (b.read_reg(gpr_reg(*r)), None),
                    Operand::Mem(addr) => {
                        let base = emit_address(&mut b, inst, addr, addr_mask);
                        let shift = b.const_int(Width::W64, u64::from(log2_bits));
                        let word_off = b.binop(BinOp::Sar, Width::W64, off, shift, FlagSet::EMPTY);
                        let scale = b.const_int(Width::W64, u64::from(log2_bits - 3));
                        let byte_off =
                            b.binop(BinOp::Shl, Width::W64, word_off, scale, FlagSet::EMPTY);
                        let a = b.binop(BinOp::Add, Width::W64, base, byte_off, FlagSet::EMPTY);
                        let a = mask_addr(&mut b, a, addr_mask);
                        (b.load(*width, a), Some(a))
                    }
                    Operand::Imm(_) => unreachable!("bit test destination cannot be an immediate"),
                };

                let bit_w = b.trunc(*width, bit);
                let shifted = b.binop(BinOp::Shr, *width, read_val, bit_w, FlagSet::EMPTY);
                let one = b.const_int(*width, 1);
                let cf = b.binop(BinOp::And, *width, shifted, one, FlagSet::EMPTY);
                let cf = b.trunc(Width::W8, cf);
                b.write_reg(GuestReg::Flag(Flag::Cf), cf);

                if !matches!(op, BitOp::Test) {
                    let one = b.const_int(*width, 1);
                    let mask = b.binop(BinOp::Shl, *width, one, bit_w, FlagSet::EMPTY);
                    let new = match op {
                        BitOp::Set => b.binop(BinOp::Or, *width, read_val, mask, FlagSet::EMPTY),
                        BitOp::Reset => {
                            let ones = b.const_int(*width, width.mask());
                            let inv = b.binop(BinOp::Xor, *width, mask, ones, FlagSet::EMPTY);
                            b.binop(BinOp::And, *width, read_val, inv, FlagSet::EMPTY)
                        }
                        BitOp::Complement => {
                            b.binop(BinOp::Xor, *width, read_val, mask, FlagSet::EMPTY)
                        }
                        BitOp::Test => unreachable!(),
                    };
                    match (dst, mem_addr) {
                        (Operand::Reg(r), _) => b.write_reg(gpr_reg(*r), new),
                        (Operand::Mem(_), Some(a)) => b.store(*width, a, new),
                        _ => unreachable!("memory destination without an address"),
                    }
                }
            }
            InstKind::Rdtsc => {
                // EDX:EAX, and a 32-bit write clears the upper half of each, which
                // is what the architecture specifies and what Tier-0 does.
                let tsc = b.read_tsc();
                let mask = b.const_int(Width::W64, 0xffff_ffff);
                let lo = b.binop(BinOp::And, Width::W64, tsc, mask, FlagSet::EMPTY);
                let shift = b.const_int(Width::W64, 32);
                let hi = b.binop(BinOp::Shr, Width::W64, tsc, shift, FlagSet::EMPTY);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rax,
                        width: Width::W64,
                        high8: false,
                    },
                    lo,
                );
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rdx,
                        width: Width::W64,
                        high8: false,
                    },
                    hi,
                );
            }
            InstKind::BitScan {
                dst,
                src,
                width,
                forward,
            } => {
                let s = emit_read_operand(&mut b, inst, src, *width, addr_mask);
                // Only ZF, which is all `BSF`/`BSR` write — the rest are architecturally
                // undefined and Tier-0's `exec_bsf_bsr` leaves them alone, so declaring
                // more here would diverge from it.
                let zero = b.const_int(*width, 0);
                b.cmp_flags(*width, s, zero, FlagSet::ZF);
                let nonzero = b.eval_cond(Cond::Ne);
                let idx = b.bitscan(*width, s, *forward);

                let Operand::Reg(r) = dst else {
                    unreachable!("BSF/BSR write a register operand");
                };
                // Merged at 64 bits rather than written at the operand width, because
                // a zero source must leave the register *exactly* as it was. Writing
                // the old value back at 32 bits would clear the upper half, which a
                // 32-bit write does and Tier-0 — which does not write at all — does
                // not.
                let full = GuestReg::Gpr {
                    reg: r.gpr,
                    width: Width::W64,
                    high8: false,
                };
                let old64 = b.read_reg(full);
                let idx64 = b.zext(Width::W64, idx);
                let written = if width.bits() >= 32 {
                    // A 32-bit write zeroes the upper half, and the index fits, so the
                    // whole register is just the index.
                    idx64
                } else {
                    let keep = b.const_int(Width::W64, !((1u64 << width.bits()) - 1));
                    let masked = b.binop(BinOp::And, Width::W64, old64, keep, FlagSet::EMPTY);
                    b.binop(BinOp::Or, Width::W64, masked, idx64, FlagSet::EMPTY)
                };
                let result = b.select(Width::W64, nonzero, written, old64);
                b.write_reg(full, result);
            }
            InstKind::MovSx {
                dst,
                src,
                dst_width,
                src_width,
            } => {
                // No sign-extend op in the IR, and none is needed: widen, shift the
                // sign bit up to the top and arithmetic-shift it back down. The
                // shifts declare no flags, so the dead-flag pass leaves them alone.
                let v = emit_read_operand(&mut b, inst, src, *src_width, addr_mask);
                let widened = b.zext(*dst_width, v);
                let shift = b.const_int(
                    *dst_width,
                    u64::from(dst_width.bits() - src_width.bits()),
                );
                let up = b.binop(BinOp::Shl, *dst_width, widened, shift, FlagSet::EMPTY);
                let signed = b.binop(BinOp::Sar, *dst_width, up, shift, FlagSet::EMPTY);
                emit_write_operand(&mut b, inst, dst, *dst_width, signed, addr_mask);
            }
            InstKind::Lea { dst, addr, width } => {
                let a = emit_address(&mut b, inst, addr, addr_mask);
                let dst_reg = GuestReg::Gpr {
                    reg: dst.gpr,
                    width: *width,
                    high8: false,
                };
                let val = if *width == Width::W64 {
                    a
                } else {
                    b.trunc(*width, a)
                };
                b.write_reg(dst_reg, val);
            }
            InstKind::Alu {
                op,
                dst,
                src,
                width,
            } => {
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let rhs = emit_read_operand(&mut b, inst, src, *width, addr_mask);
                let res = b.binop(to_binop(*op), *width, lhs, rhs, alu_flag_set(*op));
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Shift {
                op,
                dst,
                count,
                width,
            } => {
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                // Tier1 IR requires LHS/RHS have the same width.
                let rhs = b.const_int(*width, *count as u64);
                let res = b.binop(
                    to_shift_binop(*op),
                    *width,
                    lhs,
                    rhs,
                    shift_flag_set(*op),
                );
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::ShiftCl { op, dst, width } => {
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                // Read CX/ECX/RCX at the operand width so the IR types match.
                // CL is the low 8 bits; the architectural mask then isolates it.
                let raw = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rcx,
                    width: *width,
                    high8: false,
                });
                let mask_bits = if *width == Width::W64 { 0x3f } else { 0x1f };
                let mask = b.const_int(*width, mask_bits);
                let rhs = b.binop(BinOp::And, *width, raw, mask, FlagSet::EMPTY);
                let res = b.binop(
                    to_shift_binop(*op),
                    *width,
                    lhs,
                    rhs,
                    shift_flag_set(*op),
                );
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Cmp { lhs, rhs, width } => {
                let l = emit_read_operand(&mut b, inst, lhs, *width, addr_mask);
                let r = emit_read_operand(&mut b, inst, rhs, *width, addr_mask);
                b.cmp_flags(*width, l, r, FlagSet::ALU);
            }
            InstKind::Test { lhs, rhs, width } => {
                let l = emit_read_operand(&mut b, inst, lhs, *width, addr_mask);
                let r = emit_read_operand(&mut b, inst, rhs, *width, addr_mask);
                b.test_flags(*width, l, r, FlagSet::ALU);
            }
            InstKind::Inc { dst, width } => {
                let one = b.const_int(*width, 1);
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let res = b.binop(
                    BinOp::Add,
                    *width,
                    lhs,
                    one,
                    FlagSet::ALU.without(FlagSet::CF),
                );
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Dec { dst, width } => {
                let one = b.const_int(*width, 1);
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let res = b.binop(
                    BinOp::Sub,
                    *width,
                    lhs,
                    one,
                    FlagSet::ALU.without(FlagSet::CF),
                );
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Not { dst, width } => {
                // NOT does not update flags.
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let ones = b.const_int(*width, width.mask());
                let res = b.binop(BinOp::Xor, *width, lhs, ones, FlagSet::EMPTY);
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Neg { dst, width } => {
                let lhs = emit_read_operand(&mut b, inst, dst, *width, addr_mask);
                let zero = b.const_int(*width, 0);
                let res = b.binop(BinOp::Sub, *width, zero, lhs, FlagSet::ALU);
                emit_write_operand(&mut b, inst, dst, *width, res, addr_mask);
            }
            InstKind::Imul3 {
                dst,
                src,
                imm,
                width,
            } => {
                let lhs = emit_read_operand(&mut b, inst, src, *width, addr_mask);
                let rhs = b.const_int(*width, *imm);
                // IMUL writes CF/OF; SF/ZF/AF/PF are undefined. The CIM hash
                // and CRT memcpy immediately overwrite flags with SAR/AND.
                let res = b.binop(BinOp::Mul, *width, lhs, rhs, FlagSet::CF.union(FlagSet::OF));
                emit_write_operand(
                    &mut b,
                    inst,
                    &Operand::Reg(*dst),
                    *width,
                    res,
                    addr_mask,
                );
            }
            InstKind::Bswap { dst, width } => {
                let dst_op = Operand::Reg(*dst);
                let lhs = emit_read_operand(&mut b, inst, &dst_op, *width, addr_mask);
                let zero = b.const_int(*width, 0);
                let res = b.binop(BinOp::Bswap, *width, lhs, zero, FlagSet::EMPTY);
                emit_write_operand(&mut b, inst, &dst_op, *width, res, addr_mask);
            }
            InstKind::Push { src } => {
                let rsp = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rsp,
                    width: Width::W64,
                    high8: false,
                });
                let rsp = mask_addr(&mut b, rsp, addr_mask);
                let slot = b.const_int(Width::W64, stack_width.bytes() as u64);
                let new_rsp = b.binop(BinOp::Sub, Width::W64, rsp, slot, FlagSet::EMPTY);
                let new_rsp = mask_addr(&mut b, new_rsp, addr_mask);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rsp,
                        width: Width::W64,
                        high8: false,
                    },
                    new_rsp,
                );
                let v = emit_read_operand(&mut b, inst, src, stack_width, addr_mask);
                b.store(stack_width, new_rsp, v);
            }
            InstKind::Pop { dst } => {
                let rsp = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rsp,
                    width: Width::W64,
                    high8: false,
                });
                let rsp = mask_addr(&mut b, rsp, addr_mask);
                let v = b.load(stack_width, rsp);
                let slot = b.const_int(Width::W64, stack_width.bytes() as u64);
                let new_rsp = b.binop(BinOp::Add, Width::W64, rsp, slot, FlagSet::EMPTY);
                let new_rsp = mask_addr(&mut b, new_rsp, addr_mask);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rsp,
                        width: Width::W64,
                        high8: false,
                    },
                    new_rsp,
                );
                emit_write_operand(&mut b, inst, dst, stack_width, v, addr_mask);
            }
            InstKind::Setcc { cond, dst } => {
                let c = b.eval_cond(*cond);
                // SETcc writes 0/1 into an 8-bit destination.
                emit_write_operand(&mut b, inst, dst, Width::W8, c, addr_mask);
            }
            InstKind::Cmovcc {
                cond,
                dst,
                src,
                width,
            } => {
                let cond_v = b.eval_cond(*cond);
                let dst_reg = GuestReg::Gpr {
                    reg: dst.gpr,
                    width: *width,
                    high8: false,
                };
                let old = b.read_reg(dst_reg);
                let new = emit_read_operand(&mut b, inst, src, *width, addr_mask);
                let sel = b.select(*width, cond_v, new, old);
                b.write_reg(dst_reg, sel);
            }
            InstKind::JmpRel { target } => {
                if !is_last {
                    // Discovery followed this jump, so the instruction decoded after it
                    // *is* its target: control already goes where the jump would send
                    // it, and there is nothing to emit. Unlike fusing a conditional,
                    // this needs no side exit and no guess — the edge is unconditional.
                    continue;
                }
                terminator = IrTerminator::Jump {
                    target: *target & ip_mask,
                };
                break;
            }
            InstKind::JccRel { cond, target } => {
                let c = b.eval_cond(*cond);
                if is_last {
                    terminator = IrTerminator::CondJump {
                        cond: c,
                        target: *target & ip_mask,
                        fallthrough: inst.next_rip() & ip_mask,
                    };
                    break;
                }
                // Not the last instruction: the block was decoded past this branch,
                // so the taken side leaves and the fall-through carries on here.
                b.side_exit(c, *target & ip_mask);
            }
            InstKind::CallRel { target } => {
                let return_rip = inst.next_rip() & ip_mask;
                let rsp = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rsp,
                    width: Width::W64,
                    high8: false,
                });
                let rsp = mask_addr(&mut b, rsp, addr_mask);
                let slot = b.const_int(Width::W64, stack_width.bytes() as u64);
                let new_rsp = b.binop(BinOp::Sub, Width::W64, rsp, slot, FlagSet::EMPTY);
                let new_rsp = mask_addr(&mut b, new_rsp, addr_mask);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rsp,
                        width: Width::W64,
                        high8: false,
                    },
                    new_rsp,
                );
                let ret = b.const_int(stack_width, return_rip);
                b.store(stack_width, new_rsp, ret);
                if !is_last {
                    // Discovery followed this call, so the instruction decoded after it is
                    // the callee's first: the push above is the whole architectural effect
                    // and control already goes where the call would send it.
                    continue;
                }
                terminator = IrTerminator::Jump {
                    target: *target & ip_mask,
                };
                break;
            }
            InstKind::Ret { pop } => {
                let rsp = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rsp,
                    width: Width::W64,
                    high8: false,
                });
                let rsp = mask_addr(&mut b, rsp, addr_mask);
                let target = b.load(stack_width, rsp);
                // The return address is read before RSP moves, so `pop` folds into
                // the same adjustment rather than needing a second one.
                let slot =
                    b.const_int(Width::W64, stack_width.bytes() as u64 + u64::from(*pop));
                let new_rsp = b.binop(BinOp::Add, Width::W64, rsp, slot, FlagSet::EMPTY);
                let new_rsp = mask_addr(&mut b, new_rsp, addr_mask);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rsp,
                        width: Width::W64,
                        high8: false,
                    },
                    new_rsp,
                );
                terminator = IrTerminator::IndirectJump { target };
                break;
            }
            InstKind::CallIndirect { target } => {
                // The operand is read before the push: `call [rsp]` must take the
                // target from the old stack top, not from the slot the return
                // address is about to occupy.
                let dest = emit_read_operand(&mut b, inst, target, stack_width, addr_mask);
                let return_rip = inst.next_rip() & ip_mask;
                let rsp = b.read_reg(GuestReg::Gpr {
                    reg: Gpr::Rsp,
                    width: Width::W64,
                    high8: false,
                });
                let rsp = mask_addr(&mut b, rsp, addr_mask);
                let slot = b.const_int(Width::W64, stack_width.bytes() as u64);
                let new_rsp = b.binop(BinOp::Sub, Width::W64, rsp, slot, FlagSet::EMPTY);
                let new_rsp = mask_addr(&mut b, new_rsp, addr_mask);
                b.write_reg(
                    GuestReg::Gpr {
                        reg: Gpr::Rsp,
                        width: Width::W64,
                        high8: false,
                    },
                    new_rsp,
                );
                let ret = b.const_int(stack_width, return_rip);
                b.store(stack_width, new_rsp, ret);
                terminator = IrTerminator::IndirectJump { target: dest };
                break;
            }
            InstKind::JmpIndirect { target } => {
                let dest = emit_read_operand(&mut b, inst, target, stack_width, addr_mask);
                terminator = IrTerminator::IndirectJump { target: dest };
                break;
            }
            InstKind::Invalid => {
                // The block only reaches here with an assist available — otherwise
                // discovery would have ended it — so hand the instruction to the host
                // and carry on. `next_rip` is what the rest of this block was decoded
                // against; the host reports where the guest really went and the block
                // leaves if they differ.
                if is_last {
                    terminator = IrTerminator::ExitToInterpreter {
                        next_rip: inst.rip & ip_mask,
                    };
                    break;
                }
                b.assist(inst.rip & ip_mask, inst.next_rip() & ip_mask);
            }
        }
    }

    let block = b.finish(terminator);
    if let Err(e) = block.validate() {
        panic!("invalid IR generated: {e}\n{}", block.to_text());
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::capture_panic_location;

    #[test]
    fn translate_block_panics_at_call_site_on_invalid_bitness() {
        let bb = BasicBlock {
            entry_rip: 0,
            bitness: 0,
            insts: Vec::new(),
            end_kind: BlockEndKind::Limit { next_rip: 0 },
        };

        let expected_file = file!();
        let expected_line = line!() + 2;
        let (file, line) = capture_panic_location(|| {
            let _ = translate_block(&bb);
        });
        assert_eq!(file, expected_file);
        assert_eq!(line, expected_line);
    }
}

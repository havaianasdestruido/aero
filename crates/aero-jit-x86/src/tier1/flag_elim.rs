//! Drop flag writes nothing in the block goes on to read.
//!
//! x86 arithmetic defines six flags and most code reads none of them: a `cmp`
//! feeds one branch, and the `add`s around it set CF, PF, AF, ZF, SF and OF for
//! nobody. Tier-1 materialises every declared flag, and computing them is a large
//! part of what compiled code spends its time on — measured at ~100ns per guest
//! instruction against ~7ns for the same engine on a loop whose flags are never
//! read.
//!
//! This is the cheap half of the classic fix. Real lazy flags defer the
//! computation and reconstruct a flag from its operands when something finally
//! reads it, which also wins across block boundaries. This pass only removes
//! writes that are provably overwritten *within the block* before any read, which
//! needs no new IR and cannot change observable state.
//!
//! Flags are architecturally visible after the block, so all six are treated as
//! live at the exit; only a write another write shadows is removed.

use aero_types::{Flag, FlagSet};

use super::ir::{BinOp, GuestReg, IrBlock, IrInst};

fn flag_bit(flag: Flag) -> FlagSet {
    match flag {
        Flag::Cf => FlagSet::CF,
        Flag::Pf => FlagSet::PF,
        Flag::Af => FlagSet::AF,
        Flag::Zf => FlagSet::ZF,
        Flag::Sf => FlagSet::SF,
        Flag::Of => FlagSet::OF,
    }
}

/// Remove dead flag writes from `block`, returning how many flag bits were dropped.
///
/// Semantics are unchanged: a removed write is one a later write in the same block
/// replaces with no read in between.
pub fn eliminate_dead_flags(block: &mut IrBlock) -> u32 {
    // Everything is live leaving the block — the next block, or the interpreter,
    // may read any of it.
    let mut live = FlagSet::ALU;
    let mut removed = 0u32;

    for inst in block.insts.iter_mut().rev() {
        // A helper call can observe or clobber anything; give up on everything
        // above it rather than reason about what it touches.
        //
        // A side exit leaves the block, which makes every flag architecturally
        // visible on that path exactly as the block's own exit does — so nothing
        // above one may be dropped either.
        if matches!(
            inst,
            IrInst::CallHelper { .. } | IrInst::SideExit { .. }
        ) {
            live = FlagSet::ALU;
            continue;
        }

        let reads = match inst {
            IrInst::EvalCond { cond, .. } => cond.uses_flags(),
            IrInst::ReadReg {
                reg: GuestReg::Flag(flag),
                ..
            } => flag_bit(*flag),
            // `adc`/`sbb` take the incoming carry as an operand, and the IR does not
            // say so: `translate` emits a plain `BinOp`, and the code generator reads
            // CF itself when it lowers one. A pass that believed the IR alone would
            // find no reader for the CF a `cmp` writes ahead of an `sbb`, call the
            // write dead, and delete it — which is exactly what `cmp rax,rcx; sbb
            // eax,eax` does for a living.
            IrInst::BinOp {
                op: BinOp::Adc | BinOp::Sbb,
                ..
            } => FlagSet::CF,
            _ => FlagSet::EMPTY,
        };

        let written = match inst {
            IrInst::BinOp { flags, .. }
            | IrInst::CmpFlags { flags, .. }
            | IrInst::TestFlags { flags, .. } => {
                let original = *flags;
                let kept = original.intersect(live);
                removed += original.without(kept).iter().count() as u32;
                *flags = kept;
                original
            }
            IrInst::WriteReg {
                reg: GuestReg::Flag(flag),
                ..
            } => flag_bit(*flag),
            _ => FlagSet::EMPTY,
        };

        // Anything this instruction writes is dead for instructions above it,
        // except what it also reads.
        live = live.without(written).union(reads);
    }

    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::ir::{BinOp, IrBuilder, IrTerminator};
    use aero_types::{Gpr, Width};

    fn gpr(reg: Gpr) -> GuestReg {
        GuestReg::Gpr {
            reg,
            width: Width::W64,
            high8: false,
        }
    }

    /// Two arithmetic ops in a row: the first one's flags are replaced by the
    /// second's before anything reads them.
    #[test]
    fn a_shadowed_write_is_removed() {
        let mut b = IrBuilder::new(0x1000);
        let a = b.read_reg(gpr(Gpr::Rax));
        let c = b.read_reg(gpr(Gpr::Rcx));
        let first = b.binop(BinOp::Add, Width::W64, a, c, FlagSet::ALU);
        let second = b.binop(BinOp::Add, Width::W64, first, c, FlagSet::ALU);
        b.write_reg(gpr(Gpr::Rax), second);
        let mut block = b.finish(IrTerminator::Jump { target: 0x2000 });

        let removed = eliminate_dead_flags(&mut block);
        assert_eq!(removed, 6, "every flag of the first add is shadowed");

        let mut sets: Vec<FlagSet> = block
            .insts
            .iter()
            .filter_map(|i| match i {
                IrInst::BinOp { flags, .. } => Some(*flags),
                _ => None,
            })
            .collect();
        assert_eq!(sets.len(), 2);
        let last = sets.pop().unwrap();
        assert!(sets[0].is_empty(), "the shadowed write should be gone");
        assert_eq!(last, FlagSet::ALU, "the surviving write reaches the exit");
    }

    /// The same pair with a condition between them: the first write is read, so it
    /// has to stay.
    #[test]
    fn a_write_that_is_read_survives() {
        let mut b = IrBuilder::new(0x1000);
        let a = b.read_reg(gpr(Gpr::Rax));
        let c = b.read_reg(gpr(Gpr::Rcx));
        let first = b.binop(BinOp::Add, Width::W64, a, c, FlagSet::ALU);
        let cond = b.eval_cond(aero_types::Cond::E);
        let second = b.binop(BinOp::Add, Width::W64, first, c, FlagSet::ALU);
        b.write_reg(gpr(Gpr::Rax), second);
        let mut block = b.finish(IrTerminator::CondJump {
            cond,
            target: 0x2000,
            fallthrough: 0x3000,
        });

        eliminate_dead_flags(&mut block);
        let sets: Vec<FlagSet> = block
            .insts
            .iter()
            .filter_map(|i| match i {
                IrInst::BinOp { flags, .. } => Some(*flags),
                _ => None,
            })
            .collect();
        assert!(
            sets[0].contains(FlagSet::ZF),
            "ZF feeds the `eq` between the two adds and must survive"
        );
    }

    /// `cmp` then `sbb`: the carry crosses between them without the IR saying so.
    ///
    /// This is the shape that made the compiled tier diverge from the interpreter —
    /// `cmp rax, rcx; sbb eax, eax; sbb eax, -1`, the usual way to turn a comparison
    /// into 0/-1. Both `sbb`s write all six flags, so on the IR alone the `cmp`'s
    /// writes look shadowed and removable; they are not, because the code generator
    /// reads CF when it lowers an `sbb`.
    #[test]
    fn a_carry_an_sbb_consumes_is_not_dead() {
        let mut b = IrBuilder::new(0x1000);
        let a = b.read_reg(gpr(Gpr::Rax));
        let c = b.read_reg(gpr(Gpr::Rcx));
        b.cmp_flags(Width::W64, a, c, FlagSet::ALU);
        let first = b.binop(BinOp::Sbb, Width::W32, a, a, FlagSet::ALU);
        let minus_one = b.const_int(Width::W32, u64::MAX);
        let second = b.binop(BinOp::Sbb, Width::W32, first, minus_one, FlagSet::ALU);
        b.write_reg(gpr(Gpr::Rax), second);
        let mut block = b.finish(IrTerminator::Jump { target: 0x2000 });

        eliminate_dead_flags(&mut block);

        let cmp_flags = block
            .insts
            .iter()
            .find_map(|i| match i {
                IrInst::CmpFlags { flags, .. } => Some(*flags),
                _ => None,
            })
            .expect("the cmp is still there");
        assert!(
            cmp_flags.contains(FlagSet::CF),
            "the carry the first sbb consumes must survive; got {cmp_flags:?}"
        );
        let sbb_flags: Vec<FlagSet> = block
            .insts
            .iter()
            .filter_map(|i| match i {
                IrInst::BinOp { op: BinOp::Sbb, flags, .. } => Some(*flags),
                _ => None,
            })
            .collect();
        assert!(
            sbb_flags[0].contains(FlagSet::CF),
            "the first sbb's carry feeds the second one"
        );
    }

    /// Nothing may be dropped when it is the last write before the block exits.
    #[test]
    fn the_final_write_always_survives() {
        let mut b = IrBuilder::new(0x1000);
        let a = b.read_reg(gpr(Gpr::Rax));
        let c = b.read_reg(gpr(Gpr::Rcx));
        let sum = b.binop(BinOp::Add, Width::W64, a, c, FlagSet::ALU);
        b.write_reg(gpr(Gpr::Rax), sum);
        let mut block = b.finish(IrTerminator::Jump { target: 0x2000 });

        assert_eq!(eliminate_dead_flags(&mut block), 0);
    }
}

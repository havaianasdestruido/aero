//! Minimal Tier-1 decode / normalization layer.
//!
//! This module exists primarily to support the Tier-1 JIT front-end unit tests
//! without requiring the full interpreter decode pipeline.
//!
//! This decoder only supports a subset of x86-64 sufficient for building and
//! testing basic-block discovery + translation. It is **not** intended to be
//! complete or particularly fast.

use aero_types::{Cond, Gpr, Width};
mod decode;
mod display;

pub use decode::{decode_one, decode_one_mode, force_bitscan, force_hint_ops, force_rmw_ops};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub gpr: Gpr,
    pub width: Width,
    pub high8: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    pub base: Option<Gpr>,
    pub index: Option<Gpr>,
    pub scale: u8,
    pub disp: i32,
    pub rip_relative: bool,
    /// Segment override, when the instruction carried one.
    ///
    /// Only FS and GS matter in long mode — the rest have a zero base — and Windows
    /// reaches its per-CPU and per-thread blocks through `gs:`, so a Tier-1 that
    /// cannot express this bails out of most of the kernel.
    pub segment: Option<SegmentPrefix>,
}

/// A segment override that still has a base in long mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentPrefix {
    Fs,
    Gs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    Imm(u64),
    Mem(Address),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Adc,
    Sub,
    Sbb,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Imul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftOp {
    Rol,
    Ror,
    Shl,
    Shr,
    Sar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstKind {
    Nop,
    Mov {
        dst: Operand,
        src: Operand,
        width: Width,
    },
    /// `MOVZX dst, src` — read `src` at `src_width`, write `dst` at `dst_width`
    /// with the high bits cleared.
    ///
    /// Distinct from [`InstKind::Mov`] only in reading and writing at different
    /// widths; the zero extension is what reading a narrow operand into a wide one
    /// already means.
    MovZx {
        dst: Operand,
        src: Operand,
        dst_width: Width,
        src_width: Width,
    },
    Lea {
        dst: Reg,
        addr: Address,
        width: Width,
    },
    Alu {
        op: AluOp,
        dst: Operand,
        src: Operand,
        width: Width,
    },
    Shift {
        op: ShiftOp,
        dst: Operand,
        count: u8,
        width: Width,
    },
    /// Group2 shift/rotate by `CL` (`0xD2` / `0xD3`).
    ShiftCl {
        op: ShiftOp,
        dst: Operand,
        width: Width,
    },
    Cmp {
        lhs: Operand,
        rhs: Operand,
        width: Width,
    },
    Test {
        lhs: Operand,
        rhs: Operand,
        width: Width,
    },
    Inc {
        dst: Operand,
        width: Width,
    },
    Dec {
        dst: Operand,
        width: Width,
    },
    Not {
        dst: Operand,
        width: Width,
    },
    /// Two's complement negate (`F6 /3`, `F7 /3`): `dst = 0 - dst`.
    Neg {
        dst: Operand,
        width: Width,
    },
    Bswap {
        dst: Reg,
        width: Width,
    },
    /// Three-operand `IMUL r, r/m, imm` (`0x69` / `0x6B`): `dst = src * imm`.
    Imul3 {
        dst: Reg,
        src: Operand,
        imm: u64,
        width: Width,
    },
    Push {
        src: Operand,
    },
    Pop {
        dst: Operand,
    },
    JmpRel {
        target: u64,
    },
    JccRel {
        cond: Cond,
        target: u64,
    },
    CallRel {
        target: u64,
    },
    /// Near return. `pop` is the extra byte count `RET imm16` releases.
    Ret { pop: u16 },
    /// Bit test, with an optional write-back. `bit` is the *constant* offset the
    /// decoder resolved: for a memory operand the address adjustment Intel specifies
    /// for offsets past the operand size is folded into `dst` at decode time, so
    /// what remains is a plain mask-and-test.
    BitTest {
        op: BitOp,
        dst: Operand,
        bit: u8,
        width: Width,
    },
    /// `XADD`: exchange and add. `dst` gets `dst + src`, `src` gets the old `dst`,
    /// and the flags are ADD's. Common under `LOCK` in Windows' reference counting.
    Xadd {
        dst: Operand,
        src: Reg,
        width: Width,
    },
    /// `CMPXCHG`: compare the accumulator with `dst`; on equal store `src` into `dst`,
    /// otherwise load `dst` into the accumulator. Flags are CMP's either way.
    Cmpxchg {
        dst: Operand,
        src: Reg,
        width: Width,
    },
    /// Bit test whose offset is a register, so the bit position — and, for a memory
    /// operand, the bit base — are only known at run time.
    BitTestReg {
        op: BitOp,
        dst: Operand,
        index: Reg,
        width: Width,
    },
    /// `RDTSC`: read the timestamp counter into EDX:EAX.
    Rdtsc,
    /// Sign-extending move. Distinct from [`InstKind::MovZx`] because the widening
    /// is arithmetic, not a re-typing of an already-clear value.
    MovSx {
        dst: Operand,
        src: Operand,
        dst_width: Width,
        src_width: Width,
    },
    /// `BSF`/`BSR`: index of the lowest or highest set bit of `src` into `dst`.
    ///
    /// Only `ZF` is written, and when `src` is zero the destination is left alone —
    /// the manual calls it undefined, and leaving it is what both the hardware and
    /// Tier-0's `exec_bsf_bsr` do.
    BitScan {
        dst: Operand,
        src: Operand,
        width: Width,
        forward: bool,
    },
    /// Near indirect call through a register or memory operand (`FF /2`).
    CallIndirect { target: Operand },
    /// Near indirect jump through a register or memory operand (`FF /4`).
    JmpIndirect { target: Operand },
    Setcc {
        cond: Cond,
        dst: Operand,
    },
    Cmovcc {
        cond: Cond,
        dst: Reg,
        src: Operand,
        width: Width,
    },
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedInst {
    pub rip: u64,
    pub len: u8,
    pub kind: InstKind,
    /// Whether an [`InstKind::Invalid`] here may be handed to the host from inside a
    /// compiled block rather than ending it.
    ///
    /// False for the interrupt-related ones — `CLI`, `STI`, `INT`, `IRET` and their
    /// kin. Performing those needs the interpreter's interrupt state, not just
    /// architectural registers, so a block that tried would bail every time: `cli`
    /// alone accounted for 2,423 of the 2,802 failed assists in one session, and a
    /// block that bails is worse than one that never contained the instruction.
    pub assistable: bool,
    next_rip: u64,
}

impl DecodedInst {
    #[must_use]
    pub fn next_rip(&self) -> u64 {
        self.next_rip
    }

    #[must_use]
    pub fn is_block_terminator(&self) -> bool {
        matches!(
            self.kind,
            InstKind::JmpRel { .. }
                | InstKind::JccRel { .. }
                | InstKind::CallRel { .. }
                | InstKind::Ret { .. }
                | InstKind::CallIndirect { .. }
                | InstKind::JmpIndirect { .. }
                | InstKind::Invalid
        )
    }
}

/// What a bit-test instruction does to the bit it tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    /// `BT`: test only.
    Test,
    /// `BTS`: set it.
    Set,
    /// `BTR`: clear it.
    Reset,
    /// `BTC`: flip it.
    Complement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    pub message: &'static str,
}

impl std::error::Error for DecodeError {}

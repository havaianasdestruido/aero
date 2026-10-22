//! Debug-only IR interpreter used for validating the x86→IR translation.

use super::{BinOp, GuestReg, IrBlock, IrInst, IrTerminator};
use aero_cpu_core::state::{
    CpuState, RFLAGS_AF, RFLAGS_CF, RFLAGS_OF, RFLAGS_PF, RFLAGS_SF, RFLAGS_ZF,
};
use aero_types::{Cond, Flag, FlagSet, Width};

use crate::abi;
use crate::Tier1Bus;

#[derive(Debug, Clone, Copy)]
struct FlagVals {
    cf: bool,
    pf: bool,
    af: bool,
    zf: bool,
    sf: bool,
    of: bool,
}

fn parity_even(byte: u8) -> bool {
    byte.count_ones().is_multiple_of(2)
}

fn compute_logic_flags(width: Width, result: u64) -> FlagVals {
    let r = width.truncate(result);
    let sign_bit = 1u64 << (width.bits() - 1);
    FlagVals {
        cf: false,
        pf: parity_even(r as u8),
        af: false,
        zf: r == 0,
        sf: (r & sign_bit) != 0,
        of: false,
    }
}

fn compute_add_flags(width: Width, lhs: u64, rhs: u64, result: u64) -> FlagVals {
    compute_add_flags_cin(width, lhs, rhs, 0, result)
}

fn compute_add_flags_cin(width: Width, lhs: u64, rhs: u64, cin: u64, result: u64) -> FlagVals {
    let mask = width.mask();
    let lhs = lhs & mask;
    let rhs = rhs & mask;
    let result = result & mask;
    let bits = width.bits();
    let sign_bit = 1u64 << (bits - 1);

    let wide = (lhs as u128) + (rhs as u128) + (cin as u128);
    let cf = (wide >> bits) != 0;
    let of = ((lhs ^ result) & (rhs ^ result) & sign_bit) != 0;
    let af = ((lhs ^ rhs ^ result) & 0x10) != 0;
    FlagVals {
        cf,
        pf: parity_even(result as u8),
        af,
        zf: result == 0,
        sf: (result & sign_bit) != 0,
        of,
    }
}

fn compute_sub_flags(width: Width, lhs: u64, rhs: u64, result: u64) -> FlagVals {
    compute_sub_flags_bin(width, lhs, rhs, 0, result)
}

fn compute_sub_flags_bin(width: Width, lhs: u64, rhs: u64, bin: u64, result: u64) -> FlagVals {
    let mask = width.mask();
    let lhs = lhs & mask;
    let rhs = rhs & mask;
    let result = result & mask;
    let bits = width.bits();
    let sign_bit = 1u64 << (bits - 1);

    let subtrahend = (rhs as u128) + (bin as u128);
    let cf = (lhs as u128) < subtrahend;
    let of = ((lhs ^ rhs) & (lhs ^ result) & sign_bit) != 0;
    let af = ((lhs ^ rhs ^ result) & 0x10) != 0;
    FlagVals {
        cf,
        pf: parity_even(result as u8),
        af,
        zf: result == 0,
        sf: (result & sign_bit) != 0,
        of,
    }
}

fn rotate_left(width: Width, val: u64, shift_amt: u32) -> u64 {
    let bits = width.bits();
    let val = width.truncate(val);
    let amt = shift_amt % bits;
    if amt == 0 {
        val
    } else {
        width.truncate((val << amt) | (val >> (bits - amt)))
    }
}

fn rotate_right(width: Width, val: u64, shift_amt: u32) -> u64 {
    let bits = width.bits();
    let val = width.truncate(val);
    let amt = shift_amt % bits;
    if amt == 0 {
        val
    } else {
        width.truncate((val >> amt) | (val << (bits - amt)))
    }
}

fn bswap_width(width: Width, val: u64) -> u64 {
    match width {
        Width::W64 => val.swap_bytes(),
        Width::W32 => (val as u32).swap_bytes() as u64,
        Width::W16 => (val as u16).swap_bytes() as u64,
        Width::W8 => val & 0xff,
    }
}

fn update_rotate_flags(
    cpu: &mut TestCpu,
    width: Width,
    op: BinOp,
    _lhs: u64,
    shift_amt: u32,
    result: u64,
    flags: FlagSet,
) {
    let bits = width.bits();
    let amt = shift_amt % bits;
    if amt == 0 {
        return;
    }
    let result = width.truncate(result);
    let sign_bit = 1u64 << (bits - 1);
    if flags.contains(FlagSet::CF) {
        let cf = match op {
            BinOp::Rol => (result & 1) != 0,
            BinOp::Ror => (result & sign_bit) != 0,
            _ => false,
        };
        write_flag(cpu, Flag::Cf, cf);
    }
    if flags.contains(FlagSet::OF) && amt == 1 {
        let msb = (result & sign_bit) != 0;
        let of = match op {
            BinOp::Rol => msb ^ ((result & 1) != 0),
            BinOp::Ror => {
                let msb2 = (result & (sign_bit >> 1)) != 0;
                msb ^ msb2
            }
            _ => false,
        };
        write_flag(cpu, Flag::Of, of);
    }
}

fn update_shift_flags(
    cpu: &mut TestCpu,
    width: Width,
    op: BinOp,
    lhs: u64,
    shift_amt: u32,
    result: u64,
    flags: FlagSet,
) {
    debug_assert!(matches!(op, BinOp::Shl | BinOp::Shr | BinOp::Sar));

    // x86 shifts do not update any flags when the shift count is 0.
    if shift_amt == 0 {
        return;
    }

    let result = width.truncate(result);
    let sign_bit = 1u64 << (width.bits() - 1);

    if flags.contains(FlagSet::ZF) {
        write_flag(cpu, Flag::Zf, result == 0);
    }
    if flags.contains(FlagSet::SF) {
        write_flag(cpu, Flag::Sf, (result & sign_bit) != 0);
    }
    if flags.contains(FlagSet::PF) {
        write_flag(cpu, Flag::Pf, parity_even(result as u8));
    }
    // AF is undefined here; Tier-0 clears it, so match that rather than leaving a
    // value the other tier would not have produced.
    if flags.contains(FlagSet::AF) {
        write_flag(cpu, Flag::Af, false);
    }

    // CF is defined for shift counts in the range [1, width.bits()]. For counts above the operand
    // width, CF is architecturally undefined; we conservatively leave it unchanged.
    if flags.contains(FlagSet::CF) && shift_amt <= width.bits() {
        let cf = match op {
            BinOp::Shl => ((lhs >> (width.bits() - shift_amt)) & 1) != 0,
            BinOp::Shr | BinOp::Sar => ((lhs >> (shift_amt - 1)) & 1) != 0,
            _ => unreachable!(),
        };
        write_flag(cpu, Flag::Cf, cf);
    }

    // OF is only defined for a shift count of 1. For counts > 1, OF is undefined; leave unchanged.
    if flags.contains(FlagSet::OF) && shift_amt == 1 {
        let of = match op {
            // For SHL count==1: OF = new MSB XOR CF (where CF is the old MSB).
            BinOp::Shl => ((lhs ^ result) & sign_bit) != 0,
            // For SHR count==1: OF = old MSB.
            BinOp::Shr => (lhs & sign_bit) != 0,
            // For SAR count==1: OF = 0.
            BinOp::Sar => false,
            _ => unreachable!(),
        };
        write_flag(cpu, Flag::Of, of);
    }
}

fn write_flagset(cpu: &mut TestCpu, mask: FlagSet, vals: FlagVals) {
    if mask.contains(FlagSet::CF) {
        write_flag(cpu, Flag::Cf, vals.cf);
    }
    if mask.contains(FlagSet::PF) {
        write_flag(cpu, Flag::Pf, vals.pf);
    }
    if mask.contains(FlagSet::AF) {
        write_flag(cpu, Flag::Af, vals.af);
    }
    if mask.contains(FlagSet::ZF) {
        write_flag(cpu, Flag::Zf, vals.zf);
    }
    if mask.contains(FlagSet::SF) {
        write_flag(cpu, Flag::Sf, vals.sf);
    }
    if mask.contains(FlagSet::OF) {
        write_flag(cpu, Flag::Of, vals.of);
    }
}

fn eval_cond(cpu: &TestCpu, cond: Cond) -> bool {
    cond.eval(
        read_flag(cpu, Flag::Cf),
        read_flag(cpu, Flag::Pf),
        read_flag(cpu, Flag::Zf),
        read_flag(cpu, Flag::Sf),
        read_flag(cpu, Flag::Of),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecResult {
    Continue,
    ExitToInterpreter { next_rip: u64 },
}

/// Minimal CPU state subset used by the debug Tier-1 IR interpreter.
///
/// This intentionally matches the stable `aero_cpu_core::state::CpuState` *in-memory ABI* that
/// Tier-1 WASM blocks operate on (as defined by [`crate::abi`]), but only models the architectural
/// registers and flags that the Tier-1 IR can currently touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestCpu {
    pub gpr: [u64; abi::GPR_COUNT],
    pub rip: u64,
    pub rflags: u64,
    /// Read-only here; `fs:`/`gs:` addressing needs them, nothing writes them.
    pub fs_base: u64,
    pub gs_base: u64,
}

impl Default for TestCpu {
    fn default() -> Self {
        Self {
            gpr: [0; abi::GPR_COUNT],
            rip: 0,
            rflags: abi::RFLAGS_RESERVED1,
            fs_base: 0,
            gs_base: 0,
        }
    }
}

impl TestCpu {
    /// Loads a [`TestCpu`] from a canonical `CpuState` ABI byte buffer.
    #[must_use]
    pub fn from_abi_mem(mem: &[u8]) -> Self {
        assert!(
            mem.len() >= abi::CPU_STATE_SIZE as usize,
            "CpuState ABI buffer too small"
        );

        let mut gpr = [0u64; abi::GPR_COUNT];
        for (i, slot) in gpr.iter_mut().enumerate() {
            let off = abi::CPU_GPR_OFF[i] as usize;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&mem[off..off + 8]);
            *slot = u64::from_le_bytes(buf);
        }

        let mut buf = [0u8; 8];
        let rip_off = abi::CPU_RIP_OFF as usize;
        buf.copy_from_slice(&mem[rip_off..rip_off + 8]);
        let rip = u64::from_le_bytes(buf);

        let read_u64 = |off: usize| -> u64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&mem[off..off + 8]);
            u64::from_le_bytes(b)
        };
        let fs_base = read_u64(abi::CPU_FS_BASE_OFF as usize);
        let gs_base = read_u64(abi::CPU_GS_BASE_OFF as usize);

        let rflags_off = abi::CPU_RFLAGS_OFF as usize;
        buf.copy_from_slice(&mem[rflags_off..rflags_off + 8]);
        let rflags = u64::from_le_bytes(buf);

        Self {
            gpr,
            rip,
            rflags,
            fs_base,
            gs_base,
        }
    }

    /// Writes this [`TestCpu`] into a canonical `CpuState` ABI byte buffer.
    pub fn write_to_abi_mem(&self, mem: &mut [u8]) {
        assert!(
            mem.len() >= abi::CPU_STATE_SIZE as usize,
            "CpuState ABI buffer too small"
        );

        for (i, val) in self.gpr.iter().enumerate() {
            let off = abi::CPU_GPR_OFF[i] as usize;
            mem[off..off + 8].copy_from_slice(&val.to_le_bytes());
        }

        let rip_off = abi::CPU_RIP_OFF as usize;
        mem[rip_off..rip_off + 8].copy_from_slice(&self.rip.to_le_bytes());

        // Bit 1 always reads as 1 on real hardware.
        let rflags = self.rflags | abi::RFLAGS_RESERVED1;
        let rflags_off = abi::CPU_RFLAGS_OFF as usize;
        mem[rflags_off..rflags_off + 8].copy_from_slice(&rflags.to_le_bytes());
    }

    #[must_use]
    pub fn from_cpu_state(cpu: &CpuState) -> Self {
        Self {
            gpr: cpu.gpr,
            rip: cpu.rip,
            rflags: cpu.rflags_snapshot(),
            fs_base: cpu.msr.fs_base,
            gs_base: cpu.msr.gs_base,
        }
    }

    pub fn write_to_cpu_state(&self, cpu: &mut CpuState) {
        cpu.gpr = self.gpr;
        cpu.rip = self.rip;
        cpu.set_rflags(self.rflags);
    }
}

pub fn execute_block<B: Tier1Bus>(block: &IrBlock, cpu_mem: &mut [u8], bus: &mut B) -> ExecResult {
    let mut cpu = TestCpu::from_abi_mem(cpu_mem);
    let res = execute_block_cpu(block, &mut cpu, bus);
    cpu.write_to_abi_mem(cpu_mem);
    res
}

/// Execute a pre-decoded IR block directly against a [`CpuState`], reading/writing
/// guest memory through `bus`.
///
/// This bypasses the WASM compilation pipeline entirely — the IR interpreter
/// executes the pre-decoded block instruction-by-instruction. While slower than
/// compiled WASM, it eliminates per-instruction decode and dispatch overhead,
/// giving a meaningful speedup for hot blocks without requiring a WASM engine
/// or memory synchronization.
pub fn execute_block_cpu_state<B: Tier1Bus>(
    block: &IrBlock,
    cpu: &mut CpuState,
    bus: &mut B,
) -> ExecResult {
    let mut test_cpu = TestCpu::from_cpu_state(cpu);
    let res = execute_block_cpu(block, &mut test_cpu, bus);
    test_cpu.write_to_cpu_state(cpu);
    res
}

/// Execute one IR block against an already-materialized [`TestCpu`].
///
/// Used by the machine's cached-block chain so a 3-block guest loop does not
/// copy [`CpuState`] in and out on every two-instruction `Jcc`.
pub fn execute_block_cpu<B: Tier1Bus>(
    block: &IrBlock,
    cpu: &mut TestCpu,
    bus: &mut B,
) -> ExecResult {
    let mut temps = Vec::new();
    execute_block_cpu_with_scratch(block, cpu, bus, &mut temps)
}

/// Execute a block reusing a caller-owned value buffer.
///
/// The buffer holds one entry per IR value, so it has to be sized per block —
/// which, allocated here, is a zeroing `calloc` and a `free` on *every block
/// execution*. That is a few dozen bytes at a time and it showed up as ~3% of a
/// profile spent in the allocator, on a path whose whole purpose is to be cheaper
/// than interpreting the same instructions one at a time.
///
/// Callers that execute blocks in a loop (a chain, most importantly) should hoist
/// one buffer out of the loop and pass it here: after the first block it is a
/// capacity check and a memset rather than a trip to the allocator.
pub fn execute_block_cpu_with_scratch<B: Tier1Bus>(
    block: &IrBlock,
    cpu: &mut TestCpu,
    bus: &mut B,
    temps: &mut Vec<u64>,
) -> ExecResult {
    temps.clear();
    temps.resize(block.value_types.len(), 0);
    let temps = &mut temps[..];

    for inst in &block.insts {
        match inst {
            // The taken path leaves the block; the fall-through carries on with the
            // guest registers still in temporaries.
            IrInst::SideExit { cond, target } => {
                if temps[cond.0 as usize] != 0 {
                    cpu.rip = *target;
                    return ExecResult::ExitToInterpreter { next_rip: *target };
                }
            }
            IrInst::Const { dst, value, width } => {
                temps[dst.0 as usize] = width.truncate(*value);
            }
            IrInst::ReadReg { dst, reg } => {
                let v = match *reg {
                    GuestReg::Rip => cpu.rip,
                    GuestReg::Gpr { reg, width, high8 } => read_gpr_part(cpu, reg, width, high8),
                    GuestReg::Flag(flag) => read_flag(cpu, flag) as u64,
                    GuestReg::SegBase(seg) => read_seg_base(cpu, seg),
                };
                temps[dst.0 as usize] = v;
            }
            IrInst::WriteReg { reg, src } => {
                let v = temps[src.0 as usize];
                match *reg {
                    GuestReg::Rip => cpu.rip = v,
                    GuestReg::Gpr { reg, width, high8 } => {
                        write_gpr_part(cpu, reg, width, high8, v)
                    }
                    GuestReg::Flag(flag) => write_flag(cpu, flag, v != 0),
                    // The IR never emits one: a segment base changes only through
                    // instructions Tier-1 declines to translate.
                    GuestReg::SegBase(_) => unreachable!("segment bases are read-only in Tier-1"),
                }
            }
            IrInst::Trunc { dst, src, width } => {
                let v = temps[src.0 as usize];
                temps[dst.0 as usize] = width.truncate(v);
            }
            IrInst::BitScan {
                dst,
                src,
                width,
                forward,
            } => {
                // Undefined for a zero source, and the translator never uses the
                // result there — it selects the destination's old value, which is what
                // Tier-0's `exec_bsf_bsr` does. Mirror `63 - leading_zeros` exactly.
                let v = temps[src.0 as usize];
                let idx = if *forward {
                    u64::from(v.trailing_zeros())
                } else {
                    63u64.wrapping_sub(u64::from(v.leading_zeros()))
                };
                temps[dst.0 as usize] = width.truncate(idx);
            }
            IrInst::Zext { dst, src, width } => {
                // A narrow value is already stored with its high bits clear, so
                // widening it is re-typing rather than any actual work.
                let v = temps[src.0 as usize];
                temps[dst.0 as usize] = width.truncate(v);
            }
            IrInst::Load { dst, addr, width } => {
                let a = temps[addr.0 as usize];
                temps[dst.0 as usize] = bus.read(a, *width);
            }
            IrInst::Store { addr, src, width } => {
                let a = temps[addr.0 as usize];
                let v = temps[src.0 as usize];
                bus.write(a, *width, v);
            }
            IrInst::BinOp {
                dst,
                op,
                lhs,
                rhs,
                width,
                flags,
            } => {
                let l = temps[lhs.0 as usize];
                let r = temps[rhs.0 as usize];
                let w = *width;
                // x86 shift counts are masked to 5 bits for 8/16/32-bit shifts and 6 bits for
                // 64-bit shifts (regardless of the operand width).
                let shift_mask: u32 = if w == Width::W64 { 63 } else { 31 };
                let shift_amt = (r as u32) & shift_mask;
                let (res, flag_vals) = match op {
                    BinOp::Add => {
                        let res = w.truncate(l.wrapping_add(r));
                        (res, Some(compute_add_flags(w, l, r, res)))
                    }
                    BinOp::Adc => {
                        let cin = read_flag(cpu, Flag::Cf) as u64;
                        let wide = (l as u128) + (r as u128) + (cin as u128);
                        let res = w.truncate(wide as u64);
                        (res, Some(compute_add_flags_cin(w, l, r, cin, res)))
                    }
                    BinOp::Sub => {
                        let res = w.truncate(l.wrapping_sub(r));
                        (res, Some(compute_sub_flags(w, l, r, res)))
                    }
                    BinOp::Sbb => {
                        let bin = read_flag(cpu, Flag::Cf) as u64;
                        let wide = (l as u128)
                            .wrapping_sub(r as u128)
                            .wrapping_sub(bin as u128);
                        let res = w.truncate(wide as u64);
                        (res, Some(compute_sub_flags_bin(w, l, r, bin, res)))
                    }
                    BinOp::And => {
                        let res = w.truncate(l & r);
                        (res, Some(compute_logic_flags(w, res)))
                    }
                    BinOp::Or => {
                        let res = w.truncate(l | r);
                        (res, Some(compute_logic_flags(w, res)))
                    }
                    BinOp::Xor => {
                        let res = w.truncate(l ^ r);
                        (res, Some(compute_logic_flags(w, res)))
                    }
                    BinOp::Shl => {
                        let res = w.truncate(l << shift_amt);
                        (res, None)
                    }
                    BinOp::Shr => {
                        let res = w.truncate(l >> shift_amt);
                        (res, None)
                    }
                    BinOp::Sar => {
                        let signed = w.sign_extend(w.truncate(l)) as i64;
                        let res = w.truncate((signed >> shift_amt) as u64);
                        (res, None)
                    }
                    BinOp::Rol => (rotate_left(w, l, shift_amt), None),
                    BinOp::Ror => (rotate_right(w, l, shift_amt), None),
                    BinOp::Bswap => (bswap_width(w, l), None),
                    BinOp::Mul => {
                        let mask = w.mask();
                        let lhs = l & mask;
                        let rhs = r & mask;
                        let bits = w.bits();
                        let wide = (lhs as u128).wrapping_mul(rhs as u128);
                        let res = w.truncate(wide as u64);
                        let sign = 1u64 << (bits - 1);
                        let lhs_s = if (lhs & sign) != 0 {
                            (lhs as i128) | !((mask) as i128)
                        } else {
                            lhs as i128
                        };
                        let rhs_s = if (rhs & sign) != 0 {
                            (rhs as i128) | !((mask) as i128)
                        } else {
                            rhs as i128
                        };
                        let full = lhs_s.wrapping_mul(rhs_s);
                        let trunc_s = if (res & sign) != 0 {
                            (res as i128) | !((mask) as i128)
                        } else {
                            res as i128
                        };
                        let overflow = full != trunc_s;
                        (
                            res,
                            Some(FlagVals {
                                cf: overflow,
                                pf: false,
                                af: false,
                                zf: false,
                                sf: false,
                                of: overflow,
                            }),
                        )
                    }
                };
                temps[dst.0 as usize] = res;
                if !flags.is_empty() {
                    if let Some(vals) = flag_vals {
                        // `imul` defines CF and OF and leaves SF, ZF, AF and PF as
                        // they were — undefined architecturally, and untouched by
                        // Tier-0, which is the behaviour the compiled tier has to
                        // match. Honouring a wider declared set here would make the
                        // IR mean two different things depending on who ran it.
                        let written = if matches!(op, BinOp::Mul) {
                            flags.intersect(FlagSet::CF.union(FlagSet::OF))
                        } else {
                            *flags
                        };
                        write_flagset(cpu, written, vals);
                    } else if matches!(op, BinOp::Shl | BinOp::Shr | BinOp::Sar) {
                        update_shift_flags(cpu, w, *op, l, shift_amt, res, *flags);
                    } else if matches!(op, BinOp::Rol | BinOp::Ror) {
                        update_rotate_flags(cpu, w, *op, l, shift_amt, res, *flags);
                    }
                }
            }
            IrInst::CmpFlags {
                lhs,
                rhs,
                width,
                flags,
            } => {
                let l = temps[lhs.0 as usize];
                let r = temps[rhs.0 as usize];
                let w = *width;
                let res = w.truncate(l.wrapping_sub(r));
                write_flagset(cpu, *flags, compute_sub_flags(w, l, r, res));
            }
            IrInst::TestFlags {
                lhs,
                rhs,
                width,
                flags,
            } => {
                let l = temps[lhs.0 as usize];
                let r = temps[rhs.0 as usize];
                let w = *width;
                let res = w.truncate(l & r);
                write_flagset(cpu, *flags, compute_logic_flags(w, res));
            }
            IrInst::EvalCond { dst, cond } => {
                temps[dst.0 as usize] = eval_cond(cpu, *cond) as u64;
            }
            IrInst::Select {
                dst,
                cond,
                if_true,
                if_false,
                width,
            } => {
                let c = temps[cond.0 as usize] != 0;
                let t = temps[if_true.0 as usize];
                let e = temps[if_false.0 as usize];
                temps[dst.0 as usize] = width.truncate(if c { t } else { e });
            }
            IrInst::Assist { rip, .. } => {
                // No machine here to perform it; this interpreter exists to validate
                // x86 -> IR translation, and the assist's effect is the host's.
                return ExecResult::ExitToInterpreter { next_rip: *rip };
            }
            IrInst::ReadTsc { dst } => {
                // This interpreter exists to validate x86 -> IR translation and has
                // no machine to ask, so it reports zero. A timestamp cannot be
                // compared against a second execution in any case, which is why
                // blocks that read one are excluded from the runtime verifier too.
                temps[dst.0 as usize] = 0;
            }
            IrInst::CallHelper { .. } => {
                // The Tier-1 debug interpreter is only used to validate x86→IR translation. It has
                // no access to the runtime helper implementations, so treat helper calls as a
                // conservative interpreter bailout (matching Tier-1 WASM codegen).
                return ExecResult::ExitToInterpreter { next_rip: cpu.rip };
            }
        }
    }

    match block.terminator {
        IrTerminator::Jump { target } => {
            cpu.rip = target;
            ExecResult::Continue
        }
        IrTerminator::CondJump {
            cond,
            target,
            fallthrough,
        } => {
            let c = temps[cond.0 as usize] != 0;
            cpu.rip = if c { target } else { fallthrough };
            ExecResult::Continue
        }
        IrTerminator::IndirectJump { target } => {
            cpu.rip = temps[target.0 as usize];
            ExecResult::Continue
        }
        IrTerminator::ExitToInterpreter { next_rip } => {
            cpu.rip = next_rip;
            ExecResult::ExitToInterpreter { next_rip }
        }
    }
}

#[inline]
fn read_gpr_part(cpu: &TestCpu, reg: aero_types::Gpr, width: Width, high8: bool) -> u64 {
    let idx = reg.as_u8() as usize;
    let val = cpu.gpr[idx];
    match width {
        Width::W8 => {
            if high8 {
                debug_assert!(matches!(
                    reg,
                    aero_types::Gpr::Rax
                        | aero_types::Gpr::Rcx
                        | aero_types::Gpr::Rdx
                        | aero_types::Gpr::Rbx
                ));
                (val >> 8) & 0xff
            } else {
                val & 0xff
            }
        }
        Width::W16 => val & 0xffff,
        Width::W32 => val & 0xffff_ffff,
        Width::W64 => val,
    }
}

#[inline]
fn write_gpr_part(cpu: &mut TestCpu, reg: aero_types::Gpr, width: Width, high8: bool, value: u64) {
    let idx = reg.as_u8() as usize;
    let prev = cpu.gpr[idx];
    let masked = width.truncate(value);
    cpu.gpr[idx] = match width {
        Width::W8 => {
            if high8 {
                debug_assert!(matches!(
                    reg,
                    aero_types::Gpr::Rax
                        | aero_types::Gpr::Rcx
                        | aero_types::Gpr::Rdx
                        | aero_types::Gpr::Rbx
                ));
                (prev & !0xff00) | ((masked & 0xff) << 8)
            } else {
                (prev & !0xff) | (masked & 0xff)
            }
        }
        Width::W16 => (prev & !0xffff) | (masked & 0xffff),
        // Writes to a 32-bit GPR clear the upper 32 bits, even in 64-bit mode.
        Width::W32 => masked & 0xffff_ffff,
        Width::W64 => masked,
    };
}

#[inline]
fn flag_mask(flag: Flag) -> u64 {
    match flag {
        Flag::Cf => RFLAGS_CF,
        Flag::Pf => RFLAGS_PF,
        Flag::Af => RFLAGS_AF,
        Flag::Zf => RFLAGS_ZF,
        Flag::Sf => RFLAGS_SF,
        Flag::Of => RFLAGS_OF,
    }
}

#[inline]
fn read_seg_base(cpu: &TestCpu, seg: crate::tier1::ir::SegBase) -> u64 {
    match seg {
        crate::tier1::ir::SegBase::Fs => cpu.fs_base,
        crate::tier1::ir::SegBase::Gs => cpu.gs_base,
    }
}

#[inline]
fn read_flag(cpu: &TestCpu, flag: Flag) -> bool {
    (cpu.rflags & flag_mask(flag)) != 0
}

#[inline]
fn write_flag(cpu: &mut TestCpu, flag: Flag, value: bool) {
    let mask = flag_mask(flag);
    if value {
        cpu.rflags |= mask;
    } else {
        cpu.rflags &= !mask;
    }
    cpu.rflags |= abi::RFLAGS_RESERVED1;
}

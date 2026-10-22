use super::BitOp;
use super::{
    Address, AluOp, DecodeError, DecodedInst, InstKind, Operand, Reg, SegmentPrefix, ShiftOp,
};
use aero_types::{Cond, Gpr, Width};

/// Prefix state that affects how the rest of an instruction decodes.
///
/// This is threaded to every operand decoder already, which is why the segment
/// override lives here rather than as a twenty-third argument: `Operand::Mem` is
/// built in one place, and that place needs to know.
#[derive(Debug, Clone, Copy)]
struct Rex {
    present: bool,
    w: bool,
    r: bool,
    x: bool,
    b: bool,
    /// `fs:`/`gs:` override, when one was present.
    seg: Option<SegmentPrefix>,
}

impl Rex {
    fn none() -> Self {
        Self {
            present: false,
            w: false,
            r: false,
            x: false,
            b: false,
            seg: None,
        }
    }

    fn from_byte(b: u8) -> Self {
        debug_assert!((0x40..=0x4f).contains(&b));
        Self {
            present: true,
            w: (b & 0x08) != 0,
            r: (b & 0x04) != 0,
            x: (b & 0x02) != 0,
            b: (b & 0x01) != 0,
            seg: None,
        }
    }
}

fn read_u8(bytes: &[u8], offset: usize) -> Result<u8, DecodeError> {
    bytes.get(offset).copied().ok_or(DecodeError {
        message: "unexpected EOF",
    })
}

fn read_le(bytes: &[u8], offset: usize, len: usize) -> Result<u64, DecodeError> {
    if bytes.len() < offset + len {
        return Err(DecodeError {
            message: "unexpected EOF",
        });
    }
    let mut out = 0u64;
    for i in 0..len {
        out |= (bytes[offset + i] as u64) << (i * 8);
    }
    Ok(out)
}

fn decode_gpr(code: u8) -> Result<Gpr, DecodeError> {
    Gpr::from_u4(code).ok_or(DecodeError {
        message: "invalid register encoding",
    })
}

fn decode_reg8(code: u8, rex_present: bool) -> Result<(Gpr, bool), DecodeError> {
    let code = code & 0x0f;
    if code >= 8 {
        return Ok((decode_gpr(code)?, false));
    }
    match (code, rex_present) {
        (0..=3, _) => Ok((decode_gpr(code)?, false)),
        (4..=7, true) => Ok((decode_gpr(code)?, false)),
        (4..=7, false) => Ok((decode_gpr(code - 4)?, true)),
        _ => Err(DecodeError {
            message: "invalid 8-bit register encoding",
        }),
    }
}

#[derive(Debug, Clone, Copy)]
struct ModRm {
    mod_bits: u8,
    reg: u8,
    rm: u8,
}

fn parse_modrm(byte: u8, rex: Rex) -> ModRm {
    let mod_bits = (byte >> 6) & 0x3;
    let reg = ((byte >> 3) & 0x7) | if rex.r { 8 } else { 0 };
    let rm = (byte & 0x7) | if rex.b { 8 } else { 0 };
    ModRm { mod_bits, reg, rm }
}

fn parse_sib(byte: u8, rex: Rex) -> (u8, u8, u8) {
    let scale_bits = (byte >> 6) & 0x3;
    let index = ((byte >> 3) & 0x7) | if rex.x { 8 } else { 0 };
    let base = (byte & 0x7) | if rex.b { 8 } else { 0 };
    (scale_bits, index, base)
}

fn op_width(bitness: u32, rex: Rex, operand_override: bool) -> Width {
    match bitness {
        64 => {
            if rex.w {
                Width::W64
            } else if operand_override {
                Width::W16
            } else {
                Width::W32
            }
        }
        32 => {
            if operand_override {
                Width::W16
            } else {
                Width::W32
            }
        }
        16 => {
            if operand_override {
                Width::W32
            } else {
                Width::W16
            }
        }
        _ => Width::W32,
    }
}

fn stack_width(bitness: u32, operand_override: bool) -> Width {
    match bitness {
        64 => {
            if operand_override {
                Width::W16
            } else {
                Width::W64
            }
        }
        32 => {
            if operand_override {
                Width::W16
            } else {
                Width::W32
            }
        }
        16 => {
            if operand_override {
                Width::W32
            } else {
                Width::W16
            }
        }
        _ => Width::W32,
    }
}

fn ip_mask(bitness: u32) -> u64 {
    match bitness {
        64 => u64::MAX,
        32 => 0xffff_ffff,
        16 => 0xffff,
        _ => u64::MAX,
    }
}

fn near_branch_disp_width(bitness: u32, operand_override: bool) -> Width {
    match bitness {
        // In 64-bit mode, near branches always use a 32-bit relative displacement.
        64 => Width::W32,
        // In 16/32-bit modes, the displacement size follows the operand-size attribute.
        32 => {
            if operand_override {
                Width::W16
            } else {
                Width::W32
            }
        }
        16 => {
            if operand_override {
                Width::W32
            } else {
                Width::W16
            }
        }
        _ => Width::W32,
    }
}

fn near_branch_ip_mask(bitness: u32, operand_override: bool) -> u64 {
    match bitness {
        // Near branches in 64-bit mode always update RIP.
        64 => u64::MAX,
        // In 32-bit mode, the operand-size override selects a 16-bit offset (IP) for near
        // branches.
        32 => {
            if operand_override {
                0xffff
            } else {
                0xffff_ffff
            }
        }
        // In 16-bit mode, the operand-size override selects a 32-bit offset (EIP) for near
        // branches, which Tier-1 does not currently model (we bail out in that case).
        16 => {
            if operand_override {
                0xffff_ffff
            } else {
                0xffff
            }
        }
        _ => u64::MAX,
    }
}

fn read_rel(bytes: &[u8], offset: usize, width: Width) -> Result<i64, DecodeError> {
    match width {
        Width::W16 => Ok(read_le(bytes, offset, 2)? as u16 as i16 as i64),
        Width::W32 => Ok(read_le(bytes, offset, 4)? as u32 as i32 as i64),
        _ => Err(DecodeError {
            message: "invalid rel width",
        }),
    }
}

fn sign_extend_imm(width: Width, imm: u64) -> u64 {
    width.sign_extend(width.truncate(imm))
}

fn group2_shift_op(group: u8, what: &'static str) -> Result<ShiftOp, DecodeError> {
    match group {
        0 => Ok(ShiftOp::Rol),
        1 => Ok(ShiftOp::Ror),
        4 => Ok(ShiftOp::Shl),
        5 => Ok(ShiftOp::Shr),
        7 => Ok(ShiftOp::Sar),
        _ => Err(DecodeError {
            message: match what {
                "0xC0" => "unsupported 0xC0 group",
                "0xC1" => "unsupported 0xC1 group",
                "0xD0" => "unsupported 0xD0 group",
                "0xD2" => "unsupported 0xD2 group",
                "0xD3" => "unsupported 0xD3 group",
                _ => "unsupported 0xD1 group",
            },
        }),
    }
}

fn decode_modrm_operand(
    bytes: &[u8],
    offset: &mut usize,
    bitness: u32,
    rex: Rex,
    rex_present: bool,
    width: Width,
) -> Result<(Operand, ModRm), DecodeError> {
    let modrm_byte = read_u8(bytes, *offset)?;
    *offset += 1;
    let modrm = parse_modrm(modrm_byte, rex);
    if modrm.mod_bits == 3 {
        let reg = if width == Width::W8 {
            let (gpr, high8) = decode_reg8(modrm.rm, rex_present)?;
            Reg { gpr, width, high8 }
        } else {
            Reg {
                gpr: decode_gpr(modrm.rm)?,
                width,
                high8: false,
            }
        };
        return Ok((Operand::Reg(reg), modrm));
    }

    // Memory operand.
    let mut base: Option<Gpr> = None;
    let mut index: Option<Gpr> = None;
    let mut scale: u8 = 1;
    let mut disp: i32 = 0;
    let mut rip_relative = false;

    // Address decoding depends on the address-size, which for this Tier1 decoder matches the
    // requested `bitness`.
    //
    // The address-size override prefix (`0x67`) is not currently supported; the Tier-1 decoder
    // conservatively returns `InstKind::Invalid` when it is present to avoid mis-decoding
    // ModRM/SIB operands.
    if bitness == 16 {
        // 16-bit ModRM addressing (no SIB byte). Encoding table:
        //   rm=0: [bx+si]   rm=4: [si]
        //   rm=1: [bx+di]   rm=5: [di]
        //   rm=2: [bp+si]   rm=6: [disp16] if mod=0 else [bp]
        //   rm=3: [bp+di]   rm=7: [bx]
        let rm = modrm.rm & 0x7;
        match rm {
            0 => {
                base = Some(Gpr::Rbx);
                index = Some(Gpr::Rsi);
            }
            1 => {
                base = Some(Gpr::Rbx);
                index = Some(Gpr::Rdi);
            }
            2 => {
                base = Some(Gpr::Rbp);
                index = Some(Gpr::Rsi);
            }
            3 => {
                base = Some(Gpr::Rbp);
                index = Some(Gpr::Rdi);
            }
            4 => base = Some(Gpr::Rsi),
            5 => base = Some(Gpr::Rdi),
            6 => {
                if modrm.mod_bits == 0 {
                    base = None;
                    index = None;
                    let disp16 = read_le(bytes, *offset, 2)? as u16;
                    *offset += 2;
                    // In `mod=0 rm=6`, the displacement is an absolute 16-bit address, not a
                    // signed offset.
                    disp = disp16 as i32;
                } else {
                    base = Some(Gpr::Rbp);
                }
            }
            7 => base = Some(Gpr::Rbx),
            _ => unreachable!(),
        }

        match modrm.mod_bits {
            0 => {}
            1 => {
                let d8 = read_u8(bytes, *offset)? as i8;
                *offset += 1;
                disp = disp.wrapping_add(d8 as i32);
            }
            2 => {
                let d16 = read_le(bytes, *offset, 2)? as u16;
                *offset += 2;
                disp = disp.wrapping_add(d16 as i16 as i32);
            }
            _ => unreachable!(),
        }
    } else {
        let rm_low3 = modrm.rm & 0x7;
        if rm_low3 == 4 {
            let sib_byte = read_u8(bytes, *offset)?;
            *offset += 1;
            let (scale_bits, index_code, base_code) = parse_sib(sib_byte, rex);
            scale = 1u8 << scale_bits;
            // Index 100b means "no index" only without REX.X. With it the field
            // names r12, a perfectly ordinary index register, so masking the REX bit
            // off before the comparison silently drops the index and turns
            // `[r13+r12*2]` into `[r13]`.
            if index_code != 4 {
                index = Some(decode_gpr(index_code)?);
            }
            if (base_code & 0x7) == 5 && modrm.mod_bits == 0 {
                base = None;
            } else {
                base = Some(decode_gpr(base_code)?);
            }
            if (base_code & 0x7) == 5 && modrm.mod_bits == 0 && base.is_none() {
                // No base, disp32 follows.
                let disp32 = read_le(bytes, *offset, 4)? as u32;
                *offset += 4;
                disp = disp32 as i32;
            }
        } else if rm_low3 == 5 && modrm.mod_bits == 0 {
            if bitness == 64 {
                // RIP-relative (64-bit mode).
                rip_relative = true;
            } else {
                // Absolute disp32 addressing (32-bit mode).
                base = None;
            }
            let disp32 = read_le(bytes, *offset, 4)? as u32;
            *offset += 4;
            disp = disp32 as i32;
        } else {
            base = Some(decode_gpr(modrm.rm)?);
        }

        match modrm.mod_bits {
            0 => {}
            1 => {
                let d8 = read_u8(bytes, *offset)? as i8;
                *offset += 1;
                disp = disp.wrapping_add(d8 as i32);
            }
            2 => {
                let d32 = read_le(bytes, *offset, 4)? as u32;
                *offset += 4;
                disp = disp.wrapping_add(d32 as i32);
            }
            _ => unreachable!(),
        }
    }

    Ok((
        Operand::Mem(Address {
            base,
            index,
            scale,
            disp,
            rip_relative,
            segment: rex.seg,
        }),
        modrm,
    ))
}

fn decode_reg_from_modrm(
    modrm: ModRm,
    rex_present: bool,
    width: Width,
) -> Result<Reg, DecodeError> {
    if width == Width::W8 {
        let (gpr, high8) = decode_reg8(modrm.reg, rex_present)?;
        Ok(Reg { gpr, width, high8 })
    } else {
        Ok(Reg {
            gpr: decode_gpr(modrm.reg)?,
            width,
            high8: false,
        })
    }
}

/// Decode a single instruction at `rip` from `bytes`.
///
/// The caller is expected to provide up to 15 bytes (the architectural maximum
/// length). If decoding fails, an [`InstKind::Invalid`] instruction is returned
/// with a conservative 1-byte length so front-ends can always make progress.
#[must_use]
pub fn decode_one(rip: u64, bytes: &[u8]) -> DecodedInst {
    decode_one_mode(rip, bytes, 64)
}

/// Decode a single instruction at `rip` from `bytes`, using the requested x86 bitness.
///
/// `bitness` must be one of 16, 32, or 64.
#[must_use]
pub fn decode_one_mode(rip: u64, bytes: &[u8], bitness: u32) -> DecodedInst {
    match decode_one_inner(rip, bytes, bitness) {
        Ok(inst) => inst,
        // Tier-1's decoder covers the instructions Tier-1 can translate. For anything
        // else the *length* still matters — it is where the instruction after it
        // begins, and a caller that means to step over this one needs that. Guessing
        // one byte is wrong for most of them: `out`, `cli`, `mov cr8` and the locked
        // compare-exchanges are all longer. The full decoder knows, so ask it.
        Err(_) => {
            let full = crate::decode(bytes, rip, bitness).ok();
            let len = full.as_ref().map_or(1, |d| d.len.max(1));
            let assistable = full
                .as_ref()
                .is_some_and(|d| !is_interrupt_related(d.instr.mnemonic()));
            DecodedInst {
                rip,
                len,
                kind: InstKind::Invalid,
                assistable,
                next_rip: rip.wrapping_add(u64::from(len)) & ip_mask(bitness),
            }
        }
    }
}

fn decode_one_inner(rip: u64, bytes: &[u8], bitness: u32) -> Result<DecodedInst, DecodeError> {
    if !matches!(bitness, 16 | 32 | 64) {
        return Err(DecodeError {
            message: "unsupported bitness",
        });
    }

    let mut offset = 0usize;
    let mut rex = Rex::none();
    let mut seg: Option<SegmentPrefix> = None;
    let mut operand_override = false;
    let mut address_override = false;
    let mut lock_prefix = false;

    loop {
        let b = read_u8(bytes, offset)?;
        // A REX prefix only has effect when it is the *last* prefix before the opcode.
        // `48 F2 B8 ...` is `MOV EAX, imm32` with an ignored REX, not `MOV RAX, imm64`:
        // the `F2` after the `48` discards it. Getting this wrong made the instruction
        // four bytes longer than it is, so the next one decoded from the middle of this
        // one — found by the proptest against iced, not by anything running.
        let is_rex = bitness == 64 && (0x40..=0x4f).contains(&b);
        match b {
            0x66 => {
                operand_override = true;
                offset += 1;
            }
            0x67 => {
                // Address-size override affects ModRM/SIB decoding and effective-address
                // masking. Tier-1 translation does not currently model per-instruction address
                // size, so we conservatively bail out to the interpreter if this prefix is
                // present.
                address_override = true;
                offset += 1;
            }
            0xf0 => {
                // LOCK. Aero runs its CPUs one at a time under a cooperative
                // BSP-driven scheduler in a single thread, so no two of them execute
                // concurrently and a read-modify-write is already indivisible with
                // respect to the others. The prefix is therefore recorded and the
                // instruction translated as the plain RMW it wraps — but only for
                // the forms LOCK is legal on, checked below, so that a prefix on
                // anything else goes to the interpreter rather than being dropped.
                lock_prefix = true;
                offset += 1;
            }
            0xf2 | 0xf3 => {
                // Ignored for this subset.
                offset += 1;
            }
            // Segment overrides. Only FS and GS have a base worth honouring in long
            // mode; the others are zero-based there, so ignoring them is correct.
            // Windows reaches its per-CPU block through `gs:`, and before this every
            // block containing one bailed — the largest single cause of them.
            0x64 => {
                seg = Some(SegmentPrefix::Fs);
                offset += 1;
            }
            0x65 => {
                seg = Some(SegmentPrefix::Gs);
                offset += 1;
            }
            0x2e | 0x36 | 0x3e | 0x26 if bitness == 64 => {
                // CS/SS/DS/ES are zero-based in long mode: no effect on the address.
                offset += 1;
            }
            0x40..=0x4f if bitness == 64 => {
                rex = Rex::from_byte(b);
                offset += 1;
            }
            _ => break,
        }
        if !is_rex {
            // A prefix followed this REX, so the REX is not the last one and does not
            // apply. Only reached when a prefix byte was consumed; `_ => break` leaves
            // the loop before it.
            rex = Rex::none();
        }
    }

    // The segment override is carried on the REX record for convenience, and unlike the
    // REX bits it survives whatever order the prefixes came in.
    rex.seg = seg;

    let opcode1 = read_u8(bytes, offset)?;
    offset += 1;
    if address_override {
        let next_rip = rip.wrapping_add(offset as u64) & ip_mask(bitness);
        return Ok(DecodedInst {
            rip,
            len: offset as u8,
            kind: InstKind::Invalid,
            next_rip,
                    assistable: true,
        });
    }

    let width = op_width(bitness, rex, operand_override);
    let stack_width = stack_width(bitness, operand_override);

    let mut next_rip_mask = ip_mask(bitness);
    let kind = match opcode1 {
        0x90 => {
            // 0x90 is `xchg (e)ax, (e)ax` which is a NOP. In 64-bit mode, REX.B can extend the
            // register operand (eg. `0x49 0x90` => `xchg eax, r8d`), which is not a NOP.
            //
            // Tier-1 does not currently model XCHG, so we only accept the canonical NOP form.
            if bitness == 64 && rex.b {
                InstKind::Invalid
            } else {
                InstKind::Nop
            }
        }
        0x40..=0x47 if bitness != 64 => {
            let reg_code = opcode1 - 0x40;
            let w = op_width(bitness, Rex::none(), operand_override);
            InstKind::Inc {
                dst: Operand::Reg(Reg {
                    gpr: decode_gpr(reg_code)?,
                    width: w,
                    high8: false,
                }),
                width: w,
            }
        }
        0x48..=0x4f if bitness != 64 => {
            let reg_code = opcode1 - 0x48;
            let w = op_width(bitness, Rex::none(), operand_override);
            InstKind::Dec {
                dst: Operand::Reg(Reg {
                    gpr: decode_gpr(reg_code)?,
                    width: w,
                    high8: false,
                }),
                width: w,
            }
        }
        0xb8..=0xbf => {
            let reg_code = (opcode1 - 0xb8) | if rex.b { 8 } else { 0 };
            let dst = Reg {
                gpr: decode_gpr(reg_code)?,
                width,
                high8: false,
            };
            let imm_len = width.bytes();
            let imm = read_le(bytes, offset, imm_len)?;
            offset += imm_len;
            InstKind::Mov {
                dst: Operand::Reg(dst),
                src: Operand::Imm(imm),
                width,
            }
        }
        0xb0..=0xb7 => {
            let reg_code = (opcode1 - 0xb0) | if rex.b { 8 } else { 0 };
            let (gpr, high8) = decode_reg8(reg_code, rex.present)?;
            let dst = Reg {
                gpr,
                width: Width::W8,
                high8,
            };
            let imm = read_u8(bytes, offset)? as u64;
            offset += 1;
            InstKind::Mov {
                dst: Operand::Reg(dst),
                src: Operand::Imm(imm),
                width: Width::W8,
            }
        }
        0x89 | 0x88 => {
            let w = if opcode1 == 0x88 { Width::W8 } else { width };
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
            let src_reg = decode_reg_from_modrm(modrm, rex.present, w)?;
            InstKind::Mov {
                dst,
                src: Operand::Reg(src_reg),
                width: w,
            }
        }
        0x8b | 0x8a => {
            let w = if opcode1 == 0x8a { Width::W8 } else { width };
            let (src, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
            let dst_reg = decode_reg_from_modrm(modrm, rex.present, w)?;
            InstKind::Mov {
                dst: Operand::Reg(dst_reg),
                src,
                width: w,
            }
        }
        0xc7 | 0xc6 => {
            let w = if opcode1 == 0xc6 { Width::W8 } else { width };
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
            let group = modrm.reg & 0x7;
            if group != 0 {
                return Err(DecodeError {
                    message: "unsupported group for C6/C7",
                });
            }
            let imm_len = if w == Width::W16 {
                2
            } else if w == Width::W8 {
                1
            } else {
                4
            };
            let imm_raw = read_le(bytes, offset, imm_len)?;
            offset += imm_len;
            let imm = if w == Width::W64 && imm_len == 4 {
                sign_extend_imm(Width::W32, imm_raw)
            } else if imm_len == 1 && w != Width::W8 {
                sign_extend_imm(Width::W8, imm_raw)
            } else {
                w.truncate(imm_raw)
            };
            InstKind::Mov {
                dst,
                src: Operand::Imm(imm),
                width: w,
            }
        }
        0x8d => {
            let (src, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W64)?;
            let Operand::Mem(addr) = src else {
                return Err(DecodeError {
                    message: "LEA requires memory operand",
                });
            };
            let dst = decode_reg_from_modrm(modrm, rex.present, width)?;
            InstKind::Lea { dst, addr, width }
        }
        0x00 | 0x01 | 0x02 | 0x03 | 0x08 | 0x09 | 0x0a | 0x0b | 0x10 | 0x11 | 0x12 | 0x13
        | 0x18 | 0x19 | 0x1a | 0x1b | 0x20 | 0x21 | 0x22 | 0x23 | 0x28 | 0x29 | 0x2a | 0x2b
        | 0x30 | 0x31 | 0x32 | 0x33 | 0x38 | 0x39 | 0x3a | 0x3b | 0x84 | 0x85 => {
            // Even opcodes in this group are the 8-bit forms. WinSetup's LZX
            // bitstream refill is `sub dil, cl` / `test dil, dil` (0x2A / 0x84).
            let w = if matches!(
                opcode1,
                0x00 | 0x02
                    | 0x08
                    | 0x0a
                    | 0x10
                    | 0x12
                    | 0x18
                    | 0x1a
                    | 0x20
                    | 0x22
                    | 0x28
                    | 0x2a
                    | 0x30
                    | 0x32
                    | 0x38
                    | 0x3a
                    | 0x84
            ) {
                Width::W8
            } else {
                width
            };
            let (rm_op, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
            let reg_op = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, w)?);

            let (op, is_cmp, is_test, dst, src) = match opcode1 {
                0x00 | 0x01 => (AluOp::Add, false, false, rm_op, reg_op),
                0x02 | 0x03 => (AluOp::Add, false, false, reg_op, rm_op),
                0x10 | 0x11 => (AluOp::Adc, false, false, rm_op, reg_op),
                0x12 | 0x13 => (AluOp::Adc, false, false, reg_op, rm_op),
                0x18 | 0x19 => (AluOp::Sbb, false, false, rm_op, reg_op),
                0x1a | 0x1b => (AluOp::Sbb, false, false, reg_op, rm_op),
                0x20 | 0x21 => (AluOp::And, false, false, rm_op, reg_op),
                0x22 | 0x23 => (AluOp::And, false, false, reg_op, rm_op),
                0x08 | 0x09 => (AluOp::Or, false, false, rm_op, reg_op),
                0x0a | 0x0b => (AluOp::Or, false, false, reg_op, rm_op),
                0x30 | 0x31 => (AluOp::Xor, false, false, rm_op, reg_op),
                0x32 | 0x33 => (AluOp::Xor, false, false, reg_op, rm_op),
                0x28 | 0x29 => (AluOp::Sub, false, false, rm_op, reg_op),
                0x2a | 0x2b => (AluOp::Sub, false, false, reg_op, rm_op),
                0x38 | 0x39 => (AluOp::Sub, true, false, rm_op, reg_op),
                0x3a | 0x3b => (AluOp::Sub, true, false, reg_op, rm_op),
                0x84 | 0x85 => (AluOp::And, false, true, rm_op, reg_op),
                _ => unreachable!(),
            };

            if is_cmp {
                InstKind::Cmp {
                    lhs: dst,
                    rhs: src,
                    width: w,
                }
            } else if is_test {
                InstKind::Test {
                    lhs: dst,
                    rhs: src,
                    width: w,
                }
            } else {
                InstKind::Alu {
                    op,
                    dst,
                    src,
                    width: w,
                }
            }
        }
        // Group2 shifts (0xC0/0xC1/0xD0/0xD1) are decoded below as `InstKind::Shift` so Tier1
        // can keep the shift count as a `u8` instead of embedding it into an `Operand::Imm`.
        // Accumulator-with-byte-immediate forms: `op AL, imm8`. The word-width
        // siblings (0x05, 0x0D, ... 0x3D, 0xA9) are handled below; these read a
        // one-byte immediate instead, which is why they cannot share that arm.
        0x04 | 0x0c | 0x14 | 0x1c | 0x24 | 0x2c | 0x34 | 0x3c | 0xa8 => {
            let (op, is_cmp, is_test) = match opcode1 {
                0x04 => (AluOp::Add, false, false),
                0x0c => (AluOp::Or, false, false),
                0x14 => (AluOp::Adc, false, false),
                0x1c => (AluOp::Sbb, false, false),
                0x24 => (AluOp::And, false, false),
                0x2c => (AluOp::Sub, false, false),
                0x34 => (AluOp::Xor, false, false),
                0x3c => (AluOp::Sub, true, false),
                0xa8 => (AluOp::And, false, true),
                _ => unreachable!(),
            };
            let imm = read_le(bytes, offset, 1)?;
            offset += 1;
            // Always AL: these opcodes have no ModRM, so REX.B cannot redirect them
            // to SPL/BPL/SIL/DIL the way a byte register operand otherwise would.
            let al = Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W8,
                high8: false,
            });
            if is_cmp {
                InstKind::Cmp {
                    lhs: al,
                    rhs: Operand::Imm(imm),
                    width: Width::W8,
                }
            } else if is_test {
                InstKind::Test {
                    lhs: al,
                    rhs: Operand::Imm(imm),
                    width: Width::W8,
                }
            } else {
                InstKind::Alu {
                    op,
                    dst: al,
                    src: Operand::Imm(imm),
                    width: Width::W8,
                }
            }
        }
        0x05 | 0x25 | 0x0d | 0x35 | 0x2d | 0x3d | 0xa9 => {
            let (op, is_cmp, is_test, acc) = match opcode1 {
                0x05 => (AluOp::Add, false, false, Gpr::Rax),
                0x25 => (AluOp::And, false, false, Gpr::Rax),
                0x0d => (AluOp::Or, false, false, Gpr::Rax),
                0x35 => (AluOp::Xor, false, false, Gpr::Rax),
                0x2d => (AluOp::Sub, false, false, Gpr::Rax),
                0x3d => (AluOp::Sub, true, false, Gpr::Rax),
                0xa9 => (AluOp::And, false, true, Gpr::Rax),
                _ => unreachable!(),
            };
            // Immediate size follows the operand-size attribute in 16/32-bit modes; in 64-bit
            // mode the immediate is always 32-bit and sign-extended if the destination is 64.
            let imm = if width == Width::W16 {
                let imm16 = read_le(bytes, offset, 2)? as u16;
                offset += 2;
                imm16 as u64
            } else {
                let imm32 = read_le(bytes, offset, 4)? as u32;
                offset += 4;
                if width == Width::W64 {
                    sign_extend_imm(Width::W32, imm32 as u64)
                } else {
                    imm32 as u64
                }
            };
            let acc_reg = Operand::Reg(Reg {
                gpr: acc,
                width,
                high8: false,
            });
            if is_cmp {
                InstKind::Cmp {
                    lhs: acc_reg,
                    rhs: Operand::Imm(imm),
                    width,
                }
            } else if is_test {
                InstKind::Test {
                    lhs: acc_reg,
                    rhs: Operand::Imm(imm),
                    width,
                }
            } else {
                InstKind::Alu {
                    op,
                    dst: acc_reg,
                    src: Operand::Imm(imm),
                    width,
                }
            }
        }
        0x69 | 0x6b => {
            // IMUL r, r/m, imm16/32 (0x69) or imm8 (0x6B). Destination is the
            // ModRM.reg field; the r/m operand is the multiplicand.
            let (src, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            let dst = decode_reg_from_modrm(modrm, rex.present, width)?;
            let imm = if opcode1 == 0x6b {
                let imm8 = read_u8(bytes, offset)? as i8 as i64 as u64;
                offset += 1;
                width.truncate(imm8)
            } else if width == Width::W16 {
                let imm16 = read_le(bytes, offset, 2)? as u16;
                offset += 2;
                imm16 as u64
            } else {
                let imm32 = read_le(bytes, offset, 4)? as u32;
                offset += 4;
                if width == Width::W64 {
                    sign_extend_imm(Width::W32, imm32 as u64)
                } else {
                    imm32 as u64
                }
            };
            InstKind::Imul3 {
                dst,
                src,
                imm,
                width,
            }
        }
        0x80 | 0x81 | 0x83 => {
            // 0x80 is the 8-bit r/m8,imm8 group (WinSetup expand scans
            // `cmp byte [rdx], 0xe8` and bitstream refill uses `add dil, 0x10`).
            // 0x83 sign-extends imm8; 0x81 uses imm16/imm32.
            let width = if opcode1 == 0x80 { Width::W8 } else { width };
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            let group = modrm.reg & 0x7;
            let imm = if opcode1 == 0x80 {
                let imm8 = read_u8(bytes, offset)?;
                offset += 1;
                imm8 as u64
            } else if opcode1 == 0x83 {
                let imm8 = read_u8(bytes, offset)? as i8 as i64 as u64;
                offset += 1;
                width.truncate(imm8)
            } else if width == Width::W16 {
                let imm16 = read_le(bytes, offset, 2)? as u16;
                offset += 2;
                imm16 as u64
            } else {
                let imm32 = read_le(bytes, offset, 4)? as u32;
                offset += 4;
                if width == Width::W64 {
                    sign_extend_imm(Width::W32, imm32 as u64)
                } else {
                    imm32 as u64
                }
            };

            match group {
                0 => InstKind::Alu {
                    op: AluOp::Add,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                1 => InstKind::Alu {
                    op: AluOp::Or,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                2 => InstKind::Alu {
                    op: AluOp::Adc,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                3 => InstKind::Alu {
                    op: AluOp::Sbb,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                4 => InstKind::Alu {
                    op: AluOp::And,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                5 => InstKind::Alu {
                    op: AluOp::Sub,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                6 => InstKind::Alu {
                    op: AluOp::Xor,
                    dst,
                    src: Operand::Imm(imm),
                    width,
                },
                7 => InstKind::Cmp {
                    lhs: dst,
                    rhs: Operand::Imm(imm),
                    width,
                },
                _ => {
                    return Err(DecodeError {
                        message: "unsupported 0x80/0x81/0x83 group",
                    })
                }
            }
        }
        0xd1 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            InstKind::Shift {
                op: group2_shift_op(modrm.reg & 0x7, "0xD1")?,
                dst,
                count: 1,
                width,
            }
        }
        0xd0 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W8)?;
            InstKind::Shift {
                op: group2_shift_op(modrm.reg & 0x7, "0xD0")?,
                dst,
                count: 1,
                width: Width::W8,
            }
        }
        0xd2 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W8)?;
            InstKind::ShiftCl {
                op: group2_shift_op(modrm.reg & 0x7, "0xD2")?,
                dst,
                width: Width::W8,
            }
        }
        0xd3 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            InstKind::ShiftCl {
                op: group2_shift_op(modrm.reg & 0x7, "0xD3")?,
                dst,
                width,
            }
        }
        0xc1 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            let op = group2_shift_op(modrm.reg & 0x7, "0xC1")?;
            let imm8 = read_u8(bytes, offset)?;
            offset += 1;
            InstKind::Shift {
                op,
                dst,
                count: imm8,
                width,
            }
        }
        0xc0 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W8)?;
            let op = group2_shift_op(modrm.reg & 0x7, "0xC0")?;
            let imm8 = read_u8(bytes, offset)?;
            offset += 1;
            InstKind::Shift {
                op,
                dst,
                count: imm8,
                width: Width::W8,
            }
        }
        0xf6 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W8)?;
            match modrm.reg & 0x7 {
                // `/0` and `/1` are both TEST r/m8, imm8. Common enough in real code
                // that leaving it out walls off whole blocks: a block Tier-1 cannot
                // finish stops every chain that reaches it.
                0 | 1 => {
                    let imm = read_le(bytes, offset, 1)?;
                    offset += 1;
                    InstKind::Test {
                        lhs: dst,
                        rhs: Operand::Imm(imm),
                        width: Width::W8,
                    }
                }
                2 => InstKind::Not {
                    dst,
                    width: Width::W8,
                },
                3 => InstKind::Neg {
                    dst,
                    width: Width::W8,
                },
                _ => {
                    return Err(DecodeError {
                        message: "unsupported 0xF6 group",
                    })
                }
            }
        }
        0xf7 => {
            let (dst, modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
            match modrm.reg & 0x7 {
                // TEST r/m, imm. The immediate follows the operand-size attribute,
                // except in 64-bit mode where it is imm32 sign-extended.
                0 | 1 => {
                    let imm = if width == Width::W16 {
                        let imm16 = read_le(bytes, offset, 2)? as u16;
                        offset += 2;
                        imm16 as u64
                    } else {
                        let imm32 = read_le(bytes, offset, 4)? as u32;
                        offset += 4;
                        if width == Width::W64 {
                            sign_extend_imm(Width::W32, imm32 as u64)
                        } else {
                            imm32 as u64
                        }
                    };
                    InstKind::Test {
                        lhs: dst,
                        rhs: Operand::Imm(imm),
                        width,
                    }
                }
                2 => InstKind::Not { dst, width },
                3 => InstKind::Neg { dst, width },
                _ => {
                    return Err(DecodeError {
                        message: "unsupported 0xF7 group",
                    })
                }
            }
        }
        0xff => {
            // Operand size for INC/DEC (groups 0/1) follows the instruction operand width,
            // while PUSH (group 6) uses the mode's stack width.
            let modrm_byte = read_u8(bytes, offset)?;
            let group = (modrm_byte >> 3) & 0x7;
            // Near call/jump default to the stack width in long mode, as PUSH does.
            let operand_width = if matches!(group, 2 | 4 | 6) {
                stack_width
            } else {
                width
            };
            let (opnd, _modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, operand_width)?;
            match group {
                0 => InstKind::Inc { dst: opnd, width },
                1 => InstKind::Dec { dst: opnd, width },
                6 => {
                    if operand_override {
                        // See `0x50..=0x57` case below.
                        InstKind::Invalid
                    } else {
                        InstKind::Push { src: opnd }
                    }
                }
                // Near indirect call/jump. `call qword [rip+disp32]` alone is the
                // second-largest wall in a Windows desktop session: it is how every
                // import-table call dispatches.
                2 | 4 => {
                    if operand_override {
                        // The branch operand size would no longer match the stack
                        // width the push assumes; see `0x50..=0x57` below.
                        InstKind::Invalid
                    } else if group == 2 {
                        InstKind::CallIndirect { target: opnd }
                    } else {
                        InstKind::JmpIndirect { target: opnd }
                    }
                }
                _ => {
                    return Err(DecodeError {
                        message: "unsupported 0xFF group",
                    })
                }
            }
        }
        0x50..=0x57 => {
            if operand_override {
                // Operand-size overridden PUSH/POP change the stack operand width (16-bit in
                // 32/64-bit modes, 32-bit in 16-bit mode). Tier-1 translation currently assumes
                // the stack width is derived solely from the block bitness, so we conservatively
                // bail out to the interpreter instead of miscompiling.
                InstKind::Invalid
            } else {
                let reg_code = (opcode1 - 0x50) | if rex.b { 8 } else { 0 };
                let reg = Operand::Reg(Reg {
                    gpr: decode_gpr(reg_code)?,
                    width: stack_width,
                    high8: false,
                });
                InstKind::Push { src: reg }
            }
        }
        0x58..=0x5f => {
            if operand_override {
                // See `0x50..=0x57` case above.
                InstKind::Invalid
            } else {
                let reg_code = (opcode1 - 0x58) | if rex.b { 8 } else { 0 };
                let reg = Operand::Reg(Reg {
                    gpr: decode_gpr(reg_code)?,
                    width: stack_width,
                    high8: false,
                });
                InstKind::Pop { dst: reg }
            }
        }
        0x6a => {
            let imm8 = read_u8(bytes, offset)? as i8 as i64 as u64;
            offset += 1;
            if operand_override {
                InstKind::Invalid
            } else {
                InstKind::Push {
                    src: Operand::Imm(imm8),
                }
            }
        }
        0x68 => {
            // PUSH imm16/imm32: immediate length follows the operand-size attribute in
            // 16/32-bit modes; in 64-bit mode the immediate is always 32-bit and
            // sign-extended to 64.
            let imm = if stack_width == Width::W16 {
                let imm16 = read_le(bytes, offset, 2)? as u16;
                offset += 2;
                imm16 as u64
            } else {
                let imm32 = read_le(bytes, offset, 4)? as u32;
                offset += 4;
                if stack_width == Width::W64 {
                    sign_extend_imm(Width::W32, imm32 as u64)
                } else {
                    imm32 as u64
                }
            };
            if operand_override {
                InstKind::Invalid
            } else {
                InstKind::Push {
                    src: Operand::Imm(imm),
                }
            }
        }
        0xe9 => {
            let rel_width = near_branch_disp_width(bitness, operand_override);
            let rel = read_rel(bytes, offset, rel_width)?;
            offset += rel_width.bytes();
            if bitness == 16 && operand_override {
                // In 16-bit mode, the operand-size override selects a 32-bit near-branch offset
                // (EIP) which can escape the 16-bit IP space. Tier-1 currently assumes a fixed
                // IP width derived from the block bitness, so we bail out to the interpreter
                // instead of miscompiling.
                InstKind::Invalid
            } else {
                let mask = near_branch_ip_mask(bitness, operand_override);
                if bitness == 32 && operand_override {
                    next_rip_mask = mask;
                }
                let target = (rip + offset as u64).wrapping_add(rel as u64) & mask;
                InstKind::JmpRel { target }
            }
        }
        0xeb => {
            let rel8 = read_u8(bytes, offset)? as i8;
            offset += 1;
            if bitness == 16 && operand_override {
                // In 16-bit mode, the operand-size override selects a 32-bit IP (EIP) for near
                // branches, which Tier-1 does not currently model (see `near_branch_ip_mask`).
                InstKind::Invalid
            } else {
                let mask = near_branch_ip_mask(bitness, operand_override);
                if bitness == 32 && operand_override {
                    next_rip_mask = mask;
                }
                let target = (rip + offset as u64).wrapping_add(rel8 as i64 as u64) & mask;
                InstKind::JmpRel { target }
            }
        }
        0xe8 => {
            match bitness {
                // In 64-bit mode, CALL rel32 is the only near relative encoding.
                64 => {
                    let rel32 = read_rel(bytes, offset, Width::W32)?;
                    offset += 4;
                    let target =
                        (rip + offset as u64).wrapping_add(rel32 as u64) & ip_mask(bitness);
                    InstKind::CallRel { target }
                }
                32 => {
                    // In 32-bit mode, the CALL displacement and pushed return-address size
                    // follow the operand-size attribute (rel32/ret32 by default, rel16/ret16
                    // when overridden). Tier-1 translation does not currently model per-call
                    // return-address width, so we decode the full instruction length but bail
                    // out to the interpreter when the operand-size override is present.
                    let rel_width = if operand_override {
                        Width::W16
                    } else {
                        Width::W32
                    };
                    let rel = read_rel(bytes, offset, rel_width)?;
                    offset += rel_width.bytes();
                    if operand_override {
                        InstKind::Invalid
                    } else {
                        let target =
                            (rip + offset as u64).wrapping_add(rel as u64) & ip_mask(bitness);
                        InstKind::CallRel { target }
                    }
                }
                16 => {
                    // In 16-bit mode, CALL uses rel16/ret16 by default, and rel32/ret32 when
                    // the operand-size override is present. Tier-1 translation does not
                    // currently model per-call return-address width, so we decode the full
                    // instruction length but bail out to the interpreter for the overridden
                    // form.
                    let rel_width = if operand_override {
                        Width::W32
                    } else {
                        Width::W16
                    };
                    let rel = read_rel(bytes, offset, rel_width)?;
                    offset += rel_width.bytes();
                    if operand_override {
                        InstKind::Invalid
                    } else {
                        let target =
                            (rip + offset as u64).wrapping_add(rel as u64) & ip_mask(bitness);
                        InstKind::CallRel { target }
                    }
                }
                _ => unreachable!(),
            }
        }
        0xc3 => {
            if operand_override && bitness != 64 {
                // Operand-size overridden RET changes the popped return-address width in
                // 16/32-bit modes. Tier-1 translation assumes stack width == block bitness, so
                // bail out to the interpreter to avoid miscompilation.
                InstKind::Invalid
            } else {
                InstKind::Ret { pop: 0 }
            }
        }
        0x63 => {
            // MOVSXD r64, r/m32. Without REX.W it is a plain 32-bit move, which the
            // interpreter can have rather than a special case here.
            if width != Width::W64 {
                InstKind::Invalid
            } else {
                let (src, modrm) =
                    decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W32)?;
                let dst = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, Width::W64)?);
                InstKind::MovSx {
                    dst,
                    src,
                    dst_width: Width::W64,
                    src_width: Width::W32,
                }
            }
        }
        0xfe => {
            // INC/DEC r/m8. The wider forms live at 0xFF; this is the byte case.
            let modrm_byte = read_u8(bytes, offset)?;
            let group = (modrm_byte >> 3) & 0x7;
            let (opnd, _modrm) =
                decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, Width::W8)?;
            match group {
                0 => InstKind::Inc {
                    dst: opnd,
                    width: Width::W8,
                },
                1 => InstKind::Dec {
                    dst: opnd,
                    width: Width::W8,
                },
                _ => InstKind::Invalid,
            }
        }
        0xc2 => {
            // RET imm16: pop the return address, then release `imm16` more bytes of
            // arguments. Common in stdcall-convention Windows code, and a block
            // ending in one is a wall every chain reaching it stops at.
            if operand_override && bitness != 64 {
                // See `0xC3` above.
                InstKind::Invalid
            } else {
                let pop = read_le(bytes, offset, 2)? as u16;
                offset += 2;
                InstKind::Ret { pop }
            }
        }
        0x70..=0x7f => {
            let cc = opcode1 - 0x70;
            let cond = Cond::from_cc(cc).ok_or(DecodeError {
                message: "invalid condition code",
            })?;
            let rel8 = read_u8(bytes, offset)? as i8;
            offset += 1;
            if bitness == 16 && operand_override {
                // See `0xEB` case above.
                InstKind::Invalid
            } else {
                let mask = near_branch_ip_mask(bitness, operand_override);
                if bitness == 32 && operand_override {
                    next_rip_mask = mask;
                }
                let target = (rip + offset as u64).wrapping_add(rel8 as i64 as u64) & mask;
                InstKind::JccRel { cond, target }
            }
        }
        0x0f => {
            let opcode2 = read_u8(bytes, offset)?;
            offset += 1;
            match opcode2 {
                0x80..=0x8f => {
                    let cc = opcode2 - 0x80;
                    let cond = Cond::from_cc(cc).ok_or(DecodeError {
                        message: "invalid condition code",
                    })?;
                    let rel_width = near_branch_disp_width(bitness, operand_override);
                    let rel = read_rel(bytes, offset, rel_width)?;
                    offset += rel_width.bytes();
                    if bitness == 16 && operand_override {
                        // See `0xE9` case above.
                        InstKind::Invalid
                    } else {
                        let mask = near_branch_ip_mask(bitness, operand_override);
                        if bitness == 32 && operand_override {
                            next_rip_mask = mask;
                        }
                        let target = (rip + offset as u64).wrapping_add(rel as u64) & mask;
                        InstKind::JccRel { cond, target }
                    }
                }
                0x90..=0x9f => {
                    let cc = opcode2 - 0x90;
                    let cond = Cond::from_cc(cc).ok_or(DecodeError {
                        message: "invalid condition code",
                    })?;
                    let (dst, _modrm) = decode_modrm_operand(
                        bytes,
                        &mut offset,
                        bitness,
                        rex,
                        rex.present,
                        Width::W8,
                    )?;
                    InstKind::Setcc { cond, dst }
                }
                0x1f => {
                    // Multi-byte NOP (0F 1F /0).
                    // The operand is ignored, but we still need to parse ModRM/SIB/displacement
                    // to compute the correct instruction length.
                    let (_opnd, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
                    let group = modrm.reg & 0x7;
                    if group != 0 {
                        InstKind::Invalid
                    } else {
                        InstKind::Nop
                    }
                }
                0x40..=0x4f => {
                    let cc = opcode2 - 0x40;
                    let cond = Cond::from_cc(cc).ok_or(DecodeError {
                        message: "invalid condition code",
                    })?;
                    let (src, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
                    let dst = decode_reg_from_modrm(modrm, rex.present, width)?;
                    InstKind::Cmovcc {
                        cond,
                        dst,
                        src,
                        width,
                    }
                }
                0x31 => InstKind::Rdtsc,
                // LFENCE/MFENCE/SFENCE. Aero runs one CPU at a time, so ordering
                // between them is not observable and these are hints.
                0xae if rmw_ops_enabled() => {
                    let modrm_byte = read_u8(bytes, offset)?;
                    let group = (modrm_byte >> 3) & 0x7;
                    if (modrm_byte >> 6) == 0b11 && matches!(group, 5 | 6 | 7) {
                        offset += 1;
                        InstKind::Nop
                    } else {
                        InstKind::Invalid
                    }
                }
                // XADD r/m, r
                0xc0 | 0xc1 if rmw_ops_enabled() => {
                    let w = if opcode2 == 0xc0 { Width::W8 } else { width };
                    let (dst, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
                    let src = decode_reg_from_modrm(modrm, rex.present, w)?;
                    InstKind::Xadd { dst, src, width: w }
                }
                // CMPXCHG r/m, r
                0xb0 | 0xb1 if rmw_ops_enabled() => {
                    let w = if opcode2 == 0xb0 { Width::W8 } else { width };
                    let (dst, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, w)?;
                    let src = decode_reg_from_modrm(modrm, rex.present, w)?;
                    InstKind::Cmpxchg { dst, src, width: w }
                }
                // PREFETCH/PREFETCHW/PREFETCHNTA and friends: architecturally hints,
                // and between them the single largest entry left on the coverage
                // wall. The ModRM still has to be consumed to get the length right.
                0x0d | 0x18 if !hint_ops_disabled() => {
                    let (_opnd, _modrm) = decode_modrm_operand(
                        bytes,
                        &mut offset,
                        bitness,
                        rex,
                        rex.present,
                        Width::W8,
                    )?;
                    InstKind::Nop
                }
                // MOVNTI: a store, with a non-temporal hint that is only a hint.
                0xc3 if !hint_ops_disabled() => {
                    if width == Width::W8 || width == Width::W16 {
                        InstKind::Invalid
                    } else {
                        let (dst, modrm) = decode_modrm_operand(
                            bytes,
                            &mut offset,
                            bitness,
                            rex,
                            rex.present,
                            width,
                        )?;
                        if matches!(dst, Operand::Reg(_)) {
                            // Register destination is not a valid encoding.
                            InstKind::Invalid
                        } else {
                            let src = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, width)?);
                            InstKind::Mov { dst, src, width }
                        }
                    }
                }
                // BT/BTS/BTR/BTC with a register bit offset.
                0xa3 | 0xab | 0xb3 | 0xbb if !hint_ops_disabled() => {
                    let op = match opcode2 {
                        0xa3 => BitOp::Test,
                        0xab => BitOp::Set,
                        0xb3 => BitOp::Reset,
                        _ => BitOp::Complement,
                    };
                    let (dst, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
                    let index = decode_reg_from_modrm(modrm, rex.present, width)?;
                    InstKind::BitTestReg {
                        op,
                        dst,
                        index,
                        width,
                    }
                }
                0xba => {
                    // Group 8: BT/BTS/BTR/BTC with an 8-bit immediate offset.
                    // `lock bts qword [rdi+0x88], imm8` is the busiest entry left on
                    // the coverage wall.
                    let modrm_byte = read_u8(bytes, offset)?;
                    let group = (modrm_byte >> 3) & 0x7;
                    let op = match group {
                        4 => Some(BitOp::Test),
                        5 => Some(BitOp::Set),
                        6 => Some(BitOp::Reset),
                        7 => Some(BitOp::Complement),
                        _ => None,
                    };
                    match op {
                        None => InstKind::Invalid,
                        Some(op) => {
                            let (dst, _modrm) = decode_modrm_operand(
                                bytes,
                                &mut offset,
                                bitness,
                                rex,
                                rex.present,
                                width,
                            )?;
                            let imm = read_u8(bytes, offset)?;
                            offset += 1;
                            // Intel adjusts the address for an offset past the
                            // operand size, and Tier-0 does it for the immediate
                            // form too; with a constant offset the whole adjustment
                            // folds here. `bits` is a power of two, so the euclidean
                            // divide and remainder are a shift and a mask.
                            let bits = width.bits();
                            let bit = (u32::from(imm) % bits) as u8;
                            let word_off = (u32::from(imm) / bits) as i32;
                            let dst = match dst {
                                Operand::Mem(mut addr) if word_off != 0 => {
                                    let unit_bytes = (bits / 8) as i32;
                                    addr.disp = addr.disp.wrapping_add(word_off * unit_bytes);
                                    Operand::Mem(addr)
                                }
                                other => other,
                            };
                            InstKind::BitTest {
                                op,
                                dst,
                                bit,
                                width,
                            }
                        }
                    }
                }
                0xbc | 0xbd if bitscan_enabled() => {
                    // BSF (0xBC) / BSR (0xBD): `reg` is the destination, `r/m` the
                    // source, both at the operand width. ~17,400 wanted executions of
                    // blocks refused for want of these, the largest translatable
                    // entry left on the coverage wall.
                    let (src, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
                    let dst = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, width)?);
                    InstKind::BitScan {
                        dst,
                        src,
                        width,
                        forward: opcode2 == 0xbc,
                    }
                }
                0xbe | 0xbf => {
                    // MOVSX r, r/m8 (0xBE) and r, r/m16 (0xBF). The zero-extending
                    // pair a few lines down was already taken; these are just as
                    // common in Windows code that handles `char` and `short`.
                    let src_width = if opcode2 == 0xbe {
                        Width::W8
                    } else {
                        Width::W16
                    };
                    if opcode2 == 0xbf && width == Width::W16 {
                        // A 16-bit destination would be a plain move.
                        InstKind::Invalid
                    } else {
                        let (src, modrm) = decode_modrm_operand(
                            bytes,
                            &mut offset,
                            bitness,
                            rex,
                            rex.present,
                            src_width,
                        )?;
                        let dst = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, width)?);
                        InstKind::MovSx {
                            dst,
                            src,
                            dst_width: width,
                            src_width,
                        }
                    }
                }
                0xb6 | 0xb7 => {
                    // MOVZX r, r/m8 (0xB6) and r, r/m16 (0xB7). Among the most
                    // common instructions in compiled Windows code, and a block
                    // Tier-1 cannot finish stops every chain reaching it.
                    let src_width = if opcode2 == 0xb6 {
                        Width::W8
                    } else {
                        Width::W16
                    };
                    // A 16-bit destination for 0xB7 would be a no-op move; the
                    // interpreter can have that rather than a special case here.
                    if opcode2 == 0xb7 && width == Width::W16 {
                        InstKind::Invalid
                    } else {
                        let (src, modrm) = decode_modrm_operand(
                            bytes,
                            &mut offset,
                            bitness,
                            rex,
                            rex.present,
                            src_width,
                        )?;
                        let dst = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, width)?);
                        InstKind::MovZx {
                            dst,
                            src,
                            dst_width: width,
                            src_width,
                        }
                    }
                }
                0xaf => {
                    // IMUL r, r/m  (two-operand). Destination is ModRM.reg.
                    let (src, modrm) =
                        decode_modrm_operand(bytes, &mut offset, bitness, rex, rex.present, width)?;
                    let dst = Operand::Reg(decode_reg_from_modrm(modrm, rex.present, width)?);
                    InstKind::Alu {
                        op: AluOp::Imul,
                        dst,
                        src,
                        width,
                    }
                }
                0xc8..=0xcf => {
                    // BSWAP r32/r64. 16-bit operand size is undefined.
                    if width == Width::W16 {
                        InstKind::Invalid
                    } else {
                        let reg_code = (opcode2 - 0xc8) | if rex.b { 8 } else { 0 };
                        InstKind::Bswap {
                            dst: Reg {
                                gpr: decode_gpr(reg_code)?,
                                width,
                                high8: false,
                            },
                            width,
                        }
                    }
                }
                _ => {
                    return Err(DecodeError {
                        message: "unsupported 0F xx opcode",
                    })
                }
            }
        }
        _ => {
            return Err(DecodeError {
                message: "unsupported opcode",
            })
        }
    };

    // A LOCK prefix is only legal on a read-modify-write with a memory destination.
    // Translating anything else while silently dropping it would be a miscompile, so
    // the interpreter gets those.
    let kind = if lock_prefix && !lockable(&kind) {
        InstKind::Invalid
    } else {
        kind
    };

    let next_rip = rip.wrapping_add(offset as u64) & next_rip_mask;
    Ok(DecodedInst {
        rip,
        len: offset as u8,
        kind,
        // Reached here means the decoder recognised the form, so it is not one of the
        // interrupt-related opcodes below; those never decode this far.
        assistable: true,
        next_rip,
    })
}


/// Whether `LOCK` is legal on this instruction, which is also exactly the set whose
/// plain read-modify-write translation is correct on a machine that runs one CPU at
/// a time. `CMP` is excluded: it has no destination to lock.
fn lockable(kind: &InstKind) -> bool {
    match kind {
        InstKind::Alu { op, dst, .. } => {
            matches!(dst, Operand::Mem(_))
                && matches!(
                    op,
                    AluOp::Add
                        | AluOp::Adc
                        | AluOp::And
                        | AluOp::Or
                        | AluOp::Sbb
                        | AluOp::Sub
                        | AluOp::Xor
                )
        }
        InstKind::Inc { dst, .. } | InstKind::Dec { dst, .. } => matches!(dst, Operand::Mem(_)),
        InstKind::Neg { dst, .. } | InstKind::Not { dst, .. } => matches!(dst, Operand::Mem(_)),
        // Both are read-modify-writes and both are overwhelmingly seen under `LOCK`.
        InstKind::Xadd { dst, .. } | InstKind::Cmpxchg { dst, .. } => {
            matches!(dst, Operand::Mem(_))
        }
        // `BT` has no write-back, so `LOCK` on it is illegal; the other three are
        // read-modify-writes like any other.
        InstKind::BitTest { op, dst, .. } | InstKind::BitTestReg { op, dst, .. } => {
            matches!(dst, Operand::Mem(_)) && !matches!(op, BitOp::Test)
        }
        _ => false,
    }
}


/// The marginal opcodes — `PREFETCH*`, `MOVNTI`, and the register-offset bit tests.
///
/// **On now, and they used to be off.** They measured as a 3% *loss* once: they let
/// ~1,700 more blocks compile while the wall barely moved, because fixing the first
/// instruction a block cannot translate often just moves the wall to a later one in
/// the same block, and the admission cost is paid either way.
///
/// That was a judgement about admission economics, and admission is not what it was —
/// a compiled module can be kept between runs (`AERO_CT_CACHE_DIR`) and the tier is
/// two and a half times faster than when this was measured. Re-measured: **105.2
/// against 116.0** in the 80M→200M window, and better in all four windows tried
/// (89.0/89.2, 101.7/102.4, 92.5/97.0). They are also 12% and 9% of the remaining
/// coverage wall by wanted executions, which is what made them worth re-testing.
///
/// The read-modify-writes next door are a different matter and stay off: 672.1.
/// `AERO_X86_NO_HINT_OPS=1` goes back.
static BITSCAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// `BSF`/`BSR`. Built, correct, and **off, because it measured as a loss.**
///
/// It was the largest translatable entry left on the coverage wall — ~17,400 wanted
/// executions — and it does what it should: the wall falls from 195,698 to 174,273 and
/// compiled coverage rises. It is still worse in every window tried: **277.7 against
/// 105.2** (80M→200M), 107.0 against 89.0, 104.1 against 102.4, and 322.3 against 92.5.
///
/// The reason is the one already written above the hint opcodes: fixing the first
/// instruction a block cannot translate usually just moves the wall to a later one in
/// the same block, so the extra blocks are admitted — and paid for — without becoming
/// much more of the running. `BSF` differs from the hint opcodes only in that it lost
/// where they won.
///
/// The translation is kept because it is correct and cheap to re-test: wasm has `ctz`
/// and `clz`, and `tests/bitscan_equivalence.rs` pins the two things that are easy to
/// get wrong — a zero source must leave the *whole* destination register alone (a
/// 32-bit write would clear the upper half; Tier-0 does not write at all), and the
/// architecturally undefined flags must be left exactly as Tier-0 leaves them.
/// `AERO_X86_BITSCAN=1` turns it on.
fn bitscan_enabled() -> bool {
    *BITSCAN.get_or_init(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("AERO_X86_BITSCAN").is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    })
}

/// Turn `BSF`/`BSR` on for a test, without going through the environment.
pub fn force_bitscan(on: bool) {
    let _ = BITSCAN.set(on);
}

fn hint_ops_disabled() -> bool {
    !*HINT_OPS.get_or_init(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("AERO_X86_NO_HINT_OPS").is_none()
        }
        #[cfg(target_arch = "wasm32")]
        {
            true
        }
    })
}

static HINT_OPS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Turn the hint opcodes on for a test, without going through the environment.
///
/// Only takes effect before the first decode; a second call with the same value is
/// harmless, which is what lets several tests in one binary each call it.
pub fn force_hint_ops(on: bool) {
    let _ = HINT_OPS.set(on);
}


/// Whether performing this instruction needs the interpreter's interrupt state
/// rather than only architectural registers. See [`DecodedInst::assistable`].
/// Measurement gates for the read-modify-writes, which are the ones whose effect on
/// block shape is hard to predict.
/// Off, and measured. Translating these instead of leaving them to the interpreter
/// makes blocks that bail: enabling `XADD` adds 4,464 blocks and *loses* 3.4M guest
/// instructions of coverage, with the average block retiring 4.33 down to 4.21.
/// Isolated, they cost 147.9, 15.4 and 1.5 host instructions per guest instruction
/// against a 147.7 baseline. `AERO_X86_RMW_OPS=1` turns all three on.
fn rmw_ops_enabled() -> bool {
    *RMW_OPS.get_or_init(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("AERO_X86_RMW_OPS").is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    })
}

static RMW_OPS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Turn them on for a test, without going through the environment. Only takes effect
/// before the first decode; repeated calls with the same value are harmless.
pub fn force_rmw_ops(on: bool) {
    let _ = RMW_OPS.set(on);
}

fn is_interrupt_related(m: crate::Mnemonic) -> bool {
    use crate::Mnemonic as M;
    matches!(
        m,
        M::Cli
            | M::Sti
            | M::Int
            | M::Int3
            | M::Into
            | M::Iret
            | M::Iretd
            | M::Iretq
            | M::Hlt
            | M::Syscall
            | M::Sysret
            | M::Sysretq
            | M::Sysenter
            | M::Sysexit
    )
}

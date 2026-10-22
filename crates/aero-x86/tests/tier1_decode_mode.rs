use aero_types::{Cond, Gpr, Width};
use aero_x86::tier1::{decode_one_mode, AluOp, InstKind, Operand, Reg, ShiftOp};

#[test]
fn decode_one_mode_32bit_dec_ecx_is_not_rex() {
    let inst = decode_one_mode(0x1000, &[0x49], 32);
    assert_eq!(inst.len, 1);
    assert_eq!(
        inst.kind,
        InstKind::Dec {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rcx,
                width: Width::W32,
                high8: false,
            }),
            width: Width::W32,
        }
    );
}

#[test]
fn decode_one_mode_32bit_modrm_disp32_is_not_rip_relative() {
    // mov eax, [0x1234]
    //
    // In 32-bit mode this is absolute disp32 addressing.
    // In 64-bit mode the same encoding would be RIP-relative.
    let inst = decode_one_mode(0x1000, &[0x8b, 0x05, 0x34, 0x12, 0x00, 0x00], 32);
    assert_eq!(inst.len, 6);
    assert_eq!(
        inst.kind,
        InstKind::Mov {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W32,
                high8: false
            }),
            src: Operand::Mem(aero_x86::tier1::Address {
                base: None,
                index: None,
                scale: 1,
                disp: 0x1234,
                rip_relative: false, segment: None }),
            width: Width::W32,
        }
    );
}

#[test]
fn decode_one_mode_32bit_d0_disp32_is_not_rip_relative() {
    // shl byte ptr [0x1234], 1
    //
    // `0xD0 /4` uses the Group2 shift decoder path in Tier1.
    // The ModRM `mod=00 rm=101` encoding is absolute disp32 addressing in 32-bit mode.
    let inst = decode_one_mode(0x1000, &[0xd0, 0x25, 0x34, 0x12, 0x00, 0x00], 32);
    assert_eq!(inst.len, 6);
    assert_eq!(
        inst.kind,
        InstKind::Shift {
            op: ShiftOp::Shl,
            dst: Operand::Mem(aero_x86::tier1::Address {
                base: None,
                index: None,
                scale: 1,
                disp: 0x1234,
                rip_relative: false, segment: None }),
            count: 1,
            width: Width::W8,
        }
    );
}

#[test]
fn decode_one_mode_32bit_jmp_rel8_wraps_eip() {
    // jmp +1 at 0xFFFF_FFFF:
    // next EIP would be 0x1_0000_0001, plus rel8(1) => 0x1_0000_0002, then wrapped to 0x0000_0002.
    let inst = decode_one_mode(0xffff_ffff, &[0xeb, 0x01], 32);
    assert_eq!(inst.len, 2);
    assert_eq!(inst.kind, InstKind::JmpRel { target: 0x2 });
}

#[test]
fn decode_one_mode_16bit_inc_ax_is_not_rex() {
    // 0x40 is INC AX in 16-bit mode; it must not be consumed as a REX prefix.
    let inst = decode_one_mode(0x1000, &[0x40], 16);
    assert_eq!(inst.len, 1);
    assert_eq!(
        inst.kind,
        InstKind::Inc {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W16,
                high8: false,
            }),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_16bit_opsize_override_inc_eax() {
    // In 16-bit mode, 0x66 selects 32-bit operand size.
    // 66 40 = inc eax
    let inst = decode_one_mode(0x1000, &[0x66, 0x40], 16);
    assert_eq!(inst.len, 2);
    assert_eq!(
        inst.kind,
        InstKind::Inc {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W32,
                high8: false,
            }),
            width: Width::W32,
        }
    );
}

#[test]
fn decode_one_mode_32bit_opsize_override_dec_cx() {
    // In 32-bit mode, 0x66 selects 16-bit operand size.
    // 66 49 = dec cx
    let inst = decode_one_mode(0x1000, &[0x66, 0x49], 32);
    assert_eq!(inst.len, 2);
    assert_eq!(
        inst.kind,
        InstKind::Dec {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rcx,
                width: Width::W16,
                high8: false,
            }),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_16bit_modrm_bx_si_disp8() {
    // mov ax, [bx+si+0x10]
    //
    // 16-bit addressing uses the ModRM rm encoding table (no SIB).
    let inst = decode_one_mode(0x1000, &[0x8b, 0x40, 0x10], 16);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Mov {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W16,
                high8: false,
            }),
            src: Operand::Mem(aero_x86::tier1::Address {
                base: Some(Gpr::Rbx),
                index: Some(Gpr::Rsi),
                scale: 1,
                disp: 0x10,
                rip_relative: false, segment: None }),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_32bit_push_ebx_uses_w32() {
    let inst = decode_one_mode(0x1000, &[0x53], 32);
    assert_eq!(inst.len, 1);
    assert_eq!(
        inst.kind,
        InstKind::Push {
            src: Operand::Reg(Reg {
                gpr: Gpr::Rbx,
                width: Width::W32,
                high8: false,
            }),
        }
    );
}

#[test]
fn decode_one_mode_32bit_pop_ebx_uses_w32() {
    let inst = decode_one_mode(0x1000, &[0x5b], 32);
    assert_eq!(inst.len, 1);
    assert_eq!(
        inst.kind,
        InstKind::Pop {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rbx,
                width: Width::W32,
                high8: false,
            }),
        }
    );
}

#[test]
fn decode_one_mode_64bit_push_pop_use_w64() {
    let push = decode_one_mode(0x1000, &[0x53], 64);
    assert_eq!(push.len, 1);
    assert_eq!(
        push.kind,
        InstKind::Push {
            src: Operand::Reg(Reg {
                gpr: Gpr::Rbx,
                width: Width::W64,
                high8: false,
            }),
        }
    );

    let pop = decode_one_mode(0x1000, &[0x5b], 64);
    assert_eq!(pop.len, 1);
    assert_eq!(
        pop.kind,
        InstKind::Pop {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rbx,
                width: Width::W64,
                high8: false,
            }),
        }
    );
}

#[test]
fn decode_one_mode_16bit_jmp_rel16_uses_imm16() {
    // jmp near -2 (rel16) in 16-bit mode:
    // next IP = 0x1003, target = 0x1001
    let inst = decode_one_mode(0x1000, &[0xe9, 0xfe, 0xff], 16);
    assert_eq!(inst.len, 3);
    assert_eq!(inst.kind, InstKind::JmpRel { target: 0x1001 });
}

#[test]
fn decode_one_mode_32bit_opsize_override_jmp_rel16() {
    // In 32-bit mode, 0x66 selects 16-bit operand size, and near JMP uses a rel16.
    // 66 E9 FC FF = jmp -4
    // next EIP = 0x1004, target = 0x1000
    let inst = decode_one_mode(0x1000, &[0x66, 0xe9, 0xfc, 0xff], 32);
    assert_eq!(inst.len, 4);
    assert_eq!(inst.kind, InstKind::JmpRel { target: 0x1000 });
}

#[test]
fn decode_one_mode_32bit_opsize_override_jmp_rel16_zero_extends_ip() {
    // In 32-bit mode, near branches with operand-size override use a 16-bit offset (IP) and
    // zero-extend the result into EIP.
    //
    // 66 E9 01 00 at 0x0001_FFFB:
    // next EIP would be 0x0001_FFFF, but IP=0xFFFF and +1 wraps to 0x0000.
    let inst = decode_one_mode(0x0001_fffb, &[0x66, 0xe9, 0x01, 0x00], 32);
    assert_eq!(inst.len, 4);
    assert_eq!(inst.kind, InstKind::JmpRel { target: 0x0 });
}

#[test]
fn decode_one_mode_32bit_opsize_override_jmp_rel8_zero_extends_ip() {
    // Like `decode_one_mode_32bit_opsize_override_jmp_rel16_zero_extends_ip`, but for JMP rel8.
    //
    // 66 EB 00 at 0x0001_FFFD:
    // next EIP would be 0x0002_0000, but IP wraps to 0x0000.
    let inst = decode_one_mode(0x0001_fffd, &[0x66, 0xeb, 0x00], 32);
    assert_eq!(inst.len, 3);
    assert_eq!(inst.kind, InstKind::JmpRel { target: 0x0 });
}

#[test]
fn decode_one_mode_16bit_call_rel16_uses_imm16() {
    // call near +0 (rel16) in 16-bit mode:
    // next IP = 0x2003, target = 0x2003
    let inst = decode_one_mode(0x2000, &[0xe8, 0x00, 0x00], 16);
    assert_eq!(inst.len, 3);
    assert_eq!(inst.kind, InstKind::CallRel { target: 0x2003 });
}

#[test]
fn decode_one_mode_16bit_jcc_rel16_uses_imm16() {
    // jnz near -2 (rel16): 0F 85 FE FF
    // next IP = 0x3004, target = 0x3002
    let inst = decode_one_mode(0x3000, &[0x0f, 0x85, 0xfe, 0xff], 16);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::JccRel {
            cond: Cond::Ne,
            target: 0x3002
        }
    );
}

#[test]
fn decode_one_mode_32bit_opsize_override_jcc_rel16() {
    // In 32-bit mode, 0x66 selects 16-bit operand size, and near Jcc uses a rel16.
    // 66 0F 84 02 00 = jz +2
    // next EIP = 0x4005, target = 0x4007
    let inst = decode_one_mode(0x4000, &[0x66, 0x0f, 0x84, 0x02, 0x00], 32);
    assert_eq!(inst.len, 5);
    assert_eq!(
        inst.kind,
        InstKind::JccRel {
            cond: Cond::E,
            target: 0x4007
        }
    );
}

#[test]
fn decode_one_mode_32bit_opsize_override_jcc_rel16_zero_extends_fallthrough() {
    // Like `decode_one_mode_32bit_opsize_override_jmp_rel16_zero_extends_ip`, but for near Jcc.
    //
    // 66 0F 84 01 00 at 0x0001_FFFA:
    // next EIP would be 0x0001_FFFF, but IP=0xFFFF and +1 wraps to 0x0000.
    let inst = decode_one_mode(0x0001_fffa, &[0x66, 0x0f, 0x84, 0x01, 0x00], 32);
    assert_eq!(inst.len, 5);
    assert_eq!(
        inst.kind,
        InstKind::JccRel {
            cond: Cond::E,
            target: 0x0
        }
    );
    assert_eq!(inst.next_rip(), 0xffff);
}

#[test]
fn decode_one_mode_32bit_opsize_override_jcc_rel8_zero_extends_ip() {
    // Like `decode_one_mode_32bit_opsize_override_jcc_rel16_zero_extends_fallthrough`, but for
    // short Jcc (rel8).
    //
    // 66 70 00 at 0x0001_FFFD => JO +0
    // next EIP would be 0x0002_0000, but IP wraps to 0x0000.
    let inst = decode_one_mode(0x0001_fffd, &[0x66, 0x70, 0x00], 32);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::JccRel {
            cond: Cond::O,
            target: 0x0
        }
    );
    assert_eq!(inst.next_rip(), 0x0);
}

#[test]
fn decode_one_mode_16bit_opsize_override_near_jmp_bails() {
    // In 16-bit mode, 0x66 selects 32-bit operand size, and near JMP uses a rel32 offset.
    //
    // Tier-1 currently assumes a fixed 16-bit instruction pointer in `bitness=16`, so it bails
    // out instead of mis-compiling.
    let inst = decode_one_mode(0x1000, &[0x66, 0xe9, 0x01, 0x00, 0x00, 0x00], 16);
    assert_eq!(inst.len, 6);
    assert_eq!(inst.kind, InstKind::Invalid);
}

#[test]
fn decode_one_mode_16bit_add_ax_imm16_is_not_imm32() {
    // add ax, 0x1234
    let inst = decode_one_mode(0x1000, &[0x05, 0x34, 0x12], 16);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Add,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W16,
                high8: false,
            }),
            src: Operand::Imm(0x1234),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_32bit_opsize_override_add_ax_imm16() {
    // 66 05 34 12 = add ax, 0x1234
    let inst = decode_one_mode(0x1000, &[0x66, 0x05, 0x34, 0x12], 32);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Add,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W16,
                high8: false,
            }),
            src: Operand::Imm(0x1234),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_16bit_81_group_reads_imm16() {
    // 81 /0 = add r/m16, imm16
    // add ax, 0x1234
    let inst = decode_one_mode(0x1000, &[0x81, 0xc0, 0x34, 0x12], 16);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Add,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W16,
                high8: false,
            }),
            src: Operand::Imm(0x1234),
            width: Width::W16,
        }
    );
}

#[test]
fn decode_one_mode_16bit_push_imm16_reads_imm16() {
    // push 0x1234
    let inst = decode_one_mode(0x1000, &[0x68, 0x34, 0x12], 16);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Push {
            src: Operand::Imm(0x1234),
        }
    );
}

#[test]
fn decode_one_mode_32bit_opsize_override_push_imm16() {
    // In 32-bit mode, 0x66 selects 16-bit operand size, and PUSH imm uses an imm16.
    let inst = decode_one_mode(0x1000, &[0x66, 0x68, 0x34, 0x12], 32);
    assert_eq!(inst.len, 4);
    // Tier-1 translation does not currently model operand-size overridden stack ops, so the
    // minimal decoder treats them as unsupported to avoid miscompilation.
    assert_eq!(inst.kind, InstKind::Invalid);
}

#[test]
fn decode_one_mode_bails_on_address_size_override_prefix() {
    // 67 8B 07 would be `mov eax, [bx]`-style (16-bit) addressing in 32-bit mode, which Tier-1
    // does not currently model. Ensure we bail out instead of mis-decoding the address.
    let inst = decode_one_mode(0x1000, &[0x67, 0x8b, 0x07], 32);
    assert_eq!(inst.len, 2);
    assert_eq!(inst.kind, InstKind::Invalid);
}

#[test]
fn decode_one_mode_multi_byte_nop_0f1f_parses_modrm_len() {
    // nop dword ptr [rax+0x1234]
    let inst = decode_one_mode(0x1000, &[0x0f, 0x1f, 0x80, 0x34, 0x12, 0x00, 0x00], 64);
    assert_eq!(inst.len, 7);
    assert_eq!(inst.kind, InstKind::Nop);
}

#[test]
fn decode_one_mode_multi_byte_nop_0f1f_32bit_disp8_len() {
    // nop dword ptr [eax+0x10]
    let inst = decode_one_mode(0x1000, &[0x0f, 0x1f, 0x40, 0x10], 32);
    assert_eq!(inst.len, 4);
    assert_eq!(inst.kind, InstKind::Nop);
}

#[test]
fn decode_one_mode_multi_byte_nop_0f1f_16bit_disp8_len() {
    // nop word ptr [bx+si+0x10]
    let inst = decode_one_mode(0x1000, &[0x0f, 0x1f, 0x40, 0x10], 16);
    assert_eq!(inst.len, 4);
    assert_eq!(inst.kind, InstKind::Nop);
}

#[test]
fn decode_long_mode_adc_rdx_imm8_is_adc_not_invalid() {
    // cryptsp bignum loops use `adc rdx, 0` / `adc r10, 0` (REX.W 83 /2 ib).
    // Those used to be an unsupported 0x83 group, so the Tier-1 JIT split the
    // 64-limb add/sub carry chain and produced a wrong 4096-bit RSA decrypt.
    let inst = decode_one_mode(0, &[0x48, 0x83, 0xd2, 0x00], 64);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Adc,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rdx,
                width: Width::W64,
                high8: false,
            }),
            src: Operand::Imm(0),
            width: Width::W64,
        }
    );
}

#[test]
fn decode_long_mode_cmp_byte_ptr_rdx_imm8_is_cmp() {
    // WinSetup.dll apply scan (vds-copy-93b RIP 0x7fefc0c127c):
    // `cmp byte [rdx], 0xe8` is 80 /7, not 81/83.
    let inst = decode_one_mode(0, &[0x80, 0x3a, 0xe8], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Cmp {
            lhs: Operand::Mem(aero_x86::tier1::Address {
                base: Some(Gpr::Rdx),
                index: None,
                scale: 1,
                disp: 0,
                rip_relative: false, segment: None }),
            rhs: Operand::Imm(0xe8),
            width: Width::W8,
        }
    );
}

#[test]
fn decode_long_mode_rex_add_dil_imm8_is_alu() {
    // WinSetup bitstream refill: `add dil, 0x10` (40 80 C7 10).
    let inst = decode_one_mode(0, &[0x40, 0x80, 0xc7, 0x10], 64);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Add,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rdi,
                width: Width::W8,
                high8: false,
            }),
            src: Operand::Imm(0x10),
            width: Width::W8,
        }
    );
}

#[test]
fn decode_long_mode_imul_eax_eax_imm32_is_imul3() {
    // ntoskrnl CIM name-hash: `imul eax, eax, 0xffff9e5f` (69 C0 5F 9E FF FF).
    let inst = decode_one_mode(0, &[0x69, 0xc0, 0x5f, 0x9e, 0xff, 0xff], 64);
    assert_eq!(inst.len, 6);
    assert_eq!(
        inst.kind,
        InstKind::Imul3 {
            dst: Reg {
                gpr: Gpr::Rax,
                width: Width::W32,
                high8: false,
            },
            src: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W32,
                high8: false,
            }),
            imm: 0xffff_9e5f,
            width: Width::W32,
        }
    );
}

#[test]
fn decode_long_mode_imul_rdx_r9_is_two_operand_imul() {
    // msvcrt memcpy splat: `imul rdx, r9` (REX.W 0F AF D1).
    let inst = decode_one_mode(0, &[0x49, 0x0f, 0xaf, 0xd1], 64);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Imul,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rdx,
                width: Width::W64,
                high8: false,
            }),
            src: Operand::Reg(Reg {
                gpr: Gpr::R9,
                width: Width::W64,
                high8: false,
            }),
            width: Width::W64,
        }
    );
}

#[test]
fn decode_long_mode_ror_rax_imm8_is_ror_not_invalid() {
    // bcryptprimitives SHA-512 Sigma0/1 uses `ror r64, imm8` with counts
    // 14/18/28/34/39/41 (and 19/61 in the message schedule).
    let inst = decode_one_mode(0, &[0x48, 0xc1, 0xc8, 0x22], 64);
    assert_eq!(inst.len, 4);
    assert_eq!(
        inst.kind,
        InstKind::Shift {
            op: ShiftOp::Ror,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W64,
                high8: false,
            }),
            count: 0x22,
            width: Width::W64,
        }
    );
}

#[test]
fn decode_long_mode_ror_rax_1_is_ror() {
    let inst = decode_one_mode(0, &[0x48, 0xd1, 0xc8], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Shift {
            op: ShiftOp::Ror,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rax,
                width: Width::W64,
                high8: false,
            }),
            count: 1,
            width: Width::W64,
        }
    );
}

#[test]
fn decode_long_mode_sub_dil_cl_is_alu() {
    // WinSetup LZX bitstream: `sub dil, cl` (40 2A F9).
    let inst = decode_one_mode(0, &[0x40, 0x2a, 0xf9], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Alu {
            op: AluOp::Sub,
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rdi,
                width: Width::W8,
                high8: false,
            }),
            src: Operand::Reg(Reg {
                gpr: Gpr::Rcx,
                width: Width::W8,
                high8: false,
            }),
            width: Width::W8,
        }
    );
}

#[test]
fn decode_long_mode_test_dil_dil_is_test() {
    // Same refill: `test dil, dil` (40 84 FF).
    let inst = decode_one_mode(0, &[0x40, 0x84, 0xff], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Test {
            lhs: Operand::Reg(Reg {
                gpr: Gpr::Rdi,
                width: Width::W8,
                high8: false,
            }),
            rhs: Operand::Reg(Reg {
                gpr: Gpr::Rdi,
                width: Width::W8,
                high8: false,
            }),
            width: Width::W8,
        }
    );
}

#[test]
fn decode_long_mode_shl_r11d_cl_is_shift_cl() {
    // Same refill: `shl r11d, cl` (41 D3 E3).
    let inst = decode_one_mode(0, &[0x41, 0xd3, 0xe3], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::ShiftCl {
            op: ShiftOp::Shl,
            dst: Operand::Reg(Reg {
                gpr: Gpr::R11,
                width: Width::W32,
                high8: false,
            }),
            width: Width::W32,
        }
    );
}

#[test]
fn decode_long_mode_neg_ecx_is_neg() {
    // WinSetup refill: `neg ecx` (F7 D9).
    let inst = decode_one_mode(0, &[0xf7, 0xd9], 64);
    assert_eq!(inst.len, 2);
    assert_eq!(
        inst.kind,
        InstKind::Neg {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rcx,
                width: Width::W32,
                high8: false,
            }),
            width: Width::W32,
        }
    );
}

#[test]
fn decode_long_mode_not_rcx_is_not() {
    let inst = decode_one_mode(0, &[0x48, 0xf7, 0xd1], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Not {
            dst: Operand::Reg(Reg {
                gpr: Gpr::Rcx,
                width: Width::W64,
                high8: false,
            }),
            width: Width::W64,
        }
    );
}

#[test]
fn decode_long_mode_bswap_r8_is_bswap() {
    let inst = decode_one_mode(0, &[0x49, 0x0f, 0xc8], 64);
    assert_eq!(inst.len, 3);
    assert_eq!(
        inst.kind,
        InstKind::Bswap {
            dst: Reg {
                gpr: Gpr::R8,
                width: Width::W64,
                high8: false,
            },
            width: Width::W64,
        }
    );
}

/// A `REX` prefix only applies when it is the last prefix before the opcode.
///
/// `48 F2 B8 imm32` is `MOV EAX, imm32` with the `REX.W` discarded by the `F2` that
/// follows it — 7 bytes — not `MOV RAX, imm64` at 11. Decoding it as the latter runs
/// four bytes of the *next* instruction as part of this one, so the whole block after
/// it is garbage. Real compilers do not emit this ordering, which is why it survived;
/// a proptest against iced generated it.
///
/// ```text
/// cargo test --release -p aero-x86 --test tier1_decode_mode
/// ```
#[test]
fn a_rex_followed_by_another_prefix_does_not_apply() {
    // 48 (REX.W) F2 (REPNE) B8 (MOV eAX, imm) 00000000
    let bytes = [0x48u8, 0xf2, 0xb8, 0x00, 0x00, 0x00, 0x00];
    let inst = decode_one_mode(0, &bytes, 64);
    assert_eq!(
        inst.len, 7,
        "REX.W must be discarded by the following prefix, making this a 32-bit MOV"
    );

    // And with the prefixes the other way round it *is* the 64-bit form, so the fix
    // is about ordering rather than about ignoring REX whenever F2 appears.
    let bytes = [
        0xf2u8, 0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let inst = decode_one_mode(0, &bytes, 64);
    assert_eq!(
        inst.len, 11,
        "REX.W is the last prefix here, so this is MOV RAX, imm64"
    );
}

//! cryptsp Win7 x64 bignum inner loops vs a host reference.
//!
//! lsass dies because `cryptsp` rejects `rsaenh.dll`'s Authenticode PKCS#7:
//! the 2048-bit RSA public decrypt matches `pow(sig, e, n)` and PKCS#1
//! verifies, but the following 4096-bit decrypt returns a random-looking
//! block (`NTE_BAD_SIGNATURE`). These are the exact leaf routines that
//! 4096-bit path uses (64-bit limbs in `bfe0`/`bf90`/`c0a0`/`c030`,
//! 32-bit limbs in `bdf0`).

use aero_cpu_core::interp::tier0::exec::{run_batch, BatchExit};
use aero_cpu_core::mem::{CpuBus, FlatTestBus};
use aero_cpu_core::state::{CpuMode, CpuState, FLAG_CF};
use aero_x86::Register;

const CODE: u64 = 0x1000;
const STACK: u64 = 0x8000;
const HLT: u64 = 0x1F00;
const DEST: u64 = 0x4000;
const SRC: u64 = 0x5000;
const SRC_B: u64 = 0x6000;

/// cryptsp `0xbfe0`: dest[0..n) += src[0..n) * m, return carry in RAX.
const BFE0: &[u8] = &[
    0x4c, 0x8b, 0xd2, 0x41, 0xc1, 0xe1, 0x03, 0x74, 0x3a, 0x49, 0x03, 0xc9, 0x4d, 0x03, 0xc1, 0x49,
    0xf7, 0xd9, 0x4d, 0x2b, 0xdb, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x4b, 0x8b, 0x04, 0x01, 0x49, 0xf7, 0xe2, 0x49, 0x03, 0xc3, 0x48, 0x83, 0xd2, 0x00, 0x49, 0x01,
    0x04, 0x09, 0x48, 0x83, 0xd2, 0x00, 0x49, 0x83, 0xc1, 0x08, 0x4c, 0x8b, 0xda, 0x75, 0xe1, 0x48,
    0x8b, 0xc2, 0xc3,
];

/// cryptsp `0xbf90`: add the squaring diagonal `src[i]^2` into dest[2i..].
const BF90: &[u8] = &[
    0x41, 0xc1, 0xe0, 0x03, 0x74, 0x3e, 0x4a, 0x8d, 0x0c, 0x41, 0x4d, 0x8d, 0x0c, 0x10, 0x49, 0xf7,
    0xd8, 0x4d, 0x2b, 0xdb, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x4b, 0x8b, 0x04, 0x08, 0x48, 0xf7, 0xe0, 0x49, 0x03, 0xc3, 0x48, 0x83, 0xd2, 0x00, 0x4d, 0x33,
    0xdb, 0x4a, 0x01, 0x04, 0x41, 0x4a, 0x11, 0x54, 0x41, 0x08, 0x49, 0x83, 0xd3, 0x00, 0x49, 0x83,
    0xc0, 0x08, 0x75, 0xdc, 0xc3,
];

/// cryptsp `0xc0a0`: dest[0..n) = a[0..n) + b[0..n), return carry in RAX.
const C0A0: &[u8] = &[
    0x45, 0x0b, 0xc9, 0x74, 0x41, 0x4a, 0x8d, 0x0c, 0xc9, 0x4a, 0x8d, 0x14, 0xca, 0x4f, 0x8d, 0x04,
    0xc8, 0x49, 0xf7, 0xd9, 0x4d, 0x33, 0xd2, 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x4a, 0x8b, 0x04, 0xca, 0x49, 0x03, 0xc2, 0x41, 0xba, 0x00, 0x00, 0x00, 0x00, 0x49, 0x83, 0xd2,
    0x00, 0x4b, 0x03, 0x04, 0xc8, 0x49, 0x83, 0xd2, 0x00, 0x4a, 0x89, 0x04, 0xc9, 0x49, 0xff, 0xc1,
    0x75, 0xde, 0x49, 0x8b, 0xc2, 0xc3,
];

/// cryptsp `0xc030`: dest[0..n) -= src[0..n) * m, return borrow-ish in RAX.
const C030: &[u8] = &[
    0x48, 0x83, 0xfa, 0x01, 0x74, 0x4d, 0x72, 0x50, 0x45, 0x0b, 0xc9, 0x4c, 0x8b, 0xd2, 0x49, 0x8b,
    0x00, 0x4a, 0x8d, 0x0c, 0xc9, 0x4f, 0x8d, 0x04, 0xc8, 0x74, 0x3d, 0x49, 0xf7, 0xd9, 0x49, 0xf7,
    0xe2, 0x4a, 0x29, 0x04, 0xc9, 0x48, 0x83, 0xd2, 0x00, 0x49, 0xff, 0xc1, 0x74, 0x21, 0x66, 0x90,
    0x4b, 0x8b, 0x04, 0xc8, 0x4c, 0x8b, 0xda, 0x49, 0xf7, 0xe2, 0x4a, 0x29, 0x04, 0xc9, 0x48, 0x83,
    0xd2, 0x00, 0x4e, 0x29, 0x1c, 0xc9, 0x48, 0x83, 0xd2, 0x00, 0x49, 0xff, 0xc1, 0x75, 0xe1, 0x48,
    0x8b, 0xc2, 0xc3,
];

/// cryptsp `0xbdf0` (edx != 1 path): dest[i] = dest[i] * m + carry, 32-bit limbs.
const BDF0: &[u8] = &[
    0x40, 0x53, 0x48, 0x83, 0xec, 0x20, 0x33, 0xdb, 0x4d, 0x8b, 0xd8, 0x4c, 0x8b, 0xd1, 0x83, 0xfa,
    0x01, 0x75, 0x17, 0x45, 0x8b, 0xc1, 0x49, 0x8b, 0xd3, 0x49, 0xc1, 0xe0, 0x02, 0xe8, 0xb2, 0x0f,
    0x00, 0x00, 0x8b, 0xc3, 0x48, 0x83, 0xc4, 0x20, 0x5b, 0xc3, 0x45, 0x85, 0xc9, 0x74, 0x33, 0x44,
    0x8b, 0xc2, 0x4c, 0x2b, 0xd9, 0x41, 0x8b, 0xd1, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x43, 0x8b, 0x0c, 0x13, 0x8b, 0xc3, 0x49, 0x83, 0xc2, 0x04, 0x49, 0x0f, 0xaf, 0xc8, 0x48, 0x03,
    0xc8, 0x48, 0x8b, 0xd9, 0x41, 0x89, 0x4a, 0xfc, 0x48, 0xc1, 0xeb, 0x20, 0x48, 0x83, 0xea, 0x01,
    0x75, 0xde, 0x8b, 0xc3, 0x48, 0x83, 0xc4, 0x20, 0x5b, 0xc3,
];

fn new_long() -> (CpuState, FlatTestBus) {
    let mut bus = FlatTestBus::new(0x1_0000);
    bus.load(HLT, &[0xF4]);
    let mut state = CpuState::new(CpuMode::Long);
    state.set_rip(CODE);
    state.write_reg(Register::RSP, STACK);
    bus.write_u64(STACK - 8, HLT).unwrap();
    state.write_reg(Register::RSP, STACK - 8);
    (state, bus)
}

fn run_fn(state: &mut CpuState, bus: &mut FlatTestBus, code: &[u8]) {
    bus.load(CODE, code);
    state.halted = false;
    state.set_rip(CODE);
    bus.write_u64(STACK - 8, HLT).unwrap();
    state.write_reg(Register::RSP, STACK - 8);
    let mut steps = 0u64;
    while steps < 2_000_000 {
        let res = run_batch(state, bus, 256);
        steps += res.executed;
        match res.exit {
            BatchExit::Halted => return,
            BatchExit::Completed | BatchExit::Branch => continue,
            other => panic!("unexpected exit {other:?} rip={:#x}", state.rip()),
        }
    }
    panic!("function did not halt");
}

fn write_limbs(bus: &mut FlatTestBus, addr: u64, limbs: &[u64]) {
    for (i, &limb) in limbs.iter().enumerate() {
        bus.write_u64(addr + (i as u64) * 8, limb).unwrap();
    }
}

fn read_limbs(bus: &mut FlatTestBus, addr: u64, n: usize) -> Vec<u64> {
    (0..n)
        .map(|i| bus.read_u64(addr + (i as u64) * 8).unwrap())
        .collect()
}

fn ref_muladd(dest: &mut [u64], src: &[u64], m: u64) -> u64 {
    let mut carry = 0u128;
    for i in 0..src.len() {
        let t = dest[i] as u128 + src[i] as u128 * m as u128 + carry;
        dest[i] = t as u64;
        carry = t >> 64;
    }
    carry as u64
}

fn ref_add(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0u64; a.len()];
    let mut carry = 0u128;
    for i in 0..a.len() {
        let t = a[i] as u128 + b[i] as u128 + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    (out, carry as u64)
}

fn ref_mulsub_guest(dest: &mut [u64], src: &[u64], m: u64) -> u64 {
    // Emulate the exact 0xc030 carry protocol with wrapping 64-bit math.
    let mut rdx = 0u64;
    // first limb is special: mul r10; sub [dest], rax; adc rdx, 0
    if src.is_empty() {
        return 0;
    }
    let mut prod = (src[0] as u128) * (m as u128);
    let mut rax = prod as u64;
    rdx = (prod >> 64) as u64;
    let (d0, brw) = dest[0].overflowing_sub(rax);
    dest[0] = d0;
    rdx = rdx.wrapping_add(brw as u64);
    for i in 1..src.len() {
        let r11 = rdx;
        prod = (src[i] as u128) * (m as u128);
        rax = prod as u64;
        rdx = (prod >> 64) as u64;
        let (d1, b1) = dest[i].overflowing_sub(rax);
        dest[i] = d1;
        rdx = rdx.wrapping_add(b1 as u64);
        let (d2, b2) = dest[i].overflowing_sub(r11);
        dest[i] = d2;
        rdx = rdx.wrapping_add(b2 as u64);
    }
    rdx
}

fn ref_square_diag(dest: &mut [u64], src: &[u64]) {
    // 0xbf90: for i, dest[2i] += lo(src[i]^2)+cin; dest[2i+1] += hi + CF; cin = new CF
    let mut cin = 0u64;
    for (i, &limb) in src.iter().enumerate() {
        let prod = (limb as u128) * (limb as u128);
        let mut rax = (prod as u64) as u128 + cin as u128;
        let mut rdx = (prod >> 64) as u128 + ((rax >> 64) as u128);
        rax &= u64::MAX as u128;
        let t0 = dest[2 * i] as u128 + rax;
        dest[2 * i] = t0 as u64;
        let t1 = dest[2 * i + 1] as u128 + rdx + (t0 >> 64);
        dest[2 * i + 1] = t1 as u64;
        cin = (t1 >> 64) as u64;
    }
}

fn schoolbook(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len();
    let mut out = vec![0u64; 2 * n];
    for i in 0..n {
        let carry = ref_muladd(&mut out[i..i + n], b, a[i]);
        out[i + n] = carry;
    }
    out
}

fn pattern_limbs(n: usize, salt: u64) -> Vec<u64> {
    (0..n)
        .map(|i| {
            0x9E37_79B9_7F4A_7C15u64
                .wrapping_mul(i as u64 + 1)
                .wrapping_add(salt)
                ^ (u64::MAX.wrapping_shr((i % 17) as u32))
        })
        .collect()
}

#[test]
fn adc_mem64_src_all_ones_with_carry_in() {
    // Memory-dest ADC used the RMW path `old + (src + cin)` with wrapping
    // add of src+cin. When src == u64::MAX and CF=1 that sum wraps; flags
    // were computed separately. This is the exact edge the 4096-bit mul
    // high-half `adc [mem], rdx` can hit.
    let (mut state, mut bus) = new_long();
    bus.write_u64(DEST, u64::MAX).unwrap();
    state.write_reg(Register::RAX, DEST);
    state.write_reg(Register::RDX, u64::MAX);
    state.set_flag(FLAG_CF, true);
    // adc qword ptr [rax], rdx
    run_fn(&mut state, &mut bus, &[0x48, 0x11, 0x10, 0xC3]);
    // MAX + MAX + 1 = 2^65 - 1 → store MAX, CF=1
    assert_eq!(bus.read_u64(DEST).unwrap(), u64::MAX);
    assert!(state.get_flag(FLAG_CF), "ADC MAX+MAX+1 must set CF");
}

#[test]
fn cryptsp_bfe0_matches_reference_for_32_and_64_limbs() {
    for &n in &[32usize, 64] {
        for &salt in &[0u64, 0xFFFF_FFFF_FFFF_FFFF, 0x0123_4567_89AB_CDEF] {
            let src = if salt == u64::MAX {
                vec![u64::MAX; n]
            } else {
                pattern_limbs(n, salt)
            };
            let mut dest = if salt == 0 {
                vec![0u64; n]
            } else {
                pattern_limbs(n, salt.wrapping_mul(3))
            };
            let m = if salt == u64::MAX {
                u64::MAX
            } else {
                0xFFFF_FFFD_0000_0001
            };

            let mut expect = dest.clone();
            let expect_carry = ref_muladd(&mut expect, &src, m);

            let (mut state, mut bus) = new_long();
            write_limbs(&mut bus, SRC, &src);
            write_limbs(&mut bus, DEST, &dest);
            state.write_reg(Register::RCX, DEST);
            state.write_reg(Register::RDX, m);
            state.write_reg(Register::R8, SRC);
            state.write_reg(Register::R9, n as u64);
            run_fn(&mut state, &mut bus, BFE0);

            let got = read_limbs(&mut bus, DEST, n);
            let got_carry = state.read_reg(Register::RAX);
            assert_eq!(got, expect, "bfe0 dest n={n} salt={salt:#x}");
            assert_eq!(got_carry, expect_carry, "bfe0 carry n={n} salt={salt:#x}");
            dest.clone_from(&got); // keep dest used
            let _ = dest;
        }
    }
}

#[test]
fn cryptsp_c0a0_matches_reference_for_64_limbs() {
    let n = 64;
    let a = vec![u64::MAX; n];
    let b = pattern_limbs(n, 7);
    let (expect, expect_carry) = ref_add(&a, &b);

    let (mut state, mut bus) = new_long();
    write_limbs(&mut bus, SRC, &a);
    write_limbs(&mut bus, SRC_B, &b);
    write_limbs(&mut bus, DEST, &vec![0xA5A5_A5A5_A5A5_A5A5; n]);
    state.write_reg(Register::RCX, DEST);
    state.write_reg(Register::RDX, SRC);
    state.write_reg(Register::R8, SRC_B);
    state.write_reg(Register::R9, n as u64);
    run_fn(&mut state, &mut bus, C0A0);

    assert_eq!(read_limbs(&mut bus, DEST, n), expect);
    assert_eq!(state.read_reg(Register::RAX), expect_carry);
}

#[test]
fn cryptsp_c030_matches_guest_protocol_for_64_limbs() {
    let n = 64;
    let src = vec![u64::MAX; n];
    let mut dest = pattern_limbs(n, 99);
    let m = u64::MAX;
    let mut expect = dest.clone();
    let expect_rdx = ref_mulsub_guest(&mut expect, &src, m);

    let (mut state, mut bus) = new_long();
    write_limbs(&mut bus, SRC, &src);
    write_limbs(&mut bus, DEST, &dest);
    state.write_reg(Register::RCX, DEST);
    state.write_reg(Register::RDX, m);
    state.write_reg(Register::R8, SRC);
    state.write_reg(Register::R9, n as u64);
    run_fn(&mut state, &mut bus, C030);

    assert_eq!(read_limbs(&mut bus, DEST, n), expect, "c030 dest");
    assert_eq!(state.read_reg(Register::RAX), expect_rdx, "c030 rdx");
}

#[test]
fn cryptsp_bf90_square_diag_64_limbs() {
    let n = 64;
    let src = pattern_limbs(n, 0xC0FFEE);
    let mut expect = vec![0u64; 2 * n];
    ref_square_diag(&mut expect, &src);

    let (mut state, mut bus) = new_long();
    write_limbs(&mut bus, SRC, &src);
    write_limbs(&mut bus, DEST, &vec![0u64; 2 * n]);
    state.write_reg(Register::RCX, DEST);
    state.write_reg(Register::RDX, SRC);
    state.write_reg(Register::R8, n as u64);
    run_fn(&mut state, &mut bus, BF90);

    assert_eq!(read_limbs(&mut bus, DEST, 2 * n), expect);
}

#[test]
fn cryptsp_bfe0_schoolbook_64x64_matches_python_style_product() {
    // Full 64-limb × 64-limb product via the same muladd leaf 4096-bit RSA uses.
    let n = 64;
    let a = pattern_limbs(n, 1);
    let b = pattern_limbs(n, 2);
    let expect = schoolbook(&a, &b);

    let (mut state, mut bus) = new_long();
    write_limbs(&mut bus, SRC, &a);
    write_limbs(&mut bus, SRC_B, &b);
    write_limbs(&mut bus, DEST, &vec![0u64; 2 * n]);

    for i in 0..n {
        state.write_reg(Register::RCX, DEST + (i as u64) * 8);
        state.write_reg(Register::RDX, a[i]);
        state.write_reg(Register::R8, SRC_B);
        state.write_reg(Register::R9, n as u64);
        run_fn(&mut state, &mut bus, BFE0);
        bus.write_u64(DEST + ((i + n) as u64) * 8, state.read_reg(Register::RAX))
            .unwrap();
    }

    assert_eq!(read_limbs(&mut bus, DEST, 2 * n), expect);
}

#[test]
fn cryptsp_bdf0_32bit_limbs_128_matches_reference() {
    // Montgomery setup for 4096-bit uses 128 thirty-two-bit limbs.
    let n = 128usize;
    let src: Vec<u32> = (0..n)
        .map(|i| {
            0xFFFF_FFFFu32
                .wrapping_sub(i as u32)
                .wrapping_mul(0x9E37_79B9)
        })
        .collect();
    let m = 0xFFFF_FFFDu32;
    let mut expect = src.clone();
    let mut carry = 0u64;
    for limb in expect.iter_mut() {
        let t = *limb as u64 * m as u64 + carry;
        *limb = t as u32;
        carry = t >> 32;
    }

    let (mut state, mut bus) = new_long();
    for (i, &v) in src.iter().enumerate() {
        bus.write_u32(DEST + (i as u64) * 4, v).unwrap();
    }
    // in-place: rcx = dest, edx = m, r8 unused as src (uses dest), r9d = n
    // bdf0: dest[i] = dest[i] * m + carry
    state.write_reg(Register::RCX, DEST);
    state.write_reg(Register::RDX, m as u64);
    state.write_reg(Register::R8, DEST);
    state.write_reg(Register::R9, n as u64);
    run_fn(&mut state, &mut bus, BDF0);

    for (i, &v) in expect.iter().enumerate() {
        assert_eq!(
            bus.read_u32(DEST + (i as u64) * 4).unwrap(),
            v,
            "bdf0 limb {i}"
        );
    }
    assert_eq!(state.read_reg(Register::EAX), carry as u32 as u64);
}

#[test]
fn cryptsp_bfe0_live_4096_modulus_times_max() {
    // Live-dumped 4096-bit modulus from plus2b-1200m (LE bytes).
    let bytes = hex_to_bytes(concat!(
        "35c1cebf00fd86aa6c6bb6e54016d47c2604bb8fc8385b7a1e60c16558e19c74",
        "40dec4369b116eb0d141a823aa168f99301b6721ddca7b2d40ab2eea32077062",
        "59f90033d1fc5b32fa734bfd891415ef94a524d515eb11cb97b52e1f7d18410e",
        "602d06bb318955a5068723a545b2d9393381df9004efdb4028c295acfdb7e2e6",
        "8853cede5b97119a9e86ecfc2ac90451c4db45ec2d8ead887ecbd7a488451d0f",
        "49fc20c70471773f03cde4b11f2ae36a287f10ef9542c091b0160a96714730b9",
        "61a3f95ac0c1fcd0a04236c7e386b06f4bd56f92744468ce902297117438d69e",
        "4d04247c450ffd57355942d864533fefbf58e433075bdb3d4d044ea5c0364059",
        "3263beb5d55379ca344e0835c89931953312eee4ace7c80625679606ff264dc2",
        "dea39b9795a26ba18bdb3698737d52cbc9339ff86eb5f36de179de8236b5003c",
        "d292e57ec3d57ee1e4a6d6c57c149179ac6011e1b705e76c2f9bbfb747d870da",
        "f590af6d7d0470147d796a8f0c48f55b77b49194b71aa8db75d595db9994136f",
        "5ce44fadb8ebe64f32480be1fc4bcba8d2d7326d2968fc5803ca079fd8085988",
        "aea39207b779e1195875fcfe14bc9009e15e1d91223c6434f7912277e3059de5",
        "932b0b58e9ba80f7afdfe6363404fa580e947625c8e0c27c76182971949e5a68",
        "91fa60c3cededa899c8907b7cd84753c0835d020902c0ca9a75ad46780fa5df3",
    ));
    assert_eq!(bytes.len(), 512);
    let src: Vec<u64> = bytes
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let dest = vec![0u64; 64];
    let m = u64::MAX;
    let mut expect = dest.clone();
    let expect_carry = ref_muladd(&mut expect, &src, m);

    let (mut state, mut bus) = new_long();
    write_limbs(&mut bus, SRC, &src);
    write_limbs(&mut bus, DEST, &dest);
    state.write_reg(Register::RCX, DEST);
    state.write_reg(Register::RDX, m);
    state.write_reg(Register::R8, SRC);
    state.write_reg(Register::R9, 64);
    run_fn(&mut state, &mut bus, BFE0);
    assert_eq!(read_limbs(&mut bus, DEST, 64), expect);
    assert_eq!(state.read_reg(Register::RAX), expect_carry);
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

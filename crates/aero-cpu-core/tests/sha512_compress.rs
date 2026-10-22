//! Win7 `bcryptprimitives` SHA-512 compress (`RVA 0xb000`) vs FIPS 180-4.
//!
//! After the JIT ADC/SBB fix, lsass stays alive but a +2B `--jit` slice from
//! `plus2b-1600m` spends every progress sample in this function (or the MSHA
//! name match just before `GetHashInterface`). The guest re-inits the SHA-512
//! IV and rehashes the same 0x80/0xc0-byte buffers — a KAT/retry signature.
//! This test runs the exact boot.wim leaf against a host reference so a
//! ROR/BSWAP/ADD bug cannot hide behind "the guest is just hashing".

use aero_cpu_core::interp::tier0::exec::{run_batch, BatchExit};
use aero_cpu_core::mem::{CpuBus, FlatTestBus};
use aero_cpu_core::state::{CpuMode, CpuState};
use aero_x86::Register;

const FN: &[u8] = include_bytes!("data/bcryptprimitives_sha512_compress.bin");
const K_LE: &[u8] = &aero_cpu_core::sha2_constants::SHA512_K_LE;

/// Original guest RVA so the function's RIP-relative K-table LEAs stay valid.
const CODE: u64 = 0xb000;
const K_ADDR: u64 = 0x3b400;
const HLT: u64 = 0x1f00;
const STACK: u64 = 0x8000;
const STATE: u64 = 0x2000;
const BLOCK: u64 = 0x3000;

const SHA512_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

fn k_table() -> [u64; 80] {
    let mut k = [0u64; 80];
    for (i, slot) in k.iter_mut().enumerate() {
        let off = i * 8;
        *slot = u64::from_le_bytes(K_LE[off..off + 8].try_into().unwrap());
    }
    k
}

fn rotr(x: u64, n: u32) -> u64 {
    x.rotate_right(n)
}

fn sha512_compress(state: &mut [u64; 8], block: &[u8; 128]) {
    let k = k_table();
    let mut w = [0u64; 80];
    for i in 0..16 {
        w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    for i in 16..80 {
        let s0 = rotr(w[i - 15], 1) ^ rotr(w[i - 15], 8) ^ (w[i - 15] >> 7);
        let s1 = rotr(w[i - 2], 19) ^ rotr(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = s1
            .wrapping_add(w[i - 7])
            .wrapping_add(s0)
            .wrapping_add(w[i - 16]);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    for i in 0..80 {
        let s1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k[i])
            .wrapping_add(w[i]);
        let s0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn pad_message(msg: &[u8]) -> [u8; 128] {
    assert!(msg.len() < 112, "single-block messages only");
    let mut block = [0u8; 128];
    block[..msg.len()].copy_from_slice(msg);
    block[msg.len()] = 0x80;
    let bits = (msg.len() as u128) * 8;
    block[112..].copy_from_slice(&bits.to_be_bytes());
    block
}

fn write_state(bus: &mut FlatTestBus, addr: u64, words: &[u64; 8]) {
    for (i, &word) in words.iter().enumerate() {
        bus.write_u64(addr + (i as u64) * 8, word).unwrap();
    }
}

fn read_state(bus: &mut FlatTestBus, addr: u64) -> [u64; 8] {
    let mut out = [0u64; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = bus.read_u64(addr + (i as u64) * 8).unwrap();
    }
    out
}

fn new_machine() -> (CpuState, FlatTestBus) {
    let mut bus = FlatTestBus::new(0x4_0000);
    bus.load(HLT, &[0xF4]);
    bus.load(CODE, FN);
    bus.load(K_ADDR, K_LE);
    let mut state = CpuState::new(CpuMode::Long);
    state.set_rip(CODE);
    bus.write_u64(STACK - 8, HLT).unwrap();
    state.write_reg(Register::RSP, STACK - 8);
    (state, bus)
}

fn run_compress(state: &mut CpuState, bus: &mut FlatTestBus) {
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
            other => panic!(
                "unexpected exit {other:?} rip={:#x} after {steps} insts",
                state.rip()
            ),
        }
    }
    panic!("SHA-512 compress did not halt; rip={:#x}", state.rip());
}

fn run_guest_compress(iv: [u64; 8], block: &[u8; 128]) -> [u64; 8] {
    let (mut state, mut bus) = new_machine();
    write_state(&mut bus, STATE, &iv);
    bus.load(BLOCK, block);
    state.write_reg(Register::RCX, STATE);
    state.write_reg(Register::RDX, BLOCK);
    run_compress(&mut state, &mut bus);
    read_state(&mut bus, STATE)
}

#[test]
fn sha512_k_table_matches_fips_first_and_last() {
    let k = k_table();
    assert_eq!(k[0], 0x428a2f98d728ae22);
    assert_eq!(k[1], 0x7137449123ef65cd);
    assert_eq!(k[79], 0x6c44198c4a475817);
}

#[test]
fn ror64_sha512_counts_match_rotate_right() {
    // Sigma0/1 and the message schedule use counts that do not fit in 5 bits.
    // A 32-bit-style `count & 31` would turn ROR 34/39/41/61 into 2/7/9/29.
    let cases: &[(u64, u32)] = &[
        (0x6a09e667f3bcc908, 14),
        (0x6a09e667f3bcc908, 18),
        (0x6a09e667f3bcc908, 28),
        (0x6a09e667f3bcc908, 34),
        (0x6a09e667f3bcc908, 39),
        (0x6a09e667f3bcc908, 41),
        (0xbb67ae8584caa73b, 1),
        (0xbb67ae8584caa73b, 8),
        (0xbb67ae8584caa73b, 19),
        (0xbb67ae8584caa73b, 61),
        (u64::MAX, 41),
        (0x0123_4567_89ab_cdef, 34),
    ];
    for &(val, count) in cases {
        let (mut state, mut bus) = new_machine();
        state.write_reg(Register::RAX, val);
        // ror rax, imm8 ; hlt   (REX.W C1 /1)
        bus.load(0x1000, &[0x48, 0xc1, 0xc8, count as u8, 0xF4]);
        state.set_rip(0x1000);
        state.halted = false;
        let mut steps = 0u64;
        while steps < 8 {
            let res = run_batch(&mut state, &mut bus, 1);
            steps += res.executed;
            match res.exit {
                BatchExit::Halted => break,
                BatchExit::Completed | BatchExit::Branch => continue,
                other => panic!("ror rax,{count}: {other:?}"),
            }
        }
        assert_eq!(
            state.read_reg(Register::RAX),
            val.rotate_right(count),
            "ROR r64, {count} of {val:#x}"
        );
    }
}

#[test]
fn bswap64_reverses_bytes() {
    let (mut state, mut bus) = new_machine();
    state.write_reg(Register::R8, 0x0123_4567_89ab_cdef);
    bus.load(0x1000, &[0x49, 0x0f, 0xc8, 0xF4]); // bswap r8 ; hlt
    state.set_rip(0x1000);
    state.halted = false;
    loop {
        match run_batch(&mut state, &mut bus, 1).exit {
            BatchExit::Halted => break,
            BatchExit::Completed | BatchExit::Branch => continue,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(state.read_reg(Register::R8), 0xefcd_ab89_6745_2301);
}

#[test]
fn bcryptprimitives_sha512_compress_empty_message_matches_fips() {
    let block = pad_message(b"");
    let mut expect = SHA512_IV;
    sha512_compress(&mut expect, &block);
    // SHA-512("") from FIPS 180-4 / hashlib
    let fips = [
        0xcf83e1357eefb8bd,
        0xf1542850d66d8007,
        0xd620e4050b5715dc,
        0x83f4a921d36ce9ce,
        0x47d0d13c5d85f2b0,
        0xff8318d2877eec2f,
        0x63b931bd47417a81,
        0xa538327af927da3e,
    ];
    assert_eq!(expect, fips, "host reference must match hashlib");

    let got = run_guest_compress(SHA512_IV, &block);
    assert_eq!(got, fips, "guest compress of empty-message pad block");
}

#[test]
fn bcryptprimitives_sha512_compress_abc_matches_fips() {
    let block = pad_message(b"abc");
    let mut expect = SHA512_IV;
    sha512_compress(&mut expect, &block);
    let fips = [
        0xddaf35a193617aba,
        0xcc417349ae204131,
        0x12e6fa4e89a97ea2,
        0x0a9eeee64b55d39a,
        0x2192992a274fc1a8,
        0x36ba3c23a3feebbd,
        0x454d4423643ce80e,
        0x2a9ac94fa54ca49f,
    ];
    assert_eq!(expect, fips, "host reference must match hashlib");

    let got = run_guest_compress(SHA512_IV, &block);
    assert_eq!(got, fips, "guest compress of 'abc' pad block");
}

#[test]
fn bcryptprimitives_sha512_compress_nonzero_state_matches_reference() {
    // The live plus2b-3600m path also compresses already-advanced states
    // (HMAC inner/outer, multi-block). Use a non-IV chaining value.
    let mut iv = SHA512_IV;
    for (i, word) in iv.iter_mut().enumerate() {
        *word ^= 0x1111_1111_1111_1111u64.wrapping_mul(i as u64 + 1);
    }
    let mut block = [0u8; 128];
    for (i, byte) in block.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
    }
    let mut expect = iv;
    sha512_compress(&mut expect, &block);
    let got = run_guest_compress(iv, &block);
    assert_eq!(got, expect);
}

#[test]
fn bcryptprimitives_sha512_compress_live_hmac_ipad_matches_guest() {
    // plus2b-3600m watch: SHA512Init + first HMAC-SHA512 inner block
    // (key xor 0x36, then 0x36 padding). Guest store-back matched this
    // host compress, so the stall is HMAC volume, not a bad hash.
    let block: [u8; 128] = [
        0x03, 0xe9, 0x88, 0x8f, 0x88, 0xfe, 0x3e, 0x5e, 0x97, 0x8c, 0x1b, 0x67, 0x35, 0xe8, 0xa8,
        0x39, 0x63, 0x2b, 0x52, 0x94, 0x89, 0x1b, 0xea, 0x07, 0xfc, 0xb0, 0xac, 0xcc, 0xd8, 0xfe,
        0x1a, 0xc3, 0x1c, 0x48, 0x19, 0xbd, 0x69, 0xeb, 0x51, 0xa8, 0xae, 0x78, 0x86, 0x08, 0x35,
        0x91, 0x46, 0x79, 0x8e, 0x08, 0x56, 0x7c, 0xcd, 0xa6, 0xc2, 0xad, 0x0d, 0xad, 0x0c, 0x06,
        0x99, 0x1d, 0x6f, 0xc6, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
        0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
        0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
        0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
        0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36,
    ];
    let guest = [
        0x5fc726dce25bf263,
        0x6a13d136119e1058,
        0x0283689867077647,
        0x1da0ef54722928e4,
        0x0de6041316147287,
        0x88aa69ea32032765,
        0x9e4555d61b3f0687,
        0x6183ace566c5f204,
    ];
    let mut expect = SHA512_IV;
    sha512_compress(&mut expect, &block);
    assert_eq!(expect, guest);
    let got = run_guest_compress(SHA512_IV, &block);
    assert_eq!(got, guest);
}

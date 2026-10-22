//! Win7 `cryptsp` SHA-256 compress (`RVA 0xe210`) vs FIPS 180-4.
//!
//! VDS `ServiceMain` dies with `NTE_BAD_SIGNATURE` (`0x80090006`) during
//! `AtlModuleRegisterClassObjects`. The service worker is in this exact
//! SHA-256 leaf (CALG_SHA_256 / `0x800c`) while that call is on the stack.
//! A wrong 32-bit ROR/BSWAP/ADD here would make every SHA-256 Authenticode
//! (the 2018 Win7 dual-sig on `rsaenh.dll`) fail even when 4096-bit RSA is
//! correct.

use aero_cpu_core::interp::tier0::exec::{run_batch, BatchExit};
use aero_cpu_core::mem::{CpuBus, FlatTestBus};
use aero_cpu_core::state::{CpuMode, CpuState};
use aero_x86::Register;

const FN: &[u8] = include_bytes!("data/cryptsp_sha256_compress.bin");
const K_LE: &[u8] = &aero_cpu_core::sha2_constants::SHA256_K_LE;

/// Original guest RVAs so RIP-relative K-table LEAs stay valid.
const CODE: u64 = 0xe210;
const K_ADDR: u64 = 0x1590;
const HLT: u64 = 0x1f00;
const STACK: u64 = 0x8000;
const STATE: u64 = 0x2000;
const BLOCK: u64 = 0x3000;

const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn k_table() -> [u32; 64] {
    let mut k = [0u32; 64];
    for (i, slot) in k.iter_mut().enumerate() {
        let off = i * 4;
        *slot = u32::from_le_bytes(K_LE[off..off + 4].try_into().unwrap());
    }
    k
}

fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let k = k_table();
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
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
    for i in 0..64 {
        let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k[i])
            .wrapping_add(w[i]);
        let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
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

fn new_machine() -> (CpuState, FlatTestBus) {
    let mut bus = FlatTestBus::new(0x1_0000);
    bus.load(HLT, &[0xF4]);
    bus.load(CODE, FN);
    bus.load(K_ADDR, K_LE);
    let mut state = CpuState::new(CpuMode::Long);
    state.set_rip(CODE);
    (state, bus)
}

fn run_compress(state_words: [u32; 8], block: [u8; 64]) -> [u32; 8] {
    let (mut cpu, mut bus) = new_machine();
    for (i, word) in state_words.iter().enumerate() {
        bus.write_u32(STATE + (i as u64) * 4, *word).unwrap();
    }
    bus.load(BLOCK, &block);
    bus.write_u64(STACK - 8, HLT).unwrap();
    cpu.write_reg(Register::RSP, STACK - 8);
    cpu.write_reg(Register::RCX, STATE);
    cpu.write_reg(Register::RDX, BLOCK);
    cpu.set_rip(CODE);
    cpu.halted = false;
    let mut steps = 0u64;
    while steps < 200_000 {
        let res = run_batch(&mut cpu, &mut bus, 64);
        steps += res.executed;
        match res.exit {
            BatchExit::Halted => break,
            BatchExit::Completed | BatchExit::Branch => continue,
            other => panic!(
                "compress exit {other:?} after {steps} inst rip={:#x}",
                cpu.rip()
            ),
        }
    }
    assert!(cpu.halted, "compress did not halt after {steps} inst");
    let mut out = [0u32; 8];
    for (i, word) in out.iter_mut().enumerate() {
        *word = bus.read_u32(STATE + (i as u64) * 4).unwrap();
    }
    out
}

#[test]
fn ror32_sha256_counts_match_rotate_right() {
    let cases = [
        (0x6a09e667u32, 2u32),
        (0x6a09e667, 6),
        (0x6a09e667, 11),
        (0x6a09e667, 13),
        (0x6a09e667, 22),
        (0x6a09e667, 25),
        (0xffff_ffff, 7),
        (0x0123_4567, 18),
    ];
    for &(val, count) in &cases {
        let (mut state, mut bus) = new_machine();
        state.write_reg(Register::EAX, val as u64);
        // ror eax, imm8 ; hlt   (C1 /1)
        bus.load(0x1000, &[0xc1, 0xc8, count as u8, 0xF4]);
        state.set_rip(0x1000);
        state.halted = false;
        loop {
            match run_batch(&mut state, &mut bus, 1).exit {
                BatchExit::Halted => break,
                BatchExit::Completed | BatchExit::Branch => continue,
                other => panic!("ror eax,{count}: {other:?}"),
            }
        }
        assert_eq!(
            state.read_reg(Register::EAX) as u32,
            val.rotate_right(count),
            "ROR r32, {count} of {val:#x}"
        );
        assert_eq!(
            state.read_reg(Register::RAX) >> 32,
            0,
            "32-bit ROR must zero-extend"
        );
    }
}

#[test]
fn bswap64_then_ror32_matches_cryptsp_message_load() {
    // cryptsp loads two BE dwords via `bswap rax; ror rax, 32`.
    let (mut state, mut bus) = new_machine();
    state.write_reg(Register::RAX, 0x0123_4567_89ab_cdef);
    bus.load(
        0x1000,
        &[
            0x48, 0x0f, 0xc8, // bswap rax
            0x48, 0xc1, 0xc8, 0x20, // ror rax, 32
            0xF4,
        ],
    );
    state.set_rip(0x1000);
    state.halted = false;
    loop {
        match run_batch(&mut state, &mut bus, 1).exit {
            BatchExit::Halted => break,
            BatchExit::Completed | BatchExit::Branch => continue,
            other => panic!("{other:?}"),
        }
    }
    // bytes 01 23 45 67 89 ab cd ef → bswap efcdab8967452301 → ror32 67452301efcdab89
    assert_eq!(state.read_reg(Register::RAX), 0x6745_2301_efcd_ab89);
}

#[test]
fn cryptsp_sha256_compress_zero_block_matches_fips() {
    let block = [0u8; 64];
    let mut expect = SHA256_IV;
    sha256_compress(&mut expect, &block);
    let got = run_compress(SHA256_IV, block);
    assert_eq!(got, expect, "cryptsp SHA-256 compress of a zero block");
}

#[test]
fn cryptsp_sha256_compress_abc_block_matches_fips() {
    // One-block padded "abc" (FIPS 180-4 B.1).
    let mut block = [0u8; 64];
    block[0] = b'a';
    block[1] = b'b';
    block[2] = b'c';
    block[3] = 0x80;
    block[63] = 24;
    let mut expect = SHA256_IV;
    sha256_compress(&mut expect, &block);
    assert_eq!(
        expect,
        [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad
        ],
        "host reference must match the published SHA-256(\"abc\")"
    );
    let got = run_compress(SHA256_IV, block);
    assert_eq!(got, expect, "cryptsp SHA-256 compress of padded \"abc\"");
}

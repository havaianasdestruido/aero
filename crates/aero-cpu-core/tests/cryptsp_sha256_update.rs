//! Win7 `cryptsp` SHA-256 init/update/final vs hashlib, including the
//! Authenticode PE hasher (`0xa298`) that `CheckSignatureInFile` uses.
//!
//! VDS `CryptAcquireContext` dies with `CRYPT_E_HASH_VALUE` (`0x80091007`)
//! because the 32-byte digest `0x859c` writes does not match the PKCS#7
//! SHA-256 of `rsaenh.dll`. Single-block compress already matches FIPS; this
//! file runs the streamed update/final path and the real skip-list hasher.

use aero_cpu_core::interp::tier0::exec::{run_batch, BatchExit};
use aero_cpu_core::mem::{CpuBus, FlatTestBus};
use aero_cpu_core::state::{CpuMode, CpuState};
use aero_x86::Register;

/// The mapped `cryptsp.dll` image these tests execute against is a copy of a
/// Microsoft binary and is deliberately **not committed**. Supply one locally to
/// run them — point `AERO_CRYPTSP_IMAGE` at it, or drop it at
/// `tests/data/cryptsp_mapped.bin` — and without it these tests skip rather than
/// fail. See `tests/data/README.md`.
fn cryptsp_image() -> Option<Vec<u8>> {
    // Setting the variable is a statement of intent: if it points at something
    // unreadable that is a mistake worth failing on, not a reason to skip. Only
    // the *default* path is allowed to be absent.
    match std::env::var("AERO_CRYPTSP_IMAGE") {
        Ok(path) => {
            Some(std::fs::read(&path).unwrap_or_else(|e| panic!("AERO_CRYPTSP_IMAGE={path}: {e}")))
        }
        Err(_) => {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/cryptsp_mapped.bin");
            match std::fs::read(path) {
                Ok(bytes) => Some(bytes),
                Err(_) => {
                    // Say so out loud. A silent early return reports "ok" and is
                    // indistinguishable from a test that actually ran.
                    eprintln!("SKIP: no cryptsp image at {path} (set AERO_CRYPTSP_IMAGE)");
                    None
                }
            }
        }
    }
}

const HLT: u64 = 0x1_f000;
const STUB_MEMCPY: u64 = 0x1_9000;
const STUB_MEMSET: u64 = 0x1_9080;
const HASH_OBJ: u64 = 0x1_e000;
const DIGEST: u64 = 0x1_e800;
const STACK: u64 = 0x1_f800;
const FILE: u64 = 0x2_0000;

const SHA256_INIT: u64 = 0xe1c0;
const SHA256_UPDATE: u64 = 0xecd0;
const SHA256_FINAL: u64 = 0xee20;
const PE_HASH: u64 = 0xa298;
const COOKIE_CHECK: u64 = 0xaa90;
const CALG_SHA_256: u32 = 0x800c;

fn load_cryptsp(bus: &mut FlatTestBus, image: &[u8]) {
    bus.load(0, image);
}

/// Windows x64 `memcpy`/`memmove` (rcx=dst, rdx=src, r8=len) and `memset`.
fn memcpy_bytes() -> Vec<u8> {
    vec![
        0x56, // push rsi
        0x57, // push rdi
        0x48, 0x89, 0xcf, // mov rdi, rcx
        0x48, 0x89, 0xd6, // mov rsi, rdx
        0x4c, 0x89, 0xc1, // mov rcx, r8
        0x48, 0x89, 0xf8, // mov rax, rdi
        0xf3, 0xa4, // rep movsb
        0x5f, // pop rdi
        0x5e, // pop rsi
        0xc3, // ret
    ]
}

fn memset_bytes() -> Vec<u8> {
    vec![
        0x57, // push rdi
        0x48, 0x89, 0xcf, // mov rdi, rcx
        0x48, 0x89, 0xd0, // mov rax, rdx
        0x4c, 0x89, 0xc1, // mov rcx, r8
        0xf3, 0xaa, // rep stosb
        0x5f, // pop rdi
        0xc3, // ret
    ]
}

fn new_cryptsp() -> Option<(CpuState, FlatTestBus)> {
    let image = cryptsp_image()?;
    let mut bus = FlatTestBus::new(0x8_0000);
    bus.load(HLT, &[0xF4]);
    load_cryptsp(&mut bus, &image);
    bus.load(STUB_MEMCPY, &memcpy_bytes());
    bus.load(STUB_MEMSET, &memset_bytes());
    // IAT slots used by memcpy (0xcdc4), memset (0xcdd0), memmove (0x1019a).
    bus.write_u64(0x1268, STUB_MEMCPY).unwrap();
    bus.write_u64(0x1270, STUB_MEMSET).unwrap();
    bus.write_u64(0x12b0, STUB_MEMCPY).unwrap();
    bus.load(COOKIE_CHECK, &[0xC3]);
    let cpu = CpuState::new(CpuMode::Long);
    Some((cpu, bus))
}

fn run_until_hlt(cpu: &mut CpuState, bus: &mut FlatTestBus, limit: u64) {
    let mut steps = 0u64;
    cpu.halted = false;
    while steps < limit {
        let res = run_batch(cpu, bus, 64);
        steps += res.executed;
        match res.exit {
            BatchExit::Halted => return,
            BatchExit::Completed | BatchExit::Branch => continue,
            other => panic!(
                "guest exit {other:?} after {steps} inst rip={:#x}",
                cpu.rip()
            ),
        }
    }
    panic!("did not halt after {steps} inst rip={:#x}", cpu.rip());
}

fn call(cpu: &mut CpuState, bus: &mut FlatTestBus, rip: u64, limit: u64) {
    bus.write_u64(STACK - 8, HLT).unwrap();
    cpu.write_reg(Register::RSP, STACK - 8);
    cpu.set_rip(rip);
    run_until_hlt(cpu, bus, limit);
}

fn sha256_via_cryptsp(data: &[u8]) -> Option<[u8; 32]> {
    let (mut cpu, mut bus) = new_cryptsp()?;
    bus.load(FILE, data);
    cpu.write_reg(Register::RCX, HASH_OBJ);
    call(&mut cpu, &mut bus, SHA256_INIT, 10_000);
    cpu.write_reg(Register::RCX, HASH_OBJ);
    cpu.write_reg(Register::RDX, FILE);
    cpu.write_reg(Register::R8, data.len() as u64);
    call(&mut cpu, &mut bus, SHA256_UPDATE, 2_000_000);
    cpu.write_reg(Register::RCX, HASH_OBJ);
    cpu.write_reg(Register::RDX, DIGEST);
    call(&mut cpu, &mut bus, SHA256_FINAL, 200_000);
    let mut out = [0u8; 32];
    out.copy_from_slice(bus.slice(DIGEST, 32));
    Some(out)
}

fn rotr(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

fn host_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    // The published FIPS round constants, from the one place they are defined
    // and re-derived: a fourth hand-copy is a fourth chance for a typo.
    const K: [u32; 64] = aero_cpu_core::sha2_constants::SHA256_K;
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
            .wrapping_add(K[i])
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

fn host_sha256(data: &[u8]) -> [u8; 32] {
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut i = 0;
    while i + 64 <= data.len() {
        host_compress(&mut state, data[i..i + 64].try_into().unwrap());
        i += 64;
    }
    let mut block = [0u8; 64];
    let rem = data.len() - i;
    block[..rem].copy_from_slice(&data[i..]);
    block[rem] = 0x80;
    let bit_len = (data.len() as u64).saturating_mul(8);
    if rem >= 56 {
        host_compress(&mut state, &block);
        block = [0u8; 64];
    }
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    host_compress(&mut state, &block);
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn tiny_pe() -> (Vec<u8>, [u8; 32]) {
    let mut dos = vec![0u8; 0x80];
    dos[0] = b'M';
    dos[1] = b'Z';
    dos[0x3c] = 0x80;
    let mut pe = vec![0u8; 0x108];
    pe[0..4].copy_from_slice(b"PE\0\0");
    pe[4..6].copy_from_slice(&0x8664u16.to_le_bytes());
    pe[6..8].copy_from_slice(&1u16.to_le_bytes());
    // COFF: SizeOfOptionalHeader at +16, Characteristics at +18, then optional.
    pe[20..22].copy_from_slice(&0x00f0u16.to_le_bytes());
    pe[22..24].copy_from_slice(&0x0022u16.to_le_bytes());
    pe[24..26].copy_from_slice(&0x020bu16.to_le_bytes());
    pe[26] = 8; // MajorLinkerVersion — cryptsp rejects < 3
    pe[24 + 32..24 + 36].copy_from_slice(&0x1000u32.to_le_bytes());
    pe[24 + 36..24 + 40].copy_from_slice(&0x200u32.to_le_bytes());
    pe[24 + 56..24 + 60].copy_from_slice(&0x2000u32.to_le_bytes());
    pe[24 + 60..24 + 64].copy_from_slice(&0x200u32.to_le_bytes());
    pe[24 + 64..24 + 68].copy_from_slice(&0x1234_5678u32.to_le_bytes());
    pe[24 + 108..24 + 112].copy_from_slice(&16u32.to_le_bytes());
    let cert_off = 0x400u32;
    let cert_sz = 0x80u32;
    let cert_dir = 24 + 112 + 4 * 8;
    pe[cert_dir..cert_dir + 4].copy_from_slice(&cert_off.to_le_bytes());
    pe[cert_dir + 4..cert_dir + 8].copy_from_slice(&cert_sz.to_le_bytes());
    // SizeOfOptionalHeader is 0xf0; SizeOfHeaders 0x200.
    let mut sect = vec![0u8; 40];
    sect[0..5].copy_from_slice(b".text");
    sect[8..12].copy_from_slice(&0x200u32.to_le_bytes());
    sect[12..16].copy_from_slice(&0x1000u32.to_le_bytes());
    sect[16..20].copy_from_slice(&0x200u32.to_le_bytes());
    sect[20..24].copy_from_slice(&0x200u32.to_le_bytes());
    sect[36..40].copy_from_slice(&0x0000_0020u32.to_le_bytes());
    let mut file = dos;
    file.extend_from_slice(&pe);
    file.extend_from_slice(&sect);
    file.resize(0x200, 0);
    file.extend((0..0x200).map(|i| i as u8));
    file.extend(std::iter::repeat(0x30u8).take(0x80));
    file[cert_off as usize] = 0x80;
    file[cert_off as usize + 4] = 0x00;
    file[cert_off as usize + 5] = 0x02;
    file[cert_off as usize + 6] = 0x02;
    let checksum = 0x80 + 24 + 64;
    let cdir = 0x80 + cert_dir;
    let mut stream = Vec::new();
    stream.extend_from_slice(&file[..checksum]);
    stream.extend_from_slice(&file[checksum + 4..cdir]);
    stream.extend_from_slice(&file[cdir + 8..cert_off as usize]);
    stream.extend_from_slice(&file[cert_off as usize + cert_sz as usize..]);
    (file, host_sha256(&stream))
}

#[test]
fn cryptsp_sha256_update_final_abc_matches_fips_if_fixture_present() {
    let expect = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    // Pin the host reference to the published vector first — this needs no
    // fixture, and every other assertion in this file trusts it.
    assert_eq!(host_sha256(b"abc"), expect, "host SHA-256(\"abc\") is FIPS");
    let Some(got) = sha256_via_cryptsp(b"abc") else {
        return;
    };
    assert_eq!(got, expect, "update+final of \"abc\"");
}

#[test]
fn cryptsp_sha256_update_final_unaligned_multiblock_matches_host_if_fixture_present() {
    let data: Vec<u8> = (0..10_000).map(|i| (i * 17 + 3) as u8).collect();
    let Some(got) = sha256_via_cryptsp(&data) else {
        return;
    };
    assert_eq!(got, host_sha256(&data), "10k unaligned-length SHA-256");
}

#[test]
fn cryptsp_sha256_update_final_authenticode_stream_matches_host_if_fixture_present() {
    let (file, expect) = tiny_pe();
    let checksum = 0x80 + 24 + 64;
    let cdir = 0x80 + 24 + 112 + 32;
    let cert_off = 0x400;
    let cert_sz = 0x80;
    let mut stream = Vec::new();
    stream.extend_from_slice(&file[..checksum]);
    stream.extend_from_slice(&file[checksum + 4..cdir]);
    stream.extend_from_slice(&file[cdir + 8..cert_off]);
    stream.extend_from_slice(&file[cert_off + cert_sz..]);
    assert_eq!(host_sha256(&stream), expect);
    let Some(got) = sha256_via_cryptsp(&stream) else {
        return;
    };
    assert_eq!(got, expect, "SHA-256 of the Authenticode byte stream");
}

#[test]
fn cryptsp_pe_hasher_a298_matches_authenticode_sha256_if_fixture_present() {
    let (file, expect) = tiny_pe();
    let Some((mut cpu, mut bus)) = new_cryptsp() else {
        return;
    };
    bus.load(FILE, &file);
    bus.write_u32(HASH_OBJ, CALG_SHA_256).unwrap();
    for off in 0u64..0x48 {
        bus.write_u64(STACK + off, 0).unwrap();
    }
    // 7th argument (entry [rsp+0x38]) is the digest destination for 0xa21c.
    bus.write_u64(STACK - 8 + 0x38, DIGEST).unwrap();
    bus.write_u64(STACK - 8, HLT).unwrap();
    cpu.write_reg(Register::RSP, STACK - 8);
    cpu.write_reg(Register::RCX, HASH_OBJ);
    cpu.write_reg(Register::RDX, FILE);
    cpu.write_reg(Register::R8, file.len() as u64);
    cpu.write_reg(Register::R9, file.len() as u64);
    cpu.set_rip(PE_HASH);
    run_until_hlt(&mut cpu, &mut bus, 2_000_000);
    assert_eq!(
        cpu.read_reg(Register::RAX) as u32,
        0,
        "0xa298 returned {:#x}",
        cpu.read_reg(Register::RAX)
    );
    let mut digest = [0u8; 32];
    digest.copy_from_slice(bus.slice(DIGEST, 32));
    assert_eq!(digest, expect, "0xa298 Authenticode SHA-256 of the tiny PE");
}

#[test]
fn cryptsp_pe_hasher_a298_matches_rsaenh_if_fixture_present() {
    let path = match std::env::var("AERO_RSAENH") {
        Ok(p) => p,
        Err(_) => return,
    };
    let file = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let e_lfanew = u32::from_le_bytes(file[0x3c..0x40].try_into().unwrap()) as usize;
    let checksum = e_lfanew + 24 + 64;
    let cdir = e_lfanew + 24 + 112 + 32;
    let cert_off = u32::from_le_bytes(file[cdir..cdir + 4].try_into().unwrap()) as usize;
    let cert_sz = u32::from_le_bytes(file[cdir + 4..cdir + 8].try_into().unwrap()) as usize;
    let mut stream = Vec::new();
    stream.extend_from_slice(&file[..checksum]);
    stream.extend_from_slice(&file[checksum + 4..cdir]);
    stream.extend_from_slice(&file[cdir + 8..cert_off]);
    stream.extend_from_slice(&file[cert_off + cert_sz..]);
    let expect = host_sha256(&stream);

    let Some((mut cpu, mut bus)) = new_cryptsp() else {
        return;
    };
    bus.load(FILE, &file);
    bus.write_u32(HASH_OBJ, CALG_SHA_256).unwrap();
    for off in 0u64..0x48 {
        bus.write_u64(STACK + off, 0).unwrap();
    }
    bus.write_u64(STACK - 8 + 0x38, DIGEST).unwrap();
    bus.write_u64(STACK - 8, HLT).unwrap();
    cpu.write_reg(Register::RSP, STACK - 8);
    cpu.write_reg(Register::RCX, HASH_OBJ);
    cpu.write_reg(Register::RDX, FILE);
    cpu.write_reg(Register::R8, file.len() as u64);
    cpu.write_reg(Register::R9, file.len() as u64);
    cpu.set_rip(PE_HASH);
    run_until_hlt(&mut cpu, &mut bus, 20_000_000);
    assert_eq!(
        cpu.read_reg(Register::RAX) as u32,
        0,
        "0xa298 rsaenh returned {:#x}",
        cpu.read_reg(Register::RAX)
    );
    let mut digest = [0u8; 32];
    digest.copy_from_slice(bus.slice(DIGEST, 32));
    assert_eq!(
        digest,
        expect,
        "0xa298 SHA-256 of rsaenh.dll\ngot {}\nexp {}",
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        expect
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
}

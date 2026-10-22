use crate::Tier1Bus;
use aero_x86::tier1::{decode_one_mode, DecodedInst, InstKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLimits {
    pub max_insts: usize,
    pub max_bytes: usize,
    /// Keep decoding past a conditional branch, along its fall-through.
    ///
    /// A basic block ends at the first branch, which is why blocks here average four
    /// guest instructions — and every end costs a register spill, a host call and a
    /// reload. Continuing turns the branch into a side exit and keeps the guest
    /// registers in wasm locals across it.
    pub follow_conditionals: bool,
    /// Whether discovery continues into the target of a direct `call` instead of ending
    /// the block there.
    ///
    /// `call` is **55.6% of block ends** once conditionals and unconditional jumps are
    /// followed — by far the largest remaining limit on block length, and block length is
    /// what the per-entry cost is amortised over. The translation needs nothing new: a
    /// `CallRel` already pushes the return address and *then* sets a terminator, so
    /// following it is a matter of not terminating.
    pub follow_calls: bool,
    /// Whether `RDTSC` may be translated rather than ending the block.
    ///
    /// It is the single largest coverage wall in a Windows desktop session, but it
    /// reads the machine's clock through a host call — so a caller with no host to
    /// call leaves it off, and so does one running with `AERO_RDTSC_QUANTUM`, whose
    /// tight-poll detection lives in the interpreter's assist and would not see the
    /// reads a compiled block makes.
    pub allow_rdtsc: bool,
    /// Whether an instruction Tier-1 cannot translate may be handed to the host
    /// mid-block instead of ending it. Requires the runtime to provide `env.assist`.
    pub allow_assists: bool,
}

impl Default for BlockLimits {
    fn default() -> Self {
        Self {
            max_insts: 64,
            max_bytes: 1024,
            follow_conditionals: false,
            follow_calls: false,
            allow_rdtsc: false,
            allow_assists: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockEndKind {
    Jmp,
    Jcc,
    Call,
    Ret,
    ExitToInterpreter { next_rip: u64 },
    Limit { next_rip: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub entry_rip: u64,
    pub bitness: u32,
    pub insts: Vec<DecodedInst>,
    pub end_kind: BlockEndKind,
}

/// Decode a basic block starting at `entry_rip`.
///
/// Decoding stops when one of the following conditions is met:
/// - a control-flow instruction is decoded (branch/call/ret)
/// - an unsupported/invalid instruction is decoded (must exit to interpreter)
/// - `limits` are exceeded
#[must_use]
pub fn discover_block<B: Tier1Bus>(bus: &B, entry_rip: u64, limits: BlockLimits) -> BasicBlock {
    discover_block_mode(bus, entry_rip, limits, 64)
}

/// Decode a basic block starting at `entry_rip` using the requested x86 bitness.
///
/// This is a thin wrapper around the Tier1 minimal decoder (`aero_x86::tier1`) that allows
/// front-ends/tests to run 16/32-bit guest payloads without mis-decoding `0x40..=0x4F` as REX.
#[must_use]
#[track_caller]
pub fn discover_block_mode<B: Tier1Bus>(
    bus: &B,
    entry_rip: u64,
    limits: BlockLimits,
    bitness: u32,
) -> BasicBlock {
    let follow_jmp = follow_jmp_enabled();
    let mut next_rip_override: Option<u64> = None;
    let mut insts = Vec::new();
    let ip_mask = match bitness {
        32 => 0xffff_ffff,
        64 => u64::MAX,
        // Tier-1 decoder only partially models 16-bit mode (eg. some 16-bit ModRM addressing
        // forms), but applying the 16-bit IP mask here keeps instruction fetch consistent with
        // the architectural IP width if callers experiment with `bitness=16`.
        16 => 0xffff,
        other => panic!("invalid x86 bitness {other}"),
    };
    let entry_rip = entry_rip & ip_mask;
    let mut rip = entry_rip;
    let mut total_bytes = 0usize;

    loop {
        if insts.len() >= limits.max_insts || total_bytes >= limits.max_bytes {
            return BasicBlock {
                entry_rip,
                bitness,
                insts,
                end_kind: BlockEndKind::Limit { next_rip: rip },
            };
        }

        // Instruction fetch must respect architectural IP width for 16/32-bit guests.
        //
        // `Tier1Bus` helpers use `u64` wrapping arithmetic; for 16/32-bit modes we instead mask
        // each byte address so instructions that straddle the architectural wrap boundary (e.g.
        // EIP=0xFFFF_FFFF) decode consistently with the guest IP size.
        //
        // When the full 15-byte decode window is contiguous within the masked IP space, we can
        // use `Tier1Bus::fetch_window()` directly (avoids per-byte masking and heap allocation).
        let bytes = if ip_mask == u64::MAX || rip <= ip_mask - 14 {
            bus.fetch_window(rip)
        } else {
            let mut buf = [0u8; 15];
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = bus.read_u8(rip.wrapping_add(i as u64) & ip_mask);
            }
            buf
        };
        let inst = decode_one_mode(rip, &bytes, bitness);
        total_bytes += inst.len as usize;
        rip = inst.next_rip() & ip_mask;

        let end_kind = match inst.kind {
            // An unconditional direct jump can be followed instead of ending the
            // block: the target is known and always taken, so unlike fusing a
            // conditional there is no side exit and nothing speculative about it.
            // 10.9% of executed chain steps enter a block that ends this way, and a
            // block boundary costs ~283 host instructions.
            //
            // Not backwards and not into what has already been decoded: a jump back
            // into this block is a loop, and following it would decode the body over
            // and over until the limits stopped it.
            InstKind::JmpRel { target }
                if follow_jmp
                    && insts.len() + 1 < limits.max_insts
                    && !already_decoded(&insts, target & ip_mask) =>
            {
                next_rip_override = Some(target & ip_mask);
                None
            }
            InstKind::JmpRel { .. } => Some(BlockEndKind::Jmp),
            // In superblock mode the fall-through continues in the same block and the
            // taken side becomes an exit; the branch only *ends* the block when it is
            // the last instruction the limits allow.
            InstKind::JccRel { .. } if limits.follow_conditionals => None,
            InstKind::JccRel { .. } => Some(BlockEndKind::Jcc),
            // As for `JmpRel`: the target is known and always taken. The pushed return
            // address is an ordinary store the translation already emits, so the only
            // difference is that decoding continues at the callee instead of stopping.
            InstKind::CallRel { target }
                if limits.follow_calls
                    && insts.len() + 1 < limits.max_insts
                    && !already_decoded(&insts, target & ip_mask) =>
            {
                next_rip_override = Some(target & ip_mask);
                None
            }
            InstKind::CallRel { .. } => Some(BlockEndKind::Call),
            InstKind::Rdtsc if !limits.allow_rdtsc => {
                Some(BlockEndKind::ExitToInterpreter { next_rip: inst.rip })
            }
            InstKind::Ret { .. } => Some(BlockEndKind::Ret),
            InstKind::CallIndirect { .. } => Some(BlockEndKind::Call),
            InstKind::JmpIndirect { .. } => Some(BlockEndKind::Jmp),
            // With an assist available the block keeps going: the host performs the
            // instruction and the block carries on with what it leaves. Without one
            // there is nothing to do but stop here.
            InstKind::Invalid if !limits.allow_assists || !inst.assistable => {
                Some(BlockEndKind::ExitToInterpreter { next_rip: inst.rip })
            }
            InstKind::Invalid => None,
            _ => None,
        };

        insts.push(inst);

        if let Some(target) = next_rip_override.take() {
            rip = target;
            continue;
        }

        if let Some(kind) = end_kind {
            return BasicBlock {
                entry_rip,
                bitness,
                insts,
                end_kind: kind,
            };
        }
    }
}

/// Whether `target` is an instruction this block has already decoded.
///
/// The point is to refuse a jump that re-enters the block, because following one would
/// decode the same instructions again and the block would then retire more than the guest
/// executes at those addresses.
///
/// This used to be `!(entry_rip..=rip).contains(&target)`, which is only right while the
/// block is contiguous. Once *one* jump has been followed the decoded set is two disjoint
/// ranges and that test models neither: a second jump landing back in the first segment
/// passes it. Checking the instructions actually decoded is exact, and `max_insts` is 64
/// so the scan is bounded and runs only where a jump is being followed.
fn already_decoded(insts: &[DecodedInst], target: u64) -> bool {
    insts.iter().any(|i| i.rip == target)
}

/// Whether discovery follows an unconditional direct jump instead of ending the block.
///
/// **Off, and measured: 249.3 / 89.6 / 99.4 / 272.3 against 104.9 / 88.8 / 101.5 /
/// 92.3** — worse in three of four windows, two of them badly.
///
/// This is the fifth attempt in this codebase at making blocks longer, and the one
/// worth remembering, because every objection raised against the previous four does
/// not apply to it. Fusing across an unconditional direct jump is not speculative:
/// the edge is always taken, there is no side exit, and nothing is decoded that might
/// not run. It works exactly as intended — `jmp`-terminated blocks fall from 3,398 to
/// 722, executed block entries from 6.07M to 5.67M (**6.6% fewer**), guest
/// instructions per block from 4.41 to 4.79, and compiled coverage *rises* 1.1%. The
/// oracles match and the per-block verifier is clean.
///
/// And it is still slower. So "fewer block boundaries" is not the win the per-block
/// cost model suggests it should be, and the earlier superblock failures were not
/// really about side exits or lost coverage. Whatever the cost of a longer block is —
/// more emitted bytes to compile, worse locality in the host's instruction cache, a
/// block that now spans two pages — it is at least as large as the boundary it
/// removes. `AERO_X86_FOLLOW_JMP=1` turns it on.
///
/// **Re-measured 2026-08-29 with superblocks on, and it is no longer a speed question.**
/// In deep windows it is now a consistent small *win* — 66.4/67.1, 78.5/79.9, 80.8/81.6,
/// 80.2/81.0, 67.0/67.8 across five paired windows, about 1.0-1.8% — with the mechanism
/// visible: guest instructions per block 6.80 -> 7.30, in-wasm chain rate 67.9% -> 70.2%.
/// (A sixth window read 49.8 against 75.2. That was a path artifact, not a 34% win; five
/// windows agreeing at ~1% is the believable answer.)
///
/// **It used to fail the verifier, and the verifier was wrong.** `AERO_CT_VERIFY=1` gave
/// 4 mismatches with this on and 0 without, which read as a translation bug for several
/// rounds. It was not: `verify_block` bounded the *host* chain loop to one block but never
/// bounded the module's own in-wasm dispatcher, so it inherited the last real chain's
/// `chain_max_blocks` and ran hundreds. A nine-instruction block reported `retired=1302`,
/// and Tier-0 was then replayed 1302 instructions — the whole-machine comparison, which
/// legitimately diverges on interrupt timing. With the verifier bounded to one block this
/// is **0 mismatches over 30M**, as is the default.
///
/// **It is still off, and now for an honest reason.** With the verifier fixed it is clean,
/// so the correctness objection is gone — but it is also worth nothing: 50.2/50.1,
/// 49.8/49.7, 61.4/63.0, 64.6/64.7, 63.3/63.2 across six paired deep windows. The ~1-1.8%
/// it once measured was against a baseline three times slower, where a saved block boundary
/// was a larger share of the total.
///
/// It does improve the structure — guest instructions per block 6.80 -> 7.35, in-wasm chain
/// rate 67.9% -> 70.5% — and turning it on breaks a dozen tests whose fixtures assume a
/// `jmp` ends a block. Twelve test rewrites for no measured throughput is the wrong trade,
/// so `AERO_X86_FOLLOW_JMP=1` stays opt-in.
fn follow_jmp_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_X86_FOLLOW_JMP").is_some());
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::capture_panic_location;

    #[derive(Default)]
    struct DummyBus;

    impl Tier1Bus for DummyBus {
        fn read_u8(&self, _addr: u64) -> u8 {
            0
        }

        fn write_u8(&mut self, _addr: u64, _value: u8) {}
    }

    #[test]
    fn discover_block_mode_panics_at_call_site_on_invalid_bitness() {
        let bus = DummyBus;

        let expected_file = file!();
        let expected_line = line!() + 2;
        let (file, line) = capture_panic_location(|| {
            let _ = discover_block_mode(&bus, 0, BlockLimits::default(), 0);
        });
        assert_eq!(file, expected_file);
        assert_eq!(line, expected_line);
    }
}

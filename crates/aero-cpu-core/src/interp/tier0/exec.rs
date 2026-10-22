use core::mem::ManuallyDrop;

use super::{exec_decoded, ExecOutcome, Tier0Config};
use crate::assist::{handle_assist_decoded, has_addr_size_override, AssistContext};
use crate::exception::{AssistReason, Exception};
use crate::interrupts;
use crate::interrupts::CpuCore;
use crate::linear_mem::{
    fetch_wrapped_seg_ip, fetch_wrapped_seg_ip_first_page, read_bytes_wrapped,
};
use crate::mem::CpuBus;
use crate::state::{mask_bits, CpuState, FLAG_AF, FLAG_CF, FLAG_OF, FLAG_PF, FLAG_SF, FLAG_ZF};
use aero_x86::{DecodedInst, MemorySize, Mnemonic, OpKind, Register};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Global retired-instruction counter used to correlate a windowed execution
/// trace (`AERO_TRACE_FROM`/`AERO_TRACE_TO`) with the machine-level
/// `total_executed` reported on a fault. Bring-up/debug only — not bumped when
/// tracing is off (atomic RMW on every instruction was measurable).
static INSN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Current value of the global retired-instruction counter (the same index the
/// windowed trace uses). `pub(crate)` so sibling instrumentation (read
/// watchpoints in `linear_mem`) can tag log lines with a resumable index.
pub fn current_insn_index() -> u64 {
    INSN_COUNTER.load(Ordering::Relaxed)
}

/// Returns the configured `[from, to]` instruction-index trace window, if the
/// `AERO_TRACE_FROM` env var is set. Parsed once.
fn trace_window() -> Option<(u64, u64)> {
    static WINDOW: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    WINDOW
        .get_or_init(|| {
            let from = std::env::var("AERO_TRACE_FROM").ok()?.parse::<u64>().ok()?;
            let to = std::env::var("AERO_TRACE_TO")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(from + 200);
            Some((from, to))
        })
        .as_ref()
        .copied()
}

/// Whether to emit the compact (address-only) trace. Parsed once from
/// `AERO_TRACE_COMPACT`.
fn trace_compact() -> bool {
    static COMPACT: OnceLock<bool> = OnceLock::new();
    *COMPACT.get_or_init(|| std::env::var_os("AERO_TRACE_COMPACT").is_some())
}

/// Optional linear instruction-fetch watchpoints for bring-up/debug.
///
/// `AERO_WATCH_EXEC` accepts comma-separated hexadecimal or decimal linear
/// addresses. Hits include the Windows x64 argument registers so a one-shot
/// driver callback can be distinguished from merely paging the image into RAM.
fn watch_exec_addrs() -> &'static [u64] {
    static WATCH: OnceLock<Vec<u64>> = OnceLock::new();
    WATCH.get_or_init(|| {
        std::env::var("AERO_WATCH_EXEC")
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|raw| {
                        let raw = raw.trim();
                        u64::from_str_radix(raw.trim_start_matches("0x"), 16)
                            .ok()
                            .or_else(|| raw.parse::<u64>().ok())
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

#[derive(Clone, Copy)]
enum WatchExecDumpBase {
    Register(Register),
    Linear(u64),
}

struct WatchExecDump {
    label: String,
    base: WatchExecDumpBase,
    len: usize,
}

/// Optional memory dumps emitted only when `AERO_WATCH_EXEC` hits.
///
/// `AERO_WATCH_EXEC_DUMP` is a comma-separated list of `<base>:<length>`
/// entries. A base can be one of the Windows x64 argument/general registers
/// (`rax`, `rbx`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`, `r9`, or `rsp`) or a
/// linear address. Lengths are decimal unless prefixed with `0x` and are capped
/// at 256 bytes.
fn watch_exec_dumps() -> &'static [WatchExecDump] {
    static DUMPS: OnceLock<Vec<WatchExecDump>> = OnceLock::new();
    DUMPS.get_or_init(|| {
        std::env::var("AERO_WATCH_EXEC_DUMP")
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|raw| {
                        let (raw_base, raw_len) = raw.trim().rsplit_once(':')?;
                        let raw_base = raw_base.trim();
                        let base = match raw_base.to_ascii_lowercase().as_str() {
                            "rax" => WatchExecDumpBase::Register(Register::RAX),
                            "rbx" => WatchExecDumpBase::Register(Register::RBX),
                            "rcx" => WatchExecDumpBase::Register(Register::RCX),
                            "rdx" => WatchExecDumpBase::Register(Register::RDX),
                            "rsi" => WatchExecDumpBase::Register(Register::RSI),
                            "rdi" => WatchExecDumpBase::Register(Register::RDI),
                            "r8" => WatchExecDumpBase::Register(Register::R8),
                            "r9" => WatchExecDumpBase::Register(Register::R9),
                            "r10" => WatchExecDumpBase::Register(Register::R10),
                            "r11" => WatchExecDumpBase::Register(Register::R11),
                            "rsp" => WatchExecDumpBase::Register(Register::RSP),
                            _ => WatchExecDumpBase::Linear(
                                u64::from_str_radix(raw_base.trim_start_matches("0x"), 16)
                                    .ok()
                                    .or_else(|| raw_base.parse::<u64>().ok())?,
                            ),
                        };
                        let raw_len = raw_len.trim();
                        let len = raw_len
                            .strip_prefix("0x")
                            .and_then(|v| usize::from_str_radix(v, 16).ok())
                            .or_else(|| raw_len.parse::<usize>().ok())?
                            .min(256);
                        (len != 0).then(|| WatchExecDump {
                            label: raw_base.to_owned(),
                            base,
                            len,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

#[inline]
fn watch_exec<B: CpuBus>(state: &CpuState, bus: &mut B, linear: u64) {
    let watches = watch_exec_addrs();
    if watches.is_empty() || !watches.contains(&linear) {
        return;
    }

    static HITS: AtomicU64 = AtomicU64::new(0);
    let hit = HITS.fetch_add(1, Ordering::Relaxed) + 1;
    if hit <= 64 {
        eprintln!(
            "AERO_WATCH_EXEC[{hit}]: linear={linear:#x} rip={:#x} cr3={:#x} gs_base={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} r8={:#x} r9={:#x} r10={:#x} r11={:#x} rsp={:#x}",
            state.rip(),
            state.control.cr3,
            state.msr.gs_base,
            state.read_reg(Register::RAX),
            state.read_reg(Register::RBX),
            state.read_reg(Register::RCX),
            state.read_reg(Register::RDX),
            state.read_reg(Register::RSI),
            state.read_reg(Register::RDI),
            state.read_reg(Register::R8),
            state.read_reg(Register::R9),
            state.read_reg(Register::R10),
            state.read_reg(Register::R11),
            state.read_reg(Register::RSP),
        );
        for dump in watch_exec_dumps() {
            let addr = match dump.base {
                WatchExecDumpBase::Register(reg) => state.read_reg(reg),
                WatchExecDumpBase::Linear(addr) => addr,
            };
            let mut bytes = vec![0u8; dump.len];
            match read_bytes_wrapped(state, bus, addr, &mut bytes) {
                Ok(()) => {
                    let hex = bytes
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "AERO_WATCH_EXEC[{hit}] dump[{}]: linear={addr:#x} len={:#x} bytes={hex}",
                        dump.label, dump.len
                    );
                }
                Err(err) => eprintln!(
                    "AERO_WATCH_EXEC[{hit}] dump[{}]: linear={addr:#x} len={:#x} error={err:?}",
                    dump.label, dump.len
                ),
            }
        }
    }
}

/// Direct-mapped decode cache. iced `Decoder::with_ip` is ~10–15% of host cycles
/// on the kernel path when constructed per instruction; caching the decoded form
/// keyed by (ip, bitness, raw bytes) removes that for hot loops. RIP-relative
/// operands bake `ip` into the iced `Instruction`, so the tag must be the exact
/// decode IP (not just a byte hash).
///
/// Sized so a Windows working set of hot code pages mostly avoids conflict
/// misses, since each miss costs a full iced decode. 4 K and 64 K were both
/// measured against this: 64 K retires fewer host instructions but costs more
/// cycles, because the slot array stops fitting in L2.
const DECODE_CACHE_SIZE: usize = 1 << 14;

#[derive(Clone)]
struct DecodeCacheSlot {
    ip: u64,
    /// The instruction's bytes packed little-endian, so validating a hit is a
    /// masked 128-bit compare instead of a variable-length `memcmp` call.
    key: u128,
    bitness: u32,
    decoded: DecodedInst,
    addr_size_override: bool,
}

struct DecodeCache {
    slots: Box<[Option<DecodeCacheSlot>]>,
}

/// Pack a decode window into the slot key. Instruction fetch zero-fills the tail
/// of its 15-byte window, so equal instructions always pack identically.
#[inline]
fn decode_key(bytes: &[u8]) -> u128 {
    let mut padded = [0u8; 16];
    // Copy a *constant* 15 bytes on the path that almost always runs. The
    // variable-length form compiles to a `memcpy` call, and at one call per
    // decoded instruction that showed up as a libc `memmove` inside the
    // interpreter's hot loop. A fixed-size copy is two loads and two stores
    // inline.
    if bytes.len() >= 15 {
        let head: &[u8; 15] = bytes[..15].try_into().expect("length checked");
        padded[..15].copy_from_slice(head);
    } else {
        let len = bytes.len();
        padded[..len].copy_from_slice(&bytes[..len]);
    }
    u128::from_le_bytes(padded)
}

/// Keep only the low `n` bytes of a packed decode window.
const fn prefix_mask(n: u32) -> u128 {
    if n >= 16 {
        u128::MAX
    } else {
        (1u128 << (n * 8)) - 1
    }
}

const PREFIX_MASK_6: u128 = prefix_mask(6);
const PREFIX_MASK_8: u128 = prefix_mask(8);

/// `prefix_mask` for every instruction length, indexed by byte count.
///
/// The obvious `(1u128 << (len * 8)) - 1` has a *runtime* shift amount, and a
/// 128-bit variable shift lowers to a `shrd` pair plus branches. That ran on every
/// decode-cache probe — once per instruction — and a single `shrd` was over 6% of
/// the interpreter's hot loop. Indexing a table is one load.
const PREFIX_MASKS: [u128; 16] = {
    let mut masks = [0u128; 16];
    let mut i = 0usize;
    while i < 16 {
        masks[i] = prefix_mask(i as u32);
        i += 1;
    }
    masks
};

impl DecodeCache {
    fn new() -> Self {
        let mut slots = Vec::with_capacity(DECODE_CACHE_SIZE);
        slots.resize_with(DECODE_CACHE_SIZE, || None);
        Self {
            slots: slots.into_boxed_slice(),
        }
    }

    /// Fold in bits above the low 12 so two hot code pages do not alias every
    /// instruction against each other.
    #[inline]
    fn index(ip: u64) -> usize {
        ((ip ^ (ip >> 13)) as usize) & (DECODE_CACHE_SIZE - 1)
    }

    #[inline]
    fn slot_matches(
        slot: &DecodeCacheSlot,
        ip: u64,
        bitness: u32,
        key: u128,
        avail: usize,
    ) -> bool {
        let len = usize::from(slot.decoded.len);
        if slot.ip != ip || slot.bitness != bitness || len > avail {
            return false;
        }
        // Only the instruction's own bytes are part of its identity; anything the
        // fetch window happened to carry past it is not.
        let mask = if len >= 16 {
            u128::MAX
        } else {
            PREFIX_MASKS[len]
        };
        (slot.key ^ key) & mask == 0
    }

    /// `key` must be [`decode_key`] of `bytes`; the batch loop already needs it for
    /// its fuse gates, so it is passed in rather than recomputed here.
    fn decode_ref(
        &mut self,
        ip: u64,
        bitness: u32,
        bytes: &[u8],
        key: u128,
    ) -> Result<(&DecodedInst, bool), aero_x86::DecodeError> {
        debug_assert_eq!(key, decode_key(bytes));
        let idx = Self::index(ip);
        let hit = self.slots[idx]
            .as_ref()
            .is_some_and(|slot| Self::slot_matches(slot, ip, bitness, key, bytes.len()));

        if !hit {
            let mut padded = [0u8; 15];
            let len = bytes.len().min(padded.len());
            padded[..len].copy_from_slice(&bytes[..len]);
            let addr_size_override = has_addr_size_override(&padded, bitness);
            let decoded = aero_x86::decode(bytes, ip, bitness)?;
            self.slots[idx] = Some(DecodeCacheSlot {
                ip,
                key,
                bitness,
                decoded,
                addr_size_override,
            });
        }

        let slot = self.slots[idx]
            .as_ref()
            .expect("decode cache slot just stored or matched");
        Ok((&slot.decoded, slot.addr_size_override))
    }
}

thread_local! {
    /// Deliberately never dropped.
    ///
    /// A `thread_local!` whose value implements `Drop` makes std register a destructor for it, and
    /// that registration allocates through [`std::alloc::System`] rather than the crate's
    /// `#[global_allocator]`. In the browser runtime that distinction matters: `aero-wasm`
    /// installs an allocator bounded to a reserved low region so guest RAM cannot be corrupted,
    /// and the linear memory is created non-growable so guest RAM keeps a fixed home. `System` on
    /// wasm32 can only obtain memory by growing, so the registration fails — and an allocation
    /// failure is an abort, which surfaced as a bare `RuntimeError: unreachable` the first time
    /// the interpreter decoded an instruction.
    ///
    /// Holding the cache in [`ManuallyDrop`] keeps it out of that path. It is allocated once and
    /// lives as long as the thread, which is what a decode cache wants anyway.
    static DECODE_CACHE: ManuallyDrop<RefCell<DecodeCache>> =
        ManuallyDrop::new(RefCell::new(DecodeCache::new()));
}

// Enforce the note above: if this type ever needs dropping again, std will register a destructor
// through `System` and the browser runtime will abort on the first decode.
const _: () = assert!(!core::mem::needs_drop::<ManuallyDrop<RefCell<DecodeCache>>>());

#[derive(Debug, Clone)]
pub enum StepExit {
    Continue,
    /// The instruction completed normally, and maskable interrupts should be
    /// inhibited for exactly one subsequent instruction (MOV SS / POP SS shadow).
    ContinueInhibitInterrupts,
    Branch,
    Halted,
    BiosInterrupt(u8),
    /// Tier-0 could decode the instruction but does not implement its semantics
    /// and wants the caller to emulate it via the assist layer.
    ///
    /// The faulting instruction has already been fetched + decoded, so callers
    /// can avoid a second fetch/decode pass by using [`crate::assist::handle_assist_decoded`]
    /// (or [`crate::interrupts::exec_interrupt_assist_decoded`] for interrupt-related
    /// instructions).
    Assist {
        reason: AssistReason,
        decoded: aero_x86::DecodedInst,
        /// Whether an address-size override prefix (0x67) was present. This is
        /// tracked separately because Tier-0 currently only uses `iced-x86`'s
        /// decoded instruction, which does not expose a simple "was 0x67 seen"
        /// query that matches our string-op/IO assist needs.
        addr_size_override: bool,
    },
}

impl PartialEq for StepExit {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Continue, Self::Continue)
            | (Self::ContinueInhibitInterrupts, Self::ContinueInhibitInterrupts)
            | (Self::Branch, Self::Branch)
            | (Self::Halted, Self::Halted) => true,
            (Self::BiosInterrupt(a), Self::BiosInterrupt(b)) => a == b,
            (Self::Assist { reason: a, .. }, Self::Assist { reason: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for StepExit {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchExit {
    Completed,
    Branch,
    Halted,
    BiosInterrupt(u8),
    Assist(AssistReason),
    Exception(Exception),
    CpuExit(interrupts::CpuExit),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchResult {
    pub executed: u64,
    pub exit: BatchExit,
}

pub fn step_with_config<B: CpuBus>(
    cfg: &Tier0Config,
    state: &mut CpuState,
    bus: &mut B,
) -> Result<StepExit, Exception> {
    step_with_config_and_decoder(cfg, state, bus, |bytes, ip, bitness| {
        aero_x86::decode(bytes, ip, bitness)
    })
}

pub fn step<B: CpuBus>(state: &mut CpuState, bus: &mut B) -> Result<StepExit, Exception> {
    let cfg = Tier0Config::default();
    step_with_config(&cfg, state, bus)
}

pub(crate) fn fetch_and_decode<B, F>(
    state: &CpuState,
    bus: &mut B,
    cs_base: u64,
    ip: u64,
    mut decode: F,
) -> Result<([u8; 15], aero_x86::DecodedInst), Exception>
where
    B: CpuBus,
    F: FnMut(&[u8], u64, u32) -> Result<aero_x86::DecodedInst, aero_x86::DecodeError>,
{
    let (mut bytes, available) = fetch_wrapped_seg_ip_first_page(state, bus, cs_base, ip, 15)?;
    let bitness = state.bitness();
    match decode(&bytes[..available], ip, bitness) {
        Ok(decoded) => Ok((bytes, decoded)),
        Err(aero_x86::DecodeError::UnexpectedEof) if available < 15 => {
            bytes = fetch_wrapped_seg_ip(state, bus, cs_base, ip, 15)?;
            let decoded = decode(&bytes, ip, bitness).map_err(|_| Exception::InvalidOpcode)?;
            Ok((bytes, decoded))
        }
        Err(_) => Err(Exception::InvalidOpcode),
    }
}

pub(crate) fn step_with_config_and_decoder<B, F>(
    cfg: &Tier0Config,
    state: &mut CpuState,
    bus: &mut B,
    mut decode: F,
) -> Result<StepExit, Exception>
where
    B: CpuBus,
    F: FnMut(&[u8], u64, u32) -> Result<aero_x86::DecodedInst, aero_x86::DecodeError>,
{
    bus.sync(state);

    let ip = state.rip();
    let cs_base = state.seg_base_reg(Register::CS);
    let (bytes, decoded) = match fetch_and_decode(state, bus, cs_base, ip, &mut decode) {
        Ok(v) => v,
        Err(e) => {
            state.apply_exception_side_effects(&e);
            return Err(e);
        }
    };
    let bitness = state.bitness();
    let addr_size_override = has_addr_size_override(&bytes, bitness);
    let next_ip = ip.wrapping_add(decoded.len as u64) & mask_bits(bitness);

    let outcome = match exec_decoded(cfg, state, bus, &decoded, next_ip, addr_size_override) {
        Ok(v) => v,
        Err(e) => {
            // x87 opcodes (D8-DF, optionally preceded by FWAIT=9B) should still obey CR0.EM/TS
            // gating even if Tier-0 doesn't implement the specific mnemonic yet.
            if matches!(e, Exception::InvalidOpcode) && is_x87_opcode(&bytes, bitness) {
                if let Err(fp_e) = super::check_fp_available(state, crate::fpu::FpKind::X87) {
                    state.apply_exception_side_effects(&fp_e);
                    return Err(fp_e);
                }
            }

            state.apply_exception_side_effects(&e);
            return Err(e);
        }
    };

    match outcome {
        ExecOutcome::Continue => {
            state.set_rip(next_ip);
            Ok(StepExit::Continue)
        }
        ExecOutcome::ContinueInhibitInterrupts => {
            state.set_rip(next_ip);
            Ok(StepExit::ContinueInhibitInterrupts)
        }
        ExecOutcome::Halt => {
            state.set_rip(next_ip);
            if let Some(vector) = state.take_pending_bios_int() {
                Ok(StepExit::BiosInterrupt(vector))
            } else {
                state.halted = true;
                Ok(StepExit::Halted)
            }
        }
        ExecOutcome::Branch => Ok(StepExit::Branch),
        ExecOutcome::Assist(reason) => Ok(StepExit::Assist {
            reason,
            decoded,
            addr_size_override,
        }),
    }
}

fn is_x87_opcode(bytes: &[u8; 15], bitness: u32) -> bool {
    // Skip legacy prefixes + REX to find the first opcode byte.
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        let is_legacy_prefix = matches!(
            b,
            0xF0 | 0xF2 | 0xF3 // lock/rep
                | 0x2E | 0x36 | 0x3E | 0x26 | 0x64 | 0x65 // segment overrides
                | 0x66 // operand-size override
                | 0x67 // address-size override
        );
        let is_rex = bitness == 64 && (0x40..=0x4F).contains(&b);
        if !(is_legacy_prefix || is_rex) {
            break;
        }
        i += 1;
    }

    if i >= bytes.len() {
        return false;
    }

    match bytes[i] {
        0xD8..=0xDF => true,
        0x9B => {
            // Many x87 "wait" forms are encoded with an FWAIT prefix (9B) followed by an x87 opcode.
            i + 1 < bytes.len() && matches!(bytes[i + 1], 0xD8..=0xDF)
        }
        _ => false,
    }
}

pub fn run_batch_with_config<B: CpuBus>(
    cfg: &Tier0Config,
    state: &mut CpuState,
    bus: &mut B,
    max_insts: u64,
) -> BatchResult {
    if state.halted {
        return BatchResult {
            executed: 0,
            exit: BatchExit::Halted,
        };
    }

    let mut executed = 0u64;
    while executed < max_insts {
        match step_with_config(cfg, state, bus) {
            Ok(StepExit::Continue) => executed += 1,
            Ok(StepExit::ContinueInhibitInterrupts) => executed += 1,
            Ok(StepExit::Branch) => {
                executed += 1;
                return BatchResult {
                    executed,
                    exit: BatchExit::Branch,
                };
            }
            Ok(StepExit::Halted) => {
                executed += 1;
                return BatchResult {
                    executed,
                    exit: BatchExit::Halted,
                };
            }
            Ok(StepExit::BiosInterrupt(vector)) => {
                executed += 1;
                return BatchResult {
                    executed,
                    exit: BatchExit::BiosInterrupt(vector),
                };
            }
            Ok(StepExit::Assist { reason, .. }) => {
                return BatchResult {
                    executed,
                    exit: BatchExit::Assist(reason),
                };
            }
            Err(e) => {
                return BatchResult {
                    executed,
                    exit: BatchExit::Exception(e),
                };
            }
        }
    }

    BatchResult {
        executed,
        exit: BatchExit::Completed,
    }
}

pub fn run_batch<B: CpuBus>(state: &mut CpuState, bus: &mut B, max_insts: u64) -> BatchResult {
    let cfg = Tier0Config::default();
    run_batch_with_config(&cfg, state, bus, max_insts)
}

/// Tier-0 batch execution wrapper that resolves [`StepExit::Assist`] exits via
/// the [`crate::assist`] module.
///
/// This keeps the core Tier-0 interpreter minimal while still allowing it to
/// execute privileged/IO/time instructions required by OS boot code.
///
/// Note: this helper intentionally does not execute interrupt-related assists
/// (`CLI`/`STI`/`INT*`/`IRET*`). Those instructions require the architectural
/// interrupt engine in [`crate::interrupts`] (including interrupt shadow and
/// IRET bookkeeping). When an interrupt assist is encountered, this helper
/// returns [`BatchExit::Assist`] with [`AssistReason::Interrupt`].
///
/// This helper will still deliver any already-queued events in
/// [`crate::interrupts::PendingEventState`] (pending exceptions + external
/// interrupt FIFO) at instruction boundaries, including waking the CPU from
/// `HLT` when a maskable interrupt is delivered.
///
/// If event delivery fails (e.g. triple fault), this helper returns
/// [`BatchExit::CpuExit`].
///
/// Use [`run_batch_cpu_core_with_assists`] if you need Tier-0 to resolve
/// interrupt assists.
pub fn run_batch_with_assists<B: CpuBus>(
    ctx: &mut AssistContext,
    cpu: &mut CpuCore,
    bus: &mut B,
    max_insts: u64,
) -> BatchResult {
    let cfg = Tier0Config::from_cpuid(&ctx.features);
    run_batch_with_assists_with_config(&cfg, ctx, cpu, bus, max_insts)
}

pub fn run_batch_with_assists_with_config<B: CpuBus>(
    cfg: &Tier0Config,
    ctx: &mut AssistContext,
    cpu: &mut CpuCore,
    bus: &mut B,
    max_insts: u64,
) -> BatchResult {
    if max_insts == 0 {
        return BatchResult {
            executed: 0,
            exit: if cpu.state.halted {
                BatchExit::Halted
            } else {
                BatchExit::Completed
            },
        };
    }

    let mut executed = 0u64;
    while executed < max_insts {
        // Give pending exceptions/interrupts a chance at instruction boundaries.
        if cpu.pending.has_pending_event() {
            match cpu.deliver_pending_event(bus) {
                Ok(()) => continue,
                Err(exit) => {
                    return BatchResult {
                        executed,
                        exit: BatchExit::CpuExit(exit),
                    };
                }
            }
        }
        if !cpu.pending.external_interrupts().is_empty() {
            let before = cpu.pending.external_interrupts().len();
            match cpu.deliver_external_interrupt(bus) {
                Ok(()) => {
                    if cpu.pending.external_interrupts().len() != before {
                        continue;
                    }
                }
                Err(exit) => {
                    return BatchResult {
                        executed,
                        exit: BatchExit::CpuExit(exit),
                    };
                }
            }
        }
        if cpu.state.halted {
            return BatchResult {
                executed,
                exit: BatchExit::Halted,
            };
        }

        bus.sync(&cpu.state);

        let ip = cpu.state.rip();
        let cs_base = cpu.state.seg_base_reg(Register::CS);
        let (bytes, decoded) =
            match fetch_and_decode(&cpu.state, bus, cs_base, ip, aero_x86::decode) {
                Ok(result) => result,
                Err(e) => {
                    cpu.state.apply_exception_side_effects(&e);
                    return BatchResult {
                        executed,
                        exit: BatchExit::Exception(e),
                    };
                }
            };

        let addr_size_override = has_addr_size_override(&bytes, cpu.state.bitness());
        let next_ip_raw = ip.wrapping_add(decoded.len as u64);
        let next_ip = next_ip_raw & mask_bits(cpu.state.bitness());

        let outcome = match exec_decoded(
            cfg,
            &mut cpu.state,
            bus,
            &decoded,
            next_ip,
            addr_size_override,
        ) {
            Ok(v) => v,
            Err(e) => {
                if matches!(e, Exception::InvalidOpcode)
                    && is_x87_opcode(&bytes, cpu.state.bitness())
                {
                    if let Err(fp_e) =
                        super::check_fp_available(&cpu.state, crate::fpu::FpKind::X87)
                    {
                        cpu.state.apply_exception_side_effects(&fp_e);
                        return BatchResult {
                            executed,
                            exit: BatchExit::Exception(fp_e),
                        };
                    }
                }
                cpu.state.apply_exception_side_effects(&e);
                return BatchResult {
                    executed,
                    exit: BatchExit::Exception(e),
                };
            }
        };

        match outcome {
            ExecOutcome::Continue => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
            }
            ExecOutcome::ContinueInhibitInterrupts => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                cpu.pending.inhibit_interrupts_for_one_instruction();
            }
            ExecOutcome::Branch => {
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                return BatchResult {
                    executed,
                    exit: BatchExit::Branch,
                };
            }
            ExecOutcome::Halt => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                if let Some(vector) = cpu.state.take_pending_bios_int() {
                    return BatchResult {
                        executed,
                        exit: BatchExit::BiosInterrupt(vector),
                    };
                }
                cpu.state.halted = true;
                return BatchResult {
                    executed,
                    exit: BatchExit::Halted,
                };
            }
            ExecOutcome::Assist(reason) => {
                // Interrupt-related assists (CLI/STI/INT*/IRET*) require access to
                // `interrupts::PendingEventState`, which is intentionally not part
                // of the public `CpuState` ABI.
                if reason == AssistReason::Interrupt {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Assist(reason),
                    };
                }

                // Execute the instruction via the assist layer using the already decoded form.
                let inhibits_interrupt = matches!(
                    decoded.instr.mnemonic(),
                    aero_x86::Mnemonic::Mov | aero_x86::Mnemonic::Pop
                ) && decoded.instr.op_count() > 0
                    && decoded.instr.op_kind(0) == aero_x86::OpKind::Register
                    && decoded.instr.op0_register() == aero_x86::Register::SS;
                if let Err(e) = handle_assist_decoded(
                    ctx,
                    &mut cpu.time,
                    &mut cpu.state,
                    bus,
                    &decoded,
                    addr_size_override,
                ) {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Exception(e),
                    };
                }
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                if inhibits_interrupt {
                    cpu.pending.inhibit_interrupts_for_one_instruction();
                }

                if bus.take_execution_yield_request() {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }

                // Preserve the "basic block" behavior of `run_batch`: treat any
                // control-transfer assist (i.e. RIP != fallthrough) as a branch.
                let expected_next = next_ip_raw & mask_bits(cpu.state.bitness());
                if cpu.state.rip() != expected_next {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }

                // A tight RDTSC polling loop can otherwise keep HAL clock-sync work inside one
                // batch before RTC/PIC can raise IRQ8. Yield only after the assist recognizes
                // sustained polling; RDTSC itself remains a pure time sample.
                if matches!(
                    decoded.instr.mnemonic(),
                    aero_x86::Mnemonic::Rdtsc | aero_x86::Mnemonic::Rdtscp
                ) && crate::assist::rdtsc_assist_yields_for_timers(ctx)
                {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }
            }
        }
    }

    BatchResult {
        executed,
        exit: BatchExit::Completed,
    }
}

/// Tier-0 batch execution wrapper that resolves assists with access to
/// [`interrupts::CpuCore`] (architectural state + interrupt bookkeeping).
///
/// This is the canonical Tier-0 runner for tests/embeddings that need correct
/// interrupt semantics (`CLI`/`STI`/`INT*`/`IRET*`), including interrupt shadows
/// and IRET frame bookkeeping.
///
/// Like [`crate::exec::Vcpu::maybe_deliver_interrupt`], this helper also gives
/// any already-queued exceptions/interrupts in [`interrupts::PendingEventState`]
/// a chance at instruction boundaries.
///
/// If event delivery fails (e.g. triple fault), this helper returns
/// [`BatchExit::CpuExit`].
/// Deliver a Tier-0 exception through the guest IDT when it is architectural
/// (#PF, #GP, …). Non-architectural failures (`MemoryFault`, unimplemented)
/// still abort the batch as [`BatchExit::Exception`] / [`BatchExit::CpuExit`].
///
/// On successful delivery returns `Ok(())` so the batch can continue at the
/// handler. Windows demand-paging depends on this for ordinary #PF.
fn deliver_or_abort_exception<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    faulting_rip: u64,
    e: Exception,
) -> Result<(), BatchExit> {
    use crate::exceptions::Exception as ArchException;

    let (arch, error_code, cr2) = match e {
        Exception::PageFault { addr, error_code } => {
            (ArchException::PageFault, Some(error_code), Some(addr))
        }
        Exception::GeneralProtection(code) => {
            (ArchException::GeneralProtection, Some(code as u32), None)
        }
        Exception::DivideError => (ArchException::DivideError, None, None),
        Exception::SegmentNotPresent(code) => {
            (ArchException::SegmentNotPresent, Some(code as u32), None)
        }
        Exception::StackSegment(code) => (ArchException::StackFault, Some(code as u32), None),
        Exception::InvalidTss(code) => (ArchException::InvalidTss, Some(code as u32), None),
        Exception::InvalidOpcode => (ArchException::InvalidOpcode, None, None),
        Exception::DeviceNotAvailable => (ArchException::DeviceNotAvailable, None, None),
        Exception::X87Fpu => (ArchException::X87Fpu, None, None),
        Exception::SimdFloatingPointException => (ArchException::SimdFloatingPoint, None, None),
        Exception::MemoryFault => {
            cpu.state.apply_exception_side_effects(&e);
            return Err(BatchExit::CpuExit(interrupts::CpuExit::MemoryFault));
        }
        Exception::Unimplemented(name) => {
            cpu.state.apply_exception_side_effects(&e);
            return Err(BatchExit::CpuExit(
                interrupts::CpuExit::UnimplementedInstruction(name),
            ));
        }
    };

    // Bring-up probe: sample #PF deliveries (error_code + CR2 + RIP).
    // Gated by AERO_LOG_PF. Use to diagnose soft-fault RSVD livelocks and
    // session-space prototype-PTE handling (P=0 vs P=1|RSVD).
    // User demand-paging floods a small cap; kernel-space / raised-IRQL faults
    // are rare and are the ones that precede HIGH_LEVEL parks, so they are
    // logged separately (still capped).
    if matches!(e, Exception::PageFault { .. }) && std::env::var_os("AERO_LOG_PF").is_some() {
        static PF_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static PF_K_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if let Exception::PageFault { addr, error_code } = e {
            let p = error_code & 1;
            let wr = (error_code >> 1) & 1;
            let us = (error_code >> 2) & 1;
            let rsvd = (error_code >> 3) & 1;
            let id = (error_code >> 4) & 1;
            let cr8 = cpu.state.control.cr8;
            let kernel = addr >= 0xffff_8000_0000_0000 || cr8 != 0;
            if kernel {
                let n = PF_K_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 64 {
                    eprintln!(
                        "AERO_LOG_PF_K: n={n} rip={faulting_rip:#x} cr2={addr:#x} err={error_code:#x} P={p} W/R={wr} U/S={us} RSVD={rsvd} I/D={id} cr8={cr8:#x} cr3={:#x} rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rsp={:#x}",
                        cpu.state.control.cr3,
                        cpu.state.read_reg(Register::RAX),
                        cpu.state.read_reg(Register::RBX),
                        cpu.state.read_reg(Register::RCX),
                        cpu.state.read_reg(Register::RDX),
                        cpu.state.read_reg(Register::RSI),
                        cpu.state.read_reg(Register::RDI),
                        cpu.state.read_reg(Register::RSP),
                    );
                }
            } else {
                let n = PF_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 128 {
                    eprintln!(
                        "AERO_LOG_PF: n={n} rip={faulting_rip:#x} cr2={addr:#x} err={error_code:#x} P={p} W/R={wr} U/S={us} RSVD={rsvd} I/D={id} cr8={cr8:#x}"
                    );
                }
            }
        }
    }

    // Bring-up probe: log only #UD (not demand-paging #PF, which floods the cap).
    // Gated by AERO_LOG_UD. Proves true host #UD vs guest-raised ILLEGAL_INSTRUCTION.
    if matches!(e, Exception::InvalidOpcode) && std::env::var_os("AERO_LOG_UD").is_some() {
        static UD_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = UD_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 64 {
            let mut bytes = [0u8; 15];
            let linear = cpu
                .state
                .seg_base_reg(aero_x86::Register::CS)
                .wrapping_add(faulting_rip);
            let _ = bus.read_bytes(linear, &mut bytes);
            let hex: String = bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            let cs = cpu.state.segments.cs.selector;
            let cpl = cpu.state.cpl();
            eprintln!("AERO_LOG_UD[{n}]: rip={faulting_rip:#x} cs={cs:#x} cpl={cpl} bytes={hex}");
        }
    }

    // Bring-up probe: user-mode #GP (movaps misalign, etc.). Kernel #GP is too
    // noisy for a first look. Gated by AERO_LOG_GP.
    if matches!(e, Exception::GeneralProtection(_)) && std::env::var_os("AERO_LOG_GP").is_some() {
        static GP_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let cpl = cpu.state.cpl();
        if cpl == 3 {
            let n = GP_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 64 {
                let mut bytes = [0u8; 15];
                let linear = cpu
                    .state
                    .seg_base_reg(aero_x86::Register::CS)
                    .wrapping_add(faulting_rip);
                let _ = bus.read_bytes(linear, &mut bytes);
                let hex: String = bytes
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                let cs = cpu.state.segments.cs.selector;
                let code = match e {
                    Exception::GeneralProtection(c) => c,
                    _ => 0,
                };
                eprintln!(
                    "AERO_LOG_GP[{n}]: rip={faulting_rip:#x} cs={cs:#x} cpl={cpl} err={code:#x} bytes={hex}"
                );
            }
        }
    }

    cpu.pending
        .raise_exception_fault(&mut cpu.state, arch, faulting_rip, error_code, cr2);
    match cpu.deliver_pending_event(bus) {
        Ok(()) => Ok(()),
        Err(exit) => Err(BatchExit::CpuExit(exit)),
    }
}

/// WinSetup apply-scan (`inc r64; inc dword [r64+disp]; cmp byte [r64], imm8; jnz`).
///
/// That 4-instruction loop is ~77% of WIM-expand samples. Decoding and
/// dispatching it every iteration is the expand wall; the fused path keeps
/// the same faults, flags, and RIP as stepping the four x86 insns.
struct ByteScanLoop {
    loop_head: u64,
    scan_reg: Register,
    counter_base: Register,
    counter_disp: u64,
    needle: u8,
    inc_mem_rip: u64,
    cmp_rip: u64,
    fallthrough: u64,
}

fn match_byte_scan_loop(
    ip: u64,
    bitness: u32,
    bytes: &[u8],
    first: &DecodedInst,
) -> Option<ByteScanLoop> {
    if bitness != 64 {
        return None;
    }
    let instr = &first.instr;
    if instr.mnemonic() != Mnemonic::Inc
        || instr.has_lock_prefix()
        || instr.op_kind(0) != OpKind::Register
        || super::ops_data::reg_bits(instr.op0_register()) != Ok(64)
    {
        return None;
    }
    let scan_reg = instr.op0_register();
    let mut off = first.len as usize;
    if off >= bytes.len() {
        return None;
    }

    let inc_mem_rip = ip.wrapping_add(first.len as u64);
    let inc_mem = aero_x86::decode(&bytes[off..], inc_mem_rip, bitness).ok()?;
    if inc_mem.instr.mnemonic() != Mnemonic::Inc
        || inc_mem.instr.has_lock_prefix()
        || inc_mem.instr.op_kind(0) != OpKind::Memory
        || inc_mem.instr.memory_index() != Register::None
        || inc_mem.instr.memory_base() == Register::None
        || inc_mem.instr.memory_base() == Register::RIP
        || inc_mem.instr.memory_size() != MemorySize::UInt32
    {
        return None;
    }
    let counter_base = inc_mem.instr.memory_base();
    let counter_disp = inc_mem.instr.memory_displacement64();
    off += inc_mem.len as usize;
    if off >= bytes.len() {
        return None;
    }

    let cmp_rip = inc_mem_rip.wrapping_add(inc_mem.len as u64);
    let cmp = aero_x86::decode(&bytes[off..], cmp_rip, bitness).ok()?;
    if cmp.instr.mnemonic() != Mnemonic::Cmp
        || cmp.instr.op_kind(0) != OpKind::Memory
        || cmp.instr.op_kind(1) != OpKind::Immediate8
        || cmp.instr.memory_base() != scan_reg
        || cmp.instr.memory_index() != Register::None
        || cmp.instr.memory_displacement64() != 0
        || cmp.instr.memory_size() != MemorySize::UInt8
    {
        return None;
    }
    let needle = cmp.instr.immediate8();
    off += cmp.len as usize;
    if off >= bytes.len() {
        return None;
    }

    let jcc_rip = cmp_rip.wrapping_add(cmp.len as u64);
    let jcc = aero_x86::decode(&bytes[off..], jcc_rip, bitness).ok()?;
    if jcc.instr.mnemonic() != Mnemonic::Jne || jcc.instr.near_branch_target() != ip {
        return None;
    }
    Some(ByteScanLoop {
        loop_head: ip,
        scan_reg,
        counter_base,
        counter_disp,
        needle,
        inc_mem_rip,
        cmp_rip,
        fallthrough: jcc_rip.wrapping_add(jcc.len as u64),
    })
}

fn exec_byte_scan_loop<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    spec: &ByteScanLoop,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    let mut retired = 0u64;
    while remaining.saturating_sub(retired) >= 4 {
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            break;
        }
        if cpu.state.halted {
            break;
        }

        let cf = cpu.state.get_flag(FLAG_CF);
        let old_ptr = cpu.state.read_reg(spec.scan_reg);
        let ptr = old_ptr.wrapping_add(1);
        let (_inc, inc_flags) = super::ops_alu::add_with_flags(&cpu.state, old_ptr, 1, 0, 64);
        cpu.state
            .set_rflags((inc_flags & !FLAG_CF) | (cf as u64 * FLAG_CF));
        cpu.state.write_reg(spec.scan_reg, ptr);

        let counter_addr = cpu.state.apply_a20(
            cpu.state
                .read_reg(spec.counter_base)
                .wrapping_add(spec.counter_disp),
        );
        let old = match bus.atomic_rmw::<u32, _>(counter_addr, |old| (old.wrapping_add(1), old)) {
            Ok(old) => old,
            Err(e) => {
                cpu.state.set_rip(spec.inc_mem_rip);
                return Err((spec.inc_mem_rip, e));
            }
        };
        let cf = cpu.state.get_flag(FLAG_CF);
        let (_inc, inc_flags) = super::ops_alu::add_with_flags(&cpu.state, old as u64, 1, 0, 32);
        cpu.state
            .set_rflags((inc_flags & !FLAG_CF) | (cf as u64 * FLAG_CF));

        let byte = match bus.read_u8(cpu.state.apply_a20(ptr)) {
            Ok(b) => b,
            Err(e) => {
                cpu.state.set_rip(spec.cmp_rip);
                return Err((spec.cmp_rip, e));
            }
        };
        let (_res, cmp_flags) =
            super::ops_alu::sub_with_flags(&cpu.state, byte as u64, spec.needle as u64, 0, 8);
        cpu.state.set_rflags(cmp_flags);

        cpu.pending.retire_instructions(4);
        cpu.time.advance_cycles(4);
        cpu.state.msr.tsc = cpu.time.read_tsc();
        retired += 4;

        if byte == spec.needle {
            cpu.state.set_rip(spec.fallthrough);
            break;
        }
        cpu.state.set_rip(spec.loop_head);
    }
    Ok(retired)
}

/// WinSetup LZX match-copy (`encode_verbatim_block` @ +0x1b3ac4 /
/// `encode_aligned_block` @ +0x1b4038). After the E8 fuse these two
/// counted byte-copies are the next expand wall.
enum LzxCopyLoop {
    /// `mov r8,[src]; dec c32; inc pos32; mov [dst+idx],r8; inc dst; inc src; test c32; jg`
    Verbatim {
        loop_head: u64,
        src: Register,
        count: Register,
        pos: Register,
        dst: Register,
        idx: Register,
        byte_reg: Register,
        load_rip: u64,
        store_rip: u64,
        fallthrough: u64,
        insts_per_iter: u64,
    },
    /// `mov base,[ptr]; dec c32; inc pos32; mov r8,[off+base]; inc off; mov [base+dst],r8; inc dst; test c32; jg`
    Aligned {
        loop_head: u64,
        ptr: Register,
        base: Register,
        count: Register,
        pos: Register,
        off: Register,
        dst: Register,
        byte_reg: Register,
        reload_rip: u64,
        load_rip: u64,
        store_rip: u64,
        fallthrough: u64,
        insts_per_iter: u64,
    },
}

fn decode_at(bytes: &[u8], ip: u64, bitness: u32) -> Option<(aero_x86::DecodedInst, usize)> {
    let decoded = aero_x86::decode(bytes, ip, bitness).ok()?;
    let len = decoded.len as usize;
    Some((decoded, len))
}

fn take_next(
    bytes: &[u8],
    off: &mut usize,
    ip: u64,
    bitness: u32,
) -> Option<aero_x86::DecodedInst> {
    let at = ip.wrapping_add(*off as u64);
    let (decoded, len) = decode_at(&bytes[*off..], at, bitness)?;
    *off += len;
    Some(decoded)
}

fn is_inc_reg(inst: &aero_x86::Instruction, reg: Register) -> bool {
    inst.mnemonic() == Mnemonic::Inc
        && !inst.has_lock_prefix()
        && inst.op_kind(0) == OpKind::Register
        && inst.op0_register() == reg
}

fn is_dec_reg(inst: &aero_x86::Instruction, reg: Register) -> bool {
    inst.mnemonic() == Mnemonic::Dec
        && !inst.has_lock_prefix()
        && inst.op_kind(0) == OpKind::Register
        && inst.op0_register() == reg
}

fn is_test_reg_self(inst: &aero_x86::Instruction, reg: Register) -> bool {
    inst.mnemonic() == Mnemonic::Test
        && inst.op_kind(0) == OpKind::Register
        && inst.op_kind(1) == OpKind::Register
        && inst.op0_register() == reg
        && inst.op1_register() == reg
}

fn collect_code_window<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    already: &[u8],
    want: usize,
) -> Option<Vec<u8>> {
    let mut out = already.to_vec();
    if out.len() >= want {
        out.truncate(want);
        return Some(out);
    }
    let linear = cpu
        .state
        .apply_a20(cpu.state.seg_base_reg(Register::CS).wrapping_add(ip));
    while out.len() < want {
        let addr = linear.wrapping_add(out.len() as u64);
        let chunk = bus.fetch(addr, (want - out.len()).min(15)).ok()?;
        let take = (want - out.len()).min(15);
        // `fetch` zero-fills the tail; only the requested prefix is valid.
        let before = out.len();
        out.extend_from_slice(&chunk[..take]);
        if out.len() == before {
            return None;
        }
    }
    Some(out)
}

fn match_lzx_copy_loop<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    bitness: u32,
    already: &[u8],
    first: &DecodedInst,
) -> Option<LzxCopyLoop> {
    if bitness != 64 {
        return None;
    }
    let instr = &first.instr;
    if instr.mnemonic() != Mnemonic::Mov || instr.has_lock_prefix() {
        return None;
    }
    let looks_like_copy_head = instr.op_kind(0) == OpKind::Register
        && instr.op_kind(1) == OpKind::Memory
        && instr.memory_index() == Register::None
        && instr.memory_base() != Register::None
        && instr.memory_base() != Register::RIP
        && instr.memory_displacement64() == 0
        && (instr.memory_size() == MemorySize::UInt8 || instr.memory_size() == MemorySize::UInt64);
    if !looks_like_copy_head {
        return None;
    }
    let bytes = collect_code_window(cpu, bus, ip, already, 32)?;
    let mut off;

    // Variant A: 8-bit load from [src]
    if instr.op_kind(0) == OpKind::Register
        && instr.op_kind(1) == OpKind::Memory
        && instr.memory_index() == Register::None
        && instr.memory_base() != Register::None
        && instr.memory_base() != Register::RIP
        && instr.memory_displacement64() == 0
        && instr.memory_size() == MemorySize::UInt8
        && super::ops_data::reg_bits(instr.op0_register()) == Ok(8)
    {
        let byte_reg = instr.op0_register();
        let src = instr.memory_base();
        off = first.len as usize;
        let load_rip = ip;
        let dec = take_next(&bytes, &mut off, ip, bitness)?;
        if dec.instr.op_kind(0) != OpKind::Register
            || !is_dec_reg(&dec.instr, dec.instr.op0_register())
            || super::ops_data::reg_bits(dec.instr.op0_register()) != Ok(32)
        {
            return None;
        }
        let count = dec.instr.op0_register();
        let inc_pos = take_next(&bytes, &mut off, ip, bitness)?;
        if inc_pos.instr.op_kind(0) != OpKind::Register
            || !is_inc_reg(&inc_pos.instr, inc_pos.instr.op0_register())
            || super::ops_data::reg_bits(inc_pos.instr.op0_register()) != Ok(32)
        {
            return None;
        }
        let pos = inc_pos.instr.op0_register();
        let store_rip = ip.wrapping_add(off as u64);
        let store = take_next(&bytes, &mut off, ip, bitness)?;
        if store.instr.mnemonic() != Mnemonic::Mov
            || store.instr.op_kind(0) != OpKind::Memory
            || store.instr.op_kind(1) != OpKind::Register
            || store.instr.op1_register() != byte_reg
            || store.instr.memory_size() != MemorySize::UInt8
            || store.instr.memory_index() == Register::None
            || store.instr.memory_index_scale() != 1
            || store.instr.memory_displacement64() != 0
        {
            return None;
        }
        let dst = store.instr.memory_base();
        let idx = store.instr.memory_index();
        let inc_dst = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_inc_reg(&inc_dst.instr, dst) || super::ops_data::reg_bits(dst) != Ok(64) {
            return None;
        }
        let inc_src = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_inc_reg(&inc_src.instr, src) || super::ops_data::reg_bits(src) != Ok(64) {
            return None;
        }
        let test = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_test_reg_self(&test.instr, count) {
            return None;
        }
        let jcc_rip = ip.wrapping_add(off as u64);
        let jcc = take_next(&bytes, &mut off, ip, bitness)?;
        if jcc.instr.mnemonic() != Mnemonic::Jg || jcc.instr.near_branch_target() != ip {
            return None;
        }
        return Some(LzxCopyLoop::Verbatim {
            loop_head: ip,
            src,
            count,
            pos,
            dst,
            idx,
            byte_reg,
            load_rip,
            store_rip,
            fallthrough: jcc_rip.wrapping_add(jcc.len as u64),
            insts_per_iter: 8,
        });
    }

    // Variant B: 64-bit reload of the window base, then indexed 8-bit copy.
    if instr.op_kind(0) == OpKind::Register
        && instr.op_kind(1) == OpKind::Memory
        && instr.memory_index() == Register::None
        && instr.memory_base() != Register::None
        && instr.memory_base() != Register::RIP
        && instr.memory_displacement64() == 0
        && instr.memory_size() == MemorySize::UInt64
        && super::ops_data::reg_bits(instr.op0_register()) == Ok(64)
    {
        let base = instr.op0_register();
        let ptr = instr.memory_base();
        off = first.len as usize;
        let reload_rip = ip;
        let dec = take_next(&bytes, &mut off, ip, bitness)?;
        if dec.instr.op_kind(0) != OpKind::Register
            || !is_dec_reg(&dec.instr, dec.instr.op0_register())
            || super::ops_data::reg_bits(dec.instr.op0_register()) != Ok(32)
        {
            return None;
        }
        let count = dec.instr.op0_register();
        let inc_pos = take_next(&bytes, &mut off, ip, bitness)?;
        if inc_pos.instr.op_kind(0) != OpKind::Register
            || !is_inc_reg(&inc_pos.instr, inc_pos.instr.op0_register())
            || super::ops_data::reg_bits(inc_pos.instr.op0_register()) != Ok(32)
        {
            return None;
        }
        let pos = inc_pos.instr.op0_register();
        let load_rip = ip.wrapping_add(off as u64);
        let load = take_next(&bytes, &mut off, ip, bitness)?;
        if load.instr.mnemonic() != Mnemonic::Mov
            || load.instr.op_kind(0) != OpKind::Register
            || load.instr.op_kind(1) != OpKind::Memory
            || super::ops_data::reg_bits(load.instr.op0_register()) != Ok(8)
            || load.instr.memory_size() != MemorySize::UInt8
            || load.instr.memory_index() == Register::None
            || load.instr.memory_index_scale() != 1
            || load.instr.memory_displacement64() != 0
        {
            return None;
        }
        let byte_reg = load.instr.op0_register();
        let mb = load.instr.memory_base();
        let mi = load.instr.memory_index();
        if mb != base && mi != base {
            return None;
        }
        let off_reg = if mb == base { mi } else { mb };
        let inc_off = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_inc_reg(&inc_off.instr, off_reg) {
            return None;
        }
        let store_rip = ip.wrapping_add(off as u64);
        let store = take_next(&bytes, &mut off, ip, bitness)?;
        if store.instr.mnemonic() != Mnemonic::Mov
            || store.instr.op_kind(0) != OpKind::Memory
            || store.instr.op_kind(1) != OpKind::Register
            || store.instr.op1_register() != byte_reg
            || store.instr.memory_size() != MemorySize::UInt8
            || store.instr.memory_index() == Register::None
            || store.instr.memory_index_scale() != 1
            || store.instr.memory_displacement64() != 0
        {
            return None;
        }
        let (sbase, sidx) = (store.instr.memory_base(), store.instr.memory_index());
        let dst = if sbase == base {
            sidx
        } else if sidx == base {
            sbase
        } else {
            return None;
        };
        let inc_dst = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_inc_reg(&inc_dst.instr, dst) || super::ops_data::reg_bits(dst) != Ok(64) {
            return None;
        }
        let test = take_next(&bytes, &mut off, ip, bitness)?;
        if !is_test_reg_self(&test.instr, count) {
            return None;
        }
        let jcc_rip = ip.wrapping_add(off as u64);
        let jcc = take_next(&bytes, &mut off, ip, bitness)?;
        if jcc.instr.mnemonic() != Mnemonic::Jg || jcc.instr.near_branch_target() != ip {
            return None;
        }
        return Some(LzxCopyLoop::Aligned {
            loop_head: ip,
            ptr,
            base,
            count,
            pos,
            off: off_reg,
            dst,
            byte_reg,
            reload_rip,
            load_rip,
            store_rip,
            fallthrough: jcc_rip.wrapping_add(jcc.len as u64),
            insts_per_iter: 9,
        });
    }
    None
}

fn inc_reg_flags(cpu: &mut interrupts::CpuCore, reg: Register) {
    let bits = super::ops_data::reg_bits(reg).unwrap_or(64);
    let old = cpu.state.read_reg(reg);
    let cf = cpu.state.get_flag(FLAG_CF);
    let (res, flags) = super::ops_alu::add_with_flags(&cpu.state, old, 1, 0, bits);
    cpu.state
        .set_rflags((flags & !FLAG_CF) | (cf as u64 * FLAG_CF));
    cpu.state.write_reg(reg, res);
}

fn dec_reg_flags(cpu: &mut interrupts::CpuCore, reg: Register) {
    let bits = super::ops_data::reg_bits(reg).unwrap_or(32);
    let old = cpu.state.read_reg(reg);
    let cf = cpu.state.get_flag(FLAG_CF);
    let (res, flags) = super::ops_alu::sub_with_flags(&cpu.state, old, 1, 0, bits);
    cpu.state
        .set_rflags((flags & !FLAG_CF) | (cf as u64 * FLAG_CF));
    cpu.state.write_reg(reg, res);
}

fn test_reg_self_flags(cpu: &mut interrupts::CpuCore, reg: Register) {
    let bits = super::ops_data::reg_bits(reg).unwrap_or(32);
    let val = cpu.state.read_reg(reg) & mask_bits(bits);
    let mut flags =
        cpu.state.rflags() & !(FLAG_CF | FLAG_OF | FLAG_SF | FLAG_ZF | FLAG_PF | FLAG_AF);
    if val == 0 {
        flags |= FLAG_ZF;
    }
    if (val & (1u64 << (bits - 1))) != 0 {
        flags |= FLAG_SF;
    }
    if (val as u8).count_ones().is_multiple_of(2) {
        flags |= FLAG_PF;
    }
    cpu.state.set_rflags(flags);
}

fn exec_lzx_copy_loop<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    spec: &LzxCopyLoop,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    let per = match spec {
        LzxCopyLoop::Verbatim { insts_per_iter, .. }
        | LzxCopyLoop::Aligned { insts_per_iter, .. } => *insts_per_iter,
    };
    let mut retired = 0u64;
    while remaining.saturating_sub(retired) >= per {
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            break;
        }
        if cpu.state.halted {
            break;
        }
        match spec {
            LzxCopyLoop::Verbatim {
                loop_head,
                src,
                count,
                pos,
                dst,
                idx,
                byte_reg,
                load_rip,
                store_rip,
                fallthrough,
                insts_per_iter,
            } => {
                let src_addr = cpu.state.apply_a20(cpu.state.read_reg(*src));
                let byte = match bus.read_u8(src_addr) {
                    Ok(b) => b,
                    Err(e) => {
                        cpu.state.set_rip(*load_rip);
                        return Err((*load_rip, e));
                    }
                };
                cpu.state.write_reg(*byte_reg, byte as u64);
                dec_reg_flags(cpu, *count);
                inc_reg_flags(cpu, *pos);
                let dst_addr = cpu.state.apply_a20(
                    cpu.state
                        .read_reg(*dst)
                        .wrapping_add(cpu.state.read_reg(*idx)),
                );
                if let Err(e) = bus.write_u8(dst_addr, byte) {
                    cpu.state.set_rip(*store_rip);
                    return Err((*store_rip, e));
                }
                inc_reg_flags(cpu, *dst);
                inc_reg_flags(cpu, *src);
                test_reg_self_flags(cpu, *count);
                cpu.pending.retire_instructions(*insts_per_iter);
                cpu.time.advance_cycles(*insts_per_iter);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                retired += *insts_per_iter;
                if (cpu.state.read_reg(*count) as i32) <= 0 {
                    cpu.state.set_rip(*fallthrough);
                    break;
                }
                cpu.state.set_rip(*loop_head);
            }
            LzxCopyLoop::Aligned {
                loop_head,
                ptr,
                base,
                count,
                pos,
                off,
                dst,
                byte_reg,
                reload_rip,
                load_rip,
                store_rip,
                fallthrough,
                insts_per_iter,
            } => {
                let ptr_addr = cpu.state.apply_a20(cpu.state.read_reg(*ptr));
                let window = match bus.read_u64(ptr_addr) {
                    Ok(v) => v,
                    Err(e) => {
                        cpu.state.set_rip(*reload_rip);
                        return Err((*reload_rip, e));
                    }
                };
                cpu.state.write_reg(*base, window);
                dec_reg_flags(cpu, *count);
                inc_reg_flags(cpu, *pos);
                let load_addr = cpu
                    .state
                    .apply_a20(window.wrapping_add(cpu.state.read_reg(*off)));
                let byte = match bus.read_u8(load_addr) {
                    Ok(b) => b,
                    Err(e) => {
                        cpu.state.set_rip(*load_rip);
                        return Err((*load_rip, e));
                    }
                };
                cpu.state.write_reg(*byte_reg, byte as u64);
                inc_reg_flags(cpu, *off);
                let store_addr = cpu
                    .state
                    .apply_a20(window.wrapping_add(cpu.state.read_reg(*dst)));
                if let Err(e) = bus.write_u8(store_addr, byte) {
                    cpu.state.set_rip(*store_rip);
                    return Err((*store_rip, e));
                }
                inc_reg_flags(cpu, *dst);
                test_reg_self_flags(cpu, *count);
                cpu.pending.retire_instructions(*insts_per_iter);
                cpu.time.advance_cycles(*insts_per_iter);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                retired += *insts_per_iter;
                if (cpu.state.read_reg(*count) as i32) <= 0 {
                    cpu.state.set_rip(*fallthrough);
                    break;
                }
                cpu.state.set_rip(*loop_head);
            }
        }
    }
    Ok(retired)
}

/// WinSetup LZX Huffman walk (`encode_uncompressed_block` @ +0x1b384e):
/// `neg r32; movsxd i,r32; test bits,mask; je even; movsx r32,word [b+i*4+odd];
/// jmp join; even: movsx r32,word [b+i*4+even]; join: shr mask,1; test r32; js`.
struct HuffmanWalk {
    loop_head: u64,
    node: Register,
    idx: Register,
    bits: Register,
    mask: Register,
    table_base: Register,
    disp_even: u64,
    disp_odd: u64,
    load_even_rip: u64,
    load_odd_rip: u64,
    fallthrough: u64,
}

fn match_huffman_walk<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    bitness: u32,
    already: &[u8],
    first: &DecodedInst,
) -> Option<HuffmanWalk> {
    if bitness != 64 {
        return None;
    }
    let instr = &first.instr;
    if instr.mnemonic() != Mnemonic::Neg
        || instr.has_lock_prefix()
        || instr.op_kind(0) != OpKind::Register
        || super::ops_data::reg_bits(instr.op0_register()) != Ok(32)
    {
        return None;
    }
    let node = instr.op0_register();
    let bytes = collect_code_window(cpu, bus, ip, already, 48)?;
    let mut off = first.len as usize;

    let movsxd = take_next(&bytes, &mut off, ip, bitness)?;
    if movsxd.instr.mnemonic() != Mnemonic::Movsxd
        || movsxd.instr.op_kind(0) != OpKind::Register
        || movsxd.instr.op_kind(1) != OpKind::Register
        || super::ops_data::reg_bits(movsxd.instr.op0_register()) != Ok(64)
        || movsxd.instr.op1_register() != node
    {
        return None;
    }
    let idx = movsxd.instr.op0_register();

    let test_bits = take_next(&bytes, &mut off, ip, bitness)?;
    if test_bits.instr.mnemonic() != Mnemonic::Test
        || test_bits.instr.op_kind(0) != OpKind::Register
        || test_bits.instr.op_kind(1) != OpKind::Register
        || super::ops_data::reg_bits(test_bits.instr.op0_register()) != Ok(32)
        || super::ops_data::reg_bits(test_bits.instr.op1_register()) != Ok(32)
    {
        return None;
    }
    let bits = test_bits.instr.op0_register();
    let mask = test_bits.instr.op1_register();

    let je = take_next(&bytes, &mut off, ip, bitness)?;
    if je.instr.mnemonic() != Mnemonic::Je {
        return None;
    }
    let even_rip = je.instr.near_branch_target();

    let load_odd_rip = ip.wrapping_add(off as u64);
    let load_odd = take_next(&bytes, &mut off, ip, bitness)?;
    if load_odd.instr.mnemonic() != Mnemonic::Movsx
        || load_odd.instr.op_kind(0) != OpKind::Register
        || load_odd.instr.op0_register() != node
        || load_odd.instr.op_kind(1) != OpKind::Memory
        || load_odd.instr.memory_size() != MemorySize::Int16
        || load_odd.instr.memory_index() != idx
        || load_odd.instr.memory_index_scale() != 4
        || load_odd.instr.memory_base() == Register::None
        || load_odd.instr.memory_base() == Register::RIP
    {
        return None;
    }
    let table_base = load_odd.instr.memory_base();
    let disp_odd = load_odd.instr.memory_displacement64();

    let jmp = take_next(&bytes, &mut off, ip, bitness)?;
    if jmp.instr.mnemonic() != Mnemonic::Jmp {
        return None;
    }
    let join_rip = jmp.instr.near_branch_target();

    if ip.wrapping_add(off as u64) != even_rip {
        return None;
    }
    let load_even = take_next(&bytes, &mut off, ip, bitness)?;
    if load_even.instr.mnemonic() != Mnemonic::Movsx
        || load_even.instr.op0_register() != node
        || load_even.instr.op_kind(1) != OpKind::Memory
        || load_even.instr.memory_size() != MemorySize::Int16
        || load_even.instr.memory_index() != idx
        || load_even.instr.memory_index_scale() != 4
        || load_even.instr.memory_base() != table_base
    {
        return None;
    }
    let disp_even = load_even.instr.memory_displacement64();
    if ip.wrapping_add(off as u64) != join_rip {
        return None;
    }

    let shr = take_next(&bytes, &mut off, ip, bitness)?;
    if shr.instr.mnemonic() != Mnemonic::Shr
        || shr.instr.op_kind(0) != OpKind::Register
        || shr.instr.op0_register() != mask
        || shr.instr.op_kind(1) != OpKind::Immediate8
        || shr.instr.immediate8() != 1
    {
        return None;
    }

    let test_node = take_next(&bytes, &mut off, ip, bitness)?;
    if !is_test_reg_self(&test_node.instr, node) {
        return None;
    }

    let jcc_rip = ip.wrapping_add(off as u64);
    let jcc = take_next(&bytes, &mut off, ip, bitness)?;
    if jcc.instr.mnemonic() != Mnemonic::Js || jcc.instr.near_branch_target() != ip {
        return None;
    }

    Some(HuffmanWalk {
        loop_head: ip,
        node,
        idx,
        bits,
        mask,
        table_base,
        disp_even,
        disp_odd,
        load_even_rip: even_rip,
        load_odd_rip,
        fallthrough: jcc_rip.wrapping_add(jcc.len as u64),
    })
}

fn exec_huffman_walk<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    spec: &HuffmanWalk,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    let mut retired = 0u64;
    while remaining.saturating_sub(retired) >= 8 {
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            break;
        }
        if cpu.state.halted {
            break;
        }

        let old = cpu.state.read_reg(spec.node) as u32;
        let neg = old.wrapping_neg();
        let idx = neg as i32 as i64 as u64;
        let bits = cpu.state.read_reg(spec.bits) as u32;
        let mask = cpu.state.read_reg(spec.mask) as u32;
        let odd = (bits & mask) != 0;
        let (disp, load_rip, insts) = if odd {
            (spec.disp_odd, spec.load_odd_rip, 9u64)
        } else {
            (spec.disp_even, spec.load_even_rip, 8u64)
        };
        if remaining.saturating_sub(retired) < insts {
            break;
        }
        cpu.state.write_reg(spec.node, neg as u64);
        cpu.state.write_reg(spec.idx, idx);
        let addr = cpu.state.apply_a20(
            cpu.state
                .read_reg(spec.table_base)
                .wrapping_add(idx.wrapping_mul(4))
                .wrapping_add(disp),
        );
        let word = match bus.read_u16(addr) {
            Ok(w) => w,
            Err(e) => {
                cpu.state.set_rip(load_rip);
                return Err((load_rip, e));
            }
        };
        let leaf = word as i16 as i32 as u32;
        cpu.state.write_reg(spec.node, leaf as u64);
        cpu.state.write_reg(spec.mask, (mask >> 1) as u64);
        test_reg_self_flags(cpu, spec.node);

        cpu.pending.retire_instructions(insts);
        cpu.time.advance_cycles(insts);
        cpu.state.msr.tsc = cpu.time.read_tsc();
        retired += insts;

        if (leaf as i32) >= 0 {
            cpu.state.set_rip(spec.fallthrough);
            break;
        }
        cpu.state.set_rip(spec.loop_head);
    }
    Ok(retired)
}

/// WinSetup LZX bitstream refill (`encode_verbatim_block` @ +0x1b3891 /
/// `encode_aligned_block` @ +0x1b3ea7):
/// `movzx hi,byte [src+1]; movzx lo,byte [src]; movsx sh,bit8; shl hi,8;
///  neg sh; add src,2; or hi,lo; shl hi,cl; or bits,hi; add bit8,0x10`.
///
/// After the E8 / Huffman / match-copy fuses this 10-instruction refill is
/// the next expand tax (two copies, ~12% of early-expand samples).
struct LzxBitRefill {
    src: Register,
    hi: Register,
    lo: Register,
    shift: Register,
    bits: Register,
    bitcount: Register,
    load_hi_rip: u64,
    load_lo_rip: u64,
    fallthrough: u64,
    insts: u64,
}

fn is_movzx_byte_mem(
    inst: &aero_x86::Instruction,
    dest: Option<Register>,
    base: Option<Register>,
    disp: u64,
) -> bool {
    inst.mnemonic() == Mnemonic::Movzx
        && !inst.has_lock_prefix()
        && inst.op_kind(0) == OpKind::Register
        && inst.op_kind(1) == OpKind::Memory
        && super::ops_data::reg_bits(inst.op0_register()) == Ok(32)
        && inst.memory_size() == MemorySize::UInt8
        && inst.memory_index() == Register::None
        && inst.memory_base() != Register::None
        && inst.memory_base() != Register::RIP
        && inst.memory_displacement64() == disp
        && dest.map(|r| inst.op0_register() == r).unwrap_or(true)
        && base.map(|r| inst.memory_base() == r).unwrap_or(true)
}

fn match_lzx_bit_refill<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    bitness: u32,
    already: &[u8],
    first: &DecodedInst,
) -> Option<LzxBitRefill> {
    if bitness != 64 {
        return None;
    }
    if !is_movzx_byte_mem(&first.instr, None, None, 1) {
        return None;
    }
    let hi = first.instr.op0_register();
    let src = first.instr.memory_base();
    let bytes = collect_code_window(cpu, bus, ip, already, 40)?;
    let mut off = first.len as usize;
    let load_hi_rip = ip;

    let load_lo_rip = ip.wrapping_add(off as u64);
    let load_lo = take_next(&bytes, &mut off, ip, bitness)?;
    if !is_movzx_byte_mem(&load_lo.instr, None, Some(src), 0) {
        return None;
    }
    let lo = load_lo.instr.op0_register();
    if lo == hi {
        return None;
    }

    let movsx = take_next(&bytes, &mut off, ip, bitness)?;
    if movsx.instr.mnemonic() != Mnemonic::Movsx
        || movsx.instr.has_lock_prefix()
        || movsx.instr.op_kind(0) != OpKind::Register
        || movsx.instr.op_kind(1) != OpKind::Register
        || super::ops_data::reg_bits(movsx.instr.op0_register()) != Ok(32)
        || super::ops_data::reg_bits(movsx.instr.op1_register()) != Ok(8)
    {
        return None;
    }
    let shift = movsx.instr.op0_register();
    let bitcount = movsx.instr.op1_register();
    if shift != Register::ECX {
        // `shl hi, cl` below hard-wires CL.
        return None;
    }
    if shift == hi || shift == lo {
        return None;
    }

    let shl8 = take_next(&bytes, &mut off, ip, bitness)?;
    if shl8.instr.mnemonic() != Mnemonic::Shl
        || shl8.instr.op_kind(0) != OpKind::Register
        || shl8.instr.op0_register() != hi
        || shl8.instr.op_kind(1) != OpKind::Immediate8
        || shl8.instr.immediate8() != 8
    {
        return None;
    }

    let neg = take_next(&bytes, &mut off, ip, bitness)?;
    if neg.instr.mnemonic() != Mnemonic::Neg
        || neg.instr.has_lock_prefix()
        || neg.instr.op_kind(0) != OpKind::Register
        || neg.instr.op0_register() != shift
    {
        return None;
    }

    let add_src = take_next(&bytes, &mut off, ip, bitness)?;
    if add_src.instr.mnemonic() != Mnemonic::Add
        || add_src.instr.has_lock_prefix()
        || add_src.instr.op_kind(0) != OpKind::Register
        || add_src.instr.op0_register() != src
        || super::ops_data::reg_bits(src) != Ok(64)
        || !matches!(
            add_src.instr.op_kind(1),
            OpKind::Immediate8 | OpKind::Immediate8to64
        )
        || add_src.instr.immediate8() != 2
    {
        return None;
    }

    let or_hi = take_next(&bytes, &mut off, ip, bitness)?;
    if or_hi.instr.mnemonic() != Mnemonic::Or
        || or_hi.instr.op_kind(0) != OpKind::Register
        || or_hi.instr.op_kind(1) != OpKind::Register
        || or_hi.instr.op0_register() != hi
        || or_hi.instr.op1_register() != lo
    {
        return None;
    }

    let shl_cl = take_next(&bytes, &mut off, ip, bitness)?;
    if shl_cl.instr.mnemonic() != Mnemonic::Shl
        || shl_cl.instr.op_kind(0) != OpKind::Register
        || shl_cl.instr.op0_register() != hi
        || shl_cl.instr.op_kind(1) != OpKind::Register
        || shl_cl.instr.op1_register() != Register::CL
    {
        return None;
    }

    let or_bits = take_next(&bytes, &mut off, ip, bitness)?;
    if or_bits.instr.mnemonic() != Mnemonic::Or
        || or_bits.instr.op_kind(0) != OpKind::Register
        || or_bits.instr.op_kind(1) != OpKind::Register
        || or_bits.instr.op1_register() != hi
        || super::ops_data::reg_bits(or_bits.instr.op0_register()) != Ok(32)
    {
        return None;
    }
    let bits = or_bits.instr.op0_register();
    if bits == hi || bits == lo || bits == shift {
        return None;
    }

    let add_bits = take_next(&bytes, &mut off, ip, bitness)?;
    if add_bits.instr.mnemonic() != Mnemonic::Add
        || add_bits.instr.has_lock_prefix()
        || add_bits.instr.op_kind(0) != OpKind::Register
        || add_bits.instr.op0_register() != bitcount
        || super::ops_data::reg_bits(bitcount) != Ok(8)
        || add_bits.instr.op_kind(1) != OpKind::Immediate8
        || add_bits.instr.immediate8() != 0x10
    {
        return None;
    }

    Some(LzxBitRefill {
        src,
        hi,
        lo,
        shift,
        bits,
        bitcount,
        load_hi_rip,
        load_lo_rip,
        fallthrough: ip.wrapping_add(off as u64),
        insts: 10,
    })
}

fn exec_lzx_bit_refill<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    spec: &LzxBitRefill,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    if remaining < spec.insts {
        return Ok(0);
    }
    if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
        return Ok(0);
    }
    if cpu.state.halted {
        return Ok(0);
    }

    let src = cpu.state.read_reg(spec.src);
    let hi_byte = match bus.read_u8(cpu.state.apply_a20(src.wrapping_add(1))) {
        Ok(b) => b,
        Err(e) => {
            cpu.state.set_rip(spec.load_hi_rip);
            return Err((spec.load_hi_rip, e));
        }
    };
    cpu.state.write_reg(spec.hi, hi_byte as u64);

    let lo_byte = match bus.read_u8(cpu.state.apply_a20(src)) {
        Ok(b) => b,
        Err(e) => {
            cpu.state.set_rip(spec.load_lo_rip);
            return Err((spec.load_lo_rip, e));
        }
    };
    cpu.state.write_reg(spec.lo, lo_byte as u64);

    let bit8 = cpu.state.read_reg(spec.bitcount) as u8;
    let sx = bit8 as i8 as i32 as u32;
    cpu.state.write_reg(spec.shift, sx as u64);

    let mut tmp = (hi_byte as u32) << 8;
    let neg = sx.wrapping_neg();
    cpu.state.write_reg(spec.shift, neg as u64);
    cpu.state.write_reg(spec.src, src.wrapping_add(2));
    tmp |= lo_byte as u32;
    let count = neg & 0x1f;
    tmp = tmp.wrapping_shl(count);
    cpu.state.write_reg(spec.hi, tmp as u64);

    let bits = (cpu.state.read_reg(spec.bits) as u32) | tmp;
    cpu.state.write_reg(spec.bits, bits as u64);

    let old = cpu.state.read_reg(spec.bitcount);
    let (res, flags) = super::ops_alu::add_with_flags(&cpu.state, old, 0x10, 0, 8);
    cpu.state.write_reg(spec.bitcount, res);
    cpu.state.set_rflags(flags);
    cpu.state.set_rip(spec.fallthrough);

    cpu.pending.retire_instructions(spec.insts);
    cpu.time.advance_cycles(spec.insts);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(spec.insts)
}

/// Win7 `bcryptprimitives` SHA-512 compress (RVA `0xb000`, live RIP
/// `0x7fefc35bxxx` = base `0x7fefc350000` + inner σ). CryptoSysPrep spends
/// whole 1.5 B slices here. Calling convention: `rcx` = H[8], `rdx` = 128-byte
/// block. Prefix is the home-space + `sub rsp, 98h` prologue — 32 bytes so a
/// generic `mov [rsp+8], rcx; push` does not match. K-table is FIPS 180-4
/// (`crate::sha2_constants::SHA512_K`).
const BCRYPT_SHA512_COMPRESS_PREFIX: [u8; 32] = [
    0x48, 0x89, 0x4c, 0x24, 0x08, 0x53, 0x55, 0x56, 0x57, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41,
    0x57, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0x09, 0x4c, 0x8b,
];
const BCRYPT_SHA512_COMPRESS_INSNS: u64 = 64;
/// The first 15 bytes of the compress prologue, packed for the batch loop's gate.
const BCRYPT_SHA512_COMPRESS_KEY: u128 = packed_prefix(&BCRYPT_SHA512_COMPRESS_PREFIX, 15);

/// Pack the first `n` bytes of a fuse signature the same way [`decode_key`] packs
/// a live decode window.
const fn packed_prefix(bytes: &[u8], n: usize) -> u128 {
    let mut key = 0u128;
    let mut i = 0;
    while i < n {
        key |= (bytes[i] as u128) << (i * 8);
        i += 1;
    }
    key
}

fn sha512_compress_block(state: &mut [u64; 8], block: &[u8; 128]) {
    let k = &crate::sha2_constants::SHA512_K;
    let mut w = [0u64; 80];
    for i in 0..16 {
        w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    for i in 16..80 {
        let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
        let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
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
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(k[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
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

fn match_bcrypt_sha512_compress<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    already: &[u8],
) -> bool {
    // Must include `sub rsp, 98h` (bytes 17..24). Matching only the push
    // frame (15+3) false-hit other Win64 leaves, wrote 64 bytes through rcx,
    // and produced MEMORY_MANAGEMENT 0x1A (102b KiBugCheckData).
    // Use data reads for the tail so a paging `fetch` cannot false-negative.
    if already.len() >= 15 && already[..15] != BCRYPT_SHA512_COMPRESS_PREFIX[..15] {
        return false;
    }
    if already.len() >= 15 && already[..15] == BCRYPT_SHA512_COMPRESS_PREFIX[..15] {
        let linear = cpu
            .state
            .apply_a20(cpu.state.seg_base_reg(Register::CS).wrapping_add(ip));
        for i in 15..BCRYPT_SHA512_COMPRESS_PREFIX.len() {
            let Ok(b) = bus.read_u8(linear.wrapping_add(i as u64)) else {
                return false;
            };
            if b != BCRYPT_SHA512_COMPRESS_PREFIX[i] {
                return false;
            }
        }
        return true;
    }
    let Some(bytes) =
        collect_code_window(cpu, bus, ip, already, BCRYPT_SHA512_COMPRESS_PREFIX.len())
    else {
        return false;
    };
    bytes.as_slice() == BCRYPT_SHA512_COMPRESS_PREFIX
}

fn exec_bcrypt_sha512_compress<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    // Do not require a large remaining budget and do not refuse a queued but
    // undelivered IRQ: one Ok(0) at the entry prefix spends the only match
    // this CALL will ever see, and the rest of the leaf is interpreted.
    let _ = remaining;
    if cpu.state.halted {
        return Ok(0);
    }
    let state_addr = cpu.state.read_reg(Register::RCX);
    let block_addr = cpu.state.read_reg(Register::RDX);
    if state_addr == 0 || block_addr == 0 {
        return Ok(0);
    }
    let mut state = [0u64; 8];
    let mut block = [0u8; 128];
    for i in 0..8 {
        state[i] = bus
            .read_u64(cpu.state.apply_a20(state_addr.wrapping_add((i as u64) * 8)))
            .map_err(|e| (ip, e))?;
    }
    for i in 0..128 {
        block[i] = bus
            .read_u8(cpu.state.apply_a20(block_addr.wrapping_add(i as u64)))
            .map_err(|e| (ip, e))?;
    }
    sha512_compress_block(&mut state, &block);
    for i in 0..8 {
        bus.write_u64(
            cpu.state.apply_a20(state_addr.wrapping_add((i as u64) * 8)),
            state[i],
        )
        .map_err(|e| (ip, e))?;
    }
    // `ret`: caller sees callee-saved regs unchanged (we never touched them).
    let rsp = cpu.state.read_reg(Register::RSP);
    let ret = bus
        .read_u64(cpu.state.apply_a20(rsp))
        .map_err(|e| (ip, e))?;
    cpu.state.write_reg(Register::RSP, rsp.wrapping_add(8));
    cpu.state.set_rip(ret);
    cpu.pending
        .retire_instructions(BCRYPT_SHA512_COMPRESS_INSNS);
    cpu.time.advance_cycles(BCRYPT_SHA512_COMPRESS_INSNS);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(BCRYPT_SHA512_COMPRESS_INSNS)
}

/// KERNELBASE UTF-16 NLS mapping loop (live RIP `0x7fefcf46ba2` /
/// `KernelBase!NLSOEMUP` style table walk). 19 insns / wchar; ~10–15% of
/// specialize `drvinst` samples. Match is a 8-byte prefix plus a 72-byte
/// memcmp — not a 19-insn decode — because `mov ebx, edi` is common and a
/// decode-heavy matcher lost 4.28→4.19 Minst/s on this same snap.
const KERNELBASE_NLS_UPCASE: [u8; 72] = [
    0x8b, 0xdf, 0x45, 0x0f, 0xb7, 0x04, 0x19, 0x49, 0x83, 0xc1, 0x02, 0x41, 0x0f, 0xb6, 0xd0, 0x41,
    0x0f, 0xb7, 0xc0, 0x66, 0xc1, 0xe8, 0x08, 0x0f, 0xb6, 0xc0, 0x41, 0x0f, 0xb7, 0x0c, 0x42, 0x48,
    0x8b, 0xc2, 0x83, 0xe2, 0x0f, 0x48, 0xc1, 0xe8, 0x04, 0x48, 0x03, 0xc8, 0x41, 0x0f, 0xb7, 0x04,
    0x4a, 0x48, 0x03, 0xc2, 0x41, 0x0f, 0xb7, 0x04, 0x42, 0x66, 0x41, 0x03, 0xc0, 0x49, 0x83, 0xeb,
    0x01, 0x66, 0x41, 0x89, 0x41, 0xfe, 0x75, 0xba,
];
const KERNELBASE_NLS_UPCASE_INSNS: u64 = 19;
const KERNELBASE_NLS_UPCASE_KEY: u128 = packed_prefix(&KERNELBASE_NLS_UPCASE, 8);

/// Win7 SP1 x64 `KiSwapContext` UP spin: `pause; cmp byte [rsi+0x49], 0`.
/// `KTHREAD.Running` is +0x49. On a uniprocessor the flag is sometimes left 1
/// from a missed remote CPU, and this loop never exits (IRQL DISPATCH, so a
/// clock DPC cannot preempt it). Store 0 so the following `je` can leave —
/// but never on the thread that is already Current (that self-reschedules the
/// hog). When Current and the live 14-byte `je; jmp` encoding is present,
/// retire the spin in a tight helper instead of decode/dispatch per pause.
const WIN7_KTHREAD_RUNNING_SPIN: [u8; 6] = [0xF3, 0x90, 0x80, 0x7E, 0x49, 0x00];
const WIN7_KTHREAD_RUNNING_SPIN_KEY: u128 = packed_prefix(&WIN7_KTHREAD_RUNNING_SPIN, 6);
/// Live tail after the 6-byte prefix at ntos `0xfffff800026eb41e`:
/// `je rel32` (exit) + `jmp rel8` (back to pause).
const WIN7_KTHREAD_RUNNING_SPIN_JE_JMP: [u8; 8] = [0x0F, 0x84, 0xE1, 0xFC, 0xFF, 0xFF, 0xEB, 0xF2];
const WIN7_KTHREAD_RUNNING_SPIN_INSNS: u64 = 4; // pause, cmp, je, jmp

fn kthread_running_is_current<B: CpuBus>(cpu: &interrupts::CpuCore, bus: &mut B, rsi: u64) -> bool {
    let gs = cpu.state.msr.gs_base;
    if gs <= 0xffff_8000_0000_0000 {
        return false;
    }
    match bus.read_u64(cpu.state.apply_a20(gs.wrapping_add(0x188))) {
        Ok(current) => current == rsi,
        Err(_) => false,
    }
}

fn exec_win7_kthread_running_spin<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    remaining: u64,
    already: &[u8],
) -> Result<u64, (u64, Exception)> {
    if cpu.state.halted {
        return Ok(0);
    }
    let rsi = cpu.state.read_reg(Register::RSI);
    let addr = cpu.state.apply_a20(rsi.wrapping_add(0x49));
    let running = match bus.read_u8(addr) {
        Ok(b) => b,
        Err(e) => return Err((ip, e)),
    };
    // Do not clear KTHREAD.Running on the thread that is already Current.
    // That turns a swap-to-RIT into a self-reschedule of the hog (sppsvc 316).
    let is_current = kthread_running_is_current(cpu, bus, rsi);
    if is_current && running != 0 {
        return exec_win7_kthread_running_spin_current(cpu, ip, remaining, already);
    }
    if running != 0 {
        if let Err(e) = bus.write_u8(addr, 0) {
            return Err((ip, e));
        }
    }
    cpu.state.set_rip(ip.wrapping_add(2));
    cpu.pending.retire_instructions(1);
    cpu.time.advance_cycles(1);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(1)
}

/// Batch `pause; cmp [rsi+0x49],0; je; jmp` while Current.Running stays 1.
///
/// A queued but undeliverable IRQ (typical at CR8=2) must not refuse the
/// helper — falling back to per-insn dispatch is the desktop hog. Only a
/// pending exception already accepted by the core aborts the batch.
fn exec_win7_kthread_running_spin_current(
    cpu: &mut interrupts::CpuCore,
    ip: u64,
    remaining: u64,
    already: &[u8],
) -> Result<u64, (u64, Exception)> {
    if remaining < WIN7_KTHREAD_RUNNING_SPIN_INSNS {
        return Ok(0);
    }
    if already.len() < 14 || already[6..14] != WIN7_KTHREAD_RUNNING_SPIN_JE_JMP {
        return Ok(0);
    }
    if cpu.pending.has_pending_event() {
        return Ok(0);
    }

    let iters = remaining / WIN7_KTHREAD_RUNNING_SPIN_INSNS;
    let retired = iters * WIN7_KTHREAD_RUNNING_SPIN_INSNS;
    // cmp byte [rsi+0x49], 0 with the live 1: ZF/SF/CF/OF/AF clear, PF clear
    // (odd parity). PAUSE and the not-taken je / jmp do not touch flags.
    let (_res, flags) = super::ops_alu::sub_with_flags(&cpu.state, 1, 0, 0, 8);
    cpu.state.set_rflags(flags);
    cpu.state.set_rip(ip);
    cpu.pending.retire_instructions(retired);
    cpu.time.advance_cycles(retired);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(retired)
}

fn match_kernelbase_nls_upcase<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    already: &[u8],
) -> bool {
    let Some(bytes) = collect_code_window(cpu, bus, ip, already, KERNELBASE_NLS_UPCASE.len())
    else {
        return false;
    };
    bytes.as_slice() == KERNELBASE_NLS_UPCASE
}

fn exec_kernelbase_nls_upcase<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    const LOAD_RIP: u64 = 2;
    const T1_RIP: u64 = 26;
    const T2_RIP: u64 = 44;
    const T3_RIP: u64 = 52;
    const STORE_RIP: u64 = 65;

    if remaining < KERNELBASE_NLS_UPCASE_INSNS {
        return Ok(0);
    }
    if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
        return Ok(0);
    }
    if cpu.state.halted {
        return Ok(0);
    }

    let mut count = cpu.state.read_reg(Register::R11);
    if count == 0 || count > 1_000_000 {
        return Ok(0);
    }
    let max_chars = remaining / KERNELBASE_NLS_UPCASE_INSNS;
    if max_chars == 0 {
        return Ok(0);
    }

    let edi = cpu.state.read_reg(Register::EDI) & 0xffff_ffff;
    cpu.state.write_reg(Register::RBX, edi);
    let table = cpu.state.read_reg(Register::R10);
    let mut r9 = cpu.state.read_reg(Register::R9);
    let mut last_ch = 0u16;
    let mut last_edx = 0u64;
    let mut last_rcx = 0u64;
    let mut last_ax = 0u16;
    let mut did = 0u64;

    while did < max_chars && count != 0 {
        if cpu.pending.has_pending_event() || !cpu.pending.external_interrupts().is_empty() {
            break;
        }
        let src = r9.wrapping_add(edi);
        let ch = match bus.read_u16(cpu.state.apply_a20(src)) {
            Ok(v) => v,
            Err(e) => {
                cpu.state.set_rip(ip.wrapping_add(LOAD_RIP));
                return Err((ip.wrapping_add(LOAD_RIP), e));
            }
        };
        r9 = r9.wrapping_add(2);
        let lo = u64::from(ch & 0xff);
        let hi = u64::from(ch >> 8);
        let t1 = match bus.read_u16(cpu.state.apply_a20(table.wrapping_add(hi.wrapping_mul(2)))) {
            Ok(v) => u64::from(v),
            Err(e) => {
                cpu.state.set_rip(ip.wrapping_add(T1_RIP));
                return Err((ip.wrapping_add(T1_RIP), e));
            }
        };
        let edx = lo & 0x0f;
        let rcx = t1.wrapping_add(lo >> 4);
        let t2 = match bus.read_u16(cpu.state.apply_a20(table.wrapping_add(rcx.wrapping_mul(2)))) {
            Ok(v) => u64::from(v),
            Err(e) => {
                cpu.state.set_rip(ip.wrapping_add(T2_RIP));
                return Err((ip.wrapping_add(T2_RIP), e));
            }
        };
        let rax = t2.wrapping_add(edx);
        let t3 = match bus.read_u16(cpu.state.apply_a20(table.wrapping_add(rax.wrapping_mul(2)))) {
            Ok(v) => v,
            Err(e) => {
                cpu.state.set_rip(ip.wrapping_add(T3_RIP));
                return Err((ip.wrapping_add(T3_RIP), e));
            }
        };
        let ax = t3.wrapping_add(ch);
        if let Err(e) = bus.write_u16(cpu.state.apply_a20(r9.wrapping_sub(2)), ax) {
            cpu.state.set_rip(ip.wrapping_add(STORE_RIP));
            return Err((ip.wrapping_add(STORE_RIP), e));
        }
        count = count.wrapping_sub(1);
        last_ch = ch;
        last_edx = edx;
        last_rcx = rcx;
        last_ax = ax;
        did += 1;
    }

    if did == 0 {
        return Ok(0);
    }

    cpu.state.write_reg(Register::R9, r9);
    cpu.state.write_reg(Register::R11, count);
    cpu.state.write_reg(Register::R8, u64::from(last_ch));
    cpu.state.write_reg(Register::RDX, last_edx);
    cpu.state.write_reg(Register::RAX, u64::from(last_ax));
    cpu.state.write_reg(Register::RCX, last_rcx);

    let (res, flags) = super::ops_alu::sub_with_flags(&cpu.state, count.wrapping_add(1), 1, 0, 64);
    debug_assert_eq!(res, count);
    cpu.state.set_rflags(flags);

    let retired = did.saturating_mul(KERNELBASE_NLS_UPCASE_INSNS);
    if count == 0 {
        cpu.state
            .set_rip(ip.wrapping_add(KERNELBASE_NLS_UPCASE.len() as u64));
    } else {
        cpu.state.set_rip(ip);
    }
    cpu.pending.retire_instructions(retired);
    cpu.time.advance_cycles(retired);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(retired)
}

/// ntoskrnl PAGELK 3-byte name-hash insert (live RIP `0xfffff8000d4cdba0`,
/// RVA `0x2b5ba0`). Called once per 4 KiB WIM/XPRESS window slot; ~55% of
/// samples on the 155b+ expand slices. No CALLs; flags are clobbered.
const NTOS_NAME_HASH_PREFIX: [u8; 32] = [
    0x48, 0x89, 0x5c, 0x24, 0x08, 0x48, 0x89, 0x6c, 0x24, 0x10, 0x48, 0x89, 0x74, 0x24, 0x18, 0x48,
    0x89, 0x7c, 0x24, 0x20, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x44, 0x0f, 0xb6, 0x39,
];

fn match_ntos_name_hash<B: CpuBus>(
    cpu: &interrupts::CpuCore,
    bus: &mut B,
    ip: u64,
    first: &DecodedInst,
) -> bool {
    let instr = &first.instr;
    if instr.mnemonic() != Mnemonic::Mov
        || instr.op_kind(0) != OpKind::Memory
        || instr.op_kind(1) != OpKind::Register
        || instr.op1_register() != Register::RBX
        || instr.memory_base() != Register::RSP
        || instr.memory_index() != Register::None
        || instr.memory_displacement64() != 8
        || instr.memory_size() != MemorySize::UInt64
    {
        return false;
    }
    let Some(bytes) = collect_code_window(cpu, bus, ip, &[], NTOS_NAME_HASH_PREFIX.len()) else {
        return false;
    };
    bytes == NTOS_NAME_HASH_PREFIX
}

// The `loop` below never iterates: every branch ends in `break`, and there is
// no `continue`. It is an early-exit block written before labelled blocks were
// available, and `'blk: { … break 'blk; }` would say so properly. Left as-is
// deliberately — this is a fused-leaf helper, and restructuring one is how the
// SHA-512 fuse came to corrupt guest memory. The lint is denied by default and
// aborts clippy for every crate downstream of this one, so it is suppressed
// rather than left to block the gate.
#[allow(clippy::never_loop)]
fn exec_ntos_name_hash<B: CpuBus>(
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    remaining: u64,
) -> Result<u64, (u64, Exception)> {
    if remaining < 40 {
        return Ok(0);
    }
    let head = cpu.state.rip();
    let mut n = 0u64;

    let rd8 = |bus: &mut B, cpu: &interrupts::CpuCore, addr: u64, rip: u64| {
        bus.read_u8(cpu.state.apply_a20(addr)).map_err(|e| (rip, e))
    };
    let rd64 = |bus: &mut B, cpu: &interrupts::CpuCore, addr: u64, rip: u64| {
        bus.read_u64(cpu.state.apply_a20(addr))
            .map_err(|e| (rip, e))
    };
    let rd32 = |bus: &mut B, cpu: &interrupts::CpuCore, addr: u64, rip: u64| {
        bus.read_u32(cpu.state.apply_a20(addr))
            .map_err(|e| (rip, e))
    };
    let wr64 = |bus: &mut B, cpu: &interrupts::CpuCore, addr: u64, val: u64, rip: u64| {
        bus.write_u64(cpu.state.apply_a20(addr), val)
            .map_err(|e| (rip, e))
    };

    // Home-space saves of rbx/rbp/rsi/rdi, then push r12-r15.
    let rsp0 = cpu.state.read_reg(Register::RSP);
    wr64(
        bus,
        cpu,
        rsp0.wrapping_add(8),
        cpu.state.read_reg(Register::RBX),
        head,
    )?;
    wr64(
        bus,
        cpu,
        rsp0.wrapping_add(0x10),
        cpu.state.read_reg(Register::RBP),
        head.wrapping_add(5),
    )?;
    wr64(
        bus,
        cpu,
        rsp0.wrapping_add(0x18),
        cpu.state.read_reg(Register::RSI),
        head.wrapping_add(0xa),
    )?;
    wr64(
        bus,
        cpu,
        rsp0.wrapping_add(0x20),
        cpu.state.read_reg(Register::RDI),
        head.wrapping_add(0xf),
    )?;
    n += 4;
    let mut rsp = rsp0;
    for (reg, off) in [
        (Register::R12, 0x14u64),
        (Register::R13, 0x16),
        (Register::R14, 0x18),
        (Register::R15, 0x1a),
    ] {
        rsp = rsp.wrapping_sub(8);
        wr64(
            bus,
            cpu,
            rsp,
            cpu.state.read_reg(reg),
            head.wrapping_add(off),
        )?;
        n += 1;
    }
    cpu.state.write_reg(Register::RSP, rsp);

    let key = cpu.state.read_reg(Register::RCX);
    let table = cpu.state.read_reg(Register::RDX);
    let b0 = rd8(bus, cpu, key, head.wrapping_add(0x1c))? as u64;
    let lo = rd64(bus, cpu, table, head.wrapping_add(0x20))?;
    let b2 = rd8(bus, cpu, key.wrapping_add(2), head.wrapping_add(0x23))? as u64;
    let hi = rd64(bus, cpu, table.wrapping_add(8), head.wrapping_add(0x28))?;
    let len = rd32(bus, cpu, table.wrapping_add(0x10), head.wrapping_add(0x2c))? as u64;
    let b1 = rd8(bus, cpu, key.wrapping_add(1), head.wrapping_add(0x33))? as u64;
    n += 9; // movzx/mov through movzx edx, [rcx+1] plus mov rdi/eax/r8 (3 more below)

    let mut eax = b0 as u32;
    eax <<= 4;
    eax ^= b1 as u32;
    eax <<= 4;
    eax ^= b2 as u32;
    eax = eax.wrapping_mul(0xffff9e5f);
    eax = ((eax as i32) >> 4) as u32;
    eax &= 0xfff;
    n += 12; // mov eax,r15d .. and eax,0xfff (approx; exact path below)

    let r12 = (eax as u64).wrapping_mul(2);
    let r13 = (eax as u64).wrapping_add(2).wrapping_mul(2);
    let rbp = rd64(
        bus,
        cpu,
        table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
        head.wrapping_add(0x66),
    )?;
    let rsi = rd64(
        bus,
        cpu,
        table.wrapping_add(r13.wrapping_mul(8)),
        head.wrapping_add(0x6e),
    )?;
    n += 6;

    cpu.state.write_reg(Register::R8, key);
    cpu.state.write_reg(Register::RDI, table);
    cpu.state.write_reg(Register::R9, len);
    cpu.state.write_reg(Register::R14, lo);
    cpu.state.write_reg(Register::RBX, hi);
    cpu.state.write_reg(Register::R12, r12);
    cpu.state.write_reg(Register::R13, r13);
    cpu.state.write_reg(Register::R15, b0);
    cpu.state.write_reg(Register::RDX, b1);
    cpu.state.write_reg(Register::RAX, eax as u64);

    let mut ecx: u32 = 0;
    let mut r10d: u32 = 3;
    let mut r11 = b2;
    let mut edx = b1;

    // cmp rsi, r14 / jae 0xa8
    n += 2;
    if rsi >= lo {
        // 0xa8: cmp rsi, r8 / jae 0x77 — fall into walk after optional prefix
        n += 2;
        if rsi < key {
            let hit = prefix_eq(bus, cpu, rsi, b0, edx, r11, head.wrapping_add(0xad))?;
            n += 6;
            if hit {
                ecx = r10d;
                n += 2; // mov ecx,r10d; cmp r9d,ecx
                if len > ecx as u64 {
                    r11 = rsi;
                    let mut rdx = key.wrapping_add(3);
                    r11 = r11.wrapping_sub(key);
                    n += 3;
                    loop {
                        n += 3; // mov eax,ecx; add rax,r8; cmp rax,rbx
                        let rax = (ecx as u64).wrapping_add(key);
                        if rax >= hi {
                            break;
                        }
                        let a = rd8(bus, cpu, r11.wrapping_add(rdx), head.wrapping_add(0xd9))?;
                        let b = rd8(bus, cpu, rdx, head.wrapping_add(0xde))?;
                        n += 2;
                        if a != b {
                            break;
                        }
                        ecx = ecx.wrapping_add(1);
                        rdx = rdx.wrapping_add(1);
                        n += 3; // inc ecx; inc rdx; cmp ecx,r9d
                        if (ecx as u64) >= len {
                            break;
                        }
                    }
                    edx = rd8(bus, cpu, key.wrapping_add(1), head.wrapping_add(0xec))? as u64;
                    r11 = rd8(bus, cpu, key.wrapping_add(2), head.wrapping_add(0xf1))? as u64;
                    n += 3; // two movzx + jmp
                }
            }
        }
    }

    // Walk at 0x77 until insert.
    loop {
        n += 2; // cmp rbp,r14 / jae
        if rbp < lo {
            // insert at 0x7c
            wr64(
                bus,
                cpu,
                table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
                rsi,
                head.wrapping_add(0x7c),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(r13.wrapping_mul(8)),
                key,
                head.wrapping_add(0x81),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(0x18),
                rsi,
                head.wrapping_add(0x87),
            )?;
            cpu.state.write_reg(Register::RAX, ecx as u64);
            n += 4;
            break;
        }

        // 0xfb
        n += 2; // cmp rbp,r8 / jae
        if rbp >= key {
            wr64(
                bus,
                cpu,
                table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
                rsi,
                head.wrapping_add(0x7c),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(r13.wrapping_mul(8)),
                key,
                head.wrapping_add(0x81),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(0x18),
                rsi,
                head.wrapping_add(0x87),
            )?;
            cpu.state.write_reg(Register::RAX, ecx as u64);
            n += 4;
            break;
        }
        let hit = prefix_eq(bus, cpu, rbp, b0, edx, r11, head.wrapping_add(0x104))?;
        n += 6;
        if !hit {
            wr64(
                bus,
                cpu,
                table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
                rsi,
                head.wrapping_add(0x7c),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(r13.wrapping_mul(8)),
                key,
                head.wrapping_add(0x81),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(0x18),
                rsi,
                head.wrapping_add(0x87),
            )?;
            cpu.state.write_reg(Register::RAX, ecx as u64);
            n += 4;
            break;
        }
        n += 2; // cmp r9d,r10d / jbe
        if len > r10d as u64 {
            r11 = rbp;
            let mut rdx = key.wrapping_add(3);
            r11 = r11.wrapping_sub(key);
            let mut eax = r10d;
            n += 4;
            loop {
                let rax = (eax as u64).wrapping_add(key);
                n += 3;
                if rax >= hi {
                    break;
                }
                let a = rd8(bus, cpu, r11.wrapping_add(rdx), head.wrapping_add(0x13b))?;
                let b = rd8(bus, cpu, rdx, head.wrapping_add(0x140))?;
                n += 2;
                if a != b {
                    break;
                }
                r10d = r10d.wrapping_add(1);
                rdx = rdx.wrapping_add(1);
                eax = r10d;
                n += 3;
                if (r10d as u64) >= len {
                    break;
                }
            }
        }
        n += 2; // cmp ecx,r10d / jae
        if ecx >= r10d {
            wr64(
                bus,
                cpu,
                table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
                rsi,
                head.wrapping_add(0x7c),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(r13.wrapping_mul(8)),
                key,
                head.wrapping_add(0x81),
            )?;
            wr64(
                bus,
                cpu,
                table.wrapping_add(0x18),
                rsi,
                head.wrapping_add(0x87),
            )?;
            cpu.state.write_reg(Register::RAX, ecx as u64);
            n += 4;
            break;
        }
        wr64(
            bus,
            cpu,
            table.wrapping_add(r12.wrapping_mul(8)).wrapping_add(0x28),
            rsi,
            head.wrapping_add(0x158),
        )?;
        wr64(
            bus,
            cpu,
            table.wrapping_add(r13.wrapping_mul(8)),
            key,
            head.wrapping_add(0x15d),
        )?;
        wr64(
            bus,
            cpu,
            table.wrapping_add(0x18),
            rbp,
            head.wrapping_add(0x164),
        )?;
        cpu.state.write_reg(Register::RAX, r10d as u64);
        n += 5;
        break;
    }

    // Restore rbx/rbp/rsi/rdi from home space, pop r15-r12, RET.
    let rbx = rd64(bus, cpu, rsp.wrapping_add(0x28), head.wrapping_add(0x8b))?;
    let rbp = rd64(bus, cpu, rsp.wrapping_add(0x30), head.wrapping_add(0x90))?;
    let rsi = rd64(bus, cpu, rsp.wrapping_add(0x38), head.wrapping_add(0x95))?;
    let rdi = rd64(bus, cpu, rsp.wrapping_add(0x40), head.wrapping_add(0x9a))?;
    cpu.state.write_reg(Register::RBX, rbx);
    cpu.state.write_reg(Register::RBP, rbp);
    cpu.state.write_reg(Register::RSI, rsi);
    cpu.state.write_reg(Register::RDI, rdi);
    n += 4;
    let r15 = rd64(bus, cpu, rsp, head.wrapping_add(0x9f))?;
    let r14 = rd64(bus, cpu, rsp.wrapping_add(8), head.wrapping_add(0xa1))?;
    let r13 = rd64(bus, cpu, rsp.wrapping_add(0x10), head.wrapping_add(0xa3))?;
    let r12v = rd64(bus, cpu, rsp.wrapping_add(0x18), head.wrapping_add(0xa5))?;
    cpu.state.write_reg(Register::R15, r15);
    cpu.state.write_reg(Register::R14, r14);
    cpu.state.write_reg(Register::R13, r13);
    cpu.state.write_reg(Register::R12, r12v);
    n += 4;
    let ret = rd64(bus, cpu, rsp.wrapping_add(0x20), head.wrapping_add(0xa7))?;
    cpu.state.write_reg(Register::RSP, rsp.wrapping_add(0x28));
    cpu.state.set_rip(ret);
    n += 1;

    // Volatiles left as the real function would: r8=key, rdi restored,
    // eax = match length, r9/r10/r11/rdx/rcx last used. The x64 ABI
    // clobbers them; the oracle checks rax/rsp/rip/callee-saved + table.

    cpu.state.write_reg(Register::RCX, ecx as u64);
    cpu.state.write_reg(Register::R10, r10d as u64);
    cpu.state.write_reg(Register::R11, r11);
    cpu.state.write_reg(Register::RDX, edx);
    cpu.state.write_reg(Register::R8, key);
    cpu.state.write_reg(Register::R9, len);
    // rdi/rsi already restored.

    cpu.pending.retire_instructions(n);
    cpu.time.advance_cycles(n);
    cpu.state.msr.tsc = cpu.time.read_tsc();
    Ok(n)
}

fn prefix_eq<B: CpuBus>(
    bus: &mut B,
    cpu: &interrupts::CpuCore,
    ptr: u64,
    b0: u64,
    b1: u64,
    b2: u64,
    rip: u64,
) -> Result<bool, (u64, Exception)> {
    let a0 = bus
        .read_u8(cpu.state.apply_a20(ptr))
        .map_err(|e| (rip, e))?;
    if a0 as u64 != b0 {
        return Ok(false);
    }
    let a1 = bus
        .read_u8(cpu.state.apply_a20(ptr.wrapping_add(1)))
        .map_err(|e| (rip.wrapping_add(5), e))?;
    if a1 as u64 != b1 {
        return Ok(false);
    }
    let a2 = bus
        .read_u8(cpu.state.apply_a20(ptr.wrapping_add(2)))
        .map_err(|e| (rip.wrapping_add(10), e))?;
    Ok(a2 as u64 == b2)
}

/// A caller-supplied hook offered each Tier-0 basic-block boundary.
///
/// `Machine::run_slice` runs Tier-0 with `exit_on_branch: false`, because ending
/// the batch on every branch forces a device re-poll and paging-bus rebuild that
/// costs far more than the branch itself. The consequence is that the caller sees
/// exactly one RIP per batch — whichever the instruction budget expired at — so a
/// compiled block is entered only if the budget happens to land on its entry. At a
/// few hundred instructions of coverage per batch that caps a compiled tier well
/// below what it is worth.
///
/// Deliberately **not** generic. Making the batch generic over a hook type gives it
/// a second monomorphisation, and that alone measured as a 4% regression on the
/// untiered path even with the hook compiled away. A pointer and a function pointer
/// keep one body and cost one predictable branch per taken branch.
///
/// The context pointer is opaque here; only the callback that was built alongside
/// it ever interprets it.
pub struct Tier0BranchSeam {
    ctx: *mut (),
    /// Called at a taken branch. Returns how many guest instructions it executed,
    /// having already advanced RIP, retirement and the time source for them; 0
    /// means it did nothing and Tier-0 should carry on.
    ///
    /// The CPU and bus are handed over as arguments rather than captured, because
    /// the batch holds `&mut` to both for its whole duration — a seam that captured
    /// them would alias those borrows. The bus arrives type-erased for the same
    /// reason the seam is not generic; the callback was built knowing its type.
    run: fn(
        ctx: *mut (),
        cpu: &mut interrupts::CpuCore,
        bus: *mut (),
        assist: *mut AssistContext,
        budget: u64,
    ) -> u64,
    /// How many more instructions to run before offering the tier again, outside a
    /// taken branch. Zero means never.
    ///
    /// The seam is otherwise only offered at taken branches, so when a chain stops
    /// because the tier refused a block the interpreter carries on to the next
    /// branch — about thirteen instructions, every time, and that is most of what
    /// the compiled tier still spends on interpretation. The tier sets this to 1
    /// when a chain stops somewhere it refused, meaning "step over that instruction
    /// and hand back". A count rather than a RIP because the length of an
    /// instruction Tier-1 could not decode is exactly what is not known.
    resume_in: std::rc::Rc<std::cell::Cell<u64>>,
}

impl Tier0BranchSeam {
    /// `ctx` is passed back to `run` unchanged. The caller is responsible for it
    /// outliving this seam.
    pub fn new(
        ctx: *mut (),
        run: fn(
            ctx: *mut (),
            cpu: &mut interrupts::CpuCore,
            bus: *mut (),
            assist: *mut AssistContext,
            budget: u64,
        ) -> u64,
        resume_in: std::rc::Rc<std::cell::Cell<u64>>,
    ) -> Self {
        Self {
            ctx,
            run,
            resume_in,
        }
    }

    #[inline]
    fn offer(
        &mut self,
        cpu: &mut interrupts::CpuCore,
        bus: *mut (),
        assist: *mut AssistContext,
        budget: u64,
    ) -> u64 {
        (self.run)(self.ctx, cpu, bus, assist, budget)
    }

    /// Instructions left before the tier is offered again outside a branch.
    #[inline]
    fn resume_in(&self) -> u64 {
        self.resume_in.get()
    }

    #[inline]
    fn set_resume_in(&self, n: u64) {
        self.resume_in.set(n);
    }
}

pub fn run_batch_cpu_core_with_assists<B: CpuBus>(
    cfg: &Tier0Config,
    ctx: &mut AssistContext,
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    max_insts: u64,
) -> BatchResult {
    run_batch_cpu_core_with_assists_seamed(cfg, ctx, cpu, bus, max_insts, None)
}

pub fn run_batch_cpu_core_with_assists_seamed<B: CpuBus>(
    cfg: &Tier0Config,
    ctx: &mut AssistContext,
    cpu: &mut interrupts::CpuCore,
    bus: &mut B,
    max_insts: u64,
    mut seam: Option<&mut Tier0BranchSeam>,
) -> BatchResult {
    use aero_x86::{Mnemonic, OpKind};

    if max_insts == 0 {
        return BatchResult {
            executed: 0,
            exit: if cpu.state.halted {
                BatchExit::Halted
            } else {
                BatchExit::Completed
            },
        };
    }

    // Borrow the decode cache once per batch (not once per instruction) so the
    // TLS `LocalKey::with` / RefCell overhead stays off the per-insn hot path.
    DECODE_CACHE.with(|cell| {
    let mut cache = cell.borrow_mut();
    let mut executed = 0u64;
    while executed < max_insts {
        // Give pending exceptions/interrupts a chance at instruction boundaries.
        if cpu.pending.has_pending_event() {
            match cpu.deliver_pending_event(bus) {
                Ok(()) => continue,
                Err(exit) => {
                    return BatchResult {
                        executed,
                        exit: BatchExit::CpuExit(exit),
                    };
                }
            }
        }
        if !cpu.pending.external_interrupts().is_empty() {
            let before = cpu.pending.external_interrupts().len();
            match cpu.deliver_external_interrupt(bus) {
                Ok(()) => {
                    if cpu.pending.external_interrupts().len() != before {
                        continue;
                    }
                }
                Err(exit) => {
                    return BatchResult {
                        executed,
                        exit: BatchExit::CpuExit(exit),
                    };
                }
            }
        }
        if cpu.state.halted {
            return BatchResult {
                executed,
                exit: BatchExit::Halted,
            };
        }

        // The one place the tier is offered outside a taken branch. It asked to be
        // handed back once the instruction it could not translate has been stepped
        // over, rather than after the interpreter has run on to the next branch.
        if let Some(seam) = seam.as_deref_mut() {
            let left = seam.resume_in();
            if left > 0 {
                seam.set_resume_in(left - 1);
                if left == 1 {
                    let budget = max_insts.saturating_sub(executed);
                    if budget > 0 {
                        let erased = bus as *mut B as *mut ();
                        // Release the decode cache across the offer. It is held for
                        // the whole batch to keep a TLS access and a `RefCell` borrow
                        // off the per-instruction path, but the compiled tier may
                        // step an instruction of its own through here, and that needs
                        // the cache too. An offer happens about once in two hundred
                        // instructions, so re-taking it is not on any hot path.
                        drop(cache);
                        let ran = seam.offer(cpu, erased, ctx as *mut AssistContext, budget);
                        cache = cell.borrow_mut();
                        if ran > 0 {
                            executed = executed.saturating_add(ran);
                            continue;
                        }
                    }
                }
            }
        }

        bus.sync(&cpu.state);

        let ip = cpu.state.rip();
        let cs_base = cpu.state.seg_base_reg(Register::CS);
        watch_exec(&cpu.state, bus, cs_base.wrapping_add(ip));
        let (mut bytes, available) =
            match fetch_wrapped_seg_ip_first_page(&cpu.state, bus, cs_base, ip, 15) {
            Ok(result) => result,
            Err(e) => match deliver_or_abort_exception(cpu, bus, ip, e) {
                Ok(()) => continue,
                Err(exit) => {
                    return BatchResult {
                        executed,
                        exit,
                    };
                }
            },
        };

        let bitness = cpu.state.bitness();
        // Every fused leaf below is keyed on an exact byte prefix, and the decode
        // cache validates a hit against the same bytes. Packing the window once
        // turns both into masked integer compares.
        let mut window_key = decode_key(&bytes[..available]);
        let decoded_result = match cache.decode_ref(ip, bitness, &bytes[..available], window_key) {
            Err(aero_x86::DecodeError::UnexpectedEof) if available < 15 => {
                bytes = match fetch_wrapped_seg_ip(&cpu.state, bus, cs_base, ip, 15) {
                    Ok(bytes) => bytes,
                    Err(e) => match deliver_or_abort_exception(cpu, bus, ip, e) {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                };
                window_key = decode_key(&bytes);
                cache.decode_ref(ip, bitness, &bytes, window_key)
            }
            result => result,
        };
        let (decoded, addr_size_override) = match decoded_result {
            Ok(result) => result,
            Err(_) => {
                let e = Exception::InvalidOpcode;
                match deliver_or_abort_exception(cpu, bus, ip, e) {
                    Ok(()) => continue,
                    Err(exit) => {
                        return BatchResult {
                            executed,
                            exit,
                        };
                    }
                }
            }
        };

        if available >= 6 && window_key & PREFIX_MASK_6 == WIN7_KTHREAD_RUNNING_SPIN_KEY {
            match exec_win7_kthread_running_spin(
                cpu,
                bus,
                ip,
                max_insts - executed,
                &bytes[..available],
            ) {
                Ok(0) => {}
                Ok(n) => {
                    executed += n;
                    continue;
                }
                Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e) {
                    Ok(()) => continue,
                    Err(exit) => {
                        return BatchResult {
                            executed,
                            exit,
                        };
                    }
                },
            }
        }

        if available >= 15
            && window_key == BCRYPT_SHA512_COMPRESS_KEY
            && match_bcrypt_sha512_compress(cpu, bus, ip, &bytes[..available])
        {
            match exec_bcrypt_sha512_compress(cpu, bus, ip, max_insts - executed) {
                Ok(0) => {}
                Ok(n) => {
                    executed += n;
                    continue;
                }
                Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e) {
                    Ok(()) => continue,
                    Err(exit) => {
                        return BatchResult {
                            executed,
                            exit,
                        };
                    }
                },
            }
        }

        if (max_insts - executed) >= KERNELBASE_NLS_UPCASE_INSNS
            && available >= 8
            && window_key & PREFIX_MASK_8 == KERNELBASE_NLS_UPCASE_KEY
            && match_kernelbase_nls_upcase(cpu, bus, ip, &bytes[..available])
        {
            match exec_kernelbase_nls_upcase(cpu, bus, ip, max_insts - executed) {
                Ok(0) => {}
                Ok(n) => {
                    executed += n;
                    continue;
                }
                Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e) {
                    Ok(()) => continue,
                    Err(exit) => {
                        return BatchResult {
                            executed,
                            exit,
                        };
                    }
                },
            }
        }

        // Each of the remaining fuses requires a specific opening mnemonic, so read
        // it once and skip the matchers that cannot possibly apply. Without this
        // every fuse re-derives the same iced operand shape on every instruction.
        let fuse_mnemonic = decoded.instr.mnemonic();
        let try_mov = matches!(fuse_mnemonic, Mnemonic::Mov);
        let try_neg = matches!(fuse_mnemonic, Mnemonic::Neg);
        let try_movzx = matches!(fuse_mnemonic, Mnemonic::Movzx);
        let try_inc = matches!(fuse_mnemonic, Mnemonic::Inc);

        if (max_insts - executed) >= 8 && (try_mov || try_neg || try_movzx) {
            if try_mov && match_ntos_name_hash(cpu, bus, ip, decoded) {
                match exec_ntos_name_hash(cpu, bus, max_insts - executed) {
                    Ok(0) => {}
                    Ok(n) => {
                        executed += n;
                        continue;
                    }
                    Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e)
                    {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                }
            }
            if let Some(spec) = try_neg
                .then(|| match_huffman_walk(cpu, bus, ip, bitness, &bytes[..available], decoded))
                .flatten()
            {
                match exec_huffman_walk(cpu, bus, &spec, max_insts - executed) {
                    Ok(0) => {}
                    Ok(n) => {
                        executed += n;
                        continue;
                    }
                    Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e)
                    {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                }
            }
            if let Some(spec) = try_movzx
                .then(|| match_lzx_bit_refill(cpu, bus, ip, bitness, &bytes[..available], decoded))
                .flatten()
            {
                match exec_lzx_bit_refill(cpu, bus, &spec, max_insts - executed) {
                    Ok(0) => {}
                    Ok(n) => {
                        executed += n;
                        continue;
                    }
                    Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e)
                    {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                }
            }
            if let Some(spec) = try_mov
                .then(|| match_lzx_copy_loop(cpu, bus, ip, bitness, &bytes[..available], decoded))
                .flatten()
            {
                match exec_lzx_copy_loop(cpu, bus, &spec, max_insts - executed) {
                    Ok(0) => {}
                    Ok(n) => {
                        executed += n;
                        continue;
                    }
                    Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e)
                    {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                }
            }
        }

        if try_inc && (max_insts - executed) >= 4 {
            if let Some(spec) =
                match_byte_scan_loop(ip, bitness, &bytes[..available], decoded)
            {
                match exec_byte_scan_loop(cpu, bus, &spec, max_insts - executed) {
                    Ok(0) => {}
                    Ok(n) => {
                        executed += n;
                        continue;
                    }
                    Err((fault_rip, e)) => match deliver_or_abort_exception(cpu, bus, fault_rip, e)
                    {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    },
                }
            }
        }

        if let Some((from, to)) = trace_window() {
            let insn_idx = INSN_COUNTER.fetch_add(1, Ordering::Relaxed);
            if (from..=to).contains(&insn_idx) {
                let st = &cpu.state;
                if trace_compact() {
                    // Fast path for whole-run control-flow capture (diffed against a QEMU
                    // reference): just the linear fetch address per instruction.
                    let linear = st.seg_base_reg(Register::CS).wrapping_add(ip);
                    eprintln!("{linear:#x}");
                } else {
                    let nbytes = &bytes[..(decoded.len as usize).min(15)];
                    let mut hex = String::with_capacity(nbytes.len() * 2);
                    for b in nbytes {
                        hex.push_str(&format!("{b:02x}"));
                    }
                    let ss_base = st.seg_base_reg(Register::SS);
                    let esp = st.read_reg(Register::RSP);
                    let eax = st.read_reg(Register::RAX);
                    let ebx = st.read_reg(Register::RBX);
                    let ecx = st.read_reg(Register::RCX);
                    let edx = st.read_reg(Register::RDX);
                    let ebp = st.read_reg(Register::RBP);
                    let esi = st.read_reg(Register::RSI);
                    let edi = st.read_reg(Register::RDI);
                    let fl = st.rflags();
                    let mode = st.mode;
                    eprintln!(
                        "[{insn_idx}] {mode:?}{bitness} {cs_base:#06x}:{ip:#06x} {hex:<30} ssb={ss_base:#07x} esp={esp:#x} eax={eax:#x} ebx={ebx:#x} ecx={ecx:#x} edx={edx:#x} ebp={ebp:#x} esi={esi:#x} edi={edi:#x} fl={fl:#x}"
                    );
                }
            }
        }

        let next_ip_raw = ip.wrapping_add(decoded.len as u64);
        let next_ip = next_ip_raw & mask_bits(cpu.state.bitness());

        let outcome = match exec_decoded(
            cfg,
            &mut cpu.state,
            bus,
            decoded,
            next_ip,
            addr_size_override,
        ) {
            Ok(v) => v,
            Err(e) => {
                let e = if matches!(e, Exception::InvalidOpcode)
                    && is_x87_opcode(&bytes, bitness)
                {
                    match super::check_fp_available(&cpu.state, crate::fpu::FpKind::X87) {
                        Err(fp_e) => fp_e,
                        Ok(()) => e,
                    }
                } else {
                    e
                };
                match deliver_or_abort_exception(cpu, bus, ip, e) {
                    Ok(()) => continue,
                    Err(exit) => {
                        return BatchResult {
                            executed,
                            exit,
                        };
                    }
                }
            }
        };

        match outcome {
            ExecOutcome::Continue => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
            }
            ExecOutcome::ContinueInhibitInterrupts => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                cpu.pending.inhibit_interrupts_for_one_instruction();
            }
            ExecOutcome::Branch => {
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                // Only end the batch on a branch when the caller asked for it
                // (`exit_on_branch`). `Machine::run_slice` clears that flag so
                // branch-dense boot code does not force a full device re-poll +
                // paging-bus rebuild after nearly every branch — measured as the
                // dominant native-interpreter cost (~95% of cycles vs ~5% for
                // decode+exec). Interrupts are still checked at the top of every
                // iteration, and real exits (halt/assist/exception/completed) still
                // end the batch, so correctness is unchanged either way.
                if cfg.exit_on_branch {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }
                // Block boundary: RIP is the branch target.
                if let Some(seam) = seam.as_deref_mut() {
                    let budget = max_insts.saturating_sub(executed);
                    if budget > 0 {
                        let erased = bus as *mut B as *mut ();
                        // See the note at the resume-point offer above.
                        drop(cache);
                        let ran = seam.offer(cpu, erased, ctx as *mut AssistContext, budget);
                        cache = cell.borrow_mut();
                        executed = executed.saturating_add(ran);
                    }
                }
            }
            ExecOutcome::Halt => {
                cpu.state.set_rip(next_ip);
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                if let Some(vector) = cpu.state.take_pending_bios_int() {
                    return BatchResult {
                        executed,
                        exit: BatchExit::BiosInterrupt(vector),
                    };
                }
                cpu.state.halted = true;
                return BatchResult {
                    executed,
                    exit: BatchExit::Halted,
                };
            }
            ExecOutcome::Assist(reason) => {
                if reason == AssistReason::Interrupt {
                    let assist_outcome = match interrupts::exec_interrupt_assist_decoded(
                        cpu,
                        bus,
                        decoded,
                        addr_size_override,
                    ) {
                        Ok(outcome) => outcome,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit: BatchExit::CpuExit(exit),
                            };
                        }
                    };

                    match assist_outcome {
                        interrupts::InterruptAssistOutcome::Retired {
                            block_boundary,
                            inhibit_interrupts,
                        } => {
                            executed += 1;
                            cpu.pending.retire_instruction();
                            cpu.time.advance_cycles(1);
                            cpu.state.msr.tsc = cpu.time.read_tsc();
                            if inhibit_interrupts {
                                cpu.pending.inhibit_interrupts_for_one_instruction();
                            }
                            if block_boundary {
                                return BatchResult {
                                    executed,
                                    exit: BatchExit::Branch,
                                };
                            }
                            continue;
                        }
                        interrupts::InterruptAssistOutcome::FaultDelivered => {
                            return BatchResult {
                                executed,
                                exit: BatchExit::Branch,
                            };
                        }
                    }
                }

                let inhibits_interrupt =
                    matches!(decoded.instr.mnemonic(), Mnemonic::Mov | Mnemonic::Pop)
                        && decoded.instr.op_count() > 0
                        && decoded.instr.op_kind(0) == OpKind::Register
                        && decoded.instr.op0_register() == Register::SS;

                if let Err(e) = handle_assist_decoded(
                    ctx,
                    &mut cpu.time,
                    &mut cpu.state,
                    bus,
                    decoded,
                    addr_size_override,
                ) {
                    match deliver_or_abort_exception(cpu, bus, ip, e) {
                        Ok(()) => continue,
                        Err(exit) => {
                            return BatchResult {
                                executed,
                                exit,
                            };
                        }
                    }
                }
                executed += 1;
                cpu.pending.retire_instruction();
                cpu.time.advance_cycles(1);
                cpu.state.msr.tsc = cpu.time.read_tsc();
                if inhibits_interrupt {
                    cpu.pending.inhibit_interrupts_for_one_instruction();
                }

                if bus.take_execution_yield_request() {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }

                // Preserve the "basic block" behavior of `run_batch`: treat any
                // control-transfer assist (i.e. RIP != fallthrough) as a branch.
                let expected_next = next_ip_raw & mask_bits(cpu.state.bitness());
                if cpu.state.rip() != expected_next {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }

                // A tight RDTSC polling loop can otherwise keep HAL clock-sync work inside one
                // batch before RTC/PIC can raise IRQ8. Yield only after the assist recognizes
                // sustained polling; RDTSC itself remains a pure time sample.
                if matches!(
                    decoded.instr.mnemonic(),
                    Mnemonic::Rdtsc | Mnemonic::Rdtscp
                ) && crate::assist::rdtsc_assist_yields_for_timers(ctx)
                {
                    return BatchResult {
                        executed,
                        exit: BatchExit::Branch,
                    };
                }

                // An assist is where a compiled block gave up. Offer the seam here as
                // well as at branches: RIP is now the instruction *after* the one the
                // compiled tier could not translate, which is exactly where it could
                // start again. Without this the interpreter carries on to the next
                // taken branch — about thirteen instructions each time, and the
                // instructions a compiled tier does not cover are what its remaining
                // cost is mostly made of.
                if seam_after_assist() {
                  if let Some(seam) = seam.as_deref_mut() {
                    let budget = max_insts.saturating_sub(executed);
                    if budget > 0 {
                        let erased = bus as *mut B as *mut ();
                        // See the note at the resume-point offer above.
                        drop(cache);
                        let ran = seam.offer(cpu, erased, ctx as *mut AssistContext, budget);
                        cache = cell.borrow_mut();
                        executed = executed.saturating_add(ran);
                    }
                  }
                }
            }
        }
    }

    BatchResult {
        executed,
        exit: BatchExit::Completed,
    }
    })
}

#[cfg(test)]
mod lzx_bit_refill_match_tests {
    use super::*;
    use crate::mem::FlatTestBus;
    use crate::state::CpuMode;

    const VERBATIM: [u8; 34] = [
        0x41, 0x0f, 0xb6, 0x54, 0x24, 0x01, 0x41, 0x0f, 0xb6, 0x04, 0x24, 0x40, 0x0f, 0xbe, 0xcf,
        0xc1, 0xe2, 0x08, 0xf7, 0xd9, 0x49, 0x83, 0xc4, 0x02, 0x0b, 0xd0, 0xd3, 0xe2, 0x0b, 0xda,
        0x40, 0x80, 0xc7, 0x10,
    ];

    #[test]
    fn bcrypt_sha512_fuse_runs_with_queued_irq() {
        // A queued but undelivered IRQ used to Ok(0) at the entry prefix; the
        // next insn left the match window and CryptoSysPrep interpreted the
        // whole leaf. Completing the hash with IRQs pending is correct.
        let leaf = include_bytes!("../../../tests/data/bcryptprimitives_sha512_compress.bin");
        let mut cpu = CpuCore::new(CpuMode::Long);
        let mut bus = FlatTestBus::new(0x1_0000);
        bus.load(0xb000, &leaf[..32]);
        // FIPS IV + empty-message padded block (0x80 at [0], bitlen 0)
        let iv: [u64; 8] = [
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
            0x510e527fade682d1,
            0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179,
        ];
        for (i, w) in iv.iter().enumerate() {
            bus.write_u64(0x2000 + (i as u64) * 8, *w).unwrap();
        }
        let mut block = [0u8; 128];
        block[0] = 0x80;
        bus.load(0x3000, &block);
        bus.write_u64(0x7ff8, 0x10_000).unwrap(); // ret target
        cpu.state.set_rip(0xb000);
        cpu.state.write_reg(Register::RCX, 0x2000);
        cpu.state.write_reg(Register::RDX, 0x3000);
        cpu.state.write_reg(Register::RSP, 0x7ff8);
        cpu.pending.inject_external_interrupt(0xD1);
        assert!(match_bcrypt_sha512_compress(
            &cpu,
            &mut bus,
            0xb000,
            &leaf[..15]
        ));
        let n = exec_bcrypt_sha512_compress(&mut cpu, &mut bus, 0xb000, 1).unwrap();
        assert_eq!(n, BCRYPT_SHA512_COMPRESS_INSNS);
        assert_eq!(cpu.state.rip(), 0x10_000);
        assert_eq!(cpu.state.read_reg(Register::RSP), 0x8000);
        // SHA-512("")
        let expect: [u64; 8] = [
            0xcf83e1357eefb8bd,
            0xf1542850d66d8007,
            0xd620e4050b5715dc,
            0x83f4a921d36ce9ce,
            0x47d0d13c5d85f2b0,
            0xff8318d2877eec2f,
            0x63b931bd47417a81,
            0xa538327af927da3e,
        ];
        for (i, w) in expect.iter().enumerate() {
            assert_eq!(bus.read_u64(0x2000 + (i as u64) * 8).unwrap(), *w);
        }
    }

    #[test]
    fn bcrypt_sha512_prefix_matches_guest_leaf() {
        let leaf = include_bytes!("../../../tests/data/bcryptprimitives_sha512_compress.bin");
        assert!(leaf.starts_with(&BCRYPT_SHA512_COMPRESS_PREFIX));
        let mut cpu = CpuCore::new(CpuMode::Long);
        let mut bus = FlatTestBus::new(0x1_0000);
        bus.load(0xb000, &leaf[..BCRYPT_SHA512_COMPRESS_PREFIX.len()]);
        cpu.state.set_rip(0xb000);
        assert!(match_bcrypt_sha512_compress(
            &cpu,
            &mut bus,
            0xb000,
            &leaf[..15]
        ));
        let mut miss = BCRYPT_SHA512_COMPRESS_PREFIX;
        miss[16] ^= 1;
        bus.load(0xb000, &miss);
        assert!(!match_bcrypt_sha512_compress(
            &cpu,
            &mut bus,
            0xb000,
            &miss[..15]
        ));
        // Same push frame, different stack alloc — the 102b false-positive shape.
        let mut almost = BCRYPT_SHA512_COMPRESS_PREFIX;
        almost[20] = 0x90; // sub rsp, 90h not 98h
        bus.load(0xb000, &almost);
        assert!(
            !match_bcrypt_sha512_compress(&cpu, &mut bus, 0xb000, &almost[..15]),
            "must not match a generic home-space prologue"
        );
    }

    #[test]
    fn matches_kernelbase_nls_upcase_bytes() {
        let mut cpu = CpuCore::new(CpuMode::Long);
        let mut bus = FlatTestBus::new(0x4000);
        bus.load(0x1000, &KERNELBASE_NLS_UPCASE);
        cpu.state.set_rip(0x1000);
        assert!(match_kernelbase_nls_upcase(
            &cpu,
            &mut bus,
            0x1000,
            &KERNELBASE_NLS_UPCASE[..15]
        ));
        let mut miss = KERNELBASE_NLS_UPCASE;
        miss[10] ^= 1;
        bus.load(0x1000, &miss);
        assert!(!match_kernelbase_nls_upcase(
            &cpu,
            &mut bus,
            0x1000,
            &miss[..15]
        ));
    }

    #[test]
    fn kernelbase_nls_upcase_zero_table_is_identity() {
        let mut cpu = CpuCore::new(CpuMode::Long);
        let mut bus = FlatTestBus::new(0x8000);
        bus.load(0x1000, &KERNELBASE_NLS_UPCASE);
        // input string
        let input: [u16; 4] = [0x0061, 0x0041, 0x00e9, 0x2122];
        let mut raw = [0u8; 8];
        for (i, ch) in input.iter().enumerate() {
            raw[i * 2..i * 2 + 2].copy_from_slice(&ch.to_le_bytes());
        }
        bus.load(0x3000, &raw);
        // zero NLS table: every lookup is 0, so dest = src
        bus.load(0x4000, &[0u8; 512]);
        cpu.state.set_rip(0x1000);
        cpu.state.write_reg(Register::R9, 0x3000);
        cpu.state.write_reg(Register::R10, 0x4000);
        cpu.state.write_reg(Register::R11, 4);
        cpu.state.write_reg(Register::EDI, 0);
        let n = exec_kernelbase_nls_upcase(&mut cpu, &mut bus, 0x1000, 200).unwrap();
        assert_eq!(n, 19 * 4);
        assert_eq!(cpu.state.rip(), 0x1000 + KERNELBASE_NLS_UPCASE.len() as u64);
        assert_eq!(cpu.state.read_reg(Register::R11), 0);
        assert_eq!(cpu.state.read_reg(Register::R9), 0x3008);
        let out = bus.slice(0x3000, 8);
        assert_eq!(out, raw.as_slice());
    }

    #[test]
    fn matches_winsetup_verbatim_refill_bytes() {
        let mut cpu = CpuCore::new(CpuMode::Long);
        let mut bus = FlatTestBus::new(0x2000);
        bus.load(0x1000, &VERBATIM);
        cpu.state.set_rip(0x1000);
        let first = aero_x86::decode(&VERBATIM, 0x1000, 64).unwrap();
        let spec = match_lzx_bit_refill(&cpu, &mut bus, 0x1000, 64, &VERBATIM, &first)
            .expect("verbatim refill must match");
        assert_eq!(spec.insts, 10);
        assert_eq!(spec.src, Register::R12);
        assert_eq!(spec.hi, Register::EDX);
        assert_eq!(spec.lo, Register::EAX);
        assert_eq!(spec.bits, Register::EBX);
        assert_eq!(spec.bitcount, Register::DIL);
        assert_eq!(spec.fallthrough, 0x1000 + VERBATIM.len() as u64);
    }
}

/// Whether the compiled tier is offered the instruction after an assist.
///
/// See the call site. Off, and measured: the idea is right — the instruction after
/// the one a compiled block could not translate is exactly where it could start
/// again — but the points are diverse and mostly cold, so it admits blocks that do
/// not repay their compilation. 187.4 against 184.4 host instructions per guest
/// instruction. `AERO_CT_SEAM_AFTER_ASSIST=1` turns it on.
fn seam_after_assist() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| std::env::var_os("AERO_CT_SEAM_AFTER_ASSIST").is_some());
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

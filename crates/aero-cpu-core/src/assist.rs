use aero_x86::{DecodedInst, Instruction, Mnemonic, OpKind, Register};

use crate::cpuid::{self, CpuFeatures};
use crate::exception::{AssistReason, Exception};
use crate::linear_mem::{
    read_u16_wrapped, read_u32_wrapped, read_u64_wrapped, watch_read, write_u16_wrapped,
    write_u32_wrapped, write_u64_wrapped,
};
use crate::mem::CpuBus;
use crate::msr;
use crate::segmentation::{LoadReason, Seg};
use crate::state::{
    mask_bits, CpuMode, CpuState, RFLAGS_IF, RFLAGS_IOPL_MASK, RFLAGS_ZF, SEG_ACCESS_DB,
    SEG_ACCESS_L, SEG_ACCESS_PRESENT,
};
use crate::time::TimeSource;

/// Maximum number of `INVLPG` addresses retained in [`AssistContext::invlpg_log`].
///
/// This log is only intended for tests/debugging; bounding it prevents unbounded
/// memory growth in long-running test workloads that execute many `INVLPG`s.
pub const MAX_INVLPG_LOG_ENTRIES: usize = 4096;

// Backwards-compat alias for the constant name used in older discussions/specs.
// (The "B" is a typo, but keeping it avoids churn in downstream code/tests.)
pub const MAX_INVLPGB_LOG_ENTRIES: usize = MAX_INVLPG_LOG_ENTRIES;

/// Runtime context needed by the Tier-0 assist layer.
///
/// Tier-0 is intentionally minimal and does not model system state like MSRs,
/// descriptor tables, or deterministic time. When it encounters an instruction
/// that depends on that state, it exits with [`AssistReason`]. The caller feeds
/// those exits into [`handle_assist`], which emulates the instruction against
/// the JIT ABI [`CpuState`].
#[derive(Debug, Clone, Default)]
pub struct AssistContext {
    /// CPUID feature policy used for `CPUID` and for masking MSR writes (e.g.
    /// keeping `IA32_EFER` coherent with advertised features).
    pub features: CpuFeatures,
    /// Optional log of `INVLPG` linear addresses (useful for integration tests).
    ///
    /// The log is bounded to [`MAX_INVLPG_LOG_ENTRIES`]. When the capacity is
    /// reached, new entries are dropped and
    /// [`AssistContext::dropped_invlpg_log_entries`] is incremented.
    pub invlpg_log: Vec<u64>,
    /// Number of `INVLPG` log entries dropped because [`AssistContext::invlpg_log`] hit
    /// [`AssistContext::INVLPG_LOG_CAP`].
    pub dropped_invlpg_log_entries: u64,
    /// Last instruction pointer that executed `RDTSC`/`RDTSCP`.
    ///
    /// This is scheduler-only history used to recognize a tight timestamp polling loop before
    /// applying the opt-in `AERO_RDTSC_QUANTUM` scheduler yield. A one-off timestamp read must
    /// remain an ordinary sample: Windows also uses RDTSC as entropy, and advancing time on every
    /// sample changes calibration, allocator placement, and boot correctness.
    rdtsc_poll_rip: Option<u64>,
    /// Architectural TSC returned by the preceding timestamp read.
    rdtsc_poll_tsc: u64,
    /// Number of consecutive timestamp reads from the same instruction pointer with a small
    /// instruction-time gap.
    rdtsc_poll_streak: u8,
    /// Whether the most recent timestamp assist requested a polling-loop scheduler yield.
    rdtsc_poll_yield_requested: bool,
}

impl AssistContext {
    /// Hard cap on the number of `INVLPG` addresses recorded in [`AssistContext::invlpg_log`].
    ///
    /// This log is a debug/testing facility and must never grow without bound (guest kernels can
    /// execute `INVLPG` frequently).
    pub const INVLPG_LOG_CAP: usize = MAX_INVLPG_LOG_ENTRIES;

    #[inline]
    pub fn invlpg_log_dropped(&self) -> u64 {
        self.dropped_invlpg_log_entries
    }

    #[inline]
    pub fn clear_invlpg_log(&mut self) {
        self.invlpg_log.clear();
        self.dropped_invlpg_log_entries = 0;
    }

    #[inline]
    fn record_invlpg(&mut self, addr: u64) {
        if self.invlpg_log.len() < Self::INVLPG_LOG_CAP {
            self.invlpg_log.push(addr);
        } else {
            self.dropped_invlpg_log_entries = self.dropped_invlpg_log_entries.saturating_add(1);
        }
    }
}

/// Execute a Tier-0 assist exit.
///
/// The assist handler:
/// - fetches + decodes the faulting instruction at the current RIP
/// - executes its architectural semantics using [`CpuState`]
/// - advances RIP (or transfers control) so the caller can resume execution
pub fn handle_assist<B: CpuBus>(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    bus: &mut B,
    _reason: AssistReason,
) -> Result<(), Exception> {
    // Keep the bus paging/MMU view coherent with architectural state even when
    // `handle_assist` is used outside of `tier0::exec::step` / `run_batch_with_assists`.
    bus.sync(state);
    let ip = state.rip();
    let cs_base = state.seg_base_reg(Register::CS);
    let (bytes, decoded) =
        crate::interp::tier0::exec::fetch_and_decode(state, bus, cs_base, ip, aero_x86::decode)
            .inspect_err(|e| state.apply_exception_side_effects(e))?;
    let addr_size_override = has_addr_size_override(&bytes, state.bitness());

    exec_decoded(ctx, time, state, bus, &decoded, addr_size_override)
        .inspect_err(|e| state.apply_exception_side_effects(e))?;
    // Assists can update paging-related state (CR0/CR3/CR4/EFER/CPL). Sync the
    // bus so the next instruction boundary observes those updates even if the
    // caller doesn't go through `tier0::exec::step` immediately.
    bus.sync(state);
    Ok(())
}

/// How many decodes the assist path remembers. A compiled block hands the same
/// instruction over every time it runs — 1.56M assists came from 898 blocks in one
/// session, so roughly 1,700 visits each — and decoding is a large part of what an
/// assist costs.
const ASSIST_DECODE_ENTRIES: usize = 1024;

#[derive(Clone)]
struct AssistDecodeEntry {
    rip: u64,
    /// The bytes the entry was decoded from, so a page that has been rewritten under
    /// the same address is not decoded from memory that has since changed.
    key: u64,
    decoded: aero_x86::DecodedInst,
}

thread_local! {
    static ASSIST_DECODE: core::cell::RefCell<Vec<Option<AssistDecodeEntry>>> =
        core::cell::RefCell::new(vec![None; ASSIST_DECODE_ENTRIES]);
}

/// Pack the first eight bytes of the fetch window, as the Tier-0 decode cache does.
fn assist_decode_key(bytes: &[u8]) -> u64 {
    let mut k = 0u64;
    for (i, b) in bytes.iter().take(8).enumerate() {
        k |= u64::from(*b) << (i * 8);
    }
    k
}

fn assist_decode(
    bytes: &[u8],
    ip: u64,
    bitness: u32,
) -> Result<aero_x86::DecodedInst, aero_x86::DecodeError> {
    let key = assist_decode_key(bytes);
    let slot = (ip as usize >> 2) & (ASSIST_DECODE_ENTRIES - 1);
    let hit = ASSIST_DECODE.with(|c| {
        c.borrow()[slot]
            .as_ref()
            .filter(|e| e.rip == ip && e.key == key)
            .map(|e| e.decoded.clone())
    });
    if let Some(d) = hit {
        return Ok(d);
    }
    let decoded = aero_x86::decode(bytes, ip, bitness)?;
    ASSIST_DECODE.with(|c| {
        c.borrow_mut()[slot] = Some(AssistDecodeEntry {
            rip: ip,
            key,
            decoded: decoded.clone(),
        });
    });
    Ok(decoded)
}

/// Execute exactly one guest instruction, whatever kind it is.
///
/// The compiled tier calls this for an instruction Tier-1 cannot translate, from the
/// middle of a compiled block: the block has written the architectural state out, this
/// advances it by one instruction, and the block picks it up and carries on. Which is
/// why it has to handle *any* instruction and not only the assist mnemonics — Tier-1
/// refuses plenty that Tier-0 executes directly, `bsf` and `movnti` among them.
///
/// Returns the RIP the guest ended up at. `None` means the instruction did not
/// complete — an exception, a halt, a BIOS interrupt — and the caller must leave the
/// architectural state alone and let the interpreter own it. Interrupt-related
/// assists are refused for the same reason: delivering one is Tier-0's business, at
/// the point in its loop where it does that.
pub fn assist_one_instruction<B: CpuBus>(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    bus: &mut B,
) -> Option<u64> {
    use crate::exception::AssistReason;
    use crate::interp::tier0::exec::{step_with_config_and_decoder, StepExit};
    use crate::interp::tier0::Tier0Config;

    let cfg = Tier0Config::from_cpuid(&ctx.features);
    match step_with_config_and_decoder(&cfg, state, bus, assist_decode) {
        Ok(StepExit::Continue | StepExit::ContinueInhibitInterrupts | StepExit::Branch) => {
            Some(state.rip())
        }
        Ok(StepExit::Assist {
            reason,
            decoded,
            addr_size_override,
        }) => {
            if reason == AssistReason::Interrupt {
                return None;
            }
            handle_assist_decoded(ctx, time, state, bus, &decoded, addr_size_override).ok()?;
            Some(state.rip())
        }
        Ok(StepExit::Halted | StepExit::BiosInterrupt(_)) | Err(_) => None,
    }
}

/// Execute an assist using a pre-decoded instruction.
///
/// This is used by Tier-0 execution glue that already fetched/decoded the
/// instruction bytes and wants to avoid an extra decode pass. Callers should
/// also supply the already-parsed address-size override prefix state so assists
/// like `INS*`/`OUTS*` can pick the correct implicit index/counter registers.
pub fn handle_assist_decoded<B: CpuBus>(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    bus: &mut B,
    decoded: &DecodedInst,
    addr_size_override: bool,
) -> Result<(), Exception> {
    exec_decoded(ctx, time, state, bus, decoded, addr_size_override)
        .inspect_err(|e| state.apply_exception_side_effects(e))
}

fn exec_decoded<B: CpuBus>(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    bus: &mut B,
    decoded: &DecodedInst,
    addr_size_override: bool,
) -> Result<(), Exception> {
    let instr = &decoded.instr;
    let ip = state.rip();
    let next_ip_raw = ip.wrapping_add(decoded.len as u64);

    match instr.mnemonic() {
        Mnemonic::Cpuid => {
            instr_cpuid(ctx, state);
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Rdmsr => {
            instr_rdmsr(ctx, time, state)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Wrmsr => {
            instr_wrmsr(ctx, time, state)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Rdtsc => {
            instr_rdtsc(ctx, time, state);
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Rdtscp => {
            instr_rdtscp(ctx, time, state);
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Lfence | Mnemonic::Sfence | Mnemonic::Mfence => {
            // Tier-0 uses assists for fence opcodes so the runtime can model them
            // as serializing points if needed. For now they are treated as NOPs.
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Pause => {
            // NOP with a spin-loop hint.
            core::hint::spin_loop();
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::In | Mnemonic::Out => {
            instr_in_out(time, state, bus, instr)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Insb | Mnemonic::Insw | Mnemonic::Insd => {
            instr_ins(state, bus, instr, addr_size_override)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Outsb | Mnemonic::Outsw | Mnemonic::Outsd => {
            instr_outs(state, bus, instr, addr_size_override)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Cli
        | Mnemonic::Sti
        | Mnemonic::Int
        | Mnemonic::Int1
        | Mnemonic::Int3
        | Mnemonic::Into
        | Mnemonic::Iret
        | Mnemonic::Iretd
        | Mnemonic::Iretq => Err(Exception::Unimplemented(
            "interrupt assist requires CpuCore",
        )),
        Mnemonic::Mov => {
            instr_mov_privileged(ctx, state, bus, instr, next_ip_raw)?;
            Ok(())
        }
        Mnemonic::Pop => {
            instr_pop_privileged(ctx, state, bus, instr)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Jmp => {
            instr_jmp_far(ctx, state, bus, instr, next_ip_raw)?;
            Ok(())
        }
        Mnemonic::Call => {
            instr_call_far(ctx, state, bus, instr, next_ip_raw)?;
            Ok(())
        }
        Mnemonic::Retf => {
            instr_retf(ctx, state, bus, instr)?;
            Ok(())
        }
        Mnemonic::Lgdt | Mnemonic::Lidt => {
            instr_lgdt_lidt(state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Sgdt | Mnemonic::Sidt => {
            instr_sgdt_sidt(state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Ltr | Mnemonic::Lldt => {
            instr_ltr_lldt(ctx, state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Str | Mnemonic::Sldt => {
            instr_str_sldt(state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Lar | Mnemonic::Lsl => {
            instr_lar_lsl(state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Lmsw | Mnemonic::Smsw => {
            instr_lmsw_smsw(state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Invlpg => {
            instr_invlpg(ctx, state, bus, instr, next_ip_raw)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        // WBINVD / INVD: privileged cache flushes. Emulators without a host
        // cache model treat them as CPL0 no-ops. Win7 PAGELK issues WBINVD
        // while programming MTRRs immediately before SSDT conversion.
        Mnemonic::Wbinvd | Mnemonic::Invd => {
            require_cpl0(state)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Swapgs => {
            instr_swapgs(state)?;
            state.set_rip(next_ip_raw);
            Ok(())
        }
        Mnemonic::Syscall => {
            instr_syscall(ctx, state, next_ip_raw)?;
            Ok(())
        }
        Mnemonic::Sysret | Mnemonic::Sysretq => {
            // iced-x86 maps legacy SYSRET (0F 07) to Sysret and REX.W SYSRETQ
            // (48 0F 07) to Sysretq. Win7 x64 only uses SYSRETQ; treating it as
            // #UD left GS user after SWAPGS and nested #PF → TripleFault.
            instr_sysret(ctx, state)?;
            Ok(())
        }
        Mnemonic::Sysenter => {
            instr_sysenter(state)?;
            Ok(())
        }
        Mnemonic::Sysexit => {
            instr_sysexit(state)?;
            Ok(())
        }
        Mnemonic::Rsm => Err(Exception::InvalidOpcode),
        _ => Err(Exception::InvalidOpcode),
    }
}

// -------------------------------------------------------------------------------------------------
// Basic helpers
// -------------------------------------------------------------------------------------------------

fn require_cpl0(state: &CpuState) -> Result<(), Exception> {
    if state.cpl() != 0 {
        return Err(Exception::gp0());
    }
    Ok(())
}

fn require_iopl(state: &CpuState) -> Result<(), Exception> {
    let cpl = state.cpl();
    let iopl = ((state.rflags() & RFLAGS_IOPL_MASK) >> 12) as u8;
    if cpl > iopl {
        return Err(Exception::gp0());
    }
    Ok(())
}

fn is_seg_reg(reg: Register) -> bool {
    matches!(
        reg,
        Register::ES | Register::CS | Register::SS | Register::DS | Register::FS | Register::GS
    )
}

fn seg_from_reg(reg: Register) -> Result<Seg, Exception> {
    Ok(match reg {
        Register::ES => Seg::ES,
        Register::CS => Seg::CS,
        Register::SS => Seg::SS,
        Register::DS => Seg::DS,
        Register::FS => Seg::FS,
        Register::GS => Seg::GS,
        _ => return Err(Exception::InvalidOpcode),
    })
}

fn is_ctrl_reg(reg: Register) -> bool {
    matches!(
        reg,
        Register::CR0 | Register::CR2 | Register::CR3 | Register::CR4 | Register::CR8
    )
}

fn is_debug_reg(reg: Register) -> bool {
    matches!(
        reg,
        Register::DR0
            | Register::DR1
            | Register::DR2
            | Register::DR3
            | Register::DR6
            | Register::DR7
    )
}

fn reg_bits(reg: Register) -> Result<u32, Exception> {
    use Register::*;
    Ok(match reg {
        AL | CL | DL | BL | AH | CH | DH | BH | SPL | BPL | SIL | DIL | R8L | R9L | R10L | R11L
        | R12L | R13L | R14L | R15L => 8,
        AX | CX | DX | BX | SP | BP | SI | DI | R8W | R9W | R10W | R11W | R12W | R13W | R14W
        | R15W => 16,
        EAX | ECX | EDX | EBX | ESP | EBP | ESI | EDI | R8D | R9D | R10D | R11D | R12D | R13D
        | R14D | R15D => 32,
        RAX | RCX | RDX | RBX | RSP | RBP | RSI | RDI | R8 | R9 | R10 | R11 | R12 | R13 | R14
        | R15 => 64,
        _ => return Err(Exception::InvalidOpcode),
    })
}

fn io_size_from_reg(reg: Register) -> Result<u32, Exception> {
    let bits = reg_bits(reg)?;
    Ok(bits / 8)
}

fn calc_ea(
    state: &CpuState,
    instr: &Instruction,
    next_ip: u64,
    include_seg: bool,
) -> Result<u64, Exception> {
    let base = instr.memory_base();
    let index = instr.memory_index();
    let scale = instr.memory_index_scale() as u64;
    let mut disp = instr.memory_displacement64() as i128;
    if base == Register::RIP {
        disp -= next_ip as i128;
    }

    let addr_bits = if base == Register::RIP {
        64
    } else if base != Register::None {
        reg_bits(base)?
    } else if index != Register::None {
        reg_bits(index)?
    } else {
        match instr.memory_displ_size() {
            2 => 16,
            4 => 32,
            8 => 64,
            _ => state.bitness(),
        }
    };

    let mut offset: i128 = disp;
    if base != Register::None {
        let base_val = if base == Register::RIP {
            next_ip
        } else {
            state.read_reg(base)
        };
        offset += (base_val & mask_bits(addr_bits)) as i128;
    }
    if index != Register::None {
        let idx_val = state.read_reg(index) & mask_bits(addr_bits);
        offset += (idx_val as i128) * (scale as i128);
    }

    let addr = (offset as u64) & mask_bits(addr_bits);
    if include_seg {
        Ok(state.apply_a20(
            state
                .seg_base_reg(instr.memory_segment())
                .wrapping_add(addr),
        ))
    } else {
        Ok(addr)
    }
}

fn read_mem<B: CpuBus>(
    state: &CpuState,
    bus: &mut B,
    addr: u64,
    bits: u32,
) -> Result<u64, Exception> {
    match bits {
        8 => {
            let v = bus.read_u8(state.apply_a20(addr))?;
            watch_read(state, addr, 1, u64::from(v));
            Ok(u64::from(v))
        }
        16 => Ok(read_u16_wrapped(state, bus, addr)? as u64),
        32 => Ok(read_u32_wrapped(state, bus, addr)? as u64),
        64 => Ok(read_u64_wrapped(state, bus, addr)?),
        _ => Err(Exception::InvalidOpcode),
    }
}

fn write_mem<B: CpuBus>(
    state: &CpuState,
    bus: &mut B,
    addr: u64,
    bits: u32,
    val: u64,
) -> Result<(), Exception> {
    match bits {
        8 => bus.write_u8(state.apply_a20(addr), val as u8),
        16 => write_u16_wrapped(state, bus, addr, val as u16),
        32 => write_u32_wrapped(state, bus, addr, val as u32),
        64 => write_u64_wrapped(state, bus, addr, val),
        _ => Err(Exception::InvalidOpcode),
    }
}

fn read_op_u16<B: CpuBus>(
    state: &CpuState,
    bus: &mut B,
    instr: &Instruction,
    op: u32,
    next_ip: u64,
) -> Result<u16, Exception> {
    match instr.op_kind(op) {
        OpKind::Register => Ok(state.read_reg(instr.op_register(op)) as u16),
        OpKind::Memory => {
            let addr = calc_ea(state, instr, next_ip, true)?;
            read_u16_wrapped(state, bus, addr)
        }
        OpKind::Immediate16 => Ok(instr.immediate16()),
        OpKind::Immediate8to16 => Ok(instr.immediate8to16() as u16),
        _ => Err(Exception::InvalidOpcode),
    }
}

fn write_op_u16<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    op: u32,
    value: u16,
    next_ip: u64,
) -> Result<(), Exception> {
    match instr.op_kind(op) {
        OpKind::Register => {
            state.write_reg(instr.op_register(op), value as u64);
            Ok(())
        }
        OpKind::Memory => {
            let addr = calc_ea(state, instr, next_ip, true)?;
            write_u16_wrapped(state, bus, addr, value)
        }
        _ => Err(Exception::InvalidOpcode),
    }
}

// -------------------------------------------------------------------------------------------------
// CPUID
// -------------------------------------------------------------------------------------------------

fn instr_cpuid(ctx: &AssistContext, state: &mut CpuState) {
    let leaf = state.read_reg(Register::EAX) as u32;
    let subleaf = state.read_reg(Register::ECX) as u32;
    let res = cpuid::cpuid(&ctx.features, leaf, subleaf);
    state.write_reg(Register::EAX, res.eax as u64);
    state.write_reg(Register::EBX, res.ebx as u64);
    state.write_reg(Register::ECX, res.ecx as u64);
    state.write_reg(Register::EDX, res.edx as u64);
}

// -------------------------------------------------------------------------------------------------
// MSRs
// -------------------------------------------------------------------------------------------------

fn msr_read(
    _ctx: &AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    msr_index: u32,
) -> Result<u64, Exception> {
    match msr_index {
        msr::IA32_TSC => {
            let tsc = time.read_tsc();
            state.msr.tsc = tsc;
            Ok(tsc)
        }
        _ => state.msr.read(msr_index),
    }
}

fn msr_write(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
    msr_index: u32,
    value: u64,
) -> Result<(), Exception> {
    match msr_index {
        msr::IA32_EFER => {
            state.msr.write(&ctx.features, msr_index, value)?;
            state.update_mode();
            Ok(())
        }
        msr::IA32_FS_BASE => {
            state.msr.write(&ctx.features, msr_index, value)?;
            if state.mode == CpuMode::Long {
                state.segments.fs.base = value;
            }
            Ok(())
        }
        msr::IA32_GS_BASE => {
            state.msr.write(&ctx.features, msr_index, value)?;
            if state.mode == CpuMode::Long {
                state.segments.gs.base = value;
            }
            Ok(())
        }
        msr::IA32_TSC => {
            time.set_tsc(value);
            state.msr.tsc = value;
            Ok(())
        }
        _ => state.msr.write(&ctx.features, msr_index, value),
    }
}

fn instr_rdmsr(
    ctx: &AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    let msr_index = state.read_reg(Register::ECX) as u32;
    let value = msr_read(ctx, time, state, msr_index)?;
    state.write_reg(Register::EAX, value as u32 as u64);
    state.write_reg(Register::EDX, (value >> 32) as u32 as u64);
    Ok(())
}

fn instr_wrmsr(
    ctx: &mut AssistContext,
    time: &mut TimeSource,
    state: &mut CpuState,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    let msr_index = state.read_reg(Register::ECX) as u32;
    let eax = state.read_reg(Register::EAX) as u32 as u64;
    let edx = state.read_reg(Register::EDX) as u32 as u64;
    let value = (edx << 32) | eax;
    msr_write(ctx, time, state, msr_index, value)?;
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Time
// -------------------------------------------------------------------------------------------------

/// Allocate one line from the bounded timestamp/PM-timer diagnostic stream.
///
/// `AERO_TIME_TRACE=<lines>` is intentionally off by default. It records the
/// architectural samples Windows uses while calibrating the TSC without
/// enabling the much heavier whole-instruction trace.
fn time_trace_slot() -> Option<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static LIMIT: OnceLock<u64> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("AERO_TIME_TRACE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    });
    if limit == 0 {
        return None;
    }
    let slot = NEXT.fetch_add(1, Ordering::Relaxed);
    (slot < limit).then_some(slot)
}

/// Whether sustained interruptible RDTSC polling should yield to the machine scheduler.
///
/// The numeric `AERO_RDTSC_QUANTUM` spelling is retained for compatibility with existing
/// bring-up recipes, but the value is now only an on/off control. Earlier code added it to the
/// architectural TSC on every timestamp read, corrupting Windows' TSC calibration and entropy.
fn rdtsc_poll_yield_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AERO_RDTSC_QUANTUM")
            .ok()
            .and_then(|s| s.parse().ok())
            .is_some_and(|value: u64| value != 0)
    })
}

/// Whether the most recent RDTSC assist recognized a tight polling loop and requested a scheduler
/// yield.
///
/// Tier-0 yields so `Machine::run_slice` can tick RTC/PIT and poll IRQ8 before a HAL
/// RDTSC-bounded clock-sync wait expires entirely inside one batch (0x5C / 0x10B). RDTSC itself
/// remains a pure architectural sample; it never advances time. Isolated timestamp/entropy
/// samples do not force a yield.
#[inline]
pub(crate) fn rdtsc_assist_yields_for_timers(ctx: &AssistContext) -> bool {
    ctx.rdtsc_poll_yield_requested
}

/// Maximum architectural-time gap between samples that can belong to one tight timestamp polling
/// loop. Real HAL delay/calibration loops return to the same RDTSC within a few dozen retired
/// instructions; calls separated by ordinary kernel work must not be classified as polling.
const RDTSC_POLL_MAX_GAP_CYCLES: u64 = 64;

/// Require several consecutive tight-loop samples before yielding. This keeps short bursts of
/// timestamp reads cheap while adding negligible latency to long waits.
const RDTSC_POLL_WARMUP_SAMPLES: u8 = 8;

fn update_rdtsc_poll_yield(ctx: &mut AssistContext, time: &mut TimeSource, state: &CpuState) {
    let yield_enabled = rdtsc_poll_yield_enabled();
    let rip = state.rip();
    let before = time.read_tsc();
    let tight_repeat = ctx.rdtsc_poll_rip == Some(rip)
        && before.wrapping_sub(ctx.rdtsc_poll_tsc) <= RDTSC_POLL_MAX_GAP_CYCLES;

    ctx.rdtsc_poll_streak = if tight_repeat {
        ctx.rdtsc_poll_streak.saturating_add(1)
    } else {
        1
    };
    // The optimization exists to let platform timers and IRQ delivery progress during an
    // interruptible timestamp-bounded wait. Yielding while IF=0 cannot let an ISR satisfy the
    // wait and just adds host overhead.
    ctx.rdtsc_poll_yield_requested = yield_enabled
        && (state.rflags() & RFLAGS_IF) != 0
        && ctx.rdtsc_poll_streak >= RDTSC_POLL_WARMUP_SAMPLES;
    ctx.rdtsc_poll_rip = Some(rip);
    ctx.rdtsc_poll_tsc = time.read_tsc();
}

fn instr_rdtsc(ctx: &mut AssistContext, time: &mut TimeSource, state: &mut CpuState) {
    update_rdtsc_poll_yield(ctx, time, state);
    let tsc = time.read_tsc();
    if let Some(slot) = time_trace_slot() {
        eprintln!(
            "AERO_TIME_TRACE[{slot}] RDTSC rip={:#x} tsc={tsc:#x}",
            state.rip()
        );
    }
    state.msr.tsc = tsc;
    state.write_reg(Register::EAX, tsc as u32 as u64);
    state.write_reg(Register::EDX, (tsc >> 32) as u32 as u64);
}

fn instr_rdtscp(ctx: &mut AssistContext, time: &mut TimeSource, state: &mut CpuState) {
    update_rdtsc_poll_yield(ctx, time, state);
    let tsc = time.read_tsc();
    if let Some(slot) = time_trace_slot() {
        eprintln!(
            "AERO_TIME_TRACE[{slot}] RDTSCP rip={:#x} tsc={tsc:#x}",
            state.rip()
        );
    }
    state.msr.tsc = tsc;
    state.write_reg(Register::EAX, tsc as u32 as u64);
    state.write_reg(Register::EDX, (tsc >> 32) as u32 as u64);
    state.write_reg(Register::ECX, state.msr.tsc_aux as u64);
}

// -------------------------------------------------------------------------------------------------
// Port I/O
// -------------------------------------------------------------------------------------------------

fn instr_in_out<B: CpuBus>(
    time: &mut TimeSource,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
) -> Result<(), Exception> {
    require_iopl(state)?;
    match instr.mnemonic() {
        Mnemonic::In => {
            let dst = instr.op0_register();
            let size = io_size_from_reg(dst)?;
            let port = match instr.op_kind(1) {
                OpKind::Immediate8 => instr.immediate8() as u16,
                OpKind::Register => state.read_reg(instr.op1_register()) as u16,
                _ => return Err(Exception::InvalidOpcode),
            };
            let val = bus.io_read(port, size)?;
            state.write_reg(dst, val);
            let tsc_before = time.read_tsc();
            let extra_cycles = bus.io_read_time_advance_cycles(port, size);
            time.advance_cycles(extra_cycles);
            if port == 0x0408 {
                if let Some(slot) = time_trace_slot() {
                    eprintln!(
                        "AERO_TIME_TRACE[{slot}] PM_TMR rip={:#x} size={size} value={val:#x} tsc_before={tsc_before:#x} extra={extra_cycles:#x} tsc_after={:#x}",
                        state.rip(),
                        time.read_tsc()
                    );
                }
            }
            Ok(())
        }
        Mnemonic::Out => {
            let port = match instr.op_kind(0) {
                OpKind::Immediate8 => instr.immediate8() as u16,
                OpKind::Register => state.read_reg(instr.op0_register()) as u16,
                _ => return Err(Exception::InvalidOpcode),
            };
            let src = instr.op1_register();
            let size = io_size_from_reg(src)?;
            let val = state.read_reg(src);
            bus.io_write(port, size, val)?;
            Ok(())
        }
        _ => Err(Exception::InvalidOpcode),
    }
}

pub(crate) fn has_addr_size_override(bytes: &[u8; 15], bitness: u32) -> bool {
    let mut i = 0usize;
    let mut seen = false;
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
        if b == 0x67 {
            seen = true;
        }
        i += 1;
    }
    seen
}

fn effective_addr_size(state: &CpuState, addr_size_override: bool) -> Result<u32, Exception> {
    Ok(match state.bitness() {
        16 => {
            if addr_size_override {
                32
            } else {
                16
            }
        }
        32 => {
            if addr_size_override {
                16
            } else {
                32
            }
        }
        64 => {
            if addr_size_override {
                32
            } else {
                64
            }
        }
        _ => return Err(Exception::InvalidOpcode),
    })
}

fn string_count_reg(addr_bits: u32) -> Register {
    match addr_bits {
        16 => Register::CX,
        32 => Register::ECX,
        _ => Register::RCX,
    }
}

fn string_src_index_reg(addr_bits: u32) -> Register {
    match addr_bits {
        16 => Register::SI,
        32 => Register::ESI,
        _ => Register::RSI,
    }
}

fn string_dst_index_reg(addr_bits: u32) -> Register {
    match addr_bits {
        16 => Register::DI,
        32 => Register::EDI,
        _ => Register::RDI,
    }
}

fn instr_ins<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    addr_size_override: bool,
) -> Result<(), Exception> {
    require_iopl(state)?;
    let size = match instr.mnemonic() {
        Mnemonic::Insb => 1,
        Mnemonic::Insw => 2,
        Mnemonic::Insd => 4,
        _ => return Err(Exception::InvalidOpcode),
    };

    let port = state.read_reg(Register::DX) as u16;
    let df = state.get_flag(crate::state::RFLAGS_DF);
    let step: i64 = if df { -(size as i64) } else { size as i64 };

    let addr_size = effective_addr_size(state, addr_size_override)?;
    let count_reg = string_count_reg(addr_size);
    let index_reg = string_dst_index_reg(addr_size);
    let addr_mask = mask_bits(addr_size);

    let has_rep = instr.has_rep_prefix() || instr.has_repne_prefix();
    let mut count = if has_rep {
        state.read_reg(count_reg)
    } else {
        1
    };
    if has_rep && count == 0 {
        return Ok(());
    }
    let mut di = state.read_reg(index_reg) & addr_mask;
    let seg_base = state.seg_base_reg(Register::ES);

    while count != 0 {
        let val = bus.io_read(port, size)?;
        let addr = state.apply_a20(seg_base.wrapping_add(di & addr_mask));
        match size {
            1 => bus.write_u8(addr, val as u8)?,
            2 => write_u16_wrapped(state, bus, addr, val as u16)?,
            4 => write_u32_wrapped(state, bus, addr, val as u32)?,
            _ => unreachable!(),
        }
        di = (di as i64).wrapping_add(step) as u64;
        di &= addr_mask;
        if has_rep {
            count = count.wrapping_sub(1);
        } else {
            break;
        }
    }

    state.write_reg(index_reg, di);
    if has_rep {
        state.write_reg(count_reg, count);
    }
    Ok(())
}

fn instr_outs<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    addr_size_override: bool,
) -> Result<(), Exception> {
    require_iopl(state)?;
    let size = match instr.mnemonic() {
        Mnemonic::Outsb => 1,
        Mnemonic::Outsw => 2,
        Mnemonic::Outsd => 4,
        _ => return Err(Exception::InvalidOpcode),
    };

    let port = state.read_reg(Register::DX) as u16;
    let df = state.get_flag(crate::state::RFLAGS_DF);
    let step: i64 = if df { -(size as i64) } else { size as i64 };

    let addr_size = effective_addr_size(state, addr_size_override)?;
    let count_reg = string_count_reg(addr_size);
    let index_reg = string_src_index_reg(addr_size);
    let addr_mask = mask_bits(addr_size);

    let has_rep = instr.has_rep_prefix() || instr.has_repne_prefix();
    let mut count = if has_rep {
        state.read_reg(count_reg)
    } else {
        1
    };
    if has_rep && count == 0 {
        return Ok(());
    }
    let mut si = state.read_reg(index_reg) & addr_mask;
    let seg = instr.segment_prefix();
    let seg_reg = if seg == Register::None {
        Register::DS
    } else {
        seg
    };
    let seg_base = state.seg_base_reg(seg_reg);

    while count != 0 {
        let addr = state.apply_a20(seg_base.wrapping_add(si & addr_mask));
        let val: u64 = match size {
            1 => bus.read_u8(addr)? as u64,
            2 => read_u16_wrapped(state, bus, addr)? as u64,
            4 => read_u32_wrapped(state, bus, addr)? as u64,
            _ => unreachable!(),
        };
        bus.io_write(port, size, val)?;
        si = (si as i64).wrapping_add(step) as u64;
        si &= addr_mask;
        if has_rep {
            count = count.wrapping_sub(1);
        } else {
            break;
        }
    }

    state.write_reg(index_reg, si);
    if has_rep {
        state.write_reg(count_reg, count);
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Stack helpers
// -------------------------------------------------------------------------------------------------

fn push_u16<B: CpuBus>(state: &mut CpuState, bus: &mut B, val: u16) -> Result<(), Exception> {
    push_sized(state, bus, val as u64, 2)
}

fn pop_u16<B: CpuBus>(state: &mut CpuState, bus: &mut B) -> Result<u16, Exception> {
    Ok(pop_sized(state, bus, 2)? as u16)
}

fn set_real_mode_seg(seg: &mut crate::state::Segment, selector: u16) {
    seg.selector = selector;
    seg.base = (selector as u64) << 4;
    seg.limit = 0xFFFF;
    seg.access = 0;
}

// -------------------------------------------------------------------------------------------------
// Privileged MOV / POP / segment loads
// -------------------------------------------------------------------------------------------------

fn instr_mov_privileged<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    // Segment loads in protected/long mode.
    if instr.op_kind(0) == OpKind::Register && is_seg_reg(instr.op0_register()) {
        let seg = instr.op0_register();
        if seg == Register::CS {
            return Err(Exception::InvalidOpcode);
        }
        let selector = read_op_u16(state, bus, instr, 1, next_ip)?;
        let seg_enum = seg_from_reg(seg)?;
        let reason = if seg_enum == Seg::SS {
            LoadReason::Stack
        } else {
            LoadReason::Data
        };
        state.load_seg(bus, seg_enum, selector, reason)?;
        state.set_rip(next_ip);
        return Ok(());
    }

    // MOV to/from control/debug registers.
    if instr.op_kind(0) == OpKind::Register
        && (is_ctrl_reg(instr.op0_register()) || is_debug_reg(instr.op0_register()))
        || instr.op_kind(1) == OpKind::Register
            && (is_ctrl_reg(instr.op1_register()) || is_debug_reg(instr.op1_register()))
    {
        instr_mov_cr_dr(state, bus, instr)?;
        state.set_rip(next_ip);
        return Ok(());
    }

    Err(Exception::InvalidOpcode)
}

fn instr_mov_cr_dr<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    let dst = instr.op0_register();
    let src = instr.op1_register();
    if is_ctrl_reg(dst) {
        // mov crX, reg
        let val = state.read_reg(src);
        match dst {
            Register::CR0 => state.control.cr0 = val,
            Register::CR2 => state.control.cr2 = val,
            Register::CR3 => {
                state.control.cr3 = val;
                bus.write_cr3(val);
            }
            Register::CR4 => state.control.cr4 = val,
            Register::CR8 => state.control.cr8 = val,
            _ => return Err(Exception::InvalidOpcode),
        }
        state.update_mode();
        return Ok(());
    }
    if is_ctrl_reg(src) {
        // mov reg, crX
        let val = match src {
            Register::CR0 => state.control.cr0,
            Register::CR2 => state.control.cr2,
            Register::CR3 => state.control.cr3,
            Register::CR4 => state.control.cr4,
            Register::CR8 => state.control.cr8,
            _ => return Err(Exception::InvalidOpcode),
        };
        state.write_reg(dst, val);
        return Ok(());
    }

    if is_debug_reg(dst) {
        let val = state.read_reg(src);
        match dst {
            Register::DR0 => state.debug.dr[0] = val,
            Register::DR1 => state.debug.dr[1] = val,
            Register::DR2 => state.debug.dr[2] = val,
            Register::DR3 => state.debug.dr[3] = val,
            Register::DR6 => state.debug.dr6 = val,
            Register::DR7 => state.debug.dr7 = val,
            _ => return Err(Exception::InvalidOpcode),
        }
        return Ok(());
    }
    if is_debug_reg(src) {
        let val = match src {
            Register::DR0 => state.debug.dr[0],
            Register::DR1 => state.debug.dr[1],
            Register::DR2 => state.debug.dr[2],
            Register::DR3 => state.debug.dr[3],
            Register::DR6 => state.debug.dr6,
            Register::DR7 => state.debug.dr7,
            _ => return Err(Exception::InvalidOpcode),
        };
        state.write_reg(dst, val);
        return Ok(());
    }

    Err(Exception::InvalidOpcode)
}

fn instr_pop_privileged<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
) -> Result<(), Exception> {
    if instr.op_kind(0) != OpKind::Register || !is_seg_reg(instr.op0_register()) {
        return Err(Exception::InvalidOpcode);
    }
    let seg = instr.op0_register();
    if seg == Register::CS {
        return Err(Exception::InvalidOpcode);
    }
    // POP segment always pops a 16-bit selector, even in 32-bit mode.
    let selector = pop_u16(state, bus)?;
    let seg_enum = seg_from_reg(seg)?;
    let reason = if seg_enum == Seg::SS {
        LoadReason::Stack
    } else {
        LoadReason::Data
    };
    state.load_seg(bus, seg_enum, selector, reason)
}

// -------------------------------------------------------------------------------------------------
// Far control transfers (real + protected mode)
// -------------------------------------------------------------------------------------------------

fn instr_jmp_far<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    match instr.op_kind(0) {
        OpKind::FarBranch16 => {
            let sel = instr.far_branch_selector();
            let off = instr.far_branch16();
            far_jump(state, bus, sel, off as u64)
        }
        OpKind::FarBranch32 => {
            let sel = instr.far_branch_selector();
            let off = instr.far_branch32();
            far_jump(state, bus, sel, off as u64)
        }
        _ => {
            // Near JMP should have been handled by Tier-0 already.
            state.set_rip(next_ip);
            Ok(())
        }
    }
}

fn instr_call_far<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    match instr.op_kind(0) {
        OpKind::FarBranch16 => {
            let sel = instr.far_branch_selector();
            let off = instr.far_branch16() as u64;
            far_call(state, bus, sel, off, next_ip)
        }
        OpKind::FarBranch32 => {
            let sel = instr.far_branch_selector();
            let off = instr.far_branch32() as u64;
            far_call(state, bus, sel, off, next_ip)
        }
        _ => Err(Exception::InvalidOpcode),
    }
}

fn instr_retf<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
) -> Result<(), Exception> {
    let pop_imm = if instr.op_count() == 1 && instr.op_kind(0) == OpKind::Immediate16 {
        instr.immediate16() as u32
    } else {
        0
    };

    // The number of bytes RETF pops for the offset and for the CS selector slot is the *operand
    // size*, not the code-segment default: a 0x66 prefix toggles it in 16/32-bit mode (e.g. `66 cb`
    // in a 16-bit code segment is RETFD, popping a 4-byte offset and a 4-byte selector slot).
    // Deriving it from `state.bitness()` would mis-split the return frame whenever 0x66 is used.
    // Decode it from the instruction's Code, matching the real-mode path in ops_cf.
    let off_size = match instr.code() {
        aero_x86::Code::Retfd | aero_x86::Code::Retfd_imm16 => 4,
        aero_x86::Code::Retfq | aero_x86::Code::Retfq_imm16 => 8,
        _ => 2,
    };
    let off_bits = off_size * 8;
    let off = pop_sized(state, bus, off_size)? & mask_bits(off_bits);
    // The CS selector slot is `off_size` bytes wide (selector in the low word, upper bytes
    // discarded); reading the full slot keeps the stack pointer correctly aligned.
    let cs_slot = pop_sized(state, bus, off_size)?;
    let cs = (cs_slot & 0xFFFF) as u16;
    if std::env::var_os("AERO_SEG_DEBUG").is_some() {
        eprintln!(
            "[retf] popped off={off:#x} cs={cs:#06x} (slot={cs_slot:#x}, sp={:#x}, cpl={})",
            state.stack_ptr(),
            state.cpl(),
        );
    }
    let sp = state.stack_ptr().wrapping_add(pop_imm as u64);
    state.set_stack_ptr(sp);
    far_jump(state, bus, cs, off)
}

fn push_sized<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    val: u64,
    size: u32,
) -> Result<(), Exception> {
    // Same atomicity rule as tier0 `push`: only commit RSP after the store
    // succeeds so a stack #PF leaves the instruction restartable.
    let sp_bits = state.stack_ptr_bits();
    let new_sp = state.stack_ptr().wrapping_sub(size as u64) & mask_bits(sp_bits);
    let addr = state.apply_a20(state.seg_base_reg(Register::SS).wrapping_add(new_sp));
    write_mem(state, bus, addr, size * 8, val)?;
    state.set_stack_ptr(new_sp);
    Ok(())
}

fn pop_sized<B: CpuBus>(state: &mut CpuState, bus: &mut B, size: u32) -> Result<u64, Exception> {
    let sp_bits = state.stack_ptr_bits();
    let sp = state.stack_ptr();
    let addr = state.apply_a20(state.seg_base_reg(Register::SS).wrapping_add(sp));
    let v = read_mem(state, bus, addr, size * 8)?;
    let next = sp.wrapping_add(size as u64) & mask_bits(sp_bits);
    state.set_stack_ptr(next);
    Ok(v)
}

fn far_jump<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    selector: u16,
    offset: u64,
) -> Result<(), Exception> {
    match state.mode {
        CpuMode::Real | CpuMode::Vm86 => {
            set_real_mode_seg(&mut state.segments.cs, selector);
            state.set_rip(offset);
            Ok(())
        }
        CpuMode::Protected | CpuMode::Long => {
            state.load_seg(bus, Seg::CS, selector, LoadReason::FarControlTransfer)?;
            state.set_rip(offset);
            Ok(())
        }
    }
}

fn far_call<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    selector: u16,
    offset: u64,
    return_ip: u64,
) -> Result<(), Exception> {
    // Push current CS then return IP.
    let cs = state.segments.cs.selector;
    push_u16(state, bus, cs)?;
    let ret_size = state.bitness() / 8;
    push_sized(state, bus, return_ip, ret_size)?;
    far_jump(state, bus, selector, offset)
}

// -------------------------------------------------------------------------------------------------
// Descriptor table ops
// -------------------------------------------------------------------------------------------------

fn instr_lgdt_lidt<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    if instr.op_kind(0) != OpKind::Memory {
        return Err(Exception::InvalidOpcode);
    }
    let addr = calc_ea(state, instr, next_ip, true)?;
    let limit = read_u16_wrapped(state, bus, addr)?;
    let base = match state.bitness() {
        64 => read_u64_wrapped(state, bus, addr.wrapping_add(2))?,
        16 => (read_u32_wrapped(state, bus, addr.wrapping_add(2))? & 0x00FF_FFFF) as u64,
        _ => read_u32_wrapped(state, bus, addr.wrapping_add(2))? as u64,
    };
    match instr.mnemonic() {
        Mnemonic::Lgdt => {
            state.tables.gdtr.limit = limit;
            state.tables.gdtr.base = base;
        }
        Mnemonic::Lidt => {
            state.tables.idtr.limit = limit;
            state.tables.idtr.base = base;
        }
        _ => return Err(Exception::InvalidOpcode),
    }
    Ok(())
}

fn instr_sgdt_sidt<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    if instr.op_kind(0) != OpKind::Memory {
        return Err(Exception::InvalidOpcode);
    }
    let addr = calc_ea(state, instr, next_ip, true)?;
    let (limit, base) = match instr.mnemonic() {
        Mnemonic::Sgdt => (state.tables.gdtr.limit, state.tables.gdtr.base),
        Mnemonic::Sidt => (state.tables.idtr.limit, state.tables.idtr.base),
        _ => return Err(Exception::InvalidOpcode),
    };
    write_u16_wrapped(state, bus, addr, limit)?;
    match state.bitness() {
        64 => write_u64_wrapped(state, bus, addr.wrapping_add(2), base)?,
        16 => write_u32_wrapped(state, bus, addr.wrapping_add(2), base as u32 & 0x00FF_FFFF)?,
        _ => write_u32_wrapped(state, bus, addr.wrapping_add(2), base as u32)?,
    }
    Ok(())
}

fn instr_ltr_lldt<B: CpuBus>(
    _ctx: &mut AssistContext,
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    let selector = read_op_u16(state, bus, instr, 0, next_ip)?;
    match instr.mnemonic() {
        Mnemonic::Ltr => state.load_tr(bus, selector),
        Mnemonic::Lldt => state.load_ldtr(bus, selector),
        _ => Err(Exception::InvalidOpcode),
    }
}

fn instr_str_sldt<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    let selector = match instr.mnemonic() {
        Mnemonic::Str => state.tables.tr.selector,
        Mnemonic::Sldt => state.tables.ldtr.selector,
        _ => return Err(Exception::InvalidOpcode),
    };
    write_op_u16(state, bus, instr, 0, selector, next_ip)
}

/// LAR/LSL: load access rights / segment limit from a selector.
///
/// Soft-fault semantics (Intel SDM Vol. 2): invalid selector, bad type, or
/// privilege failure clears ZF and does **not** raise #GP. Destination is left
/// unchanged when ZF=0. Used by Win7 ntdll in user mode (`LSL EAX, EAX`).
fn instr_lar_lsl<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    // Operand 1 is the source selector (r/m16); operand 0 is the destination.
    let selector = read_op_u16(state, bus, instr, 1, next_ip)?;
    let null = (selector >> 3) == 0;

    let result = if null {
        None
    } else {
        lar_lsl_probe(state, bus, selector, instr.mnemonic())
    };

    match result {
        Some(value) => {
            // `write_reg` applies 16/32/64 write semantics from the iced register
            // id (32-bit dest clears the upper half in long mode).
            match instr.op_kind(0) {
                OpKind::Register => state.write_reg(instr.op0_register(), value),
                _ => return Err(Exception::InvalidOpcode),
            }
            state.set_flag(RFLAGS_ZF, true);
        }
        None => {
            state.set_flag(RFLAGS_ZF, false);
        }
    }
    Ok(())
}

/// Probe GDT/LDT for LAR/LSL. Returns `Some(dest_value)` on success (ZF=1 path).
fn lar_lsl_probe<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    selector: u16,
    mnemonic: Mnemonic,
) -> Option<u64> {
    // Soft checks: anything that would be #GP for a hard load becomes ZF=0.
    let desc = soft_read_descriptor_8(state, bus, selector)?;
    let attrs = desc.attrs();
    if !attrs.present {
        return None;
    }
    if !lar_lsl_type_ok(attrs.s, attrs.typ) {
        return None;
    }
    // Privilege: conforming code is always ok; otherwise max(CPL,RPL) <= DPL.
    let rpl = (selector & 0b11) as u8;
    let cpl = state.cpl();
    let conforming = attrs.s && (attrs.typ & 0b1100) == 0b1100; // code + conforming
    if !conforming && (cpl > attrs.dpl || rpl > attrs.dpl) {
        return None;
    }

    match mnemonic {
        Mnemonic::Lsl => {
            // Segment limit already expanded for G bit by parse_descriptor_8.
            let limit = match desc {
                crate::descriptors::Descriptor::Segment(s) => s.limit,
                crate::descriptors::Descriptor::System(s) => s.limit,
            };
            Some(u64::from(limit))
        }
        Mnemonic::Lar => {
            // Access rights: bits 16..23 = access byte, 20..23 high flags partially.
            // Architectural LAR returns: hidden high word of descriptor (access rights
            // shifted): bits [23:16] = access, [15:0] = 0, with some high flags.
            // Classic form: DEST[23:0] = descriptor[55:40] << 8? Actually:
            //   DEST = (access_rights << 8) with bits 19:16 = flags low nibble
            // SDM: "DEST ← access rights" as the high 4 bytes of the descriptor
            // (bytes 4-7 of the 8-byte descriptor) with low 8 bits cleared:
            //   bits 15:0 = 0, bits 23:16 = access byte, bits 31:24 from flags area.
            // Effective: value = (raw_low >> 32) & 0x00FFFF00
            // We reconstruct from attrs.
            let access = (attrs.typ & 0xF)
                | (u8::from(attrs.s) << 4)
                | ((attrs.dpl & 3) << 5)
                | (u8::from(attrs.present) << 7);
            let flags = u8::from(attrs.avl)
                | (u8::from(attrs.long) << 1)
                | (u8::from(attrs.default_big) << 2)
                | (u8::from(attrs.granularity) << 3);
            // LAR result: bits[23:16]=access, bits[19:16] also include flags in
            // the upper nibble of the third byte — SDM packs as descriptor[55:40]
            // left-shifted by 8 with low byte zero:
            //   result = ((flags & 0xF) << 20) | ((access as u32) << 8)
            let rights = (u32::from(access) << 8) | (u32::from(flags & 0xF) << 20);
            Some(u64::from(rights))
        }
        _ => None,
    }
}

fn lar_lsl_type_ok(s: bool, typ: u8) -> bool {
    if s {
        // All code/data segment types.
        return true;
    }
    // System types valid for LAR/LSL (Intel SDM table).
    matches!(typ, 0x1 | 0x2 | 0x3 | 0x9 | 0xB)
}

/// Soft GDT/LDT read for LAR/LSL: returns None instead of #GP on OOB/null.
fn soft_read_descriptor_8<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    selector: u16,
) -> Option<crate::descriptors::Descriptor> {
    let index = selector >> 3;
    if index == 0 {
        return None;
    }
    let ti = (selector & 0b100) != 0;
    let (base, limit) = if ti {
        if state.tables.ldtr.is_unusable() {
            return None;
        }
        (state.tables.ldtr.base, state.tables.ldtr.limit)
    } else {
        (state.tables.gdtr.base, u32::from(state.tables.gdtr.limit))
    };
    let byte_off = u64::from(index) * 8;
    if byte_off + 7 > u64::from(limit) {
        return None;
    }
    let addr = base.wrapping_add(byte_off);
    let raw_low = state
        .with_supervisor_access(bus, |bus, st| read_u64_wrapped(st, bus, addr))
        .ok()?;
    Some(crate::descriptors::parse_descriptor_8(raw_low))
}

fn instr_lmsw_smsw<B: CpuBus>(
    state: &mut CpuState,
    bus: &mut B,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    match instr.mnemonic() {
        Mnemonic::Lmsw => {
            require_cpl0(state)?;
            let msw = read_op_u16(state, bus, instr, 0, next_ip)? as u64;
            let old = state.control.cr0;
            let mut next = (old & !0xF) | (msw & 0xF);
            // LMSW cannot clear PE once set.
            if (old & crate::state::CR0_PE) != 0 {
                next |= crate::state::CR0_PE;
            }
            state.control.cr0 = next;
            state.update_mode();
            Ok(())
        }
        Mnemonic::Smsw => {
            let msw = (state.control.cr0 & 0xFFFF) as u16;
            match instr.op_kind(0) {
                OpKind::Register => {
                    state.write_reg(instr.op0_register(), msw as u64);
                    Ok(())
                }
                OpKind::Memory => {
                    let addr = calc_ea(state, instr, next_ip, true)?;
                    write_u16_wrapped(state, bus, addr, msw)
                }
                _ => Err(Exception::InvalidOpcode),
            }
        }
        _ => Err(Exception::InvalidOpcode),
    }
}

fn instr_invlpg(
    ctx: &mut AssistContext,
    state: &CpuState,
    bus: &mut impl CpuBus,
    instr: &Instruction,
    next_ip: u64,
) -> Result<(), Exception> {
    require_cpl0(state)?;
    if instr.op_kind(0) != OpKind::Memory {
        return Err(Exception::InvalidOpcode);
    }
    // INVLPG takes a linear address operand. In non-long modes, linear addresses
    // are 32-bit and wrap around on overflow of `segment_base + offset`.
    let mut addr = calc_ea(state, instr, next_ip, true)?;
    if state.mode != CpuMode::Long {
        addr &= 0xffff_ffff;
    }
    bus.invlpg(addr);
    ctx.record_invlpg(addr);
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// SYSCALL/SYSRET/SYSENTER/SYSEXIT/SWAPGS
// -------------------------------------------------------------------------------------------------

/// Bring-up: `AERO_LOG_SYSCALL=aa,82` logs matching `eax` syscall numbers
/// (hex, optional `0x`) at the architectural `SYSCALL` assist, so JIT and
/// Tier-0 share one view. Caps at 32 lines per number.
fn log_filtered_syscall(state: &CpuState, return_ip: u64) {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::OnceLock;

    let want = SYSCALL_LOG_FILTER.get_or_init(parse_syscall_log_filter);
    if want.is_empty() {
        return;
    }
    let eax = state.read_reg(Register::RAX) as u32;
    if !want.contains(&eax) {
        return;
    }
    static COUNTS: OnceLock<Vec<AtomicU32>> = OnceLock::new();
    let counts = COUNTS.get_or_init(|| want.iter().map(|_| AtomicU32::new(0)).collect());
    let Some(idx) = want.iter().position(|&n| n == eax) else {
        return;
    };
    let n = counts[idx].fetch_add(1, Ordering::Relaxed);
    if n >= 32 {
        return;
    }
    eprintln!(
        "AERO_LOG_SYSCALL: eax={eax:#x} rip={return_ip:#x} cr3={:#x} rcx={:#x} rdx={:#x} r8={:#x} r9={:#x} rsp={:#x}",
        state.control.cr3,
        state.read_reg(Register::RCX),
        state.read_reg(Register::RDX),
        state.read_reg(Register::R8),
        state.read_reg(Register::R9),
        state.read_reg(Register::RSP),
    );
}

fn parse_syscall_log_filter() -> Vec<u32> {
    parse_syscall_log_filter_spec(std::env::var("AERO_LOG_SYSCALL").ok().as_deref())
}

fn parse_syscall_log_filter_spec(spec: Option<&str>) -> Vec<u32> {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return Vec::new();
    };
    spec.split(',')
        .filter_map(|raw| {
            let raw = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
            u32::from_str_radix(raw, 16).ok()
        })
        .collect()
}

#[cfg(test)]
mod syscall_log_filter_tests {
    use super::parse_syscall_log_filter_spec;

    #[test]
    fn parses_hex_ntcreateuserprocess_and_alpc() {
        let got = parse_syscall_log_filter_spec(Some("aa,0x82"));
        assert_eq!(
            got,
            vec![0xaa, 0x82],
            "Win7 NtCreateUserProcess=0xAA, NtAlpcSendWaitReceivePort=0x82"
        );
    }

    #[test]
    fn empty_or_unset_disables_the_log() {
        assert!(parse_syscall_log_filter_spec(None).is_empty());
        assert!(parse_syscall_log_filter_spec(Some("")).is_empty());
    }
}

static SYSCALL_LOG_FILTER: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

fn is_canonical(addr: u64, bits: u8) -> bool {
    if bits >= 64 {
        return true;
    }
    let sign_bit = 1u64 << (bits - 1);
    let mask = (!0u64) << bits;
    if (addr & sign_bit) == 0 {
        (addr & mask) == 0
    } else {
        (addr & mask) == mask
    }
}

fn instr_syscall(
    ctx: &AssistContext,
    state: &mut CpuState,
    return_ip: u64,
) -> Result<(), Exception> {
    if state.mode != CpuMode::Long {
        return Err(Exception::InvalidOpcode);
    }
    if (state.msr.efer & msr::EFER_SCE) == 0 {
        return Err(Exception::InvalidOpcode);
    }

    log_filtered_syscall(state, return_ip);

    state.write_reg(Register::RCX, return_ip);
    state.write_reg(Register::R11, state.rflags());

    let star = state.msr.star;
    let syscall_cs = ((star >> 32) & 0xFFFF) as u16;
    // SDM SYSCALL: load flat kernel CS/SS descriptor caches (CPL0, L=1 for CS).
    // Type=0xB (exec/read/accessed), S=1, DPL=0, P=1, L=1, G=1.
    const KERNEL_CS_AR: u32 = 0xB | (1 << 4) | SEG_ACCESS_PRESENT | SEG_ACCESS_L | (1 << 11);
    // Type=0x3 (read/write/accessed), S=1, DPL=0, P=1, B=1, G=1.
    const KERNEL_SS_AR: u32 = 0x3 | (1 << 4) | SEG_ACCESS_PRESENT | SEG_ACCESS_DB | (1 << 11);
    state.segments.cs.selector = syscall_cs & !0b11;
    state.segments.cs.base = 0;
    state.segments.cs.limit = 0xFFFFF;
    state.segments.cs.access = KERNEL_CS_AR;
    state.segments.ss.selector = syscall_cs.wrapping_add(8) & !0b11;
    state.segments.ss.base = 0;
    state.segments.ss.limit = 0xFFFFF;
    state.segments.ss.access = KERNEL_SS_AR;

    let fmask = state.msr.fmask;
    state.set_rflags(state.rflags() & !fmask);

    let target = state.msr.lstar;
    if !is_canonical(target, ctx.features.linear_address_bits) {
        return Err(Exception::gp0());
    }
    state.set_rip(target);
    Ok(())
}

fn instr_sysret(ctx: &AssistContext, state: &mut CpuState) -> Result<(), Exception> {
    require_cpl0(state)?;
    if state.mode != CpuMode::Long {
        return Err(Exception::InvalidOpcode);
    }
    if (state.msr.efer & msr::EFER_SCE) == 0 {
        return Err(Exception::InvalidOpcode);
    }

    let target = state.read_reg(Register::RCX);
    if !is_canonical(target, ctx.features.linear_address_bits) {
        return Err(Exception::gp0());
    }

    // SDM SYSRET (64-bit): RFLAGS ← (R11 & 0x3C7FD7) | 2  (RF/VM cleared, bit1 set).
    let r11 = state.read_reg(Register::R11);
    state.set_rflags((r11 & 0x3C7FD7) | 2);

    let star = state.msr.star;
    let base = ((star >> 48) & 0xFFFF) as u16;
    let user_cs = base.wrapping_add(16);
    let user_ss = base.wrapping_add(8);
    // SDM: fixed flat user CS/SS caches (DPL=3, CS.L=1, CS.D=0, SS.B=1).
    // Type=0xB (exec/read/accessed), S=1, DPL=3, P=1, L=1, G=1.
    const USER_CS_AR: u32 =
        0xB | (1 << 4) | (3 << 5) | SEG_ACCESS_PRESENT | SEG_ACCESS_L | (1 << 11);
    // Type=0x3 (read/write/accessed), S=1, DPL=3, P=1, B=1, G=1.
    const USER_SS_AR: u32 =
        0x3 | (1 << 4) | (3 << 5) | SEG_ACCESS_PRESENT | SEG_ACCESS_DB | (1 << 11);
    state.segments.cs.selector = (user_cs & !0b11) | 0b11;
    state.segments.cs.base = 0;
    state.segments.cs.limit = 0xFFFFF;
    state.segments.cs.access = USER_CS_AR;
    state.segments.ss.selector = (user_ss & !0b11) | 0b11;
    state.segments.ss.base = 0;
    state.segments.ss.limit = 0xFFFFF;
    state.segments.ss.access = USER_SS_AR;

    state.set_rip(target);
    Ok(())
}

fn instr_sysenter(state: &mut CpuState) -> Result<(), Exception> {
    if state.mode == CpuMode::Real || state.mode == CpuMode::Vm86 {
        return Err(Exception::InvalidOpcode);
    }

    let cs = state.msr.sysenter_cs as u16;
    if cs == 0 {
        return Err(Exception::gp0());
    }
    state.segments.cs.selector = cs & !0b11;
    state.segments.ss.selector = state.segments.cs.selector.wrapping_add(8) & !0b11;

    match state.mode {
        CpuMode::Protected => {
            state.write_reg(Register::ESP, state.msr.sysenter_esp);
            state.set_rip(state.msr.sysenter_eip);
        }
        CpuMode::Long => {
            state.write_reg(Register::RSP, state.msr.sysenter_esp);
            state.set_rip(state.msr.sysenter_eip);
        }
        _ => {}
    }
    Ok(())
}

fn instr_sysexit(state: &mut CpuState) -> Result<(), Exception> {
    require_cpl0(state)?;
    if state.mode == CpuMode::Real || state.mode == CpuMode::Vm86 {
        return Err(Exception::InvalidOpcode);
    }

    let cs = state.msr.sysenter_cs as u16;
    if cs == 0 {
        return Err(Exception::gp0());
    }
    let cs_base = cs & !0b11;
    state.segments.cs.selector = cs_base.wrapping_add(16) | 0b11;
    state.segments.ss.selector = cs_base.wrapping_add(24) | 0b11;

    match state.mode {
        CpuMode::Protected => {
            let new_rip = state.read_reg(Register::EDX) as u32 as u64;
            let new_rsp = state.read_reg(Register::ECX) as u32 as u64;
            state.write_reg(Register::ESP, new_rsp);
            state.set_rip(new_rip);
        }
        CpuMode::Long => {
            // SDM SYSEXIT (IA-32e): RIP ← RDX, RSP ← RCX. Matches the 32-bit
            // Protected mapping (RIP←EDX, RSP←ECX) above; the prior code had
            // these two swapped, which silently corrupts the return target.
            let new_rip = state.read_reg(Register::RDX);
            let new_rsp = state.read_reg(Register::RCX);
            state.write_reg(Register::RSP, new_rsp);
            state.set_rip(new_rip);
        }
        _ => {}
    }
    Ok(())
}

fn instr_swapgs(state: &mut CpuState) -> Result<(), Exception> {
    require_cpl0(state)?;
    if state.mode != CpuMode::Long {
        return Err(Exception::InvalidOpcode);
    }
    core::mem::swap(&mut state.msr.gs_base, &mut state.msr.kernel_gs_base);
    state.segments.gs.base = state.msr.gs_base;
    Ok(())
}

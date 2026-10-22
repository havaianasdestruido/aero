#![cfg(all(debug_assertions, not(target_arch = "wasm32")))]

mod tier1_common;

use aero_cpu_core::state::CpuState;
use aero_jit_x86::abi;
use aero_jit_x86::tier1::ir::interp::execute_block;
use aero_jit_x86::tier1::ir::{BinOp, GuestReg, IrBlock, IrBuilder, IrTerminator, ValueId};
use aero_jit_x86::tier1::{Tier1WasmCodegen, EXPORT_TIER1_BLOCK_FN};
use aero_jit_x86::{discover_block, translate_block, BlockLimits, Tier1Bus};
use aero_jit_x86::tier1::block::discover_block_mode;
use aero_jit_x86::wasm::{
    IMPORT_JIT_EXIT, IMPORT_MEMORY, IMPORT_MEM_READ_U16, IMPORT_MEM_READ_U32, IMPORT_MEM_READ_U64,
    IMPORT_MEM_READ_U8, IMPORT_MEM_WRITE_U16, IMPORT_MEM_WRITE_U32, IMPORT_MEM_WRITE_U64,
    IMPORT_MEM_WRITE_U8, IMPORT_MODULE, IMPORT_PAGE_FAULT, JIT_EXIT_SENTINEL_I64,
};
use aero_types::{Cond, Flag, FlagSet, Gpr, Width};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tier1_common::{write_cpu_to_wasm_bytes, CpuSnapshot, SimpleBus};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module, Store, TypedFunc};

const CPU_PTR: i32 = 0x1_0000;
const JIT_CTX_PTR: i32 = CPU_PTR + abi::CPU_STATE_SIZE as i32;

// Page 0 is guest RAM. Keep it one page so we can cheaply compare the entire region and still
// catch stray/out-of-window writes due to codegen bugs.
const RAM_SIZE: usize = 0x1_0000;

// Keep random load/store addresses in a smaller window so we can safely generate in-bounds
// accesses without having to prove address arithmetic properties.
const RAM_WINDOW_BASE: u64 = 0x2000;
const RAM_WINDOW_LEN: u64 = 0x1000;

const MAX_IR_INSTS: usize = 50;
const ITERS: usize = 300;
const SEED: u64 = 0x6c0a_9f3a_77d5_1d2b;

fn validate_wasm(bytes: &[u8]) {
    let mut validator = wasmparser::Validator::new();
    validator.validate_all(bytes).unwrap();
}

fn define_mem_helpers(store: &mut Store<()>, linker: &mut Linker<()>, memory: Memory) {
    fn read<const N: usize>(caller: &mut Caller<'_, ()>, memory: &Memory, addr: usize) -> u64 {
        let mut buf = [0u8; N];
        memory
            .read(caller, addr, &mut buf)
            .expect("memory read in bounds");
        let mut v = 0u64;
        for (i, b) in buf.iter().enumerate() {
            v |= (*b as u64) << (i * 8);
        }
        v
    }

    fn write<const N: usize>(
        caller: &mut Caller<'_, ()>,
        memory: &Memory,
        addr: usize,
        value: u64,
    ) {
        let mut buf = vec![0u8; N];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (value >> (i * 8)) as u8;
        }
        memory
            .write(caller, addr, &buf)
            .expect("memory write in bounds");
    }

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_READ_U8,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64| -> i32 {
                    read::<1>(&mut caller, &mem, addr as usize) as i32
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_READ_U16,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64| -> i32 {
                    read::<2>(&mut caller, &mem, addr as usize) as i32
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_READ_U32,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64| -> i32 {
                    read::<4>(&mut caller, &mem, addr as usize) as i32
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_READ_U64,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64| -> i64 {
                    read::<8>(&mut caller, &mem, addr as usize) as i64
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_WRITE_U8,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64, value: i32| {
                    write::<1>(&mut caller, &mem, addr as usize, value as u64);
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_WRITE_U16,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64, value: i32| {
                    write::<2>(&mut caller, &mem, addr as usize, value as u64);
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_WRITE_U32,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64, value: i32| {
                    write::<4>(&mut caller, &mem, addr as usize, value as u64);
                },
            ),
        )
        .unwrap();

    let mem = memory;
    linker
        .define(
            IMPORT_MODULE,
            IMPORT_MEM_WRITE_U64,
            Func::wrap(
                &mut *store,
                move |mut caller: Caller<'_, ()>, _cpu_ptr: i32, addr: i64, value: i64| {
                    write::<8>(&mut caller, &mem, addr as usize, value as u64);
                },
            ),
        )
        .unwrap();
}

fn instantiate(engine: &Engine, bytes: &[u8]) -> (Store<()>, Memory, TypedFunc<(i32, i32), i64>) {
    let module = Module::new(engine, bytes).unwrap();

    let mut store = Store::new(engine, ());
    let mut linker = Linker::new(engine);

    // Guest memory in page 0, CpuState at CPU_PTR in page 1, and room for the JIT context.
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
    linker.define(IMPORT_MODULE, IMPORT_MEMORY, memory).unwrap();

    define_mem_helpers(&mut store, &mut linker, memory);

    linker
        .define(
            IMPORT_MODULE,
            IMPORT_PAGE_FAULT,
            Func::wrap(
                &mut store,
                |_caller: Caller<'_, ()>, _cpu_ptr: i32, _addr: i64| -> i64 {
                    panic!("page_fault should not be called by tier1_random_ir_equiv tests");
                },
            ),
        )
        .unwrap();

    linker
        .define(
            IMPORT_MODULE,
            IMPORT_JIT_EXIT,
            Func::wrap(
                &mut store,
                |_caller: Caller<'_, ()>, _kind: i32, rip: i64| -> i64 { rip },
            ),
        )
        .unwrap();

    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let block = instance
        .get_typed_func::<(i32, i32), i64>(&store, EXPORT_TIER1_BLOCK_FN)
        .unwrap();
    (store, memory, block)
}

fn run_wasm(
    engine: &Engine,
    ir: &IrBlock,
    cpu: &CpuState,
    bus: &SimpleBus,
) -> (i64, CpuSnapshot, Vec<u8>) {
    let wasm = Tier1WasmCodegen::new().compile_block(ir);
    validate_wasm(&wasm);

    let (mut store, memory, func) = instantiate(engine, &wasm);

    // Initialize guest memory.
    memory.write(&mut store, 0, bus.mem()).unwrap();

    // Initialize CpuState.
    let mut cpu_bytes = vec![0u8; abi::CPU_STATE_SIZE as usize];
    write_cpu_to_wasm_bytes(cpu, &mut cpu_bytes);
    memory
        .write(&mut store, CPU_PTR as usize, &cpu_bytes)
        .unwrap();

    let ret = func.call(&mut store, (CPU_PTR, JIT_CTX_PTR)).unwrap();

    // Read back guest memory region (page 0).
    let mut out_mem = vec![0u8; bus.mem().len()];
    memory.read(&store, 0, &mut out_mem).unwrap();

    // Read back CpuState.
    let mut out_cpu_bytes = vec![0u8; abi::CPU_STATE_SIZE as usize];
    memory
        .read(&store, CPU_PTR as usize, &mut out_cpu_bytes)
        .unwrap();
    let out_cpu = CpuSnapshot::from_wasm_bytes(&out_cpu_bytes);

    (ret, out_cpu, out_mem)
}

fn run_interp(ir: &IrBlock, cpu: &CpuState, bus: &SimpleBus) -> CpuSnapshotAndMem {
    let mut interp_bus = bus.clone();

    let mut cpu_bytes = vec![0u8; abi::CPU_STATE_SIZE as usize];
    write_cpu_to_wasm_bytes(cpu, &mut cpu_bytes);

    let _ = execute_block(ir, &mut cpu_bytes, &mut interp_bus);

    CpuSnapshotAndMem {
        cpu: CpuSnapshot::from_wasm_bytes(&cpu_bytes),
        mem: interp_bus.mem().to_vec(),
    }
}

struct CpuSnapshotAndMem {
    cpu: CpuSnapshot,
    mem: Vec<u8>,
}

struct GenCtx {
    b: IrBuilder,
    insts: usize,
    // Value ids, plus their declared type.
    values: Vec<(ValueId, Width)>,
    // Known-safe guest addresses within the RAM window, always safe for 8-byte accesses.
    addr_values: Vec<ValueId>,
}

impl GenCtx {
    fn new(entry_rip: u64) -> Self {
        Self {
            b: IrBuilder::new(entry_rip),
            insts: 0,
            values: Vec::new(),
            addr_values: Vec::new(),
        }
    }

    fn can_add(&self, n: usize) -> bool {
        self.insts + n <= MAX_IR_INSTS
    }

    fn record(&mut self, id: ValueId, width: Width) {
        self.values.push((id, width));
    }

    fn const_int(&mut self, width: Width, value: u64) -> ValueId {
        self.insts += 1;
        let id = self.b.const_int(width, value);
        self.record(id, width);
        id
    }

    fn read_reg(&mut self, reg: GuestReg) -> ValueId {
        let width = match reg {
            GuestReg::Gpr { width, .. } => width,
            GuestReg::Flag(_) => Width::W8,
            GuestReg::Rip | GuestReg::SegBase(_) => Width::W64,
        };
        self.insts += 1;
        let id = self.b.read_reg(reg);
        self.record(id, width);
        id
    }

    fn write_reg(&mut self, reg: GuestReg, src: ValueId) {
        self.insts += 1;
        self.b.write_reg(reg, src);
    }

    fn trunc(&mut self, width: Width, src: ValueId) -> ValueId {
        self.insts += 1;
        let id = self.b.trunc(width, src);
        self.record(id, width);
        id
    }

    fn load(&mut self, width: Width, addr: ValueId) -> ValueId {
        self.insts += 1;
        let id = self.b.load(width, addr);
        self.record(id, width);
        id
    }

    fn store(&mut self, width: Width, addr: ValueId, src: ValueId) {
        self.insts += 1;
        self.b.store(width, addr, src);
    }

    fn binop(
        &mut self,
        op: BinOp,
        width: Width,
        lhs: ValueId,
        rhs: ValueId,
        flags: FlagSet,
    ) -> ValueId {
        self.insts += 1;
        let id = self.b.binop(op, width, lhs, rhs, flags);
        self.record(id, width);
        id
    }

    fn cmp_flags(&mut self, width: Width, lhs: ValueId, rhs: ValueId, flags: FlagSet) {
        self.insts += 1;
        self.b.cmp_flags(width, lhs, rhs, flags);
    }

    fn test_flags(&mut self, width: Width, lhs: ValueId, rhs: ValueId, flags: FlagSet) {
        self.insts += 1;
        self.b.test_flags(width, lhs, rhs, flags);
    }

    fn eval_cond(&mut self, cond: Cond) -> ValueId {
        self.insts += 1;
        let id = self.b.eval_cond(cond);
        self.record(id, Width::W8);
        id
    }

    fn select(
        &mut self,
        width: Width,
        cond: ValueId,
        if_true: ValueId,
        if_false: ValueId,
    ) -> ValueId {
        self.insts += 1;
        let id = self.b.select(width, cond, if_true, if_false);
        self.record(id, width);
        id
    }

    fn finish(self, terminator: IrTerminator) -> IrBlock {
        self.b.finish(terminator)
    }

    fn pick_any_value(&self, rng: &mut impl Rng) -> (ValueId, Width) {
        let idx = rng.random_range(0..self.values.len());
        self.values[idx]
    }

    fn pick_value_of_width(&self, rng: &mut impl Rng, width: Width) -> Option<ValueId> {
        // Reservoir sample without allocating.
        let mut chosen = None;
        let mut seen = 0usize;
        for (id, w) in &self.values {
            if *w != width {
                continue;
            }
            seen += 1;
            if rng.random_range(0..seen) == 0 {
                chosen = Some(*id);
            }
        }
        chosen
    }

    fn pick_or_const(&mut self, rng: &mut impl Rng, width: Width) -> ValueId {
        if let Some(v) = self.pick_value_of_width(rng, width) {
            return v;
        }
        // Ensure progress even if the generator forgot to seed something.
        self.const_int(width, rng.random())
    }

    fn pick_safe_addr(&self, rng: &mut impl Rng) -> ValueId {
        self.addr_values[rng.random_range(0..self.addr_values.len())]
    }

    fn pick_w8(&mut self, rng: &mut impl Rng) -> ValueId {
        self.pick_or_const(rng, Width::W8)
    }
}

fn random_width(rng: &mut impl Rng) -> Width {
    match rng.random_range(0..4u8) {
        0 => Width::W8,
        1 => Width::W16,
        2 => Width::W32,
        3 => Width::W64,
        _ => unreachable!(),
    }
}

fn random_gpr(rng: &mut impl Rng) -> Gpr {
    Gpr::from_u4(rng.random_range(0..16)).unwrap()
}

fn random_flag(rng: &mut impl Rng) -> Flag {
    match rng.random_range(0..6u8) {
        0 => Flag::Cf,
        1 => Flag::Pf,
        2 => Flag::Af,
        3 => Flag::Zf,
        4 => Flag::Sf,
        5 => Flag::Of,
        _ => unreachable!(),
    }
}

fn random_cond(rng: &mut impl Rng) -> Cond {
    match rng.random_range(0..16u8) {
        0 => Cond::O,
        1 => Cond::No,
        2 => Cond::B,
        3 => Cond::Ae,
        4 => Cond::E,
        5 => Cond::Ne,
        6 => Cond::Be,
        7 => Cond::A,
        8 => Cond::S,
        9 => Cond::Ns,
        10 => Cond::P,
        11 => Cond::Np,
        12 => Cond::L,
        13 => Cond::Ge,
        14 => Cond::Le,
        15 => Cond::G,
        _ => unreachable!(),
    }
}

/// Every `BinOp` the IR can hold.
///
/// This used to stop at `Sar`, which left `Adc`, `Sbb`, `Rol`, `Ror`, `Bswap` and
/// `Mul` untested against the interpreter — and the code generator computed the wrong
/// carry *out* of `adc`/`sbb` for as long as that was true. Anything the IR can
/// express belongs here, or this test says less than it appears to.
fn random_binop(rng: &mut impl Rng) -> BinOp {
    match rng.random_range(0..14u8) {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::And,
        3 => BinOp::Or,
        4 => BinOp::Xor,
        5 => BinOp::Shl,
        6 => BinOp::Shr,
        7 => BinOp::Sar,
        8 => BinOp::Adc,
        9 => BinOp::Sbb,
        10 => BinOp::Rol,
        11 => BinOp::Ror,
        12 => BinOp::Bswap,
        13 => BinOp::Mul,
        _ => unreachable!(),
    }
}

fn random_flagset(rng: &mut impl Rng) -> FlagSet {
    let mut set = FlagSet::EMPTY;
    let bits = [
        FlagSet::CF,
        FlagSet::PF,
        FlagSet::AF,
        FlagSet::ZF,
        FlagSet::SF,
        FlagSet::OF,
    ];
    for bit in bits {
        // Bias towards non-empty sets.
        if rng.random_bool(0.4) {
            set = set.union(bit);
        }
    }
    set
}

fn random_guest_gpr_reg(rng: &mut impl Rng) -> GuestReg {
    let reg = random_gpr(rng);
    let width = random_width(rng);
    let high8 = width == Width::W8
        && matches!(reg, Gpr::Rax | Gpr::Rcx | Gpr::Rdx | Gpr::Rbx)
        && rng.random_bool(0.25);
    GuestReg::Gpr { reg, width, high8 }
}

fn random_ir_block(rng: &mut impl Rng, entry_rip: u64) -> IrBlock {
    let mut ctx = GenCtx::new(entry_rip);

    // Seed one value of each width so later instructions can always find operands without having
    // to insert extra consts (keeping blocks bounded in size).
    for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
        if !ctx.can_add(1) {
            break;
        }
        ctx.const_int(w, rng.random());
    }

    // Seed a few safe memory addresses (safe for 8-byte loads/stores).
    let addr_max = RAM_WINDOW_BASE + RAM_WINDOW_LEN - 8;
    for _ in 0..8 {
        if !ctx.can_add(1) {
            break;
        }
        let addr = rng.random_range(RAM_WINDOW_BASE..=addr_max);
        let v = ctx.const_int(Width::W64, addr);
        ctx.addr_values.push(v);
    }

    while ctx.insts < MAX_IR_INSTS {
        // Stop early if we haven't seeded safe addresses for memory ops (shouldn't happen unless
        // MAX_IR_INSTS is very small).
        let have_addrs = !ctx.addr_values.is_empty();

        let choice: u8 = rng.random_range(0..100);
        if choice < 8 {
            // Const
            if !ctx.can_add(1) {
                break;
            }
            let w = random_width(rng);
            ctx.const_int(w, rng.random());
        } else if choice < 18 {
            // ReadReg (GPR/Flag)
            if !ctx.can_add(1) {
                break;
            }
            if rng.random_bool(0.8) {
                let reg = random_guest_gpr_reg(rng);
                ctx.read_reg(reg);
            } else {
                ctx.read_reg(GuestReg::Flag(random_flag(rng)));
            }
        } else if choice < 28 {
            // WriteReg (GPR/Flag)
            if !ctx.can_add(1) {
                break;
            }
            if rng.random_bool(0.8) {
                let reg = random_guest_gpr_reg(rng);
                let GuestReg::Gpr { width, .. } = reg else {
                    unreachable!()
                };
                let src = ctx.pick_or_const(rng, width);
                ctx.write_reg(reg, src);
            } else {
                let src = ctx.pick_w8(rng);
                ctx.write_reg(GuestReg::Flag(random_flag(rng)), src);
            }
        } else if choice < 36 {
            // Trunc
            if !ctx.can_add(1) {
                break;
            }
            let (src, src_w) = ctx.pick_any_value(rng);
            let dst_w = match src_w {
                Width::W8 => Width::W8,
                Width::W16 => {
                    if rng.random_bool(0.5) {
                        Width::W8
                    } else {
                        Width::W16
                    }
                }
                Width::W32 => match rng.random_range(0..3u8) {
                    0 => Width::W8,
                    1 => Width::W16,
                    2 => Width::W32,
                    _ => unreachable!(),
                },
                Width::W64 => match rng.random_range(0..4u8) {
                    0 => Width::W8,
                    1 => Width::W16,
                    2 => Width::W32,
                    3 => Width::W64,
                    _ => unreachable!(),
                },
            };
            ctx.trunc(dst_w, src);
        } else if choice < 56 {
            // BinOp (all ops)
            if !ctx.can_add(1) {
                break;
            }
            let op = random_binop(rng);
            let w = random_width(rng);
            let lhs = ctx.pick_or_const(rng, w);
            let rhs = ctx.pick_or_const(rng, w);
            let flags = random_flagset(rng);
            ctx.binop(op, w, lhs, rhs, flags);
        } else if choice < 66 {
            // CmpFlags
            if !ctx.can_add(1) {
                break;
            }
            let w = random_width(rng);
            let lhs = ctx.pick_or_const(rng, w);
            let rhs = ctx.pick_or_const(rng, w);
            let flags = random_flagset(rng).union(FlagSet::ZF); // bias towards ZF usage
            ctx.cmp_flags(w, lhs, rhs, flags);
        } else if choice < 72 {
            // TestFlags
            if !ctx.can_add(1) {
                break;
            }
            let w = random_width(rng);
            let lhs = ctx.pick_or_const(rng, w);
            let rhs = ctx.pick_or_const(rng, w);
            let flags = random_flagset(rng).union(FlagSet::PF);
            ctx.test_flags(w, lhs, rhs, flags);
        } else if choice < 78 {
            // EvalCond
            if !ctx.can_add(1) {
                break;
            }
            ctx.eval_cond(random_cond(rng));
        } else if choice < 88 {
            // Select
            if !ctx.can_add(1) {
                break;
            }
            let w = random_width(rng);
            let cond = ctx.pick_w8(rng);
            let if_true = ctx.pick_or_const(rng, w);
            let if_false = ctx.pick_or_const(rng, w);
            ctx.select(w, cond, if_true, if_false);
        } else if choice < 94 {
            // Load
            if !have_addrs || !ctx.can_add(1) {
                continue;
            }
            let w = random_width(rng);
            let addr = ctx.pick_safe_addr(rng);
            ctx.load(w, addr);
        } else {
            // Store
            if !have_addrs || !ctx.can_add(1) {
                continue;
            }
            let w = random_width(rng);
            let addr = ctx.pick_safe_addr(rng);
            let src = ctx.pick_or_const(rng, w);
            ctx.store(w, addr, src);
        }
    }

    // Random-ish terminator without inserting new values.
    let term_choice: u8 = rng.random_range(0..100);
    let term = if term_choice < 50 {
        IrTerminator::Jump {
            target: entry_rip.wrapping_add(rng.random_range(0..0x1000u64)),
        }
    } else if term_choice < 80 {
        let cond = ctx.pick_w8(rng);
        IrTerminator::CondJump {
            cond,
            target: entry_rip.wrapping_add(rng.random_range(0..0x1000u64)),
            fallthrough: entry_rip.wrapping_add(rng.random_range(0..0x1000u64)),
        }
    } else if term_choice < 95 {
        let (v, _w) = ctx.pick_any_value(rng);
        IrTerminator::IndirectJump { target: v }
    } else {
        IrTerminator::ExitToInterpreter {
            next_rip: entry_rip.wrapping_add(rng.random_range(0..0x1000u64)),
        }
    };

    let block = ctx.finish(term);
    block
        .validate()
        .unwrap_or_else(|e| panic!("generated invalid IR: {e}\n{}", block.to_text()));
    block
}

fn format_cpu_diff(expected: &CpuSnapshot, actual: &CpuSnapshot) -> String {
    let mut out = String::new();

    if expected.rip != actual.rip {
        out.push_str(&format!(
            "  rip: expected=0x{:016x} actual=0x{:016x}\n",
            expected.rip, actual.rip
        ));
    }
    if expected.rflags != actual.rflags {
        out.push_str(&format!(
            "  rflags: expected=0x{:016x} actual=0x{:016x} (xor=0x{:016x})\n",
            expected.rflags,
            actual.rflags,
            expected.rflags ^ actual.rflags
        ));
    }

    for i in 0..16 {
        if expected.gpr[i] == actual.gpr[i] {
            continue;
        }
        let reg = Gpr::from_u4(i as u8).unwrap();
        out.push_str(&format!(
            "  {reg}: expected=0x{:016x} actual=0x{:016x}\n",
            expected.gpr[i], actual.gpr[i]
        ));
    }

    out
}

fn format_mem_diff(expected: &[u8], actual: &[u8]) -> String {
    if expected.len() != actual.len() {
        return format!(
            "  len: expected={} actual={}\n",
            expected.len(),
            actual.len()
        );
    }

    if let Some((idx, (a, b))) = expected
        .iter()
        .copied()
        .zip(actual.iter().copied())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        let window_start = idx.saturating_sub(8);
        let window_end = (idx + 8).min(expected.len());
        let exp = &expected[window_start..window_end];
        let act = &actual[window_start..window_end];
        return format!(
            "  first_diff: off=0x{idx:04x} expected=0x{a:02x} actual=0x{b:02x}\n  window[0x{window_start:04x}..0x{window_end:04x}]:\n    expected: {exp:02x?}\n    actual:   {act:02x?}\n",
        );
    }

    String::new()
}

fn random_cpu_state(rng: &mut impl Rng, entry_rip: u64) -> CpuState {
    let mut cpu = CpuState {
        rip: entry_rip,
        ..Default::default()
    };

    for slot in &mut cpu.gpr {
        *slot = rng.random();
    }

    cpu.set_rflags(rng.random::<u64>() | abi::RFLAGS_RESERVED1);

    cpu
}

fn random_bus(rng: &mut impl Rng) -> SimpleBus {
    let mut bus = SimpleBus::new(RAM_SIZE);
    let mut bytes = vec![0u8; RAM_SIZE];
    rng.fill(&mut bytes[..]);
    bus.load(0, &bytes);
    bus
}

#[test]
fn tier1_random_ir_matches_wasm_codegen() {
    // Keep the wasmi engine shared across iterations for performance.
    let engine = Engine::default();

    for iter in 0..ITERS {
        // Per-iteration RNG so failures can be reproduced by iteration number without depending on
        // prior test cases.
        let mut rng =
            ChaCha8Rng::seed_from_u64(SEED ^ (iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        let entry_rip = 0x1000u64;
        let cpu = random_cpu_state(&mut rng, entry_rip);
        let bus = random_bus(&mut rng);

        let ir = random_ir_block(&mut rng, entry_rip);

        let interp = run_interp(&ir, &cpu, &bus);
        let (ret, wasm_cpu, wasm_mem) = run_wasm(&engine, &ir, &cpu, &bus);

        let wasm_next_rip = if ret == JIT_EXIT_SENTINEL_I64 {
            wasm_cpu.rip as i64
        } else {
            ret
        };
        if wasm_next_rip as u64 != wasm_cpu.rip {
            panic!(
                "tier1_random_ir_equiv internal error: ret/CPU.RIP mismatch at iter={iter}\nret={ret} cpu.rip=0x{:x}\n{}",
                wasm_cpu.rip,
                ir.to_text()
            );
        }

        if interp.cpu != wasm_cpu || interp.mem != wasm_mem {
            let cpu_diff = format_cpu_diff(&interp.cpu, &wasm_cpu);
            let mem_diff = format_mem_diff(&interp.mem, &wasm_mem);
            panic!(
                "tier1 random IR equivalence failed\n  iter={iter}\n  seed=0x{SEED:016x}\n\nIR:\n{}\nCPU diff:\n{}\nMemory diff:\n{}",
                ir.to_text(),
                cpu_diff,
                mem_diff
            );
        }
    }
}

/// Every `BinOp`, at every width, on the operand pairs where the rules differ.
///
/// The random generator above finds disagreements but hands back a fifty-instruction
/// block to read. This walks one operation at a time over the values that actually
/// distinguish the flag rules — zero, one, the width's maximum, the sign bit, and a
/// pair that is equal, which is the case `sbb eax, eax` lands on — with the carry in
/// set both ways. A failure here names the operation, the width and the operands.
#[test]
fn every_binop_matches_the_interpreter_on_boundary_operands() {
    let engine = Engine::default();
    let ops = [
        BinOp::Add,
        BinOp::Adc,
        BinOp::Sub,
        BinOp::Sbb,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Sar,
        BinOp::Rol,
        BinOp::Ror,
        BinOp::Bswap,
        BinOp::Mul,
    ];
    let widths = [Width::W8, Width::W16, Width::W32, Width::W64];
    let entry_rip = 0x1000u64;
    let mut failures: Vec<String> = Vec::new();

    for op in ops {
        for width in widths {
            let max = match width {
                Width::W8 => 0xffu64,
                Width::W16 => 0xffffu64,
                Width::W32 => 0xffff_ffffu64,
                Width::W64 => u64::MAX,
            };
            let sign = (max >> 1) + 1;
            let interesting = [0u64, 1, 2, 7, max, max - 1, sign, sign - 1, sign + 1];
            for &a in &interesting {
                for &b in &interesting {
                    for cf in [false, true] {
                        // One operation, its own operands, all six flags declared.
                        let mut ir_b = IrBuilder::new(entry_rip);
                        let lhs = ir_b.const_int(width, a);
                        let rhs = ir_b.const_int(width, b);
                        let res = ir_b.binop(op, width, lhs, rhs, FlagSet::ALU);
                        ir_b.write_reg(
                            GuestReg::Gpr {
                                reg: Gpr::Rax,
                                width,
                                high8: false,
                            },
                            res,
                        );
                        let ir = ir_b.finish(IrTerminator::Jump { target: 0x2000 });

                        let mut cpu = CpuState::default();
                        cpu.rip = entry_rip;
                        cpu.set_rflags(
                            if cf { 1 } else { 0 } | abi::RFLAGS_RESERVED1,
                        );
                        let bus = SimpleBus::new(RAM_SIZE);

                        let interp = run_interp(&ir, &cpu, &bus);
                        let (_, wasm_cpu, _) = run_wasm(&engine, &ir, &cpu, &bus);

                        if interp.cpu != wasm_cpu {
                            failures.push(format!(
                                "  {op:?}.{width:?} lhs={a:#x} rhs={b:#x} cf_in={} -> {}",
                                cf as u8,
                                format_cpu_diff(&interp.cpu, &wasm_cpu).trim_end()
                            ));
                        }
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} operand/flag combinations disagree with the IR interpreter:\n{}",
        failures.len(),
        failures
            .iter()
            .take(200)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// A block taken out of the running guest, decoded and compiled for real.
///
/// The generator above builds IR directly, so it exercises the code generator but
/// never the decoder or the translator ahead of it. This is the SHA-256 message
/// schedule out of `bcryptprimitives` — the block a 1M-instruction whole-machine
/// divergence bisected to — run through `discover_block` → `translate_block` →
/// codegen and compared against the IR interpreter.
///
/// It grows the block one instruction at a time and reports the first length that
/// disagrees, so a failure names the instruction rather than the block.

/// The captured block with its one RIP-relative `lea` retargeted into the test's
/// memory window.
///
/// `lea rax, [rip+0x5b82a]` points at a constant table tens of kilobytes away, and a
/// later instruction dereferences it. Only the displacement is rewritten: the
/// instruction keeps its length and its addressing form, which is what is under test.
fn real_block_bytes(entry: u64) -> Vec<u8> {
    let mut code: Vec<u8> = (0..REAL_BLOCK.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&REAL_BLOCK[i..i + 2], 16).unwrap())
        .collect();
    let at = code
        .windows(3)
        .position(|w| w == [0x48, 0x8d, 0x05])
        .expect("the block contains one rip-relative lea");
    let next = entry + at as u64 + 7;
    let disp = (0x2000u64.wrapping_sub(next)) as u32;
    code[at + 3..at + 7].copy_from_slice(&disp.to_le_bytes());
    code
}

/// Whether every decoded instruction lies inside the captured bytes.
///
/// Past the end the decoder is reading whatever the surrounding memory holds, and
/// the two engines are then not being given the same program.
fn fits(block: &aero_jit_x86::tier1::BasicBlock, entry: u64, len: usize) -> bool {
    block
        .insts
        .last()
        .is_some_and(|i| i.rip + u64::from(i.len) <= entry + len as u64)
}

const REAL_BLOCK: &str = "83e00f458bcd448b4484104183e10f418d45fe83e00f8b4c84108bc18bd1c1e90ac1ca13c1c81133d0418bc033d1418bc841c1e803c1c912c1c80733c8418d45f983e00f4133c803d18bcd03548410f7d18bc54201548c104123cb8bd5c1ca19c1c80b458bdf33d04533de8bc5c1c806458bc333d04423c7418bc423c533c8488d052ab8050003d1418bce420314a841ffc5418bc64203548c10458bcd03d34183e10f418bdf03f24123dec1c916c1c80d4433c333c8418bc6c1c80233c8";

#[test]
fn a_real_guest_block_matches_the_interpreter() {
    let entry_rip = 0x800u64;
    let code = real_block_bytes(entry_rip);
    let engine = Engine::default();

    // Registers land in a window where `[rsp + reg*4 + disp]` stays inside page 0.
    // The block masks its indices with `and reg, 0xf` before using them.
    let base_state = |rng: &mut ChaCha8Rng| {
        let mut cpu = CpuState::default();
        cpu.rip = entry_rip;
        for g in cpu.gpr.iter_mut() {
            *g = 0x3000 + (rng.random::<u64>() & 0x3ff);
        }
        cpu.set_rflags(rng.random::<u64>() | abi::RFLAGS_RESERVED1);
        cpu
    };

    let mut checked = 0usize;
    for len in 1..=64usize {
        let mut disc_bus = SimpleBus::new(RAM_SIZE);
        disc_bus.load(entry_rip, &code);
        let block = discover_block(
            &disc_bus,
            entry_rip,
            BlockLimits {
                max_insts: len,
                max_bytes: 1024,
                ..BlockLimits::default()
            },
        );
        if block.insts.len() < len || !fits(&block, entry_rip, code.len()) {
            break; // decoding would run past the captured bytes
        }
        let ir = translate_block(&block);

        for iter in 0..24u64 {
            let mut rng =
                ChaCha8Rng::seed_from_u64(0x51ed_2701 ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let cpu = base_state(&mut rng);
            let mut bus = SimpleBus::new(RAM_SIZE);
            let mut fill = vec![0u8; RAM_SIZE];
            rng.fill(&mut fill[..]);
            bus.load(0, &fill);
            bus.load(entry_rip, &code);

            let interp = run_interp(&ir, &cpu, &bus);
            let (_, wasm_cpu, wasm_mem) = run_wasm(&engine, &ir, &cpu, &bus);

            if interp.cpu != wasm_cpu || interp.mem != wasm_mem {
                let last = block.insts.last().unwrap();
                panic!(
                    "real block diverges at instruction {len} (rip={:#x}, {} bytes)\n\
                     iter={iter}\nIR:\n{}\nCPU diff:\n{}\nMemory diff:\n{}",
                    last.rip,
                    last.len,
                    ir.to_text(),
                    format_cpu_diff(&interp.cpu, &wasm_cpu),
                    format_mem_diff(&interp.mem, &wasm_mem)
                );
            }
        }
        checked = len;
    }
    println!("\n  checked {checked} instruction prefixes of the real block\n");
    assert!(checked >= 44, "expected to check most of the block");
}

/// The same real block, against Tier-0 rather than the IR interpreter.
///
/// The check above compares two consumers of the *same* IR, so it cannot see a
/// decoder or translator that agrees with itself and disagrees with x86. Tier-0 is
/// the machine's own interpreter and the thing the compiled tier has to match, so it
/// is the oracle that counts. Each engine gets its own copy of memory, which is what
/// makes a block with a read-modify-write store comparable at all.
#[test]
fn a_real_guest_block_matches_tier0() {
    let entry_rip = 0x800u64;
    check_block_against_tier0("sha256-schedule", &real_block_bytes(entry_rip), entry_rip);
}

/// `cmp r9, 0x1c00` then `jae`. Two instructions, no memory access.
///
/// Bisecting the divergence that appears with `inline_tlb_cross_page_fastpath`
/// enabled lands on this block, which cannot be affected by that option — so either
/// the translation is wrong for a reason of its own, or the bisection is naming the
/// block whose admission shifts interrupt timing rather than the block that is wrong.
#[test]
fn a_compare_and_branch_block_matches_tier0() {
    let code: Vec<u8> = (0..CMP_JAE_BLOCK.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&CMP_JAE_BLOCK[i..i + 2], 16).unwrap())
        .collect();
    check_block_against_tier0("cmp-jae", &code, 0x800);
}

const CMP_JAE_BLOCK: &str = "4981f9001c00007330";

/// `mov rax,[rsp+8]` then `call qword [rip+0x20]` — the shape every Windows import
/// call has, and the second-largest coverage wall before Tier-1 could translate it.
#[test]
fn an_indirect_call_through_memory_matches_tier0() {
    check_block_against_tier0("call-mem", &[0x48, 0x8b, 0x44, 0x24, 0x08, 0xff, 0x15, 0x20, 0x00, 0x00, 0x00], 0x800);
}

/// `call rax` — the register form takes the target before the push, which matters
/// only for the stack forms but is the same code path.
#[test]
fn an_indirect_call_through_a_register_matches_tier0() {
    check_block_against_tier0("call-reg", &[0x48, 0x83, 0xc0, 0x10, 0xff, 0xd0], 0x800);
}

/// `call qword [rsp]` — reads its target from the very slot the push overwrites, so
/// it fails if the operand is read after RSP moves.
#[test]
fn an_indirect_call_through_the_stack_top_matches_tier0() {
    check_block_against_tier0("call-rsp", &[0xff, 0x14, 0x24], 0x800);
}

/// `jmp qword [rip+0x10]` — the tail-call/thunk form.
#[test]
fn an_indirect_jump_through_memory_matches_tier0() {
    check_block_against_tier0("jmp-mem", &[0xff, 0x25, 0x10, 0x00, 0x00, 0x00], 0x800);
}

/// `movsx eax,al` then `inc rcx` — the busiest wall entry after the indirect call.
#[test]
fn a_sign_extending_move_matches_tier0() {
    check_block_against_tier0("movsx-8", &[0x0f, 0xbe, 0xc0, 0x48, 0xff, 0xc1, 0xc3], 0x800);
}

/// `movsx eax,[rdx]` — the memory form, and `movsx r8d,[rdx]` right after it.
#[test]
fn a_sign_extending_load_matches_tier0() {
    check_block_against_tier0("movsx-mem", &[0x0f, 0xbe, 0x02, 0x45, 0x0f, 0xbe, 0x02, 0xc3], 0x800);
}

/// `movsx rax,ax` — 16-bit source into a 64-bit destination.
#[test]
fn a_sign_extending_move_from_16_bits_matches_tier0() {
    check_block_against_tier0("movsx-16", &[0x48, 0x0f, 0xbf, 0xc0, 0xc3], 0x800);
}

/// `movsxd rbp,ebp` — sign-extends the low half of a register over itself.
#[test]
fn a_movsxd_matches_tier0() {
    check_block_against_tier0("movsxd", &[0x48, 0x63, 0xed, 0xc3], 0x800);
}

/// `inc dil` then `shr eax,8` — the byte increment form at 0xFE.
#[test]
fn an_eight_bit_increment_matches_tier0() {
    check_block_against_tier0("inc-r8", &[0x40, 0xfe, 0xc7, 0xc1, 0xe8, 0x08, 0xc3], 0x800);
}

/// `dec byte [rdx]` — the read-modify-write byte form.
#[test]
fn an_eight_bit_decrement_in_memory_matches_tier0() {
    check_block_against_tier0("dec-m8", &[0xfe, 0x0a, 0xc3], 0x800);
}

/// `lock and qword [rdi+8], rsi` — the busiest wall entry once rdtsc was compiled.
/// Aero runs one CPU at a time, so the lock is satisfied by the plain RMW.
#[test]
fn a_locked_and_to_memory_matches_tier0() {
    check_block_against_tier0("lock-and", &[0xf0, 0x48, 0x21, 0x77, 0x08, 0xc3], 0x800);
}

/// `lock or dword [rsp], 0` — the fence idiom Windows uses constantly.
#[test]
fn a_locked_or_fence_matches_tier0() {
    check_block_against_tier0("lock-or", &[0xf0, 0x83, 0x0c, 0x24, 0x00, 0xc3], 0x800);
}

/// `lock add qword [rdi], rax` then `lock inc dword [rsi]`.
#[test]
fn locked_add_and_increment_match_tier0() {
    check_block_against_tier0("lock-add", &[0xf0, 0x48, 0x01, 0x07, 0xf0, 0xff, 0x06, 0xc3], 0x800);
}

/// `lock` on a register destination is illegal and must not be translated as if the
/// prefix were absent.
#[test]
fn a_lock_prefix_on_a_register_destination_is_refused() {
    use aero_x86::tier1::decode_one_mode;
    let inst = decode_one_mode(0x800, &[0xf0, 0x48, 0x21, 0xf0], 64);
    assert!(
        matches!(inst.kind, aero_x86::tier1::InstKind::Invalid),
        "lock and rax,rsi should not translate: {:?}",
        inst.kind
    );
}

/// `bt rdx,10` then `jb` — the read-only bit test, which writes CF and nothing else.
#[test]
fn a_bit_test_matches_tier0() {
    check_block_against_tier0("bt-imm", &[0x48, 0x0f, 0xba, 0xe2, 0x0a, 0xc3], 0x800);
}

/// `lock bts qword [rdi+8],3` — the write-back form under a lock prefix.
#[test]
fn a_locked_bit_set_in_memory_matches_tier0() {
    check_block_against_tier0("lock-bts", &[0xf0, 0x48, 0x0f, 0xba, 0x6f, 0x08, 0x03, 0xc3], 0x800);
}

/// `btr dword [rsi],17` and `btc dword [rsi],5` — clear and flip.
#[test]
fn bit_reset_and_complement_match_tier0() {
    check_block_against_tier0(
        "btr-btc",
        &[0x0f, 0xba, 0x36, 0x11, 0x0f, 0xba, 0x3e, 0x05, 0xc3],
        0x800,
    );
}

/// `bts rax,70` — an offset past the operand size. Intel adjusts the bit base for a
/// memory operand; for a register it is taken modulo the width, and Tier-0 is the
/// oracle for which happens here.
#[test]
fn a_bit_offset_past_the_operand_size_matches_tier0() {
    check_block_against_tier0("bts-wide", &[0x48, 0x0f, 0xba, 0xe8, 0x46, 0xc3], 0x800);
}

/// The same, against memory, where the bit base moves.
#[test]
fn a_memory_bit_offset_past_the_operand_size_matches_tier0() {
    check_block_against_tier0("bts-mem-wide", &[0x48, 0x0f, 0xba, 0x2e, 0x46, 0xc3], 0x800);
}

/// `bt [rax],rdx` — a register bit offset against memory, so the bit base moves.
#[test]
fn a_bit_test_with_a_register_offset_in_memory_matches_tier0() {
    aero_x86::tier1::force_hint_ops(true);
    check_block_against_tier0("bt-reg-mem", &[0x48, 0x0f, 0xa3, 0x10, 0xc3], 0x800);
}

/// `bts rcx,rdx` — register destination, where the offset is masked instead.
#[test]
fn a_bit_set_with_a_register_offset_matches_tier0() {
    aero_x86::tier1::force_hint_ops(true);
    check_block_against_tier0("bts-reg", &[0x48, 0x0f, 0xab, 0xd1, 0xc3], 0x800);
}

/// `btr [rsi],eax` then `btc [rsi],eax`, 32-bit, exercising the negative-offset
/// path: the operands are random, so the offset is signed and often negative.
#[test]
fn bit_reset_and_complement_with_register_offsets_match_tier0() {
    aero_x86::tier1::force_hint_ops(true);
    check_block_against_tier0("btr-reg-mem", &[0x0f, 0xb3, 0x06, 0x0f, 0xbb, 0x06, 0xc3], 0x800);
}

/// `prefetchw [rbx+0x628]` — a hint, and the biggest single opcode left on the wall.
#[test]
fn a_prefetch_is_a_nop_matching_tier0() {
    aero_x86::tier1::force_hint_ops(true);
    check_block_against_tier0("prefetchw", &[0x0f, 0x0d, 0x8b, 0x28, 0x06, 0x00, 0x00, 0xc3], 0x800);
}

/// `movnti [rcx],rax` — a store whose non-temporal hint carries no architecture.
#[test]
fn a_non_temporal_store_matches_tier0() {
    aero_x86::tier1::force_hint_ops(true);
    check_block_against_tier0("movnti", &[0x48, 0x0f, 0xc3, 0x01, 0xc3], 0x800);
}

/// `lock xadd [rbx+8], eax` — Windows' reference counting, and the busiest
/// read-modify-write the compiled tier could not take.
#[test]
fn a_locked_exchange_and_add_matches_tier0() {
    aero_x86::tier1::force_rmw_ops(true);
    check_block_against_tier0("lock-xadd", &[0xf0, 0x0f, 0xc1, 0x43, 0x08, 0xc3], 0x800);
}

/// `xadd rcx, rdx` — the register form, where source and destination alias nothing.
#[test]
fn an_exchange_and_add_between_registers_matches_tier0() {
    aero_x86::tier1::force_rmw_ops(true);
    check_block_against_tier0("xadd-reg", &[0x48, 0x0f, 0xc1, 0xd1, 0xc3], 0x800);
}

/// `lock cmpxchg [rbx+0x10], rdx` — both outcomes matter: on equal the destination
/// takes the source, otherwise the accumulator takes the destination, and the flags
/// are CMP's either way.
#[test]
fn a_locked_compare_exchange_matches_tier0() {
    aero_x86::tier1::force_rmw_ops(true);
    check_block_against_tier0("lock-cmpxchg", &[0xf0, 0x48, 0x0f, 0xb1, 0x53, 0x10, 0xc3], 0x800);
}

/// The same against a register, and with RAX deliberately set from the destination
/// first so the equal case is exercised too.
#[test]
fn a_compare_exchange_between_registers_matches_tier0() {
    aero_x86::tier1::force_rmw_ops(true);
    check_block_against_tier0("cmpxchg-reg", &[0x48, 0x8b, 0xc1, 0x48, 0x0f, 0xb1, 0xd1, 0xc3], 0x800);
}

/// `lfence`/`mfence`/`sfence` decode to nothing.
///
/// Asserted on the decode rather than against Tier-0, because Tier-0 reaches its own
/// no-op for these through the *assist* path, and the equivalence harness above runs
/// Tier-0 with plain `step`, which reports an assist instead of performing it. The
/// machine-level oracles cover the pair actually running.
#[test]
fn the_fences_decode_to_nothing() {
    use aero_x86::tier1::{decode_one_mode, InstKind};
    aero_x86::tier1::force_rmw_ops(true);
    for bytes in [
        [0x0f, 0xae, 0xe8], // lfence
        [0x0f, 0xae, 0xf0], // mfence
        [0x0f, 0xae, 0xf8], // sfence
    ] {
        let inst = decode_one_mode(0x800, &bytes, 64);
        assert!(
            matches!(inst.kind, InstKind::Nop),
            "{bytes:02x?} should be a nop, got {:?}",
            inst.kind
        );
        assert_eq!(inst.len, 3, "{bytes:02x?} is three bytes");
    }
}

/// `ret 0x10` — pops the return address and releases 16 more bytes.
#[test]
fn a_ret_with_a_pop_count_matches_tier0() {
    check_block_against_tier0("ret-imm16", &[0x48, 0x83, 0xc0, 0x01, 0xc2, 0x10, 0x00], 0x800);
}

fn check_block_against_tier0(name: &str, code: &[u8], entry_rip: u64) {
    use aero_cpu_core::mem::FlatTestBus;

    let engine = Engine::default();

    for len in 1..=64usize {
        let mut disc_bus = SimpleBus::new(RAM_SIZE);
        disc_bus.load(entry_rip, code);
        let block = discover_block(
            &disc_bus,
            entry_rip,
            BlockLimits {
                max_insts: len,
                max_bytes: 1024,
                ..BlockLimits::default()
            },
        );
        if block.insts.len() < len || !fits(&block, entry_rip, code.len()) {
            break;
        }
        let ir = translate_block(&block);

        for iter in 0..16u64 {
            let mut rng =
                ChaCha8Rng::seed_from_u64(0xb10c_0de5 ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            // 64-bit mode: the block is 64-bit guest code, and in 32-bit mode Tier-0
            // reads its `45` REX prefixes as `inc ebp`.
            let mut cpu = CpuState::new(aero_cpu_core::state::CpuMode::Long);
            cpu.rip = entry_rip;
            for g in cpu.gpr.iter_mut() {
                *g = 0x3000 + (rng.random::<u64>() & 0x3ff);
            }
            cpu.set_rflags(rng.random::<u64>() | abi::RFLAGS_RESERVED1);

            let mut fill = vec![0u8; RAM_SIZE];
            rng.fill(&mut fill[..]);

            // Tier-0 on its own copy.
            let mut t0_bus = FlatTestBus::new(RAM_SIZE);
            t0_bus.load(0, &fill);
            t0_bus.load(entry_rip, code);
            let mut t0_cpu = cpu.clone();
            let mut faulted = false;
            for _ in 0..len {
                if aero_cpu_core::interp::tier0::exec::step(&mut t0_cpu, &mut t0_bus).is_err() {
                    faulted = true;
                    break;
                }
            }
            if faulted {
                continue;
            }
            t0_cpu.commit_lazy_flags();

            // Compiled, on an identical copy.
            let mut bus = SimpleBus::new(RAM_SIZE);
            bus.load(0, &fill);
            bus.load(entry_rip, code);
            let (_, wasm_cpu, wasm_mem) = run_wasm(&engine, &ir, &cpu, &bus);

            // Tier-0 leaves RIP at the next instruction; so does the block.
            let t0_snapshot = CpuSnapshot::from_cpu(&t0_cpu);
            // RF/TF are delivery state Tier-1 does not model.
            const STATUS: u64 = 0x8d5;
            let regs_match = t0_snapshot.gpr == wasm_cpu.gpr
                && t0_snapshot.rip == wasm_cpu.rip
                && (t0_snapshot.rflags & STATUS) == (wasm_cpu.rflags & STATUS);
            let mem_match = t0_bus.slice(0, RAM_SIZE) == &wasm_mem[..];
            if !regs_match || !mem_match {
                let last = block.insts.last().unwrap();
                let mut detail = String::new();
                if t0_snapshot.rip != wasm_cpu.rip {
                    detail += &format!(
                        "  rip: tier0={:#x} tier1={:#x}\n",
                        t0_snapshot.rip, wasm_cpu.rip
                    );
                }
                for (i, name) in [
                    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10",
                    "r11", "r12", "r13", "r14", "r15",
                ]
                .iter()
                .enumerate()
                {
                    if t0_snapshot.gpr[i] != wasm_cpu.gpr[i] {
                        detail += &format!(
                            "  {name}: tier0={:#x} tier1={:#x} (in={:#x})\n",
                            t0_snapshot.gpr[i], wasm_cpu.gpr[i], cpu.gpr[i]
                        );
                    }
                }
                let fd = (t0_snapshot.rflags ^ wasm_cpu.rflags) & STATUS;
                if fd != 0 {
                    detail += &format!("  flags differ: {fd:#x}\n");
                }
                if !mem_match {
                    let a = t0_bus.slice(0, RAM_SIZE);
                    for (off, (x, y)) in a.iter().zip(wasm_mem.iter()).enumerate() {
                        if x != y {
                            detail +=
                                &format!("  mem[{off:#x}]: tier0={x:#04x} tier1={y:#04x}\n");
                            break;
                        }
                    }
                }
                panic!(
                    "{name}: instruction {len} ({} bytes at {:#x}) disagrees with Tier-0, iter={iter}\n{detail}",
                    last.len, last.rip
                );
            }
        }
    }
}

/// `mov rax, gs:[disp32]`, the shape Windows reaches its per-CPU block through.
///
/// Segment prefixes were the single largest cause of Tier-1 bailing out on this
/// guest — about 500 blocks, all of them kernel code. The base has to come from the
/// same place Tier-0 reads it, or the two tiers disagree about where `gs:[0x88]` is.
#[test]
fn a_gs_relative_load_matches_tier0() {
    use aero_cpu_core::mem::FlatTestBus;

    let entry_rip = 0x800u64;
    // 65 48 8b 04 25 88 00 00 00   mov rax, gs:[0x88]
    // eb 00                        jmp +0
    let code: Vec<u8> = vec![0x65, 0x48, 0x8b, 0x04, 0x25, 0x88, 0x00, 0x00, 0x00, 0xeb, 0x00];

    let mut disc_bus = SimpleBus::new(RAM_SIZE);
    disc_bus.load(entry_rip, &code);
    let block = discover_block(&disc_bus, entry_rip, BlockLimits::default());
    assert_eq!(
        block.insts.len(),
        2,
        "the gs-prefixed load must decode, not bail: {:?}",
        block.end_kind
    );
    let ir = translate_block(&block);

    let engine = Engine::default();
    for (gs_base, value) in [(0x3000u64, 0xdead_beefu64), (0x4000, 0x1234_5678)] {
        let mut cpu = CpuState::new(aero_cpu_core::state::CpuMode::Long);
        cpu.rip = entry_rip;
        cpu.msr.gs_base = gs_base;
        cpu.set_rflags(abi::RFLAGS_RESERVED1);

        let mut fill = vec![0u8; RAM_SIZE];
        fill[(gs_base + 0x88) as usize..(gs_base + 0x88) as usize + 8]
            .copy_from_slice(&value.to_le_bytes());

        let mut t0_bus = FlatTestBus::new(RAM_SIZE);
        t0_bus.load(0, &fill);
        t0_bus.load(entry_rip, &code);
        let mut t0_cpu = cpu.clone();
        for _ in 0..2 {
            aero_cpu_core::interp::tier0::exec::step(&mut t0_cpu, &mut t0_bus)
                .expect("tier0 runs the block");
        }
        t0_cpu.commit_lazy_flags();
        assert_eq!(
            t0_cpu.gpr[0], value,
            "the test itself is wrong if Tier-0 does not read {value:#x} from gs:[0x88]"
        );

        let mut bus = SimpleBus::new(RAM_SIZE);
        bus.load(0, &fill);
        bus.load(entry_rip, &code);
        let (_, wasm_cpu, _) = run_wasm(&engine, &ir, &cpu, &bus);

        assert_eq!(
            wasm_cpu.gpr[0], t0_cpu.gpr[0],
            "gs_base={gs_base:#x}: compiled code read {:#x}, Tier-0 read {:#x}",
            wasm_cpu.gpr[0], t0_cpu.gpr[0]
        );
    }
}

/// A superblock taken out of the guest, against Tier-0.
///
/// `follow_conditionals` keeps decoding past a branch and turns it into a side exit,
/// so a block stops being a basic block: control can leave from the middle. This one
/// begins `test cl, 7; jne` — a memcpy checking alignment — and the whole point is
/// that the compiled block must leave exactly where Tier-0 leaves.
const SUPERBLOCK: &str = "f6c1070f85970200004d8bc849c1e90575234d8bc849c1e90374cf488b040a4889014883c10849ffc975f04983e007ebb9";

#[test]
fn a_superblock_matches_tier0() {
    use aero_cpu_core::mem::FlatTestBus;

    let entry_rip = 0x800u64;
    let code: Vec<u8> = (0..SUPERBLOCK.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&SUPERBLOCK[i..i + 2], 16).unwrap())
        .collect();

    let limits = BlockLimits {
        max_insts: 64,
        max_bytes: 1024,
        follow_conditionals: true,
        ..BlockLimits::default()
    };
    let mut disc_bus = SimpleBus::new(RAM_SIZE);
    disc_bus.load(entry_rip, &code);
    let block = discover_block_mode(&disc_bus, entry_rip, limits, 64);
    assert!(
        block.insts.len() > 3,
        "decoding should continue past the first branch, got {} instructions",
        block.insts.len()
    );
    let ir = translate_block(&block);
    assert!(
        ir.insts
            .iter()
            .any(|i| matches!(i, aero_jit_x86::tier1::ir::IrInst::SideExit { .. })),
        "a block decoded past a conditional must carry a side exit"
    );

    let engine = Engine::default();
    for iter in 0..32u64 {
        let mut rng =
            ChaCha8Rng::seed_from_u64(0x5b10_cc05 ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut cpu = CpuState::new(aero_cpu_core::state::CpuMode::Long);
        cpu.rip = entry_rip;
        for g in cpu.gpr.iter_mut() {
            *g = 0x3000 + (rng.random::<u64>() & 0x3ff);
        }
        cpu.set_rflags(rng.random::<u64>() | abi::RFLAGS_RESERVED1);

        let mut fill = vec![0u8; RAM_SIZE];
        rng.fill(&mut fill[..]);

        let mut t0_bus = FlatTestBus::new(RAM_SIZE);
        t0_bus.load(0, &fill);
        t0_bus.load(entry_rip, &code);
        let mut t0_cpu = cpu.clone();
        // Step until Tier-0 leaves the range the block covers.
        let end = entry_rip + code.len() as u64;
        let mut steps = 0u32;
        while t0_cpu.rip >= entry_rip && t0_cpu.rip < end && steps < 200 {
            if aero_cpu_core::interp::tier0::exec::step(&mut t0_cpu, &mut t0_bus).is_err() {
                break;
            }
            steps += 1;
        }
        t0_cpu.commit_lazy_flags();

        let mut bus = SimpleBus::new(RAM_SIZE);
        bus.load(0, &fill);
        bus.load(entry_rip, &code);
        let (_, wasm_cpu, _) = run_wasm(&engine, &ir, &cpu, &bus);

        // The compiled block runs once and leaves; Tier-0 was stepped to the same
        // point, so RIP is the thing that has to agree.
        assert_eq!(
            wasm_cpu.rip,
            CpuSnapshot::from_cpu(&t0_cpu).rip,
            "iter {iter}: compiled block left at {:#x}, Tier-0 left at {:#x}",
            wasm_cpu.rip,
            t0_cpu.rip
        );
    }
}

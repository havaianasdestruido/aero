//! What it costs to admit a block into the backend.
//!
//! On the desktop checkpoint the compiled tier is a net loss, and raising the
//! hotness thresholds until almost nothing compiles removes almost all of the
//! difference — so the cost is admission, not execution. A block retires about four
//! guest instructions, so if admitting one costs as much as running thousands, the
//! tier cannot pay for itself at any coverage.
//!
//! This measures that directly, in seconds rather than the minutes a full machine
//! run takes.
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test compile_cost -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use aero_cpu_core::state::CpuState;
use aero_jit_x86::backend::WasmtimeBackend;
use aero_jit_x86::{
    discover_block, translate_block, BlockLimits, Tier1Bus, Tier1WasmCodegen, Tier1WasmOptions,
};

const ENTRY: u64 = 0x100;

/// Distinct blocks so nothing is deduplicated: each is `add eax, N` then `ret`.
fn block_wasm(n: u8) -> Vec<u8> {
    let code = [0x83, 0xc0, n, 0xc3];
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
    for (i, b) in code.iter().enumerate() {
        backend.write_u8(ENTRY + i as u64, *b);
    }
    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    Tier1WasmCodegen::new().compile_block_with_options(
        &ir,
        Tier1WasmOptions {
            inline_tlb: true,
            ..Default::default()
        },
    )
}

#[test]
fn admitting_a_block_costs() {
    const N: usize = 200;
    let modules: Vec<Vec<u8>> = (0..N).map(|i| block_wasm((i % 200) as u8 + 1)).collect();

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);

    // Warm the engine so the first module's one-off costs are not charged to all.
    let _ = backend.add_compiled_block(&modules[0]);

    let start = Instant::now();
    for m in modules.iter().skip(1) {
        let _ = backend.add_compiled_block(m);
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() * 1e6 / (N - 1) as f64;

    println!("\n  {} blocks admitted in {:?}", N - 1, elapsed);
    println!("  {per:.1} us per block");
    // A guest instruction interpreted costs roughly 95ns, so this is the number of
    // interpreted instructions one admission is worth.
    println!("  ≈ {:.0} interpreted guest instructions per admission\n", per * 1000.0 / 95.0);

    assert!(
        elapsed.as_secs_f64() > 0.0,
        "the timed region must have done work"
    );
}

/// What batching blocks into one module saves.
#[test]
fn batching_amortises_admission() {
    use aero_jit_x86::tier1::ir::IrBlock;
    use wasmtime::{Config, Engine, Module};

    const N: usize = 256;
    // Distinct IR blocks, so nothing is shared or deduplicated.
    let irs: Vec<IrBlock> = (0..N)
        .map(|i| {
            let code = [0x83, 0xc0, (i % 200) as u8 + 1, 0xc3];
            let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
            for (j, b) in code.iter().enumerate() {
                backend.write_u8(ENTRY + j as u64, *b);
            }
            let block = discover_block(&backend, ENTRY, BlockLimits::default());
            translate_block(&block)
        })
        .collect();

    let opts = Tier1WasmOptions {
        inline_tlb: true,
        ..Default::default()
    };
    let codegen = Tier1WasmCodegen::new();

    let mut config = Config::new();
    config.wasm_simd(true);
    let cranelift = std::env::var("AERO_JIT_COMPILER").as_deref() == Ok("cranelift");
    if !cranelift {
        let _ = config.strategy(wasmtime::Strategy::Winch);
    }
    let engine = Engine::new(&config).expect("engine");
    let _ = Module::new(&engine, &codegen.compile_block_with_options(&irs[0], opts));
    println!("\n  compiler: {}", if cranelift { "cranelift" } else { "winch" });

    for per_module in [1usize, 16, 64, 256] {
        let modules: Vec<Vec<u8>> = irs
            .chunks(per_module)
            .map(|chunk| {
                let refs: Vec<&IrBlock> = chunk.iter().collect();
                codegen.compile_blocks_with_options(&refs, opts)
            })
            .collect();
        let bytes: usize = modules.iter().map(|m| m.len()).sum();
        let start = Instant::now();
        for m in &modules {
            let _ = Module::new(&engine, m).expect("module");
        }
        let elapsed = start.elapsed();
        println!(
            "  {per_module:>3} blocks/module: {:>6.1} us/block  ({} modules, {bytes} bytes total)",
            elapsed.as_secs_f64() * 1e6 / N as f64,
            modules.len()
        );
    }
    println!();
}

/// Split admission into its two halves.
///
/// `Module::new` compiles; `Linker::instantiate` resolves this module's imports and
/// builds an instance. Both are paid per block today because each block is its own
/// module, so knowing which dominates says whether the fix is a cheaper compiler or
/// fewer modules.
#[test]
fn where_admission_time_goes() {
    use wasmtime::{Config, Engine, Linker, Module, Store};

    const N: usize = 400;
    let modules: Vec<Vec<u8>> = (0..N).map(|i| block_wasm((i % 200) as u8 + 1)).collect();
    println!(
        "\n  module bytes: {} (min {}, max {})",
        modules.iter().map(|m| m.len()).sum::<usize>() / N,
        modules.iter().map(|m| m.len()).min().unwrap(),
        modules.iter().map(|m| m.len()).max().unwrap()
    );

    let mut config = Config::new();
    config.wasm_simd(true);
    if std::env::var("AERO_JIT_COMPILER").as_deref() != Ok("cranelift") {
        let _ = config.strategy(wasmtime::Strategy::Winch);
    }
    let engine = Engine::new(&config).expect("engine");

    // Warm up so one-off costs are not charged to the measurement.
    let _ = Module::new(&engine, &modules[0]).expect("warmup module");

    let start = Instant::now();
    let compiled: Vec<Module> = modules
        .iter()
        .map(|m| Module::new(&engine, m).expect("module"))
        .collect();
    let compile = start.elapsed();

    let mut store = Store::new(&engine, ());
    let linker: Linker<()> = Linker::new(&engine);
    // Instantiation needs the module's imports defined; without them this fails,
    // which is itself the answer for how much of the cost is import resolution.
    let start = Instant::now();
    let mut instantiated = 0usize;
    for m in &compiled {
        if linker.instantiate(&mut store, m).is_ok() {
            instantiated += 1;
        }
    }
    let instantiate = start.elapsed();

    println!("
  compile:     {:.1} us/block", compile.as_secs_f64() * 1e6 / N as f64);
    println!(
        "  instantiate: {:.1} us/block ({instantiated}/{N} succeeded; failures still pay the lookup)
",
        instantiate.as_secs_f64() * 1e6 / N as f64
    );
}

/// Does admission get slower as the cache fills?
///
/// The block cache holds tens of thousands of blocks, and every one is a live
/// instance in a single `Store`. If per-admission cost grows with the number of
/// instances already there, the cache size is self-limiting for a reason that has
/// nothing to do with compilation, and the fix is a different one.
#[test]
fn admission_cost_against_a_filling_store() {
    use aero_jit_x86::tier1::ir::IrBlock;

    const BATCH: usize = 64;
    const ROUNDS: usize = 24;

    let opts = Tier1WasmOptions {
        inline_tlb: true,
        ..Default::default()
    };
    let codegen = Tier1WasmCodegen::new();

    // One representative block, re-emitted so each module is distinct work but the
    // same size — the variable under test is how full the store already is.
    let ir_of = |n: u8| -> IrBlock {
        let code = [0x83, 0xc0, n, 0xc3];
        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
        for (j, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + j as u64, *b);
        }
        let block = discover_block(&backend, ENTRY, BlockLimits::default());
        translate_block(&block)
    };

    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
    println!("\n  blocks in store -> us per block admitted");
    let mut emit_total = 0.0;
    for round in 0..ROUNDS {
        let irs: Vec<IrBlock> = (0..BATCH).map(|i| ir_of(((round * BATCH + i) % 200) as u8 + 1)).collect();
        let refs: Vec<&IrBlock> = irs.iter().collect();

        let t0 = Instant::now();
        let wasm = codegen.compile_blocks_with_options(&refs, opts);
        let emit = t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        emit_total += emit;

        let t1 = Instant::now();
        let _ = backend.add_compiled_blocks(&wasm, BATCH);
        let admit = t1.elapsed().as_secs_f64() * 1e6 / BATCH as f64;

        if round % 4 == 0 || round == ROUNDS - 1 {
            println!(
                "  {:>6} -> emit {emit:>6.1} us + admit {admit:>7.1} us = {:>7.1} us  ({} bytes/block)",
                round * BATCH,
                emit + admit,
                wasm.len() / BATCH
            );
        }
    }
    println!("  mean emission cost {:.1} us per block\n", emit_total / ROUNDS as f64);
}

/// How much of a block's emitted size is fixed overhead.
///
/// Admission cost tracks emitted bytes almost exactly, so what a block pays before
/// it does any work sets the floor on how many blocks the tier can afford. A short
/// block that is mostly prologue is one the tier should probably not take.
#[test]
fn what_a_block_pays_before_it_does_anything() {
    for (label, opts) in [
        (
            "as the tier emits them today",
            Tier1WasmOptions {
                inline_tlb: true,
                single_fast_path: true,
                shared_mem_helpers: true,
                dispatcher: true,
                ..Default::default()
            },
        ),
        (
            "inline probe at every access",
            Tier1WasmOptions {
                inline_tlb: true,
                single_fast_path: true,
                ..Default::default()
            },
        ),
        (
            "one shared probe per module",
            Tier1WasmOptions {
                inline_tlb: true,
                single_fast_path: true,
                shared_mem_helpers: true,
                ..Default::default()
            },
        ),
    ] {
        println!("\n=== {label} ===");
        block_size_table(opts);
    }
}

fn block_size_table(opts: Tier1WasmOptions) {
    let codegen = Tier1WasmCodegen::new();

    println!("\n  guest insts -> emitted bytes");
    let mut prev = 0usize;
    for n in [1usize, 2, 4, 8, 16, 32] {
        // `add eax,1` repeated, then `ret`.
        let mut code: Vec<u8> = Vec::new();
        for _ in 0..n.saturating_sub(1) {
            code.extend_from_slice(&[0x83, 0xc0, 0x01]);
        }
        code.push(0xc3);

        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
        for (j, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + j as u64, *b);
        }
        let block = discover_block(&backend, ENTRY, BlockLimits::default());
        let ir = translate_block(&block);
        let wasm = codegen.compile_block_with_options(&ir, opts);
        let delta = wasm.len().saturating_sub(prev);
        println!(
            "  {n:>3} -> {:>5} bytes  ({:>4} per guest inst, +{delta} over previous)",
            wasm.len(),
            wasm.len() / n
        );
        prev = wasm.len();
    }

    // The same, for a block whose instructions touch memory.
    println!("\n  guest loads -> emitted bytes");
    prev = 0;
    for n in [1usize, 2, 4, 8, 16] {
        let mut code: Vec<u8> = Vec::new();
        for _ in 0..n {
            code.extend_from_slice(&[0x8b, 0x06]); // mov eax,[rsi]
        }
        code.push(0xc3);

        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
        for (j, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + j as u64, *b);
        }
        let block = discover_block(&backend, ENTRY, BlockLimits::default());
        let ir = translate_block(&block);
        let wasm = codegen.compile_block_with_options(&ir, opts);
        let delta = wasm.len().saturating_sub(prev);
        println!(
            "  {n:>3} -> {:>5} bytes  (+{delta} over previous)",
            wasm.len()
        );
        prev = wasm.len();
    }
    println!();
}

/// Where a minimal block's bytes actually go, section by section.
#[test]
fn the_shape_of_a_minimal_block() {
    let opts = Tier1WasmOptions {
        inline_tlb: true,
        ..Default::default()
    };
    let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
    for (j, b) in [0x83u8, 0xc0, 0x01, 0xc3].iter().enumerate() {
        backend.write_u8(ENTRY + j as u64, *b);
    }
    let block = discover_block(&backend, ENTRY, BlockLimits::default());
    let ir = translate_block(&block);
    let wasm = Tier1WasmCodegen::new().compile_block_with_options(&ir, opts);

    // Walk the section headers by hand: id (1 byte) + LEB length + payload.
    let mut i = 8usize; // magic + version
    println!("\n  total {} bytes", wasm.len());
    while i < wasm.len() {
        let id = wasm[i];
        i += 1;
        let (len, used) = {
            let (mut v, mut s, mut n) = (0u32, 0u32, 0usize);
            loop {
                let b = wasm[i + n];
                v |= u32::from(b & 0x7f) << s;
                n += 1;
                if b & 0x80 == 0 {
                    break;
                }
                s += 7;
            }
            (v as usize, n)
        };
        i += used;
        let name = match id {
            1 => "type",
            2 => "import",
            3 => "function",
            4 => "table",
            5 => "memory",
            6 => "global",
            7 => "export",
            10 => "code",
            _ => "other",
        };
        println!("  section {id:>2} {name:<9} {len:>5} bytes");
        if id == 10 {
            // Inside the code section: body size, then the locals declaration.
            let mut j = i;
            let count = wasm[j];
            j += 1;
            let _ = count;
            // skip body size LEB
            while wasm[j] & 0x80 != 0 {
                j += 1;
            }
            j += 1;
            let local_groups = wasm[j];
            let start = j;
            j += 1;
            let mut locals_total = 0u32;
            for _ in 0..local_groups {
                let mut v = 0u32;
                let mut s = 0u32;
                loop {
                    let b = wasm[j];
                    v |= u32::from(b & 0x7f) << s;
                    j += 1;
                    if b & 0x80 == 0 {
                        break;
                    }
                    s += 7;
                }
                j += 1; // valtype
                locals_total += v;
            }
            println!(
                "    locals: {locals_total} in {local_groups} groups ({} bytes of declaration)",
                j - start
            );
            println!("    body:   {} bytes of instructions", i + len - j);
        }
        i += len;
    }
    println!();
}

/// What declaring locals costs.
///
/// Every block declares a fixed 34 i64 locals plus one per IR value, whether it uses
/// them or not, and WASM requires them zero-initialised — so the compiler emits a
/// store for each at function entry. If admission tracks that count, trimming the
/// layout to what a block actually touches is worth more than any peephole.
#[test]
fn what_declaring_locals_costs() {
    use wasmtime::{Config, Engine, Module};
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module as M,
        TypeSection, ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");

    println!("\n  locals -> us per function compiled (64 functions per module)");
    for locals in [0u32, 8, 16, 32, 43, 64, 128] {
        let mut module = M::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        for _ in 0..64 {
            funcs.function(0);
        }
        module.section(&funcs);
        let mut exports = ExportSection::new();
        for i in 0..64u32 {
            exports.export(&format!("f{i}"), ExportKind::Func, i);
        }
        module.section(&exports);
        let mut code = CodeSection::new();
        for i in 0..64u32 {
            let mut f = if locals == 0 {
                Function::new(vec![])
            } else {
                Function::new(vec![(locals, ValType::I64)])
            };
            // A little real work so the body is not empty.
            f.instruction(&Instruction::I64Const(i64::from(i)));
            f.instruction(&Instruction::End);
            code.function(&f);
        }
        module.section(&code);
        let wasm = module.finish();

        // Warm, then time.
        let _ = Module::new(&engine, &wasm).expect("module");
        let start = Instant::now();
        for _ in 0..8 {
            let _ = Module::new(&engine, &wasm).expect("module");
        }
        println!(
            "  {locals:>3} -> {:>6.1} us  ({} bytes)",
            start.elapsed().as_secs_f64() * 1e6 / (8.0 * 64.0),
            wasm.len()
        );
    }
    println!();
}

/// Is admission per module, per function, or per byte?
///
/// Three different answers imply three different fixes — bigger batches, fewer
/// blocks, or smaller code — so it is worth knowing which one is true.
#[test]
fn where_admission_cost_actually_lives() {
    use wasmtime::{Config, Engine, Module};
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module as M,
        TypeSection, ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");

    let build = |n_funcs: u32, body_reps: u32| -> Vec<u8> {
        let mut module = M::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        for _ in 0..n_funcs {
            funcs.function(0);
        }
        module.section(&funcs);
        let mut exports = ExportSection::new();
        for i in 0..n_funcs {
            exports.export(&format!("f{i}"), ExportKind::Func, i);
        }
        module.section(&exports);
        let mut code = CodeSection::new();
        for _ in 0..n_funcs {
            let mut f = Function::new(vec![(8, ValType::I64)]);
            f.instruction(&Instruction::I64Const(0));
            for _ in 0..body_reps {
                f.instruction(&Instruction::I64Const(7));
                f.instruction(&Instruction::I64Add);
            }
            f.instruction(&Instruction::End);
            code.function(&f);
        }
        module.section(&code);
        module.finish()
    };

    let time = |wasm: &[u8]| -> f64 {
        let _ = Module::new(&engine, wasm).expect("module");
        let start = Instant::now();
        for _ in 0..8 {
            let _ = Module::new(&engine, wasm).expect("module");
        }
        start.elapsed().as_secs_f64() * 1e6 / 8.0
    };

    println!("\n  functions per module (tiny bodies) -> total us, us per function");
    for n in [1u32, 4, 16, 64, 256] {
        let wasm = build(n, 1);
        let t = time(&wasm);
        println!(
            "  {n:>4} funcs -> {t:>8.1} us total, {:>7.1} us/func  ({} bytes)",
            t / f64::from(n),
            wasm.len()
        );
    }

    println!("\n  body size (64 functions per module) -> total us");
    for reps in [1u32, 8, 32, 128] {
        let wasm = build(64, reps);
        let t = time(&wasm);
        println!(
            "  {reps:>4} ops/func -> {t:>8.1} us total, {:>7.1} us/func  ({} bytes)",
            t / 64.0,
            wasm.len()
        );
    }
    println!();
}

/// Is it the export, or the function?
#[test]
fn what_an_export_costs() {
    use wasmtime::{Config, Engine, Module};
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module as M,
        TypeSection, ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");

    let build = |n_funcs: u32, n_exports: u32| -> Vec<u8> {
        let mut module = M::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        for _ in 0..n_funcs {
            funcs.function(0);
        }
        module.section(&funcs);
        let mut exports = ExportSection::new();
        for i in 0..n_exports {
            exports.export(&format!("f{i}"), ExportKind::Func, i);
        }
        module.section(&exports);
        let mut code = CodeSection::new();
        for _ in 0..n_funcs {
            let mut f = Function::new(vec![(8, ValType::I64)]);
            f.instruction(&Instruction::I64Const(0));
            f.instruction(&Instruction::End);
            code.function(&f);
        }
        module.section(&code);
        module.finish()
    };

    let time = |wasm: &[u8]| -> f64 {
        let _ = Module::new(&engine, wasm).expect("module");
        let start = Instant::now();
        for _ in 0..8 {
            let _ = Module::new(&engine, wasm).expect("module");
        }
        start.elapsed().as_secs_f64() * 1e6 / 8.0
    };

    println!("\n  64 functions, varying how many are exported");
    for n_exports in [0u32, 1, 8, 32, 64] {
        let wasm = build(64, n_exports);
        let t = time(&wasm);
        println!(
            "  {n_exports:>3} exported -> {t:>8.1} us total, {:>6.1} us per function",
            t / 64.0
        );
    }
    println!();
}

/// Does reaching blocks through a table avoid the per-export cost?
#[test]
fn a_table_of_blocks_instead_of_64_exports() {
    use wasmtime::{Config, Engine, Module};
    use wasm_encoder::{
        CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
        FunctionSection, Instruction, Module as M, RefType, TableSection, TableType, TypeSection,
        ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");

    const N: u32 = 64;
    let mut module = M::new();
    let mut types = TypeSection::new();
    // 0: block signature. 1: dispatcher signature.
    types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I64]);
    module.section(&types);

    let mut funcs = FunctionSection::new();
    for _ in 0..N {
        funcs.function(0);
    }
    funcs.function(1); // the dispatcher
    module.section(&funcs);

    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: u64::from(N),
        maximum: Some(u64::from(N)),
        table64: false,
        shared: false,
    });
    module.section(&tables);

    let mut exports = ExportSection::new();
    exports.export("dispatch", ExportKind::Func, N);
    module.section(&exports);

    let idxs: Vec<u32> = (0..N).collect();
    let mut elems = ElementSection::new();
    elems.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(idxs.into()));
    module.section(&elems);

    let mut code = CodeSection::new();
    for _ in 0..N {
        let mut f = Function::new(vec![(8, ValType::I64)]);
        f.instruction(&Instruction::I64Const(0));
        f.instruction(&Instruction::End);
        code.function(&f);
    }
    // dispatch(cpu_ptr, jit_ctx_ptr, index) -> call_indirect
    let mut d = Function::new(vec![]);
    d.instruction(&Instruction::LocalGet(0));
    d.instruction(&Instruction::LocalGet(1));
    d.instruction(&Instruction::LocalGet(2));
    d.instruction(&Instruction::CallIndirect {
        type_index: 0,
        table_index: 0,
    });
    d.instruction(&Instruction::End);
    code.function(&d);
    module.section(&code);

    let wasm = module.finish();
    let _ = Module::new(&engine, &wasm).expect("module");
    let start = Instant::now();
    for _ in 0..8 {
        let _ = Module::new(&engine, &wasm).expect("module");
    }
    let t = start.elapsed().as_secs_f64() * 1e6 / 8.0;
    println!(
        "\n  {N} blocks in a table, 1 export -> {t:.1} us total, {:.1} us per block ({} bytes)\n",
        t / f64::from(N),
        wasm.len()
    );
}

/// One exported dispatcher that reaches blocks by *direct* call.
///
/// A function that is exported, or merely present in a table, gets a trampoline
/// compiled for it — ~80us each, which is most of what admitting a block costs.
/// A function only ever reached by a direct `call` needs none.
#[test]
fn a_br_table_dispatcher_of_direct_calls() {
    use wasmtime::{Config, Engine, Module};
    use wasm_encoder::{
        BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
        Module as M, TypeSection, ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");

    for n in [16u32, 64, 256] {
        let mut module = M::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
        types
            .ty()
            .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);

        let mut funcs = FunctionSection::new();
        for _ in 0..n {
            funcs.function(0);
        }
        funcs.function(1);
        module.section(&funcs);

        let mut exports = ExportSection::new();
        exports.export("dispatch", ExportKind::Func, n);
        module.section(&exports);

        let mut code = CodeSection::new();
        for _ in 0..n {
            let mut f = Function::new(vec![(8, ValType::I64)]);
            f.instruction(&Instruction::I64Const(0));
            f.instruction(&Instruction::End);
            code.function(&f);
        }
        // dispatch: nested blocks + br_table, each arm a direct call.
        let mut d = Function::new(vec![]);
        for _ in 0..=n {
            d.instruction(&Instruction::Block(BlockType::Empty));
        }
        d.instruction(&Instruction::LocalGet(2));
        let targets: Vec<u32> = (0..n).collect();
        d.instruction(&Instruction::BrTable(targets.into(), n));
        for i in 0..n {
            d.instruction(&Instruction::End);
            d.instruction(&Instruction::LocalGet(0));
            d.instruction(&Instruction::LocalGet(1));
            d.instruction(&Instruction::Call(i));
            d.instruction(&Instruction::Return);
        }
        d.instruction(&Instruction::End);
        d.instruction(&Instruction::I64Const(-1));
        d.instruction(&Instruction::End);
        code.function(&d);
        module.section(&code);

        let wasm = module.finish();
        let _ = Module::new(&engine, &wasm).expect("module");
        let start = Instant::now();
        for _ in 0..8 {
            let _ = Module::new(&engine, &wasm).expect("module");
        }
        let t = start.elapsed().as_secs_f64() * 1e6 / 8.0;
        println!(
            "  {n:>4} blocks, 1 export, direct calls -> {t:>8.1} us total, {:>5.1} us per block ({} bytes)",
            t / f64::from(n),
            wasm.len()
        );
    }
    println!();
}

/// What unused locals cost at *run* time.
///
/// WASM requires locals to start at zero, so a function that declares 43 of them
/// stores zero to 43 stack slots every time it is entered — and a Tier-1 block is
/// entered once per four guest instructions. If that is real, trimming the layout to
/// what a block actually touches is worth more than any peephole in the body.
#[test]
fn what_unused_locals_cost_to_enter() {
    for winch in [true, false] {
        println!("\n=== {} ===", if winch { "winch" } else { "cranelift" });
        unused_locals_table(winch);
    }
}

fn unused_locals_table(winch: bool) {
    use wasmtime::{Config, Engine, Instance, Linker, Module, Store};
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
        Module as M, TypeSection, ValType,
    };

    let mut config = Config::new();
    config.wasm_simd(true);
    if winch {
        let _ = config.strategy(wasmtime::Strategy::Winch);
    }
    let engine = Engine::new(&config).expect("engine");

    println!("  declared locals -> ns per call");
    let mut baseline = 0.0f64;
    for locals in [1u32, 8, 16, 34, 43, 64] {
        let mut module = M::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);
        module.section(&types);
        let mut funcs = FunctionSection::new();
        funcs.function(0);
        module.section(&funcs);
        let mut exports = ExportSection::new();
        exports.export("f", ExportKind::Func, 0);
        module.section(&exports);
        let mut code = CodeSection::new();
        // Touch one local so the body is not trivially empty, and return it.
        let mut f = Function::new(vec![(locals, ValType::I64)]);
        f.instruction(&Instruction::I64Const(7));
        f.instruction(&Instruction::LocalSet(2));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::End);
        code.function(&f);
        module.section(&code);
        let wasm = module.finish();

        let module = Module::new(&engine, &wasm).expect("module");
        let mut store = Store::new(&engine, ());
        let linker: Linker<()> = Linker::new(&engine);
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let _ = linker;
        let f = instance
            .get_typed_func::<(i32, i32), i64>(&mut store, "f")
            .expect("func");

        const N: usize = 2_000_000;
        // Warm.
        for _ in 0..10_000 {
            let _ = f.call(&mut store, (0, 0));
        }
        let start = Instant::now();
        for _ in 0..N {
            let _ = f.call(&mut store, (0, 0)).expect("call");
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / N as f64;
        if locals == 1 {
            baseline = ns;
        }
        println!(
            "  {locals:>3} -> {ns:>6.1} ns  (+{:.1} ns over 1 local)",
            ns - baseline
        );
    }
    println!();
}

/// Can two engines be made to share one guest memory?
///
/// Only if the module *defines* the memory. `Memory::new` builds its own instance
/// with a default on-demand allocator whose memory creator is `None`, so a memory
/// created that way and imported ignores `Config::with_host_memory` entirely.
#[test]
fn two_engines_share_a_module_defined_memory() {
    use aero_jit_x86::backend::shared_memory::SharedGuestMemory;
    use std::sync::Arc;
    use wasmtime::{Config, Engine, Instance, Module, Store};
    use wasm_encoder::{
        ExportKind, ExportSection, MemorySection, MemoryType as EncMemoryType, Module as M,
    };

    const PAGES: u64 = 4;
    const GUARD: usize = 1 << 20;
    let shared = SharedGuestMemory::new((PAGES as usize) * 65_536, GUARD);

    let mut module = M::new();
    let mut mems = MemorySection::new();
    mems.memory(EncMemoryType {
        minimum: PAGES,
        maximum: Some(PAGES),
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&mems);
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    module.section(&exports);
    let wasm = module.finish();

    let make = |winch: bool| {
        let mut config = Config::new();
        config.wasm_simd(true);
        if winch {
            let _ = config.strategy(wasmtime::Strategy::Winch);
        }
        config.memory_reservation(PAGES * 65_536);
        config.memory_guard_size(GUARD as u64);
        config.with_host_memory(Arc::new(shared.clone()));
        let engine = Engine::new(&config).expect("engine");
        let mut store = Store::new(&engine, ());
        let m = Module::new(&engine, &wasm).expect("module");
        let inst = Instance::new(&mut store, &m, &[]).expect("instance");
        let mem = inst.get_memory(&mut store, "memory").expect("memory");
        let ptr = mem.data_ptr(&store);
        (store, ptr)
    };

    let (_sa, pa) = make(true);
    let (_sb, pb) = make(false);
    println!("\n  winch memory at {pa:?}, cranelift memory at {pb:?}");
    assert_eq!(pa, pb, "the two engines did not map the same pages");
    assert_eq!(pa, shared.base(), "neither mapped the host's pages");
    println!("  both engines mapped the host's pages\n");
}

/// How many locals a real block declares, and where they go.
///
/// A block zero-initialises every declared local on entry, and it is entered once
/// per four guest instructions — so the split between the fixed layout and the IR's
/// value temporaries decides which is worth trimming.
#[test]
fn how_many_locals_a_real_block_declares() {
    use aero_jit_x86::tier1::ir::IrBlock;

    // Representative shapes, in the proportions Windows code actually has.
    let cases: [(&str, &[u8]); 6] = [
        ("mov/cmp/jne", &[0x48, 0x8b, 0x46, 0x08, 0x48, 0x3b, 0xc1, 0x75, 0x10]),
        ("push/mov/call", &[0x53, 0x48, 0x8b, 0xd9, 0xe8, 0x00, 0x01, 0x00, 0x00]),
        ("alu chain", &[0x48, 0x03, 0xc1, 0x48, 0x33, 0xc2, 0x48, 0x2b, 0xc3, 0xc3]),
        ("load/store pair", &[0x48, 0x8b, 0x06, 0x48, 0x89, 0x47, 0x08, 0xc3]),
        ("test/jz", &[0x85, 0xc0, 0x74, 0x20]),
        ("ret", &[0xc3]),
    ];

    println!("\n  block            guest  IR values");
    let mut vals = 0usize;
    let mut n = 0usize;
    for (name, code) in cases {
        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
        for (j, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + j as u64, *b);
        }
        let block = discover_block(&backend, ENTRY, BlockLimits::default());
        let ir: IrBlock = translate_block(&block);
        println!(
            "  {name:<16} {:>4}  {:>6}",
            block.insts.len(),
            ir.value_types.len()
        );
        vals += ir.value_types.len();
        n += 1;
    }
    let mean = vals as f64 / n as f64;
    println!(
        "\n  mean {mean:.1} value locals + 34 fixed = {:.1} declared\n  fixed is {:.0}% of them\n",
        mean + 34.0,
        100.0 * 34.0 / (mean + 34.0)
    );
}

/// What a block costs in a module the size the tier actually builds.
///
/// The single-block figures elsewhere in this file charge one block for the shared
/// memory helper, the dispatcher and the imports. A real module holds sixteen, so
/// those amortise; this is the number that multiplies out to the 39 MiB a session
/// emits.
#[test]
fn what_a_block_costs_in_a_real_module() {
    use aero_jit_x86::tier1::ir::IrBlock;

    let opts = Tier1WasmOptions {
        inline_tlb: true,
        single_fast_path: true,
        shared_mem_helpers: true,
        dispatcher: true,
        ..Default::default()
    };
    let codegen = Tier1WasmCodegen::new();

    let build = |code: &[u8]| -> IrBlock {
        let mut backend = WasmtimeBackend::<CpuState>::new_with_memory_pages(4, 0x8000);
        for (j, b) in code.iter().enumerate() {
            backend.write_u8(ENTRY + j as u64, *b);
        }
        let block = discover_block(&backend, ENTRY, BlockLimits::default());
        translate_block(&block)
    };

    let cases: [(&str, &[u8]); 5] = [
        // No guest registers touched at all: the floor.
        ("jmp $+2", &[0xeb, 0x00]),
        // One register, no flags, no memory.
        ("mov eax,1 / jmp", &[0xb8, 0x01, 0x00, 0x00, 0x00, 0xeb, 0x00]),
        // Flags, no memory.
        ("cmp/jne", &[0x48, 0x3b, 0xc1, 0x75, 0x10]),
        // One load.
        ("mov rax,[rsi] / jmp", &[0x48, 0x8b, 0x06, 0xeb, 0x00]),
        // Representative: load, compare, branch.
        (
            "load/cmp/jne",
            &[0x48, 0x8b, 0x46, 0x08, 0x48, 0x3b, 0xc1, 0x75, 0x10],
        ),
    ];

    println!("\n  sixteen to a module:");
    for (name, code) in cases {
        let ir = build(code);
        let irs: Vec<IrBlock> = (0..16).map(|_| ir.clone()).collect();
        let refs: Vec<&IrBlock> = irs.iter().collect();
        let many = codegen.compile_blocks_with_options(&refs, opts);
        let one = codegen.compile_blocks_with_options(&refs[..1], opts);
        // (16 blocks + shared) - (1 block + shared) = 15 blocks.
        let per_block = (many.len() - one.len()) / 15;
        println!(
            "  {name:<22} {per_block:>4} bytes per block   (module of 16: {} bytes)",
            many.len()
        );
    }
    println!();
}

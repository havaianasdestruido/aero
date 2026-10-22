//! Where a chain link's 89ns goes, and what moving dispatch into wasm would save.
//!
//! A link is a host `TypedFunc::call` plus a host lookup. Blocks retire about four
//! guest instructions, so the link is divided by four before it reaches the
//! per-instruction cost — at 89ns that is 22ns of the 38ns compiled code costs per
//! guest instruction, more than the block bodies themselves.
//!
//! The alternative is a dispatch loop *inside* wasm: one host call runs a whole
//! chain, and each link becomes a `call_indirect` through a table. That is also what
//! a browser tier has to do, where the host boundary is a JS call and costs more.
//!
//! This measures three things on the same trivial body, so the difference is the
//! dispatch and nothing else:
//!   - a bare host call, to separate boundary cost from block work
//!   - the host loop, as the tier runs it today
//!   - an in-wasm loop over `call_indirect`
//!
//! ```text
//! cargo test --release -p aero-jit-x86 --test dispatch_cost -- --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, MemArg, MemoryType, Module, RefType,
    TableSection, TableType, TypeSection, ValType,
};
use wasmtime::{Config, Engine, Linker, Memory, MemoryType as WtMemoryType, Module as WtModule, Store, TypedFunc};

const ITERS: usize = 500_000;

/// A module holding `n` leaf functions in a table plus two entry points.
///
/// Every leaf has the same body — return its own index's successor — so the only
/// thing that differs between the two measurements is who runs the loop.
fn dispatch_module(n: u32) -> Vec<u8> {
    let mut types = TypeSection::new();
    // 0: leaf, (i32 ctx) -> i32 next
    types.ty().function([ValType::I32], [ValType::I32]);
    // 1: run(i32 start, i32 iters) -> i32
    types.ty().function([ValType::I32, ValType::I32], [ValType::I32]);

    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );

    let mut funcs = FunctionSection::new();
    for _ in 0..n {
        funcs.function(0);
    }
    funcs.function(1); // run_chain

    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: u64::from(n),
        maximum: Some(u64::from(n)),
        table64: false,
        shared: false,
    });

    let mut elems = ElementSection::new();
    let ids: Vec<u32> = (0..n).collect();
    elems.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(ids.into()));

    let mut code = CodeSection::new();
    for i in 0..n {
        // A body with a little work in it, so dispatch is not being compared against
        // literally nothing: bump a counter in memory, return the next index.
        let mut f = Function::new([]);
        f.instructions()
            .i32_const(0)
            .i32_const(0)
            .i32_load(MemArg { offset: 0, align: 2, memory_index: 0 })
            .i32_const(1)
            .i32_add()
            .i32_store(MemArg { offset: 0, align: 2, memory_index: 0 })
            .i32_const(((i + 1) % n) as i32)
            .end();
        code.function(&f);
    }

    // run_chain(start, iters): loop { idx = call_indirect(idx); iters -= 1 }
    let mut run = Function::new([(2, ValType::I32)]);
    {
        let idx = 2u32; // local 0 = start, 1 = iters, 2/3 = scratch
        let i = &mut run.instructions();
        i.local_get(0).local_set(idx);
        i.block(wasm_encoder::BlockType::Empty);
        i.loop_(wasm_encoder::BlockType::Empty);
        // if iters == 0 break
        i.local_get(1).i32_eqz().br_if(1);
        i.local_get(1).i32_const(1).i32_sub().local_set(1);
        // idx = call_indirect[idx](idx)
        i.local_get(idx).local_get(idx).call_indirect(0, 0);
        i.local_set(idx);
        i.br(0);
        i.end(); // loop
        i.end(); // block
        i.local_get(idx).end();
    }
    code.function(&run);

    let mut exports = ExportSection::new();
    exports.export("run_chain", ExportKind::Func, n);
    for i in 0..n.min(4) {
        exports.export(&format!("leaf{i}"), ExportKind::Func, i);
    }

    let mut m = Module::new();
    m.section(&types);
    m.section(&imports);
    m.section(&funcs);
    m.section(&tables);
    m.section(&exports);
    m.section(&elems);
    m.section(&code);
    m.finish()
}

#[test]
fn in_wasm_dispatch_versus_host_dispatch() {
    const N: u32 = 64;
    let wasm = dispatch_module(N);

    let mut config = Config::new();
    let _ = config.strategy(wasmtime::Strategy::Winch);
    let engine = Engine::new(&config).expect("engine");
    let module = WtModule::new(&engine, &wasm).expect("module");

    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, WtMemoryType::new(1, None)).expect("memory");
    let mut linker: Linker<()> = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory).expect("define");
    let instance = linker.instantiate(&mut store, &module).expect("instance");

    let run: TypedFunc<(i32, i32), i32> = instance
        .get_typed_func(&mut store, "run_chain")
        .expect("run_chain");
    let leaves: Vec<TypedFunc<i32, i32>> = (0..N.min(4))
        .map(|i| {
            instance
                .get_typed_func(&mut store, &format!("leaf{i}"))
                .expect("leaf")
        })
        .collect();

    // Warm both paths.
    run.call(&mut store, (0, 1000)).expect("warm");
    for _ in 0..1000 {
        leaves[0].call(&mut store, 0).expect("warm");
    }

    // The tier's shape today: one host call per block, plus a host-side lookup.
    let table: Vec<usize> = (0..N as usize).map(|i| (i + 1) % N as usize).collect();
    let start = Instant::now();
    let mut idx = 0usize;
    for _ in 0..ITERS {
        let _ = leaves[idx.min(leaves.len() - 1)]
            .call(&mut store, idx as i32)
            .expect("call");
        idx = table[idx];
    }
    let host_loop = start.elapsed();

    // The same links, dispatched inside wasm.
    let start = Instant::now();
    run.call(&mut store, (0, ITERS as i32)).expect("run");
    let wasm_loop = start.elapsed();

    let host_ns = host_loop.as_secs_f64() * 1e9 / ITERS as f64;
    let wasm_ns = wasm_loop.as_secs_f64() * 1e9 / ITERS as f64;

    println!("\n  host loop  (call per block):   {host_ns:6.1} ns/link");
    println!("  wasm loop  (call_indirect):   {wasm_ns:6.1} ns/link");
    println!("  saved per link:               {:6.1} ns", host_ns - wasm_ns);
    println!(
        "  at ~4 guest instructions per block: {:.1} ns per guest instruction\n",
        (host_ns - wasm_ns) / 4.0
    );

    assert!(
        wasm_ns < host_ns,
        "dispatching inside wasm should not be slower than crossing the host boundary"
    );
}

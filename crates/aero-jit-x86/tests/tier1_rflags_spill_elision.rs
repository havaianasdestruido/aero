use aero_jit_x86::abi;
use aero_jit_x86::tier1::ir::{IrBuilder, IrTerminator};
use aero_jit_x86::tier1::Tier1WasmCodegen;
use aero_types::{Cond, FlagSet, Width};

use wasmparser::{Operator, Parser, Payload};

fn wasm_accesses_cpu_rflags(wasm: &[u8]) -> (bool, bool) {
    let mut has_load = false;
    let mut has_store = false;
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::CodeSectionEntry(body) = payload.expect("parse wasm") {
            let mut reader = body.get_operators_reader().expect("operators reader");
            while !reader.eof() {
                match reader.read().expect("read operator") {
                    Operator::I64Load { memarg } => {
                        if memarg.offset == u64::from(abi::CPU_RFLAGS_OFF) {
                            has_load = true;
                        }
                    }
                    Operator::I64Store { memarg } => {
                        if memarg.offset == u64::from(abi::CPU_RFLAGS_OFF) {
                            has_store = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    (has_load, has_store)
}

#[test]
fn tier1_readonly_rflags_loads_without_spill() {
    let entry = 0x1000u64;

    // A block that only reads flags (via EvalCond) but does not write them.
    let mut b = IrBuilder::new(entry);
    let cond = b.eval_cond(Cond::E);
    let block = b.finish(IrTerminator::CondJump {
        cond,
        target: entry + 0x10,
        fallthrough: entry + 0x8,
    });
    block.validate().unwrap();

    let wasm = Tier1WasmCodegen::new().compile_block(&block);
    let (has_load, has_store) = wasm_accesses_cpu_rflags(&wasm);
    assert!(has_load, "expected Tier-1 block to load CpuState.rflags");
    assert!(
        !has_store,
        "expected Tier-1 block to avoid spilling CpuState.rflags when read-only"
    );
}

#[test]
fn tier1_writes_rflags_spills_back() {
    let entry = 0x1000u64;

    // A flag-producing operation must hand its flags on — but not necessarily through
    // RFLAGS. With lazy flags the whole point is that an ALU operation is *recorded* in
    // the descriptor and RFLAGS is left exactly as it was found, so the block writes
    // the descriptor and skips the RFLAGS round trip entirely. What has to hold is that
    // one of the two happens, not which.
    let mut b = IrBuilder::new(entry);
    let lhs = b.const_int(Width::W64, 1);
    let rhs = b.const_int(Width::W64, 2);
    b.cmp_flags(Width::W64, lhs, rhs, FlagSet::ALU);
    let block = b.finish(IrTerminator::Jump { target: entry + 1 });
    block.validate().unwrap();

    let wasm = Tier1WasmCodegen::new().compile_block(&block);
    let (_, has_rflags_store) = wasm_accesses_cpu_rflags(&wasm);
    let spills_descriptor = wasm_stores_offset(&wasm, aero_jit_x86::abi::CPU_LAZY_FLAGS_OFF);
    assert!(
        has_rflags_store || spills_descriptor,
        "a block that writes flags must hand them on, through RFLAGS or the lazy \
         descriptor"
    );
}

/// Whether the module stores to `cpu_ptr + off` at i64 width.
fn wasm_stores_offset(wasm: &[u8], off: u32) -> bool {
    use wasmparser::{Operator, Parser, Payload};
    let mut found = false;
    for payload in Parser::new(0).parse_all(wasm) {
        let Ok(Payload::CodeSectionEntry(body)) = payload else {
            continue;
        };
        let Ok(mut reader) = body.get_operators_reader() else {
            continue;
        };
        while let Ok(op) = reader.read() {
            if let Operator::I64Store { memarg } = op {
                if memarg.offset == u64::from(off) {
                    found = true;
                }
            }
        }
    }
    found
}

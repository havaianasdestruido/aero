# Performance

> Where Aero's time goes and how that is measured: the optimisation strategy
> per subsystem, the telemetry and tracing surfaces, the benchmark harnesses,
> and the guest-CPU microbenchmark suite.
>
> Headline targets on this page are design goals, not measurements. Measured
> numbers live in [../history/windows-7-bring-up.md](../history/windows-7-bring-up.md) and
> `bench/history.json`.

## Overview

Achieving acceptable performance for Windows 7 emulation requires aggressive optimization at every layer. This document details strategies across CPU, memory, graphics, and I/O subsystems.

---

## Performance Budget

### Target Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Boot Time | < 60s | < 120s |
| Desktop FPS | ≥ 30 | ≥ 15 |
| Application Launch | < 5s | < 15s |
| Input Latency | < 50ms | < 100ms |
| Memory Overhead | < 1.5x | < 2x |
| IPS (MIPS) | ≥ 500 | ≥ 200 |

For CPU throughput tracking, prefer the **guest CPU microbenchmark suite ** rather than full-system boot/app workloads. See: [Guest CPU Instruction Throughput Benchmarks](performance.md).

### Performance Breakdown (Target)

```
┌─────────────────────────────────────────────────────────────────┐
│ CPU Time Distribution │
├─────────────────────────────────────────────────────────────────┤
│ │
│ JIT-compiled code execution 40% ████████████████ │
│ Memory access / MMU 25% ██████████ │
│ Device emulation 15% ██████ │
│ Interpreter (cold code) 10% ████ │
│ JIT compilation 5% ██ │
│ Synchronization / IPC 5% ██ │
│ │
└─────────────────────────────────────────────────────────────────┘
```

---

## CPU Optimization

### JIT Compilation Strategy

#### Block Detection and Compilation Thresholds

```rust
pub struct JitConfig {
 // Tier thresholds
 pub tier1_threshold: u32, // 10 executions → baseline JIT
 pub tier2_threshold: u32, // 1000 executions → optimizing JIT
 
 // Block limits
 pub max_block_size: usize, // 1000 instructions
 pub max_trace_length: usize, // 5000 instructions
 
 // Cache sizes
 pub code_cache_size: usize, // 256 MB
 pub max_compiled_blocks: usize, // 100,000
 
 // Optimization settings
 pub inline_threshold: usize, // Inline functions < 50 instructions
 pub unroll_threshold: usize, // Unroll loops < 16 iterations
}
```

#### Hot Path Optimization

```rust
pub struct HotPathOptimizer {
 profile: ProfileData,
}

impl HotPathOptimizer {
 pub fn optimize_trace(&self, trace: &Trace) -> OptimizedTrace {
 let mut ir = self.build_ir(trace);
 
 // Optimization passes in order
 self.constant_folding(&mut ir);
 self.dead_code_elimination(&mut ir);
 self.common_subexpression_elimination(&mut ir);
 self.flag_elimination(&mut ir); // x86-specific
 self.strength_reduction(&mut ir);
 self.loop_invariant_code_motion(&mut ir);
 self.register_allocation(&mut ir);
 self.instruction_scheduling(&mut ir);
 
 self.generate_wasm(&ir)
 }
 
 /// Eliminate redundant flag computations
 fn flag_elimination(&self, ir: &mut IrGraph) {
 // Track which flags are live at each point
 let liveness = self.compute_flag_liveness(ir);
 
 for block in ir.blocks_mut() {
 for inst in block.instructions_mut() {
 if let Some(flags_written) = inst.flags_written() {
 let live_flags = liveness.at(inst.id());
 let needed_flags = flags_written & live_flags;
 
 if needed_flags.is_empty() {
 // No flags needed - remove computation entirely
 inst.remove_flag_computation();
 } else if needed_flags != flags_written {
 // Only some flags needed - optimize
 inst.set_required_flags(needed_flags);
 }
 }
 }
 }
 }
}
```

#### WASM SIMD Utilization

```rust
impl WasmCodegen {
 fn emit_sse_operation(&mut self, op: &SseOp) {
 match op {
 SseOp::Addps { dst, src } => {
 // Use WASM SIMD v128 operations
 self.emit_v128_load(src);
 self.emit_v128_load(dst);
 self.emit(Instruction::F32x4Add);
 self.emit_v128_store(dst);
 }
 SseOp::Mulps { dst, src } => {
 self.emit_v128_load(src);
 self.emit_v128_load(dst);
 self.emit(Instruction::F32x4Mul);
 self.emit_v128_store(dst);
 }
 SseOp::Shufps { dst, src, imm } => {
 // WASM doesn't have direct shuffle - use swizzle
 self.emit_v128_load(src);
 self.emit_v128_load(dst);
 self.emit_shuffle_pattern(imm);
 self.emit(Instruction::I8x16Swizzle);
 self.emit_v128_store(dst);
 }
 // ... more SSE operations
 }
 }
}
```

#### What the ≥500 MIPS target would require

The target at the top of this page is a design goal, and the measurements now bound
it. Three numbers, all measured on the desktop checkpoint: the interpreter costs
~210ns per guest instruction, compiled code ~39ns, and coverage plateaus near 60% —
lowering the admission thresholds compiles four times as many blocks for one extra
point of coverage and a worse per-instruction cost, so that plateau is a property of
the workload rather than of the tuning.

Blending those gives what is reachable:

| coverage | compiled at 38.8ns | at the 7ns floor | compiled free |
|---|---|---|---|
| 60% (measured) | 9.2 MIPS | 11.3 | 11.9 |
| 90% | 18.6 | 36.6 | 47.6 |
| 100% | 25.8 | **142.9** | — |

Two things follow. At today's coverage the interpreted remainder dominates
completely: **even free compiled code caps the machine at 11.9 MIPS**, so work on
code quality is worth less than work on coverage until coverage is much higher.
And 500 MIPS means 2.0ns per guest instruction blended — below the 7ns the compiled
engine reaches on a synthetic ALU loop with perfect chaining, which is its best
case, not its average. **That target is not reachable by x86-to-WASM translation**;
something in the region of 50–150 MIPS is what a mature version of this design
supports, and that is the number worth planning against.

#### What actually executes today (measured, 2026-08-26)

The strategy above describes the intended tiering. None of it is reached by the
machine that boots Windows 7. Measured on the desktop checkpoint, 100M guest
instructions, differencing two instruction budgets so snapshot load cancels:

| Path | host insns / guest insn |
|---|---|
| Tier-0 only | 1326 |
| `--jit` (Tier-0 + IR block cache) | 1267 |

At an IPC of ~2.0 that is **about 4.5 MIPS**, against the ≥500 MIPS target above —
a factor of roughly 110. A steady-state `perf` profile puts ~50% of host
instructions in interpretation (`run_batch_cpu_core_with_assists`, `exec_decoded`,
`ops_alu`, decode) and ~15% in the memory/MMU path, so the gap is where the
strategy says it should be; it simply is not being closed yet.

Two separate things are missing.

**The Tier-1 WASM JIT is not wired to the full-system machine at all.** The whole
tiered stack exists and is tested — `ExecDispatcher`, `JitRuntime` with hotness,
compile queue and page-version invalidation, `Tier1Compiler` (x86 → WASM), a
`WasmtimeBackend` for native and a `WasmBackend` for the browser. Every consumer
of it is a CPU-only harness: the crates' own tests, and `WasmTieredVm`
(`crates/aero-wasm/src/tiered_vm.rs`), which its own header calls "CPU-only (not a
full-system machine)". `aero_machine::Machine` — the only full-system path, native
*and* browser — uses none of it. It runs Tier-0 plus `crate::jit::IrBlockCache`,
an IR *interpreter*. So compiled code never executes in either place, and the
browser desktop is limited by the same interpreter the native build is.

Note also that `WasmtimeBackend` is a reference oracle, not an engine: it is
created with a 128KiB linear memory and copies the whole of guest RAM on *every*
block execution to support rollback (`backend/wasmtime.rs`). That is fine at test
scale and unusable at guest scale. The browser backend has no such copy — JS
signals rollback through a commit flag — so the browser is the nearer of the two
to a working compiled path.

**The IR cache that does exist is nearly dormant.** Its hotness counter was
sampled once per outer-loop batch, at whichever RIP the 1024-instruction budget
happened to expire at — an arbitrary address, not a block entry. Over 100M
instructions that discovered 230 blocks and ran **0.14%** of instructions from the
cache. Growing the cache along executing chains (recording the RIP where a chain
falls off the end of the cache, which is by construction hot) raises that to ~1.4%
and 843 blocks. Throughput is unchanged, and arithmetic says it must be: the IR
interpreter is 3.86x Tier-0 (`crates/aero-machine/tests/ir_vs_tier0_throughput.rs`
measures both on one loop and asserts they agree architecturally), so 1.4%
coverage can return at most ~1%, which compilation cost absorbs.

Coverage is entry-limited, not discovery-limited. A batch contains on the order of
200 taken branches and the cache is offered exactly one of them, so a compiled
block is entered only if the budget happens to expire on its entry.

#### What of this reaches the browser

The compiled tier drives Wasmtime natively. In the browser the engine *is* the
browser, reached through `WebAssembly.Module` and `WebAssembly.Instance` rather than a
host crate — and that is the whole of the difference. It sits behind
`aero_jit_x86::compiled_backend::CompiledBackend`, a trait compiled for **every**
target including `wasm32`, and `CompiledTier` is generic over it. Block discovery,
translation, dead-flag elimination, the hotness policy and chain execution are shared
code, not a port.

`aero_jit_wasm::browser_backend::BrowserBackend` implements that trait, and
`scripts/browser-tier1-smoke.mjs` runs it in headless Chromium: it builds the module
with `wasm-pack`, serves `apps/web/tier1-browser`, and asserts on the guest registers
after a block has run. The block it uses stores to guest memory and reads it back, so
the inline TLB is on the path — a block of pure register arithmetic would pass without
guest memory working at all. It also asserts the tier compiled a block, which is what
separates "the tier ran it" from "nothing happened".

Two things about the browser layout are worth knowing before touching it.

Guest RAM, `CpuState` and the JIT context are ordinary allocations in this module's
linear memory, and compiled blocks import that same memory — so `ram_base` is simply
where the guest RAM buffer happens to live, and a guest load compiles to a wasm load
against a pointer the emulator already holds.

`CpuState` still has to be copied into an ABI-shaped region rather than passed by
pointer, because compiled code reads the code-version table pointer and length from
the twelve bytes *after* it. Handing it a bare `&mut CpuState` leaves those reading
whatever the allocator put there; a nonzero length sends the version bump on every RAM
store to a wild address, which corrupts the heap rather than failing. That presented
as the browser hanging on any block containing a store while loads worked fine.

The refactor is behaviour-preserving on the native side: same blocks, same coverage,
same guest state as the interpreter.

The parts under it do, because the browser's Tier-1 compiles through
`aero-jit-wasm`, which calls the same `compile_tier1_block_with_options` and the
same code generator:

- the AF fix, so its compiled blocks agree with Tier-0 on flags a guest can observe
  across a tier boundary;
- `TEST` with an immediate and `MOVZX`, which between them removed a third of the
  blocks Tier-1 could not finish — a block it cannot finish is a wall for the
  browser's chains exactly as for the native ones;
- `eliminate_dead_flags`, which is applied in that shared entry point rather than in
  the native tier's caller, precisely so every consumer gets it.

`cargo check -p aero-jit-wasm --target wasm32-unknown-unknown` passes with all of
it. What the browser still lacks is the tiering *around* the code generator — the
seam, the chaining, guest RAM as linear memory — which is native-only today.

#### The compiled tier runs, and what it costs

`aero_machine::compiled_tier::CompiledTier` assembles the pieces below: hot blocks
are translated to WASM, chained through `execute_chain`, and reach guest RAM by
direct load because that RAM is the backend's linear memory. Translation comes from
the machine's own MMU, and the TLB is flushed before every chain — Tier-1 never
compiles `mov cr3` or `INVLPG`, so page tables cannot change inside one, and a store
to a PTE without an `INVLPG` leaves a stale TLB exactly as hardware does. Enabled
with `AERO_JIT_RAM=1 AERO_COMPILED_TIER=1`.

**It executes Windows correctly.** At a fixed budget the guest's architectural state
after compiled execution is *identical* to the interpreter's — same RIP, RSP, GPRs
and CR3 at 100K and at 1M instructions — and a 120M-instruction run completes with
no bugcheck. Two admission bugs had to be fixed first: a block whose entry
instruction Tier-1 cannot translate compiles to nothing but `exit_to_interp` at its
own RIP, which retires nothing and spins the machine forever; and retirement must be
counted from the chain rather than estimated, which `execute_chain`'s lookup makes
exact because it is called once per *completed* block.

**Offered every taken branch, it covers a third of the workload and saves ~15%.**
Entering only at outer-loop boundaries caps coverage at what one arbitrary RIP per
batch can reach — about 1.5%. `Tier0BranchSeam` offers the tier every taken branch
instead. Measured over 120M instructions on the desktop checkpoint:

| | blocks | covered | ns/insn | net |
|---|---|---|---|---|
| batch boundaries only | 125 | 1.5% | 57.7 | +189 ms (+0.8%) |
| every taken branch | 611 | **32.6%** | 100.9 | **+3887 ms (+15.4%)** |

Two things made that possible where the same seam failed for the IR tier. It is
deliberately **not generic**: making the batch generic over a hook type gives it a
second monomorphisation, which alone measured as a 4% regression on the untiered
path even with the hook compiled away. A pointer and a function pointer keep one
body, and the seam measures free when absent (1259.4 against 1257.8). And the CPU
and bus are passed *to* the callback rather than captured by it, because the batch
holds `&mut` to both for its whole duration.

The IR tier's version of this bugchecked the guest; this one does not, because the
fault was specific to `run_ir_chain`'s rollback — a block that stores and then
faults leaves the store applied. Compiled blocks do not have that shape: they write
through and exit at a precise instruction with RIP already correct.

##### The instructions Tier-1 could not decode were most of the ceiling

Coverage sat at 34.6% and the reason was not the hotness threshold. Instrumenting
where admission is refused showed 7.45M *chain misses* — a chain stopping because
its successor was not compiled — against only 171 blocks refused for ending in a
bail to the interpreter. A block Tier-1 cannot finish is a wall: every chain that
reaches it stops there, so a handful of them in hot paths cost far more coverage
than their number suggests.

Dumping the opcode that ended each bail named the culprits directly. The largest
was `TEST` with an immediate — `0xA8` (`TEST AL, imm8`) and `0xF6 /0` and `0xF7 /0`
(`TEST r/m, imm`). Tier-1 decoded `TEST r/m, reg` and `TEST eAX, imm32` but none of
those, and they are everywhere in real code. Adding all three:

| | coverage | ns/insn | net |
|---|---|---|---|
| before | 34.6% | 84.8 | +19.5% |
| after | **56.9%** | **56.6** | **+40.3%** |

Both numbers moved because they are the same effect: chains that used to stop at a
`test` now run through it, so more instructions are covered *and* the per-chain
overhead is amortised over more of them.

Adding the byte-width accumulator ALU forms (`0x04`/`0x0C`/`0x24`/... `op AL, imm8`,
also missing) was worth almost nothing by comparison — 56.6 to 55.5ns. Worth
recording: the two changes look equally plausible from the source, and only the
measurement separates them.

`MOVZX` (`0F B6`/`0F B7`) came next, and needed more than a decode case. Reading
narrow and writing wide looks like it should fall out of `Mov`, but IR values carry
a width and `WriteReg` insists the value match the register — `WriteReg width
mismatch: i16 -> i32`. Reusing `Trunc` to widen is rejected too, by design. So the
IR gained a `Zext`, which in WASM lowers identically to `Trunc` because locals are
i64 and a narrow value is already held with its high bits clear.

Every `IrInst` match in the tree is exhaustive, so adding the variant produced
compile errors at each site that needed a decision rather than silently defaulting
somewhere — including the reference x86 model in `tests/translate.rs`, which is the
differential oracle and would otherwise have started agreeing with a wrong
translation. Tier-2 refuses traces containing it rather than mis-lowering them.

That took coverage to **59.1%** at 54.5ns and dropped bail sites from 133 to 103,
for a tier net of **+42.1%**.

What still bails is dominated by `IN`/`OUT`, which must exit by nature.

##### Invalidating cached translations only when something invalidates them

The tier flushed the inline TLB before every chain. That is trivially correct and,
at 1.5% coverage, measured free — which is why it stayed. At 34.6% coverage it is
not: 89.6ns per guest instruction against 80.0 with the flush removed, about 11% of
the compiled path, because every chain re-fills the TLB through host calls.

`Mmu::tlb_generation` is a counter bumped by exactly the events that invalidate the
MMU's own TLB — `INVLPG`, a CR3 write, and a CR0/CR4/EFER change. It reaches the
tier through `CpuBus::jit_tlb_generation`, and the tier drops its cached
translations only when the counter moves. Deliberately conservative: a bump that
did not strictly need to happen costs a re-translation, a missed one leaves
compiled code on a stale physical page.

**89.6 to 84.8ns**, about half the gap to the unsafe measurement, while still
producing guest state identical to the interpreter. The tier's net goes to
**+19.5%**.

Worth noting how this was nearly missed: the same experiment run earlier, at 1.5%
coverage, said the flush cost nothing, and that was recorded as a settled result. It
was true of that regime and not of this one. Cost measurements taken at one coverage
should be re-taken when coverage changes.

##### Dropping flag writes nobody reads

x86 arithmetic defines six flags and most code reads none of them: a `cmp` feeds
one branch, and the arithmetic around it sets CF, PF, AF, ZF, SF and OF for nobody.
Tier-1 materialised every declared flag.

`tier1::eliminate_dead_flags` removes a flag write that a later write in the same
block shadows with no read in between. It needs no new IR and cannot change
observable state: flags are architecturally visible after the block, so all six are
live at the exit, and only a shadowed write goes. Helper calls reset liveness to
everything, since a helper can observe anything.

Applied to the compiled tier: **100.1 to 92.3ns per guest instruction**, 1,393 dead
writes dropped across 436 blocks, and coverage up to 34.6%. Net for the tier goes
from +15.6% to **+18.3%** of the run.

This is the cheap half of the classic fix. Real lazy flags defer the computation
and reconstruct a flag from its operands when something finally reads it, which
also wins across block boundaries — most of the gap to the 7ns floor is still
there.

##### Why coverage stops at 60%

Instrumenting the seam answers this directly. Over 120M instructions it is offered
9.54M branch targets and:

- declines **0** for a pending interrupt — the guard that looked like it might cost
  coverage costs none;
- declines **3.93M** because the target is not compiled;
- runs the rest.

The obvious reading is that the 103 blocks Tier-1 cannot finish absorb those
declines: 103 walls at ~38,000 hits each is almost exactly 3.93M. It is wrong.
Compiling the *prefix* of a bailing block up to its untranslatable instruction —
which is correct, since the compiled block leaves RIP at the bail and the
interpreter resumes there — added 67 blocks for 0.3 points of coverage and made the
compiled path slower, 38.8ns to 40.9ns. It was reverted.

So the declines are a long tail: many distinct addresses each reached a handful of
times. Windows runs an enormous variety of code once. No admission policy makes
cold code worth a ~600us compile, which is why lowering the thresholds compiles
four times as many blocks for one extra point of coverage.

**The 60% plateau is a property of the workload, not of the tuning**, and it is the
term that dominates the ceilings above: at 60%, even free compiled code caps the
machine near 12 MIPS.

##### Two costs that were mine, not the guest's

Profiling the tier once it was carrying real work turned up `[vdso]` at 1.3% and
`RegisteredType::clone` plus `malloc` at ~2.6%. Neither was emulation:

- `execute_chain` pulled its `TypedFunc` out of the table with `.cloned()` — a
  refcount bump and an allocation once per block — because the table and the store
  are both fields of the backend and the obvious loop borrows one while the other is
  in use. Borrowing them as *disjoint fields* removes the clone entirely.
- The per-path cost accounting read the clock twice per chain entry. Useful, and
  not free at millions of entries; it is behind `AERO_CT_TIMING` now.

Together: **53.1 to 38.8ns per compiled guest instruction**, nearly all of it the
clone. Worth stating plainly — the instrumentation added to find where time went
was itself 1.3% of the run, and the clone was in code written earlier in the same
session. Profiles of a tier should be read for the tier's *own* overhead before its
emulation cost.

##### What the remaining per-instruction cost is not

Compiled code costs ~53ns per guest instruction against ~7ns for the same engine on
a synthetic chained loop. A block is about four instructions, so ~212ns per block:
roughly 69ns of host call, and the rest the body — which includes reloading the
block's registers from `CpuState` into locals on entry and storing them back on
exit, six memory operations of overhead around four instructions of work.

The obvious cheap fix is a self-looping block: emit the loop inside the compiled
function and neither the host call nor the reload is paid per iteration. Measured
first, and it is not there — **76,875 self-loop steps out of 17.5M block
executions, 0.4%**. Hot code here is chains of *distinct* blocks, so removing that
overhead needs general superblock formation — compiling a run of successors into one
function with state loaded once — rather than a special case.

That is the remaining lever, and it is the one that would also help the browser,
since it lives in the shared code generator.

##### Where the remaining time goes

Profiling with the tier live puts 7.4% of the run in `CompiledTier::run` — but
annotating it shows the hot instruction is `callq *(%r14)`, the call into the
compiled function, plus Wasmtime's thread-local entry sequence. That is not
overhead to remove; it is the per-block call boundary, attributed to the caller
because JIT code is not symbolised. Chaining already amortises the CPU-state
marshalling across a chain, but a host call per *block* remains, which is what
`call_indirect` between compiled blocks inside one module would remove.

Compiled code costs ~100ns per guest instruction against the 7ns the same engine
reaches on a synthetic ALU loop. That gap — not coverage — is where the next order
of magnitude lives: block-to-block calls inside WASM, and code quality (lazy flags,
the register allocator in the unused Tier-2).

Two guesses about that 7.4% were measured and disproven, which is the reason to
annotate rather than reason about it: flushing the inline TLB before every chain
costs nothing (101.0ns/insn with, 107.7 without), and allocating the translator
closure per chain entry cost nothing either (98.6 before removing it, 100.1 after).
What *did* pay was the same fix the IR cache needed — the tier's RIP-keyed maps used
the default SipHash and were probed on every taken branch through the seam;
`RipHasher` removed it from the profile entirely.

##### Two correctness holes found by turning coverage up

Lowering the admission threshold to 4 panicked: `index out of bounds: the len is
2147549184 but the index is 18446736380380979196`. The slow memory helpers
(`env.mem_read_*` / `env.mem_write_*`) index linear memory with the raw guest
address — an identity mapping. The inline TLB translates; its fallback did not, so
a kernel address indexed far outside the linear memory. Higher thresholds had
simply never compiled a block that took that path, which is the worst kind of
latent bug: correct-looking until coverage rises.

The first fix was to refuse such blocks, which was sound and cost more than half
the coverage — 32.6% down to 8.6%. That is the measurement that matters: it showed
the earlier 33%-coverage result had been *achieved by running that unsound path*.
The helpers translate now, per byte, because the accesses that reach them are the
ones the inline TLB declined — notably page-crossing ones, whose halves live on
pages that need not be adjacent. Coverage came back to 32.6%, soundly.

**Admission threshold.** Cost and benefit
can be computed from a single run, which sidesteps the divergence problem entirely:
benefit is covered instructions times the per-instruction saving, cost is blocks
admitted times admission time. Both are measured, so no cross-configuration
comparison is needed. Swept against the desktop checkpoint over 120M instructions:

| hotness threshold | blocks | covered | ns/insn | net |
|---|---|---|---|---|
| 24 | 649 | 1.81% | 60.7 | −27 ms |
| 48 | 241 | 1.51% | 55.5 | +158 ms |
| 96 | 125 | 1.51% | 57.7 | **+189 ms** |
| 160 | 83 | 1.15% | 58.4 | +147 ms |

The shape is the whole lesson: lowering the threshold compiles *more* blocks and
covers *fewer* instructions, because the extra blocks are cold ones whose admission
is never repaid. With the seam the optimum moves down to 12, which is the default:
entry is cheap enough there that more blocks pay for themselves.

Compiled code runs at **57.7ns per guest instruction against the interpreter's
~210ns — 3.6x**. That is the number that matters, and unlike the IR tier's 3.86x it
is measured inside the real machine rather than on a flat bus. The ceiling on the
whole workload is coverage: at 1.5%, saving 73% of that is ~1.1%, and +189ms against
a 25s run is 0.75%. Raising coverage is now worth doing in a way it never was for
the IR tier, because the covered path is genuinely cheaper.

**What it cost to get there.** Measured honestly —
see the warning below about why the obvious measurement is wrong — over a window
where both configurations execute identical code, the tier costs **1852 host
instructions per guest instruction against 1324 for the interpreter**. Raising the
hotness thresholds until almost nothing compiles removes almost all of the
difference, which is what identifies compilation rather than execution as the cost.

`tests/compile_cost.rs` measures admission directly, in seconds rather than the
minutes a machine run takes, and locates it precisely:

| | per block |
|---|---|
| `Module::new`, Cranelift | 1090 us |
| `Module::new`, Winch | 577 us |
| `Linker::instantiate` | 0.3 us |

The module being compiled is 663 bytes. Compilation is essentially all of it, and
it is a cost per *module* rather than per instruction compiled — every block is its
own module today. Switching to Wasmtime's baseline compiler, Winch (worse code, far
cheaper to produce, which is the right side of this trade) nearly halves it and
takes the machine workload from 1811 to **1456**, still short of the interpreter's
1324.

`Tier1WasmCodegen::compile_blocks_with_options` now emits several blocks into one
module, which is worth **3.2x** — 592us to 185us per block at 64 blocks per module.
Less than hoped: only ~400us of the 592 was per-module overhead, and the remaining
~190us is genuine per-function compilation, which no amount of batching removes.
(A single-block module is byte-identical to what the codegen produced before, so
nothing else had to change.)

Two other things were measured and turned out not to matter, which is worth
recording so they are not tried again: Wasmtime's parallel compilation is not
enabled in this build, so it is not the cost; and flushing the inline TLB before
every chain — chosen for safety — costs nothing at all (101.0ns/insn with the flush
against 107.7 without). Tracking real invalidation points would have bought
nothing.

An earlier estimate here put admission at 217,000 host instructions (~36us). That
was wrong: it divided the extra cost seen in an 800,000-instruction window by a
block count taken from a 120M-instruction run. Two measurements of different things
divided into each other.

##### A measurement that lies

The obvious comparison — run both configurations to the same `--max-insts` and
difference two budgets — reports the tier as **10.9% faster**. That number is an
artifact and should not be believed.

Enabling the tier changes how many guest instructions run between outer-loop
boundaries, which changes when platform timers tick, which changes where interrupts
land. The runs are identical at 100K and 1M instructions and have diverged
completely by 5M: same instruction count, different RIP, different registers. After
that the two configurations are executing *different guest code*, and the cheaper
number belongs to the cheaper code, not the faster engine.

The Amdahl check is what catches it: the tier covered 1.2% of retired instructions,
and 1.2% coverage cannot yield a 10.9% whole-workload gain however fast the covered
path is. Any future comparison here has to either stay inside the window where the
guest states still match, or verify that they do.

#### Guest RAM now lives in the JIT's linear memory

The prerequisite for compiled code being worth anything — see the two sections
below — is that guest RAM *is* the WASM linear memory compiled blocks address, so a
guest load becomes a load inside the block rather than a call back out to the host.
That is done, and it cost nothing:

- `WasmtimeBackend::claim_guest_ram` hands out the one permitted host view of its
  linear memory. Exactly one may exist — compiled blocks writing the same bytes is
  the design, but two *host-side* views would be two `&mut` to one region — and
  enforcing that in the backend is what lets the wrapper be a safe constructor.
- `aero_machine::jit_ram::WasmGuestRam` implements `memory::GuestMemory` over that
  region, with the CPU state at the bottom and RAM above it from `RAM_BASE`
  (0x10000). Putting RAM above the state region keeps it clear of the 2GiB ceiling a
  signed `i32` ABI pointer would impose; `JitContext::ram_base` already exists to
  carry the offset to compiled code.
- No machine surgery was needed. `Machine::new_with_guest_memory` already accepted
  a `Box<dyn GuestMemory>`, so the whole change is choosing a different backing.

Verified on the desktop checkpoint with `AERO_JIT_RAM=1`: 2GiB of guest RAM in a
32,769-page linear memory, and the run is **byte-identical** to the normal backing
(same cached blocks, block executions and retired instructions at a fixed budget)
at **1233.0 against 1232.6** host instructions per guest instruction — no
measurable cost.

There is no shortcut around the address-translation machinery, which is worth
knowing before building for one. A compiled tier restricted to blocks that touch no
guest memory would need no inline TLB, no MMIO exit and nothing to roll back — but
on the desktop checkpoint only **852 guest instructions** live in such blocks. They
are 50% of cached blocks by count (424 of 843) and average two instructions each,
and a chain through them stops at the first successor that touches memory. Real x86
interleaves memory access far too finely for that to be a tier.

#### Compiled code can now follow the guest's own page tables

The backend's built-in inline-TLB behaviour identity-maps, which is only correct
for a guest that is not paging — and Windows pages from very early on. That made
the inline TLB, the one thing that turns a guest load into a load *inside* the
compiled block, unusable for the real machine.

`WasmtimeBackend::set_translator` installs the embedder's own translation:
`FnMut(vaddr, access) -> Option<(phys_page_base, flags)>`, called on a TLB miss,
with `None` meaning the access faults and the block should exit to the interpreter
to raise it architecturally. Alongside it, `set_ram_base` says where guest physical
0 lands in linear memory (0x10000 under the layout above), and `flush_tlb` bumps
the tag salt so every cached translation misses at once — what a CR3 write, an
`INVLPG`, or a paging-mode change needs. Missing one of those would leave compiled
code using a stale physical page, which corrupts a guest silently rather than
loudly, so it is worth being deliberate about the invalidation points.

Covered by `tests/backend_translator.rs`: a non-identity mapping is actually
followed by a compiled load, the result is cached rather than re-translated, a
flush forces re-translation, and embedders with no translator still get the
identity behaviour every other test in the crate relies on. These need
`--features tier1-inline-tlb`, because the code generator drops the inline TLB
without it and every access takes the helper path.

`CpuBus::jit_translate` supplies that translation from the machine's own MMU, with
`MemoryBus::is_ram` classifying the result so compiled code knows whether it may
address the page directly or must take an MMIO exit. Verified against a booted
guest rather than a harness: real Windows user-mode addresses (`0x77829e83`,
`0x7fefde711cb`) resolve to plausible physical pages, marked RAM.

What remains for a compiled tier is the tiering itself: compile hot blocks through
the machine's bus, drive them with `execute_chain`, point `JitContext::ram_base` at
`RAM_BASE`, and fill the inline TLB on miss. None of that is blocked on the memory
model any more.

#### What compiled code actually costs, and what blocks it

Measured by `crates/aero-jit-x86/tests/tier1_compiled_throughput.rs`, which runs
one guest ALU loop through Tier-0 and through x86-compiled-to-WASM under Wasmtime,
and checks both agree on the architectural result. Block length is a parameter, so
the two costs separate:

| instructions per block | Tier-0 | compiled | speedup |
|---|---|---|---|
| 5 | 97 ns/insn | 66 ns/insn | 1.5x |
| 24 | 93 ns/insn | 18.7 ns/insn | 5.0x |
| 64 | 90 ns/insn | 11.6 ns/insn | 7.8x |

Fitting `block_cost = fixed + per_insn × n` across those points:

- **compiled code costs about 7 ns per guest instruction — roughly 140 MIPS**
- **entering and leaving a block costs about 293 ns**

A second sweep confirms the fixed cost is not the guest-RAM snapshot: varying
`cpu_ptr` (which sets how many bytes the backend copies per block) gives a slope of
about 0.02 ns/byte and an intercept of ~345 ns at zero bytes. The fixed cost is the
round trip itself — marshalling the architectural registers into linear memory,
calling into WASM, and copying them back out.

#### Chaining blocks, and what it recovers

`WasmtimeBackend::execute_chain` marshals the architectural state once for a whole
chain instead of once per block: the state stays in linear memory, successive
blocks read each other's results directly, and only the next entry point comes back
to the host to be resolved. Measured on the same loop, with the chain bounded to
4096 blocks the way `run_ir_chain` is:

| instructions per block | Tier-0 | per-block | chained | chained vs Tier-0 |
|---|---|---|---|---|
| 5 | 9.9 MIPS | 1.4x | **49.8 MIPS** | **5.0x** |
| 12 | 10.2 MIPS | 3.2x | **84.2 MIPS** | **8.3x** |
| 24 | 10.7 MIPS | 4.9x | **107.4 MIPS** | **10.1x** |

At a realistic five-instruction block that is 1.4x → 5.0x from one change, and the
machine's whole-workload figure today is about 4 MIPS.

The same measurement with a load and a store per iteration (`AERO_BENCH_MEM=1`)
stays at about 10x, so helper-routed memory access does not by itself sink the
compiled path. That result does **not** transfer to the machine, and the difference
is the whole design question: in this benchmark the helper indexes a flat byte
array, while in `Machine` it would call `CpuBus` → `PagingBus` → MMU translation —
the interpreter's entire memory cost, plus a WASM-to-host boundary on every access.

Since roughly 21% of the machine's host instructions are already memory and MMU
work and ~53% is interpretation, a compiled tier that removes interpretation but
keeps that memory path is bounded by Amdahl at about 2x. Getting past that requires
guest RAM to live *in* the WASM linear memory with the inline TLB enabled, so a
guest load becomes a bounds-checked load in compiled code rather than a call out.
`Tier1WasmOptions` already supports this (`inline_tlb`, `memory_min_pages`,
`memory_shared`); what does not exist is a `Machine` whose guest RAM *is* that
linear memory. That is the change the headline number waits on.

About 69ns of per-block cost survives: the host-to-WASM call itself, which chaining
reduces the *count* of but does not remove — the blocks still return to the host to
have their successor looked up. Removing it means letting compiled blocks
`call_indirect` each other inside WASM against a shared function table, which is
what would approach the ~7ns/instruction the compiled code itself costs.

That earlier number explains every negative result on this page. Real basic blocks
are 3–5 instructions, so a ~293 ns round trip is amortised over ~30 ns of work: the
compiled path measures 1.5x at that block length, and a 1.5x tier cannot repay the
cost of reaching it. It is the same shape as the IR tier's failure, one order of
magnitude further along.

It also says the architecture is sound. The compiled code itself runs at ~140 MIPS
against the ~4 MIPS the machine achieves today — the ceiling is not the problem,
the per-block boundary is. **The change that matters is letting compiled blocks
branch directly to each other inside WASM instead of returning to the host between
them**, so one host call runs a long chain. Nothing else on this page — coverage,
hotness tuning, entry placement, a better interpreter tier — moves the number,
because all of them leave that round trip in place once per block.

Two caveats for whoever builds it. Block discovery stops at
`BlockLimits::default().max_insts` (64), so the table above is measured at the
limit, not past it; and the compiled code computes all six arithmetic flags for
every ALU op, which is a large part of the 7 ns — lazy flags (computing only what a
later instruction consumes) is the obvious next lever once chaining lands.

#### Why widening entry does not help, and what would

Two seams were built to fix that, and both were withdrawn. What they measured is
worth more than either of them.

The first executed a chain *inside* the Tier-0 loop at each taken branch. It
bugchecks a booted guest, and the cause is now known.

`run_ir_chain` rolls a faulting block's *CPU* state back and lets Tier-0
re-execute it, which is only sound if the block wrote nothing. It is not: the
adapter's stores land directly, so a block that stores and *then* faults on a
later operation leaves the store applied, and the re-execution reads back its own
write. `inc dword [mem]` increments twice. Confirmed by experiment — excluding
store-carrying blocks from chains makes the same seam run clean, and allowing them
resets the guest.

An earlier reading of these experiments wrongly claimed store rollback had been
ruled out; the run that "restricted to store-free blocks" only inspected the
chain's *entry* block, leaving successors free to write.

Excluding those blocks is not the fix. The block that matters most here — the
WinSetup E8 scan, 77% of expand samples — is `inc dword [r9+0x2ed8]`, exactly the
read-modify-write shape being excluded, and excluding all store blocks costs 5%
outright. The fix is to buffer a block's stores and commit them only once it
completes, with reads consulting the buffer so a block still sees its own writes.
That is bounded work: a block stores a handful of bytes.

The second seam only *signalled*: Tier-0 ended the batch when control reached a
cached entry, and the machine's existing post-batch path ran the chain with the
fresh bus and device re-poll it always had. That version is **correct** — no
bugcheck — and lifts coverage from 1.4% to about 25%. It is also four times
slower: 5100 host instructions per guest instruction against a 1267 baseline, and
still 2847 with entries gated on measured chain yield.

The arithmetic explains both results, and generalises:

- One outer-loop iteration — paging-bus rebuild, device re-poll, platform tick, AP
  run — costs about **23,700 host instructions**. Measured from the slope of
  `AERO_BATCH_INSTS`: 1376.6 host/guest at a 1024-instruction batch against 1354.9
  at 16384 gives `C·(1/1024 − 1/16384) = 21.7`.
- The IR interpreter is 3.86x Tier-0, which on this workload saves roughly **490
  host instructions per guest instruction** executed from the cache.
- So entering the cache pays for itself only once a chain covers about **48 guest
  instructions**.

Measured chains do not get near that: blocks average 3.1 guest instructions and a
chain stops at its first uncached successor.

But the deeper result came from running the in-batch seam, which pays *no* entry
cost at all, at several coverage levels:

| hotness threshold | coverage | host insns / guest |
|---|---|---|
| 8 | 8.7% | 1976 |
| 50 | 13.6% | 1606 |
| 200 | 16.5% | 1521 |
| 2000 | 20.4% | 1321 |

Cost is not monotonic in coverage — 20.4% is far cheaper than 8.7% — because the
variable those thresholds really move is how many blocks get *translated*, most of
which never run hot. And the best of them, 1321, is still worse than the 1267
baseline even before subtracting the ~4% that making the batch function generic
costs on its own.

So the IR tier is, at best, cost-neutral in this machine at any coverage. The
3.86x from `ir_vs_tier0_throughput` is a `FlatTestBus` artifact: with no paging or
MMU, Tier-0's decode and dispatch dominate and the IR interpreter wins easily. In
the machine both paths pay the same memory costs, which dominate, and the IR
path's per-op interpretation overhead and chain bookkeeping cancel the rest.

**An interpreter tier cannot pay for itself here** — not because entry is
expensive, though it is, but because the tier underneath is not actually cheaper
on the workload that matters. Compiled Tier-1 blocks are not a better interpreter;
they remove interpretation. That is the difference that matters, and it is the
reason to spend effort wiring `JitRuntime` into `Machine` rather than on anything
described above.

#### Three costs removed from the Tier-0 hot loop

Instruction-level profiling of the interpreter (`perf annotate` on the batch
closure, rather than symbol-level self time) found three things that are not
inherent to interpreting x86, all of them paid once per guest instruction:

- **A `memcpy` call to pack the decode key.** `decode_key` copied the fetch window
  with a runtime length, which cannot inline and became a libc `memmove` call. The
  common case is a full 15-byte window, so it now copies a constant 15 bytes.
- **A `memcpy` call to fetch the window itself.** `MemoryBus::read_physical` takes
  a slice, so the callee only ever sees a runtime length. A `read_insn_window`
  method returning `[u8; 15]` lets the contiguous-RAM backend — already reachable
  through `get_slice` — copy a constant size inline. Added to both `MemoryBus`
  traits with a delegating default, so backends that do not care are unaffected.
- **A 128-bit variable shift per decode-cache probe.** `slot_matches` built its
  comparison mask as `(1u128 << (len * 8)) - 1`. A 128-bit shift by a runtime
  amount lowers to a `shrd` pair with branches; a single `shrd` was over 6% of the
  loop. It is a 16-entry `const` table now.

A fourth followed from the second, and doubled it. `read_insn_window` was
implemented at both *ends* of the bus chain — `PhysicalMemoryBus`, which can copy a
constant size out of contiguous RAM, and `PerCpuSystemMemoryBus`, which the
interpreter calls — but not in the two wrappers between them, `SystemMemory` and
`aero_platform::memory::MemoryBus`. The default implementation there fell back to
the slice form, so the fast path was never reached. Forwarding it through both took
the gain from 2.5% to **5.1%**, and `PhysicalMemoryBus::read_physical` disappeared
from the profile entirely. Worth remembering when adding a method to a trait with a
useful default: the default silently swallows the optimisation at every level that
does not override it.

Together these are **5.1% off Tier-0 and 5.4% off the `--jit` path** on the desktop
checkpoint — 1325.8 to 1257.8, and 1266.5 to 1198.1 host instructions per guest
instruction. Small against the gap, but they are the whole workload rather than the
1.4% a tier covers, and they stay fixed.

A fifth was tried and reverted: memoising the *host pointer* of the fetch page
alongside its translation, to skip the wrapper chain entirely. `aero-cpu-core` is
`#![forbid(unsafe_code)]`, which cannot be locally overridden and should not be
worked around. Caching the page's *bytes* instead is not an option either — x86
has a coherent instruction cache, so a guest that writes its own code must see the
change, and a byte cache would break that.

A caution for anyone continuing this: much of what a flat `perf annotate` blames on
one instruction is sample skid onto the first instruction after a branch join. The
`shrd` above was real — the fix moved the number — but the next candidate on the
same list turned out to be a whole basic block's cost attributed to the
instruction that happened to follow it.

#### Two costs found in the IR interpreter's own hot path

Profiled in isolation rather than as 1.4% of a machine run, the IR path spent about
3% of its time in `calloc`/`free` and another 3% in SipHash:

- `execute_block_cpu` sized its IR-value buffer with `vec![0u64; n]` — a zeroing
  allocation and a free **per block execution**, on a path whose whole purpose is
  to be cheaper than interpreting the same instructions one at a time.
  `execute_block_cpu_with_scratch` now takes a caller-owned buffer, and
  `run_ir_chain` hoists one out of its loop.
- `IrBlockCache`'s maps are keyed by guest address and looked up once per block.
  They used the default SipHash, which exists to resist collision attacks on
  untrusted keys — irrelevant for a RIP. They now use a multiply-shift hasher.

Together those took the IR interpreter from 33.6 to 26.4 ns per guest instruction,
**22% faster**, measured by `ir_vs_tier0_throughput`. It does not move the machine
number and cannot: at 1.4% coverage, 22% off the covered fraction is 0.3%. It does
lower the break-even chain length above, and it removes a real cost from whatever
eventually drives this interpreter.

The differential harness built to rule the above out did find one real bug, below.

#### Tiers must agree on undefined flags

`SHL`/`SHR`/`SAR` leave AF architecturally undefined. Tier-0 clears it; Tier-1 —
both the IR interpreter and the WASM code generator — deliberately preserved it,
each with a comment saying the value was undefined and so did not matter.

It does matter. A guest can set flags in one tier and read them in another, and on
real hardware that is one machine, so the tiers have to make the *same* choice
even where the architecture permits any. Tier-0's choice is the reference:
`shift_flag_set` in `tier1/translate.rs` now declares AF for true shifts (not
rotates, which leave it alone on both sides), and both Tier-1 backends write zero.
Declaring it in the `FlagSet` rather than doing it silently also keeps Tier-2's
flag elimination honest, since that pass uses the set to decide what a block
writes.

This was invisible while the cache covered 0.14% of instructions. Any future work
that widens JIT coverage should expect more of this class, and the differential
approach — run a block under both engines from one state, compare — is what finds
it.

### Interpreter Optimization

The canonical interpreter is **Tier-0** (`aero_cpu_core::interp::tier0`), which is designed to be the
baseline execution tier shared by:

- the unit-test harness (`exec::Vcpu` + `exec::Tier0Interpreter`), and
- future tiered/JIT execution (`exec::ExecDispatcher`).

Tier-0’s hot loop is essentially repeated calls to `interp::tier0::exec::step(&mut CpuState, &mut impl CpuBus)`
plus out-of-line handling for:

- assists (`aero_cpu_core::assist`), and
- interrupt/exception delivery (`aero_cpu_core::interrupts`).

See [`cpu-and-jit.md`](cpu-and-jit.md) for the concrete execution-loop shape and the
JIT ABI constraints around `CpuState`.

> Note: A legacy “threaded interpreter” implementation exists behind `--features legacy-interp`
> (`aero_cpu_core::cpu` + `aero_cpu_core::bus`). It is not the primary path and is mainly kept for
> regression benchmarking.

```rust
/// (Legacy, `legacy-interp`) Threaded interpreter using a dispatch table.
pub struct ThreadedInterpreter {
 // Dispatch table indexed by opcode
 dispatch_table: [fn(&mut CpuState, &mut MemoryBus, u64) -> u64; 256],
}

impl ThreadedInterpreter {
 pub fn run(&mut self, cpu: &mut CpuState, memory: &mut MemoryBus) {
 loop {
 let opcode = memory.read_u8(cpu.rip);
 let next_ip = self.dispatch_table[opcode as usize](cpu, memory, cpu.rip);
 cpu.rip = next_ip;
 
 // Check for events periodically
 if cpu.instruction_count % 1000 == 0 {
 if self.check_events(cpu) {
 break;
 }
 }
 }
 }
}

// Pre-decode common instruction sequences
pub struct DecodedBlock {
 instructions: Vec<DecodedInstruction>,
 cached_addresses: Vec<u64>,
}
```

---

## Memory Optimization

### JIT Memory Access Fast Path (Inline TLB + Direct RAM)

The single biggest baseline-JIT overhead in an emulator is usually **memory helper calls** (virtual→physical translation + MMIO routing) emitted for every guest load/store.

For common-case RAM accesses we should instead generate:

- **inline TLB lookup** against a compact JIT-visible TLB stored in linear memory
- **direct WASM `load/store`** from the guest RAM region on hit
- **clean exits** back to the runtime for MMIO / ROM / unmapped addresses
- **slow-path translation** (`mmu_translate(cpu_ptr, jit_ctx_ptr, vaddr, access)`) only on TLB miss or permission failure

This reduces imported calls from “every memory op” to “once per page” for tight loops, and turns the hot path into a small amount of integer work plus a final linear-memory load/store.

Key metrics to track:

- `mmu_translate_calls / guest_mem_ops` (should approach `~1 / (page_bytes / access_bytes)` for streaming loops)
- `jit_exit_mmio_calls` (should match actual device accesses; validate that it does **not** trigger for RAM)
- instruction throughput delta for synthetic memory loops (interpreter vs baseline JIT vs optimized JIT)

### TLB Optimization

```rust
pub struct OptimizedTlb {
 // Hot entries (most recently used)
 hot_cache: [TlbEntry; 16],
 hot_count: usize,
 
 // Main TLB (set-associative)
 main_cache: [[TlbEntry; 4]; 256], // 1024 entries, 4-way
 
 // Large page TLB
 large_cache: [TlbEntry; 32],
}

impl OptimizedTlb {
 #[inline(always)]
 pub fn lookup(&mut self, vaddr: u64) -> Option<u64> {
 let vpn = vaddr >> 12;
 
 // Check hot cache first (most common case)
 for entry in &self.hot_cache[..self.hot_count] {
 if entry.vpn == vpn && entry.valid {
 return Some(entry.ppn | (vaddr & 0xFFF));
 }
 }
 
 // Check main cache
 let set = (vpn as usize) & 0xFF;
 for entry in &self.main_cache[set] {
 if entry.vpn == vpn && entry.valid {
 // Promote to hot cache
 self.promote_to_hot(*entry);
 return Some(entry.ppn | (vaddr & 0xFFF));
 }
 }
 
 // Check large page TLB
 let large_vpn = vaddr >> 21; // 2MB pages
 for entry in &self.large_cache {
 if entry.vpn == large_vpn && entry.valid && entry.large_page {
 return Some(entry.ppn | (vaddr & 0x1F_FFFF));
 }
 }
 
 None
 }
}
```

### Memory Access Batching

```rust
/// Batch memory operations for better cache utilization
pub struct MemoryBatcher {
 pending_reads: Vec<(u64, usize, usize)>, // (addr, size, result_idx)
 pending_writes: Vec<(u64, usize, u64)>, // (addr, size, value)
 results: Vec<u64>,
}

impl MemoryBatcher {
 pub fn queue_read(&mut self, addr: u64, size: usize) -> usize {
 let idx = self.results.len();
 self.results.push(0);
 self.pending_reads.push((addr, size, idx));
 idx
 }
 
 pub fn flush(&mut self, memory: &MemoryBus) {
 // Sort by address for sequential access
 self.pending_reads.sort_by_key(|(addr, _, _)| *addr);
 
 for (addr, size, idx) in &self.pending_reads {
 self.results[*idx] = memory.read(*addr, *size);
 }
 
 // Process writes
 self.pending_writes.sort_by_key(|(addr, _, _)| *addr);
 for (addr, size, value) in &self.pending_writes {
 memory.write(*addr, *size, *value);
 }
 
 self.pending_reads.clear();
 self.pending_writes.clear();
 }
}
```

### Sparse Memory with Lazy Allocation

```rust
pub struct LazyMemory {
 // Page table (1 entry per 4KB page)
 pages: Vec<Option<Box<[u8; 4096]>>>,
 total_pages: usize,
 allocated_pages: usize,
 
 // Zero page for unallocated reads
 zero_page: [u8; 4096],
}

impl LazyMemory {
 pub fn read(&self, addr: u64) -> u8 {
 let page_idx = (addr >> 12) as usize;
 let offset = (addr & 0xFFF) as usize;
 
 match &self.pages[page_idx] {
 Some(page) => page[offset],
 None => 0, // Unallocated = zero
 }
 }
 
 pub fn write(&mut self, addr: u64, value: u8) {
 let page_idx = (addr >> 12) as usize;
 let offset = (addr & 0xFFF) as usize;
 
 // Allocate on first write
 if self.pages[page_idx].is_none() {
 self.pages[page_idx] = Some(Box::new([0u8; 4096]));
 self.allocated_pages += 1;
 }
 
 self.pages[page_idx].as_mut().unwrap()[offset] = value;
 }
}
```

---

## Graphics Optimization

### Draw Call Batching

```rust
pub struct DrawBatcher {
 batches: Vec<DrawBatch>,
 current_batch: Option<DrawBatch>,
}

impl DrawBatcher {
 pub fn add_draw(&mut self, draw: DrawCall) {
 if let Some(ref mut batch) = self.current_batch {
 if batch.can_merge(&draw) {
 batch.merge(draw);
 return;
 }
 self.batches.push(self.current_batch.take().unwrap());
 }
 self.current_batch = Some(DrawBatch::from(draw));
 }
 
 pub fn flush(&mut self, encoder: &mut GPUCommandEncoder) {
 if let Some(batch) = self.current_batch.take() {
 self.batches.push(batch);
 }
 
 // Sort batches to minimize state changes
 self.batches.sort_by_key(|b| (b.pipeline_hash, b.texture_hash));
 
 let mut current_pipeline = 0;
 let mut current_texture = 0;
 
 for batch in &self.batches {
 if batch.pipeline_hash != current_pipeline {
 encoder.set_pipeline(&batch.pipeline);
 current_pipeline = batch.pipeline_hash;
 }
 if batch.texture_hash != current_texture {
 encoder.set_bind_group(0, &batch.textures);
 current_texture = batch.texture_hash;
 }
 
 encoder.draw(batch.vertex_count, batch.instance_count, 0, 0);
 }
 
 self.batches.clear();
 }
}
```

### Shader Compilation Caching

```rust
pub const CACHE_SCHEMA_VERSION: u32 = 1;

pub struct ShaderCache {
 // In-memory (session) cache for WebGPU objects we cannot persist.
 compiled: HashMap<ShaderKey, CompiledShader>,
 
 // In-memory cache for derived artifacts (WGSL + reflection).
 derived: HashMap<CacheKey, CachedShaderTranslation>,
 
 // Persistent cache (IndexedDB; optional OPFS indirection for large blobs).
 persistent: PersistentGpuCache,
 
 // Compilation queue
 pending: VecDeque<ShaderCompileRequest>,

 telemetry: ShaderCacheTelemetry,
}

impl ShaderCache {
 /// Returns a compiled shader module, persisting only CPU-derived artifacts.
 pub async fn get_shader(&mut self, dxbc: &[u8], device: &GPUDevice) -> Result<&CompiledShader> {
 let key = ShaderKey::from_bytecode(dxbc); // includes backend_kind + schema_version
 let cache_key = CacheKey::new(CACHE_SCHEMA_VERSION, BackendKind::DxbcToWgsl, dxbc, device);
 
 // (1) Persistent cache first (cross-session derived artifacts).
 if let Some(cached) = self.persistent.get(&cache_key).await? {
 self.telemetry.persistent_hits += 1;
 self.telemetry.bytes_read += cached.size_bytes;

 // Corruption defense: validate cached WGSL with Naga.
 if validate_wgsl_with_naga(&cached.wgsl).is_ok() {
 return Ok(self.compile_from_translation(key, cached, device));
 }

 // Treat as miss and delete corrupted entry.
 self.persistent.remove(&cache_key).await?;
 }
 self.telemetry.persistent_misses += 1;

 // (2) In-memory cache for derived artifacts.
 if let Some(cached) = self.derived.get(&cache_key).cloned() {
 return Ok(self.compile_from_translation(key, cached, device));
 }

 // (3) Translate + reflect + validate, then populate caches.
 let translated = translate_and_reflect_dxbc(dxbc)?;
 validate_wgsl_with_naga(&translated.wgsl)?;
 self.persistent.put(&cache_key, &translated).await?; // may trigger LRU eviction
 self.telemetry.bytes_written += translated.size_bytes;
 self.derived.insert(cache_key, translated.clone());
 
 Ok(self.compile_from_translation(key, translated, device))
 }

 fn compile_from_translation(
 &mut self,
 key: ShaderKey,
 translated: CachedShaderTranslation,
 device: &GPUDevice,
 ) -> &CompiledShader {
 // If we already compiled this module this session, reuse it.
 if let Some(existing) = self.compiled.get(&key) {
 return existing;
 }

 let module = device.create_shader_module(&GPUShaderModuleDescriptor {
 code: &translated.wgsl,
 });

 self.compiled.insert(key.clone(), CompiledShader {
 module,
 wgsl: translated.wgsl,
 reflection: translated.reflection,
 });
 self.compiled.get(&key).unwrap()
 }
}
```

Key points:

- The persistent cache stores **WGSL + reflection**, not `GPUShaderModule`/pipelines.
- Cached WGSL is **validated via Naga** before use.
- Persistent cache is **bounded** via a byte budget + LRU timestamps.
- Expose a `clear_cache()` API and emit telemetry (hit/miss, bytes read/written) to verify behavior.

### Framebuffer Optimization

```rust
pub struct OptimizedFramebuffer {
 // Double buffering
 front: GPUTexture,
 back: GPUTexture,
 
 // Dirty region tracking
 dirty_regions: Vec<Rect>,
 
 // Previous frame for delta encoding
 previous_frame: Option<Vec<u8>>,
}

impl OptimizedFramebuffer {
 pub fn present(&mut self, queue: &GPUQueue) {
 if self.dirty_regions.is_empty() {
 return; // No changes
 }
 
 // Only copy dirty regions
 for region in &self.dirty_regions {
 queue.copy_texture_to_texture(
 &self.back,
 &self.front,
 region.to_extent(),
 );
 }
 
 self.dirty_regions.clear();
 std::mem::swap(&mut self.front, &mut self.back);
 }
 
 pub fn mark_dirty(&mut self, region: Rect) {
 // Merge overlapping regions
 for existing in &mut self.dirty_regions {
 if existing.intersects(&region) {
 *existing = existing.union(&region);
 return;
 }
 }
 self.dirty_regions.push(region);
 }
}
```

---

## I/O Optimization

### Async Storage with Prefetching

Note: this is illustrative pseudocode. In the actual repo:

- The canonical synchronous disk trait is `aero_storage::VirtualDisk` (plus `StorageBackend` for
 resizable byte stores).
- The async browser host layer uses TypeScript `AsyncSectorDisk` and Rust/wasm `st_idb::DiskBackend`.
- Avoid introducing new disk/backend traits; see [`20-storage-trait-consolidation.md`](storage.md).

```rust
pub struct PrefetchingStorage {
 backend: Box<dyn DiskBackend>,
 cache: LruCache<u64, Vec<u8>>,
 prefetch_queue: VecDeque<u64>,
 pending_reads: HashMap<u64, oneshot::Receiver<Vec<u8>>>,
}

impl PrefetchingStorage {
 pub async fn read_sector(&mut self, lba: u64) -> Vec<u8> {
 // Check cache
 if let Some(data) = self.cache.get(&lba) {
 // Prefetch next sectors
 self.queue_prefetch(lba + 1);
 self.queue_prefetch(lba + 2);
 return data.clone();
 }
 
 // Check pending reads
 if let Some(receiver) = self.pending_reads.remove(&lba) {
 return receiver.await.unwrap();
 }
 
 // Read from backend
 let data = self.backend.read_sectors(lba, 1).await;
 self.cache.put(lba, data.clone());
 
 // Prefetch
 self.queue_prefetch(lba + 1);
 self.queue_prefetch(lba + 2);
 
 data
 }
 
 fn queue_prefetch(&mut self, lba: u64) {
 if !self.cache.contains(&lba) && !self.pending_reads.contains_key(&lba) {
 let (tx, rx) = oneshot::channel();
 self.pending_reads.insert(lba, rx);
 
 // Spawn prefetch task
 let backend = self.backend.clone();
 spawn_local(async move {
 let data = backend.read_sectors(lba, 1).await;
 tx.send(data).ok();
 });
 }
 }
}
```

### Network Packet Coalescing

```rust
pub struct PacketCoalescer {
 pending_packets: Vec<Vec<u8>>,
 max_delay: Duration,
 last_flush: Instant,
}

impl PacketCoalescer {
 pub fn queue_packet(&mut self, packet: Vec<u8>) {
 self.pending_packets.push(packet);
 
 // Flush if too many packets or too much time
 if self.pending_packets.len() >= 10 || 
 self.last_flush.elapsed() > self.max_delay {
 self.flush();
 }
 }
 
 pub fn flush(&mut self) {
 if self.pending_packets.is_empty() {
 return;
 }
 
 // Combine packets into single WebSocket message
 let combined = self.encode_combined(&self.pending_packets);
 self.websocket.send(&combined);
 
 self.pending_packets.clear();
 self.last_flush = Instant::now();
 }
}
```

---

## Threading Optimization

### Work Distribution

```rust
pub struct WorkDistributor {
 cpu_worker: Worker,
 gpu_worker: Worker,
 io_worker: Worker,
 jit_worker: Worker,
}

impl WorkDistributor {
 pub fn distribute(&mut self) {
 // CPU emulation runs continuously on dedicated worker
 // GPU work submitted in batches
 // I/O operations are async with callbacks
 // JIT compilation happens in background
 
 // Balance work based on queue depths
 let cpu_queue_depth = self.cpu_worker.queue_depth();
 let gpu_queue_depth = self.gpu_worker.queue_depth();
 
 if gpu_queue_depth > cpu_queue_depth * 2 {
 // GPU bottleneck - reduce graphics quality
 self.reduce_graphics_quality();
 } else if cpu_queue_depth > 1000 {
 // CPU bottleneck - prioritize JIT compilation
 self.jit_worker.increase_priority();
 }
 }
}
```

### Lock-Free Communication

```rust
pub struct LockFreeQueue<T> {
 buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
 head: AtomicUsize,
 tail: AtomicUsize,
 capacity: usize,
}

impl<T> LockFreeQueue<T> {
 pub fn push(&self, value: T) -> Result<(), T> {
 let tail = self.tail.load(Ordering::Relaxed);
 let next_tail = (tail + 1) % self.capacity;
 
 if next_tail == self.head.load(Ordering::Acquire) {
 return Err(value); // Queue full
 }
 
 unsafe {
 (*self.buffer[tail].get()).write(value);
 }
 
 self.tail.store(next_tail, Ordering::Release);
 Ok(())
 }
 
 pub fn pop(&self) -> Option<T> {
 let head = self.head.load(Ordering::Relaxed);
 
 if head == self.tail.load(Ordering::Acquire) {
 return None; // Queue empty
 }
 
 let value = unsafe {
 (*self.buffer[head].get()).assume_init_read()
 };
 
 self.head.store((head + 1) % self.capacity, Ordering::Release);
 Some(value)
 }
}
```

---

## Profiling and Monitoring

### Nightly benchmark history

> **These workflows no longer exist.** Continuous integration was purged and is
> on the ban list — see [the working agreements](../meta/working-agreements.md).
> What follows describes the jobs as they were configured, which is still the
> best record of *what should be run and in what combination*; run them by
> hand. Nothing here runs automatically.

This repo includes a versioned performance history (`bench/history.json`) and a scheduled workflow (`.github/workflows/perf-nightly.yml`) that:

- runs a stable browser benchmark suite (Playwright/Chromium via `tools/perf/`)
- appends results into the `history.json` time series (mean + variance indicators)
- publishes a static dashboard (trend graphs + commit links) as a workflow artifact and (optionally) to `gh-pages`

### Built-in Profiler

#### Instruction counting semantics

- `instructions_executed` counts **retired guest architectural instructions**.
 - This is intentionally **not** micro-ops.
 - x86 string/`REP*` instructions retire as **1** architectural instruction; use
 `rep_iterations` (or a similar auxiliary counter) to track iteration counts
 if needed.
- For performance, CPU workers should maintain **non-atomic local counters** in
 the interpreter/JIT hot loops and **batch flush** to shared/atomic totals at a
 coarse granularity (e.g. every 1k–10k retired instructions).
- For JIT execution, attribute counts at **block granularity** using a
 precomputed `block_instruction_count` per compiled block.
- Compute MIPS as `instructions_delta / wall_time_delta / 1e6` and report both
 the rolling-window average and p95 to smooth jitter and catch tail latency.

```rust
pub struct Profiler {
 samples: Vec<ProfileSample>,
 current_frame: FrameProfile,
 enabled: bool,
 // Graphics profiling is split out so we can attribute frame time to
 // (a) CPU-side translation/encoding vs (b) GPU-side execution.
 //
 // GPU timing is best-effort: it is only populated when the backend
 // supports timestamp queries and they are explicitly enabled.
 gpu_timing_supported: bool,
 gpu_timing_enabled: bool,
}

#[derive(Default, Clone)]
pub struct GraphicsFrameMetrics {
 // Command buffer / state churn
 pub draw_calls: u32,
 pub render_passes: u32,
 pub pipeline_switches: u32,
 pub bind_group_changes: u32,

 // CPU → GPU upload pressure
 pub upload_bytes: u64, // buffers + textures

 // CPU-side cost of producing a frame
 pub cpu_translate_time: Duration, // DirectX → WebGPU translation
 pub cpu_encode_time: Duration, // WebGPU command encoding

 // Best-effort GPU-side execution time (timestamp-query), else None.
 pub gpu_time: Option<Duration>,
}

pub struct FrameProfile {
 start_time: Instant,
 cpu_time: Duration,
 // Best-effort. When WebGPU timestamp queries are unavailable, this is None
 // and exports should encode this as `null` (not 0.0ms).
 gpu_time: Option<Duration>,
 io_time: Duration,
 jit_time: Duration,
 instructions_executed: u64,
 rep_iterations: u64,
 graphics: GraphicsFrameMetrics,
 memory_allocated: usize,
}

impl Profiler {
 pub fn begin_frame(&mut self) {
 self.current_frame = FrameProfile {
 start_time: Instant::now(),
 ..Default::default()
 };
 }
 
 pub fn end_frame(&mut self) {
 let frame_time = self.current_frame.start_time.elapsed();
 
 // Log slow frames
 if frame_time > Duration::from_millis(33) {
 log::warn!(
 "Slow frame: {:?} (CPU: {:?}, GPU: {:?}, IO: {:?})",
 frame_time,
 self.current_frame.cpu_time,
 self.current_frame.gpu_time,
 self.current_frame.io_time
 );
 }
 
 self.samples.push(ProfileSample {
 frame_time,
 profile: self.current_frame.clone(),
 });
 }
 
 pub fn get_stats(&self) -> ProfileStats {
 let recent = &self.samples[self.samples.len().saturating_sub(100)..];
 
 ProfileStats {
 avg_frame_time: recent.iter().map(|s| s.frame_time).sum::<Duration>() / recent.len() as u32,
 avg_fps: 1.0 / recent.iter().map(|s| s.frame_time.as_secs_f64()).sum::<f64>() * recent.len() as f64,
 avg_mips: recent.iter().map(|s| s.profile.instructions_executed).sum::<u64>() as f64 /
 recent.iter().map(|s| s.frame_time.as_secs_f64()).sum::<f64>() / 1_000_000.0,
 }
 }
}
```

### JIT optimisation analysis metrics

To optimize a tiered JIT you need visibility into **(a)** compile cost (per tier and per pass), **(b)** cache behavior (is the JIT saving work or thrashing), and **(c)** stability (deopts/guard failures).

the JIT telemetry surface defines a small, cheap-to-update telemetry surface that the rest of the performance tooling (the performance HUD and JSON export HUD + JSON export, the guest CPU benchmark suite benchmarks) can consume.

#### Metrics to collect (exported totals + rolling)

The minimum required set is:

| Metric | Kind | Unit | Notes |
| --- | --- | --- | --- |
| `jit.tier1.compile_time_total` | duration | ms | Total time spent compiling Tier 1 (baseline) blocks this run |
| `jit.tier2.compile_time_total` | duration | ms | Total time spent compiling Tier 2 (optimizing) blocks this run |
| `jit.tier2.pass.<name>.time_total` | duration | ms | Total time spent in major Tier 2 passes (`const_fold`, `dce`, `regalloc`, …) |
| `jit.tier1.blocks_compiled_total` | counter | blocks | Count of compiled Tier 1 blocks/regions |
| `jit.tier2.blocks_compiled_total` | counter | blocks | Count of compiled Tier 2 blocks/regions |
| `jit.cache.capacity_bytes` | gauge | bytes | Configured code cache capacity |
| `jit.cache.used_bytes` | gauge | bytes | Current code cache usage (after inserts/evictions) |
| `jit.cache.lookup_hit_total` | counter | lookups | Cache lookup found an existing compiled block |
| `jit.cache.lookup_miss_total` | counter | lookups | Cache lookup did not find compiled code and triggered compilation |
| `jit.deopt_total` | counter | events | Number of deoptimizations (Tier 2 → Tier 1/0) |
| `jit.guard_fail_total` | counter | events | Number of guard failures (if guards are implemented) |

**Rolling metrics** (derived from the totals above) are required for HUD and for "what changed right now?" debugging:

| Rolling view | Derived from | Use |
| --- | --- | --- |
| `jit.cache_hit_rate_1s` | hits/misses deltas over 1s | Detect cache thrash or poor block formation |
| `jit.compile_ms_per_s_1s` | compile-time delta over 1s | Detect compile spikes and tune thresholds |
| `jit.blocks_compiled_per_s_1s` | block count delta over 1s | Detect pathological compilation churn |

The "1s" window is an example; the performance HUD and JSON export can choose another HUD window (e.g. last 60 frames) as long as it is consistent and documented in exports.

#### Data model + hot-path considerations

The JIT cache lookup is on the execution hot path, so updates must be **amortized and lock-free**:

- Use simple **counters** for events (`lookup_hit_total`, `lookup_miss_total`, `blocks_compiled_total`, `deopt_total`).
- Use **durations accumulated as integer nanoseconds** (or milliseconds) for compilation work; compilation runs off-thread, but we still want cheap aggregation.
- Prefer **per-thread counters** (or per-worker counters) that are merged periodically into a global view, to avoid high-frequency atomics on every lookup.

One workable shape (illustrative):

```rust
/// Collected by the JIT worker and periodically merged into the global telemetry sink.
pub struct JitMetricsTotals {
 // Cache behavior.
 pub cache_lookup_hit_total: u64,
 pub cache_lookup_miss_total: u64,
 pub cache_install_total: u64,
 pub cache_evict_total: u64,
 pub cache_invalidate_total: u64,
 pub cache_stale_install_reject_total: u64,
 pub compile_request_total: u64,

 // Tier distribution.
 pub tier1_blocks_compiled_total: u64,
 pub tier2_blocks_compiled_total: u64,

 // Compile time (accumulated).
 pub tier1_compile_ns_total: u64,
 pub tier2_compile_ns_total: u64,

 // Tier 2 pass breakdown (accumulated).
 pub tier2_pass_const_fold_ns_total: u64,
 pub tier2_pass_dce_ns_total: u64,
 pub tier2_pass_regalloc_ns_total: u64,

 // Deopts/guards.
 pub deopt_total: u64,
 pub guard_fail_total: u64,

 // Code cache footprint.
 pub code_cache_capacity_bytes: u64,
 pub code_cache_used_bytes: u64,
}
```

Compilation-phase timings should be captured at **pass boundaries** (not per instruction/IR node), so instrumentation stays negligible relative to actual compile work.

The repository includes a small, self-contained implementation of these counters + exports in `crates/aero-perf` (`aero_perf::jit` and `aero_perf::telemetry`). `crates/perf` remains as a deprecated compatibility shim that re-exports the same modules and includes a synthetic demo you can run via:

```sh
cargo run --locked -p perf --example jit_metrics_demo
```

#### Reporting integration with the HUD and JSON export

the performance HUD and JSON export should expose these metrics in two surfaces:

1) **HUD (small panel)**

Required fields:

- `JIT cache hit`: `hits / (hits + misses)` over rolling window
- `JIT blocks`: total blocks compiled (optionally `(+X/s)` from rolling window)
- `JIT compile`: compile ms/s over rolling window (sum Tier1+Tier2), with optional breakdown (`t1=… t2=…`)

2) **JSON export**

Include a `jit` section that contains both totals (since run start) and a rolling snapshot used by the HUD.

Example shape:

```json
{
 "jit": {
 "enabled": true,
 "totals": {
 "tier1": { "blocks_compiled": 1234, "compile_ms": 87.4 },
 "tier2": {
 "blocks_compiled": 56,
 "compile_ms": 45.1,
 "passes_ms": { "const_fold": 4.2, "dce": 6.7, "regalloc": 18.9 }
 },
 "cache": {
 "lookup_hit": 98765,
 "lookup_miss": 1234,
 "install": 5678,
 "evict": 123,
 "invalidate": 45,
 "stale_install_reject": 3,
 "compile_request": 789,
 "capacity_bytes": 268435456,
 "used_bytes": 104857600
 },
 "deopt": { "count": 0, "guard_fail": 0 }
 },
 "rolling": {
 "window_ms": 1000,
 "cache_hit_rate": 0.9875,
 "compile_ms_per_s": 3.4,
 "blocks_compiled_per_s": 12.0
 }
 }
}
```

The exact JSON schema is less important than: (a) stability, (b) explicit units, (c) totals + rolling included together.

#### Benchmark/regression linkage

the guest CPU benchmark suite benchmarks should print a one-line summary of key `jit` metrics alongside throughput numbers so regressions can be attributed quickly (e.g. "MIPS dropped because Tier2 compile ms doubled").

See "Performance Benchmarks" in [Testing Strategy](testing.md) for suggested output format.

#### Verification expectations

the JIT telemetry surface should be validated with at least:

- A synthetic benchmark that executes enough distinct blocks to force compilation → **non-zero compile ms and block counts**.
- A run with JIT compilation disabled (interpreted-only mode) → **all `jit.*` counters and durations remain 0**, and the code path avoids doing cache lookups/instrumentation work.

### Graphics bottleneck analysis

To optimize rendering we need to answer:

1. **Are we CPU-bound?** (DirectX translation, state setup, WebGPU command encoding, uploads)
2. **Are we GPU-bound?** (shader execution / fragment workload / bandwidth)
3. **What patterns are expensive?** (too many draws, too many pipeline/bind churn, too much upload)

graphics bottleneck analysis adds per-frame and rolling-window metrics to make those questions measurable.

#### Core per-frame counters

Collected every frame and exported both as **instantaneous** values and as a **rolling window** (e.g. last 60 frames):

| Metric | Meaning | Typical source |
| --- | --- | --- |
| `draw_calls` | Count of `draw*()` issued | WebGPU render/compute pass wrapper |
| `render_passes` | Count of `beginRenderPass()` | WebGPU command encoder wrapper |
| `pipeline_switches` | Count of pipeline changes (state churn) | `setPipeline()` wrapper + "last pipeline" tracking |
| `bind_group_changes` | Count of bind group changes | `setBindGroup()` wrapper + per-index "last bind group" tracking |
| `upload_bytes` | Bytes uploaded from CPU to GPU (buffers + textures) | `queue.writeBuffer`, `queue.writeTexture`, staging uploads |
| `cpu_translate_time` | CPU time in DirectX → WebGPU translation | timer around translator entrypoints |
| `cpu_encode_time` | CPU time in WebGPU command encoding | timer around command encoder/pass building |

#### WebGPU instrumentation touch points

To avoid sprinkling counters throughout the renderer, route WebGPU access through a thin "instrumented" facade:

```rust
pub struct InstrumentedCommandEncoder<'a> {
 inner: wgpu::CommandEncoder,
 metrics: &'a mut GraphicsFrameMetrics,
}

impl<'a> InstrumentedCommandEncoder<'a> {
 pub fn begin_render_pass<'p>(
 &'p mut self,
 desc: &wgpu::RenderPassDescriptor<'p>,
 ) -> InstrumentedRenderPass<'p> {
 self.metrics.render_passes += 1;
 let inner = self.inner.begin_render_pass(desc);
 InstrumentedRenderPass::new(inner, self.metrics)
 }
}

pub struct InstrumentedRenderPass<'a> {
 inner: wgpu::RenderPass<'a>,
 metrics: &'a mut GraphicsFrameMetrics,
 last_pipeline: Option<u64>,
 last_bind_groups: [Option<u64>; 4],
}

impl<'a> InstrumentedRenderPass<'a> {
 fn new(inner: wgpu::RenderPass<'a>, metrics: &'a mut GraphicsFrameMetrics) -> Self {
 Self {
 inner,
 metrics,
 last_pipeline: None,
 last_bind_groups: [None, None, None, None],
 }
 }

 pub fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
 let id = pipeline.global_id(); // any stable identifier works here
 if self.last_pipeline != Some(id) {
 self.metrics.pipeline_switches += 1;
 self.last_pipeline = Some(id);
 }
 self.inner.set_pipeline(pipeline);
 }

 pub fn set_bind_group(&mut self, index: u32, bind_group: &wgpu::BindGroup, offsets: &[u32]) {
 let id = bind_group.global_id();
 let slot = index as usize;
 if slot < self.last_bind_groups.len() && self.last_bind_groups[slot] != Some(id) {
 self.metrics.bind_group_changes += 1;
 self.last_bind_groups[slot] = Some(id);
 }
 self.inner.set_bind_group(index, bind_group, offsets);
 }

 pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
 self.metrics.draw_calls += 1;
 self.inner.draw(vertices, instances);
 }
}
```

Uploads are counted at the queue/resource layer:

```rust
pub struct InstrumentedQueue<'a> {
 inner: &'a wgpu::Queue,
 metrics: &'a mut GraphicsFrameMetrics,
}

impl<'a> InstrumentedQueue<'a> {
 pub fn write_buffer(&mut self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
 self.metrics.upload_bytes += data.len() as u64;
 self.inner.write_buffer(buffer, offset, data);
 }

 pub fn write_texture(
 &mut self,
 dst: wgpu::ImageCopyTexture,
 data: &[u8],
 layout: wgpu::ImageDataLayout,
 size: wgpu::Extent3d,
 ) {
 self.metrics.upload_bytes += data.len() as u64;
 self.inner.write_texture(dst, data, layout, size);
 }
}
```

#### Best-effort GPU timing (timestamp-query)

When supported, GPU timing uses timestamp queries to measure **GPU execution time** without guessing from CPU submission time:

- Request the WebGPU `timestamp-query` feature only when a runtime flag enables GPU timing.
- If the feature is unavailable, keep all other graphics metrics active and export `gpu_time_ms: null`.
- Prefer **asynchronous readback** (e.g. ring-buffered query resolve/readback) so the profiler does not stall on `mapAsync`.

At a high level:

```rust
// frame start
encoder.write_timestamp(&query_set, START);

// ... encode major passes ...

// frame end
encoder.write_timestamp(&query_set, END);
encoder.resolve_query_set(&query_set, 0..2, &resolve_buffer, 0);

// submit, then read back later (N frames delayed) to populate `gpu_time`.
```

If we want per-major-pass timing, reserve additional query indices (e.g. `GBUFFER_START/END`, `COMPOSITE_START/END`) and export a pass breakdown alongside the per-frame total.

#### Perf HUD surfacing

Add a compact "graphics" panel to the on-screen perf HUD (values should be rolling-window averages):

- `Draws`: draw calls / frame
- `Pipes`: pipeline switches / frame
- `Upload`: MB/s (from `upload_bytes` / time window)
- `GPU`: indicator for timestamp query state
 - `GPU: on` when supported + enabled
 - `GPU: off` when supported but disabled
 - `GPU: n/a` when unsupported

#### JSON export schema (`graphics` section)

Graphics telemetry should be exported under a dedicated `graphics` section so perf tooling can attribute regressions:

```json
{
 "frame_index": 12345,
 "frame_time_ms": 16.7,
 "graphics": {
 "draw_calls": 102,
 "render_passes": 3,
 "pipeline_switches": 12,
 "bind_group_changes": 58,
 "upload_bytes": 1048576,
 "cpu_translate_ms": 1.2,
 "cpu_encode_ms": 0.9,
 "gpu_time_ms": null,
 "rolling": {
 "window_frames": 60,
 "draw_calls_avg": 98.4,
 "pipeline_switches_avg": 10.7,
 "upload_mib_per_s": 45.2
 },
 "gpu_timing": {
 "supported": false,
 "enabled": false
 }
 }
}
```

#### CI / benchmark considerations

- Keep GPU timing **opt-in** (e.g. config flag or environment variable) so headless CI and smoke perf runs remain stable.
- Treat `gpu_time_ms` as **informational** unless a benchmark explicitly enables GPU timing and asserts it.
### Hot-path identification

To guide interpreter and JIT optimization, Aero records **hot basic blocks** (by guest PC) with **bounded memory**:

- **Collection granularity:** update counters once per *basic-block entry* (not per instruction) to keep overhead low.
- **Attribution:** `hits += 1` and `instructions += decoded_block_len` for the block starting at `pc`.
- **Bounded Top-K:** use a streaming heavy-hitter algorithm (e.g. **Space-Saving**) to keep only the most relevant PCs.

The CPU worker periodically (or on-demand during perf capture/export) sends a snapshot to the main thread. The public export surface includes this under `window.aero.perf.export().hotspots`:

```js
{
 hotspots: [
 { pc: "0x1000", hits: 10000, instructions: 30000, percent_of_total: 96.77 },
 // ...
 ],
}
```

The HUD can render a simple “Hotspots” panel showing the top N PCs and their percentage of total executed instructions.

### Standard performance metric definitions

To avoid drift between in-app HUD, exported perf summaries, and benchmark tooling, compute metrics using a single shared implementation (`packages/aero-stats`) and the following definitions:

- `avg_fps = frames / total_time_s` (equivalently `1000 / mean_frame_time_ms`)
- `median_fps = 1000 / p50_frame_time_ms`
- `1% low FPS = 1000 / p99_frame_time_ms`
- `0.1% low FPS = 1000 / p99.9_frame_time_ms`
- Variance/stdev/CoV are computed over frame times (ms) using Welford’s algorithm (population variance).

---

## Next Steps

- See [Browser APIs](web-host.md) for platform-specific optimizations
- See [Testing Strategy](testing.md) for performance testing
- See [Performance Tooling](performance.md) for profiling HUD, trace capture, and benchmarks
- See [Task Breakdown](../history/sprint-era-record.md) for optimization tasks
- See [Guest CPU Instruction Throughput Benchmarks](performance.md) for instruction-throughput regression tracking

## Performance Tooling (Profiling, HUD, Traces, Benchmarks)

### Overview

Aero ships with **first-party performance tooling** that contributors should use for:

- **Interactive profiling** while developing locally (Perf HUD)
- **Bug reports / PR evidence** (JSON exports + traces)
- **Regression detection** (benchmarks + baselines in CI)

This doc describes the expected workflows and how to interpret the outputs.

---

### Perf tooling unit tests

Perf tooling source lives in `apps/web/src/perf/`, and its unit tests live alongside the implementation as `apps/web/src/perf/**/*.test.ts`.

These tests are executed in CI by the **root** Vitest suite. To run them locally (including coverage):

```bash
pnpm run test:unit:coverage
```

If a perf unit test needs DOM APIs (e.g. HUD tests), opt into `jsdom` per-file:

```ts
// @vitest-environment jsdom
```

### Perf HUD

The Perf HUD is a lightweight overlay that shows a rolling snapshot of key emulator metrics while the system is running.

#### Toggling the HUD

- **In-app UI:** Click the **Perf HUD** button in the developer menu.
- **Keyboard shortcut:** `F2` or `Ctrl+Shift+P` (when focus is *not* in a text input).

If you don’t see the toggle, make sure you’re running a build that includes the developer menu / HUD.

#### Update frequency / sampling window

To avoid the HUD itself becoming a source of overhead:

- The HUD **refreshes at a fixed rate** (currently **5 Hz**, i.e. every **200ms**) while visible.
- Most values are computed over a **rolling window** (a fixed-size ring buffer of recent samples).

#### Metrics (what they mean)

The exact fields may evolve, but contributors should expect the HUD to include:

| Metric | Meaning | How to use it |
| --- | --- | --- |
| **FPS (avg / 1% low)** | Average FPS plus “1% low” FPS (tail latency) | Tail drops are often more important than averages for “jank” |
| **Frame time (avg / p95)** | Mean and p95 frame time | Compare to 16.7ms (60Hz) / 33.3ms (30Hz); p95 captures stutters |
| **MIPS (avg / p95)** | Estimated guest throughput (million instructions/s), reported as avg / p95 | Useful for CPU/JIT regressions independent of rendering; p95 captures stutters |
| **CPU / GPU / IO / JIT (avg)** | Accounted time buckets (ms) | Use to attribute slow frames; buckets may not sum to frame time if work overlaps |
| **Draw calls (avg/frame)** | Approximate rendering work submitted | Spikes often correlate with state churn or missed batching |
| **IO throughput** | Bytes/sec of emulated I/O | Correlate stutters with disk/network activity |
| **Host heap** | JS heap used/total | Watch for leaks and GC-triggering spikes |
| **Guest RAM** | Guest memory size (if known) | Helps validate test configurations |
| **Capture** | Recording state + sample counts | Use Start/Stop to capture a window and Download to export it |

Interpretation tip: **one bad frame is not a regression**. Look for sustained changes in the rolling averages and for increases in long-frame counts.

#### Capture controls (Start / Stop / Reset / Download)

The HUD includes basic capture tooling:

- **Start / Stop:** begin/end a capture window (aim for ~5–15s around the issue).
- **Reset:** clears the current capture buffer.
- **Download:** downloads the current `window.aero.perf.export()` payload as JSON.

---

### Capturing and downloading JSON exports

JSON exports are meant to be attached to issues/PRs. They should include:

- Build metadata (commit SHA, build mode, feature flags)
- Environment metadata (browser version, OS, device info)
- Aggregated counters and histograms for the current run

#### Capturing a window of data

`window.aero.perf.export()` exports the **most recent capture buffer**. To collect a useful per-frame trace:

- Start a capture
- Reproduce the workload for ~5–15 seconds
- Stop the capture
- Export / download

You can do this either:

- via the Perf HUD buttons (**Start**, **Stop**, then **Download**), or
- via the console API:

```js
window.aero.perf.captureStart();
// Reproduce the workload for ~5–15 seconds.
window.aero.perf.captureStop();
const data = window.aero.perf.export();
```

#### Export from the DevTools console

Run this in the page console:

```js
const data = window.aero.perf.export();
```

Expected behavior:

- `export()` returns a plain JSON-serializable object.
- In builds that include the Perf HUD, clicking **Download** triggers a JSON download using this same payload.

#### Manual download (if your build doesn’t auto-download)

```js
const data = window.aero.perf.export();
const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
const a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = `aero-perf-${new Date().toISOString()}.json`;
a.click();
```

#### Perf export schema (canonical, v2)

`window.aero.perf.export()` returns a **versioned**, **JSON-serializable** object with:

- `kind: "aero-perf-capture"`
- `version: 2`

Top-level fields (v2):

- `build`: build metadata (`git_sha`, `mode`, and optional `features` when available)
- `env`: environment metadata (`now_epoch_ms`, `userAgent`, `platform`, `hardwareConcurrency`, `devicePixelRatio`, `webgpu`)
- `capture`: capture timing (`startUnixMs`, `endUnixMs`, `durationMs`)
- `capture_control`: capture bounds/counters (`startFrameId`, `endFrameId`, `droppedRecords`, `records`)
- `summary`: `{ frameTime, mipsAvg }` (frame-time summary + MIPS average)
- `frameTime`: `{ summary, stats }` (histogram payload; `stats` is `FrameTimeStats.toJSON()`)
- `records[]`: per-frame samples:
 - `tMs`: milliseconds since `captureStart()`
 - `frameTimeMs`
 - `instructions` (number when safe, otherwise string; `null` if unavailable)
 - `cpuMs`, `gpuMs`, `ioMs`, `jitMs`, `drawCalls`, `ioBytes` (nullable)
- `memory`: memory telemetry (`MemoryTelemetryExport`)
- `responsiveness`: input/long-task telemetry (`ResponsivenessExport`)
- `jit`: JIT telemetry snapshot (currently a placeholder in most builds, but always present)
- optional: `benchmarks` (attached by `apps/web/src/aero.ts` when bench suites are enabled)

The JSON schema used by CI and tooling lives at:

- `bench/schema/perf-output.schema.json`

CI perf workflows validate `perf_export.json` artifacts against this schema (using `tools/perf/validate_perf_export.mjs`) to catch breaking changes early.

---

### Trace mode (timeline capture)

Traces answer questions like:

- “What exactly was the main thread doing during a stutter?”
- “Are we blocked on GPU submission, I/O, or cross-worker synchronization?”
- “Is JIT compilation happening on the hot path?”

Aero traces should be emitted in **Chrome Trace Event** JSON format so they can be opened in:

- **Perfetto UI:** https://ui.perfetto.dev/
- **Chrome tracing:** `chrome://tracing`

#### Capturing a trace

In the DevTools console:

```js
window.aero.perf.traceStart();
// Reproduce the problem for ~5-15 seconds.
window.aero.perf.traceStop();
const trace = await window.aero.perf.exportTrace();
```

To download the trace as a file:

```js
const json = await window.aero.perf.exportTrace({ asString: true });
const blob = new Blob([typeof json === "string" ? json : JSON.stringify(json)], { type: "application/json" });
const a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = `aero-trace-${new Date().toISOString()}.json`;
a.click();
```

In builds that include the Perf HUD, the **Trace JSON** button performs this same export asynchronously via `exportTrace({ asString: true })`.
Large traces can take a moment to serialize (events are gathered from multiple workers), so the HUD disables the button while the export is in-flight.

Notes:

- Keep traces short (seconds, not minutes). Long traces are hard to analyze and expensive to record.
- Prefer capturing a trace immediately after a cold start if you’re investigating startup costs (shader compilation, caches).
- Trace recording uses bounded ring buffers; if you capture too long, you may see dropped events under `trace.otherData.aero.droppedRecordsByThread`.

For more details on trace instrumentation, worker lanes, and dropped-event accounting, see:

- [`performance.md`](performance.md)

#### Opening a trace in Perfetto

1. Save the trace JSON returned by `exportTrace()` to a file.
2. Open https://ui.perfetto.dev/
3. Drag-and-drop the JSON file into the UI.

#### Opening a trace in `chrome://tracing`

1. Open `chrome://tracing` in a Chromium-based browser.
2. Click **Load**.
3. Select the trace JSON file you saved from `exportTrace()`.

#### Capturing a trace from startup

Some builds support starting tracing automatically via URL param:

- `http://127.0.0.1:4173/?trace`

This is useful when investigating early startup costs (shader compilation, worker startup, cache warmup).

#### Interpreting lanes / tracks

Contributors should expect traces to include separate tracks for:

- **Main thread:** UI, input routing, canvas presentation, coordination
- **CPU worker:** guest instruction execution, translation/JIT, interrupts
- **GPU worker:** command encoding, texture/shader management
- **I/O worker:** storage reads/writes, network packets, audio buffers

Common patterns:

- **Wide slices on the main thread** often indicate rendering/UI work or accidental synchronous waits.
- **Periodic “sawtooth” patterns** in memory/GC regions often indicate allocation churn.
- **Gaps on the CPU worker** with activity elsewhere suggest the emulator is waiting (sync, I/O, frame pacing).

---

### Running benchmarks locally

Benchmarks are the **repeatable** way to validate performance changes. They should:

- Use a production-like build
- Run multiple iterations
- Warn when results are noisy

Related references:

- CI browser perf runner: [`tools/perf/README.md`](../decisions/README.md)
- Bench harness + GPU scenarios: [`bench/README.md`](../decisions/README.md)

#### Build + serve (production-like)

If you want to benchmark the *app* (not just synthetic microbenches), run against a built preview server.

```bash
pnpm install --frozen-lockfile

# Production bundle + preview server
pnpm run build
pnpm run preview
```

Note: the default preview server (`apps/web/bringup.html` → `apps/web/src/main.ts`) installs the Perf HUD and exposes
`window.aero.perf.captureStart/captureStop/export()` (and trace APIs). CI's Playwright perf runner
(`tools/perf/run.mjs`) relies on this API to write `perf_export.json`.

For perf-sensitive work (SharedArrayBuffer / threads / COOP+COEP), prefer the COI preview helper:

```bash
pnpm run serve:coi
```

#### Run the Playwright benchmark runner (CI parity)

The CI browser perf harness lives under `tools/perf/` and can be run locally.

One-time setup:

```bash
pnpm install --frozen-lockfile
node scripts/playwright_install.mjs chromium
```

On minimal Linux environments, you may need system dependencies:

```bash
node scripts/playwright_install.mjs chromium --with-deps
```

CI-parity run (build + `vite preview` + perf harness, matching `.github/workflows/perf*.yml`):

```bash
node scripts/ci/run_browser_perf.mjs --preview --out-dir perf-results/local --iterations 7
```

Quick run (no build; uses the perf harness' internal `data:` URL):

```bash
pnpm run bench:browser -- --iterations 7 --out-dir perf-results/local
```

Capture an Aero trace alongside the run (best-effort; requires the page to expose `window.aero.perf.traceStart/traceStop/exportTrace`):

```bash
node tools/perf/run.mjs --trace --out-dir perf-results/local --iterations 7 --url http://127.0.0.1:4173/
```

Capture a fixed-duration trace window (useful when you want a longer trace than a single benchmark run):

```bash
node tools/perf/run.mjs --trace-duration-ms 5000 --out-dir perf-results/local --iterations 7 --url http://127.0.0.1:4173/
```

Include app-provided microbenches (best-effort; requires `window.aero.bench.runMicrobenchSuite`):

```bash
node tools/perf/run.mjs --include-aero-bench --out-dir perf-results/local --iterations 7 --url http://127.0.0.1:4173/
```

What to expect:

- Results are written to `perf-results/` (raw samples + `summary.json`).
- Summaries include **median-of-N** numbers and a **CV** (coefficient of variation) to help spot noise.

To benchmark a specific URL (e.g. your local preview server), pass `--url`:

```bash
pnpm run bench:browser -- --iterations 7 --out-dir perf-results/local --url http://127.0.0.1:4173/
```

#### Run the GPU benchmark scenarios (Playwright + WebGPU/WebGL2)

The GPU bench runner executes scenario scripts in a real browser and emits a JSON report:

```bash
pnpm run bench:gpu -- --scenarios vga_text_scroll,vbe_lfb_blit --iterations 7 --headless false --output gpu_bench.json
```

Compare two GPU bench reports (baseline vs candidate) and emit a Markdown report (plus machine-readable JSON):

```bash
node --experimental-strip-types --import ./scripts/register-ts-strip-loader.mjs scripts/compare_gpu_benchmarks.ts \
 --baseline gpu_bench_base.json \
 --candidate gpu_bench_head.json \
 --out-dir gpu_bench_compare \
 --thresholds-file bench/perf_thresholds.json \
 --profile pr-smoke
```

#### Run + compare the storage I/O benchmark suite (OPFS + IndexedDB)

The storage bench is a Playwright-driven macrobench (`bench/runner.ts storage_io`) that writes a raw `storage_bench.json`.

Note: this suite measures host-side OPFS/IndexedDB throughput. IndexedDB is async-only and does not
currently back the synchronous Rust disk/controller path (`aero_storage::{StorageBackend, VirtualDisk}`).
See: [`19-indexeddb-storage-story.md`](storage.md) and
[`20-storage-trait-consolidation.md`](storage.md).

To reproduce the CI compare locally:

```bash
pnpm run bench:storage -- --out-dir storage-perf-results/base
pnpm run bench:storage -- --out-dir storage-perf-results/head

pnpm run compare:storage -- \
 --baseline storage-perf-results/base/storage_bench.json \
 --candidate storage-perf-results/head/storage_bench.json \
 --out-dir storage-perf-results/compare \
 --thresholds-file bench/perf_thresholds.json \
 --profile pr-smoke
```

Outputs:
- `storage-perf-results/compare/compare.md`
- `storage-perf-results/compare/summary.json` (machine-readable)

#### Run + compare the gateway benchmark suite (backend networking)

The gateway bench (`services/gateway/bench/run.mjs`) runs loopback-only benchmarks for Aero's backend networking paths:

- TCP proxy RTT (p50/p90/p99)
- TCP proxy throughput (MiB/s)
- DoH QPS + cache hit ratio

It does **not** require Playwright (pure Node + local sockets), but it does require the gateway build artifacts in `services/gateway/dist/`:

```bash
pnpm -C services/gateway run build

node services/gateway/bench/run.mjs --mode smoke --json gateway-perf-results/base/raw.json
node services/gateway/bench/run.mjs --mode smoke --json gateway-perf-results/head/raw.json

node --experimental-strip-types --import ./scripts/register-ts-strip-loader.mjs scripts/compare_gateway_benchmarks.ts \
 --baseline gateway-perf-results/base/raw.json \
 --candidate gateway-perf-results/head/raw.json \
 --out-dir gateway-perf-results/compare \
 --thresholds-file bench/perf_thresholds.json \
 --profile pr-smoke
```

Outputs:
- `gateway-perf-results/compare/compare.md`
- `gateway-perf-results/compare/summary.json` (machine-readable)

#### Interpreting summary output and variance warnings

When the summary shows a high **CV** (coefficient of variation):

- **Re-run** with more iterations (e.g. 15+).
- Ensure your machine is not thermally throttling and nothing heavy is running in the background.
- Prefer the same browser version you use for baselines.

Treat small single-digit % changes as suspicious unless variance is low and the change reproduces across runs.

To reproduce the CI comparison locally (baseline vs candidate):

```bash
node tools/perf/run.mjs --out-dir perf-results/base --iterations 7
node tools/perf/run.mjs --out-dir perf-results/head --iterations 7

node tools/perf/compare.mjs \
 --baseline perf-results/base/summary.json \
 --candidate perf-results/head/summary.json \
 --out-dir perf-results/compare \
 --thresholds-file bench/perf_thresholds.json \
 --profile pr-smoke
```

---

### Updating baselines and thresholds

Baselines are used to detect regressions in CI. Update them when:

- You intentionally improve performance (so CI stops failing once the improvement lands)
- A known, unavoidable regression is accepted (with explicit acknowledgement)
- The baseline environment changes (e.g. major browser version change)

In CI:

- The **browser CI perf** workflow (`tools/perf`) compares **PR vs base commit** (no committed “golden baseline” file).
- Thresholds live in [`bench/perf_thresholds.json`](../../bench/perf_thresholds.json) (versioned, shared across CI perf tooling: browser/GPU/storage/gateway/Node microbench/Criterion).
 - PR gating uses profile `pr-smoke`
 - Nightly runs should use profile `nightly`

Separately, the checked-in benchmark baseline adds a **checked-in baseline** file (`bench/baseline.json`) and a Node microbench compare tool (`node bench/compare`).
It uses the same threshold policy file (`bench/perf_thresholds.json`) under the `node` suite.

#### Updating the benchmark baseline (`bench/baseline.json`)

Use the baseline updater script:

```bash
pnpm run bench:update-baseline -- --scenario all --iterations 15
```

This re-runs the lightweight Node microbench (`bench/run.js`) and updates `bench/baseline.json` in-place, printing a
before/after summary table to make PR review easier.

Sanity check the updated baseline:

```bash
pnpm run bench:node
node bench/compare --fail-on-regression --json
```

Guidelines:

- Update baselines on a **quiet, stable machine**.
- Include the **before/after summary** in the PR description.
- Avoid “baseline churn”: if results are noisy, fix the source of noise first.

---

### CI perf jobs

> **These workflows no longer exist.** Continuous integration was purged and is
> on the ban list — see [the working agreements](../meta/working-agreements.md).
> What follows describes the jobs as they were configured, which is still the
> best record of *what should be run and in what combination*; run them by
> hand. Nothing here runs automatically.

CI performance jobs are split by intent:

- **PR perf (gating):** ``/.github/workflows/perf.yml`` runs a small browser-only suite and compares PR vs base commit.
- **Nightly perf (non-gating + data collection):** ``/.github/workflows/perf-nightly.yml`` runs more iterations and publishes history/dashboard artifacts.
- **PR GPU perf (gating):** ``/.github/workflows/gpu-perf.yml`` runs a small GPU scenario set and compares PR vs base commit.
- **PR storage perf (gating):** ``/.github/workflows/storage-perf.yml`` runs the storage bench and compares PR vs base commit.
- **PR gateway perf (gating):** ``/.github/workflows/gateway-perf.yml`` runs the Aero Gateway bench and compares PR vs base commit.
- **PR CPU microbenches (gating):** ``/.github/workflows/bench.yml`` runs Criterion benchmarks and compares PR vs base commit.

Where to find artifacts:

- In GitHub Actions, open the workflow run and download artifacts such as:
 - JSON exports (`*.json`)
 - Trace captures (`trace.json`)
 - Benchmark summaries (`summary.json`, `compare.md`, history dashboards)

What CI currently uploads (subject to change):

- **PR perf:** `perf-smoke-<run_id>` (includes baseline + candidate summaries and `compare.md`)
- **Nightly perf:** `perf-nightly-<run_id>` (browser perf run output) and `perf-history-dashboard` (time-series history + static dashboard bundle)

For the canonical repo, the nightly workflow also publishes the dashboard to `gh-pages` (GitHub Pages) so trends are visible without downloading artifacts.

---

### Pitfalls / best practices

#### Noisy machines

Performance measurements are extremely sensitive to background load.

Best practices:

- Close CPU-heavy apps (video calls, IDE indexing, games).
- Plug laptops into power and disable battery-saver modes.
- Avoid running benchmarks right after boot (let background tasks settle).

#### Browser updates

Browser engine updates can cause large swings in WASM and WebGPU performance.

Best practices:

- Record the browser version (exports/traces should include it).
- Re-run baselines after major browser upgrades.

#### Headless vs headed differences

Headless runs can differ due to:

- Different compositor behavior
- Different GPU selection / SwiftShader fallback
- Different timer granularity and scheduling

Best practices:

- Use **headed Chromium** for local investigations unless you are specifically debugging CI.
- Note that CI perf intentionally runs **headless Chromium with GPU/WebGPU disabled** for stability; expect different numbers locally.
- Compare “headless vs headed” only against baselines collected the same way.

---

### Quick-start checklist (copy/paste friendly)

Capture a perf export:

1. Run Aero locally and reproduce the workload.
2. Start a capture (HUD **Start**, or run `window.aero.perf.captureStart()`).
3. Reproduce the issue for ~5–15 seconds, then stop capture (HUD **Stop**, or `window.aero.perf.captureStop()`).
4. Open DevTools → Console and run: `window.aero.perf.export()`
5. Save/download the JSON and attach it to your issue/PR.

Run the microbench benchmark locally:

1. `pnpm install --frozen-lockfile && node scripts/playwright_install.mjs chromium` (one-time; use `--with-deps` on minimal Linux hosts)
2. `pnpm run bench:run -- --scenario microbench --iterations 7`

Alternative (run the underlying runner directly, without the wrapper):

- `node tools/perf/run.mjs --out-dir perf-results/local --iterations 7`

To benchmark your locally served build instead of the built-in `data:` page:

1. `pnpm install --frozen-lockfile && pnpm run serve:coi`
2. `node bench/run --scenario microbench --iterations 7 --url http://127.0.0.1:4173/`

## Trace-mode Profiling (Chrome Trace / Perfetto Export)

### Overview

Aero includes an **opt-in trace capture mode** that records:

- Nested span begin/end events
- Counter events (e.g. instructions, bytes, draw calls)
- Instant events for notable state transitions (boot milestones, tier switches)

The capture can be exported in **Chrome Trace Event** format and viewed in:

- `chrome://tracing`
- https://ui.perfetto.dev

This document focuses on trace capture/export. For the broader performance tooling workflows (Perf HUD, perf exports, benchmarks, CI perf jobs), see:

- [`performance.md`](performance.md)

### API

When running on the main thread, the profiler is exposed as:

```ts
window.aero.perf.traceStart();
window.aero.perf.traceStop();
const trace = await window.aero.perf.exportTrace(); // Chrome Trace JSON object
```

`exportTrace({ asString: true })` returns a JSON string.

For quick manual testing, `apps/web/src/main.ts` also supports enabling tracing at boot via `?trace` in the URL.

### Workers / multi-thread timelines

To give each worker its own lane in the exported trace:

1) Register the worker on the main thread:

```ts
import { perf } from "./perf/perf";

const vmRuntime: "legacy" | "machine" = "legacy";
const cpuWorkerEntrypoint = vmRuntime === "machine" ? "./workers/machine_cpu.worker.ts" : "./workers/cpu.worker.ts";
const worker = new Worker(new URL(cpuWorkerEntrypoint, import.meta.url), { type: "module" });
perf.registerWorker(worker, { threadName: "cpu" });
```

`apps/web/src/runtime/coordinator.ts` registers the default `cpu/gpu/io/jit` workers automatically.

2) Install trace handlers in the worker entrypoint:

```ts
import { installWorkerPerfHandlers } from "./perf/worker";

void installWorkerPerfHandlers();
```

### Bounded memory & dropped events

Trace recording uses fixed-size ring buffers. When buffers overflow, older events are overwritten and
the number of overwritten records is reported via `trace.otherData.aero.droppedRecordsByThread`.

### Minimal initial instrumentation

Current placeholder instrumentation points:

- Main thread: requestAnimationFrame present loops (`apps/web/src/display/vga_presenter.ts`, plus any callers of `startFrameScheduler`)
- Main thread: `wasm:init` span for runtime WASM loader (`apps/web/src/main.ts`)
- Workers: `wasm:init` + `worker:init` spans (`apps/web/src/workers/*.worker.ts`, `apps/web/src/workers/gpu-worker.ts`)

### Audio telemetry (ring buffer health)

When `startAudioPerfSampling()` is enabled for an `EnabledAudioOutput`, the trace will contain counter (`ph: "C"`) events with:

- `audio.bufferLevelFrames` — current queued frames in the output ring buffer
- `audio.underrunFrames` — total missing output frames rendered as silence due to underruns
- `audio.overrunFrames` — total frames dropped due to producer writes exceeding available capacity
- `audio.sampleRate` — AudioContext sample rate
- `audio.baseLatencySeconds` — `AudioContext.baseLatency` (seconds) when available
- `audio.outputLatencySeconds` — `AudioContext.outputLatency` (seconds) when available

These counters are what we typically want to capture during manual end-to-end audio validation (for example, the Windows 7
in-box HDA driver smoke test in [`audio.md`](audio.md)).

See:

- `apps/web/src/platform/audio.ts` (`startAudioPerfSampling`, `getMetrics`)
- `apps/web/src/main.ts` (demo UI auto-starts sampling when audio output is initialized and `window.aero.perf` is present)

## Native interpreter throughput

This is the cost of Tier-0 executing one guest instruction in the native runner,
which sets the wall-clock price of every Windows bring-up experiment.

### Measure retired host instructions, not wall clock

Wall clock is not a usable signal on a shared development host: the same fixed
slice varies by roughly ±15% between runs depending on what else is executing.
**Retired host instructions per guest instruction** is stable to three
significant figures across runs, so it is the metric to optimise against.

Take it by running the same snapshot twice with different `--max-insts` budgets
and differencing the two `perf stat` instruction counts. The difference cancels
the fixed snapshot-load cost and leaves only the steady-state loop.
`tools/win7-bringup/bench-insns.sh` does exactly this, over as many binaries as
you pass it, so two interpreter builds can be compared directly. Cycles are
worth recording alongside it — they capture stalls and branch-prediction effects
that instruction counts miss — but they inherit some of wall clock's noise, so
treat a cycle win that contradicts the instruction count as unproven.

### What the number says

Tier-0 costs on the order of **1,300 host instructions per guest instruction** at
an IPC near 2.0. That IPC matters: it means the interpreter is **code-volume
bound, not stall bound**. There is no memory-latency problem to fix; there is
simply too much work per instruction, and the only thing that helps is executing
fewer host instructions.

Roughly: ~22% batch loop, ~19% operand decode and execute, ~12% physical memory,
~10% MMU translation, ~9% instruction decode.

### What moved it

Five changes took the figure from ~1,720 to ~1,270 (−26%), with cycles falling
further (−32%) because several of them removed branchy code:

1. **A release profile with `lto = "fat"` and `codegen-units = 1`.** The
   workspace previously had no `[profile.release]` at all. The hot path crosses a
   crate boundary on nearly every guest instruction — Tier-0 calls `CpuBus`,
   which is `aero-machine` → `aero-pc-platform` → `memory` → `aero-platform`
   before a byte of guest RAM is touched — and with default codegen units none of
   that inlines. This alone is worth about a fifth of the total.
2. **A decode cache that validates a hit once.** It previously compared
   instruction bytes twice per hit through a variable-length `memcmp`, as a
   borrow-checker workaround. It now packs the 15-byte fetch window into a `u128`
   and validates with a masked compare, and indexes with `ip ^ (ip >> 13)` so two
   hot code pages do not alias each instruction against each other. Size was
   measured, not guessed: 64 K entries retires fewer host instructions than 16 K
   (better hit rate) but costs more cycles because the slot array stops fitting in
   L2, so 16 K is the peak.
3. **Gating the fused-leaf matchers.** They ran on every instruction. The
   byte-prefix fuses now test the same packed `u128` the decode cache built, and
   the shape matchers are gated on the opening mnemonic they already required.
4. **Memoising the instruction-fetch translation.** Straight-line execution
   re-fetches from one page for many consecutive instructions; the resolved page
   is now remembered and dropped on anything that requires a TLB flush.
5. **Memoising opcode dispatch.** Dispatch walked nine module predicates in a
   fixed order, so a `mov` paid five and an SSE instruction paid all nine. The
   answer depends only on the mnemonic, so it is computed once and cached.

### What is not the bottleneck

The outer loop rebuilds the per-vCPU memory bus and paging bus and re-polls every
device backend at each batch boundary, which looks expensive. It is not:
`AERO_BATCH_INSTS` overrides the batch quantum, and 1024 against 8192 against
65536 measure within noise. The cost is genuinely per-instruction.

### Where the remaining headroom is

The ~19% spent deriving operand shape (`op_bits`, `calc_ea`, `read_op_sized`)
re-derives the same facts from the iced `Instruction` on every execution. Cutting
it means caching a pre-decoded operand plan — which is the IR path in
[cpu-and-jit.md](./cpu-and-jit.md), not further micro-tuning of Tier-0.

### How far this has to go, and why Tier-0 cannot get there

Around 1,300 host instructions per guest instruction puts Tier-0 roughly two to
three orders of magnitude off native. Published figures for comparable systems
say where a *compiled* tier lands: QEMU's TCG runs about 8× native, CheerpX about
20× on scalar code — and around 2,500× on SIMD-heavy code, which is the more
sobering number and a warning about where vector workloads land — and JSLinux's
in-browser x86-64 is user-measured near 50×.
WebAssembly itself costs only about 1.45–1.55× native, which is the important
part — **the browser is not what sets the floor here, the execution tier is.**

That is the argument for the JIT stated as a number rather than an intuition. No
amount of interpreter micro-optimisation closes a gap that large; the published
range for a compiled tier is 8–50×, and Tier-0 is nowhere near it by
construction. So the division of labour is deliberate: Tier-0's job is to be
correct and to make bring-up experiments cheap, and throughput is the JIT's job.
The JIT becomes the gate once the desktop milestone is held, with a differential
interpreter-versus-JIT harness as its entry condition — a fast tier that is
subtly wrong is worse than a slow tier that is right.

Sources: WebAssembly overhead, [Jangda et al., USENIX ATC'19](https://www.usenix.org/system/files/atc19-jangda.pdf);
TCG, [Linaro](https://www.linaro.org/blog/qemu-a-tale-of-performance-analysis/);
JSLinux and CheerpX, [user measurements](https://hn.nuxt.dev/item/47311484).

## Guest CPU instruction throughput benchmarks

### Overview

Host-side microbenchmarks (e.g. WASM loops) are useful for tracking runtime/compiler regressions, but they **do not measure the emulator’s actual core work** (fetch/decode/execute, memory emulation, tiered JIT overhead).

the guest CPU benchmark suite adds a **deterministic guest instruction throughput suite** that runs small x86/x86-64 instruction streams directly inside the CPU core (no OS images required).

Goals:

1. Measure **guest IPS/MIPS** for representative instruction mixes.
2. Work as soon as the interpreter exists (CI-001..CI-005) and scales to baseline/optimizing JIT.
3. Provide correctness guardrails via **checksums** (fast-but-wrong regressions fail the run).
4. Export results via `window.aero.perf.export()` and exercise via a Playwright scenario.

Non-goals:

 - Measuring full-system “Windows boot” performance (covered by image-based benchmarks).
 - Measuring host JS/WASM performance in isolation.

---

### Public API (JS)

Expose a JS-callable bench entrypoint:

```ts
type GuestCpuMode = "interpreter" | "jit_baseline" | "jit_opt";

type GuestCpuBenchVariant =
 | "alu64"
 | "alu32"
 | "branch_pred64"
 | "branch_pred32"
 | "branch_unpred64"
 | "branch_unpred32"
 | "mem_seq64"
 | "mem_seq32"
 | "mem_stride64"
 | "mem_stride32"
 | "call_ret64"
 | "call_ret32";

type GuestCpuBenchOpts = {
 variant: GuestCpuBenchVariant;
 mode: GuestCpuMode;

 // Preferred: run until seconds budget is reached (multiple runs).
 // Must be a finite number > 0.
 seconds?: number;

 // Optional: run a single *measured* payload invocation for N iterations.
 // The harness will also perform an unmeasured "reference run" (same variant/iters)
 // to derive `expected_checksum` and assert determinism.
 // This is mainly for debugging; the main suite uses `seconds`.
 // Must be an integer in the u32 range (> 0).
 iters?: number;
};

type GuestCpuBenchRun = {
 variant: GuestCpuBenchVariant;
 mode: GuestCpuMode;

 // Configuration
 iters_per_run: number;
 warmup_runs: number;
 measured_runs: number;

 // Correctness
 expected_checksum: string; // hex string (e.g. "0xf935...")
 observed_checksum: string;

 // Counters
 total_instructions: number; // u64-ish; in JS as number if safe or bigint
 total_seconds: number;

 // Result
 ips: number;
 mips: number;

 // Variance indicators (per-run)
 run_mips: number[];
 mips_mean: number;
 mips_stddev: number;
 mips_min: number;
 mips_max: number;
};

// JS API surface:
window.aero.bench.runGuestCpuBench(opts: GuestCpuBenchOpts): Promise<GuestCpuBenchRun>;
```

Notes:

 - If `seconds` is specified, the harness repeats deterministic runs until the budget is reached.
 - `seconds` and `iters` are mutually exclusive; if provided they must be finite and > 0.
 - If checksum verification fails, `runGuestCpuBench` **throws** (and Playwright marks the scenario failed).
 - `mode` selects interpreter vs baseline JIT vs optimizing JIT. If a mode is not implemented yet, the API should fail loudly (or optionally downgrade while clearly reporting the actual mode used).

---

### Guest payload ABI (minimal, OS-less)

Each payload is a flat blob of machine code that:

 - consumes an iteration count (`ECX`/`RCX`)
 - optionally consumes a pointer to a scratch buffer (`EDI`/`RDI`)
 - returns a checksum (`EAX`/`RAX`)
 - ends with `ret`

The harness is responsible for:

 - allocating + initializing a scratch memory buffer for memory variants
 - setting up CPU state (stack pointer, code location, execution mode)
 - selecting execution tier (interp/JIT)
 - counting retired instructions 

#### Why `ret`?

It avoids needing privileged instructions (`hlt`) or exception-based exits (`ud2`). The harness can start the CPU at the payload entrypoint with a valid stack and stop when it returns.

---

### Payload set (required)

All payloads are:

 - deterministic
 - self-contained (no syscalls, no I/O)
 - checksum-returning

The suite includes:

1. **Tight integer ALU loop** (`alu32`, `alu64`)
2. **Branch-heavy loop (predictable)** (`branch_pred32`, `branch_pred64`)
3. **Branch-heavy loop (unpredictable)** (`branch_unpred32`, `branch_unpred64`)
4. **Memory load/store (sequential)** (`mem_seq32`, `mem_seq64`)
5. **Memory load/store (strided)** (`mem_stride32`, `mem_stride64`)
6. **Call/ret heavy (stack)** (`call_ret32`, `call_ret64`)

#### Canonical checksum verification

To keep correctness checks fast and deterministic, the suite defines a canonical iteration count:

 - `iters_per_run = 10_000`

For each variant, we include:

 - the canonical machine code bytes
 - the expected checksum after `iters_per_run`

The harness should reset guest-visible state between runs (registers and, for memory benches, the scratch buffer) so that each run is identical and checksum verification remains valid.

---

### Payload definitions (assembly + bytes + expected checksums)

All byte arrays below are flat binaries assembled with NASM (e.g. `nasm -f bin file.asm -o file.bin`).

> **Important:** These payloads assume a *flat address space* from the harness. They are designed to be run inside the CPU core test harness, not in a full BIOS/OS boot flow.

#### `alu64`

Tight loop over integer ALU + shifts.

**Input:** `RCX = iters`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000):** `0xf935f9482b8f99b8`

Assembly:

```nasm
BITS 64
mov rax, 0x123456789ABCDEF0
mov rdx, 0x9E3779B97F4A7C15
.loop:
 add rax, rdx
 mov rbx, rax
 shr rbx, 13
 xor rax, rbx
 shl rax, 1
 dec rcx
 jnz .loop
ret
```

Bytes:

```text
0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x48, 0xba, 0x15, 0x7c, 0x4a, 0x7f,
0xb9, 0x79, 0x37, 0x9e, 0x48, 0x01, 0xd0, 0x48, 0x89, 0xc3, 0x48, 0xc1, 0xeb, 0x0d, 0x48, 0x31,
0xd8, 0x48, 0xd1, 0xe0, 0x48, 0xff, 0xc9, 0x75, 0xeb, 0xc3
```

---

#### `alu32`

32-bit variant of the ALU loop.

**Input:** `ECX = iters`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000):** `0x30aae0b8`

Assembly:

```nasm
BITS 32
mov eax, 0x9ABCDEF0
mov edx, 0x7F4A7C15
.loop:
 add eax, edx
 mov ebx, eax
 shr ebx, 13
 xor eax, ebx
 shl eax, 1
 dec ecx
 jnz .loop
ret
```

Bytes:

```text
0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0xba, 0x15, 0x7c, 0x4a, 0x7f, 0x01, 0xd0, 0x89, 0xc3, 0xc1, 0xeb,
0x0d, 0x31, 0xd8, 0xd1, 0xe0, 0x49, 0x75, 0xf2, 0xc3
```

---

#### `branch_pred64`

Branch-heavy loop with *predictable* conditional branches (always not taken), plus the loop back-edge.

**Input:** `RCX = iters`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000):** `0xd7ab5d5aaad6afab`

Assembly:

```nasm
BITS 64
mov rax, 0x123456789ABCDEF0
mov rbx, 0x9E3779B97F4A7C15
.loop:
 xor rdx, rdx
 jnz .skip_add
 add rax, rbx
.skip_add:
 xor rdx, rdx
 jnz .skip_xor
 xor rax, rbx
.skip_xor:
 shl rax, 1
 add rax, 1
 dec rcx
 jnz .loop
ret
```

Bytes:

```text
0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x48, 0xbb, 0x15, 0x7c, 0x4a, 0x7f,
0xb9, 0x79, 0x37, 0x9e, 0x48, 0x31, 0xd2, 0x75, 0x03, 0x48, 0x01, 0xd8, 0x48, 0x31, 0xd2, 0x75,
0x03, 0x48, 0x31, 0xd8, 0x48, 0xd1, 0xe0, 0x48, 0x83, 0xc0, 0x01, 0x48, 0xff, 0xc9, 0x75, 0xe4,
0xc3
```

---

#### `branch_pred32`

32-bit predictable branch variant.

**Input:** `ECX = iters`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000):** `0xaad6afab`

Assembly:

```nasm
BITS 32
mov eax, 0x9ABCDEF0
mov ebx, 0x7F4A7C15
.loop:
 xor edx, edx
 jnz .skip_add
 add eax, ebx
.skip_add:
 xor edx, edx
 jnz .skip_xor
 xor eax, ebx
.skip_xor:
 shl eax, 1
 add eax, 1
 dec ecx
 jnz .loop
ret
```

Bytes:

```text
0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0xbb, 0x15, 0x7c, 0x4a, 0x7f, 0x31, 0xd2, 0x75, 0x02, 0x01, 0xd8,
0x31, 0xd2, 0x75, 0x02, 0x31, 0xd8, 0xd1, 0xe0, 0x83, 0xc0, 0x01, 0x49, 0x75, 0xec, 0xc3
```

---

#### `branch_unpred64`

Branch-heavy loop with pseudo-random branch direction (xorshift64).

**Input:** `RCX = iters`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000):** `0xdf14f128b035d3f0`

Assembly:

```nasm
BITS 64
mov rax, 0x123456789ABCDEF0 ; RNG state
mov rbx, 0x0F0E0D0C0B0A0908 ; accumulator / checksum
.loop:
 mov rdx, rax
 shl rdx, 13
 xor rax, rdx
 mov rdx, rax
 shr rdx, 7
 xor rax, rdx
 mov rdx, rax
 shl rdx, 17
 xor rax, rdx

 mov rdx, rax
 and rdx, 1
 jz .even
 add rbx, rax
 jmp .after
.even:
 xor rbx, rax
 jmp .after
.after:
 dec rcx
 jnz .loop
 mov rax, rbx
ret
```

Bytes:

```text
0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x48, 0xbb, 0x08, 0x09, 0x0a, 0x0b,
0x0c, 0x0d, 0x0e, 0x0f, 0x48, 0x89, 0xc2, 0x48, 0xc1, 0xe2, 0x0d, 0x48, 0x31, 0xd0, 0x48, 0x89,
0xc2, 0x48, 0xc1, 0xea, 0x07, 0x48, 0x31, 0xd0, 0x48, 0x89, 0xc2, 0x48, 0xc1, 0xe2, 0x11, 0x48,
0x31, 0xd0, 0x48, 0x89, 0xc2, 0x48, 0x83, 0xe2, 0x01, 0x74, 0x05, 0x48, 0x01, 0xc3, 0xeb, 0x05,
0x48, 0x31, 0xc3, 0xeb, 0x00, 0x48, 0xff, 0xc9, 0x75, 0xca, 0x48, 0x89, 0xd8, 0xc3
```

---

#### `branch_unpred32`

32-bit xorshift branch variant.

**Input:** `ECX = iters`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000):** `0xb1fdf341`

Assembly:

```nasm
BITS 32
mov eax, 0x9ABCDEF0 ; RNG state
mov ebx, 0x0B0A0908 ; accumulator
.loop:
 mov edx, eax
 shl edx, 13
 xor eax, edx
 mov edx, eax
 shr edx, 7
 xor eax, edx
 mov edx, eax
 shl edx, 17
 xor eax, edx

 mov edx, eax
 and edx, 1
 jz .even
 add ebx, eax
 jmp .after
.even:
 xor ebx, eax
 jmp .after
.after:
 dec ecx
 jnz .loop
 mov eax, ebx
ret
```

Bytes:

```text
0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0xbb, 0x08, 0x09, 0x0a, 0x0b, 0x89, 0xc2, 0xc1, 0xe2, 0x0d, 0x31,
0xd0, 0x89, 0xc2, 0xc1, 0xea, 0x07, 0x31, 0xd0, 0x89, 0xc2, 0xc1, 0xe2, 0x11, 0x31, 0xd0, 0x89,
0xc2, 0x83, 0xe2, 0x01, 0x74, 0x04, 0x01, 0xc3, 0xeb, 0x04, 0x31, 0xc3, 0xeb, 0x00, 0x49, 0x75,
0xd9, 0x89, 0xd8, 0xc3
```

---

#### `mem_seq64`

Memory load/store loop over a 4096-byte ring buffer (sequential stride of 8 bytes).

**Input:** `RCX = iters`, `RDI = buffer_base (4096 bytes, 8-byte aligned)`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000, buffer initialized to zero):** `0xb744761306865560`

Assembly:

```nasm
BITS 64
mov rax, 0x0123456789ABCDEF
xor rsi, rsi
.loop:
 mov rdx, [rdi + rsi]
 add rax, rdx
 xor rdx, rax
 mov [rdi + rsi], rdx
 add rsi, 8
 and rsi, 0x0FFF
 dec rcx
 jnz .loop
ret
```

Bytes:

```text
0x48, 0xb8, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x48, 0x31, 0xf6, 0x48, 0x8b, 0x14,
0x37, 0x48, 0x01, 0xd0, 0x48, 0x31, 0xc2, 0x48, 0x89, 0x14, 0x37, 0x48, 0x83, 0xc6, 0x08, 0x48,
0x81, 0xe6, 0xff, 0x0f, 0x00, 0x00, 0x48, 0xff, 0xc9, 0x75, 0xe2, 0xc3
```

---

#### `mem_seq32`

32-bit sequential memory loop over a 4096-byte ring buffer (stride of 4 bytes).

**Input:** `ECX = iters`, `EDI = buffer_base (4096 bytes, 4-byte aligned)`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000, buffer initialized to zero):** `0x0cc50aff`

Assembly:

```nasm
BITS 32
mov eax, 0x89ABCDEF
xor esi, esi
.loop:
 mov edx, [edi + esi]
 add eax, edx
 xor edx, eax
 mov [edi + esi], edx
 add esi, 4
 and esi, 0x0FFF
 dec ecx
 jnz .loop
ret
```

Bytes:

```text
0xb8, 0xef, 0xcd, 0xab, 0x89, 0x31, 0xf6, 0x8b, 0x14, 0x37, 0x01, 0xd0, 0x31, 0xc2, 0x89, 0x14,
0x37, 0x83, 0xc6, 0x04, 0x81, 0xe6, 0xff, 0x0f, 0x00, 0x00, 0x49, 0x75, 0xea, 0xc3
```

---

#### `mem_stride64`

Strided memory load/store loop (64-byte stride) over a 4096-byte ring buffer.

**Input:** `RCX = iters`, `RDI = buffer_base (4096 bytes, 8-byte aligned)`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000, buffer initialized to zero):** `0xd8d5ee9d0da7ebb4`

Assembly:

```nasm
BITS 64
mov rax, 0x0123456789ABCDEF
xor rsi, rsi
.loop:
 mov rdx, [rdi + rsi]
 add rax, rdx
 xor rdx, rax
 mov [rdi + rsi], rdx
 add rsi, 64
 and rsi, 0x0FFF
 dec rcx
 jnz .loop
ret
```

Bytes:

```text
0x48, 0xb8, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x48, 0x31, 0xf6, 0x48, 0x8b, 0x14,
0x37, 0x48, 0x01, 0xd0, 0x48, 0x31, 0xc2, 0x48, 0x89, 0x14, 0x37, 0x48, 0x83, 0xc6, 0x40, 0x48,
0x81, 0xe6, 0xff, 0x0f, 0x00, 0x00, 0x48, 0xff, 0xc9, 0x75, 0xe2, 0xc3
```

---

#### `mem_stride32`

32-bit strided memory loop (64-byte stride) over a 4096-byte ring buffer.

**Input:** `ECX = iters`, `EDI = buffer_base (4096 bytes, 4-byte aligned)`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000, buffer initialized to zero):** `0x0da7ebb4`

Assembly:

```nasm
BITS 32
mov eax, 0x89ABCDEF
xor esi, esi
.loop:
 mov edx, [edi + esi]
 add eax, edx
 xor edx, eax
 mov [edi + esi], edx
 add esi, 64
 and esi, 0x0FFF
 dec ecx
 jnz .loop
ret
```

Bytes:

```text
0xb8, 0xef, 0xcd, 0xab, 0x89, 0x31, 0xf6, 0x8b, 0x14, 0x37, 0x01, 0xd0, 0x31, 0xc2, 0x89, 0x14,
0x37, 0x83, 0xc6, 0x40, 0x81, 0xe6, 0xff, 0x0f, 0x00, 0x00, 0x49, 0x75, 0xea, 0xc3
```

---

#### `call_ret64`

Call/ret + stack pressure (push/pop) in the callee.

**Input:** `RCX = iters`
**Output:** `RAX = checksum`
**Expected checksum (iters=10_000):** `0x0209be0771df5500`

Assembly:

```nasm
BITS 64
mov rax, 0xFEEDFACECAFEBEEF
mov rbx, 0x9E3779B97F4A7C15
.loop:
 call callee
 dec rcx
 jnz .loop
ret

callee:
 push rbx
 push rsi
 add rax, rbx
 xor rax, 0x1F123BB5
 shl rax, 3
 pop rsi
 pop rbx
ret
```

Bytes:

```text
0x48, 0xb8, 0xef, 0xbe, 0xfe, 0xca, 0xce, 0xfa, 0xed, 0xfe, 0x48, 0xbb, 0x15, 0x7c, 0x4a, 0x7f,
0xb9, 0x79, 0x37, 0x9e, 0xe8, 0x06, 0x00, 0x00, 0x00, 0x48, 0xff, 0xc9, 0x75, 0xf6, 0xc3, 0x53,
0x56, 0x48, 0x01, 0xd8, 0x48, 0x35, 0xb5, 0x3b, 0x12, 0x1f, 0x48, 0xc1, 0xe0, 0x03, 0x5e, 0x5b,
0xc3
```

---

#### `call_ret32`

32-bit call/ret + stack pressure.

**Input:** `ECX = iters`
**Output:** `EAX = checksum`
**Expected checksum (iters=10_000):** `0x71df5500`

Assembly:

```nasm
BITS 32
mov eax, 0xCAFEBEEF
mov ebx, 0x7F4A7C15
.loop:
 call callee
 dec ecx
 jnz .loop
ret

callee:
 push ebx
 push esi
 add eax, ebx
 xor eax, 0x1F123BB5
 shl eax, 3
 pop esi
 pop ebx
ret
```

Bytes:

```text
0xb8, 0xef, 0xbe, 0xfe, 0xca, 0xbb, 0x15, 0x7c, 0x4a, 0x7f, 0xe8, 0x04, 0x00, 0x00, 0x00, 0x49,
0x75, 0xf8, 0xc3, 0x53, 0x56, 0x01, 0xd8, 0x35, 0xb5, 0x3b, 0x12, 0x1f, 0xc1, 0xe0, 0x03, 0x5e,
0x5b, 0xc3
```

---

### Execution harness requirements (emulator core)

Add a CPU-core-level helper that can:

1. Load a byte slice into a scratch executable region (or use an existing “execute bytes” harness).
2. Configure the CPU state for x86-32 vs x86-64 payload execution.
3. Run in a chosen tier: interpreter, baseline JIT, optimizing JIT.
4. Return:
 - checksum (`EAX`/`RAX`)
 - retired instruction count delta 
 - wall-clock time delta

Measurement must compute throughput from **retired guest instructions**, not from static “instructions in the payload”.

Suggested internal structure:

```rust
pub enum GuestCpuMode { Interpreter, JitBaseline, JitOpt }

pub struct GuestCpuRunResult {
 pub checksum: u64,
 pub retired_instructions: u64,
 pub seconds: f64,
}

pub fn run_guest_payload(
 mode: GuestCpuMode,
 payload: &[u8],
 iters: u32,
 scratch_base: u64, // for mem_* benches
 scratch_len: u32,
) -> Result<GuestCpuRunResult>;
```

#### Time budgeting (`seconds`)

Recommended approach for stable results:

1. Run `warmup_runs` (default 3) and discard.
2. For measured runs:
 - repeat deterministic invocations (`iters_per_run = 10_000`) until the elapsed time budget is reached.
 - collect per-run MIPS to compute variance.

This avoids needing to forcibly interrupt guest execution mid-loop.

---

### Perf export integration

Include results in `window.aero.perf.export()`:

```ts
type AeroPerfExport = {
 // ... other perf fields ...
 benchmarks: {
 guest_cpu: {
 iters_per_run: number;
 warmup_runs: number;
 measured_runs: number;
 results: Array<{
 variant: GuestCpuBenchVariant;
 mode: GuestCpuMode;
 mips_mean: number;
 mips_stddev: number;
 mips_min: number;
 mips_max: number;
 expected_checksum: string;
 observed_checksum: string;
 }>;
 };
 };
};
```

The export should be structured to allow baseline comparison in CI .

---

### Playwright scenario: `guest_cpu`

Add a benchmark runner scenario `guest_cpu` (Playwright) that:

 - runs headless
 - calls `window.aero.bench.runGuestCpuBench(...)`
 - writes the exported perf blob to the benchmark artifact directory
 - compares against baseline and fails on large regressions

#### PR (fast) subset

Run a small subset with short budgets (example):

 - `alu64` in `interpreter` for `0.25s`
 - `branch_pred64` in `interpreter` for `0.25s`

#### Nightly (full) suite

Run all variants and all available modes with longer budgets (example):

 - `seconds = 1.0` per variant/mode
 - `warmup_runs = 3`, `measured_runs >= 10`

#### Baseline comparison

Compare `mips_mean` per `(variant, mode)`:

 - PR threshold: e.g. fail if regression > 10% (configurable)
 - nightly threshold: tighter thresholds + trend tracking

---

### Correctness guardrails

The guest CPU bench **must fail on checksum mismatches**:

 - If `observed_checksum != expected_checksum`, throw an error and include:
 - variant + mode
 - expected vs observed
 - a short hint (likely an emulator correctness regression)

This prevents “fast but wrong” changes (especially in JIT) from silently improving benchmark numbers.

---

---

## The compiled tier's correctness bar, and what it cost

The compiled tier only means anything if the guest cannot tell it apart from the
interpreter. The bar used here is architectural state — RIP, RSP, the GPRs, CR3 —
identical to Tier-0 at 100K, 600K and 1M retired instructions. Past a few million the
two legitimately drift, because a compiled block runs to its end without the
per-instruction interrupt checks Tier-0 makes, so delivery lands a few instructions
apart; at 2M the states sit in the same function with the same RSP and CR3 and RIPs a
few instructions apart, which is that drift and not a wrong answer.

Raising coverage broke that bar, and the interesting part is *why*: the tier had been
compiling 628 blocks, and every bug below was in code it had simply never translated.
Coverage is not only a performance lever. It is how much of the translator is under
test.

### Five real bugs, and the shape they share

- **A SIB index of `r12` was dropped.** `index == 100b` means "no index" only without
  REX.X; with it the field names `r12`. The decoder masked the REX bit off before the
  comparison, so `[r13+r12*2]` decoded as `[r13]`.
- **`adc`/`sbb` carried the wrong flag out.** CF was computed as a plain add or
  subtract, which is right except when the carry *in* is what pushes the result over
  the edge — exactly where `cmp; sbb eax, eax` lands, the standard way to turn a
  comparison into 0/-1.
- **Rotates never wrote CF or OF at all**, on the reasoning that nothing read them.
- **`imul` never wrote CF or OF** either, for the same reason.
- **Dead-flag elimination deleted a carry that `sbb` consumes.** The IR does not say
  `sbb` reads CF — the code generator reads it when lowering — so a pass that
  believed the IR found no reader for the CF a `cmp` produced and removed it.

Four of the five are the same mistake in different places: **a declared flag set that
does not match what the code actually writes.** That is harmless while nothing reads
it and fatal the moment a liveness pass is added, because such a pass reads those sets
as "what this instruction overwrites". Rotates declared PF, ZF and SF they never
wrote, so they shadowed the writes that really produced them. `imul` declared all six.

The rule this leaves: **an instruction's declared flag set must be exactly what it
writes, and where the architecture leaves a flag undefined, Tier-0's choice is the
specification** — a guest can set a flag in one tier and read it in another, and it
has to look like one machine. Tier-0 writes only CF and OF for `imul` and rotates, so
Tier-1 does too, and so does the IR interpreter.

### What found them

Three tools, in increasing order of what they cost to build:

- **An exhaustive per-operation check.** Every `BinOp`, every width, the operand
  values where the rules differ (zero, one, the maximum, the sign bit, an equal pair),
  with carry in both ways. `tier1_random_ir_equiv`'s generator had covered eight of
  the fourteen operations, which is why `adc` and `sbb` went untested for as long as
  they did. Anything the IR can express belongs in that list.
- **Real guest blocks against Tier-0.** The generator builds IR directly, so it never
  exercises the decoder or the translator. `a_real_guest_block_matches_tier0` takes a
  byte sequence captured out of the running guest with `AERO_CT_TRACE_ADMIT=1`, grows
  it one instruction at a time, and compares against Tier-0 with each engine on its
  own copy of memory. A failure names the instruction.
- **A per-block verifier in the live machine.** `AERO_CT_VERIFY=1` runs Tier-0 over a
  block's instructions, then the compiled block from the same registers, and reports
  any difference. Tier-0 goes first and its result is the one kept, so the machine
  follows the interpreter while being checked; a correct block then writes the same
  bytes Tier-0 just wrote and memory is left as Tier-0 had it.

Bisecting a whole-machine divergence down to one block costs a dozen multi-minute runs
and finds one bug. The verifier finds every mis-translated block in a single run and
names it. Build the oracle before the third bisection, not after.

**Its blind spot is worth stating**, because it explains a divergence it could not
see: it only replays blocks that do not both load and store, since a read-modify-write
applied twice is not the same as applied once. Detecting that by matching the load's
address value against the store's is not enough — `sub dword ptr [rip+0xfac1a], 1`
builds the address twice and gets two value ids, so the check passed and the verifier
reported its own double-subtraction as a mismatch.

### The chain has to retire exactly what the slice allowed

Three separate divergences turned out to be one accounting mistake, and none of them
were mistranslations.

`max_blocks` for a chain was derived from the *entry* block's length, so a chain whose
later blocks were longer ran past the slice's budget — 262 instructions over a 100,000
instruction slice. The machine steps devices between slices, so overrunning moves
interrupt delivery later than the interpreter would put it. The fix is to stop on
actual retirement, and to decline a block that will not fit in what is left rather
than notice afterwards.

Separately, a block ending at the decoder's instruction limit rather than at a branch
returns the same exit sentinel as a block that bailed part-way, and the tier treated
both as bails: their instructions were retired by the guest and never counted. The two
are distinguishable — a real bail leaves an exit flag set, a terminator exit does not —
and a limit-terminated block has run everything and can be chained straight through.

The tell was that shortening blocks to two or four instructions, where nearly every
block ends at the limit, made the tier disagree while eight or more agreed. **A knob
whose failures are not monotonic in the knob is measuring timing, not semantics** —
and inconsistency across budgets says the same thing. Chain length looked guilty for a
while on exactly that evidence.

With retirement exact, the tier matches the interpreter through 2M instructions, which
it had never done.

### Coverage, and the price of correctness

The slow memory helpers translate a guest address and then read or write the engine's
linear memory at the result. That is right for RAM and wrong for everything else: a
device register has no backing there, so the access reads the wrong bytes and the
device never sees it. Blocks that can reach those helpers are now refused, which is
what restores agreement with the interpreter at a million instructions.

It was expensive. `needs.slow_mem` is set when a module merely *imports* the helpers,
which the inline TLB does as a fallback, so refusing those blocks took out 12,724
against 8,192 compiled, and coverage fell from 59% to 21.6%.

That refusal is gone. A slow-path access that translates to something other than RAM
is now handed to the embedder's bus — the same bus Tier-0 uses — at its natural width,
through a `GuestAccessor` the tier installs alongside its translator. RAM keeps the
direct path, which is also where guest writes get their code-version bookkeeping, so
self-modifying code stays coherent. Refusals are at zero and coverage is 74%.

`inline_tlb_cross_page_fastpath` deserves a note because it was on for a while. It
keeps an access the code generator cannot prove stays inside one page on the inline
path, by splitting it and translating both pages, which was the only way to avoid the
slow-memory refusal. With that refusal gone it buys nothing and is not free: the split
emits a second translate-and-access path for every wide load and store, and that code
has to be generated, validated and compiled. Turning it off cut admission from 612us
to 249us a block and compiled code from 108ns to 78ns per guest instruction at
identical coverage.

### What admission actually costs

Admission is the tier's second-largest expense and it is not what it looks like.
Profiling a real run shows `winch_codegen`, `cranelift_assembler_x64` and
`wasmparser::visit_operator` — genuine compilation and validation of the wasm the tier
emits, not per-module bookkeeping. So the lever is emitting less of it, which is what
the cross-page finding above is really about.

Blocks are compiled in batches of sixteen into one module rather than one at a time,
which took admission from 909us to 598us a block with the split paths still on. Batch
size beyond sixteen makes no difference, confirming the cost is per-block codegen.
The block cache holds 32,768 entries, which is where `at_cap` reaches zero — this
workload's working set is 32,014 blocks. Wasmtime allows 10,000 instances per store by
default, and since every module is an instance that ceiling, not policy, was what
capped the cache at 8,192.

### What the compiled tier is not

Profiling a full run says the thing worth knowing before optimising anything else:
**19% of the time is spent in JIT-generated code, and that code runs 74% of the guest
instructions.** The other 81% is the emulator around it — the interpreter for what is
left, the MMU, the device model, the memory bus, the chain driver. Making compiled
code twice as fast would buy under 10%.

So the lever is coverage, and after that it is the emulator itself. Going from 500
MIPS-class performance to what this is would need the whole machine to cost on the
order of ten host instructions per guest instruction; it costs 630. That is an
architectural gap, not a tuning one.

### Segment prefixes were the largest single bail

`AERO_CT_BAILS=1` prints the bytes that ended a block in a bailout, most frequent
first, and the answer was not subtle: `65 48 8b 04 25 ...` — `mov rax, gs:[disp32]` —
and its siblings, about 500 blocks. The Tier-1 decoder had no segment-prefix handling
at all, and Windows reaches its per-CPU and per-thread blocks through `gs:`, so most
of the kernel was unreachable.

Supporting it is `Address::segment`, a `GuestReg::SegBase` the code generator lowers
to a load from `CpuState`, and the base added after the address-size mask — `gs:[0x88]`
in long mode reaches gs_base+0x88, not a wrapped sum. Only FS and GS need it; the rest
are zero-based in long mode, so their prefixes can be skipped outright.

It also needed a fix nothing had noticed: **`sync_cpu_to_wasm` never copied the
segment bases**, because until now nothing read them. Compiled code saw zero. The
lesson is the ABI's, not this feature's — a field the code generator can read must be
in the sync, and there is nothing that checks the two agree.

The bail list after that is a genuine long tail: `syscall`, `lock`-prefixed atomics,
one-operand `imul`, `ret imm16`, control-register moves, string operations. The top
ten together are about 4% of remaining bails, so no single opcode is worth much now.

### Where the time goes, honestly

Two windows, because a JIT has two regimes. Compilation is paid once; coverage pays
for as long as the machine runs.

| | interpreter | compiled tier |
|---|---|---|
| warm-up (20M-80M) | 1255.0 | **800.1** |
| steady state (80M-200M) | 1276.0 | **182.1** |
| whole session (20M-200M) | 1253.1 | **365.2** |

Host instructions per guest instruction, lower is better: about **2x**. Coverage is
75% and admission is ~258us a block.

Getting the warm-up number there was a threshold decision, not a code change. The
hotness threshold's real job is how *soon* the tier reaches its plateau, not how much
it compiles in the end: at 12 the two windows measured 941 and 638, at 4 they measure
630 and 686. Warm-up improves monotonically as it falls while the steady-state
differences sit inside the run-to-run spread, so 4 is the default.

Both numbers matter. An earlier configuration reached the same steady state while
being 31% *slower* than the interpreter during warm-up, because it spent twenty
seconds compiling; quoting only the steady-state figure would have hidden that
completely.

### One fast path per memory access

The first structural piece of a TCG-class core, and the largest single win found:
**whole session 674.5 -> 631.4 host instructions per guest instruction, steady state
714 -> 585.**

`emitted_code_size` counts what the code generator produces. A single
`mov eax,[rsi]` cost about 128 wasm operations, against five to ten host instructions
for the equivalent soft-MMU hit in a mature emulator, and a real block averaged 204
operations per guest instruction. Breaking one load down: 51 `local.get`, 28
`i64.const`, 24 `local.set` — 58% of it moving values around — with five `if`s and
four calls.

The reason was structural. Each access inlined a path per case: a cross-page check
wrapping an IS_RAM check wrapping the TLB probe and an MMIO exit, with the same-page
sequence emitted inside two of those branches. Four paths at every access site, one of
them taken.

A mature emulator emits one fast path and branches everything else to a shared
out-of-line stub. The slow helper here already handles misses, page crossings and
device memory — it reaches non-RAM through the embedder's bus — so the inline path
reduces to a single predicate: *the TLB knows this page, it is backed by RAM, and the
access stays inside it*. Anything else is one call.

Worth noting what the measurement said versus what was expected. Collapsing the
nested paths cut emitted code by only 8-10%, because the bulk was never the branches —
it is the TLB probe, which computed the entry address twice, once for the tag load and
again for the data load. Keeping it in a local took a load from 112 operations to 90
and a whole real block from 204 to 189. Yet the end-to-end win was 6.4%, far more than
the code-size change suggests: smaller, flatter code is worth more to a baseline
compiler than the operation count implies.

**The same change is a win for loads and a loss for stores**, which no amount of
reasoning would have predicted. Whole session, host instructions per guest instruction:
nested paths 674.7, loads on the single fast path 632.5, loads *and* stores 763.3. The
load half is on by default and the store half is not (`AERO_JIT_STORE_FP=1` enables
it). The likely reason is that a non-RAM store used to leave the block and now calls
the helper, and that every storing block must import that helper — but it is a guess,
and the code carries it as one.

One thing the store attempt taught, which was not a guess: an entry caches only the
permission actually exercised, so a page first touched by a read carries no write bit.
Folding the permission into the fast-path predicate means a mismatch merely *fails*
and never refills the entry, so every later store to that page goes to the slow helper
for as long as it stays cached. That cost 50% before it was found, and it is why the
permission re-check has to live inside the probe.

It costs warm-up — 594.7 to 724.1 — because the helper can no longer be elided, so
more blocks import it. Over a whole session it is well ahead, and steady state is the
best measured.

A load is now 90 operations, from 112. What is left in it is mostly the probe's tag
compare and the address arithmetic either side, and the ~27 operations of per-block
bookkeeping that any block pays.

### The ceiling of this design, added up

Every lever below was measured, not estimated. Assuming each lands in full and they
compose — which is optimistic, since they overlap — this is where the current design
ends up:

| | host instructions per guest instruction |
|---|---|
| interpreter | 1277 |
| today | **631** |
| in-wasm dispatch (`call_indirect`; 29.5ns -> 7.0ns a link) | -5% |
| Cranelift tier-up over a `SharedMemory` (9% steady) | -9% |
| hot-edge traces (penalties larger than the gain) | -3% |
| the remaining missing opcodes (top ten are 4% of bails) | -2% |
| best case | **552** |

500 MIPS needs roughly 10-18. **The entire remaining known headroom in this design is
about 17%, against a gap of 40-60x.** That is the number worth carrying: the distance
is not a backlog of optimisations, it is the design.

For scale, a mature TCG-class emulator runs at roughly 45-90 host instructions per
guest instruction. Reaching that would mean the three things this does not do — direct
block chaining with patched jumps rather than a host round-trip per block, register
allocation into host registers rather than wasm locals through a baseline compiler,
and a soft-MMU lookup of a handful of instructions. Wasm makes the first two
structurally harder: there are no patchable jumps and no host registers to allocate
into. Closing the rest of the way needs hardware virtualisation, which the browser
rules out by construction.

### What the control-flow graph actually looks like

Before building profile-guided traces on top of the superblock machinery, it was
worth measuring whether the branches are predictable enough to be worth following.
`AERO_CT_EDGES=1` records every block-to-block transition and reports three things
over 80M guest instructions:

- **93.0%** of 14.7M transitions go to the block's most frequent successor. The graph
  is strongly biased, which is the precondition for tracing to be worth anything.
- Only **23.6%** of those hot edges go backwards, so "backward branches are taken" —
  the classic static prediction — accounts for a quarter of them. A static heuristic
  is not enough.
- **73.4%** of hot edges out of conditionally-ending blocks are the fall-through. So
  fusing blindly along the fall-through, which is what superblocks do today, already
  follows the hot edge about three times in four.

That last number is the one that settles it. Profile-guided selection would raise
edge accuracy from 73.4% to 93%, which for a three-edge trace is the difference
between staying on it 40% and 80% of the time — a real improvement. But the two costs
that made superblocks lose are consequences of block *size*, not of which edge was
chosen: admission goes from 253us to 385us a block, and a cache holding the same
*number* of bigger blocks covers fewer entry points, so coverage falls. Both survive
perfect edge selection.

What is left to win is the boundary saving: about 50ns times the 8.6M fewer block
executions, roughly 3% of the run. That is smaller than the penalties it would have to
overcome, so hot-edge tracing is not worth building on this workload — and the
measurement is here so that stays a decision rather than a hunch.

### Superblocks: built, correct, and slower

`AERO_CT_SUPERBLOCKS=1` keeps decoding past a conditional branch along its
fall-through and turns the branch into an `IrInst::SideExit`, so the guest registers
stay in wasm locals across it instead of being spilled and reloaded. It works, it
agrees with the interpreter through 2M instructions, and it is **off by default,
because it is slower**.

The structure does what it was meant to. Block executions fall from 14.7M to 6.1M
over 80M guest instructions and instructions retired per execution goes from 4.2 to
8.4 — 2.4x fewer boundaries. But:

| | basic blocks | superblocks |
|---|---|---|
| warm-up (20M-80M) | **591.6** | 990.8 |
| steady state (80M-200M) | **711.9** | 785.2 |
| coverage of 80M | 61.9M | 51.4M |
| admission | 253us/block | 385us/block |

Two things eat it. Admission costs half again as much per block, and the cache holds
the same *number* of blocks, so a cache full of superblocks covers fewer distinct
entry points and chains break more often — coverage falls.

The deeper reason is that fall-through is the wrong direction to fuse blindly. A loop's
back-edge is taken on nearly every iteration, so the side exit fires and the block is
merely bigger. Making this pay needs the trace to follow the *hot* edge, which means
profiling branch directions first — the blocks are already there to hang that on.

Two real bugs came out of it, and one of them was not superblocks' fault:

- The register-liveness rule "load what you read, spill what you write" is only sound
  if the whole block runs. A register written near the end has no load, so its local
  is zero until that write — and an exit *before* it spills the zero over the guest's
  value. Superblocks made it constant; a mid-block bail-out could always hit it. Only
  registers whose first write follows the earliest possible exit need the load.
- A block that bailed part-way used to be discarded whole, on the grounds that its
  retirement was unknowable. Now that a block reports what it got through, discarding
  it is wrong: the prefix's register writes are already committed and only the count
  would be thrown away.

The retirement reporting those rest on is worth its own note. A block used to retire a
fixed count — its own length — and the runtime could assume it. One with side exits
cannot, so it writes what actually ran into the JIT context at whichever exit it
leaves through. That reporting costs about 2.5%, which is why the count is only
maintained from the first instruction that could exit; before that it is unobservable.

### The block boundary, measured

Running the *same* instruction mix in blocks of 1, 2, 4 ... 64 separates what an
instruction costs to execute from what a block boundary costs: the slope is one, the
intercept is the other. `block_length_cost` does exactly that, and the answer decided
what to work on next.

The first run said 270ns per boundary against 7-18ns per guest instruction — at the
4.2-instruction blocks this workload really has, **78-91% of compiled-code cost was
the boundary, not the code**.

Most of it was not fundamental. Entering a chain wrote `CpuState` field by field
through `Memory::write`: twenty-one bounds-checked calls, each re-borrowing the store.
Leaving one allocated a `Vec` the size of `CpuState` and copied it back. The linear
memory is created with `min == max` and a matching `Config::memory_reservation`, so
its base never moves — caching it and copying directly took the boundary from **270ns
to 50ns**, and the machine from 635/690 to 610/664 host instructions per guest
instruction.

A caution from doing it: `execute` and `execute_chain` are not the same path.
`execute` snapshots guest RAM for rollback, so the first version of this benchmark
measured a half-megabyte `memcpy` and reported 14us a block. Measure the path the tier
actually takes.

What is left at a boundary is the host-to-wasm call and the register spill and reload
around it, and that is what superblock formation would remove: at 4.2 instructions a
block the boundary is still about half the cost, and the same benchmark shows the
per-instruction cost falling from ~72ns at four instructions to ~15ns at sixteen.

### Lazy flags: measured, and it did not pay

x86 defines six flags on most ALU instructions and real code reads one or two, so
deferring the computation and rebuilding a flag only where it is read is the classic
JIT win. It was the first step of the execution-core rewrite. **It is worth about 1%
here**, and the reason is worth writing down so nobody spends the week again.

The mechanism is in place and on by default: an ALU operation records what produced
the flags — operation, width, operands, result — in wasm locals instead of computing
six bits; a read rebuilds just the flag it wants; what is still owed at the block's
exit goes to `CpuState.lazy_flags`, which Tier-0 already materialises on demand. A
block that only reads flags takes the one it needs and passes the record on
untouched, so `je`/`jne` cost almost nothing: ZF is "the result was zero" whatever
produced it. `AERO_JIT_NO_LAZY_FLAGS=1` forces the eager path.

Measured, host instructions per guest instruction:

| | eager | deferred |
|---|---|---|
| warm-up (20M-80M) | 644.6 | **635.5** |
| steady state (80M-200M) | 695.6 | **689.9** |

A second design, where pure readers pass the record through instead of settling it,
measured 636.1 and 692.4 — the same to within noise.

**Why the estimate was wrong.** `eliminate_dead_flags` already removes flag writes
that a later write shadows inside a block, which is most of what laziness would have
saved. What is left is a block computing flags its *successor* reads, and moving that
work into a record does not remove it — it relocates it, and the consumer pays to
rebuild what the producer no longer computed.

**What that says about the rest.** Flags are not where compiled code spends its time,
so the remaining 22% must be the memory path, the register spill and reload at every
block boundary, and dispatch. Blocks average 4.2 guest instructions, so those
boundary costs are divided by almost nothing — which points at superblock formation,
not at better arithmetic, as the next thing to try.

Four bugs were found on the way, all of them in code that had never run:

- Wasm `select` yields its **first** pushed operand when the condition is true. Three
  guards were written as `[value, 0, is_logic]`, which reads exactly backwards.
- `cmp` and `test` compute their result into the shared scratch local, which the
  inline TLB reuses long before the block ends, so a record pointing at it captured
  whatever came later.
- The record has to live in locals, not in what the code generator last saw: a block
  can leave through a bail-out in the middle, and the shared epilogue would otherwise
  describe arithmetic that path never ran.
- `sync_cpu_to_wasm` wrote a materialised RFLAGS but left the record region alone, so
  a block reading flags on arrival settled a **stale** record over correct flags.

And one in code that had been dormant since it was written: `LazyFlags::af()` and
`of()` folded the carry into the operand before applying the XOR identity, which is
wrong for `adc`/`sbb` whenever the low nibble of `op2 + carry` wraps. `adc al, 0xff`
with AL=0 and CF=1 has a real carry out of bit 3 and the folded form reported none.

### What is left, with numbers

Four independent measurements agree that compiled-code quality is not where the
remaining time is.

**The profile.** 19% of samples are in JIT-generated code, which runs 74% of the guest
instructions. Everything else is the emulator around it.

**Winch against Cranelift.** The code generator can be swapped with
`AERO_JIT_COMPILER=cranelift`. Cranelift's output is better and it costs much more to
produce, and which wins depends entirely on the window:

| window | Winch | Cranelift |
|---|---|---|
| warm-up (20M-80M) | **630.6** | 714.6 |
| steady state (80M-200M) | 685.2 | **624.6** |
| whole session (20M-200M) | 667.3 | **653.9** |

Reproducible to better than a percent on repeat runs. Over a whole session the
difference is 2%: Cranelift's 9% better steady state is very nearly eaten by its
compile cost. Winch stays the default because it is the better balance and the
simpler arrangement.

The way to have both is tier-up — Winch on admission, Cranelift for blocks that go on
to prove hot. Worth about 9% of steady state, and it is not free to build: a Wasmtime
`Engine` fixes its strategy, two engines cannot share one linear memory, so it needs
the guest's memory to become a `SharedMemory` (which also changes what the code
generator emits for code-version bumps). That is the trade, stated so it can be
decided rather than guessed at.

**Coverage is at its plateau.** `at_cap` is zero — every block that gets hot is
compiled — and the bail list is a long tail where the top ten opcodes together are
about 4% of what remains.

**The arithmetic.** Desktop-class throughput (500 MIPS) means roughly ten host
instructions per guest instruction. This costs 630. Closing that is not a matter of
better code generation for the 19%; it would take a different design for the
interpreter, the MMU and the device model.

Two measurement notes that cost real time here:

- **Wall-clock MIPS is unusable on a loaded machine.** At load average 22 on 64 cores
  the interpreter baseline moved 10% between runs. Host instructions retired per guest
  instruction, differenced across two `--max-insts` budgets to cancel start-up, is
  immune to that and is the number to quote.
- **A diverged run is not a measurement.** An intermediate state here read 13.16 MIPS
  against 7.73 and it was not a speedup: the run had diverged and was executing
  different, cheaper guest code. Coverage was *lower* in the faster run, which is the
  tell — check the throughput claim against Amdahl before believing it.


## What blocks coverage, and what admitting a block costs

Two thirds of the compiled tier's remaining cost is not in compiled code at all. At
72% coverage the interpreter still runs 28% of guest instructions at ~1275 host
instructions each, which is ~360 of the ~550 the whole thing averages. Coverage is
the lever; what follows is what actually holds it down.

### Ranking the missing opcodes by how often the guest arrives at them

`AERO_CT_BAILS=1` used to report which instruction stopped translation, counted once
per *compile attempt* — which flatters rare instructions in cold code and hides the
ones the guest hits constantly. `AERO_CT_WALL` counts the other thing: every time
execution wanted a block Tier-1 had refused, over seam offers and chain steps alike.

The two orderings are not remotely the same. Ranked by wanted executions over a
200M-instruction desktop session:

| instruction | wanted executions |
| --- | --- |
| `rdtsc` | 5,083,926 |
| `call qword [rip+disp32]` | 1,315,336 |
| `lock` RMW (`bts`/`and`/`or`/`xadd`) | ~930,000 |
| `bt`/`bts`/`btr`/`btc` with an immediate | ~344,000 |
| `ret imm16` | ~200,000 |
| `movsx` / `movsxd` | ~265,000 |
| `bsf` / `bsr` | ~143,000 |
| `inc`/`dec r/m8` | ~48,000 |
| `movnti` | ~47,000 |

`rdtsc` dominates by four times, which is unsurprising once stated: Windows reads the
timestamp counter in its scheduler, its spinlocks and its performance counters.

### What was implemented, and what it left

Tier-1 gained the ones needing no host state: near indirect `call`/`jmp` through a
register or memory (`FF /2`, `FF /4`), `ret imm16`, `movsx`/`movsxd`, and the byte
forms of `inc`/`dec`. None needed new IR — indirect branches reuse
`IrTerminator::IndirectJump`, which `ret` already required, and sign extension is a
shift up and an arithmetic shift back down. Blocks Tier-1 refuses fell from 6,248 to
3,776 and wanted executions of them from 5.18M to 3.35M.

What is left needs a mechanism rather than an opcode. `rdtsc` reads the machine's
`TimeSource`, and `lock`-prefixed RMW needs the machine's own bus. The IR already has
`IrInst::CallHelper` for exactly this, but it is currently a *bailout* — both the
WASM code generator and the debug interpreter treat it as an exit to the interpreter.
Making it a real host call is the piece of work that unlocks the top of that table.

### Admission is per exported function, not per byte

Admitting a block cost ~222us against ~4.8us to generate the WASM, so essentially all
of it was `Module::new`. The obvious reading — Winch compiles about 2.6 MB/s, so emit
fewer bytes — is wrong, and following it wastes effort.

Compiling 64 functions of one operation each costs 93.2us per function. Compiling 64
functions of 128 operations each — twenty-three times the bytes — costs 102.8us.
Body size is nearly free. What is not free is being *nameable from outside*:

| of 64 functions, exported | us per function |
| --- | --- |
| 0 | 12.1 |
| 1 | 14.2 |
| 8 | 21.5 |
| 32 | 52.8 |
| 64 | 91.6 |

~80us per export, because a function reachable by name needs a host-callable
trampoline compiled for it. Putting the blocks in a table instead of exporting them
does not help — 82.6us — since a `funcref` is nameable too.

A function only ever reached by a direct `call` needs none. So a module now exports
one `dispatch(cpu_ptr, jit_ctx_ptr, index)` that `br_table`s to a direct call per
block. Measured, interleaved: warm-up 951.2 against 1055.5, whole session 681.9
against 714.9, steady state 547.3 against 544.7 — it wins wherever compilation is
being paid for and costs half a percent in the extra call where it is not.
`AERO_CT_NO_DISPATCH=1` goes back to one export per block.

Batching is not a substitute. Bigger batches do cut admission further (105.7us a
block at 64, 97.0us at 256, against 153.7us at 16) but lose end to end — 684.2,
690.0 and 711.1 host instructions per guest instruction over a whole session —
because a block waits for the rest of its batch before it can run.

### The cache holds the working set now

The desktop working set is ~51,600 blocks. `AERO_CT_MAX_BLOCKS` was 32,768, so the
cache filled and then refused 6.3M admissions, and coverage stopped where it stopped.
At 65,536 nothing is refused; at 131,072 nothing changes, because nothing more asks
to be compiled. Steady state 549.9 against 633.5 — 13%.

It is not free: holding 51,600 blocks means compiling 51,600 blocks, and warm-up
went from 723.8 to 951.1 host instructions per guest instruction. Break-even against
the smaller cache is a few hundred million guest instructions, which a desktop
session passes in under a minute.

### Sharing the load fast path between blocks

Every memory access inlined the whole TLB probe — ~218 emitted bytes, and with `ret`
making a load of nearly every block, that was most of what a block emitted. Emitting
it once per module as a plain function and calling it takes a load to 21 bytes and
admission from 246us to 222us.

End to end it is a wash: 688.6 against 691.2 over a whole session, 519.1 against
516.7 in steady state — it wins exactly where compilation is paid for. It is on for
the browser, where module compilation happens on the worker thread and a third of the
bytes is worth more than half a percent of steady state. `AERO_CT_INLINE_MEM=1`
inlines again.

The same change applied to *stores* is a clear loss — 763.3 against 632.5 — and is
off. See the store fast path note above.

### Measuring this at all

The guest's timestamp counter follows the host wall clock, so a loaded host makes the
guest execute *different code* — more idle, different timer paths. Within a window
the measurement is exact: the same configuration repeated three times gives 592.1,
592.1, 592.1. Across hours it drifts by several percent with no change to the binary.

**Only interleaved A/B comparisons are valid.** Run configuration A, then B, then A
again, and check that A repeats before believing the difference. Several conclusions
in this file's history were drawn from runs taken far apart and had to be redone.


## Compiling `RDTSC`, and what it was worth

`rdtsc` was the largest single entry in the coverage wall — 5,083,926 wanted
executions of blocks Tier-1 had to refuse, four times the next one — because Windows
reads the timestamp counter in its scheduler, its spinlocks and its performance
counters, and a block that cannot finish is a wall every chain reaching it stops at.

It is not a computation. The value comes from the machine's `TimeSource`, which
generated code has no way to reach, so it is a host call: a new `env.rdtsc(cpu_ptr)`
import, an `IrInst::ReadTsc`, and a `GuestClock` the tier installs on the backend.

Three things made it tractable:

**The clock is already next to the chain.** `TimeSource` is a field of `CpuCore`, and
`CompiledTier::run` is handed `&mut CpuCore`. The hook points at `&mut cpu.time`,
which is disjoint from the `&mut cpu.state` the backend borrows — so no second
pointer to the same data. The architectural copy in `CpuState::msr.tsc` is settled
after the chain returns rather than written during it, for the same reason.

**The poll-yield bookkeeping is opt-in.** Tier-0's assist tracks tight `rdtsc` polling
loops to request a scheduler yield, and that state lives in `AssistContext`, which a
compiled block cannot see. But it only does anything under `AERO_RDTSC_QUANTUM`,
which is off by default — so `CompiledTier` compiles `rdtsc` only when that is unset,
and otherwise leaves it to the interpreter where the bookkeeping works.
`BlockLimits::allow_rdtsc` carries the decision, so a caller with no host to call
gets the old behaviour.

**A timestamp cannot be verified by replay.** `AERO_CT_VERIFY` runs a block under
both engines and compares, which a second `rdtsc` would always fail. Blocks that read
the counter join the read-modify-write blocks as non-replayable, and are checked by
the machine-level oracles instead.

The result is the largest single change measured in this file:

| | interpreted `rdtsc` | compiled `rdtsc` |
| --- | --- | --- |
| coverage | 144.7M of 200M (72%) | **183.4M of 200M (92%)** |
| guest instructions interpreted | 55.3M | **16.6M** |
| wanted executions of refused blocks | 3.35M | **2.01M** |
| steady state | 550.5 | **301.9** |

45% off the steady state, and the reason is arithmetic rather than subtlety: the
interpreter costs ~1275 host instructions per guest instruction against the compiled
tier's ~200, so moving 39M guest instructions across is worth more than any amount of
tuning on the ones already there. `AERO_CT_NO_RDTSC=1` puts it back.

What is left on the wall is the `lock`-prefixed read-modify-write family — `lock bts`,
`lock and`, `lock or` — at about 930,000 wanted executions between them. Those need
the machine's own bus rather than the clock, and on a machine with AP CPUs running
they need to mean something, so they are a harder question than `rdtsc` was.


## `LOCK`, the bit tests, and where opcode work stops paying

With `rdtsc` compiled the wall's top was the `lock`-prefixed read-modify-writes —
`lock and`, `lock or`, `lock bts` — at about 930,000 wanted executions between them.

**`LOCK` needs no machinery here.** Aero runs its CPUs one at a time under a
cooperative BSP-driven scheduler in a single thread (`Machine::run_ap_cpus`), so no
two of them execute concurrently and a read-modify-write is already indivisible with
respect to the others. The prefix is recorded and the instruction translated as the
plain RMW it wraps. What matters is not translating it where the prefix would be
illegal: `lockable()` in the decoder restricts it to memory-destination RMW forms, so
`lock and rax,rsi` — which faults on real hardware — goes to the interpreter rather
than being silently translated as if the prefix were absent.

**The bit tests fold most of their work at translation time.** Intel adjusts the bit
base for an offset past the operand size, and Tier-0 does it for the immediate form
too, so `0F BA` group 8 resolves the whole thing in the decoder: the bit position
becomes a constant and any displacement folds into the operand. The register-offset
forms (`0F A3`/`AB`/`B3`/`BB`) need it at run time, but `bits` is a power of two, so
Intel's euclidean divide and remainder are an arithmetic shift and a mask. Only CF is
written — Tier-0 leaves the flags Intel calls undefined alone, so Tier-1 must too.

Together: coverage 91.7% -> 94.1%, and every window improved — steady state 301.9 ->
264.6, whole session 517.6 -> 459.0, warm-up 949.2 -> 848.1.

### Where it stops paying

The next three — `PREFETCH*`, `MOVNTI`, and the register-offset bit tests — are each
correct, each on the wall, and together a **3.1% loss**: 272.6 against 264.5 host
instructions per guest instruction, interleaved and repeated. They are off by
default behind `AERO_X86_HINT_OPS=1`.

The reason is worth keeping. They let ~1,700 more blocks compile while the wall
barely moved, 843K wanted executions to 833K — because the wall names the *first*
instruction a block cannot translate, and fixing it often just moves the wall to a
later one in the same block. The admission cost is paid either way. So an entry high
on that list is a *candidate*, not a promise; only the measurement says which.

Raising the admission threshold does not rescue them, and is worth recording because
the policy was last tuned at 59% coverage: at 94%, thresholds of 4/8/16/32 measure
272.7 / 557.5 / 538.5 / 601.9. Four is right by a wide margin — coverage lost to a
higher bar costs far more than the compilation it avoids.


## Two compilers at once

At 94.7% coverage the compiled code is finally the majority of what is left. Taking
the difference of two profiles, so the snapshot load cancels the way the headline
metric makes it, steady state divides as: **[JIT] 112.7, the emulator around it 95.2,
libc 10.2** host instructions per guest instruction. That is ~119 per *compiled*
guest instruction, against the 45-90 a mature TCG achieves — so the code generator
itself is now worth attacking.

Wasmtime has two, and the choice had been made once, globally:

| | admission | steady state |
| --- | --- | --- |
| Winch | ~144us a block | 267.7 |
| Cranelift | ~1.5ms a block | **197.6** |
| Cranelift, warm-up window | | **2283.0** |

Cranelift's code is a quarter cheaper to run and its compile time is worse than the
interpreter it replaces during warm-up. Neither alone is right, and execution is
concentrated enough to have both: the busiest **1% of blocks account for 82.3%** of
all block executions, and the busiest 5% for 91.9% (`AERO_CT_HOTNESS=1`). So blocks
that pass 2,000 executions are recompiled by Cranelift and everything else stays with
Winch — 1,175 blocks over a 200M-instruction session, at 1.48ms each.

Measured, interleaved: **steady state 228.7 against 267.7**, 14.6%. Whole session
456.9 against 462.3, and warm-up 851.2 to 912.5 — it costs where compilation is being
paid for and returns it several times over afterwards. `AERO_JIT_NO_TIERUP=1` turns it
off; `AERO_CT_PROMOTE` sets the threshold, and 200/500/2000/8000 measure
250.3/251.3/229.2/230.0 — promoting more is worse, because the extra compilation costs
more than the better code returns.

### The part that does not work by itself

Wasmtime picks its compiler per `Engine`, so this needs two of them, and a `Store`
belongs to one engine. Guest RAM, `CpuState` and the JIT context all live in linear
memory, so the two engines have to be looking at the same pages.

`Config::with_host_memory` is the hook for that, and `backend::shared_memory` maps
one region — accessible bytes followed by an unmapped guard, because Winch refuses to
start without signals-based traps and so needs an out-of-range access to fault.

**The trap: a memory created by `Memory::new` ignores that configuration entirely.**
It builds its own instance with a default on-demand allocator whose memory creator is
`None`; only a memory a *module defines* is routed through the host creator. Nothing
fails when this is wrong — the second engine quietly gets its own private copy of
guest memory, and the machine resets about a million instructions later. So the
memory is defined and exported by a tiny anchor module, and that memory is what the
block modules import as `env.memory`, exactly as before. There is an assertion at
construction that the two engines' data pointers are equal, so a future Wasmtime that
changes this says so immediately.

Two more things that cost a debugging cycle each, both worth stating because they are
not obvious from the diff:

- The chain loop read `CpuState.rip` back through the store's `Memory` handle. A
  handle belongs to the store that made it, and a promoted block runs in the other
  one. It reads through the mapping's base pointer now, which is the one thing true
  for either store.
- The IR is kept from admission rather than re-derived at promotion. A block's entry
  is a *linear* address; Windows changes CR3 constantly, and re-decoding at promotion
  time can read a completely different process's code. Re-decoding from saved bytes
  is closer but still not identical — a block that ended at the decoder's instruction
  limit re-decodes shorter when the bytes run out.


## The chain step becomes the hot path

With Cranelift on the hot blocks, the profile moved again, and further than expected.
Differencing two profiles over the steady window: **[JIT] 38.8, the emulator around
it 113.0, libc 7.4** host instructions per guest instruction. Compiled code had gone
from ~119 per compiled guest instruction to ~41 — TCG-class — and the emulator around
it was suddenly 71% of the cost.

The busiest thing in it was the closure that resolves the next block in a chain. It
runs once per chain step, 42 million times over a 200M-instruction session, and it
was doing **two** map lookups: the block cache, and a `HashMap<u64, u64>` of execution
counts added for the tier-up's promotion policy. That second one used the default
hasher, which is SipHash.

Folding the counter into the block entry, so the lookup the step already performs
serves the whole step, took steady state from **229.4 to 186.9 — 18.5%**. Moving the
reporting counters (`AERO_CT_EDGES`, chain locality, self-loops) behind their flag
took it to 184.4.

Worth stating plainly, because it is easy to add and hard to see: at 42M calls a
session, a single extra hash lookup on the chain step is worth more than any of the
opcode work in this file. The closure now does one lookup and nothing that is not
needed to choose the next block.

### The marginal opcodes are still a loss, and more so

`PREFETCH*`, `MOVNTI` and the register-offset bit tests measured 3.1% worse when they
were added. Re-measured against everything since — the tier-up, the cheaper chain
step — they are **6.2% worse**: 195.7 against 184.3, interleaved. The reason is the
same and has only sharpened: they let ~1,700 more blocks compile without moving the
coverage wall, because the wall names the first instruction a block cannot translate
and fixing it moves the wall later in the same block. Admission is paid either way,
and every other cost around it has come down, so the admission that buys nothing is a
larger share of what is left.


## Three more ideas, all measured, all rejected

Worth writing down because each looks obviously right and each is not, on this
workload, at this point on the curve. All are interleaved A/B, repeated.

**Offering the compiled tier the instruction after an assist.** When a block cannot
translate an instruction the interpreter takes over, and it carries on to the *next
taken branch* — about thirteen instructions — because that is where the seam is
offered. Offering it immediately after the assist instead should have cut that to one.
It does get offered (1.76M offers to 1.91M) and the tier does run more often, but
coverage moved 188.2M to 188.7M and the result is **187.6 against 184.4**. The
post-assist points are diverse and mostly cold, so what it buys is admission that
does not repay itself. `AERO_CT_SEAM_AFTER_ASSIST=1`.

**Superblocks, re-measured.** They were rejected long ago at 72% coverage, with
admission eight times more expensive, no tier-up, and a chain step that cost twice
what it does now. Every one of those has changed, so the question was worth asking
again: **199.0 against 184.4**. The answer is the same and for the same reason —
fall-through is the wrong edge to fuse, and a same-sized cache of larger blocks covers
fewer entry points.

**The marginal opcodes, re-measured.** `PREFETCH*`, `MOVNTI` and the register-offset
bit tests were 3.1% worse when added and are now **6.2% worse** (195.7 against 184.3).
Everything around them got cheaper, so admission that buys no coverage is a larger
share of what remains.

The pattern across all three, and worth carrying forward: at this point the tier's
cost is not in any one mechanism but in *how many blocks get compiled for how much
running*. Anything that admits more blocks without moving coverage loses, and the
margin against it widens every time something else gets faster.


### A cache in front of the chain step

The step's map lookup is what is left of it, so it now goes through a 4096-entry
direct-mapped cache holding `(rip, table_index, guest_insts, exec_count)` — 128KiB,
which stays in L2. The execution count lives in the entry, which is what keeps a hit
from touching the map at all; a block evicted before it is promoted loses its count,
and that is the right trade, because a block cold enough to be evicted from a cache
this size is not one worth recompiling.

Worth 1.3% — 182.0 against 184.4 — which is less than the map lookup looked like it
was costing, so most of the closure's share is the call and the checks around it
rather than the lookup itself. It is kept because it is also what makes the tier-up's
execution counting free.

Blocks are only ever inserted, never removed or rewritten, so the only way an entry
can be wrong is by being for a different RIP, which the tag compare catches. The two
insertion paths clear the slot anyway, in case a RIP is ever admitted twice.

### A profiling correction worth keeping

An earlier note in this file said `lz4_decompress_into` appeared to scale with run
length, and that the headline number might therefore include snapshot page-in. **It
does not.** Measured directly: 1,560.6M host instructions at `--max-insts 20000000`
and 1,618.4M at 200000000 — a 0.3-per-guest-instruction difference over 180M
instructions. Snapshot loading is a single streaming pass and cancels exactly in the
two-budget differencing.

What produced the wrong impression was per-symbol differencing between an 80M and a
200M profile, which put lz4 at 17% of the delta. `perf record`'s period sums are not
reliable at these sampling rates — the same pair reported a total five times the
measured value. **Use the two-budget `perf stat` difference for magnitudes and treat
per-symbol profiles as ordering only**, and check any surprising symbol directly
before believing it.


### Trimming the per-block locals: worth 2% before the tier-up, 0.4% after

Every block declares a fixed 34 i64 locals plus one per IR value, and WASM requires
them zero-initialised, so a block stores zero to every slot each time it is entered —
once per four guest instructions. Measured on Winch: **+12.2ns per call at 43 locals
against 1**, roughly linear. The fixed layout is the part worth attacking, not the IR
values: a real block declares 34 fixed and about 10 values, so **77% of them are
fixed**.

That sized the change at about 2%, and it is not a small change — the layout assigns
indices positionally, so trimming means renumbering, and a group wrongly marked
absent is a silent miscompile rather than an error.

It is now worth 0.4%, because the blocks that matter moved to a compiler that does
not have the problem:

| declared locals | Winch | Cranelift |
| --- | --- | --- |
| 1 | 24.7 ns | 21.3 ns |
| 16 | 29.2 ns | 21.5 ns |
| 34 | 33.6 ns | 23.9 ns |
| 43 | 37.0 ns | 24.8 ns |
| 64 | 42.2 ns | 21.1 ns |

Cranelift's register allocator sees the locals are never read and does not emit the
stores; Winch, translating a stack machine straight through, does. The hot blocks are
Cranelift's, so what is left applies only to the 18% of executions still on Winch.

Worth recording as a shape rather than a number: **the tier-up did not just make
compiled code faster, it changed which optimisations are worth doing.** An estimate
made before it is not valid after it.


## The coverage tail: settled

Interpretation is still the largest single term — 5.3% of guest instructions at about
1240 host instructions each, which is 37% of the total. The relationship is linear and
measured: compiling `RDTSC` moved coverage 20 points and the total 248.6, so each
point of coverage is worth ~12.4 host instructions per guest instruction. Closing the
remaining 5.3 points would be worth ~66, taking 182 to about 116.

Six attempts, every one measured, every one a loss:

| attempt | result |
| --- | --- |
| `PREFETCH*`, `MOVNTI`, register-offset bit tests | 195.7 vs 184.3 |
| offering the seam after an assist | 187.6 vs 184.4 |
| handing back right after the refused instruction | 184.8 vs 182.2 |
| superblocks (re-measured) | 199.0 vs 184.4 |
| raising the admission threshold | 557.5 / 538.5 / 601.9 vs 272.7 |
| lowering the promotion threshold | 250.3 / 251.3 vs 229.2 |

The last of these is the clearest. `AERO_CT_RESUME=1` gives the tier a one-shot resume
point: when a chain stops at a block Tier-1 refused, the interpreter hands back as soon
as it has stepped over the offending instruction, instead of running on to the next
taken branch. It is a count rather than a RIP, because the length of an instruction
Tier-1 could not decode is exactly what is not known, and it costs one load and one
compare per interpreted instruction.

It works, and it does not help — because there is no compiled block at the resume
point either, so the seam declines and the interpreter carries on regardless. Making
it pay means *admitting* a block there, which is the trade that lost the other five
times.

**The conclusion is not "these particular ideas were wrong" but that the tail is
priced out.** A block admitted for a site hit ~280 times cannot repay ~144us of
compilation against ~13 interpreted instructions a visit, and every improvement
elsewhere makes that arithmetic worse, not better. The remaining 5.3% is the part of a
Windows desktop that a block-at-a-time compiler cannot profitably reach.


## What the metric cannot see, and what that leaves us able to say

Host instructions retired per guest instruction is the headline here because it is
*exact*: the same configuration measured three times gives 1280.0, 1280.0, 1280.0 for
the interpreter and 182.1, 182.2, 182.1 for the tier, on a box under load average 32.

It is also **blind to the memory system by construction**. A change that improves
cache or TLB behaviour retires the same instructions and cannot show up at all.
Asking the kernel for huge pages behind guest RAM (`MADV_HUGEPAGE`, 2MiB instead of
4KiB for a multi-gigabyte mapping, so 512x fewer faults and far less host TLB
pressure) measures **exactly neutral — 182.1 both ways** — which is what a blind
metric returns whether the change helped or not. The `madvise` is kept because the
reasoning is sound and it costs nothing, but it is unverified: this host cannot tell.

Cycles would see it. They are not usable here. Differencing cycles across two budgets
gives, for three back-to-back pairs:

| | interpreter | tier |
| --- | --- | --- |
| r1 | 657.5 | 65.5 |
| r2 | 697.4 | 245.2 |
| r3 | 701.8 | 51.8 |

The interpreter holds to about 3%; the tier does not, because its total is small
enough relative to the fixed startup that differencing amplifies whatever else the
machine is doing. A single early reading of 80.6 suggested the tier was 8.4x the
interpreter in cycles against 7.0x in instructions — that was one sample and does not
survive repetition.

**What can be said** is bounded rather than precise: at 182.1 host instructions per
guest instruction and an IPC somewhere between 1.8 and 3.5, the tier runs this guest
at roughly **30-58 MIPS on a 3.04GHz EPYC 7502P**. The target is >=500. A quiet
machine would be needed to say anything sharper, and to settle whether huge pages
matter.


## Three more measured, three more rejected

**The inline TLB has no headroom.** 256 entries of 16 bytes covers 1MiB of guest
address space, and every guest access probes it while a miss is a *host call* to
`mmu_translate` — which the instruction metric does see. Enlarging it to 1024 entries
measures **182.1, identical**. The hit rate is already high enough that misses are not
what the memory path costs, and smaller is better for the host cache.

**Batching Cranelift promotions halves their cost and still loses.** Most of
Cranelift's ~1.5ms is per module, so recompiling sixteen hot blocks into one module
takes it to **730.4us each**. Steady state goes the wrong way: **184.8 at sixteen a
module, 183.5 at four, against 182.1 at one.** A block waits for the rest of its batch
before it gets the better code, and that delay costs more than the compilation saves.
The batching path is kept behind `AERO_CT_PROMOTE_BATCH` because it is what in-WASM
chaining between hot blocks would need — a promoted block currently lives in a
single-block module, so 82% of executions are in modules that cannot chain internally.

**Huge pages cannot be judged here.** See the section above on what the metric cannot
see; the `madvise` is kept, unverified.

### Making a cycles measurement work, and one that did not

Cycles see what instructions cannot, so it is worth saying how far they can be pushed
on a box at load average 32. Pinning to a core and taking the least-contended sample
helps — contention only ever *adds* cycles — but the minimum has to be over **paired**
differences. Taking the minimum at each budget separately and subtracting pairs a
lucky short run at one budget with a lucky short run at the other: that reported 27.7
cycles per guest instruction and an IPC of 6.58, which no Zen 2 core can sustain.

Even done correctly it is not tight enough to A/B a few percent: repeated paired
minima give 129.2 and 141.0. `scripts`-adjacent `hostcycles-min.sh` in the scratchpad
does it properly and is the starting point if a quiet machine ever becomes available.


## The axis the headline metric cannot see

Host instructions retired is exact and load-immune, which is why it is the headline —
and it is blind to the memory system. That blindness turns out to matter, and the way
to see past it is not cycles (unusable here) but **event counts**: a cache or TLB miss
is something this process *caused*, not time it took, so it differences across two
budgets exactly as instructions do.

### The compiled tier's footprint

`AERO_CT_FOOTPRINT` reports what has been emitted. A 200M-instruction desktop session
produces **39.0 MiB of WASM across 60,144 blocks — 679 bytes each.** The machine code
Winch and Cranelift make of that is larger again. This host is an EPYC 7502P: 16 MiB
of L3 per CCX. The compiled tier does not fit in any cache, and never will at
block-at-a-time granularity, so a cold block execution is a walk through DRAM.

Measured, `cache-misses` per guest instruction:

| | footprint | cache misses (mean) |
| --- | --- | --- |
| shared load helpers (default) | 39.0 MiB | **~0.70** |
| the same probe inlined at each access | 48.8 MiB | ~1.16 |
| superblocks | 75.2 MiB | ~1.27 |

### What that changes

**The shared load helpers are worth far more than they measured.** Emitting the TLB
probe once per module rather than at every access took a load from 218 emitted bytes
to 21, and measured a *wash* on host instructions — 688.6 against 691.2 over a whole
session. On cache misses it is **~40% fewer**, consistently, over eight paired runs.
They were kept "for the browser's sake"; the better reason is that a fifth of the
tier's cache footprint goes with them.

**Superblocks were rejected for the right reason and a second one.** They nearly
double the footprint — 75.2 MiB against 39.0, because fusing along the fall-through
duplicates code — and double the cache misses with it. Worse on both metrics.

**Emitted size is a lever on execution, not just on admission.** An earlier section
here concludes that admission cost is per exported function and nearly independent of
body size, and that remains true. It does not follow that body size is free: it is not
free at *run* time, through the cache, and that cost is invisible to the metric this
file leads with.

The biggest remaining component is the fixed ~440 bytes every block emits before it
does any work — 65% of the footprint at 4.3 guest instructions a block. Out-of-lining
more of it, as the load probe was, is the concrete next step on this axis. It cannot
be evaluated properly here: `cache-misses` shows direction reliably but its magnitude
swings (0.365 to 0.996 for one configuration), and cycles are unusable at load
average 32.


### What the footprint is made of

Measured in a module of sixteen, which is what the tier builds — the single-block
figures elsewhere in this file charge one block for the shared memory helper, the
dispatcher and the imports, and so overstate the fixed cost badly:

| block | bytes each, sixteen to a module |
| --- | --- |
| `jmp $+2` — no registers, no flags, no memory | **92** |
| `mov eax,1; jmp` | 123 |
| `cmp rax,rcx; jne` | 187 |
| `mov rax,[rsi]; jmp` | 167 |
| `mov rax,[rsi+8]; cmp rax,rcx; jne` | 269 |

So the floor is 92 bytes, a guest register costs about 31 (its load at entry, its
spill at exit, both already gated on whether the block reads or writes it), flags cost
about 95, and a memory access about 75. A real session averages 679 bytes across 4.3
guest instructions a block, so real blocks touch considerably more than these samples.

There is no single dominant waste in that, which is the useful finding: the shape is
about as compact as a design that keeps guest registers in WASM locals allows.

The 66 is mostly the lazy-flag record, and that is worth checking rather than
assuming: recording the operation and its operands looks expensive next to an `add`,
but computing six flags eagerly is worse. Measured, whole session: **39.0 MiB lazy
against 44.5 MiB eager**, and lazy is also 2% ahead on host instructions (686.2
against 700.5). It was already the default and it is right on both counts.

Two ways out of the register load/spill were considered and neither survives contact:

**Leave guest registers in memory** and let Cranelift's register allocator do the work
— it recompiles promoted blocks from IR anyway, so they could use different codegen
from the Winch ones. It cannot work: `CpuState` lives *in* the same linear memory as
guest RAM, and WASM offers no way to say they do not alias, so a guest store forces a
reload of every register after it. Blocks with stores are most of them.

**Pass the registers as arguments and returns**, so the dispatcher holds them across a
whole chain and no block loads or spills anything. WASM supports the multi-value
returns this needs. But eighteen arguments and eighteen returns per call is not fewer
bytes at the call site than the prologue it removes, and x86-64 has about fourteen
usable registers, so most of them are going to memory regardless.


## What an outer-loop boundary costs

The machine's outer loop re-polls devices, rebuilds the paging bus and checks for
interrupts at every batch boundary, and the batch was 1,024 guest instructions —
chosen for timer responsiveness, never for throughput. A boundary costs **about
35,000-42,000 host instructions.**

That figure is worth trusting because it falls out of two measurements that have no
reason to agree. Raising the batch to 4,096 saves 26 host instructions per guest
instruction under the interpreter and 30.6 under the compiled tier. Over the same
120M-instruction window those are 88,000 fewer boundaries, so they imply 35,500 and
41,700 host instructions per boundary — from engines whose per-instruction costs
differ by eightfold.

It matters far more to the compiled tier for exactly that reason: its own cost per
instruction is low enough that this fixed overhead is a large share of it.

| | 1,024 | 4,096 |
| --- | --- | --- |
| warm-up | 827.0 | 793.2 |
| steady state | 182.2 | **151.6** |
| whole session | 397.2 | 365.4 |

**17% off the steady state**, and the largest single win since the tier-up.

### Why this is doing less work rather than different work

Changing the quantum changes *when* devices are polled, which can change what the
guest does — and then two measurements are not comparable at all. Two of the values
tried are exactly that trap: at 2,048 the block cache hit its 65,536 ceiling and at
8,192 the working set had moved by 200M, and both measured far worse for reasons that
have nothing to do with batch overhead.

4,096 is not one of those. The guest executes **identically** to 1,024 through 20M
instructions — same final RIP, RSP, registers and CR3 — and over a full 200M session
the compiled tier's working set is unchanged: 60,016 blocks against 60,144, coverage
within 0.002%, blocks executed within 0.13%, `at_cap` zero either way. The measurement
repeats exactly (151.6, 151.5).

8,192 is also guest-identical at 20M and is *not* taken, because its working set has
moved by 200M — the point at which a longer run would start measuring something else.
The value has to stay well under the 16,384 that once let a HAL 0x5C clock-sync wait
expire entirely inside one batch. `AERO_BATCH_INSTS` overrides it.


### What else is in a boundary

A device poll costs about **1,500 host instructions** whether or not the controller
has anything to do — recovered from removing four duplicated ones, which is worth 1%
on its own (151.6 to 150.1, guest execution identical through 20M). The outer loop
made thirteen such calls; `process_ahci`, `process_nvme`, `process_virtio_blk` and
`process_aerogpu` were each called twice, once for the halted-CPU case and again after
`poll_network`, and nothing `poll_network` does gives them new work.

Thirteen polls is roughly 20,000 of the ~41,700 instructions a boundary costs. The
rest is the bus and MMU construction, platform timer ticks, AP scheduling and the
batch's own setup — `Mmu::new()` alone zeroes a ~16KiB two-bank TLB every boundary,
though the durable MMU is swapped in immediately afterwards so only the construction
is wasted, and at ~600 instructions it is 0.1%.

**Polling the non-boot devices less often does not work.** Every eighth boundary
instead of every one measures 153.4 against 150.1 — worse — and part of that is the
guest doing different work once completions are delayed (compiled coverage moves
188.21M to 187.19M), which makes the comparison worth less than it looks. Either way
there is nothing in it, and the mechanism was removed rather than left as an unused
knob.


### The rest of the boundary is work, not waste

The obvious way to size what remains is to switch each part off and measure. It does
not work here, and the way it fails is worth recording:

| gated off | steady state |
| --- | --- |
| nothing | 150.8 |
| `sync_legacy_vga_scanout_state` | 150.9 |
| `poll_platform_interrupt` | **1295.0** |
| the device polls | **473.3** |

The last two are not measurements of what those calls cost. Without interrupt polling
the guest never receives one, and without device polling its I/O never completes; in
both cases it spins, and the number reflects a broken machine rather than a cheaper
one. Disable-to-size only works on work that is genuinely redundant — which is what
the duplicated storage polls were, and why removing those was worth a real 1%.

VGA scanout sync is free, and that is the whole of what the technique found.

So the ~21,700 host instructions a boundary spends outside the device polls are
functional: interrupt delivery, timer ticks, the paging bus and MMU handed to the
batch, AP scheduling. Making them cheaper means making each of them cheaper, not
skipping them, and none is large enough on its own to be the next thing worth doing.


## What a point of coverage is actually worth

The interpreted share is 5.9% of guest instructions, and it is easy to price that at
the interpreter's own rate — 1,252 host instructions each, so 5.9% is 74 of the total.
That is wrong, and the error is large enough to have misdirected several rounds of
work here.

Measured two ways, at different points and by different means:

| | coverage | steady state |
| --- | --- | --- |
| `RDTSC` compiled | 94.1% | 150.2 |
| `RDTSC` interpreted | 78.8% | 428.2 |
| block cache 65,536 | 94.1% | 150.1 |
| block cache 40,000 | 73.5% | 522.3 |

18.2 host instructions per guest instruction per point of coverage, and 18.1. So the
5.9% still interpreted contributes about **107 of the 150** — over two thirds — and
everything else in the emulator, the compiled code included, is about 43.

An instruction costs more when interpreted *in a mixed run* than it does in a
pure-interpreter run, because moving it there also breaks a chain, pollutes caches and
costs a seam offer. Costing coverage at the interpreter's own rate understates it by
about 45%.

### Stepping over what stops a chain

A chain ends where the tier has no block, and for a block it *refused* that is a
specific instruction Tier-1 cannot translate. The tier now steps that one instruction
itself, through Tier-0, and compiles a block wherever the guest lands — so the chain
resumes instead of the interpreter running on to the next branch.

This is the third attempt at that idea and the first that works. Offering the seam
after an assist failed because nothing had compiled the resume point; arming a
one-shot resume failed for the same reason. What is different here is that the tier
stays in control afterwards and admits the block itself.

**147.8 against 150.1**, with 700,998 instructions stepped over and 672 blocks added.

Two things had to be got right. Tier-0 holds the decode cache across a whole batch to
keep a TLS access off the per-instruction path, and the seam is offered from inside
that borrow — so stepping an instruction from the seam panicked with `RefCell already
borrowed` until the offer sites release it. And **one step is the whole of it**: the
untranslatable instructions cluster, so stepping further to get through a run of them
is tempting, and it is sharply negative — 303.5 at two steps and 670.6 at four, against
147.8 at one. A single-instruction Tier-0 batch is expensive, and the extra blocks blow
the cache: at four, coverage *fell* to 85% and the block cache hit its ceiling.
`AERO_CT_STEP_OVER_MAX` sets it, `AERO_CT_NO_STEP_OVER=1` turns it off.

Two things go wrong at once past one step, and it is worth separating them because
only one is obvious. Steps triple — 700,998 to 2,553,813 — and coverage *falls*, from
188.97M guest instructions to 184.57M, because a block admitted at every landing point
displaces better ones. At 18.1 per point that lost coverage accounts for about 40 of
the 156-instruction regression; the rest is the cost of the steps themselves. Doubling
the block cache does not rescue it (303.9 at 131,072 against 303.5 at 65,536), so the
cache ceiling is not what binds.

The per-step cost does not come out of that arithmetic consistently — the same model
applied at one step predicts a loss where a gain is measured — so it is left as an
observation rather than a number. What is solid is the shape: one step pays, two do
not, and the reason is at least half about which blocks end up in the cache.


## The mid-block assist: built, correct on the small oracles, and not finished

Every attempt above to close the coverage tail ends the block at the instruction
Tier-1 cannot translate, and then needs something compiled at wherever the guest
lands. The instructions it refuses come in short runs, so that lands on another one.
The mechanism that does not have that shape is an assist *inside* the block: the host
performs the instruction and the block carries on with what it leaves.

It is built. `IrInst::Assist { rip, next_rip }` spills the registers and flags to
`CpuState`, calls `env.assist`, and reloads them. The host overlays those registers
onto the machine's real `CpuState` — which has CR3, the MSRs, the segment bases and
the paging mode, none of which a block maintains — runs one instruction through
Tier-0, and reports where the guest went. The block compares that against where it was
decoded to continue and leaves if they differ, so anything unexpected is a clean exit
rather than a wrong one.

It works: `bails_to_interp` goes to **zero**, the first time no block is refused at
all, and the small oracles match at 100K.

It is also **correct**, in the end: the machine-level oracles match at 100K, 600K, 1M
and 2M. Getting there needed three more things, each of which is the sort of thing
this design makes easy to miss:

- **The inline TLB.** An assisted instruction can write CR3, CR0, CR4 or EFER, and the
  block then carries on against entries filled under the old translation. The host
  bumps the TLB salt when the MMU generation moves and the block reloads it after the
  call, which makes every existing tag mismatch — the same thing a flush does. The
  next chain entry re-flushes anyway, because the generation it compares has moved.
- **The segment bases.** A block reads FS and GS out of linear memory for
  `fs:`/`gs:`-relative addressing, and an assisted `swapgs` changes them. Windows
  follows `swapgs` with `mov gs:[...], rsp` *immediately*, so not writing them back
  put the kernel stack pointer through a user-mode base and the guest died a few
  hundred thousand instructions later.
- **The interrupt-class instructions.** `CLI`, `STI`, `INT`, `IRET` and their kin need
  the interpreter's interrupt state, not just architectural registers, so an assist
  can only report failure and the block bails — every time. `cli` alone was 2,423 of
  2,802 failed assists. They are excluded now (`DecodedInst::assistable`), and with
  that the oracles match.

**And it is a loss, decisively.** With the interrupt class excluded, coverage reaches
**97.2%** — but at **20.8 million assists, one in ten of every guest instruction
executed** — and the cost is 597.4 host instructions per guest instruction against
147.7.

The arithmetic is the whole story: an assist costs about **6,850 host instructions**
and interpreting the same instruction costs about **1,250**. A host call per
instruction is five times worse than the interpreter, and this workload has about a
tenth of its instructions in the class Tier-1 cannot translate. No amount of coverage
pays for that. Caching the decode inside the assist path — the same instruction is
handed over roughly 1,700 times per site — changes nothing measurable (151.9 against
152.0), so it is the marshalling and the boundary, not the decoding.

That also explains why every cheaper approximation of this idea failed. They were not
worse implementations of a good mechanism; the mechanism itself is priced out, because
the instructions in question are *frequent*. The way to reach them is to translate
them, not to call out for them. `AERO_CT_ASSISTS=1` if that ever changes.

Three bugs on the way there, each of which presented as "the feature silently does
nothing" rather than as an error, and each worth remembering:

- The early-exit path pushed a value onto the stack *and* branched. Every other exit
  in this code generator sets `scratch_local` and branches with the stack empty. The
  resulting module does not validate — which showed up as 336 blocks admitted out of
  60,000, not as a compile error.
- The import looked the guest memory up with `Caller::get_export("memory")`. A block
  module *imports* memory; it exports only its dispatcher. So the lookup found
  nothing and the closure reported every assist as a fault, which is indistinguishable
  from correct behaviour with the assist never happening.
- The host was handed a fresh `CpuState` rather than the machine's. A blank one reads
  CR3 as zero and writes MSRs into something discarded, so `wrmsr` of `IA32_GS_BASE`
  quietly went nowhere and the guest stopped sixty thousand instructions later.


## Translating the tail does not work either

The assist result said the way to reach these instructions is to *translate* them, not
to call out for them. So: which ones, and how much?

`AERO_CT_ASSIST_HIST=1` answers the first part properly. It counts what is actually
handed to the host, weighted by how often it runs — which is not the coverage wall,
because the wall names only the *first* instruction a block could not take. Ranked:

| | executions per 200M |
| --- | --- |
| `MOVNTI` | ~750,000 |
| `MOV CR8, r` / `MOV r, CR8` | ~390,000 |
| `PREFETCHW` | ~240,000 |
| `BSF` | ~235,000 |
| `LOCK CMPXCHG` | ~210,000 |
| `LOCK XADD` | ~125,000 |
| `LFENCE` | ~40,000 |

`XADD`, `CMPXCHG` and the fences were implemented — decode, IR, translation, five new
equivalence tests, guest state identical to the interpreter at 100K through 3M, and
**zero verifier mismatches**. Correct, and every one of them a loss:

| | steady state |
| --- | --- |
| none of them (baseline) | **147.7** |
| fences only | 149.2 |
| fences + `CMPXCHG` | 297.1 |
| all three | 312.5 |

`XADD` alone is 147.9 of that, which is not a per-execution cost — 125,000 executions
cannot account for it. It is structural: enabling `XADD` admits **4,464 more blocks**
and *loses* **3.4 million guest instructions of coverage**, with the average block
retiring 4.33 instructions down to 4.21. The blocks get shorter, which means they are
bailing at run time, and a block that bails is worse than one that was never taken.

`AERO_X86_RMW_OPS=1` turns them on.

### What that settles

This was the last identified route. Coverage is worth 18.1 host instructions per guest
instruction per point and 5.5 points remain, so the arithmetic keeps saying the tail is
worth ~100 — and every mechanism for reaching it has now been built and measured:

- **assist from the interpreter** — priced out, a host call is 5x an interpreted
  instruction and a tenth of instructions need one
- **assist inside the block** — same, 597.4
- **hand back after the refusal** (seam-after-assist, one-shot resume, step-over) —
  one step pays 1.6%, more is sharply negative
- **translate them** — correct, and 2.1x worse

They fail for one reason wearing different clothes: an instruction Tier-1 cannot take
is not isolated. It sits in code that is *full* of them, and any change that lets a
block reach one lets it reach the next, so blocks grow, bail, and multiply. The tail
is not a list of missing opcodes. It is a region of the guest that this block-at-a-time
design does not fit.


## The interpreter, looked at directly

The interpreted share is two thirds of what the compiled tier still costs, so the
other way at it is to make interpreting cheaper rather than to interpret less. Profiled
on its own (`AERO_X=1`, 20M against 80M), by share:

| | share |
| --- | --- |
| the batch loop body | 23% |
| `read_insn_window` | 12% |
| `fetch_wrapped_seg_ip_first_page` | 10% |
| `MachineCpuBus::fetch` | 8% |
| `exec_decoded` | 8% |
| `aero_x86::decode` | 6% |
| `ops_alu::exec` | 5% |
| the decode cache | 4% |
| MMU/TLB | 7% |

Instruction fetch is about 30% of it, which looks like a lever until the code is read:
the translation is already memoised per page (`PagingBus::fetch_page`), and the RAM
read already has a fast path that copies a constant fifteen bytes.

**The three fetch frames are not call overhead.** They are already inlined — `perf`
attributes an inlined body to the symbol it came from, which reads as three separate
costs. Marking them `#[inline]` explicitly changes nothing: 1255.0 before, 1255.0
after, and the compiled tier 147.8 to 147.7. The attributes were removed again rather
than left asserting a benefit that is not there.

So ~1,250 host instructions per interpreted guest instruction is what this interpreter
costs: a full fetch, a validated decode-cache lookup, an `iced-x86` decode on a miss,
the execute, and the MMU. That is high against a figure like 100-300 for a simple
interpreter, and the comparison is not fair — those do not model paging, MMIO device
dispatch, or x86 decoding of this generality. There is no single item in it to remove.


## A correction: the block cache is not a monotonic knob, and the 18.1 figure rests on it

This file has said, in several places, that a point of compiled-tier coverage is worth
about 18.1 host instructions per guest instruction, on the strength of two
measurements that agreed: compiling `RDTSC` (18.2) and shrinking the block cache
(18.1). The second one is not sound, and it takes some of the confidence in the first
with it.

Sweeping `AERO_CT_MAX_BLOCKS` near the operating point:

| cache | coverage | steady state |
| --- | --- | --- |
| 65,536 | 94.5% | **147.8** |
| 58,000 | **33.5%** | 1080.0 |
| 52,000 | 86.2% | 441.8 |

A *smaller* cache gives more than twice the coverage. Once the cache is full nothing
more is admitted, so which blocks are in it is decided by arrival order, and a small
change in capacity changes which ones win in a way that has nothing to do with
capacity. It is not a dial that trades coverage against admission cost; it is a dial
that reshuffles the working set.

So a cap sweep cannot measure the value of coverage — the two configurations differ in
*which* blocks are compiled, not only how many. The `RDTSC` comparison is cleaner but
not clean: it too moves the working set.

The one genuinely local datum is the step-over change, which strictly *adds* blocks
and adds 0.38 points of coverage for a 2.3 improvement — about 6 per point before
attributing its own admission cost, and nearer 16 after. So the honest range is 6 to
18, not the 18.1 quoted elsewhere here, and the decomposition that follows from it —
"the interpreted 5.5% is about 100 of the 147.7" — is an upper bound rather than a
figure. At the low end it is nearer 33, and then the compiled path and the emulator
around it are the larger share instead.

That does not change any of the measured results above: every mechanism for reaching
the tail was measured end to end, and end-to-end numbers do not depend on this model.
It does change what the remaining headroom is worth, and it is the sort of thing worth
knowing before spending another effort on coverage.

**`AERO_CT_MAX_BLOCKS` should be left at 65,536**, and not treated as a tuning knob
without re-measuring coverage at the same time.

---

## Chaining inside wasm: 147.7 to 132.7, and where the ceiling actually is

The correction above said the decomposition was an upper bound and that the compiled
path and the emulator around it might be the larger share. Measuring that properly is
what produced the next win.

### First, a measurement that was quietly wrong

Splitting the run by DSO needs `perf record` at a fixed period, differenced across two
guest-instruction budgets the same way `hostinsns2.sh` does. At `-c 500000` the buckets
summed to **5.5** host instructions per guest instruction against a known total of
147.7 — the kernel was throttling the sample rate and the shortfall grew with the run
length, so the difference lost almost everything. At `-c 5000000` the same method
reproduces the total to within 0.2:

| bucket | host insns per guest insn | share |
| --- | --- | --- |
| `aero-machine` (emulator + interpreter) | 81.8 | 55.3% |
| `[JIT]` (compiled code) | 61.1 | 41.3% |
| `libc` | 5.5 | 3.7% |

**Always check the bucket sum against the independently known total.** A profile that
silently drops 96% of its events still produces a plausible-looking ranking.

Wasmtime writes a perf map when `AERO_JIT_PERFMAP=1` is set (diagnostic only, off by
default), which is what makes `[JIT]` resolve rather than showing up as one anonymous
mapping.

### The cost that was hiding in plain sight

Sorted by symbol, the top of the steady-state profile is:

| host insns per guest insn | symbol |
| --- | --- |
| 22.0 | `CompiledBackend::…` (the block call path) |
| 12.2 | `run_chain_once::{{closure}}` |
| 7.8 | `wasmtime::…::HostContext::array_call_trampoline` |
| 6.2 | `PagingBus::jit_translate` |
| 4.8 | `drop_in_place<Cell<…call_thread_state…>>` |
| 4.6 | `Machine::run_slice_inner` |

Rows 1, 3 and 5 are one thing: **entering and leaving wasm**. A block retires 4.28
guest instructions (`executed_blocks` against `executed_guest_instructions`), so 34.6
per guest instruction is about 150 host instructions per block call — and the chain
loop was doing one `TypedFunc::call` per block, always returning to the host in
between, even when the successor was a block of the same module.

### What was built

`Tier1WasmOptions::chain_module_id` makes the module export a second dispatcher that
calls the first in a loop: run a block, fold its retired count into
`JitContext::chain_retired`, stop on the exit sentinel or the budget or the block cap,
otherwise probe a direct-mapped RIP map in the JIT context and call the next block
directly. The map entry carries the module id, so a RIP belonging to another module is
not followed. Blocks are untouched — the dispatcher folds in the count the block
already wrote, which is why none of this needed a change to block codegen.

`AERO_CT_NO_WASM_CHAIN=1` turns it off. Steady state: **132.7 with, 148.9 without.**

Two things about the budget. The loop tests it *after* running a block, so it is given
`budget - max_block_insts`; the machine's accounting has to land exactly on
`--max-insts` for the interpreter oracles to hold at a count. And chains are long —
268,192 chains averaging 22.77 blocks — so only 3,686 of them lack the budget to chain
at all. The budget is not what bounds this.

### The map is populated at promotion, and that is not a detail

A block entered inside wasm never reaches the host's chain-step closure, and that
closure is what counts executions and queues a block for Cranelift. Chaining to a block
therefore *hides it from promotion*.

Populating the map at admission as well raises the in-wasm share from 20.1% of block
entries to 26.5% — and costs **319.6 against 132.8**. Promotions fall from 237 to 215,
and the losses are concentrated at the top, because the hottest blocks are exactly the
ones most often reached by chaining. Cranelift over Winch is worth 77 host instructions
per guest instruction here (147.7 against 224.7 with `AERO_JIT_NO_TIERUP=1`), which
dwarfs what chaining saves. It is off behind `AERO_CT_CHAIN_ADMISSION=1`.

Promoted blocks cannot suffer this, having already been promoted. Hence: the map is
filled at promotion only.

### The ceiling, measured rather than assumed

`AERO_CT_PROMOTED_EDGES` (needs `AERO_CT_EDGES=1`) reports that 36.3% of chain steps
land on a promoted block and 28.7% run promoted-to-promoted.

**Those two figures are undercounts, and the reason is worth remembering.** The edge
recorder runs in the *host's* chain-step closure, and a step taken inside wasm never
reaches it — so the traffic it misses is exactly the promoted-to-promoted traffic, which
is the traffic being counted. Adding the in-wasm steps back: of 6,106,460 block entries,
about 2,899,000 (**47.5%**) land on a promoted block and about 2,549,000 (**41.7%**) run
promoted-to-promoted. A counter that is blind to the case it measures will always read
low; check what the instrumentation can see before quoting it as a ceiling.

That killed the obvious next idea. Blocks cross the promotion threshold a few at a time,
so the promoted set is scattered across many small modules, and two blocks chain only
when they share one — so gathering the whole promoted set into one module periodically
should have paid. It is implemented (`AERO_CT_CONSOLIDATE=1`,
`promote_blocks_replacing`) and it is **239.6 against 132.8**: it moves the in-wasm
share from 20.1% to 20.3% and triples the Cranelift cost per block (2041us against
708us) recompiling the set each time.

The premise was wrong, and the figure that misled was one already in this file: "the
busiest 188 blocks cover 89% of chain steps" is read from the 4096-entry chain cache,
so it describes the resident set at the end of a run, not the session. **Execution here
is much flatter than that makes it look.** Where a concentration figure comes from
matters as much as its value.

Even with the corrected 41.7%, gathering the set still loses: the recompile is paid now
and in full, and the chaining it buys is spread thinly over the rest of the run.

### Knobs that were swept and left alone

- `AERO_CT_PROMOTE` 2000 is right: 800 gives 302.9 and 300 gives 279.2 *with* chaining.
  Promotion is already saturated — 237 blocks — which is why lowering it only adds
  Cranelift compile time.
- `AERO_CT_BATCH` 16 is right: 64 gives 292.8, 256 gives 170.3, 1024 gives 626.2. The
  non-monotonicity is the same working-set reshuffling that
  `AERO_CT_MAX_BLOCKS` shows, so a sweep of it cannot be read as a measurement of
  module size either.
- `AERO_CT_PROMOTE_BATCH` defaults to 256 when chaining is available, and the promotion
  queue waits for a full group (with `PROMOTE_STALL_CHAINS` as the escape hatch, or the
  tail of the hot set would never leave Winch). Without that wait the queue drains four
  at a time and `chunks` never sees more than four — which is how in-wasm chaining can
  be built, be correct, and change nothing at all. It measured exactly 147.7 before that
  was found.

### Where it stands

| window | interpreter | tier | |
| --- | --- | --- | --- |
| warm-up (20M→80M) | 1255.0 | 790.1 | 1.59x |
| steady (80M→200M) | 1252.1 | **132.7** | **9.44x** |
| whole (20M→200M) | 1253.1 | 351.8 | 3.56x |

Still far from the ~10–20 that 500 MIPS needs. The remaining crossing cost is bounded
by module membership, and module membership is bounded by admission economics: the only
population that can be afforded in one module is the promoted set, and that set is 36%
of steps.

---

## The measurement floor: a tier change moves the guest, so small effects cannot be measured here

`hostinsns2.sh` is exactly reproducible — the same configuration gives the same number
every time, to the tenth. That reproducibility is easy to mistake for precision. It is
not: two *different* tier configurations run **different guest code**, deterministically,
and the difference between them is partly a comparison of what Windows happened to do.

The mechanism is already documented above in another context: a compiled block runs to
its end without the per-instruction interrupt checks Tier-0 makes, so delivery lands a
few instructions later. Change where block boundaries fall — which nearly every tier
change does — and an interrupt lands somewhere else, and the guest diverges from there
on. Guest time is derived from retired instructions (wall-clock TSC is off by default),
so this is not host-timing noise; it is deterministic path selection.

What it looks like in practice. The chaining dispatcher was rewritten to hoist the two
loop-invariant limits and the running totals into locals — **strictly fewer memory
operations per step, identical semantics**, and the in-wasm chain rate (20.1% -> 20.2%)
and promotion count (237 -> 237) confirmed nothing had changed behaviourally. Measured:

| window | before | after |
| --- | --- | --- |
| 80M→200M | 132.7 | 277.8 (twice, identical) |
| 100M→160M | 112.8 | 314.5 |
| 140M→200M | 105.2 | 112.2 |

It was reverted. Not because doing less work is bad, but because there is no way to
show it helps and three windows say the paths it lands on are worse.

**How to measure anything here:**

- Only trust an effect that holds across **several independent windows** with a
  consistent sign. In-wasm chaining was −10.9%, −19.5% and −23.0% across three; that is
  believable. Anything below a few percent is under the floor and cannot be settled
  this way at all.
- A jagged sweep is a warning, not a result. `AERO_CT_PROMOTE` reads 132.8 at 2000,
  129.5 at 1600, 269.1 at 1200 and 230.4 at 1000, while at 30M all four look healthy and
  monotonic (237/288/354 blocks promoted, comparable admission cost). Picking 1600 out
  of that would be fitting arrival-order noise. The same caution already applies to
  `AERO_CT_MAX_BLOCKS` and `AERO_CT_BATCH`.
- **Microbenchmarks in this tree do not stand in for it.** Two were written and both
  measured the wrong thing: `chain_link_insns` reported 450 host instructions for a host
  chain link and `wasm_chain_step_insns` 474 for an in-wasm step — larger than a link,
  which cannot be true when chaining improves the machine by 20%. Both build their
  blocks in the *Winch* store, so they measure baseline-compiler output rather than the
  Cranelift-promoted code that actually chains. They are kept, with that written on
  them.

### What the rate actually is

The 80M→200M window still contains warm-up. Later windows:

| window | interpreter | tier | |
| --- | --- | --- | --- |
| 80M→200M | 1252.1 | 132.8 | 9.4x |
| 100M→160M | — | 112.8 | — |
| 140M→200M | — | 105.4 | ~11.9x |
| 200M→300M | 1247.3 | 114.0 | 10.9x |

It oscillates rather than continuing to fall, so the honest figure is **about 105–115
host instructions per guest instruction, roughly 11x the interpreter** — not a single
number, because which guest code a window covers moves it by ten.

---

## What actually bounds this, measured

Two numbers set the ceiling for the current design, and both were re-checked across
several windows so they are above the measurement floor.

**Cranelift against Winch is worth about half the run.** With `AERO_JIT_NO_TIERUP=1`:
225.1 against 132.7 in the 80M→200M window, and 216.0 against 105.4 in 140M→200M. That
switch also disables chaining (`chain_in_wasm` requires `can_optimize`), so isolating
it against the chaining-only measurement puts Cranelift alone at ~76. Only ~47.5% of
block entries run Cranelift code; the rest run baseline Winch output, and Winch code is
*bad* — this is the largest single lever left.

**And it cannot be pushed further, because Wasmtime compilation is the constraint.**

- A promoted block costs ~630–730us of Cranelift. Dropping the optimisation level to
  `none` (`AERO_JIT_CRANELIFT_OPT=none`) takes that to 588us — 19%, not the several-fold
  saving that would make wider promotion affordable. The cost is not optimisation.
- Nor is it per-module overhead. Those wall-clock figures — 145.0us per block at
  `AERO_CT_BATCH=16` down to 94.9 at 1024 — come from `Instant` on a machine at load
  average 32, so they measure contention as much as work. Measured in host instructions
  instead (`tests/admission_insns.rs`, differencing two block counts):

  | batch | host insns per block admitted |
  | --- | --- |
  | 16 | 416,619 |
  | 64 | 336,425 |
  | 256 | 316,394 |
  | 1024 | 311,612 |

  Batching wins a quarter and flattens. **~310,000 host instructions per block is the
  floor**, for a block of four guest instructions.

  And of that, almost none is ours:

  | stage | host insns per block |
  | --- | --- |
  | decode + translate to IR | 4,621 |
  | Tier-1 wasm codegen | 19,127 |
  | **`Module::new` (validate + compile)** | **390,883** |
  | instantiate | 2,303 |

  A first attempt at this table said codegen was 54% of admission. It was wrong: the
  harness built a whole `WasmtimeBackend` — engine, store, anchor module — per block
  inside the measured region, and that dwarfed what it was trying to measure. Watch for
  setup inside the timed span; it made our own emitter look like the problem when it is
  4.6% of admission.
- So paying for more Cranelift does not work. `AERO_CT_PROMOTE=1000` measures 230.5,
  127.4 and 111.2 against 132.7, 105.4 and 114.0 across three windows — worse in two of
  three. The compile is paid immediately and in full; the better code repays slowly.

That is the shape of the wall. Coverage and code quality are both rationed by a
compiler that costs ~95us per four guest instructions, and the tier is already at the
economic optimum of that trade — which is why the thresholds all sweep jagged and none
of them beat the defaults.

`AERO_JIT_JITDUMP=1` (needs `--features aero-jit-x86/wasmtime-jitdump`) makes Wasmtime
emit a jitdump for `perf inject --jit`; the intent was to annotate inside generated code
and find out what the ~72 host instructions per guest instruction of compiled code are
spent on. Injection runs but the generated-code symbols do not survive it, so that
question is still open, and it is the one worth answering next: it is the only remaining
term big enough to matter that has never been broken down.

### The break-even, and why every knob is already at its optimum

A block admitted costs ~310,000–417,000 host instructions. Executing one saves roughly
`4.28 x (1250 - 72)` ≈ 5,000 against interpreting it. So **a block must run about 60–80
times before it has paid for its own compilation**.

That number explains the shape of every sweep in this file. Blocks here average 166
executions (6,106,494 executions over 36,848 blocks), comfortably past break-even — so
admitting aggressively is right, and raising the bar loses badly. On the whole session
(20M→200M), `AERO_CT_HOT_THRESHOLD` reads 352.4 at 4, 586.6 at 16 and 543.3 at 64. A
block that is not admitted is not merely uncompiled; its instructions are interpreted at
~1250 host instructions each *and* it breaks the chain around it.

Lowering the bar does not work either, for the same reason in reverse — the tail is
where blocks fall short of 60 executions — and `AERO_CT_PROMOTE` is bounded by the same
arithmetic one tier up.

So: admission threshold 4, promote threshold 2000, batch 16, cache 65,536. Every one of
them sits at a local optimum of the same trade, and the trade is set by a compiler that
charges ~310,000 host instructions per four guest instructions of code. That is the
wall, stated as precisely as this project can state it.

---

## Compiling off the CPU thread: the win is real, the determinism cost is why it is opt-in

Admission is ~417,000 host instructions a block and 94% of it is `Module::new`, and
until now the guest waited through every one of them. `Engine` is `Send + Sync` and
compiles concurrently and `Module` is `Send`, so that work can move to worker threads
while the guest keeps running; only instantiation (2,303 instructions) has to stay on
the CPU thread, because `Store` is not `Send`. `AERO_CT_ASYNC=1` turns it on.

**The metric this file is built on cannot see the win**, which is the first thing worth
saying. Host instructions per guest instruction counts the whole process, and moving
work between cores does not change it. What changes is how much of it is on the thread
the guest is waiting for, which is what throughput depends on. Sampling by *thread*
(`SORT=comm`) separates them:

| window | sync, CPU thread | async, CPU thread | async, workers |
| --- | --- | --- | --- |
| 80M→200M | 132.9 | 111.0 | 18.0 |
| 20M→200M | 353.7 | 241.6 | 119.0 |

So a third of the whole session's critical-path work is compilation, and it can be
taken off the critical path.

**And it is off by default, because it makes the machine non-reproducible.** Three runs
of the same 60M-instruction workload: 88.428G, 86.570G, 88.603G host instructions —
2.3% apart. Synchronous: 87.541G and 87.546G, five significant figures. When a module
lands depends on thread scheduling, a block becoming runnable earlier or later moves
where an interrupt falls, and the guest diverges. Every oracle here checks state at an
exact instruction count, and the whole measurement method differences two runs — one
async attempt read *negative* 188.4 for a window because the two budgets took different
paths.

A deterministic variant was built and rejected on measurement. Instead of installing a
module when a thread finishes it, install it at a fixed point in *guest* time: the
result is reproducible to 0.001%, and measures **269.6 against 132.7**. The deadline
almost always arrives before the compile has finished, so the guest waits anyway, and
the blocks land later than they would have. Both halves of the trade, none of the gain.

Three clock choices failed before that one worked at all, all the same shape:
- guest instructions *executed by the tier* — stays at zero until something is
  installed, so nothing ever becomes due, so nothing is ever installed;
- offers — go quiet once coverage is high, stalling the clock exactly when modules are
  waiting for it (fine at 30M, then 649.3 in the steady window);
- chains — only reached once there is a block to enter a chain at, so nothing starts.

The clock has to advance whether or not anything is compiled *and* whether or not new
blocks are being offered. Driving it from both the offer path and the batch loop does.

One more trap: `poll_admissions` sits on the offer path, which runs millions of times a
session, and scanning the in-flight map on each call — even only to find what was due —
cost 54% of the whole run on its own.

---

## A compiled-module cache: 42% off the warm-up, 32% off the whole session

The wall above is compile cost — ~417,000 host instructions a block, 94% of it inside
`Module::new`. There is one thing to do about it that does not involve replacing the
compiler: **stop paying it twice.** The guest image and the checkpoint are the same
every run, so the same blocks are compiled from scratch every time.

Wasmtime can serialize a compiled module and load it back. Measured the same way as
everything else (`tests/admission_insns.rs`, differencing two block counts):

| | host insns per block |
| --- | --- |
| `Module::new` | 390,883 |
| serialize + deserialize | **4,560** |

About 86x cheaper, so admission falls from ~417,000 to roughly 28,000 — the decode,
the IR and our own codegen, which are 24,000 of it and still paid.

`AERO_CT_CACHE_DIR=<dir>` turns it on. The key is the BLAKE3 hash of the generated
wasm, so a hit can only be bytes that would have produced exactly this module; a miss,
a truncated file or an incompatible engine all end the same way, by compiling. Modules
are written to a temporary name and renamed, so a run that dies part way through cannot
leave a half-written module for the next one.

| window | no cache | warm cache | |
| --- | --- | --- | --- |
| warm-up (20M→80M) | 790.3 | **461.7** | −42% |
| whole session (20M→200M) | 352.0 | **239.4** | −32% |
| steady (80M→200M) | 132.8 | **128.2** | −3.5% |

The shape is what it should be: nearly all of the gain is in the window where blocks are
being admitted, and almost none in the window where they already have been. On this
workload the cache holds 3,789 modules and 389MB, and a second run reuses 100% of them.

**It stays deterministic**, which is what separates it from compiling on worker threads
(above): two runs of the same 60M workload give 44.4323G and 44.4296G host instructions,
0.006% apart, against 2.3% for async. Nothing about *when* a block becomes runnable
changes — only how long it took to prepare it.

It is off unless a directory is named, deliberately: it writes hundreds of megabytes,
and `Module::deserialize` is `unsafe` — it must never be pointed at data another program
could have written. Both of those are the operator's decision, not a default.

---

## What compiled code actually spends its instructions on

The `[JIT]` bucket is ~72 host instructions per guest instruction and had never been
broken down. Whole-machine A/B cannot do it — any codegen change moves block boundaries
and the guest takes a different path — so `tests/compiled_code_cost.rs` measures it
where the guest cannot move: a block of `AERO_BODY` identical guest instructions, run as
a chain of `AERO_LINKS` blocks, differenced two ways.

| body | compiler | host insns/block | per guest insn |
| --- | --- | --- | --- |
| 4 ALU | Winch | 597 | 119.4 |
| 4 ALU | Cranelift | 272 | 54.4 |
| 32 ALU | Winch | 1493 | 45.2 |
| 32 ALU | Cranelift | 328 | **9.9** |

Read as a line, Cranelift is **~275 host instructions fixed per block execution plus
~2.0 per ALU guest instruction**. Two things follow.

**The per-instruction quality is already desktop-class.** Two host instructions for an
`add eax, imm` with its flags is what a good JIT produces; at a 32-instruction body the
whole thing is 9.9 per guest instruction. Nothing about replacing the code generator
would improve that much.

**The cost is the fixed per-block term, and blocks here average 4.28 guest
instructions.** 275 + 2 x 4.28 = 283 per block, or ~66 per guest instruction — which is
the 72 the whole-machine profile measures. That is the whole story of the `[JIT]`
bucket.

**And it is not the engine crossing.** Running the same block through the module's own
chaining dispatcher instead of returning to the host measures 294 against 283 (body 4)
and 406 against 339 (body 32) — no cheaper, slightly worse. So the fixed cost lives in
the block prologue and epilogue and the dispatcher, which is *this project's* codegen,
not Wasmtime's compiler. (In-wasm chaining still wins in the real machine, because there
the host path also carries the tier's chain-step lookup, which this harness stubs out.)

That reverses the recommendation this file gave earlier. The next thing to look at is
not a native backend: it is what a block does on entry and exit for four guest
instructions of work.

### Superblocks, a fourth time, and why the answer is still no

If the fixed cost is per block, longer blocks amortise it — which is the superblock
argument, and the measured slope says the prize is large. It was tried again anyway,
this time with a **warm module cache**, which removes the admission cost that sank it
before: **377.9 against 239.4 over a whole session.** Still a loss.

The reason is the one already recorded: fall-through fusion makes blocks bigger without
making the *executed* run longer, because the loop back-edge fires the side exit almost
every iteration. Longer blocks are the right target; decoding past the fall-through is
not the way to get them on this workload.

### Where the fixed per-block cost actually is

Profiling the harness at a one-guest-instruction body, with `AERO_JIT_PERFMAP=1` so
generated code resolves, splits the ~277 host instructions a block execution costs:

| host path | share | | in-wasm chained path | share |
| --- | --- | --- | --- | --- |
| `execute_chain` | 35.6% | | chaining dispatcher | 44.8% |
| `array_to_wasm_trampoline` | 23.3% | | the block | 26.0% |
| the block | 24.4% | | `dispatch_one` | 19.3% |
| trap thread state | 4.4% | | | |
| host lookup closure | 3.3% | | | |

So on the host path **two thirds of a block execution is the crossing**, and the block
itself is about 70 host instructions. And the chaining dispatcher costs very nearly what
the crossing did — which is why chaining measures flat *in this harness*, while still
winning in the real machine, where the host path additionally carries the tier's
chain-step lookup that this harness stubs out.

The dispatcher's cost is not its memory traffic. Hoisting its two loop-invariant limits
and both running totals into locals — six fewer loads a step — measures **294 against
294**, to the instruction. It is the two nested wasm calls per block. That change has now
been written twice and reverted twice; the second time with a measurement that could
actually see it, which is the useful part.

**The conclusion for the whole effort:** a block execution costs ~190–280 host
instructions of fixed overhead whichever way it is entered, against ~2.0 for a guest ALU
instruction, and blocks here average 4.28 instructions. Nothing about instruction
selection or the code generator moves that. Only fewer, longer block *executions* would,
and the four superblock attempts say this workload's control flow does not offer them.

---

## The number that settles the direction: KVM is on this machine, and it is 3,693 MIPS

Everything above is about making a software translator faster. The floor that analysis
arrives at is ~45 host instructions per guest instruction — roughly 150 MIPS — because a
block execution costs ~190 host instructions however it is entered and blocks average
4.28 instructions. Against a target near 500 MIPS that is not a tuning gap.

This machine has `/dev/kvm` (AMD EPYC 7502P, `kvm_amd` loaded, 64 cores). Measured with
`scratchpad/kvmspeed.c` — a real-mode loop with a known instruction count, so guest
instructions are exact:

```
589,818,001 guest instructions in 0.160 s = 3,693 MIPS
```

Reproducible across runs, and this at load average 21–35. That is **~50x the compiled
tier** and about seven times the target. It is a pure compute loop with no VM exits, so
it is an upper bound — but the order of magnitude is not in doubt, and it is the reason
every desktop VM uses hardware virtualisation rather than translation.

The other half of the picture is what device work costs, since Aero's device model would
run on each exit to userspace. `scratchpad/kvmexit.c` measures a port-I/O exit at
**~5.1us, about 200,000 exits a second**. So a KVM backend's design constraint is the
exit budget, not instruction throughput: an idle Windows desktop doing tens of thousands
of device accesses a second would spend a few percent there, and anything that turns a
hot guest loop into per-access exits would not work at all.

### What this means for the two halves of the goal

They cannot be the same code path. A browser has no hardware virtualisation, so the
browser tier stays exactly what it is — the Tier-1 WASM JIT, which is the only thing that
can run there. Desktop-class speed on native means KVM, which cannot run in a browser.
The architecture the goal implies is therefore **both**: KVM acceleration on native, and
this compiled tier in the browser, sharing the device model, the memory bus and the
snapshot format that already exist.

That is a large build — vCPU setup and CPUID, memory slots, restoring an architectural
snapshot exactly, routing PIO/MMIO exits into the existing buses, interrupt injection and
the choice between an in-kernel and a userspace irqchip, and TSC and timer handling. It
is QEMU-scale work and not something to half-do: a KVM backend that is subtly wrong
produces a guest that dies unpredictably, and none of the oracles this project relies on
(per-block replay, state comparison against Tier-0) apply to it.

The measurements above are here so that decision can be made on numbers: 3,693 MIPS
against a 150 MIPS ceiling for the alternative, and a 5.1us exit to design the device
path around.

---

## Pricing a guest memory access, and a default that had expired

With a harness that prices a block where the guest cannot diverge
(`tests/compiled_code_cost.rs`), the marginal cost of a guest instruction under
Cranelift is:

| | host insns per guest instruction |
| --- | --- |
| ALU (`add eax, imm`) | 2.0 |
| ALU with eager flags (`AERO_JIT_NO_LAZY_FLAGS=1`) | 59.5 |
| memory read, inlined TLB probe | 43.6 |
| memory read, shared load helper | 90.0 |

Three things worth keeping from that table.

**Lazy flags are worth 30x, not the ~1% recorded elsewhere here.** The small end-to-end
figure is because `eliminate_dead_flags` already removes most flag writes; what lazy
flags save on top is only what survives it. But the mechanism itself is enormous, so
neither should be removed.

**A memory access is the most expensive ordinary thing a block does** — twenty times an
ALU instruction — which makes the memory path, not instruction selection, the place to
look inside compiled code.

**And the shared load helper costs more than twice what inlining the probe does**, because
it is a wasm call per access. It was chosen when that trade ran the other way: inlining
emits ten times the bytes, so it costs more to compile, and the two used to cancel. They
no longer do, and `AERO_CT_CACHE_DIR` is why — the compile is paid once ever.

| window | shared helper | inlined probe |
| --- | --- | --- |
| 80M→200M | 132.8 | **129.0** |
| 140M→200M | 105.4 | **101.1** |
| 200M→300M | 114.0 | 114.6 |
| whole session (no cache) | 352.0 | 359.8 |
| whole session (warm cache) | 239.3 | **236.1** |

So it is now inlined on native and still the shared helper on wasm32, where there is no
module cache, compilation is on the main thread, and a tenth of the emitted bytes is
worth more than a few percent of steady state. `AERO_CT_SHARED_MEM=1` restores the old
native behaviour.

### What did not work, and what that says

Three attempts to make the probe itself cheaper measured **exactly nothing**:

- index arithmetic in i32 instead of i64-then-wrap, multiply replaced by a shift, and
  the constant `TLB_OFFSET` folded into the load's `memarg`: 43.7 against 43.6;
- computing the entry address once instead of twice: no change (Cranelift was already
  CSE-ing it);
- folding the permission test into the tag test so the fast path has one branch instead
  of two: 43.7 against 43.7.

The TLB is genuinely hit while this is measured — one `mmu_translate` call across
320,000 accesses — so this is the hit path, not a disguised miss.

A fourth followed: inlining the `br_table` into the chaining loop so a step is one wasm
call instead of two measured **300 host instructions a block against 282** — removing a
call made it *slower*, presumably because the loop body grows and is scheduled worse.

The lesson is the same one the dispatcher gave earlier: **Cranelift already removes the
kind of waste that is visible in the emitted wasm.** A block of four `add eax, imm`
emits about twenty wasm instructions each and costs 2.0 host instructions each. Reading
the generated wasm and finding it verbose says nothing about what it costs. Only the
first and third of the changes above were kept, and only because they emit fewer bytes;
the branch fold was reverted, since it bought a reader the obligation to check that a
speculative read of another page's TLB entry is safe, in exchange for nothing.

### In-wasm execution counting, so admission-time chaining could work

The chain map holds only *promoted* blocks because a block entered inside wasm never
reaches the host's chain-step closure, and that closure is what counts executions and
queues a block for Cranelift — so chaining to a block hides it from promotion, and
admission-time chaining measured 319.6 against 132.8 for exactly that reason.

The fix is obvious enough: count in the map entry, and have the dispatcher append a RIP
to a small ring when it crosses the threshold, for the host to drain. Adding memory
operations to the dispatcher had already measured free, so the counting should have been
free too. **It was, and the mechanism worked exactly as intended:**

- in-wasm chaining rose from 20.1% of block entries to **32.2%**;
- promotions rose from 237 to **265** — up, not down, so the starvation really was fixed.

**And the run collapsed: 498.3 against 128.9 (80M→200M), 431.6 against 100.9 later.**

The reason is the map, not the counter. Restricted to ~240 promoted blocks, a
2,048-entry direct-mapped table is empty enough to keep them all. Feed it all 36,832
admitted blocks and it thrashes: the promoted entries — the ones that route a hot block
to its Cranelift copy — are evicted by cold blocks that will run twice. Widening the
entry to 32 bytes to hold the counter cost **1.6% on its own** even with the feature
switched off, from the halved cache density.

Reverted. A bigger map would trade against exactly the locality that makes the probe
cheap, and the entry is on the hot path of every chain step.

**Watch out when reverting a change of this size.** The revert left a single stray
`write_unaligned(e.add(16), 0)` in `chain_map_insert` — harmless at 32-byte entries,
and at 16 bytes it writes into the *next* entry's RIP. Everything looked right
afterwards: the in-wasm chain rate was 20.0% and promotions 237, both exactly baseline,
and the late window measured 102.5 against 100.9. Only the 80M→200M window, at 269.4
against 128.8, said anything was wrong. Diffing the working tree against the patch taken
before the change found it in seconds; guessing would not have.

### The memcpy in the steady-state profile is compilation, not execution

`__memmove_avx_unaligned_erms` shows up at ~4.0 host instructions per guest instruction
in the steady window, which looks like a copy on a per-instruction path and was flagged
here three times without being chased. It is not: rebuilt with
`-C force-frame-pointers=yes` and profiled with `--call-graph fp`, its callers resolve
to

```
Machine::run_slice_inner -> tier0 seam -> CompiledTier::compile
  -> add_dispatched_blocks -> compile_module -> winch_codegen / wasmparser
```

— Wasmtime compiling modules. So it is part of admission, already characterised and
already addressed by `AERO_CT_CACHE_DIR`, not a separate lever.

Worth noting for next time: `--call-graph dwarf` left 3.74 of the 4.77 points
unattributed and would have supported almost any guess. A four-minute rebuild with frame
pointers answered it exactly.

### Skipping the chain accounting for blocks that cannot chain

Only a promoted block's module carries the chaining dispatcher, so four in five block
entries pay for accounting they can never use: the host zeroes `chain_retired` and
`chain_blocks` before every call and reads one back after.

Gating that on whether the module chains — which needs a `chains` flag carried on the
block reference, because inferring it from `is_optimized` would be silently wrong the
moment anything other than a promoted module is compiled with chaining — measured
**129.2 against 128.8, and 101.2 against 100.9**. Very slightly worse: the branch costs
what the two stores and the read cost.

Reverted, including the flag through five trait methods. The bookkeeping is unconditional
again, which is also the version that cannot be wrong.

---

## The inline TLB was caching half a permission, and it cost 10%

Chasing ~1% items turned up a 10% one. The route to it was measuring a rate rather than
guessing at code.

The MMU miss path is ~10.2 host instructions per guest instruction — `jit_translate`
4.9, `Mmu::translate` 2.8, `TlbSet::lookup` 2.5 — which is only worth attacking if the
inline TLB misses often. It does: **0.069 misses per guest instruction**, one every 14.5
instructions, at ~148 host instructions each.

Two obvious explanations, both eliminated by measurement:

- **Capacity.** Rebuilding with `JIT_TLB_ENTRIES` at 1024 and 4096 gives 0.068 and
  0.068. Sixteen times the table removes 2% of the misses.
- **Invalidation.** 778 full flushes across 26M guest instructions, one per 33,566. At
  256 entries refilled each, that is at most 11% of the misses.

The cause was in the tier's translator, and its comment said so plainly:

```rust
// Cache only the permission actually exercised.
flags |= if write { TLB_FLAG_WRITE } else { TLB_FLAG_READ };
```

So a page written and then read misses on the read, and re-translates. Writing and then
reading is what a stack does, and what every buffer does.

Half of that is fixable with nothing given up: **x86 has no write-only page**, so a
mapping this translator has just let a write through is by construction readable, and an
entry filled by a write can carry `READ` as well. The converse stays as it was — a read
says nothing about whether a page is writable, and a write to a read-only page has to
miss so that it can fault.

One line. **Misses fell from 1,808,903 to 97,737, a factor of 18.5** (0.069 to 0.004 per
guest instruction), and:

| window | before | after | |
| --- | --- | --- | --- |
| 80M→200M | 128.9 | **116.0** | −10.0% |
| 140M→200M | 101.0 | **89.2** | −11.7% |
| 200M→300M | 114.0 | **102.4** | −10.2% |

Consistent in sign and size across three independent windows, which is the bar for
believing anything here. Guest state still matches the interpreter at 100K/600K/1M/2M
and the per-block verifier is clean over 3M.

`AERO_CT_TLB` reports the miss and flush counts, which is what made this findable at all
— seven codegen changes before it measured nothing, because they were guesses at where
the cost was rather than measurements of it.

### Three follow-ups to the TLB fix, all measured, none taken

**The other half of the permission cache must stay as it is.** A read-filled entry still
cannot carry `WRITE`, and not only because a read says nothing about writability: the MMU
sets the accessed and dirty bits as it translates (`aero-mmu`, `if !entry.dirty()`), so
granting write on the strength of the page tables alone would let the first write through
the inline fast path without ever setting `D` — which the guest OS believes. The
asymmetry is load-bearing.

**The TLB is the right size, re-checked now that misses are rare.** 1024 and 4096 entries
leave the miss rate at 0.068 against 256's 0.069; 64 and 128 measure 116.7 against 116.0.

**The outer-loop quantum looks better at 16,384 and should not be moved.** It measures
113.6 / 94.4 / 79.5 / 91.9 against 4,096's 116.0 / 97.0 / 89.2 / 102.4 across four
windows — better in all four, and the oracles pass. But its immediate neighbours
*collapse*: 8,192 gives 707.0 in the 140M→200M window and 32,768 gives 629.7 in the same
one. A setting whose neighbours on both sides are catastrophic is a lucky spot, not an
optimum, and the same rule was already applied to `AERO_CT_PROMOTE` and
`AERO_CT_MAX_BLOCKS`. Left at 4,096.

**And the device polls cannot be bounded by skipping them.** `AERO_NO_POLL=1` measures
393.7 against 116.0 — a guest whose storage completions never arrive spins waiting for
them, so that is not an upper bound on what cheaper idle polls would be worth, just a
different workload. Combined with the earlier every-eighth-boundary result (153.4 against
150.1), there is no evidence of anything to win here.

---

## The marginal opcodes were off for a reason that expired

`AERO_CT_BAILS=1` ranks what Tier-1 refuses by *wanted executions*. At 116.0 the wall was
290,092, and its top entries were instructions the decoder already understands but that
are switched off:

| bytes | what | wanted executions |
| --- | --- | --- |
| `48 0f c3` | `MOVNTI` | 35,608 |
| `0f 0d …` | `PREFETCH*` | ~27,400 |
| `f0 48 0f b1` | `LOCK CMPXCHG` | ~14,700 |
| `0f bc` | `BSF` | ~17,400 |

`PREFETCH*`, `MOVNTI` and the register-offset bit tests sit behind one gate, and it was
closed on a measurement: a 3% loss, because they let ~1,700 more blocks compile while
the wall barely moved — fixing the first instruction a block cannot translate often just
moves the wall to a later one in the same block, and the admission cost is paid either
way.

That was a judgement about admission economics, and admission is not what it was. A
compiled module can now be kept between runs, and the tier is two and a half times
faster than when the gate was closed, so the same admission cost weighs differently
against the same coverage. Re-measured:

| window | gate closed | gate open |
| --- | --- | --- |
| 80M→200M | 116.0 | **105.2** |
| 100M→160M | 97.0 | **92.5** |
| 140M→200M | 89.2 | **89.0** |
| 200M→300M | 102.4 | **101.7** |

Better in all four, by 9.3% in the largest. The wall falls from 290,092 to 195,698 and
compiled coverage rises 2.9%. **On by default now**; `AERO_X86_NO_HINT_OPS=1` goes back.

The read-modify-writes next door stay off, and it is not close: **672.1** with
`AERO_X86_RMW_OPS=1`, and 303.5 with both groups on.

Correctness: guest state matches the interpreter at 100K/600K/1M/2M and the per-block
verifier is clean over 3M. `AERO_CT_VERIFY_ALL` reports 17 mismatches against a baseline
of **25** — that mode replays store-bearing blocks and double-applies their writes, so
its mismatches are its own; the count going *down* is the check that matters.

**The general lesson, and it has now paid twice in a row:** a default chosen by
measurement is only as current as the economics it was measured against. The shared
memory helper and this gate were both closed for admission cost, and both flipped once a
module could be compiled once and kept. It is worth re-running the old A/Bs after
anything that changes what compilation costs.

### The rest of the gate sweep, at the 105.2 baseline

Having found two expired defaults, the sensible thing was to re-run the others rather
than guess which. Steady window (80M→200M) against 105.2:

| gate | result |
| --- | --- |
| `AERO_JIT_STORE_FP=1` (stores on the single fast path) | 111.3, and 88.4 against 89.0 in the late window — mixed, left off |
| `AERO_CT_MMIO_VIA_BUS=1` | 110.7 |
| `AERO_CT_ASSISTS=1` (mid-block assist) | 276.5 |
| `AERO_CT_RESUME=1` | 233.7 |
| `AERO_X86_RMW_OPS=1` | 672.1 |

So the space is swept: of everything switched off in the tier and the decoder, two had
expired — the shared memory helper and the hint opcodes — and the rest are still right.
The store fast path is the interesting near-miss, better in one window and worse in the
other, which is exactly the shape the measurement floor produces; it stays off because
"better in all the windows tried" is the bar.

### What is left on the wall

After the hint opcodes, `AERO_CT_BAILS=1` ranks the remainder at 195,698 wanted
executions:

| bytes | what | wanted |
| --- | --- | --- |
| `0f bc` | `BSF` | ~17,400 |
| `f0 48 0f b1` | `LOCK CMPXCHG` | ~14,700 |
| `fa f0 48 0f ba` | `CLI` + `LOCK BTS` | ~7,000 |
| `f2 ae` | `REPNE SCASB` | ~3,000 |

`BSF`/`BSR` is the one worth building: wasm has `ctz` and `clz` natively, and Tier-0's
semantics are narrow enough to mirror exactly — only `ZF` is written, and the
destination is left *unchanged* when the source is zero (`exec_bsf_bsr` in
`ops_alu.rs`), which the IR can express with `Select` and a single
`WriteReg { reg: Flag(ZF) }`. The rest of the wall is `LOCK`-prefixed read-modify-writes,
which measured 672.1 as a group.

### `BSF`/`BSR`: built, correct, and a loss

The largest translatable entry left on the wall — ~17,400 wanted executions — and the
one place where wasm gives the operation for free, since it has `ctz` and `clz`. Built
across the decoder (`0F BC`/`0F BD`), a new `IrInst::BitScan`, and codegen.

It works: the wall falls from 195,698 to 174,273 and compiled coverage rises. It is
**worse in every window tried**:

| window | off | on |
| --- | --- | --- |
| 80M→200M | 105.2 | 277.7 |
| 100M→160M | 92.5 | 322.3 |
| 140M→200M | 89.0 | 107.0 |
| 200M→300M | 102.4 | 104.1 |

Which is the reason already written above the hint opcodes, only landing the other way:
fixing the first instruction a block cannot translate usually moves the wall to a later
one in the same block, so the extra blocks are admitted — and paid for — without becoming
much more of the running. The hint opcodes happened to clear whole blocks; `BSF` mostly
does not.

Kept behind `AERO_X86_BITSCAN=1` because it is correct and cheap to re-test if the
economics move again. `tests/bitscan_equivalence.rs` pins the two things that are easy to
get wrong and silent when wrong:

- **a zero source must leave the whole destination register alone.** Tier-0's
  `exec_bsf_bsr` sets `ZF` and returns *without writing*; writing the old value back at
  a 32-bit operand size would clear the upper half of the register, because that is what
  a 32-bit write does. The translation therefore merges at 64 bits rather than writing at
  the operand width.
- **the undefined flags must be left as Tier-0 leaves them** — untouched, not set to any
  particular value — so the IR declares `FlagSet::ZF` alone.

What remains on the wall after it is `LOCK`-prefixed read-modify-writes, measured at
672.1 as a group. There is nothing left there worth taking.

### Trimming the lazy-flag spill

At 105.2 the `[JIT]` bucket is 61.6% of the run (64.8 against the emulator's 40.0),
having overtaken it as the emulator dropped 59.1 → 40.0 while generated code fell only
71.9 → 64.8. So the per-block cost inside wasm is where what is left lives.

Every block's epilogue writes the lazy-flag descriptor — a header and three operands —
unconditionally. A block that never *touched* it (neither loaded what the previous block
owed nor deferred an operation of its own) is storing three zeros nobody reads. Skipping
those three is worth **12 host instructions of the 384** a one-instruction load block
costs, and end to end:

| window | before | after |
| --- | --- | --- |
| 80M→200M | 105.2 | **104.9** |
| 100M→160M | 92.5 | **92.3** |
| 140M→200M | 89.0 | **88.8** |
| 200M→300M | 102.4 | **101.5** |

Better in all four, and small — most blocks *do* touch the descriptor, because they end
in a compare and a branch. On by default; `AERO_JIT_NO_TRIM_SPILL=1` goes back.

**The header store stays**, and that is the part worth writing down: it may have to
clear a record an earlier block left owed, and a stale header is read as work still due.
Only the operands are don't-care when the header says nothing is owed.

### Following unconditional jumps: the cleanest "longer blocks" attempt, and it still loses

The per-block cost model says a block boundary is ~283 host instructions and blocks
average 4.41 guest instructions, so fewer boundaries should be worth a lot. `AERO_CT_EDGES`
says **10.9% of executed chain steps enter a block that ends at an unconditional jump**
(against 8.6% of blocks *admitted* that way — worth weighting, since hot blocks are loops
that end in `jcc`).

An unconditional direct jump can be followed without any of the objections the superblock
attempts ran into: the edge is always taken, so there is no side exit, nothing
speculative is decoded, and no coverage is lost. Built in `discover_block_mode` (follow
the target instead of ending, refusing to jump backwards into what is already decoded)
and in `translate` (a `JmpRel` that is not the last instruction emits nothing, because
the instruction decoded after it *is* its target).

It does exactly what it should:

| | before | after |
| --- | --- | --- |
| blocks ending in `jmp` | 3,398 | 722 |
| executed block entries | 6,073,450 | **5,673,036** (−6.6%) |
| guest instructions per block | 4.41 | **4.79** |
| compiled coverage | 26.88M | **27.17M** (+1.1%) |

Oracles match at 100K/600K/1M/2M and the per-block verifier is clean over 3M. And it
measures **249.3 / 89.6 / 99.4 / 272.3 against 104.9 / 88.8 / 101.5 / 92.3** — worse in
three of four windows.

**That is the useful result.** Four superblock attempts could be blamed on side exits
firing, on speculation decoding code that never runs, or on coverage falling as bigger
blocks crowded a fixed cache. None of those apply here, and it still loses. So the
per-block cost model is not wrong about what a boundary costs — it is wrong to conclude
that removing boundaries by lengthening blocks recovers it. Whatever a longer block costs
— more bytes to compile, worse host instruction-cache locality, spanning two guest pages
— is at least as large as the boundary it removes.

The "longer blocks" family is now five for five. It should not be attempted a sixth time
without a mechanism that makes blocks longer *without* making them bigger.

### Two dead stores in the block epilogue

Both found by reading what the epilogue writes and asking who reads it. Both are small,
and both measure *cleanly* — better in all four windows by very similar amounts — which
is itself the point: neither moves a block boundary, and block boundaries are what move
the guest onto a different path and swamp everything under a few percent.

**The lazy-flag descriptor's three operands.** A block that never touched the descriptor
— neither loaded what the previous block owed nor deferred an operation of its own — is
storing three zeros nobody reads. The *header* still has to be written: it may have to
clear a record an earlier block left owed, and a stale header reads as work still due.
Worth 12 host instructions of the 384 a one-instruction load block costs, and
104.9 / 92.3 / 88.8 / 101.5 against 105.2 / 92.5 / 89.0 / 102.4.

**The `CpuState.rip` store.** The block already *returns* where the guest goes next, and
the chain loop uses that return value to find the next block. So the store is dead for a
block that runs to its end — as long as the caller writes the return value into
`CpuState.rip` when the chain ends, which it now does in both `execute_chain` and the
single-block `execute`. 104.3 / 91.8 / 88.4 / 100.9 against 104.9 / 92.3 / 88.8 / 101.5.

Two blocks must keep it, and the second was a real bug caught by
`jit_backend_exit_to_interpreter_uses_sentinel` rather than by reasoning:

- a block that can **leave part-way** — a bail returns the sentinel and the interpreter
  has to resume at the instruction that faulted, which only the store records;
- a block whose **terminator is `ExitToInterpreter`** — it runs to its end, so it has no
  early exit and the first test misses it, but it *also* returns the sentinel, and the
  sentinel is precisely what a RIP cannot be recovered from.

The first version of this shipped with only the first condition, and the machine-level
oracles did not catch it: `AERO_CT_STATS` was normal, the guest ran, and state matched at
every checkpoint. It took a unit test that runs one block and asserts `cpu.rip`.

**And the RFLAGS store, which is the biggest of the three.** `state_usage.rflags_written`
is a static over-approximation: it is set for any instruction that writes flags,
*including* one whose flags are deferred into the lazy descriptor and never materialised
into the RFLAGS local at all. Such a block leaves RFLAGS exactly as it found it, so
loading it, setting a bit that is architecturally already set, and storing it back is a
round trip to nowhere — and deferred flags are the common case, so this is most ALU
blocks. Tracking whether anything *actually* wrote the local and storing only then:

| window | before | after |
| --- | --- | --- |
| 80M→200M | 104.3 | **102.5** |
| 100M→160M | 91.8 | **90.2** |
| 140M→200M | 88.4 | **86.8** |
| 200M→300M | 100.9 | **99.1** |

Better in all four by 1.7–1.8%. `AERO_JIT_NO_TRIM_RFLAGS=1` goes back.

Two codegen-inspection tests asserted the older contract — "a block that writes flags
spills RFLAGS" — which predates lazy flags. They now assert what actually has to hold,
that the flags are handed on *by one mechanism or the other*, RFLAGS or the descriptor.
Worth reading as a warning: a test that pins a mechanism rather than a property will
fail the day the mechanism legitimately changes, and the temptation is to conclude the
change is wrong.

Together the three epilogue trims are worth about 2.7%: 105.2 → 102.5 in the 80M→200M
window, with the oracles matching at 100K/600K/1M/2M and the per-block verifier clean
over 3M throughout.

### The rule the epilogue trims taught: a dead store pays, a dead load does not

With the RFLAGS *store* gone, its *load* in the prologue looked dead too, and the
analysis can tell statically which flag writes are deferred. Removing it measured
**102.5 / 86.7 / 99.0 / 90.2 against 102.5 / 86.7 / 99.1 / 90.2** — nothing, in four
windows.

That is the whole pattern of this area in one line. A store to linear memory is
observable, so Cranelift has to keep it however obviously pointless it is; a load whose
result goes unused it simply deletes. Every dead *store* removed here was worth
0.4–1.8%, and the one dead *load* was worth nothing — as were, earlier, four attempts to
rearrange arithmetic and control flow that Cranelift had already optimised.

So when looking for work to remove from generated code: **enumerate the stores.** The
loads are already handled, and so is everything that is merely verbose.

---

## The outer loop was copying a 16.5KB MMU on every batch boundary

The steady-state symbol profile has a bucket that had never been chased: about **9.6% of
runtime in `__memmove_avx_unaligned_erms`**, 4.9 host instructions per guest instruction.
A warm module cache cut it to 3.5, so ~1.4 of it was compilation copying code buffers and
the remaining 3.5 was steady state. Nothing in the tier obviously copies memory per
instruction, so the first job was to find out what did.

Two attempts to attribute it failed and are worth recording, because both look like they
should work:

- **`perf --call-graph dwarf`** returned `memmove` with no callers at all. The stacks
  cross JIT-generated frames that have no unwind information, and the unwinder gives up
  below them.
- **`perf --call-graph lbr`** is not available on this machine.

What worked was an `LD_PRELOAD` interposer on `memmove`/`memcpy` that records
`__builtin_return_address(0)` per call site with a call count and a byte total, differenced
across two `--max-insts` budgets exactly like the instruction metric.

**One mistake there is worth more than the result.** The first pass differenced raw
return addresses across two runs, which under ASLR are in different address spaces: nothing
matched, every "delta" was simply the larger run's totals, and the top entry was reported
with a byte count identical to its own total. The tell was exactly that — a delta that
equals a total. Interposers must record their own load base (`dl_iterate_phdr`) and
difference by file offset.

Done properly, the answer was unambiguous. Seven call sites, all inlined into
`Machine::run_slice_inner`, all with the same call count — 37,251 across a 120M
instruction window, or one per ~3,200 guest instructions, which is one per outer-loop
batch boundary:

| bytes/call | what it was |
| --- | --- |
| 16,728 | moving an `Mmu` into the freshly built `PagingBus` |
| 8,258 ×2, 8,256 ×3, 8,192 | `Mmu::new()` building its TLB arrays from a template |

66KB per boundary. **2.5GB across the window, 21.2 bytes per guest instruction.**

The cause is a single line of intent in the batch loop. `PagingBus` owned its `Mmu`
inline, and an `Mmu` is 16,584 bytes — almost all of it software TLB. `Machine` keeps a
durable warm MMU across batches, so every boundary built a *fresh empty* `Mmu`, handed it
to a new `PagingBus`, and `std::mem::swap`ped the warm one in; at the end of the batch it
swapped back and dropped the placeholder. The swap moved the MMU, and constructing the
placeholder to swap against cost six more copies.

Neither copy is necessary, and the fix is in two parts.

**Box the MMU inside `PagingBus`.** The struct goes from ~8.3KB to 88 bytes, so it moves
freely into the wrapping bus, and the swap moves a pointer. This alone took the 16,728-byte
copy out: 102.5 → 99.0 steady, 86.7 → 83.0 late.

**Then stop building a placeholder at all.** With the MMU boxed, the placeholder is the
only reason `Mmu::new()` runs per boundary — and it is only needed because a swap requires
something to swap against. `Machine` now holds `Option<Box<Mmu>>` and *lends* the MMU to
the bus for the batch, taking it back at each exit. There is no placeholder, so the six
template copies go too: **99.0 → 97.2 steady, 83.0 → 81.2 late.**

`Option` rather than a plain `Box` on purpose. A batch exit that forgets to hand the MMU
back leaves `None`, and the next boundary starts on a fresh cold MMU — which is exactly
what a missed `swap` did before. That is a performance bug either way, and not worth a
panic.

Measured end to end, with the guest doing identical work (this changes no block boundary,
so it is one of the changes that measures cleanly):

| window | before | after |
| --- | --- | --- |
| steady (80M→200M) | 102.5 | **97.2** |
| late (140M→200M) | 86.7 | **81.2** |
| mid (100M→160M) | 86.8 | **85.0** |
| warm-up (20M→80M) | 729.6 | **721.7** |
| whole session (20M→200M) | 311.5 | **305.4** |
| steady, warm cache | 97.6 | **92.2** |
| whole session, warm cache | 191.7 | **185.5** |

and the interposer confirms the mechanism rather than just the outcome: **21.20 bytes of
`memmove` per guest instruction before, 0.99 after.**

`crates/aero-cpu-core/tests/paging_bus_is_small.rs` pins it. The cost was invisible in the
source — the loop reads as one struct construction and one swap — so a comment would not
have survived. A size assertion will.

### What this one says about the others

Every previous item in this document was found by reasoning about the tier: coverage,
block length, epilogue stores, chain structure. This one was in the machine's outer loop,
had been there the whole time, cost 3.4 host instructions per guest instruction — more
than the three epilogue trims put together — and no amount of thinking about the JIT would
have found it. The symbol profile named it in one line.

The lesson is narrow and practical: **profile the whole process, not the component under
study.** The `memmove` bucket was in every profile taken this whole time. It was skipped
because it did not look like it belonged to the tier, which was true, and irrelevant.

### Two things priced this session and not built

Both were sized before any code was written, which is the point.

**Keeping the inline TLB across address-space switches.** The tier takes 627,478 TLB
misses per 200M guest instructions, and 14,978 full flushes at 41.9 misses each accounts
for every one of them — so all misses are post-flush refills, and tagging the TLB by
address space would remove most of them. Priced with `AERO_CT_ALWAYS_FLUSH_TLB=1`, which
raises flushes 53x (796,407) and misses 6.7x (4.23M): **93.7 against 92.2**. That is 1.5
host instructions per guest instruction for 3.6M extra misses, so **a miss costs ~50 host
instructions and the entire baseline miss load is ~0.16 per guest instruction** — under
0.2%. A correctness-sensitive change to the TLB tagging, closed for the price of one run.

**Closing the coverage wall opcode by opcode.** `AERO_CT_WALL` is 588,757 wanted
executions and `AERO_CT_STEPPED` 527,269, i.e. about one interpreted instruction per
refusal, so the whole wall is ~3.9 host instructions per guest instruction — **~4% of
runtime, for all of it**. And it is flat: the top 16 byte-sequences are 27% of the total.
Grouped, `LOCK` RMW (`CMPXCHG`, `BTS`, `CMPXCHG16B`, `XADD`) is 59K wanted executions —
**~0.4%** — then `OUT` 18,495, `INVLPG` 12,649, `BSF` 13,386 (already built, already a
loss). No family is worth its implementation risk, before even counting that admitting
more blocks costs compile time.

### A `getenv` on the refusal path

`CompiledTier::record_hit_with` read `AERO_CT_BAILS` from the environment on every
*refused* offer — 588,757 times per run — instead of caching it in a `OnceLock` like every
other gate in the crate. `getenv` walks `environ` with `strncmp`, which is what
`__strncmp_avx2` was doing in the steady profile at 0.6 host instructions per guest
instruction.

Cached, it is worth **97.2 → 97.0 steady, 81.2 → 80.9 late, 85.0 → 84.7 mid, 92.2 → 92.0
warm-cache** — about 0.3%, consistent in direction across all four windows, and it cannot
move a block boundary so it measures cleanly. Smaller than the profile bucket suggested,
which means the bucket had other users; the change is right regardless, and it removes a
trap for anyone who later adds work to that path.

## Winch is 14x Cranelift, and the promotion cache is what lets you have more of it

Priced with `compiled_code_cost`, which measures where the guest cannot move — a body of
identical guest instructions in a chain of blocks, differencing two chain lengths and two
body lengths to separate the per-instruction cost from the per-block one:

| engine | per guest ALU op | per block entry |
| --- | --- | --- |
| Winch | **32.00** | 443 |
| Cranelift | **2.25** | 270 |

(Take care driving that harness: `AERO_OPT=0` still satisfies `is_some()`, so it selects
Cranelift. The Winch column needs the variable *unset*.)

Weighting by the machine's own split — `AERO_CT_WASM_CHAIN` says 56.9% of block entries
are in promoted modules — gives ~410 host instructions per block entry, and the machine
independently shows 17.6G host instructions over 43.66M block entries, or ~403. The model
holds. Since a block averages 4.38 guest instructions and an ALU op costs 2.25 on
Cranelift, **nearly all of compiled-tier time is per-block entry cost, not the guest's
work** — and about a third of the total is Winch being 14x worse than Cranelift on the
43.1% of entries it serves.

### Why the promotion threshold had never really been tested

The module cache covered `add_dispatched_cached` — **admission only**. `promote_blocks`
and `promote_blocks_replacing` never went through it, so Cranelift recompiled every
promotion on every run. That is why lowering `AERO_CT_PROMOTE` measured as a large loss
*even "with a warm cache"*: the cache did not cover the thing being increased. This is
[[aero-defaults-expire]] with a twist — not a default whose economics changed, but one
tested against a mechanism that never applied to it.

`promote_blocks_serialized` / `promote_blocks_precompiled` fix that, keyed `.opt.cwasm`
against admission's `.cwasm` because both engines compile the *same* wasm bytes and
`Module::deserialize` checks a version header, not which engine produced the body.
`tests/promotion_cache.rs` pins both the equivalence and the key separation.

**The cache alone is worth 5.7%** on the warm path: steady 92.0 → **86.8**, whole session
183.5 → **174.6**, late → **78.7**. The no-cache path is unchanged at 97.0, as it must be.

### And with it, the threshold is still 2000

Re-run properly, with a cache warmed separately for each setting:

| `AERO_CT_PROMOTE` | promoted blocks | in-wasm chain rate | steady |
| --- | --- | --- | --- |
| 2000 | 463 | 56.9% | **86.8** |
| 500 | 1,777 | 48.0% | 169.2 |
| 100 | 4,804 | 43.0% | 102.2 |

Promoting *more* blocks lowers the in-wasm chain rate, because in-wasm steps only stay
inside one module and more promotions spread the hot set across more of them. The
Winch→Cranelift gain is real but it is bought with host round trips, and the round trips
cost more.

The obvious follow-up — wait longer so each promoted module is bigger — is worse, and
jaggedly so. `AERO_CT_PROMOTE_STALL` (new, default 20,000) against the in-wasm rate:
200 → 46.9%, 2,000 → 25.9%, 6,000 → 37.2%, **20,000 → 56.9%**, 200,000 → 17.3%,
2,000,000 → 12.5%. Non-monotonic in both directions, which is the same signature as the
chain-map size sweep: **not a dial**. Waiting produces bigger modules but promotes later,
and the blocks carrying the run spend that time on Winch.

### Consolidation, re-tested now that its cost is cached — and why it cannot help

Gathering the promoted set into one module is the obvious way to raise the in-wasm chain
rate, and it was rejected earlier at **239.6**. That rejection was mostly its compile
bill: consolidation recompiles the *whole* promoted set every time it gathers, and until
`promote_blocks_replacing` went through the cache, every gather was paid again on every
run.

Cached, it is far closer — **89.3 steady against 86.8, 79.3 late against 78.7** — but
still a loss, and the interesting part is why it does not help. With consolidation on, the
in-wasm rate moves 56.9% → **57.1%**. Essentially nothing. So the 43% of block entries
that leave wasm are **not** steps between promoted blocks in different modules, which
consolidation would fix; they are steps into blocks that are **not promoted at all**.

That leaves only one shape that could work: promote more *and* keep them together. Tested,
and it does not:

| config | promoted | in-wasm rate | steady |
| --- | --- | --- | --- |
| T=2000, no consolidate | 463 | **56.9%** | **86.8** |
| T=2000, consolidate | 451 | 57.1% | 89.3 |
| T=500, consolidate | 1,807 | 51.0% | 182.7 |
| T=100, consolidate | 4,828 | 45.3% | — |

The in-wasm rate is *maximised* by the current setting, and every attempt to raise it
lowers it. The reason is that `promote_batch` is 256 while the set that matters is ~190
blocks: at a threshold of 2000 those blocks cross it earliest and land in the first module
**together**, and any lower threshold mixes them with less-hot blocks and splits them
across modules. The configuration works because of an ordering accident, and consolidation
cannot recreate it after the fact.

**Six configurations, and the shipping one is the best of them.** What would actually move
this is a promoted set chosen by hotness rather than by arrival order — but that is the
in-wasm execution counter, already built and measured at 498.3.

## Superblocks, on by default — the same measurement, reversed

Superblocks were measured at **199.0 against 184.4** and rejected. They are now **on by
default**, because the thing they trade against has changed twice: admission is cached,
promotion is cached, and the cost that actually dominates has been identified as per-block
*entry*, not codegen.

What they do, over a 200M-instruction run:

| | off | on |
| --- | --- | --- |
| block entries | 43.7M | **26.7M** |
| guest instructions per block | 4.38 | **6.80** |
| in-wasm chain rate | 56.9% | **67.9%** |
| coverage | 95.5% | 90.9% |

Fewer, longer blocks, bought with coverage: a longer span is likelier to contain an
instruction Tier-1 refuses, and one refusal refuses the whole block. That trade used to
lose. It now wins, because a block entry costs ~270 host instructions on Cranelift while a
guest ALU op costs 2.25 — so removing 17M block entries is worth more than the 4.6 points
of coverage it costs.

**With a warm module cache, six post-warm-up windows, six wins:**

| window | off | on |
| --- | --- | --- |
| warm-up (20M→80M) | 350.3 | 485.8 |
| steady (80M→200M) | 86.8 | **84.3** |
| mid (100M→160M) | 81.2 | **71.1** |
| late (140M→200M) | 78.7 | **67.8** |
| 200M→300M | 93.8 | **90.8** |
| 250M→350M | 82.6 | **75.2** |
| 500M→600M | 92.5 | **81.0** |

**Without a cache, three of four:** 250M→350M 84.8 → **76.0**, 500M→600M 92.7 → **81.2**,
late 80.9 → **71.9** — and 80M→200M goes the other way, 97.0 → 100.9, because that window
still has admission running through it.

Warm-up is the honest cost: **485.8 against 350.3**, and it is not compile time (the cache
was warm for both). It is coverage — early on, more of the running is code the tier has
refused. The crossover is around 1.4G guest instructions, which is a few seconds of use.
**`AERO_CT_NO_SUPERBLOCKS=1` turns them off**, which is the right setting for something
that only boots.

### What this says about the five earlier "longer blocks" losses

Superblocks were the fifth of five attempts in that family, all measured, all rejected. At
least this one was rejected for a reason that expired — it was paying Cranelift for every
promoted block on every run, and competing against a per-block cost nobody had priced. The
others (jump-following, fusion, block-limit changes) are worth re-running on the same
suspicion, with the deep windows rather than 80M→200M, which is still admission-heavy.

## Pricing a chain step by engine, which explains a rejection nobody had explained

`wasm_chain_step_insns` against `chain_link_insns` — same guest block body, one chaining
inside wasm and one returning to the host each step:

| | host instructions per step |
| --- | --- |
| in-wasm chain step, **Cranelift** | **136** |
| host round trip | 420 |
| in-wasm chain step, **Winch** | **526** |

Chaining is a 3x saving, and *only* in a promoted module. On Winch a chain step costs
**more than the host round trip it replaces**.

That is the explanation for admission-time chaining measuring 319.6. It was recorded as
thrashing a 2,048-entry direct-mapped table, and that may also have been true, but the
first-order reason is simpler: every step it converted from a host round trip into an
in-wasm step made that step 25% more expensive. The mechanism was never about the map.

It also warns about the harness: run `wasm_chain_step_insns` without `AERO_OPT=1` and it
measures Winch, which is an engine the machine never chains on. The test now says so.

### What the 32% that still leaves wasm actually is

With superblocks the in-wasm rate is 67.9%. Weighted, a block entry costs
`0.679×136 + 0.321×420` ≈ **227 host instructions**, over 6.80 guest instructions.

Raising that rate is worth roughly 284 host instructions per entry converted — about 12%
for a rate of 90%. Three routes are measured and all closed:

- **Promote more blocks.** Lowers the rate, because `promote_batch` is 256 and any lower
  threshold splits the ~190-block hot set across modules. 500 → 48.0%, 100 → 43.0%.
- **Consolidate the promoted set into one module.** Moves the rate 56.9% → 57.1%. The
  steps leaving wasm are into blocks that are **not promoted at all**, so gathering the
  promoted ones changes nothing.
- **Chain on Winch too.** 526 against 420 per step. A loss before any map is involved.

A shared `funcref` table imported by every module — so a dispatcher could `call_indirect`
across module boundaries instead of `br_table`ing within its own — is the obvious fourth
route, and it is not worth building: consolidation already ran the experiment it would
run, and a table cannot span two engines' stores anyway.

The binding constraint is therefore exact: **Winch code is 14x Cranelift, and there is no
cheap way to have more Cranelift code.** Promotion is now free on a warm run
(`.opt.cwasm`), so the cost is no longer compilation — it is that promoting more fragments
the modules that chaining depends on.

## Fragmentation was not the constraint — concentration is

The previous section said the in-wasm chain rate could not be raised because promoting
more blocks fragments the modules chaining depends on. That was measured, and it was the
wrong diagnosis. The fragmentation is real and it is now fixed, and fixing it changes
nothing.

**The fix.** A block joined `promote_queue` the instant its count hit the threshold, so
the queue was in *arrival* order and every entry looked equally hot when it joined. By the
time a 256-block group gathered they no longer were. Sorting the queue by what its blocks
have actually executed since joining — the count is in the chain cache — makes the hot set
land in one module by rule instead of by luck:

| `AERO_CT_PROMOTE` | in-wasm rate, arrival order | in-wasm rate, hotness order |
| --- | --- | --- |
| 2000 | 67.9% | 67.9% |
| 500 | 48.0% | **67.4%** |
| 100 | 43.0% | 38.1% |

At 500 that is 864 promoted blocks instead of 270, with the chain rate intact — exactly
the thing the earlier sweep could not have.

**And it still does not pay.** Four paired deep windows, hotness-ordered 500 against the
shipping 2000: 79.4/67.1, 79.8/80.0, 81.8/81.7, 72.5/67.7. Two worse, two even.

So the binding constraint is not module layout. It is that **execution is concentrated
enough that there is nothing left to promote**: the busiest ~190 blocks are 94% of chain
steps, the hot set is already promoted at threshold 2000, and the 594 blocks between 500
and 2000 are cold enough that promoting them buys no steps while adding emitted code —
which [[aero-code-footprint-matters]] says is not free.

That also re-reads the 32% of block entries still leaving wasm. They are not steps the
design *cannot* capture; they are steps into a long tail of genuinely cold blocks, each
executed too few times for any promotion policy to repay. The distribution, not the
mechanism, is the wall.

The ordering change is kept even though it is worth nothing at the default (67.9% either
way). The shipping threshold worked because arrival order happened to coincide with
hotness order at 2000, and a default that holds by accident is one bad tuning change away
from a 20-point regression nobody would connect to the cause.

## Where the 82 actually goes, and what that means for the target

I had started reasoning toward "the software TLB sets the floor" — if a guest load costs
43.6 host instructions and a third of guest instructions touch memory, the floor would be
~17 host instructions per guest instruction, about 410 MIPS, and the target would be
unreachable by construction. **That is wrong, and the machine says so.**

A/B in a deep window (400M→500M): inlined probe **81.7**, `AERO_CT_SHARED_MEM=1` **86.9**.
Inlining is worth 5.2 host instructions per guest instruction. Inlining took a load from
90.0 to 43.6, so 5.2 / 46.4 ≈ **0.11 loads per guest instruction** — not a third. The
memory path therefore costs ≈ 0.11 × 43.6 ≈ **4.9**, and ALU ≈ 0.89 × 2.25 ≈ **2.0**.

**Instruction bodies are ~7 of the 82 — about 8% of runtime.** Everything else is overhead:

| term | host instructions per guest instruction | how it was derived |
| --- | --- | --- |
| block entry | ~33 | 227 per entry ÷ 6.80 guest instructions per block |
| Winch bodies | ~10 | 32.1% of entries at 32 vs 2.25 per ALU op |
| instruction bodies | ~7 | the A/B above |
| Tier-0, device polls, batch boundaries | ~32 | the remainder |

So the guest's own work is a rounding error and the emulator is almost entirely overhead.
That is worth stating plainly because it points somewhere different from where the last
several sections were digging: **not codegen quality, not coverage, not the memory path —
block entry, at 227 host instructions for 6.8 guest instructions of work.**

### What would have to be true for 500 MIPS

The measured conversion on this host is roughly `MIPS ≈ 6900 / (host instructions per
guest instruction)`, which the numbers bear out (6900/70 ≈ 98). So 500 MIPS means **~14
host instructions per guest instruction**, against 82 today.

Instruction bodies alone are ~7, so it is not arithmetically excluded — but it requires
block entry, the Winch penalty, and the machine's own per-batch overhead to *together*
fall from ~75 to ~7. Nothing measured here suggests a route to that. The largest single
piece, the 136-instruction in-wasm chain step, is mostly the block's prologue and epilogue:
each block loads its guest registers out of linear memory and spills them back, and a chain
of blocks in one module does that repeatedly for registers that never left. Holding guest
registers in wasm locals *across* chained blocks — inlining the bodies into the dispatcher
rather than calling them — is the one untried structural change with a real target, and a
generous estimate puts it at 136 → ~40, worth about 13%.

That is the honest position: 13% is available from the biggest remaining item, and 5.9x is
required. The gap is structural, and KVM on this machine measures 3,693 MIPS.

## Every CpuState access was bounds-checked, and reserving the i32 range removes it

This is the largest single win of the effort, and it was found by disassembling rather
than reasoning. Wasmtime can write a jitdump (`--features wasmtime-jitdump`,
`AERO_JIT_JITDUMP=1`), `perf inject --jit` reconstructs it, and `perf annotate` then
disassembles generated code — which is the only way any of the preceding sections could
have been checked against what the CPU actually runs.

A promoted block of four guest ALU instructions compiled to **111 native instructions**.
The reason was visible immediately, repeated about nine times:

```
leaq 0x88(%rcx,%rdx), %r9    ; address = memory_base + cpu_ptr + field
cmpq %rax, %rdx              ; cpu_ptr against the memory length
cmovaq %r8, %r9              ; out of bounds -> NULL, so the access faults
movq (%r9), %r8              ; the actual load
```

Three instructions of bounds check per `CpuState` field. The Tier-1 ABI passes
`cpu_ptr`/`jit_ctx_ptr` as runtime `i32` parameters, so every field access is a *dynamic*
index, and with a 32MiB guard an `i32` can plainly escape the mapping — so Wasmtime is
obliged to check. The dispatcher had the same idiom.

**The fix is one constant.** Reserve the whole `i32` range as guard, and no `i32` index can
leave the mapping, so the proof obligation disappears and no check is emitted. The region
is already `PROT_NONE` with `MAP_NORESERVE`, so it costs address space and nothing else —
measured RSS is unchanged at 2.73GiB for a 2GiB guest.

| | before | after |
| --- | --- | --- |
| block, 4 guest ALU ops | 111 native instructions | **64** |
| dispatcher | 98 | **68** |
| bounds-check `cmov`s | present | **0** |
| per-block overhead (`compiled_code_cost`) | 268.0 | **157.7** |

Whole machine, A/B in one binary with a cache warmed per configuration
(`AERO_JIT_SMALL_GUARD=1` restores the 32MiB guard):

| window | 4GiB guard | 32MiB guard |
| --- | --- | --- |
| 140M→200M | **50.7** | 67.7 |
| 200M→300M | **70.2** | 90.8 |
| 300M→400M | **64.9** | 81.2 |
| 400M→500M | **65.7** | 81.9 |
| 500M→600M | **65.1** | 81.2 |

**Five windows, 20-25% every time.** Oracle matches at five checkpoints, verifier clean
over 12M, suites unchanged.

### What this says about the preceding sections

Several of them priced things in host instructions per guest instruction and then reasoned
about *why* the number was what it was — per-block entry cost, prologue and epilogue,
dispatcher work. None of that reasoning would have found this, because the cost was not in
any of those places: it was three instructions attached to every field access, invisible in
the wasm and invisible in the profile, which only says which function the time is in.

The lesson is the same one the `memmove` section records, one level down: **look at what
actually runs.** A jitdump is 20 minutes of setup and it should have been the first thing
done, not the last.

## Two wasm calls a step became one, and a decoder bug the fuzzer had never reached

Disassembling the chain step showed three functions, not two: the chaining loop called a
`br_table` function, which called the block. That middle function is 28 native
instructions — its own frame, its own stack-limit check, the jump table, a call and a
return — out of a ~94 instruction step.

Inlining it was measured at **300 against 282** and rejected. That was before the guard
covered the `i32` range, when every context access carried three instructions of bounds
check and the middle call was a small share of a much larger step. Re-run at the new
baseline it is a consistent win: **49.4/50.7, 49.1/50.4, 64.1/65.6, 63.5/65.0,
62.6/64.1** across five paired deep windows, about **2.4%**. (A sixth read 43.6 against
63.9; 300M→400M swings, and one window is not a result.) On by default;
`AERO_JIT_TWO_CALL_DISPATCH=1` restores the old shape.

That is the third rejection this session that reversed once the thing it was competing
against changed. The pattern is now reliable enough to state as a rule: **a rejected
optimisation is only rejected against the baseline it was measured on.**

### `48 F2 B8` — a REX prefix that should not have applied

Running the iced proptest repeatedly (it generates fresh cases each run, and persists
failures) turned up a real decoder bug that has nothing to do with any of this work:

```
48 F2 B8 00000000        REX.W, REPNE, MOV eAX, imm32
```

A `REX` prefix only has effect when it is the **last** prefix before the opcode. The `F2`
discards it, so this is `MOV EAX, imm32` and seven bytes long. Aero honoured the `REX.W`
regardless of order and decoded `MOV RAX, imm64` at eleven — which means it consumed four
bytes of the *next* instruction, and everything after it in the block is garbage.

The prefix loop kept a `REX` across subsequent prefixes; it now clears it whenever another
prefix follows. There was even a comment two lines below saying "REX must be the last
prefix", attached to something else. Real compilers never emit this ordering, which is why
it survived, and why the fuzzer found it rather than the machine.
`tests/tier1_decode_mode.rs::a_rex_followed_by_another_prefix_does_not_apply` pins both
directions — the same bytes reordered still decode as the 64-bit form.

The same runs surfaced a second, benign mismatch: with hint opcodes on by default, `0F 0D`
and `0F 18` decode to `InstKind::Nop` while iced names them `Reservednop` and `Prefetch*`.
Those are architecturally side-effect-free, so folding them into `Nop` is right; the test
now accepts that family, and only that family.

## The composition has inverted: host-side machine code is now 44%

With the tier three times faster than it was, the split has moved. Deep window
(500M→600M), by DSO:

| | host instructions per guest instruction | share |
| --- | --- | --- |
| `[JIT]` (compiled wasm) | 36.7 | 57.7% |
| `aero-machine` (host) | 28.2 | 44.3% |

Every section above this one was about the JIT. It is now barely half the cost, and the
next items are on the machine side.

### The guest-load path, priced properly at last

Three harness traps had to be cleared first, and all three had been quietly producing
nonsense:

- **`compiled_code_cost`'s load body reported a TLB *miss* every time** — `translates`
  was 3,216,000 for 800,000 loads. Its comment claimed it measured a hit.
- **The inline TLB is behind the `tier1-inline-tlb` cargo feature**, which the harness
  runs did not enable, so every access went through the host helper. That is the whole
  explanation for the miss count, and for the 490-per-load figure quoted earlier in this
  document. (`aero-machine` enables the feature, so the machine itself was always fine.)
- **`block_prologue` printed the helper path**, because `inline_tlb: true` alone is not
  enough — `shared_mem_helpers` defaults on and wins.

With the feature on and `AERO_INLINE_MEM=1`, `translates` drops to **1** and a guest load
costs **28.79 host instructions**, against 4.07 for an ALU op. The fast path disassembles
to ~24 instructions, six of which are stack spills forced by the slow path's liveness.

And it does not matter much: the machine A/B against `AERO_CT_SHARED_MEM=1` puts loads at
about **8.5% of guest instructions**, so the whole memory path is ~2.4 host instructions
per guest instruction of ~50. **The load path is not where the time is**, which is worth
knowing before optimising it — the earlier reasoning in this document assumed a third of
instructions touched memory and was wrong by a factor of four.

### PCI config writes: the mechanism found, the end-to-end effect not measurable

`PciConfigSpace::write_with_effects` was **3.0 host instructions per guest instruction** in
the deep window, plus 0.7 for MSI capability handling — about **6% of runtime**.
Instrumented: **3,103,926 config-space writes** over 580M guest instructions, one every
187, at roughly **555 host instructions each**, and concentrated on five consecutive dwords
— **0x40, 0x44, 0x48, 0x4c, 0x50 at ~620,780 writes each**, nothing else above 14. One
20-byte MSI capability block, rewritten 620,780 times.

`perf annotate` on the host binary named the cost: `pcmpeqb` against a `.rodata` constant,
`pmovmskb`, `cmpl $0xffff` and a 16-byte-stride loop. That is a `TypeId` comparison — a
`downcast_ref::<MsiCapability>()` — inside a walk of the capability list, and
`is_read_only_byte` was calling it **once per byte written**, alongside a virtual call per
capability for its offset.

Two changes, both semantics-preserving:

- **`read_only_mask(offset, size)`** computes the predicate for the whole write in one pass
  over the capability list instead of one pass per byte. The predicate reads capability
  state and the loop only writes `self.bytes`, so it is invariant across the loop.
  `write_with_effects` fell **3.0 → 2.4**.
- **Ranged capability syncs.** The pre-write sync exists so the read-modify-write sees
  device-managed bits, and the post-write read-back exists so the device sees the guest's
  write — both concern only the capability that owns those bytes, and `PciCapability::len`
  makes the overlap exact. The *final* sync stays whole, because a device may react to one
  capability by changing another. `write_with_effects` then fell below the profile's 0.5
  cutoff entirely.

**And the machine did not move**: 50.0/63.0/64.1 against 49.8/62.7/63.8 on the same three
windows. The symbol profile says the targeted cost went from ~3.7 to ~2.1, which should be
~2.5%, and the windows cannot resolve it. Kept anyway — strictly less work for the same
result, guarded by `read_only_mask_tests` — but recorded as **not measurable end to end**
rather than claimed as a win.

**Still open, and it is the bigger half:** why does a guest rewrite an MSI capability block
620,780 times? That is one per ~4,700 guest instructions and looks like a driver retrying,
which would make it a device-model behaviour bug worth more than the microseconds spent
servicing it. The five offsets in `pci_ahci.rs` that match exactly turn out to be inside
`mod tests`, so the writes really do arrive through `PciBus::io_write`.


## The verifier was comparing whole chains, and had been for a long time

`AERO_X86_FOLLOW_JMP` produced 4 `AERO_CT_VERIFY` mismatches where the default produced 0.
I spent two rounds treating that as a translation bug. It was the verifier.

`verify_block` runs the block with `run_chain(..., max_blocks = 1)` and then replays exactly
`retired` instructions under Tier-0. But `max_blocks` bounds the **host** loop only — the
module's own dispatcher chains *inside wasm* up to `chain_max_blocks` in the JIT context,
which this path never set, so it inherited whatever the last real chain left there.

Printing `retired` alongside the block's length settled it in one run:

```
AERO_CT_MISMATCH rip=0xfffff8000274511b insts=9 retired=1302 ...
AERO_CT_MISMATCH rip=0xfffff800026e0668 insts=6 retired=2164 ...
```

A nine-instruction block "verifying" 1302 instructions. Tier-0 was then replayed 1302
instructions — which is the whole-machine comparison, and that legitimately diverges on
interrupt timing, because a compiled chain runs to its end while Tier-0 checks between
instructions. Hence four mismatches with a shape that looked like a loop running one extra
iteration: only `rcx`/`rbp` differing, Tier-0 always ahead.

Bounding the in-wasm chain to one block in `verify_block` gives **0 mismatches over 30M**
for both the default and `AERO_X86_FOLLOW_JMP=1`.

This matters beyond the immediate result. Every "verifier clean" claim in this document
before now was a claim that *chains* agreed with Tier-0 replays, not that blocks did — a
weaker property, and one that also raised false alarms. It is now the intended one.

### Jump-following is clean now, and still off — for an honest reason

With the verifier fixed the correctness objection is gone. It is also worth nothing:
**50.2/50.1, 49.8/49.7, 61.4/63.0, 64.6/64.7, 63.3/63.2** across six paired deep windows.
The ~1-1.8% it measured earlier was against a baseline three times slower, where a saved
block boundary was a larger share of the total.

It does improve the structure — guest instructions per block **6.80 → 7.35**, in-wasm chain
rate **67.9% → 70.5%**. I turned it on, and it broke **12 tests** whose fixtures assume a
`jmp` ends a block. Twelve test rewrites for no measured throughput is the wrong trade, so
it went back to opt-in (`AERO_X86_FOLLOW_JMP=1`). Worth revisiting only if something later
makes block length pay again.

## Following calls: the biggest structural lever left, and it loses

Once conditionals and jumps are followed, **`call` is 55.6% of block ends** — much the
largest limit on block length. The translation needed nothing new: `InstKind::CallRel`
already pushes the return address and *then* sets a terminator, so following it is a matter
of not terminating, with discovery mirroring the `JmpRel` arm.

Built, correct — **0 verifier mismatches over 12M, oracle matches at 100K/1M/3M** — and a
decisive loss. Six paired deep windows, `AERO_CT_FOLLOW_CALLS=1` against the default:

| window | following calls | off |
| --- | --- | --- |
| 140M→200M | 61.0 | **50.2** |
| 200M→300M | 58.5 | **49.8** |
| 300M→400M | 43.1 | 61.4 |
| 400M→500M | 73.5 | **64.6** |
| 500M→600M | 72.5 | **64.1** |
| 600M→700M | 70.9 | **63.2** |

Five of six worse by 12-21%; 300M→400M is the window that swings.

The mechanism is measured, not guessed:

| | off | following calls |
| --- | --- | --- |
| guest instructions per block | 7.35 | 7.75 |
| emitted WASM | 98.4 MiB | **183.2 MiB** |
| in-wasm chain rate | 70.5% | **58.4%** |
| block ends: `ret` / `exit` | 22.8% / 16% | **35.7% / 26.4%** |

Blocks get only 5% longer, because a followed callee reaches a `ret` or an instruction
Tier-1 refuses almost immediately — those two become 62% of block ends. Meanwhile the
callee is duplicated at **every** call site, so emitted code nearly doubles, and the
promoted set fragments enough to cost 12 points of chain rate. Inlining pays when the
callee is hot relative to its size; here it is neither.

That is the sixth measured attempt in the "longer blocks" family, and the first with a
clear account of *why*: the cost is duplication, not the extra instructions.

## `& 0xffff_ffff` versus wrap/extend: the same meaning, very different code

Back to the disassembly. In the block Cranelift emits for four guest ALU instructions,
five of ~26 instructions were:

```
and 0x78(%rip), %r11        # 0xffff_ffff, out of the constant pool
```

That is 32-bit operand truncation. The codegen wrote it as `i64.const 0xffff_ffff` +
`i64.and`, and Cranelift needs the 64-bit constant, which it materialises from the
constant pool as a memory operand. The canonical wasm idiom for the same thing —
`i32.wrap_i64` then `i64.extend_i32_u` — becomes a 32-bit move, which x86-64 zero-extends
for free and which folds away entirely when the producer already wrote a 32-bit result.

Applied to `emit_read_gpr_part` and `emit_write_gpr_part` (the 128-bit multiply helper
keeps the mask; it is cold):

| | mask | wrap/extend |
| --- | --- | --- |
| block, 4 guest ALU ops | 64 native instructions | **50** |
| rip-relative masks in it | 5 | **1** |
| emitted WASM, whole session | 98.4 MiB | 97.3 MiB |

Whole machine, six paired deep windows against `AERO_JIT_MASK32=1`:

| window | wrap/extend | mask |
| --- | --- | --- |
| 140M→200M | **49.7** | 50.1 |
| 200M→300M | **49.4** | 49.7 |
| 300M→400M | 44.0 | 63.0 |
| 400M→500M | **64.3** | 64.7 |
| 500M→600M | **63.8** | 64.1 |
| 600M→700M | **62.9** | 63.1 |

Five windows, all improving, 0.3-0.8% — small, but unanimous in direction and with the
mechanism visible in the disassembly. (300M→400M is the window that swings; it is not
evidence either way.) On by default.

**`compiled_code_cost` cannot price this one.** It reports the per-ALU-op cost falling from
4.07 to *zero*, because with the truncation written as wrap/extend Cranelift can finally
see through the operations and constant-folds the harness's 32 dependent `add eax, imm`
into one. Real guest code does not look like that. The harness measures codegen quality by
adding identical instructions, and this change is precisely one that makes identical
instructions collapse — so the machine is the only honest measurement here.

### The dead RFLAGS load: right diagnosis, wrong fix

The same disassembly shows the block prologue loading `CpuState.rflags` and nothing
consuming it — `mov 0x88(%r8,%r9),%r11` immediately overwritten by the next instruction.
One instruction in twenty.

This document previously explained that away: "a dead store pays to remove and a dead load
does not — Cranelift deletes it". **That reasoning is wrong.** Cranelift *cannot* delete
it: a wasm load may fault into the guard region, so it is a trapping operation and has to
stay. The earlier measurement found nothing because a block was ~64 native instructions
then and one load was invisible.

The load is gated on `uses_rflags`, which is true for a block that merely *writes* flags.
`reads_incoming_flags` looks like the precise condition — a partial flag write with nothing
produced yet, `adc`/`sbb` consuming carry, a condition evaluated before flags are produced.
Narrowing to it removes the load, and is **wrong**: the oracle mismatches at 100K
instructions and the machine falls to **1243.2** host instructions per guest instruction,
near the interpreter. Some path reads the RFLAGS local without any of those three marking
it, and a wasm local starts at zero, so it reads flags of zero.

Reverted — and then tracked down. The RFLAGS local has exactly three readers:

- the epilogue store, already gated on the block having written flags;
- **`emit_spill_all_state`, which stores it unconditionally** — and that runs on every bail
  path, so an MMIO exit, an assist or a page fault in a block whose flags were all deferred
  would spill a *zeroed* local over the guest's real RFLAGS. That is the divergence;
- `emit_rflags_bit`, which extracts a single flag, and which must not run when the flags
  live in the descriptor rather than the local.

So the load cannot be narrowed on its own: the prologue load, the bail-path spill and every
local reader have to move together, and "does this block's flag write reach the local"
is a property of the *emitter's* deferral decision, which is made while emitting the body —
after the prologue has already been written. Doing it properly means emitting the body
first and prepending the prologue once the answer is known.

Left undone: one instruction in twenty, against a three-site refactor in code that just
produced a silent 25x regression. The diagnosis is here so the next attempt starts from it
rather than from the disassembly.

**And the way it failed matters more than the change.** `AERO_CT_VERIFY` reported **0**
mismatches over 3M while the oracle caught it on its first checkpoint. The per-block
verifier only checks blocks marked `replayable`, so a whole class of divergence is outside
it. It is not a substitute for the oracle, and this document has leaned on it as though it
were.

## The admission path costs 4%, and making it cheaper does not help

The deep-window profile shows `CompiledTier::compile` at 1.5 host instructions per guest
instruction and `record_hit_with` at 1.3 — about **4.4% spent deciding not to compile
anything**. Instrumented over a 600M-instruction session:

| | |
| --- | --- |
| `compile()` calls | 61,064 |
| `record_hit_with` calls | **3,613,054** (one per 161 guest instructions) |
| cost per `record_hit_with` | ~209 host instructions |
| `hotness` map | **109,543 entries** |
| `rejected` / `blocks` | 8,291 / 52,732 |

So `compile()` is barely called — the 1.5 attributed to it is inlined neighbours, not
compilation, which the block counts confirm: 52,752 blocks at 200M and 52,768 at 600M, so
sixteen new blocks in 400M instructions and no recompilation churn at all.

The real cost is `record_hit_with` probing three maps per offer, the largest of which has
grown past a megabyte — a guaranteed cache miss to bump a counter for a RIP that will never
be compiled. Only 52,732 of those 109,543 RIPs ever became blocks.

**A direct-mapped counter, the same trick the chain cache uses, does not pay.** At 16,384
entries it thrashes: blocks admitted fall 52,752 → 39,216 and coverage 90.9% → 89.6%, with
throughput flat (49.8/49.7, 63.0/62.8). At 131,072 entries coverage falls *further* (88.8%)
even though more blocks are admitted, and the windows disagree violently — 53.3 against
49.7 in one, 39.9 against 62.8 in another.

That disagreement is the real finding. Changing when a block is admitted changes which
blocks exist, which changes the guest's path — the measurement floor this document opens
with. An optimisation that perturbs admission cannot be measured by these windows at all,
whatever its local merit. Reverted; the map keeps the best coverage, and the diagnostics
came out with it rather than being left on a 3.6M-call path.

## A KVM backend, started

Every item the profile and the disassembly still surface is worth a few percent, and the
gap to the target is 4x. That is not a tuning gap. The structure says why: a block entry
costs far more than the four to seven guest instructions it covers, and that is inherent to
translating into WebAssembly and calling into an engine.

Hardware virtualisation does not have that structure, and this is no longer a remembered
figure from a scratch C program — it is in the tree and it passes:

```
  KVM ran 159907840 guest instructions in 0.045s = 3589 MIPS
  test kvm_runs_a_known_number_of_guest_instructions ... ok
```

**3,589 MIPS against the tier's 110-140** — 26x, and 7x the target. `crates/aero-machine/src/kvm.rs`
opens `/dev/kvm`, gives the VM a slice of host memory as guest RAM, creates a vCPU, runs it,
and decodes the exits that matter (`HLT`, port I/O, MMIO, shutdown, failed entry).
`crates/aero-machine/tests/kvm_runs_guest_code.rs` runs a real-mode loop whose instruction
count is exact by construction and skips — rather than fails — where `/dev/kvm` is absent.

Two things the ABI does not forgive, both found by doing it:

- The ioctl request numbers encode struct sizes, so a transcribed constant is a silently
  wrong ioctl rather than a compile error. They are read out of `linux/kvm.h` on this
  machine and the sizes are pinned in the module.
- `KVM_SET_USER_MEMORY_REGION` requires a **page-aligned** userspace address and fails with
  a bare `EINVAL` otherwise. A `Vec<u8>` is byte-aligned, so the obvious thing does not
  work; `GuestRam` owns an `mmap` instead. That is also what the integration needs, since
  KVM holds host addresses rather than copying and the region must not move.

### The guest's own state, on hardware

Running code written *for* KVM proves the ioctls work. Running the emulator's own
architectural state is what would let it replace the compiled tier, so
`load_cpu_state` / `store_cpu_state` came next — and that conversion is where the silent
failures live:

- **`CpuState::gpr` is in architectural encoding order** (RAX, RCX, RDX, RBX, RSP, RBP,
  RSI, RDI, R8..) while `struct kvm_regs` is RAX, RBX, RCX, RDX, RSI, RDI, RSP, RBP, R8...
  A straight copy swaps four registers and the guest keeps running, wrongly.
- **Segments carry hidden state.** A selector is not enough: base, limit and access rights
  are what the CPU uses, and after a `mov ds, ax` they need not agree with the descriptor
  table any more. `Segment::access` is already the VMX AR-byte layout, which is exactly the
  set of fields `struct kvm_segment` splits out, so it decodes rather than translates.
- **`unusable` is not "absent".** A null-but-loaded segment has to arrive unusable or the
  guest faults on its first access.

`tests/kvm_cpu_state.rs` pins both directions: a populated state round-trips with every
GPR distinct (so a swapped pair is visible rather than plausible), and then a **64-bit
long-mode guest with page tables built in the test runs from a loaded `CpuState`** and
computes the right answer:

```
  long-mode guest ran from a loaded CpuState: rax=0x1111333344445555
  test a_cpu_state_survives_a_round_trip_through_the_vcpu ... ok
  test a_long_mode_guest_runs_from_a_loaded_cpu_state ... ok
```

### MSRs, and exits that reach a device model

Two more pieces, both of which the booted guest needs.

**MSRs.** `EFER` travels in `kvm_sregs`; everything else goes through `KVM_SET_MSRS`, whose
argument is a count followed by a flexible array of 16-byte `(index, data)` entries built by
hand. `STAR`/`LSTAR`/`CSTAR`/`FMASK`, the `FS`/`GS`/`KernelGS` bases, the three `SYSENTER`
registers, `TSC` and `TSC_AUX` now round-trip, each with a distinct value in the test so a
mis-mapped index shows as the wrong field rather than a plausible number. A guest restored
without these lands in the wrong syscall entry point on its first `syscall`. `EFER` is
deliberately *not* in that list — setting it through both paths is how you get a VM that
enters long mode in one place and not the other.

**Routed exits.** `run_routed` turns a vCPU exit into the call the device models already
answer — by port, or by address — and puts the result where the guest will look:

```
  routed: in 0x71 -> 0x5a, out 0x70 <- 0x5a
  routed: MMIO read at 0x100000
```

The first test has the guest read a port, then write the same byte back out; the `out`
carrying `0x5a` is what proves the read's answer reached the guest rather than being
dropped. The second reads one byte past the mapped region and it arrives as an MMIO read at
`0x100000` — which is how every device BAR will be reached.

Two details that are quiet when wrong, both handled: a read's answer must be in the
`kvm_run` buffer *before the next entry*, and a port exit's `count` may exceed one for a
string instruction, so each iteration is a separate access rather than one look at the
buffer.

`inject_interrupt` is in place too, treating `EEXIST` — one already pending — as "not now"
rather than an error, since the caller simply retries after the next exit.

### A guest under hardware virtualisation, driving a real device

Routing to closures proves the plumbing; it does not prove the device models can serve it.
So `tests/kvm_drives_a_real_device.rs` registers an actual `DebugCon` on an actual
`IoPortBus` — the same bus the interpreter and the compiled tier use — and has a guest
running under KVM write to port `0xE9`:

```
  a KVM guest wrote "KVM" to a real DebugCon through the real IoPortBus
```

The bridge is four lines: `KVM_EXIT_IO` carries port, direction, size and count, and
`IoPortBus::read`/`write` take exactly those. What remains on the port side is breadth —
every device — rather than a question of shape.

### MMIO to a real device, and one memory

**MMIO.** `tests/kvm_mmio_reaches_a_device.rs` maps a real `MmioHandler` on a real
`MemoryBus` — the bus `read_physical`/`write_physical` that the interpreter and the compiled
tier call — and a guest under KVM reads and writes it:

```
  MMIO: device answered 0xA5, guest wrote back 0xA6 through MemoryBus
```

The device answers `0xA5`, the guest increments and writes back `0xA6`. That round trip is
the point: a stale `kvm_run` buffer or a dropped answer gives something else, so the write
proves the device's value actually reached the guest. This is how every PCI BAR is reached.

**One memory.** The machine's RAM is not a slice it can lend out — under `AERO_JIT_RAM=1` it
lives inside the JIT backend's linear memory, reached through a raw pointer so the
interpreter and compiled code can both touch it. That memory is an `mmap`, hence already
page-aligned and contiguous, which is exactly what `KVM_SET_USER_MEMORY_REGION` demands. So
`new_from_raw` takes it directly rather than copying, and
`tests/kvm_shares_guest_ram.rs` checks the property an execution mode depends on:

```
  host wrote 0x41 at 0x8000; guest read it, incremented, and the host sees 0x42 at 0x8100
```

Both directions, no synchronisation, no flush. A device doing DMA, a snapshot being
restored and the vCPU all see one memory.

### Waking a halted guest

Aero's PIC and IOAPIC already decide *which* vector to raise. What the backend needs from
the hypervisor is only the timing: a vector cannot go in while the guest has interrupts
masked or is part-way through an instruction. `request_interrupt_window` turns that from
"retry until it takes" into a single exit — KVM returns `KVM_EXIT_IRQ_WINDOW_OPEN` at the
first moment injection is possible — and `KVM_INTERRUPT` hands the vector over.

`tests/kvm_delivers_an_interrupt.rs` halts a guest with interrupts enabled, so it makes no
further progress on its own, then injects:

```
  injected vector 0x20; the halted guest woke and ran its handler
```

`interrupts_enabled()` is checked separately from "halted", because a guest sitting in
`cli; hlt` is halted and *cannot* take a vector — waking it would be wrong.
`KVM_GET_LAPIC`/`KVM_SET_LAPIC` are in place for the 1KiB register file a snapshot restore
has to write.

### A slice, driven the way the machine drives a tier

`KvmRunner::run_slice` is the composition: decide an interrupt, enter the guest, service
what comes back, hand control out with the CPU state updated. It takes the machine's own
`IoPortBus` and `MemoryBus`, so nothing in it knows what a device is.

`tests/kvm_runner_slice.rs` runs a guest that does all of it in one slice — reads a port
from a real device, writes the value to a real MMIO handler, then idles on `hlt` — and
takes a vector on the next:

```
  slice 1: in 0x64 -> 0x3c -> MMIO write, then halt
  slice 2 -> Halted at rip=0x507, sp=0xfffa (three words pushed by the vector)
```

The MMIO value proves the port read's answer survived to the next exit; the stack pointer
proves the vector was delivered as a real-mode interrupt rather than the guest merely
resuming.

Two things that had to be right, and were not at first:

- **The interrupt handshake.** `KVM_INTERRUPT` accepts a vector whenever nothing else is
  queued, but accepting is not the same as the guest being able to *take* it. Injecting at
  an arbitrary moment faults during delivery and a real-mode guest triple-faults back to
  `0xffff0000:0xfff0`. The order is: request a window, wait for `KVM_EXIT_IRQ_WINDOW_OPEN`,
  then inject.
- **Resuming versus reloading.** When KVM owns the CPU the vCPU *is* the architectural
  state. `resume_slice` continues from it; `run_slice` is for entering from a state another
  executor produced.

### `IDTR.limit = 0`, and a wrong conclusion retracted

The slice test triple-faulted on its second slice and came back at the reset vector. I
wrote that up as a gap in `load_cpu_state` — a real-mode state it loads cannot take an
interrupt — and that was **wrong**.

Bisected properly, by loading the vCPU's own state back through the conversion and then
each group of a hand-built one in turn:

| what was loaded | vector delivered? |
| --- | --- |
| the vCPU's reset state, untouched | yes |
| its own state, round-tripped through `store`/`load` | **yes** |
| a hand-built state, whole | no |
| hand-built + the vCPU's real MSRs | no |
| real descriptors + zeroed MSRs | **yes** |
| hand-built **segments** only | yes |
| hand-built **control registers** only | yes |
| hand-built **descriptor tables** only | **no** |

The conversion is faithful — the round trip delivers, and an `sregs` diff across it shows
no field changing. What was wrong was the state I built: `CpuState::new` leaves the
descriptor tables at zero, so **`IDTR.limit` was 0**. The guest runs perfectly and then
triple-faults the instant a vector arrives, because the vector's entry lies outside the
table. In real mode that table is the IVT at physical zero, and it still goes through IDTR.

Set `idtr.limit = 0x3FF` and every case above delivers. The slice test now uses the full
`run_slice` load path.

The general form is worth keeping: **a vCPU does not tolerate a half-filled state the way
an interpreter does.** Segment descriptors of all zeros are *not present*; a zero `IDTR`
limit means no interrupt is ever deliverable. Both are silent until the guest happens to
need them.

## The booted Windows guest, executing on the CPU

Everything above ran code written for the test. This takes the architectural state of the
*real* Windows 7 desktop — restored from the same snapshot the compiled tier is measured
against — puts it in a vCPU, and lets the hardware execute it:

```
  restored: rip=0x77829e83 cr0=0x80050021 cr3=0x1c465000 cr4=0x6f8 efer=0xd01 mode=Long
  ran: KVM_EXIT_SET_TPR after 1 exits — the guest is in its own kernel
  rip 0x77829e83 -> 0xfffff80002699a59   (moved: true)
```

It starts in user-mode `ntdll` at `0x77829e83` and comes back inside the Windows kernel at
`0xfffff800_02699a59`, having lowered its task-priority register. So a long-mode Windows
state — its page tables, segments with hidden parts, MSRs and descriptor tables — is
**acceptable to hardware and makes forward progress from where the emulator left it.**
That was the thing genuinely in doubt, and it is the reason `KVM_EXIT_SET_TPR` is now a
named variant: left as `Other(11)` it reads as "the guest did not run".

`tests/kvm_runs_the_booted_guest.rs` skips where `/dev/kvm` or the snapshot is absent, so
it is inert anywhere but this machine. The 2GiB memory copy is a one-off, not a per-run
cost.

### Taking a spinning guest off the CPU

With `KVM_CREATE_IRQCHIP` the guest gets past its first task-priority write and runs on
into its own kernel — and then never comes back, because it is waiting for a timer tick
nobody delivers. `KVM_RUN` does not return, and **no budget expressed in exits can reach a
guest that is not exiting.** That is a gap any slice-based execution mode has to close, not
a detail of this test.

The remedy is a signal: with no `SA_RESTART` the in-flight ioctl fails with `EINTR`, which
`run()` now reports as `KvmExit::Interrupted` rather than an error. The part that matters
is *which thread gets it*. `setitimer` raises `SIGALRM` process-wide and any thread may
take it — under a test harness reliably the wrong one, and the guest runs on undisturbed
while the timer fires perfectly. `preempt_this_thread` records the calling thread and has a
helper `pthread_kill` that thread specifically; it returns a guard that stops the helper on
drop.

With that in place the booted guest is bounded and legible:

```
  restored: rip=0x77829e83 cr0=0x80050021 cr3=0x1c465000 cr4=0x6f8 efer=0xd01 mode=Long
  exits: 5 (200 preemptions) in 0.415s; ports []; mmio [e0000138, e0000134, e0000128, e0000118]
  rip 0x77829e83 -> 0xfffff80002615fad
```

It is executing its own kernel, reaching device MMIO in the `0xe0000000` BAR window, and
spinning because nothing answers. That is the expected shape with no devices behind the
exits, and it is now observable instead of a hang.

### One memory, real devices, the guest's own APIC — and then it runs

Three more pieces, each of which moved the guest measurably:

**One memory.** `SharedGuestRam` implements `memory::GuestMemory` over a page-aligned
`mmap` behind an `Arc`, so the same pages go to `Machine::new_with_guest_memory` and to
`KvmVm::new_from_raw`. With two allocations the vCPU writes a descriptor into its own copy
while a device model reads the machine's, and nothing the guest asks for comes back right.

**Real devices behind the exits.** Ports to `Machine::io_read`/`io_write`, MMIO to
`read_physical_bytes`/`write_physical` — the same entry points the other tiers use.

**The guest's own APIC.** `kvm_lapic_state` is the APIC page's first kilobyte on a 16-byte
stride, the same addressing the machine's model uses, so the transfer is a loop. Without it
the vCPU's APIC sits at power-on defaults while the guest waits on a timer it programmed
before the snapshot.

### It is running Windows

I had written this up as "stops in a kernel poll loop". That was **wrong**, and disassembling
at the sampled RIP is what showed it:

```
  0xfffff800026f17e0  [48, 0f, c3, 01]        MOVNTI [rcx], rax
  0xfffff800026f17e4  [48, 0f, c3, 41, 08]    MOVNTI [rcx+8], rax
  ... eight of them, [rcx] through [rcx+56]
```

Eight non-temporal stores filling 64 bytes — the kernel's page-zeroing loop. RIP lands
there because it is hot, not because it is stuck. Sampling confirms it:

```
  RIP samples: 400 taken, 128 distinct, spanning 0x776c3380..0xfffff960000f8f50
```

User mode, kernel, and Win32k session space. The guest is executing Windows.

### The number

`perf stat -e instructions:G` counts guest-mode instructions, which is the only honest way
to measure this — the host-instruction metric the rest of this document uses cannot see
inside a vCPU.

| window | guest instructions | rate |
| --- | --- | --- |
| first 1.00s | **1,045,059,747** | **~1,045 MIPS** |
| first 3.00s | 1,175,765,604 | ~392 MIPS |
| first 5.00s | 1,175,880,901 | ~235 MIPS |

**About 1,050 MIPS of real Windows in the first second**, against the compiled tier's
110-140.

### It is idle, not stuck

After that first second it does essentially nothing more — the same ~1.176G total whether
the run is 3 seconds or 5, with zero halts and no new exits. I called that a spinlock under
pause-loop exiting. It is not. `KVM_GET_MP_STATE` says **`HALTED`**:

```
  vCPU mp_state at the end: Ok(3)   (0 runnable, 1 uninit, 2 init, 3 halted, 4 sipi)
  after 2.1s: 430 samples, 1 distinct — 0xfffff8000262a762 x430   (a `ret`)
```

The guest ran out of work and executed `HLT`. It is waiting, correctly, for a timer
interrupt — and that interrupt never comes, because **Windows programmed *Aero's* IOAPIC
before the snapshot was taken while KVM's in-kernel chips are at their reset defaults.**
Raising IRQ 0 into them goes nowhere, which is why doing so changed nothing.

Two things made this hard to see and are worth remembering:

- **With an in-kernel irqchip, `HLT` never reaches userspace.** KVM blocks in the kernel
  until an interrupt arrives, so there is no `KVM_EXIT_HLT` to count. The preemption signal
  turns the wait into `EINTR`, which reads as "running but doing nothing". `mp_state` is the
  only thing that distinguishes idle from stuck.
- **A frozen RIP at a `ret`** looks like a fault loop and is nothing of the kind — it is
  simply where the guest happened to be when it halted.

### What is still missing

- **The interrupt controllers' state.** Either `KVM_SET_IRQCHIP` from Aero's PIC/IOAPIC, or
  the split-irqchip arrangement so Aero's models stay authoritative and deliver through
  `KVM_INTERRUPT`. That is the single thing between here and a guest that keeps running.
- **`Machine` owning the loop**, rather than a test approximating it.

**It still cannot boot Windows unattended.** But the claim that the alternative cannot run
the real workload is no longer true: it runs it, and while it has work to do it runs it
about eight times faster than the compiled tier.


## Where this stands

Current defaults, measured the same way throughout — host instructions retired per guest
instruction, differenced across two `--max-insts` budgets:

| window | interpreter | tier | tier + `AERO_CT_CACHE_DIR` |
| --- | --- | --- | --- |
| late (140M→200M) | — | — | **49.4** |
| 300M→400M | — | — | **43.6** |
| 500M→600M | — | — | **63.4** |
| 600M→700M | — | — | **62.6** |

Roughly 108-158 MIPS, from 55-75 when this effort began.

Superblocks are on, so the 80M→200M row is the one window they cost; every deeper window
is 10-12% better. The columns are the interpreter, the tier, and the tier with
`AERO_CT_CACHE_DIR`.

About 13x the interpreter in steady state, from 1.9x when this work began and 8.5x at
the start of the most recent effort.

Two things that were re-checked at this baseline and stay as they are:

- **`AERO_CT_PROMOTE` stays at 2000.** Widening it to 1000 was rejected earlier on
  Cranelift compile cost, which a warm module cache removes — so it was worth re-testing
  under the rule that a default is only as current as the economics it was measured
  against. It is not that: **204.4 against 97.6 with both caches warm.**
- **The chain map stays promoted-only**, for the reasons already recorded: extending it
  to admitted blocks thrashes a 2,048-entry direct-mapped table, with or without the
  in-wasm execution counter that fixes the promotion starvation.

The gap that remains is not a tuning gap. KVM on this machine measures 3,693 MIPS
against the tier's ~60–75, and the structural floor of this design — ~190 host
instructions of per-block overhead over blocks of ~4.4 guest instructions — is around
150 MIPS. Closing it needs a different execution structure on native, with this tier
kept for the browser, where no hardware virtualisation exists.

### The chain map cannot be grown, which closes admission-time chaining for good

In-wasm chaining is capped at ~20% of block entries because the map holds only promoted
blocks, and extending it to admitted blocks thrashes a 2,048-entry direct-mapped table.
The obvious fix is a bigger table, and the reason for expecting that to be free is that a
direct-mapped table's cache footprint is set by the entries actually *touched*, not by
its size — growing it should only reduce collisions.

Measured, and it is not:

| map entries | promoted-only | admission chaining |
| --- | --- | --- |
| 2,048 | **102.5** | 144.1 |
| 8,192 | 628.2 | 319.7 |
| 32,768 | out-of-bounds trap | — |

The promoted-only column is the one to read, because that configuration uses ~240
entries whatever the table size and should barely care. It is six times worse at 8,192 —
consistently, across four windows (628–638), with the cost host-side rather than in
generated code and the tier's own statistics unchanged.

**But shrinking is also worse, and unequally:** 512 gives 120.4 and 1024 gives 258.3,
against 2,048's 102.5. Jagged in both directions is the signature every size-like
parameter here has — the context lives in guest linear memory, so changing its size moves
what follows it and the guest ends up on a different path. So the honest reading is not
"growing the map costs X for reason Y"; it is that this constant is not a dial, and 2,048
happens to be where it sits. At 32,768 there is at least an unambiguous answer: the
context outgrows its reserved region and the machine traps.

So: the map stays at 2,048, chaining stays promoted-only and capped near 20%, and the
last route to raising it is closed. Together with the in-wasm execution counter (built,
works, 498.3) and the promoted-set consolidation (built, 239.6), that is the whole of
what can be done about the engine crossing from this side.

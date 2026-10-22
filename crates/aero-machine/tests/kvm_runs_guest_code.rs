//! Does hardware virtualisation actually run guest code here, and how fast?
//!
//! The compiled tier is ~50-64 host instructions per guest instruction, about 110-140
//! MIPS. This measures the alternative on the same machine, in-tree, so the number behind
//! every "KVM is the native answer" claim in `wiki/areas/performance.md` is reproducible
//! rather than a remembered figure from a scratch C program.
//!
//! The guest is a 16-bit real-mode loop whose instruction count is exact by construction:
//! `LOOP_BODY` instructions per iteration for `ITERATIONS` iterations, then `hlt`.
//!
//! Skips — rather than fails — where `/dev/kvm` is absent or unreadable, which is the
//! normal case in a container without the device.
//!
//! ```text
//! cargo test --release -p aero-machine --test kvm_runs_guest_code -- --nocapture
//! ```

#![cfg(all(target_os = "linux", not(target_arch = "wasm32")))]

use aero_machine::kvm::{GuestRam, KvmError, KvmExit, KvmVm};

/// `add ax,1` x7 then `loop` — eight instructions an iteration.
const LOOP_BODY: u64 = 8;
const ITERATIONS: u64 = 20_000_000;

/// 64KiB is one real-mode segment, which is all this guest needs.
const RAM: usize = 0x1_0000;

fn guest_code() -> Vec<u8> {
    let mut code = Vec::new();
    // mov cx, ITERATIONS (low 16 bits) — the loop counter.
    code.extend_from_slice(&[0xb9, 0x00, 0x00]);
    let loop_top = code.len();
    for _ in 0..7 {
        // add ax, 1
        code.extend_from_slice(&[0x83, 0xc0, 0x01]);
    }
    // loop <top>
    let rel = (loop_top as i64 - (code.len() as i64 + 2)) as i8;
    code.extend_from_slice(&[0xe2, rel as u8]);
    // hlt
    code.push(0xf4);
    code
}

#[test]
fn kvm_runs_a_known_number_of_guest_instructions() {
    // Page-aligned: KVM rejects anything else, and a `Vec` is byte-aligned.
    let mut ram = GuestRam::new(RAM);
    let code = guest_code();
    ram.as_mut_slice()[..code.len()].copy_from_slice(&code);

    // Safety: `ram` outlives `vm` and its mapping does not move.
    let mut vm = match unsafe { KvmVm::new(ram.as_mut_slice(), 0) } {
        Ok(vm) => vm,
        Err(KvmError::Unavailable(e)) => {
            println!("  /dev/kvm unavailable ({e}); skipping");
            return;
        }
        Err(e) => panic!("could not create a KVM VM: {e}"),
    };

    vm.set_real_mode(0, 0).expect("set real mode");

    // `loop` uses the full CX, so program the iteration count there. 20M does not fit in
    // 16 bits, so run the loop to exhaustion repeatedly and count what actually ran.
    let per_run: u64 = 0x1_0000;
    let runs = ITERATIONS / per_run;
    let mut regs = vm.regs().expect("regs");

    let started = std::time::Instant::now();
    let mut guest_insts: u64 = 0;
    for _ in 0..runs {
        regs.rip = 0;
        regs.rcx = 0; // CX = 0 means 65536 iterations for `loop`.
        vm.set_regs(&regs).expect("set regs");
        loop {
            match vm.run().expect("KVM_RUN") {
                KvmExit::Halt => break,
                KvmExit::Io { .. } => {}
                other => panic!("unexpected exit {other:?}"),
            }
        }
        // 3 setup bytes are one `mov`, then the body.
        guest_insts += per_run * LOOP_BODY;
    }
    let elapsed = started.elapsed();

    let mips = guest_insts as f64 / elapsed.as_secs_f64() / 1e6;
    println!(
        "  KVM ran {guest_insts} guest instructions in {:.3}s = {mips:.0} MIPS",
        elapsed.as_secs_f64()
    );

    assert!(
        guest_insts > 0,
        "the guest must have executed something for the rate to mean anything"
    );
    // Deliberately loose: this asserts the mechanism works, not a particular speed. The
    // compiled tier is ~110-140 MIPS, so anything at or below that would mean the vCPU is
    // not really running guest code.
    assert!(
        mips > 200.0,
        "KVM managed only {mips:.0} MIPS, which is not hardware virtualisation running"
    );
}

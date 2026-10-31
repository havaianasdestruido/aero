/// <reference lib="webworker" />

// Emulator worker behind the public player page.
//
// One worker owns the whole machine: it instantiates the Rust/WASM core, attaches
// the chosen boot medium, runs the guest in execution slices, and publishes the
// display. Running the guest off the main thread keeps the page responsive even
// when the guest spins (DOS guests do), and it means the page never has to touch
// emulator internals directly.
//
// Deliberately standalone (not the multi-worker `WorkerCoordinator` runtime that
// the Windows 7 bring-up shell uses): the player must work without
// `SharedArrayBuffer`/cross-origin isolation so the deployed static site boots
// anywhere, even before (or without) the isolation shim.

import { initWasmForContext, type WasmApi } from "../runtime/wasm_context";
import { assertWasmMemoryWiring } from "../runtime/wasm_memory_probe";
import { formatOneLineError } from "../../../../packages/transport-safety/src/text.js";
import { negateI32Saturating } from "../input/int32";
import { unrefBestEffort } from "../unrefSafe";
import {
  FRAMEBUFFER_FORMAT_RGBA8888,
  HEADER_BYTE_LENGTH,
  HEADER_INDEX_CONFIG_COUNTER,
  HEADER_INDEX_FORMAT,
  HEADER_INDEX_FRAME_COUNTER,
  HEADER_INDEX_HEIGHT,
  HEADER_INDEX_STRIDE_BYTES,
  HEADER_INDEX_WIDTH,
  addHeaderI32,
  initFramebufferHeader,
  storeHeaderI32,
  wrapSharedFramebuffer,
} from "../display/framebuffer_protocol";
import type {
  PlayerInboundMessage,
  PlayerOutboundMessage,
  PlayerPhase,
  PlayerStartMessage,
} from "../player/protocol";

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// --- tunables -------------------------------------------------------------

/** Instructions per `run_slice` call. Large enough that JS overhead is small. */
const SLICE_INSTRUCTIONS = 1_000_000;
/** Wall-clock budget of guest execution per loop turn before yielding. */
const SLICE_BUDGET_MS = 10;
/** Minimum interval between display refreshes. */
const PRESENT_INTERVAL_MS = 33;
/** Minimum interval between `player.stats` messages. */
const STATS_INTERVAL_MS = 1000;
/** Frame bytes we are willing to copy; larger scanouts are ignored. */
const MAX_FRAME_BYTES = 32 * 1024 * 1024;
/** Flat heap the Rust runtime gets on top of guest RAM. */
const RUNTIME_RESERVED_BYTES = 128 * 1024 * 1024;
const WASM_PAGE_BYTES = 64 * 1024;
const DEFAULT_RAM_BYTES = 64 * 1024 * 1024;
const MIN_RAM_BYTES = 2 * 1024 * 1024;

/** `RunExit.kind` values exported by `aero-wasm` (`RunExitKind`). */
const RUN_EXIT_HALTED = 1;
const RUN_EXIT_RESET_REQUESTED = 2;

type RunExitLike = { readonly kind?: number; readonly executed?: number; readonly detail?: string };

/** The subset of the `aero-wasm` `Machine` API the player depends on. */
type MachineLike = Record<string, unknown>;

// --- module state ---------------------------------------------------------

let machine: MachineLike | null = null;
let wasmApi: WasmApi | null = null;
let wasmMemory: WebAssembly.Memory | null = null;
let preferredMemory: WebAssembly.Memory | null = null;

let running = false;
let paused = false;
let loopScheduled = false;
let loopTimer: number | null = null;
let runToken = 0;

let transport: "shared" | "copy" = "copy";
let sharedFb: ReturnType<typeof wrapSharedFramebuffer> | null = null;
let sharedWidth = 0;
let sharedHeight = 0;
let sharedStrideBytes = 0;
let sharedFrameCounter = 0;
let copyFrameCounter = 0;

let presentWidth = 0;
let presentHeight = 0;
let presentedFrames = 0;

let executedTotal = 0;
let startedAtMs = 0;
let lastPresentMs = 0;
let lastStatsMs = 0;
let lastStatsExecuted = 0;
let lastFingerprint = 0;

const serialDecoder = new TextDecoder("utf-8", { fatal: false });

// --- messaging ------------------------------------------------------------

function post(msg: PlayerOutboundMessage, transfer: Transferable[] = []): void {
  if (transfer.length > 0) ctx.postMessage(msg, transfer);
  else ctx.postMessage(msg);
}

function phase(next: PlayerPhase, detail?: string): void {
  post({ type: "player.phase", phase: next, ...(detail !== undefined ? { detail } : {}) });
}

function fail(err: unknown): void {
  post({ type: "player.error", message: formatOneLineError(err, 512) });
}

/** Whether the machine exposes a given method. */
function hasMethod(name: string): boolean {
  const m = machine;
  return !!m && typeof m[name] === "function";
}

/** Call an optional machine method, returning `undefined` when unavailable. */
function call<T>(name: string, ...args: unknown[]): T | undefined {
  const m = machine;
  if (!m) return undefined;
  const fn = m[name];
  if (typeof fn !== "function") return undefined;
  return (fn as (...rest: unknown[]) => T).apply(m, args);
}

// --- display --------------------------------------------------------------

function ensureSharedFramebuffer(): ReturnType<typeof wrapSharedFramebuffer> | null {
  const SharedArrayBufferCtor = globalThis.SharedArrayBuffer;
  if (typeof SharedArrayBufferCtor === "undefined") return null;

  // 1024x768 is the practical upper bound for DOS-era guests; larger modes are
  // re-allocated on demand in `publishSharedFrame`.
  const bytes = HEADER_BYTE_LENGTH + 1024 * 768 * 4;
  let sab: SharedArrayBuffer;
  try {
    sab = new SharedArrayBufferCtor(bytes);
  } catch {
    return null;
  }

  const fb = wrapSharedFramebuffer(sab, 0);
  initFramebufferHeader(fb.header, { width: 1, height: 1, strideBytes: 4, format: FRAMEBUFFER_FORMAT_RGBA8888 });
  sharedFb = fb;
  sharedWidth = 1;
  sharedHeight = 1;
  sharedStrideBytes = 4;
  sharedFrameCounter = 0;
  return fb;
}

function publishSharedFrame(width: number, height: number, strideBytes: number, pixels: Uint8Array): void {
  const requiredBytes = strideBytes * height;
  let fb = sharedFb;
  if (!fb) return;

  if (requiredBytes > fb.pixelsU8.byteLength) {
    const SharedArrayBufferCtor = globalThis.SharedArrayBuffer;
    if (typeof SharedArrayBufferCtor === "undefined") {
      downgradeToCopyTransport();
      return;
    }
    try {
      const grown = new SharedArrayBufferCtor(HEADER_BYTE_LENGTH + requiredBytes);
      const next = wrapSharedFramebuffer(grown, 0);
      initFramebufferHeader(next.header, { width, height, strideBytes, format: FRAMEBUFFER_FORMAT_RGBA8888 });
      storeHeaderI32(next.header, HEADER_INDEX_FRAME_COUNTER, sharedFrameCounter);
      fb = next;
      sharedFb = next;
      sharedWidth = width;
      sharedHeight = height;
      sharedStrideBytes = strideBytes;
      post({ type: "player.ready", variant: variantName, transport: "shared", width, height, framebuffer: grown, bootDrive: currentBootDrive() });
    } catch {
      downgradeToCopyTransport();
      return;
    }
  }

  if (sharedWidth !== width || sharedHeight !== height || sharedStrideBytes !== strideBytes) {
    storeHeaderI32(fb.header, HEADER_INDEX_WIDTH, width);
    storeHeaderI32(fb.header, HEADER_INDEX_HEIGHT, height);
    storeHeaderI32(fb.header, HEADER_INDEX_STRIDE_BYTES, strideBytes);
    storeHeaderI32(fb.header, HEADER_INDEX_FORMAT, FRAMEBUFFER_FORMAT_RGBA8888);
    addHeaderI32(fb.header, HEADER_INDEX_CONFIG_COUNTER, 1);
    sharedWidth = width;
    sharedHeight = height;
    sharedStrideBytes = strideBytes;
  }

  fb.pixelsU8.set(pixels.subarray(0, requiredBytes), 0);
  sharedFrameCounter = (sharedFrameCounter + 1) >>> 0;
  storeHeaderI32(fb.header, HEADER_INDEX_FRAME_COUNTER, sharedFrameCounter);
}

function downgradeToCopyTransport(): void {
  if (transport === "copy") return;
  transport = "copy";
  sharedFb = null;
  copyFrameCounter = sharedFrameCounter;
  post({ type: "player.ready", variant: variantName, transport: "copy", width: presentWidth, height: presentHeight, bootDrive: currentBootDrive() });
}

function publishCopyFrame(width: number, height: number, strideBytes: number, pixels: Uint8Array): void {
  const requiredBytes = strideBytes * height;
  if (requiredBytes > MAX_FRAME_BYTES) return;
  const copy = new Uint8Array(requiredBytes);
  copy.set(pixels.subarray(0, requiredBytes));
  copyFrameCounter = (copyFrameCounter + 1) >>> 0;
  post(
    {
      type: "player.frame",
      width,
      height,
      strideBytes,
      frameCounter: copyFrameCounter,
      pixels: copy.buffer,
    },
    [copy.buffer],
  );
}

/**
 * Cheap change detector used on the copy transport so we do not ship identical
 * frames across the thread boundary 20 times a second.
 */
function fingerprint(pixels: Uint8Array, strideBytes: number, height: number): number {
  let hash = 0x811c9dc5;
  const step = 97; // prime-ish stride: samples the whole frame without a full scan
  for (let i = 0; i < pixels.length; i += step) {
    hash ^= pixels[i] ?? 0;
    hash = Math.imul(hash, 0x01000193);
  }
  hash ^= strideBytes & 0xff;
  hash = Math.imul(hash, 0x01000193);
  hash ^= height & 0xff;
  return hash >>> 0;
}

function readScanout(): { width: number; height: number; strideBytes: number; pixels: Uint8Array } | null {
  if (!hasMethod("display_present")) return null;
  try {
    call<void>("display_present");
  } catch {
    return null;
  }

  const width = (call<number>("display_width") ?? 0) >>> 0;
  const height = (call<number>("display_height") ?? 0) >>> 0;
  if (width === 0 || height === 0) return null;

  let strideBytes = (call<number>("display_stride_bytes") ?? width * 4) >>> 0;
  const tightBytes = width * height * 4;
  if (strideBytes < width * 4) return null;
  if (strideBytes * height > MAX_FRAME_BYTES) return null;

  const ptr = (call<number>("display_framebuffer_ptr") ?? 0) >>> 0;
  const len = (call<number>("display_framebuffer_len_bytes") ?? 0) >>> 0;
  const memory = wasmMemory;
  if (ptr !== 0 && len !== 0 && memory) {
    // Some builds report a padded stride but expose a tightly packed buffer.
    if (len < strideBytes * height && len >= tightBytes) strideBytes = width * 4;
    const requiredBytes = strideBytes * height;
    if (len >= requiredBytes && ptr + requiredBytes <= memory.buffer.byteLength) {
      return { width, height, strideBytes, pixels: new Uint8Array(memory.buffer, ptr, requiredBytes) };
    }
  }
  return null;
}

function presentIfDue(force: boolean): void {
  const now = performance.now();
  if (!force && now - lastPresentMs < PRESENT_INTERVAL_MS) return;

  const frame = readScanout();
  if (!frame) return;
  lastPresentMs = now;

  if (presentWidth !== frame.width || presentHeight !== frame.height) {
    presentWidth = frame.width;
    presentHeight = frame.height;
  }

  if (transport === "shared") {
    publishSharedFrame(frame.width, frame.height, frame.strideBytes, frame.pixels);
    presentedFrames += 1;
    return;
  }

  const hash = fingerprint(frame.pixels, frame.strideBytes, frame.height);
  if (!force && hash === lastFingerprint) return;
  lastFingerprint = hash;
  publishCopyFrame(frame.width, frame.height, frame.strideBytes, frame.pixels);
  presentedFrames += 1;
}

// --- serial ---------------------------------------------------------------

function drainSerial(): void {
  if (hasMethod("serial_output_len")) {
    let pending = 0;
    try {
      pending = (call<number>("serial_output_len") ?? 0) >>> 0;
    } catch {
      pending = 1;
    }
    if (pending === 0) return;
  }
  if (!hasMethod("serial_output")) return;
  let bytes: Uint8Array;
  try {
    const out = call<Uint8Array>("serial_output");
    if (!out) return;
    bytes = out;
  } catch {
    return;
  }
  if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) return;
  const text = serialDecoder.decode(bytes, { stream: true });
  if (text.length > 0) post({ type: "player.serial", text });
}

// --- run loop -------------------------------------------------------------

let variantName = "unknown";

function currentBootDrive(): number {
  if (hasMethod("boot_drive")) {
    try {
      return (call<number>("boot_drive") ?? 0x80) >>> 0;
    } catch {
      /* ignore */
    }
  }
  return 0x80;
}

function scheduleLoop(delayMs: number): void {
  if (!running || loopScheduled) return;
  loopScheduled = true;
  const token = runToken;
  loopTimer = ctx.setTimeout(() => {
    loopScheduled = false;
    loopTimer = null;
    if (!running || token !== runToken) return;
    runLoop();
  }, delayMs) as unknown as number;
  unrefBestEffort(loopTimer);
}

function runLoop(): void {
  const m = machine;
  if (!running || !m) return;

  if (paused) {
    presentIfDue(true);
    scheduleLoop(50);
    return;
  }

  const runSlice = m["run_slice"];
  if (typeof runSlice !== "function") {
    fail(new Error("Machine is missing run_slice()."));
    phase("error");
    running = false;
    return;
  }

  const start = performance.now();
  let haltSpins = 0;
  try {
    // Execute for a short wall-clock budget, then yield to the event loop so
    // input messages and stop/pause requests are still handled promptly.
    while (performance.now() - start < SLICE_BUDGET_MS) {
      const exit = (runSlice as (n: number) => RunExitLike | undefined).call(m, SLICE_INSTRUCTIONS);
      const executed = typeof exit?.executed === "number" ? exit.executed : 0;
      executedTotal += executed;
      const kind = exit?.kind;
      if (kind === RUN_EXIT_HALTED) {
        haltSpins += 1;
        if (haltSpins >= 2) break; // idle guest: wait for an interrupt
      } else {
        haltSpins = 0;
      }
      if (kind === RUN_EXIT_RESET_REQUESTED) {
        const reset = m["reset"];
        if (typeof reset === "function") (reset as () => void).call(m);
      }
      if (executed === 0 && haltSpins === 0) break;
    }
  } catch (err) {
    running = false;
    fail(err);
    phase("error");
    return;
  }

  drainSerial();
  presentIfDue(false);

  const now = performance.now();
  if (now - lastStatsMs >= STATS_INTERVAL_MS) {
    const elapsedMs = Math.max(1, now - (lastStatsMs || startedAtMs));
    const delta = executedTotal - lastStatsExecuted;
    lastStatsExecuted = executedTotal;
    lastStatsMs = now;
    post({
      type: "player.stats",
      instructionsPerSecond: Math.round((delta / elapsedMs) * 1000),
      executed: executedTotal,
      frames: presentedFrames,
      width: presentWidth,
      height: presentHeight,
      uptimeMs: Math.round(now - startedAtMs),
    });
  }

  scheduleLoop(haltSpins > 0 ? 4 : 0);
}

// --- lifecycle ------------------------------------------------------------

function stopMachine(): void {
  running = false;
  paused = false;
  runToken += 1;
  if (loopTimer !== null) {
    ctx.clearTimeout(loopTimer);
    loopTimer = null;
  }
  loopScheduled = false;

  const free = machine?.["free"];
  if (typeof free === "function") {
    try {
      (free as () => void).call(machine);
    } catch {
      /* ignore */
    }
  }
  machine = null;
  wasmApi = null;
  wasmMemory = null;
  sharedFb = null;
  presentWidth = 0;
  presentHeight = 0;
  executedTotal = 0;
  lastStatsExecuted = 0;
  lastStatsMs = 0;
  lastPresentMs = 0;
  lastFingerprint = 0;
}

function selectMemory(ramSizeBytes: number): WebAssembly.Memory {
  const wantedBytes = Math.max(RUNTIME_RESERVED_BYTES, ramSizeBytes + RUNTIME_RESERVED_BYTES);
  const pages = Math.ceil(wantedBytes / WASM_PAGE_BYTES);
  // `maximum` is what the module was linked with (4 GiB by default); keep the
  // initial mapping small so instantiation stays fast.
  return new WebAssembly.Memory({ initial: pages, maximum: 65536 });
}

function constructMachine(ramSizeBytes: number, display: "vga" | "aerogpu"): MachineLike {
  const api = wasmApi;
  if (!api?.Machine) throw new Error("This build has no Machine export (WASM package missing?).");

  const MachineCtor = api.Machine as unknown as {
    new (ram: number): MachineLike;
    new_with_config?: (ram: number, aerogpu: boolean, vga: boolean | undefined, cpuCount: number | undefined) => MachineLike;
  };

  if (typeof MachineCtor.new_with_config === "function") {
    // Explicit device selection: the DOS-era demo wants the classic VGA device,
    // not the AeroGPU register interface (which needs a guest driver).
    const enableAerogpu = display === "aerogpu";
    return MachineCtor.new_with_config(ramSizeBytes >>> 0, enableAerogpu, !enableAerogpu, 1);
  }
  return new MachineCtor(ramSizeBytes >>> 0);
}

function normaliseImage(bytes: ArrayBuffer, kind: "hdd" | "cd"): Uint8Array {
  const image = new Uint8Array(bytes);
  if (kind === "cd") return image;
  // The IDE/ATA path requires whole 512-byte sectors.
  if (image.byteLength === 0) throw new Error("Disk image is empty.");
  if (image.byteLength % 512 === 0) return image;
  const padded = new Uint8Array(Math.ceil(image.byteLength / 512) * 512);
  padded.set(image, 0);
  return padded;
}

async function start(msg: PlayerStartMessage, token: number): Promise<void> {
  stopMachine();
  running = true;
  runToken = token;
  startedAtMs = performance.now();
  phase("starting", "Loading the emulator core…");

  const ramSizeBytes = Math.max(MIN_RAM_BYTES, Math.trunc(msg.ramSizeBytes ?? DEFAULT_RAM_BYTES));
  const display = msg.display === "aerogpu" ? "aerogpu" : "vga";

  let init: Awaited<ReturnType<typeof initWasmForContext>>;
  try {
    if (!preferredMemory) preferredMemory = selectMemory(ramSizeBytes);
    init = await initWasmForContext({ variant: "single", memory: preferredMemory });
    assertWasmMemoryWiring({ api: init.api, memory: preferredMemory, context: "player.worker" });
  } catch (err) {
    // Older/newer builds may not import memory; fall back to wasm-bindgen's
    // default instantiation so the player still works.
    console.warn("[player.worker] preallocated-memory init failed, falling back:", err);
    init = await initWasmForContext();
  }
  if (!running || token !== runToken) return;

  wasmApi = init.api;
  wasmMemory = init.wasmMemory ?? preferredMemory;
  variantName = String(init.variant ?? "unknown");

  phase("starting", "Powering on the machine…");
  machine = constructMachine(ramSizeBytes, display);

  const disk = msg.disk ?? null;
  let bootDrive = 0x80;
  if (disk) {
    const image = normaliseImage(disk.bytes, disk.kind);
    if (disk.kind === "cd") {
      const attach = machine["attach_install_media_iso_bytes"];
      if (typeof attach !== "function") throw new Error("This build cannot attach CD media.");
      (attach as (b: Uint8Array) => void).call(machine, image);
      bootDrive = 0xe0;
      call("set_boot_from_cd_if_present", true);
    } else {
      const attach = machine["set_disk_image"];
      if (typeof attach !== "function") throw new Error("This build cannot attach a hard disk.");
      (attach as (b: Uint8Array) => void).call(machine, image);
    }
    phase("booting", `${disk.kind === "cd" ? "CD-ROM" : "Hard disk"}: ${disk.name ?? "image"} (${Math.round(image.byteLength / 1024)} KiB)`);
  }

  // Boot drive selection is consumed by POST, so it must be set before reset.
  call("set_boot_drive", bootDrive);
  call("reset");

  transport = ensureSharedFramebuffer() ? "shared" : "copy";
  copyFrameCounter = 0;
  presentedFrames = 0;

  presentIfDue(true);
  post({
    type: "player.ready",
    variant: variantName,
    transport,
    width: presentWidth,
    height: presentHeight,
    ...(transport === "shared" && sharedFb ? { framebuffer: sharedFb.buffer as SharedArrayBuffer } : {}),
    bootDrive,
  });

  phase("running");
  scheduleLoop(0);
}

// --- message entry point ---------------------------------------------------

ctx.onmessage = (ev: MessageEvent<unknown>) => {
  const raw = ev.data;
  if (!raw || typeof raw !== "object") return;
  const msg = raw as Partial<PlayerInboundMessage> & { type?: unknown };
  if (typeof msg.type !== "string") return;

  switch (msg.type) {
    case "player.start": {
      const token = runToken + 1;
      void start(msg as PlayerStartMessage, token).catch((err) => {
        running = false;
        fail(err);
        phase("error");
      });
      return;
    }
    case "player.stop": {
      stopMachine();
      phase("stopped");
      return;
    }
    case "player.pause": {
      const next = (msg as { paused?: unknown }).paused === true;
      paused = next && running;
      phase(paused ? "paused" : running ? "running" : "stopped");
      if (!paused && running) scheduleLoop(0);
      return;
    }
    case "player.key": {
      const input = msg as { code?: unknown; pressed?: unknown };
      if (typeof input.code !== "string" || !machine) return;
      const inject = machine["inject_browser_key"];
      if (typeof inject !== "function") return;
      try {
        (inject as (code: string, pressed: boolean) => void).call(machine, input.code, input.pressed === true);
      } catch {
        /* ignore malformed key codes */
      }
      return;
    }
    case "player.mouseMotion": {
      const input = msg as { dx?: unknown; dy?: unknown; wheel?: unknown };
      const dx = typeof input.dx === "number" ? input.dx | 0 : 0;
      const dy = typeof input.dy === "number" ? input.dy | 0 : 0;
      const wheel = typeof input.wheel === "number" ? input.wheel | 0 : 0;
      if (!machine || (dx === 0 && dy === 0 && wheel === 0)) return;
      const motion = machine["inject_mouse_motion"];
      const ps2Motion = machine["inject_ps2_mouse_motion"];
      try {
        if (typeof motion === "function") {
          (motion as (dx: number, dy: number, wheel: number) => void).call(machine, dx, dy, wheel);
        } else if (typeof ps2Motion === "function") {
          (ps2Motion as (dx: number, dy: number, wheel: number) => void).call(machine, dx, negateI32Saturating(dy), wheel);
        }
      } catch {
        /* ignore */
      }
      return;
    }
    case "player.mouseButton": {
      const input = msg as { button?: unknown; pressed?: unknown };
      if (!machine) return;
      const button = typeof input.button === "number" ? input.button | 0 : -1;
      if (button < 0) return;
      const inject = machine["inject_mouse_button"];
      if (typeof inject !== "function") return;
      try {
        (inject as (button: number, pressed: boolean) => void).call(machine, button, input.pressed === true);
      } catch {
        /* ignore */
      }
      return;
    }
    default:
      return;
  }
};

// Main-thread controller for the public player page (repo-root `index.html`).
//
// Responsibilities, in order of importance:
//   1. Start the emulator worker and route the guest display to the canvas.
//   2. Forward keyboard/mouse input.
//   3. Keep the surrounding chrome (status, serial log, statistics) honest.
//
// All emulation happens in `workers/player.worker.ts`; this file never touches
// emulator internals, so the guest cannot stall the UI thread.
//
// The page deliberately keeps working without cross-origin isolation (and
// therefore without `SharedArrayBuffer`): frames then arrive as copies at a
// slightly lower rate. That fallback is what makes plain GitHub Pages hosting
// viable, since it cannot send COOP/COEP headers.

import "./player.css";

import { VgaPresenter } from "../display/vga_presenter";
import { FRAMEBUFFER_FORMAT_RGBA8888, wrapSharedFramebuffer } from "../display/framebuffer_protocol";
import { formatOneLineError } from "../../../../packages/transport-safety/src/text.js";
import { buildDemoBootSector } from "./demo_disk";
import {
  isPlayerOutboundMessage,
  type PlayerBootDisk,
  type PlayerInboundMessage,
  type PlayerInputMode,
  type PlayerStartMessage,
} from "./protocol";

// --- DOM -------------------------------------------------------------------

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`player: missing #${id}`);
  return element as T;
}

const canvas = byId<HTMLCanvasElement>("vm-canvas");
const screen = byId<HTMLDivElement>("screen");
const overlay = byId<HTMLDivElement>("screen-overlay");
const overlaySpinner = byId<HTMLDivElement>("overlay-spinner");
const overlayTitle = byId<HTMLHeadingElement>("overlay-title");
const overlayText = byId<HTMLParagraphElement>("overlay-text");
const overlayDetail = byId<HTMLPreElement>("overlay-detail");
const statusPill = byId<HTMLDivElement>("status-pill");
const statusText = byId<HTMLSpanElement>("status-text");
const coiPill = byId<HTMLDivElement>("coi-pill");
const coiText = byId<HTMLSpanElement>("coi-text");
const serialOut = byId<HTMLPreElement>("serial-out");
const bootBanner = byId<HTMLDivElement>("boot-banner");
const fileInput = byId<HTMLInputElement>("file-input");
const metaVariant = byId<HTMLElement>("meta-variant");
const metaDisplay = byId<HTMLElement>("meta-display");
const metaFrames = byId<HTMLElement>("meta-frames");
const metaIps = byId<HTMLElement>("meta-ips");
const metaRam = byId<HTMLElement>("meta-ram");
const metaTransport = byId<HTMLElement>("meta-transport");
const btnReboot = byId<HTMLButtonElement>("btn-reboot");
const btnPause = byId<HTMLButtonElement>("btn-pause");
const btnFullscreen = byId<HTMLButtonElement>("btn-fullscreen");
const btnResetCoi = byId<HTMLButtonElement>("btn-reset-coi");
const btnBootDemo = byId<HTMLButtonElement>("boot-demo");
const btnBootFile = byId<HTMLButtonElement>("boot-file");
const btnBootFirmware = byId<HTMLButtonElement>("boot-firmware");

// --- tunables --------------------------------------------------------------

/** Images are copied into the wasm heap, so the page caps what it accepts. */
const MAX_INMEMORY_IMAGE_BYTES = 128 * 1024 * 1024;
/** Guest RAM for the built-in demo. */
const DEMO_RAM_BYTES = 32 * 1024 * 1024;
/** Guest RAM when the visitor supplies their own image. */
const IMAGE_RAM_BYTES = 64 * 1024 * 1024;
/** Keep the serial panel bounded even for chatty guests. */
const SERIAL_LOG_LIMIT = 60_000;

/**
 * How keystrokes reach the guest; `?input=ps2|bios|both` overrides the default.
 *
 * The default is `bios`: the firmware installs no INT 9 handler, so ps/2 scan
 * codes raised as IRQ1 cannot reach the INT 16h queue that BIOS-era guests poll.
 * `?input=ps2` (or `both`) is there for guests that drive the i8042 themselves.
 */
const inputMode: PlayerInputMode = (() => {
  const requested = new URLSearchParams(location.search).get("input");
  return requested === "ps2" || requested === "bios" || requested === "both" ? requested : "bios";
})();

// --- state -----------------------------------------------------------------

type Phase = "idle" | "starting" | "booting" | "running" | "paused" | "stopped" | "error";

let worker: Worker | null = null;
let presenter: VgaPresenter | null = null;
let phase: Phase = "idle";
let paused = false;
let serialLog = "";
let bootGeneration = 0;
/** What `Reboot` should re-attach. */
let lastBoot: { disk: PlayerBootDisk | null; ramSizeBytes: number } | null = null;

/**
 * Live state mirrored onto `window.__aeroPlayer`.
 *
 * The headless smoke test in CI reads this instead of scraping the DOM: a
 * canvas assertion can only say "some pixels changed", while this can say the
 * guest consumed a keystroke and echoed it on COM1.
 */
const testState: PlayerTestState = {
  ready: false,
  phase: "idle",
  serial: "",
  frames: 0,
  width: 0,
  height: 0,
  transport: "none",
  variant: "",
  error: null,
  transportCopies: 0,
  keysSent: 0,
  inputMode: "bios",
  ips: 0,
};

/** See {@link testState}. */
type PlayerTestState = {
  ready: boolean;
  phase: Phase;
  serial: string;
  frames: number;
  width: number;
  height: number;
  transport: "shared" | "copy" | "none";
  variant: string;
  error: string | null;
  transportCopies: number;
  keysSent: number;
  inputMode: PlayerInputMode;
  /** Last reported guest speed, in instructions/second. */
  ips: number;
};

// --- chrome ----------------------------------------------------------------

function setStatus(next: Phase, detail?: string): void {
  phase = next;
  testState.phase = next;
  const labels: Record<Phase, string> = {
    idle: "idle",
    starting: "starting…",
    booting: "booting…",
    running: "running",
    paused: "paused",
    stopped: "stopped",
    error: "error",
  };
  statusText.textContent = detail ? `${labels[next]} — ${detail}` : labels[next];
  const state =
    next === "running" ? "running" : next === "error" ? "error" : next === "idle" || next === "stopped" ? "off" : "loading";
  statusPill.dataset.state = state;
  statusPill.title = detail ?? labels[next];
}

function showOverlay(options: {
  title: string;
  text?: string;
  detail?: string;
  spinner?: boolean;
  error?: boolean;
}): void {
  overlay.hidden = false;
  overlay.classList.toggle("error", options.error === true);
  overlaySpinner.hidden = options.spinner !== true;
  overlayTitle.textContent = options.title;
  overlayText.textContent = options.text ?? "";
  overlayDetail.textContent = options.detail ?? "";
}

function hideOverlay(): void {
  overlay.hidden = true;
}

function setBanner(text: string, kind: "info" | "error" = "info"): void {
  if (!text) {
    bootBanner.hidden = true;
    bootBanner.textContent = "";
    return;
  }
  bootBanner.hidden = false;
  bootBanner.textContent = text;
  bootBanner.classList.toggle("error", kind === "error");
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function formatRate(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)} GIPS`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)} MIPS`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)} kIPS`;
  return `${Math.round(value)} IPS`;
}

function reportEnvironment(): void {
  const isolated = typeof window.crossOriginIsolated === "boolean" && window.crossOriginIsolated;
  const shared = typeof SharedArrayBuffer !== "undefined";
  if (isolated && shared) {
    coiPill.dataset.state = "on";
    coiText.textContent = "shared memory: on";
  } else if (shared) {
    coiPill.dataset.state = "loading";
    coiText.textContent = "shared memory: waiting for isolation";
  } else {
    coiPill.dataset.state = "off";
    coiText.textContent = "shared memory: off (copy transport)";
  }
}

// --- presenter -------------------------------------------------------------

function ensurePresenter(): VgaPresenter {
  if (presenter) return presenter;
  presenter = new VgaPresenter(canvas, {
    scaleMode: "auto",
    integerScaling: true,
    maxPresentHz: 60,
    clearColor: "#000000",
  });
  presenter.start();
  return presenter;
}

// --- serial ----------------------------------------------------------------

function appendSerial(text: string): void {
  if (!text) return;
  serialLog += text;
  if (serialLog.length > SERIAL_LOG_LIMIT) serialLog = serialLog.slice(-SERIAL_LOG_LIMIT);
  serialOut.textContent = serialLog;
  serialOut.scrollTop = serialOut.scrollHeight;
}

// --- worker ----------------------------------------------------------------

function postToWorker(message: PlayerInboundMessage, transfer: Transferable[] = []): void {
  worker?.postMessage(message, transfer);
}

function teardownWorker(): void {
  const current = worker;
  worker = null;
  if (!current) return;
  try {
    current.postMessage({ type: "player.stop" } satisfies PlayerInboundMessage);
  } catch {
    /* channel may already be closed */
  }
  current.terminate();
}

function handleMessage(data: unknown): void {
  if (!isPlayerOutboundMessage(data)) return;

  if (data.type === "player.phase") {
    setStatus(data.phase, data.detail);
    if (data.phase === "running") hideOverlay();
    if (data.phase === "error") {
      testState.error = data.detail ?? "guest error";
      showOverlay({
        title: "The guest stopped",
        text: "Something went wrong while running the emulator.",
        detail: data.detail,
        spinner: false,
        error: true,
      });
    }
    return;
  }

  if (data.type === "player.ready") {
    testState.ready = true;
    testState.variant = data.variant;
    testState.transport = data.transport;
    metaVariant.textContent = data.variant;
    if (data.width > 0 && data.height > 0) {
      metaDisplay.textContent = `${data.width} × ${data.height}`;
      testState.width = data.width;
      testState.height = data.height;
    }
    const p = ensurePresenter();
    if (data.transport === "shared" && data.framebuffer) {
      p.setSharedFramebuffer(wrapSharedFramebuffer(data.framebuffer, 0));
      metaTransport.textContent = "shared";
      coiPill.dataset.state = "on";
      coiText.textContent = "shared memory: on";
    } else {
      p.setSharedFramebuffer(null);
      metaTransport.textContent = "copy";
    }
    return;
  }

  if (data.type === "player.frame") {
    testState.transportCopies += 1;
    const p = ensurePresenter();
    p.pushCopyFrame({
      width: data.width,
      height: data.height,
      strideBytes: data.strideBytes,
      format: FRAMEBUFFER_FORMAT_RGBA8888,
      frameCounter: data.frameCounter,
      pixelsU8: new Uint8Array(data.pixels),
    });
    return;
  }

  if (data.type === "player.serial") {
    appendSerial(data.text);
    testState.serial = serialLog;
    return;
  }

  if (data.type === "player.stats") {
    testState.frames = data.frames;
    testState.ips = data.instructionsPerSecond;
    testState.width = data.width || testState.width;
    testState.height = data.height || testState.height;
    metaFrames.textContent = String(data.frames);
    metaIps.textContent = formatRate(data.instructionsPerSecond);
    if (data.width > 0 && data.height > 0) metaDisplay.textContent = `${data.width} × ${data.height}`;
    return;
  }

  if (data.type === "player.error") {
    testState.error = data.message;
    setStatus("error", "worker error");
    showOverlay({
      title: "The emulator could not start",
      text: "This usually means the WASM bundle is missing or stale. Rebuilding the site fixes it.",
      detail: data.message,
      spinner: false,
      error: true,
    });
  }
}

function startMachine(disk: PlayerBootDisk | null, ramSizeBytes: number): void {
  teardownWorker();
  bootGeneration += 1;
  const generation = bootGeneration;

  paused = false;
  btnPause.textContent = "Pause";
  serialLog = "";
  serialOut.textContent = "";
  testState.ready = false;
  testState.serial = "";
  testState.frames = 0;
  testState.error = null;
  metaFrames.textContent = "0";
  metaIps.textContent = "—";
  metaRam.textContent = formatBytes(ramSizeBytes);
  metaVariant.textContent = "—";
  metaTransport.textContent = "—";
  lastBoot = { disk, ramSizeBytes };

  showOverlay({ title: "Powering on…", text: "Instantiating the emulator core.", spinner: true });
  setStatus("starting");

  const w = new Worker(new URL("../workers/player.worker.ts", import.meta.url), { type: "module" });
  worker = w;
  w.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (generation !== bootGeneration) return;
    handleMessage(event.data);
  });
  w.addEventListener("error", (event: ErrorEvent) => {
    if (generation !== bootGeneration) return;
    setStatus("error", "worker crashed");
    showOverlay({
      title: "The emulator worker failed",
      text: "The worker could not be loaded. If you are running from a file:// URL, serve the site over HTTP.",
      detail: event.message || String(event.error ?? ""),
      spinner: false,
      error: true,
    });
  });

  const message: PlayerStartMessage = {
    type: "player.start",
    ramSizeBytes,
    disk,
    display: "vga",
    // The INT 16h queue by default (see `inputMode`); `?input=` switches it.
    input: inputMode,
  };
  w.postMessage(message);
}

function startDemo(): void {
  const sector = buildDemoBootSector();
  const bytes = sector.buffer.slice(sector.byteOffset, sector.byteOffset + sector.byteLength) as ArrayBuffer;
  setBanner("Booting the generated demo disk: 512 bytes of real-mode code.");
  startMachine({ kind: "hdd", bytes, name: "demo-sector.bin" }, DEMO_RAM_BYTES);
}

function startFirmwareOnly(): void {
  setBanner("No disk attached. The firmware will POST and then look for a boot device.");
  startMachine(null, DEMO_RAM_BYTES);
}

function startImage(bytes: ArrayBuffer, name: string): void {
  if (bytes.byteLength === 0) {
    setBanner(`${name} is empty.`, "error");
    return;
  }
  if (bytes.byteLength > MAX_INMEMORY_IMAGE_BYTES) {
    setBanner(
      `${name} is ${formatBytes(bytes.byteLength)}; this page keeps images in guest memory and tops out at ` +
        `${formatBytes(MAX_INMEMORY_IMAGE_BYTES)}. Use the bring-up shell for larger media.`,
      "error",
    );
    return;
  }
  // 512-byte multiples only: the ATA model rejects anything else.
  let image = bytes;
  if (image.byteLength % 512 !== 0) {
    const padded = new Uint8Array(Math.ceil(image.byteLength / 512) * 512);
    padded.set(new Uint8Array(image));
    image = padded.buffer;
  }
  const kind: "hdd" | "cd" = /\.(iso|cdr)(?:$|\?)/i.test(name) ? "cd" : "hdd";
  setBanner(`Attached ${name} (${formatBytes(image.byteLength)}) as ${kind === "cd" ? "a CD-ROM" : "a hard disk"}.`);
  startMachine({ kind, bytes: image, name }, IMAGE_RAM_BYTES);
}

async function loadFile(file: File): Promise<void> {
  setBanner(`Reading ${file.name} (${formatBytes(file.size)})…`);
  try {
    const bytes = await file.arrayBuffer();
    startImage(bytes, file.name);
  } catch (err) {
    setBanner(`Could not read ${file.name}: ${formatOneLineError(err, 200)}`, "error");
  }
}

// --- input -----------------------------------------------------------------

/** Keys the browser must keep for itself. */
const RESERVED_KEYS = new Set(["F5", "F11", "F12"]);

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function inputAllowed(): boolean {
  return phase === "running" || phase === "booting";
}

window.addEventListener("keydown", (event) => {
  if (!inputAllowed() || isTypingTarget(event.target)) return;
  if (RESERVED_KEYS.has(event.code)) return;
  // Leave browser shortcuts (Ctrl/Cmd) alone, except for the one chord that is
  // conventionally "reboot the guest".
  if (event.ctrlKey && event.altKey && event.code === "Delete") {
    event.preventDefault();
    reboot();
    return;
  }
  if (event.ctrlKey || event.metaKey) return;

  postToWorker({ type: "player.key", code: event.code, pressed: true });
  testState.keysSent += 1;
  event.preventDefault();
});

window.addEventListener("keyup", (event) => {
  if (!inputAllowed() || isTypingTarget(event.target)) return;
  if (event.ctrlKey || event.metaKey) return;
  postToWorker({ type: "player.key", code: event.code, pressed: false });
  if (!RESERVED_KEYS.has(event.code)) event.preventDefault();
});

// Release everything when the page loses focus, otherwise the guest keeps
// believing a modifier is held down.
window.addEventListener("blur", () => {
  for (const code of ["ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight"]) {
    postToWorker({ type: "player.key", code, pressed: false });
  }
});

let pointerLocked = false;

canvas.addEventListener("mousemove", (event) => {
  if (!inputAllowed()) return;
  const dx = event.movementX | 0;
  const dy = event.movementY | 0;
  if (dx === 0 && dy === 0) return;
  postToWorker({ type: "player.mouseMotion", dx, dy, wheel: 0 });
});

canvas.addEventListener("mousedown", (event) => {
  canvas.focus();
  if (!inputAllowed()) return;
  postToWorker({ type: "player.mouseButton", button: event.button, pressed: true });
  if (!pointerLocked && event.button === 0) {
    // Pointer lock makes relative motion usable (a DOS-era mouse has no
    // absolute cursor); ignore failure, motion still works without it.
    void canvas.requestPointerLock?.();
  }
});

canvas.addEventListener("mouseup", (event) => {
  if (!inputAllowed()) return;
  postToWorker({ type: "player.mouseButton", button: event.button, pressed: false });
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  if (!inputAllowed()) return;
  const wheel = Math.sign(event.deltaY) * -1;
  if (wheel === 0) return;
  postToWorker({ type: "player.mouseMotion", dx: 0, dy: 0, wheel });
  event.preventDefault();
});

document.addEventListener("pointerlockchange", () => {
  pointerLocked = document.pointerLockElement === canvas;
});
document.addEventListener("pointerlockerror", () => {
  pointerLocked = false;
});

// --- controls --------------------------------------------------------------

function reboot(): void {
  if (!lastBoot) {
    startDemo();
    return;
  }
  // The worker owns the machine; a fresh boot is both simpler and more honest
  // than trying to reset a wedged guest in place.
  startMachine(lastBoot.disk, lastBoot.ramSizeBytes);
}

btnBootDemo.addEventListener("click", () => startDemo());
btnBootFirmware.addEventListener("click", () => startFirmwareOnly());
btnBootFile.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (file) void loadFile(file);
});

btnReboot.addEventListener("click", () => reboot());

btnPause.addEventListener("click", () => {
  if (!worker) return;
  paused = !paused;
  btnPause.textContent = paused ? "Resume" : "Pause";
  postToWorker({ type: "player.pause", paused });
});

btnFullscreen.addEventListener("click", () => {
  if (document.fullscreenElement) {
    void document.exitFullscreen();
    return;
  }
  void screen.requestFullscreen?.().catch((err: unknown) => {
    setBanner(`Fullscreen was refused: ${formatOneLineError(err, 160)}`, "error");
  });
});

btnResetCoi.addEventListener("click", async () => {
  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((registration) => registration.unregister()));
    for (const key of await caches.keys()) await caches.delete(key);
    sessionStorage.removeItem("aero.coiReloaded");
  } catch {
    /* best effort */
  }
  location.reload();
});

// --- boot ------------------------------------------------------------------

window.addEventListener("unhandledrejection", (event) => {
  console.error("[aero] unhandled rejection", event.reason);
});

reportEnvironment();
ensurePresenter();

if (new URLSearchParams(location.search).get("noboot") === "1") {
  showOverlay({ title: "Ready", text: "Pick a boot medium to start the machine.", spinner: false });
  setStatus("idle");
  setBanner("Nothing booted yet.");
} else {
  try {
    startDemo();
  } catch (err) {
    setStatus("error", "boot failed");
    showOverlay({
      title: "Could not boot the demo disk",
      detail: formatOneLineError(err, 500),
      spinner: false,
      error: true,
    });
  }
}

// Expose live state for debugging and for the headless smoke test. The page
// itself never reads it.
declare global {
  interface Window {
    __aeroPlayer?: PlayerTestState;
  }
}

testState.inputMode = inputMode;
window.__aeroPlayer = testState;

export {};

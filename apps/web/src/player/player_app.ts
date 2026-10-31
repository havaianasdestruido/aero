// Main-thread controller for the public player page.
//
// Responsibilities, in order of importance:
//  1. Boot something on arrival (the generated demo disk), so a visitor sees a
//     live computer without configuring anything.
//  2. Let the visitor choose other boot media — a bundled DOS image or a local
//     disk image — and stay out of the way while it boots.
//  3. Present the guest display and forward keyboard/mouse input.
//
// The guest itself never runs here: `workers/player.worker.ts` owns the machine.
// This file only talks to it over the small protocol in `./protocol.ts`.

import "./player.css";

import { VgaPresenter } from "../display/vga_presenter";
import { FRAMEBUFFER_FORMAT_RGBA8888, wrapSharedFramebuffer } from "../display/framebuffer_protocol";
import { formatOneLineError } from "../../../../packages/transport-safety/src/text.js";

import { buildDemoBootSector } from "./demo_disk";
import { isPlayerOutboundMessage, type PlayerBootDisk, type PlayerInboundMessage, type PlayerStartMessage } from "./protocol";

const MAX_INMEMORY_IMAGE_BYTES = 128 * 1024 * 1024;
const DEMO_RAM_BYTES = 64 * 1024 * 1024;
const SERIAL_LOG_LIMIT = 24_000;

type Phase = "idle" | "starting" | "booting" | "running" | "paused" | "stopped" | "error";

/** State mirrored onto `window.__aeroPlayer` for headless smoke tests. */
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
};

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
};

declare global {
  interface Window {
    __aeroPlayer?: PlayerTestState;
  }
}

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
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
const btnBootFreedos = byId<HTMLButtonElement>("boot-freedos");
const btnBootFile = byId<HTMLButtonElement>("boot-file");
const btnBootFirmware = byId<HTMLButtonElement>("boot-firmware");

const presenter = new VgaPresenter(canvas, {
  scaleMode: "pixelated",
  integerScaling: false,
  autoResizeToClient: true,
  maxPresentHz: 60,
});

let worker: Worker | null = null;
let currentDisk: PlayerBootDisk | null = null;
let pendingDiskPromise: Promise<PlayerBootDisk | null> | null = null;
let paused = false;
let running = false;
let serialBuffer = "";

// --- small helpers ---------------------------------------------------------

function setPhase(next: Phase, detail?: string): void {
  testState.phase = next;
  statusPill.dataset.state = next === "booting" || next === "starting" ? "booting" : next;
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
  btnPause.disabled = !running;
  btnPause.textContent = paused ? "Resume" : "Pause";
}

function showOverlay(opts: { title: string; text?: string; detail?: string; spinner?: boolean; error?: boolean }): void {
  overlay.hidden = false;
  overlay.classList.toggle("error", opts.error === true);
  overlaySpinner.hidden = opts.spinner !== true;
  overlayTitle.textContent = opts.title;
  overlayText.textContent = opts.text ?? "";
  overlayDetail.textContent = opts.detail ?? "";
}

function hideOverlay(): void {
  overlay.hidden = true;
}

function showBanner(message: string, kind: "info" | "error" = "info"): void {
  bootBanner.hidden = false;
  bootBanner.classList.toggle("error", kind === "error");
  bootBanner.textContent = message;
}

function hideBanner(): void {
  bootBanner.hidden = true;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

function formatRate(ips: number): string {
  if (ips >= 1_000_000_000) return `${(ips / 1_000_000_000).toFixed(2)} GIPS`;
  if (ips >= 1_000_000) return `${(ips / 1_000_000).toFixed(1)} MIPS`;
  if (ips >= 1_000) return `${(ips / 1_000).toFixed(1)} KIPS`;
  return `${ips} IPS`;
}

// --- worker plumbing -------------------------------------------------------

function postToWorker(msg: PlayerInboundMessage, transfer: Transferable[] = []): void {
  const w = worker;
  if (!w) return;
  try {
    if (transfer.length > 0) w.postMessage(msg, transfer);
    else w.postMessage(msg);
  } catch (err) {
    console.warn("[aero] failed to post message to worker", err);
  }
}

function stopWorker(): void {
  const w = worker;
  worker = null;
  running = false;
  paused = false;
  if (!w) return;
  try {
    w.postMessage({ type: "player.stop" } satisfies PlayerInboundMessage);
  } catch {
    /* ignore */
  }
  try {
    w.terminate();
  } catch {
    /* ignore */
  }
  presenter.setSharedFramebuffer(null);
  testState.transport = "none";
}

function handleWorkerMessage(data: unknown): void {
  if (!isPlayerOutboundMessage(data)) return;

  switch (data.type) {
    case "player.phase": {
      setPhase(data.phase as Phase, data.detail);
      if (data.phase === "starting" || data.phase === "booting") {
        showOverlay({ title: "Powering on…", text: data.detail, spinner: true });
      } else if (data.phase === "running") {
        hideOverlay();
        running = true;
      } else if (data.phase === "error") {
        showOverlay({ title: "The machine stopped", text: data.detail ?? "", spinner: false, error: true });
      }
      return;
    }

    case "player.ready": {
      testState.ready = true;
      testState.variant = data.variant;
      testState.transport = data.transport;
      metaVariant.textContent = data.variant;
      metaTransport.textContent = data.transport === "shared" ? "shared memory" : "copied frames";
      if (data.width > 0 && data.height > 0) {
        metaDisplay.textContent = `${data.width} × ${data.height}`;
      }
      if (data.framebuffer && typeof SharedArrayBuffer !== "undefined") {
        try {
          presenter.setSharedFramebuffer(wrapSharedFramebuffer(data.framebuffer, 0));
          metaTransport.textContent = "shared memory";
        } catch (err) {
          console.warn("[aero] shared framebuffer unusable, waiting for copies", err);
          presenter.setSharedFramebuffer(null);
          metaTransport.textContent = "copied frames";
        }
      } else {
        presenter.setSharedFramebuffer(null);
        metaTransport.textContent = "copied frames";
      }
      return;
    }

    case "player.frame": {
      presenter.pushCopyFrame({
        width: data.width,
        height: data.height,
        strideBytes: data.strideBytes,
        format: FRAMEBUFFER_FORMAT_RGBA8888,
        frameCounter: data.frameCounter,
        pixelsU8: new Uint8Array(data.pixels),
      });
      testState.transportCopies += 1;
      return;
    }

    case "player.serial": {
      serialBuffer += data.text;
      if (serialBuffer.length > SERIAL_LOG_LIMIT) serialBuffer = serialBuffer.slice(-SERIAL_LOG_LIMIT);
      testState.serial = serialBuffer;
      serialOut.textContent = serialBuffer;
      serialOut.scrollTop = serialOut.scrollHeight;
      return;
    }

    case "player.stats": {
      testState.frames = data.frames;
      testState.width = data.width;
      testState.height = data.height;
      metaFrames.textContent = String(data.frames);
      metaIps.textContent = formatRate(data.instructionsPerSecond);
      if (data.width > 0 && data.height > 0) {
        metaDisplay.textContent = `${data.width} × ${data.height}`;
      }
      return;
    }

    case "player.error": {
      testState.error = data.message;
      setPhase("error");
      showOverlay({
        title: "The emulator hit an error",
        text: "This is almost always a missing or stale WASM bundle.",
        detail: data.message,
        spinner: false,
        error: true,
      });
      return;
    }

    default:
      return;
  }
}

async function bootWith(disk: PlayerBootDisk | null, opts: { ramSizeBytes?: number } = {}): Promise<void> {
  stopWorker();
  hideBanner();
  serialBuffer = "";
  serialOut.textContent = "";
  testState.serial = "";
  testState.error = null;
  testState.ready = false;
  testState.frames = 0;
  testState.transportCopies = 0;

  currentDisk = disk;
  running = false;
  paused = false;
  setPhase("starting");

  const ramBytes = opts.ramSizeBytes ?? DEMO_RAM_BYTES;
  metaRam.textContent = formatBytes(ramBytes);
  metaFrames.textContent = "0";
  metaIps.textContent = "—";
  metaTransport.textContent = "—";
  metaVariant.textContent = "—";

  const message: PlayerStartMessage = {
    type: "player.start",
    ramSizeBytes: ramBytes,
    disk,
    display: "vga",
  };

  const w = new Worker(new URL("../workers/player.worker.ts", import.meta.url), { type: "module" });
  worker = w;
  let sawReady = false;
  w.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (worker !== w) return;
    if (!sawReady && isPlayerOutboundMessage(event.data) && event.data.type === "player.ready") sawReady = true;
    handleWorkerMessage(event.data);
  });
  w.addEventListener("messageerror", () => {
    if (worker !== w) return;
    testState.error = "worker message could not be deserialised";
    showOverlay({ title: "Worker failure", text: testState.error, spinner: false, error: true });
  });
  w.addEventListener("error", (event: ErrorEvent) => {
    if (worker !== w) return;
    const message = event.message || "worker crashed";
    testState.error = message;
    setPhase("error");
    showOverlay({
      title: "The emulator worker crashed",
      text: "Reload the page. If it persists, the deployed WASM bundle may be stale.",
      detail: formatOneLineError(message, 512),
      spinner: false,
      error: true,
    });
  });

  const transfer: Transferable[] = disk ? [disk.bytes] : [];
  postToWorker(message, transfer);
  presentSoon();
}

function presentSoon(): void {
  presenter.start();
}

// --- boot media ------------------------------------------------------------

function bootDemo(): void {
  const sector = buildDemoBootSector();
  void bootWith({ kind: "hdd", bytes: sector.buffer as ArrayBuffer, name: "aero-demo.img" });
}

function bootFirmwareOnly(): void {
  void bootWith(null);
}

async function bootFromFile(file: File): Promise<void> {
  if (file.size === 0) {
    showBanner("That file is empty.", "error");
    return;
  }
  if (file.size > MAX_INMEMORY_IMAGE_BYTES) {
    showBanner(
      `That image is ${formatBytes(file.size)}. The player keeps the whole image in guest memory, so it is capped at ${formatBytes(
        MAX_INMEMORY_IMAGE_BYTES,
      )}. Use the bring-up shell for larger media (streamed from OPFS).`,
      "error",
    );
    return;
  }

  showBanner(`Reading ${file.name} (${formatBytes(file.size)})…`);
  const bytes = await file.arrayBuffer();
  const kind: PlayerBootDisk["kind"] = /\.iso$/i.test(file.name) ? "cd" : "hdd";
  await bootWith({ kind, bytes, name: file.name });
  showBanner(
    kind === "cd"
      ? `${file.name} attached as ATAPI CD-ROM; the firmware boots CD first while a disc is present.`
      : `${file.name} attached as the primary ATA disk.`,
  );
}

/** Where the deployed site keeps optional demo media (copied in by CI). */
function demoMediaUrl(name: string): string {
  const base = import.meta.env.BASE_URL ?? "/";
  return new URL(`${base.replace(/\/?$/, "/")}demo/${name}`, location.href).href;
}

async function bootBundledFreeDos(): Promise<void> {
  if (pendingDiskPromise) return;
  const url = demoMediaUrl("freedos.img");
  showBanner("Downloading FreeDOS (1.4 MB)…");
  const promise = (async (): Promise<PlayerBootDisk | null> => {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
    const bytes = await response.arrayBuffer();
    return { kind: "hdd", bytes, name: "freedos.img" };
  })();
  pendingDiskPromise = promise;
  try {
    const disk = await promise;
    if (!disk) return;
    await bootWith(disk, { ramSizeBytes: 64 * 1024 * 1024 });
    showBanner("FreeDOS attached. It is a 1.44 MB floppy image mounted as a hard disk; if it does not boot, try the demo disk.");
  } catch (err) {
    showBanner(`Could not load FreeDOS: ${formatOneLineError(err, 300)}`, "error");
  } finally {
    pendingDiskPromise = null;
  }
}

/** Show the FreeDOS option only when the deployed site actually ships the image. */
async function probeBundledMedia(): Promise<void> {
  try {
    const response = await fetch(demoMediaUrl("freedos.img"), { method: "HEAD" });
    if (!response.ok) return;
    btnBootFreedos.hidden = false;
    const size = Number(response.headers.get("content-length") ?? "0");
    if (size > 0) byId<HTMLSpanElement>("boot-freedos-sub").textContent = `A real 16-bit OS · ${formatBytes(size)} download`;
  } catch {
    /* optional */
  }
}

// --- input -----------------------------------------------------------------

const BROWSER_OWNED_KEYS = new Set(["F5", "F11", "F12"]);

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function reboot(): void {
  if (currentDisk) void bootWith(currentDisk);
  else bootFirmwareOnly();
}

window.addEventListener("keydown", (event) => {
  if (!running) return;
  if (isTypingTarget(event.target)) return;

  // Ctrl+Alt+Del is intercepted by the browser on many platforms, so offer the
  // guest reset through an explicit chord instead.
  if (event.ctrlKey && event.altKey && event.code === "Delete") {
    event.preventDefault();
    reboot();
    return;
  }

  if (event.metaKey || event.ctrlKey) return; // leave browser shortcuts alone
  if (BROWSER_OWNED_KEYS.has(event.code)) return;
  if (event.repeat) {
    event.preventDefault();
    return;
  }

  postToWorker({ type: "player.key", code: event.code, pressed: true });
  event.preventDefault();
});

window.addEventListener("keyup", (event) => {
  if (!running) return;
  if (isTypingTarget(event.target)) return;
  if (event.metaKey || event.ctrlKey) {
    if (event.code !== "ControlLeft" && event.code !== "ControlRight") return;
  }
  postToWorker({ type: "player.key", code: event.code, pressed: false });
  if (!BROWSER_OWNED_KEYS.has(event.code)) event.preventDefault();
});

canvas.addEventListener("mousemove", (event) => {
  if (!running) return;
  const dx = event.movementX | 0;
  const dy = event.movementY | 0;
  if (dx === 0 && dy === 0) return;
  postToWorker({ type: "player.mouseMotion", dx, dy, wheel: 0 });
});

canvas.addEventListener("wheel", (event) => {
  if (!running) return;
  const wheel = -Math.sign(event.deltaY) | 0;
  if (wheel === 0) return;
  postToWorker({ type: "player.mouseMotion", dx: 0, dy: 0, wheel });
  event.preventDefault();
});

canvas.addEventListener("mousedown", (event) => {
  canvas.focus();
  if (!running) return;
  postToWorker({ type: "player.mouseButton", button: event.button | 0, pressed: true });
});

window.addEventListener("mouseup", (event) => {
  if (!running) return;
  postToWorker({ type: "player.mouseButton", button: event.button | 0, pressed: false });
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("dblclick", () => {
  const requestLock = (canvas as unknown as { requestPointerLock?: () => void }).requestPointerLock;
  if (typeof requestLock !== "function") return;
  try {
    requestLock.call(canvas);
  } catch {
    /* ignore */
  }
});

canvas.addEventListener("click", () => {
  canvas.focus();
  screen.classList.add("capture");
  window.setTimeout(() => screen.classList.remove("capture"), 1200);
});

// --- controls --------------------------------------------------------------

btnReboot.addEventListener("click", () => reboot());
btnPause.addEventListener("click", () => {
  if (!running) return;
  paused = !paused;
  postToWorker({ type: "player.pause", paused });
  setPhase(paused ? "paused" : "running");
});
btnFullscreen.addEventListener("click", () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void screen.requestFullscreen?.();
});
btnBootDemo.addEventListener("click", () => bootDemo());
btnBootFirmware.addEventListener("click", () => bootFirmwareOnly());
btnBootFile.addEventListener("click", () => fileInput.click());
btnBootFreedos.addEventListener("click", () => void bootBundledFreeDos());
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (file) void bootFromFile(file);
});
btnResetCoi.addEventListener("click", () => {
  try {
    sessionStorage.removeItem("aero.coiReloaded");
  } catch {
    /* ignore */
  }
  const url = new URL(location.href);
  url.searchParams.set("no-coi", "1");
  location.href = url.href;
});

// --- environment readout ---------------------------------------------------

function reportEnvironment(): void {
  const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  const hasSab = typeof SharedArrayBuffer !== "undefined";
  const canShare = isolated && hasSab;
  coiPill.dataset.state = canShare ? "running" : "idle";
  coiText.textContent = canShare
    ? "shared memory: on"
    : hasSab
      ? "shared memory: not isolated"
      : "shared memory: unavailable";
  coiPill.title = canShare
    ? "Cross-origin isolated: the emulator publishes frames through SharedArrayBuffer."
    : "Not cross-origin isolated: the isolation service worker may still be installing, or this browser blocks it. Frames are copied instead.";
}

// --- start -----------------------------------------------------------------

window.__aeroPlayer = testState;
presenter.start();
reportEnvironment();
setPhase("idle");

const params = new URLSearchParams(location.search);
const autoBoot = params.get("noboot") !== "1" && params.get("noboot") !== "true";

// `?disk=<url>` boots a same-site image directly, which is handy for sharing a
// link to a specific guest without re-implementing the UI.
const diskParam = params.get("disk");
if (diskParam) {
  showBanner(`Loading ${diskParam}…`);
  void (async () => {
    try {
      const response = await fetch(diskParam);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength > MAX_INMEMORY_IMAGE_BYTES) throw new Error(`image is ${formatBytes(bytes.byteLength)} (max ${formatBytes(MAX_INMEMORY_IMAGE_BYTES)})`);
      await bootWith({ kind: /\.iso(?:$|\?)/i.test(diskParam) ? "cd" : "hdd", bytes, name: diskParam });
      showBanner(`${diskParam} attached.`);
    } catch (err) {
      showBanner(`Could not load ${diskParam}: ${formatOneLineError(err, 300)}`, "error");
    }
  })();
} else if (autoBoot) {
  bootDemo();
} else {
  setPhase("stopped");
  showOverlay({
    title: "Ready",
    text: "Pick a boot medium on the right, or press Reboot to start the demo disk.",
    spinner: false,
  });
}

void probeBundledMedia();

// Message protocol between the player UI (`play_main.ts`) and the emulator
// worker (`workers/player.worker.ts`).
//
// The player runs a full `aero_machine::Machine` inside a dedicated worker: the
// guest executes there, the display is published as a framebuffer (shared memory
// when the page is cross-origin isolated, plain copies otherwise), and the UI
// thread only paints and forwards input.

/** A boot medium supplied by the page (already downloaded, or generated). */
export type PlayerBootDisk = {
  /** `hdd` attaches the bytes as the canonical ATA disk, `cd` as ATAPI media. */
  readonly kind: "hdd" | "cd";
  /** Raw image bytes. Transferred, so the page must not reuse the buffer. */
  readonly bytes: ArrayBuffer;
  /** Human-readable label, shown in the UI/log. */
  readonly name?: string;
};

export type PlayerStartMessage = {
  readonly type: "player.start";
  /**
   * Guest RAM in bytes. Defaults to 64 MiB: enough for DOS-era guests and for
   * the firmware's own structures, while keeping the wasm heap modest.
   */
  readonly ramSizeBytes?: number;
  /** Boot medium. `null`/omitted boots the firmware with no disk attached. */
  readonly disk?: PlayerBootDisk | null;
  /**
   * Which display model to instantiate.
   *
   * `vga` (default) is the classic Bochs-compatible VGA: text modes, VBE, the
   * memory-mapped `0xB8000` window. `aerogpu` selects the experimental AeroGPU
   * device, which needs a driver in the guest and is therefore not useful for
   * DOS-era media.
   */
  readonly display?: "vga" | "aerogpu";
};

export type PlayerInboundMessage =
  | PlayerStartMessage
  | { readonly type: "player.stop" }
  | { readonly type: "player.pause"; readonly paused: boolean }
  | { readonly type: "player.key"; readonly code: string; readonly pressed: boolean }
  | { readonly type: "player.mouseMotion"; readonly dx: number; readonly dy: number; readonly wheel?: number }
  | { readonly type: "player.mouseButton"; readonly button: number; readonly pressed: boolean };

export type PlayerPhase = "starting" | "booting" | "running" | "paused" | "stopped" | "error";

export type PlayerOutboundMessage =
  /** Lifecycle transitions, for the status line. */
  | { readonly type: "player.phase"; readonly phase: PlayerPhase; readonly detail?: string }
  /** Emitted once the machine is constructed and the run loop is live. */
  | {
      readonly type: "player.ready";
      readonly variant: string;
      readonly transport: "shared" | "copy";
      readonly width: number;
      readonly height: number;
      readonly framebuffer?: SharedArrayBuffer;
      readonly bootDrive: number;
    }
  /** Copy-transport frames (no SharedArrayBuffer available). */
  | {
      readonly type: "player.frame";
      readonly width: number;
      readonly height: number;
      readonly strideBytes: number;
      readonly frameCounter: number;
      readonly pixels: ArrayBuffer;
    }
  /** Decoded COM1 output. */
  | { readonly type: "player.serial"; readonly text: string }
  /** Periodic telemetry for the status bar. */
  | {
      readonly type: "player.stats";
      readonly instructionsPerSecond: number;
      readonly executed: number;
      readonly frames: number;
      readonly width: number;
      readonly height: number;
      readonly uptimeMs: number;
    }
  | { readonly type: "player.error"; readonly message: string };

/** Narrow an arbitrary `MessageEvent.data` value to the worker->page protocol. */
export function isPlayerOutboundMessage(value: unknown): value is PlayerOutboundMessage {
  if (!value || typeof value !== "object") return false;
  const type = (value as { type?: unknown }).type;
  return (
    type === "player.phase" ||
    type === "player.ready" ||
    type === "player.frame" ||
    type === "player.serial" ||
    type === "player.stats" ||
    type === "player.error"
  );
}

# Aero

A Windows 7 SP1 (32/64-bit) system emulator that runs entirely in the
browser: Rust → WASM core, Web Workers + SharedArrayBuffer, D3D9/10/11 →
WebGPU translation, virtio Windows guest drivers.

**This repository is greenfield and agent-owned.** The wiki is the source of
truth for what the project is, where it stands, and how work happens here:

- Start: [`wiki/README.md`](./wiki/README.md)
- The system in one page: [`wiki/overview.md`](./wiki/overview.md)
- Repo state & structure: [`wiki/state/repo-state-and-structure.md`](./wiki/state/repo-state-and-structure.md)
- Resume the Windows 7 bring-up: [`wiki/state/repo-state-and-structure.md`](./wiki/state/repo-state-and-structure.md)
- Working agreements: [`wiki/meta/working-agreements.md`](./wiki/meta/working-agreements.md)
- Agents: see [`AGENTS.md`](./AGENTS.md)

## Play it in a browser

The emulator is published as a **static site** — no server, no local build
step, nothing to install:

**https://havaianasdestruido.github.io/aero/**

That page boots a complete x86 machine: the emulator core (Rust → WebAssembly)
runs in a Web Worker, the guest's VGA output is painted on a canvas, and the
keyboard and mouse are wired straight into the emulated hardware. The demo disk
is a 512-byte boot sector generated in the page (see
`apps/web/src/player/demo_disk.ts`), so it works offline and instantly; you can
also mount your own IMG/ISO, or open the bring-up shell from the same page to
attach larger media.

### How the site is built and published

`.github/workflows/pages.yml` runs three jobs on every push to `main` (and to
`arena/**` branches):

1. **Build WASM** — compiles `aero-wasm` twice, single-threaded and threaded,
   with `wasm-pack`, and uploads the packages as an artifact.
2. **Build site** — downloads that artifact, runs `pnpm run build` (Vite,
   with `VITE_BASE=/aero/` so asset URLs resolve under the project sub-path),
   then boots the *built* artifact in headless Chromium and asserts that the
   guest comes up: the player becomes ready, frames are presented, the bundled
   demo announces itself on COM1, and typed keys reach the guest and are echoed
   back (`scripts/ci/player-smoke.mjs`). The same test re-runs the page without
   `COOP`/`COEP` headers to prove the copy-framebuffer fallback works on a host
   that cannot set headers.
3. **Deploy** — publishes `dist/` to GitHub Pages.

One-time setup: the repository must have Pages enabled with **Source: GitHub
Actions** (Settings → Pages). After that, every push publishes automatically.

### Cross-origin isolation on a static host

Shared memory (`SharedArrayBuffer`) needs the `COOP`/`COEP` response headers,
which GitHub Pages cannot send. `apps/web/public/coi-serviceworker.js` re-issues
same-origin responses with those headers, so the deployed page becomes
cross-origin isolated after a one-time reload and shares frames with zero
copies. When that is not possible the player falls back to copying frames, and
everything still works: `?no-coi=1` forces the fallback, and *Reload clean*
unregisters the worker if a stale one is ever suspected.

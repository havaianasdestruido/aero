// NOTE: This Vite config is for the *repo-root Vite app* (canonical).
//
// It exists primarily for:
// - Playwright E2E that exercises low-level primitives (workers, COOP/COEP, etc.)
// - Importing source modules across the repo (e.g. `/apps/web/src/...`) in a browser context
//
// The `web/` directory contains shared runtime modules and WASM build tooling.
// Its Vite entrypoint (`web/index.html`) is legacy/experimental.
//
// This config is also responsible for emitting `aero.version.json` in production builds,
// which is used for provenance/debugging in deployed artifacts.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Connect, type Plugin } from 'vite';

import {
  baselineSecurityHeaders,
  crossOriginIsolationHeaders,
  cspHeaders,
} from './scripts/security_headers.mjs';

const coopCoepSetting = (process.env.VITE_DISABLE_COOP_COEP ?? '').toLowerCase();
const coopCoepDisabled = coopCoepSetting === '1' || coopCoepSetting === 'true';

type AeroBuildInfo = Readonly<{
  version: string;
  gitSha: string;
  builtAt: string;
}>;

const rootDir = fileURLToPath(new URL('.', import.meta.url));

function resolveGitSha(): string {
  const fromEnv = process.env.GIT_SHA || process.env.GITHUB_SHA;
  if (fromEnv && fromEnv.trim().length > 0) return fromEnv.trim();

  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: rootDir, encoding: 'utf8' }).trim();
  } catch {
    return 'dev';
  }
}

function resolveBuildTimestamp(): string {
  const explicit = process.env.BUILD_TIMESTAMP;
  if (explicit && explicit.trim().length > 0) return explicit.trim();

  // Support reproducible builds when SOURCE_DATE_EPOCH is set (common in release pipelines).
  const sourceDateEpoch = process.env.SOURCE_DATE_EPOCH;
  if (sourceDateEpoch && /^\d+$/.test(sourceDateEpoch)) {
    return new Date(Number(sourceDateEpoch) * 1000).toISOString();
  }

  return new Date().toISOString();
}

function resolveVersion(gitSha: string): string {
  const fromEnv = process.env.AERO_VERSION || process.env.GITHUB_REF_NAME;
  if (fromEnv && fromEnv.trim().length > 0) return fromEnv.trim();
  return gitSha.length ? gitSha.slice(0, 12) : 'dev';
}

function aeroBuildInfoPlugin(): Plugin {
  const gitSha = resolveGitSha();
  const buildInfo: AeroBuildInfo = {
    version: resolveVersion(gitSha),
    gitSha,
    builtAt: resolveBuildTimestamp(),
  };

  const jsonBody = `${JSON.stringify(buildInfo, null, 2)}\n`;

  return {
    name: 'aero-build-info',
    config: () => ({
      define: {
        __AERO_BUILD_INFO__: JSON.stringify(buildInfo),
      },
    }),
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'aero.version.json',
        source: jsonBody,
      });
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = req.url?.split('?', 1)[0];
        if (pathname !== '/aero.version.json') return next();
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(jsonBody);
      });
    },
  };
}

function wasmMimeTypePlugin(): Plugin {
  const installWasmMiddleware = (middlewares: Connect.Server) => {
    middlewares.use((req, res, next) => {
      // `instantiateStreaming` requires the correct MIME type.
      const pathname = req.url?.split('?', 1)[0];
      if (pathname?.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
      }
      next();
    });
  };

  return {
    name: 'wasm-mime-type',
    configureServer(server) {
      installWasmMiddleware(server.middlewares);
    },
    configurePreviewServer(server) {
      installWasmMiddleware(server.middlewares);
    },
  };
}

function audioWorkletDependenciesPlugin(): Plugin {
  // Vite treats AudioWorklet modules loaded via `audioWorklet.addModule(new URL(...))` as static
  // assets and does not follow their ESM imports. Our worklets have runtime ESM imports:
  // - `web/src/audio/mic-worklet-processor.js` imports `./mic_ring.js`
  // - `web/src/platform/audio-worklet-processor.js` imports `./audio_worklet_ring_layout.js`
  //
  // Emit copies into `dist/assets/` so the browser can resolve them at runtime.
  const srcMicRingPath = resolve(rootDir, 'apps/web/src/audio/mic_ring.js');
  const source = readFileSync(srcMicRingPath, 'utf8');
  const srcAudioWorkletRingLayoutPath = resolve(rootDir, 'apps/web/src/platform/audio_worklet_ring_layout.js');
  const audioWorkletRingLayoutSource = readFileSync(srcAudioWorkletRingLayoutPath, 'utf8');
  return {
    name: 'aero-audio-worklet-deps',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'assets/mic_ring.js',
        source,
      });
      this.emitFile({
        type: 'asset',
        fileName: 'assets/audio_worklet_ring_layout.js',
        source: audioWorkletRingLayoutSource,
      });
    },
  };
}

function persistentCacheShimPlugin(): Plugin {
  // `wasm-bindgen` supports importing "external modules" via absolute specifiers.
  // `aero-d3d9` uses `#[wasm_bindgen(module = "/js/persistent_cache_shim.js")]`,
  // so we need to ensure that file exists in:
  //  - `vite dev` (served by the dev server)
  //  - `vite build` output (emitted into `dist/`)
  //  - `vite preview` (served by the preview server)
  const srcShimPath = resolve(rootDir, 'crates/aero-d3d9/js/persistent_cache_shim.js');
  const source = readFileSync(srcShimPath, 'utf8');

  const installShimMiddleware = (middlewares: Connect.Server) => {
    middlewares.use((req, res, next) => {
      const pathname = req.url?.split('?', 1)[0];
      if (pathname !== '/js/persistent_cache_shim.js') return next();
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      res.end(source);
    });
  };

  return {
    name: 'aero-persistent-cache-shim',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'js/persistent_cache_shim.js',
        source,
      });
    },
    configureServer(server) {
      installShimMiddleware(server.middlewares);
    },
    configurePreviewServer(server) {
      installShimMiddleware(server.middlewares);
    },
  };
}

// The deployed static site may live under a sub-path (GitHub Pages project
// sites are served from `https://<owner>.github.io/<repo>/`). `VITE_BASE` lets
// the deploy workflow inject that prefix; the default keeps local dev at `/`.
const basePath = (() => {
  const raw = (process.env.VITE_BASE ?? '/').trim();
  if (!raw) return '/';
  if (raw === './' || raw.startsWith('/')) return raw.endsWith('/') ? raw : `${raw}/`;
  return `/${raw.replace(/^\/+/, '').replace(/\/+$/, '')}/`;
})();

export default defineConfig({
  base: basePath,
  assetsInclude: ['**/*.wasm'],
  build: {
    // Ensure `.wasm` is always emitted as a file so `fetch()`/`instantiateStreaming()`
    // behaves consistently across dev/preview/prod.
    assetsInlineLimit: 0,
    rollupOptions: {
      // The harness preview server (port 4173) is used for COOP/COEP + CSP matrix tests.
      // Both application entry points are built so the matrix covers each of them.
      input: {
        // The public player page (what the deployed static site serves at `/`).
        player: fileURLToPath(new URL('./index.html', import.meta.url)),
        main: fileURLToPath(new URL('./apps/web/index.html', import.meta.url)),
        // The fuller shell used for manual Windows 7 bring-up.
        bringup: fileURLToPath(new URL('./apps/web/bringup.html', import.meta.url)),
        // Standalone pages linked from the bring-up UI (keep them available in
        // `vite preview` runs of the harness).
        webusb_diagnostics: fileURLToPath(new URL('./apps/web/webusb_diagnostics.html', import.meta.url)),
        webgl2_fallback_demo: fileURLToPath(new URL('./apps/web/webgl2_fallback_demo.html', import.meta.url)),
        ipc_demo: fileURLToPath(new URL('./apps/web/demo/ipc_demo.html', import.meta.url)),
        vm_boot_vga_serial_smoke: fileURLToPath(new URL('./apps/web/vm-boot-vga-serial-smoke.html', import.meta.url)),
        wddm_scanout_smoke: fileURLToPath(new URL('./apps/web/wddm-scanout-smoke.html', import.meta.url)),
        wddm_scanout_vram_smoke: fileURLToPath(new URL('./apps/web/wddm-scanout-vram-smoke.html', import.meta.url)),
        wddm_scanout_debug: fileURLToPath(new URL('./apps/web/wddm-scanout-debug.html', import.meta.url)),
      },
    },
  },
  // Reuse `web/public` across the repo so test assets and `_headers` templates
  // are consistently available in `vite preview` runs.
  publicDir: 'apps/web/public',
  plugins: [
    aeroBuildInfoPlugin(),
    wasmMimeTypePlugin(),
    audioWorkletDependenciesPlugin(),
    persistentCacheShimPlugin(),
  ],
  // The repo heavily relies on module workers (`type: 'module'` + `import.meta.url`).
  // Keep the harness build aligned with `web/vite.config.ts` so worker bundling
  // supports code-splitting.
  worker: {
    format: 'es',
  },
  server: {
    watch: {
      // Vite's watcher walks the whole project root. This repository's Rust build directory holds
      // well over a million files — past the inotify limit on its own — so watching it does not
      // merely waste descriptors, it takes the dev server down with `ENOSPC` the moment anyone has
      // run a build. Every path here is a build product or a scratch area; none is a source input.
      ignored: [
        "**/target/**",
        "**/node_modules/**",
        "**/dist/**",
        "**/.attic/**",
        "**/test-results/**",
        "**/playwright-report/**",
        "**/.git/**",
      ],
    },
    headers: {
      ...(coopCoepDisabled ? {} : crossOriginIsolationHeaders),
      ...baselineSecurityHeaders,
    },
  },
  preview: {
    headers: {
      ...(coopCoepDisabled ? {} : crossOriginIsolationHeaders),
      ...baselineSecurityHeaders,
      ...cspHeaders,
    },
  },
});

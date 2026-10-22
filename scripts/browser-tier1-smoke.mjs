#!/usr/bin/env node
// Runs Aero's compiled Tier-1 tier inside a real browser and checks what the guest
// computed.
//
// The tier above the engine is the same code the native build runs: block discovery,
// translation, dead-flag elimination and WASM code generation are shared, and only
// the thing that turns those bytes into machine code differs — Wasmtime natively,
// `WebAssembly.Module`/`Instance` here. This asserts on guest registers rather than
// on "it loaded", so a tier that silently compiled nothing fails.
//
//   node scripts/browser-tier1-smoke.mjs [--keep]
//
// `--keep` leaves the built package in apps/web/tier1-browser for poking at by hand.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from '@playwright/test';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pageDir = path.join(repoRoot, 'apps', 'web', 'tier1-browser');

// mov rcx,7 ; add rax,rcx ; mov [rsp-16],rax ; mov rbx,[rsp-16] ; add rax,rbx ; ret
//
// Starting from RAX=5: 5+7 = 12, stored to guest memory, read back, 12+12 = 24. The
// store and the load both go through the inline TLB, so this covers the interesting
// half — a block of pure register arithmetic would pass without guest memory working
// at all.
const EXPECTED = { rax: 24, rcx: 7, retired: 6, compiledBlocks: 1 };

function build() {
  const r = spawnSync(
    'wasm-pack',
    ['build', '--target', 'web', '--out-dir', pageDir, 'crates/aero-jit-wasm'],
    { cwd: repoRoot, stdio: 'inherit' },
  );
  if (r.status !== 0) {
    throw new Error(`wasm-pack build failed (exit ${r.status})`);
  }
}

function serve(root) {
  const types = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.wasm': 'application/wasm',
  };
  return http.createServer((req, res) => {
    const rel = req.url === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]);
    const file = path.join(root, rel);
    if (!file.startsWith(root)) {
      res.writeHead(403);
      res.end();
      return;
    }
    fs.readFile(file, (err, buf) => {
      if (err) {
        res.writeHead(404);
        res.end('not found');
        return;
      }
      res.writeHead(200, { 'content-type': types[path.extname(file)] ?? 'application/octet-stream' });
      res.end(buf);
    });
  });
}

const keep = process.argv.includes('--keep');
build();

const server = serve(pageDir);
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();

const browser = await chromium.launch();
const page = await browser.newPage();
const noise = [];
page.on('console', (m) => noise.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => noise.push(`[pageerror] ${e.message}`));

let result;
try {
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => window.__aeroResult !== undefined, null, { timeout: 60_000 });
  result = await page.evaluate(() => window.__aeroResult);
} catch (err) {
  result = { ok: false, error: String(err).split('\n')[0] };
}

await browser.close();
server.close();
if (!keep) {
  for (const f of ['aero_jit_wasm.js', 'aero_jit_wasm_bg.wasm', 'aero_jit_wasm.d.ts',
                   'aero_jit_wasm_bg.wasm.d.ts', 'package.json', '.gitignore']) {
    fs.rmSync(path.join(pageDir, f), { force: true });
  }
}

if (noise.length) {
  console.log(noise.join('\n'));
}
console.log(JSON.stringify(result, null, 2));

if (!result?.ok) {
  console.error('tier did not run in the browser');
  process.exit(1);
}
const wrong = Object.entries(EXPECTED).filter(([k, v]) => result[k] !== v);
if (wrong.length) {
  console.error(
    'guest state is wrong: ' +
      wrong.map(([k, v]) => `${k}=${result[k]} (expected ${v})`).join(', '),
  );
  process.exit(1);
}
console.log('compiled Tier-1 ran in the browser and the guest state is right');

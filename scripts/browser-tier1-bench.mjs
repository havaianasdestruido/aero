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
//   node scripts/browser-tier1-bench.mjs [--keep]
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
// Throughput, not state: the smoke script already asserts the guest arithmetic. What
// this checks is that the run actually executed the work it claims — a tier that
// compiled nothing and fell back would report a tiny `retired` and a huge MIPS.
const MIN_RETIRED = 5_000_000;

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
  await page.goto(`http://127.0.0.1:${port}/bench.html`);
  await page.waitForFunction(() => window.__aeroResult !== undefined, null, { timeout: 120_000 });
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
if (result.compiledBlocks < 1) {
  console.error('nothing was compiled; this measured the interpreter, not the tier');
  process.exit(1);
}
if (result.retired < MIN_RETIRED) {
  console.error(
    `only ${result.retired} instructions retired (expected >= ${MIN_RETIRED}): the run ` +
      'ended early, so the rate is measuring something other than steady execution',
  );
  process.exit(1);
}
if (result.unchainedRetired < MIN_RETIRED) {
  console.error(
    `the unchained side only retired ${result.unchainedRetired}: the two sides did not ` +
      'run the same work, so the ratio means nothing',
  );
  process.exit(1);
}
if (!result.chainSteps) {
  console.error(
    'the chained side took 0 in-wasm chain steps: the dispatcher never chained, so the ' +
      'ratio is not measuring chaining at all',
  );
  process.exit(1);
}
console.log(
  `in-wasm chain steps: ${result.chainSteps} chained, ${result.unchainedChainSteps} unchained`,
);
console.log(
  `browser tier-1 throughput: ${result.chainedMips} MIPS chained vs ` +
    `${result.unchainedMips} unchained on the same program and binary ` +
    `(${result.speedup}x), ${result.compiledBlocks} blocks, ${result.retired} retired`,
);

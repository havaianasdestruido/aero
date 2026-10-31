#!/usr/bin/env node
/**
 * Headless boot smoke test for the public player page.
 *
 * The deployed artifact is the only thing that matters, so this script serves
 * the *built* `dist/` over HTTP (with the COOP/COEP headers a real static host
 * cannot set) and drives it in Chromium:
 *
 *   1. the page must reach `player.ready` and start presenting frames,
 *   2. the bundled demo boot sector must announce itself on COM1,
 *   3. the canvas must actually show the guest (non-black pixels),
 *   4. typing must change what the guest displays (keyboard path end to end).
 *
 * It prints a machine-readable report plus a luminance ASCII sketch of the
 * screen, because CI logs (and the API annotations derived from them) are the
 * only view of the result available to some reviewers.
 *
 * Exit code 0 = pass, 1 = fail.
 */

import { createServer } from "node:http";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, normalize, resolve } from "node:path";
import process from "node:process";

const DIST = resolve(process.cwd(), "dist");
const PORT = Number(process.env.AERO_SMOKE_PORT ?? 8123);
const HOST = "127.0.0.1";
const reportPath = process.env.AERO_SMOKE_REPORT || "";
const inActions = process.env.GITHUB_ACTIONS === "true";

/** Workflow-command escaping (see GitHub Actions "setting an error message"). */
function escapeCommandData(value) {
  return String(value).replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
}

/**
 * The emulated screen, as coarse luminance, so a reviewer reading only the
 * checks API can see what the guest actually drew.
 */
function sketchSummary(lines) {
  return lines
    .slice(0, 14)
    .map((line) => line.replace(/\s+$/, ""))
    .filter((line, index, all) => line.length > 0 || all.slice(index + 1).some((next) => next.length > 0))
    .join("\n")
    .slice(0, 1200);
}

function annotate(level, title, message) {
  if (!inActions) return;
  console.log(`::${level} title=${escapeCommandData(title)}::${escapeCommandData(message)}`);
}
const reportScreenshot = process.env.AERO_SMOKE_SCREENSHOT || "";

const MIME = new Map(
  Object.entries({
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".img": "application/octet-stream",
    ".iso": "application/octet-stream",
    ".txt": "text/plain; charset=utf-8",
  }),
);

/**
 * Base path the artifact was built for (`/aero/` on GitHub Pages, `/` locally).
 *
 * Derived from the built index.html rather than hard-coded: a mismatch here is
 * exactly the class of bug this test exists to catch, and a test that guessed
 * the prefix would have hidden it.
 */
function detectBasePrefix() {
  const explicit = (process.env.AERO_SMOKE_BASE ?? "").trim();
  if (explicit) return explicit.endsWith("/") ? explicit : `${explicit}/`;
  try {
    const html = readFileSync(join(DIST, "index.html"), "utf8");
    const match =
      html.match(/<script[^>]+type="module"[^>]+src="([^"]+)"/i) ??
      html.match(/<link[^>]+href="([^"]*\/assets\/[^"]+)"/i);
    const src = match?.[1] ?? "";
    if (src.startsWith("/")) {
      const idx = src.indexOf("/assets/");
      return idx <= 0 ? "/" : `${src.slice(0, idx)}/`;
    }
  } catch {
    /* fall through */
  }
  return "/";
}

function startStaticServer(basePrefix) {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${HOST}:${PORT}`);
    let pathname = decodeURIComponent(url.pathname);

    // Mirror GitHub Pages: only the project sub-path is served.
    if (!pathname.startsWith(basePrefix)) {
      res.statusCode = 404;
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.end(`not found: ${pathname} (site is served from ${basePrefix})`);
      return;
    }
    pathname = `/${pathname.slice(basePrefix.length)}`;
    if (pathname.endsWith("/")) pathname += "index.html";

    const filePath = join(DIST, normalize(pathname).replace(/^(\.\.[/\\])+/, ""));
    if (!filePath.startsWith(DIST) || !existsSync(filePath) || !statSync(filePath).isFile()) {
      res.statusCode = 404;
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.end(`not found: ${pathname}`);
      return;
    }

    res.statusCode = 200;
    res.setHeader("Content-Type", MIME.get(extname(filePath).toLowerCase()) ?? "application/octet-stream");
    // Mirrors what the isolation service worker does on a static host.
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    res.setHeader("Cache-Control", "no-store");
    createReadStream(filePath).pipe(res);
  });

  return new Promise((resolvePromise, rejectPromise) => {
    server.once("error", rejectPromise);
    server.listen(PORT, HOST, () => resolvePromise(server));
  });
}

/** Read back a coarse luminance sketch of the emulated screen (80x25 cells). */
const SAMPLE_SCREEN = `(() => {
  const canvas = document.getElementById('vm-canvas');
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  let nonBlack = 0, bright = 0, checksum = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r || g || b) nonBlack++;
    if (r + g + b > 96) bright++;
    // Cheap order-sensitive checksum: enough to notice a repaint, cheap enough
    // to run on every sampled pixel (640x400 = 256k pixels).
    checksum = (checksum * 31 + ((r << 16) | (g << 8) | b)) | 0;
  }
  const rows = 25, cols = 80;
  const lines = [];
  for (let ry = 0; ry < rows; ry++) {
    let line = '';
    for (let rx = 0; rx < cols; rx++) {
      const x0 = Math.floor((rx * w) / cols), x1 = Math.max(x0 + 1, Math.floor(((rx + 1) * w) / cols));
      const y0 = Math.floor((ry * h) / rows), y1 = Math.max(y0 + 1, Math.floor(((ry + 1) * h) / rows));
      let sum = 0, count = 0;
      for (let y = y0; y < y1; y += Math.max(1, Math.floor((y1 - y0) / 4))) {
        for (let x = x0; x < x1; x += Math.max(1, Math.floor((x1 - x0) / 4))) {
          const i = (y * w + x) * 4;
          sum += data[i] + data[i + 1] + data[i + 2];
          count++;
        }
      }
      const avg = count ? sum / count : 0;
      line += avg > 180 ? '#' : avg > 48 ? '+' : avg > 8 ? '.' : ' ';
    }
    lines.push(line.replace(/\\s+$/, ''));
  }
  return { width: w, height: h, nonBlack, bright, checksum, lines };
})()`;

async function main() {
  if (!existsSync(join(DIST, "index.html"))) {
    console.error(`error: ${DIST}/index.html is missing — build the site before running the smoke test.`);
    process.exit(1);
  }

  // Imported lazily so a missing browser install fails with a clear message.
  const { chromium } = await import("@playwright/test");

  const basePrefix = detectBasePrefix();
  const server = await startStaticServer(basePrefix);
  const browser = await chromium.launch({ args: ["--no-sandbox", "--disable-dev-shm-usage"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  const consoleLines = [];
  page.on("console", (msg) => {
    const text = `${msg.type()}: ${msg.text()}`;
    if (consoleLines.length < 200) consoleLines.push(text);
  });
  page.on("pageerror", (err) => consoleLines.push(`pageerror: ${err.message}`));
  const failedRequests = [];
  page.on("response", (response) => {
    if (response.status() >= 400) failedRequests.push(`${response.status()} ${response.url()}`);
  });
  page.on("requestfailed", (request) => failedRequests.push(`failed ${request.url()}`));

  const report = { pass: false, base: basePrefix, steps: {}, consoleLines, failedRequests };
  const fail = (step, detail) => {
    report.steps[step] = { ok: false, detail };
    throw new Error(`${step}: ${detail}`);
  };

  try {
    await page.goto(`http://${HOST}:${PORT}${basePrefix}`, { waitUntil: "domcontentloaded", timeout: 60_000 });

    await page
      .waitForFunction(() => window.__aeroPlayer && window.__aeroPlayer.ready === true, undefined, { timeout: 120_000 })
      .catch((err) => {
        const detail = failedRequests.length
          ? `${err} (failed requests: ${failedRequests.slice(0, 6).join(", ")})`
          : String(err);
        fail("ready", detail);
      });
    report.steps.ready = { ok: true, variant: await page.evaluate(() => window.__aeroPlayer.variant) };

    await page
      .waitForFunction(() => (window.__aeroPlayer?.frames ?? 0) > 0, undefined, { timeout: 60_000 })
      .catch((err) => fail("frames", String(err)));
    report.steps.frames = { ok: true, frames: await page.evaluate(() => window.__aeroPlayer.frames) };

    await page
      .waitForFunction(() => (window.__aeroPlayer?.serial ?? "").includes("AERO-DEMO-BOOT-OK"), undefined, {
        timeout: 90_000,
      })
      .catch(async (err) => {
        const serial = await page.evaluate(() => window.__aeroPlayer?.serial ?? "");
        fail("serial", `expected AERO-DEMO-BOOT-OK; got ${JSON.stringify(serial.slice(0, 300))} (${err})`);
      });
    report.steps.serial = { ok: true, serial: await page.evaluate(() => window.__aeroPlayer.serial) };

    // Give the guest a moment to paint the banner.
    await page.waitForTimeout(2_500);

    const before = await page.evaluate(SAMPLE_SCREEN);
    if (!before) fail("canvas", "no canvas");
    report.steps.screen = {
      ok: before.nonBlack > 500,
      nonBlack: before.nonBlack,
      bright: before.bright,
      width: before.width,
      height: before.height,
      sketch: before.lines,
    };
    if (before.nonBlack <= 500) fail("screen", `canvas is (nearly) black: ${before.nonBlack} non-black pixels`);

    // Keyboard path, end to end: the demo echoes every accepted character to
    // COM1 as well as to the screen, so the serial log proves the guest *ran*
    // code in response to the key rather than merely repainting.
    const typed = "aero ci";
    const beforeKeys = await page.evaluate(() => window.__aeroPlayer.keysSent ?? 0);
    const beforeFrames = await page.evaluate(() => window.__aeroPlayer.frames ?? 0);
    await page.keyboard.type(typed, { delay: 60 });
    await page
      .waitForFunction((text) => (window.__aeroPlayer?.serial ?? "").includes(text), typed, { timeout: 4_000 })
      .catch(() => {
        /* the report below explains what did happen */
      });

    const after = await page.evaluate(SAMPLE_SCREEN);
    if (after) report.screenSketch = after.lines;
    const state = await page.evaluate(() => ({
      serial: window.__aeroPlayer.serial ?? "",
      keysSent: window.__aeroPlayer.keysSent ?? 0,
      frames: window.__aeroPlayer.frames ?? 0,
      ips: window.__aeroPlayer.ips ?? 0,
      inputMode: window.__aeroPlayer.inputMode,
    }));

    const serialEchoed = state.serial.includes(typed);
    const repainted = before.checksum !== after.checksum;
    report.steps.keyboard = {
      ok: serialEchoed,
      serialEchoed,
      repainted,
      inputMode: state.inputMode,
      keysForwarded: state.keysSent - beforeKeys,
      instructionsPerSecond: state.ips,
      framesBefore: beforeFrames,
      framesAfter: state.frames,
      checksumBefore: before.checksum,
      checksumAfter: after.checksum,
      detail: serialEchoed
        ? "guest echoed the typed text on COM1"
        : `guest never echoed. keys=${state.keysSent - beforeKeys} ips=${state.ips} ` +
          `repainted=${repainted} serial=${JSON.stringify(state.serial.slice(-200))}`,
    };
    if (!report.steps.keyboard.ok) fail("keyboard", report.steps.keyboard.detail);

    report.pass = true;
  } catch (err) {
    report.error = String(err && err.message ? err.message : err);
  } finally {
    // Artifacts are captured while the page is still alive: CI annotates
    // failures from this log, and reviewers can open the PNG after the fact.
    if (reportScreenshot) {
      try {
        mkdirSync(dirname(reportScreenshot), { recursive: true });
        await page.screenshot({ path: reportScreenshot, fullPage: false });
        report.screenshot = reportScreenshot;
      } catch (err) {
        report.screenshotError = String(err);
      }
    }
    if (reportPath) {
      try {
        mkdirSync(dirname(reportPath), { recursive: true });
        writeFileSync(reportPath, JSON.stringify({ ...report, screenSketch: report.screenSketch }, null, 2));
        report.reportFile = reportPath;
      } catch (err) {
        report.reportError = String(err);
      }
    }

    await context.close().catch(() => {});
    await browser.close().catch(() => {});
    server.close();
  }

  console.log("=== aero player smoke report ===");
  console.log(JSON.stringify(report, null, 2));
  if (report.failedRequests?.length) {
    console.log("=== failed requests ===");
    for (const line of report.failedRequests.slice(0, 20)) console.log(line);
  }
  if (report.screenSketch) {
    console.log("=== screen (80x25 luminance sketch) ===");
    for (const line of report.screenSketch) console.log(`|${line}|`);
  }

  if (!report.pass) {
    console.log("RESULT: FAIL");
    if (report.error) console.log(`error: ${report.error}`);
    // Make the failure readable from the checks API, where annotations are the
    // only triage surface some reviewers have.
    for (const [step, value] of Object.entries(report.steps ?? {})) {
      if (value && value.ok === false) annotate("error", `player smoke: ${step}`, value.detail ?? "step failed");
    }
    if (report.error) annotate("error", "player smoke: failure", report.error.slice(0, 1500));
    for (const line of report.consoleLines.slice(-10)) annotate("error", "player console", line.slice(0, 500));
    const sketch = report.screenSketch ?? report.steps.screen?.sketch;
    if (sketch) annotate("error", "player screen", sketchSummary(sketch));
    if (report.consoleLines.length) {
      console.log("=== console ===");
      for (const line of report.consoleLines) console.log(line);
    }
    process.exit(1);
  }

  const steps = Object.entries(report.steps)
    .map(([name, value]) => `${name}=${value.ok ? "ok" : "FAIL"}`)
    .join(" ");
  annotate(
    "notice",
    "player smoke passed",
    `${steps} frames=${report.steps.frames?.frames ?? "?"} serial=${JSON.stringify(report.steps.serial?.serial ?? "")} ` +
      `keyboard=${JSON.stringify(report.steps.keyboard ?? null)}`,
  );
  if (report.screenSketch) annotate("notice", "player screen", sketchSummary(report.screenSketch));
  console.log("RESULT: PASS");
}

main().catch((err) => {
  console.error("smoke test crashed:", err);
  process.exit(1);
});

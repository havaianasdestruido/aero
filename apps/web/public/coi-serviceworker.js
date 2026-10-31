/*
 * Cross-origin isolation shim for static hosting.
 *
 * Why this exists
 * ---------------
 * `SharedArrayBuffer` (and therefore fast, zero-copy display sharing with the
 * emulator worker) is only exposed to pages that are "cross-origin isolated",
 * which normally requires the server to send:
 *
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 *
 * GitHub Pages cannot set response headers, so this service worker re-issues
 * same-origin responses with those headers attached. That is the standard
 * technique used by wasm projects hosted on static file hosts.
 *
 * Properties that matter for us:
 * - No caching: every request goes to the network, so a deploy is visible on
 *   the next reload (the site must never serve stale emulator binaries).
 * - Opt-out: adding `?no-coi=1` (or running on an origin where service workers
 *   are unavailable) leaves the page working without shared memory; the player
 *   then falls back to copying frames.
 * - The page reloads itself exactly once, when the worker first takes control.
 */

/* eslint-env serviceworker */

const COOP = "same-origin";
const COEP = "require-corp";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // Ignore requests that cannot be re-wrapped (opaque/no-cors) and the
  // cache-only special case, which throws when handled manually.
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    (async () => {
      let response;
      try {
        response = await fetch(request);
      } catch (err) {
        return Response.error();
      }

      // Opaque responses (status 0) cannot be rebuilt; hand them back untouched.
      if (response.status === 0) return response;

      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Opener-Policy", COOP);
      headers.set("Cross-Origin-Embedder-Policy", COEP);
      headers.set("Cross-Origin-Resource-Policy", "cross-origin");

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    })(),
  );
});

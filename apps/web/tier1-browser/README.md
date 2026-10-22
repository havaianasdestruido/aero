# Tier-1 compiled tier, in the browser

`index.html` runs a block of guest x86 through the *same* compiled tier the native
build uses: block discovery, translation, dead-flag elimination and WASM code
generation are shared code, and only the engine underneath differs — Wasmtime
natively, `WebAssembly.Module`/`Instance` here.

Build the module next to this page and open it from a server (module scripts and
`application/wasm` do not load over `file://`):

```sh
wasm-pack build --target web --out-dir apps/web/tier1-browser crates/aero-jit-wasm
```

`scripts/browser-tier1-smoke.mjs` does both and asserts the result, which is what CI
would run:

```sh
node scripts/browser-tier1-smoke.mjs
```

The page reports the guest registers after the run, plus how many blocks the tier
compiled — the latter is what separates "the tier ran it" from "nothing happened".

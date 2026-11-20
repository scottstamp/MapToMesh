// Generation in the browser (docs/plans/browser-wasm.md): Pyodide runs the maptomesh package in
// this worker. Messages: in {type: "run", opts, memory_mb, cache_mb}; out {type: "setup" |
// "progress" | "done" | "error", ...}. The network and cache functions below are called from
// maptomesh/browser.py.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide.mjs";
import { PackCache, mountCache } from "./opfs-cache.js";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
// Python and its packages are kept in the browser's Cache Storage (not left to the HTTP cache,
// which browsers may skip or evict): ~23 MB downloaded once per Pyodide version. Same name in
// web/app.js, which checks it to tell phone users what a build will download.
const RUNTIME_CACHE = "maptomesh-runtime-pyodide-314.0.8";
const RUNTIME_DONE = "/__runtime_complete__";
const PACKAGES = ["numpy", "pyproj", "pydantic", "pillow", "requests", "affine"];
// SciPy comes trimmed from our server (native/trim_scipy.py: ~6.5 of 14 MB). A build that needs a
// part left out restarts the worker with Pyodide's full SciPy (fullScipy, set by the page).
let fullScipy = false;
let PROXY = null;                // "/api/proxy?url=": the server streams hosts without CORS
let PROXY_FIRST = new Set();     // ... and these never answer a browser directly
const CONCURRENCY = 12;
if (new URLSearchParams(self.location.search).has("nojspi")) {   // test the path other browsers take
  delete WebAssembly.Suspending;
  delete WebAssembly.promising;
}

function headersOf(res) {
  const o = {};
  res.headers.forEach((v, k) => { o[k] = v; });
  return o;
}

function viaProxy(url) {
  return PROXY + encodeURIComponent(url);
}

// At most HOST_LIMIT requests in flight per host (agency servers rate-limit bursts: IGN answers
// 429 to a dozen at once); the server's own proxy allows 16 per client.
const HOST_LIMIT = 6, PROXY_LIMIT = 12;
const hostActive = new Map(), hostWait = new Map();

async function hostSlot(host, limit) {
  const n = hostActive.get(host) || 0;
  if (n < limit) { hostActive.set(host, n + 1); return; }
  await new Promise((resolve) => {
    if (!hostWait.has(host)) hostWait.set(host, []);
    hostWait.get(host).push(resolve);       // the releasing request hands its slot over
  });
}

function hostRelease(host) {
  const q = hostWait.get(host);
  if (q && q.length) q.shift()();
  else hostActive.set(host, (hostActive.get(host) || 1) - 1);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchOnce(url, headers, method = "GET", body = null) {
  const init = { method, headers: headers || {} };
  if (body) init.body = body;
  let proxied = false;
  if (PROXY && PROXY_FIRST.has(new URL(url, self.location.href).hostname)) { url = viaProxy(url); proxied = true; }
  for (let attempt = 0; ; attempt++) {
    const host = proxied ? self.location.host : new URL(url, self.location.href).host;
    await hostSlot(host, proxied ? PROXY_LIMIT : HOST_LIMIT);
    let res, body;
    try {
      res = await fetch(url, init);
      if (!(res.status === 429 || res.status >= 500) || attempt >= 4) {
        body = method === "HEAD" ? new Uint8Array(0) : new Uint8Array(await res.arrayBuffer());
      }
    } catch (e) {               // CORS refusal or network failure: the server forwards it
      hostRelease(host);
      if (PROXY && !proxied) { url = viaProxy(url); proxied = true; continue; }
      if (attempt < 3) { await sleep(500 * 2 ** attempt); continue; }
      throw e;
    }
    hostRelease(host);
    if (body === undefined) {   // rate limited / server error: back off (Retry-After if given)
      const ra = parseFloat(res.headers.get("retry-after"));
      await sleep(Number.isFinite(ra) ? Math.min(ra, 30) * 1000 : 1000 * 2 ** attempt + Math.random() * 500);
      continue;
    }
    return { status: res.status, headers: headersOf(res), body };
  }
}

// Downloads Python announced (mtmHint) start right away, a few at a time, while Python works;
// a later request for the same URL + range takes the result. At most HINT_BUFFER bytes wait.
const HINTS = new Map();         // key -> promise of a response
const hintQueue = [];
let hintActive = 0, hintBuffered = 0;
const HINT_BUFFER = 96 << 20;

function hintKey(url, headers) {
  return `${url}|${(headers && (headers.Range || headers.range)) || ""}`;
}

function pumpHints() {
  while (hintActive < CONCURRENCY && hintQueue.length && hintBuffered < HINT_BUFFER) {
    const [key, url, headers] = hintQueue.shift();
    hintActive++;
    const p = fetchOnce(url, headers, "GET")
      .then((r) => { hintBuffered += r.body.length; return r; })
      .finally(() => { hintActive--; pumpHints(); });
    p.catch(() => {});           // a failed hint is retried by the real request
    HINTS.set(key, p);
  }
}

self.mtmHint = (reqs) => {
  for (const [url, headers] of Array.from(reqs)) {
    const key = hintKey(url, headers);
    if (!HINTS.has(key) && !hintQueue.some((q) => q[0] === key)) hintQueue.push([key, url, headers]);
  }
  pumpHints();
};

async function fetchHinted(url, headers, method = "GET", body = null) {
  if (method === "GET" && !body) {
    const key = hintKey(url, headers);
    const p = HINTS.get(key);
    if (p) {
      HINTS.delete(key);
      try {
        const r = await p;
        hintBuffered -= r.body.length;
        pumpHints();
        return r;
      } catch (e) { /* fetch it again below */ }
    } else {
      const i = hintQueue.findIndex((q) => q[0] === key);
      if (i >= 0) hintQueue.splice(i, 1);   // needed now: don't fetch it twice
    }
  }
  return fetchOnce(url, headers, method, body);
}

function resetHints() {
  hintQueue.length = 0;
  HINTS.clear();
  hintBuffered = 0;
}

self.mtmFetch = fetchHinted;

self.mtmFetchMany = async (reqs) => {
  const list = Array.from(reqs);
  const out = new Array(list.length);
  let next = 0;
  const lane = async () => {
    while (next < list.length) {
      const i = next++;
      out[i] = await fetchHinted(list[i][0], list[i][1], "GET");
    }
  };
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, list.length) }, lane));
  return out;
};

function xhr(url, headers, method, body) {
  const x = new XMLHttpRequest();
  x.open(method, url, false);
  x.responseType = "arraybuffer";
  for (const [k, v] of Object.entries(headers || {})) x.setRequestHeader(k, v);
  x.send(body || null);
  return x;
}

self.mtmFetchSync = (url, headers, method = "GET", body = null) => {   // no Promise Integration: one at a time
  let x;
  if (PROXY && PROXY_FIRST.has(new URL(url, self.location.href).hostname)) {
    x = xhr(viaProxy(url), headers, method, body);
  } else {
    try {
      x = xhr(url, headers, method, body);
    } catch (e) {
      if (!PROXY) throw e;
      x = xhr(viaProxy(url), headers, method, body);
    }
  }
  const h = {};
  for (const line of x.getAllResponseHeaders().trim().split(/[\r\n]+/)) {
    const i = line.indexOf(":");
    if (i > 0) h[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  }
  return { status: x.status, headers: h, body: new Uint8Array(x.response || new ArrayBuffer(0)) };
};

let py = null;
let ready = null;

// ---- the runtime cache: cache-first for files that never change under their URL (the
// versioned Pyodide CDN path; our wheels and package under content-versioned URLs)
const netFetch = globalThis.fetch.bind(globalThis);
let runtimeCache = null;
const RUNTIME = { hits: 0, misses: 0, keep: new Set(), errors: [], puts: [] };

function immutableUrl(url) {
  if (url.startsWith(PYODIDE_URL)) return true;
  const u = new URL(url);
  return u.origin === self.location.origin
    && (u.pathname.startsWith("/api/browser/files/") || (u.pathname === "/api/browser/package.zip" && u.searchParams.has("v")));
}

async function cachedFetch(input, init) {
  const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url, self.location.href).href;
  if (!runtimeCache || (init && init.method && init.method !== "GET") || !immutableUrl(url)) return netFetch(input, init);
  RUNTIME.keep.add(url);
  try {
    const hit = await runtimeCache.match(url);
    if (hit) { RUNTIME.hits++; return hit; }
  } catch (e) { /* storage unavailable: the network */ }
  const res = await netFetch(input, init);   // (Pyodide passes its packages' integrity hashes)
  RUNTIME.misses++;
  if (res.ok) {
    RUNTIME.puts.push(runtimeCache.put(url, res.clone()).catch((e) => {
      RUNTIME.errors.push(`${url.split("/").pop()}: ${e && e.name}: ${e && e.message}`);
    }));
  }
  return res;
}
globalThis.fetch = cachedFetch;     // Pyodide fetches its wasm, standard library and packages

async function openRuntimeCache() {
  try {
    runtimeCache = await caches.open(RUNTIME_CACHE);
    for (const name of await caches.keys()) {   // earlier Pyodide versions
      if (name.startsWith("maptomesh-runtime-") && name !== RUNTIME_CACHE) await caches.delete(name);
    }
  } catch (e) {
    RUNTIME.errors.push(`Cache Storage: ${e && e.name}: ${e && e.message}`);
    runtimeCache = null;    // no Cache Storage (private windows in some browsers): HTTP cache only
  }
}

async function finishRuntimeCache() {   // forget our superseded wheels / packages; mark complete
  if (!runtimeCache) return;
  try {
    await Promise.all(RUNTIME.puts);
    if (RUNTIME.errors.length) return;     // incomplete: not marked, the next start fills the gaps
    for (const req of await runtimeCache.keys()) {
      if (new URL(req.url).origin === self.location.origin && !RUNTIME.keep.has(req.url) && !req.url.endsWith(RUNTIME_DONE)) {
        await runtimeCache.delete(req);
      }
    }
    await runtimeCache.put(new URL(RUNTIME_DONE, self.location.href).href, new Response("1"));
  } catch (e) { /* best effort */ }
}

async function runtimeKept() {    // the runtime was completely stored by an earlier visit
  try {
    return !!runtimeCache && !!(await runtimeCache.match(new URL(RUNTIME_DONE, self.location.href).href));
  } catch (e) { return false; }
}

async function runtimeModule() {   // pyodide.asm.mjs through the cache (Pyodide would import() it)
  try {
    const src = await (await cachedFetch(PYODIDE_URL + "pyodide.asm.mjs")).text();
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    try { return (await import(url)).default; } finally { URL.revokeObjectURL(url); }
  } catch (e) {
    return undefined;       // let Pyodide load it itself
  }
}

// Persistent data cache (web/opfs-cache.js): held while a job runs, released after it so another
// tab can use it; maptomesh/browser.py reads and writes it through these functions.
const CACHE = new PackCache();
let mounted = null;              // what /cache is: "cache" (persistent) or "memory"
self.mtmCacheOn = () => CACHE.open;
self.mtmCacheGet = (key) => CACHE.get(key) ?? undefined;   // undefined -> Python None
self.mtmCachePut = (key, data, time) => CACHE.put(key, data, time || Date.now());
self.mtmCacheHas = (key) => CACHE.has(key);
self.mtmCacheDelPrefix = (prefix) => CACHE.delPrefix(prefix);

async function cacheBegin(mb) {
  let ok = false, changed = false;
  if (mb > 0 && navigator.storage && navigator.storage.getDirectory) {
    try {
      let budget = mb * 1e6;
      try {
        const est = await navigator.storage.estimate();
        if (est.quota) budget = Math.min(budget, est.quota * 0.3);
      } catch (e) { /* no estimate: the page's budget */ }
      changed = await CACHE.begin(budget);
      ok = true;
    } catch (e) {               // no OPFS, or another tab is using the cache
      console.warn("Data cache not available for this build:", e && e.message);
    }
  }
  if (ok && (changed || mounted !== "cache")) { mountCache(py.FS, "/cache", CACHE); mounted = "cache"; }
  else if (!ok && mounted !== "memory") { mountCache(py.FS, "/cache", null); mounted = "memory"; }
  return ok;
}

function downloadedMB() {
  return performance.getEntriesByType("resource")
    .reduce((s, e) => s + (e.transferSize || e.encodedBodySize || 0), 0) / 1e6;
}

async function init(say) {
  const t0 = performance.now();
  await openRuntimeCache();
  const kept = await runtimeKept();
  say(kept ? "Starting Python (kept on this device)"
    : runtimeCache ? "Downloading Python (~23 MB, first time only)"
      : "Downloading Python (~23 MB; this page can't keep it on this device)");
  const cfg = await (await netFetch("/api/browser/config", { cache: "no-cache" })).json();
  PROXY = cfg.proxy || null;
  PROXY_FIRST = new Set(cfg.proxy_first || []);
  py = await loadPyodide({ createPyodideModule: await runtimeModule() });
  const t1 = performance.now();
  const wheels = { ...(cfg.wheels || {}) };
  if (fullScipy || !wheels.scipy) delete wheels.scipy;
  await py.loadPackage(wheels.scipy ? PACKAGES : [...PACKAGES, "scipy"], { messageCallback: () => {} });
  const t2 = performance.now();
  // our wheels: maptomesh_native (Rust: meshing, 3MF, tile codecs; ~2x faster), lazrs (LAZ),
  // laspy (point clouds; from PyPI, pinned), SciPy (trimmed)
  const loaded = [];
  for (const [name, url] of Object.entries(wheels)) {
    try {
      await py.loadPackage(new URL(url, self.location.href).href, { messageCallback: () => {} });
      loaded.push(name);
    } catch (e) {
      console.warn(`${name} not loaded:`, e);
      if (name === "scipy") await py.loadPackage("scipy", { messageCallback: () => {} });   // the full one
    }
  }
  const native = loaded.includes("maptomesh_native");
  const zip = await (await cachedFetch(new URL(cfg.package || "/api/browser/package.zip", self.location.href).href)).arrayBuffer();
  py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import maptomesh.browser as b; b.install()");
  await finishRuntimeCache();
  const t3 = performance.now();
  return {
    runtime_s: (t1 - t0) / 1000, packages_s: (t2 - t1) / 1000, code_s: (t3 - t2) / 1000,
    downloaded_mb: downloadedMB(), jspi: typeof WebAssembly.Suspending === "function", native, wheels: loaded,
    pyodide: py.version, scipy: loaded.includes("scipy") ? "trimmed" : "full", runtime_cached: runtimeCache !== null && RUNTIME.misses === 0, runtime_kept_before: kept,
    secure: self.isSecureContext,
    runtime_files: { cached: RUNTIME.hits, downloaded: RUNTIME.misses },
    runtime_errors: RUNTIME.errors.slice(0, 5),
  };
}

let readySettled = false;
let say = () => {};                // where init's status goes: the running job's progress, if any

function startPython() {
  if (!ready) {
    ready = init((msg) => say(msg));
    ready.then(() => { readySettled = true; }, () => { readySettled = true; });
  }
  return ready;
}

self.onmessage = async (e) => {
  if (!ready && e.data.full_scipy) fullScipy = true;   // (only before Python is loaded)
  if (e.data.type === "init") {      // the page starts Python ahead of the first build
    try {
      self.postMessage({ type: "setup", setup: await startPython(), early: true });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err && err.message || err).slice(0, 500), fatal: true, early: true });
      self.close();
    }
    return;
  }
  if (e.data.type !== "run") return;
  try {
    const t0 = performance.now();
    say = (msg) => self.postMessage({ type: "progress", f: 0, msg });
    if (ready && !readySettled) say("Starting Python…");   // started early, not ready yet
    const tw = performance.now();
    const setup = await startPython();
    say = () => {};
    self.postMessage({ type: "setup", setup, waited_s: (performance.now() - tw) / 1000 });
    const persistent = await cacheBegin(e.data.cache_mb || 0);
    const hit0 = CACHE.stats.hitBytes;
    const tJob = performance.now();
    const netBefore = downloadedMB();
    const runJob = py.pyimport("maptomesh.browser").run_job;
    const progress = (f, msg) => self.postMessage({ type: "progress", f, msg });
    resetHints();
    const args = [JSON.stringify(e.data.opts), progress, e.data.memory_mb || 0];
    const out = setup.jspi && runJob.callPromising ? await runJob.callPromising(...args) : runJob(...args);
    runJob.destroy();
    const res = JSON.parse(out);
    const cache = { persistent, hit_mb: (CACHE.stats.hitBytes - hit0) / 1e6, size_mb: CACHE.bytes() / 1e6 };
    CACHE.end();
    if (res.error) {             // refused up front (limits.check): too large for this device
      self.postMessage({ type: "error", ...res.error });
      return;
    }
    const files = {};
    for (const name of py.FS.readdir(res.outdir)) {
      if (name === "." || name === "..") continue;
      const p = `${res.outdir}/${name}`;
      files[name] = py.FS.readFile(p);
      py.FS.unlink(p);                       // free the in-memory copy
    }
    py.FS.rmdir(res.outdir);
    self.postMessage({
      type: "done", res, files,
      job_s: (performance.now() - tJob) / 1000,
      total_s: (performance.now() - t0) / 1000,
      heap_mb: py._module.HEAPU8.length / 1e6,
      job_downloaded_mb: downloadedMB() - netBefore,
      cache,
    }, Object.values(files).map(f => f.buffer));
  } catch (err) {
    try { CACHE.end(); } catch (e) { /* closing anyway */ }
    const text = String(err && err.message || err);
    // out of WebAssembly memory: Python's MemoryError, or the runtime aborting (then it's unusable)
    const memory = /MemoryError|out of memory|Cannot allocate|Aborted\(|memory access out of bounds/i.test(text);
    const fatal = memory || /Aborted\(|fatal error/i.test(text) || (py && py._api && py._api.fatal_error_occurred);
    const last = text.trim().split(/\r?\n/).filter((l) => l.trim()).pop() || text;
    // a part of SciPy the trimmed wheel left out: the page restarts with the full SciPy
    const missing = /No module named '(scipy[\w.]*)'/.exec(text);
    self.postMessage({ type: "error", message: memory ? "Out of memory" : last.slice(0, 500),
                       error_kind: memory ? "memory" : null, fatal: !!fatal, trace: text.slice(-4000),
                       missing_scipy: missing && !fullScipy ? missing[1] : null });
    if (fatal) self.close();
  }
};

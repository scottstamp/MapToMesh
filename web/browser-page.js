// web/browser.html (test page for building in the browser): a file, not inline (the page's CSP)
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const log = (s) => { $("log").textContent += s + "\n"; $("log").scrollTop = 1e9; };
let worker = null, t0 = 0, urls = [];

function opts() {
  const f = new FormData($("f"));
  const num = (k) => Number(f.get(k));
  return {
    lat: num("lat"), lon: num("lon"), width_m: num("width_m"), height_m: num("height_m"),
    grid_cells: num("grid_cells"), size_mm: num("size_mm"), color_mode: f.get("color_mode"),
    osm_roads: f.get("osm_roads"), osm_water: f.get("osm_water") === "on", name: f.get("name") || "MapToMesh",
    surface: "terrain", dem_preference: f.get("dem_preference"), source: f.get("source"),
  };
}

function fmtMB(n) { return (n / 1e6).toFixed(1) + " MB"; }

function showResult(m) {
  const r = m.res, files = m.files;
  urls.forEach(URL.revokeObjectURL);
  urls = [];
  const link = (name, label, type) => {
    if (!files[name]) return "";
    const u = URL.createObjectURL(new Blob([files[name]], { type }));
    urls.push(u);
    return `<a href="${u}" download="${esc(name)}">${label} (${fmtMB(files[name].length)})</a>`;
  };
  const thumb = files["thumbnail.png"] ? URL.createObjectURL(new Blob([files["thumbnail.png"]], { type: "image/png" })) : "";
  if (thumb) urls.push(thumb);
  const b = r.browser, net = b.net;
  const rows = [
    ["Triangles", r.stats.triangles.toLocaleString()],
    ["Job time (this device)", m.job_s.toFixed(1) + " s"],
    ["Data downloaded by the job", `${fmtMB(net.bytes)} in ${net.requests} requests (${net.seconds} s waiting)`],
    ["Peak WebAssembly memory", Math.round(m.heap_mb) + " MB"],
    ["Elevation", (r.data.sources || []).map(s => `${s.label} ${(100 * s.share).toFixed(0)}%`).join("; ")],
  ];
  $("result").innerHTML =
    `<div class="links">${link(r.file, "3MF", "model/3mf")} ${link(r.stl_file, "STL zip", "application/zip")}` +
    ` ${link(r.base_file, "Base 3MF", "model/3mf")}</div>` +
    `<table>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>` +
    (thumb ? `<img src="${thumb}" alt="Top view of the model">` : "");
  log(`stages: ${b.marks.map(([t, , msg]) => `${t}s ${msg}`).join("\n        ")}`);
  window.__last = { job_s: m.job_s, total_s: m.total_s, heap_mb: m.heap_mb, net, triangles: r.stats.triangles,
                    marks: b.marks, sources: (r.data.sources || []).map(s => `${s.label} ${(100 * s.share).toFixed(0)}%`), errors: r.data.errors, sizes: Object.fromEntries(Object.entries(files).map(([k, v]) => [k, v.length])) };
}

$("go").onclick = () => {
  if (!worker) {
    worker = new Worker("/static/browser-worker.js" + location.search, { type: "module" });
    worker.onmessage = (e) => {
      const m = e.data;
      const s = ((performance.now() - t0) / 1000).toFixed(1);
      if (m.type === "progress") {
        $("bar").value = m.f;
        if ($("status").textContent !== m.msg) log(`${s}s  ${m.msg}`);
        $("status").textContent = m.msg;
      } else if (m.type === "setup") {
        const x = m.setup;
        log(`${s}s  Python ready: runtime ${x.runtime_s.toFixed(1)} s, packages ${x.packages_s.toFixed(1)} s, ` +
            `code ${x.code_s.toFixed(1)} s, ~${x.downloaded_mb.toFixed(1)} MB fetched, Pyodide ${x.pyodide}, ` +
            `Promise Integration ${x.jspi ? "yes" : "no (downloads one at a time)"}, ` +
            `Rust extension ${x.native ? "yes" : "no"}`);
        window.__setup = x;
      } else if (m.type === "done") {
        $("status").textContent = `Done in ${s} s`;
        $("bar").hidden = true;
        $("go").disabled = false;
        showResult(m);
        window.__done = true;
      } else if (m.type === "error") {
        $("status").innerHTML = `<span class="err">Failed: ${esc(m.message)}</span>`;
        log(m.message);
        $("bar").hidden = true;
        $("go").disabled = false;
        window.__error = m.message;
      }
    };
  }
  t0 = performance.now();
  window.__done = false; window.__error = null;
  $("go").disabled = true;
  $("bar").hidden = false;
  $("bar").value = 0;
  $("result").innerHTML = "";
  worker.postMessage({ type: "run", opts: opts() });
};

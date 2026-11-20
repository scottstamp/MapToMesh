import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

const $ = (id) => document.getElementById(id);
const R_EARTH = 6371008.8;
const DEG = Math.PI / 180;
const STORE_KEY = 'maptomesh.settings.v1';

const PRESETS = {
  landcover: {
    'Classic city': { base: '#2B2B2B', ground: '#E9E4D8', water: '#2F8FD8', buildings: '#C8553D', roads: '#FAFAFA', vegetation: '#4F8A3C', frame: '#111111' },
    'Amsterdam': { base: '#1A1A1A', ground: '#F4F1EA', water: '#3AA6D0', buildings: '#C4553A', roads: '#F4F1EA', vegetation: '#F4F1EA', frame: '#111111' },
    'Gold streets': { base: '#3A3A3A', ground: '#6E6E6E', water: '#1F3B57', buildings: '#9E9E9E', roads: '#E3B23C', vegetation: '#6E6E6E', frame: '#1A1A1A' },
    'Park & water': { base: '#2B2B2B', ground: '#D8C3A5', water: '#2F8FD8', buildings: '#F2F2F2', roads: '#F2F2F2', vegetation: '#3E7B3A', frame: '#111111' },
    'All white': { base: '#F2F2F2', ground: '#F2F2F2', water: '#F2F2F2', buildings: '#F2F2F2', roads: '#F2F2F2', vegetation: '#F2F2F2', frame: '#F2F2F2' },
  },
  elevation: {
    'Topographic': ['#6FA35A', '#C9B26B', '#B07A3E', '#7A4E2A', '#F4F4F4'],
    'Canyon': ['#7A3B22', '#9E4A26', '#B5562A', '#E08E45', '#F6C99A'],
    'Island': ['#E8D9A8', '#9CCB86', '#3E8E5A', '#1F5E3B', '#5B4636'],
    'Greyscale': ['#3A3A3A', '#5E5E5E', '#858585', '#ABABAB', '#E0E0E0'],
  },
};
const LANDCOVER_KEYS = [['ground', 'Ground'], ['water', 'Water'], ['buildings', 'Buildings'], ['roads', 'Roads & bridges'], ['vegetation', 'Trees']];

const state = {
  center: { lat: 43.2557, lon: -79.8711 },
  drawing: false,
  jobId: null,
  pollTimer: null,
  lastResult: null,
};

// ------------------------------------------------------------------ colour inputs
function colorInput(container, key, label, value) {
  const l = document.createElement('label');
  const input = document.createElement('input');   // values may come from a share link: no innerHTML
  input.type = 'color';
  input.dataset.color = key;
  input.value = value;
  l.append(input, ` ${label}`);
  container.appendChild(l);
  return input;
}

function buildColorInputs() {
  const lc = PRESETS.landcover['Classic city'];
  colorInput($('colorsSingle'), 'single', 'Model', '#E9E4D8');
  for (const [k, label] of LANDCOVER_KEYS) colorInput($('colorsLandcover'), k, label, lc[k]);
  colorInput($('colorsCommon'), 'base', 'Base', lc.base);
  colorInput($('colorsCommon'), 'frame', 'Frame', lc.frame);
  colorInput($('colorsCommon'), 'label', 'Label', '#FAFAFA');
  renderBandInputs(PRESETS.elevation.Topographic);
}

function renderBandInputs(colors) {
  const n = Math.max(2, Math.min(10, parseInt($('bandCount').value) || 5));
  const box = $('bandColors');
  const prev = [...box.querySelectorAll('input')].map((i) => i.value);
  const src = colors || prev;
  box.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const c = src[i] || src[src.length - 1] || '#888888';
    colorInput(box, `band${i}`, `Band ${i + 1}${i === 0 ? ' (lowest)' : i === n - 1 ? ' (highest)' : ''}`, c);
  }
}

function fillPresets() {
  const mode = $('colorMode').value;
  const sel = $('preset');
  setHtml(sel, html`<option value="">— choose —</option>`);
  for (const name of Object.keys(PRESETS[mode] || {})) sel.add(new Option(name, name));
  $('presetRow').hidden = !PRESETS[mode];
}

function applyPreset() {
  const mode = $('colorMode').value, name = $('preset').value;
  if (!name) return;
  const p = PRESETS[mode][name];
  if (mode === 'landcover') {
    for (const [k, v] of Object.entries(p)) {
      const el = document.querySelector(`[data-color="${k}"]`);
      if (el) el.value = v;
    }
  } else if (mode === 'elevation') {
    $('bandCount').value = p.length;
    renderBandInputs(p);
  }
  saveSettings();
}

function colorVal(key) { return document.querySelector(`[data-color="${key}"]`).value.toUpperCase(); }

// ------------------------------------------------------------------ geometry helpers
function areaDims() {
  return { w: parseFloat($('widthKm').value) * 1000 || 1000, h: parseFloat($('heightKm').value) * 1000 || 1000 };
}
function toLatLng(x, y) {
  const c = state.center;
  return [c.lat + (y / R_EARTH) / DEG, c.lon + (x / (R_EARTH * Math.cos(c.lat * DEG))) / DEG];
}
function metresBetween(a, b) {
  const lat = ((a.lat + b.lat) / 2) * DEG;
  return { dx: (b.lng - a.lng) * DEG * R_EARTH * Math.cos(lat), dy: (b.lat - a.lat) * DEG * R_EARTH };
}
function outlineMetres() {
  const { w, h } = areaDims();
  const hw = w / 2, hh = h / 2, shape = $('shape').value;
  const pts = [];
  if (shape === 'circle') {
    for (let i = 0; i < 96; i++) { const t = (i / 96) * 2 * Math.PI; pts.push([hw * Math.cos(t), hh * Math.sin(t)]); }
  } else if (shape === 'hexagon') {
    pts.push([hw, 0], [hw / 2, hh], [-hw / 2, hh], [-hw, 0], [-hw / 2, -hh], [hw / 2, -hh]);
  } else if (shape === 'rounded') {
    const mPerMm = Math.max(w, h) / (parseFloat($('sizeMm').value) || 200);
    const r = Math.min(parseFloat($('cornerMm').value) * mPerMm || 0, hw, hh);
    const corners = [[hw - r, hh - r, 0], [-hw + r, hh - r, 90], [-hw + r, -hh + r, 180], [hw - r, -hh + r, 270]];
    for (const [cx, cy, a0] of corners) for (let k = 0; k <= 8; k++) {
      const a = (a0 + (k / 8) * 90) * DEG; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
  } else {
    pts.push([hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]);
  }
  return pts;
}
function enforceAspect(changed) {
  const shape = $('shape').value;
  const w = parseFloat($('widthKm').value), h = parseFloat($('heightKm').value);
  if (shape === 'circle') {
    if (changed === 'h') $('widthKm').value = h; else $('heightKm').value = w;
  } else if (shape === 'hexagon') {
    if (changed === 'h') $('widthKm').value = +(h / 0.8660254).toFixed(3);
    else $('heightKm').value = +(w * 0.8660254).toFixed(3);
  }
}

// ------------------------------------------------------------------ map
const map = L.map('map', { zoomControl: true }).setView([state.center.lat, state.center.lon], 14);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · Elevation: NRCan, USGS, national mapping agencies, Mapterhorn, Copernicus (credits listed with each model)',
}).addTo(map);

const selection = L.polygon([], { color: '#c8553d', weight: 2, fillOpacity: 0.12, className: 'aoi' }).addTo(map);
const centerMarker = L.circleMarker([0, 0], { radius: 4, color: '#c8553d', fillOpacity: 1, className: 'aoi' }).addTo(map);
let coverageLayer = null;
let hintEl = null;
const COARSE = matchMedia('(pointer: coarse)').matches;   // touch screen: bigger handles
// phones held upright: the panel is a bottom sheet over a full-width map (same query as style.css)
const MOBILE = matchMedia('(orientation: portrait) and (max-width: 900px)');

// ---- resize handles: corners and edge midpoints of the area's bounding box
const HANDLE_DIRS = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]];
const HANDLE_CURSOR = { '-1,-1': 'nesw', '0,-1': 'ns', '1,-1': 'nwse', '1,0': 'ew', '1,1': 'nesw', '0,1': 'ns', '-1,1': 'nwse', '-1,0': 'ew' };
const HANDLE_PX = COARSE ? 18 : 12;
let resizing = null;
const handles = HANDLE_DIRS.map(([sx, sy]) => {
  const m = L.marker([0, 0], {
    draggable: true, keyboard: false, zIndexOffset: 1000,
    icon: L.divIcon({ className: `resize-handle cursor-${HANDLE_CURSOR[`${sx},${sy}`]}`, iconSize: [HANDLE_PX, HANDLE_PX] }),
  }).addTo(map);
  m.on('dragstart', () => {
    const { w, h } = areaDims();
    resizing = { sx, sy, w, h, lat: state.center.lat, lon: state.center.lon, marker: m };
  });
  m.on('drag', (e) => resizeTo(e.target.getLatLng()));
  m.on('dragend', () => { resizing = null; redrawSelection(); saveSettings(); });
  return { m, sx, sy };
});

function resizeTo(ll) {
  const r = resizing;
  if (!r) return;
  const cosl = Math.cos(r.lat * DEG);
  // handle position in metres from the original centre
  const px = (ll.lng - r.lon) * DEG * R_EARTH * cosl;
  const py = (ll.lat - r.lat) * DEG * R_EARTH;
  // the opposite side stays put; size = signed distance from it (never flips through)
  let w = r.sx ? Math.max(50, r.sx * px + r.w / 2) : r.w;
  let h = r.sy ? Math.max(50, r.sy * py + r.h / 2) : r.h;
  const shape = $('shape').value;
  if (shape === 'circle') {
    const v = r.sx && r.sy ? Math.max(w, h) : r.sx ? w : h;
    w = h = v;
  } else if (shape === 'hexagon') {
    if (r.sx) h = w * 0.8660254; else w = h / 0.8660254;
  }
  // new centre: opposite side + half the new size towards the handle (axes without a handle stay centred)
  const cx = r.sx ? -r.sx * r.w / 2 + r.sx * w / 2 : 0;
  const cy = r.sy ? -r.sy * r.h / 2 + r.sy * h / 2 : 0;
  state.center = { lat: r.lat + (cy / R_EARTH) / DEG, lon: r.lon + (cx / (R_EARTH * cosl)) / DEG };
  $('widthKm').value = +(w / 1000).toFixed(3);
  $('heightKm').value = +(h / 1000).toFixed(3);
  redrawSelection();
}

function redrawSelection() {
  selection.setLatLngs(outlineMetres().map(([x, y]) => toLatLng(x, y)));
  centerMarker.setLatLng([state.center.lat, state.center.lon]);
  const { w, h } = areaDims();
  for (const { m, sx, sy } of handles) {
    if (resizing && resizing.marker === m) continue;  // the one under the mouse follows the cursor
    m.setLatLng(toLatLng(sx * w / 2, sy * h / 2));
  }
  updateDerived();
  scheduleCoverage();
}

function showHint(text) {
  if (!hintEl) { hintEl = document.createElement('div'); hintEl.className = 'map-hint'; $('map').appendChild(hintEl); }
  hintEl.textContent = text; hintEl.hidden = !text;
}

function setDrawing(on) {
  state.drawing = on;
  $('drawBtn').classList.toggle('active', on);
  map.getContainer().classList.toggle('drawing', on);
  const shape = $('shape').value;
  showHint(on ? (shape === 'circle' || shape === 'hexagon' ? 'Drag from the centre outwards' : 'Drag a box') : '');
  if (on) { map.dragging.disable(); setSheet(false); } else map.dragging.enable();
}

// Drawing and moving the area use pointer events (mouse, touch and pen alike): Leaflet's map
// mousedown / mousemove never arrive for a finger, so on phones the map panned instead.
// pointer: the one pointer doing it; downPointers: every pointer currently down (pinch-zoom check)
let drawStart = null, dragStart = null;
const downPointers = new Set();
window.addEventListener('pointerdown', (e) => {
  downPointers.add(e.pointerId);
  if (dragStart && e.pointerId !== dragStart.pointer) endMove();   // second finger: let Leaflet pinch-zoom
}, true);
for (const t of ['pointerup', 'pointercancel']) window.addEventListener(t, (e) => downPointers.delete(e.pointerId), true);

function startMove(e) {
  if (state.drawing || e.button > 0 || downPointers.size > 1) return;
  e.stopPropagation(); e.preventDefault();   // the map must not start panning (preventDefault: no mousedown)
  dragStart = { pointer: e.pointerId, ll: map.mouseEventToLatLng(e), center: { ...state.center } };
  map.dragging.disable();
}
function endMove() {
  if (!dragStart) return;
  dragStart = null;
  map.dragging.enable();
  saveSettings();
}
selection.getElement().addEventListener('pointerdown', startMove);
centerMarker.getElement().addEventListener('pointerdown', startMove);

map.getContainer().addEventListener('pointerdown', (e) => {
  if (!state.drawing || e.button > 0 || e.target.closest('.leaflet-control')) return;
  e.stopPropagation(); e.preventDefault();
  drawStart = { pointer: e.pointerId, ll: map.mouseEventToLatLng(e) };
}, true);   // capture: before the area's own handler

window.addEventListener('pointermove', (e) => {
  if (drawStart && e.pointerId === drawStart.pointer) {
    const a = drawStart.ll, b = map.mouseEventToLatLng(e);
    const shape = $('shape').value;
    const { dx, dy } = metresBetween(a, b);
    if (shape === 'circle' || shape === 'hexagon') {
      const r = Math.max(25, Math.hypot(dx, dy));
      state.center = { lat: a.lat, lon: a.lng };
      $('widthKm').value = +(2 * r / 1000).toFixed(3);
      enforceAspect('w');
    } else {
      state.center = { lat: (a.lat + b.lat) / 2, lon: (a.lng + b.lng) / 2 };
      $('widthKm').value = +(Math.max(50, Math.abs(dx)) / 1000).toFixed(3);
      $('heightKm').value = +(Math.max(50, Math.abs(dy)) / 1000).toFixed(3);
    }
    redrawSelection();
  } else if (dragStart && e.pointerId === dragStart.pointer) {
    const ll = map.mouseEventToLatLng(e);
    state.center = { lat: dragStart.center.lat + (ll.lat - dragStart.ll.lat), lon: dragStart.center.lon + (ll.lng - dragStart.ll.lng) };
    redrawSelection();
  }
});
for (const t of ['pointerup', 'pointercancel']) {
  window.addEventListener(t, (e) => {
    if (drawStart && e.pointerId === drawStart.pointer) { drawStart = null; setDrawing(false); saveSettings(); }
    if (dragStart && e.pointerId === dragStart.pointer) endMove();
  });
}
if (!COARSE) selection.bindTooltip('Drag to move · drag the white handles to resize', { sticky: true });

// USGS 3DEP has no national footprint file on Planetary Computer: tile footprints are searched
// per map view, only when zoomed in far enough for one search to cover it
const US_REGIONS = [[-125.0, 24.0, -66.5, 49.5], [-180.0, 51.0, -129.5, 71.6], [-160.5, 18.8, -154.6, 22.4], [-67.5, 17.5, -64.4, 18.6]];
let usLayer = null, usTimer = null, usKey = '';
function viewInUs(b) {
  return US_REGIONS.some(([w, s, e, n]) => b.getWest() <= e && b.getEast() >= w && b.getSouth() <= n && b.getNorth() >= s);
}
function scheduleUsCoverage() {
  clearTimeout(usTimer);
  usTimer = setTimeout(loadUsCoverage, 500);
}
let usgsLayer = null, usgsLoading = false;
function loadUsgsProjects() {
  if (usgsLayer || usgsLoading) return;
  usgsLoading = true;
  $('usCovHint').textContent = '· loading USGS 3DEP coverage…';
  fetch('/api/projects/usgs').then((r) => (r.ok ? r.json() : null)).then((gj) => {
    usgsLoading = false;
    $('usCovHint').textContent = gj ? '' : '· USGS coverage unavailable';
    if (!gj) return;
    usgsLayer = L.geoJSON(gj, {
      renderer: L.canvas({ padding: 0.5 }),  // 2,000+ project outlines
      style: { color: '#2b7a78', weight: 0.6, opacity: 0.6, fillColor: '#3aafa9', fillOpacity: 0.08 },
      onEachFeature: (f, layer) => layer.bindTooltip(tooltipText(f.properties), { sticky: true }),
    });
    loadUsCoverage();
  }).catch(() => { usgsLoading = false; });
}
async function loadUsCoverage() {
  const hint = $('usCovHint');
  const b = map.getBounds();
  const show = $('showCoverage').checked && viewInUs(b);
  if ($('usSource').value === 'usgs') {
    // every AWS Entwine project footprint at once (no per-view search)
    if (usLayer) map.removeLayer(usLayer);
    if (!show) { if (usgsLayer) map.removeLayer(usgsLayer); if (!usgsLoading) hint.textContent = ''; return; }
    if (!usgsLayer) { loadUsgsProjects(); return; }
    hint.textContent = '';
    usgsLayer.addTo(map); selection.bringToFront();
    return;
  }
  if (usgsLayer) map.removeLayer(usgsLayer);
  hint.textContent = '';
  if (!show) { if (usLayer) map.removeLayer(usLayer); return; }
  if ((b.getEast() - b.getWest()) * (b.getNorth() - b.getSouth()) > 0.35) {
    hint.textContent = '· zoom in to see USGS 3DEP tiles';
    if (usLayer) map.removeLayer(usLayer);
    return;
  }
  const q = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(3));
  const key = q.join(',');
  if (key === usKey && usLayer) { usLayer.addTo(map); selection.bringToFront(); return; }
  try {
    const r = await fetch(`/api/coverage/usgs?w=${q[0]}&s=${q[1]}&e=${q[2]}&n=${q[3]}`);
    if (!r.ok) { hint.textContent = '· USGS coverage unavailable'; return; }
    const gj = await r.json();
    if (usLayer) map.removeLayer(usLayer);
    usKey = key;
    usLayer = L.geoJSON(gj, {
      style: { stroke: false, fillColor: '#3aafa9', fillOpacity: 0.12 },
      onEachFeature: (f, layer) => layer.bindTooltip(tooltipText(f.properties), { sticky: true }),
    });
    if ($('showCoverage').checked) { usLayer.addTo(map); selection.bringToFront(); }
  } catch (e) { hint.textContent = ''; }
}

function loadCoverage() {
  fetch('/api/projects').then((r) => (r.ok ? r.json() : null)).then((gj) => {
    if (!gj) return;
    coverageLayer = L.geoJSON(gj, {
      style: { color: '#2b7a78', weight: 1, fillColor: '#3aafa9', fillOpacity: 0.12 },
      interactive: true,
      onEachFeature: (f, layer) => layer.bindTooltip(tooltipText(f.properties), { sticky: true }),
    });
    if ($('showCoverage').checked) coverageLayer.addTo(map);
    selection.bringToFront();
  });
}

// ------------------------------------------------------------------ derived info
function updateDerived() {
  const { w, h } = areaDims();
  const longM = Math.max(w, h);
  const size = parseFloat($('sizeMm').value) || 200;
  const cells = parseInt($('gridCells').value);
  $('gridCellsOut').textContent = cells;
  updateGridCap();
  $('smoothingOut').textContent = $('smoothing').value;
  $('cliffOut').textContent = $('cliffSmoothing').value;
  $('zExOut').textContent = parseFloat($('zEx').value).toFixed(1);
  $('scaleOut').value = `1:${Math.round((longM * 1000) / size).toLocaleString()}`;
  const mPerCell = longM / cells, mmPerCell = size / cells;
  let msg = `${mPerCell.toFixed(2)} m per cell on the ground · ${mmPerCell.toFixed(2)} mm per cell printed · model ${size.toFixed(0)} × ${(size * Math.min(w, h) / longM).toFixed(0)} mm`;
  if (mmPerCell < 0.2) msg += ' — finer than a 0.4 mm nozzle can show; lower the detail to save time.';
  if (mPerCell < 0.5) msg += ' — finer than typical LiDAR density (expect interpolation).';
  $('detailInfo').textContent = msg;
  const tx = parseInt($('tilesX').value) || 1, ty = parseInt($('tilesY').value) || 1;
  const pw = size * (w >= h ? 1 : w / h), ph = size * (h >= w ? 1 : h / w);
  $('tileInfo').textContent = tx * ty > 1 ? `${tx * ty} tiles of ≈ ${(pw / tx).toFixed(0)} × ${(ph / ty).toFixed(0)} mm` : '';
  $('frameOpts').hidden = !$('frame').checked;
  $('labelOpts').hidden = !$('frame').checked;
  $('labelMore').hidden = !$('label').checked;
  $('lipOpts').hidden = !$('baseSeparate').checked;
  $('cornerWrap').hidden = $('shape').value !== 'rounded';
  const mode = $('colorMode').value;
  $('colorsSingle').hidden = mode !== 'single';
  $('colorsLandcover').hidden = mode !== 'landcover';
  $('colorsElevation').hidden = mode !== 'elevation';
  $('depthWrap').hidden = mode === 'single';
  // the water picker lives with the land-cover colours; show a copy in elevation mode
  const wl = document.querySelector('[data-color="water"]').parentElement;
  const target = mode === 'elevation' ? $('waterColorWrap') : $('colorsLandcover');
  if (wl.parentElement !== target) target.insertBefore(wl, mode === 'elevation' ? null : target.children[1] || null);
  $('waterColorWrap').hidden = mode !== 'elevation' || !$('elevWater').checked;
  document.querySelector('[data-color="base"]').parentElement.hidden = mode === 'single';
}

let covTimer = null;
function scheduleCoverage() {
  clearTimeout(covTimer);
  covTimer = setTimeout(checkCoverage, 350);
}
async function checkCoverage() {
  const { w, h } = areaDims();
  const box = $('coverageInfo');
  try {
    const r = await fetch('/api/coverage', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: state.center.lat, lon: state.center.lon, width_m: w, height_m: h,
        grid_cells: parseInt($('gridCells').value), us_source: $('usSource').value, source: $('source').value,
        ground_only: $('surface').value === 'terrain' && $('colorMode').value !== 'landcover',
        dem_preference: $('demPref').value }),
    });
    const d = await r.json();
    box.className = 'info';
    if (!d.ready) { box.textContent = `LiDAR index loading… ${d.message || ''}`; return; }
    // elevation models that fill whatever the point clouds don't cover (best resolution first)
    const dems = (d.dem_sources || []).filter((x) => x.usable).map((x) => x.label);
    const demText = dems.length ? `Elevation models: ${dems.join(' → ')}` : '';
    if (d.models_suffice) {
      box.className = 'info';
      box.innerText = 'Bare terrain at this detail: the elevation models are as fine as the grid, no point clouds needed.\n' + demText;
      return;
    }
    if (!d.tiles && !d.ept_projects) {
      box.className = 'info';
      box.innerText = ($('source').value === 'lidar' ? 'No LiDAR point clouds here.\n' : '') + demText
        + (d.errors && d.errors.length ? `\n${d.errors[0]}` : '');
      return;
    }
    const projects = d.projects.map((p) => `${p.name}${p.year ? ' (' + p.year + ')' : ''}` + (p.tiles != null ? `: ${p.tiles}` : '')).join('\n');
    const parts = [];
    if (d.tiles) parts.push(`${d.tiles} LiDAR tile${d.tiles > 1 ? 's' : ''}`);
    if (d.ept_projects) parts.push(`${d.ept_projects} USGS 3DEP project${d.ept_projects > 1 ? 's' : ''} (AWS)`);
    const tooBig = d.tiles > d.max_tiles || (d.ept_projects && d.area_km2 > (d.max_ept_km2 ?? d.max_tiles));
    const big = d.tiles > 300 || (d.ept_projects && d.area_km2 > 300);
    box.className = tooBig ? 'info err' : big ? 'info warn' : 'info';
    box.innerText = parts.join(' + ') +
      (tooBig ? ` — over the limit at this detail (${d.tiles > d.max_tiles ? d.max_tiles + ' tiles' : (d.max_ept_km2 ?? d.max_tiles) + ' km²'}): elevation models will be used` : big ? ' — large request' : '') +
      `\n${projects}` + (demText ? `\nGaps: ${demText}` : '');
  } catch (e) { box.textContent = ''; }
}

var LIMITS = null;   // var: read by updateDerived during start-up
// Models are built in the browser ("this device"). The server builds them only for browsers that
// can't (``deviceSupported``) and, one model at a time, when a build on the device fails and the
// user takes the failure dialog's offer.
let buildTarget = 'device';
let deviceOk = true;
function onDevice() { return buildTarget === 'device'; }
// Building on this device needs WebAssembly, module workers and gzip streams (the private file
// system, for the data cache, is optional); without them, or with under 2 GB of RAM, the server
// builds.
function deviceSupported() {
  if (typeof WebAssembly !== 'object' || typeof Worker !== 'function' || !('DecompressionStream' in window)) return false;
  if (navigator.deviceMemory && navigator.deviceMemory < 2) return false;
  let module = false;
  try {
    const u = URL.createObjectURL(new Blob([''], { type: 'text/javascript' }));
    new Worker(u, { get type() { module = true; return 'module'; } }).terminate();
    URL.revokeObjectURL(u);
  } catch (e) { return false; }
  return module;
}
function initBuildTarget() {
  deviceOk = deviceSupported();
  buildTarget = deviceOk ? 'device' : 'server';
}
// space for downloaded data kept between visits (web/opfs-cache.js; at most 30% of the quota)
function deviceCacheMb() { return COARSE || MOBILE.matches ? 512 : 2048; }
// Memory a model may use when built in this browser: WebAssembly (32-bit) stops at 4 GB, and a
// tab gets much less on phones. navigator.deviceMemory (Chrome, Edge; GB, at most 8) when known.
function deviceMemoryMb() {
  const ram = navigator.deviceMemory ? navigator.deviceMemory * 1024 : null;
  const phone = COARSE || MOBILE.matches;
  let mb = ram ? ram * (phone ? 0.3 : 0.5) : (phone ? 1200 : 2500);
  return Math.round(Math.max(800, Math.min(phone ? 1600 : 3600, mb)));
}
function gridCap(target = buildTarget) {   // largest Detail this device (or the server) accepts (limits.max_cells)
  if (!LIMITS) return null;
  const memory = target === 'device' ? deviceMemoryMb() : LIMITS.memory_mb;
  const { w, h } = areaDims();
  const aspect = Math.min(w, h) / Math.max(w, h);
  const mode = $('colorMode').value;
  const groundOnly = $('surface').value === 'terrain' && mode !== 'landcover';
  let per = LIMITS.build_mb[mode] || LIMITS.build_mb.landcover;
  if ($('source').value === 'lidar' && target !== 'device') per = Math.max(per, groundOnly ? LIMITS.fetch_mb.ground : LIMITS.fetch_mb.classes);
  const mv = Math.max(0.05, (memory * 0.9 - LIMITS.base_mb) / per);
  return Math.min(2400, Math.floor(Math.sqrt(mv * 1e6 / Math.max(aspect, 0.05)) / 50) * 50);
}
function updateGridCap() {
  const cap = gridCap(), el = $('gridCap');
  if (cap == null) { el.hidden = true; return; }
  const cells = parseInt($('gridCells').value);
  el.hidden = cells <= cap;
  el.textContent = `Over ${onDevice() ? 'this device' : "this server"}'s limit for these colours: ${cap} cells at most`;
  el.classList.toggle('warn', cells > cap);
}
async function pollIndex() {
  try {
    const d = await (await fetch('/api/status')).json();
    if (d.limits) { LIMITS = d.limits; updateGridCap(); }
    const el = $('indexStatus');
    if (d.index.ready) {
      el.textContent = `● ${d.index.message}`; el.className = 'status';
      if (!coverageLayer) loadCoverage();
      scheduleCoverage();
      return;
    }
    if (d.index.state === 'error') { el.textContent = `Index error: ${d.index.error}`; el.className = 'status err'; return; }
    el.textContent = `Preparing LiDAR index (one-time, ~230 MB): ${d.index.message || ''}`;
    if (d.projects_ready && !coverageLayer) loadCoverage();
  } catch (e) { $('indexStatus').textContent = 'Server unreachable'; }
  setTimeout(pollIndex, 1500);
}

// ------------------------------------------------------------------ search (Nominatim)
let searchTimer = null;
$('search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = $('search').value.trim();
  if (q.length < 3) { $('searchResults').innerHTML = ''; return; }
  searchTimer = setTimeout(async () => {
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&q=${encodeURIComponent(q)}`;
    try {
      const res = await (await fetch(url, { headers: { 'Accept-Language': navigator.language || 'en' } })).json();
      $('searchResults').innerHTML = '';
      for (const r of res) {
        const li = document.createElement('li');
        li.textContent = r.display_name;
        li.onclick = () => {
          state.center = { lat: parseFloat(r.lat), lon: parseFloat(r.lon) };
          $('searchResults').innerHTML = ''; $('search').value = r.display_name.split(',')[0];
          $('search').blur(); setSheet(false);   // phones: show the map with the new area
          redrawSelection();
          map.fitBounds(selection.getBounds().pad(0.4));
          saveSettings();
        };
        $('searchResults').appendChild(li);
      }
    } catch (e) { /* offline */ }
  }, 450);
});

// ------------------------------------------------------------------ settings
const FIELDS = ['shape', 'widthKm', 'heightKm', 'source', 'surface', 'gridCells', 'smoothing', 'cliffSmoothing', 'baseSeparate',
  'baseLip', 'lipH', 'lipW', 'lipC', 'label', 'labelText', 'labelScale', 'labelNorth', 'layerH', 'firstLayer',
  'minFeature', 'simplify', 'sizeMm', 'baseMm', 'floorMm',
  'zEx', 'maxHeight', 'cornerMm', 'frame', 'frameW', 'frameH', 'tilesX', 'tilesY', 'flattenWater', 'voidsWater', 'osmWater',
  'usSource', 'demPref', 'waterRecess', 'shoreSlope', 'osmStreams', 'riverMin', 'osmRoads', 'roadMin', 'roadEmboss', 'colorMode', 'bandCount', 'bandMode', 'bandThresholds', 'minRegion',
  'colorDepth', 'swapBudget', 'elevWater',
  'name', 'layout', 'showCoverage'];

// option name (server) -> form element id
const OPTION_IDS = { shape: 'shape', corner_radius_mm: 'cornerMm', source: 'source', us_source: 'usSource', dem_preference: 'demPref', surface: 'surface', grid_cells: 'gridCells',
  smoothing: 'smoothing', cliff_smoothing: 'cliffSmoothing', base_separate: 'baseSeparate', base_lip: 'baseLip',
  lip_height_mm: 'lipH', lip_width_mm: 'lipW', lip_clearance_mm: 'lipC', label: 'label', label_text: 'labelText',
  label_scale_bar: 'labelScale', label_north: 'labelNorth', layer_height_mm: 'layerH', first_layer_mm: 'firstLayer',
  min_feature_mm: 'minFeature', simplify_mm: 'simplify', size_mm: 'sizeMm', base_mm: 'baseMm', floor_mm: 'floorMm', z_exaggeration: 'zEx',
  max_height_mm: 'maxHeight', frame: 'frame', frame_width_mm: 'frameW', frame_height_mm: 'frameH', tiles_x: 'tilesX',
  tiles_y: 'tilesY', flatten_water: 'flattenWater', water_recess_mm: 'waterRecess', shore_slope_deg: 'shoreSlope', voids_as_water: 'voidsWater',
  osm_water: 'osmWater', osm_streams: 'osmStreams', river_min_width_mm: 'riverMin', osm_roads: 'osmRoads', road_min_width_mm: 'roadMin', road_emboss_mm: 'roadEmboss',
  color_mode: 'colorMode', min_region_mm2: 'minRegion', color_depth_mm: 'colorDepth', swap_patch_budget_mm2: 'swapBudget', elevation_water: 'elevWater', layout: 'layout', name: 'name' };

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// Markup goes into the page only through html`...` and setHtml: every ${value} is escaped (text
// from the server or upstream data services - names, credits, errors - is never markup), except
// html`...` fragments and arrays of them, which are joined as they are. null / undefined / false
// add nothing.
class SafeHtml { constructor(s) { this.s = s; } toString() { return this.s; } }
function htmlValue(v) {
  if (v instanceof SafeHtml) return v.s;
  if (Array.isArray(v)) return v.map(htmlValue).join('');
  return v == null || v === false ? '' : escapeHtml(v);
}
function html(strings, ...values) {
  return new SafeHtml(strings.reduce((out, s, i) => out + htmlValue(values[i - 1]) + s));
}
function joinHtml(items, sep) {   // items (escaped) with a markup separator between them
  return items.flatMap((x, i) => (i ? [sep, x] : [x]));
}
function setHtml(el, h) {
  if (!(h instanceof SafeHtml)) throw new TypeError('setHtml: markup must come from html``');
  el.innerHTML = h.s;
  // colour swatches: set through the CSSOM (the CSP allows no inline style attributes)
  for (const sw of el.querySelectorAll('[data-swatch]')) sw.style.background = sw.dataset.swatch;
}
// a Leaflet tooltip's text (Leaflet treats a string as markup)
function tooltipText(p) { return escapeHtml(`${p.project}${p.year ? ` (${p.year})` : ''}`); }

// visible label of a form field (its <label>'s own text, without option texts)
function fieldLabel(el) {
  if (el.dataset.label) return el.dataset.label;   // where the visible label is too terse out of context
  const lab = el.closest('label');
  if (!lab) return el.id;
  const t = [...lab.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
  return t || el.id;
}
function rangeText(el) {
  return el.min !== '' && el.max !== '' ? `${el.min} to ${el.max}` : el.min !== '' ? `at least ${el.min}` : `at most ${el.max}`;
}
// a saved value still fits the field (inputs change between versions: ranges, options)
function fitsField(el, v) {
  if (el.type === 'checkbox') return typeof v === 'boolean';
  if (el.type === 'number' || el.type === 'range') {
    if (v === '' || v === null) return el.type === 'number';
    const x = parseFloat(v);
    if (!Number.isFinite(x)) return false;
    return !(el.min !== '' && x < parseFloat(el.min)) && !(el.max !== '' && x > parseFloat(el.max));
  }
  if (el.tagName === 'SELECT') return [...el.options].some((o) => o.value === String(v));
  return true;
}
function showSettingsNote(msg) {
  const el = $('settingsNote');
  el.hidden = !msg;
  el.innerText = msg || '';
}
// out-of-range number fields (hidden ones too: the server checks every option)
function invalidFields() {
  const bad = [];
  for (const id of new Set([...FIELDS, ...Object.values(OPTION_IDS)])) {
    const el = $(id);
    if (!el || el.type !== 'number' || el.value === '') continue;
    if (el.validity.badInput || el.validity.rangeUnderflow || el.validity.rangeOverflow) bad.push(el);
  }
  return bad;
}

// ------------------------------------------------------------------ Simple / Advanced settings
// Advanced settings (class="adv") are only hidden in Simple mode: their values still apply, so
// Simple mode lists any that differ from the defaults, with a reset.
const MODE_KEY = 'maptomesh.mode';
const MODE_HINT = { simple: 'Essentials only; the defaults suit most prints', advanced: 'Every setting, for fine-tuning' };
const ADV_IDS = [...new Set([...FIELDS, ...Object.values(OPTION_IDS)])].filter((id) => $(id)?.closest('.adv'));

function isSimple() { return document.documentElement.dataset.mode !== 'advanced'; }
function setMode(mode) {
  document.documentElement.dataset.mode = mode;
  try { localStorage.setItem(MODE_KEY, mode); } catch (e) { /* private mode */ }
  refreshMode();
}
function defaultOf(el) {   // the value in the HTML
  if (el.type === 'checkbox') return el.defaultChecked;
  if (el.tagName === 'SELECT') return ([...el.options].find((o) => o.defaultSelected) || el.options[0]).value;
  return el.defaultValue;
}
function differsFromDefault(el) {
  const d = defaultOf(el);
  if (el.type === 'checkbox') return el.checked !== d;
  if ((el.type === 'number' || el.type === 'range') && el.value !== '' && d !== '') return parseFloat(el.value) !== parseFloat(d);
  return el.value.trim() !== String(d).trim();
}
// changed advanced settings that matter now (not those of a switched-off feature, e.g. lip sizes
// without a separate base plate)
function changedAdvanced() {
  return ADV_IDS.map($).filter((el) => !el.closest('[hidden]') && differsFromDefault(el));
}
function shownValue(el) {
  if (el.type === 'checkbox') return el.checked ? 'on' : 'off';
  if (el.tagName === 'SELECT') return el.selectedOptions[0]?.textContent.split(/[(:]/)[0].trim() ?? el.value;
  return el.value === '' ? 'none' : el.value;
}
function refreshMode() {
  const simple = isSimple();
  document.querySelectorAll('.seg [data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === (simple ? 'simple' : 'advanced'))));
  $('modeHint').textContent = MODE_HINT[simple ? 'simple' : 'advanced'];
  const changed = simple ? changedAdvanced() : [];
  const note = $('advNote');
  note.hidden = !changed.length;
  if (!changed.length) { note.innerHTML = ''; return; }
  const list = joinHtml(changed.map((el) => html`${fieldLabel(el)} <b>${shownValue(el)}</b>`), ', ');
  setHtml(note, html`<span class="adv-list">Also using ${changed.length} changed advanced setting${changed.length > 1 ? 's' : ''}: ${list}</span><button
    type="button" class="secondary" data-act="show">Show them</button><button
    type="button" class="secondary" data-act="reset">Reset to defaults</button>`);
}
function resetAdvanced() {
  for (const el of changedAdvanced()) {
    if (el.type === 'checkbox') el.checked = defaultOf(el); else el.value = defaultOf(el);
  }
  redrawSelection();   // updates the derived fields and the coverage check
  scheduleUsCoverage();
  saveSettings();
  refreshMode();
}
document.querySelectorAll('.seg [data-mode]').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
$('advNote').addEventListener('click', (e) => {
  const act = e.target.closest('button')?.dataset.act;
  if (act === 'show') setMode('advanced');
  else if (act === 'reset') resetAdvanced();
});
// FastAPI / pydantic validation errors -> "Simplify (mm): must be less than or equal to 0.5 (got 1)"
function describeError(d) {
  if (typeof d.detail === 'string') return d.detail;
  if (!Array.isArray(d.detail)) return JSON.stringify(d.detail ?? d);
  return d.detail.map((e) => {
    const key = (e.loc || []).slice(-1)[0];
    const el = OPTION_IDS[key] && $(OPTION_IDS[key]);
    const msg = String(e.msg || '').replace(/^Input should be /, 'must be ');
    return `${el ? fieldLabel(el) : key}: ${msg}` + ('input' in e ? ` (got ${JSON.stringify(e.input)})` : '');
  }).join('\n');
}

function saveSettings() {
  const s = { center: state.center, zoom: map.getZoom(), mapCenter: map.getCenter(), colors: {} };
  for (const f of FIELDS) { const el = $(f); s[f] = el.type === 'checkbox' ? el.checked : el.value; }
  document.querySelectorAll('[data-color]').forEach((el) => { s.colors[el.dataset.color] = el.value; });
  localStorage.setItem(STORE_KEY, JSON.stringify(s));
}
function loadSettings() {
  let s;
  try { s = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch (e) { s = null; }
  if (!s) return;
  const reset = [];
  for (const f of FIELDS) {
    if (!(f in s)) continue;
    const el = $(f);
    if (!fitsField(el, s[f])) { reset.push(`${fieldLabel(el)} (${JSON.stringify(s[f])})`); continue; }
    if (el.type === 'checkbox') el.checked = s[f]; else el.value = s[f];
  }
  if (reset.length) {
    showSettingsNote(`Reset to defaults, the saved values no longer fit: ${reset.join(', ')}.`);
  }
  if (s.center) state.center = s.center;
  const bands = Object.keys(s.colors || {}).filter((k) => k.startsWith('band')).sort((a, b) => a.slice(4) - b.slice(4)).map((k) => s.colors[k]);
  renderBandInputs(bands.length ? bands : null);
  for (const [k, v] of Object.entries(s.colors || {})) {
    const el = document.querySelector(`[data-color="${k}"]`); if (el) el.value = v;
  }
  if (s.mapCenter) map.setView(s.mapCenter, s.zoom || 14);
}

// ------------------------------------------------------------------ generate
function num(id, fallback) { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : fallback; }

function buildOptions() {
  const { w, h } = areaDims();
  const mode = $('colorMode').value;
  const bands = [...$('bandColors').querySelectorAll('input')].map((i) => i.value.toUpperCase());
  const th = $('bandThresholds').value.split(/[,;\s]+/).map(parseFloat).filter(Number.isFinite);
  return {
    lat: state.center.lat, lon: state.center.lon, width_m: w, height_m: h,
    shape: $('shape').value, corner_radius_mm: num('cornerMm', 12),
    source: $('source').value, us_source: $('usSource').value, dem_preference: $('demPref').value, surface: $('surface').value,
    grid_cells: parseInt($('gridCells').value), smoothing: num('smoothing', 0),
    cliff_smoothing: num('cliffSmoothing', 1),
    size_mm: num('sizeMm', 200), base_mm: num('baseMm', 3), floor_mm: num('floorMm', 1.5),
    base_separate: $('baseSeparate').checked, base_lip: $('baseLip').checked,
    lip_height_mm: num('lipH', 1.2), lip_width_mm: num('lipW', 1.6), lip_clearance_mm: num('lipC', 0.2),
    label: $('label').checked, label_text: $('labelText').value, label_scale_bar: $('labelScale').checked,
    label_north: $('labelNorth').checked, layer_height_mm: num('layerH', 0.2), first_layer_mm: num('firstLayer', 0.2),
    min_feature_mm: num('minFeature', 0.5), simplify_mm: num('simplify', 0.05),
    z_exaggeration: num('zEx', 1), max_height_mm: $('maxHeight').value ? num('maxHeight', null) : null,
    frame: $('frame').checked, frame_width_mm: num('frameW', 6), frame_height_mm: num('frameH', 2),
    tiles_x: parseInt($('tilesX').value) || 1, tiles_y: parseInt($('tilesY').value) || 1,
    flatten_water: $('flattenWater').checked, water_recess_mm: num('waterRecess', 0.6),
    shore_slope_deg: num('shoreSlope', 70),
    voids_as_water: $('voidsWater').checked, osm_water: $('osmWater').checked,
    osm_streams: $('osmStreams').value, river_min_width_mm: num('riverMin', 0.8),
    osm_roads: $('osmRoads').value, road_min_width_mm: num('roadMin', 0.6), road_emboss_mm: num('roadEmboss', 0),
    color_mode: mode, min_region_mm2: num('minRegion', 1), color_depth_mm: num('colorDepth', 1), swap_patch_budget_mm2: num('swapBudget', 50), elevation_water: $('elevWater').checked,
    colors: {
      single: colorVal('single'), base: colorVal('base'), frame: colorVal('frame'),
      ground: colorVal('ground'), water: colorVal('water'), buildings: colorVal('buildings'),
      roads: colorVal('roads'), vegetation: colorVal('vegetation'), label: colorVal('label'),
      bands, band_mode: $('bandMode').value,
      band_thresholds_m: th.length === bands.length - 1 ? th : null,
    },
    layout: $('layout').value, name: $('name').value || 'MapToMesh',
  };
}

// restore the form from a job's options (inverse of buildOptions)
function applyOptions(o) {
  if (!o) return;
  const set = (id, v) => {
    const el = $(id);
    if (v === null || v === undefined) el.value = '';
    else if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v;
  };
  state.center = { lat: o.lat, lon: o.lon };
  for (const [k, id] of Object.entries(OPTION_IDS)) if (k in o) set(id, o[k]);
  $('widthKm').value = +(o.width_m / 1000).toFixed(3);
  $('heightKm').value = +(o.height_m / 1000).toFixed(3);
  const c = o.colors || {};
  for (const k of ['single', 'base', 'frame', 'label', 'ground', 'water', 'buildings', 'roads', 'vegetation']) {
    const el = document.querySelector(`[data-color="${k}"]`);
    if (el && c[k]) el.value = c[k];
  }
  if (c.bands) { $('bandCount').value = c.bands.length; renderBandInputs(c.bands); }
  if (c.band_mode) $('bandMode').value = c.band_mode;
  $('bandThresholds').value = (c.band_thresholds_m || []).join(', ');
  fillPresets();
  redrawSelection();
  map.fitBounds(selection.getBounds().pad(0.3));
  saveSettings();
  refreshMode();
}

// progress shows in the panel and, on phones, in the sheet's always-visible header
function setProgress(frac, msg, isError = false) {
  if (frac != null) { const w = `${(frac * 100).toFixed(1)}%`; $('bar').style.width = w; $('miniBar').style.width = w; }
  if (msg != null) {
    $('progressMsg').textContent = msg;
    $('miniMsg').textContent = msg.split('\n')[0];
    $('miniMsg').classList.toggle('err', isError);
    $('miniStatus').hidden = false;
  }
}
function setRunning(on) {
  state.running = on;
  $('generateBtn').disabled = on;
  $('cancelBtn').hidden = !on;
  const mg = $('mobileGenerate');
  mg.textContent = on ? 'Cancel' : 'Generate 3MF';
  mg.classList.toggle('primary', !on); mg.classList.toggle('secondary', on);
}

async function generate({ keepUrl = false, target = buildTarget } = {}) {
  if (!keepUrl && location.search) history.replaceState(null, '', location.pathname);
  const bad = invalidFields();
  if (bad.length) {
    showSettingsNote(`Fix these settings first:\n${bad.map((el) => `${fieldLabel(el)}: ${el.value} (allowed ${rangeText(el)})`).join('\n')}`);
    if (isSimple() && bad.some((el) => el.closest('.adv'))) setMode('advanced');   // show the culprits
    setSheet(true);
    $('settingsNote').scrollIntoView({ block: 'center', behavior: 'smooth' });
    return;
  }
  showSettingsNote('');
  saveSettings();
  const body = buildOptions();
  setRunning(true);
  $('progress').hidden = false; $('log').hidden = false;
  $('log').textContent = '';
  $('miniDownload').hidden = true; $('miniDetails').hidden = true;
  setProgress(0, 'Starting…');
  $('result').hidden = true;
  hideReview();
  if (target === 'device') { runOnDevice(body); return; }
  if (!deviceOk) setProgress(0, "Building on the server: this browser can't build models itself");
  runOnServer(body);
}

async function runOnServer(body) {
  try {
    const r = await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok && d.error_kind) {   // refused up front: too large / server busy
      finish();
      setProgress(null, `Not started: ${d.detail}`, true);
      showFailure({ error_kind: d.error_kind, error: d.detail, advice: d.advice, suggest: d.suggest });
      return;
    }
    if (!r.ok) throw new Error(describeError(d));
    state.jobId = d.id;
    state.cancelToken = d.cancel_token;   // the server cancels a job only with this
    poll();
  } catch (e) {
    finish();
    setProgress(null, `Error: ${e.message}`, true);
    $('miniDetails').hidden = false;
  }
}

function finish() {
  setRunning(false);
  clearTimeout(state.pollTimer);
}

async function poll() {
  const id = state.jobId;
  try {
    const j = await (await fetch(`/api/jobs/${id}`)).json();
    setProgress(j.progress, j.message);
    $('log').textContent = j.log.join('\n');
    $('log').scrollTop = 1e9;
    if (j.state === 'done') {
      finish(); showResult(j); showReview(j);
      history.replaceState(null, '', `?job=${id}`);
      return;
    }
    if (j.state === 'error' || j.state === 'cancelled') {
      finish();
      setProgress(null, j.state === 'error' ? `Error: ${j.error}` : 'Cancelled', j.state === 'error');
      $('miniDetails').hidden = j.state !== 'error';
      if (j.state === 'error') showReview(j);
      if (j.state === 'error' && FAIL_TITLES[j.error_kind]) showFailure(j);
      return;
    }
  } catch (e) { /* transient */ }
  state.pollTimer = setTimeout(poll, 700);
}

// ------------------------------------------------------------------ building on this device
// "Build on: this device": a Web Worker (web/browser-worker.js) runs the maptomesh Python package
// compiled to WebAssembly (Pyodide); the server only serves files and forwards data from hosts
// without CORS. The results stay in the browser (blob URLs).
// fullScipy: this device needed a part of SciPy that the trimmed copy leaves out (remembered)
const FULL_SCIPY_KEY = 'maptomesh.fullScipy';
const device = { worker: null, ready: false, t0: 0, log: [], urls: [], opts: null,
                 fullScipy: (() => { try { return localStorage.getItem(FULL_SCIPY_KEY) === '1'; } catch (e) { return false; } })() };

function cancelJob() {
  if (!state.running) return;
  if (state.onDevice) {
    if (device.worker) device.worker.terminate();   // its Python goes too; the next run starts afresh
    device.worker = null;
    state.onDevice = false;
    finish();
    setProgress(null, 'Cancelled');
    return;
  }
  if (state.jobId) fetch(`/api/jobs/${state.jobId}/cancel`, { method: 'POST', headers: { 'X-Cancel-Token': state.cancelToken || '' } });
}

function localJobId() {   // the server's job-id format, so a review of this model can be sent
  return [...crypto.getRandomValues(new Uint8Array(6))].map((x) => x.toString(16).padStart(2, '0')).join('');
}

function deviceLog(msg) {
  if (device.log.length && device.log[device.log.length - 1].endsWith(`  ${msg}`)) return;
  device.log.push(`${((performance.now() - device.t0) / 1000).toFixed(1)}s  ${msg}`);
  $('log').textContent = device.log.join('\n');
  $('log').scrollTop = 1e9;
}

async function runOnDevice(body, { retry = false } = {}) {
  const cap = gridCap();
  if (cap != null && body.grid_cells > cap) {
    finish();
    const msg = `${body.grid_cells} cells of detail need more memory than a browser tab on this device can use.`;
    setProgress(null, `Not started: ${body.grid_cells} cells is over this device's limit (${cap})`, true);
    reportDeviceFailure('too_large', msg, '', body);
    showFailure({ error_kind: 'too_large', device: true, title: 'This model is too large for this device',
                  error: msg, advice: [`Lower Detail to ${cap} cells or less.`], suggest: { grid_cells: cap }, server: body });
    return;
  }
  if (!retry && onPhone() && !localStorage.getItem(NOASK_KEY) && !builtBefore(body)) {
    setProgress(0, 'Waiting for your go-ahead to download the data…');
    if (!(await confirmDownload(body))) { finish(); setProgress(null, 'Not started'); return; }
  }
  state.onDevice = true;
  state.jobId = null;
  if (!retry) {                     // a retry keeps the first attempt's log
    device.t0 = performance.now();
    device.log = [];
  }
  device.opts = body;
  device.setup = null;
  startWorker();
  // Chromium grants persistent storage silently (no prompt; Firefox would ask): the kept Python
  // and data then survive storage pressure on phones
  if (navigator.userAgentData && navigator.storage?.persist) navigator.storage.persist().catch(() => {});
  device.worker.postMessage({ type: 'run', opts: body, memory_mb: deviceMemoryMb(), cache_mb: deviceCacheMb(),
                             full_scipy: device.fullScipy });
}

function startWorker() {
  if (device.worker) return;
  device.ready = false;   // until its Python has loaded
  device.worker = new Worker('/static/browser-worker.js', { type: 'module' });
  device.worker.onmessage = onDeviceMessage;
  device.worker.onerror = (e) => {
    if (state.onDevice) deviceFailed({ message: e.message || 'The build worker could not start', fatal: true });
    else { device.worker?.terminate(); device.worker = null; }
  };
}

// Python takes a few seconds to start even when it's kept on the device (more on phones): start
// it in the background once the page is idle, for people building here - never when that would
// download it (first visit: the build asks first on phones)
async function prewarmDevice() {
  if (device.worker || !onDevice() || state.running || document.hidden) return;
  if (!(await runtimeCached())) return;
  if (onPhone() && !builtList().length) return;
  startWorker();
  device.worker.postMessage({ type: 'init', full_scipy: device.fullScipy });
}

async function gunzipToUrl(bytes, type) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return URL.createObjectURL(new Blob([await new Response(stream).arrayBuffer()], { type }));
}

async function onDeviceMessage(e) {
  const m = e.data;
  if (m.type === 'setup') { device.ready = true; device.setup = m.setup; }
  if (m.early) {                      // started ahead of a build
    if (m.type === 'error') { device.worker?.terminate(); device.worker = null; }
    return;
  }
  if (!state.onDevice) return;
  if (m.type === 'progress') {
    setProgress(m.f, m.msg);
    deviceLog(m.msg);
  } else if (m.type === 'setup') {
    const x = m.setup;
    const secs = (x.runtime_s + x.packages_s + x.code_s).toFixed(1);
    const where = m.waited_s < 0.5 ? 'started earlier in the background' : `started in ${secs} s`;
    deviceLog(x.runtime_cached
      ? `Python ready (${where} from this device: nothing downloaded${x.native ? '' : ', without the Rust extension'})`
      : `Python ready (${where}${x.native ? '' : ', without the Rust extension'})`);
    const rf = x.runtime_files;
    if (rf && !x.runtime_cached && x.secure !== false) {
      deviceLog(`Python files: ${rf.cached} kept on this device, ${rf.downloaded} downloaded`
        + (x.downloaded_mb ? ` (${x.downloaded_mb.toFixed(1)} MB)` : '') + (x.scipy ? `, ${x.scipy} SciPy` : '')
        + ((x.runtime_errors || []).length ? ` (couldn't keep: ${x.runtime_errors.join('; ')})` : ''));
    }
    if (x.secure === false) deviceLog('This page is not on a secure connection (https): browsers don\'t let it keep Python or data on this device.');
  } else if (m.type === 'done') {
    device.urls.forEach((u) => URL.revokeObjectURL(u));
    device.urls = [];
    const f = m.files, r = m.res;
    const blob = (name, type) => {
      if (!name || !f[name]) return null;
      const u = URL.createObjectURL(new Blob([f[name]], { type }));
      device.urls.push(u);
      return u;
    };
    const urls = { model: blob(r.file, 'model/3mf'), stl: blob(r.stl_file, 'application/zip'), base: blob(r.base_file, 'model/3mf'),
                   thumb: blob('thumbnail.png', 'image/png'),
                   preview: await gunzipToUrl(f[`preview_${PREVIEW_LOD}.bin.gz`], 'application/octet-stream') };
    device.urls.push(urls.preview);
    const j = { id: localJobId(), state: 'done', result: r, options: device.opts, log: device.log, urls,
                device: { seconds: m.total_s, downloaded_mb: (r.browser?.net?.bytes || 0) / 1e6,
                          cache_mb: m.cache?.persistent ? m.cache.hit_mb : null } };
    state.onDevice = false;
    rememberBuilt(device.opts);
    finish();
    showResult(j);
    showReview(j);
  } else if (m.type === 'error') {
    deviceFailed(m);
  }
}

function deviceFailed(m) {
  if (m.missing_scipy && !device.fullScipy && device.opts) {
    // the trimmed SciPy (native/trim_scipy.py) lacks a part this model needs: reported so it can be
    // added; this device switches to the full SciPy (~7 MB more, once) and the build runs again
    reportDeviceFailure('error', `Trimmed SciPy lacks ${m.missing_scipy}: retried with the full SciPy`, m.trace || '', device.opts);
    device.fullScipy = true;
    try { localStorage.setItem(FULL_SCIPY_KEY, '1'); } catch (e) { /* private mode */ }
    device.worker?.terminate();
    device.worker = null;
    deviceLog(`Loading the rest of SciPy (${m.missing_scipy} is needed) and building again`);
    state.onDevice = false;
    runOnDevice(device.opts, { retry: true });
    return;
  }
  const started = device.ready;
  const body = device.opts;
  const kind = m.error_kind === 'memory' || m.error_kind === 'too_large' ? m.error_kind : started ? 'error' : 'startup';
  reportDeviceFailure(kind, m.message, m.trace || '', body);
  state.onDevice = false;
  finish();
  if (m.fatal && device.worker) { device.worker.terminate(); device.worker = null; }
  setProgress(null, `Error: ${m.message}`, true);
  deviceLog(`Error: ${m.message}`);
  $('miniDetails').hidden = false;
  if (kind === 'memory') {
    const cells = Math.max(150, Math.floor(parseInt($('gridCells').value) * 0.7 / 50) * 50);
    showFailure({ error_kind: 'memory', device: true, error: 'This browser tab ran out of memory building the model.',
                  advice: [`Lower Detail (e.g. ${cells} cells).`, 'Close other tabs.'], suggest: { grid_cells: cells }, server: body });
  } else if (kind === 'too_large') {
    showFailure({ error_kind: 'too_large', device: true, title: 'This model is too large for this device', error: m.message,
                  advice: (m.advice || []).filter((a) => !/server/i.test(a)), suggest: m.suggest || {}, server: body });
  } else {
    showFailure({ error_kind: kind, device: true, error: m.message, server: body,
                  title: kind === 'startup' ? "This browser couldn't start the build" : "The model couldn't be built on this device" });
  }
}

// a failed build on the device, for review on the server (maptomesh/device_failures.py): the
// error, the model's settings and facts about the browser - no personal data
function reportDeviceFailure(kind, message, trace, options) {
  const x = device.setup;
  const body = {
    kind, message: String(message || '').slice(0, 2000), trace: String(trace || '').slice(-8000),
    stage: (device.log[device.log.length - 1] || '').slice(0, 300), options: options || null,
    env: {
      memory_mb: deviceMemoryMb(), device_memory_gb: navigator.deviceMemory ?? null, cores: navigator.hardwareConcurrency ?? null,
      phone: onPhone(), secure: isSecureContext, seconds: device.t0 ? +((performance.now() - device.t0) / 1000).toFixed(1) : null,
      python: x ? { ready: true, jspi: x.jspi, native: x.native, pyodide: x.pyodide, runtime_cached: x.runtime_cached } : { ready: false },
      log: device.log.slice(-15),
    },
  };
  fetch('/api/device-failures', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify(body), keepalive: true }).catch(() => {});
}

// ------------------------------------------------------------------ phones: confirm the download
// Before building on a phone (mobile data), show what the build will download: the area's data
// (server estimate: maptomesh/estimate.py) and, the first time, Python itself. Not asked again for
// an area already built on this device (its data is in the cache), or after "Don't ask again".
const RUNTIME_CACHE = 'maptomesh-runtime-pyodide-314.0.8';   // as in web/browser-worker.js
const RUNTIME_MB = 23;   // Python, its packages and the trimmed SciPy
const NOASK_KEY = 'maptomesh.dataNoAsk';
const BUILT_KEY = 'maptomesh.deviceBuilt.v1';

function onPhone() { return COARSE || MOBILE.matches; }
function dataKey(o) {
  return [o.lat.toFixed(5), o.lon.toFixed(5), Math.round(o.width_m), Math.round(o.height_m), o.grid_cells, o.source,
          o.us_source, o.dem_preference, o.surface, o.color_mode].join('|');
}
function builtList() { try { return JSON.parse(localStorage.getItem(BUILT_KEY) || '[]'); } catch (e) { return []; } }
function builtBefore(o) { return builtList().includes(dataKey(o)); }
function rememberBuilt(o) {
  if (!o) return;
  const k = dataKey(o);
  try { localStorage.setItem(BUILT_KEY, JSON.stringify(builtList().filter((x) => x !== k).concat(k).slice(-100))); } catch (e) { /* full */ }
}
async function runtimeCached() {
  try {
    if (!(await caches.has(RUNTIME_CACHE))) return false;
    return !!(await (await caches.open(RUNTIME_CACHE)).match('/__runtime_complete__'));
  } catch (e) { return false; }
}
const fmtMb = (mb) => (mb < 0.1 ? 'under 0.1 MB' : mb < 10 ? `${mb.toFixed(1)} MB` : `${Math.round(mb)} MB`);

function confirmDownload(body) {
  return new Promise((resolve) => {
    const dlg = $('dataDialog');
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      if (v && $('dataNoAsk').checked) localStorage.setItem(NOASK_KEY, '1');
      if (dlg.open) dlg.close();
      resolve(v);
    };
    setHtml($('dataParts'), html`<li>Working out the download size…</li>`);
    $('dataNote').textContent = '';
    $('dataNoAsk').checked = false;
    $('dataTitle').textContent = 'Download the data on this device?';
    $('dataGo').onclick = () => done(true);
    $('dataCancel').onclick = () => done(false);
    $('dataClose').onclick = () => done(false);
    dlg.onclose = () => done(false);             // Esc / back gesture
    dlg.showModal();
    fillDownload(body).catch(() => {
      $('dataParts').innerHTML = '';
      $('dataNote').textContent = "Couldn't work out the download size. Areas with LiDAR point clouds can need 100 MB or more.";
    });
  });
}

async function fillDownload(body) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 25000);
  try {
    const [est, cached] = await Promise.all([
      fetch('/api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status)))),
      runtimeCached(),
    ]);
    const rows = est.parts.map((p) => [p.label, p.mb, p.exact]);
    if (!cached) rows.push([isSecureContext ? 'Python and its libraries (first build only, then kept)' : 'Python and its libraries (every visit on this connection)',
                            RUNTIME_MB + (device.fullScipy ? 7 : 0), false]);
    const total = rows.reduce((a, r) => a + r[1], 0);
    $('dataTitle').textContent = `Download about ${fmtMb(total)}?`;
    setHtml($('dataParts'), html`${rows.map(([l, mb, exact]) => html`<li><span>${l}</span><b>${exact ? '' : '≈ '}${fmtMb(mb)}</b></li>`)}
      <li class="total"><span>Total</span><b>≈ ${fmtMb(total)}</b></li>`);
    $('dataNote').textContent = [...(est.notes || []),
      isSecureContext ? 'Data this device downloaded before is reused.'
        : "This page isn't on a secure connection (https), so this device can't keep the downloads: they repeat on every build."].join(' ');
  } finally {
    clearTimeout(timer);
  }
}

function fmtBytes(n) { return n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(0)} kB`; }

function jobUrls(j) {
  if (j.urls) return j.urls;     // built on this device: blob URLs
  const id = encodeURIComponent(j.id);
  return { model: `/api/jobs/${id}/model.3mf`, stl: `/api/jobs/${id}/stl.zip`, base: `/api/jobs/${id}/base.3mf`,
           thumb: `/api/jobs/${id}/thumbnail.png?${Date.now()}`, preview: `/api/jobs/${id}/preview.bin?lod=${PREVIEW_LOD}` };
}

function showResult(j) {
  const r = j.result, s = r.stats, u = jobUrls(j);
  state.lastResult = j;
  const br = (items) => joinHtml(items, html`<br>`);
  const swatch = (color) => html`<span class="swatch" data-swatch="${color}"></span>`;
  const fil = r.filaments.map((f) => html`<li>${swatch(f.color)}<b>Filament ${f.slot}</b> ${f.color} — ${(f.used_by || []).join(', ')}</li>`);
  const cc = r.color_changes.length
    ? html`<h2>Single-extruder colour changes</h2><ul class="filaments">${r.color_changes.map((c) => html`<li>${swatch(c.color)}at Z = ${c.z_mm.toFixed(2)} mm${c.elevation_m != null ? ` (${c.elevation_m} m)` : ''}</li>`)}</ul><p class="hint">Add a filament change / pause at these heights (round to your layer height).</p>`
    : '';
  const notes = [...(r.notes || []).filter((n) => !n.startsWith('Licence restriction:')),  // own row below
    ...(r.data.failed_tiles ? [`${r.data.failed_tiles} tile(s) failed to download.`] : [])];
  const projects = (r.data.projects || []).map((p) => p.name + (p.year ? ` (${p.year})` : '')).join(', ');
  const lidar = [r.data.tiles ? `${r.data.tiles} tiles` : '', r.data.ept_nodes ? `${r.data.ept_nodes} USGS octree nodes (AWS)` : ''].filter(Boolean).join(' + ');
  setHtml($('result'), html`
    <h2>Result</h2>
    <img src="${u.thumb}" alt="Top view">
    <div class="downloads">
      <a href="${u.model}" download="${r.file}">⬇ 3MF (${fmtBytes(r.file_size)})</a>
      <a class="alt" href="${u.stl}" download="${r.stl_file || 'model_stl.zip'}">STL per colour (.zip)</a>
    </div>
    ${r.base_file ? html`<div class="downloads"><a class="alt" href="${u.base}" download="${r.base_file}">⬇ Base plate 3MF (${fmtBytes(r.base_file_size)}, ${s.base_plate_mm.join(' × ')} mm)</a></div>` : ''}
    <div class="share-row">
      <button type="button" class="secondary" id="shareBtn">Share link</button>
      <input type="text" id="shareUrl" readonly hidden aria-label="Share link">
    </div>
    <p class="hint" id="shareMsg">The link holds this model's settings: opening it builds the same model.</p>
    <dl class="stats">
      <dt>Model size</dt><dd>${s.size_mm[0]} × ${s.size_mm[1]} × ${s.size_mm[2]} mm</dd>
      ${s.tile_size_mm ? html`<dt>Tiles</dt><dd>${s.tiles} × ${s.tile_size_mm[0]} × ${s.tile_size_mm[1]} mm</dd>` : ''}
      <dt>Scale</dt><dd>${s.scale} (z ×${$('zEx').value})</dd>
      <dt>Elevation</dt><dd>${s.elevation_range_m[0]} – ${s.elevation_range_m[1]} m → relief ${s.relief_mm} mm</dd>
      <dt>Resolution</dt><dd>${s.ground_m_per_cell} m / ${s.cell_mm} mm per cell</dd>
      <dt>Mesh</dt><dd>${s.triangles.toLocaleString()} triangles, ${s.parts} parts</dd>
      ${j.device ? html`<dt>Built on</dt><dd>this device in ${j.device.seconds.toFixed(0)} s (${j.device.downloaded_mb.toFixed(1)} MB of data downloaded${j.device.cache_mb >= 0.05 ? `, ${j.device.cache_mb.toFixed(1)} MB reused from earlier downloads` : ''})</dd>` : ''}
      ${s.colour_changes_est != null ? html`<dt>Colour changes</dt><dd>≈ ${s.colour_changes_est.toLocaleString()} at 0.2 mm layers (max ${s.max_colours_per_layer} colours in a layer)</dd>` : ''}
      ${r.data.points ? html`<dt>LiDAR</dt><dd>${(r.data.points / 1e6).toFixed(1)} M points from ${lidar}</dd>` : ''}
      ${(r.data.sources || []).length ? html`<dt>Data sources</dt><dd>${br(r.data.sources.filter((x) => x.share >= 0.0005).map((x) => `${x.label} ${(100 * x.share).toFixed(0)}%`))}</dd>` : ''}
      ${(r.data.licence_warnings || []).length ? html`<dt>Licence</dt><dd class="licence-warn">${br(r.data.licence_warnings.map((w) => `⚠ ${w}`))}</dd>` : ''}
      ${(r.data.attribution || []).length ? html`<dt>Credits</dt><dd class="credits">${br([...r.data.attribution, 'Map data © OpenStreetMap contributors (ODbL)'])}</dd>` : ''}
      ${projects ? html`<dt>Projects</dt><dd>${projects}</dd>` : ''}
      ${r.data.items ? html`<dt>HRDEM</dt><dd>${[].concat(r.data.items).join(', ')}</dd>` : ''}
    </dl>
    <h2>Filaments</h2><ul class="filaments">${fil}</ul>
    ${cc}
    ${(r.tips || []).length ? html`<h2>Print tips</h2><ul class="tips">${r.tips.map((tip) => html`<li>${tip}</li>`)}</ul>` : ''}
    ${notes.length ? html`<div class="info warn">${br(notes)}</div>` : ''}`);
  $('result').hidden = false;
  setProgress(1, `Done · ${s.size_mm[0]} × ${s.size_mm[1]} mm, ${r.filaments.length} colour${r.filaments.length > 1 ? 's' : ''}`);
  $('miniDownload').href = u.model; $('miniDownload').hidden = false;
  if (j.urls) $('miniDownload').download = r.file; else $('miniDownload').removeAttribute('download');
  $('miniDetails').hidden = false;
  $('topImg').src = u.thumb;
  $('shareBtn').onclick = () => shareModel(j);
  viewer.load(u.preview);
  selectTab('view');
}

// ------------------------------------------------------------------ share links
// Only the settings go to the server (maptomesh/shares.py), never model files: the link rebuilds
// the model when it's opened.
async function shareModel(j) {
  const msg = $('shareMsg'), btn = $('shareBtn');
  btn.disabled = true;
  msg.textContent = 'Creating the link…';
  try {
    const r = await fetch('/api/shares', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                           body: JSON.stringify(j.options) });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(describeError(d));
    const url = `${location.origin}/?s=${d.id}`;
    const box = $('shareUrl');
    box.value = url; box.hidden = false;
    if (navigator.share && (COARSE || MOBILE.matches)) {
      try {
        await navigator.share({ title: `${j.options.name} - MapToMesh`, url });
        msg.textContent = 'Opening the link builds this model.';
        return;
      } catch (e) { /* dismissed: copy it instead */ }
    }
    try {
      await navigator.clipboard.writeText(url);
      msg.textContent = 'Link copied. Opening it builds this model.';
    } catch (e) {
      box.select();
      msg.textContent = 'Copy the link above. Opening it builds this model.';
    }
  } catch (e) {
    msg.textContent = `Couldn't create the link: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// a share link (?s=<id>): load its settings and build the model straight away
async function openShare(id) {
  $('progress').hidden = false;
  setProgress(0, 'Opening the shared model…');
  let d;
  try {
    const r = await fetch(`/api/shares/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error(r.status === 404 ? 'this link is unknown' : `the server answered ${r.status}`);
    d = await r.json();
  } catch (e) {
    setProgress(null, `Couldn't open the shared model: ${e.message}.`, true);
    return;
  }
  applyOptions(d.options);
  for (let i = 0; i < 50 && !LIMITS; i++) await new Promise((res) => setTimeout(res, 100));   // the Detail cap
  generate({ keepUrl: true });
}

// ------------------------------------------------------------------ review / problem report
const PROBLEMS = [
  ['terrain', 'Terrain / elevation looks wrong'],
  ['missing_data', 'Missing or low-detail data for my area'],
  ['buildings', 'Buildings or trees are wrong'],
  ['water', 'Water, rivers or shoreline problems'],
  ['roads', 'Roads are wrong'],
  ['colors', 'Colours / filament changes'],
  ['slicer', "Won't open, slice or print properly"],
  ['slow_failed', 'Generation was slow or failed'],
  ['other', 'Something else'],
];
const REVIEWED_KEY = 'maptomesh.reviewed.v1';
function reviewedJobs() {
  try { return JSON.parse(localStorage.getItem(REVIEWED_KEY) || '[]'); } catch (e) { return []; }
}
function markReviewed(id) {
  const ids = reviewedJobs().filter((x) => x !== id).concat(id).slice(-200);
  localStorage.setItem(REVIEWED_KEY, JSON.stringify(ids));
}

function hideReview() { $('review').hidden = true; $('review').innerHTML = ''; }

function showReview(j) {
  const box = $('review');
  state.reviewJob = j;
  box.hidden = false;
  if (reviewedJobs().includes(j.id)) {
    setHtml(box, html`<p class="hint review-done">Thanks for reviewing this model.</p>`);
    return;
  }
  const failed = j.state !== 'done';
  const stars = [5, 4, 3, 2, 1].map((n) => html`<input type="radio" id="star${n}" name="rating" value="${n}"><label
    for="star${n}" title="${n} star${n > 1 ? 's' : ''}" aria-label="${n} star${n > 1 ? 's' : ''}">★</label>`);
  setHtml(box, html`
    <h2>${failed ? 'Report this problem' : 'How did it turn out?'}</h2>
    <form id="reviewForm" novalidate>
      ${failed ? '' : html`<fieldset class="stars"><legend>Your score</legend><div class="star-row">${stars}</div></fieldset>`}
      <fieldset class="problems"><legend>${failed ? 'What went wrong?' : 'Anything wrong? (optional)'}</legend>
        ${PROBLEMS.map(([k, l]) => html`<label class="check"><input type="checkbox" name="problem" value="${k}" ${failed && k === 'slow_failed' ? html`checked` : ''}> ${l}</label>`)}
      </fieldset>
      <label>Details (optional)<textarea id="reviewComment" maxlength="4000" rows="3"
        placeholder="What did you expect, what went wrong, and where?"></textarea></label>
      <label class="check"><input type="checkbox" id="reviewFollowup"> I'd like a follow-up about this</label>
      <div id="reviewContact" hidden>
        <div class="row">
          <label>Name (optional)<input type="text" id="reviewName" maxlength="200" autocomplete="name"></label>
          <label>Email<input type="email" id="reviewEmail" maxlength="254" autocomplete="email"></label>
        </div>
      </div>
      <input type="text" id="reviewWebsite" class="hp" tabindex="-1" autocomplete="off" aria-hidden="true">
      <p class="hint">Your settings and this review are saved to help improve MapToMesh. Contact details are only
        stored if you ask for a follow-up, and only used for that.</p>
      <button type="submit" class="secondary" id="reviewSend">Send</button>
      <div id="reviewMsg" class="hint" role="status"></div>
    </form>`);
  $('reviewFollowup').onchange = () => {
    $('reviewContact').hidden = !$('reviewFollowup').checked;
    if ($('reviewFollowup').checked) $('reviewEmail').focus();
  };
  $('reviewForm').onsubmit = submitReview;
}

async function submitReview(e) {
  e.preventDefault();
  const form = $('reviewForm'), msg = $('reviewMsg');
  const rating = form.querySelector('input[name=rating]:checked');
  const problems = [...form.querySelectorAll('input[name=problem]:checked')].map((i) => i.value);
  const comment = $('reviewComment').value.trim();
  const followup = $('reviewFollowup').checked;
  const email = $('reviewEmail').value.trim();
  if (!rating && !problems.length && !comment) { msg.textContent = 'Pick a score or a problem, or write a comment.'; return; }
  if (followup && (!email || !$('reviewEmail').checkValidity())) {
    msg.textContent = 'Enter your email address for the follow-up (or untick it).';
    $('reviewEmail').focus();
    return;
  }
  const j = state.reviewJob;
  const body = {
    rating: rating ? Number(rating.value) : null, problems, comment, followup,
    name: followup ? $('reviewName').value.trim() : '', email: followup ? email : '',
    website: $('reviewWebsite').value,
    // used only if the server no longer has this job in memory (e.g. after a restart)
    snapshot: { state: j.state, error: j.error, options: j.options, result: j.result, log: j.log },
  };
  $('reviewSend').disabled = true;
  msg.textContent = 'Sending…';
  try {
    const r = await fetch(`/api/jobs/${j.id}/review`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(describeError(d));
    markReviewed(j.id);
    setHtml($('review'), html`<p class="hint review-done">Thanks! Your ${problems.length ? 'report' : 'review'} was saved${
      followup ? " and we'll get back to you by email" : ''}.</p>`);
  } catch (err) {
    $('reviewSend').disabled = false;
    msg.textContent = `Couldn't send: ${err.message}`;
  }
}

// ------------------------------------------------------------------ 3D viewer
const viewer = {
  renderer: null, scene: null, camera: null, controls: null, group: null, pending: null, frame: 0,
  init() {
    if (this.renderer) return;
    const el = $('viewer');
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.setBackground();
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-1, 2, 1.2);
    this.scene.add(sun);
    new ResizeObserver(() => this.resize()).observe(el);
    this.controls.addEventListener('change', () => this.redraw());
    this.redraw();
  },
  // draw only when something changed (the camera moved, a model loaded, a resize), not every
  // frame: an idle view costs no GPU time or battery
  redraw() {
    if (this.frame || !this.renderer) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const easing = this.controls.update();   // damping: keeps moving a little after release
      this.renderer.render(this.scene, this.camera);
      if (easing) this.redraw();
    });
  },
  setBackground() {   // follows the colour theme
    if (this.scene) this.scene.background = new THREE.Color(getComputedStyle($('viewer')).backgroundColor);
    this.redraw();
  },
  resize() {
    const el = $('viewer');
    if (!this.renderer || !el.clientWidth) return;
    this.renderer.setSize(el.clientWidth, el.clientHeight);
    this.camera.aspect = el.clientWidth / el.clientHeight;
    this.camera.updateProjectionMatrix();
    this.redraw();
  },
  async load(url) {
    if ($('viewer').hidden) { this.pending = url; return; }
    this.init(); this.resize();
    $('viewerMsg').hidden = false; $('viewerMsg').textContent = 'Loading preview…';
    $('viewerNote').hidden = true;
    const r = await fetch(url);
    if (!r.ok) { $('viewerMsg').textContent = 'Preview not available.'; return; }
    const buf = await r.arrayBuffer();   // gzip-encoded on the wire: the browser inflates it
    const hlen = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const base = 4 + hlen;
    if (this.group) { this.scene.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); }); }
    this.group = new THREE.Group();
    this.group.rotation.x = -Math.PI / 2; // 3MF is Z-up
    for (const p of header.parts) {
      const g = new THREE.BufferGeometry();
      const [pos, idx] = header.version === 2 ? decodePart(buf, base, p, header)
        : [new Float32Array(buf, base + p.vOffset, p.vCount * 3), new Uint32Array(buf, base + p.tOffset, p.tCount * 3)];
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      const mat = new THREE.MeshStandardMaterial({ color: p.color, roughness: 0.85, metalness: 0.0 });
      // smooth terrain, crisp walls / building edges
      const mesh = new THREE.Mesh(toCreasedNormals(g, Math.PI / 5), mat);
      g.dispose();
      mesh.name = p.name;
      this.group.add(mesh);
    }
    this.scene.add(this.group);
    const box = new THREE.Box3().setFromObject(this.group);
    const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    // optional camera from the URL: zoom, az/el (degrees, az 0 = from the south), fx/fy focus (0..1)
    const q = new URLSearchParams(location.search);
    const zoom = parseFloat(q.get('zoom') || '1');
    // a portrait view (phones) is narrower than it is tall: back off so the model's width fits
    const d = Math.max(size.x, size.z) * 1.1 / zoom / Math.min(1, this.camera.aspect);
    if (q.has('fx')) c.x = box.min.x + parseFloat(q.get('fx')) * size.x;
    if (q.has('fy')) c.z = box.max.z - parseFloat(q.get('fy')) * size.z;
    this.controls.target.copy(c);
    if (q.has('az')) {
      const az = parseFloat(q.get('az')) * Math.PI / 180, el = parseFloat(q.get('el') || '35') * Math.PI / 180;
      this.camera.position.set(c.x + d * Math.sin(az) * Math.cos(el), c.y + d * Math.sin(el), c.z + d * Math.cos(az) * Math.cos(el));
    } else {
      this.camera.position.set(c.x - d * 0.1, c.y + d * 0.75, c.z + d * 0.85);
    }
    this.camera.near = d / 100; this.camera.far = d * 20; this.camera.updateProjectionMatrix();
    this.controls.update();
    this.redraw();
    $('viewerMsg').hidden = true;
    if (header.simplified) {
      const m = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : `${Math.round(n / 1e3)} k`);
      $('viewerNote').textContent = `Preview simplified to ${m(header.triangles)} of ${m(header.triangles_full)} triangles; the 3MF has full detail`;
      $('viewerNote').hidden = false;
    }
  },
};

// preview format 2 (maptomesh/export.py preview_bytes): uint16 positions quantised over the model's
// bounding box, int32 index deltas
function decodePart(buf, base, p, header) {
  const q = new Uint16Array(buf, base + p.vOffset, p.vCount * 3);
  const pos = new Float32Array(q.length);
  const [lx, ly, lz] = header.lo, [sx, sy, sz] = header.scale;
  for (let i = 0; i < q.length; i += 3) {
    pos[i] = lx + q[i] * sx; pos[i + 1] = ly + q[i + 1] * sy; pos[i + 2] = lz + q[i + 2] * sz;
  }
  const d = new Int32Array(buf, base + p.tOffset, p.tCount * 3);
  const idx = new Uint32Array(d.length);
  let acc = 0;
  for (let i = 0; i < d.length; i++) { acc += d[i]; idx[i] = acc; }
  return [pos, idx];
}
// phones and low-memory devices get the lighter preview
const PREVIEW_LOD = COARSE || MOBILE.matches || (navigator.deviceMemory && navigator.deviceMemory <= 4) ? 'low' : 'high';

// ------------------------------------------------------------------ tabs
function selectTab(name) {
  setSheet(false);
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $('map').hidden = name !== 'map';
  $('viewer').hidden = name !== 'view';
  $('topview').hidden = name !== 'top';
  if (name === 'map') map.invalidateSize();
  if (name === 'view') {
    viewer.init(); viewer.resize();
    if (viewer.pending) { const u = viewer.pending; viewer.pending = null; viewer.load(u); }
  }
}
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));

// ------------------------------------------------------------------ phone layout: bottom sheet
// The panel's header stays visible at the bottom (--peek); the map / 3D view fill the space above
// it. Tap the header or "Settings" to expand the sheet, swipe it up or down, or tap a tab to close.
const sheetHead = $('sheetHead');
function sheetOpen() { return document.body.classList.contains('sheet-open'); }
function setSheet(open) {
  if (!MOBILE.matches) open = false;
  document.body.classList.toggle('sheet-open', open);
  $('sheetToggle').textContent = open ? 'Map ▾' : 'Settings ▴';
  $('sheetToggle').setAttribute('aria-expanded', String(open));
}
function layoutVars() {
  const root = document.documentElement.style;
  root.setProperty('--peek', `${Math.ceil(sheetHead.getBoundingClientRect().height)}px`);
  root.setProperty('--tabs-h', `${Math.ceil($('tabs').getBoundingClientRect().height)}px`);
}
new ResizeObserver(layoutVars).observe(sheetHead);
// the map's size changes with the phone sheet's header (progress row); keep the view centred.
// Not while it's hidden behind the 3D / Top tab: a 0 x 0 map has no centre to keep, and
// selectTab('map') re-measures it when it comes back.
new ResizeObserver(() => { if (!$('map').hidden && $('map').clientWidth) map.invalidateSize(); }).observe($('map'));
MOBILE.addEventListener('change', () => { setSheet(false); layoutVars(); });
$('sheetToggle').onclick = () => setSheet(!sheetOpen());
$('miniDetails').onclick = (e) => {
  e.preventDefault();
  setSheet(true);
  const target = $('result').hidden ? $('actions') : $('result');
  setTimeout(() => target.scrollIntoView({ block: 'start', behavior: 'smooth' }), 280);
};
$('mobileGenerate').onclick = () => {
  if (state.running) cancelJob(); else generate();
};
// swipe / tap on the header (not on its buttons and links)
let sheetDrag = null;
sheetHead.addEventListener('pointerdown', (e) => {
  if (!MOBILE.matches || e.button > 0 || e.target.closest('button, a, input, select')) return;
  sheetDrag = { pointer: e.pointerId, y: e.clientY, open: sheetOpen(), moved: 0 };
  document.body.classList.add('sheet-dragging');
});
window.addEventListener('pointermove', (e) => {
  if (!sheetDrag || e.pointerId !== sheetDrag.pointer) return;
  const dy = e.clientY - sheetDrag.y;
  sheetDrag.moved = Math.max(sheetDrag.moved, Math.abs(dy));
  const sb = $('sidebar');
  const closedY = sb.offsetHeight - sheetHead.offsetHeight;   // follow the finger
  const y = Math.min(closedY, Math.max(0, (sheetDrag.open ? 0 : closedY) + dy));
  sb.style.transform = `translateY(${y}px)`;
});
for (const t of ['pointerup', 'pointercancel']) {
  window.addEventListener(t, (e) => {
    if (!sheetDrag || e.pointerId !== sheetDrag.pointer) return;
    const dy = e.clientY - sheetDrag.y, d = sheetDrag;
    sheetDrag = null;
    document.body.classList.remove('sheet-dragging');
    $('sidebar').style.transform = '';
    if (t === 'pointercancel') return;
    if (d.moved < 8) setSheet(!d.open);           // tap
    else if (dy < -40) setSheet(true);
    else if (dy > 40) setSheet(false);
    else setSheet(d.open);
  });
}

// ------------------------------------------------------------------ colour theme
// Follows the system setting until the button is used, then the saved choice (light / dark).
const THEME_KEY = 'maptomesh.theme';
const DARK_MQ = matchMedia('(prefers-color-scheme: dark)');
function savedTheme() { try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; } }
function applyTheme() {
  const t = savedTheme();
  const dark = t === 'dark' || (t !== 'light' && DARK_MQ.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const btn = $('themeBtn');
  btn.textContent = dark ? '☀︎' : '☾';
  btn.title = dark ? 'Switch to the light theme' : 'Switch to the dark theme';
  btn.setAttribute('aria-label', btn.title);
  document.querySelector('meta[name=theme-color]').content = dark ? '#1d2024' : '#c8553d';
  viewer.setBackground();
}
$('themeBtn').onclick = () => {
  localStorage.setItem(THEME_KEY, document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  applyTheme();
};
DARK_MQ.addEventListener('change', applyTheme);
applyTheme();

// ------------------------------------------------------------------ about / donate
const ABOUT_KEY = 'maptomesh.about.seen.v1';
// ------------------------------------------------------------------ failure dialog
const FAIL_TITLES = {
  timeout: 'The model took too long',
  memory: 'The model needed too much memory',
  too_large: 'This model is too large for the server',
  busy: 'The server is busy',
};
let failSuggest = null;
let failServerBody = null;
function showFailure(j) {
  const dlg = $('failDialog');
  $('failTitle').textContent = j.title || FAIL_TITLES[j.error_kind] || "The model couldn't be built";
  $('failCause').textContent = j.error || '';
  const advice = j.advice || [];
  setHtml($('failAdvice'), html`${advice.map((a) => html`<li>${a}</li>`)}`);
  $('failAdviceHead').hidden = !advice.length;
  const stats = [];
  if (j.seconds) stats.push(`stopped after ${Math.round(j.seconds)} s`);
  if (j.peak_mb) stats.push(`peak memory ${(j.peak_mb / 1000).toFixed(1)} GB`);
  if (LIMITS && !j.device && (j.error_kind === 'timeout' || j.error_kind === 'memory'))
    stats.push(`limits: ${Math.round(LIMITS.timeout_s / 60)} min, ${(LIMITS.memory_mb / 1000).toFixed(1)} GB`);
  $('failStats').textContent = stats.length ? stats.join(' · ') : '';
  failSuggest = j.suggest && j.suggest.grid_cells ? j.suggest : null;
  const retry = $('failRetry');
  retry.hidden = !failSuggest && j.error_kind !== 'busy';
  retry.textContent = failSuggest ? `Retry at ${failSuggest.grid_cells} cells` : 'Retry';
  // a build that failed on this device can go to the server (if the server accepts that detail)
  const scap = j.server ? gridCap('server') : null;
  failServerBody = j.server && (scap == null || j.server.grid_cells <= scap) ? j.server : null;
  $('failServer').hidden = !failServerBody;
  if (!dlg.open) dlg.showModal();
}
$('failOk').onclick = () => $('failDialog').close();
$('failClose').onclick = () => $('failDialog').close();
$('failDialog').addEventListener('click', (e) => { if (e.target === $('failDialog')) $('failDialog').close(); });
$('failRetry').onclick = () => {
  $('failDialog').close();
  if (failSuggest) {
    $('gridCells').value = failSuggest.grid_cells;
    $('gridCells').dispatchEvent(new Event('input', { bubbles: true }));
    saveSettings();
  }
  generate({ keepUrl: true });
};
$('failServer').onclick = () => {   // this model only: the next one is built on the device again
  $('failDialog').close();
  if (failServerBody) generate({ keepUrl: true, target: 'server' });
};

const aboutDialog = $('aboutDialog');
function openAbout() { if (!aboutDialog.open) aboutDialog.showModal(); }
aboutDialog.addEventListener('close', () => localStorage.setItem(ABOUT_KEY, '1'));
aboutDialog.addEventListener('click', (e) => { if (e.target === aboutDialog) aboutDialog.close(); });  // backdrop
$('aboutClose').onclick = () => aboutDialog.close();
$('aboutStart').onclick = () => aboutDialog.close();
$('aboutLink').onclick = (e) => { e.preventDefault(); openAbout(); };
aboutDialog.querySelectorAll('.copy').forEach((btn) => {
  btn.onclick = async () => {
    const text = btn.dataset.copy;
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {  // no clipboard API (non-secure context): select the address instead
      const sel = window.getSelection(); const r = document.createRange();
      r.selectNodeContents(btn.previousElementSibling); sel.removeAllRanges(); sel.addRange(r);
      document.execCommand('copy');
    }
    btn.textContent = 'Copied'; btn.classList.add('done');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1600);
  };
});

// ------------------------------------------------------------------ wiring
buildColorInputs();
initBuildTarget();
loadSettings();
fillPresets();
enforceAspect('w');
redrawSelection();
updateDerived();
refreshMode();
pollIndex();
scheduleUsCoverage();

// reopen a finished job from the URL (?job=<id>); &embed=1 shows only the 3D view (link previews,
// screenshots: the preview image in the page's <head> metadata is rendered this way)
const jobParam = new URLSearchParams(location.search).get('job');
const shareParam = new URLSearchParams(location.search).get('s');
if (jobParam && new URLSearchParams(location.search).get('embed') === '1') document.body.classList.add('embed');
// first visit: introduce the tool (not over a shared link - the About link is always there)
if (!jobParam && !shareParam && !localStorage.getItem(ABOUT_KEY)) openAbout();
if (shareParam && !jobParam) openShare(shareParam);
if (jobParam) {
  fetch(`/api/jobs/${jobParam}`).then((r) => (r.ok ? r.json() : null)).then((j) => {
    if (j && j.state === 'done') { state.jobId = j.id; applyOptions(j.options); showResult(j); showReview(j); }
  });
}

$('drawBtn').onclick = () => setDrawing(!state.drawing);
$('centerBtn').onclick = () => { const c = map.getCenter(); state.center = { lat: c.lat, lon: c.lng }; redrawSelection(); saveSettings(); };
$('shape').onchange = () => { enforceAspect('w'); redrawSelection(); saveSettings(); };
$('widthKm').oninput = () => { enforceAspect('w'); redrawSelection(); };
$('heightKm').oninput = () => { enforceAspect('h'); redrawSelection(); };
$('colorMode').onchange = () => { fillPresets(); updateDerived(); saveSettings(); scheduleCoverage(); };
$('preset').onchange = applyPreset;
$('bandCount').oninput = () => { renderBandInputs(); saveSettings(); };
$('usSource').addEventListener('change', () => { scheduleUsCoverage(); scheduleCoverage(); });
$('source').addEventListener('change', scheduleCoverage);
$('demPref').addEventListener('change', scheduleCoverage);
for (const id of ['gridCells', 'colorMode', 'surface', 'source', 'widthKm', 'heightKm'])
  $(id).addEventListener('input', updateGridCap);
$('surface').addEventListener('change', scheduleCoverage);
$('showCoverage').onchange = () => {
  scheduleUsCoverage();
  if (!coverageLayer) return;
  if ($('showCoverage').checked) { coverageLayer.addTo(map); selection.bringToFront(); } else map.removeLayer(coverageLayer);
};
$('generateBtn').onclick = () => generate();
if (!shareParam) (window.requestIdleCallback || ((f) => setTimeout(f, 2000)))(() => prewarmDevice(), { timeout: 5000 });
$('cancelBtn').onclick = cancelJob;
for (const id of ['baseSeparate', 'label', 'gridCells', 'smoothing', 'cliffSmoothing', 'sizeMm', 'zEx', 'tilesX', 'tilesY', 'frame', 'cornerMm', 'elevWater']) {
  $(id).addEventListener('input', () => { if (id === 'sizeMm' || id === 'cornerMm') redrawSelection(); else updateDerived(); });
}
$('gridCells').addEventListener('change', scheduleCoverage);
document.getElementById('sidebar').addEventListener('change', saveSettings);
for (const t of ['input', 'change']) document.getElementById('sidebar').addEventListener(t, refreshMode);
map.on('moveend', saveSettings);
map.on('moveend', scheduleUsCoverage);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.drawing) setDrawing(false); });

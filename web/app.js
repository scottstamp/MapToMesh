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
  l.innerHTML = `<input type="color" data-color="${key}" value="${value}"> ${label}`;
  container.appendChild(l);
  return l.querySelector('input');
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
  sel.innerHTML = '<option value="">— choose —</option>';
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
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors · LiDAR: <a href="https://open.canada.ca/data/en/dataset/7069387e-9986-4297-9f55-0288e9676947">NRCan CanElevation</a> (OGL-Canada)',
}).addTo(map);

const selection = L.polygon([], { color: '#c8553d', weight: 2, fillOpacity: 0.12 }).addTo(map);
const centerMarker = L.circleMarker([0, 0], { radius: 4, color: '#c8553d', fillOpacity: 1 }).addTo(map);
let coverageLayer = null;
let hintEl = null;

// ---- resize handles: corners and edge midpoints of the area's bounding box
const HANDLE_DIRS = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]];
const HANDLE_CURSOR = { '-1,-1': 'nesw', '0,-1': 'ns', '1,-1': 'nwse', '1,0': 'ew', '1,1': 'nesw', '0,1': 'ns', '-1,1': 'nwse', '-1,0': 'ew' };
let resizing = null;
const handles = HANDLE_DIRS.map(([sx, sy]) => {
  const m = L.marker([0, 0], {
    draggable: true, keyboard: false, zIndexOffset: 1000,
    icon: L.divIcon({ className: `resize-handle cursor-${HANDLE_CURSOR[`${sx},${sy}`]}`, iconSize: [12, 12] }),
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
  if (on) map.dragging.disable(); else map.dragging.enable();
}

let drawStart = null, dragStart = null;
map.on('mousedown', (e) => {
  if (state.drawing) { drawStart = e.latlng; }
});
map.on('mousemove', (e) => {
  if (drawStart) {
    const shape = $('shape').value;
    if (shape === 'circle' || shape === 'hexagon') {
      const { dx, dy } = metresBetween(drawStart, e.latlng);
      const r = Math.max(25, Math.hypot(dx, dy));
      state.center = { lat: drawStart.lat, lon: drawStart.lng };
      $('widthKm').value = +(2 * r / 1000).toFixed(3);
      enforceAspect('w');
    } else {
      const { dx, dy } = metresBetween(drawStart, e.latlng);
      state.center = { lat: (drawStart.lat + e.latlng.lat) / 2, lon: (drawStart.lng + e.latlng.lng) / 2 };
      $('widthKm').value = +(Math.max(50, Math.abs(dx)) / 1000).toFixed(3);
      $('heightKm').value = +(Math.max(50, Math.abs(dy)) / 1000).toFixed(3);
    }
    redrawSelection();
  } else if (dragStart) {
    state.center = { lat: dragStart.center.lat + (e.latlng.lat - dragStart.ll.lat), lon: dragStart.center.lon + (e.latlng.lng - dragStart.ll.lng) };
    redrawSelection();
  }
});
map.on('mouseup', () => {
  if (drawStart) { drawStart = null; setDrawing(false); saveSettings(); }
  if (dragStart) { dragStart = null; map.dragging.enable(); saveSettings(); }
});
selection.on('mousedown', (e) => {
  if (state.drawing) return;
  L.DomEvent.stop(e);
  dragStart = { ll: e.latlng, center: { ...state.center } };
  map.dragging.disable();
});
selection.bindTooltip('Drag to move · drag the white handles to resize', { sticky: true });

function loadCoverage() {
  fetch('/api/projects').then((r) => (r.ok ? r.json() : null)).then((gj) => {
    if (!gj) return;
    coverageLayer = L.geoJSON(gj, {
      style: { color: '#2b7a78', weight: 1, fillColor: '#3aafa9', fillOpacity: 0.12 },
      interactive: true,
      onEachFeature: (f, layer) => layer.bindTooltip(`${f.properties.project}${f.properties.year ? ' (' + f.properties.year + ')' : ''}`, { sticky: true }),
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
  $('cornerWrap').style.visibility = $('shape').value === 'rounded' ? 'visible' : 'hidden';
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
      body: JSON.stringify({ lat: state.center.lat, lon: state.center.lon, width_m: w, height_m: h, grid_cells: parseInt($('gridCells').value) }),
    });
    const d = await r.json();
    box.className = 'info';
    if (!d.ready) { box.textContent = `LiDAR index loading… ${d.message || ''}`; return; }
    if (!d.tiles) {
      box.className = 'info warn';
      box.textContent = 'No LiDAR point-cloud tiles here. Try the HRDEM source, or move the area onto the shaded coverage.';
      return;
    }
    const projects = d.projects.map((p) => `${p.name}${p.year ? ' (' + p.year + ')' : ''}: ${p.tiles}`).join('\n');
    box.className = d.tiles > d.max_tiles ? 'info err' : d.tiles > 300 ? 'info warn' : 'info';
    box.innerText = `${d.tiles} LiDAR tile${d.tiles > 1 ? 's' : ''}` +
      (d.tiles > d.max_tiles ? ` — over the ${d.max_tiles} limit, use HRDEM` : d.tiles > 300 ? ' — large request, HRDEM will be faster' : '') +
      `\n${projects}`;
  } catch (e) { box.textContent = ''; }
}

async function pollIndex() {
  try {
    const d = await (await fetch('/api/status')).json();
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
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&countrycodes=ca&limit=6&q=${encodeURIComponent(q)}`;
    try {
      const res = await (await fetch(url, { headers: { 'Accept-Language': navigator.language || 'en' } })).json();
      $('searchResults').innerHTML = '';
      for (const r of res) {
        const li = document.createElement('li');
        li.textContent = r.display_name;
        li.onclick = () => {
          state.center = { lat: parseFloat(r.lat), lon: parseFloat(r.lon) };
          $('searchResults').innerHTML = ''; $('search').value = r.display_name.split(',')[0];
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
  'waterRecess', 'osmStreams', 'riverMin', 'osmRoads', 'roadMin', 'roadEmboss', 'colorMode', 'bandCount', 'bandMode', 'bandThresholds', 'minRegion',
  'colorDepth', 'swapBudget', 'elevWater',
  'name', 'layout', 'showCoverage'];

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
  for (const f of FIELDS) {
    if (!(f in s)) continue;
    const el = $(f);
    if (el.type === 'checkbox') el.checked = s[f]; else el.value = s[f];
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
    source: $('source').value, surface: $('surface').value,
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
  const fields = { shape: 'shape', corner_radius_mm: 'cornerMm', source: 'source', surface: 'surface', grid_cells: 'gridCells',
    smoothing: 'smoothing', cliff_smoothing: 'cliffSmoothing', base_separate: 'baseSeparate', base_lip: 'baseLip',
    lip_height_mm: 'lipH', lip_width_mm: 'lipW', lip_clearance_mm: 'lipC', label: 'label', label_text: 'labelText',
    label_scale_bar: 'labelScale', label_north: 'labelNorth', layer_height_mm: 'layerH', first_layer_mm: 'firstLayer',
    min_feature_mm: 'minFeature', simplify_mm: 'simplify', size_mm: 'sizeMm', base_mm: 'baseMm', floor_mm: 'floorMm', z_exaggeration: 'zEx',
    max_height_mm: 'maxHeight', frame: 'frame', frame_width_mm: 'frameW', frame_height_mm: 'frameH', tiles_x: 'tilesX',
    tiles_y: 'tilesY', flatten_water: 'flattenWater', water_recess_mm: 'waterRecess', voids_as_water: 'voidsWater',
    osm_water: 'osmWater', osm_streams: 'osmStreams', river_min_width_mm: 'riverMin', osm_roads: 'osmRoads', road_min_width_mm: 'roadMin', road_emboss_mm: 'roadEmboss',
    color_mode: 'colorMode', min_region_mm2: 'minRegion', color_depth_mm: 'colorDepth', swap_patch_budget_mm2: 'swapBudget', elevation_water: 'elevWater', layout: 'layout', name: 'name' };
  for (const [k, id] of Object.entries(fields)) if (k in o) set(id, o[k]);
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
}

async function generate() {
  saveSettings();
  const body = buildOptions();
  $('generateBtn').disabled = true;
  $('cancelBtn').hidden = false;
  $('progress').hidden = false; $('log').hidden = false;
  $('bar').style.width = '0%'; $('log').textContent = ''; $('progressMsg').textContent = 'Starting…';
  $('result').hidden = true;
  try {
    const r = await fetch('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail));
    state.jobId = d.id;
    poll();
  } catch (e) {
    finish();
    $('progressMsg').textContent = `Error: ${e.message}`;
  }
}

function finish() {
  $('generateBtn').disabled = false;
  $('cancelBtn').hidden = true;
  clearTimeout(state.pollTimer);
}

async function poll() {
  const id = state.jobId;
  try {
    const j = await (await fetch(`/api/jobs/${id}`)).json();
    $('bar').style.width = `${(j.progress * 100).toFixed(1)}%`;
    $('progressMsg').textContent = j.message;
    $('log').textContent = j.log.join('\n');
    $('log').scrollTop = 1e9;
    if (j.state === 'done') {
      finish(); showResult(j);
      history.replaceState(null, '', `?job=${id}`);
      return;
    }
    if (j.state === 'error' || j.state === 'cancelled') {
      finish(); $('progressMsg').textContent = j.state === 'error' ? `Error: ${j.error}` : 'Cancelled'; return;
    }
  } catch (e) { /* transient */ }
  state.pollTimer = setTimeout(poll, 700);
}

function fmtBytes(n) { return n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(0)} kB`; }

function showResult(j) {
  const r = j.result, s = r.stats, id = j.id;
  state.lastResult = j;
  const fil = r.filaments.map((f) => `<li><span class="swatch" style="background:${f.color}"></span><b>Filament ${f.slot}</b> ${f.color} — ${f.used_by.join(', ')}</li>`).join('');
  const cc = r.color_changes.length
    ? `<h2>Single-extruder colour changes</h2><ul class="filaments">${r.color_changes.map((c) => `<li><span class="swatch" style="background:${c.color}"></span>at Z = ${c.z_mm.toFixed(2)} mm${c.elevation_m != null ? ` (${c.elevation_m} m)` : ''}</li>`).join('')}</ul><p class="hint">Add a filament change / pause at these heights (round to your layer height).</p>`
    : '';
  const notes = [...(r.notes || []), ...(r.data.failed_tiles ? [`${r.data.failed_tiles} tile(s) failed to download.`] : [])];
  const projects = (r.data.projects || []).map((p) => p.name + (p.year ? ` (${p.year})` : '')).join(', ');
  $('result').innerHTML = `
    <h2>Result</h2>
    <img src="/api/jobs/${id}/thumbnail.png?${Date.now()}" alt="Top view">
    <div class="downloads">
      <a href="/api/jobs/${id}/model.3mf" download>⬇ 3MF (${fmtBytes(r.file_size)})</a>
      <a class="alt" href="/api/jobs/${id}/stl.zip" download>STL per colour (.zip)</a>
    </div>
    ${r.base_file ? `<div class="downloads"><a class="alt" href="/api/jobs/${id}/base.3mf" download>⬇ Base plate 3MF (${fmtBytes(r.base_file_size)}, ${s.base_plate_mm.join(' × ')} mm)</a></div>` : ''}
    <dl class="stats">
      <dt>Model size</dt><dd>${s.size_mm[0]} × ${s.size_mm[1]} × ${s.size_mm[2]} mm</dd>
      ${s.tile_size_mm ? `<dt>Tiles</dt><dd>${s.tiles} × ${s.tile_size_mm[0]} × ${s.tile_size_mm[1]} mm</dd>` : ''}
      <dt>Scale</dt><dd>${s.scale} (z ×${$('zEx').value})</dd>
      <dt>Elevation</dt><dd>${s.elevation_range_m[0]} – ${s.elevation_range_m[1]} m → relief ${s.relief_mm} mm</dd>
      <dt>Resolution</dt><dd>${s.ground_m_per_cell} m / ${s.cell_mm} mm per cell</dd>
      <dt>Mesh</dt><dd>${s.triangles.toLocaleString()} triangles, ${s.parts} parts</dd>
      ${s.colour_changes_est != null ? `<dt>Colour changes</dt><dd>≈ ${s.colour_changes_est.toLocaleString()} at 0.2 mm layers (max ${s.max_colours_per_layer} colours in a layer)</dd>` : ''}
      ${r.data.points ? `<dt>LiDAR</dt><dd>${(r.data.points / 1e6).toFixed(1)} M points from ${r.data.tiles} tiles</dd>` : ''}
      ${projects ? `<dt>Projects</dt><dd>${projects}</dd>` : ''}
      ${r.data.items ? `<dt>HRDEM</dt><dd>${r.data.items.join(', ')}</dd>` : ''}
    </dl>
    <h2>Filaments</h2><ul class="filaments">${fil}</ul>
    ${cc}
    ${(r.tips || []).length ? `<h2>Print tips</h2><ul class="tips">${r.tips.map((t) => `<li>${t}</li>`).join('')}</ul>` : ''}
    ${notes.length ? `<div class="info warn">${notes.join('<br>')}</div>` : ''}`;
  $('result').hidden = false;
  $('topImg').src = `/api/jobs/${id}/thumbnail.png?${Date.now()}`;
  viewer.load(`/api/jobs/${id}/preview.bin`);
  selectTab('view');
}

// ------------------------------------------------------------------ 3D viewer
const viewer = {
  renderer: null, scene: null, camera: null, controls: null, group: null, pending: null,
  init() {
    if (this.renderer) return;
    const el = $('viewer');
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#e9e7e2');
    this.camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-1, 2, 1.2);
    this.scene.add(sun);
    new ResizeObserver(() => this.resize()).observe(el);
    const loop = () => { requestAnimationFrame(loop); this.controls.update(); this.renderer.render(this.scene, this.camera); };
    loop();
  },
  resize() {
    const el = $('viewer');
    if (!this.renderer || !el.clientWidth) return;
    this.renderer.setSize(el.clientWidth, el.clientHeight);
    this.camera.aspect = el.clientWidth / el.clientHeight;
    this.camera.updateProjectionMatrix();
  },
  async load(url) {
    if ($('viewer').hidden) { this.pending = url; return; }
    this.init(); this.resize();
    $('viewerMsg').hidden = false; $('viewerMsg').textContent = 'Loading preview…';
    const buf = await (await fetch(url)).arrayBuffer();
    const hlen = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const base = 4 + hlen;
    if (this.group) { this.scene.remove(this.group); this.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); }); }
    this.group = new THREE.Group();
    this.group.rotation.x = -Math.PI / 2; // 3MF is Z-up
    for (const p of header.parts) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, base + p.vOffset, p.vCount * 3), 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, base + p.tOffset, p.tCount * 3), 1));
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
    const d = Math.max(size.x, size.z) * 1.1 / zoom;
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
    $('viewerMsg').hidden = true;
  },
};

// ------------------------------------------------------------------ tabs
function selectTab(name) {
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

// ------------------------------------------------------------------ wiring
buildColorInputs();
loadSettings();
fillPresets();
enforceAspect('w');
redrawSelection();
updateDerived();
pollIndex();

// reopen a finished job from the URL (?job=<id>)
const jobParam = new URLSearchParams(location.search).get('job');
if (jobParam) {
  fetch(`/api/jobs/${jobParam}`).then((r) => (r.ok ? r.json() : null)).then((j) => {
    if (j && j.state === 'done') { state.jobId = j.id; applyOptions(j.options); showResult(j); }
  });
}

$('drawBtn').onclick = () => setDrawing(!state.drawing);
$('centerBtn').onclick = () => { const c = map.getCenter(); state.center = { lat: c.lat, lon: c.lng }; redrawSelection(); saveSettings(); };
$('shape').onchange = () => { enforceAspect('w'); redrawSelection(); saveSettings(); };
$('widthKm').oninput = () => { enforceAspect('w'); redrawSelection(); };
$('heightKm').oninput = () => { enforceAspect('h'); redrawSelection(); };
$('colorMode').onchange = () => { fillPresets(); updateDerived(); saveSettings(); };
$('preset').onchange = applyPreset;
$('bandCount').oninput = () => { renderBandInputs(); saveSettings(); };
$('showCoverage').onchange = () => {
  if (!coverageLayer) return;
  if ($('showCoverage').checked) { coverageLayer.addTo(map); selection.bringToFront(); } else map.removeLayer(coverageLayer);
};
$('generateBtn').onclick = generate;
$('cancelBtn').onclick = () => state.jobId && fetch(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
for (const id of ['baseSeparate', 'label', 'gridCells', 'smoothing', 'cliffSmoothing', 'sizeMm', 'zEx', 'tilesX', 'tilesY', 'frame', 'cornerMm', 'elevWater']) {
  $(id).addEventListener('input', () => { if (id === 'sizeMm' || id === 'cornerMm') redrawSelection(); else updateDerived(); });
}
$('gridCells').addEventListener('change', scheduleCoverage);
document.getElementById('sidebar').addEventListener('change', saveSettings);
map.on('moveend', saveSettings);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.drawing) setDrawing(false); });

// Persistent data cache for models built on this device (web/browser-worker.js): downloaded
// elevation / point-cloud blocks and the server code's cache files (/cache in Python) survive
// page reloads, so building the same area again downloads little or nothing.
//
// Storage: the Origin Private File System (OPFS), through synchronous access handles (dedicated
// workers only). Synchronous, because Python reads it synchronously, with or without JavaScript
// Promise Integration. Four "segment" files hold append-only records:
//
//   segment: [magic u32 "MTMS"][version u32][generation f64] record*
//   record:  [magic u32 "MTMR"][kind u8, 3 pad][time f64][key length u32][data length u32][key][data]
//            kind 1 = put (key -> data), 2 = delete (key), 3 = move (key -> the key in data)
//
// New records go to the newest segment; when it is full, the oldest segment is emptied and reused
// (its entries are dropped), so the cache never exceeds its budget. Entries read from the oldest
// segment are written again, so data in regular use survives. An in-memory index (key -> place)
// is rebuilt by scanning the segments; a torn record at a segment's end (tab closed mid-write) is
// cut off. Only one tab can hold the files at a time: another tab builds without the cache.

const DIR = "maptomesh-cache-v1";
const SEGMENTS = 4;
const SEG_MAGIC = 0x534d544d, REC_MAGIC = 0x524d544d, VERSION = 1;
const SEG_HEAD = 16, REC_HEAD = 24;
const PUT = 1, DEL = 2, MOVE = 3;
const enc = new TextEncoder(), dec = new TextDecoder();

export class PackCache {
  constructor() {
    this.h = null;              // sync access handles while open
    this.index = new Map();     // key -> {seg, off (of the data), len, time}
    this.gen = []; this.size = [];
    this.cur = 0;
    this.segMax = 64 << 20;
    this.stamp = null;          // segment sizes / generations when we last closed
    this.onDrop = null;         // (keys) => void: entries removed to make room
    this.stats = { hitBytes: 0, putBytes: 0 };
  }

  get open() { return this.h !== null; }

  // Open the segment files (async: OPFS handles). Returns true if the index was (re)built, i.e.
  // the files changed since we last held them (another tab) or this is the first open.
  async begin(budgetBytes) {
    if (this.h) return false;
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    const handles = [];
    try {
      for (let i = 0; i < SEGMENTS; i++) {
        const fh = await dir.getFileHandle(`seg${i}`, { create: true });
        handles.push(await fh.createSyncAccessHandle());   // fails while another tab holds it
      }
    } catch (e) {
      for (const h of handles) h.close();
      throw e;
    }
    this.h = handles;
    this.segMax = Math.max(8 << 20, Math.floor(budgetBytes / SEGMENTS));
    const stamp = this._stamp();
    if (this.stamp !== null && stamp === this.stamp) return false;
    this._scan();
    return true;
  }

  end() {   // flush and release the files (another tab may use them meanwhile)
    if (!this.h) return;
    for (const h of this.h) { try { h.flush(); } catch (e) { /* closing anyway */ } }
    this.stamp = this._stamp();
    for (const h of this.h) h.close();
    this.h = null;
  }

  bytes() {
    let n = 0;
    for (let i = 0; i < SEGMENTS; i++) n += Math.max(0, (this.size[i] || 0) - SEG_HEAD);
    return n;
  }

  _stamp() {
    const head = new Uint8Array(SEG_HEAD);
    return this.h.map((h) => {
      const n = h.getSize();
      if (n >= SEG_HEAD) h.read(head, { at: 0 });
      return `${n}:${n >= SEG_HEAD ? new DataView(head.buffer).getFloat64(8, true) : -1}`;
    }).join(",");
  }

  _initSeg(s, gen) {
    const head = new DataView(new ArrayBuffer(SEG_HEAD));
    head.setUint32(0, SEG_MAGIC, true);
    head.setUint32(4, VERSION, true);
    head.setFloat64(8, gen, true);
    this.h[s].truncate(0);
    this.h[s].write(new Uint8Array(head.buffer), { at: 0 });
    this.gen[s] = gen;
    this.size[s] = SEG_HEAD;
  }

  _scan() {
    this.index.clear();
    const head = new Uint8Array(SEG_HEAD), hv = new DataView(head.buffer);
    for (let s = 0; s < SEGMENTS; s++) {
      const n = this.h[s].getSize();
      if (n >= SEG_HEAD && this.h[s].read(head, { at: 0 }) === SEG_HEAD
          && hv.getUint32(0, true) === SEG_MAGIC && hv.getUint32(4, true) === VERSION) {
        this.gen[s] = hv.getFloat64(8, true);
        this.size[s] = n;
      } else {
        this._initSeg(s, 0);
      }
    }
    const buf = new Uint8Array(REC_HEAD + 1024), bv = new DataView(buf.buffer);
    for (const s of this._byAge()) {
      let off = SEG_HEAD;
      const end = this.size[s];
      while (off + REC_HEAD <= end) {
        const got = this.h[s].read(buf, { at: off });
        if (got < REC_HEAD || bv.getUint32(0, true) !== REC_MAGIC) break;
        const kind = buf[4], time = bv.getFloat64(8, true);
        const klen = bv.getUint32(16, true), dlen = bv.getUint32(20, true);
        const next = off + REC_HEAD + klen + dlen;
        if (next > end || klen > 1 << 16) break;
        let kb;
        if (klen <= got - REC_HEAD) kb = buf.subarray(REC_HEAD, REC_HEAD + klen);
        else { kb = new Uint8Array(klen); this.h[s].read(kb, { at: off + REC_HEAD }); }
        const key = dec.decode(kb);
        if (kind === PUT) {
          this.index.set(key, { seg: s, off: off + REC_HEAD + klen, len: dlen, time });
        } else if (kind === DEL) {
          this.index.delete(key);
        } else if (kind === MOVE) {
          const d = new Uint8Array(dlen);
          this.h[s].read(d, { at: off + REC_HEAD + klen });
          const e = this.index.get(key);
          if (e) { this.index.delete(key); this.index.set(dec.decode(d), e); }
        }
        off = next;
      }
      if (off < end) {           // torn or corrupt tail: cut it off
        this.h[s].truncate(off);
        this.size[s] = off;
      }
    }
    const order = this._byAge();
    this.cur = order[order.length - 1];
  }

  _byAge() {
    return [...Array(SEGMENTS).keys()].sort((a, b) => this.gen[a] - this.gen[b]);
  }

  _rotate() {
    const order = this._byAge();
    const old = order[0];
    const dropped = [];
    for (const [k, e] of this.index) if (e.seg === old) { this.index.delete(k); dropped.push(k); }
    this._initSeg(old, this.gen[order[order.length - 1]] + 1);
    this.cur = old;
    if (dropped.length && this.onDrop) this.onDrop(dropped);
  }

  _append(kind, key, data, time) {
    const kb = enc.encode(key);
    const need = REC_HEAD + kb.length + data.length;
    if (need > this.segMax - SEG_HEAD) return -1;
    if (this.size[this.cur] + need > this.segMax) {
      try { this._rotate(); } catch (e) { return -1; }   // storage refused even the empty segment
    }
    const s = this.cur, off = this.size[s];
    const head = new Uint8Array(REC_HEAD + kb.length), hv = new DataView(head.buffer);
    hv.setUint32(0, REC_MAGIC, true);
    head[4] = kind;
    hv.setFloat64(8, time, true);
    hv.setUint32(16, kb.length, true);
    hv.setUint32(20, data.length, true);
    head.set(kb, REC_HEAD);
    try {
      if (this.h[s].write(head, { at: off }) !== head.length
          || (data.length && this.h[s].write(data, { at: off + head.length }) !== data.length)) {
        throw new Error("short write");
      }
    } catch (e) {               // quota exceeded / disk full: keep the segment consistent
      try { this.h[s].truncate(off); } catch (e2) { /* nothing more to do */ }
      return -1;
    }
    this.size[s] = off + need;
    return off + head.length;
  }

  has(key) { return this.h !== null && this.index.has(key); }

  time(key) { const e = this.index.get(key); return e ? e.time : null; }

  get(key) {
    if (!this.h) return null;
    const e = this.index.get(key);
    if (!e) return null;
    const out = new Uint8Array(e.len);
    if (e.len && this.h[e.seg].read(out, { at: e.off }) !== e.len) { this.index.delete(key); return null; }
    this.stats.hitBytes += e.len;
    if (e.seg === this._byAge()[0] && e.seg !== this.cur) this.put(key, out, e.time);   // keep it
    return out;
  }

  put(key, data, time = Date.now()) {
    if (!this.h) return false;
    const off = this._append(PUT, key, data, time);
    if (off < 0) return false;
    this.index.set(key, { seg: this.cur, off, len: data.length, time });
    this.stats.putBytes += data.length;
    return true;
  }

  del(key) {
    if (!this.h || !this.index.has(key)) return;
    if (this._append(DEL, key, new Uint8Array(0), Date.now()) >= 0) this.index.delete(key);
  }

  move(from, to) {
    if (!this.h || !this.index.has(from)) return false;
    if (this._append(MOVE, from, enc.encode(to), Date.now()) < 0) return false;
    const e = this.index.get(from);          // (gone if making room dropped it)
    if (!e) return false;
    this.index.delete(from);
    this.index.set(to, e);
    return true;
  }

  delPrefix(prefix) {
    for (const k of [...this.index.keys()]) if (k.startsWith(prefix)) this.del(k);
  }

  clear() {
    if (!this.h) return;
    const keys = [...this.index.keys()];
    this.index.clear();
    for (let s = 0; s < SEGMENTS; s++) this._initSeg(s, 0);
    this.cur = 0;
    if (keys.length && this.onDrop) this.onDrop(keys);
  }
}

// ------------------------------------------------------------------ file system
// Python's /cache as an Emscripten MEMFS whose files live in the pack cache: a file's contents
// are read from the cache when it is opened and dropped from memory when it is closed (written
// back first if changed). Directories exist in memory only (from the cached files' paths).
// Top-level folders in `memoryOnly` (job outputs) are plain in-memory files.
export function mountCache(FS, mountpoint, cache, memoryOnly = ["output"]) {
  const MEMFS = FS.filesystems.MEMFS;
  try { FS.unmount(mountpoint); } catch (e) { /* not mounted */ }
  try { FS.mkdirTree(mountpoint); } catch (e) { /* exists */ }
  if (!cache) {
    FS.mount(MEMFS, {}, mountpoint);
    return;
  }
  const M = MEMFS.node_ops, S = MEMFS.stream_ops;
  const keyOf = (node) => `f|${FS.getPath(node).slice(mountpoint.length + 1)}`;
  const childPersists = (dir, name) => (dir.isCacheRoot ? !memoryOnly.includes(name) : dir.persist);
  const isFile = (node) => FS.isFile(node.mode);

  const load = (node) => {
    node.busy = true;              // reading may make room (refresh), which mustn't drop this node
    const data = cache.get(keyOf(node));
    node.busy = false;
    if (data === null) throw new FS.ErrnoError(44);      // dropped meanwhile: ENOENT
    node.contents = data;
    node.usedBytes = data.length;
    node.lazy = false;
    node.stored = true;            // the cache holds these contents
  };
  const save = (node) => {      // write back, then free the memory (kept if the cache refused it)
    const data = node.contents.subarray(0, node.usedBytes);
    node.busy = true;
    const ok = cache.put(keyOf(node), data, node.mtime);
    node.busy = false;
    node.stored = ok;
    if (ok) {
      node.dirty = false;
      node.contents = new Uint8Array(0);
      node.lazy = true;
    }
  };

  const fileStream = {
    ...MEMFS.ops_table.file.stream,
    open(stream) {
      const node = stream.node;
      if (!node.persist) return;
      if (node.lazy) load(node);
      node.opens = (node.opens || 0) + 1;
      if (stream.flags & 3) node.dirty = true;           // opened for writing
    },
    close(stream) {
      const node = stream.node;
      if (!node.persist) return;
      node.opens -= 1;
      if (node.opens > 0) return;
      if (node.deleted) { node.contents = new Uint8Array(0); return; }
      if (node.dirty) save(node);
      else if (node.stored) { node.contents = new Uint8Array(0); node.lazy = true; }   // else: memory only
    },
    write(stream, buffer, offset, length, position, canOwn) {
      stream.node.dirty = true;
      return S.write(stream, buffer, offset, length, position, canOwn);
    },
  };
  const fileNode = {
    ...MEMFS.ops_table.file.node,
    setattr(node, attr) {
      if (node.persist && attr.size !== undefined) {
        if (node.lazy && attr.size > 0) load(node);
        if (node.lazy) { node.lazy = false; node.contents = new Uint8Array(0); node.usedBytes = 0; }
        node.dirty = true;
        if (!node.opens) {            // truncated without being open: store it now
          M.setattr(node, attr);
          save(node);
          return;
        }
      }
      M.setattr(node, attr);
    },
  };
  const dirNode = {
    ...MEMFS.ops_table.dir.node,
    mknod(parent, name, mode, dev) {
      const node = M.mknod(parent, name, mode, dev);
      adopt(node, childPersists(parent, name));
      return node;
    },
    rename(node, newDir, newName) {
      const before = node.persist ? keyOf(node) : null;
      M.rename(node, newDir, newName);
      node.persist = childPersists(newDir, newName);
      const after = node.persist ? keyOf(node) : null;
      if (FS.isDir(node.mode)) {
        if (before && after) {
          for (const k of [...cache.index.keys()]) if (k.startsWith(`${before}/`)) cache.move(k, after + k.slice(before.length));
        }
        return;
      }
      if (!isFile(node)) return;
      if (before && after) {
        if (node.lazy || node.stored) {
          if (!cache.move(before, after)) { if (node.lazy) dropNode(node); else node.stored = false; }
        } else {
          cache.del(before);          // memory only: written back under the new name on close
        }
      } else if (before) {           // into a memory-only folder
        if (node.lazy) load(node);
        cache.del(before);
        node.stored = false;
      } else if (after && !node.opens) {
        save(node);
      }
    },
    unlink(parent, name) {
      const node = parent.contents[name];
      if (node && node.persist && isFile(node)) { cache.del(keyOf(node)); node.deleted = true; }
      M.unlink(parent, name);
    },
  };

  function adopt(node, persist) {
    node.persist = persist;
    if (FS.isDir(node.mode)) node.node_ops = dirNode;
    else if (isFile(node)) { node.node_ops = fileNode; node.stream_ops = fileStream; }
  }

  function dropNode(node) {
    try { M.unlink(node.parent, node.name); FS.destroyNode(node); } catch (e) { /* gone */ }
  }

  const type = {
    mount(mount) {
      const root = MEMFS.mount(mount);
      root.node_ops = dirNode;
      root.isCacheRoot = true;
      root.persist = true;
      return root;
    },
  };
  FS.mount(type, {}, mountpoint);

  // a node for every cached file, contents left in the cache until opened
  for (const [key, e] of cache.index) {
    if (!key.startsWith("f|")) continue;
    const rel = key.slice(2);
    const top = rel.split("/")[0];
    if (!rel || memoryOnly.includes(top) || rel.split("/").some((p) => !p || p === "." || p === "..")) continue;
    const path = `${mountpoint}/${rel}`;
    try {
      const dir = path.slice(0, path.lastIndexOf("/"));
      if (dir !== mountpoint) FS.mkdirTree(dir);
      const node = FS.mknod(path, 0o100666, 0);
      node.lazy = true;
      node.stored = true;
      node.usedBytes = e.len;
      node.atime = node.mtime = node.ctime = e.time;
    } catch (err) { /* a file and a folder of the same name: skip */ }
  }

  // entries dropped to make room: remove their files (unless open; they're written back on close)
  cache.onDrop = (keys) => {
    for (const key of keys) {
      if (!key.startsWith("f|")) continue;
      let node;
      try { node = FS.lookupPath(`${mountpoint}/${key.slice(2)}`).node; } catch (e) { continue; }
      if (node.busy || !isFile(node)) continue;
      node.stored = false;
      if (node.opens > 0) node.dirty = true;
      else if (node.lazy) dropNode(node);
    }
  };
}

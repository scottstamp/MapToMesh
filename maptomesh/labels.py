"""Frame labels: a small single-stroke font, a scale bar and a north arrow.

Everything is built from line segments (in mm) with a stroke width; ``stroke_field``
turns them into a signed distance on the vertex grid (positive inside a stroke),
which the partition cuts exactly - so labels have smooth outlines at any grid size.
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np

# Glyphs on a 4 x 6 grid (baseline y = 0, cap height 6), as polylines.
_O = [(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)]
_P = [(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]
GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], [(0, 3), (4, 3)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]],
    "D": [[(0, 0), (0, 6), (2, 6), (4, 4), (4, 2), (2, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(1, 6), (3, 6)], [(2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "J": [[(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 2)], [(1, 3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [_O],
    "P": [_P],
    "Q": [_O, [(2, 2), (4, 0)]],
    "R": [_P, [(2, 3), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 4), (3, 0), (4, 6)]],
    "X": [[(0, 6), (4, 0)], [(0, 0), (4, 6)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    "0": [_O, [(0.5, 1), (3.5, 5)]],
    "1": [[(1, 5), (2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "2": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)]],
    "3": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3)],
          [(3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(4, 6), (0, 6), (0, 3), (3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "6": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)]],
    "7": [[(0, 6), (4, 6), (1, 0)]],
    "8": [[(1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0),
           (3, 0), (4, 1), (4, 2), (3, 3)]],
    "9": [[(0, 1), (1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (4, 3)]],
    "-": [[(1, 3), (3, 3)]],
    ".": [[(2, 0), (2, 0.4)]],
    ",": [[(2, 0.5), (1.5, -1)]],
    ":": [[(2, 1), (2, 1.4)], [(2, 4.2), (2, 4.6)]],
    "'": [[(2, 6), (2, 4.5)]],
    "/": [[(0, 0), (4, 6)]],
    "(": [[(3, 6), (2, 5), (2, 1), (3, 0)]],
    ")": [[(1, 6), (2, 5), (2, 1), (1, 0)]],
    "&": [[(4, 0), (1, 5), (2, 6), (3, 5), (0, 2), (0, 1), (1, 0), (2, 0), (4, 2)]],
}
ADVANCE, SPACE = 5.0, 3.0


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return "".join(ch if ch in GLYPHS or ch == " " else " " for ch in t).strip()


def text_width(text: str, height: float) -> float:
    s = height / 6.0
    w = sum(SPACE if ch == " " else ADVANCE for ch in text)
    return max(w - 1.0, 0.0) * s


def text_segments(text: str, height: float, x0: float, y0: float) -> list[tuple[float, float, float, float]]:
    """Segments (x1, y1, x2, y2) in mm for ``text`` with its baseline-left at (x0, y0)."""
    s = height / 6.0
    out, x = [], x0
    for ch in text:
        if ch == " ":
            x += SPACE * s
            continue
        for line in GLYPHS[ch]:
            for (a, b), (c, d) in zip(line[:-1], line[1:]):
                out.append((x + a * s, y0 + b * s, x + c * s, y0 + d * s))
        x += ADVANCE * s
    return out


def nice_length(max_m: float) -> float:
    """Largest 1/2/5 x 10^n length not exceeding ``max_m``."""
    if max_m <= 0:
        return 0.0
    e = 10 ** math.floor(math.log10(max_m))
    for m in (5, 2, 1):
        if m * e <= max_m:
            return m * e
    return e


def length_text(m: float) -> str:
    return f"{m / 1000:g} KM" if m >= 1000 else f"{m:g} M"


def scale_bar(length_mm: float, length_m: float, height: float, x0: float, yc: float):
    """Bar with end ticks and its length label; returns (segments, total width)."""
    th = height * 0.5
    segs = [(x0, yc, x0 + length_mm, yc),
            (x0, yc - th / 2, x0, yc + th / 2), (x0 + length_mm, yc - th / 2, x0 + length_mm, yc + th / 2)]
    txt = length_text(length_m)
    th_txt = height * 0.8
    gap = height * 0.6
    segs += text_segments(txt, th_txt, x0 + length_mm + gap, yc - th_txt / 2)
    return segs, length_mm + gap + text_width(txt, th_txt)


def north_arrow(height: float, xc: float, yc: float):
    h = height
    segs = [(xc, yc - h / 2, xc, yc + h / 2),                      # shaft
            (xc, yc + h / 2, xc - h * 0.3, yc + h * 0.1),          # head
            (xc, yc + h / 2, xc + h * 0.3, yc + h * 0.1)]
    segs += text_segments("N", h * 0.8, xc + h * 0.75, yc - h * 0.4)
    return segs


def stroke_field(segments, half_width: float, shape: tuple[int, int], cell: float) -> np.ndarray:
    """Signed distance (cells) to the strokes on the (R+1, C+1) vertex grid; positive inside."""
    H, W = shape
    out = np.full(shape, -1e9)
    hw = half_width / cell
    for x1, y1, x2, y2 in segments:
        a = np.array([x1, y1]) / cell
        b = np.array([x2, y2]) / cell
        lo = np.floor(np.minimum(a, b) - hw - 2).astype(int)
        hi = np.ceil(np.maximum(a, b) + hw + 2).astype(int)
        c0, r0 = max(lo[0], 0), max(lo[1], 0)
        c1, r1 = min(hi[0], W - 1), min(hi[1], H - 1)
        if c1 < c0 or r1 < r0:
            continue
        yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1].astype(np.float64)
        ab = b - a
        L2 = float(ab @ ab)
        t = np.zeros_like(xx) if L2 == 0 else np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2, 0, 1)
        d = np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))
        out[r0:r1 + 1, c0:c1 + 1] = np.maximum(out[r0:r1 + 1, c0:c1 + 1], hw - d)
    return out

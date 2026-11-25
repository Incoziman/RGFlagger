"""Drawing helpers: LED boxes with state labels, mask previews, tiled grids."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import classify, color_masks, merge_thresholds

__all__ = ["STATE_BGR", "draw", "mask_image", "tile", "label"]

# BGR colours for box outlines.
STATE_BGR = {
    "OK": (0, 200, 0),
    "FAULT": (0, 140, 255),
    "DARK": (150, 150, 150),
    "UNKNOWN": (255, 150, 0),
}
# How each mask is painted in mask_image().
_MASK_BGR = {"green": (0, 220, 0), "amber": (0, 150, 255), "red": (0, 0, 255)}


def label(img: np.ndarray, text: str, org, color=(255, 255, 255), scale: float = 0.5) -> None:
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def mask_image(bgr: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    """What the filter sees: red/amber/green pixels in their colour, lit-but-white in grey, rest black."""
    m = color_masks(bgr, thresholds)
    out = np.zeros_like(bgr)
    out[m["lit"]] = (90, 90, 90)
    for key, color in _MASK_BGR.items():
        out[m[key]] = color
    return out


def _mark_spots(out: np.ndarray, roi: np.ndarray, x: int, y: int, thresholds: dict | None) -> None:
    """Circle every red/amber spot (>= min_spot_px pixels) inside an "any" zone."""
    t = merge_thresholds(thresholds)
    m = color_masks(roi, t)
    for key in ("red", "amber"):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m[key].astype(np.uint8), connectivity=8)
        for sx, sy, sw, sh, area in stats[1:]:
            if area >= max(1, t["min_spot_px"] // 2):
                cx, cy = x + sx + sw // 2, y + sy + sh // 2
                cv2.circle(out, (int(cx), int(cy)), int(max(sw, sh) // 2 + 5), _MASK_BGR[key], 2)


def draw(frame: np.ndarray, leds: list, statuses: dict | None = None,
         thresholds: dict | None = None, debug: bool = False) -> np.ndarray:
    """Copy of `frame` with every LED box outlined and labelled.

    statuses: {port: PortStatus} from Monitor.status() / check(). Without it,
    each box is classified on the spot and labelled with its colour only.
    debug: also paste a zoomed mask of each box next to it.
    "any" zones are tagged [any] and every red/amber spot in them is circled.
    """
    out = frame.copy()
    H, W = out.shape[:2]
    for led in leds:
        x, y, w, h = (int(v) for v in led["box"])
        mode = led.get("mode", "led")
        tag = " [any]" if mode == "any" else ""
        status = (statuses or {}).get(led["name"])
        roi = frame[max(0, y):y + h, max(0, x):x + w]
        if status is not None:
            state = str(status.state)
            text = f"{led['name']}{tag} {state} {status.color}"
        else:
            state = "UNKNOWN"
            text = f"{led['name']}{tag} {classify(roi, thresholds, mode).color}"
        color = STATE_BGR.get(state, STATE_BGR["UNKNOWN"])
        thick = 3 if state in ("FAULT", "DARK") else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
        label(out, text, (x, max(14, y - 6)), color)
        if mode == "any" and roi.size:
            _mark_spots(out, roi, max(0, x), max(0, y), thresholds)

        if debug and roi.size:
            zoom = max(1, min(8, 64 // max(1, max(w, h))))
            inset = cv2.resize(mask_image(roi, thresholds), None, fx=zoom, fy=zoom,
                               interpolation=cv2.INTER_NEAREST)
            ih, iw = inset.shape[:2]
            ix = x + w + 4 if x + w + 4 + iw <= W else max(0, x - 4 - iw)
            iy = min(max(0, y), max(0, H - ih))
            if iy + ih <= H and ix + iw <= W:
                out[iy:iy + ih, ix:ix + iw] = inset
                cv2.rectangle(out, (ix, iy), (ix + iw - 1, iy + ih - 1), color, 1)
    return out


def tile(frames: list, tile_w: int = 640, tile_h: int = 360, cols: int | None = None,
         names: list | None = None) -> np.ndarray:
    """Resize frames and arrange them in a grid. None -> 'no signal' tile."""
    if cols is None:
        cols = 1 if len(frames) == 1 else 2
    tiles = []
    for i, frame in enumerate(frames):
        if frame is None:
            t = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            label(t, "no signal", (20, 60), (0, 0, 255), 1.0)
        else:
            t = cv2.resize(frame, (tile_w, tile_h))
        if names:
            label(t, names[i], (10, 26), (255, 255, 255), 0.7)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)

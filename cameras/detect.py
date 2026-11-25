"""Classify the colour of one LED from a small image region.

Pure image logic: no cameras, no threads. Everything works on BGR numpy
arrays as returned by OpenCV.

How it decides:
  1. "lit" pixels are bright (V >= v_min). Too few lit pixels -> OFF.
  2. Bright LEDs often blow out to a white core with a coloured ring around
     it, so the colour is taken only from pixels that are reasonably bright
     (V >= v_color) AND saturated (S >= s_min). These are mostly the ring.
  3. Those coloured pixels vote RED / AMBER / GREEN by hue. Red sits at
     both ends of OpenCV's hue scale (0-179), so it is two ranges OR'd.
  4. Lit but (almost) no saturated pixels -> OTHER (white / unknown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

__all__ = ["LedColor", "LedReading", "DEFAULTS", "MODES", "merge_thresholds", "color_masks",
           "classify_led", "classify_any", "classify"]


class LedColor(str, Enum):
    OFF = "OFF"
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    OTHER = "OTHER"

    def __str__(self) -> str:
        return self.value


# All tunable numbers in one place. Override per camera in leds.json
# ("thresholds": {...}) or live with `python -m cameras tune <camera>`.
DEFAULTS = {
    "v_min": 170,        # brightness (0-255) for a pixel to count as lit
    "v_color": 110,      # minimum brightness for a pixel to vote on colour
    "s_min": 90,         # minimum saturation (0-255) for a pixel to vote on colour
    "red_max": 8,        # red   = hue 0..red_max  OR  red_wrap..179
    "red_wrap": 165,
    "amber_min": 9,      # amber/orange = amber_min..amber_max
    "amber_max": 28,
    "green_min": 40,     # green = green_min..green_max
    "green_max": 90,
    "min_lit_pct": 3,    # % of the box that must be lit to count as "on"
    "min_color_pct": 10, # % of lit pixels that must be coloured, else OTHER (white)
    "min_spot_px": 6,    # "any" boxes: this many red/amber pixels anywhere in the box = a hit
}

# Box modes (per box in leds.json, "mode": ...):
#   "led"  one LED per box; the majority colour of the box wins (default)
#   "any"  a zone (e.g. a whole switch); ANY red/amber spot in it is a hit
MODES = ("led", "any")


def merge_thresholds(overrides: dict | None = None) -> dict:
    """DEFAULTS updated with any overrides (unknown keys are ignored)."""
    t = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key in t:
            t[key] = int(value)
    return t


def _in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return (h >= lo) & (h <= hi)


def color_masks(bgr: np.ndarray, thresholds: dict | None = None) -> dict:
    """Boolean masks for one image: lit, colored, red, amber, green.

    Works on a small LED box or on a whole frame (used by the tune view).
    """
    t = merge_thresholds(thresholds)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = v >= t["v_min"]
    colored = (v >= t["v_color"]) & (s >= t["s_min"])
    red = colored & (_in_range(h, 0, t["red_max"]) | _in_range(h, t["red_wrap"], 179))
    amber = colored & _in_range(h, t["amber_min"], t["amber_max"])
    green = colored & _in_range(h, t["green_min"], t["green_max"])
    return {"lit": lit, "colored": colored, "red": red, "amber": amber, "green": green}


@dataclass
class LedReading:
    color: LedColor
    lit_pct: float                       # % of the box that is lit
    votes: dict = field(default_factory=dict)  # coloured pixel count per colour

    @property
    def is_lit(self) -> bool:
        return self.color != LedColor.OFF


def classify_led(roi_bgr: np.ndarray, thresholds: dict | None = None) -> LedReading:
    """Classify one LED box as OFF / GREEN / AMBER / RED / OTHER."""
    t = merge_thresholds(thresholds)
    if roi_bgr is None or roi_bgr.size == 0:
        return LedReading(LedColor.OFF, 0.0)
    m = color_masks(roi_bgr, t)
    total = m["lit"].size
    lit_count = int(m["lit"].sum())
    votes = {
        LedColor.RED: int(m["red"].sum()),
        LedColor.AMBER: int(m["amber"].sum()),
        LedColor.GREEN: int(m["green"].sum()),
    }
    colored_count = int(m["colored"].sum())
    lit_pct = 100.0 * lit_count / total
    # A strongly coloured LED may not reach v_min everywhere; count its coloured
    # pixels as "on" too so a saturated red/green LED is never read as OFF.
    on_pct = 100.0 * max(lit_count, sum(votes.values())) / total
    if on_pct < t["min_lit_pct"]:
        return LedReading(LedColor.OFF, lit_pct, votes)

    best = max(votes, key=votes.get)
    reference = max(lit_count, colored_count, 1)
    if votes[best] == 0 or 100.0 * votes[best] / reference < t["min_color_pct"]:
        return LedReading(LedColor.OTHER, lit_pct, votes)
    return LedReading(best, lit_pct, votes)


def classify_any(roi_bgr: np.ndarray, thresholds: dict | None = None) -> LedReading:
    """Zone check: RED/AMBER if at least `min_spot_px` red/amber pixels exist ANYWHERE in the box.

    No majority vote, so one small red LED among many green ones still counts.
    Without red/amber: GREEN if there is a green spot, OTHER if anything is lit, else OFF.
    """
    t = merge_thresholds(thresholds)
    if roi_bgr is None or roi_bgr.size == 0:
        return LedReading(LedColor.OFF, 0.0)
    m = color_masks(roi_bgr, t)
    votes = {
        LedColor.RED: int(m["red"].sum()),
        LedColor.AMBER: int(m["amber"].sum()),
        LedColor.GREEN: int(m["green"].sum()),
    }
    lit_count = int(m["lit"].sum())
    lit_pct = 100.0 * lit_count / m["lit"].size
    spot = t["min_spot_px"]
    red, amber = votes[LedColor.RED], votes[LedColor.AMBER]
    if red >= spot or amber >= spot:
        return LedReading(LedColor.RED if red >= amber else LedColor.AMBER, lit_pct, votes)
    if votes[LedColor.GREEN] >= spot:
        return LedReading(LedColor.GREEN, lit_pct, votes)
    if lit_count >= spot:
        return LedReading(LedColor.OTHER, lit_pct, votes)
    return LedReading(LedColor.OFF, lit_pct, votes)


def classify(roi_bgr: np.ndarray, thresholds: dict | None = None, mode: str = "led") -> LedReading:
    """classify_led for "led" boxes, classify_any for "any" boxes."""
    return classify_any(roi_bgr, thresholds) if mode == "any" else classify_led(roi_bgr, thresholds)

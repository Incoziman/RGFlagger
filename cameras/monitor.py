"""Watch calibrated port LEDs over time and raise events when they change.

Per port the state is decided over a rolling window (default 5 s), so normal
activity blinking doesn't trigger anything:

    FAULT  amber/red in >= fault_ratio of the window   (default 60 %)
    DARK   port was lit at calibration but is lit in <= dark_ratio of the window (10 %)
    OK     otherwise

A new state must then hold for `hold_seconds` (default 10) before an Event fires.

API:
    check("camera1")                          one-shot, sync -> {port: PortStatus}
    Monitor(on_event=callback).run()          blocking
    with Monitor(on_event=cb) as m: m.status()  background threads
    async for ev in Monitor().events(): ...   asyncio
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import LEDS_FILE, _resolve_name, open_camera
from .detect import LedColor, classify, merge_thresholds

__all__ = [
    "PortState", "PortStatus", "Event", "PortTracker", "Monitor", "check",
    "load_leds", "save_leds", "setup_logging", "crop", "baseline_from",
]

log = logging.getLogger("cameras.leds")

FAULT_COLORS = {LedColor.RED, LedColor.AMBER}


class PortState(str, Enum):
    UNKNOWN = "UNKNOWN"   # not enough data yet
    OK = "OK"
    FAULT = "FAULT"       # LED amber/red
    DARK = "DARK"         # LED should be lit but is off
    # camera-level events
    CAMERA_ONLINE = "CAMERA_ONLINE"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"

    def __str__(self) -> str:
        return self.value


# --------------------------------------------------------------------------- config

def load_leds(path: str | Path | None = None) -> dict:
    """Read leds.json: {camera: {"thresholds": {...}, "leds": [{"name", "box", "baseline"}]}}."""
    path = Path(path or LEDS_FILE)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else {}


def save_leds(config: dict, path: str | Path | None = None) -> Path:
    path = Path(path or LEDS_FILE)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def crop(frame: np.ndarray, box) -> np.ndarray:
    """Cut an [x, y, w, h] box out of a frame (clipped to the frame)."""
    x, y, w, h = (int(v) for v in box)
    H, W = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return frame[y0:y1, x0:x1]


def baseline_from(colors: list) -> LedColor:
    """The 'normal' colour of a port from samples taken at calibration.

    Lit in at least 20 % of samples -> its most common lit colour (so a
    blinking activity LED counts as lit), otherwise OFF (unused port).
    """
    if not colors:
        return LedColor.OFF
    lit = [c for c in colors if c != LedColor.OFF]
    if len(lit) < 0.2 * len(colors):
        return LedColor.OFF
    return Counter(lit).most_common(1)[0][0]


# --------------------------------------------------------------------------- data

@dataclass
class PortStatus:
    camera: str
    port: str
    state: PortState
    color: LedColor             # latest reading
    baseline: LedColor
    fault_pct: float = 0.0      # % of window samples amber/red
    lit_pct: float = 0.0        # % of window samples lit
    since: float = 0.0          # epoch time the current state started

    def __str__(self) -> str:
        return (f"{self.camera}/{self.port}: {self.state} (now {self.color}, normal {self.baseline}, "
                f"fault {self.fault_pct:.0f}%, lit {self.lit_pct:.0f}%)")


@dataclass
class Event:
    camera: str
    port: Optional[str]          # None for camera online/offline events
    old_state: PortState
    new_state: PortState
    color: Optional[LedColor] = None
    time: float = field(default_factory=time.time)
    snapshot: Optional[bytes] = field(default=None, repr=False)  # JPEG of the annotated frame
    mode: str = "led"            # "led" box or "any" zone

    @property
    def is_problem(self) -> bool:
        return self.new_state in (PortState.FAULT, PortState.DARK, PortState.CAMERA_OFFLINE)

    @property
    def message(self) -> str:
        if self.port is None:
            return f"{self.camera}: {'camera offline' if self.new_state == PortState.CAMERA_OFFLINE else 'camera online'}"
        fault = (f"{self.color} spot in zone - possible connection fault" if self.mode == "any"
                 else f"LED {self.color} - possible connection fault")
        what = {
            PortState.FAULT: fault,
            PortState.DARK: "LED went dark - link lost?",
            PortState.OK: "back to normal" if self.old_state != PortState.UNKNOWN else "OK",
        }.get(self.new_state, str(self.new_state))
        return f"{self.camera}/{self.port}: {what} ({self.old_state} -> {self.new_state})"

    def __str__(self) -> str:
        return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.time))} {self.message}"


# --------------------------------------------------------------------------- state machine

class PortTracker:
    """Rolling-window state for one LED. Time is passed in, so it is easy to test."""

    def __init__(self, camera: str, port: str, baseline: LedColor | str = LedColor.GREEN,
                 window: float = 5.0, hold: float = 10.0,
                 fault_ratio: float = 0.6, dark_ratio: float = 0.1, mode: str = "led"):
        self.camera, self.port, self.mode = camera, port, mode
        self.baseline = LedColor(baseline)
        self.window, self.hold = window, hold
        self.fault_ratio, self.dark_ratio = fault_ratio, dark_ratio
        if mode == "any":
            # Zone: any red/amber is a fault, and "dark" means nothing for a group of LEDs.
            self.fault_colors = set(FAULT_COLORS)
            self.can_go_dark = False
        else:
            # A port whose normal colour is amber (e.g. a 100 Mb link) only faults on red.
            self.fault_colors = FAULT_COLORS - {self.baseline}
            self.can_go_dark = self.baseline != LedColor.OFF
        self.samples: deque = deque()
        self.state = PortState.UNKNOWN
        self.since = 0.0
        self.color = LedColor.OFF
        self._first_t: Optional[float] = None
        self._candidate: Optional[PortState] = None
        self._candidate_since = 0.0

    def reset(self) -> None:
        """Forget samples (e.g. after the camera dropped); keeps the current state."""
        self.samples.clear()
        self._first_t = None
        self._candidate = None

    def evaluate(self) -> tuple[PortState, float, float]:
        """(raw state, fault %, lit %) for the samples currently in the window."""
        n = len(self.samples)
        if n == 0:
            return PortState.UNKNOWN, 0.0, 0.0
        fault = sum(1 for _, c in self.samples if c in self.fault_colors) / n
        lit = sum(1 for _, c in self.samples if c != LedColor.OFF) / n
        if fault >= self.fault_ratio:
            state = PortState.FAULT
        elif self.can_go_dark and lit <= self.dark_ratio:
            state = PortState.DARK
        else:
            state = PortState.OK
        return state, 100 * fault, 100 * lit

    def add(self, t: float, color: LedColor) -> Optional[tuple[PortState, PortState]]:
        """Add one reading. Returns (old, new) when the confirmed state changes."""
        self.color = color
        self.samples.append((t, color))
        while self.samples and self.samples[0][0] < t - self.window:
            self.samples.popleft()
        if self._first_t is None:
            self._first_t = t
        if t - self._first_t < self.window:      # still filling the window
            return None

        raw, _, _ = self.evaluate()
        if raw == self.state:
            self._candidate = None
            return None
        if self.state == PortState.UNKNOWN:      # first decision: the window is debounce enough
            return self._switch(raw, t)
        if raw != self._candidate:
            self._candidate, self._candidate_since = raw, t
        if t - self._candidate_since >= self.hold:
            return self._switch(raw, t)
        return None

    def _switch(self, new: PortState, t: float) -> tuple[PortState, PortState]:
        old, self.state, self.since, self._candidate = self.state, new, t, None
        return old, new

    def status(self) -> PortStatus:
        _, fault, lit = self.evaluate()
        return PortStatus(self.camera, self.port, self.state, self.color, self.baseline, fault, lit, self.since)


# --------------------------------------------------------------------------- frame grabbing

class _Grabber(threading.Thread):
    """Keeps only the newest frame of one camera; reconnects with backoff."""

    def __init__(self, camera: str, opener: Callable, on_change: Callable[[str, bool], None]):
        super().__init__(name=f"grab-{camera}", daemon=True)
        self.camera, self._opener, self._on_change = camera, opener, on_change
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._stop_evt = threading.Event()
        self.online = False

    def latest(self) -> tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._frame_id, self._frame

    def stop(self) -> None:
        self._stop_evt.set()

    def _set_online(self, online: bool) -> None:
        """Report the state; the Monitor ignores repeated offline reports."""
        self.online = online
        if not online:
            with self._lock:
                self._frame = None
        self._on_change(self.camera, online)

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_evt.is_set():
            try:
                cap = self._opener(self.camera)
            except Exception as e:  # noqa: BLE001 - any failure means "offline, retry"
                log.debug("open %s failed: %s", self.camera, e)
                self._set_online(False)
                self._stop_evt.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            failures = 0
            try:
                while not self._stop_evt.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        failures += 1
                        if failures >= 5:
                            break
                        self._stop_evt.wait(0.2)
                        continue
                    failures = 0
                    with self._lock:
                        self._frame, self._frame_id = frame, self._frame_id + 1
                    if not self.online:
                        self._set_online(True)
            finally:
                cap.release()
            if not self._stop_evt.is_set():
                self._set_online(False)
                self._stop_evt.wait(1.0)


# --------------------------------------------------------------------------- monitor

class Monitor:
    """Continuously watch the calibrated LEDs of one or more cameras.

    on_event: called (from a background thread) with each Event.
    """

    def __init__(self, cameras: list | None = None, leds_file: str | Path | None = None,
                 on_event: Callable[[Event], None] | None = None,
                 window: float = 5.0, hold_seconds: float = 10.0, fps: float = 5.0,
                 fault_ratio: float = 0.6, dark_ratio: float = 0.1,
                 snapshots: bool = True, opener: Callable = open_camera):
        self.config = load_leds(leds_file)
        names = cameras if cameras else list(self.config)
        if not names:
            raise RuntimeError("No calibrated LEDs. Run: python -m cameras calibrate <camera>")
        self.cameras = [_resolve_name(c) if opener is open_camera else str(c) for c in names]
        for cam in self.cameras:
            if not self.leds(cam):
                log.warning("%s has no calibrated LEDs; only online/offline is watched. "
                            "Run: python -m cameras calibrate %s", cam, cam)
        self.fps, self.snapshots = fps, snapshots
        self._opener = opener
        self._trackers = {
            cam: {led["name"]: PortTracker(cam, led["name"], led.get("baseline", "GREEN"),
                                           window, hold_seconds, fault_ratio, dark_ratio,
                                           led.get("mode", "led"))
                  for led in self.leds(cam)}
            for cam in self.cameras
        }
        self._listeners: list[Callable[[Event], None]] = [on_event] if on_event else []
        self._lock = threading.RLock()
        self._grabbers: dict[str, _Grabber] = {}
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._offline_reported: set = set()

    # ---- config helpers (also used by the overlay)
    def leds(self, camera: str) -> list:
        return self.config.get(camera, {}).get("leds", [])

    def thresholds(self, camera: str) -> dict:
        return merge_thresholds(self.config.get(camera, {}).get("thresholds"))

    # ---- lifecycle
    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> "Monitor":
        if self.running:
            return self
        self._stop_evt.clear()
        self._grabbers = {cam: _Grabber(cam, self._opener, self._camera_changed) for cam in self.cameras}
        for g in self._grabbers.values():
            g.start()
        self._worker = threading.Thread(target=self._loop, name="led-monitor", daemon=True)
        self._worker.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        for g in self._grabbers.values():
            g.stop()
        if self._worker:
            self._worker.join(timeout=5)
        self._worker = None

    def run(self) -> None:
        """Start and block until Ctrl+C (or stop() from another thread)."""
        self.start()
        try:
            while not self._stop_evt.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> "Monitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- reading results
    def add_listener(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def status(self) -> dict:
        """{camera: {port: PortStatus}} snapshot. Safe to call from any thread."""
        with self._lock:
            return {cam: {p: t.status() for p, t in ports.items()} for cam, ports in self._trackers.items()}

    def problems(self) -> list:
        """All ports currently FAULT or DARK."""
        return [s for ports in self.status().values() for s in ports.values()
                if s.state in (PortState.FAULT, PortState.DARK)]

    def online(self, camera: str) -> bool:
        g = self._grabbers.get(camera)
        return bool(g and g.online)

    def latest_frame(self, camera: str) -> Optional[np.ndarray]:
        g = self._grabbers.get(camera)
        return None if g is None else g.latest()[1]

    async def events(self):
        """Async generator of Events. Starts the monitor if needed (and stops it afterwards)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def push(event: Event) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        self.add_listener(push)
        started_here = not self.running
        if started_here:
            self.start()
        try:
            while True:
                yield await queue.get()
        finally:
            self.remove_listener(push)
            if started_here:
                await loop.run_in_executor(None, self.stop)

    # ---- internals
    def _emit(self, event: Event) -> None:
        level = logging.WARNING if event.is_problem else logging.INFO
        log.log(level, event.message)
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(event)
            except RuntimeError:  # e.g. asyncio loop already closed
                self.remove_listener(cb)
            except Exception:  # noqa: BLE001 - a broken listener must not stop monitoring
                log.exception("event listener failed")

    def _camera_changed(self, camera: str, online: bool) -> None:
        with self._lock:
            if not online:
                for t in self._trackers[camera].values():
                    t.reset()
                if camera in self._offline_reported:
                    return
                self._offline_reported.add(camera)
            else:
                self._offline_reported.discard(camera)
        new = PortState.CAMERA_ONLINE if online else PortState.CAMERA_OFFLINE
        old = PortState.CAMERA_OFFLINE if online else PortState.CAMERA_ONLINE
        self._emit(Event(camera, None, old, new))

    def _snapshot(self, camera: str, frame: np.ndarray) -> Optional[bytes]:
        if not self.snapshots:
            return None
        from .overlay import draw

        ok, jpg = cv2.imencode(".jpg", draw(frame, self.leds(camera), self.status().get(camera)))
        return jpg.tobytes() if ok else None

    def _loop(self) -> None:
        last_ids = {cam: 0 for cam in self.cameras}
        period = 1.0 / self.fps
        while not self._stop_evt.is_set():
            started = time.monotonic()
            for cam, grabber in self._grabbers.items():
                frame_id, frame = grabber.latest()
                if frame is None or frame_id == last_ids[cam]:
                    continue
                last_ids[cam] = frame_id
                thresholds = self.thresholds(cam)
                now = time.time()
                changes = []
                with self._lock:
                    for led in self.leds(cam):
                        tracker = self._trackers[cam][led["name"]]
                        reading = classify(crop(frame, led["box"]), thresholds, led.get("mode", "led"))
                        change = tracker.add(now, reading.color)
                        if change:
                            changes.append((tracker, change))
                if changes:
                    snapshot = self._snapshot(cam, frame)
                    for tracker, (old, new) in changes:
                        self._emit(Event(cam, tracker.port, old, new, tracker.color, now, snapshot,
                                         tracker.mode))
            self._stop_evt.wait(max(0.0, period - (time.monotonic() - started)))


# --------------------------------------------------------------------------- one-shot

def check(camera: str | int, seconds: float = 5.0, leds_file: str | Path | None = None,
          fps: float = 5.0, fault_ratio: float = 0.6, dark_ratio: float = 0.1,
          opener: Callable = open_camera) -> dict:
    """Watch one camera for `seconds` and return {port: PortStatus}. Blocking.

    Uses the same window rules as Monitor, but without the hold time.
    """
    cam = _resolve_name(camera) if opener is open_camera else str(camera)
    config = load_leds(leds_file).get(cam, {})
    leds = config.get("leds", [])
    if not leds:
        raise RuntimeError(f"{cam} has no calibrated LEDs. Run: python -m cameras calibrate {cam}")
    thresholds = merge_thresholds(config.get("thresholds"))
    trackers = {led["name"]: PortTracker(cam, led["name"], led.get("baseline", "GREEN"),
                                         window=float("inf"), hold=0,
                                         fault_ratio=fault_ratio, dark_ratio=dark_ratio,
                                         mode=led.get("mode", "led"))
                for led in leds}
    cap = opener(cam)
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tick = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()
            for led in leds:
                reading = classify(crop(frame, led["box"]), thresholds, led.get("mode", "led"))
                trackers[led["name"]].add(now, reading.color)
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - tick)))
    finally:
        cap.release()
    result = {}
    for name, t in trackers.items():
        if not t.samples:
            raise ConnectionError(f"Connected to {cam} but could not read any frames.")
        state, fault, lit = t.evaluate()
        result[name] = PortStatus(cam, name, state, t.color, t.baseline, fault, lit, time.time())
    return result


# --------------------------------------------------------------------------- logging

def setup_logging(log_file: str | Path | None = "logs/leds.log", level: int = logging.INFO) -> logging.Logger:
    """Console + rotating file (5 MB x 5) for the 'cameras.leds' logger. Pass log_file=None for console only."""
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(level)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log

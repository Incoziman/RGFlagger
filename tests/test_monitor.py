import asyncio
import json
import threading
import time

import cv2
import numpy as np

from cameras.detect import LedColor
from cameras.monitor import Monitor, PortState, PortTracker, baseline_from, check

G, A, R, OFF = LedColor.GREEN, LedColor.AMBER, LedColor.RED, LedColor.OFF
FPS = 5


def feed(tracker, colors, start=0.0):
    """Feed colours at FPS; return the list of (time, (old, new)) transitions."""
    out = []
    for i, c in enumerate(colors):
        t = start + i / FPS
        change = tracker.add(t, c)
        if change:
            out.append((t, change))
    return out


def seconds(color, s):
    return [color] * int(s * FPS)


def test_initial_state_decided_after_window():
    t = PortTracker("cam", "p1", G, window=5, hold=10)
    changes = feed(t, seconds(G, 6))
    assert [c for _, c in changes] == [(PortState.UNKNOWN, PortState.OK)]
    assert changes[0][0] >= 5


def test_blinking_green_stays_ok():
    t = PortTracker("cam", "p1", G, window=5, hold=10)
    blink = [G, OFF, OFF, G, OFF, OFF, OFF] * 20      # lit ~30 % of the time
    changes = feed(t, blink)
    assert [c for _, c in changes] == [(PortState.UNKNOWN, PortState.OK)]


def test_sustained_amber_faults_only_after_hold():
    t = PortTracker("cam", "p1", G, window=5, hold=10)
    changes = feed(t, seconds(G, 6) + seconds(A, 20))
    fault = [(ts, c) for ts, c in changes if c[1] == PortState.FAULT]
    assert len(fault) == 1
    # amber starts at 6 s, window needs >=60 % amber (~3 s), then hold 10 s
    assert fault[0][0] >= 6 + 10


def test_short_amber_burst_is_ignored():
    t = PortTracker("cam", "p1", G, window=5, hold=10)
    changes = feed(t, seconds(G, 6) + seconds(A, 8) + seconds(G, 20))
    assert all(c[1] != PortState.FAULT for _, c in changes)


def test_lit_then_off_goes_dark_and_recovers():
    t = PortTracker("cam", "p1", G, window=5, hold=10)
    changes = feed(t, seconds(G, 6) + seconds(OFF, 25) + seconds(G, 25))
    states = [c[1] for _, c in changes]
    assert states == [PortState.OK, PortState.DARK, PortState.OK]


def test_unused_port_never_goes_dark():
    t = PortTracker("cam", "p1", OFF, window=5, hold=10)
    changes = feed(t, seconds(OFF, 60))
    assert [c[1] for _, c in changes] == [PortState.OK]


def test_amber_baseline_only_faults_on_red():
    t = PortTracker("cam", "p1", A, window=5, hold=10)
    assert all(c[1] != PortState.FAULT for _, c in feed(t, seconds(A, 30)))
    changes = feed(t, seconds(R, 20), start=30)
    assert [c[1] for _, c in changes] == [PortState.FAULT]


def test_red_faults():
    t = PortTracker("cam", "p1", G, window=5, hold=2)
    changes = feed(t, seconds(G, 6) + seconds(R, 10))
    assert changes[-1][1] == (PortState.OK, PortState.FAULT)


def test_baseline_from():
    assert baseline_from([G, OFF, OFF, G, OFF]) == G
    assert baseline_from([OFF] * 9 + [G]) == OFF
    assert baseline_from([]) == OFF
    assert baseline_from([A, A, G]) == A


# ---------------------------------------------------------------- Monitor with a fake camera

class FakeCap:
    """Stands in for cv2.VideoCapture: a dark frame with one LED whose colour we control."""

    def __init__(self, source):
        self.source = source

    def read(self):
        time.sleep(0.02)
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        hue = {G: 60, A: 15, R: 3}.get(self.source.color)
        if hue is not None:
            bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            frame[20:30, 20:30] = bgr
        return True, frame

    def release(self):
        pass


class Source:
    color = G

    def open(self, name):
        return FakeCap(self)


def leds_file(tmp_path):
    path = tmp_path / "leds.json"
    path.write_text(json.dumps({"fake": {"leds": [{"name": "p1", "box": [18, 18, 14, 14], "baseline": "GREEN"}]}}))
    return path


def test_monitor_callback_and_status(tmp_path):
    src = Source()
    events = []
    got_fault = threading.Event()

    def on_event(ev):
        events.append(ev)
        if ev.new_state == PortState.FAULT:
            got_fault.set()

    mon = Monitor(leds_file=leds_file(tmp_path), on_event=on_event, window=0.5, hold_seconds=0.5,
                  fps=20, opener=src.open)
    with mon:
        time.sleep(1.0)
        assert mon.status()["fake"]["p1"].state == PortState.OK
        src.color = A
        assert got_fault.wait(5), [str(e) for e in events]
    states = [e.new_state for e in events]
    assert PortState.CAMERA_ONLINE in states and PortState.FAULT in states
    fault = next(e for e in events if e.new_state == PortState.FAULT)
    assert fault.port == "p1" and fault.snapshot and fault.snapshot[:2] == b"\xff\xd8"  # JPEG


def test_monitor_async_events(tmp_path):
    src = Source()
    mon = Monitor(leds_file=leds_file(tmp_path), window=0.5, hold_seconds=0.5, fps=20, opener=src.open)

    async def main():
        seen = []
        async for ev in mon.events():
            seen.append(ev.new_state)
            if ev.new_state == PortState.OK:
                src.color = OFF
            if ev.new_state == PortState.DARK:
                return seen

    seen = asyncio.run(asyncio.wait_for(main(), 10))
    assert seen[-1] == PortState.DARK
    assert not mon.running   # events() started it, so it stopped it too


def test_check_one_shot(tmp_path):
    src = Source()
    src.color = R
    result = check("fake", seconds=0.5, leds_file=leds_file(tmp_path), fps=20, opener=src.open)
    assert result["p1"].state == PortState.FAULT and result["p1"].color == R


def test_any_zone_faults_on_amber_even_with_amber_baseline_and_never_dark():
    t = PortTracker("cam", "zone1", A, window=5, hold=2, mode="any")
    changes = feed(t, seconds(A, 10) + seconds(OFF, 30))
    states = [c[1] for _, c in changes]
    assert states[0] == PortState.FAULT
    assert PortState.DARK not in states

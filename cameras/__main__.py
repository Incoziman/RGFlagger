"""Command line tool.

    python -m cameras list
    python -m cameras url camera1
    python -m cameras view [camera ...]          live video grid (q = quit)
    python -m cameras snapshot camera1 out.jpg

    python -m cameras tune webcam                live colour-mask debugger with sliders
    python -m cameras calibrate camera1          draw a box around each port LED
    python -m cameras check camera1              one-shot status table (exit 1 on problems)
    python -m cameras watch [camera ...] [--show | --debug]   continuous monitoring + log
"""

import argparse
import sys

from . import _resolve_name, get_url, is_local, list_cameras, open_camera, read_frame

ESC = 27
TUNE_SLIDERS = {  # key -> slider maximum
    "v_min": 255, "v_color": 255, "s_min": 255,
    "red_max": 30, "red_wrap": 179,
    "amber_min": 60, "amber_max": 60,
    "green_min": 179, "green_max": 179,
    "min_lit_pct": 50, "min_color_pct": 100, "min_spot_px": 200,
}


# --------------------------------------------------------------------------- helpers

def _window_closed(cv2, window):
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _quit_pressed(cv2, window, delay=1):
    key = cv2.waitKey(delay) & 0xFF
    return key in (ord("q"), ESC) or _window_closed(cv2, window), key


def _grid(frames, names):
    """Tile frames keeping the camera's aspect ratio (webcams are often 4:3)."""
    from .overlay import tile

    cols = 1 if len(frames) == 1 else 2
    tile_w = 1280 if cols == 1 else 640
    sample = next((f for f in frames if f is not None), None)
    tile_h = int(tile_w * sample.shape[0] / sample.shape[1]) if sample is not None else tile_w * 9 // 16
    return tile(frames, tile_w, tile_h, cols, names)


def _warmup(name):
    """Webcams need a few frames to settle exposure / white balance."""
    return 15 if is_local(name) else 0


# --------------------------------------------------------------------------- view / snapshot

def _view(names):
    import cv2

    caps = {}
    for name in names:
        try:
            caps[name] = open_camera(name)
        except ConnectionError as e:
            print(f"Warning: {e}", file=sys.stderr)
            caps[name] = None
    window = "cameras (press q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = []
            for cap in caps.values():
                ok, frame = cap.read() if cap is not None else (False, None)
                frames.append(frame if ok else None)
            cv2.imshow(window, _grid(frames, list(caps)))
            if _quit_pressed(cv2, window)[0]:
                break
    finally:
        for cap in caps.values():
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- tune

def _tune(name):
    import cv2
    import numpy as np

    from .detect import DEFAULTS, color_masks, merge_thresholds
    from .monitor import load_leds, save_leds
    from .overlay import draw

    cam = _resolve_name(name)
    config = load_leds()
    entry = config.get(cam, {})
    start = merge_thresholds(entry.get("thresholds"))
    leds = entry.get("leds", [])

    view_win = f"tune {cam}  |  s = save   r = reset   q = quit"
    ctrl_win = "tune controls"
    cv2.namedWindow(view_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 500, 40 * len(TUNE_SLIDERS))
    for key, maximum in TUNE_SLIDERS.items():
        cv2.createTrackbar(key, ctrl_win, start[key], maximum, lambda _v: None)

    print("Sliders (window 'tune controls'):")
    print("  v_min      brightness for a pixel to count as lit")
    print("  v_color/s_min  brightness/saturation for a pixel to vote on colour")
    print("  red_max/red_wrap, amber_*, green_*  hue ranges (0-179; red wraps around)")
    print("Panels: frame with boxes | RED mask | AMBER mask | GREEN mask")
    if not leds:
        print(f"(No LED boxes for {cam} yet; the masks cover the whole frame.)")

    cap = open_camera(cam)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if _quit_pressed(cv2, view_win, 100)[0]:
                    break
                continue
            t = {k: cv2.getTrackbarPos(k, ctrl_win) for k in TUNE_SLIDERS}
            masks = color_masks(frame, t)
            panels = [draw(frame, leds, thresholds=t, debug=True)]
            names = [cam]
            for key in ("red", "amber", "green"):
                panels.append(np.where(masks[key][..., None], frame, 0).astype(frame.dtype))
                names.append(f"{key.upper()}  {int(masks[key].sum())} px")
            cv2.imshow(view_win, _grid(panels, names))

            quit_, key = _quit_pressed(cv2, view_win)
            if quit_:
                break
            if key == ord("s"):
                config = load_leds()
                changed = {k: v for k, v in t.items() if v != DEFAULTS[k]}
                config.setdefault(cam, {})["thresholds"] = changed
                path = save_leds(config)
                print(f"Saved thresholds for {cam} to {path}: {changed or '(defaults)'}")
            elif key == ord("r"):
                for k in TUNE_SLIDERS:
                    cv2.setTrackbarPos(k, ctrl_win, DEFAULTS[k])
    finally:
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- calibrate

def _calibrate(name, seconds, append, scale, any_mode=False):
    import time

    import cv2

    from .detect import LedColor, classify, merge_thresholds
    from .monitor import baseline_from, crop, load_leds, save_leds
    from .overlay import draw

    cam = _resolve_name(name)
    config = load_leds()
    entry = config.get(cam, {})
    thresholds = merge_thresholds(entry.get("thresholds"))
    existing = entry.get("leds", []) if append else []

    cap = open_camera(cam)
    try:
        for _ in range(_warmup(cam)):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise ConnectionError(f"Connected to {cam} but could not read a frame.")

        H, W = frame.shape[:2]
        if scale is None:
            scale = min(4.0, max(0.5, 1600 / W))
        shown = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        for led in existing:  # show the boxes we keep
            x, y, w, h = (int(v * scale) for v in led["box"])
            cv2.rectangle(shown, (x, y), (x + w, y + h), (255, 150, 0), 1)

        print(f"Calibrating {cam} ({W}x{H}, shown at {scale:.2f}x).")
        mode = "any" if any_mode else "led"
        if any_mode:
            print("  ZONE mode: drag a box around a whole area (e.g. a switch); ANY red/amber spot")
            print("  inside it will alert. Press ENTER or SPACE after each box, ESC when done.")
        else:
            print("  Drag a TIGHT box around one port LED, then press ENTER or SPACE.")
            print("  Repeat for every LED. Press ESC when done.")
        what = "zone" if any_mode else "LED"
        win = f"calibrate {cam}: box each {what}, ENTER after each, ESC when done"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, shown.shape[1], shown.shape[0])
        rois = cv2.selectROIs(win, shown, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(win)
        cv2.waitKey(1)

        boxes = [[int(round(v / scale)) for v in r] for r in (rois if rois is not None else [])]
        boxes = [[x, y, max(1, w), max(1, h)] for x, y, w, h in boxes if w > 0 and h > 0]
        if not boxes:
            print("No boxes drawn; nothing changed.")
            return 0

        taken = {led["name"] for led in existing}
        leds = []
        if sys.stdin.isatty():
            print(f"\nName the {len(boxes)} box(es) (this is NOT a shell prompt; press Enter for the default).")
        for i, box in enumerate(boxes):
            n = len(existing) + i + 1
            prefix = "zone" if any_mode else "p"
            default = f"{prefix}{n}"
            while default in taken:
                n += 1
                default = f"{prefix}{n}"
            label = default
            if sys.stdin.isatty():
                label = input(f"  {what} name for box {i + 1}/{len(boxes)} [{default}]: ").strip() or default
            if label in taken:
                print(f"  '{label}' already used, using {default}")
                label = default
            taken.add(label)
            leds.append({"name": label, "box": box, "mode": mode})

        print(f"Sampling the LEDs for {seconds:.0f} s to learn their normal colour. "
              "Make sure every port is in its HEALTHY state now...")
        samples = {led["name"]: [] for led in leds}
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            for led in leds:
                samples[led["name"]].append(classify(crop(frame, led["box"]), thresholds, mode).color)
            time.sleep(0.1)

        for led in leds:
            baseline = baseline_from(samples[led["name"]])
            led["baseline"] = baseline.value
            counts = {}
            for c in samples[led["name"]]:
                counts[c.value] = counts.get(c.value, 0) + 1
            note = ""
            if any_mode:
                if baseline in (LedColor.AMBER, LedColor.RED):
                    note = (f"  (WARNING: {baseline} is ALREADY visible in this zone, so it will alert. "
                            "Redraw the zone to leave out red/orange labels or cables, or raise min_spot_px in `tune`)")
            elif baseline == LedColor.OFF:
                note = "  (unused port: will not alert when dark)"
            elif baseline in (LedColor.AMBER, LedColor.RED):
                note = f"  (WARNING: {baseline} treated as normal here; only a different fault colour alerts)"
            elif baseline == LedColor.OTHER:
                note = "  (white/unclear: try `tune` to adjust thresholds, or lower camera exposure)"
            print(f"  {led['name']:<12} normal = {baseline.value:<6} samples {counts}{note}")

        config = load_leds()
        config.setdefault(cam, {})["leds"] = existing + leds
        path = save_leds(config)
        print(f"Saved {len(existing) + len(leds)} LED(s) for {cam} to {path}")

        all_leds = existing + leds
        preview = f"preview {cam} (live colour per box) - any key to close"
        cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if ok:
                cv2.imshow(preview, draw(frame, all_leds, thresholds=thresholds, debug=True))
            key = cv2.waitKey(30)
            if key != -1 or _window_closed(cv2, preview):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------- check / watch

def _check(name, seconds):
    from .monitor import PortState, check

    statuses = check(name, seconds=seconds)
    print(f"{'PORT':<14}{'STATE':<8}{'NOW':<7}{'NORMAL':<8}{'FAULT%':>7}{'LIT%':>6}")
    for s in statuses.values():
        print(f"{s.port:<14}{s.state.value:<8}{s.color.value:<7}{s.baseline.value:<8}"
              f"{s.fault_pct:>7.0f}{s.lit_pct:>6.0f}")
    bad = [s for s in statuses.values() if s.state in (PortState.FAULT, PortState.DARK)]
    print(f"{len(bad)} problem(s)" if bad else "All OK")
    return 1 if bad else 0


def _status_line(mon):
    parts = []
    for cam, ports in mon.status().items():
        online = "online" if mon.online(cam) else "OFFLINE/connecting"
        port_text = ", ".join(f"{s.port}={s.state.value}({s.color.value})" for s in ports.values())
        parts.append(f"{cam} [{online}] {port_text or 'no LEDs'}")
    return "status: " + " | ".join(parts)


def _watch(names, show, debug, log_file, hold, window, status_every):
    import time

    from .monitor import Monitor, setup_logging

    log = setup_logging(log_file or None)
    mon = Monitor(names or None, window=window, hold_seconds=hold)
    log.info("Watching %s (window %gs, hold %gs)%s. Ctrl+C to stop. Events are logged only when a state "
             "changes; a status line follows every %gs.", ", ".join(mon.cameras), window, hold,
             f", logging to {log_file}" if log_file else "", status_every)
    next_status = time.monotonic() + status_every

    def heartbeat():
        nonlocal next_status
        if status_every > 0 and time.monotonic() >= next_status:
            next_status = time.monotonic() + status_every
            log.info(_status_line(mon))

    if not (show or debug):
        mon.start()
        try:
            while mon.running:
                time.sleep(0.5)
                heartbeat()
        except KeyboardInterrupt:
            pass
        finally:
            mon.stop()
        return 0

    import cv2

    from .overlay import draw, label

    win = "LED monitor (q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mon.start()
    try:
        while True:
            heartbeat()
            status = mon.status()
            frames = []
            for cam in mon.cameras:
                f = mon.latest_frame(cam)
                frames.append(None if f is None else
                              draw(f, mon.leds(cam), status.get(cam), mon.thresholds(cam), debug))
            img = _grid(frames, mon.cameras)
            problems = mon.problems()
            text = f"{len(problems)} problem(s): " + ", ".join(f"{p.camera}/{p.port} {p.state}" for p in problems) \
                if problems else "all OK"
            label(img, text, (10, img.shape[0] - 12), (0, 140, 255) if problems else (0, 200, 0), 0.6)
            cv2.imshow(win, img)
            if _quit_pressed(cv2, win, 100)[0]:
                break
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cameras",
                                     description="Watch switch port LEDs through the IP cameras.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print all camera names")
    p_url = sub.add_parser("url", help="print the RTSP URL of a camera")
    p_url.add_argument("name")
    p_view = sub.add_parser("view", help="show live video (all cameras if no name given)")
    p_view.add_argument("names", nargs="*")
    p_snap = sub.add_parser("snapshot", help="save one frame to an image file")
    p_snap.add_argument("name")
    p_snap.add_argument("output", nargs="?", default="snapshot.jpg")

    p_tune = sub.add_parser("tune", help="live colour-mask debugger with sliders")
    p_tune.add_argument("name", nargs="?", default="webcam")
    p_cal = sub.add_parser("calibrate", help="draw a box around each port LED and learn its normal colour")
    p_cal.add_argument("name")
    p_cal.add_argument("--seconds", type=float, default=5, help="how long to sample the normal colour (default 5)")
    p_cal.add_argument("--append", action="store_true", help="keep existing boxes and add new ones")
    p_cal.add_argument("--scale", type=float, help="zoom factor for the selection window (default: auto)")
    p_cal.add_argument("--any", action="store_true",
                       help="zone mode: alert if ANY red/amber spot appears anywhere in the box")
    p_check = sub.add_parser("check", help="watch one camera briefly and print each port's state")
    p_check.add_argument("name")
    p_check.add_argument("--seconds", type=float, default=5)
    p_watch = sub.add_parser("watch", help="monitor continuously (all calibrated cameras if none given)")
    p_watch.add_argument("names", nargs="*")
    p_watch.add_argument("--show", action="store_true", help="live window with boxes coloured by state")
    p_watch.add_argument("--debug", action="store_true", help="like --show, plus a zoomed mask per box")
    p_watch.add_argument("--log", default="logs/leds.log", help="log file ('' = console only)")
    p_watch.add_argument("--hold", type=float, default=10, help="seconds a new state must last before alerting")
    p_watch.add_argument("--window", type=float, default=5, help="seconds of history per decision")
    p_watch.add_argument("--status-every", type=float, default=30,
                         help="log a status line every N seconds (0 = only on changes)")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            print("\n".join(list_cameras()))
        elif args.command == "url":
            print(get_url(args.name))
        elif args.command == "view":
            _view(args.names or list_cameras())
        elif args.command == "snapshot":
            import cv2

            cv2.imwrite(args.output, read_frame(args.name, warmup=_warmup(args.name)))
            print(f"Saved {args.output}")
        elif args.command == "tune":
            _tune(args.name)
        elif args.command == "calibrate":
            return _calibrate(args.name, args.seconds, args.append, args.scale, args.any)
        elif args.command == "check":
            return _check(args.name, args.seconds)
        elif args.command == "watch":
            return _watch(args.names, args.show, args.debug, args.log, args.hold, args.window,
                          args.status_every)
    except (KeyError, ValueError, RuntimeError, ConnectionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

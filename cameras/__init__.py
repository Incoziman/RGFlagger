"""Server-room port-LED monitor for the office IP cameras.

The cameras watch network switches. For each calibrated port LED the monitor
reports OK, FAULT (LED turned amber/red) or DARK (LED that should be lit
went off).

    import cameras

    # one-shot check (sync)
    for port, status in cameras.check("camera1").items():
        print(port, status.state, status.color)

    # continuous monitoring with a callback
    with cameras.Monitor(on_event=print) as mon:
        ...
        print(mon.status())

    # continuous monitoring, async
    async for event in cameras.Monitor().events():
        print(event)

Plain stream access is still here:

    cameras.list_cameras()               # ['camera1', ..., 'webcam']
    cap = cameras.open_camera("camera1") # cv2.VideoCapture
    frame = cameras.read_frame(1)        # single image (numpy array, BGR)

Cameras live in cameras.json next to this file. A camera may be an IP
address (RTSP) or an integer device index (e.g. "webcam": 0 for testing).
The RTSP login comes from CAMERA_USER / CAMERA_PASSWORD or a .env file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

__all__ = [
    "list_cameras", "get_url", "open_camera", "read_frame", "is_local", "CONFIG_FILE", "LEDS_FILE",
    # LED monitoring (imported lazily, see __getattr__ below)
    "Monitor", "check", "Event", "PortStatus", "PortState", "LedColor", "LedReading",
    "classify_led", "classify_any", "classify", "color_masks", "load_leds", "save_leds", "setup_logging",
]

CONFIG_FILE = Path(__file__).with_name("cameras.json")
LEDS_FILE = Path(__file__).with_name("leds.json")
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Give up on an unreachable / stalled RTSP camera after this many ms.
_TIMEOUT_MS = 8000


def _load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def _read_env_file() -> dict:
    """Read KEY=VALUE lines from .env in the current folder or the repo root."""
    for folder in (Path.cwd(), _REPO_ROOT):
        env_file = folder / ".env"
        if env_file.is_file():
            values = {}
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
            return values
    return {}


def _credentials() -> tuple[str, str]:
    env_file = _read_env_file()
    user = os.environ.get("CAMERA_USER") or env_file.get("CAMERA_USER") or "root"
    password = os.environ.get("CAMERA_PASSWORD") or env_file.get("CAMERA_PASSWORD")
    if not password:
        raise RuntimeError(
            "No camera password found. Either set the environment variable "
            "CAMERA_PASSWORD (Windows: setx CAMERA_PASSWORD yourpassword, then "
            "open a new terminal) or create a .env file containing "
            "CAMERA_PASSWORD=yourpassword (see .env.example)."
        )
    return user, password


def _resolve_name(name: str | int) -> str:
    cameras = _load_config()["cameras"]
    key = f"camera{name}" if isinstance(name, int) or str(name).isdigit() else str(name)
    if key not in cameras:
        raise KeyError(f"Unknown camera {name!r}. Available: {', '.join(cameras)}")
    return key


def list_cameras() -> list[str]:
    """Names of all configured cameras (including local test devices)."""
    return list(_load_config()["cameras"])


def is_local(name: str | int) -> bool:
    """True if the camera is a local device (e.g. the laptop webcam), not RTSP."""
    return isinstance(_load_config()["cameras"][_resolve_name(name)], int)


def get_url(name: str | int) -> str:
    """Full RTSP URL (including login) for a camera, e.g. get_url("camera1") or get_url(1)."""
    config = _load_config()
    key = _resolve_name(name)
    host = config["cameras"][key]
    if isinstance(host, int):
        raise ValueError(f"{key} is a local device (index {host}); it has no RTSP URL.")
    user, password = _credentials()
    return (
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{config['port']}{config['path']}"
    )


def open_camera(name: str | int):
    """Open a camera and return a cv2.VideoCapture.

    Usage:
        cap = cameras.open_camera("camera1")   # or "webcam"
        ok, frame = cap.read()
        cap.release()
    """
    # TCP is more reliable than UDP for RTSP (no torn/grey frames).
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2

    key = _resolve_name(name)
    source = _load_config()["cameras"][key]
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        hint = f"Is device {source} connected and not in use by another app?"
    else:
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, _TIMEOUT_MS]
        cap = cv2.VideoCapture(get_url(key), cv2.CAP_FFMPEG, params)
        hint = "Check the network connection and the password."
    if not cap.isOpened():
        cap.release()
        raise ConnectionError(f"Could not open {key}. {hint}")
    return cap


def read_frame(name: str | int, warmup: int = 0):
    """Grab a single frame (numpy array, BGR) from a camera.

    warmup: frames to skip first (webcams need a few to settle their exposure).
    """
    cap = open_camera(name)
    try:
        for _ in range(warmup):
            cap.read()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ConnectionError(f"Connected to {_resolve_name(name)} but could not read a frame.")
    return frame


# The LED-monitoring API is imported on first use, so `python -m cameras url`
# and friends stay fast and don't load numpy/OpenCV.
_LAZY = {
    "Monitor": "monitor", "check": "monitor", "Event": "monitor", "PortStatus": "monitor",
    "PortState": "monitor", "load_leds": "monitor", "save_leds": "monitor",
    "setup_logging": "monitor",
    "LedColor": "detect", "LedReading": "detect", "classify_led": "detect", "classify_any": "detect",
    "classify": "detect", "color_masks": "detect",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

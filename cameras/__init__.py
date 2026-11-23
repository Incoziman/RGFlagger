"""Easy access to the office IP cameras.

    import cameras
    cameras.list_cameras()               # ['camera1', 'camera2', ...]
    url = cameras.get_url("camera1")     # rtsp://... (use with VLC, ffmpeg, etc.)
    cap = cameras.open_camera("camera1") # cv2.VideoCapture
    frame = cameras.read_frame(1)        # single image (numpy array)

The camera list lives in cameras.json next to this file.
The login comes from the environment variables CAMERA_USER / CAMERA_PASSWORD,
or from a .env file (see .env.example).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

__all__ = ["list_cameras", "get_url", "open_camera", "read_frame", "CONFIG_FILE"]

CONFIG_FILE = Path(__file__).with_name("cameras.json")
_REPO_ROOT = Path(__file__).resolve().parent.parent


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
    """Names of all configured cameras."""
    return list(_load_config()["cameras"])


def get_url(name: str | int) -> str:
    """Full RTSP URL (including login) for a camera, e.g. get_url("camera1") or get_url(1)."""
    config = _load_config()
    host = config["cameras"][_resolve_name(name)]
    user, password = _credentials()
    return (
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{config['port']}{config['path']}"
    )


def open_camera(name: str | int):
    """Open a camera stream and return a cv2.VideoCapture.

    Usage:
        cap = cameras.open_camera("camera1")
        ok, frame = cap.read()
        cap.release()
    """
    # TCP is more reliable than UDP for RTSP (no torn/grey frames).
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    import cv2

    cap = cv2.VideoCapture(get_url(name), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ConnectionError(
            f"Could not open {_resolve_name(name)}. Check the network connection "
            f"and the password."
        )
    return cap


def read_frame(name: str | int):
    """Grab a single frame (numpy array, BGR) from a camera."""
    cap = open_camera(name)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ConnectionError(f"Connected to {_resolve_name(name)} but could not read a frame.")
    return frame

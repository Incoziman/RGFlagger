# cameras: server-room port-LED monitor

IP cameras watch the network switches. This package spots connection problems from the Ethernet port LEDs:

| State   | Meaning |
|---------|---------|
| `OK`    | LED shows its normal colour (blinking activity is fine) |
| `FAULT` | LED is **amber/red** for most of the last few seconds |
| `DARK`  | LED was lit when calibrated but is now **off**. The link is probably lost. |

Only the boxes you draw around each port LED are analysed, so red labels, cables and reflections elsewhere in the picture are ignored. Each port is judged over a rolling window (5 s), and a new state must last a hold time (10 s) before an alert fires, so normal blinking doesn't cause alarms.

| Name    | Source        |
|---------|---------------|
| camera1 | 192.168.50.91 |
| camera2 | 192.168.50.92 |
| camera3 | 192.168.50.93 |
| camera4 | 192.168.50.94 |
| webcam  | local device 0 (laptop webcam, for testing) |

The camera list is in [`cameras/cameras.json`](cameras/cameras.json). A value can be an IP address (RTSP) or an integer device index.

## 1. Install

```
pip install -e .            # or: pip install git+<repo-url>
pip install -e .[test]      # with pytest
```

## 2. Set the camera password (once per PC)

The password is **not** stored in this repository. Do one of the following:

- `setx CAMERA_PASSWORD yourpassword` (Windows), then open a new terminal.
- Copy `.env.example` to `.env` and fill it in.

`CAMERA_USER` defaults to `root`. The webcam needs no password.

## 3. Try it on your laptop webcam first

Use a phone showing a full-screen red, orange or green image, or any coloured LED or flashlight.

```
python -m cameras tune webcam          # see what the colour filter sees, adjust sliders
python -m cameras calibrate webcam     # draw a box around the "LED"
python -m cameras watch webcam --debug # live view: change colour -> FAULT, cover it -> DARK
```

## 4. Set up a real camera

1. **Tune** (optional): `python -m cameras tune camera1`.
   The window shows the frame with boxes, plus what the RED, AMBER and GREEN masks pick up. The sliders change the brightness, saturation and hue limits live. Press `s` to save them for that camera, `r` to reset and `q` to quit.
2. **Calibrate**: `python -m cameras calibrate camera1`.
   Drag a tight box around one port LED and press ENTER. Repeat for each LED, then press ESC. Name each box when asked, e.g. `sw1-p1`. The tool then watches for 5 s to learn each port's **normal** colour, so **make sure everything is healthy while it does that**.
   - Ports that are off during calibration count as unused and never raise `DARK`.
   - A port that is normally amber (e.g. a 100 Mb link) only raises `FAULT` on red.
   - Use `--append` to add boxes without removing the existing ones.
   - **Zone mode** (`--any`): draw one big box around a whole switch or panel. Normal boxes go by the colour most of the box shows. A zone alerts as soon as **any** red/amber spot of at least `min_spot_px` pixels (default 6) appears anywhere inside it, so one red LED among 20 green ones still fires. Zones never raise `DARK`. In the live view, every red/amber spot is circled. Make sure no red/orange labels or cables are inside the zone; calibration warns you if it sees any. You can mix both kinds of box: `calibrate camera1 --any --append`.

   Everything is saved in [`cameras/leds.json`](cameras/leds.json).
3. **Check**: `python -m cameras check camera1` prints a table. The exit code is 1 if any port is `FAULT`/`DARK`, so it can be used from scripts or Task Scheduler.
4. **Watch**: run it continuously:
   ```
   python -m cameras watch                    # all calibrated cameras, console + logs/leds.log
   python -m cameras watch camera1 --show     # plus live window, boxes coloured by state
   python -m cameras watch --debug            # plus a zoomed colour mask next to each box
   python -m cameras watch --hold 30 --log "" # slower alerts, console only
   ```
   Box colours: green = OK, orange = FAULT, grey = DARK, blue = not decided yet.

**Tip:** bright LEDs often show up as white blobs on camera. If calibration reports `OTHER`, lower the camera's exposure or turn off auto-exposure in its web settings. Darker images give much cleaner LED colours.

## 5. Use it from your own Python code

```python
import cameras

# One-shot (blocking): watch for 5 s, then return each port's state
for port, s in cameras.check("camera1").items():
    print(port, s.state, s.color)          # e.g. sw1-p1 OK GREEN

# Continuous, with a callback (called from a background thread)
def alert(event):
    if event.is_problem:
        print(event.message)               # "camera1/sw1-p3: LED went dark - link lost? (OK -> DARK)"
        # event.snapshot is a JPEG (bytes) of the annotated frame, handy for alerts

with cameras.Monitor(on_event=alert) as mon:   # all calibrated cameras
    ...                                        # your program keeps running
    print(mon.status())                        # {camera: {port: PortStatus}}, any time
    print(mon.problems())                      # just the FAULT / DARK ports

# Continuous, asyncio
import asyncio

async def main():
    async for event in cameras.Monitor(["camera1"]).events():
        print(event)

asyncio.run(main())

# Logging to console + rotating file, same as the CLI
cameras.setup_logging("logs/leds.log")
```

`Monitor` options: `cameras=[...]`, `window=5`, `hold_seconds=10`, `fps=5`, `fault_ratio=0.6`, `dark_ratio=0.1`, `snapshots=True`. It reconnects to cameras that drop and reports `CAMERA_OFFLINE` / `CAMERA_ONLINE` events.

Lower-level pieces: `cameras.classify_led(bgr_crop)` (majority colour) and `cameras.classify_any(bgr_crop)` (any red/amber spot) → `OFF/GREEN/AMBER/RED/OTHER`, plus `cameras.color_masks(bgr)`.

## 6. Plain stream access

```
python -m cameras list                      # camera names
python -m cameras url camera1               # print RTSP URL
python -m cameras view                      # live grid of all cameras (q = quit)
python -m cameras snapshot camera1 pic.jpg  # save one frame
```

```python
cap = cameras.open_camera("camera1")   # cv2.VideoCapture (or open_camera(1), open_camera("webcam"))
frame = cameras.read_frame("camera2")  # one BGR numpy image
```

RTSP URL format, for other tools: `rtsp://<user>:<password>@<ip>:554/mpeg4/media.amp`. Each camera can only serve a few viewers at once, so close streams when you're done.

## Tests

```
python -m pytest
```
The tests use synthetic LED images and a fake camera, so no hardware is needed.

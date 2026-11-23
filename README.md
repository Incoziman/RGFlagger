# cameras

Easy access to the 4 IP cameras from Python or any other language.

| Name    | IP            |
|---------|---------------|
| camera1 | 192.168.50.91 |
| camera2 | 192.168.50.92 |
| camera3 | 192.168.50.93 |
| camera4 | 192.168.50.94 |

The list lives in [`cameras/cameras.json`](cameras/cameras.json). To add or change a camera, edit that file.

## 1. Install

```
pip install git+<repo-url>
```

Or, from a local copy of this folder:

```
pip install -e .
```

## 2. Set the camera password (once per PC)

The password is **not** stored in this repository. Ask a colleague for it, then do one of the following:

- Set an environment variable (Windows). Open a new terminal afterwards:
  ```
  setx CAMERA_PASSWORD yourpassword
  ```
- Or copy `.env.example` to `.env` in your project folder and fill in the password.

`CAMERA_USER` is optional. It defaults to `root`.

## 3. Use it from Python

```python
import cameras

print(cameras.list_cameras())          # ['camera1', 'camera2', 'camera3', 'camera4']

cap = cameras.open_camera("camera1")   # a cv2.VideoCapture, or use open_camera(1)
while True:
    ok, frame = cap.read()             # frame is a numpy array (BGR)
    if not ok:
        break
    # ... do something with frame ...
cap.release()

frame = cameras.read_frame("camera2")  # just one image
url = cameras.get_url("camera3")       # the raw rtsp:// URL
```

## 4. Command line

```
python -m cameras list                      # camera names
python -m cameras url camera1               # print RTSP URL
python -m cameras view                      # live 2x2 grid of all cameras (q = quit)
python -m cameras view camera1              # one camera
python -m cameras snapshot camera1 pic.jpg  # save one frame
```

## 5. Web apps (React etc.)

Browsers can't play `rtsp://` streams, and the camera password must never be in frontend code. So one PC on the camera network runs a small **relay** (MediaMTX). The relay reads the cameras and serves them in a format browsers can play.

**On the relay PC (once):**

1. Clone this repo and do steps 1 and 2 above.
2. Download `mediamtx_..._windows_amd64.zip` from https://github.com/bluenviron/mediamtx/releases and unzip it into `relay/mediamtx/`.
3. Check the video codec: `python -m cameras info camera1`
4. Start the relay:
   ```
   python -m cameras relay              # if info said H.264
   python -m cameras relay --transcode  # otherwise (needs ffmpeg: winget install ffmpeg)
   ```
   Allow it through the Windows firewall when asked. It uses TCP ports 8889 and 8888 and UDP port 8189.

Each camera is now available, with no password, at:

| What                     | URL                                         |
|--------------------------|---------------------------------------------|
| Web page (WebRTC, ~0.5s delay) | `http://<relay-pc>:8889/camera1`      |
| HLS (for `<video>` + hls.js)   | `http://<relay-pc>:8888/camera1/index.m3u8` |
| RTSP (no password)             | `rtsp://<relay-pc>:8554/camera1`      |

Open the first URL in a browser to test it.

**In React:** copy [`web/CameraStream.jsx`](web/CameraStream.jsx) into your app and set `RELAY_URL` to the relay PC:

```jsx
import { CameraStream, CameraGrid } from "./CameraStream";

<CameraStream name="camera1" />   // one camera
<CameraGrid />                    // all four in a 2x2 grid
```

The relay connects to a camera only while someone is watching it. However many people are watching, each camera gets only one connection.

## 6. Other languages

Every camera URL has this form:

```
rtsp://<CAMERA_USER>:<CAMERA_PASSWORD>@<ip>:554/mpeg4/media.amp
```

Read the IPs from `cameras/cameras.json` and the login from the environment variables. Alternatively, run `python -m cameras url camera1` and use its output.

**C# (OpenCvSharp)**

```csharp
var pw = Environment.GetEnvironmentVariable("CAMERA_PASSWORD");
var cap = new VideoCapture($"rtsp://root:{pw}@192.168.50.91:554/mpeg4/media.amp");
```

**JavaScript / Node (via ffmpeg)**

```js
const cfg = require("./cameras/cameras.json");
const url = `rtsp://root:${process.env.CAMERA_PASSWORD}@${cfg.cameras.camera1}:${cfg.port}${cfg.path}`;
```

**VLC / ffplay**

```
vlc  rtsp://root:<password>@192.168.50.91:554/mpeg4/media.amp
ffplay -rtsp_transport tcp rtsp://root:<password>@192.168.50.91:554/mpeg4/media.amp
```

## Note

Each camera can only serve a few viewers at the same time. Close streams (`cap.release()`) when you are done. If many people or apps need the streams at once, have everyone use the relay URLs (section 5) instead of connecting to the cameras directly.

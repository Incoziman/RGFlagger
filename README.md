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

## 5. Other languages

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

Each camera can only serve a few viewers at the same time. Close streams (`cap.release()`) when you are done. If too many people need the streams at once, a relay server such as MediaMTX can be added later. The way you use this library would stay the same.

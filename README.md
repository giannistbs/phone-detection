# Phone Detection

![CI](https://github.com/giannistbs/phone-detection/actions/workflows/docker.yml/badge.svg)

Real-time driver phone detection using a Roboflow model and OpenCV.

## Prerequisites

Create a `.env` file in this directory:

```
ROBOFLOW_API_KEY=your_key_here
```

---

## Option 1 — Docker Compose (headless, no window)

Best for servers, background processes, or running against an RTSP camera stream.
The container always runs with `--headless`, so no display is needed.

### RTSP stream

Add `RTSP_URL` to your `.env`:

```
ROBOFLOW_API_KEY=your_key_here
RTSP_URL=rtsp://user:pass@192.168.1.100/stream
```

Then build and run:

```bash
docker compose up --build
```

### Webcam

In `docker-compose.yml`, swap the `command` and uncomment `devices`:

```yaml
command: ["--source", "0"]
devices:
  - /dev/video0:/dev/video0
```

Then:

```bash
docker compose up --build
```

> Note: webcam passthrough requires Linux. On macOS, Docker Desktop does not
> expose USB/webcam devices to containers — use the native approach below instead.

---

## Option 2 — Native macOS (with live window)

Run the script directly in a virtual environment to get the `cv2.imshow` preview window.

### Setup (first time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Webcam

```bash
source .venv/bin/activate
python src/detect.py --source 0
```

Press `q` in the video window to quit.

### RTSP stream

```bash
source .venv/bin/activate
python src/detect.py --rtsp rtsp://user:pass@192.168.1.100/stream
```

### Headless (no window) — native

```bash
python src/detect.py --source 0 --headless
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index or path to a video file |
| `--rtsp` | — | RTSP URL; overrides `--source` |
| `--conf` | `0.5` | Confidence threshold (0–1) |
| `--headless` | off | Disable the preview window |

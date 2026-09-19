# ShortsAI

ShortsAI is a local-first video editor and deterministic FFmpeg rendering engine designed for crafting YouTube Shorts. 

The repository provides a PC-based FastAPI development server. You edit your video using a responsive HTML5 web interface (which provides a lightweight native video preview) on your main device, while the FFmpeg processing happens locally on your rendering machine over Wi-Fi.

## Key Features

* **Lightweight Native Preview:** The frontend utilizes an optimized HTML5 `<video>` element and a CSS overlay for cropping, ensuring smooth scrubbing without heavy canvas rendering.
* **Millisecond Precision Timeline:** Control Start, Cut, and End timestamps down to the millisecond (`MM:SS.mmm`).
* **Fixed Crop Architecture:** Deterministic square crop targeting 1080x1920 outputs.
* **Emoji Typography Support:** Built-in Calistoga font rendering with emoji wrapping via `pilmoji` (which dynamically fetches Apple-style emoji image assets from a CDN at runtime).
* **Local Processing:** Local rendering over Wi-Fi. No video data is uploaded to external cloud services.

## Video Modes

### 1. Main Video
* Select a continuous clip (Start → End).
* Apply a fixed square crop.
* A single, bold Black **Main Caption** overlaid on a clean white background.
* Optional emoji alignment.

### 2. Curiosity Video
* **Hook Segment (Start → Cut):** Features a "Main Caption" where text before the first comma is Black, and text after is Red. 
* **Reveal Segment (Cut → End):** Cross-fades into the remaining footage featuring a bold Black "Curiosity Caption" (the reveal).
* Preserves independent emojis for both captions.

---

## Requirements

* **Network:** Both the editing device and the rendering device must be on the same local Wi-Fi network.
* **Dependencies:** Python 3.10+, FFmpeg, and FFprobe.

---

## Setup: PC Development Environment

This is the primary environment supported natively by the repository.

### 1. Clone the Repository
```bash
git clone https://github.com/sahilsleem/shortsAi.git
cd shortsAi
```

### 2. FFmpeg Installation
* **Windows:** Download FFmpeg from a reputable source, extract it, and add the `bin` folder to your Environment Variables PATH.

### 3. Python Dependencies
Install the required packages from the repository:
```bash
pip install -r requirements.txt
```
*(This installs `fastapi`, `uvicorn`, `pillow`, `emoji`, `pilmoji`, and `python-multipart`).*

### 4. Font Setup
ShortsAI requires the `Calistoga-Regular.ttf` font to correctly render captions. You must procure this font yourself and ensure you respect its licensing terms. Place `Calistoga-Regular.ttf` in the `fonts/` directory (e.g., `fonts/Calistoga-Regular.ttf`).

### 5. Starting the PC Server
Start the FastAPI application using Uvicorn:
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

---

## Setup: Custom Android / Termux Render Server

While the repository provides the PC FastAPI server, it is possible to run the rendering pipeline on a dedicated Android device (like a OnePlus Nord) via Termux using a custom standalone standard-library script (e.g., `shortsai_server.py`). *Note: This custom wrapper script is not bundled in this repository, but this documents the environment we successfully use.*

### 1. Termux Preparation
Install Termux on the rendering Android device and run:
```bash
pkg update
pkg install python ffmpeg
pip install pillow emoji pilmoji
```

### 2. File Placement
Create a directory `~/shortsai/` in Termux. You must place the following files into this directory:
* Your custom `shortsai_server.py` standalone script.
* `index.html` (The frontend UI, copied from `static/index.html`).
* `Calistoga-Regular.ttf` (The required font).

### 3. Creating Termux:Widget Shortcuts
To start the server from your Android home screen without opening the terminal, ensuring the device doesn't sleep during rendering:
1. Install the **Termux:Widget** app.
2. Create a `~/.shortcuts/` directory in Termux.
3. Create `Start ShortsAI.sh`:
   ```bash
   #!/bin/bash
   termux-wake-lock
   cd ~/shortsai
   python shortsai_server.py
   ```
4. Create `Stop ShortsAI.sh`:
   ```bash
   #!/bin/bash
   pkill -f shortsai_server.py
   termux-wake-unlock
   ```
5. Make both executable (`chmod +x ~/.shortcuts/*.sh`). Add the Termux Widget to your home screen.

---

## Usage Instructions

1. **Find the Server IP:** On your Android rendering device, find its local Wi-Fi IP address (e.g., `192.168.1.82`) reliably by navigating to **Settings -> Network & internet -> Wi-Fi -> Network details**.
2. **Connect:** Open the browser on your main phone/device and navigate to `http://<SERVER_IP>:8000/`.
3. **Upload & Edit:** Upload a video, set your crops and captions, and select your timestamps.
4. **Render:** Tap "Render Video". The FFmpeg processing happens entirely on the server device.
5. **Download:** The finished MP4 will be served back to your browser.

---

## Output Specifications
* **Resolution:** 1080x1920 (9:16)
* **Crop Area:** 1080x1080 (1:1 Square) centered dynamically at `Y=420`
* **Framerate:** 30 FPS
* **Codec:** H.264 (`libx264`, preset `medium`, CRF `18`)
* **Audio:** AAC 192k, 44.1kHz
* **Color Grading:** Subtly enhanced via SDR FFmpeg `eq` filters (Contrast 1.03, Saturation 1.08, Gamma 1.02) and unsharp masking.

---

## Project Structure
* `src/` - Core Python backend modules (FastAPI PC version).
* `static/index.html` - The HTML5 frontend web application.
* `fonts/` - Expected directory for `Calistoga-Regular.ttf`.

---

## Troubleshooting

* **Server crashes on start:** Ensure all packages in `requirements.txt` are installed.
* **HTTP 500 Plain-Text Errors:** Indicates an FFmpeg subprocess failure. The server validates and clamps crop dimensions against the video's actual `ffprobe` bounds to prevent out-of-bounds crashes, but extreme values or missing dependencies can still cause failures.
* **Font Issues:** Ensure `Calistoga-Regular.ttf` is placed exactly in `fonts/` for the PC server, or `~/shortsai/` for the standalone Termux server.

---

## Privacy
ShortsAI processes your video files completely locally. All original uploads and rendered videos stay on your local rendering device and local Wi-Fi network; ShortsAI does not upload your videos to any cloud video-processing service.

However, note that the current emoji rendering implementation (`pilmoji`) may make external network requests to retrieve emoji image assets from its configured source CDN at runtime. Users who require completely offline or air-gapped operation must manually provide and configure appropriate local emoji assets and dependencies.

## License
MIT License. Please note that third-party fonts (like Calistoga) or emoji packs used by `pilmoji` are subject to their own respective licenses. Ensure you have the rights to use any assets you provide to the renderer.

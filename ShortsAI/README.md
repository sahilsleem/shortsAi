# ShortsAI - Milestone 1

Deterministic video renderer for YouTube Shorts.

## Setup
1. Ensure Python 3.10+ is installed.
2. Install FFmpeg manually and add it to your System PATH:
   - Download from https://gyan.dev/ffmpeg/builds/
   - Extract and add the `bin` folder to your Environment Variables PATH.
3. Install Python dependencies:
   ```bash
   pip install Pillow pilmoji
   ```
4. Place the required font `Calistoga-Regular.ttf` in the `fonts/` directory.

## Usage
Run the renderer by pointing it to a configuration JSON file. You can also override input/output paths via command line.

```bash
python -m src.render --config job.json
```

Or with overrides:
```bash
python -m src.render --input input/my_vid.mp4 --output output/done.mp4 --config job.json
```

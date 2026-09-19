import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import CaptionData
from src.image_ops import generate_text_overlay, compute_best_font_size
from src.video_ops import render_main_video, render_curiosity_video, get_ffprobe_path

app = FastAPI(title="ShortsAI Minimal Backend")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

MAX_UPLOAD_SIZE = 100 * 1024 * 1024

def cleanup_workspace(workspace_dir: str):
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir, ignore_errors=True)

def get_duration(video_path: str) -> float:
    cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        if duration <= 0:
            for stream in data.get("streams", []):
                if stream.get("duration"):
                    duration = float(stream["duration"])
                    break
        return duration
    except Exception:
        return 0.0

@app.post("/render")
async def render_short(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    cut_time: Optional[float] = Form(None),
    crop_x: int = Form(...),
    crop_y: int = Form(...),
    crop_size: int = Form(...),
    caption_main: str = Form(...),
    caption_curiosity: str = Form(""),
    caption_main_emoji: str = Form(""),
    caption_curiosity_emoji: str = Form("")
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")

    req_id = str(uuid.uuid4())
    workspace = Path(f"working/{req_id}").resolve()
    os.makedirs(workspace, exist_ok=True)
    background_tasks.add_task(cleanup_workspace, str(workspace))
    
    input_path = workspace / "input.mp4"
    output_path = workspace / "output.mp4"
    
    try:
        file_size = 0
        with open(input_path, "wb") as f:
            while chunk := await video.read(8192):
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="Video too large")
                f.write(chunk)
                
        actual_dur = get_duration(str(input_path))
        if actual_dur <= 0:
            raise HTTPException(status_code=400, detail="Invalid video duration")
            
        start_time = max(0.0, min(start_time, actual_dur))
        end_time = max(start_time + 0.5, min(end_time, actual_dur))
                
        font_path = str(Path("fonts/Calistoga-Regular.ttf").resolve())
        
        if mode == "main":
            cap = CaptionData(main_text=caption_main, curiosity_text="", emoji=caption_main_emoji)
            fs = compute_best_font_size([cap], font_path, 880, 3)
            overlay_path = workspace / "overlay.png"
            generate_text_overlay(cap, font_path, fs, str(overlay_path))
            
            render_main_video(
                str(input_path), str(output_path), str(overlay_path),
                start_time, end_time, crop_x, crop_y, crop_size
            )
        else:
            if "," in caption_main:
                split_idx = caption_main.index(",") + 1
                hook_black = caption_main[:split_idx].strip()
                hook_red = caption_main[split_idx:].strip()
            else:
                hook_black = caption_main.strip()
                hook_red = ""
                
            cap_hook = CaptionData(main_text=hook_black, curiosity_text=hook_red, emoji=caption_main_emoji)
            # Curiosity Caption / Reveal = BLACK (mapped to main_text)
            cap_reveal = CaptionData(main_text=caption_curiosity, curiosity_text="", emoji=caption_curiosity_emoji)
            fs = compute_best_font_size([cap_hook, cap_reveal], font_path, 880, 3)
            
            hook_overlay = workspace / "hook.png"
            reveal_overlay = workspace / "reveal.png"
            generate_text_overlay(cap_hook, font_path, fs, str(hook_overlay))
            generate_text_overlay(cap_reveal, font_path, fs, str(reveal_overlay))
            
            # Validate and clamp cut_time
            if cut_time is None:
                cut_time = start_time + (end_time - start_time) / 2.0
            else:
                cut_time = max(start_time + 0.001, min(cut_time, end_time - 0.001))
            
            render_curiosity_video(
                str(input_path), str(output_path), str(hook_overlay), str(reveal_overlay),
                start_time, cut_time, end_time, crop_x, crop_y, crop_size
            )
            
    except Exception as e:
        cleanup_workspace(str(workspace))
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"shortsai_{mode}_{req_id[:8]}.mp4"
    )

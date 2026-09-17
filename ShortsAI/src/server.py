import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from src.config import CaptionData
from src.image_ops import generate_text_overlay, compute_best_font_size
from src.video_ops import render_video, get_ffprobe_path

app = FastAPI(title="ShortsAI Minimal Backend")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

def cleanup_workspace(workspace_dir: str):
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir, ignore_errors=True)

def validate_and_get_duration(video_path: str) -> float:
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail="Invalid video file: FFprobe failed to inspect")
        
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid video file: Unreadable FFprobe output")
        
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
            
    if not video_stream:
        raise HTTPException(status_code=400, detail="Invalid video file: No video stream found")
        
    # Get duration
    duration = None
    if "duration" in data.get("format", {}):
        duration = float(data["format"]["duration"])
    elif video_stream.get("duration"):
        duration = float(video_stream["duration"])
        
    if duration is None or duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid video file: Could not determine valid duration")
        
    return duration

@app.post("/render")
async def render_short(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    hook_main_text: str = Form(...),
    hook_start: float = Form(...),
    hook_end: float = Form(...),
    reveal_main_text: str = Form(...),
    reveal_start: float = Form(...),
    reveal_end: float = Form(...),
    hook_curiosity_text: str = Form(""),
    hook_emoji: str = Form(""),
    reveal_curiosity_text: str = Form(""),
    reveal_emoji: str = Form("")
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
        
    if hook_start < 0 or reveal_start < 0:
        raise HTTPException(status_code=400, detail="Timings cannot be negative")
        
    if hook_start >= hook_end:
        raise HTTPException(status_code=400, detail="hook_start must be less than hook_end")
        
    if reveal_start >= reveal_end:
        raise HTTPException(status_code=400, detail="reveal_start must be less than reveal_end")

    req_id = str(uuid.uuid4())
    workspace = Path(f"working/{req_id}").resolve()
    os.makedirs(workspace, exist_ok=True)
    
    # Schedule cleanup for successful requests
    background_tasks.add_task(cleanup_workspace, str(workspace))
    
    input_path = workspace / "input.mp4"
    output_path = workspace / "output.mp4"
    hook_overlay_path = workspace / "hook_overlay.png"
    reveal_overlay_path = workspace / "reveal_overlay.png"
    
    try:
        # Bounded upload reading
        file_size = 0
        with open(input_path, "wb") as f:
            while chunk := await video.read(8192):
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="Video exceeds 50MB limit")
                f.write(chunk)
                
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        duration = validate_and_get_duration(str(input_path))
        
        if hook_end > duration or reveal_end > duration:
            raise HTTPException(status_code=400, detail="Timings exceed video duration")
            
        font_path = str(Path("fonts/Calistoga-Regular.ttf").resolve())
        if not os.path.exists(font_path):
            raise HTTPException(status_code=500, detail="Server error: Default font missing")
            
        hook_caption = CaptionData(
            main_text=hook_main_text,
            curiosity_text=hook_curiosity_text,
            emoji=hook_emoji
        )
        
        reveal_caption = CaptionData(
            main_text=reveal_main_text,
            curiosity_text=reveal_curiosity_text,
            emoji=reveal_emoji
        )
        
        best_font_size = compute_best_font_size(
            captions=[hook_caption, reveal_caption],
            font_path=font_path,
            max_width=880,
            max_lines=2
        )
        
        generate_text_overlay(
            caption=hook_caption,
            font_path=font_path,
            font_size=best_font_size,
            output_path=str(hook_overlay_path)
        )
        
        generate_text_overlay(
            caption=reveal_caption,
            font_path=font_path,
            font_size=best_font_size,
            output_path=str(reveal_overlay_path)
        )
        
        render_video(
            source_video=str(input_path),
            output_video=str(output_path),
            hook_overlay=str(hook_overlay_path),
            reveal_overlay=str(reveal_overlay_path),
            hook_start=hook_start,
            hook_end=hook_end,
            reveal_start=reveal_start,
            reveal_end=reveal_end
        )
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Render failed: Output file missing")
            
    except Exception as e:
        # Cleanup synchronously since BackgroundTasks won't run on HTTPException
        cleanup_workspace(str(workspace))
        if isinstance(e, HTTPException):
            raise e
        
        error_msg = str(e)
        if "Validation failed" in error_msg:
            raise HTTPException(status_code=400, detail="Render output validation failed.")
        raise HTTPException(status_code=500, detail="Render failed due to an internal error.")

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"shortsai_{req_id[:8]}.mp4"
    )

import subprocess
import os
import shutil
from pathlib import Path

def get_ffmpeg_path() -> str:
    # 1. Environment variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    # 2. shutil.which
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path
        
    # 3. Verified Windows fallback
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(fallback):
        return fallback
        
    # Default to "ffmpeg" and hope for the best
    return "ffmpeg"

def get_ffprobe_path() -> str:
    # 1. Environment variable
    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    # 2. shutil.which
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path
        
    # 3. Verified Windows fallback
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"
    if os.path.exists(fallback):
        return fallback
        
    # Default to "ffprobe"
    return "ffprobe"

def build_ffmpeg_command(
    source_video: str,
    output_video: str,
    hook_overlay: str,
    reveal_overlay: str,
    hook_start: float,
    hook_end: float,
    reveal_start: float,
    reveal_end: float,
    transition_dur: float = 0.15
):
    hook_dur = hook_end - hook_start
    rev_dur = reveal_end - reveal_start
    
    video_filters = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,eq=contrast=1.05:saturation=1.1,unsharp=5:5:0.5:5:5:0.0"
    
    cmd = [get_ffmpeg_path(), "-y"]
    
    # Input 0: Hook video
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    
    # Input 1: Reveal video
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    
    # Input 2: Hook overlay
    cmd.extend(["-i", hook_overlay])
    
    # Input 3: Reveal overlay
    cmd.extend(["-i", reveal_overlay])
    
    fc = []
    
    # Hook segment (timestamps are reset to 0 because of -ss)
    fc.append(f"[0:v]setpts=PTS-STARTPTS,{video_filters}[hook_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={hook_dur}:r=30[hook_base];")
    fc.append(f"[hook_base][hook_v_proc]overlay=0:420:eof_action=pass[hook_bg1];")
    fc.append(f"[hook_bg1][2:v]overlay=0:0[hook_bg2];")
    fc.append(f"[hook_bg2]fade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}:c=black[hook_v_final];")
    
    fc.append(f"[0:a]asetpts=PTS-STARTPTS,afade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}[hook_a_final];")
    
    # Reveal segment
    fc.append(f"[1:v]setpts=PTS-STARTPTS,{video_filters}[rev_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={rev_dur}:r=30[rev_base];")
    fc.append(f"[rev_base][rev_v_proc]overlay=0:420:eof_action=pass[rev_bg1];")
    fc.append(f"[rev_bg1][3:v]overlay=0:0[rev_bg2];")
    fc.append(f"[rev_bg2]fade=t=in:st=0:d={transition_dur}:c=black[rev_v_final];")
    
    fc.append(f"[1:a]asetpts=PTS-STARTPTS,afade=t=in:st=0:d={transition_dur}[rev_a_final];")
    
    fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[outv][outa]")
    
    filter_complex = "".join(fc)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-maxrate", "12M",
        "-bufsize", "24M",
        "-r", "30",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        output_video
    ])
    
    return cmd

def render_video(
    source_video: str,
    output_video: str,
    hook_overlay: str,
    reveal_overlay: str,
    hook_start: float,
    hook_end: float,
    reveal_start: float,
    reveal_end: float
):
    cmd = build_ffmpeg_command(
        source_video=source_video,
        output_video=output_video,
        hook_overlay=hook_overlay,
        reveal_overlay=reveal_overlay,
        hook_start=hook_start,
        hook_end=hook_end,
        reveal_start=reveal_start,
        reveal_end=reveal_end
    )
    
    print("Running FFmpeg with command:")
    try:
        print(" ".join(cmd))
    except UnicodeEncodeError:
        print(" ".join(cmd).encode('ascii', 'replace').decode('ascii'))
    
    subprocess.run(cmd, check=True)

import json

def validate_render(output_video: str):
    print("Validating output video...")
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        output_video
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    video_stream = None
    audio_stream = None
    
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
        elif stream.get("codec_type") == "audio":
            audio_stream = stream
            
    if not video_stream:
        raise ValueError("Validation failed: No video stream found.")
        
    width = video_stream.get("width")
    height = video_stream.get("height")
    codec = video_stream.get("codec_name")
    
    fps_str = video_stream.get("r_frame_rate", "0/1")
    
    if width != 1080 or height != 1920:
        raise ValueError(f"Validation failed: Invalid resolution {width}x{height}, expected 1080x1920")
        
    if codec != "h264":
        raise ValueError(f"Validation failed: Invalid codec {codec}, expected h264")
        
    parts = fps_str.split('/')
    if len(parts) == 2 and parts[1] != '0':
        fps = float(parts[0]) / float(parts[1])
        if abs(fps - 30.0) > 0.01:
            raise ValueError(f"Validation failed: Invalid fps {fps}, expected 30.0")
    else:
        raise ValueError(f"Validation failed: Could not parse fps string {fps_str}")
        
    if not audio_stream:
        raise ValueError("Validation failed: No audio stream found.")
        
    print("Validation passed: 1080x1920, 30fps, H.264, Audio OK.")

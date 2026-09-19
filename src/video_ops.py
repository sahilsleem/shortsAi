import subprocess
import os
import shutil
import json
from pathlib import Path

def get_ffmpeg_path() -> str:
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path): return env_path
    which_path = shutil.which("ffmpeg")
    if which_path: return which_path
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(fallback): return fallback
    return "ffmpeg"

def get_ffprobe_path() -> str:
    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and os.path.exists(env_path): return env_path
    which_path = shutil.which("ffprobe")
    if which_path: return which_path
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"
    if os.path.exists(fallback): return fallback
    return "ffprobe"

def get_video_dimensions(video_path: str):
    cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    return 1080, 1920

def render_main_video(
    source_video: str,
    output_video: str,
    caption_overlay: str,
    start_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int
):
    duration = end_time - start_time
    video_filters = "eq=contrast=1.03:brightness=0.01:saturation=1.08:gamma=1.02,unsharp=3:3:0.3:3:3:0.0"
    
    cmd = [get_ffmpeg_path(), "-y"]
    cmd.extend(["-ss", str(start_time), "-t", str(duration), "-i", source_video])
    cmd.extend(["-i", caption_overlay])
    
    fc = []
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1080:1080,{video_filters}[v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={duration}:r=30[base];")
    fc.append(f"[base][v_proc]overlay=0:420:eof_action=pass[bg];")
    fc.append(f"[bg][1:v]overlay=0:0[outv]")
    
    cmd.extend([
        "-filter_complex", "".join(fc),
        "-map", "[outv]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        output_video
    ])
    print("Running Main Video FFmpeg command")
    subprocess.run(cmd, check=True)

def render_curiosity_video(
    source_video: str,
    output_video: str,
    hook_overlay: str,
    reveal_overlay: str,
    start_time: float,
    cut_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    transition_dur: float = 0.15
):
    hook_dur = cut_time - start_time
    rev_dur = end_time - cut_time
    hook_start = start_time
    reveal_start = cut_time
    
    video_filters = "eq=contrast=1.03:brightness=0.01:saturation=1.08:gamma=1.02,unsharp=3:3:0.3:3:3:0.0"
    
    cmd = [get_ffmpeg_path(), "-y"]
    
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    cmd.extend(["-i", hook_overlay])
    cmd.extend(["-i", reveal_overlay])
    
    # We must explicitly read the audio streams starting at hook_start and reveal_start to avoid sync issues with concat
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    
    fc = []
    
    # Hook segment
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1080:1080,{video_filters}[hook_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={hook_dur}:r=30[hook_base];")
    fc.append(f"[hook_base][hook_v_proc]overlay=0:420:eof_action=pass[hook_bg1];")
    fc.append(f"[hook_bg1][2:v]overlay=0:0[hook_bg2];")
    fc.append(f"[hook_bg2]fade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}:c=black[hook_v_final];")
    fc.append(f"[4:a]afade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}[hook_a_final];")
    
    # Reveal segment
    fc.append(f"[1:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1080:1080,{video_filters}[rev_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={rev_dur}:r=30[rev_base];")
    fc.append(f"[rev_base][rev_v_proc]overlay=0:420:eof_action=pass[rev_bg1];")
    fc.append(f"[rev_bg1][3:v]overlay=0:0[rev_bg2];")
    fc.append(f"[rev_bg2]fade=t=in:st=0:d={transition_dur}:c=black[rev_v_final];")
    fc.append(f"[5:a]afade=t=in:st=0:d={transition_dur}[rev_a_final];")
    
    fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[outv][outa]")
    
    cmd.extend([
        "-filter_complex", "".join(fc),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        output_video
    ])
    print("Running Curiosity Video FFmpeg command")
    subprocess.run(cmd, check=True)

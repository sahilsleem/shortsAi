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

DEFAULT_WATERMARK_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "saba_bollywood_watermark.png")

# Video Enhancement Filters
# Base/Default: Existing subtle polish
DEFAULT_VIDEO_FILTERS = "eq=contrast=1.03:brightness=0.01:saturation=1.08:gamma=1.02,unsharp=3:3:0.3:3:3:0.0"

# Enhanced Video: Cleaner, punchier social-video finishing pass
# 1. Subtle denoise (hqdn3d) to eliminate digital sensor grain and compression artifacts
# 2. Balanced contrast and modest saturation for punchy social colors while preserving natural skin tones
# 3. Clean clarity and sharpening (unsharp) without crunchy halos or color fringing
ENHANCED_VIDEO_FILTERS = "hqdn3d=1.5:1.5:3:3,eq=contrast=1.06:brightness=0.015:saturation=1.12:gamma=1.02,unsharp=5:5:0.5:3:3:0.0"

# Audio Enhancement Filters
# 1. Enhanced Mix: Balanced speech clarity, dynamic compression, rumble/hiss cleanup, -16 LUFS loudness normalization
ENHANCED_AUDIO_FILTERS = (
    "highpass=f=80,lowpass=f=12000,"
    "equalizer=f=250:t=q:w=1.2:g=-1.5,"
    "equalizer=f=3000:t=q:w=1.2:g=2.5,"
    "acompressor=threshold=0.1:ratio=3:attack=15:release=120:makeup=2,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)

# 2. Voice Focus: Speech-priority shaping, attenuating competing music frequencies, targeted dialogue presence, -16 LUFS
VOICE_FOCUS_AUDIO_FILTERS = (
    "highpass=f=120,lowpass=f=7500,"
    "equalizer=f=200:t=q:w=1.0:g=-3.0,"
    "equalizer=f=600:t=q:w=1.5:g=-2.5,"
    "equalizer=f=2800:t=q:w=1.2:g=4.0,"
    "equalizer=f=4500:t=q:w=1.5:g=2.5,"
    "acompressor=threshold=0.08:ratio=4:attack=10:release=100:makeup=2.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=8"
)

def get_video_filters(enhance: bool = False) -> str:
    """Return video filter string based on enhancement toggle."""
    return ENHANCED_VIDEO_FILTERS if enhance else DEFAULT_VIDEO_FILTERS

def get_audio_filters(audio_mode: str = "enhanced") -> str:
    """Return audio filter chain for the specified mode ('enhanced' or 'voice_focus')."""
    if audio_mode == "voice_focus":
        return VOICE_FOCUS_AUDIO_FILTERS
    return ENHANCED_AUDIO_FILTERS

def has_audio_stream(video_path: str) -> bool:
    """Detect if input video has an audio stream."""
    try:
        cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return True
    except Exception:
        # Default to True so filter graphs construct reliably in mock/tests or if probe fails gracefully
        return True
    return False

def render_main_video(
    source_video: str,
    output_video: str,
    caption_overlay: str,
    start_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    watermark_path: str = None,
    enhance: bool = False,
    audio_mode: str = "original"
):
    if watermark_path is None:
        watermark_path = DEFAULT_WATERMARK_PATH
    has_wm = bool(watermark_path and os.path.exists(watermark_path))

    duration = end_time - start_time
    video_filters = get_video_filters(enhance)
    apply_audio_enhance = enhance and (audio_mode in ("enhanced", "voice_focus")) and has_audio_stream(source_video)
    
    cmd = [get_ffmpeg_path(), "-y"]
    cmd.extend(["-ss", str(start_time), "-t", str(duration), "-i", source_video])
    cmd.extend(["-i", caption_overlay])
    if has_wm:
        cmd.extend(["-i", watermark_path])
    
    fc = []
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={duration}:r=30[base];")
    fc.append(f"[base][v_proc]overlay=39:420:eof_action=pass[bg];")
    if has_wm:
        fc.append(f"[bg][1:v]overlay=0:0[with_cap];")
        if apply_audio_enhance:
            fc.append(f"[with_cap][2:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[0:a]{a_filter}[outa]")
        else:
            fc.append(f"[with_cap][2:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]")
    else:
        if apply_audio_enhance:
            fc.append(f"[bg][1:v]overlay=0:0[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[0:a]{a_filter}[outa]")
        else:
            fc.append(f"[bg][1:v]overlay=0:0[outv]")
    
    if apply_audio_enhance:
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        maps = ["-map", "[outv]", "-map", "0:a?"]

    cmd.extend([
        "-filter_complex", "".join(fc),
        *maps,
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
    transition_dur: float = 0.15,
    watermark_path: str = None,
    enhance: bool = False,
    audio_mode: str = "original"
):
    if watermark_path is None:
        watermark_path = DEFAULT_WATERMARK_PATH
    has_wm = bool(watermark_path and os.path.exists(watermark_path))

    hook_dur = cut_time - start_time
    rev_dur = end_time - cut_time
    hook_start = start_time
    reveal_start = cut_time
    
    video_filters = get_video_filters(enhance)
    apply_audio_enhance = enhance and (audio_mode in ("enhanced", "voice_focus"))
    
    cmd = [get_ffmpeg_path(), "-y"]
    
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    cmd.extend(["-i", hook_overlay])
    cmd.extend(["-i", reveal_overlay])
    
    # We must explicitly read the audio streams starting at hook_start and reveal_start to avoid sync issues with concat
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    
    if has_wm:
        cmd.extend(["-i", watermark_path])
        wm_idx = 6
    
    fc = []
    
    # Hook segment
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[hook_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={hook_dur}:r=30[hook_base];")
    fc.append(f"[hook_base][hook_v_proc]overlay=39:420:eof_action=pass[hook_bg1];")
    fc.append(f"[hook_bg1][2:v]overlay=0:0[hook_bg2];")
    fc.append(f"[hook_bg2]fade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}:c=black[hook_v_final];")
    fc.append(f"[4:a]afade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}[hook_a_final];")
    
    # Reveal segment
    fc.append(f"[1:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[rev_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={rev_dur}:r=30[rev_base];")
    fc.append(f"[rev_base][rev_v_proc]overlay=39:420:eof_action=pass[rev_bg1];")
    fc.append(f"[rev_bg1][3:v]overlay=0:0[rev_bg2];")
    fc.append(f"[rev_bg2]fade=t=in:st=0:d={transition_dur}:c=black[rev_v_final];")
    fc.append(f"[5:a]afade=t=in:st=0:d={transition_dur}[rev_a_final];")
    
    raw_a = "raw_a" if apply_audio_enhance else "outa"

    if has_wm:
        fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[v_concat][{raw_a}];")
        if apply_audio_enhance:
            fc.append(f"[v_concat][{wm_idx}:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[{raw_a}]{a_filter}[outa]")
        else:
            fc.append(f"[v_concat][{wm_idx}:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]")
    else:
        if apply_audio_enhance:
            fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[outv][{raw_a}];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[{raw_a}]{a_filter}[outa]")
        else:
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

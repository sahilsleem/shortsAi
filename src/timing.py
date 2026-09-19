import subprocess
import json
from src.video_ops import get_ffprobe_path

def determine_reveal_timing(video_path: str, curiosity_text: str, reveal_text: str):
    """
    Determine the timings for the hook and reveal segments.
    
    NOTE: This is currently a temporary deterministic baseline that simply splits
    the video 50/50. It will later be replaced with intelligent content-aware
    reveal-point detection (computer vision).
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    duration = float(data["format"]["duration"])
    
    split = duration / 2.0
        
    return {
        "hook_start": 0.0,
        "hook_end": round(split, 3),
        "reveal_start": round(split, 3),
        "reveal_end": duration
    }

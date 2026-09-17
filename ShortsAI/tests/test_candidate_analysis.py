import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.candidate_analysis import CandidateAnalyzer
from src.video_ops import get_ffmpeg_path

def create_synthetic_video(path: str):
    # Create a 2-second 10fps video with a beep audio
    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-c:a", "aac",
        path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def test_candidate_analysis():
    print("Setting up synthetic video...")
    os.makedirs("working", exist_ok=True)
    test_video = "working/test_synthetic.mp4"
    create_synthetic_video(test_video)
    
    analyzer = CandidateAnalyzer(test_video)
    
    # Fake frames dir
    test_frames_dir = "working/test_cand_frames"
    if os.path.exists(test_frames_dir):
        shutil.rmtree(test_frames_dir)
    os.makedirs(test_frames_dir, exist_ok=True)
    
    # Generate some fake frame images
    from PIL import Image
    for i in range(2):
        img = Image.new("RGB", (320, 240), color="blue")
        img.save(os.path.join(test_frames_dir, f"frame_{i+1:04d}.jpg"))
        
    print("Testing visual analysis...")
    vis = analyzer.analyze_visuals(test_frames_dir)
    assert len(vis) == 2
    assert vis[0].timestamp == 0.0
    
    print("Testing audio analysis...")
    aud = analyzer.analyze_audio()
    assert len(aud) > 0
    assert aud[0].rms_level > 0.0
    
    print("Testing scene changes...")
    scenes = analyzer.analyze_scenes()
    # synthetic might not have scene changes, so len >= 0
    assert isinstance(scenes, list)
    
    print("Testing candidate scoring...")
    cands = analyzer.score_candidates(vis, aud, scenes, duration=2.0)
    assert len(cands) > 0
    assert 0.0 <= cands[0].score <= 1.0
    assert isinstance(cands[0].timestamp, float)
    
    print("Candidate analysis tests passed!")

if __name__ == "__main__":
    test_candidate_analysis()

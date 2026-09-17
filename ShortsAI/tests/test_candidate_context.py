import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.candidate_context import ContextExtractor
from src.video_ops import get_ffmpeg_path
import subprocess

def create_synthetic_video(path: str):
    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
        "-c:v", "libx264", "-c:a", "aac",
        path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def test_candidate_context():
    print("Setting up synthetic environment...")
    os.makedirs("working", exist_ok=True)
    test_video = "working/test_synthetic_context.mp4"
    create_synthetic_video(test_video)
    
    # Mock transcript
    mock_transcript = [
        {"start": 0.0, "end": 2.0, "text": "Hello world."},
        {"start": 2.5, "end": 4.5, "text": "Testing overlapping speech."}
    ]
    
    extractor = ContextExtractor(test_video, source_duration=5.0)
    
    # 1. Test overlapping speech logic
    speech_1 = extractor.get_overlapping_speech(0.5, 1.5, mock_transcript)
    assert len(speech_1) == 1
    assert speech_1[0].text == "Hello world."
    
    speech_2 = extractor.get_overlapping_speech(1.5, 3.5, mock_transcript)
    assert len(speech_2) == 2  # Overlaps both
    
    speech_none = extractor.get_overlapping_speech(2.1, 2.4, mock_transcript)
    assert len(speech_none) == 0
    
    # 2. Test extraction
    test_candidates_dir = "working/test_candidates"
    if os.path.exists(test_candidates_dir):
        shutil.rmtree(test_candidates_dir)
    os.makedirs(test_candidates_dir, exist_ok=True)
    
    clip_path = os.path.join(test_candidates_dir, "cand_01.mp4")
    extractor.extract_clip(1.0, 3.0, clip_path)
    assert os.path.exists(clip_path)
    
    frames_dir = os.path.join(test_candidates_dir, "cand_01")
    frames = extractor.extract_frames(1.0, 3.0, frames_dir)
    assert len(frames) > 0
    
    # Test clamped timestamps (going beyond duration)
    frames_clamped = extractor.extract_frames(4.0, 6.0, os.path.join(test_candidates_dir, "cand_02"))
    assert len(frames_clamped) > 0
    assert frames_clamped[-1].timestamp < 5.0
    
    print("Candidate context tests passed!")

if __name__ == "__main__":
    test_candidate_context()

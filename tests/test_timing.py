import pytest
from src.timing import determine_reveal_timing
import os

# Create a small dummy timing module to mock ffprobe
def test_determine_reveal_timing(monkeypatch):
    
    # Mock subprocess.run for ffprobe
    import subprocess
    import json
    
    class MockResult:
        def __init__(self, duration):
            self.stdout = json.dumps({"format": {"duration": str(duration)}})
            self.returncode = 0
            
    def mock_run(cmd, *args, **kwargs):
        # We can simulate different durations based on some fake video path
        duration = float(cmd[-1].split("_")[-1].replace(".mp4", ""))
        return MockResult(duration)
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    # 1. Normal 8-second video
    res = determine_reveal_timing("dummy_8.0.mp4", "Curiosity", "Reveal")
    assert res["hook_start"] == 0.0
    assert res["hook_end"] == 4.0
    assert res["reveal_start"] == 4.0
    assert res["reveal_end"] == 8.0
    
    # 2. Shorter video
    res = determine_reveal_timing("dummy_5.0.mp4", "", "")
    assert res["hook_end"] == 2.5
    assert res["reveal_start"] == 2.5
    assert res["reveal_end"] == 5.0
    
    # 3. Longer video
    res = determine_reveal_timing("dummy_12.0.mp4", "", "")
    assert res["hook_end"] == 6.0
    assert res["reveal_end"] == 12.0
    
    # 4. Very short video
    res = determine_reveal_timing("dummy_0.5.mp4", "", "")
    assert res["hook_end"] == 0.25
    assert res["reveal_start"] == 0.25
    assert res["reveal_end"] == 0.5
    
    # Timing boundaries check:
    # hook_end equals reveal_start
    assert res["hook_end"] == res["reveal_start"]
    
    # no timing exceeds video duration
    assert res["reveal_end"] <= 0.5
    
    # no negative values
    assert res["hook_start"] >= 0


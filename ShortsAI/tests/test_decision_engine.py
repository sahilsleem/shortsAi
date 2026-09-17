import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.decision_engine import DecisionEngine

def test_decision_engine():
    # Setup mock candidate_context.json
    mock_data = {
        "source": {},
        "candidates": [
            {
                "rank": 1,
                "timestamp": 1.0,
                "window_start": 0.0,
                "window_end": 3.0,
                "score": 0.8,
                "signals": {
                    "motion": 0.8,
                    "audio_energy": 0.9,
                    "scene_change": 0.1,
                    "visual_detail": 0.5
                },
                "speech_segments": [{"text": "Hello"}],
                "clip_path": "",
                "frames": []
            },
            {
                "rank": 2,
                "timestamp": 1.5,  # Should be suppressed due to overlap with rank 1
                "window_start": 0.5,
                "window_end": 3.5,
                "score": 0.75,
                "signals": {
                    "motion": 0.7,
                    "audio_energy": 0.8,
                    "scene_change": 0.0,
                    "visual_detail": 0.5
                },
                "speech_segments": [],
                "clip_path": "",
                "frames": []
            },
            {
                "rank": 3,
                "timestamp": 5.0,  # Valid alternative
                "window_start": 4.0,
                "window_end": 6.0,
                "score": 0.6,
                "signals": {
                    "motion": 0.2,
                    "audio_energy": 0.3,
                    "scene_change": 0.0,
                    "visual_detail": 0.2
                },
                "speech_segments": [],
                "clip_path": "",
                "frames": []
            }
        ]
    }
    
    os.makedirs("working", exist_ok=True)
    with open("working/test_mock_candidate_context.json", "w") as f:
        json.dump(mock_data, f)
        
    engine = DecisionEngine("working/test_mock_candidate_context.json")
    
    decision = engine.make_decision()
    
    # Verify basics
    assert decision.selected_candidate_rank == 1
    assert decision.selected_timestamp == 1.0
    
    # Verify score normalization
    assert 0.0 <= decision.confidence <= 1.0
    
    # Verify temporal suppression (there were 3 candidates, but 2 overlap. Output should show 3 in candidate_scores, but suppression picked rank 1 over rank 2).
    # The decision candidate_scores should have all 3.
    assert len(decision.candidate_scores) == 3
    
    # Verify speech context
    assert len(decision.caption_context.speech_segments) == 1
    assert decision.caption_context.speech_segments[0]["text"] == "Hello"
    assert decision.caption_context.transcript_reliability == "unknown"
    
    # Verify visual review flag
    assert decision.needs_visual_review is True
    
    print("Decision engine tests passed!")

if __name__ == "__main__":
    test_decision_engine()

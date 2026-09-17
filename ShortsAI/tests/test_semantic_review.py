import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.semantic_review import SemanticReviewBuilder

def test_semantic_review():
    # Setup mock decision and context files
    mock_decision = {
        "version": 1,
        "selected_candidate_rank": 2,
        "confidence": 0.85
    }
    
    mock_context = {
        "source": {"fps": 30},
        "candidates": [
            {
                "rank": 1,
                "timestamp": 1.0,
                "score": 0.5,
                "frames": [{"timestamp": 1.0, "path": "f1.jpg"}],
                "speech_segments": []
            },
            {
                "rank": 2,
                "timestamp": 2.0,
                "window_start": 1.5,
                "window_end": 4.5,
                "score": 0.9,
                "frames": [
                    {"timestamp": 1.5, "path": "f1.jpg"},
                    {"timestamp": 3.0, "path": "f2.jpg"},
                    {"timestamp": 4.5, "path": "f3.jpg"}
                ],
                "speech_segments": [
                    {"start": 2.0, "end": 3.0, "text": "Mock text"}
                ]
            }
        ]
    }
    
    os.makedirs("working", exist_ok=True)
    decision_path = "working/test_mock_decision.json"
    context_path = "working/test_mock_context.json"
    
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(mock_decision, f)
        
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(mock_context, f)
        
    builder = SemanticReviewBuilder(decision_path, context_path)
    req = builder.build_request()
    
    # Verify rank linking
    assert req.candidate.rank == 2
    assert req.candidate.timestamp == 2.0
    
    # Verify frames matched
    assert len(req.visual_evidence) == 3
    assert req.visual_evidence[1].timestamp == 3.0
    
    # Verify transcript text preserved
    assert len(req.speech_evidence) == 1
    assert req.speech_evidence[0].text == "Mock text"
    
    # Verify instructions and schema
    assert req.instructions.detect_uncertainty is True
    assert "unknown" in " ".join(req.instructions.hallucination_safeguards)
    assert "uncertainties" in req.expected_response_schema
    
    # Verify JSON serialization works implicitly
    import dataclasses
    json.dumps(dataclasses.asdict(req))
    
    print("Semantic review tests passed!")

if __name__ == "__main__":
    test_semantic_review()

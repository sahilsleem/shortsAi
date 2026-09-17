import os
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.caption_decision import (
    CaptionDecisionMaker, CaptionDecision, ValidationError, 
    DryRunCaptionProvider, GeminiCaptionProvider, CaptionDecisionValidator
)

def test_caption_decision(monkeypatch):
    # 1. Dry-run provider
    dry_provider = DryRunCaptionProvider()
    maker = CaptionDecisionMaker(dry_provider)
    
    valid_semantic = {
        "request_metadata": {"candidate_rank": 1, "timestamp": 1.0},
        "semantic_review": {
            "confidence": 0.9,
            "event": "A man is running.",
            "caption_fact_basis": ["A man is running quickly.", "He is wearing red."],
            "uncertainties": [],
            "reveal": {"idea": "He trips."}
        }
    }
    
    decision = maker.decide({}, valid_semantic)
    if not decision.usable:
        print("Dry run failed reasoning:", decision.reasoning)
    assert decision.usable is True
    assert decision.caption.main_text == "A man is running."

test_caption_decision(type("Monkey", (), {"setenv": lambda k,v: None}))

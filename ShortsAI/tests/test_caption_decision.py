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
    assert decision.usable is True
    assert decision.caption.main_text == "A man is running."
    
    # 2. Gemini provider construction without making a request
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    import google.genai as genai
    
    class MockGeminiModels:
        def __init__(self, response_text):
            self.response_text = response_text
        def generate_content(self, model, contents, config=None):
            class MockResponse:
                text = self.response_text
            return MockResponse()
            
    class MockGeminiClient:
        def __init__(self, response_text="{}"):
            self.models = MockGeminiModels(response_text)
            
    genai.Client = lambda api_key=None: MockGeminiClient()
    
    gemini_provider = GeminiCaptionProvider()
    assert gemini_provider.api_key == "mock_key"
    
    # 3. Valid mocked AI response (generic replaced by more natural wording)
    valid_ai_response = json.dumps({
        "main_text": "Ceremonial aarti ritual during a festive gathering.",
        "curiosity_text": "But something unexpected happens...",
        "emoji": "🔥",
        "reveal_text": "He makes a mistake.",
        "fact_basis": ["A man is performing aarti."],
        "unsupported_claims": ["He is doing it the wrong way."],
        "confidence": 0.9,
        "reasoning": ["Natural phrasing identified."]
    })
    
    genai.Client = lambda api_key=None: MockGeminiClient(valid_ai_response)
    maker_gemini = CaptionDecisionMaker(GeminiCaptionProvider())
    
    decision_ai = maker_gemini.decide({}, valid_semantic)
    assert decision_ai.usable is True
    assert decision_ai.caption.main_text == "Ceremonial aarti ritual during a festive gathering."
    assert decision_ai.caption.emoji == "🔥"
    
    # 4. Missing required field
    invalid_ai_response_missing = json.dumps({
        "main_text": "He is running.",
        "emoji": "🏃",
        "reveal_text": "He falls.",
        "fact_basis": ["A man is running quickly."],
        "unsupported_claims": [],
        "confidence": 0.9,
        "reasoning": ["Looks good."]
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_ai_response_missing)
    maker_missing = CaptionDecisionMaker(GeminiCaptionProvider())
    dec_missing = maker_missing.decide({}, valid_semantic)
    assert dec_missing.usable is False
    
    # 5. Invalid confidence
    invalid_conf = json.dumps({
        "main_text": "He is running.", "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["A man is running quickly."], "unsupported_claims": [],
        "confidence": 1.5, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_conf)
    dec_conf = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_conf.usable is False
    
    # 6. Newline/HTML/markdown rejection
    invalid_html = json.dumps({
        "main_text": "He is <b>running</b>.", "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["A man is running quickly."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_html)
    dec_html = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_html.usable is False
    
    # 8. On-screen text cannot be sole factual basis
    invalid_sole_text = json.dumps({
        "main_text": "He is running.", "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["The on-screen text says he runs."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_sole_text)
    dec_text = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_text.usable is False
    
    # 9. Unreliable transcript cannot be sole factual basis
    invalid_sole_transcript = json.dumps({
        "main_text": "He is running.", "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["The transcript says he runs."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_sole_transcript)
    dec_transcript = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_transcript.usable is False
    
    # 10. Graphics instructions rejected
    invalid_graphics = json.dumps({
        "main_text": "He is running. Draw a red circle.", "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["A man is running quickly."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_graphics)
    dec_graphics = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_graphics.usable is False
    
    # 11. Excessive caption length rejected
    long_text = "This is a very very long text that just keeps going and going and going and going and going and going to break the one hundred character limit easily."
    invalid_len = json.dumps({
        "main_text": long_text, "curiosity_text": "", "emoji": "", "reveal_text": "",
        "fact_basis": ["A man is running quickly."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_len)
    dec_len = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_len.usable is False
    
    # 12. More than two emoji rejected
    invalid_emoji = json.dumps({
        "main_text": "He is running.", "curiosity_text": "", "emoji": "🏃‍♂️🏃‍♀️🔥", "reveal_text": "",
        "fact_basis": ["A man is running quickly."], "unsupported_claims": [],
        "confidence": 0.9, "reasoning": []
    })
    genai.Client = lambda api_key=None: MockGeminiClient(invalid_emoji)
    dec_emoji = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_emoji.usable is False
    
    # 7. Unsupported claim rejection (checked via deterministic filtering)
    text_overlay_semantic = {
        "request_metadata": {"candidate_rank": 1, "timestamp": 1.0},
        "semantic_review": {
            "confidence": 0.8,
            "event": "A man stands there.",
            "caption_fact_basis": ["On-screen text says he is doing it wrong.", "He stands."],
            "uncertainties": ["Identity cannot be visually confirmed without text."],
            "reveal": {"idea": ""}
        }
    }
    
    # valid_ai_response simulates rejecting "wrong way"
    genai.Client = lambda api_key=None: MockGeminiClient(valid_ai_response)
    dec_unsupported = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, text_overlay_semantic)
    
    assert any("text" in claim.lower() for claim in dec_unsupported.unsupported_claims)
    assert any("wrong way" in claim.lower() for claim in dec_unsupported.unsupported_claims)
    
    # Test valid curiosity and no curiosity
    no_curiosity_ai = json.dumps({
        "main_text": "Ceremonial aarti ritual.",
        "curiosity_text": "",
        "emoji": "🪔",
        "reveal_text": "",
        "fact_basis": ["A man is performing aarti."],
        "unsupported_claims": [],
        "confidence": 0.9,
        "reasoning": ["No interesting unresolved moment, so curiosity left empty."]
    })
    genai.Client = lambda api_key=None: MockGeminiClient(no_curiosity_ai)
    dec_no_cur = CaptionDecisionMaker(GeminiCaptionProvider()).decide({}, valid_semantic)
    assert dec_no_cur.caption.curiosity_text == ""
    assert dec_no_cur.caption.emoji == "🪔"
    
    # 14. Network gate refuses Gemini without --allow-network.
    def run_cli(args):
        return subprocess.run([sys.executable, "-m", "src.caption_decision"] + args, capture_output=True, text=True)
        
    res_gemini_safe = run_cli(["--provider", "gemini"])
    assert "Network request halted for safety" in res_gemini_safe.stdout
    assert "requires --allow-network" in res_gemini_safe.stdout
    
    res_dryrun = run_cli(["--provider", "dry-run"])
    assert "Caption Decision generation complete" in res_dryrun.stdout
    
    print("All tests passed including CLI safety gates and mocked AI responses!")

if __name__ == "__main__":
    import os
    class MonkeyPatchMock:
        def setenv(self, k, v):
            os.environ[k] = v
            
    test_caption_decision(MonkeyPatchMock())

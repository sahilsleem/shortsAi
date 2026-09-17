import os
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.semantic_ai import (
    RequestValidator, 
    ResponseValidator, 
    DryRunSemanticProvider,
    OpenAISemanticProvider,
    GeminiSemanticProvider,
    ValidationError
)

def create_dummy_frame(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("dummy")

class MockOpenAIClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                class MockMessage:
                    content = json.dumps({
                        "scene_summary": "Mock OpenAI Scene",
                        "people": [{"description": "Mock Person", "identity": "unknown", "action": "Mock Action"}],
                        "interaction": "Mock Interaction",
                        "expressions": "Mock Expressions",
                        "event": "Mock Event",
                        "hook": {"available": True, "idea": "Mock Hook"},
                        "reveal": {"available": True, "idea": "Mock Reveal"},
                        "caption_fact_basis": [],
                        "confidence": 0.9,
                        "uncertainties": ["Mock Uncertainty"]
                    })
                class MockChoice:
                    message = MockMessage()
                class MockResponse:
                    choices = [MockChoice()]
                return MockResponse()

def test_semantic_ai_validation(monkeypatch):
    req_validator = RequestValidator()
    res_validator = ResponseValidator()
    provider = DryRunSemanticProvider()
    
    base_dir = "working/test_semantic_ai"
    f1 = os.path.join(base_dir, "f1.jpg")
    f2 = os.path.join(base_dir, "f2.jpg")
    f3 = os.path.join(base_dir, "f3.jpg")
    f4 = os.path.join(base_dir, "f4.jpg")
    create_dummy_frame(f1)
    create_dummy_frame(f2)
    create_dummy_frame(f3)
    create_dummy_frame(f4)
    
    valid_req = {
        "version": 1,
        "task": {},
        "source": {},
        "candidate": {
            "rank": 1, "timestamp": 1.0, "window_start": 0.0, "window_end": 3.0,
            "deterministic_score": 0.5, "decision_confidence": 0.8
        },
        "visual_evidence": [
            {"path": f1}, {"path": f2}, {"path": f3}
        ],
        "speech_evidence": [],
        "instructions": {},
        "expected_response_schema": {}
    }
    
    # Validation tests
    req_validator.validate(valid_req)
    
    invalid_req1 = valid_req.copy()
    del invalid_req1["task"]
    try:
        req_validator.validate(invalid_req1)
        assert False
    except ValidationError:
        pass
        
    invalid_req2 = valid_req.copy()
    invalid_req2["visual_evidence"] = [{"path": f1}, {"path": f2}]
    try:
        req_validator.validate(invalid_req2)
        assert False
    except ValidationError:
        pass
        
    res = provider.review(valid_req)
    info = provider.get_provider_info()
    assert info["inference_performed"] is False
    assert info["name"] == "dry-run"
    
    res_validator.validate(res)
    
    # Test OpenAI Provider Mock
    monkeypatch.setenv("OPENAI_API_KEY", "mock_key")
    import openai
    openai.OpenAI = lambda **kwargs: MockOpenAIClient()
    
    openai_provider = OpenAISemanticProvider()
    assert openai_provider.api_key == "mock_key"
    
    openai_info = openai_provider.get_provider_info()
    assert openai_info["inference_performed"] is True
    assert openai_info["name"] == "openai"
    
    openai_res = openai_provider.review(valid_req)
    res_validator.validate(openai_res)
    assert openai_res["scene_summary"] == "Mock OpenAI Scene"
    
    # Test Gemini Provider Mock
    monkeypatch.setenv("GEMINI_API_KEY", "mock_gemini_key")
    import google.genai as genai
    
    class MockGeminiModels:
        def __init__(self, should_fail_503=False):
            self.should_fail_503 = should_fail_503
            
        def generate_content(self, model, contents, config=None):
            if self.should_fail_503:
                raise Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")
                
            class MockResponse:
                text = json.dumps({
                    "scene_summary": "Mock Gemini Scene",
                    "people": [{"description": "Mock Person", "identity": "unknown", "action": "Mock Action"}],
                    "interaction": "Mock Interaction",
                    "expressions": "Mock Expressions",
                    "event": "Mock Event",
                    "hook": {"available": True, "idea": "Mock Hook"},
                    "reveal": {"available": True, "idea": "Mock Reveal"},
                    "caption_fact_basis": [],
                    "confidence": 0.8,
                    "uncertainties": ["Mock Uncertainty"]
                })
            return MockResponse()
            
    class MockGeminiClient:
        def __init__(self, api_key=None, should_fail_503=False):
            self.models = MockGeminiModels(should_fail_503=should_fail_503)
            
    # Mock PIL so it doesn't crash on dummy file
    class MockImage:
        @staticmethod
        def open(path):
            class MImg:
                def load(self):
                    pass
            return MImg()
            
    import PIL
    PIL.Image.open = MockImage.open
    
    genai.Client = MockGeminiClient
    
    gemini_provider = GeminiSemanticProvider()
    assert gemini_provider.api_key == "mock_gemini_key"
    assert gemini_provider.model_name == "gemini-3.6-flash"
    
    gemini_info = gemini_provider.get_provider_info()
    assert gemini_info["inference_performed"] is True
    assert gemini_info["name"] == "gemini"
    
    gemini_res = gemini_provider.review(valid_req)
    res_validator.validate(gemini_res)
    assert gemini_res["scene_summary"] == "Mock Gemini Scene"
    assert gemini_res["confidence"] == 0.8
    
    # Test Gemini 503 logic
    genai.Client = lambda api_key=None: MockGeminiClient(api_key=api_key, should_fail_503=True)
    gemini_provider_503 = GeminiSemanticProvider()
    try:
        gemini_provider_503.review(valid_req)
        assert False, "Should have thrown an exception for 503"
    except RuntimeError as e:
        assert "retryable temporary service-capacity error" in str(e)
    
    # Test missing image throws ValidationError
    invalid_image_req = valid_req.copy()
    invalid_image_req["visual_evidence"] = [
        {"path": "non_existent_1.jpg"},
        {"path": "non_existent_2.jpg"},
        {"path": "non_existent_3.jpg"}
    ]
    genai.Client = MockGeminiClient  # reset client mock
    gemini_provider_invalid_img = GeminiSemanticProvider()
    try:
        gemini_provider_invalid_img.review(invalid_image_req)
        assert False, "Should have thrown ValidationError for missing image"
    except ValidationError:
        pass
        
    print("Class logic tests passed.")
    
    # --- CLI SAFETY TESTS ---
    def run_cli(args):
        return subprocess.run([sys.executable, "-m", "src.semantic_ai"] + args, capture_output=True, text=True)
        
    res_gemini_safe = run_cli(["--provider", "gemini"])
    assert "Network request halted for safety" in res_gemini_safe.stdout
    assert "requires --allow-network" in res_gemini_safe.stdout

    res_openai_safe = run_cli(["--provider", "openai"])
    assert "Network request halted for safety" in res_openai_safe.stdout
    assert "requires --allow-network" in res_openai_safe.stdout
    
    res_dryrun = run_cli(["--provider", "dry-run"])
    assert "Semantic AI adapter dry-run complete" in res_dryrun.stdout
    
    print("All Semantic AI tests passed including CLI safety gates!")

if __name__ == "__main__":
    import os
    class MonkeyPatchMock:
        def setenv(self, k, v):
            os.environ[k] = v
            
    test_semantic_ai_validation(MonkeyPatchMock())

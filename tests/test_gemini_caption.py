import os
import sys
import json
import io
import time
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.gemini_caption import (
    DEFAULT_GEMINI_MODEL,
    generate_captions,
    parse_gemini_response,
    build_gemini_payload,
    clean_json_text,
    GeminiError,
    GeminiConfigError,
    GeminiQuotaError,
    GeminiAPIError,
)
from src.server import ShortsAIHandler

def test_model_configuration_and_url(monkeypatch):
    """Verify default model is gemini-3.8-flash, GEMINI_MODEL overrides it, and endpoint URL is properly constructed."""
    assert DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"

    captured_urls = []
    class MockResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({
                "candidates": [{"content": {"parts": [{"text": '{"main_caption": "A, B", "curiosity_caption": "C"}'}]}}]
            }).encode("utf-8")

    def mock_urlopen(req, timeout=15):
        captured_urls.append(req.full_url)
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # 1. Default model
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    generate_captions(context="Test context", api_key="test_key_1")
    assert len(captured_urls) == 1
    assert "models/gemini-3.8-flash:generateContent" in captured_urls[0]
    assert "key=test_key_1" in captured_urls[0]

    # 2. GEMINI_MODEL environment variable override
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-flash")
    generate_captions(context="Test context", api_key="test_key_2")
    assert len(captured_urls) == 2
    assert "models/gemini-custom-flash:generateContent" in captured_urls[1]
    assert "key=test_key_2" in captured_urls[1]

    # 3. Explicit model argument override
    generate_captions(context="Test context", api_key="test_key_3", model="gemini-explicit-model")
    assert len(captured_urls) == 3
    assert "models/gemini-explicit-model:generateContent" in captured_urls[2]
    assert "key=test_key_3" in captured_urls[2]

def test_missing_api_key_handling(monkeypatch):
    """Calling generate_captions without GEMINI_API_KEY raises GeminiConfigError without crashing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError) as exc_info:
        generate_captions(context="Salman was walking down the street")
    assert "GEMINI_API_KEY" in str(exc_info.value)

def test_empty_context_handling():
    """Empty or whitespace context raises ValueError."""
    with pytest.raises(ValueError):
        generate_captions(context="", api_key="dummy_key")
    with pytest.raises(ValueError):
        generate_captions(context="   \n  ", api_key="dummy_key")

def test_build_gemini_payload_with_and_without_retries():
    """Verify payload generation, Saba Bollywood instructions, visual composition rules, and retry avoidance list."""
    context = "Salman Khan was spotted at Mumbai airport greeting fans"
    
    # Without previous generations
    payload1 = build_gemini_payload(context)
    prompt1 = payload1["contents"][0]["parts"][0]["text"]
    assert context in prompt1
    assert "SETUP, CURIOSITY HOOK" in prompt1
    assert "MUST include a comma" in prompt1
    assert "Saba Bollywood" in prompt1
    assert "DO NOT repeat" not in prompt1

    # Verify visual composition & line requirements
    assert "TWO NATURAL LINES" in prompt1
    assert "Two lines should be the standard output" in prompt1
    assert "undesirable" in prompt1.lower()
    assert "leaves too much empty visual space" in prompt1
    assert "Three lines are allowed ONLY as an exception" in prompt1
    assert "Favor a visually balanced two-line composition over an exact character count" in prompt1
    assert "awkward filler words" in prompt1.lower()
    assert "45–70" not in prompt1 and "45-70" not in prompt1 and "35-65" not in prompt1

    # Verify capitalization rules (normal sentence capitalization, no Title Case)
    assert "normal English sentence capitalization" in prompt1
    assert "Do NOT use Title Case" in prompt1
    assert "Every-Word-Capitalized" in prompt1
    assert "proper nouns" in prompt1.lower()

    # Verify Gemini 3.8 Flash generationConfig compliance
    gen_config = payload1.get("generationConfig", {})
    assert gen_config.get("responseMimeType") == "application/json"
    assert gen_config.get("maxOutputTokens") == 1024
    assert gen_config.get("thinkingConfig") == {"thinkingLevel": "low"}

    # Assert absence of deprecated sampling parameters
    assert "temperature" not in gen_config
    assert "top_p" not in gen_config
    assert "top_k" not in gen_config
    assert "candidate_count" not in gen_config
    assert "thinking_budget" not in gen_config
    assert "thinkingBudget" not in gen_config

    # With previous generations (retry behavior)
    prev = [
        {"main_caption": "Salman Khan was at Mumbai airport, but then...", "curiosity_caption": "fans rushed over to hug him"},
        {"main_caption": "Salman stepped out of the terminal, however...", "curiosity_caption": "he stopped for a quick selfie"}
    ]
    payload2 = build_gemini_payload(context, previous_generations=prev)
    prompt2 = payload2["contents"][0]["parts"][0]["text"]
    assert "DO NOT repeat or closely mimic any of these previous generations" in prompt2
    assert "Salman Khan was at Mumbai airport, but then..." in prompt2
    assert "he stopped for a quick selfie" in prompt2

def test_structured_gemini_response_parsing():
    """Verify parsing valid JSON, markdown-wrapped JSON, and fallback extraction."""
    # 1. Clean JSON
    raw_json = json.dumps({
        "main_caption": "Salman Khan was walking down the street, but then...",
        "curiosity_caption": "he spotted his friend and stopped to shake hands"
    })
    res1 = parse_gemini_response(raw_json)
    assert res1["main_caption"] == "Salman Khan was walking down the street, but then..."
    assert res1["curiosity_caption"] == "he spotted his friend and stopped to shake hands"
    assert "," in res1["main_caption"]

    # 2. Markdown fenced JSON (```json ... ```)
    fenced_json = f"```json\n{raw_json}\n```"
    res2 = parse_gemini_response(fenced_json)
    assert res2["main_caption"] == res1["main_caption"]
    assert res2["curiosity_caption"] == res1["curiosity_caption"]

    # 3. Fenced with no language identifier
    fenced_no_lang = f"```\n{raw_json}\n```"
    res3 = parse_gemini_response(fenced_no_lang)
    assert res3["main_caption"] == res1["main_caption"]

def test_malformed_gemini_response_handling():
    """Verify clean error handling when Gemini returns invalid or incomplete JSON."""
    # Incomplete keys
    with pytest.raises(GeminiAPIError):
        parse_gemini_response('{"main_caption": "Only Main"}')
    with pytest.raises(GeminiAPIError):
        parse_gemini_response('{"curiosity_caption": "Only Curiosity"}')

    # Empty string values
    with pytest.raises(GeminiAPIError):
        parse_gemini_response('{"main_caption": "", "curiosity_caption": "Something"}')

    # Non-JSON garbage
    with pytest.raises(GeminiAPIError):
        parse_gemini_response("I cannot fulfill this request as an AI model.")

def test_quota_exhausted_error(monkeypatch):
    """Verify 429 / quota error retries with exponential backoff and raises GeminiQuotaError when exhausted."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    attempt_count = 0
    def mock_urlopen(req, *args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        fp = io.BytesIO(json.dumps({"error": {"message": "Resource has been exhausted (e.g. check quota)."}}).encode("utf-8"))
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, fp)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(GeminiQuotaError) as exc_info:
        generate_captions(context="Test context", api_key="dummy_key")
    assert "Gemini limit reached. Try again later." in str(exc_info.value)
    assert attempt_count == 4
    assert len(sleep_calls) == 3

def test_transient_error_retry_and_recovery(monkeypatch):
    """Verify transient errors (503, 408, 500, 429) retry and succeed when subsequent attempt passes."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    for status_code in (503, 408, 500, 429):
        sleep_calls.clear()
        attempt_count = 0
        expected_json = {
            "main_caption": "Salman Khan was walking down the street, but then...",
            "curiosity_caption": "he spotted his friend nearby and stopped to shake hands"
        }

        class MockSuccessResponse:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({
                    "candidates": [{"content": {"parts": [{"text": json.dumps(expected_json)}]}}]
                }).encode("utf-8")

        def mock_urlopen(req, timeout=15):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                fp = io.BytesIO(json.dumps({"error": {"message": f"Transient error {status_code}"}}).encode("utf-8"))
                raise urllib.error.HTTPError(req.full_url, status_code, "Error", {}, fp)
            return MockSuccessResponse()

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        res = generate_captions(context="Salman walking", api_key="test_key")
        assert attempt_count == 2
        assert len(sleep_calls) == 1
        assert res["main_caption"] == expected_json["main_caption"]

def test_permanent_errors_fail_immediately_without_retry(monkeypatch):
    """Verify permanent errors (400, 401, 403, 404) fail immediately without retrying."""
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    for status_code in (400, 401, 403, 404):
        sleep_calls.clear()
        attempt_count = 0

        def mock_urlopen(req, timeout=15):
            nonlocal attempt_count
            attempt_count += 1
            fp = io.BytesIO(json.dumps({"error": {"message": f"Client error {status_code}"}}).encode("utf-8"))
            raise urllib.error.HTTPError(req.full_url, status_code, "Client Error", {}, fp)

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        with pytest.raises((GeminiAPIError, GeminiConfigError)):
            generate_captions(context="Salman walking", api_key="test_key")
        
        assert attempt_count == 1
        assert len(sleep_calls) == 0

def test_successful_api_flow(monkeypatch):
    """Verify complete end-to-end generate_captions flow with mocked API response."""
    expected_data = {
        "main_caption": "Ranbir Kapoor was spotted leaving the gym, but then...",
        "curiosity_caption": "he noticed the paparazzi waiting outside"
    }
    # Response with thought part preceding the text part (standard in Gemini 3.8 thinking mode)
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": "Thinking deeply about the scene and creating an open loop..."},
                        {"text": json.dumps(expected_data)}
                    ]
                }
            }
        ]
    }

    captured_req = None
    class MockResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps(api_response).encode("utf-8")

    def mock_urlopen(req, timeout=15):
        nonlocal captured_req
        captured_req = req
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    res = generate_captions(context="Ranbir Kapoor leaving gym saw paparazzi", api_key="test_key")
    assert res["main_caption"] == expected_data["main_caption"]
    assert res["curiosity_caption"] == expected_data["curiosity_caption"]
    assert "," in res["main_caption"]
    assert res["main_caption"] != res["curiosity_caption"]

    # Verify authentication headers and endpoint
    assert captured_req is not None
    assert captured_req.get_header("X-goog-api-key") == "test_key"
    assert "key=test_key" in captured_req.full_url

def test_server_endpoint_handling(monkeypatch):
    """Verify ShortsAIHandler POST /api/generate_captions routing and responses."""
    # 1. Success response
    monkeypatch.setattr(
        "src.server.generate_captions",
        lambda context, previous_generations: {
            "main_caption": "Shah Rukh Khan arrived at the venue, but then...",
            "curiosity_caption": "thousands of fans started cheering his name"
        }
    )

    class DummyHandler(ShortsAIHandler):
        def __init__(self, path, body):
            self.path = path
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.response_code = None
            self.response_headers = {}

        def send_response(self, code, message=None):
            self.response_code = code

        def send_header(self, keyword, value):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

    req_body = json.dumps({"context": "SRK arrived at venue", "previous_generations": []}).encode("utf-8")
    handler = DummyHandler('/api/generate_captions', req_body)
    handler.do_POST()

    assert handler.response_code == 200
    res_json = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "main_caption" in res_json
    assert "curiosity_caption" in res_json
    assert "," in res_json["main_caption"]

    # 2. Config error (e.g. missing API key)
    def mock_config_err(*args, **kwargs):
        raise GeminiConfigError("GEMINI_API_KEY is not configured on the server.")
    monkeypatch.setattr("src.server.generate_captions", mock_config_err)

    handler2 = DummyHandler('/api/generate_captions', req_body)
    handler2.do_POST()
    assert handler2.response_code == 400
    res_err = json.loads(handler2.wfile.getvalue().decode("utf-8"))
    assert "GEMINI_API_KEY is not configured" in res_err["error"]

    # 3. Quota error
    def mock_quota_err(*args, **kwargs):
        raise GeminiQuotaError("Gemini limit reached. Try again later.")
    monkeypatch.setattr("src.server.generate_captions", mock_quota_err)

    handler3 = DummyHandler('/api/generate_captions', req_body)
    handler3.do_POST()
    assert handler3.response_code == 429
    res_quota = json.loads(handler3.wfile.getvalue().decode("utf-8"))
    assert "Gemini limit reached. Try again later." in res_quota["error"]

def test_gemini_failure_does_not_affect_normal_rendering(monkeypatch):
    """Confirm normal video rendering functions continue working even if Gemini fails or has no key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    # Mock subprocess.run for render_main_video
    captured_cmd = None
    def mock_run(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    from src.video_ops import render_main_video
    render_main_video(
        source_video="dummy_in.mp4",
        output_video="dummy_out.mp4",
        caption_overlay="dummy_overlay.png",
        start_time=0.0,
        end_time=5.0,
        crop_x=0,
        crop_y=0,
        crop_size=1080
    )
    assert captured_cmd is not None
    assert "-filter_complex" in captured_cmd

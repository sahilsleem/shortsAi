"""
tests/test_ai_router.py - Test suite for Multi-Provider AI Fallback System.

Tests cover:
1. Gemini success
2. Gemini 500 -> retry -> success
3. Gemini 500 -> retry -> Groq success
4. Gemini 429 -> immediately Groq (no retry on quota exhaustion)
5. Gemini timeout -> Groq
6. Gemini malformed JSON -> Groq
7. Groq failure -> Cerebras
8. Cerebras failure -> OpenRouter
9. All providers fail -> AllProvidersFailedError
10. Incomplete package rejected (missing titles, top_titles, description, or <10 captions)
11. Valid complete package accepted
12. Provider session-level cooldown behavior
13. API keys never appear in logs or exceptions
14. Frontend-compatible normalized response
15. Existing thumbnail_phrase preserved and sanitized
16. Retry/previous-generation avoidance forwarded
17. Configurable provider order via SHORTSAI_AI_PROVIDERS
18. Provider name populated strictly by router (untrusted provider spoofing prevented)
"""

import os
import sys
import json
import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.ai_router import (
    generate_content_with_fallback,
    validate_content_package,
    normalize_package,
    cooldown_tracker,
    AIProviderError,
    AIAuthError,
    AIQuotaError,
    AITransientError,
    AITimeoutError,
    AINetworkError,
    AIMalformedResponseError,
    AIIncompletePackageError,
    AllProvidersFailedError,
    PROVIDER_REGISTRY,
    get_configured_provider_order,
    DEFAULT_PROVIDER_ORDER,
    sanitize_log_text,
    format_structural_summary,
)


def make_valid_package(prefix: str = "Test") -> dict:
    """Creates a complete, valid Saba Bollywood publishing package with 10 caption pairs, 10 titles, 3 top titles."""
    return {
        "captions": [
            {
                "main_caption": f"{prefix} Salman was at airport #{i}, but then...",
                "curiosity_caption": f"he stopped to greet fan #{i} with a warm smile"
            }
            for i in range(1, 11)
        ],
        "titles": [
            f"{prefix} Salman Khan Moment #{i} Shocked Everyone"
            for i in range(1, 11)
        ],
        "top_titles": [
            f"{prefix} Top Title 1",
            f"{prefix} Top Title 2",
            f"{prefix} Top Title 3"
        ],
        "thumbnail_phrase": f"{prefix.upper()} DID THIS 😳",
        "description": f"{prefix} Salman Khan stopped to meet a fan at Mumbai airport. #Shorts #Bollywood"
    }


@pytest.fixture(autouse=True)
def reset_cooldown():
    """Ensure in-memory cooldown state is fresh for each test."""
    cooldown_tracker.reset()
    yield
    cooldown_tracker.reset()


def test_gemini_success(monkeypatch):
    """Test 1: Gemini succeeds on first attempt, returning normalized package with provider='gemini'."""
    expected = make_valid_package("Gemini")
    mock_gen = MagicMock(return_value=expected)
    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gen)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini"])

    assert result["provider"] == "gemini"
    assert result["main_caption"] == expected["captions"][0]["main_caption"]
    assert result["curiosity_caption"] == expected["captions"][0]["curiosity_caption"]
    assert result["thumbnail_phrase"] == "GEMINI DID THIS 😳"
    assert len(result["captions"]) == 10
    assert len(result["titles"]) == 10
    assert len(result["top_titles"]) == 3
    assert "description" in result
    assert mock_gen.call_count == 1


def test_gemini_transient_retry_success(monkeypatch):
    """Test 2: Gemini 500 on first try, retry succeeds."""
    expected = make_valid_package("Gemini")
    attempts = 0

    def mock_gen(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AITransientError("Gemini 500 Internal Error", provider="gemini", status_code=500)
        return expected

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gen)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini"])

    assert result["provider"] == "gemini"
    assert attempts == 2


def test_gemini_transient_retry_fails_then_groq_success(monkeypatch):
    """Test 3: Gemini 500 fails both attempts, router falls back to Groq which succeeds."""
    groq_pkg = make_valid_package("Groq")

    mock_gemini = MagicMock(side_effect=AITransientError("503 Service Unavailable", provider="gemini", status_code=503))
    mock_groq = MagicMock(return_value=groq_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    assert result["provider"] == "groq"
    assert result["main_caption"] == groq_pkg["captions"][0]["main_caption"]
    assert mock_gemini.call_count == 2  # initial + 1 retry
    assert mock_groq.call_count == 1


def test_gemini_429_quota_immediately_groq(monkeypatch):
    """Test 4: Gemini 429 quota exhaustion immediately advances to Groq without retrying Gemini."""
    groq_pkg = make_valid_package("Groq")

    mock_gemini = MagicMock(side_effect=AIQuotaError("Gemini quota exhausted (429)", provider="gemini", status_code=429))
    mock_groq = MagicMock(return_value=groq_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    assert result["provider"] == "groq"
    assert mock_gemini.call_count == 1  # Exactly 1 call, NO retry on quota!
    assert mock_groq.call_count == 1
    # Gemini should now be recorded in cooldown
    assert not cooldown_tracker.is_available("gemini")


def test_gemini_timeout_falls_back_to_groq(monkeypatch):
    """Test 5: Gemini timeout exhausts its retry and falls back to Groq."""
    groq_pkg = make_valid_package("Groq")

    mock_gemini = MagicMock(side_effect=AITimeoutError("Gemini timed out", provider="gemini", status_code=408))
    mock_groq = MagicMock(return_value=groq_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    assert result["provider"] == "groq"
    assert mock_gemini.call_count == 2
    assert mock_groq.call_count == 1


def test_gemini_malformed_json_moves_immediately_to_groq(monkeypatch):
    """Test 6: Gemini returning malformed JSON advances to Groq without repeated retries."""
    groq_pkg = make_valid_package("Groq")

    mock_gemini = MagicMock(side_effect=AIMalformedResponseError("Invalid JSON", provider="gemini"))
    mock_groq = MagicMock(return_value=groq_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    assert result["provider"] == "groq"
    assert mock_gemini.call_count == 1
    assert mock_groq.call_count == 1


def test_groq_failure_falls_back_to_cerebras(monkeypatch):
    """Test 7: Groq fails, router advances to Cerebras."""
    cerebras_pkg = make_valid_package("Cerebras")

    mock_gemini = MagicMock(side_effect=AIAuthError("No Gemini key", provider="gemini"))
    mock_groq = MagicMock(side_effect=AITransientError("Groq 500", provider="groq", status_code=500))
    mock_cerebras = MagicMock(return_value=cerebras_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)
    monkeypatch.setattr(PROVIDER_REGISTRY["cerebras"], "generate", mock_cerebras)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq", "cerebras"])

    assert result["provider"] == "cerebras"
    assert mock_cerebras.call_count == 1


def test_cerebras_failure_falls_back_to_openrouter(monkeypatch):
    """Test 8: Cerebras fails, router advances to OpenRouter."""
    openrouter_pkg = make_valid_package("OpenRouter")

    mock_gemini = MagicMock(side_effect=AIAuthError("No Gemini key", provider="gemini"))
    mock_groq = MagicMock(side_effect=AIQuotaError("Groq 429", provider="groq"))
    mock_cerebras = MagicMock(side_effect=AITransientError("Cerebras 503", provider="cerebras"))
    mock_openrouter = MagicMock(return_value=openrouter_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)
    monkeypatch.setattr(PROVIDER_REGISTRY["cerebras"], "generate", mock_cerebras)
    monkeypatch.setattr(PROVIDER_REGISTRY["openrouter"], "generate", mock_openrouter)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq", "cerebras", "openrouter"])

    assert result["provider"] == "openrouter"
    assert mock_openrouter.call_count == 1


def test_all_providers_fail_raises_all_providers_failed_error(monkeypatch):
    """Test 9: All 4 providers fail, AllProvidersFailedError is raised with user-friendly message."""
    mock_gemini = MagicMock(side_effect=AIAuthError("No key", provider="gemini"))
    mock_groq = MagicMock(side_effect=AIQuotaError("Groq quota", provider="groq"))
    mock_cerebras = MagicMock(side_effect=AITransientError("Cerebras 500", provider="cerebras"))
    mock_openrouter = MagicMock(side_effect=AINetworkError("Connection refused", provider="openrouter"))

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)
    monkeypatch.setattr(PROVIDER_REGISTRY["cerebras"], "generate", mock_cerebras)
    monkeypatch.setattr(PROVIDER_REGISTRY["openrouter"], "generate", mock_openrouter)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        generate_content_with_fallback("Salman at airport")

    assert "AI generation is temporarily unavailable" in str(exc_info.value)


def test_incomplete_package_rejected_and_router_advances(monkeypatch):
    """
    Test 10: If a provider returns only (main_caption, curiosity_caption, thumbnail_phrase)
    omitting titles, top_titles, or description, it is classified as AIIncompletePackageError
    and the router advances to the next provider.
    """
    incomplete_gemini = {
        "main_caption": "Salman walked down the street, but then...",
        "curiosity_caption": "he spotted an elderly fan",
        "thumbnail_phrase": "SALMAN STOPPED 😳"
        # Missing: captions list of 10, titles, top_titles, description!
    }
    complete_groq = make_valid_package("Groq")

    mock_gemini = MagicMock(return_value=incomplete_gemini)
    mock_groq = MagicMock(return_value=complete_groq)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    # Router must reject incomplete Gemini response and succeed with Groq
    assert result["provider"] == "groq"
    assert len(result["captions"]) == 10
    assert len(result["titles"]) == 10
    assert len(result["top_titles"]) == 3


def test_fewer_than_10_captions_rejected():
    """Verify validate_content_package strictly enforces exactly 10 caption pairs."""
    pkg = make_valid_package("Test")
    pkg["captions"] = pkg["captions"][:9]  # 9 instead of 10
    is_valid, reason = validate_content_package(pkg)
    assert not is_valid
    assert "Expected exactly 10 caption pairs" in reason


def test_fewer_than_10_titles_rejected():
    """Verify validate_content_package strictly enforces exactly 10 titles."""
    pkg = make_valid_package("Test")
    pkg["titles"] = pkg["titles"][:8]  # 8 instead of 10
    is_valid, reason = validate_content_package(pkg)
    assert not is_valid
    assert "Expected exactly 10 titles" in reason


def test_fewer_than_3_top_titles_rejected():
    """Verify validate_content_package strictly enforces exactly 3 top_titles."""
    pkg = make_valid_package("Test")
    pkg["top_titles"] = ["Top 1", "Top 2"]  # 2 instead of 3
    is_valid, reason = validate_content_package(pkg)
    assert not is_valid
    assert "Expected exactly 3 top_titles" in reason


def test_valid_complete_package_accepted():
    """Test 11: Valid complete publishing package passes validation cleanly."""
    pkg = make_valid_package("Valid")
    is_valid, reason = validate_content_package(pkg)
    assert is_valid
    assert reason == ""


def test_provider_cooldown_skips_without_network_call(monkeypatch):
    """Test 12: Provider with active cooldown is skipped immediately without making an API call."""
    groq_pkg = make_valid_package("Groq")

    mock_gemini = MagicMock()
    mock_groq = MagicMock(return_value=groq_pkg)

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    # Put Gemini in cooldown
    cooldown_tracker.record_quota_cooldown("gemini", duration=600.0)

    result = generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq"])

    assert result["provider"] == "groq"
    assert mock_gemini.call_count == 0  # Gemini was skipped entirely!


def test_api_keys_never_appear_in_logs(monkeypatch, capsys):
    """Test 13: Secrets like GEMINI_API_KEY or GROQ_API_KEY are never printed in logs."""
    fake_key = "SECRET_SUPER_CONFIDENTIAL_KEY_9999"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)

    mock_gemini = MagicMock(side_effect=AIAuthError("Invalid key", provider="gemini"))
    mock_groq = MagicMock(side_effect=AIQuotaError(f"Quota error for key {fake_key}", provider="groq"))
    mock_cerebras = MagicMock(side_effect=AIAuthError("No key", provider="cerebras"))
    mock_openrouter = MagicMock(side_effect=AIAuthError("No key", provider="openrouter"))

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)
    monkeypatch.setattr(PROVIDER_REGISTRY["cerebras"], "generate", mock_cerebras)
    monkeypatch.setattr(PROVIDER_REGISTRY["openrouter"], "generate", mock_openrouter)

    with pytest.raises(AllProvidersFailedError):
        generate_content_with_fallback("Salman at airport")

    captured = capsys.readouterr()
    assert fake_key not in captured.out
    assert fake_key not in captured.err


def test_frontend_compatible_normalized_response():
    """Test 14: Normalized response contains all fields expected by frontend."""
    raw = make_valid_package("Frontend")
    normalized = normalize_package(raw, "groq")

    # Frontend expectations:
    assert "main_caption" in normalized
    assert "curiosity_caption" in normalized
    assert "thumbnail_phrase" in normalized
    assert "provider" in normalized
    assert normalized["provider"] == "groq"
    assert normalized["main_caption"] == raw["captions"][0]["main_caption"]
    assert normalized["curiosity_caption"] == raw["captions"][0]["curiosity_caption"]


def test_thumbnail_phrase_sanitization_and_preservation():
    """Test 15: Thumbnail phrase is sanitized (no hashtags, quotes) and preserved."""
    raw = make_valid_package("Thumb")
    raw["thumbnail_phrase"] = '"SALMAN DID THIS" #Viral #Bollywood'
    normalized = normalize_package(raw, "cerebras")

    assert normalized["thumbnail_phrase"] == "SALMAN DID THIS"
    assert "#" not in normalized["thumbnail_phrase"]
    assert '"' not in normalized["thumbnail_phrase"]


def test_retry_avoidance_forwarded_to_prompt(monkeypatch):
    """Test 16: Previous generations are formatted into prompt avoidance list."""
    captured_prompt = None

    def mock_gen(context, previous_generations=None):
        nonlocal captured_prompt
        from src.ai_router import build_user_prompt
        captured_prompt = build_user_prompt(context, previous_generations)
        return make_valid_package("Groq")

    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_gen)

    prev = [
        {"main_caption": "Old Main 1", "curiosity_caption": "Old Reveal 1", "thumbnail_phrase": "OLD THUMB 1"},
        {"main_caption": "Old Main 2", "curiosity_caption": "Old Reveal 2", "thumbnail_phrase": "OLD THUMB 2"}
    ]

    generate_content_with_fallback("Context", previous_generations=prev, provider_order=["groq"])

    assert captured_prompt is not None
    assert "DO NOT repeat" in captured_prompt
    assert 'Main="Old Main 1"' in captured_prompt
    assert 'Thumbnail="OLD THUMB 1"' in captured_prompt


def test_configurable_provider_order(monkeypatch):
    """Test 17: Provider order can be customized via SHORTSAI_AI_PROVIDERS env var."""
    monkeypatch.setenv("SHORTSAI_AI_PROVIDERS", "openrouter,cerebras,groq,gemini")
    order = get_configured_provider_order()
    assert order == ["openrouter", "cerebras", "groq", "gemini"]

    monkeypatch.delenv("SHORTSAI_AI_PROVIDERS", raising=False)
    order_def = get_configured_provider_order()
    assert order_def == DEFAULT_PROVIDER_ORDER


def test_provider_name_never_trusted_from_api(monkeypatch):
    """Test 18: Populated provider name always matches router-dispatched provider, preventing spoofing."""
    spoofed_package = make_valid_package("Spoof")
    spoofed_package["provider"] = "fake_spoofed_provider"

    mock_groq = MagicMock(return_value=spoofed_package)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)

    result = generate_content_with_fallback("Context", provider_order=["groq"])

    assert result["provider"] == "groq"
    assert result["provider"] != "fake_spoofed_provider"


def test_groq_headers_user_agent_and_stripped_quotes(monkeypatch):
    """Test 19: Groq requests include User-Agent and strip surrounding quotes from GROQ_API_KEY."""
    from src.ai_router import DEFAULT_USER_AGENT
    captured_req = None

    valid_pkg = make_valid_package("Groq")
    resp_bytes = json.dumps({"choices": [{"message": {"content": json.dumps(valid_pkg)}}]}).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=None):
        nonlocal captured_req
        captured_req = req
        return MockHTTPResponse(resp_bytes)

    monkeypatch.setenv("GROQ_API_KEY", '"gsk_live_test_key_123"')
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    pkg = PROVIDER_REGISTRY["groq"].generate("Salman at airport")

    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]
    assert captured_req is not None
    # User-Agent must be present to prevent Cloudflare 403
    assert captured_req.headers.get("User-agent") == DEFAULT_USER_AGENT or captured_req.headers.get("User-Agent") == DEFAULT_USER_AGENT
    # Key must be stripped of quotes
    assert captured_req.headers.get("Authorization") == "Bearer gsk_live_test_key_123"


def test_groq_cloudflare_403_maps_to_auth_error(monkeypatch):
    """Test 20: Groq 403 error from Cloudflare or endpoint maps cleanly to AIAuthError with status 403."""
    import urllib.error

    def mock_urlopen(req, timeout=None):
        fp = io.BytesIO(b"<html><title>Just a moment...</title>Cloudflare 403 Forbidden</html>")
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, fp)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    with pytest.raises(AIAuthError) as exc_info:
        PROVIDER_REGISTRY["groq"].generate("Salman at airport")

    assert exc_info.value.status_code == 403
    assert "groq" in str(exc_info.value)


def test_openrouter_payload_and_headers(monkeypatch):
    """Test 21: OpenRouter requests include User-Agent, referer, X-Title, and require_parameters: true."""
    from src.ai_router import DEFAULT_USER_AGENT
    captured_req = None

    valid_pkg = make_valid_package("OpenRouter")
    resp_bytes = json.dumps({"choices": [{"message": {"content": json.dumps(valid_pkg)}}]}).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=None):
        nonlocal captured_req
        captured_req = req
        return MockHTTPResponse(resp_bytes)

    monkeypatch.setenv("OPENROUTER_API_KEY", "'sk-or-test-key-456'")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    pkg = PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")

    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]
    assert captured_req is not None
    assert captured_req.headers.get("Authorization") == "Bearer sk-or-test-key-456"
    assert captured_req.headers.get("Http-referer") == "https://github.com/sahilsleem/shortsAi"
    assert captured_req.headers.get("X-title") == "ShortsAI"

    payload = json.loads(captured_req.data.decode("utf-8"))
    assert payload.get("provider", {}).get("require_parameters") is True
    assert payload.get("max_tokens") == 4096


def test_openrouter_empty_content_with_reasoning_fallback(monkeypatch):
    """Test 22: If content is empty/null but reasoning_content contains JSON package, extracts it successfully."""
    valid_pkg = make_valid_package("Reasoning")
    resp_bytes = json.dumps({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "",
                "reasoning_content": f"I will now formulate the response:\n{json.dumps(valid_pkg)}"
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    pkg = PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")
    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]


def test_openrouter_multipart_content_concatenation(monkeypatch):
    """Test 23: If content is returned as a list of parts, it is joined and parsed."""
    valid_pkg = make_valid_package("Multipart")
    json_str = json.dumps(valid_pkg)
    mid = len(json_str) // 2

    resp_bytes = json.dumps({
        "choices": [{
            "message": {
                "content": [
                    {"type": "text", "text": json_str[:mid]},
                    {"type": "text", "text": json_str[mid:]}
                ]
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    pkg = PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")
    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]


def test_openrouter_refusal_raises_incomplete_package(monkeypatch):
    """Test 24: If model returns a refusal, it is raised as AIIncompletePackageError."""
    resp_bytes = json.dumps({
        "choices": [{
            "message": {
                "content": None,
                "refusal": "I cannot fulfill this request."
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    with pytest.raises(AIIncompletePackageError) as exc_info:
        PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")

    assert "model refusal" in str(exc_info.value)


def test_openrouter_empty_content_retries_without_response_format(monkeypatch):
    """Test 25: If initial call with response_format returns empty content, retries without response_format."""
    valid_pkg = make_valid_package("RetryNoFormat")
    call_count = 0

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        body = json.loads(req.data.decode("utf-8"))
        if "response_format" in body:
            # Model returns empty content when response_format is forced
            empty_resp = json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
            return MockHTTPResponse(empty_resp)
        else:
            # Model succeeds without response_format
            success_resp = json.dumps({"choices": [{"message": {"content": json.dumps(valid_pkg)}}]}).encode("utf-8")
            return MockHTTPResponse(success_resp)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    pkg = PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")
    assert call_count == 2
    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]


def test_openrouter_conversational_preamble_extracted(monkeypatch):
    """Test 26: Conversational text around JSON is handled by regex fallback."""
    valid_pkg = make_valid_package("Conversational")
    noisy_content = f"Here is the requested package:\n```json\n{json.dumps(valid_pkg)}\n```\nHope this helps!"
    resp_bytes = json.dumps({"choices": [{"message": {"content": noisy_content}}]}).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    pkg = PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")
    assert pkg["captions"][0]["main_caption"] == valid_pkg["captions"][0]["main_caption"]


def test_openrouter_top_level_error_classified(monkeypatch):
    """Test 27: OpenRouter 200 OK containing top-level error dict is classified into AIQuotaError or AIAuthError."""
    rate_limit_resp = json.dumps({"error": {"code": 429, "message": "Rate limit reached"}}).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(rate_limit_resp))

    with pytest.raises(AIQuotaError) as exc_info:
        PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")

    assert exc_info.value.status_code == 429


def test_phone_diagnostic_scenario_reproduced_and_handled(monkeypatch):
    """
    Test 28: Reproduce the exact diagnostic sequence from the real phone:
    1. Gemini: 429 Quota Exceeded
    2. Groq: Authentication/Configuration Error
    3. OpenRouter: Malformed/Empty Content
    Router tries all 3 in order, skips none, and reports clean AllProvidersFailedError with last error.
    """
    mock_gemini = MagicMock(side_effect=AIQuotaError("Gemini quota exhausted (429)", provider="gemini", status_code=429))
    mock_groq = MagicMock(side_effect=AIAuthError("groq authentication failed (403)", provider="groq", status_code=403))
    mock_openrouter = MagicMock(side_effect=AIMalformedResponseError("openrouter returned empty message content", provider="openrouter"))

    monkeypatch.setattr(PROVIDER_REGISTRY["gemini"], "generate", mock_gemini)
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", mock_groq)
    monkeypatch.setattr(PROVIDER_REGISTRY["openrouter"], "generate", mock_openrouter)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        generate_content_with_fallback("Salman at airport", provider_order=["gemini", "groq", "openrouter"])

    err_str = str(exc_info.value)
    assert "AI generation is temporarily unavailable" in err_str
    assert "openrouter returned empty message content" in err_str
    assert mock_gemini.call_count == 1
    assert mock_groq.call_count == 1
    assert mock_openrouter.call_count == 1


def test_reasoning_fallback_rejects_rambling_non_json_text(monkeypatch):
    """Test 29: Reasoning text is never accepted merely because it is non-empty; non-JSON raises AIMalformedResponseError."""
    rambling_resp = json.dumps({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "",
                "reasoning_content": "I am thinking deeply about Bollywood movies and Salman Khan at the airport..."
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(rambling_resp))

    with pytest.raises(AIMalformedResponseError) as exc_info:
        PROVIDER_REGISTRY["openrouter"].generate("Salman at airport")

    assert "empty message content" in str(exc_info.value)


def test_reasoning_fallback_validates_package_and_rejects_incomplete(monkeypatch):
    """Test 30: Valid JSON extracted from reasoning must still pass complete package validation."""
    incomplete_pkg = {
        "captions": [{"main_caption": "Only one caption", "curiosity_caption": "Incomplete"}],
        "titles": ["Title 1"]
        # Missing required 10 captions, 10 titles, 3 top titles, etc.
    }
    incomplete_reasoning_resp = json.dumps({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "",
                "reasoning_content": f"Here is the partial output:\n{json.dumps(incomplete_pkg)}"
            }
        }]
    }).encode("utf-8")

    valid_groq = make_valid_package("Groq")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(incomplete_reasoning_resp))
    monkeypatch.setattr(PROVIDER_REGISTRY["groq"], "generate", MagicMock(return_value=valid_groq))

    # OpenRouter returns incomplete package in reasoning -> router must reject it and fall back to Groq
    result = generate_content_with_fallback("Salman at airport", provider_order=["openrouter", "groq"])
    assert result["provider"] == "groq"
    assert len(result["captions"]) == 10


def test_reasoning_fallback_never_exposes_internal_thoughts(monkeypatch):
    """Test 31: Normalized package never leaks or exposes raw internal reasoning text."""
    secret_thoughts = "SECRET_INTERNAL_CHAIN_OF_THOUGHT_DO_NOT_LEAK"
    valid_pkg = make_valid_package("Safe")
    resp_bytes = json.dumps({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": "",
                "reasoning_content": f"{secret_thoughts}\n```json\n{json.dumps(valid_pkg)}\n```"
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    result = generate_content_with_fallback("Salman at airport", provider_order=["openrouter"])
    assert result["provider"] == "openrouter"
    # Ensure secret thoughts are nowhere in the output dictionary
    assert secret_thoughts not in json.dumps(result)
    assert "reasoning" not in result
    assert "reasoning_content" not in result
    assert "thought" not in result


def test_sanitize_log_text_masks_keys_and_redacts(monkeypatch):
    """Test 32: sanitize_log_text redacts configured env keys and generic key patterns, and truncates."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real_secret_key_abcdef123456")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real_secret_token_987654")

    raw_text = "Error calling with gsk_real_secret_key_abcdef123456 and Bearer sk-or-real_secret_token_987654"
    sanitized = sanitize_log_text(raw_text)

    assert "gsk_real_secret_key" not in sanitized
    assert "sk-or-real_secret" not in sanitized
    assert "[REDACTED_GROQ_API_KEY]" in sanitized or "[REDACTED_GROQ_KEY]" in sanitized
    assert "[REDACTED_OPENROUTER_API_KEY]" in sanitized or "[REDACTED_TOKEN]" in sanitized

    long_text = "x" * 600
    assert len(sanitize_log_text(long_text, max_len=200)) <= 203


def test_format_structural_summary_outputs_sanitized_keys_and_counts():
    """Test 33: format_structural_summary outputs safe structural metadata without dumping contents."""
    data = {
        "id": "gen-1234",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "private thoughts should not leak in summary",
            }
        }],
        "usage": {"total_tokens": 150}
    }
    summary = format_structural_summary(data, status_code=200)

    assert "HTTP 200" in summary
    assert "top_keys=['choices', 'id', 'usage']" in summary
    assert "choices_count=1" in summary
    assert "finish_reason='stop'" in summary
    assert "content_len=0" in summary
    assert "has_reasoning=True" in summary
    assert "private thoughts" not in summary


def test_groq_http_error_logs_status_and_body_snippet(monkeypatch, capsys):
    """Test 34: When Groq returns an HTTP error (e.g. 400 Bad Request), logs status code and sanitized body snippet."""
    import urllib.error

    def mock_urlopen(req, timeout=None):
        fp = io.BytesIO(b'{"error":{"message":"Failed to parse body: unrecognized field","type":"invalid_request_error"}}')
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, fp)

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    with pytest.raises(AllProvidersFailedError):
        generate_content_with_fallback("Salman at airport", provider_order=["groq"])

    captured = capsys.readouterr()
    assert "[Groq] HTTP 400 (AIProviderError):" in captured.out
    assert "Failed to parse body" in captured.out
    assert "Provider Error" in captured.out


def test_openrouter_malformed_empty_content_logs_structural_summary(monkeypatch, capsys):
    """Test 35: When OpenRouter returns empty message content, logs structural summary to console."""
    resp_bytes = json.dumps({
        "id": "gen-test-555",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": ""
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: MockHTTPResponse(resp_bytes))

    with pytest.raises(AllProvidersFailedError):
        generate_content_with_fallback("Salman at airport", provider_order=["openrouter"])

    captured = capsys.readouterr()
    assert "[Openrouter] Empty message content. Response structure: HTTP 200" in captured.out
    assert "choices_count=1" in captured.out
    assert "content_len=0" in captured.out
    assert "finish_reason='stop'" in captured.out

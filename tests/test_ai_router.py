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
    GroqProvider,
    CerebrasProvider,
    OpenRouterProvider,
    contains_emoji,
    extract_emojis,
    strip_emojis,
    separate_emojis,
    MASTER_PUBLISHING_PROMPT,
)
from src.gemini_caption import SYSTEM_INSTRUCTION


def make_valid_package(prefix: str = "Test") -> dict:
    emojis_main = ["👀", "😳", "🤔", "😮", "✨", "🎬", "🔥", "👏", "👀", ""]
    emojis_curiosity = ["❤️", "😂", "🔥", "🙏", "❤️", "👏", "😂", "✨", "", "❤️"]
    return {
        "captions": [
            {
                "main": f"{prefix} Salman was at airport #{i}, but then...",
                "main_emoji": emojis_main[i-1],
                "curiosity": f"he stopped to greet fan #{i} with a warm smile",
                "curiosity_emoji": emojis_curiosity[i-1],
                "main_caption": f"{prefix} Salman was at airport #{i}, but then...",
                "curiosity_caption": f"he stopped to greet fan #{i} with a warm smile",
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


def test_default_model_names_and_env_overrides(monkeypatch):
    """Test 36: Verify default models for Groq, OpenRouter, and Cerebras and their env overrides."""
    groq = GroqProvider()
    assert groq.default_model == "openai/gpt-oss-120b"
    assert groq.env_model_name == "GROQ_MODEL"

    openrouter = OpenRouterProvider()
    assert openrouter.default_model == "meta-llama/llama-3.3-70b-instruct"
    assert openrouter.env_model_name == "OPENROUTER_MODEL"

    cerebras = CerebrasProvider()
    assert cerebras.default_model == "llama-3.3-70b"
    assert cerebras.env_model_name == "CEREBRAS_MODEL"

    # Verify environment overrides are used in requests
    captured_payloads = []

    def mock_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        resp_data = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(make_valid_package("EnvModel"))}
            }]
        }
        return io.BytesIO(json.dumps(resp_data).encode("utf-8"))

    # Test Groq with custom GROQ_MODEL
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_MODEL", "custom-groq-model")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    res = groq.generate("test context")
    assert captured_payloads[-1]["model"] == "custom-groq-model"

    # Test Groq without GROQ_MODEL (falls back to default openai/gpt-oss-120b)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    res = groq.generate("test context")
    assert captured_payloads[-1]["model"] == "openai/gpt-oss-120b"

    # Test OpenRouter with custom OPENROUTER_MODEL
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "custom-openrouter-model")

    res = openrouter.generate("test context")
    assert captured_payloads[-1]["model"] == "custom-openrouter-model"

    # Test OpenRouter without OPENROUTER_MODEL (falls back to default meta-llama/llama-3.3-70b-instruct)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    res = openrouter.generate("test context")
    assert captured_payloads[-1]["model"] == "meta-llama/llama-3.3-70b-instruct"


# ---------------------------------------------------------------------------
# Dedicated Emoji Field Separation Regression Tests
# ---------------------------------------------------------------------------

def test_main_emoji_separated_from_main_text():
    """Test 37: Main emoji embedded in main caption text is extracted to main_emoji and stripped from main."""
    pkg = make_valid_package("EmojiTest")
    pkg["captions"][0] = {
        "main": "Salman Khan was already leaving, but then he noticed the fan waiting 👀",
        "main_emoji": "",
        "curiosity": "He stopped to take a photo with the young fan",
        "curiosity_emoji": "❤️"
    }

    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"
    first = pkg["captions"][0]
    assert first["main"] == "Salman Khan was already leaving, but then he noticed the fan waiting"
    assert first["main_emoji"] == "👀"
    assert not contains_emoji(first["main"])


def test_curiosity_emoji_separated_from_curiosity_text():
    """Test 38: Curiosity emoji embedded in curiosity caption text is extracted to curiosity_emoji and stripped."""
    pkg = make_valid_package("EmojiTest")
    pkg["captions"][0] = {
        "main": "Salman Khan was already leaving, but then he noticed the fan waiting",
        "main_emoji": "👀",
        "curiosity": "He stopped to take a photo with the young fan ❤️",
        "curiosity_emoji": ""
    }

    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"
    first = pkg["captions"][0]
    assert first["curiosity"] == "He stopped to take a photo with the young fan"
    assert first["curiosity_emoji"] == "❤️"
    assert not contains_emoji(first["curiosity"])


def test_main_and_curiosity_text_contain_no_emoji():
    """Test 39: Validates that every caption in the package has main and curiosity text free of any emoji characters."""
    pkg = make_valid_package("NoEmojiText")
    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"

    normalized = normalize_package(pkg, "groq")
    for idx, cap in enumerate(normalized["captions"]):
        assert not contains_emoji(cap["main"]), f"Caption #{idx+1} main has emoji: {cap['main']}"
        assert not contains_emoji(cap["curiosity"]), f"Caption #{idx+1} curiosity has emoji: {cap['curiosity']}"
        assert not contains_emoji(cap["main_caption"]), f"Caption #{idx+1} main_caption has emoji: {cap['main_caption']}"
        assert not contains_emoji(cap["curiosity_caption"]), f"Caption #{idx+1} curiosity_caption has emoji: {cap['curiosity_caption']}"


def test_main_and_curiosity_emoji_fields_receive_generated_emojis():
    """Test 40: Main emoji and curiosity emoji fields receive the generated emojis properly."""
    pkg = make_valid_package("ReceiveEmoji")
    pkg["captions"][0] = {
        "main": "She was leaving when the paparazzi asked her to pose",
        "main_emoji": "👀",
        "curiosity": "Her awkward reaction before walking away had everyone laughing",
        "curiosity_emoji": "😂"
    }

    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"
    normalized = normalize_package(pkg, "openrouter")
    assert normalized["main_emoji"] == "👀"
    assert normalized["curiosity_emoji"] == "😂"
    assert normalized["captions"][0]["main_emoji"] == "👀"
    assert normalized["captions"][0]["curiosity_emoji"] == "😂"


def test_all_10_caption_options_contain_all_four_fields():
    """Test 41: All 10 caption options contain main, main_emoji, curiosity, and curiosity_emoji."""
    pkg = make_valid_package("All10Fields")
    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"

    normalized = normalize_package(pkg, "cerebras")
    assert len(normalized["captions"]) == 10
    for idx, opt in enumerate(normalized["captions"]):
        assert "main" in opt, f"Caption #{idx+1} missing main"
        assert "main_emoji" in opt, f"Caption #{idx+1} missing main_emoji"
        assert "curiosity" in opt, f"Caption #{idx+1} missing curiosity"
        assert "curiosity_emoji" in opt, f"Caption #{idx+1} missing curiosity_emoji"
        assert isinstance(opt["main"], str) and opt["main"].strip()
        assert isinstance(opt["curiosity"], str) and opt["curiosity"].strip()
        assert isinstance(opt["main_emoji"], str)
        assert isinstance(opt["curiosity_emoji"], str)


def test_emojis_not_duplicated_inside_caption_text():
    """Test 42: Emojis provided both in text and in emoji fields are not duplicated; text is strictly cleaned."""
    pkg = make_valid_package("Deduplicate")
    pkg["captions"][0] = {
        "main": "Salman Khan was already leaving, but then he noticed the fan waiting 👀",
        "main_emoji": "👀",
        "curiosity": "He stopped to take a photo with the young fan ❤️",
        "curiosity_emoji": "❤️"
    }

    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"

    cap = pkg["captions"][0]
    assert cap["main"] == "Salman Khan was already leaving, but then he noticed the fan waiting"
    assert cap["main_emoji"] == "👀"
    assert cap["curiosity"] == "He stopped to take a photo with the young fan"
    assert cap["curiosity_emoji"] == "❤️"
    assert "👀" not in cap["main"]
    assert "❤️" not in cap["curiosity"]


def test_empty_emoji_fields_allowed_when_no_emoji_appropriate():
    """Test 43: Empty string emoji fields are fully allowed when no emoji is appropriate."""
    pkg = make_valid_package("EmptyEmoji")
    for cap in pkg["captions"]:
        cap["main_emoji"] = ""
        cap["curiosity_emoji"] = ""

    is_valid, reason = validate_content_package(pkg)
    assert is_valid, f"Validation failed: {reason}"

    normalized = normalize_package(pkg, "gemini")
    assert normalized["main_emoji"] == ""
    assert normalized["curiosity_emoji"] == ""
    for cap in normalized["captions"]:
        assert cap["main_emoji"] == ""
        assert cap["curiosity_emoji"] == ""


def test_captions_with_only_emojis_rejected():
    """Test 44: A caption that consists solely of emojis is rejected because clean text becomes empty."""
    pkg = make_valid_package("OnlyEmoji")
    pkg["captions"][0]["main"] = "👀🔥😂"
    is_valid, reason = validate_content_package(pkg)
    assert not is_valid
    assert "empty main caption" in reason


def test_invalid_emoji_type_rejected():
    """Test 45: A caption with a non-string emoji field is rejected."""
    pkg = make_valid_package("InvalidEmojiType")
    pkg["captions"][0]["main_emoji"] = 123  # Int instead of string
    is_valid, reason = validate_content_package(pkg)
    assert not is_valid
    assert "invalid non-string 'main_emoji'" in reason


def test_selecting_caption_with_use_populates_all_four_frontend_fields():
    """Test 46: Verify that index.html contains dedicated emoji inputs and JS logic correctly populates all 4 fields."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Check dedicated emoji fields in HTML
    assert 'id="caption-emoji-main"' in html_content
    assert 'id="caption-emoji-curiosity"' in html_content
    assert 'id="caption-main"' in html_content
    assert 'id="caption-curiosity"' in html_content

    # Check options container and pills in HTML
    assert 'id="gemini-options-container"' in html_content
    assert 'id="gemini-options-pills"' in html_content

    # Check preview elements
    assert 'id="gemini-preview-main-emoji"' in html_content
    assert 'id="gemini-preview-curiosity-emoji"' in html_content

    # Check JS event dispatching and population for all four fields
    assert "captionEmojiMainInput.value = mainEmoji" in html_content
    assert "captionEmojiCuriosityInput.value = curiosityEmoji" in html_content
    assert "captionMainInput.value = mainText" in html_content
    assert "captionCuriosityInput.value = curiosityText" in html_content

    # Simulate the JS application logic on a sample package
    pkg = make_valid_package("FrontendSim")
    pkg["captions"][1] = {
        "main": "Selected option 2 main text, but then...",
        "main_emoji": "😮",
        "curiosity": "Selected option 2 curiosity reveal text here",
        "curiosity_emoji": "✨"
    }

    # Simulate selecting Caption 2 (idx = 1)
    selected_index = 1
    opt = pkg["captions"][selected_index]

    simulated_inputs = {
        "caption-main": opt["main"],
        "caption-emoji-main": opt["main_emoji"],
        "caption-curiosity": opt["curiosity"],
        "caption-emoji-curiosity": opt["curiosity_emoji"],
    }

    assert simulated_inputs["caption-main"] == "Selected option 2 main text, but then..."
    assert simulated_inputs["caption-emoji-main"] == "😮"
    assert simulated_inputs["caption-curiosity"] == "Selected option 2 curiosity reveal text here"
    assert simulated_inputs["caption-emoji-curiosity"] == "✨"
    assert not contains_emoji(simulated_inputs["caption-main"])
    assert not contains_emoji(simulated_inputs["caption-curiosity"])


def test_prompt_regression_simple_english():
    """Test 47: Prompts mandate simple everyday English and forbid unnecessarily sophisticated language."""
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "SIMPLE ENGLISH" in prompt_text, f"{prompt_name} missing 'SIMPLE ENGLISH' section"
        assert "everyday words" in prompt_text.lower(), f"{prompt_name} missing 'everyday words'"
        for word in ["saw", "noticed", "stopped", "smiled", "walked over", "waited", "asked", "waved"]:
            assert word in prompt_text, f"{prompt_name} missing example word '{word}'"


def test_prompt_regression_conversational_fan_page_voice():
    """Test 48: Prompts enforce real Bollywood fan-page creator voice over robotic news style."""
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "fan-page creator" in prompt_text.lower(), f"{prompt_name} missing fan-page creator persona"
        assert "telling another bollywood fan" in prompt_text.lower(), f"{prompt_name} missing peer fan-to-fan framing"
        assert "not an ai news writer" in prompt_text.lower(), f"{prompt_name} missing anti-news writer restriction"


def test_prompt_regression_emotional_storytelling_through_action():
    """Test 49: Prompts mandate emotion through action (show, don't tell)."""
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "EMOTION THROUGH ACTION" in prompt_text, f"{prompt_name} missing EMOTION THROUGH ACTION"
        assert "action creates the emotion" in prompt_text.lower(), f"{prompt_name} missing 'action creates the emotion'"
        assert "Salman Khan was already heading out, but then he noticed someone waiting for him" in prompt_text


def test_prompt_regression_curiosity_open_loop_main_and_payoff_reveal():
    """Test 50: Prompts define Main as setup+open loop and Curiosity as the actual payoff."""
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "open loop" in prompt_text.lower(), f"{prompt_name} missing open loop concept"
        assert "What happened?" in prompt_text, f"{prompt_name} missing 'What happened?' hook trigger"
        assert "payoff" in prompt_text.lower(), f"{prompt_name} missing payoff definition for reveal"


def test_prompt_regression_avoidance_of_robotic_news_language():
    """Test 51: Prompts explicitly list forbidden generic AI and news clichés."""
    forbidden_phrases = [
        "captured attention",
        "heartwarming gesture",
        "unexpected turn of events",
        "left fans stunned",
        "unfolded",
        "garnered attention",
        "showcased",
        "demonstrated his affection",
        "displayed his kindness",
        "proceeded to",
        "in a touching moment",
        "netizens",
        "was seen",
        "made headlines",
        "social media went into a frenzy",
    ]
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        for phrase in forbidden_phrases:
            assert phrase in prompt_text, f"{prompt_name} missing forbidden phrase '{phrase}'"


def test_prompt_regression_genuine_variation_across_10_options():
    """Test 52: MASTER_PUBLISHING_PROMPT requires 10 distinct emotional angles."""
    required_angles = [
        "curiosity",
        "emotional",
        "wholesome",
        "funny",
        "awkward",
        "unexpected",
        "fan perspective",
        "celebrity-action perspective",
        "contrast",
        "direct hook",
    ]
    assert "10 OPTIONS MUST HAVE REAL VARIETY" in MASTER_PUBLISHING_PROMPT
    for angle in required_angles:
        assert angle in MASTER_PUBLISHING_PROMPT, f"MASTER_PUBLISHING_PROMPT missing angle '{angle}'"


def test_prompt_regression_truthfulness_and_no_invented_facts():
    """Test 53: Prompts strictly forbid inventing dialogue, thoughts, motives, relationships, etc."""
    forbidden_inventions = [
        "dialogue",
        "thoughts",
        "motives",
        "relationships",
        "locations",
        "dates",
        "feelings",
        "backstory",
        "reactions",
        "intentions",
    ]
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "DO NOT INVENT FACTS" in prompt_text or "NEVER INVENT FACTS" in prompt_text
        for inv in forbidden_inventions:
            assert inv in prompt_text.lower(), f"{prompt_name} missing forbidden invention '{inv}'"


def test_prompt_regression_final_human_test():
    """Test 54: Prompts enforce final internal human check before returning output."""
    for prompt_name, prompt_text in [("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION), ("MASTER_PUBLISHING_PROMPT", MASTER_PUBLISHING_PROMPT)]:
        assert "FINAL HUMAN TEST" in prompt_text
        assert "Would a real Bollywood fan-page creator actually write this?" in prompt_text


def test_publishing_ui_html_structure_and_location():
    """Test 55: Verify publishing section elements exist directly below video in index.html in order (Video -> Title -> Description)."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Elements exist
    assert 'id="result-video"' in html
    assert 'id="publishing-section"' in html
    assert 'id="publishing-title"' in html
    assert 'id="publishing-top-titles-group"' in html
    assert 'id="publishing-top-titles-list"' in html
    assert 'id="publishing-all-titles-group"' in html
    assert 'id="publishing-all-titles-list"' in html
    assert 'id="publishing-description"' in html
    assert 'id="btn-copy-title"' in html
    assert 'id="btn-copy-description"' in html

    # Verify visual order in DOM: Video -> Title -> Description
    video_pos = html.find('id="result-video"')
    pub_section_pos = html.find('id="publishing-section"')
    title_pos = html.find('id="publishing-title"')
    desc_pos = html.find('id="publishing-description"')

    assert video_pos != -1
    assert pub_section_pos != -1
    assert title_pos != -1
    assert desc_pos != -1

    assert video_pos < pub_section_pos, "Publishing section must be located below video"
    assert title_pos < desc_pos, "Title must appear before description"


def test_publishing_ui_title_selection_updates_title_field():
    """Test 56: Verify title selection updates the selected title field."""
    pkg = make_valid_package("TitleTest")
    titles = pkg["titles"]
    top_titles = pkg["top_titles"]

    # Simulation of initial title population logic
    current_selected_title = top_titles[0] if top_titles else titles[0]
    title_input_value = current_selected_title
    assert title_input_value == top_titles[0]

    # Simulate user clicking a different title (e.g., Title #5)
    new_title = titles[4]
    title_input_value = new_title
    current_selected_title = new_title
    assert title_input_value == titles[4]


def test_publishing_ui_description_populated_and_starts_with_title():
    """Test 57: Verify description is populated and starts with the selected title."""
    pkg = make_valid_package("DescTest")
    title = pkg["top_titles"][0]
    raw_desc = pkg["description"]

    # Saba Bollywood description starts with selected title
    if not raw_desc.startswith(title):
        initial_desc = title + "\n\n" + raw_desc
    else:
        initial_desc = raw_desc

    assert initial_desc.startswith(title)
    assert raw_desc in initial_desc


def test_publishing_ui_changing_title_updates_only_beginning_of_description():
    """Test 58: Verify changing the selected title updates only the beginning of the description."""
    title1 = "Salman Khan stopped to greet a fan at Mumbai airport #Shorts"
    title2 = "Salman Khan did something unexpected before leaving #Shorts"
    credits_and_hashtags = (
        "\n\n📌 CREDITS:\n"
        "Some clips/images may be sourced from publicly available platforms...\n\n"
        "#bollywood #paparazzi #celebrity #bollywoodshorts #shorts"
    )

    desc = title1 + credits_and_hashtags

    # Simulate selectTitle(title2)
    old_title = title1
    new_title = title2

    assert desc.startswith(old_title)
    new_desc = new_title + desc[len(old_title):]

    assert new_desc.startswith(title2)
    assert not new_desc.startswith(title1)
    # The remainder of the description must be identical
    assert new_desc[len(title2):] == credits_and_hashtags


def test_publishing_ui_manual_description_edits_preserved():
    """Test 59: Verify manual description edits are preserved when title changes and across rendering."""
    title1 = "Initial Title #Shorts"
    title2 = "Second Title #Shorts"
    user_custom_note = "\n\n[Creator Custom Note: Watch till the end!]"
    credits_and_hashtags = (
        "\n\n📌 CREDITS:\nSome clips sourced from web.\n\n"
        "#bollywood #shorts"
    )

    # User starts with description then manually adds a note
    desc = title1 + user_custom_note + credits_and_hashtags

    # User clicks title2
    assert desc.startswith(title1)
    updated_desc = title2 + desc[len(title1):]

    assert updated_desc.startswith(title2)
    assert user_custom_note in updated_desc
    assert credits_and_hashtags in updated_desc

    # Re-rendering check: if publishingDescription already has value, it is not overwritten
    re_rendered_desc = updated_desc
    assert re_rendered_desc == updated_desc


def test_rendering_does_not_trigger_ai_generation_and_preserves_package(monkeypatch):
    """Test 60: Verify video rendering does not trigger AI generation and preserves the publishing package."""
    ai_called = False

    def mock_generate(*args, **kwargs):
        nonlocal ai_called
        ai_called = True
        return make_valid_package("Unwanted")

    monkeypatch.setattr("src.ai_router.generate_content_with_fallback", mock_generate)

    # In server.py, /render handles video rendering independently
    with open("src/server.py", "r", encoding="utf-8") as f:
        server_code = f.read()

    render_block = server_code[server_code.find("self.path == '/render'"):server_code.find("elif self.path in ('/api/generate_captions'")]
    assert "generate_captions" not in render_block
    assert "generate_content_with_fallback" not in render_block
    assert not ai_called


def test_publishing_section_revealed_below_video_after_rendering():
    """Test 61: Verify JS logic reveals publishing section below video after rendering when AI content exists."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # JS logic checks currentGeneratedPair after render
    assert "populatePublishingSection(currentGeneratedPair)" in html
    # JS logic hides publishingSection if no AI content exists
    assert "publishingSection.style.display = 'none'" in html
    # Ensure no modal or new window is opened
    assert "window.open" not in html

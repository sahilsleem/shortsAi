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

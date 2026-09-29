"""
src/ai_router.py - Multi-Provider AI Fallback System for Saba Bollywood Content Writer.

Provider Chain:
1. Gemini (primary)
2. Groq (Llama 3.3 70B via OpenAI-compatible endpoint)
3. Cerebras (Llama 3.3 70B via OpenAI-compatible endpoint)
4. OpenRouter (fallback router / free model tier)

Features:
- Standard library urllib only (zero third-party AI runtime dependencies).
- Strict Saba Bollywood Master Publishing Package validation:
  * Exactly 10 caption pairs (main_caption + curiosity_caption)
  * Exactly 10 YouTube titles
  * Exactly 3 top_titles
  * Non-empty thumbnail_phrase
  * Non-empty description
  * Populated provider name (never trusted from API)
- Session-level in-memory cooldown to avoid hammering rate-limited or broken providers.
- Safe logging without exposing secrets, API keys, or private prompts.
"""

import os
import json
import re
import time
import socket
import urllib.request
import urllib.error
from typing import Tuple, List, Dict, Any, Optional

from src.gemini_caption import (
    SYSTEM_INSTRUCTION,
    clean_json_text,
    sanitize_thumbnail_phrase,
    GeminiConfigError,
    GeminiQuotaError,
    GeminiAPIError,
)

# ---------------------------------------------------------------------------
# Error Classification Hierarchy
# ---------------------------------------------------------------------------

class AIProviderError(Exception):
    """Base exception for all AI provider errors."""
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AIAuthError(AIProviderError):
    """Authentication or configuration error (missing key, 401, 403 invalid key)."""
    pass


class AIQuotaError(AIProviderError):
    """Rate limit or quota exhausted (429, RESOURCE_EXHAUSTED)."""
    pass


class AITransientError(AIProviderError):
    """Server-side transient error (500, 502, 503, 504)."""
    pass


class AITimeoutError(AIProviderError):
    """Request timeout (408 or socket timeout)."""
    pass


class AINetworkError(AIProviderError):
    """Network connection failure (URLError, connection reset)."""
    pass


class AIMalformedResponseError(AIProviderError):
    """Invalid, unparseable, or empty JSON response."""
    pass


class AIIncompletePackageError(AIProviderError):
    """Package missing required fields or failing quality constraints."""
    pass


class AllProvidersFailedError(Exception):
    """Raised when every configured provider in the fallback chain has failed."""
    pass


# ---------------------------------------------------------------------------
# Master Saba Bollywood Publishing Prompt
# ---------------------------------------------------------------------------

MASTER_PUBLISHING_PROMPT = SYSTEM_INSTRUCTION + """

FULL SABA BOLLYWOOD PUBLISHING PACKAGE REQUIREMENTS:
Generate a complete social publishing package for YouTube Shorts:
1. "captions": Exactly 10 distinct, high-impact caption pairs.
   - Each pair must contain "main_caption" and "curiosity_caption".
   - Each "main_caption" must have a comma separating the setup and hook, formatted for balanced two-line visual composition.
   - Each "curiosity_caption" must complete the reveal with enough substance for two lines.
   - All 10 pairs must explore genuinely different angles, hooks, and rhythms without repeating.
2. "titles": Exactly 10 punchy, viral YouTube Shorts titles (emotional, curiosity-driven, factual).
3. "top_titles": Exactly 3 best recommended title selections from the 10 titles.
4. "thumbnail_phrase": Exactly 1 short visual hook (2-6 words, max 7 words) for the cover thumbnail frame.
5. "description": An engaging 2-4 sentence YouTube Shorts description sticking strictly to context facts, ending with #Shorts #Bollywood.

OUTPUT FORMAT:
Respond with ONLY valid JSON with this exact schema:
{
  "captions": [
    {
      "main_caption": "Setup with comma, hook phrase here...",
      "curiosity_caption": "Resolution completing the action with substance here"
    }
  ],
  "titles": [
    "Title 1", "Title 2", "Title 3", "Title 4", "Title 5",
    "Title 6", "Title 7", "Title 8", "Title 9", "Title 10"
  ],
  "top_titles": [
    "Top Title 1", "Top Title 2", "Top Title 3"
  ],
  "thumbnail_phrase": "SHORT VISUAL HOOK 😳",
  "description": "Engaging context-accurate description text here... #Shorts #Bollywood"
}"""


def build_user_prompt(context: str, previous_generations: list = None) -> str:
    """Constructs user context prompt with retry avoidance list if present."""
    prompt_parts = [
        f"Creator's Video Context:\n{context.strip()}"
    ]

    if previous_generations and len(previous_generations) > 0:
        avoidance_list = []
        for i, gen in enumerate(previous_generations[-6:], 1):
            m = gen.get("main_caption", "")
            c = gen.get("curiosity_caption", "")
            t = gen.get("thumbnail_phrase", "")
            if m or c or t:
                entry = f"Previous #{i}: Main=\"{m}\" | Reveal=\"{c}\""
                if t:
                    entry += f" | Thumbnail=\"{t}\""
                avoidance_list.append(entry)

        if avoidance_list:
            prompt_parts.append(
                "\nDO NOT repeat or closely mimic any of these previous generations. "
                "Provide a meaningfully different angle, distinct hook wording, and fresh rhythm:\n"
                + "\n".join(avoidance_list)
            )

    return "\n\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Strict Package Validation & Normalization
# ---------------------------------------------------------------------------

def validate_content_package(package: dict) -> Tuple[bool, str]:
    """
    Validates that the returned package contains the complete Saba Bollywood Publishing Package:
    1. Exactly 10 caption pairs (every pair has non-empty main_caption and curiosity_caption).
    2. Exactly 10 YouTube titles (every title is non-empty string).
    3. Exactly 3 top_titles (every title is non-empty string).
    4. thumbnail_phrase is non-empty string.
    5. description is non-empty string.
    Returns (is_valid, error_reason).
    """
    if not isinstance(package, dict):
        return False, "Package is not a dictionary"

    # 1. Captions: exactly 10 pairs
    captions = package.get("captions")
    if not isinstance(captions, list):
        return False, "Missing or invalid 'captions' list"
    if len(captions) != 10:
        return False, f"Expected exactly 10 caption pairs, got {len(captions)}"
    for idx, pair in enumerate(captions):
        if not isinstance(pair, dict):
            return False, f"Caption pair #{idx+1} is not a dictionary"
        main = str(pair.get("main_caption", "")).strip()
        curiosity = str(pair.get("curiosity_caption", "")).strip()
        if not main:
            return False, f"Caption pair #{idx+1} has empty main_caption"
        if not curiosity:
            return False, f"Caption pair #{idx+1} has empty curiosity_caption"
        if "{" in main or "}" in main or "{" in curiosity or "}" in curiosity:
            return False, f"Caption pair #{idx+1} contains unparsed JSON syntax"

    # 2. Titles: exactly 10
    titles = package.get("titles")
    if not isinstance(titles, list):
        return False, "Missing or invalid 'titles' list"
    if len(titles) != 10:
        return False, f"Expected exactly 10 titles, got {len(titles)}"
    for idx, title in enumerate(titles):
        if not isinstance(title, str) or not title.strip():
            return False, f"Title #{idx+1} is empty or not a string"

    # 3. Top titles: exactly 3
    top_titles = package.get("top_titles")
    if not isinstance(top_titles, list):
        return False, "Missing or invalid 'top_titles' list"
    if len(top_titles) != 3:
        return False, f"Expected exactly 3 top_titles, got {len(top_titles)}"
    for idx, top_title in enumerate(top_titles):
        if not isinstance(top_title, str) or not top_title.strip():
            return False, f"Top title #{idx+1} is empty or not a string"

    # 4. Thumbnail phrase: non-empty string
    thumb = str(package.get("thumbnail_phrase", "")).strip()
    if not thumb:
        return False, "Missing or empty 'thumbnail_phrase'"

    # 5. Description: non-empty string
    desc = str(package.get("description", "")).strip()
    if not desc:
        return False, "Missing or empty 'description'"

    return True, ""


def normalize_package(data: dict, provider_name: str) -> dict:
    """
    Normalizes a validated publishing package:
    - Sets provider to provider_name (never trusting any provider-supplied value).
    - Sets top-level main_caption and curiosity_caption from the first caption pair
      for complete backward compatibility with the existing frontend.
    - Sanitizes thumbnail_phrase.
    - Preserves captions, titles, top_titles, and description.
    """
    captions = data["captions"]
    thumb_phrase = sanitize_thumbnail_phrase(str(data["thumbnail_phrase"]))
    if not thumb_phrase:
        thumb_phrase = str(data["thumbnail_phrase"]).strip()

    return {
        "provider": provider_name,
        "main_caption": captions[0]["main_caption"],
        "curiosity_caption": captions[0]["curiosity_caption"],
        "thumbnail_phrase": thumb_phrase,
        "captions": captions,
        "titles": [str(t).strip() for t in data["titles"]],
        "top_titles": [str(t).strip() for t in data["top_titles"]],
        "description": str(data["description"]).strip()
    }


# ---------------------------------------------------------------------------
# Session-Level In-Memory Cooldown Tracker
# ---------------------------------------------------------------------------

class ProviderCooldownTracker:
    """
    Thread-safe in-memory cooldown tracker:
    - Quota/rate-limit (429): 10 minutes cooldown.
    - Auth/missing key (401/403): 30 minutes cooldown.
    - Transient 5xx/network (after retrying): 1 minute cooldown.
    - Success: clears cooldown.
    """
    def __init__(self):
        self._cooldowns: Dict[str, float] = {}

    def is_available(self, provider_name: str) -> bool:
        until = self._cooldowns.get(provider_name, 0.0)
        return time.time() >= until

    def record_quota_cooldown(self, provider_name: str, duration: float = 600.0):
        self._cooldowns[provider_name] = time.time() + duration

    def record_auth_cooldown(self, provider_name: str, duration: float = 1800.0):
        self._cooldowns[provider_name] = time.time() + duration

    def record_transient_cooldown(self, provider_name: str, duration: float = 60.0):
        self._cooldowns[provider_name] = time.time() + duration

    def clear_cooldown(self, provider_name: str):
        self._cooldowns.pop(provider_name, None)

    def reset(self):
        self._cooldowns.clear()


cooldown_tracker = ProviderCooldownTracker()


# ---------------------------------------------------------------------------
# Provider Adapters
# ---------------------------------------------------------------------------

class BaseAIProvider:
    """Abstract base class for AI provider adapters."""
    name: str = "base"

    def generate(self, context: str, previous_generations: list = None) -> dict:
        raise NotImplementedError


class GeminiProvider(BaseAIProvider):
    """Adapter for Google Gemini API (gemini-3.8-flash default)."""
    name = "gemini"

    def generate(self, context: str, previous_generations: list = None) -> dict:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise AIAuthError("GEMINI_API_KEY is not configured.", provider=self.name)

        model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

        user_prompt = build_user_prompt(context, previous_generations)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": MASTER_PUBLISHING_PROMPT + "\n\n" + user_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "low"
                },
                "responseMimeType": "application/json",
                "maxOutputTokens": 2048
            }
        }
        req_body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=req_body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_data = json.loads(resp_body)
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                err_msg = err_data.get("error", {}).get("message", "")
            except Exception:
                pass

            if e.code in (400, 401, 403, 404):
                if e.code in (400, 403) and ("API_KEY" in err_msg or "key" in err_msg.lower()):
                    raise AIAuthError(f"Invalid GEMINI_API_KEY ({e.code}): {err_msg}", provider=self.name, status_code=e.code)
                raise AIAuthError(f"Gemini client/auth error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

            if e.code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                raise AIQuotaError(f"Gemini quota exhausted ({e.code}): {err_msg}", provider=self.name, status_code=429)

            if e.code == 408:
                raise AITimeoutError(f"Gemini timeout ({e.code})", provider=self.name, status_code=408)

            if 500 <= e.code < 600:
                raise AITransientError(f"Gemini server error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

            raise AIProviderError(f"Gemini HTTP error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
                raise AITimeoutError(f"Gemini network timeout: {e.reason}", provider=self.name)
            raise AINetworkError(f"Gemini network error: {e.reason}", provider=self.name)
        except (TimeoutError, socket.timeout):
            raise AITimeoutError("Gemini request timed out", provider=self.name)
        except Exception as e:
            raise AIProviderError(f"Gemini communication error: {str(e)}", provider=self.name)

        candidates = resp_data.get("candidates", [])
        if not candidates:
            raise AIMalformedResponseError("Gemini returned no candidates", provider=self.name)

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p and p.get("text")]
        if not text_parts:
            raise AIMalformedResponseError("Gemini returned no text content", provider=self.name)

        raw_text = clean_json_text("".join(text_parts))
        try:
            data = json.loads(raw_text)
        except Exception as e:
            raise AIMalformedResponseError(f"Gemini response could not be parsed as JSON: {str(e)}", provider=self.name)

        return data


class OpenAICompatibleProvider(BaseAIProvider):
    """Generic OpenAI-compatible chat completions provider adapter."""
    def __init__(
        self,
        name: str,
        endpoint: str,
        env_key_name: str,
        env_model_name: str,
        default_model: str,
        extra_headers: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.endpoint = endpoint
        self.env_key_name = env_key_name
        self.env_model_name = env_model_name
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

    def generate(self, context: str, previous_generations: list = None) -> dict:
        key = os.environ.get(self.env_key_name, "").strip()
        if not key:
            raise AIAuthError(f"{self.env_key_name} is not configured.", provider=self.name)

        model = os.environ.get(self.env_model_name, self.default_model).strip()
        user_prompt = build_user_prompt(context, previous_generations)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": MASTER_PUBLISHING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 2048
        }
        req_body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            **self.extra_headers
        }

        req = urllib.request.Request(
            self.endpoint,
            data=req_body,
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_data = json.loads(resp_body)
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                err_msg = err_data.get("error", {}).get("message", "")
            except Exception:
                pass

            if e.code in (401, 403):
                raise AIAuthError(f"{self.name} authentication failed ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

            if e.code == 429 or "rate_limit" in err_msg.lower() or "quota" in err_msg.lower():
                raise AIQuotaError(f"{self.name} rate limit / quota exceeded ({e.code}): {err_msg}", provider=self.name, status_code=429)

            if e.code == 408:
                raise AITimeoutError(f"{self.name} timeout ({e.code})", provider=self.name, status_code=408)

            if 500 <= e.code < 600:
                raise AITransientError(f"{self.name} server error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

            if e.code in (400, 404):
                if "key" in err_msg.lower() or "auth" in err_msg.lower():
                    raise AIAuthError(f"{self.name} auth/config error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)
                raise AIProviderError(f"{self.name} client error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

            raise AIProviderError(f"{self.name} HTTP error ({e.code}): {err_msg}", provider=self.name, status_code=e.code)

        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
                raise AITimeoutError(f"{self.name} network timeout: {e.reason}", provider=self.name)
            raise AINetworkError(f"{self.name} network error: {e.reason}", provider=self.name)
        except (TimeoutError, socket.timeout):
            raise AITimeoutError(f"{self.name} request timed out", provider=self.name)
        except Exception as e:
            raise AIProviderError(f"{self.name} communication error: {str(e)}", provider=self.name)

        choices = resp_data.get("choices", [])
        if not choices:
            raise AIMalformedResponseError(f"{self.name} returned no choices", provider=self.name)

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            raise AIMalformedResponseError(f"{self.name} returned empty message content", provider=self.name)

        raw_text = clean_json_text(content)
        try:
            data = json.loads(raw_text)
        except Exception as e:
            raise AIMalformedResponseError(f"{self.name} content could not be parsed as JSON: {str(e)}", provider=self.name)

        return data


class GroqProvider(OpenAICompatibleProvider):
    """Adapter for Groq (default: llama-3.3-70b-versatile)."""
    def __init__(self):
        super().__init__(
            name="groq",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            env_key_name="GROQ_API_KEY",
            env_model_name="GROQ_MODEL",
            default_model="llama-3.3-70b-versatile"
        )


class CerebrasProvider(OpenAICompatibleProvider):
    """Adapter for Cerebras (default: llama-3.3-70b)."""
    def __init__(self):
        super().__init__(
            name="cerebras",
            endpoint="https://api.cerebras.ai/v1/chat/completions",
            env_key_name="CEREBRAS_API_KEY",
            env_model_name="CEREBRAS_MODEL",
            default_model="llama-3.3-70b"
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """Adapter for OpenRouter (default: openrouter/auto)."""
    def __init__(self):
        super().__init__(
            name="openrouter",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            env_key_name="OPENROUTER_API_KEY",
            env_model_name="OPENROUTER_MODEL",
            default_model="openrouter/auto",
            extra_headers={
                "HTTP-Referer": "https://github.com/sahilsleem/shortsAi",
                "X-Title": "ShortsAI"
            }
        )


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER_ORDER = ["gemini", "groq", "cerebras", "openrouter"]

PROVIDER_REGISTRY: Dict[str, BaseAIProvider] = {
    "gemini": GeminiProvider(),
    "groq": GroqProvider(),
    "cerebras": CerebrasProvider(),
    "openrouter": OpenRouterProvider()
}


def get_configured_provider_order() -> List[str]:
    """Retrieves provider sequence from SHORTSAI_AI_PROVIDERS or default."""
    raw = os.environ.get("SHORTSAI_AI_PROVIDERS", "").strip()
    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)
    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = [p for p in providers if p in PROVIDER_REGISTRY]
    return valid if valid else list(DEFAULT_PROVIDER_ORDER)


# ---------------------------------------------------------------------------
# Router Orchestrator
# ---------------------------------------------------------------------------

def generate_content_with_fallback(
    context: str,
    previous_generations: list = None,
    provider_order: list = None,
    max_transient_retries: int = 1
) -> dict:
    """
    Orchestrates multi-provider fallback for the Saba Bollywood Content Writer:
    - Tries providers in order (default: Gemini -> Groq -> Cerebras -> OpenRouter).
    - Session-level in-memory cooldown prevents pounding rate-limited providers.
    - Transient server errors (500/502/503/timeout) receive at most 1 short retry.
    - Quota (429), auth/config, malformed JSON, and incomplete packages immediately advance.
    - Validates that the full publishing package (10 captions, 10 titles, 3 top titles,
      thumbnail_phrase, description) is present and non-empty.
    - Injects the true provider name into the returned package.
    """
    if not context or not context.strip():
        raise ValueError("Video context description cannot be empty.")

    providers = provider_order or get_configured_provider_order()
    last_error = None

    for provider_name in providers:
        provider = PROVIDER_REGISTRY.get(provider_name)
        if not provider:
            continue

        if not cooldown_tracker.is_available(provider_name):
            print(f"[AI Router] Skipping {provider_name.capitalize()} (in cooldown)")
            continue

        print(f"[AI Router] Trying {provider_name.capitalize()}")

        for attempt in range(max_transient_retries + 1):
            try:
                raw_data = provider.generate(context, previous_generations)

                # Strict validation of complete Saba Bollywood Publishing Package
                is_valid, reason = validate_content_package(raw_data)
                if not is_valid:
                    print(f"[{provider_name.capitalize()}] Incomplete package: {reason}")
                    raise AIIncompletePackageError(f"Incomplete package: {reason}", provider=provider_name)

                # Success: clear cooldown, normalize, and return
                cooldown_tracker.clear_cooldown(provider_name)
                print(f"[{provider_name.capitalize()}] Success")
                print(f"[AI Router] Generation completed with {provider_name.capitalize()}")
                return normalize_package(raw_data, provider_name)

            except AIQuotaError as e:
                print(f"[{provider_name.capitalize()}] 429 Quota Exceeded")
                cooldown_tracker.record_quota_cooldown(provider_name)
                last_error = e
                # Do NOT retry quota errors; immediately advance to next provider
                break

            except AIAuthError as e:
                print(f"[{provider_name.capitalize()}] Authentication/Configuration Error")
                cooldown_tracker.record_auth_cooldown(provider_name)
                last_error = e
                # Do NOT retry auth errors; disable for session and advance
                break

            except (AIMalformedResponseError, AIIncompletePackageError) as e:
                print(f"[{provider_name.capitalize()}] Invalid/Incomplete Response")
                last_error = e
                # Advance to next provider without retrying unparseable data
                break

            except (AITransientError, AITimeoutError, AINetworkError) as e:
                status_code = getattr(e, 'status_code', None)
                status_str = f" {status_code}" if status_code else ""
                print(f"[{provider_name.capitalize()}]{status_str} Transient Error")
                last_error = e

                if attempt < max_transient_retries:
                    print(f"[AI Router] Retrying {provider_name.capitalize()} once")
                    time.sleep(0.5)
                    continue
                else:
                    cooldown_tracker.record_transient_cooldown(provider_name)
                    break

            except Exception as e:
                print(f"[{provider_name.capitalize()}] Unexpected error")
                last_error = e
                break

        print(f"[AI Router] {provider_name.capitalize()} failed; trying next provider")

    print("[AI Router] All providers failed.")
    err_detail = f" Last error: {str(last_error)}" if last_error else ""
    raise AllProvidersFailedError(f"AI generation is temporarily unavailable. Please try again in a moment.{err_detail}")

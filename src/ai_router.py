"""
src/ai_router.py - Multi-Provider AI Fallback System for Saba Bollywood Content Writer.

Provider Chain:
1. Gemini (primary)
2. Groq (GPT OSS 120B via OpenAI-compatible endpoint)
3. Cerebras (Llama 3.3 70B via OpenAI-compatible endpoint)
4. OpenRouter (Llama 3.3 70B Instruct via OpenAI-compatible endpoint)

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
    contains_emoji,
    extract_emojis,
    strip_emojis,
    separate_emojis,
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
# Diagnostic Logging Helpers
# ---------------------------------------------------------------------------

def sanitize_log_text(text: Any, max_len: int = 500) -> str:
    """
    Sanitizes arbitrary text for safe console logging:
    - Never prints API keys or Bearer tokens.
    - Redacts any configured environment API keys.
    - Truncates to max_len characters.
    - Condenses whitespace to keep logs clean and single-line.
    """
    if not text:
        return ""
    s = str(text)
    # Redact configured API keys in environment
    for key_env in ("GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        val = os.environ.get(key_env, "").strip().strip("\"'")
        if val and len(val) >= 4 and val in s:
            s = s.replace(val, f"[REDACTED_{key_env}]")

    # Generic patterns
    s = re.sub(r'gsk_[A-Za-z0-9_\-]{6,}', '[REDACTED_GROQ_KEY]', s)
    s = re.sub(r'sk-or-[A-Za-z0-9_\-]{6,}', '[REDACTED_OPENROUTER_KEY]', s)
    s = re.sub(r'AIza[A-Za-z0-9_\-]{6,}', '[REDACTED_GEMINI_KEY]', s)
    s = re.sub(r'csk-[A-Za-z0-9_\-]{6,}', '[REDACTED_CEREBRAS_KEY]', s)
    s = re.sub(r'(Bearer\s+)[A-Za-z0-9_\-\.]{8,}', r'\1[REDACTED_TOKEN]', s)
    s = re.sub(r'(SECRET_[A-Za-z0-9_]+)', '[REDACTED_SECRET]', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def format_structural_summary(data: Any, status_code: int = 200) -> str:
    """
    Constructs a safe structural summary of an API response without leaking
    sensitive prompts, user context, or private internal reasoning text.
    """
    if not isinstance(data, dict):
        return f"HTTP {status_code} | data_type={type(data).__name__}"

    top_keys = sorted(list(data.keys()))
    has_error = "error" in data
    error_summary = ""
    if has_error:
        err = data.get("error")
        if isinstance(err, dict):
            err_code = err.get("code")
            err_type = err.get("type")
            err_msg = sanitize_log_text(err.get("message", ""), max_len=120)
            error_summary = f" (code={err_code}, type={err_type}, msg='{err_msg}')"
        else:
            error_summary = f" ({sanitize_log_text(str(err), max_len=120)})"

    choices = data.get("choices")
    choices_count = len(choices) if isinstance(choices, list) else 0

    first_choice_summary = []
    if choices_count > 0 and isinstance(choices[0], dict):
        fc = choices[0]
        finish_reason = fc.get("finish_reason")
        first_choice_summary.append(f"finish_reason={finish_reason!r}")
        msg = fc.get("message")
        if isinstance(msg, dict):
            msg_keys = sorted(list(msg.keys()))
            first_choice_summary.append(f"message_keys={msg_keys}")
            content = msg.get("content")
            content_type = type(content).__name__
            content_len = len(content) if hasattr(content, "__len__") else 0
            first_choice_summary.append(f"content_type={content_type}")
            first_choice_summary.append(f"content_len={content_len}")
            has_refusal = bool(msg.get("refusal"))
            first_choice_summary.append(f"has_refusal={has_refusal}")
            has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thought"))
            first_choice_summary.append(f"has_reasoning={has_reasoning}")
        else:
            first_choice_summary.append(f"message_type={type(msg).__name__}")
    elif choices is not None and not isinstance(choices, list):
        first_choice_summary.append(f"choices_type={type(choices).__name__}")

    fc_str = " | " + " | ".join(first_choice_summary) if first_choice_summary else ""
    return f"HTTP {status_code} | top_keys={top_keys} | choices_count={choices_count} | has_error={has_error}{error_summary}{fc_str}"



# ---------------------------------------------------------------------------
# Master Saba Bollywood Publishing Prompt
# ---------------------------------------------------------------------------

MASTER_PUBLISHING_PROMPT = SYSTEM_INSTRUCTION + """

FULL SABA BOLLYWOOD PUBLISHING PACKAGE REQUIREMENTS:
Generate a complete social publishing package for YouTube Shorts:
1. "captions": Exactly 10 distinct, high-impact caption pairs.
   - Each pair MUST contain four distinct fields:
     * "main": Main caption text (Setup + Hook). CLEAN TEXT ONLY. Absolutely NO emojis inside "main".
     * "main_emoji": Main caption emoji (0-2 emojis). Single string containing ONLY the intended emoji(s) (e.g. "👀" or "😳" or "🤔"). Empty string "" if no emoji fits.
     * "curiosity": Curiosity/Reveal caption text (The Reveal / Payoff). CLEAN TEXT ONLY. Absolutely NO emojis inside "curiosity".
     * "curiosity_emoji": Curiosity/Reveal caption emoji (0-2 emojis). Single string containing ONLY the intended emoji(s) (e.g. "❤️" or "😂" or "🔥"). Empty string "" if no emoji fits.
   - CRITICAL EMOJI SEPARATION:
     * Keep caption text and emojis COMPLETELY SEPARATE.
     * Do NOT put emojis into "main" or "curiosity".
     * Emojis belong ONLY in "main_emoji" and "curiosity_emoji".
     * Emojis must complement the sentence emotionally and naturally (e.g. surprise "👀", emotion "❤️", humor "😂").
     * Keep emoji strings short (0–2 emojis, usually one strong emoji). Never output long strings like "😂😂😂🔥🔥👀👀❤️❤️".
   - Each "main" must have a comma separating the setup and hook, formatted for balanced two-line visual composition.
   - Each "curiosity" must complete the reveal with enough substance for two lines.
   - 10 OPTIONS MUST HAVE REAL VARIETY:
     Do NOT create ten versions that merely replace words or rephrase the same sentence.
     Use genuinely different angles when supported by the context:
     1. curiosity (open-loop curiosity hook: makes viewer wonder what happened next)
     2. emotional (focusing on feelings, heartwarming interaction, or touching reaction)
     3. wholesome (warm, sweet, respectful interaction)
     4. funny (humorous, amusing, or playful angle)
     5. awkward (funny awkward moment, hesitation, or relatable pause)
     6. unexpected (element of surprise, sudden turn, or unexpected gesture)
     7. fan perspective (how it looked and felt from the fan's or crowd's point of view)
     8. celebrity-action perspective (what the celebrity specifically decided to do)
     9. contrast (expectation vs reality, or what could have happened vs what actually happened)
     10. direct hook (short, punchy, immediate setup)
     Do not force an angle that doesn't fit the context; choose natural angles supported by the clip.
   - Every one of the 10 options must have its own four fields and its own tailored emojis matching that specific option's emotional angle.
   - VOICE ACROSS ALL 10 OPTIONS:
     Write in a conversational, relatable Bollywood fan-page voice.
     Use simple everyday English (saw, noticed, stopped, smiled, looked back, walked over, asked, waved, etc.).
     Create emotion through action rather than telling the viewer how to feel.
     NEVER use robotic news clichés ("heartwarming gesture", "captured attention", "left fans stunned", "proceeded to", "netizens", "was seen", "made headlines", "social media went into a frenzy").
2. "titles": Exactly 10 punchy, viral YouTube Shorts titles (emotional, curiosity-driven, factual, simple English).
3. "top_titles": Exactly 3 best recommended title selections from the 10 titles.
4. "thumbnail_phrase": Exactly 1 short visual hook (2-6 words, max 7 words) for the cover thumbnail frame.
5. "description": An engaging 2-4 sentence YouTube Shorts description sticking strictly to context facts, ending with #Shorts #Bollywood.

FINAL HUMAN TEST:
Before returning the package, internally ask:
"Would a real Bollywood fan-page creator actually write this?"
If it sounds like a newspaper, generic AI, or overly complicated English, or if all ten options sound almost identical, rewrite them.

OUTPUT FORMAT:
Respond with ONLY valid JSON with this exact schema:
{
  "captions": [
    {
      "main": "Setup with comma, hook phrase here...",
      "main_emoji": "👀",
      "curiosity": "Resolution completing the action with substance here",
      "curiosity_emoji": "❤️"
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
            m = gen.get("main", "") or gen.get("main_caption", "")
            c = gen.get("curiosity", "") or gen.get("curiosity_caption", "")
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
    1. Exactly 10 caption pairs:
       - Every pair has non-empty 'main' and 'curiosity' text containing no emojis.
       - Every pair has 'main_emoji' and 'curiosity_emoji' as strings (empty string allowed).
       - Automatically normalizes embedded emojis from caption text into dedicated emoji fields when safely possible.
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

        has_main_key = "main" in pair or "main_caption" in pair
        has_curiosity_key = "curiosity" in pair or "curiosity_caption" in pair

        if not has_main_key:
            return False, f"Caption pair #{idx+1} missing 'main' caption"
        if not has_curiosity_key:
            return False, f"Caption pair #{idx+1} missing 'curiosity' caption"

        raw_main = pair.get("main") if "main" in pair else pair.get("main_caption")
        raw_curiosity = pair.get("curiosity") if "curiosity" in pair else pair.get("curiosity_caption")

        if not isinstance(raw_main, str):
            return False, f"Caption pair #{idx+1} has non-string 'main'"
        if not isinstance(raw_curiosity, str):
            return False, f"Caption pair #{idx+1} has non-string 'curiosity'"

        raw_main_emoji = pair.get("main_emoji", "")
        raw_curiosity_emoji = pair.get("curiosity_emoji", "")
        if not isinstance(raw_main_emoji, str):
            return False, f"Caption pair #{idx+1} has invalid non-string 'main_emoji'"
        if not isinstance(raw_curiosity_emoji, str):
            return False, f"Caption pair #{idx+1} has invalid non-string 'curiosity_emoji'"

        clean_main, embedded_m_emojis = separate_emojis(raw_main)
        clean_curiosity, embedded_c_emojis = separate_emojis(raw_curiosity)

        if not clean_main:
            return False, f"Caption pair #{idx+1} has empty main caption"
        if not clean_curiosity:
            return False, f"Caption pair #{idx+1} has empty curiosity caption"

        if contains_emoji(clean_main):
            return False, f"Caption pair #{idx+1} 'main' still contains emojis"
        if contains_emoji(clean_curiosity):
            return False, f"Caption pair #{idx+1} 'curiosity' still contains emojis"

        if "{" in clean_main or "}" in clean_main or "{" in clean_curiosity or "}" in clean_curiosity:
            return False, f"Caption pair #{idx+1} contains unparsed JSON syntax"

        final_main_emoji = extract_emojis(raw_main_emoji) or embedded_m_emojis
        final_curiosity_emoji = extract_emojis(raw_curiosity_emoji) or embedded_c_emojis

        # Update in-place so downstream callers receive normalized package
        pair["main"] = clean_main
        pair["main_emoji"] = final_main_emoji
        pair["curiosity"] = clean_curiosity
        pair["curiosity_emoji"] = final_curiosity_emoji
        pair["main_caption"] = clean_main
        pair["curiosity_caption"] = clean_curiosity

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
    - Sets top-level main, main_emoji, curiosity, curiosity_emoji, main_caption, and curiosity_caption
      from the first caption pair for complete backward compatibility.
    - Sanitizes thumbnail_phrase.
    - Preserves canonical captions (each containing main, main_emoji, curiosity, curiosity_emoji),
      titles, top_titles, and description.
    """
    captions = data["captions"]
    thumb_phrase = sanitize_thumbnail_phrase(str(data["thumbnail_phrase"]))
    if not thumb_phrase:
        thumb_phrase = str(data["thumbnail_phrase"]).strip()

    canonical_captions = []
    for pair in captions:
        clean_main = str(pair.get("main", "")).strip()
        clean_curiosity = str(pair.get("curiosity", "")).strip()
        main_emoji = str(pair.get("main_emoji", "")).strip()
        curiosity_emoji = str(pair.get("curiosity_emoji", "")).strip()
        canonical_captions.append({
            "main": clean_main,
            "main_emoji": main_emoji,
            "curiosity": clean_curiosity,
            "curiosity_emoji": curiosity_emoji,
            "main_caption": clean_main,
            "curiosity_caption": clean_curiosity,
        })

    first_cap = canonical_captions[0]
    return {
        "provider": provider_name,
        "main": first_cap["main"],
        "main_emoji": first_cap["main_emoji"],
        "curiosity": first_cap["curiosity"],
        "curiosity_emoji": first_cap["curiosity_emoji"],
        "main_caption": first_cap["main"],
        "curiosity_caption": first_cap["curiosity"],
        "thumbnail_phrase": thumb_phrase,
        "captions": canonical_captions,
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

DEFAULT_USER_AGENT = (
    "ShortsAI/1.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class BaseAIProvider:
    """Abstract base class for AI provider adapters."""
    name: str = "base"

    def generate(self, context: str, previous_generations: list = None) -> dict:
        raise NotImplementedError


class GeminiProvider(BaseAIProvider):
    """Adapter for Google Gemini API (gemini-3.8-flash default)."""
    name = "gemini"

    def generate(self, context: str, previous_generations: list = None) -> dict:
        key = os.environ.get("GEMINI_API_KEY", "").strip().strip("\"'")
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
                "User-Agent": DEFAULT_USER_AGENT,
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
            raw_snippet = ""
            try:
                raw_err = e.read().decode("utf-8", errors="replace")
                raw_snippet = sanitize_log_text(raw_err, max_len=500)
                try:
                    err_data = json.loads(raw_err)
                    if isinstance(err_data, dict):
                        err_obj = err_data.get("error")
                        if isinstance(err_obj, dict):
                            err_msg = err_obj.get("message", "")
                        elif isinstance(err_obj, str):
                            err_msg = err_obj
                except Exception:
                    clean_err = re.sub(r'<[^>]+>', ' ', raw_err[:300]).strip()
                    err_msg = clean_err[:120] if clean_err else str(e.reason)
            except Exception:
                err_msg = str(e.reason)
                raw_snippet = sanitize_log_text(str(e.reason), max_len=500)

            # Classify error
            if e.code in (400, 401, 403, 404) and ("API_KEY" in err_msg or "key" in err_msg.lower()):
                error_class = "AIAuthError"
            elif e.code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                error_class = "AIQuotaError"
            elif e.code == 408:
                error_class = "AITimeoutError"
            elif 500 <= e.code < 600:
                error_class = "AITransientError"
            else:
                error_class = "AIProviderError"

            print(f"[Gemini] HTTP {e.code} ({error_class}): {raw_snippet}")

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
            exc_type = type(e).__name__
            exc_msg = sanitize_log_text(str(e.reason))
            print(f"[Gemini] Network Exception ({exc_type}): {exc_msg}")
            if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
                raise AITimeoutError(f"Gemini network timeout: {e.reason}", provider=self.name)
            raise AINetworkError(f"Gemini network error: {e.reason}", provider=self.name)
        except (TimeoutError, socket.timeout) as e:
            print(f"[Gemini] Timeout Exception ({type(e).__name__}): {sanitize_log_text(str(e))}")
            raise AITimeoutError("Gemini request timed out", provider=self.name)
        except (AIProviderError, ValueError):
            raise
        except Exception as e:
            exc_type = type(e).__name__
            exc_msg = sanitize_log_text(str(e))
            print(f"[Gemini] Unexpected Exception ({exc_type}): {exc_msg}")
            raise AIProviderError(f"Gemini communication error ({exc_type}): {exc_msg}", provider=self.name)

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
        extra_headers: Optional[Dict[str, str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096
    ):
        self.name = name
        self.endpoint = endpoint
        self.env_key_name = env_key_name
        self.env_model_name = env_model_name
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.extra_payload = extra_payload or {}
        self.max_tokens = max_tokens

    def generate(self, context: str, previous_generations: list = None) -> dict:
        key = os.environ.get(self.env_key_name, "").strip().strip("\"'")
        if not key:
            raise AIAuthError(f"{self.env_key_name} is not configured.", provider=self.name)

        model = os.environ.get(self.env_model_name, self.default_model).strip()
        user_prompt = build_user_prompt(context, previous_generations)

        def _execute_http(include_json_format: bool) -> dict:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": MASTER_PUBLISHING_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": self.max_tokens,
                **self.extra_payload
            }
            if include_json_format:
                payload["response_format"] = {"type": "json_object"}

            req_body = json.dumps(payload).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
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
                with urllib.request.urlopen(req, timeout=25) as resp:
                    resp_body = resp.read().decode("utf-8")
                    return json.loads(resp_body)
            except urllib.error.HTTPError as e:
                err_msg = ""
                raw_snippet = ""
                try:
                    raw_err = e.read().decode("utf-8", errors="replace")
                    raw_snippet = sanitize_log_text(raw_err, max_len=500)
                    try:
                        err_data = json.loads(raw_err)
                        if isinstance(err_data, dict):
                            err_obj = err_data.get("error")
                            if isinstance(err_obj, dict):
                                err_msg = err_obj.get("message", "")
                            elif isinstance(err_obj, str):
                                err_msg = err_obj
                    except Exception:
                        clean_err = re.sub(r'<[^>]+>', ' ', raw_err[:300]).strip()
                        err_msg = clean_err[:120] if clean_err else str(e.reason)
                except Exception:
                    err_msg = str(e.reason)
                    raw_snippet = sanitize_log_text(str(e.reason), max_len=500)

                # Classify error
                if e.code in (401, 403):
                    error_class = "AIAuthError"
                elif e.code == 429 or "rate_limit" in err_msg.lower() or "quota" in err_msg.lower():
                    error_class = "AIQuotaError"
                elif e.code == 408:
                    error_class = "AITimeoutError"
                elif 500 <= e.code < 600:
                    error_class = "AITransientError"
                else:
                    error_class = "AIProviderError"

                print(f"[{self.name.capitalize()}] HTTP {e.code} ({error_class}): {raw_snippet}")

                # If 400 Bad Request indicates json_object response_format is unsupported, signal to retry without it
                if include_json_format and e.code == 400 and any(kw in err_msg.lower() for kw in ("response_format", "json_object", "schema", "parameter")):
                    raise ValueError(f"unsupported_response_format: {err_msg}")

                if e.code in (401, 403):
                    detail = err_msg or ("Forbidden / WAF block" if e.code == 403 else "Unauthorized")
                    raise AIAuthError(f"{self.name} authentication failed ({e.code}): {detail}", provider=self.name, status_code=e.code)

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
                exc_type = type(e).__name__
                exc_msg = sanitize_log_text(str(e.reason))
                print(f"[{self.name.capitalize()}] Network Exception ({exc_type}): {exc_msg}")
                if isinstance(e.reason, socket.timeout) or "timed out" in str(e.reason).lower():
                    raise AITimeoutError(f"{self.name} network timeout: {e.reason}", provider=self.name)
                raise AINetworkError(f"{self.name} network error: {e.reason}", provider=self.name)
            except (TimeoutError, socket.timeout) as e:
                print(f"[{self.name.capitalize()}] Timeout Exception ({type(e).__name__}): {sanitize_log_text(str(e))}")
                raise AITimeoutError(f"{self.name} request timed out", provider=self.name)
            except (AIProviderError, ValueError):
                raise
            except Exception as e:
                exc_type = type(e).__name__
                exc_msg = sanitize_log_text(str(e))
                print(f"[{self.name.capitalize()}] Execution Exception ({exc_type}): {exc_msg}")
                raise AIProviderError(f"{self.name} communication error ({exc_type}): {exc_msg}", provider=self.name)

        def _extract_choice_data(data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            """Extracts (content, refusal, finish_reason) from response dictionary."""
            # Check for top-level error object (common in proxy/aggregator 200 responses)
            if "error" in data and not data.get("choices"):
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                code = err.get("code") if isinstance(err, dict) else None
                print(f"[{self.name.capitalize()}] Top-level API error object present. Structure: {format_structural_summary(data)}")
                if code == 429 or "rate" in err_msg.lower() or "quota" in err_msg.lower():
                    raise AIQuotaError(f"{self.name} rate limit / quota exceeded: {err_msg}", provider=self.name, status_code=429)
                if code in (401, 403) or "auth" in err_msg.lower() or "key" in err_msg.lower():
                    raise AIAuthError(f"{self.name} authentication failed: {err_msg}", provider=self.name, status_code=code or 401)
                raise AIMalformedResponseError(f"{self.name} API error: {err_msg}", provider=self.name)

            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                return None, None, None

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return None, None, None

            finish_reason = first_choice.get("finish_reason")
            message = first_choice.get("message", {})
            if not isinstance(message, dict):
                return None, None, finish_reason

            refusal = message.get("refusal")
            if refusal:
                return None, str(refusal), finish_reason

            raw_content = message.get("content")
            extracted = None
            if isinstance(raw_content, list):
                # Multipart content list of dicts: [{"type": "text", "text": "..."}]
                extracted = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                )
            elif isinstance(raw_content, str) and raw_content.strip():
                extracted = raw_content
            elif raw_content is None or (isinstance(raw_content, str) and not raw_content.strip()):
                # Fallback: check reasoning or thought if content was empty
                reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thought")
                if reasoning and isinstance(reasoning, str) and "{" in reasoning and "}" in reasoning:
                    extracted = reasoning

            return extracted, None, finish_reason

        # Attempt 1: with response_format: {"type": "json_object"}
        used_response_format = True
        try:
            resp_data = _execute_http(include_json_format=True)
        except ValueError as ve:
            # Endpoint explicitly rejected response_format: {"type": "json_object"}
            print(f"[{self.name.capitalize()}] Initial call rejected response_format, retrying without it: {sanitize_log_text(str(ve))}")
            used_response_format = False
            resp_data = _execute_http(include_json_format=False)

        content, refusal, finish_reason = _extract_choice_data(resp_data)

        if refusal:
            print(f"[{self.name.capitalize()}] Model returned refusal. Structure: {format_structural_summary(resp_data)}")
            raise AIIncompletePackageError(f"{self.name} model refusal: {refusal}", provider=self.name)

        # Fallback attempt if response_format yielded empty/null content
        if (content is None or not content.strip()) and used_response_format:
            print(f"[{self.name.capitalize()}] Initial call yielded empty content with response_format. Retrying without response_format...")
            print(f"[{self.name.capitalize()}] Initial response structure: {format_structural_summary(resp_data)}")
            try:
                fallback_data = _execute_http(include_json_format=False)
                fb_content, fb_refusal, fb_finish = _extract_choice_data(fallback_data)
                if fb_refusal:
                    print(f"[{self.name.capitalize()}] Fallback model returned refusal. Structure: {format_structural_summary(fallback_data)}")
                    raise AIIncompletePackageError(f"{self.name} model refusal: {fb_refusal}", provider=self.name)
                if fb_content and fb_content.strip():
                    content = fb_content
                    resp_data = fallback_data
                else:
                    finish_reason = fb_finish or finish_reason
                    print(f"[{self.name.capitalize()}] Fallback without response_format also yielded empty content.")
                    print(f"[{self.name.capitalize()}] Fallback response structure: {format_structural_summary(fallback_data)}")
            except Exception as fb_exc:
                if isinstance(fb_exc, (AIProviderError, AIIncompletePackageError, AIMalformedResponseError)):
                    raise
                print(f"[{self.name.capitalize()}] Fallback request failed ({type(fb_exc).__name__}): {sanitize_log_text(str(fb_exc))}")

        if not content or not content.strip():
            print(f"[{self.name.capitalize()}] Empty message content. Response structure: {format_structural_summary(resp_data)}")
            reason_suffix = f" (finish_reason: {finish_reason})" if finish_reason else ""
            raise AIMalformedResponseError(f"{self.name} returned empty message content{reason_suffix}", provider=self.name)

        raw_text = clean_json_text(content)
        try:
            data = json.loads(raw_text)
        except Exception:
            # Fallback regex extraction for outermost JSON object if conversational text surrounds it
            match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except Exception as e:
                    print(f"[{self.name.capitalize()}] Regex JSON extraction failed ({type(e).__name__}). Response structure: {format_structural_summary(resp_data)}")
                    raise AIMalformedResponseError(f"{self.name} content could not be parsed as JSON: {str(e)}", provider=self.name)
            else:
                print(f"[{self.name.capitalize()}] Content is not valid JSON. Response structure: {format_structural_summary(resp_data)}")
                raise AIMalformedResponseError(f"{self.name} content could not be parsed as JSON: invalid format", provider=self.name)

        if not isinstance(data, dict):
            print(f"[{self.name.capitalize()}] Parsed JSON is not a dictionary ({type(data).__name__}). Response structure: {format_structural_summary(resp_data)}")
            raise AIMalformedResponseError(f"{self.name} content did not parse into a JSON object", provider=self.name)

        return data


class GroqProvider(OpenAICompatibleProvider):
    """Adapter for Groq (default: openai/gpt-oss-120b)."""
    def __init__(self):
        super().__init__(
            name="groq",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            env_key_name="GROQ_API_KEY",
            env_model_name="GROQ_MODEL",
            default_model="openai/gpt-oss-120b",
            max_tokens=4096
        )


class CerebrasProvider(OpenAICompatibleProvider):
    """Adapter for Cerebras (default: llama-3.3-70b)."""
    def __init__(self):
        super().__init__(
            name="cerebras",
            endpoint="https://api.cerebras.ai/v1/chat/completions",
            env_key_name="CEREBRAS_API_KEY",
            env_model_name="CEREBRAS_MODEL",
            default_model="llama-3.3-70b",
            max_tokens=4096
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """Adapter for OpenRouter (default: meta-llama/llama-3.3-70b-instruct)."""
    def __init__(self):
        super().__init__(
            name="openrouter",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            env_key_name="OPENROUTER_API_KEY",
            env_model_name="OPENROUTER_MODEL",
            default_model="meta-llama/llama-3.3-70b-instruct",
            extra_headers={
                "HTTP-Referer": "https://github.com/sahilsleem/shortsAi",
                "X-Title": "ShortsAI"
            },
            extra_payload={
                "provider": {
                    "require_parameters": True
                }
            },
            max_tokens=4096
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
                    if isinstance(raw_data, dict):
                        pkg_keys = sorted(list(raw_data.keys()))
                        c_count = len(raw_data["captions"]) if isinstance(raw_data.get("captions"), list) else 0
                        t_count = len(raw_data["titles"]) if isinstance(raw_data.get("titles"), list) else 0
                        top_count = len(raw_data["top_titles"]) if isinstance(raw_data.get("top_titles"), list) else 0
                        print(f"[{provider_name.capitalize()}] Package structure: keys={pkg_keys} | captions={c_count} | titles={t_count} | top_titles={top_count} | has_thumb={bool(raw_data.get('thumbnail_phrase'))} | has_desc={bool(raw_data.get('description'))}")
                    raise AIIncompletePackageError(f"Incomplete package: {reason}", provider=provider_name)

                # Success: clear cooldown, normalize, and return
                cooldown_tracker.clear_cooldown(provider_name)
                print(f"[{provider_name.capitalize()}] Success")
                print(f"[AI Router] Generation completed with {provider_name.capitalize()}")
                return normalize_package(raw_data, provider_name)

            except AIQuotaError as e:
                print(f"[{provider_name.capitalize()}] 429 Quota Exceeded: {sanitize_log_text(str(e))}")
                cooldown_tracker.record_quota_cooldown(provider_name)
                last_error = e
                # Do NOT retry quota errors; immediately advance to next provider
                break

            except AIAuthError as e:
                status_str = f" {e.status_code}" if getattr(e, 'status_code', None) else ""
                print(f"[{provider_name.capitalize()}]{status_str} Authentication/Configuration Error: {sanitize_log_text(str(e))}")
                cooldown_tracker.record_auth_cooldown(provider_name)
                last_error = e
                # Do NOT retry auth errors; disable for session and advance
                break

            except (AIMalformedResponseError, AIIncompletePackageError) as e:
                print(f"[{provider_name.capitalize()}] Invalid/Incomplete Response ({type(e).__name__}): {sanitize_log_text(str(e))}")
                last_error = e
                # Advance to next provider without retrying unparseable data
                break

            except (AITransientError, AITimeoutError, AINetworkError) as e:
                status_code = getattr(e, 'status_code', None)
                status_str = f" {status_code}" if status_code else ""
                print(f"[{provider_name.capitalize()}]{status_str} Transient Error ({type(e).__name__}): {sanitize_log_text(str(e))}")
                last_error = e

                if attempt < max_transient_retries:
                    print(f"[AI Router] Retrying {provider_name.capitalize()} once")
                    time.sleep(0.5)
                    continue
                else:
                    cooldown_tracker.record_transient_cooldown(provider_name)
                    break

            except AIProviderError as e:
                status_code = getattr(e, 'status_code', None)
                status_str = f" {status_code}" if status_code else ""
                print(f"[{provider_name.capitalize()}]{status_str} Provider Error ({type(e).__name__}): {sanitize_log_text(str(e))}")
                last_error = e
                break

            except Exception as e:
                print(f"[{provider_name.capitalize()}] Unexpected error ({type(e).__name__}): {sanitize_log_text(str(e))}")
                last_error = e
                break

        print(f"[AI Router] {provider_name.capitalize()} failed; trying next provider")

    print("[AI Router] All providers failed.")
    err_detail = f" Last error: {str(last_error)}" if last_error else ""
    raise AllProvidersFailedError(f"AI generation is temporarily unavailable. Please try again in a moment.{err_detail}")

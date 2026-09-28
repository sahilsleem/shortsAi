import os
import json
import re
import urllib.request
import urllib.error

class GeminiError(Exception):
    """Base exception for Gemini operations."""
    pass

class GeminiConfigError(GeminiError):
    """Raised when Gemini configuration (e.g. API key) is missing or invalid."""
    pass

class GeminiQuotaError(GeminiError):
    """Raised when API rate limits or quotas are exceeded."""
    pass

class GeminiAPIError(GeminiError):
    """Raised when Gemini API request fails or returns an unexpected response."""
    pass

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """You are the expert caption writer for Saba Bollywood YouTube Shorts.
Transform the creator's rough video description into a polished pair of high-impact captions:
1. main_caption (Setup + Hook)
2. curiosity_caption (The Reveal)

CRITICAL SABA BOLLYWOOD WRITING SPECIFICATION:
- Tone: Concise, punchy, natural Bollywood entertainment/paparazzi news tone.
- Curiosity: Create an irresistible open loop that makes viewers stay to watch the video.
- Factuality: Stick 100% strictly to the factual story supplied by the creator. NEVER invent events, dialogue, motives, emotions, private thoughts, unmentioned relationships, or fake details.
- No AI Fluff: Avoid generic AI phrases, long explanations, excessive adjectives, or clickbait nonsense.

FORMAT & STRUCTURE RULES:
1. MAIN CAPTION:
   - Must contain: SETUP, CURIOSITY HOOK
   - MUST include a comma ',' separating the setup and the hook.
   - Example: "Salman Khan Was Walking Down The Street, But Then..."
   - The setup establishes the scene. The hook opens the curiosity gap.
   - Vary the hook phrase across generations (e.g. "But Then...", "However...", "Meanwhile...", "And Then...", "Until...", "Then Suddenly...").
   - Do NOT reveal the conclusion in the Main caption.
   - Capitalize each major word (Title Case) suitable for clean video typography.

2. CURIOSITY / REVEAL CAPTION:
   - Completes the open loop and finishes the story.
   - Example: "He Spotted His Friend And Stopped To Shake Hands"
   - Do NOT repeat the main caption.
   - Capitalize each major word (Title Case).

3. LENGTH & VISUAL COMPOSITION:
   - Aim for approximately 35-65 characters per caption (soft target for perfect 2-line rendering on mobile).
   - Combined total around 90-120 characters (~6 to 12 words per caption).
   - Never write long paragraphs or 40+ word sentences.

OUTPUT FORMAT:
Respond with ONLY valid JSON with this exact schema:
{
  "main_caption": "Setup Goes Here, Hook Phrase Here...",
  "curiosity_caption": "Resolution Completing The Action Here"
}"""

def clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1 and lines[0].startswith("```"):
            lines = lines[1:]
        if len(lines) > 0 and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def parse_gemini_response(response_text: str) -> dict:
    """Parse and validate JSON response containing main_caption and curiosity_caption."""
    clean_text = clean_json_text(response_text)
    try:
        data = json.loads(clean_text)
    except Exception as e:
        # Fallback regex extraction if raw JSON parsing fails
        main_match = re.search(r'"main_caption"\s*:\s*"([^"]+)"', clean_text)
        curiosity_match = re.search(r'"curiosity_caption"\s*:\s*"([^"]+)"', clean_text)
        if main_match and curiosity_match:
            data = {
                "main_caption": main_match.group(1),
                "curiosity_caption": curiosity_match.group(1)
            }
        else:
            raise GeminiAPIError(f"Failed to parse structured JSON from Gemini response: {str(e)}")

    if not isinstance(data, dict):
        raise GeminiAPIError("Gemini response is not a valid JSON object.")

    main_cap = data.get("main_caption", "").strip()
    curiosity_cap = data.get("curiosity_caption", "").strip()

    if not main_cap or not curiosity_cap:
        raise GeminiAPIError("Gemini response is missing required caption fields.")

    return {
        "main_caption": main_cap,
        "curiosity_caption": curiosity_cap
    }

def build_gemini_payload(context: str, previous_generations: list = None) -> dict:
    """Construct Gemini API request payload with context and retry avoidance list."""
    prompt_parts = [
        f"Creator's Video Context:\n{context.strip()}"
    ]

    if previous_generations and len(previous_generations) > 0:
        avoidance_list = []
        for i, gen in enumerate(previous_generations[-6:], 1):
            m = gen.get("main_caption", "")
            c = gen.get("curiosity_caption", "")
            if m or c:
                avoidance_list.append(f"Previous #{i}: Main=\"{m}\" | Reveal=\"{c}\"")
        
        if avoidance_list:
            prompt_parts.append(
                "\nDO NOT repeat or closely mimic any of these previous generations. "
                "Provide a meaningfully different angle, distinct hook wording, and fresh rhythm:\n"
                + "\n".join(avoidance_list)
            )

    full_user_prompt = "\n\n".join(prompt_parts)

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": SYSTEM_INSTRUCTION + "\n\n" + full_user_prompt}
                ]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "low"
            },
            "responseMimeType": "application/json",
            "maxOutputTokens": 1024
        }
    }

def generate_captions(context: str, previous_generations: list = None, api_key: str = None, model: str = None) -> dict:
    """
    Generate a pair of (main_caption, curiosity_caption) using Google Gemini API.
    Uses standard library urllib without adding new dependencies.
    """
    if not context or not context.strip():
        raise ValueError("Video context description cannot be empty.")

    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiConfigError("GEMINI_API_KEY is not configured on the server. Please set the GEMINI_API_KEY environment variable.")

    gemini_model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={key}"

    payload = build_gemini_payload(context, previous_generations)
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_data = json.loads(resp_body)
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            err_msg = err_data.get("error", {}).get("message", "")
        except Exception:
            pass

        if e.code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise GeminiQuotaError("Gemini limit reached. Try again later.")
        elif e.code in (400, 403) and ("API_KEY" in err_msg or "key" in err_msg.lower()):
            raise GeminiConfigError("Invalid GEMINI_API_KEY configured.")
        else:
            detail = f": {err_msg}" if err_msg else ""
            raise GeminiAPIError(f"Gemini API request failed ({e.code}){detail}")
    except urllib.error.URLError as e:
        raise GeminiAPIError(f"Network error connecting to Gemini API: {str(e.reason)}")
    except Exception as e:
        raise GeminiAPIError(f"Failed to communicate with Gemini API: {str(e)}")

    candidates = resp_data.get("candidates", [])
    if not candidates:
        raise GeminiAPIError("Gemini returned no response candidates.")

    first_cand = candidates[0]
    parts = first_cand.get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p and p.get("text")]
    if not text_parts:
        raise GeminiAPIError("Gemini returned no text content.")

    raw_text = "".join(text_parts)
    return parse_gemini_response(raw_text)

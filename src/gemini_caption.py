import os
import json
import re
import time
import random
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

# ---------------------------------------------------------------------------
# Emoji Separation & Normalization Helpers
# ---------------------------------------------------------------------------

EMOJI_PATTERN = re.compile(
    r'['
    r'\U0001F600-\U0001F64F'  # Emoticons
    r'\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
    r'\U0001F680-\U0001F6FF'  # Transport and Map Symbols
    r'\U0001F700-\U0001F77F'  # Alchemical Symbols
    r'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
    r'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
    r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    r'\U0001FA00-\U0001FA6F'  # Chess Symbols
    r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    r'\U00002700-\U000027BF'  # Dingbats
    r'\U00002600-\U000026FF'  # Misc Symbols (including ❤️ with \uFE0F)
    r'\U00002300-\U000023FF'  # Misc Technical
    r'\U00002B50-\U00002B55'  # Stars, circles
    r'\U0000200D'              # Zero Width Joiner
    r'\U0000FE0E-\U0000FE0F'  # Variation Selectors
    r'\U0001F1E0-\U0001F1FF'  # Regional indicators (flags)
    r']+',
    re.UNICODE
)


def contains_emoji(text: str) -> bool:
    """Returns True if text contains any emoji characters."""
    if not text:
        return False
    return bool(EMOJI_PATTERN.search(str(text)))


def extract_emojis(text: str) -> str:
    """Extracts only the emoji characters from text."""
    if not text:
        return ""
    matches = EMOJI_PATTERN.findall(str(text))
    return "".join(matches).strip()


def strip_emojis(text: str) -> str:
    """Removes all emojis from text and cleans up whitespace and attached punctuation."""
    if not text:
        return ""
    cleaned = EMOJI_PATTERN.sub('', str(text))
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+([,!?\.])', r'\1', cleaned)
    return cleaned.strip()


def separate_emojis(text: str):
    """
    Separates a text into (clean_text, emojis).
    The clean_text contains NO emojis.
    The emojis string contains only extracted emoji characters.
    """
    if not text:
        return "", ""
    clean = strip_emojis(text)
    emojis = extract_emojis(text)
    return clean, emojis

SYSTEM_INSTRUCTION = """You are the expert caption writer for Saba Bollywood YouTube Shorts.
Transform the creator's rough video description into high-impact Shorts assets:
1. main (Setup + Hook - Clean text only)
2. main_emoji (0-2 emojis or empty string)
3. curiosity (The Reveal / Payoff - Clean text only)
4. curiosity_emoji (0-2 emojis or empty string)
5. thumbnail_phrase (Short visual hook for the cover thumbnail frame)

TARGET VOICE & PERSONA:
- Write like a real Bollywood fan-page creator who watched the video, NOT an AI news writer or robotic entertainment journalist.
- Think: "I watched this clip and I'm telling another Bollywood fan what made this moment interesting."
- Not: "I am generating entertainment-news copy about this event."
- Tone: emotional, human, simple, conversational, relatable, punchy, slightly dramatic when appropriate, curiosity-driven, natural, easy to understand instantly.

SIMPLE ENGLISH (HIGHEST PRIORITY RULE):
- Use everyday words.
- Prefer: saw, noticed, stopped, smiled, looked back, walked over, waited, asked, waved, hugged, laughed, reacted, was about to leave, then this happened, but then, and that's when.
- Avoid unnecessarily sophisticated, academic, or formal language.
- FORBIDDEN GENERIC AI / NEWS PHRASES (DO NOT USE AS FILLER):
  * captured attention
  * heartwarming gesture
  * unexpected turn of events
  * left fans stunned / left everyone stunned
  * unfolded
  * garnered attention
  * showcased
  * demonstrated his affection / demonstrated her affection
  * displayed his kindness / displayed her kindness
  * proceeded to
  * in a touching moment
  * the internet was left
  * netizens
  * was seen
  * made headlines
  * social media went into a frenzy

EMOTION THROUGH ACTION (SHOW, DON'T TELL):
- Do not simply tell the audience: "It was a heartwarming moment" or "He displayed his kindness."
- Show why it felt that way through physical actions. The action creates the emotion.
- Instead of: "Salman Khan made a heartwarming gesture towards a fan."
- Prefer: "Salman Khan was already heading out, but then he noticed someone waiting for him"

NATURAL FAN-PAGE VOICE (AVOID "AI PERFECTNESS"):
- Captions should sound like something a real creator would actually post.
- Too robotic: "Salman Khan surprised everyone with a heartwarming interaction with a young fan."
  Desired: "Salman Khan was already heading out, but then he noticed someone waiting for him"
- Too robotic: "The actress displayed an unexpected reaction when paparazzi requested a pose."
  Desired: "She was about to leave when the paparazzi asked her to pose… and her reaction"
- Avoid overly polished or formal sentences:
  "He could have just left, but he stopped" is much better than "Despite being ready to depart, he chose to pause and acknowledge the fan."

TRUTHFULNESS & STRICT FACTUALITY (DO NOT INVENT FACTS):
- Never invent: dialogue, thoughts, motives, relationships, locations, dates, feelings, backstory, reactions, or intentions.
- Stick 100% strictly to the factual story supplied by the creator. Only use information supported by the supplied context.

CAPTION ROLES & STRUCTURE:
1. MAIN CAPTION (main):
   - Contains: SETUP, CURIOSITY HOOK.
   - Setup + curiosity/open loop.
   - Makes the viewer want to watch. Makes them think: "What happened?"
   - Must NOT explain the complete event.
   - MUST include a comma ',' separating the setup and the hook.
   - Write enough natural wording to comfortably occupy approximately two lines at normal font size.
   - Examples of good style:
     * "Salman Khan was walking down the street, but then..."
     * "Salman Khan was casually walking, until something caught his eye..."
   - Vary hook phrases naturally (e.g., ", but then...", ", until...", ", however...", ", meanwhile...", ", then suddenly...", ", and then...").

2. CURIOSITY / REVEAL CAPTION (curiosity):
   - The actual payoff.
   - Explains what happened naturally. Feels like the satisfying second half of the Main.
   - Must have enough substance to naturally occupy approximately two lines.
   - Example:
     * "He stopped before leaving to take a photo with the young fan"
   - Do NOT add meaningless filler merely to increase length. Do NOT repeat the main caption.

3. EMOJI FIELD SEPARATION (MANDATORY):
   - main: Clean text only. Absolutely NO emojis inside main.
   - main_emoji: 0-2 relevant emojis or "" (e.g. "👀" or "😳" or "🤔").
   - curiosity: Clean text only. Absolutely NO emojis inside curiosity.
   - curiosity_emoji: 0-2 relevant emojis or "" (e.g. "❤️" or "😂" or "🔥").
   - Emojis must complement the sentence emotionally and naturally. Do not force emojis if they do not fit.

CAPTION LAYOUT & VISUAL COMPOSITION PHILOSOPHY:
- Think about the caption as visual composition, not character count.
- Target Line Behavior: The normal target is TWO NATURAL LINES. Two lines should be the standard output. The generated caption should normally be long enough that the existing renderer wraps it naturally into two visually balanced lines at the existing normal caption font size.
- Visual Balance: LINE 1: substantial amount of text; LINE 2: substantial amount of text. The two lines should feel intentional, readable, and balanced.
- One-Line Captions Are Undesirable: One line should be considered undesirable for normal generated captions because it leaves too much empty visual space and appears tiny or incomplete. Normal generated captions should not stay on one line.
- Three Lines Exception Only: Three lines are allowed ONLY as an exception when the wording genuinely cannot be expressed naturally and comfortably within two lines. Do not artificially force three lines. Do not artificially force two lines with awkward wording.
- Caption Length & No Rigid Character Count: Write enough natural text to comfortably occupy approximately two lines in the existing caption area at normal font size. Favor a visually balanced two-line composition over an exact character count. Do NOT target an arbitrary character-count range.
- No Forced Filler or Cramming: Do NOT use awkward filler words purely to increase length. Do NOT cram excessive words into the sentence just to force wrapping. Do NOT deliberately shorten the text just to guarantee two lines. Write naturally phrased, engaging text with enough substance to fill the available caption width across two lines.

CAPITALIZATION RULES:
- Use normal English sentence capitalization.
- Do NOT use Title Case / Every-Word-Capitalized style.
- Capitalize naturally:
  * Sentence beginnings
  * People's names (e.g. Salman Khan, Shah Rukh Khan, Deepika Padukone)
  * Places and venues (e.g. Mumbai Airport, Bandra)
  * Brands and studios
  * Genuine proper nouns
- Keep ordinary words lowercase.
- CRITICAL RATIONALE: The existing ShortsAI video renderer uses uppercase-first words as the visual red emphasis. Only names and genuinely important proper nouns should receive that red emphasis. If you use Title Case, every word becomes red.

4. THUMBNAIL / COVER-FRAME PHRASE (thumbnail_phrase):
   - A short, high-impact visual hook displayed on the 1:1 Saba Bollywood cover thumbnail frame.
   - Target Length:
     * Ideally 2–6 words.
     * Maximum 7 words.
     * Extremely short, punchy, and readable at a glance on mobile screens.
   - Strict Factuality (DO NOT INVENT FACTS):
     * The thumbnail phrase must be derived ONLY from the supplied context.
     * Never invent dialogue, actions, reactions, motives, relationships, events, locations, dates, or opinions not in the context.
     * For example, if context says "Salman stopped to meet a young fan and smiled for a photo":
       - Good: "SALMAN STOPPED FOR HIM ❤️"
       - Bad: "SALMAN CHANGED HIS LIFE ❤️" (invents unsupported claim).
   - Tone & Style:
     * Saba Bollywood audience: viral entertainment / paparazzi entertainment page / YouTube thumbnail.
     * Punchy, human, emotional, curiosity-driven, slightly dramatic when appropriate, instantly understandable.
     * Avoid corporate/formal news language, generic AI wording, SEO keywords, hashtags.
   - Do NOT Repeat the Title:
     * The thumbnail phrase is a short visual hook, NOT a repeat of the entire title or full sentence description.
     * For example, for "Salman Khan unexpectedly stopped while leaving and interacted with a fan":
       - Good: "SALMAN STOPPED 😳"
       - Bad: "SALMAN KHAN'S UNEXPECTED FAN MOMENT" (too long, repeats title).
   - Use the Strongest Visual / Emotional Hook:
     * Prioritize: unexpected action, strong reaction, emotional moment, funny moment, celebrity/fan interaction, surprise, wholesome moment, or dramatic reveal supported by context.
     * Examples:
       - "SALMAN DID THIS 😳"
       - "HER REACTION 😂"
       - "THIS WAS SO SWEET ❤️"
       - "HE ACTUALLY STOPPED 😳"
       - "FANS DID NOT EXPECT THIS"
       - "HE STOPPED FOR HER ❤️"
       - "NOBODY EXPECTED THIS"
   - Capitalization:
     * Use normal capitalization with strong emphasis where appropriate.
     * Celebrity names may naturally appear capitalized (e.g. "SALMAN DID THIS 😳") for strong visual punch.
     * Do NOT force every phrase into awkward Title Case (e.g. avoid "Salman Did This 😳").
     * Do not make every word uppercase automatically unless it feels natural for the moment.
   - Format Constraints:
     * 0–2 emojis when appropriate.
     * NO hashtags (e.g. no #hashtags).
     * NO quotation marks in the returned value.
     * Exactly ONE phrase in the "thumbnail_phrase" field. Do not return multiple options, explanations, or rankings.

FINAL HUMAN TEST:
Before returning, internally ask:
"Would a real Bollywood fan-page creator actually write this?"
If it sounds like a newspaper, generic AI, or overly complicated English, or lacks emotion or curiosity, rewrite it to be simpler, punchier, and more conversational!

OUTPUT FORMAT:
Respond with ONLY valid JSON with this exact schema:
{
  "main": "Setup goes here, hook phrase here...",
  "main_emoji": "👀",
  "curiosity": "Resolution completing the action with enough substance here",
  "curiosity_emoji": "❤️",
  "thumbnail_phrase": "SHORT VISUAL HOOK 😳"
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

def sanitize_thumbnail_phrase(phrase: str) -> str:
    """
    Cleans and validates the thumbnail phrase:
    - Strips leading/trailing whitespace and quotes.
    - Removes hashtags.
    - Strips internal quotation marks.
    - Normalizes internal spacing.
    """
    if not phrase:
        return ""
    p = phrase.strip().strip("\"'`“”‘’")
    p = re.sub(r'#\w+', '', p)
    for q in ('"', "'", '“', '”', '‘', '’', '`'):
        p = p.replace(q, '')
    p = re.sub(r'\s+', ' ', p).strip()
    return p

def parse_gemini_response(response_text: str) -> dict:
    """Parse and validate JSON response containing main, main_emoji, curiosity, curiosity_emoji, and thumbnail_phrase."""
    clean_text = clean_json_text(response_text)
    try:
        data = json.loads(clean_text)
    except Exception as e:
        # Fallback regex extraction if raw JSON parsing fails
        main_match = re.search(r'"(?:main|main_caption)"\s*:\s*"([^"]+)"', clean_text)
        curiosity_match = re.search(r'"(?:curiosity|curiosity_caption)"\s*:\s*"([^"]+)"', clean_text)
        thumb_match = re.search(r'"thumbnail_phrase"\s*:\s*"([^"]+)"', clean_text)
        main_emoji_match = re.search(r'"main_emoji"\s*:\s*"([^"]*)"', clean_text)
        curiosity_emoji_match = re.search(r'"curiosity_emoji"\s*:\s*"([^"]*)"', clean_text)
        if main_match and curiosity_match:
            data = {
                "main": main_match.group(1),
                "curiosity": curiosity_match.group(1),
                "main_emoji": main_emoji_match.group(1) if main_emoji_match else "",
                "curiosity_emoji": curiosity_emoji_match.group(1) if curiosity_emoji_match else "",
                "thumbnail_phrase": thumb_match.group(1) if thumb_match else ""
            }
        else:
            raise GeminiAPIError(f"Failed to parse structured JSON from Gemini response: {str(e)}")

    if not isinstance(data, dict):
        raise GeminiAPIError("Gemini response is not a valid JSON object.")

    raw_main = str(data.get("main") if "main" in data else data.get("main_caption", "")).strip()
    raw_curiosity = str(data.get("curiosity") if "curiosity" in data else data.get("curiosity_caption", "")).strip()
    raw_main_emoji = str(data.get("main_emoji", "")).strip()
    raw_curiosity_emoji = str(data.get("curiosity_emoji", "")).strip()
    raw_thumb = data.get("thumbnail_phrase", "")
    thumb_phrase = sanitize_thumbnail_phrase(str(raw_thumb)) if raw_thumb else ""

    clean_main, embedded_m_emojis = separate_emojis(raw_main)
    clean_curiosity, embedded_c_emojis = separate_emojis(raw_curiosity)

    if not clean_main or not clean_curiosity:
        raise GeminiAPIError("Gemini response is missing required caption fields.")

    final_main_emoji = extract_emojis(raw_main_emoji) or embedded_m_emojis
    final_curiosity_emoji = extract_emojis(raw_curiosity_emoji) or embedded_c_emojis

    result = {
        "main": clean_main,
        "main_emoji": final_main_emoji,
        "curiosity": clean_curiosity,
        "curiosity_emoji": final_curiosity_emoji,
        "main_caption": clean_main,
        "curiosity_caption": clean_curiosity,
        "thumbnail_phrase": thumb_phrase
    }

    return result

def build_user_prompt(context: str, previous_generations: list = None) -> str:
    """Construct user context prompt with retry avoidance list if present."""
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

def build_gemini_payload(context: str, previous_generations: list = None) -> dict:
    """Construct Gemini API request payload with context and retry avoidance list."""
    full_user_prompt = build_user_prompt(context, previous_generations)

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

def generate_captions(
    context: str,
    previous_generations: list = None,
    api_key: str = None,
    model: str = None,
    max_retries: int = 3,
    base_delay: float = 0.5
) -> dict:
    """
    Generate a pair of (main_caption, curiosity_caption) using Google Gemini API.
    Uses standard library urllib without adding new dependencies.
    Retries transient errors (503, 429, 408, 5xx) with exponential backoff and jitter.
    Permanent client errors (400, 401, 403, 404) fail immediately without retrying.
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

    resp_data = None
    for attempt in range(max_retries + 1):
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
                break
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                err_msg = err_data.get("error", {}).get("message", "")
            except Exception:
                pass

            # Permanent client errors: fail immediately without retrying
            if e.code in (400, 401, 403, 404):
                if e.code in (400, 403) and ("API_KEY" in err_msg or "key" in err_msg.lower()):
                    raise GeminiConfigError("Invalid GEMINI_API_KEY configured.")
                detail = f": {err_msg}" if err_msg else ""
                raise GeminiAPIError(f"Gemini API request failed ({e.code}){detail}")

            # Transient errors: 429, 408, 503, or other 5xx server errors
            is_transient = e.code in (408, 429, 503) or (500 <= e.code < 600)
            if is_transient:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0.05, 0.25)
                    time.sleep(delay)
                    continue
                else:
                    if e.code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        raise GeminiQuotaError("Gemini limit reached. Try again later.")
                    detail = f": {err_msg}" if err_msg else ""
                    raise GeminiAPIError(f"Gemini API request failed ({e.code}){detail}")

            # Other HTTP errors
            detail = f": {err_msg}" if err_msg else ""
            raise GeminiAPIError(f"Gemini API request failed ({e.code}){detail}")

        except urllib.error.URLError as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0.05, 0.25)
                time.sleep(delay)
                continue
            raise GeminiAPIError(f"Network error connecting to Gemini API: {str(e.reason)}")
        except Exception as e:
            raise GeminiAPIError(f"Failed to communicate with Gemini API: {str(e)}")

    if not resp_data:
        raise GeminiAPIError("Failed to obtain response from Gemini API.")

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

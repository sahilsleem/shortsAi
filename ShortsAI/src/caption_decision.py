import os
import json
import argparse
import time
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

class ValidationError(Exception):
    pass

@dataclass
class Caption:
    main_text: str
    curiosity_text: str
    emoji: str

@dataclass
class Reveal:
    text: str

@dataclass
class Constraints:
    max_lines: int
    safe_width_px: int
    graphics_allowed: bool

@dataclass
class CandidateMetadata:
    rank: int
    timestamp: float

@dataclass
class CaptionDecision:
    version: int
    candidate: CandidateMetadata
    usable: bool
    decision_confidence: float
    fact_basis: List[str]
    unsupported_claims: List[str]
    caption: Caption
    reveal: Reveal
    constraints: Constraints
    reasoning: List[str]

class CaptionProvider(ABC):
    @abstractmethod
    def generate(self, semantic_review: dict, filtered_facts: List[str], unsupported: List[str]) -> dict:
        pass

class DryRunCaptionProvider(CaptionProvider):
    def generate(self, semantic_review: dict, filtered_facts: List[str], unsupported: List[str]) -> dict:
        event = semantic_review.get("event", "")
        main_text = event if event else (filtered_facts[0] if filtered_facts else "")
        main_text = main_text.replace("\n", " ").strip()
        if len(main_text) > 80:
            main_text = main_text[:77] + "..."
            
        return {
            "main_text": main_text,
            "curiosity_text": "",
            "emoji": "🔥",
            "reveal_text": "Wait for it...",
            "fact_basis": filtered_facts,
            "unsupported_claims": unsupported,
            "confidence": semantic_review.get("confidence", 0.0),
            "reasoning": ["Generated via deterministic dry-run fallback."]
        }

class GeminiCaptionProvider(CaptionProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.model_name = os.environ.get("GEMINI_SEMANTIC_MODEL", "gemini-3.6-flash")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, semantic_review: dict, filtered_facts: List[str], unsupported: List[str]) -> dict:
        prompt_text = (
            "You are generating a caption decision for a YouTube Shorts video based on verified semantic evidence.\n"
            f"VERIFIED FACT BASIS:\n{json.dumps(filtered_facts, indent=2)}\n\n"
            f"REJECTED/UNSUPPORTED CLAIMS:\n{json.dumps(unsupported, indent=2)}\n\n"
            f"SEMANTIC EVENT:\n{semantic_review.get('event', 'unknown')}\n\n"
            "INSTRUCTIONS:\n"
            "- Generate captions only from supported evidence. Be creative in wording but strictly conservative in facts.\n"
            "- Existing on-screen captions/text are not factual proof.\n"
            "- Unreliable transcript is not factual proof.\n"
            "- Do not invent identity, relationships, actions, motives, dialogue, causes, outcomes, or events. Do not turn uncertainty into certainty.\n"
            "- Do not create circles, arrows, highlights, tracking, zooms, or graphic instructions. graphics_allowed must remain false.\n"
            "- Make the caption sound like a natural YouTube Shorts caption, not a dry computer-vision description. Avoid generic phrasing like 'A man performs...' or 'The crowd watches closely...'. Identify the recognizable subject/event with concise, human wording.\n"
            "- main_text is the factual setup.\n"
            "- curiosity_text is a short teaser ONLY when the evidence supports a genuine unresolved or interesting moment. Do not manufacture generic hooks (e.g., 'But then...', 'Wait for it...') unless truly supported by a subsequent payoff in the visual evidence. Otherwise, leave it empty.\n"
            "- emoji is optional. It must have a clear semantic relationship to the caption and visible scene. Never choose unrelated emojis. 0 emojis is completely acceptable if none fit perfectly. At most 2 emoji characters.\n"
            "- main_text + curiosity_text must be concise enough for the existing 880px / maximum-2-line renderer (under 80 chars total is ideal).\n"
            "- Do not use line breaks.\n"
            "- Do not include markdown, HTML, color instructions, or styling instructions.\n"
        )
        
        from google.genai import types
        schema = {
            "type": "OBJECT",
            "properties": {
                "main_text": {"type": "STRING"},
                "curiosity_text": {"type": "STRING"},
                "emoji": {"type": "STRING"},
                "reveal_text": {"type": "STRING"},
                "fact_basis": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "unsupported_claims": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "confidence": {"type": "NUMBER"},
                "reasoning": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": [
                "main_text", "curiosity_text", "emoji", "reveal_text",
                "fact_basis", "unsupported_claims", "confidence", "reasoning"
            ]
        }
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
        except Exception as e:
            error_message = str(e)
            if "503" in error_message and ("UNAVAILABLE" in error_message.upper() or "high demand" in error_message.lower()):
                raise RuntimeError(f"Gemini API request failed with a retryable temporary service-capacity error: {error_message}")
            raise RuntimeError(f"Gemini API request failed: {error_message}")
            
        content = response.text
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Failed to parse Gemini response as JSON: {e}")

class CaptionDecisionValidator:
    def count_emojis(self, text: str) -> int:
        count = 0
        for c in text:
            # Ignore ZWJ (0x200D), Variation Selector 16 (0xFE0F), and Skin Tones (0x1F3FB-0x1F3FF)
            if ord(c) > 0x2000 and ord(c) not in [0x200D, 0xFE0F] and not (0x1F3FB <= ord(c) <= 0x1F3FF):
                count += 1
        return count

    def validate_and_parse(self, raw_decision: dict) -> dict:
        required_keys = [
            "main_text", "curiosity_text", "emoji", "reveal_text",
            "fact_basis", "unsupported_claims", "confidence", "reasoning"
        ]
        for k in required_keys:
            if k not in raw_decision:
                raise ValidationError(f"Missing required AI response field: {k}")
                
        for k in ["main_text", "curiosity_text", "emoji", "reveal_text"]:
            if not isinstance(raw_decision[k], str):
                raise ValidationError(f"Field {k} must be a string")
            if "\n" in raw_decision[k] or "\r" in raw_decision[k]:
                raise ValidationError(f"Field {k} contains newlines")
            if "<" in raw_decision[k] and ">" in raw_decision[k]:
                # basic HTML/markdown check
                raise ValidationError(f"Field {k} contains possible HTML/styling instructions")
            if "**" in raw_decision[k] or "__" in raw_decision[k]:
                raise ValidationError(f"Field {k} contains markdown styling")
                
        if len(raw_decision["main_text"]) + len(raw_decision["curiosity_text"]) > 100:
            raise ValidationError("Caption text too long. Exceeds maximum 2-line concise constraint.")
            
        if self.count_emojis(raw_decision["emoji"]) > 2:
            raise ValidationError(f"Too many emojis provided: {raw_decision['emoji']}")
            
        conf = raw_decision["confidence"]
        if not isinstance(conf, (int, float)):
            raise ValidationError("confidence must be a number")
        if conf < 0.0 or conf > 1.0:
            raise ValidationError("confidence must be between 0.0 and 1.0")
            
        for list_key in ["fact_basis", "unsupported_claims", "reasoning"]:
            if not isinstance(raw_decision[list_key], list):
                raise ValidationError(f"Field {list_key} must be a list")
                
        # Check that on-screen text isn't the sole factual basis
        facts = raw_decision["fact_basis"]
        if facts:
            all_text_based = all("text" in f.lower() or "caption" in f.lower() or "subtitle" in f.lower() or "transcript" in f.lower() for f in facts)
            if all_text_based:
                raise ValidationError("On-screen text or unreliable transcript cannot be the sole factual basis.")
                
        # Check no graphics instructions
        combined_text = (raw_decision["main_text"] + " " + raw_decision["curiosity_text"] + " " + raw_decision["reveal_text"]).lower()
        banned_graphics = ["circle", "arrow", "highlight", "tracking", "zoom", "overlay"]
        for bg in banned_graphics:
            if bg in combined_text:
                raise ValidationError(f"Caption contains forbidden graphics instruction: {bg}")
                
        return raw_decision

class CaptionDecisionMaker:
    def __init__(self, provider: CaptionProvider):
        self.provider = provider
        self.validator = CaptionDecisionValidator()

    def decide(self, decision_json: dict, semantic_review_json: dict) -> CaptionDecision:
        semantic_data = semantic_review_json.get("semantic_review", {})
        cand_meta = semantic_review_json.get("request_metadata", {})
        
        candidate = CandidateMetadata(
            rank=cand_meta.get("candidate_rank", 1),
            timestamp=cand_meta.get("timestamp", 0.0)
        )
        
        # Deterministic preliminary safety check
        confidence = semantic_data.get("confidence", 0.0)
        uncertainties = semantic_data.get("uncertainties", [])
        facts = semantic_data.get("caption_fact_basis", [])
        
        clean_facts = []
        unsupported = []
        
        for f in facts:
            if "on-screen text" in f.lower() or "caption" in f.lower() or "subtitle" in f.lower() or "transcript" in f.lower():
                unsupported.append(f"Rejected text-based claim: {f}")
            else:
                clean_facts.append(f)
                
        unsupported.extend(uncertainties)
        
        fallback_decision = CaptionDecision(
            version=1,
            candidate=candidate,
            usable=False,
            decision_confidence=confidence,
            fact_basis=clean_facts,
            unsupported_claims=unsupported,
            caption=Caption("", "", ""),
            reveal=Reveal(""),
            constraints=Constraints(max_lines=2, safe_width_px=880, graphics_allowed=False),
            reasoning=["Insufficient semantic evidence or validation failed."]
        )
        
        if confidence < 0.6 or not clean_facts:
            # Fails preliminary safety check
            fallback_decision.reasoning = ["No purely visual facts remaining or low confidence."]
            return fallback_decision
            
        try:
            ai_decision = self.provider.generate(semantic_data, clean_facts, unsupported)
            validated = self.validator.validate_and_parse(ai_decision)
        except Exception as e:
            fallback_decision.reasoning = [f"AI Validation/Generation failed: {str(e)}"]
            return fallback_decision
            
        return CaptionDecision(
            version=1,
            candidate=candidate,
            usable=True,
            decision_confidence=validated["confidence"],
            fact_basis=validated["fact_basis"],
            unsupported_claims=unsupported + validated["unsupported_claims"],
            caption=Caption(
                main_text=validated["main_text"],
                curiosity_text=validated["curiosity_text"],
                emoji=validated["emoji"]
            ),
            reveal=Reveal(text=validated["reveal_text"]),
            constraints=Constraints(max_lines=2, safe_width_px=880, graphics_allowed=False),
            reasoning=validated["reasoning"]
        )

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Caption Decision Layer")
    parser.add_argument("--provider", type=str, default="dry-run", choices=["dry-run", "gemini"], help="AI Provider to use")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly permit real network requests")
    args = parser.parse_args()
    
    if args.provider != "dry-run" and not args.allow_network:
        print(f"Error: {args.provider.capitalize()} network access requires --allow-network flag.")
        print("Network request halted for safety. No output file created.")
        return
        
    decision_file = "working/decision.json"
    semantic_file = "working/semantic_review_gemini.json"
    output_file = "working/caption_decision.json"
    
    if not os.path.exists(decision_file) or not os.path.exists(semantic_file):
        print(f"Error: Required input files missing.")
        return
        
    with open(decision_file, "r", encoding="utf-8") as f:
        decision_data = json.load(f)
        
    with open(semantic_file, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)
        
    if args.provider == "gemini":
        try:
            provider = GeminiCaptionProvider()
        except Exception as e:
            print(f"Error initializing Gemini provider: {e}")
            return
    else:
        provider = DryRunCaptionProvider()
        
    maker = CaptionDecisionMaker(provider)
    decision = maker.decide(decision_data, semantic_data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2)
        
    print("Caption Decision generation complete.\n")
    print("Provider:")
    print(f"    {args.provider}\n")
    print("Candidate:")
    print(f"    Rank {decision.candidate.rank}")
    print(f"    Timestamp {decision.candidate.timestamp}s\n")
    print("Usable:")
    print(f"    {decision.usable}\n")
    if decision.usable:
        print("Caption:")
        print(f"    Main: {decision.caption.main_text}")
        print(f"    Curiosity: {decision.caption.curiosity_text}")
        print(f"    Emoji: {decision.caption.emoji.encode('unicode_escape').decode('utf-8')}\n")
    else:
        print("Reasoning:")
        for r in decision.reasoning:
            print(f"    - {r}")
        print()
    print(f"Output:\n    {output_file}")

if __name__ == "__main__":
    main()

import os
import json
import argparse
import time
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ValidationError(Exception):
    pass

class SemanticAIProvider(ABC):
    @abstractmethod
    def review(self, request_data: dict) -> dict:
        pass

    @abstractmethod
    def get_provider_info(self) -> dict:
        pass

    def build_prompt_text(self, request_data: dict) -> str:
        prompt_text = (
            f"TASK:\n{json.dumps(request_data.get('task'), indent=2)}\n\n"
            f"SPEECH EVIDENCE (from noisy transcript):\n{json.dumps(request_data.get('speech_evidence'), indent=2)}\n\n"
            f"INSTRUCTIONS:\n"
            f"The transcript is noisy and has unknown reliability. Use it only as secondary evidence. Never treat transcript text as verified fact when visual evidence disagrees or does not support it.\n"
            f"- Analyze only what is actually visible.\n"
            f"- Never invent a person's identity.\n"
            f"- If identity is not visually established, use 'unknown'.\n"
            f"- Never invent an event.\n"
            f"- Never infer unsupported facts from the filename.\n"
            f"- Distinguish visual evidence from transcript-derived information.\n"
            f"- If something cannot be determined, use 'unknown'.\n"
            f"- Do not create marketing language.\n"
            f"- Hook and reveal ideas must be based only on observed facts.\n"
            f"- Report uncertainty honestly.\n"
            f"- Do not use existing captions, subtitles, text overlays, filenames, or metadata as evidence for factual claims.\n"
            f"- If visible text describes an event, report it separately as 'on-screen text evidence', but do not use it to establish that the event actually occurred.\n"
            f"- Determine events from the visible people/actions/scene only.\n"
            f"- Do not assume a person performed something merely because the caption says they did.\n"
            f"- Do not suggest or mention renderer graphics such as circles, arrows, highlight overlays, tracking graphics, or visual effects unless they are genuinely part of the original source footage.\n"
            f"- Semantic review should describe factual evidence, not design the final caption or graphics. The 'hook.idea' and 'reveal.idea' fields are ONLY evidence assessments and must not contain graphic/design suggestions.\n"
        )
        instructions = request_data.get('instructions', {})
        for instruction in instructions.get("hallucination_safeguards", []):
            prompt_text += f"- {instruction}\n"
            
        prompt_text += "\nDo not write final YouTube captions. Do not write polished marketing copy. Do not exaggerate the event.\n"
        return prompt_text

class DryRunSemanticProvider(SemanticAIProvider):
    def review(self, request_data: dict) -> dict:
        return {
            "scene_summary": "DRY RUN - semantic analysis not performed",
            "people": [
                {
                    "description": "unknown",
                    "identity": "unknown",
                    "action": "unknown"
                }
            ],
            "interaction": "unknown",
            "expressions": "unknown",
            "event": "unknown",
            "hook": {
                "available": False,
                "idea": ""
            },
            "reveal": {
                "available": False,
                "idea": ""
            },
            "caption_fact_basis": [],
            "confidence": 0.0,
            "uncertainties": [
                "Semantic inference has not been performed."
            ]
        }

    def get_provider_info(self) -> dict:
        return {
            "name": "dry-run",
            "inference_performed": False
        }

class OpenAISemanticProvider(SemanticAIProvider):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Cannot use OpenAI provider.")
        self.model = os.environ.get("OPENAI_SEMANTIC_MODEL", "gpt-4o-mini")
        
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)

    def _encode_image(self, path: str) -> str:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def review(self, request_data: dict) -> dict:
        visual_evidence = request_data.get("visual_evidence", [])
        
        prompt_text = self.build_prompt_text(request_data)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        for frame in visual_evidence[:3]:
            path = frame.get("path")
            b64_image = self._encode_image(path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}",
                    "detail": "low"
                }
            })
            
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_review",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scene_summary": {"type": "string"},
                        "people": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "identity": {"type": "string"},
                                    "action": {"type": "string"}
                                },
                                "required": ["description", "identity", "action"],
                                "additionalProperties": False
                            }
                        },
                        "interaction": {"type": "string"},
                        "expressions": {"type": "string"},
                        "event": {"type": "string"},
                        "hook": {
                            "type": "object",
                            "properties": {
                                "available": {"type": "boolean"},
                                "idea": {"type": "string"}
                            },
                            "required": ["available", "idea"],
                            "additionalProperties": False
                        },
                        "reveal": {
                            "type": "object",
                            "properties": {
                                "available": {"type": "boolean"},
                                "idea": {"type": "string"}
                            },
                            "required": ["available", "idea"],
                            "additionalProperties": False
                        },
                        "caption_fact_basis": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "confidence": {"type": "number"},
                        "uncertainties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": [
                        "scene_summary", "people", "interaction", "expressions",
                        "event", "hook", "reveal", "caption_fact_basis",
                        "confidence", "uncertainties"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=schema,
            max_completion_tokens=1000
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    def get_provider_info(self) -> dict:
        return {
            "name": "openai",
            "model": self.model,
            "inference_performed": True
        }

class GeminiSemanticProvider(SemanticAIProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Cannot use Gemini provider.")
        self.model_name = os.environ.get("GEMINI_SEMANTIC_MODEL", "gemini-3.6-flash")
        
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def review(self, request_data: dict) -> dict:
        visual_evidence = request_data.get("visual_evidence", [])
        
        prompt_text = self.build_prompt_text(request_data)
        prompt_text += "\nReturn output exactly as a JSON object matching the provided expected response schema.\n"
        
        import PIL.Image
        contents = [prompt_text]
        
        for frame in visual_evidence[:3]:
            path = frame.get("path")
            if not os.path.exists(path):
                raise ValidationError(f"Failed to load semantic evidence frame: {path}")
            try:
                img = PIL.Image.open(path)
                img.load()
                contents.append(img)
            except Exception:
                raise ValidationError(f"Failed to load semantic evidence frame: {path}")
                
        from google.genai import types
        schema = {
            "type": "OBJECT",
            "properties": {
                "scene_summary": {"type": "STRING"},
                "people": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "description": {"type": "STRING"},
                            "identity": {"type": "STRING"},
                            "action": {"type": "STRING"}
                        },
                        "required": ["description", "identity", "action"]
                    }
                },
                "interaction": {"type": "STRING"},
                "expressions": {"type": "STRING"},
                "event": {"type": "STRING"},
                "hook": {
                    "type": "OBJECT",
                    "properties": {
                        "available": {"type": "BOOLEAN"},
                        "idea": {"type": "STRING"}
                    },
                    "required": ["available", "idea"]
                },
                "reveal": {
                    "type": "OBJECT",
                    "properties": {
                        "available": {"type": "BOOLEAN"},
                        "idea": {"type": "STRING"}
                    },
                    "required": ["available", "idea"]
                },
                "caption_fact_basis": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "confidence": {"type": "NUMBER"},
                "uncertainties": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": [
                "scene_summary", "people", "interaction", "expressions",
                "event", "hook", "reveal", "caption_fact_basis",
                "confidence", "uncertainties"
            ]
        }
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
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
            raise ValidationError(f"Failed to parse Gemini response as JSON: {e}. Raw content: {content}")

    def get_provider_info(self) -> dict:
        return {
            "name": "gemini",
            "model": self.model_name,
            "inference_performed": True
        }

class RequestValidator:
    def validate(self, req: dict):
        required_keys = [
            "version", "task", "source", "candidate", 
            "visual_evidence", "speech_evidence", 
            "instructions", "expected_response_schema"
        ]
        for k in required_keys:
            if k not in req:
                raise ValidationError(f"Missing required request field: {k}")
                
        candidate = req["candidate"]
        for k in ["rank", "timestamp", "window_start", "window_end", "deterministic_score", "decision_confidence"]:
            if k not in candidate:
                raise ValidationError(f"Missing candidate field: {k}")
                
        frames = req.get("visual_evidence", [])
        if len(frames) != 3:
            raise ValidationError(f"Request must have exactly 3 visual frames, got {len(frames)}")
            
        for f in frames:
            path = f.get("path")
            if not path:
                raise ValidationError("Missing frame path in visual_evidence")
            if not os.path.exists(path):
                raise ValidationError(f"Frame path does not exist on disk: {path}")

class ResponseValidator:
    def validate(self, res: dict):
        required_keys = [
            "scene_summary", "people", "interaction", "expressions",
            "event", "hook", "reveal", "caption_fact_basis",
            "confidence", "uncertainties"
        ]
        
        for k in required_keys:
            if k not in res:
                raise ValidationError(f"Missing required response field: {k}")
                
        if not isinstance(res["scene_summary"], str):
            raise ValidationError("scene_summary must be a string")
            
        if not isinstance(res["people"], list):
            raise ValidationError("people must be a list")
            
        for p in res["people"]:
            if not isinstance(p, dict):
                raise ValidationError("each person must be a dict")
            for pk in ["description", "identity", "action"]:
                if pk not in p:
                    raise ValidationError(f"Missing person field: {pk}")
                if not isinstance(p[pk], str):
                    raise ValidationError(f"Person field {pk} must be a string")
                    
        if not isinstance(res["interaction"], str):
            raise ValidationError("interaction must be a string")
        if not isinstance(res["expressions"], str):
            raise ValidationError("expressions must be a string")
        if not isinstance(res["event"], str):
            raise ValidationError("event must be a string")
            
        for hk in ["hook", "reveal"]:
            obj = res[hk]
            if not isinstance(obj, dict):
                raise ValidationError(f"{hk} must be an object")
            if "available" not in obj or not isinstance(obj["available"], bool):
                raise ValidationError(f"{hk}.available must be a boolean")
            if "idea" not in obj or not isinstance(obj["idea"], str):
                raise ValidationError(f"{hk}.idea must be a string")
                
        if not isinstance(res["caption_fact_basis"], list):
            raise ValidationError("caption_fact_basis must be a list")
            
        conf = res["confidence"]
        if not isinstance(conf, (int, float)):
            raise ValidationError("confidence must be a number")
        if conf < 0.0 or conf > 1.0:
            raise ValidationError("confidence must be between 0.0 and 1.0")
            
        if not isinstance(res["uncertainties"], list):
            raise ValidationError("uncertainties must be a list")

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Semantic AI Adapter")
    parser.add_argument("--provider", type=str, default="dry-run", choices=["dry-run", "openai", "gemini"], help="Provider to use for semantic review")
    parser.add_argument("--allow-network", action="store_true", help="Explicitly permit real network requests")
    args = parser.parse_args()
    
    if args.provider != "dry-run" and not args.allow_network:
        print(f"Error: {args.provider.capitalize()} network access requires --allow-network flag.")
        print("Network request halted for safety. No output file created.")
        return
    
    request_file = "working/semantic_review_request.json"
    
    if args.provider == "openai":
        output_file = "working/semantic_review_real.json"
    elif args.provider == "gemini":
        output_file = "working/semantic_review_gemini.json"
    else:
        output_file = "working/semantic_review.json"
    
    t0 = time.time()
    
    if not os.path.exists(request_file):
        print(f"Error: {request_file} not found.")
        return
        
    with open(request_file, "r", encoding="utf-8") as f:
        request_data = json.load(f)
        
    req_validator = RequestValidator()
    try:
        req_validator.validate(request_data)
        input_validation = "PASS"
    except ValidationError as e:
        input_validation = f"FAIL ({e})"
        print(f"Input validation failed: {e}")
        return
        
    if args.provider == "openai":
        try:
            provider = OpenAISemanticProvider()
        except Exception as e:
            print(f"Error initializing OpenAI provider: {e}")
            return
    elif args.provider == "gemini":
        try:
            provider = GeminiSemanticProvider()
        except Exception as e:
            print(f"Error initializing Gemini provider: {e}")
            return
    else:
        provider = DryRunSemanticProvider()
        
    try:
        response_data = provider.review(request_data)
    except Exception as e:
        print(f"Error during provider review: {e}")
        return
    
    res_validator = ResponseValidator()
    try:
        res_validator.validate(response_data)
        output_validation = "PASS"
    except ValidationError as e:
        output_validation = f"FAIL ({e})"
        print(f"Response validation failed: {e}")
        return
        
    wrapper = {
        "version": 1,
        "provider": provider.get_provider_info(),
        "request_metadata": {
            "candidate_rank": request_data["candidate"]["rank"],
            "timestamp": request_data["candidate"]["timestamp"]
        },
        "semantic_review": response_data
    }
    
    os.makedirs("working", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2)
        
    t1 = time.time()
    
    info = provider.get_provider_info()
    
    print("Real semantic AI test\n" if args.provider != "dry-run" else "Semantic AI adapter dry-run complete.\n")
    print("Provider:")
    print(f"    {info.get('name', args.provider)}")
    if 'model' in info:
        print(f"\nModel:\n    {info['model']}")
    print("\nInference performed:")
    print(f"    {'YES' if info.get('inference_performed') else 'NO'}\n")
    print("Input validation:")
    print(f"    {input_validation}\n")
    print("Response validation:")
    print(f"    {output_validation}\n")
    print("Visual frames:")
    print(f"    {len(request_data.get('visual_evidence', []))}\n")
    print("Candidate:")
    print(f"    Rank {request_data['candidate']['rank']}")
    print(f"    Timestamp {request_data['candidate']['timestamp']}s\n")
    print("Network request:")
    print(f"    {'1 request' if args.provider in ('openai', 'gemini') else '0 requests'}\n")
    print("Output:")
    print(f"    {output_file}\n")
    print(f"Runtime:\n    {t1-t0:.4f} seconds")

if __name__ == "__main__":
    main()

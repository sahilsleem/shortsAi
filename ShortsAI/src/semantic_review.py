import os
import json
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class TaskInfo:
    purpose: str = "Understand the selected short-form video moment and identify the most useful hook/reveal structure."
    semantic_only: bool = True

@dataclass
class CandidateInfo:
    rank: int
    timestamp: float
    window_start: float
    window_end: float
    deterministic_score: float
    decision_confidence: float

@dataclass
class VisualEvidence:
    timestamp: float
    path: str

@dataclass
class SpeechEvidence:
    start: float
    end: float
    text: str

@dataclass
class Instructions:
    identify_people: bool = True
    describe_actions: bool = True
    describe_interaction: bool = True
    describe_expression_or_emotion: bool = True
    identify_event_or_moment: bool = True
    identify_hook_opportunity: bool = True
    identify_reveal_opportunity: bool = True
    detect_uncertainty: bool = True
    hallucination_safeguards: List[str] = None

    def __post_init__(self):
        if self.hallucination_safeguards is None:
            self.hallucination_safeguards = [
                "Never invent a person's identity.",
                "If identity is not visually established, return 'unknown'.",
                "Never invent events that cannot be observed.",
                "Do not treat the Whisper transcript as verified fact.",
                "Transcript reliability is currently 'unknown'.",
                "Distinguish observed visual facts from transcript-derived claims.",
                "If evidence conflicts, report the conflict.",
                "If something cannot be determined, say 'unknown'.",
                "Do not fabricate hook/reveal claims."
            ]

@dataclass
class SemanticReviewRequest:
    version: int
    task: TaskInfo
    source: dict
    candidate: CandidateInfo
    visual_evidence: List[VisualEvidence]
    speech_evidence: List[SpeechEvidence]
    instructions: Instructions
    expected_response_schema: dict

class SemanticReviewBuilder:
    def __init__(self, decision_path: str, context_path: str):
        self.decision_path = decision_path
        self.context_path = context_path

    def build_request(self) -> SemanticReviewRequest:
        if not os.path.exists(self.decision_path):
            raise FileNotFoundError(f"Missing {self.decision_path}")
        if not os.path.exists(self.context_path):
            raise FileNotFoundError(f"Missing {self.context_path}")
            
        with open(self.decision_path, "r", encoding="utf-8") as f:
            decision_data = json.load(f)
            
        with open(self.context_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
            
        selected_rank = decision_data.get("selected_candidate_rank")
        if selected_rank is None:
            raise ValueError("No selected candidate rank found in decision.json")
            
        candidates = context_data.get("candidates", [])
        selected_candidate = next((c for c in candidates if c.get("rank") == selected_rank), None)
        
        if not selected_candidate:
            raise ValueError(f"Candidate rank {selected_rank} not found in candidate_context.json")
            
        # Build Visual Evidence
        visual_evidence = []
        for frame in selected_candidate.get("frames", []):
            visual_evidence.append(VisualEvidence(
                timestamp=frame.get("timestamp", 0.0),
                path=frame.get("path", "")
            ))
            
        # Build Speech Evidence
        speech_evidence = []
        for seg in selected_candidate.get("speech_segments", []):
            speech_evidence.append(SpeechEvidence(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "")
            ))
            
        # Expected Schema Definition
        expected_schema = {
            "scene_summary": "string",
            "people": [
                {
                    "description": "string",
                    "identity": "string or unknown",
                    "action": "string"
                }
            ],
            "interaction": "string",
            "expressions": "string",
            "event": "string",
            "hook": {
                "available": True,
                "idea": "string"
            },
            "reveal": {
                "available": True,
                "idea": "string"
            },
            "caption_fact_basis": [
                "string"
            ],
            "confidence": 0.0,
            "uncertainties": [
                "string"
            ]
        }
        
        cand_info = CandidateInfo(
            rank=selected_rank,
            timestamp=selected_candidate.get("timestamp", 0.0),
            window_start=selected_candidate.get("window_start", 0.0),
            window_end=selected_candidate.get("window_end", 0.0),
            deterministic_score=selected_candidate.get("score", 0.0),
            decision_confidence=decision_data.get("confidence", 0.0)
        )
        
        request = SemanticReviewRequest(
            version=1,
            task=TaskInfo(),
            source=context_data.get("source", {}),
            candidate=cand_info,
            visual_evidence=visual_evidence,
            speech_evidence=speech_evidence,
            instructions=Instructions(),
            expected_response_schema=expected_schema
        )
        
        return request

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Semantic Review Interface")
    args = parser.parse_args()
    
    decision_file = "working/decision.json"
    context_file = "working/candidate_context.json"
    output_file = "working/semantic_review_request.json"
    
    t0 = time.time()
    
    builder = SemanticReviewBuilder(decision_file, context_file)
    try:
        req = builder.build_request()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    os.makedirs("working", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(asdict(req), f, indent=2)
        
    t1 = time.time()
    
    print("Semantic review request created.\n")
    print("Selected candidate:")
    print(f"    Rank: {req.candidate.rank}")
    print(f"    Timestamp: {req.candidate.timestamp}s")
    print(f"    Window: {req.candidate.window_start}s - {req.candidate.window_end}s\n")
    print("Visual evidence:")
    print(f"    {len(req.visual_evidence)} frames\n")
    print("Speech evidence:")
    print(f"    {len(req.speech_evidence)} segments\n")
    print("AI inference:")
    print("    NOT RUN\n")
    print("Saved:")
    print(f"    {output_file}\n")
    print(f"Runtime: {t1-t0:.4f} seconds")

if __name__ == "__main__":
    main()

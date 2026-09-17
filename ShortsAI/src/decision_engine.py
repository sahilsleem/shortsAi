import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

WEIGHTS = {
    "audio_energy": 0.35,
    "motion": 0.30,
    "visual_detail": 0.20,
    "scene_change": 0.15
}

TEMPORAL_SUPPRESSION_THRESHOLD = 1.5

@dataclass
class ScoredCandidate:
    rank: int
    timestamp: float
    score: float
    decision_score: float
    signals: dict

@dataclass
class CaptionContext:
    speech_segments: List[dict]
    transcript_reliability: str = "unknown"
    suggested_role: str = "candidate_context_only"

@dataclass
class EditingDecision:
    version: int
    selected_candidate_rank: int
    selected_timestamp: float
    clip_start: float
    clip_end: float
    confidence: float
    reasoning: List[str]
    candidate_scores: List[ScoredCandidate]
    caption_context: CaptionContext
    needs_visual_review: bool = True
    visual_review_reason: str = "Deterministic signals cannot establish semantic content."

class DecisionEngine:
    def __init__(self, candidate_context_path: str):
        self.candidate_context_path = candidate_context_path
        
    def calculate_decision_score(self, signals: dict) -> float:
        score = (
            signals.get("audio_energy", 0.0) * WEIGHTS["audio_energy"] +
            signals.get("motion", 0.0) * WEIGHTS["motion"] +
            signals.get("visual_detail", 0.0) * WEIGHTS["visual_detail"] +
            signals.get("scene_change", 0.0) * WEIGHTS["scene_change"]
        )
        return min(1.0, max(0.0, score))

    def suppress_overlaps(self, scored_candidates: List[dict]) -> List[dict]:
        # scored_candidates is assumed to be sorted by decision_score descending
        retained = []
        for cand in scored_candidates:
            ts = cand.get("timestamp", 0.0)
            overlap = False
            for r in retained:
                r_ts = r.get("timestamp", 0.0)
                if abs(ts - r_ts) < TEMPORAL_SUPPRESSION_THRESHOLD:
                    overlap = True
                    break
            if not overlap:
                retained.append(cand)
        return retained
        
    def calculate_confidence(self, top_candidates: List[dict]) -> float:
        if not top_candidates:
            return 0.0
            
        top_cand = top_candidates[0]
        top_score = top_cand.get("decision_score", 0.0)
        
        # Base confidence from the score itself
        base_confidence = top_score
        
        # Adjust based on delta to the runner up
        if len(top_candidates) > 1:
            runner_up_score = top_candidates[1].get("decision_score", 0.0)
            delta = top_score - runner_up_score
            # Boost if clear winner, penalize if very close
            if delta > 0.1:
                base_confidence += 0.1
            elif delta < 0.02:
                base_confidence -= 0.1
                
        # Signal agreement
        signals = top_cand.get("signals", {})
        active_signals = sum(1 for k, v in signals.items() if v > 0.2)
        if active_signals >= 3:
            base_confidence += 0.1
        elif active_signals <= 1:
            base_confidence -= 0.1
            
        return min(1.0, max(0.0, base_confidence))
        
    def make_decision(self) -> EditingDecision:
        if not os.path.exists(self.candidate_context_path):
            raise FileNotFoundError(f"Missing {self.candidate_context_path}")
            
        with open(self.candidate_context_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
            
        candidates_in = context_data.get("candidates", [])
        
        # 1. Recalculate scores
        scored = []
        for c in candidates_in:
            signals = c.get("signals", {})
            decision_score = self.calculate_decision_score(signals)
            c["decision_score"] = decision_score
            scored.append(c)
            
        # Sort by decision_score descending
        scored.sort(key=lambda x: x.get("decision_score", 0.0), reverse=True)
        
        # 2. Temporal suppression
        suppressed = self.suppress_overlaps(scored)
        
        if not suppressed:
            raise ValueError("No candidates found after suppression.")
            
        winner = suppressed[0]
        
        # 3. Confidence
        confidence = self.calculate_confidence(suppressed)
        
        # 4. Reasoning
        reasoning = [
            "Highest combined deterministic evidence after temporal suppression."
        ]
        signals = winner.get("signals", {})
        for sig, val in signals.items():
            if val > 0.5:
                reasoning.append(f"Strong {sig} signal ({val:.2f})")
                
        # 5. Format candidate scores for output
        out_scores = []
        for c in scored:
            out_scores.append(ScoredCandidate(
                rank=c.get("rank", 0),
                timestamp=c.get("timestamp", 0.0),
                score=c.get("score", 0.0),
                decision_score=c.get("decision_score", 0.0),
                signals=c.get("signals", {})
            ))
            
        caption_context = CaptionContext(
            speech_segments=winner.get("speech_segments", [])
        )
        
        return EditingDecision(
            version=1,
            selected_candidate_rank=winner.get("rank", 1),
            selected_timestamp=winner.get("timestamp", 0.0),
            clip_start=winner.get("window_start", 0.0),
            clip_end=winner.get("window_end", 0.0),
            confidence=confidence,
            reasoning=reasoning,
            candidate_scores=out_scores,
            caption_context=caption_context
        )

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Decision Engine")
    args = parser.parse_args()
    
    print("Running Decision Engine...")
    import time
    t0 = time.time()
    
    context_path = "working/candidate_context.json"
    engine = DecisionEngine(context_path)
    
    try:
        decision = engine.make_decision()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    out_file = "working/decision.json"
    os.makedirs("working", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2)
        
    t1 = time.time()
    
    print("\nDecision complete.")
    print("\nSelected candidate:")
    print(f"    Rank: {decision.selected_candidate_rank}")
    print(f"    Timestamp: {decision.selected_timestamp}s")
    print(f"    Window: {decision.clip_start}s - {decision.clip_end}s")
    print(f"    Confidence: {decision.confidence:.2f}")
    
    print("\nVisual review required:")
    print(f"    {'Yes' if decision.needs_visual_review else 'No'}")
    
    print("\nReason:")
    for r in decision.reasoning:
        print(f"    - {r}")
        
    print(f"\nRuntime: {t1-t0:.4f} seconds")

if __name__ == "__main__":
    main()

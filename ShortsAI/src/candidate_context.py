import os
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from .config import RenderConfig
from .video_ops import get_ffmpeg_path

@dataclass
class CandidateFrame:
    timestamp: float
    path: str

@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str

@dataclass
class CandidateContext:
    rank: int
    timestamp: float
    window_start: float
    window_end: float
    score: float
    signals: dict
    speech_segments: List[SpeechSegment]
    clip_path: str
    frames: List[CandidateFrame]

@dataclass
class CandidateContextOutput:
    source: dict
    candidates: List[CandidateContext]

class ContextExtractor:
    def __init__(self, source_path: str, source_duration: float):
        self.source_path = str(Path(source_path).resolve())
        self.source_duration = source_duration

    def extract_clip(self, start: float, end: float, output_path: str):
        duration = end - start
        if duration <= 0:
            duration = 1.0
            
        cmd = [
            get_ffmpeg_path(), "-y",
            "-ss", str(start),
            "-i", self.source_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", # fast and lightweight
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def extract_frames(self, start: float, end: float, out_dir: str) -> List[CandidateFrame]:
        os.makedirs(out_dir, exist_ok=True)
        duration = end - start
        
        mid = start + (duration / 2.0)
        
        frames = []
        timestamps = [
            (start, "frame_start.jpg"),
            (mid, "frame_middle.jpg"),
            (end, "frame_end.jpg")
        ]
        
        for ts, filename in timestamps:
            # clamp ts to safely within video to prevent ffmpeg errors
            clamped_ts = max(0.0, min(self.source_duration - 0.05, ts))
            out_path = os.path.join(out_dir, filename)
            
            cmd = [
                get_ffmpeg_path(), "-y",
                "-ss", str(clamped_ts),
                "-i", self.source_path,
                "-vframes", "1",
                "-q:v", "2",
                out_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(out_path):
                frames.append(CandidateFrame(
                    timestamp=clamped_ts,
                    path=out_path.replace("\\", "/")
                ))
                
        return frames
        
    def get_overlapping_speech(self, start: float, end: float, transcript_segments: List[dict]) -> List[SpeechSegment]:
        overlapping = []
        for seg in transcript_segments:
            s_start = seg.get("start", 0.0)
            s_end = seg.get("end", 0.0)
            text = seg.get("text", "")
            
            # Check for overlap: max(start1, start2) < min(end1, end2)
            if max(start, s_start) < min(end, s_end):
                overlapping.append(SpeechSegment(
                    start=s_start,
                    end=s_end,
                    text=text
                ))
        return overlapping

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Candidate Context Extractor")
    args = parser.parse_args()
    
    config = RenderConfig.from_json("job.json")
    print(f"Extracting context for source video...")
    
    analysis_file = "working/candidate_analysis.json"
    transcript_file = "working/transcript.json"
    
    if not os.path.exists(analysis_file):
        print(f"Error: {analysis_file} not found. Run candidate analysis first.")
        return
        
    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)
        
    transcript_segments = []
    if os.path.exists(transcript_file):
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
                if transcript_data.get("available", False):
                    transcript_segments = transcript_data.get("segments", [])
        except Exception:
            pass

    source_info = analysis_data.get("source", {})
    source_duration = source_info.get("duration", 0.0)
    
    extractor = ContextExtractor(config.source_video, source_duration)
    
    candidates_in = analysis_data.get("candidates", [])
    candidates_out = []
    
    base_out_dir = "working/candidates"
    os.makedirs(base_out_dir, exist_ok=True)
    
    import time
    t0 = time.time()
    
    frames_generated = 0
    clips_generated = 0
    
    for idx, cand in enumerate(candidates_in):
        rank = idx + 1
        ts = cand.get("timestamp", 0.0)
        
        # Clamp timestamps
        win_start = max(0.0, min(source_duration, cand.get("window_start", 0.0)))
        win_end = max(0.0, min(source_duration, cand.get("window_end", 0.0)))
        if win_end <= win_start:
            win_end = min(source_duration, win_start + 1.0)
            
        print(f"Processing candidate {rank} (window {win_start:.1f}s - {win_end:.1f}s)...")
        
        clip_path = os.path.join(base_out_dir, f"candidate_{rank:02d}.mp4")
        extractor.extract_clip(win_start, win_end, clip_path)
        clips_generated += 1
        
        frame_dir = os.path.join(base_out_dir, f"candidate_{rank:02d}")
        extracted_frames = extractor.extract_frames(win_start, win_end, frame_dir)
        frames_generated += len(extracted_frames)
        
        overlapping_speech = extractor.get_overlapping_speech(win_start, win_end, transcript_segments)
        
        candidates_out.append(CandidateContext(
            rank=rank,
            timestamp=ts,
            window_start=win_start,
            window_end=win_end,
            score=cand.get("score", 0.0),
            signals=cand.get("signals", {}),
            speech_segments=overlapping_speech,
            clip_path=clip_path.replace("\\", "/"),
            frames=extracted_frames
        ))
        
    output = CandidateContextOutput(
        source=source_info,
        candidates=candidates_out
    )
    
    out_file = "working/candidate_context.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(output), f, indent=2)
        
    t1 = time.time()
    print(f"\nContext extraction complete in {t1-t0:.2f} seconds.")
    print(f"Processed {len(candidates_out)} candidates.")
    print(f"Generated {clips_generated} clips.")
    print(f"Generated {frames_generated} representative frames.")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()

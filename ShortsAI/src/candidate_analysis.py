import os
import json
import math
import wave
import struct
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from PIL import Image, ImageFilter, ImageStat

from .config import RenderConfig
from .video_ops import get_ffprobe_path, get_ffmpeg_path
from .analyze import MediaAnalyzer

@dataclass
class VisualSample:
    timestamp: float
    brightness: float
    contrast: float
    detail: float
    motion: float

@dataclass
class AudioSample:
    timestamp: float
    rms_level: float
    peak_level: float

@dataclass
class SceneChange:
    timestamp: float

@dataclass
class CandidateSignal:
    motion: float
    audio_energy: float
    scene_change: float
    visual_detail: float

@dataclass
class Candidate:
    timestamp: float
    window_start: float
    window_end: float
    score: float
    signals: CandidateSignal

@dataclass
class CandidateAnalysisOutput:
    source: dict
    visual_samples: List[VisualSample]
    audio_samples: List[AudioSample]
    scene_changes: List[SceneChange]
    candidates: List[Candidate]

class CandidateAnalyzer:
    def __init__(self, source_path: str):
        self.source_path = str(Path(source_path).resolve())
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source video not found: {self.source_path}")

    def analyze_visuals(self, frames_dir: str) -> List[VisualSample]:
        samples = []
        if not os.path.exists(frames_dir):
            return samples
            
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        prev_img = None
        
        for filename in frame_files:
            try:
                idx = int(filename[6:10])
                timestamp = float(idx - 1)
            except ValueError:
                continue
                
            frame_path = os.path.join(frames_dir, filename)
            img = Image.open(frame_path).convert("L")
            
            # Brightness and contrast
            stat = ImageStat.Stat(img)
            brightness = stat.mean[0] / 255.0
            contrast = stat.stddev[0] / 128.0 # roughly normalize
            
            # Detail/Edge
            edges = img.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            detail = edge_stat.mean[0] / 255.0
            
            # Motion
            motion = 0.0
            if prev_img is not None:
                # mean absolute difference
                from PIL import ImageChops
                diff = ImageChops.difference(img, prev_img)
                diff_stat = ImageStat.Stat(diff)
                motion = diff_stat.mean[0] / 255.0
                
            prev_img = img
            
            samples.append(VisualSample(
                timestamp=timestamp,
                brightness=min(1.0, max(0.0, brightness)),
                contrast=min(1.0, max(0.0, contrast)),
                detail=min(1.0, max(0.0, detail * 2.0)), # boost slightly for normalization
                motion=min(1.0, max(0.0, motion * 5.0))  # boost small motion
            ))
            
        return samples

    def analyze_audio(self) -> List[AudioSample]:
        samples = []
        temp_wav = "working/temp_audio.wav"
        
        # Check if video has audio
        probe_cmd = [
            get_ffprobe_path(), "-v", "error", "-show_entries",
            "stream=codec_type", "-of", "json", self.source_path
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True)
        try:
            streams = json.loads(res.stdout).get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            if not has_audio:
                return samples
        except Exception:
            return samples

        # Extract audio as 8kHz mono PCM
        cmd = [
            get_ffmpeg_path(), "-y", "-i", self.source_path,
            "-ac", "1", "-ar", "8000", "-acodec", "pcm_s16le", temp_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(temp_wav):
            return samples
            
        try:
            with wave.open(temp_wav, "rb") as wav:
                rate = wav.getframerate()
                nframes = wav.getnframes()
                data = wav.readframes(nframes)
                
                # 1 second windows
                window_frames = rate
                total_windows = nframes // window_frames
                
                for i in range(total_windows):
                    chunk = data[i * window_frames * 2 : (i + 1) * window_frames * 2]
                    # Unpack 16-bit PCM
                    num_samples = len(chunk) // 2
                    unpacked = struct.unpack(f"<{num_samples}h", chunk)
                    
                    if num_samples == 0:
                        continue
                        
                    rms = math.sqrt(sum(s*s for s in unpacked) / num_samples)
                    peak = max(abs(s) for s in unpacked)
                    
                    # Normalize against 16-bit max (32768)
                    norm_rms = min(1.0, rms / 32768.0)
                    # logarithmic scaling for RMS to represent perceived loudness better
                    norm_rms = math.pow(norm_rms, 0.5)
                    norm_peak = min(1.0, peak / 32768.0)
                    
                    samples.append(AudioSample(
                        timestamp=float(i),
                        rms_level=norm_rms,
                        peak_level=norm_peak
                    ))
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        return samples

    def analyze_scenes(self) -> List[SceneChange]:
        samples = []
        
        # Due to path issues in Lavfi filters on Windows, replace slashes
        clean_path = self.source_path.replace("\\", "/").replace(":", "\\:")
        
        cmd = [
            get_ffprobe_path(), "-v", "quiet", "-show_frames",
            "-show_entries", "frame=pkt_pts_time", "-of", "json",
            "-f", "lavfi", f"movie='{clean_path}',select='gt(scene,0.2)'"
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            frames = json.loads(res.stdout).get("frames", [])
            for frame in frames:
                pts = frame.get("pkt_pts_time")
                if pts is not None:
                    samples.append(SceneChange(timestamp=float(pts)))
        except Exception as e:
            pass
            
        return samples

    def score_candidates(self, vis: List[VisualSample], aud: List[AudioSample], scenes: List[SceneChange], duration: float) -> List[Candidate]:
        candidates = []
        
        # Normalize vectors if needed, but they are already mostly 0-1
        max_ts = math.ceil(duration)
        
        for t in range(max_ts):
            ts = float(t)
            
            # Find closest samples
            v_samp = next((v for v in vis if abs(v.timestamp - ts) < 0.5), None)
            a_samp = next((a for a in aud if abs(a.timestamp - ts) < 0.5), None)
            
            motion = v_samp.motion if v_samp else 0.0
            detail = v_samp.detail if v_samp else 0.0
            audio_energy = a_samp.rms_level if a_samp else 0.0
            
            # Distance to nearest scene change
            scene_prox = 0.0
            if scenes:
                min_dist = min(abs(s.timestamp - ts) for s in scenes)
                if min_dist < 2.0:
                    scene_prox = max(0.0, 1.0 - (min_dist / 2.0))
            
            # Scoring logic (weighted)
            # High audio energy + some motion + high detail + near a scene change
            score = (audio_energy * 0.4) + (motion * 0.3) + (detail * 0.2) + (scene_prox * 0.1)
            score = min(1.0, max(0.0, score))
            
            candidates.append(Candidate(
                timestamp=ts,
                window_start=max(0.0, ts - 1.0),
                window_end=min(duration, ts + 2.0),
                score=score,
                signals=CandidateSignal(
                    motion=motion,
                    audio_energy=audio_energy,
                    scene_change=scene_prox,
                    visual_detail=detail
                )
            ))
            
        # Sort by score descending and return top N (e.g. 5)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:5]

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Deterministic Candidate Analyzer")
    args = parser.parse_args()
    
    config = RenderConfig.from_json("job.json")
    
    print(f"Analyzing source video (Deterministic Mode)...")
    import time
    t0 = time.time()
    
    analyzer = CandidateAnalyzer(config.source_video)
    base_analyzer = MediaAnalyzer(config.source_video)
    meta = base_analyzer.get_metadata()
    
    # Ensure frames exist (using the media analyzer logic)
    frames_dir = str(Path("working/frames").resolve())
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        print("Extracting sparse visual samples...")
        base_analyzer.extract_frames(frames_dir, interval_seconds=1.0)
    else:
        print("Reusing existing visual samples.")
        
    print("Calculating visual signals...")
    vis_samples = analyzer.analyze_visuals(frames_dir)
    
    print("Calculating audio signals...")
    aud_samples = analyzer.analyze_audio()
    
    print("Detecting scene changes...")
    scene_samples = analyzer.analyze_scenes()
    
    print("Scoring candidate windows...")
    candidates = analyzer.score_candidates(vis_samples, aud_samples, scene_samples, meta.duration)
    
    output = CandidateAnalysisOutput(
        source={
            "duration": meta.duration,
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "has_audio": meta.has_audio
        },
        visual_samples=vis_samples,
        audio_samples=aud_samples,
        scene_changes=scene_samples,
        candidates=candidates
    )
    
    out_path = Path("working/candidate_analysis.json")
    os.makedirs("working", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(output), f, indent=2)
        
    t1 = time.time()
    print(f"\nAnalysis complete in {t1-t0:.2f} seconds.")
    print("Top 5 Candidate Timestamps:")
    for i, c in enumerate(candidates, 1):
        print(f" {i}. {c.timestamp}s (Score: {c.score:.2f}) [Start: {c.window_start}s, End: {c.window_end}s]")

if __name__ == "__main__":
    main()

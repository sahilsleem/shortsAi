import argparse
import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict

from .config import RenderConfig
from .video_ops import get_ffprobe_path, get_ffmpeg_path

@dataclass
class MediaSourceMetadata:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None

@dataclass
class FrameData:
    timestamp: float
    path: str

@dataclass
class TranscriptData:
    available: bool
    text: str
    segments: List[Dict[str, Any]]

@dataclass
class AnalysisOutput:
    source: MediaSourceMetadata
    frames: List[FrameData]
    transcript: TranscriptData

class MediaAnalyzer:
    def __init__(self, source_path: str):
        self.source_path = str(Path(source_path).resolve())
        
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source video not found: {self.source_path}")
            
    def get_metadata(self) -> MediaSourceMetadata:
        cmd = [
            get_ffprobe_path(),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            self.source_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream
                
        if not video_stream:
            raise ValueError("No video stream found in the source file.")
            
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        video_codec = video_stream.get("codec_name", "")
        
        # Duration
        duration_str = data.get("format", {}).get("duration")
        if not duration_str and video_stream:
            duration_str = video_stream.get("duration")
        duration = float(duration_str) if duration_str else 0.0
            
        # FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        parts = fps_str.split('/')
        if len(parts) == 2 and parts[1] != '0':
            fps = float(parts[0]) / float(parts[1])
        else:
            fps = 0.0
            
        # Audio
        has_audio = audio_stream is not None
        audio_sr = int(audio_stream.get("sample_rate", 0)) if has_audio else None
        audio_ch = int(audio_stream.get("channels", 0)) if has_audio else None
        
        return MediaSourceMetadata(
            path=self.source_path.replace('\\', '/'),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video_codec,
            has_audio=has_audio,
            audio_sample_rate=audio_sr,
            audio_channels=audio_ch
        )

    def extract_frames(self, output_dir: str, interval_seconds: float = 1.0) -> List[FrameData]:
        os.makedirs(output_dir, exist_ok=True)
        
        # Use FFmpeg to extract frames at 1/interval_seconds fps
        fps_filter = f"fps={1.0 / interval_seconds}"
        
        # Save frames as frame_0001.jpg, etc.
        pattern = os.path.join(output_dir, "frame_%04d.jpg")
        
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-i", self.source_path,
            "-vf", fps_filter,
            "-q:v", "2",
            pattern
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        frames = []
        
        # Collect extracted frames and assign approx timestamps
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith("frame_") and filename.endswith(".jpg"):
                idx_str = filename[6:10]
                idx = int(idx_str)
                timestamp = (idx - 1) * interval_seconds
                frames.append(FrameData(
                    timestamp=timestamp,
                    path=os.path.join(output_dir, filename).replace('\\', '/')
                ))
                
        return frames

class WhisperAdapter:
    def __init__(self, model_size="tiny", compute_type="int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.is_installed = False
        self.model = None
        
        try:
            import faster_whisper
            self.is_installed = True
        except ImportError:
            pass
            
    def transcribe(self, audio_path: str, cache_dir: str = "working") -> TranscriptData:
        cache_file = os.path.join(cache_dir, "transcript.json")
        
        if os.path.exists(cache_file):
            print("Loading cached transcript...")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return TranscriptData(
                        available=data.get("available", False),
                        text=data.get("text", ""),
                        segments=data.get("segments", [])
                    )
            except Exception as e:
                print(f"Failed to load transcript cache: {e}")
                
        if not self.is_installed:
            print("Local Whisper (faster-whisper) is not installed. Transcription is unavailable.")
            return TranscriptData(
                available=False,
                text="",
                segments=[]
            )
            
        try:
            from faster_whisper import WhisperModel
            
            if self.model is None:
                print(f"Loading Whisper model '{self.model_size}' on CPU...")
                self.model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)
                
            print("Transcribing video audio with faster-whisper...")
            # Set condition_on_previous_text=False for faster/stable transcription
            segments_gen, info = self.model.transcribe(audio_path, beam_size=5, condition_on_previous_text=False)
            
            segments = []
            full_text = []
            
            for segment in segments_gen:
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                full_text.append(segment.text.strip())
                
            result = TranscriptData(
                available=True,
                text=" ".join(full_text),
                segments=segments
            )
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)
                
            return result
            
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            return TranscriptData(
                available=False,
                text="",
                segments=[]
            )

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Source Video Analyzer")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    config = RenderConfig.from_json(args.config)
    
    analyzer = MediaAnalyzer(config.source_video)
    
    print("Analyzing source video...")
    metadata = analyzer.get_metadata()
    print("Metadata extracted.")
    
    frames_dir = str(Path("working/frames").resolve())
    print(f"Extracting frames to working/frames...")
    frames = analyzer.extract_frames(frames_dir, interval_seconds=1.0)
    print(f"Extracted {len(frames)} frames.")
    
    transcriber = WhisperAdapter()
    if metadata.has_audio:
        print("Attempting local transcription...")
        transcript = transcriber.transcribe(config.source_video)
    else:
        print("Source video has no audio. Skipping transcription.")
        transcript = TranscriptData(
            available=False,
            text="",
            segments=[]
        )
    
    output = AnalysisOutput(
        source=metadata,
        frames=frames,
        transcript=transcript
    )
    
    output_json = json.dumps(asdict(output), indent=2)
    
    os.makedirs("working", exist_ok=True)
    out_path = Path("working/analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_json)
        
    print("Analysis complete. Results written to working/analysis.json")

if __name__ == "__main__":
    main()

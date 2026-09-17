import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.analyze import MediaAnalyzer, WhisperAdapter, AnalysisOutput, TranscriptData
from src.config import RenderConfig

def test_media_analyzer():
    # Load config to get the real source video
    config = RenderConfig.from_json("job.json")
    
    print("Testing analyzer with input...", flush=True)
    analyzer = MediaAnalyzer(config.source_video)
    
    # 1. Test Metadata extraction
    metadata = analyzer.get_metadata()
    assert metadata.width > 0
    
    # 2. Test Frame extraction
    test_frames_dir = str(Path("working/test_frames").resolve())
    frames = analyzer.extract_frames(test_frames_dir, interval_seconds=5.0)
    assert len(frames) > 0
    
    # 3. Test Whisper adapter
    transcriber = WhisperAdapter(model_size="tiny")
    print(f"Whisper is_installed: {transcriber.is_installed}")
    
    cache_dir = "working/test_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # First run (no cache)
    print("Testing transcription (first run, no cache)...")
    transcript1 = transcriber.transcribe(config.source_video, cache_dir=cache_dir)
    
    if transcriber.is_installed:
        assert transcript1.available, "Transcription should be available"
        assert len(transcript1.segments) > 0, "Segments should be generated"
        assert len(transcript1.text) > 0, "Text should be generated"
        assert "start" in transcript1.segments[0]
        assert "end" in transcript1.segments[0]
        assert "text" in transcript1.segments[0]
        
        # Second run (cached)
        print("Testing transcription (cached run)...")
        transcript2 = transcriber.transcribe(config.source_video, cache_dir=cache_dir)
        assert transcript2.available == transcript1.available
        assert transcript2.text == transcript1.text
        assert len(transcript2.segments) == len(transcript1.segments)
    else:
        assert not transcript1.available, "Stub should report unavailable"
        
    # 4. Test no audio
    no_audio_analyzer = MediaAnalyzer("working/no_audio_test.mp4")
    no_audio_meta = no_audio_analyzer.get_metadata()
    assert not no_audio_meta.has_audio
    
    # Transcription on no audio
    print("Testing transcription on no-audio file...")
    # Whisper typically handles no audio gracefully by returning empty segments or throwing an error
    # Our adapter catches errors and returns available=False
    no_audio_transcript = transcriber.transcribe("working/no_audio_test.mp4", cache_dir="working/test_cache_no_audio")
    
    print("Transcription adapter OK.", flush=True)

if __name__ == "__main__":
    test_media_analyzer()
    print("All tests passed!", flush=True)

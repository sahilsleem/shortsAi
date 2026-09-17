import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.vision import OllamaVisionAdapter, VisionAnalyzer, FrameVisionData

def test_vision_analyzer():
    print("Testing OllamaVisionAdapter...", flush=True)
    adapter = OllamaVisionAdapter(model_name="qwen3-vl:2b")
    
    print(f"Ollama is_available: {adapter.is_available}")
    print(f"Model exists: {adapter.model_exists}")
    
    test_frame_path = "working/test_frame.jpg"
    os.makedirs("working", exist_ok=True)
    
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(test_frame_path)
    
    test_frames_dir = "working/test_frames_vision"
    if os.path.exists(test_frames_dir):
        shutil.rmtree(test_frames_dir)
    os.makedirs(test_frames_dir, exist_ok=True)
    
    shutil.copy(test_frame_path, os.path.join(test_frames_dir, "frame_0001.jpg"))
    shutil.copy(test_frame_path, os.path.join(test_frames_dir, "frame_0002.jpg"))
    
    cache_file = "working/test_vision_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    analyzer = VisionAnalyzer(adapter)
    
    print("Testing basic analysis (might fail gracefully if Ollama missing)...")
    output = analyzer.analyze_frames(test_frames_dir, cache_file)
    
    assert len(output.frames) > 0, "Should have returned frames"
    
    if not adapter.is_available or not adapter.model_exists:
        assert output.frames[0].error != "", "Should report error when missing"
        assert "not installed" in output.frames[0].error
        print("Missing dependency handling OK.", flush=True)
    else:
        print("Ollama is available. Testing output.", flush=True)
        
    print("Testing caching...")
    output2 = analyzer.analyze_frames(test_frames_dir, cache_file)
    assert len(output2.frames) == len(output.frames)
    print("Caching OK.", flush=True)
    
if __name__ == "__main__":
    test_vision_analyzer()
    print("All tests passed!", flush=True)

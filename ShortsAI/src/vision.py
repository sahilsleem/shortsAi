import os
import json
import base64
import argparse
import urllib.request
import urllib.error
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict

from .config import RenderConfig

@dataclass
class FrameVisionData:
    timestamp: float
    description: str = ""
    people: List[str] = None
    actions: List[str] = None
    expressions: List[str] = None
    interaction: str = ""
    interestingness: int = 0
    error: str = ""
    inference_time: float = 0.0

@dataclass
class VisionAnalysisOutput:
    frames: List[FrameVisionData]

class OllamaVisionAdapter:
    def __init__(self, model_name="qwen3-vl:2b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.is_available = self._check_availability()
        self.model_exists = self._check_model_exists() if self.is_available else False
        
    def _check_availability(self):
        try:
            req = urllib.request.Request("http://localhost:11434/", method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
            
        import shutil
        if shutil.which("ollama"):
            return True
            
        return False

    def _check_model_exists(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    models = [m.get("name") for m in data.get("models", [])]
                    for m in models:
                        if m == self.model_name or m.startswith(self.model_name + ":") or m == self.model_name + ":latest":
                            return True
            return False
        except Exception:
            return False

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def analyze_frame(self, frame_path: str, timestamp: float) -> FrameVisionData:
        if not self.is_available:
            return FrameVisionData(
                timestamp=timestamp,
                error="Ollama is not installed or not running."
            )
            
        if not self.model_exists:
            return FrameVisionData(
                timestamp=timestamp,
                error=f"Model {self.model_name} is not installed in Ollama."
            )
            
        prompt = (
            "Analyze this frame and provide a JSON response with the following keys exactly: "
            "'description' (brief summary), "
            "'people' (list of people descriptions, e.g. ['man in red shirt']), "
            "'actions' (list of actions happening), "
            "'expressions' (list of visible facial expressions), "
            "'interaction' (describe how people/objects are interacting), "
            "'interestingness' (integer 1-10 on how engaging this frame is). "
            "Only output valid JSON."
        )
        
        start_time = time.time()
        try:
            b64_img = self.encode_image(frame_path)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [b64_img],
                "format": "json"
            }
            
            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(self.api_url, data=data_bytes, headers={'Content-Type': 'application/json'}, method='POST')
            
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            inference_time = time.time() - start_time
            response_text = result.get("response", "{}")
            try:
                data = json.loads(response_text)
                return FrameVisionData(
                    timestamp=timestamp,
                    description=data.get("description", ""),
                    people=data.get("people", []),
                    actions=data.get("actions", []),
                    expressions=data.get("expressions", []),
                    interaction=data.get("interaction", ""),
                    interestingness=int(data.get("interestingness", 0)),
                    inference_time=inference_time
                )
            except json.JSONDecodeError:
                return FrameVisionData(
                    timestamp=timestamp,
                    error="Failed to parse JSON from model.",
                    inference_time=inference_time
                )
                
        except Exception as e:
            inference_time = time.time() - start_time
            return FrameVisionData(
                timestamp=timestamp,
                error=f"Model error: {str(e)}",
                inference_time=inference_time
            )

class VisionAnalyzer:
    def __init__(self, adapter: OllamaVisionAdapter):
        self.adapter = adapter
        
    def analyze_frames(self, frames_dir: str, cache_file: str) -> VisionAnalysisOutput:
        cache = {}
        
        if os.path.exists(cache_file):
            print("Loading cached vision analysis...")
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    for frame_data in cached_data.get("frames", []):
                        ts = frame_data.get("timestamp")
                        if ts is not None:
                            cache[ts] = FrameVisionData(**frame_data)
            except Exception as e:
                print(f"Failed to load cache: {e}")
                
        results = []
        
        if not os.path.exists(frames_dir):
            print(f"Frames directory not found: {frames_dir}")
            return VisionAnalysisOutput(frames=[])
            
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        
        for filename in frame_files:
            try:
                idx = int(filename[6:10])
                timestamp = float(idx - 1)
            except ValueError:
                continue
                
            if timestamp in cache and not cache[timestamp].error:
                print(f"Using cached analysis for frame {filename} (ts={timestamp})")
                results.append(cache[timestamp])
            else:
                print(f"Analyzing {filename}...")
                frame_path = os.path.join(frames_dir, filename)
                res = self.adapter.analyze_frame(frame_path, timestamp)
                results.append(res)
                if res.error:
                    print(f"Error on {filename}: {res.error}")
                    if "Ollama is not installed" in res.error or "is not installed in Ollama" in res.error:
                        print("Stopping further analysis due to missing dependency.")
                        break
                        
        output = VisionAnalysisOutput(frames=results)
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(asdict(output), f, indent=2)
            
        return output

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Vision Analyzer")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    config = RenderConfig.from_json(args.config)
    
    adapter = OllamaVisionAdapter(model_name="qwen3-vl:2b")
    
    if not adapter.is_available:
        print("ERROR: Ollama is not installed or not running on this machine.")
    elif not adapter.model_exists:
        print(f"ERROR: Model {adapter.model_name} is not installed in Ollama.")
    else:
        print(f"Ollama detected. Will use model: {adapter.model_name}")
        
    frames_dir = str(Path("working/frames").resolve())
    cache_file = str(Path("working/vision_analysis.json").resolve())
    
    analyzer = VisionAnalyzer(adapter)
    output = analyzer.analyze_frames(frames_dir, cache_file)
    
    print(f"Vision analysis complete. Results saved to {cache_file}")

if __name__ == "__main__":
    main()

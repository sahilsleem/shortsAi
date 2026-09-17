import os
import sys
import json
import pytest
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.server import app

client = TestClient(app)

def test_missing_video():
    response = client.post("/render", data={
        "hook_main_text": "Main",
        "hook_start": 0.0,
        "hook_end": 2.0,
        "reveal_main_text": "Reveal",
        "reveal_start": 2.0,
        "reveal_end": 4.0
    })
    # FastAPI automatically raises 422 Unprocessable Entity when required File is missing
    assert response.status_code == 422

def test_empty_upload(tmp_path):
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    
    with open(empty_file, "rb") as f:
        response = client.post("/render", data={
            "hook_main_text": "Main",
            "hook_start": 0.0,
            "hook_end": 2.0,
            "reveal_main_text": "Reveal",
            "reveal_start": 2.0,
            "reveal_end": 4.0
        }, files={"video": ("empty.mp4", f, "video/mp4")})
        
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_invalid_timings(tmp_path):
    # Need a small dummy file to pass the empty check and reach validation
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_bytes(b"dummy content")
    
    with open(dummy_file, "rb") as f:
        # hook_start > hook_end
        response = client.post("/render", data={
            "hook_main_text": "Main",
            "hook_start": 5.0,
            "hook_end": 2.0,
            "reveal_main_text": "Reveal",
            "reveal_start": 2.0,
            "reveal_end": 4.0
        }, files={"video": ("dummy.mp4", f, "video/mp4")})
        
    assert response.status_code == 400
    assert "hook_start must be less than hook_end" in response.json()["detail"]
    
    with open(dummy_file, "rb") as f:
        # Negative timings
        response = client.post("/render", data={
            "hook_main_text": "Main",
            "hook_start": -1.0,
            "hook_end": 2.0,
            "reveal_main_text": "Reveal",
            "reveal_start": 2.0,
            "reveal_end": 4.0
        }, files={"video": ("dummy.mp4", f, "video/mp4")})
        
    assert response.status_code == 400
    assert "negative" in response.json()["detail"].lower()

def test_successful_request_validation_without_render(monkeypatch, tmp_path):
    # We want to mock out the FFprobe duration check and the render/image generation
    # so we can test that UUID isolation and the logic flow works up to the render point.
    
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_bytes(b"some valid video content bytes")
    
    # Mock validate_and_get_duration
    monkeypatch.setattr("src.server.validate_and_get_duration", lambda path: 10.0)
    
    # Mock the image generation and render so it doesn't crash on dummy file
    monkeypatch.setattr("src.server.compute_best_font_size", lambda *args, **kwargs: 30)
    monkeypatch.setattr("src.server.generate_text_overlay", lambda *args, **kwargs: None)
    
    req_uuid_spy = []
    
    def mock_render_video(*args, **kwargs):
        # Extract the workspace path to verify UUID isolation
        out_path = kwargs.get("output_video")
        req_uuid_spy.append(out_path)
        # Create dummy output file to satisfy FileResponse
        with open(out_path, "w") as f:
            f.write("rendered")
            
    monkeypatch.setattr("src.server.render_video", mock_render_video)
    
    with open(dummy_file, "rb") as f:
        response = client.post("/render", data={
            "hook_main_text": "Main",
            "hook_start": 0.0,
            "hook_end": 2.0,
            "reveal_main_text": "Reveal",
            "reveal_start": 2.0,
            "reveal_end": 4.0
        }, files={"video": ("dummy.mp4", f, "video/mp4")})
        
    assert response.status_code == 200
    # It returned a file!
    assert response.headers["content-type"] == "video/mp4"
    
    # Verify UUID isolation was used
    out_path = req_uuid_spy[0]
    assert "working" in out_path
    
    # Extract the UUID part from the path
    # Path looks like: /something/working/uuid/output.mp4
    parts = Path(out_path).parts
    uuid_dir = parts[-2]
    
    # UUID should be 36 chars
    assert len(uuid_dir) == 36
    
    # Verify cleanup (Background tasks might take a tiny bit to run, but TestClient runs them synchronously)
    assert not os.path.exists(Path(out_path).parent)

if __name__ == "__main__":
    pytest.main(["-v", __file__])

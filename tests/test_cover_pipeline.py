"""
tests/test_cover_pipeline.py - Test suite for Phase 5 Saba Bollywood Cover Frame Pipeline

Tests:
1. Cover toggle OFF preserves normal render path without invoking cover pipeline.
2. Missing or empty thumbnail_phrase safely skips cover frame without breaking render.
3. Audio properties inspection via ffprobe.
4. Duration-controlled (0.100s) cover video segment command generation with matched audio properties.
5. Video concatenation command generation with audio sync and stream mapping.
6. Silent main video concatenation handling (a=0).
7. Frame selector operates on original source video footage.
8. Output cover image is saved with 1080x1080 dimensions.
9. Failure safety: errors during cover stage safely preserve the normal rendered Short.
10. Server /render endpoint handles cover_enabled and thumbnail_phrase correctly.
11. Server /output_cover.jpg endpoint serves saved cover artwork.
"""

import os
import sys
import io
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.video_ops import (
    get_audio_properties,
    create_cover_video_segment,
    concat_video_with_cover,
    append_cover_frame,
    render_main_video,
    render_curiosity_video,
)
from src.server import ShortsAIHandler


def test_missing_thumbnail_phrase_safe_skip(tmp_path):
    """When thumbnail_phrase is empty, append_cover_frame must skip safely without raising an error."""
    source_vid = str(tmp_path / "source.mp4")
    rendered_vid = str(tmp_path / "rendered.mp4")
    output_vid = str(tmp_path / "output.mp4")

    # Create dummy rendered file
    with open(rendered_vid, "w") as f:
        f.write("dummy video data")

    res = append_cover_frame(
        source_video=source_vid,
        rendered_video=rendered_vid,
        output_video=output_vid,
        thumbnail_phrase="",
        start_time=0.0,
        end_time=5.0,
        crop_x=0,
        crop_y=0,
        crop_size=1080
    )

    assert res["success"] is False
    assert res["reason"] == "no_thumbnail_phrase"
    assert os.path.exists(output_vid)
    with open(output_vid, "r") as f:
        assert f.read() == "dummy video data"


def test_get_audio_properties_with_mocked_ffprobe(monkeypatch):
    """Verify get_audio_properties extracts sample rate, channels, and layout."""
    mock_probe = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2, "channel_layout": "stereo"}
        ]
    }

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(mock_probe))

    monkeypatch.setattr(subprocess, "run", mock_run)

    props = get_audio_properties("dummy.mp4")
    assert props["has_audio"] is True
    assert props["sample_rate"] == 48000
    assert props["channels"] == 2
    assert props["channel_layout"] == "stereo"
    assert props["codec_name"] == "aac"


def test_get_audio_properties_silent_video(monkeypatch):
    """Verify get_audio_properties correctly identifies video without audio stream."""
    mock_probe = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920}
        ]
    }

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(mock_probe))

    monkeypatch.setattr(subprocess, "run", mock_run)

    props = get_audio_properties("dummy.mp4")
    assert props["has_audio"] is False


def test_create_cover_video_segment_command_generation(monkeypatch, tmp_path):
    """Verify create_cover_video_segment builds exact FFmpeg command with 0.100s duration controls."""
    captured_cmd = None

    def mock_run(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        # Create output file so existence checks pass
        with open(cmd[-1], "w") as f:
            f.write("cover segment")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    cover_img = str(tmp_path / "cover.jpg")
    with open(cover_img, "w") as f:
        f.write("image")
    out_seg = str(tmp_path / "segment.mp4")

    audio_props = {"has_audio": True, "sample_rate": 44100, "channel_layout": "stereo", "channels": 2}
    result_path = create_cover_video_segment(
        cover_image_path=cover_img,
        output_segment_path=out_seg,
        duration=0.100,
        audio_props=audio_props
    )

    assert captured_cmd is not None
    cmd_str = " ".join(captured_cmd)

    # 1. Image input looped with explicit duration
    assert "-loop 1" in cmd_str
    assert "-t 0.100" in cmd_str
    assert cover_img in captured_cmd

    # 2. Silent audio matching properties
    assert "anullsrc=r=44100:cl=stereo" in cmd_str

    # 3. Canvas geometry: scale=1002:1002, canvas=1080x1920, overlay=39:420
    assert "scale=1002:1002" in cmd_str
    assert "s=1080x1920" in cmd_str
    assert "overlay=39:420" in cmd_str

    # 4. Explicit duration and timestamp trimming
    assert "trim=duration=0.100" in cmd_str
    assert "atrim=duration=0.100" in cmd_str
    assert "setpts=PTS-STARTPTS" in cmd_str

    # 5. Output codec and framerate
    assert "-r 30" in cmd_str
    assert "-c:v" in captured_cmd
    assert "libx264" in captured_cmd
    assert "-pix_fmt" in captured_cmd
    assert "yuv420p" in captured_cmd


def test_concat_video_with_cover_command(monkeypatch, tmp_path):
    """Verify concat_video_with_cover builds valid concat filter complex."""
    captured_cmd = None

    def mock_run(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    main_vid = str(tmp_path / "main.mp4")
    cover_seg = str(tmp_path / "seg.mp4")
    out_vid = str(tmp_path / "final.mp4")

    # With audio
    concat_video_with_cover(main_vid, cover_seg, out_vid, has_audio=True)
    assert captured_cmd is not None
    cmd_str = " ".join(captured_cmd)
    assert "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1" in cmd_str
    assert "-map [outv]" in cmd_str
    assert "-map [outa]" in cmd_str

    # Without audio
    captured_cmd = None
    concat_video_with_cover(main_vid, cover_seg, out_vid, has_audio=False)
    assert captured_cmd is not None
    cmd_str_no_a = " ".join(captured_cmd)
    assert "[0:v][1:v]concat=n=2:v=1:a=0" in cmd_str_no_a
    assert "-map [outv]" in cmd_str_no_a
    assert "-map [outa]" not in cmd_str_no_a


def test_append_cover_frame_end_to_end_flow(monkeypatch, tmp_path):
    """Verify complete append_cover_frame orchestration with mocked FFmpeg and selector."""
    source_vid = str(tmp_path / "source.mp4")
    rendered_vid = str(tmp_path / "rendered.mp4")
    output_vid = str(tmp_path / "output.mp4")
    cover_img = str(tmp_path / "output_cover.jpg")

    with open(source_vid, "w") as f: f.write("source")
    with open(rendered_vid, "w") as f: f.write("rendered video")

    # Mock frame selection diagnostic report
    mock_report = {
        "selected_candidate": {
            "timestamp": 2.50,
            "rank": 1,
            "total_score": 85
        }
    }
    monkeypatch.setattr(
        "src.frame_selector.run_frame_selection_diagnostic",
        lambda *args, **kwargs: mock_report
    )

    # Mock extract_full_res_frame
    dummy_frame = Image.new("RGB", (1080, 1080), color=(180, 50, 50))
    monkeypatch.setattr(
        "src.frame_selector.extract_full_res_frame",
        lambda *args, **kwargs: dummy_frame
    )

    # Mock get_audio_properties
    monkeypatch.setattr(
        "src.video_ops.get_audio_properties",
        lambda *args, **kwargs: {"has_audio": True, "sample_rate": 44100, "channel_layout": "stereo", "channels": 2, "codec_name": "aac"}
    )

    # Mock get_video_dimensions
    monkeypatch.setattr(
        "src.video_ops.get_video_dimensions",
        lambda *args, **kwargs: (1080, 1920)
    )

    # Mock subprocess.run
    def mock_run(cmd, *args, **kwargs):
        out_target = cmd[-1]
        if out_target.endswith(".mp4"):
            with open(out_target, "w") as f:
                f.write("final video content")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    res = append_cover_frame(
        source_video=source_vid,
        rendered_video=rendered_vid,
        output_video=output_vid,
        thumbnail_phrase="SALMAN DID THIS 😳",
        start_time=1.0,
        end_time=6.0,
        crop_x=0,
        crop_y=0,
        crop_size=1080,
        mode="main",
        cover_image_path=cover_img
    )

    assert res["success"] is True
    assert res["best_timestamp"] == 2.50
    assert res["selected_phrase"] == "SALMAN DID THIS 😳"
    assert os.path.exists(cover_img)
    with Image.open(cover_img) as img:
        assert img.size == (1080, 1080)
    assert os.path.exists(output_vid)


def test_append_cover_frame_failure_safety_preserves_original(monkeypatch, tmp_path):
    """When an error occurs during cover generation, normal render must be preserved."""
    source_vid = str(tmp_path / "source.mp4")
    rendered_vid = str(tmp_path / "rendered.mp4")
    output_vid = str(tmp_path / "output.mp4")
    cover_img = str(tmp_path / "output_cover.jpg")

    with open(rendered_vid, "w") as f: f.write("original rendered video")

    # Force error during frame selection
    def mock_err(*args, **kwargs):
        raise RuntimeError("Simulated frame extraction crash")

    monkeypatch.setattr("src.frame_selector.run_frame_selection_diagnostic", mock_err)

    res = append_cover_frame(
        source_video=source_vid,
        rendered_video=rendered_vid,
        output_video=output_vid,
        thumbnail_phrase="SALMAN DID THIS 😳",
        start_time=1.0,
        end_time=6.0,
        crop_x=0,
        crop_y=0,
        crop_size=1080,
        cover_image_path=cover_img
    )

    assert res["success"] is False
    assert "Simulated frame extraction crash" in res["error"]
    assert os.path.exists(output_vid)
    with open(output_vid, "r") as f:
        assert f.read() == "original rendered video"


def test_server_render_endpoint_with_cover_enabled(monkeypatch):
    """Verify server POST /render executes cover pipeline when cover_enabled is True."""
    append_called = False

    def mock_render_main(*args, **kwargs):
        output_video = args[1]
        with open(output_video, "w") as f:
            f.write("rendered main video")

    def mock_append(*args, **kwargs):
        nonlocal append_called
        append_called = True
        out_v = kwargs.get("output_video") or (args[2] if len(args) > 2 else "output.mp4")
        return {"success": True, "output_video": out_v}

    monkeypatch.setattr("src.server.render_main_video", mock_render_main)
    monkeypatch.setattr("src.server.append_cover_frame", mock_append)
    monkeypatch.setattr("src.server.get_duration", lambda p: 10.0)
    monkeypatch.setattr("src.server.compute_best_font_size", lambda *args, **kwargs: 60)
    monkeypatch.setattr("src.server.generate_text_overlay", lambda *args, **kwargs: None)

    class DummyHandler(ShortsAIHandler):
        def __init__(self, path, body, boundary):
            self.path = path
            self.headers = {
                "Content-Length": str(len(body)),
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.response_code = None
            self.response_headers = {}
            self.client_address = ('127.0.0.1', 8000)

        def send_response(self, code, message=None):
            self.response_code = code

        def send_header(self, keyword, value):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

        def send_error(self, code, message=None):
            self.response_code = code

        def log_message(self, *args):
            pass

    boundary = "----WebKitFormBoundaryDummy123"
    body_parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nmain\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"start_time\"\r\n\r\n0.0\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"end_time\"\r\n\r\n5.0\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"cover_enabled\"\r\n\r\ntrue\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"thumbnail_phrase\"\r\n\r\nSALMAN DID THIS 😳\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"test.mp4\"\r\nContent-Type: video/mp4\r\n\r\nfake_video_bytes\r\n",
        f"--{boundary}--\r\n"
    ]
    raw_body = "".join(body_parts).encode("utf-8")

    handler = DummyHandler('/render', raw_body, boundary)
    handler.do_POST()

    assert handler.response_code == 200
    assert append_called is True


def test_server_render_endpoint_with_cover_disabled(monkeypatch):
    """Verify server POST /render skips cover pipeline when cover_enabled is False."""
    append_called = False

    def mock_render_main(*args, **kwargs):
        output_video = args[1]
        with open(output_video, "w") as f:
            f.write("rendered main video")

    def mock_append(*args, **kwargs):
        nonlocal append_called
        append_called = True
        return {"success": True, "output_video": args[2]}

    monkeypatch.setattr("src.server.render_main_video", mock_render_main)
    monkeypatch.setattr("src.server.append_cover_frame", mock_append)
    monkeypatch.setattr("src.server.get_duration", lambda p: 10.0)
    monkeypatch.setattr("src.server.compute_best_font_size", lambda *args, **kwargs: 60)
    monkeypatch.setattr("src.server.generate_text_overlay", lambda *args, **kwargs: None)

    class DummyHandler(ShortsAIHandler):
        def __init__(self, path, body, boundary):
            self.path = path
            self.headers = {
                "Content-Length": str(len(body)),
                "Content-Type": f"multipart/form-data; boundary={boundary}"
            }
            self.rfile = io.BytesIO(body)
            self.wfile = io.BytesIO()
            self.response_code = None
            self.response_headers = {}
            self.client_address = ('127.0.0.1', 8000)

        def send_response(self, code, message=None):
            self.response_code = code

        def send_header(self, keyword, value):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

        def send_error(self, code, message=None):
            self.response_code = code

        def log_message(self, *args):
            pass

    boundary = "----WebKitFormBoundaryDummy456"
    body_parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nmain\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"start_time\"\r\n\r\n0.0\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"end_time\"\r\n\r\n5.0\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"cover_enabled\"\r\n\r\nfalse\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"test.mp4\"\r\nContent-Type: video/mp4\r\n\r\nfake_video_bytes\r\n",
        f"--{boundary}--\r\n"
    ]
    raw_body = "".join(body_parts).encode("utf-8")

    handler = DummyHandler('/render', raw_body, boundary)
    handler.do_POST()

    assert handler.response_code == 200
    assert append_called is False


def test_server_serves_output_cover_image(tmp_path):
    """Verify server GET /output_cover.jpg serves the saved cover image."""
    cover_file = Path("working/output_cover.jpg").resolve()
    os.makedirs(cover_file.parent, exist_ok=True)
    with open(cover_file, "wb") as f:
        f.write(b"fake jpeg content")

    class DummyHandler(ShortsAIHandler):
        def __init__(self, path):
            self.path = path
            self.wfile = io.BytesIO()
            self.response_code = None
            self.response_headers = {}

        def send_response(self, code, message=None):
            self.response_code = code

        def send_header(self, keyword, value):
            self.response_headers[keyword] = value

        def end_headers(self):
            pass

    handler = DummyHandler('/output_cover.jpg')
    handler.do_GET()

    assert handler.response_code == 200
    assert handler.response_headers.get('Content-Type') == 'image/jpeg'
    assert handler.wfile.getvalue() == b"fake jpeg content"

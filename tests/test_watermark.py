import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.video_ops import DEFAULT_WATERMARK_PATH, render_main_video, render_curiosity_video

def test_watermark_asset_exists_and_valid():
    """Verify that the watermark file exists, is valid RGBA PNG, and meets size requirements."""
    assert os.path.exists(DEFAULT_WATERMARK_PATH), f"Watermark missing at {DEFAULT_WATERMARK_PATH}"
    
    with Image.open(DEFAULT_WATERMARK_PATH) as img:
        assert img.mode == "RGBA"
        w, h = img.size
        # Target height is around 18-24px (specifically 20px)
        assert 15 <= h <= 25, f"Unexpected watermark height: {h}px"
        assert w > h, f"Watermark width {w} should be greater than height {h}"
        
        # Verify opacity: watermark is subtle (semi-transparent), not solid opaque 255
        extrema = img.getextrema()
        alpha_min, alpha_max = extrema[3]
        # Should contain semi-transparent pixels (alpha < 200)
        assert alpha_max > 0, "Watermark is completely transparent"
        assert alpha_max < 220, f"Watermark alpha max is {alpha_max}, expected subtle semi-transparency (<220)"

def test_watermark_coordinates_and_bounds():
    """Verify watermark geometry and positioning inside the 1002x1002 video."""
    with Image.open(DEFAULT_WATERMARK_PATH) as img:
        wm_w, wm_h = img.size
        
    vid_x, vid_y = 39, 420
    vid_w, vid_h = 1002, 1002
    vid_bottom = vid_y + vid_h  # 1422
    vid_right = vid_x + vid_w   # 1041
    vid_center_x = vid_x + vid_w / 2.0  # 540.0
    
    # In FFmpeg overlay: x=(W-w)/2 on 1080 canvas
    wm_x = (1080 - wm_w) // 2
    wm_center_x = wm_x + wm_w / 2.0
    
    # Horizontal centering: must match video center 540
    assert wm_center_x == vid_center_x, f"Watermark center {wm_center_x} != video center {vid_center_x}"
    
    # Vertical positioning: bottom edge at 1374
    wm_y = 1374 - wm_h
    wm_bottom = wm_y + wm_h
    gap_from_video_bottom = vid_bottom - wm_bottom
    
    # Must be 40-55px above video bottom (user specified 40-55px)
    assert 40 <= gap_from_video_bottom <= 55, f"Gap from bottom {gap_from_video_bottom}px outside 40-55px range"
    
    # Must be strictly INSIDE the 1002x1002 video frame
    assert wm_x >= vid_x, f"Watermark left {wm_x} extends past video left {vid_x}"
    assert wm_x + wm_w <= vid_right, f"Watermark right {wm_x + wm_w} extends past video right {vid_right}"
    assert wm_y >= vid_y, f"Watermark top {wm_y} extends above video top {vid_y}"
    assert wm_bottom <= vid_bottom, f"Watermark bottom {wm_bottom} extends below video bottom {vid_bottom}"

def test_render_main_video_includes_watermark(monkeypatch):
    """Verify that render_main_video constructs the FFmpeg command with watermark overlay."""
    captured_cmd = None
    def mock_run(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, 0)
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    render_main_video(
        source_video="dummy_in.mp4",
        output_video="dummy_out.mp4",
        caption_overlay="dummy_overlay.png",
        start_time=0.0,
        end_time=5.0,
        crop_x=50,
        crop_y=80,
        crop_size=900
    )
    
    assert captured_cmd is not None
    cmd_str = " ".join(captured_cmd)
    
    # Watermark input included
    assert DEFAULT_WATERMARK_PATH in captured_cmd
    # Filter complex includes watermark overlay filter
    assert "overlay=x=(W-w)/2:y=1374-h:eof_action=repeat" in cmd_str
    # Output mapped from [outv]
    assert "-map" in captured_cmd and "[outv]" in captured_cmd

def test_render_curiosity_video_includes_persistent_watermark(monkeypatch):
    """Verify that render_curiosity_video overlays watermark AFTER concat so it does not fade during transitions."""
    captured_cmd = None
    def mock_run(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, 0)
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    render_curiosity_video(
        source_video="dummy_in.mp4",
        output_video="dummy_out.mp4",
        hook_overlay="dummy_hook.png",
        reveal_overlay="dummy_reveal.png",
        start_time=0.0,
        cut_time=3.0,
        end_time=6.0,
        crop_x=50,
        crop_y=80,
        crop_size=900
    )
    
    assert captured_cmd is not None
    fc_index = captured_cmd.index("-filter_complex") + 1
    fc_str = captured_cmd[fc_index]
    
    # Watermark input included
    assert DEFAULT_WATERMARK_PATH in captured_cmd
    
    # Critical requirement: watermark must be overlayed on [v_concat] (AFTER concat), NOT on hook/reveal before fade
    assert "concat=n=2:v=1:a=1[v_concat][outa]" in fc_str
    assert "[v_concat][6:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]" in fc_str

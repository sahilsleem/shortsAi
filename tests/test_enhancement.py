import os
import sys
import subprocess
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.video_ops import (
    DEFAULT_WATERMARK_PATH,
    DEFAULT_VIDEO_FILTERS,
    ENHANCED_VIDEO_FILTERS,
    ENHANCED_AUDIO_FILTERS,
    VOICE_FOCUS_AUDIO_FILTERS,
    get_video_filters,
    get_audio_filters,
    render_main_video,
    render_curiosity_video,
)

def test_filter_definitions():
    """Verify default vs enhanced video/audio filter definitions."""
    # Video filters
    assert get_video_filters(enhance=False) == DEFAULT_VIDEO_FILTERS
    assert get_video_filters(enhance=True) == ENHANCED_VIDEO_FILTERS
    assert "hqdn3d" in ENHANCED_VIDEO_FILTERS
    assert "unsharp=5:5:0.5:3:3:0.0" in ENHANCED_VIDEO_FILTERS
    assert "eq=contrast=1.06" in ENHANCED_VIDEO_FILTERS

    # Audio filters
    assert get_audio_filters("enhanced") == ENHANCED_AUDIO_FILTERS
    assert get_audio_filters("voice_focus") == VOICE_FOCUS_AUDIO_FILTERS
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in ENHANCED_AUDIO_FILTERS
    assert "loudnorm=I=-16:TP=-1.5:LRA=8" in VOICE_FOCUS_AUDIO_FILTERS
    assert "equalizer=f=2800" in VOICE_FOCUS_AUDIO_FILTERS

def test_enhance_off_preserves_current_behavior_main(monkeypatch):
    """When enhance is OFF, Main Video render pipeline behavior must be 100% preserved."""
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
        start_time=1.0,
        end_time=6.0,
        crop_x=50,
        crop_y=80,
        crop_size=900,
        enhance=False
    )

    assert captured_cmd is not None
    fc_index = captured_cmd.index("-filter_complex") + 1
    fc_str = captured_cmd[fc_index]

    # Must use DEFAULT video filters, NOT enhanced
    assert DEFAULT_VIDEO_FILTERS in fc_str
    assert "hqdn3d" not in fc_str
    # Output canvas and video positioning unchanged
    assert "color=c=white:s=1080x1920:d=5.0:r=30[base]" in fc_str
    assert "[base][v_proc]overlay=39:420:eof_action=pass[bg]" in fc_str
    # Saba Bollywood watermark unchanged
    assert "[with_cap][2:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]" in fc_str
    # Audio unchanged: mapped via 0:a? without audio filters
    assert "-map" in captured_cmd and "0:a?" in captured_cmd
    assert "[outa]" not in captured_cmd

def test_enhanced_mix_activates_audio_and_video_chain(monkeypatch):
    """When enhance is ON with Enhanced Mix, both video and audio chains are activated."""
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
        crop_x=0,
        crop_y=0,
        crop_size=1080,
        enhance=True,
        audio_mode="enhanced"
    )

    assert captured_cmd is not None
    fc_index = captured_cmd.index("-filter_complex") + 1
    fc_str = captured_cmd[fc_index]

    # Video: uses ENHANCED_VIDEO_FILTERS
    assert ENHANCED_VIDEO_FILTERS in fc_str
    assert "hqdn3d=1.5:1.5:3:3" in fc_str
    # Dimensions remain 1002x1002 and placement remains 39:420
    assert "scale=1002:1002" in fc_str
    assert "overlay=39:420" in fc_str
    # Watermark remains persistent inside square video
    assert "overlay=x=(W-w)/2:y=1374-h" in fc_str

    # Audio: mapped via [outa] with ENHANCED_AUDIO_FILTERS
    assert ENHANCED_AUDIO_FILTERS in fc_str
    assert "-map" in captured_cmd and "[outa]" in captured_cmd
    assert "-map" in captured_cmd and "[outv]" in captured_cmd

def test_voice_focus_activates_speech_priority_chain(monkeypatch):
    """When enhance is ON with Voice Focus, alternate speech-priority audio path is activated."""
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
        crop_x=0,
        crop_y=0,
        crop_size=1080,
        enhance=True,
        audio_mode="voice_focus"
    )

    assert captured_cmd is not None
    fc_index = captured_cmd.index("-filter_complex") + 1
    fc_str = captured_cmd[fc_index]

    # Video: uses ENHANCED_VIDEO_FILTERS
    assert ENHANCED_VIDEO_FILTERS in fc_str
    # Audio: uses VOICE_FOCUS_AUDIO_FILTERS
    assert VOICE_FOCUS_AUDIO_FILTERS in fc_str
    assert "highpass=f=120" in fc_str
    assert "equalizer=f=2800:t=q:w=1.2:g=4.0" in fc_str
    assert "-map" in captured_cmd and "[outa]" in captured_cmd

def test_original_mix_with_enhanced_video(monkeypatch):
    """When enhance is ON but audio_mode is Original Mix, video is enhanced while audio is untouched."""
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
        crop_x=0,
        crop_y=0,
        crop_size=1080,
        enhance=True,
        audio_mode="original"
    )

    assert captured_cmd is not None
    fc_index = captured_cmd.index("-filter_complex") + 1
    fc_str = captured_cmd[fc_index]

    # Video: enhanced
    assert ENHANCED_VIDEO_FILTERS in fc_str
    # Audio: original mix (0:a?), NO audio filter applied
    assert "0:a?" in captured_cmd
    assert "[outa]" not in captured_cmd
    assert "loudnorm" not in fc_str

def test_curiosity_mode_enhance_off_and_on(monkeypatch):
    """Curiosity mode should support both Enhance OFF and Enhance ON cleanly across hook and reveal."""
    captured_cmds = []
    def mock_run(cmd, *args, **kwargs):
        captured_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    # 1. Enhance OFF
    render_curiosity_video(
        source_video="dummy_in.mp4",
        output_video="dummy_out_off.mp4",
        hook_overlay="dummy_hook.png",
        reveal_overlay="dummy_reveal.png",
        start_time=0.0,
        cut_time=3.0,
        end_time=6.0,
        crop_x=0,
        crop_y=0,
        crop_size=900,
        enhance=False
    )

    fc_off = captured_cmds[0][captured_cmds[0].index("-filter_complex") + 1]
    assert DEFAULT_VIDEO_FILTERS in fc_off
    assert "hqdn3d" not in fc_off
    # Concatenated audio goes directly to [outa] without enhancement filter
    assert "concat=n=2:v=1:a=1[v_concat][outa]" in fc_off

    # 2. Enhance ON with Enhanced Mix
    render_curiosity_video(
        source_video="dummy_in.mp4",
        output_video="dummy_out_on.mp4",
        hook_overlay="dummy_hook.png",
        reveal_overlay="dummy_reveal.png",
        start_time=0.0,
        cut_time=3.0,
        end_time=6.0,
        crop_x=0,
        crop_y=0,
        crop_size=900,
        enhance=True,
        audio_mode="enhanced"
    )

    fc_on = captured_cmds[1][captured_cmds[1].index("-filter_complex") + 1]
    assert ENHANCED_VIDEO_FILTERS in fc_on
    # Concat produces [raw_a], which is then filtered into [outa]
    assert "concat=n=2:v=1:a=1[v_concat][raw_a]" in fc_on
    assert f"[raw_a]{ENHANCED_AUDIO_FILTERS}[outa]" in fc_on
    # Watermark remains persistent on [v_concat]
    assert "[v_concat][6:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]" in fc_on

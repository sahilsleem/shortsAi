"""
tests/test_frame_selector.py - Test suite for Saba Bollywood Cover Frame Selector

Tests:
1. Timestamp sampling (boundaries, curiosity exclusion, short videos, deterministic ordering).
2. Sharpness scoring (sharp synthetic image vs. blurred image).
3. Exposure scoring (balanced exposure vs. black/white extreme frames).
4. Composition scoring (high-activity central subject vs. blank image; does NOT assume skin=person).
5. Deduplication (near-identical frames suppressed in favor of diverse candidates).
6. Candidate ranking (higher aggregate scores rank higher).
7. Diagnostic report generation (5 JPEG files, JSON report with full score breakdowns).
8. CLI argument parsing (validation without runtime video dependencies).
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.frame_selector import (
    sample_candidate_timestamps,
    compute_sharpness_score,
    compute_composition_score,
    compute_exposure_score,
    compute_contrast_score,
    compute_saturation_score,
    compute_penalties,
    score_frame,
    compute_image_signature,
    compute_signature_distance,
    deduplicate_candidates,
    run_frame_selection_diagnostic,
    extract_candidate_frame,
    extract_full_res_frame
)

# ---------------------------------------------------------------------------
# 1. Timestamp Sampling Tests
# ---------------------------------------------------------------------------

def test_sampling_boundaries_and_exclusions():
    """Verify first 0.4s and final 0.5s are excluded in main mode."""
    start = 0.0
    end = 10.0
    interval = 0.6
    timestamps = sample_candidate_timestamps(start_time=start, end_time=end, mode="main", interval=interval)

    # First timestamp must be >= start + 0.4s
    assert timestamps[0] >= 0.4, f"First timestamp {timestamps[0]} is earlier than 0.4s buffer"
    # Last timestamp must be <= end - 0.5s
    assert timestamps[-1] <= 9.5, f"Last timestamp {timestamps[-1]} exceeds 9.5s buffer"

    # All timestamps must be strictly sorted and within range
    assert timestamps == sorted(timestamps)
    for ts in timestamps:
        assert 0.4 <= ts <= 9.5

def test_sampling_curiosity_mode_cut_exclusion():
    """Verify curiosity cut ± 0.25s is excluded from candidate timestamps."""
    start = 0.0
    end = 10.0
    cut_time = 5.0
    interval = 0.2
    timestamps = sample_candidate_timestamps(
        start_time=start,
        end_time=end,
        mode="curiosity",
        cut_time=cut_time,
        interval=interval
    )

    # Cut exclusion range is [4.75, 5.25]
    for ts in timestamps:
        assert not (4.75 <= ts <= 5.25), f"Timestamp {ts} was not excluded from curiosity cut window [4.75, 5.25]"

def test_sampling_short_video_safety():
    """Verify very short video clips do not crash or produce invalid ranges."""
    # Clip where (end - 0.5) <= (start + 0.4)
    short_timestamps = sample_candidate_timestamps(start_time=0.0, end_time=0.6, interval=0.5)
    assert len(short_timestamps) == 1
    assert short_timestamps[0] == 0.3  # Midpoint

    # Zero duration edge case
    zero_timestamps = sample_candidate_timestamps(start_time=2.0, end_time=2.0)
    assert len(zero_timestamps) == 1
    assert zero_timestamps[0] == 2.0

def test_sampling_deterministic_and_sorted():
    """Verify sample timestamps output is deterministic and strictly sorted."""
    ts1 = sample_candidate_timestamps(start_time=1.0, end_time=8.5, interval=0.5)
    ts2 = sample_candidate_timestamps(start_time=1.0, end_time=8.5, interval=0.5)
    assert ts1 == ts2
    assert ts1 == sorted(ts1)
    assert len(ts1) > 0

# ---------------------------------------------------------------------------
# 2. Metric Scoring Tests (Synthetic Images)
# ---------------------------------------------------------------------------

def test_sharpness_scoring_crisp_vs_blurred():
    """Verify crisp synthetic image scores substantially higher than blurred image."""
    # Create sharp high-contrast grid pattern
    sharp_img = Image.new("RGB", (360, 360), (255, 255, 255))
    draw = ImageDraw.Draw(sharp_img)
    for x in range(0, 360, 20):
        draw.line([(x, 0), (x, 360)], fill=(0, 0, 0), width=3)
    for y in range(0, 360, 20):
        draw.line([(0, y), (360, y)], fill=(0, 0, 0), width=3)

    # Intentionally blur the exact same image
    blurred_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=8))

    sharp_score = compute_sharpness_score(sharp_img)
    blurred_score = compute_sharpness_score(blurred_img)

    assert sharp_score > blurred_score, f"Expected sharp ({sharp_score}) > blurred ({blurred_score})"
    assert sharp_score >= 20.0
    assert blurred_score <= 10.0

def test_exposure_scoring_balanced_vs_extremes():
    """Verify properly exposed image beats nearly black and nearly white frames."""
    # Balanced mid-gray image (luminance ~ 125)
    balanced_img = Image.new("RGB", (360, 360), (125, 125, 125))
    # Pitch black frame (fade-to-black artifact)
    black_img = Image.new("RGB", (360, 360), (0, 0, 0))
    # Blown out white frame
    white_img = Image.new("RGB", (360, 360), (255, 255, 255))

    score_balanced = compute_exposure_score(balanced_img)
    score_black = compute_exposure_score(black_img)
    score_white = compute_exposure_score(white_img)

    assert score_balanced == 20.0, f"Balanced exposure should receive maximum 20.0, got {score_balanced}"
    assert score_black <= 2.0, f"Pitch black frame must be penalized, got {score_black}"
    assert score_white <= 2.0, f"Blown-out white frame must be penalized, got {score_white}"
    assert score_balanced > score_black + 15.0
    assert score_balanced > score_white + 15.0

def test_composition_scoring_subject_vs_blank():
    """
    Verify high-activity central subject region scores higher than completely blank background.
    IMPORTANT: This test verifies spatial visual structure and edge separation without assuming skin=person.
    """
    # 1. Blank background (no visual subject)
    blank_img = Image.new("RGB", (360, 360), (120, 120, 120))

    # 2. Image with framed subject structure in upper-center (clothing/geometric object, no skin colors)
    # Using blue and white patterns to strictly ensure skin chrominance is 0
    subject_img = Image.new("RGB", (360, 360), (80, 80, 80))
    draw = ImageDraw.Draw(subject_img)
    # Draw high-contrast subject in upper-center zone (X: 100-260, Y: 60-240)
    draw.rectangle([100, 60, 260, 240], fill=(240, 240, 255), outline=(0, 20, 80), width=4)
    draw.line([(100, 60), (260, 240)], fill=(0, 50, 150), width=3)
    draw.line([(100, 240), (260, 60)], fill=(0, 50, 150), width=3)

    blank_comp = compute_composition_score(blank_img)
    subject_comp = compute_composition_score(subject_img)

    assert subject_comp > blank_comp, f"Subject frame ({subject_comp}) should beat blank frame ({blank_comp})"
    assert blank_comp == 0.0
    assert subject_comp >= 12.0

def test_contrast_and_saturation_scoring():
    """Verify dynamic range and saturation scoring behavior."""
    # Flat contrast
    flat_img = Image.new("RGB", (360, 360), (100, 100, 100))
    # High contrast
    contrasty_img = Image.new("RGB", (360, 360), (0, 0, 0))
    draw = ImageDraw.Draw(contrasty_img)
    draw.rectangle([0, 0, 180, 360], fill=(255, 255, 255))

    assert compute_contrast_score(contrasty_img) > compute_contrast_score(flat_img)

    # Grayscale/monochrome vs natural color
    mono_img = Image.new("RGB", (360, 360), (120, 120, 120))
    color_img = Image.new("RGB", (360, 360), (180, 100, 60))

    assert compute_saturation_score(color_img) > compute_saturation_score(mono_img)

# ---------------------------------------------------------------------------
# 3. Deduplication & Ranking Tests
# ---------------------------------------------------------------------------

def test_deduplication_filters_near_identical_frames():
    """Verify that near-identical frames at close timestamps are deduplicated in top candidates."""
    # Candidate 1: Crisp high-scoring base frame at 8.0s
    img1 = Image.new("RGB", (360, 360), (120, 120, 120))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([80, 60, 280, 240], fill=(220, 180, 100), outline=(20, 20, 20), width=4)
    sig1 = compute_image_signature(img1)

    # Candidate 2: Virtually identical frame at 8.6s (minor 1px change)
    img2 = img1.copy()
    draw2 = ImageDraw.Draw(img2)
    draw2.point([1, 1], fill=(255, 255, 255))
    sig2 = compute_image_signature(img2)

    # Candidate 3: Virtually identical frame at 9.2s
    img3 = img1.copy()
    sig3 = compute_image_signature(img3)

    # Candidate 4: Visually completely distinct frame at 3.4s (different composition)
    img4 = Image.new("RGB", (360, 360), (40, 60, 120))
    draw4 = ImageDraw.Draw(img4)
    draw4.ellipse([50, 50, 150, 150], fill=(255, 255, 255))
    sig4 = compute_image_signature(img4)

    # Candidate 5: Another distinct frame at 12.0s
    img5 = Image.new("RGB", (360, 360), (200, 200, 200))
    draw5 = ImageDraw.Draw(img5)
    draw5.rectangle([200, 200, 350, 350], fill=(0, 0, 0))
    sig5 = compute_image_signature(img5)

    candidates = [
        {"timestamp": 8.0, "score_data": {"total_score": 88.0}, "signature": sig1, "image": img1},
        {"timestamp": 8.6, "score_data": {"total_score": 87.5}, "signature": sig2, "image": img2},
        {"timestamp": 9.2, "score_data": {"total_score": 87.0}, "signature": sig3, "image": img3},
        {"timestamp": 3.4, "score_data": {"total_score": 75.0}, "signature": sig4, "image": img4},
        {"timestamp": 12.0, "score_data": {"total_score": 72.0}, "signature": sig5, "image": img5},
    ]

    deduped = deduplicate_candidates(candidates, max_results=3)

    # Verify that the top 3 are NOT {8.0s, 8.6s, 9.2s}
    selected_timestamps = [c["timestamp"] for c in deduped]
    assert 8.0 in selected_timestamps
    assert 3.4 in selected_timestamps
    assert 12.0 in selected_timestamps
    # 8.6s and 9.2s should have been suppressed as near-duplicates
    assert not (8.6 in selected_timestamps and 9.2 in selected_timestamps)

def test_candidate_ranking_order():
    """Verify higher aggregate scores rank higher."""
    cands = [
        {"timestamp": 2.0, "score_data": {"total_score": 65.0}, "signature": [0] * 256},
        {"timestamp": 4.0, "score_data": {"total_score": 92.0}, "signature": [100] * 256},
        {"timestamp": 6.0, "score_data": {"total_score": 81.0}, "signature": [200] * 256},
    ]
    ranked = deduplicate_candidates(cands, max_results=3)
    assert ranked[0]["timestamp"] == 4.0
    assert ranked[0]["rank"] == 1
    assert ranked[1]["timestamp"] == 6.0
    assert ranked[1]["rank"] == 2
    assert ranked[2]["timestamp"] == 2.0
    assert ranked[2]["rank"] == 3

# ---------------------------------------------------------------------------
# 4. Diagnostic Pipeline & Report Generation
# ---------------------------------------------------------------------------

def test_diagnostic_pipeline_output(monkeypatch):
    """
    Mock frame extraction and verify run_frame_selection_diagnostic outputs:
    - candidate_01.jpg through candidate_05.jpg
    - diagnostic_report.json with proper schema
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Mock extract_candidate_frame to return synthetic Pillow images
        def mock_extract(source_video, timestamp, **kwargs):
            img = Image.new("RGB", (360, 360), (int(timestamp * 25) % 255, 120, 140))
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 300, 300], fill=(200, 200, 200), outline=(0, 0, 0), width=2)
            return img

        monkeypatch.setattr("src.frame_selector.extract_candidate_frame", mock_extract)

        report = run_frame_selection_diagnostic(
            source_video="dummy_test_video.mp4",
            start_time=0.0,
            end_time=6.0,
            interval=0.6,
            output_dir=temp_dir
        )

        # 1. Verify report structure
        assert report["source_path"] is not None
        assert report["duration"] == 6.0
        assert report["sampled_timestamps_count"] > 0
        assert report["selected_candidate"] is not None
        assert report["selected_candidate"]["rank"] == 1
        assert len(report["top_candidates"]) <= 5

        # 2. Verify score components in report
        scores = report["selected_candidate"]["scores"]
        assert "sharpness" in scores
        assert "composition" in scores
        assert "exposure" in scores
        assert "contrast" in scores
        assert "saturation" in scores
        assert "penalty" in scores

        # 3. Verify output files on disk
        report_file = os.path.join(temp_dir, "diagnostic_report.json")
        assert os.path.exists(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["selected_candidate"]["total_score"] > 0

        # Verify candidate image files exist
        for i in range(1, len(report["top_candidates"]) + 1):
            cand_img_path = os.path.join(temp_dir, f"candidate_{i:02d}.jpg")
            assert os.path.exists(cand_img_path)
            with Image.open(cand_img_path) as im:
                assert im.size == (360, 360)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# 5. CLI Argument Parsing Tests
# ---------------------------------------------------------------------------

def test_cli_argument_parsing(monkeypatch):
    """Verify CLI parses arguments correctly and enforces choices."""
    from src.frame_selector import main

    test_args = [
        "frame_selector.py",
        "--input", "dummy_video.mp4",
        "--start", "1.5",
        "--end", "8.0",
        "--crop-x", "40",
        "--crop-y", "100",
        "--crop-size", "960",
        "--mode", "curiosity",
        "--cut-time", "4.5",
        "--interval", "0.5",
        "--output-dir", "test_candidates_dir"
    ]

    captured_kwargs = None
    def mock_run_diag(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return {}

    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr("src.frame_selector.run_frame_selection_diagnostic", mock_run_diag)

    main()

    assert captured_kwargs is not None
    assert captured_kwargs["start_time"] == 1.5
    assert captured_kwargs["end_time"] == 8.0
    assert captured_kwargs["crop_x"] == 40
    assert captured_kwargs["crop_y"] == 100
    assert captured_kwargs["crop_size"] == 960
    assert captured_kwargs["mode"] == "curiosity"
    assert captured_kwargs["cut_time"] == 4.5
    assert captured_kwargs["interval"] == 0.5
    assert captured_kwargs["output_dir"] == "test_candidates_dir"

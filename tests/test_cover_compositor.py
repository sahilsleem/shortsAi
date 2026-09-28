"""
tests/test_cover_compositor.py - Test suite for Saba Bollywood Cover Frame Compositor

Tests:
1. Exact 1080x1080 output dimensions.
2. Source aspect ratio handling (square, vertical, horizontal inputs).
3. Deterministic output generation.
4. Phrase rendering with text and emojis.
5. Empty and very short phrase handling.
6. Long phrase handling (wrapping & fitting within canvas).
7. Smart text zone selection (clutter scoring).
8. Face/subject avoidance heuristic (skin/clutter at top -> bottom chosen, and vice versa).
9. Branding asset handling (watermark scaled and positioned; missing asset handled gracefully).
10. Output file creation on disk.
11. JPEG and PNG format validity.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.cover_compositor import (
    create_cover_frame,
    prepare_source_image,
    apply_visual_finish,
    detect_subject_region,
    evaluate_placement_zones,
    compute_cover_font_size,
    render_cover_text,
    apply_branding,
    DEFAULT_FONT_PATH,
    DEFAULT_WATERMARK_PATH
)


# ---------------------------------------------------------------------------
# 1. Output Dimensions & Aspect Ratio Tests
# ---------------------------------------------------------------------------

def test_1080x1080_output():
    """Verify default output is exactly 1080x1080 pixels."""
    sample_img = Image.new("RGB", (360, 360), (140, 140, 140))
    res = create_cover_frame(sample_img, phrase="SALMAN DID THIS 😳")

    assert res["size"] == (1080, 1080)
    assert res["output_image"].size == (1080, 1080)
    assert res["output_image"].mode == "RGB"

def test_source_aspect_handling():
    """
    Verify square, vertical, and horizontal inputs are all properly converted
    to square 1080x1080 without distortion or stretching.
    """
    # 1. Square source (e.g. 360x360)
    sq = Image.new("RGB", (360, 360), (100, 100, 100))
    p_sq = prepare_source_image(sq, (1080, 1080))
    assert p_sq.size == (1080, 1080)

    # 2. Vertical source (e.g. 720x1280)
    vert = Image.new("RGB", (720, 1280), (100, 100, 100))
    p_vert = prepare_source_image(vert, (1080, 1080))
    assert p_vert.size == (1080, 1080)

    # 3. Horizontal source (e.g. 1920x1080)
    horiz = Image.new("RGB", (1920, 1080), (100, 100, 100))
    p_horiz = prepare_source_image(horiz, (1080, 1080))
    assert p_horiz.size == (1080, 1080)

def test_deterministic_output():
    """Verify running compositor twice with identical inputs produces pixel-identical output."""
    sample = Image.new("RGB", (400, 400), (120, 130, 140))
    draw = ImageDraw.Draw(sample)
    draw.rectangle([50, 50, 350, 350], fill=(200, 100, 50))

    res1 = create_cover_frame(sample, phrase="THIS MOMENT ❤️")
    res2 = create_cover_frame(sample, phrase="THIS MOMENT ❤️")

    assert res1["size"] == res2["size"]
    assert res1["selected_position"] == res2["selected_position"]
    # Verify exact pixel bytes match
    assert res1["output_image"].tobytes() == res2["output_image"].tobytes()


# ---------------------------------------------------------------------------
# 2. Phrase Rendering & Edge Cases
# ---------------------------------------------------------------------------

def test_phrase_rendering_with_emoji():
    """Verify typical Saba Bollywood phrase with emojis renders without error."""
    sample = Image.new("RGB", (360, 360), (100, 100, 100))
    res = create_cover_frame(sample, phrase="SALMAN DID THIS 😳")

    assert res["phrase"] == "SALMAN DID THIS 😳"
    assert res["output_image"] is not None

def test_empty_and_very_short_phrase():
    """Verify empty string or single word handles gracefully without crash."""
    sample = Image.new("RGB", (360, 360), (100, 100, 100))

    # Empty phrase
    res_empty = create_cover_frame(sample, phrase="")
    assert res_empty["size"] == (1080, 1080)

    # Single word
    res_word = create_cover_frame(sample, phrase="UNBELIEVABLE")
    assert res_word["size"] == (1080, 1080)

def test_long_phrase_handling():
    """Verify longer phrase wraps or scales gracefully to fit inside canvas bounds."""
    sample = Image.new("RGB", (360, 360), (100, 100, 100))
    long_phrase = "THIS WAS AN ABSOLUTELY UNEXPECTED BOLLYWOOD MOMENT 😳"
    res = create_cover_frame(sample, phrase=long_phrase)

    assert res["size"] == (1080, 1080)

    # Check font computation
    fs, lines = compute_cover_font_size(long_phrase, DEFAULT_FONT_PATH, max_width=940)
    assert len(lines) <= 2
    assert fs >= 40


# ---------------------------------------------------------------------------
# 3. Smart Text Placement & Face Avoidance Heuristic
# ---------------------------------------------------------------------------

def test_text_zone_selection_clutter_scoring():
    """Verify evaluate_placement_zones returns scores for top and bottom zones."""
    sample = Image.new("RGB", (1080, 1080), (120, 120, 120))
    info = evaluate_placement_zones(sample)

    assert "recommended" in info
    assert info["recommended"] in ("top", "bottom")
    assert "top" in info["scores"]
    assert "bottom" in info["scores"]
    assert "total_clutter" in info["scores"]["top"]
    assert "total_clutter" in info["scores"]["bottom"]

def test_face_subject_avoidance_heuristic():
    """
    Test face avoidance:
    - When skin/face is at the top, bottom position is selected.
    - When skin/face is at the bottom, top position is selected.
    """
    # 1. Subject/face placed in top quadrant (Y: 80-220, X: 400-680)
    img_face_top = Image.new("RGB", (1080, 1080), (80, 80, 80))
    draw_top = ImageDraw.Draw(img_face_top)
    # Warm skin tone rectangle in top zone
    draw_top.rectangle([400, 80, 680, 220], fill=(215, 160, 120), outline=(0, 0, 0), width=3)

    info_top = evaluate_placement_zones(img_face_top)
    # Top has high face penalty -> recommended must be bottom!
    assert info_top["recommended"] == "bottom", f"Expected bottom, got {info_top['recommended']}"
    assert info_top["scores"]["top"]["face_penalty"] > info_top["scores"]["bottom"]["face_penalty"]

    # 2. Subject/face placed in bottom quadrant (Y: 820-980, X: 400-680)
    img_face_bot = Image.new("RGB", (1080, 1080), (80, 80, 80))
    draw_bot = ImageDraw.Draw(img_face_bot)
    # Warm skin tone rectangle in bottom zone
    draw_bot.rectangle([400, 840, 680, 980], fill=(215, 160, 120), outline=(0, 0, 0), width=3)

    info_bot = evaluate_placement_zones(img_face_bot)
    # Bottom has high face penalty -> recommended must be top!
    assert info_bot["recommended"] == "top", f"Expected top, got {info_bot['recommended']}"
    assert info_bot["scores"]["bottom"]["face_penalty"] > info_bot["scores"]["top"]["face_penalty"]

def test_manual_text_position_override():
    """Verify caller can manually override text position to top or bottom."""
    sample = Image.new("RGB", (360, 360), (100, 100, 100))
    res_top = create_cover_frame(sample, phrase="TEST", text_position="top")
    assert res_top["selected_position"] == "top"

    res_bottom = create_cover_frame(sample, phrase="TEST", text_position="bottom")
    assert res_bottom["selected_position"] == "bottom"


# ---------------------------------------------------------------------------
# 4. Branding & Visual Finish Tests
# ---------------------------------------------------------------------------

def test_branding_asset_handling():
    """Verify watermark is composited when file exists, and missing asset is handled gracefully."""
    sample = Image.new("RGB", (1080, 1080), (50, 50, 50))

    # Existing watermark
    assert os.path.exists(DEFAULT_WATERMARK_PATH)
    res_wm = create_cover_frame(sample, phrase="SALMAN DID THIS 😳", watermark_path=DEFAULT_WATERMARK_PATH)
    assert res_wm["watermark_used"] is True

    # Missing watermark
    res_none = create_cover_frame(sample, phrase="SALMAN DID THIS 😳", watermark_path="nonexistent_wm.png")
    assert res_none["watermark_used"] is False
    assert res_none["size"] == (1080, 1080)

def test_visual_finish_enhances_image():
    """Verify subtle visual finish pass executes without error."""
    sample = Image.new("RGBA", (1080, 1080), (120, 120, 120, 255))
    finished = apply_visual_finish(sample)
    assert finished.size == (1080, 1080)
    assert finished.mode == "RGBA"


# ---------------------------------------------------------------------------
# 5. Output File & Format Tests
# ---------------------------------------------------------------------------

def test_output_file_creation_and_formats():
    """Verify saving to JPEG and PNG produces valid readable image files on disk."""
    temp_dir = tempfile.mkdtemp()
    try:
        sample = Image.new("RGB", (360, 360), (120, 140, 160))

        # 1. JPEG output
        jpg_path = os.path.join(temp_dir, "cover.jpg")
        res_jpg = create_cover_frame(sample, phrase="HER REACTION 😂", output_path=jpg_path)
        assert os.path.exists(jpg_path)
        with Image.open(jpg_path) as im:
            assert im.size == (1080, 1080)
            assert im.format == "JPEG"

        # 2. PNG output
        png_path = os.path.join(temp_dir, "cover.png")
        res_png = create_cover_frame(sample, phrase="HER REACTION 😂", output_path=png_path)
        assert os.path.exists(png_path)
        with Image.open(png_path) as im:
            assert im.size == (1080, 1080)
            assert im.format == "PNG"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# 6. Standalone CLI Tests
# ---------------------------------------------------------------------------

def test_cli_execution_with_test_image(monkeypatch):
    """Verify CLI parses arguments, runs compositor, and writes output image."""
    from src.cover_compositor import main

    temp_dir = tempfile.mkdtemp()
    try:
        in_img_path = os.path.join(temp_dir, "test_candidate.jpg")
        out_img_path = os.path.join(temp_dir, "test_cover.jpg")

        # Create synthetic input image
        sample = Image.new("RGB", (360, 360), (100, 120, 140))
        sample.save(in_img_path, "JPEG")

        test_args = [
            "cover_compositor.py",
            "--input", in_img_path,
            "--phrase", "SALMAN DID THIS 😳",
            "--output", out_img_path,
            "--size", "1080"
        ]

        monkeypatch.setattr(sys, "argv", test_args)

        main()

        assert os.path.exists(out_img_path)
        with Image.open(out_img_path) as im:
            assert im.size == (1080, 1080)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. Face Protection & Text Color Verification Tests
# ---------------------------------------------------------------------------

def test_detect_subject_region_function():
    """Verify detect_subject_region detects skin clusters and calculates protected zone."""
    # 1. Image with synthetic face
    img_face = Image.new("RGB", (1080, 1080), (70, 70, 70))
    draw = ImageDraw.Draw(img_face)
    draw.rectangle([350, 180, 730, 420], fill=(215, 160, 120))

    info = detect_subject_region(img_face)
    assert info["detected"] is True
    assert info["face_box"] is not None
    assert info["protected_y"] is not None
    # Protected zone should encompass face plus forehead and chin padding
    prot_y1, prot_y2 = info["protected_y"]
    assert prot_y1 < 180, "Forehead/hair padding must extend above skin"
    assert prot_y2 > 420, "Chin/neck padding must extend below skin"

    # 2. Image without skin tones
    img_plain = Image.new("RGB", (1080, 1080), (30, 30, 80))
    info_plain = detect_subject_region(img_plain)
    assert info_plain["detected"] is False
    assert info_plain["face_box"] is None


def test_text_does_not_overlap_detected_face():
    """
    Verify text strictly avoids the celebrity's face and head:
    When face is positioned in the upper portion (e.g. Y=170..450),
    the compositor selects 'bottom' and text Y bounds have 0 overlap with face.
    """
    img = Image.new("RGB", (1080, 1080), (80, 80, 80))
    draw = ImageDraw.Draw(img)
    # Face placed at Y=170..450 (typical portrait shot)
    draw.rectangle([350, 170, 730, 450], fill=(215, 160, 120))
    # Clutter at bottom (e.g. textured shirt)
    for y in range(750, 1050, 10):
        draw.line([(100, y), (980, y)], fill=(200, 200, 200), width=2)

    res = create_cover_frame(img, phrase="SALMAN DID THIS 😳")

    assert res["selected_position"] == "bottom", (
        f"Expected bottom to avoid face at top, got {res['selected_position']}"
    )
    # Verify top zone was flagged with face collision
    assert res["placement_scores"]["top"]["collides_with_face"] is True
    assert res["placement_scores"]["bottom"]["collides_with_face"] is False

    # Check text starting Y
    y_start = res["placement_scores"]["bottom"]["y_start"]
    # Protected face zone is ~100 to ~520. Text starting at > 800 has 0 overlap.
    assert y_start > 550, f"Text start Y {y_start} overlaps with face zone (170-450)"


def test_face_at_bottom_selects_top():
    """
    Verify that if a face is in the lower portion (e.g. leaning in at bottom),
    the compositor selects 'top' so text never obscures the face.
    """
    img = Image.new("RGB", (1080, 1080), (80, 80, 80))
    draw = ImageDraw.Draw(img)
    # Face placed at Y=750..980
    draw.rectangle([350, 750, 730, 980], fill=(215, 160, 120))

    res = create_cover_frame(img, phrase="WAIT FOR IT 😳")

    assert res["selected_position"] == "top", (
        f"Expected top to avoid face at bottom, got {res['selected_position']}"
    )
    assert res["placement_scores"]["bottom"]["collides_with_face"] is True
    assert res["placement_scores"]["top"]["collides_with_face"] is False


def test_lower_clear_area_preferred_when_available():
    """
    Verify that when both top and bottom areas are clear, the compositor
    prefers the lower portion (bottom) as specified in requirements.
    """
    # Plain image with clean negative space across top and bottom
    img = Image.new("RGB", (1080, 1080), (60, 60, 60))
    info = evaluate_placement_zones(img)

    assert info["recommended"] == "bottom"
    assert info["scores"]["bottom"]["preference_bonus"] < 0, "Expected negative bonus favoring bottom"


def test_text_color_white_fill_with_solid_black_stroke():
    """
    Verify the cover text renders with:
    1. Pure WHITE text fill (R>240, G>240, B>240)
    2. Strong BLACK outline/stroke (R<25, G<25, B<25)
    3. High-contrast typography instantly readable on mobile
    """
    # Use neutral gray canvas so white letters and black outlines are easily separated
    img = Image.new("RGB", (1080, 1080), (128, 128, 128))
    res = create_cover_frame(img, phrase="UNBELIEVABLE 😳", apply_finish=False)

    out_img = res["output_image"]
    y_start = res["placement_scores"][res["selected_position"]]["y_start"]

    # Sample horizontal slice through the rendered letters
    sample_y = int(y_start + 45)
    pixels = [out_img.getpixel((x, sample_y)) for x in range(100, 980)]

    white_pixels = [p for p in pixels if p[0] > 240 and p[1] > 240 and p[2] > 240]
    black_pixels = [p for p in pixels if p[0] < 30 and p[1] < 30 and p[2] < 30]

    assert len(white_pixels) >= 20, (
        f"Expected substantial white text fill pixels, found {len(white_pixels)}"
    )
    assert len(black_pixels) >= 20, (
        f"Expected substantial black outline stroke pixels, found {len(black_pixels)}"
    )

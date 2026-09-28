"""
src/cover_compositor.py - Saba Bollywood Cover Frame Compositor

Phase 3 Compositor:
Transforms a selected clean square video frame into a polished 1080x1080 Saba Bollywood
cover frame / thumbnail image.
- Preserves clean square footage without distortion or artificial letterboxing.
- Applies subtle visual polish (contrast, mild saturation, gentle sharpening, soft vignette).
- Smart text placement (evaluates visual complexity, luminance, and face/subject presence
  to automatically place text where it never covers the celebrity's face).
- Elegant Saba Bollywood branding using existing assets, scaled proportionally.
- High-quality, readable typography with subtle drop shadow and natural background gradient.
- Standalone CLI for local diagnostic testing.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Union, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat, ImageEnhance

try:
    import emoji
except ImportError:
    emoji = None

try:
    from pilmoji import Pilmoji
    from pilmoji.source import AppleEmojiSource
except ImportError:
    Pilmoji = None
    AppleEmojiSource = None

DEFAULT_FONT_PATH = str(Path(__file__).resolve().parent.parent / "fonts" / "Calistoga-Regular.ttf")
DEFAULT_WATERMARK_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "saba_bollywood_watermark.png")


# ---------------------------------------------------------------------------
# 1. Source Image Preparation & Scaling
# ---------------------------------------------------------------------------

def prepare_source_image(img: Image.Image, target_size: Tuple[int, int] = (1080, 1080)) -> Image.Image:
    """
    Scale the source image to target_size (default 1080x1080) maintaining 1:1 aspect ratio.
    If the source is already square, performs high-quality Lanczos scaling.
    If non-square, center-crops the largest square first to prevent stretching or letterboxing.
    """
    w, h = img.size
    target_w, target_h = target_size

    # Ensure square crop if input has non-square aspect ratio
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return img.convert("RGBA")


# ---------------------------------------------------------------------------
# 2. Subtle Visual Finish Pass
# ---------------------------------------------------------------------------

def create_radial_vignette(width: int, height: int, max_alpha: int = 38) -> Image.Image:
    """
    Generates a soft, natural radial vignette that gently darkens only the outermost corners.
    Keeps center 100% transparent and natural.
    """
    grid_size = 54
    center = grid_size / 2.0
    pixels = []
    for y in range(grid_size):
        for x in range(grid_size):
            dist = math.sqrt((x - center) ** 2 + (y - center) ** 2) / center
            # Smooth falloff starting at 70% radius from center
            if dist > 0.70:
                t = (dist - 0.70) / 0.45
                alpha = int(min(max_alpha, max(0, t * max_alpha)))
            else:
                alpha = 0
            pixels.append((0, 0, 0, alpha))

    small = Image.new("RGBA", (grid_size, grid_size))
    small.putdata(pixels)
    return small.resize((width, height), Image.Resampling.BILINEAR)

def apply_visual_finish(img: Image.Image) -> Image.Image:
    """
    Applies a restrained visual polish to elevate the frame into a poster/cover image:
    - Subtle contrast (+5%)
    - Mild saturation (+6%)
    - Subtle sharpening (+15%)
    - Soft radial corner vignette
    Preserves natural skin tones and original camera lighting.
    """
    rgb = img.convert("RGB")

    # 1. Subtle contrast
    enhancer_contrast = ImageEnhance.Contrast(rgb)
    rgb = enhancer_contrast.enhance(1.05)

    # 2. Mild saturation
    enhancer_color = ImageEnhance.Color(rgb)
    rgb = enhancer_color.enhance(1.06)

    # 3. Subtle sharpening
    enhancer_sharp = ImageEnhance.Sharpness(rgb)
    rgb = enhancer_sharp.enhance(1.15)

    result = rgb.convert("RGBA")

    # 4. Soft radial vignette
    vignette = create_radial_vignette(img.width, img.height, max_alpha=35)
    result.alpha_composite(vignette)

    return result


# ---------------------------------------------------------------------------
# 3. Smart Text Placement & Face Avoidance Heuristic
# ---------------------------------------------------------------------------

def detect_subject_region(img: Image.Image) -> dict:
    """
    Detects human subject / celebrity face region using color-space chrominance
    analysis (YCbCr) in pure Python with zero external runtime dependencies.
    Identifies the primary vertical span of the face and pads it to protect
    the forehead/hair and chin/neck as a strict no-text zone.
    """
    w, h = img.size
    small_w, small_h = 108, 108
    small = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    ycbcr = small.convert("YCbCr")
    y_ch, cb_ch, cr_ch = ycbcr.split()
    y_bytes = y_ch.tobytes()
    cb_bytes = cb_ch.tobytes()
    cr_bytes = cr_ch.tobytes()

    scale_y = h / float(small_h)
    scale_x = w / float(small_w)

    skin_mask = []
    row_counts = [0] * small_h
    for y in range(small_h):
        for x in range(small_w):
            idx = y * small_w + x
            lum = y_bytes[idx]
            cb = cb_bytes[idx]
            cr = cr_bytes[idx]
            # Human skin chrominance cluster:
            # Cr > Cb (hemoglobin red chrominance) reliably separates human skin from
            # warm wooden backgrounds, beige walls, and ambient indoor lighting.
            if (75 <= cb <= 135) and (135 <= cr <= 180) and ((cr - cb) >= 10) and (lum >= 45):
                row_counts[y] += 1
                skin_mask.append((x, y))

    # A face row must have at least ~8 skin pixels in a 108px row
    face_rows = [y for y, count in enumerate(row_counts) if count >= 8]
    if not face_rows:
        return {
            "detected": False,
            "face_box": None,
            "protected_y": None,
            "skin_ratio": 0.0
        }

    # Find contiguous segments of rows where skin was detected
    segments = []
    current_seg = [face_rows[0]]
    for r in face_rows[1:]:
        if r <= current_seg[-1] + 2:  # Allow small gaps (e.g. eyes/glasses/mustache)
            current_seg.append(r)
        else:
            segments.append(current_seg)
            current_seg = [r]
    segments.append(current_seg)

    # Pick the largest/densest segment as the main face
    main_seg = max(segments, key=lambda seg: sum(row_counts[r] for r in seg))
    min_row, max_row = main_seg[0], main_seg[-1]

    face_xs = [x for (x, y) in skin_mask if min_row <= y <= max_row]
    min_col = min(face_xs) if face_xs else 0
    max_col = max(face_xs) if face_xs else small_w - 1

    y_min = int(min_row * scale_y)
    y_max = int((max_row + 1) * scale_y)
    x_min = int(min_col * scale_x)
    x_max = int((max_col + 1) * scale_x)

    # Protective padding:
    # Top padding protects forehead and hair (7% of image height)
    # Bottom padding protects chin, jaw, and collar (7% of image height)
    prot_y_min = max(0, y_min - int(h * 0.07))
    prot_y_max = min(h, y_max + int(h * 0.07))

    return {
        "detected": True,
        "face_box": (x_min, y_min, x_max, y_max),
        "protected_y": (prot_y_min, prot_y_max),
        "skin_ratio": len(skin_mask) / float(small_w * small_h)
    }


def evaluate_placement_zones(
    img: Image.Image,
    text_height: int = 150,
    force_position: Optional[str] = None
) -> dict:
    """
    Evaluates candidate text zones ('top' vs 'bottom') on the image.
    1. Detects face/subject region and treats it as a strictly protected no-text zone.
    2. Calculates face/subject collision, edge clutter, and luminance variance.
    3. Prevents text from ever covering the celebrity's face or head (massive collision penalty).
    4. Prefers the lower portion ('bottom') when clear and available.
    5. If bottom is obstructed by the subject or blocked, selects the cleanest available area (top).
    """
    w, h = img.size

    # Detect human subject / celebrity face region
    subject_info = detect_subject_region(img)

    # Candidate text placement boxes
    # Top text zone: Y: ~8% to 8% + text_height
    # Bottom text zone: Y: ~92% - text_height to 92%
    top_y_start = int(h * 0.08)
    top_y_end = int(top_y_start + text_height)
    top_box = (int(w * 0.05), top_y_start, int(w * 0.95), top_y_end)

    bot_y_end = int(h * 0.92)
    bot_y_start = int(bot_y_end - text_height)
    bot_box = (int(w * 0.05), bot_y_start, int(w * 0.95), bot_y_end)

    zones = {
        "top": top_box,
        "bottom": bot_box
    }

    scores = {}
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    ycbcr = img.convert("YCbCr")

    for name, box in zones.items():
        z_x1, z_y1, z_x2, z_y2 = box

        # 1. Subject / Face Collision Protection
        collides = False
        face_overlap_ratio = 0.0
        face_penalty = 0.0

        if subject_info["detected"]:
            prot_y1, prot_y2 = subject_info["protected_y"]
            overlap_y1 = max(z_y1, prot_y1)
            overlap_y2 = min(z_y2, prot_y2)
            overlap_h = max(0, overlap_y2 - overlap_y1)

            if overlap_h > 0:
                collides = True
                face_overlap_ratio = overlap_h / float(max(1, text_height))
                # Severe disqualifying penalty: text must NEVER cover the face
                face_penalty = 10000.0 + (face_overlap_ratio * 5000.0)

        # In addition, measure skin ratio within the candidate zone
        zone_ycbcr = ycbcr.crop(box)
        zone_y, zone_cb, zone_cr = zone_ycbcr.split()
        cb_bytes = zone_cb.tobytes()
        cr_bytes = zone_cr.tobytes()
        lum_bytes = zone_y.tobytes()
        total_pixels = max(1, len(cb_bytes))
        skin_count = sum(
            1 for b, r, l in zip(cb_bytes, cr_bytes, lum_bytes)
            if 75 <= b <= 135 and 135 <= r <= 180 and (r - b) >= 10 and l >= 45
        )
        skin_ratio = skin_count / total_pixels

        # If not already penalized for direct collision, skin presence adds mild penalty
        if not collides:
            face_penalty = skin_ratio * 70.0

        # 2. Edge / Visual Clutter in zone
        zone_edges = edges.crop(box)
        edge_mean = ImageStat.Stat(zone_edges).mean[0]

        # 3. Luminance Variance in zone
        zone_gray = gray.crop(box)
        lum_stddev = ImageStat.Stat(zone_gray).stddev[0]

        clutter_penalty = edge_mean * 1.0
        variance_penalty = lum_stddev * 0.25

        # 4. Position preference: strongly prefer bottom when clear of face
        pref_bonus = -25.0 if name == "bottom" else 0.0

        total_zone_clutter = face_penalty + clutter_penalty + variance_penalty + pref_bonus

        scores[name] = {
            "face_penalty": round(face_penalty, 2),
            "edge_clutter": round(clutter_penalty, 2),
            "variance": round(variance_penalty, 2),
            "preference_bonus": round(pref_bonus, 2),
            "total_clutter": round(total_zone_clutter, 2),
            "collides_with_face": collides,
            "skin_ratio": round(skin_ratio, 3),
            "y_start": z_y1,
            "y_end": z_y2
        }

    # Pick zone with lower clutter (cleaner space, strictly avoiding face)
    if force_position in ("top", "bottom"):
        recommended = force_position
    else:
        top_score = scores["top"]["total_clutter"]
        bottom_score = scores["bottom"]["total_clutter"]
        if bottom_score <= top_score:
            recommended = "bottom"
        else:
            recommended = "top"

    return {
        "recommended": recommended,
        "scores": scores,
        "subject_region": subject_info
    }


# ---------------------------------------------------------------------------
# 4. Typography & Text Rendering
# ---------------------------------------------------------------------------

def get_text_width(word: str, font: ImageFont.ImageFont, pilmoji_context, draw: ImageDraw.ImageDraw) -> float:
    """Measure word or phrase width with Pilmoji fallback."""
    if pilmoji_context:
        return pilmoji_context.getsize(word, font=font)[0]
    return draw.textlength(word, font=font)

def get_line_height(font: ImageFont.ImageFont, pilmoji_context, draw: ImageDraw.ImageDraw) -> float:
    """Measure single line typographic height."""
    if pilmoji_context:
        return pilmoji_context.getsize("AydY~.", font=font)[1]
    bbox = draw.textbbox((0, 0), "AydY~.", font=font)
    return bbox[3] - bbox[1]

def wrap_cover_phrase(
    phrase: str,
    font: ImageFont.ImageFont,
    max_width: int,
    pilmoji_context,
    draw: ImageDraw.ImageDraw
) -> list[str]:
    """
    Wraps short cover phrases into 1 or 2 visually balanced lines.
    """
    phrase = phrase.strip()
    if not phrase:
        return []

    # Check if full phrase fits on 1 line
    if get_text_width(phrase, font, pilmoji_context, draw) <= max_width:
        return [phrase]

    words = phrase.split()
    if len(words) <= 1:
        return [phrase]

    # Find the cleanest 2-line split point that balances line widths
    best_split = 1
    min_diff = float("inf")
    for i in range(1, len(words)):
        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])
        w1 = get_text_width(line1, font, pilmoji_context, draw)
        w2 = get_text_width(line2, font, pilmoji_context, draw)
        if max(w1, w2) <= max_width:
            diff = abs(w1 - w2)
            if diff < min_diff:
                min_diff = diff
                best_split = i

    line1 = " ".join(words[:best_split])
    line2 = " ".join(words[best_split:])
    return [line1, line2]

def compute_cover_font_size(
    phrase: str,
    font_path: str,
    max_width: int = 940,
    max_font_size: int = 74,
    min_font_size: int = 46
) -> Tuple[int, list[str]]:
    """
    Computes optimal bold font size for short cover phrase.
    Favors 1 line if possible, otherwise clean 2-line composition.
    """
    dummy_img = Image.new("RGBA", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    pilmoji_ctx = Pilmoji(dummy_img, source=AppleEmojiSource) if Pilmoji else None

    # First attempt: find single line fit down to 54px
    for fs in range(max_font_size, 53, -2):
        try:
            f = ImageFont.truetype(font_path, fs)
        except IOError:
            f = ImageFont.load_default()
        if get_text_width(phrase, f, pilmoji_ctx, dummy_draw) <= max_width:
            return fs, [phrase]

    # Second attempt: 2-line fit
    for fs in range(max_font_size - 4, min_font_size - 1, -2):
        try:
            f = ImageFont.truetype(font_path, fs)
        except IOError:
            f = ImageFont.load_default()
        lines = wrap_cover_phrase(phrase, f, max_width, pilmoji_ctx, dummy_draw)
        if len(lines) <= 2:
            all_fit = all(get_text_width(l, f, pilmoji_ctx, dummy_draw) <= max_width for l in lines)
            if all_fit:
                return fs, lines

    # Fallback to min_font_size
    try:
        f = ImageFont.truetype(font_path, min_font_size)
    except IOError:
        f = ImageFont.load_default()
    lines = wrap_cover_phrase(phrase, f, max_width, pilmoji_ctx, dummy_draw)
    return min_font_size, lines

def create_text_gradient(
    width: int,
    height: int,
    position: str = "bottom",
    max_alpha: int = 140
) -> Image.Image:
    """
    Creates a gentle, seamless gradient behind text so white typography
    remains completely legible against any image background without a harsh box.
    """
    gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    if position == "bottom":
        start_y = int(height * 0.72)
        end_y = height
        span = max(1, end_y - start_y)
        for y in range(start_y, end_y):
            t = (y - start_y) / span
            # Smooth quadratic curve for natural cinematic vignette
            alpha = int(max_alpha * (t ** 1.4))
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    else:  # top
        start_y = 0
        end_y = int(height * 0.28)
        span = max(1, end_y - start_y)
        for y in range(start_y, end_y):
            t = (end_y - y) / span
            alpha = int(max_alpha * (t ** 1.4))
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

    return gradient

def render_cover_text(
    canvas: Image.Image,
    phrase: str,
    font_path: str,
    position: str = "bottom",
    custom_y: Optional[int] = None
) -> Image.Image:
    """
    Renders high-impact cover typography onto the canvas:
    - Pure WHITE text fill.
    - Strong BLACK outline/stroke around every letter for instant mobile legibility.
    - Subtle soft drop shadow behind the outline for depth and separation from the photo.
    - Full color emoji support.
    """
    if not phrase or not phrase.strip():
        return canvas

    w, h = canvas.size
    font_size, lines = compute_cover_font_size(phrase, font_path, max_width=int(w * 0.88))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    pilmoji_ctx = Pilmoji(canvas, source=AppleEmojiSource) if Pilmoji else None

    line_h = get_line_height(font, pilmoji_ctx, draw)
    line_spacing = font_size * 0.15
    total_text_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing

    # Calculate starting Y
    if custom_y is not None:
        start_y = custom_y
    elif position == "top":
        start_y = int(h * 0.08)
    else:  # bottom
        # Leaves elegant margin from bottom (watermark sits at opposite pole)
        start_y = int(h * 0.92 - total_text_h)

    # Stroke width proportional to font size (typically 3 to 4 pixels)
    stroke_w = max(3, min(5, int(round(font_size * 0.055))))

    curr_y = start_y
    for line in lines:
        line_w = get_text_width(line, font, pilmoji_ctx, draw)
        curr_x = (w - line_w) // 2

        # 1. Subtle soft shadow behind the outline for depth & separation
        text_only = emoji.replace_emoji(line, "") if emoji else line
        if text_only.strip():
            draw.text(
                (curr_x + 3, curr_y + 4),
                text_only,
                fill=(0, 0, 0, 140),
                font=font,
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 140)
            )

        # 2. Main crisp typography with pure WHITE fill and solid BLACK outline
        if pilmoji_ctx:
            # Pilmoji handles both the white text with solid black stroke and full-color emojis in a single pass
            pilmoji_ctx.text(
                (curr_x, curr_y),
                line,
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 255)
            )
        else:
            if text_only.strip():
                draw.text(
                    (curr_x, curr_y),
                    text_only,
                    fill=(255, 255, 255, 255),
                    font=font,
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0, 255)
                )

        curr_y += int(line_h + line_spacing)

    return canvas


# ---------------------------------------------------------------------------
# 5. Saba Bollywood Branding Overlay
# ---------------------------------------------------------------------------

def apply_branding(
    canvas: Image.Image,
    watermark_path: str,
    text_position: str = "bottom"
) -> Image.Image:
    """
    Composites the Saba Bollywood watermark asset proportionally onto the cover frame.
    Places the branding at the opposite pole of the cover text (e.g. text at bottom -> branding at top)
    to achieve balanced visual hierarchy and leave the center clear for the celebrity.
    """
    if not watermark_path or not os.path.exists(watermark_path):
        return canvas

    try:
        with Image.open(watermark_path) as wm:
            orig_w, orig_h = wm.size
            # Scale proportionally: 27px height for 1080x1080 standalone image (width ~316px)
            scale = 27.0 / float(orig_h)
            new_w = int(orig_w * scale)
            new_h = 27
            wm_scaled = wm.resize((new_w, new_h), Image.Resampling.LANCZOS)

        w, h = canvas.size
        wm_x = (w - new_w) // 2

        if text_position == "bottom":
            # Text is at bottom -> branding sits elegantly at top center
            wm_y = int(h * 0.045)
        else:
            # Text is at top -> branding sits at bottom center
            wm_y = int(h * 0.955 - new_h)

        canvas.alpha_composite(wm_scaled, dest=(wm_x, wm_y))
    except Exception as e:
        print(f"[COVER COMPOSITOR WARNING] Could not apply branding: {e}")

    return canvas


# ---------------------------------------------------------------------------
# 6. Main Orchestrator: create_cover_frame
# ---------------------------------------------------------------------------

def create_cover_frame(
    input_image: Union[str, Image.Image],
    phrase: str,
    output_path: Optional[str] = None,
    target_size: Tuple[int, int] = (1080, 1080),
    font_path: Optional[str] = None,
    watermark_path: Optional[str] = None,
    text_position: str = "auto",
    apply_finish: bool = True
) -> dict:
    """
    Composes a complete 1080x1080 Saba Bollywood Cover Frame:
    1. Prepares & scales source image maintaining square aspect ratio.
    2. Applies subtle visual finish pass (contrast, saturation, sharpness, vignette).
    3. Evaluates smart text placement (or respects manual override).
    4. Applies natural text backdrop gradient.
    5. Renders high-impact cover phrase with emoji support.
    6. Applies proportional Saba Bollywood branding at opposite pole.
    7. Optionally saves high-quality JPEG or PNG.
    """
    # Load input image
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image not found: {input_image}")
        raw_img = Image.open(input_image)
        raw_img.load()
    else:
        raw_img = input_image.copy()

    # 1. Scale to square target size
    canvas = prepare_source_image(raw_img, target_size=target_size)

    # 2. Subtle visual finish
    if apply_finish:
        canvas = apply_visual_finish(canvas)

    # 3. Typography & Placement analysis
    resolved_font = font_path or DEFAULT_FONT_PATH
    if not os.path.exists(resolved_font):
        resolved_font = "fonts/Calistoga-Regular.ttf"

    text_h = 150
    if phrase and phrase.strip():
        font_size, lines = compute_cover_font_size(phrase, resolved_font, max_width=int(canvas.width * 0.88))
        try:
            meas_font = ImageFont.truetype(resolved_font, font_size)
        except IOError:
            meas_font = ImageFont.load_default()
        line_h = get_line_height(meas_font, None, ImageDraw.Draw(canvas))
        line_spacing = font_size * 0.15
        text_h = int(len(lines) * line_h + max(0, len(lines) - 1) * line_spacing)

    force_pos = text_position if text_position in ("top", "bottom") else None
    placement_info = evaluate_placement_zones(canvas, text_height=text_h, force_position=force_pos)
    selected_position = placement_info["recommended"]
    selected_y = placement_info["scores"][selected_position]["y_start"]

    # 4. Natural text backdrop gradient
    if phrase and phrase.strip():
        text_grad = create_text_gradient(canvas.width, canvas.height, position=selected_position)
        canvas.alpha_composite(text_grad)

    # 5. Render cover text with white fill and solid black outline
    canvas = render_cover_text(canvas, phrase, resolved_font, position=selected_position, custom_y=selected_y)

    # 6. Apply branding
    resolved_wm = watermark_path if watermark_path is not None else DEFAULT_WATERMARK_PATH
    canvas = apply_branding(canvas, resolved_wm, text_position=selected_position)

    # 7. Convert to RGB for final output
    final_output = canvas.convert("RGB")

    # 8. Save if output_path specified
    if output_path:
        out_p = Path(output_path).resolve()
        os.makedirs(out_p.parent, exist_ok=True)
        ext = out_p.suffix.lower()
        if ext in (".png",):
            canvas.save(str(out_p), format="PNG")
        else:
            # Default to high-quality JPEG
            final_output.save(str(out_p), format="JPEG", quality=95, subsampling=0)

    return {
        "output_image": final_output,
        "output_path": output_path,
        "size": final_output.size,
        "phrase": phrase,
        "selected_position": selected_position,
        "placement_scores": placement_info["scores"],
        "subject_region": placement_info.get("subject_region"),
        "font_used": resolved_font,
        "watermark_used": bool(resolved_wm and os.path.exists(resolved_wm))
    }


# ---------------------------------------------------------------------------
# 7. Standalone CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Saba Bollywood Cover Frame Compositor - Phase 3"
    )
    parser.add_argument("--input", required=True, type=str, help="Path to input candidate image")
    parser.add_argument("--phrase", required=True, type=str, help="Short cover phrase (e.g. 'SALMAN DID THIS 😳')")
    parser.add_argument("--output", type=str, default="thumbnail_candidates/cover_test.jpg", help="Output file path (default: thumbnail_candidates/cover_test.jpg)")
    parser.add_argument("--size", type=int, default=1080, help="Target square dimension (default: 1080)")
    parser.add_argument("--position", type=str, default="auto", choices=["auto", "top", "bottom"], help="Text position (default: auto)")
    parser.add_argument("--font", type=str, default=None, help="Path to custom font")
    parser.add_argument("--watermark", type=str, default=None, help="Path to custom watermark asset")
    parser.add_argument("--no-finish", action="store_true", help="Disable subtle visual finish pass")

    args = parser.parse_args()

    input_path = str(Path(args.input).resolve())
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at '{input_path}'")
        sys.exit(1)

    result = create_cover_frame(
        input_image=input_path,
        phrase=args.phrase,
        output_path=args.output,
        target_size=(args.size, args.size),
        font_path=args.font,
        watermark_path=args.watermark,
        text_position=args.position,
        apply_finish=not args.no_finish
    )

    top_s = result["placement_scores"]["top"]["total_clutter"]
    bot_s = result["placement_scores"]["bottom"]["total_clutter"]
    top_coll = result["placement_scores"]["top"]["collides_with_face"]
    bot_coll = result["placement_scores"]["bottom"]["collides_with_face"]

    print("=" * 60)
    print("SABA BOLLYWOOD COVER FRAME COMPOSITOR - PHASE 3")
    print("=" * 60)
    print(f"Input Image:        {args.input}")
    print(f"Output Image:       {args.output}")
    print(f"Target Size:        {result['size'][0]} x {result['size'][1]}")
    print(f"Phrase:             \"{result['phrase']}\"")
    print(f"Selected Position:  {result['selected_position']}")
    print(f"Top Clutter:        {top_s} (face collision: {'YES' if top_coll else 'NO'})")
    print(f"Bottom Clutter:     {bot_s} (face collision: {'YES' if bot_coll else 'NO'})")
    print(f"Branding Applied:   {'Yes' if result['watermark_used'] else 'No'}")
    print(f"Visual Finish:      {'No' if args.no_finish else 'Yes'}")
    print(f"Status:             Saved {result['size'][0]}x{result['size'][1]} successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()

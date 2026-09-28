import os
import string
import emoji
from PIL import Image, ImageDraw, ImageFont

try:
    from pilmoji import Pilmoji
    from pilmoji.source import AppleEmojiSource
except ImportError:
    Pilmoji = None
    AppleEmojiSource = None

def get_word_list(caption):
    """
    Returns a list of (word, color) tuples.
    Main text is black, curiosity text is red.
    The emoji is appended to the last word so it wraps inline.
    """
    words = []
    
    if caption.main_text:
        text = caption.main_text.replace("\n", " \n ")
        for w in text.split(" "):
            if w == "\n":
                words.append(("\n", None))
                continue
            if w:
                if getattr(caption, "mode", "") == "main":
                    clean_w = w.translate(str.maketrans('', '', string.punctuation))
                    if clean_w and clean_w.isupper():
                        words.append((w, (255, 0, 0, 255)))
                        continue
                words.append((w, (0, 0, 0, 255)))
                
    if caption.curiosity_text:
        text = caption.curiosity_text.replace("\n", " \n ")
        c_words = [w for w in text.split(" ") if w]
        if caption.emoji:
            for i in range(len(c_words)-1, -1, -1):
                if c_words[i] != "\n":
                    c_words[i] += " " + caption.emoji
                    break
            else:
                c_words.append(caption.emoji)
                
        for w in c_words:
            if w == "\n":
                words.append(("\n", None))
                continue
            words.append((w, (255, 0, 0, 255)))
    else:
        if caption.emoji:
            if words:
                last_w, color = words.pop()
                if last_w == "\n":
                    words.append(("\n", None))
                    words.append((caption.emoji, (0, 0, 0, 255)))
                else:
                    words.append((last_w + " " + caption.emoji, color))
            else:
                words.append((caption.emoji, (0, 0, 0, 255)))
                
    return words

def get_text_width(word, font, pilmoji_context, draw):
    if pilmoji_context:
        return pilmoji_context.getsize(word, font=font)[0]
    return draw.textlength(word, font=font)

def wrap_words(words, font, max_width, draw, pilmoji_context):
    lines = []
    current_line = []
    current_width = 0
    space_w = get_text_width(" ", font, pilmoji_context, draw)
    
    for word, color in words:
        if word == "\n":
            if current_line:
                lines.append(current_line)
            else:
                lines.append([(" ", (0,0,0,0))])
            current_line = []
            current_width = 0
            continue
            
        w_w = get_text_width(word, font, pilmoji_context, draw)
        
        if w_w > max_width:
            return None # Word wider than max width
            
        if not current_line:
            current_line.append((word, color))
            current_width = w_w
        else:
            if current_width + space_w + w_w <= max_width:
                current_line.append((word, color))
                current_width += space_w + w_w
            else:
                lines.append(current_line)
                current_line = [(word, color)]
                current_width = w_w
                
    if current_line:
        lines.append(current_line)
        
    return lines

def calculate_visual_font_scale(reference_font_path: str, alternative_font_path: str, reference_size: int, sample_text: str = "AydY~.") -> int:
    """
    Measures actual glyph geometry using Pillow.
    Calculates the alternative font size required to match the reference font's
    visible glyph bounding-box height at reference_size.
    """
    img = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(img)
    
    try:
        ref_font = ImageFont.truetype(reference_font_path, reference_size)
    except IOError:
        return reference_size
        
    ref_bbox = draw.textbbox((0, 0), sample_text, font=ref_font)
    target_h = ref_bbox[3] - ref_bbox[1]
    if target_h <= 0:
        return reference_size
        
    best_fs = reference_size
    best_diff = float('inf')
    
    min_search = max(10, int(reference_size * 0.5))
    max_search = min(200, int(reference_size * 2.0))
    
    for fs in range(max_search, min_search - 1, -1):
        try:
            alt_font = ImageFont.truetype(alternative_font_path, fs)
        except IOError:
            continue
        bbox = draw.textbbox((0, 0), sample_text, font=alt_font)
        h = bbox[3] - bbox[1]
        diff = abs(h - target_h)
        if diff < best_diff:
            best_diff = diff
            best_fs = fs
            if diff == 0:
                break
                
    return best_fs

def compute_best_font_size(caption, font_path: str, max_width: int, max_lines: int) -> int:
    img = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    words = get_word_list(caption)
    if not words:
        return 82

    # 1. Calistoga Reference Baseline
    CALISTOGA_PATH = str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "Calistoga-Regular.ttf"))
    if not os.path.exists(CALISTOGA_PATH):
        CALISTOGA_PATH = "fonts/Calistoga-Regular.ttf"

    cal_base_size = 38
    for target_lines in range(1, max_lines + 1):
        found = False
        for fs in range(82, 37, -2):
            try:
                font = ImageFont.truetype(CALISTOGA_PATH, fs)
            except IOError:
                continue
            lines = wrap_words(words, font, max_width, draw, pilmoji_context)
            if lines is not None and len(lines) <= target_lines:
                cal_base_size = fs
                found = True
                break
        if found:
            break

    # Extract representative sample text from caption for geometry measurement
    raw_text = caption.main_text if hasattr(caption, 'main_text') else str(caption)
    sample_text = raw_text.strip() if raw_text.strip() else "AydY~."

    # Calistoga reference height at base size
    try:
        cal_font = ImageFont.truetype(CALISTOGA_PATH, cal_base_size)
        cal_bbox = draw.textbbox((0, 0), sample_text, font=cal_font)
        cal_visible_h = cal_bbox[3] - cal_bbox[1]
    except IOError:
        cal_visible_h = 0

    # 2. Alternative Font Scaling based on visual glyph height
    if os.path.normpath(font_path) == os.path.normpath(CALISTOGA_PATH):
        alt_base_size = cal_base_size
        alt_visible_h = cal_visible_h
    else:
        alt_base_size = calculate_visual_font_scale(CALISTOGA_PATH, font_path, cal_base_size, sample_text)
        try:
            alt_font = ImageFont.truetype(font_path, alt_base_size)
            alt_bbox = draw.textbbox((0, 0), sample_text, font=alt_font)
            alt_visible_h = alt_bbox[3] - alt_bbox[1]
        except IOError:
            alt_visible_h = 0

        # Safety check: make sure single word does not exceed max_width
        try:
            f_check = ImageFont.truetype(font_path, alt_base_size)
            if wrap_words(words, f_check, max_width, draw, pilmoji_context) is None:
                for fs in range(alt_base_size - 1, 19, -1):
                    try:
                        f_sub = ImageFont.truetype(font_path, fs)
                        if wrap_words(words, f_sub, max_width, draw, pilmoji_context) is not None:
                            alt_base_size = fs
                            break
                    except IOError:
                        continue
        except IOError:
            pass

    # Restored historical scale (no 0.8 reduction)
    final_font_size = alt_base_size

    # Count final lines for diagnostic logging
    try:
        final_font = ImageFont.truetype(font_path, final_font_size)
        final_lines = wrap_words(words, final_font, max_width, draw, pilmoji_context) or []
        line_count = len(final_lines)
    except IOError:
        line_count = 0

    # Production Diagnostic Logging
    selected_font = os.path.basename(font_path)
    print(f"[DIAGNOSTIC] selected_font: {selected_font}")
    print(f"[DIAGNOSTIC] font_path: {font_path}")
    print(f"[DIAGNOSTIC] calistoga_reference_size: {cal_base_size}")
    print(f"[DIAGNOSTIC] calistoga_visible_height: {cal_visible_h}")
    print(f"[DIAGNOSTIC] alternative_font_size: {alt_base_size}")
    print(f"[DIAGNOSTIC] alternative_visible_height: {alt_visible_h}")
    print(f"[DIAGNOSTIC] final_font_size: {final_font_size}")
    print(f"[DIAGNOSTIC] line_count: {line_count}")

    return final_font_size

def draw_caption(img: Image.Image, caption, font_path: str, font_size: int, max_width: int):
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    words = get_word_list(caption)
    if not words:
        return
        
    CALISTOGA_PATH = str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "Calistoga-Regular.ttf"))
    try:
        best_font = ImageFont.truetype(font_path, font_size)
    except IOError:
        try:
            best_font = ImageFont.truetype(CALISTOGA_PATH, font_size)
        except IOError:
            best_font = ImageFont.load_default()
        
    best_lines = wrap_words(words, best_font, max_width, draw, pilmoji_context) or []
        
    # Baseline anchoring logic
    if pilmoji_context:
        _, line_height = pilmoji_context.getsize("AydY~.", font=best_font)
    else:
        bbox = draw.textbbox((0, 0), "AydY~.", font=best_font)
        line_height = bbox[3] - bbox[1]
        
    try:
        ascent, descent = best_font.getmetrics()
    except AttributeError:
        ascent = line_height * 0.8
        
    line_spacing = font_size * 0.1
    
    # Anchor the baseline of the LAST line consistently at Y=375 for all fonts.
    # This guarantees identical 30-35px separation from the Y=420 video frame 
    # regardless of an alternative font's descender depth.
    last_line_top = 375 - ascent
    current_y = last_line_top - (max(0, len(best_lines) - 1) * (line_height + line_spacing))
    
    space_w = get_text_width(" ", best_font, pilmoji_context, draw)
    
    for line in best_lines:
        line_width = sum(get_text_width(w, best_font, pilmoji_context, draw) for w, c in line) + space_w * (len(line) - 1)
        current_x = (img.width - line_width) // 2
        
        if pilmoji_context:
            full_line_text = " ".join(w for w, c in line)
            pilmoji_context.text((current_x, current_y), full_line_text, fill=(0, 0, 0, 0), font=best_font)
            
        for word, color in line:
            text_only = emoji.replace_emoji(word, "")
                
            if text_only.strip():
                draw.text((current_x, current_y), text_only, fill=color, font=best_font)
            
            current_x += get_text_width(word, best_font, pilmoji_context, draw) + space_w
            
        current_y += line_height + line_spacing

def generate_text_overlay(
    caption, 
    font_path: str, 
    font_size: int,
    output_path: str
):
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    
    # User requested exactly 936px max width to sit 36px inside the 1008px video frame
    draw_caption(img, caption, font_path, font_size=font_size, max_width=936)
    
    img.save(output_path)
    return output_path

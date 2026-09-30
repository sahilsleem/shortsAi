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

def is_main_word_red(word: str) -> bool:
    """
    Returns True if the first alphabetic character of the word is uppercase.
    Ignores leading punctuation, quotes, brackets, symbols, and emojis.
    Returns False if there are no alphabetic characters, if the first alphabetic
    character is lowercase, or if the word begins with a number (e.g. '123Salman').
    """
    i = 0
    while i < len(word):
        ch = word[i]
        if ch.isdigit():
            return False
        if ch.isalpha():
            return ch.isupper()
        i += 1
    return False

def get_word_list(caption):
    """
    Returns a list of (word, color) tuples.
    Main text is black, curiosity text is red.
    The emoji is appended to the last word so it wraps inline.
    """
    words = []
    
    if caption.main_text:
        text = caption.main_text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", " \n ")
        for w in text.split(" "):
            if w == "\n":
                words.append(("\n", None))
                continue
            if w:
                if getattr(caption, "mode", "") == "main":
                    if is_main_word_red(w):
                        words.append((w, (255, 0, 0, 255)))
                        continue
                words.append((w, (0, 0, 0, 255)))
                
    if caption.curiosity_text:
        text = caption.curiosity_text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", " \n ")
        c_words = [w for w in text.split(" ") if w]
        if caption.emoji:
            for i in range(len(c_words) - 1, -1, -1):
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
                for i in range(len(words) - 1, -1, -1):
                    if words[i][0] != "\n":
                        w, col = words[i]
                        words[i] = (w + " " + caption.emoji, col)
                        break
                else:
                    words.append((caption.emoji, (0, 0, 0, 255)))
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

def compute_best_font_size(caption, font_path: str, max_width: int, max_lines: int, composition_scale: float = 1.0) -> int:
    # Apply composition scale: larger scale → narrower effective width → more wrapping
    # This preserves the existing 30px fitting rule while making text appear larger
    effective_max_width = int(max_width / max(0.6, min(1.4, composition_scale))) if composition_scale != 1.0 else max_width

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
    target_line_count = max_lines
    for target_lines in range(1, max_lines + 1):
        found = False
        for fs in range(82, 37, -2):
            try:
                font = ImageFont.truetype(CALISTOGA_PATH, fs)
            except IOError:
                continue
            lines = wrap_words(words, font, effective_max_width, draw, pilmoji_context)
            if lines is not None and len(lines) <= target_lines:
                cal_base_size = fs
                target_line_count = len(lines)
                found = True
                break
        if found:
            break

    # Extract clean representative line from caption for single-line glyph height measurement
    raw_text = ""
    if hasattr(caption, 'main_text') and caption.main_text:
        raw_text += caption.main_text
    if hasattr(caption, 'curiosity_text') and caption.curiosity_text:
        raw_text += " " + caption.curiosity_text
    if not raw_text:
        raw_text = str(caption)

    # Use clean text (no newlines, no emojis) for single-line glyph height measurement
    clean_lines = [emoji.replace_emoji(l, "").strip() for l in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    clean_lines = [l for l in clean_lines if l]
    sample_text = max(clean_lines, key=len) if clean_lines else "AydY~."

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
        
        # Verify alternative font fits within max_width and target_line_count if possible
        for fs in range(alt_base_size, 19, -1):
            try:
                f_check = ImageFont.truetype(font_path, fs)
            except IOError:
                continue
            w_res = wrap_words(words, f_check, effective_max_width, draw, pilmoji_context)
            if w_res is not None and len(w_res) <= target_line_count:
                alt_base_size = fs
                break
        else:
            # Fallback if it couldn't fit target_line_count down to 20: ensure at least individual words fit
            for fs in range(alt_base_size, 19, -1):
                try:
                    f_check = ImageFont.truetype(font_path, fs)
                except IOError:
                    continue
                if wrap_words(words, f_check, effective_max_width, draw, pilmoji_context) is not None:
                    alt_base_size = fs
                    break

        try:
            alt_font = ImageFont.truetype(font_path, alt_base_size)
            alt_bbox = draw.textbbox((0, 0), sample_text, font=alt_font)
            alt_visible_h = alt_bbox[3] - alt_bbox[1]
        except IOError:
            alt_visible_h = 0

    # Restored historical scale (no 0.8 reduction)
    final_font_size = alt_base_size

    # Count final lines for diagnostic logging
    try:
        final_font = ImageFont.truetype(font_path, final_font_size)
        final_lines = wrap_words(words, final_font, effective_max_width, draw, pilmoji_context) or []
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

def draw_caption(img: Image.Image, caption, font_path: str, font_size: int, max_width: int,
                 composition_scale: float = 1.0, composition_offset_x: int = 0, composition_offset_y: int = 0):
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    words = get_word_list(caption)
    if not words:
        return

    # Apply composition scale to max_width inversely for wrapping
    effective_max_width = int(max_width / max(0.6, min(1.4, composition_scale))) if composition_scale != 1.0 else max_width
        
    CALISTOGA_PATH = str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "Calistoga-Regular.ttf"))
    try:
        best_font = ImageFont.truetype(font_path, font_size)
    except IOError:
        try:
            best_font = ImageFont.truetype(CALISTOGA_PATH, font_size)
        except IOError:
            best_font = ImageFont.load_default()
        
    best_lines = wrap_words(words, best_font, effective_max_width, draw, pilmoji_context) or []
        
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
    last_line_top = 375 - ascent + composition_offset_y
    current_y = last_line_top - (max(0, len(best_lines) - 1) * (line_height + line_spacing))

    # Clamp composition to stay within canvas bounds (prevent complete overflow)
    # Allow text to extend partially off-canvas but not completely
    min_y = -200  # Allow some overflow above
    max_y = img.height - 50  # Must keep at least some text visible
    if current_y < min_y:
        current_y = min_y
    elif current_y > max_y:
        current_y = max_y
    
    space_w = get_text_width(" ", best_font, pilmoji_context, draw)
    
    for line in best_lines:
        line_width = sum(get_text_width(w, best_font, pilmoji_context, draw) for w, c in line) + space_w * (len(line) - 1)
        current_x = (img.width - line_width) // 2 + composition_offset_x

        # Clamp horizontal position
        current_x = max(-line_width + 50, min(current_x, img.width - 50))
        
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
    output_path: str,
    composition_scale: float = 1.0,
    composition_offset_x: int = 0,
    composition_offset_y: int = 0
):
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    
    # User requested exactly 936px max width to sit 36px inside the 1008px video frame
    draw_caption(img, caption, font_path, font_size=font_size, max_width=936,
                 composition_scale=composition_scale,
                 composition_offset_x=composition_offset_x,
                 composition_offset_y=composition_offset_y)
    
    img.save(output_path)
    return output_path

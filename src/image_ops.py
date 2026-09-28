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

def measure_footprint(lines, font, pilmoji_context, draw):
    if not lines:
        return 0, 0
    space_w = get_text_width(" ", font, pilmoji_context, draw)
    max_w = 0
    for line in lines:
        w = sum(get_text_width(word, font, pilmoji_context, draw) for word, c in line) + space_w * max(0, len(line) - 1)
        if w > max_w:
            max_w = w
            
    if pilmoji_context:
        _, lh = pilmoji_context.getsize("AydY~.", font=font)
    else:
        bbox = draw.textbbox((0,0), "AydY~.", font=font)
        lh = bbox[3] - bbox[1]
        
    line_spacing = font.size * 0.1
    total_h = (len(lines) * lh) + (max(0, len(lines) - 1) * line_spacing)
    return max_w, total_h

def compute_best_font_size(caption, font_path: str, max_width: int, max_lines: int) -> int:
    img = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    words = get_word_list(caption)
    if not words:
        return int(82 * 0.8)

    # 1. Calistoga Reference Layout
    CALISTOGA_PATH = str(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts", "Calistoga-Regular.ttf"))
    if not os.path.exists(CALISTOGA_PATH):
        CALISTOGA_PATH = "fonts/Calistoga-Regular.ttf"

    cal_fs = 82
    cal_font = None
    for target_lines in range(1, max_lines + 1):
        found = False
        for fs in range(82, 37, -2):
            try:
                font = ImageFont.truetype(CALISTOGA_PATH, fs)
            except IOError:
                font = ImageFont.load_default()
            lines = wrap_words(words, font, max_width, draw, pilmoji_context)
            if lines is not None and len(lines) <= target_lines:
                cal_fs = fs
                cal_font = font
                found = True
                break
        if found:
            break
            
    if cal_font is None:
        cal_fs = 38
        cal_font = ImageFont.truetype(CALISTOGA_PATH, 38)

    # Calculate Calistoga's visual glyph height
    cal_bbox = draw.textbbox((0, 0), "AydY~.", font=cal_font)
    target_h = cal_bbox[3] - cal_bbox[1]

    # If the requested font is Calistoga, just apply the 20% reduction and return
    if os.path.normpath(font_path) == os.path.normpath(CALISTOGA_PATH):
        return int(cal_fs * 0.8)

    # 2. Dynamic Calibration for Alternative Fonts
    best_fs = 38
    best_diff = float('inf')
    
    # Alternative fonts may need to scale up dramatically to match Calistoga's massive x-height
    for fs in range(250, 20, -1):
        try:
            alt_font = ImageFont.truetype(font_path, fs)
        except IOError:
            continue
            
        alt_bbox = draw.textbbox((0, 0), "AydY~.", font=alt_font)
        alt_h = alt_bbox[3] - alt_bbox[1]
        
        diff = abs(alt_h - target_h)
        if diff < best_diff:
            # Ensure this visually matched size doesn't fail the max_width physical boundary
            if wrap_words(words, alt_font, max_width, draw, pilmoji_context) is not None:
                best_diff = diff
                best_fs = fs
                # If we hit an exact match, we can stop early
                if diff == 0:
                    break

    return int(best_fs * 0.8)

def draw_caption(img: Image.Image, caption, font_path: str, font_size: int, max_width: int):
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    words = get_word_list(caption)
    if not words:
        return
        
    try:
        best_font = ImageFont.truetype(font_path, font_size)
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

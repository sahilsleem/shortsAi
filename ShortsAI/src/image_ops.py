import os
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
        for w in caption.main_text.split(" "):
            if w:
                words.append((w, (0, 0, 0, 255)))
                
    if caption.curiosity_text:
        c_words = [w for w in caption.curiosity_text.split(" ") if w]
        if caption.emoji:
            if c_words:
                c_words[-1] += " " + caption.emoji
            else:
                c_words.append(caption.emoji)
        for w in c_words:
            words.append((w, (255, 0, 0, 255)))
    else:
        if caption.emoji:
            if words:
                last_w, color = words.pop()
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

def compute_best_font_size(captions, font_path: str, max_width: int, max_lines: int) -> int:
    img = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(img)
    pilmoji_context = None
    if Pilmoji:
        pilmoji_context = Pilmoji(img, source=AppleEmojiSource)
        
    font_size = 140
    
    while font_size > 20:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            return 30
            
        all_fit = True
        for caption in captions:
            words = get_word_list(caption)
            if not words:
                continue
            lines = wrap_words(words, font, max_width, draw, pilmoji_context)
            if lines is None or len(lines) > max_lines:
                all_fit = False
                break
                
        if all_fit:
            return font_size
            
        font_size -= 4
        
    return 30

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
        
    # Positioning logic
    if pilmoji_context:
        _, line_height = pilmoji_context.getsize("AydY~.", font=best_font)
    else:
        bbox = draw.textbbox((0, 0), "AydY~.", font=best_font)
        line_height = bbox[3] - bbox[1]
        
    line_spacing = font_size * 0.1
    total_height = (len(best_lines) * line_height) + (max(0, len(best_lines) - 1) * line_spacing)
    
    # User requested BOTTOM edge at approximately Y=390 (30px gap)
    current_y = 390 - total_height
    space_w = get_text_width(" ", best_font, pilmoji_context, draw)
    
    for line in best_lines:
        line_width = sum(get_text_width(w, best_font, pilmoji_context, draw) for w, c in line) + space_w * (len(line) - 1)
        current_x = (img.width - line_width) // 2
        
        for word, color in line:
            # explicitly fallback to draw.text for non-emoji words to guarantee identical rendering metrics
            has_emoji = any(ord(c) > 127 for c in word)
            if has_emoji and pilmoji_context:
                pilmoji_context.text((current_x, current_y), word, fill=color, font=best_font)
            else:
                draw.text((current_x, current_y), word, fill=color, font=best_font)
            
            current_x += get_text_width(word, best_font, pilmoji_context, draw) + space_w
            
        current_y += line_height + line_spacing

def generate_text_overlay(
    caption, 
    font_path: str, 
    font_size: int,
    output_path: str
):
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    
    # User requested approx 880 max width for 100px breathing room
    draw_caption(img, caption, font_path, font_size=font_size, max_width=880)
    
    img.save(output_path)
    return output_path

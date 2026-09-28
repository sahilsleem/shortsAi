import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.config import CaptionData
from src.image_ops import generate_text_overlay, compute_best_font_size

def test_caption_wrap():
    os.makedirs("working", exist_ok=True)
    
    font_path = str(Path("fonts/Calistoga-Regular.ttf").resolve())
    
    long_caption = CaptionData(
        main_text="This is an extremely long and verbose caption that would normally take up to three or four or even five lines if we did not have a dynamic font scaling algorithm that forces it to compress.",
        curiosity_text="But wait until you see how it fits! 😅"
    )
    
    short_caption = CaptionData(
        main_text="Salman Khan was wrong,",
        curiosity_text="But then... 😅"
    )
    
    fs_long = compute_best_font_size(
        caption=long_caption,
        font_path=font_path,
        max_width=936,
        max_lines=3
    )
    print(f"Computed font size for long caption: {fs_long}")
    
    output_1 = str(Path("working/test_caption_long.png").resolve())
    generate_text_overlay(long_caption, font_path, fs_long, output_1)
    print(f"Generated test image for long caption at: {output_1}")
    
    fs_short = compute_best_font_size(
        caption=short_caption,
        font_path=font_path,
        max_width=936,
        max_lines=3
    )
    print(f"Computed font size for short caption: {fs_short}")
    
    output_2 = str(Path("working/test_caption_short.png").resolve())
    generate_text_overlay(short_caption, font_path, fs_short, output_2)
    print(f"Generated test image for short caption at: {output_2}")

if __name__ == "__main__":
    test_caption_wrap()

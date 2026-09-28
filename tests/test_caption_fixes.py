import os
import sys
import unittest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CaptionData
from src.image_ops import (
    get_word_list,
    wrap_words,
    compute_best_font_size,
    generate_text_overlay,
    is_main_word_red
)

class TestCaptionFixes(unittest.TestCase):

    def test_capitalization_rule(self):
        # Basic words
        self.assertTrue(is_main_word_red("Salman"))
        self.assertTrue(is_main_word_red("SALMAN"))
        self.assertFalse(is_main_word_red("salman"))
        
        # Punctuation handling
        self.assertTrue(is_main_word_red("(Salman)"))
        self.assertTrue(is_main_word_red("Salman,"))
        self.assertTrue(is_main_word_red("Salman."))
        self.assertTrue(is_main_word_red('"Salman"'))
        self.assertTrue(is_main_word_red("Salman!"))
        
        # Numbers / digits
        self.assertFalse(is_main_word_red("123Salman"))
        self.assertFalse(is_main_word_red("123"))
        
        # Emoji prefix
        self.assertTrue(is_main_word_red("👀Salman"))
        
        # Sentences from user prompt
        cap1 = CaptionData(main_text="Salman Khan spotted in Mumbai", mode="main")
        words1 = get_word_list(cap1)
        expected1 = [
            ("Salman", (255, 0, 0, 255)),
            ("Khan", (255, 0, 0, 255)),
            ("spotted", (0, 0, 0, 255)),
            ("in", (0, 0, 0, 255)),
            ("Mumbai", (255, 0, 0, 255)),
        ]
        self.assertEqual(words1, expected1)

        cap2 = CaptionData(main_text="I saw Salman yesterday", mode="main")
        words2 = get_word_list(cap2)
        expected2 = [
            ("I", (255, 0, 0, 255)),
            ("saw", (0, 0, 0, 255)),
            ("Salman", (255, 0, 0, 255)),
            ("yesterday", (0, 0, 0, 255)),
        ]
        self.assertEqual(words2, expected2)

    def test_curiosity_coloring_unchanged(self):
        # In curiosity mode, main_text is black and curiosity_text is red
        cap = CaptionData(main_text="Hook text,", curiosity_text="reveal text", emoji="👀")
        words = get_word_list(cap)
        # Hook words should be black
        self.assertEqual(words[0], ("Hook", (0, 0, 0, 255)))
        self.assertEqual(words[1], ("text,", (0, 0, 0, 255)))
        # Reveal words should be red
        self.assertEqual(words[2], ("reveal", (255, 0, 0, 255)))
        self.assertEqual(words[3], ("text 👀", (255, 0, 0, 255)))

    def test_newline_handling(self):
        font = ImageFont.truetype("fonts/Calistoga-Regular.ttf", 46)
        draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

        # 1. Two manual lines
        cap1 = CaptionData(main_text="Hello\nWorld", mode="main")
        words1 = get_word_list(cap1)
        lines1 = wrap_words(words1, font, 936, draw, None)
        self.assertEqual(len(lines1), 2)
        self.assertEqual([w[0] for w in lines1[0]], ["Hello"])
        self.assertEqual([w[0] for w in lines1[1]], ["World"])

        # 2. Three manual lines
        cap2 = CaptionData(main_text="Hello\nWorld\nAgain", mode="main")
        words2 = get_word_list(cap2)
        lines2 = wrap_words(words2, font, 936, draw, None)
        self.assertEqual(len(lines2), 3)

        # 3. Double newline (empty line) - should not crash
        cap3 = CaptionData(main_text="Hello\n\nWorld", mode="main")
        words3 = get_word_list(cap3)
        lines3 = wrap_words(words3, font, 936, draw, None)
        self.assertEqual(len(lines3), 3)

        # 4. CRLF normalization
        cap4 = CaptionData(main_text="Hello\r\nWorld\r\nAgain", mode="main")
        words4 = get_word_list(cap4)
        lines4 = wrap_words(words4, font, 936, draw, None)
        self.assertEqual(len(lines4), 3)

        # 5. Salman Khan prompt example across all five fonts
        fonts = [
            "fonts/Calistoga-Regular.ttf",
            "fonts/Alike-Regular.ttf",
            "fonts/CaslonOS-Regular.otf",
            "fonts/FiraMono-Regular.ttf",
            "fonts/SourceSansPro-Regular.ttf"
        ]
        salman_cap = CaptionData(
            main_text="Salman Khan was spotted today\nAnd here's who he was with 👀",
            mode="main"
        )
        os.makedirs("working/test_runs", exist_ok=True)
        for fpath in fonts:
            fs = compute_best_font_size(salman_cap, fpath, 936, 3)
            out_png = f"working/test_runs/overlay_{os.path.basename(fpath)}.png"
            generate_text_overlay(salman_cap, fpath, fs, out_png)
            self.assertTrue(os.path.exists(out_png))
            self.assertGreater(os.path.getsize(out_png), 1000)
            
            # Verify line count is exactly 2 for this caption
            f = ImageFont.truetype(fpath, fs)
            w_list = get_word_list(salman_cap)
            w_lines = wrap_words(w_list, f, 936, draw, None)
            self.assertEqual(len(w_lines), 2, f"Font {fpath} wrapped to {len(w_lines)} lines instead of 2")

if __name__ == '__main__':
    unittest.main()

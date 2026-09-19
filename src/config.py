import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class CaptionData:
    main_text: str
    curiosity_text: str = ""
    emoji: str = ""

@dataclass
class RenderConfig:
    source_video: str
    output_video: str
    
    hook_caption: CaptionData
    reveal_caption: CaptionData
    
    hook_start: float
    hook_end: float
    reveal_start: float
    reveal_end: float
    
    font_path: str = "fonts/Calistoga-Regular.ttf"

    @classmethod
    def from_json(cls, path: str) -> 'RenderConfig':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return cls(
            source_video=data["source_video"],
            output_video=data["output_video"],
            hook_caption=CaptionData(**data["hook_caption"]),
            reveal_caption=CaptionData(**data["reveal_caption"]),
            hook_start=data["hook_start"],
            hook_end=data["hook_end"],
            reveal_start=data["reveal_start"],
            reveal_end=data["reveal_end"],
            font_path=data.get("font_path", "fonts/Calistoga-Regular.ttf")
        )

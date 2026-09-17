import argparse
import os
import json
from pathlib import Path
from .config import RenderConfig
from .image_ops import generate_text_overlay, compute_best_font_size
from .video_ops import render_video, validate_render

def main():
    parser = argparse.ArgumentParser(description="ShortsAI Renderer - Milestone 1")
    
    # Core paths
    parser.add_argument("--input", type=str, help="Path to source video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--config", type=str, help="Path to JSON config file with all parameters")
    
    args = parser.parse_args()
    
    if args.config:
        config = RenderConfig.from_json(args.config)
        # Override with command line if provided
        if args.input:
            config.source_video = args.input
        if args.output:
            config.output_video = args.output
    else:
        print("Error: For Milestone 1, please provide a --config JSON file.")
        print("Example: python -m src.render --config job.json")
        return
        
    os.makedirs("working", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Robust path handling via pathlib
    source_video_path = str(Path(config.source_video).resolve())
    output_video_path = str(Path(config.output_video).resolve())
    font_path_resolved = str(Path(config.font_path).resolve())
    hook_overlay_path = str(Path("working/hook_overlay.png").resolve())
    reveal_overlay_path = str(Path("working/reveal_overlay.png").resolve())
    
    print("Computing universal font size...")
    best_font_size = compute_best_font_size(
        captions=[config.hook_caption, config.reveal_caption],
        font_path=font_path_resolved,
        max_width=880,
        max_lines=2
    )
    print(f"Universal Font Size computed as: {best_font_size}")
    
    print("Generating Hook Overlay...")
    generate_text_overlay(
        caption=config.hook_caption,
        font_path=font_path_resolved,
        font_size=best_font_size,
        output_path=hook_overlay_path
    )
    
    print("Generating Reveal Overlay...")
    generate_text_overlay(
        caption=config.reveal_caption,
        font_path=font_path_resolved,
        font_size=best_font_size,
        output_path=reveal_overlay_path
    )
    
    print("Rendering Video...")
    render_video(
        source_video=source_video_path,
        output_video=output_video_path,
        hook_overlay=hook_overlay_path,
        reveal_overlay=reveal_overlay_path,
        hook_start=config.hook_start,
        hook_end=config.hook_end,
        reveal_start=config.reveal_start,
        reveal_end=config.reveal_end
    )
    
    validate_render(output_video_path)
    print(f"Success! Video saved to {output_video_path}")

if __name__ == "__main__":
    main()

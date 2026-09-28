"""
src/frame_selector.py - Saba Bollywood Cover Frame Diagnostic Selector

Phase 2 Diagnostic System:
Inspects source video with identical ShortsAI crop parameters, samples candidate frames,
evaluates lightweight local visual metrics (sharpness, spatial composition, exposure,
contrast, saturation, transition penalties), deduplicates visually similar candidates,
and outputs the top 5 diverse candidates alongside a diagnostic JSON report.

Operates purely with standard library, Pillow, and FFmpeg. No heavyweight ML dependencies.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import math
from pathlib import Path
from PIL import Image, ImageFilter, ImageStat

# Import project utilities
from src.video_ops import get_ffmpeg_path, get_ffprobe_path, get_video_filters

# ---------------------------------------------------------------------------
# 1. Candidate Timestamp Sampling
# ---------------------------------------------------------------------------

def sample_candidate_timestamps(
    start_time: float,
    end_time: float,
    mode: str = "main",
    cut_time: float = None,
    interval: float = 0.6
) -> list[float]:
    """
    Generate deterministic, sorted timestamps for candidate frame sampling.
    - Excludes first 0.4s and final 0.5s of the trimmed segment.
    - In curiosity mode, excludes [cut_time - 0.25, cut_time + 0.25] to avoid transition fades.
    - Safely handles short video clips.
    """
    duration = end_time - start_time
    if duration <= 0:
        return [round(start_time, 3)]

    valid_start = start_time + 0.4
    valid_end = end_time - 0.5

    # Safe fallback for short clips where boundaries meet or invert
    if valid_end <= valid_start:
        midpoint = round(start_time + duration / 2.0, 3)
        return [midpoint]

    timestamps = []
    curr = valid_start
    while curr <= valid_end + 1e-5:
        # Avoid curiosity transition window (cut_time ± 0.25s)
        if mode == "curiosity" and cut_time is not None:
            if (cut_time - 0.25 - 1e-5) <= curr <= (cut_time + 0.25 + 1e-5):
                curr += interval
                continue

        timestamps.append(round(curr, 3))
        curr += interval

    if not timestamps:
        timestamps = [round(start_time + duration / 2.0, 3)]

    return sorted(timestamps)

# ---------------------------------------------------------------------------
# 2. Frame Extraction via FFmpeg
# ---------------------------------------------------------------------------

def extract_candidate_frame(
    source_video: str,
    timestamp: float,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_size: int = 1080,
    enhance: bool = False,
    target_size: tuple = (360, 360),
    output_path: str = None
) -> Image.Image:
    """
    Extract a low-resolution cropped frame suitable for scoring via FFmpeg.
    Applies identical crop coordinates (crop_x, crop_y, crop_size) used by ShortsAI,
    scaled down to target_size (e.g. 360x360) for fast phone evaluation.
    """
    ffmpeg = get_ffmpeg_path()
    vf = f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale={target_size[0]}:{target_size[1]}"
    if enhance:
        vf += f",{get_video_filters(enhance=True)}"

    is_temp = output_path is None
    if is_temp:
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        target_file = temp_file.name
        temp_file.close()
    else:
        target_file = output_path

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", source_video,
        "-vf", vf,
        "-frames:v", "1",
        "-q:v", "2",
        target_file
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        img = Image.open(target_file)
        img.load()  # Load all pixels into memory
        if is_temp:
            img_copy = img.copy()
            img.close()
            try:
                os.unlink(target_file)
            except OSError:
                pass
            return img_copy
        return img
    except Exception as e:
        if is_temp and os.path.exists(target_file):
            try:
                os.unlink(target_file)
            except OSError:
                pass
        raise RuntimeError(f"Failed to extract frame at {timestamp:.3f}s from '{source_video}': {e}")

def extract_full_res_frame(
    source_video: str,
    timestamp: float,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_size: int = 1080,
    enhance: bool = False,
    output_path: str = None
) -> Image.Image:
    """
    Structure hook for future phases: extracts the selected winning candidate
    at full 1002x1002 cropped resolution for eventual Cover Frame creation.
    """
    return extract_candidate_frame(
        source_video=source_video,
        timestamp=timestamp,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_size=crop_size,
        enhance=enhance,
        target_size=(1002, 1002),
        output_path=output_path
    )

# ---------------------------------------------------------------------------
# 3. Candidate Frame Scoring Engine
# ---------------------------------------------------------------------------

def compute_sharpness_score(img: Image.Image) -> float:
    """
    Measures edge variance using Pillow FIND_EDGES on grayscale.
    Range: 0 - 30.
    - Blurry / out-of-focus: var < 500 -> 0 to 10 pts.
    - Moderately sharp: var 1000 - 3000 -> 15 to 25 pts.
    - Crisp, sharp edges: var 4500+ -> 28 to 30 pts.
    Uses square-root scaling to reflect human perceptual clarity.
    """
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    edge_var = stat.var[0]

    if edge_var <= 50.0:
        return 0.0

    normalized = math.sqrt(max(0.0, edge_var - 50.0)) / math.sqrt(4500.0)
    score = 30.0 * min(1.0, normalized)
    return round(score, 1)

def compute_composition_score(img: Image.Image) -> float:
    """
    Evaluates spatial visual activity and subject prominence.
    Range: 0 - 25.
    - Evaluates visual activity (edge energy & local contrast) in the upper-center quadrant
      (X: 15%-85%, Y: 12%-70%) where framed subjects / interactions sit.
    - Contrast separation: foreground subject distinct from background.
    - Weak supporting chrominance signal: presence of warm/skin tones adds modest bonus (max 3 pts),
      ensuring non-skin clothing or multi-person scenes can still achieve top scores.
    """
    w, h = img.size
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # Upper-center subject box
    subject_box = (int(w * 0.15), int(h * 0.12), int(w * 0.85), int(h * 0.70))
    subject_edges = edges.crop(subject_box)
    sub_stat = ImageStat.Stat(subject_edges)
    sub_energy = sub_stat.mean[0]

    overall_stat = ImageStat.Stat(edges)
    overall_energy = overall_stat.mean[0]

    if overall_energy < 1.0:
        return 0.0

    # 1. Spatial activity distribution (0 - 16 pts)
    energy_ratio = sub_energy / max(1.0, overall_energy)
    base_activity = min(12.0, (sub_energy / 25.0) * 12.0)
    focus_bonus = min(4.0, max(0.0, (energy_ratio - 0.85) * 8.0))
    spatial_score = base_activity + focus_bonus

    # 2. Central contrast / Foreground-background separation (0 - 6 pts)
    sub_gray = gray.crop(subject_box)
    sub_stddev = ImageStat.Stat(sub_gray).stddev[0]
    contrast_separation = min(6.0, (sub_stddev / 50.0) * 6.0)

    # 3. Weak chrominance / warm tone supporting signal (0 - 3 pts)
    chroma_bonus = 0.0
    try:
        ycbcr = img.convert("YCbCr")
        ycbcr_crop = ycbcr.crop(subject_box)
        _, cb, cr = ycbcr_crop.split()
        cb_bytes = cb.tobytes()
        cr_bytes = cr.tobytes()
        warm_count = sum(1 for b, r in zip(cb_bytes, cr_bytes) if 75 <= b <= 135 and 130 <= r <= 180)
        warm_fraction = warm_count / max(1, len(cb_bytes))
        if 0.05 <= warm_fraction <= 0.60:
            chroma_bonus = 3.0 * min(1.0, warm_fraction / 0.20)
    except Exception:
        chroma_bonus = 0.0

    total_comp = min(25.0, spatial_score + contrast_separation + chroma_bonus)
    return round(max(0.0, total_comp), 1)

def compute_exposure_score(img: Image.Image) -> float:
    """
    Rewards useful luminance and severely penalizes dark/blown-out frames.
    Range: 0 - 20.
    - Extremely dark (L < 25): 0 to 2 pts.
    - Blown-out / flash (L > 230): 0 to 2 pts.
    - Broad optimal plateau between 85 and 155: 20 pts.
    """
    gray = img.convert("L")
    mean_lum = ImageStat.Stat(gray).mean[0]

    if mean_lum < 25.0:
        return round(max(0.0, (mean_lum / 25.0) * 2.0), 1)
    if mean_lum > 230.0:
        return round(max(0.0, ((255.0 - mean_lum) / 25.0) * 2.0), 1)

    if 85.0 <= mean_lum <= 155.0:
        return 20.0
    elif mean_lum < 85.0:
        return round(2.0 + (mean_lum - 25.0) / 60.0 * 18.0, 1)
    else:
        return round(20.0 - (mean_lum - 155.0) / 75.0 * 18.0, 1)

def compute_contrast_score(img: Image.Image) -> float:
    """
    Rewards healthy dynamic range and clear lighting separation.
    Range: 0 - 15.
    - Low stddev (< 15) indicates muddy/flat frames: 0 to 3 pts.
    - Standard deviation 45 - 75 indicates rich dynamics: 12 to 15 pts.
    """
    gray = img.convert("L")
    stddev = ImageStat.Stat(gray).stddev[0]

    if stddev < 15.0:
        return round(max(0.0, (stddev / 15.0) * 3.0), 1)

    score = 3.0 + min(12.0, (stddev - 15.0) / 35.0 * 12.0)
    return round(min(15.0, score), 1)

def compute_saturation_score(img: Image.Image) -> float:
    """
    Supporting color richness metric.
    Range: 0 - 10.
    - Muted/monochrome (< 20): 1 to 3 pts.
    - Healthy natural color (40 - 140): 8 to 10 pts.
    - Oversaturated (> 140): gently tapers so hypersaturation does not dominate.
    """
    hsv = img.convert("HSV")
    s_stat = ImageStat.Stat(hsv)
    mean_sat = s_stat.mean[1]

    if mean_sat < 20.0:
        return round(max(1.0, (mean_sat / 20.0) * 3.0), 1)

    if 40.0 <= mean_sat <= 140.0:
        return 10.0
    elif mean_sat < 40.0:
        return round(3.0 + (mean_sat - 20.0) / 20.0 * 7.0, 1)
    else:
        return round(max(6.0, 10.0 - (mean_sat - 140.0) / 115.0 * 4.0), 1)

def compute_penalties(
    timestamp: float,
    start_time: float,
    end_time: float,
    cut_time: float = None
) -> float:
    """
    Penalizes boundary timestamps and curiosity cut proximity.
    """
    penalties = 0.0
    if timestamp < start_time + 0.45:
        penalties += 10.0
    if timestamp > end_time - 0.55:
        penalties += 10.0
    if cut_time is not None and abs(timestamp - cut_time) <= 0.30:
        penalties += 25.0
    return penalties

def score_frame(
    img: Image.Image,
    timestamp: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 0.0,
    cut_time: float = None
) -> dict:
    """
    Compute comprehensive visual score for a candidate frame on a 0 - 100 scale.
    """
    sharpness = compute_sharpness_score(img)
    composition = compute_composition_score(img)
    exposure = compute_exposure_score(img)
    contrast = compute_contrast_score(img)
    saturation = compute_saturation_score(img)
    penalty = compute_penalties(timestamp, start_time, end_time, cut_time)

    # Extra safety penalty for nearly black frames
    gray = img.convert("L")
    if ImageStat.Stat(gray).mean[0] < 20.0:
        penalty += 30.0

    raw_total = sharpness + composition + exposure + contrast + saturation - penalty
    total_score = round(max(0.0, min(100.0, raw_total)), 1)

    return {
        "total_score": total_score,
        "sharpness": sharpness,
        "composition": composition,
        "exposure": exposure,
        "contrast": contrast,
        "saturation": saturation,
        "penalty": penalty
    }

# ---------------------------------------------------------------------------
# 4. Lightweight Perceptual Deduplication
# ---------------------------------------------------------------------------

def compute_image_signature(img: Image.Image) -> list[int]:
    """16x16 grayscale thumbnail pixel vector for fast perceptual comparison."""
    thumb = img.convert("L").resize((16, 16), Image.Resampling.BILINEAR)
    return list(thumb.tobytes())

def compute_signature_distance(sig1: list[int], sig2: list[int]) -> float:
    """Mean absolute pixel difference between two 16x16 signatures (0.0 to 255.0)."""
    return sum(abs(a - b) for a, b in zip(sig1, sig2)) / 256.0

def deduplicate_candidates(
    candidates: list[dict],
    min_distance: float = 14.0,
    min_time_delta: float = 1.2,
    max_results: int = 5
) -> list[dict]:
    """
    Greedy diversity selector. Ensures top candidates are visually distinct
    and not just five consecutive frames from the same moment.
    """
    if not candidates:
        return []

    # Sort strictly descending by total_score
    sorted_pool = sorted(candidates, key=lambda c: c["score_data"]["total_score"], reverse=True)

    selected = []
    # First pass: enforce strict visual distance and time separation
    for cand in sorted_pool:
        is_distinct = True
        for sel in selected:
            time_diff = abs(cand["timestamp"] - sel["timestamp"])
            visual_diff = compute_signature_distance(cand["signature"], sel["signature"])

            # If visually near-identical AND temporally close, consider duplicate
            if visual_diff < min_distance and time_diff < min_time_delta:
                is_distinct = False
                break

        if is_distinct:
            selected.append(cand)
            if len(selected) >= max_results:
                break

    # Second pass: if pool had fewer than max_results distinct moments, fill with next best
    if len(selected) < max_results:
        for cand in sorted_pool:
            if cand not in selected:
                selected.append(cand)
                if len(selected) >= max_results:
                    break

    # Assign ranks
    for idx, item in enumerate(selected, start=1):
        item["rank"] = idx

    return selected

# ---------------------------------------------------------------------------
# 5. Diagnostic Pipeline & Output Generation
# ---------------------------------------------------------------------------

def run_frame_selection_diagnostic(
    source_video: str,
    start_time: float,
    end_time: float,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_size: int = 1080,
    mode: str = "main",
    cut_time: float = None,
    interval: float = 0.6,
    output_dir: str = "thumbnail_candidates",
    enhance: bool = False
) -> dict:
    """
    Orchestrate the Phase 2 diagnostic run:
    1. Sample timestamps.
    2. Extract low-res 360x360 candidate frames.
    3. Score each candidate frame.
    4. Deduplicate to find top 5 diverse candidates.
    5. Save candidate_01.jpg ... candidate_05.jpg.
    6. Write diagnostic_report.json and print console summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamps = sample_candidate_timestamps(
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        cut_time=cut_time,
        interval=interval
    )

    extracted_candidates = []
    for ts in timestamps:
        try:
            img = extract_candidate_frame(
                source_video=source_video,
                timestamp=ts,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_size=crop_size,
                enhance=enhance,
                target_size=(360, 360)
            )
            scores = score_frame(
                img=img,
                timestamp=ts,
                start_time=start_time,
                end_time=end_time,
                cut_time=cut_time
            )
            sig = compute_image_signature(img)
            extracted_candidates.append({
                "timestamp": ts,
                "score_data": scores,
                "signature": sig,
                "image": img
            })
        except Exception as e:
            print(f"[DIAGNOSTIC WARNING] Could not extract frame at {ts:.3f}s: {e}")

    top_candidates = deduplicate_candidates(extracted_candidates, max_results=5)

    # Save images and prepare report structure
    top_report_list = []
    for cand in top_candidates:
        rank = cand["rank"]
        filename = f"candidate_{rank:02d}.jpg"
        file_path = os.path.join(output_dir, filename)
        cand["image"].save(file_path, "JPEG", quality=90)

        report_entry = {
            "rank": rank,
            "filename": filename,
            "timestamp": cand["timestamp"],
            "total_score": cand["score_data"]["total_score"],
            "scores": {
                "sharpness": cand["score_data"]["sharpness"],
                "composition": cand["score_data"]["composition"],
                "exposure": cand["score_data"]["exposure"],
                "contrast": cand["score_data"]["contrast"],
                "saturation": cand["score_data"]["saturation"],
                "penalty": cand["score_data"]["penalty"]
            }
        }
        top_report_list.append(report_entry)

    # Clean up image objects in memory
    for cand in extracted_candidates:
        cand["image"].close()

    selected_winner = top_report_list[0] if top_report_list else None

    report = {
        "source_path": str(Path(source_video).resolve()),
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 3),
        "mode": mode,
        "cut_time": cut_time,
        "sampling_interval": interval,
        "sampled_timestamps_count": len(timestamps),
        "extracted_frames_count": len(extracted_candidates),
        "selected_candidate": selected_winner,
        "top_candidates": top_report_list
    }

    report_path = os.path.join(output_dir, "diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console Diagnostic Output
    print("=" * 60)
    print("SABA BOLLYWOOD COVER FRAME - DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Source Video:       {source_video}")
    print(f"Segment:            {start_time:.2f}s -> {end_time:.2f}s ({end_time - start_time:.2f}s duration)")
    print(f"Sampled Timestamps: {len(timestamps)} at {interval}s interval")
    print(f"Extracted Frames:   {len(extracted_candidates)}")
    print("-" * 60)

    if selected_winner:
        print(f"Selected Candidate:  {selected_winner['timestamp']:.2f}s (Rank 1)")
        print(f"Total Score:         {selected_winner['total_score']} / 100")
        s = selected_winner["scores"]
        print(f"  - Sharpness:       {s['sharpness']} / 30")
        print(f"  - Composition:     {s['composition']} / 25")
        print(f"  - Exposure:        {s['exposure']} / 20")
        print(f"  - Contrast:        {s['contrast']} / 15")
        print(f"  - Saturation:      {s['saturation']} / 10")
        if s["penalty"] > 0:
            print(f"  - Penalties:       -{s['penalty']}")
        print(f"Saved Image:         {os.path.join(output_dir, selected_winner['filename'])}")
    else:
        print("No candidates successfully extracted.")

    print(f"Diagnostic Report:   {report_path}")
    print("=" * 60)

    return report

# ---------------------------------------------------------------------------
# 6. Standalone CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Saba Bollywood Cover Frame Diagnostic Selector - Phase 2"
    )
    parser.add_argument("--input", required=True, type=str, help="Path to source video file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0)")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: probe duration)")
    parser.add_argument("--crop-x", type=int, default=0, help="Crop X coordinate (default: 0)")
    parser.add_argument("--crop-y", type=int, default=0, help="Crop Y coordinate (default: 0)")
    parser.add_argument("--crop-size", type=int, default=1080, help="Crop square size (default: 1080)")
    parser.add_argument("--mode", type=str, default="main", choices=["main", "curiosity"], help="Mode (default: main)")
    parser.add_argument("--cut-time", type=float, default=None, help="Cut time for curiosity mode in seconds")
    parser.add_argument("--interval", type=float, default=0.6, help="Sampling interval in seconds (default: 0.6)")
    parser.add_argument("--output-dir", type=str, default="thumbnail_candidates", help="Output directory (default: thumbnail_candidates)")
    parser.add_argument("--enhance", action="store_true", help="Apply video enhancement filters")

    args = parser.parse_args()

    input_path = str(Path(args.input).resolve())
    if not os.path.exists(input_path):
        print(f"Error: Source video file not found at '{input_path}'")
        sys.exit(1)

    end_time = args.end
    if end_time is None:
        # Probe duration via ffprobe
        try:
            ffprobe = get_ffprobe_path()
            cmd = [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", input_path]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            end_time = float(data.get("format", {}).get("duration", 10.0))
        except Exception:
            end_time = 10.0

    run_frame_selection_diagnostic(
        source_video=input_path,
        start_time=args.start,
        end_time=end_time,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_size=args.crop_size,
        mode=args.mode,
        cut_time=args.cut_time,
        interval=args.interval,
        output_dir=args.output_dir,
        enhance=args.enhance
    )

if __name__ == "__main__":
    main()

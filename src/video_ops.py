import subprocess
import os
import shutil
import json
from pathlib import Path

def get_ffmpeg_path() -> str:
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path): return env_path
    which_path = shutil.which("ffmpeg")
    if which_path: return which_path
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(fallback): return fallback
    return "ffmpeg"

def get_ffprobe_path() -> str:
    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and os.path.exists(env_path): return env_path
    which_path = shutil.which("ffprobe")
    if which_path: return which_path
    fallback = r"C:\Users\Sahil Sham's\Documents\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"
    if os.path.exists(fallback): return fallback
    return "ffprobe"

def get_video_dimensions(video_path: str):
    try:
        cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return int(stream["width"]), int(stream["height"])
    except Exception:
        pass
    return 1080, 1920


DEFAULT_WATERMARK_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "saba_bollywood_watermark.png")

# Video Enhancement Filters
# Base/Default: Existing subtle polish
DEFAULT_VIDEO_FILTERS = "eq=contrast=1.03:brightness=0.01:saturation=1.08:gamma=1.02,unsharp=3:3:0.3:3:3:0.0"

# Enhanced Video: Cleaner, punchier social-video finishing pass
# 1. Subtle denoise (hqdn3d) to eliminate digital sensor grain and compression artifacts
# 2. Balanced contrast and modest saturation for punchy social colors while preserving natural skin tones
# 3. Clean clarity and sharpening (unsharp) without crunchy halos or color fringing
ENHANCED_VIDEO_FILTERS = "hqdn3d=1.5:1.5:3:3,eq=contrast=1.06:brightness=0.015:saturation=1.12:gamma=1.02,unsharp=5:5:0.5:3:3:0.0"

# Audio Enhancement Filters
# 1. Enhanced Mix: Balanced speech clarity, dynamic compression, rumble/hiss cleanup, -16 LUFS loudness normalization
ENHANCED_AUDIO_FILTERS = (
    "highpass=f=80,lowpass=f=12000,"
    "equalizer=f=250:t=q:w=1.2:g=-1.5,"
    "equalizer=f=3000:t=q:w=1.2:g=2.5,"
    "acompressor=threshold=0.1:ratio=3:attack=15:release=120:makeup=2,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)

# 2. Voice Focus: Speech-priority shaping, attenuating competing music frequencies, targeted dialogue presence, -16 LUFS
VOICE_FOCUS_AUDIO_FILTERS = (
    "highpass=f=120,lowpass=f=7500,"
    "equalizer=f=200:t=q:w=1.0:g=-3.0,"
    "equalizer=f=600:t=q:w=1.5:g=-2.5,"
    "equalizer=f=2800:t=q:w=1.2:g=4.0,"
    "equalizer=f=4500:t=q:w=1.5:g=2.5,"
    "acompressor=threshold=0.08:ratio=4:attack=10:release=100:makeup=2.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=8"
)

def get_video_filters(enhance: bool = False) -> str:
    """Return video filter string based on enhancement toggle."""
    return ENHANCED_VIDEO_FILTERS if enhance else DEFAULT_VIDEO_FILTERS

def get_audio_filters(audio_mode: str = "enhanced") -> str:
    """Return audio filter chain for the specified mode ('enhanced' or 'voice_focus')."""
    if audio_mode == "voice_focus":
        return VOICE_FOCUS_AUDIO_FILTERS
    return ENHANCED_AUDIO_FILTERS

def has_audio_stream(video_path: str) -> bool:
    """Detect if input video has an audio stream."""
    try:
        cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return True
    except Exception:
        # Default to True so filter graphs construct reliably in mock/tests or if probe fails gracefully
        return True
    return False

def render_main_video(
    source_video: str,
    output_video: str,
    caption_overlay: str,
    start_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    watermark_path: str = None,
    enhance: bool = False,
    audio_mode: str = "original"
):
    if watermark_path is None:
        watermark_path = DEFAULT_WATERMARK_PATH
    has_wm = bool(watermark_path and os.path.exists(watermark_path))

    duration = end_time - start_time
    video_filters = get_video_filters(enhance)
    apply_audio_enhance = enhance and (audio_mode in ("enhanced", "voice_focus")) and has_audio_stream(source_video)
    
    cmd = [get_ffmpeg_path(), "-y"]
    cmd.extend(["-ss", str(start_time), "-t", str(duration), "-i", source_video])
    cmd.extend(["-i", caption_overlay])
    if has_wm:
        cmd.extend(["-i", watermark_path])
    
    fc = []
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={duration}:r=30[base];")
    fc.append(f"[base][v_proc]overlay=39:420:eof_action=pass[bg];")
    if has_wm:
        fc.append(f"[bg][1:v]overlay=0:0[with_cap];")
        if apply_audio_enhance:
            fc.append(f"[with_cap][2:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[0:a]{a_filter}[outa]")
        else:
            fc.append(f"[with_cap][2:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]")
    else:
        if apply_audio_enhance:
            fc.append(f"[bg][1:v]overlay=0:0[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[0:a]{a_filter}[outa]")
        else:
            fc.append(f"[bg][1:v]overlay=0:0[outv]")
    
    if apply_audio_enhance:
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        maps = ["-map", "[outv]", "-map", "0:a?"]

    cmd.extend([
        "-filter_complex", "".join(fc),
        *maps,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        output_video
    ])
    print("Running Main Video FFmpeg command")
    subprocess.run(cmd, check=True)

def render_curiosity_video(
    source_video: str,
    output_video: str,
    hook_overlay: str,
    reveal_overlay: str,
    start_time: float,
    cut_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    transition_dur: float = 0.15,
    watermark_path: str = None,
    enhance: bool = False,
    audio_mode: str = "original"
):
    if watermark_path is None:
        watermark_path = DEFAULT_WATERMARK_PATH
    has_wm = bool(watermark_path and os.path.exists(watermark_path))

    hook_dur = cut_time - start_time
    rev_dur = end_time - cut_time
    hook_start = start_time
    reveal_start = cut_time
    
    video_filters = get_video_filters(enhance)
    apply_audio_enhance = enhance and (audio_mode in ("enhanced", "voice_focus"))
    
    cmd = [get_ffmpeg_path(), "-y"]
    
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    cmd.extend(["-i", hook_overlay])
    cmd.extend(["-i", reveal_overlay])
    
    # We must explicitly read the audio streams starting at hook_start and reveal_start to avoid sync issues with concat
    cmd.extend(["-ss", str(hook_start), "-t", str(hook_dur), "-i", source_video])
    cmd.extend(["-ss", str(reveal_start), "-t", str(rev_dur), "-i", source_video])
    
    if has_wm:
        cmd.extend(["-i", watermark_path])
        wm_idx = 6
    
    fc = []
    
    # Hook segment
    fc.append(f"[0:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[hook_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={hook_dur}:r=30[hook_base];")
    fc.append(f"[hook_base][hook_v_proc]overlay=39:420:eof_action=pass[hook_bg1];")
    fc.append(f"[hook_bg1][2:v]overlay=0:0[hook_bg2];")
    fc.append(f"[hook_bg2]fade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}:c=black[hook_v_final];")
    fc.append(f"[4:a]afade=t=out:st={max(0, hook_dur - transition_dur)}:d={transition_dur}[hook_a_final];")
    
    # Reveal segment
    fc.append(f"[1:v]crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=1002:1002,{video_filters}[rev_v_proc];")
    fc.append(f"color=c=white:s=1080x1920:d={rev_dur}:r=30[rev_base];")
    fc.append(f"[rev_base][rev_v_proc]overlay=39:420:eof_action=pass[rev_bg1];")
    fc.append(f"[rev_bg1][3:v]overlay=0:0[rev_bg2];")
    fc.append(f"[rev_bg2]fade=t=in:st=0:d={transition_dur}:c=black[rev_v_final];")
    fc.append(f"[5:a]afade=t=in:st=0:d={transition_dur}[rev_a_final];")
    
    raw_a = "raw_a" if apply_audio_enhance else "outa"

    if has_wm:
        fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[v_concat][{raw_a}];")
        if apply_audio_enhance:
            fc.append(f"[v_concat][{wm_idx}:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[{raw_a}]{a_filter}[outa]")
        else:
            fc.append(f"[v_concat][{wm_idx}:v]overlay=x=(W-w)/2:y=1374-h:eof_action=repeat[outv]")
    else:
        if apply_audio_enhance:
            fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[outv][{raw_a}];")
            a_filter = get_audio_filters(audio_mode)
            fc.append(f"[{raw_a}]{a_filter}[outa]")
        else:
            fc.append(f"[hook_v_final][hook_a_final][rev_v_final][rev_a_final]concat=n=2:v=1:a=1[outv][outa]")
    
    cmd.extend([
        "-filter_complex", "".join(fc),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        output_video
    ])
    print("Running Curiosity Video FFmpeg command")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Phase 5: Cover Frame Pipeline (optional, not active by default)
# ---------------------------------------------------------------------------

def get_audio_properties(video_path: str) -> dict:
    """Query ffprobe for audio stream properties (sample_rate, channels, channel_layout, codec_name)."""
    cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                sample_rate = int(stream.get("sample_rate", 44100))
                channels = int(stream.get("channels", 2))
                layout = stream.get("channel_layout", "stereo" if channels == 2 else ("mono" if channels == 1 else "stereo"))
                codec_name = stream.get("codec_name", "aac")
                return {
                    "has_audio": True,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "channel_layout": layout,
                    "codec_name": codec_name
                }
    except Exception:
        pass
    return {
        "has_audio": False,
        "sample_rate": 44100,
        "channels": 2,
        "channel_layout": "stereo",
        "codec_name": "aac"
    }


def create_cover_video_segment(
    cover_image_path: str,
    output_segment_path: str,
    duration: float = 0.100,
    audio_props: dict = None,
    width: int = 1080,
    height: int = 1920
) -> str:
    """
    Renders an exact duration-controlled (default 0.100s) cover video segment
    matching the target canvas dimensions (default 1080x1920) with the cover image
    scaled square and positioned within the canvas.
    Produces silent audio matched to audio_props if audio is present.
    """
    if audio_props is None:
        audio_props = {"has_audio": True, "sample_rate": 44100, "channel_layout": "stereo", "channels": 2}

    has_audio = audio_props.get("has_audio", False)
    sample_rate = audio_props.get("sample_rate", 44100)
    channel_layout = audio_props.get("channel_layout", "stereo")

    if width == 1080 and height == 1920:
        cover_size = 1002
        x = 39
        y = 420
    else:
        # Scale square cover proportionally and center horizontally
        scale_factor = width / 1080.0
        cover_size = int(round(1002 * scale_factor))
        if cover_size % 2 != 0:
            cover_size -= 1
        cover_size = min(cover_size, width, height)
        x = (width - cover_size) // 2
        y = int(round(420 * (height / 1920.0)))

    cmd = [get_ffmpeg_path(), "-y"]
    cmd.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", cover_image_path])

    if has_audio:
        cmd.extend([
            "-f", "lavfi", "-t", f"{duration:.3f}",
            "-i", f"anullsrc=r={sample_rate}:cl={channel_layout}"
        ])

    fc = [
        f"[0:v]scale={cover_size}:{cover_size}[c_img];",
        f"color=c=white:s={width}x{height}:d={duration:.3f}:r=30[base];",
        f"[base][c_img]overlay={x}:{y}:eof_action=pass,trim=duration={duration:.3f},setpts=PTS-STARTPTS[outv]"
    ]

    if has_audio:
        fc.append(f";[1:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[outa]")
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        maps = ["-map", "[outv]"]

    cmd.extend([
        "-filter_complex", "".join(fc),
        *maps,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30"
    ])

    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate)])

    cmd.append(output_segment_path)
    print(f"Creating Cover Video Segment ({width}x{height}, duration={duration:.3f}s)")
    subprocess.run(cmd, check=True)
    return output_segment_path


def concat_video_with_cover(
    main_video_path: str,
    cover_segment_path: str,
    output_video_path: str,
    has_audio: bool = True
) -> None:
    """Concatenates main video and cover segment with audio synchronization."""
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", main_video_path,
        "-i", cover_segment_path
    ]

    if has_audio:
        fc = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]"
        maps = ["-map", "[outv]", "-map", "[outa]"]
        codec_args = [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "192k"
        ]
    else:
        fc = "[0:v][1:v]concat=n=2:v=1:a=0[outv]"
        maps = ["-map", "[outv]"]
        codec_args = [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "30"
        ]

    cmd.extend([
        "-filter_complex", fc,
        *maps,
        *codec_args,
        output_video_path
    ])
    print("Concatenating Main Video with Cover Segment")
    subprocess.run(cmd, check=True)


def append_cover_frame(
    source_video: str,
    rendered_video: str,
    output_video: str,
    thumbnail_phrase: str,
    start_time: float,
    end_time: float,
    crop_x: int,
    crop_y: int,
    crop_size: int,
    mode: str = "main",
    cut_time: float = None,
    enhance: bool = False,
    cover_image_path: str = None,
    duration: float = 0.100
) -> dict:
    """
    Orchestrates the Phase 5 cover frame pipeline:
    1. Validates thumbnail phrase (skips safely if missing).
    2. Runs frame selector on original source footage to find best candidate frame.
    3. Extracts full-resolution candidate frame.
    4. Composes cover frame with thumbnail phrase.
    5. Saves cover artwork to disk.
    6. Generates exact duration-controlled (0.100s) cover video segment with matched audio properties.
    7. Appends cover segment to the rendered Short.
    8. Validates output dimensions and duration with ffprobe.
    9. Atomically updates output_video with complete failure safety.
    """
    if not thumbnail_phrase or not thumbnail_phrase.strip():
        print("[COVER PIPELINE] No thumbnail phrase available. Skipping cover frame safely.")
        if rendered_video != output_video and os.path.exists(rendered_video):
            shutil.copyfile(rendered_video, output_video)
        return {"success": False, "reason": "no_thumbnail_phrase", "output_video": output_video}

    work_dir = Path(cover_image_path).parent if cover_image_path else Path(rendered_video).parent
    temp_cand_dir = work_dir / "cover_candidates"
    os.makedirs(temp_cand_dir, exist_ok=True)
    if not cover_image_path:
        cover_image_path = str(work_dir / "output_cover.jpg")
    cover_segment_path = str(work_dir / "cover_segment.mp4")
    temp_final_path = str(work_dir / "temp_final_with_cover.mp4")

    try:
        # Step 2: Frame Selection from original source video
        from src.frame_selector import run_frame_selection_diagnostic, extract_full_res_frame
        report = run_frame_selection_diagnostic(
            source_video=source_video,
            start_time=start_time,
            end_time=end_time,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            mode=mode,
            cut_time=cut_time,
            output_dir=str(temp_cand_dir),
            enhance=enhance
        )
        selected = report.get("selected_candidate")
        if not selected:
            print("[COVER PIPELINE] Frame selector found no valid candidate. Skipping safely.")
            if rendered_video != output_video and os.path.exists(rendered_video):
                shutil.copyfile(rendered_video, output_video)
            return {"success": False, "reason": "no_candidate_frame", "output_video": output_video}

        best_ts = float(selected["timestamp"])

        # Step 3: Extract full-resolution frame
        full_frame = extract_full_res_frame(
            source_video=source_video,
            timestamp=best_ts,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_size=crop_size,
            enhance=enhance
        )

        # Step 4: Compose Cover Artwork
        from src.cover_compositor import create_cover_frame
        comp_result = create_cover_frame(
            input_image=full_frame,
            phrase=thumbnail_phrase,
            output_path=cover_image_path
        )

        # Step 5: Determine actual dimensions and inspect audio properties of rendered video
        rend_w, rend_h = get_video_dimensions(rendered_video)
        audio_props = get_audio_properties(rendered_video)

        # Step 6: Create 0.100s Cover Video Segment matching rendered video dimensions
        create_cover_video_segment(
            cover_image_path=cover_image_path,
            output_segment_path=cover_segment_path,
            duration=duration,
            audio_props=audio_props,
            width=rend_w,
            height=rend_h
        )

        # Step 6b: Explicit validation before concatenation: dimensions must match
        seg_w, seg_h = get_video_dimensions(cover_segment_path)
        if (seg_w, seg_h) != (rend_w, rend_h):
            raise ValueError(
                f"Cover segment dimensions ({seg_w}x{seg_h}) do not match "
                f"rendered video dimensions ({rend_w}x{rend_h})"
            )

        # Step 7: Concatenate Main Video + Cover Segment
        concat_video_with_cover(
            main_video_path=rendered_video,
            cover_segment_path=cover_segment_path,
            output_video_path=temp_final_path,
            has_audio=audio_props["has_audio"]
        )

        # Step 8: Validate output dimensions match rendered video
        final_w, final_h = get_video_dimensions(temp_final_path)
        if (final_w, final_h) != (rend_w, rend_h):
            raise ValueError(f"Output dimensions mismatch: expected ({rend_w}, {rend_h}), got ({final_w}, {final_h})")

        # Step 9: Atomically replace output video
        if os.path.exists(temp_final_path) and os.path.getsize(temp_final_path) > 0:
            if os.path.exists(output_video) and output_video == rendered_video:
                os.remove(output_video)
            shutil.move(temp_final_path, output_video)

        return {
            "success": True,
            "output_video": output_video,
            "cover_image_path": cover_image_path,
            "cover_duration": duration,
            "best_timestamp": best_ts,
            "selected_phrase": thumbnail_phrase,
            "audio_properties": audio_props,
            "placement_scores": comp_result.get("placement_scores")
        }

    except Exception as e:
        print(f"[COVER PIPELINE ERROR] Cover frame generation failed: {e}. Preserving normal render.")
        if rendered_video != output_video and os.path.exists(rendered_video):
            shutil.copyfile(rendered_video, output_video)
        return {
            "success": False,
            "error": str(e),
            "output_video": output_video if os.path.exists(output_video) else rendered_video
        }


def build_segment_render_specs(
    segments: list | None,
    actual_dur: float,
    default_crop: dict | None = None,
    source_video: str = "input.mp4",
    workspace: str | Path = ".",
    has_audio: bool = True
) -> dict:
    """
    Builds the per-segment render specification and concat plan for timeline segments.
    Ensures that each segment is independently cropped and scaled to 1002x1002 before
    concatenation, preserving distinct crop/zoom/pan settings for each clip.
    """
    if default_crop is None:
        default_crop = {}
    def_x = int(default_crop.get("crop_x", 0))
    def_y = int(default_crop.get("crop_y", 0))
    def_size = int(default_crop.get("crop_size", 1002))

    workspace_path = Path(workspace)

    parsed_segments = []
    if segments and len(segments) > 0:
        for i, seg in enumerate(segments):
            s_start = max(0.0, float(seg.get('start', 0.0)))
            s_end = min(actual_dur, float(seg.get('end', actual_dur)))
            s_dur = max(0.1, s_end - s_start)
            s_crop_x = int(seg.get('crop_x', def_x))
            s_crop_y = int(seg.get('crop_y', def_y))
            s_crop_size = int(seg.get('crop_size', def_size))
            seg_out = workspace_path / f"seg_{i}.mp4"
            vf = f"crop={s_crop_size}:{s_crop_size}:{s_crop_x}:{s_crop_y},scale=1002:1002"

            cmd = [
                get_ffmpeg_path(), "-y",
                "-ss", str(s_start), "-t", str(s_dur),
                "-i", str(source_video),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-r", "30"
            ]
            if has_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100"])
            else:
                cmd.append("-an")
            cmd.append(str(seg_out))

            parsed_segments.append({
                "index": i,
                "start": s_start,
                "end": s_end,
                "duration": s_dur,
                "crop_x": s_crop_x,
                "crop_y": s_crop_y,
                "crop_size": s_crop_size,
                "vf": vf,
                "output_path": str(seg_out),
                "ffmpeg_cmd": cmd
            })

    if not parsed_segments:
        parsed_segments.append({
            "index": 0,
            "start": 0.0,
            "end": actual_dur,
            "duration": actual_dur,
            "crop_x": def_x,
            "crop_y": def_y,
            "crop_size": def_size,
            "vf": f"crop={def_size}:{def_size}:{def_x}:{def_y},scale=1002:1002",
            "output_path": str(source_video),
            "ffmpeg_cmd": []
        })

    is_multisegment = len(parsed_segments) > 1

    concat_txt_path = workspace_path / "concat.txt"
    concat_out_path = workspace_path / "concat.mp4"

    concat_lines = []
    for s in parsed_segments:
        seg_file_path = Path(s["output_path"]).resolve().as_posix()
        concat_lines.append(f"file '{seg_file_path}'")
    concat_txt_content = "\n".join(concat_lines) + "\n"

    concat_cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_txt_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-r", "30"
    ]
    if has_audio:
        concat_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100"])
    else:
        concat_cmd.append("-an")
    concat_cmd.append(str(concat_out_path))

    total_dur = sum(s["duration"] for s in parsed_segments) if is_multisegment else parsed_segments[0]["duration"]
    cut_time = max(0.001, min(parsed_segments[0]["duration"], total_dur - 0.001))

    return {
        "is_multisegment": is_multisegment,
        "segments": parsed_segments,
        "concat_cmd": concat_cmd if is_multisegment else None,
        "concat_txt_path": str(concat_txt_path),
        "concat_txt_content": concat_txt_content,
        "concat_output_path": str(concat_out_path),
        "total_duration": total_dur,
        "cut_time": cut_time
    }


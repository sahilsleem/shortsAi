import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
import json
import shutil
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email import message_from_bytes

from src.config import CaptionData
from src.image_ops import generate_text_overlay, compute_best_font_size
from src.video_ops import render_main_video, render_curiosity_video, get_ffprobe_path, append_cover_frame
from src.gemini_caption import generate_captions, GeminiError, GeminiConfigError, GeminiQuotaError, GeminiAPIError

MAX_UPLOAD_SIZE = 100 * 1024 * 1024

def cleanup_workspace(workspace_dir: str):
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir, ignore_errors=True)

def get_duration(video_path: str) -> float:
    cmd = [get_ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        if duration <= 0:
            for stream in data.get("streams", []):
                if stream.get("duration"):
                    duration = float(stream["duration"])
                    break
        return duration
    except Exception:
        return 0.0

class ShortsAIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open('static/index.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path.startswith('/static/'):
            filepath = self.path.lstrip('/')
            if os.path.exists(filepath) and os.path.isfile(filepath):
                self.send_response(200)
                if filepath.endswith('.css'):
                    self.send_header('Content-Type', 'text/css')
                elif filepath.endswith('.js'):
                    self.send_header('Content-Type', 'application/javascript')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'File Not Found')
        elif self.path in ('/output_cover.jpg', '/cover', '/download_cover'):
            cover_path = Path("working/output_cover.jpg").resolve()
            if cover_path.exists() and cover_path.is_file():
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Disposition', 'attachment; filename="output_cover.jpg"')
                self.send_header('Content-Length', str(cover_path.stat().st_size))
                self.end_headers()
                with open(cover_path, 'rb') as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_error(404, "Cover image not found")
        else:
            self.send_error(404, 'File Not Found')

    def do_POST(self):
        if self.path == '/render':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > MAX_UPLOAD_SIZE:
                    self.send_error(413, "Video too large")
                    return
                
                body = self.rfile.read(content_length)
                
                msg_bytes = b""
                for k, v in self.headers.items():
                    msg_bytes += f"{k}: {v}\r\n".encode('latin-1')
                msg_bytes += b"\r\n" + body
                
                msg = message_from_bytes(msg_bytes)
                
                form_data = {}
                video_file = None
                
                if msg.is_multipart():
                    for part in msg.get_payload():
                        cd = part.get("Content-Disposition", "")
                        if 'name=' in cd:
                            name = None
                            filename = None
                            for param in cd.split(';'):
                                param = param.strip()
                                if param.startswith('name='):
                                    name = param[5:].strip('"')
                                elif param.startswith('filename='):
                                    filename = param[9:].strip('"')
                            
                            payload = part.get_payload(decode=True)
                            if filename and name == 'video':
                                video_file = payload
                            elif name:
                                form_data[name] = payload.decode('utf-8')
                
                if not video_file:
                    self.send_error(400, "No video file provided")
                    return
                
                mode = form_data.get('mode', 'main')
                start_time = float(form_data.get('start_time', 0))
                end_time = float(form_data.get('end_time', 0))
                
                cut_time_str = form_data.get('cut_time', '')
                cut_time = float(cut_time_str) if cut_time_str and cut_time_str != 'NaN' else None
                
                crop_x = int(form_data.get('crop_x', 0))
                crop_y = int(form_data.get('crop_y', 0))
                crop_size = int(form_data.get('crop_size', 1080))
                
                FONT_REGISTRY = {
                    "Calistoga": "fonts/Calistoga-Regular.ttf",
                    "Alike": "fonts/Alike-Regular.ttf",
                    "Caslon OS": "fonts/CaslonOS-Regular.otf",
                    "Fira Mono": "fonts/FiraMono-Regular.ttf",
                    "Source Sans Pro": "fonts/SourceSansPro-Regular.ttf"
                }

                caption_main = form_data.get('caption_main', '').replace('\r\n', '\n').replace('\r', '\n')
                font_main_key = form_data.get('font_main', 'Calistoga')
                font_main = str(Path(FONT_REGISTRY.get(font_main_key, "fonts/Calistoga-Regular.ttf")).resolve())
                
                caption_curiosity = form_data.get('caption_curiosity', '').replace('\r\n', '\n').replace('\r', '\n')
                font_curiosity_key = form_data.get('font_curiosity', 'Calistoga')
                font_curiosity = str(Path(FONT_REGISTRY.get(font_curiosity_key, "fonts/Calistoga-Regular.ttf")).resolve())
                
                caption_main_emoji = form_data.get('caption_main_emoji', '')
                caption_curiosity_emoji = form_data.get('caption_curiosity_emoji', '')
                
                enhance_str = form_data.get('enhance', 'false').lower()
                enhance = enhance_str in ('true', '1', 'yes', 'on')
                audio_mode = form_data.get('audio_mode', 'enhanced' if enhance else 'original').lower()
                if audio_mode not in ('original', 'enhanced', 'voice_focus'):
                    audio_mode = 'enhanced' if enhance else 'original'
                
                cover_enabled_str = form_data.get('cover_enabled', 'true').lower()
                cover_enabled = cover_enabled_str in ('true', '1', 'yes', 'on')
                thumbnail_phrase = form_data.get('thumbnail_phrase', '').strip()
                
                req_id = str(uuid.uuid4())
                workspace = Path(f"working/{req_id}").resolve()
                os.makedirs(workspace, exist_ok=True)
                
                input_path = workspace / "input.mp4"
                output_path = workspace / "output.mp4"
                
                try:
                    with open(input_path, "wb") as f:
                        f.write(video_file)
                        
                    actual_dur = get_duration(str(input_path))
                    if actual_dur <= 0:
                        raise Exception("Invalid video duration")
                        
                    start_time = max(0.0, min(start_time, actual_dur))
                    end_time = max(start_time + 0.5, min(end_time, actual_dur))
                            
                    if mode == "main":
                        cap = CaptionData(main_text=caption_main, curiosity_text="", emoji=caption_main_emoji, mode="main")
                        fs = compute_best_font_size(cap, font_main, 936, 3)
                        overlay_path = workspace / "overlay.png"
                        generate_text_overlay(cap, font_main, fs, str(overlay_path))
                        
                        render_main_video(
                            str(input_path), str(output_path), str(overlay_path),
                            start_time, end_time, crop_x, crop_y, crop_size,
                            enhance=enhance, audio_mode=audio_mode
                        )
                    else:
                        if "," in caption_main:
                            split_idx = caption_main.index(",") + 1
                            hook_black = caption_main[:split_idx].rstrip(' \t')
                            hook_red = caption_main[split_idx:].lstrip(' \t')
                        else:
                            hook_black = caption_main.rstrip(' \t')
                            hook_red = ""
                            
                        cap_hook = CaptionData(main_text=hook_black, curiosity_text=hook_red, emoji=caption_main_emoji)
                        cap_reveal = CaptionData(main_text=caption_curiosity, curiosity_text="", emoji=caption_curiosity_emoji)
                        
                        fs_hook = compute_best_font_size(cap_hook, font_main, 936, 3)
                        fs_reveal = compute_best_font_size(cap_reveal, font_curiosity, 936, 3)
                        
                        hook_overlay = workspace / "hook.png"
                        reveal_overlay = workspace / "reveal.png"
                        generate_text_overlay(cap_hook, font_main, fs_hook, str(hook_overlay))
                        generate_text_overlay(cap_reveal, font_curiosity, fs_reveal, str(reveal_overlay))
                        
                        if cut_time is None:
                            cut_time = start_time + (end_time - start_time) / 2.0
                        else:
                            cut_time = max(start_time + 0.001, min(cut_time, end_time - 0.001))
                        
                        render_curiosity_video(
                            str(input_path), str(output_path), str(hook_overlay), str(reveal_overlay),
                            start_time, cut_time, end_time, crop_x, crop_y, crop_size,
                            enhance=enhance, audio_mode=audio_mode
                        )
                        
                    # Phase 5: Optional Saba Bollywood Cover Frame appending
                    if cover_enabled and os.path.exists(output_path):
                        cover_img_path = str(workspace / "output_cover.jpg")
                        cover_res = append_cover_frame(
                            source_video=str(input_path),
                            rendered_video=str(output_path),
                            output_video=str(output_path),
                            thumbnail_phrase=thumbnail_phrase,
                            start_time=start_time,
                            end_time=end_time,
                            crop_x=crop_x,
                            crop_y=crop_y,
                            crop_size=crop_size,
                            mode=mode,
                            cut_time=cut_time,
                            enhance=enhance,
                            cover_image_path=cover_img_path
                        )
                        if cover_res.get("success") and os.path.exists(cover_img_path):
                            persistent_cover = Path("working/output_cover.jpg").resolve()
                            os.makedirs(persistent_cover.parent, exist_ok=True)
                            shutil.copyfile(cover_img_path, persistent_cover)
                    
                    if os.path.exists(output_path):
                        self.send_response(200)
                        self.send_header('Content-Type', 'video/mp4')
                        self.send_header('Content-Disposition', f'attachment; filename="shortsai_{mode}_{req_id[:8]}.mp4"')
                        file_size = os.path.getsize(output_path)
                        self.send_header('Content-Length', str(file_size))
                        self.end_headers()
                        with open(output_path, 'rb') as f:
                            shutil.copyfileobj(f, self.wfile)
                    else:
                        self.send_error(500, "Render output missing")
                        
                except Exception as e:
                    self.send_error(500, f"Render failed: {str(e)}")
                finally:
                    cleanup_workspace(str(workspace))
                    
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        elif self.path in ('/api/generate_captions', '/generate_captions'):
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                try:
                    data = json.loads(body.decode('utf-8'))
                except Exception:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode('utf-8'))
                    return
                
                context = data.get('context', '')
                previous_generations = data.get('previous_generations', [])
                
                try:
                    captions = generate_captions(context=context, previous_generations=previous_generations)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(captions).encode('utf-8'))
                except GeminiConfigError as e:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                except GeminiQuotaError as e:
                    self.send_response(429)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                except (GeminiAPIError, ValueError) as e:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": f"Failed to generate captions: {str(e)}"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Server error: {str(e)}"}).encode('utf-8'))
        else:
            self.send_error(404, 'File Not Found')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(('0.0.0.0', port), ShortsAIHandler)
    print(f"Starting ShortsAI server on 0.0.0.0:{port}...")
    server.serve_forever()

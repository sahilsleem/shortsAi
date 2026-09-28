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
from src.video_ops import render_main_video, render_curiosity_video, get_ffprobe_path

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
                
                caption_main = form_data.get('caption_main', '')
                caption_curiosity = form_data.get('caption_curiosity', '')
                caption_main_emoji = form_data.get('caption_main_emoji', '')
                caption_curiosity_emoji = form_data.get('caption_curiosity_emoji', '')
                
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
                            
                    font_path = str(Path("fonts/Calistoga-Regular.ttf").resolve())
                    
                    if mode == "main":
                        cap = CaptionData(main_text=caption_main, curiosity_text="", emoji=caption_main_emoji)
                        fs = compute_best_font_size([cap], font_path, 880, 3)
                        overlay_path = workspace / "overlay.png"
                        generate_text_overlay(cap, font_path, fs, str(overlay_path))
                        
                        render_main_video(
                            str(input_path), str(output_path), str(overlay_path),
                            start_time, end_time, crop_x, crop_y, crop_size
                        )
                    else:
                        if "," in caption_main:
                            split_idx = caption_main.index(",") + 1
                            hook_black = caption_main[:split_idx].strip()
                            hook_red = caption_main[split_idx:].strip()
                        else:
                            hook_black = caption_main.strip()
                            hook_red = ""
                            
                        cap_hook = CaptionData(main_text=hook_black, curiosity_text=hook_red, emoji=caption_main_emoji)
                        cap_reveal = CaptionData(main_text=caption_curiosity, curiosity_text="", emoji=caption_curiosity_emoji)
                        fs = compute_best_font_size([cap_hook, cap_reveal], font_path, 880, 3)
                        
                        hook_overlay = workspace / "hook.png"
                        reveal_overlay = workspace / "reveal.png"
                        generate_text_overlay(cap_hook, font_path, fs, str(hook_overlay))
                        generate_text_overlay(cap_reveal, font_path, fs, str(reveal_overlay))
                        
                        if cut_time is None:
                            cut_time = start_time + (end_time - start_time) / 2.0
                        else:
                            cut_time = max(start_time + 0.001, min(cut_time, end_time - 0.001))
                        
                        render_curiosity_video(
                            str(input_path), str(output_path), str(hook_overlay), str(reveal_overlay),
                            start_time, cut_time, end_time, crop_x, crop_y, crop_size
                        )
                        
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
        else:
            self.send_error(404, 'File Not Found')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(('0.0.0.0', port), ShortsAIHandler)
    print(f"Starting ShortsAI server on 0.0.0.0:{port}...")
    server.serve_forever()

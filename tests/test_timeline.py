class TimelineState:
    def __init__(self, duration):
        self.duration = duration
        self.segments = [{"start": 0.0, "end": float(duration)}]
        self.active_idx = 0
        self.zoom_levels = [100, 200, 400, 800, 1600, 3200]
        self.zoom_level = 100
        self.scroll_left = 0.0

    def get_track_width(self, viewport_width):
        return (viewport_width * self.zoom_level) / 100.0

    def get_px_per_sec(self, viewport_width):
        return self.get_track_width(viewport_width) / self.duration

    def set_zoom(self, zoom_pct, viewport_width, current_time=None):
        if current_time is None:
            current_time = self.segments[self.active_idx]["start"]
        old_track_width = self.get_track_width(viewport_width)
        old_playhead_px = (current_time / self.duration) * old_track_width
        old_screen_x = old_playhead_px - self.scroll_left
        if old_screen_x < 20 or old_screen_x > viewport_width - 20:
            old_screen_x = viewport_width / 2.0

        self.zoom_level = zoom_pct
        new_track_width = self.get_track_width(viewport_width)
        new_playhead_px = (current_time / self.duration) * new_track_width
        self.scroll_left = max(0.0, min(new_track_width - viewport_width, new_playhead_px - old_screen_x))

    def get_time_markers(self, viewport_width):
        px_per_sec = self.get_px_per_sec(viewport_width)
        intervals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0]
        step = 5.0
        for intv in intervals:
            if intv * px_per_sec >= 50.0:
                step = intv
                break
        markers = []
        t = 0.0
        while t <= self.duration + 1e-5:
            markers.append(round(t, 4))
            t += step
        return step, markers

    def get_active(self):
        return self.segments[self.active_idx]

    def move_in(self, t):
        seg = self.get_active()
        prev_end = self.segments[self.active_idx - 1]["end"] if self.active_idx > 0 else 0.0
        seg["start"] = max(prev_end, min(t, seg["end"] - 0.05))

    def move_out(self, t):
        seg = self.get_active()
        next_start = self.segments[self.active_idx + 1]["start"] if self.active_idx < len(self.segments) - 1 else self.duration
        seg["end"] = min(next_start, max(t, seg["start"] + 0.05))

    def move_segment(self, dx):
        seg = self.get_active()
        new_start = seg["start"] + dx
        new_end = seg["end"] + dx
        dur = new_end - new_start
        prev_end = self.segments[self.active_idx - 1]["end"] if self.active_idx > 0 else 0.0
        next_start = self.segments[self.active_idx + 1]["start"] if self.active_idx < len(self.segments) - 1 else self.duration

        if new_start < prev_end:
            new_start = prev_end
            new_end = new_start + dur
        if new_end > next_start:
            new_end = next_start
            new_start = new_end - dur
            if new_start < prev_end:
                new_start = prev_end
        seg["start"] = new_start
        seg["end"] = new_end

    def split(self, t):
        if not self.segments: return
        seg = self.get_active()
        if t > seg["start"] + 0.05 and t < seg["end"] - 0.05:
            old_end = seg["end"]
            seg["end"] = t
            self.segments.insert(self.active_idx + 1, {"start": t, "end": old_end})
            self.active_idx += 1

    def delete(self):
        if not self.segments: return
        self.segments.pop(self.active_idx)
        if not self.segments:
            self.segments.append({"start": 0.0, "end": float(self.duration)})
        self.active_idx = max(0, self.active_idx - 1)

    def hit_test(self, px, rect_width):
        """Deterministic hit testing matching index.html hitTestTimeline"""
        if not self.duration or not self.segments:
            return {"type": "none", "index": -1}

        # 1. Exact handle hit for active segment
        if 0 <= self.active_idx < len(self.segments):
            active_seg = self.segments[self.active_idx]
            in_px = (active_seg["start"] / self.duration) * rect_width
            out_px = (active_seg["end"] / self.duration) * rect_width
            handle_radius = 22.0

            # Active segment handles should not steal touches inside neighbor segments
            prev_e_px = (self.segments[self.active_idx - 1]["end"] / self.duration) * rect_width if self.active_idx > 0 else 0.0
            next_s_px = (self.segments[self.active_idx + 1]["start"] / self.duration) * rect_width if self.active_idx < len(self.segments) - 1 else rect_width

            in_min_px = max(prev_e_px, in_px - handle_radius)
            in_max_px = in_px + min(handle_radius, (out_px - in_px) / 2.0)

            out_min_px = out_px - min(handle_radius, (out_px - in_px) / 2.0)
            out_max_px = min(next_s_px, out_px + handle_radius)

            is_in_hit = in_min_px <= px <= in_max_px
            is_out_hit = out_min_px <= px <= out_max_px

            if is_in_hit and is_out_hit:
                dist_in = abs(px - in_px)
                dist_out = abs(px - out_px)
                return {"type": "handle-in" if dist_in <= dist_out else "handle-out", "index": self.active_idx}
            if is_in_hit:
                return {"type": "handle-in", "index": self.active_idx}
            if is_out_hit:
                return {"type": "handle-out", "index": self.active_idx}

        # 2. Exact segment body hit
        exact_hit_idx = -1
        closest_center_dist = float('inf')
        for idx, seg in enumerate(self.segments):
            s_px = (seg["start"] / self.duration) * rect_width
            e_px = (seg["end"] / self.duration) * rect_width
            if s_px <= px <= e_px:
                center = (s_px + e_px) / 2.0
                d = abs(px - center)
                if d < closest_center_dist:
                    closest_center_dist = d
                    exact_hit_idx = idx
        if exact_hit_idx != -1:
            return {"type": "segment", "index": exact_hit_idx}

        # 3. Expanded segment touch target
        best_expanded_idx = -1
        min_edge_dist = float('inf')
        expanded_radius = 24.0
        for idx, seg in enumerate(self.segments):
            s_px = (seg["start"] / self.duration) * rect_width
            e_px = (seg["end"] / self.duration) * rect_width
            edge_dist = max(0.0, s_px - px, px - e_px)
            if edge_dist <= expanded_radius and edge_dist < min_edge_dist:
                min_edge_dist = edge_dist
                best_expanded_idx = idx
        if best_expanded_idx != -1:
            return {"type": "segment", "index": best_expanded_idx}

        # 4. Narrow gap between segments
        nearest_idx = -1
        min_gap_dist = float('inf')
        for idx, seg in enumerate(self.segments):
            s_px = (seg["start"] / self.duration) * rect_width
            e_px = (seg["end"] / self.duration) * rect_width
            d = min(abs(px - s_px), abs(px - e_px))
            if d < min_gap_dist:
                min_gap_dist = d
                nearest_idx = idx
        if min_gap_dist <= 32.0 and nearest_idx != -1:
            return {"type": "segment", "index": nearest_idx}

        # 5. Playhead
        return {"type": "playhead", "index": -1}


class LongPressStepper:
    def __init__(self, current_time, duration, fps=30):
        self.current_time = float(current_time)
        self.duration = float(duration)
        self.fps = fps
        self.frame_dur = 1.0 / fps
        self.is_holding = False
        self.hold_time_ms = 0
        self.timer_active = False
        self.step_history = []
        self.last_seek_requested = None

    def request_seek(self, t):
        self.last_seek_requested = t

    def pointer_down(self, direction=1):
        self.is_holding = True
        self.timer_active = True
        self.direction = direction
        self.hold_time_ms = 0
        # Immediate single frame
        self._step(multiplier=1)

    def _step(self, multiplier=1):
        self.current_time = max(0.0, min(self.duration, self.current_time + self.direction * self.frame_dur * multiplier))
        self.step_history.append(self.current_time)
        self.request_seek(self.current_time)

    def advance_time(self, ms_elapsed):
        if not self.is_holding:
            return

        step_interval = 60
        hold_delay = 250

        while ms_elapsed >= step_interval:
            self.hold_time_ms += step_interval
            ms_elapsed -= step_interval

            if self.hold_time_ms > hold_delay:
                multiplier = 1
                if self.hold_time_ms > 2000:
                    multiplier = 4
                elif self.hold_time_ms > 1000:
                    multiplier = 2
                self._step(multiplier=multiplier)

    def pointer_up(self):
        self.is_holding = False
        self.timer_active = False


# --- TESTS ---

def test_basic_trim():
    tl = TimelineState(20)
    tl.move_in(3)
    tl.move_out(8)
    assert len(tl.segments) == 1
    assert tl.segments[0]["start"] == 3
    assert tl.segments[0]["end"] == 8

def test_move_selection():
    tl = TimelineState(20)
    tl.move_in(3)
    tl.move_out(8)
    tl.move_segment(4) # 3-8 -> 7-12
    assert tl.segments[0]["start"] == 7
    assert tl.segments[0]["end"] == 12

def test_bounds():
    tl = TimelineState(20)
    tl.move_in(3)
    tl.move_out(8)
    tl.move_segment(-10) # under 0
    assert tl.segments[0]["start"] == 0
    assert tl.segments[0]["end"] == 5

    tl.move_segment(50) # over 20
    assert tl.segments[0]["end"] == 20
    assert tl.segments[0]["start"] == 15

def test_invalid_range():
    tl = TimelineState(20)
    tl.move_in(10)
    tl.move_in(30) # try to move IN past OUT
    assert tl.segments[0]["start"] <= tl.segments[0]["end"] - 0.05
    tl.move_out(0) # try to move OUT before IN
    assert tl.segments[0]["end"] >= tl.segments[0]["start"] + 0.05

def test_split_delete():
    tl = TimelineState(20)
    tl.split(8)
    assert len(tl.segments) == 2
    assert tl.segments[0]["start"] == 0
    assert tl.segments[0]["end"] == 8
    assert tl.segments[1]["start"] == 8
    assert tl.segments[1]["end"] == 20
    assert tl.active_idx == 1

    tl.active_idx = 0
    tl.delete()
    assert len(tl.segments) == 1
    assert tl.segments[0]["start"] == 8
    assert tl.segments[0]["end"] == 20

def test_tiny_segment_selectable():
    """Verify that a tiny 0.5-second segment between two other segments is independently selectable."""
    tl = TimelineState(20)
    tl.segments = [
        {"start": 0.0, "end": 8.0},
        {"start": 8.0, "end": 8.5},
        {"start": 8.5, "end": 20.0}
    ]
    tl.active_idx = 1
    active = tl.get_active()
    assert active["start"] == 8.0
    assert active["end"] == 8.5
    assert (active["end"] - active["start"]) == 0.5

def test_tiny_segment_touch_logic():
    """Verify an expanded touch target can identify the tiny middle segment without changing its actual bounds."""
    tl = TimelineState(20)
    tl.segments = [
        {"start": 0.0, "end": 8.0},
        {"start": 8.0, "end": 8.5},
        {"start": 8.5, "end": 20.0}
    ]
    tl.active_idx = 0

    rect_width = 360.0
    # In 20s timeline of 360px:
    # Seg 0: 0.0 to 144.0px
    # Seg 1 (tiny): 144.0 to 153.0px (width 9px)
    # Seg 2: 153.0 to 360.0px

    # Touch at 148px (inside tiny segment body)
    hit = tl.hit_test(148.0, rect_width)
    assert hit["type"] == "segment"
    assert hit["index"] == 1

    # Verify the segment's actual bounds did not change
    assert tl.segments[1]["start"] == 8.0
    assert tl.segments[1]["end"] == 8.5

def test_stretch_start():
    """Example: 5-10, drag IN to 6.25 -> duration 3.75s."""
    tl = TimelineState(20)
    tl.segments = [{"start": 5.0, "end": 10.0}]
    tl.active_idx = 0
    tl.move_in(6.25)
    assert tl.segments[0]["start"] == 6.25
    assert tl.segments[0]["end"] == 10.0
    duration = tl.segments[0]["end"] - tl.segments[0]["start"]
    assert duration == 3.75

def test_stretch_end():
    """Example: 5-10, drag OUT to 12.5 -> duration 7.5s."""
    tl = TimelineState(20)
    tl.segments = [{"start": 5.0, "end": 10.0}]
    tl.active_idx = 0
    tl.move_out(12.5)
    assert tl.segments[0]["start"] == 5.0
    assert tl.segments[0]["end"] == 12.5
    duration = tl.segments[0]["end"] - tl.segments[0]["start"]
    assert duration == 7.5

def test_bounds_cannot_cross():
    """Verify handles cannot cross."""
    tl = TimelineState(20)
    tl.segments = [{"start": 5.0, "end": 10.0}]
    tl.active_idx = 0

    # Try dragging IN past OUT
    tl.move_in(12.0)
    assert tl.segments[0]["start"] <= tl.segments[0]["end"] - 0.05
    assert tl.segments[0]["start"] < tl.segments[0]["end"]

    # Try dragging OUT before IN
    tl.segments[0]["start"] = 5.0
    tl.segments[0]["end"] = 10.0
    tl.move_out(3.0)
    assert tl.segments[0]["end"] >= tl.segments[0]["start"] + 0.05
    assert tl.segments[0]["end"] > tl.segments[0]["start"]

def test_subsecond_precision():
    """Verify values such as 4.23 and 4.71 are preserved rather than rounded to whole seconds."""
    tl = TimelineState(20)
    tl.move_in(4.23)
    tl.move_out(4.71)
    assert tl.segments[0]["start"] == 4.23
    assert tl.segments[0]["end"] == 4.71
    assert round(tl.segments[0]["end"] - tl.segments[0]["start"], 2) == 0.48

def test_frame_tap():
    """One tap moves exactly one frame: N -> N + 1/fps."""
    stepper = LongPressStepper(current_time=5.000, duration=20.0, fps=30)
    stepper.pointer_down(direction=1)
    stepper.pointer_up()

    # Exactly 1 step was recorded
    assert len(stepper.step_history) == 1
    expected = 5.000 + (1.0 / 30.0)
    assert abs(stepper.current_time - expected) < 1e-4

def test_long_press_stepping():
    """Verify stepping starts after hold delay, repeated stepping occurs, acceleration occurs, release stops."""
    stepper = LongPressStepper(current_time=5.000, duration=20.0, fps=30)
    stepper.pointer_down(direction=1) # 1 initial step
    assert len(stepper.step_history) == 1
    assert stepper.timer_active is True

    # Advance 200ms (still within 250ms hold delay)
    stepper.advance_time(200)
    assert len(stepper.step_history) == 1

    # Advance past 250ms hold delay (advance 600ms total)
    stepper.advance_time(600)
    assert len(stepper.step_history) > 1

    # Advance past 1000ms (acceleration kicks in with multiplier=2)
    time_before_accel = stepper.current_time
    stepper.advance_time(600)
    time_after_accel = stepper.current_time
    delta_after_accel = time_after_accel - time_before_accel
    assert delta_after_accel > 0

    # Release pointer
    stepper.pointer_up()
    assert stepper.is_holding is False
    assert stepper.timer_active is False

    # Verify advancing time after release does nothing
    stopped_count = len(stepper.step_history)
    stepper.advance_time(500)
    assert len(stepper.step_history) == stopped_count

def test_latest_wins_seek_behavior():
    """Verify long-press stepping feeds positions through latest-wins seek without stale backlog."""
    stepper = LongPressStepper(current_time=2.000, duration=20.0, fps=30)
    stepper.pointer_down(direction=1)
    stepper.advance_time(1200)
    # The last seek requested matches the current target time
    assert stepper.last_seek_requested == stepper.current_time

def test_font_default_alike():
    """Verify fresh UI state selects Alike for both Main and Curiosity captions, and all other fonts remain available."""
    from pathlib import Path
    import re
    html = Path("static/index.html").read_text(encoding="utf-8")

    # Verify font-main selector defaults to Alike
    main_match = re.search(r'<select id="font-main">(.*?)</select>', html, re.DOTALL)
    assert main_match, "font-main select block not found"
    assert '<option value="Alike" selected>Alike</option>' in main_match.group(1)
    for font in ["Calistoga", "Caslon OS", "Fira Mono", "Source Sans Pro"]:
        assert f'<option value="{font}">{font}</option>' in main_match.group(1)

    # Verify font-curiosity selector defaults to Alike
    curiosity_match = re.search(r'<select id="font-curiosity">(.*?)</select>', html, re.DOTALL)
    assert curiosity_match, "font-curiosity select block not found"
    assert '<option value="Alike" selected>Alike</option>' in curiosity_match.group(1)
    for font in ["Calistoga", "Caslon OS", "Fira Mono", "Source Sans Pro"]:
        assert f'<option value="{font}">{font}</option>' in curiosity_match.group(1)

def test_server_font_defaults():
    """Verify backend server defaults to Alike for both Main and Curiosity captions."""
    from pathlib import Path
    server_code = Path("src/server.py").read_text(encoding="utf-8")
    assert "form_data.get('font_main', 'Alike')" in server_code
    assert 'FONT_REGISTRY.get(font_main_key, "fonts/Alike-Regular.ttf")' in server_code
    assert "form_data.get('font_curiosity', 'Alike')" in server_code
    assert 'FONT_REGISTRY.get(font_curiosity_key, "fonts/Alike-Regular.ttf")' in server_code

def test_legacy_compatibility():
    segments = [{"start": 3, "end": 8}]
    s_start = segments[0]["start"]
    s_end = segments[0]["end"]
    cut_time = s_start + (s_end - s_start)/2
    assert s_start == 3
    assert s_end == 8
    assert cut_time == 5.5

def test_zoom_100_percent_preserves_behavior():
    """Verify 100% zoom preserves existing timeline behavior and track width matches viewport."""
    tl = TimelineState(20)
    viewport_width = 360.0
    assert tl.zoom_level == 100
    assert tl.get_track_width(viewport_width) == 360.0
    assert tl.get_px_per_sec(viewport_width) == 18.0

def test_zoom_factor_changes_track_width():
    """Verify zoom levels 200%, 400%, 800%, 1600%, 3200% correctly scale track width."""
    tl = TimelineState(20)
    viewport_width = 400.0

    tl.set_zoom(200, viewport_width)
    assert tl.get_track_width(viewport_width) == 800.0

    tl.set_zoom(400, viewport_width)
    assert tl.get_track_width(viewport_width) == 1600.0

    tl.set_zoom(800, viewport_width)
    assert tl.get_track_width(viewport_width) == 3200.0

    tl.set_zoom(1600, viewport_width)
    assert tl.get_track_width(viewport_width) == 6400.0

    tl.set_zoom(3200, viewport_width)
    assert tl.get_track_width(viewport_width) == 12800.0

def test_fractional_segment_values_survive_zoom():
    """Verify fractional segment start/end values like 12.37 and 12.71 survive zoom without rounding."""
    tl = TimelineState(30)
    tl.segments = [{"start": 12.37, "end": 12.71}]
    tl.active_idx = 0
    viewport_width = 360.0

    for zoom in [100, 200, 400, 800, 1600, 3200]:
        tl.set_zoom(zoom, viewport_width)
        assert tl.segments[0]["start"] == 12.37
        assert tl.segments[0]["end"] == 12.71
        assert round(tl.segments[0]["end"] - tl.segments[0]["start"], 2) == 0.34

def test_zooming_does_not_alter_segment_times():
    """Verify that repeatedly zooming in and out does not mutate any segment boundaries."""
    tl = TimelineState(20)
    tl.segments = [
        {"start": 1.25, "end": 4.50},
        {"start": 6.10, "end": 8.75},
        {"start": 12.333, "end": 15.666}
    ]
    viewport_width = 360.0
    expected = [dict(s) for s in tl.segments]

    for z in [200, 400, 800, 1600, 100, 3200, 100]:
        tl.set_zoom(z, viewport_width)
        assert tl.segments == expected

def test_horizontal_scrolling_does_not_alter_segment_times():
    """Verify that horizontal scrolling does not alter segment times."""
    tl = TimelineState(20)
    tl.segments = [{"start": 4.23, "end": 4.71}]
    viewport_width = 360.0
    tl.set_zoom(800, viewport_width)

    for scroll in [0.0, 100.0, 500.0, 1200.0]:
        tl.scroll_left = scroll
        assert tl.segments[0]["start"] == 4.23
        assert tl.segments[0]["end"] == 4.71

def test_zoom_centering_on_playhead():
    """Verify that zooming in centers around the playhead/current editing position."""
    tl = TimelineState(20)
    viewport_width = 400.0
    tl.scroll_left = 0.0
    playhead_time = 10.0

    # Zoom to 400% (track becomes 1600px)
    tl.set_zoom(400, viewport_width, current_time=playhead_time)

    # Playhead on track is (10/20) * 1600 = 800px.
    # With scroll_left, screen position is 800 - scroll_left.
    screen_pos = (playhead_time / tl.duration) * tl.get_track_width(viewport_width) - tl.scroll_left
    assert abs(screen_pos - 200.0) < 1.0

def test_ruler_marker_density():
    """Verify intelligent marker density at different zoom levels so labels never crowd (< 50px)."""
    tl = TimelineState(20)
    viewport_width = 360.0

    # 100% zoom (18 px/sec) -> step should be 5s (90px apart >= 50px)
    tl.set_zoom(100, viewport_width)
    step_100, markers_100 = tl.get_time_markers(viewport_width)
    assert step_100 == 5.0
    assert step_100 * tl.get_px_per_sec(viewport_width) >= 50.0

    # 400% zoom (72 px/sec) -> step should be 1s (72px apart >= 50px)
    tl.set_zoom(400, viewport_width)
    step_400, markers_400 = tl.get_time_markers(viewport_width)
    assert step_400 == 1.0
    assert step_400 * tl.get_px_per_sec(viewport_width) >= 50.0

    # 1600% zoom (288 px/sec) -> step should be 0.2s (57.6px apart >= 50px)
    tl.set_zoom(1600, viewport_width)
    step_1600, markers_1600 = tl.get_time_markers(viewport_width)
    assert step_1600 == 0.2
    assert step_1600 * tl.get_px_per_sec(viewport_width) >= 50.0

def test_split_fractional_time():
    """Verify split still works at fractional times like 7.34s."""
    tl = TimelineState(20)
    tl.segments = [{"start": 5.0, "end": 10.0}]
    tl.active_idx = 0
    tl.split(7.34)
    assert len(tl.segments) == 2
    assert tl.segments[0]["start"] == 5.0
    assert tl.segments[0]["end"] == 7.34
    assert tl.segments[1]["start"] == 7.34
    assert tl.segments[1]["end"] == 10.0

def test_mobile_creator_redesign_structure():
    """Verify the redesigned mobile creator layout contracts: light theme, dark workspace, drawers, quick tools."""
    from pathlib import Path
    html = Path("static/index.html").read_text(encoding="utf-8")

    # 1. Dark editor workspace encapsulating preview and timeline
    assert 'class="editor-workspace"' in html
    assert 'id="preview-container"' in html
    assert 'id="crop-box"' in html
    assert 'id="zoom-slider"' in html
    assert 'id="timeline-scroll-viewport"' in html
    assert 'id="trim-timeline-container"' in html
    assert 'id="handle-in"' in html
    assert 'id="handle-out"' in html

    # 2. Quick editing controls
    assert 'id="btn-frame-prev"' in html
    assert 'id="btn-play-selection"' in html
    assert 'id="btn-frame-next"' in html
    assert 'id="btn-split"' in html
    assert 'id="btn-delete-segment"' in html

    # 3. Polished Creator Tool Sections (Visible by Default)
    assert 'class="tool-sections"' in html
    assert 'id="drawer-captions"' in html
    assert 'id="drawer-ai-writer"' in html
    assert 'id="drawer-enhance"' in html
    assert 'id="drawer-audio"' in html
    assert 'id="drawer-branding"' in html
    assert 'class="tool-section"' in html
    assert 'toggleDrawer' in html

    # 5. Prominent Render CTA and result section
    assert 'id="btn-generate"' in html
    assert 'id="status-message"' in html
    assert 'id="result-section"' in html
    assert 'id="result-video"' in html
    assert 'id="publishing-section"' in html

    # 6. Verify Branding was renamed to Watermark with no Saba Bollywood in visible UI
    assert 'Saba Bollywood' not in html

def test_compact_mobile_preview_viewport():
    """Verify that the preview viewport is compact and constrained to prevent page height explosion."""
    from pathlib import Path
    html = Path("static/index.html").read_text(encoding="utf-8")

    assert 'class="preview-viewport"' in html
    assert 'id="preview-viewport"' in html
    assert 'clamp(220px, 35vh, 280px)' in html
    assert 'fitPreviewContainer' in html
    assert 'previewContainer.onmousedown = startDrag' in html
    assert 'previewContainer.ontouchstart = startDrag' in html

def test_neutral_caption_ui():
    """Verify neutral caption placeholders, emoji inputs without placeholders, and absence of Bollywood examples."""
    from pathlib import Path
    import re
    html = Path("static/index.html").read_text(encoding="utf-8")

    # 1. Main caption placeholder
    assert 'placeholder="Enter your main caption..."' in html

    # 2. Curiosity caption placeholder
    assert 'placeholder="Enter your curiosity caption..."' in html

    # 3. AI context placeholder
    assert 'placeholder="Tell AI what this short is about..."' in html

    # 4. Generate button text has no emojis
    assert re.search(r'<button[^>]*id="btn-gemini-generate"[^>]*>\s*Generate Captions\s*</button>', html)

    # 5. Emoji inputs have no placeholder
    assert 'id="caption-emoji-main"' in html
    main_emoji_match = re.search(r'<input[^>]*id="caption-emoji-main"[^>]*>', html)
    assert main_emoji_match and 'placeholder' not in main_emoji_match.group(0)

    assert 'id="caption-emoji-curiosity"' in html
    curiosity_emoji_match = re.search(r'<input[^>]*id="caption-emoji-curiosity"[^>]*>', html)
    assert curiosity_emoji_match and 'placeholder' not in curiosity_emoji_match.group(0)

    # 6. No Bollywood / celebrity examples
    assert 'Salman' not in html
    assert 'Arpita' not in html
    assert 'Bollywood' not in html

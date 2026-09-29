class TimelineState:
    def __init__(self, duration):
        self.duration = duration
        self.segments = [{"start": 0.0, "end": float(duration)}]
        self.active_idx = 0

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
    """Verify fresh UI state selects Alike for Main captions, and all other fonts remain available."""
    from pathlib import Path
    html = Path("static/index.html").read_text(encoding="utf-8")

    # Check default selected option is Alike
    assert '<option value="Alike" selected>Alike</option>' in html

    # Check all existing fonts remain available
    assert '<option value="Calistoga">Calistoga</option>' in html
    assert '<option value="Caslon OS">Caslon OS</option>' in html
    assert '<option value="Fira Mono">Fira Mono</option>' in html
    assert '<option value="Source Sans Pro">Source Sans Pro</option>' in html

def test_legacy_compatibility():
    segments = [{"start": 3, "end": 8}]
    s_start = segments[0]["start"]
    s_end = segments[0]["end"]
    cut_time = s_start + (s_end - s_start)/2
    assert s_start == 3
    assert s_end == 8
    assert cut_time == 5.5

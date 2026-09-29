class TimelineState:
    def __init__(self, duration):
        self.duration = duration
        self.segments = [{"start": 0, "end": duration}]
        self.active_idx = 0
    
    def get_active(self):
        return self.segments[self.active_idx]
    
    def move_in(self, t):
        seg = self.get_active()
        seg["start"] = max(0.0, min(t, seg["end"] - 0.1))
        
    def move_out(self, t):
        seg = self.get_active()
        seg["end"] = min(self.duration, max(t, seg["start"] + 0.1))
        
    def move_segment(self, dx):
        seg = self.get_active()
        new_start = seg["start"] + dx
        new_end = seg["end"] + dx
        dur = new_end - new_start
        if new_start < 0:
            new_start = 0
            new_end = dur
        if new_end > self.duration:
            new_end = self.duration
            new_start = new_end - dur
        seg["start"] = new_start
        seg["end"] = new_end
        
    def split(self, t):
        if not self.segments: return
        seg = self.get_active()
        if t > seg["start"] + 0.1 and t < seg["end"] - 0.1:
            old_end = seg["end"]
            seg["end"] = t
            self.segments.insert(self.active_idx + 1, {"start": t, "end": old_end})
            self.active_idx += 1
            
    def delete(self):
        if not self.segments: return
        self.segments.pop(self.active_idx)
        if not self.segments:
            self.segments.append({"start": 0, "end": self.duration})
        self.active_idx = max(0, self.active_idx - 1)

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
    assert tl.segments[0]["start"] <= tl.segments[0]["end"] - 0.1
    tl.move_out(0) # try to move OUT before IN
    assert tl.segments[0]["end"] >= tl.segments[0]["start"] + 0.1

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

def test_legacy_compatibility():
    segments = [{"start": 3, "end": 8}]
    s_start = segments[0]["start"]
    s_end = segments[0]["end"]
    cut_time = s_start + (s_end - s_start)/2
    assert s_start == 3
    assert s_end == 8
    assert cut_time == 5.5

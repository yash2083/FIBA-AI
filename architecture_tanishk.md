# 🏗️ Architecture — Tanishk
## Module: Tracker + Motion Engine + Action Inferencer + Segmentor
### FIBA AI | MIT Bangalore Hitachi Hackathon

---

## Your Role in the Pipeline

Tanishk owns **Stage 4, 5, and 6** — the analytical brain of FIBA AI. You receive bboxes from Atul and produce the action verdict with evidence that goes to Yash's UI.

```
FROM ATUL:
  hand_bbox, object_bbox per frame
         │
         ▼
[ByteTrack Tracker] ──► stable object trajectory across frames
         │
         ▼
[Motion Engine] ──► motion features (displacement, rotation, area_change, fragmentation)
         │
         ▼
[Action Inferencer] ──► action_label, confidence, evidence_trace
         │
         ▼
[MobileSAM Segmentor] ──► object mask on key frames (optional, on-demand)
         │
         ▼
TO YASH: {action, timestamp_range, key_frames, trajectory, mask, explanation}
```

---

## Stage 4: Tracker (`pipeline/tracker.py`)

### What it does
Keeps the query-relevant object tracked stably across frames, even through occlusion or momentary detection failure.

### Architecture Decision: **Simple IoU + Kalman Tracker (ByteTrack-inspired)**
For a hackathon, implement a lightweight version:
- **Kalman Filter** predicts object position when detection is missed
- **IoU matching** associates new detections to existing tracks
- Re-detect with YOLOv8 every 5 frames or when tracking confidence < 0.4
- ByteTrack paper's key innovation: keep both high+low score detections — implement this

### State maintained per track
```python
class Track:
    track_id: int
    bbox_history: List[[x1,y1,x2,y2]]   # all bboxes over time
    center_history: List[[cx,cy]]         # centroid trajectory
    area_history: List[float]             # bbox area over time
    confidence_history: List[float]
    frames_since_update: int
    state: "active" | "lost" | "dead"
```

### Tracking Parameters
```python
MAX_LOST_FRAMES = 10        # lose track after 10 missed frames
IOU_THRESHOLD = 0.3         # minimum IoU to associate
REDETECT_INTERVAL = 5       # re-run YOLO every 5 frames
HIGH_SCORE_THRESHOLD = 0.6  # ByteTrack high confidence
LOW_SCORE_THRESHOLD = 0.2   # ByteTrack low confidence (still keep)
```

### Output per frame
```python
{
  "track_id": int,
  "bbox": [x1, y1, x2, y2],
  "center": [cx, cy],
  "area": float,
  "tracking_confidence": float,
  "trajectory": [[cx,cy], ...],   # full history
}
```

---

## Stage 5: Motion Engine (`pipeline/motion_engine.py`)

### What it does
This is the **core novelty** of FIBA AI. Instead of a heavy action classifier, extract interpretable motion features from the tracked object's history and hand-object relationship.

### Motion Features to Compute

#### 5.1 Translational Motion
```python
# Displacement vector over sliding window (last N frames)
displacement = center[-1] - center[-N]
displacement_magnitude = ||displacement||
displacement_direction = atan2(dy, dx)   # in degrees
vertical_motion = dy / frame_height      # normalized

# Is the object moving toward/away from body?
motion_speed = displacement_magnitude / N  # pixels per frame
```

#### 5.2 Rotational Proxy
```python
# Use oriented bounding box (via cv2.minAreaRect on mask or contour)
# OR use hand-to-object vector rotation
angle_history = [get_obb_angle(bbox) for bbox in bbox_history]
angle_change = angle_history[-1] - angle_history[-N]
rotation_speed = abs(angle_change) / N     # degrees/frame

# High rotation → OPEN/CLOSE/UNSCREW action
```

#### 5.3 Area Change (Scale)
```python
area_history = track.area_history
area_ratio = area[-1] / area[-N]          # >1 = growing, <1 = shrinking
area_velocity = (area[-1] - area[-N]) / N
area_variance = std(area_history[-N:])     # high variance = fragmentation
```

#### 5.4 Hand-Object Interaction
```python
# Distance from hand wrist to object center
contact_distance = euclidean(hand_wrist, object_center)
normalized_contact = contact_distance / frame_diagonal

# Repetitive contact pattern (cut signature)
contact_history = [contact_distance < CONTACT_THRESHOLD for each frame]
contact_frequency = count_oscillations(contact_history)  # low freq = one action, high freq = repeated (cutting)
```

#### 5.5 Object State Change Score
```python
# Compare first-N frames vs last-N frames
early_state = features(frames[:10])
late_state = features(frames[-10:])
state_change_score = ||early_state - late_state||
```

### Motion Feature Bundle (output)
```python
{
  "displacement_magnitude": float,    # pixels
  "displacement_direction": float,    # degrees
  "vertical_motion_ratio": float,     # -1 (down) to +1 (up)
  "rotation_change": float,           # degrees
  "rotation_speed": float,            # deg/frame
  "area_ratio": float,                # end_area / start_area
  "area_variance": float,             # fragmentation indicator
  "contact_distance_mean": float,     # hand-object closeness
  "contact_frequency": float,         # oscillation count (cutting signature)
  "state_change_score": float,        # 0-1, how much object changed
}
```

---

## Stage 6: Action Inferencer (`pipeline/action_inferencer.py`)

### What it does
Maps motion features to action categories using rule-based logic. Returns a human-readable explanation.

### Rules Engine (the key innovation — explainable AI!)

```python
def infer_action(motion_features, action_verb_hint):
    f = motion_features
    
    if action_verb_hint in ["CUT", "CHOP", "SLICE"]:
        score = (
            0.4 * normalize(f.contact_frequency, 0, 5) +    # repeated contact
            0.3 * normalize(f.area_variance, 0, 500) +       # fragmentation
            0.3 * (1 - normalize(f.displacement_magnitude, 0, 200))  # stays in place
        )
        evidence = f"Object area fragmented {f.area_variance:.0f}px²; tool-object contact repeated {f.contact_frequency:.1f}x"
    
    elif action_verb_hint in ["OPEN", "UNSCREW"]:
        score = (
            0.5 * normalize(f.rotation_change, 0, 90) +     # rotation detected
            0.3 * normalize(f.area_ratio - 1, 0, 0.5) +     # object opens/expands
            0.2 * normalize(f.state_change_score, 0, 1)
        )
        evidence = f"Rotation detected: {f.rotation_change:.0f}°; area ratio: {f.area_ratio:.2f}"
    
    elif action_verb_hint in ["POUR", "FILL"]:
        score = (
            0.5 * normalize(abs(f.vertical_motion_ratio), 0, 1) +  # tilting motion
            0.3 * normalize(f.rotation_change, 0, 60) +
            0.2 * normalize(f.displacement_magnitude, 0, 100)
        )
        evidence = f"Container tilt detected; vertical displacement: {f.vertical_motion_ratio:.2f}"
    
    elif action_verb_hint in ["PICK", "GRAB"]:
        score = (
            0.5 * normalize(f.vertical_motion_ratio, 0, 1) +   # upward motion
            0.3 * (1 - normalize(f.contact_distance_mean, 0, 200)) +  # close to hand
            0.2 * normalize(f.state_change_score, 0, 1)
        )
        evidence = f"Upward motion: {f.vertical_motion_ratio:.2f}; hand-object proximity: {f.contact_distance_mean:.0f}px"
    
    return ActionResult(
        action_label=action_verb_hint,
        confidence=score,
        evidence=evidence,
        is_detected=(score > 0.5)
    )
```

### Key Frame Selector
```python
def select_key_frames(track_history, motion_history, n=3):
    """Select 3 most informative frames"""
    scores = []
    for i, (frame, motion) in enumerate(zip(track_history, motion_history)):
        # High score = high motion change = informative moment
        score = motion.area_variance + abs(motion.rotation_change) + motion.contact_frequency
        scores.append((score, i, frame))
    scores.sort(reverse=True)
    return [frame for _, _, frame in scores[:n]]
```

---

## Stage 6b: Segmentor (`pipeline/segmentor.py`)

### What it does
On selected key frames, run MobileSAM to produce a clean object mask from the bounding box.

### Architecture Decision: **MobileSAM (prompt-based)**
- Input: key frame image + bounding box prompt
- Output: binary mask for the object
- Only run on 3-5 key frames (not every frame — too slow)

### Usage
```python
from mobile_sam import sam_model_registry, SamPredictor

def segment_object(image, bbox):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=bbox,              # [x1, y1, x2, y2]
        multimask_output=False
    )
    return masks[0]  # binary mask
```

### When to run
- On the 3 selected key frames
- When `detection_confidence < 0.4` (to refine bbox → mask for better tracking)
- On the final frame to show the user what object was tracked

---

## Final Output Contract (to Yash)
```python
{
  "action_detected": bool,
  "action_label": str,               # "cutting", "opening", etc.
  "confidence": float,               # 0-1
  "timestamp_range": [start_ms, end_ms],
  "key_frames": [frame_img * 3],     # 3 numpy arrays
  "trajectory": [[cx, cy] * N],      # object path
  "object_mask": ndarray | None,     # from MobileSAM on key frame
  "evidence": str,                   # human-readable explanation
  "motion_summary": {
      "rotation": float,
      "displacement": float,
      "contact_events": int,
      "area_change": float,
  }
}
```

---

## Performance Targets
| Module | Target Latency | Notes |
|--------|---------------|-------|
| Tracker | < 5ms/frame | Pure NumPy |
| Motion Engine | < 3ms/frame | All math ops |
| Action Inferencer | < 1ms | Rule-based |
| MobileSAM | < 200ms | Only on key frames |
| **Total Stage 4-6** | **< 10ms/frame** | (excl. SAM) |

---

## Dev Notes for Hackathon Speed
1. Install: `pip install mobile-sam filterpy`
2. Use `filterpy.kalman.KalmanFilter` for the tracker
3. For rotation proxy: `cv2.minAreaRect(contour)[2]` gives OBB angle
4. For contact frequency: use `scipy.signal.find_peaks` on contact_distance signal
5. Test your motion engine on a pre-recorded video first before live integration
6. Key frame visualizer: draw bbox + trajectory on frame using OpenCV

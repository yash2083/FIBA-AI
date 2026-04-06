# 🎨 Design — Tanishk
## Implementation Spec: Tracker + Motion Engine + Action Inferencer + Segmentor
### FIBA AI | MIT Bangalore Hitachi Hackathon

---

## Complete Source Code

### `pipeline/tracker.py`

```python
"""
Object Tracker — Tanishk
Lightweight IoU + Kalman-based tracking for the query object across frames.
ByteTrack-inspired: keeps both high and low confidence detections.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class TrackResult:
    tracked: bool
    track_id: int = -1
    bbox: Optional[List[float]] = None          # [x1, y1, x2, y2]
    center: Optional[Tuple[float, float]] = None
    area: float = 0.0
    tracking_confidence: float = 0.0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    bbox_history: List[List[float]] = field(default_factory=list)
    area_history: List[float] = field(default_factory=list)


def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class SimpleKalmanBBox:
    """
    Simple 1D Kalman filter per bbox coordinate.
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """
    def __init__(self, initial_bbox):
        x1, y1, x2, y2 = initial_bbox
        self.state = np.array([x1, y1, x2, y2, 0., 0., 0., 0.], dtype=float)
        # Process noise
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] *= 5  # higher uncertainty on velocity
        # Observation noise
        self.R = np.eye(4) * 1.0
        # State covariance
        self.P = np.eye(8) * 10.0
        # State transition (constant velocity)
        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        # Observation matrix
        self.H = np.zeros((4, 8))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0
    
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:4]
    
    def update(self, bbox):
        z = np.array(bbox, dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.state[:4]
    
    def get_bbox(self):
        return self.state[:4].tolist()


class ObjectTracker:
    """
    Tracks the single query-relevant object across frames.
    Uses Kalman prediction + IoU association.
    """
    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        
        self.track_id = 0
        self.kalman: Optional[SimpleKalmanBBox] = None
        self.lost_frames = 0
        self.is_active = False
        
        # History for motion analysis
        self.bbox_history: List[List[float]] = []
        self.center_history: List[Tuple[float, float]] = []
        self.area_history: List[float] = []
        self.frame_ids: List[int] = []
    
    def _bbox_center(self, bbox):
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    
    def _bbox_area(self, bbox):
        return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
    
    def update(self, detection_result, frame_id: int) -> TrackResult:
        """
        Update tracker with new detection result.
        Returns TrackResult with current tracked state.
        """
        # Predict step
        predicted_bbox = None
        if self.is_active and self.kalman:
            predicted_bbox = self.kalman.predict().tolist()
        
        # New detection available?
        has_detection = detection_result is not None and detection_result.detected
        
        if has_detection:
            new_bbox = detection_result.object_bbox
            
            if not self.is_active:
                # Initialize new track
                self.kalman = SimpleKalmanBBox(new_bbox)
                self.track_id += 1
                self.is_active = True
                self.lost_frames = 0
                current_bbox = new_bbox
            else:
                # Check IoU with predicted position
                iou = compute_iou(predicted_bbox, new_bbox) if predicted_bbox else 0
                if iou >= self.iou_threshold:
                    # Associate and update
                    current_bbox = self.kalman.update(new_bbox).tolist()
                    self.lost_frames = 0
                else:
                    # Low IoU — might be different object, use Kalman prediction
                    current_bbox = predicted_bbox
                    self.lost_frames += 1
        else:
            if not self.is_active:
                return TrackResult(tracked=False, trajectory=[], bbox_history=[], area_history=[])
            
            # No detection — use Kalman prediction
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.is_active = False
                return TrackResult(tracked=False, trajectory=self.center_history.copy(),
                                   bbox_history=self.bbox_history.copy(),
                                   area_history=self.area_history.copy())
            current_bbox = predicted_bbox or self.bbox_history[-1]
        
        # Update history
        center = self._bbox_center(current_bbox)
        area = self._bbox_area(current_bbox)
        self.bbox_history.append(current_bbox)
        self.center_history.append(center)
        self.area_history.append(area)
        self.frame_ids.append(frame_id)
        
        tracking_conf = max(0.0, 1.0 - (self.lost_frames / self.max_lost_frames))
        
        return TrackResult(
            tracked=True,
            track_id=self.track_id,
            bbox=current_bbox,
            center=center,
            area=area,
            tracking_confidence=tracking_conf,
            trajectory=self.center_history.copy(),
            bbox_history=self.bbox_history.copy(),
            area_history=self.area_history.copy(),
        )
    
    def get_history(self):
        return {
            "bbox_history": self.bbox_history,
            "center_history": self.center_history,
            "area_history": self.area_history,
            "frame_ids": self.frame_ids,
        }
    
    def reset(self):
        self.__init__()
```

---

### `pipeline/motion_engine.py`

```python
"""
Motion Engine — Tanishk
Extracts interpretable motion features from tracked object history.
Core innovation: replaces heavy action recognition with physics-inspired features.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2


@dataclass
class MotionFeatures:
    # Translational motion
    displacement_magnitude: float = 0.0    # total pixels moved
    displacement_direction: float = 0.0    # degrees from horizontal
    vertical_motion_ratio: float = 0.0     # -1 down, +1 up
    motion_speed: float = 0.0              # pixels/frame average
    
    # Rotational proxy
    rotation_change: float = 0.0           # degrees total
    rotation_speed: float = 0.0            # degrees/frame
    
    # Area/scale change
    area_ratio: float = 1.0                # end_area / start_area
    area_variance: float = 0.0             # fragmentation indicator
    area_growth_rate: float = 0.0          # area/frame
    
    # Hand-object interaction
    contact_distance_mean: float = 999.0   # mean hand-object dist
    contact_frequency: float = 0.0         # oscillations/sec
    contact_events: int = 0                # number of close contacts
    
    # State change
    state_change_score: float = 0.0        # 0-1, how much changed
    
    # Timing
    window_frames: int = 0


class MotionEngine:
    def __init__(self, frame_window: int = 30, contact_threshold: float = 80.0):
        """
        Args:
            frame_window: Sliding window size for motion computation
            contact_threshold: Pixel distance to count as hand-object contact
        """
        self.frame_window = frame_window
        self.contact_threshold = contact_threshold
    
    def compute(self, 
                tracker_history: dict, 
                hand_history: List = None,
                frame_height: int = 480) -> MotionFeatures:
        """
        Compute motion features from tracker history.
        
        Args:
            tracker_history: dict with bbox_history, center_history, area_history
            hand_history: Optional list of hand wrist positions per frame
            frame_height: Frame height for normalization
        """
        centers = tracker_history.get("center_history", [])
        areas = tracker_history.get("area_history", [])
        bboxes = tracker_history.get("bbox_history", [])
        
        if len(centers) < 3:
            return MotionFeatures()
        
        # Use last N frames for analysis
        N = min(self.frame_window, len(centers))
        centers_w = np.array(centers[-N:])
        areas_w = np.array(areas[-N:])
        bboxes_w = bboxes[-N:]
        
        features = MotionFeatures(window_frames=N)
        
        # --- TRANSLATIONAL MOTION ---
        start = centers_w[0]
        end = centers_w[-1]
        diff = end - start
        
        features.displacement_magnitude = float(np.linalg.norm(diff))
        features.displacement_direction = float(np.degrees(np.arctan2(-diff[1], diff[0])))
        features.vertical_motion_ratio = float(-diff[1] / (frame_height * 0.5))  # up = positive
        features.vertical_motion_ratio = np.clip(features.vertical_motion_ratio, -1, 1)
        
        # Speed: mean frame-to-frame displacement
        frame_disps = np.linalg.norm(np.diff(centers_w, axis=0), axis=1)
        features.motion_speed = float(np.mean(frame_disps))
        
        # --- ROTATIONAL PROXY ---
        angles = []
        for bbox in bboxes_w:
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                angles.append(angle)
        
        if len(angles) >= 2:
            angle_diffs = np.diff(angles)
            # Unwrap to avoid ±180 jumps
            angle_diffs = np.where(angle_diffs > 90, angle_diffs - 180, angle_diffs)
            angle_diffs = np.where(angle_diffs < -90, angle_diffs + 180, angle_diffs)
            features.rotation_change = float(np.sum(angle_diffs))
            features.rotation_speed = float(np.mean(np.abs(angle_diffs)))
        
        # --- AREA / SCALE CHANGE ---
        if areas_w[0] > 0 and areas_w[-1] > 0:
            features.area_ratio = float(areas_w[-1] / areas_w[0])
        features.area_variance = float(np.std(areas_w))
        area_diffs = np.diff(areas_w)
        features.area_growth_rate = float(np.mean(area_diffs))
        
        # --- HAND-OBJECT INTERACTION ---
        if hand_history and len(hand_history) >= N:
            hand_w = np.array([h for h in hand_history[-N:] if h is not None], dtype=float)
            if len(hand_w) >= 2:
                # Compute distances
                min_len = min(len(hand_w), len(centers_w))
                dists = np.linalg.norm(centers_w[:min_len] - hand_w[:min_len], axis=1)
                features.contact_distance_mean = float(np.mean(dists))
                
                # Contact events: frames where distance < threshold
                contacts = dists < self.contact_threshold
                features.contact_events = int(np.sum(contacts))
                
                # Contact frequency: oscillations (rising edges in contact signal)
                contact_float = contacts.astype(float)
                transitions = np.diff(contact_float)
                features.contact_frequency = float(np.sum(transitions > 0))
        
        # --- STATE CHANGE SCORE ---
        # Compare early vs late features
        early_N = N // 4
        late_N = N // 4
        if early_N > 0 and late_N > 0:
            early_area = float(np.mean(areas_w[:early_N]))
            late_area = float(np.mean(areas_w[-late_N:]))
            early_y = float(np.mean(centers_w[:early_N, 1]))
            late_y = float(np.mean(centers_w[-late_N:, 1]))
            
            area_change = abs(late_area - early_area) / (early_area + 1e-5)
            pos_change = abs(late_y - early_y) / (frame_height + 1e-5)
            rot_change = abs(features.rotation_change) / 90.0  # normalize to 90 deg
            
            features.state_change_score = float(np.clip(
                0.4 * area_change + 0.3 * pos_change + 0.3 * rot_change, 0, 1))
        
        return features
    
    def select_key_frame_indices(self, 
                                  all_motion_features: List[MotionFeatures],
                                  n: int = 3) -> List[int]:
        """
        Select the n most informative frame indices.
        Picks frames with highest motion activity.
        """
        if not all_motion_features:
            return []
        
        scores = []
        for i, f in enumerate(all_motion_features):
            score = (abs(f.displacement_magnitude) + 
                    abs(f.rotation_change) * 2 + 
                    f.area_variance * 0.01 +
                    f.contact_events * 5)
            scores.append((score, i))
        
        scores.sort(reverse=True)
        selected = sorted([i for _, i in scores[:n]])
        
        # Always include first and last if less than 3 selected
        if len(selected) < n:
            if 0 not in selected:
                selected.insert(0, 0)
            if len(all_motion_features)-1 not in selected:
                selected.append(len(all_motion_features)-1)
        
        return selected[:n]
```

---

### `pipeline/action_inferencer.py`

```python
"""
Action Inferencer — Tanishk
Rule-based action inference from motion features.
Explainable: produces human-readable evidence for each decision.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pipeline.motion_engine import MotionFeatures


@dataclass
class ActionResult:
    action_label: str
    action_category: str
    is_detected: bool
    confidence: float
    evidence: str
    timestamp_range: Tuple[float, float]
    key_frame_indices: List[int] = field(default_factory=list)
    motion_summary: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)


def _norm(val: float, lo: float, hi: float) -> float:
    """Normalize val to [0,1] given expected [lo, hi] range."""
    if hi == lo:
        return 0.0
    return float(max(0.0, min(1.0, (val - lo) / (hi - lo))))


class ActionInferencer:
    DETECTION_THRESHOLD = 0.45  # minimum confidence to call "detected"
    
    def infer(self, 
              features: MotionFeatures, 
              action_category: str,
              action_verb: str,
              timestamps: Tuple[float, float] = (0, 0)) -> ActionResult:
        """
        Infer action from motion features using rule-based scoring.
        
        Args:
            features: MotionFeatures from MotionEngine
            action_category: e.g., "CUT", "OPEN", "POUR"
            action_verb: raw verb from query (e.g., "cutting")
            timestamps: (start_ms, end_ms) of analyzed segment
        """
        score, evidence = self._score_action(features, action_category)
        
        motion_summary = {
            "rotation_deg": round(features.rotation_change, 1),
            "displacement_px": round(features.displacement_magnitude, 1),
            "contact_events": features.contact_events,
            "area_change_ratio": round(features.area_ratio, 2),
            "state_change": round(features.state_change_score, 2),
            "vertical_motion": round(features.vertical_motion_ratio, 2),
        }
        
        return ActionResult(
            action_label=action_verb,
            action_category=action_category,
            is_detected=(score >= self.DETECTION_THRESHOLD),
            confidence=round(score, 3),
            evidence=evidence,
            timestamp_range=timestamps,
            motion_summary=motion_summary,
        )
    
    def _score_action(self, f: MotionFeatures, category: str):
        """Returns (score 0-1, evidence string)"""
        
        if category == "CUT":
            # Signature: repeated hand-object contact + object fragmentation + low displacement
            contact_score = _norm(f.contact_frequency, 0, 5)
            frag_score = _norm(f.area_variance, 0, 1000)
            stable_score = 1.0 - _norm(f.displacement_magnitude, 0, 150)
            score = 0.40 * contact_score + 0.35 * frag_score + 0.25 * stable_score
            evidence = (f"Tool-object contact repeated {f.contact_events}x "
                       f"(frequency={f.contact_frequency:.1f}); "
                       f"object area variance={f.area_variance:.0f}px² "
                       f"(fragmentation indicator); "
                       f"displacement={f.displacement_magnitude:.0f}px (stayed in place)")
        
        elif category == "OPEN":
            # Signature: rotation + area increase or interior reveal
            rot_score = _norm(abs(f.rotation_change), 0, 90)
            area_score = _norm(f.area_ratio - 1.0, 0, 0.5)
            state_score = f.state_change_score
            score = 0.50 * rot_score + 0.30 * area_score + 0.20 * state_score
            evidence = (f"Rotation detected: {f.rotation_change:.0f}°; "
                       f"object area ratio: {f.area_ratio:.2f} "
                       f"({'expanding' if f.area_ratio > 1 else 'stable'}); "
                       f"state change score: {f.state_change_score:.2f}")
        
        elif category == "POUR":
            # Signature: container tilt (rotation) + vertical/lateral motion
            tilt_score = _norm(abs(f.rotation_change), 0, 60)
            motion_score = _norm(f.displacement_magnitude, 0, 100)
            vert_score = _norm(abs(f.vertical_motion_ratio), 0, 1)
            score = 0.40 * tilt_score + 0.30 * motion_score + 0.30 * vert_score
            evidence = (f"Container tilt: {f.rotation_change:.0f}°; "
                       f"displacement: {f.displacement_magnitude:.0f}px; "
                       f"vertical motion ratio: {f.vertical_motion_ratio:.2f}")
        
        elif category == "PICK":
            # Signature: upward motion + close to hand + state change
            up_score = _norm(f.vertical_motion_ratio, 0.1, 1.0)
            close_score = _norm(200 - f.contact_distance_mean, 0, 200)
            score = 0.50 * up_score + 0.30 * close_score + 0.20 * f.state_change_score
            evidence = (f"Upward motion: ratio={f.vertical_motion_ratio:.2f}; "
                       f"hand-object distance: {f.contact_distance_mean:.0f}px; "
                       f"state change: {f.state_change_score:.2f}")
        
        elif category == "PLACE":
            # Signature: downward motion + object stabilizes
            down_score = _norm(-f.vertical_motion_ratio, 0.1, 1.0)
            stable_score = 1.0 - _norm(f.motion_speed, 0, 10)
            score = 0.50 * down_score + 0.30 * stable_score + 0.20 * f.state_change_score
            evidence = (f"Downward motion: ratio={f.vertical_motion_ratio:.2f}; "
                       f"motion speed: {f.motion_speed:.1f}px/frame")
        
        elif category == "MIX":
            # Signature: circular/oscillatory motion + tool contact
            # Proxy: high displacement with returning path (variance in both axes)
            circ_score = _norm(f.rotation_speed, 0, 5)
            contact_score = _norm(f.contact_frequency, 0, 8)
            speed_score = _norm(f.motion_speed, 2, 20)
            score = 0.35 * circ_score + 0.35 * contact_score + 0.30 * speed_score
            evidence = (f"Oscillatory motion detected: rotation speed={f.rotation_speed:.1f}°/frame; "
                       f"contact frequency={f.contact_frequency:.1f}; "
                       f"motion speed={f.motion_speed:.1f}px/frame")
        
        elif category == "CLOSE":
            # Inverse of OPEN
            rot_score = _norm(abs(f.rotation_change), 0, 90)
            shrink_score = _norm(1.0 - f.area_ratio, 0, 0.5)
            score = 0.50 * rot_score + 0.50 * shrink_score
            evidence = (f"Closing rotation: {f.rotation_change:.0f}°; "
                       f"area shrink: {f.area_ratio:.2f}")
        
        else:
            # Generic: use state change score
            score = f.state_change_score
            evidence = f"Motion detected with state change score: {f.state_change_score:.2f}"
        
        return float(np.clip(score, 0, 1)) if score else 0.0, evidence


import numpy as np  # for clip
```

---

### `pipeline/segmentor.py`

```python
"""
Segmentor — Tanishk
MobileSAM-based object segmentation for key frames.
Generates clean object masks from bounding box prompts.
Only runs on selected key frames (not every frame).
"""

import cv2
import numpy as np
from typing import Optional, List
import base64


def encode_frame_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV frame as base64 JPEG string for web transmission."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def draw_annotated_frame(frame: np.ndarray,
                          hand_bbox=None,
                          obj_bbox=None,
                          obj_mask=None,
                          obj_label: str = "",
                          confidence: float = 0.0,
                          frame_id: int = 0,
                          timestamp_ms: float = 0.0) -> np.ndarray:
    """
    Draw all annotations on a frame for display.
    Returns annotated copy of frame.
    """
    out = frame.copy()
    
    # Draw object mask (semi-transparent overlay)
    if obj_mask is not None:
        colored_mask = np.zeros_like(out)
        colored_mask[obj_mask > 0] = [0, 100, 255]  # Orange-blue
        out = cv2.addWeighted(out, 0.75, colored_mask, 0.25, 0)
    
    # Draw object bbox
    if obj_bbox:
        x1, y1, x2, y2 = [int(c) for c in obj_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 80, 255), 2)
        label_text = f"{obj_label} {confidence:.0%}"
        cv2.putText(out, label_text, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    
    # Draw hand bbox
    if hand_bbox:
        x1, y1, x2, y2 = [int(c) for c in hand_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, "Hand", (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    
    # Timestamp overlay
    ts = f"t={timestamp_ms/1000:.2f}s  frame={frame_id}"
    cv2.putText(out, ts, (10, out.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return out


def draw_trajectory(frame_shape, trajectory: List, 
                    color=(255, 140, 0), thickness=2) -> np.ndarray:
    """Draw object centroid trajectory on a blank canvas."""
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    if len(trajectory) < 2:
        return canvas
    
    pts = np.array([(int(cx), int(cy)) for cx, cy in trajectory], dtype=np.int32)
    
    # Draw gradient-colored path (start=blue, end=red)
    for i in range(1, len(pts)):
        progress = i / len(pts)
        color_interp = (
            int(255 * progress),      # R increases
            50,                        # G constant
            int(255 * (1-progress)),  # B decreases
        )
        cv2.line(canvas, tuple(pts[i-1]), tuple(pts[i]), color_interp, thickness)
    
    # Start point (green circle)
    cv2.circle(canvas, tuple(pts[0]), 8, (0, 255, 0), -1)
    cv2.putText(canvas, "START", tuple(pts[0]+np.array([5,0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    
    # End point (red circle)
    cv2.circle(canvas, tuple(pts[-1]), 8, (0, 0, 255), -1)
    cv2.putText(canvas, "END", tuple(pts[-1]+np.array([5,0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    
    return canvas


class MobileSAMSegmentor:
    """
    MobileSAM wrapper for key-frame segmentation.
    Falls back to GrabCut if MobileSAM not installed.
    """
    def __init__(self):
        self.model = None
        self.predictor = None
        self._try_load_mobile_sam()
    
    def _try_load_mobile_sam(self):
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            import torch
            sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
            sam.eval()
            self.predictor = SamPredictor(sam)
            print("[Segmentor] MobileSAM loaded.")
        except Exception as e:
            print(f"[Segmentor] MobileSAM not available ({e}). Using GrabCut fallback.")
            self.predictor = None
    
    def segment(self, frame: np.ndarray, bbox: Optional[List[float]]) -> Optional[np.ndarray]:
        """
        Segment object in frame given bounding box.
        Returns binary mask (same HxW as frame), or None.
        """
        if bbox is None:
            return None
        
        if self.predictor is not None:
            return self._sam_segment(frame, bbox)
        else:
            return self._grabcut_segment(frame, bbox)
    
    def _sam_segment(self, frame, bbox):
        import numpy as np
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox),
            multimask_output=False
        )
        return masks[0].astype(np.uint8) * 255
    
    def _grabcut_segment(self, frame, bbox):
        """GrabCut fallback — no model needed."""
        x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
        h, w = frame.shape[:2]
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        rect = (x1, y1, x2-x1, y2-y1)
        
        try:
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask==2)|(mask==0), 0, 255).astype(np.uint8)
            return result
        except:
            return None
```

---

## Test Script for Tanishk's Modules

```bash
# Install deps
pip install filterpy opencv-python-headless numpy scipy

# Test tracker (standalone)
python -c "
from pipeline.tracker import ObjectTracker
from pipeline.hand_detector import HandDetectionResult
from pipeline.object_detector import ObjectDetectionResult

tracker = ObjectTracker()
# Simulate 10 frames with detection
for i in range(10):
    det = ObjectDetectionResult(detected=True, 
        object_bbox=[100+i*2, 100, 150+i*2, 150], 
        center=(125+i*2, 125), area=2500, detection_confidence=0.8)
    result = tracker.update(det, i)
    print(f'Frame {i}: tracked={result.tracked}, center={result.center}')
"
```

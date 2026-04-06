"""
Object Tracker — Tanishk
========================
Lightweight IoU + Kalman-based tracking for the query object across frames.
ByteTrack-inspired: keeps both high and low confidence detections.

Owner: Tanishk
Receives: ObjectDetectionResult (from Atul's object_detector.py)
Outputs:  TrackResult (to Yash's integrator.py)

Interface contract (from common_integration.md):
    TrackResult:
        tracked: bool
        track_id: int
        bbox: Optional[List[float]]         [x1, y1, x2, y2]
        center: Optional[Tuple[float,float]]
        area: float
        tracking_confidence: float
        trajectory: List[Tuple[float,float]]
        bbox_history: List[List[float]]
        area_history: List[float]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    """Tracker output per frame. Exported to Yash's integrator."""
    tracked: bool
    track_id: int = -1
    bbox: Optional[List[float]] = None          # [x1, y1, x2, y2]
    center: Optional[Tuple[float, float]] = None
    area: float = 0.0
    tracking_confidence: float = 0.0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    bbox_history: List[List[float]] = field(default_factory=list)
    area_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, 8-state)
# ---------------------------------------------------------------------------

class SimpleKalmanBBox:
    """
    Simple constant-velocity Kalman filter for a bounding box.
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]

    Designed to be lightweight (no scipy/filterpy dependency).
    """

    def __init__(self, initial_bbox: List[float]):
        x1, y1, x2, y2 = initial_bbox
        self.state = np.array(
            [x1, y1, x2, y2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        # Process noise — higher for velocity dimensions
        self.Q = np.eye(8) * 0.1
        self.Q[4:, 4:] *= 5.0

        # Observation noise
        self.R = np.eye(4) * 1.0

        # State covariance (initial uncertainty)
        self.P = np.eye(8) * 10.0

        # State transition (constant velocity)
        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0

        # Observation matrix (we observe x1,y1,x2,y2 directly)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    def predict(self) -> np.ndarray:
        """Kalman predict step. Returns predicted bbox [x1,y1,x2,y2]."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:4].copy()

    def update(self, bbox: List[float]) -> np.ndarray:
        """Kalman update step. Returns corrected bbox [x1,y1,x2,y2]."""
        z = np.array(bbox, dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.state[:4].copy()

    def get_bbox(self) -> List[float]:
        return self.state[:4].tolist()


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Tracks the single query-relevant object across frames.

    Approach:
      - Kalman filter predicts object position each frame
      - IoU association decides whether new detection matches track
      - ByteTrack-inspired: low-score detections still considered to prevent
        losing the track when the object is partially occluded
      - When no detection for MAX_LOST_FRAMES, the track is killed

    Usage (in Yash's integrator per-frame loop):
        tracker = ObjectTracker()
        result = tracker.update(obj_detection_result, frame_id)
    """

    # Tunable parameters (ByteTrack-inspired)
    MAX_LOST_FRAMES = 10
    IOU_THRESHOLD = 0.3
    HIGH_SCORE_THRESHOLD = 0.6   # detections above this: always associate
    LOW_SCORE_THRESHOLD = 0.2    # detections above this: try to associate

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 10,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

        self.track_id: int = 0
        self.kalman: Optional[SimpleKalmanBBox] = None
        self.lost_frames: int = 0
        self.is_active: bool = False

        # Full history — used by MotionEngine
        self.bbox_history: List[List[float]] = []
        self.center_history: List[Tuple[float, float]] = []
        self.area_history: List[float] = []
        self.frame_ids: List[int] = []

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _bbox_area(bbox: List[float]) -> float:
        return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(self, detection_result, frame_id: int) -> TrackResult:
        """
        Update tracker with the latest detection from Atul's object detector.

        Args:
            detection_result: ObjectDetectionResult (or None)
                              Must have .detected: bool  and  .object_bbox: List[float]
            frame_id: integer frame index

        Returns:
            TrackResult with current tracked state for this frame.
        """
        # --- Kalman predict ---
        predicted_bbox: Optional[List[float]] = None
        if self.is_active and self.kalman is not None:
            predicted_bbox = self.kalman.predict().tolist()

        # --- Check new detection ---
        has_detection = (
            detection_result is not None
            and getattr(detection_result, "detected", False)
            and getattr(detection_result, "object_bbox", None) is not None
        )

        if has_detection:
            new_bbox: List[float] = list(detection_result.object_bbox)
            det_conf: float = getattr(detection_result, "detection_confidence", 0.8)

            if not self.is_active:
                # Initialize a brand-new track
                if det_conf >= self.LOW_SCORE_THRESHOLD:
                    self.kalman = SimpleKalmanBBox(new_bbox)
                    self.track_id += 1
                    self.is_active = True
                    self.lost_frames = 0
                    current_bbox = new_bbox
                else:
                    # Detection too weak to start a track
                    return TrackResult(
                        tracked=False,
                        trajectory=list(self.center_history),
                        bbox_history=list(self.bbox_history),
                        area_history=list(self.area_history),
                    )
            else:
                # Associate with existing track via IoU
                if predicted_bbox is not None:
                    iou = compute_iou(predicted_bbox, new_bbox)
                else:
                    iou = 0.0

                if iou >= self.iou_threshold:
                    # Good match — update Kalman
                    current_bbox = self.kalman.update(new_bbox).tolist()
                    self.lost_frames = 0
                elif det_conf >= self.HIGH_SCORE_THRESHOLD:
                    # ByteTrack: high-confidence detection accepted even with low IoU
                    # (object may have jumped/reappeared)
                    current_bbox = self.kalman.update(new_bbox).tolist()
                    self.lost_frames = 0
                else:
                    # Poor IoU + low confidence → coast on Kalman
                    current_bbox = predicted_bbox if predicted_bbox else new_bbox
                    self.lost_frames += 1

        else:
            # No detection this frame
            if not self.is_active:
                return TrackResult(
                    tracked=False,
                    trajectory=list(self.center_history),
                    bbox_history=list(self.bbox_history),
                    area_history=list(self.area_history),
                )

            self.lost_frames += 1

            if self.lost_frames > self.max_lost_frames:
                # Track is dead — return final state
                self.is_active = False
                return TrackResult(
                    tracked=False,
                    trajectory=list(self.center_history),
                    bbox_history=list(self.bbox_history),
                    area_history=list(self.area_history),
                )

            # Keep coasting on Kalman prediction
            if predicted_bbox is not None:
                current_bbox = predicted_bbox
            elif self.bbox_history:
                current_bbox = self.bbox_history[-1]
            else:
                return TrackResult(tracked=False)

        # --- Update history ---
        center = self._bbox_center(current_bbox)
        area = self._bbox_area(current_bbox)

        self.bbox_history.append(list(current_bbox))
        self.center_history.append(center)
        self.area_history.append(area)
        self.frame_ids.append(frame_id)

        tracking_conf = max(0.0, 1.0 - (self.lost_frames / self.max_lost_frames))

        return TrackResult(
            tracked=True,
            track_id=self.track_id,
            bbox=list(current_bbox),
            center=center,
            area=area,
            tracking_confidence=tracking_conf,
            trajectory=list(self.center_history),
            bbox_history=list(self.bbox_history),
            area_history=list(self.area_history),
        )

    def get_history(self) -> dict:
        """
        Returns the full tracking history dict consumed by MotionEngine.
        Keys: bbox_history, center_history, area_history, frame_ids
        """
        return {
            "bbox_history": list(self.bbox_history),
            "center_history": list(self.center_history),
            "area_history": list(self.area_history),
            "frame_ids": list(self.frame_ids),
        }

    def reset(self):
        """Full tracker reset — call at the start of a new video."""
        self.__init__(
            iou_threshold=self.iou_threshold,
            max_lost_frames=self.max_lost_frames,
        )


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class _FakeDet:
        detected: bool
        object_bbox: list
        detection_confidence: float = 0.8

    tracker = ObjectTracker()
    print("=== ObjectTracker standalone test ===")
    for i in range(12):
        det = _FakeDet(detected=(i != 5),  # simulate frame 5 miss
                       object_bbox=[100 + i * 3, 80, 160 + i * 3, 140])
        result = tracker.update(det, frame_id=i)
        print(f"  frame={i:2d}  tracked={result.tracked}  center={result.center}"
              f"  conf={result.tracking_confidence:.2f}")
    print("\nHistory frames:", len(tracker.get_history()["frame_ids"]))
    print("Tracker test PASSED ✓")

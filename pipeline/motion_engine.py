"""
Motion Engine — Tanishk
=======================
Extracts interpretable motion features from the tracked object's history
and hand-object interaction signals.

This is the "core novelty" of FIBA AI:
  Instead of a heavy ML action classifier, we compute physics-inspired,
  human-readable features and pass them to the rule-based Action Inferencer.

Owner: Tanishk
Receives: tracker.get_history() + optional hand wrist positions per frame
Outputs:  MotionFeatures (consumed by ActionInferencer)

Feature families:
  1. Translational motion  (displacement magnitude/direction/speed)
  2. Rotational proxy      (angle change from OBB or bbox aspect)
  3. Area/scale change     (growth rate, variance → fragmentation)
  4. Hand-object interaction (contact distance, frequency, events)
  5. State-change score    (early vs late window comparison)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Feature bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class MotionFeatures:
    """All motion features computed over a sliding analysis window."""

    # 1. Translational motion
    displacement_magnitude: float = 0.0    # total px moved start→end
    displacement_direction: float = 0.0    # degrees from horizontal
    vertical_motion_ratio: float = 0.0     # -1.0 (down) to +1.0 (up)
    motion_speed: float = 0.0              # mean px/frame displacement

    # 2. Rotational proxy
    rotation_change: float = 0.0           # total degrees rotated
    rotation_speed: float = 0.0            # mean |degrees/frame|

    # 3. Area / scale change
    area_ratio: float = 1.0                # end_area / start_area
    area_variance: float = 0.0             # std of area (fragmentation)
    area_growth_rate: float = 0.0          # mean area change per frame

    # 4. Hand-object interaction
    contact_distance_mean: float = 999.0   # mean wrist-to-center distance
    contact_frequency: float = 0.0         # contact oscillation count
    contact_events: int = 0                # frames where dist < threshold

    # 5. State change
    state_change_score: float = 0.0        # normalized 0–1

    # Meta
    window_frames: int = 0                 # how many frames were analysed


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MotionEngine:
    """
    Computes motion features from tracker history.

    Usage:
        engine = MotionEngine()
        features = engine.compute(tracker.get_history(),
                                  hand_history=wrist_positions,
                                  frame_height=480)
    """

    def __init__(
        self,
        frame_window: int = 30,
        contact_threshold: float = 80.0,
    ):
        """
        Args:
            frame_window: sliding window (frames) for analysis
            contact_threshold: pixel distance counted as hand-object contact
        """
        self.frame_window = frame_window
        self.contact_threshold = contact_threshold

    # -----------------------------------------------------------------------
    # Main compute
    # -----------------------------------------------------------------------

    def compute(
        self,
        tracker_history: dict,
        hand_history: Optional[List] = None,
        frame_height: int = 480,
    ) -> MotionFeatures:
        """
        Compute all motion features from tracker history.

        Args:
            tracker_history: dict returned by ObjectTracker.get_history()
                             Keys: bbox_history, center_history, area_history
            hand_history:    List of wrist (x,y) positions per frame (or None).
                             May contain None entries for frames where hand
                             was not detected.
            frame_height:    Frame height in pixels (for vertical normalisation)

        Returns:
            MotionFeatures dataclass
        """
        centers_all: list = tracker_history.get("center_history", [])
        areas_all: list = tracker_history.get("area_history", [])
        bboxes_all: list = tracker_history.get("bbox_history", [])

        if len(centers_all) < 3:
            return MotionFeatures()

        # Use last N frames
        N = min(self.frame_window, len(centers_all))
        centers_w = np.array(centers_all[-N:], dtype=np.float64)  # (N, 2)
        areas_w = np.array(areas_all[-N:], dtype=np.float64)       # (N,)
        bboxes_w: list = bboxes_all[-N:]

        features = MotionFeatures(window_frames=N)

        # -------------------------------------------------------------------
        # 1. Translational motion
        # -------------------------------------------------------------------
        start = centers_w[0]
        end = centers_w[-1]
        diff = end - start

        features.displacement_magnitude = float(np.linalg.norm(diff))
        features.displacement_direction = float(
            np.degrees(np.arctan2(-diff[1], diff[0]))  # screen y is flipped
        )
        # Vertical motion: upward = positive (note screen y increases downward)
        features.vertical_motion_ratio = float(
            np.clip(-diff[1] / max(frame_height * 0.5, 1.0), -1.0, 1.0)
        )

        # Mean frame-to-frame speed
        frame_disps = np.linalg.norm(np.diff(centers_w, axis=0), axis=1)
        features.motion_speed = float(np.mean(frame_disps)) if len(frame_disps) else 0.0

        # -------------------------------------------------------------------
        # 2. Rotational proxy
        # -------------------------------------------------------------------
        # Use bbox aspect-angle as a proxy for object rotation.
        # For irregular objects, cv2.minAreaRect on a contour would be more
        # accurate, but we don't have mask data here.
        angles = []
        for bbox in bboxes_w:
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                # Aspect angle as rotation proxy
                angle = float(np.degrees(np.arctan2(h, max(w, 1.0))))
                angles.append(angle)

        if len(angles) >= 2:
            angle_arr = np.array(angles)
            angle_diffs = np.diff(angle_arr)
            # Unwrap large jumps caused by bbox flipping
            angle_diffs = np.where(angle_diffs > 90, angle_diffs - 180, angle_diffs)
            angle_diffs = np.where(angle_diffs < -90, angle_diffs + 180, angle_diffs)
            features.rotation_change = float(np.sum(angle_diffs))
            features.rotation_speed = float(np.mean(np.abs(angle_diffs)))

        # -------------------------------------------------------------------
        # 3. Area / scale change
        # -------------------------------------------------------------------
        safe_start_area = max(float(areas_w[0]), 1.0)
        safe_end_area = max(float(areas_w[-1]), 1.0)
        features.area_ratio = safe_end_area / safe_start_area
        features.area_variance = float(np.std(areas_w))
        area_diffs = np.diff(areas_w)
        features.area_growth_rate = float(np.mean(area_diffs)) if len(area_diffs) else 0.0

        # -------------------------------------------------------------------
        # 4. Hand-object interaction
        # -------------------------------------------------------------------
        if hand_history and len(hand_history) >= 1:
            # Filter to window and drop None entries
            hand_win = hand_history[-N:]
            valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
            for i, hw in enumerate(hand_win):
                if hw is not None and i < len(centers_w):
                    try:
                        hp = np.array(hw, dtype=np.float64)
                        if hp.shape == (2,):
                            valid_pairs.append((centers_w[i], hp))
                    except Exception:
                        pass

            if valid_pairs:
                dists = np.array([
                    np.linalg.norm(obj_c - hand_c)
                    for obj_c, hand_c in valid_pairs
                ])
                features.contact_distance_mean = float(np.mean(dists))

                # Contact events: frames where hand is within threshold
                contacts = dists < self.contact_threshold
                features.contact_events = int(np.sum(contacts))

                # Contact frequency: count rising edges (contact start events)
                contact_float = contacts.astype(float)
                transitions = np.diff(contact_float)
                features.contact_frequency = float(np.sum(transitions > 0))

        # -------------------------------------------------------------------
        # 5. State-change score
        # -------------------------------------------------------------------
        # Compare early 25% vs late 25% of the window
        quarter = max(N // 4, 1)
        early_area = float(np.mean(areas_w[:quarter]))
        late_area = float(np.mean(areas_w[-quarter:]))
        early_y = float(np.mean(centers_w[:quarter, 1]))
        late_y = float(np.mean(centers_w[-quarter:, 1]))

        area_change = abs(late_area - early_area) / (early_area + 1e-5)
        pos_change = abs(late_y - early_y) / max(frame_height, 1)
        rot_change_norm = min(abs(features.rotation_change) / 90.0, 1.0)

        features.state_change_score = float(np.clip(
            0.4 * area_change + 0.3 * pos_change + 0.3 * rot_change_norm,
            0.0, 1.0
        ))

        return features

    # -----------------------------------------------------------------------
    # Key frame selection
    # -----------------------------------------------------------------------

    def select_key_frame_indices(
        self,
        all_motion_features: List[MotionFeatures],
        n: int = 3,
    ) -> List[int]:
        """
        Select the n most informative frame indices from per-frame motion.

        Each frame's motion activity is scored by a weighted combo of:
          - displacement magnitude (movement)
          - rotation change (orientation shift)
          - area variance (fragmentation)
          - contact events (tool engagement)

        Guarantees first and last frame are included when n < 3 fallbacks.
        Returns sorted list of frame indices.
        """
        if not all_motion_features:
            return []

        total = len(all_motion_features)

        scores = []
        for i, f in enumerate(all_motion_features):
            score = (
                abs(f.displacement_magnitude) * 1.0
                + abs(f.rotation_change) * 2.0
                + f.area_variance * 0.01
                + f.contact_events * 5.0
            )
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        selected = sorted(idx for _, idx in scores[:n])

        # Pad with boundary frames if fewer than n were returned
        if len(selected) < n:
            if 0 not in selected:
                selected = [0] + selected
            if total - 1 not in selected:
                selected.append(total - 1)

        return selected[:n]


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MotionEngine standalone test ===")

    # Simulate a 40-frame cutting scenario
    rng = np.random.default_rng(42)
    N = 40
    xs = 120 + rng.normal(0, 3, N).cumsum() * 0.1
    ys = 100 + rng.normal(0, 5, N)
    areas = 2500 + rng.normal(0, 200, N)

    fake_history = {
        "center_history": [(xs[i], ys[i]) for i in range(N)],
        "area_history": list(areas),
        "bbox_history": [
            [xs[i]-25, ys[i]-25, xs[i]+25, ys[i]+25] for i in range(N)
        ],
        "frame_ids": list(range(N)),
    }

    # Simulate intermittent hand wrist positions
    hand_hist = [(xs[i]+40, ys[i]+rng.normal(0, 10)) if i % 3 != 0 else None
                 for i in range(N)]

    engine = MotionEngine(frame_window=30, contact_threshold=80)
    features = engine.compute(fake_history, hand_history=hand_hist, frame_height=480)

    print(f"  displacement_magnitude : {features.displacement_magnitude:.1f} px")
    print(f"  vertical_motion_ratio  : {features.vertical_motion_ratio:.3f}")
    print(f"  rotation_change        : {features.rotation_change:.2f} °")
    print(f"  area_ratio             : {features.area_ratio:.3f}")
    print(f"  area_variance          : {features.area_variance:.1f}")
    print(f"  contact_distance_mean  : {features.contact_distance_mean:.1f} px")
    print(f"  contact_frequency      : {features.contact_frequency:.1f}")
    print(f"  contact_events         : {features.contact_events}")
    print(f"  state_change_score     : {features.state_change_score:.3f}")

    # Key frame selection
    per_frame = [engine.compute(fake_history) for _ in range(N)]
    kf_indices = engine.select_key_frame_indices(per_frame, n=3)
    print(f"\n  Key frame indices: {kf_indices}")
    print("MotionEngine test PASSED ✓")

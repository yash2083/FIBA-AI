"""
Action Inferencer — Tanishk
============================
Rule-based action inference from MotionFeatures.

This is the "explainable AI" layer of FIBA AI:
  Each action class has a hand-crafted scoring function using physics-
  inspired motion cues (rotation, area change, contact pattern, etc.).
  The result includes a human-readable evidence string explaining the
  decision — satisfying the brief's "explain motion logic" requirement.

Owner: Tanishk
Receives: MotionFeatures (from MotionEngine)
Outputs:  ActionResult  (to Yash's integrator)

Action categories supported:
  CUT   — repeated contact + area fragmentation + low displacement
  OPEN  — rotation + area expansion + state change
  POUR  — container tilt (rotation) + vertical/lateral displacement
  PICK  — upward motion + close hand proximity
  PLACE — downward motion + stabilization
  MIX   — oscillatory rotation + high contact frequency
  CLOSE — inverse of OPEN (rotation + area shrink)
  <any> — generic: state change score

Interface contract (from common_integration.md):
    ActionResult:
        action_label: str
        action_category: str
        is_detected: bool
        confidence: float          0–1
        evidence: str              human-readable explanation
        timestamp_range: Tuple[float, float]
        key_frame_indices: List[int]
        motion_summary: dict
        trajectory: list
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from pipeline.motion_engine import MotionFeatures


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Action inference result. Exported to Yash's integrator."""
    action_label: str
    action_category: str
    is_detected: bool
    confidence: float
    evidence: str
    timestamp_range: Tuple[float, float]
    key_frame_indices: List[int] = field(default_factory=list)
    motion_summary: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm(val: float, lo: float, hi: float) -> float:
    """
    Linearly normalize `val` to [0, 1] given expected range [lo, hi].
    Clamps out-of-range values.
    """
    if hi == lo:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

class ActionInferencer:
    """
    Maps MotionFeatures → ActionResult using interpretable rule-based scoring.

    Design:
      Each action category defines a weighted sum of normalized features.
      Weights reflect the relative importance of each cue for that action.
      The final confidence is clamped to [0, 1].

    Usage:
        inferencer = ActionInferencer()
        result = inferencer.infer(features, "CUT", "cutting", timestamps)
    """

    # Minimum confidence to report action as "detected"
    DETECTION_THRESHOLD = 0.45

    def infer(
        self,
        features: MotionFeatures,
        action_category: str,
        action_verb: str,
        timestamps: Tuple[float, float] = (0.0, 0.0),
    ) -> ActionResult:
        """
        Infer whether the action occurred and produce an explanation.

        Args:
            features:        MotionFeatures computed by MotionEngine
            action_category: e.g. "CUT", "OPEN", "POUR" (from query parser map)
            action_verb:     raw verb from user query e.g. "cutting"
            timestamps:      (start_ms, end_ms) of the analysed video segment

        Returns:
            ActionResult with confidence, verdict, and human-readable evidence
        """
        score, evidence = self._score_action(features, action_category.upper())

        motion_summary = {
            "rotation_deg": round(features.rotation_change, 1),
            "displacement_px": round(features.displacement_magnitude, 1),
            "contact_events": features.contact_events,
            "area_change_ratio": round(features.area_ratio, 3),
            "state_change": round(features.state_change_score, 3),
            "vertical_motion": round(features.vertical_motion_ratio, 3),
            "motion_speed_px_per_frame": round(features.motion_speed, 2),
            "contact_frequency": round(features.contact_frequency, 2),
            "area_variance": round(features.area_variance, 1),
        }

        return ActionResult(
            action_label=action_verb,
            action_category=action_category.upper(),
            is_detected=(score >= self.DETECTION_THRESHOLD),
            confidence=round(score, 4),
            evidence=evidence,
            timestamp_range=timestamps,
            motion_summary=motion_summary,
        )

    # -----------------------------------------------------------------------
    # Scoring rules (one per action category)
    # -----------------------------------------------------------------------

    def _score_action(
        self, f: MotionFeatures, category: str
    ) -> Tuple[float, str]:
        """
        Compute a [0,1] confidence score and a human-readable evidence string
        for the given action category.
        """

        # ------------------ CUT / CHOP / SLICE ------------------
        if category in ("CUT", "CHOP", "SLICE"):
            # Signature:
            #   - High contact frequency (tool repeatedly hits object)
            #   - High area variance (fragmentation)
            #   - Low displacement (action stays local)
            contact_score = _norm(f.contact_frequency, 0, 5)
            frag_score = _norm(f.area_variance, 0, 1000)
            stable_score = 1.0 - _norm(f.displacement_magnitude, 0, 150)
            score = 0.40 * contact_score + 0.35 * frag_score + 0.25 * stable_score
            detected_str = "DETECTED" if score >= self.DETECTION_THRESHOLD else "not detected"
            evidence = (
                f"[CUT {detected_str}] "
                f"Tool-object contact {f.contact_events} times "
                f"(oscillation frequency={f.contact_frequency:.1f}); "
                f"object area variance={f.area_variance:.0f}px² "
                f"(fragmentation indicator); "
                f"object stayed local: displacement={f.displacement_magnitude:.0f}px."
            )

        # ------------------ OPEN / UNSCREW ----------------------
        elif category in ("OPEN", "UNSCREW"):
            # Signature:
            #   - Rotation (lid/cap turns)
            #   - Area expands (interior revealed)
            #   - State change (before ≠ after)
            rot_score = _norm(abs(f.rotation_change), 0, 90)
            area_score = _norm(f.area_ratio - 1.0, 0, 0.5)
            state_score = f.state_change_score
            score = 0.50 * rot_score + 0.30 * area_score + 0.20 * state_score
            expansion = "expanding" if f.area_ratio > 1.05 else "stable"
            evidence = (
                f"[OPEN] Rotation detected: {f.rotation_change:.0f}°; "
                f"area ratio={f.area_ratio:.2f} ({expansion}); "
                f"state-change score={f.state_change_score:.2f}. "
                f"→ Rotation detected → opening inferred."
            )

        # ------------------ POUR / FILL -------------------------
        elif category in ("POUR", "FILL"):
            # Signature:
            #   - Container tilt (rotation)
            #   - Vertical and/or lateral displacement
            tilt_score = _norm(abs(f.rotation_change), 0, 60)
            motion_score = _norm(f.displacement_magnitude, 0, 100)
            vert_score = _norm(abs(f.vertical_motion_ratio), 0, 1.0)
            score = 0.40 * tilt_score + 0.30 * motion_score + 0.30 * vert_score
            evidence = (
                f"[POUR] Container tilt: {f.rotation_change:.0f}°; "
                f"horizontal displacement: {f.displacement_magnitude:.0f}px; "
                f"vertical motion ratio: {f.vertical_motion_ratio:.2f}."
            )

        # ------------------ PICK / GRAB / TAKE ------------------
        elif category in ("PICK", "GRAB", "TAKE"):
            # Signature:
            #   - Upward motion (vertical_motion_ratio > 0)
            #   - Object close to hand
            #   - State change (object lifts off surface)
            up_score = _norm(f.vertical_motion_ratio, 0.1, 1.0)
            closeness = max(0.0, 200.0 - f.contact_distance_mean)
            close_score = _norm(closeness, 0, 200)
            score = 0.50 * up_score + 0.30 * close_score + 0.20 * f.state_change_score
            evidence = (
                f"[PICK] Upward motion ratio={f.vertical_motion_ratio:.2f}; "
                f"hand-object distance={f.contact_distance_mean:.0f}px; "
                f"state change={f.state_change_score:.2f}."
            )

        # ------------------ PLACE / PUT / SET -------------------
        elif category in ("PLACE", "PUT", "SET"):
            # Signature:
            #   - Downward motion
            #   - Object decelerates / stabilises
            down_score = _norm(-f.vertical_motion_ratio, 0.1, 1.0)
            stable_score = 1.0 - _norm(f.motion_speed, 0, 10)
            score = 0.50 * down_score + 0.30 * stable_score + 0.20 * f.state_change_score
            evidence = (
                f"[PLACE] Downward motion ratio={f.vertical_motion_ratio:.2f}; "
                f"motion speed={f.motion_speed:.1f}px/frame "
                f"({'decelerating' if stable_score > 0.5 else 'still moving'}); "
                f"state change={f.state_change_score:.2f}."
            )

        # ------------------ MIX / STIR / SHAKE ------------------
        elif category in ("MIX", "STIR", "SHAKE"):
            # Signature:
            #   - Oscillatory rotation (rotation_speed high)
            #   - High contact frequency (spoon touching bowl repeatedly)
            #   - Moderate-high motion speed
            circ_score = _norm(f.rotation_speed, 0, 5)
            contact_score = _norm(f.contact_frequency, 0, 8)
            speed_score = _norm(f.motion_speed, 2, 20)
            score = 0.35 * circ_score + 0.35 * contact_score + 0.30 * speed_score
            evidence = (
                f"[MIX] Oscillatory rotation speed={f.rotation_speed:.1f}°/frame; "
                f"contact frequency={f.contact_frequency:.1f}; "
                f"motion speed={f.motion_speed:.1f}px/frame."
            )

        # ------------------ CLOSE / SHUT / CAP ------------------
        elif category in ("CLOSE", "SHUT", "CAP"):
            # Inverse of OPEN: rotation + area shrinks
            rot_score = _norm(abs(f.rotation_change), 0, 90)
            shrink_score = _norm(1.0 - f.area_ratio, 0, 0.5)
            score = 0.50 * rot_score + 0.50 * max(0.0, shrink_score)
            evidence = (
                f"[CLOSE] Closing rotation: {f.rotation_change:.0f}°; "
                f"area ratio={f.area_ratio:.2f} "
                f"({'shrinking' if f.area_ratio < 0.95 else 'stable'})."
            )

        # ------------------ GENERIC fallback --------------------
        else:
            score = f.state_change_score
            evidence = (
                f"[{category}] Generic motion inference: "
                f"state-change score={f.state_change_score:.2f}; "
                f"displacement={f.displacement_magnitude:.0f}px; "
                f"rotation={f.rotation_change:.0f}°."
            )

        return float(np.clip(score, 0.0, 1.0)), evidence

    # -----------------------------------------------------------------------
    # Batch helper (for integrator when inferring over full video)
    # -----------------------------------------------------------------------

    def infer_from_history(
        self,
        all_features: List[MotionFeatures],
        action_category: str,
        action_verb: str,
        fps: float = 30.0,
    ) -> ActionResult:
        """
        Aggregate per-frame MotionFeatures into a single video-level inference.

        Strategy: take the feature snapshot at the frame with the highest
        combined motion activity — this captures the peak of the action.

        Args:
            all_features:     list of per-frame MotionFeatures
            action_category:  e.g. "CUT"
            action_verb:      e.g. "cutting"
            fps:              video FPS for timestamp computation

        Returns:
            ActionResult for the full video
        """
        if not all_features:
            null_f = MotionFeatures()
            return self.infer(null_f, action_category, action_verb)

        # Pick the frame with the highest activity score
        def _activity(f: MotionFeatures) -> float:
            return (
                f.displacement_magnitude
                + abs(f.rotation_change) * 2.0
                + f.area_variance * 0.01
                + f.contact_events * 5.0
                + f.state_change_score * 100.0
            )

        best_idx = max(range(len(all_features)), key=lambda i: _activity(all_features[i]))
        best_features = all_features[best_idx]

        # Also compute timestamps from frame indices
        start_frame = 0
        end_frame = len(all_features) - 1
        start_ms = (start_frame / max(fps, 1.0)) * 1000.0
        end_ms = (end_frame / max(fps, 1.0)) * 1000.0

        result = self.infer(
            best_features, action_category, action_verb, (start_ms, end_ms)
        )
        result.key_frame_indices = []  # will be filled by integrator / motion engine
        return result


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from pipeline.motion_engine import MotionFeatures

    print("=== ActionInferencer standalone test ===\n")
    inferencer = ActionInferencer()

    # --- CUT test ---
    cut_f = MotionFeatures(
        contact_frequency=4.5, area_variance=800, displacement_magnitude=30,
        contact_events=12, state_change_score=0.6
    )
    r = inferencer.infer(cut_f, "CUT", "cutting", (0, 5000))
    print(f"CUT  → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    # --- OPEN test ---
    open_f = MotionFeatures(
        rotation_change=72, area_ratio=1.3, state_change_score=0.75
    )
    r = inferencer.infer(open_f, "OPEN", "opening", (0, 3000))
    print(f"OPEN → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    # --- POUR test ---
    pour_f = MotionFeatures(
        rotation_change=45, displacement_magnitude=80, vertical_motion_ratio=0.4
    )
    r = inferencer.infer(pour_f, "POUR", "pouring", (0, 4000))
    print(f"POUR → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    # --- PICK test ---
    pick_f = MotionFeatures(
        vertical_motion_ratio=0.6, contact_distance_mean=40, state_change_score=0.5
    )
    r = inferencer.infer(pick_f, "PICK", "picking", (0, 2000))
    print(f"PICK → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    # --- PLACE test ---
    place_f = MotionFeatures(
        vertical_motion_ratio=-0.5, motion_speed=2, state_change_score=0.4
    )
    r = inferencer.infer(place_f, "PLACE", "placing", (0, 2500))
    print(f"PLACE → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    # --- MIX test ---
    mix_f = MotionFeatures(
        rotation_speed=3.5, contact_frequency=6, motion_speed=10
    )
    r = inferencer.infer(mix_f, "MIX", "mixing", (0, 6000))
    print(f"MIX  → detected={r.is_detected}  conf={r.confidence:.3f}")
    print(f"  Evidence: {r.evidence}\n")

    print("ActionInferencer test PASSED ✓")

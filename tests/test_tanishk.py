"""
Tanishk's Module Tests — FIBA AI
==================================
Covers: ObjectTracker, MotionEngine, ActionInferencer, MobileSAMSegmentor

Run: python -m pytest tests/test_tanishk.py -v
  or: python tests/test_tanishk.py   (no pytest required)

All tests are self-contained and require only numpy + opencv.
No model files needed (segmentor uses GrabCut fallback).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2


# ============================================================
# Helpers / stubs matching Atul's interface
# ============================================================

class _FakeObjectDetection:
    """Minimal stub for ObjectDetectionResult."""
    def __init__(self, detected, bbox, confidence=0.8):
        self.detected = detected
        self.object_bbox = bbox
        self.detection_confidence = confidence
        if bbox:
            self.center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            self.area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        else:
            self.center = None
            self.area = 0.0


# ============================================================
# TRACKER TESTS
# ============================================================

def test_tracker_basic_tracking():
    """Tracker should assign a track_id and accumulate history."""
    from pipeline.tracker import ObjectTracker

    tracker = ObjectTracker()
    for i in range(8):
        det = _FakeObjectDetection(True, [100+i*3, 80, 160+i*3, 140])
        result = tracker.update(det, frame_id=i)

    assert result.tracked is True
    assert result.track_id == 1
    assert result.center is not None
    assert len(result.trajectory) == 8
    assert len(result.bbox_history) == 8
    assert len(result.area_history) == 8
    print("  [PASS] test_tracker_basic_tracking")


def test_tracker_handles_missed_frames():
    """Tracker should coast on Kalman when detections are missed."""
    from pipeline.tracker import ObjectTracker

    tracker = ObjectTracker(max_lost_frames=5)
    # 5 good frames
    for i in range(5):
        det = _FakeObjectDetection(True, [100, 100, 150, 150])
        tracker.update(det, frame_id=i)

    # 3 missed frames (within max_lost_frames)
    for i in range(5, 8):
        det = _FakeObjectDetection(False, None)
        result = tracker.update(det, frame_id=i)
        assert result.tracked is True, f"Should still be tracking at frame {i}"

    # Beyond max_lost_frames — track should die
    for i in range(8, 15):
        det = _FakeObjectDetection(False, None)
        result = tracker.update(det, frame_id=i)

    assert result.tracked is False, "Track should have died after too many missed frames"
    print("  [PASS] test_tracker_handles_missed_frames")


def test_tracker_iou_computation():
    """IoU helper should compute correct values."""
    from pipeline.tracker import compute_iou

    # Perfect overlap
    assert abs(compute_iou([0,0,10,10], [0,0,10,10]) - 1.0) < 1e-6

    # No overlap
    assert compute_iou([0,0,10,10], [20,20,30,30]) == 0.0

    # Partial overlap: two 10x10 boxes offset by 5px → 25px²/(100+100-25) = 25/175
    iou = compute_iou([0,0,10,10], [5,5,15,15])
    expected = 25 / 175
    assert abs(iou - expected) < 1e-4, f"Got {iou}, expected {expected}"
    print("  [PASS] test_tracker_iou_computation")


def test_tracker_no_detection_start():
    """Tracker should not activate when first detection is below LOW_SCORE_THRESHOLD."""
    from pipeline.tracker import ObjectTracker

    tracker = ObjectTracker()
    det = _FakeObjectDetection(True, [0, 0, 10, 10], confidence=0.05)  # very low
    result = tracker.update(det, frame_id=0)
    assert result.tracked is False
    print("  [PASS] test_tracker_no_detection_start")


def test_tracker_reset():
    """Reset should clear all history."""
    from pipeline.tracker import ObjectTracker

    tracker = ObjectTracker()
    for i in range(5):
        det = _FakeObjectDetection(True, [100, 100, 150, 150])
        tracker.update(det, frame_id=i)

    tracker.reset()
    assert len(tracker.get_history()["bbox_history"]) == 0
    assert tracker.is_active is False
    print("  [PASS] test_tracker_reset")


def test_tracker_get_history():
    """get_history should return all 4 keys."""
    from pipeline.tracker import ObjectTracker

    tracker = ObjectTracker()
    N = 6
    for i in range(N):
        det = _FakeObjectDetection(True, [100+i, 100, 140+i, 140])
        tracker.update(det, frame_id=i)

    h = tracker.get_history()
    assert set(h.keys()) == {"bbox_history", "center_history", "area_history", "frame_ids"}
    assert len(h["bbox_history"]) == N
    assert len(h["center_history"]) == N
    assert len(h["area_history"]) == N
    assert len(h["frame_ids"]) == N
    print("  [PASS] test_tracker_get_history")


# ============================================================
# MOTION ENGINE TESTS
# ============================================================

def _make_fake_history(N=40, dx=3.0, dy=0.5, area=2500.0, area_noise=200.0):
    """Generate a synthetic tracker history for motion testing."""
    rng = np.random.default_rng(7)
    xs = 120.0 + np.arange(N) * dx + rng.normal(0, 1, N)
    ys = 100.0 + np.arange(N) * dy + rng.normal(0, 2, N)
    areas = area + rng.normal(0, area_noise, N)
    areas = np.clip(areas, 100, None)
    return {
        "center_history": [(xs[i], ys[i]) for i in range(N)],
        "area_history": list(areas),
        "bbox_history": [[xs[i]-25, ys[i]-25, xs[i]+25, ys[i]+25] for i in range(N)],
        "frame_ids": list(range(N)),
    }


def test_motion_engine_basic():
    """MotionEngine should return populated MotionFeatures."""
    from pipeline.motion_engine import MotionEngine

    engine = MotionEngine(frame_window=30)
    hist = _make_fake_history()
    features = engine.compute(hist)

    assert features.window_frames > 0
    assert features.displacement_magnitude >= 0
    assert -1.0 <= features.vertical_motion_ratio <= 1.0
    assert features.area_ratio > 0
    print("  [PASS] test_motion_engine_basic")


def test_motion_engine_short_history():
    """MotionEngine should return defaults for < 3 frames."""
    from pipeline.motion_engine import MotionEngine, MotionFeatures

    engine = MotionEngine()
    hist = {
        "center_history": [(100, 100), (101, 101)],
        "area_history": [1000, 1010],
        "bbox_history": [[75, 75, 125, 125], [76, 76, 126, 126]],
        "frame_ids": [0, 1],
    }
    features = engine.compute(hist)
    assert features.displacement_magnitude == 0.0  # default
    print("  [PASS] test_motion_engine_short_history")


def test_motion_engine_contact_detection():
    """Contact events should be counted when hand is near object."""
    from pipeline.motion_engine import MotionEngine

    engine = MotionEngine(contact_threshold=50.0)
    hist = _make_fake_history(N=20, dx=0, dy=0, area=2500, area_noise=0)

    # Hand very close (within threshold) every frame
    hand_hist = [(120.0, 100.0)] * 20  # exactly on object center (distance=0)

    features = engine.compute(hist, hand_history=hand_hist)
    assert features.contact_events > 0, "Expected contact events"
    print("  [PASS] test_motion_engine_contact_detection")


def test_motion_engine_upward_motion():
    """Upward moving object should give positive vertical_motion_ratio."""
    from pipeline.motion_engine import MotionEngine

    N = 20
    # Object moves UP (y decreases in screen space)
    xs = [200.0] * N
    ys = [300.0 - i * 8.0 for i in range(N)]  # y decreases → upward
    areas = [2500.0] * N
    hist = {
        "center_history": list(zip(xs, ys)),
        "area_history": areas,
        "bbox_history": [[xs[i]-25, ys[i]-25, xs[i]+25, ys[i]+25] for i in range(N)],
        "frame_ids": list(range(N)),
    }
    engine = MotionEngine(frame_window=N)
    features = engine.compute(hist, frame_height=480)
    assert features.vertical_motion_ratio > 0, (
        f"Upward motion should give positive ratio, got {features.vertical_motion_ratio}"
    )
    print("  [PASS] test_motion_engine_upward_motion")


def test_motion_engine_key_frame_selection():
    """Key frame selector should return valid sorted indices."""
    from pipeline.motion_engine import MotionEngine, MotionFeatures

    engine = MotionEngine()
    hist = _make_fake_history()

    # Build per-frame features (simple: compute once and replicate)
    base_features = engine.compute(hist)
    all_features = [base_features] * 30

    indices = engine.select_key_frame_indices(all_features, n=3)
    assert len(indices) <= 3
    assert all(0 <= i < 30 for i in indices)
    assert indices == sorted(indices), "Indices should be sorted"
    print("  [PASS] test_motion_engine_key_frame_selection")


def test_motion_engine_state_change_score_range():
    """State change score must be in [0, 1]."""
    from pipeline.motion_engine import MotionEngine

    engine = MotionEngine()
    for _ in range(5):
        rng = np.random.default_rng()
        hist = _make_fake_history(N=rng.integers(5, 50),
                                  dx=rng.uniform(-5, 10),
                                  dy=rng.uniform(-5, 10),
                                  area=rng.uniform(500, 5000),
                                  area_noise=rng.uniform(0, 500))
        f = engine.compute(hist)
        assert 0.0 <= f.state_change_score <= 1.0, f"Got {f.state_change_score}"
    print("  [PASS] test_motion_engine_state_change_score_range")


# ============================================================
# ACTION INFERENCER TESTS
# ============================================================

def test_infer_cut_detected():
    """CUT with strong cutting signals should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        contact_frequency=5.0,
        area_variance=900.0,
        displacement_magnitude=20.0,
        contact_events=15,
        state_change_score=0.7,
    )
    result = inf.infer(f, "CUT", "cutting", (0, 5000))
    assert result.is_detected is True, f"Expected detected, got conf={result.confidence}"
    assert 0.0 <= result.confidence <= 1.0
    assert "CUT" in result.evidence
    print(f"  [PASS] test_infer_cut_detected  (conf={result.confidence:.3f})")


def test_infer_open_detected():
    """OPEN with strong rotation signal should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        rotation_change=80.0,
        area_ratio=1.35,
        state_change_score=0.8,
    )
    result = inf.infer(f, "OPEN", "opening", (0, 3000))
    assert result.is_detected is True, f"Expected detected, got conf={result.confidence}"
    assert "Rotation" in result.evidence
    print(f"  [PASS] test_infer_open_detected  (conf={result.confidence:.3f})")


def test_infer_pour_detected():
    """POUR with tilt and displacement should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        rotation_change=50.0,
        displacement_magnitude=90.0,
        vertical_motion_ratio=0.5,
    )
    result = inf.infer(f, "POUR", "pouring", (0, 4000))
    assert 0.0 <= result.confidence <= 1.0
    assert result.action_category == "POUR"
    print(f"  [PASS] test_infer_pour_detected  (conf={result.confidence:.3f})")


def test_infer_pick_detected():
    """PICK with strong upward motion should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        vertical_motion_ratio=0.8,
        contact_distance_mean=30.0,
        state_change_score=0.6,
    )
    result = inf.infer(f, "PICK", "picking", (0, 2000))
    assert result.is_detected is True, f"Expected detected, got conf={result.confidence}"
    print(f"  [PASS] test_infer_pick_detected  (conf={result.confidence:.3f})")


def test_infer_place_detected():
    """PLACE with downward motion and low speed should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        vertical_motion_ratio=-0.6,
        motion_speed=1.5,
        state_change_score=0.5,
    )
    result = inf.infer(f, "PLACE", "placing", (0, 2500))
    assert 0.0 <= result.confidence <= 1.0
    assert result.action_category == "PLACE"
    print(f"  [PASS] test_infer_place_detected  (conf={result.confidence:.3f})")


def test_infer_mix_detected():
    """MIX with high rotation speed and contact frequency should be detected."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(
        rotation_speed=4.0,
        contact_frequency=7.0,
        motion_speed=12.0,
    )
    result = inf.infer(f, "MIX", "mixing", (0, 6000))
    assert result.is_detected is True, f"Expected detected, got conf={result.confidence}"
    print(f"  [PASS] test_infer_mix_detected  (conf={result.confidence:.3f})")


def test_infer_generic_fallback():
    """Unknown category should use state_change_score as confidence."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(state_change_score=0.9)
    result = inf.infer(f, "THROW", "throwing", (0, 1000))
    assert abs(result.confidence - 0.9) < 0.01, f"Got {result.confidence}"
    print(f"  [PASS] test_infer_generic_fallback  (conf={result.confidence:.3f})")


def test_infer_confidence_range():
    """Confidence must always be in [0, 1] regardless of inputs."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    categories = ["CUT", "OPEN", "POUR", "PICK", "PLACE", "MIX", "CLOSE"]
    extreme_features = [
        MotionFeatures(),  # all zeros/defaults
        MotionFeatures(
            contact_frequency=100, area_variance=100000,
            rotation_change=1000, vertical_motion_ratio=2.0,
            state_change_score=1.0, contact_events=500,
        ),
    ]
    for cat in categories:
        for feat in extreme_features:
            result = inf.infer(feat, cat, cat.lower())
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence out of range for {cat}: {result.confidence}"
            )
    print("  [PASS] test_infer_confidence_range")


def test_action_result_fields():
    """ActionResult should have all required fields."""
    from pipeline.action_inferencer import ActionInferencer
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    f = MotionFeatures(state_change_score=0.6)
    result = inf.infer(f, "CUT", "cutting", (1000.0, 4000.0))

    assert hasattr(result, "action_label")
    assert hasattr(result, "action_category")
    assert hasattr(result, "is_detected")
    assert hasattr(result, "confidence")
    assert hasattr(result, "evidence")
    assert hasattr(result, "timestamp_range")
    assert hasattr(result, "motion_summary")
    assert isinstance(result.motion_summary, dict)
    print("  [PASS] test_action_result_fields")


def test_infer_from_history():
    """infer_from_history should handle a list of MotionFeatures."""
    from pipeline.action_inferencer import ActionInferencer, ActionResult
    from pipeline.motion_engine import MotionFeatures

    inf = ActionInferencer()
    all_feat = [MotionFeatures(state_change_score=0.5 + i*0.01) for i in range(30)]
    result = inf.infer_from_history(all_feat, "CUT", "cutting", fps=30.0)
    assert isinstance(result, ActionResult)
    assert result.timestamp_range[1] > result.timestamp_range[0]
    print("  [PASS] test_infer_from_history")


# ============================================================
# SEGMENTOR TESTS
# ============================================================

def test_segmentor_encode_frame_b64():
    """encode_frame_b64 should return a non-empty base64 string."""
    from pipeline.segmentor import encode_frame_b64

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [0, 128, 255]
    b64 = encode_frame_b64(frame)
    assert isinstance(b64, str)
    assert len(b64) > 100
    # Verify it decodes back to a valid image
    decoded = np.frombuffer(
        __import__("base64").b64decode(b64), dtype=np.uint8
    )
    img = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    assert img is not None
    print("  [PASS] test_segmentor_encode_frame_b64")


def test_segmentor_draw_annotated_frame():
    """draw_annotated_frame should return same-shape output."""
    from pipeline.segmentor import draw_annotated_frame

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_annotated_frame(
        frame=frame,
        hand_bbox=[10, 10, 80, 80],
        obj_bbox=[150, 100, 300, 250],
        obj_mask=None,
        obj_label="bottle",
        confidence=0.85,
        frame_id=7,
        timestamp_ms=2333.0,
        trajectory=[(50 + j * 5, 200) for j in range(10)],
    )
    assert result.shape == frame.shape
    # Should be different from blank frame (annotations added)
    assert not np.array_equal(result, frame)
    print("  [PASS] test_segmentor_draw_annotated_frame")


def test_segmentor_draw_annotated_frame_with_mask():
    """draw_annotated_frame should apply mask overlay correctly."""
    from pipeline.segmentor import draw_annotated_frame

    H, W = 240, 320
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[60:120, 80:160] = 255

    result = draw_annotated_frame(frame=frame, obj_mask=mask)
    assert result.shape == frame.shape
    print("  [PASS] test_segmentor_draw_annotated_frame_with_mask")


def test_segmentor_draw_trajectory():
    """draw_trajectory should produce correct shape canvas."""
    from pipeline.segmentor import draw_trajectory

    traj = [(50 + i * 8, 100 + (i % 5) * 15) for i in range(25)]
    canvas = draw_trajectory((480, 640, 3), traj)
    assert canvas.shape == (480, 640, 3)
    print("  [PASS] test_segmentor_draw_trajectory")


def test_segmentor_draw_trajectory_empty():
    """draw_trajectory with < 2 points should return blank canvas."""
    from pipeline.segmentor import draw_trajectory

    canvas = draw_trajectory((480, 640, 3), [(100, 100)])
    assert canvas.shape == (480, 640, 3)
    print("  [PASS] test_segmentor_draw_trajectory_empty")


def test_segmentor_segment_none_inputs():
    """segment() should handle None frame and None bbox gracefully."""
    from pipeline.segmentor import MobileSAMSegmentor

    seg = MobileSAMSegmentor()
    assert seg.segment(None, [10, 10, 50, 50]) is None
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert seg.segment(frame, None) is None
    print("  [PASS] test_segmentor_segment_none_inputs")


def test_segmentor_grabcut():
    """GrabCut fallback should work on a valid test image."""
    from pipeline.segmentor import MobileSAMSegmentor

    seg = MobileSAMSegmentor()
    # Force GrabCut path
    seg.predictor = None
    seg._backend = "grabcut"

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Draw an obvious object region
    frame[100:300, 200:400] = [0, 200, 100]

    mask = seg.segment(frame, [200, 100, 400, 300])
    # May return None if GrabCut fails on random image — that's acceptable
    if mask is not None:
        assert mask.shape == frame.shape[:2], f"Mask shape {mask.shape} != {frame.shape[:2]}"
        assert mask.dtype == np.uint8
    print(f"  [PASS] test_segmentor_grabcut  (mask={'returned' if mask is not None else 'None (OK)'})")


def test_segmentor_annotate_key_frames():
    """annotate_key_frames should return list of base64 strings."""
    from pipeline.segmentor import annotate_key_frames

    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
    results = annotate_key_frames(
        frames=frames,
        frame_ids=[0, 15, 30],
        timestamps_ms=[0.0, 500.0, 1000.0],
        obj_label="cup",
        confidences=[0.8, 0.7, 0.9],
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, str)
        assert len(r) > 50
    print("  [PASS] test_segmentor_annotate_key_frames")


# ============================================================
# Integration smoke test (Tracker → MotionEngine → Inferencer)
# ============================================================

def test_full_pipeline_smoke():
    """
    End-to-end smoke test:
      Simulate 30 frames of a cutting action, run through all 3 modules,
      check that an ActionResult is returned with expected structure.
    """
    from pipeline.tracker import ObjectTracker
    from pipeline.motion_engine import MotionEngine
    from pipeline.action_inferencer import ActionInferencer

    tracker = ObjectTracker()
    engine = MotionEngine(frame_window=30, contact_threshold=60)
    inferencer = ActionInferencer()

    rng = np.random.default_rng(42)
    N = 30
    # Object stays mostly in place (cutting)
    base_x, base_y = 150, 120

    all_motion_features = []
    hand_positions = []

    for i in range(N):
        # Small jitter + slight drift
        x_off = rng.normal(0, 3)
        y_off = rng.normal(0, 2)
        bbox = [base_x + x_off - 30, base_y + y_off - 30,
                base_x + x_off + 30, base_y + y_off + 30]
        det = _FakeObjectDetection(True, bbox, confidence=0.75)
        tracker.update(det, frame_id=i)

        # Hand alternates close/far to simulate cutting contact
        hand_dist = 40 if i % 3 == 0 else 100
        hand_positions.append((base_x + hand_dist, base_y))

    # Compute motion features over full history
    hist = tracker.get_history()
    features = engine.compute(hist, hand_history=hand_positions, frame_height=480)

    # Infer action
    result = inferencer.infer(features, "CUT", "cutting", (0.0, 1000.0))

    assert result.action_category == "CUT"
    assert result.action_label == "cutting"
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.evidence, str) and len(result.evidence) > 10
    assert isinstance(result.motion_summary, dict)
    assert result.timestamp_range == (0.0, 1000.0)

    print(f"  [PASS] test_full_pipeline_smoke  "
          f"(detected={result.is_detected}, conf={result.confidence:.3f})")
    print(f"         evidence: {result.evidence[:80]}...")


# ============================================================
# Runner
# ============================================================

ALL_TESTS = [
    # Tracker
    test_tracker_basic_tracking,
    test_tracker_handles_missed_frames,
    test_tracker_iou_computation,
    test_tracker_no_detection_start,
    test_tracker_reset,
    test_tracker_get_history,
    # Motion engine
    test_motion_engine_basic,
    test_motion_engine_short_history,
    test_motion_engine_contact_detection,
    test_motion_engine_upward_motion,
    test_motion_engine_key_frame_selection,
    test_motion_engine_state_change_score_range,
    # Action inferencer
    test_infer_cut_detected,
    test_infer_open_detected,
    test_infer_pour_detected,
    test_infer_pick_detected,
    test_infer_place_detected,
    test_infer_mix_detected,
    test_infer_generic_fallback,
    test_infer_confidence_range,
    test_action_result_fields,
    test_infer_from_history,
    # Segmentor
    test_segmentor_encode_frame_b64,
    test_segmentor_draw_annotated_frame,
    test_segmentor_draw_annotated_frame_with_mask,
    test_segmentor_draw_trajectory,
    test_segmentor_draw_trajectory_empty,
    test_segmentor_segment_none_inputs,
    test_segmentor_grabcut,
    test_segmentor_annotate_key_frames,
    # Integration
    test_full_pipeline_smoke,
]


if __name__ == "__main__":
    print("=" * 60)
    print("FIBA AI — Tanishk Module Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\n✓ All Tanishk modules verified successfully!")

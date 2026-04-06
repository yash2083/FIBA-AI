"""
Tests for Yash's integration modules.
Tests the integrator pipeline and Flask app routes.
"""

import os
import sys
import json
import numpy as np
import pytest

# Ensure pipeline is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test PipelineResult dataclass
# ---------------------------------------------------------------------------

from pipeline.integrator import PipelineResult


def test_pipeline_result_defaults():
    """PipelineResult should initialize with sensible defaults."""
    r = PipelineResult(success=False)
    assert r.success is False
    assert r.action_detected is False
    assert r.confidence == 0.0
    assert r.key_frames_b64 == []
    assert r.error == ""


def test_pipeline_result_populated():
    """PipelineResult should hold all fields."""
    r = PipelineResult(
        success=True,
        action_detected=True,
        action_label="cutting",
        action_category="CUT",
        confidence=0.87,
        timestamp_range=(100.0, 5000.0),
        evidence="Tool-object contact repeated 12x",
        key_frames_b64=["abc", "def"],
        trajectory_b64="xyz",
        motion_summary={"rotation_deg": 15.0},
        query_info={"raw": "cutting onion"},
        total_frames=150,
        fps=30.0,
    )
    assert r.action_detected is True
    assert r.confidence == 0.87
    assert len(r.key_frames_b64) == 2


# ---------------------------------------------------------------------------
# Test FIBAPipeline init
# ---------------------------------------------------------------------------

def test_pipeline_import():
    """FIBAPipeline class should be importable."""
    from pipeline.integrator import FIBAPipeline
    assert FIBAPipeline is not None


# ---------------------------------------------------------------------------
# Test Flask app routes (unit-level, no actual pipeline run)
# ---------------------------------------------------------------------------

def test_app_import():
    """Flask app module should be importable."""
    import app as flask_app
    assert flask_app.app is not None


def test_app_index_route():
    """GET / should return 200."""
    import app as flask_app
    client = flask_app.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"FIBA" in resp.data


def test_app_process_no_video():
    """POST /api/process without video should return 400."""
    import app as flask_app
    client = flask_app.app.test_client()
    resp = client.post("/api/process", data={"query": "cutting onion"})
    assert resp.status_code == 400


def test_app_process_no_query():
    """POST /api/process without query should return 400."""
    import app as flask_app
    client = flask_app.app.test_client()
    # Send a dummy file but no query
    from io import BytesIO
    data = {"video": (BytesIO(b"fake_video_data"), "test.mp4")}
    resp = client.post(
        "/api/process",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_app_status_not_found():
    """GET /api/status/<nonexistent_id> should return 404."""
    import app as flask_app
    client = flask_app.app.test_client()
    resp = client.get("/api/status/nonexistent123")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Integration: interface contract validation
# ---------------------------------------------------------------------------

def test_interface_contracts():
    """Verify that all inter-module interfaces match expected contracts."""
    from pipeline.query_parser import QueryResult, parse_query
    from pipeline.hand_detector import HandDetectionResult
    from pipeline.object_detector import ObjectDetectionResult
    from pipeline.tracker import TrackResult
    from pipeline.motion_engine import MotionFeatures
    from pipeline.action_inferencer import ActionResult

    # QueryResult fields
    qr = parse_query("cutting onion")
    assert hasattr(qr, "raw_query")
    assert hasattr(qr, "action_verb")
    assert hasattr(qr, "action_category")
    assert hasattr(qr, "object_noun")
    assert hasattr(qr, "tool_noun")

    # HandDetectionResult fields
    hdr = HandDetectionResult(detected=False)
    assert hasattr(hdr, "hand_bbox")
    assert hasattr(hdr, "wrist_pos")

    # ObjectDetectionResult fields
    odr = ObjectDetectionResult(detected=False)
    assert hasattr(odr, "object_bbox")
    assert hasattr(odr, "center")
    assert hasattr(odr, "area")

    # TrackResult fields
    tr = TrackResult(tracked=False)
    assert hasattr(tr, "trajectory")
    assert hasattr(tr, "bbox_history")

    # MotionFeatures fields
    mf = MotionFeatures()
    assert hasattr(mf, "displacement_magnitude")
    assert hasattr(mf, "state_change_score")

    # ActionResult fields
    ar = ActionResult(
        action_label="cutting",
        action_category="CUT",
        is_detected=True,
        confidence=0.5,
        evidence="test",
        timestamp_range=(0, 1000),
    )
    assert hasattr(ar, "motion_summary")
    assert hasattr(ar, "trajectory")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
FIBA AI — Core Pipeline Package
================================

Module ownership:
  Atul:    query_parser, hand_detector, object_detector  (Stage 1-3)
  Tanishk: tracker, motion_engine, action_inferencer, segmentor  (Stage 4-6b)
  Yash:    integrator  (Orchestration + API)
"""

# --- Atul's modules (Stage 1-3) ---
from .query_parser import QueryResult, parse_query
from .hand_detector import HandDetectionResult, HandDetector
from .object_detector import ObjectDetectionResult, ObjectDetector

# --- Tanishk's modules (Stage 4-6b) ---
from .tracker import TrackResult, ObjectTracker
from .motion_engine import MotionFeatures, MotionEngine
from .action_inferencer import ActionResult, ActionInferencer
from .segmentor import (
    MobileSAMSegmentor,
    encode_frame_b64,
    draw_annotated_frame,
    draw_trajectory,
    annotate_key_frames,
)

# --- Yash's module (Orchestration) ---
from .integrator import FIBAPipeline, PipelineResult

__all__ = [
    # Atul
    "QueryResult",
    "parse_query",
    "HandDetectionResult",
    "HandDetector",
    "ObjectDetectionResult",
    "ObjectDetector",
    # Tanishk
    "TrackResult",
    "ObjectTracker",
    "MotionFeatures",
    "MotionEngine",
    "ActionResult",
    "ActionInferencer",
    "MobileSAMSegmentor",
    "encode_frame_b64",
    "draw_annotated_frame",
    "draw_trajectory",
    "annotate_key_frames",
    # Yash
    "FIBAPipeline",
    "PipelineResult",
]

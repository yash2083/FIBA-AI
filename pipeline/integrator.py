"""
Integrator — Yash
==================
Orchestrates the full FIBA AI pipeline end-to-end.
Connects Atul's detectors, Tanishk's motion/tracking engine,
and assembles the final result for the Flask API.

Pipeline stages:
  1. Parse query  (Atul: query_parser)
  2. Init detectors  (Atul: hand_detector, object_detector)
  3. Open video + frame loop
  4. Per-frame: hand detect → object detect → track → motion features
  5. Action inference  (Tanishk: action_inferencer)
  6. Key frame selection + segmentation  (Tanishk: segmentor)
  7. Trajectory visualisation  (Tanishk: segmentor.draw_trajectory)
  8. Assemble PipelineResult
"""

import cv2
import numpy as np
import time
import traceback
from typing import Optional, Callable, List
from dataclasses import dataclass, field

# --- Atul's modules ---
from pipeline.query_parser import parse_query, QueryResult
from pipeline.hand_detector import HandDetector
from pipeline.object_detector import ObjectDetector

# --- Tanishk's modules ---
from pipeline.tracker import ObjectTracker
from pipeline.motion_engine import MotionEngine, MotionFeatures
from pipeline.action_inferencer import ActionInferencer
from pipeline.segmentor import (
    MobileSAMSegmentor,
    draw_annotated_frame,
    draw_trajectory,
    encode_frame_b64,
    annotate_key_frames,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Final output of the full FIBA AI pipeline."""
    success: bool
    action_detected: bool = False
    action_label: str = ""
    action_category: str = ""
    confidence: float = 0.0
    timestamp_range: tuple = (0.0, 0.0)
    evidence: str = ""
    key_frames_b64: list = field(default_factory=list)
    trajectory_b64: str = ""
    motion_summary: dict = field(default_factory=dict)
    query_info: dict = field(default_factory=dict)
    total_frames: int = 0
    fps: float = 0.0
    processing_time_s: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FIBAPipeline:
    """
    Full FIBA AI pipeline runner.

    Usage:
        pipeline = FIBAPipeline()
        result = pipeline.run("video.mp4", "cutting onion", progress_cb)
    """

    def __init__(self):
        self.hand_detector = HandDetector(min_detection_confidence=0.6)
        self.segmentor = MobileSAMSegmentor()
        self.motion_engine = MotionEngine(frame_window=30)
        self.action_inferencer = ActionInferencer()

    def run(
        self,
        video_path: str,
        query_text: str,
        progress_cb: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a video file.

        Args:
            video_path:   path to input video
            query_text:   natural language query e.g. "cutting onion"
            progress_cb:  optional callback(pct: int, msg: str)

        Returns:
            PipelineResult with all outputs
        """
        t0 = time.time()

        def progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        try:
            # ==================== STAGE 1: Parse query ====================
            progress(5, "Parsing query...")
            query = parse_query(query_text)

            # ==================== STAGE 2: Init per-video components ======
            progress(10, f"Loading detector for '{query.object_noun}'...")
            obj_detector = ObjectDetector(query.object_noun)
            tracker = ObjectTracker()

            # ==================== STAGE 3: Open video =====================
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return PipelineResult(
                    success=False,
                    error=f"Cannot open video: {video_path}",
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            progress(15, f"Processing {total_frames} frames at {fps:.1f} FPS...")

            # ==================== STAGE 4: Frame-by-frame =================
            all_frames: List[np.ndarray] = []
            all_hand_results = []
            all_obj_results = []
            all_track_results = []
            hand_wrist_history: List[Optional[np.ndarray]] = []
            motion_features_per_sample: List[MotionFeatures] = []

            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize for speed (max 640px width)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))

                # --- Atul's modules ---
                hand_result = self.hand_detector.detect(frame)
                obj_result = obj_detector.detect(frame, hand_result)

                # --- Tanishk's tracker ---
                track_result = tracker.update(obj_result, frame_id)

                # Store everything
                all_frames.append(frame.copy())
                all_hand_results.append(hand_result)
                all_obj_results.append(obj_result)
                all_track_results.append(track_result)

                # Hand wrist for motion engine
                wrist = hand_result.wrist_pos if hand_result.detected else None
                hand_wrist_history.append(
                    np.array(wrist, dtype=float) if wrist else None
                )

                # Sample motion features every 5 frames for efficiency
                if frame_id % 5 == 0 and len(tracker.center_history) >= 3:
                    mf = self.motion_engine.compute(
                        tracker.get_history(),
                        hand_wrist_history,
                        frame_height=frame.shape[0],
                    )
                    motion_features_per_sample.append(mf)

                frame_id += 1

                # Progress update every 30 frames
                if frame_id % 30 == 0:
                    pct = 15 + int((frame_id / max(total_frames, 1)) * 55)
                    progress(min(pct, 70), f"Processed {frame_id}/{total_frames} frames...")

            cap.release()

            if not all_frames:
                return PipelineResult(success=False, error="No frames could be read from video")

            progress(72, "Analyzing motion and inferring action...")

            # ==================== STAGE 5: Action inference ================
            # Compute final aggregate features over full track history
            final_features = self.motion_engine.compute(
                tracker.get_history(),
                hand_wrist_history,
                frame_height=all_frames[0].shape[0],
            )

            video_duration_ms = (len(all_frames) / fps) * 1000.0

            action_result = self.action_inferencer.infer(
                features=final_features,
                action_category=query.action_category,
                action_verb=query.action_verb,
                timestamps=(0.0, video_duration_ms),
            )
            # Attach trajectory for rendering
            action_result.trajectory = list(tracker.center_history)

            progress(78, "Selecting key frames...")

            # ==================== STAGE 6: Key frame selection =============
            key_indices = self.motion_engine.select_key_frame_indices(
                motion_features_per_sample, n=3
            )

            # Map from motion-sample indices to actual frame indices
            actual_key_indices = [
                min(i * 5, len(all_frames) - 1) for i in key_indices
            ]
            # Fallback: evenly spaced if nothing selected
            if not actual_key_indices:
                actual_key_indices = [
                    0,
                    len(all_frames) // 2,
                    len(all_frames) - 1,
                ]
            # Ensure exactly 3, deduplicate
            actual_key_indices = sorted(set(actual_key_indices))[:3]
            while len(actual_key_indices) < 3 and len(all_frames) > 0:
                # Pad with last frame
                actual_key_indices.append(len(all_frames) - 1)

            progress(82, "Segmenting key frames...")

            # ==================== STAGE 7: Segment + annotate key frames ===
            key_frames_raw = [all_frames[ki] for ki in actual_key_indices]
            key_obj_bboxes = [
                all_obj_results[ki].object_bbox
                if all_obj_results[ki].detected else None
                for ki in actual_key_indices
            ]
            key_hand_bboxes = [
                all_hand_results[ki].hand_bbox
                if all_hand_results[ki].detected else None
                for ki in actual_key_indices
            ]
            key_timestamps = [(ki / fps) * 1000.0 for ki in actual_key_indices]

            # Segment each key frame
            masks = []
            for kf, bbox in zip(key_frames_raw, key_obj_bboxes):
                mask = self.segmentor.segment(kf, bbox) if bbox else None
                masks.append(mask)

            # Annotate and encode
            key_frames_b64 = annotate_key_frames(
                frames=key_frames_raw,
                frame_ids=actual_key_indices,
                timestamps_ms=key_timestamps,
                object_bboxes=key_obj_bboxes,
                hand_bboxes=key_hand_bboxes,
                masks=masks,
                obj_label=query.object_noun,
                confidences=[action_result.confidence] * len(key_frames_raw),
                trajectory=list(tracker.center_history),
                quality=85,
            )

            progress(90, "Building trajectory visualization...")

            # ==================== STAGE 8: Trajectory ====================
            traj_canvas = draw_trajectory(
                frame_shape=all_frames[0].shape,
                trajectory=list(tracker.center_history),
            )
            traj_b64 = encode_frame_b64(traj_canvas)

            elapsed = time.time() - t0
            progress(100, f"Done! ({elapsed:.1f}s)")

            return PipelineResult(
                success=True,
                action_detected=action_result.is_detected,
                action_label=action_result.action_label,
                action_category=action_result.action_category,
                confidence=action_result.confidence,
                timestamp_range=action_result.timestamp_range,
                evidence=action_result.evidence,
                key_frames_b64=key_frames_b64,
                trajectory_b64=traj_b64,
                motion_summary=action_result.motion_summary,
                query_info={
                    "raw": query.raw_query,
                    "verb": query.action_verb,
                    "category": query.action_category,
                    "object": query.object_noun,
                    "tool": query.tool_noun,
                },
                total_frames=len(all_frames),
                fps=fps,
                processing_time_s=round(elapsed, 2),
            )

        except Exception as e:
            return PipelineResult(
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

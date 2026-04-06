"""
Integrator — Yash
==================
Orchestrates the full FIBA AI pipeline end-to-end.

LEAN version: no CLIP, no clip extraction, no RAG cache.
Focus: speed + accuracy + edge-readiness.

Optimizations for speed:
  - Adaptive frame sampling: skip frames on long videos
  - Single-pass processing (no double-read)
  - Hand detection runs at lower resolution when possible
"""

import cv2
import numpy as np
import time
import traceback
from typing import Optional, Callable, List
from dataclasses import dataclass, field

from pipeline.query_parser import parse_query, QueryResult
from pipeline.hand_detector import HandDetector
from pipeline.object_detector import ObjectDetector
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


@dataclass
class PipelineResult:
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
    action_description: str = ""
    # Edge deployment stats
    edge_stats: dict = field(default_factory=dict)


def _generate_description(query_info, motion_summary, confidence, detected):
    verb = query_info.get("verb", "unknown")
    obj = query_info.get("object", "object")
    cat = query_info.get("category", "")

    if not detected:
        return f"The action '{verb} {obj}' was not clearly detected in the video."

    conf_text = ("high" if confidence >= 0.7 else
                 "moderate" if confidence >= 0.5 else "low")

    details = []
    disp = motion_summary.get("displacement_px", 0)
    contact = motion_summary.get("contact_events", 0)
    rotation = motion_summary.get("rotation_deg", 0)
    grasp = motion_summary.get("grasp_change", 0)
    area_growth = motion_summary.get("area_growth_trend", 0)

    if cat == "PICK":
        if contact > 3: details.append(f"hand contacted {obj} ({contact}× events)")
        if grasp < -0.1: details.append("grasping motion observed")
        if area_growth > 0.1: details.append(f"{obj} approached camera")
        if disp > 20: details.append(f"displaced {disp:.0f}px")
    elif cat == "POUR":
        if abs(rotation) > 15: details.append(f"container tilted {abs(rotation):.0f}°")
        if disp > 20: details.append(f"moved {disp:.0f}px")
    elif cat == "CUT":
        if contact > 3: details.append(f"repeated contact ({contact}×)")
        if motion_summary.get("contact_frequency", 0) > 2:
            details.append("oscillatory cutting pattern")
    elif cat == "OPEN":
        if abs(rotation) > 20: details.append(f"rotated {abs(rotation):.0f}°")
        if motion_summary.get("area_change_ratio", 1) > 1.1:
            details.append("object expanded (opening)")
    elif cat == "MIX":
        if motion_summary.get("contact_frequency", 0) > 2:
            details.append("circular stirring motion")

    detail_str = " — " + "; ".join(details) if details else ""
    return f"**{verb.capitalize()} {obj}** detected with {conf_text} confidence ({confidence:.0%}){detail_str}."


class FIBAPipeline:
    """Lean FIBA AI pipeline. Fast single-pass processing."""

    def __init__(self):
        self.hand_detector = HandDetector(min_detection_confidence=0.5)
        self.segmentor = MobileSAMSegmentor()
        self.motion_engine = MotionEngine(frame_window=120, contact_threshold=150)
        self.action_inferencer = ActionInferencer()

    def run(
        self, video_path: str, query_text: str,
        progress_cb: Optional[Callable] = None,
    ) -> PipelineResult:
        t0 = time.time()
        mem_before = 0

        def progress(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        try:
            # ===== STAGE 1: Parse query =====
            progress(5, "Parsing query...")
            query = parse_query(query_text)

            # ===== STAGE 2: Open video =====
            progress(8, f"Loading detector for '{query.object_noun}'...")
            obj_detector = ObjectDetector(query.object_noun)
            tracker = ObjectTracker()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return PipelineResult(success=False, error=f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Adaptive frame skip: process every Nth frame for speed
            # Short videos (<150 frames): process every frame
            # Medium (150-500): every 2nd frame
            # Long (500+): every 3rd frame
            if total_frames <= 150:
                frame_skip = 1
            elif total_frames <= 500:
                frame_skip = 2
            else:
                frame_skip = 3

            self.motion_engine.frame_window = max(120, total_frames // frame_skip)

            progress(12, f"Processing {total_frames} frames (skip={frame_skip})...")

            # ===== STAGE 3: Single-pass frame processing =====
            all_frames = []
            all_hand_results = []
            all_obj_results = []
            hand_contact_history = []
            grasp_history = []
            motion_features_per_sample = []
            processed_count = 0

            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for speed
                if frame_id % frame_skip != 0:
                    frame_id += 1
                    continue

                # Resize for speed (max 640px)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))

                # Hand detection
                hand_result = self.hand_detector.detect(frame)

                # Object detection
                obj_result = obj_detector.detect(frame, hand_result, query.action_category)

                # Tracking
                track_result = tracker.update(obj_result, processed_count)

                # Store
                all_frames.append(frame.copy())
                all_hand_results.append(hand_result)
                all_obj_results.append(obj_result)

                # Hand contact point
                if hand_result.detected:
                    ftip = getattr(hand_result, "fingertip_center", None)
                    wrist = hand_result.wrist_pos
                    pt = ftip if ftip else wrist
                    hand_contact_history.append(
                        np.array(pt, dtype=float) if pt else None
                    )
                    grasp_history.append(getattr(hand_result, "grasp_openness", None))
                else:
                    hand_contact_history.append(None)
                    grasp_history.append(None)

                # Motion features every 3 processed frames
                if processed_count % 3 == 0 and len(tracker.center_history) >= 3:
                    mf = self.motion_engine.compute(
                        tracker.get_history(),
                        hand_contact_history,
                        frame_height=frame.shape[0],
                        grasp_history=grasp_history,
                    )
                    motion_features_per_sample.append(mf)

                processed_count += 1
                frame_id += 1

                if processed_count % 15 == 0:
                    pct = 12 + int((processed_count / max(total_frames // frame_skip, 1)) * 50)
                    progress(min(pct, 62), f"Frame {processed_count}/{total_frames // frame_skip}...")

            cap.release()

            if not all_frames:
                return PipelineResult(success=False, error="No frames read from video")

            t_frames = time.time()

            progress(64, "Computing action inference...")

            # ===== STAGE 4: Action inference =====
            final_features = self.motion_engine.compute(
                tracker.get_history(),
                hand_contact_history,
                frame_height=all_frames[0].shape[0],
                grasp_history=grasp_history,
            )

            video_duration_ms = (total_frames / fps) * 1000.0

            action_result = self.action_inferencer.infer(
                features=final_features,
                action_category=query.action_category,
                action_verb=query.action_verb,
                timestamps=(0.0, video_duration_ms),
            )

            # Multi-frame aggregation
            if motion_features_per_sample:
                agg_result = self.action_inferencer.infer_from_history(
                    all_features=motion_features_per_sample,
                    action_category=query.action_category,
                    action_verb=query.action_verb,
                    fps=fps,
                )
                if agg_result.confidence > action_result.confidence:
                    action_result = agg_result
                    action_result.timestamp_range = (0.0, video_duration_ms)

            action_result.trajectory = list(tracker.center_history)

            progress(72, "Selecting key frames...")

            # ===== STAGE 5: Key frames =====
            key_indices = self.motion_engine.select_key_frame_indices(
                motion_features_per_sample, n=3
            )
            actual_key_indices = [min(i * 3, len(all_frames) - 1) for i in key_indices]
            if not actual_key_indices:
                actual_key_indices = [0, len(all_frames) // 2, len(all_frames) - 1]
            actual_key_indices = sorted(set(actual_key_indices))[:3]
            while len(actual_key_indices) < 3 and all_frames:
                actual_key_indices.append(len(all_frames) - 1)

            progress(78, "Segmenting key frames...")

            # ===== STAGE 6: Segment + annotate =====
            key_frames_raw = [all_frames[ki] for ki in actual_key_indices]
            key_obj_bboxes = [
                all_obj_results[ki].object_bbox if all_obj_results[ki].detected else None
                for ki in actual_key_indices
            ]
            key_hand_bboxes = [
                all_hand_results[ki].hand_bbox if all_hand_results[ki].detected else None
                for ki in actual_key_indices
            ]
            key_timestamps = [(ki * frame_skip / fps) * 1000.0 for ki in actual_key_indices]

            masks = []
            for kf, bbox in zip(key_frames_raw, key_obj_bboxes):
                masks.append(self.segmentor.segment(kf, bbox) if bbox else None)

            key_frames_b64 = annotate_key_frames(
                frames=key_frames_raw, frame_ids=actual_key_indices,
                timestamps_ms=key_timestamps, object_bboxes=key_obj_bboxes,
                hand_bboxes=key_hand_bboxes, masks=masks,
                obj_label=query.object_noun,
                confidences=[action_result.confidence] * len(key_frames_raw),
                trajectory=list(tracker.center_history), quality=85,
            )

            progress(88, "Building trajectory...")

            traj_canvas = draw_trajectory(
                frame_shape=all_frames[0].shape,
                trajectory=list(tracker.center_history),
            )
            traj_b64 = encode_frame_b64(traj_canvas)

            elapsed = time.time() - t0
            frame_process_time = t_frames - t0

            # ===== STAGE 7: Build result =====
            query_info = {
                "raw": query.raw_query, "verb": query.action_verb,
                "category": query.action_category, "object": query.object_noun,
                "tool": query.tool_noun,
            }

            description = _generate_description(
                query_info, action_result.motion_summary,
                action_result.confidence, action_result.is_detected,
            )

            # Edge deployment stats
            edge_stats = {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "frame_skip": frame_skip,
                "effective_fps": round(processed_count / max(frame_process_time, 0.01), 1),
                "pipeline_latency_s": round(elapsed, 2),
                "frame_processing_s": round(frame_process_time, 2),
                "inference_latency_s": round(elapsed - frame_process_time, 2),
                "resolution": f"{all_frames[0].shape[1]}×{all_frames[0].shape[0]}",
                "models_used": "YOLOv8n + MediaPipe Hands",
                "edge_ready": True,
                "zero_shot": True,
            }

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
                query_info=query_info,
                total_frames=total_frames,
                fps=fps,
                processing_time_s=round(elapsed, 2),
                action_description=description,
                edge_stats=edge_stats,
            )

        except Exception as e:
            return PipelineResult(success=False, error=f"{e}\n{traceback.format_exc()}")

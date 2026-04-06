"""
Clip Extractor — Short action clip generation
===============================================
Extracts a 2-5 second video clip around the detected action peak,
encodes it as an MP4 in memory, and returns base64 for web display.

Edge-friendly: pure OpenCV, no heavy dependencies.

Features:
  - Auto-detects the action peak window from motion features
  - Draws bounding boxes and trajectory overlay on the clip
  - Encodes as H.264 MP4 for universal browser playback
  - Falls back to MJPEG if H.264 codec unavailable
"""

import base64
import os
import cv2
import numpy as np
import tempfile
from typing import List, Optional, Tuple

from pipeline.motion_engine import MotionFeatures


def find_action_window(
    motion_samples: List[MotionFeatures],
    sample_interval: int = 3,
    clip_duration_frames: int = 90,
    fps: float = 30.0,
) -> Tuple[int, int]:
    """
    Find the best window of frames containing the action peak.

    Args:
        motion_samples:       Per-sample MotionFeatures
        sample_interval:      Frames between each sample (default 3)
        clip_duration_frames: Desired clip length in frames
        fps:                  Video FPS

    Returns:
        (start_frame, end_frame) indices into the original frame list
    """
    if not motion_samples:
        return (0, clip_duration_frames)

    # Score each sample by activity level
    scores = []
    for i, f in enumerate(motion_samples):
        activity = (
            f.displacement_magnitude * 1.0
            + abs(f.rotation_change) * 2.0
            + f.contact_events * 8.0
            + f.state_change_score * 80.0
            + f.approach_score * 40.0
            + abs(f.grasp_change) * 25.0
            + f.area_variance * 0.01
        )
        scores.append(activity)

    scores = np.array(scores)

    # Find peak activity sample
    peak_sample = int(np.argmax(scores))
    peak_frame = peak_sample * sample_interval

    # Center the clip around the peak
    half_clip = clip_duration_frames // 2
    start_frame = max(0, peak_frame - half_clip)
    end_frame = start_frame + clip_duration_frames

    return (start_frame, end_frame)


def extract_action_clip(
    all_frames: List[np.ndarray],
    start_frame: int,
    end_frame: int,
    fps: float = 30.0,
    object_bboxes: Optional[List[Optional[List[float]]]] = None,
    hand_bboxes: Optional[List[Optional[List[float]]]] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    object_label: str = "",
    confidence: float = 0.0,
    max_clip_seconds: float = 5.0,
) -> Optional[str]:
    """
    Extract and encode a short action clip from video frames.

    Args:
        all_frames:    Complete list of video frames
        start_frame:   First frame index
        end_frame:     Last frame index
        fps:           Video FPS
        object_bboxes: Per-frame object bounding boxes (optional)
        hand_bboxes:   Per-frame hand bounding boxes (optional)
        trajectory:    Full object trajectory (optional)
        object_label:  Label to display on bbox
        confidence:    Detection confidence to display
        max_clip_seconds: Maximum clip length

    Returns:
        Base64-encoded MP4 string, or None if failed
    """
    if not all_frames:
        return None

    # Clamp frames
    total = len(all_frames)
    start_frame = max(0, min(start_frame, total - 1))
    max_frames = int(max_clip_seconds * fps)
    end_frame = min(end_frame, total, start_frame + max_frames)

    if end_frame <= start_frame:
        end_frame = min(start_frame + int(2 * fps), total)  # at least 2 seconds

    clip_frames = list(range(start_frame, end_frame))
    if not clip_frames:
        return None

    # Get frame dimensions
    h, w = all_frames[0].shape[:2]

    # Try different codec combinations for maximum compatibility
    codecs = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    tmp_path = None
    success = False

    for codec_str, ext in codecs:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)

            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

            if not writer.isOpened():
                os.unlink(tmp_path)
                continue

            # Write annotated frames
            for frame_idx in clip_frames:
                frame = all_frames[frame_idx].copy()

                # Draw trajectory trail (faded)
                if trajectory and len(trajectory) > 1:
                    # Draw trajectory up to current frame
                    trail_end = min(frame_idx + 1, len(trajectory))
                    trail_start = max(0, trail_end - 30)  # last 30 points
                    for j in range(trail_start, trail_end - 1):
                        alpha = (j - trail_start) / max(trail_end - trail_start, 1)
                        color = (
                            int(50 + alpha * 200),     # B
                            int(200 - alpha * 100),    # G
                            int(50 + alpha * 50),      # R
                        )
                        thickness = max(1, int(alpha * 3))
                        pt1 = (int(trajectory[j][0]), int(trajectory[j][1]))
                        pt2 = (int(trajectory[j+1][0]), int(trajectory[j+1][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)

                # Draw object bbox
                if object_bboxes and frame_idx < len(object_bboxes):
                    bbox = object_bboxes[frame_idx]
                    if bbox is not None:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        label = f"{object_label} {confidence:.0%}" if object_label else ""
                        if label:
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 255), -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw hand bbox
                if hand_bboxes and frame_idx < len(hand_bboxes):
                    hbbox = hand_bboxes[frame_idx]
                    if hbbox is not None:
                        hx1, hy1, hx2, hy2 = [int(v) for v in hbbox]
                        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 220, 0), 1)

                # Timestamp overlay
                ts = frame_idx / max(fps, 1)
                ts_text = f"{ts:.1f}s"
                cv2.putText(frame, ts_text, (w - 80, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, ts_text, (w - 80, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # "FIBA AI" watermark
                cv2.putText(frame, "FIBA AI", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                writer.write(frame)

            writer.release()

            # Read the file and encode
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    video_bytes = f.read()
                os.unlink(tmp_path)
                if len(video_bytes) > 100:  # sanity check
                    success = True
                    return base64.b64encode(video_bytes).decode("utf-8")

            os.unlink(tmp_path)

        except Exception as e:
            print(f"[ClipExtractor] Codec {codec_str} failed: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return None


def get_clip_mime_type() -> str:
    """Return the expected MIME type for the generated clip."""
    return "video/mp4"


if __name__ == "__main__":
    print("=== Clip Extractor Test ===")

    # Create some fake frames with a moving box
    frames = []
    for i in range(90):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 50)  # dark background
        x = 100 + i * 4
        y = 240 - abs(i - 45) * 2
        cv2.rectangle(frame, (x, y), (x + 60, y + 40), (0, 120, 255), -1)
        cv2.putText(frame, f"Frame {i}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        frames.append(frame)

    # Test clip extraction
    clip_b64 = extract_action_clip(
        frames, 20, 70, fps=30.0,
        object_label="test object", confidence=0.75,
    )

    if clip_b64:
        print(f"  Clip generated: {len(clip_b64)} chars base64")
        print(f"  Approx size: {len(clip_b64) * 3 / 4 / 1024:.1f} KB")
        print("  ClipExtractor test PASSED ✓")
    else:
        print("  WARNING: Clip generation failed (codec issue?)")
        print("  The system will fall back to key frames only.")

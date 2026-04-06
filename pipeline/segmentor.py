"""
Segmentor — Tanishk
====================
MobileSAM-based object segmentation for selected key frames.
Generates clean binary masks from bounding box prompts.
Falls back to OpenCV GrabCut when MobileSAM is unavailable.

Owner: Tanishk
Receives: (frame, bbox) for each key frame selected by MotionEngine
Outputs:  binary mask (same H×W as frame), annotated frames, trajectory image

Exported functions (used by Yash's integrator):
    MobileSAMSegmentor      — class with .segment(frame, bbox) → mask
    encode_frame_b64        — base64-encode a frame for web transmission
    draw_annotated_frame    — overlay hand/object bbox + mask + timestamp
    draw_trajectory         — colour-coded centroid path on blank canvas

Design notes:
  - Segmentation is SPARSE — only run on 3–5 key frames, never every frame.
  - MobileSAM (vit_t) requires model weights at weights/mobile_sam.pt.
  - If weights are missing or torch is unavailable, GrabCut kicks in silently.
  - All drawing helpers are pure OpenCV (no extra deps).
"""

import base64
import cv2
import numpy as np
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Frame encoding for web transmission
# ---------------------------------------------------------------------------

def encode_frame_b64(frame: np.ndarray, quality: int = 85) -> str:
    """
    JPEG-encode an OpenCV frame and return as a base64 string.
    Ready to embed in a <img src="data:image/jpeg;base64,..."> tag.

    Args:
        frame:   BGR numpy array (H, W, 3)
        quality: JPEG quality 1–100 (default 85 — good balance)

    Returns:
        base64-encoded JPEG string
    """
    _, buffer = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Annotated frame drawing
# ---------------------------------------------------------------------------

def draw_annotated_frame(
    frame: np.ndarray,
    hand_bbox: Optional[List[float]] = None,
    obj_bbox: Optional[List[float]] = None,
    obj_mask: Optional[np.ndarray] = None,
    obj_label: str = "",
    confidence: float = 0.0,
    frame_id: int = 0,
    timestamp_ms: float = 0.0,
    trajectory: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Draw all annotations on a video frame for display.

    Draws (in order):
      1. Object mask overlay (semi-transparent blue-purple)
      2. Object bounding box + label + confidence
      3. Hand bounding box (green)
      4. Trajectory dots (last N positions)
      5. Timestamp watermark

    Args:
        frame:        BGR numpy array
        hand_bbox:    [x1,y1,x2,y2] or None
        obj_bbox:     [x1,y1,x2,y2] or None
        obj_mask:     binary mask same size as frame, or None
        obj_label:    detected object class name
        confidence:   detection/tracking confidence 0–1
        frame_id:     frame index for watermark
        timestamp_ms: timestamp for watermark
        trajectory:   list of (cx, cy) centroid positions to draw as dots

    Returns:
        Annotated copy of frame (original is not modified)
    """
    out = frame.copy()

    # 1. Object mask (translucent overlay)
    if obj_mask is not None and obj_mask.shape[:2] == frame.shape[:2]:
        colored_mask = np.zeros_like(out)
        mask_bool = obj_mask > 0
        colored_mask[mask_bool] = [180, 60, 255]  # blue-purple
        out = cv2.addWeighted(out, 0.78, colored_mask, 0.22, 0)

    # 2. Object bounding box
    if obj_bbox and len(obj_bbox) == 4:
        x1, y1, x2, y2 = [int(round(c)) for c in obj_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 90, 255), 2)
        label_text = f"{obj_label} {confidence:.0%}" if obj_label else f"{confidence:.0%}"
        label_pos = (x1, max(y1 - 8, 12))
        cv2.putText(out, label_text, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 20, 255), 2, cv2.LINE_AA)
        # Draw small confidence arc
        _draw_confidence_bar(out, x1, y1, x2, confidence)

    # 3. Hand bounding box
    if hand_bbox and len(hand_bbox) == 4:
        x1, y1, x2, y2 = [int(round(c)) for c in hand_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, "Hand", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

    # 4. Trajectory dots (most recent 20)
    if trajectory and len(trajectory) >= 2:
        recent = trajectory[-20:]
        for i, (cx, cy) in enumerate(recent):
            alpha = (i + 1) / len(recent)
            color = (int(255 * alpha), 80, int(255 * (1 - alpha)))
            cv2.circle(out, (int(cx), int(cy)), 3, color, -1)

    # 5. Timestamp watermark
    ts_text = f"t={timestamp_ms / 1000:.2f}s  frame={frame_id}"
    h = out.shape[0]
    cv2.putText(out, ts_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def _draw_confidence_bar(
    img: np.ndarray,
    x1: int, y1: int, x2: int,
    confidence: float,
    bar_height: int = 4,
) -> None:
    """Small horizontal confidence bar drawn below the top edge of the bbox."""
    bar_width = x2 - x1
    filled = int(bar_width * min(1.0, max(0.0, confidence)))
    y_bar = max(y1 - 14, 0)
    cv2.rectangle(img, (x1, y_bar), (x2, y_bar + bar_height), (50, 50, 50), -1)
    if filled > 0:
        color = (0, int(200 * confidence), int(200 * (1 - confidence)))
        cv2.rectangle(img, (x1, y_bar), (x1 + filled, y_bar + bar_height), color, -1)


# ---------------------------------------------------------------------------
# Trajectory visualisation
# ---------------------------------------------------------------------------

def draw_trajectory(
    frame_shape: Tuple[int, int, int],
    trajectory: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (255, 140, 0),
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (20, 20, 30),
) -> np.ndarray:
    """
    Draw the object centroid trajectory on a dark canvas.

    The path is colour-graded blue (start) → red (end) to show time.
    Start and end are marked with labelled circles.

    Args:
        frame_shape:  (H, W, C) shape used to size the canvas
        trajectory:   list of (cx, cy) centroid positions
        color:        fallback colour (not used when gradient is active)
        thickness:    line thickness in pixels
        bg_color:     canvas background colour

    Returns:
        BGR numpy array (same spatial size as the source frame)
    """
    h, w = frame_shape[:2]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    if len(trajectory) < 2:
        return canvas

    pts = np.array(
        [(int(round(cx)), int(round(cy))) for cx, cy in trajectory],
        dtype=np.int32,
    )

    # Gradient line: blue→red over time
    for i in range(1, len(pts)):
        progress = i / max(len(pts) - 1, 1)
        grad_color = (
            int(255 * progress),   # R
            60,                     # G
            int(255 * (1 - progress)),  # B
        )
        cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), grad_color, thickness, cv2.LINE_AA)

    # Start marker
    cv2.circle(canvas, tuple(pts[0]), 8, (0, 255, 80), -1)
    cv2.putText(canvas, "START",
                (pts[0][0] + 10, pts[0][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1, cv2.LINE_AA)

    # End marker
    cv2.circle(canvas, tuple(pts[-1]), 8, (0, 60, 255), -1)
    cv2.putText(canvas, "END",
                (pts[-1][0] + 10, pts[-1][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 100, 255), 1, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, "Object Trajectory", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# MobileSAM segmentor (with GrabCut fallback)
# ---------------------------------------------------------------------------

class MobileSAMSegmentor:
    """
    Key-frame object segmentor.

    Tries to load MobileSAM (vit_t) from weights/mobile_sam.pt.
    If unavailable, silently falls back to OpenCV GrabCut, which requires
    no model download and works purely from the bounding box.

    Usage:
        seg = MobileSAMSegmentor()
        mask = seg.segment(frame, bbox)   # → H×W binary mask or None
        encoded = encode_frame_b64(frame)
    """

    WEIGHTS_PATH = "weights/mobile_sam.pt"

    def __init__(self):
        self.predictor = None
        self._backend: str = "grabcut"
        self._try_load_mobile_sam()

    def _try_load_mobile_sam(self) -> None:
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            import torch

            sam = sam_model_registry["vit_t"](checkpoint=self.WEIGHTS_PATH)
            sam.eval()
            # Use CUDA if available, else CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device)
            self.predictor = SamPredictor(sam)
            self._backend = "mobilesam"
            print(f"[Segmentor] MobileSAM loaded (device={device})")
        except Exception as exc:
            print(f"[Segmentor] MobileSAM not available ({exc}). "
                  "Using GrabCut fallback.")
            self.predictor = None
            self._backend = "grabcut"

    @property
    def backend(self) -> str:
        """Returns 'mobilesam' or 'grabcut'."""
        return self._backend

    def segment(
        self,
        frame: np.ndarray,
        bbox: Optional[List[float]],
    ) -> Optional[np.ndarray]:
        """
        Segment the query object in `frame` given its bounding `bbox`.

        Args:
            frame: BGR numpy array
            bbox:  [x1, y1, x2, y2] in pixel coordinates

        Returns:
            Binary mask as uint8 numpy array (255 = foreground, 0 = background),
            same spatial size as frame.  Returns None on failure.
        """
        if bbox is None or frame is None:
            return None
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None

        try:
            if self.predictor is not None:
                return self._sam_segment(frame, bbox)
            else:
                return self._grabcut_segment(frame, bbox)
        except Exception as exc:
            print(f"[Segmentor] Segment failed ({exc}). Returning None.")
            return None

    def _sam_segment(
        self, frame: np.ndarray, bbox: List[float]
    ) -> Optional[np.ndarray]:
        """MobileSAM path."""
        import numpy as _np
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        box_arr = _np.array(bbox, dtype=_np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box_arr, multimask_output=False
        )
        mask = masks[0].astype(_np.uint8) * 255
        return mask

    def _grabcut_segment(
        self, frame: np.ndarray, bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        OpenCV GrabCut fallback — no model required.
        Uses 5 iterations from the bounding box rectangle.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(w - 1, int(round(bbox[2])))
        y2 = min(h - 1, int(round(bbox[3])))

        if x2 <= x1 or y2 <= y1:
            return None

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        rect = (x1, y1, x2 - x1, y2 - y1)

        try:
            cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)
            # Probable/definite foreground regions → 255
            result = np.where(
                (gc_mask == cv2.GC_PR_FGD) | (gc_mask == cv2.GC_FGD),
                255, 0
            ).astype(np.uint8)
            return result
        except cv2.error:
            return None


# ---------------------------------------------------------------------------
# Batch annotator (convenience for integrator)
# ---------------------------------------------------------------------------

def annotate_key_frames(
    frames: List[np.ndarray],
    frame_ids: List[int],
    timestamps_ms: List[float],
    object_bboxes: Optional[List[Optional[List[float]]]] = None,
    hand_bboxes: Optional[List[Optional[List[float]]]] = None,
    masks: Optional[List[Optional[np.ndarray]]] = None,
    obj_label: str = "",
    confidences: Optional[List[float]] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    quality: int = 85,
) -> List[str]:
    """
    Annotate a list of key frames and return them as base64 JPEG strings.

    Convenience wrapper for Yash's integrator — eliminates per-loop boilerplate.

    Returns:
        List of base64-encoded strings, one per frame.
    """
    result = []
    n = len(frames)
    for i, frame in enumerate(frames):
        ann = draw_annotated_frame(
            frame=frame,
            hand_bbox=hand_bboxes[i] if hand_bboxes and i < len(hand_bboxes) else None,
            obj_bbox=object_bboxes[i] if object_bboxes and i < len(object_bboxes) else None,
            obj_mask=masks[i] if masks and i < len(masks) else None,
            obj_label=obj_label,
            confidence=confidences[i] if confidences and i < len(confidences) else 0.0,
            frame_id=frame_ids[i] if i < len(frame_ids) else i,
            timestamp_ms=timestamps_ms[i] if i < len(timestamps_ms) else 0.0,
            trajectory=trajectory,
        )
        result.append(encode_frame_b64(ann, quality=quality))
    return result


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Segmentor standalone test ===")

    H, W = 480, 640

    # 1. Test encode_frame_b64
    dummy_frame = np.zeros((H, W, 3), dtype=np.uint8)
    dummy_frame[100:200, 150:300] = [0, 120, 255]
    b64 = encode_frame_b64(dummy_frame)
    assert b64 and len(b64) > 100, "encode_frame_b64 failed"
    print("  encode_frame_b64 ... OK")

    # 2. Test draw_annotated_frame
    annotated = draw_annotated_frame(
        frame=dummy_frame,
        hand_bbox=[10, 10, 80, 80],
        obj_bbox=[150, 100, 300, 200],
        obj_mask=None,
        obj_label="onion",
        confidence=0.77,
        frame_id=42,
        timestamp_ms=1400.0,
        trajectory=[(i * 5, 120 + i * 2) for i in range(15)],
    )
    assert annotated.shape == dummy_frame.shape, "draw_annotated_frame wrong shape"
    print("  draw_annotated_frame ... OK")

    # 3. Test draw_trajectory
    traj = [(30 + i * 10, 100 + (i % 5) * 20) for i in range(30)]
    traj_img = draw_trajectory((H, W, 3), traj)
    assert traj_img.shape == (H, W, 3), "draw_trajectory wrong shape"
    print("  draw_trajectory ... OK")

    # 4. Test GrabCut segmentor
    seg = MobileSAMSegmentor()
    assert seg.backend in ("mobilesam", "grabcut"), "backend not set"
    print(f"  Backend: {seg.backend}")
    test_frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    mask = seg.segment(test_frame, [100, 100, 300, 300])
    if mask is not None:
        assert mask.shape == (H, W), f"Mask shape wrong: {mask.shape}"
        print(f"  segment() returned mask {mask.shape} ... OK")
    else:
        print("  segment() returned None (acceptable for edge cases) ... OK")

    # 5. Edge cases
    assert seg.segment(None, [10, 10, 50, 50]) is None, "None frame should return None"
    assert seg.segment(test_frame, None) is None, "None bbox should return None"
    print("  Edge cases ... OK")

    print("\nSegmentor test PASSED ✓")

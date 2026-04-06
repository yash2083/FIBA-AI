# 🎨 Design — Yash
## Implementation Spec: Web App + Integrator
### FIBA AI | MIT Bangalore Hitachi Hackathon

---

## Complete Source Code

### `pipeline/integrator.py`

```python
"""
Integrator — Yash
Orchestrates the full FIBA AI pipeline. Connects all modules end-to-end.
Designed for video file or webcam stream input.
"""

import cv2
import numpy as np
import os
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

from pipeline.query_parser import parse_query, QueryResult
from pipeline.hand_detector import HandDetector
from pipeline.object_detector import ObjectDetector
from pipeline.tracker import ObjectTracker
from pipeline.motion_engine import MotionEngine, MotionFeatures
from pipeline.action_inferencer import ActionInferencer
from pipeline.segmentor import (MobileSAMSegmentor, draw_annotated_frame, 
                                  draw_trajectory, encode_frame_b64)


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
    error: str = ""


class FIBAPipeline:
    """
    Full FIBA AI pipeline runner.
    Call run(video_path, query) to process a video.
    """
    
    def __init__(self):
        self.hand_detector = HandDetector(min_detection_confidence=0.6)
        self.segmentor = MobileSAMSegmentor()
        self.motion_engine = MotionEngine(frame_window=30)
        self.action_inferencer = ActionInferencer()
    
    def run(self, 
            video_path: str, 
            query_text: str,
            progress_cb: Optional[Callable] = None) -> PipelineResult:
        """
        Run the full FIBA AI pipeline on a video.
        
        Args:
            video_path: Path to input video file
            query_text: Natural language query e.g. "cutting onion"
            progress_cb: Optional callback(progress_pct, message) for UI updates
        
        Returns:
            PipelineResult with all outputs
        """
        
        def progress(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)
            print(f"[{pct:3d}%] {msg}")
        
        try:
            # --- STAGE 1: Parse Query ---
            progress(5, "Parsing query...")
            query = parse_query(query_text)
            
            # --- STAGE 2: Init per-video components ---
            progress(10, f"Loading detector for '{query.object_noun}'...")
            obj_detector = ObjectDetector(query.object_noun)
            tracker = ObjectTracker()
            
            # --- STAGE 3: Open video ---
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return PipelineResult(success=False, error=f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            progress(15, f"Processing {total_frames} frames at {fps:.1f} FPS...")
            
            # --- STAGE 4: Frame-by-frame processing ---
            all_frames = []
            all_hand_results = []
            all_obj_results = []
            all_track_results = []
            hand_wrist_history = []
            motion_features_per_frame = []
            
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp_ms = (frame_id / fps) * 1000.0
                
                # Resize for speed (max 640px width)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                
                # Atul's modules
                hand_result = self.hand_detector.detect(frame)
                obj_result = obj_detector.detect(frame, hand_result)
                
                # Tanishk's tracker
                track_result = tracker.update(obj_result, frame_id)
                
                # Store history
                all_frames.append(frame.copy())
                all_hand_results.append(hand_result)
                all_obj_results.append(obj_result)
                all_track_results.append(track_result)
                
                wrist = hand_result.wrist_pos if hand_result.detected else None
                hand_wrist_history.append(np.array(wrist) if wrist else None)
                
                # Compute motion features every 5 frames (efficiency)
                if frame_id % 5 == 0:
                    mf = self.motion_engine.compute(
                        tracker.get_history(),
                        hand_wrist_history,
                        frame_height=frame.shape[0]
                    )
                    motion_features_per_frame.append(mf)
                
                frame_id += 1
                if frame_id % 30 == 0:
                    pct = 15 + int((frame_id / max(total_frames, 1)) * 60)
                    progress(pct, f"Processed {frame_id}/{total_frames} frames...")
            
            cap.release()
            
            progress(75, "Analyzing motion and inferring action...")
            
            # --- STAGE 5: Action inference over full video ---
            if not motion_features_per_frame:
                return PipelineResult(success=False, error="No motion data collected")
            
            # Use aggregate features from all frames
            final_features = self.motion_engine.compute(
                tracker.get_history(),
                hand_wrist_history,
                frame_height=all_frames[0].shape[0] if all_frames else 480
            )
            
            action_result = self.action_inferencer.infer(
                features=final_features,
                action_category=query.action_category,
                action_verb=query.action_verb,
                timestamps=(0.0, (len(all_frames) / fps) * 1000.0)
            )
            action_result.trajectory = tracker.center_history
            
            progress(80, "Selecting key frames...")
            
            # --- STAGE 6: Select key frames ---
            key_indices = self.motion_engine.select_key_frame_indices(
                motion_features_per_frame, n=3)
            
            # Map from motion-sampled indices to actual frame indices
            actual_key_indices = [min(i * 5, len(all_frames)-1) for i in key_indices]
            if not actual_key_indices:
                actual_key_indices = [0, len(all_frames)//2, len(all_frames)-1]
            
            progress(85, "Segmenting key frames...")
            
            # --- STAGE 7: Segment key frames ---
            key_frames_b64 = []
            for ki in actual_key_indices[:3]:
                kf = all_frames[ki]
                obj_r = all_obj_results[ki]
                hand_r = all_hand_results[ki]
                
                # Segment object
                mask = None
                if obj_r.detected:
                    mask = self.segmentor.segment(kf, obj_r.object_bbox)
                
                # Draw annotations
                annotated = draw_annotated_frame(
                    frame=kf,
                    hand_bbox=hand_r.hand_bbox if hand_r.detected else None,
                    obj_bbox=obj_r.object_bbox if obj_r.detected else None,
                    obj_mask=mask,
                    obj_label=obj_r.object_label or query.object_noun,
                    confidence=action_result.confidence,
                    frame_id=ki,
                    timestamp_ms=(ki / fps) * 1000.0
                )
                key_frames_b64.append(encode_frame_b64(annotated))
            
            progress(92, "Building trajectory visualization...")
            
            # --- STAGE 8: Trajectory visualization ---
            traj_canvas = draw_trajectory(
                frame_shape=all_frames[0].shape if all_frames else (480, 640, 3),
                trajectory=tracker.center_history
            )
            traj_b64 = encode_frame_b64(traj_canvas)
            
            progress(98, "Done!")
            
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
            )
        
        except Exception as e:
            import traceback
            return PipelineResult(success=False, error=f"{str(e)}\n{traceback.format_exc()}")
```

---

### `app.py`

```python
"""
Flask Web App — Yash
Local web server for FIBA AI.
Runs entirely offline. No internet needed.
"""

import os
import uuid
import json
import threading
import time
from flask import Flask, request, jsonify, Response, send_file, render_template
from flask_cors import CORS
from pipeline.integrator import FIBAPipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store job progress and results in memory
job_registry = {}   # job_id -> {"progress": int, "message": str, "result": dict|None}

pipeline = FIBAPipeline()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_video():
    """
    Upload a video and start processing.
    Returns a job_id to poll for progress.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    
    query_text = request.form.get('query', '').strip()
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    video_file = request.files['video']
    job_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{video_file.filename}")
    video_file.save(video_path)
    
    job_registry[job_id] = {
        "progress": 0, 
        "message": "Queued...", 
        "result": None,
        "done": False,
        "error": None
    }
    
    def run_pipeline():
        def progress_cb(pct, msg):
            job_registry[job_id]["progress"] = pct
            job_registry[job_id]["message"] = msg
        
        result = pipeline.run(video_path, query_text, progress_cb)
        job_registry[job_id]["progress"] = 100
        job_registry[job_id]["done"] = True
        
        if result.success:
            job_registry[job_id]["result"] = {
                "action_detected": result.action_detected,
                "action_label": result.action_label,
                "confidence": result.confidence,
                "timestamp_range": result.timestamp_range,
                "evidence": result.evidence,
                "key_frames": result.key_frames_b64,
                "trajectory": result.trajectory_b64,
                "motion_summary": result.motion_summary,
                "query_info": result.query_info,
                "total_frames": result.total_frames,
                "fps": result.fps,
            }
        else:
            job_registry[job_id]["error"] = result.error
    
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "started"})


@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Poll for job progress."""
    if job_id not in job_registry:
        return jsonify({"error": "Job not found"}), 404
    
    job = job_registry[job_id]
    return jsonify({
        "job_id": job_id,
        "progress": job["progress"],
        "message": job["message"],
        "done": job["done"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route('/api/stream/<job_id>')
def stream_progress(job_id):
    """Server-Sent Events stream for real-time progress."""
    def event_stream():
        while True:
            if job_id not in job_registry:
                yield "data: {\"error\": \"not found\"}\n\n"
                break
            job = job_registry[job_id]
            data = json.dumps({
                "progress": job["progress"],
                "message": job["message"],
                "done": job["done"],
            })
            yield f"data: {data}\n\n"
            if job["done"]:
                break
            time.sleep(0.5)
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == '__main__':
    print("🚀 FIBA AI Server starting...")
    print("📡 Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## Test Script for Yash's Integration

```bash
# Install deps
pip install flask flask-cors

# Start server
python app.py

# In another terminal, test API
curl -X POST http://localhost:5000/api/process \
  -F "video=@test_video.mp4" \
  -F "query=cutting onion"

# Poll status
curl http://localhost:5000/api/status/<job_id>
```

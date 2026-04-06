# 🏗️ Architecture — Yash
## Module: Web App + Integration Orchestrator + Output Renderer
### FIBA AI | MIT Bangalore Hitachi Hackathon

---

## Your Role in the Pipeline

Yash owns **the web application, the integration layer, and all visual output**. You are the glue that connects Atul's detectors and Tanishk's motion engine into a seamless demo.

```
[Browser UI] ──► Upload Video + Type Query ──► [Flask Backend]
                                                      │
                                               [integrator.py]
                                                      │
                        ┌─────────────────────────────┤
                        ▼         ▼          ▼         ▼
                    Atul's    Tanishk's  Result    Video
                    modules   modules   builder   renderer
                        │         │          │         │
                        └─────────┴──────────┘─────────┘
                                      │
                              [JSON Response]
                                      │
                        [Browser: Results Panel]
                         • Key frames with bbox
                         • Object trajectory map
                         • Confidence bar
                         • Evidence text
                         • Downloadable clip
```

---

## Your 3 Responsibilities

### A. Integration Orchestrator (`pipeline/integrator.py`)
The main pipeline runner. Calls all modules in order per video.

### B. Flask Web Backend (`app.py`)
API server with 3 endpoints.

### C. Frontend (`templates/index.html` + `static/`)
The UI — clean, demo-ready, works offline.

---

## A. Integrator (`pipeline/integrator.py`)

### What it does
Orchestrates the entire frame-by-frame pipeline and assembles the final result.

### Pipeline Logic
```python
def run_pipeline(video_path, query_text):
    # 1. Parse query (Atul's module)
    query = query_parser.parse(query_text)
    
    # 2. Initialize hand detector + object detector
    hand_det = HandDetector()
    obj_det = ObjectDetector(query.object_noun)
    tracker = ObjectTracker()
    motion_engine = MotionEngine()
    
    # 3. Process video frame by frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp_ms = (frame_id / fps) * 1000
        
        # Atul's pipeline
        hand_result = hand_det.detect(frame)
        obj_result = obj_det.detect(frame, hand_result)
        
        # Tanishk's pipeline
        track_result = tracker.update(obj_result, frame_id)
        motion_result = motion_engine.compute(tracker.get_history())
        
        frame_results.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "frame": frame.copy(),
            "hand": hand_result,
            "object": obj_result,
            "track": track_result,
            "motion": motion_result,
        })
    
    cap.release()
    
    # 4. Action inference over full video
    action_result = action_inferencer.infer(
        motion_history=[r["motion"] for r in frame_results],
        query=query
    )
    
    # 5. Select key frames and segment
    key_frame_ids = action_result.key_frame_indices
    key_frames = [frame_results[i]["frame"] for i in key_frame_ids]
    masks = [segmentor.segment(f, frame_results[i]["object"]["bbox"]) 
             for f, i in zip(key_frames, key_frame_ids)]
    
    # 6. Build result
    return build_result(query, action_result, key_frames, masks, frame_results, fps)


def build_result(query, action, key_frames, masks, all_frames, fps):
    """Assemble the JSON response for the frontend"""
    # Encode key frames as base64 JPEG
    encoded_frames = [encode_frame_b64(f) for f in key_frames]
    
    # Generate trajectory visualization
    traj_img = draw_trajectory(all_frames, action.trajectory)
    
    # Clip extraction: start to end timestamp
    clip_path = extract_clip(all_frames, action.timestamp_range, fps)
    
    return {
        "success": True,
        "action_detected": action.is_detected,
        "action_label": action.action_label,
        "confidence": round(action.confidence, 3),
        "timestamp_range": action.timestamp_range,
        "evidence": action.evidence,
        "key_frames": encoded_frames,               # list of base64 strings
        "trajectory_image": encode_frame_b64(traj_img),
        "clip_path": clip_path,
        "motion_summary": action.motion_summary,
        "query": {
            "raw": query.raw_query,
            "verb": query.action_verb,
            "object": query.object_noun,
            "tool": query.tool_noun,
        }
    }
```

---

## B. Flask Backend (`app.py`)

### Endpoints

#### `POST /api/process`
Upload video + query → trigger pipeline → return results.

#### `GET /api/stream/<job_id>`
Server-Sent Events for real-time progress updates.

#### `GET /api/clip/<filename>`
Serve the extracted clip for download.

---

## C. Frontend Architecture

### Pages / Sections
1. **Hero / Input Section**: Video upload (drag-drop), query text input, Process button
2. **Progress Panel**: Live progress bar with status messages (SSE stream)
3. **Results Panel**: 
   - Action banner (DETECTED / NOT DETECTED + confidence)
   - 3 key frames side by side (with overlaid bbox + mask)
   - Trajectory map (bird's-eye path of object)
   - Evidence text (explainable AI output)
   - Motion stats (rotation, displacement, contact events)
   - Download clip button

### UI Tech Stack
- Pure HTML + CSS + Vanilla JS (no React/Node needed)
- Tailwind CSS via CDN (styling)
- Chart.js for motion graphs
- No external Python dependencies for frontend

---

## Performance Targets
| Component | Target |
|-----------|--------|
| Video upload | < 2s (local) |
| Pipeline (30s video) | < 45s total |
| UI response | < 100ms |
| Key frame render | Instant (base64) |

---

## Dev Notes for Hackathon Speed
1. Install: `pip install flask flask-cors opencv-python-headless`
2. Test with `flask run --host=0.0.0.0 --port=5000`
3. Use `threading.Thread` to run pipeline async so UI doesn't freeze
4. For SSE progress: use `yield f"data: {json.dumps(msg)}\n\n"`
5. Draw bbox on key frames with `cv2.rectangle + cv2.putText` before encoding
6. For trajectory: use `cv2.polylines` on a blank canvas sized to frame dims
7. Keep clip extraction simple: `cv2.VideoWriter` with frames from `timestamp_range`

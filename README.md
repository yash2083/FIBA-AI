# FIBA AI — Find it by Action

> **Edge-Ready · Zero-Shot Action Recognition · Explainable AI · SOP Compliance Validation**
>
> **Hackathon Domain:** AI-driven Manufacturing & Assembly Line Monitoring

FIBA AI is an advanced, computer-vision architecture designed to simultaneously solve two critical challenges on the factory floor:
1. **Action Search (Zero-Shot):** The ability to use natural language (e.g., *"Did the worker rotate the pen?"* or *"Placing the bottle"*) to search unindexed video footage for highly specific spatial interactions.
2. **SOP Compliance:** Strict, frame-by-frame validation of ordered assembly instructions, verifying that a worker follows exact standard operating procedures (SOPs).

---

## 🧠 Web App Pipeline Methodology & Architecture

The core python backend (`web_app/`) serves as the brain of the FIBA AI engine. It is completely offline-capable, highly modular, and skips black-box action foundation models in favor of an **interpretable, multi-stage modular pipeline**.

### 1. Zero-Shot Action Search Pipeline
The Action Search pipeline breaks down natural language queries into physical physics-based rules, allowing it to detect actions it has never explicitly been trained on.

* **NLP Query Parser (`query_parser.py`):**
  Uses rule-based NLP (and local LLM fallbacks) to dissect physical queries (e.g., "dipping tea bag") into structural targets: 
  * `Noun/Target`: "tea bag" (matches YOLO COCO classes and soft-aliases like "cup" or "bowl").
  * `Action Category`: "DIP"
* **Object Grounding (`object_detector.py`):**
  Powered by **Ultralytics YOLOv8n**. Object detection is enhanced via a robust string-similarity algorithm and expansive semantic aliases (e.g., "water" -> "bottle", "mobile" -> "cell phone"). It implements a multi-pass architecture: if standard detection fails, it creates a custom bounding box around the worker's hands (predicting the object's likely position).
* **Hand Kinematics (`hand_skeleton.py`):**
  Powered by **Google MediaPipe**. Reconstructs a 21-point 3D hand skeleton per frame. Extracts physical context: Is the hand grasping? What is the distance between the fingertip vector and the target object?
* **Physics & Motion Engine (`motion_engine.py`):**
  The backbone of our spatial understanding. Over 20+ math-driven metrics are computed across video frames:
  * *Displacement:* Pixel vector movement of the object.
  * *Area Growth/Shrink:* Maps to Z-axis movement (approaching or retreating from the camera).
  * *Grasp Openness:* Evaluates the distance between the thumb and index landmarks to determine grip closing events.
  * *Rotation Check:* Calculates the shifting center-of-mass angle over time.
* **Action Inferencer (`action_inferencer.py`):**
  Uses the raw kinematics from the Motion Engine to generate a confidence score out of `1.0`. For example, a **PICK** action requires: Hand proximity + Grasp closing (negative change) + Area growth (approaching camera).

### 2. Standard Operating Procedure (SOP) Validation Pipeline
To monitor strict sequence assembly routines, FIBA AI uses specialized, heavily fine-tuned classifiers.

* **The Dataset:**
  The classifier was trained on an internally generated dataset comprising **78 video cycles** equating to thousands of frames. The dataset captures strict step-by-step actions like "Screwing", "Placing white plastic part", and "Inflating the valve".
* **Fine-Tuning Architecture (`train_sop_classifier.py`):**
  We fine-tuned the **YOLOv8n-cls** (Classifier architecture) directly on the extracted frames of the 7 assembly steps. Using pre-trained ImageNet weights as a base, we froze the lower convolutional layers and trained the classification heads to map localized assembly patterns.
  * **Epochs:** 100+
  * **Optimization:** SGD with cosine learning rate scheduling.
  * Augmented extensively to manage varying lighting conditions and object occlusions.
* **Temporal Consistency (`sop_validator.py`):**
  Because individual frame classifications can be noisy (e.g., a hand blocks the camera for 3 frames), we implement a **Sliding Window Majority Vote**. This mathematically smooths predictions over a rolling frame window, ensuring the predicted SOP step is structurally stable before logging it.
* **Sequential Verification:**
  The pipeline compares the executed Steps (1 → 2 → 3) against a strict logical reference. Deviations or skipped steps instantly trigger an SOP violation alert.

---

## 🛠️ Tools & Tech Stack

**Web/Backend:**
* **Flask:** Asynchronous server acting through Server-Sent Events (SSE) to stream live progress to the frontend.
* **PyTorch & Ultralytics:** Core backbone for fine-tuning YOLOv8 object and classification models.
* **Google MediaPipe:** Skeletal hand tracking.
* **OpenCV:** High-speed frame extraction and matrix manipulation.
* **NumPy / SciPy:** Matrix math for bounding box Intersect-over-Union (IoU) and physics calculations.

**Android/Mobile Edge (APK):**
* **Kotlin & Jetpack Compose:** A standalone mobile port allowing local computation.
* **ONNX Runtime:** The fine-tuned YOLO and SOP classifiers from the Python backend were converted to `.onnx` models, allowing them to run 100% offline iteratively on Android smartphone silicon.

---

## 📂 Project Structure

```text
FIBA AI/
├── web_app/                   # Heavy Python/Deep Learning Environment
│   ├── app.py                 # Main Flask Server
│   ├── pipeline/              
│   │   ├── motion_engine.py   # Mathematical physics processing per-frame
│   │   ├── action_inferencer.py # Evaluates action labels from physics
│   │   ├── object_detector.py # YOLOv8 integration and alias tracking
│   │   ├── hand_skeleton.py   # MediaPipe pipeline
│   │   └── sop_validator.py   # Fine-tuned classification sequencer
│   ├── train_sop_classifier.py# Custom PyTorch YOLOv8n-cls training script
│   └── requirements.txt       
│
├── android_apk/               # Standalone Port (Native Kotlin)
│   ├── app/src/main/java/.../ml/  
│   │   ├── SOPClassifier.kt   # ONNX implementation of SOP Validation
│   │   └── YOLOClassifier.kt  # Fully native ONNX bounding-box inference
│   └── app/build.gradle.kts   # Kotlin Build system
└── README.md
```

## 🚀 Quick Start

### Python Web Environment
```bash
cd web_app
python -m venv .venv

# Activate Virtual Environment
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install required deep learning systems
pip install -r requirements.txt

# Run server
python app.py
```

### Android Native APK
The mobile application handles inference locally via `onnxruntime-android`. 
* Compile the project via Android Studio.
* Locate the debug APK in `android_apk\app\build\outputs\apk\debug\app-debug.apk`.

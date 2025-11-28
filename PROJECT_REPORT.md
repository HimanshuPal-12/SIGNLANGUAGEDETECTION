# Sign Language Detection Project - Comprehensive Report

**Project Date:** November 2025  
**Status:** Active Development  
**Team Size:** 4 engineers  

---

## 1. Executive Summary

The **Sign Language Detection** project is an AI-driven computer vision application designed to recognize hand gestures and sign language sequences in real-time. The system uses **MediaPipe** for hand pose estimation, **TensorFlow/Keras** for deep learning models (LSTM-based), and **OpenCV** for image processing. The project is structured to be developed and maintained by a 4-person engineering team with clear role separation and deliverables.

**Key Objectives:**
- Collect and preprocess sign language gesture data (25 actions: A–Z, excluding J)
- Extract hand keypoints (21 landmarks per hand × 3 coordinates = 63 features)
- Train LSTM neural network to classify sequences of keypoints
- Deploy inference pipeline via web/CLI app with Docker support
- Generate comprehensive analysis and evaluation reports

---

## 2. Project Architecture

### 2.1 Directory Structure

```
SIGNLANGUAGEDETECTION/
├── collectdata.py              # Data collection script (webcam → Image/)
├── data.py                     # Preprocessing (Image/ → MP_Data/.npy)
├── function.py                 # MediaPipe integration & keypoint extraction
├── trainmodel.py               # Model training & artifact saving
├── app.py                      # Inference app (CLI/UI)
├── analysis_and_evaluation.py  # Model evaluation & visualization
├── Image/
│   ├── A/, B/, ..., Z/        # Raw collected images per gesture
├── MP_Data/
│   ├── A/, B/, ..., Z/        # Processed keypoint .npy files per gesture
│       ├── 0/, 1/, ..., 29/   # 30 sequences per gesture
│           ├── 0.npy, 1.npy, ..., 29.npy  # 30 frames per sequence
├── Logs/                       # TensorBoard event files (training metrics)
├── analysis/                   # Evaluation outputs (plots, CSVs)
├── model.h5                    # Trained Keras model
├── model.json                  # Model architecture (JSON)
├── requirements.txt            # Python dependencies
├── .venv/                      # Virtual environment (Python 3.10)
└── README.md                   # User guide (to be created)
```

### 2.2 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION (collectdata.py)                             │
│    - Webcam input via OpenCV                                     │
│    - MediaPipe hand detection → keypoint extraction              │
│    - User presses 'a'–'z' to save frame to Image/A/–Image/Z/    │
│    - Output: Image/A/0.png, Image/A/1.png, ...                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING (data.py)                                       │
│    - Reads Image/A/0.png, ..., Image/Z/N.png                    │
│    - For each image: extract hand landmarks via MediaPipe       │
│    - Flatten 21 landmarks × 3 coords = 63-dim feature vector    │
│    - Save as MP_Data/A/0/0.npy, MP_Data/A/0/1.npy, ...          │
│    - Output: nested structure with one .npy per frame           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TRAINING (trainmodel.py)                                      │
│    - Load MP_Data/A/0/*.npy, MP_Data/A/1/*.npy, ...              │
│    - Stack 30 frames per sequence → shape (30, 63)              │
│    - Train LSTM model on sequences + one-hot encoded labels     │
│    - TensorBoard logging → Logs/                                │
│    - Save model.h5, model.json, training_history.json           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. EVALUATION & ANALYSIS (analysis_and_evaluation.py)           │
│    - Load model.h5 and test dataset                              │
│    - Generate confusion matrix, per-class metrics, ROC curves    │
│    - Plot learning curves (accuracy/loss vs epoch)              │
│    - Save visualizations to analysis/                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. INFERENCE & DEPLOYMENT (app.py)                              │
│    - Load model.h5 from disk                                     │
│    - Accept webcam stream or image input                         │
│    - Extract keypoints via function.py                          │
│    - Predict gesture → display result                            │
│    - Docker containerization for portability                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Hand Detection** | MediaPipe | 0.10.21 | Extract 21 hand landmarks in real-time |
| **Deep Learning** | TensorFlow/Keras | 2.19.1 | LSTM model for sequence classification |
| **Computer Vision** | OpenCV | 4.11.0 | Image I/O, display, preprocessing |
| **Data Processing** | NumPy | 1.26.4 | Array operations for keypoint vectors |
| **ML Utils** | scikit-learn | Latest | Train/test split, confusion matrix |
| **Visualization** | Matplotlib/Seaborn | Latest | Plots for analysis (curves, heatmaps) |
| **Python Runtime** | Python | 3.10 | Stable, well-supported for ML libs |
| **Environment** | Virtual Env (.venv) | - | Isolated dependency management |
| **Deployment** | Docker | Latest | Containerized inference |

**Installed Packages:**
```
mediapipe==0.10.21
opencv-python==4.11.0
opencv-contrib-python==4.11.0
tensorflow==2.19.1
keras==3.12.0
scikit-learn==latest
numpy==1.26.4
pandas==latest
matplotlib==3.10.7
seaborn==latest
protobuf==4.25.8 (pinned for TensorFlow compatibility)
```

---

## 4. Core Modules & Responsibilities

### 4.1 `function.py` - Feature Extraction Pipeline

**Purpose:** Integrate MediaPipe hand detection and extract keypoints from images.

**Key Functions:**
- `mediapipe_detection(image, model)` – Convert image BGR→RGB, run MediaPipe, return landmarks
- `draw_styled_landmarks(image, results)` – Render hand skeleton on image
- `extract_keypoints(results)` – Flatten 21 landmarks (x, y, z each) into 63-dim vector
- Module-level constants: `DATA_PATH`, `actions`, `sequence_length`, `no_sequences`

**Error Handling:**
- Wrapped MediaPipe import in try/except with user-friendly error message
- Returns None safely when landmarks are not detected

**Status:** ✅ Functional (ready for unit test enhancements by Person 2)

---

### 4.2 `collectdata.py` - Real-time Data Collection

**Purpose:** Capture webcam frames, extract keypoints, and save gestures by action.

**Workflow:**
1. Initialize `Image/A/`, `Image/B/`, ..., `Image/Z/` directories (done via `os.makedirs(..., exist_ok=True)`)
2. Open webcam with `cv2.VideoCapture(0)`
3. For each frame: extract hand keypoints via MediaPipe
4. Display frame with bounding box and count of saved images per action
5. User presses 'a' → save frame to `Image/A/<count>.png`, etc.
6. Press 'q' to quit

**Status:** ✅ Fixed (now uses `os.path.join` for cross-platform paths)

---

### 4.3 `data.py` - Preprocessing & Feature Export

**Purpose:** Convert raw images + keypoints into `.npy` sequences for model training.

**Workflow:**
1. Iterate over all actions and sequences
2. For each image file in `Image/<ACTION>/<SEQUENCE_ID>/<FRAME_ID>.png`:
   - Load with `cv2.imread()`
   - Extract hand keypoints (63-dim vector)
   - Save as `MP_Data/<ACTION>/<SEQUENCE_ID>/<FRAME_ID>.npy`
3. Handle missing frames gracefully (skip incomplete sequences)

**Recent Fix:** Added check for `frame is None` to skip corrupted/missing images without crashing.

**Status:** ✅ Functional (ready for optimization and batch export by Person 2)

---

### 4.4 `trainmodel.py` - Model Training & Artifact Generation

**Purpose:** Train an LSTM neural network on sequences of keypoints.

**Architecture:**
```
Input: (batch_size, 30 frames, 63 keypoints)
  ↓
LSTM(64, return_sequences=True, activation='relu')
  ↓
LSTM(128, return_sequences=True, activation='relu')
  ↓
LSTM(64, return_sequences=False, activation='relu')
  ↓
Dense(64, activation='relu')
  ↓
Dense(32, activation='relu')
  ↓
Dense(25, activation='softmax')  # 25 actions: A–Z except J
Output: (batch_size, 25) one-hot encoded predictions
```

**Training Details:**
- Loss: categorical cross-entropy
- Optimizer: Adam
- Metrics: categorical accuracy
- Epochs: 200 (with TensorBoard logging)
- Data split: 95% train, 5% test
- Callbacks: TensorBoard event logger

**Recent Enhancements:**
- Saves training history to `analysis/training_history.json` for later visualization
- Gracefully skips missing `.npy` files during data loading

**Status:** ✅ Functional, ready for hyperparameter tuning by Person 3

---

### 4.5 `analysis_and_evaluation.py` - Evaluation & Visualization

**Purpose:** Comprehensive model evaluation and dataset analysis.

**Outputs:**
1. **Class Distribution:** `original_class_distribution.png`, `balanced_class_distribution.png`
2. **Feature Correlation:** `feature_correlation.png` (heatmap of 63 keypoint features)
3. **Model Evaluation:**
   - Confusion matrix (heatmap + CSV)
   - Classification report (precision/recall/f1 per action)
   - Per-class metrics bar chart
4. **Learning Curves:**
   - `learning_curve_accuracy.png` (training vs validation accuracy)
   - `learning_curve_loss.png` (training vs validation loss)
   - `accuracy_loss_full.png` (combined 2-panel plot)

**Modes:**
- `--mode all` – Run all analyses
- `--mode distributions` – Class distribution only
- `--mode correlation` – Feature correlation only
- `--mode evaluate` – Model evaluation only

**Command:**
```bash
python analysis_and_evaluation.py --mode all --model model.h5
```

**Status:** ✅ Functional (learning curves and visual reports now working)

---

### 4.6 `app.py` - Inference & Deployment

**Purpose:** Load trained model and perform real-time inference on new inputs.

**Current Status:** ⏳ To be implemented by Person 4

**Expected Workflow:**
1. Load `model.h5` and `model.json`
2. Accept webcam stream or image input
3. Extract keypoints via `function.py`
4. Pass through LSTM model → get class probabilities
5. Display predicted action + confidence

**Deliverables (from Person 4):**
- Full `app.py` with CLI/UI
- `Dockerfile` for containerization
- Integration tests (model load + sample inference)
- User guide

---

## 5. Current Project Status

### 5.1 Completed ✅

| Component | Status | Notes |
|-----------|--------|-------|
| `function.py` | ✅ Complete | MediaPipe integration, robust error handling |
| `collectdata.py` | ✅ Fixed | Cross-platform path handling via `os.path.join` |
| `data.py` | ✅ Enhanced | Skips missing/corrupted images gracefully |
| `trainmodel.py` | ✅ Working | Saves model + history; skips incomplete sequences |
| `analysis_and_evaluation.py` | ✅ Full Featured | Distributions, correlation, learning curves, metrics |
| Virtual environment | ✅ Set up | Python 3.10 + all dependencies pinned |
| `requirements.txt` | ✅ Updated | mediapipe, tensorflow, opencv, sklearn, etc. |

### 5.2 In Progress / To Do 🔄

| Component | Owner | Status | Notes |
|-----------|-------|--------|-------|
| **Person 1:** Dataset Validation | Person 1 | Planned | Write `validate_dataset.py`, `generate_placeholders.py`, dataset README |
| **Person 2:** Unit Tests | Person 2 | Planned | Add tests for `extract_keypoints`, batch export utils |
| **Person 3:** Model Experiments | Person 3 | Planned | Iterate architecture, tune hyperparams, document final model |
| **Person 4:** Inference App | Person 4 | Planned | Implement `app.py`, Dockerfile, integration tests |
| **Project Governance** | Person 1 | Planned | `CONTRIBUTING.md`, PR template, branch strategy |

### 5.3 Known Issues & Fixes Applied

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| FileNotFoundError on `Image//B` | Missing directories; forward/backward slash mismatch | Created all `Image/A..Z` dirs; used `os.path.join` |
| `cv2.cvtColor` error (empty frame) | `cv2.imread()` returned None | Added `if frame is None: continue` check in `data.py` |
| ModuleNotFoundError: mediapipe | Python 3.13 unsupported; pip had no wheels | Used Python 3.10 with venv; pinned protobuf for compatibility |
| Indentation errors in scripts | Patch tool formatting issues | Fixed manually; verified syntax |
| Training history not saved | Missing JSON export from model.fit | Added `json.dump(history.history)` to `trainmodel.py` |

---

## 6. Team Structure & Responsibilities (4-Person Split)

### **Person 1 — Lead Data Engineer & Project Coordinator** (Extra Work)

**Role:** End-to-end dataset ownership + project coordination

**Responsibilities:**
- ✅ Run `collectdata.py` to gather sign language gesture videos from webcam
- ✅ Run `data.py` to preprocess images → extract keypoints → save to `MP_Data/`
- ✅ Create `generate_placeholders.py` to generate dummy .npy files for testing/CI
- ✅ Write `validate_dataset.py` to verify dataset integrity (check counts, file sizes)
- ✅ Produce `data/README.md` with dataset layout and reproduction steps
- ✅ Lead code reviews for data-related PRs
- ✅ Coordinate merges and maintain `develop` branch stability
- ✅ Organize weekly 30-min syncs with team

**Deliverables:**
- Fully validated `MP_Data/` with ≥ 30 sequences × 30 frames per action
- Passing validation script
- Dataset documentation
- Merged PRs from team members

**Timeline:** 4–6 days  
**Extra Work:** ~20% more (coordination + dataset automation + validation)

---

### **Person 2 — Feature Extraction & Preprocessing Engineer**

**Role:** Robustify keypoint extraction and preprocessing pipeline

**Responsibilities:**
- ✅ Harden `function.py`: improve error handling, add logging
- ✅ Write unit tests for `extract_keypoints` (test 21-landmark flattening, None handling)
- ✅ Implement batch export utilities to speed up `data.py`
- ✅ Add image/sequence I/O helpers (read sequences, validate frame counts)
- ✅ Update and pin all versions in `requirements.txt`
- ✅ Document `function.py` API (input/output shapes, exception contracts)

**Deliverables:**
- Tested `function.py` with ≥90% line coverage
- Batch export script (e.g., `batch_export.py`)
- Unit test suite passing on CI
- Updated `requirements.txt` with pinned versions

**Timeline:** 2–3 days

---

### **Person 3 — Model Engineer**

**Role:** Design, train, and evaluate deep learning model

**Responsibilities:**
- ✅ Experiment with LSTM architecture variations (layer size, dropout, etc.)
- ✅ Implement training notebook/script with hyperparameter logging
- ✅ Use TensorBoard to track experiments → save logs to `Logs/`
- ✅ Evaluate model: generate confusion matrix, per-class metrics, ROC curves
- ✅ Save final model artifacts: `model.h5`, `model.json`, `analysis/training_history.json`
- ✅ Produce evaluation report (performance summary, recommendations)

**Deliverables:**
- Final trained `model.h5` with ≥95% test accuracy (if dataset is good)
- Evaluation report with learning curves and confusion matrix
- Training notebook/script reproducible on standard hardware
- Logged experiments in TensorBoard

**Timeline:** 3–5 days

---

### **Person 4 — App, Deployment & QA Engineer**

**Role:** Build inference application and deployment pipeline

**Responsibilities:**
- ✅ Implement `app.py`: load model, accept input (webcam/image), predict action, display result
- ✅ Add CLI and/or simple web UI for testing
- ✅ Write integration tests: verify model loads, sample inference works
- ✅ Create `Dockerfile` for containerized deployment
- ✅ Write `README.md` with installation, usage, and Docker instructions
- ✅ Test on Windows + Linux (if applicable)

**Deliverables:**
- Runnable `app.py` (CLI or web)
- Passing integration tests
- `Dockerfile` that builds and runs the app
- User guide in `README.md`

**Timeline:** 2–3 days

---

### **Project Governance (Shared, led by Person 1)**

**Deliverables:**
- `CONTRIBUTING.md` with:
  - Branch naming: `feature/<name>`, `bugfix/<name>`, `hotfix/<name>`
  - PR template (description, checklist, testing notes)
  - Code owners per file/area
  - Review rotation (Person 1 coordinates)
- Sprint schedule (2 weeks) with daily standups
- Merge strategy (squash-and-merge to `develop`, no direct main pushes)

**Timeline:** 1 day

---

## 7. Development Workflow & Best Practices

### 7.1 Branch Strategy

```
main (production, tagged releases)
  ↑
develop (integration branch, always stable)
  ↑
feature/*, bugfix/*, hotfix/*  (individual PRs)
```

**Rules:**
- All work on feature branches
- PRs must pass tests + code review before merge
- Squash commits when merging to `develop`
- Person 1 coordinates merges to ensure data pipeline stability

### 7.2 Communication

- **Daily:** Asynchronous updates in shared Slack/chat channel
- **2x Weekly:** 30-min video sync (Monday + Thursday)
  - Share blockers, review progress
  - Adjust timeline if needed
  - Demo new features

### 7.3 Testing & CI

- **Unit tests:** Person 2 owns `function.py` tests; others add tests for their modules
- **Integration tests:** Person 4 owns app-level tests
- **Manual testing:** Each person validates their deliverables locally before PR

### 7.4 Documentation

- Inline code comments for complex logic
- Function docstrings with input/output types
- README per major component
- Final project `README.md` with full setup instructions

---

## 8. Deployment & Production

### 8.1 Docker Containerization

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**Build & Run:**
```bash
docker build -t sign-lang-detection .
docker run --rm -v /dev/video0:/dev/video0 sign-lang-detection  # Linux
docker run --rm sign-lang-detection  # Windows (adjust camera mapping)
```

### 8.2 Performance Expectations

- **Data Collection:** ~5–10 min to collect 30 sequences per action
- **Preprocessing:** ~2–5 min to export all keypoints to MP_Data/
- **Training:** ~5–15 min on GPU; ~1–2 hours on CPU (200 epochs)
- **Inference:** ~50–100 ms per frame on CPU; <10 ms on GPU

---

## 9. Success Metrics & Acceptance Criteria

### 9.1 Dataset

- ✅ Minimum 30 sequences × 30 frames per action (25 actions = 22,500 frames total)
- ✅ All .npy files valid (loadable, correct shape 63)
- ✅ Validation script confirms no missing/corrupt files

### 9.2 Model

- ✅ Test accuracy ≥ 90% on hold-out test set
- ✅ Per-class F1-score ≥ 0.85 (no extreme outliers)
- ✅ Training time ≤ 2 hours on standard GPU

### 9.3 Inference App

- ✅ Loads model in < 5 sec
- ✅ Runs inference < 100 ms per frame
- ✅ Handles edge cases (no hand detected, multiple hands, poor lighting)
- ✅ Docker image builds and runs without errors

### 9.4 Code Quality

- ✅ All modules have docstrings
- ✅ ≥80% test coverage for `function.py`
- ✅ No linting errors (PEP 8)
- ✅ All PRs reviewed and approved before merge

---

## 10. Timeline & Milestones

| Milestone | Target Date | Owner | Notes |
|-----------|------------|-------|-------|
| **M1: Data Collection** | Day 1–2 | Person 1 | Collect ≥30 seq/action; validate |
| **M2: Preprocessing** | Day 2–3 | Person 1 + 2 | Export MP_Data/, unit tests |
| **M3: Model Training** | Day 3–5 | Person 3 | Train, evaluate, save artifacts |
| **M4: App & Deployment** | Day 4–5 | Person 4 | Inference pipeline, Docker, tests |
| **M5: Integration & Polish** | Day 5–6 | All | Docs, final QA, prepare release |
| **Release Candidate** | Day 6 | Person 1 | Merge to `main`, tag v1.0 |

**Total Sprint Duration:** ~1 week (6 days, accounting for parallelism)

---

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Poor dataset quality | Medium | High | Person 1 validates early; use placeholder generator for CI |
| Model overfitting | Medium | Medium | Person 3 monitors val/train loss; uses dropout |
| Webcam/GPU not available | Low | High | Provide Docker + test on synthetic data |
| Dependency conflicts | Low | Medium | Lock versions in `requirements.txt`; test venv setup |
| Communication breakdown | Low | High | Weekly syncs; Person 1 coordinates; async updates |

---

## 12. Future Enhancements (Post-v1.0)

- Multi-hand gesture recognition (current: single hand)
- Continuous gesture streaming (current: fixed-length sequences)
- Mobile deployment (TensorFlow Lite)
- Real-time webcam UI with confidence scores
- Fine-tuning on user-specific data
- Data augmentation (rotations, scaling)
- Quantization for edge devices

---

## 13. Conclusion

The **Sign Language Detection** project is a well-scoped, 4-person engineering effort with clear role separation and a realistic 1-week delivery timeline. By assigning Person 1 as the lead coordinator with dataset ownership, we ensure data pipeline stability and cross-team communication. Each engineer has distinct, non-overlapping deliverables with measurable success criteria.

**Key Success Factors:**
1. Early dataset validation (Person 1, Day 1–2)
2. Robust preprocessing and testing (Person 2, Day 2–3)
3. Focused model experimentation (Person 3, Day 3–5)
4. Complete inference & deployment (Person 4, Day 4–5)
5. Regular sync-ups and transparent communication (all)

**Next Steps:**
1. Person 1: Start data collection and create validation scripts
2. Person 2: Begin unit tests for `extract_keypoints`
3. Person 3: Prepare training notebook skeleton
4. Person 4: Stub `app.py` with model loading
5. All: Attend first sync on Day 1 to clarify any questions

---

**Report Generated:** November 27, 2025  
**Contact:** Project Coordinator (Person 1)

# VisionGate — Intelligent Real-Time Visitor Counter

> An AI-driven system that detects, tracks, and counts unique visitors from video streams using YOLOv8 face detection, ArcFace embeddings, and Kalman filter tracking.

**Demo Video:** [Watch Demo](https://your-loom-or-youtube-link-here)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [AI Planning Document](#ai-planning-document)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Configuration](#configuration)
8. [Running the System](#running-the-system)
9. [Output Structure](#output-structure)
10. [Assumptions Made](#assumptions-made)
11. [Compute Load Estimates](#compute-load-estimates)
12. [Features](#features)

---

## Overview

VisionGate processes video streams (MP4 file or live RTSP camera) to:

- Detect faces in real-time using YOLOv8-face
- Generate 512-d ArcFace embeddings via InsightFace buffalo_l
- Track faces across frames using SORT + Kalman filter
- Auto-register new faces with unique IDs on first detection
- Re-identify returning faces without double-counting
- Log every entry and exit with a timestamped cropped face image
- Store all metadata in SQLite with video-derived timestamps
- Serve a live Flask dashboard showing real-time detections (bonus)
- Generate post-run pipeline and model performance charts

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Video Source                             │
│                   (MP4 file / RTSP stream)                       │
└───────────────────────────┬──────────────────────────────────────┘
                            │ frames
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FaceDetector                               │
│               YOLOv8-face  (GPU / Haar fallback)                 │
│          Outputs: List[Detection(bbox, confidence)]              │
└───────────────────────────┬──────────────────────────────────────┘
                            │ bounding boxes + confidences
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FaceTracker                                │
│            SORT + Kalman Filter (Hungarian matching)             │
│      Outputs: active_tracks, exited_tracks per frame             │
└───────────┬──────────────────────────────┬───────────────────────┘
            │ active tracks                │ exited tracks
            ▼                              ▼
┌───────────────────────┐      ┌─────────────────────────────────┐
│    FaceRecognizer     │      │       _process_exit()           │
│  InsightFace buffalo_l│      │ Save exit image + log DB event  │
│  ArcFace 512-d embeds │      └─────────────────────────────────┘
│  Cosine similarity    │
└───────────┬───────────┘
            │ face_id (matched or newly registered)
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FaceTrackingPipeline                           │
│  _process_entry()      → save entry image, log DB event         │
│  _identify_track()     → match or register new face             │
│  _save_model_metrics() → write model_metrics.json at end        │
└───────────┬──────────────────────────┬───────────────────────────┘
            │                          │
            ▼                          ▼
┌───────────────────┐      ┌──────────────────────────────────────┐
│  DatabaseManager  │      │            EventLogger               │
│  SQLite (WAL)     │      │  events.log  (rotating, 100 MB max)  │
│  faces table      │      │  logs/entries/YYYY-MM-DD/*.jpg       │
│  events table     │      │  logs/exits/YYYY-MM-DD/*.jpg         │
│  visitor_summary  │      └──────────────────────────────────────┘
└───────────────────┘
            │ (post-run)
            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    generate_metrics.py                           │
│   metrics_report.png         — pipeline metrics chart           │
│   model_metrics_report.png   — model performance chart          │
│   metrics_summary.txt        — plain-text summary               │
└──────────────────────────────────────────────────────────────────┘
```

---

## AI Planning Document

### Phase 1 — Planning

**Goal:** Count unique human visitors in a video stream with verifiable logging proof for every entry and exit.

**Key decisions made:**

- Use **YOLOv8-face** (not generic YOLOv8) — purpose-trained for face detection, significantly better on small and partial faces in CCTV footage
- Use **InsightFace buffalo_l (ArcFace)** over the `face_recognition` library — ArcFace is a SOTA model trained on millions of faces, far more accurate on side-profile and low-quality CCTV crops
- Use **SORT + Kalman filter** over pure IoU matching — the Kalman filter predicts each track's position between missed detections, dramatically reducing fragmentation on CCTV where faces briefly occlude each other
- Use **full-frame InsightFace detection** for embedding extraction instead of crop-based — InsightFace's internal detector is tuned for full images and fails frequently on pre-cropped partial faces
- Use **per-video isolated SQLite databases** — no cross-contamination between videos, clean outputs per recording
- Use **video-derived timestamps from filename** — DB records and logs reflect actual recording time (`record_20250620_184807.mp4` → `2025-06-20 18:48:07`), not notebook processing time

### Phase 2 — Feature List

| Feature | Status |
|---|---|
| YOLOv8 face detection with CUDA GPU | ✅ |
| InsightFace ArcFace embeddings (512-d) | ✅ |
| SORT + Kalman filter multi-object tracking | ✅ |
| Auto-registration of new faces | ✅ |
| Re-identification of returning faces | ✅ |
| Unique visitor counting (no double-count) | ✅ |
| Exactly one entry image per face per visit | ✅ |
| Exactly one exit image per face per visit | ✅ |
| logs/entries/YYYY-MM-DD/ folder structure | ✅ |
| events.log with all 7 required event types | ✅ |
| SQLite DB with faces / events / visitor_summary | ✅ |
| Video-derived timestamps in DB and logs | ✅ |
| Configurable skip_frames via config.json | ✅ |
| RTSP live stream support | ✅ |
| Flask live dashboard (bonus) | ✅ |
| Post-run pipeline + model performance charts | ✅ |
| Haar cascade fallback if YOLO unavailable | ✅ |
| Mock embedding fallback if InsightFace unavailable | ✅ |

### Phase 3 — Architecture Decisions

**Why SORT + Kalman over DeepSORT?**
DeepSORT requires a separate appearance re-ID model running per frame. SORT with Kalman prediction is sufficient here because InsightFace already handles identity matching — the Kalman filter handles temporal continuity between frames while InsightFace handles who the person is.

**Why per-video SQLite instead of one shared DB?**
Isolated databases ensure one video's processing cannot corrupt another's data and make it easy to inspect, share, or delete results per video independently.

**Why full-frame InsightFace instead of crop-based embedding?**
Passing a pre-cropped face to InsightFace triggers its internal re-detection pipeline which frequently fails on partial/side profiles common in CCTV. Running InsightFace on the full frame and then matching detections to YOLO tracks by centre-point proximity is significantly more reliable.

**Why video-derived timestamps?**
The recording timestamp is embedded in the filename. Using it makes the entire audit trail — DB records, log lines, image filenames — reflect when events happened in the real world rather than when a Colab notebook happened to process the file.

---

## Tech Stack

| Module | Technology |
|---|---|
| Face Detection | YOLOv8-face (ultralytics) |
| Face Recognition | InsightFace buffalo_l (ArcFace backbone) |
| Tracking | SORT + Kalman Filter (scipy, numpy) |
| Backend | Python 3.10+ |
| Database | SQLite with WAL mode (thread-safe) |
| Configuration | JSON (`config/config.json`) |
| Logging | Python logging + RotatingFileHandler |
| Frontend (bonus) | Flask + Flask-CORS |
| Metrics & Charts | matplotlib, numpy |
| Platform | Google Colab (T4 GPU) |

---

## Project Structure

```
face-tracker/
├── README.md
├── requirements.txt
├── face_tracker_colab.ipynb         ← Colab notebook (full workflow)
├── main.py                          ← Entry point
├── config/
│   └── config.json                  ← All configuration parameters
├── core/
│   ├── pipeline.py                  ← Central orchestrator
│   ├── face_detector.py             ← YOLOv8 detection + Haar fallback
│   ├── face_recognizer.py           ← InsightFace embeddings + matching
│   └── face_tracker.py              ← SORT + Kalman multi-object tracker
├── database/
│   ├── __init__.py
│   └── database_manager.py          ← SQLite / MongoDB / PostgreSQL manager
├── logging_system/
│   ├── __init__.py
│   └── event_logger.py              ← Image saving + structured log writing
├── frontend/
│   ├── app.py                       ← Flask dashboard server
│   └── templates/
│       └── index.html               ← Live dashboard UI
├── tools/
│   └── generate_metrics.py          ← Post-run metrics + visualization charts
├── models/
│   └── .gitkeep                     ← YOLOv8-face weights go here 
└── sample_outputs/
    └── record_20250620_184807/      ← Sample processed video output
        ├── database/
        │   └── face_tracker.db
        ├── logs/
        │   ├── events.log
        │   ├── entries/2025-06-20/
        │   └── exits/2025-06-20/
        ├── report.txt
        ├── metrics_report.png
        ├── model_metrics_report.png
        └── metrics_summary.txt
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU fallback available)
- Google Colab with T4 GPU (recommended)

### Option A — Google Colab (recommended)

1. Upload `face-tracker-submission.tar.gz` to Google Drive at `MyDrive/`
2. Open `face_tracker_colab.ipynb` in Google Colab
3. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
4. Run cells in order:

| Cell | Purpose |
|---|---|
| Cell 1 | Mount Google Drive |
| Cell 2 | Extract project from tar.gz |
| Cell 3 | Set up Drive symlinks for persistence across runtimes |
| Cell 4 | Install all Python packages |
| Cell 5 | Download YOLOv8-face + InsightFace models |
| Cell 6 | System check (CUDA, models, videos) |
| Cell 7 | Save configuration |
| Cell 8 | List available videos |
| Cell 9 | Select videos to process |
| Cell 10 | Run pipeline (no frontend) + generate metrics |
| Cell 11 | Run pipeline with live Flask dashboard |
| Cell 12 | Verify output folder structure |
| Cell 13 | Verify database contents |
| Cell 14 | Verify events.log coverage |
| Cell 15 | Display entry/exit image galleries |
| Cell 16 | Clean all outputs (reset before re-run) |

### Option B — Local installation

```bash
# Clone the repo
git clone https://github.com/Jelena-Heartly-X/VisionGate.git
cd face-tracker

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8-face model
mkdir -p models
wget -O models/yolov8n-face.pt \
  https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt

# Run on a video file (no preview, no frontend)
python main.py --source /path/to/video.mp4 --no-preview --no-frontend

# Run with live Flask dashboard
python main.py --source /path/to/video.mp4 --no-preview
# Open http://localhost:5000

# Generate metrics for an already-processed video
python tools/generate_metrics.py --output_dir outputs/record_20250620_184807
```

---

## Configuration

### Sample `config.json`

```json
{
  "video": {
    "source": "sample_video.mp4",
    "use_rtsp": false,
    "rtsp_url": "rtsp://username:password@ip:port/stream",
    "frame_width": 1280,
    "frame_height": 720
  },
  "detection": {
    "yolo_model": "yolov8n-face.pt",
    "device": "cuda",
    "confidence_threshold": 0.45,
    "iou_threshold": 0.4,
    "skip_frames": 1
  },
  "recognition": {
    "model_name": "buffalo_l",
    "ctx_id": 0,
    "embedding_similarity_threshold": 0.38,
    "min_face_size": 35
  },
  "tracking": {
    "iou_threshold": 0.15,
    "max_lost_frames": 30,
    "min_hits": 2
  },
  "database": {
    "type": "sqlite",
    "sqlite_path": "database/face_tracker.db",
    "mongo_uri": "mongodb://localhost:27017/face_tracker",
    "postgres_uri": "postgresql://user:pass@localhost/face_tracker"
  },
  "logging": {
    "log_file": "logs/events.log",
    "log_level": "INFO",
    "image_store_base": "logs",
    "max_log_size_mb": 100
  },
  "frontend": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 5000
  }
}
```

### Key parameters

| Parameter | Description | Default |
|---|---|---|
| `detection.skip_frames` | Frames to skip between detection cycles. `0` = every frame, `1` = every 2nd frame | `1` |
| `detection.confidence_threshold` | Minimum YOLO confidence to accept a detection | `0.45` |
| `recognition.embedding_similarity_threshold` | Cosine similarity threshold for re-identification. Lower = stricter matching | `0.38` |
| `recognition.min_face_size` | Minimum face bounding box width/height in pixels | `35` |
| `tracking.max_lost_frames` | Frames to keep a track alive without detection before triggering exit | `30` |
| `tracking.min_hits` | Minimum detections before a track is confirmed and shown | `2` |
| `video.use_rtsp` | Set `true` to switch from file to live RTSP stream | `false` |

---

## Running the System

### Video file — no frontend
```bash
python main.py --source /path/to/video.mp4 --no-preview --no-frontend
```

### Video file — with live dashboard
```bash
python main.py --source /path/to/video.mp4 --no-preview
# Open http://localhost:5000
```

### Live RTSP stream (interview mode)
Update `config.json`:
```json
"video": {
  "use_rtsp": true,
  "rtsp_url": "rtsp://username:password@camera_ip:554/stream"
}
```
Then run:
```bash
python main.py --no-preview
```

---

## Output Structure

For each processed video, all outputs are saved at `outputs/{video_stem}/`:

```
outputs/
└── record_20250620_184807/
    ├── database/
    │   └── face_tracker.db           ← SQLite database
    ├── logs/
    │   ├── events.log                ← Structured event log
    │   ├── entries/
    │   │   └── 2025-06-20/
    │   │       ├── FACE_XXXX_184815_123456.jpg
    │   │       └── FACE_YYYY_184823_456789.jpg
    │   └── exits/
    │       └── 2025-06-20/
    │           ├── FACE_XXXX_184832_789012.jpg
    │           └── FACE_YYYY_184901_234567.jpg
    ├── report.txt                    ← Plain-text run summary
    ├── model_metrics.json            ← Raw model metrics data
    ├── metrics_report.png            ← Pipeline metrics chart
    ├── model_metrics_report.png      ← Model performance chart
    └── metrics_summary.txt           ← Combined text summary
```

### Database Schema

**faces**

| Column | Type | Description |
|---|---|---|
| face_id | TEXT PK | Unique ID e.g. `FACE_A1B2C3D4E5F6` |
| first_seen | TEXT | Video-derived ISO timestamp of first detection |
| last_seen | TEXT | Video-derived ISO timestamp of last detection |
| entry_count | INTEGER | Total number of visits |
| embedding | TEXT | JSON-serialised 512-d ArcFace embedding |
| thumbnail | TEXT | Path to first entry image |

**events**

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| face_id | TEXT FK | References faces.face_id |
| event_type | TEXT | `entry` or `exit` |
| timestamp | TEXT | Video-derived ISO timestamp |
| image_path | TEXT | Path to saved face crop |
| track_id | INTEGER | Kalman tracker numeric ID |

**visitor_summary**

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Always 1 (singleton row) |
| total_unique | INTEGER | Total unique faces registered |
| last_updated | TEXT | Timestamp of last update |

### events.log format

```
[YYYY-MM-DD HH:MM:SS] [LEVEL   ] [module] EVENT_TYPE | field=value | field=value
```

Sample lines:
```
[2025-06-20 18:48:15] [INFO    ] [logging_system.event_logger] SYSTEM      | Pipeline started.
[2025-06-20 18:48:22] [INFO    ] [logging_system.event_logger] RECOGNITION | face_id=NONE | similarity=0.0000 | result=NO_MATCH
[2025-06-20 18:48:22] [INFO    ] [logging_system.event_logger] EMBEDDING   | face_id=FACE_A1B2C3D4E5F6 | dim=512 | action=GENERATED
[2025-06-20 18:48:22] [INFO    ] [logging_system.event_logger] REGISTRATION| face_id=FACE_A1B2C3D4E5F6 | action=NEW_FACE_REGISTERED
[2025-06-20 18:48:22] [INFO    ] [logging_system.event_logger] TRACKING    | track_id=1 | face_id=FACE_A1B2C3D4E5F6 | bbox=(120, 45, 280, 230)
[2025-06-20 18:48:22] [INFO    ] [logging_system.event_logger] FACE_ENTRY  | face_id=FACE_A1B2C3D4E5F6 | track_id=1 | status=NEW_REGISTRATION
[2025-06-20 18:48:45] [INFO    ] [logging_system.event_logger] FACE_EXIT   | face_id=FACE_A1B2C3D4E5F6 | track_id=1 | tracked_for=76_frames
```

---

## Assumptions Made

1. **Video filename encodes recording timestamp** — filenames follow `record_YYYYMMDD_HHMMSS.mp4`. The pipeline parses this as the base timestamp for all DB records and log entries. If the filename does not match this pattern, wall clock time is used as fallback.

2. **One embedding per unique person** — ArcFace cosine similarity with threshold `0.38` decides if two embeddings belong to the same person. This was tuned for the provided CCTV footage. Very similar-looking individuals may occasionally be counted as one.

3. **Minimum 4 stable detection frames** before a face is identified and registered. This prevents ghost tracks from single-frame noise detections.

4. **Exit triggered after `max_lost_frames` consecutive missed detections** (default 30 frames ≈ 1.2 seconds at 25fps). A face briefly disappearing and reappearing within this window is treated as continuous tracking, not a new visit.

5. **One entry + one exit per visit** — if a person fully exits and re-enters, that is a new visit (new entry/exit pair) but the same unique visitor. The unique count does not increment.

6. **CCTV footage quality** — optimized for standard CCTV resolution (720p–1080p). Very low light or heavy compression reduces detection and recognition accuracy.

7. **GPU strongly recommended** — the system falls back to CPU but real-time performance requires CUDA. On CPU, processing is approximately 10× slower than real-time.

8. **Single camera per pipeline instance** — one pipeline processes one stream. Multi-camera setups would require shared embedding databases across instances.

---

## Compute Load Estimates

### GPU — NVIDIA T4 (Google Colab)

| Component | VRAM | Time per Frame |
|---|---|---|
| YOLOv8n-face detection | ~800 MB | ~8 ms |
| InsightFace buffalo_l | ~1.2 GB | ~12 ms |
| Kalman tracker | negligible | < 1 ms |
| **Total** | **~2.0 GB** | **~20–25 ms (~40–50 FPS)** |

### CPU (fallback)

| Component | Time per Frame |
|---|---|
| YOLOv8n-face detection | ~80 ms |
| InsightFace buffalo_l | ~120 ms |
| **Total** | **~200 ms (~5 FPS)** |

### Storage per hour of video

| Output | Approximate Size |
|---|---|
| SQLite database | ~500 KB |
| events.log | ~2 MB |
| Face images (128×128 JPEG) | ~50 KB per unique face |
| model_metrics.json | ~100 KB |

---

## Features

### Core
- Real-time face detection via YOLOv8-face with configurable confidence threshold
- 512-d ArcFace embeddings via InsightFace buffalo_l
- SORT + Kalman filter tracking — reduces fragmentation vs pure IoU matching
- Auto-registration on first stable detection (4+ frames)
- Re-identification by cosine similarity — returning visitors not double-counted
- Exactly one entry image and one exit image per face per visit — enforced natively in EventLogger, not as a post-processing step
- Video-derived timestamps throughout — DB, logs, and image filenames all reflect actual recording time
- Frame skipping configurable via `skip_frames` in `config.json`
- RTSP live stream support for interview/production use

### Logging
- `events.log` covers 7 event types: FACE_ENTRY, FACE_EXIT, RECOGNITION, EMBEDDING, REGISTRATION, TRACKING, SYSTEM
- Images stored in `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`
- Rotating log file — 100 MB max per file, 5 backup files
- Thread-safe SQLite with WAL mode — resilient to unexpected interruptions

### Bonus
- **Flask live dashboard** — annotated real-time video feed, entry/exit event log, face gallery, live unique visitor count
- **Pipeline metrics chart** — visitor stat card, entry/exit bar chart, visit duration histogram, events.log breakdown
- **Model performance chart** — YOLO confidence distribution, ArcFace similarity distribution with threshold line, track length histogram, detections-per-frame timeline
- **Haar cascade fallback** — runs without GPU or YOLO weights for quick testing
- **Mock embedding fallback** — runs without InsightFace for pipeline testing

---

> This project is a part of a hackathon run by https://katomaran.com

"""
scripts/download_models.py
Downloads all required models before first run.

Models fetched:
  1. YOLOv8n-face (face detection) — from GitHub releases
  2. InsightFace buffalo_l (ArcFace recognition) — auto-downloaded by insightface SDK

Usage:
    python scripts/download_models.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_yolo_face_model():
    """Download YOLOv8n-face weights."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "yolov8n-face.pt"

    if model_path.exists():
        logger.info(f"YOLOv8-face already present at: {model_path}")
        return

    logger.info("Downloading YOLOv8n-face model...")
    url = "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
        logger.info(f"Downloaded → {model_path}")
    except Exception as e:
        logger.warning(
            f"Direct download failed ({e}). "
            "Falling back to ultralytics auto-download on first inference."
        )


def download_insightface_model():
    """Pre-download InsightFace buffalo_l model."""
    try:
        from insightface.app import FaceAnalysis
        logger.info("Pre-loading InsightFace buffalo_l model (downloads on first call)...")
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU, just to trigger download
        logger.info("InsightFace model ready.")
    except ImportError:
        logger.warning("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
    except Exception as e:
        logger.error(f"InsightFace download error: {e}")


if __name__ == "__main__":
    logger.info("=== Downloading required models ===")
    download_yolo_face_model()
    download_insightface_model()
    logger.info("=== All models ready ===")

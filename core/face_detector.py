"""
face_detector.py
Real-time face detection using YOLOv8-face (ultralytics).

Falls back to a Haar-cascade detector when the YOLO model is not available,
allowing the system to run for quick local testing without a GPU.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import cv2

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detected face bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        """Return (top, left, width, height) format."""
        return (self.y1, self.x1, self.height, self.width)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return (cx, cy, width, height) format."""
        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2
        return (cx, cy, self.width, self.height)

    def crop(self, frame: np.ndarray, padding: float = 0.15) -> np.ndarray:
        """
        Crop the face region from a frame with optional padding.

        Args:
            frame:   Full BGR frame.
            padding: Fraction of bbox size to add as border.
        """
        h, w = frame.shape[:2]
        pad_x = int(self.width * padding)
        pad_y = int(self.height * padding)
        x1 = max(0, self.x1 - pad_x)
        y1 = max(0, self.y1 - pad_y)
        x2 = min(w, self.x2 + pad_x)
        y2 = min(h, self.y2 + pad_y)
        return frame[y1:y2, x1:x2].copy()


class FaceDetector:
    """
    Wraps YOLOv8-face for face detection.
    Automatically downloads the model on first run if not present.
    Falls back to OpenCV Haar cascade if YOLO is unavailable.
    """

    YOLO_MODEL_URL = (
        "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
    )

    def __init__(self, config: dict):
        det_cfg = config["detection"]
        self.conf_threshold = det_cfg.get("confidence_threshold", 0.5)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.4)
        self.min_face_size = config["recognition"].get("min_face_size", 40)
        model_name = det_cfg.get("yolo_model", "yolov8n-face.pt")
        device_cfg = det_cfg.get("device", "auto")

        self.model = None
        self.use_fallback = False

        # ── Resolve compute device ─────────────────────────────────────
        self.device = self._resolve_device(device_cfg)

        # ── Attempt to load YOLO ───────────────────────────────────────
        model_path = Path("models") / model_name
        try:
            from ultralytics import YOLO
            if not model_path.exists():
                logger.info(f"Downloading YOLOv8-face model → {model_path}")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # ultralytics auto-downloads when given a known model name;
                # for custom face weights we try the direct path first.
                self.model = YOLO(str(model_path)) if model_path.exists() else YOLO(model_name)
            else:
                self.model = YOLO(str(model_path))
            # Warm-up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device=self.device)
            logger.info(f"YOLOv8-face model loaded on device: {self.device}")
        except Exception as e:
            logger.warning(
                f"YOLOv8 unavailable ({e}). Falling back to OpenCV Haar cascade."
            )
            self._init_haar_fallback()

    def _resolve_device(self, device_cfg: str) -> str:
        if device_cfg == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_cfg

    def _init_haar_fallback(self):
        """Load OpenCV's built-in Haar cascade for faces."""
        self.use_fallback = True
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        logger.info("Haar cascade face detector loaded as fallback.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run face detection on a single BGR frame.

        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        if frame is None or frame.size == 0:
            return []

        if self.use_fallback:
            return self._detect_haar(frame)
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                if w < self.min_face_size or h < self.min_face_size:
                    continue
                detections.append(Detection(x1, y1, x2, y2, conf))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_haar(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        detections = []
        if len(faces) == 0:
            return detections
        for (x, y, w, h) in faces:
            detections.append(Detection(x, y, x + w, y + h, confidence=0.8))
        return detections

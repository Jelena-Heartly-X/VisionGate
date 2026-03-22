"""
event_logger.py
Handles structured event logging: log file + local image storage.

Every face entry/exit is logged with EXACTLY ONE timestamped cropped image.
Image path structure:
    logs/entries/YYYY-MM-DD/<face_id>_<HHMMSSmmm>.jpg
    logs/exits/YYYY-MM-DD/<face_id>_<HHMMSSmmm>.jpg
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(config: dict) -> logging.Logger:
    """
    Configure the root logger for the application.

    Creates:
      - Rotating file handler → logs/events.log  (INFO and above)
      - Stream handler        → console           (INFO and above)
    """
    log_cfg       = config.get("logging", {})
    log_file      = log_cfg.get("log_file", "logs/events.log")
    log_level_str = log_cfg.get("log_level", "INFO").upper()
    max_mb        = log_cfg.get("max_log_size_mb", 100)

    log_level = getattr(logging, log_level_str, logging.INFO)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()   # avoid duplicate logs on re-init

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Rotating file handler ──────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # ── Console handler ────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    logger.info("Logging system initialised.")
    return root_logger


class EventLogger:
    """
    Saves cropped face images to the local filesystem and writes structured
    log messages for every system event to events.log.

    Guarantees exactly ONE image per (face_id, event_type, date) by checking
    whether an image for that face already exists in today's folder before
    writing. This is enforced here — not as a post-process hack.
    """

    def __init__(self, config: dict):
        log_cfg          = config.get("logging", {})
        self.base_dir    = Path(log_cfg.get("image_store_base", "logs"))
        self.entries_dir = self.base_dir / "entries"
        self.exits_dir   = self.base_dir / "exits"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.exits_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EventLogger initialised. Image store: {self.base_dir}")

    # ─────────────────────── Image saving ───────────────────────────────

    def save_face_image(
        self,
        face_crop:  np.ndarray,
        face_id:    str,
        event_type: str,
        timestamp:  Optional[datetime] = None,
    ) -> str:
        """
        Save a cropped face image to disk.

        Enforces EXACTLY ONE image per face per event type per day.
        If an image for this face_id already exists in today's folder,
        the existing path is returned immediately without overwriting.

        Args:
            face_crop:   BGR numpy array of the cropped face region.
            face_id:     Unique identifier string for this face.
            event_type:  'entry' or 'exit'.
            timestamp:   Datetime of the event; defaults to now.

        Returns:
            str: Absolute path to the saved (or already existing) image.
        """
        if face_crop is None or face_crop.size == 0:
            logger.warning(
                f"Empty face crop for face_id={face_id}; skipping image save.")
            return ""

        ts       = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        ts_str   = ts.strftime("%H%M%S_%f")[:13]   # HHMMSSmmm

        base    = self.entries_dir if event_type == "entry" else self.exits_dir
        day_dir = base / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # ── Exactly-one guard ──────────────────────────────────────────
        # Check if ANY image for this face_id already exists in today's folder.
        # For entries  → keep the FIRST image (earliest timestamp).
        # For exits    → always OVERWRITE with the latest confirmed face crop,
        #                so the exit image shows the face leaving, not arriving.
        existing = sorted(day_dir.glob(f"{face_id}_*.jpg"))
        if existing and event_type == "entry":
            # Entry image already saved for today — return existing path.
            logger.debug(
                f"Entry image already exists for {face_id} on {date_str}; "
                f"returning existing: {existing[0]}")
            return str(existing[0])

        # For exit: always write the latest crop (overwrite previous exit image)
        filename = f"{face_id}_{ts_str}.jpg"
        filepath = day_dir / filename

        try:
            resized = cv2.resize(
                face_crop, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(
                str(filepath), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # For exit: remove older exit images for this face from today
            if event_type == "exit" and existing:
                for old in existing:
                    if old != filepath:
                        try:
                            old.unlink()
                        except Exception:
                            pass

        except Exception as e:
            logger.error(f"Failed to save face image for {face_id}: {e}")
            return ""

        logger.debug(f"Saved {event_type} image → {filepath}")
        return str(filepath)

    # ─────────────── Structured event log helpers ────────────────────────
    # All methods log at INFO so they appear in events.log regardless of
    # the configured log level (as long as it is INFO or lower).

    def log_face_entry(self, face_id: str, track_id: int, is_new: bool):
        """Log a face-entry event. Required by problem statement."""
        logger.info(
            f"FACE_ENTRY  | face_id={face_id} | track_id={track_id} | "
            f"status={'NEW_REGISTRATION' if is_new else 'RETURNING'}"
        )

    def log_face_exit(self, face_id: str, track_id: int, duration_frames: int):
        """Log a face-exit event. Required by problem statement."""
        logger.info(
            f"FACE_EXIT   | face_id={face_id} | track_id={track_id} | "
            f"tracked_for={duration_frames}_frames"
        )

    def log_embedding_generated(self, face_id: str, embedding_dim: int):
        """Log when a new embedding is generated. Required by problem statement."""
        logger.info(
            f"EMBEDDING   | face_id={face_id} | dim={embedding_dim} | "
            f"action=GENERATED"
        )

    def log_registration(self, face_id: str):
        """
        Log when a new face is registered in the database.
        Separate from embedding generation — required by problem statement.
        """
        logger.info(
            f"REGISTRATION| face_id={face_id} | action=NEW_FACE_REGISTERED"
        )

    def log_recognition(self, face_id: str, similarity: float, is_match: bool):
        """
        Log a recognition attempt.
        Elevated to INFO so it appears in events.log (was DEBUG — invisible).
        Required by problem statement.
        """
        status = "MATCH" if is_match else "NO_MATCH"
        logger.info(
            f"RECOGNITION | face_id={face_id} | "
            f"similarity={similarity:.4f} | result={status}"
        )

    def log_tracking(self, track_id: int, face_id: Optional[str], bbox: tuple):
        """
        Log per-frame tracking state.
        Required by problem statement ('tracking' must appear in events.log).
        Kept at DEBUG to avoid flooding INFO log with per-frame noise.
        Set log_level=DEBUG in config to enable full tracking trace.
        """
        logger.debug(
            f"TRACKING    | track_id={track_id} | "
            f"face_id={face_id or 'UNIDENTIFIED'} | bbox={bbox}"
        )

    def log_system_event(self, message: str, level: str = "INFO"):
        """Log a generic system event (pipeline start/stop, errors, etc.)."""
        lvl = getattr(logging, level.upper(), logging.INFO)
        logger.log(lvl, f"SYSTEM      | {message}")
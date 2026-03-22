"""
pipeline.py
Central orchestrator of the face tracking pipeline.

Changes vs previous version:
  - Passes video-derived timestamp to all DB and EventLogger calls.
    No more datetime monkey-patching in the Colab notebook.
  - log_tracking() called once per entry (not per frame) at INFO level
    so TRACKING events appear cleanly in events.log without flooding it.
  - log_registration() called separately from log_embedding_generated().
  - All DB calls pass timestamp= so DB records reflect video time.
  - Model metrics collected during run and saved to model_metrics.json:
      detection_confidences  : every YOLO confidence score
      similarity_scores      : every embedding similarity score
      track_lengths          : frames per completed track
      detections_per_frame   : detection count per frame
"""

import json
import logging
import re
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Set, List

import cv2
import numpy as np

from core.face_detector import FaceDetector, Detection
from core.face_recognizer import FaceRecognizer
from core.face_tracker import FaceTracker, Track
from database.database_manager import DatabaseManager
from logging_system.event_logger import EventLogger

logger = logging.getLogger(__name__)


class FaceTrackingPipeline:

    def __init__(self, config: dict, show_preview: bool = True):
        self.config       = config
        self.show_preview = show_preview

        logger.info("Initialising pipeline modules...")
        self.detector     = FaceDetector(config)
        self.recognizer   = FaceRecognizer(config)
        self.tracker      = FaceTracker(config)
        self.db           = DatabaseManager(config)
        self.event_logger = EventLogger(config)

        # ── Video timestamp ──────────────────────────────────────────────
        self._video_fps:      float              = 25.0
        self._video_start_ts: Optional[datetime] = None

        # ── State ────────────────────────────────────────────────────────
        self._skip_frames: int  = config["detection"].get("skip_frames", 0)
        self._frame_count: int  = 0
        self._running:     bool = False
        self._stop_event        = threading.Event()

        self._track_to_face:  Dict[int, str]      = {}
        self._entered_faces:  Set[str]             = set()
        self._embedding_cache                      = self.db.get_all_embeddings()

        # Full-frame InsightFace cache — computed once per detection frame
        self._last_insight_frame: int  = -1
        self._last_insight_faces: list = []

        # Best (InsightFace-confirmed) crop per track — used for exit image.
        # Only updated when InsightFace confirms a real face near this track,
        # so exit images always show a real face, not a back-of-head crop.
        self._best_face_crop: Dict[int, np.ndarray] = {}
        self._best_face_ts:   Dict[int, datetime]   = {}

        self.latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # ── Runtime metrics ──────────────────────────────────────────────
        self.metrics = {
            "frames_processed": 0,
            "detections_total": 0,
            "unique_visitors":  self.db.get_unique_visitor_count(),
            "fps":              0.0,
            "start_time":       None,
        }

        # ── Model-level metrics collectors ───────────────────────────────
        # These are collected during the run and saved to model_metrics.json
        # at the end so generate_metrics.py can visualise them.
        self._model_metrics: Dict[str, List] = {
            "detection_confidences": [],   # every YOLO box confidence score
            "similarity_scores":     [],   # every embedding cosine similarity
            "track_lengths":         [],   # frames tracked per completed track
            "detections_per_frame":  [],   # number of detections each frame
        }

        logger.info("Pipeline ready.")

    # ── Video timestamp helpers ──────────────────────────────────────────

    def _parse_start_ts(self, source: str) -> Optional[datetime]:
        """Extract recording start time from filename like record_20250620_191129.mp4."""
        m = re.search(r'(\d{8})[_\-](\d{6})', Path(source).name)
        if m:
            try:
                ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                logger.info(f"Video start timestamp from filename: {ts}")
                return ts
            except ValueError:
                pass
        logger.warning(
            "Could not parse timestamp from filename — using wall clock.")
        return None

    def _video_ts(self) -> datetime:
        """Return the current video-derived timestamp."""
        if self._video_start_ts is None:
            return datetime.now()
        return self._video_start_ts + timedelta(
            seconds=self._frame_count / max(self._video_fps, 1.0))

    # ── Video source ─────────────────────────────────────────────────────

    def _open_video_source(self) -> cv2.VideoCapture:
        vid = self.config["video"]
        if vid.get("use_rtsp", False):
            src = vid.get("rtsp_url")
            logger.info(f"Opening RTSP stream: {src}")
        else:
            src = vid.get("source", "sample_video.mp4")
            if not Path(src).exists():
                raise FileNotFoundError(f"Video file not found: {src}")
            logger.info(f"Opening video file: {src}")

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {src}")

        self._video_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._video_start_ts = self._parse_start_ts(str(src))
        logger.info(f"Video FPS: {self._video_fps:.2f}")

        w = vid.get("frame_width")
        h = vid.get("frame_height")
        if w and h:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        self.metrics["start_time"] = time.time()
        self.event_logger.log_system_event("Pipeline started.")

        cap = self._open_video_source()
        last_detections: list = []
        fps_counter = 0
        fps_timer   = time.time()

        try:
            while self._running and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video stream ended.")
                    break

                self._frame_count               += 1
                self.metrics["frames_processed"] += 1

                # ── Detection every N frames (0 = every frame) ──────────
                if self._skip_frames == 0 or \
                        self._frame_count % (self._skip_frames + 1) == 0:
                    last_detections = self.detector.detect(frame)
                    self.metrics["detections_total"] += len(last_detections)

                    # ── Collect detection confidences & per-frame count ──
                    self._model_metrics["detections_per_frame"].append(
                        len(last_detections))
                    for det in last_detections:
                        self._model_metrics["detection_confidences"].append(
                            round(float(det.confidence), 4))

                    # Invalidate full-frame InsightFace cache
                    self._last_insight_frame = -1
                    self._last_insight_faces = []

                # ── Tracker update ───────────────────────────────────────
                bboxes = [d.bbox for d in last_detections]
                confs  = [d.confidence for d in last_detections]
                active_tracks, exited_tracks = self.tracker.update(bboxes, confs)

                for track in active_tracks:
                    self._process_active_track(track, frame)

                for track in exited_tracks:
                    self._process_exit(track, frame)

                annotated = self._annotate_frame(frame, active_tracks)
                with self._frame_lock:
                    self.latest_frame = annotated.copy()

                if self.show_preview:
                    cv2.imshow("Face Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self.metrics["fps"] = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer   = time.time()

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._finalize()

    def stop(self):
        self._stop_event.set()
        self._running = False

    # ── Full-frame InsightFace (cached per detection frame) ───────────────

    def _get_full_frame_faces(self, frame: np.ndarray) -> list:
        """
        Run InsightFace on the FULL frame and cache for this detection frame.
        Full-frame detection is more reliable for partial/side profiles.
        """
        if self._last_insight_frame == self._frame_count:
            return self._last_insight_faces

        try:
            faces = self.recognizer._app.get(frame)
            self._last_insight_faces = faces if faces else []
            self._last_insight_frame = self._frame_count
        except Exception as e:
            logger.debug(f"Full-frame InsightFace error: {e}")
            self._last_insight_faces = []
            self._last_insight_frame = self._frame_count

        return self._last_insight_faces

    # ── Track processing ──────────────────────────────────────────────────

    def _process_active_track(self, track: Track, frame: np.ndarray):
        """
        For each active track:
          - Try full-frame InsightFace to confirm a real face is visible.
          - Save the best confirmed crop (used for exit image later).
          - Identify the face after 4 stable hits.
          - Log entry once face_id is assigned.
        """
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]

        # ── Update best confirmed crop ────────────────────────────────
        full_faces = self._get_full_frame_faces(frame)
        track_cx   = (x1 + x2) / 2
        track_cy   = (y1 + y2) / 2

        for face in full_faces:
            try:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                fcx  = (fx1 + fx2) / 2
                fcy  = (fy1 + fy2) / 2
                dist = ((track_cx - fcx) ** 2 + (track_cy - fcy) ** 2) ** 0.5
                if dist < 100 and face.det_score > 0.5:
                    crop = frame[max(0, fy1):min(h, fy2),
                                 max(0, fx1):min(w, fx2)]
                    if crop.size > 0:
                        self._best_face_crop[track.track_id] = crop.copy()
                        self._best_face_ts[track.track_id]   = self._video_ts()
                    break
            except Exception:
                continue

        # ── Identify face after 4 stable hits ────────────────────────
        if track.face_id is None and track.hits >= 4:
            face_id = self._identify_track(track, frame)
            if face_id:
                track.face_id = face_id
                self.tracker.assign_face_id(track.track_id, face_id)
                self._track_to_face[track.track_id] = face_id

        # ── Log entry once (first time face_id is confirmed) ─────────
        if track.face_id and track.face_id not in self._entered_faces:
            self._process_entry(track, frame)

    def _identify_track(self, track: Track, frame: np.ndarray) -> Optional[str]:
        """
        Get embedding using full-frame InsightFace first, crop fallback second.
        Match against registered faces; register as new if no match found.
        Collects similarity scores into model_metrics for visualisation.
        """
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        track_cx = (x1 + x2) / 2
        track_cy = (y1 + y2) / 2

        embedding = None

        # ── Strategy 1: full-frame InsightFace ───────────────────────
        full_faces = self._get_full_frame_faces(frame)
        if full_faces:
            best_face = None
            best_dist = float('inf')
            for face in full_faces:
                try:
                    fx1, fy1, fx2, fy2 = map(int, face.bbox)
                    fcx  = (fx1 + fx2) / 2
                    fcy  = (fy1 + fy2) / 2
                    dist = ((track_cx - fcx) ** 2 + (track_cy - fcy) ** 2) ** 0.5
                    if dist < best_dist and dist < 100:
                        best_dist = dist
                        best_face = face
                except Exception:
                    continue

            if best_face is not None and best_face.embedding is not None:
                emb  = np.array(best_face.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0.1:
                    embedding = emb / norm

        # ── Strategy 2: crop fallback ─────────────────────────────────
        if embedding is None:
            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face_crop.size > 0:
                embedding = self.recognizer.get_embedding(face_crop)

        if embedding is None:
            return None

        ts = self._video_ts()

        # ── Match against registered faces ────────────────────────────
        matched_id, similarity = self.recognizer.find_best_match(
            embedding, self._embedding_cache)

        # Collect similarity score for model metrics visualisation
        self._model_metrics["similarity_scores"].append(round(float(similarity), 4))

        # Log every recognition attempt (required by problem statement)
        self.event_logger.log_recognition(
            matched_id or "NONE", similarity, matched_id is not None)

        if matched_id:
            self.db.update_face_last_seen(matched_id, timestamp=ts)
            self._update_embedding_if_better(matched_id, embedding)
            return matched_id

        # ── Register as new face ──────────────────────────────────────
        new_id    = self.recognizer.generate_new_face_id()
        face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # Save entry thumbnail
        thumbnail = self.event_logger.save_face_image(
            face_crop, new_id, "entry", ts)

        # Log embedding generation (required by problem statement)
        self.event_logger.log_embedding_generated(new_id, len(embedding))

        registered = self.db.register_face(
            new_id, embedding.tolist(), thumbnail, timestamp=ts)

        if registered:
            # Log registration separately (required by problem statement)
            self.event_logger.log_registration(new_id)

            self._embedding_cache.append(
                {"face_id": new_id, "embedding": embedding.tolist()})
            self.metrics["unique_visitors"] = self.db.get_unique_visitor_count()
            logger.info(
                f"New face registered: {new_id} | "
                f"Total unique visitors: {self.metrics['unique_visitors']}")

        return new_id

    def _update_embedding_if_better(self, face_id: str, new_emb: np.ndarray):
        """Replace stored embedding if new one has higher norm (better quality)."""
        try:
            for record in self._embedding_cache:
                if record["face_id"] == face_id:
                    old_norm = np.linalg.norm(
                        np.array(record["embedding"], dtype=np.float32))
                    new_norm = float(np.linalg.norm(new_emb))
                    if new_norm > old_norm + 0.05:
                        record["embedding"] = new_emb.tolist()
                    break
        except Exception:
            pass

    def _process_entry(self, track: Track, frame: np.ndarray):
        """
        Log exactly one entry event per face per visit.
        Also logs one TRACKING line at INFO level so it appears
        in events.log without flooding it with per-frame lines.
        """
        # Use best confirmed InsightFace crop if available, else bbox crop
        crop = self._best_face_crop.get(track.track_id)
        if crop is None or crop.size == 0:
            x1, y1, x2, y2 = track.bbox
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        ts         = self._video_ts()
        image_path = self.event_logger.save_face_image(
            crop, track.face_id, "entry", ts)

        # Pass video timestamp to DB
        self.db.log_event(
            track.face_id, "entry", image_path, track.track_id,
            timestamp=ts)

        is_new = track.face_id not in [
            r["face_id"] for r in self._embedding_cache[:-1]]

        # One TRACKING line per entry — clean, auditable, compliant
        # with problem statement without per-frame log flooding
        self.event_logger.log_tracking(
            track.track_id, track.face_id, track.bbox)

        self.event_logger.log_face_entry(
            track.face_id, track.track_id, is_new=is_new)

        self._entered_faces.add(track.face_id)

    def _process_exit(self, track: Track, frame: np.ndarray):
        """
        Log exactly one exit event per face per visit.
        Uses the best InsightFace-confirmed crop saved during active tracking,
        so the exit image always shows a real face, not a stale/empty bbox.
        Also collects track length into model_metrics.
        """
        if not track.face_id:
            self._best_face_crop.pop(track.track_id, None)
            self._best_face_ts.pop(track.track_id, None)
            return

        # Noise guard: skip track fragments (< 3 frames)
        if track.hits < 3:
            self._track_to_face.pop(track.track_id, None)
            self._best_face_crop.pop(track.track_id, None)
            self._best_face_ts.pop(track.track_id, None)
            return

        # Collect track length for model metrics
        self._model_metrics["track_lengths"].append(track.hits)

        exit_crop = self._best_face_crop.pop(track.track_id, None)
        ts        = self._best_face_ts.pop(track.track_id, self._video_ts())

        # Fallback: use current bbox if no confirmed crop available
        if exit_crop is None or exit_crop.size == 0:
            x1, y1, x2, y2 = track.bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                exit_crop = frame[y1:y2, x1:x2]
            ts = self._video_ts()

        image_path = ""
        if exit_crop is not None and exit_crop.size > 0:
            image_path = self.event_logger.save_face_image(
                exit_crop, track.face_id, "exit", ts)

        # Pass video timestamp to DB
        self.db.log_event(
            track.face_id, "exit", image_path, track.track_id,
            timestamp=ts)

        self.event_logger.log_face_exit(
            track.face_id, track.track_id, track.hits)

        self._entered_faces.discard(track.face_id)
        self._track_to_face.pop(track.track_id, None)

    # ── Frame annotation ──────────────────────────────────────────────────

    def _annotate_frame(
        self, frame: np.ndarray, active_tracks: list
    ) -> np.ndarray:
        out = frame.copy()

        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            label = (
                f"{track.face_id[-8:] if track.face_id else '?'} "
                f"#{track.track_id}"
            )
            color = (0, 200, 0) if track.face_id else (0, 140, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        vid_ts = self._video_ts().strftime("%H:%M:%S")
        hud = [
            f"Frame:  {self._frame_count}",
            f"Time:   {vid_ts}",
            f"Active: {len(active_tracks)}",
            f"Unique: {self.metrics['unique_visitors']}",
            f"FPS:    {self.metrics['fps']:.1f}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(out, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA)
        return out

    # ── Cleanup ───────────────────────────────────────────────────────────

    def _save_model_metrics(self):
        """
        Save collected model-level metrics to model_metrics.json
        in the same folder as the SQLite database.
        Read by generate_metrics.py for the model metrics chart.
        """
        try:
            db_path  = Path(self.config["database"].get(
                "sqlite_path", "database/face_tracker.db"))
            out_path = db_path.parent.parent / "model_metrics.json"

            # Compute summary statistics before saving
            confs   = self._model_metrics["detection_confidences"]
            sims    = self._model_metrics["similarity_scores"]
            lengths = self._model_metrics["track_lengths"]
            dpf     = self._model_metrics["detections_per_frame"]

            summary = {
                "detection_confidences": confs,
                "similarity_scores":     sims,
                "track_lengths":         lengths,
                "detections_per_frame":  dpf,
                "stats": {
                    "total_detections":       len(confs),
                    "avg_confidence":         round(float(np.mean(confs)), 4)  if confs   else 0,
                    "min_confidence":         round(float(np.min(confs)), 4)   if confs   else 0,
                    "max_confidence":         round(float(np.max(confs)), 4)   if confs   else 0,
                    "avg_similarity":         round(float(np.mean(sims)), 4)   if sims    else 0,
                    "avg_track_length":       round(float(np.mean(lengths)), 1) if lengths else 0,
                    "max_track_length":       int(np.max(lengths))              if lengths else 0,
                    "avg_detections_per_frame": round(float(np.mean(dpf)), 2)  if dpf     else 0,
                    "total_frames_with_detection": sum(1 for x in dpf if x > 0),
                    "total_frames_processed": len(dpf),
                }
            }

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Model metrics saved → {out_path}")

        except Exception as e:
            logger.error(f"Failed to save model metrics: {e}")

    def _finalize(self):
        total   = self.db.get_unique_visitor_count()
        elapsed = time.time() - self.metrics["start_time"]
        self.event_logger.log_system_event(
            f"Pipeline stopped. Unique visitors: {total}. "
            f"Runtime: {elapsed:.1f}s. Frames: {self._frame_count}.")
        self._save_model_metrics()
        self.db.close()
        logger.info(f"─── Session complete. Unique visitors: {total} ───")
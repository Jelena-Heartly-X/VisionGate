"""
app.py
Flask web dashboard for the face tracking system.
Provides:
  - Live MJPEG video stream
  - Real-time visitor count
  - Events table (last N entries/exits)
  - Registered faces gallery
  - REST API endpoints for programmatic access
"""

import io
import json
import logging
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, send_file, request

logger = logging.getLogger(__name__)

# Global reference to pipeline (set in main.py)
_pipeline = None


def create_app(pipeline, config: dict) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        pipeline: FaceTrackingPipeline instance (may still be initialising).
        config:   Full configuration dict.
    """
    global _pipeline
    _pipeline = pipeline

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ─────────────────────────── Routes ─────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        """MJPEG stream for live face-annotated video."""
        return Response(
            _generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/stats")
    def api_stats():
        """Return current pipeline statistics."""
        if _pipeline is None:
            return jsonify({"error": "Pipeline not initialised"}), 503
        return jsonify({
            "unique_visitors": _pipeline.db.get_unique_visitor_count(),
            "frames_processed": _pipeline.metrics.get("frames_processed", 0),
            "fps": round(_pipeline.metrics.get("fps", 0), 1),
            "detections_total": _pipeline.metrics.get("detections_total", 0),
        })

    @app.route("/api/events")
    def api_events():
        """Return recent events (query param: limit, type)."""
        limit = int(request.args.get("limit", 50))
        event_type = request.args.get("type", None)
        events = _pipeline.db.get_events(limit=limit, event_type=event_type)
        return jsonify(events)

    @app.route("/api/faces")
    def api_faces():
        """Return all registered faces."""
        faces = _pipeline.db.get_all_faces()
        return jsonify(faces)

    @app.route("/api/visitors")
    def api_visitors():
        """Return unique visitor count."""
        count = _pipeline.db.get_unique_visitor_count()
        return jsonify({"unique_visitors": count})

    @app.route("/face_image/<path:image_path>")
    def face_image(image_path):
        """Serve a saved face image."""
        full = Path(image_path)
        if not full.exists():
            return "", 404
        return send_file(str(full), mimetype="image/jpeg")

    # ────────────────────────────────────────────────────────────────────
    return app


def _generate_mjpeg():
    """Generator that yields JPEG frames as an MJPEG stream."""
    while True:
        if _pipeline is None:
            time.sleep(0.1)
            continue

        with _pipeline._frame_lock:
            frame = _pipeline.latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 25)  # ~25 fps stream cap

"""
main.py
Entry point for the Face Tracking System.

Usage:
    python main.py                          # run with default config
    python main.py --config custom.json     # run with custom config
    python main.py --source video.mp4       # override video source
    python main.py --no-preview             # headless (server/Colab mode)
    python main.py --no-frontend            # skip Flask dashboard
"""

import argparse
import sys
import threading
import logging
from pathlib import Path

from config.config_loader import load_config
from logging_system.event_logger import setup_logging
from core.pipeline import FaceTrackingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config/config.json", help="Path to config file")
    parser.add_argument("--source", default=None, help="Override video source (file path or RTSP URL)")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--no-frontend", action="store_true", help="Disable Flask web dashboard")
    parser.add_argument("--rtsp", action="store_true", help="Force RTSP mode (ignores video file)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load config ────────────────────────────────────────────────
    config = load_config(args.config)

    # ── 2. Override video source if specified ─────────────────────────
    if args.source:
        if args.source.startswith("rtsp://"):
            config["video"]["rtsp_url"] = args.source
            config["video"]["use_rtsp"] = True
        else:
            config["video"]["source"] = args.source
            config["video"]["use_rtsp"] = False

    if args.rtsp:
        config["video"]["use_rtsp"] = True

    # ── 3. Setup logging ──────────────────────────────────────────────
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Face Tracking System — Starting")
    logger.info("=" * 60)

    # ── 4. Create pipeline ────────────────────────────────────────────
    show_preview = not args.no_preview
    pipeline = FaceTrackingPipeline(config, show_preview=show_preview)

    # ── 5. Optionally start Flask in background thread ─────────────────
    if not args.no_frontend and config.get("frontend", {}).get("enabled", True):
        from frontend.app import create_app
        flask_app = create_app(pipeline, config)

        fe_cfg = config.get("frontend", {})
        host = fe_cfg.get("host", "0.0.0.0")
        port = fe_cfg.get("port", 5000)

        def run_flask():
            logger.info(f"Flask dashboard: http://{host}:{port}")
            flask_app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

    # ── 6. Run pipeline (blocking) ─────────────────────────────────────
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        pipeline.stop()

    logger.info("Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""
generate_metrics.py
Post-run metrics and visualizations for a processed video.

Reads from:
  - outputs/{video}/database/face_tracker.db  → visitor/event data
  - outputs/{video}/logs/events.log            → log event counts
  - outputs/{video}/model_metrics.json         → model-level metrics

Produces:
  - metrics_summary.txt    plain-text summary of all metrics
  - metrics_report.png     4-panel pipeline metrics chart
  - model_metrics_report.png  4-panel model performance chart

Usage:
    python generate_metrics.py --output_dir /path/to/outputs/video_stem
"""

import argparse
import json
import sqlite3
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────── Theme constants ─────────────────────────────────

DARK_BG  = "#1a1d27"
FIG_BG   = "#0f1117"
ACCENT   = "#00d4aa"
ACCENT2  = "#ff6b6b"
ACCENT3  = "#f0c040"
ACCENT4  = "#7b8cde"
TEXT_COL = "#e0e0e0"
GRID_COL = "#2a2d3a"
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4,
            "#ff9f40", "#4bc0c0", "#9966ff"]


# ─────────────────────────── Data loading ────────────────────────────────────

def load_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    faces = conn.execute(
        "SELECT face_id, first_seen, last_seen, entry_count "
        "FROM faces ORDER BY first_seen"
    ).fetchall()

    events = conn.execute(
        "SELECT face_id, event_type, timestamp "
        "FROM events ORDER BY timestamp"
    ).fetchall()

    summary = conn.execute(
        "SELECT total_unique FROM visitor_summary WHERE id=1"
    ).fetchone()

    conn.close()
    return (
        [dict(r) for r in faces],
        [dict(r) for r in events],
        summary["total_unique"] if summary else 0,
    )


def load_log(log_path: str):
    lines = []
    if not os.path.exists(log_path):
        return lines
    pattern = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?'
        r'(FACE_ENTRY|FACE_EXIT|RECOGNITION|EMBEDDING|REGISTRATION|TRACKING|SYSTEM)'
    )
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                lines.append({
                    "timestamp": m.group(1),
                    "event":     m.group(2),
                    "raw":       line.strip(),
                })
    return lines


def load_model_metrics(model_metrics_path: str) -> dict:
    if not os.path.exists(model_metrics_path):
        return {}
    with open(model_metrics_path) as f:
        return json.load(f)


# ─────────────────────────── Pipeline metrics ────────────────────────────────

def compute_pipeline_metrics(faces, events, total_unique, log_lines):
    entries = [e for e in events if e["event_type"] == "entry"]
    exits   = [e for e in events if e["event_type"] == "exit"]

    # Track durations — match each entry to its closest following exit
    durations = []
    for face in faces:
        fid = face["face_id"]
        fe  = [e["timestamp"] for e in entries if e["face_id"] == fid]
        fx  = [e["timestamp"] for e in exits   if e["face_id"] == fid]
        for ets in fe:
            after = [x for x in fx if x >= ets]
            if after:
                try:
                    t_in  = datetime.fromisoformat(ets)
                    t_out = datetime.fromisoformat(min(after))
                    durations.append((t_out - t_in).total_seconds())
                except Exception:
                    pass

    # Recognition match rate from log
    rec_lines  = [l for l in log_lines if l["event"] == "RECOGNITION"]
    matches    = sum(1 for l in rec_lines
                     if "MATCH" in l["raw"] and "NO_MATCH" not in l["raw"])
    no_matches = sum(1 for l in rec_lines if "NO_MATCH" in l["raw"])

    return {
        "total_unique":     total_unique,
        "total_entries":    len(entries),
        "total_exits":      len(exits),
        "total_faces_db":   len(faces),
        "avg_duration_s":   float(np.mean(durations))  if durations else 0,
        "max_duration_s":   float(np.max(durations))   if durations else 0,
        "min_duration_s":   float(np.min(durations))   if durations else 0,
        "durations":        durations,
        "recognition_matches":    matches,
        "recognition_no_matches": no_matches,
        "log_event_counts": {
            evt: sum(1 for l in log_lines if l["event"] == evt)
            for evt in ["FACE_ENTRY", "FACE_EXIT", "RECOGNITION",
                        "EMBEDDING", "REGISTRATION", "TRACKING", "SYSTEM"]
        },
    }


# ─────────────────────────── Text summary ────────────────────────────────────

def write_text_summary(
    pipeline: dict,
    model:    dict,
    out_path: str,
    video_name: str,
):
    stats = model.get("stats", {})
    lines = [
        "=" * 62,
        f"METRICS SUMMARY — {video_name}",
        "=" * 62,
        "",
        "── Visitor Counts ────────────────────────────────────────",
        f"  Unique visitors      : {pipeline['total_unique']}",
        f"  Total entries logged : {pipeline['total_entries']}",
        f"  Total exits logged   : {pipeline['total_exits']}",
        f"  Faces in database    : {pipeline['total_faces_db']}",
        "",
        "── Visit Duration ────────────────────────────────────────",
        f"  Average time in frame: {pipeline['avg_duration_s']:.1f}s",
        f"  Longest visit        : {pipeline['max_duration_s']:.1f}s",
        f"  Shortest visit       : {pipeline['min_duration_s']:.1f}s",
        "",
        "── Recognition Performance ───────────────────────────────",
        f"  Matches   : {pipeline['recognition_matches']}",
        f"  No-matches: {pipeline['recognition_no_matches']}",
        f"  Match rate: {pipeline['recognition_matches'] / max(1, pipeline['recognition_matches'] + pipeline['recognition_no_matches']) * 100:.1f}%",
        "",
        "── events.log Event Counts ───────────────────────────────",
    ]
    for evt, count in pipeline["log_event_counts"].items():
        status = "✓" if count > 0 else "✗"
        lines.append(f"  {status} {evt:<22}: {count}")

    if stats:
        lines += [
            "",
            "── Model-Level Metrics ───────────────────────────────────",
            f"  Total detections        : {stats.get('total_detections', 0)}",
            f"  Avg YOLO confidence     : {stats.get('avg_confidence', 0):.4f}",
            f"  Min / Max confidence    : {stats.get('min_confidence', 0):.4f} / {stats.get('max_confidence', 0):.4f}",
            f"  Avg embedding similarity: {stats.get('avg_similarity', 0):.4f}",
            f"  Avg track length        : {stats.get('avg_track_length', 0):.1f} frames",
            f"  Max track length        : {stats.get('max_track_length', 0)} frames",
            f"  Avg detections/frame    : {stats.get('avg_detections_per_frame', 0):.2f}",
            f"  Frames with detections  : {stats.get('total_frames_with_detection', 0)} / {stats.get('total_frames_processed', 0)}",
        ]

    lines += ["", "=" * 62]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Text summary saved → {out_path}")


# ─────────────────────────── Chart helpers ───────────────────────────────────

def _style_ax(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.spines[:].set_color(GRID_COL)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=11, pad=8)


def _grid(ax, axis="y"):
    if axis == "y":
        ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
    else:
        ax.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax.set_axisbelow(True)


# ─────────────────────────── Pipeline metrics chart ──────────────────────────

def generate_pipeline_chart(metrics: dict, out_path: str, video_name: str):
    """
    4-panel pipeline metrics chart:
      Top-left    : Unique visitor stat card
      Top-right   : Entry / Exit / Unique bar chart
      Bottom-left : Visit duration histogram
      Bottom-right: events.log event type breakdown
    """
    fig = plt.figure(figsize=(14, 9), facecolor=FIG_BG)
    fig.suptitle(
        f"Pipeline Metrics — {video_name}",
        fontsize=15, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Visitor stat card ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "Visitor Count")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.text(0.5, 0.78, str(metrics["total_unique"]),
             ha="center", va="center", fontsize=68, fontweight="bold",
             color=ACCENT, transform=ax1.transAxes)
    ax1.text(0.5, 0.38, "Unique Visitors",
             ha="center", va="center", fontsize=14, color=TEXT_COL,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.18,
             f"{metrics['total_entries']} entries  •  {metrics['total_exits']} exits",
             ha="center", va="center", fontsize=10, color="#888888",
             transform=ax1.transAxes)

    # ── Panel 2: Entry / Exit bar chart ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Entry / Exit Events")
    cats   = ["Entries", "Exits", "Unique"]
    vals   = [metrics["total_entries"], metrics["total_exits"],
              metrics["total_unique"]]
    colors = [ACCENT, ACCENT2, ACCENT3]
    bars   = ax2.bar(cats, vals, color=colors, width=0.5,
                     edgecolor="#000", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(vals) * 0.02,
                 str(val), ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=TEXT_COL)
    ax2.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 5)
    ax2.tick_params(axis="x", colors=TEXT_COL, labelsize=10)
    _grid(ax2)

    # ── Panel 3: Visit duration histogram ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Visit Duration Distribution")
    durations = metrics["durations"]
    if durations and len(durations) > 1:
        ax3.hist(durations, bins=min(20, len(durations)),
                 color=ACCENT, edgecolor="#000", linewidth=0.6, alpha=0.85)
        ax3.axvline(np.mean(durations), color=ACCENT2, linewidth=2,
                    linestyle="--",
                    label=f"Mean: {np.mean(durations):.1f}s")
        ax3.legend(facecolor=DARK_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Insufficient data\n(need matched entry+exit pairs)",
                 ha="center", va="center", color="#888888",
                 transform=ax3.transAxes, fontsize=10)
    ax3.set_xlabel("Duration (seconds)", color=TEXT_COL, fontsize=9)
    ax3.set_ylabel("Count",              color=TEXT_COL, fontsize=9)
    _grid(ax3)

    # ── Panel 4: events.log breakdown ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "events.log Breakdown")
    ec     = metrics["log_event_counts"]
    labels = [k for k, v in ec.items() if v > 0]
    vals4  = [ec[k] for k in labels]
    if vals4:
        bcolors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        hb = ax4.barh(labels, vals4, color=bcolors,
                      edgecolor="#000", linewidth=0.6)
        for bar, val in zip(hb, vals4):
            ax4.text(bar.get_width() + max(vals4) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=9,
                     color=TEXT_COL, fontweight="bold")
        ax4.set_xlim(0, max(vals4) * 1.20)
    else:
        ax4.text(0.5, 0.5, "No log data found",
                 ha="center", va="center", color="#888888",
                 transform=ax4.transAxes, fontsize=10)
    _grid(ax4, axis="x")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Pipeline metrics chart saved → {out_path}")


# ─────────────────────────── Model metrics chart ─────────────────────────────

def generate_model_chart(model: dict, out_path: str, video_name: str):
    """
    4-panel model performance chart:
      Top-left    : YOLO detection confidence distribution
      Top-right   : Embedding similarity score distribution
      Bottom-left : Track length (frames) distribution
      Bottom-right: Detections-per-frame over time (sampled)
    """
    if not model:
        print("✗ model_metrics.json not found — skipping model chart")
        return

    confs   = model.get("detection_confidences", [])
    sims    = model.get("similarity_scores", [])
    lengths = model.get("track_lengths", [])
    dpf     = model.get("detections_per_frame", [])
    stats   = model.get("stats", {})

    fig = plt.figure(figsize=(14, 9), facecolor=FIG_BG)
    fig.suptitle(
        f"Model Performance Metrics — {video_name}",
        fontsize=15, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: YOLO confidence distribution ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "YOLO Detection Confidence")
    if confs:
        ax1.hist(confs, bins=20, range=(0, 1),
                 color=ACCENT, edgecolor="#000", linewidth=0.6, alpha=0.85)
        mean_c = np.mean(confs)
        ax1.axvline(mean_c, color=ACCENT2, linewidth=2, linestyle="--",
                    label=f"Mean: {mean_c:.3f}")
        ax1.legend(facecolor=DARK_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=9)
        # Annotate key stats
        ax1.text(0.97, 0.92,
                 f"n={len(confs)}\nmin={stats.get('min_confidence',0):.3f}\nmax={stats.get('max_confidence',0):.3f}",
                 ha="right", va="top", transform=ax1.transAxes,
                 fontsize=8, color=TEXT_COL,
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL,
                           boxstyle="round,pad=0.3"))
    else:
        ax1.text(0.5, 0.5, "No detection data",
                 ha="center", va="center", color="#888888",
                 transform=ax1.transAxes, fontsize=10)
    ax1.set_xlabel("Confidence Score", color=TEXT_COL, fontsize=9)
    ax1.set_ylabel("Count",            color=TEXT_COL, fontsize=9)
    ax1.set_xlim(0, 1)
    _grid(ax1)

    # ── Panel 2: Embedding similarity distribution ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "Embedding Similarity Scores")
    if sims:
        ax2.hist(sims, bins=20, range=(0, 1),
                 color=ACCENT4, edgecolor="#000", linewidth=0.6, alpha=0.85)
        mean_s = np.mean(sims)
        # Draw recognition threshold line
        threshold = 0.38   # matches config default
        ax2.axvline(mean_s, color=ACCENT2, linewidth=2, linestyle="--",
                    label=f"Mean: {mean_s:.3f}")
        ax2.axvline(threshold, color=ACCENT3, linewidth=1.5,
                    linestyle=":", label=f"Threshold: {threshold}")
        ax2.legend(facecolor=DARK_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=9)
        ax2.text(0.97, 0.92,
                 f"n={len(sims)}\nmatches={sum(1 for s in sims if s >= threshold)}",
                 ha="right", va="top", transform=ax2.transAxes,
                 fontsize=8, color=TEXT_COL,
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL,
                           boxstyle="round,pad=0.3"))
    else:
        ax2.text(0.5, 0.5, "No similarity data",
                 ha="center", va="center", color="#888888",
                 transform=ax2.transAxes, fontsize=10)
    ax2.set_xlabel("Cosine Similarity", color=TEXT_COL, fontsize=9)
    ax2.set_ylabel("Count",             color=TEXT_COL, fontsize=9)
    ax2.set_xlim(0, 1)
    _grid(ax2)

    # ── Panel 3: Track length distribution ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "Track Length Distribution (frames)")
    if lengths:
        ax3.hist(lengths, bins=min(20, len(lengths)),
                 color=ACCENT2, edgecolor="#000", linewidth=0.6, alpha=0.85)
        mean_l = np.mean(lengths)
        ax3.axvline(mean_l, color=ACCENT3, linewidth=2, linestyle="--",
                    label=f"Mean: {mean_l:.1f} frames")
        ax3.legend(facecolor=DARK_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=9)
        ax3.text(0.97, 0.92,
                 f"n={len(lengths)}\nmax={max(lengths)} frames",
                 ha="right", va="top", transform=ax3.transAxes,
                 fontsize=8, color=TEXT_COL,
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL,
                           boxstyle="round,pad=0.3"))
    else:
        ax3.text(0.5, 0.5, "No track data",
                 ha="center", va="center", color="#888888",
                 transform=ax3.transAxes, fontsize=10)
    ax3.set_xlabel("Track Length (frames)", color=TEXT_COL, fontsize=9)
    ax3.set_ylabel("Count",                 color=TEXT_COL, fontsize=9)
    _grid(ax3)

    # ── Panel 4: Detections per frame over time ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "Detections per Frame (over time)")
    if dpf:
        # Downsample to max 500 points for readability
        step  = max(1, len(dpf) // 500)
        x     = list(range(0, len(dpf), step))
        y     = [dpf[i] for i in x]
        ax4.plot(x, y, color=ACCENT, linewidth=0.8, alpha=0.8)
        # Rolling mean for trend
        if len(y) >= 10:
            window = max(5, len(y) // 20)
            kernel = np.ones(window) / window
            smooth = np.convolve(y, kernel, mode='valid')
            xs     = x[:len(smooth)]
            ax4.plot(xs, smooth, color=ACCENT3, linewidth=1.5,
                     label=f"Trend (w={window})")
            ax4.legend(facecolor=DARK_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9)
        ax4.text(0.97, 0.92,
                 f"avg={stats.get('avg_detections_per_frame', 0):.2f}/frame",
                 ha="right", va="top", transform=ax4.transAxes,
                 fontsize=8, color=TEXT_COL,
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL,
                           boxstyle="round,pad=0.3"))
    else:
        ax4.text(0.5, 0.5, "No per-frame data",
                 ha="center", va="center", color="#888888",
                 transform=ax4.transAxes, fontsize=10)
    ax4.set_xlabel("Frame number",    color=TEXT_COL, fontsize=9)
    ax4.set_ylabel("Detections",      color=TEXT_COL, fontsize=9)
    _grid(ax4)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Model metrics chart saved → {out_path}")


# ─────────────────────────── Entry point ─────────────────────────────────────

def run(output_dir: str):
    output_dir   = Path(output_dir)
    db_path      = output_dir / "database" / "face_tracker.db"
    log_path     = output_dir / "logs"     / "events.log"
    model_path   = output_dir / "model_metrics.json"
    pipeline_chart = output_dir / "metrics_report.png"
    model_chart    = output_dir / "model_metrics_report.png"
    summary_path   = output_dir / "metrics_summary.txt"
    video_name   = output_dir.name

    if not db_path.exists():
        print(f"✗ Database not found: {db_path}")
        return

    faces, events, total_unique = load_db(str(db_path))
    log_lines                   = load_log(str(log_path))
    model_data                  = load_model_metrics(str(model_path))

    pipeline_metrics = compute_pipeline_metrics(
        faces, events, total_unique, log_lines)

    write_text_summary(
        pipeline_metrics, model_data, str(summary_path), video_name)

    generate_pipeline_chart(
        pipeline_metrics, str(pipeline_chart), video_name)

    generate_model_chart(
        model_data, str(model_chart), video_name)

    return pipeline_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metrics reports for a processed video.")
    parser.add_argument(
        "--output_dir", required=True,
        help="Path to the video output folder (contains database/ and logs/)")
    args = parser.parse_args()
    run(args.output_dir)
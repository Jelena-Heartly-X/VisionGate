"""
scripts/report.py
CLI tool to query the SQLite database and print a visitor report.

Usage:
    python scripts/report.py
    python scripts/report.py --db database/face_tracker.db
    python scripts/report.py --events --limit 20
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Face Tracker Report Generator")
    p.add_argument("--db", default="database/face_tracker.db", help="Path to SQLite DB")
    p.add_argument("--events", action="store_true", help="Show recent events")
    p.add_argument("--limit", type=int, default=20, help="Max rows to display")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 60)
    print("  FACE TRACKER — SESSION REPORT")
    print("=" * 60)

    # ── Unique visitor count ──────────────────────────────────────────
    row = conn.execute("SELECT total_unique, last_updated FROM visitor_summary WHERE id=1").fetchone()
    if row:
        print(f"\n  Total Unique Visitors : {row['total_unique']}")
        print(f"  Last Updated          : {row['last_updated']}")

    # ── Registered faces ─────────────────────────────────────────────
    faces = conn.execute(
        "SELECT face_id, first_seen, last_seen, entry_count FROM faces ORDER BY first_seen LIMIT ?",
        (args.limit,),
    ).fetchall()
    print(f"\n  Registered Faces ({len(faces)} shown, limit={args.limit}):")
    print(f"  {'Face ID':<22} {'First Seen':<22} {'Last Seen':<22} {'Visits'}")
    print("  " + "-" * 78)
    for f in faces:
        print(
            f"  {f['face_id']:<22} {f['first_seen']:<22} {f['last_seen']:<22} {f['entry_count']}"
        )

    # ── Recent events ─────────────────────────────────────────────────
    if args.events:
        events = conn.execute(
            "SELECT face_id, event_type, timestamp, image_path FROM events ORDER BY timestamp DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        print(f"\n  Recent Events ({len(events)} shown):")
        print(f"  {'Face ID':<22} {'Type':<8} {'Timestamp':<22} {'Image'}")
        print("  " + "-" * 90)
        for e in events:
            img = (e["image_path"] or "")[-40:] or "—"
            print(f"  {e['face_id']:<22} {e['event_type']:<8} {e['timestamp']:<22} {img}")

    print("\n" + "=" * 60 + "\n")
    conn.close()


if __name__ == "__main__":
    main()

"""
database_manager.py
Handles all database operations: face registration, event logging, visitor counting.
Supports SQLite (default), MongoDB, and PostgreSQL.

Key fix: All public methods that write timestamps now accept an explicit
`timestamp` parameter (datetime). This means the pipeline passes its
video-derived timestamp directly — no monkey-patching needed.
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Thread-safe database manager for the face tracking system.
    Default backend: SQLite (zero-config, file-based).
    Switch via config.database.type to 'mongodb' or 'postgresql'.
    """

    def __init__(self, config: dict):
        self.config  = config["database"]
        self.db_type = self.config.get("type", "sqlite").lower()
        self._lock   = threading.Lock()
        self._connection = None

        if self.db_type == "sqlite":
            self._init_sqlite()
        elif self.db_type == "mongodb":
            self._init_mongo()
        elif self.db_type == "postgresql":
            self._init_postgres()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        logger.info(f"DatabaseManager initialised with backend: {self.db_type}")

    # ─────────────────────────── SQLite ─────────────────────────────────

    def _init_sqlite(self):
        db_path = Path(self.config.get("sqlite_path", "database/face_tracker.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_path = str(db_path)
        self._create_sqlite_schema()
        logger.info(f"SQLite database initialised at: {db_path}")

    def _get_sqlite_conn(self):
        """Return a shared SQLite connection (protected by self._lock)."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self._sqlite_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection

    def _create_sqlite_schema(self):
        """Create all required tables if they don't already exist."""
        conn = sqlite3.connect(self._sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id     TEXT PRIMARY KEY,
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                entry_count INTEGER DEFAULT 1,
                embedding   TEXT,       -- JSON-serialised float list
                thumbnail   TEXT        -- path to first captured image
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id     TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,   -- 'entry' | 'exit'
                timestamp   TEXT    NOT NULL,   -- video-derived timestamp
                image_path  TEXT,
                track_id    INTEGER,
                FOREIGN KEY (face_id) REFERENCES faces(face_id)
            );

            CREATE TABLE IF NOT EXISTS visitor_summary (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                total_unique INTEGER DEFAULT 0,
                last_updated TEXT
            );

            INSERT OR IGNORE INTO visitor_summary (id, total_unique, last_updated)
            VALUES (1, 0, datetime('now'));
        """)
        conn.commit()
        conn.close()

    # ─────────────── Public API ──────────────────────────────────────────

    def register_face(
        self,
        face_id:        str,
        embedding:      List[float],
        thumbnail_path: str,
        timestamp:      Optional[datetime] = None,
    ) -> bool:
        """
        Insert a new face record.

        Args:
            face_id:        Unique face identifier.
            embedding:      512-d float list (ArcFace embedding).
            thumbnail_path: Path to the entry image.
            timestamp:      Video-derived datetime; defaults to wall clock.

        Returns:
            True on successful insert, False if face_id already exists.
        """
        now            = (timestamp or datetime.now()).isoformat()
        embedding_json = json.dumps(embedding)

        with self._lock:
            conn = self._get_sqlite_conn()
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO faces
                       (face_id, first_seen, last_seen, entry_count, embedding, thumbnail)
                       VALUES (?, ?, ?, 1, ?, ?)""",
                    (face_id, now, now, embedding_json, thumbnail_path),
                )
                inserted = conn.total_changes > 0
                if inserted:
                    conn.execute(
                        """UPDATE visitor_summary
                           SET total_unique = total_unique + 1, last_updated = ?""",
                        (now,),
                    )
                conn.commit()
                return inserted
            except Exception as e:
                logger.error(f"register_face error: {e}")
                conn.rollback()
                return False

    def update_face_last_seen(
        self,
        face_id:   str,
        timestamp: Optional[datetime] = None,
    ):
        """
        Update last_seen and increment entry_count for a returning face.

        Args:
            face_id:   Face to update.
            timestamp: Video-derived datetime; defaults to wall clock.
        """
        now = (timestamp or datetime.now()).isoformat()
        with self._lock:
            conn = self._get_sqlite_conn()
            try:
                conn.execute(
                    """UPDATE faces
                       SET last_seen = ?, entry_count = entry_count + 1
                       WHERE face_id = ?""",
                    (now, face_id),
                )
                conn.commit()
            except Exception as e:
                logger.error(f"update_face_last_seen error: {e}")

    def log_event(
        self,
        face_id:    str,
        event_type: str,
        image_path: str,
        track_id:   Optional[int]      = None,
        timestamp:  Optional[datetime] = None,
    ):
        """
        Log an entry or exit event to the events table.

        Args:
            face_id:    UUID of the face.
            event_type: 'entry' or 'exit'.
            image_path: Path to the saved cropped face image.
            track_id:   Tracker-assigned numeric ID (optional).
            timestamp:  Video-derived datetime; defaults to wall clock.
        """
        now = (timestamp or datetime.now()).isoformat()
        with self._lock:
            conn = self._get_sqlite_conn()
            try:
                conn.execute(
                    """INSERT INTO events
                       (face_id, event_type, timestamp, image_path, track_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (face_id, event_type, now, image_path, track_id),
                )
                conn.commit()
            except Exception as e:
                logger.error(f"log_event error: {e}")

    # ─────────────── Read API ─────────────────────────────────────────────

    def get_all_faces(self) -> List[Dict[str, Any]]:
        """Return all registered faces as a list of dicts (no embeddings)."""
        with self._lock:
            conn = self._get_sqlite_conn()
            rows = conn.execute(
                """SELECT face_id, first_seen, last_seen, entry_count, thumbnail
                   FROM faces ORDER BY first_seen"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_events(
        self,
        limit:      int           = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent events, optionally filtered by type."""
        with self._lock:
            conn = self._get_sqlite_conn()
            if event_type:
                rows = conn.execute(
                    """SELECT * FROM events WHERE event_type = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_unique_visitor_count(self) -> int:
        """Return the total number of unique faces registered."""
        with self._lock:
            conn = self._get_sqlite_conn()
            row  = conn.execute(
                "SELECT total_unique FROM visitor_summary WHERE id = 1"
            ).fetchone()
            return row["total_unique"] if row else 0

    def face_exists(self, face_id: str) -> bool:
        """Check whether a face_id is already in the database."""
        with self._lock:
            conn = self._get_sqlite_conn()
            row  = conn.execute(
                "SELECT 1 FROM faces WHERE face_id = ?", (face_id,)
            ).fetchone()
            return row is not None

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Return all face_id + embedding pairs for nearest-neighbour matching."""
        with self._lock:
            conn = self._get_sqlite_conn()
            rows = conn.execute(
                "SELECT face_id, embedding FROM faces"
            ).fetchall()
            result = []
            for r in rows:
                try:
                    emb = json.loads(r["embedding"])
                    result.append({"face_id": r["face_id"], "embedding": emb})
                except Exception:
                    pass
            return result

    def close(self):
        """Close the database connection gracefully."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed.")

    # ─────────────── MongoDB stub ─────────────────────────────────────────

    def _init_mongo(self):
        try:
            from pymongo import MongoClient
            uri    = self.config.get(
                "mongo_uri", "mongodb://localhost:27017/face_tracker")
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            db     = client.get_default_database()
            self._mongo_faces   = db["faces"]
            self._mongo_events  = db["events"]
            self._mongo_summary = db["visitor_summary"]
            self._mongo_faces.create_index("face_id", unique=True)
            self._mongo_events.create_index([("timestamp", -1)])
            logger.info("MongoDB connected.")
        except Exception as e:
            raise RuntimeError(f"MongoDB connection failed: {e}")

    # ─────────────── PostgreSQL stub ──────────────────────────────────────

    def _init_postgres(self):
        try:
            import psycopg2
            uri          = self.config.get("postgres_uri")
            self._pg_conn = psycopg2.connect(uri)
            logger.info("PostgreSQL connected.")
        except Exception as e:
            raise RuntimeError(f"PostgreSQL connection failed: {e}")
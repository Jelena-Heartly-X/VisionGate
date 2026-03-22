"""
face_tracker.py
Multi-object face tracker with built-in Kalman-filter SORT implementation.

No external tracking packages required.
SORT (Simple Online and Realtime Tracking) uses a Kalman filter to predict
each track's position between detections, which dramatically reduces
fragmentation compared to pure IoU matching — especially on CCTV footage
where detection skips frames or faces briefly occlude each other.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ── Kalman Filter ────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Kalman filter-based tracker for a single bounding box.
    State: [cx, cy, s, r, dcx, dcy, ds]  (centre x/y, scale, aspect ratio, velocities)
    Observation: [cx, cy, s, r]
    """
    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        from numpy.linalg import norm
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)

        # Observation matrix
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)

        # Covariances
        self.R = np.eye(4, dtype=np.float32) * 1.0   # measurement noise
        self.Q = np.eye(7, dtype=np.float32) * 0.01  # process noise
        self.Q[4:, 4:] *= 10.0  # higher noise on velocities

        self.P = np.eye(7, dtype=np.float32)
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities

        # Initialise state from bbox
        obs = self._bbox_to_obs(bbox)
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = obs.reshape(4, 1)

        KalmanBoxTracker.count += 1
        self.id          = KalmanBoxTracker.count
        self.hits        = 1
        self.age         = 0
        self.hit_streak  = 1
        self.time_since_update = 0

    def _bbox_to_obs(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s  = (x2 - x1) * (y2 - y1)        # area as scale
        r  = (x2 - x1) / float(y2 - y1 + 1e-6)   # aspect ratio
        return np.array([cx, cy, s, r], dtype=np.float32)

    def _obs_to_bbox(self, obs):
        cx, cy, s, r = obs.flatten()[:4]
        w = np.sqrt(max(s * r, 1.0))
        h = max(s / (w + 1e-6), 1.0)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)

    def predict(self):
        """Advance the Kalman filter one step (no measurement)."""
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0   # prevent negative scale velocity
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0
        return self._obs_to_bbox(self.x[:4])

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update the Kalman filter with a new measurement."""
        z = self._bbox_to_obs(bbox).reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.age = 0

    def get_bbox(self) -> Tuple[int, int, int, int]:
        return self._obs_to_bbox(self.x[:4])


# ── IoU helper ───────────────────────────────────────────────────────────────

def _iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter + 1e-8)


def _iou_matrix(trackers, detections):
    iou = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for i, t in enumerate(trackers):
        for j, d in enumerate(detections):
            iou[i, j] = _iou(t, d)
    return iou


# ── Track dataclass (same interface as before) ────────────────────────────────

@dataclass
class Track:
    """Represents an active face track."""
    track_id: int
    bbox:     Tuple[int, int, int, int]
    face_id:  Optional[str] = None
    age:      int = 0
    hits:     int = 1
    history:  List[Tuple] = field(default_factory=list)
    _kf:      object = field(default=None, repr=False)   # KalmanBoxTracker

    def update(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.age  = 0
        self.hits += 1
        self.history.append(bbox)
        if len(self.history) > 50:
            self.history.pop(0)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)


# ── SORT Tracker ─────────────────────────────────────────────────────────────

class FaceTracker:
    """
    SORT tracker with Kalman filter prediction.
    Dramatically reduces track fragmentation vs pure IoU matching because
    the Kalman filter predicts where each face will be in the next frame,
    so even when YOLO misses a detection for 1-2 frames the track survives.
    """

    def __init__(self, config: dict):
        track_cfg = config["tracking"]
        self.iou_threshold   = track_cfg.get("iou_threshold",   0.15)
        self.max_lost_frames = track_cfg.get("max_lost_frames", 30)
        self.min_hits        = track_cfg.get("min_hits",        2)

        self._kalman_tracks: Dict[int, KalmanBoxTracker] = {}
        self._track_meta:    Dict[int, Track]            = {}
        self._next_id = 1

        logger.info(
            f"FaceTracker initialised: type=SORT(Kalman) "
            f"iou_thresh={self.iou_threshold} "
            f"max_lost={self.max_lost_frames}")

    # ── Public API (same as before — pipeline doesn't change) ────────────

    def update(
        self,
        detections: List[Tuple[int, int, int, int]],
        confidences: Optional[List[float]] = None,
    ) -> Tuple[List[Track], List[Track]]:
        """
        Update tracker with current-frame detections.
        Returns (active_tracks, exited_tracks_this_frame).
        """
        # 1. Predict new bbox for every existing Kalman track
        predicted = {}
        for tid, kf in self._kalman_tracks.items():
            predicted[tid] = kf.predict()

        exited_this_frame: List[Track] = []

        if not detections:
            # No detections — age all tracks and expire old ones
            for tid in list(self._kalman_tracks.keys()):
                kf   = self._kalman_tracks[tid]
                meta = self._track_meta[tid]
                meta.age += 1
                if kf.time_since_update > self.max_lost_frames:
                    exited_this_frame.append(meta)
                    del self._kalman_tracks[tid]
                    del self._track_meta[tid]
            return list(self._track_meta.values()), exited_this_frame

        # 2. Build IoU cost matrix between predicted positions and detections
        tids      = list(predicted.keys())
        pred_boxes = [predicted[tid] for tid in tids]

        iou_mat = _iou_matrix(pred_boxes, detections)

        # 3. Hungarian algorithm matching
        if iou_mat.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        matched_tids = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self.iou_threshold:
                tid  = tids[r]
                bbox = detections[c]
                self._kalman_tracks[tid].update(bbox)
                meta      = self._track_meta[tid]
                meta.bbox = self._kalman_tracks[tid].get_bbox()
                meta.hits = self._kalman_tracks[tid].hits
                meta.age  = 0
                meta.history.append(bbox)
                matched_tids.add(tid)
                matched_dets.add(c)

        # 4. Unmatched detections → new tracks
        for ci, det in enumerate(detections):
            if ci not in matched_dets:
                kf  = KalmanBoxTracker(det)
                tid = self._next_id
                self._next_id += 1
                # Reset class counter to use our own IDs
                kf.id = tid
                self._kalman_tracks[tid] = kf
                self._track_meta[tid]    = Track(
                    track_id=tid, bbox=det, hits=1, _kf=kf)

        # 5. Expire lost tracks
        for tid in list(self._kalman_tracks.keys()):
            kf   = self._kalman_tracks[tid]
            meta = self._track_meta[tid]
            if tid not in matched_tids:
                meta.age += 1
            if kf.time_since_update > self.max_lost_frames:
                exited_this_frame.append(meta)
                del self._kalman_tracks[tid]
                del self._track_meta[tid]

        # 6. Return only tracks that have been confirmed (min_hits)
        active = [
            meta for tid, meta in self._track_meta.items()
            if self._kalman_tracks[tid].hit_streak >= self.min_hits
            or meta.face_id is not None   # always keep identified tracks
        ]

        return active, exited_this_frame

    def assign_face_id(self, track_id: int, face_id: str):
        if track_id in self._track_meta:
            self._track_meta[track_id].face_id = face_id

    def get_active_tracks(self) -> List[Track]:
        return list(self._track_meta.values())

    def get_face_id(self, track_id: int) -> Optional[str]:
        m = self._track_meta.get(track_id)
        return m.face_id if m else None

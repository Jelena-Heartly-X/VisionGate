"""
face_recognizer.py
Generates 512-d facial embeddings using InsightFace (ArcFace backbone).
Provides nearest-neighbour matching against a registered face database.
"""

import logging
import uuid
import numpy as np
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Wraps InsightFace for embedding generation and face matching.
    Falls back to a mock when InsightFace is not installed.
    """

    def __init__(self, config: dict):
        rec_cfg          = config["recognition"]
        self.model_name  = rec_cfg.get("model_name", "buffalo_l")
        self.threshold   = rec_cfg.get("embedding_similarity_threshold", 0.45)
        self.ctx_id      = rec_cfg.get("ctx_id", 0)   # 0 = GPU, -1 = CPU

        self._app       = None
        self._use_mock  = False
        self._load_model()

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            logger.info(
                f"InsightFace model '{self.model_name}' loaded on ctx_id={self.ctx_id}")
        except ImportError:
            logger.warning(
                "InsightFace not installed. Using mock embeddings. "
                "Install: pip install insightface onnxruntime-gpu")
            self._use_mock = True
        except Exception as e:
            logger.error(f"InsightFace load error: {e}. Falling back to mock.")
            self._use_mock = True

    # ── Public API ───────────────────────────────────────────────────────

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d L2-normalised embedding from a cropped face image.

        IMPORTANT: Always resize to 112x112 before passing to InsightFace.
        ArcFace's native input size is 112x112. Passing variable-size crops
        causes InsightFace to re-run its internal detector which frequently
        fails on side profiles and small/partial faces from CCTV footage.
        Fixing the size forces InsightFace to treat the whole crop as a face.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        if self._use_mock:
            return self._mock_embedding(face_crop)

        try:
            import cv2

            # Always resize to ArcFace native size (112x112).
            # This bypasses InsightFace's internal detector failure on crops.
            resized = cv2.resize(face_crop, (112, 112),
                                 interpolation=cv2.INTER_LINEAR)

            faces = self._app.get(resized)
            if faces:
                # Use highest-confidence detection from the resized crop
                face = max(faces, key=lambda f: f.det_score)
                emb  = np.array(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0.1:
                    return emb / norm

            # InsightFace still found nothing at 112x112 — try 224x224
            # (some ONNX versions handle upscaled crops better)
            resized2 = cv2.resize(face_crop, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
            faces2 = self._app.get(resized2)
            if faces2:
                face = max(faces2, key=lambda f: f.det_score)
                emb  = np.array(face.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0.1:
                    return emb / norm

            logger.debug("InsightFace found no face in crop at 112 or 224px.")
            return None

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        registered_embeddings: List[Dict],
    ) -> Tuple[Optional[str], float]:
        """
        Find the closest registered face to the query embedding using
        cosine similarity on L2-normalised vectors.
        """
        if not registered_embeddings or query_embedding is None:
            return None, 0.0

        best_id  = None
        best_sim = -1.0

        q      = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        for record in registered_embeddings:
            try:
                ref      = np.array(record["embedding"], dtype=np.float32)
                ref_norm = ref / (np.linalg.norm(ref) + 1e-8)
                sim      = float(np.dot(q_norm, ref_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_id  = record["face_id"]
            except Exception:
                continue

        if best_sim >= self.threshold:
            logger.debug(f"Match: face_id={best_id} sim={best_sim:.4f}")
            return best_id, best_sim
        else:
            logger.debug(
                f"No match above threshold {self.threshold}. "
                f"Best sim={best_sim:.4f}")
            return None, best_sim

    def generate_new_face_id(self) -> str:
        return f"FACE_{uuid.uuid4().hex[:12].upper()}"

    # ── Mock fallback ────────────────────────────────────────────────────

    def _mock_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2
        small = cv2.resize(face_crop, (32, 32)).astype(np.float32).flatten()
        emb   = np.pad(small, (0, max(0, 512 - len(small))))[:512]
        return emb / (np.linalg.norm(emb) + 1e-8)

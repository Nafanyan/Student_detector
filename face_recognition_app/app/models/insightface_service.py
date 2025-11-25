"""
InsightFace-based face detection and recognition service.

This service uses RetinaFace for detection and ArcFace (r100/r50, etc.)
for embeddings via the `insightface` library. It is designed to work both
on CPU and GPU (if available and configured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.models.face_recognizer import RecognitionResult


@dataclass
class InsightFaceRecognizedFace:
    """Combined detection + recognition result for a single face."""

    box: Tuple[int, int, int, int]  # top, right, bottom, left
    result: RecognitionResult


@dataclass
class InsightFaceService:
    """
    Service that wraps InsightFace's FaceAnalysis.

    - Uses RetinaFace (из пакета моделей, например 'buffalo_l'/'buffalo_s')
      для детекции лиц.
    - Использует ArcFace для получения эмбеддингов и последующего сравнения
      с базой известных лиц.
    """

    model_pack: str
    known_faces_dir: Path
    ctx_id: int = -1  # -1 = CPU, >=0 = GPU index
    det_size: Tuple[int, int] = (640, 640)
    tolerance: float = 0.8  # расстояние/порог для сопоставления эмбеддингов

    _app: FaceAnalysis = field(init=False)
    _encodings: List["np.ndarray"] = field(default_factory=list, init=False)
    _names: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Инициализируем единственный экземпляр FaceAnalysis
        self._app = FaceAnalysis(
            name=self.model_pack,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        self._load_known_faces()

    def _compute_embedding_for_image(self, image_bgr: "np.ndarray") -> "np.ndarray | None":
        """Получить один эмбеддинг лица из изображения (берём самое крупное лицо)."""
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        # Выбираем самое крупное лицо
        def _area(face: object) -> float:
            x1, y1, x2, y2 = face.bbox
            return float((x2 - x1) * (y2 - y1))

        largest_face = max(faces, key=_area)
        return largest_face.embedding

    def _load_known_faces(self) -> None:
        """Собираем базу эмбеддингов известных лиц из директории known_faces."""
        if not self.known_faces_dir.exists():
            return

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for image_path in person_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                embedding = self._compute_embedding_for_image(image)
                if embedding is None:
                    continue
                self._encodings.append(embedding)
                self._names.append(name)

    def _match_embedding(self, embedding: "np.ndarray") -> RecognitionResult:
        """Сопоставить эмбеддинг с базой и вернуть имя + confidence."""
        if not self._encodings:
            return RecognitionResult(name="Unknown", confidence=0.0)

        encodings_array = np.stack(self._encodings)  # (N, D)
        # Эмбеддинги ArcFace обычно L2-нормированы, поэтому можно использовать L2-дистанцию
        diff = encodings_array - embedding
        distances = np.linalg.norm(diff, axis=1)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance <= self.tolerance:
            # Чем меньше расстояние, тем выше уверенность.
            confidence = max(0.0, min(1.0, 1.0 - best_distance / self.tolerance))
            return RecognitionResult(name=self._names[best_index], confidence=confidence)

        return RecognitionResult(name="Unknown", confidence=0.0)

    def analyze(self, frame_bgr: "np.ndarray") -> List[InsightFaceRecognizedFace]:
        """
        Выполнить детекцию и распознавание лиц на кадре.

        Возвращает список объектов с координатами лица и результатом распознавания.
        """
        faces = self._app.get(frame_bgr)
        results: List[InsightFaceRecognizedFace] = []

        for face in faces:
            bbox = face.bbox.astype(int)  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox
            # Конвертируем в формат (top, right, bottom, left) для совместимости с вью
            top, right, bottom, left = y1, x2, y2, x1

            embedding = face.embedding
            recog = self._match_embedding(embedding)

            results.append(
                InsightFaceRecognizedFace(
                    box=(top, right, bottom, left),
                    result=recog,
                )
            )

        return results



"""
Face recognition model.

Responsible for encoding known faces and matching them against detected faces
on incoming frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import face_recognition
import numpy as np


@dataclass
class RecognitionResult:
    """Stores the recognition result for a single face."""

    name: str
    confidence: float  # simple similarity score in range [0, 1]


class FaceRecognizerInterface:
    """Abstraction used by controllers for face recognition."""

    def recognize(
        self,
        frame_rgb: "np.ndarray",
        face_locations: Sequence[Tuple[int, int, int, int]],
    ) -> List[RecognitionResult]:
        """Return recognition results for each face in the same order."""
        raise NotImplementedError


@dataclass
class SimpleFaceRecognizer(FaceRecognizerInterface):
    """
    Simple k-NN style recognizer based on `face_recognition` encodings.

    Known faces are loaded from a directory where each subfolder corresponds
    to a person and contains one or more images.
    """

    known_faces_dir: Path
    tolerance: float = 0.6
    _encodings: List["np.ndarray"] = field(default_factory=list, init=False)
    _names: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        if not self.known_faces_dir.exists():
            return
        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for image_path in person_dir.glob("*.jpg"):
                image = face_recognition.load_image_file(str(image_path))
                encs = face_recognition.face_encodings(image)
                if not encs:
                    continue
                self._encodings.append(encs[0])
                self._names.append(name)

    def recognize(
        self,
        frame_rgb: "np.ndarray",
        face_locations: Sequence[Tuple[int, int, int, int]],
    ) -> List[RecognitionResult]:
        if not self._encodings:
            return [RecognitionResult(name="Unknown", confidence=0.0) for _ in face_locations]

        encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        results: List[RecognitionResult] = []
        for enc in encodings:
            distances = face_recognition.face_distance(self._encodings, enc)
            best_index = int(np.argmin(distances))
            best_distance = float(distances[best_index])
            if best_distance <= self.tolerance:
                # Convert distance into a simple confidence measure (1 - normalized distance)
                confidence = max(0.0, min(1.0, 1.0 - best_distance / self.tolerance))
                results.append(
                    RecognitionResult(
                        name=self._names[best_index],
                        confidence=confidence,
                    )
                )
            else:
                results.append(RecognitionResult(name="Unknown", confidence=0.0))

        # Ensure the result count matches faces count
        if len(results) < len(face_locations):
            results.extend(
                RecognitionResult(name="Unknown", confidence=0.0)
                for _ in range(len(face_locations) - len(results))
            )

        return results



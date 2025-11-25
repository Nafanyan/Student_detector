"""
Face detection model.

Encapsulates the logic of finding face locations in a frame.
Can be backed by different algorithms (Haar cascades, HOG, DNN, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import cv2
import face_recognition
import numpy as np


FaceLocation = Tuple[int, int, int, int]  # top, right, bottom, left


class FaceDetectorInterface(Protocol):
    """Abstraction used by controllers to detect faces on images."""

    def detect_faces(self, frame_bgr: "np.ndarray") -> List[FaceLocation]:
        """Return list of face bounding boxes (top, right, bottom, left)."""


@dataclass
class SimpleFaceDetector(FaceDetectorInterface):
    """
    Simple detector using `face_recognition.face_locations`.

    This implementation is CPU-friendly and uses a downscaled copy of the
    frame for faster processing.
    """

    scale_factor: float = 0.25
    model: str = "hog"  # or "cnn" if GPU is available

    def detect_faces(self, frame_bgr: "np.ndarray") -> List[FaceLocation]:
        rgb_small = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.scale_factor != 1.0:
            rgb_small = cv2.resize(
                rgb_small,
                (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
            )
        boxes_small: Sequence[FaceLocation] = face_recognition.face_locations(rgb_small, model=self.model)
        # Rescale boxes to original frame size
        factor = 1.0 / self.scale_factor if self.scale_factor != 0 else 1.0
        boxes: List[FaceLocation] = []
        for top, right, bottom, left in boxes_small:
            boxes.append(
                (
                    int(top * factor),
                    int(right * factor),
                    int(bottom * factor),
                    int(left * factor),
                )
            )
        return boxes



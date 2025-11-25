"""
VideoView is responsible for visualizing frames and overlays.

This class contains all OpenCV UI calls (imshow, waitKey) and has no knowledge
of how faces are detected or recognized (Single Responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

from app.models.face_recognizer import RecognitionResult


@dataclass
class VideoOverlayFace:
    """Data needed by the view to draw a single face."""

    box: Tuple[int, int, int, int]  # top, right, bottom, left
    label: str
    confidence: float


@dataclass
class VideoView:
    """Simple OpenCV-based video window."""

    window_title: str = "Face Recognition App"

    def show_frame(
        self,
        frame_bgr: "np.ndarray",
        faces: Sequence[VideoOverlayFace],
        fps: float | None = None,
    ) -> None:
        """Draw overlays and show the frame."""
        output = frame_bgr.copy()

        for face in faces:
            top, right, bottom, left = face.box
            color = (0, 255, 0) if face.label != "Unknown" else (0, 0, 255)
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            label = f"{face.label} ({face.confidence*100:.0f}%)"
            cv2.rectangle(output, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                output,
                label,
                (left + 2, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if fps is not None:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.imshow(self.window_title, output)

    def process_events(self) -> bool:
        """
        Process window events.

        Returns True if the app should continue running, False if the user
        requested to quit (e.g., pressed 'q' or ESC).
        """
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return True

    def close(self) -> None:
        """Destroy OpenCV windows."""
        cv2.destroyAllWindows()



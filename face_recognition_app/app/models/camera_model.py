"""
Camera model responsible only for video capture.

This class provides a small, typed interface so that controllers can depend on
the abstraction instead of raw OpenCV calls (Single Responsibility, DIP).
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import cv2


class CameraInterface(Protocol):
    """Abstraction of a video source."""

    def read(self) -> Tuple[bool, "cv2.Mat"]:
        """Return (success, frame) from the video source."""

    def release(self) -> None:
        """Release underlying resources."""


class OpenCVCamera(CameraInterface):
    """
    OpenCV-based camera implementation.

    Responsible only for capturing frames from a camera index.
    """

    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self._cap = cv2.VideoCapture(index)
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[bool, "cv2.Mat"]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()



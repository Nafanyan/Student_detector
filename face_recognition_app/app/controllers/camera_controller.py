"""
Camera controller.

Connects the camera model with the rest of the application. The controller
itself does not know how faces are detected or recognized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2

from app.models.camera_model import CameraInterface


@dataclass
class CameraController:
    """Controller responsible for retrieving frames from a camera model."""

    camera: CameraInterface

    def get_frame(self) -> Tuple[bool, "cv2.Mat"]:
        """Return the latest frame from the camera."""
        return self.camera.read()

    def shutdown(self) -> None:
        """Release camera resources."""
        self.camera.release()



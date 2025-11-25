"""
Face controller.

Coordinates face detection and recognition. This layer converts low-level
model outputs into high-level data structures suitable for the view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.models.insightface_service import InsightFaceRecognizedFace, InsightFaceService
from app.views.video_view import VideoOverlayFace


@dataclass
class FaceController:
    """Controller that uses InsightFaceService for detection and recognition."""

    service: InsightFaceService

    def process_frame(self, frame_bgr: "np.ndarray") -> List[VideoOverlayFace]:
        """Detect and recognize faces using InsightFace, returning overlay data for the view."""
        faces: List[InsightFaceRecognizedFace] = self.service.analyze(frame_bgr)
        overlays: List[VideoOverlayFace] = []
        for face in faces:
            overlays.append(
                VideoOverlayFace(
                    box=face.box,
                    label=face.result.name,
                    confidence=face.result.confidence,
                )
            )
        return overlays



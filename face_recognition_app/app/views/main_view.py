"""
High-level view for the application.

For now this is a thin wrapper around `VideoView`, but it allows extending the
UI later (e.g., adding status panels or controls) without changing controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.views.video_view import VideoOverlayFace, VideoView


@dataclass
class MainView:
    """Main view that owns video-related subviews."""

    video_view: VideoView

    def render_frame(
        self,
        frame_bgr: "np.ndarray",
        faces: Sequence[VideoOverlayFace],
        fps: float | None = None,
    ) -> None:
        self.video_view.show_frame(frame_bgr, faces, fps)

    def tick(self) -> bool:
        """Process UI events and return whether the app should continue."""
        return self.video_view.process_events()

    def close(self) -> None:
        self.video_view.close()



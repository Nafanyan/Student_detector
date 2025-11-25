"""
Application configuration and constants.

All configurable parameters should be defined here to keep the rest of the
codebase decoupled from concrete values (paths, thresholds, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoConfig:
    """Configuration related to video capture and display."""

    camera_index: int = 0
    width: int = 640
    height: int = 480
    window_title: str = "Face Recognition App"
    target_fps: int = 25


@dataclass(frozen=True)
class FaceDetectionConfig:
    """Configuration for face detection and recognition."""

    model_dir: Path = Path("data/models")
    known_faces_dir: Path = Path("data/known_faces")
    detection_scale_factor: float = 0.25  # legacy resize factor (for non-InsightFace backends)
    recognition_tolerance: float = 0.8  # similarity threshold for recognition
    use_insightface: bool = True
    # InsightFace configuration
    insightface_model_pack: str = "buffalo_l"  # 'buffalo_l' (ArcFace r100), 'buffalo_s' (ArcFace r50), etc.
    insightface_ctx_id: int = -1  # -1 = CPU, 0 = GPU:0, 1 = GPU:1, ...
    insightface_det_size: tuple[int, int] = (640, 640)


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration object."""

    video: VideoConfig
    face: FaceDetectionConfig


# Predefined configurations
CPU_CONFIG = AppConfig(
    video=VideoConfig(),
    face=FaceDetectionConfig(
        model_dir=Path("data/models"),
        known_faces_dir=Path("data/known_faces"),
        detection_scale_factor=0.25,
        recognition_tolerance=0.8,
        use_insightface=True,
        insightface_model_pack="buffalo_s",  # ArcFace r50 — легче и быстрее, подходит для CPU
        insightface_ctx_id=-1,  # CPU
        insightface_det_size=(480, 480),
    ),
)

GPU_CONFIG = AppConfig(
    video=VideoConfig(),
    face=FaceDetectionConfig(
        model_dir=Path("data/models"),
        known_faces_dir=Path("data/known_faces"),
        detection_scale_factor=0.25,
        recognition_tolerance=0.8,
        use_insightface=True,
        insightface_model_pack="buffalo_l",  # ArcFace r100 — максимальная точность, рекомендуется для GPU
        insightface_ctx_id=0,  # GPU:0
        insightface_det_size=(640, 640),
    ),
)

# Active configuration used by the application.
# По умолчанию используется CPU-конфигурация.
# Чтобы переключиться на GPU-режим, замените CPU_CONFIG на GPU_CONFIG:
# CONFIG = GPU_CONFIG
CONFIG = CPU_CONFIG


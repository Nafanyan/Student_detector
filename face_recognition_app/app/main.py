"""
Entry point for the Face Recognition App.

This module wires together the MVC components and starts the main loop.
"""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack

from loguru import logger

# Allow running both as a module (`python -m app.main`) and as a script (`python app/main.py`)
if __package__ is None or __package__ == "":
    # Add project root (parent of `app`) to sys.path so that `import app...` works
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.controllers.camera_controller import CameraController
from app.controllers.face_controller import FaceController
from app.models.camera_model import OpenCVCamera
from app.models.insightface_service import InsightFaceService
from app.utils.config import CONFIG
from app.utils.helpers import measure_fps
from app.views.main_view import MainView
from app.views.video_view import VideoView


def _ensure_running_in_venv() -> None:
    """
    Ensure that the application is running inside the local .venv.

    Это помогает избежать ситуаций, когда используется глобальный Python
    вместо локального виртуального окружения.
    """
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    in_venv = sys.prefix != base_prefix
    venv_name = os.path.basename(sys.prefix)

    if not in_venv or venv_name != ".venv":
        # Жёсткая проверка: останавливаем приложение, чтобы явно подсветить проблему.
        raise RuntimeError(
            f"Application is not running inside the local .venv (sys.prefix={sys.prefix}). "
            "Запускайте приложение через scripts/run_app.ps1 (Windows) или scripts/run_app.sh (Linux/Mac), "
            "чтобы использовать локальное виртуальное окружение."
        )


def run() -> None:
    """Main application function."""
    logger.add("face_recognition_app.log", rotation="1 MB", enqueue=True)
    logger.info("Starting Face Recognition App")
    _ensure_running_in_venv()

    camera = OpenCVCamera(
        index=CONFIG.video.camera_index,
        width=CONFIG.video.width,
        height=CONFIG.video.height,
    )
    camera_controller = CameraController(camera=camera)

    if CONFIG.face.use_insightface:
        logger.info(
            "Using InsightFace backend: model_pack='{}', ctx_id={}, det_size={}",
            CONFIG.face.insightface_model_pack,
            CONFIG.face.insightface_ctx_id,
            CONFIG.face.insightface_det_size,
        )
        service = InsightFaceService(
            model_pack=CONFIG.face.insightface_model_pack,
            known_faces_dir=CONFIG.face.known_faces_dir,
            ctx_id=CONFIG.face.insightface_ctx_id,
            det_size=CONFIG.face.insightface_det_size,
            tolerance=CONFIG.face.recognition_tolerance,
        )
        face_controller = FaceController(service=service)
    else:
        raise RuntimeError("Non-InsightFace backends are not configured in this build. Set use_insightface=True.")

    main_view = MainView(video_view=VideoView(window_title=CONFIG.video.window_title))

    fps_gen = measure_fps()

    try:
        for delta, fps in fps_gen:
            success, frame = camera_controller.get_frame()
            if not success:
                logger.warning("Failed to read frame from camera")
                continue

            overlays = face_controller.process_frame(frame)
            main_view.render_frame(frame, overlays, fps=fps)

            if not main_view.tick():
                logger.info("User requested shutdown")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down")
        main_view.close()
        camera_controller.shutdown()


if __name__ == "__main__":
    run()



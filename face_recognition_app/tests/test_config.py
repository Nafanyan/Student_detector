from app.utils.config import CONFIG


def test_config_paths_exist_attributes() -> None:
    """Basic smoke test for config object structure."""
    assert CONFIG.video.width > 0
    assert CONFIG.face.known_faces_dir is not None



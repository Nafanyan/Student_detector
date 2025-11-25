"""
Helper utilities used across the application.

This module intentionally contains only stateless functions that can be reused
by models, views and controllers without introducing tight coupling.
"""

from __future__ import annotations

from time import perf_counter
from typing import Iterator, Tuple


def measure_fps() -> Iterator[Tuple[float, float]]:
    """
    Simple FPS measurement generator.

    Yields tuples of (delta_time_seconds, fps_estimate).
    """
    last_time = perf_counter()
    while True:
        current_time = perf_counter()
        delta = current_time - last_time
        last_time = current_time
        fps = 1.0 / delta if delta > 0 else 0.0
        yield delta, fps



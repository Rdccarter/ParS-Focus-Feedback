from __future__ import annotations

import threading
import time
from typing import Callable

from .interfaces import CameraInterface, Image2D


def run_live_monitor(
    camera: CameraInterface,
    on_frame: Callable[[Image2D], None],
    stop_event: threading.Event,
    loop_hz: float = 20.0,
) -> None:
    """Read frames in real-time and dispatch to a callback (for GUI display)."""

    dt = 1.0 / loop_hz
    while not stop_event.is_set():
        t0 = time.monotonic()
        frame = camera.get_frame()
        on_frame(frame.image)
        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

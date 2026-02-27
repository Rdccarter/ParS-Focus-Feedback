from __future__ import annotations

import time
from typing import Any, Protocol

from .interfaces import Image2D


class DcamLikeCamera(Protocol):
    """Subset of DCAM-style camera methods used by this adapter."""

    def get_latest_frame(self) -> Any:
        ...


class DcamFrameSource:
    """Frame-source adapter for Hamamatsu DCAM backends.

    Use this with `HamamatsuOrcaCamera(frame_source=...)`:

    ```python
    dcam_source = DcamFrameSource(dcam_camera)
    camera = HamamatsuOrcaCamera(frame_source=dcam_source)
    ```
    """

    def __init__(self, dcam_camera: DcamLikeCamera) -> None:
        self._camera = dcam_camera

    def start(self) -> None:
        start = getattr(self._camera, "start", None)
        if callable(start):
            start()

    def stop(self) -> None:
        stop = getattr(self._camera, "stop", None)
        if callable(stop):
            stop()

    def __call__(self) -> tuple[Image2D, float]:
        frame = self._camera.get_latest_frame()
        image = _to_image_2d(frame)
        return image, time.time()


def _to_image_2d(frame: Any) -> Image2D:
    """Convert common DCAM frame containers to Image2D.

    Supports:
    - Python nested lists/tuples
    - array-like objects exposing `.tolist()` (e.g. numpy arrays)
    """

    if hasattr(frame, "tolist") and callable(frame.tolist):
        frame = frame.tolist()
    if isinstance(frame, tuple):
        frame = list(frame)

    if not isinstance(frame, list):
        raise TypeError("Frame must be a 2D list/tuple or expose tolist()")
    if len(frame) == 0:
        raise ValueError("Frame is empty")

    first_row = frame[0]
    if isinstance(first_row, tuple):
        first_row = list(first_row)
    if not isinstance(first_row, list):
        raise TypeError("Frame must be 2D")

    out: Image2D = []
    width = len(first_row)
    if width == 0:
        raise ValueError("Frame width is zero")

    for row in frame:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise TypeError("Frame rows must be list/tuple")
        if len(row) != width:
            raise ValueError("Frame rows must have equal length")
        out.append([float(px) for px in row])
    return out

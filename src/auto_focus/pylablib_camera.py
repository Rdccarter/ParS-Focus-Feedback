from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .dcam import _to_image_2d
from .interfaces import Image2D


@dataclass(slots=True)
class PylablibFrameSource:
    """Generic frame-source wrapper for pylablib camera objects."""

    camera: Any
    read_frame: Callable[[Any], Any]

    def start(self) -> None:
        start = getattr(self.camera, "start_acquisition", None)
        if callable(start):
            start()

    def stop(self) -> None:
        stop = getattr(self.camera, "stop_acquisition", None)
        if callable(stop):
            stop()

    def __call__(self) -> tuple[Image2D, float]:
        frame = self.read_frame(self.camera)
        image = _to_image_2d(frame)
        return image, time.time()


def _default_read_frame(camera: Any) -> Any:
    candidates = [
        "read_newest_image",
        "read_oldest_image",
        "get_latest_frame",
        "read_multiple_images",
        "snap",
    ]
    for name in candidates:
        fn = getattr(camera, name, None)
        if callable(fn):
            value = fn()
            if name == "read_multiple_images" and isinstance(value, list) and value:
                return value[-1]
            return value
    raise AttributeError("Could not find a compatible pylablib frame-read method")


def create_pylablib_frame_source(camera_kind: str, **camera_kwargs: Any) -> PylablibFrameSource:
    """Create a frame source for either ORCA or Andor iXon via pylablib.

    camera_kind: "orca" | "andor"
    """

    kind = camera_kind.strip().lower()
    try:
        if kind == "orca":
            from pylablib.devices.Hamamatsu import DCAMCamera

            camera = DCAMCamera(**camera_kwargs)
        elif kind == "andor":
            from pylablib.devices.Andor import AndorSDK2Camera

            camera = AndorSDK2Camera(**camera_kwargs)
        else:
            raise ValueError("camera_kind must be either 'orca' or 'andor'")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize pylablib camera backend. Ensure pylablib and"
            " vendor drivers are installed on the microscope PC."
        ) from exc

    return PylablibFrameSource(camera=camera, read_frame=_default_read_frame)

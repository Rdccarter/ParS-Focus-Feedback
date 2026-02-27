from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# Accept list-based images and NumPy-like arrays to avoid expensive
# list<->ndarray conversion churn on high-rate camera streams.
Image2D = list[list[float]] | Any


@dataclass(slots=True)
class CameraFrame:
    """Single camera frame and acquisition metadata."""

    image: Image2D
    timestamp_s: float


class CameraInterface(Protocol):
    """Interface for a streaming camera source."""

    def start(self) -> None:
        """Start acquisition."""

    def stop(self) -> None:
        """Stop acquisition."""

    def get_frame(self) -> CameraFrame:
        """Fetch next frame from the camera stream."""


class StageInterface(Protocol):
    """Interface for an absolute Z stage controller."""

    def get_z_um(self) -> float:
        """Read current stage Z in microns."""

    def move_z_um(self, target_z_um: float) -> None:
        """Command stage to a new absolute Z position in microns."""

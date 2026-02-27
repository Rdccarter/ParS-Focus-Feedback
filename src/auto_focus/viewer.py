from __future__ import annotations

from dataclasses import dataclass

from .focus_metric import Roi
from .interfaces import Image2D


@dataclass(slots=True)
class RoiSelector:
    """Interactive ROI state holder.

    This class is UI-toolkit agnostic. A GUI can call `begin`, `update`, and
    `finalize` from mouse handlers to create an ROI rectangle.
    """

    _x0: int | None = None
    _y0: int | None = None
    roi: Roi | None = None

    def begin(self, x: int, y: int) -> None:
        self._x0 = x
        self._y0 = y

    def update(self, x: int, y: int) -> None:
        if self._x0 is None or self._y0 is None:
            return
        left = min(self._x0, x)
        top = min(self._y0, y)
        width = abs(x - self._x0) + 1
        height = abs(y - self._y0) + 1
        self.roi = Roi(x=left, y=top, width=width, height=height)

    def finalize(self, image: Image2D) -> Roi:
        if self.roi is None:
            raise ValueError("No ROI selected")
        h = len(image)
        w = len(image[0]) if image else 0
        self.roi = self.roi.clamp((h, w))
        self._x0 = None
        self._y0 = None
        return self.roi

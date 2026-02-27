from __future__ import annotations

from .autofocus import AutofocusConfig, CalibrationLike
from .interfaces import CameraInterface, StageInterface


def launch_autofocus_viewer(
    camera: CameraInterface,
    stage: StageInterface,
    *,
    calibration: CalibrationLike,
    default_config: AutofocusConfig,
    interval_ms: int = 20,
    calibration_output_path: str | None = None,
    calibration_half_range_um: float = 0.75,
    calibration_steps: int = 21,
) -> None:
    """Launch the PyQtGraph autofocus GUI."""
    _ = interval_ms
    from .pg_gui import launch_pg_autofocus_gui

    launch_pg_autofocus_gui(
        camera,
        stage,
        calibration=calibration,
        default_config=default_config,
        calibration_output_path=calibration_output_path,
        calibration_half_range_um=calibration_half_range_um,
        calibration_steps=calibration_steps,
    )


def launch_live_viewer(camera: CameraInterface) -> None:
    raise RuntimeError("Standalone live viewer was removed; use launch_autofocus_viewer with stage+calibration")

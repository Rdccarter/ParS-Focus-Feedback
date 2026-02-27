from __future__ import annotations

import ctypes
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .interfaces import CameraFrame, CameraInterface, Image2D, StageInterface


class NotConnectedError(RuntimeError):
    pass


class HamamatsuOrcaCamera(CameraInterface):
    """Camera adapter that can wrap any frame callback.

    Pass a callable returning `(image_2d, timestamp_s)` where image_2d is a
    2D list of pixel intensities. This makes it straightforward to connect to
    Micro-Manager, DCAM Python bindings, or custom SDK wrappers.
    """

    def __init__(
        self,
        frame_source: Callable[[], tuple[Image2D, float]] | None = None,
        control_source_lifecycle: bool = False,
    ) -> None:
        if frame_source is None:
            raise ValueError(
                "frame_source is required for HamamatsuOrcaCamera. "
                "Provide a callable returning (image_2d, timestamp_s)."
            )
        self._running = False
        self._frame_source = frame_source
        self._control_source_lifecycle = control_source_lifecycle

    def start(self) -> None:
        self._running = True
        if self._control_source_lifecycle and self._frame_source is not None:
            start = getattr(self._frame_source, "start", None)
            if callable(start):
                start()

    def stop(self) -> None:
        if self._control_source_lifecycle and self._frame_source is not None:
            stop = getattr(self._frame_source, "stop", None)
            if callable(stop):
                stop()
        self._running = False

    def __enter__(self) -> "HamamatsuOrcaCamera":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()

    def get_frame(self) -> CameraFrame:
        if not self._running:
            raise NotConnectedError("Camera not started")
        if self._frame_source is None:
            raise NotImplementedError(
                "No frame source configured. Provide a callable that returns"
                " (image_2d, timestamp_s) from live acquisition."
            )
        image, ts = self._frame_source()
        return CameraFrame(image=image, timestamp_s=ts)


class MclNanoZStage(StageInterface):
    """Mad City Labs Nano-Z stage adapter.

    Supports 3 integration modes:
    1) ctypes DLL path (`dll_path=...`)
    2) Python wrapper module (`wrapper_module=...`), including
       `MCL_Madlib_Wrapper.MCL_Nanodrive`
    3) in-memory simulated fallback (default)
    """

    def __init__(
        self,
        dll_path: str | None = None,
        wrapper_module: str | ModuleType | Any | None = None,
        axis_index: int = 3,
        wrapper_handle: int | None = None,
    ) -> None:
        self._z_um = 0.0
        self._dll = None
        self._handle: int | None = None
        self._axis = axis_index
        self._wrapper: Any | None = None
        self._wrapper_handle: int | None = wrapper_handle

        if wrapper_module is not None:
            self._connect_wrapper(wrapper_module)
        if dll_path:
            self._connect_sdk(dll_path)

    def _connect_wrapper(self, wrapper_module: str | ModuleType | Any) -> None:
        module_or_obj = importlib.import_module(wrapper_module) if isinstance(wrapper_module, str) else wrapper_module

        # Specific support for the provided Madlib wrapper class name.
        if hasattr(module_or_obj, "MCL_Nanodrive") and callable(getattr(module_or_obj, "MCL_Nanodrive")):
            self._wrapper = module_or_obj.MCL_Nanodrive()
        elif hasattr(module_or_obj, "NanoDrive") and callable(getattr(module_or_obj, "NanoDrive")):
            self._wrapper = module_or_obj.NanoDrive()
        else:
            self._wrapper = module_or_obj

        if self._wrapper_handle is None:
            init_handle = getattr(self._wrapper, "init_handle", None)
            if callable(init_handle):
                self._wrapper_handle = int(init_handle())

    def _connect_sdk(self, dll_path: str) -> None:
        path = Path(dll_path)
        if not path.exists():
            raise FileNotFoundError(f"MCL DLL not found: {dll_path}")

        self._dll = ctypes.CDLL(str(path))
        self._dll.MCL_InitHandle.restype = ctypes.c_int
        self._dll.MCL_SingleReadN.restype = ctypes.c_double
        self._dll.MCL_SingleWriteN.restype = ctypes.c_int

        handle = int(self._dll.MCL_InitHandle())
        if handle <= 0:
            raise NotConnectedError("Failed to initialize MCL handle")
        self._handle = handle

    def _call_wrapper_first(self, names: list[str], arg_options: list[tuple[Any, ...]]) -> Any:
        if self._wrapper is None:
            raise NotConnectedError("No MCL wrapper configured")

        last_type_error: TypeError | None = None
        attempted_callable = False
        for name in names:
            fn = getattr(self._wrapper, name, None)
            if not callable(fn):
                continue
            attempted_callable = True
            for args in arg_options:
                try:
                    return fn(*args)
                except TypeError as exc:
                    last_type_error = exc
                    continue
        if attempted_callable and last_type_error is not None:
            raise RuntimeError(
                f"Wrapper call failed for {names}; last TypeError: {last_type_error}"
            ) from last_type_error
        raise AttributeError(f"No compatible wrapper method/signature found for: {names}")

    def _wrapper_read_z(self) -> float:
        handle = self._wrapper_handle
        value = self._call_wrapper_first(
            ["get_z_um", "read_z", "single_read_z", "single_read_n", "MCL_SingleReadN"],
            [
                (self._axis, handle),
                (self._axis,),
                (handle,),
                (),
            ],
        )
        return float(value)

    def _wrapper_write_z(self, target_z_um: float) -> None:
        handle = self._wrapper_handle
        self._call_wrapper_first(
            ["move_z_um", "write_z", "single_write_z", "single_write_n", "MCL_SingleWriteN"],
            [
                (target_z_um, self._axis, handle),
                (target_z_um, self._axis),
                (target_z_um, handle),
                (target_z_um,),
            ],
        )

    def close(self) -> None:
        if self._wrapper is not None and self._wrapper_handle is not None:
            release = getattr(self._wrapper, "release_handle", None)
            if callable(release):
                try:
                    release(self._wrapper_handle)
                except Exception:
                    pass
            self._wrapper_handle = None

        if self._dll is not None and self._handle is not None:
            release_fn = getattr(self._dll, "MCL_ReleaseHandle", None)
            if callable(release_fn):
                try:
                    release_fn(ctypes.c_int(self._handle))
                except Exception:
                    pass
            self._handle = None

    def __enter__(self) -> "MclNanoZStage":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def get_z_um(self) -> float:
        if self._wrapper is not None:
            self._z_um = self._wrapper_read_z()
            return self._z_um

        if self._dll is None or self._handle is None:
            return self._z_um
        axis = ctypes.c_uint(self._axis)
        value = float(self._dll.MCL_SingleReadN(axis, ctypes.c_int(self._handle)))
        self._z_um = value
        return value

    def get_range_um(self) -> tuple[float, float] | None:
        """Query the hardware travel range for this axis.

        Returns (min_um, max_um) or None if the range cannot be determined.
        MCL piezos always start at 0; max is read from MCL_GetCalibration.
        """
        # Try wrapper first
        if self._wrapper is not None:
            handle = self._wrapper_handle
            for name in ("get_range_um", "get_calibration", "MCL_GetCalibration"):
                fn = getattr(self._wrapper, name, None)
                if not callable(fn):
                    continue
                for args in ((self._axis, handle), (self._axis,), (handle,), ()):
                    try:
                        result = fn(*args)
                        if isinstance(result, (tuple, list)) and len(result) == 2:
                            return (float(result[0]), float(result[1]))
                        val = float(result)
                        if val > 0:
                            return (0.0, val)
                    except (TypeError, ValueError, AttributeError):
                        continue
            return None

        # Try DLL MCL_GetCalibration(axis, handle) -> double (max travel in um)
        if self._dll is not None and self._handle is not None:
            get_cal = getattr(self._dll, "MCL_GetCalibration", None)
            if get_cal is not None:
                try:
                    get_cal.restype = ctypes.c_double
                    max_um = float(get_cal(ctypes.c_uint(self._axis), ctypes.c_int(self._handle)))
                    if max_um > 0:
                        return (0.0, max_um)
                except Exception:
                    pass
            return None

        # Simulated stage: no hardware limits
        return None

    def move_z_um(self, target_z_um: float) -> None:
        if self._wrapper is not None:
            self._wrapper_write_z(target_z_um)
            self._z_um = target_z_um
            return

        if self._dll is None or self._handle is None:
            self._z_um = target_z_um
            return
        axis = ctypes.c_uint(self._axis)
        status = int(self._dll.MCL_SingleWriteN(ctypes.c_double(target_z_um), axis, ctypes.c_int(self._handle)))
        if status != 0:
            hint = ""
            if abs(status) == 6:
                hint = (
                    f" (status {status} often indicates out-of-range/invalid move; "
                    "check stage axis selection and autofocus Z clamps, "
                    "and try --stage-axis 1 or --stage-axis 3 based on your controller)"
                )
            raise RuntimeError(
                f"MCL_SingleWriteN failed with status {status} for target_z_um={target_z_um:+0.6f}, axis={self._axis}.{hint}"
            )
        self._z_um = target_z_um


@dataclass(slots=True)
class SimulatedScene:
    focal_plane_um: float = 0.0
    sigma0_px: float = 1.2
    alpha_px_per_um: float = 0.25

    def render_dot(self, z_um: float, size: int = 64) -> Image2D:
        # Pure-Python O(size^2) simulation for test/demo use; real acquisition paths
        # should use hardware camera frames rather than this renderer.
        cx = cy = (size - 1) / 2

        dz = z_um - self.focal_plane_um
        sigma_x = max(0.6, self.sigma0_px + self.alpha_px_per_um * dz)
        sigma_y = max(0.6, self.sigma0_px - self.alpha_px_per_um * dz)

        try:
            import numpy as np

            y, x = np.mgrid[0:size, 0:size]
            x_term = ((x - cx) ** 2) / (2 * sigma_x**2)
            y_term = ((y - cy) ** 2) / (2 * sigma_y**2)
            return (np.exp(-(x_term + y_term)) * 4095.0).tolist()
        except Exception:
            image: Image2D = []
            for y in range(size):
                row: list[float] = []
                for x in range(size):
                    x_term = ((x - cx) ** 2) / (2 * sigma_x**2)
                    y_term = ((y - cy) ** 2) / (2 * sigma_y**2)
                    row.append(math.exp(-(x_term + y_term)) * 4095.0)
                image.append(row)
            return image


class SimulatedCamera(CameraInterface):
    def __init__(self, stage: StageInterface, scene: SimulatedScene | None = None) -> None:
        self._stage = stage
        self._scene = scene or SimulatedScene()
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False


    def __enter__(self) -> "SimulatedCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.stop()

    def get_frame(self) -> CameraFrame:
        if not self._running:
            raise NotConnectedError("Simulated camera not started")
        z = self._stage.get_z_um()
        image = self._scene.render_dot(z_um=z)
        return CameraFrame(image=image, timestamp_s=time.time())

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

Slot = getattr(QtCore, "Slot", QtCore.pyqtSlot)

from .autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusState, CalibrationLike
from .calibration import (
    CalibrationMetadata,
    CalibrationSample,
    ZhuangCalibrationSample,
    calibration_quality_issues,
    fit_linear_calibration_with_report,
    fit_zhuang_calibration,
    save_calibration_metadata_json,
    save_zhuang_calibration_samples_csv,
    _moment_sigma_fallback,
    zhuang_calibration_quality_issues,
)
from .focus_metric import Roi, astigmatic_error_signal, extract_roi, fit_gaussian_psf, roi_total_intensity
from .interfaces import CameraFrame, CameraInterface, StageInterface
from .ui_signals import AutofocusSignals


UI_THEME = {
    "bg": "#16181c",
    "bg_alt": "#1d2025",
    "panel": "#262a31",
    "panel_alt": "#2d3139",
    "panel_edge": "#3d434d",
    "text": "#edf1f5",
    "muted": "#9fa8b3",
    "focus": "#4ecdc4",
    "focus_bright": "#86fff6",
    "warn": "#f0b35a",
    "fault": "#ef6f6c",
    "ok": "#6bcf8a",
    "cyan": "#6ec5ff",
    "magenta": "#ff7ad9",
    "yellow": "#ffd369",
    "dim": "#65707d",
}


class LatestFrameQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._seq = 0

    def put(self, frame: CameraFrame) -> None:
        with self._lock:
            self._latest = frame
            self._seq += 1

    def get_latest(self) -> tuple[CameraFrame | None, int]:
        with self._lock:
            return self._latest, self._seq


@dataclass(slots=True)
class RunStats:
    loop_latency_ms: deque[float]
    dropped_frames: int = 0
    total_frames: int = 0
    faults: list[str] | None = None




@dataclass(slots=True)
class FrameTransformState:
    rotation_deg: int = 0
    flip_h: bool = False
    flip_v: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def set(self, *, rotation_deg: int | None = None, flip_h: bool | None = None, flip_v: bool | None = None) -> None:
        with self._lock:
            if rotation_deg is not None:
                self.rotation_deg = int(rotation_deg) % 360
            if flip_h is not None:
                self.flip_h = bool(flip_h)
            if flip_v is not None:
                self.flip_v = bool(flip_v)

    def get(self) -> tuple[int, bool, bool]:
        with self._lock:
            return self.rotation_deg, self.flip_h, self.flip_v


def _apply_frame_transform(frame: CameraFrame, transform: FrameTransformState) -> CameraFrame:
    rotation_deg, flip_h, flip_v = transform.get()
    if rotation_deg % 360 == 0 and (not flip_h) and (not flip_v):
        return frame

    image = frame.image
    try:
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 3 and 1 in arr.shape:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")

        k = (rotation_deg // 90) % 4
        if k:
            arr = np.rot90(arr, k=k)
        if flip_h:
            arr = np.fliplr(arr)
        if flip_v:
            arr = np.flipud(arr)
        return CameraFrame(image=arr.copy(), timestamp_s=frame.timestamp_s)
    except ImportError:
        if hasattr(image, "tolist") and callable(image.tolist):
            image = image.tolist()
        if isinstance(image, tuple):
            image = list(image)
        if not isinstance(image, list) or not image or not isinstance(image[0], (list, tuple)):
            raise ValueError("Camera frame must be 2D")
        rows = [list(r) for r in image]

        k = (rotation_deg // 90) % 4
        for _ in range(k):
            rows = [list(col) for col in zip(*rows[::-1])]
        if flip_h:
            rows = [row[::-1] for row in rows]
        if flip_v:
            rows = rows[::-1]
        return CameraFrame(image=rows, timestamp_s=frame.timestamp_s)


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraInterface, frame_queue: LatestFrameQueue, signals: AutofocusSignals, stop_evt: threading.Event, transform: FrameTransformState):
        super().__init__(daemon=True)
        self._camera = camera
        self._queue = frame_queue
        self._signals = signals
        self._stop_evt = stop_evt
        self._pause_evt = threading.Event()
        self._transform = transform

    def pause(self) -> None:
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if self._pause_evt.is_set():
                time.sleep(0.01)
                continue
            try:
                frame = self._camera.get_frame()
                oriented = _apply_frame_transform(_normalize_frame(frame), self._transform)
                self._queue.put(oriented)
                latest, _ = self._queue.get_latest()
                if latest is not None:
                    self._signals.frame_ready.emit(latest)
            except Exception as exc:  # pragma: no cover
                self._signals.fault.emit(f"Camera failure: {exc}")
                time.sleep(0.05)


def _normalize_frame(frame: CameraFrame) -> CameraFrame:
    """Ensure camera frames are 2D and detached before cross-thread use."""
    try:
        import numpy as np

        arr = np.asarray(frame.image)
        if arr.ndim == 3 and 1 in arr.shape:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")
        # Copy to detach from camera-owned buffers that may be reused.
        return CameraFrame(image=arr.copy(), timestamp_s=frame.timestamp_s)
    except ImportError:
        image = frame.image
        if hasattr(image, "tolist") and callable(image.tolist):
            image = image.tolist()
        if isinstance(image, tuple):
            image = list(image)
        if not isinstance(image, list) or not image or not isinstance(image[0], (list, tuple)):
            raise ValueError("Camera frame must be 2D")
        return CameraFrame(image=[list(row) for row in image], timestamp_s=frame.timestamp_s)


class AutofocusWorkerObject(QtCore.QObject):
    def __init__(
        self,
        controller: AstigmaticAutofocusController,
        frame_queue: LatestFrameQueue,
        signals: AutofocusSignals,
        stats: RunStats,
        stop_evt: threading.Event,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._frame_queue = frame_queue
        self._signals = signals
        self._stats = stats
        self._stop_evt = stop_evt
        self._running = False
        self._last_seq = -1
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step)

    @Slot(tuple)
    def update_roi(self, roi_bounds: tuple[int, int, int, int]) -> None:
        self._controller.update_roi(Roi(*roi_bounds))

    @Slot(dict)
    def update_config(self, values: dict) -> None:
        self._controller.update_config(**values)

    @Slot()
    def reset_lock_state(self) -> None:
        self._controller.reset_lock_state()

    @Slot()
    def start_loop(self) -> None:
        # Always re-acquire lock from the current frame when starting so
        # control corrections are relative to the present target state.
        ok, score, issues = self._controller.check_calibration_ready()
        if not ok:
            self._signals.fault.emit(
                "Calibration rejected (score="
                f"{score:0.2f}): " + "; ".join(issues)
            )
            return
        self._controller.reset_lock_state()
        self._last_seq = -1
        self._running = True
        self._timer.start(max(1, int(1000.0 / self._controller.loop_hz)))

    @Slot()
    def stop_loop(self) -> None:
        self._running = False
        self._timer.stop()

    def _step(self) -> None:
        if not self._running or self._stop_evt.is_set():
            return
        frame, seq = self._frame_queue.get_latest()
        if frame is None:
            return
        # No new camera frame since last control step: skip this timer tick
        # instead of feeding duplicate frames into the controller.
        if self._last_seq >= 0 and seq == self._last_seq:
            return
        self._stats.total_frames += 1
        if self._last_seq >= 0 and seq - self._last_seq > 1:
            self._stats.dropped_frames += (seq - self._last_seq - 1)
        self._last_seq = seq
        try:
            sample = self._controller.run_step(frame=frame)
            self._stats.loop_latency_ms.append(sample.loop_latency_ms)
            self._signals.autofocus_update.emit(sample)
            cfg = self._controller.get_config_snapshot()
            r = cfg.roi
            self._signals.roi_position_update.emit((r.x, r.y, r.width, r.height))
            self._signals.state_changed.emit(sample.state.value)
        except Exception as exc:  # pragma: no cover
            msg = f"Autofocus failure: {exc}"
            if self._stats.faults is not None:
                self._stats.faults.append(msg)
            self._signals.fault.emit(msg)


class AutofocusMainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        camera: CameraInterface,
        stage: StageInterface,
        calibration: CalibrationLike,
        default_config: AutofocusConfig,
        calibration_output_path: str | None = None,
        calibration_half_range_um: float = 0.75,
        calibration_steps: int = 21,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Autofocus Instrument Panel")
        self._camera = camera
        self._stage = stage
        self._calibration = calibration
        self._config = default_config
        self._calibration_output_path = calibration_output_path or "calibration_sweep.csv"
        self._calibration_half_range_um = max(0.05, float(calibration_half_range_um))
        self._calibration_steps = max(5, int(calibration_steps))

        self._signals = AutofocusSignals()
        self._stop_evt = threading.Event()
        self._frame_queue = LatestFrameQueue()
        self._stats = RunStats(loop_latency_ms=deque(maxlen=1000), faults=[])
        self._history_t = deque(maxlen=400)
        self._history_z = deque(maxlen=400)
        self._history_err = deque(maxlen=400)
        self._history_corr = deque(maxlen=400)
        self._history_ell = deque(maxlen=400)
        self._history_state = deque(maxlen=400)
        self._last_cmd = None
        self._image_levels: tuple[float, float] | None = None
        self._display_domain: tuple[float, float] = (0.0, 1.0)
        self._display_level_min: float = 0.0
        self._display_level_max: float = 1.0
        self._display_autoscale: bool = True
        self._gamma: float = 1.0
        self._suspend_level_sync: bool = False
        self._frame_transform = FrameTransformState()
        self._calibration_lock = threading.Lock()
        self._calibration_running = False
        self._last_state: str | None = None
        self._last_recovery_log_s: float | None = None
        self._retry_prompt_lock = threading.Lock()
        self._retry_prompt_event: threading.Event | None = None
        self._retry_prompt_result: bool = False

        self._controller = AstigmaticAutofocusController(
            camera=self._camera,
            stage=self._stage,
            config=self._config,
            calibration=self._calibration,
        )

        self._af_thread = QtCore.QThread(self)
        self._af_worker = AutofocusWorkerObject(self._controller, self._frame_queue, self._signals, self._stats, self._stop_evt)
        self._af_worker.moveToThread(self._af_thread)
        self._signals.roi_changed.connect(self._af_worker.update_roi, QtCore.Qt.QueuedConnection)
        self._signals.config_changed.connect(self._af_worker.update_config, QtCore.Qt.QueuedConnection)
        self._af_thread.start()

        self._camera_worker = CameraWorker(self._camera, self._frame_queue, self._signals, self._stop_evt, self._frame_transform)

        self._build_ui()
        self._connect_signals()
        self._load_settings()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {UI_THEME['bg']};
                color: {UI_THEME['text']};
                font-family: "Avenir Next", "Segoe UI", sans-serif;
                font-size: 12px;
            }}
            QFrame#Card {{
                background: {UI_THEME['panel']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 14px;
            }}
            QFrame#CardAlt {{
                background: {UI_THEME['panel_alt']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 14px;
            }}
            QLabel#CardTitle {{
                font-size: 15px;
                font-weight: 700;
                color: {UI_THEME['text']};
            }}
            QLabel#CardSubtitle {{
                font-size: 11px;
                color: {UI_THEME['muted']};
            }}
            QLabel#MetricLabel {{
                color: {UI_THEME['muted']};
                font-size: 11px;
                letter-spacing: 0.6px;
            }}
            QLabel#MetricValue {{
                color: {UI_THEME['text']};
                font-size: 24px;
                font-weight: 700;
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
            }}
            QLabel#MetricMeta {{
                color: {UI_THEME['muted']};
                font-size: 11px;
            }}
            QLabel#StateTitle {{
                font-size: 12px;
                color: {UI_THEME['muted']};
                letter-spacing: 0.8px;
            }}
            QLabel#StateValue {{
                font-size: 30px;
                font-weight: 800;
                color: {UI_THEME['text']};
            }}
            QLabel#StateDetail {{
                font-size: 12px;
                color: {UI_THEME['text']};
            }}
            QPushButton {{
                background: {UI_THEME['panel_alt']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 10px;
                padding: 8px 12px;
                color: {UI_THEME['text']};
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {UI_THEME['focus']};
            }}
            QPushButton#PrimaryButton {{
                background: {UI_THEME['focus']};
                color: {UI_THEME['bg']};
                border-color: {UI_THEME['focus']};
                font-size: 14px;
                font-weight: 800;
                min-height: 34px;
            }}
            QPushButton#DangerButton {{
                background: {UI_THEME['fault']};
                color: white;
                border-color: {UI_THEME['fault']};
                font-size: 14px;
                font-weight: 800;
                min-height: 34px;
            }}
            QPushButton#WarnButton {{
                background: {UI_THEME['warn']};
                color: {UI_THEME['bg']};
                border-color: {UI_THEME['warn']};
                font-weight: 700;
            }}
            QCheckBox, QLabel, QSpinBox, QDoubleSpinBox, QComboBox {{
                color: {UI_THEME['text']};
            }}
            QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTextEdit {{
                background: {UI_THEME['bg_alt']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 8px;
                padding: 5px 8px;
            }}
            QToolBox::tab {{
                background: {UI_THEME['panel_alt']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 8px;
                color: {UI_THEME['text']};
                padding: 8px 12px;
                font-weight: 700;
            }}
            QToolBox::tab:selected {{
                background: {UI_THEME['focus']};
                color: {UI_THEME['bg']};
            }}
            QProgressBar {{
                background: {UI_THEME['bg_alt']};
                border: 1px solid {UI_THEME['panel_edge']};
                border-radius: 8px;
                text-align: center;
                color: {UI_THEME['text']};
            }}
            QProgressBar::chunk {{
                background: {UI_THEME['focus']};
                border-radius: 7px;
            }}
            QListWidget {{
                outline: none;
            }}
            """
        )

    def _make_card(self, title: str, subtitle: str = "", *, alt: bool = False) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        frame = QtWidgets.QFrame()
        frame.setObjectName("CardAlt" if alt else "Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        head = QtWidgets.QLabel(title)
        head.setObjectName("CardTitle")
        layout.addWidget(head)
        if subtitle:
            sub = QtWidgets.QLabel(subtitle)
            sub.setObjectName("CardSubtitle")
            sub.setWordWrap(True)
            layout.addWidget(sub)
        return frame, layout

    def _make_metric_tile(self, label: str, value: str = "--", meta: str = "") -> tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel]:
        frame = QtWidgets.QFrame()
        frame.setObjectName("CardAlt")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        lbl = QtWidgets.QLabel(label.upper())
        lbl.setObjectName("MetricLabel")
        val = QtWidgets.QLabel(value)
        val.setObjectName("MetricValue")
        meta_lbl = QtWidgets.QLabel(meta)
        meta_lbl.setObjectName("MetricMeta")
        meta_lbl.setWordWrap(True)
        layout.addWidget(lbl)
        layout.addWidget(val)
        layout.addWidget(meta_lbl)
        return frame, val, meta_lbl

    def _style_plot(self, plot: pg.PlotWidget, *, x_label: str = "", y_label: str = "") -> None:
        plot.setBackground(UI_THEME["bg_alt"])
        item = plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.16)
        item.getAxis("left").setPen(pg.mkPen(UI_THEME["dim"]))
        item.getAxis("bottom").setPen(pg.mkPen(UI_THEME["dim"]))
        item.getAxis("left").setTextPen(pg.mkPen(UI_THEME["muted"]))
        item.getAxis("bottom").setTextPen(pg.mkPen(UI_THEME["muted"]))
        if x_label:
            item.setLabel("bottom", x_label, color=UI_THEME["muted"])
        if y_label:
            item.setLabel("left", y_label, color=UI_THEME["muted"])
        item.layout.setContentsMargins(12, 12, 12, 12)

    def _add_plot_legend(self, plot: pg.PlotWidget) -> Any:
        item = plot.getPlotItem()
        try:
            legend = item.addLegend(
                offset=(10, 10),
                brush=pg.mkBrush(22, 24, 28, 220),
                pen=pg.mkPen(UI_THEME["panel_edge"]),
                labelTextColor=UI_THEME["text"],
            )
        except TypeError:
            legend = item.addLegend(offset=(10, 10))
            for method_name, arg in [
                ("setBrush", pg.mkBrush(22, 24, 28, 220)),
                ("setPen", pg.mkPen(UI_THEME["panel_edge"])),
            ]:
                method = getattr(legend, method_name, None)
                if callable(method):
                    try:
                        method(arg)
                    except Exception:
                        pass
        return legend

    def _push_event(self, message: str, *, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime())
        text = f"{stamp}  {message}"
        if text == getattr(self, "_last_event_text", None):
            return
        self._last_event_text = text
        if not hasattr(self, "_events"):
            return
        item = QtWidgets.QListWidgetItem(text)
        color = {
            "info": UI_THEME["muted"],
            "state": UI_THEME["text"],
            "warn": UI_THEME["warn"],
            "fault": UI_THEME["fault"],
        }.get(level, UI_THEME["muted"])
        item.setForeground(QtGui.QColor(color))
        self._events.insertItem(0, item)
        while self._events.count() > 80:
            self._events.takeItem(self._events.count() - 1)

    def _set_state_card(self, state: str, detail: str = "") -> None:
        palette = {
            AutofocusState.IDLE.value: ("#39424d", UI_THEME["muted"]),
            AutofocusState.CALIBRATED_READY.value: ("#1d4366", UI_THEME["cyan"]),
            AutofocusState.LOCKING.value: ("#244a45", UI_THEME["focus_bright"]),
            AutofocusState.LOCKED.value: ("#1e4d36", UI_THEME["ok"]),
            AutofocusState.DEGRADED.value: ("#5b4a20", UI_THEME["warn"]),
            AutofocusState.RECOVERY.value: ("#6a3f14", UI_THEME["warn"]),
            AutofocusState.LOST.value: ("#4d2d5d", "#d5a6ff"),
            AutofocusState.FAULT.value: ("#5e2827", UI_THEME["fault"]),
        }
        bg, accent = palette.get(state, (UI_THEME["panel"], UI_THEME["text"]))
        self._state_card.setStyleSheet(
            f"QFrame#{self._state_card.objectName()} {{"
            f"background:{bg}; border:1px solid {accent}; border-radius:18px; }}"
        )
        self._state_badge.setText(state)
        self._state_badge.setStyleSheet(
            f"color:{accent}; font-size:30px; font-weight:800;"
        )
        self._state_detail.setText(detail or "System ready for lock.")


    def _build_ui(self) -> None:
        self._apply_theme()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        master = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        master.setChildrenCollapsible(False)
        layout.addWidget(master)

        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.setChildrenCollapsible(False)
        master.addWidget(top)

        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom.setChildrenCollapsible(False)
        master.addWidget(bottom)
        master.setStretchFactor(0, 6)
        master.setStretchFactor(1, 4)

        left_card, left_layout = self._make_card(
            "Live Image",
            "Track a single bright spot. The ROI may move in XY, but only the Z stage is driven.",
        )
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setBackground(UI_THEME["bg_alt"])
        self._view = self._graphics.addViewBox(lockAspect=True)
        self._img = pg.ImageItem()
        self._view.addItem(self._img)
        self._view.invertY(True)
        self._roi = pg.RectROI(
            [self._config.roi.x, self._config.roi.y],
            [self._config.roi.width, self._config.roi.height],
            pen=pg.mkPen(UI_THEME["focus"], width=2),
        )
        self._view.addItem(self._roi)
        self._centroid_marker = pg.ScatterPlotItem(size=11, pen=pg.mkPen(UI_THEME["focus_bright"], width=2), brush=pg.mkBrush(255, 255, 255, 40))
        self._view.addItem(self._centroid_marker)
        self._trail_curve = pg.PlotCurveItem(pen=pg.mkPen((110, 197, 255, 110), width=2))
        self._view.addItem(self._trail_curve)
        self._overlay_info = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(22, 24, 28, 220), border=pg.mkPen(UI_THEME["panel_edge"]))
        self._overlay_legend = pg.TextItem(anchor=(1, 0), fill=pg.mkBrush(22, 24, 28, 210), border=pg.mkPen(UI_THEME["panel_edge"]))
        self._view.addItem(self._overlay_info)
        self._view.addItem(self._overlay_legend)
        self._hist = pg.HistogramLUTWidget()
        self._hist.setImageItem(self._img)
        image_row = QtWidgets.QHBoxLayout()
        image_row.setSpacing(10)
        image_row.addWidget(self._graphics, 7)
        image_row.addWidget(self._hist, 2)
        left_layout.addLayout(image_row)

        image_toolbar = QtWidgets.QHBoxLayout()
        self._show_trail = QtWidgets.QCheckBox("Show track trail")
        self._show_trail.setChecked(True)
        self._show_centroid = QtWidgets.QCheckBox("Show centroid")
        self._show_centroid.setChecked(True)
        image_toolbar.addWidget(self._show_trail)
        image_toolbar.addWidget(self._show_centroid)
        image_toolbar.addStretch(1)
        left_layout.addLayout(image_toolbar)
        top.addWidget(left_card)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_body = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_body)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self._state_card = QtWidgets.QFrame()
        self._state_card.setObjectName("StateCard")
        state_layout = QtWidgets.QVBoxLayout(self._state_card)
        state_layout.setContentsMargins(16, 16, 16, 16)
        state_layout.setSpacing(8)
        state_title = QtWidgets.QLabel("AUTOFOCUS STATE")
        state_title.setObjectName("StateTitle")
        self._state_badge = QtWidgets.QLabel("CALIBRATED_READY")
        self._state_badge.setObjectName("StateValue")
        self._state_detail = QtWidgets.QLabel("Calibration loaded. Ready to acquire lock.")
        self._state_detail.setObjectName("StateDetail")
        self._state_detail.setWordWrap(True)
        self._status = QtWidgets.QLabel("Ready")
        self._status.setObjectName("CardSubtitle")
        self._status.setWordWrap(True)
        self._lock_quality = QtWidgets.QProgressBar()
        self._lock_quality.setRange(0, 100)
        self._lock_quality.setValue(0)
        self._lock_quality_lbl = QtWidgets.QLabel("Lock quality 0%")
        self._lock_quality_lbl.setObjectName("CardSubtitle")
        state_layout.addWidget(state_title)
        state_layout.addWidget(self._state_badge)
        state_layout.addWidget(self._state_detail)
        state_layout.addWidget(self._lock_quality)
        state_layout.addWidget(self._lock_quality_lbl)
        state_layout.addWidget(self._status)
        right_layout.addWidget(self._state_card)

        metrics_wrap, metrics_layout = self._make_card("Live Metrics", "Critical values stay visible while tuning.")
        metric_grid = QtWidgets.QGridLayout()
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)
        z_tile, self._z_val, self._z_meta = self._make_metric_tile("Current Z", "--", "stage command")
        err_tile, self._err_val, self._err_meta = self._make_metric_tile("Error", "--", "focus error")
        corr_tile, self._corr_val, self._corr_meta = self._make_metric_tile("Correction", "--", "last command step")
        conf_tile, self._conf_val, self._conf_meta = self._make_metric_tile("Confidence", "--", "detection pass")
        domain_tile, self._domain_val, self._domain_meta = self._make_metric_tile("Domain Margin", "--", "negative means outside")
        lag_tile, self._lag_val, self._lag_meta = self._make_metric_tile("Stage / Frames", "--", "lag and dropped")
        metric_grid.addWidget(z_tile, 0, 0)
        metric_grid.addWidget(err_tile, 0, 1)
        metric_grid.addWidget(corr_tile, 1, 0)
        metric_grid.addWidget(conf_tile, 1, 1)
        metric_grid.addWidget(domain_tile, 2, 0)
        metric_grid.addWidget(lag_tile, 2, 1)
        metrics_layout.addLayout(metric_grid)
        right_layout.addWidget(metrics_wrap)

        self._sections = QtWidgets.QToolBox()
        right_layout.addWidget(self._sections)

        af_page = QtWidgets.QWidget()
        af_layout = QtWidgets.QVBoxLayout(af_page)
        af_layout.setContentsMargins(10, 10, 10, 10)
        af_layout.setSpacing(10)
        action_row = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("Start Autofocus")
        self._start_btn.setObjectName("PrimaryButton")
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setObjectName("DangerButton")
        action_row.addWidget(self._start_btn)
        action_row.addWidget(self._stop_btn)
        af_layout.addLayout(action_row)
        self._lock_setpoint = QtWidgets.QCheckBox("Lock current focus setpoint on engage")
        self._lock_setpoint.setChecked(True)
        af_layout.addWidget(self._lock_setpoint)
        self._kp = QtWidgets.QDoubleSpinBox(); self._kp.setRange(0.0, 20.0); self._kp.setDecimals(3); self._kp.setValue(self._config.kp); self._kp.setPrefix("Kp  ")
        self._ki = QtWidgets.QDoubleSpinBox(); self._ki.setRange(0.0, 20.0); self._ki.setDecimals(3); self._ki.setValue(self._config.ki); self._ki.setPrefix("Ki  ")
        self._kd = QtWidgets.QDoubleSpinBox(); self._kd.setRange(0.0, 20.0); self._kd.setDecimals(3); self._kd.setValue(self._config.kd); self._kd.setPrefix("Kd  ")
        self._max_step = QtWidgets.QDoubleSpinBox(); self._max_step.setRange(0.001, 20.0); self._max_step.setDecimals(3); self._max_step.setValue(self._config.max_step_um); self._max_step.setPrefix("Max step (um)  ")
        self._deadband = QtWidgets.QDoubleSpinBox(); self._deadband.setRange(0.0, 1.0); self._deadband.setDecimals(4); self._deadband.setValue(self._config.command_deadband_um); self._deadband.setPrefix("Deadband (um)  ")
        self._max_speed = QtWidgets.QDoubleSpinBox(); self._max_speed.setRange(0.0, 500.0); self._max_speed.setDecimals(3); self._max_speed.setValue(self._config.max_slew_rate_um_per_s or 0.0); self._max_speed.setPrefix("Max slew (um/s)  ")
        for w in [self._kp, self._ki, self._kd, self._max_step, self._deadband, self._max_speed]:
            af_layout.addWidget(w)
        af_layout.addStretch(1)
        self._sections.addItem(af_page, "Autofocus")

        cal_page = QtWidgets.QWidget()
        cal_layout = QtWidgets.QVBoxLayout(cal_page)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        cal_layout.setSpacing(10)
        self._cal_btn = QtWidgets.QPushButton("Calibrate")
        self._cal_btn.setObjectName("PrimaryButton")
        self._cal_progress = QtWidgets.QLabel("Calibration idle")
        self._cal_progress.setObjectName("CardSubtitle")
        self._cal_half_range = QtWidgets.QDoubleSpinBox(); self._cal_half_range.setRange(0.05, 100.0); self._cal_half_range.setDecimals(3); self._cal_half_range.setValue(self._calibration_half_range_um); self._cal_half_range.setPrefix("Half-range (um)  ")
        self._cal_steps = QtWidgets.QSpinBox(); self._cal_steps.setRange(5, 1001); self._cal_steps.setSingleStep(2); self._cal_steps.setValue(self._calibration_steps); self._cal_steps.setPrefix("Steps  ")
        cal_layout.addWidget(self._cal_btn)
        cal_layout.addWidget(self._cal_progress)
        cal_layout.addWidget(self._cal_half_range)
        cal_layout.addWidget(self._cal_steps)
        cal_layout.addStretch(1)
        self._sections.addItem(cal_page, "Calibration")

        track_page = QtWidgets.QWidget()
        track_layout = QtWidgets.QVBoxLayout(track_page)
        track_layout.setContentsMargins(10, 10, 10, 10)
        track_layout.setSpacing(10)
        self._track_roi_xy = QtWidgets.QCheckBox("Move ROI to follow the spot in XY")
        self._track_roi_xy.setChecked(bool(self._config.track_roi))
        self._track_gain = QtWidgets.QDoubleSpinBox(); self._track_gain.setRange(0.0, 1.0); self._track_gain.setSingleStep(0.05); self._track_gain.setValue(float(self._config.track_gain)); self._track_gain.setPrefix("Track gain  ")
        self._track_deadband_px = QtWidgets.QDoubleSpinBox(); self._track_deadband_px.setRange(0.0, 50.0); self._track_deadband_px.setDecimals(2); self._track_deadband_px.setValue(float(self._config.track_deadband_px)); self._track_deadband_px.setPrefix("Deadband (px)  ")
        self._track_max_step_px = QtWidgets.QDoubleSpinBox(); self._track_max_step_px.setRange(0.5, 100.0); self._track_max_step_px.setDecimals(2); self._track_max_step_px.setValue(float(self._config.track_max_shift_px)); self._track_max_step_px.setPrefix("Max shift (px/frame)  ")
        trail_note = QtWidgets.QLabel("Display trail and centroid toggles live above the image.")
        trail_note.setObjectName("CardSubtitle")
        trail_note.setWordWrap(True)
        for w in [self._track_roi_xy, self._track_gain, self._track_deadband_px, self._track_max_step_px, trail_note]:
            track_layout.addWidget(w)
        track_layout.addStretch(1)
        self._sections.addItem(track_page, "ROI Tracking")

        safety_page = QtWidgets.QWidget()
        safety_layout = QtWidgets.QVBoxLayout(safety_page)
        safety_layout.setContentsMargins(10, 10, 10, 10)
        safety_layout.setSpacing(10)
        self._home_btn = QtWidgets.QPushButton("Home Stage To Centre")
        self._home_btn.setObjectName("WarnButton")
        self._home_btn.setToolTip(
            "Move stage to the midpoint of its travel range.\n"
            "Use after a power cycle or re-home to restore sweep headroom.\n"
            "Recalibrate only if the sample itself has moved."
        )
        safety_note = QtWidgets.QLabel("Use with care. This recentres stage travel but does not move the ROI or retune the calibration.")
        safety_note.setObjectName("CardSubtitle")
        safety_note.setWordWrap(True)
        safety_layout.addWidget(self._home_btn)
        safety_layout.addWidget(safety_note)
        safety_layout.addStretch(1)
        self._sections.addItem(safety_page, "Safety")

        detect_page = QtWidgets.QWidget()
        detect_layout = QtWidgets.QVBoxLayout(detect_page)
        detect_layout.setContentsMargins(10, 10, 10, 10)
        detect_layout.setSpacing(10)
        self._detection_window = QtWidgets.QSpinBox(); self._detection_window.setRange(3, 30); self._detection_window.setValue(int(self._config.detection_window_size)); self._detection_window.setPrefix("Window  ")
        self._detection_threshold = QtWidgets.QDoubleSpinBox(); self._detection_threshold.setRange(0.1, 1.0); self._detection_threshold.setSingleStep(0.05); self._detection_threshold.setValue(float(self._config.detection_pass_threshold)); self._detection_threshold.setPrefix("Pass threshold  ")
        self._use_gaussian = QtWidgets.QCheckBox("Use per-frame Gaussian fit")
        self._use_gaussian.setChecked(bool(self._config.use_gaussian_fit))
        for w in [self._detection_window, self._detection_threshold, self._use_gaussian]:
            detect_layout.addWidget(w)
        detect_layout.addStretch(1)
        self._sections.addItem(detect_page, "Detection")

        display_page = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_page)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setSpacing(10)
        self._autoscale_btn = QtWidgets.QPushButton("Autoscale image")
        self._rotation = QtWidgets.QComboBox()
        self._rotation.addItems(["Rotate 0 deg", "Rotate 90 deg", "Rotate 180 deg", "Rotate 270 deg"])
        self._flip_h = QtWidgets.QCheckBox("Flip horizontal")
        self._flip_v = QtWidgets.QCheckBox("Flip vertical")
        self._level_min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self._level_min_slider.setRange(0, 1000); self._level_min_slider.setValue(0)
        self._level_max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self._level_max_slider.setRange(0, 1000); self._level_max_slider.setValue(1000)
        self._gamma_spin = QtWidgets.QDoubleSpinBox(); self._gamma_spin.setRange(0.2, 4.0); self._gamma_spin.setSingleStep(0.05); self._gamma_spin.setValue(1.0); self._gamma_spin.setPrefix("Gamma  ")
        display_layout.addWidget(self._autoscale_btn)
        display_layout.addWidget(QtWidgets.QLabel("Display minimum"))
        display_layout.addWidget(self._level_min_slider)
        display_layout.addWidget(QtWidgets.QLabel("Display maximum"))
        display_layout.addWidget(self._level_max_slider)
        display_layout.addWidget(self._gamma_spin)
        display_layout.addWidget(self._rotation)
        display_layout.addWidget(self._flip_h)
        display_layout.addWidget(self._flip_v)
        display_layout.addStretch(1)
        self._sections.addItem(display_page, "Display")

        right_layout.addStretch(1)
        right_scroll.setWidget(right_body)
        top.addWidget(right_scroll)
        top.setStretchFactor(0, 8)
        top.setStretchFactor(1, 5)

        diag_card, diag_layout = self._make_card("Diagnostics", "Live control behavior over the most recent 10 seconds.")
        self._plot = pg.PlotWidget(title="Lock Diagnostics")
        self._style_plot(self._plot, x_label="Time (s)", y_label="Signal")
        self._add_plot_legend(self._plot)
        self._diag_zero = pg.InfiniteLine(angle=0, pen=pg.mkPen(UI_THEME["dim"], style=QtCore.Qt.DashLine))
        self._plot.addItem(self._diag_zero)
        self._diag_safe_band = pg.LinearRegionItem(values=[-0.01, 0.01], orientation='horizontal', movable=False, brush=pg.mkBrush(78, 205, 196, 36), pen=pg.mkPen(None))
        self._plot.addItem(self._diag_safe_band)
        self._z_curve = self._plot.plot(pen=pg.mkPen(UI_THEME["yellow"], width=2), name='Z command')
        self._e_curve = self._plot.plot(pen=pg.mkPen(UI_THEME["cyan"], width=2), name='Error (um)')
        self._c_curve = self._plot.plot(pen=pg.mkPen(UI_THEME["magenta"], width=2), name='Correction')
        self._ell_curve = self._plot.plot(pen=pg.mkPen(UI_THEME["focus"], width=2), name='Ellipticity')
        diag_layout.addWidget(self._plot)
        bottom.addWidget(diag_card)

        cal_card, cal_layout = self._make_card("Calibration", "Accepted samples and model fit from the most recent sweep.")
        summary_row = QtWidgets.QGridLayout()
        summary_row.setHorizontalSpacing(10)
        summary_row.setVerticalSpacing(8)
        q_tile, self._cal_quality_val, self._cal_quality_meta = self._make_metric_tile("Fit Quality", "--", "awaiting sweep")
        range_tile, self._cal_range_val, self._cal_range_meta = self._make_metric_tile("Usable Range", "--", "focus-relative")
        fall_tile, self._cal_fallback_val, self._cal_fallback_meta = self._make_metric_tile("Fallback Fraction", "--", "rejected from model")
        accept_tile, self._cal_accept_val, self._cal_accept_meta = self._make_metric_tile("Calibration", "--", "status")
        summary_row.addWidget(q_tile, 0, 0)
        summary_row.addWidget(range_tile, 0, 1)
        summary_row.addWidget(fall_tile, 1, 0)
        summary_row.addWidget(accept_tile, 1, 1)
        cal_layout.addLayout(summary_row)

        self._cal_plot = pg.PlotWidget(title="Calibration Curve")
        self._style_plot(self._cal_plot, x_label="Error signal", y_label="Z offset (um)")
        self._add_plot_legend(self._cal_plot)
        self._cal_pts = self._cal_plot.plot(pen=None, symbol='o', symbolSize=7, symbolBrush=(255, 211, 105, 220), symbolPen=pg.mkPen(UI_THEME["yellow"]), name='Forward sweep')
        self._cal_pts_rev = self._cal_plot.plot(pen=None, symbol='t', symbolSize=7, symbolBrush=(110, 197, 255, 210), symbolPen=pg.mkPen(UI_THEME["cyan"]), name='Reverse sweep')
        self._cal_fit = self._cal_plot.plot(pen=pg.mkPen(UI_THEME["ok"], width=3), name='Model fit')
        self._cal_current_z = pg.InfiniteLine(angle=0, pen=pg.mkPen(UI_THEME["fault"], width=1, style=QtCore.Qt.DashLine))
        self._cal_plot.addItem(self._cal_current_z)
        self._cal_usable_band = pg.LinearRegionItem(values=[-0.1, 0.1], orientation='horizontal', movable=False, brush=pg.mkBrush(107, 207, 138, 36), pen=pg.mkPen(None))
        self._cal_plot.addItem(self._cal_usable_band)
        cal_layout.addWidget(self._cal_plot, 2)

        self._ell_plot = pg.PlotWidget(title="Ellipticity Classes")
        self._style_plot(self._ell_plot, x_label="Z offset (um)", y_label="Ellipticity")
        self._add_plot_legend(self._ell_plot)
        self._ell_param = self._ell_plot.plot(
            pen=None,
            symbol='o',
            symbolSize=7,
            symbolBrush=(107, 207, 138, 210),
            symbolPen=pg.mkPen(UI_THEME["ok"]),
            name='Accepted for model',
        )
        self._ell_fallback = self._ell_plot.plot(
            pen=None,
            symbol='x',
            symbolSize=7,
            symbolBrush=(120, 128, 138, 120),
            symbolPen=pg.mkPen((120, 128, 138, 180)),
            name='Rejected / fallback',
        )
        self._ell_plot.addItem(pg.InfiniteLine(angle=0, pos=1.0, pen=pg.mkPen((255, 255, 255, 80), style=QtCore.Qt.DashLine)))
        self._ell_usable_band = pg.LinearRegionItem(values=[-0.1, 0.1], movable=False, brush=pg.mkBrush(107, 207, 138, 28), pen=pg.mkPen(None))
        self._ell_plot.addItem(self._ell_usable_band)
        cal_layout.addWidget(self._ell_plot, 2)
        bottom.addWidget(cal_card)

        event_card, event_layout = self._make_card("Event Feed", "Recent state transitions, faults, and operator-visible messages.")
        self._events = QtWidgets.QListWidget()
        self._events.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._events.setAlternatingRowColors(False)
        event_layout.addWidget(self._events)
        bottom.addWidget(event_card)
        bottom.setStretchFactor(0, 5)
        bottom.setStretchFactor(1, 6)
        bottom.setStretchFactor(2, 3)
        self._set_state_card(AutofocusState.CALIBRATED_READY.value, "Calibration loaded. Ready to acquire lock.")
        self._push_event("Instrument panel ready", level="state")

    def _connect_signals(self) -> None:
        self._signals.frame_ready.connect(self._on_frame)
        self._signals.autofocus_update.connect(self._on_update)
        self._signals.state_changed.connect(self._on_state)
        self._signals.fault.connect(self._on_fault)
        self._signals.status.connect(self._on_status)
        self._signals.calibration_running.connect(self._on_calibration_running)
        self._signals.calibration_plot.connect(self._on_calibration_plot)
        self._signals.calibration_retry_prompt.connect(self._on_calibration_retry_prompt)
        self._signals.roi_position_update.connect(self._on_roi_position_update)

        self._roi.sigRegionChangeFinished.connect(self._emit_roi_change)
        self._start_btn.clicked.connect(lambda: QtCore.QMetaObject.invokeMethod(self._af_worker, "start_loop", QtCore.Qt.QueuedConnection))
        self._stop_btn.clicked.connect(lambda: QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection))
        self._home_btn.clicked.connect(self._home_stage_to_centre)
        self._cal_btn.clicked.connect(self._run_calibration)
        self._lock_setpoint.toggled.connect(self._on_lock_setpoint)
        self._kp.valueChanged.connect(lambda v: self._queue_config_update(kp=float(v)))
        self._ki.valueChanged.connect(lambda v: self._queue_config_update(ki=float(v)))
        self._kd.valueChanged.connect(lambda v: self._queue_config_update(kd=float(v)))
        self._max_step.valueChanged.connect(lambda v: self._queue_config_update(max_step_um=float(v)))
        self._deadband.valueChanged.connect(lambda v: self._queue_config_update(command_deadband_um=float(v)))
        self._max_speed.valueChanged.connect(
            lambda v: self._queue_config_update(max_slew_rate_um_per_s=(None if v <= 0 else float(v)))
        )
        self._track_roi_xy.toggled.connect(lambda v: self._queue_config_update(track_roi=bool(v)))
        self._track_gain.valueChanged.connect(lambda v: self._queue_config_update(track_gain=float(v)))
        self._track_deadband_px.valueChanged.connect(lambda v: self._queue_config_update(track_deadband_px=float(v)))
        self._track_max_step_px.valueChanged.connect(lambda v: self._queue_config_update(track_max_shift_px=float(v)))
        self._detection_window.valueChanged.connect(lambda v: self._queue_config_update(detection_window_size=int(v)))
        self._detection_threshold.valueChanged.connect(lambda v: self._queue_config_update(detection_pass_threshold=float(v)))
        self._use_gaussian.toggled.connect(lambda v: self._queue_config_update(use_gaussian_fit=bool(v)))
        self._rotation.currentIndexChanged.connect(self._on_transform_changed)
        self._flip_h.toggled.connect(self._on_transform_changed)
        self._flip_v.toggled.connect(self._on_transform_changed)
        self._autoscale_btn.clicked.connect(self._on_autoscale_clicked)
        self._level_min_slider.valueChanged.connect(self._on_level_slider_changed)
        self._level_max_slider.valueChanged.connect(self._on_level_slider_changed)
        self._gamma_spin.valueChanged.connect(self._on_gamma_changed)
        for signal in [
            self._kp.valueChanged,
            self._ki.valueChanged,
            self._kd.valueChanged,
            self._max_step.valueChanged,
            self._deadband.valueChanged,
            self._max_speed.valueChanged,
            self._track_gain.valueChanged,
            self._track_deadband_px.valueChanged,
            self._track_max_step_px.valueChanged,
            self._detection_window.valueChanged,
            self._detection_threshold.valueChanged,
            self._rotation.currentIndexChanged,
            self._level_min_slider.valueChanged,
            self._level_max_slider.valueChanged,
            self._gamma_spin.valueChanged,
            self._cal_half_range.valueChanged,
            self._cal_steps.valueChanged,
        ]:
            signal.connect(lambda *_args: self._save_settings())
        for signal in [
            self._lock_setpoint.toggled,
            self._track_roi_xy.toggled,
            self._use_gaussian.toggled,
            self._flip_h.toggled,
            self._flip_v.toggled,
            self._show_trail.toggled,
            self._show_centroid.toggled,
        ]:
            signal.connect(lambda *_args: self._save_settings())

    def _on_transform_changed(self, *_args) -> None:
        rotation = int(self._rotation.currentIndex()) * 90
        self._frame_transform.set(
            rotation_deg=rotation,
            flip_h=self._flip_h.isChecked(),
            flip_v=self._flip_v.isChecked(),
        )
        self._image_levels = None

    def _queue_config_update(self, **values) -> None:
        self._signals.config_changed.emit(values)

    def _settings_path(self) -> Path:
        return Path(self._calibration_output_path).with_name("autofocus_gui_settings.json")

    def _collect_settings(self) -> dict[str, Any]:
        pos = self._roi.pos()
        size = self._roi.size()
        return {
            "kp": float(self._kp.value()),
            "ki": float(self._ki.value()),
            "kd": float(self._kd.value()),
            "max_step_um": float(self._max_step.value()),
            "command_deadband_um": float(self._deadband.value()),
            "max_slew_rate_um_per_s": float(self._max_speed.value()),
            "lock_setpoint": bool(self._lock_setpoint.isChecked()),
            "track_roi": bool(self._track_roi_xy.isChecked()),
            "track_gain": float(self._track_gain.value()),
            "track_deadband_px": float(self._track_deadband_px.value()),
            "track_max_shift_px": float(self._track_max_step_px.value()),
            "roi": [int(pos.x()), int(pos.y()), int(size.x()), int(size.y())],
            "rotation_index": int(self._rotation.currentIndex()),
            "flip_h": bool(self._flip_h.isChecked()),
            "flip_v": bool(self._flip_v.isChecked()),
            "show_trail": bool(self._show_trail.isChecked()),
            "show_centroid": bool(self._show_centroid.isChecked()),
            "cal_half_range_um": float(self._cal_half_range.value()),
            "cal_steps": int(self._cal_steps.value()),
        }

    def _save_settings(self) -> None:
        try:
            self._settings_path().write_text(json.dumps(self._collect_settings(), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_settings(self) -> None:
        path = self._settings_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        for widget, key in [
            (self._kp, "kp"),
            (self._ki, "ki"),
            (self._kd, "kd"),
            (self._max_step, "max_step_um"),
            (self._deadband, "command_deadband_um"),
            (self._max_speed, "max_slew_rate_um_per_s"),
            (self._cal_half_range, "cal_half_range_um"),
            (self._track_gain, "track_gain"),
            (self._track_deadband_px, "track_deadband_px"),
            (self._track_max_step_px, "track_max_shift_px"),
        ]:
            if key in data:
                try:
                    widget.setValue(float(data[key]))
                except Exception:
                    pass
        if "cal_steps" in data:
            try:
                self._cal_steps.setValue(int(data["cal_steps"]))
            except Exception:
                pass
        if "lock_setpoint" in data:
            self._lock_setpoint.setChecked(bool(data["lock_setpoint"]))
        if "track_roi" in data:
            self._track_roi_xy.setChecked(bool(data["track_roi"]))
        if "rotation_index" in data:
            self._rotation.setCurrentIndex(int(data["rotation_index"]))
        self._flip_h.setChecked(bool(data.get("flip_h", False)))
        self._flip_v.setChecked(bool(data.get("flip_v", False)))
        self._show_trail.setChecked(bool(data.get("show_trail", True)))
        self._show_centroid.setChecked(bool(data.get("show_centroid", True)))
        if "roi" in data and isinstance(data["roi"], list) and len(data["roi"]) == 4:
            x, y, w, h = [int(v) for v in data["roi"]]
            self._roi.setPos([x, y])
            self._roi.setSize([max(1, w), max(1, h)])
            self._emit_roi_change()
        self._on_transform_changed()

    def _emit_roi_change(self) -> None:
        pos = self._roi.pos()
        size = self._roi.size()
        roi = (max(0, int(pos.x())), max(0, int(pos.y())), max(1, int(size.x())), max(1, int(size.y())))
        self._signals.roi_changed.emit(roi)


    def _sync_level_sliders_from_levels(self) -> None:
        lo_dom, hi_dom = self._display_domain
        span = max(1e-9, hi_dom - lo_dom)
        self._suspend_level_sync = True
        self._level_min_slider.setValue(int(max(0, min(1000, round((self._display_level_min - lo_dom) / span * 1000.0)))))
        self._level_max_slider.setValue(int(max(0, min(1000, round((self._display_level_max - lo_dom) / span * 1000.0)))))
        self._suspend_level_sync = False

    def _sync_levels_from_sliders(self) -> None:
        lo_dom, hi_dom = self._display_domain
        span = max(1e-9, hi_dom - lo_dom)
        s_min = int(self._level_min_slider.value())
        s_max = int(self._level_max_slider.value())
        if s_min >= s_max:
            if self.sender() is self._level_min_slider:
                s_max = min(1000, s_min + 1)
                self._suspend_level_sync = True
                self._level_max_slider.setValue(s_max)
                self._suspend_level_sync = False
            else:
                s_min = max(0, s_max - 1)
                self._suspend_level_sync = True
                self._level_min_slider.setValue(s_min)
                self._suspend_level_sync = False
        self._display_level_min = lo_dom + (s_min / 1000.0) * span
        self._display_level_max = lo_dom + (s_max / 1000.0) * span

    def _on_autoscale_clicked(self) -> None:
        self._display_autoscale = True

    def _on_level_slider_changed(self, _value: int) -> None:
        if self._suspend_level_sync:
            return
        self._display_autoscale = False
        self._sync_levels_from_sliders()

    def _on_gamma_changed(self, value: float) -> None:
        self._gamma = max(0.2, float(value))

    def _apply_gamma_lut(self) -> None:
        try:
            import numpy as np

            gamma = max(0.2, float(self._gamma))
            x = np.linspace(0.0, 1.0, 256, dtype=float)
            y = np.power(x, 1.0 / gamma)
            lut = np.clip(np.round(y * 255.0), 0, 255).astype(np.ubyte)
            self._img.setLookupTable(lut)
        except Exception:
            pass

    @Slot(object)
    def _on_frame(self, frame: CameraFrame) -> None:
        try:
            import numpy as np

            arr = np.asarray(frame.image)
            if arr.ndim == 3 and 1 in arr.shape:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Camera frame must be 2D (received shape={arr.shape!r})")

            arr = np.asarray(arr, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
            else:
                lo, hi = 0.0, 1.0
            if hi <= lo:
                hi = lo + 1.0

            self._display_domain = (lo, hi)
            if self._display_autoscale:
                self._display_level_min = lo
                self._display_level_max = hi
                self._sync_level_sliders_from_levels()
            else:
                self._sync_levels_from_sliders()

            if self._display_level_max <= self._display_level_min:
                self._display_level_max = self._display_level_min + 1e-6

            self._apply_gamma_lut()
            self._img.setImage(
                arr,
                autoLevels=False,
                levels=(self._display_level_min, self._display_level_max),
            )
            h, w = arr.shape
            self._overlay_info.setPos(6, 6)
            self._overlay_legend.setPos(max(10, w - 6), 6)
            self._overlay_legend.setHtml(
                "<div style='color:{text}; font-size:11px;'>"
                "<span style='color:{focus};'>ROI</span>  "
                "<span style='color:{cyan};'>trail</span>  "
                "<span style='color:{ok};'>centroid</span>"
                "</div>".format(
                    text=UI_THEME["text"],
                    focus=UI_THEME["focus"],
                    cyan=UI_THEME["cyan"],
                    ok=UI_THEME["ok"],
                )
            )
        except Exception as exc:
            self._signals.fault.emit(f"Display failure: {exc}")

    @Slot(tuple)
    def _on_roi_position_update(self, bounds: tuple[int, int, int, int]) -> None:
        self._roi.blockSignals(True)
        self._roi.setPos([bounds[0], bounds[1]])
        self._roi.setSize([bounds[2], bounds[3]])
        self._roi.blockSignals(False)

    @Slot(object)
    def _on_update(self, sample) -> None:
        now = time.monotonic()
        self._history_t.append(now)
        self._history_z.append(sample.commanded_z_um)
        self._history_err.append(sample.error_um)
        corr = 0.0 if self._last_cmd is None else sample.commanded_z_um - self._last_cmd
        self._history_corr.append(corr)
        self._history_ell.append(getattr(sample, "ellipticity", 0.0))
        self._history_state.append(str(getattr(sample, "state", "")))
        self._last_cmd = sample.commanded_z_um

        conf = float(getattr(sample, "confidence_scale", 1.0))
        detect = float(getattr(sample, "detection_pass_rate", 1.0))
        lock_quality = int(max(0.0, min(100.0, 100.0 * conf * detect)))
        self._lock_quality.setValue(lock_quality)
        self._lock_quality_lbl.setText(f"Lock quality {lock_quality}%   pass {detect:0.0%}")

        self._z_val.setText(f"{sample.commanded_z_um:+.3f}")
        self._z_meta.setText(f"commanded  stage {sample.stage_z_um:+.3f} um")
        self._err_val.setText(f"{sample.error_um:+.4f}")
        self._err_meta.setText(f"raw {sample.error:+.4f}   ell {getattr(sample, 'ellipticity', 0.0):.3f}")
        self._corr_val.setText(f"{corr * 1000.0:+.1f} nm")
        self._corr_meta.setText(f"loop latency {sample.loop_latency_ms:.1f} ms")
        self._conf_val.setText(f"{detect:0.0%}")
        self._conf_meta.setText(f"R² {getattr(sample, 'fit_r_squared', 0.0):.2f}   scale {conf:0.2f}")
        self._domain_val.setText(f"{getattr(sample, 'domain_margin', 0.0):+0.4f}")
        self._domain_meta.setText("negative means outside calibration")
        self._lag_val.setText(f"{getattr(sample, 'stage_lag_um', 0.0):0.3f} um")
        self._lag_meta.setText(f"dropped {self._stats.dropped_frames} frames")
        self._status.setText(f"I={sample.roi_total_intensity:.0f} err={sample.error:+.4f} - {sample.diagnostic}")
        self._roi.blockSignals(True)
        self._roi.setPos((sample.roi.x, sample.roi.y), update=True)
        self._roi.setSize((sample.roi.width, sample.roi.height), update=True)
        self._roi.blockSignals(False)
        if sample.confidence_ok:
            self._roi.setPen(pg.mkPen(UI_THEME["focus"], width=2))
        else:
            self._roi.setPen(pg.mkPen(UI_THEME["fault"], width=2))

        cx = sample.roi.x + (sample.roi.width / 2.0)
        cy = sample.roi.y + (sample.roi.height / 2.0)
        if self._show_centroid.isChecked():
            self._centroid_marker.setData([cx], [cy])
        else:
            self._centroid_marker.setData([], [])

        if not hasattr(self, "_trail_points"):
            self._trail_points = deque(maxlen=80)
        self._trail_points.append((cx, cy))
        if self._show_trail.isChecked():
            xs = [p[0] for p in self._trail_points]
            ys = [p[1] for p in self._trail_points]
            self._trail_curve.setData(xs, ys)
        else:
            self._trail_curve.setData([], [])

        self._overlay_info.setHtml(
            "<div style='color:{text}; font-size:12px;'>"
            "<b>Live fit</b><br>"
            "I {intensity:.0f}<br>"
            "ell {ell:.3f}<br>"
            "fwhm {fwhm:.2f}px<br>"
            "R² {r2:.2f}"
            "</div>".format(
                text=UI_THEME["text"],
                intensity=sample.roi_total_intensity,
                ell=getattr(sample, "ellipticity", 0.0),
                fwhm=getattr(sample, "fwhm_px", 0.0),
                r2=getattr(sample, "fit_r_squared", 0.0),
            )
        )
        self._update_plot()

    def _update_plot(self) -> None:
        if not self._history_t:
            return
        t0 = self._history_t[-1]
        xs = [t - t0 for t in self._history_t]
        self._z_curve.setData(xs, list(self._history_z))
        self._e_curve.setData(xs, list(self._history_err))
        self._c_curve.setData(xs, list(self._history_corr))
        self._ell_curve.setData(xs, list(self._history_ell))
        self._plot.setXRange(-10.0, 0.0, padding=0.02)
        deadband = max(0.001, float(self._deadband.value()))
        self._diag_safe_band.setRegion((-deadband, deadband))


    def _event_log_path(self) -> Path:
        return Path(self._calibration_output_path).with_name("autofocus_events.log")

    def _append_event_log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            with self._event_log_path().open("a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {message}\n")
        except Exception:
            pass

    @Slot(str)
    def _on_state(self, state: str) -> None:
        detail = {
            AutofocusState.IDLE.value: "Loop stopped. Image and ROI remain live.",
            AutofocusState.CALIBRATED_READY.value: "Calibration accepted. Lock can be armed safely.",
            AutofocusState.LOCKING.value: "Good frames are accumulating and control authority is ramping in.",
            AutofocusState.LOCKED.value: "Closed-loop corrections are stable and within the deadband.",
            AutofocusState.DEGRADED.value: "Measurements are still arriving, but trust is reduced.",
            AutofocusState.RECOVERY.value: "Guardrails are holding control while the target is re-evaluated.",
            AutofocusState.LOST.value: "Target confidence collapsed. Re-select the ROI.",
            AutofocusState.FAULT.value: "Operator attention required. Inspect diagnostics and status messages.",
        }.get(state, "Autofocus state updated.")
        self._set_state_card(state, detail)
        self._push_event(f"State -> {state}", level="state")

        now = time.monotonic()
        if state == AutofocusState.RECOVERY.value:
            if self._last_state != AutofocusState.RECOVERY.value:
                self._append_event_log("Entered RECOVERY state; autofocus corrections may be unreliable.")
                self._last_recovery_log_s = now
            elif self._last_recovery_log_s is None or (now - self._last_recovery_log_s) >= 60.0:
                self._append_event_log("Still in RECOVERY state.")
                self._last_recovery_log_s = now
        elif self._last_state == AutofocusState.RECOVERY.value:
            self._append_event_log(f"Exited RECOVERY state -> {state}")
            self._last_recovery_log_s = None
        self._last_state = state

    @Slot(str)
    def _on_fault(self, message: str) -> None:
        self._status.setText(message)
        self._push_event(message, level="fault")
        self._on_state(AutofocusState.FAULT.value)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        self._status.setText(message)
        level = "warn" if ("warning" in message.lower() or "failed" in message.lower()) else "info"
        self._push_event(message, level=level)

    def _on_lock_setpoint(self, enabled: bool) -> None:
        self._queue_config_update(lock_setpoint=bool(enabled))
        QtCore.QMetaObject.invokeMethod(self._af_worker, "reset_lock_state", QtCore.Qt.QueuedConnection)

    def _home_stage_to_centre(self) -> None:
        """Move stage to the midpoint of its travel range in a background thread."""
        def _task() -> None:
            try:
                get_range = getattr(self._stage, "get_range_um", None)
                if not callable(get_range):
                    self._signals.fault.emit(
                        "Home to centre: stage does not expose get_range_um(); "
                        "cannot determine travel range automatically."
                    )
                    return

                stage_range = get_range()
                if stage_range is None:
                    self._signals.fault.emit(
                        "Home to centre: stage returned no range information. "
                        "Move the stage manually to a mid-range position before calibrating."
                    )
                    return

                lo, hi = float(stage_range[0]), float(stage_range[1])
                if hi <= lo:
                    self._signals.fault.emit(
                        f"Home to centre: invalid stage range ({lo:.2f}, {hi:.2f})."
                    )
                    return

                centre = (lo + hi) / 2.0
                self._signals.status.emit(
                    f"Homing stage to centre: {centre:.2f} µm "
                    f"(range [{lo:.2f}, {hi:.2f}] µm) …"
                )
                self._stage.move_z_um(centre)
                actual = float(self._stage.get_z_um())
                self._append_event_log(
                    f"Homed stage to centre {centre:.3f} µm (actual readback {actual:.3f} µm). "
                    f"Re-calibrate only if the sample has physically moved."
                )
                self._signals.status.emit(
                    f"Stage at centre: {actual:.3f} µm. "
                    f"Re-calibrate only if the sample has physically moved."
                )
            except Exception as exc:
                self._signals.fault.emit(f"Home to centre failed: {exc}")

        threading.Thread(target=_task, daemon=True).start()

    def _run_calibration(self) -> None:
        with self._calibration_lock:
            if self._calibration_running:
                self._signals.status.emit("Calibration already running")
                return
            self._calibration_running = True
        self._signals.calibration_running.emit(True)
        self._save_settings()
        cal_half_range = max(0.05, float(self._cal_half_range.value()))
        cal_steps = max(5, int(self._cal_steps.value()))
        # Read ROI directly from the widget on the GUI thread so the
        # calibration always uses exactly what is drawn on screen, even if
        # a queued roi_changed signal hasn't been processed by the AF worker yet.
        pos = self._roi.pos()
        size = self._roi.size()
        gui_roi = Roi(
            x=max(0, int(pos.x())),
            y=max(0, int(pos.y())),
            width=max(1, int(size.x())),
            height=max(1, int(size.y())),
        )
        # Push the same ROI to the controller so post-calibration autofocus
        # uses the identical region.
        self._signals.roi_changed.emit((gui_roi.x, gui_roi.y, gui_roi.width, gui_roi.height))

        def _task() -> None:
            try:
                QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.QueuedConnection)
                roi = gui_roi
                center = float(self._stage.get_z_um())

                # Determine stage travel limits for sweep clamping.
                stage_range: tuple[float, float] | None = None
                get_range = getattr(self._stage, "get_range_um", None)
                if callable(get_range):
                    try:
                        stage_range = get_range()
                    except Exception:
                        pass
                # Fall back to config-level clamps if hardware query unavailable
                config_snap = self._controller.get_config_snapshot()
                if stage_range is None:
                    lo = config_snap.stage_min_um
                    hi = config_snap.stage_max_um
                    if lo is not None or hi is not None:
                        stage_range = (lo if lo is not None else -1e9, hi if hi is not None else 1e9)

                def _wait_for_settled_frame(last_seq: int, settle_s: float, move_time_s: float, timeout_s: float = 2.0):
                    """Wait for both settle time and a fresh frame in one loop.

                    Blocks until at least `settle_s` has elapsed since
                    `move_time_s` AND a new frame (seq > last_seq) is
                    available, whichever comes last.  This avoids the
                    stop-and-go cadence from separate sleep + frame-wait.
                    """
                    settle_deadline = move_time_s + settle_s
                    timeout_deadline = time.monotonic() + max(0.05, timeout_s)
                    while time.monotonic() < timeout_deadline:
                        if self._stop_evt.is_set():
                            raise RuntimeError("Calibration cancelled")
                        now = time.monotonic()
                        settled = now >= settle_deadline
                        frame, seq = self._frame_queue.get_latest()
                        if settled and frame is not None and seq > last_seq:
                            return frame, seq
                        time.sleep(0.005)
                    raise RuntimeError("Timed out waiting for settled camera frame during calibration")

                def _collect_and_check(
                    *,
                    half_range_um: float,
                    n_steps: int,
                    settle_s: float,
                    bidirectional: bool = True,
                ):
                    if n_steps < 2:
                        raise ValueError("n_steps must be >= 2")
                    z_min_um = center - half_range_um
                    z_max_um = center + half_range_um

                    # Clamp sweep to stage travel limits to avoid out-of-range errors.
                    if stage_range is not None:
                        old_min, old_max = z_min_um, z_max_um
                        z_min_um = max(z_min_um, stage_range[0])
                        z_max_um = min(z_max_um, stage_range[1])
                        if z_min_um != old_min or z_max_um != old_max:
                            self._signals.status.emit(
                                f"Sweep clamped to stage range [{z_min_um:.2f}, {z_max_um:.2f}] µm"
                            )
                    if z_max_um <= z_min_um:
                        raise ValueError(
                            f"Sweep range is empty after clamping to stage limits "
                            f"(center={center:.2f}, half_range={half_range_um:.2f}, "
                            f"stage_range={stage_range}). Move stage away from travel limit."
                        )

                    step = (z_max_um - z_min_um) / float(n_steps - 1)
                    forward_targets = [z_min_um + i * step for i in range(n_steps)]
                    targets = forward_targets + list(reversed(forward_targets)) if bidirectional else forward_targets

                    samples_local: list[CalibrationSample] = []
                    zhuang_samples_local: list[ZhuangCalibrationSample] = []
                    zhuang_raw_steps: list[tuple[float, float, float, object]] = []
                    failed_moves: list[tuple[float, Exception]] = []
                    _, last_seq = self._frame_queue.get_latest()

                    for i, target_z in enumerate(targets):
                        if self._stop_evt.is_set():
                            raise RuntimeError("Calibration cancelled")
                        try:
                            self._stage.move_z_um(target_z)
                        except Exception as exc:
                            failed_moves.append((target_z, exc))
                            self._signals.status.emit(
                                f"Calibrating {i+1}/{len(targets)} - skipped z={target_z:.3f} (move failed)"
                            )
                            continue
                        move_finished_s = time.monotonic()
                        frame, last_seq = _wait_for_settled_frame(last_seq, settle_s, move_finished_s)
                        # Read stage position after waiting for the settled frame so
                        # recorded Z matches the image used for calibration fitting.
                        measured_z = target_z
                        try:
                            measured_z = float(self._stage.get_z_um())
                        except Exception:
                            pass
                        err = astigmatic_error_signal(frame.image, roi)
                        weight = roi_total_intensity(frame.image, roi)
                        samples_local.append(CalibrationSample(z_um=measured_z, error=err, weight=max(0.0, weight)))

                        patch = extract_roi(frame.image, roi)
                        zhuang_raw_steps.append((measured_z, err, max(0.0, weight), patch))
                        self._signals.status.emit(f"Calibrating {i+1}/{len(targets)}")

                    if len(samples_local) < 2:
                        if failed_moves:
                            _, first_exc = failed_moves[0]
                            raise RuntimeError(
                                f"Calibration sweep failed: {len(samples_local)} succeeded, "
                                f"{len(failed_moves)} moves failed. First: {first_exc}"
                            )
                        raise RuntimeError("Calibration sweep could not collect enough valid points.")

                    # Run Gaussian PSF fits AFTER the Z sweep so stage motion cadence
                    # stays smooth during acquisition.
                    def _fit_one(
                        step: tuple[float, float, float, object],
                        *,
                        theta_fit: float | None,
                    ) -> tuple[ZhuangCalibrationSample, float]:
                        measured_z, err, weight, patch = step
                        gauss = fit_gaussian_psf(patch, theta=theta_fit)
                        theta_meas = float("nan")
                        if gauss is not None and gauss.r_squared > 0.15:
                            sx = gauss.sigma_x
                            sy = gauss.sigma_y
                            ell = gauss.ellipticity
                            fit_r2 = gauss.r_squared
                            theta_meas = float(gauss.theta)
                        else:
                            sx, sy = _moment_sigma_fallback(patch)
                            ell = sx / sy if sy > 0 else float("nan")
                            fit_r2 = float("nan")
                        return ZhuangCalibrationSample(
                            z_um=measured_z,
                            error=err,
                            weight=weight,
                            sigma_x=sx,
                            sigma_y=sy,
                            ellipticity=ell,
                            fit_r2=fit_r2,
                        ), theta_meas

                    n_fit = len(zhuang_raw_steps)
                    # Keep the GUI responsive by reserving CPU headroom instead of
                    # saturating all cores during post-sweep fitting.
                    cpu = os.cpu_count() or 1
                    workers = max(1, min(3, cpu - 1))
                    self._signals.status.emit(
                        f"Fitting PSF ({n_fit} frames, {workers} worker{'s' if workers != 1 else ''})"
                    )

                    def _fit_batch(*, theta_fit: float | None, progress_prefix: str) -> tuple[list[ZhuangCalibrationSample], list[float]]:
                        if workers > 1 and n_fit >= 8:
                            progress_step = max(1, n_fit // 20)
                            ordered: list[tuple[ZhuangCalibrationSample, float] | None] = [None] * n_fit
                            with ThreadPoolExecutor(max_workers=workers) as ex:
                                futures = {
                                    ex.submit(_fit_one, step, theta_fit=theta_fit): idx
                                    for idx, step in enumerate(zhuang_raw_steps)
                                }
                                done = 0
                                for fut in as_completed(futures):
                                    idx = futures[fut]
                                    ordered[idx] = fut.result()
                                    done += 1
                                    if done % progress_step == 0 or done == n_fit:
                                        self._signals.status.emit(f"{progress_prefix} {done}/{n_fit}")
                            pairs = [s for s in ordered if s is not None]
                            return [p[0] for p in pairs], [p[1] for p in pairs]
                        out_fit: list[ZhuangCalibrationSample] = []
                        out_theta: list[float] = []
                        for i, step in enumerate(zhuang_raw_steps, start=1):
                            fitted, th = _fit_one(step, theta_fit=theta_fit)
                            out_fit.append(fitted)
                            out_theta.append(th)
                            if i % max(1, n_fit // 20) == 0 or i == n_fit:
                                self._signals.status.emit(f"{progress_prefix} {i}/{n_fit}")
                        return out_fit, out_theta

                    # Strict replication of the publication protocol:
                    # 1) fit with free theta, 2) compute global theta, 3) refit with fixed theta.
                    pass1_samples, pass1_theta = _fit_batch(theta_fit=None, progress_prefix="PSF pass1")
                    try:
                        import numpy as np

                        theta_candidates = [
                            float(th)
                            for s, th in zip(pass1_samples, pass1_theta)
                            if np.isfinite(float(s.fit_r2)) and float(s.fit_r2) >= 0.5 and np.isfinite(float(th))
                        ]
                        if theta_candidates:
                            theta_arr = np.asarray(theta_candidates, dtype=float)
                            theta_global = float(np.angle(np.mean(np.exp(1j * theta_arr))))
                        else:
                            theta_global = 0.0
                    except Exception:
                        theta_global = 0.0

                    self._signals.status.emit(f"Global theta fixed at {theta_global:+0.3f} rad")
                    pass2_samples, _ = _fit_batch(theta_fit=theta_global, progress_prefix="PSF pass2")
                    zhuang_samples_local.extend(pass2_samples)

                    try:
                        z_report = fit_zhuang_calibration(zhuang_samples_local)
                        z_issues = zhuang_calibration_quality_issues(z_report)
                        fallback_fraction = 1.0 - (float(z_report.n_good_fits) / max(1.0, float(z_report.n_samples)))
                        if fallback_fraction > 0.35:
                            z_issues.append(
                                f"fallback fraction too high ({fallback_fraction:0.0%}); increase SNR or tighten ROI"
                            )
                        if z_issues:
                            raise ValueError(" ; ".join(z_issues))
                        return zhuang_samples_local, z_report.calibration, [], z_report
                    except Exception as exc:
                        self._signals.status.emit(f"Zhuang fit failed ({exc}); falling back to linear fit")
                        report_local = fit_linear_calibration_with_report(samples_local, robust=True)
                        issues_local = calibration_quality_issues(samples_local, report_local)
                        return zhuang_samples_local, report_local.calibration, issues_local, None

                samples, calibration_fit, issues, z_report = _collect_and_check(
                    half_range_um=cal_half_range,
                    n_steps=cal_steps,
                    settle_s=0.05,
                    bidirectional=True,
                )

                def _prompt_retry(message: str) -> bool:
                    evt = threading.Event()
                    with self._retry_prompt_lock:
                        self._retry_prompt_result = False
                        self._retry_prompt_event = evt
                    self._signals.calibration_retry_prompt.emit(message)
                    evt.wait(timeout=120.0)
                    with self._retry_prompt_lock:
                        result = bool(self._retry_prompt_result)
                        self._retry_prompt_event = None
                    return result

                mismatch_issue = "up/down sweep mismatch is high"
                if issues and any(mismatch_issue in issue for issue in issues):
                    try:
                        import numpy as np
                        xs0 = np.array([float(s.error) for s in samples])
                        ys0_abs = np.array([float(s.z_um) for s in samples])
                        y0_ref = float(np.mean(ys0_abs))
                        ys0 = ys0_abs - y0_ref
                        x_fit0 = np.linspace(float(xs0.min()), float(xs0.max()), 300)
                        y_fit0 = np.array([float(calibration_fit.error_to_z_offset_um(x)) for x in x_fit0])
                        split0 = len(xs0) // 2
                        self._signals.calibration_plot.emit({
                            "x": xs0[:split0].tolist(),
                            "y": ys0[:split0].tolist(),
                            "x_rev": xs0[split0:].tolist(),
                            "y_rev": ys0[split0:].tolist(),
                            "x_fit": x_fit0.tolist(),
                            "y_fit": y_fit0.tolist(),
                            "quality_text": "Preview only",
                            "quality_meta": "retry decision required",
                            "range_text": "--",
                            "range_meta": "sweep not accepted",
                            "fallback_text": "--",
                            "fallback_meta": "preview",
                            "accept_text": "REVIEW",
                            "accept_meta": "confirm retry or keep current fit",
                        })
                    except Exception:
                        pass

                    if _prompt_retry(
                        "Up/down sweep mismatch is high (possible backlash/stage settling).\n"
                        "Retry with slower settle and finer sweep?"
                    ):
                        retry_half_range = max(0.05, cal_half_range * 0.8)
                        retry_steps = max(cal_steps + 10, int(cal_steps * 1.5))
                        samples_retry, calibration_retry, issues_retry, z_report_retry = _collect_and_check(
                            half_range_um=retry_half_range,
                            n_steps=retry_steps,
                            settle_s=0.25,
                            bidirectional=True,
                        )
                        if len(issues_retry) < len(issues):
                            samples, calibration_fit, issues, z_report = samples_retry, calibration_retry, issues_retry, z_report_retry
                        if issues and any(mismatch_issue in issue for issue in issues):
                            if _prompt_retry(
                                "Mismatch persists.\n"
                                "Retry with slow one-way sweep to mitigate backlash?"
                            ):
                                one_way_half_range = max(0.05, retry_half_range)
                                one_way_steps = max(retry_steps, 31)
                                samples_one_way, calibration_one_way, issues_one_way, z_report_one_way = _collect_and_check(
                                    half_range_um=one_way_half_range,
                                    n_steps=one_way_steps,
                                    settle_s=0.35,
                                    bidirectional=False,
                                )
                                if len(issues_one_way) <= len(issues):
                                    samples, calibration_fit, issues, z_report = (
                                        samples_one_way,
                                        calibration_one_way,
                                        issues_one_way,
                                        z_report_one_way,
                                    )

                if issues:
                    self._signals.fault.emit("Calibration failed: " + "; ".join(issues))
                    return
                self._calibration = calibration_fit
                self._controller.calibration = self._calibration

                try:
                    import numpy as np

                    xs = np.array([float(s.error) for s in samples])
                    ys_abs = np.array([float(s.z_um) for s in samples])
                    y_ref = float(np.mean(ys_abs))
                    ys = ys_abs - y_ref
                    z_rel = ys
                    ell_vals = np.array([float(getattr(s, "ellipticity", float("nan"))) for s in samples])
                    r2_vals = np.array([float(getattr(s, "fit_r2", float("nan"))) for s in samples])
                    x_fit = np.linspace(float(xs.min()), float(xs.max()), 300)
                    y_fit = np.array([float(calibration_fit.error_to_z_offset_um(x)) for x in x_fit])
                    split = len(xs) // 2
                    current_z = float(self._stage.get_z_um())
                    current_z_rel = current_z - y_ref
                    usable = getattr(z_report, "usable_range_um", None) if z_report is not None else None
                    n_param = int(np.isfinite(r2_vals).sum())
                    n_fallback = int((~np.isfinite(r2_vals)).sum())
                    total_samples = max(1, len(samples))
                    fallback_fraction = float(n_fallback) / float(total_samples)
                    if z_report is not None:
                        quality_text = f"R²x {z_report.r2_x:.3f} / R²y {z_report.r2_y:.3f}"
                        quality_meta = "parametric axis fit quality"
                        accept_text = "ACCEPT"
                        accept_meta = "calibration armed"
                    else:
                        quality_text = "Linear fallback"
                        quality_meta = "parametric fit unavailable"
                        accept_text = "LIMITED"
                        accept_meta = "fallback calibration only"
                    if usable is not None:
                        range_text = f"{float(usable[0]):+.3f} to {float(usable[1]):+.3f} um"
                        range_meta = "usable Z range"
                    else:
                        z_min = float(np.min(z_rel))
                        z_max = float(np.max(z_rel))
                        range_text = f"{z_min:+.3f} to {z_max:+.3f} um"
                        range_meta = "swept Z range"
                    self._signals.calibration_plot.emit({
                        "x": xs[:split].tolist(),
                        "y": ys[:split].tolist(),
                        "x_rev": xs[split:].tolist(),
                        "y_rev": ys[split:].tolist(),
                        "x_fit": x_fit.tolist(),
                        "y_fit": y_fit.tolist(),
                        "current_z_rel": current_z_rel,
                        "usable": list(usable) if usable is not None else None,
                        "ell_z_param": z_rel[np.isfinite(r2_vals)].tolist(),
                        "ell_param": ell_vals[np.isfinite(r2_vals)].tolist(),
                        "ell_z_fallback": z_rel[~np.isfinite(r2_vals)].tolist(),
                        "ell_fallback": ell_vals[~np.isfinite(r2_vals)].tolist(),
                        "quality_text": quality_text,
                        "quality_meta": quality_meta,
                        "range_text": range_text,
                        "range_meta": range_meta,
                        "fallback_text": f"{fallback_fraction:0.0%}",
                        "fallback_meta": f"{n_fallback} fallback of {total_samples} samples",
                        "accept_text": accept_text,
                        "accept_meta": accept_meta,
                    })
                    if z_report is not None:
                        self._signals.status.emit(
                            f"Calibration quality: R²x={z_report.r2_x:.3f}, R²y={z_report.r2_y:.3f}, usable={z_report.usable_range_um}"
                        )
                        self._signals.status.emit(
                            f"Ellipticity classes: parametric={n_param}, moment-fallback={n_fallback} "
                            "(fallback points can appear as a near-flat band around ell~1)."
                        )
                        if usable is not None and not (usable[0] <= current_z_rel <= usable[1]):
                            self._signals.status.emit(
                                "Warning: current Z is outside calibration usable range. Move near focus and re-run calibration."
                            )
                except Exception:
                    pass

                out = Path(self._calibration_output_path)
                save_zhuang_calibration_samples_csv(out, samples)
                meta = CalibrationMetadata(
                    roi_size=(roi.width, roi.height),
                    stage_type=type(self._stage).__name__,
                    created_at_unix_s=time.time(),
                )
                save_calibration_metadata_json(out.with_suffix('.meta.json'), meta)
                self._signals.status.emit(f"Calibration saved: {out}")
            except Exception as exc:
                self._signals.fault.emit(f"Calibration failure: {exc}")
            finally:
                with self._calibration_lock:
                    self._calibration_running = False
                self._signals.calibration_running.emit(False)

        threading.Thread(target=_task, daemon=True).start()

    def closeEvent(self, event):  # noqa: N802
        self._stop_evt.set()
        try:
            QtCore.QMetaObject.invokeMethod(self._af_worker, "stop_loop", QtCore.Qt.BlockingQueuedConnection)
        except Exception:
            pass
        self._af_thread.quit()
        self._af_thread.wait(1000)
        if self._camera_worker.is_alive():
            self._camera_worker.join(timeout=1.0)
        for name in ("stop", "close"):
            method = getattr(self._camera, name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        if self._stats.total_frames > 0:
            err_vals = list(self._history_err)
            corr_vals = list(self._history_corr)
            rms_error_um = None
            corr_std_um = None
            if err_vals:
                rms_error_um = (sum(float(v) * float(v) for v in err_vals) / len(err_vals)) ** 0.5
            if corr_vals:
                mean_corr = sum(float(v) for v in corr_vals) / len(corr_vals)
                corr_std_um = (
                    sum((float(v) - mean_corr) ** 2 for v in corr_vals) / len(corr_vals)
                ) ** 0.5
            locked_count = sum(
                1 for st in self._history_state
                if st == AutofocusState.LOCKED.value
            )
            total_hist = len(self._history_state)
            lock_fraction = (locked_count / total_hist) if total_hist else None
            suggestions: list[str] = []
            if rms_error_um is not None:
                if rms_error_um > 0.05:
                    suggestions.append("High residual noise: reduce kp/ki or increase deadband.")
                elif rms_error_um < 0.008:
                    suggestions.append("Residual is very low: deadband may be reduced for tighter lock.")
            if corr_std_um is not None and corr_std_um > 0.03:
                suggestions.append("Correction jitter is high: increase dynamic deadband coefficient.")
            report = {
                "median_loop_latency_ms": (sorted(self._stats.loop_latency_ms)[len(self._stats.loop_latency_ms)//2] if self._stats.loop_latency_ms else None),
                "dropped_frames": self._stats.dropped_frames,
                "total_frames": self._stats.total_frames,
                "faults": self._stats.faults,
                "config": asdict(self._controller.get_config_snapshot()),
                "lock_metrics": {
                    "rms_error_nm": (None if rms_error_um is None else rms_error_um * 1000.0),
                    "correction_std_nm": (None if corr_std_um is None else corr_std_um * 1000.0),
                    "time_in_lock_fraction_proxy": lock_fraction,
                },
                "retuning_suggestions": suggestions,
            }
            report_path = Path(self._calibration_output_path).with_name("autofocus_run_report.json")
            try:
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                self._append_event_log(f"Saved run report: {report_path}")
            except Exception:
                pass
        self._save_settings()
        super().closeEvent(event)

    @Slot(bool)
    def _on_calibration_running(self, running: bool) -> None:
        self._cal_btn.setEnabled(not running)
        self._cal_btn.setText("Calibrating..." if running else "Calibrate")
        self._cal_progress.setText("Calibration running..." if running else "Calibration idle")
        self._push_event("Calibration started" if running else "Calibration stopped", level="state")

    @Slot(object)
    def _on_calibration_plot(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._cal_pts.setData(payload.get("x", []), payload.get("y", []))
        self._cal_pts_rev.setData(payload.get("x_rev", []), payload.get("y_rev", []))
        self._cal_fit.setData(payload.get("x_fit", []), payload.get("y_fit", []))
        self._ell_param.setData(payload.get("ell_z_param", []), payload.get("ell_param", []))
        self._ell_fallback.setData(payload.get("ell_z_fallback", []), payload.get("ell_fallback", []))
        try:
            self._cal_current_z.setValue(float(payload.get("current_z_rel", 0.0)))
        except Exception:
            pass
        usable = payload.get("usable", None)
        if isinstance(usable, (list, tuple)) and len(usable) == 2:
            try:
                lo = float(usable[0]); hi = float(usable[1])
                self._cal_usable_band.setRegion((lo, hi))
                self._ell_usable_band.setRegion((lo, hi))
            except Exception:
                pass
        self._cal_quality_val.setText(str(payload.get("quality_text", "--")))
        self._cal_quality_meta.setText(str(payload.get("quality_meta", "awaiting sweep")))
        self._cal_range_val.setText(str(payload.get("range_text", "--")))
        self._cal_range_meta.setText(str(payload.get("range_meta", "focus-relative")))
        self._cal_fallback_val.setText(str(payload.get("fallback_text", "--")))
        self._cal_fallback_meta.setText(str(payload.get("fallback_meta", "rejected from model")))
        self._cal_accept_val.setText(str(payload.get("accept_text", "--")))
        self._cal_accept_meta.setText(str(payload.get("accept_meta", "status")))
        accept_text = str(payload.get("accept_text", "--")).upper()
        accept_color = UI_THEME["text"]
        if "ACCEPT" in accept_text:
            accept_color = UI_THEME["ok"]
        elif "LIMIT" in accept_text:
            accept_color = UI_THEME["warn"]
        elif "REJECT" in accept_text or "FAIL" in accept_text:
            accept_color = UI_THEME["fault"]
        self._cal_accept_val.setStyleSheet(
            f"color: {accept_color}; font-size: 24px; font-weight: 700; font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;"
        )

    @Slot(str)
    def _on_calibration_retry_prompt(self, message: str) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Calibration Retry",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        with self._retry_prompt_lock:
            self._retry_prompt_result = (reply == QtWidgets.QMessageBox.Yes)
            evt = self._retry_prompt_event
        if evt is not None:
            evt.set()

    def start(self) -> None:
        self._camera_worker.start()


def launch_pg_autofocus_gui(
    camera: CameraInterface,
    stage: StageInterface,
    *,
    calibration: CalibrationLike,
    default_config: AutofocusConfig,
    calibration_output_path: str | None = None,
    calibration_half_range_um: float = 0.75,
    calibration_steps: int = 21,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = AutofocusMainWindow(
        camera=camera,
        stage=stage,
        calibration=calibration,
        default_config=default_config,
        calibration_output_path=calibration_output_path,
        calibration_half_range_um=calibration_half_range_um,
        calibration_steps=calibration_steps,
    )
    win.resize(1280, 860)
    win.start()
    win.show()
    app.exec()

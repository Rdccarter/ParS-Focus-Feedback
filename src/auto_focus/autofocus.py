from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .calibration import FocusCalibration, ZhuangFocusCalibration
from .focus_metric import Roi, astigmatic_error_signal, centroid_near_edge, extract_roi, fit_gaussian_fast, roi_total_intensity
from .interfaces import CameraFrame, CameraInterface, StageInterface
from .pid_controller import PidConfig, PidController
from .zhuang import findz_analytical

# Either calibration type works - both provide error_to_z_offset_um().
CalibrationLike = FocusCalibration | ZhuangFocusCalibration


class AutofocusState(str, Enum):
    IDLE = "IDLE"
    CALIBRATED_READY = "CALIBRATED_READY"
    LOCKING = "LOCKING"
    LOCKED = "LOCKED"
    DEGRADED = "DEGRADED"
    RECOVERY = "RECOVERY"
    LOST = "LOST"
    FAULT = "FAULT"


@dataclass(slots=True)
class DetectionBounds:
    """Acceptable bounds for per-frame quality gating."""

    fwhm_min_px: float = 1.0
    fwhm_max_px: float = 20.0
    ellipticity_min: float = 0.5
    ellipticity_max: float = 2.0
    r_squared_min: float = 0.0
    intensity_min: float = 0.0
    intensity_min_fraction: float = 0.0


class DetectionFilter:
    """Sliding-window detection filter (CylLensGUI pattern)."""

    def __init__(
        self,
        window_size: int = 5,
        pass_threshold: float = 0.35,
        bounds: DetectionBounds | None = None,
    ) -> None:
        self._window: deque[bool] = deque([True] * window_size, maxlen=window_size)
        self._pass_threshold = pass_threshold
        self.bounds = bounds or DetectionBounds()
        self._intensity_baseline: float | None = None
        self._baseline_alpha: float = 0.02

    def check(self, fwhm: float, ellipticity: float, r_squared: float, total_intensity: float) -> bool:
        b = self.bounds

        if self._intensity_baseline is None:
            self._intensity_baseline = total_intensity
        else:
            self._intensity_baseline = (
                (1.0 - self._baseline_alpha) * self._intensity_baseline
                + self._baseline_alpha * total_intensity
            )

        frame_ok = True
        if math.isfinite(fwhm) and not (b.fwhm_min_px <= fwhm <= b.fwhm_max_px):
            frame_ok = False
        if math.isfinite(ellipticity) and not (b.ellipticity_min <= ellipticity <= b.ellipticity_max):
            frame_ok = False
        if math.isfinite(r_squared) and r_squared < b.r_squared_min:
            frame_ok = False
        if total_intensity < b.intensity_min:
            frame_ok = False
        if b.intensity_min_fraction > 0 and self._intensity_baseline is not None:
            if total_intensity < b.intensity_min_fraction * self._intensity_baseline:
                frame_ok = False

        self._window.append(frame_ok)
        pass_fraction = sum(self._window) / len(self._window)
        return frame_ok and (pass_fraction > self._pass_threshold)

    def reset(self) -> None:
        self._window = deque([True] * self._window.maxlen, maxlen=self._window.maxlen)
        self._intensity_baseline = None

    @property
    def pass_fraction(self) -> float:
        return sum(self._window) / len(self._window)


@dataclass(slots=True)
class AutofocusConfig:
    roi: Roi
    loop_hz: float = 30.0
    # Cap effective control-step dt to avoid large corrective jumps after
    # temporary stalls (GUI pauses, GC, device hiccups).
    max_dt_s: float = 0.2
    # PID control gains, in units of um of stage command per um equivalent error.
    kp: float = 0.6
    ki: float = 0.15
    kd: float = 0.0
    max_step_um: float = 0.25
    integral_limit_um: float = 2.0
    stage_min_um: float | None = None
    stage_max_um: float | None = None
    # Safety clamp around initial lock position to avoid runaway absolute jumps.
    max_abs_excursion_um: float | None = 5.0
    # Freeze control updates when ROI total intensity drops below threshold.
    min_roi_intensity: float | None = None
    # Relative gate: pause control when ROI intensity drops below this
    # fraction of the running baseline intensity.
    min_roi_intensity_fraction: float | None = None
    # Declare target LOST after this many consecutive low-intensity frames.
    lost_intensity_frames: int = 20
    # Additional quality gates before applying corrections.
    min_roi_variance: float | None = None
    max_saturated_fraction: float | None = None
    saturation_level: float = 65535.0
    # Exponential moving average smoothing factor for the error signal.
    # 0.0 = no filtering (raw error used), 1.0 = ignore new measurements.
    error_alpha: float = 0.0
    # Derivative filtering coefficient (EMA on d(error)/dt).
    derivative_alpha: float = 0.7
    # Compensate stage response delay by decaying D term based on command age.
    stage_latency_s: float = 0.0
    # Reject frames when PSF centroid is within this many pixels of the ROI
    # boundary, to avoid biased second moments from a truncated PSF.
    edge_margin_px: float = 0.0
    # Do not issue stage moves smaller than this threshold (um) to reduce
    # high-frequency dithering/oscillation near focus.
    command_deadband_um: float = 0.005
    # Slew limit in um/s for commanded target changes.
    max_slew_rate_um_per_s: float | None = None
    # Setpoint locking + optional guarded recentering.
    lock_setpoint: bool = True
    recenter_alpha: float = 0.0
    # Guardrail: reject control updates when measured error is outside the
    # calibration domain. This prevents lookup extrapolation/clamping from
    # producing large wrong-way moves when changing ROI/target.
    calibration_error_margin: float = 0.02
    # Optional XY ROI tracking done in the control loop before error extraction.
    track_roi: bool = True
    track_gain: float = 0.4
    track_deadband_px: float = 1.5
    track_max_shift_px: float = 8.0
    use_gaussian_fit: bool = True
    gaussian_fast_mode: bool = False
    gaussian_theta: float | None = 0.0
    use_analytical_inversion: bool = True
    # Strict Lenstra-style analytical loop: use ellipticity->findz directly
    # without lookup disagreement override and without second-moment domain
    # gating as the primary control-domain check.
    strict_lenstra_match: bool = True
    # Background percentile used for astigmatic error background subtraction.
    background_percentile: float = 20.0
    # Optional multi-peak rejection: minimum weighted excess kurtosis
    # (x and y) required to accept ROI content as a single compact spot.
    min_intensity_kurtosis: float | None = None
    # Sliding-window detection filter (replaces degraded counter heuristic).
    detection_window_size: int = 5
    detection_pass_threshold: float = 0.35
    detection_bounds: DetectionBounds | None = None
    # Calibration acceptance gates before arming autofocus.
    min_cal_lookup_error_span: float = 0.06
    min_cal_usable_half_range_um: float = 0.10
    min_cal_linear_slope_abs: float = 0.05
    # Safe engage: require consecutive good frames then ramp control authority.
    engage_hold_good_frames: int = 0
    engage_ramp_s: float = 0.0
    # Confidence-weighted control scaling.
    min_confidence_scale: float = 1.0
    # Dynamic deadband: base + c*noise_sigma (estimated from recent error_um).
    dynamic_deadband_coeff: float = 0.0
    dynamic_deadband_window: int = 30
    # Anti-runaway guards.
    runaway_sign_frames: int = 4
    runaway_growth_frames: int = 5
    # Stage lag guard.
    max_stage_lag_um: float = 0.20
    stage_lag_fault_frames: int = 5
    # Adaptive ROI sizing under low-confidence conditions.
    roi_expand_on_low_confidence: bool = False
    roi_expand_step_px: int = 2
    roi_expand_max_px: int = 8
    roi_shrink_stable_frames: int = 15
    # Temporal filtering for error/ellipticity jitter suppression.
    error_median3: bool = True
    ellipticity_ema_alpha: float = 0.3

    @property
    def track_roi_xy(self) -> bool:
        """Backward-compatible alias for ``track_roi``."""
        return self.track_roi

    @track_roi_xy.setter
    def track_roi_xy(self, value: bool) -> None:
        self.track_roi = bool(value)

    @property
    def track_roi_gain(self) -> float:
        """Backward-compatible alias for ``track_gain``."""
        return self.track_gain

    @track_roi_gain.setter
    def track_roi_gain(self, value: float) -> None:
        self.track_gain = float(value)

    @property
    def track_roi_deadband_px(self) -> float:
        """Backward-compatible alias for ``track_deadband_px``."""
        return self.track_deadband_px

    @track_roi_deadband_px.setter
    def track_roi_deadband_px(self, value: float) -> None:
        self.track_deadband_px = float(value)

    @property
    def track_roi_max_step_px(self) -> float:
        """Backward-compatible alias for ``track_max_shift_px``."""
        return self.track_max_shift_px

    @track_roi_max_step_px.setter
    def track_roi_max_step_px(self, value: float) -> None:
        self.track_max_shift_px = float(value)


@dataclass(slots=True)
class AutofocusSample:
    timestamp_s: float
    error: float
    error_um: float
    stage_z_um: float
    commanded_z_um: float
    roi_total_intensity: float
    control_applied: bool
    confidence_ok: bool
    state: AutofocusState
    loop_latency_ms: float
    roi: Roi
    diagnostic: str = ""
    ellipticity: float = 0.0
    fwhm_px: float = 0.0
    fit_r_squared: float = 0.0
    detection_pass_rate: float = 1.0
    roi_x: int = 0
    roi_y: int = 0
    domain_margin: float = 0.0
    confidence_scale: float = 1.0
    stage_lag_um: float = 0.0


class AstigmaticAutofocusController:
    """Closed-loop focus controller for a single astigmatic PSF target."""

    def __init__(
        self,
        camera: CameraInterface,
        stage: StageInterface,
        config: AutofocusConfig,
        calibration: CalibrationLike,
        initial_integral_um: float = 0.0,
    ) -> None:
        self._camera = camera
        self._stage = stage
        self._config = config
        self._validate_config()
        self._calibration = calibration
        pid_config = PidConfig(
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            integral_limit_um=config.integral_limit_um,
            max_step_um=config.max_step_um,
            command_deadband_um=config.command_deadband_um,
            max_slew_rate_um_per_s=config.max_slew_rate_um_per_s,
            derivative_alpha=config.derivative_alpha,
            stage_min_um=config.stage_min_um,
            stage_max_um=config.stage_max_um,
            max_excursion_um=config.max_abs_excursion_um,
        )
        self._pid = PidController(pid_config, initial_output=0.0)
        self._last_frame_ts: float | None = None
        self._z_lock_center_um: float | None = None
        self._setpoint_error: float | None = None
        self._setpoint_z_offset_um: float | None = None
        self._roi_intensity_baseline: float | None = None
        self._state = AutofocusState.CALIBRATED_READY
        self._low_intensity_count = 0
        self._detection_filter = DetectionFilter(
            window_size=config.detection_window_size,
            pass_threshold=config.detection_pass_threshold,
            bounds=config.detection_bounds or DetectionBounds(),
        )
        self._config_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._engage_good_frames = 0
        self._engage_started_s: float | None = None
        self._recent_error_raw: deque[float] = deque(maxlen=5)
        self._last_filtered_error: float | None = None
        self._recent_error_um: deque[float] = deque(maxlen=max(10, int(config.dynamic_deadband_window)))
        self._ell_ema: float | None = None
        self._last_commanded_z_um: float | None = None
        self._stage_lag_bad_frames = 0
        self._sign_mismatch_count = 0
        self._growth_count = 0
        self._last_abs_error_um: float | None = None
        self._nominal_roi_size = (int(config.roi.width), int(config.roi.height))
        self._stable_roi_frames = 0

    @property
    def loop_hz(self) -> float:
        with self._config_lock:
            return self._config.loop_hz

    @property
    def calibration(self) -> CalibrationLike:
        with self._state_lock:
            return self._calibration

    @calibration.setter
    def calibration(self, value: CalibrationLike) -> None:
        with self._state_lock:
            self._calibration = value
            self._state = AutofocusState.CALIBRATED_READY
            self._engage_good_frames = 0
            self._engage_started_s = None

    def get_config_snapshot(self) -> AutofocusConfig:
        with self._config_lock:
            return copy.deepcopy(self._config)

    def update_config(self, **kwargs) -> None:
        with self._config_lock:
            det_changed = False
            pid_changed = False
            pid_keys = {
                "kp", "ki", "kd", "integral_limit_um", "max_step_um",
                "command_deadband_um", "max_slew_rate_um_per_s",
                "derivative_alpha", "stage_min_um", "stage_max_um",
                "max_abs_excursion_um",
            }
            for key, value in kwargs.items():
                if not hasattr(self._config, key):
                    raise AttributeError(f"Unknown AutofocusConfig field: {key}")
                setattr(self._config, key, value)
                if key in {"detection_window_size", "detection_pass_threshold", "detection_bounds"}:
                    det_changed = True
                if key in pid_keys:
                    pid_changed = True
            self._validate_config()
            if det_changed:
                self._detection_filter = DetectionFilter(
                    window_size=self._config.detection_window_size,
                    pass_threshold=self._config.detection_pass_threshold,
                    bounds=self._config.detection_bounds or DetectionBounds(),
                )
            if pid_changed:
                lock_center = self._z_lock_center_um
                self._pid = PidController(
                    PidConfig(
                        kp=self._config.kp,
                        ki=self._config.ki,
                        kd=self._config.kd,
                        integral_limit_um=self._config.integral_limit_um,
                        max_step_um=self._config.max_step_um,
                        command_deadband_um=self._config.command_deadband_um,
                        max_slew_rate_um_per_s=self._config.max_slew_rate_um_per_s,
                        derivative_alpha=self._config.derivative_alpha,
                        stage_min_um=self._config.stage_min_um,
                        stage_max_um=self._config.stage_max_um,
                        max_excursion_um=self._config.max_abs_excursion_um,
                    ),
                    initial_output=self._pid.output,
                )
                if lock_center is not None:
                    self._pid.set_lock_center(lock_center)

    def update_roi(self, roi: Roi) -> None:
        self.update_config(roi=roi)
        self.reset_lock_state()

    def reset_lock_state(self) -> None:
        """Clear control memory so lock is re-acquired cleanly after ROI changes."""
        with self._state_lock:
            self._z_lock_center_um = None
            self._setpoint_error = None
            self._setpoint_z_offset_um = None
            self._roi_intensity_baseline = None
            self._last_frame_ts = None
            self._low_intensity_count = 0
            self._engage_good_frames = 0
            self._engage_started_s = None
            self._recent_error_raw.clear()
            self._last_filtered_error = None
            self._recent_error_um.clear()
            self._ell_ema = None
            self._last_commanded_z_um = None
            self._stage_lag_bad_frames = 0
            self._sign_mismatch_count = 0
            self._growth_count = 0
            self._last_abs_error_um = None
            self._stable_roi_frames = 0
            self._pid.reset()
            self._detection_filter.reset()
            self._state = AutofocusState.CALIBRATED_READY

    def _validate_config(self) -> None:
        if self._config.loop_hz <= 0:
            raise ValueError("loop_hz must be > 0")
        if self._config.max_dt_s <= 0:
            raise ValueError("max_dt_s must be > 0")
        if self._config.max_step_um < 0:
            raise ValueError("max_step_um must be >= 0")
        if self._config.integral_limit_um < 0:
            raise ValueError("integral_limit_um must be >= 0")
        if not 0.0 <= self._config.error_alpha <= 1.0:
            raise ValueError("error_alpha must be in [0.0, 1.0]")
        if not 0.0 <= self._config.derivative_alpha <= 1.0:
            raise ValueError("derivative_alpha must be in [0.0, 1.0]")
        if self._config.edge_margin_px < 0:
            raise ValueError("edge_margin_px must be >= 0")
        if self._config.max_abs_excursion_um is not None and self._config.max_abs_excursion_um < 0:
            raise ValueError("max_abs_excursion_um must be >= 0 when provided")
        if self._config.command_deadband_um < 0:
            raise ValueError("command_deadband_um must be >= 0")
        if self._config.max_slew_rate_um_per_s is not None and self._config.max_slew_rate_um_per_s <= 0:
            raise ValueError("max_slew_rate_um_per_s must be > 0 when provided")
        if self._config.calibration_error_margin < 0:
            raise ValueError("calibration_error_margin must be >= 0")
        if self._config.min_roi_intensity_fraction is not None and not 0.0 <= self._config.min_roi_intensity_fraction <= 1.0:
            raise ValueError("min_roi_intensity_fraction must be in [0.0, 1.0] when provided")
        if self._config.track_gain < 0:
            raise ValueError("track_gain must be >= 0")
        if self._config.track_deadband_px < 0:
            raise ValueError("track_deadband_px must be >= 0")
        if self._config.track_max_shift_px <= 0:
            raise ValueError("track_max_shift_px must be > 0")
        if not 0.0 <= self._config.background_percentile <= 100.0:
            raise ValueError("background_percentile must be in [0.0, 100.0]")
        if self._config.lost_intensity_frames < 1:
            raise ValueError("lost_intensity_frames must be >= 1")
        if self._config.detection_window_size < 1:
            raise ValueError("detection_window_size must be >= 1")
        if not 0.0 <= self._config.detection_pass_threshold <= 1.0:
            raise ValueError("detection_pass_threshold must be in [0.0, 1.0]")
        if self._config.engage_hold_good_frames < 0:
            raise ValueError("engage_hold_good_frames must be >= 0")
        if self._config.engage_ramp_s < 0:
            raise ValueError("engage_ramp_s must be >= 0")
        if not 0.0 <= self._config.min_confidence_scale <= 1.0:
            raise ValueError("min_confidence_scale must be in [0,1]")
        if self._config.dynamic_deadband_coeff < 0:
            raise ValueError("dynamic_deadband_coeff must be >= 0")
        if self._config.dynamic_deadband_window < 3:
            raise ValueError("dynamic_deadband_window must be >= 3")
        if self._config.runaway_sign_frames < 2:
            raise ValueError("runaway_sign_frames must be >= 2")
        if self._config.runaway_growth_frames < 2:
            raise ValueError("runaway_growth_frames must be >= 2")
        if self._config.max_stage_lag_um < 0:
            raise ValueError("max_stage_lag_um must be >= 0")
        if self._config.stage_lag_fault_frames < 1:
            raise ValueError("stage_lag_fault_frames must be >= 1")
        if not 0.0 <= self._config.ellipticity_ema_alpha <= 1.0:
            raise ValueError("ellipticity_ema_alpha must be in [0,1]")

    def check_calibration_ready(self) -> tuple[bool, float, list[str]]:
        issues: list[str] = []
        score = 1.0
        cal = self._calibration

        slope = float(getattr(cal, "error_to_um", 0.0))
        if abs(slope) < self._config.min_cal_linear_slope_abs:
            issues.append(f"slope too small ({slope:+0.4f})")
            score -= 0.35

        lookup = getattr(cal, "lookup", None)
        error_values = getattr(lookup, "error_values", None)
        if error_values is not None and len(error_values) >= 2:
            try:
                lo = min(float(v) for v in error_values)
                hi = max(float(v) for v in error_values)
                span = hi - lo
                if span < self._config.min_cal_lookup_error_span:
                    issues.append(f"lookup span too small ({span:0.4f})")
                    score -= 0.35
            except Exception:
                issues.append("invalid lookup domain")
                score -= 0.25

        usable = getattr(cal, "usable_range_um", None)
        if usable is not None:
            try:
                lo, hi = float(usable[0]), float(usable[1])
                if lo > -self._config.min_cal_usable_half_range_um or hi < self._config.min_cal_usable_half_range_um:
                    issues.append(f"usable range weak around focus [{lo:+0.3f},{hi:+0.3f}]")
                    score -= 0.3
            except Exception:
                pass

        ok = len(issues) == 0
        return ok, max(0.0, min(1.0, score)), issues

    def _calibration_domain_margin(self, error: float, config: AutofocusConfig) -> float:
        checker = getattr(self._calibration, "is_error_in_range", None)
        if callable(checker):
            lookup = getattr(self._calibration, "lookup", None)
            vals = getattr(lookup, "error_values", None)
            if vals is not None and len(vals) >= 2:
                lo = min(float(v) for v in vals) - config.calibration_error_margin
                hi = max(float(v) for v in vals) + config.calibration_error_margin
                return min(error - lo, hi - error)
        if getattr(self._calibration, "error_min", None) is not None and getattr(self._calibration, "error_max", None) is not None:
            lo = float(getattr(self._calibration, "error_min")) - config.calibration_error_margin
            hi = float(getattr(self._calibration, "error_max")) + config.calibration_error_margin
            return min(error - lo, hi - error)
        return float("inf")

    def _confidence_scale(
        self,
        *,
        fit_r2: float,
        total_intensity: float,
        baseline_intensity: float | None,
        detection_pass_rate: float,
        config: AutofocusConfig,
    ) -> float:
        r2_scale = 1.0
        if math.isfinite(fit_r2):
            r2_min = (config.detection_bounds.r_squared_min if config.detection_bounds is not None else 0.0)
            r2_scale = 0.0 if fit_r2 <= r2_min else min(1.0, max(0.0, (fit_r2 - r2_min) / max(1e-6, 1.0 - r2_min)))
        intensity_scale = 1.0
        if baseline_intensity is not None and baseline_intensity > 0:
            intensity_scale = min(1.0, max(0.0, total_intensity / baseline_intensity))
        det_scale = min(1.0, max(0.0, detection_pass_rate))
        score = 0.45 * r2_scale + 0.35 * intensity_scale + 0.20 * det_scale
        return max(config.min_confidence_scale, min(1.0, score))

    def _dynamic_deadband(self, config: AutofocusConfig) -> float:
        base = float(config.command_deadband_um)
        if len(self._recent_error_um) < 5:
            return base
        mean = sum(self._recent_error_um) / len(self._recent_error_um)
        var = sum((v - mean) ** 2 for v in self._recent_error_um) / len(self._recent_error_um)
        sigma = math.sqrt(max(0.0, var))
        return max(base, base + config.dynamic_deadband_coeff * sigma)

    def _safe_engage_scale(self, config: AutofocusConfig, confidence_ok: bool) -> tuple[bool, float]:
        if config.engage_hold_good_frames <= 0:
            return True, 1.0
        if not confidence_ok:
            self._engage_good_frames = 0
            self._engage_started_s = None
            return False, 0.0
        self._engage_good_frames += 1
        if self._engage_good_frames < config.engage_hold_good_frames:
            return False, 0.0
        if self._engage_started_s is None:
            self._engage_started_s = time.monotonic()
        if config.engage_ramp_s <= 0:
            return True, 1.0
        ramp = (time.monotonic() - self._engage_started_s) / config.engage_ramp_s
        return True, max(0.0, min(1.0, ramp))

    def _filtered_error(self, raw_error: float, config: AutofocusConfig) -> float:
        self._recent_error_raw.append(float(raw_error))
        out = float(raw_error)
        if config.error_median3 and len(self._recent_error_raw) >= 3:
            window = list(self._recent_error_raw)[-3:]
            out = sorted(window)[1]
        if config.error_alpha > 0.0 and self._last_filtered_error is not None:
            out = config.error_alpha * self._last_filtered_error + (1.0 - config.error_alpha) * out
        self._last_filtered_error = float(out)
        return float(out)

    def _adaptive_roi_resize(self, image, roi: Roi, config: AutofocusConfig, *, confidence_ok: bool) -> Roi:
        if not config.roi_expand_on_low_confidence:
            return roi
        step = max(1, int(config.roi_expand_step_px))
        nominal_w, nominal_h = self._nominal_roi_size
        if confidence_ok:
            self._stable_roi_frames += 1
            if self._stable_roi_frames < config.roi_shrink_stable_frames:
                return roi
            self._stable_roi_frames = 0
            if roi.width <= nominal_w and roi.height <= nominal_h:
                return roi
            new_w = max(nominal_w, roi.width - step)
            new_h = max(nominal_h, roi.height - step)
        else:
            self._stable_roi_frames = 0
            max_w = nominal_w + int(config.roi_expand_max_px)
            max_h = nominal_h + int(config.roi_expand_max_px)
            if roi.width >= max_w and roi.height >= max_h:
                return roi
            new_w = min(max_w, roi.width + step)
            new_h = min(max_h, roi.height + step)

        h = len(image)
        w = len(image[0]) if h > 0 else 0
        cx = roi.x + roi.width // 2
        cy = roi.y + roi.height // 2
        new_x = int(max(0, min(w - new_w, cx - new_w // 2)))
        new_y = int(max(0, min(h - new_h, cy - new_h // 2)))
        return Roi(new_x, new_y, int(new_w), int(new_h))

    def _roi_center_from_patch(self, patch) -> tuple[float, float] | None:
        try:
            import numpy as np

            arr = np.asarray(patch, dtype=float)
            if arr.ndim != 2 or arr.size == 0:
                return None
            total = float(np.sum(arr))
            if total <= 0:
                return None
            y_idx, x_idx = np.indices(arr.shape, dtype=float)
            cx = float(np.sum(x_idx * arr) / total)
            cy = float(np.sum(y_idx * arr) / total)
            return cx, cy
        except Exception:
            return None

    def _recenter_roi_if_enabled(
        self,
        image,
        roi: Roi,
        config: AutofocusConfig,
        *,
        force: bool = False,
        centroid: tuple[float, float] | None = None,
    ) -> Roi:
        track_enabled = bool(config.track_roi)
        if (not track_enabled) and (not force):
            return roi
        if centroid is None:
            patch = extract_roi(image, roi)
            centroid = self._roi_center_from_patch(patch)
        if centroid is None:
            return roi
        cx, cy = centroid
        target_cx = (roi.width - 1) / 2.0
        target_cy = (roi.height - 1) / 2.0
        dx = cx - target_cx
        dy = cy - target_cy
        deadband = float(config.track_deadband_px)
        if (not force) and abs(dx) <= deadband and abs(dy) <= deadband:
            return roi

        gain = float(config.track_gain)
        max_shift = float(config.track_max_shift_px)
        shift_x = max(-max_shift, min(max_shift, gain * dx))
        shift_y = max(-max_shift, min(max_shift, gain * dy))

        h = len(image)
        w = len(image[0]) if h > 0 else 0
        new_x = int(max(0.0, min(float(w - roi.width), roi.x + shift_x)))
        new_y = int(max(0.0, min(float(h - roi.height), roi.y + shift_y)))
        if new_x == roi.x and new_y == roi.y:
            return roi
        return Roi(new_x, new_y, roi.width, roi.height)

    def _tracking_centroid_ok(
        self,
        centroid: tuple[float, float] | None,
        roi: Roi,
        config: AutofocusConfig,
        *,
        fit=None,
    ) -> bool:
        if centroid is None:
            return False
        try:
            cx = float(centroid[0])
            cy = float(centroid[1])
        except Exception:
            return False
        if (not math.isfinite(cx)) or (not math.isfinite(cy)):
            return False
        if cx < 0.0 or cy < 0.0 or cx > (roi.width - 1) or cy > (roi.height - 1):
            return False

        # When a Gaussian fit is available, require it to look plausible before
        # using its centroid to move the ROI. This avoids noise-driven drift.
        if fit is not None:
            bounds = config.detection_bounds or DetectionBounds()
            try:
                fwhm = float(getattr(fit, "fwhm", float("nan")))
                ell = float(getattr(fit, "ellipticity", float("nan")))
                r2 = float(getattr(fit, "r_squared", float("nan")))
            except Exception:
                return False
            if math.isfinite(fwhm) and not (bounds.fwhm_min_px <= fwhm <= bounds.fwhm_max_px):
                return False
            if math.isfinite(ell) and not (bounds.ellipticity_min <= ell <= bounds.ellipticity_max):
                return False
            if math.isfinite(r2) and r2 < bounds.r_squared_min:
                return False
        return True

    def _is_error_in_calibration_domain(self, error: float, config: AutofocusConfig) -> bool:
        checker = getattr(self._calibration, "is_error_in_range", None)
        if callable(checker):
            try:
                return bool(checker(error, margin=config.calibration_error_margin))
            except Exception:
                pass

        lookup = getattr(self._calibration, "lookup", None)
        error_values = getattr(lookup, "error_values", None)
        if error_values is None:
            return True
        if len(error_values) < 2:
            return True
        lo = min(float(v) for v in error_values) - config.calibration_error_margin
        hi = max(float(v) for v in error_values) + config.calibration_error_margin
        return lo <= error <= hi


    def _calibration_error_at_focus(self) -> float:
        with self._state_lock:
            calibration = self._calibration
        try:
            return float(getattr(calibration, "error_at_focus", 0.0))
        except Exception:
            return 0.0

    def _weighted_excess_kurtosis_min(self, patch) -> float | None:
        try:
            import numpy as np

            arr = np.asarray(patch, dtype=float)
            if arr.ndim != 2 or arr.size == 0:
                return None
            total = float(np.sum(arr))
            if total <= 0:
                return None
            y_idx, x_idx = np.indices(arr.shape, dtype=float)
            cx = float(np.sum(x_idx * arr) / total)
            cy = float(np.sum(y_idx * arr) / total)
            vx = float(np.sum(((x_idx - cx) ** 2) * arr) / total)
            vy = float(np.sum(((y_idx - cy) ** 2) * arr) / total)
            if vx <= 1e-12 or vy <= 1e-12:
                return None
            kx = float(np.sum(((x_idx - cx) ** 4) * arr) / total) / (vx * vx) - 3.0
            ky = float(np.sum(((y_idx - cy) ** 4) * arr) / total) / (vy * vy) - 3.0
            return min(kx, ky)
        except Exception:
            return None

    def _roi_confidence_ok(self, image, total_intensity: float, config: AutofocusConfig, baseline_intensity: float | None = None) -> tuple[bool, str, bool]:
        if config.min_roi_intensity is not None and total_intensity < config.min_roi_intensity:
            return False, "low ROI confidence (absolute intensity)", True
        if (
            config.min_roi_intensity_fraction is not None
            and baseline_intensity is not None
            and total_intensity < (baseline_intensity * config.min_roi_intensity_fraction)
        ):
            return False, "low ROI confidence (relative intensity)", True

        try:
            import numpy as np

            arr = np.asarray(image, dtype=float)
            if arr.ndim != 2:
                return False, "low ROI confidence (invalid frame)", False
            y0 = max(0, int(config.roi.y))
            x0 = max(0, int(config.roi.x))
            y1 = min(arr.shape[0], y0 + int(config.roi.height))
            x1 = min(arr.shape[1], x0 + int(config.roi.width))
            if y1 <= y0 or x1 <= x0:
                return False, "low ROI confidence (invalid ROI)", False
            patch = arr[y0:y1, x0:x1]
            if patch.size == 0:
                return False, "low ROI confidence (empty ROI)", False
            pixels = [float(v) for v in patch.ravel()]
        except Exception:
            patch = [row[config.roi.x : config.roi.x + config.roi.width] for row in image[config.roi.y : config.roi.y + config.roi.height]]
            if len(patch) == 0 or len(patch[0]) == 0:
                return False, "low ROI confidence (empty ROI)", False
            pixels = [float(v) for r in patch for v in r]

        if not pixels:
            return False, "low ROI confidence (empty ROI)", False
        mean = sum(pixels) / len(pixels)
        var = sum((v - mean) ** 2 for v in pixels) / len(pixels)
        if config.min_roi_variance is not None and var < config.min_roi_variance:
            return False, "low ROI confidence (low variance)", False
        if config.max_saturated_fraction is not None:
            sat = sum(1 for v in pixels if v >= config.saturation_level)
            if sat / len(pixels) > config.max_saturated_fraction:
                return False, "low ROI confidence (saturated)", False
        if config.min_intensity_kurtosis is not None:
            kmin = self._weighted_excess_kurtosis_min(patch)
            if kmin is not None and kmin < config.min_intensity_kurtosis:
                return False, "low ROI confidence (possible multi-peak)", False
        return True, "", False

    def run_step(self, dt_s: float | None = None, frame: CameraFrame | None = None) -> AutofocusSample:
        loop_start = time.monotonic()
        if frame is None:
            frame = self._camera.get_frame()
        current_z = self._stage.get_z_um()

        config = self.get_config_snapshot()

        if dt_s is None:
            dt_s = 1.0 / config.loop_hz
        dt_s = max(0.0, min(float(dt_s), config.max_dt_s))

        with self._state_lock:
            calibration = self._calibration
            if self._z_lock_center_um is None:
                self._z_lock_center_um = float(current_z)
                self._pid.set_lock_center(current_z)
                self._pid.output = current_z

            if self._last_commanded_z_um is not None:
                stage_lag_um = abs(float(current_z) - float(self._last_commanded_z_um))
            else:
                stage_lag_um = 0.0
            if stage_lag_um > config.max_stage_lag_um:
                self._stage_lag_bad_frames += 1
            else:
                self._stage_lag_bad_frames = 0
            if self._stage_lag_bad_frames >= config.stage_lag_fault_frames:
                self._state = AutofocusState.RECOVERY
                return AutofocusSample(
                    frame.timestamp_s, 0.0, 0.0, current_z, current_z, 0.0, False, False,
                    self._state, (time.monotonic() - loop_start) * 1e3, config.roi,
                    f"stage lag high ({stage_lag_um:0.3f} um), pausing control",
                    stage_lag_um=stage_lag_um,
                )

            # Guard: skip duplicate frames (same timestamp as previous).
            if self._last_frame_ts is not None and frame.timestamp_s <= self._last_frame_ts:
                self._state = AutofocusState.DEGRADED
                return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, 0.0, False, False, self._state, (time.monotonic() - loop_start) * 1e3, config.roi, "duplicate frame")
            self._last_frame_ts = frame.timestamp_s

            roi = config.roi
            config.roi = roi
            with self._config_lock:
                self._config.roi = roi

            total_intensity = roi_total_intensity(frame.image, roi)

            fit = None
            if config.use_gaussian_fit:
                fit = fit_gaussian_fast(extract_roi(frame.image, roi), theta=config.gaussian_theta, fast_mode=config.gaussian_fast_mode)
                fit_centroid = None
                if fit is not None and hasattr(fit, "x") and hasattr(fit, "y"):
                    fit_centroid = (float(fit.x), float(fit.y))
                if self._tracking_centroid_ok(fit_centroid, roi, config, fit=fit):
                    roi = self._recenter_roi_if_enabled(frame.image, roi, config, centroid=fit_centroid)
                if roi != config.roi:
                    config.roi = roi
                    with self._config_lock:
                        self._config.roi = roi
                    total_intensity = roi_total_intensity(frame.image, roi)
                    fit = fit_gaussian_fast(extract_roi(frame.image, roi), theta=config.gaussian_theta, fast_mode=config.gaussian_fast_mode)

            if config.edge_margin_px > 0 and centroid_near_edge(frame.image, roi, config.edge_margin_px):
                edge_centroid = None
                if fit is not None:
                    candidate = (float(fit.x), float(fit.y))
                    if self._tracking_centroid_ok(candidate, roi, config, fit=fit):
                        edge_centroid = candidate
                edge_roi = self._recenter_roi_if_enabled(frame.image, roi, config, force=True, centroid=edge_centroid)
                if edge_roi != roi:
                    roi = edge_roi
                    config.roi = roi
                    with self._config_lock:
                        self._config.roi = roi
                    total_intensity = roi_total_intensity(frame.image, roi)
                    if config.use_gaussian_fit:
                        fit = fit_gaussian_fast(extract_roi(frame.image, roi), theta=config.gaussian_theta, fast_mode=config.gaussian_fast_mode)

            baseline_intensity = self._roi_intensity_baseline
            if baseline_intensity is None:
                baseline_intensity = total_intensity
            else:
                alpha = 0.98
                baseline_intensity = alpha * baseline_intensity + (1.0 - alpha) * total_intensity
            self._roi_intensity_baseline = baseline_intensity

            confidence_ok, confidence_reason, low_intensity = self._roi_confidence_ok(frame.image, total_intensity, config, baseline_intensity)
            if config.edge_margin_px > 0 and centroid_near_edge(frame.image, roi, config.edge_margin_px):
                confidence_ok = False
                confidence_reason = "low ROI confidence (near edge)"
            resized = self._adaptive_roi_resize(frame.image, roi, config, confidence_ok=confidence_ok)
            if resized != roi:
                roi = resized
                config.roi = roi
                with self._config_lock:
                    self._config.roi = roi
                total_intensity = roi_total_intensity(frame.image, roi)
                if config.use_gaussian_fit:
                    fit = fit_gaussian_fast(extract_roi(frame.image, roi), theta=config.gaussian_theta, fast_mode=config.gaussian_fast_mode)
                    if fit is not None:
                        fit_fwhm = float(fit.fwhm)
                        fit_ell = float(fit.ellipticity)
                        fit_r2 = float(fit.r_squared)

            if fit is None:
                fit_fwhm = float("nan")
                fit_ell = float("nan")
                fit_r2 = float("nan")
            else:
                fit_fwhm = float(fit.fwhm)
                fit_ell = float(fit.ellipticity)
                fit_r2 = float(fit.r_squared)

            # Keep detection-filter thresholds aligned with config intensity gates.
            self._detection_filter.bounds.intensity_min = float(config.min_roi_intensity or 0.0)
            self._detection_filter.bounds.intensity_min_fraction = float(config.min_roi_intensity_fraction or 0.0)
            detection_trusted = self._detection_filter.check(fit_fwhm, fit_ell, fit_r2, total_intensity)
            if not detection_trusted:
                confidence_ok = False
                if not confidence_reason:
                    confidence_reason = "low ROI confidence (detection window)"

            if not confidence_ok:
                if low_intensity:
                    self._low_intensity_count += 1
                else:
                    self._low_intensity_count = 0
                if self._low_intensity_count >= config.lost_intensity_frames:
                    self._state = AutofocusState.LOST
                    with self._config_lock:
                        self._config.track_roi = False
                    return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, total_intensity, False, False, self._state, (time.monotonic() - loop_start) * 1e3, roi, "target lost: reselect ROI")
                self._state = AutofocusState.RECOVERY if self._detection_filter.pass_fraction <= config.detection_pass_threshold else AutofocusState.DEGRADED
                diag = f"{confidence_reason or 'low ROI confidence'} (pass={self._detection_filter.pass_fraction:.2f})"
                return AutofocusSample(frame.timestamp_s, 0.0, 0.0, current_z, current_z, total_intensity, False, False, self._state, (time.monotonic() - loop_start) * 1e3, roi, diag, stage_lag_um=stage_lag_um)
            self._low_intensity_count = 0

            try:
                error = astigmatic_error_signal(frame.image, roi, background_percentile=config.background_percentile)
            except TypeError:
                error = astigmatic_error_signal(frame.image, roi)
            error = self._filtered_error(float(error), config)
            domain_margin = self._calibration_domain_margin(error, config)
            ell_params = getattr(calibration, "params", None)
            strict_analytic = (
                bool(config.strict_lenstra_match)
                and bool(config.use_gaussian_fit)
                and bool(config.use_analytical_inversion)
                and (fit is not None)
                and (ell_params is not None)
            )
            if (not strict_analytic) and (not self._is_error_in_calibration_domain(error, config)):
                self._state = AutofocusState.RECOVERY if self._detection_filter.pass_fraction <= config.detection_pass_threshold else AutofocusState.DEGRADED
                return AutofocusSample(
                    frame.timestamp_s,
                    error,
                    0.0,
                    current_z,
                    current_z,
                    total_intensity,
                    False,
                    False,
                    self._state,
                    (time.monotonic() - loop_start) * 1e3,
                    roi,
                    "error outside calibration domain",
                    domain_margin=domain_margin,
                    stage_lag_um=stage_lag_um,
                )

            focus_error = self._calibration_error_at_focus()
            if config.lock_setpoint:
                if self._setpoint_error is None:
                    self._setpoint_error = error - focus_error
                error -= self._setpoint_error
            elif config.recenter_alpha > 0:
                error_rel_focus = error - focus_error
                if self._setpoint_error is None:
                    # Track setpoint bias in the same focus-referenced domain as
                    # lock_setpoint so the first frame does not command an
                    # immediate move away from the calibrated focus.
                    self._setpoint_error = error_rel_focus
                self._setpoint_error = (
                    config.recenter_alpha * self._setpoint_error
                    + (1.0 - config.recenter_alpha) * error_rel_focus
                )
                error -= self._setpoint_error
            else:
                self._setpoint_error = None

            if config.use_gaussian_fit and fit is not None and config.use_analytical_inversion:
                if ell_params is not None:
                    lookup_offset_um = calibration.error_to_z_offset_um(error)
                    z_model = findz_analytical(float(fit.ellipticity), ell_params)
                    if math.isfinite(z_model):
                        z0 = getattr(ell_params, "z0", 0.0)
                        # findz_analytical() returns the modeled Z in the local
                        # calibration frame. Convert that to offset from the
                        # model focus center without an extra sign flip.
                        z_offset_um = float(z_model - z0)
                        # Guard against branch/sign mistakes in analytical inversion:
                        # if the analytical estimate disagrees strongly with the
                        # empirical calibration lookup for the same frame, trust
                        # the lookup for control.
                        if (not strict_analytic) and math.isfinite(float(lookup_offset_um)):
                            mismatch_um = abs(z_offset_um - float(lookup_offset_um))
                            mismatch_limit = max(0.25, 4.0 * float(config.max_step_um))
                            if mismatch_um > mismatch_limit:
                                z_offset_um = float(lookup_offset_um)
                        if config.lock_setpoint:
                            if self._setpoint_z_offset_um is None:
                                self._setpoint_z_offset_um = z_offset_um
                            error_um = z_offset_um - self._setpoint_z_offset_um
                        elif config.recenter_alpha > 0:
                            if self._setpoint_z_offset_um is None:
                                self._setpoint_z_offset_um = z_offset_um
                            self._setpoint_z_offset_um = (
                                config.recenter_alpha * self._setpoint_z_offset_um
                                + (1.0 - config.recenter_alpha) * z_offset_um
                            )
                            error_um = z_offset_um - self._setpoint_z_offset_um
                        else:
                            self._setpoint_z_offset_um = None
                            error_um = z_offset_um
                    else:
                        self._state = AutofocusState.DEGRADED
                        return AutofocusSample(frame.timestamp_s, float(fit.ellipticity), 0.0, current_z, current_z, total_intensity, False, False, self._state, (time.monotonic() - loop_start) * 1e3, roi, "ellipticity outside Zhuang model range", ellipticity=float(fit.ellipticity), fwhm_px=float(fit.fwhm), fit_r_squared=float(fit.r_squared), detection_pass_rate=self._detection_filter.pass_fraction, roi_x=roi.x, roi_y=roi.y, domain_margin=domain_margin, stage_lag_um=stage_lag_um)
                else:
                    self._setpoint_z_offset_um = None
                    error_um = calibration.error_to_z_offset_um(error)
            else:
                self._setpoint_z_offset_um = None
                error_um = calibration.error_to_z_offset_um(error)

            if fit is not None:
                if self._ell_ema is None:
                    self._ell_ema = float(fit.ellipticity)
                else:
                    a = config.ellipticity_ema_alpha
                    self._ell_ema = a * float(fit.ellipticity) + (1.0 - a) * self._ell_ema

            if not math.isfinite(float(error_um)):
                self._state = AutofocusState.FAULT
                raise RuntimeError("Non-finite autofocus error encountered; check ROI/calibration")

            can_control, engage_scale = self._safe_engage_scale(config, confidence_ok=True)
            if not can_control:
                self._state = AutofocusState.LOCKING
                return AutofocusSample(
                    frame.timestamp_s, error, 0.0, current_z, current_z, total_intensity,
                    False, True, self._state, (time.monotonic() - loop_start) * 1e3, roi,
                    f"safe engage hold {self._engage_good_frames}/{config.engage_hold_good_frames}",
                    ellipticity=(float(self._ell_ema) if self._ell_ema is not None else 0.0),
                    fwhm_px=(float(fit.fwhm) if fit is not None else 0.0),
                    fit_r_squared=(float(fit.r_squared) if fit is not None else 0.0),
                    detection_pass_rate=self._detection_filter.pass_fraction,
                    roi_x=roi.x,
                    roi_y=roi.y,
                    domain_margin=domain_margin,
                    stage_lag_um=stage_lag_um,
                )

            conf_scale = self._confidence_scale(
                fit_r2=fit_r2,
                total_intensity=total_intensity,
                baseline_intensity=baseline_intensity,
                detection_pass_rate=self._detection_filter.pass_fraction,
                config=config,
            )
            control_scale = conf_scale * engage_scale
            error_um_effective = float(error_um) * control_scale
            self._recent_error_um.append(error_um_effective)
            dynamic_deadband = self._dynamic_deadband(config)
            correction, applied = self._pid.update(
                error_um_effective,
                dt_s,
                deadband_um=dynamic_deadband,
            )
            if not applied:
                self._state = AutofocusState.LOCKED
                return AutofocusSample(frame.timestamp_s, error, error_um_effective, current_z, self._pid.output, total_intensity, False, True, self._state, (time.monotonic() - loop_start) * 1e3, roi, "within deadband", ellipticity=(float(self._ell_ema) if self._ell_ema is not None else 0.0), fwhm_px=(float(fit.fwhm) if fit is not None else 0.0), fit_r_squared=(float(fit.r_squared) if fit is not None else 0.0), detection_pass_rate=self._detection_filter.pass_fraction, roi_x=roi.x, roi_y=roi.y, domain_margin=domain_margin, confidence_scale=control_scale, stage_lag_um=stage_lag_um)

            commanded_z_to_apply = self._pid.output
            error_to_report = error
            error_um_to_report = error_um_effective
            total_intensity_to_report = total_intensity

            # Anti-runaway guards: stop and re-lock if corrections appear to
            # push in the wrong direction for several consecutive frames.
            if correction * error_um_effective > 0:
                self._sign_mismatch_count += 1
            else:
                self._sign_mismatch_count = 0
            abs_err = abs(error_um_effective)
            if self._last_abs_error_um is not None and abs_err > (self._last_abs_error_um + max(0.002, dynamic_deadband)):
                self._growth_count += 1
            else:
                self._growth_count = 0
            self._last_abs_error_um = abs_err
            if self._sign_mismatch_count >= config.runaway_sign_frames or self._growth_count >= config.runaway_growth_frames:
                self.reset_lock_state()
                self._state = AutofocusState.RECOVERY
                return AutofocusSample(
                    frame.timestamp_s,
                    error_to_report,
                    0.0,
                    current_z,
                    current_z,
                    total_intensity_to_report,
                    False,
                    False,
                    self._state,
                    (time.monotonic() - loop_start) * 1e3,
                    roi,
                    "runaway guard triggered; control frozen, re-lock and recalibrate",
                    ellipticity=(float(self._ell_ema) if self._ell_ema is not None else 0.0),
                    fwhm_px=(float(fit.fwhm) if fit is not None else 0.0),
                    fit_r_squared=(float(fit.r_squared) if fit is not None else 0.0),
                    detection_pass_rate=self._detection_filter.pass_fraction,
                    roi_x=roi.x,
                    roi_y=roi.y,
                    domain_margin=domain_margin,
                    confidence_scale=control_scale,
                    stage_lag_um=stage_lag_um,
                )

        self._stage.move_z_um(commanded_z_to_apply)
        self._last_commanded_z_um = commanded_z_to_apply

        with self._state_lock:
            self._state = AutofocusState.LOCKING if abs(error_um_to_report) > config.command_deadband_um else AutofocusState.LOCKED
            return AutofocusSample(
                timestamp_s=frame.timestamp_s,
                error=error_to_report,
                error_um=error_um_to_report,
                stage_z_um=current_z,
                commanded_z_um=commanded_z_to_apply,
                roi_total_intensity=total_intensity_to_report,
                control_applied=True,
                confidence_ok=True,
                state=self._state,
                loop_latency_ms=(time.monotonic() - loop_start) * 1e3,
                roi=roi,
                diagnostic="control update applied",
                ellipticity=(float(self._ell_ema) if self._ell_ema is not None else 0.0),
                fwhm_px=(float(fit.fwhm) if fit is not None else 0.0),
                fit_r_squared=(float(fit.r_squared) if fit is not None else 0.0),
                detection_pass_rate=self._detection_filter.pass_fraction,
                roi_x=roi.x,
                roi_y=roi.y,
                domain_margin=domain_margin,
                confidence_scale=control_scale,
                stage_lag_um=stage_lag_um,
            )

    def run(self, duration_s: float) -> list[AutofocusSample]:
        samples: list[AutofocusSample] = []
        loop_dt = 1.0 / self.loop_hz
        end = time.monotonic() + duration_s
        last_step_start: float | None = None
        while time.monotonic() < end:
            step_start = time.monotonic()
            dt_s = loop_dt if last_step_start is None else max(0.0, step_start - last_step_start)
            samples.append(self.run_step(dt_s=dt_s))
            last_step_start = step_start
            elapsed = time.monotonic() - step_start
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)
        return samples


class AutofocusWorker:
    """Background real-time autofocus worker."""

    def __init__(
        self,
        controller: AstigmaticAutofocusController,
        on_sample: Callable[[AutofocusSample], None] | None = None,
    ) -> None:
        self._controller = controller
        self._on_sample = on_sample
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_error: Exception | None = None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_evt.clear()
            self._last_error = None
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        self._stop_evt.set()
        if not wait:
            return
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        nominal_dt = 1.0 / self._controller.loop_hz
        last_step_start: float | None = None
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            dt = nominal_dt if last_step_start is None else max(0.0, t0 - last_step_start)
            try:
                sample = self._controller.run_step(dt_s=dt)
                if self._on_sample is not None:
                    self._on_sample(sample)
            except Exception as exc:  # pragma: no cover - exercised by tests indirectly
                self._last_error = exc
                self._stop_evt.set()
                return
            last_step_start = t0
            elapsed = time.monotonic() - t0
            if elapsed < nominal_dt:
                time.sleep(nominal_dt - elapsed)

from __future__ import annotations


from dataclasses import dataclass

from auto_focus.autofocus import AstigmaticAutofocusController, AutofocusConfig, AutofocusState, AutofocusWorker, DetectionBounds, DetectionFilter
from auto_focus.calibration import FocusCalibration
from auto_focus.focus_metric import Roi
from auto_focus.hardware import HamamatsuOrcaCamera
from auto_focus.interfaces import CameraFrame
from auto_focus.calibration import ZhuangFocusCalibration
from auto_focus.zhuang import ZhuangParams, ZhuangEllipticityParams, ZhuangLookupTable


class DummyCamera:
    def __init__(self) -> None:
        self.frame = CameraFrame(image=[[1.0]], timestamp_s=1.0)

    def get_frame(self) -> CameraFrame:
        return self.frame


class DummyStage:
    def __init__(self) -> None:
        self.z = 0.0
        self.moves: list[float] = []

    def get_z_um(self) -> float:
        return self.z

    def move_z_um(self, target_z_um: float) -> None:
        self.moves.append(target_z_um)
        self.z = target_z_um


@dataclass(slots=True)
class DummyLookup:
    error_values: list[float]


@dataclass(slots=True)
class DummyZhuangCalibration:
    lookup: DummyLookup

    def error_to_z_offset_um(self, error: float) -> float:
        return error


def _config() -> AutofocusConfig:
    return AutofocusConfig(roi=Roi(x=0, y=0, width=1, height=1), kp=1.0, ki=0.0, kd=0.0, max_step_um=1.0)


def test_out_of_range_linear_calibration_skips_control(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.8)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED
    assert stage.moves == []


def test_out_of_range_lookup_calibration_skips_control(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.9)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    # exercise lookup fallback path used by Zhuang-style calibrations
    cal = DummyZhuangCalibration(lookup=DummyLookup(error_values=[-0.3, 0.3]))
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED
    assert stage.moves == []


def test_lock_setpoint_initializes_roi_bias_from_calibration_focus(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.12)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    # At lock engagement, controller should treat current ROI bias as setpoint
    # offset so it does not command an immediate move.
    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED
    assert stage.moves == []


def test_lock_setpoint_applies_only_delta_from_initial_roi_bias(monkeypatch):
    signal = {"value": 0.12}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    first = controller.run_step()
    assert first.control_applied is False

    signal["value"] = 0.16  # +0.04 vs engagement baseline
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    second = controller.run_step()

    assert second.control_applied is True
    assert stage.moves, "expected correction move"
    assert stage.moves[-1] < 0.0



def test_out_of_range_sets_diagnostic(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.95)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.2, error_max=0.2)
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), _config(), cal)

    sample = controller.run_step()

    assert sample.diagnostic == "error outside calibration domain"


def test_transition_lock_disabled_recenter_enabled(monkeypatch):
    signal = {"value": 0.10}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    controller.run_step()  # initializes lock setpoint
    controller.update_config(lock_setpoint=False, recenter_alpha=0.5)
    controller.reset_lock_state()
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    signal["value"] = 0.18

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED


def test_roi_change_resets_setpoint_memory(monkeypatch):
    signal = {"value": 0.12}

    def _err(_img, _roi):
        return signal["value"]

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = FocusCalibration(error_at_focus=0.05, error_to_um=1.0, error_min=-0.3, error_max=0.3)
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    controller.run_step()
    signal["value"] = 0.20
    controller.update_roi(Roi(x=0, y=0, width=1, height=1))
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)

    sample = controller.run_step()

    # Fresh ROI lock should not apply an immediate move.
    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED


def test_analytical_zhuang_lock_setpoint_does_not_jump_to_calibration_focus(monkeypatch):
    class DummyFit:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.fwhm = 2.0
            self.ellipticity = 1.2
            self.r_squared = 0.99

    class DummyZhuangCal:
        def __init__(self) -> None:
            self.params = type("P", (), {"z0": 0.0})()
            self.error_at_focus = 0.0

        def error_to_z_offset_um(self, error: float) -> float:
            return error

        def is_error_in_range(self, error: float, margin: float = 0.0) -> bool:
            return True

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.2)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_a, **_k: DummyFit())
    monkeypatch.setattr("auto_focus.autofocus.findz_analytical", lambda _ell, _params: 0.25)

    cfg = _config()
    cfg.use_gaussian_fit = True
    cfg.use_analytical_inversion = True
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, cfg, DummyZhuangCal())

    sample = controller.run_step()

    # Lock-setpoint should treat the first observed model Z offset as the target,
    # not command an immediate move toward the calibration focus (z0).
    assert sample.control_applied is False
    assert sample.state == AutofocusState.LOCKED
    assert stage.moves == []


def test_analytical_zhuang_correction_uses_findz_sign_directly(monkeypatch):
    class DummyFit:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.fwhm = 2.0
            self.ellipticity = 1.2
            self.r_squared = 0.99

    class DummyZhuangCal:
        def __init__(self) -> None:
            self.params = type("P", (), {"z0": 0.0})()
            self.error_at_focus = 0.0

        def error_to_z_offset_um(self, error: float) -> float:
            return error

        def is_error_in_range(self, error: float, margin: float = 0.0) -> bool:
            return True

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.0)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_a, **_k: DummyFit())

    z_model = {"value": 0.25}

    def _findz(_ell, _params):
        return z_model["value"]

    monkeypatch.setattr("auto_focus.autofocus.findz_analytical", _findz)

    cfg = _config()
    cfg.use_gaussian_fit = True
    cfg.use_analytical_inversion = True
    cfg.lock_setpoint = True
    cfg.kp = 1.0
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, cfg, DummyZhuangCal())

    first = controller.run_step()
    assert first.control_applied is False

    z_model["value"] = 0.35
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    second = controller.run_step()

    assert second.control_applied is True
    assert stage.moves[-1] < 0.0


def test_strict_analytical_bypasses_error_domain_gate(monkeypatch):
    class DummyFit:
        x = 0.0
        y = 0.0
        fwhm = 2.0
        ellipticity = 1.1
        r_squared = 0.99

    class DummyZhuangCal:
        def __init__(self) -> None:
            self.params = type("P", (), {"z0": 0.0})()
            self.lookup = DummyLookup(error_values=[-0.3, 0.3])
            self.error_at_focus = 0.0

        def error_to_z_offset_um(self, _error: float) -> float:
            return 0.0

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.9)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_a, **_k: DummyFit())
    monkeypatch.setattr("auto_focus.autofocus.findz_analytical", lambda _ell, _params: 0.2)

    cfg = _config()
    cfg.use_gaussian_fit = True
    cfg.use_analytical_inversion = True
    cfg.strict_analytical_match = True

    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, DummyZhuangCal())
    sample = controller.run_step()

    assert sample.state != AutofocusState.DEGRADED
    assert sample.diagnostic != "error outside calibration domain"


def test_non_strict_mode_keeps_domain_guard_for_analytical_path(monkeypatch):
    class DummyFit:
        x = 0.0
        y = 0.0
        fwhm = 2.0
        ellipticity = 1.1
        r_squared = 0.99

    class DummyZhuangCal:
        def __init__(self) -> None:
            self.params = type("P", (), {"z0": 0.0})()
            self.lookup = DummyLookup(error_values=[-0.3, 0.3])
            self.error_at_focus = 0.0

        def error_to_z_offset_um(self, _error: float) -> float:
            return 0.0

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.9)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_a, **_k: DummyFit())
    monkeypatch.setattr("auto_focus.autofocus.findz_analytical", lambda _ell, _params: 0.2)

    cfg = _config()
    cfg.use_gaussian_fit = True
    cfg.use_analytical_inversion = True
    cfg.strict_analytical_match = False

    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, DummyZhuangCal())
    sample = controller.run_step()

    assert sample.state == AutofocusState.DEGRADED
    assert sample.diagnostic == "error outside calibration domain"


def test_lookup_domain_guard_handles_unsorted_error_values(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.6)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 10.0)

    cal = DummyZhuangCalibration(lookup=DummyLookup(error_values=[0.3, -0.3]))
    stage = DummyStage()
    controller = AstigmaticAutofocusController(DummyCamera(), stage, _config(), cal)

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state == AutofocusState.DEGRADED


def test_worker_uses_elapsed_dt(monkeypatch):
    class StubController:
        loop_hz = 10.0

        def __init__(self):
            self.dt_values: list[float] = []

        def run_step(self, dt_s=None):
            self.dt_values.append(float(dt_s))
            if len(self.dt_values) >= 2:
                raise RuntimeError("stop")
            return None

    clock = {"t": 100.0}

    def fake_monotonic():
        val = clock["t"]
        clock["t"] += 0.15
        return val

    monkeypatch.setattr("auto_focus.autofocus.time.monotonic", fake_monotonic)
    monkeypatch.setattr("auto_focus.autofocus.time.sleep", lambda _s: None)

    controller = StubController()
    worker = AutofocusWorker(controller)
    worker._run_loop()

    assert len(controller.dt_values) == 2
    assert controller.dt_values[0] == 0.1
    assert abs(controller.dt_values[1] - 0.3) < 1e-9


def test_hamamatsu_orca_requires_frame_source():
    try:
        HamamatsuOrcaCamera(frame_source=None)
    except ValueError as exc:
        assert "frame_source" in str(exc)
    else:
        raise AssertionError("expected ValueError when frame_source is None")


def test_zhuang_from_axis_params_uses_sigma_ratio_for_e0():
    px = ZhuangParams(sigma0=4.0, A=0.0, B=0.0, c=0.0, d=1.0)
    py = ZhuangParams(sigma0=1.0, A=0.0, B=0.0, c=0.0, d=1.0)

    q = ZhuangEllipticityParams.from_axis_params(px, py)

    assert q.e0 == 4.0


def test_zhuang_calibration_has_range_checker():
    lookup = ZhuangLookupTable(error_values=[-0.2, 0.2], z_values_um=[-1.0, 1.0], z_range=(-1.0, 1.0))
    cal = ZhuangFocusCalibration(
        params=ZhuangEllipticityParams(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        params_x=ZhuangParams(1.0, 0.0, 0.0, 0.0, 1.0),
        params_y=ZhuangParams(1.0, 0.0, 0.0, 0.0, 1.0),
        lookup=lookup,
    )

    assert cal.is_error_in_range(0.0)
    assert not cal.is_error_in_range(0.5)

def test_run_step_recenters_roi_before_error(monkeypatch):
    seen_rois: list[Roi] = []

    def _err(_img, roi):
        seen_rois.append(roi)
        return 0.0

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 50.0)
    class _GoodFit:
        x = 5.0
        y = 3.0
        fwhm = 2.0
        ellipticity = 1.0
        r_squared = 0.99
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_args, **_kwargs: _GoodFit())

    image = [[0.0 for _x in range(12)] for _y in range(12)]
    image[3][5] = 100.0
    cam = DummyCamera()
    cam.frame = CameraFrame(image=image, timestamp_s=2.0)

    cfg = AutofocusConfig(
        roi=Roi(x=0, y=0, width=6, height=6),
        kp=1.0,
        ki=0.0,
        kd=0.0,
        max_step_um=1.0,
        track_roi=True,
        track_gain=1.0,
        track_deadband_px=0.0,
        track_max_shift_px=3.0,
    )
    controller = AstigmaticAutofocusController(cam, DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    sample = controller.run_step()

    assert seen_rois, "error metric should be evaluated"
    assert seen_rois[-1].x > 0
    assert sample.roi.x == seen_rois[-1].x
    assert controller.get_config_snapshot().roi.x == sample.roi.x


def test_roi_tracking_ignores_bad_gaussian_fit_centroid(monkeypatch):
    seen_rois: list[Roi] = []

    def _err(_img, roi):
        seen_rois.append(roi)
        return 0.0

    class _BadFit:
        x = 5.0
        y = 3.0
        fwhm = 100.0       # implausible, should fail tracking guard
        ellipticity = 1.0
        r_squared = 0.99

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 50.0)
    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_args, **_kwargs: _BadFit())

    image = [[0.0 for _x in range(12)] for _y in range(12)]
    image[3][5] = 100.0
    cam = DummyCamera()
    cam.frame = CameraFrame(image=image, timestamp_s=2.0)

    cfg = AutofocusConfig(
        roi=Roi(x=0, y=0, width=6, height=6),
        kp=1.0,
        ki=0.0,
        kd=0.0,
        max_step_um=1.0,
        track_roi=True,
        track_gain=1.0,
        track_deadband_px=0.0,
        track_max_shift_px=3.0,
        use_gaussian_fit=True,
        detection_bounds=DetectionBounds(fwhm_min_px=1.0, fwhm_max_px=20.0, ellipticity_min=0.5, ellipticity_max=2.0, r_squared_min=0.0),
    )
    controller = AstigmaticAutofocusController(
        cam,
        DummyStage(),
        cfg,
        FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5),
    )

    sample = controller.run_step()

    # With an implausible fit, tracking should not move the ROI and the frame may
    # be rejected before the astigmatic error metric is evaluated.
    if seen_rois:
        assert seen_rois[-1].x == 0
    assert sample.roi.x == 0


def test_min_roi_intensity_fraction_uses_running_baseline(monkeypatch):
    intensity = {"value": 100.0}

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda _img, _roi: 0.0)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: intensity["value"])

    cfg = _config()
    cfg.min_roi_intensity_fraction = 0.6
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    first = controller.run_step()
    assert first.confidence_ok is True

    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    intensity["value"] = 40.0
    second = controller.run_step()

    assert second.control_applied is False
    assert second.confidence_ok is False
    assert second.state == AutofocusState.DEGRADED


def test_edge_margin_forces_recentering_before_drop(monkeypatch):
    seen_rois: list[Roi] = []

    def _err(_img, roi, **_kwargs):
        seen_rois.append(roi)
        return 0.0

    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", _err)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 100.0)

    image = [[0.0 for _x in range(12)] for _y in range(12)]
    image[4][2] = 200.0  # near left edge of initial ROI
    cam = DummyCamera()
    cam.frame = CameraFrame(image=image, timestamp_s=2.0)

    cfg = _config()
    cfg.roi = Roi(x=2, y=0, width=8, height=8)
    cfg.track_roi = True
    cfg.track_gain = 1.0
    cfg.track_deadband_px = 0.0
    cfg.track_max_shift_px = 4.0
    cfg.edge_margin_px = 2.0

    controller = AstigmaticAutofocusController(cam, DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))
    sample = controller.run_step()

    assert sample.roi.x < 2
    assert sample.diagnostic in {"control update applied", "within deadband", "low ROI confidence (near edge)"}


def test_low_intensity_transitions_to_lost(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda *_args, **_kwargs: 0.0)
    intensity = {"value": 10.0}
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: intensity["value"])

    cfg = _config()
    cfg.min_roi_intensity_fraction = 0.9
    cfg.lost_intensity_frames = 2
    cfg.track_roi = True
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    first = controller.run_step()
    intensity["value"] = 1.0
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=2.0)
    second = controller.run_step()
    controller._camera.frame = CameraFrame(image=[[1.0]], timestamp_s=3.0)
    third = controller.run_step()

    assert first.state == AutofocusState.LOCKED
    assert second.state == AutofocusState.DEGRADED
    assert third.state == AutofocusState.LOST
    assert "target lost" in third.diagnostic
    assert controller.get_config_snapshot().track_roi is False


def test_detection_filter_sliding_window_behavior():
    filt = DetectionFilter(window_size=4, pass_threshold=0.5, bounds=DetectionBounds(
        fwhm_min_px=1.0, fwhm_max_px=10.0, ellipticity_min=0.5, ellipticity_max=2.0, r_squared_min=0.0
    ))

    assert filt.check(2.0, 1.0, 0.9, 100.0) is True
    assert filt.check(2.0, 1.0, 0.9, 100.0) is True
    assert filt.check(50.0, 1.0, 0.9, 100.0) is False
    assert 0.0 <= filt.pass_fraction <= 1.0


def test_controller_uses_detection_window_for_degraded_state(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda _img, _roi: 100.0)
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda *_args, **_kwargs: 0.0)

    class _BadFit:
        fwhm = 100.0
        ellipticity = 10.0
        r_squared = -1.0

    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_args, **_kwargs: _BadFit())

    cfg = _config()
    cfg.detection_window_size = 3
    cfg.detection_pass_threshold = 0.6
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    sample = controller.run_step()

    assert sample.control_applied is False
    assert sample.state in {AutofocusState.DEGRADED, AutofocusState.RECOVERY}
    assert "pass=" in sample.diagnostic


def test_run_step_without_gaussian_fit_keeps_fit_diagnostics_safe(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda *_args, **_kwargs: 10.0)

    cfg = _config()
    cfg.use_gaussian_fit = False
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    sample = controller.run_step()

    assert sample.state in {AutofocusState.LOCKED, AutofocusState.LOCKING}
    assert sample.ellipticity == 0.0
    assert sample.fwhm_px == 0.0
    assert sample.fit_r_squared == 0.0


def test_run_step_calls_detection_filter_check_once_per_frame(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda *_args, **_kwargs: 10.0)

    class _Fit:
        x = 0.0
        y = 0.0
        fwhm = 3.0
        ellipticity = 1.0
        r_squared = 0.9

    monkeypatch.setattr("auto_focus.autofocus.fit_gaussian_fast", lambda *_args, **_kwargs: _Fit())

    cfg = _config()
    cfg.use_gaussian_fit = True
    controller = AstigmaticAutofocusController(DummyCamera(), DummyStage(), cfg, FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    calls = {"n": 0}
    original_check = controller._detection_filter.check

    def _check(*args, **kwargs):
        calls["n"] += 1
        return original_check(*args, **kwargs)

    controller._detection_filter.check = _check
    controller.run_step()

    assert calls["n"] == 1


def test_run_step_first_pid_correction_does_not_touch_legacy_slew_state(monkeypatch):
    monkeypatch.setattr("auto_focus.autofocus.astigmatic_error_signal", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr("auto_focus.autofocus.roi_total_intensity", lambda *_args, **_kwargs: 10.0)

    cfg = _config()
    cfg.use_gaussian_fit = False
    cfg.command_deadband_um = 0.0
    cfg.lock_setpoint = False
    stage = DummyStage()
    controller = AstigmaticAutofocusController(stage=stage, camera=DummyCamera(), config=cfg, calibration=FocusCalibration(error_at_focus=0.0, error_to_um=1.0, error_min=-0.5, error_max=0.5))

    sample = controller.run_step()

    assert sample.control_applied is True
    assert stage.moves, "expected at least one stage move"

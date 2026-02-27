from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .autofocus import AstigmaticAutofocusController, AutofocusConfig, CalibrationLike, DetectionBounds
from .calibration import (
    FocusCalibration,
    calibration_quality_issues,
    fit_calibration_model,
    fit_linear_calibration_with_report,
    fit_zhuang_calibration,
    load_calibration_samples_csv,
    load_zhuang_calibration_samples_csv,
    validate_calibration_sign,
)
from .focus_metric import Roi
from .hardware import HamamatsuOrcaCamera, MclNanoZStage, NotConnectedError, SimulatedCamera
from .interfaces import StageInterface
from .interactive import launch_autofocus_viewer
from .pylablib_camera import create_pylablib_frame_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run astigmatic autofocus loop")
    parser.add_argument("--duration", type=float, default=2.0, help="Loop runtime in seconds")
    parser.add_argument("--loop-hz", type=float, default=30.0, help="Control loop frequency")
    parser.add_argument("--max-dt-s", type=float, default=0.2, help="Maximum effective control dt per step (s) to limit jump after stalls")
    parser.add_argument(
        "--show-live",
        action="store_true",
        help="Open interactive live image viewer with ROI selection",
    )
    parser.add_argument(
        "--camera",
        choices=["simulate", "orca", "andor", "micromanager"],
        default="simulate",
        help="Camera backend selection",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera index (DCAM idx / Andor camera_index) for pylablib backend"
    )
    parser.add_argument("--mm-host", default="localhost", help="Micro-Manager pycromanager host")
    parser.add_argument("--mm-port", type=int, default=4827, help="Micro-Manager pycromanager port")
    parser.add_argument(
        "--mm-allow-standalone-core",
        action="store_true",
        help=(
            "Allow fallback to standalone pymmcore/MMCorePy CMMCore if bridge attach fails. "
            "Use only when intentionally running without attaching to MM GUI."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["mcl", "simulate", "micromanager"],
        default=None,
        help=(
            "Stage backend. Defaults: simulate camera -> simulate stage, "
            "all hardware cameras (including micromanager) -> mcl stage."
        ),
    )
    parser.add_argument("--stage-dll", default=None, help="Path to MCL stage DLL")
    parser.add_argument("--stage-axis", type=int, default=3, help="MCL axis index for Z moves (commonly 1 or 3 depending on controller/wrapper)")
    parser.add_argument(
        "--stage-wrapper", default=None, help="Python module path for MCL wrapper"
    )
    parser.add_argument("--kp", type=float, default=0.8, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.2, help="Integral gain")
    parser.add_argument("--max-step", type=float, default=0.2, help="Max correction step in um")
    parser.add_argument("--command-deadband-um", type=float, default=0.005, help="Ignore stage corrections smaller than this magnitude (um) to reduce oscillation")
    parser.add_argument("--stage-min-um", type=float, default=None, help="Lower clamp for commanded stage Z (um)")
    parser.add_argument("--stage-max-um", type=float, default=None, help="Upper clamp for commanded stage Z (um)")
    parser.add_argument("--af-max-excursion-um", type=float, default=5.0, help="Max allowed autofocus excursion from initial Z lock point (um); set negative to disable")
    parser.add_argument("--track-roi-xy", action="store_true", help="Enable ROI XY tracking in control loop")
    parser.add_argument("--track-gain", type=float, default=0.4, help="ROI tracking gain")
    parser.add_argument("--track-deadband-px", type=float, default=1.5, help="ROI tracking deadband (pixels)")
    parser.add_argument("--track-max-step-px", type=float, default=8.0, help="Max ROI tracking shift per frame (pixels)")
    parser.add_argument("--min-roi-intensity-fraction", type=float, default=None, help="Pause corrections when ROI intensity falls below this fraction of running baseline")
    parser.add_argument("--background-percentile", type=float, default=20.0, help="Background percentile for astigmatic error signal subtraction")
    parser.add_argument("--use-gaussian-fit", action="store_true", default=True,
        help="Use per-frame Gaussian fitting (default for ParS tracking)")
    parser.add_argument("--no-gaussian-fit", dest="use_gaussian_fit", action="store_false",
        help="Use second-moment error signal (original mode)")
    parser.add_argument("--fast-mode", action="store_true",
        help="Use moment-only Gaussian (faster, less accurate)")
    parser.add_argument("--use-analytical-inversion", action="store_true", default=True,
        help="Use analytical Zhuang Z inversion")
    parser.add_argument("--track-roi", action="store_true", default=True,
        help="Enable automatic ROI re-centering on target")
    parser.add_argument("--no-track-roi", dest="track_roi", action="store_false",
        help="Disable ROI tracking (fixed ROI, original bead mode)")
    parser.add_argument("--detection-window", type=int, default=5,
        help="Sliding window size for detection filter")
    parser.add_argument("--detection-threshold", type=float, default=0.35,
        help="Minimum pass fraction to trust measurement")
    parser.add_argument(
        "--calibration-csv",
        default="calibration_sweep.csv",
        help=(
            "Calibration samples CSV path. If present, it is loaded on startup to build the "
            "focus calibration automatically; GUI calibration sweeps also write to this path."
        ),
    )
    parser.add_argument("--allow-missing-calibration", action="store_true", help="Allow runtime with default calibration when no calibration file exists")
    parser.add_argument("--calibration-model", choices=["zhuang", "linear", "poly2", "piecewise"], default="zhuang", help="Calibration fit model (zhuang preferred by default)")
    parser.add_argument(
        "--calibration-expected-slope",
        choices=["auto", "positive", "negative"],
        default="auto",
        help="Expected sign of calibration slope. Use auto to disable sign warnings for unknown optical orientation.",
    )
    parser.add_argument(
        "--calibration-half-range-um",
        type=float,
        default=0.75,
        help="Half-range around current Z for GUI calibration sweep",
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=21,
        help="Number of Z points for GUI calibration sweep",
    )
    return parser


def _load_startup_calibration(
    samples_csv: str | None,
    *,
    model: str = "linear",
    expected_slope: str = "auto",
) -> CalibrationLike:
    if not samples_csv:
        raise ValueError(
            "Calibration CSV path is required. Provide --calibration-csv to reuse a saved sweep."
        )

    csv_path = Path(samples_csv)
    if not csv_path.exists():
        raise ValueError(
            f"Calibration CSV not found: {csv_path}. Run GUI calibration first to generate it."
        )

    # Try loading as Zhuang-format CSV (has sigma_x, sigma_y columns)
    zhuang_loaded = False
    has_gaussian = False
    try:
        zhuang_samples = load_zhuang_calibration_samples_csv(csv_path)
        has_gaussian = any(
            math.isfinite(s.sigma_x) for s in zhuang_samples
        )
        if has_gaussian:
            zhuang_report = fit_zhuang_calibration(zhuang_samples)
            if (zhuang_report.r2_x > 0.5 and zhuang_report.r2_y > 0.5
                    and zhuang_report.usable_range_um is not None):
                calibration = zhuang_report.calibration
                rng = zhuang_report.usable_range_um
                print(
                    f"Loaded Zhuang calibration ({csv_path}): "
                    f"R^2(sx)={zhuang_report.r2_x:.3f}, R^2(sy)={zhuang_report.r2_y:.3f}, "
                    f"range=[{rng[0]:+.3f}, {rng[1]:+.3f}] um, "
                    f"linear_slope~={zhuang_report.linear_slope:+.4f} um/error, "
                    f"good_fits={zhuang_report.n_good_fits}/{zhuang_report.n_samples}",
                    file=sys.stderr,
                )
                zhuang_loaded = True
            else:
                print(
                    f"Warning: Zhuang fit quality too low "
                    f"(R^2_x={zhuang_report.r2_x:.3f}, R^2_y={zhuang_report.r2_y:.3f}, "
                    f"usable_range={zhuang_report.usable_range_um}, "
                    f"good_fits={zhuang_report.n_good_fits}/{zhuang_report.n_samples}); "
                    f"falling back to linear.",
                    file=sys.stderr,
                )
        else:
            print(
                f"Warning: CSV has no valid Gaussian PSF data (sigma_x/sigma_y columns missing or all NaN); "
                f"cannot use Zhuang model. Re-run calibration from the GUI to generate Zhuang-compatible data.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"Warning: Zhuang calibration failed ({exc}), trying linear.", file=sys.stderr)

    if not zhuang_loaded:
        # Fall back to linear calibration
        samples = load_calibration_samples_csv(csv_path)
        fallback_model = "linear" if model == "zhuang" else model
        report = fit_calibration_model(samples, model=fallback_model, robust=True)
        calibration = report.calibration
        print(
            "Loaded linear calibration "
            f"({csv_path}): slope={calibration.error_to_um:+0.4f} um/error, "
            f"error_at_focus={calibration.error_at_focus:+0.4f}, "
            f"R^2={report.r2:0.4f}, inliers={report.n_inliers}/{report.n_samples}",
            file=sys.stderr,
        )

        issues = calibration_quality_issues(samples, report)
        if issues:
            raise ValueError(
                "Calibration CSV failed quality checks: " + " ; ".join(issues)
            )

        if report.r2 < 0.9:
            print(
                f"Warning: calibration R^2={report.r2:0.4f} is below 0.9; "
                "fit quality is poor. Consider re-running the calibration sweep "
                "with a smaller Z range or checking that the fiducial is in the ROI.",
                file=sys.stderr,
            )

        if expected_slope in {"positive", "negative"}:
            try:
                validate_calibration_sign(
                    calibration,
                    expected_positive_slope=(expected_slope == "positive"),
                )
            except ValueError as sign_err:
                print(f"Warning: {sign_err}", file=sys.stderr)
                print(
                    "Calibration slope sign does not match requested expectation. "
                    "If this setup is known to use the opposite orientation, pass "
                    "--calibration-expected-slope auto or the matching sign.",
                    file=sys.stderr,
                )
        else:
            sign_txt = "positive" if calibration.error_to_um > 0 else "negative"
            print(
                f"Calibration slope sign (auto mode): {sign_txt} ({calibration.error_to_um:+0.4f} um/error)",
                file=sys.stderr,
            )

    return calibration


def _build_stage(args, *, mm_core=None) -> StageInterface:
    stage_backend = args.stage or ("simulate" if args.camera == "simulate" else "mcl")

    if stage_backend == "simulate":
        if args.stage_dll is not None or args.stage_wrapper is not None:
            print(
                "Warning: --stage simulate ignores --stage-dll/--stage-wrapper inputs.",
                file=sys.stderr,
            )
        return MclNanoZStage(axis_index=args.stage_axis)

    if stage_backend == "micromanager":
        if mm_core is None:
            raise RuntimeError(
                "Micro-Manager stage backend requested but no Micro-Manager core is available. "
                "Use --camera micromanager or choose --stage mcl/simulate."
            )
        if args.stage_dll is not None or args.stage_wrapper is not None:
            print(
                "Warning: --stage micromanager ignores --stage-dll/--stage-wrapper inputs.",
                file=sys.stderr,
            )
        from .micromanager import MicroManagerStage

        return MicroManagerStage(core=mm_core)

    try:
        stage = MclNanoZStage(dll_path=args.stage_dll, wrapper_module=args.stage_wrapper, axis_index=args.stage_axis)
    except (NotConnectedError, OSError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Failed to initialize MCL stage: {exc}. "
            "If the stage is controlled through Micro-Manager, use --stage micromanager. "
            "To run without hardware stage control, use --stage simulate."
        ) from exc
    if args.stage_dll is None and args.stage_wrapper is None:
        print(
            "Warning: no explicit MCL stage backend configured; using in-memory simulated stage.",
            file=sys.stderr,
        )
    return stage


def _build_camera_and_stage(args):
    # Simulated camera + in-memory stage
    if args.camera == "simulate":
        stage = _build_stage(args)
        stage.move_z_um(1.5)
        camera = SimulatedCamera(stage=stage)
        return camera, stage

    # Micro-Manager camera stream (MM owns camera only)
    if args.camera == "micromanager":
        from .micromanager import create_micromanager_frame_source

        mm_source = create_micromanager_frame_source(
            host=args.mm_host,
            port=args.mm_port,
            allow_standalone_core=args.mm_allow_standalone_core,
        )
        stage = _build_stage(args, mm_core=mm_source.core)
        camera = HamamatsuOrcaCamera(
            frame_source=mm_source,
            control_source_lifecycle=False,
        )
        return camera, stage

    # pylablib camera (orca / andor) + package-controlled stage
    stage = _build_stage(args)
    camera_kwargs = {"idx": args.camera_index} if args.camera == "orca" else {"camera_index": args.camera_index}
    frame_source = create_pylablib_frame_source(args.camera, **camera_kwargs)
    camera = HamamatsuOrcaCamera(frame_source=frame_source, control_source_lifecycle=True)
    return camera, stage


def main() -> int:
    args = build_parser().parse_args()

    camera, stage = _build_camera_and_stage(args)
    camera_started = False

    try:
        camera.start()
        camera_started = True

        config = AutofocusConfig(
            roi=Roi(x=20, y=20, width=24, height=24),
            loop_hz=args.loop_hz,
            max_dt_s=args.max_dt_s,
            kp=args.kp,
            ki=args.ki,
            max_step_um=args.max_step,
            stage_min_um=args.stage_min_um,
            stage_max_um=args.stage_max_um,
            max_abs_excursion_um=(None if args.af_max_excursion_um < 0 else args.af_max_excursion_um),
            command_deadband_um=args.command_deadband_um,
            track_roi=bool(args.track_roi or args.track_roi_xy),
            track_gain=args.track_gain,
            track_deadband_px=args.track_deadband_px,
            track_max_shift_px=args.track_max_step_px,
            use_gaussian_fit=bool(args.use_gaussian_fit),
            gaussian_fast_mode=bool(args.fast_mode),
            use_analytical_inversion=bool(args.use_analytical_inversion),
            detection_window_size=int(args.detection_window),
            detection_pass_threshold=float(args.detection_threshold),
            min_roi_intensity_fraction=args.min_roi_intensity_fraction,
            background_percentile=args.background_percentile,
            engage_hold_good_frames=5,
            engage_ramp_s=1.2,
            min_confidence_scale=0.2,
            dynamic_deadband_coeff=1.5,
            dynamic_deadband_window=30,
            roi_expand_on_low_confidence=True,
            roi_expand_step_px=2,
            roi_expand_max_px=8,
            roi_shrink_stable_frames=15,
            detection_bounds=DetectionBounds(
                fwhm_min_px=1.0,
                fwhm_max_px=12.0,
                ellipticity_min=0.7,
                ellipticity_max=1.35,
                r_squared_min=0.6,
            ),
        )
        try:
            calibration = _load_startup_calibration(
                args.calibration_csv,
                model=args.calibration_model,
                expected_slope=args.calibration_expected_slope,
            )
        except ValueError as exc:
            if (not args.show_live) or (not args.allow_missing_calibration):
                raise
            print(f"Warning: {exc}", file=sys.stderr)
            print(
                "Warning: using temporary default calibration for live setup only; "
                "run calibration sweep and restart to use saved calibration.",
                file=sys.stderr,
            )
            calibration = FocusCalibration(error_at_focus=0.0, error_to_um=1.0)

        if args.show_live:
            launch_autofocus_viewer(
                camera,
                stage,
                calibration=calibration,
                default_config=config,
                calibration_output_path=args.calibration_csv,
                calibration_half_range_um=args.calibration_half_range_um,
                calibration_steps=args.calibration_steps,
            )
            return 0

        controller = AstigmaticAutofocusController(
            camera=camera,
            stage=stage,
            config=config,
            calibration=calibration,
        )

        samples = controller.run(duration_s=args.duration)

        if samples:
            final = samples[-1]
            print(
                f"camera={args.camera} steps={len(samples)} final_error={final.error:+0.4f} "
                f"final_error_um={final.error_um:+0.3f} stage={final.commanded_z_um:+0.3f} um"
            )
        return 0
    finally:
        if camera_started:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())

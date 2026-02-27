# Focus Feedback

Astigmatic autofocus for microscopy with:

- live PyQtGraph GUI,
- calibration sweep and model fitting,
- closed-loop Z control,
- software XY ROI tracking,
- simulation and hardware backends.

This package is built for the cylindrical-lens autofocus workflow where a bright spot becomes elliptical away from focus. It measures that PSF shape in a small ROI and applies only the Z correction needed to bring the spot back to the locked focus position.

## What This Package Does

The package provides two tightly connected pieces:

1. Calibration
   - Sweep Z through focus.
   - Fit the PSF at each step.
   - Build a defocus model from ellipticity vs Z.

2. Runtime autofocus
   - Track one spot in a drawn ROI.
   - Optionally move the ROI in XY if the spot drifts laterally.
   - Move only the Z stage.
   - Apply relative Z corrections, not an absolute jump to the calibration focus position.

## Current Calibration Behavior

The default calibration path now follows the publication/reference implementation closely:

- integrated elliptical Gaussian PSF model for calibration fitting,
- two-pass theta handling:
  - first pass with free theta,
  - global theta estimate from good fits,
  - second pass with theta fixed,
- Zhuang ellipticity model fit for defocus calibration,
- analytical inversion path for runtime control,
- explicit separation of:
  - `parametric_fit` samples,
  - `moment_fallback` samples.

Fallback samples are kept for diagnostics and export, but they are excluded from the parametric Zhuang fit.

## Main Runtime Safeguards

- relative lock setpoint: autofocus holds the engaged focus position instead of driving to an absolute Z from calibration,
- calibration quality gates before arming,
- safe engage ramp on startup,
- confidence-weighted control,
- dynamic deadband,
- anti-runaway checks,
- stage excursion limits,
- detection filtering on fit quality,
- calibration-domain guards,
- ROI edge and low-signal handling.

## Installation

Use Python 3.10+.

### Editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[ui]
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[camera]
```

Installed command:

```bash
focus-feedback --help
```

Repo-local alternative without install:

```bash
PYTHONPATH=src python -m auto_focus.cli --help
```

## Quick Start

### 1. Simulated GUI session

```bash
focus-feedback --camera simulate --show-live --allow-missing-calibration
```

This starts the GUI with a temporary placeholder calibration so you can:

- position the ROI,
- run a calibration sweep,
- inspect the fit plot,
- save a real calibration file.

### 2. Reuse a saved calibration

```bash
focus-feedback \
  --camera simulate \
  --show-live \
  --calibration-csv calibration_sweep.csv
```

### 3. Headless run

```bash
focus-feedback \
  --camera simulate \
  --duration 10 \
  --loop-hz 30 \
  --calibration-csv calibration_sweep.csv
```

## Recommended Workflow

1. Start live mode.
2. Draw a tight ROI around one bright stable spot.
3. Run calibration.
4. Inspect the calibration plot.
5. Accept the calibration only if the fit is clean and the usable range brackets focus.
6. Start autofocus.
7. If the spot drifts in XY, let the ROI track it instead of moving an XY stage.

## GUI Behavior

The GUI provides:

- live image view,
- draggable ROI,
- autofocus start/stop,
- stage home/center,
- calibration sweep,
- calibration plot and fit preview,
- diagnostics plots,
- display LUT controls,
- persistent GUI settings.

### XY ROI tracking

XY tracking moves only the ROI drawn in the GUI. It does not command an XY stage.

Use it when:

- the spot drifts slowly in XY,
- the PSF remains fully inside the ROI,
- you want to keep focus lock without stage recentering.

Practical starting values:

- `track_gain`: `0.3` to `0.5`
- `track_deadband_px`: `1` to `2`
- `track_max_step_px`: `5` to `10`

## Calibration Outputs

Running calibration in the GUI writes a calibration CSV and companion files next to it.

Typical files:

- `calibration_sweep.csv`
- `calibration_sweep.meta.json`
- `autofocus_gui_settings.json`
- `autofocus_events.log`
- `autofocus_run_report.json`

The CSV contains the sweep data used for later reuse and diagnostics, including the sample class labels so fallback points are not mistaken for the main calibration branch.

## CLI Options You Will Actually Use

### Core

- `--show-live`
- `--camera {simulate,orca,andor,micromanager}`
- `--stage {simulate,mcl,micromanager}`
- `--calibration-csv`
- `--allow-missing-calibration`

### Control tuning

- `--loop-hz`
- `--kp`
- `--ki`
- `--max-step`
- `--command-deadband-um`
- `--af-max-excursion-um`

### Calibration

- `--calibration-model {zhuang,linear,poly2,piecewise}`
- `--calibration-half-range-um`
- `--calibration-steps`
- `--calibration-expected-slope {auto,positive,negative}`

### ROI tracking

- `--track-roi`
- `--track-roi-xy`
- `--track-gain`
- `--track-deadband-px`
- `--track-max-step-px`

### Detection and image quality

- `--use-gaussian-fit`
- `--no-gaussian-fit`
- `--fast-mode`
- `--background-percentile`
- `--min-roi-intensity-fraction`

## Hardware Backends

Supported camera modes:

- `simulate`
- `orca`
- `andor`
- `micromanager`

Supported stage modes:

- `simulate`
- `mcl`
- `micromanager`

Notes:

- `simulate` stage uses the package’s in-memory fallback stage implementation.
- `mcl` stage supports DLL, wrapper module, or in-memory fallback.
- `micromanager` stage requires a working Micro-Manager core connection.

## Deployment Notes

Before deploying on hardware:

1. Install the package, not just `PYTHONPATH=src`.
2. Confirm the GUI launches with `focus-feedback --show-live --allow-missing-calibration --camera simulate`.
3. Run one real calibration on hardware.
4. Confirm the calibration plot shows a usable branch and acceptable fit quality.
5. Run a hardware-in-loop lock test for several minutes at the expected signal level.

## Tuning Guidance

Start conservatively.

Suggested initial values:

- `kp`: `0.6` to `0.8`
- `ki`: `0.1` to `0.2`
- `max-step`: `0.1` to `0.25` um
- `command-deadband-um`: `0.005` to `0.02` um

If the lock chatters:

- lower `kp`,
- increase deadband,
- reduce `max-step`.

If the lock is too slow:

- raise `kp` slightly,
- then add a small amount of `ki`.

## Troubleshooting

### `Calibration CSV not found`

Either:

- point `--calibration-csv` to an existing file, or
- start the GUI with `--allow-missing-calibration` and create one.

### `Calibration failed`

Common causes:

- ROI does not stay on one spot,
- low SNR,
- PSF clipped by ROI edge,
- stage settling/backlash mismatch,
- sweep not centered near focus.

### `error outside calibration domain`

This means the runtime measurement is outside the trusted calibrated range. Typical reasons:

- wrong target,
- ROI drift,
- poor calibration,
- focus already far outside the calibrated span.

### Autofocus drives the spot out of focus

Check:

- calibration quality,
- sign convention,
- usable range,
- fit quality,
- ROI tracking,
- stage axis selection and limits.

### GUI becomes slow during PSF fitting

Calibration fitting is intentionally moved after the Z sweep so acquisition remains smooth. If fitting is still too slow, reduce sweep size or increase signal quality.

## Project Layout

- `src/auto_focus/autofocus.py` — control loop and state machine
- `src/auto_focus/calibration.py` — sweep fitting and calibration models
- `src/auto_focus/focus_metric.py` — PSF fitting and image metrics
- `src/auto_focus/pg_gui.py` — live PyQtGraph GUI
- `src/auto_focus/cli.py` — CLI entrypoint
- `src/auto_focus/hardware.py` — stage and camera adapters
- `tests/` — regression tests

## License

MIT

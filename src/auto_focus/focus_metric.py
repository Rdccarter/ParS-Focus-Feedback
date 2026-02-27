from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any

import numpy as np

from .interfaces import Image2D


# --- Numba-accelerated Gaussian fitting (optional, graceful fallback) ---
if importlib.util.find_spec("numba") is not None:
    _numba_jit = importlib.import_module("numba").jit
    _HAS_NUMBA = True
else:
    _HAS_NUMBA = False

    def _numba_jit(*args, **kwargs):
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator


@_numba_jit(nopython=True, nogil=True)
def _meshgrid_numba(x, y):
    s = (len(y), len(x))
    xv = np.zeros(s)
    yv = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            xv[i, j] = x[j]
            yv[i, j] = y[i]
    return xv, yv


@_numba_jit(nopython=True, nogil=True)
def _erf_numba(x):
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


@_numba_jit(nopython=True, nogil=True)
def _erf2d_numba(x):
    s = x.shape
    y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            y[i, j] = _erf_numba(x[i, j])
    return y


@_numba_jit(nopython=True, nogil=True)
def _gaussian7_numba(p, xv, yv):
    """7-parameter integrated Gaussian on a pixel grid."""
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    if p[5] == 0:
        dx = efac
        dy = efac
    else:
        dx = efac / p[5]
        dy = efac * p[5]
    cos_t = np.cos(p[6])
    sin_t = np.sin(p[6])
    x = 2 * dx * (cos_t * (xv - p[0]) - (yv - p[1]) * sin_t)
    y = 2 * dy * (cos_t * (yv - p[1]) + (xv - p[0]) * sin_t)
    return p[3] / 4 * (_erf2d_numba(x + dx) - _erf2d_numba(x - dx)) * (_erf2d_numba(y + dy) - _erf2d_numba(y - dy)) + p[4]


def _gaussian7_python(p, xv, yv):
    """Pure-Python fallback for _gaussian7_numba."""
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    ell = p[5] if p[5] > 1e-6 else 1e-6
    dx = efac / ell
    dy = efac * ell
    cos_t = np.cos(p[6])
    sin_t = np.sin(p[6])
    x = 2 * dx * (cos_t * (xv - p[0]) - (yv - p[1]) * sin_t)
    y = 2 * dy * (cos_t * (yv - p[1]) + (xv - p[0]) * sin_t)
    erf = importlib.import_module("scipy.special").erf
    return p[3] / 4 * (erf(x + dx) - erf(x - dx)) * (erf(y + dy) - erf(y - dy)) + p[4]


@dataclass(slots=True)
class Roi:
    """Axis-aligned region of interest in image coordinates."""

    x: int
    y: int
    width: int
    height: int

    def clamp(self, image_shape: tuple[int, int]) -> "Roi":
        h, w = image_shape
        x = min(max(0, self.x), max(0, w - 1))
        y = min(max(0, self.y), max(0, h - 1))
        width = min(self.width, w - x)
        height = min(self.height, h - y)
        if width <= 0 or height <= 0:
            raise ValueError("ROI does not intersect image")
        return Roi(x=x, y=y, width=width, height=height)


def _coerce_image_2d(image: Any) -> Image2D:
    if hasattr(image, "tolist") and callable(image.tolist):
        image = image.tolist()
    if isinstance(image, tuple):
        image = list(image)
    if not isinstance(image, list):
        raise TypeError("Image must be a 2D list/tuple or expose tolist()")

    out: Image2D = []
    width: int | None = None
    for row in image:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise TypeError("Image rows must be list/tuple")
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError("Empty image")
        elif len(row) != width:
            raise ValueError("Image rows must have equal length")
        out.append([float(v) for v in row])

    if not out:
        raise ValueError("Empty image")
    return out


def _image_shape(image: Image2D) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("Empty image")
    return len(image), len(image[0])


def extract_roi(image: Image2D, roi: Roi) -> Image2D:
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            if image.ndim != 2:
                raise ValueError("Image must be 2D")
            safe_roi = roi.clamp((image.shape[0], image.shape[1]))
            return image[
                safe_roi.y : safe_roi.y + safe_roi.height,
                safe_roi.x : safe_roi.x + safe_roi.width,
            ]
    except Exception:
        pass

    safe_image = _coerce_image_2d(image)
    h, w = _image_shape(safe_image)
    safe_roi = roi.clamp((h, w))
    return [
        row[safe_roi.x : safe_roi.x + safe_roi.width]
        for row in safe_image[safe_roi.y : safe_roi.y + safe_roi.height]
    ]


def _astigmatic_error_signal_numpy(patch: Image2D, background_percentile: float = 20.0) -> float:
    try:
        import numpy as np
    except Exception:
        return _astigmatic_error_signal_python(patch, background_percentile=background_percentile)

    arr = np.asarray(patch, dtype=float)

    # Background-subtract so uniform illumination doesn't dilute the
    # intensity-weighted second moments.  Without this, background pixels
    # compress the (var_x - var_y)/(var_x + var_y) ratio toward zero,
    # making the error signal ~20× weaker than the Zhuang model expects.
    pctl = max(0.0, min(100.0, float(background_percentile)))
    bg = float(np.percentile(arr, pctl))
    arr = np.maximum(arr - bg, 0.0)

    total = float(arr.sum())
    if total <= 0:
        return 0.0

    y_idx, x_idx = np.indices(arr.shape, dtype=float)
    cx = float((x_idx * arr).sum() / total)
    cy = float((y_idx * arr).sum() / total)

    var_x = float((((x_idx - cx) ** 2) * arr).sum() / total)
    var_y = float((((y_idx - cy) ** 2) * arr).sum() / total)

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _astigmatic_error_signal_python(patch: Image2D, background_percentile: float = 20.0) -> float:
    # Flatten to estimate background percentile.
    all_vals = sorted(v for row in patch for v in row)
    if not all_vals:
        return 0.0
    pctl = max(0.0, min(100.0, float(background_percentile)))
    idx = int(round((len(all_vals) - 1) * (pctl / 100.0)))
    bg = all_vals[max(0, min(len(all_vals) - 1, idx))]

    total = 0.0
    for row in patch:
        for val in row:
            total += max(0.0, val - bg)
    if total <= 0:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            w = max(0.0, val - bg)
            sum_x += x * w
            sum_y += y * w

    cx = sum_x / total
    cy = sum_y / total

    var_x = 0.0
    var_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            w = max(0.0, val - bg)
            var_x += ((x - cx) ** 2) * w
            var_y += ((y - cy) ** 2) * w

    var_x /= total
    var_y /= total

    denom = var_x + var_y
    if denom == 0:
        return 0.0
    return (var_x - var_y) / denom


def _centroid_python(patch: Image2D) -> tuple[float, float]:
    """Return (cx, cy) intensity-weighted centroid of the patch."""
    total = sum(sum(row) for row in patch)
    if total <= 0:
        h = len(patch)
        w = len(patch[0]) if patch else 0
        return (w - 1) / 2.0, (h - 1) / 2.0
    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(patch):
        for x, val in enumerate(row):
            sum_x += x * val
            sum_y += y * val
    return sum_x / total, sum_y / total


def centroid_near_edge(image: Image2D, roi: Roi, margin_px: float) -> bool:
    """Return True if the intensity centroid is within *margin_px* of the ROI boundary.

    This is a guard against truncated PSFs: when the bead drifts near the ROI
    edge, the second-moment error signal becomes biased and can drive runaway
    corrections.
    """
    if margin_px <= 0:
        return False
    patch = extract_roi(image, roi)
    h = len(patch)
    w = len(patch[0]) if patch else 0
    if h == 0 or w == 0:
        return True
    cx, cy = _centroid_python(patch)
    if cx < margin_px or cx > (w - 1) - margin_px:
        return True
    if cy < margin_px or cy > (h - 1) - margin_px:
        return True
    return False


def roi_total_intensity(image: Image2D, roi: Roi) -> float:
    patch = extract_roi(image, roi)
    return float(sum(sum(row) for row in patch))


def astigmatic_error_signal(image: Image2D, roi: Roi, background_percentile: float = 20.0) -> float:
    """Return focus error based on anisotropic second moments.

    Uses a NumPy-accelerated path when NumPy is available; otherwise falls back
    to a pure-Python implementation.
    """

    patch = extract_roi(image, roi)
    return _astigmatic_error_signal_numpy(patch, background_percentile=background_percentile)


# ---------------------------------------------------------------------------
# Gaussian PSF fitting for calibration sweeps
# ---------------------------------------------------------------------------



@dataclass(slots=True)
class FastGaussianResult:
    """Result of the fast 7-parameter Gaussian fit."""

    x: float
    y: float
    fwhm: float
    area: float
    offset: float
    ellipticity: float
    theta: float
    r_squared: float


def fit_gaussian_fast(
    patch: Image2D,
    theta: float | None = 0.0,
    fast_mode: bool = False,
) -> FastGaussianResult | None:
    """Fast 7-parameter Gaussian fit using optional numba acceleration."""
    if importlib.util.find_spec("scipy") is None:
        return None

    np_mod = importlib.import_module("numpy")
    opt = importlib.import_module("scipy.optimize")

    patch = _coerce_image_2d(patch)
    im = np_mod.asarray(patch, dtype=float)
    h, w = im.shape

    if h < 5 or w < 5:
        return None

    q = np_mod.zeros(7, dtype=float)
    q[4] = float(np_mod.nanmin(im))
    jm = im - q[4]
    total = float(np_mod.nansum(jm))
    if total <= 0:
        return None

    x_idx, y_idx = np_mod.meshgrid(range(w), range(h))
    q[0] = float(np_mod.nansum(x_idx * jm) / total)
    q[1] = float(np_mod.nansum(y_idx * jm) / total)
    q[3] = total

    theta0 = theta if theta is not None else 0.0
    cos_t = np_mod.cos(theta0)
    sin_t = np_mod.sin(theta0)
    xr = cos_t * (x_idx - q[0]) - (y_idx - q[1]) * sin_t
    yr = cos_t * (y_idx - q[1]) + (x_idx - q[0]) * sin_t

    s2 = float(np_mod.nansum(jm ** 2))
    if s2 <= 0:
        return None
    sx = np_mod.sqrt(float(np_mod.nansum((xr * jm) ** 2) / s2))
    sy = np_mod.sqrt(float(np_mod.nansum((yr * jm) ** 2) / s2))

    q[2] = np_mod.sqrt(sx * sy) * 4 * np_mod.sqrt(np_mod.log(2))
    q[5] = np_mod.sqrt(sx / sy) if sy > 0 else 1.0
    q[6] = theta0

    if q[2] <= 0 or q[2] > max(h, w) or q[5] <= 0:
        return None

    if _HAS_NUMBA:
        xv, yv = _meshgrid_numba(np_mod.arange(w, dtype=float), np_mod.arange(h, dtype=float))
        model = _gaussian7_numba(q, xv, yv)
    else:
        xv, yv = np_mod.meshgrid(np_mod.arange(w, dtype=float), np_mod.arange(h, dtype=float))
        model = _gaussian7_python(q, xv, yv)

    ss_res = float(np_mod.nansum((im - model) ** 2))
    ss_tot = float(np_mod.nansum((im - np_mod.nanmean(im)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if fast_mode:
        return FastGaussianResult(x=q[0], y=q[1], fwhm=abs(q[2]), area=q[3], offset=q[4], ellipticity=abs(q[5]), theta=q[6], r_squared=r2)

    if q[2] > max(h, w) / 2 or q[3] < 0.1:
        return None

    if _HAS_NUMBA:
        if theta is not None:
            def cost(pf):
                p7 = np_mod.append(pf, theta)
                return float(np_mod.sum((im - _gaussian7_numba(p7, xv, yv)) ** 2))

            p0 = q[:6]
        else:
            def cost(pf):
                return float(np_mod.sum((im - _gaussian7_numba(pf, xv, yv)) ** 2))

            p0 = q
    else:
        if theta is not None:
            def cost(pf):
                p7 = np_mod.append(pf, theta)
                return float(np_mod.sum((im - _gaussian7_python(p7, xv, yv)) ** 2))

            p0 = q[:6]
        else:
            def cost(pf):
                return float(np_mod.sum((im - _gaussian7_python(pf, xv, yv)) ** 2))

            p0 = q

    try:
        result = opt.minimize(cost, p0, options={"disp": False, "maxiter": 500})
        qf = result.x
    except (RuntimeError, ValueError):
        return None

    if theta is not None:
        qf = np_mod.append(qf, theta)

    if _HAS_NUMBA:
        model_f = _gaussian7_numba(qf, xv, yv)
    else:
        model_f = _gaussian7_python(qf, xv, yv)
    ss_res_f = float(np_mod.nansum((im - model_f) ** 2))
    r2_f = 1.0 - ss_res_f / ss_tot if ss_tot > 0 else 0.0

    return FastGaussianResult(x=qf[0], y=qf[1], fwhm=abs(qf[2]), area=qf[3], offset=qf[4], ellipticity=abs(qf[5]), theta=qf[6], r_squared=r2_f)


@dataclass(slots=True)
class GaussianPsfResult:
    """Result of a 2D Gaussian PSF fit.

    Parameters follow the calibration/export convention used by the
    Lenstra/Zhuang workflow:
      sigma_x, sigma_y are the axis widths reconstructed from the integrated
      PSF fit parameterization,
      ellipticity = sigma_x / sigma_y
    """
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    offset: float
    theta: float  # rotation angle (radians)
    ellipticity: float  # sigma_x / sigma_y
    r_squared: float


def fit_gaussian_psf(patch: Image2D, theta: float | None = None) -> GaussianPsfResult | None:
    """Fit the publication-style integrated elliptical Gaussian PSF.

    This uses the same 7-parameter integrated model as the Lenstra/Zhuang
    implementation, so calibration and runtime use a consistent ellipticity
    definition.
    """
    fast = fit_gaussian_fast(patch, theta=theta, fast_mode=False)
    if fast is None:
        return None

    # Match the reference calibration conversion:
    #   s = fwhm / (2*sqrt(2*ln(2)))
    #   sigma_x = s * sqrt(e)
    #   sigma_y = s / sqrt(e)
    # where e is the fitted integrated-PSF ellipticity parameter.
    import math
    s = float(fast.fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    e = float(abs(fast.ellipticity))
    if not math.isfinite(s) or not math.isfinite(e) or s <= 0 or e <= 0:
        return None

    sqrt_e = math.sqrt(e)
    sigma_x = s * sqrt_e
    sigma_y = s / sqrt_e

    return GaussianPsfResult(
        x=float(fast.x),
        y=float(fast.y),
        sigma_x=float(sigma_x),
        sigma_y=float(sigma_y),
        amplitude=float(fast.area),
        offset=float(fast.offset),
        theta=float(fast.theta),
        ellipticity=float(e),
        r_squared=float(fast.r_squared),
    )

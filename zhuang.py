"""
Zhuang/Huang astigmatic defocus model.

Implements the standard defocus model from:
  Huang, Wang, Bates & Zhuang, Science 319, 810 (2008)

Each lateral PSF width (sigma_x, sigma_y) follows:
  sigma(z) = sigma0 * sqrt(1 + ((z-c)/d)^2 + A*((z-c)/d)^3 + B*((z-c)/d)^4)

The ellipticity ratio e = sigma_x / sigma_y varies monotonically near focus
and can be inverted to recover z.

This module includes:
  - forward model: sigma(z), ellipticity(z)
  - robust parameter fitting (recommended): least_squares + robust loss + bounds + outlier pruning
  - numerical inversion: z from ellipticity
  - theoretical and empirical lookup tables for fast runtime z estimation

Ported and cleaned up from the cylindrical-lens reference (Vliet lab).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special


# ---------------------------------------------------------------------------
# Core Zhuang defocus curve
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZhuangParams:
    """Parameters for one lateral axis of the Zhuang defocus model.

    sigma(z) = sigma0 * sqrt(1 + X^2 + A*X^3 + B*X^4)
    where X = (z - c) / d
    """
    sigma0: float   # minimum PSF width (um or px)
    A: float        # 3rd-order coefficient
    B: float        # 4th-order coefficient
    c: float        # z offset of this axis's focus (um)
    d: float        # focal depth parameter (um)


@dataclass(slots=True)
class ZhuangEllipticityParams:
    """Combined 9-parameter model: e(z) = e0 * sqrt(sigma_x^2(z) / sigma_y^2(z)).

    q = [e0, z0, c, Ax, Bx, dx, Ay, By, dy]
    where:
      e0    = sigma_x0 / sigma_y0 at focus
      z0    = center of focus
      c     = half-separation between x and y foci
      Ax,Bx = 3rd/4th order for x axis
      dx    = focal depth for x axis
      Ay,By,dy = same for y axis
    """
    e0: float
    z0: float
    c: float
    Ax: float
    Bx: float
    dx: float
    Ay: float
    By: float
    dy: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.e0, self.z0, self.c,
             self.Ax, self.Bx, self.dx,
             self.Ay, self.By, self.dy],
            dtype=float,
        )

    @classmethod
    def from_array(cls, q: np.ndarray) -> "ZhuangEllipticityParams":
        q = np.asarray(q, dtype=float)
        return cls(e0=q[0], z0=q[1], c=q[2],
                   Ax=q[3], Bx=q[4], dx=q[5],
                   Ay=q[6], By=q[7], dy=q[8])

    @classmethod
    def from_axis_params(cls, px: ZhuangParams, py: ZhuangParams) -> "ZhuangEllipticityParams":
        """Construct from independently fitted x and y axis parameters."""
        e0 = (px.sigma0 / py.sigma0) if (np.isfinite(py.sigma0) and py.sigma0 > 0) else 1.0
        z0 = (px.c + py.c) / 2.0
        c = (px.c - py.c) / 2.0
        return cls(e0=e0, z0=z0, c=c,
                   Ax=px.A, Bx=px.B, dx=px.d,
                   Ay=py.A, By=py.B, dy=py.d)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def zhuang_sigma(z: np.ndarray | float, p: ZhuangParams) -> np.ndarray | float:
    """Evaluate sigma(z) for one axis."""
    z_arr = np.asarray(z, dtype=float)
    if not np.isfinite(p.d) or p.d == 0:
        out = np.full_like(z_arr, np.nan, dtype=float)
        return float(out) if np.isscalar(z) else out

    X = (z_arr - p.c) / p.d
    arg = 1.0 + X**2 + p.A * X**3 + p.B * X**4
    with np.errstate(invalid="ignore"):
        out = p.sigma0 * np.sqrt(np.maximum(arg, 0.0))
    return float(out) if np.isscalar(z) else out


def zhuang_ellipticity(z: np.ndarray | float, q: ZhuangEllipticityParams | np.ndarray) -> np.ndarray | float:
    """Ellipticity e(z) = e0 * sqrt(sigma_x^2 / sigma_y^2).

    q: [e0, z0, c, Ax, Bx, dx, Ay, By, dy]
    """
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    q = np.asarray(q, dtype=float)

    z_arr = np.asarray(z, dtype=float)

    dx = q[5]
    dy = q[8]
    if dx == 0 or dy == 0 or (not np.isfinite(dx)) or (not np.isfinite(dy)):
        out = np.full_like(z_arr, np.nan, dtype=float)
        return float(out) if np.isscalar(z) else out

    # Model convention:
    #   x-axis focus at z = z0 + c
    #   y-axis focus at z = z0 - c
    X = (z_arr - (q[1] + q[2])) / dx
    Y = (z_arr - (q[1] - q[2])) / dy

    num = 1.0 + X**2 + q[3] * X**3 + q[4] * X**4
    den = 1.0 + Y**2 + q[6] * Y**3 + q[7] * Y**4

    with np.errstate(divide="ignore", invalid="ignore"):
        out = q[0] * np.sqrt(np.maximum(num, 0.0) / np.maximum(den, 1e-30))

    return float(out) if np.isscalar(z) else out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mad_sigma(x: np.ndarray) -> float:
    """Robust sigma estimate from MAD."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return 1.4826 * mad


def _zhuang_initial_guess(z: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Moment-based initial guess for [sigma0, A, B, c, d]."""
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    p = np.zeros(5, dtype=float)
    p[0] = float(np.nanmin(sigma))
    p[3] = float(z[np.nanargmin(sigma)])

    # crude d estimate from sigma growth; ignore invalid regions
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (sigma / max(p[0], 1e-12))**2 - 1.0
        d_est = np.sqrt((z - p[3])**2 / denom)

    d_est = d_est[np.isfinite(d_est) & (d_est > 0)]
    p[4] = float(np.nanmedian(d_est)) if d_est.size > 0 else 1.0
    if not np.isfinite(p[4]) or p[4] <= 0:
        p[4] = 1.0

    # start with no high-order terms
    p[1] = 0.0
    p[2] = 0.0
    return p


def _cov_from_jacobian(jac: np.ndarray, res: np.ndarray, dof: int) -> np.ndarray | None:
    """Covariance estimate: s^2 * inv(J^T J), with s^2 from SSE/dof."""
    if dof <= 0:
        return None
    try:
        jt = jac.T
        jtj = jt @ jac
        s2 = float(np.sum(res**2) / dof)
        cov = s2 * np.linalg.inv(jtj)
        return cov
    except np.linalg.LinAlgError:
        return None


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators algorithm for monotone increasing fit."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    result = y.copy()
    blocks = [[i] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        mean_curr = np.mean(result[blocks[i]])
        mean_next = np.mean(result[blocks[i + 1]])
        if mean_curr <= mean_next:
            i += 1
        else:
            merged = blocks[i] + blocks[i + 1]
            merged_mean = np.mean(result[merged])
            for idx in merged:
                result[idx] = merged_mean
            blocks[i] = merged
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
    return result


# ---------------------------------------------------------------------------
# Fitting: single-axis Zhuang curve (robust recommended)
# ---------------------------------------------------------------------------

def fit_zhuang_axis_robust(
    z: np.ndarray,
    sigma: np.ndarray,
    dsigma: np.ndarray | None = None,
    loss: str = "soft_l1",      # "soft_l1" or "huber"
    f_scale: float | None = None,
    max_nfev: int = 20000,
    outlier_sigma: float = 6.0,
) -> tuple[ZhuangParams, np.ndarray, tuple[float, float]]:
    """Robust least-squares fit of Zhuang sigma(z) for one axis.

    Returns (params, d_params, (chi2_reduced, R2)).

    Notes:
      - Uses bounded least_squares to enforce sigma0>0 and d>0.
      - Uses robust loss (soft_l1/huber) to reduce the impact of outliers.
      - Optionally does a second pass after pruning gross outliers by MAD.
    """
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mask = np.isfinite(z) & np.isfinite(sigma)
    if dsigma is not None:
        dsigma = np.asarray(dsigma, dtype=float)
        mask &= np.isfinite(dsigma)
        dsigma = dsigma[mask]
    z = z[mask]
    sigma = sigma[mask]

    nan5 = np.full(5, np.nan)
    if z.size < 6:
        return ZhuangParams(np.nan, np.nan, np.nan, np.nan, np.nan), nan5, (np.nan, np.nan)

    # weights: residual = (sigma - model) / dsigma
    if dsigma is None:
        w = np.ones_like(sigma, dtype=float)
    else:
        w = 1.0 / np.clip(dsigma, 1e-10, None)

    p0 = _zhuang_initial_guess(z, sigma)

    lb = np.array([1e-6, -np.inf, -np.inf, -np.inf, 1e-6], dtype=float)
    ub = np.array([np.inf,  np.inf,  np.inf,  np.inf,  np.inf], dtype=float)

    def residuals(p, z_, sigma_, w_):
        model = zhuang_sigma(z_, ZhuangParams(*p))
        return (sigma_ - model) * w_

    # choose scale for robust loss
    if f_scale is None:
        # quick unrobust residual scale guess: typical sigma noise in weighted units
        # start with 1.0 so robust loss behaves sensibly even with unknown dsigma
        f_scale_use = 1.0
    else:
        f_scale_use = float(f_scale)

    # pass 1
    r1 = scipy.optimize.least_squares(
        residuals, p0, args=(z, sigma, w),
        bounds=(lb, ub),
        loss=loss, f_scale=f_scale_use,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    p1 = r1.x
    res1 = residuals(p1, z, sigma, w)

    # robust scale and prune gross outliers (in weighted residual units)
    s1 = _mad_sigma(res1)
    if not np.isfinite(s1) or s1 <= 0:
        s1 = 1.0
    keep = np.abs(res1 - np.median(res1)) <= (outlier_sigma * s1)

    if keep.sum() >= 6 and keep.sum() < z.size:
        z2 = z[keep]
        s2 = sigma[keep]
        w2 = w[keep]
        # pass 2 with scale matched to residual dispersion
        r2 = scipy.optimize.least_squares(
            residuals, p1, args=(z2, s2, w2),
            bounds=(lb, ub),
            loss=loss, f_scale=max(s1, 1e-3),
            max_nfev=max_nfev,
            x_scale="jac",
        )
        p = r2.x
        z_fit, sigma_fit, w_fit = z2, s2, w2
        r_final = r2
    else:
        p = p1
        z_fit, sigma_fit, w_fit = z, sigma, w
        r_final = r1

    dof = int(z_fit.size - p.size)
    if dof <= 0 or (not r_final.success):
        return ZhuangParams(*p), nan5, (np.nan, np.nan)

    res = residuals(p, z_fit, sigma_fit, w_fit)
    chi2_red = float(np.sum(res**2) / dof)

    cov = _cov_from_jacobian(r_final.jac, res, dof)
    if cov is None:
        dp = nan5
    else:
        dp = np.sqrt(np.diag(cov))

    # R2 in unweighted sigma space
    model = zhuang_sigma(z_fit, ZhuangParams(*p))
    ss_res = float(np.sum((sigma_fit - model) ** 2))
    ss_tot = float(np.sum((sigma_fit - float(np.mean(sigma_fit))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return ZhuangParams(*p), dp, (chi2_red, r2)


# Backward-compatible alias (robust is the recommended default now)
def fit_zhuang_axis(
    z: np.ndarray,
    sigma: np.ndarray,
    dsigma: np.ndarray | None = None,
) -> tuple[ZhuangParams, np.ndarray, tuple[float, float]]:
    """Alias for the robust axis fit (recommended)."""
    return fit_zhuang_axis_robust(z, sigma, dsigma=dsigma)


# ---------------------------------------------------------------------------
# Fitting: combined ellipticity model (robust)
# ---------------------------------------------------------------------------

def fit_zhuang_ellipticity_robust(
    z: np.ndarray,
    ell: np.ndarray,
    initial: ZhuangEllipticityParams | np.ndarray | None = None,
    loss: str = "soft_l1",
    f_scale: float = 0.02,      # typical ellipticity noise scale
    max_nfev: int = 30000,
    outlier_sigma: float = 6.0,
) -> tuple[ZhuangEllipticityParams, np.ndarray, float]:
    """Robust fit of 9-parameter ellipticity model.

    Returns (params, d_params, chi2_reduced), where chi2_reduced is in residual space.

    Bounds enforced:
      - e0 > 0
      - dx > 0
      - dy > 0
    """
    z = np.asarray(z, dtype=float)
    ell = np.asarray(ell, dtype=float)

    mask = np.isfinite(z) & np.isfinite(ell) & (ell > 0)
    z = z[mask]
    ell = ell[mask]

    nan9 = np.full(9, np.nan)
    if z.size < 12:
        return ZhuangEllipticityParams(*nan9), nan9, np.nan

    if initial is None:
        p0 = np.array([1.0, float(np.mean(z)), 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4], dtype=float)
    elif isinstance(initial, ZhuangEllipticityParams):
        p0 = initial.to_array().astype(float)
    else:
        p0 = np.asarray(initial, dtype=float)

    lb = np.array([1e-6, -np.inf, -np.inf, -np.inf, -np.inf, 1e-6, -np.inf, -np.inf, 1e-6], dtype=float)
    ub = np.array([np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf], dtype=float)

    def residuals(p, z_, ell_):
        return ell_ - zhuang_ellipticity(z_, p)

    # pass 1
    r1 = scipy.optimize.least_squares(
        residuals, p0, args=(z, ell),
        bounds=(lb, ub),
        loss=loss, f_scale=float(f_scale),
        max_nfev=max_nfev,
        x_scale="jac",
    )
    p1 = r1.x
    res1 = residuals(p1, z, ell)

    s1 = _mad_sigma(res1)
    if not np.isfinite(s1) or s1 <= 0:
        s1 = float(f_scale)

    keep = np.abs(res1 - np.median(res1)) <= (outlier_sigma * s1)

    if keep.sum() >= 12 and keep.sum() < z.size:
        z2 = z[keep]
        e2 = ell[keep]
        r2 = scipy.optimize.least_squares(
            residuals, p1, args=(z2, e2),
            bounds=(lb, ub),
            loss=loss, f_scale=max(s1, float(f_scale)),
            max_nfev=max_nfev,
            x_scale="jac",
        )
        p = r2.x
        z_fit, ell_fit = z2, e2
        r_final = r2
    else:
        p = p1
        z_fit, ell_fit = z, ell
        r_final = r1

    dof = int(z_fit.size - p.size)
    if dof <= 0 or (not r_final.success):
        return ZhuangEllipticityParams.from_array(p), nan9, np.nan

    res = residuals(p, z_fit, ell_fit)
    chi2_red = float(np.sum(res**2) / dof)

    cov = _cov_from_jacobian(r_final.jac, res, dof)
    if cov is None:
        dp = nan9
    else:
        dp = np.sqrt(np.diag(cov))

    return ZhuangEllipticityParams.from_array(p), dp, chi2_red


# Backward-compatible alias
def fit_zhuang_ellipticity(
    z: np.ndarray,
    ell: np.ndarray,
    initial: ZhuangEllipticityParams | np.ndarray | None = None,
) -> tuple[ZhuangEllipticityParams, np.ndarray, float]:
    """Alias for robust ellipticity fit (recommended)."""
    return fit_zhuang_ellipticity_robust(z, ell, initial=initial)


# ---------------------------------------------------------------------------
# Z lookup from ellipticity (inverse mapping)
# ---------------------------------------------------------------------------

def find_z_from_ellipticity(
    ell: float,
    q: ZhuangEllipticityParams | np.ndarray,
    z_range: tuple[float, float] | None = None,
) -> float:
    """Invert e(z) to find z given measured ellipticity.

    Uses numerical minimization of (e(z) - ell)^2 over the usable monotonic range.
    Returns NaN if no valid solution can be determined.
    """
    if isinstance(q, ZhuangEllipticityParams):
        q_arr = q.to_array()
    else:
        q_arr = np.asarray(q, dtype=float)

    if not np.isfinite(ell) or ell <= 0:
        return np.nan

    if z_range is None:
        z_range = _find_zhuang_usable_range(q_arr)
        if z_range is None:
            return np.nan

    z_analytic = findz_analytical(ell, q_arr, z_range=z_range)
    if np.isfinite(z_analytic):
        return float(z_analytic)

    z_grid = np.linspace(z_range[0], z_range[1], 200)
    e_grid = zhuang_ellipticity(z_grid, q_arr)
    residuals = np.abs(e_grid - ell)
    if not np.any(np.isfinite(residuals)):
        return np.nan

    i_best = int(np.nanargmin(residuals))
    z0 = float(z_grid[i_best])

    result = scipy.optimize.minimize_scalar(
        lambda z: (zhuang_ellipticity(float(z), q_arr) - ell) ** 2,
        bounds=(float(z_range[0]), float(z_range[1])),
        method="bounded",
    )
    z_hat = float(result.x) if result.success else z0

    if z_hat < z_range[0] or z_hat > z_range[1]:
        return np.nan
    return z_hat




def findz_analytical_range(
    q: ZhuangEllipticityParams | np.ndarray,
) -> tuple[float, float] | None:
    """Exact usable range via analytical derivative roots (CylLensGUI method)."""
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    q = np.asarray(q, dtype=float)

    Lx = np.repeat(
        (q[4] / q[5]**4, q[3] / q[5]**3, 1 / q[5]**2, 0, 1),
        5,
    ).reshape((5, 5)).T
    Ly = np.repeat(
        (q[7] / q[8]**4, q[6] / q[8]**3, 1 / q[8]**2, 0, 1),
        5,
    ).reshape((5, 5)).T

    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    N1 = scipy.special.binom(T.T, T)
    N2 = scipy.special.binom(T.T - 1, T)
    N2[np.isnan(N2)] = 0
    S = (-np.ones((5, 5))) ** (T.T - T)

    cx = q[1] + q[2]
    cy = q[1] - q[2]

    za = np.sum(-N1 * cy ** (T.T - T) * Ly * S, axis=1)
    zb = np.sum(T.T * N2 * cx ** (T.T - T - 1) * Lx * S, axis=1)
    zc = np.sum(T.T * N2 * cy ** (T.T - T - 1) * Ly * S, axis=1)
    zd = np.sum(N1 * cx ** (T.T - T) * Lx * S, axis=1)

    # Build degree-8 polynomial for d(ell)/dz = 0
    poly = np.full(8, np.nan)
    poly[0] = za[0] * zb[1] + zc[1] * zd[0]
    poly[1] = za[::-1][3:5] @ zb[1:3] + zc[::-1][2:4] @ zd[0:2]
    poly[2] = za[::-1][2:5] @ zb[1:4] + zc[::-1][1:4] @ zd[0:3]
    poly[3] = za[::-1][1:5] @ zb[1:5] + zc[::-1][0:4] @ zd[0:4]
    poly[4] = za[::-1][0:4] @ zb[1:5] + zc[::-1][0:4] @ zd[1:5]
    poly[5] = za[::-1][0:3] @ zb[2:5] + zc[::-1][0:3] @ zd[2:5]
    poly[6] = za[::-1][0:2] @ zb[3:5] + zc[::-1][0:2] @ zd[3:5]
    poly[7] = za[4] * zb[4] + zc[4] * zd[4]

    rts = np.roots(poly)
    rts = np.real(rts[np.isreal(rts)])

    z0 = float(q[1])
    below = rts[rts < z0]
    above = rts[rts > z0]

    if below.size == 0 or above.size == 0:
        return None

    return (float(np.max(below)), float(np.min(above)))


def findz_analytical(
    ell: float,
    q: ZhuangEllipticityParams | np.ndarray,
    z_range: tuple[float, float] | None = None,
) -> float:
    """Analytically invert e(z) to find z using polynomial root finding."""
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    q = np.asarray(q, dtype=float)

    if not np.isfinite(ell) or ell <= 0:
        return np.nan

    if z_range is None:
        z_range = findz_analytical_range(q)
        if z_range is None:
            z_range = _find_zhuang_usable_range(q)
        if z_range is None:
            return np.nan

    R2 = (ell / q[0]) ** 2

    T = np.flip(np.repeat(range(5), 5).reshape((5, 5)))
    Lx = np.repeat(
        (q[4] / q[5]**4, q[3] / q[5]**3, 1 / q[5]**2, 0, 1),
        5,
    ).reshape((5, 5)).T
    Ly = np.repeat(
        (q[7] / q[8]**4, q[6] / q[8]**3, 1 / q[8]**2, 0, 1),
        5,
    ).reshape((5, 5)).T

    N = scipy.special.binom(T.T, T)
    S = (-np.ones((5, 5))) ** (T.T - T)

    cx = q[1] + q[2]
    cy = q[1] - q[2]

    P = np.sum(
        (R2 * Ly * cy ** (T.T - T) - Lx * cx ** (T.T - T)) * S * N,
        axis=1,
    )

    if np.any(np.isnan(P)):
        return np.nan

    rts = np.roots(P)
    rts = rts[np.isreal(rts)]

    if rts.size == 0:
        return np.nan

    rts_real = np.real(rts)
    in_range = rts_real[(rts_real >= z_range[0]) & (rts_real <= z_range[1])]

    if in_range.size == 0:
        return np.nan

    z0 = q[1]
    return float(in_range[np.argmin(np.abs(in_range - z0))])

def _find_zhuang_usable_range(q: np.ndarray, search_half_range_um: float = 5.0) -> tuple[float, float] | None:
    """Find the monotonic range of the ellipticity curve near focus.

    Returns (z_min, z_max) or None if range cannot be determined.
    Robust to invalid points: derivative/extrema detection is done on valid samples only.
    """
    q = np.asarray(q, dtype=float)
    analytic = findz_analytical_range(q)
    if analytic is not None:
        return analytic

    z0 = float(q[1])  # center of focus
    half = max(0.5, float(search_half_range_um))

    z_test = np.linspace(z0 - half, z0 + half, 3000)
    e_test = zhuang_ellipticity(z_test, q)

    valid = np.isfinite(e_test) & (e_test > 0)
    if valid.sum() < 10:
        return None

    zv = z_test[valid]
    ev = e_test[valid]

    # If valid range is fragmented, just work with the valid samples; extrema detection
    # still behaves well if you have enough points near focus.
    de = np.diff(ev)
    # protect against all-zero diffs
    if not np.any(np.isfinite(de)):
        return (float(zv[0]), float(zv[-1]))

    sign_changes = np.where(np.diff(np.sign(de)))[0]

    if sign_changes.size == 0:
        return (float(zv[0]), float(zv[-1]))

    extrema_z = zv[sign_changes + 1]
    below = extrema_z[extrema_z < z0]
    above = extrema_z[extrema_z > z0]

    z_lo = float(below[-1]) if below.size > 0 else float(zv[0])
    z_hi = float(above[0]) if above.size > 0 else float(zv[-1])

    if z_hi <= z_lo:
        return (float(zv[0]), float(zv[-1]))
    return (z_lo, z_hi)


def zhuang_usable_range(q: ZhuangEllipticityParams | np.ndarray) -> tuple[float, float] | None:
    """Public interface: get the usable Z range for a fitted model."""
    if isinstance(q, ZhuangEllipticityParams):
        q = q.to_array()
    return _find_zhuang_usable_range(np.asarray(q, dtype=float))


# ---------------------------------------------------------------------------
# Second-moment metric ↔ ellipticity bridge
# ---------------------------------------------------------------------------

def second_moment_error_from_ellipticity(ell: float) -> float:
    """Convert sigma_x/sigma_y ellipticity ratio to the second-moment error signal.

    The second-moment metric is (var_x - var_y)/(var_x + var_y).
    Since var ~ sigma^2, and ell = sigma_x/sigma_y:
      error = (ell^2 - 1) / (ell^2 + 1)
    """
    e2 = float(ell) ** 2
    return (e2 - 1.0) / (e2 + 1.0)


def ellipticity_from_second_moment_error(error: float) -> float:
    """Inverse: convert second-moment error signal to ellipticity ratio.

    error = (ell^2 - 1)/(ell^2 + 1) -> ell^2 = (1 + error)/(1 - error)
    """
    error = float(error)
    if error >= 1.0:
        return float("inf")
    if error <= -1.0:
        return 0.0
    return math.sqrt((1.0 + error) / (1.0 - error))


# ---------------------------------------------------------------------------
# Build a lookup table for fast runtime Z estimation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ZhuangLookupTable:
    """Precomputed lookup table mapping second-moment error signal -> z offset.

    Built from either:
      - a fitted Zhuang ellipticity model (theoretical), or
      - an empirical calibration sweep (recommended for control).

    The table stores Z OFFSETS relative to focus (focus = 0), making it independent
    of the absolute stage position at calibration time.
    """
    error_values: np.ndarray        # sorted ascending
    z_offsets: np.ndarray           # z offsets relative to focus (um), focus = 0
    z_range: tuple[float, float]    # usable range relative to focus (um)
    z_center: float = 0.0           # retained for backward compatibility only

    def __init__(
        self,
        error_values,
        z_offsets=None,
        z_range=(0.0, 0.0),
        z_center: float = 0.0,
        z_values_um=None,
    ) -> None:
        # Backward compatibility: older callers/tests used `z_values_um`.
        if z_offsets is None and z_values_um is not None:
            z_offsets = z_values_um
        if z_offsets is None:
            raise TypeError("ZhuangLookupTable requires `z_offsets` (or deprecated `z_values_um`).")
        self.error_values = np.asarray(error_values, dtype=float)
        self.z_offsets = np.asarray(z_offsets, dtype=float)
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.z_center = float(z_center)

    def error_to_z_offset_um(self, error: float) -> float:
        """Fast interpolated lookup: error signal -> z offset from focus in um."""
        if self.error_values.size < 2:
            return 0.0
        return float(np.interp(float(error), self.error_values, self.z_offsets))


def build_lookup_table(
    q: ZhuangEllipticityParams | np.ndarray,
    n_points: int = 1000,
) -> ZhuangLookupTable:
    """Build a theoretical lookup table from a fitted Zhuang model.

    Maps *theoretical* second-moment error signal to z offset. This is useful
    for diagnostics. For real-time control, prefer build_empirical_lookup_table()
    so the lookup matches the actual measured error signal from your ROI metric.

    Note: This function assumes best focus occurs where the theoretical error signal
    crosses 0 (i.e., ell = 1). In real systems, metric bias often shifts this; the
    empirical table avoids that problem.
    """
    if isinstance(q, ZhuangEllipticityParams):
        q_arr = q.to_array()
    else:
        q_arr = np.asarray(q, dtype=float)

    z_range = _find_zhuang_usable_range(q_arr)
    if z_range is None:
        raise ValueError("Cannot determine usable Z range from Zhuang parameters")

    z_vals = np.linspace(z_range[0], z_range[1], int(n_points))
    ell_vals = zhuang_ellipticity(z_vals, q_arr)
    error_vals = np.array([second_moment_error_from_ellipticity(float(e)) for e in ell_vals], dtype=float)

    valid = np.isfinite(error_vals) & np.isfinite(z_vals)
    error_vals = error_vals[valid]
    z_vals = z_vals[valid]

    if error_vals.size < 2:
        raise ValueError("Not enough valid points for lookup table")

    order = np.argsort(error_vals)
    error_sorted = error_vals[order]
    z_sorted = z_vals[order]

    # Focus via theoretical zero-crossing of error
    if (error_sorted[0] > 0) or (error_sorted[-1] < 0):
        # No zero crossing in this usable range -> choose mid-point as "focus"
        z_center_abs = float(np.median(z_sorted))
    else:
        z_center_abs = float(np.interp(0.0, error_sorted, z_sorted))

    z_offsets_rel = z_sorted - z_center_abs
    z_range_rel = (float(z_range[0]) - z_center_abs, float(z_range[1]) - z_center_abs)

    return ZhuangLookupTable(
        error_values=error_sorted,
        z_offsets=z_offsets_rel,
        z_range=z_range_rel,
        z_center=0.0,
    )


def build_empirical_lookup_table(
    z_data: np.ndarray,
    error_data: np.ndarray,
    z_focus: float,
    n_bins: int = 60,
    smooth_window: int = 7,
) -> ZhuangLookupTable:
    """Build a lookup table from actual calibration sweep measurements.

    Uses the *measured* second-moment error signal from the calibration sweep, so
    the lookup matches the same runtime astigmatic error metric your control loop
    computes. This avoids bias/scale mismatch relative to the ideal Gaussian model.

    The function bins and smooths the sweep data, then identifies the largest
    monotonic arm of the error curve that contains z_focus.

    Args:
        z_data: Z positions from calibration sweep.
        error_data: Measured second-moment error at each Z.
        z_focus: Z position of best focus (from visual/metric minimum/other method).
        n_bins: Number of bins for binning.
        smooth_window: Moving-average window size for smoothing.

    Returns:
        ZhuangLookupTable using empirical error values.
    """
    z_data = np.asarray(z_data, dtype=float)
    error_data = np.asarray(error_data, dtype=float)

    mask = np.isfinite(z_data) & np.isfinite(error_data)
    z_sel = z_data[mask]
    err_sel = error_data[mask]

    if z_sel.size < 4:
        raise ValueError(f"Only {z_sel.size} finite samples; need at least 4.")

    z_min, z_max = float(z_sel.min()), float(z_sel.max())
    if z_max <= z_min:
        raise ValueError("All Z values are identical")

    # Bin by Z and average
    n_bins = int(max(8, n_bins))
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_errors = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        in_bin = (z_sel >= bin_edges[i]) & (z_sel < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin |= (z_sel == bin_edges[i + 1])
        if np.any(in_bin):
            bin_errors[i] = float(np.mean(err_sel[in_bin]))

    # Interpolate empty bins
    valid = np.isfinite(bin_errors)
    if valid.sum() < 4:
        raise ValueError("Too few non-empty bins for empirical lookup")
    bin_errors = np.interp(bin_centers, bin_centers[valid], bin_errors[valid])

    # Smooth with moving average
    smooth_window = int(max(3, smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    padded = np.pad(bin_errors, smooth_window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[:n_bins]

    # Find extrema from smoothed derivative
    de = np.diff(smoothed)
    sign_changes = np.where(np.diff(np.sign(de)))[0]
    extrema_z = bin_centers[sign_changes + 1] if sign_changes.size > 0 else np.array([], dtype=float)

    below = extrema_z[extrema_z < z_focus]
    above = extrema_z[extrema_z > z_focus]
    seg_lo = float(below[-1]) if below.size > 0 else z_min
    seg_hi = float(above[0]) if above.size > 0 else z_max

    seg_mask = (bin_centers >= seg_lo) & (bin_centers <= seg_hi)
    seg_z = bin_centers[seg_mask]
    seg_err = smoothed[seg_mask]

    # If too small, fall back to the largest monotonic segment
    if seg_z.size < 6:
        segments = []
        start = 0
        for sc in sign_changes:
            segments.append((start, sc + 1))
            start = sc + 1
        segments.append((start, len(de)))
        best = max(segments, key=lambda s: s[1] - s[0])
        seg_mask_fb = np.zeros(n_bins, dtype=bool)
        seg_mask_fb[best[0]:min(best[1] + 1, n_bins)] = True
        seg_z = bin_centers[seg_mask_fb]
        seg_err = smoothed[seg_mask_fb]

    if seg_z.size < 2:
        raise ValueError("Not enough valid points for empirical lookup")

    # Enforce monotonicity via isotonic regression
    if seg_err[-1] >= seg_err[0]:
        seg_err_mono = _isotonic_increasing(seg_err)
    else:
        seg_err_mono = -_isotonic_increasing(-seg_err)

    # Sort by error for interpolation
    order = np.argsort(seg_err_mono)
    error_sorted = seg_err_mono[order]
    z_sorted = seg_z[order]

    # Determine center error at z_focus using monotone curve vs z
    # (interpolate in z-space first)
    # Ensure seg_z is increasing for interp; it is (bin_centers subset)
    z_focus_clip = float(np.clip(z_focus, float(seg_z.min()), float(seg_z.max())))
    error_at_focus = float(np.interp(z_focus_clip, seg_z, seg_err_mono))
    z_center = float(np.interp(error_at_focus, error_sorted, z_sorted))

    z_offsets_rel = z_sorted - z_center
    z_range_rel = (float(seg_z.min()) - z_center, float(seg_z.max()) - z_center)

    return ZhuangLookupTable(
        error_values=error_sorted,
        z_offsets=z_offsets_rel,
        z_range=z_range_rel,
        z_center=0.0,
    )

from __future__ import annotations

import numpy as np

from auto_focus.focus_metric import _gaussian7_python, fit_gaussian_fast, fit_gaussian_psf


def _synthetic_patch(size: int = 21) -> np.ndarray:
    yv, xv = np.indices((size, size), dtype=float)
    x0, y0 = 10.2, 9.7
    sigma_x, sigma_y = 2.4, 1.8
    amp, off = 180.0, 12.0
    g = amp * np.exp(-0.5 * (((xv - x0) / sigma_x) ** 2 + ((yv - y0) / sigma_y) ** 2)) + off
    return g


def test_fit_gaussian_fast_moment_only_returns_result() -> None:
    patch = _synthetic_patch()
    res = fit_gaussian_fast(patch, fast_mode=True)
    assert res is not None
    assert 0.0 <= res.r_squared <= 1.0
    assert res.fwhm > 0
    assert res.ellipticity > 0


def test_fit_gaussian_fast_refined_fit_returns_result() -> None:
    patch = _synthetic_patch()
    res = fit_gaussian_fast(patch, theta=0.0, fast_mode=False)
    assert res is not None
    assert res.r_squared > 0.5
    assert abs(res.x - 10.2) < 2.0
    assert abs(res.y - 9.7) < 2.0


def test_fit_gaussian_psf_reconstructs_axis_widths_consistently() -> None:
    q = np.array([15.2, 14.7, 3.9, 3256.7, 12.0, 1.44, 0.0], dtype=float)
    xv, yv = np.meshgrid(np.arange(31, dtype=float), np.arange(31, dtype=float))
    patch = _gaussian7_python(q, xv, yv)

    res = fit_gaussian_psf(patch, theta=0.0)

    assert res is not None
    assert abs(res.ellipticity - 1.44) < 0.05
    assert abs((res.sigma_x / res.sigma_y) - res.ellipticity) < 0.05

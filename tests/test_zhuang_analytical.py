from __future__ import annotations

import numpy as np

from auto_focus.zhuang import (
    ZhuangEllipticityParams,
    find_z_from_ellipticity,
    findz_analytical,
    findz_analytical_range,
    zhuang_ellipticity,
)


def _sample_params() -> ZhuangEllipticityParams:
    return ZhuangEllipticityParams(
        e0=1.1,
        z0=0.0,
        c=0.15,
        Ax=0.0,
        Bx=0.0,
        dx=0.4,
        Ay=0.0,
        By=0.0,
        dy=0.4,
    )


def test_findz_analytical_matches_numerical_inversion() -> None:
    q = _sample_params()
    z_range = findz_analytical_range(q)
    assert z_range is not None

    z_test = np.linspace(z_range[0] * 0.9, z_range[1] * 0.9, 25)
    for z in z_test:
        ell = float(zhuang_ellipticity(float(z), q))
        z_num = float(find_z_from_ellipticity(ell, q, z_range=z_range))
        z_ana = float(findz_analytical(ell, q, z_range=z_range))
        assert abs(z_num - z_ana) < 1e-3


def test_findz_analytical_range_contains_focus_center() -> None:
    q = _sample_params()
    z_range = findz_analytical_range(q)
    assert z_range is not None
    assert z_range[0] < q.z0 < z_range[1]

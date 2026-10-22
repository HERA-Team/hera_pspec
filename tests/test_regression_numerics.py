"""Numerical regression tests.

These pin the numbers hera_pspec produces for a set of fixed inputs (the
cutouts shipped in ``hera_pspec.data``) so that refactors such as the astropy
units migration can be checked against them. The inputs are built in
``tests/regression_cases.py`` and the expected values live in
``tests/data/regression/*.npz``. Read ``tests/data/README.md`` before
regenerating them.
"""

import astropy.cosmology.units as cu
import numpy as np
import pytest
from astropy import units
from astropy.cosmology import FLRW

from hera_pspec import conversions
from hera_pspec import uvpspec_utils as uvputils
from tests.regression_cases import CASES, compute_case, read_legacy_uvp


def _assert_value_matches(case_name: str, key: str, actual, expected) -> None:
    case = CASES[case_name]
    actual = np.asarray(actual)
    label = f"{case_name}[{key}]"
    if expected.dtype.kind == "U":
        assert actual.dtype.kind == "U", f"{label}: expected a string"
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        return

    assert actual.shape == expected.shape, (
        f"{label}: shape {actual.shape} != expected {expected.shape}"
    )
    assert np.iscomplexobj(actual) == np.iscomplexobj(expected), (
        f"{label}: complex/real mismatch"
    )
    rtol = case.rtol_for(key)
    # Elements that come out of large cancellations (e.g. near-zero imaginary
    # parts, high-delay bins next to foreground-dominated ones) carry an
    # absolute rounding error set by the largest values in the array, so the
    # absolute tolerance scales with the array's maximum.
    scale = np.max(np.abs(expected)) if expected.size else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * scale, err_msg=label
    )


@pytest.mark.parametrize("case_name", list(CASES))
def test_regression_case(case_name: str) -> None:
    """Compare a case's current outputs with its stored fixture."""
    fixture_path = CASES[case_name].fixture_path
    assert fixture_path.exists(), (
        f"missing regression fixture {fixture_path}; see tests/data/README.md"
    )
    actual = compute_case(case_name)
    with np.load(fixture_path) as stored:
        expected = {key: stored[key] for key in stored.files}

    assert sorted(actual) == sorted(expected), (
        f"{case_name}: output keys changed; "
        f"missing {sorted(set(expected) - set(actual))}, "
        f"new {sorted(set(actual) - set(expected))}"
    )
    for key in sorted(expected):
        _assert_value_matches(case_name, key, actual[key], expected[key])


def test_legacy_uvp_file_metadata() -> None:
    """The 2019 UVPSpec file keeps its units and cosmology on read.

    Its legacy unit strings ("h^-3 Mpc^3", "UNCALIB") are read into the
    canonical astropy forms.
    """
    uvp = read_legacy_uvp()
    assert uvp.norm_units == "Mpc3 / littleh3"
    assert uvp.vis_units == "uncalib"
    assert uvp.deltasq is False
    assert uvp.norm_unit == units.Mpc**3 / cu.littleh**3
    assert uvp.vis_unit == uvputils.uncalib
    assert uvp.units == uvputils.uncalib**2 * units.Mpc**3 / cu.littleh**3
    assert isinstance(uvp.cosmo, FLRW)
    assert uvp.cosmo.is_equivalent(conversions.DEFAULT_COSMOLOGY)
    params = conversions.params_from_cosmology(uvp.cosmo)
    assert sorted(params) == ["H0", "Om_L", "Om_M", "Om_b", "Om_c", "Om_k"]
    np.testing.assert_allclose(
        [params[key] for key in ("H0", "Om_L", "Om_M", "Om_b", "Om_c")],
        [67.27, 0.6844, 0.31353, 0.04911, 0.26442],
        rtol=1e-15,
    )
    np.testing.assert_allclose(params["Om_k"], 0.00207, rtol=1e-12)
    np.testing.assert_allclose(uvp.cosmo.h, 0.6727, rtol=1e-15)
    np.testing.assert_allclose(uvp.scalar_array[0, 0], 3748370988.976227, rtol=1e-15)
    np.testing.assert_allclose(
        uvp.data_array[0][0, :2, 0],
        [3.20990444e10 - 3.65748031e-06j, 2.16634414e11 - 3.70675734e-05j],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        uvp.get_kparas(0)[1].to_value(cu.littleh / units.Mpc),
        -0.16775573211036096,
        rtol=1e-12,
    )

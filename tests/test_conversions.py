import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
import scipy.integrate as integrate
from astropy import constants
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck18, wCDM

from hera_pspec import conversions


def test_f21_is_the_21cm_rest_frequency() -> None:
    """F21 is the 21cm rest frequency in Hz and f2z maps it to z = 0."""
    assert conversions.F21 == 1.420405751e9 * u.Hz
    assert conversions.f2z(conversions.F21) == 0.0


# Parameters of the v0.x default cosmology, the dictionary stored in the "cosmo"
# attribute of HDF5 files written by hera_pspec v0.x.
OLD_DEFAULT_COSMO_PARAMS = {
    "Om_L": 0.6844,
    "Om_b": 0.04911,
    "Om_c": 0.26442,
    "Om_M": 0.31353,
    "Om_k": 0.0020700000000000163,
    "H0": 67.27,
}


def test_legacy_cosmology_reference_values() -> None:
    """LEGACY_COSMOLOGY at z = 10 gives the numbers hera_pspec v0.x gave (h^-1 Mpc)."""
    cosmo = conversions.LEGACY_COSMOLOGY
    z = 10.0
    np.testing.assert_allclose(conversions.f2z(100e6), 13.20405751, rtol=1e-12)
    np.testing.assert_allclose(
        conversions.z2f(z).to_value(u.Hz), 129127795.54545455, rtol=1e-12
    )
    np.testing.assert_allclose(
        conversions.dRperp_dtheta(z, cosmo).value, 6510.2536925709637, rtol=1e-9
    )
    np.testing.assert_allclose(
        conversions.dRpara_df(z, cosmo).value, 1.2487605057418872e-05, rtol=1e-9
    )
    np.testing.assert_allclose(
        conversions.X2Y(z, cosmo).value, 529.26719942209002, rtol=1e-9
    )
    np.testing.assert_allclose(
        conversions.bl_to_kperp(z, cosmo).value, 0.00041570092391078579, rtol=1e-9
    )
    np.testing.assert_allclose(
        conversions.tau_to_kpara(z, cosmo).value, 503153.74952115043, rtol=1e-9
    )


# --- astropy cosmology backend ---------------------------------------------
#
# The reference values below come from the formulas hera_pspec v0.x used
# before it delegated to astropy (Hogg 1999 with scipy quadrature), written
# out here so the comparison does not go through the code under test.

ASTROPY_REDSHIFTS = [0.1, 0.5, 1.0, 2.0, 5.0, 7.9, 10.0, 20.0, 30.0]

# (Om_L, Om_b, Om_c, H0): the default (slightly open), a flat and a closed
# model (Om_k = 0 and -0.1).
ASTROPY_COSMO_PARAMS = {
    "default": (0.68440, 0.04911, 0.26442, 67.27),
    "flat": (0.75, 0.05, 0.2, 70.0),
    "closed": (0.8, 0.05, 0.25, 60.0),
}

LITTLEH_MPC = u.Mpc / cu.littleh


def _old_E(Om_L, Om_M, Om_k, z):
    return np.sqrt(Om_M * (1 + z) ** 3 + Om_k * (1 + z) ** 2 + Om_L)


def _old_DH(H0, little_h):
    return (
        constants.c.to_value(u.km / u.s) / 100.0
        if little_h
        else constants.c.to_value(u.km / u.s) / H0
    )


def _old_DC(Om_L, Om_M, Om_k, H0, z, little_h):
    d = integrate.quad(lambda zz: 1 / _old_E(Om_L, Om_M, Om_k, zz), 0, z)[0]
    return d * _old_DH(H0, little_h)


def _old_DM(Om_L, Om_M, Om_k, H0, z, little_h):
    DH = _old_DH(H0, little_h)
    DC = _old_DC(Om_L, Om_M, Om_k, H0, z, little_h)
    if Om_k > 0:
        return DH * np.sinh(np.sqrt(Om_k) * DC / DH) / np.sqrt(Om_k)
    if Om_k < 0:
        return DH * np.sin(np.sqrt(np.abs(Om_k)) * DC / DH) / np.sqrt(np.abs(Om_k))
    return DC


def _old_quantities(Om_L, Om_M, Om_k, H0, z, little_h):
    """v0.x outputs, as plain floats, in (h^-1) Mpc units."""
    f21 = conversions.F21.to_value(u.Hz)
    DM = _old_DM(Om_L, Om_M, Om_k, H0, z, little_h)
    dRpara_df = (
        (1 + z) ** 2.0 / _old_E(Om_L, Om_M, Om_k, z) * _old_DH(H0, little_h) / f21
    )
    return {
        "E": _old_E(Om_L, Om_M, Om_k, z),
        "DC": _old_DC(Om_L, Om_M, Om_k, H0, z, little_h),
        "DM": DM,
        "DA": DM / (1 + z),
        "dRperp_dtheta": DM,
        "dRpara_df": dRpara_df,
        "X2Y": DM**2 * dRpara_df,
        "bl_to_kperp": 2 * np.pi / (DM * (constants.c.value / (f21 / (z + 1)))),
        "tau_to_kpara": 2 * np.pi / dRpara_df,
    }


def _make_cosmo(name):
    """Return the LambdaCDM (no radiation) and the (Om_L, Om_M, Om_k, H0) of a case."""
    Om_L, Om_b, Om_c, H0 = ASTROPY_COSMO_PARAMS[name]
    Om_M = Om_b + Om_c
    lcdm = LambdaCDM(H0=H0, Om0=Om_M, Ode0=Om_L, Ob0=Om_b)
    return lcdm, (Om_L, Om_M, 1 - Om_L - Om_M, H0)


class TestAstropyBackend:
    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    @pytest.mark.parametrize("little_h", [True, False])
    @pytest.mark.parametrize("z", ASTROPY_REDSHIFTS)
    def test_helpers_match_old_formulas(self, name, little_h, z) -> None:
        """Check the module-level helpers reproduce the pre-astropy formulas."""
        lcdm, (Om_L, Om_M, Om_k, H0) = _make_cosmo(name)
        expected = _old_quantities(Om_L, Om_M, Om_k, H0, z, little_h)
        length_unit = LITTLEH_MPC if little_h else u.Mpc
        equiv = cu.with_H0(lcdm.H0)
        rtol = 1e-12

        distances = {
            "DC": lcdm.comoving_distance(z),
            "DM": lcdm.comoving_transverse_distance(z),
            "DA": lcdm.angular_diameter_distance(z),
        }
        for key, distance in distances.items():
            np.testing.assert_allclose(
                distance.to_value(length_unit, equiv), expected[key], rtol=rtol
            )
        np.testing.assert_allclose(lcdm.efunc(z), expected["E"], rtol=rtol)

        results = {
            "dRperp_dtheta": (
                conversions.dRperp_dtheta(z, lcdm, length_unit),
                length_unit / u.rad,
            ),
            "dRpara_df": (
                conversions.dRpara_df(z, lcdm, length_unit),
                length_unit / u.Hz,
            ),
            "X2Y": (
                conversions.X2Y(z, lcdm, length_unit),
                length_unit**3 / (u.sr * u.Hz),
            ),
            "bl_to_kperp": (
                conversions.bl_to_kperp(z, lcdm, length_unit),
                1 / length_unit / u.m,
            ),
            "tau_to_kpara": (
                conversions.tau_to_kpara(z, lcdm, length_unit),
                1 / length_unit / u.s,
            ),
        }
        for key, (result, unit) in results.items():
            assert result.unit == unit, key
            np.testing.assert_allclose(
                result.value, expected[key], rtol=rtol, err_msg=key
            )

    def test_default_cosmology_is_planck18(self) -> None:
        """Check DEFAULT_COSMOLOGY is astropy's Planck18 itself."""
        assert conversions.DEFAULT_COSMOLOGY is Planck18

    def test_legacy_cosmology(self) -> None:
        """LEGACY_COSMOLOGY has the v0.x default parameters and no radiation."""
        legacy = conversions.LEGACY_COSMOLOGY
        assert legacy.Tcmb0.value == 0
        np.testing.assert_allclose(legacy.Ok0, 0.00207, rtol=1e-10)
        assert legacy.H0.value == OLD_DEFAULT_COSMO_PARAMS["H0"]
        assert legacy.Om0 == OLD_DEFAULT_COSMO_PARAMS["Om_M"]
        assert legacy.Ode0 == OLD_DEFAULT_COSMO_PARAMS["Om_L"]
        assert legacy.Ob0 == OLD_DEFAULT_COSMO_PARAMS["Om_b"]

    def test_f2z_quantity(self) -> None:
        """Check f2z accepts Quantities in any frequency unit."""
        np.testing.assert_allclose(
            conversions.f2z(150.0 * u.MHz), conversions.f2z(150e6), rtol=1e-15
        )
        freqs = np.array([100.0, 150.0, 200.0]) * u.MHz
        z = conversions.f2z(freqs)
        assert not isinstance(z, u.Quantity)
        np.testing.assert_allclose(
            z, conversions.F21.to_value(u.Hz) / freqs.to_value(u.Hz) - 1, rtol=1e-15
        )
        np.testing.assert_allclose(
            conversions.f2z(150e6),
            conversions.F21.to_value(u.Hz) / 150e6 - 1,
            rtol=1e-15,
        )

    def test_z2f_returns_hz(self) -> None:
        """Check z2f returns a Quantity in Hz that inverts f2z."""
        freq = conversions.z2f(10.0)
        assert freq.unit == u.Hz
        np.testing.assert_allclose(freq.value, 129127795.54545455, rtol=1e-15)
        np.testing.assert_allclose(
            conversions.f2z(conversions.z2f(np.array([6.0, 8.0]))),
            [6.0, 8.0],
            rtol=1e-14,
        )


class TestAstropyCosmologyObjects:
    def test_params_from_legacy_cosmology(self) -> None:
        """params_from_cosmology(LEGACY_COSMOLOGY) is the old default dict."""
        params = conversions.params_from_cosmology(conversions.LEGACY_COSMOLOGY)
        assert params == OLD_DEFAULT_COSMO_PARAMS
        assert list(params) == list(OLD_DEFAULT_COSMO_PARAMS)
        # same string, so files keep the exact v0.x "cosmo" attribute
        assert str(params) == str(OLD_DEFAULT_COSMO_PARAMS)
        assert all(type(value) is float for value in params.values())

    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    def test_params_round_trip(self, name) -> None:
        """cosmology_from_params inverts params_from_cosmology for no-radiation models."""
        lcdm, (Om_L, Om_M, Om_k, H0) = _make_cosmo(name)
        params = conversions.params_from_cosmology(lcdm)
        assert list(params) == ["Om_L", "Om_b", "Om_c", "Om_M", "Om_k", "H0"]
        assert params["Om_L"] == Om_L
        assert params["Om_M"] == Om_M
        assert params["H0"] == H0
        np.testing.assert_allclose(params["Om_k"], Om_k, atol=1e-15)
        assert conversions.cosmology_from_params(params).is_equivalent(lcdm)

    def test_params_from_cosmology_without_ob0(self) -> None:
        """A cosmology without Ob0 uses the fixed baryon split."""
        params = conversions.params_from_cosmology(FlatLambdaCDM(H0=70.0, Om0=0.3))
        assert params["Om_M"] == 0.3
        assert params["Om_b"] == 0.3 * 0.156635
        assert params["Om_c"] == 0.3 * 0.843364
        assert params["H0"] == 70.0
        np.testing.assert_allclose(params["Om_k"], 0.0, atol=1e-15)

    def test_as_astropy_cosmology_flrw(self) -> None:
        """An astropy cosmology is returned as is."""
        cosmo = wCDM(H0=67.27, Om0=0.31353, Ode0=0.6844, w0=-0.9)
        assert conversions.as_astropy_cosmology(cosmo) is cosmo
        default = conversions.DEFAULT_COSMOLOGY
        assert conversions.as_astropy_cosmology(default) is default

    def test_params_from_planck18_have_no_radiation(self) -> None:
        """The dict of Planck18 keeps its parameters but not its radiation."""
        params = conversions.params_from_cosmology(Planck18)
        assert params["H0"] == 67.66
        assert params["Om_M"] == Planck18.Om0
        assert params["Om_L"] == Planck18.Ode0
        assert params["Om_b"] == Planck18.Ob0
        # the radiation density ends up in Om_k
        np.testing.assert_allclose(
            params["Om_k"], Planck18.Ogamma0 + Planck18.Onu0, rtol=1e-12
        )
        rebuilt = conversions.cosmology_from_params(params)
        assert rebuilt.Tcmb0.value == 0
        assert not rebuilt.is_equivalent(Planck18)
        # the approximation documented in params_from_cosmology
        z = np.linspace(5.0, 25.0, 41)
        ratio = rebuilt.comoving_transverse_distance(z) / (
            Planck18.comoving_transverse_distance(z)
        )
        assert np.all(np.abs(ratio.value - 1) < 4e-3)

    def test_as_astropy_cosmology_rejects_params_dict(self) -> None:
        """A cosmology parameter dictionary is not an accepted cosmology."""
        with pytest.raises(TypeError, match="astropy.cosmology.FLRW"):
            conversions.as_astropy_cosmology(OLD_DEFAULT_COSMO_PARAMS)

    def test_cosmology_from_params_rejects_inconsistent_om_k(self) -> None:
        """A dictionary astropy cannot represent (inconsistent Om_k) raises ValueError."""
        params = dict(OLD_DEFAULT_COSMO_PARAMS, Om_k=0.05)
        with pytest.raises(ValueError, match="Om_k"):
            conversions.cosmology_from_params(params)

    def test_cosmology_from_params_rejects_large_om_b(self) -> None:
        """Om_b larger than Om_M cannot be converted to an astropy cosmology."""
        params = dict(OLD_DEFAULT_COSMO_PARAMS, Om_M=0.03, Om_k=None)
        with pytest.raises(ValueError, match="Om_b"):
            conversions.cosmology_from_params(params)

    @pytest.mark.parametrize(
        "value", [[0.7, 0.05], "Planck18", 1.0], ids=["list", "str", "float"]
    )
    def test_as_astropy_cosmology_rejects_other_types(self, value) -> None:
        """Other types raise TypeError."""
        with pytest.raises(TypeError, match="astropy.cosmology.FLRW"):
            conversions.as_astropy_cosmology(value)


class TestResolveLengthUnit:
    def test_default(self) -> None:
        assert conversions.resolve_length_unit() == u.Mpc / cu.littleh

    @pytest.mark.parametrize(
        "unit", [u.Mpc, u.Mpc / cu.littleh, "Mpc", u.kpc, u.Gpc / cu.littleh]
    )
    def test_length_units_pass(self, unit) -> None:
        assert conversions.resolve_length_unit(unit) == u.Unit(unit)

    @pytest.mark.parametrize(
        "unit_string", ["Mpc / littleh", b"Mpc / littleh"], ids=["str", "bytes"]
    )
    def test_littleh_string(self, unit_string) -> None:
        """A littleh unit string, as str or bytes, parses to Mpc / littleh."""
        unit = conversions.resolve_length_unit(unit_string)
        assert unit == u.Mpc / cu.littleh

    @pytest.mark.parametrize("unit", [u.s, u.Mpc**2, cu.littleh])
    def test_bad_unit_raises(self, unit) -> None:
        with pytest.raises(ValueError, match="length_unit must be a length"):
            conversions.resolve_length_unit(unit)


class TestRedshiftCache:
    """The distance helpers cache by redshift, keyed on the cosmology."""

    @pytest.mark.parametrize(
        "helper",
        [
            conversions.dRperp_dtheta,
            conversions.dRpara_df,
            conversions.bl_to_kperp,
            conversions.tau_to_kpara,
        ],
    )
    def test_matches_array_path(self, helper) -> None:
        # scalar and array calls (both cached) give the same numbers
        zs = np.array([6.0, 8.0, 10.0])
        scalar = u.Quantity([helper(z) for z in zs])
        np.testing.assert_allclose(scalar.value, helper(zs).to_value(scalar.unit))

    def test_equal_cosmologies_share_entries(self) -> None:
        conversions.bl_to_kperp.cache_clear()
        first = conversions.bl_to_kperp(8.0, conversions.LEGACY_COSMOLOGY)
        same = LambdaCDM(H0=67.27, Om0=0.31353, Ode0=0.6844, Ob0=0.04911, name="x")
        assert conversions.bl_to_kperp(8.0, same) == first
        other = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)
        assert conversions.bl_to_kperp(8.0, other) != first

    def test_returned_value_is_a_copy(self) -> None:
        value = conversions.dRpara_df(8.0)
        reference = value.copy()
        value *= 2
        assert conversions.dRpara_df(8.0) == reference

    def test_length_unit_is_part_of_the_key(self) -> None:
        littleh = conversions.dRperp_dtheta(8.0, length_unit=u.Mpc / cu.littleh)
        mpc = conversions.dRperp_dtheta(8.0, length_unit=u.Mpc)
        assert littleh.unit != mpc.unit
        np.testing.assert_allclose(
            littleh.to_value(
                u.Mpc / u.rad, cu.with_H0(conversions.DEFAULT_COSMOLOGY.H0)
            ),
            mpc.to_value(u.Mpc / u.rad),
        )

    def test_array_result_is_a_copy(self) -> None:
        zs = np.array([6.0, 8.0])
        value = conversions.bl_to_kperp(zs)
        reference = value.copy()
        value[0] *= 2
        np.testing.assert_array_equal(conversions.bl_to_kperp(zs), reference)

    def test_large_arrays_bypass_the_cache(self) -> None:
        zs = np.linspace(6.0, 10.0, conversions._MAX_CACHED_Z_SIZE + 1)
        np.testing.assert_allclose(
            conversions.tau_to_kpara(zs)[[0, -1]].value,
            u.Quantity(
                [conversions.tau_to_kpara(6.0), conversions.tau_to_kpara(10.0)]
            ).value,
        )

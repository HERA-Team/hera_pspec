import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
import scipy.integrate as integrate
from astropy import constants
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, wCDM

from hera_pspec import conversions


def _cosmo_conversions(*args, **kwargs) -> conversions.Cosmo_Conversions:
    """Build the deprecated Cosmo_Conversions, checking that it warns."""
    with pytest.warns(DeprecationWarning, match="Cosmo_Conversions is deprecated"):
        return conversions.Cosmo_Conversions(*args, **kwargs)


def _from_astropy(cosmo) -> conversions.Cosmo_Conversions:
    """Cosmo_Conversions.from_astropy, checking that it warns."""
    with pytest.warns(DeprecationWarning, match="Cosmo_Conversions is deprecated"):
        return conversions.Cosmo_Conversions.from_astropy(cosmo)


@pytest.fixture
def cosmo_h100() -> conversions.Cosmo_Conversions:
    """Cosmology fixture with H0=100 used as the reference for distance/conversion checks."""
    return _cosmo_conversions(
        Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=100.0, Om_M=None, Om_k=None
    )


@pytest.fixture
def cosmo_h25() -> conversions.Cosmo_Conversions:
    """Cosmology fixture with H0=25.12, used to check little_h scaling against cosmo_h100."""
    return _cosmo_conversions(
        Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=25.12, Om_M=None, Om_k=None
    )


def test_units() -> None:
    """Check that SI and CGS unit systems report the speed of light in the right units."""
    si = conversions.units()
    cgs = conversions.cgs_units()
    np.testing.assert_almost_equal(si.c, 2.99792458e8)
    np.testing.assert_almost_equal(cgs.c, 2.99792458e10)


def test_units_match_astropy_constants() -> None:
    """Check that the float constants are the astropy.constants values."""
    si = conversions.units
    cgs = conversions.cgs_units
    assert si.c == constants.c.value
    assert si.ckm == constants.c.to_value(u.km / u.s)
    assert si.G == constants.G.value
    assert si.kb == constants.k_B.value
    assert si.hp == constants.h.value
    assert si.sb == constants.sigma_sb.value
    assert si.H0_to_SI == (1 * u.km / u.s / u.Mpc).to_value(1 / u.s)
    assert cgs.c == constants.c.cgs.value
    assert cgs.kb == constants.k_B.cgs.value
    for value in (si.c, si.ckm, si.G, si.kb, si.hp, si.sb, si.H0_to_SI, cgs.c, cgs.kb):
        assert type(value) is float
    # hera_pspec's own 21cm constants are unchanged
    assert si.f21 == 1.420405751e9
    assert si.w21 == 0.211061140542
    assert conversions.F21 == si.f21 * u.Hz


class TestCosmoConversions:
    def test_init(self, cosmo_h100: conversions.Cosmo_Conversions) -> None:
        """Check that a default instance exposes expected attributes and that constructor kwargs are applied."""
        # test parameters exist
        cosmo_h100.Om_b
        cosmo_h100.H0

        # test parameters get fed to class
        C = _cosmo_conversions(H0=25.5)
        np.testing.assert_almost_equal(C.H0, 25.5)

    @pytest.mark.parametrize(
        "method,args,kwargs,expected",
        [
            ("f2z", (100e6,), {}, 13.20405751),
            ("f2z", (0.1,), {"ghz": True}, 13.20405751),
            ("z2f", (10.0,), {}, 129127795.54545455),
            ("z2f", (10.0,), {"ghz": True}, 0.12912779554545455),
            ("E", (10.0,), {}, 20.450997530682947),
            ("DC", (10.0,), {}, 6499.708111027144),
            ("DC", (10.0,), {"little_h": False}, 6499.708111027144),
            ("DM", (10.0,), {}, 6510.2536925709637),
            ("DA", (10.0,), {}, 591.84124477917851),
            ("dRperp_dtheta", (10.0,), {}, 6510.2536925709637),
            ("dRpara_df", (10.0,), {}, 1.2487605057418872e-05),
            ("dRpara_df", (10.0,), {"ghz": True}, 12487.605057418872),
            ("X2Y", (10.0,), {}, 529.26719942209002),
        ],
    )
    def test_distances(
        self,
        cosmo_h100: conversions.Cosmo_Conversions,
        method: str,
        args: tuple,
        kwargs: dict,
        expected: float,
    ) -> None:
        """Check that distance/frequency conversion methods match reference values for H0=100."""
        result = getattr(cosmo_h100, method)(*args, **kwargs)
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "method,args,kwargs,expected",
        [
            ("f2z", (100e6,), {}, 13.20405751),
            ("z2f", (10.0,), {}, 129127795.54545455),
            ("E", (10.0,), {}, 20.450997530682947),
            ("DC", (10.0,), {"little_h": True}, 6499.708111027144),
            ("DM", (10.0,), {"little_h": True}, 6510.2536925709637),
            ("DA", (10.0,), {"little_h": True}, 591.84124477917851),
            ("dRperp_dtheta", (10.0,), {"little_h": True}, 6510.2536925709637),
            ("dRpara_df", (10.0,), {"little_h": True}, 1.2487605057418872e-05),
            ("dRpara_df", (10.0,), {"ghz": True, "little_h": True}, 12487.605057418872),
            ("X2Y", (10.0,), {"little_h": True}, 529.26719942209002),
        ],
    )
    def test_little_h_true_gives_h0_independent_results(
        self,
        cosmo_h25: conversions.Cosmo_Conversions,
        method: str,
        args: tuple,
        kwargs: dict,
        expected: float,
    ) -> None:
        """Check that little_h=True results are H0-independent."""
        result = getattr(cosmo_h25, method)(*args, **kwargs)
        np.testing.assert_almost_equal(result, expected)

    def test_little_h_false_scales_distances_by_h0_ratio(
        self, cosmo_h25: conversions.Cosmo_Conversions
    ) -> None:
        """Check that little_h=False distances scale by the H0 ratio relative to h=1."""
        result = cosmo_h25.DC(10.0, little_h=False)
        np.testing.assert_almost_equal(result, 6499.708111027144 * 100.0 / 25.12)

    def test_params(self, cosmo_h100: conversions.Cosmo_Conversions) -> None:
        """Check that get_params returns a dict matching the instance's attributes."""
        params = cosmo_h100.get_params()
        np.testing.assert_almost_equal(params["Om_L"], cosmo_h100.Om_L)

    def test_kpara_kperp(self, cosmo_h100: conversions.Cosmo_Conversions) -> None:
        """Check that bl_to_kperp and tau_to_kpara match reference values."""
        bl2kperp = cosmo_h100.bl_to_kperp(10.0, little_h=True)
        tau2kpara = cosmo_h100.tau_to_kpara(10.0, little_h=True)
        np.testing.assert_almost_equal(bl2kperp, 0.00041570092391078579)
        np.testing.assert_almost_equal(tau2kpara, 503153.74952115043)


# --- astropy cosmology backend ---------------------------------------------
#
# The reference values below come from the formulas Cosmo_Conversions used
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
    return conversions.units.ckm / 100.0 if little_h else conversions.units.ckm / H0


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
    """Old Cosmo_Conversions outputs, as plain floats, in (h^-1) Mpc units."""
    f21 = conversions.units.f21
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
        "bl_to_kperp": 2 * np.pi / (DM * (conversions.units.c / (f21 / (z + 1)))),
        "tau_to_kpara": 2 * np.pi / dRpara_df,
    }


def _make_cosmo(name):
    Om_L, Om_b, Om_c, H0 = ASTROPY_COSMO_PARAMS[name]
    return _cosmo_conversions(Om_L=Om_L, Om_b=Om_b, Om_c=Om_c, H0=H0)


class TestAstropyBackend:
    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    @pytest.mark.parametrize("little_h", [True, False])
    @pytest.mark.parametrize("z", ASTROPY_REDSHIFTS)
    def test_helpers_match_old_formulas(self, name, little_h, z) -> None:
        """Check the module-level helpers reproduce the pre-astropy formulas."""
        cosmo = _make_cosmo(name)
        lcdm = cosmo.to_astropy()
        expected = _old_quantities(
            cosmo.Om_L, cosmo.Om_M, cosmo.Om_k, cosmo.H0, z, little_h
        )
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

    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    @pytest.mark.parametrize("little_h", [True, False])
    @pytest.mark.parametrize("z", ASTROPY_REDSHIFTS)
    def test_methods_match_old_formulas(self, name, little_h, z) -> None:
        """Check the Cosmo_Conversions methods still give the old numbers."""
        cosmo = _make_cosmo(name)
        expected = _old_quantities(
            cosmo.Om_L, cosmo.Om_M, cosmo.Om_k, cosmo.H0, z, little_h
        )
        for key, value in expected.items():
            if key == "E":
                result = cosmo.E(z)
            else:
                result = getattr(cosmo, key)(z, little_h=little_h)
            assert not isinstance(result, u.Quantity)
            np.testing.assert_allclose(result, value, rtol=1e-12, err_msg=key)

    @pytest.mark.parametrize("z", [0.5, 7.9, 20.0])
    def test_inconsistent_om_k_keeps_old_formulas(self, z) -> None:
        """Check an explicit inconsistent Om_k still works via the fallback."""
        cosmo = _cosmo_conversions(Om_k=0.05)
        assert cosmo.Om_k == 0.05
        expected = _old_quantities(cosmo.Om_L, cosmo.Om_M, 0.05, cosmo.H0, z, True)
        for key, value in expected.items():
            if key == "E":
                result = cosmo.E(z)
            else:
                result = getattr(cosmo, key)(z)
            np.testing.assert_allclose(result, value, rtol=1e-12, err_msg=key)
        with pytest.raises(ValueError, match="Om_k"):
            cosmo.to_astropy()

    def test_default_cosmology(self) -> None:
        """Check DEFAULT_COSMOLOGY matches Cosmo_Conversions() defaults."""
        default = conversions.DEFAULT_COSMOLOGY
        cosmo = _cosmo_conversions()
        assert default.Tcmb0.value == 0
        np.testing.assert_allclose(default.Ok0, 0.00207, rtol=1e-10)
        np.testing.assert_allclose(default.H0.value, cosmo.H0, rtol=1e-15)
        z = np.array(ASTROPY_REDSHIFTS)
        for method, astropy_method in [
            ("DC", "comoving_distance"),
            ("DM", "comoving_transverse_distance"),
            ("DA", "angular_diameter_distance"),
        ]:
            np.testing.assert_allclose(
                getattr(default, astropy_method)(z).to_value(u.Mpc),
                getattr(cosmo, method)(z, little_h=False),
                rtol=1e-12,
            )
        np.testing.assert_allclose(conversions.X2Y(z).value, cosmo.X2Y(z), rtol=1e-12)

    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    def test_astropy_round_trip(self, name) -> None:
        """Check to_astropy/from_astropy preserve the cosmological parameters."""
        cosmo = _make_cosmo(name)
        lcdm = cosmo.to_astropy()
        assert isinstance(lcdm, LambdaCDM)
        assert lcdm.Tcmb0.value == 0
        assert lcdm.Ob0 == cosmo.Om_b
        new = _from_astropy(lcdm)
        original = cosmo.get_params()
        params = new.get_params()
        assert sorted(params) == sorted(original)
        for key, value in original.items():
            np.testing.assert_allclose(params[key], value, rtol=1e-12, atol=1e-15)

    def test_from_astropy_default(self) -> None:
        """Check DEFAULT_COSMOLOGY converts to Cosmo_Conversions()."""
        new = _from_astropy(conversions.DEFAULT_COSMOLOGY)
        for key, value in _cosmo_conversions().get_params().items():
            np.testing.assert_allclose(new.get_params()[key], value, rtol=1e-12)

    def test_from_astropy_without_ob0(self) -> None:
        """Check a missing Ob0 is split with the usual baryon fraction."""
        new = _from_astropy(FlatLambdaCDM(H0=70.0, Om0=0.3))
        assert new.Om_M == 0.3
        np.testing.assert_allclose(new.Om_b, 0.3 * 0.156635, rtol=1e-15)
        np.testing.assert_allclose(new.Om_c, 0.3 * 0.843364, rtol=1e-15)
        np.testing.assert_allclose(new.Om_L, 0.7, rtol=1e-15)
        np.testing.assert_allclose(new.Om_k, 0.0, atol=1e-15)

    def test_from_astropy_rejects_radiation(self) -> None:
        """Check a cosmology with radiation (Tcmb0 > 0) is rejected."""
        with pytest.raises(ValueError, match="Tcmb0"):
            conversions.Cosmo_Conversions.from_astropy(
                LambdaCDM(H0=67.27, Om0=0.31353, Ode0=0.6844, Tcmb0=2.725)
            )

    def test_from_astropy_rejects_other_models(self) -> None:
        """Check a non-LambdaCDM cosmology is rejected."""
        with pytest.raises(TypeError, match="LambdaCDM"):
            conversions.Cosmo_Conversions.from_astropy(
                wCDM(H0=67.27, Om0=0.31353, Ode0=0.6844, w0=-0.9)
            )

    def test_to_astropy_rejects_large_om_b(self) -> None:
        """Check Om_b > Om_M cannot be converted."""
        cosmo = _cosmo_conversions(Om_M=0.03)
        with pytest.raises(ValueError, match="Om_b"):
            cosmo.to_astropy()

    @pytest.mark.parametrize("little_h", [True, False])
    @pytest.mark.parametrize("method", ["E", "DC", "DM", "DA", "dRpara_df", "X2Y"])
    def test_array_redshifts(self, method, little_h) -> None:
        """Check array z gives the same values as element-wise scalar calls."""
        cosmo = _cosmo_conversions()
        z = np.array(ASTROPY_REDSHIFTS)
        kwargs = {} if method == "E" else {"little_h": little_h}
        result = getattr(cosmo, method)(z, **kwargs)
        expected = [getattr(cosmo, method)(zz, **kwargs) for zz in z]
        assert np.shape(result) == z.shape
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_f2z_quantity(self) -> None:
        """Check f2z accepts Quantities in any frequency unit."""
        np.testing.assert_allclose(
            conversions.f2z(150.0 * u.MHz), conversions.f2z(150e6), rtol=1e-15
        )
        freqs = np.array([100.0, 150.0, 200.0]) * u.MHz
        z = conversions.f2z(freqs)
        assert not isinstance(z, u.Quantity)
        np.testing.assert_allclose(
            z, _cosmo_conversions().f2z(freqs.to_value(u.Hz)), rtol=1e-15
        )
        np.testing.assert_allclose(
            conversions.f2z(150e6), conversions.units.f21 / 150e6 - 1, rtol=1e-15
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
    def test_cosmo_conversions_deprecated(self) -> None:
        """Building Cosmo_Conversions warns and points to astropy.cosmology."""
        with pytest.warns(DeprecationWarning) as record:
            conversions.Cosmo_Conversions()
        message = str(record[0].message)
        assert "astropy.cosmology" in message
        assert "DEFAULT_COSMOLOGY" in message

    def test_params_from_default_cosmology(self) -> None:
        """params_from_cosmology(DEFAULT_COSMOLOGY) is the old default dict."""
        params = conversions.params_from_cosmology(conversions.DEFAULT_COSMOLOGY)
        old = _cosmo_conversions().get_params()
        assert params == old
        assert list(params) == list(old)
        # same string, so files keep the exact v0.x "cosmo" attribute
        assert str(params) == str(old)
        assert all(type(value) is float for value in params.values())

    @pytest.mark.parametrize("name", list(ASTROPY_COSMO_PARAMS))
    def test_params_from_cosmology_matches_get_params(self, name) -> None:
        """params_from_cosmology agrees with Cosmo_Conversions.get_params."""
        cosmo = _make_cosmo(name)
        params = conversions.params_from_cosmology(cosmo.to_astropy())
        assert list(params) == list(cosmo.get_params())
        for key, value in cosmo.get_params().items():
            np.testing.assert_allclose(params[key], value, rtol=1e-12, atol=1e-15)

    def test_params_from_cosmology_without_ob0(self) -> None:
        """A cosmology without Ob0 uses the fixed baryon split."""
        params = conversions.params_from_cosmology(FlatLambdaCDM(H0=70.0, Om0=0.3))
        assert params["Om_M"] == 0.3
        assert params["Om_b"] == 0.3 * 0.156635
        assert params["Om_c"] == 0.3 * 0.843364
        assert params["H0"] == 70.0
        np.testing.assert_allclose(params["Om_k"], 0.0, atol=1e-15)
        assert params == _from_astropy(FlatLambdaCDM(H0=70.0, Om0=0.3)).get_params()

    def test_as_astropy_cosmology_flrw(self) -> None:
        """An astropy cosmology is returned as is."""
        cosmo = wCDM(H0=67.27, Om0=0.31353, Ode0=0.6844, w0=-0.9)
        assert conversions.as_astropy_cosmology(cosmo) is cosmo
        default = conversions.DEFAULT_COSMOLOGY
        assert conversions.as_astropy_cosmology(default) is default

    def test_as_astropy_cosmology_dict(self) -> None:
        """A parameter dict gives one shared, equivalent astropy object."""
        params = conversions.params_from_cosmology(conversions.DEFAULT_COSMOLOGY)
        first = conversions.as_astropy_cosmology(params)
        second = conversions.as_astropy_cosmology(dict(params))
        assert isinstance(first, LambdaCDM)
        assert first is second
        assert first.is_equivalent(conversions.DEFAULT_COSMOLOGY)

    def test_as_astropy_cosmology_cosmo_conversions(self) -> None:
        """A Cosmo_Conversions is converted with a DeprecationWarning."""
        old = _cosmo_conversions(H0=70.0, Om_L=0.7, Om_b=0.05, Om_c=0.25)
        with pytest.warns(DeprecationWarning, match="astropy.cosmology"):
            cosmo = conversions.as_astropy_cosmology(old)
        assert isinstance(cosmo, LambdaCDM)
        assert cosmo.is_equivalent(old.to_astropy())

    def test_as_astropy_cosmology_inconsistent_om_k(self) -> None:
        """A Cosmo_Conversions astropy cannot represent raises ValueError."""
        old = _cosmo_conversions(Om_k=0.05)
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Om_k"):
            conversions.as_astropy_cosmology(old)

    @pytest.mark.parametrize("value", [[0.7, 0.05], "Planck18", 1.0])
    def test_as_astropy_cosmology_rejects_other_types(self, value) -> None:
        """Other types raise TypeError."""
        with pytest.raises(TypeError, match="astropy.cosmology.FLRW"):
            conversions.as_astropy_cosmology(value)

    def test_length_unit_from_little_h(self) -> None:
        """little_h maps to Mpc / littleh, otherwise Mpc."""
        assert conversions.length_unit_from_little_h(True) == u.Mpc / cu.littleh
        assert conversions.length_unit_from_little_h(False) == u.Mpc

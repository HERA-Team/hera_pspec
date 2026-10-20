import numpy as np
import pytest

from hera_pspec import conversions


@pytest.fixture
def cosmo_h100() -> conversions.Cosmo_Conversions:
    """Cosmology fixture with H0=100 used as the reference for distance/conversion checks."""
    return conversions.Cosmo_Conversions(
        Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=100.0, Om_M=None, Om_k=None
    )


@pytest.fixture
def cosmo_h25() -> conversions.Cosmo_Conversions:
    """Cosmology fixture with H0=25.12, used to check little_h scaling against cosmo_h100."""
    return conversions.Cosmo_Conversions(
        Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=25.12, Om_M=None, Om_k=None
    )


# (method, args, kwargs, expected) reference values for both test cosmologies
DISTANCE_CASES = [
    ("f2z", (100e6,), {}, 13.20405751),
    ("f2z", (0.1,), {"ghz": True}, 13.20405751),
    ("z2f", (10.0,), {}, 129127795.54545455),
    ("z2f", (10.0,), {"ghz": True}, 0.12912779554545455),
    ("E", (10.0,), {}, 20.450997530682947),
    ("DC", (10.0,), {}, 6499.708111027144),
    ("DC", (10.0,), {"little_h": True}, 6499.708111027144),
    ("DM", (10.0,), {}, 6510.2536925709637),
    ("DM", (10.0,), {"little_h": True}, 6510.2536925709637),
    ("DA", (10.0,), {}, 591.84124477917851),
    ("DA", (10.0,), {"little_h": True}, 591.84124477917851),
    ("dRperp_dtheta", (10.0,), {}, 6510.2536925709637),
    ("dRperp_dtheta", (10.0,), {"little_h": True}, 6510.2536925709637),
    ("dRpara_df", (10.0,), {}, 1.2487605057418872e-05),
    ("dRpara_df", (10.0,), {"little_h": True}, 1.2487605057418872e-05),
    ("dRpara_df", (10.0,), {"ghz": True}, 12487.605057418872),
    ("dRpara_df", (10.0,), {"ghz": True, "little_h": True}, 12487.605057418872),
    ("X2Y", (10.0,), {}, 529.26719942209002),
    ("X2Y", (10.0,), {"little_h": True}, 529.26719942209002),
]


def _case_id(method: str, args: tuple, kwargs: dict, expected: float) -> str:
    return "-".join([method] + [f"{k}={v}" for k, v in kwargs.items()])


class TestCosmoConversions:
    def test_init(self) -> None:
        """Check that constructor kwargs are applied."""
        C = conversions.Cosmo_Conversions(H0=25.5)
        np.testing.assert_almost_equal(C.H0, 25.5)

    def test_om_m_sets_baryon_and_cdm_fractions(self) -> None:
        """A custom Om_M is split into Om_b and Om_c that add up to it."""
        default = conversions.Cosmo_Conversions()
        assert default.Om_M == default.Om_b + default.Om_c
        # an Om_M consistent with the default Om_b and Om_c leaves them unchanged
        C = conversions.Cosmo_Conversions(Om_M=default.Om_M)
        assert (C.Om_b, C.Om_c) == (default.Om_b, default.Om_c)

        C = conversions.Cosmo_Conversions(Om_M=0.5)
        assert C.Om_M == 0.5
        np.testing.assert_allclose(C.Om_b + C.Om_c, 0.5, rtol=1e-5)
        # the baryon fraction of matter is that of the default cosmology
        np.testing.assert_allclose(
            C.Om_b / C.Om_M, default.Om_b / default.Om_M, rtol=1e-5
        )

    @pytest.mark.parametrize("cosmo_name", ["cosmo_h100", "cosmo_h25"])
    @pytest.mark.parametrize(
        "method,args,kwargs,expected",
        DISTANCE_CASES,
        ids=[_case_id(*case) for case in DISTANCE_CASES],
    )
    def test_distances(
        self,
        cosmo_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
        expected: float,
        request: pytest.FixtureRequest,
    ) -> None:
        """Check that distance/frequency conversion methods match reference values.

        Distances are in Mpc/h by default (little_h=True), so they do not depend
        on H0.
        """
        cosmo = request.getfixturevalue(cosmo_name)
        result = getattr(cosmo, method)(*args, **kwargs)
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "cosmo_name,H0",
        [("cosmo_h100", 100.0), ("cosmo_h25", 25.12)],
        ids=["cosmo_h100", "cosmo_h25"],
    )
    def test_little_h_false_scales_distances_by_h0_ratio(
        self, cosmo_name: str, H0: float, request: pytest.FixtureRequest
    ) -> None:
        """Check that little_h=False distances scale by the H0 ratio relative to h=1."""
        cosmo = request.getfixturevalue(cosmo_name)
        result = cosmo.DC(10.0, little_h=False)
        np.testing.assert_almost_equal(result, 6499.708111027144 * 100.0 / H0)

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


# Flat, open (Om_k > 0) and closed (Om_k < 0) cosmologies, with H0 away from
# 100 so that h^-1 Mpc and Mpc differ. (The default cosmology is very slightly
# open, Om_k = 0.002.)
CURVATURE_COSMOLOGIES = {
    "flat": dict(Om_L=1 - 0.04911 - 0.26442, Om_k=0.0),
    "open": dict(Om_L=0.5, H0=70.0),
    "closed": dict(Om_L=0.9, H0=60.0),
}


@pytest.fixture(params=list(CURVATURE_COSMOLOGIES), scope="module")
def curved_cosmo(request) -> conversions.Cosmo_Conversions:
    cosmo = conversions.Cosmo_Conversions(**CURVATURE_COSMOLOGIES[request.param])
    sign = {"flat": 0, "open": 1, "closed": -1}[request.param]
    assert np.sign(cosmo.Om_k) == sign
    return cosmo


class TestCosmoPhysics:
    """Check the cosmological distances against astropy and analytic limits,
    and the delay/baseline to k conversions against their definitions."""

    @pytest.mark.parametrize("z", [0.01, 1.0, 6.0, 10.0])
    def test_distances_match_astropy(
        self, curved_cosmo: conversions.Cosmo_Conversions, z: float
    ) -> None:
        """E(z) and the comoving, transverse comoving and angular diameter
        distances agree with astropy's LambdaCDM (in Mpc for little_h=False,
        and h^-1 Mpc = h * Mpc otherwise), including the sinh/sin branches of
        the transverse distance for open and closed cosmologies."""
        C = curved_cosmo
        lcdm = C.lcdm
        np.testing.assert_allclose(C.E(z), lcdm.efunc(z), rtol=1e-12)
        for method, astropy_method in [
            ("DC", lcdm.comoving_distance),
            ("DM", lcdm.comoving_transverse_distance),
            ("DA", lcdm.angular_diameter_distance),
        ]:
            expected = astropy_method(z).to("Mpc").value
            np.testing.assert_allclose(
                getattr(C, method)(z, little_h=False), expected, rtol=1e-8
            )
            np.testing.assert_allclose(
                getattr(C, method)(z, little_h=True), expected * C.h, rtol=1e-8
            )

    def test_small_redshift_limit(
        self, curved_cosmo: conversions.Cosmo_Conversions
    ) -> None:
        """At small z the comoving distance is the Hubble law, D = c z / H0,
        with corrections of order z."""
        z = 1e-4
        c_km = conversions.units.ckm
        np.testing.assert_allclose(
            curved_cosmo.DC(z, little_h=False), c_km * z / curved_cosmo.H0, rtol=1e-3
        )
        np.testing.assert_allclose(
            curved_cosmo.DC(z, little_h=True), c_km * z / 100.0, rtol=1e-3
        )

    @pytest.mark.parametrize("little_h", [True, False])
    @pytest.mark.parametrize("z", [1.0, 8.0])
    def test_conversion_factors(
        self, curved_cosmo: conversions.Cosmo_Conversions, z: float, little_h: bool
    ) -> None:
        """The conversions from frequency, angle, delay and baseline length to
        comoving distances and wavenumbers follow from the distances:

        - dRpara_df is the derivative of the comoving distance with respect to
          the observed 21 cm frequency;
        - X2Y = DM^2 * dRpara_df converts sr Hz to comoving volume;
        - a delay tau is the Fourier dual of frequency, so
          k_para = 2 pi tau / dRpara_df;
        - a baseline of length b measures angular wavenumber 2 pi b / lambda,
          so k_perp = 2 pi b / (lambda DM).
        """
        C = curved_cosmo
        freq = C.z2f(z)
        np.testing.assert_allclose(C.f2z(freq), z, rtol=1e-12)
        dfreq = 1e3  # Hz
        dDC_df = (
            C.DC(C.f2z(freq - dfreq), little_h=little_h)
            - C.DC(C.f2z(freq + dfreq), little_h=little_h)
        ) / (2 * dfreq)
        dRpara_df = C.dRpara_df(z, little_h=little_h)
        np.testing.assert_allclose(dRpara_df, dDC_df, rtol=1e-6)
        DM = C.DM(z, little_h=little_h)
        np.testing.assert_allclose(C.dRperp_dtheta(z, little_h=little_h), DM)
        np.testing.assert_allclose(
            C.X2Y(z, little_h=little_h), DM**2 * dRpara_df, rtol=1e-12
        )
        np.testing.assert_allclose(
            C.tau_to_kpara(z, little_h=little_h) * dRpara_df, 2 * np.pi, rtol=1e-12
        )
        wavelength = conversions.units.c / freq
        np.testing.assert_allclose(
            C.bl_to_kperp(z, little_h=little_h),
            2 * np.pi / (wavelength * DM),
            rtol=1e-12,
        )

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


def test_units() -> None:
    """Check that SI and CGS unit systems report the speed of light in the right units."""
    si = conversions.units()
    cgs = conversions.cgs_units()
    np.testing.assert_almost_equal(si.c, 2.99792458e8)
    np.testing.assert_almost_equal(cgs.c, 2.99792458e10)


class TestCosmoConversions:
    def test_init(self, cosmo: conversions.Cosmo_Conversions) -> None:
        """Check that a default instance exposes expected attributes and that constructor kwargs are applied."""
        # test parameters exist
        cosmo.Om_b
        cosmo.H0

        # test parameters get fed to class
        C = conversions.Cosmo_Conversions(H0=25.5)
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

    def test_little_h_false_scales_distances_by_h0_ratio(self, cosmo_h25: conversions.Cosmo_Conversions) -> None:
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

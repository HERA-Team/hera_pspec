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

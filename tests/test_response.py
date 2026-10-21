"""Tests for user-defined response matrices (#328): hera_pspec.response and
PSpecData.get_response.

Identities are checked at 1e-12 relative to the largest element (design note
section 6, tests T2, T4, T6).
"""

import copy

import astropy.cosmology.units as cu
import numpy as np
import pytest
from astropy import units as u
from scipy.interpolate import interp1d

from hera_pspec import DelayResponse, PSpecData, conversions, response, utils

IDENTITY_RTOL = 1e-12
SPW = (400, 440)
BEAM_POL = -5  # ee


def assert_close(actual, expected, rtol=IDENTITY_RTOL, scale=None):
    """Compare at rtol, with an absolute floor of rtol times ``scale``
    (default: the largest element of ``expected``)."""
    if scale is None:
        scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale)


@pytest.fixture(scope="module")
def unflagged_uvd(uvd_zen_2458116):
    uvd = copy.deepcopy(uvd_zen_2458116)
    uvd.flag_array[:] = False
    return uvd


@pytest.fixture
def ds(unflagged_uvd, beam_nf_dipole):
    return PSpecData(
        dsets=[unflagged_uvd, unflagged_uvd], wgts=[None, None], beam=beam_nf_dipole
    )


KEY1 = (0, (23, 24), "ee")
KEY2 = (1, (24, 25), "ee")


def configure(ds, ndlys, taper, weighting, symmetric_taper, extension):
    ds.set_spw(SPW, ndlys=ndlys)
    ds.set_taper(taper)
    ds.set_weighting(weighting)
    ds.set_symmetric_taper(False)
    ds.set_filter_extension(extension)
    ds.set_symmetric_taper(symmetric_taper)
    ds.clear_cache()


def explicit_response(R1, R2, q_est, c_tensor):
    """H_ab = (1/2) tr[R_1^dagger Q^est_a R_2 C_{,b}], one trace at a time."""
    n_dlys = q_est.shape[0]
    H = np.zeros((n_dlys, n_dlys), dtype=complex)
    for a in range(n_dlys):
        E_a = 0.5 * np.conj(R1).T @ q_est[a] @ R2
        for b in range(n_dlys):
            H[a, b] = np.trace(E_a @ c_tensor[b])
    return H


def q_alt(n_dlys, n_freqs, extension=(0, 0)):
    return utils.get_Q_alt_tensor(
        n_dlys, n_freqs=n_freqs, n_extend=sum(extension), phase_correction=extension[0]
    )


# configurations: (ndlys, taper, weighting, symmetric_taper, filter extension)
LEGACY_CONFIGS = [
    (40, "none", "identity", True, (0, 0)),
    (20, "blackman-harris", "identity", True, (0, 0)),
    (20, "blackman-harris", "identity", False, (5, 3)),
    (40, "none", "iC", False, (4, 4)),
    (30, "blackman-harris", "iC", True, (0, 0)),
]


@pytest.mark.filterwarnings("ignore:Poorly conditioned covariance:UserWarning")
class TestLegacyConventions:
    """T2: the legacy kernels reproduce get_H / get_G.

    The iC configurations invert a rank-deficient empirical covariance, which
    warns (as in test_pspecdata's TestFisherMatrices).
    """

    @pytest.mark.parametrize("config", LEGACY_CONFIGS)
    @pytest.mark.parametrize("sampling", [False, True])
    def test_legacy_matches_get_H(self, ds, config, sampling) -> None:
        configure(ds, *config)
        H = ds.get_H(KEY1, KEY2, sampling=sampling)
        H_resp = ds.get_response(
            KEY1, KEY2, DelayResponse(convention="legacy", sampling=sampling)
        )
        assert_close(H_resp, H)
        if sampling:
            # with delta-function bins H is G
            assert_close(H_resp, ds.get_G(KEY1, KEY2))

    @pytest.mark.parametrize("config", [c for c in LEGACY_CONFIGS if c[4] == (0, 0)])
    @pytest.mark.parametrize("sampling", [False, True])
    def test_legacy_exact_matches_exact_norm(self, ds, config, sampling) -> None:
        # exact_norm get_G / get_H raise with filter extensions (F3), so only
        # configurations without extensions are compared
        configure(ds, *config)
        if sampling:
            expected = ds.get_G(KEY1, KEY2, exact_norm=True, pol=BEAM_POL)
        else:
            expected = ds.get_H(KEY1, KEY2, exact_norm=True, pol=BEAM_POL)
        H_resp = ds.get_response(
            KEY1,
            KEY2,
            DelayResponse(convention="legacy-exact", sampling=sampling),
            pol=BEAM_POL,
        )
        assert_close(H_resp, expected)

    def test_legacy_without_beam(self, unflagged_uvd) -> None:
        ds = PSpecData(dsets=[unflagged_uvd, unflagged_uvd], wgts=[None, None])
        configure(ds, 20, "none", "identity", True, (0, 0))
        H_resp = ds.get_response(KEY1, KEY2, DelayResponse(convention="legacy"))
        assert_close(H_resp, ds.get_H(KEY1, KEY2))

    def test_fully_flagged_gives_identity(self, ds, unflagged_uvd) -> None:
        uvd = copy.deepcopy(unflagged_uvd)
        uvd.flag_array[:] = True
        ds_flagged = PSpecData(dsets=[uvd, uvd], wgts=[None, None])
        configure(ds_flagged, 20, "none", "identity", True, (0, 0))
        H = ds_flagged.get_H(KEY1, KEY2)
        H_resp = ds_flagged.get_response(KEY1, KEY2, DelayResponse(convention="legacy"))
        np.testing.assert_array_equal(H, 0.5 * np.eye(20))
        np.testing.assert_array_equal(H_resp, H)


class TestToeplitzResponse:
    """T4: the Toeplitz algorithm equals the explicit tr(E_a C_{,b}) loop."""

    @pytest.mark.parametrize(
        "n_dlys, n_freqs, extension",
        [(12, 12, (0, 0)), (8, 16, (3, 2)), (7, 15, (0, 4))],
    )
    @pytest.mark.parametrize("user_estimator", [False, True])
    def test_random_complex_weights(
        self, n_dlys, n_freqs, extension, user_estimator
    ) -> None:
        rng = np.random.default_rng(42)
        n_ext = n_freqs + sum(extension)
        R1 = rng.normal(size=(n_freqs, n_ext)) + 1j * rng.normal(size=(n_freqs, n_ext))
        R2 = rng.normal(size=(n_freqs, n_ext)) + 1j * rng.normal(size=(n_freqs, n_ext))
        kernel = rng.normal(size=(n_ext, n_ext))
        if user_estimator:
            q_est = rng.normal(size=(n_dlys, n_freqs, n_freqs)) + 1j * rng.normal(
                size=(n_dlys, n_freqs, n_freqs)
            )
        else:
            q_est = q_alt(n_dlys, n_freqs)
        c_tensor = q_alt(n_dlys, n_freqs, extension) * kernel
        expected = explicit_response(R1, R2, q_est, c_tensor)

        H = response.toeplitz_response(
            R1, R2, kernel, n_dlys, q_est if user_estimator else None
        )
        assert_close(H, expected)
        # the generic dense path agrees too
        assert_close(response.tensor_response(R1, R2, c_tensor, q_est), expected)

    def test_estimator_phases_match_q_alt(self) -> None:
        for n_dlys, n_freqs in [(10, 10), (5, 10), (7, 9)]:
            m = response.estimator_phases(n_dlys, n_freqs)
            q_est = np.conj(m)[:, :, None] * m[:, None, :]
            assert_close(q_est, q_alt(n_dlys, n_freqs))


class UserResponse:
    """C_{,b} = Q^alt_b ⊙ K for a given float kernel, in a chosen unit."""

    def __init__(self, kernel, unit, extension=(0, 0)):
        self.kernel = kernel
        self.unit = unit
        self.extension = extension
        self.calls = 0

    def matrices(self, freqs, delays, beam, pol, cosmo, length_unit):
        self.calls += 1
        assert freqs.unit == u.Hz and delays.unit == u.s
        n_freqs = freqs.size - sum(self.extension)
        c_tensor = q_alt(delays.size, n_freqs, self.extension) * self.kernel
        return c_tensor * self.unit


def cosmological_kernel(ds, length_unit):
    delay_response = DelayResponse()
    spw = ds.get_spw(include_extension=True)
    return delay_response.kernel(
        ds.freqs[spw[0] : spw[1]],
        ds._delays_s(),
        ds.primary_beam,
        BEAM_POL,
        ds.primary_beam.cosmo,
        length_unit,
    )


class TestUserResponse:
    """T6 (part): units of user responses, q_matrices, memory cap."""

    @pytest.mark.parametrize("extension", [(0, 0), (3, 2)])
    def test_matches_delay_response(self, ds, extension) -> None:
        configure(ds, 20, "blackman-harris", "identity", extension == (0, 0), extension)
        kernel = cosmological_kernel(ds, u.Mpc)
        user = UserResponse(kernel, u.Mpc**-3, extension)
        H_user = ds.get_response(KEY1, KEY2, user, pol=BEAM_POL, length_unit=u.Mpc)
        H_delay = ds.get_response(
            KEY1, KEY2, DelayResponse(), pol=BEAM_POL, length_unit=u.Mpc
        )
        assert_close(H_user, H_delay)

    def test_equivalent_units(self, ds, beam_nf_dipole) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        kernel = cosmological_kernel(ds, u.Mpc)
        H_mpc = ds.get_response(
            KEY1, KEY2, UserResponse(kernel, u.Mpc**-3), BEAM_POL, length_unit=u.Mpc
        )
        # the same response written in 1/Gpc^3 is 1e9 larger numerically
        H_gpc = ds.get_response(
            KEY1,
            KEY2,
            UserResponse(kernel * 1e9, u.Gpc**-3),
            BEAM_POL,
            length_unit=u.Mpc,
        )
        assert_close(H_gpc, H_mpc)
        # in (Mpc/h)^-3: 1 Mpc^-3 = h^-3 (h/Mpc)^3
        h = beam_nf_dipole.cosmo.h
        H_littleh = ds.get_response(
            KEY1,
            KEY2,
            UserResponse(kernel, u.Mpc**-3),
            BEAM_POL,
            length_unit=u.Mpc / cu.littleh,
        )
        assert_close(H_littleh, H_mpc / h**3)

    def test_non_equivalent_unit_raises(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        kernel = np.ones((40, 40))
        with pytest.raises(ValueError, match="not equivalent"):
            ds.get_response(KEY1, KEY2, UserResponse(kernel, u.Mpc**-2), BEAM_POL)

    def test_non_quantity_raises(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        with pytest.raises(TypeError, match="Quantity"):
            ds.get_response(KEY1, KEY2, UserResponse(np.ones((40, 40)), 1.0), BEAM_POL)

    def test_wrong_shape_raises(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))

        class BadShape:
            def matrices(self, freqs, delays, beam, pol, cosmo, length_unit):
                return np.ones((3, 40, 40)) / length_unit**3

        with pytest.raises(ValueError, match="shape"):
            ds.get_response(KEY1, KEY2, BadShape(), BEAM_POL)

    def test_not_a_response_raises(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        with pytest.raises(TypeError, match="matrices"):
            ds.get_response(KEY1, KEY2, object(), BEAM_POL)

    def test_memory_cap(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        user = UserResponse(np.ones((40, 40)), u.Mpc**-3)
        with pytest.raises(ValueError, match="max_tensor_bytes"):
            ds.get_response(KEY1, KEY2, user, BEAM_POL, max_tensor_bytes=10_000)
        assert user.calls == 0
        # raising the cap allows it
        ds.get_response(KEY1, KEY2, user, BEAM_POL, max_tensor_bytes=10**8)
        assert user.calls == 1

    def test_user_response_called_once_per_spw_and_pol(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        user = UserResponse(np.ones((40, 40)), u.Mpc**-3)
        for _ in range(3):
            ds.get_response(KEY1, KEY2, user, BEAM_POL)
        assert user.calls == 1
        # another window gets its own entry; coming back is served from the
        # cache (the R matrices are not rebuilt by set_spw, so the kernels are
        # requested directly here)
        littleh = u.Mpc / cu.littleh
        other = UserResponse(np.ones((30, 30)), u.Mpc**-3)
        ds.set_spw((400, 430), ndlys=15)
        ds._response_kernel(other, BEAM_POL, littleh)
        ds.set_spw(SPW, ndlys=20)
        ds._response_kernel(user, BEAM_POL, littleh)
        assert (user.calls, other.calls) == (1, 1)
        ds.clear_cache()
        ds.get_response(KEY1, KEY2, user, BEAM_POL)
        assert user.calls == 2

    def test_needs_beam(self, unflagged_uvd) -> None:
        ds = PSpecData(dsets=[unflagged_uvd, unflagged_uvd], wgts=[None, None])
        configure(ds, 20, "none", "identity", True, (0, 0))
        with pytest.raises(ValueError, match="beam"):
            ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        with pytest.raises(ValueError, match="beam"):
            ds.get_response(
                KEY1, KEY2, UserResponse(np.ones((40, 40)), u.Mpc**-3), BEAM_POL
            )


class TestQMatrices:
    """User estimator matrices Q^est."""

    def test_q_alt_reproduces_defaults(self, ds) -> None:
        configure(ds, 20, "blackman-harris", "identity", True, (0, 0))
        q_est = ds.get_Q_alt_tensor()
        assert_close(ds.q_hat(KEY1, KEY2, q_matrices=q_est), ds.q_hat(KEY1, KEY2))
        assert_close(
            ds.get_unnormed_E(KEY1, KEY2, q_matrices=q_est),
            ds.get_unnormed_E(KEY1, KEY2),
        )
        legacy = DelayResponse(convention="legacy")
        assert_close(
            ds.get_response(KEY1, KEY2, legacy, q_matrices=q_est),
            ds.get_response(KEY1, KEY2, legacy),
        )

    def test_callable_and_quantity(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        q_est = ds.get_Q_alt_tensor()
        seen = {}

        def make_q(freqs, delays):
            seen["freqs"], seen["delays"] = freqs, delays
            return q_est * u.dimensionless_unscaled

        assert_close(ds.q_hat(KEY1, KEY2, q_matrices=make_q), ds.q_hat(KEY1, KEY2))
        assert seen["freqs"].unit == u.Hz and seen["freqs"].size == 40
        assert seen["delays"].unit == u.ns and seen["delays"].size == 20

    @pytest.mark.filterwarnings("ignore:Poorly conditioned covariance:UserWarning")
    def test_user_estimator_matches_explicit_trace(self, ds) -> None:
        configure(ds, 10, "none", "iC", False, (2, 3))
        rng = np.random.default_rng(1)
        q_est = rng.normal(size=(10, 40, 40))
        q_est = q_est + q_est.transpose(0, 2, 1)
        H = ds.get_response(
            KEY1, KEY2, DelayResponse(convention="legacy"), q_matrices=q_est
        )
        c_tensor = q_alt(10, 40, (2, 3)) * np.sinc(
            np.subtract.outer(np.arange(45), np.arange(45)) / 45.0
        )
        expected = explicit_response(ds.R(KEY1), ds.R(KEY2), q_est, c_tensor)
        assert_close(H, expected)

    def test_errors(self, ds) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        q_est = ds.get_Q_alt_tensor()
        with pytest.raises(ValueError, match="allow_fft"):
            ds.q_hat(KEY1, KEY2, allow_fft=True, q_matrices=q_est)
        with pytest.raises(ValueError, match="exact_norm"):
            ds.q_hat(KEY1, KEY2, exact_norm=True, q_matrices=q_est)
        with pytest.raises(ValueError, match="exact_norm"):
            ds.get_unnormed_E(KEY1, KEY2, exact_norm=True, q_matrices=q_est)
        with pytest.raises(ValueError, match="shape"):
            ds.q_hat(KEY1, KEY2, q_matrices=q_est[:5])
        with pytest.raises(ValueError, match="dimensionless"):
            ds.q_hat(KEY1, KEY2, q_matrices=q_est * u.s)
        with pytest.raises(ValueError, match="own estimator"):
            ds.get_response(
                KEY1,
                KEY2,
                DelayResponse(convention="legacy-exact"),
                pol=BEAM_POL,
                q_matrices=q_est,
            )


class TestDelayResponseKernel:
    def test_invalid_options(self) -> None:
        for kwargs in [
            {"sampling": "yes"},
            {"beam": "gauss"},
            {"x2y": "mean"},
            {"bin_width": 3},
            {"convention": "exact"},
        ]:
            with pytest.raises(ValueError, match="DelayResponse"):
                DelayResponse(**kwargs)

    def test_hashable_and_comparable(self) -> None:
        assert DelayResponse() == DelayResponse()
        assert (
            len({DelayResponse(), DelayResponse(), DelayResponse(sampling=True)}) == 2
        )
        assert DelayResponse(convention="legacy").is_legacy
        assert not DelayResponse().is_legacy

    def test_cosmological_kernel_diagonal(self, beam_nf_dipole) -> None:
        # K_ii = Delta_eta Omega_pp / Omega_p^2 / X2Y at nu_i (design note 2.1),
        # computed independently from the beam integrals
        freqs = np.linspace(140e6, 150e6, 40, endpoint=False)
        delays = utils.get_delays(freqs, n_dlys=20)
        cosmo = beam_nf_dipole.cosmo
        kernel = DelayResponse().kernel(
            freqs, delays, beam_nf_dipole, BEAM_POL, cosmo, u.Mpc
        )
        ratio = interp1d(
            beam_nf_dipole.beam_freqs / 1e6,
            beam_nf_dipole.power_beam_sq_int(BEAM_POL)
            / beam_nf_dipole.power_beam_int(BEAM_POL) ** 2,
            kind="quadratic",
            fill_value="extrapolate",
        )(freqs / 1e6)
        x2y = np.array(
            [
                conversions.X2Y(conversions.f2z(f), cosmo, u.Mpc).to_value(
                    u.Mpc**3 / (u.sr * u.Hz)
                )
                for f in freqs
            ]
        )
        delta_eta = 1.0 / (20 * (freqs[1] - freqs[0]))
        assert_close(np.diag(kernel), delta_eta * ratio / x2y)
        # geometric mean off the diagonal; sinc of the lag in units of Ndlys
        lag = np.subtract.outer(np.arange(40), np.arange(40))
        expected = (
            delta_eta
            * np.sqrt(np.outer(ratio, ratio))
            / np.sqrt(np.outer(x2y, x2y))
            * np.sinc(lag / 20)
        )
        assert_close(kernel, expected)

    def test_x2y_and_bin_width_options(self, beam_nf_dipole) -> None:
        freqs = np.linspace(140e6, 150e6, 40, endpoint=False)
        delays = utils.get_delays(freqs, n_dlys=20)
        cosmo = beam_nf_dipole.cosmo
        args = (freqs, delays, beam_nf_dipole, BEAM_POL, cosmo, u.Mpc)
        geometric = DelayResponse().kernel(*args)
        midpoint = DelayResponse(x2y="midpoint").kernel(*args)
        center = DelayResponse(x2y="center").kernel(*args)
        # the diagonal is the same for geometric and midpoint X2Y
        assert_close(np.diag(midpoint), np.diag(geometric))
        assert np.max(np.abs(midpoint / geometric - 1)) < 1e-3
        x2y_unit = u.Mpc**3 / (u.sr * u.Hz)
        x2y_centre = conversions.X2Y(
            conversions.f2z(0.5 * (freqs[0] + freqs[-1])), cosmo, u.Mpc
        ).to_value(x2y_unit)
        x2y = np.array(
            [
                conversions.X2Y(conversions.f2z(f), cosmo, u.Mpc).to_value(x2y_unit)
                for f in freqs
            ]
        )
        assert_close(np.diag(geometric) / np.diag(center), x2y_centre / x2y)
        # bin width Nfreqs: Delta_eta and the sinc use 40 instead of 20
        nfreqs = DelayResponse(bin_width="nfreqs").kernel(*args)
        lag = np.subtract.outer(np.arange(40), np.arange(40))
        assert_close(
            nfreqs, geometric / np.sinc(lag / 20) * np.sinc(lag / 40) * 20 / 40
        )
        # sampled bins: S = 1, Delta_eta the delay spacing
        sampled = DelayResponse(sampling=True).kernel(*args)
        assert_close(sampled, geometric / np.sinc(lag / 20))

    @pytest.mark.parametrize("n_dlys", [40, 20, 10, 8])
    def test_bins_tile_eta(self, n_dlys) -> None:
        # sum_b Q^alt_b ⊙ sinc((i-j)/N_d) = N_d I when N_d divides the band
        # (design note section 2.1): the aliased lags fall on sinc zeros
        n_freqs = 40
        lag = np.subtract.outer(np.arange(n_freqs), np.arange(n_freqs))
        total = np.sum(q_alt(n_dlys, n_freqs), axis=0) * np.sinc(lag / n_dlys)
        assert_close(total, n_dlys * np.eye(n_freqs))

    def test_healpix_beam(self, ds, beam_nf_dipole) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        nu = ds.freqs[SPW[0] : SPW[1]]
        np.testing.assert_array_equal(
            response.healpix_integral_beam(beam_nf_dipole, BEAM_POL, nu),
            ds.get_integral_beam(BEAM_POL),
        )
        kernel = DelayResponse(beam="healpix").kernel(
            nu, ds._delays_s(), beam_nf_dipole, BEAM_POL, beam_nf_dipole.cosmo, u.Mpc
        )
        separable = DelayResponse().kernel(
            nu, ds._delays_s(), beam_nf_dipole, BEAM_POL, beam_nf_dipole.cosmo, u.Mpc
        )
        # the two beam models agree to the per-cent level (design note 3.2)
        assert np.max(np.abs(np.diag(kernel) / np.diag(separable) - 1)) < 0.01

    def test_healpix_needs_uvbeam(self, unflagged_uvd) -> None:
        from hera_pspec import PSpecBeamGauss

        freqs = np.linspace(140e6, 150e6, 20, endpoint=False)
        gauss = PSpecBeamGauss(0.2 * u.rad, np.linspace(100e6, 200e6, 50) * u.Hz)
        delays = utils.get_delays(freqs)
        with pytest.raises(ValueError, match="HEALPix"):
            DelayResponse(beam="healpix").kernel(
                freqs, delays, gauss, "pI", gauss.cosmo, u.Mpc
            )
        # the separable beam works for any PSpecBeam
        kernel = DelayResponse().kernel(freqs, delays, gauss, "pI", gauss.cosmo, u.Mpc)
        assert kernel.shape == (20, 20) and np.all(np.isfinite(kernel))


class TestResponseCaching:
    def test_no_quantities_after_first_call(self, ds, monkeypatch) -> None:
        # astropy work happens once per spw and pol, not per baseline pair
        configure(ds, 20, "blackman-harris", "identity", True, (0, 0))
        ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        calls = []
        original = u.Quantity.__array_finalize__

        def counting(self, obj):
            calls.append(1)
            return original(self, obj)

        monkeypatch.setattr(u.Quantity, "__array_finalize__", counting)
        H = ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        assert calls == []
        assert not isinstance(H, u.Quantity)

    def test_kernel_cache_invalidation(self, ds, beam_nf_dipole) -> None:
        configure(ds, 20, "none", "identity", True, (0, 0))
        delay_response = DelayResponse()
        ds.get_response(KEY1, KEY2, delay_response, BEAM_POL)
        assert len(ds._response_kernels) == 1
        ds.get_response(KEY1, KEY2, delay_response, BEAM_POL)
        assert len(ds._response_kernels) == 1
        ds.get_response(KEY1, KEY2, delay_response, BEAM_POL, length_unit=u.Mpc)
        assert len(ds._response_kernels) == 2
        ds.set_spw((400, 420), ndlys=10)
        littleh = u.Mpc / cu.littleh
        first = ds._response_kernel(delay_response, BEAM_POL, littleh)
        assert len(ds._response_kernels) == 3
        assert first[0].shape == (20, 20)
        # an equal DelayResponse shares the entry; a new beam object does not
        assert ds._response_kernel(DelayResponse(), BEAM_POL, littleh) is first
        ds.primary_beam = copy.copy(beam_nf_dipole)
        rebuilt = ds._response_kernel(delay_response, BEAM_POL, littleh)
        assert rebuilt is not first
        np.testing.assert_array_equal(rebuilt[0], first[0])
        ds.clear_cache()
        assert ds._response_kernels == {}


PSPEC_BLS = [(23, 24), (24, 25)]


def run_pspec(ds, **kwargs):
    options = dict(
        spw_ranges=[SPW],
        input_data_weight="identity",
        taper="none",
        store_cov=True,
        verbose=False,
    )
    options.update(kwargs)
    return ds.pspec(PSPEC_BLS, PSPEC_BLS, (0, 1), [("ee", "ee")], **options)


class TestPspecOneStepLegacy:
    """The provisional one-step path with DelayResponse(convention="legacy").

    Without a taper, without flags and with Ndlys = Nfreqs, sum_b H_ab does
    not depend on a, so the two-step result is the one-step (legacy) result
    times the scalar (design note section 3.2), and the covariance scales by
    the scalar squared.
    """

    # norm="I" with iC weighting is left out: its scalar_delay_adjustment
    # depends on the delay, so two-step is not one-step times the scalar
    @pytest.mark.parametrize(
        "weighting, norm",
        [
            ("identity", "I"),
            ("identity", "H^-1"),
            ("identity", "V^-1/2"),
            ("iC", "H^-1"),
            ("iC", "V^-1/2"),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Poorly conditioned covariance:UserWarning")
    def test_matches_two_step_times_scalar(self, ds, weighting, norm) -> None:
        legacy = DelayResponse(convention="legacy")
        uvp2 = run_pspec(ds, norm=norm, input_data_weight=weighting)
        uvp1 = run_pspec(
            ds,
            norm=norm,
            input_data_weight=weighting,
            normalization="one-step",
            response=legacy,
        )
        scalar = uvp2.scalar_array[0, 0]
        assert scalar > 1.0
        np.testing.assert_array_equal(
            uvp1.scalar_array, np.ones_like(uvp1.scalar_array)
        )
        assert uvp1.norm_units == "Hz sr"
        assert uvp2.norm_units == "Mpc3 / littleh3"
        assert uvp1.normalization == "one-step"
        assert uvp1.response_model == repr(legacy)
        assert uvp2.normalization == "two-step"
        assert not hasattr(uvp2, "response_model")
        assert_close(uvp1.data_array[0] * scalar, uvp2.data_array[0])
        cov_scale = np.max(np.abs(uvp2.cov_array_real[0]))
        assert_close(uvp1.cov_array_real[0] * scalar**2, uvp2.cov_array_real[0])
        # the imaginary covariance is rounding noise here (~1e-18 of the real
        # part), so it is compared on the scale of the real part
        assert_close(
            uvp1.cov_array_imag[0] * scalar**2, uvp2.cov_array_imag[0], scale=cov_scale
        )
        assert_close(uvp1.window_function_array[0], uvp2.window_function_array[0])

    @pytest.mark.parametrize("norm", ["I", "V^-1/2"])
    def test_q_matrices_threaded_through(self, ds, norm) -> None:
        legacy = DelayResponse(convention="legacy")
        uvp_default = run_pspec(
            ds, norm=norm, normalization="one-step", response=legacy
        )

        def q_alt_matrices(freqs, delays):
            return q_alt(delays.size, freqs.size)

        uvp_q = run_pspec(
            ds,
            norm=norm,
            normalization="one-step",
            response=legacy,
            q_matrices=q_alt_matrices,
        )
        assert_close(uvp_q.data_array[0], uvp_default.data_array[0])
        assert_close(uvp_q.cov_array_real[0], uvp_default.cov_array_real[0])
        assert_close(
            uvp_q.window_function_array[0], uvp_default.window_function_array[0]
        )

    def test_default_two_step_unchanged(self, ds) -> None:
        uvp_default = run_pspec(ds, taper="blackman-harris")
        uvp_explicit = run_pspec(ds, taper="blackman-harris", normalization="two-step")
        np.testing.assert_array_equal(
            uvp_explicit.data_array[0], uvp_default.data_array[0]
        )
        assert uvp_default.normalization == "two-step"
        uvp_exact = run_pspec(ds, exact_norm=True, store_cov=False)
        assert uvp_exact.normalization == "two-step-exact"

    def test_errors(self, ds) -> None:
        legacy = DelayResponse(convention="legacy")
        with pytest.raises(ValueError, match="normalization must be"):
            run_pspec(ds, normalization="three-step")
        with pytest.raises(ValueError, match="one-step"):
            run_pspec(ds, response=legacy)
        with pytest.raises(ValueError, match="one-step"):
            run_pspec(ds, q_matrices=np.zeros((40, 40, 40)))
        with pytest.raises(ValueError, match="exact_norm"):
            run_pspec(ds, normalization="one-step", response=legacy, exact_norm=True)
        with pytest.raises(ValueError, match="allow_fft"):
            run_pspec(
                ds,
                normalization="one-step",
                response=legacy,
                allow_fft=True,
                q_matrices=np.zeros((40, 40, 40)),
            )
        with pytest.raises(ValueError, match="legacy-exact"):
            run_pspec(
                ds,
                normalization="one-step",
                response=DelayResponse(convention="legacy-exact"),
            )

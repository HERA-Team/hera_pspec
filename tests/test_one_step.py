"""Tests for the normalization label of power spectra and for one-step
normalization in cosmological units (#329).

"""

import copy
from pathlib import Path

import astropy.cosmology.units as cu
import h5py
import numpy as np
import pytest
import uvtools.dspec as dspec
from astropy import units as u
from astropy.cosmology import LambdaCDM, Planck18
from pyuvdata import UVData
from scipy.interpolate import interp1d

from hera_pspec import (
    DelayResponse,
    PSpecBeamUV,
    PSpecContainer,
    PSpecData,
    UVPSpec,
    conversions,
    grouping,
    utils,
    uvpspec,
)
from hera_pspec.data import DATA_PATH
from tests import legacy_fixtures as lf

DATA_PATH = Path(DATA_PATH)

# identities: 1e-12 relative to the largest element;
# results through a matrix inverse or eigendecomposition: 1e-9, as in
# tests/regression_cases.py
IDENTITY_RTOL = 1e-12
LINALG_RTOL = 1e-9
SPW = (400, 440)
NFREQS = SPW[1] - SPW[0]
BEAM_POL = -5  # ee, the beam's xx
KEY1 = (0, (23, 24), "ee")
KEY2 = (1, (24, 25), "ee")
PSPEC_BLS = [(23, 24), (24, 25)]
LITTLEH = u.Mpc / cu.littleh
X2Y_UNIT = LITTLEH**3 / (u.sr * u.Hz)
TAPERS = ["none", "blackman-harris"]

pytestmark = pytest.mark.filterwarnings(
    "ignore:Poorly conditioned covariance:UserWarning"
)


def assert_close(actual, expected, rtol=IDENTITY_RTOL, scale=None):
    """Compare at rtol, with an absolute floor of rtol times ``scale``
    (default: the largest element of ``expected``)."""
    expected = np.asarray(expected)
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


def configure(ds, ndlys, taper, weighting="identity"):
    ds.set_spw(SPW, ndlys=ndlys)
    ds.set_taper(taper)
    ds.set_weighting(weighting)
    ds.set_symmetric_taper(True)
    ds.set_filter_extension((0, 0))
    ds.clear_cache()


def run_pspec(ds, **kwargs):
    options = dict(
        spw_ranges=[SPW], input_data_weight="identity", taper="none", verbose=False
    )
    options.update(kwargs)
    return ds.pspec(PSPEC_BLS, PSPEC_BLS, (0, 1), [("ee", "ee")], **options)


def taper_values(taper, n):
    return np.ones(n) if taper == "none" else dspec.gen_window(taper, n)


def channel_weights(beam, pol, freqs, cosmo, length_unit=LITTLEH):
    """w_i = (Omega_pp / Omega_p^2)(nu_i) / X2Y(nu_i), computed independently of
    hera_pspec.response: the Omega ratio interpolated as in the scalar
    (quadratic in MHz), X2Y from conversions."""
    ratio = beam.power_beam_sq_int(pol) / beam.power_beam_int(pol) ** 2
    omega = interp1d(
        beam.beam_freqs / 1e6, ratio, kind="quadratic", fill_value="extrapolate"
    )(freqs / 1e6)
    x2y = conversions.X2Y(conversions.f2z(freqs), cosmo, length_unit).to_value(
        length_unit**3 / (u.sr * u.Hz)
    )
    return omega / x2y


def discrete_scalar(weights, taper, dnu):
    """s_disc = (N dnu)^2 / (dnu sum_i T_i^2 w_i): the two-step scalar
    integral evaluated as a sum over channel centres, with w = B / X2Y."""
    bandwidth = weights.size * dnu
    return bandwidth**2 / (dnu * np.sum(taper**2 * weights))


def spw_freqs(ds):
    return ds.freqs[SPW[0] : SPW[1]]


def mean_unnormed_q(ds, covariance, q_est=None):
    """<q_a> = (1/2) tr[R_1^dagger Q_a R_2 C] for a data covariance C
    (arXiv:1502.06016 Eq. 13)."""
    if q_est is None:
        q_est = ds.get_Q_alt_tensor()
    A = np.conj(ds.R(KEY1)).T @ q_est @ ds.R(KEY2)
    return 0.5 * np.einsum("aji,ij->a", A, covariance)


def one_step_M(ds, H, norm):
    if norm == "V^-1/2":
        V = ds.get_unnormed_V(KEY1, KEY2, pol=BEAM_POL)
        return ds.get_MW(H, H, mode=norm, band_covar=V)
    return ds.get_MW(H, H, mode=norm)


class UserResponse:
    """C_{,b} = Q^alt_b ⊙ K for a given float kernel, in a chosen unit."""

    def __init__(self, kernel, unit):
        self.kernel = kernel
        self.unit = unit

    def matrices(self, freqs, delays, beam, pol, cosmo, length_unit):
        q_alt = utils.get_Q_alt_tensor(delays.size, n_freqs=freqs.size)
        return q_alt * self.kernel * self.unit


def constant_w_kernel(n_freqs, n_dlys, dnu, w0):
    """K = Delta_eta sinc((i-j)/N_d) w0: a top-hat response with a constant
    w = B / X2Y."""
    lag = np.subtract.outer(np.arange(n_freqs), np.arange(n_freqs))
    return w0 * np.sinc(lag / n_dlys) / (n_dlys * dnu)


def _read(path):
    uvp = UVPSpec()
    uvp.read_hdf5(path)
    return uvp


def _labelled(uvp, normalization, response_model=None):
    uvp = copy.deepcopy(uvp)
    uvp.normalization = normalization
    if response_model is not None:
        uvp.response_model = response_model
    return uvp


class TestNormalizationParameter:
    """The normalization and response_model parameters."""

    def test_default_and_check(self, vanilla_uvp) -> None:
        assert UVPSpec().normalization == "two-step"
        assert vanilla_uvp.normalization == "two-step"
        assert not hasattr(vanilla_uvp, "response_model")
        assert set(uvpspec.NORMALIZATIONS) == {"two-step", "two-step-exact", "one-step"}
        bad = _labelled(vanilla_uvp, "three-step")
        with pytest.raises(AssertionError, match="normalization must be one of"):
            bad.check()
        missing = copy.deepcopy(vanilla_uvp)
        del missing.normalization
        with pytest.raises(AssertionError, match="normalization"):
            missing.check()

    @pytest.mark.parametrize("normalization", uvpspec.NORMALIZATIONS)
    def test_hdf5_round_trip(self, vanilla_uvp_with_beam, normalization, tmp_path):
        model = "DelayResponse()" if normalization == "one-step" else None
        uvp = _labelled(vanilla_uvp_with_beam, normalization, model)
        path = tmp_path / "uvp.h5"
        uvp.write_hdf5(path)
        with h5py.File(path, "r") as f:
            assert f.attrs["uvpspec_version"] == "2"
            assert f.attrs["normalization"] == normalization
            assert ("response_model" in f.attrs) == (model is not None)
        uvp_in = _read(path)
        assert uvp_in.normalization == normalization
        assert getattr(uvp_in, "response_model", None) == model
        assert uvp_in == uvp

    def test_pre_v1_version_2_file_reads_as_two_step(
        self, vanilla_uvp_with_beam, tmp_path
    ) -> None:
        # a version 2 file written before the normalization attribute existed
        path = tmp_path / "uvp.h5"
        _labelled(vanilla_uvp_with_beam, "one-step").write_hdf5(path)
        with h5py.File(path, "r+") as f:
            del f.attrs["normalization"]
        assert _read(path).normalization == "two-step"

    def test_legacy_files_read_as_two_step(self) -> None:
        with h5py.File(DATA_PATH / "test_uvp.h5", "r") as f:
            assert "normalization" not in f.attrs
        assert _read(DATA_PATH / "test_uvp.h5").normalization == "two-step"
        with h5py.File(lf.CONTAINER_FILE, "r") as f:
            group = f[lf.CONTAINER_GROUP]
            for name in group:
                assert "normalization" not in group[name].attrs
        psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
        for name in psc.spectra(lf.CONTAINER_GROUP):
            uvp = psc.get_pspec(lf.CONTAINER_GROUP, name)
            assert uvp.normalization == "two-step"

    def test_equality(self, vanilla_uvp) -> None:
        assert _labelled(vanilla_uvp, "one-step") != vanilla_uvp
        assert _labelled(vanilla_uvp, "one-step", "a") != _labelled(
            vanilla_uvp, "one-step", "b"
        )
        assert _labelled(vanilla_uvp, "one-step", "a") == _labelled(
            vanilla_uvp, "one-step", "a"
        )

    def test_combine_refuses_mixed_normalizations(self, vanilla_uvp) -> None:
        blpairs = np.unique(vanilla_uvp.blpair_array)
        uvp1 = vanilla_uvp.select(blpairs=blpairs[:1], inplace=False)
        uvp2 = vanilla_uvp.select(blpairs=blpairs[1:], inplace=False)
        combined = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert combined.normalization == "two-step"
        one_step = _labelled(uvp2, "one-step", "DelayResponse()")
        with pytest.raises(AssertionError, match="'normalization'"):
            uvpspec.combine_uvpspec([uvp1, one_step], verbose=False)
        both = [_labelled(uvp1, "one-step", "DelayResponse()"), one_step]
        combined = uvpspec.combine_uvpspec(both, verbose=False)
        assert combined.normalization == "one-step"
        assert combined.response_model == "DelayResponse()"
        other_model = _labelled(uvp1, "one-step", "DelayResponse(x2y='center')")
        with pytest.raises(AssertionError, match="'response_model'"):
            uvpspec.combine_uvpspec([other_model, one_step], verbose=False)

    def test_carried_through(self, vanilla_uvp) -> None:
        uvp = _labelled(vanilla_uvp, "one-step", "DelayResponse()")
        folded = copy.deepcopy(uvp)
        folded.fold_spectra()
        kbins = np.array([0.1, 0.2]) * cu.littleh / u.Mpc
        outputs = [
            uvp.select(spws=[0], inplace=False),
            uvp.average_spectra(time_avg=True, inplace=False),
            grouping.average_spectra(uvp, time_avg=True, inplace=False),
            folded,
            grouping.spherical_average(uvp, kbins, np.array([0.05, 0.05])),
        ]
        for out in outputs:
            assert out.normalization == "one-step"
            assert out.response_model == "DelayResponse()"


class TestRelationToTwoStep:
    """For norm="I", Ndlys = Nfreqs, identity weighting, no flags and the
    separable beam, one-step = two-step * s_disc / s for every delay and
    time (s_disc: see discrete_scalar)."""

    @pytest.mark.parametrize("taper", TAPERS)
    def test_eq_2(self, ds, taper) -> None:
        uvp2 = run_pspec(ds, taper=taper, store_cov=True)
        uvp1 = run_pspec(ds, taper=taper, store_cov=True, normalization="one-step")
        dnu = np.median(np.diff(spw_freqs(ds)))
        weights = channel_weights(
            ds.primary_beam, BEAM_POL, spw_freqs(ds), ds.primary_beam.cosmo
        )
        s_disc = discrete_scalar(weights, taper_values(taper, NFREQS), dnu)
        scalar = uvp2.scalar_array[0, 0]
        assert_close(uvp1.data_array[0], uvp2.data_array[0] * s_disc / scalar)
        ratio2 = (s_disc / scalar) ** 2
        assert_close(uvp1.cov_array_real[0], uvp2.cov_array_real[0] * ratio2)
        # the reference scalar is stored, computed as for two-step
        np.testing.assert_array_equal(uvp1.scalar_array, uvp2.scalar_array)
        # one-step is a few 0.01% higher here: the channel sum s_disc is
        # slightly larger than the band integral s
        assert 0 < s_disc / scalar - 1 < 3e-3
        assert uvp1.norm_units == uvp2.norm_units == "Mpc3 / littleh3"
        assert uvp1.normalization == "one-step"
        assert uvp1.response_model == repr(DelayResponse())


class TestFlatSpectrumExpectation:
    """For a flat P(k) = 1 the expected data covariance is
    C = sum_b C_{,b} = diag(w) / dnu (the top-hat bins tile eta), and one-step
    gives <p> = M <q> = 1 exactly."""

    @pytest.mark.parametrize("norm", ["I", "H^-1", "V^-1/2"])
    @pytest.mark.parametrize("taper", TAPERS)
    @pytest.mark.parametrize("ndlys", [NFREQS, NFREQS // 2])
    @pytest.mark.parametrize("weighting", ["identity", "iC"])
    def test_flat_spectrum(self, ds, norm, taper, ndlys, weighting) -> None:
        configure(ds, ndlys, taper, weighting)
        freqs = spw_freqs(ds)
        weights = channel_weights(
            ds.primary_beam, BEAM_POL, freqs, ds.primary_beam.cosmo
        )
        covariance = np.diag(weights) / np.median(np.diff(freqs))
        q_mean = mean_unnormed_q(ds, covariance)
        H = ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        M, _ = one_step_M(ds, H, norm)
        assert_close(M @ q_mean, np.ones(ndlys), rtol=LINALG_RTOL, scale=1.0)


class TestNormalizationMatrices:
    """W rows sum to 1 (I, V^-1/2), M H = I (H^-1), cov_p = M V_q M^T."""

    @pytest.mark.parametrize("norm", ["I", "V^-1/2"])
    @pytest.mark.parametrize("taper", TAPERS)
    def test_window_rows_sum_to_one(self, ds, norm, taper) -> None:
        uvp = run_pspec(ds, norm=norm, taper=taper, normalization="one-step")
        rows = uvp.window_function_array[0].sum(axis=2)
        assert_close(rows, np.ones_like(rows), rtol=LINALG_RTOL)

    @pytest.mark.parametrize("ndlys", [NFREQS, NFREQS // 2])
    @pytest.mark.parametrize("taper", TAPERS)
    def test_h_inverse(self, ds, ndlys, taper) -> None:
        configure(ds, ndlys, taper)
        H = ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        M, W = ds.get_MW(H, H, mode="H^-1")
        assert_close(M @ H, np.eye(ndlys), rtol=LINALG_RTOL, scale=1.0)
        uvp = run_pspec(
            ds, norm="H^-1", taper=taper, n_dlys=[ndlys], normalization="one-step"
        )
        window = uvp.window_function_array[0]
        assert_close(
            window,
            np.broadcast_to(np.eye(ndlys)[None, :, :, None], window.shape),
            rtol=LINALG_RTOL,
            scale=1.0,
        )

    @pytest.mark.parametrize("norm", ["I", "H^-1"])
    def test_covariance(self, ds, norm) -> None:
        uvp = ds.pspec(
            [KEY1[1]],
            [KEY2[1]],
            (0, 1),
            [("ee", "ee")],
            spw_ranges=[SPW],
            norm=norm,
            taper="blackman-harris",
            store_cov=True,
            cov_model="empirical",
            normalization="one-step",
            verbose=False,
        )
        configure(ds, NFREQS, "blackman-harris")
        H = ds.get_response(KEY1, KEY2, DelayResponse(), BEAM_POL)
        M, _ = ds.get_MW(H, H, mode=norm)
        # identity weighting with a symmetric taper gives a real M here
        assert np.max(np.abs(M.imag)) < IDENTITY_RTOL * np.max(np.abs(M))
        cov_q_real, _, _, _ = ds.get_analytic_covariance(KEY1, KEY2, M, pol=BEAM_POL)
        expected = np.einsum("ab,tbc,dc->tad", M.real, cov_q_real, M.real)
        blpair = uvp.antnums_to_blpair(((23, 24), (24, 25)))
        cov = uvp.get_cov((0, blpair, int(uvp.polpair_array[0])))
        assert_close(cov, expected, rtol=LINALG_RTOL)


class TestUserResponseAndErrors:
    """User responses in pspec(), unit checks and error paths."""

    def test_user_response_units(self, ds) -> None:
        configure(ds, NFREQS, "blackman-harris")
        spw = ds.get_spw(include_extension=True)
        kernel = DelayResponse().kernel(
            ds.freqs[spw[0] : spw[1]],
            ds._delays_s(),
            ds.primary_beam,
            BEAM_POL,
            ds.primary_beam.cosmo,
            u.Mpc,
        )
        kwargs = dict(
            taper="blackman-harris", normalization="one-step", length_unit=u.Mpc
        )
        uvp_default = run_pspec(ds, **kwargs)
        uvp_mpc = run_pspec(ds, response=UserResponse(kernel, u.Mpc**-3), **kwargs)
        # 1 / Mpc^3 = 1e9 / Gpc^3
        uvp_gpc = run_pspec(
            ds, response=UserResponse(kernel * 1e9, u.Gpc**-3), **kwargs
        )
        assert_close(uvp_mpc.data_array[0], uvp_default.data_array[0])
        assert_close(uvp_gpc.data_array[0], uvp_mpc.data_array[0])
        assert uvp_mpc.response_model == "tests.test_one_step.UserResponse"
        assert uvp_mpc.norm_units == "Mpc3"
        with pytest.raises(ValueError, match="not equivalent"):
            run_pspec(ds, response=UserResponse(kernel, u.Mpc**-2), **kwargs)

    def test_errors(self, ds, unflagged_uvd) -> None:
        no_beam = PSpecData(dsets=[unflagged_uvd, unflagged_uvd], wgts=[None, None])
        with pytest.raises(ValueError, match="needs a beam"):
            run_pspec(no_beam, normalization="one-step")
        with pytest.raises(ValueError, match="needs a beam"):
            run_pspec(
                no_beam,
                normalization="one-step",
                response=UserResponse(np.ones((NFREQS, NFREQS)), u.Mpc**-3),
            )
        # the legacy convention needs no beam
        uvp = run_pspec(
            no_beam,
            normalization="one-step",
            response=DelayResponse(convention="legacy"),
        )
        assert uvp.norm_units == "Hz sr"
        with pytest.raises(ValueError, match="legacy-exact"):
            run_pspec(
                ds,
                normalization="one-step",
                response=DelayResponse(convention="legacy-exact"),
            )
        with pytest.raises(TypeError, match="matrices"):
            run_pspec(ds, normalization="one-step", response=object())
        with pytest.raises(NotImplementedError, match="different polarizations"):
            ds.pspec(
                PSPEC_BLS,
                PSPEC_BLS,
                (0, 1),
                [("ee", "nn")],
                spw_ranges=[SPW],
                normalization="one-step",
                verbose=False,
            )

    def test_sampling_selects_default_response(self, ds) -> None:
        uvp = run_pspec(ds, normalization="one-step", sampling=True)
        assert uvp.response_model == repr(DelayResponse(sampling=True))
        explicit = run_pspec(
            ds, normalization="one-step", response=DelayResponse(sampling=True)
        )
        np.testing.assert_array_equal(uvp.data_array[0], explicit.data_array[0])

    def test_no_quantity_in_baseline_loop(self, ds, monkeypatch) -> None:
        # astropy work happens once per spw and pol: the number of Quantities
        # created does not grow with the number of baseline pairs
        calls = []
        original = u.Quantity.__array_finalize__

        def counting(self, obj):
            calls.append(1)
            return original(self, obj)

        monkeypatch.setattr(u.Quantity, "__array_finalize__", counting)
        counts = []
        for bls in (PSPEC_BLS[:1], PSPEC_BLS, PSPEC_BLS + [(23, 23)]):
            calls.clear()
            ds.pspec(
                bls,
                bls,
                (0, 1),
                [("ee", "ee")],
                spw_ranges=[SPW],
                store_cov=True,
                normalization="one-step",
                verbose=False,
            )
            counts.append(len(calls))
        assert counts[0] > 0
        assert counts[1] == counts[0] and counts[2] == counts[0]


class TestExpectedDifferences:
    """Closed forms of the two-step / one-step ratio for a flat P(k),
    with a user response of constant w = B / X2Y so they are exact (design
    note sections 1.6, 3.2 and 6). F1 and F2 are pre-existing two-step
    biases that the one-step path does not have."""

    W0 = 1e-6

    def flat_q(self, ds):
        dnu = np.median(np.diff(spw_freqs(ds)))
        return mean_unnormed_q(ds, self.W0 * np.eye(NFREQS) / dnu), dnu

    def one_step(self, ds, ndlys, norm, q_mean, dnu):
        kernel = constant_w_kernel(NFREQS, ndlys, dnu, self.W0)
        H = ds.get_response(KEY1, KEY2, UserResponse(kernel, LITTLEH**-3), BEAM_POL)
        M, _ = ds.get_MW(H, H, mode=norm)
        p = M @ q_mean
        # one-step recovers the flat spectrum
        assert_close(p, np.ones(ndlys), rtol=IDENTITY_RTOL, scale=1.0)
        return p

    def scalar(self, ds, taper, **kwargs):
        ds.set_taper(taper)
        value = ds.scalar((BEAM_POL, BEAM_POL), length_unit=LITTLEH, **kwargs)
        return value.to_value(X2Y_UNIT)

    @pytest.mark.parametrize(
        "norm, taper, ndlys, bias",
        [
            # reference: only the scalar's integral vs channel sum differs
            ("I", "none", NFREQS, 1.0),
            # F2: the N_f sinc in get_H with N_d = N_f / 2
            ("I", "none", NFREQS // 2, 1.0 / (1.0 + np.sinc(0.5))),
            # F1: H^-1 with a taper, multiplied by s(T) / s(none) below
            ("H^-1", "blackman-harris", NFREQS, 1.0),
        ],
    )
    def test_two_step(self, ds, norm, taper, ndlys, bias) -> None:
        configure(ds, ndlys, taper)
        q_mean, dnu = self.flat_q(ds)
        p1 = self.one_step(ds, ndlys, norm, q_mean, dnu)
        G = ds.get_G(KEY1, KEY2)
        H = ds.get_H(KEY1, KEY2, sampling=False)
        M, _ = ds.get_MW(G, H, mode=norm)
        # the two-step scalar as pspec() computes it: the H^-1 taper
        # override is ignored by scalar(), so it is the tapered scalar
        s_taper = self.scalar(ds, taper)
        p2 = s_taper * (M @ q_mean)
        if norm == "I":
            p2 = p2 * ds.scalar_delay_adjustment(Gv=G, Hv=H)
        s_none = self.scalar(ds, "none")
        s_disc_none = NFREQS * dnu / self.W0  # eq. (2) with T = 1, w = W0
        expected = bias * (s_taper / s_none) * (s_none / s_disc_none)
        assert_close(p2 / p1, np.full(ndlys, expected), rtol=IDENTITY_RTOL)
        if norm == "H^-1":
            # F1 is 1 / <T^2> to within the scalar's taper interpolation
            mean_t2 = np.mean(taper_values(taper, NFREQS) ** 2)
            assert abs(s_taper / s_none * mean_t2 - 1) < 2e-2

    @pytest.mark.parametrize("taper", TAPERS)
    def test_two_step_exact(self, ds, taper) -> None:
        # exact_norm today gives X2Y_bar * <1 / X2Y>_{T^2 B_ii^2} for a flat
        # spectrum; one-step with the same HEALPix
        # beam gives 1
        configure(ds, NFREQS, taper)
        freqs = spw_freqs(ds)
        dnu = np.median(np.diff(freqs))
        beam_diag = np.diag(ds.get_integral_beam(BEAM_POL)).real
        x2y = conversions.X2Y(
            conversions.f2z(freqs), ds.primary_beam.cosmo, LITTLEH
        ).to_value(X2Y_UNIT)
        covariance = np.diag(beam_diag / x2y) / dnu
        q_mean = mean_unnormed_q(ds, covariance)
        H1 = ds.get_response(KEY1, KEY2, DelayResponse(beam="healpix"), BEAM_POL)
        M1, _ = ds.get_MW(H1, H1, mode="I")
        assert_close(M1 @ q_mean, np.ones(NFREQS), rtol=IDENTITY_RTOL, scale=1.0)

        weight = np.median(np.diff(ds._delays_s())) * ds.get_integral_beam(BEAM_POL)
        q_exact = mean_unnormed_q(ds, covariance, ds.get_Q_alt_tensor() * weight)
        G = ds.get_G(KEY1, KEY2, exact_norm=True, pol=BEAM_POL)
        H = ds.get_H(KEY1, KEY2, exact_norm=True, pol=BEAM_POL)
        M, _ = ds.get_MW(G, H, mode="I", exact_norm=True)
        x2y_bar = self.scalar(ds, taper, exact_norm=True)
        p2 = x2y_bar * (M @ q_exact)
        t2b2 = taper_values(taper, NFREQS) ** 2 * beam_diag**2
        expected = x2y_bar * np.sum(t2b2 / x2y) / np.sum(t2b2)
        assert_close(p2, np.full(NFREQS, expected), rtol=IDENTITY_RTOL)
        # the stored exact-norm scalar is the band-mean X2Y
        assert abs(x2y_bar / np.mean(x2y) - 1) < 1e-3


def diagonal_errors(uvp):
    """sqrt of the diagonal of the real covariance, as a complex stats array."""
    variance = np.diagonal(uvp.cov_array_real[0], axis1=1, axis2=2)
    return np.sqrt(np.moveaxis(variance, -1, 1)).astype(complex)


NEW_COSMOLOGIES = [Planck18, LambdaCDM(H0=70.0, Om0=0.25, Ode0=0.75, Tcmb0=0.0)]


class TestSetCosmology:
    """set_cosmology dispatches on normalization."""

    @pytest.mark.parametrize("new_cosmo", NEW_COSMOLOGIES, ids=["Planck18", "H0_70"])
    @pytest.mark.parametrize("taper", TAPERS)
    @pytest.mark.parametrize("length_unit", [LITTLEH, u.Mpc], ids=["littleh", "Mpc"])
    def test_one_step_rescale_matches_rebuild(
        self, ds, unflagged_uvd, beam_nf_dipole, new_cosmo, taper, length_unit
    ) -> None:
        kwargs = dict(
            taper=taper,
            store_cov=True,
            normalization="one-step",
            length_unit=length_unit,
        )
        uvp = run_pspec(ds, **kwargs)
        # errors on the power spectra (store_cov_diag is not used: the
        # imaginary covariance of these identical datasets is rounding noise)
        uvp.stats_array = {"errors": {0: diagonal_errors(uvp)}}
        old_scalar = uvp.scalar_array.copy()
        uvp.set_cosmology(new_cosmo, overwrite=True, verbose=False)

        beam = copy.deepcopy(beam_nf_dipole)
        beam.cosmo = new_cosmo
        rebuilt = run_pspec(
            PSpecData(
                dsets=[unflagged_uvd, unflagged_uvd], wgts=[None, None], beam=beam
            ),
            **kwargs,
        )
        # the reference scalar is recomputed exactly as pspec() computes it
        assert_close(uvp.scalar_array, rebuilt.scalar_array)
        # a real change of the scalar (-1% to +16% in Mpc, more with h)
        assert np.all(np.abs(uvp.scalar_array / old_scalar - 1) > 5e-3)
        # approximate rescale vs exact rebuild: <= 1e-5, since s_disc / s
        # hardly depends on the cosmology
        assert_close(uvp.data_array[0], rebuilt.data_array[0], rtol=1e-5)
        for name in ("cov_array_real", "cov_array_imag"):
            assert_close(
                getattr(uvp, name)[0],
                getattr(rebuilt, name)[0],
                rtol=1e-5,
                scale=np.max(np.abs(rebuilt.cov_array_real[0])),
            )
        assert_close(uvp.stats_array["errors"][0], diagonal_errors(rebuilt), rtol=1e-5)
        assert uvp.cosmo is new_cosmo or uvp.cosmo.is_equivalent(new_cosmo)
        assert uvp.norm_units == rebuilt.norm_units

    def test_one_step_same_cosmology_is_identity(self, ds) -> None:
        uvp = run_pspec(ds, taper="blackman-harris", normalization="one-step")
        data = uvp.data_array[0].copy()
        uvp.set_cosmology(uvp.cosmo, overwrite=True, verbose=False)
        assert_close(uvp.data_array[0], data)

    def test_legacy_one_step_raises(self, ds) -> None:
        uvp = run_pspec(
            ds, normalization="one-step", response=DelayResponse(convention="legacy")
        )
        with pytest.raises(ValueError, match="not in cosmological units"):
            uvp.set_cosmology(Planck18, overwrite=True, verbose=False)

    @pytest.mark.parametrize("exact_norm", [False, True])
    def test_two_step_keeps_pre_v1_behaviour_known_issues(self, ds, exact_norm) -> None:
        # Pins what set_cosmology did before v1.0 for two-step spectra; it
        # does not assert that this is right. Known issues kept on purpose
        # (to be filed): covariances and stats are not rescaled, the scalar
        # uses 1000 integration steps where pspec() uses 2000, and for
        # exact_norm spectra the stored band-mean X2Y is replaced by the
        # two-step scalar, multiplying the data by ~1e6.
        uvp = run_pspec(
            ds, taper="blackman-harris", store_cov=not exact_norm, exact_norm=exact_norm
        )
        assert uvp.normalization == ("two-step-exact" if exact_norm else "two-step")
        before = copy.deepcopy(uvp)
        new_cosmo = NEW_COSMOLOGIES[1]
        uvp.set_cosmology(new_cosmo, overwrite=True, verbose=False)
        expected = copy.deepcopy(before)
        expected.cosmo = new_cosmo
        scalar = expected.compute_scalar(
            0, before.polpair_array[0], num_steps=1000
        ).to_value(LITTLEH**3 / (u.Hz * u.sr))
        np.testing.assert_array_equal(uvp.scalar_array, [[scalar]])
        assert_close(
            uvp.data_array[0], before.data_array[0] * scalar / before.scalar_array[0, 0]
        )
        if exact_norm:
            assert scalar / before.scalar_array[0, 0] > 1e5
        else:
            np.testing.assert_array_equal(
                uvp.cov_array_real[0], before.cov_array_real[0]
            )


@pytest.fixture(scope="module")
def noise_uvd():
    uvd = UVData()
    uvd.read_miriad(str(DATA_PATH / "zen.even.xx.LST.1.28828.uvOCRSA"))
    uvd.flag_array[:] = False
    uvd.nsample_array[:] = 1.0
    return uvd


class TestWhiteNoiseMonteCarlo:
    """pspec() on simulated data whose covariance is a flat P(k) = 1 in
    the one-step response model, C = diag(w) / dnu (see
    TestFlatSpectrumExpectation). Every one-step
    estimator has expectation 1, so the mean power must agree for norm I and
    H^-1, no taper and Blackman-Harris, Ndlys = Nfreqs and Nfreqs / 2 (the
    configurations where the two-step path is biased: F1, F2).

    The noise is drawn independently for every (baseline, time) and both
    datasets hold the same realisation, so each baseline-pair x time gives an
    independent estimate of the delay-averaged power. The test compares the
    mean over the N = 45 x 10 x 3 = 1350 estimates with 1 to within five
    standard errors, std / sqrt(N): a chance failure has probability
    ~6e-7 per configuration (two-sided Gaussian tail, central limit theorem).
    """

    SPW = (20, 60)
    N_REALISATIONS = 3

    @pytest.fixture(scope="class")
    def estimates(self, noise_uvd, beam_nf_dipole):
        freqs = noise_uvd.freq_array.ravel()[self.SPW[0] : self.SPW[1]]
        dnu = np.median(np.diff(freqs))
        weights = channel_weights(beam_nf_dipole, "xx", freqs, beam_nf_dipole.cosmo)
        # complex Gaussian with <x_i x_i^*> = w_i / dnu
        sigma = np.sqrt(weights / dnu / 2.0)
        rng = np.random.default_rng(329)
        bls = [bl for bl in noise_uvd.get_antpairs() if bl[0] != bl[1]]
        configs = [
            (norm, taper, ndlys)
            for norm in ("I", "H^-1")
            for taper in TAPERS
            for ndlys in (40, 20)
        ]
        samples = {config: [] for config in configs}
        for _ in range(self.N_REALISATIONS):
            uvd = copy.deepcopy(noise_uvd)
            shape = uvd.data_array.shape
            noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            uvd.data_array[:] = 0.0
            uvd.data_array[:, self.SPW[0] : self.SPW[1]] = (
                noise[:, self.SPW[0] : self.SPW[1]] * sigma[None, :, None]
            )
            ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole)
            for norm, taper, ndlys in configs:
                uvp = ds.pspec(
                    bls,
                    bls,
                    (0, 1),
                    [("xx", "xx")],
                    spw_ranges=[self.SPW],
                    n_dlys=[ndlys],
                    norm=norm,
                    taper=taper,
                    store_window=False,
                    normalization="one-step",
                    verbose=False,
                )
                power = uvp.data_array[0][:, :, 0].real.mean(axis=1)
                samples[(norm, taper, ndlys)].append(power)
        return {config: np.concatenate(values) for config, values in samples.items()}

    @pytest.mark.parametrize("norm", ["I", "H^-1"])
    @pytest.mark.parametrize("taper", TAPERS)
    @pytest.mark.parametrize("ndlys", [40, 20])
    def test_mean_power_is_one(self, estimates, norm, taper, ndlys) -> None:
        power = estimates[(norm, taper, ndlys)]
        assert power.size == 45 * 10 * self.N_REALISATIONS
        standard_error = power.std(ddof=1) / np.sqrt(power.size)
        assert abs(power.mean() - 1.0) < 5.0 * standard_error
        # the test resolves the two-step biases (x3.98, x0.61) many times over
        assert standard_error < 0.02

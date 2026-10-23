import copy

import numpy as np
import pytest
from pyuvdata import UVData

from hera_pspec import (
    PSpecBeamUV,
    UVPSpec,
    conversions,
    noise,
    pspecdata,
    testing,
    utils,
)


@pytest.fixture
def sense(
    beam_nf_pstokes: PSpecBeamUV, cosmo: conversions.Cosmo_Conversions
) -> noise.Sensitivity:
    return noise.Sensitivity(beam=beam_nf_pstokes, cosmo=cosmo)


class TestSensitivity:
    def test_set_cosmology(self) -> None:
        """Check that set_cosmology accepts both a Cosmo_Conversions object and its stringified params."""
        sense = noise.Sensitivity()
        C = conversions.Cosmo_Conversions()
        sense.set_cosmology(C)
        assert C.get_params() == sense.cosmo.get_params()
        params = str(C.get_params())
        sense.set_cosmology(params)
        assert C.get_params() == sense.cosmo.get_params()

    def test_set_beam_with_cosmo(self, beam_nf_pstokes: PSpecBeamUV) -> None:
        """Check that set_beam aligns the beam's cosmology with self.cosmo when both are present."""
        sense = noise.Sensitivity()
        C = conversions.Cosmo_Conversions()
        sense.set_cosmology(C)

        beam = copy.deepcopy(beam_nf_pstokes)
        sense.set_beam(beam)
        assert sense.cosmo.get_params() == sense.beam.cosmo.get_params()
        beam.cosmo = C
        sense.set_beam(beam)
        assert sense.cosmo.get_params() == sense.beam.cosmo.get_params()

    def test_set_beam_without_cosmo(self, beam_nf_pstokes: PSpecBeamUV) -> None:
        """Check that set_beam attaches self.cosmo to a beam lacking a cosmo attribute, rather than raising."""
        sense = noise.Sensitivity()
        sense.set_cosmology(conversions.Cosmo_Conversions())
        bm = copy.deepcopy(beam_nf_pstokes)
        delattr(bm, "cosmo")
        sense.set_beam(bm)
        assert bm.cosmo is sense.cosmo

    def test_scalar(self, sense: noise.Sensitivity) -> None:
        """Check that calc_scalar records the input subband and polarization."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        sense.calc_scalar(freqs, "pI", num_steps=5000, little_h=True)
        assert np.isclose(freqs, sense.subband).all()
        assert sense.pol == "pI"

    def test_calc_p_n(self, sense: noise.Sensitivity) -> None:
        """Check that calc_P_N returns a scalar P_N in Pk form and a properly shaped, smaller DelSq array."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        sense.calc_scalar(freqs, "pI", num_steps=5000, little_h=True)

        k = np.linspace(0, 3, 10)
        Tsys = 500.0
        t_int = 10.7
        P_N = sense.calc_P_N(Tsys, t_int, Ncoherent=1, Nincoherent=1, form="Pk")
        assert isinstance(P_N, float)
        assert np.isclose(P_N, 642386932892.2921)
        Dsq = sense.calc_P_N(Tsys, t_int, k=k, Ncoherent=1, Nincoherent=1, form="DelSq")
        assert Dsq.shape == (10,)
        assert Dsq[1] < P_N


class TestCalcPN:
    """The thermal noise power P_N = scalar Tsys^2 / (t_int Ncoherent sqrt(Nincoherent))
    of Cheng et al. (2018): the visibility noise variance scales as
    Tsys^2 / t_int, coherent averaging reduces it like a longer integration,
    and incoherent averaging of power spectra reduces their rms by
    sqrt(Nincoherent)."""

    ARGS = dict(scalar=2.0e9, Tsys=300.0, t_int=10.0)

    @pytest.mark.parametrize(
        "kwargs,factor",
        [
            (dict(Tsys=600.0), 4.0),
            (dict(t_int=40.0), 1 / 4),
            (dict(Ncoherent=4), 1 / 4),
            (dict(Nincoherent=4), 1 / 2),
            (dict(scalar=6.0e9), 3.0),
        ],
        ids=["Tsys", "t_int", "Ncoherent", "Nincoherent", "scalar"],
    )
    def test_scaling(self, kwargs: dict, factor: float) -> None:
        reference = noise.calc_P_N(**self.ARGS, Ncoherent=1, Nincoherent=1)
        scaled = noise.calc_P_N(**{**self.ARGS, **kwargs})
        np.testing.assert_allclose(scaled / reference, factor, rtol=1e-12)

    def test_components(self) -> None:
        """The real and imaginary parts of a noise power spectrum each have
        1/sqrt(2) of the rms of its absolute value."""
        real = noise.calc_P_N(**self.ARGS, component="real")
        np.testing.assert_allclose(noise.calc_P_N(**self.ARGS, component="imag"), real)
        np.testing.assert_allclose(
            noise.calc_P_N(**self.ARGS, component="abs") / real, np.sqrt(2), rtol=1e-12
        )
        # in mK^2: (Tsys in mK)^2 / t_int, times the scalar
        np.testing.assert_allclose(
            noise.calc_P_N(**self.ARGS, component="abs"),
            2.0e9 * (300.0e3) ** 2 / 10.0,
            rtol=1e-12,
        )

    def test_delta_squared(self) -> None:
        """Delta^2(k) = k^3 P(k) / (2 pi^2)."""
        k = np.linspace(0.1, 2.0, 7)
        pk = noise.calc_P_N(**self.ARGS)
        np.testing.assert_allclose(
            noise.calc_P_N(**self.ARGS, form="DelSq", k=k),
            pk * k**3 / (2 * np.pi**2),
            rtol=1e-12,
        )


# System temperature of the simulated noise in test_noise_validation, in K
NOISE_VALIDATION_TSYS = 300.0


@pytest.fixture(scope="module")
def noise_validation_ds(
    beam_nf_dipole: PSpecBeamUV, uvd_zen_even_xx: UVData
) -> tuple[pspecdata.PSpecData, list, list]:
    """PSpecData of simulated noise (in mK) on the zen.even.xx baselines, and
    the baseline pairs of one redundant group."""
    uvd = testing.noise_sim(
        uvd_zen_even_xx,
        NOISE_VALIDATION_TSYS,
        beam_nf_dipole,
        seed=0,
        whiten=True,
        inplace=False,
        Nextend=9,
    )

    # get redundant baseline group
    reds, lens, angs = utils.get_reds(
        uvd, pick_data_ants=True, bl_len_range=(10, 20), bl_deg_range=(0, 1)
    )
    bls1, bls2, blps = utils.construct_blpairs(
        reds[0], exclude_auto_bls=True, exclude_permutations=True
    )

    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    ds.Jy_to_mK()
    return ds, bls1, bls2


@pytest.mark.parametrize("taper", ["none", "bh"])
@pytest.mark.parametrize("spw", [(0, 20), (100, 150)], ids=["spw_0_20", "spw_100_150"])
def test_noise_validation(
    noise_validation_ds: tuple[pspecdata.PSpecData, list, list],
    spw: tuple[int, int],
    taper: str,
) -> None:
    """Check that the analytic noise power P_N matches the RMS of power spectra
    of simulated noise, through a real beam model and Jy -> mK conversion.

    The noise of different baselines is independent, so the cross-baseline
    spectra are pure noise, and the rms of both their real and imaginary parts
    must equal P_N (component="real"), independently of the spectral window
    and of the taper.
    """
    ds, bls1, bls2 = noise_validation_ds
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper=taper,
        sampling=False,
        little_h=True,
        spw_ranges=[spw],
        verbose=False,
    )

    # get noise spectra from one of the blpairs
    P_N = list(
        uvp.generate_noise_spectra(
            0,
            ("xx", "xx"),
            NOISE_VALIDATION_TSYS,
            blpairs=uvp.get_blpairs()[:1],
            num_steps=2000,
            component="real",
        ).values()
    )[0][0, 0]

    # power spectra of every blpair, shape (Nblpairs, Ntimes, Ndlys)
    Pspec = np.array([uvp.get_data((0, bl, ("xx", "xx"))) for bl in uvp.get_blpairs()])
    for component in [Pspec.real, Pspec.imag]:
        # Mean square power per integration: integrations are independent
        # noise realisations, while blpairs sharing a baseline and (with a
        # taper) neighbouring delays are correlated.
        mean_square = np.mean(component**2, axis=(0, 2))
        P_rms = np.sqrt(mean_square.mean())
        # relative standard error of P_rms
        error = mean_square.std(ddof=1) / np.sqrt(len(mean_square)) / (2 * P_rms**2)
        assert np.abs(P_rms / P_N - 1) < 4 * error


@pytest.fixture(
    scope="module", params=[[(0, 20)], [(119, 140)]], ids=["spw_0_20", "spw_119_140"]
)
def analytic_noise_uvps(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    beam_nf_dipole: PSpecBeamUV,
    uvd_zen_even_xx: UVData,
) -> tuple[UVPSpec, UVPSpec, UVData]:
    """uvp (cov_model='autos') and uvp_fg (cov_model='foreground_dependent') power spectra, plus auto_Tsys, with uncorrected P_N/P_SN populated on uvp."""
    spw_ranges = request.param
    uvd = copy.deepcopy(uvd_zen_even_xx)

    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
        dsets_std=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
    )
    ds.Jy_to_mK()

    reds, lens, angs = utils.get_reds(
        uvd, pick_data_ants=True, bl_len_range=(10, 20), bl_deg_range=(0, 1)
    )
    bls1, bls2, blps = utils.construct_blpairs(
        reds[0], exclude_auto_bls=True, exclude_permutations=True
    )
    taper = "bh"
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper=taper,
        sampling=False,
        little_h=True,
        spw_ranges=spw_ranges,
        verbose=False,
        cov_model="autos",
        store_cov=True,
    )
    uvp_fg = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper=taper,
        sampling=False,
        little_h=True,
        spw_ranges=spw_ranges,
        verbose=False,
        cov_model="foreground_dependent",
        store_cov=True,
    )

    auto_Tsys = utils.uvd_to_Tsys(
        uvd,
        beam_nf_dipole,
        str(tmp_path_factory.mktemp("analytic_noise") / "test_uvd.uvh5"),
    )
    utils.uvp_noise_error(
        uvp, auto_Tsys, err_type=["P_N", "P_SN"], P_SN_correction=False
    )

    return uvp, uvp_fg, auto_Tsys


class TestAnalyticNoise:
    """Tests for the two forms of analytic noise calculation: QE-propagated autos (P_N) and Tsys-estimated (P_SN)."""

    def test_pn_consistency(
        self, analytic_noise_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """Check that the analytic P_N error matches the 1-sigma std from the autos-only covariance, to 1%."""
        uvp, uvp_fg, auto_Tsys = analytic_noise_uvps
        cov_diag = np.array(
            [
                np.diag(uvp.cov_array_real[0][i][:, :, 0])[:, None]
                for i in range(uvp.cov_array_real[0].shape[0])
            ]
        )
        stats_diag = uvp.stats_array["P_N"][0]
        frac_ratio = (cov_diag**0.5 - stats_diag) / stats_diag
        assert np.abs(frac_ratio).mean() < 0.01

    def test_psn_consistency(
        self, analytic_noise_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """Check that the analytic (uncorrected) P_SN error matches the 1-sigma std from the foreground-dependent covariance, to 1%."""
        uvp, uvp_fg, auto_Tsys = analytic_noise_uvps
        cov_diag = np.array(
            [
                np.diag(uvp_fg.cov_array_real[0][i][:, :, 0])[:, None]
                for i in range(uvp_fg.cov_array_real[0].shape[0])
            ]
        )
        stats_diag = uvp.stats_array["P_SN"][0]
        frac_ratio = (cov_diag**0.5 - stats_diag) / stats_diag
        assert np.abs(frac_ratio).mean() < 0.01

    def test_psn_correction_matches_pn_at_high_k(
        self, analytic_noise_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """Check that the bias-corrected P_SN converges to P_N at high delay, where foregrounds are negligible."""
        uvp, _, auto_Tsys = analytic_noise_uvps
        uvp = copy.deepcopy(uvp)
        auto_Tsys = copy.deepcopy(auto_Tsys)
        utils.uvp_noise_error(
            uvp, auto_Tsys, err_type=["P_N", "P_SN"], P_SN_correction=True
        )
        frac_ratio = (
            uvp.stats_array["P_SN"][0] - uvp.stats_array["P_N"][0]
        ) / uvp.stats_array["P_N"][0]
        dlys = uvp.get_dlys(0) * 1e9
        select = np.abs(dlys) > 3000
        assert np.abs(frac_ratio[:, select].mean()) < 1 / np.sqrt(uvp.Nbltpairs)

    def test_psn_correction_matches_pn_at_high_k_single_time(
        self, analytic_noise_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """Check that the high-k P_SN/P_N agreement still holds after selecting down to a single time."""
        uvp, _, auto_Tsys = analytic_noise_uvps
        uvp = copy.deepcopy(uvp)
        auto_Tsys = copy.deepcopy(auto_Tsys)
        uvp.select(times=uvp.time_avg_array[:1], inplace=True)
        auto_Tsys.select(times=auto_Tsys.time_array[:1], inplace=True)
        utils.uvp_noise_error(
            uvp, auto_Tsys, err_type=["P_N", "P_SN"], P_SN_correction=True
        )
        frac_ratio = (
            uvp.stats_array["P_SN"][0] - uvp.stats_array["P_N"][0]
        ) / uvp.stats_array["P_N"][0]
        dlys = uvp.get_dlys(0) * 1e9
        select = np.abs(dlys) > 3000
        assert np.abs(frac_ratio[:, select].mean()) < 1 / np.sqrt(uvp.Nbltpairs)


def check_corr_matrix(m: np.ndarray) -> None:
    """Perform checks of a matrix that establish it as a reasonable correlation."""
    assert m.ndim == 2
    assert m.shape[0] == m.shape[1]
    assert np.all(np.diag(m) == 1)
    # correlation coefficients are bounded by 1 in magnitude
    assert np.all(np.abs(m) <= 1 + 1e-12)
    np.testing.assert_array_almost_equal(m.T, m)
    # and a correlation matrix is positive semi-definite
    assert np.all(np.linalg.eigvalsh(m) >= -1e-10)


@pytest.mark.parametrize("n", [1, 6, 7])
@pytest.mark.parametrize("taper", ["blackmanharris", "hann", "blackman", "none"])
def test_get_approximate_corr(n: int, taper: str) -> None:
    """Check that get_approximate_delay_delay_corr_matrix returns a valid correlation matrix."""
    corr = noise.get_approximate_delay_delay_corr_matrix(taper, n)
    check_corr_matrix(corr)
    if taper == "none":
        # without a taper, the delay modes of white noise are uncorrelated
        np.testing.assert_allclose(corr, np.eye(n), atol=1e-12)

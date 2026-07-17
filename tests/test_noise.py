import copy
from pathlib import Path

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
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


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


def test_noise_validation(beam_nf_dipole: PSpecBeamUV) -> None:
    """Check that the analytic noise 1-sigma amplitude matches the RMS of a noise simulation realization."""
    # get simulated noise in Jy
    uvfile = str(DATA_PATH / "zen.even.xx.LST.1.28828.uvOCRSA")
    Tsys = 300.0  # Kelvin

    # generate noise
    seed = 0
    uvd = testing.noise_sim(
        uvfile, Tsys, beam_nf_dipole, seed=seed, whiten=True, inplace=False, Nextend=9
    )

    # get redundant baseline group
    reds, lens, angs = utils.get_reds(
        uvd, pick_data_ants=True, bl_len_range=(10, 20), bl_deg_range=(0, 1)
    )
    bls1, bls2, blps = utils.construct_blpairs(
        reds[0], exclude_auto_bls=True, exclude_permutations=True
    )

    # setup PSpecData
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    ds.Jy_to_mK()

    # get pspec
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper="none",
        sampling=False,
        little_h=True,
        spw_ranges=[(0, 50)],
        verbose=False,
    )

    # get noise spectra from one of the blpairs
    P_N = list(
        uvp.generate_noise_spectra(
            0,
            ("xx", "xx"),
            Tsys,
            blpairs=uvp.get_blpairs()[:1],
            num_steps=2000,
            component="real",
        ).values()
    )[0][0, 0]

    # get P_rms of real spectra for each baseline across time axis
    Pspec = np.array(
        [uvp.get_data((0, bl, ("xx", "xx"))).real for bl in uvp.get_blpairs()]
    )
    P_rms = np.sqrt(np.mean(np.abs(Pspec) ** 2))

    # assert close to P_N: 2%
    # This should be updated to be within standard error on P_rms
    # when the spw_range-variable pspec amplitude bug is resolved
    assert np.abs(P_rms - P_N) / P_N < 0.02


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
    assert np.all(m - np.eye(m.shape[0]) <= 1)
    np.testing.assert_array_almost_equal(m.T, m)


@pytest.mark.parametrize("n", [1, 6, 7])
@pytest.mark.parametrize("taper", ["blackmanharris", "none"])
def test_get_approximate_corr(n: int, taper: str) -> None:
    """Check that get_approximate_delay_delay_corr_matrix returns a valid correlation matrix."""
    corr = noise.get_approximate_delay_delay_corr_matrix(taper, n)
    check_corr_matrix(corr)

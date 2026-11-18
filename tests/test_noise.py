import copy

import astropy.cosmology.units as cu
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FLRW
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
def sense(beam_nf_pstokes: PSpecBeamUV, cosmo: FLRW) -> noise.Sensitivity:
    return noise.Sensitivity(beam=beam_nf_pstokes, cosmo=cosmo)


class TestSensitivity:
    def test_set_cosmology(self) -> None:
        """Check that set_cosmology accepts an astropy cosmology, its params dict and its stringified params."""
        sense = noise.Sensitivity()
        # a cosmology without radiation, which its params dict represents exactly
        C = conversions.LEGACY_COSMOLOGY
        sense.set_cosmology(C)
        assert sense.cosmo is C
        params = conversions.params_from_cosmology(C)
        assert sense.cosmo_params == str(params)
        sense.set_cosmology(str(params))
        assert isinstance(sense.cosmo, FLRW)
        assert sense.cosmo.is_equivalent(C)
        sense.set_cosmology(params)
        assert sense.cosmo.is_equivalent(C)

    def test_set_cosmology_cosmo_conversions_deprecated(self) -> None:
        """A Cosmo_Conversions object still works, with a DeprecationWarning."""
        sense = noise.Sensitivity()
        with pytest.warns(DeprecationWarning):
            C = conversions.Cosmo_Conversions(Om_L=0.7, Om_c=0.25)
        with pytest.warns(DeprecationWarning, match="astropy.cosmology"):
            sense.set_cosmology(C)
        assert isinstance(sense.cosmo, FLRW)
        assert conversions.params_from_cosmology(sense.cosmo) == pytest.approx(
            C.get_params(), rel=1e-12
        )

    def test_set_beam_with_cosmo(self, beam_nf_pstokes: PSpecBeamUV) -> None:
        """Check that set_beam aligns the beam's cosmology with self.cosmo when both are present."""
        sense = noise.Sensitivity()
        C = conversions.DEFAULT_COSMOLOGY
        sense.set_cosmology(C)

        beam = copy.deepcopy(beam_nf_pstokes)
        sense.set_beam(beam)
        assert sense.cosmo is sense.beam.cosmo
        beam.cosmo = C
        sense.set_beam(beam)
        assert sense.cosmo is sense.beam.cosmo

    def test_set_beam_without_cosmo(self, beam_nf_pstokes: PSpecBeamUV) -> None:
        """Check that set_beam attaches self.cosmo to a beam lacking a cosmo attribute, rather than raising."""
        sense = noise.Sensitivity()
        sense.set_cosmology(conversions.DEFAULT_COSMOLOGY)
        bm = copy.deepcopy(beam_nf_pstokes)
        delattr(bm, "cosmo")
        sense.set_beam(bm)
        assert bm.cosmo is sense.cosmo

    def test_scalar(self, sense: noise.Sensitivity) -> None:
        """Check that calc_scalar records the input subband and polarization."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        sense.calc_scalar(
            freqs * u.Hz, "pI", num_steps=5000, length_unit=u.Mpc / cu.littleh
        )
        assert np.isclose(freqs, sense.subband).all()
        assert sense.pol == "pI"

    def test_calc_p_n(self, beam_nf_pstokes: PSpecBeamUV) -> None:
        """Check that calc_P_N returns a scalar P_N in Pk form and a properly shaped, smaller DelSq array."""
        # the reference P_N is from v0.x, with its default cosmology
        sense = noise.Sensitivity(
            beam=copy.deepcopy(beam_nf_pstokes), cosmo=conversions.LEGACY_COSMOLOGY
        )
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        sense.calc_scalar(
            freqs * u.Hz, "pI", num_steps=5000, length_unit=u.Mpc / cu.littleh
        )

        k = np.linspace(0, 3, 10)
        Tsys = 500.0 * u.K
        t_int = 10.7 * u.s
        P_N = sense.calc_P_N(Tsys, t_int, Ncoherent=1, Nincoherent=1, form="Pk")
        assert isinstance(P_N, float)
        assert np.isclose(P_N, 642386932892.2921)
        Dsq = sense.calc_P_N(Tsys, t_int, k=k, Ncoherent=1, Nincoherent=1, form="DelSq")
        assert Dsq.shape == (10,)
        assert Dsq[1] < P_N

    def test_scalar_length_unit(self, sense: noise.Sensitivity) -> None:
        """Check calc_scalar's length_unit: littleh by default, Mpc = littleh / h^3, strings accepted."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        sense.calc_scalar(freqs * u.Hz, "pI", num_steps=2000)
        default_scalar = sense.scalar
        assert isinstance(default_scalar, float)
        assert sense.length_unit == u.Mpc / cu.littleh
        sense.calc_scalar(
            freqs * u.Hz, "pI", num_steps=2000, length_unit=u.Mpc / cu.littleh
        )
        assert sense.scalar == default_scalar
        sense.calc_scalar(freqs * u.Hz, "pI", num_steps=2000, length_unit=u.Mpc)
        assert sense.length_unit == u.Mpc
        h = sense.cosmo.h
        np.testing.assert_allclose(sense.scalar, default_scalar / h**3, rtol=1e-12)
        mpc_scalar = sense.scalar
        sense.calc_scalar(freqs * u.Hz, "pI", num_steps=2000, length_unit="Mpc")
        assert sense.scalar == mpc_scalar
        sense.calc_scalar(
            freqs * u.Hz, "pI", num_steps=2000, length_unit="Mpc / littleh"
        )
        assert sense.scalar == default_scalar
        # P_N follows the scalar's length unit: mK^2 L^3
        Pk_mpc = noise.calc_P_N(mpc_scalar, 300.0 * u.K, 10.7 * u.s)
        sense.calc_scalar(freqs * u.Hz, "pI", num_steps=2000, length_unit=u.Mpc)
        assert sense.calc_P_N(300.0 * u.K, 10.7 * u.s) == Pk_mpc

    @pytest.mark.parametrize("little_h", [True, False])
    def test_scalar_little_h_deprecated(
        self, sense: noise.Sensitivity, little_h: bool
    ) -> None:
        """Check that little_h warns at the caller's line, keeps its positional slot and matches length_unit."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        length_unit = conversions.length_unit_from_little_h(little_h)
        sense.calc_scalar(freqs * u.Hz, "pI", num_steps=2000, length_unit=length_unit)
        expected = sense.scalar
        with pytest.warns(
            DeprecationWarning, match="little_h argument is deprecated"
        ) as record:
            sense.calc_scalar(freqs * u.Hz, "pI", 2000, little_h)
        assert len(record) == 1
        assert record[0].filename == __file__
        assert sense.scalar == expected
        assert sense.length_unit == length_unit
        # agreeing pair is accepted
        with pytest.warns(DeprecationWarning, match="little_h argument is deprecated"):
            sense.calc_scalar(
                freqs * u.Hz, "pI", 2000, little_h=little_h, length_unit=length_unit
            )
        assert sense.scalar == expected

    def test_scalar_length_unit_errors(self, sense: noise.Sensitivity) -> None:
        """Check that a disagreeing little_h or a non-length unit raises."""
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match="disagrees"),
        ):
            sense.calc_scalar(freqs * u.Hz, "pI", little_h=True, length_unit=u.Mpc)
        with pytest.raises(ValueError, match="length_unit must be a length"):
            sense.calc_scalar(freqs * u.Hz, "pI", length_unit=u.s)


def test_noise_validation(beam_nf_dipole: PSpecBeamUV, uvd_zen_even_xx: UVData) -> None:
    """Check that the analytic noise 1-sigma amplitude matches the RMS of a noise simulation realization."""
    # get simulated noise in Jy (on a copy of zen.even.xx.LST.1.28828.uvOCRSA)
    Tsys = 300.0  # Kelvin

    # generate noise
    seed = 0
    uvd = testing.noise_sim(
        uvd_zen_even_xx,
        Tsys,
        beam_nf_dipole,
        seed=seed,
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
        length_unit=u.Mpc / cu.littleh,
        spw_ranges=[(0, 50)],
        verbose=False,
    )

    # get noise spectra from one of the blpairs
    P_N = list(
        uvp.generate_noise_spectra(
            0,
            ("xx", "xx"),
            Tsys * u.K,
            blpairs=uvp.get_blpairs()[:1],
            num_steps=2000,
            component="real",
        ).values()
    )[0][0, 0]
    assert P_N.unit == u.mK**2 * u.Mpc**3 / cu.littleh**3
    P_N = P_N.value

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
        length_unit=u.Mpc / cu.littleh,
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
        length_unit=u.Mpc / cu.littleh,
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
        dlys = uvp.get_dlys(0).to_value(u.ns)
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
        dlys = uvp.get_dlys(0).to_value(u.ns)
        select = np.abs(dlys) > 3000
        assert np.abs(frac_ratio[:, select].mean()) < 1 / np.sqrt(uvp.Nbltpairs)


@pytest.fixture(scope="module")
def noise_error_uvps(
    uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
) -> tuple[UVPSpec, UVPSpec, UVData]:
    """The same spectra normalised in Mpc^3 / littleh^3 and in Mpc^3 (data in mK), plus auto Tsys."""
    uvd = copy.deepcopy(uvd_zen_even_xx)
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    ds.Jy_to_mK()
    bls = [(37, 38), (38, 39), (52, 53)]
    bls1, bls2, _ = utils.construct_blpairs(
        bls, exclude_auto_bls=True, exclude_permutations=True
    )
    uvps = [
        ds.pspec(
            bls1,
            bls2,
            (0, 1),
            [("xx", "xx")],
            input_data_weight="identity",
            norm="I",
            taper="bh",
            length_unit=length_unit,
            spw_ranges=[(20, 40)],
            verbose=False,
        )
        for length_unit in (u.Mpc / cu.littleh, u.Mpc)
    ]
    auto_Tsys = utils.uvd_to_Tsys(uvd, beam_nf_dipole)
    return uvps[0], uvps[1], auto_Tsys


class TestNoiseErrorUnits:
    """uvp_noise_error stores P_N / P_SN in the length unit of the object's power spectra."""

    @staticmethod
    def _noise_error(uvp: UVPSpec, auto_Tsys: UVData, **kwargs) -> UVPSpec:
        uvp = copy.deepcopy(uvp)
        utils.uvp_noise_error(
            uvp, copy.deepcopy(auto_Tsys), err_type=["P_N", "P_SN"], **kwargs
        )
        return uvp

    def test_littleh_object(
        self, noise_error_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """A littleh object gets littleh noise by default, identical to passing the unit explicitly."""
        uvp_h, _, auto_Tsys = noise_error_uvps
        assert uvp_h.units == u.mK**2 * u.Mpc**3 / cu.littleh**3
        default = self._noise_error(uvp_h, auto_Tsys)
        explicit = self._noise_error(uvp_h, auto_Tsys, length_unit=u.Mpc / cu.littleh)
        for stat in ("P_N", "P_SN"):
            np.testing.assert_array_equal(
                default.stats_array[stat][0], explicit.stats_array[stat][0]
            )

    def test_mpc_object(
        self, noise_error_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        """An Mpc^3 object stores P_N in mK^2 Mpc^3: the littleh P_N times the scalar ratio (h^-3)."""
        _, uvp_mpc, auto_Tsys = noise_error_uvps
        assert uvp_mpc.units == u.mK**2 * u.Mpc**3
        default = self._noise_error(uvp_mpc, auto_Tsys)
        explicit = self._noise_error(uvp_mpc, auto_Tsys, length_unit="Mpc")
        np.testing.assert_array_equal(
            default.stats_array["P_N"][0], explicit.stats_array["P_N"][0]
        )
        littleh = self._noise_error(uvp_mpc, auto_Tsys, length_unit=u.Mpc / cu.littleh)
        # the factor between the two is the ratio of the noise scalars
        polpair = uvp_mpc.polpair_array[0]
        scalar_mpc = uvp_mpc.compute_scalar(
            0, polpair, num_steps=2000, noise_scalar=True, length_unit=u.Mpc
        )
        scalar_h = uvp_mpc.compute_scalar(
            0,
            polpair,
            num_steps=2000,
            noise_scalar=True,
            length_unit=u.Mpc / cu.littleh,
        )
        factor = scalar_mpc.value / scalar_h.value
        np.testing.assert_allclose(factor, uvp_mpc.cosmo.h**-3, rtol=1e-10)
        np.testing.assert_allclose(
            default.stats_array["P_N"][0],
            littleh.stats_array["P_N"][0] * factor,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("little_h", [True, False])
    def test_little_h_deprecated(
        self, noise_error_uvps: tuple[UVPSpec, UVPSpec, UVData], little_h: bool
    ) -> None:
        """little_h warns at the caller's line and gives the same result as length_unit."""
        uvp_h, uvp_mpc, auto_Tsys = noise_error_uvps
        uvp = uvp_h if little_h else uvp_mpc
        expected = self._noise_error(uvp, auto_Tsys)
        uvp_dep = copy.deepcopy(uvp)
        with pytest.warns(
            DeprecationWarning, match="little_h argument is deprecated"
        ) as record:
            utils.uvp_noise_error(
                uvp_dep, copy.deepcopy(auto_Tsys), "P_N", little_h=little_h
            )
        assert len(record) == 1
        assert record[0].filename == __file__
        np.testing.assert_array_equal(
            uvp_dep.stats_array["P_N"][0], expected.stats_array["P_N"][0]
        )

    def test_little_h_disagreement_raises(
        self, noise_error_uvps: tuple[UVPSpec, UVPSpec, UVData]
    ) -> None:
        uvp_h, _, auto_Tsys = noise_error_uvps
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match="disagrees"),
        ):
            utils.uvp_noise_error(
                copy.deepcopy(uvp_h),
                copy.deepcopy(auto_Tsys),
                little_h=True,
                length_unit=u.Mpc,
            )


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
@pytest.mark.parametrize("taper", ["blackmanharris", "none"])
def test_get_approximate_corr(n: int, taper: str) -> None:
    """Check that get_approximate_delay_delay_corr_matrix returns a valid correlation matrix."""
    corr = noise.get_approximate_delay_delay_corr_matrix(taper, n)
    check_corr_matrix(corr)

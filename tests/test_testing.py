import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hera_cal import redcal
from pyuvdata import UVData

from hera_pspec import PSpecBeamUV, conversions, testing, uvpspec
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


# Path to the zen.even.xx.LST.1.28828.uvOCRSA miriad data file.
FNAME = str(DATA_PATH / "zen.even.xx.LST.1.28828.uvOCRSA")
# Path to the corresponding zen.even.std.xx.LST.1.28828.uvOCRSA std-dev file.
FNAME_STD = str(DATA_PATH / "zen.even.std.xx.LST.1.28828.uvOCRSA")
# Path to the HERA_NF_dipole_power beamfits file (for testing beam-as-path inputs).
DIPOLE_BEAMFILE_PATH = str(DATA_PATH / "HERA_NF_dipole_power.beamfits")
# Path to the HERA_NF_pstokes_power beamfits file.
PSTOKES_BEAMFILE_PATH = str(DATA_PATH / "HERA_NF_pstokes_power.beamfits")


@pytest.fixture
def uvd_dual_pol(uvd_zen_even_xx: UVData) -> UVData:
    """uvd_zen_even_xx duplicated onto a second (dummy) polarization."""
    uvd2 = copy.deepcopy(uvd_zen_even_xx)
    uvd2.polarization_array[0] = 1
    uvd2 += uvd_zen_even_xx
    return uvd2


class TestBuildVanillaUvpspec:
    def test_default(self) -> None:
        """Check that build_vanilla_uvpspec returns a UVPSpec/Cosmo_Conversions pair with the cosmology attached."""
        uvp, cosmo = testing.build_vanilla_uvpspec()
        assert isinstance(uvp, uvpspec.UVPSpec)
        assert isinstance(cosmo, conversions.Cosmo_Conversions)
        assert uvp.cosmo == cosmo

    def test_with_beam(self, beam_nf_dipole: PSpecBeamUV) -> None:
        """Check that the OmegaP array on the output UVPSpec matches the supplied beam's Omegas."""
        uvp, _ = testing.build_vanilla_uvpspec(beam=beam_nf_dipole)
        beam_OP = beam_nf_dipole.get_Omegas(uvp.polpair_array[0])[0]
        assert beam_OP.tolist() == uvp.OmegaP.tolist()


class TestUvpspecFromData:
    def test_basic_execution(self, beam_nf_dipole: PSpecBeamUV) -> None:
        """Check that uvpspec_from_data with a filename selects the right delay range and blpairs."""
        uvp = testing.uvpspec_from_data(
            FNAME,
            [(37, 38), (38, 39), (52, 53), (53, 54)],
            beam=beam_nf_dipole,
            spw_ranges=[(50, 100)],
        )
        assert uvp.Nfreqs == 50
        assert np.unique(uvp.blpair_array).tolist() == [
            137138138139,
            137138152153,
            137138153154,
            138139152153,
            138139153154,
            152153153154,
        ]

    def test_string_and_object_inputs_equivalent(
        self, beam_nf_dipole: PSpecBeamUV, uvd_zen_even_xx: UVData
    ) -> None:
        """Check that filename/UVData and beam-object/beam-path inputs are interchangeable."""
        uvp = testing.uvpspec_from_data(
            FNAME,
            [(37, 38), (38, 39), (52, 53), (53, 54)],
            beam=beam_nf_dipole,
            spw_ranges=[(50, 100)],
        )
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            [(37, 38), (38, 39), (52, 53), (53, 54)],
            beam=DIPOLE_BEAMFILE_PATH,
            spw_ranges=[(50, 100)],
        )
        uvp.history = ""
        uvp2.history = ""
        assert uvp == uvp2

    def test_multiple_bl_groups(
        self, beam_nf_dipole: PSpecBeamUV, uvd_zen_even_xx: UVData
    ) -> None:
        """Check that feeding multiple redundant baseline groups only uses the expected baselines."""
        antpos, ants = uvd_zen_even_xx.get_enu_data_ants()
        reds = redcal.get_pos_reds(dict(zip(ants, antpos)))
        uvp = testing.uvpspec_from_data(
            FNAME, reds[:3], beam=beam_nf_dipole, spw_ranges=[(50, 100)]
        )
        expected_bls = {
            137138,
            137151,
            137152,
            138139,
            138152,
            138153,
            139153,
            139154,
            151152,
            151167,
            152153,
            152167,
            152168,
            153154,
            153168,
            153169,
            154169,
            167168,
            168169,
        }
        assert set(uvp.bl_array) <= expected_bls
        assert uvp.Nblpairs == 51

    @pytest.mark.parametrize(
        "bl_grps,match",
        [
            ((37, 38), "bl_grps must be a list"),
            ([([37, 38], [38, 39])], "length of bls1 must equal"),
            ([[[37, 38], [38, 39]]], "list of lists of tuples"),
        ],
    )
    def test_raises_on_invalid_bl_grps(self, bl_grps: Any, match: str) -> None:
        """Check that malformed bl_grps inputs raise descriptive AssertionErrors."""
        with pytest.raises(AssertionError, match=match):
            testing.uvpspec_from_data(FNAME, bl_grps)

    def test_with_data_std(self, beam_nf_dipole: PSpecBeamUV) -> None:
        """Check that passing data_std enables covariance storage with cov_model='dsets'."""
        uvp = testing.uvpspec_from_data(
            FNAME,
            [(37, 38), (38, 39), (52, 53), (53, 54)],
            data_std=FNAME_STD,
            beam=beam_nf_dipole,
            spw_ranges=[(20, 28)],
        )
        assert uvp.cov_model == "dsets"
        assert uvp.cov_array_real
        assert uvp.cov_array_imag


class TestNoiseSim:
    def test_amplitude(self, uvd_dual_pol: UVData) -> None:
        """Check that noise_sim produces noise with the expected std and 1/sqrt(2) auto/cross-pol scaling."""
        uvn = testing.noise_sim(uvd_dual_pol, 300.0, seed=0, whiten=True, inplace=False)
        assert uvn.Ntimes == uvd_dual_pol.Ntimes
        assert uvn.Nfreqs == uvd_dual_pol.Nfreqs
        assert uvn.Nbls == uvd_dual_pol.Nbls
        assert uvn.Npols == uvd_dual_pol.Npols
        np.testing.assert_almost_equal(
            np.std(uvn.data_array[..., 1].real), 0.20655731998619664
        )
        np.testing.assert_almost_equal(
            np.std(uvn.data_array[..., 1].imag), 0.20728471891024444
        )
        np.testing.assert_almost_equal(
            np.std(uvn.data_array[..., 0].real) / np.std(uvn.data_array[..., 1].real),
            1 / np.sqrt(2),
            decimal=2,
        )

    def test_seed_and_inplace(self, uvd_dual_pol: UVData) -> None:
        """Check that seed=0 matches np.random.seed(0) global state used with seed=None and inplace=True."""
        uvn = testing.noise_sim(uvd_dual_pol, 300.0, seed=0, whiten=True, inplace=False)

        np.random.seed(0)
        uvn2 = copy.deepcopy(uvd_dual_pol)
        testing.noise_sim(uvn2, 300.0, seed=None, whiten=True, inplace=True)
        assert uvn == uvn2

    def test_with_beam(self, uvd_zen_even_xx: UVData) -> None:
        """Check that supplying a beam file converts the simulated noise to Jy."""
        uvn = testing.noise_sim(
            copy.deepcopy(uvd_zen_even_xx),
            300.0,
            DIPOLE_BEAMFILE_PATH,
            seed=0,
            whiten=True,
            inplace=False,
        )
        assert uvn.vis_units == "Jy"

    def test_tsys_scaling(self, uvd_dual_pol: UVData) -> None:
        """Check that doubling Tsys doubles the simulated noise amplitude."""
        uvn3 = testing.noise_sim(
            uvd_dual_pol, 2 * 300.0, seed=0, whiten=True, inplace=False
        )
        np.testing.assert_almost_equal(
            np.std(uvn3.data_array[..., 1].real), 2 * 0.20655731998619664
        )
        np.testing.assert_almost_equal(
            np.std(uvn3.data_array[..., 1].imag), 2 * 0.20728471891024444
        )

    def test_nextend(self, uvd_zen_even_xx: UVData) -> None:
        """Check that Nextend replicates the time axis by the requested factor."""
        uvn = testing.noise_sim(
            uvd_zen_even_xx, 300.0, seed=0, whiten=True, inplace=False, Nextend=4
        )
        assert uvn.Ntimes == uvd_zen_even_xx.Ntimes * 5
        assert uvn.Nfreqs == uvd_zen_even_xx.Nfreqs
        assert uvn.Nbls == uvd_zen_even_xx.Nbls
        assert uvn.Npols == uvd_zen_even_xx.Npols


def test_sky_noise_jy_autos(uvd_zen_even_xx: UVData) -> None:
    """Check that sky_noise_jy_autos returns finite noise values for an analytic OmegaP function."""
    lsts = np.unique(uvd_zen_even_xx.lst_array)
    freqs = np.unique(uvd_zen_even_xx.freq_array)
    int_time = np.median(uvd_zen_even_xx.integration_time)
    channel_width = np.mean(np.diff(freqs))

    def omega_p(freq: float) -> float:
        return 0.05 * (freq / 100.0e6) ** -1.0

    n = testing.sky_noise_jy_autos(
        lsts,
        freqs,
        autovis=150.0,
        omega_p=omega_p,
        integration_time=int_time,
        channel_width=channel_width,
    )

    assert np.all(~np.isnan(n))
    assert np.all(~np.isinf(n))


class TestSkyNoiseSim:
    def test_basic(self, uvd_zen_even_xx: UVData) -> None:
        """Check that sky_noise_sim perturbs cross-correlation data relative to the input."""
        np.random.seed(0)
        sim = testing.sky_noise_sim(
            FNAME,
            DIPOLE_BEAMFILE_PATH,
            cov_amp=1000,
            cov_length_scale=10,
            constant_in_time=True,
            divide_by_nsamp=False,
        )
        for bl in sim.get_antpairpols():
            if bl[0] != bl[1]:
                assert np.all(
                    ~np.isclose(sim.get_data(bl), uvd_zen_even_xx.get_data(bl))
                )

    def test_pseudo_stokes(self, uvd_zen_even_xx: UVData) -> None:
        """Check that sky_noise_sim works on pseudo-Stokes (dual-pol-combined) data."""
        np.random.seed(0)
        uvd2, uvd2b = copy.deepcopy(uvd_zen_even_xx), copy.deepcopy(uvd_zen_even_xx)
        uvd2.polarization_array[0] = 1
        uvd2b.polarization_array[0] = 2
        uvd2 += uvd2b
        sim2 = testing.sky_noise_sim(
            uvd2,
            PSTOKES_BEAMFILE_PATH,
            cov_amp=1000,
            cov_length_scale=10,
            constant_in_time=True,
            divide_by_nsamp=False,
        )
        for bl in sim2.get_antpairpols():
            if bl[0] != bl[1]:
                assert np.all(~np.isclose(sim2.get_data(bl), uvd2.get_data(bl)))

    def test_divide_by_nsamp(self, uvd_zen_even_xx: UVData) -> None:
        """Check that divide_by_nsamp zeroes the simulated noise where nsample is zero."""
        sim3 = testing.sky_noise_sim(
            uvd_zen_even_xx,
            DIPOLE_BEAMFILE_PATH,
            cov_amp=0,
            cov_length_scale=10,
            constant_in_time=True,
            divide_by_nsamp=True,
        )
        # assert noise in channel 104 is zero (because nsample = 0)
        assert np.isclose(sim3.get_data(53, 69)[:, 104], 0).all()

    def test_constant_in_time_and_per_bl(self, uvd_zen_even_xx: UVData) -> None:
        """Check that constant_in_time and constant_per_bl make the noise constant across time and baseline."""
        sim3 = copy.deepcopy(uvd_zen_even_xx)
        # set integration_time to inf so the nsamp-weighted noise term vanishes,
        # leaving only the (constant) foreground covariance contribution
        sim3.integration_time[:] = np.inf
        sim3 = testing.sky_noise_sim(
            sim3,
            DIPOLE_BEAMFILE_PATH,
            cov_amp=1000,
            cov_length_scale=10,
            constant_in_time=True,
            constant_per_bl=True,
            divide_by_nsamp=True,
        )
        d = sim3.get_data(52, 53)
        assert np.isclose(d - d[0], 0).all()
        assert np.isclose(sim3.get_data(52, 53) - sim3.get_data(67, 68), 0).all()

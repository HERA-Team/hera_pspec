import pytest
from hera_pspec.data import DATA_PATH
from .. import testing, uvpspec, conversions, pspecbeam, utils
import os
from pyuvdata import UVData, UVBeam
import numpy as np
from hera_cal import redcal
import copy


def test_build_vanilla_uvpspec():
    uvp, cosmo = testing.build_vanilla_uvpspec()
    assert isinstance(uvp, uvpspec.UVPSpec)
    assert isinstance(cosmo, conversions.Cosmo_Conversions)
    assert uvp.cosmo == cosmo

    beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits'))
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)
    beam_OP = beam.get_Omegas(uvp.polpair_array[0])[0]
    assert beam_OP.tolist() == uvp.OmegaP.tolist()

def test_uvpspec_from_data():
    # get data
    fname = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    fname_std = os.path.join(DATA_PATH, "zen.even.std.xx.LST.1.28828.uvOCRSA")
    uvd = UVData()
    uvd.read_miriad(fname)
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)

    # test basic execution
    uvp = testing.uvpspec_from_data(fname, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beam, spw_ranges=[(50, 100)])
    assert uvp.Nfreqs == 50
    assert np.unique(uvp.blpair_array).tolist() == [137138138139, 137138152153, 137138153154, 138139152153, 138139153154, 152153153154]
    uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beamfile, spw_ranges=[(50, 100)])
    uvp.history = ''
    uvp2.history = ''
    assert uvp == uvp2

    # test multiple bl groups
    antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
    reds = redcal.get_pos_reds(dict(zip(ants, antpos)))
    uvp = testing.uvpspec_from_data(fname, reds[:3], beam=beam, spw_ranges=[(50, 100)])
    assert len(set(uvp.bl_array) - set([137138, 137151, 137152, 138139, 138152, 138153, 139153, 139154,
                                                 151152, 151167, 152153, 152167, 152168, 153154, 153168, 153169,
                                                 154169, 167168, 168169])) == 0
    assert uvp.Nblpairs == 51

    # test exceptions
    pytest.raises(AssertionError, testing.uvpspec_from_data, fname, (37, 38))
    pytest.raises(AssertionError, testing.uvpspec_from_data, fname, [([37, 38], [38, 39])])
    pytest.raises(AssertionError, testing.uvpspec_from_data, fname, [[[37, 38], [38, 39]]])

    # test std
    uvp = testing.uvpspec_from_data(fname, [(37, 38), (38, 39), (52, 53), (53, 54)],
                                    data_std=fname_std, beam=beam, spw_ranges=[(20,28)])

def test_noise_sim():
    uvd = UVData()
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd.read_miriad(uvfile)

    # test noise amplitude
    uvd2 = copy.deepcopy(uvd)
    uvd2.polarization_array[0] = 1
    uvd2 += uvd
    uvn = testing.noise_sim(uvd2, 300.0, seed=0, whiten=True, inplace=False)
    assert uvn.Ntimes == uvd2.Ntimes
    assert uvn.Nfreqs == uvd2.Nfreqs
    assert uvn.Nbls == uvd2.Nbls
    assert uvn.Npols == uvd2.Npols
    np.testing.assert_almost_equal(np.std(uvn.data_array[:, :, :, 1].real), 0.20655731998619664)
    np.testing.assert_almost_equal(np.std(uvn.data_array[:, :, :, 1].imag), 0.20728471891024444)
    np.testing.assert_almost_equal(np.std(uvn.data_array[:, :, :, 0].real) / np.std(uvn.data_array[:, :, :, 1].real),
                           1/np.sqrt(2), decimal=2)

    # test seed and inplace
    np.random.seed(0)
    uvn2 = copy.deepcopy(uvd2)
    testing.noise_sim(uvn2, 300.0, seed=None, whiten=True, inplace=True)
    assert uvn == uvn2

    # Test with a beam!
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    uvn = testing.noise_sim(copy.deepcopy(uvd), 300.0, beamfile, seed=0, whiten=True, inplace=False)
    assert uvn.vis_units == 'Jy'

    # test Tsys scaling
    uvn3 = testing.noise_sim(uvd2, 2*300.0, seed=0, whiten=True, inplace=False)
    np.testing.assert_almost_equal(np.std(uvn3.data_array[:, :, :, 1].real), 2*0.20655731998619664)
    np.testing.assert_almost_equal(np.std(uvn3.data_array[:, :, :, 1].imag), 2*0.20728471891024444)

    # test Nextend
    uvn = testing.noise_sim(uvd, 300.0, seed=0, whiten=True, inplace=False, Nextend=4)
    assert uvn.Ntimes == uvd.Ntimes*5
    assert uvn.Nfreqs == uvd.Nfreqs
    assert uvn.Nbls == uvd.Nbls
    assert uvn.Npols == uvd.Npols


def test_sky_noise_jy_autos():
    
    # Load data
    uvd = UVData()
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd.read_miriad(uvfile)
    
    # Get input arrays
    lsts = np.unique(uvd.lst_array)
    freqs = np.unique(uvd.freq_array)
    int_time = np.median(uvd.integration_time)
    channel_width = np.mean(np.diff(freqs))
    
    # Callable beam function
    omega_p = lambda freq: 0.05 * (freq / 100.e6)**-1.
    
    # Call function
    n = testing.sky_noise_jy_autos(
            lsts,
            freqs,
            autovis=150.,
            omega_p=omega_p,
            integration_time=int_time,
            channel_width=channel_width
        )
    
    # Check that results are finite
    assert np.all(~np.isnan(n))
    assert np.all(~np.isinf(n))


def test_sky_noise_sim():
    uvd = UVData()
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd.read_miriad(uvfile)
    beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    beam_ps = os.path.join(DATA_PATH, "HERA_NF_pstokes_power.beamfits")

    # basic test
    np.random.seed(0)
    sim = testing.sky_noise_sim(uvd, beam, cov_amp=1000, cov_length_scale=10, constant_in_time=True,
                                divide_by_nsamp=False)
    # assert something was inserted
    for bl in sim.get_antpairpols():
        if bl[0] != bl[1]:
            assert np.all(~np.isclose(sim.get_data(bl), uvd.get_data(bl)))

    # try with psuedo stokes
    np.random.seed(0)
    uvd2, uvd2b = copy.deepcopy(uvd), copy.deepcopy(uvd)
    uvd2.polarization_array[0] = 1
    uvd2b.polarization_array[0] = 2
    uvd2 += uvd2b
    sim2 = testing.sky_noise_sim(uvd2, beam_ps, cov_amp=1000, cov_length_scale=10, constant_in_time=True,
                                 divide_by_nsamp=False)
    # assert something was inserted
    for bl in sim2.get_antpairpols():
        if bl[0] != bl[1]:
            assert np.all(~np.isclose(sim2.get_data(bl), uvd2.get_data(bl)))

    # try divide by nsamp : set cov_amp=0 so we are only probing noise
    sim3 = testing.sky_noise_sim(uvd, beam, cov_amp=0, cov_length_scale=10, constant_in_time=True,
                                divide_by_nsamp=True)

    # assert noise in channel 104 is zero (because nsample = 0)
    assert np.isclose(sim3.get_data(53, 69)[:, 104], 0).all()

    # test constant across time and bl
    sim3 = copy.deepcopy(uvd)
    sim3.integration_time[:] = np.inf  # set int_time to zero so we are only probing fg signal
    sim3 = testing.sky_noise_sim(sim3, beam, cov_amp=1000, cov_length_scale=10, constant_in_time=True,
                                 constant_per_bl=True, divide_by_nsamp=True)

    # assert constant in time and across baseline
    d = sim3.get_data(52, 53)
    assert np.isclose(d - d[0], 0).all()
    assert np.isclose(sim3.get_data(52, 53) - sim3.get_data(67, 68), 0).all()

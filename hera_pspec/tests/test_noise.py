import unittest
import pytest
import numpy as np
import os
import sys
import copy
import h5py
from collections import OrderedDict as odict
from pyuvdata import UVData

from hera_pspec.data import DATA_PATH
from .. import uvpspec, conversions, pspecdata, pspecbeam, noise, testing, utils


class Test_Sensitivity(unittest.TestCase):
    """
    Test noise.Sensitivity object
    """
    def setUp(self):
        self.cosmo = conversions.Cosmo_Conversions()
        self.beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                              'HERA_NF_pstokes_power.beamfits'))
        self.sense = noise.Sensitivity(beam=self.beam, cosmo=self.cosmo)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_set(self):
        sense = noise.Sensitivity()

        C = conversions.Cosmo_Conversions()
        sense.set_cosmology(C)
        assert C.get_params() == sense.cosmo.get_params()
        params = str(C.get_params())
        sense.set_cosmology(params)
        assert C.get_params() == sense.cosmo.get_params()

        sense.set_beam(self.beam)
        assert sense.cosmo.get_params() == sense.beam.cosmo.get_params()
        self.beam.cosmo = C
        sense.set_beam(self.beam)
        assert sense.cosmo.get_params() == sense.beam.cosmo.get_params()

        bm = copy.deepcopy(self.beam)
        delattr(bm, 'cosmo')
        sense.set_beam(bm)

    def test_scalar(self):
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'pI', num_steps=5000, little_h=True)
        assert np.isclose(freqs, self.sense.subband).all()
        assert self.sense.pol == 'pI'

    def test_calc_P_N(self):

        # calculate scalar
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'pI', num_steps=5000, little_h=True)

        # basic execution
        k = np.linspace(0, 3, 10)
        Tsys = 500.0
        t_int = 10.7
        P_N = self.sense.calc_P_N(Tsys, t_int, Ncoherent=1, Nincoherent=1,
                                  form='Pk')
        assert isinstance(P_N, float)
        assert np.isclose(P_N, 642386932892.2921)
        # calculate DelSq
        Dsq = self.sense.calc_P_N(Tsys, t_int, k=k, Ncoherent=1,
                                  Nincoherent=1, form='DelSq')
        assert Dsq.shape == (10,)
        assert Dsq[1] < P_N


def test_noise_validation():
    """
    make sure that the noise.py code produces
    correct noise 1-sigma amplitude using a
    noise simulation.
    """
    # get simulated noise in Jy
    bfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(bfile)
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    Tsys = 300.0  # Kelvin

    # generate noise
    seed = 0
    uvd = testing.noise_sim(uvfile, Tsys, beam, seed=seed, whiten=True,
                            inplace=False, Nextend=9)

    # get redundant baseline group
    reds, lens, angs = utils.get_reds(uvd, pick_data_ants=True,
                                      bl_len_range=(10, 20),
                                      bl_deg_range=(0, 1))
    bls1, bls2, blps = utils.construct_blpairs(reds[0], exclude_auto_bls=True,
                                               exclude_permutations=True)

    # setup PSpecData
    ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
                             wgts=[None, None], beam=beam)
    ds.Jy_to_mK()

    # get pspec
    uvp = ds.pspec(bls1, bls2, (0, 1), [('xx', 'xx')], input_data_weight='identity', norm='I',
                   taper='none', sampling=False, little_h=True, spw_ranges=[(0, 50)], verbose=False)

    # get noise spectra from one of the blpairs
    P_N = list(uvp.generate_noise_spectra(0, ('xx','xx'), Tsys,
                                          blpairs=uvp.get_blpairs()[:1], num_steps=2000,
                                          component='real').values())[0][0, 0]

    # get P_rms of real spectra for each baseline across time axis
    Pspec = np.array([uvp.get_data((0, bl, ('xx', 'xx'))).real for bl in uvp.get_blpairs()])
    P_rms = np.sqrt(np.mean(np.abs(Pspec)**2))

    # assert close to P_N: 2%
    # This should be updated to be within standard error on P_rms
    # when the spw_range-variable pspec amplitude bug is resolved
    assert np.abs(P_rms - P_N) / P_N < 0.02


def test_analytic_noise():
    """
    Test the two forms of analytic noise calculation
    one using QE propagated from auto-correlation
    second using P_N from Tsys estimate
    """
    bfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(bfile)
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd = UVData()
    uvd.read(uvfile)

    # setup PSpecData
    ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
                             wgts=[None, None], beam=beam,
                             dsets_std=[copy.deepcopy(uvd), copy.deepcopy(uvd)])
    ds.Jy_to_mK()

    # get pspec
    reds, lens, angs = utils.get_reds(uvd, pick_data_ants=True,
                                      bl_len_range=(10, 20),
                                      bl_deg_range=(0, 1))
    bls1, bls2, blps = utils.construct_blpairs(reds[0], exclude_auto_bls=True,
                                               exclude_permutations=True)
    taper = 'bh'
    Nchan = 20
    uvp = ds.pspec(bls1, bls2, (0, 1), [('xx', 'xx')], input_data_weight='identity', norm='I',
                   taper=taper, sampling=False, little_h=True, spw_ranges=[(0, Nchan)], verbose=False,
                   cov_model='autos', store_cov=True)
    uvp_fg = ds.pspec(bls1, bls2, (0, 1), [('xx', 'xx')], input_data_weight='identity', norm='I',
                   taper=taper, sampling=False, little_h=True, spw_ranges=[(0, Nchan)], verbose=False,
                   cov_model='foreground_dependent', store_cov=True)

    # get P_N estimate
    auto_Tsys = utils.uvd_to_Tsys(uvd, beam, os.path.join(DATA_PATH, "test_uvd.uvh5"))
    assert os.path.exists(os.path.join(DATA_PATH, "test_uvd.uvh5"))
    utils.uvp_noise_error(uvp, auto_Tsys, err_type=['P_N','P_SN'], P_SN_correction=False)

    # check consistency of 1-sigma standard dev. to 1%
    cov_diag = uvp.cov_array_real[0][:, range(Nchan), range(Nchan)]
    stats_diag = uvp.stats_array['P_N'][0]
    frac_ratio = (cov_diag**0.5 - stats_diag) / stats_diag
    assert np.abs(frac_ratio).mean() < 0.01

    ## check P_SN consistency of 1-sigma standard dev. to 1%
    cov_diag = uvp_fg.cov_array_real[0][:, range(Nchan), range(Nchan)]
    stats_diag = uvp.stats_array['P_SN'][0]
    frac_ratio = (cov_diag**0.5 - stats_diag) / stats_diag
    assert np.abs(frac_ratio).mean() < 0.01

    # now compute unbiased P_SN and check that it matches P_N at high-k
    utils.uvp_noise_error(uvp, auto_Tsys, err_type=['P_N','P_SN'], P_SN_correction=True)
    frac_ratio = (uvp.stats_array["P_SN"][0] - uvp.stats_array["P_N"][0]) / uvp.stats_array["P_N"][0]
    dlys = uvp.get_dlys(0) * 1e9
    select = np.abs(dlys) > 3000
    assert np.abs(frac_ratio[:, select].mean()) < 1 / np.sqrt(uvp.Nblpairts)

    # test single time
    uvp.select(times=uvp.time_avg_array[:1], inplace=True)
    auto_Tsys.select(times=auto_Tsys.time_array[:1], inplace=True)
    utils.uvp_noise_error(uvp, auto_Tsys, err_type=['P_N','P_SN'], P_SN_correction=True)
    frac_ratio = (uvp.stats_array["P_SN"][0] - uvp.stats_array["P_N"][0]) / uvp.stats_array["P_N"][0]
    dlys = uvp.get_dlys(0) * 1e9
    select = np.abs(dlys) > 3000
    assert np.abs(frac_ratio[:, select].mean()) < 1 / np.sqrt(uvp.Nblpairts)

    # clean up
    os.remove(os.path.join(DATA_PATH, "test_uvd.uvh5"))

import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import uvpspec, conversions, pspecdata, pspecbeam, noise, testing, utils
import copy
import h5py
from collections import OrderedDict as odict
from pyuvdata import UVData


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
        nt.assert_equal(C.get_params(), sense.cosmo.get_params())
        params = str(C.get_params())
        sense.set_cosmology(params)
        nt.assert_equal(C.get_params(), sense.cosmo.get_params())

        sense.set_beam(self.beam)
        nt.assert_equal(sense.cosmo.get_params(), sense.beam.cosmo.get_params())
        self.beam.cosmo = C
        sense.set_beam(self.beam)
        nt.assert_equal(sense.cosmo.get_params(), sense.beam.cosmo.get_params())

    def test_scalar(self):
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'pI', num_steps=5000, little_h=True)
        nt.assert_true(np.isclose(freqs, self.sense.subband).all())
        nt.assert_true(self.sense.pol, 'pI')

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
        nt.assert_true(isinstance(P_N, (float, np.float)))
        nt.assert_true(np.isclose(P_N, 908472312787.53491))
        # calculate DelSq
        Dsq = self.sense.calc_P_N(Tsys, t_int, k=k, Ncoherent=1, 
                                  Nincoherent=1, form='DelSq')
        nt.assert_equal(Dsq.shape, (10,))
        nt.assert_true(Dsq[1] < P_N)


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
                                          blpairs=uvp.get_blpairs()[:1], 
                                          num_steps=2000).values())[0][0, 0]

    # get P_std of real spectra for each baseline across time axis
    P_stds = np.array([np.std(uvp.get_data((0, bl, ('xx','xx'))).real, axis=1) 
                       for bl in uvp.get_blpairs()])

    # get average P_std_avg and its standard error
    P_std_avg = np.mean(P_stds)
    
    # assert close to P_N: 2%
    # This should be updated to be within standard error on P_std_avg
    # when the spw_range-variable pspec amplitude bug is resolved
    nt.assert_true(np.abs(P_std_avg - P_N) / P_N < 0.02)






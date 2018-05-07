import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import uvpspec, conversions, parameter, pspecbeam, noise
import copy
import h5py
from collections import OrderedDict as odict
from pyuvdata import UVData
from hera_pspec.tests import noise_sim

class Test_Sensitivity(unittest.TestCase):
    """ Test noise.Sensitivity object """

    def setUp(self):
        self.cosmo = conversions.Cosmo_Conversions()
        self.beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits'))
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
        self.sense.calc_scalar(freqs, 'I', num_steps=5000, little_h=True)
        nt.assert_true(np.isclose(freqs, self.sense.subband).all())
        nt.assert_true(self.sense.pol, 'I')

    def test_calc_P_N(self):
        # calculate scalar
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'I', num_steps=5000, little_h=True)
        # basic execution 
        k = np.linspace(0, 3, 10)
        Tsys = 500.0
        t_int = 10.7
        P_N = self.sense.calc_P_N(Tsys, t_int, Ncoherent=1, Nincoherent=1, form='Pk')
        nt.assert_true(isinstance(P_N, (float, np.float)))
        nt.assert_true(np.isclose(P_N, 906609626029.72791))
        # calculate DelSq
        Dsq = self.sense.calc_P_N(Tsys, t_int, k=k, Ncoherent=1, Nincoherent=1, form='DelSq')
        nt.assert_equal(Dsq.shape, (10,))
        nt.assert_true(Dsq[1] < P_N)

def test_noise_sim():
    # run noise simulation
    ds, uvp = noise_sim()

    # get standard dev of real(data)
    dstd = np.std(np.real(uvp.data_array[0].ravel()))

    # get noise.py estimate
    blp = uvp.blpair_to_antnums(uvp.blpair_array[0])
    PN = uvp.generate_noise_spectra(0, 'xx', 100.0, blpairs=[uvp.blpair_array[0]], 
                                    little_h=True, form='Pk', real=True)
    pn = PN[uvp.blpair_array[0]][0,0]

    # check that noise.py agrees w/ standard dev of real(pspectra) to within 5%
    nt.assert_true(np.abs( (dstd - pn) / pn ) < 0.05)








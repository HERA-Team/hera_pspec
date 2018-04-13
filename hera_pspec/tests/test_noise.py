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

class Test_Sense(unittest.TestCase):
    """ Test noise.Sense object """

    def setUp(self):
        self.cosmo = conversions.Cosmo_Conversions()
        self.beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits'))
        self.sense = noise.Sense(beam=self.beam, cosmo=self.cosmo)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_add(self):
        sense = noise.Sense()

        C = conversions.Cosmo_Conversions()
        sense.add_cosmology(C)
        nt.assert_equal(C.get_params(), sense.cosmo.get_params())
        params = str(C.get_params())
        sense.add_cosmology(params)
        nt.assert_equal(C.get_params(), sense.cosmo.get_params())

        sense.add_beam(self.beam)
        nt.assert_equal(sense.cosmo.get_params(), sense.beam.cosmo.get_params())
        self.beam.cosmo = C
        sense.add_beam(self.beam)
        nt.assert_equal(sense.cosmo.get_params(), sense.beam.cosmo.get_params())

    def test_scalar(self):
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'pseudo_I', num_steps=5000, little_h=True)
        nt.assert_true(np.isclose(freqs, self.sense.subband).all())
        nt.assert_true(self.sense.stokes, 'I')

    def test_calc_P_N(self):
        # calculate scalar
        freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
        self.sense.calc_scalar(freqs, 'pseudo_I', num_steps=5000, little_h=True)
        # basic execution 
        k = np.linspace(0, 3, 10)
        Tsys = 500.0
        t_int = 10.7
        P_N = self.sense.calc_P_N(k, Tsys, t_int, Ncoherent=1, Nincoherent=1, form='Pk')
        nt.assert_equal(P_N.shape, (10,))
        nt.assert_true(np.isclose(P_N, 9.07836740e+11).all())
        # calculate DelSq
        Dsq = self.sense.calc_P_N(k, Tsys, t_int, Ncoherent=1, Nincoherent=1, form='DelSq')
        nt.assert_equal(Dsq.shape, (10,))
        nt.assert_true(Dsq[1] < P_N[1])




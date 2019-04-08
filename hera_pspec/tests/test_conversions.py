import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import conversions


class Test_Cosmo(unittest.TestCase):

    def setUp(self):
        self.C = conversions.Cosmo_Conversions(Om_L=0.68440, Om_b=0.04911, 
                                               Om_c=0.26442, H0=100.0,
                                               Om_M=None, Om_k=None)
    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        # test empty instance
        C = conversions.Cosmo_Conversions()
        # test parameters exist
        C.Om_b
        C.H0
        # test parameters get fed to class
        C = conversions.Cosmo_Conversions(H0=25.5)
        self.assertAlmostEqual(C.H0, 25.5)
        print(C)

        #test equality
        self.assertEqual(C, C)

    def test_units(self):
        si = conversions.units()
        cgs = conversions.cgs_units()
        nt.assert_almost_equal(si.c, 2.99792458e8)
        nt.assert_almost_equal(cgs.c, 2.99792458e10)

    def test_distances(self):
        self.assertAlmostEqual(self.C.f2z(100e6), 13.20405751)
        self.assertAlmostEqual(self.C.f2z(0.1, ghz=True), 13.20405751)
        self.assertAlmostEqual(self.C.z2f(10.0), 129127795.54545455)
        self.assertAlmostEqual(self.C.z2f(10.0, ghz=True), 0.12912779554545455)
        self.assertAlmostEqual(self.C.E(10.0), 20.450997530682947)
        self.assertAlmostEqual(self.C.DC(10.0), 6499.708111027144)
        self.assertAlmostEqual(self.C.DC(10.0, little_h=False), 6499.708111027144)
        self.assertAlmostEqual(self.C.DM(10.0), 6510.2536925709637)
        self.assertAlmostEqual(self.C.DA(10.0), 591.84124477917851)
        self.assertAlmostEqual(self.C.dRperp_dtheta(10.0), 6510.2536925709637)
        self.assertAlmostEqual(self.C.dRpara_df(10.0), 1.2487605057418872e-05)
        self.assertAlmostEqual(self.C.dRpara_df(10.0, ghz=True), 12487.605057418872)
        self.assertAlmostEqual(self.C.X2Y(10.0), 529.26719942209002)


    def test_little_h(self):
        # Test that putting in a really low value of H0 into the conversions object
        # has no effect on the result if little_h=True units are used
        self.C = conversions.Cosmo_Conversions(Om_L=0.68440, Om_b=0.04911, 
                                               Om_c=0.26442, H0=25.12,
                                               Om_M=None, Om_k=None)
        self.assertAlmostEqual(self.C.f2z(100e6), 13.20405751)
        self.assertAlmostEqual(self.C.z2f(10.0), 129127795.54545455)
        self.assertAlmostEqual(self.C.E(10.0), 20.450997530682947)
        self.assertAlmostEqual(self.C.DC(10.0, little_h=True), 6499.708111027144)
        self.assertAlmostEqual(self.C.DC(10.0, little_h=False), 
                               6499.708111027144*100./25.12)
        self.assertAlmostEqual(self.C.DM(10.0, little_h=True), 6510.2536925709637)
        self.assertAlmostEqual(self.C.DA(10.0, little_h=True), 591.84124477917851)
        self.assertAlmostEqual(self.C.dRperp_dtheta(10.0, little_h=True), 
                               6510.2536925709637)
        self.assertAlmostEqual(self.C.dRpara_df(10.0, little_h=True), 
                               1.2487605057418872e-05)
        self.assertAlmostEqual(self.C.dRpara_df(10.0, ghz=True, little_h=True), 
                               12487.605057418872)
        self.assertAlmostEqual(self.C.X2Y(10.0, little_h=True), 
                               529.26719942209002)

    def test_params(self):
        params = self.C.get_params()
        self.assertAlmostEqual(params['Om_L'], self.C.Om_L)

    def test_kpara_kperp(self):
        bl2kperp = self.C.bl_to_kperp(10.0, little_h=True)
        tau2kpara = self.C.tau_to_kpara(10.0, little_h=True)
        self.assertAlmostEqual(bl2kperp, 0.00041570092391078579)
        self.assertAlmostEqual(tau2kpara, 503153.74952115043)

if __name__ == "__main__":
    unittest.main()

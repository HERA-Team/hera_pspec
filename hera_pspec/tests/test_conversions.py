import unittest
import nose.tools as nt
from hera_qm import ant_metrics
import numpy as np
from hera_pspec.data import DATA_PATH
import os
import sys
from hera_pspec import conversions


class Test_SIunits(unittest.TestCase):

    def setUp(self):
        self.siu = conversions.SIunits

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_units(self):
        self.assertAlmostEqual(self.siu.c, 2.99792458e8)


class Test_Cosmo(unittest.TestCase):

    def setUp(self):
        self.C = conversions.Cosmo(Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=67.31,
                                    Om_M=None, Om_k=None)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        C = conversions.Cosmo()
        # test parameters exist
        C.Om_b
        C.H0

        # test parameters get fed to class
        C = conversions.Cosmo(H0=100.0)
        self.assertAlmostEqual(C.H0, 100.0)

    def test_cosmo(self):
        self.assertAlmostEqual(self.C.f2z(100e6), 13.20405751)
        self.assertAlmostEqual(self.C.z2f(10.0), 129127795.54545455)
        self.assertAlmostEqual(self.C.E(10.0), 20.450997530682947)
        self.assertAlmostEqual(self.C.DC(10.0), 9656378117.704863)
        self.assertAlmostEqual(self.C.DM(10.0), 9672045301.6950874)
        self.assertAlmostEqual(self.C.DA(10.0), 879276845.60864437)
        self.assertAlmostEqual(self.C.dr2df(1.0, 10.0), 0.053901448428665018)
        self.assertAlmostEqual(self.C.df2dr(1e6, 10.0), 0.053901448428665018)












if __name__ == "__main__":
    unittest.main()

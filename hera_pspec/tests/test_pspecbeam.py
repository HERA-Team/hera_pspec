import unittest, os
import nose.tools as nt
import numpy as np
import pyuvdata as uv
from hera_pspec import pspecbeam
from hera_pspec.data import DATA_PATH


class Example(unittest.TestCase):
    """
    when running tests in this file by-hand in an interactive interpreter, 
    instantiate this example class as

    self = Example()

    and you can copy-paste self.assert* calls interactively.
    """
    def runTest(self):
        pass


class Test_DataSet(unittest.TestCase):

    def setUp(self):
        beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        bm = pspecbeam.PSpecBeamUV(beamfile)

    def test_UVbeam(self):
        # Precomputed results in the following tests were done "by hand" using 
        # iPython notebook "Scalar_dev2.ipynb" in tests directory
        Om_p = self.bm.power_beam_int()
        Om_pp = self.bm.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        scalar = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, stokes='pseudo_I', num_steps=2000)

        # Check array dimensionality
        self.assertEqual(Om_p.ndim, 1)
        self.assertEqual(Om_pp.ndim, 1)

        # Check that errors are raised for other Stokes parameters
        for stokes in ['Q', 'U', 'V', 'Z']:
            nt.assert_raises(NotImplementedError, self.bm.power_beam_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.bm.power_beam_sq_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.bm.compute_pspec_scalar, 
                             lower_freq, upper_freq, num_freqs, stokes=stokes)

        self.assertAlmostEqual(Om_p[0], 0.078694909518866998)
        self.assertAlmostEqual(Om_p[18], 0.065472512282419112)
        self.assertAlmostEqual(Om_p[-1], 0.029484832405240326)

        self.assertAlmostEqual(Om_pp[0], 0.035171688022986113)
        self.assertAlmostEqual(Om_pp[24], 0.024137903003171767)
        self.assertAlmostEqual(Om_pp[-1], 0.013178952686690554)

        self.assertAlmostEqual(scalar/544745630.76776004, 1.0, delta=1e-4)
        
        # convergence of integral
        scalar_large_Nsteps = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, stokes='pseudo_I', num_steps=10000) 
        self.assertAlmostEqual(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # test taper execution
        scalar = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 1793248694.8873105, 1.0, delta=1e-8)

        # test Jy_to_mK
        M = self.bm.Jy_to_mK(np.linspace(100e6, 200e6, 11))
        nt.assert_equal(len(M), 11)
        nt.assert_almost_equal(M[0], 41.33552971)
        M = self.bm.Jy_to_mK(120e6)

    def test_Gaussbeam(self):
        Om_p = self.gauss.power_beam_int()
        Om_pp = self.gauss.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        scalar = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, stokes='pseudo_I', num_steps=2000)

        # Check array dimensionality
        self.assertEqual(Om_p.ndim,1)
        self.assertEqual(Om_pp.ndim,1)

        # Check that errors are raised for other Stokes parameters
        for stokes in ['Q', 'U', 'V', 'Z']:
            nt.assert_raises(NotImplementedError, 
                             self.gauss.power_beam_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, 
                             self.gauss.power_beam_sq_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, 
                             self.gauss.compute_pspec_scalar, 
                             lower_freq, upper_freq, num_freqs, stokes=stokes)

        self.assertAlmostEqual(Om_p[4], 0.7251776226923511)
        self.assertAlmostEqual(Om_p[4], Om_p[0])

        self.assertAlmostEqual(Om_pp[4], 0.36258881134617554)
        self.assertAlmostEqual(Om_pp[4], Om_pp[0])

        self.assertAlmostEqual(scalar/6082814757.7556648, 1.0, delta=1e-4)
        
        # convergence of integral
        scalar_large_Nsteps = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000)
        self.assertAlmostEqual(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # test taper execution
        scalar = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 19974901797.178055, 1.0, delta=1e-8)


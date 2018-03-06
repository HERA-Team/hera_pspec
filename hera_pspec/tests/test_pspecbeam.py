import unittest, os
import nose.tools as nt
import numpy as np
import pyuvdata as uv
from hera_pspec import pspecbeam


# Get absolute path to data directory
DATADIR = os.path.dirname( os.path.realpath(__file__) ) + "/../data/"


class Test_DataSet(unittest.TestCase):

    def setUp(self):
        beamfile = DATADIR + 'NF_HERA_Beams.beamfits'
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.gauss = pspecbeam.PSpecBeamGauss(0.8,np.linspace(120.*10**6,128.*10**6,50))
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        beamfile = DATADIR + 'NF_HERA_Beams.beamfits'
        bm = pspecbeam.PSpecBeamUV(beamfile)
        pass

    def test_UVbeam(self):
        # Precomputed results in the following tests were done  "by hand" using iPython
        # notebook "Scalar_dev2.ipynb" in tests directory
        Om_p = self.bm.power_beam_int()
        Om_pp = self.bm.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 50
        scalar = self.bm.compute_pspec_scalar(lower_freq,upper_freq,num_freqs,stokes='I')

        # Check array dimensionality
        self.assertEqual(Om_p.ndim,1)
        self.assertEqual(Om_pp.ndim,1)

        # Check that errors are raised for other Stokes parameters
        for stokes in ['Q', 'U', 'V', 'Z']:
            nt.assert_raises(NotImplementedError, self.bm.power_beam_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.bm.power_beam_sq_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.bm.compute_pspec_scalar,lower_freq,upper_freq,num_freqs,stokes=stokes)

        self.assertAlmostEqual(Om_p[0],0.078694909518866998)
        self.assertAlmostEqual(Om_p[18],0.065472512282419112)
        self.assertAlmostEqual(Om_p[-1],0.029484832405240326)

        self.assertAlmostEqual(Om_pp[0],0.035171688022986113)
        self.assertAlmostEqual(Om_pp[24],0.024137903003171767)
        self.assertAlmostEqual(Om_pp[-1],0.013178952686690554)

        self.assertAlmostEqual(scalar/10**5,567834117.162/10**5)

        scalar_double_steps = self.bm.compute_pspec_scalar(lower_freq,upper_freq,num_freqs,num_steps=20000) # convergence of integral
        scalar /= 10**9 # assertAlmostEqual does absolute comparisons, so we want to put things on a sensible scale
        scalar_double_steps /= 10**9
        self.assertAlmostEqual(scalar,scalar_double_steps)

    def test_Gaussbeam(self):
        Om_p = self.gauss.power_beam_int()
        Om_pp = self.gauss.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 50
        scalar = self.gauss.compute_pspec_scalar(lower_freq,upper_freq,num_freqs,stokes='I')

        # Check array dimensionality
        self.assertEqual(Om_p.ndim,1)
        self.assertEqual(Om_pp.ndim,1)

        # Check that errors are raised for other Stokes parameters
        for stokes in ['Q', 'U', 'V', 'Z']:
            nt.assert_raises(NotImplementedError, self.gauss.power_beam_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.gauss.power_beam_sq_int, stokes=stokes)
            nt.assert_raises(NotImplementedError, self.gauss.compute_pspec_scalar,lower_freq,upper_freq,num_freqs,stokes=stokes)

        self.assertAlmostEqual(Om_p[4],0.7251776226923511)
        self.assertAlmostEqual(Om_p[4],Om_p[0])

        self.assertAlmostEqual(Om_pp[4],0.36258881134617554)
        self.assertAlmostEqual(Om_pp[4],Om_pp[0])

        self.assertAlmostEqual(scalar/10**5,6392750121.05/10**5)

        scalar_double_steps = self.gauss.compute_pspec_scalar(lower_freq,upper_freq,num_freqs,num_steps=20000) # convergence of integral
        scalar /= 10**9 # assertAlmostEqual does absolute comparisons, so we want to put things on a sensible scale
        scalar_double_steps /= 10**9
        self.assertAlmostEqual(scalar,scalar_double_steps)



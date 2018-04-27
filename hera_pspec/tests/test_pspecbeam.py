import unittest, os
import nose.tools as nt
import numpy as np
import pyuvdata as uv
from hera_pspec import pspecbeam, conversions
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
        self.beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(self.beamfile)
        self.gauss = pspecbeam.PSpecBeamGauss(0.8, 
                                  np.linspace(115e6, 130e6, 50, endpoint=False))

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
        scalar = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='I', num_steps=2000)
        
        # Check that user-defined cosmology can be specified
        bm = pspecbeam.PSpecBeamUV(self.beamfile,
                                   cosmo=conversions.Cosmo_Conversions())

        # Check array dimensionality
        self.assertEqual(Om_p.ndim, 1)
        self.assertEqual(Om_pp.ndim, 1)

        # Check that errors are raised for other Stokes parameters
        for pol in ['Q', 'U', 'V', 'Z']:
            nt.assert_raises(NotImplementedError, self.bm.power_beam_int, pol=pol)
            nt.assert_raises(NotImplementedError, self.bm.power_beam_sq_int, pol=pol)
            nt.assert_raises(NotImplementedError, self.bm.compute_pspec_scalar, 
                             lower_freq, upper_freq, num_freqs, pol=pol)

        self.assertAlmostEqual(Om_p[0], 0.078694909518866998)
        self.assertAlmostEqual(Om_p[18], 0.065472512282419112)
        self.assertAlmostEqual(Om_p[-1], 0.029484832405240326)

        self.assertAlmostEqual(Om_pp[0], 0.035171688022986113)
        self.assertAlmostEqual(Om_pp[24], 0.024137903003171767)
        self.assertAlmostEqual(Om_pp[-1], 0.013178952686690554)

        self.assertAlmostEqual(scalar/567871703.75268996, 1.0, delta=1e-4)
        
        # convergence of integral
        scalar_large_Nsteps = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='I', num_steps=10000) 
        self.assertAlmostEqual(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # test taper execution
        scalar = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 1986172241.1760113, 1.0, delta=1e-8)

        # test Jy_to_mK
        M = self.bm.Jy_to_mK(np.linspace(100e6, 200e6, 11))
        nt.assert_equal(len(M), 11)
        nt.assert_almost_equal(M[0], 41.360105524572283)
        
        # Extrapolation will fail
        nt.assert_raises(ValueError, self.bm.Jy_to_mK, 99e6)
        nt.assert_raises(ValueError, self.bm.Jy_to_mK, 201e6)
        
        # test exception
        nt.assert_raises(TypeError, self.bm.Jy_to_mK, [1])
        nt.assert_raises(TypeError, self.bm.Jy_to_mK, np.array([1]))

        # test noise scalar
        sclr = self.bm.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='I', num_steps=2000, noise_scalar=True)
        nt.assert_almost_equal(sclr, 70.983962969086235)


    def test_Gaussbeam(self):
        Om_p = self.gauss.power_beam_int()
        Om_pp = self.gauss.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        scalar = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='I', num_steps=2000)
        
        # Check that user-defined cosmology can be specified
        bgauss = pspecbeam.PSpecBeamGauss(0.8, 
                                 np.linspace(115e6, 130e6, 50, endpoint=False), 
                                 cosmo=conversions.Cosmo_Conversions())
        
        # Check array dimensionality
        self.assertEqual(Om_p.ndim,1)
        self.assertEqual(Om_pp.ndim,1)

        self.assertAlmostEqual(Om_p[4], 0.7251776226923511)
        self.assertAlmostEqual(Om_p[4], Om_p[0])

        self.assertAlmostEqual(Om_pp[4], 0.36258881134617554)
        self.assertAlmostEqual(Om_pp[4], Om_pp[0])

        self.assertAlmostEqual(scalar/6392750120.8657961, 1.0, delta=1e-4)
        
        # convergence of integral
        scalar_large_Nsteps = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000)
        self.assertAlmostEqual(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # test taper execution
        scalar = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 22123832163.072491, 1.0, delta=1e-8)
    
    
    def test_BeamFromArray(self):
        """
        Test PSpecBeamFromArray
        """
        # Get Gaussian beam to use as a reference
        Om_P = self.gauss.power_beam_int()
        Om_PP = self.gauss.power_beam_sq_int()
        beam_freqs = self.gauss.beam_freqs
        
        # Array specs for tests
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        
        # Check that PSpecBeamFromArray can be instantiated
        psbeam = pspecbeam.PSpecBeamFromArray(OmegaP=Om_P, OmegaPP=Om_PP, 
                                              beam_freqs=beam_freqs)
        
        psbeampol = pspecbeam.PSpecBeamFromArray(
                                OmegaP={'pseudo_I': Om_P, 'pseudo_Q': Om_P},
                                OmegaPP={'pseudo_I': Om_PP, 'pseudo_Q': Om_PP},
                                beam_freqs=beam_freqs)
        
        # Check that user-defined cosmology can be specified
        bm2 = pspecbeam.PSpecBeamFromArray(OmegaP=Om_P, OmegaPP=Om_PP, 
                                           beam_freqs=beam_freqs,
                                           cosmo=conversions.Cosmo_Conversions())
        
        # Compare scalar calculation with Gaussian case
        scalar = psbeam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, 
                                             stokes='pseudo_I', num_steps=2000)
        g_scalar = self.gauss.compute_pspec_scalar(lower_freq, upper_freq, 
                                                   num_freqs, stokes='pseudo_I', 
                                                   num_steps=2000)
        np.testing.assert_array_almost_equal(scalar, g_scalar)
        
        # Check that polarizations are recognized and invalid ones rejected
        scalarp = psbeampol.compute_pspec_scalar(lower_freq, upper_freq, 
                                                 num_freqs, stokes='pseudo_Q', 
                                                 num_steps=2000)
        
        # Test taper execution (same as Gaussian case)
        scalar = psbeam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, 
                                             num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 22123832163.072491, 1.0, delta=1e-8)
        
        # Check that invalid init args raise errors
        nt.assert_raises(TypeError, pspecbeam.PSpecBeamFromArray, OmegaP=Om_P, 
                         OmegaPP={'pseudo_I': Om_PP}, beam_freqs=beam_freqs)
        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pseudo_I': Om_P, 'pseudo_Q': Om_P},
                         OmegaPP={'pseudo_I': Om_PP,},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pseudo_A': Om_P}, 
                         OmegaPP={'pseudo_A': Om_PP,},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(TypeError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pseudo_I': Om_P,},
                         OmegaPP={'pseudo_I': 'string',},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(ValueError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pseudo_I': Om_P}, 
                         OmegaPP={'pseudo_I': Om_PP[:-2],},
                         beam_freqs=beam_freqs)
        
        # Check that invalid method args raise errors
        nt.assert_raises(KeyError, psbeam.power_beam_int, stokes='blah')
        nt.assert_raises(KeyError, psbeam.power_beam_sq_int, stokes='blah')
        nt.assert_raises(KeyError, psbeam.add_pol, pol='pseudo_A', 
                         OmegaP=Om_P, OmegaPP=Om_PP)
        
        # Check that string works
        self.assert_(len(str(psbeam)) > 0)
    
    
    def test_PSpecBeamBase(self):
        """
        Test that base class can be instantiated.
        """
        bm1 = pspecbeam.PSpecBeamBase()
        
        # Check that user-defined cosmology can be specified
        bm2 = pspecbeam.PSpecBeamBase(cosmo=conversions.Cosmo_Conversions())

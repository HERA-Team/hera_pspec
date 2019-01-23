import unittest, os
import nose.tools as nt
import numpy as np
import pyuvdata as uv
from hera_pspec import pspecbeam, conversions
from hera_pspec.data import DATA_PATH


class Test_DataSet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        bm = pspecbeam.PSpecBeamUV(beamfile)

    def test_UVbeam(self):
        # Precomputed results in the following tests were done "by hand" using 
        # iPython notebook "Scalar_dev2.ipynb" in tests directory
        pstokes_beamfile = os.path.join(DATA_PATH, "HERA_NF_pstokes_power.beamfits")
        beam = pspecbeam.PSpecBeamUV(pstokes_beamfile)
        Om_p = beam.power_beam_int()
        Om_pp = beam.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20

        # check pI polarization
        scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='pI', num_steps=2000)
        
        nt.assert_almost_equal(Om_p[0], 0.080082680885906782)
        nt.assert_almost_equal(Om_p[18], 0.031990943334017245)
        nt.assert_almost_equal(Om_p[-1], 0.03100215028171072)

        nt.assert_almost_equal(Om_pp[0], 0.036391945229980432)
        nt.assert_almost_equal(Om_pp[15], 0.018159280192894631)
        nt.assert_almost_equal(Om_pp[-1], 0.014528100116719534)

        nt.assert_almost_equal(scalar/568847837.72586381, 1.0, delta=1e-4)

        # Check array dimensionality
        nt.assert_equal(Om_p.ndim, 1)
        nt.assert_equal(Om_pp.ndim, 1)
        
        # convergence of integral
        scalar_large_Nsteps = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='pI', num_steps=10000) 
        nt.assert_almost_equal(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # Check that user-defined cosmology can be specified
        beam = pspecbeam.PSpecBeamUV(pstokes_beamfile,
                                     cosmo=conversions.Cosmo_Conversions())

        # Check that errors are not raised for other Stokes parameters
        for pol in ['pQ', 'pU', 'pV',]:
            scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol=pol, num_steps=2000)

        # test taper execution
        scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        nt.assert_almost_equal(scalar / 1989353792.1765163, 1.0, delta=1e-8)

        # test Jy_to_mK
        M = beam.Jy_to_mK(np.linspace(100e6, 200e6, 11))
        nt.assert_equal(len(M), 11)
        nt.assert_almost_equal(M[0], 40.643366654821904)
        M = beam.Jy_to_mK(150e6)
        nt.assert_true(isinstance(M, np.ndarray))
        M = beam.Jy_to_mK(np.linspace(90, 210e6, 11))

        # test exception
        nt.assert_raises(TypeError, beam.Jy_to_mK, [1])
        nt.assert_raises(TypeError, beam.Jy_to_mK, np.array([1]))

        # test noise scalar
        sclr = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='pI', num_steps=2000, noise_scalar=True)
        nt.assert_almost_equal(sclr, 71.105979715733)

        # Check that invalid polarizations raise an error
        pol = 'pZ'
        nt.assert_raises(KeyError, beam.compute_pspec_scalar, 
                         lower_freq, upper_freq, num_freqs, pol=pol)
        pol = 'XX'
        nt.assert_raises(ValueError, beam.compute_pspec_scalar, 
                         lower_freq, upper_freq, num_freqs, pol=pol)

        # check dipole beams work
        dipole_beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
        beam = pspecbeam.PSpecBeamUV(dipole_beamfile)
        scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='XX')
        nt.assert_raises(ValueError, beam.compute_pspec_scalar,
                         lower_freq, upper_freq, num_freqs, pol='pI')

        # check efield beams work
        efield_beamfile = os.path.join(DATA_PATH, "HERA_NF_efield.beamfits")
        beam = pspecbeam.PSpecBeamUV(efield_beamfile)
        scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='XX')
        nt.assert_raises(ValueError, beam.compute_pspec_scalar,
                         lower_freq, upper_freq, num_freqs, pol='pI')

    def test_Gaussbeam(self):
        gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))

        Om_p = gauss.power_beam_int()
        Om_pp = gauss.power_beam_sq_int()
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        scalar = gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol='pI', num_steps=2000)
        
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
        scalar_large_Nsteps = gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000)
        self.assertAlmostEqual(scalar / scalar_large_Nsteps, 1.0, delta=1e-5)

        # test taper execution
        scalar = gauss.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 22123832163.072491, 1.0, delta=1e-8)    
    
    def test_BeamFromArray(self):
        """
        Test PSpecBeamFromArray
        """
        # Get Gaussian beam to use as a reference
        gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))
        Om_P = gauss.power_beam_int()
        Om_PP = gauss.power_beam_sq_int()
        beam_freqs = gauss.beam_freqs
        
        # Array specs for tests
        lower_freq = 120.*10**6
        upper_freq = 128.*10**6
        num_freqs = 20
        
        # Check that PSpecBeamFromArray can be instantiated
        psbeam = pspecbeam.PSpecBeamFromArray(OmegaP=Om_P, OmegaPP=Om_PP, 
                                              beam_freqs=beam_freqs)
        
        psbeampol = pspecbeam.PSpecBeamFromArray(
                                OmegaP={'pI': Om_P, 'pQ': Om_P},
                                OmegaPP={'pI': Om_PP, 'pQ': Om_PP},
                                beam_freqs=beam_freqs)
        
        # Check that user-defined cosmology can be specified
        bm2 = pspecbeam.PSpecBeamFromArray(OmegaP=Om_P, OmegaPP=Om_PP, 
                                           beam_freqs=beam_freqs,
                                           cosmo=conversions.Cosmo_Conversions())
        
        # Compare scalar calculation with Gaussian case
        scalar = psbeam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, 
                                             pol='pI', num_steps=2000)
        g_scalar = gauss.compute_pspec_scalar(lower_freq, upper_freq, 
                                                   num_freqs, pol='pI', 
                                                   num_steps=2000)
        np.testing.assert_array_almost_equal(scalar, g_scalar)
        
        # Check that polarizations are recognized and invalid ones rejected
        scalarp = psbeampol.compute_pspec_scalar(lower_freq, upper_freq, 
                                                 num_freqs, pol='pQ', 
                                                 num_steps=2000)
        
        # Test taper execution (same as Gaussian case)
        scalar = psbeam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, 
                                             num_steps=5000, taper='blackman')
        self.assertAlmostEqual(scalar / 22123832163.072491, 1.0, delta=1e-8)
        
        # Check that invalid init args raise errors
        nt.assert_raises(TypeError, pspecbeam.PSpecBeamFromArray, OmegaP=Om_P, 
                         OmegaPP={'pI': Om_PP}, beam_freqs=beam_freqs)
        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pI': Om_P, 'pQ': Om_P},
                         OmegaPP={'pI': Om_PP,},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'A': Om_P}, 
                         OmegaPP={'A': Om_PP,},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(TypeError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pI': Om_P,},
                         OmegaPP={'pI': 'string',},
                         beam_freqs=beam_freqs)
        
        nt.assert_raises(ValueError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pI': Om_P}, 
                         OmegaPP={'pI': Om_PP[:-2],},
                         beam_freqs=beam_freqs)

        nt.assert_raises(TypeError, pspecbeam.PSpecBeamFromArray,
                         OmegaP=Om_P, 
                         OmegaPP={'pI': Om_PP[:-2],},
                         beam_freqs=beam_freqs)
 
        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'foo': Om_P}, 
                         OmegaPP={'pI': Om_PP,},
                         beam_freqs=beam_freqs)

        nt.assert_raises(KeyError, pspecbeam.PSpecBeamFromArray,
                         OmegaP={'pI': Om_P}, 
                         OmegaPP={'foo': Om_PP,},
                         beam_freqs=beam_freqs)

        nt.assert_raises(KeyError, psbeam.add_pol, 'foo', Om_P, Om_PP)
        nt.assert_raises(KeyError, psbeam.power_beam_int, 'foo')
        nt.assert_raises(KeyError, psbeam.power_beam_sq_int, 'foo')

        # Check that invalid method args raise errors
        nt.assert_raises(KeyError, psbeam.power_beam_int, pol='blah')
        nt.assert_raises(KeyError, psbeam.power_beam_sq_int, pol='blah')
        nt.assert_raises(KeyError, psbeam.add_pol, pol='A', 
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

    def test_get_Omegas(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        beam = pspecbeam.PSpecBeamUV(beamfile)
        OP, OPP = beam.get_Omegas('xx')
        nt.assert_equal(OP.shape, (26, 1))
        nt.assert_equal(OPP.shape, (26, 1))
        OP, OPP = beam.get_Omegas([-5, -6])
        nt.assert_equal(OP.shape, (26, 2))
        nt.assert_equal(OPP.shape, (26, 2))


    def test_beam_normalized_response(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        beam     = pspecbeam.PSpecBeamUV(beamfile)
        freq     = np.linspace(130.0*1e6, 140.0*1e6, 10)
        nside    = beam.primary_beam.nside #uvbeam object
        beam_res = pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol='xx', freq=freq)
        nt.assert_equal(len(beam_res[1]), len(freq)) 
        nt.assert_equal(beam_res[0].ndim, 2) 
        nt.assert_equal(np.shape(beam_res[0]), (len(freq), (12*nside**2)))

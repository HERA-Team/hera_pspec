import unittest
import nose.tools as nt
import numpy as np
import os
from hera_pspec.data import DATA_PATH
from hera_pspec import uvpspec, conversions, parameter, pspecbeam, pspecdata, testing, utils
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec import grouping, container
from pyuvdata import UVData
from hera_cal import redcal
import copy


class Test_grouping(unittest.TestCase):

    def setUp(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        uvp.check()
        self.uvp = uvp

    def tearDown(self):
        pass

    def runTest(self):
        pass
    
    def test_group_baselines(self):
        """
        Test baseline grouping behavior.
        """
        # Generate example lists of baselines
        bls1 = [(0,i) for i in range(1)]
        bls2 = [(0,i) for i in range(2)]
        bls3 = [(0,i) for i in range(4)]
        bls4 = [(0,i) for i in range(5)]
        bls5 = [(0,i) for i in range(13)]
        bls6 = [(0,i) for i in range(521)]
        
        # Check that error is raised when more groups requested than baselines
        nt.assert_raises(ValueError, grouping.group_baselines, bls1, 2)
        nt.assert_raises(ValueError, grouping.group_baselines, bls2, 5)
        nt.assert_raises(ValueError, grouping.group_baselines, bls4, 6)
        
        # Check that keep_remainder=False results in equal-sized blocks
        g1a = grouping.group_baselines(bls4, 2, keep_remainder=False, randomize=False)
        g1b = grouping.group_baselines(bls5, 5, keep_remainder=False, randomize=False)
        g1c = grouping.group_baselines(bls6, 10, keep_remainder=False, randomize=False)
        g2a = grouping.group_baselines(bls4, 2, keep_remainder=False, randomize=True)
        g2b = grouping.group_baselines(bls5, 5, keep_remainder=False, randomize=True)
        g2c = grouping.group_baselines(bls6, 10, keep_remainder=False, randomize=True)
        
        # Loop over groups and check that blocks are equal in size
        gs = [g1a, g1b, g1c, g2a, g2b, g2c]
        for g in gs:
            self.assert_(np.unique([len(grp) for grp in g]).size == 1)
        
        # Check that total no. baselines is preserved with keep_remainder=False
        for bls in [bls1, bls2, bls3, bls4, bls5, bls6]:
            for ngrp in [1, 2, 5, 10, 45]:
                for rand in [True, False]:
                    try:
                        g = grouping.group_baselines(bls, ngrp, 
                                                 keep_remainder=True, 
                                                 randomize=rand)
                    except:
                        continue
                    count = np.sum([len(_g) for _g in g])
                    self.assertEqual(count, len(bls))
        
        # Check that random seed works
        g1 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        g2 = grouping.group_baselines(bls5, 3, randomize=True, seed=11)
        g3 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        for i in range(len(g1)):
            for j in range(len(g1[i])):
                self.assertEqual(g1[i][j], g3[i][j])
    
    def test_sample_baselines(self):
        """
        Test baseline sampling (with replacement) behavior.
        """
        # Generate example lists of baselines
        bls1 = [(0,i) for i in range(1)]
        bls2 = [(0,i) for i in range(2)]
        bls3 = [(0,i) for i in range(4)]
        bls4 = [(0,i) for i in range(5)]
        bls5 = [(0,i) for i in range(13)]
        bls6 = [(0,i) for i in range(521)]
        
        # Example grouped list
        g1 = grouping.group_baselines(bls5, 3, randomize=False)
        
        # Check that returned length is the same as input length
        for bls in [bls1, bls2, bls3, bls4, bls5, bls6]:
            samp = grouping.sample_baselines(bls)
            self.assertEqual(len(bls), len(samp))
        
        # Check that returned length is the same for groups too
        samp = grouping.sample_baselines(g1)
        self.assertEqual(len(g1), len(samp))

    def test_bootstrap_average_blpairs(self):
        """
        Test bootstrap averaging over power spectra.
        """
        # Check that basic bootstrap averaging works
        blpair_groups = [list(np.unique(self.uvp.blpair_array)),]
        uvp1, wgts = grouping.bootstrap_average_blpairs([self.uvp,], 
                                                        blpair_groups, 
                                                        time_avg=False)
        
        uvp2, wgts = grouping.bootstrap_average_blpairs([self.uvp,], 
                                                        blpair_groups, 
                                                        time_avg=True)
        self.assertEqual(uvp1[0].Nblpairs, 1)
        self.assertEqual(uvp1[0].Ntimes, self.uvp.Ntimes)
        self.assertEqual(uvp2[0].Ntimes, 1)
        # Total of weights assigned should equal total no. of blpairs
        self.assertEqual(np.sum(wgts), np.array(blpair_groups).size)
        
        # Check that exceptions are raised when inputs are invalid
        self.assertRaises(AssertionError, grouping.bootstrap_average_blpairs, 
                          [np.arange(5),], blpair_groups, time_avg=False)
        self.assertRaises(KeyError, grouping.bootstrap_average_blpairs, 
                          [self.uvp,], [[200200200200,],], time_avg=False)
        
        # Reduce UVPSpec to only 3 blpairs and set them all to the same values
        _blpairs = list(np.unique(self.uvp.blpair_array)[:3])
        uvp3 = self.uvp.select(spws=0, inplace=False, blpairs=_blpairs)
        Nt = uvp3.Ntimes
        uvp3.data_array[0][Nt:2*Nt] = uvp3.data_array[0][:Nt]
        uvp3.data_array[0][2*Nt:] = uvp3.data_array[0][:Nt]
        uvp3.integration_array[0][Nt:2*Nt] = uvp3.integration_array[0][:Nt]
        uvp3.integration_array[0][2*Nt:] = uvp3.integration_array[0][:Nt]
        
        # Test that different bootstrap-sampled averages have the same value as 
        # the normal average (since the data for all blpairs has been set to 
        # the same values for uvp3)
        np.random.seed(10)
        uvp_avg = uvp3.average_spectra(blpair_groups=[_blpairs,], 
                                       time_avg=True, inplace=False)
        blpair = uvp_avg.blpair_array[0]
        for i in range(5):
            # Generate multiple samples and make sure that they are all equal 
            # to the regular average (for the cloned data in uvp3)
            uvp4, wgts = grouping.bootstrap_average_blpairs(
                                                     [uvp3,], 
                                                     blpair_groups=[_blpairs,], 
                                                     time_avg=True)
            try:
                ps_avg = uvp_avg.get_data((0, blpair, ('xx','xx')))
            except:
                print(uvp_avg.polpair_array)
                raise
            ps_boot = uvp4[0].get_data((0, blpair, ('xx','xx')))
            np.testing.assert_array_almost_equal(ps_avg, ps_boot)

def test_bootstrap_resampled_error():
    # generate a UVPSpec
    visfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    cosmo = conversions.Cosmo_Conversions()
    beam = pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)
    uvd = UVData()
    uvd.read_miriad(visfile)
    ap, a = uvd.get_ENU_antpos(pick_data_ants=True)
    reds = redcal.get_pos_reds(dict(zip(a, ap)), bl_error_tol=1.0)[:3]
    uvp = testing.uvpspec_from_data(uvd, reds, spw_ranges=[(50, 100)], beam=beam, cosmo=cosmo)

    # Lots of this function is already tested by bootstrap_run
    # so only test the stuff not already tested
    if os.path.exists("uvp.h5"):
        os.remove("uvp.h5")
    uvp.write_hdf5("uvp.h5", overwrite=True)
    ua, ub, uw = grouping.bootstrap_resampled_error("uvp.h5", blpair_groups=None, Nsamples=10, seed=0, verbose=False)
    # check number of boots
    nt.assert_equal(len(ub), 10)
    # check seed has been used properly
    nt.assert_equal(uw[0][0][:5], [1.0, 1.0, 0.0, 2.0, 1.0])
    nt.assert_equal(uw[0][1][:5], [2.0, 1.0, 1.0, 6.0, 1.0])
    nt.assert_equal(uw[1][0][:5], [2.0, 2.0, 1.0, 1.0, 2.0])
    nt.assert_equal(uw[1][1][:5], [1.0, 0.0, 1.0, 1.0, 4.0])

    if os.path.exists("uvp.h5"):
        os.remove("uvp.h5")


def test_validate_bootstrap_errorbar():
    """ This is used to test the bootstrapping code
    against the gaussian noise visibility simulator.
    The basic premise is that, if working properly,
    gaussian noise pspectra divided by their bootstrapped
    errorbars should have a standard deviation that
    converges to 1. """
    # get simulated noise in K-str
    uvfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    Tsys = 300.0  # Kelvin

    # generate complex gaussian noise
    seed = 4
    uvd1 = testing.noise_sim(uvfile, Tsys, seed=seed, whiten=True, inplace=False, Nextend=0)
    seed = 5
    uvd2 = testing.noise_sim(uvfile, Tsys, seed=seed, whiten=True, inplace=False, Nextend=0)

    # form (auto) baseline-pairs from only 14.6m bls
    reds, lens, angs = utils.get_reds(uvd1, pick_data_ants=True, bl_len_range=(10, 50),
                                      bl_deg_range=(0, 180))
    bls1, bls2 = utils.flatten(reds), utils.flatten(reds)

    # setup PSpecData and form power psectra
    ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None])
    uvp = ds.pspec(bls1, bls2, (0, 1), [('xx', 'xx')], input_data_weight='identity', norm='I',
                   taper='none', sampling=False, little_h=False, spw_ranges=[(0, 50)], verbose=False)

    # bootstrap resample
    seed = 0
    Nsamples = 200
    uvp_avg, uvp_boots, uvp_wgts = grouping.bootstrap_resampled_error(uvp, time_avg=False, Nsamples=Nsamples,
                                                                      seed=seed, normal_std=True,
                                                                      blpair_groups=[uvp.get_blpairs()])

    # assert z-score has std of ~1.0 along time ax to within 1/sqrt(Nsamples)
    bs_std_zscr_real = np.std(uvp_avg.data_array[0].real) / np.mean(uvp_avg.stats_array['bs_std'][0].real)
    nt.assert_true(np.abs(1.0 - bs_std_zscr_real) < 1/np.sqrt(Nsamples))
    bs_std_zscr_imag = np.std(uvp_avg.data_array[0].imag) / np.mean(uvp_avg.stats_array['bs_std'][0].imag)
    nt.assert_true(np.abs(1.0 - bs_std_zscr_imag) < 1/np.sqrt(Nsamples))


def test_bootstrap_run():
    # generate a UVPSpec and container
    visfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    cosmo = conversions.Cosmo_Conversions()
    beam = pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)
    uvd = UVData()
    uvd.read_miriad(visfile)
    ap, a = uvd.get_ENU_antpos(pick_data_ants=True)
    reds = redcal.get_pos_reds(dict(zip(a, ap)), bl_error_tol=1.0)[:3]
    uvp = testing.uvpspec_from_data(uvd, reds, spw_ranges=[(50, 100)], beam=beam, cosmo=cosmo)
    if os.path.exists("ex.h5"):
        os.remove("ex.h5")
    psc = container.PSpecContainer("ex.h5", mode='rw')
    psc.set_pspec("grp1", "uvp", uvp)

    # Test basic bootstrap run
    grouping.bootstrap_run(psc, time_avg=True, Nsamples=100, seed=0,
                           normal_std=True, robust_std=True, cintervals=[16, 84], keep_samples=True,
                           bl_error_tol=1.0, overwrite=True, add_to_history='hello!', verbose=False)
    spcs = psc.spectra("grp1")
    # assert all bs samples were written
    nt.assert_true(np.all(["uvp_bs{}".format(i) in spcs for i in range(100)]))
    # assert average was written
    nt.assert_true("uvp_avg" in spcs and "uvp" in spcs)
    # assert average only has one time and 3 blpairs
    uvp_avg = psc.get_pspec("grp1", "uvp_avg")
    nt.assert_equal(uvp_avg.Ntimes, 1)
    nt.assert_equal(uvp_avg.Nblpairs, 3)
    # check avg file history
    nt.assert_true("hello!" in uvp_avg.history)
    # assert original uvp is unchanged
    nt.assert_true(uvp == psc.get_pspec("grp1", 'uvp'))
    # check stats array
    np.testing.assert_array_equal([u'bs_cinterval_16.00', u'bs_cinterval_84.00', u'bs_robust_std', u'bs_std'], list(uvp_avg.stats_array.keys()))
    for stat in [u'bs_cinterval_16.00', u'bs_cinterval_84.00', u'bs_robust_std', u'bs_std']:
        nt.assert_equal(uvp_avg.get_stats(stat, (0, ((37, 38), (38, 39)), ('xx','xx'))).shape, (1, 50))
        nt.assert_false(np.any(np.isnan(uvp_avg.get_stats(stat, (0, ((37, 38), (38, 39)), ('xx','xx'))))))
        nt.assert_equal(uvp_avg.get_stats(stat, (0, ((37, 38), (38, 39)), ('xx','xx'))).dtype, np.complex128)

    # test exceptions
    del psc
    if os.path.exists("ex.h5"):
        os.remove("ex.h5")
    psc = container.PSpecContainer("ex.h5", mode='rw')
    # test empty groups
    nt.assert_raises(AssertionError, grouping.bootstrap_run, "ex.h5")
    # test bad filename
    nt.assert_raises(AssertionError, grouping.bootstrap_run, 1)
    # test fed spectra doesn't exist
    psc.set_pspec("grp1", "uvp", uvp)
    nt.assert_raises(AssertionError, grouping.bootstrap_run, psc, spectra=['grp1/foo'])
    if os.path.exists("ex.h5"):
        os.remove("ex.h5")


def test_get_bootstrap_run_argparser():
    args = grouping.get_bootstrap_run_argparser()
    a = args.parse_args(['fname', '--spectra', 'grp1/uvp1', 'grp1/uvp2', 'grp2/uvp1',
                         '--blpair_groups', '101102103104 101102102103, 102103104105',
                         '--time_avg', 'True', '--Nsamples', '100', '--cintervals', '16', '84'])
    nt.assert_equal(a.spectra, ['grp1/uvp1', 'grp1/uvp2', 'grp2/uvp1'])
    nt.assert_equal(a.blpair_groups, [[101102103104, 101102102103], [102103104105]])
    nt.assert_equal(a.cintervals, [16.0, 84.0])


if __name__ == "__main__":
    unittest.main()

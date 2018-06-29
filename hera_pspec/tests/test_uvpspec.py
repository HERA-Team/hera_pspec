import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import uvpspec, conversions, parameter, pspecbeam, pspecdata, testing
from hera_pspec import uvpspec_utils as uvputils
import copy
import h5py
from collections import OrderedDict as odict
from pyuvdata import UVData


class Test_UVPSpec(unittest.TestCase):

    def setUp(self):
        beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        self.uvp = uvp

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_param(self):
        a = parameter.PSpecParam("example", description="example", expected_type=int)

    def test_eq(self):
        # test equivalence
        nt.assert_equal(self.uvp, self.uvp)

    def test_get_funcs(self):
        # get_data
        d = self.uvp.get_data((0, ((1, 2), (1, 2)), 'xx'))
        nt.assert_equal(d.shape, (10, 50))
        nt.assert_true(d.dtype == np.complex)
        nt.assert_almost_equal(d[0,0], (1.0020010020000001+0j))
        d = self.uvp.get_data((0, ((1, 2), (1, 2)), -5))
        nt.assert_almost_equal(d[0,0], (1.0020010020000001+0j))
        d = self.uvp.get_data((0, 1002001002, -5))
        nt.assert_almost_equal(d[0,0], (1.0020010020000001+0j))
        # get_wgts
        w = self.uvp.get_wgts((0, ((1, 2), (1, 2)), 'xx'))
        nt.assert_equal(w.shape, (10, 50, 2))
        nt.assert_true(w.dtype == np.float)
        nt.assert_equal(w[0,0,0], 1.0)
        # get_integrations
        i = self.uvp.get_integrations((0, ((1, 2), (1, 2)), 'xx'))
        nt.assert_equal(i.shape, (10,))
        nt.assert_true(i.dtype == np.float)
        nt.assert_almost_equal(i[0], 1.0)
        # get nsample
        n = self.uvp.get_nsamples((0, ((1, 2), (1, 2)), 'xx'))
        nt.assert_equal(n.shape, (10,))
        nt.assert_true(n.dtype == np.float)
        nt.assert_almost_equal(n[0], 1.0)
        # get dly
        d = self.uvp.get_dlys(0)
        nt.assert_equal(len(d), 50)
        # get blpair seps
        blp = self.uvp.get_blpair_seps()
        nt.assert_equal(len(blp), 30)
        nt.assert_true(np.isclose(blp, 14.60, rtol=1e-1, atol=1e-1).all())
        # get kvecs
        k_perp, k_para = self.uvp.get_kperps(0), self.uvp.get_kparas(0)
        nt.assert_equal(len(k_perp), 30)
        nt.assert_equal(len(k_para), 50)
        # test key expansion
        key = (0, ((1, 2), (1, 2)), 'xx')
        d = self.uvp.get_data(*key)
        nt.assert_equal(d.shape, (10, 50))
        # test key as dictionary
        key = {'spw':0, 'blpair':((1, 2), (1, 2)), 'pol': 'xx'}
        d = self.uvp.get_data(key)
        nt.assert_equal(d.shape, (10, 50))
        # test get_blpairs
        blps = self.uvp.get_blpairs()
        nt.assert_equal(blps, [((1, 2), (1, 2)), ((1, 3), (1, 3)), ((2, 3), (2, 3))])
        # test get all keys
        keys = self.uvp.get_all_keys()
        nt.assert_equal(keys, [(0, ((1, 2), (1, 2)), 'XX'), (0, ((1, 3), (1, 3)), 'XX'),
                               (0, ((2, 3), (2, 3)), 'XX')])

    def test_stats_array(self):
        # test get_data and set_data
        keys = self.uvp.get_all_keys()
        nt.assert_raises(ValueError, self.uvp.set_stats, "errors", keys[0], np.linspace(0, 1, 2))
        nt.assert_raises(AttributeError, self.uvp.get_stats, "__", keys[0])
        errs = np.ones((self.uvp.Ntimes, self.uvp.Ndlys))
        self.uvp.set_stats("errors", keys[0], errs)
        e = self.uvp.get_stats("errors", keys[0])
        nt.assert_true(np.all(self.uvp.get_stats("errors", keys[0]) == errs))
        nt.assert_true(np.all(self.uvp.get_stats("errors", keys[1]) == -99.*errs))

        #self.uvp.set_stats("errors", keys[0], -99.)
        blpairs = self.uvp.get_blpairs()
        u = self.uvp.average_spectra([blpairs], time_avg=False, error_field="errors", inplace=False)
        nt.assert_true(np.all(u.get_stats("errors", keys[0])[0] == np.ones(u.Ndlys)))
        self.uvp.set_stats("who?", keys[0], errs)
        u = self.uvp.average_spectra([blpairs], time_avg=False, error_field=["errors", "who?"], inplace=False)
        nt.assert_true(np.all( u.get_stats("errors", keys[0]) == u.get_stats("who?", keys[0])))

    def test_convert_deltasq(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.convert_to_deltasq(little_h=True)
        k_perp, k_para = self.uvp.get_kperps(0), self.uvp.get_kparas(0)
        k_mag = np.sqrt(k_perp[:, None, None]**2 + k_para[None, :, None]**2)
        nt.assert_true(np.isclose(uvp.data_array[0][0,:,0], (self.uvp.data_array[0]*k_mag**3/(2*np.pi**2))[0,:,0]).all())
        nt.assert_equal(uvp.norm_units, 'k^3 / (2pi^2)')

    def test_blpair_conversions(self):
        # test blpair -> antnums
        an = self.uvp.blpair_to_antnums(1002001002)
        nt.assert_equal(an, ((1, 2), (1, 2)))
        # test antnums -> blpair
        bp = self.uvp.antnums_to_blpair(((1, 2), (1, 2)))
        nt.assert_equal(bp, 1002001002)
        # test bl -> antnums
        an = self.uvp.bl_to_antnums(1002)
        nt.assert_equal(an, (1, 2))
        # test antnums -> bl
        bp = self.uvp.antnums_to_bl((1, 2))
        nt.assert_equal(bp, 1002)

    def test_indices_funcs(self):
        # key to indices
        spw, blpairts, pol = self.uvp.key_to_indices( (0, ((1,2),(1,2)), -5) )
        nt.assert_equal(spw, 0)
        nt.assert_equal(pol, 0)
        nt.assert_true(np.isclose(blpairts, np.array([0,3,6,9,12,15,18,21,24,27])).min())
        spw, blpairts, pol = self.uvp.key_to_indices( (0, 1002001002, 'xx') )
        nt.assert_equal(spw, 0)
        nt.assert_equal(pol, 0)
        nt.assert_true(np.isclose(blpairts, np.array([0,3,6,9,12,15,18,21,24,27])).min())

        # spw to indices
        spw = self.uvp.spw_to_indices(0)
        nt.assert_equal(len(spw), self.uvp.Ndlys)

        # pol to indices
        pol = self.uvp.pol_to_indices('xx')
        nt.assert_equal(len(pol), 1)
        pol = self.uvp.pol_to_indices(-5)
        nt.assert_equal(len(pol), 1)
        pol = self.uvp.pol_to_indices(['xx', 'xx'])
        nt.assert_equal(len(pol), 1)

        # test blpair to indices
        inds = self.uvp.blpair_to_indices(1002001002)
        nt.assert_true(np.isclose(inds, np.array([0,3,6,9,12,15,18,21,24,27])).min())
        inds = self.uvp.blpair_to_indices(((1,2),(1,2)))
        nt.assert_true(np.isclose(inds, np.array([0,3,6,9,12,15,18,21,24,27])).min())

        # test time to indices
        time = self.uvp.time_avg_array[5]
        blpair = 1002001002
        inds = self.uvp.time_to_indices(time=time)
        nt.assert_equal(len(inds), 3)
        nt.assert_true(np.isclose(self.uvp.time_avg_array[inds], time, rtol=1e-10).all())
        inds = self.uvp.time_to_indices(time=time, blpairs=[blpair])
        nt.assert_equal(len(inds), 1)
        nt.assert_equal(self.uvp.blpair_array[inds], blpair)

    def test_select(self):
        # bl group select
        uvp = copy.deepcopy(self.uvp)
        uvp.select(bls=[(1, 2)], inplace=True)
        nt.assert_equal(uvp.Nblpairs, 1)
        nt.assert_equal(uvp.data_array[0].shape, (10, 50, 1))
        nt.assert_almost_equal(uvp.data_array[0][0,0,0], (1.0020010020000001+0j))
        # inplace vs not inplace, spw selection
        uvp = copy.deepcopy(self.uvp)
        uvp2 = uvp.select(spws=0, inplace=False)
        uvp2 = uvp.select(bls=[(1, 2)], inplace=False)
        nt.assert_equal(uvp2.Nblpairs, 1)
        nt.assert_equal(uvp2.data_array[0].shape, (10, 50, 1))
        nt.assert_almost_equal(uvp2.data_array[0][0,0,0], (1.0020010020000001+0j))
        # blpair select
        uvp = copy.deepcopy(self.uvp)
        uvp2 = uvp.select(blpairs=[1002001002, 2003002003], inplace=False)
        nt.assert_equal(uvp2.Nblpairs, 2)
        # pol select
        uvp2 = uvp.select(pols=[-5], inplace=False)
        nt.assert_equal(uvp2.pol_array[0], -5)
        # time select
        uvp2 = uvp.select(times=np.unique(uvp.time_avg_array)[:1], inplace=False)
        nt.assert_equal(uvp2.Ntimes, 1)

    def test_get_ENU_bl_vecs(self):
        bl_vecs = self.uvp.get_ENU_bl_vecs()
        nt.assert_true(np.isclose(bl_vecs[0], np.array([-14.6, 0.0, 0.0]), atol=1e-6).min())

    def test_check(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.check()
        # test failure modes
        del uvp.Ntimes
        nt.assert_raises(AssertionError, uvp.check)
        uvp.Ntimes = self.uvp.Ntimes
        uvp.data_array = uvp.data_array.values()[0]
        nt.assert_raises(AssertionError, uvp.check)
        uvp.data_array = copy.deepcopy(self.uvp.data_array)

    def test_clear(self):
        uvp = copy.deepcopy(self.uvp)
        uvp._clear()
        nt.assert_false(hasattr(uvp, 'Ntimes'))
        nt.assert_false(hasattr(uvp, 'data_array'))

    def test_write_read_hdf5(self):
        # test basic write execution
        uvp = copy.deepcopy(self.uvp)
        if os.path.exists('./ex.hdf5'): os.remove('./ex.hdf5')
        uvp.write_hdf5('./ex.hdf5', overwrite=True)
        nt.assert_true(os.path.exists('./ex.hdf5'))
        # test basic read
        uvp2 = uvpspec.UVPSpec()
        uvp2.read_hdf5('./ex.hdf5')
        nt.assert_true(uvp, uvp2)
        # test just meta
        uvp2 = uvpspec.UVPSpec()
        uvp2.read_hdf5('./ex.hdf5', just_meta=True)
        nt.assert_true(hasattr(uvp2, 'Ntimes'))
        nt.assert_false(hasattr(uvp2, 'data_array'))
        # test exception
        nt.assert_raises(IOError, uvp.write_hdf5, './ex.hdf5', overwrite=False)
        # test partial I/O
        uvp.read_hdf5("./ex.hdf5", bls=[(1, 2)])
        nt.assert_equal(uvp.Nblpairs, 1)
        nt.assert_equal(uvp.data_array[0].shape, (10, 50, 1))
        # test just meta
        uvp.read_hdf5("./ex.hdf5", just_meta=True)
        nt.assert_equal(uvp.Nblpairs, 3)
        nt.assert_false(hasattr(uvp, 'data_array'))
        if os.path.exists('./ex.hdf5'): os.remove('./ex.hdf5')

    def test_sense(self):
        uvp = copy.deepcopy(self.uvp)

        # test generate noise spectra
        P_N = uvp.generate_noise_spectra(0, -5, 500, form='Pk', real=True)
        nt.assert_equal(P_N[1002001002].shape, (10, 50))

        # test smaller system temp
        P_N2 = uvp.generate_noise_spectra(0, -5, 400, form='Pk', real=True)
        nt.assert_true((P_N[1002001002] > P_N2[1002001002]).all())

        # test complex
        P_N2 = uvp.generate_noise_spectra(0, -5, 500, form='Pk', real=False)
        nt.assert_true((P_N[1002001002] < P_N2[1002001002]).all())

        # test Dsq
        Dsq = uvp.generate_noise_spectra(0, -5, 500, form='DelSq', real=True)
        nt.assert_equal(Dsq[1002001002].shape, (10, 50))
        nt.assert_true(Dsq[1002001002][0, 1] < P_N[1002001002][0, 1])

        # test a blpair selection
        P_N = uvp.generate_noise_spectra(0, -5, 500, form='Pk', real=True)

    def test_average_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        # test blpair averaging
        blpairs = uvp.get_blpair_groups_from_bl_groups([[1002, 2003, 1003]], only_pairs_in_bls=False)
        uvp2 = uvp.average_spectra(blpair_groups=blpairs, time_avg=False, inplace=False)
        nt.assert_equal(uvp2.Nblpairs, 1)
        nt.assert_true(np.isclose(uvp2.get_nsamples(0, 1002001002, 'xx'), 3.0).all())
        nt.assert_equal(uvp2.get_data(0, 1002001002, 'xx').shape, (10, 50))
        
        # Test blpair averaging (with baseline-pair weights)
        # Results should be identical with different weights here, as the data 
        # are all the same)
        blpairs = [[1002001002, 1002001002]]
        blpair_wgts = [[2., 0.,]]
        uvp3a = uvp.average_spectra(blpair_groups=blpairs, time_avg=False, 
                                   blpair_weights=None,
                                   inplace=False)
        uvp3b = uvp.average_spectra(blpair_groups=blpairs, time_avg=False, 
                                   blpair_weights=blpair_wgts,
                                   inplace=False)
        #nt.assert_equal(uvp2.Nblpairs, 1)
        nt.assert_true(np.isclose(uvp3a.get_data(0, 1002001002, 'xx'), 
                                  uvp3b.get_data(0, 1002001002, 'xx')).all())
        #nt.assert_equal(uvp2.get_data(0, 1002001002, 'xx').shape, (10, 50))
        
        
        # test time averaging
        uvp2 = uvp.average_spectra(time_avg=True, inplace=False)
        nt.assert_true(uvp2.Ntimes, 1)
        nt.assert_true(np.isclose(uvp2.get_nsamples(0, 1002001002, 'xx'), 10.0).all())
        nt.assert_true(uvp2.get_data(0, 1002001002, 'xx').shape, (1, 50))
        # ensure averaging works when multiple repeated baselines are present, but only
        # if time_avg = True
        uvp.blpair_array[uvp.blpair_to_indices(2003002003)] = 1002001002
        nt.assert_raises(ValueError, uvp.average_spectra, blpair_groups=[list(np.unique(uvp.blpair_array))], time_avg=False, inplace=False)
        uvp.average_spectra(blpair_groups=[list(np.unique(uvp.blpair_array))], time_avg=True)
        nt.assert_equal(uvp.Ntimes, 1)
        nt.assert_equal(uvp.Nblpairs, 1)

    def test_fold_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.fold_spectra()
        nt.assert_true(uvp.folded)
        nt.assert_raises(AssertionError, uvp.fold_spectra)
        nt.assert_equal(len(uvp.get_dlys(0)), 24)
        nt.assert_true(np.isclose(uvp.nsample_array[0], 2.0).all())
    
    def test_str(self):
        a = str(self.uvp)
        nt.assert_true(len(a) > 0)

    def test_compute_scalar(self):
        uvp = copy.deepcopy(self.uvp)
        # test basic execution
        s = uvp.compute_scalar(0, 'xx', num_steps=1000, noise_scalar=False)
        nt.assert_almost_equal(s/552336586.23970914, 1.0, places=5)
        # test execptions
        del uvp.OmegaP
        nt.assert_raises(AssertionError, uvp.compute_scalar, 0, -5)

    def test_set_cosmology(self):
        uvp = copy.deepcopy(self.uvp)
        new_cosmo = conversions.Cosmo_Conversions(Om_L=0.0)
        # test no overwrite
        uvp.set_cosmology(new_cosmo, overwrite=False)
        nt.assert_not_equal(uvp.cosmo, new_cosmo)
        # test setting cosmology
        uvp.set_cosmology(new_cosmo, overwrite=True)
        nt.assert_equal(uvp.cosmo, new_cosmo)
        nt.assert_equal(uvp.norm_units, 'h^-3 Mpc^3')
        nt.assert_true((uvp.scalar_array>1.0).all())
        nt.assert_true((uvp.data_array[0] > 1e5).all())
        # test exception
        new_cosmo2 = conversions.Cosmo_Conversions(Om_L=1.0)
        del uvp.OmegaP
        uvp.set_cosmology(new_cosmo2, overwrite=True)
        nt.assert_not_equal(uvp.cosmo, new_cosmo2)
        # try with new beam
        uvp.set_cosmology(new_cosmo2, overwrite=True, new_beam=self.beam)
        nt.assert_equal(uvp.cosmo, new_cosmo2)
        nt.assert_true(hasattr(uvp, 'OmegaP'))

    def test_combine_uvpspec(self):
        # setup uvp build
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, "NF_HERA_Beams.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30), (60, 90)], beam=beam)

        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        # test success across pol
        uvp2.pol_array[0] = -6
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        nt.assert_equal(out.Npols, 2)
        nt.assert_true(len(set(out.pol_array) ^ set([-5, -6])) == 0)

        # test multiple non-overlapping data axes
        uvp2.freq_array[0] = 0.0
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test partial data overlap failure
        uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (53, 54)], spw_ranges=[(20, 30), (60, 90)], beam=beam)
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30), (60, 105)], beam=beam)
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = copy.deepcopy(uvp1)
        uvp2.pol_array[0] = -6
        uvp2 = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test concat across spw
        uvp2 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(85, 101)], beam=beam)
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        nt.assert_equal(out.Nspws, 3)
        nt.assert_equal(out.Nfreqs, 51)
        nt.assert_equal(out.Nspwdlys, 56)

        # test concat across blpairts
        uvp2 = testing.uvpspec_from_data(uvd, [(53, 54), (67, 68)], spw_ranges=[(20, 30), (60, 90)], beam=beam)
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        nt.assert_equal(out.Nblpairs, 8)
        nt.assert_equal(out.Nbls, 5)

        # test failure due to variable static metadata
        uvp2.weighting = 'foo'
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2.weighting = 'identity'
        del uvp2.OmegaP
        del uvp2.OmegaPP
        nt.assert_raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test feed as strings
        uvp1 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30)], beam=beam)
        uvp2 = copy.deepcopy(uvp1)
        uvp2.pol_array[0] = -6
        uvp1.write_hdf5('uvp1.hdf5', overwrite=True)
        uvp2.write_hdf5('uvp2.hdf5', overwrite=True)
        out = uvpspec.combine_uvpspec(['uvp1.hdf5', 'uvp2.hdf5'], verbose=False)
        nt.assert_true(out.Npols, 2)
        for ff in ['uvp1.hdf5', 'uvp2.hdf5']:
            if os.path.exists(ff):
                os.remove(ff)

        # test UVPSpec __add__
        uvp3 = copy.deepcopy(uvp1)
        uvp3.pol_array[0] = -7
        out = uvp1 + uvp2 + uvp3
        nt.assert_equal(out.Npols, 3)


def test_conj_blpair_int():
    conj_blpair = uvputils._conj_blpair_int(1002003004)
    nt.assert_equal(conj_blpair, 3004001002)

def test_conj_bl_int():
    conj_bl = uvputils._conj_bl_int(1002)
    nt.assert_equal(conj_bl, 2001)

def test_conj_blpair():
    blpair = uvputils._conj_blpair(1002003004, which='first')
    nt.assert_equal(blpair, 2001003004)
    blpair = uvputils._conj_blpair(1002003004, which='second')
    nt.assert_equal(blpair, 1002004003)
    blpair = uvputils._conj_blpair(1002003004, which='both')
    nt.assert_equal(blpair, 2001004003)
    nt.assert_raises(ValueError, uvputils._conj_blpair, 2001003004, which='foo')


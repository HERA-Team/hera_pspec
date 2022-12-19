import unittest
import pytest
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from .. import uvpspec, conversions, parameter, pspecbeam, pspecdata, testing, utils
from .. import uvpspec_utils as uvputils
import copy
import h5py
from collections import OrderedDict as odict
from pyuvdata import UVData
from hera_cal import redcal
import json

class Test_UVPSpec(unittest.TestCase):

    def setUp(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        self.uvp = uvp

        # setup for exact windows related tests
        self.ft_file = os.path.join(DATA_PATH, 'FT_beam_HERA_dipole_test')
        # Instantiate UVWindow()
        # obtain uvp object
        datafile = os.path.join(DATA_PATH, 'zen.2458116.31939.HH.uvh5')
        # read datafile
        uvd = UVData()
        uvd.read_uvh5(datafile)
        # Create a new PSpecData objec
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None])
        # choose baselines
        baselines1, baselines2, blpairs = utils.construct_blpairs(uvd.get_antpairs()[1:],
                                                                  exclude_permutations=False,
                                                                  exclude_auto_bls=True)
        # compute ps
        self.uvp_wf = ds.pspec(baselines1, baselines2, dsets=(0, 1), pols=[('xx','xx')], 
                               spw_ranges=(175,195), taper='bh',verbose=False)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def _add_optionals(self, uvp):
        """add dummy optional cov_array and stats_array to uvp"""
        uvp.cov_array_real = odict()
        uvp.cov_array_imag = odict()
        uvp.cov_model = 'empirical'
        stat = 'noise_err'
        uvp.stats_array = odict({stat: odict()})
        for spw in uvp.spw_array:
            ndlys = uvp.get_spw_ranges(spw)[0][-1]
            uvp.cov_array_real[spw] = np.empty((uvp.Nblpairts, ndlys, ndlys, uvp.Npols), np.float64)
            uvp.cov_array_imag[spw] = np.empty((uvp.Nblpairts, ndlys, ndlys, uvp.Npols), np.float64)
            uvp.stats_array[stat][spw] = np.empty((uvp.Nblpairts, ndlys, uvp.Npols), np.complex128)
        return uvp

    def test_param(self):
        a = parameter.PSpecParam("example", description="example", expected_type=int)

    def test_eq(self):
        # test equivalence
        assert self.uvp == self.uvp

    def test_get_funcs(self):
        # get_data
        d = self.uvp.get_data((0, ((1, 2), (1, 2)), ('xx','xx')))
        assert d.shape == (10, 30)
        assert(d.dtype == complex)
        np.testing.assert_almost_equal(d[0,0], (101.1021011020000001+0j))
        d = self.uvp.get_data((0, ((1, 2), (1, 2)), 1515))
        np.testing.assert_almost_equal(d[0,0], (101.1021011020000001+0j))
        d = self.uvp.get_data((0, 101102101102, 1515))
        np.testing.assert_almost_equal(d[0,0], (101.1021011020000001+0j))

        # get_wgts
        w = self.uvp.get_wgts((0, ((1, 2), (1, 2)), ('xx','xx')))
        assert w.shape == (10, 50, 2) # should have Nfreq dim, not Ndlys
        assert w.dtype == np.float
        assert w[0,0,0] == 1.0

        # get_integrations
        i = self.uvp.get_integrations((0, ((1, 2), (1, 2)), ('xx','xx')))
        assert i.shape == (10,)
        assert i.dtype == np.float
        np.testing.assert_almost_equal(i[0], 1.0)

        # get nsample
        n = self.uvp.get_nsamples((0, ((1, 2), (1, 2)), ('xx', 'xx')))
        assert n.shape == (10,)
        assert n.dtype == np.float
        np.testing.assert_almost_equal(n[0], 1.0)

        # get dly
        d = self.uvp.get_dlys(0)
        assert len(d) == 30

        # get blpair seps
        blp = self.uvp.get_blpair_seps()
        assert len(blp) == 30
        assert(np.isclose(blp, 14.60, rtol=1e-1, atol=1e-1).all())

        # get kvecs
        k_perp, k_para = self.uvp.get_kperps(0), self.uvp.get_kparas(0)
        assert len(k_perp) == 30
        assert len(k_para) == 30

        # test key expansion
        key = (0, ((1, 2), (1, 2)), ('xx','xx'))
        d = self.uvp.get_data(key)
        assert d.shape == (10, 30)

        # test key as dictionary
        key = {'spw':0, 'blpair':((1, 2), (1, 2)), 'polpair': ('xx','xx')}
        d = self.uvp.get_data(key)
        assert d.shape == (10, 30)

        # test get_blpairs
        blps = self.uvp.get_blpairs()
        assert blps == [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]

        # test get_blpair_vecs
        blp_vecs = self.uvp.get_blpair_blvecs()
        assert blp_vecs.shape == (self.uvp.Nblpairs, 3)
        blp_vecs2 = self.uvp.get_blpair_blvecs(use_second_bl=True)
        assert np.isclose(blp_vecs, blp_vecs2).all()

        # test get_polpairs
        polpairs = self.uvp.get_polpairs()
        assert polpairs == [('xx', 'xx')]

        # test get all keys
        keys = self.uvp.get_all_keys()
        assert keys == [(0, ((1, 2), (1, 2)), ('xx', 'xx')),
                               (0, ((2, 3), (2, 3)), ('xx', 'xx')),
                               (0, ((1, 3), (1, 3)), ('xx', 'xx'))]
        # test omit_flags
        self.uvp.integration_array[0][self.uvp.blpair_to_indices(((1, 2), (1, 2)))[:2]] = 0.0
        assert self.uvp.get_integrations((0, ((1, 2), (1, 2)), ('xx','xx')), omit_flags=True).shape == (8,)

    def test_get_covariance(self):
        dfile = os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA')
        uvd = UVData()
        uvd.read(dfile)

        cosmo = conversions.Cosmo_Conversions()
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        uvb = pspecbeam.PSpecBeamUV(beamfile, cosmo=cosmo)

        Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol='XX')
        uvd.data_array *= Jy_to_mK[None, None, :, None]

        uvd1 = uvd.select(times=np.unique(uvd.time_array)[:(uvd.Ntimes//2):1], inplace=False)
        uvd2 = uvd.select(times=np.unique(uvd.time_array)[(uvd.Ntimes//2):(uvd.Ntimes//2 + uvd.Ntimes//2):1], inplace=False)

        ds = pspecdata.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=uvb)
        ds.rephase_to_dset(0)

        spws = utils.spw_range_from_freqs(uvd, freq_range=[(160e6, 165e6), (160e6, 165e6)], bounds_error=True)
        antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
        antpos = dict(zip(ants, antpos))
        red_bls = redcal.get_pos_reds(antpos, bl_error_tol=1.0)
        bls1, bls2, blpairs = utils.construct_blpairs(red_bls[3], exclude_auto_bls=True, exclude_permutations=True)

        uvp = ds.pspec( bls1, bls2, (0, 1), [('xx', 'xx')], spw_ranges=spws, input_data_weight='identity',
         norm='I', taper='blackman-harris', store_cov = True, cov_model='autos', verbose=False)

        key = (0,blpairs[0],"xx")

        cov_real = uvp.get_cov(key, component='real')
        assert cov_real[0].shape == (50, 50)
        cov_imag = uvp.get_cov(key, component='imag')
        assert cov_imag[0].shape == (50, 50)

        uvp.fold_spectra()

        cov_real = uvp.get_cov(key, component='real')
        assert cov_real[0].shape == (24, 24)
        cov_imag = uvp.get_cov(key, component='imag')
        assert cov_imag[0].shape == (24, 24)


    def test_stats_array(self):
        # test get_data and set_data
        uvp = copy.deepcopy(self.uvp)
        keys = uvp.get_all_keys()
        pytest.raises(ValueError, uvp.set_stats, "errors", keys[0], np.linspace(0, 1, 2))
        pytest.raises(AttributeError, uvp.get_stats, "__", keys[0])
        errs = np.ones((uvp.Ntimes, uvp.Ndlys))
        for key in keys:
            uvp.set_stats("errors", key, errs)
        e = uvp.get_stats("errors", keys[0])
        assert(np.all(uvp.get_stats("errors", keys[0]) == errs))

        # self.uvp.set_stats("errors", keys[0], -99.)
        blpairs = uvp.get_blpairs()
        u = uvp.average_spectra([blpairs], time_avg=False, error_weights="errors", inplace=False)
        assert(np.all(np.isclose(u.get_stats("errors", keys[0])[0], np.ones(u.Ndlys)/np.sqrt(len(blpairs)))))
        for key in keys:
            uvp.set_stats("who?", key, errs)
        u = uvp.average_spectra([blpairs], time_avg=False, error_field=["errors", "who?"], inplace=False)
        u2 = uvp.average_spectra([blpairs], time_avg=True, error_field=["errors", "who?"], inplace=False)
        assert(np.all( u.get_stats("errors", keys[0]) == u.get_stats("who?", keys[0])))
        u.select(times=np.unique(u.time_avg_array)[:20])

        u3 = uvp.average_spectra([blpairs], time_avg=True, inplace=False)
        pytest.raises(KeyError, uvp.average_spectra, [blpairs], time_avg=True, inplace=False, error_field=["..............."])
        assert hasattr(u3, "stats_array") == False
        if os.path.exists('./ex.hdf5'): os.remove('./ex.hdf5')
        u.write_hdf5('./ex.hdf5')
        u.read_hdf5('./ex.hdf5')
        os.remove('./ex.hdf5')

        # test folding
        uvp = copy.deepcopy(self.uvp)
        errs = np.repeat(np.arange(1, 31)[None], 10, axis=0)
        uvp.set_stats("test", keys[0], errs)
        uvp.fold_spectra()
        # fold by summing in inverse quadrature
        folded_errs = np.sum([1/errs[:, 1:15][:, ::-1]**2.0, 1/errs[:, 16:]**2.0], axis=0)**(-0.5)
        np.testing.assert_array_almost_equal(uvp.get_stats("test", keys[0]), folded_errs)

        # test set_stats_slice
        uvp = copy.deepcopy(self.uvp)
        key = (0, ((1, 2), (1, 2)), ('xx', 'xx'))
        uvp.set_stats('err', key, np.ones((uvp.Ntimes, uvp.Ndlys)))
        uvp.set_stats_slice('err', 50, 0, above=True, val=10)
        # ensure all dlys above 50 * 15 ns are set to 10 and all others set to 1
        assert np.isclose(uvp.get_stats('err', key)[:, np.abs(uvp.get_dlys(0)*1e9) > 15 * 50], 10).all()
        assert np.isclose(uvp.get_stats('err', key)[:, np.abs(uvp.get_dlys(0)*1e9) < 15 * 50], 1).all()

    def test_convert_deltasq(self):
        # setup uvp build
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                               "HERA_NF_dipole_power.beamfits"))
        uvd_std = copy.deepcopy(uvd)  # dummy uvd_std
        uvd_std.data_array[:] = 1.0
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp = testing.uvpspec_from_data(uvd, bls, data_std=uvd_std,
                                        spw_ranges=[(20, 30), (60, 90)],
                                        beam=beam)
        # dummy stats_array build
        Tsys = utils.uvd_to_Tsys(uvd, beam)
        utils.uvp_noise_error(uvp, Tsys)

        # testing
        dsq = uvp.convert_to_deltasq(inplace=False)
        for spw in uvp.spw_array:
            k_perp, k_para = uvp.get_kperps(spw), uvp.get_kparas(spw)
            k_mag = np.sqrt(k_perp[:, None, None]**2 + k_para[None, :, None]**2)
            coeff = k_mag**3 / (2 * np.pi**2)
            # check data
            assert np.isclose(dsq.data_array[spw][0, :, 0], (uvp.data_array[spw]*coeff)[0, :, 0]).all()
            # check stats
            assert np.isclose(dsq.stats_array['P_N'][spw][0, :, 0],
                              (uvp.stats_array['P_N'][spw] * coeff)[0, :, 0]).all()
            # check cov
            assert np.isclose(dsq.cov_array_real[spw][0, :, :, 0].diagonal(),
                              uvp.cov_array_real[spw][0, :, :, 0].diagonal()*coeff[0, :, 0]**2).all()
        assert dsq.norm_units == uvp.norm_units + ' k^3 / (2pi^2)'

    def test_blpair_conversions(self):
        # test blpair -> antnums
        an = self.uvp.blpair_to_antnums(101102101102)
        assert an == ((1, 2), (1, 2))
        # test antnums -> blpair
        bp = self.uvp.antnums_to_blpair(((1, 2), (1, 2)))
        assert bp == 101102101102
        # test bl -> antnums
        an = self.uvp.bl_to_antnums(101102)
        assert an == (1, 2)
        # test antnums -> bl
        bp = self.uvp.antnums_to_bl((1, 2))
        assert bp == 101102

    def test_indices_funcs(self):
        # key to indices
        spw, blpairts, pol = self.uvp.key_to_indices( (0, ((1,2),(1,2)), 1515) )
        assert spw == 0
        assert pol == 0
        assert(np.isclose(blpairts,
                                  np.array([0,3,6,9,12,15,18,21,24,27])).min())
        spw, blpairts, pol = self.uvp.key_to_indices( (0, 101102101102, ('xx','xx')) )
        assert spw == 0
        assert pol == 0
        assert(np.isclose(blpairts,
                       np.array([0,3,6,9,12,15,18,21,24,27])).min())

        # Check different polpair specification methods give the same results
        s1, b1, p1 = self.uvp.key_to_indices( (0, ((1,2),(1,2)), 1515) )
        s2, b2, p2 = self.uvp.key_to_indices( (0, ((1,2),(1,2)), ('xx','xx')) )
        s3, b3, p3 = self.uvp.key_to_indices( (0, ((1,2),(1,2)), 'xx') )
        assert p1 == p2 == p3

        # spw to indices
        spw1 = self.uvp.spw_to_dly_indices(0)
        assert len(spw1) == self.uvp.Ndlys
        spw2 = self.uvp.spw_to_freq_indices(0)
        assert len(spw2) == self.uvp.Nfreqs
        spw3 = self.uvp.spw_indices(0)
        assert len(spw3) == self.uvp.Nspws

        # use spw tuple instead of spw index
        spw1b = self.uvp.spw_to_dly_indices(self.uvp.get_spw_ranges()[0])
        spw2b = self.uvp.spw_to_freq_indices(self.uvp.get_spw_ranges()[0])
        spw3b = self.uvp.spw_indices(self.uvp.get_spw_ranges()[0])
        np.testing.assert_array_equal(spw1, spw1b)
        np.testing.assert_array_equal(spw2, spw2b)
        np.testing.assert_array_equal(spw3, spw3b)

        # pol to indices
        pol = self.uvp.polpair_to_indices(('xx','xx'))
        assert len(pol) == 1
        pol = self.uvp.polpair_to_indices(1515)
        assert len(pol) == 1
        pol = self.uvp.polpair_to_indices([('xx','xx'), ('xx','xx')])
        assert len(pol) == 1
        pytest.raises(TypeError, self.uvp.polpair_to_indices, 3.14)

        # test blpair to indices
        inds = self.uvp.blpair_to_indices(101102101102)
        assert(np.isclose(inds, np.array([0,3,6,9,12,15,18,21,24,27])).min())
        inds = self.uvp.blpair_to_indices(((1,2),(1,2)))
        assert(np.isclose(inds, np.array([0,3,6,9,12,15,18,21,24,27])).min())
        inds = self.uvp.blpair_to_indices([101102101102, 101102101102])
        inds = self.uvp.blpair_to_indices([((1,2),(1,2)), ((1,2),(1,2))])

        # test time to indices
        time = self.uvp.time_avg_array[5]
        blpair = 101102101102
        inds = self.uvp.time_to_indices(time=time)
        assert len(inds) == 3
        assert(np.isclose(self.uvp.time_avg_array[inds], time, rtol=1e-10).all())
        inds = self.uvp.time_to_indices(time=time, blpairs=[blpair])
        assert len(inds) == 1
        assert self.uvp.blpair_array[inds] == blpair
        inds = self.uvp.time_to_indices(time=time, blpairs=blpair)

    def test_select(self):
        # bl group select
        uvp = copy.deepcopy(self.uvp)
        uvp.select(bls=[(1, 2)], inplace=True)
        assert uvp.Nblpairs == 1
        assert uvp.data_array[0].shape == (10, 30, 1)
        np.testing.assert_almost_equal(uvp.data_array[0][0,0,0], (101.1021011020000001+0j))

        # inplace vs not inplace, spw selection
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        rp = {'filter_centers':[0.],
              'filter_half_widths':[250e-9],
              'filter_factors':[1e-9]}
        r_params = {}
        for bl in bls:
            key1 =  bl + ('xx',)
            r_params[key1] = rp

        uvp1 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30), (60, 90)], beam=beam,
                                         r_params = r_params)
        uvp2 = uvp1.select(spws=0, inplace=False)
        assert uvp2.Nspws == 1
        uvp2 = uvp2.select(bls=[(37, 38), (38, 39)], inplace=False)
        assert uvp2.Nblpairs == 1
        assert uvp2.data_array[0].shape == (10, 10, 1)
        np.testing.assert_almost_equal(uvp2.data_array[0][0,0,0], (-3831605.3903496987+8103523.9604128916j))
        assert len(uvp2.get_r_params().keys()) == 2
        for rpkey in uvp2.get_r_params():
            assert(rpkey == (37, 38, 'xx') or rpkey == (38, 39, 'xx'))

        # blpair select
        uvp = copy.deepcopy(self.uvp)
        uvp2 = uvp.select(blpairs=[101102101102, 102103102103], inplace=False)
        assert uvp2.Nblpairs == 2

        # pol select
        uvp2 = uvp.select(polpairs=[1515,], inplace=False)
        assert uvp2.polpair_array[0] == 1515

        # time select
        uvp2 = uvp.select(times=np.unique(uvp.time_avg_array)[:1], inplace=False)
        assert uvp2.Ntimes == 1

        # test pol and blpair select, and check dimensionality of output
        uvp = copy.deepcopy(self.uvp)
        uvp.set_stats('hi', uvp.get_all_keys()[0], np.ones(300).reshape(10, 30))
        uvp2 = uvp.select(blpairs=uvp.get_blpairs(), polpairs=uvp.polpair_array,
                          inplace=False)
        assert uvp2.data_array[0].shape == (30, 30, 1)
        assert uvp2.stats_array['hi'][0].shape == (30, 30, 1)

        # test when both blp and pol array are non-sliceable
        uvp2, uvp3, uvp4 = copy.deepcopy(uvp), copy.deepcopy(uvp), copy.deepcopy(uvp)
        uvp2.polpair_array[0] = 1414
        uvp3.polpair_array[0] = 1313
        uvp4.polpair_array[0] = 1212
        uvp = uvp + uvp2 + uvp3 + uvp4
        uvp5 = uvp.select(blpairs=[101102101102], polpairs=[1515, 1414, 1313],
                          inplace=False)
        assert uvp5.data_array[0].shape == (10, 30, 3)

        # select only on lst
        uvp = copy.deepcopy(self.uvp)
        uvp2 = uvp.select(lsts=np.unique(uvp.lst_avg_array), inplace=False)
        assert uvp == uvp2

        # check non-sliceable select: both pol and blpairs are non-sliceable
        uvp = copy.deepcopy(self.uvp)
        # extend polpair_array axis
        for i in [1414, 1313, 1212]:
            _uvp = copy.deepcopy(self.uvp)
            _uvp.polpair_array[0] = i
            uvp += _uvp

        # create a purposely non-sliceable select across *both* pol and blpair
        uvp.select(polpairs=[1414, 1313, 1212], blpairs=[101102101102, 102103102103])
        assert uvp.Npols == 3
        assert uvp.Nblpairs == 2

    def test_get_ENU_bl_vecs(self):
        bl_vecs = self.uvp.get_ENU_bl_vecs()
        assert(np.isclose(bl_vecs[0], np.array([-14.6, 0.0, 0.0]), atol=1e-6).min())

    def test_check(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.check()
        # test failure modes
        del uvp.Ntimes
        pytest.raises(AssertionError, uvp.check)
        uvp.Ntimes = self.uvp.Ntimes
        uvp.data_array = list(uvp.data_array.values())[0]
        pytest.raises(AssertionError, uvp.check)
        uvp.data_array = copy.deepcopy(self.uvp.data_array)

    def test_clear(self):
        uvp = copy.deepcopy(self.uvp)
        uvp._clear()
        assert hasattr(uvp, 'Ntimes') == False
        assert hasattr(uvp, 'data_array') == False

    def test_get_r_params(self):

        # inplace vs not inplace, spw selection
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        rp = {'filter_centers':[0.],
              'filter_half_widths':[250e-9],
              'filter_factors':[1e-9]}
        r_params = {}
        for bl in bls:
            key1 =  bl + ('xx',)
            r_params[key1] = rp
        uvp = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30), (60, 90)], beam=beam,
                                         r_params = r_params)
        assert r_params == uvp.get_r_params()

    def test_write_read_hdf5(self):

        # test basic write execution
        uvp = copy.deepcopy(self.uvp)
        if os.path.exists('./ex.hdf5'): os.remove('./ex.hdf5')
        uvp.write_hdf5('./ex.hdf5', overwrite=True)
        assert os.path.exists('./ex.hdf5')

        # test basic read
        uvp2 = uvpspec.UVPSpec()
        uvp2.read_hdf5('./ex.hdf5')
        assert uvp == uvp2

        # test just meta
        uvp2 = uvpspec.UVPSpec()
        uvp2.read_hdf5('./ex.hdf5', just_meta=True)
        assert hasattr(uvp2, 'Ntimes')
        assert hasattr(uvp2, 'data_array') == False

        # test exception
        pytest.raises(IOError, uvp.write_hdf5, './ex.hdf5', overwrite=False)

        # test partial I/O
        uvp.read_hdf5("./ex.hdf5", bls=[(1, 2)])
        assert uvp.Nblpairs == 1
        assert uvp.data_array[0].shape == (10, 30, 1)

        # test just meta
        uvp.read_hdf5("./ex.hdf5", just_meta=True)
        assert uvp.Nblpairs == 3
        assert hasattr(uvp, 'data_array') == False
        if os.path.exists('./ex.hdf5'): os.remove('./ex.hdf5')

        # tests with exact windows
        # test basic write execution
        uvp = copy.deepcopy(self.uvp_wf)
        uvp.get_exact_window_functions(ftbeam_file = self.ft_file,
                                       inplace=True)
        if os.path.exists('./ex.hdf5'):
            os.remove('./ex.hdf5')
        uvp.write_hdf5('./ex.hdf5', overwrite=True)
        assert os.path.exists('./ex.hdf5')
        # test basic read
        uvp2 = uvpspec.UVPSpec()
        uvp2.read_hdf5('./ex.hdf5')
        assert uvp == uvp2

    def test_sense(self):
        uvp = copy.deepcopy(self.uvp)

        # test generate noise spectra
        polpair = ('xx', 'xx')
        P_N = uvp.generate_noise_spectra(0, polpair, 500, form='Pk', component='real')
        assert P_N[101102101102].shape == (10, 30)

        # test smaller system temp
        P_N2 = uvp.generate_noise_spectra(0, polpair, 400, form='Pk', component='real')
        assert((P_N[101102101102] > P_N2[101102101102]).all())

        # test complex
        P_N2 = uvp.generate_noise_spectra(0, polpair, 500, form='Pk', component='abs')
        assert((P_N[101102101102] < P_N2[101102101102]).all())

        # test Dsq
        Dsq = uvp.generate_noise_spectra(0, polpair, 500, form='DelSq', component='real')
        assert Dsq[101102101102].shape == (10, 30)
        assert(Dsq[101102101102][0, 1] < P_N[101102101102][0, 1])

        # test a blpair selection and int polpair
        blpairs = uvp.get_blpairs()[:1]
        P_N = uvp.generate_noise_spectra(0, 1515, 500, form='Pk', blpairs=blpairs, component='real')
        assert P_N[101102101102].shape == (10, 30)

        # test as a dictionary of arrays
        Tsys = dict([(uvp.antnums_to_blpair(k), 500 * np.ones((uvp.Ntimes, uvp.Ndlys)) * np.linspace(1, 2, uvp.Ntimes)[:, None]) for k in uvp.get_blpairs()])
        P_N = uvp.generate_noise_spectra(0, 1515, Tsys, form='Pk', blpairs=blpairs, component='real')
        # assert time gradient is captured: 2 * Tsys results in 4 * P_N
        assert(np.isclose(P_N[101102101102][0, 0] * 4, P_N[101102101102][-1, 0]))

    def test_average_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        # test blpair averaging
        blpairs = uvp.get_blpair_groups_from_bl_groups([[101102, 102103, 101103]],
                                                       only_pairs_in_bls=False)
        uvp2 = uvp.average_spectra(blpair_groups=blpairs, time_avg=False,
                                   inplace=False)
        assert uvp2.Nblpairs == 1
        assert(np.isclose(uvp2.get_nsamples((0, 101102101102, ('xx','xx'))), 3.0).all())
        assert uvp2.get_data((0, 101102101102, ('xx','xx'))).shape == (10, 30)

        # Test blpair averaging (with baseline-pair weights)
        # Results should be identical with different weights here, as the data
        # are all the same)
        blpairs = [[101102101102, 101102101102]]
        blpair_wgts = [[2., 0.,]]
        uvp3a = uvp.average_spectra(blpair_groups=blpairs, time_avg=False,
                                   blpair_weights=None,
                                   inplace=False)
        uvp3b = uvp.average_spectra(blpair_groups=blpairs, time_avg=False,
                                   blpair_weights=blpair_wgts,
                                   inplace=False)
        #assert uvp2.Nblpairs == 1
        assert(np.isclose(
                        uvp3a.get_data((0, 101102101102, ('xx','xx'))),
                        uvp3b.get_data((0, 101102101102, ('xx','xx')))).all())
        #assert uvp2.get_data((0, 101102101102, 'xx')).shape == (10, 30)

        # test time averaging
        uvp2 = uvp.average_spectra(time_avg=True, inplace=False)
        assert uvp2.Ntimes == 1
        assert(np.isclose(
                uvp2.get_nsamples((0, 101102101102, ('xx','xx'))), 10.0).all())
        assert uvp2.get_data((0, 101102101102, ('xx','xx'))).shape == (1, 30)
        # ensure averaging works when multiple repeated baselines are present, but only
        # if time_avg = True
        uvp.blpair_array[uvp.blpair_to_indices(102103102103)] = 101102101102
        pytest.raises(ValueError, uvp.average_spectra, blpair_groups=[list(np.unique(uvp.blpair_array))], time_avg=False, inplace=False)
        uvp.average_spectra(blpair_groups=[list(np.unique(uvp.blpair_array))], time_avg=True)
        assert uvp.Ntimes == 1
        assert uvp.Nblpairs == 1

    def test_get_exact_window_functions(self):

        uvp = copy.deepcopy(self.uvp_wf)

        # obtain exact_windows (fiducial usage)
        uvp.get_exact_window_functions(ftbeam_file = self.ft_file,
                                       inplace=True)
        assert uvp.exact_windows
        assert uvp.window_function_array[0].shape[0] == uvp.Nblpairts
        # if not exact window function, array dim is 4
        assert uvp.window_function_array[0].ndim == 5

        ## tests

        # obtain exact window functions for one spw only
        uvp.get_exact_window_functions(ftbeam_file = self.ft_file,
                                       spw_array=0, inplace=True, verbose=True)
        # raise error if spw not in UVPSpec object
        pytest.raises(AssertionError, uvp.get_exact_window_functions,
                      ftbeam_file = self.ft_file,
                      spw_array=2, inplace=True)

        # output exact_window functions but does not make them attributes
        kperp_bins, kpara_bins, wf_array = uvp.get_exact_window_functions(ftbeam_file = self.ft_file,
                                                                          inplace=False)
        # check if result is the same with and without inplace
        assert np.all(wf_array[0]==uvp.window_function_array[0])

    def test_fold_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.fold_spectra()
        assert(uvp.folded)
        pytest.raises(AssertionError, uvp.fold_spectra)
        assert len(uvp.get_dlys(0)) == 14
        assert(np.isclose(uvp.nsample_array[0], 2.0).all())

        # also run the odd case
        uvd = UVData()
        uvd_std = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        uvd_std.read_miriad(os.path.join(DATA_PATH,'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                               "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(uvd, bls, data_std=uvd_std,
                                         spw_ranges=[(0,17)], beam=beam)
        uvp1.fold_spectra()
        cov_folded = uvp1.get_cov((0, ((37, 38), (38, 39)), ('xx','xx')))
        data_folded = uvp1.get_data((0, ((37,38), (38, 39)), ('xx','xx')))

        # Test fold_spectra method is consistent with average_spectra()
        uvp = testing.uvpspec_from_data(uvd, bls, data_std=uvd_std,
                                         spw_ranges=[(0,17)], beam=beam)
        # Average then fold
        uvp_avg = uvp.average_spectra(time_avg=True, inplace=False)

        # Fold averaged spectra
        uvp_avg_folded = copy.deepcopy(uvp_avg)
        uvp_avg_folded.fold_spectra()

        # Fold then average
        uvp_folded = copy.deepcopy(uvp)
        uvp_folded.fold_spectra()

        # Average folded spectra
        uvp_folded_avg = uvp_folded.average_spectra(time_avg=True, inplace=False)
        assert(np.allclose(uvp_avg_folded.get_data((0, ((37, 38), (38, 39)), 'xx')), uvp_folded_avg.get_data((0, ((37, 38), (38, 39)), 'xx')), rtol=1e-5))

        uvp = copy.deepcopy(self.uvp_wf)
        # obtain exact_windows (fiducial usage)
        uvp.get_exact_window_functions(ftbeam_file = self.ft_file,
                                       inplace=True)
        uvp.fold_spectra()

    def test_str(self):
        a = str(self.uvp)
        assert(len(a) > 0)

    def test_compute_scalar(self):
        uvp = copy.deepcopy(self.uvp)
        # test basic execution
        s = uvp.compute_scalar(0, ('xx','xx'), num_steps=1000, noise_scalar=False)
        np.testing.assert_almost_equal(s/553995277.90425551, 1.0, decimal=5)
        # test execptions
        del uvp.OmegaP
        pytest.raises(AssertionError, uvp.compute_scalar, 0, -5)

    def test_set_cosmology(self):
        uvp = copy.deepcopy(self.uvp)
        new_cosmo = conversions.Cosmo_Conversions(Om_L=0.0)

        # test no overwrite
        uvp.set_cosmology(new_cosmo, overwrite=False)
        assert uvp.cosmo != new_cosmo

        # test setting cosmology
        uvp.set_cosmology(new_cosmo, overwrite=True)
        assert uvp.cosmo == new_cosmo
        assert uvp.norm_units == 'h^-3 Mpc^3'
        assert (uvp.scalar_array>1.0).all()
        assert (uvp.data_array[0] > 1e5).all()

        # test exception
        new_cosmo2 = conversions.Cosmo_Conversions(Om_L=1.0)
        del uvp.OmegaP
        uvp.set_cosmology(new_cosmo2, overwrite=True)
        assert uvp.cosmo != new_cosmo2

        # try with new beam
        uvp.set_cosmology(new_cosmo2, overwrite=True, new_beam=self.beam)
        assert uvp.cosmo == new_cosmo2
        assert hasattr(uvp, 'OmegaP')

    def test_combine_uvpspec(self):
        # setup uvp build
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                               "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(uvd, bls,
                                         spw_ranges=[(20, 30), (60, 90)],
                                         beam=beam)
        uvp1 = self._add_optionals(uvp1)

        # test concat across pol
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Npols == 2
        assert(len(set(out.polpair_array) ^ set([1515, 1414])) == 0)
        key = (0, ((37, 38), (38, 39)), ('xx','xx'))
        assert(np.all(np.isclose(out.get_nsamples(key),
                       np.ones(10, dtype=np.float64))))
        assert(np.all(np.isclose(out.get_integrations(key),
                       190 * np.ones(10, dtype=np.float64), atol=5, rtol=2)))
        # optionals
        for spw in out.spw_array:
            ndlys = out.get_spw_ranges(spw)[0][-1]
            assert out.cov_array_real[spw].shape == (30, ndlys, ndlys, 2)
            assert out.stats_array['noise_err'][spw].shape == (30, ndlys, 2)
            assert out.window_function_array[spw].shape == (30, ndlys, ndlys, 2)
            assert out.cov_model == 'empirical'

        # test concat across spw
        uvp2 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(85, 101)],
                                         beam=beam)
        uvp2 = self._add_optionals(uvp2)

        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Nspws == 3
        assert out.Nfreqs == 51
        assert out.Nspwdlys == 56

        # optionals
        assert len(out.stats_array['noise_err']) == 3
        assert len(out.window_function_array) == 3
        assert len(out.cov_array_real) == 3

        # test concat across blpairts
        uvp2 = testing.uvpspec_from_data(uvd, [(53, 54), (67, 68)],
                                         spw_ranges=[(20, 30), (60, 90)],
                                         beam=beam)
        uvp2 = self._add_optionals(uvp2)
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Nblpairs == 4
        assert out.Nbls == 5

        # optionals
        for spw in out.spw_array:
            ndlys = out.get_spw_ranges(spw)[0][-1]
            assert out.cov_array_real[spw].shape == (40, ndlys, ndlys, 1)
            assert out.stats_array['noise_err'][spw].shape == (40, ndlys, 1)
            assert out.window_function_array[spw].shape == (40, ndlys, ndlys, 1)

        # test feed as strings
        uvp1 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30)], beam=beam)
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp1.write_hdf5('uvp1.hdf5', overwrite=True)
        uvp2.write_hdf5('uvp2.hdf5', overwrite=True)
        out = uvpspec.combine_uvpspec(['uvp1.hdf5', 'uvp2.hdf5'], verbose=False)
        assert out.Npols == 2
        for ff in ['uvp1.hdf5', 'uvp2.hdf5']:
            if os.path.exists(ff):
                os.remove(ff)

        # test UVPSpec __add__
        uvp2 = copy.deepcopy(uvp1)
        uvp3 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp3.polpair_array[0] = 1313
        out = uvp1 + uvp2 + uvp3
        assert out.Npols == 3

        # Test whether n_dlys != Nfreqs works
        uvp4 = testing.uvpspec_from_data(uvd, bls, beam=beam,
                                         spw_ranges=[(20, 30), (60, 90)],
                                         n_dlys=[5, 15])
        uvp4b = copy.deepcopy(uvp4)
        uvp4b.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp4, uvp4b], verbose=False)

        # test history adding
        uvp_a = copy.deepcopy(uvp1)
        uvp_b = copy.deepcopy(uvp1)
        uvp_b.polpair_array[0] = 1414
        uvp_a.history = 'batwing'
        uvp_b.history = 'foobar'

        # w/ merge
        out = uvpspec.combine_uvpspec([uvp_a, uvp_b], merge_history=True, verbose=False)
        assert 'batwing' in out.history and 'foobar' in out.history

        # w/o merge
        out = uvpspec.combine_uvpspec([uvp_a, uvp_b], merge_history=False, verbose=False)
        assert 'batwing' in out.history and not 'foobar' in out.history

        # test no cov_array if cov_model is not consistent
        uvp_a = copy.deepcopy(uvp1)
        uvp_b = copy.deepcopy(uvp1)
        uvp_b.cov_model = 'foo'
        uvp_b.polpair_array = np.array([1414])
        out = uvpspec.combine_uvpspec([uvp_a, uvp_b], verbose=False)
        assert hasattr(out, 'cov_array_real') is False

        # for exact windows
        # test basic write execution
        uvp1 = copy.deepcopy(self.uvp_wf)
        uvp1.get_exact_window_functions(ftbeam_file = self.ft_file, inplace=True)
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)

    def test_combine_uvpspec_errors(self):
        # setup uvp build
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                               "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(uvd, bls,
                                         spw_ranges=[(20, 30), (60, 90)],
                                         beam=beam)

        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test multiple non-overlapping data axes
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp2.freq_array[0] = 0.0
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test partial data overlap failure
        uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (53, 54)],
                                         spw_ranges=[(20, 30), (60, 90)],
                                         beam=beam)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = testing.uvpspec_from_data(uvd, bls,
                                         spw_ranges=[(20, 30), (60, 105)],
                                         beam=beam)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp2 = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test failure due to variable static metadata
        uvp2.weighting = 'foo'
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2.weighting = 'identity'
        del uvp2.OmegaP
        del uvp2.OmegaPP
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])


    def test_combine_uvpspec_r_params(self):
        # setup uvp build
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH,
                                               "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]

        rp = {'filter_centers':[0.],
              'filter_half_widths':[250e-9],
              'filter_factors':[1e-9]}

        r_params = {}

        for bl in bls:
            key1 =  bl + ('xx',)
            r_params[key1] = rp

        # create an r_params copy with inconsistent weighting to test
        # error case
        r_params_inconsistent = copy.deepcopy(r_params)
        r_params[key1]['filter_half_widths'] = [100e-9]

        uvp1 = testing.uvpspec_from_data(uvd, bls,
                                         spw_ranges=[(20, 30), (60, 90)],
                                         beam=beam, r_params=r_params)

        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test success across pol
        uvp2.polpair_array[0] = 1414

        # test errors when combining with pspecs without r_params
        uvp3 = copy.deepcopy(uvp2)
        uvp3.r_params = ''
        pytest.raises(ValueError, uvpspec.combine_uvpspec, [uvp1, uvp3])

        # combining multiple uvp objects without r_params should run fine
        uvp4 = copy.deepcopy(uvp1)
        uvp4.r_params = ''
        uvpspec.combine_uvpspec([uvp3, uvp4])

        # now test error case with inconsistent weightings.
        uvp5 = copy.deepcopy(uvp2)
        uvp5.r_params = uvputils.compress_r_params(r_params_inconsistent)
        pytest.raises(ValueError, uvpspec.combine_uvpspec, [uvp1, uvp5])

    def test_combine_uvpspec_std(self):
        # setup uvp build
        uvd = UVData()
        uvd_std = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, 'zen.even.xx.LST.1.28828.uvOCRSA'))
        uvd_std.read_miriad(
                      os.path.join(DATA_PATH,'zen.even.xx.LST.1.28828.uvOCRSA'))
        beam = pspecbeam.PSpecBeamUV(
                      os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(uvd, bls, data_std=uvd_std,
                                         spw_ranges=[(20, 24), (64, 68)],
                                         beam=beam)
        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test success across pol
        uvp2.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Npols == 2
        assert len(set(out.polpair_array) ^ set([1515, 1414])) == 0

        # test multiple non-overlapping data axes
        uvp2.freq_array[0] = 0.0
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test partial data overlap failure
        uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (53, 54)],
                                         data_std=uvd_std,
                                         spw_ranges=[(20, 24), (64, 68)],
                                         beam=beam)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = testing.uvpspec_from_data(uvd, bls,
                                         spw_ranges=[(20, 24), (64, 68)],
                                         data_std=uvd_std, beam=beam)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp2 = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test concat across spw
        uvp2 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(85, 91)],
                                         data_std=uvd_std, beam=beam)
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Nspws == 3
        assert out.Nfreqs == 14
        assert out.Nspwdlys == 14

        # test concat across blpairts
        uvp2 = testing.uvpspec_from_data(uvd, [(53, 54), (67, 68)],
                                         spw_ranges=[(20, 24), (64, 68)],
                                         data_std=uvd_std, beam=beam)
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Nblpairs == 4
        assert out.Nbls == 5

        # test failure due to variable static metadata
        uvp2.weighting = 'foo'
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])
        uvp2.weighting = 'identity'
        del uvp2.OmegaP
        del uvp2.OmegaPP
        pytest.raises(AssertionError, uvpspec.combine_uvpspec, [uvp1, uvp2])

        # test feed as strings
        uvp1 = testing.uvpspec_from_data(uvd, bls, spw_ranges=[(20, 30)],
                                         data_std=uvd_std, beam=beam)
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp1.write_hdf5('uvp1.hdf5', overwrite=True)
        uvp2.write_hdf5('uvp2.hdf5', overwrite=True)
        out = uvpspec.combine_uvpspec(['uvp1.hdf5', 'uvp2.hdf5'], verbose=False)
        assert out.Npols == 2
        for ff in ['uvp1.hdf5', 'uvp2.hdf5']:
            if os.path.exists(ff): os.remove(ff)

        # test UVPSpec __add__
        uvp3 = copy.deepcopy(uvp1)
        uvp3.polpair_array[0] = 1313
        out = uvp1 + uvp2 + uvp3
        assert out.Npols == 3

def test_conj_blpair_int():
    conj_blpair = uvputils._conj_blpair_int(101102103104)
    assert conj_blpair == 103104101102

def test_conj_bl_int():
    conj_bl = uvputils._conj_bl_int(101102)
    assert conj_bl == 102101

def test_conj_blpair():
    blpair = uvputils._conj_blpair(101102103104, which='first')
    assert blpair == 102101103104
    blpair = uvputils._conj_blpair(101102103104, which='second')
    assert blpair == 101102104103
    blpair = uvputils._conj_blpair(101102103104, which='both')
    assert blpair == 102101104103
    pytest.raises(ValueError, uvputils._conj_blpair, 102101103104, which='foo')

def test_backwards_compatibility_read():
    """This is a backwards compatibility test.
    If it fails, your edits must be changed to make this test pass.
    If the hera_pspec team decides to move forward and break
    compatibility, this file can be overwritten
    and the date of the file changed in the comment below.
    """
    # test read in of a static test file dated 8/2019
    uvp = uvpspec.UVPSpec()
    uvp.read_hdf5(os.path.join(DATA_PATH, 'test_uvp.h5'))
    # assert check does not fail
    uvp.check()

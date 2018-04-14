import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import uvpspec, conversions, parameter, pspecbeam
import copy
import h5py
from collections import OrderedDict as odict


class Test_UVPSpec(unittest.TestCase):

    def setUp(self):
        uvp = uvpspec.UVPSpec()

        Ntimes = 10
        Nfreqs = 50
        Ndlys = Nfreqs
        Nspws = 1
        Nspwdlys = Nspws * Nfreqs

        # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
        blpairs = [1002001002, 2003002003, 1003001003]
        bls = [1002, 2003, 1003]
        Nbls = len(bls)
        Nblpairs = len(blpairs)
        Nblpairts = Nblpairs * Ntimes

        blpair_array = np.tile(blpairs, Ntimes)
        bl_array = np.array(bls)
        bl_vecs = np.array([[  5.33391548e+00,  -1.35907816e+01,  -7.91624188e-09],
                            [ -8.67982998e+00,   4.43554478e+00,  -1.08695203e+01],
                            [ -3.34591450e+00,  -9.15523687e+00,  -1.08695203e+01]])
        time_array = np.repeat(np.linspace(2458042.1, 2458042.2, Ntimes), Nblpairs)
        time_1_array = time_array
        time_2_array = time_array
        lst_array = np.repeat(np.ones(Ntimes, dtype=np.float), Nblpairs)
        lst_1_array = lst_array
        lst_2_array = lst_array
        time_avg_array = time_array
        lst_avg_array = lst_array
        spws = np.arange(Nspws)
        spw_array = np.tile(spws, Ndlys)
        freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), Nspws)
        dly_array = np.fft.fftshift(np.repeat(np.fft.fftfreq(Nfreqs, np.median(np.diff(freq_array))), Nspws))
        pol_array = np.array([-5])
        Npols = len(pol_array)
        units = 'unknown'
        weighting = 'identity'
        channel_width = np.median(np.diff(freq_array))
        history = 'example'
        taper = "none"
        norm = "I"
        git_hash = "random"
        scalar_array = np.ones((Nspws, Npols), np.float)

        telescope_location = np.array([5109325.85521063, 2005235.09142983, -3239928.42475397])

        cosmo = conversions.Cosmo_Conversions()

        data_array, wgt_array, integration_array, nsample_array = {}, {}, {}, {}
        for s in spws:
            data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) * blpair_array[:, None, None] / 1e9
            wgt_array[s] = np.ones((Nblpairts, Ndlys, 2, Npols), dtype=np.float)
            integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
            nsample_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)

        params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nblpairs', 'Nblpairts', 'Npols', 'Ndlys',
                  'Nbls', 'blpair_array', 'time_1_array', 'time_2_array', 'lst_1_array', 'lst_2_array',
                  'spw_array', 'dly_array', 'freq_array', 'pol_array', 'data_array', 'wgt_array',
                  'integration_array', 'bl_array', 'bl_vecs', 'telescope_location', 'units',
                  'channel_width', 'weighting', 'history', 'taper', 'norm', 'git_hash', 'nsample_array',
                  'time_avg_array', 'lst_avg_array', 'cosmo', 'scalar_array']

        for p in params:
            setattr(uvp, p, locals()[p])

        uvp.check()
        self.uvp = uvp

        # test equivalence
        nt.assert_equal(uvp, uvp)

        self.cosmo = cosmo
        self.beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits'))

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_param(self):
        a = parameter.PSpecParam("example", description="example", expected_type=int)

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
        k_perp, k_para = self.uvp.get_kvecs(0)
        nt.assert_equal(len(k_perp), 30)
        nt.assert_equal(len(k_para), 50)

    def test_convert_deltasq(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.add_cosmology(conversions.Cosmo_Conversions())
        uvp.convert_to_deltasq(little_h=True)
        k_perp, k_para = uvp.get_kvecs(0, little_h=True)
        k_mag = np.sqrt(k_perp[:, None, None]**2 + k_para[None, :, None]**2)
        nt.assert_true(np.isclose(uvp.data_array[0][0,:,0], (self.uvp.data_array[0]*k_mag**3/(2*np.pi**2))[0,:,0]).all())
        nt.assert_equal(uvp.units, 'unknown h^3 k^3 / (2pi^2)')

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

        # test exception
        nt.assert_raises(AssertionError, uvp.generate_noise_spectra, 0, 0, 0)

        # test generate_sense
        uvp.generate_sensitivity(self.beam)
        nt.assert_true(hasattr(uvp, 'sensitivity'))
        nt.assert_true(hasattr(uvp.sensitivity, 'beam'))
        nt.assert_true(hasattr(uvp.sensitivity, 'cosmo'))

        # test generate noise spectra
        P_N = uvp.generate_noise_spectra(0, -5, 500, form='Pk', real=True)
        nt.assert_equal(P_N.shape, (30, 50))

        # test smaller system temp
        P_N2 = uvp.generate_noise_spectra(0, -5, 400, form='Pk', real=True)
        nt.assert_true((P_N > P_N2).all())

        # test complex
        P_N2 = uvp.generate_noise_spectra(0, -5, 500, form='Pk', real=False)
        nt.assert_true((P_N < P_N2).all())

        # test Dsq
        Dsq = uvp.generate_noise_spectra(0, -5, 500, form='DelSq', real=True)
        nt.assert_equal(Dsq.shape, (30, 50))
        nt.assert_true(Dsq[0, 1] < P_N[0, 1])

    def test_average_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        # test blpair averaging
        blpairs = uvp.get_blpair_groups_from_bl_groups([[1002, 2003, 1003]], only_pairs_in_bls=False)
        uvp2 = uvp.average_spectra(blpair_groups=blpairs, time_avg=False, inplace=False)
        nt.assert_equal(uvp2.Nblpairs, 1)
        nt.assert_true(np.isclose(uvp2.get_nsamples(0, 1002001002, 'xx'), 3.0).all())
        nt.assert_equal(uvp2.get_data(0, 1002001002, 'xx').shape, (10, 50))
        # test time averaging
        uvp2 = uvp.average_spectra(time_avg=True, inplace=False)
        nt.assert_true(uvp2.Ntimes, 1)
        nt.assert_true(np.isclose(uvp2.get_nsamples(0, 1002001002, 'xx'), 10.0).all())
        nt.assert_true(uvp2.get_data(0, 1002001002, 'xx').shape, (1, 50))

    def test_fold_spectra(self):
        uvp = copy.deepcopy(self.uvp)
        uvp.fold_spectra()
        nt.assert_true(uvp.folded)
        nt.assert_raises(AssertionError, uvp.fold_spectra)
        nt.assert_equal(len(uvp.get_dlys(0)), 24)
        nt.assert_true(np.isclose(uvp.nsample_array[0], 2.0).all())


def test_conj_blpair_int():
    conj_blpair = uvpspec._conj_blpair_int(1002003004)
    nt.assert_equal(conj_blpair, 3004001002)

def test_conj_bl_int():
    conj_bl = uvpspec._conj_bl_int(1002)
    nt.assert_equal(conj_bl, 2001)

def test_conj_blpair():
    blpair = uvpspec._conj_blpair(1002003004, which='first')
    nt.assert_equal(blpair, 2001003004)
    blpair = uvpspec._conj_blpair(1002003004, which='second')
    nt.assert_equal(blpair, 1002004003)
    blpair = uvpspec._conj_blpair(1002003004, which='both')
    nt.assert_equal(blpair, 2001004003)
    nt.assert_raises(ValueError, uvpspec._conj_blpair, 2001003004, which='foo')




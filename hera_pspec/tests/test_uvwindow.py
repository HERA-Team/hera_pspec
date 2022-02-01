import unittest
import pytest
from pyuvdata import utils as uvutils
import uvtools.dspec as dspec
import h5py
import warnings
import numpy as np
import sys
import os
import time
import copy
from astropy import units
from scipy.interpolate import interp2d
from pyuvdata import UVBeam, UVData
from hera_pspec.data import DATA_PATH

from .. import conversions, noise, version, pspecbeam, grouping, utils, uvwindow
from .. import uvpspec_utils as uvputils

# Data files to use in tests
dfile = 'zen.2458116.30448.HH.uvh5'
ftfile = 'FT_beam_HERA_dipole_test'


class test_FTBeam(unittest.TestCase):

    def setUp(self):

        # parameters
        self.ft_file = os.path.join(DATA_PATH, ftfile)
        self.pol = 'xx'
        self.spw_range = (5, 25)
        self.verbose = False

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        assert test.ft_file == self.ft_file

        # tests related to verbose
        pytest.raises(ValueError, uvwindow.FTBeam, verbose=np.array([2, 3]),
                      pol=self.pol, spw_range=self.spw_range)

        # tests related to pol
        test = uvwindow.FTBeam(pol=-5, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        pytest.raises(AssertionError, uvwindow.FTBeam, pol='test',
                      spw_range=self.spw_range)
        pytest.raises(AssertionError, uvwindow.FTBeam, pol=12,
                      spw_range=self.spw_range)
        pytest.raises(TypeError, uvwindow.FTBeam, pol=3.4,
                      spw_range=self.spw_range)

        # tests related to ftfile
        pytest.raises(NotImplementedError, uvwindow.FTBeam, pol=self.pol,
                      spw_range=self.spw_range, ftfile=None)
        pytest.raises(ValueError, uvwindow.FTBeam, pol=self.pol,
                      spw_range=self.spw_range, ftfile=12.)
        # if ft file does not exist, raise assertion error
        pytest.raises(AssertionError, uvwindow.FTBeam, pol=self.pol,
                      spw_range=self.spw_range, ftfile='whatever')

        # tests related to spw_range
        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        assert test.spw_range == self.spw_range
        test = uvwindow.FTBeam(pol=self.pol, spw_range=None,
                               ftfile=self.ft_file, verbose=self.verbose)
        pytest.raises(AssertionError, uvwindow.FTBeam, spw_range=(13),
                      pol=self.pol, ftfile=self.ft_file)
        pytest.raises(AssertionError, uvwindow.FTBeam, spw_range=(20, 10),
                      pol=self.pol, ftfile=self.ft_file)
        pytest.raises(AssertionError, uvwindow.FTBeam, spw_range=(1001, 1022),
                      pol=self.pol, ftfile=self.ft_file)

    def test_get_bandwidth(self):

        # if ft_file is None
        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        test.ft_file = None
        pytest.raises(AssertionError, test.read_ft)

    def test_read_ft(self):

        # if ft_file is None
        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        test.ft_file = None
        pytest.raises(AssertionError, test.read_ft)

    def test_check(self):

        # if ft_file is None
        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        assert test.check()
        # if freq_array is empty, does not pass check (output is False)
        test.freq_array = []
        assert not test.check()
        # check fails if test.ft_beam.ndim!=3
        test = uvwindow.FTBeam(pol=self.pol, spw_range=self.spw_range,
                               ftfile=self.ft_file, verbose=self.verbose)
        test.ft_beam = np.zeros((2, 2))
        assert not test.check()


class Test_UVWindow(unittest.TestCase):

    def setUp(self):

        # Instantiate UVWindow()
        ft_file = os.path.join(DATA_PATH, ftfile)
        self.pol = 'xx'
        self.spw_range = (5, 25)
        self.taper = 'blackman-harris'
        self.verbose = False
        self.little_h = True
        self.cosmo = conversions.Cosmo_Conversions()

        self.ft_beam_obj = uvwindow.FTBeam(pol=self.pol, ftfile=ft_file,
                                           spw_range=None,
                                           verbose=self.verbose)
        self.uvw = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                     spw_range=self.spw_range,
                                     cosmo=self.cosmo,
                                     taper=self.taper, little_h=self.little_h,
                                     verbose=self.verbose)

        # set parameters
        # ft_beam = self.uvw.get_FT()
        self.freq_array = self.uvw.freq_array
        self.ngrid = self.ft_beam_obj.ft_beam.shape[-1]
        # HERA bandwidth
        self.HERA_bw = np.linspace(1, 2, 1024, endpoint=False)*1e8

        # define spherical kbins
        kmax, dk = 1., 0.128/2
        krange = np.arange(dk*1.5, kmax, step=dk)
        nbinsk = krange.size - 1
        kbins = (krange[1:]+krange[:-1])/2
        self.kbins = kbins * units.h / units.Mpc

        # Load datafile
        uvd = UVData()
        uvd.read(os.path.join(DATA_PATH, dfile), read_data=False)
        self.reds, self.lens, _ = utils.get_reds(uvd, bl_error_tol=1.0,
                                                 pick_data_ants=False)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # test different options for ftbeam param
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj)
        # test with ft_beam_obj that does not pass self.check
        ftb_test = copy.deepcopy(self.ft_beam_obj)
        ftb_test.freq_array = []
        pytest.raises(AssertionError, uvwindow.UVWindow, ft_beam_obj=ftb_test)

        # tests on spw_range
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=None)
        assert test.spw_range == (0, self.ft_beam_obj.freq_array.size)
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range)
        assert test.spw_range == self.spw_range

        pytest.raises(AssertionError, uvwindow.UVWindow, spw_range=(13),
                      ft_beam_obj=self.ft_beam_obj)
        pytest.raises(AssertionError, uvwindow.UVWindow, spw_range=(20, 10),
                      ft_beam_obj=self.ft_beam_obj)
        pytest.raises(AssertionError, uvwindow.UVWindow, spw_range=(100, 1022),
                      ft_beam_obj=self.ft_beam_obj)

        # test taper options
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 taper=self.taper)
        assert test.taper == self.taper
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 taper=None)
        assert test.taper is None
        pytest.raises(ValueError, uvwindow.UVWindow, taper='test',
                      ft_beam_obj=self.ft_beam_obj)

        # test on cosmo
        cosmo = conversions.Cosmo_Conversions()
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 cosmo=None)
        # test on verbose
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 verbose=False)
        assert not test.verbose
        pytest.raises(ValueError, uvwindow.UVWindow, verbose=np.array([2, 3]),
                      ft_beam_obj=self.ft_beam_obj)
        # test on little_h
        pytest.raises(ValueError, uvwindow.UVWindow, little_h=np.array([2, 3]),
                      ft_beam_obj=self.ft_beam_obj)
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj, little_h=False)
        assert test.kunits.is_equivalent(units.Mpc**(-1))

    def test_get_kgrid(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        kgrid, kperp_norm = test._get_kgrid(bl_len)
        pytest.raises(AssertionError, test._get_kgrid,
                      bl_len=bl_len, width=0.0004)

    def test_kperp4bl_freq(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        # test for correct input parameters
        k = test._kperp4bl_freq(freq=test.freq_array[12],
                               bl_len=bl_len,
                               ngrid=self.ngrid)
        # test for frequency outside of spectral window
        pytest.raises(AssertionError, test._kperp4bl_freq,
                      freq=1.35*1e8, bl_len=bl_len, ngrid=self.ngrid)
        # test for frequency in Hhz
        pytest.raises(AssertionError, test._kperp4bl_freq,
                      freq=test.freq_array[12]/1e6,
                      bl_len=bl_len,
                      ngrid=self.ngrid)

    def test_interpolate_ft_beam(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        ft_beam = np.copy(test.ft_beam_obj.ft_beam[test.spw_range[0]:test.spw_range[-1], :, :])
        interp_ft_beam, kperp_norm = test._interpolate_ft_beam(bl_len, ft_beam)

        # test for ft_beam of wrong dimensions
        pytest.raises(AssertionError, test._interpolate_ft_beam,
                      bl_len=bl_len, ft_beam=ft_beam[0, :, :])
        pytest.raises(AssertionError, test._interpolate_ft_beam,
                      bl_len=bl_len, ft_beam=ft_beam[0:10, :, :])
        pytest.raises(AssertionError, test._interpolate_ft_beam,
                      bl_len=bl_len, ft_beam=ft_beam[:, :, :].T)

    def test_take_freq_FT(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        ft_beam = np.copy(test.ft_beam_obj.ft_beam[test.spw_range[0]:test.spw_range[-1], :, :])
        interp_ft_beam, kperp_norm = test._interpolate_ft_beam(bl_len, ft_beam)
        # frequency resolution
        delta_nu = abs(test.freq_array[-1]-test.freq_array[0])/test.Nfreqs
        fnu = test._take_freq_FT(interp_ft_beam, delta_nu)
        # test for ft_beam of wrong dimensions
        pytest.raises(AssertionError, test._take_freq_FT,
                      interp_ft_beam[0, :, :], delta_nu)
        pytest.raises(AssertionError, test._take_freq_FT,
                      interp_ft_beam[:, :, :].T, delta_nu)

    def test_get_wf_for_tau(self):

        bl_len = self.lens[12]

        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        tau = test.dly_array[12]
        kperp_bins = test.get_kperp_bins([bl_len])
        kperp_bins = np.array(kperp_bins.value)
        kpara_bins = test.get_kpara_bins(test.freq_array, test.little_h,
                                         test.cosmo)
        kpara_bins = np.array(kpara_bins.value)

        wf_array1 = np.zeros((kperp_bins.size, test.Nfreqs))
        kpara, cyl_wf = test._get_wf_for_tau(tau, wf_array1,
                                             kperp_bins, kpara_bins)

    def test_get_kperp_bins(self):

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        # raise error if empty baseline array
        pytest.raises(AssertionError, test.get_kperp_bins, bl_lens=[])
        # test for unique baseline length
        kperps = test.get_kperp_bins(self.lens[12])
        assert test.kunits.is_equivalent(kperps.unit)
        # test for array of baseline lengths
        _ = test.get_kperp_bins(self.lens)
        # test for warning if large number of bins (> 200)
        _ = test.get_kperp_bins(np.r_[1., self.lens])

    def test_get_kpara_bins(self):

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)
        # raise error if empty freq array or length 1
        pytest.raises(AssertionError, test.get_kpara_bins,
                      freq_array=self.freq_array[2])
        # test for correct input
        _ = test.get_kpara_bins(self.freq_array, little_h=test.little_h,
                                cosmo=test.cosmo)
        # test for warning if large number of bins (> 200)
        _ = test.get_kpara_bins(self.HERA_bw, little_h=test.little_h,
                                cosmo=test.cosmo)
        # test if cosmo is None
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=None)
        kparas = test.get_kpara_bins(self.freq_array, little_h=test.little_h,
                                     cosmo=test.cosmo)
        assert test.kunits.is_equivalent(kparas.unit)

    def test_get_cylindrical_wf(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)

        _, _, cyl_wf = test.get_cylindrical_wf(bl_len,
                                               kperp_bins=None,
                                               kpara_bins=None,
                                               return_bins='weighted')
        cyl_wf = test.get_cylindrical_wf(bl_len,
                                         kperp_bins=None,
                                         kpara_bins=None,
                                         return_bins=None)
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len,
                                                       kperp_bins=None,
                                                       kpara_bins=None,
                                                       return_bins='unweighted')
        # check normalisation
        assert np.all(np.isclose(np.sum(cyl_wf, axis=(1, 2)), 1., atol=1e-3))
        assert kperp.size == cyl_wf.shape[1]
        assert kpara.size == cyl_wf.shape[2]
        assert test.Nfreqs == cyl_wf.shape[0]
        # test the bins are recovered by get_kperp_bins and get_kpara_bins
        assert np.all(kperp == test.get_kperp_bins(bl_len).value)
        assert np.all(kpara == test.get_kpara_bins(test.freq_array,
                                                   test.little_h,
                                                   test.cosmo).value)

        # test different key words

        # kperp bins
        kperp2, _, cyl_wf2 = test.get_cylindrical_wf(bl_len,
                                                     kperp_bins=kperp*test.kunits,
                                                     kpara_bins=None,
                                                     return_bins='unweighted')
        assert np.all(cyl_wf2 == cyl_wf)
        assert np.all(kperp2 == kperp)  # unweighted option to return_bins
        # kpara bins
        _, kpara3, cyl_wf3 = test.get_cylindrical_wf(bl_len,
                                                     kperp_bins=None,
                                                     kpara_bins=kpara*test.kunits,
                                                     return_bins='unweighted')
        assert np.all(cyl_wf3 == cyl_wf)
        assert np.all(kpara == kpara3)

        # test filling array by delay symmetry for odd number of delays
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=(self.spw_range[0],
                                            self.spw_range[1]-1))
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len,
                                                       return_bins='unweighted')

    def test_get_spherical_wf(self):

        bl_len = self.lens[12]

        # initialise object from keywords
        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 cosmo=self.cosmo,
                                 taper=self.taper, little_h=self.little_h,
                                 verbose=self.verbose)

        WF, counts = test.get_spherical_wf(kbins=self.kbins,
                                           bl_groups=self.reds[:1],
                                           bl_lens=self.lens[:1],
                                           kperp_bins=None,
                                           kpara_bins=None,
                                           return_weights=True,
                                           verbose=True)
        kperp_bins = test.get_kperp_bins(self.lens[:1])
        kpara_bins = test.get_kpara_bins(test.freq_array, test.little_h,
                                         test.cosmo)

        WF = test.get_spherical_wf(kbins=self.kbins,
                                   kperp_bins=kperp_bins,
                                   kpara_bins=kpara_bins,
                                   bl_groups=self.reds[:1],
                                   bl_lens=self.lens[:1],
                                   return_weights=False,
                                   verbose=None)

        # check inputs
        pytest.raises(AssertionError, test.get_spherical_wf, kbins=self.kbins,
                      bl_groups=self.reds[:1], bl_lens=self.lens[:2])
        pytest.raises(AttributeError, test.get_spherical_wf, kbins=self.kbins.value,
                      bl_groups=self.reds[:2], bl_lens=self.lens[:2])
        pytest.raises(AssertionError, test.get_spherical_wf, kbins=self.kbins,
                      bl_groups=None, bl_lens=self.lens[:2])
        pytest.raises(AssertionError, test.get_spherical_wf,
                      kbins=self.kbins.value[2]*test.kunits,
                      bl_groups=self.reds[:1], bl_lens=self.lens[:1])

        # test kpara bins not outside of spectral window
        # will print warning
        kpara_centre = test.cosmo.tau_to_kpara(test.avg_z,
                                               little_h=test.little_h)\
            * abs(test.dly_array).max()
        WF = test.get_spherical_wf(kbins=self.kbins, kperp_bins=kperp_bins,
                                   kpara_bins=np.arange(2.*kpara_centre,
                                                        10*kpara_centre,
                                                        step=kpara_centre)
                                   * test.kunits,
                                   bl_groups=self.reds[:1],
                                   bl_lens=self.lens[:1])

    def test_check_kunits(self):

        test = uvwindow.UVWindow(ft_beam_obj=self.ft_beam_obj,
                                 spw_range=self.spw_range,
                                 little_h=True)
        test.check_kunits(self.kbins)
        pytest.raises(AttributeError, test.check_kunits, self.kbins.value)

    def test_raise_warning(self):

        uvwindow.raise_warning('test', verbose=False)


if __name__ == "__main__":
    unittest.main()

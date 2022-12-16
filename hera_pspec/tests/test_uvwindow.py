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

from .. import conversions, noise, version, pspecbeam, grouping, utils
from .. import uvwindow, pspecbeam, PSpecData
from .. import uvpspec_utils as uvputils

# Data files to use in tests
dfile = 'zen.2458116.31939.HH.uvh5'
ftfile = 'FT_beam_HERA_dipole_test_xx.hdf5'
basename = 'FT_beam_HERA_dipole_test'
outfile = 'test.hdf5'


class test_FTBeam(unittest.TestCase):

    def setUp(self):

        # parameters
        self.ft_file = os.path.join(DATA_PATH, ftfile)
        self.pol = 'xx'
        self.spw_range = (5, 25)
        self.verbose = False
        self.x_orientation = 'east'

        self.ftbeam_obj = uvwindow.FTBeam.from_file(ftfile=self.ft_file, 
                                                    spw_range=self.spw_range)
        self.data = self.ftbeam_obj.ft_beam
        self.mapsize = self.ftbeam_obj.mapsize
        self.freq_array = self.ftbeam_obj.freq_array
        self.bandwidth = uvwindow.FTBeam.get_bandwidth(self.ft_file)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # initialise directly with array
        test = uvwindow.FTBeam(data=self.data,
                               pol=self.pol,
                               freq_array=self.freq_array,
                               mapsize=self.mapsize,
                               verbose=self.verbose,
                               x_orientation=self.x_orientation)
        assert self.pol == test.pol
        assert np.all(self.data == test.ft_beam)

        # raise assertion error if data is not dim 3
        pytest.raises(AssertionError, uvwindow.FTBeam, 
                      data=self.data[:, :, 0], pol=self.pol,
                      freq_array=self.freq_array, mapsize=self.mapsize)

        # raise assertion error if data has wrong shape
        pytest.raises(AssertionError, uvwindow.FTBeam, 
                      data=self.data[:, :, :-1], pol=self.pol,
                      freq_array=self.freq_array, mapsize=self.mapsize)

        # raise assertion error if freq_array and data not compatible
        pytest.raises(AssertionError, uvwindow.FTBeam, 
                      data=self.data[:12, :, :], pol=self.pol,
                      freq_array=self.freq_array, mapsize=self.mapsize)

        # tests related to pol
        test = uvwindow.FTBeam(data=self.data, 
                               pol=-5,
                               freq_array=self.freq_array,
                               mapsize=self.mapsize)
        pytest.raises(AssertionError, uvwindow.FTBeam, pol='test',
                      data=self.data, freq_array=self.freq_array,
                      mapsize=self.mapsize)
        pytest.raises(AssertionError, uvwindow.FTBeam, pol=12,
                      data=self.data, freq_array=self.freq_array,
                      mapsize=self.mapsize)
        pytest.raises(TypeError, uvwindow.FTBeam, pol=3.4,
                      data=self.data, freq_array=self.freq_array,
                      mapsize=self.mapsize)

    def test_from_beam(self):

        pytest.raises(NotImplementedError, uvwindow.FTBeam.from_beam, 
                      beamfile='test')

    def test_from_file(self):

        test = uvwindow.FTBeam.from_file(ftfile=self.ft_file,
                                         spw_range=self.spw_range,
                                         verbose=self.verbose,
                                         x_orientation=self.x_orientation)
        assert test.pol == self.pol

        # tests related to ftfile
        pytest.raises(TypeError, uvwindow.FTBeam.from_file, ftfile=12.)
        # if ft file does not exist, raise assertion error
        pytest.raises(ValueError, uvwindow.FTBeam.from_file, ftfile='whatever')

        # tests related to spw_range
        test1 = uvwindow.FTBeam.from_file(ftfile=self.ft_file,
                                          spw_range=self.spw_range)
        assert np.all(test1.freq_array == self.freq_array)

        test2 = uvwindow.FTBeam.from_file(ftfile=self.ft_file,
                                          spw_range=None)
        assert np.all(test2.freq_array == self.bandwidth)

        pytest.raises(AssertionError, uvwindow.FTBeam.from_file, spw_range=(13),
                      ftfile=self.ft_file)
        pytest.raises(AssertionError, uvwindow.FTBeam.from_file, spw_range=(20, 10),
                      ftfile=self.ft_file)
        pytest.raises(AssertionError, uvwindow.FTBeam.from_file, spw_range=(1001, 1022),
                      ftfile=self.ft_file)

    def test_gaussian(self):

        # fiducial use
        widths = -0.0343 * self.freq_array/1e6 + 11.30 
        test = uvwindow.FTBeam.gaussian(freq_array=self.freq_array,
                                        widths=widths,
                                        pol=self.pol)
        # if widths given as unique number, this value is used for all freqs
        test2 = uvwindow.FTBeam.gaussian(freq_array=self.freq_array,
                                        widths=np.mean(widths),
                                        pol=self.pol)

        # tests on freq_array consistency
        pytest.raises(AssertionError, uvwindow.FTBeam.gaussian,
                      freq_array=self.freq_array[:2], pol=self.pol,
                      widths=np.mean(widths))
        pytest.raises(AssertionError, uvwindow.FTBeam.gaussian,
                      freq_array=self.freq_array, pol=self.pol,
                      widths=widths[:10])

        # make sure widths are given in degrees (raises warning)
        test = uvwindow.FTBeam.gaussian(freq_array=self.freq_array,
                                        pol=self.pol, widths=0.10)

    def test_get_bandwidth(self):

        test_bandwidth = uvwindow.FTBeam.get_bandwidth(self.ft_file)
        assert np.all(test_bandwidth == self.bandwidth)
        # raise error is ft_file does not exist
        pytest.raises(ValueError, uvwindow.FTBeam.get_bandwidth, 
                      ftfile='whatever')

    def test_update_spw(self):

        # proper usage
        test = uvwindow.FTBeam.from_file(ftfile=self.ft_file,
                                         spw_range=None)
        test.update_spw(self.spw_range)

        # tests related to spw_range        
        test = uvwindow.FTBeam.from_file(ftfile=self.ft_file,
                                         spw_range=None)
        pytest.raises(AssertionError, test.update_spw, spw_range=(13))
        pytest.raises(AssertionError, test.update_spw, spw_range=(20, 10))
        pytest.raises(AssertionError, test.update_spw, spw_range=(1001, 1022))


class Test_UVWindow(unittest.TestCase):

    def setUp(self):

        # Instantiate UVWindow()
        self.ft_file = os.path.join(DATA_PATH, ftfile)
        self.pol = 'xx'
        self.polpair = (self.pol, self.pol)
        self.spw_range = (5, 25)
        self.taper = 'blackman-harris'
        self.verbose = False
        self.little_h = True
        self.cosmo = conversions.Cosmo_Conversions()

        self.ft_beam_obj_spw = uvwindow.FTBeam.from_file(ftfile=self.ft_file, 
                                                         spw_range=self.spw_range)
        self.ft_beam_obj = uvwindow.FTBeam.from_file(ftfile=self.ft_file)

        self.uvw = uvwindow.UVWindow(ftbeam_obj = self.ft_beam_obj_spw,
                                     taper=self.taper,
                                     cosmo=self.cosmo,
                                     little_h=self.little_h,
                                     verbose=self.verbose)

        # set parameters
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

        # fiducial usage
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)

        # raise error if two ftbeam_obj are not consistent
        pytest.raises(AssertionError, uvwindow.UVWindow,
                      ftbeam_obj=(self.ft_beam_obj_spw, self.ft_beam_obj))    
        ftbeam_test = copy.deepcopy(self.ft_beam_obj_spw)
        ftbeam_test.mapsize = 2.
        pytest.raises(AssertionError, uvwindow.UVWindow,
                      ftbeam_obj=(self.ft_beam_obj_spw, ftbeam_test))    
        # raise error if ftbeam_obj is wrong input
        pytest.raises(AssertionError, uvwindow.UVWindow, ftbeam_obj='test') 

        # test taper options
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw,
                                 taper=self.taper)
        assert test.taper == self.taper
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw,
                                 taper=None)
        assert test.taper is None
        pytest.raises(ValueError, uvwindow.UVWindow, taper='test',
                      ftbeam_obj=self.ft_beam_obj_spw)

        # test on cosmo
        cosmo = conversions.Cosmo_Conversions()
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw,
                                 cosmo=cosmo)
        pytest.raises(AssertionError, uvwindow.UVWindow, cosmo=None,
                      ftbeam_obj=self.ft_beam_obj_spw)

        # test on verbose
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw,
                                 verbose=True)
        assert test.verbose

        # test on little_h
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw,
                                 verbose=True)
        assert test.kunits.is_equivalent(units.h / units.Mpc)
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw, little_h=False)
        assert test.kunits.is_equivalent(units.Mpc**(-1))

    def test_from_uvpspec(self):

        # obtain uvp object
        datafile = os.path.join(DATA_PATH, dfile)
        uvd = UVData()
        uvd.read_uvh5(datafile)
        # beam 
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        uvb = pspecbeam.PSpecBeamUV(beamfile, cosmo=None)
        Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol=self.pol)
        # reshape to appropriately match a UVData.data_array object and multiply in!
        uvd.data_array *= Jy_to_mK[None, None, :, None]
        # Create a new PSpecData object, and don't forget to feed the beam object
        ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
        ds_nocosmo = PSpecData(dsets=[uvd, uvd], wgts=[None, None])
        # choose baselines
        baselines1, baselines2, blpairs = utils.construct_blpairs(uvd.get_antpairs()[1:],
                                                                  exclude_permutations=False,
                                                                  exclude_auto_bls=True)
        # compute ps
        uvp = ds.pspec(baselines1, baselines2, dsets=(0, 1), pols=[self.polpair], 
                       spw_ranges=(175,195), taper=self.taper,verbose=self.verbose)
        uvp_nocosmo = ds_nocosmo.pspec(baselines1, baselines2, dsets=(0, 1), pols=[self.polpair], 
                                       spw_ranges=self.spw_range, taper=self.taper,verbose=self.verbose)
        uvp_crosspol = ds.pspec(baselines1, baselines2, dsets=(0, 1), pols=['xx', 'yy'], 
                                spw_ranges=(175,195), taper=self.taper,verbose=self.verbose)

        # proper usage
        # initialise with ftfile (read pre-computed FT of the beam in file)
        uvw_ps = uvwindow.UVWindow.from_uvpspec(uvp, ipol=0, spw=0, verbose=True,
                                                ftfile=os.path.join(DATA_PATH, basename))
        # if cross polarisation
        test = uvwindow.UVWindow.from_uvpspec(uvp_crosspol, ipol=0, spw=0, 
                                              ftfile=os.path.join(DATA_PATH, basename))
        # if no cosmo, use default
        uvw_ps = uvwindow.UVWindow.from_uvpspec(uvp_nocosmo, ipol=0, spw=0, verbose=True,
                                                ftfile=os.path.join(DATA_PATH, basename))

        # raise error if no ftfile as option is not implemented yet
        pytest.raises(NotImplementedError, uvwindow.UVWindow.from_uvpspec, uvp=uvp,
                      ipol=0, spw=0, ftfile=None, verbose=False)
        # raise error if spw not within uvp.Nspws
        pytest.raises(AssertionError, uvwindow.UVWindow.from_uvpspec, uvp=uvp_nocosmo,
                      ipol=0, spw=2, ftfile=os.path.join(DATA_PATH, basename))

    def test_get_kgrid(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        kgrid, kperp_norm = test._get_kgrid(bl_len)
        pytest.raises(AssertionError, test._get_kgrid,
                      bl_len=bl_len, width=0.0004)

    def test_kperp4bl_freq(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
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
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        ft_beam = np.copy(test.ftbeam_obj_pol[0].ft_beam)
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
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        ft_beam = np.copy(test.ftbeam_obj_pol[0].ft_beam)
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

        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        tau = test.dly_array[12]
        kperp_bins = test.get_kperp_bins([bl_len])
        kperp_bins = np.array(kperp_bins.value)
        kpara_bins = test.get_kpara_bins(test.freq_array)
        kpara_bins = np.array(kpara_bins.value)

        wf_array1 = np.zeros((kperp_bins.size, test.Nfreqs))
        kpara, cyl_wf = test._get_wf_for_tau(tau, wf_array1,
                                             kperp_bins, kpara_bins)

    def test_get_kperp_bins(self):

        # initialise object
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
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
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        # raise error if empty freq array or length 1
        pytest.raises(AssertionError, test.get_kpara_bins,
                      freq_array=self.freq_array[2])
        # test for correct input
        _ = test.get_kpara_bins(self.freq_array)
        # test for warning if large number of bins (> 200)
        _ = test.get_kpara_bins(self.HERA_bw)
        # test if cosmo is signature
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        kparas = test.get_kpara_bins(self.freq_array)
        assert test.kunits.is_equivalent(kparas.unit)

    def test_get_cylindrical_wf(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)

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
        assert np.all(kpara == test.get_kpara_bins(test.freq_array).value)

        # test different key words

        # kperp bins
        kperp2, _, cyl_wf2 = test.get_cylindrical_wf(bl_len,
                                                     kperp_bins=kperp*test.kunits,
                                                     kpara_bins=None,
                                                     return_bins='unweighted')
        assert np.all(cyl_wf2 == cyl_wf)
        assert np.all(kperp2 == kperp)  # unweighted option to return_bins
        # ValueError raised if kperp_bins not linearly spaced
        kperp_log = np.logspace(-2, 0, 100)
        pytest.raises(ValueError, test.get_cylindrical_wf, bl_len,
                      kperp_bins=kperp_log*test.kunits, kpara_bins=None,
                      return_bins='unweighted')

        # kpara bins
        _, kpara3, cyl_wf3 = test.get_cylindrical_wf(bl_len,
                                                     kperp_bins=None,
                                                     kpara_bins=kpara*test.kunits,
                                                     return_bins='unweighted')
        assert np.all(cyl_wf3 == cyl_wf)
        assert np.all(kpara == kpara3)
        # ValueError raised if kpara_bins not linearly spaced
        kpara_log = np.logspace(-1, 1, 100)
        pytest.raises(ValueError, test.get_cylindrical_wf, bl_len,
                      kperp_bins=None, kpara_bins=kpara_log*test.kunits,
                      return_bins='unweighted')

        # test filling array by delay symmetry for odd number of delays
        ft_beam_test = uvwindow.FTBeam.from_file(ftfile=self.ft_file, 
                                                 spw_range=(self.spw_range[0],
                                                 self.spw_range[1]-1))
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_test)
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len,
                                                       return_bins='unweighted')

    def test_cylindrical_to_spherical(self):

        bl_len = self.lens[12]

        # initialise object from keywords
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len,
                                                       kperp_bins=None,
                                                       kpara_bins=None,
                                                       return_bins='unweighted')
        ktot = np.sqrt(kperp[:, None]**2+kpara**2)
        
        # proper usage
        sph_wf, weighted_k = test.cylindrical_to_spherical(cyl_wf=cyl_wf, 
                                                          kbins=self.kbins, 
                                                          ktot=ktot, 
                                                          bl_lens=bl_len,
                                                          bl_weights=[2.])
        sph_wf, weighted_k = test.cylindrical_to_spherical(cyl_wf=cyl_wf[None], 
                                                          kbins=self.kbins, 
                                                          ktot=ktot, 
                                                          bl_lens=bl_len,
                                                          bl_weights=None)

        # ktot has shape different from cyl_wf
        pytest.raises(AssertionError, test.cylindrical_to_spherical, cyl_wf=cyl_wf,
                      kbins=self.kbins, ktot=np.sqrt(kperp[:-2, None]**2+kpara**2),
                      bl_lens=bl_len)
        # only one k-bin
        pytest.raises(AssertionError, test.cylindrical_to_spherical, cyl_wf=cyl_wf,
                      kbins=self.kbins[:1], ktot=ktot, bl_lens=bl_len)
        # weights have shape different from bl_lens
        pytest.raises(AssertionError, test.cylindrical_to_spherical, cyl_wf=cyl_wf,
                      kbins=self.kbins, ktot=ktot, bl_lens=bl_len,
                      bl_weights=[1.,2.])
        # bl_lens has different size to cyl_wf.shape[0]
        pytest.raises(AssertionError, test.cylindrical_to_spherical, cyl_wf=cyl_wf[None],
                      kbins=self.kbins, ktot=ktot, bl_lens=self.lens[:2],
                      bl_weights=[1.,2.])
        # raise warning if empty bins
        kbins_test = np.arange(2,5,step=.5)*test.kunits
        test.verbose = True
        sph_wf, weighted_k = test.cylindrical_to_spherical(cyl_wf=cyl_wf, 
                                                          kbins=kbins_test, 
                                                          ktot=ktot, 
                                                          bl_lens=bl_len)        

        # raise ValueError if kbins not linearly spaced
        kbins_log = np.logspace(-2, 2, 20)
        pytest.raises(ValueError, test.cylindrical_to_spherical, cyl_wf=cyl_wf, 
                      kbins=kbins_log*test.kunits, ktot=ktot, 
                      bl_lens=bl_len)  

    def test_get_spherical_wf(self):

        bl_len = self.lens[12]

        # initialise object from keywords
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)

        WF, weighted_k = test.get_spherical_wf(kbins=self.kbins,
                                           bl_lens=self.lens[:1],
                                           bl_weights=[1],
                                           kperp_bins=None,
                                           kpara_bins=None,
                                           return_weighted_k=True,
                                           verbose=True)
        kperp_bins = test.get_kperp_bins(self.lens[:1])
        kpara_bins = test.get_kpara_bins(test.freq_array)
        print(np.diff(kpara_bins), np.diff(kperp_bins))

        WF = test.get_spherical_wf(kbins=self.kbins,
                                   kperp_bins=kperp_bins,
                                   kpara_bins=kpara_bins,
                                   bl_lens=self.lens[:1],
                                   bl_weights=None,
                                   return_weighted_k=False,
                                   verbose=None)

        # check inputs
        pytest.raises(AttributeError, test.get_spherical_wf, kbins=self.kbins.value,
                      bl_lens=self.lens[:2])
        pytest.raises(AssertionError, test.get_spherical_wf, kbins=self.kbins,
                      bl_lens=self.lens[:2], bl_weights=[1.])
        pytest.raises(AssertionError, test.get_spherical_wf,
                      kbins=self.kbins.value[2]*test.kunits,
                      bl_lens=self.lens[:1])

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
                                   bl_lens=self.lens[:1])
        # ValueError raised if kbins not linearly spaced
        kperp_log = np.logspace(-2, 0, 100)
        pytest.raises(ValueError, test.get_spherical_wf, kbins=self.kbins, 
                      kperp_bins=kperp_log*test.kunits, bl_lens=self.lens[:1])
        kpara_log = np.logspace(-1, 1, 100)
        pytest.raises(ValueError, test.get_spherical_wf, kbins=self.kbins, 
                      kpara_bins=kpara_log*test.kunits, bl_lens=self.lens[:1])
        kbins_log = np.logspace(-2, 2, 20)
        pytest.raises(ValueError, test.get_spherical_wf, kbins=kbins_log*test.kunits,
                      bl_lens=self.lens[:1])


    def test_check_kunits(self):

        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        test.check_kunits(self.kbins)
        pytest.raises(AttributeError, test.check_kunits, self.kbins.value)

    def test_run_and_write(self):

        filepath = os.path.join(DATA_PATH, outfile)

        # initialise object from keywords
        if os.path.exists(filepath):
            os.remove(filepath)
        test = uvwindow.UVWindow(ftbeam_obj=self.ft_beam_obj_spw)
        kperp_bins = test.get_kperp_bins(self.lens[:1])
        kpara_bins = test.get_kpara_bins(test.freq_array)
        # proper usage
        test.run_and_write(filepath=filepath,
                           bl_lens=self.lens[:1],
                           kperp_bins=kperp_bins,
                           kpara_bins=kpara_bins,
                           clobber=False)

        # raise error if file already exists and clobber is False
        pytest.raises(IOError, test.run_and_write, filepath=filepath,
                      bl_lens=self.lens[:1],
                      clobber=False)
        # does not if clobber is True
        test.run_and_write(filepath=filepath,
                           bl_lens=[self.lens[:1]],
                           kperp_bins=None, kpara_bins=None,
                           clobber=True)

        # check inputs
        pytest.raises(AssertionError, test.run_and_write, filepath=filepath,
                      bl_lens=self.lens[:1], bl_weights=[1.,1.], clobber=True)
        pytest.raises(AttributeError, test.run_and_write, filepath=filepath,
                      bl_lens=self.lens[:1],
                      kperp_bins=kperp_bins.value, clobber=True)
        pytest.raises(AttributeError, test.run_and_write, filepath=filepath,
                      bl_lens=self.lens[:1],
                      kpara_bins=kpara_bins.value, clobber=True)

        if os.path.exists(filepath):
            os.remove(filepath)
            
if __name__ == "__main__":
    unittest.main()

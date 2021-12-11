import unittest
import pytest
from pyuvdata import utils as uvutils
import uvtools.dspec as dspec
import h5py
import warnings
import numpy as np
import sys, os, time
from scipy.interpolate import interp2d
from pyuvdata import UVBeam, UVData 
from hera_pspec.data import DATA_PATH

from .. import conversions, noise, version, pspecbeam, grouping, utils, uvwindow, uvpspec_utils as uvputils

# Data files to use in tests
dfile = 'zen.2458116.30448.HH.uvh5'
ftfile = 'FT_beam_HERA_dipole_test'

class Test_UVWindow(unittest.TestCase):

    def setUp(self):

        # Instantiate UVWindow()
        self.ft_file = os.path.join(DATA_PATH, ftfile)
        self.uvw = uvwindow.UVWindow(ftbeam=self.ft_file)
        self.pol = 'xx'
        self.spw_range = (5,25)

        # set parameters
        self.uvw.set_polarisation(pol=self.pol)
        self.uvw.set_spw_range(spw_range=self.spw_range)
        FT_beam = self.uvw.get_FT()
        self.freq_array = self.uvw.freq_array
        self.ngrid = FT_beam.shape[-1]
        # HERA bandwidth
        self.HERA_bw = np.linspace(1,2,1024,endpoint=False)*1e8

        # define spherical kbins
        kmax, dk = 1., 0.128/2
        krange = np.arange(dk*1.5,kmax,step=dk)
        nbinsk = krange.size -1
        self.kbins = (krange[1:]+krange[:-1])/2

        # Load datafile
        self.uvd = UVData()
        self.uvd.read(os.path.join(DATA_PATH, dfile), read_data=False)
        self.reds, self.lens, _ = utils.get_reds(self.uvd,bl_error_tol=1.0,pick_data_ants=False)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # test different options for ftbeam param
        test = uvwindow.UVWindow(ftbeam='')
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        assert test.ft_file == self.ft_file
        pytest.raises(ValueError, uvwindow.UVWindow, ftbeam=2)

        # test different options for uvdata
        test = uvwindow.UVWindow(uvdata = self.uvd)
        assert test.is_uvdata
        test = uvwindow.UVWindow(uvdata = os.path.join(DATA_PATH, dfile))
        assert test.is_uvdata
        test = uvwindow.UVWindow(uvdata = '')
        assert (not test.is_uvdata)

        # test other key words
        cosmo = conversions.Cosmo_Conversions()
        test = uvwindow.UVWindow(cosmo=cosmo)
        pytest.raises(AssertionError, uvwindow.UVWindow, verbose=2)
        pytest.raises(AssertionError, uvwindow.UVWindow, little_h=2)
        pytest.raises(AssertionError, uvwindow.UVWindow, taper=2)

    def test_get_bandwidth(self):

        # initialise object 
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        # raise error if polarisation not set
        pytest.raises(AssertionError, test.get_bandwidth,file=self.ft_file)
        test.set_polarisation(pol=self.pol)
        # raise error if file does not exist
        pytest.raises(AssertionError, test.get_bandwidth,file=ftfile)


    def test_set_taper(self):

        # test for different input tapers
        test = uvwindow.UVWindow(ftbeam=self.ft_file,taper='none')
        taper = 'blackman-harris'
        test.set_taper(taper=taper,clear_cache=True)
        assert test.taper == taper

        # test warning if updating taper after computation of wf
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        cyl_wf = test.get_cylindrical_wf(self.lens[12]) 
        test.set_taper(taper='none',clear_cache=True)
        
    def test_set_spw_range(self):

        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        # test need to set polarisation first
        pytest.raises(AssertionError, test.set_spw_range, spw_range=self.spw_range)
        test.set_polarisation(pol=self.pol)
        # test input as tuple
        test.set_spw_range(spw_range=self.spw_range)
        assert test.spw_range == self.spw_range
        # test input as array
        test.set_spw_range(spw_range=np.array(self.spw_range))
        assert test.spw_range == self.spw_range
        # test wrong inputs: len(taper)!=2 
        pytest.raises(AssertionError, test.set_spw_range, spw_range=(12))
        pytest.raises(AssertionError, test.set_spw_range, spw_range=2)
        # test for spw range not compatible with bandwidth
        pytest.raises(AssertionError, test.set_spw_range, spw_range=(1059,2314))
        # test for spw range not increasing
        pytest.raises(AssertionError, test.set_spw_range, spw_range=(20,10))


    def test_set_spw_parameters(self):

        # initialise HERA bandwidth, normally read in self.ft_file
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        pytest.raises(AssertionError, test.set_spw_range, spw_range=self.spw_range)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        test.set_spw_parameters(bandwidth=self.HERA_bw)
        # test for wrong input: len(bandwidth)<2
        pytest.raises(AssertionError, test.set_spw_parameters, bandwidth=12)
        # test for comparison of bandwifth in UVData and in bandwidth
        test = uvwindow.UVWindow(uvdata = os.path.join(DATA_PATH, dfile))
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        test.set_spw_parameters(bandwidth=self.HERA_bw)

    def test_set_polarisation(self):

        # initialise object from keywords
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        # test with string polarisation
        test.set_polarisation(pol=self.pol)
        # test with int polarisation
        test.set_polarisation(pol=-5)
        # test with wrong polarisation
        pytest.raises(AssertionError, test.set_polarisation, pol='ii')
        pytest.raises(AssertionError, test.set_polarisation, pol=12)
        pytest.raises(TypeError, test.set_polarisation, pol=(3,4))

        # initialise object from data file
        test = uvwindow.UVWindow(ftbeam=self.ft_file, uvdata = self.uvd)
        test.set_polarisation(pol=self.pol)
        # test if pol asked is in data file
        pytest.raises(AssertionError, test.set_polarisation, pol=2)

    def test_get_FT(self):

        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        # need to set polarisation before calling the function
        pytest.raises(AssertionError, test.get_FT)
        test.set_polarisation(pol=self.pol)
        # need to set spectral window before calling the function
        pytest.raises(AssertionError, test.get_FT)
        test.set_spw_range(spw_range=self.spw_range)
        # test error is raised if file does not exist
        pytest.raises(AssertionError, test.get_FT, file=ftfile)
        # import FT of beam from self.ft_file attribute
        FT_beam = test.get_FT()
        # import FT beam from other file
        FT_beam2 = test.get_FT(file=self.ft_file)
        assert np.all(FT_beam==FT_beam2)
        # check if spw parameters have been properly set
        assert test.avg_z is not None

    def test_get_kgrid(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        FT_beam = test.get_FT()
        kgrid, kperp_norm = test.get_kgrid(bl_len)
        pytest.raises(AssertionError, test.get_kgrid, bl_len=bl_len, width=0.0004)

    def test_kperp4bl_freq(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        # test with FT parameters not initialised
        pytest.raises(AssertionError, test.kperp4bl_freq, freq=self.freq_array[12],bl_len=bl_len,ngrid = self.ngrid)
        FT_beam = test.get_FT()
        # test for correct input parameters
        k = test.kperp4bl_freq(freq=test.freq_array[12],bl_len=bl_len,ngrid = FT_beam.shape[-1])
        # test for frequency outside of spectral window
        pytest.raises(AssertionError, test.kperp4bl_freq, freq=1.35*1e8,bl_len=bl_len,ngrid = FT_beam.shape[-1])
        # test for frequency in Hhz
        pytest.raises(AssertionError, test.kperp4bl_freq, freq=test.freq_array[12]/1e6,bl_len=bl_len,ngrid = FT_beam.shape[-1])

    def test_interpolate_FT_beam(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        FT_beam = test.get_FT()     
        interp_FT_beam, kperp_norm = test.interpolate_FT_beam(bl_len, FT_beam)   

        # test for FT_beam of wrong dimensions
        pytest.raises(AssertionError, test.interpolate_FT_beam, bl_len=bl_len, FT_beam=FT_beam[0,:,:])

    def test_take_freq_FT(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        FT_beam = test.get_FT()     
        interp_FT_beam, kperp_norm = test.interpolate_FT_beam(bl_len, FT_beam)   
        # frequency resolution
        delta_nu = abs(test.freq_array[-1]-test.freq_array[0])/test.Nfreqs
        fnu = test.take_freq_FT(interp_FT_beam, delta_nu, taper=test.taper)
        # taper = '' should use test.taper
        fnu2 = test.take_freq_FT(interp_FT_beam, delta_nu, taper='')
        assert np.all(fnu==fnu2)
        # other input taper
        fnu = test.take_freq_FT(interp_FT_beam, delta_nu, taper='blackman-harris')
        # taper = 'none'
        fnu = test.take_freq_FT(interp_FT_beam, delta_nu, taper='none')
        assert test.taper == 'none'

    def test_get_kperp_bins(self):

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        # raise error if get_FT has not been called
        pytest.raises(AssertionError, test.get_kperp_bins, bl_lens=self.lens)
        test.set_spw_range(spw_range=self.spw_range)
        _ = test.get_FT()
        # raise error if empty baseline array
        pytest.raises(AssertionError, test.get_kperp_bins, bl_lens=[])
        # test for unique baseline length
        _ = test.get_kperp_bins(self.lens[12])
        # test for array of baseline lengths
        _ = test.get_kperp_bins(self.lens)
        # test for warning if large number of bins (> 200)
        _ = test.get_kperp_bins(np.r_[1.,self.lens])

    def test_get_kpara_bins(self):

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        # raise error if empty freq array or length 1
        pytest.raises(AssertionError, test.get_kpara_bins, freq_array=self.freq_array[2])
        # test for correct input
        _ = test.get_kpara_bins(self.freq_array, little_h=test.little_h, cosmo=test.cosmo)
        # test for warning if large number of bins (> 200)
        _ = test.get_kpara_bins(self.HERA_bw, little_h=test.little_h, cosmo=test.cosmo)


    def test_get_cylindrical_wf(self):

        bl_len = self.lens[12]

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        FT_beam = test.get_FT()     

        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len, FT_beam,
                                kperp_bins=[],kpara_bins=[],
                                return_bins='unweighted') 
        # check normalisation
        assert np.all(np.isclose(np.sum(cyl_wf,axis=(1,2)),1.,atol=1e-3))
        assert kperp.size == cyl_wf.shape[1]
        assert kpara.size == cyl_wf.shape[2]
        assert test.Nfreqs == cyl_wf.shape[0]
        # test the bins are recovered by get_kperp_bins and get_kpara_bins
        assert np.all(kperp == test.get_kperp_bins(bl_len))
        assert np.all(kpara == test.get_kpara_bins(test.freq_array,test.little_h,test.cosmo))

        #### test different key words

        # kperp bins
        kperp2, _, cyl_wf2 = test.get_cylindrical_wf(bl_len, FT_beam,
                                kperp_bins=kperp,kpara_bins=[],
                                return_bins='unweighted') 
        assert np.all(cyl_wf2==cyl_wf)
        assert np.all(kperp2==kperp) #unweighted option to return_bins
        # kpara bins
        _, kpara3, cyl_wf3 = test.get_cylindrical_wf(bl_len, FT_beam,
                                kperp_bins=[],kpara_bins=kpara,
                                return_bins='unweighted') 
        assert np.all(cyl_wf3==cyl_wf)
        assert np.all(kpara==kpara3)

        # test filling array by delay symmetry for odd number of delays
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=(self.spw_range[0],self.spw_range[1]-1))
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len,
                                return_bins='weighted') 


    def test_get_spherical_wf(self):

        bl_len = self.lens[12]

        # initialise object from keywords
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)
        FT_beam = test.get_FT()     

        WF, counts = test.get_spherical_wf(spw_range=self.spw_range,pol=self.pol,
                            kbins=self.kbins, kperp_bins=[], kpara_bins=[],
                            bl_groups=self.reds[:1],bl_lens=self.lens[:1], 
                            save_cyl_wf = False, return_weights=True,
                            verbose=True)
        assert len(test.cyl_wf)==0
        kperp_bins = test.kperp_bins
        kpara_bins = test.kpara_bins

        WF = test.get_spherical_wf(spw_range=self.spw_range,pol=self.pol,
                            kbins=self.kbins, kperp_bins=kperp_bins, kpara_bins=kpara_bins,
                            bl_groups=self.reds[:1],bl_lens=self.lens[:1], 
                            save_cyl_wf = True, return_weights=False,
                            verbose=None)
        assert len(test.cyl_wf)>0
        assert np.all(test.kperp_bins==kperp_bins)

        # initialise object from keywords
        test2 = uvwindow.UVWindow(ftbeam=self.ft_file, uvdata = self.uvd)
        test2.set_polarisation(pol=self.pol)
        test2.set_spw_range(spw_range=self.spw_range)
        # test file for UVWindow has small bandwidth to limit file size
        # so initialisation with data file will raise error as
        # data file bandwidth != FT file bandwidth
        pytest.raises(AssertionError, test2.get_FT)
        # test for no bl input
        pytest.raises(AssertionError, test2.get_spherical_wf,spw_range=self.spw_range,
                            pol=self.pol, kbins=self.kbins)
        # test for accuraste bl_groups input
        pytest.raises(AssertionError, test2.get_spherical_wf,spw_range=self.spw_range,
                            pol=self.pol, kbins=self.kbins, 
                            bl_groups = [(23, 23)], bl_lens = [14.01])
        # test for wrong bl_groups input (bls not in dat file)
        pytest.raises(AssertionError, test2.get_spherical_wf,spw_range=self.spw_range,
                            pol=self.pol, kbins=self.kbins, 
                            bl_groups = [(37, 37)], bl_lens = [14.01])

        # test kpara bins not outside of spectral window
        # will print warning
        kpara_centre = test.cosmo.tau_to_kpara(test.avg_z,little_h=test.little_h)*abs(test.dly_array).max()
        WF = test.get_spherical_wf(spw_range=self.spw_range,pol=self.pol,
                            kbins=self.kbins, kperp_bins=kperp_bins, 
                            kpara_bins=np.arange(2.*kpara_centre,10*kpara_centre,step=kpara_centre),
                            bl_groups=self.reds[:1],bl_lens=self.lens[:1])


    def test_clear_cache(self):

        # initialise object
        test = uvwindow.UVWindow(ftbeam=self.ft_file)
        test.set_polarisation(pol=self.pol)
        test.set_spw_range(spw_range=self.spw_range)

        WF = test.get_spherical_wf(spw_range=self.spw_range,pol=self.pol,
                            kbins=self.kbins, kperp_bins=[], kpara_bins=[],
                            bl_groups=self.reds[:1],bl_lens=self.lens[:1], 
                            save_cyl_wf = True, return_weights=False,
                            verbose=False)
        # test clear_cache effectively delete wf arrays and cylindrical bins arrays
        assert len(test.cyl_wf)>0
        test.clear_cache(clear_cyl_bins=False)
        assert len(test.cyl_wf)==0
        assert len(test.kperp_bins)>0
        test.clear_cache(clear_cyl_bins=True)
        assert len(test.kperp_bins)==0

    def test_raise_warning(self):

        uvwindow.raise_warning('test',verbose=False)


if __name__ == "__main__":
    unittest.main()

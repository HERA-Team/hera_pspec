from pyuvdata import utils as uvutils
import uvtools.dspec as dspec
import h5py
import warnings
import numpy as np
import sys, os, time
from scipy.interpolate import interp2d

from . import conversions, noise, version, pspecbeam, grouping, utils, uvpspec_utils as uvputils

chan_nb = 1024
HERA_bw=np.linspace(1.,2.,chan_nb,endpoint=False)*1e8

class UVWindow(object):
    """
    An object for storing window functions copmuted without the delay approximation
    """

    def __init__(self, ftbeam, cosmo=None, little_h=True,verbose=False,taper='blackman-harris'):

        # Summary attributes
        if isinstance(ftbeam, str):
            self.ft_file = ftbeam
        elif ftbeam=='default':
            self.ft_file = 'blabla'#to define
        else:
            raise Warning('No input FT beam, will compute all window functions from scratch... Will take a few hours.')
            ##### to be coded up

        if cosmo is None: cosmo = conversions.Cosmo_Conversions()
        self.cosmo = cosmo
        self.little_h = little_h
        self.verbose = verbose
        self.taper = taper

        self.freq_array = None
        self.Nfreqs = None
        self.dly_array = None
        self.spw_range = None
        self.avg_nu = None
        self.avg_z = None
        self.pol = None

    def set_spw_range(self,spw_range):
        """
        Sets the spectral range considered to compute the window functions.

        Parameters
        ----------
        spw_range : tuple
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA bandwidth).

        """

        assert len(spw_range)==2, "spw_range must be fed as a tuple of frequency indices between 0 and 1024"

        self.spw_range = tuple(spw_range)
        self.freq_array = HERA_bw[spw_range[0]:spw_range[-1]]
        self.Nfreqs = len(self.freq_array)
        self.dly_array = utils.get_delays(self.freq_array,n_dlys=len(self.freq_array))
        self.avg_nu = np.mean(self.freq_array)
        self.avg_z = self.cosmo.f2z(self.avg_nu)

    def set_polarisation(self,pol):
        """
        Sets the polarisation considered for the beam to compute the window functions.

        Parameters
        ----------
        pol : str
            Can be pseudo-Stokes or power: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
        
        """

        self.pol = pol 

    def get_FT(self):
        """
        Loads the Fourier transform (FT) of the beam from different attributes: 
            - self.pol
            - self.spw_range
        Note that the array has no physical coordinates, they will be later 
        attributed by kperp4bl_freq.

        Returns
        ----------
        Atilde : array_like
            Real part of the Fourier transform of the beam along the spectral
            window considered. Has dimensions (Nfreqs,ngrid,ngrid).
        mapsize : float
            Size of the flat map the beam was projected onto. 
            Only used for internal calculations.
        """

        f = h5py.File(self.ft_file, "r") 
        mapsize = f['parameters']['mapsize'][0]
        Atilde = f['data'][self.pol][self.spw_range[0]:self.spw_range[1],:,:]
        f.close()

        return Atilde, mapsize

    def get_kgrid(self, bl_len, mapsize):
        """
        Computes the kperp-array the FT of the beam will be interpolated over.
        Must include all the kperp covered by the spectral window for the 
        baseline considered.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        mapsize : int
            Size of the flat map the beam was projected onto. 
            Only used for internal calculations.

        Returns
        ----------
        kgrid : array_like
            (kperp_x,kperp_y) grid corresponding to a given baseline.
            Two-dimensional.
        kperp_norm : array_like
            Array of kperp vector norms corresponding to kgrid.
            Two-dimensionsal.
            Computed as sqrt(kperp_x**2+kperp_y**2).

        """

        kp_centre=self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*bl_len
        dk = 2.*np.pi/self.cosmo.dRperp_dtheta(self.cosmo.f2z(self.freq_array.max()), little_h=self.little_h)/(2.*mapsize)
        kgrid = np.arange(kp_centre-0.020,kp_centre+0.020,step=dk)# np.arange(kmin,kmax+dk,step=dk)
        kperp_norm = np.sqrt(np.power(kgrid,2)[:, None] + np.power(kgrid,2))
        return kgrid, kperp_norm

    def kperp4bl_freq(self,freq,bl_len, ngrid, mapsize):    
        """
        Computes the range of kperp corresponding to a given
        baseline-freq pair. It will be assigned to the FT of the beam.

        Parameters
        ----------
        freq : float
            Frequency (in Hz) considered along the spectral window.
        bl_len : float
            Length of the baseline considered (in meters).
        ngrid : int
            Number of pixels in the FT beam array. 
            Internal use only. Do not modify.
        mapsize : float
            Size of the flat map the beam was projected onto.
            Internal use only. Do not modify.

        Returns
        ----------
        k : array_like  
            Array of k_perp values to match to the FT of the beam.

        """
        z = self.cosmo.f2z(freq)
        R = self.cosmo.DM(z, little_h=self.little_h) #Mpc
        q = np.fft.fftshift(np.fft.fftfreq(ngrid))*ngrid/(2.*mapsize)
        k = 2.*np.pi/R*(freq*bl_len/conversions.units.c-q)
        k = np.flip(k)
        return k

    def interpolate_FT_beam(self, bl_len, Atilde, mapsize):
        """
        Interpolate the FT of the beam on a regular (kperp,kperp) grid.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        Atilde : array_like
            Array made of the FT of the beam along the spectral window.
            Must have dimensions (Nfreqs, N, N).
        mapsize : float
            Size of the flat map the beam was projected onto.
            Internal use only. Do not modify.

        Returns
        ----------
        Atilde_cube : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        kperp_norm : array_like
            Norm of the kperp vectors throughout the grid.
            Has dimensions (N,N).

        """
        kgrid, kperp_norm = self.get_kgrid(bl_len, mapsize)

        ngrid = Atilde.shape[-1]
        Atilde_cube = np.zeros((kgrid.size,kgrid.size,self.Nfreqs))
        for i in range(self.Nfreqs):
            q = np.fft.fftshift(np.fft.fftfreq(ngrid))*ngrid/(2.*mapsize)
            k = self.kperp4bl_freq(self.freq_array[i],bl_len, ngrid=ngrid, mapsize = mapsize)
            A_real = interp2d(k,k,Atilde[i,:,:],bounds_error=False,fill_value=0.)
            Atilde_cube[:,:,i] = A_real(kgrid,kgrid) 

        return Atilde_cube, kperp_norm

    def take_freq_FT(self, Atilde_cube,delta_nu):
        """
        Take the Fourier transform along frequency of the beam.
        Applies taper before taking the FT if appropriate.

        Parameters
        ----------
        Atilde_cube : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        delta_nu : float
            Frequency resolution (Channel width) in Hz
            along the spectral window.

        Returns
        ----------
        fnu : array_like
            Fourier transform of the beam in sky plane and in frequency.
            Has dimensions (Nfreqs, N, N)
        """

        if self.taper is not None:
            tf = dspec.gen_window(self.taper, self.Nfreqs)
            Atilde_cube = Atilde_cube*tf[None,None,:]

        fnu = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Atilde_cube,axes=-1),axis=-1,norm='ortho')*delta_nu**0.5,axes=-1)

        return fnu 

    def get_wf_for_tau(self,tau,wf_array1,kperp_bins,kpara_bins):
        """
        Get the cylindrical window function for a given delay
        after binning on the sky plane.

        Parameters
        ----------
        tau : float
            Delay, in secs.      
        wf_array1: array_like
            Window function after cylindrical average on kperp plane.
            Dimensions: (nbins_kperp,nfreq).
        kperp_bins : array_like
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with self.little_h.
        kpara_bins : array_like
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning.
            Make sure the values are consistent with self.little_h.


        Returns
        ----------
        wf_array : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is kperp (kperp_bins defined as global variable).
            Axis 1 is kparallel (kpara_bins defined as global variable)
        kpara : array_like
            Values of kpara corresponding to the axis=2 of wf_array.
            Note: these values are weighted by their number of counts 
            in the cylindrical binning.
        
        """

        # read kperp bins
        kperp_bins = np.array(kperp_bins)
        nbins_kperp = kperp_bins.size
        dkperp = np.diff(kperp_bins).mean()
        kperp_range = np.arange(kperp_bins.min()-dkperp/2,kperp_bins.max()+dkperp,step=dkperp)
        # read kpara bins
        kpara_bins = np.array(kpara_bins)
        nbins_kpara = kpara_bins.size
        dkpara = np.diff(kpara_bins).mean()
        kpara_range = np.arange(kpara_bins.min()-dkpara/2,kpara_bins.max()+dkpara,step=dkpara)


        #### get kparallel grid
        alpha = self.cosmo.dRpara_df(self.avg_z, little_h=self.little_h, ghz=False)
        delta_nu = abs(self.freq_array[-1]-self.freq_array[0])/self.Nfreqs
        q = np.fft.fftshift(np.fft.fftfreq(self.Nfreqs),axes=-1)/delta_nu #unit 1: FT along theta

        wf_array = np.zeros((nbins_kperp,nbins_kpara))
        kpara = np.zeros(nbins_kpara)
        kpar_norm = np.abs(2.*np.pi/alpha*(q+tau))
        for j in range(nbins_kperp):
            for m in range(nbins_kpara):
                mask= (kpara_range[m]<=kpar_norm) & (kpar_norm<kpara_range[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_array[j,m]=np.mean(wf_array1[j,mask])
                    kpara[m] = np.mean(kpar_norm[mask])

        return kpara, wf_array

    def get_cylindrical_wf(self, bl_len, pol, Atilde, mapsize,
                            kperp_bins,kpara_bins):
        """
        Get the cylindrical window function i.e. in (kperp,kpara) space
        for a given baseline and polarisation, along the spectral window.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        pol : str
            Polarisation of the beam. 
            Can be chosen among 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'.
        kperp_bins : array_like
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with self.little_h.
        kpara_bins : array_like
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning.
            Make sure the values are consistent with self.little_h.

        Returns
        ----------
        wf_array : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is the array of delays considered (self.dly_array).
            Axis 1 is kperp (kperp_bins defined as global variable).
            Axis 2 is kparallel (kpara_bins defined as global variable).
        kperp : array_like
            Values of kperp corresponding to the axis=1 of wf_array.
            Note: these values are weighted by their number of counts 
            in the cylindrical binning.
        kpara : array_like
            Values of kpara corresponding to the axis=2 of wf_array.
            Note: these values are weighted by their number of counts 
            in the cylindrical binning.

        """

        # read kperp bins
        kperp_bins = np.array(kperp_bins)
        nbins_kperp = kperp_bins.size
        dkperp = np.diff(kperp_bins).mean()
        kperp_range = np.arange(kperp_bins.min()-dkperp/2,kperp_bins.max()+dkperp,step=dkperp)
        # read kpara bins
        kpara_bins = np.array(kpara_bins)
        nbins_kpara = kpara_bins.size
        dkpara = np.diff(kpara_bins).mean()
        kpara_range = np.arange(kpara_bins.min()-dkpara/2,kpara_bins.max()+dkpara,step=dkpara)


        t0 = time.time()
        Atilde_cube, kperp_norm = self.interpolate_FT_beam(bl_len, Atilde, mapsize)
        t1 = time.time()
        delta_nu = abs(self.freq_array[-1]-self.freq_array[0])/self.Nfreqs
        fnu = self.take_freq_FT(Atilde_cube,delta_nu)
        t2 = time.time()
        ##### cylindrical average

        # on sky plane
        wf_array1 = np.zeros((nbins_kperp,self.Nfreqs))
        kperp = np.zeros(nbins_kperp)
        for i in range(self.Nfreqs):
            for m in range(nbins_kperp):
                mask= (kperp_range[m]<=kperp_norm) & (kperp_norm<kperp_range[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_array1[m,i]=np.mean(np.abs(fnu[mask,i])**2)
                    kperp[m] = np.mean(kperp_norm[mask])
        t3 = time.time()
        # in frequency direction    
        # binning
        wf_array = np.zeros((self.Nfreqs,nbins_kperp,nbins_kpara))
        # kpara, wf0 = self.get_wf_for_tau(self.dly_array[0],wf_array1)
        # for it,tau in enumerate(self.dly_array[:self.Nfreqs//2+1]):
            # wf_array[it,:,:]=np.roll(wf0,-it,axis=1)
        for it,tau in enumerate(self.dly_array[:self.Nfreqs//2+1]):
            kpara, wf_array[it,:,:] = self.get_wf_for_tau(tau,wf_array1,kperp_bins,kpara_bins)
        #fill by symmetry for tau = -tau
        if (self.Nfreqs%2==0):
            wf_array[self.Nfreqs//2+1:,:,:]=np.flip(wf_array,axis=0)[self.Nfreqs//2:-1]
        else:
            wf_array[self.Nfreqs//2+1:,:,:]=np.flip(wf_array,axis=0)[self.Nfreqs//2+1:]

        # ### normalisation of window functions
        wf_array /= np.sum(wf_array,axis=(1,2))[:,None,None]
        t4 = time.time()
        print(t1-t0,t2-t1,t3-t2,t4-t3)
        return kperp, kpara, wf_array

    def get_spherical_wf(self,bl_groups,bl_lens,spw_range,pol,
                            kbins, kperp_bins=[], kpara_bins=[]):
        """
        Get spherical window functions for a set of baselines, polarisation,
        along a given spectral range, and for a set of kbins used for averaging.

        Parameters
        ----------
        bl_groups : embedded lists.
            List of groups of baselines gathered by lengths
            (can be redundant groups from utils.get_reds).
        bl_lens : list.
            List of lengths corresponding to each group
            (can be redundant groups from utils.get_reds).
            Must have same length as bl_groups.
            
        spw_range : tuple of ints
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA bandwidth).
        pol : str
            Can be pseudo-Stokes or power: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
        kbins : array_like
            1D float array of ascending |k| bin centers in [h] Mpc^-1 units.
            Using for spherical binning.
            Make sure the values are consistent with self.little_h.
        kperp_bins : array_like
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with self.little_h.
        kpara_bins : array_like
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning.
            Make sure the values are consistent with self.little_h.

        """

        assert len(bl_groups)==len(bl_lens), "bl_groups and bl_lens must have same length"
        nbls = len(bl_groups)
        bl_lens = np.array(bl_lens)
        red_nb = np.array([len(l) for l in bl_groups])

        if not (isinstance(spw_range[0],int) and isinstance(spw_range[1],int)):
            raise Warning('spw indices given are not integers... taking their floor value')
            spw_range = (int(np.floor(spw_range[0])),int(np.floor(spw_range[1])))
        assert min(spw_range)>=0 and max(spw_range)<chan_nb, \
                "spw_range must be integers within the HERA frequency channels"
        assert spw_range[1]-spw_range[0]>0, "Require non-zero spectral range."
        self.set_spw_range(spw_range)
        
        # k-bins for spherical binning
        assert len(kbins)>1, "must feed array of k bins for spherical averasge"                                                  
        kbins = np.array(kbins)
        nbinsk = kbins.size
        dk = np.diff(kbins).mean()
        krange = np.arange(kbins.min()-dk/2,kbins.max()+dk,step=dk)

        assert pol in ['pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'], \
                "Wrong polarisation string."
        self.set_polarisation(pol)

        #k-bins for cylindrical binning
        if np.size(kperp_bins)==0 or kperp_bins is None:
            dk_perp = np.diff(self.get_kgrid(np.min(lens), mapsize)[1]).mean()*5
            kperp_max = cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*np.max(lens)*np.sqrt(2)+ 2.*dk_perp
            kperp_range = np.arange(dk_perp,kperp_max,step=dk_perp)
            nbins_kperp = kperp_range.size -1
            kperp_bins = (kperp_range[1:]+kperp_range[:-1])/2
        else:
            kperp_bins = np.array(kperp_bins)
            nbins_kperp = kperp_bins.size
            dkperp = np.diff(kperp_bins).mean()
            kperp_range = np.arange(kperp_bins.min()-dkperp/2,kperp_bins.max()+dkperp,step=dkperp)

        if np.size(kpara_bins)==0 or kpara_bins is None:
            dk_para = cosmo.tau_to_kpara(self.avg_z,little_h=self.little_h)/(abs(self.freq_array[-1]-self.freq_array[0]))
            kpara_max = cosmo.tau_to_kpara(self.avg_z,little_h=self.little_h)*abs(self.dly_array).max()+2.*dk_para
            kpara_range = np.arange(dk_para,kpara_max,step=dk_para)
            nbins_kpara = kpara_range.size -1
            kpara_bins = (kpara_range[1:]+kpara_range[:-1])/2
        else:                                              
            kpara_bins = np.array(kpara_bins)
            nbins_kpara = kpara_bins.size
            dkpara = np.diff(kpara_bins).mean()
            kpara_range = np.arange(kpara_bins.min()-dkpara/2,kpara_bins.max()+dkpara,step=dkpara)

        ktot = np.sqrt(kperp_bins[:,None]**2+kpara_bins**2)
        if (nbins_kperp>200) or (nbins_kpara>200):
            raise Warning('Large number of kperp/kpara bins. Risk of overresolving and slow computing.')

        # get FT of the beam from file
        Atilde, mapsize = self.get_FT()
        # get cylindrical window functions for each baseline length considered
        # as a function of (kperp, kpara)
        # the kperp and kpara bins are given as global parameters
        kperp_array, kpar_array = np.zeros((nbls,nbins_kperp)),np.zeros((nbls,nbins_kpara))
        wf_array = np.zeros((nbls,self.Nfreqs,nbins_kperp,nbins_kpara))
        for ib in range(nbls):
            if self.verbose: print('Computing for bl %i of %i...' %(ib+1,nbls))
            kperp_array[ib,:], kpar_array[ib,:], wf_array[ib,:,:,:] = self.get_cylindrical_wf(bl_lens[ib],pol,
                                                                        Atilde, mapsize, 
                                                                        kperp_bins, kpara_bins)

        # construct array giving the k probed by each baseline-tau pair
        kperps = bl_lens * self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) / np.sqrt(2.)
        kparas = self.dly_array * self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h) 
        kmags = np.sqrt(kperps[:,None]**2+kparas**2)

        wf_spherical = np.zeros((nbinsk,nbinsk))
        count = np.zeros(nbinsk,dtype=int)
        for m1 in range(nbinsk):
            mask2 = (krange[m1]<=kmags) & (kmags<krange[m1+1]).astype(int)
            mask2 = mask2*red_nb[:,None] #account for redundancy
            count[m1] = np.sum(mask2) 
            wf_temp = np.sum(wf_array*mask2[:,:,None,None],axis=(0,1))/np.sum(mask2)
            for m in range(nbinsk):
                mask= (krange[m]<=ktot) & (ktot<krange[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_spherical[m1,m]=np.mean(wf_temp[mask])
            wf_spherical[m1,:]/=np.sum(wf_spherical[m1,:])

        return wf_spherical, count

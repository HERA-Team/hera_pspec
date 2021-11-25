from pyuvdata import utils as uvutils
import uvtools.dspec as dspec
import h5py
import warnings
import numpy as np
import sys, os
from scipy.interpolate import interp2d

from . import conversions, noise, version, pspecbeam, grouping, utils, uvpspec_utils as uvputils

HERA_bw=np.linspace(1.,2.,1024,endpoint=False)*1e8

#k-bins

kpara_max, dk_para = 2., 0.043/2
kpara_range = np.arange(dk_para,kpara_max,step=dk_para)
nbins_kpara = kpara_range.size -1
kpara_bins = (kpara_range[1:]+kpara_range[:-1])/2

kperp_max, dk_perp = 0.11, .5e-3
kperp_range = np.arange(dk_perp,kperp_max,step=dk_perp)
nbins_kperp = kperp_range.size -1
kperp_bins = (kperp_range[1:]+kperp_range[:-1])/2

ktot = np.sqrt(kperp_bins[:,None]**2+kpara_bins**2)

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

        assert len(spw_range)==2, "spw_range must be fed as a tuple of frequency indices between 0 and 1024"

        self.spw_range = tuple(spw_range)
        self.freq_array = HERA_bw[spw_range[0]:spw_range[-1]]
        self.Nfreqs = len(self.freq_array)
        self.dly_array = utils.get_delays(self.freq_array,n_dlys=len(self.freq_array))
        self.avg_nu = np.mean(self.freq_array)
        self.avg_z = self.cosmo.f2z(self.avg_nu)

    def set_polarisation(self,pol):

        self.pol = pol 

    def get_FT(self):

        f = h5py.File(self.ft_file, "r") 
        mapsize = f['parameters']['mapsize'][0]
        Atilde = f['data'][self.pol][self.spw_range[0]:self.spw_range[1],:,:]
        f.close()

        return Atilde, mapsize

    def get_kgrid(self, bl_len, mapsize):

        kp_centre=self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*bl_len
        dk = 2.*np.pi/self.cosmo.dRperp_dtheta(self.cosmo.f2z(self.freq_array.max()), little_h=self.little_h)/(2.*mapsize)
        kgrid = np.arange(kp_centre-0.020,kp_centre+0.020,step=dk)# np.arange(kmin,kmax+dk,step=dk)
        kperp_norm = np.sqrt(np.power(kgrid,2)[:, None] + np.power(kgrid,2))
        return kgrid, kperp_norm

    def kperp4bl_freq(self,freq,bl_len, ngrid, mapsize):    

        z = self.cosmo.f2z(freq)
        R = self.cosmo.DM(z, little_h=self.little_h) #Mpc
        q = np.fft.fftshift(np.fft.fftfreq(ngrid))*ngrid/(2.*mapsize)
        k = 2.*np.pi/R*(freq*bl_len/conversions.units.c-q)
        k = np.flip(k)
        return k

    def interpolate_FT_beam(self, bl_len, Atilde, mapsize):

        kgrid, kperp_norm = self.get_kgrid(bl_len, mapsize)

        ngrid = Atilde.shape[-1]
        Atilde_cube = np.zeros((kgrid.size,kgrid.size,self.Nfreqs))
        for i in range(self.Nfreqs):
            q = np.fft.fftshift(np.fft.fftfreq(ngrid))*ngrid/(2.*mapsize)
            k = self.kperp4bl_freq(self.freq_array[i],bl_len, ngrid=ngrid, mapsize = mapsize)
            A_real = interp2d(k,k,Atilde[i,:,:],bounds_error=False,fill_value=0.)
            Atilde_cube[:,:,i] = A_real(kgrid,kgrid) 

        return Atilde_cube, kperp_norm

    def take_freq_FT(self, Atilde_cube):

        if self.taper is not None:
            tf = dspec.gen_window(self.taper, self.Nfreqs)
            Atilde_cube = Atilde_cube*tf[None,None,:]

        delta_nu = abs(self.freq_array[-1]-self.freq_array[0])/self.Nfreqs
        fnu = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Atilde_cube,axes=-1),axis=-1,norm='ortho')*delta_nu**0.5,axes=-1)

        return fnu 

    def get_cylindrical_wf(self, bl_len, pol):

        Atilde, mapsize = self.get_FT()
        Atilde_cube, kperp_norm = self.interpolate_FT_beam(bl_len, Atilde, mapsize)
        fnu = self.take_freq_FT(Atilde_cube)

        ##### cylindrical average
        if self.verbose: print('Taking cylindrical average...')

        # on sky plane
        wf_array1 = np.zeros((nbins_kperp,self.Nfreqs))
        kperp, count1 = np.zeros(nbins_kperp), np.zeros(nbins_kperp)
        for i in range(self.Nfreqs):
            for m in range(nbins_kperp):
                mask= (kperp_range[m]<=kperp_norm) & (kperp_norm<kperp_range[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_array1[m,i]=np.mean(np.abs(fnu[mask,i])**2)
                    count1[m] = np.sum(mask)
                    kperp[m] = np.mean(kperp_norm[mask])

        # in frequency direction    
        #### get kparallel grid
        alpha = self.cosmo.dRpara_df(self.avg_z, little_h=self.little_h, ghz=False)
        q = np.fft.fftshift(np.fft.fftfreq(self.Nfreqs),axes=-1)/delta_nu #unit 1: FT along theta
        # binning
        self.wf_array = np.zeros((self.Nfreqs,nbins_kperp,nbins_kpara))
        kpara, count2 = np.zeros(nbins_kpara), np.zeros(nbins_kpara)
        for it,tau in enumerate(self.dly_array[:self.Nfreqs//2+1]):
            kpar_norm = np.abs(2.*np.pi/alpha*(q+tau))
            for j in range(nbins_kperp):
                for m in range(nbins_kpara):
                    mask= (kpara_range[m]<=kpar_norm) & (kpar_norm<kpara_range[m+1])
                    if np.any(mask): #cannot compute mean if zero elements
                        self.wf_array[it,j,m]=np.mean(wf_array1[j,mask])
                        count2[m] = np.sum(mask)
                        kpara[m] = np.mean(kpar_norm[mask])
        #fill by symmetry for tau = -tau
        if (self.Nfreqs%2==0):
            self.wf_array[self.Nfreqs//2+1:,:,:]=np.flip(self.wf_array,axis=0)[self.Nfreqs//2:-1]
        else:
            self.wf_array[self.Nfreqs//2+1:,:,:]=np.flip(self.wf_array,axis=0)[self.Nfreqs//2+1:]

        # ### normalisation of window functions
        self.wf_array /= np.sum(self.wf_array,axis=(1,2))[:,None,None]

        return kperp, kpara, self.wf_array

    def get_spherical_wf(self,lens,spw_range,pol,kbins,bl_tol=0.1):

        kbins = np.array(kbins)
        nbinsk = kbins.size
        dk = np.diff(kbins).mean()
        krange = np.arange(kbins.min()-dk/2,kbins.max()+dk,step=dk)

        self.set_spw_range(spw_range)
        self.set_polarisation(pol)

        lens = np.round(lens,decimals=abs(int(np.log10(bl_tol))))
        bl_lens, red_nb = np.unique(lens,return_counts=True)
        nbls = bl_lens.size

        kperp_array, kpar_array = np.zeros((nbls,nbins_kperp)),np.zeros((nbls,nbins_kpara))
        wf_array = np.zeros((nbls,self.Nfreqs,nbins_kperp,nbins_kpara))
        for ib, bl_len in enumerate(bl_lens):
            if self.verbose: sys.stdout.write('\r Computing for bl %i of %i...' %(ib,nbls))
            kperp_array[ib,:], kpar_array[ib,:], wf_array[ib,:,:,:] = self.get_cylindrical_wf(bl_len,pol)

        ktot_instru = np.zeros((nbls,self.Nfreqs))
        for ib in range(nbls):
            for it in range(self.Nfreqs):
                kp1 = bl_lens[ib] * self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h)
                kp2 = self.dly_array[it] * self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h)
                kp1 = kp1/np.sqrt(2.)
                ktot_instru[ib,it] = np.sqrt(kp1**2+kp2**2)


        wf_spherical = np.zeros((nbinsk,nbinsk))
        count = np.zeros(nbinsk,dtype=int)
        for m1 in range(nbinsk):
            mask2 = (krange[m1]<=ktot_instru) & (ktot_instru<krange[m1+1]).astype(int)
            mask2 = mask2*red_nb[:,None] #account for redundancy
            count[m1] = np.sum(mask2) 
            wf_temp = np.sum(wf_array*mask2[:,:,None,None],axis=(0,1))/np.sum(mask2)
            for m in range(nbinsk):
                mask= (krange[m]<=ktot) & (ktot<krange[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_spherical[m1,m]=np.mean(wf_temp[mask])
            wf_spherical[m1,:]/=np.sum(wf_spherical[m1,:])

        return wf_spherical, count

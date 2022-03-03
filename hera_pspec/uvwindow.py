from pyuvdata import utils as uvutils
import uvtools.dspec as dspec
import h5py
import warnings
import numpy as np
import sys, os, time
from scipy.interpolate import interp2d
from pyuvdata import UVBeam, UVData 

from . import conversions, noise, version, pspecbeam, grouping, utils, uvpspec_utils as uvputils

class UVWindow(object):
    """
    An object for storing window functions copmuted without the delay approximation.
    """

    def __init__(self, ftbeam='', uvdata='',taper='none', 
                    cosmo=None, little_h=True,verbose=False):
        """
        Class for UVWindow objects. Provides get_spherical_wf() and
        get_cylindrical_wf() to obtain accurate window functions for a 
        given set of baselines and spectral range.

        Parameters
        ----------
        ftbeam : str
            Definition of the beam Fourier transform to be used.
            Options include;
                - Root name of the file to use, without the polarisation
                Ex : FT_beam_HERA_dipole (+ path)
                - '' for computation from beam simulations (slow)
        uvdata : str, optional
            Data file or UVDats object to be used to read baselines.
            Not used if set to ''.
            Must have all HERA frequencies in meta_data.
        taper : str
            Type of data tapering applied along bandwidth.
            If 'none', no taper is applied. 
            See uvtools.dspec.gen_window for options.
        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.
        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
        
        """

        # Summary attributes

        # initialises other attributes
        if cosmo is None: cosmo = conversions.Cosmo_Conversions()
        self.cosmo = cosmo

        assert isinstance(little_h, bool), "little_h must be boolean"
        self.little_h = little_h

        assert isinstance(verbose, bool), "verbose must be boolean"
        self.verbose = verbose

        assert isinstance(taper, str), "taper must be string of taper name"
        self.taper = taper

        # check if path the FT beam file has been given
        if isinstance(ftbeam, str):
            if (len(ftbeam)<1):
                raise_warning('No input FT beam, will compute all window functions from scratch... Will take a few hours.',
                                verbose=self.verbose)
                ##### to be coded up
            else:  
                self.ft_file = ftbeam
        else:
            raise ValueError('Wrong ftbeam input. See docstring.')
        self.mapsize = None # Size of the flat map the beam was projected onto. 
                            # Only used for internal calculations.

        # if data file is used, initialises related arguments.   
        if isinstance(uvdata, str):
            if (uvdata == ''):
                self.is_uvdata = False  
            else: 
                self.is_uvdata = True
                self.uvdata = UVData()
                self.uvdata.read(uvdata, read_data=False)
        else:
            self.is_uvdata = True
            self.uvdata = uvdata

        # Analysis-related attributes. 
        # Will be set with set_spw_range andd set_spw_parameters
        # once one of the get_wf functions is called.
        self.freq_array = None
        self.Nfreqs = None
        self.dly_array = None
        self.spw_range = None
        self.avg_nu = None
        self.avg_z = None
        self.pol = None

        # Initialise empty arrays for values potentially
        # stored later (see get_spherical_wf)
        self.kbins = []
        self.kpara_bins = []
        self.kperp_bins = []
        self.cyl_wf = []
        self.bl_lens = []
        self.bl_weights = []

    def get_bandwidth(self,file):
        """
        Read FT file to extract bandwidth it was computed along.

        Parameters
        ----------
        file : str
            Path to FT beam file.
            Root name of the file to use, without the polarisation
                Ex : FT_beam_HERA_dipole (+ path)
            If '', then the object ft_file attribute is used.

        Returns
        ----------
        bandwidth : array of floats
            List of frequencies covered by the instrument, in Hz.
        """

        assert self.pol is not None, "Need to set polarisation first."

        filename = '%s_%s.hdf5' %(file,self.pol)
        assert os.path.isfile(filename), "Cannot find FT beam file: %s." %filename
        f = h5py.File(filename, "r") 
        HERA_bw = f['freq'][...]
        f.close()

        assert len(HERA_bw)>1, "Error reading file, empty bandwidth."

        return HERA_bw

    def set_taper(self, taper, clear_cache = True):
        """
        Set data tapering type.

        Parameters
        ----------
        taper : str
            Type of data tapering. See uvtools.dspec.gen_window for options.
        clear_cache : bool, optional
            Clear saved window functions if existing
            (in case they were computed with another taper)
        """
        self.taper = taper
        if (len(self.cyl_wf)>0):
            raise_warning('New taper set but window functions have not been updated.')
            if clear_cache:
                self.clear_cache(clear_cyl_bins=True)


    def set_spw_range(self,spw_range):
        """
        Sets the spectral range considered to compute the window functions.

        Parameters
        ----------
        spw_range : tuple or array
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA bandwidth).

        """

        spw_range = np.array(spw_range,dtype=int)
        assert spw_range.size==2, "spw_range must be fed as a tuple of frequency indices."
        assert spw_range[1]-spw_range[0]>0, "Require non-zero spectral range."
        self.spw_range = tuple(spw_range)

        if self.is_uvdata:
            # Set spw parameters such as frequency range and average redshift.
            self.set_spw_parameters(self.uvdata.freq_array[0])
        else:
            HERA_bw = self.get_bandwidth(file=self.ft_file)
            self.set_spw_parameters(HERA_bw)

    def set_freq_range(self,freq_array):
        """
        Uses the frequencies of the spectral range given as innput
        to set spectral window parameters.

        Parameters
        ----------
        fmin : array of floats
            Array of frequencies (in Hz) to be considered to compute the window functions.
            Similar to uvp.freq_array.

        """

        freq_array = np.array(freq_array)
        assert freq_array.size > 1, "freq_array must be a list or array of length > 1"

        if self.is_uvdata:
            bandwidth = self.uvdata.freq_array[0]
        else:
            bandwidth = self.get_bandwidth(file=self.ft_file)

        # check if freq_array is in the instrument bandwidth
        assert (np.min(abs(bandwidth-np.min(freq_array)))<1e6) and (np.min(abs(bandwidth-np.max(freq_array)))<1e6), \
                "input freq_array not in instrument bandwidth"
        self.spw_range = (np.argmin(abs(bandwidth-np.min(freq_array))), np.argmin(abs(bandwidth-np.max(freq_array)))+1)
        self.set_spw_parameters(bandwidth)

    def set_spw_parameters(self,bandwidth):
        """
        Sets the parameters related to the spectral window considered.

        Parameters
        ----------
        bandwidth : array of floats
            List of frequencies covered by the instrument, in Hz.

        """

        bandwidth = np.array(bandwidth)
        assert bandwidth.size>1, "Must feed bandwidth as an array of frequencies."
        assert min(self.spw_range)>=0 and max(self.spw_range)<len(bandwidth), \
                "spw_range must be integers within the given bandwith."

        self.freq_array = bandwidth[self.spw_range[0]:self.spw_range[-1]]
        self.Nfreqs = len(self.freq_array)
        self.dly_array = utils.get_delays(self.freq_array, n_dlys=len(self.freq_array))
        self.avg_nu = np.mean(self.freq_array)
        self.avg_z = self.cosmo.f2z(self.avg_nu)        
        if self.is_uvdata:
            assert self.uvdata.Nfreqs==len(bandwidth), "Data file does share bandwidth with FT beam file."

    def set_polarisation(self,pol,x_orientation=None):
        """
        Sets the polarisation considered for the beam to compute the window functions.

        Parameters
        ----------
        pol : str or int
            Can be pseudo-Stokes or power: 
             in str form: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
             in number form: 1, 2, 4, 3, -5, -6, -7, -8
        """

        if isinstance(pol, str):
            assert pol in ['pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'], "Wrong polarisation"
        elif isinstance(pol,int):
            assert pol in [1, 2, 4, 3, -5, -6, -7, -8], "Wrong polarisation"
            if self.is_uvdata: assert pol in self.uvdata.polarization_array, "Polarisation not in data file."
            # convert pol number to str according to AIPS Memo 117.
            pol = uvutils.polnum2str(pol,x_orientation=x_orientation)
        else:
            raise TypeError("Must feed pol as str or int.")
        self.pol = pol 


    def set_bl_lens(self,bl_lens):
        """
        Sets the list of baseline lengths used to compute the window functions.

        Parameters
        ----------
        bl_lens : list of floats
            List of baseline lengths.
        """
        self.bl_lens = bl_lens


    def get_FT(self,file='',return_FT=True):
        """
        Loads the Fourier transform (FT) of the beam from different attributes: 
            - self.pol
            - self.spw_range
        and initialises freqncy array from the latter.
        Note that the array has no physical coordinates, they will be later 
        attributed by kperp4bl_freq.

        Parameters
        ----------
        file : str
            Path to FT beam file.
            Root name of the file to use, without the polarisation
                Ex : FT_beam_HERA_dipole (+ path)
            If '', then the object ft_file attribute is used.
        return_FT : bool
            If True, returns the FT of the beam along the spectral window.
            If False, only initialises FT properties (useful to call 
            get_kperp_bins).

        Returns
        ----------
        FT_beam : array_like
            Real part of the Fourier transform of the beam along the spectral
            window considered. Has dimensions (Nfreqs,ngrid,ngrid).
        """

        if len(file)==0:
            file = self.ft_file
        assert self.pol is not None, "Need to set polarisation first."
        assert self.spw_range is not None, "Need to set spectral window first."

        filename = '%s_%s.hdf5' %(file,self.pol)
        assert os.path.isfile(filename), "Cannot find FT beam file: %s." %filename
        f = h5py.File(filename, "r") 
        self.mapsize = f['mapsize'][0] 
        FT_beam = f['FT_beam'][self.spw_range[0]:self.spw_range[1],:,:]
        HERA_bw = f['freq'][...]
        f.close()

        # Set spw parameters such as frequency range and average redshift.
        self.set_spw_parameters(HERA_bw)

        if return_FT:
            return FT_beam

    def get_kgrid(self, bl_len, width=0.020):
        """
        Computes the kperp-array the FT of the beam will be interpolated over.
        Must include all the kperp covered by the spectral window for the 
        baseline considered.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        width : float
            Distance between central kperp for given bl_len
            and edge of the kgrid.
            Increasing it will slow down the computation.

        Returns
        ----------
        kgrid : array_like
            (kperp_x) grid corresponding to a given baseline.
            One-dimensional.
        kperp_norm : array_like
            Array of kperp vector norms corresponding to kgrid.
            Two-dimensionsal.
            Computed as sqrt(kperp_x**2+kperp_y**2).

        """

        assert self.mapsize is not None, "Need to set spw and FT parameters with get_FT()."
        # central kperp for given baseline (i.e. 2pi*b*nu/c/R) 
        kp_centre=self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*bl_len
        # spacing of the numerical Fourier grid, in cosmological units
        dk = 2.*np.pi/self.cosmo.dRperp_dtheta(self.cosmo.f2z(self.freq_array.max()), little_h=self.little_h)/(2.*self.mapsize)
        assert width>dk, 'Change width to resolve full window function (dk=%.2e).' %dk
        # defines kgrid (kperp_x).
        kgrid = np.arange(kp_centre-width,kp_centre+width,step=dk)
        # array of kperp norms.
        kperp_norm = np.sqrt(np.power(kgrid,2)[:, None] + np.power(kgrid,2))

        return kgrid, kperp_norm

    def kperp4bl_freq(self,freq,bl_len, ngrid):    
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

        Returns
        ----------
        k : array_like  
            Array of k_perp values to match to the FT of the beam.

        """

        assert self.mapsize is not None, "Need to set spw and FT parameters with get_FT()."
        assert (freq<=self.freq_array.max()) and (freq>=self.freq_array.min()), "Choose \
                frequency within spectral window."
        assert (freq/1e6>=1.), "Frequency must be given in Hz."

        z = self.cosmo.f2z(freq)
        R = self.cosmo.DM(z, little_h=self.little_h) #Mpc
        # Fourier dual of sky angle theta
        q = np.fft.fftshift(np.fft.fftfreq(ngrid))*ngrid/(2.*self.mapsize)
        k = 2.*np.pi/R*(freq*bl_len/conversions.units.c/np.sqrt(2.)-q)
        k = np.flip(k)

        return k

    def interpolate_FT_beam(self, bl_len, FT_beam):
        """
        Interpolate the FT of the beam on a regular (kperp,kperp) grid.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        FT_beam : array_like
            Array made of the FT of the beam along the spectral window.
            Must have dimensions (Nfreqs, N, N).

        Returns
        ----------
        interp_FT_beam : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        kperp_norm : array_like
            Norm of the kperp vectors throughout the grid.
            Has dimensions (N,N).

        """

        FT_beam = np.array(FT_beam)
        assert FT_beam.ndim==3, "FT_beam must be dimension 3."
        assert FT_beam.shape[0]==self.Nfreqs, "FT_beam must have shape (Nfreqs,N,N)"
        assert FT_beam.shape[2]==FT_beam.shape[1], "FT beam must be square in sky plane"

        # regular kperp_x grid the FT of the beam will be interpolated over.
        # kperp_norm is the corresponding total kperp:
        # kperp = sqrt(kperp_x**2 + kperp_y**2)
        kgrid, kperp_norm = self.get_kgrid(bl_len)

        # assign FT of beam to appropriate kperp grid for each frequency
        # and given baseline length
        ngrid = FT_beam.shape[-1]
        interp_FT_beam = np.zeros((kgrid.size,kgrid.size,self.Nfreqs))
        for i in range(self.Nfreqs):
            # kperp values over frequency array for bl_len
            k = self.kperp4bl_freq(self.freq_array[i],bl_len, ngrid=ngrid)
            # interpolate the FT beam over values onto regular kgrid
            A_real = interp2d(k,k,FT_beam[i,:,:],bounds_error=False,fill_value=0.)
            interp_FT_beam[:,:,i] = A_real(kgrid,kgrid) 

        return interp_FT_beam, kperp_norm

    def take_freq_FT(self, interp_FT_beam, delta_nu, taper=''):
        """
        Take the Fourier transform along frequency of the beam.
        Applies taper before taking the FT if appropriate.

        Parameters
        ----------
        interp_FT_beam : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        delta_nu : float
            Frequency resolution (Channel width) in Hz
            along the spectral window.
        taper : str
            Type of data tapering applied along frequency direction. 
            See uvtools.dspec.gen_window for options.
            If 'none', no taper is used.
            If '', self.taper is used.

        Returns
        ----------
        fnu : array_like
            Fourier transform of the beam in sky plane and in frequency.
            Has dimensions (Nfreqs, N, N)
        """

        interp_FT_beam = np.array(interp_FT_beam)
        assert interp_FT_beam.ndim==3, "interp_FT_beam must be dimension 3."
        assert interp_FT_beam.shape[-1]==self.Nfreqs, "interp_FT_beam must have shape (N,N,Nfreqs)"


        # apply taper along frequency direction
        if not (taper =='none'):
            if len(taper)==0: 
                taper = self.taper    
            else:
                # set taper to new value
                self.set_taper(taper)                 
            tf = dspec.gen_window(taper, self.Nfreqs)
            interp_FT_beam = interp_FT_beam*tf[None,None,:]
        else: 
            # set taper to new value
            self.set_taper(taper)  

        # take numerical FT along frequency axis
        # normalise to appropriate units and recentre
        fnu = np.fft.fftshift(np.fft.fft(np.fft.fftshift(interp_FT_beam,axes=-1),axis=-1,norm='ortho')*delta_nu**0.5,axes=-1)

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
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is kperp (kperp_bins defined as global variable).
            Axis 1 is kparallel (kpara_bins defined as global variable)
        kpara : array_like
            Values of kpara corresponding to the axis=2 of cyl_wf.
            Note: these values are weighted by their number of counts 
            in the cylindrical binning.
        
        """

        # read kperp bins and find bin_edges
        kperp_bins = np.array(kperp_bins)
        nbins_kperp = kperp_bins.size
        dk_perp = np.diff(kperp_bins).mean()
        kperp_bin_edges = np.arange(kperp_bins.min()-dk_perp/2,kperp_bins.max()+dk_perp,step=dk_perp)
        # read kpara bins
        kpara_bins = np.array(kpara_bins)
        nbins_kpara = kpara_bins.size
        dk_para = np.diff(kpara_bins).mean()
        kpara_bin_edges = np.arange(kpara_bins.min()-dk_para/2,kpara_bins.max()+dk_para,step=dk_para)


        #### get kparallel grid
        # conversion factor to cosmological units
        alpha = self.cosmo.dRpara_df(self.avg_z, little_h=self.little_h, ghz=False)
        # frequency resolution
        delta_nu = abs(self.freq_array[-1]-self.freq_array[0])/self.Nfreqs
        # Fourier dual of frequency
        eta = np.fft.fftshift(np.fft.fftfreq(self.Nfreqs),axes=-1)/delta_nu #unit 1: FT along theta
        # construct array of |kpara| values for given delay tau
        kpar_norm = np.abs(2.*np.pi/alpha*(eta+tau))

        # perform binning along k_parallel
        cyl_wf = np.zeros((nbins_kperp,nbins_kpara))
        kpara = np.zeros(nbins_kpara)
        for j in range(nbins_kperp):
            for m in range(nbins_kpara):
                mask= (kpara_bin_edges[m]<=kpar_norm) & (kpar_norm<kpara_bin_edges[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    cyl_wf[j,m]=np.mean(wf_array1[j,mask])
                    kpara[m] = np.mean(kpar_norm[mask])

        return kpara, cyl_wf


    def get_kperp_bins(self,bl_lens):

        """
        Get spherical k_perp bins for a given set of baseline lengths,
        making sure all values probed by bl_lens are included and 
        there is no over-sampling.

        Parameters
        ----------
        bl_lens : list, optional.
            List of baseline lengths.
            Can be only one value.

        Returns
        ----------
        kperp_bins : array.
            Array of kperp bins to use.
    
        """

        # FT beam file must have been read to call get_kgrid
        assert self.mapsize is not None, "Need to set FT parameters with get_FT()."

        bl_lens = np.array(bl_lens)
        assert bl_lens.size>0, "get_kperp_bins() requires array of baseline lengths."
        
        dk_perp = np.diff(self.get_kgrid(np.min(bl_lens))[1]).mean()*5
        kperp_max = self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*np.max(bl_lens) + 10.*dk_perp
        kperp_bin_edges = np.arange(dk_perp,kperp_max,step=dk_perp)
        kperp_bins = (kperp_bin_edges[1:]+kperp_bin_edges[:-1])/2
        nbins_kperp = kperp_bins.size        
        if (nbins_kperp>200):
            raise_warning('get_kperp_bins: Large number of kperp/kpara bins. Risk of overresolving and slow computing.',
                            verbose=self.verbose)

        return kperp_bins

    def get_kpara_bins(self, freq_array, little_h=True, cosmo=None):


        """
        Get spherical k_para bins for a given spectra window initialisaed
        with set_spw_range and set_spw_parameters, making sure all values 
        probed by freq array are included and there is no over-sampling

        Parameters
        ----------
        freq_array : array 
            List of frequencies you want to compute window functions over.
            In Hz.
        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc.
        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.

        Returns
        ----------
        kpara_bins : array.
            Array of k_parallel bins to use.

    
        """

        if cosmo is None: cosmo = conversions.Cosmo_Conversions()

        freq_array = np.array(freq_array)
        assert freq_array.size>1, "Must feed list of frequencies."

        dly_array = utils.get_delays(freq_array,n_dlys=len(freq_array))
        avg_z = cosmo.f2z(np.mean(freq_array))                    

        # define default kperp bins,
        dk_para = cosmo.tau_to_kpara(avg_z,little_h=little_h)/(abs(freq_array[-1]-freq_array[0]))
        kpara_max = cosmo.tau_to_kpara(avg_z,little_h=little_h)*abs(dly_array).max()+10.*dk_para
        kpara_bin_edges = np.arange(dk_para,kpara_max,step=dk_para)
        kpara_bins = (kpara_bin_edges[1:]+kpara_bin_edges[:-1])/2
        nbins_kpara = kpara_bins.size

        if (nbins_kpara>200):
            raise_warning('get_kpara_bins: Large number of kperp/kpara bins. Risk of overresolving and slow computing.',
                            verbose=self.verbose)

        return kpara_bins

    def get_cylindrical_wf(self, bl_len, FT_beam=[],
                            kperp_bins=[],kpara_bins=[],
                            return_bins='none'):
        """
        Get the cylindrical window function i.e. in (kperp,kpara) space
        for a given baseline and polarisation, along the spectral window.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
            Must be larger than one meter.
        FT_beam : array_like
            Array made of the FT of the beam along the spectral window.
            Must have dimensions (Nfreqs, N, N).
            If empty array, FT_beam called from self.get_FT().
        kperp_bins : array_like
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with self.little_h.
            If computing for different baselines, make sure to input identical arrays.
        kpara_bins : array_like
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning.
            Make sure the values are consistent with self.little_h.
        return_bins : str
            If 'weighted', return bins weighted by the actual modes inside of bin.
            If 'unweighted', return bins used to build the histogram.
            If 'none', does not return anything. Bins can later be retrieved with 
            get_kperp_bins() and get_kpara_bins().

        Returns
        ----------
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is the array of delays considered (self.dly_array).
            Axis 1 is kperp.
            Axis 2 is kparallel.
        kperp : array_like
            Values of kperp corresponding to the axis=1 of cyl_wf.
            Note: if return_bins='weighted', these values are weighted 
            by their number of counts in the cylindrical binning.
        kpara : array_like
            Values of kpara corresponding to the axis=2 of cyl_wf.
            Note: if return_bins='weighted', these values are weighted 
            by their number of counts in the cylindrical binning.

        """

        ##### INITIALISE PARAMETERS

        assert bl_len>1., "Baseline length must be given in meters and be above 1m."

        FT_beam = np.array(FT_beam)
        if (len(FT_beam)==0): 
            # get FT of the beam from file and set frequency_related attributed (such as avg_z...)
            FT_beam = self.get_FT()

        #k-bins for cylindrical binning
        if np.size(kperp_bins)==0 or kperp_bins is None:
            kperp_bins = self.get_kperp_bins([bl_len])
        else:
            kperp_bins = np.array(kperp_bins)
        nbins_kperp = kperp_bins.size
        dk_perp = np.diff(kperp_bins).mean()
        kperp_bin_edges = np.arange(kperp_bins.min()-dk_perp/2,kperp_bins.max()+dk_perp,step=dk_perp)
        kperp_centre = self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*bl_len
        if (kperp_bin_edges.max()<kperp_centre+9.*dk_perp) or (kperp_bin_edges.min()>kperp_centre-9.*dk_perp):
            raise_warning('get_cylindrical_wf: The bin centre is not included in the array of kperp bins given as input.',
                            verbose=self.verbose)

        if np.size(kpara_bins)==0 or kpara_bins is None:
            kpara_bins = self.get_kpara_bins(self.freq_array,self.little_h,self.cosmo)
        else:                                              
            kpara_bins = np.array(kpara_bins)
        nbins_kpara = kpara_bins.size
        dk_para = np.diff(kpara_bins).mean()
        kpara_bin_edges = np.arange(kpara_bins.min()-dk_para/2,kpara_bins.max()+dk_para,step=dk_para)
        kpara_centre = self.cosmo.tau_to_kpara(self.avg_z,little_h=self.little_h)*abs(self.dly_array).max()
        if (kpara_bin_edges.max()<kpara_centre+9*dk_para) or (kpara_bin_edges.min()>kpara_centre-9.*dk_para):
            raise_warning('get_cylindrical_wf: The bin centre is not included in the array of kpara bins given as input.',
                            verbose=self.verbose)


        ##### COMPUTE CYLINDRICAL WINDOW FUNCTIONS

        # interpolate FT of beam onto regular grid of (kperp_x,kperp_y)
        interp_FT_beam, kperp_norm = self.interpolate_FT_beam(bl_len, FT_beam)
        # frequency resolution
        delta_nu = abs(self.freq_array[-1]-self.freq_array[0])/self.Nfreqs
        # obtain FT along frequency 
        fnu = self.take_freq_FT(interp_FT_beam,delta_nu)

        ##### cylindrical average

        # on sky plane
        wf_array1 = np.zeros((nbins_kperp,self.Nfreqs))
        kperp = np.zeros(nbins_kperp)
        for i in range(self.Nfreqs):
            for m in range(nbins_kperp):
                mask= (kperp_bin_edges[m]<=kperp_norm) & (kperp_norm<kperp_bin_edges[m+1])
                if np.any(mask): #cannot compute mean if zero elements
                    wf_array1[m,i]=np.mean(np.abs(fnu[mask,i])**2)
                    kperp[m] = np.mean(kperp_norm[mask])

        # in frequency direction    
        cyl_wf = np.zeros((self.Nfreqs,nbins_kperp,nbins_kpara))
        for it,tau in enumerate(self.dly_array[:self.Nfreqs//2+1]):
            kpara, cyl_wf[it,:,:] = self.get_wf_for_tau(tau,wf_array1,kperp_bins,kpara_bins)
        #fill by symmetry for tau = -tau
        if (self.Nfreqs%2==0):
            cyl_wf[self.Nfreqs//2+1:,:,:]=np.flip(cyl_wf,axis=0)[self.Nfreqs//2:-1]
        else:
            cyl_wf[self.Nfreqs//2+1:,:,:]=np.flip(cyl_wf,axis=0)[self.Nfreqs//2+1:]

        ### normalisation of window functions
        sum_per_bin = np.sum(cyl_wf,axis=(1,2))[:,None,None]
        cyl_wf = np.divide(cyl_wf,sum_per_bin,where=sum_per_bin!=0)

        if (return_bins=='unweighted'):
            return kperp_bins, kpara_bins, cyl_wf
        elif (return_bins=='weighted'):
            return kperp, kpara, cyl_wf
        else:
            return cyl_wf

    def cylindrical2spherical(self,cyl_wf,kbins,ktot,bl_weights=None):

        """
        Take spherical average of cylindrical window functions, 
        given as an array (Nblpair_groups, Ndlys, nbins_kperp, nbins_kpara).

        Parameters
        ----------
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is the array of baseline lengths (self.bl_lens)
            Axis 1 is the array of delays considered (self.dly_array).
            Axis 2 is kperp.
            Axis 3 is kparallel.
        kbins : array_like
            1D float array of ascending |k| bin centers in [h] Mpc^-1 units.
            Using for spherical binning.
            Make sure the values are consistent with self.little_h.
        ktot : array_like
            2-dimensional array giving the magnitude of k corresponding to 
            kperp and kpara in cyl_wf.
        bl_weights : list of weights (float or int), optional
            Relative weight of each baseline-pair when performing the average. This
            is useful for bootstrapping. This should have the same shape as
            blpair_groups if specified. The weights are automatically normalized
            within each baseline-pair group. Default: None (all baseline pairs have
            unity weights).

        Returns
        ----------
        wf_spherical : array
            Array of spherical window functions.
            Shape (nbinsk, nbinsk).
        kweights : array
            If reutrn_weights is True.
            Returns number of k-modes per k-bin.

        """

        assert (ktot.shape == cyl_wf.shape[2:]), \
                "k magnitude grid does not match (kperp,kpara) grid in cyl_wf"
        assert len(kbins)>1, "must feed array of k bins for spherical average"                                                  

        if bl_weights is None: 
            bl_weights = np.ones(cyl_wf.shape[0])
        else:
            assert len(bl_weights)==cyl_wf.shape[0], \
                    "Blpair weights and cylindrical window functions do not match"

        nbinsk = kbins.size
        dk = np.diff(kbins).mean()
        kbin_edges = np.arange(kbins.min()-dk/2,kbins.max()+dk,step=dk)

        # construct array giving the k probed by each baseline-tau pair
        kperps = self.bl_lens * self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h)
        kparas = self.dly_array * self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h) 
        kmags = np.sqrt(kperps[:,None]**2+kparas**2)

        wf_spherical = np.zeros((nbinsk,nbinsk))
        kweights = np.zeros(nbinsk,dtype=int)
        for m1 in range(nbinsk):
            mask2 = ((kbin_edges[m1]<=kmags) & (kmags<kbin_edges[m1+1])).astype(int)
            if (np.sum(mask2)!=0):
                mask2 = mask2*bl_weights[:,None] #add weights for redundancy
                kweights[m1] = np.sum(mask2) 
                wf_temp = np.sum(cyl_wf*mask2[:,:,None,None],axis=(0,1))/np.sum(mask2)
                if np.sum(wf_temp) > 0.: 
                    for m in range(nbinsk):
                        mask = (kbin_edges[m]<=ktot) & (ktot<kbin_edges[m+1])
                        if np.any(mask): #cannot compute mean if zero elements
                            wf_spherical[m1,m]=np.mean(wf_temp[mask])
                    # normalisation
                    wf_spherical[m1,:] = np.divide(wf_spherical[m1,:],np.sum(wf_spherical[m1,:]),
                                            where = np.sum(wf_spherical[m1,:])!=0)
            
        if np.any(kweights==0.) and self.verbose:
            raise_warning('Some spherical bins are empty. Add baselines or expand spectral window.')

        return wf_spherical, kweights


    def get_spherical_wf(self,spw_range,pol,
                            kbins, kperp_bins=[], kpara_bins=[],
                            bl_groups=[],bl_lens=[], blpair_weights=None,
                            save_cyl_wf = False, return_weights=False,
                            verbose=None):
        """
        Get spherical window functions for a set of baselines, polarisation,
        along a given spectral range, and for a set of kbins used for averaging.

        Parameters
        ----------
        spw_range : tuple of ints
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA bandwidth).
        pol : str
            Can be pseudo-Stokes or power: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
        kbins : array_like
            1D float array of ascending |k| bin centers in [h] Mpc^-1 units.
            Using for spherical binning.
            Make sure the values are consistent with self.little_h.
        kperp_bins : array_like, optional.
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with self.little_h.
        kpara_bins : array_like, optional.
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning.
            Make sure the values are consistent with self.little_h.
        bl_groups : embedded lists, optional.
            List of groups of baselines gathered by lengths
            (can be redundant groups from utils.get_reds).
            Can be optional if self.is_uvdata was given.
        bl_lens : list, optional.
            List of lengths corresponding to each group
            (can be redundant groups from utils.get_reds).
            Must have same length as bl_groups.
            Can be optional if self.is_uvdata was given.
        blpair_weights : list of weights (float or int), optional
            Relative weight of each baseline-pair when performing the average. This
            is useful for bootstrapping. This should have the same shape as
            blpair_groups if specified. The weights are automatically normalized
            within each baseline-pair group. Default: None (all baseline pairs have
            unity weights).
        save_cyl_wf : bool, optional
            Bool to choose if the cylindrical window functions get saved or not.
            Default is False.
        return_weights : bool, optional
            Save the weights associated with the different kbins when spherically
            binning the window functions.
            Default is False.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
            If None, value used is the class attribute.


        Returns
        ----------
        wf_spherical : array
            Array of spherical window functions.
            Shape (nbinsk, nbinsk).
        kweights : array
            If reutrn_weights is True.
            Returns number of k-modes per k-bin.

        """


        ##### INITIALISE PARAMETERS

        if verbose is None:
            verbose = self.verbose

        # check if bl_groups is nested list
        # if not any(isinstance(i, list) for i in bl_groups):
        #     bl_groups = [bl_groups]
        # check consistency of baseline-related inputs
        assert len(bl_groups)==len(bl_lens), "bl_groups and bl_lens must have same length"
        nbls = len(bl_groups) # number of redudant groups

        # Create baseline-pair weights list if not specified
        if blpair_weights is None:
            # Assign unity weights to baseline-pair groups that were specified
            self.bl_weights = np.array([len(l) for l in bl_groups])
        else:
            # Check that blpair_weights has the same shape as blpair_groups
            assert len(blpair_weights) == len(bl_groups), raise_warning("blpair_weights must \
                                                                            have the same shape as blpair_groups")
            self.bl_weights = blpair_weights

        # read baseline groups from data file if given
        if self.is_uvdata:
            if len(bl_groups)==0:
                bl_groups, bl_lens, _ = utils.get_reds(self.uvdata,bl_error_tol=1.0,pick_data_ants=True)
            else:
                # check baselines given as input are in data file
                baselines_in_file = [self.uvdata.baseline_to_antnums(bl) for bl in self.uvdata.baseline_array]
                assert np.all([bl in baselines_in_file for bl in sum(bl_groups, [])]), \
                        "Baselines given as input are not in data file."
        else:
            assert len(bl_groups)>0, "Must give list of baselines as input"

        # sets basic properties of UVWindow object
        self.set_bl_lens(np.array(bl_lens))

        # consistency checks for spw range given
        self.set_spw_range(spw_range)
        self.set_polarisation(pol)
        # get FT of the beam from file and set frequency_related attributed (such as avg_z...)
        FT_beam = self.get_FT()

        #k-bins for cylindrical binning
        if (np.size(kperp_bins)==0) or (kperp_bins is None):
            # define default kperp bins, making sure all values probed by bl_lens are
            # included and there is no over-sampling
            self.kperp_bins = self.get_kperp_bins(self.bl_lens)
        else:
            # read from input
            self.kperp_bins = np.array(kperp_bins)
        nbins_kperp = self.kperp_bins.size
        dk_perp = np.diff(self.kperp_bins).mean()
        kperp_bin_edges = np.arange(self.kperp_bins.min()-dk_perp/2,self.kperp_bins.max()+dk_perp,step=dk_perp)
        # make sure proper kperp values are included in given bins, raise warning otherwise
        kperp_max = self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*np.max(self.bl_lens) + 10.*dk_perp
        kperp_min = self.cosmo.bl_to_kperp(self.avg_z,little_h=self.little_h)*np.min(self.bl_lens) + 10.*dk_perp
        if (kperp_bin_edges.max()<=kperp_max): 
            raise_warning('get_spherical_wf: Max kperp bin centre not included in binning array',
                            verbose=verbose)
        if (kperp_bin_edges.min()>=kperp_min): 
            raise_warning('get_spherical_wf: Min kperp bin centre not included in binning array',
                            verbose=verbose)

        if np.size(kpara_bins)==0 or kpara_bins is None:
            # define default kperp bins, making sure all values probed by freq array are
            # included and there is no over-sampling
            self.kpara_bins = self.get_kpara_bins(self.freq_array,self.little_h,self.cosmo)
        else:                                              
            self.kpara_bins = np.array(kpara_bins)
        nbins_kpara = self.kpara_bins.size
        dk_para = np.diff(self.kpara_bins).mean()
        kpara_bin_edges = np.arange(self.kpara_bins.min()-dk_para/2,self.kpara_bins.max()+dk_para,step=dk_para)
        kpara_centre = self.cosmo.tau_to_kpara(self.avg_z,little_h=self.little_h)*abs(self.dly_array).max()
        # make sure proper kpara values are included in given bins, raise warning otherwise
        if (kpara_bin_edges.max()<=kpara_centre+5*dk_para) or (kpara_bin_edges.min()>=kpara_centre-5.*dk_para):
            raise_warning('get_spherical_wf: The bin centre is not included in the array of kpara bins given as input.',
                            verbose=verbose)

        # array of |k|=sqrt(kperp**2+kpara**2)
        ktot = np.sqrt(self.kperp_bins[:,None]**2+self.kpara_bins**2)


        # k-bins for spherical binning
        assert len(kbins)>1, "must feed array of k bins for spherical averasge"                                                  
        self.kbins = np.array(kbins)
        nbinsk = self.kbins.size
        dk = np.diff(self.kbins).mean()
        kbin_edges = np.arange(self.kbins.min()-dk/2,self.kbins.max()+dk,step=dk)
        # make sure proper ktot values are included in given bins, raise warning otherwise
        if (kbin_edges.max()<=ktot.max()): 
            raise_warning('Max spherical k probed is not included in bins.',
                            verbose=verbose)
        if (kbin_edges.min()>=ktot.min()): 
            raise_warning('Min spherical k probed is not included in bins.',
                            verbose=verbose)

        ##### COMPUTE THE WINDOW FUNCTIONS

        # get cylindrical window functions for each baseline length considered
        # as a function of (kperp, kpara)
        # the kperp and kpara bins are given as global parameters
        cyl_wf = np.zeros((nbls,self.Nfreqs,nbins_kperp,nbins_kpara))
        for ib in range(nbls):
            if verbose: 
                sys.stdout.write('\rComputing for bl %i of %i...' %(ib+1,nbls))
            cyl_wf[ib,:,:,:] = self.get_cylindrical_wf(self.bl_lens[ib], FT_beam,
                                                                        self.kperp_bins, self.kpara_bins)
        if verbose: sys.stdout.write('\rComputing for bl %i of %i... \n' %(nbls,nbls))
        if save_cyl_wf: 
            if verbose: print('Saving cylindrical window functions...')
            self.cyl_wf = cyl_wf


        # perform spherical binning
        wf_spherical, kweights = self.cylindrical2spherical(cyl_wf,self.kbins,ktot,self.bl_weights)

        if return_weights:
            return wf_spherical, kweights
        else:
            return wf_spherical


    def clear_cache(self,clear_cyl_bins=True):
        """
        Clear stored window function arrays (and cylindrical bins).

        Parameters
        ----------
        clear_cyl_bins : bool, optional
            Set to True if you want to clear arrays of (kperp,kparallel)
            bins used for cylindrical binning of the window functions.

        """
        self.cyl_wf = []
        if clear_cyl_bins:
            self.kperp_bins, self.kpara_bins = [], []


def raise_warning(warning, verbose=True):
    """
    Warning function.
    """
    if verbose:
        print('Warning: %s' %warning)



import numpy as np
import parameter as prm
from pspecbase import pspecBase

class pspec(pspecBase):
    """
    A class for storing power spectrum data.
    """
    
    def __init__(self, Ndims, length_units=None, temp_units=None,\
        freq_units=None):
        """
        Container class for power spectrum data.
        
        Parameters
        ----------
        Ndims : int
            Number of dimensions of the power spectrum.
        
        length_units, temp_units, freq_units : str
            String specifying the units that have been assumed for lengths 
            (e.g. Mpc or Mpc/h), temperatures (e.g. K or mK), and frequencies 
            (e.g. MHz or GHz). These arguments are compulsory.
        """
        # Test whether units have been specified
        if length_units is None:
            raise KeyError('Must specify length_units e.g. Mpc/h')
        if temp_units is None:
            raise KeyError('Must specify temp_units e.g. mK')
        if freq_units is None:
            raise KeyError('Must specify freq_units e.g. GHz')
        self.check_units(length_units,temp_units)

        if Ndims not in (1, 2, 3):
            raise ValueError("Power spectrum cannot have %d dimensions." \
                % Ndims)

        # Units used
        desc = ('Units used in the power spectrum object'
                'Acceptable length units: Mpc, Mpc/h, Gpc, Gpc/h'
                'Acceptable temperature units: K, mK, uK, muK, microK')
        self._lengthunits = prm.psparam('lengthunits', description=desc,\
                            expected_type=str)
        self._tempunits = prm.psparam('tempunits', description=desc,\
                            expected_type=str)
        
        # Get dimensionality of power spectrum
        desc = ('Type of pspec: 1 = spherically averaged P(k); '
                '2 = cylindrically averaged P(kperp, kpara); '
                '3 = full 3D cube P(kx,ky,kz)')
        self._Ndims = prm.psparam('Ndims', description=desc, expected_type=int)
        
        # Define binning of power spectrum in wavenumber, k
        desc = ('Number of bins along each k direction')
        self._Nks = prm.psparam('Nks', description=desc, expected_type=int, 
                                form=('Ndims',))
        desc = ('Boundaries of k bins. List (length Ndims) of arrays, '
                'each of which is of shape (Nks[i],2), where each row '
                'of the array gives the bottom edge and top edge of a '
                'k bin.')
        self._kbounds = prm.psparam('kbounds', description=desc, 
                                    expected_type=np.float, form=('Ndims',),
                                    units='(%s)^-1' % length_units)
        
        # Define array for storing the power spectrum data
        desc = ('Power spectrum data. An array that gives the power spectrum '
                'values in each kbin. 1D, 2D, or 3D array depending on Ndims '
                'parameter.')
        self._pspec = prm.psparam('pspec', description=desc, 
                                  expected_type=np.float,
                                  units='(%s)^2 (%s)^3'
                                          % (temp_units, length_units))
        
        # Specify central redshift of power spectrum
        desc = ('Central redshift of power spectrum')
        self._z0 = prm.psparam('z0', description=desc, expected_type=np.float)
        
        # Specify frequency bins that were used to form the power spectrum
        desc = ('Frequencies used in forming the power spectrum')
        self._freqs = prm.psparam('freqs', description=desc, 
                                  expected_type=np.float, 
                                  units='%s' % freq_units)
        
        # Specify window function array
        desc = ('Window functions. An array where each row corresponds to '
                'the linear combination of all the other k bins of the '
                'true power spectrum that is probed when a power spectrum '
                'estimate is made at a particular k bin.')
        self._window = prm.psparam('window', description=desc, 
                                   expected_type=np.float)
        
        # Initialize base class with the set of parameters specified above
        super(pspec, self).__init__()

        # Set the number of dimensions of the power spectrum
        # E.g. Ndim = 1 is P(k), while Ndim = 3 is P(\vec{k})
        self._Ndims = Ndims

        self._lengthunits = length_units
        self._tempunits = temp_units

        
    def set_pspec(self, pk_data,kbin_edges=None):
        """
        Set the power spectrum data to a new set of values. If kbins are not
        provided, it is assumed that the input power spectrum is replacing
        an existing power spectrum and has the same binning. The input data are
        assumed to be in the same units that were used to initialize this 
        object.
        
        Parameters
        ----------
        pkdata : array_like
            New power spectrum data, assumed to have the same dimensionality, 
            binning, and units as the current data that it will replace.
        kbin_edges : array_like
            Edges of the k-bins. List (length Ndims) of arrays.
        """
        if pk_data.ndim != self._Ndims:
            raise ValueError("Input power spectrum has %s dimensions, but\
                the power spectrum container has %s dimensions." \
                % (pk_data.ndim,self._Ndims))

        if kbin_edges is not None:
            self._kbounds = kbin_edges
            self._Nks = np.zeros(self._Ndims, dtype=int)
            for i in range(self._Ndims):
                self._Nks[i] = kbin_edges[i].shape[0]




        # elif pk_data.ndim != len(kbin_edges):
        #     raise ValueError("Input power spectrum has dimensions %s, so\
        #         we expect %s separate arrays for the kbin edges, but instead\
        #         got %s arrays" \
        #         % (pk_data.ndim, pk_data.ndim, len(kbin_edges)))

        # for i in range(self._Ndims):
        #     if pk_data.

        self._pspec = pk_data
        
    def bin_centers(self):
        """
        Return the central k values for the power spectrum bins.
        
        Returns
        -------
        k_centers : array_like
            Central k values for each bin, in units of (length_units)^-1.
        """
        k_centers = []
        for i in range(self._Ndims):
            k_centers.append(np.mean(self._kbounds[i], axis=1))
        return k_centers
        
    def DeltaSq(self):
        """
        Return the 'dimensionless' power spectrum, related to the usual power 
        spectrum by Delta^2 = k^3 P(k) / (2\pi^2). The units are actually 
        (temp_units)^2.
        
        Returns
        -------
        DeltaSq : array_like
            'Dimensionless' power spectrum, in (temp_units)^2.
        """

        pspec = np.atleast_3d(self._pspec)
        if self._Ndims == 1:
            pspec = np.rollaxis(pspec,0,-1)
        k_mag = np.zeros_like(self._pspec)
        k_mids = self.bin_centers()
        for i in range(self._Nks[0]):
            for j in range(self._Nks[1]):
                for k in range(self._Nks[2]):
                    k_mag[i,j,k] = k_mids[i]**2 + k_mids[j]**2 + k_mids[k]**2
                    k_mag[i,j,k] = np.sqrt(k_mag[i,j,k])

        return np.squeeze(pspec * k_mag**3 / (2. * np.pi))

    def check_units(self, length_units, temp_units):
        """
        Checks a set of units to see if they are understood by our object

        
        Parameters
        ----------
        length_units, temp_units : str
            Units that the power spectrum should be returned in.
            Acceptable length units: 'Mpc', 'Mpc/h', 'Gpc', 'Gpc/h'
            Acceptable temperature units: 'K', 'mK', 'uK', 'muK', 'microK'

        """

        # Define acceptable units (i.e. that the code knows how to deal with)
        acceptable_length = ['Mpc', 'Mpc/h', 'h^-1 Mpc', 'h^-1Mpc', 
                             'Gpc', 'Gpc/h']
        acceptable_temp = ['K', 'mK', 'uK', 'muK', 'microK']

        # Sanity checks on units
        if length_units.lower() not in [l.lower() for l in acceptable_length]:
            raise ValueError("Length units '%s' not recognized." \
                % length_units)
        if temp_units.lower() not in [l.lower() for l in acceptable_temp]:
            raise ValueError("Temperature units '%s' not recognized." \
                % temp_units)

        
    def in_units(self, length_units, temp_units):
        """
        Return the power spectrum in a different set of units.
        
        Parameters
        ----------
        length_units, temp_units : str
            Units that the power spectrum should be returned in.
            Acceptable length units: 'Mpc', 'Mpc/h', 'Gpc', 'Gpc/h'
            Acceptable temperature units: 'K', 'mK', 'uK', 'muK', 'microK'
        
        Returns
        -------
        pspec : array_like
            Array containing the power spectrum converted to the new set of 
            units (the stored power spectrum is not modified).
        """

        pspec = self._pspec.copy()
        # Convert old and new units to a standard set of units (Mpc, K)
        # Variables length_units and temp_units refer to the new units
        # while self._lengthunits and self._tempunits are the old units
        if 'h' in self._lengthunits and 'h' not in length_units:
            pspec /= (cosmo_units.Ho)**3
        elif 'h' not in self._lengthunits and 'h' in length_units:
            pspec *= (cosmo_units.Ho)**3

        micro_old = False
        if ('m' in self._lengthunits) or ('u' in self._lengthunits):
            micro_old = True
        micro_new = False
        if ('m' in length_units) or ('u' in length_units):
            micro_new = True

        if micro_old and not micro_new:
            pspec /= 1000000.
        elif micro_new and not micro_old:
            pspec *= 1000000.

        # QUESTION: MAYBE WE SHOULD RETURN K VALUES AS WELL?

        return pspec
        
    def rebin(self, kbounds, method=None):
        """
        Rebin the power spectrum. 
        
        Parameters
        ----------
        kbounds : array_like
            Boundaries of k bins. List (length Ndims) of arrays, each of which 
            is of shape (Nks[i],2), where each row of the array gives the 
            bottom edge and top edge of a k bin.
        
        method : str
            Rebinning method to use. Options are:
              'simple': Combine bins using a simple average.
              'Ninv': Combine bins using inverse noise weights.
        
        Returns
        -------
        pspec : array_like
            Power spectrum array with the new k binning.
        """
        raise NotImplementedError()
        # rebins the pspec without changing dimensionality, produces a new object
        
    def cylindrical(self):
        """
        Return the cylindrically-averaged (i.e. 2D) power spectrum. This method 
        will only work if the stored power spectrum is at least 2D.
        
        If the stored power spectrum is 3D, it will perform a uniformly-
        weighted cylindrical average of the data.
        
        Returns
        -------
        pspec : pspec object
            New 2D power spectrum object in cylindrical (k_par, k_perp) bins.
        """
        raise NotImplementedError()
        # returns a P(kperp, kpara) if available or does a uniform summing.
        
    def spherical(self):
        """
        Return the spherically-averaged (i.e. 1D) power spectrum. This method 
        will only work if the stored power spectrum is at least 1D.
        
        If the stored power spectrum is 2D or 3D, it will perform a uniformly-
        weighted spherical average of the data.
        
        Returns
        -------
        pspec : pspec object
            New 1D power spectrum object in spherical (|k|) bins.
        """
        raise NotImplementedError()
        # same as cylindrical
        
    def reweight(self):
        """
        Apply a new set of weights to the stored power spectrum.
        
        Parameters
        ----------
        w : array_like
            Matrix operator for the new weights.
        
        Returns
        -------
        pspec : pspec object
            New power spectrum object, with power spectrum P_new = w^T P_old 
            and covariance C_new = w^T C_old w.
        """
        raise NotImplementedError()
        # applying a new M matrix, returns a new object
        

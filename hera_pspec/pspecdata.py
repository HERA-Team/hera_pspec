import numpy as np
import aipy
import pyuvdata
from .utils import hash, cov

class PSpecData(object):

    def __init__(self, dsets=[], wgts=[], beam=None):
        """
        Object to store multiple sets of UVData visibilities and perform
        operations such as power spectrum estimation on them.

        Parameters
        ----------
        dsets : List of UVData objects, optional
            List of UVData objects containing the data that will be used to
            compute the power spectrum. Default: Empty list.

        wgts : List of UVData objects, optional
            List of UVData objects containing weights for the input data.
            Default: Empty list.

        beam : PspecBeam object, optional
            PspecBeam object containing information about the primary beam
            Default: None.
        """
        self.clear_cov_cache() # Covariance matrix cache
        self.dsets = []; self.wgts = []
        self.Nfreqs = None
        
        # Set R to identity by default
        self.R = self.I

        # Store the input UVData objects if specified
        if len(dsets) > 0:
            self.add(dsets, wgts)

        # Store a primary beam
        self.primary_beam = beam

    def add(self, dsets, wgts):
        """
        Add a dataset to the collection in this PSpecData object.

        Parameters
        ----------
        dsets : UVData or list
            UVData object or list of UVData objects containing data to add to
            the collection.

        wgts : UVData or list
            UVData object or list of UVData objects containing weights to add
            to the collection. Must be the same length as dsets. If a weight is
            set to None, the flags of the corresponding
        """
        # Convert input args to lists if possible
        if isinstance(dsets, pyuvdata.UVData): dsets = [dsets,]
        if isinstance(wgts, pyuvdata.UVData): wgts = [wgts,]
        if wgts is None: wgts = [wgts,]
        if isinstance(dsets, tuple): dsets = list(dsets)
        if isinstance(wgts, tuple): wgts = list(wgts)

        # Only allow UVData or lists
        if not isinstance(dsets, list) or not isinstance(wgts, list):
            raise TypeError("dsets and wgts must be UVData or lists of UVData")

        # Make sure enough weights were specified
        assert(len(dsets) == len(wgts))

        # Check that everything is a UVData object
        for d, w in zip(dsets, wgts):
            if not isinstance(d, pyuvdata.UVData):
                raise TypeError("Only UVData objects can be used as datasets.")
            if not isinstance(w, pyuvdata.UVData) and w is not None:
                raise TypeError("Only UVData objects (or None) can be used as "
                                "weights.")

        # Append to list
        self.dsets += dsets
        self.wgts += wgts
        
        # Store no. frequencies and no. times
        self.Nfreqs = self.dsets[0].Nfreqs
        self.Ntimes = self.dsets[0].Ntimes

        # Store the actual frequencies
        self.freqs = self.dsets[0].freq_array[0]
        
    def validate_datasets(self):
        """
        Validate stored datasets and weights to make sure they are consistent
        with one another (e.g. have the same shape, baselines etc.).
        """
        # Sanity checks on input data
        assert len(self.dsets) > 1
        assert len(self.dsets) == len(self.wgts)

        # Check if data are all the same shape
        Nfreqs = [d.Nfreqs for d in self.dsets]
        assert np.unique(Nfreqs).size == 1

    def clear_cov_cache(self, keys=None):
        """
        Clear stored covariance data (or some subset of it).

        Parameters
        ----------
        keys : list of tuples, optional
            List of keys to remove from covariance matrix cache. If None, all
            keys will be removed. Default: None.
        """
        if keys is None:
            self._C, self._Cempirical, self._I, self._iC = {}, {}, {}, {}
            self._iCt = {}
        else:
            for k in keys:
                try: del(self._C[k])
                except(KeyError): pass
                try: del(self._Cempirical[k])
                except(KeyError): pass
                try: del(self._I[k])
                except(KeyError): pass
                try: del(self._iC[k])
                except(KeyError): pass

    def x(self, key):
        """
        Get data for a given dataset and baseline, as specified in a standard
        key format.

        Parameters
        ----------
        key : tuple
            Tuple containing dataset ID and baseline index. The first element
            of the tuple is the dataset index, and the subsequent elements are
            the baseline ID.

        Returns
        -------
        x : array_like
            Array of data from the requested UVData dataset and baseline.
        """
        assert isinstance(key, tuple)
        dset = key[0]; bl = key[1:]
        return self.dsets[dset].get_data(bl).T # FIXME: Transpose?

    def w(self, key):
        """
        Get weights for a given dataset and baseline, as specified in a
        standard key format.

        Parameters
        ----------
        key : tuple
            Tuple containing dataset ID and baseline index. The first element
            of the tuple is the dataset index, and the subsequent elements are
            the baseline ID.

        Returns
        -------
        x : array_like
            Array of weights for the requested UVData dataset and baseline.
        """
        assert isinstance(key, tuple)

        dset = key[0]; bl = key[1:]
        if self.wgts[dset] is not None:
            return self.wgts[dset].get_data(bl).T # FIXME: Transpose?
        else:
            # If weights were not specified, use the flags built in to the
            # UVData dataset object
            flags = self.dsets[dset].get_flags(bl).astype(float).T # FIXME: .T?
            return 1. - flags # Flag=1 => weight=0

    def C(self, key):
        """
        Estimate covariance matrices from the data.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        Returns
        -------
        C : array_like
            (Weighted) empirical covariance of data for baseline 'bl'.
        """
        assert isinstance(key, tuple)

        # Set covariance if it's not in the cache
        if not self._C.has_key(key):
            self.set_C( {key : cov(self.x(key), self.w(key))} )
            self._Cempirical[key] = self._C[key]

        # Return cached covariance
        return self._C[key]

    def set_C(self, cov):
        """
        Set the cached covariance matrix to a set of user-provided values.

        Parameters
        ----------
        cov : dict
            Dictionary containing new covariance values for given datasets and
            baselines. Keys of the dictionary are tuples, with the first item
            being the ID (index) of the dataset, and subsequent items being the
            baseline indices.
        """
        self.clear_cov_cache(cov.keys())
        for key in cov: self._C[key] = cov[key]

    def C_empirical(self, key):
        """
        Calculate empirical covariance from the data (with appropriate
        weighting).

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        Returns
        -------
        C_empirical : array_like
            Empirical covariance for the specified key.
        """
        assert isinstance(key, tuple)

        # Check cache for empirical covariance
        if not self._Cempirical.has_key(key):
            self._Cempirical[key] = cov(self.x(key), self.w(key))
        return self._Cempirical[key]

    def I(self, key):
        """
        Return identity covariance matrix.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        Returns
        -------
        I : array_like
            Identity covariance matrix, dimension (Nfreqs, Nfreqs).
        """
        assert isinstance(key, tuple)

        if not self._I.has_key(key):
            self._I[key] = np.identity(self.Nfreqs)
        return self._I[key]

    def iC(self, key):
        """
        Return the inverse covariance matrix, C^-1.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        Returns
        -------
        iC : array_like
            Inverse covariance matrix for specified dataset and baseline.
        """
        assert isinstance(key, tuple)

        # Calculate inverse covariance if not in cache
        if not self._iC.has_key(key):
            C = self.C(key)
            U,S,V = np.linalg.svd(C.conj()) # conj in advance of next step

            # FIXME: Not sure what these are supposed to do
            #if self.lmin is not None: S += self.lmin # ensure invertibility
            #if self.lmode is not None: S += S[self.lmode-1]

            # FIXME: Is series of dot products quicker?
            self.set_iC({key:np.einsum('ij,j,jk', V.T, 1./S, U.T)})
        return self._iC[key]

    def set_iC(self, d):
        """
        Set the cached inverse covariance matrix for a given dataset and
        baseline to a specified value. For now, you should already have applied
        weights to this matrix.

        Parameters
        ----------
        d : dict
            Dictionary containing data to insert into inverse covariance matrix
            cache. Keys are tuples, following the same format as the input to
            self.iC().
        """
        for k in d: self._iC[k] = d[k]
    
    def set_R(self, R_matrix):
        """
        Set the weighting matrix R for later use in q_hat.

        Parameters
        ----------
        R_matrix : string or matrix
            If set to "identity", sets R = I
            If set to "iC", sets R = C^-1
            Otherwise, accepts a user inputted dictionary
        """

        if R_matrix == "identity":
            self.R = self.I
        elif R_matrix == "iC":
            self.R = self.iC
        else:
            self.R = R_matrix
      
    def q_hat(self, key1, key2, use_fft=True, taper='none'):
        """
        Construct an unnormalized bandpower, q_hat, from a given pair of
        visibility vectors. Returns the following quantity:
            
          \hat{q}_a = (1/2) conj(x_1) R_1 Q_a R_2 x_2 (arXiv:1502.06016, Eq. 13)
        
        Note that the R matrix need not be set to C^-1. This is something that
        is set by the user in the set_R method.
        
        Parameters
        ----------
        key1, key2 : tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors.

        use_fft : bool, optional
            Whether to use a fast FFT summation trick to construct q_hat, or
            a simpler brute-force matrix multiplication. The FFT method assumes
            a delta-fn bin in delay space. Default: True.

        taper : str, optional
            Tapering (window) function to apply to the data. Takes the same
            arguments as aipy.dsp.gen_window(). Default: 'none'.

        Returns
        -------
        q_hat : array_like
            Unnormalized bandpowers
        """
        assert isinstance(key1, tuple)
        assert isinstance(key2, tuple)
        
        # Calculate R x_1 and R x_2
        #iC1x, iC2x = 0, 0
        #for _k in k1: iC1x += np.dot(icov_fn(_k), self.x(_k))
        #for _k in k2: iC2x += np.dot(icov_fn(_k), self.x(_k))
        Rx1 = np.dot(self.R(key1), self.x(key1))
        Rx2 = np.dot(self.R(key2), self.x(key2))
            
        # Whether to use FFT or slow direct method
        if use_fft:

            if taper != 'none':
                tapering_fct = aipy.dsp.gen_window(self.Nfreqs, taper)
                Rx1 *= tapering_fct
                Rx2 *= tapering_fct

            _Rx1 = np.fft.fft(Rx1.conj(), axis=0)
            _Rx2 = np.fft.fft(Rx2.conj(), axis=0)
          
            # Conjugated because inconsistent with pspec_cov_v003 otherwise
            # FIXME: Check that this should actually be conjugated
            return 0.5 * np.conj(  np.fft.fftshift(_Rx1, axes=0).conj() 
                           * np.fft.fftshift(_Rx2, axes=0) )
        else:
            # Slow method, used to explicitly cross-check FFT code
            q = []
            for i in xrange(self.Nfreqs):
                Q = self.get_Q(i, self.Nfreqs, taper=taper)
                RQR = np.einsum('ab,bc,cd', 
                                self.R(key1).T.conj(), Q, self.R(key2))
                qi = np.sum(self.x(key1).conj()*np.dot(RQR, self.x(key2)), axis=0)
                q.append(qi)
            return 0.5 * np.array(q)

    def get_G(self, key1, key2, taper='none'):
        """
        Calculates the response matrix G of the unnormalized band powers q
        to the true band powers p, i.e.,

            <q_a> = \sum_b G_{ab} p_b

        This is given by

            G_ab = (1/2) Tr[R_1 Q_a R_2 Q_b]

        Note that in the limit that R_1 = R_2 = C^-1, this reduces to the Fisher
        matrix

            F_ab = 1/2 Tr [C^-1 Q_a C^-1 Q_b] (arXiv:1502.06016, Eq. 17)

        Parameters
        ----------
        key1, key2 : tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors.

        taper : str, optional
            Tapering (window) function used when calculating Q. Takes the same
            arguments as aipy.dsp.gen_window(). Default: 'none'.

        Returns
        -------
        G : array_like, complex
            Fisher matrix, with dimensions (Nfreqs, Nfreqs).
        """
        assert isinstance(key1, tuple)
        assert isinstance(key2, tuple)
        
        G = np.zeros((self.Nfreqs, self.Nfreqs), dtype=np.complex)
        R1 = self.R(key1)
        R2 = self.R(key2)
        
        iR1Q, iR2Q = {}, {}
        for ch in xrange(self.Nfreqs): # this loop is nchan^3
            Q = self.get_Q(ch, self.Nfreqs, taper=taper)
            iR1Q[ch] = np.dot(R1, Q) # R_1 Q
            iR2Q[ch] = np.dot(R2, Q) # R_2 Q

        for i in xrange(self.Nfreqs): # this loop goes as nchan^4
            for j in xrange(self.Nfreqs):
                # tr(R_2 Q_i R_1 Q_j)
                G[i,j] += np.einsum('ij,ji', iR1Q[i], iR2Q[j])

        return G.real / 2.

    def get_V_gaussian(self, key1, key2):
        """
        Calculates the bandpower covariance matrix
        V_ab = tr(C E_a C E_b)
        # FIXME: Must check factor of 2 with Wick's thm for complex vectors
        # and also check expression for when x_1 != x_2.
        
        Parameters
        ----------
        key1, key2 : tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors.
        
        Returns
        -------
        V : array_like, complex
            bandpower covariance matrix, with dimensions (Nfreqs, Nfreqs).
        """
        raise NotImplementedError()
   
    def get_MW(self, G, mode='I'):
        """
        Construct the normalization matrix M and window function matrix W for
        the power spectrum estimator. These are defined through Eqs. 14-16 of
        arXiv:1502.06016:

            \hat{p} = M \hat{q}
            <\hat{p}> = W p
            W = M G,
        
        where p is the true band power and G is the response matrix (defined above
        in get_G) of unnormalized bandpowers to normed bandpowers. The G matrix
        is the Fisher matrix when R = C^-1

        Several choices for M are supported:
        
            'G^-1':   Set M = G^-1, the (pseudo)inverse response matrix.
            'G^-1/2': Set M = G^-1/2, the root-inverse response matrix (using SVD).
            'I':      Set M = I, the identity matrix.
            'L^-1':   Set M = L^-1, Cholesky decomposition.

        Note that when we say (e.g., M = I), we mean this before normalization.
        The M matrix needs to be normalized such that each row of W sums to 1.
        
        Parameters
        ----------
        G : array_like or dict of array_like
            Response matrix for the bandpowers, with dimensions (Nfreqs, Nfreqs).
            If a dict is specified, M and W will be calculated for each G 
            matrix in the dict.

        mode : str, optional
            Definition to use for M. Must be one of the options listed above. 
            Default: 'I'.
        
        Returns
        -------
        M : array_like
            Normalization matrix, M. (If G was passed in as a dict, a dict of 
            array_like will be returned.)

        W : array_like
            Window function matrix, W. (If G was passed in as a dict, a dict of 
            array_like will be returned.)
        """
        # Recursive case, if many F's were specified at once
        if type(G) is dict:
            M,W = {}, {}
            for key in G: M[key],W[key] = self.get_MW(G[key], mode=mode)
            return M, W

        # Check that mode is supported
        modes = ['G^-1', 'G^-1/2', 'I', 'L^-1']
        assert(mode in modes)

        # Build M matrix according to specified mode
        if mode == 'G^-1':
            M = np.linalg.pinv(G, rcond=1e-12)
            #U,S,V = np.linalg.svd(F)
            #M = np.einsum('ij,j,jk', V.T, 1./S, U.T)
            
        elif mode == 'G^-1/2':
            U,S,V = np.linalg.svd(G)
            M = np.einsum('ij,j,jk', V.T, 1./np.sqrt(S), U.T)

        elif mode == 'I':
            M = np.identity(G.shape[0], dtype=G.dtype)
            
        else:
            """
            # Cholesky decomposition to get M (XXX: Needs generalizing)
            #order = np.array([10, 11, 9, 12, 8, 20, 0,
            #                  13, 7, 14, 6, 15, 5, 16,
            #                  4, 17, 3, 18, 2, 19, 1])
            """
            order = np.arange(G.shape[0]) - np.ceil((G.shape[0]-1.)/2.)
            order[order < 0] = order[order < 0] - 0.1
            
            # Negative integers have larger absolute value so they are sorted
            # after positive integers.
            order = (np.abs(order)).argsort()
            if np.mod(G.shape[0], 2) == 1:
                endindex = -2
            else:
                endindex = -1
            order = np.hstack([order[:5], order[endindex:], order[5:endindex]])
            iorder = np.argsort(order)
            
            G_o = np.take(np.take(G, order, axis=0), order, axis=1)
            L_o = np.linalg.cholesky(G_o)
            U,S,V = np.linalg.svd(L_o.conj())
            M_o = np.dot(np.transpose(V), np.dot(np.diag(1./S), np.transpose(U)))
            M = np.take(np.take(M_o, iorder, axis=0), iorder, axis=1)

        # Calculate (normalized) W given Fisher matrix and choice of M
        W = np.dot(M, G)
        norm = W.sum(axis=-1); norm.shape += (1,)
        M /= norm; W = np.dot(M, G)
        return M, W

    def get_Q(self, mode, n_k, taper='none'):
        """
        Response of the covariance to a given bandpower, dC / dp_alpha.

        Assumes that Q will operate on a visibility vector in frequency space.
        In other words, produces a matrix Q that performs a two-sided Fourier
        transform and extracts a particular Fourier mode.

        (Computing x^t Q y is equivalent to Fourier transforming x and y
        separately, extracting one element of the Fourier transformed vectors,
        and then multiplying them.)

        Parameters
        ----------
        mode : int
            Central wavenumber (index) of the bandpower, p_alpha.

        n_k : int
            Number of k bins that will be .

        taper : str, optional
            Type of tapering (window) function to use. Valid options are any
            window function supported by aipy.dsp.gen_window(). Default: 'none'.

        Returns
        -------
        Q : array_like
            Response matrix for bandpower p_alpha.
        """
        _m = np.zeros((n_k,), dtype=np.complex)
        _m[mode] = 1. # delta function at specific delay mode

        # FFT to transform to frequency space, and apply window function
        m = np.fft.fft(np.fft.ifftshift(_m)) * aipy.dsp.gen_window(n_k, taper)
        Q = np.einsum('i,j', m, m.conj()) # dot it with its conjugate
        return Q


    def p_hat(self, M, q):
        """
        Optimal estimate of bandpower p_alpha, defined as p_hat = M q_hat.

        Parameters
        ----------
        M : array_like
            Normalization matrix, M.

        q : array_like
            Unnormalized bandpowers, \hat{q}.

        Returns
        -------
        p_hat : array_like
            Optimal estimate of bandpower, \hat{p}.
        """
        return np.dot(M, q)

    def scalar(self, stokes='I', taper='none', little_h=True, num_steps=10000):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units

        See arxiv:1304.4991 and HERA memo #27 for details.

        Currently this is only for Stokes I.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'

        taper : str, optional
                Whether a tapering function (e.g. Blackman-Harris) is being
                used in the power spectrum estimation.
                Default: none

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        num_steps : int, optional
                Number of steps to use when interpolating primary beams for
                numerical integral
                Default: 10000

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """
        scalar = self.primary_beam.compute_pspec_scalar(\
                self.freqs[0],self.freqs[-1],self.Nfreqs,\
                stokes,taper,little_h,num_steps)
        return scalar

    def pspec(self, bls, beam=None, input_data_weight='identity', norm='I', 
              taper='none', little_h=True, verbose=False):
        """
        Estimate the power spectrum from the datasets contained in this object,
        using the optimal quadratic estimator (OQE) from arXiv:1502.06016.

        Parameters
        ----------
        bls : list of tuples
            List of baselines to include in the power spectrum calculation.
            Each baseline is specified as a tuple of antenna IDs.

        beam : PspecBeam object
            Primary beam information for the data being inputted.

        input_data_weight : str, optional
            String specifying what weighting matrix to apply to the input
            data. See the options in the set_R method for details.
            
        norm : str, optional
            String specifying how to choose the normalization matrix, M. See 
            the 'mode' argument of get_MW() for options.

        taper : str, optional
            Tapering (window) function to apply to the data. Takes the same
            arguments as aipy.dsp.gen_window(). Default: 'none'.

        little_h : boolean, options
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        verbose : bool, optional
            If True, print progress/debugging information.

        Returns
        -------
        pspec : list of np.ndarray
            Optimal quadratic estimate of the power spectrum for the datasets
            stored in this PSpecData and baselines specified in 'keys'.

        pairs : list of tuples
            List of the pairs of datasets and baselines that were used to
            calculate each element of the 'pspec' list.
        """
        #FIXME: Define sensible grouping behaviors.
        #FIXME: Check that requested keys exist in all datasets

        # Validate the input data to make sure it's sensible
        self.validate_datasets()

        # Compute the scalar to convert from "telescope units" to "cosmo units"
        # once and for all
        if self.primary_beam != None:
            scalar = self.scalar(taper=taper, little_h=True)

        pvs = []; pairs = []
        # Loop over pairs of datasets
        for m in xrange(len(self.dsets)):
            for n in xrange(m+1, len(self.dsets)):
                # Datasets should not be cross-correlated with themselves, and
                # dataset pair (m, n) gives the same result as (n, m)

                # Loop over baselines
                for bl in bls:
                    key1 = (m,) + bl
                    key2 = (n,) + bl

                    if verbose: print("Baselines:", key1, key2)
                    
                    # Set covariance weighting scheme for input data
                    if verbose: print (" Setting weighting matrix for input data...")
                    self.set_R(input_data_weight)
                    
                    # Build Fisher matrix
                    if verbose: print("  Building G...")
                    Gv = self.get_G(key1, key2, taper=taper)
                    
                    # Calculate unnormalized bandpowers
                    if verbose: print("  Building q_hat...")
                    qv = self.q_hat(key1, key2, taper=taper)
                    
                    # Normalize power spectrum estimate
                    if verbose: print("  Normalizing power spectrum...")
                    Mv, Wv = self.get_MW(Gv, mode=norm)
                    pv = self.p_hat(Mv, qv)

                    # Multiply by scalar
                    if self.primary_beam != None:
                        if verbose: print("  Computing and multiplying scalar...")
                        pv *= scalar

                    # Save power spectra and dataset/baseline pairs
                    pvs.append(pv)
                    pairs.append((key1, key2))
        return np.array(pvs).real, pairs

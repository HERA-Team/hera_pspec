import numpy as np
import aipy
import pyuvdata
from hera_pspec import utils
import itertools
import copy
import hera_cal as hc
from hera_pspec import uvpspec, version
from collections import OrderedDict as odict
from pyuvdata import utils as uvutils
import operator


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
        self.clear_cov_cache()  # Covariance matrix cache
        self.dsets = []; self.wgts = []
        self.Nfreqs = None
        self.spw_range = None
        self.spw_Nfreqs = None
        
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
        self.spw_range = (0, self.Nfreqs)
        self.spw_Nfreqs = self.Nfreqs
        
    def validate_datasets(self, verbose=True):
        """
        Validate stored datasets and weights to make sure they are consistent
        with one another (e.g. have the same shape, baselines etc.).
        """
        # check dsets and wgts have same number of elements
        if len(self.dsets) != len(self.wgts):
            raise ValueError("self.wgts does not have same length as self.dsets")

        # Check if dsets are all the same shape along freq axis
        Nfreqs = [d.Nfreqs for d in self.dsets]
        if np.unique(Nfreqs).size > 1:
            raise ValueError("all dsets must have the same Nfreqs")

        # Check shape along time axis
        Ntimes = [d.Ntimes for d in self.dsets]
        if np.unique(Ntimes).size > 1:
            raise ValueError("all dsets must have the same Ntimes")

        # raise warnings if times don't match
        lst_diffs = np.array(map(lambda dset: np.unique(self.dsets[0].lst_array) - np.unique(dset.lst_array), self.dsets[1:]))
        if np.max(np.abs(lst_diffs)) > 0.001:
            raise_warning("Warning: taking power spectra between LST bins misaligned by more than 15 seconds",
                            verbose=verbose)

        # raise warning if frequencies don't match       
        freq_diffs = np.array(map(lambda dset: np.unique(self.dsets[0].freq_array) - np.unique(dset.freq_array), self.dsets[1:]))
        if np.max(np.abs(freq_diffs)) > 0.001e6:
            raise_warning("Warning: taking power spectra between frequency bins misaligned by more than 0.001 MHz",
                          verbose=verbose)

        # Check for the same polarizations
        pols = []
        for d in self.dsets: pols.extend(d.polarization_array)
        if np.unique(pols).size > 1:
            raise ValueError("all dsets must have the same number and kind of polarizations: \n{}".format(pols))

        # Check phase type
        phase_types = []
        for d in self.dsets: phase_types.append(d.phase_type)
        if np.unique(phase_types).size > 1:
            raise ValueError("all datasets must have the same phase type (i.e. 'drift', 'phased', ...)\ncurrent phase types are {}".format(phase_types))

        # Check phase centers if phase type is phased
        if 'phased' in set(phase_types):
            phase_ra = map(lambda d: d.phase_center_ra_degrees, self.dsets)
            phase_dec = map(lambda d: d.phase_center_dec_degrees, self.dsets)
            max_diff_ra = np.max(map(lambda d: np.diff(d), itertools.combinations(phase_ra, 2)))
            max_diff_dec = np.max(map(lambda d: np.diff(d), itertools.combinations(phase_dec, 2)))
            max_diff = np.sqrt(max_diff_ra**2 + max_diff_dec**2)
            if max_diff > 0.15: raise_warning("Warning: maximum phase-center difference between datasets is > 10 arcmin", verbose=verbose)

    def check_key_in_dset(self, key, dset_ind):
        """
        Check 'key' exists in the UVData object self.dsets[dset_ind]

        Parameters
        ----------
        key : tuple
            if length 1: assumed to be polarization number or string
            elif length 2: assumed to be antenna-number tuple (ant1, ant2)
            elif length 3: assumed ot be antenna-number-polarization tuple (ant1, ant2, pol)

        dset_ind : int, the index of the dataset to-be-checked

        Returns
        -------
        exists : bool
            True if the key exists, False otherwise
        """
        # get iterable
        key = pyuvdata.utils.get_iterable(key)
        if isinstance(key, str):
            key = (key,)

        # check key is a tuple
        if isinstance(key, tuple) == False or len(key) not in (1, 2, 3):
            raise KeyError("key {} must be a length 1, 2 or 3 tuple".format(key))

        try:
            _ = self.dsets[dset_ind]._key2inds(key)
            return True
        except KeyError:
            return False

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
        return self.dsets[dset].get_data(bl).T[self.spw_range[0]:self.spw_range[1], :] # FIXME: Transpose?

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
            return self.wgts[dset].get_data(bl).T[self.spw_range[0]:self.spw_range[1], :] # FIXME: Transpose?
        else:
            # If weights were not specified, use the flags built in to the
            # UVData dataset object
            flags = self.dsets[dset].get_flags(bl).astype(float).T[self.spw_range[0]:self.spw_range[1], :]
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
            self.set_C( {key : utils.cov(self.x(key), self.w(key))} )
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
            self._Cempirical[key] = utils.cov(self.x(key), self.w(key))
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
            self._I[key] = np.identity(self.spw_Nfreqs)
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

    def set_spw(self, spw_range):
        """
        Set the spectral window range

        Parameters
        ----------
        spw_range : tuple, contains start and end of spw in channel indices
            used to slice the frequency array
        """
        assert isinstance(spw_range, tuple), "spw_range must be fed as a len-2 integer tuple"
        assert isinstance(spw_range[0], (int, np.int)), "spw_range must be fed as len-2 integer tuple"
        self.spw_range = spw_range
        self.spw_Nfreqs = spw_range[1] - spw_range[0]

    def q_hat(self, key1, key2, use_fft=True, taper='none'):
        """
        Construct an unnormalized bandpower, q_hat, from a given pair of
        visibility vectors. Returns the following quantity:

          \hat{q}_a = (1/2) conj(x_1) R_1 Q_a R_2 x_2 (arXiv:1502.06016, Eq. 13)

        Note that the R matrix need not be set to C^-1. This is something that
        is set by the user in the set_R method.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors. If a list of tuples is provided, the baselines 
            in the list will be combined with inverse noise weights.
            
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
        Rx1, Rx2 = 0, 0
        
        # Calculate R x_1
        if isinstance(key1, list):
            for _key in key1: Rx1 += np.dot(self.R(_key), self.x(_key))
        else:
            Rx1 = np.dot(self.R(key1), self.x(key1))
        
        # Calculate R x_2
        if isinstance(key2, list):
            for _key in key2: Rx2 += np.dot(self.R(_key), self.x(_key))
        else:
            Rx2 = np.dot(self.R(key2), self.x(key2))
        
        # Whether to use FFT or slow direct method
        if use_fft:
            if taper != 'none':
                tapering_fct = aipy.dsp.gen_window(self.spw_Nfreqs, taper)
                Rx1 *= tapering_fct[:, None]
                Rx2 *= tapering_fct[:, None]

            _Rx1 = np.fft.fft(Rx1.conj(), axis=0)
            _Rx2 = np.fft.fft(Rx2.conj(), axis=0)
            
            return 0.5 * np.conj(  np.fft.fftshift(_Rx1, axes=0).conj() 
                                 * np.fft.fftshift(_Rx2, axes=0) )
        else:
            # get taper if provided
            if taper != 'none':
                tapering_fct = aipy.dsp.gen_window(self.spw_Nfreqs, taper)

            # Slow method, used to explicitly cross-check FFT code
            q = []
            for i in xrange(self.spw_Nfreqs):
                Q = self.get_Q(i, self.spw_Nfreqs)
                RQR = np.einsum('ab,bc,cd',
                                self.R(key1).T.conj(), Q, self.R(key2))
                x1 = self.x(key1).conj()
                x2 = self.x(key2)
                if taper != 'none':
                    x1 = x1 * tapering_fct[:, None]
                    x2 = x2 * tapering_fct[:, None]
                qi = np.sum(x1*np.dot(RQR, x2), axis=0)
                q.append(qi)
            return 0.5 * np.array(q)

    def get_G(self, key1, key2):
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
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors. If a list of tuples is provided, the baselines 
            in the list will be combined with inverse noise weights.

        Returns
        -------
        G : array_like, complex
            Fisher matrix, with dimensions (Nfreqs, Nfreqs).
        """
        G = np.zeros((self.spw_Nfreqs, self.spw_Nfreqs), dtype=np.complex)
        R1 = self.R(key1)
        R2 = self.R(key2)

        iR1Q, iR2Q = {}, {}
        for ch in xrange(self.spw_Nfreqs): # this loop is nchan^3
            Q = self.get_Q(ch, self.spw_Nfreqs)
            iR1Q[ch] = np.dot(R1, Q) # R_1 Q
            iR2Q[ch] = np.dot(R2, Q) # R_2 Q

        for i in xrange(self.spw_Nfreqs): # this loop goes as nchan^4
            for j in xrange(self.spw_Nfreqs):
                # tr(R_2 Q_i R_1 Q_j)
                G[i,j] += np.einsum('ab,ba', iR1Q[i], iR2Q[j])

        return G / 2.
    
    def get_V_gaussian(self, key1, key2):
        """
        Calculates the bandpower covariance matrix,
        
            V_ab = tr(C E_a C E_b)
            
        FIXME: Must check factor of 2 with Wick's theorem for complex vectors,
        and also check expression for when x_1 != x_2.
        
        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors. If a list of tuples is provided, the baselines 
            in the list will be combined with inverse noise weights.
        
        Returns
        -------
        V : array_like, complex
            Bandpower covariance matrix, with dimensions (Nfreqs, Nfreqs).
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
        # Recursive case, if many G's were specified at once
        if type(G) is dict:
            M,W = {}, {}
            for key in G: M[key], W[key] = self.get_MW(G[key], mode=mode)
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
            # Cholesky decomposition
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

    def get_Q(self, mode, n_k):
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

        Returns
        -------
        Q : array_like
            Response matrix for bandpower p_alpha.
        """
        _m = np.zeros((n_k,), dtype=np.complex)
        _m[mode] = 1. # delta function at specific delay mode

        # FFT to transform to frequency space
        m = np.fft.fft(np.fft.ifftshift(_m))
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

    def units(self, little_h=True):
        """
        Return the units of the power spectrum. These are inferred from the 
        units reported by the input visibilities (UVData objects).

        Parameters
        ----------
        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        Returns
        -------
        pspec_units : str
            Units of the power spectrum that is returned by pspec().
        """
        # Work out the power spectrum units
        if len(self.dsets) == 0:
            raise IndexError("No datasets have been added yet; cannot "
                             "calculate power spectrum units.")
        if self.primary_beam is None:
            pspec_units = "({})^2 Hz [beam normalization not specified]".format(self.dsets[0].vis_units)
        else:
            if little_h:
                h_unit = "h^-3 "
            else:
                h_unit = ""
            pspec_units = "({})^2 {}Mpc^3".format(self.dsets[0].vis_units, h_unit)
        
        return pspec_units
    
    def delays(self):
        """
        Return an array of delays, tau, corresponding to the bins of the delay 
        power spectrum output by pspec() using self.spw_range to specify the spectral window.
        
        Returns
        -------
        delays : array_like
            Delays, tau. Units: ns.
        """
        # Calculate the delays
        if len(self.dsets) == 0:
            raise IndexError("No datasets have been added yet; cannot "
                             "calculate delays.")
        else:
            return utils.get_delays(self.freqs[self.spw_range[0]:self.spw_range[1]]) * 1e9 # convert to ns    
    
    
    def scalar(self, pol='I', taper='none', little_h=True, num_steps=2000, beam=None):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units, using self.spw_range to set spectral window.

        See arxiv:1304.4991 and HERA memo #27 for details.

        Parameters
        ----------
        pol: str, optional
                Which polarization to compute the scalar for.
                e.g. 'I', 'Q', 'U', 'V', 'XX', 'YY'...
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

        beam : PSpecBeam object
            Option to use a manually-fed PSpecBeam object instead of using self.primary_beam.

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """
        # set spw_range and get freqs
        freqs = self.freqs[self.spw_range[0]:self.spw_range[1]]
        start = freqs[0]
        end = freqs[0] + np.median(np.diff(freqs)) * len(freqs)

        # calculate scalar
        if beam is None:
            scalar = self.primary_beam.compute_pspec_scalar(
                                    start, end, len(freqs), pol=pol,
                                    taper=taper, little_h=little_h, 
                                    num_steps=num_steps)
        else:
            scalar = beam.compute_pspec_scalar(start, end, len(freqs), 
                                               pol=pol, taper=taper, 
                                               little_h=little_h, 
                                               num_steps=num_steps)
        return scalar

    def pspec(self, bls1, bls2, dsets, input_data_weight='identity', norm='I', 
              taper='none', little_h=True, avg_group=False, 
              exclude_auto_bls=False, exclude_conjugated_blpairs=False,
              spw_ranges=None, verbose=True, history=''):
        """
        Estimate the delay power spectrum from a pair of datasets contained in 
        this object, using the optimal quadratic estimator from 
        arXiv:1502.06016.

        In this formulation, the power spectrum is proportional to the 
        visibility data via
        
        P = M data_{LH} E data_{RH}

        where E contains the data weighting and FT matrices, M is a 
        normalization matrix, and the two separate datasets are denoted as 
        "left-hand" and "right-hand".

        Each power spectrum is generated by taking a baseline (specified by 
        an antenna-pair and polarization key) from bls1 out of dsets[0] and 
        assigning it as data_LH, and a bl from bls2 out of the dsets[1] and 
        assigning it as data_RH.

        If the bl chosen from bls1 is (ant1, ant2) and the bl chosen from bls2 
        is (ant3, ant4), the "baseline-pair" describing their cross 
        multiplication is ((ant1, ant2), (ant3, ant4)).

        Parameters
        ----------
        bls1, bls2 : list of bl tuples 
            (or list of bl groups, which themselves are list of bl tuples)
            List of baseline tuples to use in the power spectrum calculation 
            for the "left-hand" dataset (bls1) and "right-hand" dataset (bls2). 
            A baseline tuple is specified as an antenna pair, Ex: (ant1, ant2)

            Alternatively, bls1 and bls2 can contain lists of baselines groups, 
            which are themselves lists of baseline tuples. A bl-group in bls1 
            should have the same positional index in bls2, i.e. a group at 
            bls1[i] should correspond to a group at bls2[i]. 
            
            In this case, pspec will do one of two things:

              1) If avg_group=True, the data in each baseline-group will be 
                 averaged together before squaring, and then crossed with the 
                 baseline groups with the same positions (indices) in bls1 and 
                 bls2, e.g. bls1[i] x bls2[i]

              2) If avg_group=False, all permutations of cross-spectra between 
                 bls in a group in bls1 and bls in the corresponding group in 
                 bls2 are calculated. (Note that baselines are never crossed 
                 between different baseline groups, which are defined based on 
                 their positional index in the list, i.e. we never do 
                 bls1[i] x bls2[j] for i != j.
                 
                 Example: if bls1 = bls2 = [[(1, 2), (2, 3)], [(1, 5), (2, 6)]] 
                 then we will find all permutations of:
                     [(1, 2), (2, 3)] x [(1, 2), (2, 3)] 
                 and [(1, 5), (2, 6)] x [(1, 5), (2, 6)].

                 If exclude_auto_bls = True, bl-pairs that have a repeated 
                 baseline are eliminated. Example: ((1, 2), (1, 2)) would be 
                 eliminated from the bl_pairs array.

                 If exclude_conjugated_blpairs = True, if a baseline pair 
                 exists as well as its conjugate, the latter is eliminated. 
                 Example: ((1, 2), (2, 3)) and ((2, 3), (1, 2)), in which case 
                 ((2, 3), (1, 2)) would be eliminated from the bl_pairs array.

        dsets : length-2 tuple or list
            Contains indices of self.dsets to use in forming power spectra, 
            where the first index is for the Left-Hand dataset and second index 
            is used for the Right-Hand dataset (see above).

        input_data_weight : str, optional
            String specifying which weighting matrix to apply to the input
            data. See the options in the set_R() method for details. 
            Default: 'identity'.

        norm : str, optional
            String specifying how to choose the normalization matrix, M. See 
            the 'mode' argument of get_MW() for options. Default: 'I'.

        taper : str, optional
            Tapering (window) function to apply to the data. Takes the same
            arguments as aipy.dsp.gen_window(). Default: 'none'.

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        exclude_conjugated_blpairs : boolean, optional
            If bls1 and bls2 are lists of bl groups, exclude conjugated 
            baseline-pairs. Example: If ((1, 2), (2,3)) and ((2, 3), (1,2)) 
            exist, exclude the latter.

        exclude_auto_bls : boolean, optional
            If bls1 and bls2 are lists of bl groups, exclude bl-pairs when a bl 
            is paired with itself. Used to prevent the inclusion of power 
            spectra with noise biases.

        avg_group : boolean, optional
            If bls1 and bls2 contain a list of bl groups, average data in each 
            group before cross-multiplying.

        spw_ranges : list of tuples, optional
            A list of spectral window channel ranges to select within the total 
            bandwidth of the datasets, each of which forms an independent power 
            spectrum estimate. Example: [(220, 320), (650, 775)].
            
            Each tuple should contain a start and stop channel used to index 
            the `freq_array` of each dataset. The default (None) is to use the 
            entire band provided in each dataset.

        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.

        history : str, optional
            history string to attach to UVPSpec object

        Returns
        -------
        uvp : UVPSpec object
            Instance of UVPSpec that holds the output power spectrum data.
        """
        # Validate the input data to make sure it's sensible
        self.validate_datasets(verbose=verbose)

        # get datasets
        assert isinstance(dsets, (list, tuple)), "dsets must be fed as length-2 tuple of integers"
        assert len(dsets) == 2, "len(dsets) must be 2"
        assert isinstance(dsets[0], (int, np.int)) and isinstance(dsets[1], (int, np.int)), "dsets must contain integer indices"
        dset1 = self.dsets[dsets[0]]
        dset2 = self.dsets[dsets[1]]

        # get polarization array from zero'th dset
        pol_arr = map(lambda p: pyuvdata.utils.polnum2str(p), dset1.polarization_array)

        # ensure both bls1 and bls2 are the same type
        if isinstance(bls1[0], tuple) and isinstance(bls1[0][0], (int, np.int)) \
            and isinstance(bls2[0], tuple) and isinstance(bls2[0][0], (int, np.int)):
            # bls1 and bls2 fed as list of bl tuples
            fed_bl_group = False

        elif isinstance(bls1[0], list) and isinstance(bls1[0][0], tuple) and isinstance(bls2[0], list) \
            and isinstance(bls2[0][0], tuple):
            # bls1 and bls2 fed as list of bl groups
            fed_bl_group = True
            assert len(bls1) == len(bls2), "if fed as list of bl groups, len(bls1) must equal len(bls2)"

        else:
            raise TypeError("bls1 and bls2 must both be fed as either a list of bl tuples, or a list of bl groups")

        # validate bl-pair redundancy
        validate_bls(bls1, bls2, dset1, dset2, baseline_tol=1.0)

        # construct list of baseline pairs
        bl_pairs = []
        for i in range(len(bls1)):
            if fed_bl_group:
                bl_grp = []
                for j in range(len(bls1[i])):
                    bl_grp.extend(itertools.combinations(bls2[i] + [bls1[i][j]], 2))
                # eliminate duplicates
                bl_grp = sorted(set(bl_grp))
                bl_pairs.append(bl_grp)
            else:
                bl_pairs.append((bls1[i], bls2[i]))

        # iterate through all bl pairs and ensure it exists in the specified dsets, else remove
        new_bl_pairs = []
        for i, blg in enumerate(bl_pairs):
            if fed_bl_group:
                new_blg = []
                for blp in blg:
                    if self.check_key_in_dset(blp[0], dsets[0]) and self.check_key_in_dset(blp[1], dsets[1]):
                        new_blg.append(blp)
                if len(new_blg) > 0:
                    new_bl_pairs.append(new_blg)
            else:
                if self.check_key_in_dset(blg[0], dsets[1]) and self.check_key_in_dset(blg[1], dsets[1]):
                    new_bl_pairs.append(blg)
        bl_pairs = new_bl_pairs

        # exclude autos or conjugated blpairs if desired
        if fed_bl_group:
            new_bl_pairs = []
            for i, blg in enumerate(bl_pairs):
                new_blg = []
                for blp in blg:
                    if exclude_auto_bls:
                        if blp[0] == blp[1]:
                            continue
                    if (blp[1], blp[0]) in new_blg and exclude_conjugated_blpairs:
                        continue
                    new_blg.append(blp)
                if len(new_blg) > 0:
                    new_bl_pairs.append(new_blg)
            bl_pairs = new_bl_pairs

        # flatten bl_pairs list if bls fed as bl groups but no averaging is desired
        if avg_group == False and fed_bl_group:
            bl_pairs = [item for sublist in bl_pairs for item in sublist]

        if avg_group:
            # bl group averaging currently fails at self.get_G() function
            raise NotImplementedError

        # configure spectral window selections
        if spw_ranges is None:
            spw_ranges = [(0, self.Nfreqs)]
        else:
            assert np.isclose(map(lambda t: len(t), spw_ranges), 2).all(), "spw_ranges must be fed as a list of length-2 tuples"

        # initialize empty lists
        data_array = odict()
        wgt_array = odict()
        integration_array = odict()
        time1 = []
        time2 = []
        lst1 = []
        lst2 = []
        spws = []
        dlys = []
        freqs = []
        sclr_arr = np.ones((len(spw_ranges), len(pol_arr)), np.float)
        blp_arr = []
        bls_arr = []

        # Loop over spectral windows
        for i in range(len(spw_ranges)):
            # set spectral range
            if verbose:
                print( "\nSetting spectral range: {}".format(spw_ranges[i]))
            self.set_spw(spw_ranges[i])

            # clear covariance cache
            self.clear_cov_cache()
            built_G = False  # haven't built Gv matrix in this spw loop yet

            # setup emtpy data arrays
            spw_data = []
            spw_wgts = []
            spw_ints = []
            
            d = self.delays() * 1e-9
            dlys.extend(d)
            spws.extend(np.ones_like(d, np.int) * i)
            freqs.extend(
                dset1.freq_array.flatten()[spw_ranges[i][0]:spw_ranges[i][1]] )

            # Loop over polarizations
            for j, p in enumerate(pol_arr):
                pol_data = []
                pol_wgts = []
                pol_ints = []

                # Compute scalar to convert "telescope units" to "cosmo units"
                if self.primary_beam is not None:
                    scalar = self.scalar(taper=taper, little_h=True)
                else: 
                    raise_warning("Warning: self.primary_beam is not defined, "
                                  "so pspectra are not properly normalized", 
                                  verbose=verbose)
                    scalar = 1.0
                sclr_arr[i, j] = scalar

                # Loop over baseline pairs
                for k, blp in enumerate(bl_pairs):

                    # assign keys
                    if avg_group and fed_bl_group:
                        key1 = [(dsets[0],) + _blp[0] + (p,) for _blp in blp]
                        key2 = [(dsets[1],) + _blp[1] + (p,) for _blp in blp]
                    else:
                        key1 = (dsets[0],) + blp[0] + (p,)
                        key2 = (dsets[1],) + blp[1] + (p,)
                        
                    if verbose:
                        print("\n(bl1, bl2) pair: {}\npol: {}".format(blp, p))

                    # Set covariance weighting scheme for input data
                    if verbose: print("  Setting weight matrix for input data...")
                    self.set_R(input_data_weight)

                    # Build Fisher matrix
                    if input_data_weight == 'identity' and built_G:
                        # in this case, all Gv are the same, so skip if already built for this spw!
                        pass
                    else:
                        if verbose: print("  Building G...")
                        Gv = self.get_G(key1, key2)
                        built_G = True

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

                    # Get baseline keys
                    if avg_group and fed_bl_group:
                        bl1 = blp[0][0]
                        bl2 = blp[0][1]
                    else:
                        bl1 = blp[0]
                        bl2 = blp[1]

                    # append bls
                    bls_arr.extend([bl1, bl2])

                    # insert pspectra
                    pol_data.extend(pv.T)

                    # get weights
                    wgts1 = self.w(key1).T
                    wgts2 = self.w(key2).T

                    # get average of nsample across frequency axis, weighted by wgts
                    nsamp1 = np.sum(dset1.get_nsamples(bl1)[:, self.spw_range[0]:self.spw_range[1]] * wgts1, axis=1) / np.sum(wgts1, axis=1).clip(1, np.inf)
                    nsamp2 = np.sum(dset2.get_nsamples(bl2)[:, self.spw_range[0]:self.spw_range[1]] * wgts2, axis=1) / np.sum(wgts2, axis=1).clip(1, np.inf)

                    # take average of nsamp1 and nsamp2 and multiply by integration time [seconds] to get total integration
                    pol_ints.extend(np.mean([nsamp1, nsamp2], axis=0) * dset1.integration_time)

                    # combined weight is geometric mean
                    pol_wgts.extend(np.concatenate([wgts1[:, :, None], wgts2[:, :, None]], axis=2))

                    # insert time and blpair info only once
                    if i < 1 and j < 1:
                        # insert time info
                        inds1 = dset1.antpair2ind(*bl1)
                        inds2 = dset1.antpair2ind(*bl2)
                        time1.extend(dset1.time_array[inds1])
                        time2.extend(dset2.time_array[inds2])
                        lst1.extend(dset1.lst_array[inds1])
                        lst2.extend(dset2.lst_array[inds2])

                        # insert blpair info
                        blp_arr.extend(np.ones_like(inds1, np.int) * uvpspec._antnums_to_blpair(blp))

                # insert into data and wgts integrations dictionaries
                spw_data.append(pol_data)
                spw_wgts.append(pol_wgts)
                spw_ints.append(pol_ints)

            # insert into data and integration dictionaries
            spw_data = np.moveaxis(np.array(spw_data), 0, -1)
            spw_wgts = np.moveaxis(np.array(spw_wgts), 0, -1)
            spw_ints = np.moveaxis(np.array(spw_ints), 0, -1)
            data_array[i] = spw_data
            wgt_array[i] = spw_wgts
            integration_array[i] = spw_ints

        # fill uvp object
        uvp = uvpspec.UVPSpec()

        # fill meta-data
        uvp.time_1_array = np.array(time1)
        uvp.time_2_array = np.array(time2)
        uvp.time_avg_array = np.mean([uvp.time_1_array, uvp.time_2_array], axis=0)
        uvp.lst_1_array = np.array(lst1)
        uvp.lst_2_array = np.array(lst2)
        uvp.lst_avg_array = np.mean([np.unwrap(uvp.lst_1_array), np.unwrap(uvp.lst_2_array)], axis=0) % (2*np.pi)
        uvp.blpair_array = np.array(blp_arr)
        uvp.Nblpairs = len(np.unique(blp_arr))
        uvp.Ntimes = len(np.unique(time1))
        uvp.Nblpairts = len(time1)
        bls_arr = sorted(set(bls_arr))
        uvp.bl_array = np.array(map(lambda bl: uvp.antnums_to_bl(bl), bls_arr))
        antpos = dict(zip(dset1.antenna_numbers, dset1.antenna_positions))
        uvp.bl_vecs = np.array(map(lambda bl: antpos[bl[0]] - antpos[bl[1]], bls_arr))
        uvp.Nbls = len(uvp.bl_array)
        uvp.spw_array = np.array(spws)
        uvp.freq_array = np.array(freqs)
        uvp.dly_array = np.array(dlys)
        uvp.Nspws = len(np.unique(spws))
        uvp.Ndlys = len(np.unique(dlys))
        uvp.Nspwdlys = len(spws)
        uvp.Nfreqs = len(np.unique(freqs))
        uvp.pol_array = np.array(map(lambda p: uvutils.polstr2num(p), pol_arr))
        uvp.Npols = len(pol_arr)
        uvp.scalar_array = np.array(sclr_arr)
        uvp.channel_width = dset1.channel_width
        uvp.weighting = input_data_weight
        uvp.units = self.units(little_h=little_h)
        uvp.telescope_location = dset1.telescope_location
        uvp.history = dset1.history + dset2.history + history
        uvp.taper = taper
        uvp.norm = norm
        uvp.git_hash = version.git_hash
        if self.primary_beam is not None:
            uvp.cosmo_params = str(self.primary_beam.cosmo.get_params())
        if self.primary_beam is not None and hasattr(self.primary_beam, 'filename'): 
            uvp.beamfile = self.primary_beam.filename
        if hasattr(dset1.extra_keywords, 'filename'): uvp.filename1 = dset1.extra_keywords['filename']
        if hasattr(dset2.extra_keywords, 'filename'): uvp.filename2 = dset2.extra_keywords['filename']
        if hasattr(dset1.extra_keywords, 'tag'): uvp.tag1 = dset1.extra_keywords['tag']
        if hasattr(dset2.extra_keywords, 'tag'): uvp.tag2 = dset2.extra_keywords['tag']

        # fill data arrays
        uvp.data_array = data_array
        uvp.integration_array = integration_array
        uvp.wgt_array = wgt_array
        uvp.nsample_array = dict(map(lambda k: (k, np.ones_like(uvp.integration_array[k], np.float)), uvp.integration_array.keys()))

        # run check
        uvp.check()

        return uvp

    def rephase_to_dset(self, dset_index=0, inplace=True):
        """
        Rephase visibility data in self.dsets to the LST grid of dset[dset_index] 
        using hera_cal.utils.lst_rephase. 

        Each integration in all other dsets are phased to the center of the 
        corresponding LST bin (by index) in dset[dset_index].

        Will only phase if the dataset's phase type is 'drift'.

        Parameters
        ----------
        dset_index : int
            index of dataset in self.dset to phase other datasets to.

        inplace : bool, optional
            If True, edits data in dsets in-memory. Else, makes a copy of
            dsets, edits data in the copy and returns to user.

        Returns
        -------
        if inplace:
            return new_dsets
        else:
            return None
        """
        # run dataset validation
        self.validate_datasets()

        # assign dsets
        if inplace:
            dsets = self.dsets
        else:
            dsets = copy.deepcopy(self.dsets)

        # get LST grid we are phasing to
        lst_grid = []
        lst_array = dsets[dset_index].lst_array.ravel()
        for l in lst_array:
            if l not in lst_grid:
                lst_grid.append(l)
        lst_grid = np.array(lst_grid)

        # get polarization list
        pol_list = dsets[dset_index].polarization_array.tolist()

        # iterate over dsets
        for i, dset in enumerate(dsets):
            # don't rephase dataset we are using as our LST anchor
            if i == dset_index:
                # even though not phasing this dset, must set to match all other 
                # dsets due to phasing-check validation
                dset.phase_type = 'unknown'
                continue

            # skip if dataset is not drift phased
            if dset.phase_type != 'drift':
                print "skipping dataset {} b/c it isn't drift phased".format(i)

            # convert UVData to DataContainers. Note this doesn't make
            # a copy of the data
            (data, flgs, antpos, ants, freqs, times, lsts, 
             pols) = hc.io.load_vis(dset, return_meta=True)

            # make bls dictionary
            bls = dict(map(lambda k: (k, antpos[k[0]] - antpos[k[1]]), data.keys()))

            # Get dlst array
            dlst = lst_grid - lsts

            # get telescope latitude
            lat = dset.telescope_location_lat_lon_alt_degrees[0]

            # rephase
            hc.utils.lst_rephase(data, bls, freqs, dlst, lat=lat)

            # re-insert into dataset
            for j, k in enumerate(data.keys()):
                # get blts indices of basline
                indices = dset.antpair2ind(*k[:2])
                # get index in polarization_array for this polarization
                polind = pol_list.index(hc.io.polstr2num[k[-1]])
                # insert into dset
                dset.data_array[indices, 0, :, polind] = data[k]

            # set phasing in UVData object to unknown b/c there isn't a single
            # consistent phasing for the entire data set.
            dset.phase_type = 'unknown'

        if inplace is False:
            return dsets


def validate_bls(bls1, bls2, uvd1, uvd2, baseline_tol=1.0, verbose=True):
    """
    Validate baseline pairings between bls1 and bls2 are redundant within the 
    specified tolerance.

    Parameters
    ----------
    bls1 : list of baseline tuples, or list of bl-groups.
        See docstring of PSpecData.pspec() for details on format.

    bls2 : list of baseline tuples, or list of bl-groups.
        See docstring of PSpecData.pspec() for details on format.

    uvd1 : pyuvdata.UVData instance containing visibility data that bls1 will draw from

    uvd2 : pyuvdata.UVData instance containing visibility data that bls2 will draw from

    baseline_tol : float, distance tolerance for notion of baseline "redundancy" in meters

    verbose : bool, if True report feedback to stdout
    """
    # ensure both bls1 and bls2 are the same type
    if isinstance(bls1[0], tuple) and isinstance(bls1[0][0], (int, np.int)) \
        and isinstance(bls2[0], tuple) and isinstance(bls2[0][0], (int, np.int)):
        # bls1 and bls2 fed as list of bl tuples
        fed_bl_group = False

    elif isinstance(bls1[0], list) and isinstance(bls1[0][0], tuple) and isinstance(bls2[0], list) \
        and isinstance(bls2[0][0], tuple):
        # bls1 and bls2 fed as list of bl groups
        fed_bl_group = True
        assert len(bls1) == len(bls2), "if fed as list of bl groups, len(bls1) must equal len(bls2)"

    else:
        raise TypeError("bls1 and bls2 must both be fed as either a list of bl tuples, or a list of bl groups")

    # ensure uvd1 and uvd2 are UVData objects
    if isinstance(uvd1, pyuvdata.UVData) == False:
        raise TypeError("uvd1 must be a pyuvdata.UVData instance")
    if isinstance(uvd2, pyuvdata.UVData) == False:
        raise TypeError("uvd2 must be a pyuvdata.UVData instance")

    # get antenna position dictionary
    ap1, a1 = uvd1.get_ENU_antpos(pick_data_ants=True)
    ap2, a2 = uvd1.get_ENU_antpos(pick_data_ants=True)
    ap1 = dict(zip(a1, ap1))
    ap2 = dict(zip(a2, ap2))

    # ensure shared antenna keys match within tolerance
    shared = sorted(set(ap1.keys()) & set(ap2.keys()))
    for k in shared:
        assert np.linalg.norm(ap1[k] - ap2[k]) <= baseline_tol, "uvd1 and uvd2 don't agree on antenna positions within tolerance of {} m".format(baseline_tol)
    ap = ap1
    ap.update(ap2)

    # iterate through baselines and 1) check baselines crossed with each other are within tolerance
    # and 2) check baselines within a single group (if grouped) are within tolerance
    for i in range(len(bls1)):
        if fed_bl_group:
            # get baseline vectors for each bl in the i'th group
            blvecs1 = map(lambda bl: ap[bl[0]] - ap[bl[1]], bls1[i])
            blvecs2 = map(lambda bl: ap[bl[0]] - ap[bl[1]], bls2[i])
            # get maximum residual between all pairs
            resid = map(lambda p: np.linalg.norm(reduce(operator.sub, p)), itertools.combinations(blvecs1+blvecs2, 2))
            if np.max(np.abs(resid)) >= baseline_tol:
                raise_warning("baseline-pair residual(s) in the {}'th bl group exceed a bl tol of {} m".format(i, baseline_tol), verbose=verbose)
        else:
            blvec1 = ap[bls1[i][0]] - ap[bls1[i][1]]
            blvec2 = ap[bls2[i][0]] - ap[bls2[i][1]]
            if np.linalg.norm(blvec1 - blvec2) >= baseline_tol:
                raise_warning("bl1 {} and bl2 {} separation exceeds the bl tol of {} m".format(bls1[i], bls2[i], baseline_tol), verbose=verbose)


def raise_warning(warning, verbose=True):
    '''warning function'''
    if verbose:
        print(warning)

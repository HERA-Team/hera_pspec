import numpy as np
import aipy
import pyuvdata
#from .utils import hash, cov
from utils import hash, cov

class PSpecData(object):
    
    def __init__(self, dsets=[], wgts=[]):
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
        """
        self.clear_cov_cache() # Covariance matrix cache
        self.dsets = []; self.wgts = []
        self.Nfreqs = None
        
        # Store the input UVData objects if specified
        if len(dsets) > 0:
            self.add(dsets, wgts)
    
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
        
        # Store no. frequencies
        self.Nfreqs = self.dsets[0].Nfreqs
    
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
        keys : list, optional
            List of keys to remove from covariance matrix cache. If 'None', all 
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
        dset = key[0]; bl = key[1:]
        if self.wgts[dset] is not None:
            return self.wgts[dset].get_data(bl).T # FIXME: Transpose?
        else:
            # If weights were not specified, use the flags built in to the 
            # UVData dataset object
            return self.dsets[dset].get_flags(bl).astype(float).T # FIXME: Transpose?
    
    def C(self, k):
        """
        Estimate covariance matrices from the data.
        
        Parameters
        ----------
        k : tuple
            Tuple containing indices of dataset and baselines. The first item 
            specifies the index (ID) of a dataset in the collection, while 
            subsequent indices specify the baseline index, in _key2inds format.
        
        Returns
        -------
        C : array_like
            (Weighted) empirical covariance of data for baseline 'bl'.
        """
        # Set covariance if it's not in the cache
        if not self._C.has_key(k):
            self.set_C( {k : cov(self.x(k), self.w(k))} )
            self._Cempirical[k] = self._C[k]
        
        # Return cached covariance
        return self._C[k]
    
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
    
    def C_empirical(self, k):
        """
        Calculate empirical covariance from the data (with appropriate 
        weighting).
        
        Parameters
        ----------
        k : tuple
            Tuple containing indices of dataset and baselines. The first item 
            specifies the index (ID) of a dataset in the collection, while 
            subsequent indices specify the baseline index, in _key2inds format.
        
        Returns
        -------
        C_empirical : array_like
            Empirical covariance for the specified key.
        """
        # Check cache for empirical covariance
        if not self._Cempirical.has_key(k):
            self._Cempirical[k] = cov(self.x(k), self.w(k))
        return self._Cempirical[k]
    
    def I(self, k):
        """
        Return identity covariance matrix.
        
        Parameters
        ----------
        k : tuple
            Tuple containing indices of dataset and baselines. The first item 
            specifies the index (ID) of a dataset in the collection, while 
            subsequent indices specify the baseline index, in _key2inds format.
        
        Returns
        -------
        I : array_like
            Identity covariance matrix, dimension (Nfreqs, Nfreqs).
        """
        if not self._I.has_key(k):
            self._I[k] = np.identity(self.Nfreqs)
        return self._I[k]
        
    def iC(self, k):
        """
        Return the inverse covariance matrix, C^-1.
        
        Parameters
        ----------
        k : tuple
            Tuple containing indices of dataset and baselines. The first item 
            specifies the index (ID) of a dataset in the collection, while 
            subsequent indices specify the baseline index, in _key2inds format.
        
        Returns
        -------
        iC : array_like
            Inverse covariance matrix for specified dataset and baseline.
        """
        # Calculate inverse covariance if not in cache
        if not self._iC.has_key(k):
            C = self.C(k)
            U,S,V = np.linalg.svd(C.conj()) # conj in advance of next step
            
            # FIXME: Not sure what these are supposed to do
            #if self.lmin is not None: S += self.lmin # ensure invertibility
            #if self.lmode is not None: S += S[self.lmode-1]
            
            # FIXME: Is series of dot products quicker?
            self.set_iC({k:np.einsum('ij,j,jk', V.T, 1./S, U.T)})
        return self._iC[k]
    
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
    
    def q_hat(self, k1, k2, use_identity=True, use_fft=True):
        """
        Construct an unnormalized bandpower, q_hat, from a given pair of 
        visibility vectors. Returns the following quantity:
            
            \hat{q}_a = conj(x_1) C^-1 Q_a C^-1 x_2 (arXiv:1502.06016, Eq. 13)
        
        (Note the missing factor of 1/2.)
        N.B. The inverse covariance should already include the weights.
        
        Parameters
        ----------
        k1, k2 : tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors.
            
        use_identity : bool, optional
            Use the identity matrix to weight the data, instead of the 
            covariance matrix. Default: False.
            
        use_fft : bool, optional
            Whether to use a fast FFT summation trick to construct q_hat, or 
            a simpler brute-force matrix multiplication. The FFT method assumes 
            a delta-fn bin in delay space. Default: True.
        
        Returns
        -------
        q_hat : array_like
            Unnormalized bandpowers
        """
        # Whether to use look-up fn. for identity or inverse covariance matrix
        icov_fn = self.I if use_identity else self.iC
        
        # Calculate C^-1 x_1 and C^-1 x_2
        iC1x, iC2x = 0, 0
        for _k in k1: iC1x += np.dot(icov_fn(_k), self.x(_k))
        for _k in k2: iC2x += np.dot(icov_fn(_k), self.x(_k))
            
        # Whether to use FFT or slow direct method
        if use_fft:
            _iC1x = np.fft.fft(iC1x.conj(), axis=0)
            _iC2x = np.fft.fft(iC2x.conj(), axis=0)
            
            # FIXME: Should include window function (see get_Q)
            
            # Conjugated because inconsistent with pspec_cov_v003 otherwise
            # FIXME: Check that this should actually be conjugated
            return np.conj(  np.fft.fftshift(_iC1x, axes=0).conj() 
                           * np.fft.fftshift(_iC2x, axes=0) )
        else:
            # Slow method, used to explicitly cross-check FFT code
            q = []
            for i in xrange(self.Nfreqs):
                Q = self.get_Q(i, self.Nfreqs)
                iCQiC = np.einsum('ab,bc,cd', iC1.T.conj(), Q, iC2) # C^-1 Q C^-1
                qi = np.sum(self.x(k1).conj() * np.dot(iCQiC, self.x(k2)), axis=0)
                q.append(qi)
            return np.array(q)
    
    def get_F(self, k1, k2, use_identity=False, true_fisher=False):
        """
        Calculate the Fisher matrix for the power spectrum bandpowers, p_alpha. 
        The Fisher matrix is defined as:
        
            F_ab = 1/2 Tr [C^-1 Q_a C^-1 Q_b] (arXiv:1502.06016, Eq. 17)
        
        Parameters
        ----------
        k1, k2 : tuples
            Tuples containing indices of dataset and baselines for the two 
            input datavectors.
        
        use_identity : bool, optional
            Use the identity matrix to weight the data, instead of the 
            covariance matrix. Default: False.
        
        true_fisher : bool, optional
            Whether to calculate the "true" Fisher matrix, or the "effective" 
            matrix s.t. W=MF and p=Mq. Default: False. (FIXME)
        
        Returns
        -------
        F : array_like, complex
            Fisher matrix, with dimensions (Nfreqs, Nfreqs).
        """
        F = np.zeros((self.Nfreqs, self.Nfreqs), dtype=np.complex)
        
        # Whether to use look-up fn. for identity or inverse covariance matrix
        icov_fn = self.I if use_identity else self.iC
        
        iC1, iC2 = 0, 0
        for _k in k1: iC1 += icov_fn(_k)
        for _k in k2: iC2 += icov_fn(_k)
        
        # Multiply terms to get the true or effective Fisher matrix
        # FIXME: I think effective <=> true have been mixed up here
        if true_fisher:
            # This is for the "true" Fisher matrix
            # FIXME: What is this for?
            CE1, CE2 = {}, {}
            Cemp1, Cemp2 = self.I(k1), self.I(k2)
            
            for ch in xrange(self.Nfreqs):
                Q = self.get_Q(ch, self.Nfreqs)
                # C1 Cbar1^-1 Q Cbar2^-1; C2 Cbar2^-1 Q Cbar1^-1
                CE1[ch] = np.dot(Cemp1, np.dot(iC1, np.dot(Q, iC2)))
                CE2[ch] = np.dot(Cemp2, np.dot(iC2, np.dot(Q, iC1)))
            
            for i in xrange(self.Nfreqs):
                for j in xrange(self.Nfreqs):
                    F[i,j] += np.einsum('ij,ji', CE1[i], CE2[j]) # C E C E
        else:
            # This is for the "effective" matrix s.t. W=MF and p=Mq
            iCQ1, iCQ2 = {}, {}
            
            for ch in xrange(self.Nfreqs): # this loop is nchan^3
                Q = self.get_Q(ch, self.Nfreqs)
                iCQ1[ch] = np.dot(iC1, Q) #C^-1 Q
                iCQ2[ch] = np.dot(iC2, Q) #C^-1 Q
            
            for i in xrange(self.Nfreqs): # this loop goes as nchan^4
                for j in xrange(self.Nfreqs):
                    F[i,j] += np.einsum('ij,ji', iCQ1[i], iCQ2[j]) #C^-1 Q C^-1 Q 
        return F
    
    def get_MW(self, F, mode='F^-1'):
        """
        Construct the normalization matrix M and window function matrix W for 
        the power spectrum estimator. These are defined through Eqs. 14-16 of 
        arXiv:1502.06016:
            
            \hat{p} = M \hat{q}
            \hat{p} = W p
            W = M F,
        
        where p is the true band power and F is the Fisher matrix. Several 
        choices for M are supported:
        
            'F^-1':   Set M = F^-1, the (pseudo)inverse Fisher matrix.
            'F^-1/2': Set M = F^-1/2, the root-inverse Fisher matrix (using SVD).
            'I':      Set M = I, the identity matrix.
            'L^-1':   Set M = L^-1, Cholesky decomposition.
        
        Parameters
        ----------
        F : array_like or dict of array_like
            Fisher matrix for the bandpowers, with dimensions (Nfreqs, Nfreqs).
            If a dict is specified, M and W will be calculated for each F 
            matrix in the dict.
            
        mode : str, optional
            Definition to use for M. Must be one of the options listed above. 
            Default: 'F^-1'.
        
        Returns
        -------
        M : array_like
            Normalization matrix, M. (If F was passed in as a dict, a dict of 
            array_like will be returned.)
        
        W : array_like
            Window function matrix, W. (If F was passed in as a dict, a dict of 
            array_like will be returned.)
        """
        # Recursive case, if many F's were specified at once
        if type(F) is dict:
            M,W = {}, {}
            for key in F: M[key],W[key] = self.get_MW(F[key], mode=mode)
            return M, W
        
        # Check that mode is supported
        modes = ['F^-1', 'F^-1/2', 'I', 'L^-1']
        assert(mode in modes)
        
        # Build M matrix according to specified mode
        if mode == 'F^-1':
            M = np.linalg.pinv(F, rcond=1e-12)
            #U,S,V = np.linalg.svd(F)
            #M = np.einsum('ij,j,jk', V.T, 1./S, U.T)
            
        elif mode == 'F^-1/2':
            U,S,V = np.linalg.svd(F)
            M = np.einsum('ij,j,jk', V.T, 1./np.sqrt(S), U.T)
            
        elif mode == 'I':
            M = np.identity(F.shape[0], dtype=F.dtype)
            
        else:
            # Cholesky decomposition to get M (XXX: Needs generalizing)
            order = np.array([10, 11, 9, 12, 8, 20, 0, 
                              13, 7, 14, 6, 15, 5, 16, 
                              4, 17, 3, 18, 2, 19, 1])
            iorder = np.argsort(order)
            F_o = np.take(np.take(F,order, axis=0), order, axis=1)
            L_o = np.linalg.cholesky(F_o)
            U,S,V = np.linalg.svd(L_o.conj())
            M_o = np.dot(np.transpose(V), np.dot(np.diag(1./S), np.transpose(U)))
            M = np.take(np.take(M_o, iorder, axis=0), iorder, axis=1)
        
        # Calculate (normalized) W given Fisher matrix and choice of M
        W = np.dot(M, F)
        norm = W.sum(axis=-1); norm.shape += (1,)
        M /= norm; W = np.dot(M, F)
        return M, W
    
    def get_Q(self, mode, n_k, window='none'):
        """
        Response of the covariance to a given bandpower, dC / dp_alpha. 
        Assumes that Q will operate on a visibility vector in frequency space.
        
        Parameters
        ----------
        mode : int
            Central wavenumber (index) of the bandpower, p_alpha.
            
        n_k : int
            Number of k bins that will be .
            
        window : str, optional
            Type of window function to use. Valid options are any window 
            function supported by aipy.dsp.gen_window(). Default: 'none'.
        
        Returns
        -------
        Q : array_like
            Response matrix for bandpower p_alpha.
        """
        _m = np.zeros((n_k,), dtype=np.complex)
        _m[mode] = 1. # delta function at specified delay mode
        
        # FFT to convert to frequency domain
        m = np.fft.fft(np.fft.ifftshift(_m)) * aipy.dsp.gen_window(n_k, window)
        Q = np.einsum('i,j', m, m.conj()) # dot it with its conjugate
        return Q

    def p_hat(self, M, q):
        """
        Optimal estimate of bandpower p_alpha, defined as p_hat = M q_hat.
        
        Parameters
        ----------
        M : array_like or dict of array_like
            Normalization matrix, M. If different M's were chosen for different 
            times, 'M' can be passed as a dict where the keys are (FIXME).
            
        q : array_like
            Unnormalized bandpowers, \hat{q}.
        
        Returns
        -------
        p_hat : array_like
            Optimal estimate of bandpower, \hat{p}.
        """
        if type(M) is dict:
            # Specified different M's for different times
            (k1, m1, k2, m2) = M.keys()[0]
            
            w1, w2 = self.w(k1), self.w(k2)
            m1s = [hash(w1[:,i]) for i in xrange(w1.shape[1])]
            m2s = [hash(w2[:,i]) for i in xrange(w2.shape[1])]
            
            inds = {}
            for i, (m1,m2) in enumerate(zip(m1s, m2s)):
                inds[(k1,m1,k2,m2)] = inds.get((k1,m1,k2,m2),[]) + [i]
            
            p = np.zeros_like(q)
            for key in inds:
                qi = q[:,inds[key]]
                p[:,inds[key]] = np.dot(M[key], qi)
            return p
        else:
            # Simple case where M is the same for all times
            return np.dot(M, q)

    def pspec(self, keys, weights='none'):
        """
        Estimate the power spectrum from the datasets contained in this object, 
        using the optimal quadratic estimator (OQE) from arXiv:1502.06016.
        
        Parameters
        ----------
        keys : list
            TODO.
            
        weights : str, optional
            String specifying how to choose the normalization matrix, M. See 
            the 'mode' argument of get_MW() for options.
        
        Returns
        -------
        pspec : list
            Optimal quadratic estimate of the power spectrum for the datasets 
            and baselines specified in 'keys'.
        """
        #FIXME: Define sensible grouping behaviors.
        #FIXME: Check that requested keys exist in all datasets
        
        # Validate the input data to make sure it's sensible
        self.validate_datasets()
        
        pvs = []
        for k, key1 in enumerate(keys):
            if k == 1 and len(keys) == 2: 
                # NGPS = 1 (skip 'odd' with 'even' if we already did 'even' 
                # with 'odd')
                continue
            
            for key2 in keys[k:]:
                if len(keys) > 2 and (key1[0] == key2[0] or key1[1] == key2[1]):
                    # NGPS > 1
                    continue
                if key1[0] == key2[0]: # don't do 'even' with 'even', for example
                    continue
                else:
                    Fv = self.get_F(key1, key2)
                    qv = self.q_hat(key1, key2)
                    Mv, Wv = self.get_MW(Fv, mode=weights)  
                    pv = self.p_hat(Mv, qv)
                    pvs.append(pv)
        return pvs
        

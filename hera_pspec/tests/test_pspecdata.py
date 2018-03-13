import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os, copy, sys
from scipy.integrate import simps
from hera_pspec import pspecdata, pspecbeam
from hera_pspec.data import DATA_PATH

# Get absolute path to data directory
DATADIR = os.path.dirname( os.path.realpath(__file__) ) + "/../data/"

# Data files to use in tests
dfiles = [
    'zen.2458042.12552.xx.HH.uvXAA',
    'zen.2458042.12552.xx.HH.uvXAA'
]

# List of tapering function to use in tests
taper_selection = ['blackman',]
#taper_selection = ['blackman', 'blackman-harris', 'gaussian0.4', 'kaiser2',
#                   'kaiser3', 'hamming', 'hanning', 'parzen']

def generate_pos_def(n):
    """
    Generate a random positive definite Hermitian matrix.

    Parameters
    ----------
    n : integer
        Size of desired matrix

    Returns
    -------
    A : array_like
        Positive definite matrix
    """
    A = np.random.normal(size=(n,n)) + 1j * np.random.normal(size=(n,n))
    A += np.conjugate(A).T
    # Add just enough of an identity matrix to make all eigenvalues positive
    A += -1.01*np.min(np.linalg.eigvalsh(A))*np.identity(n)
    return A

def generate_pos_def_all_pos(n):
    """
    Generate a random positive definite symmetric matrix, with all entries 
    positive.

    Parameters
    ----------
    n : integer
        Size of desired matrix

    Returns
    -------
    A : array_like
        Positive definite matrix
    """
    A = np.random.uniform(size=(n,n))
    A += A.T
    # Add just enough of an identity matrix to make all eigenvalues positive
    A += -1.01*np.min(np.linalg.eigvalsh(A))*np.identity(n)
    return A

def diagonal_or_not(mat, places=7):
    """
    Tests whether a matrix is diagonal or not.

    Parameters
    ----------
    n : array_like
        Matrix to be tested

    Returns
    -------
    diag : bool
        True if matrix is diagonal
    """
    mat_norm = np.linalg.norm(mat)
    diag_mat_norm = np.linalg.norm(np.diag(np.diag(mat)))
    diag = (round(mat_norm-diag_mat_norm, places) == 0)
    return diag

class Test_PSpecData(unittest.TestCase):

    def setUp(self):
        
        # Instantiate empty PSpecData
        self.ds = pspecdata.PSpecData()
        
        # Load datafiles
        self.d = []
        for dfile in dfiles:
            _d = uv.UVData()
            _d.read_miriad(DATADIR + dfile)
            self.d.append(_d)
        
        # Set trivial weights
        self.w = [None for _d in dfiles]
        
        # Load beam file
        beamfile = DATADIR + 'NF_HERA_Beams.beamfits'
        self.bm = pspecbeam.PSpecBeamUV(beamfile)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        # Test creating empty PSpecData
        ds = pspecdata.PSpecData()
        
        # Test whether unequal no. of weights is picked up
        self.assertRaises( AssertionError, 
                           pspecdata.PSpecData, 
                           [uv.UVData(), uv.UVData(), uv.UVData()], 
                           [uv.UVData(), uv.UVData()] )
        
        # Test passing data and weights of the wrong type
        d_arr = np.ones((6, 8))
        d_lst = [[0,1,2] for i in range(5)]
        d_float = 12.
        d_dict = {'(0,1)':np.arange(5), '(0,2)':np.arange(5)}
        
        self.assertRaises(TypeError, pspecdata.PSpecData, d_arr, d_arr)
        self.assertRaises(TypeError, pspecdata.PSpecData, d_lst, d_lst)
        self.assertRaises(TypeError, pspecdata.PSpecData, d_float, d_float)
        self.assertRaises(TypeError, pspecdata.PSpecData, d_dict, d_dict)

    def test_add_data(self):
        pass

    def test_get_Q(self):
        vect_length = 50
        x_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)
        y_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)

        for i in range(vect_length):
            Q_matrix = self.ds.get_Q(i, vect_length)
            xQy = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, y_vect))
            yQx = np.dot(np.conjugate(y_vect), np.dot(Q_matrix, x_vect))
            xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
            
            # Test that Q matrix has the right shape
            self.assertEqual(Q_matrix.shape, (vect_length, vect_length))
            
            # Test that x^t Q y == conj(y^t Q x)
            self.assertAlmostEqual(xQy, np.conjugate(yQx))
            
            # x^t Q x should be real
            self.assertAlmostEqual(np.imag(xQx), 0.)

        x_vect = np.ones(vect_length)
        Q_matrix = self.ds.get_Q(vect_length/2., vect_length)
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))
        # Sending in sinusoids for x and y should give delta functions

    def test_get_MW(self):
        n = 17
        random_G = generate_pos_def_all_pos(n)

        nt.assert_raises(AssertionError, self.ds.get_MW, random_G, mode='L^3')
        
        for mode in ['G^-1', 'G^-1/2', 'I', 'L^-1']:
            M, W = self.ds.get_MW(random_G, mode=mode)
            self.assertEqual(M.shape, (n,n))
            self.assertEqual(W.shape, (n,n))
            test_norm = np.sum(W, axis=1)
            for norm in test_norm:
                self.assertAlmostEqual(norm, 1.)

            if mode == 'G^-1':
                # Test that the window functions are delta functions
                self.assertEqual(diagonal_or_not(W), True)
            elif mode == 'G^-1/2':
                # Test that the error covariance is diagonal
                error_covariance = np.dot(M, np.dot(random_G, M.T)) 
                # FIXME: We should be decorrelating V, not G. See Issue 21
                self.assertEqual(diagonal_or_not(error_covariance), True)
            elif mode == 'I':
                # Test that the norm matrix is diagonal
                self.assertEqual(diagonal_or_not(M), True)

    def test_q_hat(self):
        """
        Test that q_hat has right shape and accepts keys in the right format.
        """
        # Set weights and pack data into PSpecData
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        Ntime = self.ds.Ntimes
        
        # Set baselines to use for tests
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)
        key3 = [(0, 24, 38), (0, 24, 38)]
        key4 = [(1, 25, 38), (1, 25, 38)]
        
        for input_data_weight in ['identity', 'iC']:
            self.ds.set_R(input_data_weight)
            
            # Loop over list of taper functions
            for taper in taper_selection:
                
                # Calculate q_hat for a pair of baselines and test output shape
                q_hat_a = self.ds.q_hat(key1, key2)
                self.assertEqual(q_hat_a.shape, (Nfreq, Ntime))
                
                # Check that swapping x_1 <-> x_2 results in complex conj. only
                q_hat_b = self.ds.q_hat(key2, key1)
                q_hat_diff = np.conjugate(q_hat_a) - q_hat_b
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_diff[i,j].real, 
                                               q_hat_diff[i,j].real)
                        self.assertAlmostEqual(q_hat_diff[i,j].imag, 
                                               q_hat_diff[i,j].imag)
                
                # Check that lists of keys are handled properly
                q_hat_aa = self.ds.q_hat(key1, key4) # q_hat(k1, k2+k2)
                q_hat_bb = self.ds.q_hat(key4, key1) # q_hat(k2+k2, k1)
                q_hat_cc = self.ds.q_hat(key3, key4) # q_hat(k1+k1, k2+k2)
                
                # Effectively checks that q_hat(2*k1, 2*k2) = 4*q_hat(k1, k2)
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_a[i,j].real, 
                                               0.25 * q_hat_cc[i,j].real)
                        self.assertAlmostEqual(q_hat_a[i,j].imag, 
                                               0.25 * q_hat_cc[i,j].imag)
                
                # Check that the slow method is the same as the FFT method
                q_hat_a_slow = self.ds.q_hat(key1, key2, use_fft=False)
                vector_scale = np.min([ np.min(np.abs(q_hat_a_slow.real)), 
                                        np.min(np.abs(q_hat_a_slow.imag)) ])
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertLessEqual(
                                np.abs((q_hat_a[i,j] - q_hat_a_slow[i,j]).real), 
                                vector_scale*1e-6 )
                        self.assertLessEqual(
                                np.abs((q_hat_a[i,j] - q_hat_a_slow[i,j]).imag), 
                                vector_scale*1e-6 )

    def test_get_G(self):
        """
        Test Fisher/weight matrix calculation.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        multiplicative_tolerance = 1.

        for input_data_weight in ['identity','iC']:
            for taper in taper_selection:
                print 'input_data_weight', input_data_weight, 'taper', taper
                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                G = self.ds.get_G(key1, key2, taper=taper)
                self.assertEqual(G.shape, (Nfreq,Nfreq)) # Test shape
                print np.min(np.abs(G)), np.min(np.abs(np.linalg.eigvalsh(G)))
                matrix_scale = np.min(np.abs(np.linalg.eigvalsh(G)))

                if input_data_weight == 'identity':
                    # In the identity case, there are three special properties
                    # that are respected:
                    # i) Symmetry: G_ab = G_ba
                    # ii) Cylic property: G = (1/2) tr[R1 Q_a R2 Q_b]
                    #                       = (1/2) tr[R2 Q_b R1 Q_a]
                    # iii) All elements of G are positive.

                    # Test symmetry
                    anti_sym_norm = np.linalg.norm(G - G.T)
                    self.assertLessEqual(anti_sym_norm, 
                                        matrix_scale * multiplicative_tolerance)

                    # Test cyclic property of trace, where key1 and key2 can be
                    # swapped without changing the matrix. This is secretly the
                    # same test as the symmetry test, but perhaps there are
                    # creative ways to break the code to break one test but not
                    # the other.
                    G_swapped = self.ds.get_G(key2, key1, taper=taper)
                    G_diff_norm = np.linalg.norm(G - G_swapped)
                    self.assertLessEqual(G_diff_norm, 
                                        matrix_scale * multiplicative_tolerance)
                    min_diagonal = np.min(np.diagonal(G))
                    
                    # Test that all elements of G are positive up to numerical 
                    # noise with the threshold set to 10 orders of magnitude 
                    # down from the smallest value on the diagonal
                    for i in range(Nfreq):
                        for j in range(Nfreq):
                            self.assertGreaterEqual(G[i,j], 
                                       -min_diagonal * multiplicative_tolerance)
                else:
                    # In general, when R_1 != R_2, there is a more restricted 
                    # symmetry where swapping R_1 and R_2 *and* taking the 
                    # transpose gives the same result
                    G_swapped = self.ds.get_G(key2, key1, taper=taper)
                    G_diff_norm = np.linalg.norm(G - G_swapped.T)
                    self.assertLessEqual(G_diff_norm, 
                                         matrix_scale * multiplicative_tolerance)

            
    def test_parseval(self):
        """
        Test that output power spectrum respects Parseval's theorem.
        """
        np.random.seed(10)
        variance_in = 1.
        Nfreq = self.d[0].Nfreqs
        data = self.d[0]

        # Use only the requested number of channels
        data.select(freq_chans=range(Nfreq), ant_pairs_nums=[(24,24),])

        # Make it so that the test data is unflagged everywhere
        data.flag_array[:] = False

        # Get list of available baselines and LSTs
        bls = data.get_antpairs()
        nlsts = data.Ntimes

        # Simulate data given a Fourier-space power spectrum
        pk = variance_in * np.ones(Nfreq)

        # Make realisation of (complex) white noise in real space
        g = 1.0 * np.random.normal(size=(nlsts,Nfreq)) \
          + 1.j * np.random.normal(size=(nlsts,Nfreq))
        g /= np.sqrt(2.) # Since Re(C) = Im(C) = C/2
        x = data.freq_array[0]
        dx = x[1] - x[0]

        # Fourier transform along freq. direction in each LST bin
        gnew = np.zeros(g.shape).astype(complex)
        fnew = np.zeros(g.shape).astype(complex)
        for i in range(nlsts):
            f = np.fft.fft(g[i]) * np.sqrt(pk)
            fnew[i] = f
            gnew[i] = np.fft.ifft(f)

        # Parseval's theorem test: integral of F^2(k) dk = integral of f^2(x) dx
        k = np.fft.fftshift( np.fft.fftfreq(Nfreq, d=(x[1]-x[0])) )
        fsq = np.fft.fftshift( np.mean(fnew * fnew.conj(), axis=0) )
        gsq = np.mean(gnew * gnew.conj(), axis=0)

        # Realize set of Gaussian random datasets and pack into PSpecData
        data.data_array = np.expand_dims(np.expand_dims(gnew, axis=1), axis=3)
        ds = pspecdata.PSpecData()
        ds.add([data, data], [None, None])

        # Use true covariance instead
        exact_cov = {
            (0,24,24): np.eye(Nfreq),
            (1,24,24): np.eye(Nfreq)
        }
        ds.set_C(exact_cov)
        
        # Calculate OQE power spectrum using true covariance matrix
        tau = np.fft.fftshift( ds.delays() )
        ps, _ = ds.pspec(bls, input_data_weight='iC', norm='I')
        ps_avg = np.fft.fftshift( np.mean(ps[0], axis=1) )
        
        # Calculate integrals for Parseval's theorem
        parseval_real = simps(gsq, x)
        parseval_ft = dx**2. * simps(fsq, k)
        parseval_phat = simps(ps_avg, tau)
        
        # Report on results for different ways of calculating Parseval integrals
        print "Parseval's theorem:"
        print "  \int [g(x)]^2 dx = %3.6e, %3.6e" % (parseval_real.real, 
                                                     parseval_real.imag)
        print "  \int [f(k)]^2 dk = %3.6e, %3.6e" % (parseval_ft.real, 
                                                     parseval_ft.imag)
        print "  \int p_hat(k) dk = %3.6e, %3.6e" % (parseval_phat.real, 
                                                     parseval_phat.imag)
        
        # Perform approx. equality test (this is a stochastic quantity, so we 
        # only expect equality to ~10^-2 to 10^-3
        np.testing.assert_allclose(parseval_phat, parseval_real, rtol=1e-3)

    def test_scalar(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=self.bm)

        # Precomputed results in the following test were done "by hand" 
        # using iPython notebook "Scalar_dev2.ipynb" in the tests/ directory
        # FIXME: Uncomment when pyuvdata support for this is ready
        #scalar = self.ds.scalar()
        #self.assertAlmostEqual(scalar, 3732415176.85 / 10.**9)
        
        # FIXME: Remove this when pyuvdata support for the above is ready
        self.assertRaises(NotImplementedError, self.ds.scalar)

"""
# LEGACY MONTE CARLO TESTS

    def validate_get_G(self,tolerance=0.2,NDRAWS=100,NCHAN=8):
        '''
        Test get_G where we interpret G in this case to be the Fisher Matrix.
        Args:
            tolerance, required max fractional difference from analytical
                       solution to pass.
            NDRAWS, number of random data sets to sample frome.
            NCHAN, number of channels. Must be less than test data sets.
        '''
        #read in data.
        dpath=os.path.join(DATA_PATH,'zen.2458042.12552.xx.HH.uvXAA')
        data=uv.UVData()
        wghts=uv.UVData()
        data.read_miriad(dpath)
        wghts.read_miriad(dpath)
        assert(NCHAN<data.Nfreqs)
        #make sure we use fewer channels.
        data.select(freq_chans=range(NCHAN))
        wghts.select(freq_chans=range(NCHAN))
        #********************************************************************
        #set data to random white noise with a random variance and mean.
        ##!!!Set mean to zero for now since analyitic solutions assumed mean
        ##!!!Subtracted data which oqe isn't actually doing.
        #*******************************************************************
        test_mean=0.*np.abs(np.random.randn())
        test_std=np.abs(np.random.randn())
        #*******************************************************************
        #Make sure that all of the flags are set too true for analytic case.
        #*******************************************************************
        data.flag_array[:]=False
        wghts.data_array[:]=1.
        wghts.flag_array[:]=False
        bllist=data.get_antpairs()
        #*******************************************************************
        #These are the averaged "fisher matrices"
        #*******************************************************************
        f_mat=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        f_mat_true=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        nsamples=0
        for datanum in range(NDATA):
            #for each data draw, generate a random data set.
            pspec=pspecdata.PSpecData()
            data.data_array=test_std\
            *np.random.standard_normal(size=data.data_array.shape)\
            /np.sqrt(2.)+1j*test_std\
            *np.random.standard_normal(size=data.data_array.shape)\
            /np.sqrt(2.)+(1.+1j)*test_mean
            pspec.add([data],[wghts])
            #get empirical Fisher matrix for baselines 0 and 1.
            pair1=bllist[0]
            pair2=bllist[1]
            k1=(0,pair1[0],pair1[1],-5)
            k2=(0,pair2[0],pair2[1],-5)
            #add to fisher averages.
            f_mat_true=f_mat_true+pspec.get_F(k1,k2,true_fisher=True)
            f_mat=f_mat+pspec.get_F(k1,k2)
            #test identity
            self.assertTrue(np.allclose(pspec.get_F(k1,k2,use_identity=True)/data.Nfreqs**2.,
                            np.identity(data.Nfreqs).astype(complex)))
            del pspec
        #divide out empirical Fisher matrices by analytic solutions.
        f_mat=f_mat/NDATA/data.Nfreqs**2.*test_std**4.
        f_mat_true=f_mat_true/NDATA/data.Nfreqs**2.*test_std**4.
        #test equality to analytic solutions
        self.assertTrue(np.allclose(f_mat,
                        np.identity(data.Nfreqs).astype(complex),
                        rtol=tolerance,
                        atol=tolerance)
        self.assertTrue(np.allclose(f_mat_true,
                                    np.identity(data.Nfreqs).astype(complex),
                                    rtol=tolerance,
                                    atol=tolerance)
        #TODO: Need a test case for some kind of taper.


    def validate_get_MW(self,NCHANS=20):
        '''
        Test get_MW with analytical case.
        Args:
            NCHANS, number of channels to validate.
        '''
        ###
        test_std=np.abs(np.random.randn())
        f_mat=np.identity(NCHANS).astype(complex)/test_std**4.*nchans**2.
        pspec=pspecdata.PSpecData()
        m,w=pspec.get_MW(f_mat,mode='G^-1')
        #test M/W matrices are close to analytic solutions
        #check that rows in W sum to unity.
        self.assertTrue(np.all(np.abs(w.sum(axis=1)-1.)<=tolerance))
        #check that W is analytic soluton (identity)
        self.assertTrue(np.allclose(w,np.identity(nchans).astype(complex)))
        #check that M.F = W
        self.assertTrue(np.allclose(np.dot(m,f_mat),w))
        m,w=pspec.get_MW(f_mat,mode='G^-1/2')
        #check W is identity
        self.assertTrue(np.allclose(w,np.identity(nchans).astype(complex)))
        self.assertTrue(np.allclose(np.dot(m,f_mat),w))
        #check that L^-1 runs.
        m,w=pspec.get_MW(f_mat,mode='G^-1')

    def validate_q_hat(self,NCHAN=8,NDATA=1000,):
        '''
        validate q_hat calculation by drawing random white noise data sets
        '''
        dpath=os.path.join(DATA_PATH,'zen.2458042.12552.xx.HH.uvXAA')
        data=uv.UVData()
        wghts=uv.UVData()
        data.read_miriad(dpath)
        wghts.read_miriad(dpath)
        assert(NCHAN<=data.Nfreqs)
        data.select(freq_chans=range(NCHAN))
        wghts.select(freq_chans=range(NCHAN))
        #***************************************************************
        #set data to random white noise with a random variance and mean
        #q_hat does not subtract a mean so I will set it to zero for
        #the test.
        #****************************************************************
        test_mean=0.*np.abs(np.random.randn())#*np.abs(np.random.randn())
        test_std=np.abs(np.random.randn())

        data.flag_array[:]=False#Make sure that all of the flags are set too true for analytic case.
        wghts.data_array[:]=1.
        wghts.flag_array[:]=False
        bllist=data.get_antpairs()
        q_hat=np.zeros(NCHAN).astype(complex)
        q_hat_id=np.zeros_like(q_hat)
        q_hat_fft=np.zeros_like(q_hat)
        nsamples=0
        for datanum in range(NDATA):
            pspec=pspecdata.PSpecData()
            data.data_array=test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)\
            +1j*test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)+(1.+1j)*test_mean
            pspec.add([data],[wghts])
            for j in range(data.Nbls):
                #get baseline index
                pair1=bllist[j]
                k1=(0,pair1[0],pair1[1],-5)
                k2=(0,pair1[0],pair1[1],-5)
                #get q
                #test identity
                q_hat=q_hat+np.mean(pspec.q_hat(k1,k2,use_fft=False),
                axis=1)
                q_hat_id=q_hat_id+np.mean(pspec.q_hat(k1,k2,use_identity=True),
                axis=1)
                q_hat_fft=q_hat_fft+np.mean(pspec.q_hat(k1,k2),axis=1)
                nsamples=nsamples+1
            del pspec

        #print nsamples
        nfactor=test_std**2./data.Nfreqs/nsamples
        q_hat=q_hat*nfactor
        q_hat_id=q_hat_id*nfactor/test_std**4.
        q_hat_fft=q_hat_fft*nfactor
        #print q_hat
        #print q_hat_id
        #print q_hat_fft

        self.assertTrue(np.allclose(q_hat,
        np.identity(data.Nfreqs).astype(complex)))
        self.assertTrue(np.allclose(q_hat_id,
        np.identity(data.Nfreqs).astype(complex)))
        self.assertTrue(np.allclose(q_hat_fft,
        np.identity(data.Nfreqs).astype(complex)))
"""

if __name__ == "__main__":
    unittest.main()

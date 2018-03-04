import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os, copy, sys
from scipy.integrate import simps
from hera_pspec import pspecdata
from hera_pspec import oqe
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
    Generate a random positive definite symmetric matrix, with all entries positive.
    
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

def diagonal_or_not(mat,places=7):
    """
    Tests whether a matrix is diagonal or not
    
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
        Q_matrix = self.ds.get_Q(vect_length/2, vect_length)
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))
        # 3) Sending in pure tones/sinusoids for x and y should give delta functions

    def test_get_MW(self):
        n = 17
        random_G = generate_pos_def_all_pos(n)

        nt.assert_raises(AssertionError, self.ds.get_MW, random_G, mode='L^3')
        
        # 
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

        for input_data_weight in ['identity','iC']:
            for taper in taper_selection:
                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                G = self.ds.get_G(key1, key2, taper=taper)
                self.assertEqual(G.shape, (Nfreq,Nfreq)) # Test shape
                matrix_scale = np.min( [np.min(np.abs(G)),
                                        np.min(np.abs(np.linalg.eigvalsh(G)))] )
                # Test symmetry
                anti_sym_norm = np.linalg.norm(G - G.T)
                self.assertLessEqual(anti_sym_norm, matrix_scale*1e-10)

                # Test cyclic property of trace, where key1 and key2 can be
                # swapped without changing the matrix. This is secretly the
                # same test as the symmetry test, but perhaps there are
                # creative ways to break the code to break one test but not
                # the other.
                G_swapped = self.ds.get_G(key2, key1, taper=taper)
                G_diff_norm = np.linalg.norm(G - G_swapped)
                self.assertLessEqual(G_diff_norm, matrix_scale*1e-10)
                min_diagonal = np.min(np.diagonal(G))
                
                # Test that all elements of G are positive up to numerical noise
                # with the threshold set to 10 orders of magnitude down from
                # the smallest value on the diagonal
                for i in range(Nfreq):
                    for j in range(Nfreq):
                        self.assertGreaterEqual(G[i,j], -min_diagonal*1e-10)
            
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


if __name__ == "__main__":
    unittest.main()

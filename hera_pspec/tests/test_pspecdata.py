import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os
import copy
import sys
from hera_pspec import pspecdata, pspecbeam
from hera_pspec import oqe
from hera_pspec.data import DATA_PATH
import pyuvdata as uv
import pylab as plt

# Get absolute path to data directory
DATADIR = os.path.dirname( os.path.realpath(__file__) ) + "/../data/"

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

taper_selection = ['blackman'] #,'blackman-harris','gaussian0.4','kaiser2',\
                            #'kaiser3','hamming','hanning','parzen']

class Test_DataSet(unittest.TestCase):

    def setUp(self):
        self.ds = pspecdata.PSpecData()

        beamfile = DATADIR + 'NF_HERA_Beams.beamfits'
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        # Test creating empty DataSet
        ds = pspecdata.PSpecData()
        pass

    def test_add_data(self):
        pass

    def test_get_Q(self):
        vect_length = 50
        x_vect = np.random.normal(size=vect_length) + 1j * np.random.normal(size=vect_length)
        y_vect = np.random.normal(size=vect_length) + 1j * np.random.normal(size=vect_length)

        for i in range(vect_length):
            Q_matrix = self.ds.get_Q(i, vect_length)
            xQy = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,y_vect))
            yQx = np.dot(np.conjugate(y_vect),np.dot(Q_matrix,x_vect))
            xQx = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,x_vect))
            self.assertEqual(Q_matrix.shape,(vect_length,vect_length)) # Q matrix right shape
            self.assertAlmostEqual(xQy,np.conjugate(yQx)) # x^t Q y == conj(y^t Q x)
            self.assertAlmostEqual(np.imag(xQx), 0.) # x^t Q x should be real

        x_vect = np.ones(vect_length)
        Q_matrix = self.ds.get_Q(vect_length/2, vect_length)
        xQx = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,x_vect))
        self.assertAlmostEqual(xQx,abs(vect_length**2))
        # 3) Sending in pure tones/sinusoids for x and y should give delta functions

    def test_get_MW(self):
        n = 17
        random_G = generate_pos_def_all_pos(n)

        nt.assert_raises(AssertionError, self.ds.get_MW, random_G, mode='L^3')

        for mode in ['G^-1', 'G^-1/2', 'I', 'L^-1']:
            M, W = self.ds.get_MW(random_G, mode=mode)
            self.assertEqual(M.shape,(n,n))
            self.assertEqual(W.shape,(n,n))
            test_norm = np.sum(W, axis=1)
            for norm in test_norm:
                self.assertAlmostEqual(norm, 1.)

            if mode == 'G^-1':
                # Test that the window functions are delta functions
                self.assertEqual(diagonal_or_not(W),True)
            elif mode == 'G^-1/2':
                # Test that the error covariance is diagonal
                error_covariance = np.dot(M, np.dot(random_G, M.T)) # FIXME: We should be decorrelating V, not G. See Issue 21
                self.assertEqual(diagonal_or_not(error_covariance),True)
            elif mode == 'I':
                # Test that the norm matrix is diagonal
                self.assertEqual(diagonal_or_not(M),True)

    def test_q_hat(self):
        dfiles = [
            'zen.2458042.12552.xx.HH.uvXAA',
            'zen.2458042.12552.xx.HH.uvXAA'
        ]
        d = []
        for dfile in dfiles:
            _d = uv.UVData()
            _d.read_miriad(DATADIR + dfile)
            d.append(_d)
        w = [None for _d in dfiles]
        self.ds = pspecdata.PSpecData(dsets=d, wgts=w)
        Nfreq = self.ds.Nfreqs
        Ntime = self.ds.Ntimes

        for input_data_weight in ['identity', 'iC']:
            for taper in taper_selection:

                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                q_hat_a = self.ds.q_hat(key1, key2)
                self.assertEqual(q_hat_a.shape,(Nfreq,Ntime)) # Test shape
                q_hat_b = self.ds.q_hat(key2, key1)
                q_hat_diff = np.conjugate(q_hat_a) - q_hat_b

                # Check that swapping x_1 and x_2 results in just a complex conjugation
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_diff[i,j].real,q_hat_diff[i,j].real)
                        self.assertAlmostEqual(q_hat_diff[i,j].imag,q_hat_diff[i,j].imag)

                # Check that the slow method is the same as the FFT method
                q_hat_a_slow = self.ds.q_hat(key1, key2, use_fft=False)
                vector_scale = np.min([np.min(abs(q_hat_a_slow.real)),np.min(abs(q_hat_a_slow.imag))])
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertLessEqual(abs((q_hat_a[i,j]-q_hat_a_slow[i,j]).real),vector_scale*10**-6)
                        self.assertLessEqual(abs((q_hat_a[i,j]-q_hat_a_slow[i,j]).imag),vector_scale*10**-6)


    def test_get_G(self):
        dfiles = [
            'zen.2458042.12552.xx.HH.uvXAA',
            'zen.2458042.12552.xx.HH.uvXAA'
        ]
        d = []
        for dfile in dfiles:
            _d = uv.UVData()
            _d.read_miriad(DATADIR + dfile)
            d.append(_d)
        w = [None for _d in dfiles]
        self.ds = pspecdata.PSpecData(dsets=d, wgts=w)
        Nfreq = self.ds.Nfreqs

        for input_data_weight in ['identity','iC']:
            for taper in taper_selection:
                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                G = self.ds.get_G(key1, key2, taper=taper)
                self.assertEqual(G.shape,(Nfreq,Nfreq)) # Test shape
                matrix_scale = np.min([np.min(abs(G)),np.min(abs(np.linalg.eigvalsh(G)))])
                anti_sym_norm = np.linalg.norm(G - G.T)
                self.assertLessEqual(anti_sym_norm, matrix_scale*10**-10) # Test symmetry

                # Test cyclic property of trace, where key1 and key2 can be
                # swapped without changing the matrix. This is secretly the
                # same test as the symmetry test, but perhaps there are
                # creative ways to break the code to break one test but not
                # the other.
                G_swapped = self.ds.get_G(key2, key1, taper=taper)
                G_diff_norm = np.linalg.norm(G - G_swapped)
                self.assertLessEqual(G_diff_norm, matrix_scale*10**-10)


                min_diagonal = np.min(np.diagonal(G))
                # Test that all elements of G are positive up to numerical noise
                # with the threshold set to 10 orders of magnitude down from
                # the smallest value on the diagonal
                for i in range(Nfreq):
                    for j in range(Nfreq):
                        self.assertGreaterEqual(G[i,j], -min_diagonal*10**-10)

    def test_scalar(self):
        dfiles = [
            'zen.2458042.12552.xx.HH.uvXAA',
            'zen.2458042.12552.xx.HH.uvXAA'
        ]

        d = []
        for dfile in dfiles:
            _d = uv.UVData()
            _d.read_miriad(DATADIR + dfile)
            d.append(_d)
        w = [None for _d in dfiles]
        self.ds = pspecdata.PSpecData(dsets=d, wgts=w, beam=self.bm)

        scalar = self.ds.scalar()       
        self.assertAlmostEqual(scalar,3732415176.85 / 10**9) 

if __name__ == "__main__":
    unittest.main()

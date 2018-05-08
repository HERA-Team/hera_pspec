import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os, copy, sys
from scipy.integrate import simps, trapz
from hera_pspec import pspecdata, pspecbeam, conversions
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData
from hera_cal import redcal
from scipy.signal import windows
from scipy.interpolate import interp1d

# Data files to use in tests
dfiles = [
    'zen.2458042.12552.xx.HH.uvXAA',
    'zen.2458042.12552.xx.HH.uvXAA'
]

# List of tapering function to use in tests
taper_selection = ['none', 'blackman',]
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
            _d.read_miriad(os.path.join(DATA_PATH, dfile))
            self.d.append(_d)
        
        # Set trivial weights
        self.w = [None for _d in dfiles]
        
        # Load beam file
        beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.bm.filename = 'NF_HERA_Beams.beamfits'

        # load another data xx and yy polarized file
        self.uvd = uv.UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, 
                                          "zen.2458042.17772.xx.HH.uvXA"))

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

        # Test exception when not a UVData instance
        self.assertRaises(TypeError, ds.add, [1], [None])

    def test_add_data(self):
        """
        Test adding non UVData object.
        """
        nt.assert_raises(TypeError, self.ds.add, 1, 1)
    
    def test_labels(self):
        """
        Test that dataset labels work.
        """
        # Check that specifying labels does work
        psd = pspecdata.PSpecData( dsets=[self.d[0], self.d[1],], 
                                   wgts=[self.w[0], self.w[1], ],
                                   labels=['red', 'blue'] )
        np.testing.assert_array_equal( psd.x(('red', 24, 38)), 
                                       psd.x((0, 24, 38)) )
        
        # Check specifying labels using dicts
        dsdict = {'a':self.d[0], 'b':self.d[1]}
        psd = pspecdata.PSpecData(dsets=dsdict, wgts=dsdict)
        self.assertRaises(ValueError, pspecdata.PSpecData, dsets=dsdict, 
                          wgts=dsdict, labels=['a', 'b'])
        
        # Check that invalid labels raise errors
        self.assertRaises(KeyError, psd.x, ('green', 24, 38))
    
    def test_str(self):
        """
        Check that strings can be output.
        """
        ds = pspecdata.PSpecData()
        print(ds) # print empty psd
        ds.add(self.uvd, None)
        print(ds) # print populated psd
    
    def test_get_Q(self):
        """
        Test the Q = dC/dp function.
        """
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
        # Sending in sinusoids for x and y should give delta functions

    def test_get_MW(self):
        n = 17
        random_G = generate_pos_def_all_pos(n)
        random_H = generate_pos_def_all_pos(n)

        nt.assert_raises(AssertionError, self.ds.get_MW, random_G, random_H, mode='L^3')
        
        for mode in ['G^-1', 'G^-1/2', 'I', 'L^-1']:
            if mode == 'G^-1':
                # # Test that the window functions are delta functions
                # self.assertEqual(diagonal_or_not(W), True)
                nt.assert_raises(NotImplementedError, self.ds.get_MW, random_G, random_H, mode=mode)
            elif mode == 'G^-1/2':
                # # Test that the error covariance is diagonal
                # error_covariance = np.dot(M, np.dot(random_G, M.T)) 
                # # FIXME: We should be decorrelating V, not G. See Issue 21
                # self.assertEqual(diagonal_or_not(error_covariance), True)
                nt.assert_raises(NotImplementedError, self.ds.get_MW, random_G, random_H, mode=mode)
            elif mode == 'I':
                # Test that the norm matrix is diagonal
                M, W = self.ds.get_MW(random_G, random_H, mode=mode)
                self.assertEqual(diagonal_or_not(M), True)
                self.assertEqual(M.shape, (n,n))
                self.assertEqual(W.shape, (n,n))
                test_norm = np.sum(W, axis=1)
                for norm in test_norm:
                    self.assertAlmostEqual(norm, 1.)

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
                q_hat_a = self.ds.q_hat(key1, key2, taper=taper)
                self.assertEqual(q_hat_a.shape, (Nfreq, Ntime))
                
                # Check that swapping x_1 <-> x_2 results in complex conj. only
                q_hat_b = self.ds.q_hat(key2, key1, taper=taper)
                q_hat_diff = np.conjugate(q_hat_a) - q_hat_b
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_diff[i,j].real, 
                                               q_hat_diff[i,j].real)
                        self.assertAlmostEqual(q_hat_diff[i,j].imag, 
                                               q_hat_diff[i,j].imag)
                
                # Check that lists of keys are handled properly
                q_hat_aa = self.ds.q_hat(key1, key4, taper=taper) # q_hat(k1, k2+k2)
                q_hat_bb = self.ds.q_hat(key4, key1, taper=taper) # q_hat(k2+k2, k1)
                q_hat_cc = self.ds.q_hat(key3, key4, taper=taper) # q_hat(k1+k1, k2+k2)
                
                # Effectively checks that q_hat(2*k1, 2*k2) = 4*q_hat(k1, k2)
                for i in range(Nfreq):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_a[i,j].real, 
                                               0.25 * q_hat_cc[i,j].real)
                        self.assertAlmostEqual(q_hat_a[i,j].imag, 
                                               0.25 * q_hat_cc[i,j].imag)
                
                # Check that the slow method is the same as the FFT method
                q_hat_a_slow = self.ds.q_hat(key1, key2, use_fft=False, taper=taper)
                self.assertTrue(np.isclose(np.real(q_hat_a/q_hat_a_slow), 1).all())
                self.assertTrue(np.isclose(np.imag(q_hat_a/q_hat_a_slow), 0, atol=1e-6).all())

    def test_get_H(self):
        """
        Test Fisher/weight matrix calculation.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        multiplicative_tolerance = 1.

        for input_data_weight in ['identity','iC']:
            for taper in taper_selection:
                print 'input_data_weight', input_data_weight
                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                H = self.ds.get_H(key1, key2, taper=taper)
                self.assertEqual(H.shape, (Nfreq,Nfreq)) # Test shape

    def test_get_G(self):
        """
        Test Fisher/weight matrix calculation.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        multiplicative_tolerance = 1.

        for input_data_weight in ['identity','iC']:
            for taper in taper_selection:
                print 'input_data_weight', input_data_weight
                self.ds.set_R(input_data_weight)
                key1 = (0, 24, 38)
                key2 = (1, 25, 38)

                G = self.ds.get_G(key1, key2)
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
                    G_swapped = self.ds.get_G(key2, key1)
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
                    G_swapped = self.ds.get_G(key2, key1)
                    G_diff_norm = np.linalg.norm(G - G_swapped.T)
                    self.assertLessEqual(G_diff_norm, 
                                         matrix_scale * multiplicative_tolerance)

            
    '''
    Under Construction

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
    '''
    
    def test_get_V_gaussian(self):
        nt.assert_raises(NotImplementedError, self.ds.get_V_gaussian, 
                         (0,1), (0,1))
        

    def test_scalar(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=self.bm)
        
        gauss = pspecbeam.PSpecBeamGauss(0.8, 
                                  np.linspace(115e6, 130e6, 50, endpoint=False))
        ds2 = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=gauss)
        
        # Precomputed results in the following test were done "by hand" 
        # using iPython notebook "Scalar_dev2.ipynb" in the tests/ directory
        # FIXME: Uncomment when pyuvdata support for this is ready
        #scalar = self.ds.scalar()
        #self.assertAlmostEqual(scalar, 3732415176.85 / 10.**9)
        
        # FIXME: Remove this when pyuvdata support for the above is ready
        #self.assertRaises(NotImplementedError, self.ds.scalar)

    def test_validate_datasets(self):
        # test freq exception
        uvd = copy.deepcopy(self.d[0])
        uvd2 = uvd.select(frequencies=np.unique(uvd.freq_array)[:10], inplace=False)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)
        # test time exception
        uvd2 = uvd.select(times=np.unique(uvd.time_array)[:10], inplace=False)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)
        # test wgt exception
        ds.wgts = ds.wgts[:1]
        nt.assert_raises(ValueError, ds.validate_datasets)
        # test warnings
        uvd = copy.deepcopy(self.d[0])
        uvd2 = copy.deepcopy(self.d[0])
        uvd.select(frequencies=np.unique(uvd.freq_array)[:10], times=np.unique(uvd.time_array)[:10])
        uvd2.select(frequencies=np.unique(uvd2.freq_array)[10:20], times=np.unique(uvd2.time_array)[10:20])
        # test phasing
        uvd = copy.deepcopy(self.d[0])
        uvd2 = copy.deepcopy(self.d[0])
        uvd.phase_to_time(2458042)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)
        uvd2.phase_to_time(2458042.5)
        ds.validate_datasets()

    def test_rephase_to_dst(self):
        # generate two uvd objects w/ different LST grids
        uvd1 = copy.deepcopy(self.uvd)
        uvd2 = uv.UVData()
        uvd2.read_miriad(os.path.join(DATA_PATH, "zen.2458042.19263.xx.HH.uvXA"))

        # null test: check nothing changes when dsets contain same UVData object
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd1)], wgts=[None, None])
        # get normal pspec
        bls = [(37, 39)]
        uvp1 = ds.pspec(bls, bls, (0, 1), pols=[('xx','xx')], verbose=False)
        # rephase and get pspec
        ds.rephase_to_dset(0)
        uvp2 = ds.pspec(bls, bls, (0, 1), pols=[('xx','xx')], verbose=False)
        blp = (0, ((37,39),(37,39)), 'XX')
        nt.assert_true(np.isclose(np.abs(uvp2.get_data(blp)/uvp1.get_data(blp)), 1.0).min())

    def test_units(self):
        ds = pspecdata.PSpecData()
        # test exception
        nt.assert_raises(IndexError, ds.units)
        ds.add(self.uvd, None)
        # test basic execution
        vis_u, norm_u = ds.units()
        nt.assert_equal(vis_u, "UNCALIB")
        nt.assert_equal(norm_u, "Hz str [beam normalization not specified]")

    def test_delays(self):
        ds = pspecdata.PSpecData()
        # test exception
        nt.assert_raises(IndexError, ds.delays)
        ds.add([self.uvd, self.uvd], [None, None])
        d = ds.delays()
        nt.assert_true(len(d), ds.dsets[0].Nfreqs)

    def test_check_in_dset(self):
        # generate ds
        uvd = copy.deepcopy(self.d[0])
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None])
        # check for existing key
        nt.assert_true(ds.check_key_in_dset(('xx'), 0))
        nt.assert_true(ds.check_key_in_dset((24, 25), 0))
        nt.assert_true(ds.check_key_in_dset((24, 25, 'xx'), 0))
        # check for non-existing key
        nt.assert_false(ds.check_key_in_dset('yy', 0))
        nt.assert_false(ds.check_key_in_dset((24, 26), 0))
        nt.assert_false(ds.check_key_in_dset((24, 26, 'yy'), 0))
        # check exception
        nt.assert_raises(KeyError, ds.check_key_in_dset, (1,2,3,4,5), 0)

    def test_pspec(self):
        # generate ds
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm, labels=['red', 'blue'])

        # check basic execution with baseline list
        bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
        uvp = ds.pspec(bls, bls, (0, 1), pols=[('xx','xx')], input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
 

        nt.assert_equal(len(uvp.bl_array), len(bls))
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (24, 25))) in uvp.blpair_array)
        nt.assert_equal(uvp.data_array[0].dtype, np.complex128)
        nt.assert_equal(uvp.data_array[0].shape, (240, 64, 1)) 

        # check with redundant baseline group list
        antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
        antpos = dict(zip(ants, antpos))
        red_bls = map(lambda blg: sorted(blg), redcal.get_pos_reds(antpos, low_hi=True))[2]
        bls1, bls2, blps = pspecdata.construct_blpairs(red_bls, exclude_permutations=True)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (37, 38))) in uvp.blpair_array)
        nt.assert_equal(uvp.Nblpairs, 10)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (52, 53))) in uvp.blpair_array)
        nt.assert_true(uvp.antnums_to_blpair(((52, 53), (24, 25))) not in uvp.blpair_array)
        nt.assert_equal(uvp.Nblpairs, 10)

        # test mixed bl group and non blgroup, currently bl grouping of more than 1 blpair doesn't work
        bls1 = [[(24, 25)], (52, 53)]
        bls2 = [[(24, 25)], (52, 53)]
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)

        # test select
        red_bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
        bls1, bls2, blp = pspecdata.construct_blpairs(red_bls, exclude_permutations=False, exclude_auto_bls=False)
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], spw_ranges=[(20,30), (30,40)], verbose=False)
        nt.assert_equal(uvp.Nblpairs, 16)
        nt.assert_equal(uvp.Nspws, 2)
        uvp2 = uvp.select(spws=[0], bls=[(24, 25)], only_pairs_in_bls=False, inplace=False)
        nt.assert_equal(uvp2.Nspws, 1)
        nt.assert_equal(uvp2.Nblpairs, 7)
        uvp.select(spws=0, bls=(24, 25), only_pairs_in_bls=True, inplace=True)
        nt.assert_equal(uvp.Nspws, 1)
        nt.assert_equal(uvp.Nblpairs, 1)

        # check w/ multiple spectral ranges
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), pols=[('xx','xx')], spw_ranges=[(10, 24), (30, 40), (45, 64)], verbose=False)
        nt.assert_equal(uvp.Nspws, 3)
        nt.assert_equal(uvp.Nspwdlys, 43)
        nt.assert_equal(uvp.data_array[0].shape, (240, 14, 1))
        nt.assert_equal(uvp.get_data(0, 24025024025, 'xx').shape, (60, 14))

        # check select
        uvp.select(spws=[1])
        nt.assert_equal(uvp.Nspws, 1)
        nt.assert_equal(uvp.Ndlys, 10)
        nt.assert_equal(len(uvp.data_array), 1)

        # test multiple polarization pairs
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), pols=[('xx','xx'), ('yy','yy')], spw_ranges=[(10, 24)], verbose=False)        
        # test polarizations specified by numbers
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), pols=[(-5,-5)], spw_ranges=[(10, 24)], verbose=False)

        # test exceptions
        nt.assert_raises(AssertionError, ds.pspec, bls1[:1], bls2, (0, 1), pols=[('xx','xx')])
        nt.assert_raises(NotImplementedError, ds.pspec, bls, bls, (0, 1), pols=[('xx','yy')])

        # test when polarization pair is not specified
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), spw_ranges=[(10, 24)], verbose=False)
              
    def test_normalization(self):
        # Test Normalization of pspec() compared to PAPER legacy techniques
        d1 = self.uvd.select(times=np.unique(self.uvd.time_array)[:-1:2], 
                             frequencies=np.unique(self.uvd.freq_array)[40:51], inplace=False)
        d2 = self.uvd.select(times=np.unique(self.uvd.time_array)[1::2], 
                             frequencies=np.unique(self.uvd.freq_array)[40:51], inplace=False)
        freqs = np.unique(d1.freq_array)

        # Setup baselines
        bls1 = [(24, 25)]
        bls2 = [(37, 38)]

        # Get beam
        beam = copy.deepcopy(self.bm)
        cosmo = conversions.Cosmo_Conversions()

        # Set to mK scale
        d1.data_array *= beam.Jy_to_mK(freqs)[None, None, :, None]
        d2.data_array *= beam.Jy_to_mK(freqs)[None, None, :, None]

        # Compare using no taper
        OmegaP = beam.power_beam_int()
        OmegaPP = beam.power_beam_sq_int()
        OmegaP = interp1d(beam.beam_freqs/1e6, OmegaP)(freqs/1e6)
        OmegaPP = interp1d(beam.beam_freqs/1e6, OmegaPP)(freqs/1e6)
        NEB = 1.0
        Bp = np.median(np.diff(freqs)) * len(freqs)
        scalar = cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2/OmegaPP) * Bp * NEB
        data1 = d1.get_data(bls1[0])
        data2 = d2.get_data(bls2[0])
        legacy = np.fft.fftshift(np.fft.ifft(data1, axis=1) * np.conj(np.fft.ifft(data2, axis=1)) * scalar, axes=1)[0]
        # hera_pspec OQE
        ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], taper='none', input_data_weight='identity', norm='I')
        oqe = uvp.get_data(0, ((24, 25), (37, 38)), 'xx')[0]
        # assert answers are same to within 3%
        nt.assert_true(np.isclose(np.real(oqe)/np.real(legacy), 1, atol=0.03, rtol=0.03).all())

        # taper
        window = windows.blackmanharris(len(freqs))
        NEB = Bp / trapz(window**2, x=freqs)
        scalar = cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2/OmegaPP) * Bp * NEB
        data1 = d1.get_data(bls1[0])
        data2 = d2.get_data(bls2[0])
        legacy = np.fft.fftshift(np.fft.ifft(data1*window[None, :], axis=1) * np.conj(np.fft.ifft(data2*window[None, :], axis=1)) * scalar, axes=1)[0]
        # hera_pspec OQE
        ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=[('xx','xx')], taper='blackman-harris', input_data_weight='identity', norm='I')
        oqe = uvp.get_data(0, ((24, 25), (37, 38)), 'xx')[0]
        # assert answers are same to within 3%
        nt.assert_true(np.isclose(np.real(oqe)/np.real(legacy), 1, atol=0.03, rtol=0.03).all())

    def test_validate_blpairs(self):
        # test exceptions
        uvd = copy.deepcopy(self.uvd)
        nt.assert_raises(TypeError, pspecdata.validate_blpairs, [((1, 2), (2, 3))], None, uvd)
        nt.assert_raises(TypeError, pspecdata.validate_blpairs, [((1, 2), (2, 3))], uvd, None)

        bls = [(24,25),(37,38)]
        bls1, bls2, blpairs = pspecdata.construct_blpairs(bls, exclude_permutations=False, exclude_auto_bls=True)
        pspecdata.validate_blpairs(blpairs, uvd, uvd)
        bls1, bls2, blpairs = pspecdata.construct_blpairs(bls, exclude_permutations=False, exclude_auto_bls=True,
                                                          group=True)

        pspecdata.validate_blpairs(blpairs, uvd, uvd)

        # test non-redundant
        blpairs = [((24, 25), (24, 38))]
        pspecdata.validate_blpairs(blpairs, uvd, uvd)

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

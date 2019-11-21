import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os, copy, sys
from scipy.integrate import simps, trapz
from hera_pspec import pspecdata, pspecbeam, conversions, container, utils
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData, UVCal
from hera_cal import redcal
from scipy.signal import windows
from scipy.interpolate import interp1d
from astropy.time import Time
import warnings
import glob
from uvtools import dspec

# Data files to use in tests
dfiles = [
    'zen.2458042.12552.xx.HH.uvXAA',
    'zen.2458042.12552.xx.HH.uvXAA'
]
dfiles_std = [
    'zen.2458042.12552.std.xx.HH.uvXAA',
    'zen.2458042.12552.std.xx.HH.uvXAA'
]

# List of tapering function to use in tests
taper_selection = ['none', 'bh7',]
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

        # Load standard deviations
        self.d_std = []
        for dfile in dfiles_std:
            _d = uv.UVData()
            _d.read_miriad(os.path.join(DATA_PATH, dfile))
            self.d_std.append(_d)

        # Set trivial weights
        self.w = [None for _d in dfiles]

        # Load beam file
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.bm.filename = 'HERA_NF_dipole_power.beamfits'

        #Load isotropic beam file
        beamfile_Q = os.path.join(DATA_PATH, 'isotropic_beam.beamfits')
        self.bm_Q  = pspecbeam.PSpecBeamUV(beamfile_Q)
        self.bm_Q.filename = 'isotropic_beam.beamfits'

        # load another data file
        self.uvd = uv.UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH,
                                          "zen.2458042.17772.xx.HH.uvXA"))

        self.uvd_std = uv.UVData()
        self.uvd_std.read_miriad(os.path.join(DATA_PATH,
                                          "zen.2458042.17772.std.xx.HH.uvXA"))

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

        # Test get weights when fed a UVData for weights
        ds = pspecdata.PSpecData(dsets=[self.uvd, self.uvd], wgts=[self.uvd, self.uvd])
        key = (0, (24, 25), 'xx')
        nt.assert_true(np.all(np.isclose(ds.x(key), ds.w(key))))

        # Test labels when adding dsets
        uvd = self.uvd
        ds = pspecdata.PSpecData()
        nt.assert_equal(len(ds.labels), 0)
        ds.add([uvd, uvd], [None, None])
        nt.assert_equal(len(ds.labels), 2)
        ds.add(uvd, None, labels='foo')
        nt.assert_equal(len(ds.dsets), len(ds.labels), 3)
        nt.assert_equal(ds.labels, ['dset0', 'dset1', 'foo'])
        ds.add(uvd, None)
        nt.assert_equal(ds.labels, ['dset0', 'dset1', 'foo', 'dset3'])

        # Test some exceptions
        ds = pspecdata.PSpecData()
        nt.assert_raises(ValueError, ds.get_G, key, key)
        nt.assert_raises(ValueError, ds.get_H, key, key)

    def test_add_data(self):
        """
        Test PSpecData add()
        """
        uv = self.d[0]
        # test adding non list objects
        nt.assert_raises(TypeError, self.ds.add, 1, 1)
        # test adding non UVData objects
        nt.assert_raises(TypeError, self.ds.add, [1], None)
        nt.assert_raises(TypeError, self.ds.add, [uv], [1])
        nt.assert_raises(TypeError, self.ds.add, [uv], None, dsets_std=[1])
        # test adding non UVCal for cals
        nt.assert_raises(TypeError, self.ds.add, [uv], None, cals=[1])
        # test TypeError if dsets is dict but other inputs are not
        nt.assert_raises(TypeError, self.ds.add, {'d':uv}, [0])
        nt.assert_raises(TypeError, self.ds.add, {'d':uv}, {'d':uv}, dsets_std=[0])
        nt.assert_raises(TypeError, self.ds.add, {'d':uv}, {'d':uv}, cals=[0])
        # specifying labels when dsets is a dict is a ValueError
        nt.assert_raises(ValueError, self.ds.add, {'d':uv}, None, labels=['d'])
        # use lists, but not appropriate lengths
        nt.assert_raises(AssertionError, self.ds.add, [uv], [uv, uv])
        nt.assert_raises(AssertionError, self.ds.add, [uv], None, dsets_std=[uv, uv])
        nt.assert_raises(AssertionError, self.ds.add, [uv], None, cals=[None, None])
        nt.assert_raises(AssertionError, self.ds.add, [uv], None, labels=['foo', 'bar'])


    def test_set_symmetric_taper(self):
        """
        Make sure that you can't set a symmtric taper with an truncated R matrix
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.spw_Nfreqs
        Ntime = self.ds.Ntimes
        Ndlys = Nfreq - 3
        self.ds.spw_Ndlys = Ndlys


        # Set baselines to use for tests
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)
        key3 = [(0, 24, 38), (0, 24, 38)]
        key4 = [(1, 25, 38), (1, 25, 38)]

        rpk1 = {'filter_centers':[0.],'filter_half_widths':[100e-9],'filter_factors':[1e-9]}
        rpk2 = {'filter_centers':[0.],'filter_half_widths':[100e-9],'filter_factors':[1e-9]}
        self.ds.set_weighting('dayenu')
        self.ds.set_r_param(key1,rpk1)
        self.ds.set_r_param(key2,rpk2)
        ds1 = copy.deepcopy(self.ds)
        ds1.set_spw((10,Nfreq-10))
        ds1.set_symmetric_taper(False)
        ds1.set_filter_extension([10,10])
        ds1.set_filter_extension((10,10))
        rm1 = self.ds.R(key1)
        self.ds.set_symmetric_taper(True)
        nt.assert_raises(ValueError, ds1.set_symmetric_taper, True)
        #now make sure warnings are raised when we extend filter with
        #symmetric tapering and that symmetric taper is set to false.
        with warnings.catch_warnings(record=True) as w:
            self.ds.set_filter_extension((10,10))
        assert len(w) > 0
        self.assertTrue(not(self.ds.symmetric_taper))

        """
        Now directly compare results to expectations.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.spw_Nfreqs
        Ntime = self.ds.Ntimes
        Ndlys = Nfreq - 3
        self.ds.spw_Ndlys = Ndlys
        key1 = (0, 24, 38)
        key2 = (1,25, 38)

        rpk1 = {'filter_centers':[0.],'filter_half_widths':[100e-9],'filter_factors':[1e-9]}
        self.ds.set_weighting('dayenu')
        self.ds.set_taper('bh7')
        self.ds.set_r_param(key1,rpk1)
        #get the symmetric tapering
        rmat_symmetric = self.ds.R(key1)
        #now set taper to be asymmetric
        self.ds.set_symmetric_taper(False)
        rmat_a = self.ds.R(key1)
        #check against independent solution
        bh_taper = np.sqrt(dspec.gen_window('bh7', Nfreq).reshape(1,-1))
        rmat = dspec.dayenu_mat_inv(x=self.ds.freqs[self.ds.spw_range[0]:self.ds.spw_range[1]],
        filter_centers=[0.], filter_half_widths=[100e-9], filter_factors=[1e-9])
        wmat = np.outer(np.diag(np.sqrt(self.ds.Y(key1))), np.diag(np.sqrt(self.ds.Y(key1))))
        rmat = np.linalg.pinv(wmat * rmat)
        self.assertTrue(np.all(np.isclose(rmat_symmetric, bh_taper.T * rmat * bh_taper,atol=1e-6)))
        self.assertTrue(np.all(np.isclose(rmat_a, bh_taper.T ** 2. * rmat,atol=1e-6)))
        self.assertTrue(not np.all(np.isclose(rmat_symmetric, rmat_a,atol=1e-6)))


    def test_labels(self):
        """
        Test that dataset labels work.
        """
        # Check that specifying labels does work
        psd = pspecdata.PSpecData( dsets=[self.d[0], self.d[1],],
                                   wgts=[self.w[0], self.w[1], ],
                                   labels=['red', 'blue'])
        np.testing.assert_array_equal( psd.x(('red', 24, 38)),
                                       psd.x((0, 24, 38)) )

        # Check specifying labels using dicts
        dsdict = {'a':self.d[0], 'b':self.d[1]}
        psd = pspecdata.PSpecData(dsets=dsdict, wgts=dsdict)
        nt.assert_raises(ValueError, pspecdata.PSpecData, dsets=dsdict,
                          wgts=dsdict, labels=['a', 'b'])

        # Check that invalid labels raise errors
        nt.assert_raises(KeyError, psd.x, ('green', 24, 38))

    def test_parse_blkey(self):
        # make a double-pol UVData
        uvd = copy.deepcopy(self.uvd)
        uvd.polarization_array[0] = -7
        uvd = uvd + self.uvd
        # check parse_blkey
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], labels=['red', 'blue'])
        dset, bl = ds.parse_blkey((0, (24, 25)))
        nt.assert_equal(dset, 0)
        nt.assert_equal(bl, (24, 25))
        dset, bl = ds.parse_blkey(('red', (24, 25), 'xx'))
        nt.assert_equal(dset, 0)
        nt.assert_equal(bl, (24, 25, 'xx'))
        # check PSpecData.x works
        nt.assert_equal(ds.x(('red', (24, 25))).shape, (2, 64, 60))
        nt.assert_equal(ds.x(('red', (24, 25), 'xx')).shape, (64, 60))
        nt.assert_equal(ds.w(('red', (24, 25))).shape, (2, 64, 60))
        nt.assert_equal(ds.w(('red', (24, 25), 'xx')).shape, (64, 60))

    def test_str(self):
        """
        Check that strings can be output.
        """
        ds = pspecdata.PSpecData()
        print(ds) # print empty psd
        ds.add(self.uvd, None)
        print(ds) # print populated psd

    def test_get_Q_alt(self):
        """
        Test the Q = dC/dp function.
        """
        vect_length = 50
        x_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)
        y_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)

        self.ds.spw_Nfreqs = vect_length

        for i in range(vect_length):
            Q_matrix = self.ds.get_Q_alt(i)
            # Test that if the number of delay bins hasn't been set
            # the code defaults to putting that equal to Nfreqs
            self.assertEqual(self.ds.spw_Ndlys, self.ds.spw_Nfreqs)

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
        Q_matrix = self.ds.get_Q_alt(vect_length//2)
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))
        # Sending in sinusoids for x and y should give delta functions

        # Now do all the same tests from above but for a different number
        # of delay channels
        self.ds.set_Ndlys(vect_length-3)
        for i in range(vect_length-3):
            Q_matrix = self.ds.get_Q_alt(i)
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
        Q_matrix = self.ds.get_Q_alt((vect_length-2)//2-1)
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))
        # Sending in sinusoids for x and y should give delta functions

        # Make sure that error is raised when asking for a delay mode outside
        # of the range of delay bins
        nt.assert_raises(IndexError, self.ds.get_Q_alt, vect_length-1)

        # Ensure that in the special case where the number of channels equals
        # the number of delay bins, the FFT method gives the same answer as
        # the explicit construction method
        multiplicative_tolerance = 0.001
        self.ds.set_Ndlys(vect_length)
        for alpha in range(vect_length):
            Q_matrix_fft = self.ds.get_Q_alt(alpha)
            Q_matrix = self.ds.get_Q_alt(alpha, allow_fft=False)
            Q_diff_norm = np.linalg.norm(Q_matrix - Q_matrix_fft)
            self.assertLessEqual(Q_diff_norm, multiplicative_tolerance)

        # Check for error handling
        nt.assert_raises(ValueError, self.ds.set_Ndlys, vect_length+100)


    def test_get_Q(self):
        """
        Test the Q = dC_ij/dp function.

        A general comment here:
        I would really want to do away with try and exception statements. The reason to use them now
        was that current unittests throw in empty datasets to these functions. Given that we are computing
        the actual value of tau/freq/taper etc. we do need datasets! Currently, if there is no dataset,
        Q_matrix is simply an identity matrix with same dimensions as that of vector length.
        It will be very helpful if we can have more elegant solution for this.

        """
        vect_length = 50
        x_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)
        y_vect = np.random.normal(size=vect_length) \
               + 1.j * np.random.normal(size=vect_length)

        self.ds.spw_Nfreqs = vect_length
        #Test if there is a warning if user does not pass the beam
        key1 = (0, 24, 38)
        key2 = (1, 24, 38)
        uvd = copy.deepcopy(self.uvd)
        ds_t = pspecdata.PSpecData(dsets=[uvd, uvd])

        for i in range(vect_length):
            try:
                Q_matrix = self.ds.get_Q(i)
                # Test that if the number of delay bins hasn't been set
                # the code defaults to putting that equal to Nfreqs
                self.assertEqual(self.ds.spw_Ndlys, self.ds.spw_Nfreqs)
            except IndexError:
                Q_matrix = np.ones((vect_length, vect_length))

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
        try:
            Q_matrix = self.ds.get_Q(vect_length/2)
        except IndexError:
            Q_matrix = np.ones((vect_length, vect_length))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))

        # Now do all the same tests from above but for a different number
        # of delay channels
        self.ds.set_Ndlys(vect_length-3)
        for i in range(vect_length-3):
            try:
                Q_matrix = self.ds.get_Q(i)
            except IndexError:
                Q_matrix = np.ones((vect_length,vect_length))
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
        try:
            Q_matrix = self.ds.get_Q((vect_length-2)/2-1)
        except IndexError:
            Q_matrix = np.ones((vect_length,vect_length))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
        self.assertAlmostEqual(xQx, np.abs(vect_length**2.))

        # Make sure that error is raised when asking for a delay mode outside
        # of the range of delay bins
        nt.assert_raises(IndexError, self.ds.get_Q, vect_length-1)

    def test_get_integral_beam(self):
        """
        Test the integral of the beam and tapering function in Q.
        """
        pol = 'xx'
        #Test if there is a warning if user does not pass the beam
        uvd = copy.deepcopy(self.uvd)
        ds_t = pspecdata.PSpecData(dsets=[uvd, uvd])
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], beam=self.bm)

        with warnings.catch_warnings(record=True) as w:
            ds_t.get_integral_beam(pol)
        assert len(w) > 0

        try:
            integral_matrix = ds.get_integral_beam(pol)
            # Test that if the number of delay bins hasn't been set
            # the code defaults to putting that equal to Nfreqs
            self.assertEqual(ds.spw_Ndlys, ds.spw_Nfreqs)
        except IndexError:
            integral_matrix = np.ones((ds.spw_Ndlys, ds.spw_Ndlys))

        # Test that integral matrix has the right shape
        self.assertEqual(integral_matrix.shape, (ds.spw_Nfreqs, ds.spw_Nfreqs))

    def test_get_unnormed_E(self):
        """
        Test the E function
        """
        # Test that error is raised if spw_Ndlys is not set
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], labels=['red', 'blue'])
        ds.spw_Ndlys = None
        nt.assert_raises(ValueError, ds.get_unnormed_E, 'placeholder', 'placeholder',0 )

        # Test that if R1 = R2, then the result is Hermitian
        ds.spw_Ndlys = 7
        random_R = generate_pos_def_all_pos(ds.spw_Nfreqs)
        wgt_matrix_dict = {} # The keys here have no significance except they are formatted right
        wgt_matrix_dict[('red', (24, 25))] = random_R
        wgt_matrix_dict[('blue', (24, 25))] = random_R
        ds.set_R(wgt_matrix_dict)
        E_matrices = ds.get_unnormed_E(('red', (24, 25)), ('blue', (24, 25)), time_index=0)
        multiplicative_tolerance = 0.0000001
        for matrix in E_matrices:
            diff_norm = np.linalg.norm(matrix.T.conj() - matrix)
            self.assertLessEqual(diff_norm, multiplicative_tolerance)

        #Test for the correct shape when exact_norm is True
        ds_c = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], labels=['red', 'blue'], beam=self.bm)
        ds_c.spw_Ndlys = 10
        random_R = generate_pos_def_all_pos(ds_c.spw_Nfreqs)
        wgt_matrix_dict = {}
        wgt_matrix_dict[('red', (24, 25))] = random_R
        wgt_matrix_dict[('blue', (24, 25))] = random_R
        ds_c.set_R(wgt_matrix_dict)
        E_matrices = ds_c.get_unnormed_E(('red', (24, 25)), ('blue', (24, 25)), exact_norm=True, pol='xx', time_index=0)
        self.assertEqual(E_matrices.shape, (ds_c.spw_Ndlys, ds_c.spw_Nfreqs, ds_c.spw_Nfreqs))

        # Test that if R1 != R2, then i) E^{12,dagger} = E^{21}
        random_R2 = generate_pos_def_all_pos(ds.spw_Nfreqs)
        wgt_matrix_dict = {}
        wgt_matrix_dict[('red', (24, 25))] = random_R
        wgt_matrix_dict[('blue', (24, 25))] = random_R2
        ds.set_R(wgt_matrix_dict)
        E12_matrices = ds.get_unnormed_E(('red', (24, 25)), ('blue', (24, 25)), time_index=0)
        E21_matrices = ds.get_unnormed_E(('blue', (24, 25)), ('red', (24, 25)), time_index=0)
        multiplicative_tolerance = 0.0000001
        for mat12,mat21 in zip(E12_matrices,E21_matrices):
            diff_norm = np.linalg.norm(mat12.T.conj() - mat21)
            self.assertLessEqual(diff_norm, multiplicative_tolerance)

        # Test that if there is only one delay bin and R1 = R2 = I, then
        # the E matrices are all 0.5s exept in flagged channels.
        ds.spw_Ndlys = 1
        wgt_matrix_dict = {}
        wgt_matrix_dict[('red', (24, 25))] = np.eye(ds.spw_Nfreqs)
        wgt_matrix_dict[('blue', (24, 25))] = np.eye(ds.spw_Nfreqs)
        flags1 = ds.Y(('red', (24, 25)))[:,0]
        flags2 = ds.Y(('blue', (24, 25)))[:,0]
        ds.set_R(wgt_matrix_dict)
        E_matrices = ds.get_unnormed_E(('red', (24, 25)), ('blue', (24, 25)), time_index=0)
        multiplicative_tolerance = 0.0000001
        for matrix in E_matrices:
            for i in range(ds.spw_Nfreqs):
                for j in range(ds.spw_Nfreqs):
                    if flags1[i] * flags2[j] == 0: # either channel flagged
                        self.assertAlmostEqual(matrix[i,j], 0.)
                    else:
                        self.assertAlmostEqual(matrix[i,j], 0.5)

    def test_cross_covar_model(self):
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], labels=['red', 'blue'])
        key1 = ('red', (24, 25), 'xx')
        key2 = ('blue', (25, 38), 'xx')

        nt.assert_raises(AssertionError, ds.cross_covar_model, key1, key2, model='other_string')
        nt.assert_raises(AssertionError, ds.cross_covar_model, key1, 'a_string')

        conj1_conj1 = ds.cross_covar_model(key1, key1, conj_1=True, conj_2=True)
        conj1_real1 = ds.cross_covar_model(key1, key1, conj_1=True, conj_2=False)
        real1_conj1 = ds.cross_covar_model(key1, key1, conj_1=False, conj_2=True)
        real1_real1 = ds.cross_covar_model(key1, key1, conj_1=False, conj_2=False)

        # Check matrix sizes
        for matrix in [conj1_conj1, conj1_real1, real1_conj1, real1_real1]:
            self.assertEqual(matrix.shape, (ds.spw_Nfreqs, ds.spw_Nfreqs))

        for i in range(ds.spw_Nfreqs):
            for j in range(ds.spw_Nfreqs):
                # Check that the matrices that ought to be Hermitian are indeed Hermitian
                self.assertAlmostEqual(conj1_real1.conj().T[i,j], conj1_real1[i,j])
                self.assertAlmostEqual(real1_conj1.conj().T[i,j], real1_conj1[i,j])
                # Check that real_real and conj_conj are complex conjugates of each other
                # Also check that they are symmetric
                self.assertAlmostEqual(real1_real1.conj()[i,j], conj1_conj1[i,j])
                self.assertAlmostEqual(real1_real1[j,i], real1_real1[i,j])
                self.assertAlmostEqual(conj1_conj1[j,i], conj1_conj1[i,j])


        real1_real2 = ds.cross_covar_model(key1, key2, conj_1=False, conj_2=False)
        real2_real1 = ds.cross_covar_model(key2, key1, conj_1=False, conj_2=False)
        conj1_conj2 = ds.cross_covar_model(key1, key2, conj_1=True, conj_2=True)
        conj2_conj1 = ds.cross_covar_model(key2, key1, conj_1=True, conj_2=True)
        conj1_real2 = ds.cross_covar_model(key1, key2, conj_1=True, conj_2=False)
        conj2_real1 = ds.cross_covar_model(key2, key1, conj_1=True, conj_2=False)
        real1_conj2 = ds.cross_covar_model(key1, key2, conj_1=False, conj_2=True)
        real2_conj1 = ds.cross_covar_model(key2, key1, conj_1=False, conj_2=True)

        # And some similar tests for cross covariances
        for i in range(ds.spw_Nfreqs):
            for j in range(ds.spw_Nfreqs):
                self.assertAlmostEqual(real1_real2.T[i,j], real2_real1[i,j])
                self.assertAlmostEqual(conj1_conj2.T[i,j], conj2_conj1[i,j])
                self.assertAlmostEqual(conj1_real2.conj()[j,i], conj2_real1[i,j])
                self.assertAlmostEqual(real1_conj2.conj()[j,i], real2_conj1[i,j])

    def test_get_unnormed_V(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, labels=['red', 'blue'])
        key1 = ('red', (24, 25), 'xx')
        key2 = ('blue', (25, 38), 'xx')
        self.ds.spw_Ndlys = 5

        V = self.ds.get_unnormed_V(key1, key2, time_index=0)
        # Check size
        self.assertEqual(V.shape, (self.ds.spw_Ndlys,self.ds.spw_Ndlys))
        # Test hermiticity. Generally this is only good to about 1 part in 10^15.
        # If this is an issue downstream, should investigate more in the future.
        tol = 1e-10
        frac_non_herm = abs(V.conj().T - V)/abs(V)
        for i in range(self.ds.spw_Ndlys):
            for j in range(self.ds.spw_Ndlys):
                self.assertLessEqual(frac_non_herm[i,j], tol)

    def test_get_MW(self):
        n = 17
        random_G = generate_pos_def_all_pos(n)
        random_H = generate_pos_def_all_pos(n)
        random_V = generate_pos_def_all_pos(n)

        nt.assert_raises(AssertionError, self.ds.get_MW, random_G, random_H, mode='L^3')
        nt.assert_raises(NotImplementedError, self.ds.get_MW, random_G, random_H, mode='H^-1', exact_norm=True)

        for mode in ['H^-1', 'V^-1/2', 'I', 'L^-1']:
            if mode == 'H^-1':
                # Test that if we have full-rank matrices, the resulting window functions
                # are indeed delta functions
                M, W = self.ds.get_MW(random_G, random_H, mode=mode, average_times=True)
                Hinv = np.linalg.inv(random_H)
                for i in range(n):
                    self.assertAlmostEqual(W[i,i], 1.)
                    for j in range(n):
                        self.assertAlmostEqual(M[i,j], Hinv[i,j])

                # When the matrices are not full rank, test that the window functions
                # are at least properly normalized.
                deficient_H = np.ones((3,3))
                M, W = self.ds.get_MW(deficient_H, deficient_H, mode=mode, average_times=True)
                norm = np.sum(W, axis=1)
                for i in range(3):
                    self.assertAlmostEqual(norm[i], 1.)

                # Check that a warning is raised when H matrix is not square
                rectangle_H = np.ones((4,3))
                nt.assert_raises(np.linalg.LinAlgError, self.ds.get_MW, random_G, rectangle_H, mode=mode, average_times=True)

                # Check that the method ignores G
                M, W = self.ds.get_MW(random_G, random_H, mode=mode, average_times=True)
                M_other, W_other = self.ds.get_MW(random_H, random_H, mode=mode, average_times=True)
                for i in range(n):
                    for j in range(n):
                        self.assertAlmostEqual(M[i,j], M_other[i,j])
                        self.assertAlmostEqual(W[i,j], W_other[i,j])

            elif mode == 'V^-1/2':
                # Test that we are checking for the presence of a covariance matrix
                nt.assert_raises(ValueError, self.ds.get_MW, random_G, random_H, mode=mode, average_times=True)
                # Test that the error covariance is diagonal
                M, W = self.ds.get_MW(random_G, random_H, mode=mode, band_covar=random_V, average_times=True)
                band_covar = np.dot(M, np.dot(random_V, M.T))
                self.assertEqual(diagonal_or_not(band_covar), True)

            elif mode == 'I':
                # Test that the norm matrix is diagonal
                M, W = self.ds.get_MW(random_G, random_H, mode=mode, average_times=True)
                self.assertEqual(diagonal_or_not(M), True)
            elif mode == 'L^-1':
                # Test that Cholesky mode is disabled
                nt.assert_raises(NotImplementedError,
                                 self.ds.get_MW, random_G, random_H, mode=mode, average_times=True)

            # Test sizes for everyone
            self.assertEqual(M.shape, (n,n))
            self.assertEqual(W.shape, (n,n))

            # Window function matrices should have each row sum to unity
            # regardless of the mode chosen
            test_norm = np.sum(W, axis=1)
            for norm in test_norm:
                self.assertAlmostEqual(norm, 1.)

    def test_cov_q(self, ndlys=13):
        """
        Test that q_hat_cov has the right shape and accepts keys in correct
        format. Also validate with arbitrary number of delays.
        """
        for d in self.d:
            d.flag_array[:] = False #ensure that there are no flags!
            d.select(times=np.unique(d.time_array)[:10], frequencies=d.freq_array[0, :16])
        for d_std in self.d_std:
            d_std.flag_array[:] = False
            d_std.select(times=np.unique(d_std.time_array)[:10], frequencies=d_std.freq_array[0, :16])
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, dsets_std=self.d_std)
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, dsets_std=self.d_std)
        Ntime = self.ds.Ntimes
        self.ds.set_Ndlys(ndlys)
        # Here is the analytic covariance matrix...
        chan_x, chan_y = np.meshgrid(range(self.ds.Nfreqs), range(self.ds.Nfreqs))
        cov_analytic = np.zeros((self.ds.spw_Ndlys, self.ds.spw_Ndlys), dtype=np.complex128)
        for alpha in range(self.ds.spw_Ndlys):
            for beta in range(self.ds.spw_Ndlys):
                cov_analytic[alpha, beta] = np.exp(-2j*np.pi*(alpha-beta)*(chan_x-chan_y)/self.ds.spw_Ndlys).sum()
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)
        #print(cov_analytic)

        for input_data_weight in ['identity','iC', 'dayenu']:
            self.ds.set_weighting(input_data_weight)
            #check error raised
            if input_data_weight == 'dayenu':
                nt.assert_raises(ValueError,self.ds.R, key1)
                rpk = {'filter_centers':[0.],'filter_half_widths':[0.],'filter_factors':[0.]}
                self.ds.set_r_param(key1,rpk)
                self.ds.set_r_param(key2,rpk)
            for taper in taper_selection:
                qc = self.ds.cov_q_hat(key1,key2,model='dsets')
                self.assertTrue(np.allclose(np.array(list(qc.shape)),
                np.array([self.ds.Ntimes, self.ds.spw_Ndlys, self.ds.spw_Ndlys]), atol=1e-6))
                qc = self.ds.cov_q_hat(key1,key2,model='empirical')
                self.assertTrue(np.allclose(np.array(list(qc.shape)),
                np.array([self.ds.Ntimes, self.ds.spw_Ndlys, self.ds.spw_Ndlys]), atol=1e-6))

        """
        Now test that analytic Error calculation gives Nchan^2
        """
        self.ds.set_weighting('identity')
        qc = self.ds.cov_q_hat(key1, key2, model='dsets')
        self.assertTrue(np.allclose(qc,
                        np.repeat(cov_analytic[np.newaxis, :, :], self.ds.Ntimes, axis=0), atol=1e-6))
        """
        Test lists of keys
        """
        self.ds.set_weighting('identity')
        qc=self.ds.cov_q_hat([key1], [key2], time_indices=[0], model='dsets')
        self.assertTrue(np.allclose(qc,
                        np.repeat(cov_analytic[np.newaxis, :, :], self.ds.Ntimes, axis=0), atol=1e-6))
        self.assertRaises(ValueError, self.ds.cov_q_hat, key1, key2, time_indices=200)
        self.assertRaises(ValueError, self.ds.cov_q_hat, key1, key2, time_indices="watch out!")


    def test_cov_p_hat(self):
        """
        Test cov_p_hat, verify on identity.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, dsets_std=self.d_std)
        cov_p = self.ds.cov_p_hat(np.asarray([np.sqrt(6.)*np.identity(10)]),
                                  np.array([5.*np.identity(10)]))
        for p in range(10):
            for q in range(10):
                if p == q:
                    self.assertTrue(np.isclose(30., cov_p[0, p, q], atol=1e-6))
                else:
                    self.assertTrue(np.isclose(0., cov_p[0, p, q], atol=1e-6))

    def test_R_interpolation(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.spw_Nfreqs
        Ntime = self.ds.Ntimes
        Ndlys = Nfreq - 20
        self.ds.spw_Ndlys = Ndlys


        # Set baselines to use for tests
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)

        rpk1 = {'filter_centers':[0.],'filter_widths':[100e-9],'filter_factors':[1e-9], 'restore_width':100e-9}
        self.ds.set_weighting('sinc_downweight')
        self.ds.set_r_param(key1,rpk1)
        df = np.mean(np.diff(self.ds.freqs))
        nd = 2 * int(df * 100e-9 / (Nfreq / Ndlys) * Nfreq)
        wgt = self.ds.w(key1)[:,0].squeeze()
        imat=dspec.delay_interpolation_matrix(Nfreq, nd, wgts=wgt, dres=Nfreq/Ndlys)
        fmati=dspec.sinc_downweight_mat_inv(nchan=Nfreq, df=df, filter_centers=[0.],
                                            filter_widths=[100e-9], filter_factors=[1e-9])
        fmat = np.linalg.pinv(fmati * np.outer(wgt, wgt))
        rmat = self.ds.R(key1)[0].squeeze()
        mymat = imat @ (np.identity(Nfreq, dtype=complex) - fmat) + fmat
        assert(np.all(np.isclose(mymat, rmat, atol=1e-3)))



    def test_R_truncation(self):
        """
        Test truncation of R-matrices. These should give a q_hat that is all
        zeros outside of the with f-start and f-end.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.spw_Nfreqs
        Ntime = self.ds.Ntimes
        Ndlys = Nfreq - 20
        self.ds.spw_Ndlys = Ndlys


        # Set baselines to use for tests
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)

        rpk1 = {'filter_centers':[0.],'filter_half_widths':[100e-9],'filter_factors':[1e-9]}
        rpk2 = {'filter_centers':[0.],'filter_half_widths':[100e-9],'filter_factors':[1e-9]}
        self.ds.set_weighting('dayenu')
        self.ds.set_r_param(key1,rpk1)
        self.ds.set_r_param(key2,rpk2)
        ds1 = copy.deepcopy(self.ds)
        ds1.set_spw((10,Nfreq-10))
        ds1.set_symmetric_taper(False)
        ds1.set_filter_extension([10,10])
        ds1.set_filter_extension((10,10))
        rm1 = self.ds.R(key1)
        rm2 = ds1.R(key2)
        rm3 = ds1.R(key1)
        for m in range(self.ds.Ntimes):
            self.assertTrue(np.shape(rm2[m]) == (ds1.spw_Nfreqs, self.ds.spw_Nfreqs))
            #check that all values that are not truncated match values of untrancated matrix.
            self.assertTrue(np.all(np.isclose(rm1[m][10:-10], rm2[m], atol=1e-6)))
            #make sure no errors are thrown by get_V, get_E, etc...
        ds1.get_unnormed_E(key1, key2, time_index=0)
        ds1.get_unnormed_V(key1, key2, time_index=0)
        h=ds1.get_H(key1, key2)
        g=ds1.get_G(key1, key2)
        print(h.shape)
        print(g.shape)
        print(ds1.spw_Nfreqs)
        print(ds1.spw_Ndlys)
        print(ds1.Ntimes)
        ds1.get_MW(g, h)
        #make sure identity weighting isn't broken.
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        ds1 = copy.deepcopy(self.ds)
        ds1.set_spw((10,Nfreq-10))
        ds1.set_weighting('identity')
        ds1.set_symmetric_taper(False)
        ds1.set_filter_extension([10,10])
        rm1 = ds1.R(key1)



    def test_q_hat(self):
        """
        Test that q_hat has right shape and accepts keys in the right format.
        """
        # Set weights and pack data into PSpecData
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        Ntime = self.ds.Ntimes
        Ndlys = Nfreq - 3
        self.ds.spw_Ndlys = Ndlys


        # Set baselines to use for tests
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)
        key3 = [(0, 24, 38), (0, 24, 38)]
        key4 = [(1, 25, 38), (1, 25, 38)]

        for input_data_weight in ['identity', 'iC', 'dayenu']:
            self.ds.set_weighting(input_data_weight)
            if input_data_weight == 'dayenu':
                nt.assert_raises(ValueError,self.ds.R, key1)
                rpk = {'filter_centers':[0.],'filter_half_widths':[0.],'filter_factors':[0.]}
                self.ds.set_r_param(key1,rpk)
                self.ds.set_r_param(key2,rpk)
            # Loop over list of taper functions
            for taper in taper_selection:
                self.ds.set_taper(taper)

                # Calculate q_hat for a pair of baselines and test output shape
                q_hat_a = self.ds.q_hat(key1, key2)
                self.assertEqual(q_hat_a.shape, (Ndlys, Ntime))


                # Check that swapping x_1 <-> x_2 results in complex conj. only
                q_hat_b = self.ds.q_hat(key2, key1)
                q_hat_diff = np.conjugate(q_hat_a) - q_hat_b
                for i in range(Ndlys):
                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_diff[i,j].real,
                                               q_hat_diff[i,j].real)
                        self.assertAlmostEqual(q_hat_diff[i,j].imag,
                                               q_hat_diff[i,j].imag)

                # Check that lists of keys are handled properly
                q_hat_aa = self.ds.q_hat(key1, key4) # q_hat(x1, x2+x2)
                q_hat_bb = self.ds.q_hat(key4, key1) # q_hat(x2+x2, x1)
                q_hat_cc = self.ds.q_hat(key3, key4) # q_hat(x1+x1, x2+x2)

                # Effectively checks that q_hat(2*x1, 2*x2) = 4*q_hat(x1, x2)
                for i in range(Ndlys):

                    for j in range(Ntime):
                        self.assertAlmostEqual(q_hat_a[i,j].real,
                                               0.25 * q_hat_cc[i,j].real)
                        self.assertAlmostEqual(q_hat_a[i,j].imag,
                                               0.25 * q_hat_cc[i,j].imag)


        self.ds.spw_Ndlys = Nfreq
        # Check that the slow method is the same as the FFT method
        for input_data_weight in ['identity', 'iC', 'dayenu']:
            self.ds.set_weighting(input_data_weight)
            # Loop over list of taper functions
            for taper in taper_selection:

                self.ds.set_taper(taper)
                q_hat_a_slow = self.ds.q_hat(key1, key2, allow_fft=False)
                q_hat_a = self.ds.q_hat(key1, key2, allow_fft=True)
                self.assertTrue(np.isclose(np.real(q_hat_a/q_hat_a_slow), 1).all())
                self.assertTrue(np.isclose(np.imag(q_hat_a/q_hat_a_slow), 0, atol=1e-6).all())

        #Test if error is raised when one tried FFT approach on exact_norm
        nt.assert_raises(NotImplementedError, self.ds.q_hat, key1, key2, exact_norm=True, allow_fft = True)

    def test_get_H(self):
        """
        Test Fisher/weight matrix calculation.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        multiplicative_tolerance = 1.
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)

        for input_data_weight in ['identity','iC', 'dayenu']:
            self.ds.set_weighting(input_data_weight)
            if input_data_weight == 'dayenu':
                nt.assert_raises(ValueError,self.ds.R, key1)
                rpk = {'filter_centers':[0.],'filter_half_widths':[0.],'filter_factors':[0.]}
                self.ds.set_r_param(key1,rpk)
                self.ds.set_r_param(key2,rpk)
            for taper in taper_selection:
                self.ds.set_taper(taper)

                self.ds.set_Ndlys(Nfreq//3)
                H = self.ds.get_H(key1, key2, average_times=True)
                self.assertEqual(H.shape, (Nfreq//3, Nfreq//3)) # Test shape

                self.ds.set_Ndlys()
                H = self.ds.get_H(key1, key2, average_times=True)
                self.assertEqual(H.shape, (Nfreq, Nfreq)) # Test shape

    def test_get_G(self):
        """
        Test Fisher/weight matrix calculation.
        """
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w)
        Nfreq = self.ds.Nfreqs
        multiplicative_tolerance = 1.
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)

        for input_data_weight in ['identity','iC', 'dayenu']:
            self.ds.set_weighting(input_data_weight)
            if input_data_weight == 'dayenu':
                nt.assert_raises(ValueError,self.ds.R, key1)
                rpk = {'filter_centers':[0.],'filter_half_widths':[0.],'filter_factors':[0.]}
                self.ds.set_r_param(key1,rpk)
                self.ds.set_r_param(key2,rpk)
            for taper in taper_selection:
                self.ds.clear_cache()
                self.ds.set_taper(taper)
                #print 'input_data_weight', input_data_weight
                self.ds.set_Ndlys(Nfreq-2)
                G = self.ds.get_G(key1, key2, average_times=True)
                self.assertEqual(G.shape, (Nfreq-2, Nfreq-2)) # Test shape
                #print np.min(np.abs(G)), np.min(np.abs(np.linalg.eigvalsh(G)))
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
                    G_swapped = self.ds.get_G(key2, key1, average_times=True)
                    G_diff_norm = np.linalg.norm(G - G_swapped)
                    self.assertLessEqual(G_diff_norm,
                                        matrix_scale * multiplicative_tolerance)
                    min_diagonal = np.min(np.diagonal(G))

                    # Test that all elements of G are positive up to numerical
                    # noise with the threshold set to 10 orders of magnitude
                    # down from the smallest value on the diagonal
                    for i in range(Nfreq-2):
                        for j in range(Nfreq-2):
                            self.assertGreaterEqual(G[i,j],

                                       -min_diagonal * multiplicative_tolerance)
                else:
                    # In general, when R_1 != R_2, there is a more restricted
                    # symmetry where swapping R_1 and R_2 *and* taking the
                    # transpose gives the same result
                    #UPDATE: Taper now occurs after filter so this
                    #symmetry only holds when taper = 'none'.
                    if taper_selection == 'none':
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
        data.select(freq_chans=range(Nfreq), bls=[(24,24),])
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

    def test_scalar_delay_adjustment(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=self.bm)
        key1 = (0, 24, 38)
        key2 = (1, 25, 38)

        # Test that when:
        # i) Nfreqs = Ndlys, ii) Sampling, iii) No tapering, iv) R is identity
        # are all satisfied, the scalar adjustment factor is unity
        self.ds.set_weighting('identity')
        self.ds.spw_Ndlys = self.ds.spw_Nfreqs
        adjustment = self.ds.scalar_delay_adjustment(key1, key2, sampling=True)
        self.assertAlmostEqual(adjustment, 1.0)
        self.ds.set_weighting('iC')
        #if weighting is not identity, then the adjustment should be a vector.
        adjustment = self.ds.scalar_delay_adjustment(key1, key2, sampling=True)
        self.assertTrue(len(adjustment) == self.ds.spw_Ndlys)

    def test_scalar(self):
        self.ds = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=self.bm)

        gauss = pspecbeam.PSpecBeamGauss(0.8,
                                  np.linspace(115e6, 130e6, 50, endpoint=False))
        ds2 = pspecdata.PSpecData(dsets=self.d, wgts=self.w, beam=gauss)

        # Check normal execution
        scalar = self.ds.scalar(('xx','xx'))
        scalar_xx = self.ds.scalar('xx') # Can use single pol string as shorthand
        nt.assert_equal(scalar, scalar_xx)
        scalar = self.ds.scalar(1515) # polpair-integer = ('xx', 'xx')
        scalar = self.ds.scalar(('xx','xx'), taper_override='none')
        scalar = self.ds.scalar(('xx','xx'), beam=gauss)
        nt.assert_raises(NotImplementedError, self.ds.scalar, ('xx','yy'))

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
        uvd2 = uvd.select(frequencies=np.unique(uvd.freq_array)[:10],
                          inplace=False)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)

        # test time exception
        uvd2 = uvd.select(times=np.unique(uvd.time_array)[:10], inplace=False)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)

        # test label exception
        _labels = ds.labels
        ds.labels = ds.labels[:1]
        nt.assert_raises(ValueError, ds.validate_datasets)
        ds.labels = _labels

        # test std exception
        _std = ds.dsets_std
        ds.dsets_std = ds.dsets_std[:1]
        nt.assert_raises(ValueError, ds.validate_datasets)
        ds.dsets_std = _std

        # test wgt exception
        _wgts = ds.wgts
        ds.wgts = ds.wgts[:1]
        nt.assert_raises(ValueError, ds.validate_datasets)
        ds.wgts = _wgts

        # test warnings
        uvd = copy.deepcopy(self.d[0])
        uvd2 = copy.deepcopy(self.d[0])
        uvd.select(frequencies=np.unique(uvd.freq_array)[:10],
                   times=np.unique(uvd.time_array)[:10])
        uvd2.select(frequencies=np.unique(uvd2.freq_array)[10:20],
                    times=np.unique(uvd2.time_array)[10:20])
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        ds.validate_datasets()

        # test phasing
        uvd = copy.deepcopy(self.d[0])
        uvd2 = copy.deepcopy(self.d[0])
        uvd.phase_to_time(Time(2458042, format='jd'))
        ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)
        uvd2.phase_to_time(Time(2458042.5, format='jd'))
        ds.validate_datasets()

        # test polarization
        ds.validate_pol((0,1), ('xx', 'xx'))

        # test channel widths
        uvd2.channel_width *= 2.
        ds2 = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds2.validate_datasets)

    def test_rephase_to_dset(self):
        # get uvd
        uvd1 = copy.deepcopy(self.uvd)

        # give the uvd an x_orientation to test x_orientation propagation
        uvd1.x_orienation = 'east'

        # null test: check nothing changes when dsets contain same UVData object
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd1)], wgts=[None, None])
        # get normal pspec
        bls = [(37, 39)]
        uvp1 = ds.pspec(bls, bls, (0, 1), pols=('xx','xx'), verbose=False)
        # rephase and get pspec
        ds.rephase_to_dset(0)
        uvp2 = ds.pspec(bls, bls, (0, 1), pols=('xx','xx'), verbose=False)
        blp = (0, ((37,39),(37,39)), ('xx','xx'))
        nt.assert_true(np.isclose(np.abs(uvp2.get_data(blp)/uvp1.get_data(blp)), 1.0).min())

    def test_Jy_to_mK(self):
        # test basic execution
        uvd = self.uvd
        uvd.vis_units = 'Jy'
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
                                 wgts=[None, None], beam=self.bm)
        ds.Jy_to_mK()
        nt.assert_true(ds.dsets[0].vis_units, 'mK')
        nt.assert_true(ds.dsets[1].vis_units, 'mK')
        nt.assert_true(uvd.get_data(24, 25, 'xx')[30, 30] \
                    / ds.dsets[0].get_data(24, 25, 'xx')[30, 30] < 1.0)

        # test feeding beam
        ds2 = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
                                 wgts=[None, None], beam=self.bm)
        ds2.Jy_to_mK(beam=self.bm)
        nt.assert_equal(ds.dsets[0], ds2.dsets[0])

        # test vis_units no Jansky
        uvd2 = copy.deepcopy(uvd)
        uvd2.polarization_array[0] = -6
        uvd2.vis_units = 'UNCALIB'
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd2)],
                                 wgts=[None, None], beam=self.bm)
        ds.Jy_to_mK()
        nt.assert_equal(ds.dsets[0].vis_units, "mK")
        nt.assert_equal(ds.dsets[1].vis_units, "UNCALIB")
        nt.assert_not_equal(ds.dsets[0].get_data(24, 25, 'xx')[30, 30], ds.dsets[1].get_data(24, 25, 'yy')[30, 30])

    def test_trim_dset_lsts(self):
        fname = os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA")
        uvd1 = UVData()
        uvd1.read_miriad(fname)
        uvd2 = copy.deepcopy(uvd1)
        uvd2.lst_array = (uvd2.lst_array + 10 * np.median(np.diff(np.unique(uvd2.lst_array)))) % (2*np.pi)
        # test basic execution
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None])
        ds.trim_dset_lsts()
        nt.assert_true(ds.dsets[0].Ntimes, 52)
        nt.assert_true(ds.dsets[1].Ntimes, 52)
        nt.assert_true(np.all( (2458042.178948477 < ds.dsets[0].time_array) \
                        + (ds.dsets[0].time_array < 2458042.1843023109)))
        # test exception
        uvd2.lst_array += np.linspace(0, 1e-3, uvd2.Nblts)
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None])
        ds.trim_dset_lsts()
        nt.assert_true(ds.dsets[0].Ntimes, 60)
        nt.assert_true(ds.dsets[1].Ntimes, 60)

    def test_units(self):
        ds = pspecdata.PSpecData()
        # test exception
        nt.assert_raises(IndexError, ds.units)
        ds.add(self.uvd, None)
        # test basic execution
        vis_u, norm_u = ds.units(little_h=False)
        vis_u, norm_u = ds.units()
        nt.assert_equal(vis_u, "UNCALIB")
        nt.assert_equal(norm_u, "Hz str [beam normalization not specified]")
        ds_b = pspecdata.PSpecData(dsets=[self.uvd, self.uvd],
                wgts=[None, None], beam=self.bm)
        vis_u, norm_u = ds_b.units(little_h=False)
        nt.assert_equal(norm_u,"Mpc^3")

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

        # test dset_idx
        nt.assert_raises(TypeError, ds.dset_idx, (1,2))

    def test_pspec(self):
        # generate ds
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None],beam=self.bm, labels=['red', 'blue'])

        # check basic execution with baseline list
        bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
        uvp = ds.pspec(bls, bls, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        nt.assert_equal(len(uvp.bl_array), len(bls))
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (24, 25))) in uvp.blpair_array)
        nt.assert_equal(uvp.data_array[0].dtype, np.complex128)
        nt.assert_equal(uvp.data_array[0].shape, (240, 64, 1))

        #test for different forms of input parameters
        ds.pspec(bls, bls, (0, 1), ('xx','xx'), spw_ranges=(10,20))
        ds.pspec(bls, bls, (0, 1), ('xx','xx'), n_dlys=10, spw_ranges=[(10,20)])
        ds.pspec(bls, bls, (0, 1), ('xx','xx'), n_dlys=1)

        my_r_params = {}
        my_r_params_dset0_only = {}
        rp = {'filter_centers':[0.],
              'filter_half_widths':[250e-9],
              'filter_factors':[1e-9]}
        for bl in bls:
            key1 = (0,) + bl + ('xx',)
            key2 = (1,) + bl + ('xx',)
            my_r_params[key1] = rp
            my_r_params_dset0_only[key1] = rp
            my_r_params[key2] = rp
        #test inverse sinc weighting.
        ds.pspec(bls,bls,(0, 1), ('xx','xx'),
        spw_ranges = (10,20), input_data_weight  = 'dayenu',
        r_params = my_r_params)
        #test value error
        nt.assert_raises(ValueError, ds.pspec, bls, bls, (0, 1), ('xx','xx'),
        spw_ranges = (10,20), input_data_weight  = 'dayenu', r_params = {})
        #test value error no dset1 keys
        nt.assert_raises(ValueError, ds.pspec, bls, bls, (0, 1), ('xx','xx'),
        spw_ranges = (10,20), input_data_weight  = 'dayenu',
        r_params = my_r_params_dset0_only)

        #assert error if baselines are not provided in the right format
        nt.assert_raises(NotImplementedError, ds.pspec, [[(24,25),(38,39)]],[[(24,25),(38,39)]],
                (0,1),[('xx','xx')])

        # compare the output of get_Q function with analytical estimates

        ds_Q  = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None],beam=self.bm_Q)
        bls_Q   = [(24, 25)]
        uvp = ds_Q.pspec(bls_Q, bls_Q, (0, 1), [('xx', 'xx')], input_data_weight='identity',
                                       norm='I', taper='none', verbose=True, exact_norm=False)
        Q_sample = ds_Q.get_integral_beam('xx') #Get integral beam for pol 'xx'

        nt.assert_equal(np.shape(Q_sample), (ds_Q.spw_range[1] - ds_Q.spw_range[0],\
                                             ds_Q.spw_range[1] - ds_Q.spw_range[0])) #Check for the right shape

        estimated_Q = (1.0/(4*np.pi)) * np.ones_like(Q_sample)

        nt.assert_true(np.allclose(np.real(estimated_Q), np.real(Q_sample), rtol=1e-05))

        #Test if the two pipelines match

        ds_t    = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm_Q)
        uvp_new = ds_t.pspec(bls_Q, bls_Q, (0, 1), [('xx', 'xx')], input_data_weight='identity',
                                       norm='I', taper='none', verbose=True, exact_norm=True)
        uvp_ext = ds_t.pspec(bls_Q, bls_Q, (0, 1), [('xx', 'xx')], input_data_weight='identity',
                                       norm='I', taper='none', verbose=True, exact_norm=False)
        spw         = 0
        blp         = (bls_Q[0], bls_Q[0])
        key         = (spw, blp, 'xx')
        power_real_new  = (np.real(uvp_new.get_data(key)))
        power_real_ext  = (np.real(uvp_ext.get_data(key)))

        diff = np.median((power_real_new-power_real_ext)/power_real_ext)
        nt.assert_true((diff <= 0.05))

        # check with redundant baseline group list
        antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
        antpos = dict(zip(ants, antpos))
        red_bls = [sorted(blg) for blg in redcal.get_pos_reds(antpos)][2]
        bls1, bls2, blps = utils.construct_blpairs(red_bls, exclude_permutations=True)
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (37, 38))) in uvp.blpair_array)
        nt.assert_equal(uvp.Nblpairs, 10)
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        nt.assert_true(uvp.antnums_to_blpair(((24, 25), (52, 53))) in uvp.blpair_array)
        nt.assert_true(uvp.antnums_to_blpair(((52, 53), (24, 25))) not in uvp.blpair_array)
        nt.assert_equal(uvp.Nblpairs, 10)

        # test mixed bl group and non blgroup, currently bl grouping of more than 1 blpair doesn't work
        bls1 = [[(24, 25)], (52, 53)]
        bls2 = [[(24, 25)], (52, 53)]
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        # test select
        red_bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
        bls1, bls2, blp = utils.construct_blpairs(red_bls, exclude_permutations=False, exclude_auto_bls=False)
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), spw_ranges=[(20,30), (30,40)], verbose=False)
        nt.assert_equal(uvp.Nblpairs, 16)
        nt.assert_equal(uvp.Nspws, 2)
        uvp2 = uvp.select(spws=0, bls=[(24, 25)], only_pairs_in_bls=False, inplace=False)
        nt.assert_equal(uvp2.Nspws, 1)
        nt.assert_equal(uvp2.Nblpairs, 7)
        uvp.select(spws=0, bls=(24, 25), only_pairs_in_bls=True, inplace=True)
        nt.assert_equal(uvp.Nspws, 1)
        nt.assert_equal(uvp.Nblpairs, 1)

        # check w/ multiple spectral ranges
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), ('xx','xx'), spw_ranges=[(10, 24), (30, 40), (45, 64)], verbose=False)
        nt.assert_equal(uvp.Nspws, 3)
        nt.assert_equal(uvp.Nspwdlys, 43)
        nt.assert_equal(uvp.data_array[0].shape, (240, 14, 1))
        nt.assert_equal(uvp.get_data((0, 124125124125, ('xx','xx'))).shape, (60, 14))

        uvp.select(spws=[1])
        nt.assert_equal(uvp.Nspws, 1)
        nt.assert_equal(uvp.Ndlys, 10)
        nt.assert_equal(len(uvp.data_array), 1)

        # test polarization pairs
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), ('xx','xx'), spw_ranges=[(10, 24)], verbose=False)
        #nt.assert_raises(NotImplementedError, ds.pspec, bls, bls, (0, 1), pols=[('xx','yy')])
        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), [('xx','xx'), ('yy','yy')], spw_ranges=[(10, 24)], verbose=False)

        uvd = copy.deepcopy(self.uvd)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), (-5, -5), spw_ranges=[(10, 24)], verbose=False)

        # test exceptions
        nt.assert_raises(AssertionError, ds.pspec, bls1[:1], bls2, (0, 1), ('xx','xx'))
        nt.assert_raises(ValueError, ds.pspec, bls, bls, (0, 1), pols=('yy','yy'))
        uvd1 = copy.deepcopy(self.uvd)
        uvd1.polarization_array = np.array([-6])
        ds = pspecdata.PSpecData(dsets=[uvd, uvd1], wgts=[None, None], beam=self.bm)
        nt.assert_raises(ValueError, ds.pspec, bls, bls, (0, 1), ('xx','xx'))

        # test files with more than one polarizations
        uvd1 = copy.deepcopy(self.uvd)
        uvd1.polarization_array = np.array([-6])
        uvd2 = self.uvd + uvd1
        ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), [('xx','xx'), ('yy','yy')], spw_ranges=[(10, 24)], verbose=False)

        uvd1 = copy.deepcopy(self.uvd)
        uvd1.polarization_array = np.array([-6])
        uvd2 = self.uvd + uvd1
        ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec(bls, bls, (0, 1), [('xx','xx'), ('xy','xy')], spw_ranges=[(10, 24)], verbose=False)

        # test with nsamp set to zero
        uvd = copy.deepcopy(self.uvd)
        uvd.nsample_array[uvd.antpair2ind(24, 25, ordered=False)] = 0.0
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        uvp = ds.pspec([(24, 25)], [(37, 38)], (0, 1), [('xx', 'xx')])
        nt.assert_true(np.all(np.isclose(uvp.integration_array[0], 0.0)))

        # test covariance calculation runs with small number of delays
        uvd = copy.deepcopy(self.uvd)
        uvd_std = copy.deepcopy(self.uvd_std)
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None],
                                 dsets_std=[uvd_std, uvd_std], beam=self.bm)
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=True, spw_ranges=[(10,14)], store_cov=True)
        nt.assert_true(hasattr(uvp, 'cov_array'))

        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=True, spw_ranges=[(10,14)], store_cov=True)
        nt.assert_true(hasattr(uvp, 'cov_array'))
        # test identity_Y caching works
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(self.uvd), copy.deepcopy(self.uvd)], wgts=[None, None],
                                 beam=self.bm)
        # assert caching is used when appropriate
        uvp = ds.pspec([(24, 25), (24, 25)], [(24, 25), (24, 25)], (0, 1), ('xx', 'xx'),
                       input_data_weight='identity', norm='I', taper='none', verbose=False,
                       spw_ranges=[(20, 30)])
        nt.assert_equal(len(ds._identity_Y), len(ds._identity_G), len(ds._identity_H))
        nt.assert_equal(len(ds._identity_Y), 1)
        nt.assert_equal(list(ds._identity_Y.keys())[0], ((0, 24, 25, 'xx'), (1, 24, 25, 'xx')))
        # assert caching is not used when inappropriate
        ds.dsets[0].flag_array[ds.dsets[0].antpair2ind(37, 38, ordered=False), :, 25, :] = True
        uvp = ds.pspec([(24, 25), (37, 38)], [(24, 25), (37, 38)], (0, 1), ('xx', 'xx'),
                       input_data_weight='identity', norm='I', taper='none', verbose=False,
                       spw_ranges=[(20, 30)])
        nt.assert_equal(len(ds._identity_Y), len(ds._identity_G), len(ds._identity_H))
        nt.assert_equal(len(ds._identity_Y), 2)
        nt.assert_true(((0, 24, 25, 'xx'), (1, 24, 25, 'xx')) in ds._identity_Y.keys())
        nt.assert_true(((0, 37, 38, 'xx'), (1, 37, 38, 'xx')) in ds._identity_Y.keys())

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
        d1.data_array *= beam.Jy_to_mK(freqs, pol='XX')[None, None, :, None]
        d2.data_array *= beam.Jy_to_mK(freqs, pol='XX')[None, None, :, None]

        # Compare using no taper
        OmegaP = beam.power_beam_int(pol='XX')
        OmegaPP = beam.power_beam_sq_int(pol='XX')
        OmegaP = interp1d(beam.beam_freqs/1e6, OmegaP)(freqs/1e6)
        OmegaPP = interp1d(beam.beam_freqs/1e6, OmegaPP)(freqs/1e6)
        NEB = 1.0
        Bp = np.median(np.diff(freqs)) * len(freqs)
        scalar = cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2/OmegaPP) * Bp * NEB
        data1 = d1.get_data(bls1[0])
        data2 = d2.get_data(bls2[0])
        legacy = np.fft.fftshift(np.conj(np.fft.fft(data1, axis=1)) * np.fft.fft(data2, axis=1) * scalar / len(freqs)**2, axes=1)[0]
        # hera_pspec OQE
        ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
        uvp = ds.pspec(bls1, bls2, (0, 1), pols=('xx','xx'), taper='none', input_data_weight='identity', norm='I', sampling=True)
        oqe = uvp.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]
        # assert answers are same to within 3%
        nt.assert_true(np.isclose(np.real(oqe)/np.real(legacy), 1, atol=0.03, rtol=0.03).all())
        # taper
        window = windows.blackmanharris(len(freqs))
        NEB = Bp / trapz(window**2, x=freqs)
        scalar = cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2/OmegaPP) * Bp * NEB
        data1 = d1.get_data(bls1[0])
        data2 = d2.get_data(bls2[0])
        legacy = np.fft.fftshift(np.conj(np.fft.fft(data1*window[None, :], axis=1)) * np.fft.fft(data2*window[None, :], axis=1) * scalar / len(freqs)**2, axes=1)[0]
        # hera_pspec OQE
        ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
        uvp = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), taper='blackman-harris', input_data_weight='identity', norm='I')
        oqe = uvp.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]
        # assert answers are same to within 3%
        nt.assert_true(np.isclose(np.real(oqe)/np.real(legacy), 1, atol=0.03, rtol=0.03).all())

    def test_broadcast_dset_flags(self):
        # setup
        fname = os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA")
        uvd = UVData()
        uvd.read_miriad(fname)
        Nfreq = uvd.data_array.shape[2]

        # test basic execution w/ a spw selection
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None])
        ds.broadcast_dset_flags(spw_ranges=[(400, 800)], time_thresh=0.2)
        nt.assert_false(ds.dsets[0].get_flags(24, 25)[:, 550:650].any())

        # test w/ no spw selection
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None])
        ds.broadcast_dset_flags(spw_ranges=None, time_thresh=0.2)
        nt.assert_true(ds.dsets[0].get_flags(24, 25)[:, 550:650].any())

        # test unflagging
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None])
        ds.broadcast_dset_flags(spw_ranges=None, time_thresh=0.2, unflag=True)
        nt.assert_false(ds.dsets[0].get_flags(24, 25)[:, :].any())

        # test single integration being flagged within spw
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None])
        ds.dsets[0].flag_array[ds.dsets[0].antpair2ind(24, 25, ordered=False)[3], 0, 600, 0] = True
        ds.broadcast_dset_flags(spw_ranges=[(400, 800)], time_thresh=0.25, unflag=False)
        nt.assert_true(ds.dsets[0].get_flags(24, 25)[3, 400:800].all())
        nt.assert_false(ds.dsets[0].get_flags(24, 25)[3, :].all())

        # test pspec run sets flagged integration to have zero weight
        uvd.flag_array[uvd.antpair2ind(24, 25, ordered=False)[3], 0, 400, :] = True
        ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None])
        ds.broadcast_dset_flags(spw_ranges=[(400, 450)], time_thresh=0.25)
        uvp = ds.pspec([(24, 25), (37, 38), (38, 39)], [(24, 25), (37, 38), (38, 39)], (0, 1), ('xx', 'xx'),
                        spw_ranges=[(400, 450)], verbose=False)
        # assert flag broadcast above hits weight arrays in uvp
        nt.assert_true(np.all(np.isclose(uvp.get_wgts((0, ((24, 25), (24, 25)), ('xx','xx')))[3], 0.0)))
        # assert flag broadcast above hits integration arrays
        nt.assert_true(np.isclose(uvp.get_integrations((0, ((24, 25), (24, 25)), ('xx','xx')))[3], 0.0))
        # average spectra
        avg_uvp = uvp.average_spectra(blpair_groups=[sorted(np.unique(uvp.blpair_array))], time_avg=True, inplace=False)
        # repeat but change data in flagged portion
        ds.dsets[0].data_array[uvd.antpair2ind(24, 25, ordered=False)[3], 0, 400:450, :] *= 100
        uvp2 = ds.pspec([(24, 25), (37, 38), (38, 39)], [(24, 25), (37, 38), (38, 39)], (0, 1), ('xx', 'xx'),
                        spw_ranges=[(400, 450)], verbose=False)
        avg_uvp2 = uvp.average_spectra(blpair_groups=[sorted(np.unique(uvp.blpair_array))], time_avg=True, inplace=False)
        # assert average before and after are the same!
        print(avg_uvp.data_array)
        print(avg_uvp2.data_array)
        nt.assert_equal(avg_uvp, avg_uvp2)

    def test_RFI_flag_propagation(self):
        # generate ds and weights
        uvd = copy.deepcopy(self.uvd)
        uvd.flag_array[:] = False
        Nfreq = uvd.data_array.shape[2]
        # Basic test of shape
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm)
        test_R = ds.R((1, 37, 38, 'XX'), average_times=True)
        nt.assert_equal(test_R.shape, (Nfreq, Nfreq))
        # First test that turning-off flagging does nothing if there are no flags in the data
        bls1 = [(24, 25)]
        bls2 = [(37, 38)]
        ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=self.bm, labels=['red', 'blue'])
        uvp_flagged = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)
        ds.broadcast_dset_flags(unflag=True)
        uvp_unflagged = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)

        qe_unflagged = uvp_unflagged.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]
        qe_flagged = uvp_flagged.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]
        # assert answers are same to within 0.1%
        nt.assert_true(np.isclose(np.real(qe_unflagged)/np.real(qe_flagged), 1, atol=0.001, rtol=0.001).all())

        # Test that when flagged, the data within a channel really don't have any effect on the final result
        uvd2 = copy.deepcopy(uvd)
        uvd2.flag_array[uvd.antpair2ind(24, 25, ordered=False)] = True
        ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=self.bm)
        uvp_flagged = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)

        uvd2.data_array[uvd.antpair2ind(24, 25, ordered=False)] *= 9234.913
        ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=self.bm)
        uvp_flagged_mod = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='identity', norm='I', taper='none',
                                little_h=True, verbose=False)

        qe_flagged_mod = uvp_flagged_mod.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]
        qe_flagged = uvp_flagged.get_data((0, ((24, 25), (37, 38)), ('xx','xx')))[0]

        # assert answers are same to within 0.1%
        nt.assert_true(np.isclose(np.real(qe_flagged_mod), np.real(qe_flagged), atol=0.001, rtol=0.001).all())

        # Test below commented out because this sort of aggressive symmetrization is not yet implemented.
        # # Test that flagging a channel for one dataset (e.g. just left hand dataset x2)
        # # is equivalent to flagging for both x1 and x2.
        # test_wgts_flagged = copy.deepcopy(test_wgts)
        # test_wgts_flagged.data_array[:,:,40:60] = 0. # Flag 20 channels
        # ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[test_wgts_flagged, test_wgts_flagged], beam=self.bm)
        # print "mode alpha"
        # uvp_flagged = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='diagonal', norm='I', taper='none',
        #                         little_h=True, verbose=False)
        # ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, test_wgts_flagged], beam=self.bm)
        # print "mode beta"
        # uvp_flagged_asymm = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='diagonal', norm='I', taper='none',
        #                         little_h=True, verbose=False)

        # qe_flagged_asymm = uvp_flagged_asymm .get_data(0, ((24, 25), (37, 38)), 'xx')[0]
        # qe_flagged = uvp_flagged.get_data(0, ((24, 25), (37, 38)), 'xx')[0]

        # #print np.real(qe_flagged_asymm)/np.real(qe_flagged)

        # # assert answers are same to within 3%
        # nt.assert_true(np.isclose(np.real(qe_flagged_asymm)/np.real(qe_flagged), 1, atol=0.03, rtol=0.03).all())

        #print(uvd.data_array.shape)

    def test_validate_blpairs(self):
        # test exceptions
        uvd = copy.deepcopy(self.uvd)
        nt.assert_raises(TypeError, pspecdata.validate_blpairs, [((1, 2), (2, 3))], None, uvd)
        nt.assert_raises(TypeError, pspecdata.validate_blpairs, [((1, 2), (2, 3))], uvd, None)

        bls = [(24,25),(37,38)]
        bls1, bls2, blpairs = utils.construct_blpairs(bls, exclude_permutations=False, exclude_auto_bls=True)
        pspecdata.validate_blpairs(blpairs, uvd, uvd)
        bls1, bls2, blpairs = utils.construct_blpairs(bls, exclude_permutations=False, exclude_auto_bls=True,
                                                          group=True)

        pspecdata.validate_blpairs(blpairs, uvd, uvd)

        # test non-redundant
        blpairs = [((24, 25), (24, 38))]
        pspecdata.validate_blpairs(blpairs, uvd, uvd)


def test_pspec_run():
    fnames = [os.path.join(DATA_PATH, d)
              for d in ['zen.even.xx.LST.1.28828.uvOCRSA',
                        'zen.odd.xx.LST.1.28828.uvOCRSA']]

    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    fnames_std = [os.path.join(DATA_PATH,d)
                  for d in ['zen.even.std.xx.LST.1.28828.uvOCRSA',
                            'zen.odd.std.xx.LST.1.28828.uvOCRSA']]

    # test basic execution
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    ds = pspecdata.pspec_run(fnames, "./out.h5", Jy2mK=False,
                             verbose=False, overwrite=True, dset_pairs=[(0, 1)],
                             bl_len_range=(14, 15), bl_deg_range=(50, 70),
                             psname_ext='_0', spw_ranges=[(0, 25)])
    psc =  container.PSpecContainer('./out.h5')
    nt.assert_true(isinstance(psc, container.PSpecContainer))
    nt.assert_equal(psc.groups(), ['dset0_dset1'])
    nt.assert_equal(psc.spectra(psc.groups()[0]), ['dset0_x_dset1_0'])
    nt.assert_true(os.path.exists("./out.h5"))

    # test Jy2mK, blpairs, cosmo, cov_array, spw_ranges, dset labeling
    cosmo = conversions.Cosmo_Conversions(Om_L=0.0)
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    ds = pspecdata.pspec_run(fnames, "./out.h5",
                             dsets_std=fnames_std,
                             Jy2mK=True,
                             beam=beamfile,
                             blpairs=[((37, 38), (37, 38)),
                                      ((37, 38), (52, 53))],
                             verbose=False,
                             overwrite=True,
                             pol_pairs=[('xx', 'xx'), ('xx', 'xx')],
                             dset_labels=["foo", "bar"],
                             dset_pairs=[(0, 0), (0, 1)],
                             spw_ranges=[(50, 75), (120, 140)],
                             n_dlys=[20, 20],
                             cosmo=cosmo,
                             trim_dset_lsts=False,
                             broadcast_dset_flags=False,
                             store_cov=True)

    # assert groupname is dset1_dset2
    psc =  container.PSpecContainer('./out.h5')
    nt.assert_true("foo_bar" in psc.groups())

    # assert uvp names are labeled by dset_pairs
    nt.assert_equal(sorted(psc.spectra('foo_bar')),
                    sorted([u'foo_x_bar', u'foo_x_foo']))

    # get UVPSpec for further inspection
    uvp = psc.get_pspec("foo_bar", "foo_x_bar")

    # assert Jy2mK worked
    nt.assert_true(uvp.vis_units, "mK")

    # assert only blpairs that were fed are present
    nt.assert_equal(uvp.bl_array.tolist(), [137138, 152153])
    nt.assert_equal(uvp.polpair_array.tolist(), [1515, 1515])

    # assert weird cosmology was passed
    nt.assert_equal(uvp.cosmo, cosmo)

    # assert cov_array was calculated b/c std files were passed and store_cov
    nt.assert_true(hasattr(uvp, 'cov_array'))

    # assert dset labeling propagated
    nt.assert_equal(set(uvp.labels), set(['bar', 'foo']))

    # assert spw_ranges and n_dlys specification worked
    np.testing.assert_array_equal(uvp.get_spw_ranges(), [(163476562.5, 165917968.75, 25, 20), (170312500.0, 172265625.0, 20, 20)])

    # test single_dset, time_interleaving, rephasing, flag broadcasting
    uvd = UVData()
    uvd.read_miriad(fnames[0])
    # interleave the data by hand, and add some flags in
    uvd.flag_array[:] = False
    uvd.flag_array[uvd.antpair2ind(37, 38, ordered=False)[0], 0, 10, 0] = True
    uvd.flag_array[uvd.antpair2ind(37, 38, ordered=False)[:3], 0, 15, 0] = True
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[::2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    if os.path.exists("./out2.h5"):
        os.remove("./out2.h5")
    ds = pspecdata.pspec_run([copy.deepcopy(uvd)], "./out2.h5",
                             blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))], interleave_times=True,
                             verbose=False, overwrite=True, spw_ranges=[(0, 25)], rephase_to_dset=0,
                             broadcast_dset_flags=True, time_thresh=0.3)
    psc =  container.PSpecContainer('./out2.h5')
    nt.assert_true(isinstance(psc, container.PSpecContainer))
    nt.assert_equal(psc.groups(), ['dset0_dset1'])
    nt.assert_equal(psc.spectra(psc.groups()[0]), ['dset0_x_dset1'])
    nt.assert_true(os.path.exists("./out2.h5"))

    # assert dsets are properly interleaved
    nt.assert_true(np.isclose((np.unique(ds.dsets[0].time_array) - np.unique(ds.dsets[1].time_array))[0],
                              -np.diff(np.unique(uvd.time_array))[0]))
    # assert first integration flagged across entire spw
    nt.assert_true(ds.dsets[0].get_flags(37, 38)[0, 0:25].all())
    # assert first integration flagged *ONLY* across spw
    nt.assert_false(ds.dsets[0].get_flags(37, 38)[0, :0].any() + ds.dsets[0].get_flags(37, 38)[0, 25:].any())
    # assert channel 15 flagged for all ints
    nt.assert_true(ds.dsets[0].get_flags(37, 38)[:, 15].all())
    # assert phase errors decreased after re-phasing
    phserr_before = np.mean(np.abs(np.angle(uvd1.data_array / uvd2.data_array)))
    phserr_after = np.mean(np.abs(np.angle(ds.dsets[0].data_array / ds.dsets[1].data_array)))
    nt.assert_true(phserr_after < phserr_before)

    # repeat feeding dsets_std and wgts
    if os.path.exists("./out2.h5"):
        os.remove("./out2.h5")
    ds = pspecdata.pspec_run([copy.deepcopy(uvd)], "./out2.h5", dsets_std=[copy.deepcopy(uvd)],
                             blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))], interleave_times=True,
                             verbose=False, overwrite=True, spw_ranges=[(0, 25)], rephase_to_dset=0,
                             broadcast_dset_flags=True, time_thresh=0.3)
    # assert ds passes validation
    psc =  container.PSpecContainer('./out2.h5')
    assert ds.dsets_std[0] is not None
    ds.validate_datasets()
    assert os.path.exists("./out2.h5")
    os.remove("./out2.h5")

    # test lst trimming
    if os.path.exists("./out2.h5"):
        os.remove("./out2.h5")
    uvd1 = copy.deepcopy(uvd)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[2:], inplace=False)
    ds = pspecdata.pspec_run([copy.deepcopy(uvd1), copy.deepcopy(uvd2)], "./out2.h5",
                             blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
                             verbose=False, overwrite=True, spw_ranges=[(50, 100)],
                             trim_dset_lsts=True)
    # assert first uvd1 lst_array got trimmed by 2 integrations
    psc =  container.PSpecContainer('./out2.h5')
    nt.assert_equal(ds.dsets[0].Ntimes, 8)
    nt.assert_true(np.isclose(np.unique(ds.dsets[0].lst_array), np.unique(uvd2.lst_array)).all())

    if os.path.exists("./out2.h5"):
        os.remove("./out2.h5")

    # test when no data is loaded in dset
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    ds = pspecdata.pspec_run(fnames, "./out.h5", Jy2mK=False, verbose=False, overwrite=True,
                             blpairs=[((500, 501), (600, 601))])  # blpairs that don't exist
    nt.assert_equal(ds, None)
    nt.assert_false(os.path.exists("./out.h5"))

    # same test but with pre-loaded UVDatas
    uvds = []
    for f in fnames:
        uvd = UVData()
        uvd.read_miriad(f)
        uvds.append(uvd)
    ds = pspecdata.pspec_run(uvds, "./out.h5", dsets_std=fnames_std, Jy2mK=False, verbose=False, overwrite=True,
                             blpairs=[((500, 501), (600, 601))])
    nt.assert_equal(ds, None)
    nt.assert_false(os.path.exists("./out.h5"))

    # test when data is loaded, but no blpairs match
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    ds = pspecdata.pspec_run(fnames, "./out.h5", Jy2mK=False, verbose=False, overwrite=True,
                             blpairs=[((37, 38), (600, 601))])
    nt.assert_true(isinstance(ds, pspecdata.PSpecData))
    nt.assert_false(os.path.exists("./out.h5"))

    # test glob-parseable input dataset
    dsets = [os.path.join(DATA_PATH, "zen.2458042.?????.xx.HH.uvXA"),
             os.path.join(DATA_PATH, "zen.2458042.?????.xx.HH.uvXA")]
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    ds = pspecdata.pspec_run(dsets, "./out.h5", Jy2mK=False, verbose=True, overwrite=True,
                             blpairs=[((24, 25), (37, 38))])
    psc = container.PSpecContainer('./out.h5', 'rw')
    uvp = psc.get_pspec("dset0_dset1", "dset0_x_dset1")
    nt.assert_equal(uvp.Ntimes, 120)
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")

    # test input calibration
    dfile = os.path.join(DATA_PATH, "zen.2458116.30448.HH.uvh5")
    cfile = os.path.join(DATA_PATH, "zen.2458116.30448.HH.flagged_abs.calfits")
    ds = pspecdata.pspec_run([dfile, dfile], "./out.h5", cals=cfile, dsets_std=[dfile, dfile],
                             verbose=False, overwrite=True, blpairs=[((23, 24), (24, 25))],
                             pol_pairs=[('xx', 'xx')], interleave_times=False,
                             file_type='uvh5', spw_ranges=[(100, 150)], cal_flag=True)
    psc = container.PSpecContainer('./out.h5', 'rw')
    uvp = psc.get_pspec('dset0_dset1', 'dset0_x_dset1')
    # test calibration flags were propagated to test that cal was applied
    assert ds.dsets[0].flag_array.any()
    assert ds.dsets[1].flag_array.any()
    assert ds.dsets_std[0].flag_array.any()
    assert ds.dsets_std[1].flag_array.any()
    assert ds.dsets[0].extra_keywords['filename'] is not '""'
    assert ds.dsets[0].extra_keywords['calibration'] is not '""'
    assert 'cal: /' in uvp.history

    # test exceptions
    nt.assert_raises(AssertionError, pspecdata.pspec_run, 'foo', "./out.h5")
    nt.assert_raises(AssertionError, pspecdata.pspec_run, fnames, "./out.h5", blpairs=(1, 2), verbose=False)
    nt.assert_raises(AssertionError, pspecdata.pspec_run, fnames, "./out.h5", blpairs=[1, 2], verbose=False)
    nt.assert_raises(AssertionError, pspecdata.pspec_run, fnames, "./out.h5", beam=1, verbose=False)

    # test execution with list of files for each dataset and list of cals
    if os.path.exists("./out.h5"):
        os.remove("./out.h5")
    fnames = glob.glob(os.path.join(DATA_PATH, "zen.2458116.*.HH.uvh5"))
    cals = glob.glob(os.path.join(DATA_PATH, "zen.2458116.*.HH.flagged_abs.calfits"))
    ds = pspecdata.pspec_run([fnames, fnames], "./out.h5", Jy2mK=False,
                             verbose=False, overwrite=True, file_type='uvh5',
                             bl_len_range=(14, 15), bl_deg_range=(0, 1),
                             psname_ext='_0', spw_ranges=[(0, 25)], cals=[cals, cals])
    psc = container.PSpecContainer('./out.h5', 'rw')
    nt.assert_true(isinstance(psc, container.PSpecContainer))
    nt.assert_equal(psc.groups(), ['dset0_dset1'])
    nt.assert_equal(psc.spectra(psc.groups()[0]), ['dset0_x_dset1_0'])
    nt.assert_true(os.path.exists("./out.h5"))

    if os.path.exists("./out.h5"):
        os.remove("./out.h5")


def test_input_calibration():
    dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458116.30*.HH.uvh5")))
    cfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458116.30*.HH.flagged_abs.calfits")))
    for i, f in enumerate(zip(dfiles, cfiles)):
        uvd = UVData()
        uvd.read(f[0])
        dfiles[i] = uvd
        uvc = UVCal()
        uvc.read_calfits(f[1])
        cfiles[i] = uvc

    # test add
    pd = pspecdata.PSpecData()
    pd.add(dfiles, None)  # w/o cal
    pd.add([copy.deepcopy(uv) for uv in dfiles], None, cals=cfiles, cal_flag=False)  # with cal
    g = (cfiles[0].get_gains(23, 'x') * np.conj(cfiles[0].get_gains(24, 'x'))).T
    np.testing.assert_array_almost_equal(pd.dsets[0].get_data(23, 24, 'xx') / g,
                                         pd.dsets[1].get_data(23, 24, 'xx'))

    # test add with dictionaries
    pd.add({'one': copy.deepcopy(dfiles[0])}, {'one': None}, cals={'one':cfiles[0]}, cal_flag=False)
    np.testing.assert_array_almost_equal(pd.dsets[0].get_data(23, 24, 'xx') / g,
                                         pd.dsets[2].get_data(23, 24, 'xx'))

    # test dset_std calibration
    pd.add([copy.deepcopy(uv) for uv in dfiles], None, dsets_std=[copy.deepcopy(uv) for uv in dfiles],
           cals=cfiles, cal_flag=False)
    np.testing.assert_array_almost_equal(pd.dsets[0].get_data(23, 24, 'xx') / g,
                                         pd.dsets_std[3].get_data(23, 24, 'xx'))

    # test exceptions
    pd = pspecdata.PSpecData()
    nt.assert_raises(TypeError, pd.add, {'one': copy.deepcopy(dfiles[0])}, {'one': None},
                     cals='foo', cal_flag=False)
    nt.assert_raises(AssertionError, pd.add, dfiles, [None], cals=[None, None])
    nt.assert_raises(TypeError, pd.add, dfiles, [None], cals=['foo'])


def test_get_argparser():
    args = pspecdata.get_pspec_run_argparser()
    a = args.parse_args([['foo'], 'bar', '--dset_pairs', '0 0, 1 1', '--pol_pairs', 'xx xx, yy yy',
                         '--spw_ranges', '300 400, 600 800', '--blpairs', '24 25 24 25, 37 38 37 38'])
    nt.assert_equal(a.pol_pairs, [('xx', 'xx'), ('yy', 'yy')])
    nt.assert_equal(a.dset_pairs, [(0, 0), (1, 1)])
    nt.assert_equal(a.spw_ranges, [(300, 400), (600, 800)])
    nt.assert_equal(a.blpairs, [((24, 25), (24, 25)), ((37, 38), (37, 38))])

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

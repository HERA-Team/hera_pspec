import unittest
import nose.tools as nt
import numpy as np
import pyuvdata as uv
import os
import copy
import sys
from hera_pspec import pspecdata
from hera_pspec import oqe
from hera_pspec.data import DATA_PATH
import pyuvdata as uv
import pylab as plt
from hera_cal import redcal

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
        # setup empty PSpecData
        self.ds = pspecdata.PSpecData()

        # read in miriad file
        self.uvd1 = uv.UVData()
        self.uvd1.read_miriad(os.path.join(DATA_PATH, 'zen.2458042.12552.xx.HH.uvXAA'))

        # read in another miriad file
        self.uvd2 = uv.UVData()
        self.uvd2.read_miriad(os.path.join(DATA_PATH, 'zen.2458042.19263.xx.HH.uvXA'))
        self.antpos, self.ants = self.uvd2.get_ENU_antpos(pick_data_ants=True)
        self.antpos -= np.median(self.antpos, axis=0)
        self.antpos_dict = dict(zip(self.ants, self.antpos))
        red_bls = redcal.get_reds(self.antpos_dict, low_hi=True)
        for i, blg in enumerate(red_bls):
            for j, bl in enumerate(blg):
                red_bls[i][j] = red_bls[i][j][:2]
        self.red_bls = red_bls

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):

        # Test creating empty DataSet
        ds = pspecdata.PSpecData()
        #self.assertAlmostEqual(C.H0, 25.5)
        pass

    def test_add_data(self):
        pass

    def test_validate_datasets(self):
        # test freq exception
        uvd2 = self.uvd1.select(frequencies=np.unique(self.uvd1.freq_array)[:10], inplace=False)
        ds = pspecdata.PSpecData(dsets=[self.uvd1, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)
        # test time exception
        uvd2 = self.uvd1.select(times=np.unique(self.uvd1.time_array)[:10], inplace=False)
        ds = pspecdata.PSpecData(dsets=[self.uvd1, uvd2], wgts=[None, None])
        nt.assert_raises(ValueError, ds.validate_datasets)


    def test_pspec(self):
        # generate two uvd objects w/ adjacent integrations stripped out
        uvd1 = self.uvd2.select(times=np.unique(self.uvd2.time_array)[0::2], inplace=False)
        uvd2 = self.uvd2.select(times=np.unique(self.uvd2.time_array)[1::2], inplace=False)

        # test basic setup
        ds = pspecdata.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None])
        nt.assert_equal(uvd1, ds.dsets[0])
        nt.assert_equal(uvd2, ds.dsets[1])

        # basic execution
        pspecs, pairs = ds.pspec(self.red_bls[0], input_data_weight='identity', norm='I', 
                                 rev_bl_pair=False, verbose=False)
        # assert shapes
        nt.assert_equal(pspecs.shape, (4, 64, 30))
        nt.assert_equal(len(pairs), 4)
        # assert auto-baseline pspec
        nt.assert_equal(pairs[0][0][1:], pairs[0][1][1:])

        # execution w/ red_bl_group
        pspecs, pairs = ds.pspec(self.red_bls, input_data_weight='identity', norm='I', 
                                 rev_bl_pair=True, verbose=False)
        # assert shapes
        nt.assert_equal(pspecs.shape, (63, 64, 30))


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
                        atol=tolerance))
        self.assertTrue(np.allclose(f_mat_true,
                                    np.identity(data.Nfreqs).astype(complex),
                                    rtol=tolerance,
                                    atol=tolerance))
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




if __name__ == "__main__":
    unittest.main()

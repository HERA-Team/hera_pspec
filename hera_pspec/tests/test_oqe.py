import unittest
import nose.tools as nt
import numpy as np, oqe


class Test_OQE(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


    
    def test_noise(self):
        # Test that the noise amplitude is generated properly.
        # 0) !!I'm not sure we want this function in oqe.py 
        # 1) Real and imaginary part have the same rms (or close to)
        # 2) The sqrt(2) is correctly applied.
        vect_length = 5000000 # Need a large number to not get unlucky
        noise_vect = oqe.noise(vect_length)
        self.assertEqual(noise_vect.shape[0], vect_length) # correct shape
        real_part, imag_part = np.real(noise_vect), np.imag(noise_vect)
        real_ms, imag_ms = np.mean(real_part**2), np.mean(imag_part**2)
        self.assertAlmostEqual(real_ms, imag_ms, places=2) # Real and imaginary have same rms
        self.assertAlmostEqual(real_ms + imag_ms, 1., places=2) # Sqrt(2) is correctly applied

    def test_get_Q(self):
        vect_length = 50
        x_vect = np.random.normal(size=vect_length) + 1j * np.random.normal(size=vect_length)
        y_vect = np.random.normal(size=vect_length) + 1j * np.random.normal(size=vect_length)

        for i in range(vect_length):
            Q_matrix = oqe.get_Q(i, vect_length)
            xQy = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,y_vect))
            yQx = np.dot(np.conjugate(y_vect),np.dot(Q_matrix,x_vect))
            xQx = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,x_vect))
            self.assertEqual(Q_matrix.shape,(vect_length,vect_length)) # Q matrix right shape
            self.assertAlmostEqual(xQy,np.conjugate(yQx)) # x^t Q y == conj(y^t Q x)
            self.assertAlmostEqual(np.imag(xQx), 0.) # x^t Q x should be real

        x_vect = np.ones(vect_length)
        Q_matrix = oqe.get_Q(0, vect_length)
        xQx = np.dot(np.conjugate(x_vect),np.dot(Q_matrix,x_vect))
        self.assertAlmostEqual(xQx,1.)
        # 3) Sending in pure tones/sinusoids for x and y should give delta functions

    def test_cov(self):
        # Test for code that takes vectors and generates covariance matrices
        # 1) cov(d1,w,d1,w) should give the same as cov(d1,w)
        # 2  Test with non unity weights. 
        # 3) Diagonal elements should be real and positive
        # 4) Example covariance that should give the same
        # 5) Overall scaling of weights should not affect the final covariance
        # 6) Test that matrices of the right size are outputted
        # 7) Error raised if complex or negative weights are inputted
        n_times = 83
        n_freqs = 27
        d1_vect = np.random.normal(size=(n_freqs,n_times)) + 1j * np.random.normal(size=(n_freqs,n_times))
        w1_vect = np.random.uniform(size=(n_freqs,n_times))
        #print d1_vect.shape, w1_vect.shape
        auto_cov_1 = oqe.cov(d1_vect,w1_vect)
        auto_cov_2 = oqe.cov(d1_vect,w1_vect,d1_vect,w1_vect)
        d2_vect = np.random.normal(size=(n_freqs,n_times)) + 1j * np.random.normal(size=(n_freqs,n_times))
        w2_vect = np.random.uniform(size=(n_freqs,n_times))
        cross_cov = oqe.cov(d1_vect,w1_vect,d2_vect,w2_vect)
        cross_cov_scale_weights = oqe.cov(d1_vect,6.4*w1_vect,d2_vect,8.2*w2_vect)
        self.assertEqual(cross_cov.shape,(n_freqs,n_freqs)) # cross covariance matrix has right shape
        self.assertEqual(auto_cov_1.shape,(n_freqs,n_freqs))# auto covariance matrix has right shape
        self.assertEqual(auto_cov_2.shape,(n_freqs,n_freqs))# auto covariance matrix has right shape
        self.assertAlmostEqual(np.imag(np.diag(auto_cov_1)).sum(), 0., places=10) # diagonal real

        for i in range(n_freqs):
            self.assertGreaterEqual(auto_cov_1[i,i], 0.) # diagonal positive
            for j in range(n_freqs):
                self.assertEqual(auto_cov_1[i,j],auto_cov_2[i,j]) #cov(d1,w,d1,w) == cov(d1,w)
                self.assertAlmostEqual(cross_cov[i,j],cross_cov_scale_weights[i,j])
  
        nt.assert_raises(TypeError, oqe.cov, d1_vect,d1_vect) # weights cannot be complex
        nt.assert_raises(TypeError, oqe.cov, d1_vect,w1_vect,d2_vect,d2_vect) # weights cannot be complex
        nt.assert_raises(ValueError, oqe.cov, d1_vect,-1.*w1_vect) # weights cannot be negative
        nt.assert_raises(ValueError, oqe.cov, d1_vect,w1_vect,d2_vect,-1.*w2_vect) # weights cannot be negative
        
        n_times = 3
        n_freqs = 2
        d1_vect = np.array([[ 1.67441499-0.27071976j,0.14381232+1.6668588j,1.84615382-0.27232576j],\
                            [ 0.99703942+0.10531244j,-0.10412385-0.02644308j,-1.32641920+0.2714418j ]])
        d2_vect = np.array([[-0.53269830-0.71510325j,  1.01470283+1.73205161j,0.14592263-0.62538893j],\
                            [-3.07935528+0.80395593j, -1.04302723+1.29690401j,0.27884803-0.49442625j]])
        w1_vect = np.array([[ 0.51258983,  0.347186  ,  0.79923525],[ 0.50345938,  0.91413427,  0.55338137]])
        w2_vect = np.array([[ 0.7318251 ,  0.09832265,  0.49238467],[ 0.20412305,  0.39804246,  0.09623617]])
        cross_cov_ans = np.array([[ 0.32223255+0.5387318j,  1.23792556+0.69841395j],\
                                  [-0.44181934+0.15350595j, -0.91383982-0.18630296j]])
        auto_cov_ans = np.array([[ 1.16374256+0.j, -0.47224677-0.07411756j],\
                                 [-0.47224677+0.07411756j,  0.55598680+0.j]])
        cross_cov = oqe.cov(d1_vect,w1_vect,d2_vect,w2_vect)
        auto_cov = oqe.cov(d1_vect,w1_vect)
        for i in range(n_freqs):
            for j in range(n_freqs):
                self.assertAlmostEqual(np.real(cross_cov[i,j]),np.real(cross_cov_ans[i,j]))
                self.assertAlmostEqual(np.imag(cross_cov[i,j]),np.imag(cross_cov_ans[i,j]))
                self.assertAlmostEqual(np.real(auto_cov[i,j]),np.real(auto_cov_ans[i,j]))
                self.assertAlmostEqual(np.imag(auto_cov[i,j]),np.imag(auto_cov_ans[i,j]))



    def test_get_Q(self):
        #1) Compare input tones against analytic sinusoids with
        #2) both a blackmanharris and unity window function
        #3) Should we get rid of the DELAY argument? 

        
    def test_lst_grid(self):
        #0) !!!Im not sure we want this function in oqe.py
        #1) Test gridding of a single baseline at the equator.
        #2) Delay-rate transform of this baseline should be a gaussian
        #3) Unity weights should give same answer as no weights.
        #4) throw error if weights has different dimensions as data. 
        

    def test_lst_grid_cheap(self):
        #0) !!!I'm not sure if we want this function in oqe.py
        #1) same tests for test_lst_grid. Test equivalence of lst_grid_cheap and lst_grid

    def test_lst_align(self):
        #0) Document. Does this belong in oqe?
        #1) test on four lst lists which are overlapping, offset by less then resolution,
        #2) and one is larger then the other. Should give indices of intersection of lst values
        #3) (within lst resolution).


    def test_lst_align_data(self):
        #0) Document. Does this belong in oqe.py?
        #1) Should give same answer with unity weights.
        #2) 
        
    def test_boot_waterfall(self):
        #0) Document. Does this belong in oqe?
    
        

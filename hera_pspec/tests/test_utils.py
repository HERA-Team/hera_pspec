import unittest
import nose.tools as nt
import numpy as np, oqe

class Test_OQE(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_noise(self):
        vect_length = 5000000 # Need a large number to not get unlucky
        noise_vect = oqe.noise(vect_length)
        self.assertEqual(noise_vect.shape[0], vect_length) # correct shape
        real_part, imag_part = np.real(noise_vect), np.imag(noise_vect)
        real_ms, imag_ms = np.mean(real_part**2), np.mean(imag_part**2)
        self.assertAlmostEqual(real_ms, imag_ms, places=2) # Real and imaginary have same rms
        self.assertAlmostEqual(real_ms + imag_ms, 1., places=2) # Sqrt(2) is correctly applied

    def test_cov(self):
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



import unittest, random, glob, os
import capo
import aipy as a, numpy as np
from mpl_toolkits.basemap import Basemap
import capo.oqe as oqe

DATADIR = '/Users/aparsons/projects/eor/psa6942/omnical_v2_xtalk/lstbin000'
TESTFILE = 'test.npz'
CAL = 'psa6622_v003'
POL = 'I'
random.seed(0)
args = glob.glob(DATADIR+'/even/lst*uvGAL') + glob.glob(DATADIR+'/odd/lst*uvGAL')

class TestMethods(unittest.TestCase):
    def test_cov(self):
        NSAMP = 1024
        factor = NSAMP/float(NSAMP-1)
        fq = np.exp(1j*np.linspace(0,2*np.pi,128)); fq.shape = (-1,1)
        ts = np.exp(1j*np.linspace(0,2*np.pi,NSAMP+1)[:-1]); ts.shape = (1,-1)
        x = fq * ts
        w = np.ones_like(x)
        C = oqe.cov(x,w)
        np.testing.assert_allclose(np.diag(C), factor)
        np.testing.assert_allclose(C[0].conj(), factor*fq.flatten())
        x = np.concatenate([x,x], axis=1)
        w = np.concatenate([w,np.zeros_like(w)], axis=1)
        C = oqe.cov(x,w)
        np.testing.assert_allclose(np.diag(C), factor)
        np.testing.assert_allclose(C[0].conj(), factor*fq.flatten())
    def test_noise(self):
        n = oqe.noise((1024,1024))
        self.assertEqual(n.shape, (1024,1024))
        self.assertAlmostEqual(np.var(n), 1, 2)
    def test_lst_grid(self):
        SZ = 300
        #lsts = np.linspace(np.pi/2, 3*np.pi/2, SZ)
        lsts = np.linspace(0, 2*np.pi, SZ)
        x,y = np.indices((SZ, SZ/3))
        t = 2*np.pi*x/SZ
        data = np.exp(1j*y*t)
        wgts = np.ones(data.shape, dtype=np.float)
        lst_g, data_g, wgt_g = oqe.lst_grid(lsts, data, wgts, lstbins=400)
        x,y = np.indices(data_g.shape)
        ans = np.exp(2j*np.pi*x/data_g.shape[0]*y)
        from scipy.interpolate import interp1d
        f = interp1d(t[:,0], data, kind=1, axis=0, fill_value=0, bounds_error=False, assume_sorted=True)
        data_g2 = f(lst_g)
        
        #np.testing.assert_almost_equal(data_g, ans, 3)
        import pylab as plt
        plt.subplot(141); capo.plot.waterfall(data, mode='real', mx=1, drng=2)
        plt.subplot(142); capo.plot.waterfall(data_g, mode='real', mx=1, drng=2)
        plt.subplot(143); capo.plot.waterfall(data_g2, mode='real', mx=1, drng=2)
        plt.subplot(144); capo.plot.waterfall(data_g2-ans, mode='real', mx=1, drng=2)
        plt.show()

class TestDataSet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        NFREQ = 32
        NSAMP = 1024
        fq = np.exp(1j*np.linspace(0,2*np.pi,NFREQ)); fq.shape = (1,-1)
        ts = np.exp(1j*np.linspace(0,2*np.pi,NSAMP+1)[:-1]); ts.shape = (-1,1)
        self.sky = fq * ts
        self.eor = oqe.noise(size=(NSAMP,NFREQ))
        sets = ['even','odd']
        bls = [(0,1),(1,2),(2,3),(3,4),(4,5)]
        pols = 'I'
        self.data = {}
        for s in sets:
            self.data[s] = {}
            for bl in bls:
                self.data[s][bl] = {}
                for pol in pols:
                    self.data[s][bl][pol] = self.sky + self.eor + oqe.noise((NSAMP,NFREQ))
    def test_init(self):
        ds1 = capo.oqe.DataSet(self.data)
        for k in self.data:
            for bl in self.data[k]:
                for pol in self.data[k][bl]:
                    self.assertTrue(ds1.x.has_key((k,bl,pol)))
        ds2 = capo.oqe.DataSet()
        dflat = ds2.flatten_data(self.data)
        ds2 = capo.oqe.DataSet(dflat)
        for k in dflat:
            self.assertTrue(ds2.x.has_key(k))
        ds2 = capo.oqe.DataSet(dflat, conj={(0,1):True})
        k1 = ('even',(0,1),'I')
        k2 = ('odd',(0,1),'I')
        k3 = ('even',(1,2),'I')
        np.testing.assert_equal(ds1.x[k1], ds2.x[k1].conj())
        np.testing.assert_equal(ds1.x[k2], ds2.x[k2].conj())
        np.testing.assert_equal(ds1.x[k3], ds2.x[k3])
    #def test_to_from_npz(self):
    #    ds1 = capo.oqe.DataSet(self.data,self.wgts,self.lsts,self.conj)
    #    ds1.to_npz(TESTFILE)
    #    ds2 = capo.oqe.DataSet(npzfile=TESTFILE)
    #    for k in ds1.x:
    #        self.assertTrue(ds2.x.has_key(k))
    def test_lst_align(self):
        ds = capo.oqe.DataSet()
        lst1,lst2 = np.arange(0,544), np.arange(5,600)
        i1,i2 = ds.lst_align(lst1, lst2)
        np.testing.assert_array_equal(lst1[i1], lst2[i2])
    def test_boots(self):
        ds = capo.oqe.DataSet(self.data)
        boots = [i for i in ds.gen_bl_boots(20, ngps=5)]
        self.assertEqual(len(boots), 20)
        self.assertEqual(len(boots[0]), 5)
        self.assertNotEqual(boots[0], boots[1])
        for boot in boots:
            for i, gp1 in enumerate(boot):
                for gp2 in boot[i+1:]:
                    for bl in gp1:
                        self.assertFalse(bl in gp2)
    def test_q_fft(self):
        k1,k2 = ('a',(0,1),'I'), ('b',(0,1),'I')
        ds = oqe.DataSet({k1:self.eor, k2:self.eor})
        qnofft = ds.q_hat(k1,k2,use_cov=False,use_fft=False)
        qfft = ds.q_hat(k1,k2,use_cov=False,use_fft=True)
        np.testing.assert_array_almost_equal(qnofft,qfft)
    def test_q_eor_nocov(self):
        k1,k2 = ('a',(0,1),'I'), ('b',(0,1),'I')
        ds = oqe.DataSet({k1:self.eor, k2:self.eor})
        q = ds.q_hat(k1,k2,use_cov=False)
        self.assertTrue(np.all(q > 0))
        self.assertAlmostEqual(np.average(q).real, q.shape[0], 0)
        n1,n2 = oqe.noise(self.eor.shape), oqe.noise(self.eor.shape)
        ds = oqe.DataSet({k1:self.eor+n1, k2:self.eor+n2})
        qn = ds.q_hat(k1,k2,use_cov=False)
        self.assertFalse(np.all(qn > 0))
        self.assertAlmostEqual(np.average(qn).real, qn.shape[0], 0)
        self.assertAlmostEqual(np.average(qn).real, np.average(q).real, 0)
        ds = oqe.DataSet({k1:n1, k2:n2})
        qn = ds.q_hat(k1,k2,use_cov=False)
        self.assertFalse(np.all(qn > 0))
        self.assertAlmostEqual(np.average(qn).real, 0, 0)
    def test_q_eor_cov(self):
        k1,k2 = ('a',(0,1),'I'), ('b',(0,1),'I')
        ds = oqe.DataSet({k1:self.eor, k2:self.eor})
        C1,C2 = ds.C(k1), ds.C(k2)
        I = np.identity(C1.shape[0])
        np.testing.assert_array_almost_equal(C1, I, 1)
        np.testing.assert_array_equal(C1, C2)
        qI = ds.q_hat(k1,k2,use_cov=False)
        qC = ds.q_hat(k1,k2,use_cov=True)
        self.assertTrue(np.all(qC > 0))
        ds.set_C({k1:I,k2:I})
        qCI = ds.q_hat(k1,k2,use_cov=True)
        np.testing.assert_array_equal(qCI, qI)
        self.assertAlmostEqual(np.average(qC), np.average(qI), -1)
    def tearDown(self):
        if os.path.exists(TESTFILE): os.remove(TESTFILE)
        

if __name__ == '__main__':
    unittest.main()

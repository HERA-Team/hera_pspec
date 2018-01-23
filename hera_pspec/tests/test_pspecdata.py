import unittest
import nose.tools as nt
import numpy as np
import os
import copy
import sys
from hera_pspec import pspecdata
from hera_pspec import oqe
from hera_pspec.data import DATA_PATH
import pyuvdata as uv
import pylab as plt


class Test_DataSet(unittest.TestCase):

    def setUp(self):
        self.ds = pspecdata.PSpecData()
        pass

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

    def test_get_F(self):
        NDATA=100#must average many data sets for convergence. 
        dpath=os.path.join(DATA_PATH,'zen.2458042.12552.xx.HH.uvXAA')
        print dpath
        data=uv.UVData()
        wghts=uv.UVData()
        data.read_miriad(dpath)
        wghts.read_miriad(dpath)
        #only use 8 channels
        data.select(freq_chans=range(8))
        wghts.select(freq_chans=range(8))
        #add times


        #set data to random white noise with a random variance and mean
        test_mean=0.#*np.abs(np.random.randn())
        test_std=np.abs(np.random.randn())
        data.flag_array[:]=False#Make sure that all of the flags are set too true for analytic case.
        wghts.data_array[:]=1.
        wghts.flag_array[:]=False
        #pspec=pspecdata.PSpecData()
        #pspec.add([data],
        #          [wghts])#create pspec data set from data.
      
        #for data in pspec.dsets:
        bllist=data.get_antpairs()
        f_mat=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        f_mat_true=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        nsamples=0
        icvals=[]
        icvals_true=[]
        cvals=[]
        cvals_true=[]
        for datanum in range(NDATA):
            for j in range(1,data.Nbls):
                #get baseline index
                pair1=bllist[j]
                for k in range(j):
                    pspec=pspecdata.PSpecData()
                    data.data_array=test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)\
                    +1j*test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)
                    pspec.add([data],[wghts])
                    pair2=bllist[k]
                    k1=(0,pair1[0],pair1[1],-5)
                    k2=(0,pair2[0],pair2[1],-5)
                    icvals=icvals+list(pspec.iC(k1).diagonal()*test_std**2.)
                    icvals_true=icvals_true+list(pspec.iC(k2).diagonal()*test_std**2.)
                    cvals=cvals+list(pspec.C(k1).diagonal()/test_std**2.)
                    cvals_true=cvals_true+list(pspec.C(k2).diagonal()/test_std**2.)
                    f_mat_true=f_mat_true+pspec.get_F(k1,k2,true_fisher=True)
                    f_mat=f_mat+pspec.get_F(k1,k2)
                    nsamples=nsamples+1
                    #test identity
                    self.assertTrue(np.allclose(pspec.get_F(k1,k2,use_identity=True)/data.Nfreqs**2.,
                                    np.identity(data.Nfreqs).astype(complex)))
                    del pspec

        f_mat=f_mat/nsamples/data.Nfreqs**2.*test_std**4.
        f_mat_true=f_mat_true/nsamples/data.Nfreqs**2.*test_std**4.
        '''
        #make diagnostic histograms for testing purposes
        print f_mat.diagonal()
        print f_mat_true.diagonal()
        plt.pcolor(f_mat.astype(float))
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.hist(np.abs(np.array(icvals))**2.,100)
        plt.show()
        plt.figure()
        plt.hist(1./np.abs(np.array(cvals))**2.,100)
        plt.show()
        '''
        self.assertTrue(np.allclose(f_mat,
                        np.identity(pspec.dsets[0].Nfreqs).astype(complex),
                        rtol=2./pspec.dsets[0].Ntimes,
                        atol=test_std*2./pspec.dsets[0].Ntimes))
        #test for true fisher
        self.assertTrue(np.allclose(f_mat_true,
                                    np.identity(pspec.dsets[0].Nfreqs).astype(complex),
                                    rtol=2./pspec.dsets[0].Ntimes,
                                    atol=test_std*2./pspec.dsets[0].Ntimes))
            
                                                
        #TODO: Need a test case for some kind of taper.
            

        

if __name__ == "__main__":
    unittest.main()

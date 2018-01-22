import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec import pspecdata
from hera_pspec import oqe
from hera_pspec.data import DATA_PATH
import pyuvdata as uv


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
        dfile=DATA_PATH+'zen.2458042.12552.xx.HH.uvXAA'
        data=uv.UVData()
        data.read_miriad(dfile)
        #set data to random white noise with a random variance and mean
        test_mean=np.abs(np.random.randn())
        test_var=np.abs(np.random.randn())**2.
        test_noise=np.random.randn(test_mean,test_var/2.,size=data.data_array.shape)\
                  +1j*np.random.randn(test_mean,test_var/2.,size=data.data_array.shape)
        data.data_array=test_noise#set data equal to test noise.
        data.flag_array[:]=True#Make sure that all of the flags are set too true for analytic case. 
        pspec=PSpecData()
        pspec.add(data)#create pspec data set from data.
        for j in range(1,len(pspec.dsets[0].Nbls)):
            for k in range(j):
                #test equality
                self.assertTrue(np.allclose(pspec.get_F((i,j),(i,k)),
                                            np.identity(pspec.dsets[0].Nfreqs)\
                                            *pspec.dsets[0]**2./(2.*test_var),
                                            rtol=2./pspec.dsets[0].Ntimes,
                                            atol=test_var*2./pspec.dsets[0].Ntimes))
                #test for identity
                self.assertTrue(np.allclose(pspec.get_F((i,j),(i,k),use_identity=True),
                                            np.identity(pspec.dsets[0].Nfreqs)\
                                            *pspec.dsets[0].Nfreqs**2./2.))
                self.assertTrue(np.allclose(pspec.get_F((i,j),(i,k),use_identity=True),
                                            np.identity(pspec.dsets[0].Nfreqs)))
                #test for true fisher
                self.assertTrue(np.allclose(pspec.get_F((i,j),(i,k),true_fisher=True),
                                            np.identity(pspec.dsets[0].Nfreqs)\
                                            *pspec.dsets[0].Nfreqs**2./(2.*test_var),
                                            rtol=2./pspec.dsets[0].Ntimes,
                                            atol=test_var*2./pspec.dsets[0].Ntimes))
                
                                                
        #TODO: Need a test case for some kind of taper.
            

        

if __name__ == "__main__":
    unittest.main()

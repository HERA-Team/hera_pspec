import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import conversions

class Test_OQE(unittest.OQE):
    #Should test DataSets should be a property of Test_OQE or should we have separate tests for wherever
    #we decide to put DataSet.py


    
    def test_noise(self):
        # Test that the noise amplitude is generated properly.
        # 0) !!I'm not sure we want this function in oqe.py 
        # 1) Real and imaginary part have the same rms (or close to)
        # 2) The sqrt(2) is correctly applied.

    def test_get_Q(self):
        # Test that the delay Fourier transform matrix is correct
        # 1) x^t Q y should be the same as y^t Q x but complex conjugated
        # 2) x^t Q x should be real
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

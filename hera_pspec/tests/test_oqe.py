import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import conversions

class Test_OQE(unittest.OQE):

    def test_noise(self):
        # Test that the noise amplitude is generated properly.
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
        # 2) Diagonal elements should be real and positive
        # 3) Example covariance that should give the same
        # 4) Overall scaling of weights should not affect the final covariance
        # 5) Test that matrices of the right size are outputted
        # 6) Error raised if complex or negative weights are inputted
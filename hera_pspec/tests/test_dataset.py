import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import conversions

class Test_dataset(unittest.dataset):

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_flatten_data(self):
        # ---Test to see if inputting "None" does indeed output "None" for the data
        # ---Check that the number of keys is equal to the total number of elements in the original array
        # ---Check that there is a key for each element of the original array

    def test_add_data(self):
        # 

    def test_q_hat(self):
        # First, an awfully confusing thing. The "k1" and "k2" labels...are they just generic labels
        # for things like times and different baselines? They're certainly not wavenumbers, I don't think.
        # ---Swapping k1 and k2 should give the complex conjugated result.
        # ---The "use_fft" method should basically give the same result as the other method.
        # ---If k1 == k2, the outputs should be real.
        # ---Size of output is appropriate for the extent of the delay axis.
        # Need to think of something that tests the scenario with C = I

    def test_get_F(self):
        # Don't quite understand the += self.iC(k1i) bits. Is this just an averaging over, say, time?
        # Need to "fix" the "if False" statements
        # --For white noise with covariance amplitude alpha and nchans, F should be nchans^2/(2*alpha^2)*IDENTITY

    def test_get_MW(self):
        # ---MF should be correctly normalized.
        # ---W should be equal to MF.
        # ---If F is singular, does the code handle it gracefully?
        # ---For F^-1 mode, MF should be identity.
        # ---For I mode, M should be diagonal.
        # ---For Cholesky mode, M should be upper (or lower?) triangular.
        # ---Tests for specific numerical examples.
        # ---W For white noise is the identity
        # ---For mode=F^-1, M for white noise with variance amplitude alpha should be 2*alpha^2/nchans^2*IDENTITY
        # ---For node=F^-1/2, M for white noise with variance amplitude alpha should be 2*alpha^2/nchans^2*IDENTITY

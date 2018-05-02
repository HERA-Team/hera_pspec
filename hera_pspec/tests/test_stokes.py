import unittest
import nose.tools as nt
import os, sys
import pyuvdata as uv
from hera_pspec.data import DATA_PATH
from hera_pspec import stokes 
import pyuvdata

# Data files to use in tests
dset1 = 'zen.all.xx.LST.1.06964.uvA' # xx miriad file
dset2 = 'zen.all.yy.LST.1.06964.uvA' # yy miriad file
# Loading pyuvdata objects
uvd1 = pyuvdata.UVData()
uvd1.read_miriad(os.path.join(DATA_PATH, dset1))
pol1 = uvd1.get_pols()[0]
uvd2 = pyuvdata.UVData()
uvd2.read_miriad(os.path.join(DATA_PATH, dset2))
pol2 = uvd2.get_pols()[0]

def test_combine_pol():
   stokes.combine_pol( uvd1, uvd2, pol1, pol2)   
   # check exceptions
   nt.assert_raises(TypeError, stokes.combine_pol, dset1, dset2, pol1, pol2 )
   nt.assert_raises(AssertionError, stokes.combine_pol, uvd1, uvd2, pol1, 1)
   
def test_construct_stokes():   
   # test to form I and Q from single polarized UVData objects
   uvdI = stokes.construct_stokes(dset1=uvd1, dset2=uvd2, stokes='I')
   uvdQ = stokes.construct_stokes(dset1=uvd1, dset2=uvd2, stokes='Q')

   # check exceptions
   nt.assert_raises(AssertionError, stokes.construct_stokes, uvd1, 1)   
   nt.assert_raises(AssertionError, stokes.construct_stokes, uvd1, uvd1)

   # combining two polarizations (dset1 and dset2) together
   uvd3 = uvd1 + uvd2

   # test to form I and Q from dual polarized UVData objects
   uvdI = stokes.construct_stokes(dset1=uvd3, dset2=uvd3, stokes='I')

if __name__ == "__main__":
    unittest.main()

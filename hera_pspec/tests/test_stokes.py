import unittest
import nose.tools as nt
import os, sys
import pyuvdata as uv
from hera_pspec.data import DATA_PATH
from hera_pspec import stokes 

# Data files to use in tests
dset1 = 'zen.all.xx.LST.1.06964.uvA' # xx miriad file
dset2 = 'zen.all.yy.LST.1.06964.uvA' # yy miriad file

def test_combine_pol():
   # check polarization inputs
   nt.assert_raises(TypeError, test_combine_pol ,dset1, dset2,1,'YY' )
   nt.assert_raises(TypeError, test_combine_pol ,dset1, dset2, 'XX',1 )

def test_construct_stokes():   
   # loading first UVData object
   uvd1 = uv.UVData()
   nt.assert_true(type(dset1), str)
   uvd1.read_miriad(os.path.join(DATA_PATH, dset1))
   pol1 = uvd1.get_pols()[0]

   # loading second UVData object
   uvd2 = uv.UVData()
   nt.assert_true(type(dset1), str)
   uvd2.read_miriad(os.path.join(DATA_PATH, dset2))
   pol2 = uvd2.get_pols()[0]

   # check inputs
   nt.assert_raises(TypeError, test_construct_stokes, 1, uvd2 )
   nt.assert_raises(TypeError, test_construct_stokes , dset1, uvd1 )

   # check polarizations
   nt.assert_equal(pol1,'XX')
   nt.assert_equal(pol2,'YY')

   print ('Combining XX and YY visibilities to form Stokes I visibilities')      
   uvdI = stokes.construct_stokes(dset1=uvd1,dset2=uvd2,stokes='I')

   print ('Combining XX and YY visibilities to form Stokes Q visibilities')
   uvdQ = stokes.construct_stokes(dset1=uvd1,dset2=uvd2,stokes='Q')

if __name__ == "__main__":
    unittest.main()

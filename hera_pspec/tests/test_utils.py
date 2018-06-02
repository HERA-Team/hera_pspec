import unittest
import nose.tools as nt
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from hera_pspec import utils, testing
from collections import OrderedDict as odict
from pyuvdata import UVData



def test_cov():
    # load another data file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"))

    # test basic execution
    d1 = uvd.get_data(24, 25)
    w1 = (~uvd.get_flags(24, 25)).astype(np.float)
    cov = utils.cov(d1, w1)
    nt.assert_equal(cov.shape, (60, 60))
    nt.assert_equal(cov.dtype, np.complex)
    d2 = uvd.get_data(37, 38)
    w2 = (~uvd.get_flags(37, 38)).astype(np.float)
    cov = utils.cov(d1, w2, d2=d2, w2=w2)
    nt.assert_equal(cov.shape, (60, 60))
    nt.assert_equal(cov.dtype, np.complex)
    # test exception
    nt.assert_raises(TypeError, utils.cov, d1, w1*1j)
    nt.assert_raises(TypeError, utils.cov, d1, w1, d2=d2, w2=w2*1j)
    w1 *= -1.0
    nt.assert_raises(ValueError, utils.cov, d1, w1)

def test_load_config():
    """
    Check YAML config file handling.
    """
    fname = os.path.join(DATA_PATH, '_test_utils.yaml')
    cfg = utils.load_config(fname)
    
    # Check that expected keys exist
    assert('data' in cfg.keys())
    assert('pspec' in cfg.keys())
    
    # Check that boolean values are read in correctly
    assert(cfg['pspec']['overwrite'] == True)
    
    # Check that lists are read in as lists
    assert(len(cfg['data']['subdirs']) == 1)
    
    # Check that missing files cause an error
    nt.assert_raises(IOError, utils.load_config, "file_that_doesnt_exist")


class Test_Utils(unittest.TestCase):

    def setUp(self):
        # Load data into UVData object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, 
                                          "zen.2458042.17772.xx.HH.uvXA"))
        
        # Create UVPSpec object
        self.uvp, cosmo = testing.build_vanilla_uvpspec()

    def tearDown(self):
        pass

    def runTest(self):
        pass
    
    def test_spw_range_from_freqs(self):
        """
        Test that spectral window ranges are correctly recovered from UVData and 
        UVPSpec files.
        """
        # Check that type errors and bounds errors are raised
        nt.assert_raises(AttributeError, utils.spw_range_from_freqs, np.arange(3), 
                         freq_range=(100e6, 110e6))
        for obj in [self.uvd, self.uvp]:
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(98e6, 110e6)) # lower bound
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(190e6, 202e6)) # upper bound
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(190e6, 180e6)) # wrong order
            
        # Check that valid frequency ranges are returned
        freq_list = [(100e6, 120e6), (120e6, 140e6), (140e6, 160e6)]
        spw1 = utils.spw_range_from_freqs(self.uvd, freq_range=(110e6, 130e6))
        spw2 = utils.spw_range_from_freqs(self.uvd, freq_range=freq_list)
        spw3 = utils.spw_range_from_freqs(self.uvd, freq_range=(98e6, 120e6), 
                                          bounds_error=False)
        spw4 = utils.spw_range_from_freqs(self.uvd, freq_range=(100e6, 120e6))
        
        # Make sure tuple vs. list arguments were handled correctly
        nt.ok_( isinstance(spw1, tuple) )
        nt.ok_( isinstance(spw2, list) )
        nt.ok_( len(spw2) == len(freq_list) )
        
        # Make sure that bounds_error=False works
        nt.ok_( spw3 == spw4 )
        
        # Make sure that this also works for UVPSpec objects
        spw5 = utils.spw_range_from_freqs(self.uvp, freq_range=(100e6, 104e6))
        nt.ok_( isinstance(spw5, tuple) )
        nt.ok_( spw5[0] is not None )
    
    def test_spw_range_from_redshifts(self):
        """
        Test that spectral window ranges are correctly recovered from UVData and 
        UVPSpec files (when redshift range is specified).
        """
        # Check that type errors and bounds errors are raised
        nt.assert_raises(AttributeError, utils.spw_range_from_redshifts, 
                         np.arange(3), z_range=(9.7, 12.1))
        for obj in [self.uvd, self.uvp]:
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(5., 8.)) # lower bound
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(10., 20.)) # upper bound
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(11., 10.)) # wrong order
            
        # Check that valid frequency ranges are returned
        z_list = [(6.5, 7.5), (7.5, 8.5), (8.5, 9.5)]
        spw1 = utils.spw_range_from_redshifts(self.uvd, z_range=(7., 8.))
        spw2 = utils.spw_range_from_redshifts(self.uvd, z_range=z_list)
        spw3 = utils.spw_range_from_redshifts(self.uvd, z_range=(12., 14.), 
                                              bounds_error=False)
        spw4 = utils.spw_range_from_redshifts(self.uvd, z_range=(6.2, 7.2))
        
        # Make sure tuple vs. list arguments were handled correctly
        nt.ok_( isinstance(spw1, tuple) )
        nt.ok_( isinstance(spw2, list) )
        nt.ok_( len(spw2) == len(z_list) )
        
        # Make sure that this also works for UVPSpec objects
        spw5 = utils.spw_range_from_redshifts(self.uvp, z_range=(13.1, 13.2))
        nt.ok_( isinstance(spw5, tuple) )
        nt.ok_( spw5[0] is not None )
        
def test_log():
    """
    Test that log() prints output.
    """
    # print
    utils.log("message")
    utils.log("message", lvl=2)

    # logfile
    logf = open("logf.log", "w")
    utils.log("message", f=logf, verbose=False)
    logf.close()
    with open("logf.log", "r") as f:
        nt.assert_equal(f.readlines()[0], "message")

    # traceback
    logf = open("logf.log", "w")
    try:
        raise NameError
    except NameError:
        err, _, tb = sys.exc_info()
        utils.log("raised an exception", f=logf, tb=tb, verbose=False)
    logf.close()
    with open("logf.log", "r") as f:
        log = ''.join(f.readlines())
        nt.assert_true("NameError" in log and "raised an exception" in log)
    os.remove("logf.log")


def test_hash():
    """
    Check that MD5 hashing works.
    """
    hsh = utils.hash(np.ones((8,16)))

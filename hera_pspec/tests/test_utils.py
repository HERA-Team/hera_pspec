import unittest
import nose.tools as nt
import numpy as np
import os, sys, copy
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import utils
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

def test_log():
    """
    Test that log() prints output.
    """
    utils.log("message")
    utils.log("message", lvl=2)

def test_hash():
    """
    Check that MD5 hashing works.
    """
    hsh = utils.hash(np.ones((8,16)))
    

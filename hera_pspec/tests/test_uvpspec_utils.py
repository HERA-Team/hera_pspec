import unittest
import nose.tools as nt
import numpy as np
from hera_pspec import uvpspec_utils as uvputils

def test_conj_blpair_int():
    conj_blpair = uvputils._conj_blpair_int(1002003004)
    nt.assert_equal(conj_blpair, 3004001002)

def test_conj_bl_int():
    conj_bl = uvputils._conj_bl_int(1002)
    nt.assert_equal(conj_bl, 2001)

def test_conj_blpair():
    blpair = uvputils._conj_blpair(1002003004, which='first')
    nt.assert_equal(blpair, 2001003004)
    blpair = uvputils._conj_blpair(1002003004, which='second')
    nt.assert_equal(blpair, 1002004003)
    blpair = uvputils._conj_blpair(1002003004, which='both')
    nt.assert_equal(blpair, 2001004003)
    nt.assert_raises(ValueError, uvputils._conj_blpair, 2001003004, which='foo')

def test_fast_is_in():
    blps = [ 2001003004, 2001003004, 2001003004, 2001003004, 
             1002004003, 1002004003, 1002004003, 1002004003,
             2001004003, 2001004003, 1002004003, 2001003004 ]
    times = [ 0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(zip(blps, times))

    nt.assert_true(uvputils._fast_is_in(src_blpts, [(1002004003, 0.2)])[0])

def test_fast_lookup_blpairts():
    # Construct array of blpair-time tuples (including some out of order)
    blps = [ 2001003004, 2001003004, 2001003004, 2001003004, 
             1002004003, 1002004003, 1002004003, 1002004003,
             2001004003, 2001004003, 1002004003, 2001003004 ]
    times = [ 0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(zip(blps, times))
    
    # List of blpair-times to look up
    query_blpts = [(2001003004, 0.1), (1002004003, 0.1), (1002004003, 0.25)]
    
    # Look up indices, compare with expected result
    idxs = uvputils._fast_lookup_blpairts(src_blpts, np.array(query_blpts))
    np.testing.assert_array_equal(idxs, np.array([0, 4, 7]))

import unittest
import nose.tools as nt
import numpy as np
from hera_pspec import uvpspec_utils as uvputils

def test_conj_blpair_int():
    conj_blpair = uvputils._conj_blpair_int(101102103104)
    nt.assert_equal(conj_blpair, 103104101102)

def test_conj_bl_int():
    conj_bl = uvputils._conj_bl_int(101102)
    nt.assert_equal(conj_bl, 102101)

def test_conj_blpair():
    blpair = uvputils._conj_blpair(101102103104, which='first')
    nt.assert_equal(blpair, 102101103104)
    blpair = uvputils._conj_blpair(101102103104, which='second')
    nt.assert_equal(blpair, 101102104103)
    blpair = uvputils._conj_blpair(101102103104, which='both')
    nt.assert_equal(blpair, 102101104103)
    nt.assert_raises(ValueError, uvputils._conj_blpair, 102101103104, which='foo')

def test_fast_is_in():
    blps = [ 102101103104, 102101103104, 102101103104, 102101103104, 
             101102104103, 101102104103, 101102104103, 101102104103,
             102101104103, 102101104103, 102101104103, 102101104103 ]
    times = [ 0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(zip(blps, times))

    nt.assert_true(uvputils._fast_is_in(src_blpts, [(101102104103, 0.2)])[0])

def test_fast_lookup_blpairts():
    # Construct array of blpair-time tuples (including some out of order)
    blps = [ 102101103104, 102101103104, 102101103104, 102101103104, 
             101102104103, 101102104103, 101102104103, 101102104103,
             102101104103, 102101104103, 102101104103, 102101104103 ]
    times = [ 0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.2, 0.25, 
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(zip(blps, times))
    
    # List of blpair-times to look up
    query_blpts = [(102101103104, 0.1), (101102104103, 0.1), (101102104103, 0.25)]
    
    # Look up indices, compare with expected result
    idxs = uvputils._fast_lookup_blpairts(src_blpts, np.array(query_blpts))
    np.testing.assert_array_equal(idxs, np.array([0, 4, 7]))

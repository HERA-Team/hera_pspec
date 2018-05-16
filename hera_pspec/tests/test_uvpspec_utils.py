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




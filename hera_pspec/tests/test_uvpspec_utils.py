import unittest
import nose.tools as nt
import numpy as np
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec import testing, pspecbeam, UVPSpec
from pyuvdata import UVData
from hera_pspec.data import DATA_PATH
import os


def test_select_common():
    """
    Test selecting power spectra that two UVPSpec objects have in common.
    """
    # setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # Carve up some example UVPSpec objects
    uvp1 = uvp.select(times=np.unique(uvp.time_avg_array)[:-1], 
                           inplace=False)
    uvp2 = uvp.select(times=np.unique(uvp.time_avg_array)[1:], 
                           inplace=False)
    uvp3 = uvp.select(blpairs=np.unique(uvp.blpair_array)[1:], 
                           inplace=False)
    uvp4 = uvp.select(blpairs=np.unique(uvp.blpair_array)[:2], 
                           inplace=False)
    uvp5 = uvp.select(blpairs=np.unique(uvp.blpair_array)[:1], 
                           inplace=False)
    uvp6 = uvp.select(times=np.unique(uvp.time_avg_array)[:1], 
                           inplace=False)
    
    # Check that selecting on common times works
    uvp_list = [uvp1, uvp2]
    uvp_new = uvputils.select_common(uvp_list, spws=True, blpairs=True, 
                                     times=True, pols=True, inplace=False)
    nt.assert_equal(uvp_new[0], uvp_new[1])
    np.testing.assert_array_equal(uvp_new[0].time_avg_array, 
                                  uvp_new[1].time_avg_array)
    
    # Check that selecting on common baseline-pairs works
    uvp_list_2 = [uvp1, uvp2, uvp3]
    uvp_new_2 = uvputils.select_common(uvp_list_2, spws=True, blpairs=True, 
                                       times=True, pols=True, inplace=False)
    nt.assert_equal(uvp_new_2[0], uvp_new_2[1])
    nt.assert_equal(uvp_new_2[0], uvp_new_2[2])
    np.testing.assert_array_equal(uvp_new_2[0].time_avg_array, 
                                  uvp_new_2[1].time_avg_array)
    
    # Check that zero overlap in times raises a ValueError
    nt.assert_raises(ValueError, uvputils.select_common, [uvp2, uvp6], 
                                  spws=True, blpairs=True, times=True, 
                                  pols=True, inplace=False)
    
    # Check that zero overlap in times does *not* raise a ValueError if 
    # not selecting on times
    uvp_new_3 = uvputils.select_common([uvp2, uvp6], spws=True, 
                                       blpairs=True, times=False, 
                                       pols=True, inplace=False)
    
    # Check that zero overlap in baselines raises a ValueError
    nt.assert_raises(ValueError, uvputils.select_common, [uvp3, uvp5], 
                                  spws=True, blpairs=True, times=True, 
                                  pols=True, inplace=False)
    
    # Check that matching times are ignored when set to False
    uvp_new = uvputils.select_common(uvp_list, spws=True, blpairs=True, 
                                     times=False, pols=True, inplace=False)
    nt.assert_not_equal( np.sum(uvp_new[0].time_avg_array 
                              - uvp_new[1].time_avg_array), 0.)
    nt.assert_equal(len(uvp_new), len(uvp_list))
    
    # Check that in-place selection works
    uvputils.select_common(uvp_list, spws=True, blpairs=True, 
                           times=True, pols=True, inplace=True)
    nt.assert_equal(uvp1, uvp2)


def test_subtract_uvp():
    """ Test subtraction of two UVPSpec objects """
    # setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # add a dummy stats_array
    for k in uvp.get_all_keys():
        uvp.set_stats('mystat', k, np.ones((10, 30), dtype=np.complex))

    # test execution
    uvs = uvputils.subtract_uvp(uvp, uvp, run_check=True)
    nt.assert_true(isinstance(uvs, UVPSpec))
    nt.assert_true(hasattr(uvs, "stats_array"))
    nt.assert_true(hasattr(uvs, "cov_array"))

    # we subtracted uvp from itself, so data_array should be zero
    nt.assert_true(np.isclose(uvs.data_array[0], 0.0).all())

    # check stats_array is np.sqrt(2)
    nt.assert_true(np.isclose(uvs.stats_array['mystat'][0], np.sqrt(2)).all())
    

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

import unittest
import pytest
import numpy as np
from .. import uvpspec_utils as uvputils
from .. import testing, pspecbeam, UVPSpec
from pyuvdata import UVData
from hera_pspec.data import DATA_PATH
import os
import copy
import json


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
                                     times=True, polpairs=True, inplace=False)
    assert uvp_new[0] == uvp_new[1]
    np.testing.assert_array_equal(uvp_new[0].time_avg_array,
                                  uvp_new[1].time_avg_array)

    # Check that selecting on common baseline-pairs works
    uvp_list_2 = [uvp1, uvp2, uvp3]
    uvp_new_2 = uvputils.select_common(uvp_list_2, spws=True, blpairs=True,
                                       times=True, polpairs=True, inplace=False)
    assert uvp_new_2[0] == uvp_new_2[1]
    assert uvp_new_2[0] == uvp_new_2[2]
    np.testing.assert_array_equal(uvp_new_2[0].time_avg_array,
                                  uvp_new_2[1].time_avg_array)

    # Check that zero overlap in times raises a ValueError
    pytest.raises(ValueError, uvputils.select_common, [uvp2, uvp6],
                                  spws=True, blpairs=True, times=True,
                                  polpairs=True, inplace=False)

    # Check that zero overlap in times does *not* raise a ValueError if
    # not selecting on times
    uvp_new_3 = uvputils.select_common([uvp2, uvp6], spws=True,
                                       blpairs=True, times=False,
                                       polpairs=True, inplace=False)

    # Check that zero overlap in baselines raises a ValueError
    pytest.raises(ValueError, uvputils.select_common, [uvp3, uvp5],
                                  spws=True, blpairs=True, times=True,
                                  polpairs=True, inplace=False)

    # Check that matching times are ignored when set to False
    uvp_new = uvputils.select_common(uvp_list, spws=True, blpairs=True,
                                     times=False, polpairs=True, inplace=False)
    assert  np.sum(uvp_new[0].time_avg_array
                   - uvp_new[1].time_avg_array) != 0.
    assert len(uvp_new) == len(uvp_list)

    # Check that in-place selection works
    uvputils.select_common(uvp_list, spws=True, blpairs=True,
                           times=True, polpairs=True, inplace=True)
    assert uvp1 == uvp2

    # check uvplist > 2
    pytest.raises(IndexError, uvputils.select_common, uvp_list[:1])

    # check no spw overlap
    uvp7 = copy.deepcopy(uvp1)
    uvp7.freq_array += 10e6
    pytest.raises(ValueError, uvputils.select_common, [uvp1, uvp7], spws=True)

    # check no lst overlap
    uvp7 = copy.deepcopy(uvp1)
    uvp7.lst_avg_array += 0.1
    pytest.raises(ValueError, uvputils.select_common, [uvp1, uvp7], lsts=True)

    # check pol overlap
    uvp7 = copy.deepcopy(uvp1)
    uvp7.polpair_array[0] = 1212 # = (-8,-8)
    pytest.raises(ValueError, uvputils.select_common, [uvp1, uvp7],
                                 polpairs=True)

def test_get_blpairs_from_bls():
    """
    Test conversion of bls to set of blpairs.
    """
    # setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # Check that bls can be specified in several different ways
    blps = uvputils._get_blpairs_from_bls(uvp, bls=101102)
    blps = uvputils._get_blpairs_from_bls(uvp, bls=(101,102))
    blps = uvputils._get_blpairs_from_bls(uvp, bls=[101102, 101103])


def test_get_red_bls():
    """
    Test retrieval of redundant baseline groups.
    """
    # Setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # Get redundant baseline groups
    bls, lens, angs = uvp.get_red_bls()

    assert len(bls) == 3 # three red grps in this file
    assert len(bls) == len(lens) # Should be one length for each group
    assert len(bls) == len(angs) # Ditto, for angles

    # Check that number of grouped baselines = total no. of baselines
    num_bls = 0
    for grp in bls:
        for bl in grp:
            num_bls += 1
    assert num_bls == np.unique(uvp.bl_array).size


def test_get_red_blpairs():
    """
    Test retrieval of redundant baseline groups for baseline-pairs.
    """
    # Setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # Get redundant baseline groups
    blps, lens, angs = uvp.get_red_blpairs()

    assert len(blps) == 3 # three red grps in this file
    assert len(blps) == len(lens) # Should be one length for each group
    assert len(blps) == len(angs) # Ditto, for angles

    # Check output type
    assert isinstance(blps[0][0], int)

    # Check that number of grouped blps = total no. of blps
    num_blps = 0
    for grp in blps:
        for blp in grp:
            num_blps += 1
    assert num_blps == np.unique(uvp.blpair_array).size


def test_polpair_int2tuple():
    """
    Test conversion of polpair ints to tuples.
    """
    # List of polpairs to test
    polpairs = [('xx','xx'), ('xx','yy'), ('xy', 'yx'),
                ('pI','pI'), ('pI','pQ'), ('pQ','pQ'), ('pU','pU'),
                ('pV','pV') ]

    # Check that lists and single items work
    pol_ints = uvputils.polpair_tuple2int(polpairs)
    uvputils.polpair_tuple2int(polpairs[0])
    uvputils.polpair_int2tuple(1515)
    uvputils.polpair_int2tuple([1515,1414])
    uvputils.polpair_int2tuple(np.array([1515,1414]))

    # Test converting to int and then back again
    pol_pairs_returned = uvputils.polpair_int2tuple(pol_ints, pol_strings=True)
    for i in range(len(polpairs)):
        assert polpairs[i] == pol_pairs_returned[i]

    # Check that errors are raised appropriately
    pytest.raises(AssertionError, uvputils.polpair_int2tuple, ('xx','xx'))
    pytest.raises(AssertionError, uvputils.polpair_int2tuple, 'xx')
    pytest.raises(AssertionError, uvputils.polpair_int2tuple, 'pI')
    pytest.raises(ValueError, uvputils.polpair_int2tuple, 999)
    pytest.raises(ValueError, uvputils.polpair_int2tuple, [999,])


def test_subtract_uvp():
    """
    Test subtraction of two UVPSpec objects
    """
    # setup uvp
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)

    # add a dummy stats_array
    for k in uvp.get_all_keys():
        uvp.set_stats('mystat', k, np.ones((10, 30), dtype=complex))

    # test execution
    uvs = uvputils.subtract_uvp(uvp, uvp, run_check=True)
    assert isinstance(uvs, UVPSpec)
    assert hasattr(uvs, "stats_array")
    assert hasattr(uvs, "cov_array_real")

    # we subtracted uvp from itself, so data_array should be zero
    assert np.isclose(uvs.data_array[0], 0.0).all()

    # check stats_array is np.sqrt(2)
    assert np.isclose(uvs.stats_array['mystat'][0], np.sqrt(2)).all()


def test_conj_blpair_int():
    conj_blpair = uvputils._conj_blpair_int(101102103104)
    assert conj_blpair == 103104101102


def test_conj_bl_int():
    conj_bl = uvputils._conj_bl_int(101102)
    assert conj_bl == 102101


def test_conj_blpair():
    blpair = uvputils._conj_blpair(101102103104, which='first')
    assert blpair == 102101103104
    blpair = uvputils._conj_blpair(101102103104, which='second')
    assert blpair == 101102104103
    blpair = uvputils._conj_blpair(101102103104, which='both')
    assert blpair == 102101104103
    pytest.raises(ValueError, uvputils._conj_blpair, 102101103104, which='foo')


def test_fast_is_in():
    blps = [ 102101103104, 102101103104, 102101103104, 102101103104,
             101102104103, 101102104103, 101102104103, 101102104103,
             102101104103, 102101104103, 102101104103, 102101104103 ]
    times = [ 0.1, 0.15, 0.2, 0.25,
              0.1, 0.15, 0.2, 0.25,
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(list(zip(blps, times)))

    assert uvputils._fast_is_in(src_blpts, [(101102104103, 0.2)])[0]


def test_fast_lookup_blpairts():
    # Construct array of blpair-time tuples (including some out of order)
    blps = [ 102101103104, 102101103104, 102101103104, 102101103104,
             101102104103, 101102104103, 101102104103, 101102104103,
             102101104103, 102101104103, 102101104103, 102101104103 ]
    times = [ 0.1, 0.15, 0.2, 0.25,
              0.1, 0.15, 0.2, 0.25,
              0.1, 0.15, 0.3, 0.3, ]
    src_blpts = np.array(list(zip(blps, times)))

    # List of blpair-times to look up
    query_blpts = [(102101103104, 0.1), (101102104103, 0.1), (101102104103, 0.25)]

    # Look up indices, compare with expected result
    idxs = uvputils._fast_lookup_blpairts(src_blpts, np.array(query_blpts))
    np.testing.assert_array_equal(idxs, np.array([0, 4, 7]))

def test_r_param_compression():
    baselines = [(24,25), (37,38), (38,39)]

    rp = {'filter_centers':[0.],
          'filter_half_widths':[250e-9],
          'filter_factors':[1e-9]}

    r_params = {}

    for bl in baselines:
        key1 =  bl + ('xx',)
        key2 =  bl + ('xx',)
        r_params[key1] = rp
        r_params[key2] = rp

    rp_str_1 = uvputils.compress_r_params(r_params)
    rp = uvputils.decompress_r_params(rp_str_1)
    for rpk in rp:
        for rpfk in rp[rpk]:
            assert rp[rpk][rpfk] == r_params[rpk][rpfk]
    for rpk in r_params:
        for rpfk in r_params[rpk]:
            assert r_params[rpk][rpfk] == rp[rpk][rpfk]

    rp_str_2 = uvputils.compress_r_params(rp)
    assert json.loads(rp_str_1) == json.loads(rp_str_2)

    assert uvputils.compress_r_params({}) == ''
    assert uvputils.decompress_r_params('') == {}

import copy
import os

import numpy as np
import pytest
import pyuvdata
import pyuvdata.utils as uvutils

from hera_pspec import pstokes
from hera_pspec.data import DATA_PATH

dset1 = os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA")
dset2 = os.path.join(DATA_PATH, "zen.all.yy.LST.1.06964.uvA")
multipol_dset = os.path.join(DATA_PATH, "zen.2458116.31193.HH.uvh5")
multipol_dset_cal = os.path.join(DATA_PATH, "zen.2458116.31193.HH.flagged_abs.calfits")


def _load_uvd(path):
    uvd = pyuvdata.UVData()
    uvd.read_miriad(path)
    uvd.vis_units = "Jy"
    uvd.pol_convention = "avg"
    return uvd


@pytest.fixture()
def uvd1():
    return _load_uvd(dset1)


@pytest.fixture()
def uvd2():
    return _load_uvd(dset2)


def test_combine_pol(uvd1, uvd2):
    uvd1 = copy.deepcopy(uvd1)
    uvd2 = copy.deepcopy(uvd2)

    # basic execution on pol strings
    out1 = pstokes._combine_pol(uvd1, uvd2, "XX", "YY")
    # again w/ pol ints
    out2 = pstokes._combine_pol(uvd1, uvd2, -5, -6)
    # assert equivalence
    assert out1 == out2
    # basic execution with different polarization conventions
    # out1 assumed avg by default
    setattr(uvd1, "pol_convention", "sum")
    setattr(uvd2, "pol_convention", "sum")

    out3 = pstokes._combine_pol(uvd1, uvd2, "XX", "YY")
    assert np.allclose(out3.data_array, out1.data_array * 2.0)
    assert np.allclose(out3.nsample_array, out1.nsample_array)

    # check exceptions
    with pytest.raises(AssertionError, match="uvd1 must be a pyuvdata.UVData instance"):
        pstokes._combine_pol(dset1, dset2, "XX", "YY")
    with pytest.raises(AssertionError, match="pol2.*not used in constructing pstokes"):
        pstokes._combine_pol(uvd1, uvd2, "XX", 1)
    # if different polarization conventions
    setattr(uvd1, "pol_convention", "avg")
    with pytest.raises(
        ValueError, match="pol_convention of uvd1 and uvd2 are different"
    ):
        pstokes._combine_pol(uvd1, uvd2, "XX", "YY")


def test_combine_pol_arrays(uvd1, uvd2):
    uvd1 = copy.deepcopy(uvd1)
    uvd2 = copy.deepcopy(uvd2)

    # proper usage
    d1, f1, ns1 = pstokes._combine_pol_arrays(
        pol1=-5,
        pol2=-6,
        pstokes="pI",
        pol_convention="avg",
        data_list=[uvd1.data_array, uvd2.data_array],
        flags_list=[uvd1.flag_array, uvd2.flag_array],
        nsamples_list=[uvd1.nsample_array, uvd2.nsample_array],
    )
    # inputs can be None
    d2, f2, ns2 = pstokes._combine_pol_arrays(
        pol1=-5,
        pol2=-6,
        pstokes="pI",
        pol_convention="avg",
        data_list=None,
        flags_list=None,
        nsamples_list=None,
    )
    assert d2 is None
    assert f2 is None
    assert ns2 is None
    # polarizations can be strings
    d3, f3, ns3 = pstokes._combine_pol_arrays(
        pol1="XX",
        pol2="YY",
        pstokes="pI",
        pol_convention="avg",
        data_list=[uvd1.data_array, uvd2.data_array],
        flags_list=[uvd1.flag_array, uvd2.flag_array],
        nsamples_list=[uvd1.nsample_array, uvd2.nsample_array],
    )
    assert np.allclose(d1, d3)
    assert np.allclose(f1, f3)
    assert np.allclose(ns1, ns3)

    # check exceptions
    with pytest.raises(AssertionError, match="pol2.*not used in constructing pstokes"):
        pstokes._combine_pol_arrays("XX", "pI", "pI")
    with pytest.raises(AssertionError, match="pol1.*not used in constructing pstokes"):
        pstokes._combine_pol_arrays("pI", "YY", "pI")
    with pytest.raises(ValueError, match="pol_convention must be avg or sum"):
        pstokes._combine_pol_arrays("XX", "YY", "pI", pol_convention="blah")
    with pytest.raises(ValueError, match="Can only combine two arrays"):
        pstokes._combine_pol_arrays("XX", "YY", "pI", data_list=uvd1.data_array)
    with pytest.raises(ValueError, match="Can only combine two arrays"):
        pstokes._combine_pol_arrays(
            "XX",
            "YY",
            "pI",
            data_list=[uvd1.data_array, uvd1.data_array, uvd1.data_array],
        )
    with pytest.raises(
        AssertionError, match="Arrays in list must have identical shape"
    ):
        pstokes._combine_pol_arrays(
            "XX", "YY", "pI", data_list=[uvd1.data_array, uvd1.data_array[0]]
        )


def test_construct_pstokes(uvd1, uvd2):
    # test to form I and Q from single polarized UVData objects
    uvdI = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes="pI")
    uvdQ = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes="pQ")

    # check exceptions
    with pytest.raises(
        AssertionError, match="dset2 must be fed as a string or UVData object"
    ):
        pstokes.construct_pstokes(uvd1, 1)

    # check baselines
    uvd3 = uvd2.select(ant_str="auto", inplace=False)
    with pytest.raises(
        ValueError, match="dset1 and dset2 must have the same timestamps"
    ):
        pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    # check frequencies
    uvd3 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[:10], inplace=False)
    with pytest.raises(
        ValueError, match="dset1 and dset2 must have the same frequencies"
    ):
        pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    uvd3 = uvd1.select(frequencies=np.unique(uvd1.freq_array)[:10], inplace=False)
    uvd4 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[10:20], inplace=False)
    with pytest.raises(
        ValueError, match="dset1 and dset2 must have the same frequencies"
    ):
        pstokes.construct_pstokes(dset1=uvd3, dset2=uvd4)

    # check times
    uvd3 = uvd2.select(times=np.unique(uvd2.time_array)[0:3], inplace=False)
    with pytest.raises(
        ValueError, match="dset1 and dset2 must have the same timestamps"
    ):
        pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    uvd3 = uvd1.select(times=np.unique(uvd1.time_array)[0:3], inplace=False)
    uvd4 = uvd2.select(times=np.unique(uvd2.time_array)[1:4], inplace=False)
    with pytest.raises(
        ValueError, match="dset1 and dset2 must have the same timestamps"
    ):
        pstokes.construct_pstokes(dset1=uvd3, dset2=uvd4)

    # combining two polarizations (dset1 and dset2) together
    uvd3 = uvd1 + uvd2

    # test to form I and Q from dual polarized UVData objects
    uvdI = pstokes.construct_pstokes(dset1=uvd3, dset2=uvd3, pstokes="pI")

    # check except for same polarizations
    with pytest.raises(AssertionError, match="not found in dset2 object"):
        pstokes.construct_pstokes(dset1=uvd1, dset2=uvd1, pstokes="pI")


def test_construct_pstokes_multipol():
    """test construct_pstokes on multi-polarization files"""
    uvd = pyuvdata.UVData()
    uvd.read(multipol_dset)
    uvc = pyuvdata.UVCal()
    uvc.read_calfits(multipol_dset_cal)
    uvc.gain_scale = "Jy"
    uvc.pol_convention = "avg"
    uvutils.uvcalibrate(uvd, uvc)
    wgts = [(0.5, 0.5), (0.5, -0.5)]

    for i, ps in enumerate(["pI", "pQ"]):
        uvp = pstokes.construct_pstokes(dset1=uvd, dset2=uvd, pstokes=ps)
        # assert polarization array is correct
        assert uvp.polarization_array == np.array([i + 1])
        # assert data are properly summmed
        pstokes_vis = (
            uvd.get_data(23, 24, "xx") * wgts[i][0]
            + uvd.get_data(23, 24, "yy") * wgts[i][1]
        )
        assert np.isclose(pstokes_vis, uvp.get_data(23, 24, ps)).all()


def test_filter_dset_on_stokes_pol(uvd1, uvd2):
    dsets = [uvd1, uvd2]
    out = pstokes.filter_dset_on_stokes_pol(dsets, "pI")
    assert out[0].polarization_array[0] == -5
    assert out[1].polarization_array[0] == -6
    with pytest.raises(
        AssertionError, match="necessary input pols.*not found in dsets"
    ):
        pstokes.filter_dset_on_stokes_pol(dsets, "pV")
    dsets = [uvd2, uvd1]
    out2 = pstokes.filter_dset_on_stokes_pol(dsets, "pI")
    assert out == out2


def test_generate_pstokes_argparser():
    # test argparser for noise error bars.
    ap = pstokes.generate_pstokes_argparser()
    args = ap.parse_args(["input.uvh5", "--pstokes", "pI", "pQ", "--clobber"])
    assert args.inputdata == "input.uvh5"
    assert args.outputdata is None
    assert args.clobber

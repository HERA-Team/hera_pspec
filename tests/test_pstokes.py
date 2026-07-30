import copy
from pathlib import Path

import numpy as np
import pytest
import pyuvdata
import pyuvdata.utils as uvutils

from hera_pspec import pstokes
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

dset1 = str(DATA_PATH / "zen.all.xx.LST.1.06964.uvA")
dset2 = str(DATA_PATH / "zen.all.yy.LST.1.06964.uvA")
multipol_dset = DATA_PATH / "zen.2458116.31193.HH.uvh5"
multipol_dset_cal = DATA_PATH / "zen.2458116.31193.HH.flagged_abs.calfits"


def _load_uvd(path: str) -> pyuvdata.UVData:
    uvd = pyuvdata.UVData()
    uvd.read_miriad(path)
    uvd.vis_units = "Jy"
    uvd.pol_convention = "avg"
    return uvd


@pytest.fixture()
def uvd1(uvd_zen_all_xx: pyuvdata.UVData) -> pyuvdata.UVData:
    uvd = copy.deepcopy(uvd_zen_all_xx)
    uvd.vis_units = "Jy"
    uvd.pol_convention = "avg"
    return uvd


@pytest.fixture()
def uvd2() -> pyuvdata.UVData:
    return _load_uvd(dset2)


@pytest.fixture(scope="module")
def multipol_uvd() -> pyuvdata.UVData:
    """Calibrated multi-polarization UVData used to test pstokes weighting in construct_pstokes."""
    uvd = pyuvdata.UVData()
    uvd.read(multipol_dset)
    uvc = pyuvdata.UVCal()
    uvc.read_calfits(multipol_dset_cal)
    uvc.gain_scale = "Jy"
    uvc.pol_convention = "avg"
    uvutils.uvcalibrate(uvd, uvc)
    return uvd


class TestCombinePol:
    def test_basic_execution(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that specifying polarizations as strings or integer pol codes gives equivalent results."""
        out1 = pstokes._combine_pol(uvd1, uvd2, "XX", "YY")
        out2 = pstokes._combine_pol(uvd1, uvd2, -5, -6)
        assert out1 == out2

    def test_sum_convention_doubles_data(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that pol_convention='sum' doubles the data relative to the default 'avg' convention."""
        out1 = pstokes._combine_pol(uvd1, uvd2, "XX", "YY")
        uvd1.pol_convention = "sum"
        uvd2.pol_convention = "sum"
        out3 = pstokes._combine_pol(uvd1, uvd2, "XX", "YY")
        assert np.allclose(out3.data_array, out1.data_array * 2.0)
        assert np.allclose(out3.nsample_array, out1.nsample_array)

    def test_raises_on_non_uvdata_input(self) -> None:
        """Check that a non-UVData uvd1 raises an AssertionError."""
        with pytest.raises(
            AssertionError, match="uvd1 must be a pyuvdata.UVData instance"
        ):
            pstokes._combine_pol(dset1, dset2, "XX", "YY")

    def test_raises_on_invalid_pol2(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that a pol2 not used in constructing the requested pstokes raises an AssertionError."""
        with pytest.raises(
            AssertionError, match="pol2.*not used in constructing pstokes"
        ):
            pstokes._combine_pol(uvd1, uvd2, "XX", 1)

    def test_raises_on_mismatched_pol_convention(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that differing pol_convention values on uvd1/uvd2 raise a ValueError."""
        uvd1.pol_convention = "avg"
        uvd2.pol_convention = "sum"
        with pytest.raises(
            ValueError, match="pol_convention of uvd1 and uvd2 are different"
        ):
            pstokes._combine_pol(uvd1, uvd2, "XX", "YY")


class TestCombinePolArrays:
    def test_basic_execution(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that _combine_pol_arrays averages data, ORs flags, and harmonic-combines nsamples."""
        d1, f1, ns1 = pstokes._combine_pol_arrays(
            pol1=-5,
            pol2=-6,
            pstokes="pI",
            pol_convention="avg",
            data_list=[uvd1.data_array, uvd2.data_array],
            flags_list=[uvd1.flag_array, uvd2.flag_array],
            nsamples_list=[uvd1.nsample_array, uvd2.nsample_array],
        )
        # pI with the "avg" convention is (XX + YY) / 2
        assert np.allclose(d1, 0.5 * (uvd1.data_array + uvd2.data_array))
        # flags are combined with a logical OR
        np.testing.assert_array_equal(
            f1, np.logical_or(uvd1.flag_array, uvd2.flag_array)
        )
        # nsamples are combined to preserve proper variance (see hera_pspec issue #406)
        valid = (uvd1.nsample_array > 0) & (uvd2.nsample_array > 0)
        expected_ns = np.zeros_like(uvd1.nsample_array, dtype=float)
        expected_ns[valid] = 4.0 / (
            1.0 / uvd1.nsample_array[valid] + 1.0 / uvd2.nsample_array[valid]
        )
        assert np.allclose(ns1, expected_ns)

    def test_none_inputs_return_none(self) -> None:
        """Check that all-None array inputs propagate through as None outputs."""
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

    def test_string_and_int_pols_equivalent(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that specifying pol1/pol2 as strings gives the same result as integer pol codes."""
        kwargs = dict(
            pstokes="pI",
            pol_convention="avg",
            data_list=[uvd1.data_array, uvd2.data_array],
            flags_list=[uvd1.flag_array, uvd2.flag_array],
            nsamples_list=[uvd1.nsample_array, uvd2.nsample_array],
        )
        d1, f1, ns1 = pstokes._combine_pol_arrays(pol1=-5, pol2=-6, **kwargs)
        d3, f3, ns3 = pstokes._combine_pol_arrays(pol1="XX", pol2="YY", **kwargs)
        assert np.allclose(d1, d3)
        assert np.allclose(f1, f3)
        assert np.allclose(ns1, ns3)

    @pytest.mark.parametrize(
        "pol1,pol2,match",
        [
            ("XX", "pI", "pol2.*not used in constructing pstokes"),
            ("pI", "YY", "pol1.*not used in constructing pstokes"),
        ],
    )
    def test_raises_on_invalid_pols(self, pol1: str, pol2: str, match: str) -> None:
        """Check that pol1/pol2 not consistent with the requested pstokes raises an AssertionError."""
        with pytest.raises(AssertionError, match=match):
            pstokes._combine_pol_arrays(pol1, pol2, "pI")

    def test_raises_on_invalid_pol_convention(self) -> None:
        """Check that an unrecognized pol_convention raises a ValueError."""
        with pytest.raises(ValueError, match="pol_convention must be avg or sum"):
            pstokes._combine_pol_arrays("XX", "YY", "pI", pol_convention="blah")

    @pytest.mark.parametrize("n_arrays", [1, 3])
    def test_raises_on_wrong_number_of_arrays(
        self, uvd1: pyuvdata.UVData, n_arrays: int
    ) -> None:
        """Check that data_list must contain exactly two arrays."""
        data_list = uvd1.data_array if n_arrays == 1 else [uvd1.data_array] * n_arrays
        with pytest.raises(ValueError, match="Can only combine two arrays"):
            pstokes._combine_pol_arrays("XX", "YY", "pI", data_list=data_list)

    def test_raises_on_mismatched_array_shapes(self, uvd1: pyuvdata.UVData) -> None:
        """Check that two arrays of differing shape raise an AssertionError."""
        with pytest.raises(
            AssertionError, match="Arrays in list must have identical shape"
        ):
            pstokes._combine_pol_arrays(
                "XX", "YY", "pI", data_list=[uvd1.data_array, uvd1.data_array[0]]
            )


class TestConstructPstokes:
    @pytest.mark.parametrize(
        "pstokes_str,pstokes_num,weight2", [("pI", 1, 1.0), ("pQ", 2, -1.0)]
    )
    def test_basic_execution(
        self,
        uvd1: pyuvdata.UVData,
        uvd2: pyuvdata.UVData,
        pstokes_str: str,
        pstokes_num: int,
        weight2: float,
    ) -> None:
        """Check that pI/pQ are correctly weighted-combined from XX (dset1) and YY (dset2) data."""
        uvdS = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes=pstokes_str)
        assert uvdS.Npols == 1
        assert uvdS.polarization_array[0] == pstokes_num
        expected_data = 0.5 * (uvd1.data_array + weight2 * uvd2.data_array)
        assert np.allclose(uvdS.data_array, expected_data)

    def test_raises_on_invalid_dset2_type(self, uvd1: pyuvdata.UVData) -> None:
        """Check that a non-UVData, non-string dset2 raises an AssertionError."""
        with pytest.raises(
            AssertionError, match="dset2 must be fed as a string or UVData object"
        ):
            pstokes.construct_pstokes(uvd1, 1)

    def test_raises_when_baselines_differ(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that restricting dset2 to auto-correlations (different baselines) raises a mismatch error."""
        uvd3 = uvd2.select(ant_str="auto", inplace=False)
        with pytest.raises(
            ValueError, match="dset1 and dset2 must have the same timestamps"
        ):
            pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    def test_raises_on_mismatched_frequencies_one_sided(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that restricting only dset2's frequencies raises a frequency-mismatch error."""
        uvd3 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[:10], inplace=False)
        with pytest.raises(
            ValueError, match="dset1 and dset2 must have the same frequencies"
        ):
            pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    def test_raises_on_mismatched_frequencies_partial_overlap(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that non-overlapping frequency subsets on both datasets raise a frequency-mismatch error."""
        uvd3 = uvd1.select(frequencies=np.unique(uvd1.freq_array)[:10], inplace=False)
        uvd4 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[10:20], inplace=False)
        with pytest.raises(
            ValueError, match="dset1 and dset2 must have the same frequencies"
        ):
            pstokes.construct_pstokes(dset1=uvd3, dset2=uvd4)

    def test_raises_on_mismatched_timestamps_one_sided(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that restricting only dset2's times raises a timestamp-mismatch error."""
        uvd3 = uvd2.select(times=np.unique(uvd2.time_array)[0:3], inplace=False)
        with pytest.raises(
            ValueError, match="dset1 and dset2 must have the same timestamps"
        ):
            pstokes.construct_pstokes(dset1=uvd1, dset2=uvd3)

    def test_raises_on_mismatched_timestamps_partial_overlap(
        self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
    ) -> None:
        """Check that non-overlapping time subsets on both datasets raise a timestamp-mismatch error."""
        uvd3 = uvd1.select(times=np.unique(uvd1.time_array)[0:3], inplace=False)
        uvd4 = uvd2.select(times=np.unique(uvd2.time_array)[1:4], inplace=False)
        with pytest.raises(
            ValueError, match="dset1 and dset2 must have the same timestamps"
        ):
            pstokes.construct_pstokes(dset1=uvd3, dset2=uvd4)

    def test_dual_pol_input(self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData) -> None:
        """Check that pI can be constructed from a single dual-polarization UVData object."""
        uvd3 = uvd1 + uvd2
        pstokes.construct_pstokes(dset1=uvd3, dset2=uvd3, pstokes="pI")

    def test_raises_when_same_polarization_used_twice(
        self, uvd1: pyuvdata.UVData
    ) -> None:
        """Check that using the same single-pol dataset for both dset1 and dset2 raises an AssertionError."""
        with pytest.raises(AssertionError, match="not found in dset2 object"):
            pstokes.construct_pstokes(dset1=uvd1, dset2=uvd1, pstokes="pI")

    @pytest.mark.parametrize(
        "i,ps,weights", [(0, "pI", (0.5, 0.5)), (1, "pQ", (0.5, -0.5))]
    )
    def test_multipol(
        self,
        multipol_uvd: pyuvdata.UVData,
        i: int,
        ps: str,
        weights: tuple[float, float],
    ) -> None:
        """Check that construct_pstokes on multi-pol data applies the correct per-Stokes weighting."""
        uvp = pstokes.construct_pstokes(
            dset1=multipol_uvd, dset2=multipol_uvd, pstokes=ps
        )
        assert uvp.polarization_array == np.array([i + 1])
        pstokes_vis = (
            multipol_uvd.get_data(23, 24, "xx") * weights[0]
            + multipol_uvd.get_data(23, 24, "yy") * weights[1]
        )
        assert np.isclose(pstokes_vis, uvp.get_data(23, 24, ps)).all()


def test_filter_dset_on_stokes_pol(
    uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData
) -> None:
    """Check that filter_dset_on_stokes_pol picks out the right input pols, order-independently."""
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


def test_generate_pstokes_argparser() -> None:
    """Check that the pstokes argparser correctly parses input/output/pstokes/clobber arguments."""
    ap = pstokes.generate_pstokes_argparser()
    args = ap.parse_args(["input.uvh5", "--pstokes", "pI", "pQ", "--clobber"])
    assert args.inputdata == "input.uvh5"
    assert args.outputdata is None
    assert args.clobber

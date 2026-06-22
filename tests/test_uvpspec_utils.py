import copy
import json

import numpy as np
import pytest

from hera_pspec import UVPSpec, grouping
from hera_pspec import uvpspec_utils as uvputils


@pytest.fixture
def uvp_with_stats(vanilla_uvp_with_beam: UVPSpec) -> UVPSpec:
    """Copy of vanilla_uvp_with_beam with a 'mystat' stats_array set for all keys."""
    uvp = copy.deepcopy(vanilla_uvp_with_beam)
    for k in uvp.get_all_keys():
        uvp.set_stats("mystat", k, np.ones_like(uvp.get_data(k), dtype=complex))
    return uvp


@pytest.fixture
def common_select_uvps(
    vanilla_uvp_with_beam: UVPSpec,
) -> tuple[UVPSpec, UVPSpec, UVPSpec, UVPSpec, UVPSpec]:
    """Five UVPSpec selections of vanilla_uvp_with_beam (by time or blpair) for testing select_common."""
    uvp1 = vanilla_uvp_with_beam.select(
        times=np.unique(vanilla_uvp_with_beam.time_avg_array)[:-1], inplace=False
    )
    uvp2 = vanilla_uvp_with_beam.select(
        times=np.unique(vanilla_uvp_with_beam.time_avg_array)[1:], inplace=False
    )
    uvp3 = vanilla_uvp_with_beam.select(
        blpairs=np.unique(vanilla_uvp_with_beam.blpair_array)[1:], inplace=False
    )
    uvp5 = vanilla_uvp_with_beam.select(
        blpairs=np.unique(vanilla_uvp_with_beam.blpair_array)[:1], inplace=False
    )
    uvp6 = vanilla_uvp_with_beam.select(
        times=np.unique(vanilla_uvp_with_beam.time_avg_array)[:1], inplace=False
    )
    return uvp1, uvp2, uvp3, uvp5, uvp6


class TestSelectCommon:
    def test_common_times(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that selecting on common times gives identical, time-aligned UVPSpecs."""
        uvp1, uvp2, _, _, _ = common_select_uvps
        uvp_new = uvputils.select_common(
            [uvp1, uvp2],
            spws=True,
            blpairs=True,
            times=True,
            polpairs=True,
            inplace=False,
        )
        assert uvp_new[0] == uvp_new[1]
        np.testing.assert_array_equal(
            uvp_new[0].time_avg_array, uvp_new[1].time_avg_array
        )

    def test_common_blpairs(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that selecting on common baseline-pairs gives identical, time-aligned UVPSpecs."""
        uvp1, uvp2, uvp3, _, _ = common_select_uvps
        uvp_new = uvputils.select_common(
            [uvp1, uvp2, uvp3],
            spws=True,
            blpairs=True,
            times=True,
            polpairs=True,
            inplace=False,
        )
        assert uvp_new[0] == uvp_new[1]
        assert uvp_new[0] == uvp_new[2]
        np.testing.assert_array_equal(
            uvp_new[0].time_avg_array, uvp_new[1].time_avg_array
        )

    def test_raises_on_no_time_overlap(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that zero overlap in times raises a ValueError when times=True."""
        _, uvp2, _, _, uvp6 = common_select_uvps
        with pytest.raises(ValueError, match="No times were found"):
            uvputils.select_common(
                [uvp2, uvp6],
                spws=True,
                blpairs=True,
                times=True,
                polpairs=True,
                inplace=False,
            )

    def test_ignores_time_overlap_when_times_false(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that zero overlap in times does not raise when times=False, and leaves times unselected."""
        _, uvp2, _, _, uvp6 = common_select_uvps
        uvp_new = uvputils.select_common(
            [uvp2, uvp6],
            spws=True,
            blpairs=True,
            times=False,
            polpairs=True,
            inplace=False,
        )
        np.testing.assert_array_equal(uvp_new[0].time_avg_array, uvp2.time_avg_array)
        np.testing.assert_array_equal(uvp_new[1].time_avg_array, uvp6.time_avg_array)

    def test_raises_on_no_blpair_overlap(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that zero overlap in baseline-pairs raises a ValueError."""
        _, _, uvp3, uvp5, _ = common_select_uvps
        with pytest.raises(ValueError, match="No baseline-pairs were found"):
            uvputils.select_common(
                [uvp3, uvp5],
                spws=True,
                blpairs=True,
                times=True,
                polpairs=True,
                inplace=False,
            )

    def test_ignores_matching_times_when_times_false(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that matching times are left unaligned when times=False."""
        uvp1, uvp2, _, _, _ = common_select_uvps
        uvp_new = uvputils.select_common(
            [uvp1, uvp2],
            spws=True,
            blpairs=True,
            times=False,
            polpairs=True,
            inplace=False,
        )
        assert np.sum(uvp_new[0].time_avg_array - uvp_new[1].time_avg_array) != 0.0
        assert len(uvp_new) == 2

    def test_inplace_selection(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that inplace=True mutates the input UVPSpecs to match."""
        uvp1, uvp2, _, _, _ = common_select_uvps
        uvp_list = [uvp1, uvp2]
        uvputils.select_common(
            uvp_list,
            spws=True,
            blpairs=True,
            times=True,
            polpairs=True,
            inplace=True,
        )
        assert uvp1 == uvp2

    def test_raises_on_fewer_than_two_uvps(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that a single-element uvp_list raises an IndexError."""
        uvp1, *_ = common_select_uvps
        with pytest.raises(IndexError, match="uvp_list must contain two or more"):
            uvputils.select_common([uvp1])

    def test_raises_on_no_spw_overlap(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that non-overlapping frequencies raise a ValueError when spws=True."""
        uvp1, *_ = common_select_uvps
        uvp7 = copy.deepcopy(uvp1)
        uvp7.freq_array += 10e6
        with pytest.raises(ValueError, match="No spectral windows were found"):
            uvputils.select_common([uvp1, uvp7], spws=True)

    def test_raises_on_no_lst_overlap(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that non-overlapping LSTs raise a ValueError when lsts=True."""
        uvp1, *_ = common_select_uvps
        uvp7 = copy.deepcopy(uvp1)
        uvp7.lst_avg_array += 0.1
        with pytest.raises(ValueError, match="No lsts were found"):
            uvputils.select_common([uvp1, uvp7], lsts=True)

    def test_raises_on_no_polpair_overlap(
        self, common_select_uvps: tuple[UVPSpec, ...]
    ) -> None:
        """Check that non-overlapping polarization-pairs raise a ValueError when polpairs=True."""
        uvp1, *_ = common_select_uvps
        uvp7 = copy.deepcopy(uvp1)
        uvp7.polpair_array[0] = 1212  # = (-8, -8)
        with pytest.raises(ValueError, match="No polarization-pairs were found"):
            uvputils.select_common([uvp1, uvp7], polpairs=True)


def test_get_blpairs_from_bls(vanilla_uvp_with_beam: UVPSpec) -> None:
    """Check that integer, tuple, and list baseline specifications select consistent, non-empty blpair subsets."""
    blp_select_int = uvputils._get_blpairs_from_bls(vanilla_uvp_with_beam, bls=101102)
    # baseline ints encode antnums as (ant+100) concatenated, so (1, 2) <-> 101102
    blp_select_tuple = uvputils._get_blpairs_from_bls(vanilla_uvp_with_beam, bls=(1, 2))
    blp_select_list = uvputils._get_blpairs_from_bls(vanilla_uvp_with_beam, bls=[101102, 101103])

    assert blp_select_int.any()
    # an integer baseline and the equivalent antenna-pair tuple select the same blpairs
    np.testing.assert_array_equal(blp_select_int, blp_select_tuple)
    # selecting on a superset of baselines selects a superset of blpairs
    assert np.all(blp_select_list[blp_select_int])


class TestRedundantGroups:
    def test_get_red_bls(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that get_red_bls groups all baselines into the right number of redundant groups."""
        bls, lens, angs = vanilla_uvp_with_beam.get_red_bls()

        assert len(bls) == 3  # three red grps in this file
        assert len(bls) == len(lens)  # Should be one length for each group
        assert len(bls) == len(angs)  # Ditto, for angles

        # Check that number of grouped baselines = total no. of baselines
        num_bls = sum(len(grp) for grp in bls)
        assert num_bls == np.unique(vanilla_uvp_with_beam.bl_array).size

    def test_get_red_blpairs(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that get_red_blpairs groups all baseline-pairs into the right number of redundant groups."""
        blps, lens, angs = vanilla_uvp_with_beam.get_red_blpairs()

        assert len(blps) == 3  # three red grps in this file
        assert len(blps) == len(lens)  # Should be one length for each group
        assert len(blps) == len(angs)  # Ditto, for angles

        # Check output type
        assert isinstance(blps[0][0], int)

        # Check that number of grouped blps = total no. of blps
        num_blps = sum(len(grp) for grp in blps)
        assert num_blps == np.unique(vanilla_uvp_with_beam.blpair_array).size


# Polpairs to round-trip through polpair_tuple2int / polpair_int2tuple.
POLPAIRS = [
    ("xx", "xx"),
    ("xx", "yy"),
    ("xy", "yx"),
    ("pI", "pI"),
    ("pI", "pQ"),
    ("pQ", "pQ"),
    ("pU", "pU"),
    ("pV", "pV"),
]


class TestPolpairInt2Tuple:
    def test_tuple2int_scalar_matches_list(self) -> None:
        """Check that polpair_tuple2int on a single polpair matches the corresponding entry from the list-based call."""
        pol_ints = uvputils.polpair_tuple2int(POLPAIRS)
        assert uvputils.polpair_tuple2int(POLPAIRS[0]) == pol_ints[0]

    def test_scalar_list_and_array_inputs_agree(self) -> None:
        """Check that scalar, list, and ndarray inputs to polpair_int2tuple are mutually consistent."""
        single = uvputils.polpair_int2tuple(1515)
        from_list = uvputils.polpair_int2tuple([1515, 1414])
        from_array = uvputils.polpair_int2tuple(np.array([1515, 1414]))
        assert from_list[0] == single
        assert tuple(from_array[0]) == tuple(single)

    @pytest.mark.parametrize("polpair", POLPAIRS)
    def test_round_trip(self, polpair: tuple[str, str]) -> None:
        """Check that converting a polpair to int and back to strings reproduces the original tuple."""
        pol_int = uvputils.polpair_tuple2int(polpair)
        assert uvputils.polpair_int2tuple(pol_int, pol_strings=True) == polpair

    @pytest.mark.parametrize(
        "value",
        [("xx", "xx"), "xx", "pI"],
        ids=["tuple", "single_str", "stokes_str"],
    )
    def test_int2tuple_raises_on_non_integer_input(self, value) -> None:
        """Check that non-integer inputs raise an AssertionError."""
        with pytest.raises(AssertionError, match="polpair must be integer"):
            uvputils.polpair_int2tuple(value)

    @pytest.mark.parametrize("value", [999, [999]], ids=["scalar", "list"])
    def test_int2tuple_raises_on_invalid_code(self, value) -> None:
        """Check that an integer not corresponding to a valid polpair code raises a ValueError."""
        with pytest.raises(
            ValueError, match="polpair integer evaluates to an invalid"
        ):
            uvputils.polpair_int2tuple(value)


class TestSubtractUvp:
    def test_basic(self, uvp_with_stats: UVPSpec) -> None:
        """Check that subtracting a UVPSpec from itself zeroes the data and scales stats_array by sqrt(2)."""
        uvp = uvp_with_stats
        uvs = uvputils.subtract_uvp(uvp, uvp, run_check=True)
        assert isinstance(uvs, UVPSpec)
        assert hasattr(uvs, "stats_array")
        assert hasattr(uvs, "cov_array_real")
        assert hasattr(uvs, "window_function_array")
        # we subtracted uvp from itself, so data_array should be zero
        assert np.isclose(uvs.data_array[0], 0.0).all()
        # check stats_array is np.sqrt(2)
        assert np.isclose(uvs.stats_array["mystat"][0], np.sqrt(2)).all()

    def test_delay_averaged(self, uvp_with_stats: UVPSpec) -> None:
        """Check that subtract_uvp works on delay-averaged spectra, but not when only one side is delay-averaged."""
        uvp = uvp_with_stats
        averaged_uvp = grouping.average_in_delay_bins(uvp, kernel=np.array([1, 1, 1]))
        uvs = uvputils.subtract_uvp(averaged_uvp, averaged_uvp, run_check=True)
        assert isinstance(uvs, UVPSpec)
        assert hasattr(uvs, "stats_array")
        assert hasattr(uvs, "cov_array_real")
        assert hasattr(uvs, "window_function_array")
        # make sure you cannot combine spectra if only one has been delay averaged
        with pytest.raises(ValueError, match="No spectral windows were found"):
            uvputils.subtract_uvp(averaged_uvp, uvp)


class TestConjugation:
    def test_conj_blpair_int(self) -> None:
        """Check that _conj_blpair_int swaps the two baselines in a blpair integer code."""
        conj_blpair = uvputils._conj_blpair_int(101102103104)
        assert conj_blpair == 103104101102

    def test_conj_bl_int(self) -> None:
        """Check that _conj_bl_int swaps the two antennas in a baseline integer code."""
        conj_bl = uvputils._conj_bl_int(101102)
        assert conj_bl == 102101

    @pytest.mark.parametrize(
        "which,expected",
        [("first", 102101103104), ("second", 101102104103), ("both", 102101104103)],
    )
    def test_conj_blpair(self, which: str, expected: int) -> None:
        """Check that _conj_blpair conjugates the first, second, or both baselines as requested."""
        assert uvputils._conj_blpair(101102103104, which=which) == expected

    def test_conj_blpair_raises_on_invalid_which(self) -> None:
        """Check that an unrecognized `which` value raises a ValueError."""
        with pytest.raises(ValueError, match="didn't recognize foo"):
            uvputils._conj_blpair(102101103104, which="foo")


@pytest.fixture
def src_blpts() -> np.ndarray:
    """Array of (blpair, time) tuples, including some out-of-order entries, for blpair-time lookup tests."""
    blps = [
        102101103104,
        102101103104,
        102101103104,
        102101103104,
        101102104103,
        101102104103,
        101102104103,
        101102104103,
        102101104103,
        102101104103,
        102101104103,
        102101104103,
    ]
    times = [0.1, 0.15, 0.2, 0.25, 0.1, 0.15, 0.2, 0.25, 0.1, 0.15, 0.3, 0.3]
    return np.array(list(zip(blps, times)))


class TestBlpairTimeLookup:
    def test_is_in(self, src_blpts: np.ndarray) -> None:
        """Check that _is_in correctly reports membership of a (blpair, time) tuple."""
        assert uvputils._is_in(src_blpts, [(101102104103, 0.2)])[0]

    def test_fast_lookup_blpairts(self, src_blpts: np.ndarray) -> None:
        """Check that _fast_lookup_blpairts returns the correct indices for a list of queries."""
        query_blpts = [(102101103104, 0.1), (101102104103, 0.1), (101102104103, 0.25)]
        idxs = uvputils._fast_lookup_blpairts(src_blpts, np.array(query_blpts))
        np.testing.assert_array_equal(idxs, np.array([0, 4, 7]))


def test_r_param_compression() -> None:
    """Check that compress_r_params/decompress_r_params round-trip exactly, and that empty input round-trips to empty."""
    baselines = [(24, 25), (37, 38), (38, 39)]
    rp = {
        "filter_centers": [0.0],
        "filter_half_widths": [250e-9],
        "filter_factors": [1e-9],
    }
    r_params = {bl + ("xx",): rp for bl in baselines}

    rp_str_1 = uvputils.compress_r_params(r_params)
    decompressed = uvputils.decompress_r_params(rp_str_1)
    assert decompressed == r_params

    rp_str_2 = uvputils.compress_r_params(decompressed)
    assert json.loads(rp_str_1) == json.loads(rp_str_2)

    assert uvputils.compress_r_params({}) == ""
    assert uvputils.decompress_r_params("") == {}

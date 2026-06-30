import copy
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hera_cal import redcal
from pyuvdata import UVData

from hera_pspec import PSpecBeamUV, UVPSpec, testing, utils
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Path to the zen.2458042.17772.xx.HH.uvXA miriad data file.
ZEN_2458042_XX_PATH = str(DATA_PATH / "zen.2458042.17772.xx.HH.uvXA")
# Path to the zen.all.xx.LST.1.06964.uvA miriad data file.
ZEN_ALL_XX_PATH = str(DATA_PATH / "zen.all.xx.LST.1.06964.uvA")


@pytest.fixture(scope="module")
def uvd_zen_all_xx_meta() -> UVData:
    """zen.all.xx.LST.1.06964.uvA, metadata only (read_data=False)."""
    uvdata = UVData()
    uvdata.read_miriad(ZEN_ALL_XX_PATH, read_data=False)
    return uvdata


class TestCircularAverage:
    def test_handles_wrap_near_2pi(self) -> None:
        """Check that wrapping near 2*pi averages near 0/2*pi rather than pi."""
        angles_wrap = np.array([6.2, 0.1])
        result = utils.circular_average(angles_wrap)
        assert (result < 0.5) or (result > 5.5)

        # Verify the buggy arithmetic mean would give the wrong (pi-region) result
        buggy_result = np.mean(angles_wrap)
        assert 2.5 < buggy_result < 3.5

    def test_matches_arithmetic_mean_without_wrap(self) -> None:
        """Check that circular and arithmetic means agree when there's no wrap."""
        angles_no_wrap = np.array([1.0, 1.5, 2.0])
        circular_result = utils.circular_average(angles_no_wrap)
        arithmetic_result = np.mean(angles_no_wrap)
        assert np.isclose(circular_result, arithmetic_result, atol=1e-10)

    def test_handles_multiple_wrapping_values(self) -> None:
        """Check that multiple angles spanning the wrap point still average near 0/2*pi."""
        angles_multi = np.array([6.28, 0.01, 6.25, 0.05])
        result_multi = utils.circular_average(angles_multi)
        assert (result_multi < 0.5) or (result_multi > 5.5)

    def test_averages_along_axis(self) -> None:
        """Check that averaging a 2D array along axis=0 wraps per-column where needed."""
        angles_2d = np.array([[6.2, 1.0, 3.0], [0.1, 1.5, 3.5]])
        result_2d = utils.circular_average(angles_2d, axis=0)
        assert result_2d.shape == (3,)
        assert (result_2d[0] < 0.5) or (result_2d[0] > 5.5)
        assert np.isclose(result_2d[1], 1.25, atol=1e-10)
        assert np.isclose(result_2d[2], 3.25, atol=1e-10)

    def test_single_value_returns_itself(self) -> None:
        """Check that averaging a single angle returns that angle unchanged."""
        single_angle = np.array([1.5])
        result_single = utils.circular_average(single_angle)
        assert np.isclose(result_single, 1.5, atol=1e-10)


class TestCov:
    def test_basic_execution(self, uvd_zen_2458042_xx: UVData) -> None:
        """Check that cov() returns a complex Nfreq x Nfreq matrix for both auto- and cross-baseline inputs."""
        d1 = uvd_zen_2458042_xx.get_data(24, 25)
        w1 = (~uvd_zen_2458042_xx.get_flags(24, 25)).astype(float)
        cov = utils.cov(d1, w1)
        assert cov.shape == (60, 60)
        assert cov.dtype == complex

        d2 = uvd_zen_2458042_xx.get_data(37, 38)
        w2 = (~uvd_zen_2458042_xx.get_flags(37, 38)).astype(float)
        cov = utils.cov(d1, w2, d2=d2, w2=w2)
        assert cov.shape == (60, 60)
        assert cov.dtype == complex

    def test_raises_on_complex_weights(self, uvd_zen_2458042_xx: UVData) -> None:
        """Check that complex-valued weight matrices raise a TypeError."""
        d1 = uvd_zen_2458042_xx.get_data(24, 25)
        w1 = (~uvd_zen_2458042_xx.get_flags(24, 25)).astype(float)
        d2 = uvd_zen_2458042_xx.get_data(37, 38)
        w2 = (~uvd_zen_2458042_xx.get_flags(37, 38)).astype(float)
        with pytest.raises(TypeError, match="Weight matrices must be real"):
            utils.cov(d1, w1 * 1j)
        with pytest.raises(TypeError, match="Weight matrices must be real"):
            utils.cov(d1, w1, d2=d2, w2=w2 * 1j)

    def test_raises_on_negative_weights(self, uvd_zen_2458042_xx: UVData) -> None:
        """Check that a negative weight matrix raises a ValueError."""
        d1 = uvd_zen_2458042_xx.get_data(24, 25)
        w1 = -(~uvd_zen_2458042_xx.get_flags(24, 25)).astype(float)
        with pytest.raises(ValueError, match="Weight matrices must be positive"):
            utils.cov(d1, w1)


class TestLoadConfig:
    def test_parses_expected_structure(self) -> None:
        """Check that load_config reads keys, bools, lists, Nones, and list-of-lists-as-tuples correctly."""
        fname = DATA_PATH / "_test_utils.yaml"
        cfg = utils.load_config(fname)

        # Check that expected keys exist
        assert "data" in cfg.keys()
        assert "pspec" in cfg.keys()
        # Check that boolean values are read in correctly
        assert cfg["pspec"]["overwrite"]
        # Check that lists are read in as lists
        assert len(cfg["data"]["subdirs"]) == 1
        # Check 'None' and list of lists become Nones and list of tuples
        assert cfg["data"]["pairs"] == [("xx", "xx"), ("yy", "yy")]
        assert cfg["pspec"]["taper"] == "none"
        assert cfg["pspec"]["groupname"] is None
        assert cfg["pspec"]["options"]["bar"] == [("foo", "bar")]
        assert cfg["pspec"]["options"]["foo"] is None

    def test_raises_on_missing_file(self) -> None:
        """Check that a missing config file raises an IOError."""
        with pytest.raises(IOError, match="No such file or directory"):
            utils.load_config("file_that_doesnt_exist")


class TestSpwRange:
    @pytest.mark.parametrize(
        "func,kwarg,range_value",
        [
            (utils.spw_range_from_freqs, "freq_range", (100e6, 110e6)),
            (utils.spw_range_from_redshifts, "z_range", (9.7, 12.1)),
        ],
        ids=["freqs", "redshifts"],
    )
    def test_raises_on_object_without_freq_array(
        self, func: Callable, kwarg: str, range_value: tuple[float, float]
    ) -> None:
        """Check that an object lacking a freq_array attribute raises an AttributeError, for both freq- and redshift-range lookups."""
        with pytest.raises(
            AttributeError, match="does not have a freq_array attribute"
        ):
            func(np.arange(3), **{kwarg: range_value})

    @pytest.mark.parametrize("obj_name", ["uvd_zen_2458042_xx", "vanilla_uvp"])
    @pytest.mark.parametrize(
        "func,kwarg,range_value,match",
        [
            (
                utils.spw_range_from_freqs,
                "freq_range",
                (98e6, 110e6),
                "Lower bound of spectral window is below",
            ),
            (
                utils.spw_range_from_freqs,
                "freq_range",
                (190e6, 202e6),
                "Upper bound of spectral window is above",
            ),
            (
                utils.spw_range_from_freqs,
                "freq_range",
                (190e6, 180e6),
                "Upper bound of spectral window is less than the lower bound",
            ),
            (
                utils.spw_range_from_redshifts,
                "z_range",
                (10.0, 20.0),
                "Lower bound of spectral window is below",
            ),
            (
                utils.spw_range_from_redshifts,
                "z_range",
                (5.0, 8.0),
                "Upper bound of spectral window is above",
            ),
            (
                utils.spw_range_from_redshifts,
                "z_range",
                (11.0, 10.0),
                "Upper bound of spectral window is less than the lower bound",
            ),
        ],
        ids=[
            "freqs_lower",
            "freqs_upper",
            "freqs_order",
            "redshifts_lower",
            "redshifts_upper",
            "redshifts_order",
        ],
    )
    def test_raises_on_invalid_range(
        self,
        request: pytest.FixtureRequest,
        obj_name: str,
        func: Callable,
        kwarg: str,
        range_value: tuple[float, float],
        match: str,
    ) -> None:
        """Check that out-of-bounds or inverted ranges raise descriptive ValueErrors, for both freq/redshift lookups and UVData/UVPSpec inputs."""
        obj = request.getfixturevalue(obj_name)
        with pytest.raises(ValueError, match=match):
            func(obj, **{kwarg: range_value})

    @pytest.mark.parametrize(
        "func,kwarg,range_value,range_list,bounds_false_value,equiv_value",
        [
            (
                utils.spw_range_from_freqs,
                "freq_range",
                (110e6, 130e6),
                [(100e6, 120e6), (120e6, 140e6), (140e6, 160e6)],
                (98e6, 120e6),
                (100e6, 120e6),
            ),
            (
                utils.spw_range_from_redshifts,
                "z_range",
                (7.0, 8.0),
                [(6.5, 7.5), (7.5, 8.5), (8.5, 9.5)],
                (12.0, 14.0),
                (6.2, 7.2),
            ),
        ],
        ids=["freqs", "redshifts"],
    )
    def test_valid_range_returns_correct_types(
        self,
        uvd_zen_2458042_xx: UVData,
        func: Callable,
        kwarg: str,
        range_value: tuple[float, float],
        range_list: list[tuple[float, float]],
        bounds_false_value: tuple[float, float],
        equiv_value: tuple[float, float],
    ) -> None:
        """Check that tuple vs. list range arguments return the right output types, and bounds_error=False matches the equivalent in-bounds call."""
        spw1 = func(uvd_zen_2458042_xx, **{kwarg: range_value})
        spw2 = func(uvd_zen_2458042_xx, **{kwarg: range_list})
        spw3 = func(uvd_zen_2458042_xx, **{kwarg: bounds_false_value}, bounds_error=False)
        spw4 = func(uvd_zen_2458042_xx, **{kwarg: equiv_value})

        # Make sure tuple vs. list arguments were handled correctly
        assert isinstance(spw1, tuple)
        assert isinstance(spw2, list)
        assert len(spw2) == len(range_list)
        # Make sure that bounds_error=False works
        if kwarg == "freq_range":
            assert spw3 == spw4

    @pytest.mark.parametrize(
        "func,kwarg,range_value",
        [
            (utils.spw_range_from_freqs, "freq_range", (100.1e6, 100.74e6)),
            (utils.spw_range_from_redshifts, "z_range", (13.1, 13.2)),
        ],
        ids=["freqs", "redshifts"],
    )
    def test_works_for_uvpspec_input(
        self,
        vanilla_uvp: UVPSpec,
        func: Callable,
        kwarg: str,
        range_value: tuple[float, float],
    ) -> None:
        """Check that both spw_range_from_freqs and spw_range_from_redshifts accept a UVPSpec object."""
        spw5 = func(vanilla_uvp, **{kwarg: range_value})
        assert isinstance(spw5, tuple)
        assert spw5 == (1, 7)


class TestCalcBlpairReds:
    def test_basic_execution(self, uvd_zen_all_xx: UVData) -> None:
        """Check basic baseline-pair-redundancy calculation, including grouping/length/angle bookkeeping."""
        (bls1, bls2, blps, xants1, xants2, rgrps, lens, angs) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            extra_info=True,
            exclude_auto_bls=False,
            exclude_permutations=True,
        )
        assert len(bls1) == len(bls2) == 15
        assert blps == list(zip(bls1, bls2))
        assert xants1 == xants2
        assert len(xants1) == 42
        assert len(rgrps) == len(bls1)  # assert rgrps matches bls1 shape
        assert np.max(rgrps) == len(lens) - 1  # assert rgrps indexes lens / angs

    def test_xant_flag_thresh(self, uvd_zen_all_xx: UVData) -> None:
        """Check that a zero xant_flag_thresh excludes every antenna."""
        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            exclude_auto_bls=True,
            exclude_permutations=True,
            xant_flag_thresh=0.0,
        )
        assert len(bls1) == len(bls2) == 0

    def test_bl_len_range(self, uvd_zen_all_xx: UVData) -> None:
        """Check that bl_len_range filters baselines, with and without excluding auto-baselines."""
        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            exclude_auto_bls=False,
            exclude_permutations=True,
            bl_len_range=(0, 15.0),
        )
        assert len(bls1) == len(bls2) == 12

        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            exclude_auto_bls=True,
            exclude_permutations=True,
            bl_len_range=(0, 15.0),
        )
        assert len(bls1) == len(bls2) == 5
        assert np.all([bls1[i] != bls2[i] for i in range(len(blps))])

    def test_grouping(self, uvd_zen_all_xx: UVData) -> None:
        """Check that Nblps_per_group batches blpairs into lists of the requested size."""
        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            exclude_auto_bls=False,
            exclude_permutations=True,
            Nblps_per_group=2,
        )
        assert len(blps) == 10
        assert isinstance(blps[0], list)
        assert blps[0] == [((24, 37), (25, 38)), ((24, 37), (24, 37))]

    def test_baseline_select_on_input_uvd(self, uvd_zen_all_xx: UVData) -> None:
        """Check that pre-selecting baselines on the input UVData restricts the output blpairs accordingly."""
        uvd2 = copy.deepcopy(uvd_zen_all_xx)
        uvd2.select(bls=[(24, 25), (37, 38), (24, 39)])
        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd2,
            uvd2,
            filter_blpairs=True,
            exclude_auto_bls=True,
            exclude_permutations=True,
            bl_len_range=(10.0, 20.0),
        )
        assert blps == [((24, 25), (37, 38))]

    def test_exclude_cross_bls(self, uvd_zen_all_xx: UVData) -> None:
        """Check that exclude_cross_bls keeps only auto-baseline pairs."""
        (bls1, bls2, blps, xants1, xants2) = utils.calc_blpair_reds(
            uvd_zen_all_xx, uvd_zen_all_xx, filter_blpairs=True, exclude_cross_bls=True
        )
        for bl1, bl2 in blps:
            assert bl1 == bl2

    def test_raises_on_mismatched_antenna_positions(
        self, uvd_zen_all_xx: UVData
    ) -> None:
        """Check that differing antenna positions between uvd1 and uvd2 raise an AssertionError."""
        uvd2 = copy.deepcopy(uvd_zen_all_xx)
        uvd2.telescope.antenna_positions[0] += 2
        with pytest.raises(
            AssertionError, match="antenna positions from uvd1 and uvd2 do not agree"
        ):
            utils.calc_blpair_reds(uvd_zen_all_xx, uvd2)

    def test_raises_on_exclude_both_auto_and_cross(
        self, uvd_zen_all_xx: UVData
    ) -> None:
        """Check that excluding both auto and cross blpairs simultaneously raises an AssertionError."""
        with pytest.raises(
            AssertionError, match="Can't exclude both auto and cross blpairs"
        ):
            utils.calc_blpair_reds(
                uvd_zen_all_xx,
                uvd_zen_all_xx,
                exclude_auto_bls=True,
                exclude_cross_bls=True,
            )

    def test_autos_only(self, uvd_zen_all_xx: UVData) -> None:
        """Check that include_crosscorrs=False, include_autocorrs=True restricts output to auto-baselines."""
        (bls1, bls2, blps, xants1, xants2, rgrps, lens, angs) = utils.calc_blpair_reds(
            uvd_zen_all_xx,
            uvd_zen_all_xx,
            filter_blpairs=True,
            extra_info=True,
            exclude_auto_bls=False,
            exclude_permutations=True,
            include_crosscorrs=False,
            include_autocorrs=True,
        )
        assert len(bls1) > 0
        for bl1, bl2 in zip(bls1, bls2):
            assert bl1[0] == bl1[1]
            assert bl2[0] == bl2[1]


@pytest.mark.parametrize("n_dlys", [None, 30])
def test_get_delays(n_dlys: int | None) -> None:
    """Check that get_delays returns a sorted array of delays with the requested (or default) length."""
    freqs = np.linspace(100.0, 200.0, 50) * 1e6
    delays = utils.get_delays(freqs, n_dlys=n_dlys)
    assert delays.size == (n_dlys if n_dlys is not None else freqs.size)
    assert np.isclose(delays[0], -2.450e-07, atol=1e-10)


class TestGetReds:
    def test_basic_execution(self) -> None:
        """Check that excluded antennas are absent from every redundant group, and group counts match."""
        xants = [0, 1, 2]
        r, l, a = utils.get_reds(ZEN_ALL_XX_PATH, xants=xants)
        assert np.all(
            [
                np.all([bl[0] not in xants and bl[1] not in xants for bl in _r])
                for _r in r
            ]
        )
        assert len(r) == len(a) == len(l)
        assert len(r) == 104

    @pytest.mark.parametrize("input_type", ["uvdata", "antpos_dict"])
    def test_input_types_are_equivalent(
        self, uvd_zen_all_xx_meta: UVData, input_type: str
    ) -> None:
        """Check that UVData and antenna-position-dict inputs give the same redundant groups as a filename."""
        xants = [0, 1, 2]
        r, l, a = utils.get_reds(ZEN_ALL_XX_PATH, xants=xants)

        if input_type == "uvdata":
            other = uvd_zen_all_xx_meta
        else:
            antpos = uvd_zen_all_xx_meta.telescope.get_enu_antpos()
            other = dict(zip(uvd_zen_all_xx_meta.telescope.antenna_numbers, antpos))

        r2, l2, a2 = utils.get_reds(other, xants=xants)
        for _r1, _r2 in zip(r, r2):
            np.testing.assert_array_equal(_r1, _r2)

    def test_bl_len_and_deg_range(self, uvd_zen_all_xx_meta: UVData) -> None:
        """Check that bl_len_range/bl_deg_range restrict the returned lengths/angles accordingly."""
        bl_len_range = (14, 16)
        bl_deg_range = (55, 65)
        r, l, a = utils.get_reds(
            uvd_zen_all_xx_meta, bl_len_range=bl_len_range, bl_deg_range=bl_deg_range
        )
        assert np.all([_l > bl_len_range[0] and _l < bl_len_range[1] for _l in l])
        assert np.all([_a > bl_deg_range[0] and _a < bl_deg_range[1] for _a in a])

    def test_min_ew_cut(self, uvd_zen_all_xx_meta: UVData) -> None:
        """Check that min_EW_cut restricts to baselines along (or near) the EW axis."""
        r, l, a = utils.get_reds(
            uvd_zen_all_xx_meta, bl_len_range=(14, 16), min_EW_cut=14
        )
        assert len(l) == len(a) == 1
        assert np.isclose(a[0] % 180, 0, atol=1)

    def test_add_autos(self) -> None:
        """Check that add_autos prepends a zero-length, zero-angle auto-correlation group."""
        xants = [0, 1, 2]
        r, l, a = utils.get_reds(ZEN_ALL_XX_PATH, xants=xants, add_autos=True)
        np.testing.assert_almost_equal(l[0], 0)
        np.testing.assert_almost_equal(a[0], 0)
        assert len(r) == 105

    def test_raises_on_invalid_uvd_type(self) -> None:
        """Check that a non-UVData, non-path, non-dict input raises a TypeError."""
        with pytest.raises(TypeError, match="uvd must be a UVData object"):
            utils.get_reds([1.0, 2.0])

    def test_autos_only(self) -> None:
        """Check that autos_only restricts every returned group to auto-correlation baselines."""
        xants = [0, 1, 2]
        r, l, a = utils.get_reds(
            ZEN_ALL_XX_PATH, xants=xants, autos_only=True, add_autos=True
        )
        assert len(r) == 1
        for bl in r[0]:
            assert bl[0] == bl[1]


class TestConfigPspecBlpairs:
    UV_TEMPLATE = str(DATA_PATH / "zen.{group}.{pol}.LST.1.28828.uvOCRSA")

    def test_basic_execution(self) -> None:
        """Check that a single (group, pol) pairing returns the expected blpair count."""
        groupings = utils.config_pspec_blpairs(
            self.UV_TEMPLATE,
            [("xx", "xx")],
            [("even", "odd")],
            verbose=False,
            exclude_auto_bls=True,
        )
        assert len(groupings) == 1
        assert list(groupings.keys())[0] == (("even", "odd"), ("xx", "xx"))
        assert len(list(groupings.values())[0]) == 11833

    def test_drops_pairs_with_missing_files(self) -> None:
        """Check that requesting a (group, pol) combination with no matching files drops it from the result."""
        groupings = utils.config_pspec_blpairs(
            self.UV_TEMPLATE,
            [("xx", "xx"), ("yy", "yy")],
            [("even", "odd"), ("even", "odd")],
            verbose=False,
            exclude_auto_bls=True,
        )
        assert len(groupings) == 1
        assert list(groupings.keys())[0] == (("even", "odd"), ("xx", "xx"))

    def test_xants(self) -> None:
        """Check that excluding antennas reduces the blpair count."""
        groupings = utils.config_pspec_blpairs(
            self.UV_TEMPLATE,
            [("xx", "xx")],
            [("even", "odd")],
            xants=[0, 1, 2],
            verbose=False,
            exclude_auto_bls=True,
        )
        assert len(list(groupings.values())[0]) == 9735

    def test_exclude_patterns(self) -> None:
        """Check that exclude_patterns filters out all matching files, leaving an empty result."""
        groupings = utils.config_pspec_blpairs(
            self.UV_TEMPLATE,
            [("xx", "xx"), ("yy", "yy")],
            [("even", "odd"), ("even", "odd")],
            exclude_patterns=["1.288"],
            verbose=False,
            exclude_auto_bls=True,
        )
        assert len(groupings) == 0

    def test_raises_on_mismatched_pol_and_group_lengths(self) -> None:
        """Check that mismatched pol-pair and group-pair list lengths raise an AssertionError."""
        with pytest.raises(AssertionError, match="must equal len"):
            utils.config_pspec_blpairs(
                self.UV_TEMPLATE,
                [("xx", "xx"), ("xx", "xx")],
                [("even", "odd")],
                verbose=False,
            )


class TestUvdToTsys:
    def test_equivalent_beam_inputs(
        self, uvd_zen_2458042_xx: UVData, beam_nf_dipole: PSpecBeamUV
    ) -> None:
        """Check that PSpecBeamBase, beamfits-path, and UVPSpec-with-beam inputs give equivalent Tsys estimates."""
        tsys_estimate = utils.uvd_to_Tsys(uvd_zen_2458042_xx, beam_nf_dipole)
        tsys_estimate2 = utils.uvd_to_Tsys(
            uvd_zen_2458042_xx, str(DATA_PATH / "HERA_NF_dipole_power.beamfits")
        )
        assert np.allclose(tsys_estimate.data_array, tsys_estimate2.data_array)

        uvp2, _ = testing.build_vanilla_uvpspec(beam=beam_nf_dipole)
        tsys_estimate3 = utils.uvd_to_Tsys(uvd_zen_2458042_xx, uvp2)
        assert np.allclose(tsys_estimate.data_array, tsys_estimate3.data_array)

    def test_raises_on_uvpspec_without_beam(
        self, uvd_zen_2458042_xx: UVData, vanilla_uvp: UVPSpec
    ) -> None:
        """Check that a UVPSpec without OmegaP/OmegaPP raises a ValueError."""
        with pytest.raises(
            ValueError, match="UVPSpec must have OmegaP and OmegaPP to make a beam"
        ):
            utils.uvd_to_Tsys(uvd_zen_2458042_xx, vanilla_uvp)

    def test_raises_on_invalid_beam_type(self, uvd_zen_2458042_xx: UVData) -> None:
        """Check that a beam argument of the wrong type raises a ValueError."""
        with pytest.raises(
            ValueError, match="beam must be a string, PSpecBeamBase subclass"
        ):
            utils.uvd_to_Tsys(uvd_zen_2458042_xx, 12.0)


class TestLog:
    def test_prints_message(self) -> None:
        """Check that log() runs without raising for both default and explicit verbosity levels."""
        utils.log("message")
        utils.log("message", lvl=2)

    def test_writes_to_logfile(self, tmp_path: Path) -> None:
        """Check that log() writes the message verbatim to a given file handle."""
        logf_path = tmp_path / "logf.log"
        with open(logf_path, "w") as logf:
            utils.log("message", f=logf, verbose=False)
        with open(logf_path) as f:
            assert f.readlines()[0] == "message"

    def test_writes_traceback_to_logfile(self, tmp_path: Path) -> None:
        """Check that log() includes the exception traceback when tb= is passed."""
        logf_path = tmp_path / "logf.log"
        with open(logf_path, "w") as logf:
            try:
                raise NameError
            except NameError:
                utils.log(
                    "raised an exception", f=logf, tb=sys.exc_info(), verbose=False
                )
        with open(logf_path) as f:
            log = "".join(f.readlines())
        assert "NameError" in log and "raised an exception" in log


@pytest.fixture(scope="module")
def blvec_reds(uvd_zen_2458042_xx: UVData) -> list:
    """Redundant-baseline groups (by antenna position) for zen.2458042.17772.xx.HH.uvXA."""
    antpos, ants = uvd_zen_2458042_xx.get_enu_data_ants()
    return redcal.get_pos_reds(dict(zip(ants, antpos)))


@pytest.fixture(scope="module")
def uvp_two_red_grps(blvec_reds: list) -> UVPSpec:
    """UVPSpec built from the first two redundant baseline groups, for get_blvec_reds tests."""
    return testing.uvpspec_from_data(
        ZEN_2458042_XX_PATH, blvec_reds[:2], spw_ranges=[(10, 40)]
    )


class TestGetBlvecReds:
    def test_with_dict_input(self, uvp_two_red_grps: UVPSpec) -> None:
        """Check that get_blvec_reds groups a dict of baseline vectors into the expected redundant tags."""
        blvecs = dict(
            zip(uvp_two_red_grps.bl_array, uvp_two_red_grps.get_ENU_bl_vecs())
        )
        red_bl_grp, red_bl_len, red_bl_ang, red_bl_tag = utils.get_blvec_reds(
            blvecs, bl_error_tol=1.0
        )
        assert len(red_bl_grp) == 2
        assert red_bl_tag == ["015_060", "015_120"]

    def test_with_uvpspec_input(self, uvp_two_red_grps: UVPSpec) -> None:
        """Check that get_blvec_reds gives the same grouping when fed a UVPSpec directly."""
        red_bl_grp, red_bl_len, red_bl_ang, red_bl_tag = utils.get_blvec_reds(
            uvp_two_red_grps, bl_error_tol=1.0
        )
        assert len(red_bl_grp) == 2
        assert red_bl_tag == ["015_060", "015_120"]

    def test_zero_tolerance_separates_every_baseline(
        self, uvp_two_red_grps: UVPSpec
    ) -> None:
        """Check that bl_error_tol=0.0 puts every blpair into its own group."""
        red_bl_grp, red_bl_len, red_bl_ang, red_bl_tag = utils.get_blvec_reds(
            uvp_two_red_grps, bl_error_tol=0.0
        )
        assert len(red_bl_grp) == uvp_two_red_grps.Nblpairs

    def test_match_bl_lens_combines_angles(self, blvec_reds: list) -> None:
        """Check that match_bl_lens=True combines same-length groups across different angles."""
        uvp = testing.uvpspec_from_data(
            ZEN_2458042_XX_PATH, blvec_reds[:3], spw_ranges=[(10, 40)]
        )
        red_bl_grp, red_bl_len, red_bl_ang, red_bl_tag = utils.get_blvec_reds(
            uvp, bl_error_tol=1.0, match_bl_lens=True
        )
        assert len(red_bl_grp) == 1


def test_uvp_noise_error_parser() -> None:
    """Check that the noise-error argparser parses container/auto-file/beam/groups arguments."""
    ap = utils.uvp_noise_error_parser()
    args = ap.parse_args(
        ["container.hdf5", "autos.uvh5", "beam.uvbeam", "--groups", "dset0_dset1"]
    )
    assert args.pspec_container == "container.hdf5"
    assert args.auto_file == "autos.uvh5"
    assert args.beam == "beam.uvbeam"
    assert args.groups == ["dset0_dset1"]
    assert args.spectra is None


@pytest.fixture
def job_monitor_datafiles(tmp_path: Path) -> list[str]:
    """Four empty files for job_monitor's run_func to write into."""
    datafiles = [str(tmp_path / name) for name in ["a", "b", "c", "d"]]
    for df in datafiles:
        open(df, "w").close()
    return datafiles


def _make_run_func(datafiles: list[str]) -> Callable[[int], int]:
    """A run_func that writes to datafiles[i], failing ~30% of the time (raises ValueError above rand=0.7)."""

    def run_func(i: int) -> int:
        try:
            rand_num = np.random.rand(1)[0]
            if rand_num > 0.7:
                raise ValueError
            with open(datafiles[i], "a") as f:
                f.write("Hello World")
        except ValueError:
            return 1
        return 0

    return run_func


class TestJobMonitor:
    def test_records_failures_without_rerun(
        self, job_monitor_datafiles: list[str]
    ) -> None:
        """Check that a single pass (maxiter=1) records the seeded random failure."""
        np.random.seed(0)
        # run over datafiles
        run_func = _make_run_func(job_monitor_datafiles)
        failures = utils.job_monitor(
            run_func,
            range(len(job_monitor_datafiles)),
            "test",
            maxiter=1,
            verbose=False,
        )
        # assert job 1 failed
        np.testing.assert_array_equal(failures, np.array([1]))

    def test_reruns_clear_failures(self, job_monitor_datafiles: list[str]) -> None:
        """Check that allowing reruns (maxiter=10) eventually clears all failures."""
        np.random.seed(0)
        run_func = _make_run_func(job_monitor_datafiles)
        failures = utils.job_monitor(
            run_func,
            range(len(job_monitor_datafiles)),
            "test",
            maxiter=10,
            verbose=False,
        )
        # assert no failures now
        assert len(failures) == 0

import copy
from pathlib import Path

import numpy as np
import pytest
from hera_cal import redcal
from pytest_cases import parametrize_with_cases
from pyuvdata import UVData

from hera_pspec import (
    PSpecBeamUV,
    UVPSpec,
    container,
    conversions,
    grouping,
    pspecdata,
    testing,
    utils,
    uvpspec,
    uvwindow,
)
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


def case_vanilla_uvp(vanilla_uvp: UVPSpec) -> UVPSpec:
    return vanilla_uvp


def case_vanilla_uvp_with_beam(vanilla_uvp_with_beam: UVPSpec) -> UVPSpec:
    return vanilla_uvp_with_beam


def case_vanilla_uvp_alternating_times(
    vanilla_uvp_alternating_times: UVPSpec,
) -> UVPSpec:
    return vanilla_uvp_alternating_times


# def case_uvp_exact_wfs(uvp_exact_wfs: UVPSpec):
#     return uvp_exact_wfs


@pytest.fixture(scope="session")
def redundant_blpairs(uvd_zen_even_xx: UVData) -> list:
    """Redundant baseline groups from zen.even.xx.LST.1.28828.uvOCRSA."""
    ap, a = uvd_zen_even_xx.get_enu_data_ants()
    return redcal.get_pos_reds(dict(zip(a, ap)), bl_error_tol=1.0)


@pytest.fixture(scope="session")
def uvp_from_miriad(
    redundant_blpairs: list,
    beam_nf_dipole_wcosmo: PSpecBeamUV,
    cosmo: conversions.Cosmo_Conversions,
    uvd_zen_even_xx: UVData,
) -> UVPSpec:
    """UVPSpec from zen.even.xx.LST.1.28828.uvOCRSA, first 3 redundant groups."""
    return testing.uvpspec_from_data(
        uvd_zen_even_xx,
        redundant_blpairs[:3],
        spw_ranges=[(50, 100)],
        beam=beam_nf_dipole_wcosmo,
        cosmo=cosmo,
    )


@pytest.fixture
def delay_bins_uvp_nocov(vanilla_uvp_with_beam: UVPSpec) -> UVPSpec:
    """Copy of vanilla_uvp_with_beam with cov arrays removed."""
    uvp = copy.deepcopy(vanilla_uvp_with_beam)
    del uvp.cov_array_real
    del uvp.cov_array_imag
    return uvp


@pytest.fixture
def delay_bins_exact_wf_uvp(vanilla_uvp_with_beam: UVPSpec) -> UVPSpec:
    """Copy of vanilla_uvp_with_beam with exact window functions."""
    uvp = copy.deepcopy(vanilla_uvp_with_beam)
    gaussian_beam = uvwindow.FTBeam.gaussian(
        freq_array=uvp.freq_array, widths=8.0, pol=1
    )
    del uvp.window_function_array
    uvp.get_exact_window_functions(ftbeam=gaussian_beam, inplace=True)
    return uvp


def test_grouping_input_validation(vanilla_uvp: UVPSpec) -> None:
    """Check that invalid arguments to average_spectra, spherical_average, spherical_wf_from_uvp, and fold_spectra raise appropriate errors."""
    wrong_uvp = pspecdata.PSpecData(dsets=[], wgts=[])
    uvp_with_stats = copy.deepcopy(vanilla_uvp)
    uvp_with_stats.stats_array = {
        "missing": {
            spw: np.ones(
                (uvp_with_stats.Nbltpairs, uvp_with_stats.Ndlys, uvp_with_stats.Npols)
            )
            for spw in uvp_with_stats.spw_array
        }
    }

    with pytest.raises(TypeError, match="uvp_in must be a UVPSpec object"):
        grouping.average_spectra(wrong_uvp, inplace=False)

    with pytest.raises(
        TypeError, match="blpair_groups must be a sequence of baseline-pair groups"
    ):
        grouping.average_spectra(vanilla_uvp, blpair_groups="bad", inplace=False)

    with pytest.raises(
        TypeError, match="blpair_groups must be a sequence of baseline-pair groups"
    ):
        grouping.average_spectra(
            vanilla_uvp, blpair_groups=[101102101102], inplace=False
        )

    with pytest.raises(ValueError, match="blpair_groups cannot contain empty groups"):
        grouping.average_spectra(vanilla_uvp, blpair_groups=[[]], inplace=False)

    with pytest.raises(
        TypeError, match="error_field must be a string or a sequence of strings"
    ):
        grouping.average_spectra(vanilla_uvp, error_field=1, inplace=False)

    with pytest.raises(
        TypeError, match="error_field must be a string or a sequence of strings"
    ):
        grouping.average_spectra(vanilla_uvp, error_field=[1], inplace=False)

    with pytest.raises(TypeError, match="error_weights must be a string or None"):
        grouping.average_spectra(vanilla_uvp, error_weights=1, inplace=False)

    grouping.average_spectra(uvp_with_stats, error_field=["missing"], inplace=False)

    with pytest.raises(ValueError, match="blpair_weights must have the same shape"):
        grouping.average_spectra(
            vanilla_uvp,
            blpair_groups=[[101102101102, 102103102103]],
            blpair_weights=[[1.0]],
            inplace=False,
        )

    with pytest.raises(
        ValueError, match="Cannot specify blpair_weights if blpair_groups is None"
    ):
        grouping.average_spectra(vanilla_uvp, blpair_weights=[[1.0]], inplace=False)

    with pytest.raises(TypeError, match="uvp_in must be a UVPSpec object"):
        grouping.spherical_average(wrong_uvp, np.array([0.1, 0.2]), 0.1)

    with pytest.raises(TypeError, match="A must be a mutable mapping"):
        grouping.spherical_average(vanilla_uvp, np.array([0.1, 0.2]), 0.1, A=[])

    grouping.spherical_average(
        vanilla_uvp, np.array([0.1, 0.2]), np.array([0.05, 0.05])
    )

    with pytest.raises(TypeError, match="bin_widths must be numeric and array-like"):
        grouping.spherical_average(vanilla_uvp, np.array([0.1, 0.2]), "bad")

    with pytest.raises(ValueError, match="bin_widths must be one-dimensional"):
        grouping.spherical_average(
            vanilla_uvp, np.array([0.1, 0.2]), np.array([[0.05, 0.05]])
        )

    with pytest.raises(TypeError, match="uvp_in must be a UVPSpec object"):
        grouping.spherical_wf_from_uvp(wrong_uvp, np.array([0.1, 0.2]))

    with pytest.raises(
        TypeError, match="kbin_edges_theory must be numeric and array-like"
    ):
        grouping.spherical_wf_from_uvp(
            vanilla_uvp, np.array([0.1, 0.2]), kbin_edges_theory="bad"
        )

    with pytest.raises(TypeError, match="uvp must be a UVPSpec object"):
        grouping.fold_spectra("not-a-uvp")


class TestGroupBaselines:
    @pytest.mark.parametrize("n,ngrps", [(1, 2), (2, 5), (5, 6)])
    def test_too_many_groups(self, n: int, ngrps: int) -> None:
        """Check that requesting more groups than baselines raises a ValueError."""
        bls = [(0, i) for i in range(n)]
        with pytest.raises(ValueError, match="Can't have more groups than baselines"):
            grouping.group_baselines(bls, ngrps)

    @pytest.mark.parametrize("n,ngrps,randomize", [
        (5, 2, False), (13, 5, False), (521, 10, False),
        (5, 2, True), (13, 5, True), (521, 10, True),
    ])
    def test_equal_sized_blocks(self, n: int, ngrps: int, randomize: bool) -> None:
        """Check that keep_remainder=False produces groups of equal size."""
        bls = [(0, i) for i in range(n)]
        g = grouping.group_baselines(bls, ngrps, keep_remainder=False, randomize=randomize)
        assert np.unique([len(grp) for grp in g]).size == 1

    @pytest.mark.parametrize("n,ngrp,randomize", [
        (n, ngrp, rand)
        for n in [1, 2, 4, 5, 13, 521]
        for ngrp in [1, 2, 5, 10, 45]
        for rand in [True, False]
        if ngrp <= n
    ])
    def test_preserves_count(self, n: int, ngrp: int, randomize: bool) -> None:
        """Check that keep_remainder=True preserves the total number of baselines across all groups."""
        bls = [(0, i) for i in range(n)]
        g = grouping.group_baselines(bls, ngrp, keep_remainder=True, randomize=randomize)
        assert np.sum([len(_g) for _g in g]) == len(bls)

    def test_random_seed(self) -> None:
        """Check that the same seed produces identical baseline groupings."""
        bls5 = [(0, i) for i in range(13)]
        g1 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        g3 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        for i in range(len(g1)):
            for j in range(len(g1[i])):
                assert g1[i][j] == g3[i][j]


@pytest.fixture(scope="session")
def uvp_with_stats(beam_nf_dipole_wcosmo: PSpecBeamUV) -> tuple[UVPSpec, list]:
    """UVPSpec from zen.all.xx.LST.1.06964.uvA with noise and simple stats."""
    dfile = str(DATA_PATH / "zen.all.xx.LST.1.06964.uvA")
    # Load into UVData objects
    uvd = UVData()
    uvd.read_miriad(dfile)
    # find conversion factor from Jy to mK
    Jy_to_mK = beam_nf_dipole_wcosmo.Jy_to_mK(np.unique(uvd.freq_array), pol="XX")
    uvd.data_array *= Jy_to_mK[None, :, None]
    # slide the time axis of uvd by one integration
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    # Create a new PSpecData object, and don't forget to feed the beam object
    ds = pspecdata.PSpecData(
        dsets=[uvd1, uvd2], wgts=[None, None], beam=beam_nf_dipole_wcosmo
    )
    ds.rephase_to_dset(0)
    # change units of UVData objects
    ds.dsets[0].vis_units = "mK"
    ds.dsets[1].vis_units = "mK"
    baselines = [(24, 25), (37, 38), (38, 39)]
    # calculate all baseline pairs from group
    baselines1, baselines2, blpairs = utils.construct_blpairs(
        baselines, exclude_auto_bls=True, exclude_permutations=True
    )
    uvp = ds.pspec(
        baselines1,
        baselines2,
        (0, 1),
        [("xx", "xx")],
        spw_ranges=[(300, 350)],
        input_data_weight="identity",
        norm="I",
        taper="blackman-harris",
        store_cov=True,
        cov_model="autos",
        verbose=False,
    )
    keys = uvp.get_all_keys()
    # Add the analytic noise to stat_array
    Pn = uvp.generate_noise_spectra(0, "xx", 220)
    for key in keys:
        blp = uvp.antnums_to_blpair(key[1])
        uvp.set_stats("noise", key, Pn[blp])
    # Add the simple error bar (all are set to be one) to stat_array
    errs = np.ones((uvp.Ntpairs, uvp.Ndlys))
    for key in keys:
        uvp.set_stats("simple", key, errs)
    return uvp, blpairs


@pytest.fixture(scope="session")
def uvp_with_exact_wf(beam_nf_dipole_wcosmo: PSpecBeamUV, uvd_zen_2458116: UVData) -> tuple[UVPSpec, list]:
    """UVPSpec from uvd_zen_2458116 with exact window functions."""
    ds = pspecdata.PSpecData(
        dsets=[uvd_zen_2458116, uvd_zen_2458116],
        wgts=[None, None],
        beam=beam_nf_dipole_wcosmo,
    )
    baselines1, baselines2, _ = utils.construct_blpairs(
        uvd_zen_2458116.get_antpairs()[1:],
        exclude_permutations=False,
        exclude_auto_bls=True,
    )
    # compute ps
    uvp = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=(175, 195),
        taper="bh",
        verbose=False,
    )
    # get exact window functions
    uvp.get_exact_window_functions(
        ftbeam=DATA_PATH / "FT_beam_HERA_dipole_test",
        spw_array=None,
        inplace=True,
        verbose=False,
    )
    blpair_groups, _, _ = uvp.get_red_blpairs()
    return uvp, blpair_groups


class TestAverageSpectra:
    def test_uniform_weights_reduce_errors(self, uvp_with_stats: tuple[UVPSpec, list]) -> None:
        """Check that uniform error weights reduce the averaged error bar by 1/√N."""
        uvp, blpairs = uvp_with_stats
        blpair_groups = [blpairs]
        uvp_avg_simple_wgts = grouping.average_spectra(
            uvp,
            blpair_groups=blpair_groups,
            time_avg=True,
            error_weights="simple",
            inplace=False,
        )
        # For using uniform error bars as weights, the error bar on the average
        # is 1/sqrt{N} times the error bar on one single sample.
        averaged_stat = uvp_avg_simple_wgts.stats_array["simple"][0][0, 0, 0]
        initial_stat = (
            uvp.stats_array["simple"][0][0, 0, 0]
            / np.sqrt(uvp.Ntpairs)
            / np.sqrt(len(blpairs))
        )
        assert np.all(np.isclose(initial_stat, averaged_stat))

    def test_nonuniform_weights_reduce_errors(self, uvp_with_stats: tuple[UVPSpec, list]) -> None:
        """Check that noise-based error weights yield a smaller averaged error bar than any single sample."""
        uvp, blpairs = uvp_with_stats
        blpair_groups = [blpairs]
        uvp_avg_ints_wgts = grouping.average_spectra(
            uvp,
            blpair_groups=blpair_groups,
            error_field="noise",
            time_avg=True,
            inplace=False,
        )
        uvp_avg_noise_wgts = grouping.average_spectra(
            uvp,
            time_avg=True,
            blpair_groups=blpair_groups,
            error_weights="noise",
            inplace=False,
        )
        # For non-uniform weights, we test the error bar on the average power
        # spectra should be smaller than one on single sample.
        assert abs(uvp_avg_ints_wgts.stats_array["noise"][0][0, 0, 0]) < abs(
            uvp.stats_array["noise"][0][0, 0, 0]
        )
        assert abs(uvp_avg_noise_wgts.stats_array["noise"][0][0, 0, 0]) < abs(
            uvp.stats_array["noise"][0][0, 0, 0]
        )

    def test_inf_variance_single_blpair(self, uvp_with_stats: tuple[UVPSpec, list]) -> None:
        """Check that a single blpair with infinite variance is ignored and the average matches 1/√(N−1)."""
        # Test stats inf variance for all times, single blpair doesn't result
        # in nans and that the avg effectively ignores its presence: e.g. check
        # it matches initial over sqrt(Nblpairs - 1)
        uvp, blpairs = uvp_with_stats
        blpair_groups = [blpairs]
        uvp_inf_var = copy.deepcopy(uvp)
        initial_stat = uvp.get_stats("simple", (0, blpairs[0], "xx"))
        inf_var_stat = np.ones((uvp_inf_var.Ntpairs, uvp_inf_var.Ndlys)) * np.inf
        uvp_inf_var.set_stats("simple", (0, blpairs[1], "xx"), inf_var_stat)
        uvp_inf_var_avg = uvp_inf_var.average_spectra(
            blpair_groups=blpair_groups, error_weights="simple", inplace=False
        )
        final_stat = uvp_inf_var_avg.get_stats("simple", (0, blpairs[0], "xx"))
        assert np.isclose(final_stat, initial_stat / np.sqrt(len(blpairs) - 1)).all()

    def test_inf_variance_single_time(self, uvp_with_stats: tuple[UVPSpec, list]) -> None:
        """Check that all-infinite variance at one time integration propagates as inf rather than NaN or zero."""
        # Test infinite variance for single time, all blpairs doesn't result in nans
        # and check that averaged stat for that time is inf (not zero)
        uvp, blpairs = uvp_with_stats
        blpair_groups = [blpairs]
        uvp_inf_var = copy.deepcopy(uvp)
        initial_stat = uvp.get_stats("simple", (0, blpairs[0], "xx"))
        inf_var_stat = np.ones((uvp_inf_var.Ntpairs, uvp_inf_var.Ndlys))
        inf_var_stat[0] = np.inf
        for blp in blpairs:
            uvp_inf_var.set_stats("simple", (0, blp, "xx"), inf_var_stat)
        uvp_inf_var_avg = uvp_inf_var.average_spectra(
            blpair_groups=blpair_groups, error_weights="simple", inplace=False
        )
        final_stat = uvp_inf_var_avg.get_stats("simple", (0, blpairs[0], "xx"))
        assert np.isclose(final_stat[1:], initial_stat[1:] / np.sqrt(len(blpairs))).all()
        assert np.all(~np.isfinite(final_stat[0]))

    def test_exact_wf_time_average(self, uvp_with_exact_wf: tuple[UVPSpec, list]) -> None:
        """Check that time-averaging with exact window functions collapses Ntpairs to Nblpairs and preserves the WF array shape."""
        uvp, _ = uvp_with_exact_wf
        # time average
        uvp_time_avg = grouping.average_spectra(
            uvp,
            blpair_groups=None,
            time_avg=True,
            blpair_weights=None,
            error_field=None,
            error_weights=None,
            normalize_weights=True,
            inplace=False,
            add_to_history="",
        )
        assert uvp_time_avg.Nbltpairs == uvp_time_avg.Nblpairs
        assert uvp_time_avg.window_function_array[0].shape[0] == uvp_time_avg.Nbltpairs

    def test_exact_wf_redundant_average(self, uvp_with_exact_wf: tuple[UVPSpec, list]) -> None:
        """Check that redundant-baseline averaging with exact window functions gives Nbltpairs == Ntpairs."""
        uvp, blpair_groups = uvp_with_exact_wf
        # redundant average
        uvp_red_avg = grouping.average_spectra(
            uvp,
            blpair_groups=blpair_groups,
            time_avg=False,
            blpair_weights=None,
            error_field=None,
            error_weights=None,
            normalize_weights=True,
            inplace=False,
            add_to_history="",
        )
        assert uvp_red_avg.Nbltpairs == uvp_red_avg.Ntpairs

    def test_exact_wf_combined_average(self, uvp_with_exact_wf: tuple[UVPSpec, list]) -> None:
        """Check that combined time and redundant averaging with error_field runs without error."""
        uvp, blpair_groups = uvp_with_exact_wf
        uvp = copy.deepcopy(uvp)  # don't mutate the shared session fixture
        # both time+redundant avg + error_weights
        keys = uvp.get_all_keys()
        # Add the analytic noise to stat_array
        Pn = uvp.generate_noise_spectra(0, "xx", 220)
        for key in keys:
            blp = uvp.antnums_to_blpair(key[1])
            error = Pn[blp]
            uvp.set_stats("noise", key, error)
        _ = grouping.average_spectra(
            uvp,
            blpair_groups=blpair_groups,
            time_avg=True,
            blpair_weights=None,
            error_field="noise",
            error_weights=None,
            normalize_weights=True,
            inplace=False,
            add_to_history="",
        )


class TestSampleBaselines:
    @pytest.mark.parametrize("n", [1, 2, 4, 5, 13, 521])
    def test_length_preserved(self, n: int) -> None:
        """Check that sample_baselines returns a list of the same length as its input."""
        bls = [(0, i) for i in range(n)]
        samp = grouping.sample_baselines(bls)
        assert len(bls) == len(samp)

    def test_groups(self) -> None:
        """Check that sample_baselines preserves the number of groups when given a grouped baseline list."""
        bls5 = [(0, i) for i in range(13)]
        g1 = grouping.group_baselines(bls5, 3, randomize=False)
        samp = grouping.sample_baselines(g1)
        assert len(g1) == len(samp)


@parametrize_with_cases("uvp", cases=".")
def test_bootstrap_average_blpairs(uvp: UVPSpec) -> None:
    """Check that bootstrap_average_blpairs correctly resamples blpairs and produces consistent averages for identical data."""
    # Check that basic bootstrap averaging works
    blpair_groups = [list(np.unique(uvp.blpair_array))]
    uvp1, wgts = grouping.bootstrap_average_blpairs(
        [uvp], blpair_groups, time_avg=False
    )
    uvp2, wgts = grouping.bootstrap_average_blpairs([uvp], blpair_groups, time_avg=True)
    assert uvp1[0].Nblpairs == 1
    assert uvp1[0].Ntpairs == uvp.Ntpairs
    assert uvp2[0].Ntpairs == 1

    # Total of weights assigned should equal total no. of blpairs
    assert np.sum(wgts) == np.array(blpair_groups).size

    # Check that exceptions are raised when inputs are invalid
    with pytest.raises(
        AssertionError, match="uvp_list must be a list of UVPSpec objects"
    ):
        grouping.bootstrap_average_blpairs(
            [np.arange(5)], blpair_groups, time_avg=False
        )
    with pytest.raises(
        KeyError, match="do not exist in any of the input UVPSpec objects"
    ):
        grouping.bootstrap_average_blpairs([uvp], [[200200200200]], time_avg=False)

    # Reduce UVPSpec to only 3 blpairs and set them all to the same values
    _blpairs = list(np.unique(uvp.blpair_array)[:3])
    uvp3 = uvp.select(spws=0, inplace=False, blpairs=_blpairs)

    Nt = uvp3.Ntpairs
    uvp3.data_array[0][Nt : 2 * Nt] = uvp3.data_array[0][:Nt]
    uvp3.data_array[0][2 * Nt :] = uvp3.data_array[0][:Nt]
    uvp3.integration_array[0][Nt : 2 * Nt] = uvp3.integration_array[0][:Nt]
    uvp3.integration_array[0][2 * Nt :] = uvp3.integration_array[0][:Nt]

    # Test that different bootstrap-sampled averages have the same value as
    # the normal average (since the data for all blpairs has been set to
    # the same values for uvp3)
    np.random.seed(10)
    uvp_avg = uvp3.average_spectra(
        blpair_groups=[_blpairs], time_avg=True, inplace=False
    )
    blpair = uvp_avg.blpair_array[0]
    for _ in range(5):
        # Generate multiple samples and make sure that they are all equal
        # to the regular average (for the cloned data in uvp3)
        uvp4, wgts = grouping.bootstrap_average_blpairs(
            [uvp3], blpair_groups=[_blpairs], time_avg=True
        )
        try:
            ps_avg = uvp_avg.get_data((0, blpair, ("xx", "xx")))
        except:
            print(uvp_avg.polpair_array)
            raise
        ps_boot = uvp4[0].get_data((0, blpair, ("xx", "xx")))
        np.testing.assert_array_almost_equal(ps_avg, ps_boot)


class TestBootstrapResampledError:
    def test_basic(self, tmp_path: Path, uvp_from_miriad: UVPSpec) -> None:
        """Check that bootstrap_resampled_error returns the expected number of samples and respects the random seed."""
        # generate a UVPSpec
        uvp = copy.deepcopy(uvp_from_miriad)

        # Lots of this function is already tested by bootstrap_run
        # so only test the stuff not already tested
        uvp_file = str(tmp_path / "uvp.h5")
        uvp.write_hdf5(uvp_file, overwrite=True)
        _, ub, uw = grouping.bootstrap_resampled_error(
            uvp_file, blpair_groups=None, Nsamples=10, seed=0, verbose=False
        )

        # check number of boots
        assert len(ub) == 10

        # check seed has been used properly
        assert uw[0][0][:5] == [1.0, 1.0, 0.0, 2.0, 1.0]
        assert uw[0][1][:5] == [2.0, 1.0, 1.0, 6.0, 1.0]
        assert uw[1][0][:5] == [2.0, 2.0, 1.0, 1.0, 2.0]
        assert uw[1][1][:5] == [1.0, 0.0, 1.0, 1.0, 4.0]

    def test_gaussian_noise(self) -> None:
        """Check that bootstrapped error bars on Gaussian noise power spectra yield z-scores with standard deviation converging to 1."""
        # get simulated noise in K-str
        uvfile = str(DATA_PATH / "zen.even.xx.LST.1.28828.uvOCRSA")
        Tsys = 300.0  # Kelvin

        # generate complex gaussian noise
        seed = 4
        uvd1 = testing.noise_sim(
            uvfile, Tsys, seed=seed, whiten=True, inplace=False, Nextend=0
        )
        seed = 5
        uvd2 = testing.noise_sim(
            uvfile, Tsys, seed=seed, whiten=True, inplace=False, Nextend=0
        )

        # form (auto) baseline-pairs from only 14.6m bls
        reds, _, _ = utils.get_reds(
            uvd1, pick_data_ants=True, bl_len_range=(10, 50), bl_deg_range=(0, 180)
        )
        bls1, bls2 = utils.flatten(reds), utils.flatten(reds)

        # setup PSpecData and form power psectra
        ds = pspecdata.PSpecData(
            dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None]
        )
        uvp = ds.pspec(
            bls1,
            bls2,
            (0, 1),
            [("xx", "xx")],
            input_data_weight="identity",
            norm="I",
            taper="none",
            sampling=False,
            little_h=False,
            spw_ranges=[(0, 50)],
            verbose=False,
        )

        # bootstrap resample
        Nsamples = 100
        seed = 0
        uvp_avg, _, _ = grouping.bootstrap_resampled_error(
            uvp,
            time_avg=False,
            Nsamples=Nsamples,
            seed=seed,
            normal_std=True,
            blpair_groups=[uvp.get_blpairs()],
        )
        # assert z-score has std of ~1.0 along time ax to within 1/sqrt(Nsamples)
        zscr_real = np.std(
            uvp_avg.data_array[0].real / uvp_avg.stats_array["bs_std"][0].real
        )
        zscr_imag = np.std(
            uvp_avg.data_array[0].imag / uvp_avg.stats_array["bs_std"][0].imag
        )
        assert np.abs(1.0 - zscr_real) < 1 / np.sqrt(Nsamples)
        assert np.abs(1.0 - zscr_imag) < 1 / np.sqrt(Nsamples)


class TestBootstrapRun:
    def test_run_output(self, tmp_path: Path, uvp_from_miriad: UVPSpec) -> None:
        """Check that bootstrap_run writes all bootstrap samples, the average, and correct stats arrays to the container."""
        # generate a UVPSpec and container
        uvp = copy.deepcopy(uvp_from_miriad)
        outfile = tmp_path / "ex.h5"
        psc = container.PSpecContainer(outfile, mode="rw", keep_open=False, swmr=False)
        psc.set_pspec("grp1", "uvp", uvp)

        grouping.bootstrap_run(
            psc,
            time_avg=True,
            Nsamples=100,
            seed=0,
            normal_std=True,
            robust_std=True,
            cintervals=[16, 84],
            keep_samples=True,
            bl_error_tol=1.0,
            overwrite=True,
            add_to_history="hello!",
            verbose=False,
        )
        spcs = psc.spectra("grp1")

        # assert all bs samples were written
        assert np.all([f"uvp_bs{i}" in spcs for i in range(100)])

        # assert average was written
        assert "uvp_avg" in spcs and "uvp" in spcs

        # assert average only has one time and 3 blpairs
        uvp_avg = psc.get_pspec("grp1", "uvp_avg")
        assert uvp_avg.Ntpairs == 1
        assert uvp_avg.Nblpairs == 3

        # check avg file history
        assert "hello!" in uvp_avg.history

        # assert original uvp is unchanged
        assert uvp == psc.get_pspec("grp1", "uvp")

        # check stats array
        np.testing.assert_array_equal(
            ["bs_cinterval_16.00", "bs_cinterval_84.00", "bs_robust_std", "bs_std"],
            list(uvp_avg.stats_array.keys()),
        )

        for stat in ["bs_cinterval_16.00", "bs_cinterval_84.00", "bs_robust_std", "bs_std"]:
            assert uvp_avg.get_stats(
                stat, (0, ((37, 38), (38, 39)), ("xx", "xx"))
            ).shape == (1, 50)
            assert not np.any(
                np.isnan(uvp_avg.get_stats(stat, (0, ((37, 38), (38, 39)), ("xx", "xx"))))
            )
            assert (
                uvp_avg.get_stats(stat, (0, ((37, 38), (38, 39)), ("xx", "xx"))).dtype
                == np.complex128
            )

    def test_exceptions(self, tmp_path: Path, uvp_from_miriad: UVPSpec) -> None:
        """Check that bootstrap_run raises appropriate errors for an empty container, bad filename, missing spectra, and SWMR mode."""
        uvp = copy.deepcopy(uvp_from_miriad)
        outfile = tmp_path / "ex.h5"
        psc = container.PSpecContainer(outfile, mode="rw", keep_open=False, swmr=False)

        # test empty groups
        with pytest.raises(AssertionError, match="No groups exist in PSpecContainer"):
            grouping.bootstrap_run(str(outfile))

        # test bad filename
        with pytest.raises(
            AssertionError, match="filename must be a PSpecContainer or filepath"
        ):
            grouping.bootstrap_run(1)

        # test fed spectra doesn't exist
        psc.set_pspec("grp1", "uvp", uvp)
        with pytest.raises(
            AssertionError, match="no specified spectra exist in PSpecContainer"
        ):
            grouping.bootstrap_run(psc, spectra=["grp1/foo"])

        # test assertionerror if SWMR
        psc = container.PSpecContainer(
            tmp_path / "ex2.h5", mode="rw", keep_open=False, swmr=True
        )
        with pytest.raises(AssertionError, match="should not be in SWMR mode"):
            grouping.bootstrap_run(psc, spectra=["grp1/foo"])

    def test_argparser(self) -> None:
        """Check that the bootstrap_run argparser correctly parses spectra, blpair_groups, and cintervals arguments."""
        args = grouping.get_bootstrap_run_argparser()
        a = args.parse_args(
            [
                "fname",
                "--spectra",
                "grp1/uvp1",
                "grp1/uvp2",
                "grp2/uvp1",
                "--blpair_groups",
                "101102103104 101102102103, 102103104105",
                "--time_avg",
                "True",
                "--Nsamples",
                "100",
                "--cintervals",
                "16",
                "84",
            ]
        )
        assert a.spectra == ["grp1/uvp1", "grp1/uvp2", "grp2/uvp1"]
        assert a.blpair_groups == [[101102103104, 101102102103], [102103104105]]
        assert a.cintervals == [16.0, 84.0]


@pytest.fixture(scope="session")
def uvp_spherical(
    redundant_blpairs: list,
    beam_nf_dipole_wcosmo: PSpecBeamUV,
    cosmo: conversions.Cosmo_Conversions,
    uvd_zen_even_xx: UVData,
) -> UVPSpec:
    """Two-polarization UVPSpec with identity cov and unit stats for spherical_average tests."""
    uvd = copy.deepcopy(uvd_zen_even_xx)
    reds = [r[:2] for r in redundant_blpairs]
    uvp = testing.uvpspec_from_data(
        uvd,
        reds,
        spw_ranges=[(50, 75), (100, 125)],
        beam=beam_nf_dipole_wcosmo,
        cosmo=cosmo,
    )
    uvd.polarization_array[0] = -6
    uvp += testing.uvpspec_from_data(
        uvd,
        reds,
        spw_ranges=[(50, 75), (100, 125)],
        beam=beam_nf_dipole_wcosmo,
        cosmo=cosmo,
    )
    uvp.cov_model = "empirical"
    uvp.cov_array_real = {
        s: np.repeat(
            np.repeat(
                np.eye(uvp.Ndlys, dtype=np.float64)[None, :, :, None], uvp.Nbltpairs, 0
            ),
            uvp.Npols,
            -1,
        )
        for s in range(uvp.Nspws)
    }
    uvp.cov_array_imag = {
        s: np.repeat(
            np.repeat(
                np.eye(uvp.Ndlys, dtype=np.float64)[None, :, :, None], uvp.Nbltpairs, 0
            ),
            uvp.Npols,
            -1,
        )
        for s in range(uvp.Nspws)
    }
    uvp.stats_array = {
        "err": {
            s: np.ones((uvp.Nbltpairs, uvp.Ndlys, uvp.Npols), dtype=np.complex128)
            for s in range(uvp.Nspws)
        }
    }
    return uvp


class TestSpherical:
    KBINS = np.arange(0, 2.9, 0.25)
    BIN_WIDTHS = 0.25

    def test_average_metadata(self, uvp_spherical: UVPSpec) -> None:
        """Check that spherical_average output has correct Nblpairs, history, zero kperp, and stats key."""
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, add_to_history="checking 1 2 3"
        )
        assert sph.Nblpairs == 1
        assert "checking 1 2 3" in sph.history
        assert np.isclose(sph.get_blpair_seps(), 0).all()
        assert "err" in sph.stats_array

    def test_average_kbins_and_normalization(self, uvp_spherical: UVPSpec) -> None:
        """Check kbin values, WF normalization, data smell test, errorbar reduction, and array-shape bug checks."""
        Nk = len(self.KBINS)
        A = {}
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, add_to_history="checking 1 2 3", A=A
        )
        for spw in sph.spw_array:
            # binning and normalization
            assert np.isclose(sph.get_kparas(spw), self.KBINS).all()
            assert np.isclose(sph.window_function_array[spw].sum(axis=2), 1).all()
            # basic "averaged data smell test": low k modes > high k modes
            assert np.all(
                sph.data_array[spw][:, 0, :].real / sph.data_array[spw][:, 10, :] > 1e3
            )
            # errorbars are 1/sqrt(N) what they used to be
            assert np.isclose(
                np.sqrt(sph.cov_array_real[spw])[:, range(Nk), range(Nk)],
                1 / np.sqrt(A[spw].sum(axis=1)),
            ).all()
            assert np.isclose(
                sph.stats_array["err"][spw], 1 / np.sqrt(A[spw].sum(axis=1))
            ).all()
        # bug check: time_avg_array was not down-selected to new Nbltpairs
        assert sph.time_avg_array.size == sph.Nbltpairs
        # bug check: cov_array_imag was not updated
        assert sph.cov_array_real[0].shape == sph.cov_array_imag[0].shape

    def test_average_little_h(
        self, uvp_spherical: UVPSpec, cosmo: conversions.Cosmo_Conversions
    ) -> None:
        """Check that little_h=False with scaled kbins gives identical kparas as the default."""
        sph = grouping.spherical_average(uvp_spherical, self.KBINS, self.BIN_WIDTHS)
        sph2 = grouping.spherical_average(
            uvp_spherical, self.KBINS * cosmo.h, self.BIN_WIDTHS * cosmo.h, little_h=False
        )
        for spw in sph.spw_array:
            assert np.isclose(sph.get_kparas(spw), sph2.get_kparas(spw)).all()

    def test_average_time_avg(self, uvp_spherical: UVPSpec) -> None:
        """Check that time_avg=True reduces Ntpairs to 1."""
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, time_avg=True
        )
        assert sph.Ntpairs == 1

    def test_average_error_weights(self, uvp_spherical: UVPSpec) -> None:
        """Check that error_weights='err' produces normalized window functions."""
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, error_weights="err"
        )
        for spw in sph.spw_array:
            assert np.isclose(sph.window_function_array[spw].sum(axis=2), 1).all()

    def test_average_cov_weights(self, uvp_spherical: UVPSpec) -> None:
        """Check that cov-weighted result matches stats-weighted result for identity covariance."""
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, error_weights="err"
        )
        sph2 = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, weight_by_cov=True
        )
        for spw in sph2.spw_array:
            assert np.isclose(sph2.window_function_array[spw].sum(axis=2), 1).all()
            assert np.isclose(sph.data_array[spw], sph2.data_array[spw]).all()

    def test_average_inf_variance(self, uvp_spherical: UVPSpec) -> None:
        """Check that infinite-variance stats zero out low-k modes and leave valid WF normalization at higher k."""
        uvp2 = copy.deepcopy(uvp_spherical)
        uvp2.set_stats_slice("err", 0, 1000, above=False, val=np.inf)
        sph2 = grouping.spherical_average(uvp2, self.KBINS, self.BIN_WIDTHS, error_weights="err")
        # assert low k modes are zeroed!
        assert np.isclose(sph2.data_array[0][:, :3, :], 0).all()
        # assert bins that weren't nulled still have proper window normalization
        for spw in sph2.spw_array:
            assert np.isclose(
                sph2.window_function_array[spw].sum(axis=2)[:, 3:, :], 1
            ).all()
        # assert resultant stats are not nan
        assert (~np.isnan(sph2.stats_array["err"][0])).all()

    def test_average_combine_uvpspec(self, uvp_spherical: UVPSpec) -> None:
        """Check that combine_uvpspec on per-spw spherical averages reproduces the joint result."""
        sph = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, error_weights="err"
        )
        sph_a, sph_b = (
            sph.select(spws=[0], inplace=False),
            sph.select(spws=[1], inplace=False),
        )
        sph_c = uvpspec.combine_uvpspec([sph_a, sph_b], merge_history=False, verbose=False)
        # bug check: in the past, combine after spherical average erroneously changed dly_array
        assert sph == sph_c

    def test_average_inf_in_cov(self, uvp_spherical: UVPSpec) -> None:
        """Check that inserting inf into cov and stats arrays still produces a finite spherical average."""
        uvp2 = copy.deepcopy(uvp_spherical)
        uvp2.cov_array_real[0][0], uvp2.cov_array_imag[0][0] = np.inf, np.inf
        uvp2.stats_array["err"][0][0] = np.inf
        sph = grouping.spherical_average(uvp_spherical, self.KBINS, self.BIN_WIDTHS)
        assert np.isfinite(sph.cov_array_real[0]).all()

    def test_average_overlapping_kbins_error(self, uvp_spherical: UVPSpec) -> None:
        """Check that bin_widths larger than the kbin spacing raises an AssertionError."""
        with pytest.raises(AssertionError, match="kbins must not overlap"):
            grouping.spherical_average(uvp_spherical, self.KBINS, 1.0)

    def test_average_kbins_theory(self, uvp_spherical: UVPSpec) -> None:
        """Check that kbins_theory subsets the window function array to the given theory bins."""
        sph = grouping.spherical_average(uvp_spherical, self.KBINS, self.BIN_WIDTHS)
        sph2 = grouping.spherical_average(
            uvp_spherical, self.KBINS, self.BIN_WIDTHS, kbins_theory=self.KBINS[:4]
        )
        assert np.allclose(
            sph.window_function_array[0][:, :, :4, :], sph2.window_function_array[0]
        )

    def test_average_exact_windows(self, uvp_with_exact_wf: tuple[UVPSpec, list]) -> None:
        """Check that spherical_average runs for a UVPSpec with exact window functions, with and without blpair_groups."""
        uvp, blpair_groups = uvp_with_exact_wf
        grouping.spherical_average(uvp, self.KBINS, self.BIN_WIDTHS)
        grouping.spherical_average(uvp, self.KBINS, self.BIN_WIDTHS, blpair_groups=blpair_groups)

    def test_wf_shape(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that spherical_wf_from_uvp returns an array with the expected (Ntimes, Nk, Nk, Npols) shape."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        Nk = kbin_edges.size - 1
        wf_array = grouping.spherical_wf_from_uvp(
            uvp_exact_wfs, kbin_edges=kbin_edges, little_h="h^-3" in uvp_exact_wfs.norm_units
        )
        assert wf_array[0].shape == (uvp_exact_wfs.Ntimes, Nk, Nk, uvp_exact_wfs.Npols)

    def test_wf_nonsquare(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that kbin_edges_theory produces a non-square (Nk, Nk_in) window function array."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        kbin_edges_theory = np.arange(0.075, 2.9, dk / 2.0)
        Nk = kbin_edges.size - 1
        Nk_in = kbin_edges_theory.size - 1
        wf_array2 = grouping.spherical_wf_from_uvp(
            uvp_exact_wfs,
            kbin_edges=kbin_edges,
            kbin_edges_theory=kbin_edges_theory,
            little_h="h^-3" in uvp_exact_wfs.norm_units,
        )
        assert wf_array2[0].shape == (uvp_exact_wfs.Ntimes, Nk, Nk_in, uvp_exact_wfs.Npols)

    def test_wf_little_h(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that mismatched little_h raises a UserWarning about unit conversion."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        with pytest.warns(UserWarning, match="Changed little_h units"):
            grouping.spherical_wf_from_uvp(
                uvp_exact_wfs, kbin_edges=kbin_edges / uvp_exact_wfs.cosmo.h, little_h=True
            )

    def test_wf_spw_array(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that a valid spw_array argument runs and an invalid one raises an AssertionError."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        grouping.spherical_wf_from_uvp(
            uvp_exact_wfs, kbin_edges, spw_array=0, little_h="h^-3" in uvp_exact_wfs.norm_units
        )
        with pytest.raises(AssertionError, match="input spw is not in UVPSpec.spw_array"):
            grouping.spherical_wf_from_uvp(
                uvp_exact_wfs, kbin_edges, spw_array=2, little_h="h^-3" in uvp_exact_wfs.norm_units
            )

    def test_wf_blpair_groups(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that blpair_groups runs correctly and an un-grouped (flat) input raises a TypeError."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        blpair_groups, blpair_lens, _ = uvp_exact_wfs.get_red_blpairs()
        grouping.spherical_wf_from_uvp(
            uvp_exact_wfs,
            kbin_edges,
            blpair_groups=blpair_groups,
            blpair_lens=blpair_lens,
            little_h="h^-3" in uvp_exact_wfs.norm_units,
        )
        with pytest.raises(TypeError, match="blpair_groups must be a sequence"):
            grouping.spherical_wf_from_uvp(
                uvp_exact_wfs,
                kbin_edges=kbin_edges,
                blpair_groups=blpair_groups[0],
                little_h="h^-3" in uvp_exact_wfs.norm_units,
            )

    def test_wf_inconsistent_blpair_lens(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check warnings and errors for mismatched blpair_groups and blpair_lens combinations."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        blpair_groups, blpair_lens, _ = uvp_exact_wfs.get_red_blpairs()
        # blpair_lens given without blpair_groups → warning
        with pytest.warns(UserWarning, match="blpair_lens given but blpair_groups is None"):
            grouping.spherical_wf_from_uvp(
                uvp_exact_wfs,
                kbin_edges,
                blpair_groups=None,
                blpair_lens=blpair_lens,
                little_h="h^-3" in uvp_exact_wfs.norm_units,
            )
        # blpair_groups given without blpair_lens → no error
        grouping.spherical_wf_from_uvp(
            uvp_exact_wfs,
            kbin_edges,
            blpair_groups=blpair_groups,
            blpair_lens=None,
            little_h="h^-3" in uvp_exact_wfs.norm_units,
        )
        # inconsistent blpair_lens → error
        with pytest.raises(
            AssertionError,
            match="Baseline-pair groups are inconsistent with baseline lengths",
        ):
            grouping.spherical_wf_from_uvp(
                uvp_exact_wfs,
                kbin_edges,
                blpair_groups=blpair_groups,
                blpair_lens=[blpair_lens[0], blpair_lens[0]],
                little_h="h^-3" in uvp_exact_wfs.norm_units,
            )

    def test_wf_overlapping_kbins(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that overlapping kbin_edges raise an AssertionError (with a little_h warning)."""
        with pytest.warns(UserWarning, match="Changed little_h units"):
            with pytest.raises(AssertionError, match="kbins must not overlap"):
                grouping.spherical_wf_from_uvp(uvp_exact_wfs, kbin_edges=np.array([1.0, 2.0, 1.5]))

    def test_wf_no_exact_windows(self, uvp_exact_wfs: UVPSpec) -> None:
        """Check that spherical_wf_from_uvp raises an AssertionError when exact_windows=False."""
        dk = 0.25
        kbin_edges = np.arange(0.075, 2.9, dk)
        uvp = copy.deepcopy(uvp_exact_wfs)
        uvp.exact_windows = False
        with pytest.raises(
            AssertionError, match="Need to compute exact window functions first"
        ):
            grouping.spherical_wf_from_uvp(
                uvp, kbin_edges, little_h="h^-3" in uvp.norm_units
            )


class TestAverageInDelayBins:
    def test_delay_slices_errors(self) -> None:
        """Check that _get_delay_slices raises a ValueError when the zero_kernel has even length."""
        with pytest.raises(ValueError, match="nzero must be odd!"):
            grouping._get_delay_slices(
                dly=np.linspace(0, 1, 10),
                kernel=np.array([1, 1, 1]),
                zero_kernel=np.array([1, 0, 1, 0]),
            )

    def test_invalid_kernel(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Test that proper exceptions are raised for bad inputs."""
        with pytest.raises(ValueError, match="The kernel must be 1D"):
            grouping.average_in_delay_bins(
                vanilla_uvp_with_beam, kernel=np.array([[1, 1, 1]])
            )

        with pytest.raises(ValueError, match="The kernel size must be smaller than half"):
            grouping.average_in_delay_bins(
                vanilla_uvp_with_beam, kernel=np.zeros(vanilla_uvp_with_beam.Ndlys)
            )

        with pytest.raises(ValueError, match="The zero bin kernel must be symmetric"):
            grouping.average_in_delay_bins(
                vanilla_uvp_with_beam,
                kernel=np.array([1, 1]),
                zero_bin_kernel=np.array([1, 0, 1, 0]),
            )

    def test_basic(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that average_in_delay_bins reduces the number of delay bins by the kernel size."""
        new = grouping.average_in_delay_bins(
            vanilla_uvp_with_beam, kernel=np.array([1, 1, 1])
        )
        assert len(new.get_dlys(0)) - 1 == (len(vanilla_uvp_with_beam.get_dlys(0)) - 1) // 3

    def test_wf_propagation(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that average_in_delay_bins averages window functions over the kernel."""
        new = grouping.average_in_delay_bins(
            vanilla_uvp_with_beam, kernel=np.array([1, 1, 1])
        )
        assert np.allclose(
            np.mean(vanilla_uvp_with_beam.window_function_array[0][:, 1:4], axis=1),
            new.window_function_array[0][:, 0, :, :],
        )

    def test_fold_propagation(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that fold_spectra correctly zeroes the lower-half window functions."""
        ispw = 0
        folded_uvp = copy.deepcopy(vanilla_uvp_with_beam)
        grouping.fold_spectra(folded_uvp)
        Ndlys = vanilla_uvp_with_beam.data_array[ispw].shape[1]
        assert not folded_uvp.window_function_array[ispw][:, : Ndlys // 2].any(), (
            "Window functions wrongly propagated by grouping.fold_spectra"
        )

    def test_average_spectra_propagation(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that average_spectra correctly propagates delay-binned window functions."""
        new = grouping.average_in_delay_bins(
            vanilla_uvp_with_beam, kernel=np.array([1, 1, 1])
        )
        averaged_uvp = grouping.average_spectra(
            vanilla_uvp_with_beam, time_avg=True, inplace=False
        )
        averaged_new = grouping.average_spectra(new, time_avg=True, inplace=False)
        assert np.allclose(
            np.mean(averaged_uvp.window_function_array[0][:, 1:4], axis=1),
            averaged_new.window_function_array[0][:, 0, :, :],
        ), "Window functions wrongly propagated by grouping.average_spectra"

    def test_spherical_average_propagation(self, vanilla_uvp_with_beam: UVPSpec) -> None:
        """Check that spherical_average correctly propagates delay-binned window functions, with and without theory kbins."""
        new = grouping.average_in_delay_bins(
            vanilla_uvp_with_beam, kernel=np.array([1, 1, 1])
        )
        dk = 0.08959223 * 3.0
        kbin_left = np.arange(dk / 3 / 2, 2.3, dk)
        kbin_right = kbin_left + dk
        kbins = (kbin_left + kbin_right) / 2.0
        sph_uvp = grouping.spherical_average(
            vanilla_uvp_with_beam, kbins, dk, time_avg=True
        )
        sph_new = grouping.spherical_average(new, kbins, dk, time_avg=True)
        assert np.allclose(
            sph_uvp.window_function_array[0][0, :, :-1, 0],
            # given the bin edges, the final bin is empty in the delay averaged spectrum
            sph_new.window_function_array[0][0, :, :-1, 0],
        ), "Window functions wrongly propagated by grouping.spherical_average"
        # if theory kbins are given by the user
        sph_uvp2 = grouping.spherical_average(
            vanilla_uvp_with_beam,
            kbins,
            bin_widths=dk,
            kbins_theory=kbins[::2],
            time_avg=True,
        )
        sph_new2 = grouping.spherical_average(
            new, kbins, bin_widths=dk, kbins_theory=kbins[::2], time_avg=True
        )
        assert np.allclose(
            sph_uvp2.window_function_array[0][0, :, :-1, 0],
            # given the bin edges, the final bin is empty in the delay averaged spectrum
            sph_new2.window_function_array[0][0, :, :-1, 0],
        ), "Window functions wrongly propagated by grouping.spherical_average"

    def test_exact_wf_propagation(self, delay_bins_exact_wf_uvp: UVPSpec) -> None:
        """Check that average_in_delay_bins correctly propagates the delay array and window functions when using exact window functions."""
        new = grouping.average_in_delay_bins(
            delay_bins_exact_wf_uvp, kernel=np.array([1, 1, 1])
        )
        # delay array propagation
        assert np.isclose(
            new.dly_array[0], np.mean(delay_bins_exact_wf_uvp.dly_array[1:4])
        )
        # window functions propagation
        assert np.allclose(
            np.mean(delay_bins_exact_wf_uvp.window_function_array[0][:, 1:4], axis=1),
            new.window_function_array[0][:, 0, ...],
        ), (
            "Window functions wrongly propagated by "
            "grouping.average_in_delay_bins with exact window functions"
        )

    @pytest.mark.parametrize("with_cov", [True, False])
    @pytest.mark.parametrize("cov_weighted_stats", [(), ("P_N",)])
    def test_pn_weighting(
        self,
        vanilla_uvp_with_beam: UVPSpec,
        delay_bins_uvp_nocov: UVPSpec,
        with_cov: bool,
        cov_weighted_stats: tuple,
    ) -> None:
        """Check that P_N stat errors scale correctly after delay binning, with and without covariance weighting."""
        if with_cov:
            uvp = copy.deepcopy(vanilla_uvp_with_beam)
        else:
            uvp = copy.deepcopy(delay_bins_uvp_nocov)

        uvp.stats_array = {
            "P_N": {spw: np.ones_like(uvp.data_array[spw]) for spw in uvp.spw_array}
        }

        new = grouping.average_in_delay_bins(
            uvp,
            kernel=np.array([1, 1, 1]),
            zero_bin_kernel=np.array([1, 1, 1]),
            cov_weighted_stats=cov_weighted_stats,
        )
        assert new.stats_array["P_N"][0].shape == new.data_array[0].shape
        np.testing.assert_allclose(
            new.stats_array["P_N"][0],
            1 / np.sqrt(3) if (with_cov and cov_weighted_stats) else 1,
        )

    def test_without_cov(self, delay_bins_uvp_nocov: UVPSpec) -> None:
        """Test that the function works without cov_array."""
        new = grouping.average_in_delay_bins(
            delay_bins_uvp_nocov, kernel=np.array([1, 1, 1])
        )

        assert len(new.get_dlys(0)) - 1 == (len(delay_bins_uvp_nocov.get_dlys(0)) - 1) // 3

        # Check that the stats_array is empty
        assert new.stats_array == {}

    def test_exact_window_functions(self, delay_bins_exact_wf_uvp: UVPSpec) -> None:
        """Check that average_in_delay_bins produces a window function array of the correct shape when using exact window functions."""
        new = grouping.average_in_delay_bins(
            delay_bins_exact_wf_uvp, kernel=np.array([0, 1, 1, 0])
        )
        oldshape = delay_bins_exact_wf_uvp.window_function_array[0].shape
        newshape = list(oldshape)
        newshape[1] = len(new.get_dlys(0))

        assert new.window_function_array[0].shape == tuple(newshape)

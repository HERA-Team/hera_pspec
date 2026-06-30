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
    conversions,
    grouping,
    parameter,
    pspecbeam,
    pspecdata,
    testing,
    utils,
    uvpspec,
    uvwindow,
)
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


# Setup Test Cases for this module
def case_vanilla_uvp(vanilla_uvp: UVPSpec) -> UVPSpec:
    return vanilla_uvp


def case_vanilla_uvp_with_beam(
    beam_nf_dipole: PSpecBeamUV, vanilla_uvp_with_beam: UVPSpec
) -> UVPSpec:
    return vanilla_uvp_with_beam


def case_vanilla_uvp_w_ndlys(vanilla_uvp_w_ndlys: UVPSpec) -> UVPSpec:
    return vanilla_uvp_w_ndlys


def case_vanilla_uvp_delay_binned(vanilla_uvp_delay_binned: UVPSpec) -> UVPSpec:
    return vanilla_uvp_delay_binned


def case_vanilla_uvp_alternating_times(
    beam_nf_dipole: PSpecBeamUV, vanilla_uvp_alternating_times: UVPSpec
) -> UVPSpec:
    return vanilla_uvp_alternating_times


def case_uvp_exact_wfs(uvp_example_data: UVPSpec, uvp_exact_wfs: UVPSpec) -> UVPSpec:
    return uvp_exact_wfs


@pytest.fixture
def uvp_with_covariance(
    beam_nf_dipole_wcosmo: PSpecBeamUV, uvd_zen_even_xx: UVData
) -> uvpspec.UVPSpec:
    """UVPSpec from zen.even.xx.LST.1.28828.uvOCRSA with covariance computed (store_cov=True)."""
    uvd = copy.deepcopy(uvd_zen_even_xx)

    Jy_to_mK = beam_nf_dipole_wcosmo.Jy_to_mK(np.unique(uvd.freq_array), pol="XX")
    uvd.data_array *= Jy_to_mK[None, :, None]

    uvd1 = uvd.select(times=np.unique(uvd.time_array)[: uvd.Ntimes // 2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[uvd.Ntimes // 2 :], inplace=False)

    ds = pspecdata.PSpecData(
        dsets=[uvd1, uvd2], wgts=[None, None], beam=beam_nf_dipole_wcosmo
    )
    ds.rephase_to_dset(0)

    spws = utils.spw_range_from_freqs(
        uvd, freq_range=[(160e6, 165e6), (160e6, 165e6)], bounds_error=True
    )
    antpos, ants = uvd.get_enu_data_ants()
    red_bls = redcal.get_pos_reds(dict(zip(ants, antpos)), bl_error_tol=1.0)
    bls1, bls2, _ = utils.construct_blpairs(
        red_bls[3], exclude_auto_bls=True, exclude_permutations=True
    )

    return ds.pspec(
        bls1,
        bls2,
        (0, 1),
        [("xx", "xx")],
        spw_ranges=spws,
        input_data_weight="identity",
        norm="I",
        taper="blackman-harris",
        store_cov=True,
        cov_model="autos",
        verbose=False,
    )


def _add_optionals(uvp: uvpspec.UVPSpec) -> uvpspec.UVPSpec:
    """Add dummy optional cov_array and stats_array to uvp."""
    uvp.cov_array_real = {}
    uvp.cov_array_imag = {}
    uvp.cov_model = "empirical"
    stat = "noise_err"
    uvp.stats_array = {stat: {}}
    for spw in uvp.spw_array:
        ndlys = uvp.get_spw_ranges(spw)[0][-1]
        uvp.cov_array_real[spw] = np.empty(
            (uvp.Nbltpairs, ndlys, ndlys, uvp.Npols), np.float64
        )
        uvp.cov_array_imag[spw] = np.empty(
            (uvp.Nbltpairs, ndlys, ndlys, uvp.Npols), np.float64
        )
        uvp.stats_array[stat][spw] = np.empty(
            (uvp.Nbltpairs, ndlys, uvp.Npols), np.complex128
        )
    return uvp


def assert_uvpspec_equal(uvp1: UVPSpec, uvp2: UVPSpec) -> None:
    """Helper to compare two UVPSpec objects."""
    assert np.all(uvp1.spw_array == uvp2.spw_array)
    assert np.all(uvp1.polpair_array == uvp2.polpair_array)
    for k in uvp1.data_array:
        assert np.allclose(uvp1.data_array[k], uvp2.data_array[k])
        assert np.allclose(uvp1.nsample_array[k], uvp2.nsample_array[k])
        assert np.allclose(uvp1.integration_array[k], uvp2.integration_array[k])


def test_param() -> None:
    parameter.PSpecParam("example", description="example", expected_type=int)


@parametrize_with_cases("uvp", cases=".")
def test_eq(uvp: uvpspec.UVPSpec) -> None:
    assert uvp == uvp


@pytest.mark.parametrize(
    "key",
    [
        (0, ((1, 2), (1, 2)), ("xx", "xx")),
        (0, ((1, 2), (1, 2)), 1515),
        (0, 101102101102, 1515),
    ],
)
@parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
def test_get_data_key_formats(uvp: uvpspec.UVPSpec, key: tuple) -> None:
    d = uvp.get_data(key)
    assert d.shape == (uvp.Ntimes, uvp.get_dlys(0).size)
    assert d.dtype == complex
    np.testing.assert_almost_equal(d[0, 0], (101.1021011020000001 + 0j))


@pytest.mark.parametrize(
    "method,expected_shape,first_idx",
    [
        ("get_wgts", (10, 50, 2), (0, 0, 0)),  # Nfreq dim, not Ndlys
        ("get_integrations", (10,), (0,)),
        ("get_nsamples", (10,), (0,)),
    ],
)
@parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
def test_get_array_funcs(
    uvp: uvpspec.UVPSpec,
    method: str,
    expected_shape: tuple[int, ...],
    first_idx: tuple[int, ...],
) -> None:
    key = (0, ((1, 2), (1, 2)), ("xx", "xx"))
    result = getattr(uvp, method)(key)
    assert result.shape == expected_shape
    assert result.dtype == float
    np.testing.assert_almost_equal(result[first_idx], 1.0)


class TestGetFuncs:
    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_dlys(self, uvp: uvpspec.UVPSpec) -> None:
        d = uvp.get_dlys(0)
        assert len(d) == uvp.get_dlys(0).size

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_blpair_seps(self, uvp: uvpspec.UVPSpec) -> None:
        blp = uvp.get_blpair_seps()
        assert len(blp) == 30
        assert np.isclose(blp, 14.60, rtol=1e-1, atol=1e-1).all()

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_kperps_and_kparas(self, uvp: uvpspec.UVPSpec) -> None:
        k_perp, k_para = uvp.get_kperps(0), uvp.get_kparas(0)
        assert len(k_perp) == 30
        assert len(k_para) == uvp.get_dlys(0).size

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_data_tuple_key(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that get_data accepts a (spw, blpair, polpair) tuple key."""
        key = (0, ((1, 2), (1, 2)), ("xx", "xx"))
        d = uvp.get_data(key)
        assert d.shape == (uvp.Ntimes, uvp.get_dlys(0).size)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_data_dict_key(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that get_data accepts an equivalent {spw, blpair, polpair} dict key."""
        key = {"spw": 0, "blpair": ((1, 2), (1, 2)), "polpair": ("xx", "xx")}
        d = uvp.get_data(key)
        assert d.shape == (uvp.Ntimes, uvp.get_dlys(0).size)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_blpairs(self, uvp: uvpspec.UVPSpec) -> None:
        blps = uvp.get_blpairs()
        assert blps == [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_blpair_blvecs(self, uvp: uvpspec.UVPSpec) -> None:
        """Check get_blpair_blvecs' output shape, and that use_second_bl gives the same vectors here."""
        blp_vecs = uvp.get_blpair_blvecs()
        assert blp_vecs.shape == (uvp.Nblpairs, 3)
        blp_vecs2 = uvp.get_blpair_blvecs(use_second_bl=True)
        assert np.isclose(blp_vecs, blp_vecs2).all()

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_polpairs(self, uvp: uvpspec.UVPSpec) -> None:
        polpairs = uvp.get_polpairs()
        assert polpairs == [("xx", "xx")]

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_all_keys(self, uvp: uvpspec.UVPSpec) -> None:
        keys = uvp.get_all_keys()
        assert keys == [
            (0, ((1, 2), (1, 2)), ("xx", "xx")),
            (0, ((2, 3), (2, 3)), ("xx", "xx")),
            (0, ((1, 3), (1, 3)), ("xx", "xx")),
        ]

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_get_integrations_omit_flags(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that omit_flags drops zero-integration entries from the returned array."""
        # deepcopy: case_* functions return the shared session-scoped fixture directly,
        # and this test mutates integration_array in place.
        uvp = copy.deepcopy(uvp)
        uvp.integration_array[0][uvp.blpair_to_indices(((1, 2), (1, 2)))[:2]] = 0.0
        assert uvp.get_integrations(
            (0, ((1, 2), (1, 2)), ("xx", "xx")), omit_flags=True
        ).shape == (8,)


def test_get_covariance(uvp_with_covariance: uvpspec.UVPSpec) -> None:
    blpairs = uvp_with_covariance.get_blpairs()
    key = (0, blpairs[0], "xx")

    cov_real = uvp_with_covariance.get_cov(key, component="real")
    assert cov_real[0].shape == (50, 50)
    cov_imag = uvp_with_covariance.get_cov(key, component="imag")
    assert cov_imag[0].shape == (50, 50)

    uvp_with_covariance.fold_spectra()

    cov_real = uvp_with_covariance.get_cov(key, component="real")
    assert cov_real[0].shape == (24, 24)
    cov_imag = uvp_with_covariance.get_cov(key, component="imag")
    assert cov_imag[0].shape == (24, 24)


class TestStatsArray:
    def test_set_get_average_and_io(
        self, vanilla_uvp_with_beam: uvpspec.UVPSpec, tmp_path: Path
    ) -> None:
        """Check set_stats/get_stats validation, error-weighted averaging, and stats round-tripping through HDF5."""
        uvp = copy.deepcopy(vanilla_uvp_with_beam)
        keys = uvp.get_all_keys()
        with pytest.raises(ValueError, match="must match data_array shape"):
            uvp.set_stats("errors", keys[0], np.linspace(0, 1, 2))
        with pytest.raises(AttributeError, match="No stats have been entered"):
            uvp.get_stats("__", keys[0])
        errs = np.ones((uvp.Ntimes, uvp.get_dlys(0).size))
        for key in keys:
            uvp.set_stats("errors", key, errs)
        uvp.get_stats("errors", keys[0])
        assert np.all(uvp.get_stats("errors", keys[0]) == errs)

        blpairs = uvp.get_blpairs()
        u = uvp.average_spectra(
            [blpairs], time_avg=False, error_weights="errors", inplace=False
        )
        assert np.all(
            np.isclose(
                u.get_stats("errors", keys[0])[0],
                np.ones(u.Ndlys) / np.sqrt(len(blpairs)),
            )
        )
        for key in keys:
            uvp.set_stats("who?", key, errs)
        u = uvp.average_spectra(
            [blpairs], time_avg=False, error_field=["errors", "who?"], inplace=False
        )
        uvp.average_spectra(
            [blpairs], time_avg=True, error_field=["errors", "who?"], inplace=False
        )
        assert np.all(u.get_stats("errors", keys[0]) == u.get_stats("who?", keys[0]))
        u.select(times=np.unique(u.time_avg_array)[:20])

        u3 = uvp.average_spectra([blpairs], time_avg=True, inplace=False)
        with pytest.raises(KeyError, match="not found in stats_array keys"):
            uvp.average_spectra(
                [blpairs], time_avg=True, inplace=False, error_field=["..............."]
            )
        assert not hasattr(u3, "stats_array")

        u.write_hdf5(tmp_path / "ex.hdf5")
        u.read_hdf5(tmp_path / "ex.hdf5")

    def test_fold_propagates_stats_in_inverse_quadrature(
        self, vanilla_uvp_with_beam: uvpspec.UVPSpec
    ) -> None:
        """Check that fold_spectra combines stats_array errors by summing in inverse quadrature."""
        uvp = copy.deepcopy(vanilla_uvp_with_beam)
        key = uvp.get_all_keys()[0]
        Ndlys = uvp.get_dlys(0).size
        errs = np.repeat(np.arange(1, Ndlys + 1)[None], uvp.Ntimes, axis=0)
        uvp.set_stats("test", key, errs)
        uvp.fold_spectra()
        # fold by summing in inverse quadrature
        folded_errs = np.sum(
            [
                1 / errs[:, 1 : Ndlys // 2][:, ::-1] ** 2.0,
                1 / errs[:, Ndlys // 2 + 1 :] ** 2.0,
            ],
            axis=0,
        ) ** (-0.5)
        np.testing.assert_array_almost_equal(uvp.get_stats("test", key), folded_errs)

    def test_set_stats_slice(self, vanilla_uvp_with_beam: uvpspec.UVPSpec) -> None:
        """Check that set_stats_slice sets values above a delay threshold while leaving values below it untouched."""
        uvp = copy.deepcopy(vanilla_uvp_with_beam)
        key = (0, ((1, 2), (1, 2)), ("xx", "xx"))
        uvp.set_stats("err", key, np.ones((uvp.Ntimes, uvp.get_dlys(0).size)))
        uvp.set_stats_slice("err", 50, 0, above=True, val=10)
        # ensure all dlys above 50 * 15 ns are set to 10 and all others set to 1
        assert np.isclose(
            uvp.get_stats("err", key)[:, np.abs(uvp.get_dlys(0) * 1e9) > 15 * 50], 10
        ).all()
        assert np.isclose(
            uvp.get_stats("err", key)[:, np.abs(uvp.get_dlys(0) * 1e9) < 15 * 50], 1
        ).all()


def test_convert_deltasq(uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV) -> None:
    uvd_std = copy.deepcopy(uvd_zen_even_xx)  # dummy uvd_std
    uvd_std.data_array[:] = 1.0
    bls = [(37, 38), (38, 39), (52, 53)]
    uvp = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        bls,
        data_std=uvd_std,
        spw_ranges=[(20, 30), (60, 90)],
        beam=beam_nf_dipole,
    )
    # dummy stats_array build
    Tsys = utils.uvd_to_Tsys(uvd_zen_even_xx, beam_nf_dipole)
    utils.uvp_noise_error(uvp, Tsys)

    # testing
    dsq = uvp.convert_to_deltasq(inplace=False)
    for spw in uvp.spw_array:
        k_perp, k_para = uvp.get_kperps(spw), uvp.get_kparas(spw)
        k_mag = np.sqrt(k_perp[:, None, None] ** 2 + k_para[None, :, None] ** 2)
        coeff = k_mag**3 / (2 * np.pi**2)
        # check data
        assert np.isclose(
            dsq.data_array[spw][0, :, 0], (uvp.data_array[spw] * coeff)[0, :, 0]
        ).all()
        # check stats
        assert np.isclose(
            dsq.stats_array["P_N"][spw][0, :, 0],
            (uvp.stats_array["P_N"][spw] * coeff)[0, :, 0],
        ).all()
        # check cov
        assert np.isclose(
            dsq.cov_array_real[spw][0, :, :, 0].diagonal(),
            uvp.cov_array_real[spw][0, :, :, 0].diagonal() * coeff[0, :, 0] ** 2,
        ).all()
    assert dsq.norm_units == uvp.norm_units + " k^3 / (2pi^2)"


def test_blpair_conversions(vanilla_uvp: uvpspec.UVPSpec) -> None:
    uvp = vanilla_uvp

    # test blpair -> antnums
    an = uvp.blpair_to_antnums(101102101102)
    assert an == ((1, 2), (1, 2))
    # test antnums -> blpair
    bp = uvp.antnums_to_blpair(((1, 2), (1, 2)))
    assert bp == 101102101102
    # test bl -> antnums
    an = uvp.bl_to_antnums(101102)
    assert an == (1, 2)
    # test antnums -> bl
    bp = uvp.antnums_to_bl((1, 2))
    assert bp == 101102


class TestIndicesFuncs:
    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_key_to_indices(self, uvp: uvpspec.UVPSpec) -> None:
        spw, blpairts, pol = uvp.key_to_indices((0, ((1, 2), (1, 2)), 1515))
        assert spw == 0
        assert pol == 0
        assert np.isclose(
            blpairts, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
        ).min()
        spw, blpairts, pol = uvp.key_to_indices((0, 101102101102, ("xx", "xx")))
        assert spw == 0
        assert pol == 0
        assert np.isclose(
            blpairts, np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
        ).min()

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_key_to_indices_polpair_formats_agree(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that int, tuple, and single-pol-string polpair specs give the same indices."""
        s1, b1, p1 = uvp.key_to_indices((0, ((1, 2), (1, 2)), 1515))
        s2, b2, p2 = uvp.key_to_indices((0, ((1, 2), (1, 2)), ("xx", "xx")))
        s3, b3, p3 = uvp.key_to_indices((0, ((1, 2), (1, 2)), "xx"))
        assert p1 == p2 == p3

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_spw_to_indices(self, uvp: uvpspec.UVPSpec) -> None:
        spw1 = uvp.spw_to_dly_indices(0)
        assert len(spw1) == uvp.get_dlys(0).size
        spw2 = uvp.spw_to_freq_indices(0)
        assert len(spw2) == uvp.Nfreqs
        spw3 = uvp.spw_indices(0)
        assert len(spw3) == uvp.Nspws

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_spw_to_indices_accepts_spw_tuple(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that feeding a spw tuple (from get_spw_ranges()) matches feeding the spw index."""
        spw1b = uvp.spw_to_dly_indices(uvp.get_spw_ranges()[0])
        spw2b = uvp.spw_to_freq_indices(uvp.get_spw_ranges()[0])
        spw3b = uvp.spw_indices(uvp.get_spw_ranges()[0])
        np.testing.assert_array_equal(uvp.spw_to_dly_indices(0), spw1b)
        np.testing.assert_array_equal(uvp.spw_to_freq_indices(0), spw2b)
        np.testing.assert_array_equal(uvp.spw_indices(0), spw3b)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_polpair_to_indices(self, uvp: uvpspec.UVPSpec) -> None:
        pol = uvp.polpair_to_indices(("xx", "xx"))
        print(pol)
        assert len(pol) == 1
        pol = uvp.polpair_to_indices(1515)
        print(pol)
        assert len(pol) == 1
        pol = uvp.polpair_to_indices([("xx", "xx"), ("xx", "xx")])
        print(pol)
        assert len(pol) == 1

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_polpair_to_indices_raises_on_invalid_type(
        self, uvp: uvpspec.UVPSpec
    ) -> None:
        with pytest.raises(
            TypeError, match="polpair must be list of tuple or int or str"
        ):
            uvp.polpair_to_indices(3.14)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_blpair_to_indices(self, uvp: uvpspec.UVPSpec) -> None:
        """Check blpair_to_indices for scalar int/tuple inputs, and that list-of-duplicates inputs agree with the scalar result."""
        expected = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
        inds = uvp.blpair_to_indices(101102101102)
        assert np.isclose(inds, expected).min()
        inds_tuple = uvp.blpair_to_indices(((1, 2), (1, 2)))
        assert np.isclose(inds_tuple, expected).min()

        inds_list = uvp.blpair_to_indices([101102101102, 101102101102])
        np.testing.assert_array_equal(inds_list, inds)
        inds_tuple_list = uvp.blpair_to_indices([((1, 2), (1, 2)), ((1, 2), (1, 2))])
        np.testing.assert_array_equal(inds_tuple_list, inds_tuple)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_time_to_indices(self, uvp: uvpspec.UVPSpec) -> None:
        time = uvp.time_avg_array[5]
        blpair = 101102101102
        inds = uvp.time_to_indices(time=time)
        assert len(inds) == 3
        assert np.isclose(uvp.time_avg_array[inds], time, rtol=1e-10).all()
        inds_blp = uvp.time_to_indices(time=time, blpairs=[blpair])
        assert len(inds_blp) == 1
        assert uvp.blpair_array[inds_blp] == blpair
        # a scalar blpair should be equivalent to a single-element list
        inds_scalar = uvp.time_to_indices(time=time, blpairs=blpair)
        np.testing.assert_array_equal(inds_scalar, inds_blp)


class TestSelect:
    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_bl_group_select(self, uvp: uvpspec.UVPSpec) -> None:
        uvp1 = copy.deepcopy(uvp)
        uvp1.select(bls=[(1, 2)], inplace=True)
        assert uvp1.Nblpairs == 1
        assert uvp1.data_array[0].shape == (uvp.Ntimes, uvp.get_dlys(0).size, 1)
        np.testing.assert_almost_equal(
            uvp.data_array[0][0, 0, 0], (101.1021011020000001 + 0j)
        )

    def test_spw_and_bl_select_with_r_params(
        self, beam_nf_dipole: PSpecBeamUV, uvd_zen_even_xx: UVData
    ) -> None:
        """Check chained spw- then bl-select (inplace=False), including that get_r_params follows the selection."""
        bls = [(37, 38), (38, 39), (52, 53)]
        rp = {
            "filter_centers": [0.0],
            "filter_half_widths": [250e-9],
            "filter_factors": [1e-9],
        }
        r_params = {bl + ("xx",): rp for bl in bls}

        uvp1 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            bls,
            spw_ranges=[(20, 30), (60, 90)],
            beam=beam_nf_dipole,
            r_params=r_params,
        )
        uvp2 = uvp1.select(spws=0, inplace=False)
        assert uvp2.Nspws == 1
        uvp2 = uvp2.select(bls=[(37, 38), (38, 39)], inplace=False)
        assert uvp2.Nblpairs == 1
        assert uvp2.data_array[0].shape == (10, 10, 1)
        np.testing.assert_almost_equal(
            uvp2.data_array[0][0, 0, 0], (-3831605.3903496987 + 8103523.9604128916j)
        )
        assert len(uvp2.get_r_params().keys()) == 2
        for rpkey in uvp2.get_r_params():
            assert rpkey == (37, 38, "xx") or rpkey == (38, 39, "xx")

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_blpair_select(self, uvp: uvpspec.UVPSpec) -> None:
        uvp1 = copy.deepcopy(uvp)
        uvp2 = uvp1.select(blpairs=[101102101102, 102103102103], inplace=False)
        assert uvp2.Nblpairs == 2

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_polpair_select(self, uvp: uvpspec.UVPSpec) -> None:
        uvp2 = uvp.select(polpairs=[1515], inplace=False)
        assert uvp2.polpair_array[0] == 1515

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_time_select(self, uvp: uvpspec.UVPSpec) -> None:
        uvp2 = uvp.select(times=np.unique(uvp.time_avg_array)[:1], inplace=False)
        assert uvp2.Ntimes == 1

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_blpair_and_polpair_select_preserves_stats_shape(
        self, uvp: uvpspec.UVPSpec
    ) -> None:
        """Check that selecting on the full blpair/polpair set preserves data_array and stats_array shapes."""
        Ndlys = uvp.get_dlys(0).size
        uvp1 = copy.deepcopy(uvp)
        uvp1.set_stats(
            "hi",
            uvp.get_all_keys()[0],
            np.ones(Ndlys * uvp1.Ntimes).reshape(uvp1.Ntimes, Ndlys),
        )
        uvp2 = uvp1.select(
            blpairs=uvp1.get_blpairs(), polpairs=uvp1.polpair_array, inplace=False
        )
        assert uvp2.data_array[0].shape == (uvp1.Nbltpairs, Ndlys, 1)
        assert uvp2.stats_array["hi"][0].shape == (uvp1.Nbltpairs, Ndlys, 1)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_non_sliceable_select_single_blpair_multi_pol(
        self, uvp: uvpspec.UVPSpec
    ) -> None:
        """Check select() on a non-sliceable combination of one blpair and a subset of polpairs."""
        Ndlys = uvp.get_dlys(0).size
        uvp2, uvp3, uvp4 = copy.deepcopy(uvp), copy.deepcopy(uvp), copy.deepcopy(uvp)
        uvp2.polpair_array[0] = 1414
        uvp3.polpair_array[0] = 1313
        uvp4.polpair_array[0] = 1212
        uvp1 = uvp + uvp2 + uvp3 + uvp4
        uvp5 = uvp1.select(
            blpairs=[101102101102], polpairs=[1515, 1414, 1313], inplace=False
        )
        assert uvp5.data_array[0].shape == (uvp.Ntimes * 1, Ndlys, 3)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_lst_select(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that selecting on the full set of lsts is a no-op."""
        uvp2 = uvp.select(lsts=np.unique(uvp.lst_avg_array), inplace=False)
        assert uvp == uvp2

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_non_sliceable_select_multi_blpair_multi_pol(
        self, uvp: uvpspec.UVPSpec
    ) -> None:
        """Check select() on a non-sliceable combination of multiple blpairs and multiple polpairs."""
        uvp1 = copy.deepcopy(uvp)
        for i in [1414, 1313, 1212]:
            _uvp = copy.deepcopy(uvp)
            _uvp.polpair_array[0] = i
            uvp1 += _uvp

        uvp1.select(polpairs=[1414, 1313, 1212], blpairs=[101102101102, 102103102103])
        assert uvp1.Npols == 3
        assert uvp1.Nblpairs == 2


def test_get_ENU_bl_vecs(vanilla_uvp: uvpspec.UVPSpec) -> None:
    bl_vecs = vanilla_uvp.get_ENU_bl_vecs()
    assert np.isclose(bl_vecs[0], np.array([-14.6, 0.0, 0.0]), atol=1e-6).min()


@parametrize_with_cases("uvp", cases=".")
def test_check(uvp: uvpspec.UVPSpec) -> None:
    uvp = copy.deepcopy(uvp)
    uvp.check()

    # test failure modes
    nt = uvp.Ntimes
    del uvp.Ntimes
    with pytest.raises(AssertionError, match="required parameter Ntimes doesn't exist"):
        uvp.check()

    uvp.Ntimes = nt
    uvp.data_array = list(uvp.data_array.values())[0]
    with pytest.raises(
        AssertionError, match="attribute data_array needs to be a dictionary"
    ):
        uvp.check()


def test_clear(mutable_uvp: uvpspec.UVPSpec) -> None:
    uvp = mutable_uvp
    uvp._clear()
    assert not hasattr(uvp, "Ntimes")
    assert not hasattr(uvp, "data_array")


def test_get_r_params(uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV) -> None:
    bls = [(37, 38), (38, 39), (52, 53)]
    rp = {
        "filter_centers": [0.0],
        "filter_half_widths": [250e-9],
        "filter_factors": [1e-9],
    }
    r_params = {}
    for bl in bls:
        key1 = bl + ("xx",)
        r_params[key1] = rp
    uvp = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        bls,
        spw_ranges=[(20, 30), (60, 90)],
        beam=beam_nf_dipole,
        r_params=r_params,
    )
    assert r_params == uvp.get_r_params()


@parametrize_with_cases("uvp", cases=".")
def test_write_read_hdf5(uvp: uvpspec.UVPSpec, tmp_path: Path) -> None:
    uvp = copy.deepcopy(uvp)

    out = tmp_path / "ex.hdf5"
    # test basic write execution
    uvp.write_hdf5(out, overwrite=True)
    assert out.exists()

    # test basic read
    uvp2 = uvpspec.UVPSpec()
    uvp2.read_hdf5(out)
    assert uvp == uvp2

    # test just meta
    uvp2 = uvpspec.UVPSpec()
    uvp2.read_hdf5(out, just_meta=True)
    assert hasattr(uvp2, "Ntimes")
    assert not hasattr(uvp2, "data_array")

    # test exception
    with pytest.raises(OSError, match="exists, not overwriting"):
        uvp.write_hdf5(out, overwrite=False)

    # test partial I/O
    uvp.read_hdf5(out, blpairs=uvp.blpair_array[:1])
    assert uvp.Nblpairs == 1
    assert uvp.data_array[0].shape == (uvp.Nbltpairs, uvp.get_dlys(0).size, uvp.Npols)


class TestSense:
    BASE_TSYS = 500
    POLPAIR = ("xx", "xx")

    def test_basic_execution(self, mutable_uvp_with_beam: uvpspec.UVPSpec) -> None:
        """Check that generate_noise_spectra returns a P_N array of shape (Ntimes, Ndlys)."""
        Ndlys = mutable_uvp_with_beam.get_dlys(0).size
        P_N = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="Pk", component="real"
        )
        assert P_N[101102101102].shape == (mutable_uvp_with_beam.Ntimes, Ndlys)

    def test_lower_tsys_gives_lower_noise(
        self, mutable_uvp_with_beam: uvpspec.UVPSpec
    ) -> None:
        """Check that a lower system temperature gives a lower noise power."""
        P_N = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="Pk", component="real"
        )
        P_N2 = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS - 100, form="Pk", component="real"
        )
        assert (P_N[101102101102] > P_N2[101102101102]).all()

    def test_abs_component_exceeds_real(
        self, mutable_uvp_with_beam: uvpspec.UVPSpec
    ) -> None:
        """Check that the abs() component is larger than the real component."""
        P_N = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="Pk", component="real"
        )
        P_N_abs = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="Pk", component="abs"
        )
        assert (P_N[101102101102] < P_N_abs[101102101102]).all()

    def test_delsq_form(self, mutable_uvp_with_beam: uvpspec.UVPSpec) -> None:
        """Check that the DelSq form has the right shape and is smaller than Pk at low k."""
        Ndlys = mutable_uvp_with_beam.get_dlys(0).size
        P_N = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="Pk", component="real"
        )
        Dsq = mutable_uvp_with_beam.generate_noise_spectra(
            0, self.POLPAIR, self.BASE_TSYS, form="DelSq", component="real"
        )
        assert Dsq[101102101102].shape == (mutable_uvp_with_beam.Ntimes, Ndlys)
        assert Dsq[101102101102][0, 1] < P_N[101102101102][0, 1]

    def test_blpair_selection_and_int_polpair(
        self, mutable_uvp_with_beam: uvpspec.UVPSpec
    ) -> None:
        """Check that blpairs= restricts output and an integer polpair code is accepted."""
        Ndlys = mutable_uvp_with_beam.get_dlys(0).size
        blpairs = mutable_uvp_with_beam.get_blpairs()[:1]
        P_N = mutable_uvp_with_beam.generate_noise_spectra(
            0, 1515, self.BASE_TSYS, form="Pk", blpairs=blpairs, component="real"
        )
        assert P_N[101102101102].shape == (mutable_uvp_with_beam.Ntimes, Ndlys)

    def test_tsys_as_dict_of_arrays_captures_time_gradient(
        self, mutable_uvp_with_beam: uvpspec.UVPSpec
    ) -> None:
        """Check that a per-time Tsys array propagates: 2x Tsys gives 4x P_N."""
        uvp = mutable_uvp_with_beam
        blpairs = uvp.get_blpairs()[:1]
        Tsys = {
            uvp.antnums_to_blpair(k): self.BASE_TSYS
            * np.ones((uvp.Ntimes, uvp.get_dlys(0).size))
            * np.linspace(1, 2, uvp.Ntimes)[:, None]
            for k in uvp.get_blpairs()
        }
        P_N = uvp.generate_noise_spectra(
            0, 1515, Tsys, form="Pk", blpairs=blpairs, component="real"
        )
        # assert time gradient is captured: 2 * Tsys results in 4 * P_N
        assert np.isclose(P_N[101102101102][0, 0] * 4, P_N[101102101102][-1, 0])


class TestAverageSpectra:
    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_blpair_averaging(self, uvp: uvpspec.UVPSpec) -> None:
        uvp1 = copy.deepcopy(uvp)
        Ndlys = uvp1.get_dlys(0).size
        blpairs = uvp1.get_blpair_groups_from_bl_groups(
            [[101102, 102103, 101103]], only_pairs_in_bls=False
        )
        uvp2 = uvp1.average_spectra(
            blpair_groups=blpairs, time_avg=False, inplace=False
        )
        assert uvp2.Nblpairs == 1
        assert np.isclose(uvp2.get_nsamples((0, 101102101102, ("xx", "xx"))), 3.0).all()
        assert uvp2.get_data((0, 101102101102, ("xx", "xx"))).shape == (10, Ndlys)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_blpair_averaging_with_weights(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that blpair_weights don't change the result here, since the weighted baselines hold identical data."""
        uvp1 = copy.deepcopy(uvp)
        blpairs = [[101102101102, 101102101102]]
        blpair_wgts = [[2.0, 0.0]]
        uvp3a = uvp1.average_spectra(
            blpair_groups=blpairs, time_avg=False, blpair_weights=None, inplace=False
        )
        uvp3b = uvp1.average_spectra(
            blpair_groups=blpairs,
            time_avg=False,
            blpair_weights=blpair_wgts,
            inplace=False,
        )
        assert np.isclose(
            uvp3a.get_data((0, 101102101102, ("xx", "xx"))),
            uvp3b.get_data((0, 101102101102, ("xx", "xx"))),
        ).all()

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_time_averaging(self, uvp: uvpspec.UVPSpec) -> None:
        uvp1 = copy.deepcopy(uvp)
        Ndlys = uvp1.get_dlys(0).size
        uvp2 = uvp1.average_spectra(time_avg=True, inplace=False)
        assert uvp2.Ntimes == 1
        assert np.isclose(
            uvp2.get_nsamples((0, 101102101102, ("xx", "xx"))), 10.0
        ).all()
        assert uvp2.get_data((0, 101102101102, ("xx", "xx"))).shape == (1, Ndlys)

    @parametrize_with_cases("uvp", cases=".", glob="*vanilla*")
    def test_repeated_baselines_require_time_avg(self, uvp: uvpspec.UVPSpec) -> None:
        """Check that averaging repeated baselines without time_avg raises, but works (and collapses Nblpairs) with time_avg=True."""
        uvp1 = copy.deepcopy(uvp)
        uvp1.blpair_array[uvp1.blpair_to_indices(102103102103)] = 101102101102
        with pytest.raises(ValueError):
            uvp1.average_spectra(
                blpair_groups=[list(np.unique(uvp1.blpair_array))],
                time_avg=False,
                inplace=False,
            )
        uvp1.average_spectra(
            blpair_groups=[list(np.unique(uvp1.blpair_array))], time_avg=True
        )
        assert uvp1.Ntimes == 1
        assert uvp1.Nblpairs == 1


def test_get_blpair_groups_from_bl_groups_input_validation(
    vanilla_uvp: uvpspec.UVPSpec,
) -> None:
    with pytest.raises(
        TypeError, match="blgroups must be a sequence of baseline groups"
    ):
        vanilla_uvp.get_blpair_groups_from_bl_groups("bad")

    with pytest.raises(
        TypeError, match="blgroups must be a sequence of baseline groups"
    ):
        vanilla_uvp.get_blpair_groups_from_bl_groups([101102])

    with pytest.raises(ValueError, match="blgroups cannot contain empty groups"):
        vanilla_uvp.get_blpair_groups_from_bl_groups([[]])


class TestGetExactWindowFunctions:
    FT_FILE = DATA_PATH / "FT_beam_HERA_dipole_test"

    def test_basic_execution(self, uvp_example_data: uvpspec.UVPSpec) -> None:
        """Check that the fiducial inplace call sets exact_windows and the right array shape."""
        uvp = copy.deepcopy(uvp_example_data)
        uvp.get_exact_window_functions(ftbeam=self.FT_FILE, inplace=True)
        assert uvp.exact_windows
        assert uvp.window_function_array[0].shape[0] == uvp.Nbltpairs
        # if not exact window function, array dim is 4
        assert uvp.window_function_array[0].ndim == 5

    def test_inplace_false_matches_inplace_true(
        self, uvp_exact_wfs: uvpspec.UVPSpec
    ) -> None:
        """Check that inplace=False returns the same window functions as the inplace=True attribute."""
        kperp_bins, kpara_bins, wf_array = uvp_exact_wfs.get_exact_window_functions(
            ftbeam=self.FT_FILE, inplace=False
        )
        assert np.allclose(wf_array[0], uvp_exact_wfs.window_function_array[0])

    def test_recompute_single_spw_warns(self, uvp_exact_wfs: uvpspec.UVPSpec) -> None:
        """Check that recomputing for one spw on an object with exact_windows already set warns."""
        uvp = copy.deepcopy(uvp_exact_wfs)
        with pytest.warns(
            UserWarning, match="Exact window functions already computed, overwriting"
        ):
            uvp.get_exact_window_functions(
                ftbeam=self.FT_FILE, spw_array=0, inplace=True, verbose=True
            )

    def test_raises_on_spw_not_in_object(self, uvp_exact_wfs: uvpspec.UVPSpec) -> None:
        """Check that an out-of-range spw raises a ValueError (after the overwrite warning fires)."""
        uvp = copy.deepcopy(uvp_exact_wfs)
        with pytest.warns(
            UserWarning, match="Exact window functions already computed, overwriting"
        ):
            with pytest.raises(
                ValueError, match="input spw is not in UVPSpec.spw_array"
            ):
                uvp.get_exact_window_functions(
                    ftbeam=self.FT_FILE, spw_array=2, inplace=True
                )

    def test_raises_on_invalid_ftbeam_type(
        self, uvp_example_data: uvpspec.UVPSpec
    ) -> None:
        with pytest.raises(TypeError, match="ftbeam must be a path-like object"):
            uvp_example_data.get_exact_window_functions(ftbeam=3.14, inplace=False)

    def test_accepts_spw_array_as_list(self, uvp_example_data: uvpspec.UVPSpec) -> None:
        """Check that spw_array can be fed as a single-element list."""
        uvp_example_data.get_exact_window_functions(
            ftbeam=self.FT_FILE, spw_array=[0], inplace=False
        )

    @pytest.mark.parametrize(
        "bad_spw_array", [["bad"], 3.14], ids=["list_of_str", "float"]
    )
    def test_raises_on_invalid_spw_array_type(
        self, uvp_example_data: uvpspec.UVPSpec, bad_spw_array
    ) -> None:
        with pytest.raises(
            TypeError, match="spw_array must be an integer or a sequence"
        ):
            uvp_example_data.get_exact_window_functions(
                ftbeam=self.FT_FILE, spw_array=bad_spw_array, inplace=False
            )

    def test_raises_on_invalid_x_orientation_type(
        self, uvp_example_data: uvpspec.UVPSpec
    ) -> None:
        with pytest.raises(TypeError, match="x_orientation must be a string or None"):
            uvp_example_data.get_exact_window_functions(
                ftbeam=self.FT_FILE, x_orientation=1, inplace=False
            )

    def test_partial_spw_selection_forces_inplace_false_warns(
        self, uvp_example_data: uvpspec.UVPSpec
    ) -> None:
        """Check that requesting one spw of a multi-spw object silently overrides inplace=True to False, with a warning."""
        uvp_multi = copy.deepcopy(uvp_example_data)
        uvp_multi.spw_array = np.array([0, 1])
        uvp_multi.Nspws = 2
        with pytest.warns(
            UserWarning,
            match="inplace set to False because you are not considering all spectral windows in object.",
        ):
            uvp_multi.get_exact_window_functions(
                ftbeam=self.FT_FILE, spw_array=0, inplace=True
            )

    def test_gaussian_beam_input_warns_on_recompute(
        self, uvp_exact_wfs: uvpspec.UVPSpec
    ) -> None:
        """Check that an FTBeam.gaussian object is accepted as ftbeam, and recomputation still warns."""
        uvp = copy.deepcopy(uvp_exact_wfs)
        widths = -0.0343 * uvp.freq_array / 1e6 + 11.30
        gaussian_beam = uvwindow.FTBeam.gaussian(
            freq_array=uvp.freq_array, widths=widths, pol="xx"
        )
        with pytest.warns(
            UserWarning, match="Exact window functions already computed, overwriting"
        ):
            uvp.get_exact_window_functions(
                ftbeam=gaussian_beam, spw_array=0, inplace=True, verbose=True
            )


class TestFoldSpectra:
    @parametrize_with_cases("uvp", cases=".")
    def test_fold_spectra(self, uvp: uvpspec.UVPSpec) -> None:
        uvp1 = copy.deepcopy(uvp)

        uvp1.fold_spectra()
        assert uvp1.folded
        with pytest.raises(
            AssertionError, match="cannot fold power spectra if uvp.folded is True"
        ):
            uvp1.fold_spectra()

        if uvp.get_dlys(0).size % 2 == 0:
            assert len(uvp1.get_dlys(0)) == len(uvp.get_dlys(0)) // 2 - 1
        else:
            assert len(uvp1.get_dlys(0)) == len(uvp.get_dlys(0)) // 2
        assert np.isclose(uvp1.nsample_array[0], 2.0).all()

    def test_fold_spectra_odd_cases(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
    ) -> None:
        # also run the odd case
        uvd_std = copy.deepcopy(uvd_zen_even_xx)
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            bls,
            data_std=uvd_std,
            spw_ranges=[(0, 17)],
            beam=beam_nf_dipole,
        )
        uvp1.fold_spectra()
        cov_folded = uvp1.get_cov((0, ((37, 38), (38, 39)), ("xx", "xx")))
        data_folded = uvp1.get_data((0, ((37, 38), (38, 39)), ("xx", "xx")))

        # Test fold_spectra method is consistent with average_spectra()
        uvp = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            bls,
            data_std=uvd_std,
            spw_ranges=[(0, 17)],
            beam=beam_nf_dipole,
        )
        # Average then fold
        uvp_avg = uvp.average_spectra(time_avg=True, inplace=False)

        # Fold averaged spectra
        uvp_avg_folded = copy.deepcopy(uvp_avg)
        uvp_avg_folded.fold_spectra()

        # Fold then average
        uvp_folded = copy.deepcopy(uvp)
        uvp_folded.fold_spectra()

        # Average folded spectra
        uvp_folded_avg = uvp_folded.average_spectra(time_avg=True, inplace=False)
        assert np.allclose(
            uvp_avg_folded.get_data((0, ((37, 38), (38, 39)), "xx")),
            uvp_folded_avg.get_data((0, ((37, 38), (38, 39)), "xx")),
            rtol=1e-5,
        )


def test_str(vanilla_uvp: uvpspec.UVPSpec) -> None:
    assert str(vanilla_uvp) != ""


def test_compute_scalar(
    vanilla_uvp: uvpspec.UVPSpec, vanilla_uvp_with_beam: uvpspec.UVPSpec
) -> None:
    uvp = copy.deepcopy(vanilla_uvp_with_beam)
    # test basic execution
    s = uvp.compute_scalar(0, ("xx", "xx"), num_steps=1000, noise_scalar=False)
    np.testing.assert_almost_equal(s / 553995277.90425551, 1.0, decimal=5)

    # test no cosmo
    uvp_no_cosmo = copy.deepcopy(vanilla_uvp_with_beam)
    del uvp_no_cosmo.cosmo
    with pytest.raises(AssertionError, match="self.cosmo object must exist"):
        uvp_no_cosmo.compute_scalar(0, ("xx", "xx"))

    # test no beam (vanilla_uvp has cosmo but no OmegaP/OmegaPP/beam_freqs)
    with pytest.raises(
        AssertionError, match="self.OmegaP, self.OmegaPP and self.beam_freqs must exist"
    ):
        vanilla_uvp.compute_scalar(0, -5)


def test_set_cosmology(
    mutable_uvp_with_beam: uvpspec.UVPSpec, beam_nf_dipole: PSpecBeamUV
) -> None:
    uvp = mutable_uvp_with_beam
    new_cosmo = conversions.Cosmo_Conversions(Om_L=0.0)

    # test no overwrite
    uvp.set_cosmology(new_cosmo, overwrite=False)
    assert uvp.cosmo != new_cosmo

    # test setting cosmology
    uvp.set_cosmology(new_cosmo, overwrite=True)
    assert uvp.cosmo == new_cosmo
    assert uvp.norm_units == "h^-3 Mpc^3"
    assert (uvp.scalar_array > 1.0).all()
    assert (uvp.data_array[0] > 1e5).all()

    # test exception
    new_cosmo2 = conversions.Cosmo_Conversions(Om_L=1.0)
    del uvp.OmegaP
    uvp.set_cosmology(new_cosmo2, overwrite=True)
    assert uvp.cosmo != new_cosmo2

    # try with new beam
    uvp.set_cosmology(new_cosmo2, overwrite=True, new_beam=beam_nf_dipole)
    assert uvp.cosmo == new_cosmo2
    assert hasattr(uvp, "OmegaP")


@pytest.fixture
def uvp1_with_optionals(
    uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
) -> uvpspec.UVPSpec:
    """uvp1 (bls=[(37,38),(38,39),(52,53)], spw_ranges=[(20,30),(60,90)]) with dummy cov/stats/window-function optionals, for combine_uvpspec tests."""
    uvp1 = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(37, 38), (38, 39), (52, 53)],
        spw_ranges=[(20, 30), (60, 90)],
        beam=beam_nf_dipole,
    )
    return _add_optionals(uvp1)


@pytest.fixture
def uvp1_no_optionals(
    uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
) -> uvpspec.UVPSpec:
    """uvp1 (bls=[(37,38),(38,39),(52,53)], spw_ranges=[(20,30)]) without optionals, for combine_uvpspec __add__/history/cov_model tests."""
    return testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(37, 38), (38, 39), (52, 53)],
        spw_ranges=[(20, 30)],
        beam=beam_nf_dipole,
    )


@pytest.fixture
def uvp1_uvp2_blpairts_combined(
    uvp1_with_optionals: uvpspec.UVPSpec,
    uvd_zen_even_xx: UVData,
    beam_nf_dipole: PSpecBeamUV,
) -> tuple[uvpspec.UVPSpec, uvpspec.UVPSpec, uvpspec.UVPSpec]:
    """(uvp1, uvp2, out): uvp2 extends uvp1's baselines with two new (non-overlapping) baselines, and out is their combine_uvpspec result."""
    uvp2 = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(53, 54), (67, 68)],
        spw_ranges=[(20, 30), (60, 90)],
        beam=beam_nf_dipole,
    )
    uvp2 = _add_optionals(uvp2)
    out = uvpspec.combine_uvpspec([uvp1_with_optionals, uvp2], verbose=False)
    return uvp1_with_optionals, uvp2, out


class TestCombineUvpspec:
    def test_single_uvpspec_returned_unchanged(
        self, uvp1_with_optionals: uvpspec.UVPSpec
    ) -> None:
        """Check that combining a length-1 list returns the same object, not a copy."""
        out = uvpspec.combine_uvpspec([uvp1_with_optionals], verbose=False)
        assert id(out) == id(uvp1_with_optionals)

    def test_concat_across_pol(self, uvp1_with_optionals: uvpspec.UVPSpec) -> None:
        uvp1 = uvp1_with_optionals
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Npols == 2
        assert len(set(out.polpair_array) ^ set([1515, 1414])) == 0
        key = (0, ((37, 38), (38, 39)), ("xx", "xx"))
        assert np.all(np.isclose(out.get_nsamples(key), np.ones(10, dtype=np.float64)))
        assert np.all(
            np.isclose(
                out.get_integrations(key),
                190 * np.ones(10, dtype=np.float64),
                atol=5,
                rtol=2,
            )
        )
        # optionals
        for spw in out.spw_array:
            ndlys = out.get_spw_ranges(spw)[0][-1]
            assert out.cov_array_real[spw].shape == (30, ndlys, ndlys, 2)
            assert out.stats_array["noise_err"][spw].shape == (30, ndlys, 2)
            assert out.window_function_array[spw].shape == (30, ndlys, ndlys, 2)
            assert out.cov_model == "empirical"

    def test_concat_across_spw(
        self,
        uvp1_with_optionals: uvpspec.UVPSpec,
        uvd_zen_even_xx: UVData,
        beam_nf_dipole: PSpecBeamUV,
    ) -> None:
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx, bls, spw_ranges=[(85, 101)], beam=beam_nf_dipole
        )
        uvp2 = _add_optionals(uvp2)

        out = uvpspec.combine_uvpspec([uvp1_with_optionals, uvp2], verbose=False)
        assert out.Nspws == 3
        assert out.Nfreqs == 51
        assert out.Nspwdlys == 56

        # optionals
        assert len(out.stats_array["noise_err"]) == 3
        assert len(out.window_function_array) == 3
        assert len(out.cov_array_real) == 3

    def test_concat_across_blpairts(
        self,
        uvp1_uvp2_blpairts_combined: tuple[
            uvpspec.UVPSpec, uvpspec.UVPSpec, uvpspec.UVPSpec
        ],
    ) -> None:
        _, _, out = uvp1_uvp2_blpairts_combined
        assert out.Nblpairs == 4
        assert out.Nbls == 5

    def test_combine_with_delay_averaging(
        self,
        uvp1_uvp2_blpairts_combined: tuple[
            uvpspec.UVPSpec, uvpspec.UVPSpec, uvpspec.UVPSpec
        ],
    ) -> None:
        """Check that combine-then-delay-average matches delay-average-then-combine, and that combining mismatched delay-averaging raises."""
        uvp1, uvp2, out = uvp1_uvp2_blpairts_combined
        new = grouping.average_in_delay_bins(out, kernel=np.array([1, 1, 1]))
        new1 = grouping.average_in_delay_bins(uvp1, kernel=np.array([1, 1, 1]))
        new2 = grouping.average_in_delay_bins(uvp2, kernel=np.array([1, 1, 1]))
        combined_new = uvpspec.combine_uvpspec([new1, new2], merge_history=False)
        assert np.allclose(combined_new.data_array[0], new.data_array[0]), (
            "There was an issue combining two delay-averaged UVPSpec objects."
        )
        with pytest.raises(
            AssertionError, match="non-overlapping across multiple data axes"
        ):
            uvpspec.combine_uvpspec([uvp1, new2])

        # optionals
        for spw in out.spw_array:
            ndlys = out.get_spw_ranges(spw)[0][-1]
            assert out.cov_array_real[spw].shape == (40, ndlys, ndlys, 1)
            assert out.stats_array["noise_err"][spw].shape == (40, ndlys, 1)
            assert out.window_function_array[spw].shape == (40, ndlys, ndlys, 1)

    def test_feed_as_strings(
        self, uvp1_no_optionals: uvpspec.UVPSpec, tmp_path: Path
    ) -> None:
        """Check that combine_uvpspec accepts a list of HDF5 file paths instead of UVPSpec objects."""
        uvp1 = uvp1_no_optionals
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp1.write_hdf5(str(tmp_path / "uvp1.hdf5"), overwrite=True)
        uvp2.write_hdf5(str(tmp_path / "uvp2.hdf5"), overwrite=True)
        out = uvpspec.combine_uvpspec(
            [str(tmp_path / "uvp1.hdf5"), str(tmp_path / "uvp2.hdf5")], verbose=False
        )
        assert out.Npols == 2

    def test_uvpspec_add_operator(self, uvp1_no_optionals: uvpspec.UVPSpec) -> None:
        """Check that the UVPSpec __add__ operator chains to combine more than two objects."""
        uvp1 = uvp1_no_optionals
        uvp2 = copy.deepcopy(uvp1)
        uvp3 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp3.polpair_array[0] = 1313
        out2 = uvp1 + uvp2 + uvp3
        assert out2.Npols == 3

    def test_combine_with_different_n_dlys_per_spw(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
    ) -> None:
        """Check that combining works when Ndlys differs from Nfreqs per spw (n_dlys != Nfreqs)."""
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp4 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            bls,
            beam=beam_nf_dipole,
            spw_ranges=[(20, 30), (60, 90)],
            n_dlys=[5, 15],
        )
        uvp4b = copy.deepcopy(uvp4)
        uvp4b.polpair_array[0] = 1414
        uvpspec.combine_uvpspec([uvp4, uvp4b], verbose=False)

    def test_history_merging(self, uvp1_no_optionals: uvpspec.UVPSpec) -> None:
        uvp_a = copy.deepcopy(uvp1_no_optionals)
        uvp_b = copy.deepcopy(uvp1_no_optionals)
        uvp_b.polpair_array[0] = 1414
        uvp_a.history = "batwing"
        uvp_b.history = "foobar"

        # w/ merge
        out = uvpspec.combine_uvpspec([uvp_a, uvp_b], merge_history=True, verbose=False)
        assert "batwing" in out.history and "foobar" in out.history

        # w/o merge
        out = uvpspec.combine_uvpspec(
            [uvp_a, uvp_b], merge_history=False, verbose=False
        )
        assert "batwing" in out.history and "foobar" not in out.history

    def test_no_cov_array_if_cov_model_inconsistent(
        self, uvp1_no_optionals: uvpspec.UVPSpec
    ) -> None:
        uvp_a = copy.deepcopy(uvp1_no_optionals)
        uvp_b = copy.deepcopy(uvp1_no_optionals)
        uvp_b.cov_model = "foo"
        uvp_b.polpair_array = np.array([1414])
        out = uvpspec.combine_uvpspec([uvp_a, uvp_b], verbose=False)
        assert hasattr(out, "cov_array_real") is False

    def test_combine_uvpspec_exact_windows(self, uvp_exact_wfs: UVPSpec) -> None:
        # for exact windows
        uvp1 = copy.deepcopy(uvp_exact_wfs)
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)

    def test_combine_uvpspec_errors(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
    ) -> None:
        bls = [(37, 38), (38, 39), (52, 53)]
        uvp1 = testing.uvpspec_from_data(
            uvd_zen_even_xx, bls, spw_ranges=[(20, 30), (60, 90)], beam=beam_nf_dipole
        )

        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        with pytest.raises(AssertionError, match="completely overlapping data"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

        # test multiple non-overlapping data axes
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp2.freq_array[0] = 0.0
        with pytest.raises(AssertionError, match="partial overlap across spw"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

        # test partial data overlap failure
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            [(37, 38), (38, 39), (53, 54)],
            spw_ranges=[(20, 30), (60, 90)],
            beam=beam_nf_dipole,
        )
        with pytest.raises(AssertionError, match="partial overlap"):
            uvpspec.combine_uvpspec([uvp1, uvp2])
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx, bls, spw_ranges=[(20, 30), (60, 105)], beam=beam_nf_dipole
        )
        with pytest.raises(AssertionError, match="partial overlap"):
            uvpspec.combine_uvpspec([uvp1, uvp2])
        uvp2 = copy.deepcopy(uvp1)
        uvp2.polpair_array[0] = 1414
        uvp2 = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        with pytest.raises(AssertionError, match="partial overlap"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

        # test failure due to variable static metadata
        uvp2.weighting = "foo"
        with pytest.raises(
            AssertionError, match="not all agree on 'weighting' attribute"
        ):
            uvpspec.combine_uvpspec([uvp1, uvp2])
        uvp2.weighting = "identity"
        del uvp2.OmegaP
        del uvp2.OmegaPP
        with pytest.raises(AssertionError, match="not all agree on 'OmegaP' attribute"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

    def test_combine_uvpspec_r_params(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
    ) -> None:
        bls = [(37, 38), (38, 39), (52, 53)]

        rp = {
            "filter_centers": [0.0],
            "filter_half_widths": [250e-9],
            "filter_factors": [1e-9],
        }

        r_params = {}

        for bl in bls:
            key1 = bl + ("xx",)
            r_params[key1] = rp

        # create an r_params copy with inconsistent weighting to test error case
        r_params_inconsistent = copy.deepcopy(r_params)
        r_params[key1]["filter_half_widths"] = [100e-9]

        uvp1 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            bls,
            spw_ranges=[(20, 30), (60, 90)],
            beam=beam_nf_dipole,
            r_params=r_params,
        )

        # test failure due to overlapping data
        uvp2 = copy.deepcopy(uvp1)
        with pytest.raises(AssertionError, match="completely overlapping data"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

        # test success across pol
        uvp2.polpair_array[0] = 1414

        # test errors when combining with pspecs without r_params
        uvp3 = copy.deepcopy(uvp2)
        uvp3.r_params = ""
        with pytest.raises(ValueError, match="All r_params must be set or empty"):
            uvpspec.combine_uvpspec([uvp1, uvp3])

        # combining multiple uvp objects without r_params should run fine
        uvp4 = copy.deepcopy(uvp1)
        uvp4.r_params = ""
        uvpspec.combine_uvpspec([uvp3, uvp4])

        # now test error case with inconsistent weightings.
        uvp5 = copy.deepcopy(uvp2)
        uvp5.r_params = uvputils.compress_r_params(r_params_inconsistent)
        with pytest.raises(ValueError, match="Conflict between weightings"):
            uvpspec.combine_uvpspec([uvp1, uvp5])


@pytest.fixture
def uvp1_std(uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV) -> uvpspec.UVPSpec:
    """uvp1 (bls=[(37,38),(38,39),(52,53)], spw_ranges=[(20,24),(64,68)]) with a data_std dataset, for combine_uvpspec_std tests."""
    uvd_std = copy.deepcopy(uvd_zen_even_xx)
    return testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(37, 38), (38, 39), (52, 53)],
        data_std=uvd_std,
        spw_ranges=[(20, 24), (64, 68)],
        beam=beam_nf_dipole,
    )


@pytest.fixture
def uvp1_uvp2_std_blpairts(
    uvp1_std: uvpspec.UVPSpec, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
) -> tuple[uvpspec.UVPSpec, uvpspec.UVPSpec]:
    """(uvp1, uvp2): uvp2 extends uvp1_std's baselines with two new (non-overlapping) baselines, both with data_std."""
    uvd_std = copy.deepcopy(uvd_zen_even_xx)
    uvp2 = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(53, 54), (67, 68)],
        spw_ranges=[(20, 24), (64, 68)],
        data_std=uvd_std,
        beam=beam_nf_dipole,
    )
    return uvp1_std, uvp2


@pytest.fixture
def uvp1_uvp2_std_strings(
    uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV
) -> tuple[uvpspec.UVPSpec, uvpspec.UVPSpec]:
    """(uvp1, uvp2) (bls=[(37,38),(38,39),(52,53)], spw_ranges=[(20,30)], with data_std), uvp2 differing only in polpair."""
    uvd_std = copy.deepcopy(uvd_zen_even_xx)
    uvp1 = testing.uvpspec_from_data(
        uvd_zen_even_xx,
        [(37, 38), (38, 39), (52, 53)],
        spw_ranges=[(20, 30)],
        data_std=uvd_std,
        beam=beam_nf_dipole,
    )
    uvp2 = copy.deepcopy(uvp1)
    uvp2.polpair_array[0] = 1414
    return uvp1, uvp2


class TestCombineUvpspecStd:
    def test_raises_on_fully_overlapping_data(self, uvp1_std: uvpspec.UVPSpec) -> None:
        """Check that combining a dataset with itself (with a data_std present) raises on full overlap."""
        uvp2 = copy.deepcopy(uvp1_std)
        with pytest.raises(AssertionError, match="completely overlapping data"):
            uvpspec.combine_uvpspec([uvp1_std, uvp2])

    def test_succeeds_across_pol(self, uvp1_std: uvpspec.UVPSpec) -> None:
        uvp2 = copy.deepcopy(uvp1_std)
        uvp2.polpair_array[0] = 1414
        out = uvpspec.combine_uvpspec([uvp1_std, uvp2], verbose=False)
        assert out.Npols == 2
        assert len(set(out.polpair_array) ^ set([1515, 1414])) == 0

    def test_raises_on_multiple_non_overlapping_axes(
        self, uvp1_std: uvpspec.UVPSpec
    ) -> None:
        """Check that differing on both polpair and freq at once (two non-overlapping axes) raises."""
        uvp2 = copy.deepcopy(uvp1_std)
        uvp2.polpair_array[0] = 1414
        uvp2.freq_array[0] = 0.0
        with pytest.raises(AssertionError, match="partial overlap across spw"):
            uvpspec.combine_uvpspec([uvp1_std, uvp2])

    def test_raises_on_partial_baseline_overlap(
        self,
        uvp1_std: uvpspec.UVPSpec,
        uvd_zen_even_xx: UVData,
        beam_nf_dipole: PSpecBeamUV,
    ) -> None:
        """Check that a uvp2 sharing only some baselines with uvp1 raises a partial-overlap error."""
        uvd_std = copy.deepcopy(uvd_zen_even_xx)
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            [(37, 38), (38, 39), (53, 54)],
            data_std=uvd_std,
            spw_ranges=[(20, 24), (64, 68)],
            beam=beam_nf_dipole,
        )
        with pytest.raises(AssertionError, match="partial overlap"):
            uvpspec.combine_uvpspec([uvp1_std, uvp2])

    def test_raises_on_fully_overlapping_data_rebuilt(
        self,
        uvp1_std: uvpspec.UVPSpec,
        uvd_zen_even_xx: UVData,
        beam_nf_dipole: PSpecBeamUV,
    ) -> None:
        """Check that a uvp2 independently rebuilt with identical bls/spw_ranges still raises on full overlap."""
        uvd_std = copy.deepcopy(uvd_zen_even_xx)
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            [(37, 38), (38, 39), (52, 53)],
            spw_ranges=[(20, 24), (64, 68)],
            data_std=uvd_std,
            beam=beam_nf_dipole,
        )
        with pytest.raises(AssertionError, match="completely overlapping data"):
            uvpspec.combine_uvpspec([uvp1_std, uvp2])

    def test_raises_on_partial_overlap_with_multi_pol_uvp(
        self, uvp1_std: uvpspec.UVPSpec
    ) -> None:
        """Check that combining uvp1 (single pol) with a pre-merged two-pol uvp2 raises a partial-overlap error."""
        uvp2 = copy.deepcopy(uvp1_std)
        uvp2.polpair_array[0] = 1414
        uvp2 = uvpspec.combine_uvpspec([uvp1_std, uvp2], verbose=False)
        with pytest.raises(AssertionError, match="partial overlap"):
            uvpspec.combine_uvpspec([uvp1_std, uvp2])

    def test_concat_across_spw(
        self,
        uvp1_std: uvpspec.UVPSpec,
        uvd_zen_even_xx: UVData,
        beam_nf_dipole: PSpecBeamUV,
    ) -> None:
        uvd_std = copy.deepcopy(uvd_zen_even_xx)
        uvp2 = testing.uvpspec_from_data(
            uvd_zen_even_xx,
            [(37, 38), (38, 39), (52, 53)],
            spw_ranges=[(85, 91)],
            data_std=uvd_std,
            beam=beam_nf_dipole,
        )
        out = uvpspec.combine_uvpspec([uvp1_std, uvp2], verbose=False)
        assert out.Nspws == 3
        assert out.Nfreqs == 14
        assert out.Nspwdlys == 14

    def test_concat_across_blpairts(
        self, uvp1_uvp2_std_blpairts: tuple[uvpspec.UVPSpec, uvpspec.UVPSpec]
    ) -> None:
        uvp1, uvp2 = uvp1_uvp2_std_blpairts
        out = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert out.Nblpairs == 4
        assert out.Nbls == 5

    def test_raises_on_inconsistent_static_metadata(
        self, uvp1_uvp2_std_blpairts: tuple[uvpspec.UVPSpec, uvpspec.UVPSpec]
    ) -> None:
        """Check that mismatched static metadata (weighting, then a missing OmegaP/OmegaPP) raises during combine."""
        uvp1, uvp2 = uvp1_uvp2_std_blpairts
        uvp2.weighting = "foo"
        with pytest.raises(
            AssertionError, match="not all agree on 'weighting' attribute"
        ):
            uvpspec.combine_uvpspec([uvp1, uvp2])
        uvp2.weighting = "identity"
        del uvp2.OmegaP
        del uvp2.OmegaPP
        with pytest.raises(AssertionError, match="not all agree on 'OmegaP' attribute"):
            uvpspec.combine_uvpspec([uvp1, uvp2])

    def test_feed_as_strings(
        self,
        uvp1_uvp2_std_strings: tuple[uvpspec.UVPSpec, uvpspec.UVPSpec],
        tmp_path: Path,
    ) -> None:
        """Check that combine_uvpspec accepts a list of HDF5 file paths instead of UVPSpec objects (with data_std present)."""
        uvp1, uvp2 = uvp1_uvp2_std_strings
        uvp1.write_hdf5(str(tmp_path / "uvp1.hdf5"), overwrite=True)
        uvp2.write_hdf5(str(tmp_path / "uvp2.hdf5"), overwrite=True)
        out = uvpspec.combine_uvpspec(
            [str(tmp_path / "uvp1.hdf5"), str(tmp_path / "uvp2.hdf5")], verbose=False
        )
        assert out.Npols == 2

    def test_uvpspec_add_operator(
        self, uvp1_uvp2_std_strings: tuple[uvpspec.UVPSpec, uvpspec.UVPSpec]
    ) -> None:
        """Check that the UVPSpec __add__ operator chains to combine more than two objects (with data_std present)."""
        uvp1, uvp2 = uvp1_uvp2_std_strings
        uvp3 = copy.deepcopy(uvp1)
        uvp3.polpair_array[0] = 1313
        out = uvp1 + uvp2 + uvp3
        assert out.Npols == 3


class TestRecursiveCombineUvpspec:
    @parametrize_with_cases("uvp", cases=".")
    def test_single(self, uvp: uvpspec.UVPSpec) -> None:
        """Test recursive_combine_uvpspec with a single UVPSpec object."""
        uvps_list = [copy.deepcopy(uvp)]
        combined_recursive = uvpspec.recursive_combine_uvpspec(uvps_list)
        assert_uvpspec_equal(combined_recursive, uvp)

    @parametrize_with_cases("uvp", cases=".")
    def test_pair(self, uvp: uvpspec.UVPSpec) -> None:
        """Test recursive_combine_uvpspec with a pair of UVPSpec objects."""
        uvp_copy = copy.deepcopy(uvp)
        uvp_copy.polpair_array[0] = 1414  # Slight modification for differentiation
        uvps_list = [uvp, uvp_copy]

        combined_recursive = uvpspec.recursive_combine_uvpspec(uvps_list)
        combined_standard = uvpspec.combine_uvpspec(
            uvps_list, merge_history=False, verbose=False
        )

        assert_uvpspec_equal(combined_recursive, combined_standard)

    @parametrize_with_cases("uvp", cases=".")
    def test_multiple(self, uvp: uvpspec.UVPSpec) -> None:
        """Test recursive_combine_uvpspec with multiple UVPSpec objects."""
        uvp1 = copy.deepcopy(uvp)
        uvp2 = copy.deepcopy(uvp)
        uvp2.polpair_array[0] = 1414
        uvp3 = copy.deepcopy(uvp)
        uvp3.polpair_array[0] = 1313

        uvps_list = [uvp1, uvp2, uvp3]
        combined_recursive = uvpspec.recursive_combine_uvpspec(uvps_list)
        combined_standard = uvpspec.combine_uvpspec(
            uvps_list, merge_history=False, verbose=False
        )

        assert_uvpspec_equal(combined_recursive, combined_standard)

    def test_empty(self) -> None:
        """Test recursive_combine_uvpspec with an empty list."""
        with pytest.raises(
            ValueError,
            match="Cannot run recursive_combine_uvpspec on length-0 objects.",
        ):
            uvpspec.recursive_combine_uvpspec([])


def test_backwards_compatibility_read() -> None:
    """This is a backwards compatibility test.
    If it fails, your edits must be changed to make this test pass.
    If the hera_pspec team decides to move forward and break
    compatibility, this file can be overwritten
    and the date of the file changed in the comment below.
    """
    # test read in of a static test file dated 8/2019
    uvp = uvpspec.UVPSpec()
    uvp.read_hdf5(DATA_PATH / "test_uvp.h5")
    for dattr in uvp._meta_deprecated:
        with pytest.raises(AttributeError) as excinfo:
            raise AttributeError("'UVPSpec' object has no attribute")
        assert "'UVPSpec' object has no attribute" in str(excinfo.value)
    for dattr in uvp._meta_dsets_deprecated:
        with pytest.raises(AttributeError) as excinfo:
            raise AttributeError("'UVPSpec' object has no attribute")
        assert "'UVPSpec' object has no attribute" in str(excinfo.value)
    # assert check does not fail
    uvp.check()


def test_add_approximate_cov() -> None:
    uvp = uvpspec.UVPSpec()
    uvp.read_hdf5(DATA_PATH / "test_uvp.h5")
    uvp.stats_array = {
        "P_N": {
            spw: np.ones((uvp.Nbltpairs, len(uvp.get_dlys(spw)), uvp.Npols))
            for spw in uvp.spw_array
        }
    }

    uvp.add_approximate_covariance(inplace=True)
    assert hasattr(uvp, "cov_array_real")
    ndly = len(uvp.get_dlys(0))
    assert uvp.cov_array_real[0].shape == (uvp.Nbltpairs, ndly, ndly, uvp.Npols)
    assert np.allclose(np.diagonal(uvp.cov_array_real[0], axis1=1, axis2=2), 1.0)

    # test that inplace=False works, not changing the original.
    uvp.stats_array["P_N"][0] *= 2
    uvp2 = uvp.add_approximate_covariance(inplace=False)
    assert hasattr(uvp, "cov_array_real")
    assert np.allclose(np.diagonal(uvp.cov_array_real[0], axis1=1, axis2=2), 1.0)
    assert np.allclose(np.diagonal(uvp2.cov_array_real[0], axis1=1, axis2=2), 4.0)

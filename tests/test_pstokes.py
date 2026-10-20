from pathlib import Path

import numpy as np
import pytest
import pyuvdata
import pyuvdata.utils as uvutils

from hera_pspec import pstokes
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

multipol_dset = DATA_PATH / "zen.2458116.31193.HH.uvh5"
multipol_dset_cal = DATA_PATH / "zen.2458116.31193.HH.flagged_abs.calfits"


@pytest.fixture(scope="module")
def stokes_vis(synthetic_uvd) -> tuple[pyuvdata.UVData, dict[str, np.ndarray]]:
    """Linear-polarization visibilities of a known random pseudo-Stokes sky.

    Returns a 4-pol (xx, yy, xy, yx) UVData and the pseudo-Stokes data it was
    built from. I, Q, U and V are independent random complex arrays, and

        XX = I + Q,  YY = I - Q,  XY = U + iV,  YX = U - iV,

    which is the inverse of the "avg" convention used in pstokes, e.g.
    pI = (XX + YY) / 2 and pV = (-i XY + i YX) / 2. Nsamples are random and
    differ between polarizations, and the xx and yy flags differ.
    """
    uvd = synthetic_uvd(
        nfreq=24,
        ntimes=6,
        antpairs=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
        pols=("xx", "yy", "xy", "yx"),
        seed=7,
    )
    stokes = {
        ps: uvd.data_array[..., k].copy()
        for k, ps in enumerate(["pI", "pQ", "pU", "pV"])
    }
    i_, q_, u_, v_ = (stokes[ps] for ps in ["pI", "pQ", "pU", "pV"])
    uvd.data_array = np.stack([i_ + q_, i_ - q_, u_ + 1j * v_, u_ - 1j * v_], axis=-1)
    rng = np.random.default_rng(8)
    uvd.nsample_array = rng.uniform(0.5, 3.0, size=uvd.nsample_array.shape)
    uvd.flag_array[:, 3, 0] = True  # xx
    uvd.flag_array[:, 7, 1] = True  # yy
    uvd.pol_convention = "avg"
    return uvd, stokes


def _single_pol(uvd: pyuvdata.UVData, pol: str) -> pyuvdata.UVData:
    return uvd.select(polarizations=[pol], inplace=False)


@pytest.fixture()
def uvd1(stokes_vis) -> pyuvdata.UVData:
    """The XX visibilities of stokes_vis."""
    return _single_pol(stokes_vis[0], "xx")


@pytest.fixture()
def uvd2(stokes_vis) -> pyuvdata.UVData:
    """The YY visibilities of stokes_vis."""
    return _single_pol(stokes_vis[0], "yy")


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
            pstokes._combine_pol("xx.uvh5", "yy.uvh5", "XX", "YY")

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

    @pytest.mark.parametrize(
        "pstokes_str,pstokes_num", [("pI", 1), ("pQ", 2), ("pU", 3), ("pV", 4)]
    )
    def test_recovers_known_stokes(
        self, stokes_vis, pstokes_str: str, pstokes_num: int
    ) -> None:
        """Pseudo-Stokes visibilities formed from XX/YY/XY/YX recover the Stokes
        parameters the linear polarizations were built from."""
        uvd, stokes = stokes_vis
        uvdS = pstokes.construct_pstokes(dset1=uvd, dset2=uvd, pstokes=pstokes_str)
        np.testing.assert_array_equal(uvdS.polarization_array, [pstokes_num])
        np.testing.assert_allclose(
            uvdS.data_array[..., 0], stokes[pstokes_str], rtol=1e-12
        )

    @pytest.mark.parametrize("pstokes_str", ["pQ", "pU", "pV"])
    def test_unpolarized_sky(self, stokes_vis, pstokes_str: str) -> None:
        """An unpolarized sky (XX = YY = I, XY = YX = 0) has no polarized
        pseudo-Stokes power."""
        uvd = stokes_vis[0].copy()
        stokes_i = stokes_vis[1]["pI"]
        uvd.data_array = np.stack(
            [stokes_i, stokes_i, 0 * stokes_i, 0 * stokes_i], axis=-1
        )
        uvdS = pstokes.construct_pstokes(dset1=uvd, dset2=uvd, pstokes=pstokes_str)
        np.testing.assert_array_equal(uvdS.data_array, 0.0)

    @pytest.mark.parametrize("pstokes_str", ["pI", "pQ"])
    def test_nsamples_describe_the_noise(self, synthetic_uvd, pstokes_str: str) -> None:
        """The combined nsamples n of pseudo-Stokes visibilities describe their
        noise: if XX and YY have independent noise of variance sigma^2 / n_xx
        and sigma^2 / n_yy, pI = (XX + YY) / 2 has variance sigma^2 / n.

        The variance is estimated from 16 x 1000 independent visibilities, so
        its relative standard error is 1/sqrt(16000) ~ 0.8%; we allow five
        standard errors.
        """
        uvd = synthetic_uvd(
            nfreq=16, ntimes=1000, antpairs=[(0, 1)], pols=("xx", "yy"), seed=3
        )
        # unit-variance noise, scaled to the variance of n_xx and n_yy samples
        nsamples = np.array([1.0, 3.0])
        uvd.nsample_array[:] = nsamples
        uvd.data_array /= np.sqrt(nsamples)
        uvd.pol_convention = "avg"
        uvdS = pstokes.construct_pstokes(dset1=uvd, dset2=uvd, pstokes=pstokes_str)
        n_eff = np.unique(uvdS.nsample_array)
        assert len(n_eff) == 1
        variance = np.mean(np.abs(uvdS.data_array) ** 2)
        error = variance / np.sqrt(uvdS.data_array.size)
        assert abs(variance - 1 / n_eff[0]) < 5 * error

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

    @pytest.mark.parametrize(
        "axis,array_name,subset1,subset2,match",
        [
            pytest.param(
                "frequencies",
                "freq_array",
                None,
                slice(0, 10),
                "same frequencies",
                id="frequencies-one_sided",
            ),
            pytest.param(
                "frequencies",
                "freq_array",
                slice(0, 10),
                slice(10, 20),
                "same frequencies",
                id="frequencies-partial_overlap",
            ),
            pytest.param(
                "times",
                "time_array",
                None,
                slice(0, 3),
                "same timestamps",
                id="timestamps-one_sided",
            ),
            pytest.param(
                "times",
                "time_array",
                slice(0, 3),
                slice(1, 4),
                "same timestamps",
                id="timestamps-partial_overlap",
            ),
        ],
    )
    def test_raises_on_mismatched_axis(
        self,
        uvd1: pyuvdata.UVData,
        uvd2: pyuvdata.UVData,
        axis: str,
        array_name: str,
        subset1: slice | None,
        subset2: slice,
        match: str,
    ) -> None:
        """Check that datasets covering different frequencies or times raise a
        mismatch error, whether only dset2 is restricted (one-sided) or both are
        restricted to partially overlapping subsets."""

        def select(uvd: pyuvdata.UVData, subset: slice | None) -> pyuvdata.UVData:
            if subset is None:
                return uvd
            values = np.unique(getattr(uvd, array_name))[subset]
            return uvd.select(**{axis: values}, inplace=False)

        with pytest.raises(ValueError, match=f"dset1 and dset2 must have the {match}"):
            pstokes.construct_pstokes(
                dset1=select(uvd1, subset1), dset2=select(uvd2, subset2)
            )

    def test_dual_pol_input(self, uvd1: pyuvdata.UVData, uvd2: pyuvdata.UVData) -> None:
        """Check that pI can be constructed from a single dual-polarization UVData object."""
        uvd3 = uvd1 + uvd2
        uvdI = pstokes.construct_pstokes(dset1=uvd3, dset2=uvd3, pstokes="pI")
        # same as constructing it from the two single-polarization objects
        uvdI_ref = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes="pI")
        np.testing.assert_array_equal(uvdI.polarization_array, [1])
        np.testing.assert_allclose(uvdI.data_array, uvdI_ref.data_array)
        np.testing.assert_array_equal(uvdI.flag_array, uvdI_ref.flag_array)

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

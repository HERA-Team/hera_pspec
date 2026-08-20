import copy
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy import units
from pyuvdata import UVBeam, UVData
from pyuvdata import utils as uvutils

from hera_pspec import PSpecData, UVPSpec, conversions, utils, uvwindow
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Data files to use in tests
dfile = "zen.2458116.31939.HH.uvh5"
ftfile = "FT_beam_HERA_dipole_test_xx.hdf5"
basename = "FT_beam_HERA_dipole_test"
outfile = "test.hdf5"


@pytest.fixture()
def make_ft_beam_obj() -> Callable[[tuple[int, int] | None], uvwindow.FTBeam]:
    def _factory(spw_range: tuple[int, int] | None = None) -> uvwindow.FTBeam:
        # channel spans are expressed via freq_array (spw_range is deprecated)
        if spw_range is None:
            return uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        bandwidth = uvwindow.FTBeam.get_bandwidth(DATA_PATH / ftfile)
        return uvwindow.FTBeam.from_file(
            ftfile=DATA_PATH / ftfile, freq_array=bandwidth[spw_range[0] : spw_range[1]]
        )

    return _factory


@pytest.fixture()
def ft_beam_spw(
    make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam],
) -> uvwindow.FTBeam:
    """FTBeam loaded with spw_range=(5, 25)."""
    return make_ft_beam_obj(spw_range=(5, 25))


@pytest.fixture()
def ft_bandwidth() -> np.ndarray:
    return uvwindow.FTBeam.get_bandwidth(DATA_PATH / ftfile)


@pytest.fixture()
def uvwindow_obj(
    ft_beam_spw: uvwindow.FTBeam, cosmo: conversions.Cosmo_Conversions
) -> uvwindow.UVWindow:
    return uvwindow.UVWindow(
        ftbeam_obj=ft_beam_spw,
        taper="blackman-harris",
        cosmo=cosmo,
        little_h=True,
        verbose=False,
    )


@pytest.fixture(scope="module")
def red_bl_lens(uvd_zen_2458116: UVData) -> np.ndarray:
    """Redundant baseline lengths for zen.2458116.31939.HH.uvh5."""
    return utils.get_reds(uvd_zen_2458116, bl_error_tol=1.0, pick_data_ants=False)[1]


@pytest.fixture()
def kbins() -> units.Quantity:
    # kmax, dk = 1.0, 0.128 / 2
    # krange = np.arange(dk * 1.5, kmax, step=dk)
    krange = np.arange(0.1, 5.0, step=0.3)
    return ((krange[1:] + krange[:-1]) / 2) * units.h / units.Mpc


@pytest.fixture()
def cyl_wf_result(
    uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Cylindrical window function, k-bins, and baseline length for red_bl_lens[12]."""
    bl_len = red_bl_lens[12]
    kperp, kpara, cyl_wf = uvwindow_obj.get_cylindrical_wf(
        bl_len, kperp_bins=None, kpara_bins=None, return_bins="unweighted"
    )
    return bl_len, kperp, kpara, cyl_wf


@pytest.fixture(scope="session")
def uvp_for_uvwindow(
    beam_nf_dipole, uvd_zen_2458116: UVData
) -> tuple[UVPSpec, UVPSpec, UVPSpec]:
    """UVPSpec objects (uvp, uvp_nocosmo, uvp_crosspol) for UVWindow.from_uvpspec tests."""
    uvd = copy.deepcopy(uvd_zen_2458116)
    uvd.data_array *= beam_nf_dipole.Jy_to_mK(np.unique(uvd.freq_array), pol="xx")[
        None, :, None
    ]
    ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole)
    ds_nocosmo = PSpecData(dsets=[uvd, uvd], wgts=[None, None])
    baselines1, baselines2, _ = utils.construct_blpairs(
        uvd.get_antpairs()[1:], exclude_permutations=False, exclude_auto_bls=True
    )
    taper = "blackman-harris"
    uvp = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=(175, 195),
        taper=taper,
        verbose=False,
    )
    uvp_nocosmo = ds_nocosmo.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        # NB: must be covered by the test FT-beam file (data channels
        # 170-199): from_uvpspec checks that the data frequencies are
        # within the FTBeam bandwidth
        spw_ranges=(175, 195),
        taper=taper,
        verbose=False,
    )
    uvp_crosspol = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=["xx", "yy"],
        spw_ranges=(175, 195),
        taper=taper,
        verbose=False,
    )
    return uvp, uvp_nocosmo, uvp_crosspol


class TestFTBeamInit:
    def test_from_array(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        test = uvwindow.FTBeam(
            data=data,
            pol="xx",
            freq_array=freq_array,
            mapsize=mapsize,
            verbose=False,
            x_orientation="east",
        )
        assert test.pol == "xx"
        assert np.allclose(data, test.ft_beam)

    @pytest.mark.parametrize("bad_data", ["2d", "wrong_shape"])
    def test_invalid_data_shape(
        self, ft_beam_spw: uvwindow.FTBeam, bad_data: str
    ) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        sliced = data[:, :, 0] if bad_data == "2d" else data[:, :, :-1]
        with pytest.raises(ValueError, match="Wrong dimensions for data input"):
            uvwindow.FTBeam(
                data=sliced, pol="xx", freq_array=freq_array, mapsize=mapsize
            )

    def test_freq_mismatch(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        with pytest.raises(ValueError, match="data must have shape"):
            uvwindow.FTBeam(
                data=data[:12, :, :], pol="xx", freq_array=freq_array, mapsize=mapsize
            )

    def test_int_pol(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        test = uvwindow.FTBeam(
            data=data, pol=-5, freq_array=freq_array, mapsize=mapsize
        )
        assert test.pol == uvutils.polnum2str(-5)

    @pytest.mark.parametrize("bad_pol", ["test", 12])
    def test_invalid_pol(
        self, ft_beam_spw: uvwindow.FTBeam, bad_pol: str | int
    ) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        with pytest.raises(ValueError, match="Wrong polarisation"):
            uvwindow.FTBeam(
                pol=bad_pol, data=data, freq_array=freq_array, mapsize=mapsize
            )

    def test_float_pol_raises_typeerror(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        data, freq_array, mapsize = (
            ft_beam_spw.ft_beam,
            ft_beam_spw.freq_array,
            ft_beam_spw.mapsize,
        )
        with pytest.raises(TypeError, match="Must feed pol as str or int"):
            uvwindow.FTBeam(pol=3.4, data=data, freq_array=freq_array, mapsize=mapsize)


nf_dipole_beamfits = DATA_PATH / "HERA_NF_dipole_power.beamfits"


@pytest.fixture(scope="module")
def nf_dipole_beam_freqs() -> np.ndarray:
    beam = UVBeam()
    beam.read_beamfits(str(nf_dipole_beamfits))
    return np.unique(beam.freq_array)


class TestFTBeamFromBeam:
    @pytest.fixture(scope="class")
    def small_ft_beam(self, nf_dipole_beam_freqs: np.ndarray) -> uvwindow.FTBeam:
        freq_array = np.linspace(
            nf_dipole_beam_freqs.min(), nf_dipole_beam_freqs.max(), 5
        )
        return uvwindow.FTBeam.from_beam(
            beamfile=nf_dipole_beamfits,
            pol="xx",
            freq_array=freq_array,
            mapsize=1.0,
            npix=29,
        )

    def test_attributes(self, small_ft_beam: uvwindow.FTBeam) -> None:
        assert small_ft_beam.pol == "xx"
        assert small_ft_beam.ft_beam.ndim == 3
        assert small_ft_beam.ft_beam.shape[0] == small_ft_beam.freq_array.size == 5
        assert small_ft_beam.ft_beam.shape[1] == small_ft_beam.ft_beam.shape[2]
        assert np.all(np.isfinite(small_ft_beam.ft_beam))

    def test_ft_peaks_at_zero_mode(self, small_ft_beam: uvwindow.FTBeam) -> None:
        # the FT of a positive beam peaks at the zero mode (grid centre)
        ngrid = small_ft_beam.ft_beam.shape[-1]
        for i in range(small_ft_beam.freq_array.size):
            assert np.argmax(small_ft_beam.ft_beam[i]) == (ngrid**2) // 2

    def test_pol_as_int(self, small_ft_beam: uvwindow.FTBeam) -> None:
        test = uvwindow.FTBeam.from_beam(
            beamfile=nf_dipole_beamfits,
            pol=-5,
            freq_array=small_ft_beam.freq_array,
            mapsize=1.0,
            npix=29,
        )
        assert test.pol == "xx"
        assert np.allclose(test.ft_beam, small_ft_beam.ft_beam)

    def test_too_few_frequencies(self, nf_dipole_beam_freqs: np.ndarray) -> None:
        with pytest.raises(ValueError, match="at least three frequencies"):
            uvwindow.FTBeam.from_beam(
                beamfile=nf_dipole_beamfits,
                pol="xx",
                freq_array=nf_dipole_beam_freqs[:2],
            )

    def test_out_of_coverage_uses_edge_beam(
        self, nf_dipole_beam_freqs: np.ndarray
    ) -> None:
        # frequencies slightly outside the simulation coverage: warn, and
        # evaluate the beam at the nearest covered frequency while keeping
        # the requested frequencies as the FTBeam coordinates
        df = np.diff(nf_dipole_beam_freqs).mean()
        fmin = nf_dipole_beam_freqs.min()
        freqs_over = np.array([fmin - df / 2, fmin, fmin + df])
        with pytest.warns(UserWarning, match="outside the beam simulation"):
            test = uvwindow.FTBeam.from_beam(
                beamfile=nf_dipole_beamfits,
                pol="xx",
                freq_array=freqs_over,
                mapsize=1.0,
                npix=29,
            )
        assert np.allclose(test.freq_array, freqs_over)
        # clamped channel = beam evaluated at the edge frequency
        assert np.allclose(test.ft_beam[0], test.ft_beam[1])

    def test_select_freqs_matches_direct_computation(
        self, nf_dipole_beam_freqs: np.ndarray
    ) -> None:
        # interpolating a from_beam FT beam onto frequencies between its
        # channels must agree with computing the FT beam directly at those
        # frequencies. The 6.25 MHz source grid used here is far coarser
        # than any production grid (~122 kHz), so the 2%-of-peak bound is
        # loose.
        fine = np.linspace(nf_dipole_beam_freqs.min(), nf_dipole_beam_freqs.max(), 17)
        target = 0.5 * (fine[:-1] + fine[1:])[::2]
        ft_fine = uvwindow.FTBeam.from_beam(
            beamfile=nf_dipole_beamfits, pol="xx", freq_array=fine, mapsize=1.0, npix=29
        )
        ft_direct = uvwindow.FTBeam.from_beam(
            beamfile=nf_dipole_beamfits,
            pol="xx",
            freq_array=target,
            mapsize=1.0,
            npix=29,
        )
        with pytest.warns(UserWarning, match="interpolating"):
            ft_interp = ft_fine.select_freqs(target, inplace=False)
        assert np.allclose(
            ft_interp.ft_beam,
            ft_direct.ft_beam,
            atol=0.02 * np.abs(ft_direct.ft_beam).max(),
        )


class TestFTBeamFromFile:
    def test_happy_path(self, ft_bandwidth: np.ndarray) -> None:
        test = uvwindow.FTBeam.from_file(
            ftfile=DATA_PATH / ftfile,
            freq_array=ft_bandwidth[5:25],
            verbose=False,
            x_orientation="east",
        )
        assert test.pol == "xx"

    def test_invalid_ftfile_type(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"expected str, bytes or os\.PathLike object, not float"
            r"|argument should be a str or an os\.PathLike object where __fspath__ returns a str, not 'float'",
        ):
            uvwindow.FTBeam.from_file(ftfile=12.0)

    def test_invalid_ftfile_path(self) -> None:
        with pytest.raises(ValueError, match="Wrong ftfile input"):
            uvwindow.FTBeam.from_file(ftfile="whatever")

    def test_spw_range_matches_fixture(
        self, make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam]
    ) -> None:
        # deprecated channel-index selection agrees with freq_array selection
        ft_file = DATA_PATH / ftfile
        spw_range = (5, 25)
        with pytest.warns(DeprecationWarning, match="spw_range"):
            test = uvwindow.FTBeam.from_file(ftfile=ft_file, spw_range=spw_range)
        assert np.allclose(
            test.freq_array, make_ft_beam_obj(spw_range=spw_range).freq_array
        )

    def test_no_spw_range_uses_full_bandwidth(self, ft_bandwidth: np.ndarray) -> None:
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, spw_range=None)
        assert np.allclose(test.freq_array, ft_bandwidth)

    @pytest.mark.parametrize(
        "bad_freqs",
        [np.array([250e6, 251e6]), np.array([100e6])],  # out of coverage; single
    )
    def test_invalid_freq_array(self, bad_freqs: np.ndarray) -> None:
        with pytest.raises(ValueError):
            uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, freq_array=bad_freqs)

    @pytest.mark.parametrize("bad_spw", [(13,), (20, 10), (1001, 1022)])
    def test_invalid_spw_range(self, bad_spw: tuple[int, ...]) -> None:
        with (
            pytest.warns(DeprecationWarning, match="spw_range"),
            pytest.raises(ValueError, match="Wrong spw range format"),
        ):
            uvwindow.FTBeam.from_file(spw_range=bad_spw, ftfile=DATA_PATH / ftfile)


class TestFTBeamGaussian:
    def test_array_widths(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        test = uvwindow.FTBeam.gaussian(freq_array=freq_array, widths=widths, pol="xx")
        assert test.freq_array.shape == freq_array.shape

    def test_scalar_width(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        test = uvwindow.FTBeam.gaussian(
            freq_array=freq_array, widths=np.mean(widths), pol="xx"
        )
        assert test.freq_array.shape == freq_array.shape

    def test_too_few_frequencies(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        with pytest.raises(ValueError, match="Must use at least three frequencies"):
            uvwindow.FTBeam.gaussian(
                freq_array=freq_array[:2], pol="xx", widths=np.mean(widths)
            )

    def test_widths_length_mismatch(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        with pytest.raises(
            ValueError, match="There must be as many frequencies as widths"
        ):
            uvwindow.FTBeam.gaussian(
                freq_array=freq_array, pol="xx", widths=widths[:10]
            )

    def test_small_widths_warns(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        with pytest.warns(UserWarning, match="Small widths"):
            uvwindow.FTBeam.gaussian(
                freq_array=ft_beam_spw.freq_array, pol="xx", widths=0.10
            )


class TestFTBeamGetBandwidth:
    def test_matches_fixture(self, ft_bandwidth: np.ndarray) -> None:
        result = uvwindow.FTBeam.get_bandwidth(DATA_PATH / ftfile)
        assert np.all(result == ft_bandwidth)

    def test_invalid_file(self) -> None:
        with pytest.raises(ValueError, match="Wrong ftfile input"):
            uvwindow.FTBeam.get_bandwidth(ftfile="whatever")


class TestFTBeamUpdateSpw:
    def test_happy_path(self) -> None:
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, spw_range=None)
        with pytest.warns(DeprecationWarning, match="select_freqs"):
            test.update_spw((5, 25))

    @pytest.mark.parametrize("bad_spw", [(13,), (20, 10), (1001, 1022)])
    def test_invalid_range(self, bad_spw: tuple[int, ...]) -> None:
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, spw_range=None)
        with (
            pytest.warns(DeprecationWarning, match="select_freqs"),
            pytest.raises(ValueError, match="Wrong spw range format"),
        ):
            test.update_spw(spw_range=bad_spw)


class TestUVWindowInit:
    def test_happy_path(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw)
        assert test is not None

    @pytest.mark.parametrize("bad_spw", [None, (0, 20)])
    def test_inconsistent_ftbeam_spectral_range(
        self,
        bad_spw: tuple[int, int] | None,
        make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam],
        ft_beam_spw: uvwindow.FTBeam,
    ) -> None:
        ft_beam_full = make_ft_beam_obj(spw_range=bad_spw)
        with pytest.raises(
            ValueError, match="Spectral ranges of the two FTBeam objects do not match"
        ):
            uvwindow.UVWindow(ftbeam_obj=(ft_beam_spw, ft_beam_full))

    def test_inconsistent_ftbeam_physical(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        ftbeam_test = copy.deepcopy(ft_beam_spw)
        ftbeam_test.mapsize = 2.0
        with pytest.raises(
            ValueError,
            match="Physical properties of the two FTBeam objects do not match",
        ):
            uvwindow.UVWindow(ftbeam_obj=(ft_beam_spw, ftbeam_test))

    def test_wrong_ftbeam_type(self) -> None:
        with pytest.raises(ValueError, match="Wrong input given in ftbeam_obj"):
            uvwindow.UVWindow(ftbeam_obj="test")

    @pytest.mark.parametrize("taper", ["blackman-harris", None])
    def test_taper(self, ft_beam_spw: uvwindow.FTBeam, taper: str | None) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, taper=taper)
        assert test.taper == taper

    def test_invalid_taper(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        with pytest.raises(ValueError, match="Wrong taper"):
            uvwindow.UVWindow(taper="test", ftbeam_obj=ft_beam_spw)

    def test_cosmo(
        self, ft_beam_spw: uvwindow.FTBeam, cosmo: conversions.Cosmo_Conversions
    ) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, cosmo=cosmo)
        assert test.cosmo is not None

    def test_cosmo_none_raises(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        with pytest.raises(ValueError, match="If no preferred cosmology"):
            uvwindow.UVWindow(cosmo=None, ftbeam_obj=ft_beam_spw)

    def test_verbose(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, verbose=True)
        assert test.verbose

    def test_little_h_true(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, little_h=True)
        assert test.kunits.is_equivalent(units.h / units.Mpc)

    def test_little_h_false(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, little_h=False)
        assert test.kunits.is_equivalent(units.Mpc ** (-1))


class TestUVWindowFromUvpspec:
    def test_happy_path(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        _ = uvwindow.UVWindow.from_uvpspec(
            uvp, ipol=0, spw=0, verbose=True, ftbeam=DATA_PATH / basename
        )

    def test_crosspol(self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]) -> None:
        _, _, uvp_crosspol = uvp_for_uvwindow
        _ = uvwindow.UVWindow.from_uvpspec(
            uvp_crosspol, ipol=0, spw=0, ftbeam=DATA_PATH / basename
        )

    def test_no_cosmo_warns(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]
    ) -> None:
        _, uvp_nocosmo, _ = uvp_for_uvwindow
        with pytest.warns(UserWarning, match="uvp has no cosmo attribute"):
            _ = uvwindow.UVWindow.from_uvpspec(
                uvp_nocosmo, ipol=0, spw=0, verbose=True, ftbeam=DATA_PATH / basename
            )

    def test_no_ftbeam_not_implemented(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        with pytest.raises(NotImplementedError, match="Construct the FTBeam"):
            uvwindow.UVWindow.from_uvpspec(
                uvp=uvp, ipol=0, spw=0, ftbeam=None, verbose=False
            )

    def test_wrong_ftbeam_type(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        with pytest.raises(TypeError, match="Check your ftbeam input"):
            uvwindow.UVWindow.from_uvpspec(
                uvp=uvp, ipol=0, spw=0, ftbeam=np.zeros(12), verbose=False
            )

    def test_spw_out_of_range(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec]
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        with pytest.raises(ValueError, match="Input spw must be smaller or equal"):
            uvwindow.UVWindow.from_uvpspec(
                uvp=uvp, ipol=0, spw=2, ftbeam=DATA_PATH / basename
            )

    def test_ftbeam_object(
        self,
        uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec],
        ft_beam_spw: uvwindow.FTBeam,
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        gaussian_beam = uvwindow.FTBeam.gaussian(
            freq_array=freq_array, widths=widths, pol="xx"
        )
        _ = uvwindow.UVWindow.from_uvpspec(
            uvp, ipol=0, spw=0, verbose=True, ftbeam=gaussian_beam
        )


class TestUVWindowGetKgrid:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        _ = uvwindow_obj._get_kgrid(red_bl_lens[12])

    def test_too_narrow_width(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        with pytest.raises(
            ValueError, match="Change width to resolve full window function"
        ):
            uvwindow_obj._get_kgrid(bl_len=red_bl_lens[12], width=0.0004)


class TestUVWindowKperp4blFreq:
    def test_happy_path(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam],
    ) -> None:
        bl_len = red_bl_lens[12]
        ngrid = make_ft_beam_obj().ft_beam.shape[-1]
        _ = uvwindow_obj._kperp4bl_freq(
            freq=uvwindow_obj.freq_array[12], bl_len=bl_len, ngrid=ngrid
        )

    def test_outside_spectral_window(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam],
    ) -> None:
        ngrid = make_ft_beam_obj().ft_beam.shape[-1]
        with pytest.raises(ValueError, match="Choose frequency within spectral window"):
            uvwindow_obj._kperp4bl_freq(
                freq=1.35e8, bl_len=red_bl_lens[12], ngrid=ngrid
            )

    def test_not_in_hz(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        make_ft_beam_obj: Callable[[tuple[int, int] | None], uvwindow.FTBeam],
    ) -> None:
        ngrid = make_ft_beam_obj().ft_beam.shape[-1]
        with pytest.raises(ValueError, match="Frequency must be given in Hz"):
            uvwindow_obj._kperp4bl_freq(
                freq=uvwindow_obj.freq_array[12] / 1e6,
                bl_len=red_bl_lens[12],
                ngrid=ngrid,
            )


class TestUVWindowInterpolateFtBeam:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        _ = uvwindow_obj._interpolate_ft_beam(red_bl_lens[12], ft_beam)

    def test_not_3d(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        with pytest.raises(ValueError, match="ft_beam must be dimension 3"):
            uvwindow_obj._interpolate_ft_beam(
                bl_len=red_bl_lens[12], ft_beam=ft_beam[0, :, :]
            )

    @pytest.mark.parametrize("bad_slice", ["truncated", "transposed"])
    def test_wrong_shape(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, bad_slice: str
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        sliced = ft_beam[0:10, :, :] if bad_slice == "truncated" else ft_beam[:, :, :].T
        with pytest.raises(ValueError, match="ft_beam must have shape"):
            uvwindow_obj._interpolate_ft_beam(bl_len=red_bl_lens[12], ft_beam=sliced)

    def test_not_square(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        with pytest.raises(ValueError, match="ft_beam must be square in sky plane"):
            uvwindow_obj._interpolate_ft_beam(
                bl_len=red_bl_lens[12], ft_beam=ft_beam[:, :1, :]
            )


class TestUVWindowTakeFreqFT:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        interp_ft_beam, _ = uvwindow_obj._interpolate_ft_beam(red_bl_lens[12], ft_beam)
        delta_nu = np.median(np.diff(uvwindow_obj.freq_array))
        _ = uvwindow_obj._take_freq_FT(interp_ft_beam, delta_nu)

    def test_not_3d(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        interp_ft_beam, _ = uvwindow_obj._interpolate_ft_beam(red_bl_lens[12], ft_beam)
        delta_nu = np.median(np.diff(uvwindow_obj.freq_array))
        with pytest.raises(ValueError, match="interp_ft_beam must be dimension 3"):
            uvwindow_obj._take_freq_FT(interp_ft_beam[0, :, :], delta_nu)

    def test_wrong_shape(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
        interp_ft_beam, _ = uvwindow_obj._interpolate_ft_beam(red_bl_lens[12], ft_beam)
        delta_nu = np.median(np.diff(uvwindow_obj.freq_array))
        with pytest.raises(ValueError, match="interp_ft_beam must have shape"):
            uvwindow_obj._take_freq_FT(interp_ft_beam[:, :, :].T, delta_nu)


class TestUVWindowGetWfForTau:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        bl_len = red_bl_lens[12]
        tau = uvwindow_obj.dly_array[12]
        kperp_bins = np.array(uvwindow_obj.get_kperp_bins([bl_len]).value)
        kpara_bins = np.array(
            uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array).value
        )
        wf_array = np.zeros((kperp_bins.size, uvwindow_obj.Nfreqs))
        _ = uvwindow_obj._get_wf_for_tau(tau, wf_array, kperp_bins, kpara_bins)


class TestUVWindowGetKperpBins:
    def test_empty_list_error(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        with pytest.raises(
            ValueError, match="get_kperp_bins\\(\\) requires array of baseline lengths"
        ):
            uvwindow_obj.get_kperp_bins(bl_lens=[])

    def test_scalar_returns_units(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        kperps = uvwindow_obj.get_kperp_bins(red_bl_lens[12])
        assert uvwindow_obj.kunits.is_equivalent(kperps.unit)

    def test_array(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        _ = uvwindow_obj.get_kperp_bins(red_bl_lens)

    def test_large_array_warns(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        with pytest.warns(UserWarning, match="Large number of kperp/kpara bins"):
            _ = uvwindow_obj.get_kperp_bins(np.r_[1.0, red_bl_lens])


class TestUVWindowGetKparaBins:
    def test_scalar_error(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        with pytest.raises(ValueError, match="Must feed list of frequencies"):
            uvwindow_obj.get_kpara_bins(freq_array=uvwindow_obj.freq_array[2])

    def test_happy_path(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        _ = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)

    def test_returns_units(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        kparas = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        assert uvwindow_obj.kunits.is_equivalent(kparas.unit)

    def test_large_bandwidth_warns(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        HERA_bw = np.linspace(1, 2, 1024, endpoint=False) * 1e8
        with pytest.warns(UserWarning, match="Large number of kperp/kpara bins"):
            _ = uvwindow_obj.get_kpara_bins(HERA_bw)


class TestUVWindowGetCylindricalWf:
    def test_return_bins_weighted(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        _, _, cyl_wf = uvwindow_obj.get_cylindrical_wf(
            red_bl_lens[12], kperp_bins=None, kpara_bins=None, return_bins="weighted"
        )
        assert cyl_wf is not None

    def test_return_bins_none(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        result = uvwindow_obj.get_cylindrical_wf(
            red_bl_lens[12], kperp_bins=None, kpara_bins=None, return_bins=None
        )
        assert result is not None

    def test_normalisation(
        self, cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        _, _, _, cyl_wf = cyl_wf_result
        assert np.allclose(np.sum(cyl_wf, axis=(1, 2)), 1.0, atol=1e-3)

    def test_output_shapes(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        assert kperp.size == cyl_wf.shape[1]
        assert kpara.size == cyl_wf.shape[2]
        assert uvwindow_obj.Nfreqs == cyl_wf.shape[0]

    def test_bins_consistent_with_getters(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, _ = cyl_wf_result
        assert np.allclose(kperp, uvwindow_obj.get_kperp_bins(bl_len).value)
        assert np.allclose(
            kpara, uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array).value
        )

    def test_custom_kperp_bins(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, _, cyl_wf = cyl_wf_result
        kperp2, _, cyl_wf2 = uvwindow_obj.get_cylindrical_wf(
            bl_len,
            kperp_bins=kperp * uvwindow_obj.kunits,
            kpara_bins=None,
            return_bins="unweighted",
        )
        assert np.allclose(cyl_wf2, cyl_wf)
        assert np.allclose(kperp2, kperp)

    def test_custom_kpara_bins(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        _, kpara3, cyl_wf3 = uvwindow_obj.get_cylindrical_wf(
            bl_len,
            kperp_bins=None,
            kpara_bins=kpara * uvwindow_obj.kunits,
            return_bins="unweighted",
        )
        assert np.allclose(cyl_wf3, cyl_wf)
        assert np.allclose(kpara, kpara3)

    def test_nonlinear_kperp_error(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        with pytest.raises(
            ValueError, match="get_cylindrical_wf: kperp_bins must be linearly spaced"
        ):
            uvwindow_obj.get_cylindrical_wf(
                red_bl_lens[12],
                kperp_bins=np.logspace(-2, 0, 100) * uvwindow_obj.kunits,
                kpara_bins=None,
                return_bins="unweighted",
            )

    def test_nonlinear_kpara_error(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        with pytest.raises(
            ValueError, match="get_cylindrical_wf: kpara_bins must be linearly spaced"
        ):
            uvwindow_obj.get_cylindrical_wf(
                red_bl_lens[12],
                kperp_bins=None,
                kpara_bins=np.logspace(-1, 1, 100) * uvwindow_obj.kunits,
                return_bins="unweighted",
            )

    def test_odd_number_of_delays(
        self, red_bl_lens: np.ndarray, ft_bandwidth: np.ndarray
    ) -> None:
        ft_beam_test = uvwindow.FTBeam.from_file(
            ftfile=DATA_PATH / ftfile, freq_array=ft_bandwidth[5:24]
        )
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_test)
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(
            red_bl_lens[12], return_bins="unweighted"
        )
        assert cyl_wf is not None


class TestUVWindowCylindricalToSpherical:
    def test_with_weights(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        _ = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=[2.0]
        )

    def test_no_weights(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        _ = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf[None], kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=None
        )

    def test_ktot_shape_mismatch(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        with pytest.raises(ValueError, match="k magnitude grid does not match"):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf,
                kbins=kbins,
                ktot=np.sqrt(kperp[:-2, None] ** 2 + kpara**2),
                bl_lens=bl_len,
            )

    def test_single_kbin_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        with pytest.raises(ValueError, match="must feed array of k bins"):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf, kbins=kbins[:1], ktot=ktot, bl_lens=bl_len
            )

    def test_weights_mismatch(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        with pytest.raises(ValueError, match="Blpair weights and lengths do not match"):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf,
                kbins=kbins,
                ktot=ktot,
                bl_lens=bl_len,
                bl_weights=[1.0, 2.0],
            )

    def test_single_bl_lens(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        _, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        with pytest.raises(ValueError, match="If only one bl_lens is given,"):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=red_bl_lens
            )

    def test_bl_lens_mismatch(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        with pytest.raises(ValueError, match="bl_lens size must match cyl_wf.shape"):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf[None],
                kbins=kbins,
                ktot=ktot,
                bl_lens=red_bl_lens[:2],
                bl_weights=[1.0, 2.0],
            )

    def test_empty_bins_warns(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        kbins_test = np.arange(2, 5, step=0.5) * uvwindow_obj.kunits
        uvwindow_obj.verbose = True
        with pytest.warns(UserWarning, match="Some spherical bins are empty"):
            _ = uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf, kbins=kbins_test, ktot=ktot, bl_lens=bl_len
            )
        uvwindow_obj.verbose = False

    def test_nonlinear_kbins_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        with pytest.raises(
            ValueError, match="cylindrical_to_spherical: kbins must be linearly spaced"
        ):
            uvwindow_obj.cylindrical_to_spherical(
                cyl_wf=cyl_wf,
                kbins=np.logspace(-2, 2, 20) * uvwindow_obj.kunits,
                ktot=ktot,
                bl_lens=bl_len,
            )


class TestUVWindowGetSphericalWf:
    @pytest.mark.parametrize(
        "kbins", [np.arange(1.0, 5.0, step=0.3), np.arange(0.1, 3.0, step=0.03)]
    )
    def test_minmax_k_warning(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        with pytest.warns(
            UserWarning, match="spherical k probed is not included in bins"
        ):
            _ = uvwindow_obj.get_spherical_wf(
                kbins=kbins * units.h / units.Mpc,
                bl_lens=red_bl_lens[:1],
                bl_weights=[1],
                kperp_bins=None,
                kpara_bins=None,
                return_weighted_k=True,
                verbose=True,
            )

    def test_happy_path(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        kperp_bins = uvwindow_obj.get_kperp_bins(red_bl_lens[:1])
        kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        dk = np.diff(kbins.value).mean()
        ktot_max = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2).max()
        full_kbins = (
            np.arange(kbins.value.min(), ktot_max + dk, step=dk) * uvwindow_obj.kunits
        )
        _ = uvwindow_obj.get_spherical_wf(
            kbins=full_kbins,
            kperp_bins=kperp_bins,
            kpara_bins=kpara_bins,
            bl_lens=red_bl_lens[:1],
            bl_weights=None,
            return_weighted_k=False,
            verbose=None,
        )

    def test_kbins_no_units_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        with pytest.raises(AttributeError, match="Feed k array with units"):
            uvwindow_obj.get_spherical_wf(kbins=kbins.value, bl_lens=red_bl_lens[:2])

    def test_weights_mismatch_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        with pytest.raises(
            ValueError, match="bl_weights and bl_lens must have same length"
        ):
            uvwindow_obj.get_spherical_wf(
                kbins=kbins, bl_lens=red_bl_lens[:2], bl_weights=[1.0]
            )

    def test_single_kbin_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        with pytest.raises(ValueError, match="must feed array of k bins"):
            uvwindow_obj.get_spherical_wf(
                kbins=kbins.value[2] * uvwindow_obj.kunits, bl_lens=red_bl_lens[:1]
            )

    def test_kpara_outside_window_warns(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
    ) -> None:
        kperp_bins = uvwindow_obj.get_kperp_bins(red_bl_lens[:1])
        kpara_centre = (
            uvwindow_obj.cosmo.tau_to_kpara(
                uvwindow_obj.avg_z, little_h=uvwindow_obj.little_h
            )
            * abs(uvwindow_obj.dly_array).max()
        )
        bad_kpara_bins = (
            np.arange(2.0 * kpara_centre, 10 * kpara_centre, step=kpara_centre)
            * uvwindow_obj.kunits
        )
        dk = np.diff(kbins.value).mean()
        bad_kmax = np.sqrt(
            kperp_bins.value[:, None] ** 2 + bad_kpara_bins.value**2
        ).max()
        bad_full_kbins = (
            np.arange(kbins.value.min(), bad_kmax + dk, step=dk) * uvwindow_obj.kunits
        )
        with pytest.warns(
            UserWarning,
            match="The bin centre is not included in the array of kpara bins",
        ):
            _ = uvwindow_obj.get_spherical_wf(
                kbins=bad_full_kbins,
                kperp_bins=kperp_bins,
                kpara_bins=bad_kpara_bins,
                bl_lens=red_bl_lens[:1],
            )

    @pytest.mark.parametrize("ktype", ["wrong_min", "wrong_max"])
    def test_wrong_kperp_bins(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
        ktype: str,
    ) -> None:
        kperp_bins = [0, 1e-12] if ktype == "wrong_min" else [2e12, 3e12]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Max spherical k probed is not included in bins.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="get_cylindrical_wf: The bin centre is not included in the array of kperp bins given as input.",
                category=UserWarning,
            )
            with pytest.warns(
                UserWarning, match="kperp bin centre not included in binning array"
            ):
                _ = uvwindow_obj.get_spherical_wf(
                    kbins=kbins,
                    bl_lens=red_bl_lens[:1],
                    bl_weights=[1],
                    kperp_bins=kperp_bins * uvwindow_obj.kunits,
                )

    @pytest.mark.parametrize(
        "bad_kwarg,error_msg",
        [
            ("kperp_bins", "get_spherical_wf: kperp_bins must be linearly spaced"),
            ("kpara_bins", "get_spherical_wf: kpara_bins must be linearly spaced"),
            ("kbins", "get_spherical_wf: kbins must be linearly spaced"),
        ],
    )
    def test_nonlinear_bins_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        kbins: units.Quantity,
        bad_kwarg: str,
        error_msg: str,
    ) -> None:
        bad_bins = np.logspace(-2, 2, 20) * uvwindow_obj.kunits
        if bad_kwarg == "kbins":
            call_kw = {"kbins": bad_bins, "bl_lens": red_bl_lens[:1]}
        else:
            call_kw = {"kbins": kbins, bad_kwarg: bad_bins, "bl_lens": red_bl_lens[:1]}
        with pytest.raises(ValueError, match=error_msg):
            uvwindow_obj.get_spherical_wf(**call_kw)


class TestUVWindowCheckKunits:
    def test_with_units(
        self, uvwindow_obj: uvwindow.UVWindow, kbins: units.Quantity
    ) -> None:
        uvwindow_obj.check_kunits(kbins)

    def test_without_units_raises(
        self, uvwindow_obj: uvwindow.UVWindow, kbins: units.Quantity
    ) -> None:
        with pytest.raises(AttributeError, match="Feed k array with units"):
            uvwindow_obj.check_kunits(kbins.value)


class TestUVWindowRunAndWrite:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        kperp_bins = uvwindow_obj.get_kperp_bins(red_bl_lens[:1])
        kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile),
            bl_lens=red_bl_lens[:1],
            bl_weights=[1.0],
            kperp_bins=kperp_bins,
            kpara_bins=kpara_bins,
            clobber=False,
        )

    def test_no_taper(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        uvw = copy.deepcopy(uvwindow_obj)
        uvwindow_obj.taper = None
        kperp_bins = uvw.get_kperp_bins(red_bl_lens[:1])
        kpara_bins = uvw.get_kpara_bins(uvw.freq_array)
        uvw.run_and_write(
            filepath=str(tmp_path / outfile),
            bl_lens=red_bl_lens[:1],
            bl_weights=[1.0],
            kperp_bins=kperp_bins,
            kpara_bins=kpara_bins,
            clobber=False,
        )

    def test_clobber_false_raises(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile), bl_lens=red_bl_lens[:1], clobber=True
        )
        with pytest.raises(IOError, match="exists, not overwriting"):
            uvwindow_obj.run_and_write(
                filepath=str(tmp_path / outfile), bl_lens=red_bl_lens[:1], clobber=False
            )

    def test_clobber_true_overwrites(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile), bl_lens=red_bl_lens[:1], clobber=True
        )
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile),
            bl_lens=[red_bl_lens[:1]],
            kperp_bins=None,
            kpara_bins=None,
            clobber=True,
        )

    def test_weights_mismatch_error(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ValueError, match="bl_weights and bl_lens must have same length"
        ):
            uvwindow_obj.run_and_write(
                filepath=str(tmp_path / outfile),
                bl_lens=red_bl_lens[:1],
                bl_weights=[1.0, 1.0],
                clobber=True,
            )

    @pytest.mark.parametrize("bad_kwarg", ["kperp_bins", "kpara_bins"])
    def test_kperp_no_units_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        tmp_path: Path,
        bad_kwarg: str,
    ) -> None:
        kperp_bins = uvwindow_obj.get_kperp_bins(red_bl_lens[:1])
        kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="get_cylindrical_wf: The bin centre is not included in the array of kperp bins given as input.",
                category=UserWarning,
            )
            with pytest.raises(AttributeError, match="Feed k array with units"):
                uvwindow_obj.run_and_write(
                    filepath=str(tmp_path / outfile),
                    bl_lens=red_bl_lens[:1],
                    kperp_bins=kperp_bins.value
                    if bad_kwarg == "kperp_bins"
                    else kperp_bins,
                    kpara_bins=kpara_bins.value
                    if bad_kwarg == "kpara_bins"
                    else kpara_bins,
                    clobber=True,
                )

    def test_k_wrong_units_error(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        with pytest.raises(
            ValueError, match="k array units not consistent with little_h"
        ):
            uvwindow_obj.run_and_write(
                filepath=str(tmp_path / outfile),
                bl_lens=red_bl_lens[:1],
                kpara_bins=kpara_bins.value * units.Mpc,
                clobber=True,
            )


class TestFTBeamSelectFreqs:
    def test_exact_match_extracts_channels(self, ft_bandwidth: np.ndarray) -> None:
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        sub = full.select_freqs(ft_bandwidth[3:10], inplace=False)
        assert np.allclose(sub.freq_array, ft_bandwidth[3:10])
        assert np.array_equal(sub.ft_beam, full.ft_beam[3:10])
        # with inplace=False the original is untouched
        assert full.freq_array.size == ft_bandwidth.size

    def test_mismatched_grid_interpolates_with_warning(
        self, ft_bandwidth: np.ndarray
    ) -> None:
        # midpoints of a linear grid: linear interpolation is exactly the
        # average of the neighbouring channels
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        midpoints = 0.5 * (ft_bandwidth[3:10] + ft_bandwidth[4:11])
        with pytest.warns(UserWarning, match="interpolating"):
            interp = full.select_freqs(midpoints, inplace=False)
        assert np.allclose(interp.freq_array, midpoints)
        assert np.allclose(
            interp.ft_beam, 0.5 * (full.ft_beam[3:10] + full.ft_beam[4:11])
        )

    def test_out_of_coverage_raises(self, ft_bandwidth: np.ndarray) -> None:
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        with pytest.raises(ValueError, match="beyond the bandwidth"):
            full.select_freqs(np.array([ft_bandwidth.max() + 1e6] * 3))

    def test_single_frequency_raises(self, ft_bandwidth: np.ndarray) -> None:
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        with pytest.raises(ValueError, match="at least two frequencies"):
            full.select_freqs(ft_bandwidth[:1])


class TestFTBeamFromFileFreqArray:
    def test_partial_read_matches_full_read(self, ft_bandwidth: np.ndarray) -> None:
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        target = ft_bandwidth[5:25]
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, freq_array=target)
        assert np.allclose(test.freq_array, target)
        assert np.array_equal(test.ft_beam, full.ft_beam[5:25])

    def test_both_selections_raise(self, ft_bandwidth: np.ndarray) -> None:
        with pytest.raises(ValueError, match="both spw_range and freq_array"):
            uvwindow.FTBeam.from_file(
                ftfile=DATA_PATH / ftfile,
                spw_range=(5, 25),
                freq_array=ft_bandwidth[5:25],
            )

    def test_interpolation_through_partial_read(self, ft_bandwidth: np.ndarray) -> None:
        # requested frequencies fall between the file's channels, so
        # select_freqs interpolates using only the padded channel range
        # read from disk; must equal loading the whole file and
        # interpolating. Starts at the first channel pair to stress the
        # edge padding.
        full = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)
        midpoints = 0.5 * (ft_bandwidth[0:8] + ft_bandwidth[1:9])
        with pytest.warns(UserWarning, match="interpolating"):
            part = uvwindow.FTBeam.from_file(
                ftfile=DATA_PATH / ftfile, freq_array=midpoints
            )
        with pytest.warns(UserWarning, match="interpolating"):
            ref = full.select_freqs(midpoints, inplace=False)
        assert np.allclose(part.freq_array, ref.freq_array)
        assert np.allclose(part.ft_beam, ref.ft_beam)


class TestFTBeamWriteHdf5:
    def test_roundtrip(self, ft_beam_spw: uvwindow.FTBeam, tmp_path: Path) -> None:
        # the filename carries no pol suffix: the round trip relies on the
        # pol attribute written by write_hdf5
        fname = tmp_path / "ft_beam_roundtrip.hdf5"
        ft_beam_spw.write_hdf5(fname, extra_attrs={"beam_file": "sim.fits"})
        back = uvwindow.FTBeam.from_file(ftfile=fname)
        assert back == ft_beam_spw

    def test_no_overwrite_by_default(
        self, ft_beam_spw: uvwindow.FTBeam, tmp_path: Path
    ) -> None:
        fname = tmp_path / "ft_beam.hdf5"
        ft_beam_spw.write_hdf5(fname)
        with pytest.raises(FileExistsError, match="overwrite"):
            ft_beam_spw.write_hdf5(fname)
        ft_beam_spw.write_hdf5(fname, overwrite=True)


class TestFTBeamDataGridMismatch:
    """Regression tests: an FTBeam on a frequency grid with a different
    channel width than the data must yield the same window functions as an
    FTBeam defined directly on the data channels (historically, the window
    functions' k_parallel axis silently came out scaled by the ratio of
    the two channel widths)."""

    # "data" channels: 500 kHz; FT beam computed on 400 kHz channels
    freqs_data = np.linspace(155e6, 165e6, 20, endpoint=False)
    freqs_beam = np.arange(150e6, 170e6, 0.4e6)

    @staticmethod
    def _gaussian(freqs: np.ndarray) -> uvwindow.FTBeam:
        widths = -0.0343 * freqs / 1e6 + 11.30
        return uvwindow.FTBeam.gaussian(freqs, widths, pol="xx", npix=101)

    def test_delay_grid_matches_data_channel_width(self) -> None:
        with pytest.warns(UserWarning, match="interpolating"):
            ftb = self._gaussian(self.freqs_beam).select_freqs(
                self.freqs_data, inplace=False
            )
        uvw = uvwindow.UVWindow(ftbeam_obj=ftb, taper="blackman-harris")
        assert np.allclose(uvw.freq_array, self.freqs_data)
        assert np.isclose(np.median(np.diff(uvw.dly_array)), 1.0 / (20 * 0.5e6))

    def test_cylindrical_wf_matches_direct_grid(self) -> None:
        bl_len = 15.0
        with pytest.warns(UserWarning, match="interpolating"):
            ftb_interp = self._gaussian(self.freqs_beam).select_freqs(
                self.freqs_data, inplace=False
            )
        uvw = uvwindow.UVWindow(ftbeam_obj=ftb_interp, taper="blackman-harris")
        uvw_direct = uvwindow.UVWindow(
            ftbeam_obj=self._gaussian(self.freqs_data), taper="blackman-harris"
        )
        kperp_bins = uvw_direct.get_kperp_bins([bl_len])
        kpara_bins = uvw_direct.get_kpara_bins(self.freqs_data)
        wf_interp = uvw.get_cylindrical_wf(
            bl_len, kperp_bins=kperp_bins, kpara_bins=kpara_bins
        )
        wf_direct = uvw_direct.get_cylindrical_wf(
            bl_len, kperp_bins=kperp_bins, kpara_bins=kpara_bins
        )
        # up to interpolation accuracy of the slowly-varying gaussian beam
        assert np.allclose(wf_interp, wf_direct, atol=1e-4)


class TestGetWfForTauKnownExamples:
    """Check the k_parallel binning at its two granularity extremes.

    At the coarsest extreme (one bin wide enough to hold the whole band)
    binning must reduce to a plain mean over frequency; at the finest
    (one bin per k_parallel value) it must return the input unchanged.
    Together these pin the bin assignment, the averaging and the
    empty-bin behavior with answers that can be checked by hand.
    """

    @staticmethod
    def _kpar_norm(uvw: uvwindow.UVWindow, tau: float) -> np.ndarray:
        alpha = uvw.cosmo.dRpara_df(uvw.avg_z, little_h=uvw.little_h, ghz=False)
        delta_nu = np.median(np.diff(uvw.freq_array))
        eta = np.fft.fftshift(np.fft.fftfreq(uvw.Nfreqs)) / delta_nu
        return np.abs(2.0 * np.pi / alpha * (eta + tau))

    def test_single_bin_gets_row_means(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        # coarsest extreme: one bin holds every kpar value (flanked by
        # empty bins), so its column must be the plain frequency-mean of
        # each row of wf_array1, and the empty bins must stay zero
        uvw = uvwindow_obj
        # any tau works here; 2/delta_nu keeps kpar_norm strictly positive
        tau = 2.0 / np.median(np.diff(uvw.freq_array))
        kpar = self._kpar_norm(uvw, tau)
        dk = (kpar.max() - kpar.min()) + 1.0
        kpara_bins = np.array([kpar.mean() - dk, kpar.mean(), kpar.mean() + dk])
        kperp_bins = np.arange(1.0, 6.0)
        rng = np.random.default_rng(0)
        wf1 = rng.random((kperp_bins.size, uvw.Nfreqs))

        kpara_out, cyl = uvw._get_wf_for_tau(tau, wf1, kperp_bins, kpara_bins)
        assert np.allclose(cyl[:, 1], wf1.mean(axis=1))
        assert np.allclose(cyl[:, [0, 2]], 0.0)
        assert np.isclose(kpara_out[1], kpar.mean())
        assert np.allclose(kpara_out[[0, 2]], 0.0)

    def test_resolved_bins_are_identity(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        # finest extreme: bins centred on each kpar value, so binning
        # must be the identity. tau = 2/delta_nu exceeds max|eta| =
        # 1/(2 delta_nu), keeping eta + tau positive so kpar_norm
        # inherits eta's equal spacing (each value lands in its own bin)
        uvw = uvwindow_obj
        tau = 2.0 / np.median(np.diff(uvw.freq_array))
        kpar = self._kpar_norm(uvw, tau)
        assert np.all(np.diff(kpar) > 0)  # monotonic by construction
        kpara_bins = kpar  # equally spaced since eta is
        kperp_bins = np.arange(1.0, 6.0)
        rng = np.random.default_rng(1)
        wf1 = rng.random((kperp_bins.size, uvw.Nfreqs))

        kpara_out, cyl = uvw._get_wf_for_tau(tau, wf1, kperp_bins, kpara_bins)
        assert np.allclose(cyl, wf1)
        assert np.allclose(kpara_out, kpar)

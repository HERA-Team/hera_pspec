import copy
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from pyuvdata.analytic_beam import GaussianBeam
from uvtools import dspec

from hera_pspec import PSpecData, UVPSpec, conversions, utils, uvwindow
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Data files to use in tests
dfile = "zen.2458116.31939.HH.uvh5"
ftfile = "FT_beam_HERA_dipole_test_xx.hdf5"
basename = "FT_beam_HERA_dipole_test"
outfile = "test.hdf5"

# Spectral window ranges that FTBeam must reject: wrong length, reversed, and
# outside the band
BAD_SPWS = [(13,), (20, 10), (1001, 1022)]


# Channels of the HERA band (100-200 MHz in 1024 channels) covered by the FT beam
# file, and the 20 channels used by most tests. The latter match the spectral
# window (175, 195) of the UVPSpecs built from dfile below.
FT_FILE_FREQS = 1e8 + np.arange(170, 200) * 1e8 / 1024
FT_FREQS = FT_FILE_FREQS[5:25]


def gaussian_ft_beam(freq_array: np.ndarray) -> uvwindow.FTBeam:
    """Analytic FTBeam of a Gaussian beam with a HERA-like width (about 7 deg),
    on the same flat-sky grid size as the FT beam file."""
    widths = -0.0343 * freq_array / 1e6 + 11.30
    return uvwindow.FTBeam.gaussian(
        freq_array=freq_array, widths=widths, pol="xx", mapsize=0.5, npix=213
    )


# Most tests only need *some* valid FTBeam and use the analytic one; the tests
# of reading the file use ft_beam_file. Reading the file is slow, so it is read
# once per module. Tests that modify an FTBeam or UVWindow fixture must
# deepcopy it.
@pytest.fixture(scope="module")
def ft_beam_file() -> uvwindow.FTBeam:
    """FTBeam read from the whole FT beam file (channels FT_FILE_FREQS)."""
    return uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile)


@pytest.fixture(scope="module")
def ft_beam_spw() -> uvwindow.FTBeam:
    """Analytic Gaussian FTBeam on the 20 channels FT_FREQS."""
    return gaussian_ft_beam(FT_FREQS)


@pytest.fixture()
def ft_beam_full() -> uvwindow.FTBeam:
    """Fresh analytic Gaussian FTBeam on all the channels of the FT beam file."""
    return gaussian_ft_beam(FT_FILE_FREQS)


@pytest.fixture()
def ft_bandwidth() -> np.ndarray:
    return uvwindow.FTBeam.get_bandwidth(DATA_PATH / ftfile)


@pytest.fixture(scope="module")
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
    krange = np.arange(0.1, 5.0, step=0.3)
    return ((krange[1:] + krange[:-1]) / 2) * units.h / units.Mpc


@pytest.fixture(scope="module")
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


@pytest.fixture()
def ft_beam_init_kwargs(ft_beam_spw: uvwindow.FTBeam) -> dict:
    """FTBeam constructor arguments that rebuild ft_beam_spw from its arrays."""
    return dict(
        data=ft_beam_spw.ft_beam,
        pol="xx",
        freq_array=ft_beam_spw.freq_array,
        mapsize=ft_beam_spw.mapsize,
    )


class TestFTBeamInit:
    def test_from_array(self, ft_beam_init_kwargs: dict) -> None:
        test = uvwindow.FTBeam(
            **ft_beam_init_kwargs, verbose=False, x_orientation="east"
        )
        assert test.pol == "xx"
        assert np.allclose(ft_beam_init_kwargs["data"], test.ft_beam)

    @pytest.mark.parametrize(
        "slice_data,match",
        [
            pytest.param(
                lambda data: data[:, :, 0], "Wrong dimensions for data input", id="2d"
            ),
            pytest.param(
                lambda data: data[:, :, :-1],
                "Wrong dimensions for data input",
                id="wrong_shape",
            ),
            pytest.param(
                lambda data: data[:12, :, :], "data must have shape", id="freq_mismatch"
            ),
        ],
    )
    def test_invalid_data_shape(
        self,
        ft_beam_init_kwargs: dict,
        slice_data: Callable[[np.ndarray], np.ndarray],
        match: str,
    ) -> None:
        data = slice_data(ft_beam_init_kwargs["data"])
        with pytest.raises(ValueError, match=match):
            uvwindow.FTBeam(**{**ft_beam_init_kwargs, "data": data})

    def test_int_pol(self, ft_beam_init_kwargs: dict) -> None:
        test = uvwindow.FTBeam(**{**ft_beam_init_kwargs, "pol": -5})
        assert test.pol == uvutils.polnum2str(-5)

    @pytest.mark.parametrize(
        "bad_pol,exc_type,match",
        [
            ("test", ValueError, "Wrong polarisation"),
            (12, ValueError, "Wrong polarisation"),
            (3.4, TypeError, "Must feed pol as str or int"),
        ],
        ids=["unknown_str", "unknown_int", "float"],
    )
    def test_invalid_pol(
        self,
        ft_beam_init_kwargs: dict,
        bad_pol: str | int | float,
        exc_type: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(exc_type, match=match):
            uvwindow.FTBeam(**{**ft_beam_init_kwargs, "pol": bad_pol})


nf_dipole_beamfits = DATA_PATH / "HERA_NF_dipole_power.beamfits"


@pytest.fixture(scope="module")
def analytic_beamfits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Beamfits files of one analytic Gaussian beam, as an efield and as a power
    beam (keyed by beam type), on an NSIDE=16 HEALPix map at 120-180 MHz."""
    tmp_path = tmp_path_factory.mktemp("analytic_beams")
    files = {}
    for beam_type in ["efield", "power"]:
        uvb = GaussianBeam(sigma=0.15).to_uvbeam(
            freq_array=np.linspace(120e6, 180e6, 3),
            beam_type=beam_type,
            pixel_coordinate_system="healpix",
            nside=16,
        )
        files[beam_type] = tmp_path / f"gaussian_{beam_type}.beamfits"
        uvb.write_beamfits(files[beam_type])
    return files


@pytest.fixture(scope="module")
def nf_dipole_beam_freqs(beam_nf_dipole) -> np.ndarray:
    # beam_nf_dipole is the session-cached PSpecBeamUV of nf_dipole_beamfits
    return np.unique(beam_nf_dipole.primary_beam.freq_array)


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

    def test_attributes(
        self, small_ft_beam: uvwindow.FTBeam, nf_dipole_beam_freqs: np.ndarray
    ) -> None:
        assert small_ft_beam.pol == "xx"
        assert small_ft_beam.mapsize == 1.0
        np.testing.assert_array_equal(
            small_ft_beam.freq_array,
            np.linspace(nf_dipole_beam_freqs.min(), nf_dipole_beam_freqs.max(), 5),
        )
        assert small_ft_beam.ft_beam.ndim == 3
        assert small_ft_beam.ft_beam.shape[0] == small_ft_beam.freq_array.size == 5
        assert small_ft_beam.ft_beam.shape[1] == small_ft_beam.ft_beam.shape[2]
        assert np.all(np.isfinite(small_ft_beam.ft_beam))

    def test_ft_peaks_at_zero_mode(self, small_ft_beam: uvwindow.FTBeam) -> None:
        # the FT of a positive beam peaks at the zero mode (grid centre)
        ngrid = small_ft_beam.ft_beam.shape[-1]
        for i in range(small_ft_beam.freq_array.size):
            assert np.argmax(small_ft_beam.ft_beam[i]) == (ngrid**2) // 2

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
        # ...which is not trivially true: the beam changes with frequency
        assert not np.allclose(test.ft_beam[1], test.ft_beam[2])

    @pytest.mark.parametrize(
        "override",
        [
            pytest.param({"pol": -5}, id="pol_as_int"),
            # npix=30 is adjusted down to 29
            pytest.param({"npix": 30}, id="even_npix_adjusted_to_odd"),
        ],
    )
    def test_equivalent_arguments(
        self, small_ft_beam: uvwindow.FTBeam, override: dict
    ) -> None:
        """Check that equivalent from_beam arguments reproduce small_ft_beam exactly."""
        kwargs = dict(
            beamfile=nf_dipole_beamfits,
            pol="xx",
            freq_array=small_ft_beam.freq_array,
            mapsize=1.0,
            npix=29,
        )
        test = uvwindow.FTBeam.from_beam(**{**kwargs, **override})
        assert test.pol == "xx"
        assert test == small_ft_beam

    @pytest.mark.parametrize(("mapsize", "ngrid"), [(0.35, 15), (uvwindow._MAX_R, 29)])
    def test_mapsize_within_beam_coverage(
        self, small_ft_beam: uvwindow.FTBeam, mapsize: float, ngrid: int
    ) -> None:
        # mapsize < _MAX_R crops the beam map; mapsize = _MAX_R uses it as is
        test = uvwindow.FTBeam.from_beam(
            beamfile=nf_dipole_beamfits,
            pol="xx",
            freq_array=small_ft_beam.freq_array,
            mapsize=mapsize,
            npix=29,
        )
        # the 29-pixel map spans +-_MAX_R, so the (odd) FT grid holds
        # ~2 * mapsize / pixel width pixels
        assert test.ft_beam.shape == (small_ft_beam.freq_array.size, ngrid, ngrid)
        assert np.all(np.isfinite(test.ft_beam))

    @pytest.mark.parametrize("pol", ["xx", "pI"])
    def test_efield_beam_is_converted(
        self, analytic_beamfits: dict[str, Path], pol: str
    ) -> None:
        # efield beams are converted to power (xx) or pseudo-Stokes (pI). The
        # x and y feeds of the analytic beam have the same power response, so
        # once peak-normalized both equal the xx power beam of the same model.
        kwargs = dict(freq_array=np.linspace(120e6, 180e6, 3), mapsize=1.0, npix=29)
        test = uvwindow.FTBeam.from_beam(
            beamfile=analytic_beamfits["efield"], pol=pol, **kwargs
        )
        expected = uvwindow.FTBeam.from_beam(
            beamfile=analytic_beamfits["power"], pol="xx", **kwargs
        )
        assert test.pol == pol
        assert test.ft_beam.shape == expected.ft_beam.shape
        np.testing.assert_allclose(
            test.ft_beam, expected.ft_beam, atol=1e-12 * expected.ft_beam.max()
        )

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

    def test_gaussian_beam_has_analytic_transform(self, tmp_path: Path) -> None:
        """The flat-sky Fourier transform of a Gaussian power beam
        exp(-theta^2 / w^2) is pi w^2 exp(-pi^2 w^2 |u|^2), a Gaussian of width
        1 / (pi w) in u (the Fourier dual of the sky angle in radians), on a grid
        of spacing 1 / (2 mapsize). FTBeam stores it divided by 2 mapsize, the
        normalisation FTBeam.gaussian uses for the same beam. This pins the
        projection of the beam onto the flat sky, the Fourier grid and the
        normalisation of from_beam.

        pyuvdata's GaussianBeam(sigma) is an efield beam whose power beam is
        exp(-theta^2 / sigma^2), so w = sigma. On an NSIDE=64 HEALPix map the
        transform agrees with the analytic one to ~0.5% of its peak."""
        sigma, mapsize, freqs = 0.15, 1.0, np.linspace(120e6, 180e6, 3)
        beamfile = tmp_path / "gaussian.beamfits"
        GaussianBeam(sigma=sigma).to_uvbeam(
            freq_array=freqs,
            beam_type="efield",
            pixel_coordinate_system="healpix",
            nside=64,
        ).write_beamfits(beamfile)
        ftbeam = uvwindow.FTBeam.from_beam(
            beamfile=beamfile, pol="xx", freq_array=freqs, mapsize=mapsize, npix=141
        )
        ngrid = ftbeam.ft_beam.shape[-1]
        u = np.fft.fftshift(np.fft.fftfreq(ngrid)) * ngrid / (2 * mapsize)
        usq = u[:, None] ** 2 + u[None, :] ** 2
        expected = (
            np.pi * sigma**2 * np.exp(-(np.pi**2) * sigma**2 * usq) / (2 * mapsize)
        )
        for ft in ftbeam.ft_beam:
            np.testing.assert_allclose(ft, expected, rtol=0, atol=1e-2 * expected.max())
        # the analytic beam of FTBeam.gaussian is the same transform
        analytic = uvwindow.FTBeam.gaussian(
            freq_array=freqs,
            widths=np.degrees(sigma),
            pol="xx",
            mapsize=mapsize,
            npix=ngrid,
        )
        for ft in analytic.ft_beam:
            np.testing.assert_allclose(ft, expected, rtol=1e-10, atol=1e-300)


class TestFTBeamFromFile:
    def test_happy_path(self, ft_bandwidth: np.ndarray) -> None:
        test = uvwindow.FTBeam.from_file(
            ftfile=DATA_PATH / ftfile,
            freq_array=ft_bandwidth[5:25],
            verbose=False,
            x_orientation="east",
        )
        assert test.pol == "xx"
        assert test.x_orientation == "east"
        np.testing.assert_array_equal(test.freq_array, ft_bandwidth[5:25])

    def test_spw_range_selects_channels(self, ft_beam_file: uvwindow.FTBeam) -> None:
        # deprecated channel-index selection picks channels 5-24 of the file
        with pytest.warns(DeprecationWarning, match="spw_range"):
            test = uvwindow.FTBeam.from_file(
                ftfile=DATA_PATH / ftfile, spw_range=(5, 25)
            )
        np.testing.assert_allclose(test.freq_array, FT_FREQS)
        np.testing.assert_array_equal(test.ft_beam, ft_beam_file.ft_beam[5:25])

    def test_no_spw_range_uses_full_bandwidth(self, ft_bandwidth: np.ndarray) -> None:
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, spw_range=None)
        assert np.allclose(test.freq_array, ft_bandwidth)

    @pytest.mark.parametrize(
        "make_kwargs,exc_type,match",
        [
            # The TypeError comes from pathlib; its wording varies between
            # Python versions, so only the exception type is checked.
            pytest.param(
                lambda bandwidth: dict(ftfile=12.0),
                TypeError,
                None,
                id="ftfile_not_path_like",
            ),
            pytest.param(
                lambda bandwidth: dict(ftfile="whatever"),
                ValueError,
                "Wrong ftfile input",
                id="ftfile_missing",
            ),
            pytest.param(
                lambda bandwidth: dict(
                    ftfile=DATA_PATH / ftfile, freq_array=np.array([250e6, 251e6])
                ),
                ValueError,
                "beyond the bandwidth",
                id="freqs_beyond_bandwidth",
            ),
            pytest.param(
                lambda bandwidth: dict(
                    ftfile=DATA_PATH / ftfile, freq_array=np.array([100e6])
                ),
                ValueError,
                "at least two frequencies",
                id="single_frequency",
            ),
            pytest.param(
                lambda bandwidth: dict(
                    ftfile=DATA_PATH / ftfile,
                    spw_range=(5, 25),
                    freq_array=bandwidth[5:25],
                ),
                ValueError,
                "both spw_range and freq_array",
                id="spw_range_and_freq_array",
            ),
        ],
    )
    def test_invalid_input(
        self,
        ft_bandwidth: np.ndarray,
        make_kwargs: Callable[[np.ndarray], dict],
        exc_type: type[Exception],
        match: str | None,
    ) -> None:
        with pytest.raises(exc_type, match=match):
            uvwindow.FTBeam.from_file(**make_kwargs(ft_bandwidth))

    @pytest.mark.parametrize("bad_spw", BAD_SPWS)
    def test_invalid_spw_range(self, bad_spw: tuple[int, ...]) -> None:
        with (
            pytest.warns(DeprecationWarning, match="spw_range"),
            pytest.raises(ValueError, match="Wrong spw range format"),
        ):
            uvwindow.FTBeam.from_file(spw_range=bad_spw, ftfile=DATA_PATH / ftfile)


class TestFTBeamGaussian:
    @pytest.mark.parametrize(
        "reduce_widths", [lambda w: w, np.mean], ids=["array_widths", "scalar_width"]
    )
    def test_widths(
        self,
        ft_beam_spw: uvwindow.FTBeam,
        reduce_widths: Callable[[np.ndarray], np.ndarray | float],
    ) -> None:
        """Check that widths can be given per frequency or as a single value."""
        freq_array = ft_beam_spw.freq_array
        widths = -0.0343 * freq_array / 1e6 + 11.30
        test = uvwindow.FTBeam.gaussian(
            freq_array=freq_array, widths=reduce_widths(widths), pol="xx"
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

    def test_cosmo_deprecated(
        self, ft_beam_spw: uvwindow.FTBeam, cosmo: conversions.Cosmo_Conversions
    ) -> None:
        """The cosmo argument is unused, so passing it warns and changes nothing."""
        kwargs = dict(freq_array=ft_beam_spw.freq_array, pol="xx", widths=7.0)
        with pytest.warns(DeprecationWarning, match="cosmo argument.*is unused"):
            test = uvwindow.FTBeam.gaussian(cosmo=cosmo, **kwargs)
        np.testing.assert_array_equal(
            test.ft_beam, uvwindow.FTBeam.gaussian(**kwargs).ft_beam
        )


class TestFTBeamGetBandwidth:
    def test_matches_fixture(self, ft_bandwidth: np.ndarray) -> None:
        result = uvwindow.FTBeam.get_bandwidth(DATA_PATH / ftfile)
        assert np.all(result == ft_bandwidth)
        # the analytic FT beams used in this module are on the same channels
        np.testing.assert_array_equal(result, FT_FILE_FREQS)

    def test_invalid_file(self) -> None:
        with pytest.raises(ValueError, match="Wrong ftfile input"):
            uvwindow.FTBeam.get_bandwidth(ftfile="whatever")


class TestFTBeamUpdateSpw:
    def test_happy_path(self, ft_beam_full: uvwindow.FTBeam) -> None:
        test = ft_beam_full
        full_freqs, full_ft_beam = test.freq_array.copy(), test.ft_beam.copy()
        with pytest.warns(DeprecationWarning, match="select_freqs"):
            test.update_spw((5, 25))
        assert test.spw_range == (5, 25)
        np.testing.assert_array_equal(test.freq_array, full_freqs[5:25])
        np.testing.assert_array_equal(test.ft_beam, full_ft_beam[5:25])

    @pytest.mark.parametrize("bad_spw", BAD_SPWS)
    def test_invalid_range(
        self, ft_beam_full: uvwindow.FTBeam, bad_spw: tuple[int, ...]
    ) -> None:
        test = ft_beam_full
        with (
            pytest.warns(DeprecationWarning, match="select_freqs"),
            pytest.raises(ValueError, match="Wrong spw range format"),
        ):
            test.update_spw(spw_range=bad_spw)


class TestUVWindowInit:
    def test_happy_path(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw)
        # a single FTBeam is used for both polarizations
        assert test.ftbeam_obj_pol == [ft_beam_spw, ft_beam_spw]
        assert test.pols == (ft_beam_spw.pol, ft_beam_spw.pol)
        np.testing.assert_array_equal(test.freq_array, ft_beam_spw.freq_array)
        assert test.Nfreqs == ft_beam_spw.freq_array.size
        assert test.taper is None
        np.testing.assert_allclose(
            test.dly_array, utils.get_delays(test.freq_array, n_dlys=test.Nfreqs)
        )
        assert test.avg_z == test.cosmo.f2z(np.mean(ft_beam_spw.freq_array))

    @pytest.mark.parametrize(
        "freqs", [FT_FILE_FREQS, FT_FILE_FREQS[:20]], ids=["wider", "shifted"]
    )
    def test_inconsistent_ftbeam_spectral_range(
        self, freqs: np.ndarray, ft_beam_spw: uvwindow.FTBeam
    ) -> None:
        ft_beam_full = gaussian_ft_beam(freqs)
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
        assert test.cosmo is cosmo

    def test_cosmo_none_raises(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        with pytest.raises(ValueError, match="If no preferred cosmology"):
            uvwindow.UVWindow(cosmo=None, ftbeam_obj=ft_beam_spw)

    def test_verbose(self, ft_beam_spw: uvwindow.FTBeam) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, verbose=True)
        assert test.verbose

    @pytest.mark.parametrize(
        "little_h,kunits",
        [(True, units.h / units.Mpc), (False, units.Mpc ** (-1))],
        ids=["little_h_true", "little_h_false"],
    )
    def test_little_h(
        self, ft_beam_spw: uvwindow.FTBeam, little_h: bool, kunits: units.Unit
    ) -> None:
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_spw, little_h=little_h)
        assert test.kunits.is_equivalent(kunits)


class TestUVWindowFromUvpspec:
    @staticmethod
    def _assert_matches_uvp(uvw: uvwindow.UVWindow, uvp: UVPSpec, ipol: int) -> None:
        """Check that a UVWindow built from a UVPSpec inherits its properties."""
        np.testing.assert_allclose(
            uvw.freq_array, uvp.freq_array[uvp.spw_to_freq_indices(0)]
        )
        assert uvw.taper == uvp.taper
        assert uvw.little_h == ("h^-3" in uvp.norm_units)
        assert uvw.pols == uvputils.polpair_int2tuple(
            uvp.polpair_array[ipol], pol_strings=True
        )

    # uvp_for_uvwindow holds (uvp, uvp without cosmology, cross-pol uvp)
    @pytest.mark.parametrize("iuvp", [0, 2], ids=["happy_path", "crosspol"])
    def test_matches_uvp(
        self, uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec], iuvp: int
    ) -> None:
        uvp = uvp_for_uvwindow[iuvp]
        uvw = uvwindow.UVWindow.from_uvpspec(
            uvp, ipol=0, spw=0, verbose=True, ftbeam=DATA_PATH / basename
        )
        self._assert_matches_uvp(uvw, uvp, ipol=0)
        assert uvw.cosmo == uvp.cosmo

    def test_no_cosmo_warns(
        self,
        uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec],
        ft_beam_full: uvwindow.FTBeam,
    ) -> None:
        _, uvp_nocosmo, _ = uvp_for_uvwindow
        with pytest.warns(UserWarning, match="uvp has no cosmo attribute"):
            uvw = uvwindow.UVWindow.from_uvpspec(
                uvp_nocosmo, ipol=0, spw=0, verbose=True, ftbeam=ft_beam_full
            )
        # the fiducial cosmology is used instead
        assert uvw.cosmo == conversions.Cosmo_Conversions()

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
        self,
        uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec],
        ft_beam_spw: uvwindow.FTBeam,
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        with pytest.raises(ValueError, match="Input spw must be smaller or equal"):
            uvwindow.UVWindow.from_uvpspec(uvp=uvp, ipol=0, spw=2, ftbeam=ft_beam_spw)

    def test_ftbeam_object(
        self,
        uvp_for_uvwindow: tuple[UVPSpec, UVPSpec, UVPSpec],
        ft_beam_spw: uvwindow.FTBeam,
    ) -> None:
        uvp, _, _ = uvp_for_uvwindow
        uvw = uvwindow.UVWindow.from_uvpspec(
            uvp, ipol=0, spw=0, verbose=True, ftbeam=ft_beam_spw
        )
        self._assert_matches_uvp(uvw, uvp, ipol=0)


class TestUVWindowGetKgrid:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        bl_len = red_bl_lens[12]
        width = 0.020
        kgrid, kperp_norm = uvwindow_obj._get_kgrid(bl_len, width=width)
        # regular grid spanning +/- width around the (x or y) kperp of the baseline
        dk = np.diff(kgrid)
        np.testing.assert_allclose(dk, dk[0])
        kp_centre = (
            uvwindow_obj.cosmo.bl_to_kperp(uvwindow_obj.avg_z, little_h=True)
            * bl_len
            / np.sqrt(2.0)
        )
        np.testing.assert_allclose(kgrid[0], kp_centre - width)
        assert kp_centre + width - dk[0] <= kgrid[-1] < kp_centre + width
        np.testing.assert_allclose(
            kperp_norm, np.sqrt(kgrid[:, None] ** 2 + kgrid[None, :] ** 2)
        )

    def test_too_narrow_width(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        with pytest.raises(
            ValueError, match="Change width to resolve full window function"
        ):
            uvwindow_obj._get_kgrid(bl_len=red_bl_lens[12], width=0.0004)


class TestUVWindowKperp4blFreq:
    def test_happy_path(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        bl_len = red_bl_lens[12]
        ngrid = uvwindow_obj.ftbeam_obj_pol[0].ft_beam.shape[-1]
        k = uvwindow_obj._kperp4bl_freq(
            freq=uvwindow_obj.freq_array[12], bl_len=bl_len, ngrid=ngrid
        )
        # a regular, increasing grid of ngrid kperp values
        assert k.shape == (ngrid,)
        assert np.all(np.diff(k) > 0)
        np.testing.assert_allclose(np.diff(k), np.diff(k)[0])

    def test_outside_spectral_window(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ngrid = uvwindow_obj.ftbeam_obj_pol[0].ft_beam.shape[-1]
        with pytest.raises(ValueError, match="Choose frequency within spectral window"):
            uvwindow_obj._kperp4bl_freq(
                freq=1.35e8, bl_len=red_bl_lens[12], ngrid=ngrid
            )

    def test_not_in_hz(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        ngrid = uvwindow_obj.ftbeam_obj_pol[0].ft_beam.shape[-1]
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
        interp_ft_beam, kperp_norm = uvwindow_obj._interpolate_ft_beam(
            red_bl_lens[12], ft_beam
        )
        kgrid, kperp_norm_expected = uvwindow_obj._get_kgrid(red_bl_lens[12])
        assert interp_ft_beam.shape == (kgrid.size, kgrid.size, uvwindow_obj.Nfreqs)
        np.testing.assert_array_equal(kperp_norm, kperp_norm_expected)
        assert np.all(np.isfinite(interp_ft_beam))
        assert np.any(interp_ft_beam != 0)

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
        fnu = uvwindow_obj._take_freq_FT(interp_ft_beam, delta_nu)
        assert fnu.shape == interp_ft_beam.shape
        # Parseval: the (orthonormal) FT along frequency preserves the power of
        # the tapered beam, up to the delta_nu factor
        taper = dspec.gen_window(uvwindow_obj.taper, uvwindow_obj.Nfreqs)
        np.testing.assert_allclose(
            np.sum(np.abs(fnu) ** 2, axis=-1),
            delta_nu * np.sum(np.abs(interp_ft_beam * taper) ** 2, axis=-1),
            rtol=1e-10,
        )

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
        # a constant input gives a constant output in every populated kpara bin
        wf_array = np.ones((kperp_bins.size, uvwindow_obj.Nfreqs))
        kpara, cyl_wf = uvwindow_obj._get_wf_for_tau(
            tau, wf_array, kperp_bins, kpara_bins
        )
        assert cyl_wf.shape == (kperp_bins.size, kpara_bins.size)
        assert kpara.shape == (kpara_bins.size,)
        populated = kpara > 0
        assert np.any(populated)
        np.testing.assert_array_equal(cyl_wf[:, populated], 1.0)
        np.testing.assert_array_equal(cyl_wf[:, ~populated], 0.0)
        # the mean kpara of each populated bin lies within that bin
        dk = np.diff(kpara_bins)[0]
        assert np.all(np.abs(kpara[populated] - kpara_bins[populated]) <= dk / 2)


class TestUVWindowGetKperpBins:
    def test_empty_list_error(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        with pytest.raises(
            ValueError, match="get_kperp_bins\\(\\) requires array of baseline lengths"
        ):
            uvwindow_obj.get_kperp_bins(bl_lens=[])

    @pytest.mark.parametrize(
        "select_bl_lens",
        [lambda bl_lens: bl_lens[12], lambda bl_lens: bl_lens],
        ids=["scalar", "array"],
    )
    def test_linear_bins_with_units(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        select_bl_lens: Callable[[np.ndarray], np.ndarray | float],
    ) -> None:
        bl_lens = select_bl_lens(red_bl_lens)
        kperps = uvwindow_obj.get_kperp_bins(bl_lens)
        assert uvwindow_obj.kunits.is_equivalent(kperps.unit)
        kperp_bins = kperps.value
        # linearly spaced bins covering the longest baseline
        np.testing.assert_allclose(np.diff(kperp_bins), np.diff(kperp_bins)[0])
        kperp_max = uvwindow_obj.cosmo.bl_to_kperp(
            uvwindow_obj.avg_z, little_h=True
        ) * np.max(bl_lens)
        assert kperp_bins.max() > kperp_max

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
        kparas = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
        assert uvwindow_obj.kunits.is_equivalent(kparas.unit)
        kpara_bins = kparas.value
        # linearly spaced bins, one per delay step, covering the largest delay
        kpara_per_tau = uvwindow_obj.cosmo.tau_to_kpara(
            uvwindow_obj.avg_z, little_h=True
        )
        dly_array = uvwindow_obj.dly_array
        np.testing.assert_allclose(
            np.diff(kpara_bins), kpara_per_tau * np.diff(dly_array)[0]
        )
        assert kpara_bins.max() > kpara_per_tau * np.abs(dly_array).max()

    def test_large_bandwidth_warns(self, uvwindow_obj: uvwindow.UVWindow) -> None:
        HERA_bw = np.linspace(1, 2, 1024, endpoint=False) * 1e8
        with pytest.warns(UserWarning, match="Large number of kperp/kpara bins"):
            _ = uvwindow_obj.get_kpara_bins(HERA_bw)


class TestUVWindowGetCylindricalWf:
    def test_return_bins_weighted(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray
    ) -> None:
        kperp, kpara, cyl_wf = uvwindow_obj.get_cylindrical_wf(
            red_bl_lens[12], kperp_bins=None, kpara_bins=None, return_bins="weighted"
        )
        assert cyl_wf.shape == (uvwindow_obj.Nfreqs, kperp.size, kpara.size)
        assert np.allclose(np.sum(cyl_wf, axis=(1, 2)), 1.0, atol=1e-3)

    def test_return_bins_none(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, _, _, cyl_wf = cyl_wf_result
        result = uvwindow_obj.get_cylindrical_wf(
            bl_len, kperp_bins=None, kpara_bins=None, return_bins=None
        )
        # only the window function is returned
        np.testing.assert_array_equal(result, cyl_wf)

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

    @pytest.mark.parametrize("axis", ["kperp", "kpara"])
    def test_custom_bins(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
        axis: str,
    ) -> None:
        """Check that passing the default bins explicitly reproduces the default result."""
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        bins = {"kperp_bins": None, "kpara_bins": None}
        bins[f"{axis}_bins"] = {"kperp": kperp, "kpara": kpara}[
            axis
        ] * uvwindow_obj.kunits
        kperp2, kpara2, cyl_wf2 = uvwindow_obj.get_cylindrical_wf(
            bl_len, **bins, return_bins="unweighted"
        )
        assert np.allclose(cyl_wf2, cyl_wf)
        assert np.allclose(kperp2, kperp)
        assert np.allclose(kpara2, kpara)

    @pytest.mark.parametrize(
        "bad_kwarg,bad_bins",
        [
            ("kperp_bins", np.logspace(-2, 0, 100)),
            ("kpara_bins", np.logspace(-1, 1, 100)),
        ],
        ids=["kperp", "kpara"],
    )
    def test_nonlinear_bins_error(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        bad_kwarg: str,
        bad_bins: np.ndarray,
    ) -> None:
        bins = {"kperp_bins": None, "kpara_bins": None}
        bins[bad_kwarg] = bad_bins * uvwindow_obj.kunits
        with pytest.raises(
            ValueError, match=f"get_cylindrical_wf: {bad_kwarg} must be linearly spaced"
        ):
            uvwindow_obj.get_cylindrical_wf(
                red_bl_lens[12], **bins, return_bins="unweighted"
            )

    def test_odd_number_of_delays(self, red_bl_lens: np.ndarray) -> None:
        ft_beam_test = gaussian_ft_beam(FT_FREQS[:19])
        test = uvwindow.UVWindow(ftbeam_obj=ft_beam_test)
        kperp, kpara, cyl_wf = test.get_cylindrical_wf(
            red_bl_lens[12], return_bins="unweighted"
        )
        assert cyl_wf.shape == (19, kperp.size, kpara.size)
        assert np.allclose(np.sum(cyl_wf, axis=(1, 2)), 1.0, atol=1e-3)


class TestUVWindowCylindricalToSpherical:
    def test_with_weights(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        wf, weighted_k = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=[2.0]
        )
        # with a single baseline, its weight doesn't change the result
        wf_unweighted, weighted_k_unweighted = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=None
        )
        np.testing.assert_allclose(wf, wf_unweighted)
        np.testing.assert_allclose(weighted_k, weighted_k_unweighted)

    def test_no_weights(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        wf, weighted_k = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf[None], kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=None
        )
        assert wf.shape == (kbins.size, kbins.size)
        # each populated k bin has a normalized window function and a mean k
        # within the bin; empty bins are zero
        row_sums = wf.sum(axis=1)
        populated = weighted_k > 0
        assert np.any(populated)
        np.testing.assert_allclose(row_sums[populated], 1.0)
        np.testing.assert_array_equal(row_sums[~populated], 0.0)
        dk = np.diff(kbins.value)[0]
        assert np.all(np.abs(weighted_k[populated] - kbins.value[populated]) <= dk / 2)

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
        uvw = copy.deepcopy(uvwindow_obj)
        uvw.verbose = True
        with pytest.warns(UserWarning, match="Some spherical bins are empty"):
            _ = uvw.cylindrical_to_spherical(
                cyl_wf=cyl_wf, kbins=kbins_test, ktot=ktot, bl_lens=bl_len
            )

    def test_fractional_weights_do_not_empty_bins(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        kbins: units.Quantity,
        cyl_wf_result: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Bins populated by baselines with fractional weights are not reported empty."""
        bl_len, kperp, kpara, cyl_wf = cyl_wf_result
        ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)
        kbins = kbins[:6]  # all populated by this baseline
        uvw = copy.deepcopy(uvwindow_obj)
        uvw.verbose = True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wf, weighted_k = uvw.cylindrical_to_spherical(
                cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=[0.1]
            )
        assert np.all(weighted_k > 0)
        np.testing.assert_allclose(wf.sum(axis=1), 1.0)

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
        wf = uvwindow_obj.get_spherical_wf(
            kbins=full_kbins,
            kperp_bins=kperp_bins,
            kpara_bins=kpara_bins,
            bl_lens=red_bl_lens[:1],
            bl_weights=None,
            return_weighted_k=False,
            verbose=None,
        )
        assert wf.shape == (full_kbins.size, full_kbins.size)
        # rows are normalized, except for k bins not probed by the baseline
        row_sums = wf.sum(axis=1)
        assert np.all(np.isclose(row_sums, 1) | (row_sums == 0))
        assert np.any(np.isclose(row_sums, 1))

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
        # consistent units pass without raising
        uvwindow_obj.check_kunits(kbins)

    def test_inconsistent_little_h_raises(
        self, uvwindow_obj: uvwindow.UVWindow, kbins: units.Quantity
    ) -> None:
        assert uvwindow_obj.little_h
        with pytest.raises(ValueError, match="not consistent with little_h"):
            uvwindow_obj.check_kunits(kbins.value / units.Mpc)

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
            bl_weights=[2.0],
            kperp_bins=kperp_bins,
            kpara_bins=kpara_bins,
            clobber=False,
        )
        with h5py.File(tmp_path / outfile, "r") as f:
            assert f.attrs["taper"] == uvwindow_obj.taper
            np.testing.assert_allclose(f["kperp_bins"][:], kperp_bins.value)
            np.testing.assert_allclose(f["kpara_bins"][:], kpara_bins.value)
            np.testing.assert_allclose(f["bl_weights"][:], [2.0])
            # the stored window function is the weighted cylindrical one
            cyl_wf = uvwindow_obj.get_cylindrical_wf(
                red_bl_lens[0], kperp_bins, kpara_bins
            )
            np.testing.assert_allclose(f["cyl_wf"][0], 2.0 * cyl_wf)

    def test_no_taper(
        self, uvwindow_obj: uvwindow.UVWindow, red_bl_lens: np.ndarray, tmp_path: Path
    ) -> None:
        uvw = copy.deepcopy(uvwindow_obj)
        uvw.taper = None
        kperp_bins = uvw.get_kperp_bins(red_bl_lens[:1])
        kpara_bins = uvw.get_kpara_bins(uvw.freq_array)
        files = {}
        for name, obj in [("no_taper", uvw), ("taper", uvwindow_obj)]:
            files[name] = tmp_path / f"{name}.hdf5"
            obj.run_and_write(
                filepath=str(files[name]),
                bl_lens=red_bl_lens[:1],
                bl_weights=[1.0],
                kperp_bins=kperp_bins,
                kpara_bins=kpara_bins,
                clobber=False,
            )
        with (
            h5py.File(files["no_taper"], "r") as f_none,
            h5py.File(files["taper"], "r") as f_taper,
        ):
            assert f_none.attrs["taper"] == "none"
            assert f_taper.attrs["taper"] == "blackman-harris"
            cyl_wf_none, cyl_wf_taper = f_none["cyl_wf"][0], f_taper["cyl_wf"][0]
        # both are normalized window functions, but the taper changes their shape
        assert np.allclose(np.sum(cyl_wf_none, axis=(1, 2)), 1.0, atol=1e-3)
        assert not np.allclose(cyl_wf_none, cyl_wf_taper)

    @pytest.mark.parametrize(
        "clobber,second_write",
        [
            (False, pytest.raises(IOError, match="exists, not overwriting")),
            (True, nullcontext()),
        ],
        ids=["clobber_false_raises", "clobber_true_overwrites"],
    )
    def test_clobber(
        self,
        uvwindow_obj: uvwindow.UVWindow,
        red_bl_lens: np.ndarray,
        tmp_path: Path,
        clobber: bool,
        second_write,
    ) -> None:
        """Check that writing to an existing file needs clobber=True."""
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile), bl_lens=red_bl_lens[:1], clobber=True
        )
        with second_write:
            uvwindow_obj.run_and_write(
                filepath=str(tmp_path / outfile),
                bl_lens=[red_bl_lens[:1]],
                kperp_bins=None,
                kpara_bins=None,
                clobber=clobber,
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
    def test_exact_match_extracts_channels(self, ft_beam_full: uvwindow.FTBeam) -> None:
        full = ft_beam_full
        sub = full.select_freqs(FT_FILE_FREQS[3:10], inplace=False)
        assert np.allclose(sub.freq_array, FT_FILE_FREQS[3:10])
        assert np.array_equal(sub.ft_beam, full.ft_beam[3:10])
        # with inplace=False the original is untouched
        assert full.freq_array.size == FT_FILE_FREQS.size

    def test_mismatched_grid_interpolates_with_warning(
        self, ft_beam_full: uvwindow.FTBeam
    ) -> None:
        # a quarter of the way between channels of a linear grid: linear
        # interpolation weighs the lower channel by 3/4 and the upper by 1/4
        # (unequal weights, so swapping them would be caught)
        full = ft_beam_full
        targets = 0.75 * FT_FILE_FREQS[3:10] + 0.25 * FT_FILE_FREQS[4:11]
        with pytest.warns(UserWarning, match="interpolating"):
            interp = full.select_freqs(targets, inplace=False)
        assert np.allclose(interp.freq_array, targets)
        assert np.allclose(
            interp.ft_beam, 0.75 * full.ft_beam[3:10] + 0.25 * full.ft_beam[4:11]
        )

    @pytest.mark.parametrize(
        "make_bad_freqs,match",
        [
            pytest.param(
                lambda bandwidth: np.array([bandwidth.max() + 1e6] * 3),
                "beyond the bandwidth",
                id="out_of_coverage",
            ),
            pytest.param(
                lambda bandwidth: bandwidth[:1],
                "at least two frequencies",
                id="single_frequency",
            ),
        ],
    )
    def test_invalid_freqs_raise(
        self,
        ft_beam_full: uvwindow.FTBeam,
        make_bad_freqs: Callable[[np.ndarray], np.ndarray],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ft_beam_full.select_freqs(make_bad_freqs(FT_FILE_FREQS))


class TestFTBeamFromFileFreqArray:
    def test_partial_read_matches_full_read(
        self, ft_beam_file: uvwindow.FTBeam, ft_bandwidth: np.ndarray
    ) -> None:
        full = ft_beam_file
        target = ft_bandwidth[5:25]
        test = uvwindow.FTBeam.from_file(ftfile=DATA_PATH / ftfile, freq_array=target)
        assert np.allclose(test.freq_array, target)
        assert np.array_equal(test.ft_beam, full.ft_beam[5:25])

    def test_interpolation_through_partial_read(
        self, ft_beam_file: uvwindow.FTBeam, ft_bandwidth: np.ndarray
    ) -> None:
        # requested frequencies fall between the file's channels, so
        # select_freqs interpolates using only the padded channel range
        # read from disk; must equal loading the whole file and
        # interpolating. Starts at the first channel pair to stress the
        # edge padding.
        full = ft_beam_file
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
        with h5py.File(fname, "r") as f:
            assert f.attrs["beam_file"] == "sim.fits"
        assert back != 1  # non-FTBeam comparison falls through NotImplemented

    def test_no_overwrite_by_default(
        self, ft_beam_spw: uvwindow.FTBeam, tmp_path: Path
    ) -> None:
        fname = tmp_path / "ft_beam.hdf5"
        ft_beam_spw.write_hdf5(fname)
        with pytest.raises(FileExistsError, match="overwrite"):
            ft_beam_spw.write_hdf5(fname)
        # with overwrite=True the file is replaced by the new content
        other = ft_beam_spw.select_freqs(ft_beam_spw.freq_array[:5], inplace=False)
        other.write_hdf5(fname, overwrite=True)
        assert uvwindow.FTBeam.from_file(ftfile=fname) == other


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


class TestUVWindowPhysics:
    """The exact window function of the delay spectrum of a baseline b at delay
    tau must be centred on the (k_perp, k_para) that the baseline and delay
    probe, k_perp = bl_to_kperp(z) |b| and k_para = tau_to_kpara(z) |tau|, with
    a k_perp spread set by the beam: the beam's Fourier transform has a width
    inversely proportional to the width of the beam on the sky."""

    BL_LEN = 29.2  # m

    @staticmethod
    def window(width: float) -> tuple[uvwindow.UVWindow, tuple]:
        # A map half-width of 1.5 rad gives a fine enough k_perp grid to
        # resolve the window function of beams a few degrees wide.
        ftbeam = uvwindow.FTBeam.gaussian(
            freq_array=FT_FREQS, widths=width, pol="xx", mapsize=1.5, npix=301
        )
        uvw = uvwindow.UVWindow(
            ftbeam_obj=ftbeam, taper="blackman-harris", little_h=True, verbose=False
        )
        return uvw, uvw.get_cylindrical_wf(
            TestUVWindowPhysics.BL_LEN, return_bins="unweighted"
        )

    @pytest.fixture(scope="class")
    def window_7deg(self):
        return self.window(7.0)

    def test_kpara_peak(self, window_7deg) -> None:
        """For every delay the window function peaks at k_para = tau_to_kpara |tau|
        (to within one k_para bin)."""
        uvw, (_, kpara, cyl_wf) = window_7deg
        expected = uvw.cosmo.tau_to_kpara(uvw.avg_z) * np.abs(uvw.dly_array)
        peak = kpara[np.argmax(cyl_wf.sum(axis=1), axis=1)]
        assert np.all(np.abs(peak - expected) <= np.diff(kpara).mean() * (1 + 1e-6))

    def test_kpara_peak_with_ftbeam_on_another_grid(self, vanilla_uvp) -> None:
        """The k_para of a delay is set by the data's frequency channels, not by
        those the FT beam was computed on. With an FT beam on 80 kHz channels
        and data on 100 kHz channels, UVWindow.from_uvpspec must interpolate the
        beam onto the data channels, so that the window function of every
        nonzero data delay tau peaks in the k_para bin containing
        tau_to_kpara |tau| (the bins are centred on multiples of that value).
        Taking the FT beam's channels as they are would stretch the k_para
        axis by the ratio of the channel widths."""
        ftbeam = uvwindow.FTBeam.gaussian(
            freq_array=np.arange(99e6, 106e6, 80e3), widths=10.0, pol="xx", npix=151
        )
        with pytest.warns(UserWarning, match="interpolating"):
            uvw = uvwindow.UVWindow.from_uvpspec(vanilla_uvp, 0, 0, ftbeam=ftbeam)
        _, kpara, cyl_wf = uvw.get_cylindrical_wf(14.6, return_bins="unweighted")
        dlys = vanilla_uvp.get_dlys(0)
        expected = uvw.cosmo.tau_to_kpara(uvw.avg_z, little_h=uvw.little_h) * np.abs(
            dlys
        )
        peak = kpara[np.argmax(cyl_wf.sum(axis=1), axis=1)]
        nonzero = dlys != 0
        assert nonzero.sum() == dlys.size - 1
        assert np.all(np.abs(peak - expected)[nonzero] < 0.5 * np.diff(kpara).mean())

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "UVWindow._interpolate_ft_beam evaluates the FT beam only along the "
            "diagonal of the k_perp grid, so the window function peaks at "
            "~0.77 of the baseline's k_perp, see #571"
        ),
    )
    def test_kperp_peak(self, window_7deg) -> None:
        """The window function peaks at the k_perp of the baseline, within one
        k_perp bin."""
        uvw, (kperp, _, cyl_wf) = window_7deg
        expected = uvw.cosmo.bl_to_kperp(uvw.avg_z) * self.BL_LEN
        peak = kperp[np.argmax(cyl_wf.sum(axis=2), axis=1)]
        assert np.all(np.abs(peak - expected) <= np.diff(kperp).mean())

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "UVWindow._interpolate_ft_beam evaluates the FT beam only along the "
            "diagonal of the k_perp grid, so the k_perp spread does not depend "
            "on the beam width, see #571"
        ),
    )
    def test_kperp_spread_scales_with_inverse_beam_width(self, window_7deg) -> None:
        """The k_perp spread of the window function is inversely proportional
        to the width of the beam on the sky: doubling the width halves it.

        FTBeam.gaussian(widths=w) is the beam exp(-theta^2 / w^2), whose Fourier
        transform has a width 1 / (pi w) in u. The window function goes as the
        square of the transform, so its spread in u is 1 / (2 pi w), i.e.
        1 / (w DM) in k_perp = 2 pi u / DM, with DM the transverse comoving
        distance. The tolerance allows for the binning in k_perp and the
        evolution of DM across the band."""
        for width, (uvw, (kperp, _, cyl_wf)) in [
            (7.0, window_7deg),
            (14.0, self.window(14.0)),
        ]:
            profile = cyl_wf.sum(axis=2)
            profile /= profile.sum(axis=1, keepdims=True)
            mean = (profile * kperp).sum(axis=1)
            spread = np.sqrt((profile * kperp**2).sum(axis=1) - mean**2)
            expected = 1 / (np.radians(width) * uvw.cosmo.DM(uvw.avg_z, little_h=True))
            np.testing.assert_allclose(spread, expected, rtol=0.1)

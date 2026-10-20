"""Builders for the legacy (v0.x) HDF5 fixtures in ``tests/data/legacy/``.

The files in ``tests/data/legacy/`` were written by hera_pspec before the HDF5
format gained a version number and unit attributes (commit f942d1a). The tests
read them with the current code and compare against the same objects rebuilt
in memory with the builders below, so a reader change that breaks old files is
caught. ``tests/data/regenerate_legacy_fixtures.py`` uses the same builders;
see ``tests/data/legacy/README.md`` before running it.
"""

import copy
from pathlib import Path

import numpy as np

from hera_pspec import PSpecContainer, PSpecData, UVPSpec, testing, utils, uvwindow
from tests.regression_cases import FT_BEAM_FILE, FT_BEAM_ROOT, _read_uvh5, dipole_beam

LEGACY_DIR = Path(__file__).parent / "data" / "legacy"

CONTAINER_FILE = LEGACY_DIR / "container_v1.h5"
CONTAINER_GROUP = "legacy"
UVP_WINDOWS_NAME = "exact_windows_cov"
UVP_PLAIN_NAME = "vanilla_plain"

UVWINDOW_FILE = LEGACY_DIR / "uvwindow_v1.h5"
UVWINDOW_BL_LENS = np.array([14.6, 29.2])
UVWINDOW_BL_WEIGHTS = [2.0, 1.0]

FTBEAM_FILE = LEGACY_DIR / "ftbeam_v1_xx.hdf5"
# a few channels inside the band of FT_beam_HERA_dipole_test_xx.hdf5
FTBEAM_CHANNELS = slice(10, 13)
# central pixels kept of the 213 x 213 FT grid (odd and centred, so the
# Fourier modes of the kept pixels are unchanged)
FTBEAM_NPIX = 61
UVWINDOW_CHANNELS = slice(5, 15)


def build_uvp_exact_windows_cov() -> UVPSpec:
    """pspec() with exact window functions and an empirical covariance."""
    uvd = copy.deepcopy(_read_uvh5("zen.2458116.31939.HH.uvh5"))
    uvd.select(times=np.unique(uvd.time_array)[:1])
    ds = PSpecData(
        dsets=[uvd, copy.deepcopy(uvd)], wgts=[None, None], beam=dipole_beam()
    )
    bls1, bls2, _ = utils.construct_blpairs(
        uvd.get_antpairs()[1:], exclude_permutations=False, exclude_auto_bls=True
    )
    return ds.pspec(
        bls1,
        bls2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=(175, 185),
        taper="bh",
        exact_windows=True,
        ftbeam=str(FT_BEAM_ROOT),
        store_cov=True,
        cov_model="empirical",
        verbose=False,
    )


def build_uvp_plain() -> UVPSpec:
    """The vanilla UVPSpec (with beam metadata) without covariance or windows."""
    uvp, _ = testing.build_vanilla_uvpspec(beam=dipole_beam(), Ndlys=10)
    del uvp.cov_array_real
    del uvp.cov_array_imag
    del uvp.window_function_array
    uvp.check()
    return uvp


def build_ftbeam() -> uvwindow.FTBeam:
    """FT_beam_HERA_dipole_test_xx.hdf5 cut to three channels and 61 x 61 pixels."""
    bandwidth = uvwindow.FTBeam.get_bandwidth(FT_BEAM_FILE)
    full = uvwindow.FTBeam.from_file(
        ftfile=FT_BEAM_FILE, freq_array=bandwidth[FTBEAM_CHANNELS]
    )
    npix = full.ft_beam.shape[-1]
    start = (npix - FTBEAM_NPIX) // 2
    crop = slice(start, start + FTBEAM_NPIX)
    return uvwindow.FTBeam(
        data=full.ft_beam[:, crop, crop],
        pol=full.pol,
        freq_array=full.freq_array,
        mapsize=full.mapsize,
    )


def build_uvwindow() -> uvwindow.UVWindow:
    """UVWindow (little_h) on ten channels of the test FT beam."""
    bandwidth = uvwindow.FTBeam.get_bandwidth(FT_BEAM_FILE)
    ftbeam = uvwindow.FTBeam.from_file(
        ftfile=FT_BEAM_FILE, freq_array=bandwidth[UVWINDOW_CHANNELS]
    )
    return uvwindow.UVWindow(
        ftbeam_obj=ftbeam, taper="blackman-harris", little_h=True, verbose=False
    )


def write_container(path: Path = CONTAINER_FILE) -> None:
    psc = PSpecContainer(str(path), mode="rw", keep_open=False)
    psc.set_pspec(CONTAINER_GROUP, UVP_WINDOWS_NAME, build_uvp_exact_windows_cov())
    psc.set_pspec(CONTAINER_GROUP, UVP_PLAIN_NAME, build_uvp_plain())


def write_uvwindow(path: Path = UVWINDOW_FILE) -> None:
    build_uvwindow().run_and_write(
        path, bl_lens=UVWINDOW_BL_LENS, bl_weights=UVWINDOW_BL_WEIGHTS
    )


def write_ftbeam(path: Path = FTBEAM_FILE) -> None:
    build_ftbeam().write_hdf5(path)


LEGACY_FIXTURES = {
    CONTAINER_FILE.name: write_container,
    UVWINDOW_FILE.name: write_uvwindow,
    FTBEAM_FILE.name: write_ftbeam,
}

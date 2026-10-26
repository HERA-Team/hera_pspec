"""Input builders for the numerical regression tests.

Each case below is a function that builds its inputs from the data shipped in
``hera_pspec.data``, runs the library code and returns a flat dict mapping a
key to a numpy array (or a numpy string scalar). The expected values live in
``tests/data/regression/<case>.npz``.

Both ``tests/test_regression_numerics.py`` and
``tests/data/regenerate_regression_fixtures.py`` call :func:`compute_case`, so
a fixture can never be built from different inputs than the test that reads
it. See ``tests/data/README.md`` before regenerating anything.
"""

import copy
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import astropy.cosmology.units as cu
import numpy as np
from astropy import units
from pyuvdata import UVData

from hera_pspec import (
    PSpecBeamFromArray,
    PSpecBeamGauss,
    PSpecBeamUV,
    PSpecData,
    UVPSpec,
    conversions,
    grouping,
    noise,
    unit_helpers,
    utils,
    uvwindow,
)
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)
FIXTURE_DIR = Path(__file__).parent / "data" / "regression"
LITTLEH_LENGTH = units.Mpc / cu.littleh

# Default relative tolerance for deterministic code paths.
DEFAULT_RTOL = 1e-12

# Matrix inverses (H^-1), eigendecompositions (V^-1/2), covariance products and
# the exact-normalisation / exact-window beam integrals go through LAPACK/BLAS
# and FFT routines whose last bits differ between the BLAS builds used on the
# CI runners (Ubuntu/macOS, Python 3.11-3.13). This looser tolerance still
# catches any real change (e.g. a factor of h, a 1e-9 ns/s slip, or the 3.5e-7
# CODATA k_B difference).
LINALG_RTOL = 1e-9


@dataclass(frozen=True)
class RegressionCase:
    """A named regression case and the tolerances used to compare it."""

    name: str
    build: Callable[[], dict[str, np.ndarray]]
    rtol: float = DEFAULT_RTOL
    # Per-key overrides: any key starting with one of these prefixes uses the
    # associated rtol instead of ``rtol``.
    rtol_overrides: dict[str, float] = field(default_factory=dict)

    def rtol_for(self, key: str) -> float:
        """Return the relative tolerance to use for ``key``."""
        for prefix, rtol in self.rtol_overrides.items():
            if key.startswith(prefix):
                return rtol
        return self.rtol

    @property
    def fixture_path(self) -> Path:
        return FIXTURE_DIR / f"{self.name}.npz"


CASES: dict[str, RegressionCase] = {}


def regression_case(
    name: str,
    rtol: float = DEFAULT_RTOL,
    rtol_overrides: dict[str, float] | None = None,
) -> Callable:
    """Register a case builder under ``name``."""

    def decorator(func: Callable[[], dict[str, np.ndarray]]) -> Callable:
        if name in CASES:
            raise ValueError(f"duplicate regression case {name!r}")
        CASES[name] = RegressionCase(
            name=name, build=func, rtol=rtol, rtol_overrides=rtol_overrides or {}
        )
        return func

    return decorator


def compute_case(name: str) -> dict[str, np.ndarray]:
    """Build the inputs for case ``name``, run it and return its outputs."""
    outputs = CASES[name].build()
    return {key: np.asarray(value) for key, value in outputs.items()}


# --------------------------------------------------------------------------
# Shared inputs (cached; callers get deep copies so nothing leaks between cases)
# --------------------------------------------------------------------------


@functools.cache
def _read_miriad(filename: str) -> UVData:
    uvd = UVData()
    uvd.read_miriad(str(DATA_PATH / filename))
    return uvd


@functools.cache
def _read_uvh5(filename: str) -> UVData:
    return UVData.from_file(DATA_PATH / filename)


@functools.cache
def _beam_uv(filename: str) -> PSpecBeamUV:
    return PSpecBeamUV(str(DATA_PATH / filename))


def dipole_beam() -> PSpecBeamUV:
    return copy.deepcopy(_beam_uv("HERA_NF_dipole_power.beamfits"))


def pstokes_beam() -> PSpecBeamUV:
    return copy.deepcopy(_beam_uv("HERA_NF_pstokes_power.beamfits"))


# Only the first few integrations are used, to keep fixtures small.
N_LST_TIMES = 5


@functools.cache
def _lst_uvdata(half: str, std: bool, n_times: int | None) -> UVData:
    std_tag = "std." if std else ""
    uvd = _read_miriad(f"zen.{half}.{std_tag}xx.LST.1.28828.uvOCRSA")
    if n_times is None:
        return uvd
    times = np.unique(uvd.time_array)[:n_times]
    return uvd.select(times=times, inplace=False)


def lst_uvdata(
    half: str, std: bool = False, n_times: int | None = N_LST_TIMES
) -> UVData:
    """One LST-binned cutout (Jy, xx), trimmed to ``n_times`` integrations."""
    return copy.deepcopy(_lst_uvdata(half, std, n_times))


def lst_datasets(
    halves: tuple[str, str] = ("even", "odd"), n_times: int | None = N_LST_TIMES
) -> tuple[list[UVData], list[UVData]]:
    """LST-binned even/odd visibilities and their std files."""
    dsets = [lst_uvdata(half, n_times=n_times) for half in halves]
    dsets_std = [lst_uvdata(half, std=True, n_times=n_times) for half in halves]
    return dsets, dsets_std


def even_odd_blpairs() -> tuple[list, list]:
    """Three baseline-pairs from the first 14.6 m east-west redundant group."""
    uvd = _lst_uvdata("even", False, N_LST_TIMES)
    reds, _, _ = utils.get_reds(
        uvd, pick_data_ants=True, bl_len_range=(10, 20), bl_deg_range=(0, 1)
    )
    bls1, bls2, _ = utils.construct_blpairs(
        reds[0][:3], exclude_auto_bls=True, exclude_permutations=True
    )
    return bls1, bls2


# Two spectral windows; the second one has fewer delays than channels.
EVEN_ODD_SPW_RANGES = [(20, 40), (100, 124)]
EVEN_ODD_N_DLYS = [20, 12]


def run_even_odd_pspec(
    jy_to_mk: bool = True, halves: tuple[str, str] = ("even", "odd"), **pspec_kwargs
) -> UVPSpec:
    """Run PSpecData.pspec on the LST-binned cutouts with the NF dipole beam.

    The even and odd files carry different JDs, so an even x odd UVPSpec has
    ``time_1_array != time_2_array``. Cases that feed grouping or noise code
    which assumes matching times use ``halves=("even", "even")``.
    """
    dsets, dsets_std = lst_datasets(halves)
    ds = PSpecData(
        dsets=dsets, wgts=[None, None], dsets_std=dsets_std, beam=dipole_beam()
    )
    if jy_to_mk:
        ds.Jy_to_mK()
    bls1, bls2 = even_odd_blpairs()
    kwargs = dict(
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=EVEN_ODD_SPW_RANGES,
        n_dlys=EVEN_ODD_N_DLYS,
        verbose=False,
    )
    kwargs.update(pspec_kwargs)
    return ds.pspec(bls1, bls2, **kwargs)


FT_BEAM_ROOT = DATA_PATH / "FT_beam_HERA_dipole_test"
FT_BEAM_FILE = DATA_PATH / "FT_beam_HERA_dipole_test_xx.hdf5"


def run_exact_windows_pspec() -> UVPSpec:
    """pspec with exact window functions on zen.2458116 (inside the FT beam band)."""
    uvd = copy.deepcopy(_read_uvh5("zen.2458116.31939.HH.uvh5"))
    # two integrations are enough: the exact window functions do not depend
    # on time, and they dominate the fixture size
    uvd.select(times=np.unique(uvd.time_array)[:2])
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
        spw_ranges=(175, 195),
        taper="bh",
        exact_windows=True,
        ftbeam=str(FT_BEAM_ROOT),
        verbose=False,
    )


def uvpspec_outputs(uvp: UVPSpec, prefix: str = "") -> dict[str, np.ndarray]:
    """Collect the unit-carrying outputs of a UVPSpec into a flat dict."""
    out = {
        f"{prefix}norm_units": np.array(uvp.norm_units),
        f"{prefix}vis_units": np.array(uvp.vis_units),
        f"{prefix}dly_array": uvp.dly_array,
        f"{prefix}freq_array": uvp.freq_array,
        f"{prefix}scalar_array": uvp.scalar_array,
    }
    for spw in uvp.spw_array:
        out[f"{prefix}data_array_{spw}"] = uvp.data_array[spw]
        out[f"{prefix}get_dlys_{spw}"] = uvp.get_dlys(spw).to_value(units.s)
        for little_h in (True, False):
            tag = "h" if little_h else "noh"
            length_unit = conversions.length_unit_from_little_h(little_h)
            out[f"{prefix}get_kparas_{tag}_{spw}"] = uvp.get_kparas(
                spw, length_unit=length_unit
            ).to_value(1 / length_unit)
            out[f"{prefix}get_kperps_{tag}_{spw}"] = uvp.get_kperps(
                spw, length_unit=length_unit
            ).to_value(1 / length_unit)
        if hasattr(uvp, "cov_array_real"):
            out[f"{prefix}cov_array_real_{spw}"] = uvp.cov_array_real[spw]
            out[f"{prefix}cov_array_imag_{spw}"] = uvp.cov_array_imag[spw]
        if hasattr(uvp, "window_function_array"):
            out[f"{prefix}window_function_array_{spw}"] = uvp.window_function_array[spw]
        if getattr(uvp, "exact_windows", False):
            out[f"{prefix}window_function_kperp_{spw}"] = uvp.window_function_kperp[spw]
            out[f"{prefix}window_function_kpara_{spw}"] = uvp.window_function_kpara[spw]
        if hasattr(uvp, "stats_array"):
            for stat in sorted(uvp.stats_array):
                out[f"{prefix}stats_{stat}_{spw}"] = uvp.stats_array[stat][spw]
    return out


# --------------------------------------------------------------------------
# PSpecData.pspec
# --------------------------------------------------------------------------


@regression_case("pspec_norm_I_littleh_mK")
def _pspec_norm_i_littleh_mk() -> dict[str, np.ndarray]:
    return uvpspec_outputs(run_even_odd_pspec(norm="I", length_unit=LITTLEH_LENGTH))


@regression_case("pspec_norm_I_no_littleh_mK")
def _pspec_norm_i_no_littleh_mk() -> dict[str, np.ndarray]:
    return uvpspec_outputs(run_even_odd_pspec(norm="I", length_unit=units.Mpc))


@regression_case("pspec_norm_I_littleh_Jy")
def _pspec_norm_i_littleh_jy() -> dict[str, np.ndarray]:
    return uvpspec_outputs(run_even_odd_pspec(jy_to_mk=False, norm="I"))


@regression_case("pspec_norm_I_bh_cov_autos", rtol_overrides={"cov_array": LINALG_RTOL})
def _pspec_norm_i_bh_cov_autos() -> dict[str, np.ndarray]:
    uvp = run_even_odd_pspec(
        norm="I",
        taper="bh",
        store_cov=True,
        cov_model="autos",
        length_unit=LITTLEH_LENGTH,
    )
    return uvpspec_outputs(uvp)


@regression_case("pspec_norm_Hinv_bh_cov_empirical", rtol=LINALG_RTOL)
def _pspec_norm_hinv_bh_cov_empirical() -> dict[str, np.ndarray]:
    uvp = run_even_odd_pspec(
        norm="H^-1", taper="bh", store_cov=True, cov_model="empirical"
    )
    return uvpspec_outputs(uvp)


@regression_case("pspec_norm_Vhalf", rtol=LINALG_RTOL)
def _pspec_norm_vhalf() -> dict[str, np.ndarray]:
    # All ten integrations, one spw with Ndlys == Nfreqs and one blpair keep
    # the unnormalised bandpower covariance positive definite (smallest
    # eigenvalue ~1e-5 of the largest), so no eigenvalue sits near the
    # truncation threshold in PSpecData.get_MW. With only five integrations
    # it is rank deficient and the truncation would make the result
    # platform dependent.
    bls1, bls2 = even_odd_blpairs()
    dsets, dsets_std = lst_datasets(n_times=None)
    ds = PSpecData(
        dsets=dsets, wgts=[None, None], dsets_std=dsets_std, beam=dipole_beam()
    )
    ds.Jy_to_mK()
    uvp = ds.pspec(
        bls1[:1],
        bls2[:1],
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=[(20, 40)],
        norm="V^-1/2",
        cov_model="autos",
        verbose=False,
    )
    return uvpspec_outputs(uvp)


@regression_case("pspec_exact_norm_bh", rtol=LINALG_RTOL)
def _pspec_exact_norm_bh() -> dict[str, np.ndarray]:
    return uvpspec_outputs(run_even_odd_pspec(norm="I", taper="bh", exact_norm=True))


@regression_case("pspec_one_step_bh", rtol=LINALG_RTOL)
def _pspec_one_step_bh() -> dict[str, np.ndarray]:
    # one-step normalization (#329) with the default DelayResponse; the second
    # spw has Ndlys < Nfreqs. Covariances and window functions go through
    # BLAS, hence LINALG_RTOL.
    uvp = run_even_odd_pspec(
        norm="I",
        taper="bh",
        store_cov=True,
        cov_model="autos",
        normalization="one-step",
    )
    out = uvpspec_outputs(uvp)
    out["normalization"] = np.array(uvp.normalization)
    out["response_model"] = np.array(uvp.response_model)
    return out


@regression_case("pspec_exact_windows", rtol=LINALG_RTOL)
def _pspec_exact_windows() -> dict[str, np.ndarray]:
    return uvpspec_outputs(run_exact_windows_pspec())


# --------------------------------------------------------------------------
# Beams: compute_pspec_scalar and Jy_to_mK
# --------------------------------------------------------------------------

SCALAR_LOWER_FREQ = 120.0e6
SCALAR_UPPER_FREQ = 128.0e6
SCALAR_NUM_FREQS = 20
JY_TO_MK_FREQS = np.linspace(110e6, 190e6, 9)


def gauss_beam() -> PSpecBeamGauss:
    return PSpecBeamGauss(
        fwhm=0.25 * units.rad, beam_freqs=np.linspace(100e6, 200e6, 51) * units.Hz
    )


def from_array_beam() -> PSpecBeamFromArray:
    # Smooth, analytic solid angles (sr) over the band: Omega ~ nu^-2.
    beam_freqs = np.linspace(100e6, 200e6, 51)
    omega_p = 0.05 * (150e6 / beam_freqs) ** 2
    omega_pp = 0.025 * (150e6 / beam_freqs) ** 2
    return PSpecBeamFromArray(
        OmegaP=omega_p * units.sr,
        OmegaPP=omega_pp * units.sr,
        beam_freqs=beam_freqs * units.Hz,
    )


def beam_outputs(beam, pol: str) -> dict[str, np.ndarray]:
    out = {}
    for taper in ("none", "blackman-harris"):
        for little_h in (True, False):
            for noise_scalar in (False, True):
                for exact_norm in (False, True):
                    key = (
                        f"scalar_{taper}_h{int(little_h)}_noise{int(noise_scalar)}"
                        f"_exact{int(exact_norm)}"
                    )
                    out[key] = beam.compute_pspec_scalar(
                        SCALAR_LOWER_FREQ * units.Hz,
                        SCALAR_UPPER_FREQ * units.Hz,
                        SCALAR_NUM_FREQS,
                        num_steps=2000,
                        pol=pol,
                        taper=taper,
                        length_unit=conversions.length_unit_from_little_h(little_h),
                        noise_scalar=noise_scalar,
                        exact_norm=exact_norm,
                    )
    out["Jy_to_mK"] = beam.Jy_to_mK(JY_TO_MK_FREQS * units.Hz, pol=pol)
    out["power_beam_int"] = beam.power_beam_int(pol=pol)
    out["power_beam_sq_int"] = beam.power_beam_sq_int(pol=pol)
    return out


@regression_case("beam_uv_dipole_xx")
def _beam_uv_dipole_xx() -> dict[str, np.ndarray]:
    return beam_outputs(dipole_beam(), pol="xx")


@regression_case("beam_uv_pstokes_pI")
def _beam_uv_pstokes_pi() -> dict[str, np.ndarray]:
    return beam_outputs(pstokes_beam(), pol="pI")


@regression_case("beam_gauss_pI")
def _beam_gauss_pi() -> dict[str, np.ndarray]:
    return beam_outputs(gauss_beam(), pol="pI")


@regression_case("beam_from_array_pI")
def _beam_from_array_pi() -> dict[str, np.ndarray]:
    return beam_outputs(from_array_beam(), pol="pI")


# --------------------------------------------------------------------------
# UVWindow: cylindrical and spherical window functions
# --------------------------------------------------------------------------

WINDOW_BL_LENS = np.array([14.6, 29.2])
WINDOW_BL_WEIGHTS = [2.0, 1.0]


def make_uvwindow(little_h: bool) -> uvwindow.UVWindow:
    bandwidth = uvwindow.FTBeam.get_bandwidth(FT_BEAM_FILE)
    ftbeam = uvwindow.FTBeam.from_file(
        ftfile=FT_BEAM_FILE, freq_array=bandwidth[5:25] * units.Hz
    )
    kunits = cu.littleh / units.Mpc if little_h else 1 / units.Mpc
    return uvwindow.UVWindow(
        ftbeam_obj=ftbeam, taper="blackman-harris", kunits=kunits, verbose=False
    )


def window_outputs(little_h: bool) -> dict[str, np.ndarray]:
    uvw = make_uvwindow(little_h)
    out = {}
    kperp, kpara, cyl_wf = uvw.get_cylindrical_wf(
        WINDOW_BL_LENS[0] * units.m,
        kperp_bins=None,
        kpara_bins=None,
        return_bins="unweighted",
    )
    out["cyl_kperp"] = kperp
    out["cyl_kpara"] = kpara
    out["cyl_wf"] = cyl_wf

    # spherical bins that cover every |k| probed, so no "not included" warning
    kperp_bins = uvw.get_kperp_bins(WINDOW_BL_LENS * units.m)
    kpara_bins = uvw.get_kpara_bins(uvw.freq_array * units.Hz)
    ktot_max = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2).max()
    ktot_min = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2).min()
    dk = 0.1 if little_h else 0.1 * uvw.cosmo.h
    kbins = np.arange(ktot_min, ktot_max + dk, step=dk) * uvw.kunits
    sph_wf, weighted_k = uvw.get_spherical_wf(
        kbins=kbins,
        bl_lens=WINDOW_BL_LENS * units.m,
        bl_weights=WINDOW_BL_WEIGHTS,
        kperp_bins=kperp_bins,
        kpara_bins=kpara_bins,
        return_weighted_k=True,
    )
    out["sph_kbins"] = kbins.value
    out["sph_kunits"] = np.array(str(uvw.kunits))
    out["sph_wf"] = sph_wf
    out["sph_weighted_k"] = weighted_k
    out["kperp_bins"] = kperp_bins.value
    out["kpara_bins"] = kpara_bins.value
    return out


@regression_case("uvwindow_littleh", rtol=LINALG_RTOL)
def _uvwindow_littleh() -> dict[str, np.ndarray]:
    return window_outputs(little_h=True)


@regression_case("uvwindow_no_littleh", rtol=LINALG_RTOL)
def _uvwindow_no_littleh() -> dict[str, np.ndarray]:
    return window_outputs(little_h=False)


# --------------------------------------------------------------------------
# grouping.spherical_average and grouping.spherical_wf_from_uvp
# --------------------------------------------------------------------------

SPH_K_UNIT = cu.littleh / units.Mpc
SPH_KBINS = np.arange(0.05, 2.0, 0.1) * SPH_K_UNIT
SPH_BIN_WIDTHS = 0.1 * SPH_K_UNIT


@regression_case("spherical_average", rtol_overrides={"weighted_cov": LINALG_RTOL})
def _spherical_average() -> dict[str, np.ndarray]:
    # even x even: spherical_average sizes its arrays with uvp.Ntimes, which
    # counts time_1 and time_2 separately and breaks for even x odd inputs.
    # Ndlys == Nfreqs in every spw: the window-function averaging in
    # spherical_average does not support Ndlys < Nfreqs.
    uvp = run_even_odd_pspec(
        halves=("even", "even"),
        n_dlys=None,
        norm="I",
        taper="bh",
        store_cov=True,
        cov_model="autos",
        length_unit=LITTLEH_LENGTH,
    )
    out = {}
    design = {}
    sph = grouping.spherical_average(uvp, SPH_KBINS, SPH_BIN_WIDTHS, A=design)
    out.update(uvpspec_outputs(sph, prefix="uniform_"))
    for spw in sph.spw_array:
        out[f"uniform_A_{spw}"] = design[spw]

    # the same bins given in Mpc^-1 must give the same grid
    sph_noh = grouping.spherical_average(
        uvp,
        SPH_KBINS.value * uvp.cosmo.h / units.Mpc,
        SPH_BIN_WIDTHS.value * uvp.cosmo.h / units.Mpc,
    )
    out.update(uvpspec_outputs(sph_noh, prefix="noh_"))

    sph_cov = grouping.spherical_average(
        uvp, SPH_KBINS, SPH_BIN_WIDTHS, weight_by_cov=True, time_avg=True
    )
    out.update(uvpspec_outputs(sph_cov, prefix="weighted_cov_"))
    return out


SPH_WF_KBIN_EDGES = np.arange(0.075, 2.9, 0.25) * SPH_K_UNIT


@regression_case("spherical_exact_windows", rtol=LINALG_RTOL)
def _spherical_exact_windows() -> dict[str, np.ndarray]:
    uvp = run_exact_windows_pspec()
    out = {}
    wf = grouping.spherical_wf_from_uvp(uvp, kbin_edges=SPH_WF_KBIN_EDGES)
    for spw, arr in wf.items():
        out[f"wf_from_uvp_{spw}"] = arr
    wf_theory = grouping.spherical_wf_from_uvp(
        uvp,
        kbin_edges=SPH_WF_KBIN_EDGES,
        kbin_edges_theory=np.arange(0.075, 2.9, 0.125) * SPH_K_UNIT,
    )
    for spw, arr in wf_theory.items():
        out[f"wf_from_uvp_theory_{spw}"] = arr
    kbins = (SPH_WF_KBIN_EDGES[1:] + SPH_WF_KBIN_EDGES[:-1]) / 2
    sph = grouping.spherical_average(uvp, kbins, np.diff(SPH_WF_KBIN_EDGES))
    out.update(uvpspec_outputs(sph, prefix="sph_"))
    return out


# --------------------------------------------------------------------------
# noise: calc_P_N, Sensitivity, utils.uvp_noise_error
# --------------------------------------------------------------------------

NOISE_TSYS = 300.0 * units.K
NOISE_T_INT = 10.7 * units.s
NOISE_K = np.linspace(0.05, 2.0, 8)


@regression_case("noise_calc_P_N")
def _noise_calc_p_n() -> dict[str, np.ndarray]:
    out = {}
    scalar = 568847837.72586381
    for component in ("real", "abs"):
        out[f"Pk_{component}"] = noise.calc_P_N(
            scalar,
            NOISE_TSYS,
            NOISE_T_INT,
            Ncoherent=2,
            Nincoherent=4,
            form="Pk",
            component=component,
        )
        out[f"DelSq_{component}"] = noise.calc_P_N(
            scalar,
            NOISE_TSYS,
            NOISE_T_INT,
            Ncoherent=2,
            Nincoherent=4,
            form="DelSq",
            k=NOISE_K,
            component=component,
        )
    return out


@regression_case("noise_sensitivity")
def _noise_sensitivity() -> dict[str, np.ndarray]:
    out = {}
    freqs = np.linspace(150e6, 160e6, 100, endpoint=False)
    for little_h in (True, False):
        tag = "h" if little_h else "noh"
        sense = noise.Sensitivity(beam=pstokes_beam())
        sense.calc_scalar(
            freqs * units.Hz,
            "pI",
            num_steps=2000,
            length_unit=conversions.length_unit_from_little_h(little_h),
        )
        out[f"scalar_{tag}"] = sense.scalar
        out[f"P_N_Pk_{tag}"] = sense.calc_P_N(
            NOISE_TSYS, NOISE_T_INT, Ncoherent=1, Nincoherent=1, form="Pk"
        )
        out[f"P_N_DelSq_{tag}"] = sense.calc_P_N(
            NOISE_TSYS, NOISE_T_INT, Ncoherent=1, Nincoherent=1, form="DelSq", k=NOISE_K
        )
    return out


@regression_case("noise_uvp_noise_error")
def _noise_uvp_noise_error() -> dict[str, np.ndarray]:
    out = {}
    auto_tsys = utils.uvd_to_Tsys(lst_uvdata("even"), dipole_beam())
    for little_h in (True, False):
        tag = "h" if little_h else "noh"
        # even x even so the auto Tsys times match both time arrays
        uvp = run_even_odd_pspec(
            halves=("even", "even"),
            norm="I",
            taper="bh",
            length_unit=conversions.length_unit_from_little_h(little_h),
        )
        utils.uvp_noise_error(
            uvp,
            copy.deepcopy(auto_tsys),
            err_type=["P_N", "P_SN"],
            P_SN_correction=True,
            length_unit=conversions.length_unit_from_little_h(little_h),
        )
        for spw in uvp.spw_array:
            for stat in ("P_N", "P_SN"):
                out[f"{stat}_{tag}_{spw}"] = uvp.stats_array[stat][spw]
    out["auto_Tsys"] = auto_tsys.data_array
    return out


# --------------------------------------------------------------------------
# Legacy UVPSpec file (written 2019)
# --------------------------------------------------------------------------

LEGACY_UVP_FILE = DATA_PATH / "test_uvp.h5"


def read_legacy_uvp() -> UVPSpec:
    uvp = UVPSpec()
    uvp.read_hdf5(LEGACY_UVP_FILE)
    return uvp


@regression_case("legacy_test_uvp_h5")
def _legacy_test_uvp_h5() -> dict[str, np.ndarray]:
    uvp = read_legacy_uvp()
    out = uvpspec_outputs(uvp)
    # keep the fixture small: the full data array is 240 x 64
    out["data_array_0"] = uvp.data_array[0][:12]
    out["get_kperps_h_0"] = out["get_kperps_h_0"][:12]
    out["get_kperps_noh_0"] = out["get_kperps_noh_0"][:12]
    out["units"] = np.array(unit_helpers.unit_to_string(uvp.units))
    cosmo_params = conversions.params_from_cosmology(uvp.cosmo)
    out["cosmo_params"] = np.array([cosmo_params[key] for key in sorted(cosmo_params)])
    out["cosmo_param_names"] = np.array(sorted(cosmo_params))
    return out

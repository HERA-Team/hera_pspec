"""Time realistic hera_pspec workflows.

Run from the repository root::

    uv run python benchmarks/run_benchmarks.py --output /path/outside/repo/bench.json

Each workflow builds its inputs once, outside the timed region. Before every
timed call an untimed ``prepare`` step hands the call fresh copies of anything
it mutates, so every call does the same work. One warm-up call is made and
discarded, then ``--repeat`` calls are timed with ``time.perf_counter``. The
garbage collector is run before, and disabled during, every timed call.

See ``benchmarks/README.md`` for how to compare two runs.
"""

import argparse
import copy
import datetime as dt
import functools
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy
import astropy.cosmology.units as cu
import numpy as np
import pyuvdata
import scipy
from astropy import units as u
from pyuvdata import UVData

import hera_pspec
from hera_pspec import PSpecBeamUV, PSpecData, UVPSpec, grouping, utils, uvwindow
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)
REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

# LST-binned HERA cutout: 55 baselines, 10 integrations, 150 channels, xx.
# Both datasets in the pspec are the "even" file (same JDs), so the output has
# time_1 == time_2 and can be fed to grouping.spherical_average.
LST_FILE = "zen.even.xx.LST.1.28828.uvOCRSA"
LST_STD_FILE = "zen.even.std.xx.LST.1.28828.uvOCRSA"
BEAM_FILE = "HERA_NF_dipole_power.beamfits"
FT_BEAM_FILE = "FT_beam_HERA_dipole_test_xx.hdf5"

# Two 40-channel spectral windows; Ndlys == Nfreqs in both (spherical_average
# with stored window functions needs that).
PSPEC_SPW_RANGES = [(20, 60), (90, 130)]
PSPEC_TAPER = "blackman-harris"

# |k| bins for spherical_average, in h Mpc^-1.
SPH_KBINS = np.arange(0.05, 2.0, 0.05) * cu.littleh / u.Mpc
SPH_BIN_WIDTH = 0.05 * cu.littleh / u.Mpc

# Baseline lengths (m) for the UVWindow spherical window functions. Many
# lengths, as for a full set of HERA redundant groups.
UVWINDOW_BL_LENS = np.linspace(14.6, 140.0, 120)
UVWINDOW_DK = 0.05

# The HDF5 workflows use a pspec output over the same blpairs with three spws
# and no covariance or window functions (~6 MB on disk). With cov and window
# functions the file is ~20x bigger and the timing measures disk bandwidth,
# not hera_pspec. One write takes ~4 ms and one read ~12 ms, so each timed
# call repeats the write/read this many times to one reused file.
HDF5_SPW_RANGES = [(10, 50), (55, 95), (100, 140)]
HDF5_WRITE_NCALLS = 200
HDF5_READ_NCALLS = 100


def read_miriad(filename: str) -> UVData:
    uvd = UVData()
    uvd.read_miriad(str(DATA_PATH / filename))
    return uvd


def redundant_blpairs(uvd: UVData) -> tuple[list, list]:
    """All ordered cross-baseline pairs within every redundant group."""
    reds, _, _ = utils.get_reds(uvd, pick_data_ants=True)
    bls1, bls2 = [], []
    for group in reds:
        if len(group) < 2:
            continue
        group_bls1, group_bls2, _ = utils.construct_blpairs(
            group, exclude_auto_bls=True, exclude_permutations=False
        )
        bls1.extend(group_bls1)
        bls2.extend(group_bls2)
    return bls1, bls2


class PSpecInputs:
    """Visibilities, beam and baseline pairs for the pspec workflows."""

    def __init__(self):
        self.uvd = read_miriad(LST_FILE)
        self.uvd_std = read_miriad(LST_STD_FILE)
        self.beam = PSpecBeamUV(str(DATA_PATH / BEAM_FILE))
        self.bls1, self.bls2 = redundant_blpairs(self.uvd)

    def fresh(self) -> dict[str, Any]:
        """Deep copies of everything PSpecData/Jy_to_mK/pspec may mutate."""
        return {
            "dsets": [copy.deepcopy(self.uvd), copy.deepcopy(self.uvd)],
            "dsets_std": [copy.deepcopy(self.uvd_std), copy.deepcopy(self.uvd_std)],
            "beam": copy.deepcopy(self.beam),
            "bls1": copy.deepcopy(self.bls1),
            "bls2": copy.deepcopy(self.bls2),
        }


def run_pspec(state: dict[str, Any], exact_norm: bool) -> UVPSpec:
    """Build a PSpecData, convert to mK and estimate the power spectra."""
    ds = PSpecData(
        dsets=state["dsets"],
        wgts=[None, None],
        dsets_std=state["dsets_std"],
        beam=state["beam"],
    )
    ds.Jy_to_mK()
    return ds.pspec(
        state["bls1"],
        state["bls2"],
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=PSPEC_SPW_RANGES,
        taper=PSPEC_TAPER,
        store_cov=True,
        cov_model="autos",
        exact_norm=exact_norm,
        verbose=False,
    )


@functools.cache
def pspec_inputs() -> PSpecInputs:
    return PSpecInputs()


@functools.cache
def pspec_output() -> UVPSpec:
    """The pspec_full output, used as input by spherical_average."""
    return run_pspec(pspec_inputs().fresh(), exact_norm=False)


def uvpspec_sizes(uvp: UVPSpec) -> dict[str, Any]:
    spw_ranges = uvp.get_spw_ranges()
    return {
        "Nblpairs": int(uvp.Nblpairs),
        "Ntimes": int(uvp.Ntimes),
        "Nbltpairs": int(uvp.Nbltpairs),
        "Npols": int(uvp.Npols),
        "Nspws": int(uvp.Nspws),
        "spws": [
            {"Nfreqs": int(spw_range[2]), "Ndlys": int(spw_range[3])}
            for spw_range in spw_ranges
        ],
        "has_cov": bool(hasattr(uvp, "cov_array_real")),
        "has_window": bool(hasattr(uvp, "window_function_array")),
    }


# --------------------------------------------------------------------------
# Workflows
# --------------------------------------------------------------------------


@dataclass
class Workflow:
    """A named workflow.

    ``setup`` runs once (untimed) and returns a context. ``prepare(context)``
    runs before every call (untimed) and returns the state for that call.
    ``run(state)`` is the timed call. ``sizes(context)`` describes the inputs.
    """

    name: str
    description: str
    setup: Callable[[], Any]
    run: Callable[[Any], Any]
    prepare: Callable[[Any], Any] = field(default=lambda context: context)
    sizes: Callable[[Any], dict[str, Any]] = field(default=lambda context: {})
    teardown: Callable[[Any], None] = field(default=lambda context: None)


def _pspec_sizes(inputs: PSpecInputs) -> dict[str, Any]:
    sizes = uvpspec_sizes(pspec_output())
    sizes["Nbls_input"] = int(inputs.uvd.Nbls)
    sizes["Nfreqs_input"] = int(inputs.uvd.Nfreqs)
    return sizes


PSPEC_FULL = Workflow(
    name="pspec_full",
    description=(
        "PSpecData + Jy_to_mK + pspec over every redundant blpair of the LST "
        "cutout (even x even), 2 spws, blackman-harris taper, PSpecBeamUV, "
        "store_cov (autos), little_h default"
    ),
    setup=pspec_inputs,
    prepare=lambda inputs: inputs.fresh(),
    run=lambda state: run_pspec(state, exact_norm=False),
    sizes=_pspec_sizes,
)

PSPEC_FULL_EXACT_NORM = Workflow(
    name="pspec_full_exact_norm",
    description="pspec_full with exact_norm=True",
    setup=pspec_inputs,
    prepare=lambda inputs: inputs.fresh(),
    run=lambda state: run_pspec(state, exact_norm=True),
    sizes=_pspec_sizes,
)

SPHERICAL_AVERAGE = Workflow(
    name="spherical_average",
    description=(
        "grouping.spherical_average of the pspec_full output (with cov and "
        "window functions) onto a |k| grid, no time averaging"
    ),
    setup=pspec_output,
    prepare=lambda uvp: copy.deepcopy(uvp),
    run=lambda uvp: grouping.spherical_average(uvp, SPH_KBINS, SPH_BIN_WIDTH),
    sizes=lambda uvp: {**uvpspec_sizes(uvp), "Nkbins": len(SPH_KBINS)},
)


def _uvwindow_setup() -> dict[str, Any]:
    ft_file = DATA_PATH / FT_BEAM_FILE
    bandwidth = uvwindow.FTBeam.get_bandwidth(ft_file)
    ftbeam = uvwindow.FTBeam.from_file(ftfile=ft_file, freq_array=bandwidth)
    uvw = uvwindow.UVWindow(
        ftbeam_obj=ftbeam, taper=PSPEC_TAPER, little_h=True, verbose=False
    )
    kperp_bins = uvw.get_kperp_bins(UVWINDOW_BL_LENS)
    kpara_bins = uvw.get_kpara_bins(uvw.freq_array)
    ktot = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2)
    # bins that cover every |k| probed, so nothing falls outside the grid
    kbins = (
        np.arange(ktot.min(), ktot.max() + UVWINDOW_DK, step=UVWINDOW_DK) * uvw.kunits
    )
    return {
        "uvw": uvw,
        "kbins": kbins,
        "kperp_bins": kperp_bins,
        "kpara_bins": kpara_bins,
    }


def _uvwindow_run(state: dict[str, Any]):
    return state["uvw"].get_spherical_wf(
        kbins=state["kbins"],
        bl_lens=UVWINDOW_BL_LENS,
        kperp_bins=state["kperp_bins"],
        kpara_bins=state["kpara_bins"],
        return_weighted_k=True,
    )


UVWINDOW_SPHERICAL_WF = Workflow(
    name="uvwindow_spherical_wf",
    description=(
        "UVWindow.get_spherical_wf from the HERA dipole FT beam over many "
        "baseline lengths, blackman-harris taper, little_h"
    ),
    setup=_uvwindow_setup,
    prepare=lambda context: {**context, "uvw": copy.deepcopy(context["uvw"])},
    run=_uvwindow_run,
    sizes=lambda context: {
        "Nbl_lens": len(UVWINDOW_BL_LENS),
        "Nfreqs": int(context["uvw"].Nfreqs),
        "Nkbins": len(context["kbins"]),
        "Nkperp_bins": len(context["kperp_bins"]),
        "Nkpara_bins": len(context["kpara_bins"]),
    },
)


@functools.cache
def hdf5_uvpspec() -> UVPSpec:
    state = pspec_inputs().fresh()
    ds = PSpecData(
        dsets=state["dsets"],
        wgts=[None, None],
        dsets_std=state["dsets_std"],
        beam=state["beam"],
    )
    ds.Jy_to_mK()
    return ds.pspec(
        state["bls1"],
        state["bls2"],
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=HDF5_SPW_RANGES,
        taper=PSPEC_TAPER,
        store_window=False,
        verbose=False,
    )


def _hdf5_setup() -> dict[str, Any]:
    tmpdir = Path(tempfile.mkdtemp(prefix="hera_pspec_bench_"))
    uvp = hdf5_uvpspec()
    path = tmpdir / "uvp.h5"
    uvp.write_hdf5(str(path))
    return {"tmpdir": tmpdir, "uvp": uvp, "path": path}


def _hdf5_write_run(state: dict[str, Any]) -> None:
    # write_hdf5(overwrite=True) prints a line per call, so remove the file
    # instead; unlinking is negligible next to the write itself.
    for _ in range(HDF5_WRITE_NCALLS):
        state["path"].unlink(missing_ok=True)
        state["uvp"].write_hdf5(str(state["path"]))


def _hdf5_read_prepare(context: dict[str, Any]) -> dict[str, Any]:
    if not context["path"].exists():
        context["uvp"].write_hdf5(str(context["path"]))
    return context


def _hdf5_read_run(state: dict[str, Any]) -> None:
    for _ in range(HDF5_READ_NCALLS):
        uvp = UVPSpec()
        uvp.read_hdf5(str(state["path"]))


def _hdf5_sizes(context: dict[str, Any]) -> dict[str, Any]:
    sizes = uvpspec_sizes(context["uvp"])
    sizes["Nwrite_calls"] = HDF5_WRITE_NCALLS
    sizes["Nread_calls"] = HDF5_READ_NCALLS
    sizes["file_size_MB"] = round(context["path"].stat().st_size / 1e6, 2)
    return sizes


def _hdf5_teardown(context: dict[str, Any]) -> None:
    shutil.rmtree(context["tmpdir"], ignore_errors=True)


HDF5_WRITE = Workflow(
    name="hdf5_write",
    description=(
        f"{HDF5_WRITE_NCALLS} x UVPSpec.write_hdf5 of a 3-spw pspec output "
        "(same blpairs as pspec_full, no cov or window functions)"
    ),
    setup=_hdf5_setup,
    run=_hdf5_write_run,
    sizes=_hdf5_sizes,
    teardown=_hdf5_teardown,
)

HDF5_READ = Workflow(
    name="hdf5_read",
    description=(f"{HDF5_READ_NCALLS} x UVPSpec.read_hdf5 of the hdf5_write file"),
    setup=_hdf5_setup,
    prepare=_hdf5_read_prepare,
    run=_hdf5_read_run,
    sizes=_hdf5_sizes,
    teardown=_hdf5_teardown,
)

WORKFLOWS: dict[str, Workflow] = {
    workflow.name: workflow
    for workflow in [
        PSPEC_FULL,
        PSPEC_FULL_EXACT_NORM,
        SPHERICAL_AVERAGE,
        UVWINDOW_SPHERICAL_WF,
        HDF5_WRITE,
        HDF5_READ,
    ]
}

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def timed_call(workflow: Workflow, context: Any) -> float:
    state = workflow.prepare(context)
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        workflow.run(state)
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed


def benchmark(workflow: Workflow, repeat: int) -> dict[str, Any]:
    print(f"[{workflow.name}] setting up", flush=True)
    context = workflow.setup()
    try:
        sizes = workflow.sizes(context)
        warmup = timed_call(workflow, context)
        print(f"[{workflow.name}] warm-up {warmup:.3f} s", flush=True)
        times = []
        for index in range(repeat):
            elapsed = timed_call(workflow, context)
            times.append(elapsed)
            print(
                f"[{workflow.name}] repeat {index + 1}/{repeat} {elapsed:.3f} s",
                flush=True,
            )
    finally:
        workflow.teardown(context)
    return {
        "description": workflow.description,
        "sizes": sizes,
        "warmup_s": warmup,
        "times_s": times,
        "min_s": min(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def blas_info() -> str | None:
    try:
        config = np.show_config(mode="dicts")
        blas = config["Build Dependencies"]["blas"]
        return f"{blas.get('name')} {blas.get('version')}"
    except Exception:
        return None


def environment() -> dict[str, Any]:
    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
        "pyuvdata": pyuvdata.__version__,
        "hera_pspec": hera_pspec.__version__,
        "blas": blas_info(),
        "thread_env": {name: os.environ.get(name) for name in thread_vars},
        "git": git_info(),
    }


def print_summary(results: dict[str, Any]) -> None:
    header = (
        f"{'workflow':<24} {'min':>9} {'median':>9} {'mean':>9} {'std':>8} {'std%':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        std_pct = 100 * result["std_s"] / result["mean_s"]
        print(
            f"{name:<24} {result['min_s']:9.3f} {result['median_s']:9.3f} "
            f"{result['mean_s']:9.3f} {result['std_s']:8.3f} {std_pct:6.1f}"
        )
    print("(times in seconds)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="number of timed calls per workflow, after one warm-up (default 5)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(WORKFLOWS),
        metavar="NAME",
        help=f"run only these workflows (choices: {', '.join(WORKFLOWS)})",
    )
    parser.add_argument(
        "--output", type=Path, help="write the results as JSON to this path"
    )
    parser.add_argument(
        "--list", action="store_true", help="list the workflows and exit"
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for workflow in WORKFLOWS.values():
            print(f"{workflow.name}: {workflow.description}")
        return 0

    names = args.only or list(WORKFLOWS)
    env = environment()
    total_start = time.perf_counter()
    results = {name: benchmark(WORKFLOWS[name], args.repeat) for name in names}
    total = time.perf_counter() - total_start

    print_summary(results)
    print(f"total wall time {total:.1f} s")

    if args.output is not None:
        payload = {
            "environment": env,
            "settings": {"repeat": args.repeat, "total_wall_s": total},
            "workflows": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

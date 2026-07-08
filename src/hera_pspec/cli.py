"""A CLI interface for hera_pspec."""

import copy
import datetime
import glob
import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

import h5py
import numpy as np
from cyclopts import App, Parameter
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

cns = Console()
logger = logging.getLogger("hera_pspec")

app = App(name="pspec", version_flags=[], help_flags=["--help"])
# cyclopts pattern: register subcommands after app is constructed
import pyuvdata  # noqa: E402
from pyuvdata import UVData  # noqa: E402

from . import container, grouping, pspecdata, pstokes, utils  # noqa: E402
from .uvpspec import recursive_combine_uvpspec  # noqa: E402


@app.command
def fast_merge_baselines(
    *,
    pattern: str,
    group: str,
    names: list[str],
    outpath: Path,
    progress: bool = True,
    extras: list[str] | None = None,
    batch_size: int | None = None,
) -> None:
    """Merge a set of hera_pspec files each representing a single baseline, into one.

    This can be useful because reading a single file with many baselines is much much
    faster than reading many files each with a single baseline currently.

    Parameters
    ----------
    pattern
        A glob pattern to match the files to be merged. For example,
        '/path/to/files/blpair.*.h5'. Each file should be a valid PspecContainer file.
    group
        The group name wihtin the PSpecContainer in which the UVPSpec objects that you
        wish to merge are stored.
    names
        The names of the UVPSpec objects within the group to be merged. These should be
        the same for all files. Multiple names can be provided (via multiple --names
        flags), and they will be merged into the same file.
    outpath
        The basename of the output file. This can be a full path, but note that the
        final output pspec file will have an extension of '.pspec.h5' added to it.An
        --extras specified will be written to separate files with the same basenamebut
        a suffix of '.{extraname}.pkl'.
    progress
        Whether to show a progress bar while loading the files. This is useful for large
        datasets, but can be turned off for small datasets.
    extras
        A list of extra attributes to be saved from the header of the files. These will
        be saved to separate files with the same basename as the output file, but with a
        suffix of '.{extraname}.pkl'. This is useful for saving metadata that is not
        stored in the UVPSpec objects themselves.
    batch_size
        Number of files to load and merge at a time. Smaller batch sizes use less memory
        but may be slightly slower. If None (default), all files are loaded at once.
        Adjust this based on available RAM and file sizes.
    """
    if extras is None:
        extras = []
    extra_attrs = {extra: {} for extra in extras}

    files = sorted(glob.glob(pattern))
    cns.print(f"Found {len(files)} files matching pattern.")

    # Determine if we're processing in batches
    if batch_size is None:
        batch_size = len(files)  # Process all at once

    # Initialize accumulated merged results for each name
    merged_uvps = {name: None for name in names}

    # Process files in batches
    num_batches = (len(files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        if num_batches > 1:
            cns.print(
                f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)"
            )

        # Load UVPSpec objects for this batch
        uvps_batch = {name: [] for name in names}

        for df in tqdm(
            batch_files,
            desc=f"Loading batch {batch_idx + 1}/{num_batches}",
            unit="file",
            disable=not progress,
        ):
            # load power spectra
            psc = container.PSpecContainer(df, mode="r", keep_open=False)

            # Load both the time-averaged and not-time-averaged power spectra.
            # The time-averaging done in the single-baseline notebook has more
            # accurate noise calculations that can only be done when the interleaves
            # are separate.
            for name in names:
                uvp = psc.get_pspec(group, name)
                blp = uvp.get_blpairs()[0]
                uvps_batch[name].append(uvp)

            if extras:
                # load additional metadata stored in header
                with h5py.File(df, "r") as f:
                    for extra in extras:
                        extra_attrs[extra][blp] = f["header"].attrs[extra]

        # Merge the batch
        for name, uvplist in uvps_batch.items():
            batch_merged = recursive_combine_uvpspec(uvplist)

            # Combine with previous batches
            if merged_uvps[name] is None:
                merged_uvps[name] = batch_merged
            else:
                merged_uvps[name] = recursive_combine_uvpspec(
                    [merged_uvps[name], batch_merged]
                )

        # Clear batch data to free memory
        del uvps_batch

    cns.print("Writing merged power spectra to file")
    outspec = outpath.parent / f"{outpath.name}.pspec.h5"
    psc = container.PSpecContainer(outspec, mode="rw", keep_open=False)
    for name, uvp in merged_uvps.items():
        psc.set_pspec(group, name, uvp, overwrite=True)

    cns.print(f"Wrote pspecs to file: {outspec}")
    for name, extra in extra_attrs.items():
        fname = outpath.parent / f"{outpath.name}.{name}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(extra, f)
        cns.print(f"Wrote {fname}")


# `run` and `bootstrap` are thin wrappers: their parameters deliberately use the
# wrapped library function's names so cyclopts maps each option's --help text from
# the reused __doc__ (assigned just after each definition). Annotated
# Parameter(name=...) preserves the original CLI flag spelling (e.g. --output/-o,
# --blpair-group) even though the Python parameter was renamed to the library name.
@app.command
def run(
    dsets: list[Path],
    /,
    *,
    filename: Annotated[Path, Parameter(name=["--output", "-o"])],
    dsets_std: list[Path] | None = None,
    groupname: str | None = None,
    dset_pairs: list[tuple[int, int]] | None = None,
    dset_labels: list[str] | None = None,
    spw_ranges: list[tuple[int, int]] | None = None,
    n_dlys: list[int] | None = None,
    pol_pairs: list[tuple[str, str]] | None = None,
    blpairs: list[tuple[int, int, int, int]] | None = None,
    input_data_weight: Literal["identity", "iC", "dayenu"] = "identity",
    norm: Literal["I", "H^-1", "V^-1/2"] = "I",
    taper: str = "none",
    beam: Path | None = None,
    cosmo: list[float] | None = None,
    rephase_to_dset: int | None = None,
    trim_dset_lsts: bool = False,
    broadcast_dset_flags: bool = False,
    time_thresh: float = 0.2,
    Jy2mK: bool = False,
    exclude_auto_bls: bool = False,
    exclude_cross_bls: bool = False,
    exclude_permutations: bool = False,
    Nblps_per_group: int | None = None,
    bl_len_range: tuple[float, float] = (0.0, 1e10),
    bl_deg_range: tuple[float, float] = (0.0, 180.0),
    bl_error_tol: float = 1.0,
    store_cov: bool = False,
    store_cov_diag: bool = False,
    return_q: bool = False,
    overwrite: bool = False,
    cov_model: Literal[
        "empirical", "dsets", "autos", "foreground_dependent"
    ] = "empirical",
    psname_ext: str = "",
    verbose: bool = False,
    file_type: str = "uvh5",
    filter_extensions: list[tuple[int, int]] | None = None,
    symmetric_taper: bool = True,
    include_autocorrs: bool = False,
    include_crosscorrs: bool = True,
    interleave_times: bool = False,
    xant_flag_thresh: float = 0.95,
    store_window: bool = False,
    allow_fft: bool = False,
) -> None:
    pspecdata.pspec_run(
        dsets=[str(d) for d in dsets],
        filename=str(filename),
        dsets_std=[str(d) for d in dsets_std] if dsets_std is not None else None,
        groupname=groupname,
        dset_pairs=dset_pairs,
        dset_labels=dset_labels,
        spw_ranges=spw_ranges,
        n_dlys=n_dlys,
        pol_pairs=pol_pairs,
        blpairs=(
            [((a, b), (c, d)) for (a, b, c, d) in blpairs]
            if blpairs is not None
            else None
        ),
        input_data_weight=input_data_weight,
        norm=norm,
        taper=taper,
        beam=str(beam) if beam is not None else None,
        cosmo=cosmo,
        rephase_to_dset=rephase_to_dset,
        trim_dset_lsts=trim_dset_lsts,
        broadcast_dset_flags=broadcast_dset_flags,
        time_thresh=time_thresh,
        Jy2mK=Jy2mK,
        exclude_auto_bls=exclude_auto_bls,
        exclude_cross_bls=exclude_cross_bls,
        exclude_permutations=exclude_permutations,
        Nblps_per_group=Nblps_per_group,
        bl_len_range=bl_len_range,
        bl_deg_range=bl_deg_range,
        bl_error_tol=bl_error_tol,
        store_cov=store_cov,
        store_cov_diag=store_cov_diag,
        return_q=return_q,
        overwrite=overwrite,
        cov_model=cov_model,
        psname_ext=psname_ext,
        verbose=verbose,
        file_type=file_type,
        filter_extensions=filter_extensions,
        symmetric_taper=symmetric_taper,
        include_autocorrs=include_autocorrs,
        include_crosscorrs=include_crosscorrs,
        interleave_times=interleave_times,
        xant_flag_thresh=xant_flag_thresh,
        store_window=store_window,
        allow_fft=allow_fft,
        history=" ".join(sys.argv),
    )


run.__doc__ = pspecdata.pspec_run.__doc__


@app.command
def bootstrap(
    filename: Path,
    /,
    *,
    spectra: list[str] | None = None,
    blpair_groups: Annotated[list[str] | None, Parameter(name="--blpair-group")] = None,
    time_avg: bool = False,
    Nsamples: int = 100,
    seed: int = 0,
    normal_std: bool = True,
    robust_std: bool = False,
    cintervals: list[float] | None = None,
    keep_samples: bool = False,
    bl_error_tol: float = 1.0,
    overwrite: bool = False,
    add_to_history: str = "",
    verbose: bool = False,
) -> None:
    grouping.bootstrap_run(
        str(filename),
        spectra=spectra,
        blpair_groups=(
            [[int(tok) for tok in grp.split()] for grp in blpair_groups]
            if blpair_groups is not None
            else None
        ),
        time_avg=time_avg,
        Nsamples=Nsamples,
        seed=seed,
        normal_std=normal_std,
        robust_std=robust_std,
        cintervals=cintervals,
        keep_samples=keep_samples,
        bl_error_tol=bl_error_tol,
        overwrite=overwrite,
        add_to_history=add_to_history,
        verbose=verbose,
    )


bootstrap.__doc__ = grouping.bootstrap_run.__doc__


@app.command
def auto_noise(
    pspec_container: Path,
    auto_file: Path,
    beam: Path,
    /,
    *,
    groups: list[str] | None = None,
    spectra: list[str] | None = None,
    err_type: list[str] | None = None,
) -> None:
    """Compute noise error bars from autocorrelations (was auto_noise_run.py).

    Parameters
    ----------
    pspec_container
        HDF5 PSpecContainer with the input power spectra.
    auto_file
        UVData file of autocorr baselines for thermal-noise estimation.
    beam
        UVBeam file storing the primary beam.
    groups
        Power-spectrum groups to compute noise for (repeatable). Default: all.
    spectra
        Power-spectrum names to compute noise for (repeatable). Default: all in group.
    err_type
        Noise components to compute: 'P_N' and/or 'P_SN' (repeatable). Default ['P_N'].
    """
    if err_type is None:
        err_type = ["P_N"]
    uvd = UVData()
    uvd.read(str(auto_file))
    auto_Tsys = utils.uvd_to_Tsys(uvd, beam=str(beam))
    psc = container.PSpecContainer(
        str(pspec_container), keep_open=False, mode="rw", swmr=False
    )
    for group in groups if groups is not None else psc.groups():
        specs = spectra if spectra is not None else psc.spectra(group)
        for spec in specs:
            uvp = psc.get_pspec(group, spec)
            utils.uvp_noise_error(uvp, auto_Tsys, err_type=err_type)
            psc.set_pspec(group, spec, uvp, overwrite=True)
    psc.save()


@app.command
def generate_pstokes(
    inputdata: Path,
    /,
    *,
    pstokes_params: Annotated[list[str] | None, Parameter(name="--pstokes")] = None,
    outputdata: Path | None = None,
    clobber: bool = False,
    keep_vispols: bool = False,
) -> None:
    """Generate pseudo-Stokes visibilities from linpol files (was generate_pstokes_run.py).

    Parameters
    ----------
    inputdata
        UVData file with linearly polarized data to add pseudo-Stokes to.
    pstokes_params
        Pseudo-Stokes parameters to calculate (repeatable). Default ['pI'].
    outputdata
        Output filename. Defaults to inputdata (appends pstokes to linpols).
    clobber
        Overwrite the output file if it exists.
    keep_vispols
        Keep the original linear polarizations in the output.
    """
    if pstokes_params is None:
        pstokes_params = ["pI"]
    uvd = UVData()
    uvd.read(str(inputdata))
    out_path = str(outputdata) if outputdata is not None else str(inputdata)
    if keep_vispols:
        # if inplace, append new pstokes onto existing file.
        uvd_output = copy.deepcopy(uvd)
    else:
        # otherwise, output uvd does not contain original polarizations.
        uvd_output = pstokes.construct_pstokes(uvd, uvd, pstokes_params[0])
    for p in pstokes_params:
        if pyuvdata.utils.polstr2num(p) not in uvd_output.polarization_array:
            uvd_output += pstokes.construct_pstokes(uvd, uvd, pstokes=p)
    uvd_output.write_uvh5(out_path, clobber=clobber)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging output for CLI commands."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=cns, show_path=False)],
    )


def _load_uvp(pspec_file: Path, group: str | None, name: str | None):
    """Load a UVPSpec from a PSpecContainer, inferring group/name if unique."""
    psc = container.PSpecContainer(str(pspec_file), mode="r", keep_open=False)
    groups = psc.groups()
    if group is None:
        if len(groups) != 1:
            raise ValueError(
                f"{pspec_file} contains several groups ({groups}): "
                "specify one with --group."
            )
        group = groups[0]
    spectra = psc.spectra(group)
    if name is None:
        if len(spectra) != 1:
            raise ValueError(
                f"Group '{group}' contains several spectra ({spectra}): "
                "specify one with --name."
            )
        name = spectra[0]
    return psc.get_pspec(group, name)


def ft_beam_filename(
    label: str, pol: str, freq_array: np.ndarray, mapsize: float, npix: int
) -> str:
    """Standard FT-beam filename, encoding the differentiating parameters.

    The polarization must be the last underscore-separated token before
    ".hdf5": `FTBeam.from_file` falls back to extracting it from the
    filename for files that do not carry a `pol` attribute.
    """
    fmin = freq_array.min() / 1e6
    fmax = freq_array.max() / 1e6
    return (
        f"FT_beam_{label}_{fmin:.1f}-{fmax:.1f}MHz_"
        f"{freq_array.size}ch_ms{mapsize:g}_N{npix}_{pol}.hdf5"
    )


def _find_in_dirs(dirs: list[Path], filename: str) -> Path | None:
    """Return the first `dir/filename` that exists, searching left to right."""
    for d in dirs:
        cand = Path(d) / filename
        if cand.is_file():
            return cand
    return None


def wf_filename(polpair: str, dataset_label: str, spw: int) -> str:
    """Standard window-function filename (differentiating info baked in)."""
    return f"wf_exact_{polpair}_{dataset_label}_spw{spw:02d}.hdf5"


@app.command
def compute_ft_beam(
    beam_file: Path,
    pol: str,
    pspec_file: Path,
    /,
    *,
    out_dir: Path,
    group: str | None = None,
    name: str | None = None,
    mapsize: float = 1.0,
    npix: int = 301,
    label: str = "HERA",
    search_dirs: Annotated[list[Path] | None, Parameter(consume_multiple=True)] = None,
    force_recompute: bool = False,
    verbose: bool = False,
) -> None:
    """Ensure an FT-beam HDF5 file exists for the requested parameters.

    The file is readable by FTBeam.from_file and computed only if no
    matching file is found in the search directories. The frequency grid
    is taken from the data (pspec_file), so that the window functions'
    k_parallel axis comes out on the data's grid; to build an FT beam on
    an arbitrary grid, use FTBeam.from_beam directly. On success, prints
    a `FT_BEAM_PATH=<path>` line with the file to feed to
    `pspec compute-window-functions`.

    Parameters
    ----------
    beam_file
        Path to the UVBeam .fits/.beamfits beam simulation file.
    pol
        Polarization string, e.g. 'pI', 'xx', 'yy'.
    pspec_file
        PSpecContainer file whose frequency channels define the FT-beam
        grid.
    out_dir
        Directory where freshly computed FT-beam HDF5 files are written.
    group
        Group within pspec_file. May be omitted if the file has one group.
    name
        Spectrum name within the group. May be omitted if unique.
    mapsize
        Half-width of the flat-sky map the beam is projected onto (rad).
    npix
        Pixels per side of the Cartesian beam projection (odd preferred).
    label
        Instrument/beam label used in the output filename, e.g.
        'HERA_Vivaldi'.
    search_dirs
        Directories searched (in order, before computing anything) for a
        previously computed FT beam with matching parameters
        (space-separated).
        Matching is filename-based: the differentiating parameters are
        encoded in the filename. Use --force-recompute if anything changed
        without the filename reflecting it.
    force_recompute
        Skip the reuse lookup and recompute.
    verbose
        Debug-level logging.
    """
    from .uvwindow import FTBeam

    _setup_logging(verbose)

    # the frequency grid is the data's
    uvp = _load_uvp(pspec_file, group, name)
    freq_array = np.unique(uvp.freq_array)
    logger.info(
        "Using the %d frequency channels of %s (%.2f-%.2f MHz, channel width %.2f kHz)",
        freq_array.size,
        pspec_file,
        freq_array.min() / 1e6,
        freq_array.max() / 1e6,
        np.median(np.diff(freq_array)) / 1e3,
    )

    fname = ft_beam_filename(label, pol, freq_array, mapsize, npix)

    if not force_recompute:
        hit = _find_in_dirs([*(search_dirs or []), out_dir], fname)
        if hit is not None:
            logger.info("Reusing existing FT beam: %s", hit)
            print(f"FT_BEAM_PATH={hit.absolute()}")
            return

    ftbeam = FTBeam.from_beam(
        beamfile=beam_file, pol=pol, freq_array=freq_array, mapsize=mapsize, npix=npix
    )
    out_path = Path(out_dir) / fname
    ftbeam.write_hdf5(
        out_path,
        overwrite=True,
        extra_attrs={
            "beam_file": str(beam_file),
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "producer": "pspec compute-ft-beam",
        },
    )
    print(f"FT_BEAM_PATH={out_path.absolute()}")


def _compute_one_wf(
    pspec_file: str,
    group: str | None,
    name: str | None,
    spw: int,
    polpair: str,
    ft_beam_file: str,
    dataset_label: str,
    out_dir: str,
    wf_dirs: list[str],
    force_recompute: bool,
) -> str:
    """Compute (or reuse) the exact window functions of one (spw, polpair).

    Worker function for `compute_window_functions`; module-level so it can
    be used with ProcessPoolExecutor. Returns the path of the WF file.
    """
    from .uvwindow import FTBeam

    fname = wf_filename(polpair, dataset_label, spw)

    if not force_recompute:
        hit = _find_in_dirs([Path(d) for d in [*wf_dirs, out_dir]], fname)
        if hit is not None:
            logger.info("Reusing existing WF (spw=%d, pol=%s): %s", spw, polpair, hit)
            return str(hit.absolute())

    logger.info("Computing window functions for spw=%d, pol=%s", spw, polpair)
    uvp = _load_uvp(Path(pspec_file), group, name)
    sub = uvp.select(polpairs=[polpair], spws=[spw], inplace=False)
    del uvp

    # load the FT beam on the exact frequency channels of the spectral
    # window (reads only the relevant channel range from disk)
    spw_freqs = np.unique(sub.freq_array)
    ftbeam = FTBeam.from_file(ft_beam_file, freq_array=spw_freqs)

    kperp, kpara, wf = sub.get_exact_window_functions(ftbeam=ftbeam, inplace=False)
    # after select, the only spectral window is index 0
    kperp, kpara, wf = kperp[0], kpara[0], wf[0]

    out_path = Path(out_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("wf", data=wf, compression="gzip")
        f.create_dataset("kperp", data=kperp)
        f.create_dataset("kpara", data=kpara)
        f.attrs["algo"] = "exact"
        f.attrs["polpair"] = polpair
        f.attrs["spw_index"] = int(spw)
        f.attrs["dataset_label"] = dataset_label
        f.attrs["taper"] = str(sub.taper)
        f.attrs["spw_freq_min_hz"] = float(spw_freqs.min())
        f.attrs["spw_freq_max_hz"] = float(spw_freqs.max())
        f.attrs["pspec_file"] = str(pspec_file)
        f.attrs["ft_beam_file"] = str(ft_beam_file)
        f.attrs["created_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        f.attrs["producer"] = "pspec compute-window-functions"
    logger.info("Wrote %s", out_path)
    return str(out_path.absolute())


@app.command
def compute_window_functions(
    pspec_file: Path,
    ft_beam_file: Path,
    /,
    *,
    dataset_label: str,
    out_dir: Path,
    group: str | None = None,
    name: str | None = None,
    spws: Annotated[list[int] | None, Parameter(consume_multiple=True)] = None,
    polpairs: Annotated[list[str] | None, Parameter(consume_multiple=True)] = None,
    wf_dirs: Annotated[list[Path] | None, Parameter(consume_multiple=True)] = None,
    workers: int = 1,
    force_recompute: bool = False,
    verbose: bool = False,
) -> None:
    """Compute exact window functions per (spectral window, polpair).

    Window functions are computed with UVPSpec.get_exact_window_functions
    (Gorce et al. 2023 'exact' method) and written to one HDF5 file per
    (spw, polpair) (datasets: wf, kperp, kpara + provenance attrs).
    Existing files are reused unless --force-recompute. On success, prints
    one `WF_PATH=<path>` line per (spw, polpair) processed.

    Parameters
    ----------
    pspec_file
        PSpecContainer file to compute window functions for.
    ft_beam_file
        FT-beam HDF5 file (from `pspec compute-ft-beam`). It should be
        computed on the frequency channels of the data: if the grids
        differ, the FT of the beam is interpolated onto the data channels
        (with a warning).
    dataset_label
        Free-form string disambiguating runs whose data content differs
        (e.g. '131-nights'). Becomes part of the WF filename, so different
        labels never collide on disk.
    out_dir
        Directory to write fresh WF HDF5 files.
    group
        Group within pspec_file. May be omitted if the file has one group.
    name
        Spectrum name within the group. May be omitted if unique.
    spws
        Spectral window indices, 0-indexed, space-separated (e.g.
        `--spws 0 1 2`). Default: all.
    polpairs
        Polarization pairs, space-separated, e.g. 'pI'. Default: all in the
        file.
    wf_dirs
        Directories searched (in order, before computing anything) for
        previously computed WF files (space-separated). Matching is
        filename-based; use --force-recompute if upstream inputs changed
        without a new --dataset-label.
    workers
        Number of parallel worker processes over (spw, polpair) tasks.
        Each worker holds one spectral window's UVPSpec and FT beam in
        memory, so memory use scales with this.
    force_recompute
        Skip the reuse lookup and recompute.
    verbose
        Debug-level logging.
    """
    from . import uvpspec_utils as uvputils

    _setup_logging(verbose)

    # peek at the file to enumerate spws/polpairs
    uvp = _load_uvp(pspec_file, group, name)
    if spws is None or len(spws) == 0:
        spws = list(range(uvp.Nspws))
    else:
        bad = [s for s in spws if s >= uvp.Nspws]
        if bad:
            raise ValueError(f"spws {bad} out of range: file has Nspws={uvp.Nspws}.")
    if polpairs is None or len(polpairs) == 0:
        # for equal-pol pairs (e.g. ('pI','pI')), the single pol string is
        # the representation accepted by UVPSpec.select
        polpairs = [
            pp[0] if pp[0] == pp[1] else f"{pp[0]}{pp[1]}"
            for pp in [
                uvputils.polpair_int2tuple(p, pol_strings=True)
                for p in uvp.polpair_array
            ]
        ]
    del uvp

    jobs = [
        (
            str(pspec_file),
            group,
            name,
            spw,
            polpair,
            str(ft_beam_file),
            dataset_label,
            str(out_dir),
            [str(d) for d in (wf_dirs or [])],
            force_recompute,
        )
        for spw in spws
        for polpair in polpairs
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(_compute_one_wf, *zip(*jobs, strict=True)))
    else:
        paths = [_compute_one_wf(*job) for job in jobs]

    for path in paths:
        print(f"WF_PATH={path}")

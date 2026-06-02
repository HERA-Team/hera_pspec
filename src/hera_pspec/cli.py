"""A CLI interface for hera_pspec."""

import glob
import pickle
import sys
from pathlib import Path

import click
import h5py
import typer
from rich.console import Console
from tqdm import tqdm

cns = Console()

app = typer.Typer()
# typer pattern: register subcommands after app is constructed
from . import container, grouping, pspecdata  # noqa: E402
from .uvpspec import recursive_combine_uvpspec  # noqa: E402


@app.command()
def hello() -> None:
    # This is a test command which we need for the CLI interface to be broken into
    # subcommands (at least two commands need to be defined for it to be used as subc's)
    cns.print("Hi! :wave:")


_pattern_help = ()
_group_help = """The group name wihtin the PSpecContainer in which the UVPSpec objects
that you wish to merge are stored.
"""


@app.command()
def fast_merge_baselines(
    pattern: str = typer.Option(
        help=(
            "A glob pattern to match the files to be merged. For example, "
            "'/path/to/files/blpair.*.h5'. Each file should be a valid PspecContainer "
            "file."
        )
    ),
    group: str = typer.Option(
        help=(
            "The group name wihtin the PSpecContainer in which the UVPSpec objects "
            "that you wish to merge are stored."
        )
    ),
    names: list[str] = typer.Option(
        help=(
            "The names of the UVPSpec objects within the group to be merged. "
            "These should be the same for all files. Multiple names can be provided (via "
            "multiple --names flags), and they will be merged into the same file."
        )
    ),
    outpath: Path = typer.Option(
        help=(
            "The basename of the output file. This can be a full path, but note that "
            "the final output pspec file will have an extension of '.pspec.h5' added to it."
            "An --extras specified will be written to separate files with the same basename"
            "but a suffix of '.{extraname}.pkl'."
        )
    ),
    progress: bool = typer.Option(
        default=True,
        help=(
            "Whether to show a progress bar while loading the files. This is useful "
            "for large datasets, but can be turned off for small datasets."
        ),
    ),
    extras: list[str] | None = typer.Option(
        default=None,
        help=(
            "A list of extra attributes to be saved from the header of the files. "
            "These will be saved to separate files with the same basename as the output "
            "file, but with a suffix of '.{extraname}.pkl'. This is useful for saving "
            "metadata that is not stored in the UVPSpec objects themselves."
        ),
    ),
    batch_size: int | None = typer.Option(
        default=None,
        help=(
            "Number of files to load and merge at a time. Smaller batch sizes use less "
            "memory but may be slightly slower. If None (default), all files are loaded "
            "at once. Adjust this based on available RAM and file sizes."
        ),
    ),
) -> None:
    """Merge a set of hera_pspec files each representing a single baseline, into one.

    This can be useful because reading a single file with many baselines is much much
    faster than reading many files each with a single baseline currently.
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


@app.command()
def run(
    dsets: list[Path] = typer.Argument(
        ...,
        help="Input UVData files (miriad/uvh5 paths) to estimate power spectra from.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output filename of the HDF5 PSpecContainer."
    ),
    dset_std: list[Path] | None = typer.Option(
        None, "--dset-std", help="Visibility-stddev files, one --dset-std per dataset."
    ),
    groupname: str | None = typer.Option(
        None, help="Groupname for the UVPSpec objects in the HDF5 container."
    ),
    # typer 0.25.1 rejects `list[tuple[int, int]]` annotations, so we annotate the
    # repeatable tuple options as list[str] and supply the real element types via
    # click_type=click.Tuple([...]).
    dset_pair: list[str] | None = typer.Option(
        None,
        "--dset-pair",
        click_type=click.Tuple([int, int]),
        help="Dataset pairing for OQE, e.g. --dset-pair 0 1 (repeatable).",
    ),
    dset_label: list[str] | None = typer.Option(
        None, "--dset-label", help="String label for each input dataset (repeatable)."
    ),
    spw_range: list[str] | None = typer.Option(
        None,
        "--spw-range",
        click_type=click.Tuple([int, int]),
        help="Spectral-window channel selection, e.g. --spw-range 200 300 (repeatable).",
    ),
    n_dlys: list[int] | None = typer.Option(
        None, "--n-dlys", help="Number of delays per spectral window (repeatable)."
    ),
    pol_pair: list[str] | None = typer.Option(
        None,
        "--pol-pair",
        click_type=click.Tuple([str, str]),
        help="Polarization-string pair, e.g. --pol-pair xx xx (repeatable).",
    ),
    blpair: list[str] | None = typer.Option(
        None,
        "--blpair",
        click_type=click.Tuple([int, int, int, int]),
        help="Baseline-pair antennas, e.g. --blpair 1 2 3 4 -> ((1,2),(3,4)) (repeatable).",
    ),
    input_data_weight: str = typer.Option(
        "identity", help="Data weighting for OQE. See PSpecData.pspec."
    ),
    norm: str = typer.Option("I", help="M-matrix normalization type for OQE."),
    taper: str = typer.Option(
        "none", help="Taper function for the OQE delay transform."
    ),
    beam: Path | None = typer.Option(
        None, help="Filepath to a UVBeam healpix beam map."
    ),
    cosmo: list[float] | None = typer.Option(
        None,
        help="Cosmology [Om_L, Om_b, Om_c, H0, Om_M, Om_k] (repeatable, 6 values).",
    ),
    rephase_to_dset: int | None = typer.Option(
        None, help="Dataset index to phase all other dsets to. Default: no rephasing."
    ),
    trim_dset_lsts: bool = typer.Option(
        False, help="Trim non-overlapping dataset LSTs."
    ),
    broadcast_dset_flags: bool = typer.Option(
        False, help="Broadcast dataset flags across time per time_thresh."
    ),
    time_thresh: float = typer.Option(
        0.2, help="Fractional time-flagging threshold to trigger flag broadcast."
    ),
    jy2mk: bool = typer.Option(
        False,
        "--Jy2mK/--no-Jy2mK",
        help="Convert datasets from Jy to mK if a beam is given.",
    ),
    exclude_auto_bls: bool = typer.Option(
        False, help="If blpairs not given, exclude baselines paired with themselves."
    ),
    exclude_cross_bls: bool = typer.Option(
        False,
        help="If blpairs not given, exclude baselines paired with a different baseline.",
    ),
    exclude_permutations: bool = typer.Option(
        False, help="If blpairs not given, exclude baseline-pair permutations."
    ),
    nblps_per_group: int | None = typer.Option(
        None,
        "--nblps-per-group",
        help="If blpairs not given and grouping, blpairs per group.",
    ),
    bl_len_range: tuple[float, float] = typer.Option(
        (0.0, 1e10), help="If blpairs not given, min/max baseline length in meters."
    ),
    bl_deg_range: tuple[float, float] = typer.Option(
        (0.0, 180.0), help="If blpairs not given, min/max baseline angle (ENU degrees)."
    ),
    bl_error_tol: float = typer.Option(
        1.0, help="If blpairs not given, redundant-group error tolerance in meters."
    ),
    store_cov: bool = typer.Option(
        False, help="Compute and store bandpower covariance."
    ),
    store_cov_diag: bool = typer.Option(
        False, help="Compute and store QE-formalism error bars."
    ),
    return_q: bool = typer.Option(False, help="Return unnormalized bandpowers."),
    overwrite: bool = typer.Option(False, help="Overwrite output if it exists."),
    cov_model: str = typer.Option(
        "empirical", help="Covariance model: 'empirical' or 'dsets'."
    ),
    psname_ext: str = typer.Option(
        "", help="Extension for pspectra name in the container."
    ),
    verbose: bool = typer.Option(False, help="Report feedback to stdout."),
    file_type: str = typer.Option(
        "uvh5", help="Filetype of input UVData. Default 'uvh5'."
    ),
    filter_extension: list[str] | None = typer.Option(
        None,
        "--filter-extension",
        click_type=click.Tuple([int, int]),
        help="Per-spw filter extension, e.g. --filter-extension 20 20 (repeatable).",
    ),
    symmetric_taper: bool = typer.Option(
        True,
        "--symmetric-taper/--no-symmetric-taper",
        help="Apply sqrt(taper) before and after filtering (True) vs full taper after (False).",
    ),
    include_autocorrs: bool = typer.Option(
        False,
        "--include-autocorrs/--no-include-autocorrs",
        help="Include power spectra of autocorr visibilities.",
    ),
    include_crosscorrs: bool = typer.Option(
        True,
        "--include-crosscorrs/--no-include-crosscorrs",
        help="Include cross-correlations in power spectra.",
    ),
    interleave_times: bool = typer.Option(
        False, help="Cross-multiply even/odd time intervals."
    ),
    xant_flag_thresh: float = typer.Option(
        0.95,
        help="Flagged-fraction of a baseline waterfall to exclude the whole baseline.",
    ),
    store_window: bool = typer.Option(False, help="Store the window-function array."),
    allow_fft: bool = typer.Option(False, help="Use an FFT to compute q-hat."),
) -> None:
    """Run OQE power-spectrum estimation over datasets (was pspec_run.py)."""
    blpairs = (
        [((a, b), (c, d)) for (a, b, c, d) in blpair] if blpair is not None else None
    )
    history = " ".join(sys.argv)
    pspecdata.pspec_run(
        dsets=[str(d) for d in dsets],
        filename=str(output),
        dsets_std=[str(d) for d in dset_std] if dset_std is not None else None,
        groupname=groupname,
        dset_pairs=dset_pair,
        dset_labels=dset_label,
        spw_ranges=spw_range,
        n_dlys=n_dlys,
        pol_pairs=pol_pair,
        blpairs=blpairs,
        input_data_weight=input_data_weight,
        norm=norm,
        taper=taper,
        beam=str(beam) if beam is not None else None,
        cosmo=cosmo,
        rephase_to_dset=rephase_to_dset,
        trim_dset_lsts=trim_dset_lsts,
        broadcast_dset_flags=broadcast_dset_flags,
        time_thresh=time_thresh,
        Jy2mK=jy2mk,
        exclude_auto_bls=exclude_auto_bls,
        exclude_cross_bls=exclude_cross_bls,
        exclude_permutations=exclude_permutations,
        Nblps_per_group=nblps_per_group,
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
        filter_extensions=filter_extension,
        symmetric_taper=symmetric_taper,
        include_autocorrs=include_autocorrs,
        include_crosscorrs=include_crosscorrs,
        interleave_times=interleave_times,
        xant_flag_thresh=xant_flag_thresh,
        store_window=store_window,
        allow_fft=allow_fft,
        history=history,
    )


@app.command()
def bootstrap(
    filename: Path = typer.Argument(
        ..., help="HDF5 PSpecContainer with the input power spectra."
    ),
    spectra: list[str] | None = typer.Option(
        None,
        help="Power-spectrum names (with group prefix) to bootstrap over (repeatable).",
    ),
    blpair_group: list[str] | None = typer.Option(
        None,
        "--blpair-group",
        help="A baseline-pair group as space-separated blpair integers, e.g. "
        "--blpair-group '101 102' (repeatable). Default: solve for redundant groups.",
    ),
    time_avg: bool = typer.Option(
        False, help="Perform a time-average in the averaging step."
    ),
    nsamples: int = typer.Option(
        100, "--nsamples", help="Number of bootstrap resamples."
    ),
    seed: int = typer.Option(0, help="Random seed for bootstrap resampling."),
    normal_std: bool = typer.Option(True, help="Calculate a 'normal' std (np.std)."),
    robust_std: bool = typer.Option(
        False, help="Calculate a 'robust' std (biweight_midvariance)."
    ),
    cintervals: list[float] | None = typer.Option(
        None, help="Confidence intervals (0<ci<100) to calculate (repeatable)."
    ),
    keep_samples: bool = typer.Option(
        False, help="Store bootstrap resamples with a *_bs# extension."
    ),
    bl_error_tol: float = typer.Option(
        1.0, help="Baseline-redundancy tolerance when computing redundant groups."
    ),
    overwrite: bool = typer.Option(False, help="Overwrite outputs if they exist."),
    add_to_history: str = typer.Option(
        "", help="String to add to the power-spectra history."
    ),
    verbose: bool = typer.Option(False, help="Report feedback to stdout."),
) -> None:
    """Bootstrap over redundant baseline-pair groups (was bootstrap_run.py)."""
    blpair_groups = (
        [[int(tok) for tok in grp.split()] for grp in blpair_group]
        if blpair_group is not None
        else None
    )
    grouping.bootstrap_run(
        str(filename),
        spectra=spectra,
        blpair_groups=blpair_groups,
        time_avg=time_avg,
        Nsamples=nsamples,
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

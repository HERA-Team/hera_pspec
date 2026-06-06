"""A CLI interface for hera_pspec."""

import copy
import glob
import pickle
import sys
from pathlib import Path
from typing import Annotated, Literal

import h5py
from cyclopts import App, Parameter
from rich.console import Console
from tqdm import tqdm

cns = Console()

app = App(name="pspec", version_flags=[], help_flags=["--help"])
# cyclopts pattern: register subcommands after app is constructed
import pyuvdata  # noqa: E402
from pyuvdata import UVData  # noqa: E402

from . import container, grouping, pspecdata, pstokes, utils  # noqa: E402
from .uvpspec import recursive_combine_uvpspec  # noqa: E402


@app.command
def hello() -> None:
    # A trivial command, kept so the app stays a multi-command app.
    cns.print("Hi! :wave:")


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


@app.command
def run(
    dsets: list[Path],
    /,
    *,
    output: Annotated[Path, Parameter(name=["--output", "-o"])],
    dset_std: list[Path] | None = None,
    groupname: str | None = None,
    dset_pair: list[tuple[int, int]] | None = None,
    dset_label: list[str] | None = None,
    spw_range: list[tuple[int, int]] | None = None,
    n_dlys: list[int] | None = None,
    pol_pair: list[tuple[str, str]] | None = None,
    blpair: list[tuple[int, int, int, int]] | None = None,
    input_data_weight: Literal["identity", "iC", "dayenu"] = "identity",
    norm: Literal["I", "H^-1", "V^-1/2"] = "I",
    taper: str = "none",
    beam: Path | None = None,
    cosmo: list[float] | None = None,
    rephase_to_dset: int | None = None,
    trim_dset_lsts: bool = False,
    broadcast_dset_flags: bool = False,
    time_thresh: float = 0.2,
    jy2mk: Annotated[bool, Parameter(name="--Jy2mK", negative="--no-Jy2mK")] = False,
    exclude_auto_bls: bool = False,
    exclude_cross_bls: bool = False,
    exclude_permutations: bool = False,
    nblps_per_group: int | None = None,
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
    filter_extension: list[tuple[int, int]] | None = None,
    symmetric_taper: bool = True,
    include_autocorrs: bool = False,
    include_crosscorrs: bool = True,
    interleave_times: bool = False,
    xant_flag_thresh: float = 0.95,
    store_window: bool = False,
    allow_fft: bool = False,
) -> None:
    """Run OQE power-spectrum estimation over datasets (was pspec_run.py).

    Parameters
    ----------
    dsets
        Input UVData files (miriad/uvh5 paths) to estimate power spectra from.
    output
        Output filename of the HDF5 PSpecContainer.
    dset_std
        Visibility-stddev files, one --dset-std per dataset.
    groupname
        Groupname for the UVPSpec objects in the HDF5 container.
    dset_pair
        Dataset pairing for OQE, e.g. --dset-pair 0 1 (repeatable).
    dset_label
        String label for each input dataset (repeatable).
    spw_range
        Spectral-window channel selection, e.g. --spw-range 200 300 (repeatable).
    n_dlys
        Number of delays per spectral window (repeatable).
    pol_pair
        Polarization-string pair, e.g. --pol-pair xx xx (repeatable).
    blpair
        Baseline-pair antennas, e.g. --blpair 1 2 3 4 -> ((1,2),(3,4)) (repeatable).
    input_data_weight
        Data weighting for OQE. See PSpecData.pspec.
    norm
        M-matrix normalization type for OQE.
    taper
        Taper function for the OQE delay transform.
    beam
        Filepath to a UVBeam healpix beam map.
    cosmo
        Cosmology [Om_L, Om_b, Om_c, H0, Om_M, Om_k] (repeatable, 6 values).
    rephase_to_dset
        Dataset index to phase all other dsets to. Default: no rephasing.
    trim_dset_lsts
        Trim non-overlapping dataset LSTs.
    broadcast_dset_flags
        Broadcast dataset flags across time per time_thresh.
    time_thresh
        Fractional time-flagging threshold to trigger flag broadcast.
    jy2mk
        Convert datasets from Jy to mK if a beam is given.
    exclude_auto_bls
        If blpairs not given, exclude baselines paired with themselves.
    exclude_cross_bls
        If blpairs not given, exclude baselines paired with a different baseline.
    exclude_permutations
        If blpairs not given, exclude baseline-pair permutations.
    nblps_per_group
        If blpairs not given and grouping, blpairs per group.
    bl_len_range
        If blpairs not given, min/max baseline length in meters.
    bl_deg_range
        If blpairs not given, min/max baseline angle (ENU degrees).
    bl_error_tol
        If blpairs not given, redundant-group error tolerance in meters.
    store_cov
        Compute and store bandpower covariance.
    store_cov_diag
        Compute and store QE-formalism error bars.
    return_q
        Return unnormalized bandpowers.
    overwrite
        Overwrite output if it exists.
    cov_model
        Covariance model: 'empirical' or 'dsets'.
    psname_ext
        Extension for pspectra name in the container.
    verbose
        Report feedback to stdout.
    file_type
        Filetype of input UVData. Default 'uvh5'.
    filter_extension
        Per-spw filter extension, e.g. --filter-extension 20 20 (repeatable).
    symmetric_taper
        Apply sqrt(taper) before and after filtering (True) vs full taper after (False).
    include_autocorrs
        Include power spectra of autocorr visibilities.
    include_crosscorrs
        Include cross-correlations in power spectra.
    interleave_times
        Cross-multiply even/odd time intervals.
    xant_flag_thresh
        Flagged-fraction of a baseline waterfall to exclude the whole baseline.
    store_window
        Store the window-function array.
    allow_fft
        Use an FFT to compute q-hat.
    """
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


@app.command
def bootstrap(
    filename: Path,
    /,
    *,
    spectra: list[str] | None = None,
    blpair_group: list[str] | None = None,
    time_avg: bool = False,
    nsamples: int = 100,
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
    """Bootstrap over redundant baseline-pair groups (was bootstrap_run.py).

    Parameters
    ----------
    filename
        HDF5 PSpecContainer with the input power spectra.
    spectra
        Power-spectrum names (with group prefix) to bootstrap over (repeatable).
    blpair_group
        A baseline-pair group as space-separated blpair integers, e.g.
        --blpair-group '101 102' (repeatable). Default: solve for redundant groups.
    time_avg
        Perform a time-average in the averaging step.
    nsamples
        Number of bootstrap resamples.
    seed
        Random seed for bootstrap resampling.
    normal_std
        Calculate a 'normal' std (np.std).
    robust_std
        Calculate a 'robust' std (biweight_midvariance).
    cintervals
        Confidence intervals (0<ci<100) to calculate (repeatable).
    keep_samples
        Store bootstrap resamples with a *_bs# extension.
    bl_error_tol
        Baseline-redundancy tolerance when computing redundant groups.
    overwrite
        Overwrite outputs if they exist.
    add_to_history
        String to add to the power-spectra history.
    verbose
        Report feedback to stdout.
    """
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

"""A CLI interface for hera_pspec."""
import typer
from pathlib import Path
import warnings
from rich.console import Console
import h5py
import pickle
import glob
from tqdm import tqdm
from typing import Annotated

cns = Console()

app = typer.Typer()
from . import container
from .uvpspec import recursive_combine_uvpspec

@app.command()
def hello():
    # This is a test command which we need for the CLI interface to be broken into
    # subcommands (at least two commands need to be defined for it to be used as subc's)
    cns.print("Hi! :wave:")
    
_pattern_help=(
    
)
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
    progress: bool = typer.Option(default=True,
        help=(
            "Whether to show a progress bar while loading the files. This is useful "
            "for large datasets, but can be turned off for small datasets."
        )
    ),
    extras: list[str] = typer.Option(
        default=None,
        help=(
            "A list of extra attributes to be saved from the header of the files. "
            "These will be saved to separate files with the same basename as the output "
            "file, but with a suffix of '.{extraname}.pkl'. This is useful for saving "
            "metadata that is not stored in the UVPSpec objects themselves."
        )
    ),
    batch_size: int = typer.Option(
        default=None,
        help=(
            "Number of files to load and merge at a time. Smaller batch sizes use less "
            "memory but may be slightly slower. If None (default), all files are loaded "
            "at once. Adjust this based on available RAM and file sizes."
        )
    ),
):
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
            cns.print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)")

        # Load UVPSpec objects for this batch
        uvps_batch = {name: [] for name in names}

        for df in tqdm(batch_files, desc=f"Loading batch {batch_idx + 1}/{num_batches}", unit="file", disable=not progress):
            # load power spectra
            psc = container.PSpecContainer(df, mode='r', keep_open=False)

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
                with h5py.File(df, 'r') as f:
                    for extra in extras:
                        extra_attrs[extra][blp] = f['header'].attrs[extra]

        # Merge the batch
        for name, uvplist in uvps_batch.items():
            batch_merged = recursive_combine_uvpspec(uvplist)

            # Combine with previous batches
            if merged_uvps[name] is None:
                merged_uvps[name] = batch_merged
            else:
                merged_uvps[name] = recursive_combine_uvpspec([merged_uvps[name], batch_merged])

        # Clear batch data to free memory
        del uvps_batch

    cns.print("Writing merged power spectra to file")
    outspec = outpath.parent / f"{outpath.name}.pspec.h5"
    psc = container.PSpecContainer(outspec, mode='rw', keep_open=False)
    for name, uvp in merged_uvps.items():
        psc.set_pspec(group, name, uvp, overwrite=True)

    cns.print(f"Wrote pspecs to file: {outspec}")
    for name, extra in extra_attrs.items():
        fname = outpath.parent / f"{outpath.name}.{name}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(extra, f)
        cns.print(f"Wrote {fname}")
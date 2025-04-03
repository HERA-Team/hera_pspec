"""A CLI interface for hera_pspec."""
import typer
from pathlib import Path
import warnings
from rich.console import Console
import h5py
import pickle
import glob
from tqdm import tqdm
cns = Console()

app = typer.Typer()
from . import container
from .uvpspec import recursive_combine_uvpspec

@app.command()
def hello():
    # This is a test command which we need for the CLI interface to be broken into
    # subcommands (at least two commands need to be defined for it to be used as subc's)
    cns.print("Hi! :wave:")
    
@app.command()
def fast_merge_baselines(
    pattern: str = typer.Option(),
    group: str = typer.Option(),
    names: list[str] = typer.Option(),
    outpath: Path = typer.Option(),
    progress: bool = True,
    extras: list[str] = None,
):
    """Merge a set of hera_pspec files each representing a single baseline, into one.
    
    This can be useful because reading a single file with many baselines is much much 
    faster than reading many files each with a single baseline currently.
    """
    uvps = {name: [] for name in names}
    if extras is None:
        extras = []
    extra_attrs = {extra: {} for extra in extras}
    
    files = sorted(glob.glob(pattern))
    cns.print(f"Found {len(files)} files matching pattern.")

    for df in tqdm(files, desc="Loading files", unit="file", disable=not progress):
        # load power spectra
        psc = container.PSpecContainer(df, mode='r', keep_open=False)

        # Load both the time-averaged and not-time-averaged power spectra.
        # The time-averaging done in the single-baseline notebook has more
        # accurate noise calculations that can only be done when the interleaves
        # are separate.
        for name in names:
            uvp = psc.get_pspec(group, name)
            blp = uvp.get_blpairs()[0]
            uvps[name].append(uvp)

        if extras:
            # load additional metadata stored in header
            with h5py.File(df, 'r') as f:
                for extra in extras:
                    extra_attrs[extra][blp] = f['header'].attrs[extra]

    cns.print("Merging power spectra")
    outspec = outpath.parent / f"{outpath.name}.pspec.h5"
    psc = container.PSpecContainer(outspec, mode='rw', keep_open=False)
    for name, uvplist in uvps.items():
        uvp = recursive_combine_uvpspec(uvplist)
        psc.set_pspec(group, name, uvp, overwrite=True)

    cns.print(f"Wrote pspecs to file: {outspec}")
    for name, extra in extra_attrs.items():
        fname = outpath.parent / f"{outpath.name}.{name}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(extra, f)
        cns.print(f"Wrote {fname}")
"""A CLI interface for hera_pspec."""
import typer
from pathlib import Path
import warnings
from rich.console import Console
import h5py
import pickle
import glob
cns = Console()

app = typer.Typer()
from . import container
from .uvpspec import recursive_combine_uvpspec

@app.command()
def hello():
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
    """Merge a set of hera_pspec files each representing a single baseline, into one."""
    uvps = {name: [] for name in names}
    if extras is None:
        extras = []
    extra_attrs = {extra: {} for extra in extras}
    
    if progress:
        try:
            from tqdm import tqdm
        except ImportError:
            progress = False
            warnings.warn("tqdm not found, progress bar will not be shown.")
    else:
        tqdm = lambda x: x
        
    files = sorted(glob.glob(pattern))
    cns.print(f"Found {len(files)} files matching pattern.")
    
    for df in tqdm(files, desc="Loading files", unit="file"):
        # load power spectra
        psc = container.PSpecContainer(df, mode='r', keep_open=False)
        
        # Load both the time-averaged and not-time-averaged power spectra.
        # The time-averaging done in the single-baseline notebook has more
        # accurate noise calculations that can only be done when the interleaves
        # are separate.
        for name in names:
            uvp = psc.get_pspec(group, name)
            blp = uvp.get_blpairs()[0]

            if name not in uvps:
                uvps[name] = []
            uvps[name].append(uvp)
        
        if extras:
            # load additional metadata stored in header
            with h5py.File(df, 'r') as f:
                for extra in extras:
                    extra_attrs[extra][blp] = f['header'].attrs[extra]

    cns.print("Merging power spectra")        
    for name, uvplist in uvps.items():
        uvp = recursive_combine_uvpspec(uvplist)
        psc = container.PSpecContainer(outpath.with_suffix(".pspec.h5"), mode='rw', keep_open=False)
        psc.set_pspec(group, name, uvp, overwrite=True)
        
    for name, extra in extras.items():
        with open(outpath.with_suffix(f".{name}.pkl"), 'wb') as f:
            pickle.dump(extra, f)

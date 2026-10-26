# AGENTS.md

Guidance for AI coding assistants (Claude Code, Codex, Cursor, …) working in this repository. Human contributors may also find it useful as a quick reference.

## Repository

`hera_pspec` is the HERA collaboration's delay-spectrum / power-spectrum estimation library. Input visibilities come in via `pyuvdata`; output power spectra are persisted as HDF5. The package is in a `src/` layout (`src/hera_pspec/`); `setuptools_scm` writes `src/hera_pspec/_version.py` from git tags.

Python support: `>=3.11,<3.14`. CI tests on 3.11/3.12/3.13 across Ubuntu and macOS.

## Environment & common commands

The project uses **uv** for environment and lockfile management (matching CI). Prefer `uv` over `pip` here.

- Install for development: `uv sync --all-extras --dev`
- Run the test suite the way CI does: `MPLBACKEND=agg uv run pytest`
  - `pytest` config (`pyproject.toml`) auto-adds `--cov hera_pspec --cov-config=.coveragerc --cov-report xml:./coverage.xml --verbose` and points `testpaths = tests`.
- Run a single test file / test: `uv run pytest tests/test_pspecdata.py` or `uv run pytest tests/test_pspecdata.py::test_name`.
  - To bypass coverage while iterating: `uv run pytest --no-cov tests/test_pspecdata.py::test_name`.
- Warnings job (matches `warnings-tests.yml`, fails on any warning): `MPLBACKEND=agg uv run pytest -Werror`. New code must not emit warnings under this run.
- Build docs locally: `uv run --group docs sphinx-build docs docs/_build/html` (RTD config at `.readthedocs.yaml`). Requires `pandoc` on your `PATH` (e.g. `brew install pandoc`) for `nbsphinx` to convert the example notebooks; the Read the Docs image already ships it.
- CLI entrypoint installed as `pspec` (defined at `src/hera_pspec/cli.py`, `app = cyclopts.App()`).

## Linting

Ruff is configured in `pyproject.toml` with a deliberately small ruleset (E + W + F with empirical ignores; see `[tool.ruff.lint]` for rationale per ignored rule). Run scoped to the file you're editing whenever possible:

```bash
uv run --with ruff ruff check src/hera_pspec/pspecdata.py   # one file
uv run --with ruff ruff check                               # whole repo
uv run --with ruff ruff check --fix                         # safe autofixes
```

The ruleset will tighten over time in follow-up PRs (one rule per PR). Don't pre-emptively add new ignores or expand the `select` list without coordination.

## Architecture

The library is organized around three persistent object types and a computation engine. Understanding how they interact is the fastest route to being productive.

### Core objects

- **`PSpecData`** (`pspecdata.py`, ~6k lines) — the estimator. Holds a list of `pyuvdata.UVData` datasets (`dsets`), matching weight/std datasets, an optional `PSpecBeam`, and `UVCal` calibration. The `pspec()` method is the main entry point: it iterates baseline pairs, polarization pairs, and spectral windows and computes power spectra into a `UVPSpec`, normalized either `"two-step"` (the default: `G`/`H` matrices, then the beam scalar) or `"one-step"` (a response matrix in cosmological units, see `response.py`). The cosmological length unit is the `length_unit` argument (`u.Mpc / cu.littleh` by default, or `u.Mpc`). State on the instance (`spw_range`, `spw_Ndlys`, `data_weighting`, `taper`, `r_params`, `filter_extension`, `cov_regularization`, `symmetric_taper`) controls the estimator and is mutated by helper methods rather than passed through every call — be careful when refactoring not to break that implicit contract. `pspec_run()` is the high-level batch driver behind the `pspec run` CLI command (`src/hera_pspec/cli.py`).

- **`UVPSpec`** (`uvpspec.py`, ~4.6k lines) — the output container for a set of power spectra plus metadata. Attributes are declared as `PSpecParam` descriptors (`parameter.py`), each carrying a `description`, `expected_type`, array `form` and (for physical quantities) a `unit` — this is what enables `check()` validation and HDF5 round-tripping. Heavy operations (averaging, redundancy grouping, delay binning, fold, exact window functions) live partly here and partly in `grouping.py` and `uvpspec_utils.py` (imported as `uvputils`). Note the comment near the top of `UVPSpec.__init__`: `Ntimes` and `Ntpairs` are now identical aliases (post-v0.5); pre-v0.5 files used a different convention.

- **`PSpecContainer`** (`container.py`) — HDF5-backed dict-of-`UVPSpec`s organized as `group/name`. The `@transactional` decorator opens/closes the file per call when `keep_open=False`; nested transactional calls pass `nested=True` to keep the outer caller's lifecycle intact. Don't break that pattern when adding methods.

### Beams, windows, and supporting modules

- `pspecbeam.py` — `PSpecBeamUV` (from `.beamfits`), `PSpecBeamGauss`, `PSpecBeamFromArray`. Provides scalar normalizations and unit conversions used by `PSpecData.pspec()`.
- `uvwindow.py` — `UVWindow` and `FTBeam` for exact window-function calculations attached to `UVPSpec`s.
- `response.py` — `DelayResponse` and helpers for the response matrices `dC/dp` (#328) used by `PSpecData.get_response` and one-step normalization (#329).
- `conversions.py` — `DEFAULT_COSMOLOGY` and the 21 cm helpers (`f2z`, `z2f`, `X2Y`, `bl_to_kperp`, `tau_to_kpara`, …) that take an astropy cosmology and a length unit and return Quantities; also the deprecated `Cosmo_Conversions`.
- `grouping.py` — averaging, bootstrap, redundant-baseline grouping, time/delay binning that operates on `UVPSpec`.
- `noise.py`, `loss.py`, `pstokes.py`, `plot.py`, `utils.py`, `uvpspec_utils.py` — supporting math, pseudo-Stokes formation, plotting helpers, and shared utilities (`construct_blpairs` etc.).
- `testing.py` — `build_vanilla_uvpspec()` is the canonical synthetic `UVPSpec` factory; tests in `tests/conftest.py` expose it via session-scoped fixtures (`vanilla_uvp`, `vanilla_uvp_with_beam`, `uvp_example_data`, …). Reuse these instead of constructing `UVPSpec`s ad hoc.

### Data files

`src/hera_pspec/data/` is shipped with the wheel (`tool.setuptools.package-data`). It holds beamfits files and small `.uvh5` / `.uvA` visibility cutouts that the test suite and example notebooks load by absolute path via `from hera_pspec.data import DATA_PATH`. Don't move or rename these without updating both tests and notebooks under `docs/examples/`.

### Scripts vs CLI

There are two parallel entry-point styles:

- The `pspec` cyclopts app (`src/hera_pspec/cli.py`, `app = cyclopts.App()`) — currently exposes `fast-merge-baselines`, `run`, `bootstrap`, `auto-noise`, `generate-pstokes` and `compute-ft-beam`. Add new subcommands here.
- The historical `scripts/*.py` — formerly installed via `script-files` and built around `argparse` parsers returned from inside the package. None are installed now. The `pspec_run.py`, `bootstrap_run.py`, `auto_noise_run.py`, and `generate_pstokes_run.py` scripts have been **removed** and replaced by the corresponding `pspec` subcommands (`run`, `bootstrap`, `auto-noise`, `generate-pstokes`) — a breaking change to their CLI syntax (profiling via `hera_cal._cli_tools` was dropped pending typer support in hera-cli-utils). `scripts/pspec_red.py`, which had been broken since `pspec()` changed in 2018, was removed as well; `scripts/psc_merge_spectra.py` remains on disk but is not installed. New commands should be added under `cli.py` rather than as fresh scripts.

## Conventions to respect

- Imports go through the package's `__init__.py` (e.g. `from hera_pspec import UVPSpec, PSpecData, utils, grouping`); follow that re-export pattern when adding public surface.
- `UVPSpec` attributes must be declared as `PSpecParam` for validation/IO to work — adding a plain attribute will silently break `check()` and HDF5 read/write.
- Tests live under `tests/` (not inside the package). New `UVPSpec`-based fixtures should be added to `tests/conftest.py` and built via `testing.build_vanilla_uvpspec` when possible.
- The warnings-tests job is enforcing — when adding code that calls into `pyuvdata`, `numpy`, or `scipy`, suppress or fix any new warnings before merging.

### Units and cosmology

- Quantities at the boundary only. Public functions accept `astropy.units.Quantity` inputs and getters return Quantities (`get_dlys`, `get_kparas`, `get_kperps`, `get_blpair_seps`, `get_spw_ranges`, `PSpecData.delays()`, `scalar()`, `UVPSpec.units`). Inside the estimator (`q_hat`, `get_G`, `get_H`, `get_MW`, covariances, anything in a per-baseline loop) keep plain floats in fixed units: Hz, s, m, sr, Mpc. Strip units once on the way in, attach them once on the way out. `pyuvdata` objects always get plain Hz.
- `UVPSpec` attributes are plain arrays in the unit declared on their `PSpecParam` (`unit=`); `check()` validates it and `UVPSpec.get_param_unit(name)` returns it. The exact window function k grids use `UVPSpec.window_k_unit()` instead.
- little h is `astropy.cosmology.units.littleh` (`import astropy.cosmology.units as cu`). Convert with `cu.with_H0(cosmo.H0)`, always passing the H0 of the cosmology in use (`with_H0()` alone falls back to Planck18). Never use `astropy.units.h`: it is the hour.
- Cosmological length units come from a `length_unit` argument (`u.Mpc / cu.littleh` default, or `u.Mpc`). The old `little_h=` argument is deprecated; resolve both through `conversions.resolve_length_unit`. Physical inputs given as plain numbers are deprecated; convert them with `utils.to_value_in(value, unit, name)`, which warns for plain numbers and points the warning at the caller. Internal code must never take a deprecated path (the `-Werror` job enforces this).
- Cosmologies are `astropy.cosmology` objects; the default is `conversions.DEFAULT_COSMOLOGY` (a LambdaCDM with H0 = 67.27, Om0 = 0.31353, Ode0 = 0.6844, Tcmb0 = 0; not Planck18). `Cosmo_Conversions` is deprecated: don't use it in library code. Use the module helpers in `conversions.py` and `as_astropy_cosmology` for user input.
- `norm_units` and `vis_units` are canonical astropy strings (`"Mpc3 / littleh3"`, `"mK"`). Parse them with `uvputils.parse_pspec_unit`. Never hand a legacy string such as `"h^-3 Mpc^3"` to `u.Unit`: astropy reads that `h` as hour. Test for little h with `uvputils.is_littleh`.
- HDF5 format version 2: `uvpspec_version = "2"`, a `unit` attribute on each dataset, and the cosmology as both the v0.x-readable `cosmo` dict string and astropy YAML in `cosmo_yaml`. Files without a version are read as v0.x. Keep v0.x able to read new files.
- `tests/test_regression_numerics.py` pins numerical outputs against `tests/data/regression/*.npz` (see `tests/data/README.md`). A failing case means the numbers moved: fix the code. Regenerating a fixture needs the maintainer's approval and a commit body that says what changed and why.
- `benchmarks/` (see `benchmarks/README.md`) times realistic workflows. A workflow more than 10% slower than the baseline blocks a change; confirm a slowdown by running the baseline commit and the new one back to back in separate worktrees before calling it a regression.
- Commit messages use **conventional commits** (`type: description`, e.g. `fix:`, `feat:`, `chore:`, `docs:`, `test:`, `refactor:`).

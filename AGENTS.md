# AGENTS.md

Guidance for AI coding assistants (Claude Code, Codex, Cursor, …) working in this repository. Human contributors may also find it useful as a quick reference.

## Repository

`hera_pspec` is the HERA collaboration's delay-spectrum / power-spectrum estimation library. Input visibilities come in via `pyuvdata`; output power spectra are persisted as HDF5. The package is in a `src/` layout (`src/hera_pspec/`); `setuptools_scm` writes `src/hera_pspec/_version.py` from git tags.

## Environment & common commands

The project uses **uv** for environment and lockfile management (matching CI). Prefer `uv` over `pip` here.

- Install for development: `uv sync --all-extras --dev`
- Run the test suite the way CI does: `MPLBACKEND=agg uv run pytest`
  - `pytest` config (`pyproject.toml`) auto-adds `--cov hera_pspec --cov-config=.coveragerc --cov-report xml:./coverage.xml --verbose` and points `testpaths = tests`.
- Run a single test file / test: `uv run pytest tests/test_pspecdata.py` or `uv run pytest tests/test_pspecdata.py::test_name`.
  - To bypass coverage while iterating: `uv run pytest --no-cov tests/test_pspecdata.py::test_name`.
- Warnings job (matches `warnings-tests.yml`, fails on any warning): `MPLBACKEND=agg uv run pytest -Werror`. New code must not emit warnings under this run.
- Build docs locally: `uv run --group docs sphinx-build docs docs/_build/html` (RTD config at `.readthedocs.yaml`).
- CLI entrypoint installed as `pspec` (defined at `src/hera_pspec/cli.py`)

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

- **`PSpecData`** (`pspecdata.py`, ~4.6k lines) — the estimator. Holds a list of `pyuvdata.UVData` datasets (`dsets`), matching weight/std datasets, an optional `PSpecBeam`, and `UVCal` calibration. The `pspec()` method is the main entry point: it iterates baseline pairs, polarization pairs, and spectral windows and computes power spectra into a `UVPSpec`. State on the instance (`spw_range`, `spw_Ndlys`, `data_weighting`, `taper`, `r_params`, `filter_extension`, `cov_regularization`, `symmetric_taper`) controls the estimator and is mutated by helper methods rather than passed through every call — be careful when refactoring not to break that implicit contract. `pspec_run()` is the high-level batch driver used by `scripts/pspec_run.py`.

- **`UVPSpec`** (`uvpspec.py`, ~3k lines) — the output container for a set of power spectra plus metadata. Attributes are declared as `PSpecParam` descriptors (`parameter.py`), each carrying a `description`, `expected_type`, and array `form` — this is what enables `check()` validation and HDF5 round-tripping. Heavy operations (averaging, redundancy grouping, delay binning, fold, exact window functions) live partly here and partly in `grouping.py` and `uvpspec_utils.py` (imported as `uvputils`). Note the comment near the top of `UVPSpec.__init__`: `Ntimes` and `Ntpairs` are now identical aliases (post-v0.5); pre-v0.5 files used a different convention.

- **`PSpecContainer`** (`container.py`) — HDF5-backed dict-of-`UVPSpec`s organized as `group/name`. The `@transactional` decorator opens/closes the file per call when `keep_open=False`; nested transactional calls pass `nested=True` to keep the outer caller's lifecycle intact. Don't break that pattern when adding methods.

### Beams, windows, and supporting modules

- `pspecbeam.py` — `PSpecBeamUV` (from `.beamfits`), `PSpecBeamGauss`, `PSpecBeamFromArray`. Provides scalar normalizations and unit conversions used by `PSpecData.pspec()`.
- `uvwindow.py` — `UVWindow` and `FTBeam` for exact window-function calculations attached to `UVPSpec`s.
- `grouping.py` — averaging, bootstrap, redundant-baseline grouping, time/delay binning that operates on `UVPSpec`.
- `noise.py`, `loss.py`, `pstokes.py`, `conversions.py`, `plot.py`, `utils.py`, `uvpspec_utils.py` — supporting math, pseudo-Stokes formation, plotting helpers, and shared utilities (`construct_blpairs` etc.).
- `testing.py` — `build_vanilla_uvpspec()` is the canonical synthetic `UVPSpec` factory; tests in `tests/conftest.py` expose it via session-scoped fixtures (`vanilla_uvp`, `vanilla_uvp_with_beam`, `uvp_example_data`, …). Reuse these instead of constructing `UVPSpec`s ad hoc.

### Data files

`src/hera_pspec/data/` is shipped with the wheel (`tool.setuptools.package-data`). It holds beamfits files and small `.uvh5` / `.uvA` visibility cutouts that the test suite and example notebooks load by absolute path via `from hera_pspec.data import DATA_PATH`. Don't move or rename these without updating both tests and notebooks under `examples/`.

## Conventions to respect

- Imports go through the package's `__init__.py` (e.g. `from hera_pspec import UVPSpec, PSpecData, utils, grouping`); follow that re-export pattern when adding public surface.
- `UVPSpec` attributes must be declared as `PSpecParam` for validation/IO to work — adding a plain attribute will silently break `check()` and HDF5 read/write.
- Tests live under `tests/` (not inside the package). New `UVPSpec`-based fixtures should be added to `tests/conftest.py` and built via `testing.build_vanilla_uvpspec` when possible.
- The warnings-tests job is enforcing — when adding code that calls into `pyuvdata`, `numpy`, or `scipy`, suppress or fix any new warnings before merging.
- Commit messages use **conventional commits** (`type: description`, e.g. `fix:`, `feat:`, `chore:`, `docs:`, `test:`, `refactor:`).

## Dos / Don'ts

- DON'T: reply to PR comments/issues without direct approval by a human.
- DO: when writing new features, write tests first. Prioritise *physically meaningful* tests, not just functionality tests / coverage tests.
- DON'T: write tests that simply re-write a library function and then test the output against the library function.
- DO: keep documentation in scope. Docstrings and tutorials are *user facing* and should not refer to e.g. PR's or Issues, and should not refer to
  'history' e.g. "This method avoids previous errors where...". Comments are developer-facing and may refer to details, but should not be over-long.
- DO: always consider refining the implementation to make it as easy to understand for humans as possible.
- DON'T: multiple internal methods starting with _ which are called less than 3 times and whose content is less than 3 lines, unless it allows a significant memory gain / speed-up (x1.2 improvement and above).
- DO: when modifying methods, modify the example notebooks accordingly.

### Writing tests

Tests are read by people checking that the code is right, so they should read top to bottom without opening other code.

- DO: parametrize only when the cases really are the same test with different inputs for a given function argument. Avoid parametrized test over the same function but for different function arguments. Keep a table to about three parameters of plain values (numbers, strings, units), and give every case a readable `id`.
- DO: add a one-sentence docstring to explain what the test is about. Ensure it is consistent with the test name.
- DO: when possible, include a numerical check to a (simple) expected output.
- DON'T: put callables, expected error-message fragments or whole objects into a parametrize table to merge tests that differ in what they set up or check. Write two or three short separate tests instead.
- DON'T: pass fixtures by name through `request.getfixturevalue` unless there is no simpler way.
- DON'T: hide a test's assertions in a helper. A reader should see what is being checked in the test body.
- DO: when merging or deleting tests, check that every assertion they made is still made somewhere, and say where in the PR description.

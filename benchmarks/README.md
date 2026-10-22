# Benchmarks

Timings for a few realistic `hera_pspec` workflows, used to check that a change
(for example the move to astropy units) does not slow the code down. These are
plain scripts with no extra dependencies. They are not part of the pytest run
or CI.

## Workflows

| name | what is timed |
| --- | --- |
| `pspec_full` | `PSpecData` + `Jy_to_mK` + `pspec` over all 152 redundant baseline pairs of the shipped LST cutout (even x even), 2 spws of 40 channels, Blackman-Harris taper, `PSpecBeamUV`, `store_cov=True` |
| `pspec_full_exact_norm` | the same with `exact_norm=True` |
| `pspec_full_one_step` | the same with `normalization="one-step"` (default `DelayResponse`, no scalar) |
| `pspec_full_ic` | the same with `input_data_weight="iC"` (two-step), which builds `G` and `H` for every baseline pair |
| `spherical_average` | `grouping.spherical_average` of the `pspec_full` output |
| `uvwindow_spherical_wf` | `UVWindow.get_spherical_wf` from the HERA dipole FT beam for 120 baseline lengths |
| `hdf5_write` | 200 x `UVPSpec.write_hdf5` of a 3-spw pspec output over the same 152 blpairs x 10 times (~6 MB; no cov or window functions, which would make this a disk benchmark) |
| `hdf5_read` | 100 x `UVPSpec.read_hdf5` of that file |

`uv run python benchmarks/run_benchmarks.py --list` prints the same list.

Inputs are built once, outside the timed region. Each workflow gets one
warm-up call, then `--repeat` timed calls. The JSON output records every
timing (min, median, mean, std), the input sizes, and the environment
(package versions, BLAS, git commit).

## Running

From the repository root, on an otherwise idle machine:

```bash
uv run python benchmarks/run_benchmarks.py --output ~/bench/new.json
uv run python benchmarks/run_benchmarks.py --only pspec_full spherical_average --repeat 10
```

A full run takes a couple of minutes. Write the JSON outside the repository.
The baseline JSON is not committed: it depends on the machine, so keep it
next to your other local files and regenerate it if you change machines or
package versions.

## Comparing

```bash
uv run python benchmarks/compare_benchmarks.py ~/bench/baseline.json ~/bench/new.json
```

This prints the median of each workflow in both runs and the relative change.
It exits with status 1 if any workflow is more than 10% slower than the
baseline (`--threshold 0.10`). A slowdown above 10% on any workflow is a
blocker for the change being tested.

Only compare runs from the same machine and environment; the script lists any
differences in package versions or platform.

## Noise

Within one run the spread of each workflow is small (std of 1-4% of the mean).
Between runs a few minutes apart on a laptop with other applications open,
the medians can shift by more than 10% together, because of other load and
CPU frequency changes. So a stored baseline is a first check only. If a
workflow fails the 10% rule, check out the baseline commit in a separate
worktree and run both versions back to back (baseline, new, baseline, new),
with nothing else running, before calling it a regression.

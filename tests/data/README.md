# Regression fixtures

`regression/*.npz` hold the expected outputs for
`tests/test_regression_numerics.py`. They were generated from unmodified
library code before the astropy-units migration, so they record what
hera_pspec computed at that point: `PSpecData.pspec()` across normalisations,
tapers, `little_h`, Jy -> mK, `exact_norm`, exact windows and stored
covariances; the beam scalars and `Jy_to_mK` for each beam class; `UVWindow`
cylindrical and spherical window functions; `grouping.spherical_average` and
`grouping.spherical_wf_from_uvp`; the noise helpers; and a read of the 2019
file `src/hera_pspec/data/test_uvp.h5`. The cases that use the default
cosmology (all but `legacy_test_uvp_h5` and `noise_calc_P_N`) were
regenerated when the default became `astropy.cosmology.Planck18`.

The inputs for every case are built in `tests/regression_cases.py`. The tests
and the regeneration script both import it, so a fixture is always produced by
the same code that checks it.

## Regenerating

A failing regression test means the numbers changed. Fix the code first. Only
regenerate when the change is intended, and only with explicit approval from
the maintainer. Say in the commit body which cases changed, by how much, and
why.

```bash
# write fixtures that do not exist yet (e.g. a newly added case)
uv run python tests/data/regenerate_regression_fixtures.py NEW_CASE

# overwrite existing fixtures (needs approval)
uv run python tests/data/regenerate_regression_fixtures.py --force CASE [CASE ...]
```

The script refuses to overwrite an existing fixture without `--force`.

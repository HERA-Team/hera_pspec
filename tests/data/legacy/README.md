# Legacy HDF5 fixtures

These files were written by hera_pspec at commit `f942d1a` (branch
`astropy-units-v1`), the last commit before the HDF5 writers gained a format
version and unit attributes. They pin what v0.x files look like, so the readers
can be checked against real old files. **Never regenerate them with newer
code**: a newer writer produces the new format, and the tests reading these
files would then no longer cover the legacy reader. If one is lost, check it
out from git history instead.

They were made with

```bash
uv run python tests/data/regenerate_legacy_fixtures.py
```

which calls the builders in `tests/legacy_fixtures.py` (the tests use the same
builders to rebuild the objects in memory and compare):

| file | made by | contents |
|---|---|---|
| `container_v1.h5` | `write_container` | `PSpecContainer` with group `legacy` holding two `UVPSpec`s: `exact_windows_cov` (`PSpecData.pspec()` on one integration of `zen.2458116.31939.HH.uvh5`, channels 175-185, `taper="bh"`, `exact_windows=True` with the `FT_beam_HERA_dipole_test` beam, `store_cov=True`, `cov_model="empirical"`) and `vanilla_plain` (`testing.build_vanilla_uvpspec(beam=<NF dipole beam>, Ndlys=10)` with the covariance and window function arrays removed) |
| `uvwindow_v1.h5` | `write_uvwindow` | `UVWindow.run_and_write` with `little_h=True`, Blackman-Harris taper, channels 5-15 of `FT_beam_HERA_dipole_test_xx.hdf5`, baseline lengths 14.6 and 29.2 m with weights 2 and 1, default k bins |
| `ftbeam_v1_xx.hdf5` | `write_ftbeam` | `FTBeam.write_hdf5` of `FT_beam_HERA_dipole_test_xx.hdf5` cut to channels 10-13 and the central 61 x 61 pixels |

The `PSpecContainer` header records the installed package version string at the
time (`0.5.1.dev207+g9dfd7f490`, a stale `setuptools_scm` value); the code that
wrote the files is commit `f942d1a`.

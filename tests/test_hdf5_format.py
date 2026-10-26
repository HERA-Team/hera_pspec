"""Tests for the versioned HDF5 format (units, cosmology) and legacy reads."""

import ast
import copy
import shutil
import warnings
from pathlib import Path

import astropy.cosmology.units as cu
import h5py
import numpy as np
import pytest
from astropy import units
from astropy.cosmology import FLRW, Cosmology, FlatLambdaCDM, LambdaCDM, wCDM

from hera_pspec import PSpecContainer, UVPSpec, conversions, unit_helpers, uvwindow
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec.data import DATA_PATH
from tests import legacy_fixtures as lf

DATA_PATH = Path(DATA_PATH)

# Cosmo_Conversions().get_params() of hera_pspec v0.x, i.e. the "cosmo"
# attribute of files written with the default cosmology.
OLD_DEFAULT_COSMO_PARAMS = {
    "Om_L": 0.6844,
    "Om_b": 0.04911,
    "Om_c": 0.26442,
    "Om_M": 0.31353,
    "Om_k": 0.0020700000000000163,
    "H0": 67.27,
}

# Parameters with a declared unit, by where write_to_group puts them
UNIT_DSETS = {
    "lst_1_array": "rad",
    "lst_2_array": "rad",
    "lst_avg_array": "rad",
    "time_1_array": "d",
    "time_2_array": "d",
    "time_avg_array": "d",
    "bl_vecs": "m",
    "OmegaP": "sr",
    "OmegaPP": "sr",
}
UNIT_ATTRS = {
    "freq_array": "Hz",
    "dly_array": "s",
    "channel_width": "Hz",
    "telescope_location": "m",
    "beam_freqs": "Hz",
}


@pytest.fixture(scope="module")
def uvp_windows() -> UVPSpec:
    """A UVPSpec with exact window functions and covariances."""
    return lf.build_uvp_exact_windows_cov()


def _write(uvp: UVPSpec, path: Path) -> Path:
    uvp.write_hdf5(path, overwrite=True)
    return path


def _read(path: Path, **kwargs) -> UVPSpec:
    uvp = UVPSpec()
    uvp.read_hdf5(path, **kwargs)
    return uvp


class TestWriteVersion2:
    def test_round_trip(self, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path) -> None:
        path = _write(vanilla_uvp_with_beam, tmp_path / "uvp.h5")
        assert _read(path) == vanilla_uvp_with_beam

    def test_version_and_unit_attrs(
        self, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        path = _write(vanilla_uvp_with_beam, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            assert f.attrs["uvpspec_version"] == "2"
            assert f.attrs["pspec_type"] == "UVPSpec"
            for name, unit in UNIT_DSETS.items():
                assert f[name].attrs["unit"] == unit, name
            for name, unit in UNIT_ATTRS.items():
                assert f.attrs[f"{name}_unit"] == unit, name
            assert f["integration_spw0"].attrs["unit"] == "s"
            # parameters without a declared unit get no unit attribute
            assert "unit" not in f["blpair_array"].attrs
            # power spectra carry the unit set by vis_units and norm_units
            assert f["data_spw0"].attrs["unit"] == "uncalib2 Hz sr"
            assert "scalar_array_unit" not in f.attrs
            assert "norm_units_unit" not in f.attrs

    def test_every_declared_unit_is_written(
        self, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        uvp = vanilla_uvp_with_beam
        path = _write(uvp, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            for name in uvp._meta:
                unit = uvp.get_param_unit(name)
                if unit is None or not hasattr(uvp, name):
                    continue
                if name in uvp._meta_dsets:
                    stored = f[name].attrs["unit"]
                else:
                    stored = f.attrs[f"{name}_unit"]
                assert unit_helpers.parse_unit_string(stored) == unit, name

    def test_window_function_k_units(
        self, uvp_windows: UVPSpec, tmp_path: Path
    ) -> None:
        path = _write(uvp_windows, tmp_path / "uvp.h5")
        assert uvp_windows.window_k_unit() == cu.littleh / units.Mpc
        with h5py.File(path, "r") as f:
            for name in ("window_function_kperp_spw0", "window_function_kpara_spw0"):
                assert f[name].attrs["unit"] == "littleh / Mpc"
        assert _read(path) == uvp_windows

    def test_window_function_k_units_inverse_mpc(
        self, uvp_exact_wfs: UVPSpec, tmp_path: Path
    ) -> None:
        # an "Hz sr" object: the window functions are on a 1 / Mpc grid
        uvp = uvp_exact_wfs
        assert uvp.window_k_unit() == units.Mpc ** (-1)
        path = _write(uvp, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            for spw in uvp.spw_array:
                for name in ("kperp", "kpara"):
                    dset = f[f"window_function_{name}_spw{spw}"]
                    assert dset.attrs["unit"] == "1 / Mpc"
                    np.testing.assert_array_equal(
                        dset[()], getattr(uvp, f"window_function_{name}")[spw]
                    )
        new = _read(path)
        assert new == uvp
        for spw in uvp.spw_array:
            for got, expected in zip(
                new.get_window_function_kgrid(spw),
                uvp.get_window_function_kgrid(spw),
                strict=True,
            ):
                assert got.unit == expected.unit
                np.testing.assert_array_equal(got.value, expected.value)

    def test_window_k_unit_with_littleh_on_inverse_mpc_raises(
        self, uvp_exact_wfs: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            f["window_function_kpara_spw0"].attrs["unit"] = "littleh / Mpc"

        src = _write(uvp_exact_wfs, tmp_path / "windows.h5")
        path = _edited_copy(src, tmp_path / "bad.h5", edit)
        with pytest.raises(ValueError, match="window_function_kpara"):
            _read(path)

    def test_cosmo_attrs(self, vanilla_uvp: UVPSpec, tmp_path: Path) -> None:
        path = _write(vanilla_uvp, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            cosmo_params = f.attrs["cosmo"]
            cosmo_yaml = f.attrs["cosmo_yaml"]
        # the dict string v0.x reads with ast.literal_eval is unchanged
        assert cosmo_params == str(OLD_DEFAULT_COSMO_PARAMS)
        assert ast.literal_eval(cosmo_params) == conversions.params_from_cosmology(
            vanilla_uvp.cosmo
        )
        cosmology = Cosmology.from_format(cosmo_yaml, format="yaml")
        assert cosmology == vanilla_uvp.cosmo
        assert cosmology == conversions.DEFAULT_COSMOLOGY
        read_cosmo = _read(path).cosmo
        assert isinstance(read_cosmo, FLRW)
        assert read_cosmo == conversions.DEFAULT_COSMOLOGY

    def test_quantity_attribute_written_plain(
        self, vanilla_uvp: UVPSpec, tmp_path: Path
    ) -> None:
        uvp = copy.deepcopy(vanilla_uvp)
        uvp.dly_array = (uvp.dly_array * units.s).to(units.ns)
        uvp.lst_1_array = (uvp.lst_1_array * units.rad).to(units.deg)
        uvp.integration_array = {
            spw: (arr * units.s).to(units.ms)
            for spw, arr in vanilla_uvp.integration_array.items()
        }
        path = _write(uvp, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            assert f.attrs["dly_array_unit"] == "s"
            np.testing.assert_allclose(
                f.attrs["dly_array"], vanilla_uvp.dly_array, rtol=1e-14
            )
            assert not isinstance(f.attrs["dly_array"], units.Quantity)
            np.testing.assert_allclose(
                f["lst_1_array"][()], vanilla_uvp.lst_1_array, rtol=1e-14
            )
            np.testing.assert_allclose(
                f["integration_spw0"][()], vanilla_uvp.integration_array[0], rtol=1e-14
            )
        uvp_read = _read(path)
        assert not isinstance(uvp_read.dly_array, units.Quantity)
        assert uvp_read == vanilla_uvp

    def test_inconsistent_om_k_cosmo_conversions_rejected(
        self, vanilla_uvp: UVPSpec, tmp_path: Path
    ) -> None:
        # astropy cannot represent an Om_k inconsistent with Om_L and Om_M
        uvp = copy.deepcopy(vanilla_uvp)
        with pytest.warns(DeprecationWarning):
            uvp.cosmo = conversions.Cosmo_Conversions(Om_k=0.05)
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(AssertionError, match="Om_k"),
        ):
            uvp.check()

    def test_cosmo_conversions_converted_on_write(
        self, vanilla_uvp: UVPSpec, tmp_path: Path
    ) -> None:
        uvp = copy.deepcopy(vanilla_uvp)
        with pytest.warns(DeprecationWarning):
            uvp.cosmo = conversions.Cosmo_Conversions(H0=70.0, Om_L=0.7)
        with pytest.warns(DeprecationWarning, match="astropy.cosmology"):
            path = _write(uvp, tmp_path / "uvp.h5")
        assert isinstance(uvp.cosmo, FLRW)
        assert _read(path).cosmo == uvp.cosmo

    @pytest.mark.parametrize(
        "cosmo",
        [
            conversions.DEFAULT_COSMOLOGY,
            # the cosmology of the 2019 file test_uvp.h5, built from its dict
            conversions.cosmology_from_params(
                {
                    "Om_L": 0.6844,
                    "Om_b": 0.04911,
                    "Om_c": 0.26442,
                    "Om_M": 0.31353,
                    "Om_k": 0.0020700000000000163,
                    "H0": 67.27,
                }
            ),
            LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7, Ob0=0.05, name="custom"),
            FlatLambdaCDM(H0=70.0, Om0=0.3),
            wCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05),
        ],
        ids=["default", "test_uvp_h5", "custom", "flat", "wcdm"],
    )
    def test_cosmo_yaml_round_trip(
        self, vanilla_uvp: UVPSpec, tmp_path: Path, cosmo: FLRW
    ) -> None:
        uvp = copy.deepcopy(vanilla_uvp)
        uvp.cosmo = cosmo
        path = _write(uvp, tmp_path / "uvp.h5")
        with h5py.File(path, "r") as f:
            assert "cosmo_yaml" in f.attrs
            assert ast.literal_eval(f.attrs["cosmo"]) == (
                conversions.params_from_cosmology(cosmo)
            )
        read_cosmo = _read(path).cosmo
        assert type(read_cosmo) is type(cosmo)
        assert read_cosmo == cosmo

    def test_cosmo_attrs_follow_cosmology_class(
        self, vanilla_uvp: UVPSpec, tmp_path: Path
    ) -> None:
        # same parameter dict, different cosmologies: the cached attributes
        # must not mix them up
        lcdm = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7, Ob0=0.05)
        wcdm = wCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05)
        for cosmo in (lcdm, wcdm, lcdm):
            uvp = copy.deepcopy(vanilla_uvp)
            uvp.cosmo = cosmo
            path = _write(uvp, tmp_path / "uvp.h5")
            assert _read(path).cosmo == cosmo


def _edited_copy(src: Path, dst: Path, edit) -> Path:
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        edit(f)
    return dst


class TestReadVersion2:
    @pytest.fixture()
    def v2_file(self, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path) -> Path:
        return _write(vanilla_uvp_with_beam, tmp_path / "v2.h5")

    def test_equivalent_attr_unit_converts(
        self, v2_file: Path, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            f.attrs["dly_array"] = f.attrs["dly_array"] * 1e9
            f.attrs["dly_array_unit"] = "ns"

        uvp = _read(_edited_copy(v2_file, tmp_path / "ns.h5", edit))
        np.testing.assert_allclose(
            uvp.dly_array, vanilla_uvp_with_beam.dly_array, rtol=1e-14
        )
        assert uvp == vanilla_uvp_with_beam

    def test_equivalent_dset_unit_converts(
        self, v2_file: Path, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            lst = f["lst_avg_array"][()]
            del f["lst_avg_array"]
            dset = f.create_dataset("lst_avg_array", data=np.rad2deg(lst))
            dset.attrs["unit"] = "deg"
            ints = f["integration_spw0"]
            ints[...] = ints[()] * 1e3
            ints.attrs["unit"] = "ms"

        uvp = _read(_edited_copy(v2_file, tmp_path / "deg.h5", edit))
        np.testing.assert_allclose(
            uvp.lst_avg_array, vanilla_uvp_with_beam.lst_avg_array, rtol=1e-14
        )
        np.testing.assert_allclose(
            uvp.integration_array[0],
            vanilla_uvp_with_beam.integration_array[0],
            rtol=1e-14,
        )

    def test_missing_unit_attr_assumes_declared(
        self, v2_file: Path, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            del f.attrs["freq_array_unit"]
            del f["bl_vecs"].attrs["unit"]

        uvp = _read(_edited_copy(v2_file, tmp_path / "nounit.h5", edit))
        assert uvp == vanilla_uvp_with_beam

    @pytest.mark.parametrize(
        "where,name,unit",
        [
            ("attr", "dly_array", "m"),
            ("attr", "freq_array", "s"),
            ("dset", "bl_vecs", "Hz"),
            ("dset", "integration_spw0", "m"),
        ],
    )
    def test_non_equivalent_unit_raises(
        self, v2_file: Path, tmp_path: Path, where: str, name: str, unit: str
    ) -> None:
        def edit(f):
            if where == "attr":
                f.attrs[f"{name}_unit"] = unit
            else:
                f[name].attrs["unit"] = unit

        path = _edited_copy(v2_file, tmp_path / "bad.h5", edit)
        param = name.replace("_spw0", "_array")
        with pytest.raises(ValueError, match=f"{param} is stored in unit '{unit}'"):
            _read(path)

    def test_unparsable_unit_raises(self, v2_file: Path, tmp_path: Path) -> None:
        def edit(f):
            f["time_1_array"].attrs["unit"] = "not_a_unit"

        path = _edited_copy(v2_file, tmp_path / "bad.h5", edit)
        with pytest.raises(ValueError, match="time_1_array .*not a valid unit"):
            _read(path)

    def test_window_k_unit_without_littleh_raises(
        self, uvp_windows: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            f["window_function_kperp_spw0"].attrs["unit"] = "1 / Mpc"

        src = _write(uvp_windows, tmp_path / "windows.h5")
        path = _edited_copy(src, tmp_path / "bad.h5", edit)
        with pytest.raises(ValueError, match="window_function_kperp"):
            _read(path)

    @pytest.mark.parametrize("version", ["3", "2.1", "1"])
    def test_unknown_version_raises(
        self, v2_file: Path, tmp_path: Path, version: str
    ) -> None:
        def edit(f):
            f.attrs["uvpspec_version"] = version

        path = _edited_copy(v2_file, tmp_path / "future.h5", edit)
        with pytest.raises(ValueError, match="format version"):
            _read(path)

    def test_cosmo_disagreement_raises(self, v2_file: Path, tmp_path: Path) -> None:
        def edit(f):
            params = ast.literal_eval(f.attrs["cosmo"])
            params["H0"] = 70.0
            f.attrs["cosmo"] = str(params)

        path = _edited_copy(v2_file, tmp_path / "cosmo.h5", edit)
        with pytest.raises(ValueError, match="disagree on H0"):
            _read(path)

    def test_cosmo_yaml_preferred(self, v2_file: Path, tmp_path: Path) -> None:
        other = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7, Ob0=0.05)
        other_params = conversions.params_from_cosmology(other)

        def edit(f):
            # a tiny difference, inside the tolerance, keeps the YAML value
            params = ast.literal_eval(f.attrs["cosmo"])
            params["H0"] = other_params["H0"] * (1 + 1e-14)
            for key in ("Om_L", "Om_b", "Om_c", "Om_M", "Om_k"):
                params[key] = other_params[key]
            f.attrs["cosmo"] = str(params)
            f.attrs["cosmo_yaml"] = other.to_format("yaml")

        uvp = _read(_edited_copy(v2_file, tmp_path / "cosmo.h5", edit))
        assert uvp.cosmo == other

    def test_cosmo_yaml_without_dict(
        self, v2_file: Path, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        def edit(f):
            del f.attrs["cosmo"]

        uvp = _read(_edited_copy(v2_file, tmp_path / "yaml_only.h5", edit))
        assert uvp.cosmo == vanilla_uvp_with_beam.cosmo

    def test_partial_read(self, uvp_windows: UVPSpec, tmp_path: Path) -> None:
        path = _write(uvp_windows, tmp_path / "windows.h5")
        blpairs = np.unique(uvp_windows.blpair_array)[:1]
        uvp = _read(path, blpairs=blpairs)
        assert uvp == uvp_windows.select(blpairs=blpairs, inplace=False)
        meta = _read(path, just_meta=True)
        assert not hasattr(meta, "data_array")
        np.testing.assert_array_equal(meta.dly_array, uvp_windows.dly_array)


class TestLegacyFiles:
    """Files written by hera_pspec v0.x (tests/data/legacy, commit f942d1a)."""

    def test_test_uvp_h5_has_no_version(self) -> None:
        with h5py.File(DATA_PATH / "test_uvp.h5", "r") as f:
            assert "uvpspec_version" not in f.attrs
        uvp = _read(DATA_PATH / "test_uvp.h5")
        assert isinstance(uvp.cosmo, FLRW)
        assert uvp.cosmo.is_equivalent(conversions.DEFAULT_COSMOLOGY)
        assert conversions.params_from_cosmology(uvp.cosmo) == OLD_DEFAULT_COSMO_PARAMS
        # repeated reads of the same legacy dict reuse one cosmology object
        assert _read(DATA_PATH / "test_uvp.h5").cosmo is uvp.cosmo

    @pytest.mark.parametrize(
        "name,builder",
        [
            (lf.UVP_WINDOWS_NAME, lf.build_uvp_exact_windows_cov),
            (lf.UVP_PLAIN_NAME, lf.build_uvp_plain),
        ],
    )
    def test_container(self, name: str, builder) -> None:
        with h5py.File(lf.CONTAINER_FILE, "r") as f:
            assert "uvpspec_version" not in f[lf.CONTAINER_GROUP][name].attrs
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
            uvp = psc.get_pspec(lf.CONTAINER_GROUP, name)
        expected = builder()
        # pspec() records the time of the run in history
        params = [p for p in expected._all_params if p != "history"]
        assert uvp.__eq__(expected, params=params)
        # the v0.x cosmology dict is read as an astropy cosmology
        assert isinstance(uvp.cosmo, FLRW)
        assert uvp.cosmo.is_equivalent(conversions.DEFAULT_COSMOLOGY)

    def test_container_rewritten_as_v2(self, tmp_path: Path) -> None:
        psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
        legacy = psc.get_pspec(lf.CONTAINER_GROUP, lf.UVP_WINDOWS_NAME)
        path = _write(legacy, tmp_path / "v2.h5")
        assert _read(path) == legacy

    def test_ftbeam(self) -> None:
        with h5py.File(lf.FTBEAM_FILE, "r") as f:
            assert "ftbeam_version" not in f.attrs
            assert "unit" not in f["freq"].attrs
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ftbeam = uvwindow.FTBeam.from_file(lf.FTBEAM_FILE)
        assert ftbeam == lf.build_ftbeam()

    def test_uvwindow(self, tmp_path: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lf.write_uvwindow(tmp_path / "uvwindow_v2.h5")
        with (
            h5py.File(lf.UVWINDOW_FILE, "r") as legacy,
            h5py.File(tmp_path / "uvwindow_v2.h5", "r") as new,
        ):
            assert "uvwindow_version" not in legacy.attrs
            for key in ("avg_nu", "avg_z", "nfreqs", "little_h", "polpair", "taper"):
                assert legacy.attrs[key] == new.attrs[key], key
            for key in legacy:
                # the window functions span many orders of magnitude, and
                # float re-association (~1e-16) in the k grid moves the tiny
                # tail values by more than rtol, so allow an atol scaled to
                # the array maximum too
                expected = legacy[key][()]
                np.testing.assert_allclose(
                    new[key][()],
                    expected,
                    rtol=1e-9,
                    atol=1e-12 * np.max(np.abs(expected)),
                    err_msg=key,
                )


class TestContainer:
    def test_round_trip_and_header(
        self, vanilla_uvp_with_beam: UVPSpec, tmp_path: Path
    ) -> None:
        path = tmp_path / "container.h5"
        psc = PSpecContainer(str(path), mode="rw", keep_open=False)
        psc.set_pspec("grp", "uvp", vanilla_uvp_with_beam)
        assert psc.get_pspec("grp", "uvp") == vanilla_uvp_with_beam
        with h5py.File(path, "r") as f:
            assert sorted(f["header"].attrs) == ["hera_pspec.version"]
            assert f.attrs["pspec_type"] == "PSpecContainer"
            assert "uvpspec_version" not in f.attrs
            assert f["grp/uvp"].attrs["uvpspec_version"] == "2"


class TestFTBeamFiles:
    def test_v2_attrs(self, tmp_path: Path) -> None:
        path = tmp_path / "ftbeam_xx.hdf5"
        lf.build_ftbeam().write_hdf5(path)
        with h5py.File(path, "r") as f:
            assert f.attrs["ftbeam_version"] == "2"
            assert f["freq"].attrs["unit"] == "Hz"
            assert f["mapsize"].attrs["unit"] == "rad"

    def test_v1_and_v2_read_the_same(self, tmp_path: Path) -> None:
        path = tmp_path / "ftbeam_xx.hdf5"
        uvwindow.FTBeam.from_file(lf.FTBEAM_FILE).write_hdf5(path)
        v1 = uvwindow.FTBeam.from_file(lf.FTBEAM_FILE)
        v2 = uvwindow.FTBeam.from_file(path)
        assert v1 == v2
        np.testing.assert_array_equal(
            uvwindow.FTBeam.get_bandwidth(path),
            uvwindow.FTBeam.get_bandwidth(lf.FTBEAM_FILE),
        )

    def test_equivalent_units_convert(self, tmp_path: Path) -> None:
        path = tmp_path / "ftbeam_xx.hdf5"
        ftbeam = lf.build_ftbeam()
        ftbeam.write_hdf5(path)
        with h5py.File(path, "r+") as f:
            f["freq"][...] = f["freq"][()] / 1e6
            f["freq"].attrs["unit"] = "MHz"
            f["mapsize"][...] = np.rad2deg(f["mapsize"][()])
            f["mapsize"].attrs["unit"] = "deg"
        back = uvwindow.FTBeam.from_file(path)
        np.testing.assert_allclose(back.freq_array, ftbeam.freq_array, rtol=1e-14)
        assert back.mapsize == pytest.approx(ftbeam.mapsize, rel=1e-14)

    @pytest.mark.parametrize("dset,unit", [("freq", "m"), ("mapsize", "Hz")])
    def test_wrong_unit_raises(self, tmp_path: Path, dset: str, unit: str) -> None:
        path = tmp_path / "ftbeam_xx.hdf5"
        lf.build_ftbeam().write_hdf5(path)
        with h5py.File(path, "r+") as f:
            f[dset].attrs["unit"] = unit
        with pytest.raises(ValueError, match=f"{dset} is stored in unit '{unit}'"):
            uvwindow.FTBeam.from_file(path)

    def test_unknown_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ftbeam_xx.hdf5"
        lf.build_ftbeam().write_hdf5(path)
        with h5py.File(path, "r+") as f:
            f.attrs["ftbeam_version"] = "3"
        with pytest.raises(ValueError, match="FTBeam format version '3'"):
            uvwindow.FTBeam.from_file(path)
        with pytest.raises(ValueError, match="FTBeam format version '3'"):
            uvwindow.FTBeam.get_bandwidth(path)


class TestUVWindowFiles:
    @pytest.mark.parametrize(
        "little_h,kunit", [(True, "littleh / Mpc"), (False, "1 / Mpc")]
    )
    def test_v2_attrs(self, tmp_path: Path, little_h: bool, kunit: str) -> None:
        uvw = uvwindow.UVWindow(
            ftbeam_obj=lf.build_uvwindow().ftbeam_obj_pol[0],
            taper="blackman-harris",
            kunits=kunit,
        )
        path = tmp_path / "uvwindow.h5"
        uvw.run_and_write(path, bl_lens=lf.UVWINDOW_BL_LENS[:1] * units.m)
        with h5py.File(path, "r") as f:
            assert f.attrs["uvwindow_version"] == "2"
            # kept for v0.x readers, derived from kunits
            assert f.attrs["little_h"] == little_h
            assert f.attrs["avg_nu_unit"] == "Hz"
            assert f["kperp_bins"].attrs["unit"] == kunit
            assert f["kpara_bins"].attrs["unit"] == kunit
            assert f["bl_lens"].attrs["unit"] == "m"
            assert f["dly_array"].attrs["unit"] == "s"
            assert unit_helpers.parse_unit_string(kunit) == uvw.kunits


class TestUnitHelpers:
    def test_parse_littleh(self) -> None:
        assert unit_helpers.parse_unit_string("littleh / Mpc") == cu.littleh / units.Mpc
        assert unit_helpers.parse_unit_string(b"Hz") == units.Hz
        with pytest.raises(ValueError):
            units.Unit("littleh / Mpc")

    def test_enabled_hera_units_is_scoped(self) -> None:
        with unit_helpers.enabled_hera_units():
            assert units.Unit("littleh") == cu.littleh
        with pytest.raises(ValueError):
            units.Unit("littleh")

    def test_convert_from_stored_unit(self) -> None:
        values = np.array([1.0, 2.0])
        same = unit_helpers.convert_from_stored_unit(values, "s", units.s, "x")
        assert same is values
        assert (
            unit_helpers.convert_from_stored_unit(values, None, units.s, "x") is values
        )
        np.testing.assert_allclose(
            unit_helpers.convert_from_stored_unit(values, b"ns", units.s, "x"),
            values * 1e-9,
        )
        with pytest.raises(ValueError, match="x is stored in unit 'Hz'"):
            unit_helpers.convert_from_stored_unit(values, "Hz", units.s, "x")


class TestNormUnitsFromLegacyString:
    @pytest.mark.parametrize(
        "string,unit,is_deltasq",
        [
            ("h^-3 Mpc^3", units.Mpc**3 / cu.littleh**3, False),
            ("Mpc^3", units.Mpc**3, False),
            ("Hz str [beam normalization not specified]", units.Hz * units.sr, False),
            ("Hz str", units.Hz * units.sr, False),
            ("h^-3 Mpc^3 k^3 / (2pi^2)", units.dimensionless_unscaled, True),
            ("Mpc^3 k^3 / (2pi^2)", units.dimensionless_unscaled, True),
            (
                "Hz str [beam normalization not specified] k^3 / (2pi^2)",
                units.Hz * units.sr / units.Mpc**3,
                True,
            ),
            ("Hz str k^3 / (2pi^2)", units.Hz * units.sr / units.Mpc**3, True),
            (b"h^-3 Mpc^3", units.Mpc**3 / cu.littleh**3, False),
        ],
    )
    def test_known_forms(self, string, unit, is_deltasq: bool) -> None:
        result, deltasq = unit_helpers.norm_units_from_legacy_string(string)
        assert isinstance(result, units.UnitBase)
        assert result == unit
        assert deltasq is is_deltasq

    def test_littleh_is_not_hour(self) -> None:
        unit, _ = unit_helpers.norm_units_from_legacy_string("h^-3 Mpc^3")
        assert not unit.is_equivalent(units.Mpc**3 / units.hour**3)

    @pytest.mark.parametrize(
        "string", ["h^-3 Mpc^3 ", "Mpc3", "mK^2 h^-3 Mpc^3", "", "k^3 / (2pi^2)"]
    )
    def test_unknown_raises(self, string: str) -> None:
        with pytest.raises(ValueError, match="Known forms are"):
            unit_helpers.norm_units_from_legacy_string(string)


class TestCosmologyFromParams:
    def test_matches_default(self) -> None:
        lcdm = conversions.cosmology_from_params(OLD_DEFAULT_COSMO_PARAMS)
        assert lcdm.is_equivalent(conversions.DEFAULT_COSMOLOGY)
        assert lcdm.H0.value == OLD_DEFAULT_COSMO_PARAMS["H0"]
        assert lcdm.Om0 == OLD_DEFAULT_COSMO_PARAMS["Om_M"]
        assert lcdm.Ode0 == OLD_DEFAULT_COSMO_PARAMS["Om_L"]
        assert lcdm.Ob0 == OLD_DEFAULT_COSMO_PARAMS["Om_b"]
        assert lcdm.Tcmb0.value == 0

    def test_round_trip(self) -> None:
        params = {"H0": 70.0, "Om_L": 0.7, "Om_b": 0.04911, "Om_c": 0.26442}
        params["Om_M"] = params["Om_b"] + params["Om_c"]
        params["Om_k"] = 1 - params["Om_L"] - params["Om_M"]
        lcdm = conversions.cosmology_from_params(params)
        back = conversions.params_from_cosmology(lcdm)
        for key, value in params.items():
            assert back[key] == pytest.approx(value, rel=1e-15, abs=1e-15), key

    def test_missing_om_m_and_om_k(self) -> None:
        lcdm = conversions.cosmology_from_params(
            {"H0": 70.0, "Om_L": 0.7, "Om_b": 0.05, "Om_c": 0.25}
        )
        assert lcdm.Om0 == pytest.approx(0.3)
        assert isinstance(lcdm, conversions.LambdaCDM)
        assert not isinstance(lcdm, FlatLambdaCDM)

    def test_inconsistent_om_k_raises(self) -> None:
        params = dict(OLD_DEFAULT_COSMO_PARAMS)
        params["Om_k"] = 0.05
        with pytest.raises(ValueError, match="Om_k"):
            conversions.cosmology_from_params(params)

    def test_om_b_out_of_range_raises(self) -> None:
        params = dict(OLD_DEFAULT_COSMO_PARAMS)
        params["Om_b"] = 0.5
        with pytest.raises(ValueError, match="Om_b"):
            conversions.cosmology_from_params(params)

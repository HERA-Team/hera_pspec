"""Tests for the power spectrum units (vis_units, norm_units, deltasq)."""

import copy
import shutil
from pathlib import Path

import astropy.cosmology.units as cu
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units
from pyuvdata import UVData

from hera_pspec import (
    PSpecBeamUV,
    PSpecContainer,
    PSpecData,
    UVPSpec,
    conversions,
    plot,
    uvpspec,
)
from hera_pspec import uvpspec_utils as uvputils
from hera_pspec.data import DATA_PATH
from tests import legacy_fixtures as lf

DATA_PATH = Path(DATA_PATH)
MPC3_H3 = units.Mpc**3 / cu.littleh**3


def _pspec(
    uvd: UVData, beam=None, vis_units: str | None = None, little_h: bool = True
) -> UVPSpec:
    uvd = copy.deepcopy(uvd)
    if vis_units is not None:
        uvd.vis_units = vis_units
    ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam)
    return ds.pspec(
        [(37, 38), (38, 39)],
        [(38, 39), (37, 38)],
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=[(20, 30)],
        length_unit=units.Mpc / cu.littleh if little_h else units.Mpc,
        store_cov=True,
        verbose=False,
    )


def _read(path: Path, **kwargs) -> UVPSpec:
    uvp = UVPSpec()
    uvp.read_hdf5(path, **kwargs)
    return uvp


def _set_attrs(path: Path, **attrs) -> None:
    with h5py.File(path, "a") as f:
        for key, value in attrs.items():
            f.attrs[key] = value


class TestParsePspecUnit:
    @pytest.mark.parametrize(
        "string,unit",
        [
            ("h^-3 Mpc^3", MPC3_H3),
            ("Mpc^3", units.Mpc**3),
            ("Hz str [beam normalization not specified]", units.Hz * units.sr),
            ("Hz str", units.Hz * units.sr),
            ("h^-3 Mpc^3 k^3 / (2pi^2)", units.dimensionless_unscaled),
            ("Mpc3 / littleh3", MPC3_H3),
            ("Mpc3", units.Mpc**3),
            ("Hz sr", units.Hz * units.sr),
            ("", units.dimensionless_unscaled),
            ("uncalib", uvputils.uncalib),
            ("UNCALIB", uvputils.uncalib),
            ("unknown", uvputils.uncalib),
            ("K str", units.K * units.sr),
            ("k STR", units.K * units.sr),
            ("K sr", units.K * units.sr),
            ("Jy", units.Jy),
            ("JY", units.Jy),
            ("mK", units.mK),
            ("MK", units.mK),
            (b"mK", units.mK),
            ("Mpc3 uncalib2 / littleh3", uvputils.uncalib**2 * MPC3_H3),
        ],
    )
    def test_known_strings(self, string, unit) -> None:
        assert uvputils.parse_pspec_unit(string) == unit

    @pytest.mark.parametrize("string", ["not a unit", "Hz str [other note]", "str"])
    def test_unknown_strings_raise(self, string: str) -> None:
        with pytest.raises(ValueError, match="Cannot interpret"):
            uvputils.parse_pspec_unit(string)

    @pytest.mark.parametrize("string", ["h^-1 Mpc", "Mpc3 / h3", "h"])
    def test_hours_raise(self, string: str) -> None:
        with pytest.raises(ValueError, match="hours"):
            uvputils.parse_pspec_unit(string)

    @pytest.mark.parametrize(
        "string,canonical",
        [
            ("h^-3 Mpc^3", "Mpc3 / littleh3"),
            ("Mpc^3", "Mpc3"),
            ("Hz str [beam normalization not specified]", "Hz sr"),
            ("UNCALIB", "uncalib"),
            ("unknown", "uncalib"),
            ("K str", "K sr"),
            ("jy", "Jy"),
            ("mK", "mK"),
            ("Mpc3 / littleh3", "Mpc3 / littleh3"),
        ],
    )
    def test_canonical_unit_string(self, string: str, canonical: str) -> None:
        assert uvputils.canonical_unit_string(string) == canonical
        # canonical strings are fixed points
        assert uvputils.canonical_unit_string(canonical) == canonical

    @pytest.mark.parametrize(
        "string,canonical,is_deltasq",
        [
            ("h^-3 Mpc^3", "Mpc3 / littleh3", False),
            ("h^-3 Mpc^3 k^3 / (2pi^2)", "Mpc3 / littleh3", True),
            ("Mpc^3 k^3 / (2pi^2)", "Mpc3", True),
            ("Hz str k^3 / (2pi^2)", "Hz sr", True),
            (b"Mpc3", "Mpc3", False),
            ("Hz sr", "Hz sr", False),
        ],
    )
    def test_canonical_norm_units(
        self, string, canonical: str, is_deltasq: bool
    ) -> None:
        assert uvputils.canonical_norm_units(string) == (canonical, is_deltasq)

    def test_is_littleh(self) -> None:
        assert uvputils.is_littleh(MPC3_H3)
        assert uvputils.is_littleh(cu.littleh / units.Mpc)
        assert uvputils.is_littleh("Mpc3 / littleh3")
        assert uvputils.is_littleh("h^-3 Mpc^3")
        assert not uvputils.is_littleh(units.Mpc**3)
        assert not uvputils.is_littleh("Mpc3")
        assert not uvputils.is_littleh("Hz sr")
        assert not uvputils.is_littleh(units.dimensionless_unscaled)

    def test_uncalib_enabled(self) -> None:
        with pytest.raises(ValueError):
            units.Unit("uncalib")
        with uvputils.enabled_hera_units():
            assert units.Unit("uncalib2 Hz sr") == uvputils.uncalib**2 * (
                units.Hz * units.sr
            )
        assert "uncalib" in uvputils.uncalib.to_string("latex_inline")


class TestPspecUnits:
    @pytest.mark.parametrize(
        "vis_units,vis_unit",
        [
            ("UNCALIB", uvputils.uncalib),
            ("Jy", units.Jy),
            ("mK", units.mK),
            ("K str", units.K * units.sr),
        ],
    )
    @pytest.mark.parametrize("little_h", [True, False])
    def test_with_beam(
        self,
        uvd_zen_even_xx: UVData,
        beam_nf_dipole: PSpecBeamUV,
        vis_units,
        vis_unit,
        little_h: bool,
    ) -> None:
        uvp = _pspec(uvd_zen_even_xx, beam_nf_dipole, vis_units, little_h=little_h)
        norm_unit = MPC3_H3 if little_h else units.Mpc**3
        assert uvp.vis_unit == vis_unit
        assert uvp.norm_unit == norm_unit
        assert uvp.units == vis_unit**2 * norm_unit
        assert uvp.vis_units == uvputils.unit_to_string(vis_unit)
        assert uvp.norm_units == ("Mpc3 / littleh3" if little_h else "Mpc3")
        assert uvp.deltasq is False

    def test_without_beam(self, uvd_zen_even_xx: UVData) -> None:
        uvp = _pspec(uvd_zen_even_xx, None, "mK")
        assert uvp.norm_units == "Hz sr"
        assert uvp.vis_units == "mK"
        assert uvp.units == units.mK**2 * units.Hz * units.sr

    def test_unknown_vis_units(self, uvd_zen_even_xx: UVData) -> None:
        uvd = copy.deepcopy(uvd_zen_even_xx)
        uvd.vis_units = "furlongs"
        ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None])
        with pytest.raises(ValueError, match="furlongs"):
            ds.units()

    def test_vanilla(self, vanilla_uvp: UVPSpec) -> None:
        assert vanilla_uvp.vis_units == "uncalib"
        assert vanilla_uvp.norm_units == "Hz sr"
        assert vanilla_uvp.units == uvputils.uncalib**2 * units.Hz * units.sr

    def test_units_is_cached_unit(self, vanilla_uvp: UVPSpec) -> None:
        assert isinstance(vanilla_uvp.units, units.UnitBase)
        assert vanilla_uvp.units is vanilla_uvp.units


class TestDeltasq:
    @pytest.mark.parametrize("little_h", [True, False])
    def test_cosmological(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV, little_h: bool
    ) -> None:
        uvp = _pspec(uvd_zen_even_xx, beam_nf_dipole, "mK", little_h=little_h)
        dsq = uvp.convert_to_deltasq(inplace=False)
        assert dsq.deltasq is True
        assert uvp.deltasq is False
        # norm_units keeps the P(k) normalisation; norm_unit includes k^3
        assert dsq.norm_units == uvp.norm_units
        assert dsq.norm_unit == units.dimensionless_unscaled
        assert dsq.units == units.mK**2
        # k was taken in littleh / Mpc only for the littleh normalisation
        length_unit = conversions.length_unit_from_little_h(little_h)
        k_para = uvp.get_kparas(0, length_unit=length_unit).to_value(1 / length_unit)
        k_perp = uvp.get_kperps(0, length_unit=length_unit).to_value(1 / length_unit)
        coeff = (k_perp[:, None] ** 2 + k_para[None, :] ** 2) ** 1.5 / (2 * np.pi**2)
        np.testing.assert_allclose(
            dsq.data_array[0][..., 0], uvp.data_array[0][..., 0] * coeff, rtol=1e-12
        )

    def test_unnormalised(self, uvd_zen_even_xx: UVData) -> None:
        uvp = _pspec(uvd_zen_even_xx, None, "mK")
        uvp.cosmo = conversions.DEFAULT_COSMOLOGY
        uvp.convert_to_deltasq()
        assert uvp.norm_units == "Hz sr"
        assert uvp.norm_unit == units.Hz * units.sr / units.Mpc**3
        assert uvp.units == units.mK**2 * units.Hz * units.sr / units.Mpc**3

    def test_second_call_raises(self, vanilla_uvp: UVPSpec) -> None:
        uvp = copy.deepcopy(vanilla_uvp)
        uvp.convert_to_deltasq()
        with pytest.raises(ValueError, match="already been converted"):
            uvp.convert_to_deltasq()
        with pytest.raises(ValueError, match="already been converted"):
            uvp.convert_to_deltasq(inplace=False)

    def test_hdf5_round_trip(
        self, uvd_zen_even_xx: UVData, beam_nf_dipole: PSpecBeamUV, tmp_path: Path
    ) -> None:
        uvp = _pspec(uvd_zen_even_xx, beam_nf_dipole, "mK")
        uvp.convert_to_deltasq()
        uvp.write_hdf5(tmp_path / "dsq.h5")
        with h5py.File(tmp_path / "dsq.h5", "r") as f:
            assert bool(f.attrs["deltasq"]) is True
            assert f["data_spw0"].attrs["unit"] == "mK2"
        new = _read(tmp_path / "dsq.h5")
        assert new.deltasq is True
        assert new.units == units.mK**2
        assert new == uvp

    def test_combine_requires_same_deltasq(self, vanilla_uvp: UVPSpec) -> None:
        uvp1 = copy.deepcopy(vanilla_uvp)
        uvp2 = copy.deepcopy(vanilla_uvp)
        uvp2.select(spws=[0], inplace=True)
        uvp2.convert_to_deltasq()
        with pytest.raises(AssertionError, match="deltasq"):
            uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)

    @pytest.mark.parametrize(
        "legacy,canonical,unit",
        [
            (
                "h^-3 Mpc^3 k^3 / (2pi^2)",
                "Mpc3 / littleh3",
                units.dimensionless_unscaled,
            ),
            ("Mpc^3 k^3 / (2pi^2)", "Mpc3", units.dimensionless_unscaled),
            (
                "Hz str [beam normalization not specified] k^3 / (2pi^2)",
                "Hz sr",
                units.Hz * units.sr / units.Mpc**3,
            ),
        ],
    )
    def test_legacy_suffix_read(
        self, tmp_path: Path, legacy: str, canonical: str, unit
    ) -> None:
        path = tmp_path / "legacy.h5"
        shutil.copy(DATA_PATH / "test_uvp.h5", path)
        _set_attrs(path, norm_units=legacy)
        uvp = _read(path)
        assert uvp.deltasq is True
        assert uvp.norm_units == canonical
        assert uvp.norm_unit == unit
        assert uvp.units == uvputils.uncalib**2 * unit
        # little h of k is still known from norm_units
        assert uvputils.is_littleh(uvp.norm_units) is ("h^-3" in legacy)


class TestHdf5Units:
    def test_legacy_test_uvp(self) -> None:
        uvp = _read(DATA_PATH / "test_uvp.h5")
        assert uvp.norm_units == "Mpc3 / littleh3"
        assert uvp.vis_units == "uncalib"
        assert uvp.deltasq is False
        assert uvp.units == uvputils.uncalib**2 * MPC3_H3

    @pytest.mark.parametrize(
        "name,stored_norm,stored_vis,norm_units,vis_units",
        [
            (
                lf.UVP_WINDOWS_NAME,
                "h^-3 Mpc^3",
                "uncalib",
                "Mpc3 / littleh3",
                "uncalib",
            ),
            (lf.UVP_PLAIN_NAME, "Hz str", "unknown", "Hz sr", "uncalib"),
        ],
    )
    def test_legacy_container(
        self, name, stored_norm, stored_vis, norm_units, vis_units
    ) -> None:
        with h5py.File(lf.CONTAINER_FILE, "r") as f:
            grp = f[lf.CONTAINER_GROUP][name]
            assert grp.attrs["norm_units"] == stored_norm
            assert grp.attrs["vis_units"] == stored_vis
            assert "deltasq" not in grp.attrs
        psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
        uvp = psc.get_pspec(lf.CONTAINER_GROUP, name)
        assert uvp.deltasq is False
        assert uvp.norm_units == norm_units
        assert uvp.vis_units == vis_units

    @pytest.mark.parametrize(
        "norm_units,vis_units,canonical_norm,canonical_vis",
        [
            ("h^-3 Mpc^3", "UNCALIB", "Mpc3 / littleh3", "uncalib"),
            ("Mpc^3", "Jy", "Mpc3", "Jy"),
            ("Hz str [beam normalization not specified]", "mK", "Hz sr", "mK"),
            ("Hz str", "unknown", "Hz sr", "uncalib"),
        ],
    )
    def test_v2_with_old_strings(
        self,
        vanilla_uvp: UVPSpec,
        tmp_path: Path,
        norm_units: str,
        vis_units: str,
        canonical_norm: str,
        canonical_vis: str,
    ) -> None:
        """Version 2 files written before the strings became canonical."""
        path = tmp_path / "uvp.h5"
        vanilla_uvp.write_hdf5(path)
        with h5py.File(path, "a") as f:
            f.attrs["norm_units"] = norm_units
            f.attrs["vis_units"] = vis_units
            # files written before this change had no deltasq attribute and
            # no unit attributes on the power spectrum datasets
            del f.attrs["deltasq"]
            for name in f:
                if name.startswith(("data", "cov", "stats")):
                    del f[name].attrs["unit"]
        uvp = _read(path)
        assert uvp.norm_units == canonical_norm
        assert uvp.vis_units == canonical_vis
        assert uvp.deltasq is False

    def test_pspec_dataset_units(self, tmp_path: Path) -> None:
        psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
        uvp = psc.get_pspec(lf.CONTAINER_GROUP, lf.UVP_WINDOWS_NAME)
        uvp.stats_array = {"P_N": {0: np.ones_like(uvp.data_array[0].real)}}
        uvp.write_hdf5(tmp_path / "uvp.h5")
        unit = uvputils.unit_to_string(uvp.units)
        cov_unit = uvputils.unit_to_string(uvp.units**2)
        with h5py.File(tmp_path / "uvp.h5", "r") as f:
            assert f["data_spw0"].attrs["unit"] == unit
            assert f["stats_P_N_0"].attrs["unit"] == unit
            assert f["cov_real_spw0"].attrs["unit"] == cov_unit
            assert f["cov_imag_spw0"].attrs["unit"] == cov_unit
            assert f["window_function_spw0"].attrs["unit"] == ""
            assert "unit" not in f["wgt_spw0"].attrs
            assert "unit" not in f["nsample_spw0"].attrs
        assert unit == "Mpc3 uncalib2 / littleh3"
        assert cov_unit == "Mpc6 uncalib4 / littleh6"
        assert _read(tmp_path / "uvp.h5") == uvp

    @pytest.mark.parametrize("dset", ["data_spw0", "cov_real_spw0", "stats_P_N_0"])
    def test_inconsistent_dataset_unit_raises(
        self, vanilla_uvp: UVPSpec, tmp_path: Path, dset: str
    ) -> None:
        uvp = copy.deepcopy(vanilla_uvp)
        uvp.stats_array = {"P_N": {0: np.ones_like(uvp.data_array[0].real)}}
        uvp.cov_array_real = {0: np.zeros((uvp.Nbltpairs, 50, 50, 1))}
        uvp.cov_array_imag = {0: np.zeros((uvp.Nbltpairs, 50, 50, 1))}
        uvp.cov_model = "empirical"
        path = tmp_path / "uvp.h5"
        uvp.write_hdf5(path)
        with h5py.File(path, "a") as f:
            f[dset].attrs["unit"] = "mK2 Mpc3"
        with pytest.raises(ValueError, match=f"Dataset {dset} is stored with unit"):
            _read(path)
        # just_meta reads do not look at the data
        _read(path, just_meta=True)


class TestPlotLabels:
    def test_delay_spectrum_label(self, uvd_zen_even_xx, beam_nf_dipole) -> None:
        uvp = _pspec(uvd_zen_even_xx, beam_nf_dipole, "mK")
        blps = [[uvp.blpair_to_antnums(blp) for blp in np.unique(uvp.blpair_array)]]
        fig = plot.delay_spectrum(
            uvp, blps, spw=0, pol=("xx", "xx"), average_blpairs=True, delay=False
        )
        label = fig.axes[0].get_ylabel()
        assert r"\mathrm{Mpc^{3}" in label
        assert "h_{100}^{-3}" in label
        assert "mK^{2}" in label
        fig.canvas.draw()
        plt.close(fig)

    def test_waterfall_and_wedge_labels(self, vanilla_uvp: UVPSpec) -> None:
        blps = [
            [
                vanilla_uvp.blpair_to_antnums(b)
                for b in np.unique(vanilla_uvp.blpair_array)
            ]
        ]
        fig = plot.delay_waterfall(
            vanilla_uvp, blps, spw=0, pol=("xx", "xx"), average_blpairs=True
        )
        title = fig._suptitle.get_text()
        assert r"\mathrm{uncalib}^{2}" in title
        assert "Hz" in title
        fig.canvas.draw()
        plt.close(fig)

    def test_units_latex_dimensionless(self) -> None:
        assert plot._units_latex(units.dimensionless_unscaled) == ""
        assert plot._units_latex(units.mK**2) == r"\mathrm{mK^{2}}"

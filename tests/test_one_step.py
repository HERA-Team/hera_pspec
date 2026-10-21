"""Tests for the normalization label of power spectra and for one-step
normalization in cosmological units (#329).

Test names follow the design note's section 6 (T1, T3, T5-T9, T11).
"""

import copy
from pathlib import Path

import astropy.cosmology.units as cu
import h5py
import numpy as np
import pytest
from astropy import units as u

from hera_pspec import PSpecContainer, UVPSpec, grouping, uvpspec
from hera_pspec.data import DATA_PATH
from tests import legacy_fixtures as lf

DATA_PATH = Path(DATA_PATH)


def _read(path):
    uvp = UVPSpec()
    uvp.read_hdf5(path)
    return uvp


def _labelled(uvp, normalization, response_model=None):
    uvp = copy.deepcopy(uvp)
    uvp.normalization = normalization
    if response_model is not None:
        uvp.response_model = response_model
    return uvp


class TestNormalizationParameter:
    """T9: the normalization and response_model parameters."""

    def test_default_and_check(self, vanilla_uvp) -> None:
        assert UVPSpec().normalization == "two-step"
        assert vanilla_uvp.normalization == "two-step"
        assert not hasattr(vanilla_uvp, "response_model")
        assert set(uvpspec.NORMALIZATIONS) == {"two-step", "two-step-exact", "one-step"}
        bad = _labelled(vanilla_uvp, "three-step")
        with pytest.raises(AssertionError, match="normalization must be one of"):
            bad.check()
        missing = copy.deepcopy(vanilla_uvp)
        del missing.normalization
        with pytest.raises(AssertionError, match="normalization"):
            missing.check()

    @pytest.mark.parametrize("normalization", uvpspec.NORMALIZATIONS)
    def test_hdf5_round_trip(self, vanilla_uvp_with_beam, normalization, tmp_path):
        model = "DelayResponse()" if normalization == "one-step" else None
        uvp = _labelled(vanilla_uvp_with_beam, normalization, model)
        path = tmp_path / "uvp.h5"
        uvp.write_hdf5(path)
        with h5py.File(path, "r") as f:
            assert f.attrs["uvpspec_version"] == "2"
            assert f.attrs["normalization"] == normalization
            assert ("response_model" in f.attrs) == (model is not None)
        uvp_in = _read(path)
        assert uvp_in.normalization == normalization
        assert getattr(uvp_in, "response_model", None) == model
        assert uvp_in == uvp

    def test_pre_v1_version_2_file_reads_as_two_step(
        self, vanilla_uvp_with_beam, tmp_path
    ) -> None:
        # a version 2 file written before the normalization attribute existed
        path = tmp_path / "uvp.h5"
        _labelled(vanilla_uvp_with_beam, "one-step").write_hdf5(path)
        with h5py.File(path, "r+") as f:
            del f.attrs["normalization"]
        assert _read(path).normalization == "two-step"

    def test_legacy_files_read_as_two_step(self) -> None:
        with h5py.File(DATA_PATH / "test_uvp.h5", "r") as f:
            assert "normalization" not in f.attrs
        assert _read(DATA_PATH / "test_uvp.h5").normalization == "two-step"
        with h5py.File(lf.CONTAINER_FILE, "r") as f:
            group = f[lf.CONTAINER_GROUP]
            for name in group:
                assert "normalization" not in group[name].attrs
        psc = PSpecContainer(str(lf.CONTAINER_FILE), mode="r", keep_open=False)
        for name in psc.spectra(lf.CONTAINER_GROUP):
            uvp = psc.get_pspec(lf.CONTAINER_GROUP, name)
            assert uvp.normalization == "two-step"

    def test_equality(self, vanilla_uvp) -> None:
        assert _labelled(vanilla_uvp, "one-step") != vanilla_uvp
        assert _labelled(vanilla_uvp, "one-step", "a") != _labelled(
            vanilla_uvp, "one-step", "b"
        )
        assert _labelled(vanilla_uvp, "one-step", "a") == _labelled(
            vanilla_uvp, "one-step", "a"
        )

    def test_combine_refuses_mixed_normalizations(self, vanilla_uvp) -> None:
        blpairs = np.unique(vanilla_uvp.blpair_array)
        uvp1 = vanilla_uvp.select(blpairs=blpairs[:1], inplace=False)
        uvp2 = vanilla_uvp.select(blpairs=blpairs[1:], inplace=False)
        combined = uvpspec.combine_uvpspec([uvp1, uvp2], verbose=False)
        assert combined.normalization == "two-step"
        one_step = _labelled(uvp2, "one-step", "DelayResponse()")
        with pytest.raises(AssertionError, match="'normalization'"):
            uvpspec.combine_uvpspec([uvp1, one_step], verbose=False)
        both = [_labelled(uvp1, "one-step", "DelayResponse()"), one_step]
        combined = uvpspec.combine_uvpspec(both, verbose=False)
        assert combined.normalization == "one-step"
        assert combined.response_model == "DelayResponse()"
        other_model = _labelled(uvp1, "one-step", "DelayResponse(x2y='center')")
        with pytest.raises(AssertionError, match="'response_model'"):
            uvpspec.combine_uvpspec([other_model, one_step], verbose=False)

    def test_carried_through(self, vanilla_uvp) -> None:
        uvp = _labelled(vanilla_uvp, "one-step", "DelayResponse()")
        folded = copy.deepcopy(uvp)
        folded.fold_spectra()
        kbins = np.array([0.1, 0.2]) * cu.littleh / u.Mpc
        outputs = [
            uvp.select(spws=[0], inplace=False),
            uvp.average_spectra(time_avg=True, inplace=False),
            grouping.average_spectra(uvp, time_avg=True, inplace=False),
            folded,
            grouping.spherical_average(uvp, kbins, np.array([0.05, 0.05])),
        ]
        for out in outputs:
            assert out.normalization == "one-step"
            assert out.response_model == "DelayResponse()"

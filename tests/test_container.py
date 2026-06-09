from pathlib import Path

import numpy as np
import pytest

from hera_pspec import PSpecContainer, UVPSpec, container, testing
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

_MODES = pytest.mark.parametrize(
    "keep_open,swmr", [(True, False), (False, True)], ids=["default", "transactional"]
)

_GROUP_NAMES = ["group1", "group2", "group3"]
_PSPEC_NAMES = ["pspec_dset(0,1)", "pspec_dset(1,0)", "pspec_dset(1,1)"]


def _container_fname(tmp_path: Path) -> Path:
    return tmp_path / "_test_container.hdf5"


def _fill_container(fname: Path, uvp: UVPSpec) -> None:
    """Populate a container with _GROUP_NAMES × _PSPEC_NAMES entries."""
    ps_store = PSpecContainer(fname, mode="rw", swmr=False)
    for grp in _GROUP_NAMES:
        for psname in _PSPEC_NAMES:
            ps_store.set_pspec(group=grp, psname=psname, pspec=uvp, overwrite=False)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_invalid_mode(tmp_path: Path) -> None:
    fname = _container_fname(tmp_path)
    with pytest.raises(ValueError, match="Must set mode to either"):
        PSpecContainer(fname, mode="x")


# ---------------------------------------------------------------------------
# set_pspec
# ---------------------------------------------------------------------------


class TestSetPSpec:
    @_MODES
    def test_overwrite(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        for psname in _PSPEC_NAMES:
            ps_store.set_pspec(
                group=_GROUP_NAMES[2], psname=psname, pspec=vanilla_uvp, overwrite=True
            )
        with pytest.raises(AttributeError, match="already exists and overwrite=False"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[2],
                psname=_PSPEC_NAMES[-1],
                pspec=vanilla_uvp,
                overwrite=False,
            )

    @_MODES
    @pytest.mark.parametrize("bad", [np.arange(11), 1, "abc"])
    def test_invalid_type(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, bad, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        with pytest.raises(TypeError, match="pspec must be a UVPSpec object"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[2],
                psname=_PSPEC_NAMES[-1],
                pspec=bad,
                overwrite=True,
            )

    @_MODES
    def test_list(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        with pytest.raises(ValueError, match="Only one group can be specified"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[:2],
                psname=_PSPEC_NAMES[0],
                pspec=vanilla_uvp,
                overwrite=True,
            )
        with pytest.raises(ValueError, match="If psname is a list"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[0],
                psname=_PSPEC_NAMES,
                pspec=vanilla_uvp,
                overwrite=True,
            )
        with pytest.raises(ValueError, match="If pspec is a list"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[0],
                psname=_PSPEC_NAMES[0],
                pspec=[vanilla_uvp, vanilla_uvp, vanilla_uvp],
                overwrite=True,
            )
        with pytest.raises(TypeError, match="pspec lists must only contain UVPSpec objects"):
            ps_store.set_pspec(
                group=_GROUP_NAMES[0],
                psname=_PSPEC_NAMES,
                pspec=[vanilla_uvp, None, vanilla_uvp],
                overwrite=True,
            )
        # valid list write
        ps_store.set_pspec(
            group=_GROUP_NAMES[0],
            psname=_PSPEC_NAMES,
            pspec=[vanilla_uvp, vanilla_uvp, vanilla_uvp],
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# get_pspec
# ---------------------------------------------------------------------------


class TestGetPSpec:
    @_MODES
    def test_retrieval(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        for i in range(len(_GROUP_NAMES)):
            ps = ps_store.get_pspec(_GROUP_NAMES[i], psname=_PSPEC_NAMES[i])
            assert isinstance(ps, UVPSpec)
        ps_list = ps_store.get_pspec(_GROUP_NAMES[0])
        assert len(ps_list) == len(_PSPEC_NAMES)
        for p in ps_list:
            assert isinstance(p, UVPSpec)

    @_MODES
    @pytest.mark.parametrize("bad_group", ["x", 1, ["x", "y"]])
    def test_invalid_group_key(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, bad_group, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        with pytest.raises(KeyError, match="No group named"):
            ps_store.get_pspec(bad_group, _PSPEC_NAMES[0])

    @_MODES
    @pytest.mark.parametrize("bad_pspec", ["x", 1, ["x", "y"]])
    def test_invalid_pspec_key(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, bad_pspec, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        with pytest.raises(KeyError, match="No pspec named"):
            ps_store.get_pspec(_GROUP_NAMES[0], bad_pspec)

    @_MODES
    def test_partial_io(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
        ps = ps_store.get_pspec(_GROUP_NAMES[0], _PSPEC_NAMES[0], just_meta=True)
        assert not hasattr(ps, "data_array")
        assert hasattr(ps, "time_avg_array")
        ps = ps_store.get_pspec(
            _GROUP_NAMES[0], _PSPEC_NAMES[0], blpairs=[((1, 2), (1, 2))]
        )
        assert hasattr(ps, "data_array")
        assert np.all(np.isclose(ps.blpair_array, 101102101102))


# ---------------------------------------------------------------------------
# Misc container interface
# ---------------------------------------------------------------------------


@_MODES
def test_readonly_mode(
    tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
) -> None:
    fname = _container_fname(tmp_path)
    _fill_container(fname, vanilla_uvp)
    ps_readonly = PSpecContainer(fname, mode="r", keep_open=keep_open, swmr=swmr)
    ps = ps_readonly.get_pspec(_GROUP_NAMES[0], _PSPEC_NAMES[0])
    assert isinstance(ps, UVPSpec)
    with pytest.raises(IOError, match="HDF5 file was opened read-only"):
        ps_readonly.set_pspec(
            group=_GROUP_NAMES[2],
            psname=_PSPEC_NAMES[2],
            pspec=vanilla_uvp,
            overwrite=True,
        )


@_MODES
def test_groups_and_spectra(
    tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
) -> None:
    fname = _container_fname(tmp_path)
    _fill_container(fname, vanilla_uvp)
    ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
    grplist = ps_store.groups()
    pslist = ps_store.spectra(group=_GROUP_NAMES[0])
    assert len(grplist) == len(_GROUP_NAMES)
    assert len(pslist) == len(_PSPEC_NAMES)
    for g in grplist:
        assert g in _GROUP_NAMES
    with pytest.raises(KeyError, match="No group named"):
        ps_store.spectra("x")
    for g in ps_store.groups():
        for psname in ps_store.spectra(group=g):
            ps = ps_store.get_pspec(g, psname=psname)
            assert isinstance(ps, UVPSpec)


@_MODES
def test_container_repr(
    tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
) -> None:
    fname = _container_fname(tmp_path)
    _fill_container(fname, vanilla_uvp)
    ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
    print(ps_store)
    assert len(ps_store.tree()) > 0


@_MODES
def test_save(
    tmp_path: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
) -> None:
    fname = _container_fname(tmp_path)
    _fill_container(fname, vanilla_uvp)
    ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)
    ps_store.save()


# ---------------------------------------------------------------------------
# SWMR / transactional mode
# ---------------------------------------------------------------------------


class TestSWMR:
    def test_concurrent_access(self, tmp_path: Path, vanilla_uvp: UVPSpec) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        psc_rw = PSpecContainer(fname, mode="rw", keep_open=False, swmr=True)
        psc_ro = PSpecContainer(fname, mode="r", keep_open=False, swmr=True)
        assert len(psc_ro.groups()) == len(_GROUP_NAMES)
        psc_ro_noatom = PSpecContainer(fname, mode="r", keep_open=True, swmr=True)
        assert len(psc_ro_noatom.groups()) == len(_GROUP_NAMES)
        # transactional RO still works; RW fails while non-transactional reader is open
        assert len(psc_ro.groups()) == len(_GROUP_NAMES)
        with pytest.raises(OSError, match="Failed to open HDF5 file"):
            psc_rw.groups()
        with pytest.raises(OSError, match="Failed to open HDF5 file"):
            psc_rw.groups()
        # once the non-transactional reader closes, the RW handle works again
        psc_ro_noatom._close()
        psc_rw.set_pspec(
            group=_GROUP_NAMES[0],
            psname=_PSPEC_NAMES[0],
            pspec=vanilla_uvp,
            overwrite=True,
        )

    def test_blocks_new_writes(self, tmp_path: Path, vanilla_uvp: UVPSpec) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        psc_rw = PSpecContainer(fname, mode="rw", keep_open=False, swmr=True)
        with pytest.raises(ValueError, match="Cannot write new group or dataset with SWMR"):
            psc_rw.set_pspec(
                group="new_group",
                psname=_PSPEC_NAMES[0],
                pspec=vanilla_uvp,
                overwrite=True,
            )
        with pytest.raises(ValueError, match="Cannot write new group or dataset with SWMR"):
            psc_rw.set_pspec(
                group=_GROUP_NAMES[0],
                psname="new_psname",
                pspec=vanilla_uvp,
                overwrite=True,
            )

    @pytest.mark.parametrize("mode", ["r", "rw"])
    def test_attr_propagation(
        self, tmp_path: Path, vanilla_uvp: UVPSpec, mode: str
    ) -> None:
        fname = _container_fname(tmp_path)
        _fill_container(fname, vanilla_uvp)
        psc = PSpecContainer(fname, mode=mode, keep_open=True, swmr=True)
        assert psc.swmr
        assert psc.data.swmr_mode
        psc._close()
        psc = PSpecContainer(fname, mode=mode, keep_open=True, swmr=False)
        assert not psc.swmr
        assert not psc.data.swmr_mode
        psc._close()


# ---------------------------------------------------------------------------
# combine_psc_spectra
# ---------------------------------------------------------------------------


class TestCombinePscSpectra:
    def test_basic(self, tmp_path: Path) -> None:
        fname = str(DATA_PATH / "zen.2458042.17772.xx.HH.uvXA")
        uvp1 = testing.uvpspec_from_data(fname, [(24, 25), (37, 38)], spw_ranges=[(10, 40)])
        uvp2 = testing.uvpspec_from_data(fname, [(38, 39), (52, 53)], spw_ranges=[(10, 40)])
        psc = PSpecContainer(tmp_path / "ex.h5", mode="rw")
        psc.set_pspec("grp1", "uvp_a", uvp1, overwrite=True)
        psc.set_pspec("grp1", "uvp_b", uvp2, overwrite=True)
        container.combine_psc_spectra(psc, dset_split_str=None, ext_split_str="_")
        assert psc.spectra("grp1") == ["uvp"]

    @pytest.mark.parametrize("name_a,name_b,expected", [
        ("d1_x_d2_a", "d1_x_d2_b", "d1_x_d2"),
        ("d2_x_d3_a", "d2_x_d3_b", "d2_x_d3"),
    ])
    def test_dset_names(
        self, tmp_path: Path, name_a: str, name_b: str, expected: str
    ) -> None:
        data_fname = str(DATA_PATH / "zen.2458042.17772.xx.HH.uvXA")
        uvp1 = testing.uvpspec_from_data(
            data_fname, [(24, 25), (37, 38)], spw_ranges=[(10, 40)]
        )
        uvp2 = testing.uvpspec_from_data(
            data_fname, [(38, 39), (52, 53)], spw_ranges=[(10, 40)]
        )
        psc = PSpecContainer(tmp_path / "ex.h5", mode="rw")
        psc.set_pspec("grp1", name_a, uvp1, overwrite=True)
        psc.set_pspec("grp1", name_b, uvp2, overwrite=True)
        container.combine_psc_spectra(
            str(tmp_path / "ex.h5"), dset_split_str="_x_", ext_split_str="_"
        )
        assert psc.spectra("grp1") == [expected]

    def test_errors(self, tmp_path: Path) -> None:
        fname = str(DATA_PATH / "zen.2458042.17772.xx.HH.uvXA")
        uvp1 = testing.uvpspec_from_data(fname, [(24, 25), (37, 38)], spw_ranges=[(10, 40)])
        psc = PSpecContainer(tmp_path / "ex.h5", mode="rw")
        with pytest.raises(AssertionError, match="no specified groups exist"):
            container.combine_psc_spectra(psc)
        # incompatible uvpspecs should leave originals intact
        psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
        psc.set_pspec("grp1", "d1_x_d2_b", uvp1, overwrite=True)
        container.combine_psc_spectra(psc, dset_split_str="_x_", ext_split_str="_")
        assert psc.spectra("grp1") == ["d1_x_d2_a", "d1_x_d2_b"]

    def test_argparser(self) -> None:
        args = container.get_combine_psc_spectra_argparser()
        a = args.parse_args(
            ["filename", "--dset_split_str", "_x_", "--ext_split_str", "_"]
        )
        assert a.filename == "filename"
        assert a.dset_split_str == "_x_"
        assert a.ext_split_str == "_"

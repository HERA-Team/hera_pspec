from pathlib import Path

import numpy as np
import pytest

from hera_pspec import PSpecContainer, UVPSpec, container, testing
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


@pytest.fixture
def container_fname(tmp_path: Path) -> Path:
    """Empty PSpecContainer HDF5 file at a temporary path."""
    fname = tmp_path / "_test_container.hdf5"
    _ = PSpecContainer(fname, mode="rw", swmr=False)
    return fname


def _fill_container(fname: Path, uvp: UVPSpec) -> tuple[list[str], list[str]]:
    """Helper: populate container and return (group_names, pspec_names)."""
    group_names = ["group1", "group2", "group3"]
    pspec_names = ["pspec_dset(0,1)", "pspec_dset(1,0)", "pspec_dset(1,1)"]
    ps_store = PSpecContainer(fname, mode="rw", swmr=False)
    for grp in group_names:
        for psname in pspec_names:
            ps_store.set_pspec(group=grp, psname=psname, pspec=uvp, overwrite=False)
    return group_names, pspec_names


@pytest.mark.parametrize(
    "keep_open,swmr", [(True, False), (False, True)], ids=["default", "transactional"]
)
def test_PSpecContainer(
    container_fname: Path, vanilla_uvp: UVPSpec, keep_open: bool, swmr: bool
) -> None:
    """
    Test that PSpecContainer works properly.
    """
    # setup, fill and open container
    group_names, pspec_names = _fill_container(container_fname, vanilla_uvp)
    fname = container_fname
    ps_store = PSpecContainer(fname, mode="rw", keep_open=keep_open, swmr=swmr)

    # Make sure that invalid mode arguments are caught
    with pytest.raises(ValueError, match="Must set mode to either"):
        PSpecContainer(fname, mode="x")

    # Check that power spectra can be overwritten
    for psname in pspec_names:
        ps_store.set_pspec(
            group=group_names[2], psname=psname, pspec=vanilla_uvp, overwrite=True
        )

    # Check that overwriting fails if overwrite=False
    with pytest.raises(AttributeError, match="already exists and overwrite=False"):
        ps_store.set_pspec(
            group=group_names[2], psname=psname, pspec=vanilla_uvp, overwrite=False
        )

    # Check that wrong pspec types are rejected by the set() method
    with pytest.raises(TypeError, match="pspec must be a UVPSpec object"):
        ps_store.set_pspec(
            group=group_names[2], psname=psname, pspec=np.arange(11), overwrite=True
        )
    with pytest.raises(TypeError, match="pspec must be a UVPSpec object"):
        ps_store.set_pspec(group=group_names[2], psname=psname, pspec=1, overwrite=True)
    with pytest.raises(TypeError, match="pspec must be a UVPSpec object"):
        ps_store.set_pspec(
            group=group_names[2], psname=psname, pspec="abc", overwrite=True
        )

    # Check that power spectra can be retrieved one by one
    for i in range(len(group_names)):
        # Get one pspec from each group
        ps = ps_store.get_pspec(group_names[i], psname=pspec_names[i])
        assert isinstance(ps, UVPSpec)

    # Check that power spectra can be retrieved from group as a list
    ps_list = ps_store.get_pspec(group_names[0])
    assert len(ps_list) == len(pspec_names)
    for p in ps_list:
        assert isinstance(p, UVPSpec)

    # Check that asking for an invalid group or pspec raises an error
    with pytest.raises(KeyError, match="No group named"):
        ps_store.get_pspec("x", pspec_names[0])
    with pytest.raises(KeyError, match="No group named"):
        ps_store.get_pspec(1, pspec_names[0])
    with pytest.raises(KeyError, match="No group named"):
        ps_store.get_pspec(["x", "y"], pspec_names[0])
    with pytest.raises(KeyError, match="No pspec named"):
        ps_store.get_pspec(group_names[0], "x")
    with pytest.raises(KeyError, match="No pspec named"):
        ps_store.get_pspec(group_names[0], 1)
    with pytest.raises(KeyError, match="No pspec named"):
        ps_store.get_pspec(group_names[0], ["x", "y"])

    # Check that printing functions work
    print(ps_store)
    assert len(ps_store.tree()) > 0

    # Check that read-only mode is respected
    ps_readonly = PSpecContainer(fname, mode="r", keep_open=keep_open, swmr=swmr)
    ps = ps_readonly.get_pspec(group_names[0], pspec_names[0])
    assert isinstance(ps, UVPSpec)
    with pytest.raises(IOError, match="HDF5 file was opened read-only"):
        ps_readonly.set_pspec(
            group=group_names[2],
            psname=pspec_names[2],
            pspec=vanilla_uvp,
            overwrite=True,
        )

    # Check that spectra() and groups() methods return the things we put in
    print("ps_store:", ps_store.data)
    grplist = ps_store.groups()
    pslist = ps_store.spectra(group=group_names[0])
    assert len(grplist) == len(group_names)
    assert len(pslist) == len(pspec_names)
    for g in grplist:
        assert g in group_names

    # Check that spectra() raises an error if group doesn't exist
    with pytest.raises(KeyError, match="No group named"):
        ps_store.spectra("x")

    # Check that spectra() and groups() can be iterated over to retrieve ps
    for g in ps_store.groups():
        for psname in ps_store.spectra(group=g):
            ps = ps_store.get_pspec(g, psname=psname)
            assert isinstance(ps, UVPSpec)

    # check partial IO in get_pspec
    ps = ps_store.get_pspec(group_names[0], pspec_names[0], just_meta=True)
    assert not hasattr(ps, "data_array")
    assert hasattr(ps, "time_avg_array")
    ps = ps_store.get_pspec(group_names[0], pspec_names[0], blpairs=[((1, 2), (1, 2))])
    assert hasattr(ps, "data_array")
    assert np.all(np.isclose(ps.blpair_array, 101102101102))

    # Check that invalid list arguments raise errors in set_pspec()
    with pytest.raises(ValueError, match="Only one group can be specified"):
        ps_store.set_pspec(
            group=group_names[:2],
            psname=pspec_names[0],
            pspec=vanilla_uvp,
            overwrite=True,
        )
    with pytest.raises(ValueError, match="If psname is a list"):
        ps_store.set_pspec(
            group=group_names[0], psname=pspec_names, pspec=vanilla_uvp, overwrite=True
        )
    with pytest.raises(ValueError, match="If pspec is a list"):
        ps_store.set_pspec(
            group=group_names[0],
            psname=pspec_names[0],
            pspec=[vanilla_uvp, vanilla_uvp, vanilla_uvp],
            overwrite=True,
        )
    with pytest.raises(
        TypeError, match="pspec lists must only contain UVPSpec objects"
    ):
        ps_store.set_pspec(
            group=group_names[0],
            psname=pspec_names,
            pspec=[vanilla_uvp, None, vanilla_uvp],
            overwrite=True,
        )

    # Check that lists can be used to set pspec
    ps_store.set_pspec(
        group=group_names[0],
        psname=pspec_names,
        pspec=[vanilla_uvp, vanilla_uvp, vanilla_uvp],
        overwrite=True,
    )

    # Check that save() can be called
    ps_store.save()


def test_container_transactional_mode(
    container_fname: Path, vanilla_uvp: UVPSpec
) -> None:
    """
    Test transactional operations on PSpecContainer objects.
    """
    # setup, fill and open container
    group_names, pspec_names = _fill_container(container_fname, vanilla_uvp)
    fname = container_fname

    # Test to see whether concurrent read/write works
    psc_rw = PSpecContainer(fname, mode="rw", keep_open=False, swmr=True)

    # Try to open container read-only (transactional)
    psc_ro = PSpecContainer(fname, mode="r", keep_open=False, swmr=True)
    assert len(psc_ro.groups()) == len(group_names)

    # Open container read-only in non-transactional mode
    psc_ro_noatom = PSpecContainer(fname, mode="r", keep_open=True, swmr=True)
    assert len(psc_ro_noatom.groups()) == len(group_names)

    # Original RO handle should work fine; RW handle will throw an error
    assert len(psc_ro.groups()) == len(group_names)
    with pytest.raises(OSError, match="Failed to open HDF5 file"):
        psc_rw.groups()
    with pytest.raises(OSError, match="Failed to open HDF5 file"):
        psc_rw.groups()

    # Close the non-transactional file; the RW file should now work
    psc_ro_noatom._close()
    psc_rw.set_pspec(
        group=group_names[0], psname=pspec_names[0], pspec=vanilla_uvp, overwrite=True
    )

    # test that write of new group or dataset with SWMR is blocked
    with pytest.raises(ValueError, match="Cannot write new group or dataset with SWMR"):
        psc_rw.set_pspec(
            group="new_group", psname=pspec_names[0], pspec=vanilla_uvp, overwrite=True
        )
    with pytest.raises(ValueError, match="Cannot write new group or dataset with SWMR"):
        psc_rw.set_pspec(
            group=group_names[0], psname="new_psname", pspec=vanilla_uvp, overwrite=True
        )

    # ensure SWMR attr is propagated
    for m in ["r", "rw"]:
        psc = PSpecContainer(fname, mode=m, keep_open=True, swmr=True)
        assert psc.swmr
        assert psc.data.swmr_mode
        psc._close()
        psc = PSpecContainer(fname, mode=m, keep_open=True, swmr=False)
        assert not psc.swmr
        assert not psc.data.swmr_mode
        psc._close()


def test_combine_psc_spectra(tmp_path: Path) -> None:
    fname = str(DATA_PATH / "zen.2458042.17772.xx.HH.uvXA")
    uvp1 = testing.uvpspec_from_data(fname, [(24, 25), (37, 38)], spw_ranges=[(10, 40)])
    uvp2 = testing.uvpspec_from_data(fname, [(38, 39), (52, 53)], spw_ranges=[(10, 40)])

    # test basic execution
    psc = PSpecContainer(tmp_path / "ex1.h5", mode="rw")
    psc.set_pspec("grp1", "uvp_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "uvp_b", uvp2, overwrite=True)
    container.combine_psc_spectra(psc, dset_split_str=None, ext_split_str="_")
    assert psc.spectra("grp1") == ["uvp"]

    # test dset name handling
    psc = PSpecContainer(tmp_path / "ex2.h5", mode="rw")
    psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d1_x_d2_b", uvp2, overwrite=True)
    psc.set_pspec("grp1", "d2_x_d3_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d2_x_d3_b", uvp2, overwrite=True)
    container.combine_psc_spectra(
        str(tmp_path / "ex2.h5"), dset_split_str="_x_", ext_split_str="_"
    )
    spec_list = psc.spectra("grp1")
    spec_list.sort()
    assert spec_list == ["d1_x_d2", "d2_x_d3"]

    # test exceptions
    psc = PSpecContainer(tmp_path / "ex3.h5", mode="rw")
    # test no group exception
    with pytest.raises(AssertionError, match="no specified groups exist"):
        container.combine_psc_spectra(psc)
    # test failed combine_uvpspec
    psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d1_x_d2_b", uvp1, overwrite=True)
    container.combine_psc_spectra(psc, dset_split_str="_x_", ext_split_str="_")
    assert psc.spectra("grp1") == ["d1_x_d2_a", "d1_x_d2_b"]


def test_combine_psc_spectra_argparser() -> None:
    args = container.get_combine_psc_spectra_argparser()
    a = args.parse_args(["filename", "--dset_split_str", "_x_", "--ext_split_str", "_"])
    assert a.filename == "filename"
    assert a.dset_split_str == "_x_"
    assert a.ext_split_str == "_"

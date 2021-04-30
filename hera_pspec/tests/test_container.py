import unittest
import pytest
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from .. import container, PSpecContainer, UVPSpec, testing

class Test_PSpecContainer(unittest.TestCase):

    def setUp(self, fill_pspec=True):
        # create container: first without SWMR to add header
        self.fname = os.path.join(DATA_PATH, '_test_container.hdf5')
        self.uvp, self.cosmo = testing.build_vanilla_uvpspec()

        # Create empty container without SWMR to fill header
        if os.path.exists(self.fname):
            os.remove(self.fname)
        ps_store = PSpecContainer(self.fname, mode='rw', swmr=False)
        del ps_store

    def tearDown(self):
        # Remove HDF5 file
        try:
            os.remove(self.fname)
        except:
            print("No HDF5 file to remove.")

    def runTest(self):
        pass

    def fill_container(self):
        ps_store = PSpecContainer(self.fname, mode='rw', swmr=False)
        self.group_names = ['group1', 'group2', 'group3']
        self.pspec_names = ['pspec_dset(0,1)', 'pspec_dset(1,0)', 'pspec_dset(1,1)']
        for grp in self.group_names:
            for psname in self.pspec_names:
                ps_store.set_pspec(group=grp, psname=psname,
                                   pspec=self.uvp, overwrite=False)
        del ps_store


    def test_PSpecContainer(self, keep_open=True, swmr=False):
        """
        Test that PSpecContainer works properly.
        """
        # setup, fill and open container
        self.fill_container()
        fname = self.fname
        group_names, pspec_names = self.group_names, self.pspec_names
        ps_store = PSpecContainer(self.fname, mode='rw', keep_open=keep_open, swmr=swmr)

        # Make sure that invalid mode arguments are caught
        pytest.raises(ValueError, PSpecContainer, fname, mode='x')

        # Check that power spectra can be overwritten
        for psname in pspec_names:
            ps_store.set_pspec(group=group_names[2], psname=psname,
                               pspec=self.uvp, overwrite=True)

        # Check that overwriting fails if overwrite=False
        pytest.raises(AttributeError, ps_store.set_pspec, group=group_names[2],
                      psname=psname, pspec=self.uvp, overwrite=False)

        # Check that wrong pspec types are rejected by the set() method
        pytest.raises(TypeError, ps_store.set_pspec, group=group_names[2],
                      psname=psname, pspec=np.arange(11), overwrite=True)
        pytest.raises(TypeError, ps_store.set_pspec, group=group_names[2],
                      psname=psname, pspec=1, overwrite=True)
        pytest.raises(TypeError, ps_store.set_pspec, group=group_names[2],
                      psname=psname, pspec="abc", overwrite=True)

        # Check that power spectra can be retrieved one by one
        for i in range(len(group_names)):
            # Get one pspec from each group
            ps = ps_store.get_pspec(group_names[i], psname=pspec_names[i])
            assert(isinstance(ps, UVPSpec))

        # Check that power spectra can be retrieved from group as a list
        ps_list = ps_store.get_pspec(group_names[0])
        assert(len(ps_list) == len(pspec_names))
        for p in ps_list: assert(isinstance(p, UVPSpec))

        # Check that asking for an invalid group or pspec raises an error
        pytest.raises(KeyError, ps_store.get_pspec, 'x', pspec_names[0])
        pytest.raises(KeyError, ps_store.get_pspec, 1, pspec_names[0])
        pytest.raises(KeyError, ps_store.get_pspec, ['x', 'y'], pspec_names[0])
        pytest.raises(KeyError, ps_store.get_pspec, group_names[0], 'x')
        pytest.raises(KeyError, ps_store.get_pspec, group_names[0], 1)
        pytest.raises(KeyError, ps_store.get_pspec, group_names[0], ['x','y'])

        # Check that printing functions work
        print(ps_store)
        assert(len(ps_store.tree()) > 0)

        # Check that read-only mode is respected
        ps_readonly = PSpecContainer(fname, mode='r', keep_open=keep_open, swmr=swmr)
        ps = ps_readonly.get_pspec(group_names[0], pspec_names[0])
        assert(isinstance(ps, UVPSpec))
        pytest.raises(IOError, ps_readonly.set_pspec, group=group_names[2],
                      psname=pspec_names[2], pspec=self.uvp, overwrite=True)

        # Check that spectra() and groups() methods return the things we put in
        print("ps_store:", ps_store.data)
        grplist = ps_store.groups()
        pslist = ps_store.spectra(group=group_names[0])
        assert( len(grplist) == len(group_names) )
        assert( len(pslist) == len(pspec_names) )
        for g in grplist: assert(g in group_names)

        # Check that spectra() raises an error if group doesn't exist
        pytest.raises(KeyError, ps_store.spectra, "x")

        # Check that spectra() and groups() can be iterated over to retrieve ps
        for g in ps_store.groups():
            for psname in ps_store.spectra(group=g):
                ps = ps_store.get_pspec(g, psname=psname)
                assert(isinstance(ps, UVPSpec))

        # check partial IO in get_pspec
        ps = ps_store.get_pspec(group_names[0], pspec_names[0], just_meta=True)
        assert not hasattr(ps, 'data_array')
        assert hasattr(ps, 'time_avg_array')
        ps = ps_store.get_pspec(group_names[0], pspec_names[0], blpairs=[((1, 2), (1, 2))])
        assert hasattr(ps, 'data_array')
        assert np.all(np.isclose(ps.blpair_array, 101102101102))

        # Check that invalid list arguments raise errors in set_pspec()
        pytest.raises(ValueError, ps_store.set_pspec, group=group_names[:2],
                      psname=pspec_names[0], pspec=self.uvp, overwrite=True)
        pytest.raises(ValueError, ps_store.set_pspec, group=group_names[0],
                      psname=pspec_names, pspec=self.uvp, overwrite=True)
        pytest.raises(ValueError, ps_store.set_pspec, group=group_names[0],
                      psname=pspec_names[0], pspec=[self.uvp, self.uvp, self.uvp],
                      overwrite=True)
        pytest.raises(TypeError, ps_store.set_pspec, group=group_names[0],
                      psname=pspec_names, pspec=[self.uvp, None, self.uvp],
                      overwrite=True)

        # Check that lists can be used to set pspec
        ps_store.set_pspec(group=group_names[0], psname=pspec_names,
                           pspec=[self.uvp, self.uvp, self.uvp], overwrite=True)

        # Check that save() can be called
        ps_store.save()

    def test_PSpecContainer_transactional(self):
        """
        Test that PSpecContainer works properly (transactional mode).
        """
        self.test_PSpecContainer(keep_open=False, swmr=True)

    def test_container_transactional_mode(self):
        """
        Test transactional operations on PSpecContainer objects.
        """
        # setup, fill and open container
        self.fill_container()
        fname = self.fname
        group_names, pspec_names = self.group_names, self.pspec_names

        # Test to see whether concurrent read/write works
        psc_rw = PSpecContainer(fname, mode='rw', keep_open=False, swmr=True)

        # Try to open container read-only (transactional)
        psc_ro = PSpecContainer(fname, mode='r', keep_open=False, swmr=True)
        assert len(psc_ro.groups()) == len(group_names)

        # Open container read-only in non-transactional mode
        psc_ro_noatom = PSpecContainer(fname, mode='r', keep_open=True, swmr=True)
        assert len(psc_ro_noatom.groups()) == len(group_names)

        # Original RO handle should work fine; RW handle will throw an error
        assert len(psc_ro.groups()) == len(group_names)
        pytest.raises(OSError, psc_rw.groups)
        pytest.raises(OSError, psc_rw.groups)

        # Close the non-transactional file; the RW file should now work
        psc_ro_noatom._close()
        psc_rw.set_pspec(group=group_names[0], psname=pspec_names[0],
                         pspec=self.uvp, overwrite=True)

        # test that write of new group or dataset with SWMR is blocked
        pytest.raises(ValueError, psc_rw.set_pspec, group='new_group', psname=pspec_names[0],
                      pspec=self.uvp, overwrite=True)
        pytest.raises(ValueError, psc_rw.set_pspec, group=group_names[0], psname='new_psname',
                      pspec=self.uvp, overwrite=True)

        # ensure SWMR attr is propagated
        for m in ['r', 'rw']:
            psc = PSpecContainer(fname, mode=m, keep_open=True, swmr=True)
            assert psc.swmr
            assert psc.data.swmr_mode
            psc._close()
            psc = PSpecContainer(fname, mode=m, keep_open=True, swmr=False)
            assert not psc.swmr
            assert not psc.data.swmr_mode
            psc._close()

def test_combine_psc_spectra():
    fname = os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA")
    uvp1 = testing.uvpspec_from_data(fname, [(24, 25), (37, 38)],
                                     spw_ranges=[(10, 40)])
    uvp2 = testing.uvpspec_from_data(fname, [(38, 39), (52, 53)],
                                     spw_ranges=[(10, 40)])

    # test basic execution
    if os.path.exists('ex.h5'):
        os.remove('ex.h5')
    psc = PSpecContainer("ex.h5", mode='rw')
    psc.set_pspec("grp1", "uvp_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "uvp_b", uvp2, overwrite=True)
    container.combine_psc_spectra(psc, dset_split_str=None, ext_split_str='_')
    assert psc.spectra('grp1') == [u'uvp']

    # test dset name handling
    if os.path.exists('ex.h5'):
        os.remove('ex.h5')
    psc = PSpecContainer("ex.h5", mode='rw')
    psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d1_x_d2_b", uvp2, overwrite=True)
    psc.set_pspec("grp1", "d2_x_d3_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d2_x_d3_b", uvp2, overwrite=True)
    container.combine_psc_spectra('ex.h5', dset_split_str='_x_', ext_split_str='_')
    spec_list = psc.spectra('grp1')
    spec_list.sort()
    assert spec_list == [u'd1_x_d2', u'd2_x_d3']

    # test exceptions
    if os.path.exists('ex.h5'):
        os.remove('ex.h5')
    psc = PSpecContainer("ex.h5", mode='rw')
    # test no group exception
    pytest.raises(AssertionError, container.combine_psc_spectra, psc)
    # test failed combine_uvpspec
    psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d1_x_d2_b", uvp1, overwrite=True)
    container.combine_psc_spectra(psc, dset_split_str='_x_', ext_split_str='_')
    assert psc.spectra('grp1') == [u'd1_x_d2_a', u'd1_x_d2_b']

    if os.path.exists("ex.h5"):
        os.remove("ex.h5")


def test_combine_psc_spectra_argparser():
    args = container.get_combine_psc_spectra_argparser()
    a = args.parse_args(["filename", "--dset_split_str", "_x_", "--ext_split_str", "_"])
    assert a.filename == "filename"
    assert a.dset_split_str == "_x_"
    assert a.ext_split_str == "_"

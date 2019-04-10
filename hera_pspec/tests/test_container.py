import unittest
from nose.tools import assert_raises
import nose.tools as nt
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from hera_pspec import container, PSpecContainer, UVPSpec, testing


class Test_PSpecContainer(unittest.TestCase):
    
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, '_test_container.hdf5')
        self.uvp, self.cosmo = testing.build_vanilla_uvpspec()

    def tearDown(self):
        # Remove HDF5 file
        try:
            os.remove(self.fname)
        except:
            print("No HDF5 file to remove.")

    def runTest(self):
        pass

    def test_PSpecContainer(self, keep_open=True):
        """
        Test that PSpecContainer works properly.
        """
        fname = self.fname
        group_names = ['group1', 'group2', 'group3']
        pspec_names = ['pspec_dset(0,1)', 'pspec_dset(1,0)', 'pspec_dset(1,1)']
        
        # Create a new container
        ps_store = PSpecContainer(fname, mode='rw', keep_open=keep_open)
        
        # Make sure that invalid mode arguments are caught
        assert_raises(ValueError, PSpecContainer, fname, mode='x')
        
        # Check that multiple power spectra can be stored in the container
        for grp in group_names:
            for psname in pspec_names:
                ps_store.set_pspec(group=grp, psname=psname, 
                                   pspec=self.uvp, overwrite=False)
        
        # Check that power spectra can be overwritten
        for psname in pspec_names:
            ps_store.set_pspec(group=group_names[2], psname=psname, 
                               pspec=self.uvp, overwrite=True)
        
        # Check that overwriting fails if overwrite=False
        assert_raises(AttributeError, ps_store.set_pspec, group=group_names[2], 
                      psname=psname, pspec=self.uvp, overwrite=False)
        
        # Check that wrong pspec types are rejected by the set() method
        assert_raises(TypeError, ps_store.set_pspec, group=group_names[2], 
                      psname=psname, pspec=np.arange(11), overwrite=True)
        assert_raises(TypeError, ps_store.set_pspec, group=group_names[2], 
                      psname=psname, pspec=1, overwrite=True)
        assert_raises(TypeError, ps_store.set_pspec, group=group_names[2], 
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
        assert_raises(KeyError, ps_store.get_pspec, 'x', pspec_names[0])
        assert_raises(KeyError, ps_store.get_pspec, 1, pspec_names[0])
        assert_raises(KeyError, ps_store.get_pspec, ['x', 'y'], pspec_names[0])
        assert_raises(KeyError, ps_store.get_pspec, group_names[0], 'x')
        assert_raises(KeyError, ps_store.get_pspec, group_names[0], 1)
        assert_raises(KeyError, ps_store.get_pspec, group_names[0], ['x','y'])
        
        # Check that printing functions work
        print(ps_store)
        assert(len(ps_store.tree()) > 0)
        
        # Check that read-only mode is respected
        ps_readonly = PSpecContainer(fname, mode='r', keep_open=keep_open)
        ps = ps_readonly.get_pspec(group_names[0], pspec_names[0])
        assert(isinstance(ps, UVPSpec))
        assert_raises(IOError, ps_readonly.set_pspec, group=group_names[2], 
                      psname=pspec_names[2], pspec=self.uvp, overwrite=True)
        
        # Check that spectra() and groups() methods return the things we put in
        print("ps_store:", ps_store.data)
        grplist = ps_store.groups()
        pslist = ps_store.spectra(group=group_names[0])
        assert( len(grplist) == len(group_names) )
        assert( len(pslist) == len(pspec_names) )
        for g in grplist: assert(g in group_names)
        
        # Check that spectra() raises an error if group doesn't exist
        assert_raises(KeyError, ps_store.spectra, "x")
        
        # Check that spectra() and groups() can be iterated over to retrieve ps
        for g in ps_store.groups():
            for psname in ps_store.spectra(group=g):
                ps = ps_store.get_pspec(g, psname=psname)
                assert(isinstance(ps, UVPSpec))
        
        # Check that invalid list arguments raise errors in set_pspec()
        assert_raises(ValueError, ps_store.set_pspec, group=['g1', 'g2'], 
                      psname=pspec_names[0], pspec=self.uvp, overwrite=True)
        assert_raises(ValueError, ps_store.set_pspec, group='g1', 
                      psname=pspec_names, pspec=self.uvp, overwrite=True)
        assert_raises(ValueError, ps_store.set_pspec, group='g1', 
                      psname=pspec_names[0], pspec=[self.uvp,self.uvp,self.uvp], 
                      overwrite=True)
        assert_raises(TypeError, ps_store.set_pspec, group='g1', 
                      psname=pspec_names, pspec=[self.uvp,None,self.uvp], 
                      overwrite=True)
                      
        # Check that lists can be used to set pspec
        ps_store.set_pspec(group='g1', psname=pspec_names, 
                           pspec=[self.uvp,self.uvp,self.uvp], overwrite=True)
        
        # Check that save() can be called
        ps_store.save()
        
    
    def test_PSpecContainer_transactional(self):
        """
        Test that PSpecContainer works properly (transactional mode).
        """
        self.test_PSpecContainer(keep_open=False)
    
    
    def test_container_transactional_mode(self):
        """
        Test transactional operations on PSpecContainer objects.
        """
        fname = self.fname
        group_names = ['group1', 'group2', 'group3']
        pspec_names = ['pspec_dset(0,1)', 'pspec_dset(1,0)', 'pspec_dset(1,1)']
        
        # Test to see whether concurrent read/write works
        psc_rw = PSpecContainer(fname, mode='rw', keep_open=False)
        
        # Check that multiple power spectra can be stored in the container
        for grp in group_names:
            for psname in pspec_names:
                psc_rw.set_pspec(group=grp, psname=psname, 
                                 pspec=self.uvp, overwrite=False)
        
        # Try to open container read-only (transactional)
        psc_ro = PSpecContainer(fname, mode='r', keep_open=False)
        nt.assert_equal(len(psc_ro.groups()), len(group_names))
        
        # Open container read-only in non-transactional mode
        psc_ro_noatom = PSpecContainer(fname, mode='r', keep_open=True)
        nt.assert_equal(len(psc_ro_noatom.groups()), len(group_names))
        
        # Original RO handle should work fine; RW handle will throw an error
        nt.assert_equal(len(psc_ro.groups()), len(group_names))
        if sys.version_info[0] < 3: # check Python version
            assert_raises(IOError, psc_rw.groups)
        else:
            assert_raises(OSError, psc_rw.groups)
        
        # Close the non-transactional file; the RW file should now work
        psc_ro_noatom._close()
        psc_rw.set_pspec(group='group4', psname=psname, 
                         pspec=self.uvp, overwrite=False)
        nt.assert_equal(len(psc_rw.groups()), len(group_names)+1)
        

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
    nt.assert_equal(psc.spectra('grp1'), [u'uvp'])

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
    nt.assert_equal(spec_list, [u'd1_x_d2', u'd2_x_d3'])

    # test exceptions
    if os.path.exists('ex.h5'):
        os.remove('ex.h5')
    psc = PSpecContainer("ex.h5", mode='rw')
    # test no group exception
    nt.assert_raises(AssertionError, container.combine_psc_spectra, psc)
    # test failed combine_uvpspec
    psc.set_pspec("grp1", "d1_x_d2_a", uvp1, overwrite=True)
    psc.set_pspec("grp1", "d1_x_d2_b", uvp1, overwrite=True)
    container.combine_psc_spectra(psc, dset_split_str='_x_', ext_split_str='_')
    nt.assert_equal(psc.spectra('grp1'), [u'd1_x_d2_a', u'd1_x_d2_b'])

    if os.path.exists("ex.h5"):
        os.remove("ex.h5")
        

def test_combine_psc_spectra_argparser():
    args = container.get_combine_psc_spectra_argparser()
    a = args.parse_args(["filename", "--dset_split_str", "_x_", "--ext_split_str", "_"])
    nt.assert_equal(a.filename, "filename")
    nt.assert_equal(a.dset_split_str, "_x_")
    nt.assert_equal(a.ext_split_str, "_")



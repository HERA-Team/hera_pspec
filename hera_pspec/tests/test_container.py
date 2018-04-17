import unittest
from nose.tools import assert_raises
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from hera_pspec import PSpecContainer, UVPSpec
from test_uvpspec import build_example_uvpspec

class Test_PSpecContainer(unittest.TestCase):
    
    def setUp(self):
        self.fname = os.path.join(DATA_PATH, '_test_container.hdf5')
        self.uvp, self.cosmo = build_example_uvpspec()

    def tearDown(self):
        # Remove HDF5 file
        try:
            os.remove(self.fname)
        except:
            print("No HDF5 file to remove.")

    def runTest(self):
        pass

    def test_PSpecContainer(self):
        """
        Test that PSpecContainer works properly.
        """
        fname = self.fname
        group_names = ['group1', 'group2', 'group3']
        pspec_names = ['pspec_dset(0,1)', 'pspec_dset(1,0)', 'pspec_dset(1,1)']
        
        # Create a new container
        ps_store = PSpecContainer(fname, mode='rw')
        
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
        assert_raises(ValueError, ps_store.set_pspec, group=group_names[2], 
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
        ps_readonly = PSpecContainer(fname, mode='r')
        ps = ps_readonly.get_pspec(group_names[0], pspec_names[0])
        assert(isinstance(ps, UVPSpec))
        assert_raises(IOError, ps_readonly.set_pspec, group=group_names[2], 
                      psname=pspec_names[2], pspec=self.uvp, overwrite=True)
        
        # Check that spectra() and groups() methods return the things we put in
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
        

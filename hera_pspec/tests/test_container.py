import unittest
from nose.tools import assert_raises
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from hera_pspec import PSpecContainer, UVPSpec

class Test_PSpecContainer(unittest.TestCase):
    
    def setUp(self):
        self.fname = "%s/_test_container.hdf5" % DATA_PATH
        self.setUp_uvpspec()    
    
    def setUp_uvpspec(self):
        """
        Build an example UVPSpec object.
        """
        uvp = UVPSpec()

        Ntimes = 10
        Nfreqs = 50
        Ndlys = Nfreqs
        Nspws = 1
        Nspwdlys = Nspws * Nfreqs

        # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
        blpairs = [1002001002, 2003002003, 1003001003]
        bls = [1002, 2003, 1003]
        Nbls = len(bls)
        Nblpairs = len(blpairs)
        Nblpairts = Nblpairs * Ntimes

        blpair_array = np.tile(blpairs, Ntimes)
        bl_array = np.array(bls)
        bl_vecs = [[  5.33391548e+00,  -1.35907816e+01,  -7.91624188e-09],
                   [ -8.67982998e+00,   4.43554478e+00,  -1.08695203e+01],
                   [ -3.34591450e+00,  -9.15523687e+00,  -1.08695203e+01]]
        bl_vecs = np.array(bl_vecs)
        time_array = np.repeat(np.linspace(2458042.1, 2458042.2, Ntimes), Nblpairs)
        time_1_array = time_array
        time_2_array = time_array
        lst_array = np.repeat(np.ones(Ntimes, dtype=np.float), Nblpairs)
        lst_1_array = lst_array
        lst_2_array = lst_array
        spws = np.arange(Nspws)
        spw_array = np.tile(spws, Ndlys)
        freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), 
                               Nspws)
        dly_array = np.fft.fftshift(np.repeat(
                        np.fft.fftfreq(Nfreqs, np.median(np.diff(freq_array))), 
                        Nspws))
        pol_array = np.array([-5])
        Npols = len(pol_array)
        units = 'unknown'
        weighting = 'identity'
        channel_width = np.median(np.diff(freq_array))
        history = 'example'
        taper = "none"
        norm = "I"
        git_hash = "random"

        telescope_location = np.array([5109325.85521063, 
                                       2005235.09142983, 
                                      -3239928.42475397])

        data_array, wgt_array, integration_array = {}, {}, {}
        for s in spws:
            data_array[s] = np.ones((Nblpairts,Ndlys,Npols), dtype=np.complex) \
                          * blpair_array[:, None, None] / 1e9
            wgt_array[s] = np.ones((Nblpairts, Ndlys, 2, Npols), dtype=np.float)
            integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)

        params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nblpairs', 
                  'Nblpairts', 'Npols', 'Ndlys', 'Nbls', 'blpair_array', 
                  'time_1_array', 'time_2_array', 'lst_1_array', 'lst_2_array',
                  'spw_array', 'dly_array', 'freq_array', 'pol_array', 
                  'data_array', 'wgt_array', 'integration_array', 'bl_array', 
                  'bl_vecs', 'telescope_location', 'units', 'channel_width', 
                  'weighting', 'history', 'taper', 'norm', 'git_hash']
        for p in params: setattr(uvp, p, locals()[p])
        self.uvp = uvp

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
        
        # Check that overwriting fails if disallowed
        assert_raises(AttributeError, ps_store.set_pspec, group=group_names[2], 
                      psname=psname, pspec=self.uvp, overwrite=False)
        
        # Check that wrong pspec types are rejected by the set() method
        assert_raises(ValueError, ps_store.set_pspec, group=group_names[2], 
                      psname=psname, pspec=np.arange(11), overwrite=True)
        
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
        # Check that lists can be used to set pspec
        ps_store.set_pspec(group='g1', psname=pspec_names, 
                           pspec=[self.uvp,self.uvp,self.uvp], overwrite=True)
        
        # Check that save() can be called
        ps_store.save()
        

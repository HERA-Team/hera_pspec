import numpy as np
import h5py
from uvpspec import UVPSpec

class PSpecContainer(object):
    """
    Container class for managing multiple UVPSpec objects.
    """
    
    def __init__(self, filename, mode='r'):
        """
        Manage a collection of UVPSpec objects that are stored in a structured 
        HDF5 file. 
        
        Parameters
        ----------
        filename : str
            Path to HDF5 file to store power spectra in.
        
        mode : str
            Whether to load the HDF5 file as read/write ('rw') or read-only 
            ('r'). If 'rw' is specified and the file doesn't exist, an empty 
            one will be created.
        """
        self.filename = filename
        self.mode = mode
        if mode not in ['r', 'rw']:
            raise ValueError("Must set mode to either 'r' or 'rw'.")
        
        # Open file ready for reading and/or writing
        self.data = None
        self._open()
    
    
    def _open(self):
        """
        Open HDF5 file ready for reading/writing.
        """
        # Convert user-specified mode to a mode that HDF5 recognizes. We only 
        # allow non-destructive operations!
        mode = 'a' if self.mode == 'rw' else 'r'
        self.data = h5py.File(self.filename, mode)
    
    
    def _store_pspec(self, pspec, ps):
        """
        Store a UVPSpec object as group of datasets within the HDF5 file.
        
        Parameters
        ----------
        pspec : HDF5 group
            HDF5 group to store power spectrum data in.
        
        ps : UVPSpec
            Object containing power spectrum and related data.
        """
        # Get data and attributes from UVPSpec object (stored in dicts)
        data, attrs = ps.serialize()
        
        # Store data into HDF5 file
        for key in data.keys():
            ds = pspec.create_dataset("%s" % key, 
                                      data[key].shape, 
                                      data[key].dtype)
            ds[:] = data[key][:]
        
        # Store attributes into HDF5 file
        for key in attrs.keys():
            pspec.attrs[key] = attrs[key]
    
    
    self _load_pspec(self, grp):
        """
        Load a new UVPSpec object from a HDF5 group.
        
        Parameters
        ----------
        grp : HDF5 group
            Group containing datasets that contain power spectrum and 
            supporting information in a standard format expected by UVPSpec.
        """
        # Load data and attributes from HDF5 group
        data_dict = {key : grp[key] for key in grp.keys()}
        attr_dict = {key : grp.attrs[key] for key in grp.attrs.keys()}
        
        # Package data into a new UVPspec object
        pspec = UVPSpec(data_dict=data, attr_dict=attrs)
        return pspec
    
    
    def set_pspec(self, group, pspec, ps):
        """
        Store a delay power spectrum in the container.
        
        Parameters
        ----------
        group : str, optional
            Which group the power spectrum belongs to.
        
        pspec : str, optional
            The name of the power spectrum to return from within the group.
            
        ps : UVPSpec
            Power spectrum object to store in the container.
        """
        key1 = "%s" % group
        key2 = "%s" % pspec
        
        # Check that the group exists
        if key1 not in self.data.keys():
            grp = self.data.create_group(key1)
        else:
            grp = self.data[key1]
        
        # Check that the pspec exists
        if key2 not in grp.keys():
            pspec = grp.create_group(key2)
        else:
            pspec = grp[key2]
        
        # Add power spectrum to this group
        self._store_pspec(pspec, ps)
    
    
    def get_pspec(self, group, pspec=None):
        """
        Get a UVPSpec power spectrum object from a given group.
        
        Parameters
        ----------
        group : str, optional
            Which group the power spectrum belongs to.
        
        pspec : str, optional
            The name of the power spectrum to return from within the group.
        
        Returns
        -------
        pspec : UVPSpec
            The specified power spectrum, as a UVPSpec object.
        """
        # Check that group is in keys and extract it if so
        key1 = "%s" % group
        if key1 in self.data.keys():
            grp = self.data[key1]
        else:
            raise KeyError("No group named '%s'" % key1)
        
        # Return the whole group if pspec not specified
        if pspec is None: return grp
        
        # Check that pspec is in keys and extract it if so
        key2 = "%s" % pspec
        if key2 in grp.keys():
            return self._load_pspec(grp[key2])
        else:
            raise KeyError("No pspec named '%s' in group '%s'" % (key2, key1))
        

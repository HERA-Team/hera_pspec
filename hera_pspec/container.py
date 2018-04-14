import numpy as np
import h5py
from hera_pspec.uvpspec import UVPSpec
import hera_pspec.version as version

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
        if self.mode == 'rw':
            self._update_header()
    
    
    def _store_pspec(self, pspec_group, ps):
        """
        Store a UVPSpec object as group of datasets within the HDF5 file.
        
        Parameters
        ----------
        pspec : HDF5 group
            HDF5 group to store power spectrum data in.
        
        ps : UVPSpec
            Object containing power spectrum and related data.
        """
        if self.mode == 'r':
            raise IOError("HDF5 file was opened read-only; cannot write to file.")
        
        # Get data and attributes from UVPSpec object (stored in dicts)
        assert isinstance(ps, UVPSpec)
        
        # Write UVPSpec to group
        ps.write_to_group(pspec_group, run_check=True)
        
    
    def _load_pspec(self, grp):
        """
        Load a new UVPSpec object from a HDF5 group.
        
        Parameters
        ----------
        grp : HDF5 group
            Group containing datasets that contain power spectrum and 
            supporting information, in a standard format expected by UVPSpec.
        """
        # Check that group is tagged as containing UVPSpec (pspec_type attribute)
        if 'pspec_type' in grp.attrs.keys():
            if grp.attrs['pspec_type'] != UVPSpec.__name__:
                raise TypeError("HDF5 group is not tagged as a UVPSpec object.")
        else:
            raise TypeError("HDF5 group is not tagged as a UVPSpec object.")
        
        # Create new UVPSpec object and fill with data from this group
        pspec = UVPSpec()
        pspec.read_from_group(grp)
        return pspec
    
    
    def _update_header(self):
        """
        Update the header in the HDF5 file with useful metadata, including the 
        git version of hera_pspec.
        """
        if 'header' not in self.data.keys():
            hdr = self.data.create_group('header')
        else:
            hdr = self.data['header']
        
        # Check if versions of hera_pspec are the same
        if 'hera_pspec.git_hash' in hdr.attrs.keys():
            if hdr.attrs['hera_pspec.git_hash'] != version.git_hash:
                print("WARNING: HDF5 file was created by a different version "
                      "of hera_pspec.")
        hdr.attrs['hera_pspec.git_hash'] = version.git_hash
        
    
    def set_pspec(self, group, psname, pspec, overwrite=False):
        """
        Store a delay power spectrum in the container.
        
        Parameters
        ----------
        group : str
            Which group the power spectrum belongs to.
        
        psname : str or list of str
            The name(s) of the power spectrum to return from within the group.
            
        pspec : UVPSpec or list of UVPSpec
            Power spectrum object(s) to store in the container.
        
        overwrite : bool, optional
            If the power spectrum already exists in the file, whether it should 
            overwrite it or raise an error. Default: False (does not overwrite).
        """
        if self.mode == 'r':
            raise IOError("HDF5 file was opened read-only; cannot write to file.")
        
        if getattr(group, '__iter__', False):
            raise ValueError("Only one group can be specified at a time.")
        
        # Handle input arguments that are iterable (i.e. sequences, but not str)
        if getattr(psname, '__iter__', False):
            if getattr(pspec, '__iter__', False) and len(pspec) == len(psname):
                # Recursively call set_pspec() on each item of the list
                for _psname, _pspec in zip(psname, pspec):
                    if not isinstance(_pspec, UVPSpec):
                        raise TypeError("pspec lists must only contain UVPSpec "
                                        "objects.")
                    self.set_pspec(group, _psname, _pspec, overwrite=overwrite)
                return
            else:
                # Raise exception if psname is a list, but pspec is not
                raise ValueError("If psname is a list, pspec must be a list of "
                                 "the same length.")
        if getattr(pspec, '__iter__', False) \
          and not getattr(psname, '__iter__', False):
            raise ValueError("If pspec is a list, psname must also be a list.")
        # No lists should pass beyond this point
        
        # Check that input is of the correct type
        if not isinstance(pspec, UVPSpec):
            raise TypeError("pspec must be a UVPSpec object.")
        
        key1 = "%s" % group
        key2 = "%s" % psname
        
        # Check that the group exists
        if key1 not in self.data.keys():
            grp = self.data.create_group(key1)
        else:
            grp = self.data[key1]
        
        # Check that the psname exists
        if key2 not in grp.keys():
            # Create group if it doesn't exist
            psgrp = grp.create_group(key2)
        else:
            if overwrite:
                # Delete group and recreate
                del grp[key2]
                psgrp = grp.create_group(key2)
            else:
                raise AttributeError(
                   "Power spectrum %s/%s already exists and overwrite=False." \
                   % (key1, key2) )
        
        # Add power spectrum to this group
        self._store_pspec(psgrp, pspec)
        
        # Store info about what kind of power spectra are in the group
        psgrp.attrs['pspec_type'] = pspec.__class__.__name__
    
    
    def get_pspec(self, group, psname=None):
        """
        Get a UVPSpec power spectrum object from a given group.
        
        Parameters
        ----------
        group : str, optional
            Which group the power spectrum belongs to.
        
        psname : str, optional
            The name of the power spectrum to return.
        
        Returns
        -------
        pspec : UVPSpec or list of UVPSpec
            The specified power spectrum, as a UVPSpec object (or a list, if 
            pname was not specified).
        """
        # Check that group is in keys and extract it if so
        key1 = "%s" % group
        if key1 in self.data.keys():
            grp = self.data[key1]
        else:
            raise KeyError("No group named '%s'" % key1)
        
        # If psname was specified, check that it exists and extract
        if psname is not None:
            key2 = "%s" % psname
            
            # Load power spectrum if it exists
            if key2 in grp.keys():
                return self._load_pspec(grp[key2])
            else:
                raise KeyError("No pspec named '%s' in group '%s'" % (key2, key1))
        
        
        # Otherwise, extract all available power spectra
        spectra = []
        def pspec_filter(n, obj):
            if u'pspec_type' in obj.attrs.keys():
                spectra.append(self._load_pspec(obj))
        
        # Traverse the entire set of groups/datasets looking for pspecs
        grp.visititems(pspec_filter)
        return spectra
        
    
    def spectra(self, group):
        """
        Return list of available power spectra.
        
        Parameters
        ----------
        group : str
            Which group to list power spectra from.
        
        Returns
        -------
        ps_list : list of str
            List of names of power spectra in the group.
        """
        # Check that group is in keys and extract it if so
        key1 = "%s" % group
        if key1 in self.data.keys():
            grp = self.data[key1]
        else:
            raise KeyError("No group named '%s'" % key1)
        
        # Filter to look for pspec objects
        ps_list = []
        def pspec_filter(n, obj):
            if u'pspec_type' in obj.attrs.keys():
                ps_list.append(n)
        
        # Traverse the entire set of groups/datasets looking for pspecs
        grp.visititems(pspec_filter)
        return ps_list
    
    def groups(self):
        """
        Return list of groups in the container.
        
        Returns
        -------
        group_list : list of str
            List of group names.
        """
        groups = self.data.keys()
        if u'header' in groups: groups.remove(u'header')
        return groups
    
    def tree(self):
        """
        Output a string containing a tree diagram of groups and the power 
        spectra that they contain.
        """
        s = ""
        for grp in self.groups():
            s += "(%s)\n" % grp
            for pspec in self.spectra(grp):
                s += "  |--%s\n" % pspec
        return s
    
    def save(self):
        """
        Force HDF5 file to flush to disk.
        """
        self.data.flush()
    
    def __del__(self):
        """
        Make sure that HDF5 file is closed on destruct.
        """
        try:
            self.data.close()
        except:
            pass

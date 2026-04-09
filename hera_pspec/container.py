import numpy as np
import h5py
import argparse
import time
from functools import wraps
import warnings

from . import uvpspec, __version__, utils


def transactional(fn):
    """
    Handle 'transactional' operations on PSpecContainer, where the HDF5 file is 
    opened and then closed again for every operation. This is done when 
    keep_open = False.

    For calling a @transactional function within another @transactional function,
    feed the kwarg nested=True in the nested function call, and it will not close
    the container upon exit even if keep_open = False, until the outer-most
    @transactional function exits, in which case it will close the container
    if keep_open = False.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        psc = args[0] # self object
        
        # Open HDF5 file if needed
        if psc.data is None:
            psc._open()

        # if passed 'nested' kwarg get it
        nested = kwargs.pop('nested', False)

        # Run function
        try:
            f = fn(*args, **kwargs)
        except Exception as err:
            # Close file before raising error
            if not psc.keep_open:
                psc._close()
            raise err

        # Close HDF5 file if necessary
        if not psc.keep_open and not nested:
            psc._close()
        
        # Return function result
        return f
    
    return wrapper


class PSpecContainer(object):
    """
    Container class for managing multiple UVPSpec objects.
    """
    def __init__(self, filename, mode='r', keep_open=True, swmr=False,
                 tsleep=0.1, maxiter=1):
        """
        Manage a collection of UVPSpec objects that are stored in a structured
        HDF5 file.

        Note: one should not create new groups or datasets with SWMR. See page 6 of
        https://support.hdfgroup.org/HDF5/docNewFeatures/SWMR/HDF5_SWMR_Users_Guide.pdf
        for SWMR limitations.

        Parameters
        ----------
        filename : str
            Path to HDF5 file to store power spectra in.

        mode : str
            Whether to load the HDF5 file as read/write ('rw') or read-only
            ('r'). If 'rw' is specified and the file doesn't exist, an empty
            one will be created.
        
        keep_open : bool, optional
            Whether the HDF5 file should be kept open, or opened and then 
            closed again each time an operation is performed. Setting 
            `keep_open=False` is helpful for multi-process access patterns.
            Default: True (keep file open).

        swmr : bool, optional
            Enable Single-Writer Multiple-Reader (SWMR) feature of HDF5.
            Note that SWMR can only be used on POSIX-compliant 
            filesystems, and so may not work on some network filesystems.
            Default: False (do not use SWMR)

        tsleep : float, optional
            Time to wait in seconds after each attempt at opening the file.

        maxiter : int, optional
            Maximum number of attempts to open file (useful for concurrent 
            access when file may be locked temporarily by other processes).
        """
        self.filename = filename
        self.keep_open = keep_open
        self.mode = mode
        self.tsleep = tsleep
        self.maxiter = maxiter
        self.swmr = swmr
        if mode not in ['r', 'rw']:
            raise ValueError("Must set mode to either 'r' or 'rw'.")

        # Open file ready for reading and/or writing (if not in transactional mode)
        self.data = None
        if keep_open:
            self._open()

    def _open(self):
        """
        Open HDF5 file ready for reading/writing. Does nothing if the file is 
        already open.
        
        This method uses HDF5's single writer, multiple reader (swmr) mode, 
        which allows multiple handles to exist for the same file at the same 
        time, as long as only one is in rw mode. The rw instance should be the 
        *first* one that is created; if a read-only instance is already open 
        when a rw instance is created, an error will be raised by h5py. 
        """
        if self.data is not None: return

        # Convert user-specified mode to a mode that HDF5 recognizes. We only
        # allow non-destructive operations!
        if self.mode == 'rw':
            mode = 'a'
        else:
            mode = 'r'
        if self.swmr and self.mode == 'r':
            swmr = True
        else:
            swmr = False

        # check HDF5 version if swmr
        if swmr:
            hdf5_v = float('.'.join(h5py.version.hdf5_version.split('.')[:2]))
            if hdf5_v < 1.1:
                raise NotImplementedError("HDF5 version is {}: must "
                                          "be >= 1.10 for SWMR".format(hdf5_v))

        # Try to open the file
        Ncount = 0
        while True:
            try:
                self.data = h5py.File(self.filename, mode, libver='latest', 
                                      swmr=swmr)
                if self.mode == 'rw':
                    try:
                        # Enable single writer, multiple reader mode on HDF5 file. 
                        # This allows multiple handles to exist for the same file 
                        # at the same time, as long as only one is in rw mode
                        if self.swmr:
                            self.data.swmr_mode = True
                    except ValueError:
                        pass
                break
            except (IOError, OSError):
                # raise Exception if exceeded maxiter
                if Ncount >= self.maxiter:
                    if self.mode == 'rw':
                        raise OSError(
                            "Failed to open HDF5 file. Another process may "
                            "be holding it open; use \nkeep_open=False to "
                            "help prevent this from happening (single "
                            "process), or use the\nlock kwarg (multiple "
                            "processes).")
                    else:
                        raise
                
                # sleep and try again
                Ncount += 1
                time.sleep(self.tsleep)

        # Update header info
        if self.mode == 'rw':
            # Update header
            self._update_header()

            # Denote as Container
            if 'pspec_type' not in list(self.data.attrs.keys()):
                self.data.attrs['pspec_type'] = self.__class__.__name__

    def _close(self):
        """
        Close HDF5 file. DOes nothing if file is already closed.
        """
        if self.data is None:
            return
        self.data.close()
        self.data = None
    
    def _store_pspec(self, pspec_group, uvp):
        """
        Store a UVPSpec object as group of datasets within the HDF5 file.

        Parameters
        ----------
        pspec_group : HDF5 group
            HDF5 group to store power spectrum data in.

        uvp : UVPSpec
            Object containing power spectrum and related data.
        """
        if self.mode == 'r':
            raise IOError("HDF5 file was opened read-only; cannot write to file.")

        # Get data and attributes from UVPSpec object (stored in dicts)
        assert isinstance(uvp, uvpspec.UVPSpec)

        # Write UVPSpec to group
        uvp.write_to_group(pspec_group, run_check=True)

    def _load_pspec(self, pspec_group, **kwargs):
        """
        Load a new UVPSpec object from a HDF5 group.

        Parameters
        ----------
        pspec_group : HDF5 group
            Group containing datasets that contain power spectrum and
            supporting information, in a standard format expected by UVPSpec.
        
        kwargs : dict, optional
            UVPSpec.read_from_group partial IO keyword arguments

        Returns
        -------
        uvp : UVPSpec
            Returns a UVPSpec object constructed from the input HDF5 group.
        """
        # Check that group is tagged as containing UVPSpec (pspec_type attribute)
        if 'pspec_type' in list(pspec_group.attrs.keys()):
            
            # Convert bytes -> str if needed
            pspec_type = pspec_group.attrs['pspec_type']
            if isinstance(pspec_type, bytes): pspec_type = pspec_type.decode()
            if pspec_type != uvpspec.UVPSpec.__name__:
                raise TypeError("HDF5 group is not tagged as a UVPSpec object.")
        else:
            raise TypeError("HDF5 group is not tagged as a UVPSpec object.")

        # Create new UVPSpec object and fill with data from this group
        uvp = uvpspec.UVPSpec()
        uvp.read_from_group(pspec_group, **kwargs)
        return uvp

    def _update_header(self):
        """
        Update the header in the HDF5 file with useful metadata, including the
        version of hera_pspec.
        """
        if 'header' in list(self.data.keys()):
            hdr = self.data['header']
        elif not self.swmr:
            hdr = self.data.create_group('header')
        else:
            raise ValueError("Cannot create a header group with SWMR")

        # Check if versions of hera_pspec are the same
        # with backward-compatibility if file generated 
        # before version control was introduced
        if 'hera_pspec.git_hash' in hdr.attrs or (
            'hera_pspec.version' in hdr.attrs and hdr.attrs['hera_pspec.version'] != __version__
        ):
            warnings.warn("HDF5 file was created by a different version of hera_pspec")

        if not self.swmr:
            hdr.attrs['hera_pspec.version'] = __version__
        
        
    @transactional
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

        if isinstance(group, (tuple, list, dict)):
            raise ValueError("Only one group can be specified at a time.")

        # Handle input arguments that are iterable (i.e. sequences, but not str)
        if isinstance(psname, list):
            if isinstance(pspec, list) and len(pspec) == len(psname):
                # Recursively call set_pspec() on each item of the list
                for _psname, _pspec in zip(psname, pspec):
                    if not isinstance(_pspec, uvpspec.UVPSpec):
                        raise TypeError("pspec lists must only contain UVPSpec "
                                        "objects.")
                    self.set_pspec(group, _psname, _pspec, overwrite=overwrite)
                return
            else:
                # Raise exception if psname is a list, but pspec is not
                raise ValueError("If psname is a list, pspec must be a list of "
                                 "the same length.")
        if isinstance(pspec, list) and not isinstance(psname, list):
            raise ValueError("If pspec is a list, psname must also be a list.")
        # No lists should pass beyond this point

        # check for swmr write of new group or dataset
        if self.swmr:
            tree = self.tree(return_str=False, nested=True)
            if group not in tree or psname not in tree[group]:
                raise ValueError("Cannot write new group or dataset with SWMR")

        # Check that input is of the correct type
        if not isinstance(pspec, uvpspec.UVPSpec):
            print("pspec:", type(pspec), pspec)
            raise TypeError("pspec must be a UVPSpec object.")

        key1 = "%s" % group
        key2 = "%s" % psname

        # Check that the group exists
        if key1 not in list(self.data.keys()):
            grp = self.data.create_group(key1)
        else:
            grp = self.data[key1]

        # Check that the psname exists
        if key2 not in list(grp.keys()):
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

    @transactional
    def get_pspec(self, group, psname=None, **kwargs):
        """
        Get a UVPSpec power spectrum object from a given group.

        Parameters
        ----------
        group : str
            Which group the power spectrum belongs to.

        psname : str, optional
            The name of the power spectrum to return. If None, extract all
            available power spectra.
        
        kwargs : dict
            UVPSpec.read_from_group partial IO keyword arguments

        Returns
        -------
        uvp : UVPSpec or list of UVPSpec
            The specified power spectrum as a UVPSpec object (or a list of all
            power spectra in the group, if psname was not specified).
        """
        # Check that group is in keys and extract it if so
        key1 = "%s" % group
        if key1 in list(self.data.keys()):
            grp = self.data[key1]
        else:
            raise KeyError("No group named '%s'" % key1)

        # If psname was specified, check that it exists and extract
        if psname is not None:
            key2 = "%s" % psname

            # Load power spectrum if it exists
            if key2 in list(grp.keys()):
                return self._load_pspec(grp[key2], **kwargs)
            else:
                raise KeyError("No pspec named '%s' in group '%s'" % (key2, key1))

        # Otherwise, extract all available power spectra
        uvp = []
        def pspec_filter(n, obj):
            if u'pspec_type' in list(obj.attrs.keys()):
                uvp.append(self._load_pspec(obj, **kwargs))

        # Traverse the entire set of groups/datasets looking for pspecs
        grp.visititems(pspec_filter) # This adds power spectra to the uvp list
        return uvp
    
    @transactional
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
        if key1 in list(self.data.keys()):
            grp = self.data[key1]
        else:
            raise KeyError("No group named '%s'" % key1)

        # Filter to look for pspec objects
        ps_list = []
        def pspec_filter(n, obj):
            if u'pspec_type' in list(obj.attrs.keys()):
                ps_list.append(n)

        # Traverse the entire set of groups/datasets looking for pspecs
        grp.visititems(pspec_filter)
        return ps_list

    @transactional
    def groups(self):
        """
        Return list of groups in the container.

        Returns
        -------
        group_list : list of str
            List of group names.
        """
        groups = list(self.data.keys())
        if u'header' in groups:
            groups.remove(u'header')
        return groups

    @transactional
    def tree(self, return_str=True):
        """
        Output a string containing a tree diagram of groups and the power
        spectra that they contain.

        Parameters
        ----------
        return_str : bool, optional
            If True, return the tree as a string, otherwise
            return as a dictionary

        Returns
        -------
        str or dict
            Tree structure of HDF5 file
        """
        grps = self.groups(nested=True)
        tree = dict()
        for grp in grps:
            tree[grp] = self.spectra(grp, nested=True)
        if not return_str:
            return tree
        else:
            s = ""
            for grp in grps:
                s += "(%s)\n" % grp
                for pspec in tree[grp]:
                    s += "  |--%s\n" % pspec
            return s
    
    @transactional
    def save(self):
        """
        Force HDF5 file to flush to disk.
        """
        self.data.flush()
    
    def __del__(self):
        """
        Make sure that HDF5 file is closed on destruct.
        """
        # Uses try-except construct just as a safeguard
        try:
            self.data.close()
        except:
            pass


def combine_psc_spectra(psc, groups=None, dset_split_str='_x_', ext_split_str='_',
                        merge_history=True, verbose=True, overwrite=False):
    """
    Iterate through a PSpecContainer and, within each specified group, combine 
    UVPSpec (i.e. spectra) of similar name but varying psname extension.

    Power spectra to-be-merged are assumed to follow the naming convention

    dset1_x_dset2_ext1, dset1_x_dset2_ext2, ...

    where _x_ is the default dset_split_str, and _ is the default ext_split_str.
    The spectra names are first split by dset_split_str, and then by 
    ext_split_str. In this particular case, all instances of dset1_x_dset2* 
    will be merged together.

    In order to merge spectra names with no dset distinction and only an 
    extension, feed dset_split_str as '' or None. Example, to merge together: 
    uvp_1, uvp_2, uvp_3, feed dset_split_str=None and ext_split_str='_'.

    Note this is a destructive and inplace operation, all of the *_ext1 objects 
    are removed after merge.

    Parameters
    ----------
    psc : PSpecContainer object
        A PSpecContainer object with one or more groups and spectra.

    groups : list
        A list of groupnames to operate on. Default is all groups.

    dset_split_str : str
        The pattern used to split dset1 from dset2 in the psname.

    ext_split_str : str
        The pattern used to split the dset name from its extension in the psname.

    merge_history : bool
        If True merge UVPSpec histories. Else use zeroth object's history.

    verbose : bool
        If True, report feedback to stdout.

    overwrite : bool
        If True, overwrite output spectra if they exist.
    """
    # Load container
    if isinstance(psc, str):
        psc = PSpecContainer(psc, mode='rw')
    else:
        assert isinstance(psc, PSpecContainer)
    
    # Get groups
    _groups = psc.groups()
    if groups is None:
        groups = _groups
    else:
        groups = [grp for grp in groups if grp in _groups]
    assert len(groups) > 0, "no specified groups exist in this Container object"

    # Iterate over groups
    for grp in groups:
        # Get spectra in this group
        spectra = list(psc.data[grp].keys())

        # Get unique spectra by splitting and then re-joining
        unique_spectra = []
        for spc in spectra:
            if dset_split_str == '' or dset_split_str is None:
                sp = spc.split(ext_split_str)[0]
            else:
                sp = utils.flatten([s.split(ext_split_str) 
                                    for s in spc.split(dset_split_str)])[:2]
                sp = dset_split_str.join(sp)
            if sp not in unique_spectra:
                unique_spectra.append(sp)

        # Iterate over each unique spectra, and merge all spectra extensions
        for spc in unique_spectra:
            # check for overwrite
            if spc in spectra and overwrite == False:
                if verbose:
                    print("spectra {}/{} already exists and overwrite == False, "
                          "skipping...".format(grp, spc))
                continue

            # get merge list
            to_merge = [spectra[i] for i in \
                                np.where([spc in _sp for _sp in spectra])[0]]
            try:
                # merge
                uvps = [psc.get_pspec(grp, uvp) for uvp in to_merge]
                if len(uvps) > 1:
                    merged_uvp = uvpspec.combine_uvpspec(uvps, merge_history=merge_history, verbose=verbose)
                else:
                    merged_uvp = uvps[0]
                # write to file
                psc.set_pspec(grp, spc, merged_uvp, overwrite=True)
                # if successful merge, remove uvps
                for uvp_name in to_merge:
                    if uvp_name != spc:
                        del psc.data[grp][uvp_name]
            except Exception as exc:
                # merge failed, so continue
                if verbose:
                    print("uvp merge failed for spectra {}/{}, exception: " \
                          "{}".format(grp, spc, exc))
    


def get_combine_psc_spectra_argparser():
    a = argparse.ArgumentParser(
        description="argument parser for hera_pspec.container.combine_psc_spectra")

    # Add list of arguments
    a.add_argument("filename", type=str,
                   help="Filename of HDF5 container (PSpecContainer) containing "
                        "groups / input power spectra.")
    a.add_argument("--dset_split_str", default='_x_', type=str, 
                   help='The pattern used to split dset1 from dset2 in the '
                        'psname.')
    a.add_argument("--ext_split_str", default='_', type=str, 
                   help='The pattern used to split the dset names from their '
                        'extension in the psname (if it exists).')
    a.add_argument("--verbose", default=False, action='store_true', 
                   help='Report feedback to stdout.')
    return a

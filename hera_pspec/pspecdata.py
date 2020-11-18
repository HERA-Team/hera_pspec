import numpy as np
from pyuvdata import UVData, UVCal, UVFlag
import copy, operator, itertools, sys
from collections import OrderedDict as odict
import hera_cal as hc
from pyuvdata import utils as uvutils
import datetime
import time
import argparse
import ast
import glob
import warnings
import json
import uvtools.dspec as dspec

# This is a list of weights that will depend on baseline.
# If data_weighting is in this list, G and H will be hashed per
# baseline pair.
R_PARAM_WEIGHTINGS = ['dayenu']

from . import uvpspec, utils, version, pspecbeam, conversions, container, uvpspec_utils as uvputils

class PSpecData(object):

    def __init__(self, dsets=[], wgts=None, dsets_std=None, labels=None,
                 beam=None, cals=None, cal_flag=True, external_flags=None,
                 cov_model='empirical'):
        """
        Object to store multiple sets of UVData visibilities and perform
        operations such as power spectrum estimation on them.

        Parameters
        ----------
        dsets : list or dict of UVData objects, optional
            Set of UVData objects containing the data that will be used to
            compute the power spectrum. If specified as a dict, the key names
            will be used to tag each dataset. Default: Empty list.

        dsets_std: list or dict of UVData objects, optional
            Set of UVData objects containing the standard deviations of each
            data point in UVData objects in dsets. If specified as a dict,
            the key names will be used to tag each dataset. Default: [].

        wgts : list or dict of UVData objects, optional
            Set of UVData objects containing weights for the input data.
            Default: None (will use the flags of each input UVData object).

        labels : list of str, optional
            An ordered list of names/labels for each dataset, if dsets was
            specified as a list. If None, names will not be assigned to the
            datasets. If dsets was specified as a dict, the keys
            of that dict will be used instead of this. Default: None.

        beam : PspecBeam object, optional
            PspecBeam object containing information about the primary beam
            Default: None.

        cals : list of UVCal objects, optional
            Calibration objects to apply to data. One per dset or
            one for all dsets.

        cal_flag : bool, optional
            If True, propagate flags from calibration into data

        r_cache : dict, optional
            Dictionary of pre-computed R-matrices with appropriate keys.

        cov_model : str, optional
            Model method for computing data covariances. Default is 'empirical'.
        """
        self.clear_cache(clear_r_params=True)  # clear matrix cache
        self.dsets = []; self.wgts = []; self.labels = []
        self.dsets_std = []
        self.Nfreqs = None
        self.spw_range = None
        self.spw_Nfreqs = None
        self.spw_Ndlys = None
        #Here are some cache dictionaries
        #Necesary for speeding up calculations
        # r_params is a dictionary that stores parameters for
        # parametric R matrices.
        self.filter_extension = (0, 0)
        self.cov_regularization = 0.
        # set data weighting to identity by default
        # and taper to none by default
        self.data_weighting = 'identity'
        self.taper = 'none'
        self.symmetric_taper = True
        # Set all weights to None if wgts=None
        if wgts is None:
            wgts = [None for dset in dsets]

        # set dsets_std to None if any are None.
        if not dsets_std is None and None in dsets_std:
            dsets_std = None

        # Store the input UVData objects if specified
        self._time_independent_weights = {}
        if len(dsets) > 0:
            self.add(dsets, wgts, dsets_std=dsets_std, labels=labels, cals=cals, cal_flag=cal_flag, external_flags=external_flags)

        # Store a primary beam
        self.primary_beam = beam
        self.cov_model = cov_model
        self.r_cov_model = cov_model

    def add(self, dsets, wgts, labels=None, dsets_std=None, cals=None, cal_flag=True, external_flags=None):
        """
        Add a dataset to the collection in this PSpecData object.

        Parameters
        ----------
        dsets : UVData or list or dict
            UVData object or list of UVData objects containing data to add to
            the collection.

        wgts : UVData or list or dict
            UVData object or list of UVData objects containing weights to add
            to the collection. Must be the same length as dsets. If a weight is
            set to None, the flags of the corresponding dset are used.

        labels : list of str
            An ordered list of names/labels for each dataset, if dsets was
            specified as a list. If dsets was specified as a dict, the keys
            of that dict will be used instead.

        dsets_std: UVData or list or dict
            Optional UVData object or list of UVData objects containing the
            standard deviations (real and imaginary) of data to add to the
            collection. If dsets is a dict, will assume dsets_std is a dict
            and if dsets is a list, will assume dsets_std is a list.

        cals : UVCal or list, optional
            UVCal objects to apply to data.

        cal_flag : bool, optional
            If True, propagate flags from calibration into data
        """
        # Check for dicts and unpack into an ordered list if found
        if isinstance(dsets, dict):
            # Disallow labels kwarg if a dict was passed
            if labels is not None:
                raise ValueError("If 'dsets' is a dict, 'labels' cannot be "
                                 "specified.")
            labels = list(dsets.keys())

            if wgts is None:
                wgts = dict([(l, None) for l in labels])
            elif not isinstance(wgts, dict):
                raise TypeError("If 'dsets' is a dict, 'wgts' must also be "
                                "a dict")

            if dsets_std is None:
                dsets_std = dict([(l, None) for l in labels])
            elif not isinstance(dsets_std, dict):
                raise TypeError("If 'dsets' is a dict, 'dsets_std' must also be "
                                "a dict")

            if cals is None:
                cals = dict([(l, None) for l in labels])
            elif not isinstance(cals, dict):
                raise TypeError("If 'cals' is a dict, 'cals' must also be "
                                "a dict")

            # Unpack dsets and wgts dicts
            dsets = [dsets[key] for key in labels]
            dsets_std = [dsets_std[key] for key in labels]
            wgts = [wgts[key] for key in labels]
            cals = [cals[key] for key in labels]

        # Convert input args to lists if possible
        if isinstance(dsets, UVData): dsets = [dsets,]
        if isinstance(wgts, UVData): wgts = [wgts,]
        if isinstance(labels, str): labels = [labels,]
        if isinstance(dsets_std, UVData): dsets_std = [dsets_std,]
        if isinstance(cals, UVCal): cals = [cals,]
        if isinstance(external_flags, UVFlag): external_flags = [external_flag for m in range(len(dsets))]
        if wgts is None: wgts = [wgts,]
        if dsets_std is None: dsets_std = [dsets_std for m in range(len(dsets))]
        if cals is None: cals = [cals for m in range(len(dsets))]
        if external_flags is None: external_flags = [external_flags for m in range(len(dsets))]
        if isinstance(dsets, tuple): dsets = list(dsets)
        if isinstance(wgts, tuple): wgts = list(wgts)
        if isinstance(dsets_std, tuple): dsets_std = list(dsets_std)
        if isinstance(cals, tuple): cals = list(cals)

        # Only allow UVData or lists
        if not isinstance(dsets, list) or not isinstance(wgts, list)\
        or not isinstance(dsets_std, list) or not isinstance(cals, list):
            raise TypeError("dsets, dsets_std, wgts and cals must be UVData"
                            "UVCal, or lists of UVData or UVCal")

        # Make sure enough weights were specified
        assert len(dsets) == len(wgts), \
            "The dsets and wgts lists must have equal length"
        assert len(dsets_std) == len(dsets), \
            "The dsets and dsets_std lists must have equal length"
        assert len(cals) == len(dsets), \
            "The dsets and cals lists must have equal length"
        assert len(external_flags) == len(dsets), \
            "The dsets and external flags lists must have equal length"
        if labels is not None:
            assert len(dsets) == len(labels), \
                "If labels are specified, the dsets and labels lists " \
                "must have equal length"

        # Check that everything is a UVData object
        for d, w, s in zip(dsets, wgts, dsets_std):
            if not isinstance(d, UVData):
                raise TypeError("Only UVData objects can be used as datasets.")
            if not isinstance(w, UVData) and w is not None:
                raise TypeError("Only UVData objects (or None) can be used as "
                                "weights.")
            if not isinstance(s, UVData) and s is not None:
                raise TypeError("Only UVData objects (or None) can be used as "
                                "error sets")
        for c in cals:
            if not isinstance(c, UVCal) and c is not None:
                raise TypeError("Only UVCal objects can be used for calibration.")

        # Store labels (if they were set)
        if self.labels is None:
            self.labels = []
        if labels is None:
            labels = ["dset{:d}".format(i)
                    for i in range(len(self.dsets), len(dsets) + len(self.dsets))]

        # Apply calibration if provided
        for dset, dset_std, cal, uvf in zip(dsets, dsets_std, cals, external_flags):
            if cal is not None:
                if dset is not None:
                    uvutils.uvcalibrate(dset, cal, inplace=True, prop_flags=cal_flag, flag_missing=cal_flag)
                    dset.extra_keywords['calibration'] = cal.extra_keywords.get('filename', '""')
                if dset_std is not None:
                    uvutils.uvcalibrate(dset_std, cal, inplace=True, prop_flags=cal_flag, flag_missing=cal_flag)
                    dset_std.extra_keywords['calibration'] = cal.extra_keywords.get('filename', '""')
            if uvf is not None:
                # apply external flags.
                if uvf.type == 'waterfall':
                    uvf.to_baseline(dset, force_pol=True)
                if dset is not None:
                    dset.data_array = dset.data_array | uvf.data_array
                if dset_std is not None:
                    dset_std.data_array = dset_std.data_array | uvf.data_array
                # collapse uvf again to prevent memory leak.
                uvf.to_waterfall(method='and', keep_pol=False)

        # Append to list
        self.dsets += dsets
        self.wgts += wgts
        self.dsets_std += dsets_std
        self.labels += labels

        # Check for repeated labels, and make them unique
        for i, l in enumerate(self.labels):
            ext = 1
            while ext < 1e5:
                if l in self.labels[:i]:
                    l = self.labels[i] + ".{:d}".format(ext)
                    ext += 1
                else:
                    self.labels[i] = l
                    break

        # Store no. frequencies and no. times
        self.Nfreqs = self.dsets[0].Nfreqs
        self.Ntimes = self.dsets[0].Ntimes

        # Store the actual frequencies
        self.freqs = self.dsets[0].freq_array[0]
        self.df = np.median(np.diff(self.freqs))
        self.spw_range = (0, self.Nfreqs)
        self.spw_Nfreqs = self.Nfreqs
        self.spw_Ndlys = self.spw_Nfreqs

        # clear time independent weights.
        self._time_independent_weights = {}
        self._unflagged_time_integration = {} # stores smallest unflagged time integration.
        # Check weights. If they are time independent, then set _time_independent_weights = True.
        for dind, dset in enumerate(self.dsets):
            bls_conj = []
            if len(dset.get_pols()) > 1:
                # hash antpairpol if there is more then one polarization
                # along with conjugates
                bls = dset.get_antpairpols()
                for bl in bls:
                    bls_conj.append(bl[:2][::-1] + (bl[-1][::-1],))
            else:
                # otherwise, hash antpair and antpairpol
                # along with their conjugates
                # this is done because other parts of hera_pspec play
                # fast and loose with hasing (sometimes don't explicitly hash pols)
                bls = dset.get_antpairs()
                for bl in bls:
                    bls_conj.append(bl[::-1])
            for bl, blc in zip(bls, bls_conj):
                if len(bl) == 3:
                    # only hash xx / yy strings
                    # since these are how the other matrices seem to be hashed.
                    bl = bl[:2] + (uvutils.polnum2str(uvutils.polstr2num(bl[-1], x_orientation=dset.x_orientation)),)
                    blc = blc[:2] + (bl[-1][::-1],)
                blkey = (dind, bl)
                blkeyc = (dind, blc)
                wwf = self.w(blkey).T
                template = None
                for rownum, wrow in enumerate(wwf):
                    if not np.all(np.isclose(wrow, 0.0)):
                        template = wrow
                        break
                if template is None:
                    template = np.zeros_like(wrow)
                self._time_independent_weights[blkey] = np.all([np.all(np.isclose(wrow, template)) or np.all(np.isclose(wrow, 0.0)) for wrow in wwf])
                self._time_independent_weights[blkeyc] = self._time_independent_weights[blkey]
                self._unflagged_time_integration[blkey] = rownum # smallest unflagged integration
                self._unflagged_time_integration[blkeyc] = self._unflagged_time_integration[blkey]
                if len(bl) == 2:
                    pol = uvutils.polnum2str(uvutils.polstr2num(dset.get_pols()[0], x_orientation=dset.x_orientation))
                    self._time_independent_weights[(dind, bl + (pol,))] = self._time_independent_weights[blkey]
                    self._unflagged_time_integration[(dind, bl + (pol,))] = self._unflagged_time_integration[blkey]
                    self._time_independent_weights[(dind, blc + (pol[::-1],))] = self._time_independent_weights[blkey]
                    self._unflagged_time_integration[(dind, blc + (pol[::-1],))] = self._unflagged_time_integration[blkey]


    def __str__(self):
        """
        Print basic info about this PSpecData object.
        """
        # Basic info
        s = "PSpecData object\n"
        s += "  %d datasets" % len(self.dsets)
        if len(self.dsets) == 0: return s

        # Dataset summary
        for i, d in enumerate(self.dsets):
            if self.labels[i] is None:
                s += "  dset (%d): %d bls (freqs=%d, times=%d, pols=%d)\n" \
                      % (i, d.Nbls, d.Nfreqs, d.Ntimes, d.Npols)
            else:
                s += "  dset '%s' (%d): %d bls (freqs=%d, times=%d, pols=%d)\n" \
                      % (self.labels[i], i, d.Nbls, d.Nfreqs, d.Ntimes, d.Npols)
        return s

    def validate_datasets(self, verbose=True):
        """
        Validate stored datasets and weights to make sure they are consistent
        with one another (e.g. have the same shape, baselines etc.).
        """
        # check dsets and wgts have same number of elements
        if len(self.dsets) != len(self.wgts):
            raise ValueError("self.wgts does not have same len as self.dsets")

        if len(self.dsets_std) != len(self.dsets):
            raise ValueError("self.dsets_std does not have the same len as "
                             "self.dsets")
        if len(self.labels) != len(self.dsets):
            raise ValueError("self.labels does not have same len as self.dsets")

        # Check if dsets are all the same shape along freq axis
        Nfreqs = [d.Nfreqs for d in self.dsets]
        channel_widths = [d.channel_width for d in self.dsets]
        if np.unique(Nfreqs).size > 1:
            raise ValueError("all dsets must have the same Nfreqs")
        if np.unique(channel_widths).size > 1:
            raise ValueError("all dsets must have the same channel_widths")

        # Check shape along time axis
        Ntimes = [d.Ntimes for d in self.dsets]
        if np.unique(Ntimes).size > 1:
            raise ValueError("all dsets must have the same Ntimes")

        # raise warnings if times don't match
        if len(self.dsets) > 1:
            lst_diffs = np.array( [ np.unique(self.dsets[0].lst_array)
                                  - np.unique(dset.lst_array)
                                   for dset in self.dsets[1:]] )
            if np.max(np.abs(lst_diffs)) > 0.001:
                raise_warning("Warning: LST bins in dsets misaligned by more than 15 seconds",
                              verbose=verbose)

            # raise warning if frequencies don't match
            freq_diffs = np.array( [ np.unique(self.dsets[0].freq_array)
                                   - np.unique(dset.freq_array)
                                    for dset in self.dsets[1:]] )
            if np.max(np.abs(freq_diffs)) > 0.001e6:
                raise_warning("Warning: frequency bins in dsets misaligned by more than 0.001 MHz",
                              verbose=verbose)

        # Check phase type
        phase_types = []
        for d in self.dsets: phase_types.append(d.phase_type)
        if np.unique(phase_types).size > 1:
            raise ValueError("all datasets must have the same phase type "
                             "(i.e. 'drift', 'phased', ...)\ncurrent phase "
                             "types are {}".format(phase_types))

        # Check phase centers if phase type is phased
        if 'phased' in set(phase_types):
            phase_ra = [d.phase_center_ra_degrees for d in self.dsets]
            phase_dec = [d.phase_center_dec_degrees for d in self.dsets]
            max_diff_ra = np.max( [np.diff(d)
                                   for d in itertools.combinations(phase_ra, 2)])
            max_diff_dec = np.max([np.diff(d)
                                  for d in itertools.combinations(phase_dec, 2)])
            max_diff = np.sqrt(max_diff_ra**2 + max_diff_dec**2)
            if max_diff > 0.15:
                raise_warning("Warning: maximum phase-center difference "
                              "between datasets is > 10 arcmin",
                              verbose=verbose)

    def check_key_in_dset(self, key, dset_ind):
        """
        Check 'key' exists in the UVData object self.dsets[dset_ind]

        Parameters
        ----------
        key : tuple
            if length 1: assumed to be polarization number or string
            elif length 2: assumed to be antenna-number tuple (ant1, ant2)
            elif length 3: assumed ot be antenna-number-polarization tuple (ant1, ant2, pol)

        dset_ind : int, the index of the dataset to-be-checked

        Returns
        -------
        exists : bool
            True if the key exists, False otherwise
        """
        #FIXME: Fix this to enable label keys
        # get iterable
        key = uvutils._get_iterable(key)
        if isinstance(key, str):
            key = (key,)

        # check key is a tuple
        if isinstance(key, tuple) == False or len(key) not in (1, 2, 3):
            raise KeyError("key {} must be a length 1, 2 or 3 tuple".format(key))

        try:
            _ = self.dsets[dset_ind]._key2inds(key)
            return True
        except KeyError:
            return False

    def clear_cache(self, clear_r_params=False, keys=None):
        """
        Clear stored matrix data (or some subset of it).

        Parameters
        ----------
        clear_r_params : bool, optional
            If True, clear r_parameters in cache. If False, don't clear them.

        keys : list of tuples, optional
            List of keys to remove from matrix cache. If None, all
            keys will be removed. Default: None.
        """
        if clear_r_params:
            self.r_params = {}
        if keys is None:
            self._C, self._I, self._iC, self._Y, self._R = {}, {}, {}, {}, {}
            self._H, self._G = {}, {}
            self._W, self._M = {}, {}
            self._E, self._V = {}, {}
        else:
            for k in keys:
                try: del(self._C[k])
                except(KeyError): pass
                try: del(self._I[k])
                except(KeyError): pass
                try: del(self._iC[k])
                except(KeyError): pass
                try: del(self.r_params[k])
                except(KeyError): pass
                try: del(self._Y[k])
                except(KeyError): pass
                try: del(self._R[k])
                except(KeyError): pass

    def dset_idx(self, dset):
        """
        Return the index of a dataset, regardless of whether it was specified
        as an integer of a string.

        Parameters
        ----------
        dset : int or str
            Index or name of a dataset belonging to this PSpecData object.

        Returns
        -------
        dset_idx : int
            Index of dataset.
        """
        # Look up dset label if it's a string
        if isinstance(dset, str):
            if dset in self.labels:
                return self.labels.index(dset)
            else:
                raise KeyError("dset '%s' not found." % dset)
        elif isinstance(dset, (int, np.integer)):
            return dset
        else:
            raise TypeError("dset must be either an int or string")

    def r_param_hash(self, rpkey):
        """Function for hashing parameters for parameteric R-matrices.
        Parameters
        ----------
        rpkey : tuple
            tuple of r-parameters dictionary entry that we wish to hash.

        Returns
        -------
            hashable tuple based on values of r-parameters.
        """
        if rpkey in self.r_params:
            if self.data_weighting == 'dayenu':
                rp = self.r_params[rpkey]
                key = tuple((np.asarray(rp['filter_centers']) * self.df * self.spw_Nfreqs).astype(int)) + \
                      tuple((np.asarray(rp['filter_half_widths']) * self.df * self.spw_Nfreqs).astype(int)) + \
                      tuple((np.asarray(rp['filter_factors']) * 1e9).astype(int))
            else:
                raise ValueError("only 'dayenu' currently supported for r_param hashing.")
        else:
            raise KeyError("%s not in self.r_params"%str(rpkey))
        return key



    def weighting_hash(self, time_index, key1, key2=None):
        """hash baseline-times based on weights and flagging patterns.

        Function for hashing matrices for a particular time integration
        based on their flagging patterns and any additional necessary information
        that depends on the weighting being used.

        Currently, identity weights are only hashed on the flagging pattern of
        key1 and (optional) key2

        parametric weights are hashed on flagging patterns of key1 and (optional)
        key 2 and the r_params of key1 / key2. Since different baselines can have
        identical r_params, the number of unique hashes is less then the number
        of baseline (pairs).

        inverse covariance weights are hashed on flagging pattern and baseline pair
        if empirical. This means that hte number of unique hashes is equal to the
        number of baseline (pairs).

        Since we use binary weights, we hash as bools for efficient storage use.

        Parameters
        ----------

        time_index : int, integer specifying the time-index to be hashed.

        key1 : 3 or 4-tuple, first index is dset index, second/third indices are
               antenna indices of baseline pair. Fourth index is polarization.

        key2 : 3 or 4-tuple (optional) see key1 description. This optional arg
               should be provided when hashing baseline pairs rather then a
               single baseline.

        Returns
        -------
            key : tuple whose items depend on the time index, baseline keys and
            data_weighting being used.

        """
        # for identity weights, only hash by flagging patterns.
        if self.data_weighting == 'identity':
            key = tuple(self.Y(key1, time_index).astype(bool))
            if key2 is not None:
                key = key + tuple(self.Y(key2, time_index).astype(bool))
        elif self.data_weighting in R_PARAM_WEIGHTINGS:
            # for parametric weights, hash on r_params and flagging pattern.
            r_param_key = (self.data_weighting, ) + key1
            key = self.r_param_hash(r_param_key) + tuple(self.Y(key1, time_index).astype(bool))
            if key2 is not None:
                r_param_key = (self.data_weighting, ) + key2
                key += self.r_param_hash(r_param_key) + tuple(self.Y(key2, time_index).astype(bool))
        elif self.data_weighting == 'iC':
            # empirical covariance is hashed by baseline pair but not
            # by time since the covariance is computed across time.
            if self.r_cov_model in ['empirical', 'autos']:
                    key = key1
                    if key2 is not None:
                        key += key2
            # for dset covariances, the time_index must be hashed.
            elif self.r_cov_model == 'dsets':
                    key = key1 + (time_index, )
                    if key2 is not None:
                        key += key2
            else:
                raise ValueError("Invalid cov model. Cannot hash.")

        else:
            raise ValueError("Invalid data weighting. Cannot hash.")
        return key + (self.spw_Nfreqs, self.symmetric_taper, self.filter_extension, self.taper, self.data_weighting)

    def fisher_hash(self, time_index, key1, key2, sampling, exact_norm, pol):
        """hashing function for fisher matrices.

        Parameters
        ----------
        time_index: int
            index of time to hash.

        key1 : 3 or 4-tuple, first index is dset index, second/third indices are
               antenna indices of baseline pair. Fourth index is polarization.

        key2 : 3 or 4-tuple (optional) see key1 description.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        Returns
        -------
        tuple for hashing normalization level matrices (that involve baseline-pairs)
        keys may or may not actually be used in hashing (see docstring for weighting_hash)
        """
        return self.weighting_hash(time_index, key1, key2)\
         + (sampling, exact_norm, pol, self.spw_Ndlys)

    def cov_hash(self, time_index, key1, key2, exact_norm, pol, model):
        """hashing function for covariances of q-hat.

        Parameters
        ----------
        time_index: int
            index of time to hash
        key1 : 3 or 4-tuple, first index is dset index, second/third indices are
               antenna indices of baseline pair. Fourth index is polarization.

        key2 : 3 or 4-tuple (optional) see key1 description.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        model : str, default: 'empirical'
            How the covariances of the input data should be estimated.
            in 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations. If 'empirical' is
            set, then error bars are estimated from the data by calculating the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
             frequency-domain covariances.

        Returns
        -------
        tuple for hashing covariances of q-hat.
        """
        key = self.fisher_hash(time_index, key1, key2, sampling=False, exact_norm=exact_norm, pol=pol)
        # if covariance is empirical, then must be hashed on
        # baseline pair but does not necessarily include time-index.
        if model in ['empirical', 'autos']:
            key = key + key1 + key2
        # if dsets are being used, then must hash on time index as well.
        elif model == 'dsets':
            key = key + key1 + key2 + (time_index, )
        key = key + (model, )
        return key



    def parse_blkey(self, key):
        """
        Parse a dataset + baseline key in the form (dataset, baseline, [pol])
        into a dataset index and a baseline(pol) key, where pol is optional.

        Parameters
        ----------
        key : tuple

        Returns
        -------
        dset : int
            dataset index

        bl : tuple
            baseline (pol) key
        """
        # type check
        assert isinstance(key, tuple), "key must be fed as a tuple"
        assert len(key) > 1, "key must have len >= 2"

        # get dataset index
        dset_idx = self.dset_idx(key[0])
        key = key[1:]

        # get baseline
        bl = key[0]
        if isinstance(bl, (int, np.integer)):
            assert len(key) > 1, "baseline must be fed as a tuple"
            bl = tuple(key[:2])
            key = key[2:]
        else:
            key = key[1:]
        assert isinstance(bl, tuple), "baseline must be fed as a tuple, %s" % bl

        # put pol into bl key if it exists
        if len(key) > 0:
            pol = key[0]
            assert isinstance(pol, (str, int, np.integer)), \
                "pol must be fed as a str or int"
            bl += (key[0],)

        return dset_idx, bl

    def x(self, key, include_extension=False):
        """
        Get data for a given dataset and baseline, as specified in a standard
        key format.

        Parameters
        ----------
        key : tuple
            Tuple containing dataset ID and baseline index. The first element
            of the tuple is the dataset index (or label), and the subsequent
            elements are the baseline ID.

        include_extension : bool (optional)
            default=False
            If True, extend spw to include filtering window extensions.

        Returns
        -------
        x : array_like
            Array of data from the requested UVData dataset and baseline.
        """
        dset, bl = self.parse_blkey(key)
        spw = slice(*self.get_spw(include_extension=include_extension))
        return self.dsets[dset].get_data(bl).T[spw]

    def dx(self, key, include_extension=False):
        """
        Get standard deviation of data for given dataset and baseline as
        pecified in standard key format.

        Parameters
        ----------
        key : tuple
            Tuple containing datset ID and baseline index. The first element
            of the tuple is the dataset index (or label), and the subsequent
            elements are the baseline ID.

        include_extension : bool (optional)
            default=False
            If True, extend spw to include filtering window extensions.

        Returns
        -------
        dx : array_like
            Array of std data from the requested UVData dataset and baseline.
        """
        assert isinstance(key, tuple)
        dset,bl = self.parse_blkey(key)
        spw = slice(*self.get_spw(include_extension=include_extension))
        return self.dsets_std[dset].get_data(bl).T[spw]

    def w(self, key, include_extension=False):
        """
        Get weights for a given dataset and baseline, as specified in a
        standard key format.

        Parameters
        ----------
        key : tuple
            Tuple containing dataset ID and baseline index. The first element
            of the tuple is the dataset index, and the subsequent elements are
            the baseline ID.

        include_extension : bool (optional)
            default=False
            If True, extend spw to include filtering window extensions.

        Returns
        -------
        w : array_like
            Array of weights for the requested UVData dataset and baseline.
        """
        dset, bl = self.parse_blkey(key)
        spw = slice(*self.get_spw(include_extension=include_extension))
        if self.wgts[dset] is not None:
            wgts = self.wgts[dset].get_data(bl).T[spw]
        else:
            # If weights were not specified, use the flags built in to the
            # UVData dataset object
            wgts = (~self.dsets[dset].get_flags(bl)).astype(float).T[spw]
        return wgts

    def set_C(self, cov):
        """
        Set the cached covariance matrix to a set of user-provided values.

        This method also resets cached values that depend on the covariances
        being reset.

        Parameters
        ----------
        cov : dict
            Covariance keys and ndarrays.
            The key should conform to
            (dset_pair_index, blpair_int, model, time_index, conj_1, conj_2).
            e.g. ((0, 1), ((25,37,"xx"), (25, 37, "xx")), 'empirical', False, True)
            while the ndarrays should have shape (spw_Nfreqs, spw_Nfreqs)
        """
        for key in cov:
            #clear_cache() was originally being called here
            #this was deleting the _V cache and since get_unnormed_V
            #calls this function, this was causing caching to fail.
            #However, if we expand the use of set_C outside of
            #get_unnormed_V and allow for arbitrary covariances to
            #be set, then we could run into trouble
            #since _V, _iC and other pre-cached quanties depend on _C.
            self._C[key] = cov[key]

    def get_spw(self, include_extension=False):
        """
        Get self.spw_range with or without spw extension (self.filter_extension)

        Parameters
        ----------
        include_extension : bool
            If True, include self.filter_extension in spw_range

        Returns
        -------
        spectral_window : tuple
            In (start_chan, end_chan).
        """
        if sum(self.filter_extension) > 0:
            include_extension = True
        # if there is non-zero self.filter_extension, include_extension is automatically set to be True
        if include_extension:
            return (self.spw_range[0] - self.filter_extension[0], self.spw_range[1] + self.filter_extension[1])
        else:
            return self.spw_range

    def C_model(self, key, model='empirical', time_index=None, known_cov=None, include_extension=False):
        """
        Return a covariance model having specified a key and model type.
        Note: Time-dependent flags that differ from frequency channel-to-channel
        can create spurious spectral structure. Consider factorizing the flags with
        self.broadcast_dset_flags() before using model='empirical'.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos',
            (other model names in known_cov)]
            How the covariances of the input data should be estimated.
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations.
            If 'empirical' is set, then error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            If 'autos' is set, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas over channel bandwidth
            and integration time.

        time_index : integer, compute covariance at specific time-step in dset
            supported if mode == 'dsets' or 'autos'

        known_cov : dicts of covariance matrices
            Covariance matrices that are imported from a outer dict instead of
            using data stored or calculated inside the PSpecData object.
            known_cov could be initialized when using PSpecData.pspec() method.
            See PSpecData.pspec() for more details.

        include_extension : bool (optional)
            default=False
            If True, extend spw to include filtering window extensions.

        Returns
        -------
        C : ndarray, (spw_Nfreqs, spw_Nfreqs)
            Covariance model for the specified key.
        """
        # type check
        assert isinstance(key, tuple), "key must be fed as a tuple"
        assert isinstance(model, (str, np.str)), "model must be a string"

        # parse key
        dset, bl = self.parse_blkey(key)
        if model == 'empirical':
            # add model to key
            Ckey = ((dset, dset), (bl,bl), ) + (model, None, False, True,)
        else:
            assert isinstance(time_index, int), "time_index must be integer if cov-model=={}".format(model)
            # add model to key
            Ckey = ((dset, dset), (bl,bl), ) + (model, time_index, False, True,)

        # Check if Ckey exists in known_cov. If so, just update self._C[Ckey] with known_cov.
        if known_cov is not None:
            if Ckey in known_cov.keys():
                spw = slice(*self.get_spw(include_extension=include_extension))
                covariance = known_cov[Ckey][spw, spw]
                self.set_C({Ckey: covariance})

        # check cache
        if Ckey not in self._C:
            # calculate covariance model
            if model == 'empirical':
                self.set_C({Ckey: utils.cov(self.x(key, include_extension=include_extension), self.w(key, include_extension=include_extension))})
            elif model == 'dsets':
                self.set_C({Ckey: np.diag( np.abs(self.w(key, include_extension=include_extension)[:,time_index] * self.dx(key, include_extension=include_extension)[:,time_index]) ** 2. )})
            elif model == 'autos':
                spw_range = self.get_spw(include_extension=include_extension)
                self.set_C({Ckey: np.diag(utils.variance_from_auto_correlations(self.dsets[dset], bl, spw_range, time_index))})
            else:
                raise ValueError("didn't recognize Ckey {}".format(Ckey))

        return self._C[Ckey]

    def cross_covar_model(self, key1, key2, model='empirical',
                          time_index=None, conj_1=False, conj_2=True, known_cov=None, include_extension=False):
        """
        Return a covariance model having specified a key and model type.
        Note: Time-dependent flags that differ from frequency channel-to-channel
        can create spurious spectral structure. Consider factorizing the flags
        with self.broadcast_dset_flags() before using model='time_average'.

        Parameters
        ----------
        key1, key2 : tuples
            Tuples containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos',
            (other model names in known_cov)]
            How the covariances of the input data should be estimated.
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations.
            If 'empirical' is set, then error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            If 'autos' is set, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas over channel bandwidth
            and integration time.

        time_index : integer, compute covariance at specific time-step

        conj_1 : boolean, optional
            Whether to conjugate first copy of data in covar or not.
            Default: False

        conj_2 : boolean, optional
            Whether to conjugate second copy of data in covar or not.
            Default: True

        known_cov : dicts of covariance matrices
            Covariance matrices that are imported from a outer dict instead of
            using data stored or calculated inside the PSpecData object.
            known_cov could be initialized when using PSpecData.pspec() method.
            See PSpecData.pspec() for more details.

        include_extension : bool (optional)
            default=False
            If True, extend spw to include filtering window extensions.

        Returns
        -------
        cross_covar : ndarray, (spw_Nfreqs, spw_Nfreqs)
            Cross covariance model for the specified key.
        """
        # type check
        assert isinstance(key1, tuple), "key1 must be fed as a tuple"
        assert isinstance(key2, tuple), "key2 must be fed as a tuple"
        assert isinstance(model, (str, np.str)), "model must be a string"

        # parse key
        dset1, bl1 = self.parse_blkey(key1)
        dset2, bl2 = self.parse_blkey(key2)
        covar = None

        if model == 'empirical':
            covar = utils.cov(self.x(key1, include_extension=include_extension), self.w(key1, include_extension=include_extension),
                              self.x(key2, include_extension=include_extension), self.w(key2, include_extension=include_extension),
                              conj_1=conj_1, conj_2=conj_2)
        if model in ['dsets','autos']:
            covar = np.zeros((np.diff(self.get_spw(include_extension=include_extension))[0],
                np.diff(self.get_spw(include_extension=include_extension))[0]), dtype=np.float64)
        # Check if model exists in known_cov. If so, just overwrite covar with known_cov.
        if known_cov is not None:
            Ckey = ((dset1, dset2), (bl1,bl2), ) + (model, time_index, conj_1, conj_2,)
            if Ckey in known_cov.keys():
                spw = slice(*self.get_spw(include_extension=include_extension))
                covar = known_cov[Ckey][spw, spw]

        if covar is None:
            raise ValueError("didn't recognize model {}".format(model))

        return covar

    def I(self, key):
        """
        Return identity covariance matrix.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        Returns
        -------
        I : array_like
            Identity covariance matrix, dimension (Nfreqs, Nfreqs).
        """
        assert isinstance(key, tuple)
        # parse key
        dset, bl = self.parse_blkey(key)
        key = (dset,) + (bl,)
        nfreq = self.spw_Nfreqs + np.sum(self.filter_extension)
        if key not in self._I:
            self._I[key] = np.identity(nfreq)
        return self._I[key]

    def iC(self, key, model='empirical', time_index=None):
        """
        Return the inverse covariance matrix, C^-1.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos']
            How the covariances of the input data should be estimated.
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations.
            If 'empirical' is set, then error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            If 'autos' is set, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas over channel bandwidth
            and integration time.

        time_index : integer or list
            time indices to obtain iC for.

        Returns
        -------
        iC : array_like
            Inverse covariance matrix for specified dataset and baseline.
            This will be an Ntimes x Nfreq x Nfreq
        """
        assert isinstance(key, tuple)
        # parse key
        dset, bl = self.parse_blkey(key)
        key = (dset,) + (bl,)
        tindep = self._time_independent_weights[key]
        Ckey = ((dset, dset), (bl,bl), ) + (model, time_index, False, True, tindep)
        nfreq = self.spw_Nfreqs + np.sum(self.filter_extension)
        # Calculate inverse covariance if not in cache
        if Ckey not in self._iC:
            #Note that C by default gives us an Ndly x Ndly matrix (so it is per time)
            #In the following lines, we calculate the inverse or psuedo-inverse
            #of C multiplied by weights.
            #empirically determined C is not per-time but weights are.
            #Thus, we need to loop through each set of per-time weights and take
            #a psuedo-inverse at each time.
            wgts = self.Y(key, time_index)
            _iC = np.zeros(nfreq, nfreq)
            wgts_sq = np.outer(wgts, wgts)
            C = self.C_model(key, model=model, time_index=time_index)
            _iC = wgts_sq * C
            if np.linalg.cond(_iC) >= 1e9:
                warnings.warn("Poorly conditioned covariance. Computing Psuedo-Inverse")
                _iC = np.linalg.pinv(_iC)
            else:
                _iC = np.linalg.inv(_iC)
            self.set_iC({Ckey:_iC})
        return self._iC[Ckey][time_index]

    def Y(self, key, time_index=None):
        """
        Return the weighting (diagonal) matrix, Y. This matrix
        is calculated by taking the logical AND of flags across all times
        given the dset-baseline-pol specification in 'key', converted
        into a float, and inserted along the diagonal of an
        spw_Nfreqs x spw_Nfreqs matrix.

        The logical AND step implies that all time-dependent flagging
        patterns are automatically broadcasted across all times. This broadcasting
        follows the principle that, for each freq channel, if at least a single time
        is unflagged, then the channel is treated as unflagged for all times. Power
        spectra from certain times, however, can be given zero weight by setting the
        nsample array to be zero at those times (see self.broadcast_dset_flags).

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.
        time_index : integer
            time index to retrieve. If None, returns 2d array of all times.
        Returns
        -------
        Y : array_like
        If time_index is None
            (spw_Nfreqs + sum(filter_extension)) x Ntimes matrix of weights.
        Otherwise
            spw_Nfreqs
        """
        assert isinstance(key, tuple)
        # parse key
        dset, bl = self.parse_blkey(key)
        key = (dset,) + (bl,)

        if key not in self._Y:
            self._Y[key] = self.w(key, include_extension=True)
            if not np.all(np.isclose(self._Y[key], 0.0) \
                        + np.isclose(self._Y[key], 1.0)):
                raise NotImplementedError("Non-binary weights not currently implmented")

        if time_index is None:
            return self._Y[key]
        else:
            return self._Y[key][:, time_index].squeeze()

    def set_iC(self, d):
        """
        Set the cached inverse covariance matrix for a given dataset and
        baseline to a specified value. For now, you should already have applied
        weights to this matrix.

        Parameters
        ----------
        d : dict
            Dictionary containing data to insert into inverse covariance matrix
            cache. Keys are tuples, following the same format as the input to
            self.iC().
        """
        for k in d:
            self._iC[k] = d[k]

    def set_R(self, d):
        """
        Set the data-weighting matrix for a given dataset and baseline to
        a specified value for later use in q_hat.

        Parameters
        ----------
        d : dict
            Dictionary containing data to insert into data-weighting R matrix
            cache. Keys are tuples with the following form
            key = (dset_index, bl_ant_pair_pol_tuple, data_weighting, taper)
            Ex: (0, (37, 38, 'xx'), 'bh')
            If data_weight == 'dayenu' then additional elements are appended
                key + (filter_extension, spw_Nfreqs, symmetric_taper)
        """
        for k in d:
            self._R[k] = d[k]

    def R(self, key, time_indices=None):
        """
        Return the data-weighting matrix R, which is a product of
        data covariance matrix (I or C^-1), diagonal flag matrix (Y) and
        diagonal tapering matrix (T):

        If self.symmetric_taper == False:
            R = T K Y
        if self.symmetric_taper == True:
            R = sqrt(T^t) sqrt(Y^t) K sqrt(Y) sqrt(T)

        where T is a diagonal matrix holding the taper and Y is a diagonal
        matrix holding flag weights. The K matrix comes from either `I` or `iC`
        or a `dayenu`
        depending on self.data_weighting, T is informed by self.taper and Y
        is taken from self.Y().

        Right now, the data covariance can be identity ('I'), C^-1 ('iC'), or
        dayenu weighting 'dayenu'.

        Parameters
        ----------
        key : tuple
            Tuple containing indices of dataset and baselines. The first item
            specifies the index (ID) of a dataset in the collection, while
            subsequent indices specify the baseline index, in _key2inds format.

        time_indices : integer or list of integers

        Returns
            If weights for key are time-independent, returns an spw_Nfreqs x spw_Nfreqs array.
            Otherwise, return an Ntimes x spw_Nfreqs x spw_Nfreqs array.
        """
        if isinstance(time_indices, (int, np.integer)):
            time_indices = [time_indices]
        # type checks
        if time_indices is None:
            time_indices = np.arange(self.Ntimes).astype(int)
        assert isinstance(key, tuple)
        tindep = self._time_independent_weights[self.parse_blkey(key)]
        fext = self.filter_extension
        nfreq = np.sum(fext) + self.spw_Nfreqs
        # Only add to Rkey if a particular mode is enabled
        # If you do add to this, you need to specify this in self.set_R docstring!
        output = np.zeros((len(time_indices), self.spw_Nfreqs, nfreq), dtype=np.complex128)
        for mindex, tindex in enumerate(time_indices):
            wgt = self.Y(key, tindex)
            wgt_sq = np.outer(wgt, wgt)
            Rkey = self.weighting_hash(tindex, key)
            if Rkey not in self._R:
                if self.data_weighting == 'identity':
                        rmat = self.I(key) * wgt_sq
                elif self.data_weighting == 'dayenu':
                        r_param_key = (self.data_weighting,) + key
                        r_params = self.r_params[r_param_key]
                        _freqs = self.freqs[self.spw_range[0]-fext[0]:self.spw_range[1]+fext[1]]
                        rmat = dspec.dayenu_mat_inv(x=_freqs,
                                        filter_centers=r_params['filter_centers'],
                                        filter_half_widths=r_params['filter_half_widths'],
                                        filter_factors=r_params['filter_factors']) * wgt_sq
                        rmat = np.linalg.pinv(rmat)
                elif self.data_weighting == 'iC':
                    # model for R covariance is specified in r_params
                    # rather then self.cov_model which sets how error bars
                    # are computed.
                    rmat = self.iC(key, model=self.r_cov_model, time_index=tindex)
                else:
                    raise ValueError("data_weighting must be in ['identity', 'iC', 'dayenu']")

                myTaper = dspec.gen_window(self.taper, self.spw_Nfreqs)
                tmat = np.zeros((self.spw_Nfreqs, nfreq), dtype=complex)
                tmat[:,fext[0]:fext[0] + self.spw_Nfreqs] = np.identity(self.spw_Nfreqs,dtype=np.complex128)

                rmat = tmat @ rmat
                rmat = rmat.T
                if self.symmetric_taper:
                    sqrtT = np.sqrt(myTaper)
                    rmat = np.transpose(sqrtT[None, :] * rmat * sqrtT[None, :])
                else:
                    rmat = np.transpose(myTaper[None, :] * rmat)
                self._R[Rkey] = rmat

            output[mindex] = self._R[Rkey]

        return output.squeeze()

    def set_symmetric_taper(self, use_symmetric_taper):
        """
        Set the symmetric taper parameter
        If true, square R matrix will be computed as
        R=sqrtT  K sqrt T
        where sqrtT is a diagonal matrix with the square root of the taper.
        This is only possible when K is a square-matrix (no filter extensions).

        If set to false, then the R-matrix will implement the taper as
        R = sqrtT ** 2 K
        Parameters
        ----------
        use_taper : bool,
            do you want to use a symmetric taper? True or False?
        """
        if use_symmetric_taper and (self.filter_extension[0] > 0 or self.filter_extension[1] > 0):
            raise ValueError("You cannot use a symmetric taper when there are nonzero filter extensions.")
        else:
            self.symmetric_taper = use_symmetric_taper



    def set_filter_extension(self, filter_extension):
        """
        Set extensions to filtering matrix

        Parameters
        ----------
        filter_extension: 2-tuple or 2-list
            must be integers. Specify how many channels below spw_min/max
            filter will be applied to data.
            filter_extensions will be clipped to not extend beyond data range.
        """
        if self.symmetric_taper and not filter_extension[0] == 0 and not filter_extension[1]==0:
            raise_warning("You cannot set filter extensions greater then zero when symmetric_taper==True! Setting symmetric_taper==False!")
            self.symmetric_taper = False
        assert isinstance(filter_extension, (list, tuple)), "filter_extension must a tuple or list"
        assert len(filter_extension) == 2, "filter extension must be length 2"
        assert isinstance(filter_extension[0], (int, np.integer)) and\
               isinstance(filter_extension[1], (int, np.integer)) and \
               filter_extension[0] >= 0 and\
               filter_extension[1] >=0, "filter extension must contain only positive integers"
        filter_extension=list(filter_extension)
        if filter_extension[0] > self.spw_range[0]:
            warnings.warn("filter_extension[0] exceeds data spw_range. Defaulting to spw_range[0]!")
        if filter_extension[1] > self.Nfreqs - self.spw_range[1]:
            warnings.warn("filter_extension[1] exceeds channels between spw_range[1] and Nfreqs. Defaulting to Nfreqs-spw_range[1]!")
        filter_extension[0] = np.min([self.spw_range[0], filter_extension[0]])#clip extension to not extend beyond data range
        filter_extension[1] = np.min([self.Nfreqs - self.spw_range[1], filter_extension[1]])#clip extension to not extend beyond data range
        self.filter_extension = tuple(filter_extension)

    def set_weighting(self, data_weighting, r_cov_model='empirical'):
        """
        Set data weighting type.

        Parameters
        ----------
        data_weighting : str
            Type of data weightings. Options=['identity', 'iC', dayenu]
        cov_model : str, optional
            Specify method for computing covariance for iC weighting.
            Default is 'empirical'.
        """
        self.data_weighting = data_weighting
        if data_weighting == 'iC':
            self.r_cov_model = r_cov_model
        else:
            self.r_cov_model = self.cov_model

    def set_r_param(self, key, r_params):
        """
        Set the weighting parameters for baseline at (dset,bl, [pol])

        Parameters
        ----------
        key: tuple (dset, bl, [pol]), where dset is the index of the dataset
             bl is a 2-tuple
             pol is a float or string specifying polarization

        r_params: dictionary with parameters for weighting matrix.
                  Proper fields
                  and formats depend on the mode of data_weighting.
                data_weighting == 'dayenu':
                                dictionary with fields
                                'filter_centers', list of floats (or float) specifying the (delay) channel numbers
                                                  at which to center filtering windows. Can specify fractional channel number.
                                'filter_half_widths', list of floats (or float) specifying the width of each
                                                 filter window in (delay) channel numbers. Can specify fractional channel number.
                                'filter_factors', list of floats (or float) specifying how much power within each filter window
                                                  is to be suppressed.
                Absence of r_params dictionary will result in an error!
        """
        key = (self.data_weighting,) + key
        # make sure that the key doesn't use numpy integers.
        # numpy integers break json.
        key = (key[0], key[1], int(key[2]), int(key[3]), key[4])
        self.r_params[key] = r_params

    def set_taper(self, taper):
        """
        Set data tapering type.

        Parameters
        ----------
        taper : str
            Type of data tapering. See uvtools.dspec.gen_window for options.
        """
        self.taper = taper

    def set_spw(self, spw_range, ndlys=None, set_Ndlys=True):
        """
        Set the spectral window range.

        Parameters
        ----------
        spw_range : tuple, contains start and end of spw in channel indices
            used to slice the frequency array
        ndlys : integer
            Number of delay bins. Default: None, sets number of delay
            bins equal to the number of frequency channels in the spw.
        set_Ndlys : bool, optional
            This argument exists so that we can change the spw
            while continuing to hold the number of delays constant.
            One place we need to do this is broadcasting data set flags
            temporarily to compute empirical covariance estimates
            so that error bar estimates are not contaminated by foregrounds.
            If we do this and the Ndlys are set, then our covariance matrix
            is computed for the wrong number of delays. (see cov_q_hat)
            If True, set the number of delays equal to ndlys.
            If False, don't modify the number of delays.
        """
        assert isinstance(spw_range, tuple), \
            "spw_range must be fed as a len-2 integer tuple"
        assert isinstance(spw_range[0], (int, np.integer)), \
            "spw_range must be fed as len-2 integer tuple"
        self.spw_range = spw_range
        self.spw_Nfreqs = spw_range[1] - spw_range[0]
        if set_Ndlys:
            self.set_Ndlys(ndlys=ndlys)

    def set_Ndlys(self, ndlys=None):
        """
        Set the number of delay bins used.

        Parameters
        ----------
        ndlys : integer
            Number of delay bins. Default: None, sets number of delay
            bins equal to the number of frequency channels in the current spw
        """

        if ndlys == None:
            self.spw_Ndlys = self.spw_Nfreqs
        else:
            # Check that one is not trying to estimate more delay channels than there are frequencies
            if self.spw_Nfreqs < ndlys:
                raise ValueError("Cannot estimate more delays than there are frequency channels")
            self.spw_Ndlys = ndlys

    def cov_q_hat(self, key1, key2, model='empirical', exact_norm=False, pol=False,
                  time_indices=None, allow_fft=False):
        """
        Compute the un-normalized covariance matrix for q_hat for a given pair
        of visibility vectors. Returns the following matrix:

        Cov(\hat{q}_a,\hat{q}_b)

        !!!Only supports covariance between same power-spectrum estimates!!!
        (covariance between pair of baselines with the same pair of baselines)
        !!!Assumes that both baselines used in power-spectrum estimate
        !!!have independent noise relizations!!!

        #updated to be a multi-time wrapper to get_unnormed_V

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos']
            How the covariances of the input data should be estimated.
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations.
            If 'empirical' is set, then error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            If 'autos' is set, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas over channel bandwidth
            and integration time.


        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            The H, G, V and by extension M matrices can be computed with 2d ffts replacing the majority of
            matrix products provided that spw_Ndlys == spw_Nfreqs.
            Default: False.

        time_indices: list of indices of times to include or just a single time.
        default is None -> compute covariance for all times.

        Returns
        -------
        cov_q_hat: array_like
            Matrix with covariances between un-normalized band powers (Ntimes, Nfreqs, Nfreqs)
        """
        # type check
        if time_indices is None:
            time_indices = [tind for tind in range(self.Ntimes)]
        elif isinstance(time_indices, (int, np.integer)):
            time_indices = [time_indices]
        if not isinstance(time_indices, list):
            raise ValueError("time_indices must be an integer or list of integers.")
        if isinstance(key1,list):
            assert isinstance(key2, list), "key1 is a list, key2 must be a list"
            assert len(key2) == len(key1), "key1 length must equal key2 length"
        if isinstance(key2,list):
            assert isinstance(key1, list), "key2 is a list, key1 must be a list"
        #check time_indices
        for tind in time_indices:
            if not (tind >= 0 and tind <= self.Ntimes):
                raise ValueError("Invalid time index provided.")

        if not isinstance(key1,list):
            key1 = [key1]
        if not isinstance(key2,list):
            key2 = [key2]
        output = np.zeros((len(time_indices), self.spw_Ndlys, self.spw_Ndlys), dtype=complex)
        for k1, k2 in zip(key1, key2):
            if model in ['dsets']:
                output+=1./np.asarray([self.get_unnormed_V(k1, k2, model=model,
                                  exact_norm=exact_norm, pol=pol, time_index=t, allow_fft=allow_fft)\
                                  for t in time_indices])

            elif model in ['empirical']:
                #empirical error bars require broadcasting flags
                if model == 'empirical':
                    flag_backup=self.broadcast_dset_flags(spw_ranges=[self.spw_range], set_time_independent_weights=False)
                vm = self.get_unnormed_V(k1, k2, model=model,
                                  exact_norm=exact_norm, time_index = 0, pol=pol, allow_fft=allow_fft)
                output += 1./np.asarray([vm for t in time_indices])

        return float(len(key1)) / output

    def q_hat(self, key1, key2, allow_fft=False, exact_norm=False, pol=False):
        """

        If exact_norm is False:
        Construct an unnormalized bandpower, q_hat, from a given pair of
        visibility vectors. Returns the following quantity:

          \hat{q}_a = (1/2) conj(x_1) R_1 Q^alt_a R_2 x_2

        Note that the R matrix need not be set to C^-1. This is something that
        is set by the user in the set_R method.

        This is related to Equation 13 of arXiv:1502.06016. However, notice
        that there is a Q^alt_a instead of Q_a. The latter is defined as
        Q_a \equiv dC/dp_a. Since this is the derivative of the covariance
        with respect to the power spectrum, it contains factors of the primary
        beam etc. Q^alt_a strips away all of this, leaving only the barebones
        job of taking a Fourier transform. See HERA memo #44 for details.

        This function uses the state of self.data_weighting and self.taper
        in constructing q_hat. See PSpecData.pspec for details.

        If exact_norm is True:
        Takes beam factors into account (Eq. 14 in HERA memo #44)

        Parameters
        ----------
        key1, key2: tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors for each power spectrum estimate.
            q_a formed from key1, key2
            If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        allow_fft : bool, optional
            Whether to use a fast FFT summation trick to construct q_hat, or
            a simpler brute-force matrix multiplication. The FFT method assumes
            a delta-fn bin in delay space. It also only works if the number
            of delay bins is equal to the number of frequencies. Default: False.

        exact_norm: bool, optional
            If True, beam and spectral window factors are taken
            in the computation of Q_matrix (dC/dp = Q, and not Q_alt)
            (HERA memo #44, Eq. 11). Q matrix, for each delay mode,
            is weighted by the integral of beam over theta,phi.

        pol: str/int/bool, optional
            Used only if exact_norm is True. This argument is passed to get_integral_beam
            to extract the requested beam polarization. Default is the first
            polarization passed to pspec.

        Returns
        -------
        q_hat : array_like
            Unnormalized/normalized bandpowers
        """
        Rx1 = np.zeros((self.Ntimes, self.spw_Nfreqs),
                            dtype=complex)
        Rx2 = np.zeros_like(Rx1)
        # Calculate R x_1
        if not isinstance(key1, list):
            key1 = [key1]
        if not isinstance(key2, list):
            key2 = [key2]
        for _key in key1:
            tindep = self._time_independent_weights[self.parse_blkey(_key)]
            unflagged_ind = self._unflagged_time_integration[self.parse_blkey(_key)]
            x_wf = self.x(_key, include_extension=True)
            if tindep:
                rmat = self.R(_key, unflagged_ind)
                Rx1 += (rmat @ x_wf).T
            else:
                rmat = self.R(_key)
                Rx1 += np.asarray([rmat[m] @ x_wf[:,m] for m in range(self.Ntimes)])

        # Calculate R x_2
        for _key in key2:
            tindep = self._time_independent_weights[self.parse_blkey(_key)]
            unflagged_ind = self._unflagged_time_integration[self.parse_blkey(_key)]
            x_wf = self.x(_key, include_extension=True)
            if tindep:
                rmat = self.R(_key, unflagged_ind)
                Rx2 += (rmat @ x_wf).T
            else:
                rmat = self.R(_key)
                Rx2 += np.asarray([rmat[m] @ x_wf[:, m] for m in range(self.Ntimes)])

        # The set of operations for exact_norm == True are drawn from Equations
        # 11(a) and 11(b) from HERA memo #44. We are incorporating the
        # multiplicatives to the exponentials, and sticking to quantities in
        # their physical units.

        # exact norm can be used with use_fft.
        if exact_norm:
            del_tau = np.median(np.diff(self.delays()))*1e-9  #Get del_eta in Eq.11(a) (HERA memo #44) (seconds)
            integral_beam = self.get_integral_beam(pol) #Integral of beam in Eq.11(a) (HERA memo #44)
            qnorm = del_tau * integral_beam
        else:
            qnorm = 1.
        if not allow_fft:
            q  = []
            for i in range(self.spw_Ndlys):
                # Ideally, del_tau and integral_beam should be part of get_Q. We use them here to
                # avoid their repeated computation for each delay mode.
                Q =  self.get_Q_alt(i) * qnorm
                QRx2 = np.dot(Q, Rx2.T).T

                # Square and sum over columns
                #qi = 0.5 * np.einsum('i...,i...->...', Rx1.conj(), QRx2)
                qi =  0.5 * np.sum(Rx1.conj() * QRx2, axis=1)
                q.append(qi)

            q = np.asarray(q) #(Ndlys X Ntime)
            return q

        # use FFT
        elif allow_fft:
            lst_grtr_mult = int(np.ceil(self.spw_Nfreqs / self.spw_Ndlys)) # lowest multiple of Ndlys greater then Nfreq
            npad = self.spw_Ndlys * lst_grtr_mult - self.spw_Nfreqs # padding necessary for fft to have harmonics of spw_Ndlys.
            _Rx1 = np.fft.fft(np.pad(Rx1, [(0, 0), (0, npad)], mode='constant'), axis=1)[:, ::lst_grtr_mult]
            _Rx2 = np.fft.fft(np.pad(Rx2, [(0, 0), (0, npad)], mode='constant'), axis=1)[:, ::lst_grtr_mult]
            if exact_norm:
                qnorm = np.diag(np.sqrt(qnorm))
            #We are applying the exact norm after the R matrix consistent with above.
            #We may want to think if the order should be reversed but I doubt it
            #matters much.
            return (0.5 * np.fft.fftshift(_Rx1 * qnorm, axes=1).conj() \
                       * np.fft.fftshift(_Rx2 * qnorm, axes=1)).T

        else:
            raise ValueError("spw_Nfreqs + extensions must equal spw_Ndlys if using fft.")

    def _get_G(self, key1, key2, time_index, exact_norm=False, pol=False,
               allow_fft=False):
        """
        helper function for get_G that uses caching to speed things up.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        time_indices : list of integers or array like integer list
            List of time indices to compute G. Default, compute for all times.

        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            and sampling False
            Default: False.

        Returns
        -------
        G : array_like, complex
            Fisher matrix, with dimensions (spw_Nfreqs, spw_Nfreqs).
        """
        assert isinstance(time_index, (int, np.int32, np.int64)), "time_index must be an integer. Supplied %s"%(time_index)
        Gkey = self.fisher_hash(time_index, key1, key2, pol=pol, exact_norm=exact_norm, sampling=False)
        if not Gkey in self._G:
            R1 = self.R(key1, time_index)
            R2 = self.R(key2, time_index)
            if (exact_norm):
                integral_beam1 = self.get_integral_beam(pol)
                integral_beam2 = self.get_integral_beam(pol, include_extension=True)
                del_tau = np.median(np.diff(self.delays()))*1e-9
                qnorm1 = del_tau * integral_beam1
                qnorm2 = del_tau * integral_beam2
            else:
                qnorm1 = 1.
                qnorm2 = 1.
            if allow_fft:
                #if not (self.spw_Nfreqs + np.sum(self.filter_extension) == self.spw_Ndlys):
                #    raise ValueError("Nfreqs with extensions must equal Nspw for allow_fft")
                #We can calculate H much faster with an fft if we
                #don't have sampling
                # find least common multiple
                lcm0 = int(np.ceil(R1.shape[0] / self.spw_Ndlys)) # lowest multiple of Ndlys greater then Nfreq
                pad0 = self.spw_Ndlys * lcm0 - R1.shape[0] # padding necessary for fft to have harmonics of spw_Ndlys.
                lcm1 = int(np.ceil(R1.shape[1] / self.spw_Ndlys))
                pad1 = self.spw_Ndlys * lcm1 - R1.shape[1]
                r1_fft = np.fft.fftshift(np.fft.fft2(np.pad((R1 * qnorm1)[:,::-1], [(0, pad0), (0, pad1)], mode='constant'))[::lcm0, ::lcm1])
                r2_fft = np.fft.fftshift(np.fft.fft2(np.pad((R2 * qnorm2)[:,::-1], [(0, pad0), (0, pad1)], mode='constant'))[::lcm0, ::lcm1])
                G = np.conj(r1_fft) * r2_fft
            else:
                G = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
                iR1Q1, iR2Q2 = {}, {}

                for ch in range(self.spw_Ndlys):
                    Q1 = self.get_Q_alt(ch) * qnorm1
                    Q2 = self.get_Q_alt(ch, include_extension=True) * qnorm2
                    #G is given by Tr[E^\alpha C,\beta]
                    #where E^\alpha = R_1^\dagger Q^\apha R_2
                    #C,\beta = Q2 and Q^\alpha = Q1
                    #Note that we conjugate transpose R
                    #because we want to E^\alpha to
                    #give the absolute value squared of z = m_\alpha \dot R @ x
                    #where m_alpha takes the FT from frequency to the \alpha fourier mode.
                    #Q is essentially m_\alpha^\dagger m
                    # so we need to sandwhich it between R_1^\dagger and R_2
                    iR1Q1[ch] = np.dot(np.conj(R1).T, Q1) # R_1 Q_alt
                    iR2Q2[ch] = np.dot(R2, Q2) # R_2 Q
                for i in range(self.spw_Ndlys): # this loop goes as nchan^4
                    for j in range(self.spw_Ndlys):
                        G[i,j] = np.trace(np.dot(iR1Q1[i], iR2Q2[j]))
            self._G[Gkey] = G / 2.

        return self._G[Gkey]

    def get_G(self, key1, key2, exact_norm=False, pol=False,
              time_indices=None, allow_fft=False):
        """
        Calculates

            G_ab = (1/2) Tr[R_1 Q^alt_a R_2 Q^alt_b],

        which is needed for normalizing the power spectrum (see HERA memo #44).

        Note that in the limit that R_1 = R_2 = C^-1, this reduces to the Fisher
        matrix

            F_ab = 1/2 Tr [C^-1 Q^alt_a C^-1 Q^alt_b] (arXiv:1502.06016, Eq. 17)

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        time_indices : list of integers or array like integer list
            List of time indices to compute G. Default, compute for all times.

        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            Default: False.


        Returns
        -------
        G : array_like, complex
            Fisher matrix, with dimensions (Ntimes, spw_Nfreqs, spw_Nfreqs).
        """
        #This check is kind of arbitrary. I see it in a lot of places but there
        #really isn't anything special about Ndlys versus Ntimes, spw_Nfreqs,
        #and other ``essential" fields. I think we should remove this in future
        #releases. -AEW
        if self.spw_Ndlys == None:
            raise ValueError("Number of delay bins should have been set"
                             "by now! Cannot be equal to None.")
        if time_indices is None:
            time_indices = np.arange(self.Ntimes).astype(int)
        G = np.zeros((len(time_indices), self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
        for tind, time_index in enumerate(time_indices):
            G[tind] = self._get_G(key1=key1, key2=key2, time_index=time_index,
            exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
        # check if all zeros, in which case turn into identity
        if np.count_nonzero(G) == 0:
            G = np.asarray([np.eye(self.spw_Ndlys) for m in range(len(time_indices))])
        return G

    def _get_H(self, key1, key2, time_index, sampling=False, exact_norm=False, pol=False,
               allow_fft=False):
        """
        Helper function that calculates the response matrix H at a single time.
        takes advantage of caching.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        time_index : integer
            Specify time-step index to calculate H for.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            and sampling False
            Default: False.

        Returns
        -------
        H : array_like, complex
            Dimensions (Ndlys, Ndlys).
        """
        #Each H with fixed taper, input data weight, and weightings on key1 and key2
        #pol, and sampling bool is unique. This reduces need to recompute H over multiple times.
        assert isinstance(time_index, (int, np.int32, np.int64)), "time_index must be an integer. Supplied %s"%(time_index)
        Hkey = self.fisher_hash(time_index, key1, key2, sampling, exact_norm, pol)
        if not Hkey in self._H:
            H = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
            R1 = self.R(key1, time_index)
            R2 = self.R(key2, time_index)
            if (exact_norm):
                integral_beam1 = self.get_integral_beam(pol)
                integral_beam2 = self.get_integral_beam(pol, include_extension=True)
                del_tau = np.median(np.diff(self.delays()))*1e-9
                qnorm1 = del_tau * integral_beam1
                qnorm2 = del_tau * integral_beam2
            else:
                qnorm1 = 1.
                qnorm2 = 1.
            if allow_fft:
                if not sampling:
                    raise ValueError("sampling must equal True for allow_fft")
                #We can calculate H much faster with an fft if we
                #don't have sampling
                # find least common multiple
                lcm0 = int(np.ceil(R1.shape[0] / self.spw_Ndlys)) # lowest multiple of Ndlys greater then Nfreq
                pad0 = self.spw_Ndlys * lcm0 - R1.shape[0] # padding necessary for fft to have harmonics of spw_Ndlys.
                lcm1 = int(np.ceil(R1.shape[1] / self.spw_Ndlys))
                pad1 = self.spw_Ndlys * lcm1 - R1.shape[1]
                r1_fft = np.fft.fftshift(np.fft.fft2(np.pad((R1 * qnorm1)[:,::-1], [(0, pad0), (0, pad1)], mode='constant'))[::lcm0, ::lcm1])
                r2_fft = np.fft.fftshift(np.fft.fft2(np.pad((R2 * qnorm2)[:,::-1], [(0, pad0), (0, pad1)], mode='constant'))[::lcm0, ::lcm1])

                # restrict r1 and r2 to wavenumbers
                H = np.conj(r1_fft) * r2_fft
            else:
                if not sampling:
                    nfreq=np.sum(self.filter_extension) + self.spw_Nfreqs
                    sinc_matrix = np.zeros((nfreq, nfreq), dtype=complex)
                    for i in range(nfreq):
                        for j in range(nfreq):
                            sinc_matrix[i,j] = np.float(i - j)
                    sinc_matrix = np.sinc(sinc_matrix / np.float(nfreq))
                iR1Q1, iR2Q2 = {}, {}
                for ch in range(self.spw_Ndlys):
                    Q1 = self.get_Q_alt(ch) * qnorm1
                    Q2 = self.get_Q_alt(ch, include_extension=True) * qnorm2
                    if not sampling:
                        Q2 *= sinc_matrix
                    #H is given by Tr[E^\alpha C,\beta]
                    #where E^\alpha = R_1^\dagger Q^\apha R_2
                    #C,\beta = Q2 and Q^\alpha = Q1
                    #Note that we conjugate transpose R
                    #because we want to E^\alpha to
                    #give the absolute value squared of z = m_\alpha \dot R @ x
                    #where m_alpha takes the FT from frequency to the \alpha fourier mode.
                    #Q is essentially m_\alpha^\dagger m
                    # so we need to sandwhich it between R_1^\dagger and R_2
                    iR1Q1[ch] = np.dot(np.conj(R1).T, Q1) # R_1 Q_alt
                    iR2Q2[ch] = np.dot(R2, Q2) # R_2 Q
                for i in range(self.spw_Ndlys): # this loop goes as nchan^4
                    for j in range(self.spw_Ndlys):
                        H[i,j] = np.trace(np.dot(iR1Q1[i], iR2Q2[j]))

            self._H[Hkey] = H / 2.
        return self._H[Hkey]

    def get_H(self, key1, key2, sampling=False, exact_norm = False, pol=False,
              time_indices=None, allow_fft=False):
        """
        Calculates the response matrix H of the unnormalized band powers q
        to the true band powers p, i.e.,

            <q_a> = \sum_b H_{ab} p_b

        This is given by

            H_ab = (1/2) Tr[R_1 Q_a^alt R_2 Q_b]

        (See HERA memo #44). As currently implemented, this approximates the
        primary beam as frequency independent. Under this approximation, the
        our H_ab is defined using the equation above *except* we have
        Q^tapered rather than Q_b, where

            \overline{Q}^{tapered,beta}
            = e^{i 2pi eta_beta (nu_i - nu_j)} gamma(nu_i) gamma(nu_j)

        where gamma is the tapering function. Again, see HERA memo #44 for
        details.

        The sampling option determines whether one is assuming that the
        output points are integrals over k bins or samples at specific
        k values. The effect is to add a dampening of widely separated
        frequency correlations with the addition of the term

            sinc(pi \Delta eta (nu_i - nu_j))

        Note that numpy uses the engineering definition of sinc, where
        sinc(x) = sin(pi x) / (pi x), whereas in the line above and in
        all of our documentation, we use the physicist definition where
        sinc(x) = sin(x) / x

        Note that in the limit that R_1 = R_2 = C^-1 and Q_a is used instead
        of Q_a^alt, this reduces to the Fisher matrix

            F_ab = 1/2 Tr [C^-1 Q_a C^-1 Q_b] (arXiv:1502.06016, Eq. 17)

        This function uses the state of self.taper in constructing H.
        See PSpecData.pspec for details.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            Requires sampling False
            Default: False.


        Returns
        -------
        H : array_like, complex
            Dimensions (Ntimes, Nfreqs, Nfreqs).
        """
        #This check is kind of arbitrary. I see it in a lot of places but there
        #really isn't anything special about Ndlys versus Ntimes, Nfreqs,
        #and other ``essential" fields. I think we should remove this in future
        #releases. -AEW
        if self.spw_Ndlys == None:
            raise ValueError("Number of delay bins should have been set"
                             "by now! Cannot be equal to None.")
        if time_indices is None:
            time_indices = np.arange(self.Ntimes).astype(int)
        if self.spw_Ndlys == None:
            raise ValueError("Number of delay bins should have been set"
                             "by now! Cannot be equal to None.")
        H = np.zeros((len(time_indices),self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)

        for tind,time_index in enumerate(time_indices):
            H[tind] = self._get_H(key1=key1, key2=key2, time_index=time_index,
            sampling=sampling, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)

        # check if all zeros, in which case turn into identity
        if np.count_nonzero(H) == 0:
            H = np.asarray([np.eye(self.spw_Ndlys) for m in range(len(time_indices))])
        return H

    def get_unnormed_E(self, key1, key2, time_index, exact_norm=False, pol=False):
        """
        Calculates a series of unnormalized E matrices, such that

            q_a = x_1^* E^{12,a} x_2

        so that

            E^{12,a} = (1/2) R_1 Q^a R_2.

        In principle, this could be used to actually estimate q-hat. In other
        words, we could call this function to get E, and then sandwich it with
        two data vectors to get q_a. However, this should be slower than the
        implementation in q_hat. So this should only be used as a helper
        method for methods such as get_unnormed_V.

        Note for the future: There may be advantages to doing the
        matrix multiplications separately


        Parameters
        ----------
        key1, key2 : tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        time_index : integer,
            time interval to compute E-matrix for.

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        Returns
        -------
        E : array_like, complex
            Set of E matrices, with dimensions (Ndlys, Nfreqs, Nfreqs).

        """
        #This check is kind of arbitrary. I see it in a lot of places but there
        #really isn't anything special about Ndlys versus Ntimes, Nfreqs,
        #and other ``essential" fields. I think we should remove this in future
        #releases. -AEW
        if self.spw_Ndlys == None:
            raise ValueError("Number of delay bins should have been set"
                             "by now! Cannot be equal to None")

        Ekey = self.fisher_hash(time_index, key1, key2, pol=pol, exact_norm=exact_norm, sampling=False)
        if not Ekey in self._E:
            assert time_index >= 0 and time_index < self.Ntimes, "time_index must be between 0 and Ntimes"
            nfreq = self.spw_Nfreqs + np.sum(self.filter_extension)
            E_matrices = np.zeros((self.spw_Ndlys, nfreq, nfreq),
                                   dtype=np.complex)
            R1 = self.R(key1, time_index)
            R2 = self.R(key2, time_index)
            if (exact_norm):
                integral_beam = self.get_integral_beam(pol)
                del_tau = np.median(np.diff(self.delays()))*1e-9
            for dly_idx in range(self.spw_Ndlys):
                if exact_norm: QR2 = del_tau * integral_beam * np.dot(self.get_Q_alt(dly_idx), R2)
                else: QR2 = np.dot(self.get_Q_alt(dly_idx), R2)
                E_matrices[dly_idx] = np.dot(np.conj(R1).T, QR2)

            self._E[Ekey] = 0.5 * E_matrices

        return self._E[Ekey]

    def get_analytic_covariance(self, key1, key2, M=None, exact_norm=False, pol=False, model='empirical', known_cov=None):
        """
        Calculates the auto-covariance matrix for both the real and imaginary
        parts of bandpowers (i.e., the q vectors and the p vectors).

        Define

        .. math ::
            Real part of q_a = (1/2) (q_a + q_a^*)
            Imaginary part of q_a = (1/2i) (q_a - q_a^dagger)
            Real part of p_a = (1/2) (p_a + p_a^dagger)
            Imaginary part of p_a = (1/2i) (p_a - p_a^dagger)

            E^{12,a} = (1/2) R_1 Q^a R_2
            C^{12} = <x1 x2^dagger> - <x1><x2^dagger>
            P^{12} = <x1 x2> - <x1><x2>
            S^{12} = <x1^* x2^*> - <x1^*> <x2^*>
            p_a = M_{ab} q_b

        Then

        .. math ::
        The variance of (1/2) (q_a + q_a^dagger):
        (1/4){ (<q_a q_a> - <q_a><q_a>) + 2(<q_a q_a^dagger> - <q_a><q_a^dagger>)
        + (<q_a^dagger q_a^dagger> - <q_a^dagger><q_a^dagger>) }

        The variance of (1/2i) (q_a - q_a^dagger) :
        (-1/4){ (<q_a q_a> - <q_a><q_a>) - 2(<q_a q_a^dagger> - <q_a><q_a^dagger>)
        + (<q_a^dagger q_a^dagger> - <q_a^dagger><q_a^dagger>) }

        The variance of (1/2) (p_a + p_a^dagger):
        (1/4) { M_{ab} M_{ac} (<q_b q_c> - <q_b><q_c>) +
        M_{ab} M_{ac}^* (<q_b q_c^dagger> - <q_b><q_c^dagger>) +
        M_{ab}^* M_{ac} (<q_b^dagger q_c> - <q_b^dagger><q_c>) +
        M_{ab}^* M_{ac}^* (<q_b^dagger q_c^dagger> - <q_b^dagger><q_c^dagger>) }

        The variance of (1/2i) (p_a - p_a^dagger):
        (-1/4) { M_{ab} M_{ac} (<q_b q_c> - <q_b><q_c>) -
        M_{ab} M_{ac}^* (<q_b q_c^dagger> - <q_b><q_c^dagger>) -
        M_{ab}^* M_{ac} (<q_b^dagger q_c> - <q_b^dagger><q_c>) +
        M_{ab}^* M_{ac}^* (<q_b^dagger q_c^dagger> - <q_b^dagger><q_c^dagger>) }

        where
        <q_a q_b> - <q_a><q_b> =
                    tr(E^{12,a} C^{21} E^{12,b} C^{21})
                    + tr(E^{12,a} P^{22} E^{21,b*} S^{11})
        <q_a q_b^dagger> - <q_a><q_b^dagger> =
                    tr(E^{12,a} C^{22} E^{21,b} C^{11})
                    + tr(E^{12,a} P^{21} E^{12,b *} S^{21})
        <q_a^dagger q_b^dagger> - <q_a^dagger><q_b^dagger> =
                    tr(E^{21,a} C^{12} E^{21,b} C^{12})
                    + tr(E^{21,a} P^{11} E^{12,b *} S^{22})

        Note that

        .. math ::
            E^{12,a}_{ij}.conj = E^{21,a}_{ji}

        This function estimates C^1, C^2, P^{12}, and S^{12} empirically by
        default. (So while the pointy brackets <...> should in principle be
        ensemble averages, in practice the code performs averages in time.)

        Note: Time-dependent flags that differ from frequency channel-to-channel
        can create spurious spectral structure. Consider factorizing the flags with
        self.broadcast_dset_flags() before using model='time_average'

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        M : array_like
            Normalization matrix, M. Ntimes x Ndlys x Ndlys

        exact_norm : boolean
            If True, beam and spectral window factors are taken
            in the computation of Q_matrix (dC/dp = Q, and not Q_alt)
            (HERA memo #44, Eq. 11). Q matrix, for each delay mode,
            is weighted by the integral of beam over theta,phi.
            Therefore the output power spectra is, by construction, normalized.
            If True, it returns normalized power spectrum, except for X2Y term.
            If False, Q_alt is used (HERA memo #44, Eq. 16), and the power
            spectrum is normalized separately.

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        model : string, optional
            Type of covariance model to use. if not cached. Options=['empirical', 'dsets', 'autos', 'foreground_dependent',
            (other model names in known_cov)]
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations. In 'empirical' mode,
            error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances. In 'autos' mode, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas forming the baseline
            across channel bandwidth and integration time.
            In 'foreground_dependent' mode, it involves using auto-correlation amplitudes to model the input noise covariance
            and visibility outer products to model the input systematics covariance.

            ############
            When model is chosen as "autos" or "dsets", only C^{11} and C^{22} are accepted as non-zero values,
            and the two matrices are also expected to be diagonal,
            thus only <q_a q_b^dagger> - <q_a><q_b^dagger> = tr[ E^{12,a} C^{22} E^{21,b} C^{11} ] exists
            in the covariance terms of q vectors.
            When model is chosen as 'foreground_dependent', we further include the signal-noise coupling term
            besides the noise in the output covariance. Still only <q_a q_b^dagger> - <q_a><q_b^dagger> is non-zero,
            while it takes a form of tr[ E^{12,a} Cn^{22} E^{21,b} Cn^{11} +
            E^{12,a} Cs^{22} E^{21,b} Cn^{11} +
            E^{12,a} Cn^{22} E^{21,b} Cs^{11} ],
            where Cn is just Cautos, the input noise covariance estimated by the auto-correlation amplitudes (by calling C_model(model='autos')), and
            Cs uses the outer product of input visibilities to model the covariance on systematics.
            To construct a symmetric and unbiased covariance matrix, we choose
            Cs^{11}_{ij} = Cs^{22}_{ij} = 1/2 * [ x1_i x2_j^{*} + x2_i x1_j^{*} ], which preserves the property Cs_{ij}^* = Cs_{ji}.
            #############

        known_cov : dicts of covariance matrices
            Covariance matrices that are not calculated internally from data.

        Returns
        -------
        V : array_like, complex
            Bandpower covariance, with dimension (Ntimes, spw_Ndlys, spw_Ndlys).
        """
        # Collect all the relevant pieces
        if M.ndim == 2:
            M = np.asarray([M for time in range(self.Ntimes)])
        # M has a shape of (Ntimes, spw_Ndlys,spw_Ndlys)

        # check if the covariance matrix is uniform along the time axis. If so, we just calculate the result for one timestamp and duplicate its copies
        # along the time axis.
        check_uniform_input = False
        if model != 'foreground_dependent':
        # When model is 'foreground_dependent', since we are processing the outer products of visibilities from different times,
        # we are expected to have time-dependent inputs, thus check_uniform_input is always set to be False here.
            C11_first = self.C_model(key1, model=model, known_cov=known_cov, time_index=0)
            C11_last = self.C_model(key1, model=model, known_cov=known_cov, time_index=self.dsets[0].Ntimes-1)
            if np.isclose(C11_first, C11_last).all():
                check_uniform_input = True

        cov_q_real, cov_q_imag, cov_p_real, cov_p_imag = [], [], [], []
        for time_index in range(self.dsets[0].Ntimes):
            E_matrices = self.get_unnormed_E(key1, key2, time_index=time_index, exact_norm=exact_norm, pol=pol)
            # E_matrices has a shape of (spw_Ndlys, spw_Nfreqs, spw_Nfreqs)

            # using numpy.einsum_path to speed up the array products with numpy.einsum
            einstein_path_0 =  np.einsum_path('bij, cji->bc', E_matrices, E_matrices, optimize='optimal')[0]
            einstein_path_1 = np.einsum_path('bi, ci,i->bc', E_matrices[:,:,0], E_matrices[:,:,0],E_matrices[0,:,0], optimize='optimal')[0]
            einstein_path_2 =  np.einsum_path('ab,cd,bd->ac', M[0], M[0], M[0], optimize='optimal')[0]
            if model in ['dsets','autos']:
                # calculate <q_a q_b^dagger> - <q_a><q_b^dagger> = tr[ E^{12,a} C^{22} E^{21,b} C^{11} ]
                # We have used tr[A D_1 B D_2] = \sum_{ijkm} A_{ij} d_{1j} \delta_{jk} B_{km} d_{2m} \delta_{mi} = \sum_{ik} [A_{ik}*d_{1k}] * [B_{ki}*d_{2i}]
                # to simplify the computation.
                C11 = self.C_model(key1, model=model, known_cov=known_cov, time_index=time_index)
                C22 = self.C_model(key2, model=model, known_cov=known_cov, time_index=time_index)
                E21C11 = np.multiply(np.transpose(E_matrices.conj(), (0,2,1)), np.diag(C11))
                E12C22 = np.multiply(E_matrices, np.diag(C22))
                # Get q_q, q_qdagger, qdagger_qdagger
                q_q, qdagger_qdagger = 0.+1.j*0, 0.+1.j*0
                q_qdagger = np.einsum('bij, cji->bc', E12C22, E21C11, optimize=einstein_path_0)
            elif model == 'foreground_dependent':
                # calculate tr[ E^{12,b} Cautos^{22} E^{21,c} Cautos^{11} +
                # E^{12,b} Cs E^{21,c} Cautos^{11} +
                # E^{12,b} Cautos^{22} E^{21,c} Cs ],
                # and we take Cs_{ij} = 1/2 * [ x1_i x2_j^{*} + x2_i x1_j^{*} ].
                # For terms like E^{12,b} Cs E^{21,c} Cautos^{11},
                # we have used tr[A u u*^t B D_2] = \sum_{ijkm} A_{ij} u_j u*_k B_{km} D_{2mi} \\
                # = \sum_{i} [ \sum_j A_{ij} u_j ] * [\sum_k u*_k B_{ki} ] * d_{2i}
                # to simplify the computation.
                C11_autos = self.C_model(key1, model='autos', known_cov=known_cov, time_index=time_index)
                C22_autos = self.C_model(key2, model='autos', known_cov=known_cov, time_index=time_index)
                E21C11_autos = np.multiply(np.transpose(E_matrices.conj(), (0,2,1)), np.diag(C11_autos))
                E12C22_autos = np.multiply(E_matrices, np.diag(C22_autos))
                # Get q_q, q_qdagger, qdagger_qdagger
                q_q, qdagger_qdagger = 0.+1.j*0, 0.+1.j*0
                q_qdagger = np.einsum('bij, cji->bc', E12C22_autos, E21C11_autos, optimize=einstein_path_0)
                x1 = self.w(key1)[:,time_index] * self.x(key1)[:,time_index]
                x2 = self.w(key2)[:,time_index] * self.x(key2)[:,time_index]
                E12_x1 = np.dot(E_matrices, x1)
                E12_x2 = np.dot(E_matrices, x2)
                x2star_E21 = E12_x2.conj()
                x1star_E21 = E12_x1.conj()
                x1star_E12 = np.dot(np.transpose(E_matrices,(0,2,1)), x1.conj())
                x2star_E12 = np.dot(np.transpose(E_matrices,(0,2,1)), x2.conj())
                E21_x1 = x1star_E12.conj()
                E21_x2 = x2star_E12.conj()
                SN_cov = np.einsum('bi,ci,i->bc', E12_x1, x2star_E21, np.diag(C11_autos), optimize=einstein_path_1)/2. + np.einsum('bi,ci,i->bc', E12_x2, x1star_E21, np.diag(C11_autos), optimize=einstein_path_1)/2.\
                            + np.einsum('bi,ci,i->bc', x2star_E12, E21_x1, np.diag(C22_autos), optimize=einstein_path_1)/2. + np.einsum('bi,ci,i->bc', x1star_E12, E21_x2, np.diag(C22_autos), optimize=einstein_path_1)/2.
                # Apply zero clipping on the columns and rows containing negative diagonal elements
                SN_cov[np.real(np.diag(SN_cov))<=0., :] = 0. + 1.j*0
                SN_cov[:, np.real(np.diag(SN_cov))<=0.,] = 0. + 1.j*0
                q_qdagger += SN_cov
            else:
                # for general case (which is the slowest without simplification)
                C11 = self.C_model(key1, model=model, known_cov=known_cov, time_index=time_index)
                C22 = self.C_model(key2, model=model, known_cov=known_cov, time_index=time_index)
                C21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=True, known_cov=known_cov, time_index=time_index)
                C12 = self.cross_covar_model(key1, key2, model=model, conj_1=False, conj_2=True, known_cov=known_cov, time_index=time_index)
                P11 = self.cross_covar_model(key1, key1, model=model, conj_1=False, conj_2=False, known_cov=known_cov, time_index=time_index)
                S11 = self.cross_covar_model(key1, key1, model=model, conj_1=True, conj_2=True, known_cov=known_cov, time_index=time_index)
                P22 = self.cross_covar_model(key2, key2, model=model, conj_1=False, conj_2=False, known_cov=known_cov, time_index=time_index)
                S22 = self.cross_covar_model(key2, key2, model=model, conj_1=True, conj_2=True, known_cov=known_cov, time_index=time_index)
                P21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=False, known_cov=known_cov, time_index=time_index)
                S21 = self.cross_covar_model(key2, key1, model=model, conj_1=True, conj_2=True, known_cov=known_cov, time_index=time_index)
                # Get q_q, q_qdagger, qdagger_qdagger
                if np.isclose(P22, 0).all() or np.isclose(S11,0).all():
                    q_q = 0.+1.j*0
                else:
                    E12P22 = np.matmul(E_matrices, P22)
                    E21starS11 = np.matmul(np.transpose(E_matrices, (0,2,1)), S11)
                    q_q = np.einsum('bij, cji->bc', E12P22, E21starS11, optimize=einstein_path_0)
                if np.isclose(C21, 0).all():
                    q_q += 0.+1.j*0
                else:
                    E12C21 = np.matmul(E_matrices, C21)
                    q_q += np.einsum('bij, cji->bc', E12C21, E12C21, optimize=einstein_path_0)
                E21C11 = np.matmul(np.transpose(E_matrices.conj(), (0,2,1)), C11)
                E12C22 = np.matmul(E_matrices, C22)
                q_qdagger = np.einsum('bij, cji->bc', E12C22, E21C11, optimize=einstein_path_0)
                if np.isclose(P21, 0).all() or np.isclose(S21,0).all():
                    q_qdagger += 0.+1.j*0
                else:
                    E12P21 = np.matmul(E_matrices, P21)
                    E12starS21 = np.matmul(E_matrices.conj(), S21)
                    q_qdagger += np.einsum('bij, cji->bc', E12P21, E12starS21, optimize=einstein_path_0)
                if np.isclose(C12, 0).all():
                    qdagger_qdagger = 0.+1.j*0
                else:
                    E21C12 = np.matmul(np.transpose(E_matrices.conj(), (0,2,1)), C12)
                    qdagger_qdagger = np.einsum('bij, cji->bc', E21C12, E21C12, optimize=einstein_path_0)
                if np.isclose(P11, 0).all() or np.isclose(S22,0).all():
                    qdagger_qdagger += 0.+1.j*0
                else:
                    E21P11 = np.matmul(np.transpose(E_matrices.conj(), (0,2,1)), P11)
                    E12starS22 = np.matmul(E_matrices.conj(), S22)
                    qdagger_qdagger += np.einsum('bij, cji->bc', E21P11, E12starS22, optimize=einstein_path_0)

            cov_q_real_temp = (q_q + qdagger_qdagger + q_qdagger + q_qdagger.conj() ) / 4.
            cov_q_imag_temp = -(q_q + qdagger_qdagger - q_qdagger - q_qdagger.conj() ) / 4.

            m = M[time_index]
            # calculate \sum_{bd} [ M_{ab} M_{cd} (<q_b q_d> - <q_b><q_d>) ]
            if np.isclose([q_q], 0).all():
                MMq_q = np.zeros((E_matrices.shape[0],E_matrices.shape[0])).astype(np.complex128)
            else:
                assert np.shape(q_q) == np.shape(m), "covariance matrix and normalization matrix has different shapes."
                MMq_q = np.einsum('ab,cd,bd->ac', m, m, q_q, optimize=einstein_path_2)
            # calculate \sum_{bd} [ M_{ab} M_{cd}^* (<q_b q_d^dagger> - <q_b><q_d^dagger>) ]
            # and \sum_{bd} [ M_{ab}^* M_{cd} (<q_b^dagger q_d> - <q_b^dagger><q_d>) ]
            if np.isclose([q_qdagger], 0).all():
                MM_q_qdagger = 0.+1.j*0
                M_Mq_qdagger_ = 0.+1.j*0
            else:
                assert np.shape(q_qdagger) == np.shape(m), "covariance matrix and normalization matrix has different shapes."
                MM_q_qdagger = np.einsum('ab,cd,bd->ac', m, m.conj(), q_qdagger, optimize=einstein_path_2)
                M_Mq_qdagger_ = np.einsum('ab,cd,bd->ac', m.conj(), m, q_qdagger.conj(), optimize=einstein_path_2)
            # calculate \sum_{bd} [ M_{ab}^* M_{cd}^* (<q_b^dagger q_d^dagger> - <q_b^dagger><q_d^dagger>) ]
            if np.isclose([qdagger_qdagger], 0).all():
                M_M_qdagger_qdagger = 0.+1.j*0
            else:
                assert np.shape(qdagger_qdagger) == np.shape(m), "covariance matrix and normalization matrix has different shapes."
                M_M_qdagger_qdagger = np.einsum('ab,cd,bd->ac', m.conj(), m.conj(), qdagger_qdagger, optimize=einstein_path_2)

            cov_p_real_temp = ( MMq_q + MM_q_qdagger + M_Mq_qdagger_ + M_M_qdagger_qdagger)/ 4.
            cov_p_imag_temp = -( MMq_q - MM_q_qdagger - M_Mq_qdagger_ + M_M_qdagger_qdagger)/ 4.
            # cov_p_real_temp has a shaoe of (spw_Ndlys, spw_Ndlys)

            if check_uniform_input:
            # if the covariance matrix is uniform along the time axis, we just calculate the result for one timestamp and duplicate its copies
            # along the time axis.
                cov_q_real.extend([cov_q_real_temp]*self.dsets[0].Ntimes)
                cov_q_imag.extend([cov_q_imag_temp]*self.dsets[0].Ntimes)
                cov_p_real.extend([cov_p_real_temp]*self.dsets[0].Ntimes)
                cov_p_imag.extend([cov_p_imag_temp]*self.dsets[0].Ntimes)
                warnings.warn("Producing time-uniform covariance matrices between bandpowers.")
                break
            else:
                cov_q_real.append(cov_q_real_temp)
                cov_q_imag.append(cov_q_imag_temp)
                cov_p_real.append(cov_p_real_temp)
                cov_p_imag.append(cov_p_imag_temp)

        cov_q_real = np.asarray(cov_q_real)
        cov_q_imag = np.asarray(cov_q_imag)
        cov_p_real = np.asarray(cov_p_real)
        cov_p_imag = np.asarray(cov_p_imag)
        # (Ntimes, spw_Ndlys, spw_Ndlys)

        return cov_q_real, cov_q_imag, cov_p_real, cov_p_imag


    def get_unnormed_V(self, key1, key2, time_index, model='empirical', exact_norm=False,
                       pol=False, allow_fft=False):
        """

        ATTENTION: This function computes the complex covariance of qhat which can be useful
                   for normalization but does not have all the information on the covariances of the real
                   and imaginary components of qHat. For these, see get_analytic_covariances.

        Calculates the covariance matrix for unnormed bandpowers (i.e., the q
        vectors). If the data were real and x_1 = x_2, the expression would be

        .. math ::
            V_ab = 2 tr(C E_a C E_b), where E_a = (1/2) R Q^a R

        When the data are complex, the expression becomes considerably more
        complicated. Define

        .. math ::
            E^{12,a} = (1/2) R_1 Q^a R_2
            C^1 = <x1 x1^dagger> - <x1><x1^dagger>
            C^2 = <x2 x2^dagger> - <x2><x2^dagger>
            P^{12} = <x1 x2> - <x1><x2>
            S^{12} = <x1^* x2^*> - <x1^*> <x2^*>

        Then

        .. math ::
            V_ab = tr(E^{12,a} C^2 E^{21,b} C^1)
                    + tr(E^{12,a} P^{21} E^{12,b *} S^{21})

        Note that

        .. math ::
            E^{12,a}_{ij}.conj = E^{21,a}_{ji}

        This function estimates C^1, C^2, P^{12}, and S^{12} empirically by
        default. (So while the pointy brackets <...> should in principle be
        ensemble averages, in practice the code performs averages in time.)

        Empirical covariance estimates are in principle a little risky, as they
        can potentially induce signal loss. This is probably ok if we are just
        looking intending to look at V. It is most dangerous when C_emp^-1 is
        applied to the data. The application of using this to form do a V^-1/2
        decorrelation is probably medium risk. But this has yet to be proven,
        and results coming from V^-1/2 should be interpreted with caution.

        Note for future: Although the V matrix should be Hermitian by
        construction, in practice there are precision issues and the
        Hermiticity is violated at ~ 1 part in 10^15. (Which is ~the expected
        roundoff error). If something messes up, it may be worth investigating
        this more.

        Note for the future: If this ends up too slow, Cholesky tricks can be
        employed to speed up the computation by a factor of a few.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        time_index : integer,
            time interval to compute E-matrix for.

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos']
            How the covariances of the input data should be estimated.
            In 'dsets' mode, error bars are estimated from user-provided
            per baseline and per channel standard deivations.
            If 'empirical' is set, then error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            If 'autos' is set, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas over channel bandwidth
            and integration time.

        time_index : integer, compute covariance at specific time-step

        allow_fft : bool, optional
            If True, speed things up with ffts. Requires spw_Ndlys == spw_Nfreqs.
            Default: False.



        Returns
        -------
        V : array_like, complex
            Bandpower covariance matrix, with dimensions (Ndlys, Ndlys).
        """
        # Collect all the relevant pieces
        Vkey = self.cov_hash(time_index, key1, key2, exact_norm, pol, model)
        if not Vkey in self._V:
            if model in ['dsets', 'empirical', 'autos']:
                C1 = self.C_model(key1, model=model, time_index=time_index)
                C2 = self.C_model(key2, model=model, time_index=time_index)
                P21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=False)
                S21 = self.cross_covar_model(key2, key1, model=model, conj_1=True, conj_2=True)
                if not allow_fft:
                    E_matrices = self.get_unnormed_E(key1, key2, exact_norm=exact_norm,
                                                     time_index=time_index, pol=pol)
                    E21C1 = np.dot(np.transpose(E_matrices.conj(), (0,2,1)), C1)
                    E12C2 = np.dot(E_matrices, C2)
                    auto_term = np.einsum('aij,bji', E12C2, E21C1)
                    E12starS21 = np.dot(E_matrices.conj(), S21)
                    E12P21 = np.dot(E_matrices, P21)
                    cross_term = np.einsum('aij,bji', E12P21, E12starS21)
                    output = auto_term + cross_term
                else:
                    if exact_norm:
                        qnorm = self.get_integral_beam(pol) * np.median(np.diff(self.delays()))*1e-9
                    else:
                        qnorm = 1.
                    R1 = self.R(key1, time_index)
                    R2 = self.R(key2, time_index)
                    C1_filtered = R1 @ C1 @ np.conj(R1).T * qnorm
                    C2_filtered = R2 @ C2 @ np.conj(R2).T * qnorm
                    S21_filtered = R1 @ S21 @ np.conj(R1).T * qnorm
                    P21_filtered = R2 @ P21 @ np.conj(R2).T * qnorm
                    lcm = int(np.ceil(self.spw_Nfreqs / self.spw_Ndlys)) # lowest multiple of Ndlys greater then Nfreq
                    pad = self.spw_Ndlys * lcm - self.spw_Nfreqs
                    c1_fft = np.fft.fftshift(np.fft.fft2(np.pad(C_1_filtered[:,::-1], [(0, pad), (0, pad)], mode='constant'))[::lcm, ::lcm])
                    c2_fft = np.fft.fftshift(np.fft.fft2(np.pad(C_2_filtered[:,::-1], [(0, pad), (0, pad)], mode='constant'))[::lcm, ::lcm])
                    s21_fft = np.fft.fftshift(np.fft.fft2(np.pad(S21_filtered[:,::-1], [(0, pad), (0, pad)], mode='constant'))[::lcm, ::lcm])
                    p21_fft = np.fft.fftshift(np.fft.fft2(np.pad(P21_filtered[:,::-1], [(0, pad), (0, pad)], mode='constant'))[::lcm, ::lcm])
                    auto_term = c1_fft * np.conj(c2_fft)
                    cross_term = p21_fft @ np.conj(s21_fft)

            self._V[Vkey] = output

        return self._V[Vkey]

    def _get_M(self, key1, key2, time_index, mode='I', sampling=False, exact_norm=False,
               pol=False, allow_fft=False, rcond=1e-15):
        """
        Helper function that returns M-matrix for a single time step.
        Several choices for M are supported:
            'I':      Set M to be diagonal (e.g. HERA Memo #44)
            'H^-1':   Set M = H^-1, the (pseudo)inverse response matrix.
            'V^-1/2': Set M = V^-1/2, the root-inverse response matrix (using SVD).

        These choices will be supported very soon:
            'L^-1':   Set M = L^-1, Cholesky decomposition.

        As written, the window functions will not be correclty normalized; it needs
        to be adjusted by the pspec scalar for it to be approximately correctly
        normalized. If the beam is being provided, this will be done in the pspec
        function.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        time_index : integer
            Specify time-step index to calculate H for.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        allow_fft : boolean, optional
            If set to True, allows a shortcut FFT method when
            the number of delay bins equals the number of delay channels.
            Default: False

        Returns
        -------
        M : array_like, complex
            Dimensions (Ndlys, Ndlys).

        """
        modes = ['H^-1', 'V^-1/2', 'I', 'L^-1', 'H^-1/2']
        assert(mode in modes)
        Mkey = self.fisher_hash(time_index, key1, key2, sampling, exact_norm, pol) + (mode, )
        if not Mkey in self._M:
            if mode != 'I' and exact_norm==True:
                raise NotImplementedError("Exact norm is not supported for non-I modes")
            if mode == 'H^-1':
                H = self._get_H(key1, key2, time_index, sampling=sampling, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
                try:
                    M = np.linalg.inv(H)
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        M = np.linalg.pinv(H, rcond=rcond)
                        raise_warning("Warning: Window function matrix is singular "
                                      "and cannot be inverted, so using "
                                      " pseudoinverse instead.")
                    else:
                        raise np.linalg.LinAlgError("Linear algebra error with H matrix "
                                                    "during MW computation.")

            elif mode == 'V^-1/2':
                H = self._get_H(key1, key2, time_index, sampling=sampling, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
                # First find the eigenvectors and eigenvalues of the unnormalizd covariance
                # Then use it to compute V^-1/2
                band_covar = self.get_unnormed_V(key1, key2, time_index, model=self.cov_model, exact_norm=exact_norm,
                                                 pol = pol)
                eigvals, eigvects = np.linalg.eig(band_covar)
                nonpos_eigvals = eigvals <= 1e-20
                if (nonpos_eigvals).any():
                    raise_warning("At least one non-positive eigenvalue for the "
                                  "unnormed bandpower covariance matrix.")
                    # truncate them
                    eigvals = eigvals[~nonpos_eigvals]
                    eigvects = eigvects[:, ~nonpos_eigvals]
                V_minus_half = np.dot(eigvects, np.dot(np.diag(1./np.sqrt(eigvals)), eigvects.T))

                W_norm = np.diag(1. / np.sum(np.dot(V_minus_half, H), axis=1))
                M = np.dot(W_norm, V_minus_half)

            elif mode == 'H^-1/2':
                H = self._get_H(key1, key2, time_index, sampling=sampling, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
                eigvals, eigvects = np.linalg.eig(H)
                nonpos_eigvals = eigvals <= 1e-20
                if (nonpos_eigvals).any():
                    raise_warning("At least one non-positive eigenvalue for the "
                                  "unnormed bandpower covariance matrix.")
                    # truncate them
                    eigvals = eigvals[~nonpos_eigvals]
                    eigvects = eigvects[:, ~nonpos_eigvals]
                H_minus_half = np.dot(eigvects, np.dot(np.diag(1./np.sqrt(eigvals)), eigvects.T))
                W_norm = np.diag(1. / np.sum(np.dot(H_minus_half, H), axis=1))
                M = np.dot(W_norm, H_minus_half)

            elif mode == 'I':
                G = self._get_G(key1, key2, time_index, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
                # This is not the M matrix as is rigorously defined in the
                # OQE formalism, because the power spectrum scalar is excluded
                # in this matrix normalization (i.e., M doesn't do the full
                # normalization)
                M = np.diag(1. / np.sum(G, axis=1))
            else:
                raise NotImplementedError("Cholesky decomposition mode not currently supported.")

            M[np.isnan(M)] = 0.
            M[np.isinf(M)] = 0.
            self._M[Mkey] = M

        return self._M[Mkey]

    def _get_W(self, key1, key2, time_index, mode='I', sampling=False, exact_norm=False, pol=False,
               allow_fft=False, rcond=1e-15):
        """
        Helper function that returns W-matrix for a single time step.
        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        time_index : integer
            Specify time-step index to calculate H for.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        allow_fft : boolean, optional
            If set to True, allows a shortcut FFT method when
            the number of delay bins equals the number of delay channels.
            Default: False

        Returns
        -------
        W : array_like, complex
            Dimensions (Ndlys, Ndlys).
        """
        Wkey = self.fisher_hash(time_index, key1, key2, sampling, exact_norm, pol) + (mode, )
        if not Wkey in self._W:
            M = self._get_M(key1, key2, time_index, mode=mode, sampling=sampling,
                            exact_norm=exact_norm, pol=pol, allow_fft=allow_fft, rcond=rcond)
            H = self._get_H(key1, key2, time_index, sampling=sampling, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)
            W = np.dot(M, H)
            if mode == 'H^-1':
                W_norm = np.sum(W, axis=1)
                W = (W.T / W_norm).T
            elif mode == 'I':
                W_norm = np.diag(1. / np.sum(H, axis=1))
                W = np.dot(W_norm, H)
            W[~np.isfinite(W)] = 0. # H can have NaNs!
            self._W[Wkey] = W
        return self._W[Wkey]

    def get_M(self, key1, key2, mode='I', exact_norm=False,
              sampling=False, time_indices=None, pol=False,
              allow_fft=False, rcond=1e-15):
        """
        Construct the normalization matrix M This is defined through Eqs. 14-16 of
        arXiv:1502.06016:

            \hat{p} = M \hat{q}
            <\hat{p}> = W p
            W = M H,

        where p is the true band power and H is the response matrix (defined above
        in get_H) of unnormalized bandpowers to normed bandpowers.

        Several choices for M are supported:
            'I':      Set M to be diagonal (e.g. HERA Memo #44)
            'H^-1':   Set M = H^-1, the (pseudo)inverse response matrix.
            'V^-1/2': Set M = V^-1/2, the root-inverse response matrix (using SVD).

        These choices will be supported very soon:
            'L^-1':   Set M = L^-1, Cholesky decomposition.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        time_indices : list, optional
            List of time-indices to calculate M-matrices for. Default: All times.

        allow_fft : boolean, optional
            If set to True, allows a shortcut FFT method when
            the number of delay bins equals the number of delay channels.
            Default: False

        Returns
        -------
        M : array_like, complex
            Dimensions (Ntimes Ndlys, Ndlys) or (Ndlys, Ndlys) if average_times is True.
        """
        if time_indices is None:
            time_indices = np.arange(self.Ntimes).astype(int)
        M = np.zeros((len(time_indices),self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
        for tind, time_index in enumerate(time_indices):
            M[tind] = self._get_M(key1, key2, time_index, mode=mode, sampling=sampling, exact_norm=exact_norm,
                                  pol=pol, allow_fft=allow_fft, rcond=rcond)
        return M

    def get_W(self, key1, key2, mode='I', sampling=False, exact_norm=False, time_indices=None, pol=False, allow_fft=False, rcond=1e-15):
        """
        Construct the Window function matrix W. This is defined through Eqs. 14-16 of
        arXiv:1502.06016:

            \hat{p} = M \hat{q}
            <\hat{p}> = W p
            W = M H,

        where p is the true band power and H is the response matrix (defined above
        in get_H) of unnormalized bandpowers to normed bandpowers.

        Several choices for M are supported:
            'I':      Set M to be diagonal (e.g. HERA Memo #44)
            'H^-1':   Set M = H^-1, the (pseudo)inverse response matrix.
            'V^-1/2': Set M = V^-1/2, the root-inverse response matrix (using SVD).

        These choices will be supported very soon:
            'L^-1':   Set M = L^-1, Cholesky decomposition.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights.

        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False

        exact_norm : boolean, optional
            Exact normalization (see HERA memo #44, Eq. 11 and documentation
            of q_hat for details).

        pol : str/int/bool, optional
            Polarization parameter to be used for extracting the correct beam.
            Used only if exact_norm is True.

        time_indices : list, optional
            List of time-indices to calculate M-matrices for. Default: All times.

        allow_fft : boolean, optional
            If set to True, allows a shortcut FFT method when
            the number of delay bins equals the number of delay channels.
            Default: False

        Returns
        -------
        W : array_like, complex
            Dimensions (Ntimes Ndlys, Ndlys) where Ntimes is the length of time_indices.
        """
        if time_indices is None:
            time_indices = np.arange(self.Ntimes).astype(int)
        W = np.zeros((len(time_indices),self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
        for tind, time_index in enumerate(time_indices):
            W[tind] = self._get_W(key1, key2, time_index, mode=mode, sampling=sampling, exact_norm=exact_norm, pol=pol, rcond=rcond)
        return W

    def get_Q_alt(self, mode, allow_fft=True, include_extension=False):
        """
        Response of the covariance to a given bandpower, dC / dp_alpha,
        EXCEPT without the primary beam factors. This is Q_alt as defined
        in HERA memo #44, so it's not dC / dp_alpha, strictly, but is just
        the part that does the Fourier transforms.

        Assumes that Q will operate on a visibility vector in frequency space.
        In the limit that self.spw_Ndlys equals self.spw_Nfreqs, this will
        produce a matrix Q that performs a two-sided FFT and extracts a
        particular Fourier mode.

        (Computing x^t Q y is equivalent to Fourier transforming x and y
        separately, extracting one element of the Fourier transformed vectors,
        and then multiplying them.)

        When self.spw_Ndlys < self.spw_Nfreqs, the effect is similar except
        the delay bins need not be in the locations usually mandated
        by the FFT algorithm.

        Parameters
        ----------
        mode : int
            Central wavenumber (index) of the bandpower, p_alpha.

        allow_fft : boolean, optional
            If set to True, allows a shortcut FFT method when
            the number of delay bins equals the number of delay channels.
            Default: True

        include_extension: If True, return a matrix that is spw_Nfreq x spw_Nfreq
        (required if using \partial C_{ij} / \partial p_\alpha since C_{ij} is
        (spw_Nfreq x spw_Nfreq).

        Return
        -------
        Q : array_like
            Response matrix for bandpower p_alpha.
        """
        if self.spw_Ndlys == None:
            self.set_Ndlys()

        if mode >= self.spw_Ndlys:
            raise IndexError("Cannot compute Q matrix for a mode outside"
                             "of allowed range of delay modes.")
        nfreq = self.spw_Nfreqs
        if include_extension:
            nfreq = nfreq + np.sum(self.filter_extension)
            phase_correction = self.filter_extension[0]
        else:
            phase_correction = 0.
        if (self.spw_Ndlys == nfreq) and (allow_fft == True):
            _m = np.zeros((nfreq,), dtype=np.complex)
            _m[mode] = 1. # delta function at specific delay mode
            # FFT to transform to frequency space
            m = np.fft.fft(np.fft.ifftshift(_m))
        else:
            if self.spw_Ndlys % 2 == 0:
                start_idx = -self.spw_Ndlys/2
            else:
                start_idx = -(self.spw_Ndlys - 1)/2
            m = (start_idx + mode) * (np.arange(nfreq) - phase_correction)
            m = np.exp(-2j * np.pi * m / self.spw_Ndlys)

        Q_alt = np.einsum('i,j', m.conj(), m) # dot it with its conjugate
        return Q_alt

    def get_integral_beam(self, pol=False, include_extension=False):
        """
        Computes the integral containing the spectral beam and tapering
        function in Q_alpha(i,j).

        Parameters
        ----------

        pol : str/int/bool, optional
            Which beam polarization to use. If the specified polarization
            doesn't exist, a uniform isotropic beam (with integral 4pi for all
            frequencies) is assumed. Default: False (uniform beam).

        Return
        -------
        integral_beam : array_like
            integral containing the spectral beam and tapering.
        """
        if include_extension:
            nu = self.freqs[self.spw_range[0]-self.filter_extension[0]:self.spw_range[1]+self.filter_extension[1]]
        else:
            nu  = self.freqs[self.spw_range[0]:self.spw_range[1]] # in Hz

        try:
            # Get beam response in (frequency, pixel), beam area(freq) and
            # Nside, used in computing dtheta
            beam_res, beam_omega, N = \
                self.primary_beam.beam_normalized_response(pol, nu)
            prod = 1. / beam_omega
            beam_prod = beam_res * prod[:, np.newaxis]

            # beam_prod has omega subsumed, but taper is still part of R matrix
            # The nside term is dtheta^2, where dtheta is the resolution in
            # healpix map
            integral_beam = np.pi/(3.*N*N) * np.dot(beam_prod, beam_prod.T)

        except(AttributeError):
            warnings.warn("The beam response could not be calculated. "
                          "PS will not be normalized!")
            integral_beam = np.ones((len(nu), len(nu)))

        return integral_beam

    def get_Q(self, mode):
        """
        Computes Q_alt(i,j), which is the exponential part of the
        response of the data covariance to the bandpower (dC/dP_alpha).

        Note: This function is not being used right now, since get_q_alt and
        get_Q are essentially the same functions. However, since we want to attempt
        non-uniform bins, we do intend to use get_Q (which uses physical
        units, and hence there is not contraint of uniformly spaced
        data).

        Parameters
        ----------
        mode : int
            Central wavenumber (index) of the bandpower, p_alpha.

        Return
        -------
        Q_alt : array_like
            Exponential part of Q (HERA memo #44, Eq. 11).
        """
        if self.spw_Ndlys == None:
            self.set_Ndlys()
        if mode >= self.spw_Ndlys:
            raise IndexError("Cannot compute Q matrix for a mode outside"
                             "of allowed range of delay modes.")

        tau = self.delays()[int(mode)] * 1.0e-9 # delay in seconds
        nu  = self.freqs[self.spw_range[0]:self.spw_range[1]] # in Hz

        eta_int = np.exp(-2j * np.pi * tau * nu) # exponential part
        Q_alt = np.einsum('i,j', eta_int.conj(), eta_int) # dot with conjugate
        return Q_alt

    def p_hat(self, M, q):
        """
        Optimal estimate of bandpower p_alpha, defined as p_hat = M q_hat.

        Parameters
        ----------
        M : array_like
            Normalization matrix, M.

        q : array_like
            Unnormalized bandpowers, \hat{q}.

        Returns
        -------
        p_hat : array_like
            Optimal estimate of bandpower, \hat{p}.
        """
        if M.ndim == 3:
            return np.asarray([np.dot(M[tind, :, :], q[:,tind]) for tind in range(self.Ntimes)]).T
        else:
            return np.dot(M, q)

    def cov_p_hat(self, M, q_cov):
        """
        Covariance estimate between two different band powers p_alpha and p_beta
        given by M_{alpha i} M^*_{beta,j} C_q^{ij} where C_q^{ij} is the
        q-covariance.

        Parameters
        ----------
        M : array_like
            Normalization matrix, M. Ntimes x Ndlys x Ndlys

        q_cov : array_like
            covariance between bandpowers in q_alpha and q_beta, Ntimes x Ndlys x Ndlys
        """
        p_cov = np.zeros_like(q_cov)
        for tnum in range(len(p_cov)):
            p_cov[tnum] = np.einsum('ab,cd,bd->ac', M[tnum], M[tnum], q_cov[tnum])
        return p_cov

    def broadcast_dset_flags(self, spw_ranges=None, time_thresh=0.2,
                             unflag=False, set_time_independent_weights=True):
        """
        For each dataset in self.dset, update the flag_array such that
        the flagging patterns are time-independent for each baseline given
        a selection for spectral windows.

        For each frequency pixel in a selected spw, if the fraction of flagged
        times exceeds time_thresh, then all times are flagged. If it does not,
        the specific integrations which hold flags in the spw are flagged across
        all frequencies in the spw.

        Additionally, one can also unflag the flag_array entirely if desired.

        Note: although technically allowed, this function may give unexpected
        results if multiple spectral windows in spw_ranges have frequency
        overlap.

        Note: it is generally not recommended to set time_thresh > 0.5, which
        could lead to substantial amounts of data being flagged.

        Parameters
        ----------
        spw_ranges : list of tuples
            list of len-2 spectral window tuples, specifying the start (inclusive)
            and stop (exclusive) index of the frequency array for each spw.
            Default is to use the whole band.

        time_thresh : float
            Fractional threshold of flagged pixels across time needed to flag
            all times per freq channel. It is not recommend to set this greater
            than 0.5.

        unflag : bool
            If True, unflag all data in the spectral window.
        """
        # validate datasets
        self.validate_datasets()

        # clear matrix cache (which may be holding weight matrices Y)
        self.clear_cache()
        # spw type check
        if spw_ranges is None:
            spw_ranges = [(0, self.Nfreqs)]
        assert isinstance(spw_ranges, list), \
            "spw_ranges must be fed as a list of tuples"
        backup_flags = []
        # iterate over datasets
        for dset in self.dsets:
            # iterate over spw ranges
            backup_flags.append(copy.deepcopy(dset.flag_array))
            for spw in spw_ranges:
                self.set_spw(spw, set_Ndlys=False)
                # unflag
                if unflag:
                    # unflag for all times
                    ext = (self.spw_range[0]-self.filter_extension[0],
                           self.spw_range[1]+self.filter_extension[1])
                    dset.flag_array[:, :, ext[0]:ext[1], :] = False
                    continue
                # enact time threshold on flag waterfalls
                # iterate over polarizations
                for i in range(dset.Npols):
                    # iterate over unique baselines
                    ubl = np.unique(dset.baseline_array)
                    for bl in ubl:
                        # get baseline-times indices
                        bl_inds = np.where(np.in1d(dset.baseline_array, bl))[0]
                        # get flag waterfall
                        flags = dset.flag_array[bl_inds, 0, :, i].copy()
                        Ntimes = float(flags.shape[0])
                        Nfreqs = float(flags.shape[1])
                        # get time- and freq-continguous flags
                        freq_contig_flgs = np.sum(flags, axis=1) / Nfreqs > 0.999999
                        Ntimes_noncontig = np.sum(~freq_contig_flgs, dtype=np.float)
                        # get freq channels where non-contiguous flags exceed threshold
                        exceeds_thresh = np.sum(flags[~freq_contig_flgs], axis=0, dtype=np.float) / Ntimes_noncontig > time_thresh
                        # flag channels for all times that exceed time_thresh
                        dset.flag_array[bl_inds, :, np.where(exceeds_thresh)[0][:, None], i] = True
                        # for pixels that have flags but didn't meet broadcasting limit
                        # flag the integration within the spw
                        flags[:, np.where(exceeds_thresh)[0]] = False
                        ext = (self.spw_range[0]-self.filter_extension[0],
                               self.spw_range[1]+self.filter_extension[1])
                        flag_ints = np.max(flags[:,ext[0]:ext[1]], axis=1)
                        dset.flag_array[bl_inds[flag_ints],: , ext[0]:ext[1], i] = True
                if set_time_independent_weights:
                    for key in self._time_independent_weights:
                        self._time_independent_weights[key] = True #all weights are time independent.
        return backup_flags


    def units(self, little_h=True):
        """
        Return the units of the power spectrum. These are inferred from the
        units reported by the input visibilities (UVData objects).

        Parameters
        ----------
        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        Returns
        -------
        pspec_units : str
            Units of the power spectrum that is returned by pspec().
        """
        # Work out the power spectrum units
        if len(self.dsets) == 0:
            raise IndexError("No datasets have been added yet; cannot "
                             "calculate power spectrum units.")

        # get visibility units
        vis_units = self.dsets[0].vis_units

        # set pspec norm units
        if self.primary_beam is None:
            norm_units = "Hz str [beam normalization not specified]"
        else:
            if little_h:
                h_unit = "h^-3 "
            else:
                h_unit = ""
            norm_units = "{}Mpc^3".format(h_unit)

        return vis_units, norm_units

    def delays(self):
        """
        Return an array of delays, tau, corresponding to the bins of the delay
        power spectrum output by pspec() using self.spw_range to specify the
        spectral window.

        Returns
        -------
        delays : array_like
            Delays, tau. Units: ns.
        """
        # Calculate the delays
        if len(self.dsets) == 0:
            raise IndexError("No datasets have been added yet; cannot "
                             "calculate delays.")
        else:
            return utils.get_delays(self.freqs[self.spw_range[0]:self.spw_range[1]],
                                    n_dlys=self.spw_Ndlys) * 1e9 # convert to ns

    def scalar(self, polpair, little_h=True, num_steps=2000, beam=None,
               taper_override='no_override', exact_norm=False):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units, using self.spw_range to set
        spectral window.

        See arxiv:1304.4991 and HERA memo #27 for details.

        This function uses the state of self.taper in constructing scalar.
        See PSpecData.pspec for details.

        Parameters
        ----------
        polpair: tuple, int, or str
                Which pair of polarizations to compute the beam scalar for,
                e.g. ('pI', 'pI') or ('XX', 'YY'). If string, will assume that
                the specified polarization is to be cross-correlated with
                itself, e.g. 'XX' implies ('XX', 'XX').

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        num_steps : int, optional
                Number of steps to use when interpolating primary beams for
                numerical integral
                Default: 10000

        beam : PSpecBeam object
                Option to use a manually-fed PSpecBeam object instead of using
                self.primary_beam.

        taper_override : str, optional
                Option to override the taper chosen in self.taper (does not
                overwrite self.taper; just applies to this function).
                Default: no_override

        exact_norm : boolean, optional
                If True, scalar would just be the X2Y term, as the beam and
                spectral terms are taken into account while constructing
                power spectrum.

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """
        # make sure polarizations are the same
        if isinstance(polpair, (int, np.integer)):
            polpair = uvputils.polpair_int2tuple(polpair)
        if isinstance(polpair, str):
            polpair = (polpair, polpair)
        if polpair[0] != polpair[1]:
            raise NotImplementedError(
                    "Polarizations don't match. Beam scalar can only be "
                    "calculated for auto-polarization pairs at the moment.")
        pol = polpair[0]

        # set spw_range and get freqs
        freqs = self.freqs[self.spw_range[0]:self.spw_range[1]]
        start = freqs[0]
        end = freqs[0] + np.median(np.diff(freqs)) * len(freqs)

        # Override the taper if desired
        if taper_override == 'no_override':
            taper = self.taper
        else:
            taper = taper_override

        # calculate scalar
        if beam is None:
            scalar = self.primary_beam.compute_pspec_scalar(
                                        start, end, len(freqs), pol=pol,
                                        taper=self.taper, little_h=little_h,
                                        num_steps=num_steps, exact_norm=exact_norm)
        else:
            scalar = beam.compute_pspec_scalar(start, end, len(freqs),
                                               pol=pol, taper=self.taper,
                                               little_h=little_h,
                                               num_steps=num_steps, exact_norm=exact_norm)
        return scalar

    def scalar_delay_adjustment(self, key1=None, key2=None, time_index=None, sampling=False,
                                Gv=None, Hv=None):
        """
        Computes an adjustment factor for the pspec scalar that is needed
        when the number of delay bins is not equal to the number of
        frequency channels.

        This adjustment is necessary because
        \sum_gamma tr[Q^alt_alpha Q^alt_gamma] = N_freq**2
        is something that is true only when N_freqs = N_dlys.

        If the data weighting is equal to "identity",
        then the result is independent of alpha, but is
        no longer given by N_freq**2. (Nor is it just N_dlys**2!)

        If the data weighting is not equal to "identity" then
        we generally need a separate scalar adjustment for each
        alpha.

        This function uses the state of self.taper in constructing adjustment.
        See PSpecData.pspec for details.

        Parameters
        ----------
        key1, key2 : tuples or lists of tuples, optional
            Tuples containing indices of dataset and baselines for the two
            input datavectors. If a list of tuples is provided, the baselines
            in the list will be combined with inverse noise weights. If Gv and
            Hv are specified, these arguments will be ignored. Default: None.
        time_index : int, optional
            the index of the time to calculate scaler_delay_adjustment for
            (it is flagging dependent).
        sampling : boolean, optional
            Whether to sample the power spectrum or to assume integrated
            bands over wide delay bins. Default: False
        Gv, Hv : array_like, optional
            If specified, use these arrays instead of calling self.get_G() and
            self.get_H(). Using precomputed Gv and Hv will speed up this
            function significantly. Default: None.

        Returns
        -------
        adjustment : float if the data_weighting is 'identity'
                     1d array of floats with length spw_Ndlys otherwise.
        """
        if Gv is None or Hv is None:
            assert isinstance(time_index,(int,np.int64, np.int32)),"Must provide valid time index! None supplied!"
        if Gv is None: Gv = self.get_G(key1, key2, time_indices=[time_index])[0]
        if Hv is None: Hv = self.get_H(key1, key2, time_indices=[time_index])[0]

        # get ratio
        summed_G = np.sum(Gv, axis=1)
        summed_H = np.sum(Hv, axis=1)
        ratio = summed_H.real / summed_G.real

        # fill infs and nans from zeros in summed_G
        ratio[np.isnan(ratio)] = 1.0
        ratio[np.isinf(ratio)] = 1.0

        ## XXX: Adjustments like this are hacky and wouldn't be necessary
        ## if we deprecate the incorrectly normalized
        ## Q and M matrix definitions.
        #In the future, we need to do our normalizations properly and
        #stop introducing arbitrary normalization factors.
        #if the input identity weighting is diagonal, then the
        #adjustment factor is independent of alpha.
        # get mean ratio.
        if self.data_weighting == 'identity':
            mean_ratio = np.mean(ratio)
            scatter = np.abs(ratio - mean_ratio)
            if (scatter > 10**-4 * mean_ratio).any():
                raise ValueError("The normalization scalar is band-dependent!")
            adjustment = self.spw_Ndlys / (self.spw_Nfreqs * mean_ratio)
        #otherwise, the adjustment factor is dependent on alpha.
        else:
            adjustment = self.spw_Ndlys / (self.spw_Nfreqs * ratio)
        if self.taper != 'none':
            tapering_fct = dspec.gen_window(self.taper, self.spw_Nfreqs)
            adjustment *= np.mean(tapering_fct**2)

        return adjustment

    def validate_pol(self, dsets, pol_pair):
        """
        Validate polarization and returns the index of the datasets so that
        the polarization pair is consistent with the UVData objects.

        Parameters
        ----------
        dsets : length-2 list or length-2 tuple of integers or str
            Contains indices of self.dsets to use in forming power spectra,
            where the first index is for the Left-Hand dataset and second index
            is used for the Right-Hand dataset (see above).

        pol_pair : length-2 tuple of integers or strings
            Contains polarization pair which will be used in estiamting the power
            spectrum e,g (-5, -5) or  ('xy', 'xy'). Only auto-polarization pairs
            are implemented for the time being.

        Returns
        -------
        valid : boolean
            True if the UVData objects polarizations are consistent with the
            pol_pair (user specified polarizations) else False.
        """
        err_msg = "polarization must be fed as len-2 tuple of strings or ints"
        assert isinstance(pol_pair, tuple), err_msg

        # take x_orientation from first dset
        x_orientation = self.dsets[0].x_orientation

        # convert elements to integers if fed as strings
        if isinstance(pol_pair[0], (str, np.str)):
            pol_pair = (uvutils.polstr2num(pol_pair[0], x_orientation=x_orientation), pol_pair[1])
        if isinstance(pol_pair[1], (str, np.str)):
            pol_pair = (pol_pair[0], uvutils.polstr2num(pol_pair[1], x_orientation=x_orientation))

        assert isinstance(pol_pair[0], (int, np.integer)), err_msg
        assert isinstance(pol_pair[1], (int, np.integer)), err_msg

        #if pol_pair[0] != pol_pair[1]:
        #    raise NotImplementedError("Only auto/equal polarizations are implement at the moment.")

        dset_ind1 = self.dset_idx(dsets[0])
        dset_ind2 = self.dset_idx(dsets[1])
        dset1 = self.dsets[dset_ind1]  # first UVData object
        dset2 = self.dsets[dset_ind2]  # second UVData object

        valid = True
        if pol_pair[0] not in dset1.polarization_array:
            print("dset {} does not contain data for polarization {}".format(dset_ind1, pol_pair[0]))
            valid = False

        if pol_pair[1] not in dset2.polarization_array:
            print("dset {} does not contain data for polarization {}".format(dset_ind2, pol_pair[1]))
            valid = False

        return valid

    def pspec(self, bls1, bls2, dsets, pols, n_dlys=None,
              input_data_weight='identity', norm='I', taper='none', exclude_flagged_edge_channels=False,
              sampling=False, little_h=True, spw_ranges=None, symmetric_taper=True, Nspws=1, fullband_filter=False,
              baseline_tol=1.0, store_cov=False, store_cov_diag=False, return_q=False, store_window=True, verbose=True,
              filter_extensions=None, exact_norm=False, history='', r_params=None, standoff=None, suppression_factor=None,
              cov_model='empirical', r_cov_model='empirical', known_cov=None,allow_fft=False, rcond=1e-15):
        """
        Estimate the delay power spectrum from a pair of datasets contained in
        this object, using the optimal quadratic estimator of arXiv:1502.06016.

        In this formulation, the power spectrum is proportional to the
        visibility data via

        P = M data_{LH} E data_{RH}

        where E contains the data weighting and FT matrices, M is a
        normalization matrix, and the two separate datasets are denoted as
        "left-hand" and "right-hand".

        Each power spectrum is generated by taking a baseline (specified by
        an antenna-pair and polarization key) from bls1 out of dsets[0] and
        assigning it as data_LH, and a bl from bls2 out of the dsets[1] and
        assigning it as data_RH.

        If the bl chosen from bls1 is (ant1, ant2) and the bl chosen from bls2
        is (ant3, ant4), the "baseline-pair" describing their cross
        multiplication is ((ant1, ant2), (ant3, ant4)).

        Parameters
        ----------
        bls1, bls2 : list
            List of baseline groups, each group being a list of ant-pair tuples.

        dsets : length-2 tuple or list
            Contains indices of self.dsets to use in forming power spectra,
            where the first index is for the Left-Hand dataset and second index
            is used for the Right-Hand dataset (see above).

        pols : tuple or list of tuple
            Contains polarization pairs to use in forming power spectra
            e.g. ('XX','XX') or [('XX','XX'),('pI','pI')] or a list of
            polarization pairs. Individual strings are also supported, and will
            be expanded into a matching pair of polarizations, e.g. 'xx'
            becomes ('xx', 'xx').

            If a primary_beam is specified, only equal-polarization pairs can
            be cross-correlated, as the beam scalar normalization is only
            implemented in this case. To obtain unnormalized spectra for pairs
            of different polarizations, set the primary_beam to None.

        n_dlys : list of integer, optional
            The number of delay bins to use. The order in the list corresponds
            to the order in spw_ranges.
            Default: None, which then sets n_dlys = number of frequencies.

        input_data_weight : str, optional
            String specifying which weighting matrix to apply to the input
            data. See the options in the set_R() method for details.
            Default: 'identity'.

        norm : str, optional
            String specifying how to choose the normalization matrix, M. See
            the 'mode' argument of get_MW() for options. Default: 'I'.

        taper : str, optional
            Tapering (window) function to apply to the data. Takes the same
            arguments as uvtools.dspec.gen_window(). Default: 'none'.

        sampling : boolean, optional
            Whether output pspec values are samples at various delay bins
            or are integrated bandpowers over delay bins. Default: False

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc or Mpc
            Default: h^-1 Mpc

        spw_ranges : list of tuples, optional
            A list of spectral window channel ranges to select within the total
            bandwidth of the datasets, each of which forms an independent power
            spectrum estimate. Example: [(220, 320), (650, 775)].
            Each tuple should contain a start (inclusive) and stop (exclusive)
            channel used to index the `freq_array` of each dataset. The default
            (None) is to use the entire band provided in each dataset.

        symmetric_taper : bool, optional
            Specify if taper should be applied symmetrically to K-matrix (if true)
            or on the left (if False). Default: True

        baseline_tol : float, optional
            Distance tolerance for notion of baseline "redundancy" in meters.
            Default: 1.0.

        store_cov : bool, optional
            If True, calculate an analytic covariance between bandpowers
            given an input visibility noise model, and store the output
            in the UVPSpec object.

        store_cov_diag : bool, optional
            If True, store the square root of the diagonal of the output covariance matrix
            calculated by using get_analytic_covariance(). The error bars will
            be stored in the form of: sqrt(diag(cov_array_real)) + 1.j*sqrt(diag(cov_array_imag)).
            It's a way to save the disk space since the whole cov_array data with a size of Ndlys x Ndlys x Ntimes x Nblpairs x Nspws
            is too large.

        return_q : bool, optional
            If True, return the results (delay spectra and covariance
            matrices) for the unnormalized bandpowers in the UVPSpec object.

        store_window : bool, optional
            If True, store the window function of the bandpowers.
            Default: True

        cov_model : string, optional
            Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos', 'foreground_dependent',
            (other model names in known_cov)]
            In 'dsets' mode, error bars are estimated from user-provided per baseline and per channel standard deivations.
            In 'empirical' mode, error bars are estimated from the data by averaging the
            channel-channel covariance of each baseline over time and
            then applying the appropriate linear transformations to these
            frequency-domain covariances.
            In 'autos' mode, the covariances of the input data
            over a baseline is estimated from the autocorrelations of the two antennas forming the baseline
            across channel bandwidth and integration time.
            In 'foreground_dependent' mode, it involves using auto-correlation amplitudes to model the input noise covariance
            and visibility outer products to model the input systematics covariance.
            For more details see ds.get_analytic_covariance().

        known_cov : dicts of input covariance matrices
            known_cov has a type {Ckey:covariance}, which is the same with
            ds._C. The matrices stored in known_cov are constructed
            outside the PSpecData object, different from those in ds._C which are constructed
            internally.
            The Ckey should conform to
            (dset_pair_index, blpair_int, model, time_index, conj_1, conj_2),
            e.g. ((0, 1), ((25,37,"xx"), (25, 37, "xx")), 'empirical', False, True),
            while covariance are ndarrays with shape (Nfreqs, Nfreqs).
            Also see PSpecData.set_C() for more details.

        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.

        filter_extensions : list of 2-tuple or 2-list, optional
            Set number of channels to extend filtering width.

        exact_norm : bool, optional
            If True, estimates power spectrum using Q instead of Q_alt
            (HERA memo #44). The default options is False. Beware that
            computing a power spectrum when exact_norm is set to
            False runs two times faster than setting it to True.

        history : str, optional
            history string to attach to UVPSpec object

        r_params: dictionary with parameters for weighting matrix.
                  Proper fields
                  and formats depend on the mode of data_weighting.
                data_weighting == 'dayenu':
                                dictionary with fields
                                'filter_centers', list of floats (or float) specifying the (delay) channel numbers
                                                  at which to center filtering windows. Can specify fractional channel number.
                                'filter_half_widths', list of floats (or float) specifying the width of each
                                                 filter window in (delay) channel numbers. Can specify fractional channel number.
                                'filter_factors', list of floats (or float) specifying how much power within each filter window
                                                  is to be suppressed.
                Absence of r_params dictionary will result in an error!

        allow_fft : bool, optional
                Whether to use a fast FFT summation trick to construct q_hat
                along with H, M, W, G, and V matrices. If False,
                use (significantly slower) brute-force matrix multiplication.
                The FFT method assumes
                a delta-fn bin in delay space. It also only works if the number
                of delay bins is equal to the number of frequencies. Default: False.

        Returns
        -------
        uvp : UVPSpec object
            Instance of UVPSpec that holds the normalized output power spectrum
            data.

        Examples
        --------
        *Example 1:* No grouping; i.e. each baseline is its own group, no
        brackets needed for each bl. If::

            A = (1, 2); B = (2, 3); C = (3, 4); D = (4, 5); E = (5, 6); F = (6, 7)

        and::

            bls1 = [ A, B, C ]
            bls2 = [ D, E, F ]

        then::

            blpairs = [ (A, D), (B, E), (C, F) ]

        *Example 2:* Grouping; blpairs come in lists of blgroups, which are
        considered "grouped" in OQE.
        If::

            bls1 = [ [A, B], [C, D] ]
            bls2 = [ [C, D], [E, F] ]

        then::

            blpairs = [ [(A, C), (B, D)], [(C, E), (D, F)] ]

        *Example 3:* Mixed grouping; i.e. some blpairs are grouped, others are
        not. If::

            bls1 = [ [A, B], C ]
            bls2 = [ [D, E], F ]

        then::

            blpairs = [ [(A, D), (B, E)], (C, F)]

        """
        if r_params is None:
            r_params = {}
        # set taper and data weighting
        self.set_taper(taper)
        self.set_symmetric_taper(symmetric_taper)
        self.set_weighting(input_data_weight, r_cov_model=r_cov_model)

        # Validate the input data to make sure it's sensible
        self.validate_datasets(verbose=verbose)

        # Currently the "pspec normalization scalar" doesn't work if a
        # non-identity data weighting AND a non-trivial taper are used
        if taper != 'none' and input_data_weight != 'identity':
            raise_warning("Warning: Scalar power spectrum normalization "
                                  "doesn't work with current implementation "
                                  "if the tapering AND non-identity "
                                  "weighting matrices are both used.",
                                  verbose=verbose)

        # get datasets
        assert isinstance(dsets, (list, tuple)), \
            "dsets must be fed as length-2 tuple of integers"
        assert len(dsets) == 2, "len(dsets) must be 2"
        assert isinstance(dsets[0], (int, np.integer)) \
            and isinstance(dsets[1], (int, np.integer)), \
                "dsets must contain integer indices"
        dset1 = self.dsets[self.dset_idx(dsets[0])]
        dset2 = self.dsets[self.dset_idx(dsets[1])]

        # assert form of bls1 and bls2
        assert isinstance(bls1, list), \
            "bls1 and bls2 must be fed as a list of antpair tuples"
        assert isinstance(bls2, list), \
            "bls1 and bls2 must be fed as a list of antpair tuples"
        assert len(bls1) == len(bls2) and len(bls1) > 0, \
            "length of bls1 must equal length of bls2 and be > 0"

        for i in range(len(bls1)):
            if isinstance(bls1[i], tuple):
                assert isinstance(bls2[i], tuple), \
                    "bls1[{}] type must match bls2[{}] type".format(i, i)
            else:
                assert len(bls1[i]) == len(bls2[i]), \
                    "len(bls1[{}]) must match len(bls2[{}])".format(i, i)


        # construct list of baseline pairs
        bl_pairs = []
        for i in range(len(bls1)):
            if isinstance(bls1[i], tuple):
                bl_pairs.append( (bls1[i], bls2[i]) )
            elif isinstance(bls1[i], list) and len(bls1[i]) == 1:
                bl_pairs.append( (bls1[i][0], bls2[i][0]) )
            else:
                bl_pairs.append(
                    [ (bls1[i][j], bls2[i][j]) for j in range(len(bls1[i])) ] )

        # validate bl-pair redundancy
        validate_blpairs(bl_pairs, dset1, dset2, baseline_tol=baseline_tol)

        if exclude_flagged_edge_channels:
            # find max and min unflagged channels.
            unflagged_channels = np.zeros(self.dsets[0].Nfreqs, dtype=bool)
            for dsnum in range(len(self.dsets)):
                unflagged_channels = unflagged_channels | np.any(np.any(~self.dsets[dsnum].flag_array[:, 0, :, :], axis=-1), axis=0).squeeze()
            if np.any(unflagged_channels):
                min_chan = np.where(unflagged_channels)[0].min()
                max_chan = np.where(unflagged_channels)[0].max() + 1
            else:
                min_chan = 0
                max_chan = self.Nfreqs
        else:
            min_chan = 0
            max_chan = self.Nfreqs
        # configure spectral window selections
        if spw_ranges is None:
            # now divide up range between min and max chans
            # into Nspw windows
            spw_width = (max_chan - min_chan) // Nspws
            spw_ranges = [[min_chan + spw_width * m, min_chan + spw_width * (m+1)] for m in range(Nspws)]
            # extend final spw to include max chan
            spw_ranges[-1][-1] = max_chan
            spw_ranges = [tuple(spw) for spw in spw_ranges]

        if isinstance(spw_ranges, tuple):
            spw_ranges = [spw_ranges,]

        if filter_extensions is None:
            if fullband_filter:
                filter_extensions = [(spw[0] - chan_min, chan_max - spw[1]) for spw in spw_ranges]
            else:
                filter_extensions = [(0, 0) for m in range(len(spw_ranges))]
        # convert to list if only a tuple was given
        if isinstance(filter_extensions, tuple):
            filter_extensions = [filter_extensions,]
        # truncate filter extensions to only extend to max_chan and min_chan.
        for spw in range(len(spw_ranges)):
            fx = list(filter_extensions[spw])
            fx[0] = np.min([spw_ranges[spw][0] - min_chan, filter_extensions[spw][0]])
            fx[1] = np.min([max_chan - spw_ranges[spw][1], filter_extensions[spw][1]])
            filter_extensions[spw] = tuple(fx)


        assert len(spw_ranges) == len(filter_extensions), "must provide same number of spw_ranges as filter_extensions"

        # Check that spw_ranges is list of len-2 tuples
        assert np.isclose([len(t) for t in spw_ranges], 2).all(), \
                "spw_ranges must be fed as a list of length-2 tuples"

        # if using default setting of number of delay bins equal to number
        # of frequency channels
        if n_dlys is None:
            n_dlys = [ int(np.abs(spw_ranges[i][1] - spw_ranges[i][0])) for i in range(len(spw_ranges))]
        elif isinstance(n_dlys, (int, np.integer)):
            n_dlys = [n_dlys]

        # if using the whole band in the dataset, then there should just be
        # one n_dly parameter specified
        if spw_ranges is None and n_dlys != None:
            assert len(n_dlys) == 1, \
                "Only one spw, so cannot specify more than one n_dly value"

        # assert that the same number of ndlys has been specified as the
        # number of spws
        assert len(spw_ranges) == len(n_dlys), \
            "Need to specify number of delay bins for each spw"

        # It's nice to have the stats array and the cov array available!
        #if store_cov_diag and store_cov:
            #store_cov = False
            # Only store diagnonal parts of the cov_array to save the disk space if store_cov_diag==True,
            # no matter what the initial choice for store_cov.
        #check that the number of frequencies in each spectral window
        #equals the number of delays
        if allow_fft:
            #for spw, ndly, fext in zip(spw_ranges, n_dlys ,filter_extensions):
            #    nf_spw = spw[1]-spw[0] + np.sum(filter_extensions)
            #    if not nf_spw == ndly:
            #        raise ValueError("allow_fft is True! Number of delays in each spw must equal the number of frequencies in each spw.")
            if not sampling:
                raise ValueError("allow_fft is True! Sampling must also be set to True for allow_fft!")

        # setup polarization selection
        if isinstance(pols, (tuple, str)): pols = [pols]

        # convert all polarizations to integers if fed as strings
        _pols = []
        for p in pols:
            if isinstance(p, str):
                # Convert string to pol-integer pair
                p = (uvutils.polstr2num(p, x_orientation=self.dsets[0].x_orientation),
                     uvutils.polstr2num(p, x_orientation=self.dsets[0].x_orientation))
            if isinstance(p[0], (str, np.str)):
                p = (uvutils.polstr2num(p[0], x_orientation=self.dsets[0].x_orientation), p[1])
            if isinstance(p[1], (str, np.str)):
                p = (p[0], uvutils.polstr2num(p[1], x_orientation=self.dsets[0].x_orientation))
            _pols.append(p)
        pols = _pols

        # initialize empty lists
        data_array = odict()
        wgt_array = odict()
        integration_array = odict()
        cov_array_real = odict()
        cov_array_imag = odict()
        stats_array_cov_model = odict()
        window_function_array = odict()
        time1 = []
        time2 = []
        lst1 = []
        lst2 = []
        dly_spws = []
        freq_spws = []
        dlys = []
        freqs = []
        sclr_arr = []
        blp_arr = []
        bls_arr = []
        self.cov_model = cov_model
        # Loop over spectral windows
        for i in range(len(spw_ranges)):
            # set spectral range
            if verbose:
                print( "\nSetting spectral range: {}".format(spw_ranges[i]))
            self.set_spw(spw_ranges[i], ndlys=n_dlys[i])
            self.set_filter_extension(filter_extensions[i])

            # clear covariance cache
            self.clear_cache()

            # setup empty data arrays
            spw_data = []
            spw_wgts = []
            spw_ints = []
            spw_scalar = []
            spw_polpair = []
            spw_cov_real = []
            spw_cov_imag = []
            spw_stats_array_cov_model = []
            spw_window_function = []

            d = self.delays() * 1e-9
            f = dset1.freq_array.flatten()[spw_ranges[i][0]:spw_ranges[i][1]]
            dlys.extend(d)
            dly_spws.extend(np.ones_like(d, np.int16) * i)
            freq_spws.extend(np.ones_like(f, np.int16) * i)
            freqs.extend(f)

            # Loop over polarizations
            for j, p in enumerate(pols):
                p_str = tuple([uvutils.polnum2str(_p) for _p in p])
                if verbose: print( "\nUsing polarization pair: {}".format(p_str))

                # validating polarization pair on UVData objects
                valid = self.validate_pol(dsets, tuple(p))
                if not valid:
                   # Polarization pair is invalid; skip
                   print("Polarization pair: {} failed the validation test, "
                         "continuing...".format(p_str))
                   continue

                spw_polpair.append( uvputils.polpair_tuple2int(p) )
                pol_data = []
                pol_wgts = []
                pol_ints = []
                pol_cov_real = []
                pol_cov_imag = []
                pol_stats_array_cov_model = []
                pol_window_function = []

                # Compute scalar to convert "telescope units" to "cosmo units"
                if self.primary_beam is not None:

                    # Raise error if cross-pol is requested
                    if (p[0] != p[1]):
                        raise NotImplementedError(
                            "Visibilities with different polarizations can only "
                            "be cross-correlated if primary_beam = None. Cannot "
                            "compute beam scalar for mixed polarizations.")

                    # using zero'th indexed polarization, as cross-polarized
                    # beams are not yet implemented
                    if norm == 'H^-1':
                        # If using decorrelation, the H^-1 normalization
                        # already deals with the taper, so we need to override
                        # the taper when computing the scalar
                        scalar = self.scalar(p, little_h=little_h,
                                             taper_override='none',
                                             exact_norm=exact_norm)
                    else:
                        scalar = self.scalar(p, little_h=little_h,
                                exact_norm=exact_norm)
                else:
                    raise_warning("Warning: self.primary_beam is not defined, "
                                  "so pspectra are not properly normalized",
                                  verbose=verbose)
                    scalar = 1.0

                pol = (p[0]) # used in get_integral_beam function to specify the correct polarization for the beam
                spw_scalar.append(scalar)

                # Loop over baseline pairs
                for k, blp in enumerate(bl_pairs):
                    # assign keys
                    if isinstance(blp, list):
                        # interpet blp as group of baseline-pairs
                        raise NotImplementedError("Baseline lists bls1 and bls2"
                                " must be lists of tuples (not lists of lists"
                                " of tuples).\n"
                                "Use hera_pspec.pspecdata.construct_blpairs()"
                                " to construct appropriately grouped baseline"
                                " lists.")
                        #key1 = [(dsets[0],) + _blp[0] + (p[0],) for _blp in blp]
                        #key2 = [(dsets[1],) + _blp[1] + (p[1],) for _blp in blp]
                    elif isinstance(blp, tuple):
                        # interpret blp as baseline-pair
                        key1 = (dsets[0],) + blp[0] + (p_str[0],)
                        key2 = (dsets[1],) + blp[1] + (p_str[1],)

                    if verbose:
                        print("\n(bl1, bl2) pair: {}\npol: {}".format(blp, tuple(p)))

                    # Check that number of non-zero weight chans >= n_dlys
                    key1_dof = np.sum(~np.isclose(self.w(key1).T, 0.0), axis=1)
                    key2_dof = np.sum(~np.isclose(self.w(key2).T, 0.0), axis=1)
                    if np.any(key1_dof < self.spw_Ndlys) or np.any(key2_dof < self.spw_Ndlys):
                        if verbose:
                            print("WARNING: Number of unflagged chans for key1 "
                                  "and/or key2 < n_dlys\n which may lead to "
                                  "normalization instabilities.")
                    #if using inverse sinc weighting, set r_params
                    if input_data_weight == 'dayenu':
                        key1 = (dsets[0],) + blp[0] + (p_str[0],)
                        key2 = (dsets[1],) + blp[1] + (p_str[1],)
                        if not key1 in r_params:
                            if standoff is None or suppression_factor is None:
                                raise ValueError("No r_param dictionary supplied"
                                                 " for baseline %s"%(str(key1)))
                            else:
                                antpos, ants = self.dsets[dsets[0]].get_ENU_antpos()
                                p1 = antpos[np.where(ants==blp[0][0])[0][0]]
                                p2 = antpos[np.where(ants==blp[0][1])[0][0]]
                                bl_dly = float(np.linalg.norm(p2 - p1) / conversions.units.c)
                                rp1 = {'filter_half_widths':[standoff * 1e-9 + bl_dly],
                                                  'filter_centers':[0.0],
                                                  'filter_factors':[float(suppression_factor)]}
                        else:
                            rp1 = r_params[key1]
                        if not key2 in r_params:
                            if standoff is None or suppression_factor is None:
                                raise ValueError("No r_param dictionary supplied"
                                                 " for baseline %s"%(str(key2)))
                            else:
                                antpos, ants = self.dsets[dsets[1]].get_ENU_antpos()
                                p1 = antpos[np.where(ants==blp[1][0])[0][0]]
                                p2 = antpos[np.where(ants==blp[1][1])[0][0]]
                                bl_dly = float(np.linalg.norm(p2 - p1) / conversions.units.c)
                                rp2 = {'filter_half_widths':[standoff * 1e-9 + bl_dly],
                                                  'filter_centers':[0.0],
                                                  'filter_factors':[float(suppression_factor)]}
                        else:
                            rp1 = r_params[key2]

                        self.set_r_param(key1, rp1)
                        self.set_r_param(key2, rp2)

                    if verbose: print("  Building G...")
                    Gv = self.get_G(key1, key2, exact_norm=exact_norm, pol = pol, allow_fft=allow_fft)
                    Hv = self.get_H(key1, key2, sampling=sampling, exact_norm=exact_norm, pol = pol, allow_fft=allow_fft)

                    # Calculate unnormalized bandpowers
                    if verbose: print("  Building q_hat...")
                    qv = self.q_hat(key1, key2, exact_norm=exact_norm, pol=pol, allow_fft=allow_fft)

                    if verbose: print("  Normalizing power spectrum...")
                    #if norm == 'V^-1/2':
                    #    V_mat = self.cov_q_hat(key1, key2, exact_norm=exact_norm, pol = pol, model=cov_model)
                        #Mv, Wv = self.get_MW(Gv, Hv, mode=norm, band_covar=V_mat, exact_norm=exact_norm)
                        #Mv, Wv = self.get_MW(Gv, Hv, mode=norm, band_covar=V_mat, exact_norm=exact_norm)
                    #else:
                    #    Mv, Wv = self.get_MW(Gv, Hv, mode=norm, exact_norm=exact_norm)
                    Mv = self.get_M(key1, key2, mode=norm, sampling=sampling, exact_norm=exact_norm, pol=pol,
                                    allow_fft=allow_fft, rcond=rcond)
                    pv = self.p_hat(Mv, qv)
                    Wv = self.get_W(key1, key2, mode=norm, sampling=sampling, exact_norm=exact_norm, pol=pol,
                                    allow_fft=allow_fft, rcond=rcond)
                    # Multiply by scalar
                    if self.primary_beam != None:
                        if verbose: print("  Computing and multiplying scalar...")
                        pv *= scalar

                    # Wide bin adjustment of scalar, which is only needed for
                    # the diagonal norm matrix mode (i.e., norm = 'I')
                    if norm == 'I' and not(exact_norm):
                        for t in range(self.Ntimes):
                            sa = self.scalar_delay_adjustment(Gv=Gv[t], Hv=Hv[t], sampling=sampling)
                            pv[:,t] = pv[:,t] * sa

                    #Generate the covariance matrix if error bars provided
                    if store_cov or store_cov_diag:
                        if verbose: print(" Building q_hat covariance...")
                        if cov_model != 'empirical_pspec':
                            cov_q_real, cov_q_imag, cov_real, cov_imag \
                                = self.get_analytic_covariance(key1, key2, Mv,
                                                               exact_norm=exact_norm,
                                                               pol=pol,
                                                               model=cov_model,
                                                               known_cov=known_cov, )

                            if self.primary_beam != None:
                                cov_real = cov_real * (scalar)**2.
                                cov_imag = cov_imag * (scalar)**2.

                            if norm == 'I' and not(exact_norm):
                                if isinstance(sa, (np.float, float)):
                                    cov_real = cov_real * (sa)**2.
                                    cov_imag = cov_imag * (sa)**2.
                                else:
                                    cov_real = cov_real * np.outer(sa, sa)[None]
                                    cov_imag = cov_imag * np.outer(sa, sa)[None]
                        else:
                            # empirical_pspec mode just lets us compute covs from data directly.
                            pwgts = (~np.all(np.isclose(pv, 0), axis=0)).astype(float)
                            if not np.all(pwgts == 0.):
                                cov_real = np.cov(pv.real, aweights=pwgts)
                                cov_imag = np.cov(pv.imag, aweights=pwgts)
                            else:
                                cov_real=np.zeros((self.spw_Ndlys, self.spw_Ndlys))
                                cov_imag=np.zeros((self.spw_Ndlys, self.spw_Ndlys))

                            qwgts = (~np.all(np.isclose(pv, 0), axis=0)).astype(float)
                            if not np.all(qwgts == 0.):
                                cov_q_real = np.cov(qv.real, aweights=qwgts)
                                cov_q_imag = np.cov(qv.imag, aweights=qwgts)
                            else:
                                cov_q_real=np.zeros((self.spw_Ndlys, self.spw_Ndlys))
                                cov_q_imag=np.zeros((self.spw_Ndlys, self.spw_Ndlys))
                            # broadcast in time

                            cov_real = pwgts[:, None, None] * np.asarray([cov_real for t in range(self.Ntimes)])
                            cov_imag = pwgts[:, None, None] * np.asarray([cov_imag for t in range(self.Ntimes)])
                            cov_q_real = qwgts[:, None, None] * np.asarray([cov_q_real for t in range(self.Ntimes)])
                            cov_q_imag = qwgts[:, None, None] * np.asarray([cov_q_imag for t in range(self.Ntimes)])

                        if not return_q:
                            if store_cov:
                                pol_cov_real.extend(np.real(cov_real).astype(np.float64))
                                pol_cov_imag.extend(np.real(cov_imag).astype(np.float64))
                            if store_cov_diag:
                                stats = np.sqrt(np.diagonal(np.real(cov_real), axis1=1, axis2=2)) + 1.j*np.sqrt(np.diagonal(np.real(cov_imag), axis1=1, axis2=2))
                                pol_stats_array_cov_model.extend(stats)
                        else:
                            if store_cov:
                                pol_cov_real.extend(np.real(cov_q_real).astype(np.float64))
                                pol_cov_imag.extend(np.real(cov_q_imag).astype(np.float64))
                            if store_cov_diag:
                                stats = np.sqrt(np.diagonal(np.real(cov_q_real), axis1=1, axis2=2)) + 1.j*np.sqrt(np.diagonal(np.real(cov_q_imag), axis1=1, axis2=2))
                                pol_stats_array_cov_model.extend(stats)

                    # store the window_function
                    pol_window_function.extend(Wv.astype(np.float64))

                    # Get baseline keys
                    if isinstance(blp, list):
                        bl1 = blp[0][0]
                        bl2 = blp[0][1]
                    else:
                        bl1 = blp[0]
                        bl2 = blp[1]

                    # append bls
                    bls_arr.extend([bl1, bl2])

                    # insert pspectra
                    if not return_q:
                        pol_data.extend(pv.T)
                    else:
                        pol_data.extend(qv.T)

                    # get weights
                    wgts1 = self.w(key1).T
                    wgts2 = self.w(key2).T

                    # get avg of nsample across frequency axis, weighted by wgts
                    # we may want to use the total integration time, rather then the average.
                    nsamp1 = np.sum(dset1.get_nsamples(bl1 + (p[0],))[:, slice(*self.get_spw())] * wgts1, axis=1) \
                             / np.sum(wgts1, axis=1).clip(1, np.inf)
                    nsamp2 = np.sum(dset2.get_nsamples(bl2 + (p[1],))[:, slice(*self.get_spw())] * wgts2, axis=1) \
                             / np.sum(wgts2, axis=1).clip(1, np.inf)

                    # if the sum of wgts is zero, then set nsamps to zero!
                    nsamp1[np.isclose(np.sum(wgts1, axis=1), 0.)] = 0.
                    nsamp2[np.isclose(np.sum(wgts2, axis=1), 0.)] = 0.


                    # get integ1
                    blts1 = dset1.antpair2ind(bl1, ordered=False)
                    integ1 = dset1.integration_time[blts1] * nsamp1

                    # get integ2
                    blts2 = dset2.antpair2ind(bl2, ordered=False)
                    integ2 = dset2.integration_time[blts2] * nsamp2

                    # take inverse avg of integ1 and integ2 to get total integ
                    # inverse avg is done b/c integ ~ 1/noise_var
                    # and due to non-linear operation of V_1 * V_2
                    pol_ints.extend(1./np.mean([1./integ1, 1./integ2], axis=0))

                    # combined weight is geometric mean
                    pol_wgts.extend(np.concatenate([wgts1[:, :, None],
                                                    wgts2[:, :, None]], axis=2))

                    # insert time and blpair info only once per blpair
                    if i < 1 and j < 1:
                        # insert time info
                        inds1 = dset1.antpair2ind(bl1, ordered=False)
                        inds2 = dset2.antpair2ind(bl2, ordered=False)
                        time1.extend(dset1.time_array[inds1])
                        time2.extend(dset2.time_array[inds2])
                        lst1.extend(dset1.lst_array[inds1])
                        lst2.extend(dset2.lst_array[inds2])

                        # insert blpair info
                        blp_arr.extend(np.ones_like(inds1, np.int) \
                                       * uvputils._antnums_to_blpair(blp))

                # insert into data and wgts integrations dictionaries
                spw_data.append(pol_data)
                spw_wgts.append(pol_wgts)
                spw_ints.append(pol_ints)
                spw_stats_array_cov_model.append(pol_stats_array_cov_model)
                spw_cov_real.append(pol_cov_real)
                spw_cov_imag.append(pol_cov_imag)
                spw_window_function.append(pol_window_function)

            # insert into data and integration dictionaries
            spw_data = np.moveaxis(np.array(spw_data), 0, -1)
            spw_wgts = np.moveaxis(np.array(spw_wgts), 0, -1)
            spw_ints = np.moveaxis(np.array(spw_ints), 0, -1)
            spw_stats_array_cov_model = np.moveaxis(np.array(spw_stats_array_cov_model), 0, -1)
            spw_cov_real = np.moveaxis(np.array(spw_cov_real), 0, -1)
            spw_cov_imag = np.moveaxis(np.array(spw_cov_imag), 0, -1)
            spw_window_function = np.moveaxis(np.array(spw_window_function), 0, -1)

            data_array[i] = spw_data
            stats_array_cov_model[i] = spw_stats_array_cov_model
            cov_array_real[i] = spw_cov_real
            cov_array_imag[i] = spw_cov_imag
            window_function_array[i] = spw_window_function
            wgt_array[i] = spw_wgts
            integration_array[i] = spw_ints
            sclr_arr.append(spw_scalar)

            # raise error if none of pols are consistent with the UVData objects
            if len(spw_polpair) == 0:
                raise ValueError("None of the specified polarization pairs "
                                 "match that of the UVData objects")
            self.set_filter_extension((0,0))
            # set filter_extension to be zero when ending the loop

        # fill uvp object
        uvp = uvpspec.UVPSpec()
        uvp.symmetric_taper=symmetric_taper
        # fill meta-data
        uvp.time_1_array = np.array(time1)
        uvp.time_2_array = np.array(time2)
        uvp.time_avg_array = np.mean([uvp.time_1_array, uvp.time_2_array], axis=0)
        uvp.lst_1_array = np.array(lst1)
        uvp.lst_2_array = np.array(lst2)
        uvp.lst_avg_array = np.mean([np.unwrap(uvp.lst_1_array),
                                     np.unwrap(uvp.lst_2_array)], axis=0) \
                                     % (2*np.pi)
        uvp.blpair_array = np.array(blp_arr)
        uvp.Nblpairs = len(np.unique(blp_arr))
        uvp.Ntimes = len(np.unique(time1))
        uvp.Nblpairts = len(time1)
        bls_arr = sorted(set(bls_arr))
        uvp.bl_array = np.array([uvp.antnums_to_bl(bl) for bl in bls_arr])
        antpos = dict(zip(dset1.antenna_numbers, dset1.antenna_positions))
        uvp.bl_vecs = np.array([antpos[bl[0]] - antpos[bl[1]] for bl in bls_arr])
        uvp.Nbls = len(uvp.bl_array)
        uvp.spw_dly_array = np.array(dly_spws)
        uvp.spw_freq_array = np.array(freq_spws)
        uvp.Nspws = len(np.unique(dly_spws))
        uvp.spw_array = np.arange(uvp.Nspws, dtype=np.int16)
        uvp.freq_array = np.array(freqs)
        uvp.dly_array = np.array(dlys)
        uvp.Ndlys = len(np.unique(dlys))
        uvp.Nspwdlys = len(uvp.spw_dly_array)
        uvp.Nspwfreqs = len(uvp.spw_freq_array)
        uvp.Nfreqs = len(np.unique(freqs))
        uvp.polpair_array = np.array(spw_polpair, np.int)
        uvp.Npols = len(spw_polpair)
        uvp.scalar_array = np.array(sclr_arr)
        uvp.channel_width = dset1.channel_width  # all dsets validated to agree
        uvp.weighting = input_data_weight
        uvp.vis_units, uvp.norm_units = self.units(little_h=little_h)
        uvp.telescope_location = dset1.telescope_location
        filename1 = json.loads(dset1.extra_keywords.get('filename', '""'))
        cal1 = json.loads(dset1.extra_keywords.get('calibration', '""'))
        filename2 = json.loads(dset2.extra_keywords.get('filename', '""'))
        cal2 = json.loads(dset2.extra_keywords.get('calibration', '""'))
        label1 = self.labels[self.dset_idx(dsets[0])]
        label2 = self.labels[self.dset_idx(dsets[1])]
        uvp.labels = sorted(set([label1, label2]))
        uvp.label_1_array = np.ones((uvp.Nspws, uvp.Nblpairts, uvp.Npols), np.int) \
                            * uvp.labels.index(label1)
        uvp.label_2_array = np.ones((uvp.Nspws, uvp.Nblpairts, uvp.Npols), np.int) \
                            * uvp.labels.index(label2)
        uvp.labels = np.array(uvp.labels, np.str)
        uvp.history = "UVPSpec written on {} with hera_pspec git hash {}\n{}\n" \
                      "dataset1: filename: {}, label: {}, cal: {}, history:\n{}\n{}\n" \
                      "dataset2: filename: {}, label: {}, cal: {}, history:\n{}\n{}\n" \
                      "".format(datetime.datetime.utcnow(), version.git_hash, '-'*20,
                                filename1, label1, cal1, dset1.history, '-'*20,
                                filename2, label2, cal2, dset2.history, '-'*20)
        uvp.r_params = uvputils.compress_r_params(self.r_params)
        uvp.taper = taper
        if not return_q:
            uvp.norm = norm
        else:
            uvp.norm = 'Unnormalized'

        if self.primary_beam is not None:
            # attach cosmology
            uvp.cosmo = self.primary_beam.cosmo
            # attach beam info
            uvp.beam_freqs = self.primary_beam.beam_freqs
            uvp.OmegaP, uvp.OmegaPP = \
                                self.primary_beam.get_Omegas(uvp.polpair_array)
            if hasattr(self.primary_beam, 'filename'):
                uvp.beamfile = self.primary_beam.filename

        # fill data arrays
        uvp.data_array = data_array
        uvp.integration_array = integration_array
        uvp.wgt_array = wgt_array
        # set nsamples = 0 where we didn't take data.
        uvp.nsample_array = dict(
                        [ (k, (~np.isclose(uvp.integration_array[k], 0.0)).astype(float))
                         for k in uvp.integration_array.keys() ] )

        if store_cov:
            uvp.cov_array_real = cov_array_real
            uvp.cov_array_imag = cov_array_imag
            uvp.cov_model = cov_model
        if store_cov_diag:
            uvp.stats_array = odict()
            uvp.stats_array[cov_model+"_diag"] = stats_array_cov_model
        if store_window:
            uvp.window_function_array = window_function_array

        # run check
        uvp.check()
        return uvp

    def rephase_to_dset(self, dset_index=0, inplace=True):
        """
        Rephase visibility data in self.dsets to the LST grid of
        dset[dset_index] using hera_cal.utils.lst_rephase.

        Each integration in all other dsets is phased to the center of the
        corresponding LST bin (by index) in dset[dset_index].

        Will only phase if the dataset's phase type is 'drift'. This is because
        the rephasing algorithm assumes the data is drift-phased when applying
        the phasor term.

        Note that PSpecData.Jy_to_mK() must be run *after* rephase_to_dset(),
        if one intends to use the former capability at any point.

        Parameters
        ----------
        dset_index : int or str
            Index or label of dataset in self.dset to phase other datasets to.

        inplace : bool, optional
            If True, edits data in dsets in-memory. Else, makes a copy of
            dsets, edits data in the copy and returns to user.

        Returns
        -------
        if inplace:
            return new_dsets
        else:
            return None
        """
        # run dataset validation
        self.validate_datasets()

        # assign dsets
        if inplace:
            dsets = self.dsets
        else:
            dsets = copy.deepcopy(self.dsets)

        # Parse dset_index
        dset_index = self.dset_idx(dset_index)

        # get LST grid we are phasing to
        lst_grid = []
        lst_array = dsets[dset_index].lst_array.ravel()
        for l in lst_array:
            if l not in lst_grid:
                lst_grid.append(l)
        lst_grid = np.array(lst_grid)

        # get polarization list
        pol_list = dsets[dset_index].polarization_array.tolist()

        # iterate over dsets
        for i, dset in enumerate(dsets):
            # don't rephase dataset we are using as our LST anchor
            if i == dset_index:
                # even though not phasing this dset, must set to match all other
                # dsets due to phasing-check validation
                dset.phase_type = 'unknown'
                continue

            # skip if dataset is not drift phased
            if dset.phase_type != 'drift':
                print("skipping dataset {} b/c it isn't drift phased".format(i))

            # convert UVData to DataContainers. Note this doesn't make
            # a copy of the data
            (data, flgs, antpos, ants, freqs, times, lsts,
             pols) = hc.io.load_vis(dset, return_meta=True)

            # make bls dictionary
            bls = dict([(k, antpos[k[0]] - antpos[k[1]]) for k in data.keys()])

            # Get dlst array
            dlst = lst_grid - lsts

            # get telescope latitude
            lat = dset.telescope_location_lat_lon_alt_degrees[0]

            # rephase
            hc.utils.lst_rephase(data, bls, freqs, dlst, lat=lat)

            # re-insert into dataset
            for j, k in enumerate(data.keys()):
                # get blts indices of basline
                indices = dset.antpair2ind(k[:2], ordered=False)

                # get index in polarization_array for this polarization
                polind = pol_list.index(uvutils.polstr2num(k[-1], x_orientation=self.dsets[0].x_orientation))

                # insert into dset
                dset.data_array[indices, 0, :, polind] = data[k]

            # set phasing in UVData object to unknown b/c there isn't a single
            # consistent phasing for the entire data set.
            dset.phase_type = 'unknown'

        if inplace is False:
            return dsets

    def Jy_to_mK(self, average_conversion=False, beam=None):
        """
        Convert internal datasets from a Jy-scale to mK scale using a primary
        beam model if available. Note that if you intend to rephase_to_dset(),
        Jy to mK conversion must be done *after* that step.

        Parameters
        ----------
        beam : PSpecBeam object
            Beam object.

        average_conversion : bool, optional
            If True, average conversion factor in frequency before multiplying by data.
        """
        # get all unique polarizations of all the datasets
        pols = set(np.ravel([dset.polarization_array for dset in self.dsets]))

        # assign beam
        if beam is None:
            beam = self.primary_beam
        else:
            if self.primary_beam is not None:
                print("Warning: feeding a beam model when self.primary_beam "
                      "already exists...")

        # Check beam is not None
        assert beam is not None, \
            "Cannot convert Jy --> mK b/c beam object is not defined..."

        # assert type of beam
        assert isinstance(beam, pspecbeam.PSpecBeamBase), \
            "beam model must be a subclass of pspecbeam.PSpecBeamBase"

        # iterate over all pols and get conversion factors
        factors = {}
        for p in pols:
            factors[p] = beam.Jy_to_mK(self.freqs, pol=p)
            if average_conversion:
                factors[p] = np.mean(factors[p]) * np.ones_like(factors[p])

        # iterate over datasets and apply factor
        for i, dset in enumerate(self.dsets):
            # check dset vis units
            if dset.vis_units.upper() != 'JY':
                print("Cannot convert dset {} Jy -> mK because vis_units = {}".format(i, dset.vis_units))
                continue
            for j, p in enumerate(dset.polarization_array):
                dset.data_array[:, :, :, j] *= factors[p][None, None, :]
            dset.vis_units = 'mK'

    def trim_dset_lsts(self, lst_tol=6):
        """
        Assuming all datasets in self.dsets are locked to the same LST grid
        (but each may have a constant offset), trim LSTs from each dset that
        aren't found in all other dsets (within some decimal tolerance
        specified by lst_tol).

        Warning: this edits the data in dsets in-place, and is not reversible.

        Parameters
        ----------
        lst_tol : float
            Decimal tolerance [radians] for comparing float-valued LST bins.
        """
        # ensure each dset has same dLST within tolerance / Ntimes
        dlst = np.median(np.diff(np.unique(self.dsets[0].lst_array)))
        for dset in self.dsets:
            _dlst = np.median(np.diff(np.unique(dset.lst_array)))
            if not np.isclose(dlst, _dlst, atol=10**(-lst_tol) / dset.Ntimes):
                print("not all datasets in self.dsets are on the same LST "
                      "grid, cannot LST trim.")
                return

        # get lst array of each dataset, turn into string and add to common_lsts
        lst_arrs = []
        common_lsts = set()
        for i, dset in enumerate(self.dsets):
            lsts = ["{lst:0.{tol}f}".format(lst=l, tol=lst_tol)
                    for l in dset.lst_array]
            lst_arrs.append(lsts)
            if i == 0:
                common_lsts = common_lsts.union(set(lsts))
            else:
                common_lsts = common_lsts.intersection(set(lsts))

        # iterate through dsets and trim off integrations whose lst isn't
        # in common_lsts
        for i, dset in enumerate(self.dsets):
            trim_inds = np.array([l not in common_lsts for l in lst_arrs[i]])
            if np.any(trim_inds):
                self.dsets[i].select(times=dset.time_array[~trim_inds])
        self.Ntimes = self.dsets[i].Ntimes


def pspec_run(dsets, filename, dsets_std=None, cals=None, cal_flag=True,
              groupname=None, dset_labels=None, dset_pairs=None, psname_ext=None,
              spw_ranges=None, n_dlys=None, pol_pairs=None, blpairs=None,
              input_data_weight='identity', norm='I', taper='none', sampling=False,
              exclude_auto_bls=False, exclude_cross_bls=False, exclude_permutations=True,
              Nblps_per_group=None, bl_len_range=(0, 1e10), exclude_flagged_edge_channels=False, Nspws=1,
              bl_deg_range=(0, 180), bl_error_tol=1.0, store_window=True, Jy2mK_avg=False,
              allow_fft=False, time_avg=False, vis_units="UNCALIB", standoff=0.0, suppression_factor=1e-9,
              beam=None, cosmo=None, interleave_times=False, rephase_to_dset=None, rcond=1e-15,
              trim_dset_lsts=False, broadcast_dset_flags=True, external_flags=None, include_autocorrs=False,
              time_thresh=0.2, Jy2mK=False, overwrite=True, symmetric_taper=True, fullband_filter=False,
              file_type='miriad', verbose=True, exact_norm=False, store_cov=False, store_cov_diag=False, filter_extensions=None,
              history='', r_params=None, tsleep=0.1, maxiter=1, return_q=False, known_cov=None, cov_model='empirical'):
    """
    Create a PSpecData object, run OQE delay spectrum estimation and write
    results to a PSpecContainer object.

    Warning: if dsets is a list of UVData objects, they might be edited in place!

    Parameters
    ----------
    dsets : list
        Contains UVData objects or string filepaths to UVData-compatible files

    filename : str
        Output filepath for HDF5 PSpecContainer object

    groupname : str
        Groupname of the subdirectory in the HDF5 container to store the
        UVPSpec objects in. Default is a concatenation the dset_labels.

    dsets_std : list
        Contains UVData objects or string filepaths to miriad files.
        Default is none.

    cals : list
        List of UVCal objects or calfits filepaths. Default is None.

    cal_flag : bool
        If True, use flags in calibration to flag data.

    dset_labels : list
        List of strings to label the input datasets. These labels form
        the psname of each UVPSpec object. Default is "dset0_x_dset1"
        where 0 and 1 are replaced with the dset index in dsets.
        Note: it is not advised to put underscores in the dset label names,
        as some downstream functions use this as a special character.

    dset_pairs : list of len-2 integer tuples
        List of tuples specifying the dset pairs to use in OQE estimation.
        Default is to form all N_choose_2 pairs from input dsets.

    psname_ext : string
        A string extension for the psname in the PSpecContainer object.
        Example: 'group/psname{}'.format(psname_ext)

    spw_ranges : list of len-2 integer tuples
        List of tuples specifying the spectral window range. See
        PSpecData.pspec() for details. Default is the entire band.

    n_dlys : list
        List of integers denoting number of delays to use per spectral window.
        Same length as spw_ranges.

    pol_pairs : list of len-2 tuples
        List of string or integer tuples specifying the polarization
        pairs to use in OQE with each dataset pair in dset_pairs.
        Default is to get all unique pols in the datasets and to form
        all auto-pol pairs. See PSpecData.pspec() for details.

    blpairs : list of tuples
        List of tuples specifying the desired baseline pairs to use in OQE.
        Ex. [((1, 2), (3, 4)), ((1, 2), (5, 6)), ...]
        The first bl in a tuple is drawn from zeroth index of a tuple in
        dset_pairs, while the second bl is drawn from the first index.
        See pspecdata.construct_blpairs for details. If None, the default
        behavior is to use the antenna positions in each UVData object to
        construct lists of redundant baseline groups to to take all
        cross-multiplies in each redundant baseline group.

    input_data_weight : string
        Data weighting to use in OQE. See PSpecData.pspec for details.
        Default: 'identity'

    norm : string
        Normalization scheme to use in OQE. See PSpecData.pspec for details.
        Default: 'I'

    taper : string
        Tapering to apply to data in OQE. See PSpecData.pspec for details.
        Default: 'none'

    sampling : boolean, optional
            Whether output pspec values are samples at various delay bins
            or are integrated bandpowers over delay bins. Default: False

    exclude_auto_bls : boolean
        If blpairs is None, redundant baseline groups will be formed and
        all cross-multiplies will be constructed. In doing so, if
        exclude_auto_bls is True, eliminate all instances of a bl crossed
        with itself. Default: False

    exclude_cross_bls : boolean
        If True and if blpairs is None, exclude all bls crossed with a
        different baseline. Note if this and exclude_auto_bls are True
        then no blpairs will exist. Default: False

    exclude_permutations : boolean
        If blpairs is None, redundant baseline groups will be formed and
        all cross-multiplies will be constructed. In doing so, if
        exclude_permutations is True, eliminates instances of
        (bl_B, bl_A) if (bl_A, bl_B) also exists. Default: False

    Nblps_per_group : integer
        If blpairs is None, group blpairs into sub-groups of baseline-pairs
        of this size. See utils.calc_blpair_reds() for details. Default: None

    bl_len_range : len-2 float tuple
        A tuple containing the minimum and maximum baseline length to use
        in utils.calc_blpair_reds call. Only used if blpairs is None.

    exclude_flagged_edge_channels : bool, optional
        If True, set spw_ranges to exclude edge channels in the data that are entirely flagged.
        overridden by spw_ranges.

    Nspws : int, optional
        If spw_ranges is not provided, split up frequency band of data into Nspw sub-ranges
        if exclude_flagged_edge_channels is True, these ranges are limited to within the
        entirely flagged edges.

    bl_deg_range : len-2 float tuple
        A tuple containing the min and max baseline angle (ENU frame in degrees)
        to use in utils.calc_blpair_reds. Total range is between 0 and 180
        degrees.

    bl_error_tol : float, optional
        Baseline vector error tolerance when constructing redundant groups.
        Default: 1.0.

    store_window : bool
        If True, store computed window functions (warning, these can be large!)
        in UVPSpec objects.

    allow_fft : bool, optional
        Whether to use a fast FFT summation trick to construct q_hat
        along with H, M, W, G, and V matrices. If False,
        use (significantly slower) brute-force matrix multiplication.
        The FFT method assumes
        a delta-fn bin in delay space. It also only works if the number
        of delay bins is equal to the number of frequencies. Default: False.

    sampling : boolean, optional
        Whether output pspec values are samples at various delay bins
        or are integrated bandpowers over delay bins. Default: False


    beam : PSpecBeam object, UVBeam object or string
        Beam model to use in OQE. Can be a PSpecBeam object or a filepath
        to a beamfits healpix map (see UVBeam)

    cosmo : conversions.Cosmo_Conversions object
        A Cosmo_Conversions object to use as the cosmology when normalizing
        the power spectra. Default is a Planck cosmology.
        See conversions.Cosmo_Conversions for details.

    interleave_times : bool
        Only applicable if Ndsets == 1 or Ndsets==2. If True, copy dset[0] into
        a dset[1] slot and interleave their time arrays. This updates
        dset_pairs to [(0, 1)]. If two dsets, switch the even and odd time-steps
        in dset number 2.

    rephase_to_dset : integer
        Integer index of the anchor dataset when rephasing all other datasets.
        This adds a phasor correction to all others dataset to phase the
        visibility data to the LST-grid of this dataset. Default behavior
        is no rephasing.

    trim_dset_lsts : boolean
        If True, look for constant offset in LST between all dsets, and trim
        non-overlapping LSTs.

    broadcast_dset_flags : boolean
        If True, broadcast dset flags across time using fractional time_thresh.

    time_thresh : float
        Fractional flagging threshold, above which a broadcast of flags across
        time is triggered (if broadcast_dset_flags == True). This is done
        independently for each baseline's visibility waterfall.

    Jy2mK : boolean
        If True, use the beam model provided to convert the units of each
        dataset from Jy to milli-Kelvin. If the visibility data are not in Jy,
        this correction is not applied.

    exact_norm : bool, optional
        If True, estimates power spectrum using Q instead of Q_alt
        (HERA memo #44), where q = R_1 x_1 Q R_2 x_2.
        The default options is False. Beware that
        turning this True would take ~ 7 sec for computing
        power spectrum for 100 channels per time sample per baseline.

    store_cov : boolean, optional
        If True, solve for covariance between bandpowers and store in
        output UVPSpec object.

    store_cov_diag : bool, optional
            If True, store the square root of the diagonal of the output covariance matrix
            calculated by using get_analytic_covariance(). The error bars will
            be stored in the form of: sqrt(diag(cov_array_real)) + 1.j*sqrt(diag(cov_array_imag)).
            It's a way to save the disk space since the whole cov_array data with a size of Ndlys x Ndlys x Ntimes x Nblpairs x Nspws
            is too large.

    return_q : bool, optional
        If True, return the results (delay spectra and covariance matrices)
        for the unnormalized bandpowers in the separate UVPSpec object.

    known_cov : dicts of input covariance matrices
        known_cov has the type {Ckey:covariance}, which is the same with ds._C. The matrices
        stored in known_cov must be constructed externally, different from those in ds._C which
        are constructed internally.

    return_q : bool, optional
        If True, return the results (delay spectra and covariance matrices)
        for the unnormalized bandpowers in the separate UVPSpec object.

    known_cov : dicts of input covariance matrices
        known_cov has the type {Ckey:covariance}, which is the same with ds._C. The matrices
        stored in known_cov must be constructed externally, different from those in ds._C which
        are constructed internally.

    filter_extensions : list of 2-tuple or 2-list, optional
        Set number of channels to extend filtering width.

    overwrite : boolean
        If True, overwrite outputs if they exist on disk.

    symmetric_taper : bool, optional
        speicfy if taper should be applied symmetrically to K-matrix (if true)
        or on the left (if False). default is True

    file_type : str, optional
        If dsets passed as a list of filenames, specify which file format
        the files use. Default: 'miriad'.

    verbose : boolean
        If True, report feedback to standard output.

    history : str
        String to add to history of each UVPSpec object.

    tsleep : float, optional
        Time to wait in seconds after each attempt at opening the container file.

    maxiter : int, optional
        Maximum number of attempts to open container file (useful for concurrent
        access when file may be locked temporarily by other processes).

    cov_model : string, optional
        Type of covariance model to calculate, if not cached. Options=['empirical', 'dsets', 'autos', 'foreground_dependent',
        (other model names in known_cov)]
        In 'dsets' mode, error bars are estimated from user-provided per baseline and per channel standard deivations.
        In 'empirical' mode, error bars are estimated from the data by averaging the
        channel-channel covariance of each baseline over time and
        then applying the appropriate linear transformations to these
        frequency-domain covariances.
        In 'autos' mode, the covariances of the input data
        over a baseline is estimated from the autocorrelations of the two antennas forming the baseline
        across channel bandwidth and integration time.
        In 'foreground_dependent' mode, it involves using auto-correlation amplitudes to model the input noise covariance
        and visibility outer products to model the input systematics covariance.
        For more details see ds.get_analytic_covariance().

        Note: if dsets are str and cov_model is autos or fg_dependent, will also load auto correlations.

    r_params: dict, optional
        Dictionary with parameters for weighting matrix. Required fields and
        formats depend on the mode of `data_weighting`. Default: None.

        - `sinc_downweight` fields:
            - `filter_centers`: list of floats (or float) specifying the
                                (delay) channel numbers at which to center
                                filtering windows. Can specify fractional
                                channel number.

            - `filter_half_widths`:  list of floats (or float) specifying the width
                                of each filter window in (delay) channel
                                numbers. Can specify fractional channel number.

            - `filter_factors`: list of floats (or float) specifying how much
                                power within each filter window is to be
                                suppressed.

    Returns
    -------
    ds : PSpecData object
        The PSpecData object used for OQE of power spectrum, with cached
        weighting matrices.
    """
    # type check
    assert isinstance(dsets, (list, tuple, np.ndarray)), \
        "dsets must be fed as a list of dataset string paths or UVData objects."

    # parse psname
    if psname_ext is not None:
        assert isinstance(psname_ext, (str, np.str))
    else:
        psname_ext = ''

    # polarizations check
    if pol_pairs is not None:
        pols = sorted(set(np.ravel(pol_pairs)))
    else:
        pols = None

    # baselines check
    if blpairs is not None:
        err_msg = "blpairs must be fed as a list of baseline-pair tuples, Ex: [((1, 2), (3, 4)), ...]"
        assert isinstance(blpairs, list), err_msg
        assert np.all([isinstance(blp, tuple) for blp in blpairs]), err_msg
        bls1 = [blp[0] for blp in blpairs]
        bls2 = [blp[1] for blp in blpairs]
        bls = sorted(set(bls1 + bls2))
    else:
        # get redundant baseline groups
        bls = None

    # check cov_model
    if cov_model in ["autos", "foreground_dependent"] and bls is not None:
        # include autos if cov_model is autos
        bls += [(ant, ant) for ant in np.unique(utils.flatten(bls))]

    # Construct dataset pairs to operate on
    Ndsets = len(dsets)
    if dset_pairs is None:
        dset_pairs = list(itertools.combinations(range(Ndsets), 2))

    if dset_labels is None:
        dset_labels = ["dset{}".format(i) for i in range(Ndsets)]
    else:
        assert not np.any(['_' in dl for dl in dset_labels]), \
          "cannot accept underscores in input dset_labels: {}".format(dset_labels)

    # if dsets are not UVData, assume they are filepaths or list of filepaths
    if not isinstance(dsets[0], UVData):
        try:
            # load data into UVData objects if fed as list of strings
            t0 = time.time()
            dsets = _load_dsets(dsets, bls=bls, pols=pols, file_type=file_type, verbose=verbose)
            utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
                      lvl=1, verbose=verbose)
        except ValueError:
            # at least one of the dset loads failed due to no data being present
            utils.log("One of the dset loads failed due to no data overlap given "
                      "the bls and pols selection", verbose=verbose)
            return None

    assert np.all([isinstance(d, UVData) for d in dsets]), \
        "dsets must be fed as a list of dataset string paths or UVData objects."

    # check dsets_std input
    if dsets_std is not None:
        err_msg = "input dsets_std must be a list of UVData objects or " \
                  "filepaths to miriad files"
        assert isinstance(dsets_std,(list, tuple, np.ndarray)), err_msg
        assert len(dsets_std) == Ndsets, "len(dsets_std) must equal len(dsets)"

        # load data if not UVData
        if not isinstance(dsets_std[0], UVData):
            try:
                # load data into UVData objects if fed as list of strings
                t0 = time.time()
                dsets_std = _load_dsets(dsets_std, bls=bls, pols=pols, file_type=file_type, verbose=verbose)
                utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
                          lvl=1, verbose=verbose)
            except ValueError:
                # at least one of the dsets_std loads failed due to no data
                # being present
                utils.log("One of the dsets_std loads failed due to no data overlap given "
                          "the bls and pols selection", verbose=verbose)
                return None

        assert np.all([isinstance(d, UVData) for d in dsets_std]), err_msg
    if vis_units != "UNCALIB":
        for dset in dsets:
            dset.vis_units = vis_units
        if dsets_std is not None:
            for dset in dsets_std:
                dset.vis_units = vis_units
    # read calibration if provided (calfits partial IO not yet supported)
    if cals is not None:
        if not isinstance(cals, (list, tuple)):
            cals = [cals for d in dsets]
        if not isinstance(cals[0], UVCal):
            t0 = time.time()
            cals = _load_cals(cals, verbose=verbose)
            utils.log("Loaded calibration in %1.1f sec." % (time.time() - t0),
                      lvl=1, verbose=verbose)
        err_msg = "cals must be a list of UVCal, filepaths, or list of filepaths"
        assert np.all([isinstance(c, UVCal) for c in cals]), err_msg

    if external_flags is not None:
        if not isinstance(external_flags, (list, tuple)):
            external_flags = [external_flags for d in dsets]
        if not isinstance(external_flags[0], UVFlag):
            external_flags = [UVFlag(extf) for extf in external_flags]

    # configure polarization
    if pol_pairs is None:
        unique_pols = np.unique(np.hstack([d.polarization_array for d in dsets]))
        unique_pols = [uvutils.polnum2str(up) for up in unique_pols]
        pol_pairs = [(up, up) for up in unique_pols]
    assert len(pol_pairs) > 0, "no pol_pairs specified"

    # load beam
    if isinstance(beam, (str, np.str)):
        beam = pspecbeam.PSpecBeamUV(beam, cosmo=cosmo)

    # beam and cosmology check
    if beam is not None:
        assert isinstance(beam, pspecbeam.PSpecBeamBase)
        if cosmo is not None:
            beam.cosmo = cosmo

    # package into PSpecData
    ds = PSpecData(dsets=dsets, wgts=[None for d in dsets], labels=dset_labels,
                   dsets_std=dsets_std, beam=beam, cals=cals, cal_flag=cal_flag)

    # erase calibration as they are no longer needed
    del cals

    # trim dset LSTs
    if trim_dset_lsts:
        ds.trim_dset_lsts()
    # interleave times
    if interleave_times:
        if len(ds.dsets) not in [1, 2]:
            raise ValueError("interleave_times only applicable for Ndsets == 1 and Ndsets==2")

        # update dsets
        if len(ds.dsets) == 1:
            Ntimes = ds.dsets[0].Ntimes # get smallest Ntimes
            Ntimes -= Ntimes % 2  # make it an even number
            ds.dsets.append(ds.dsets[0].select(times=np.unique(ds.dsets[0].time_array)[1:Ntimes:2], inplace=False))
            ds.dsets[0].select(times=np.unique(ds.dsets[0].time_array)[0:Ntimes:2], inplace=True)
            # update _time_independent_weights
            keys = list(ds._time_independent_weights.keys())
            for key in keys:
                ds._time_independent_weights[(1,) + key[1:]] = ds._time_independent_weights[key]
                ds._unflagged_time_integration[(1,) + key[1:]] = ds._unflagged_time_integration[key]
            ds.labels.append("dset1")
            ds.Ntimes = ds.Ntimes // 2#divide number of times by two.
            # update dsets_std
            if ds.dsets_std[0] is None:
                ds.dsets_std.append(None)
            else:
                ds.dsets_std.append(ds.dsets_std[0].select(times=np.unique(ds.dsets_std[0].time_array)[1:Ntimes:2], inplace=False))
                ds.dsets_std[0].select(times=np.unique(ds.dsets_std[0].time_array)[0:Ntimes:2], inplace=True)
                # wgts is currently always None
            ds.wgts.append(None)
        elif len(ds.dsets) == 2:
            Ntimes = ds.dsets[0].Ntimes # get smallest Ntimes
            Ntimes -= Ntimes % 2
            ds.dsets[0].select(times=np.unique(ds.dsets[0].time_array)[:Ntimes])
            ds.dsets[1].select(times=np.unique(ds.dsets[1].time_array)[:Ntimes])
            if ds.dsets_std[0] is not None:
                ds.dsets_std[0].select(times=np.unique(ds.dsets_std[0].time_array)[:Ntimes])
                ds.dsets_std[1].select(times=np.unique(ds.dsets_std[1].time_array)[:Ntimes])
            # reshuffle the data in the second data set.
            # get even / odd time-indices
            Nbls = ds.dsets[0].Nbls
            even_data = np.asarray([ds.dsets[1].data_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
            odd_data = np.asarray([ds.dsets[1].data_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
            for tind in range(0, Ntimes//2):
                ds.dsets[1].data_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_data[tind]
                ds.dsets[1].data_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_data[tind]
            even_flags = np.asarray([ds.dsets[1].flag_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
            odd_flags = np.asarray([ds.dsets[1].flag_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
            for tind in range(Ntimes//2):
                ds.dsets[1].flag_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_flags[tind]
                ds.dsets[1].flag_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_flags[tind]
            even_nsamples = np.asarray([ds.dsets[1].nsample_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
            odd_nsamples = np.asarray([ds.dsets[1].nsample_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
            for tind in range(Ntimes//2):
                ds.dsets[1].nsample_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_nsamples[tind]
                ds.dsets[1].nsample_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_nsamples[tind]
            if ds.dsets_std[1] is not None:
                even_data = np.asarray([ds.dsets_std[1].data_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
                odd_data = np.asarray([ds.dsets_std[1].data_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
                for tind in range(0, Ntimes//2):
                    ds.dsets_std[1].data_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_data[tind]
                    ds.dsets_std[1].data_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_data[tind]
                even_flags = np.asarray([ds.dsets_std[1].flag_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
                odd_flags = np.asarray([ds.dsets_std[1].flag_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
                for tind in range(Ntimes//2):
                    ds.dsets_std[1].flag_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_flags[tind]
                    ds.dsets_std[1].flag_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_flags[tind]
                even_nsamples = np.asarray([ds.dsets_std[1].nsample_array[2*tind*Nbls:(2*tind+1)*Nbls] for tind in range(Ntimes//2)])
                odd_nsamples = np.asarray([ds.dsets_std[1].nsample_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] for tind in range(Ntimes//2)])
                for tind in range(Ntimes//2):
                    ds.dsets_std[1].nsample_array[2*tind*Nbls:(2*tind+1)*Nbls] = odd_nsamples[tind]
                    ds.dsets_std[1].nsample_array[(2*tind+1)*Nbls:(2*tind+2)*Nbls] = even_nsamples[tind]
        # swap unflagged time integration index!
        for blkey in self._unflagged_time_integration:
            if blkey[0] == 1:
                if self._unflagged_time_integration[blkey] % 2 == 0:
                    self._unflagged_time_integration[blkey] += 1
                else:
                    self._unflagged_time_integration[blkey] -= 1
        dset_pairs = [(0, 1)]
        dsets = ds.dsets
        dsets_std = ds.dsets_std
        wgts = ds.wgts
        dset_labels = ds.labels

    # rephase if desired
    if rephase_to_dset is not None:
        ds.rephase_to_dset(rephase_to_dset)

    # broadcast flags
    if broadcast_dset_flags:
        ds.broadcast_dset_flags(time_thresh=time_thresh, spw_ranges=spw_ranges,
                                set_time_independent_weights=True)

    # perform Jy to mK conversion if desired
    if Jy2mK:
        ds.Jy_to_mK(average_conversion=Jy2mK_avg)

    # Print warning if auto_bls is set to exclude correlations of the
    # same baseline with itself, because this may cause a bias if one
    # is already cross-correlating different times to avoid noise bias.
    # See issue #160 on hera_pspec repo
    if exclude_auto_bls:
        raise_warning("Skipping the cross-multiplications of a baseline "
                      "with itself may cause a bias if one is already "
                      "cross-correlating different times to avoid the "
                      "noise bias. Please see hera_pspec github issue 160 "
                      "to make sure you know what you are doing! "
                      "https://github.com/HERA-Team/hera_pspec/issues/160",
                      verbose=verbose)

    # check dset pair type
    err_msg = "dset_pairs must be fed as a list of len-2 integer tuples"
    assert isinstance(dset_pairs, list), err_msg
    assert np.all([isinstance(d, tuple) for d in dset_pairs]), err_msg

    # Get baseline-pairs to use for each dataset pair
    bls1_list, bls2_list = [], []
    for i, dsetp in enumerate(dset_pairs):
        # get bls if blpairs not fed
        if blpairs is None:
            (bls1, bls2, blps, xants1,
             xants2) = utils.calc_blpair_reds(
                                      dsets[dsetp[0]], dsets[dsetp[1]],
                                      filter_blpairs=True,
                                      exclude_auto_bls=exclude_auto_bls,
                                      exclude_cross_bls=exclude_cross_bls,
                                      exclude_permutations=exclude_permutations,
                                      Nblps_per_group=Nblps_per_group,
                                      bl_len_range=bl_len_range,
                                      bl_deg_range=bl_deg_range,
                                      include_autocorrs=include_autocorrs,
                                      bl_tol=bl_error_tol)
            bls1_list.append(bls1)
            bls2_list.append(bls2)

        # ensure fed blpairs exist in each of the datasets
        else:
            dset1_bls = dsets[dsetp[0]].get_antpairs()
            dset2_bls = dsets[dsetp[1]].get_antpairs()
            _bls1 = []
            _bls2 = []
            for _bl1, _bl2 in zip(bls1, bls2):
                if (_bl1 in dset1_bls or _bl1[::-1] in dset1_bls) \
                    and (_bl2 in dset2_bls or _bl2[::-1] in dset2_bls):
                    _bls1.append(_bl1)
                    _bls2.append(_bl2)

            bls1_list.append(_bls1)
            bls2_list.append(_bls2)

    # Open PSpecContainer to store all output in
    if verbose: print("Opening {} in transactional mode".format(filename))
    psc = container.PSpecContainer(filename, mode='rw', keep_open=False, tsleep=tsleep, maxiter=maxiter)

    # assign group name
    if groupname is None:
        groupname = '_'.join(dset_labels)

    # Loop over dataset combinations
    for i, dset_idxs in enumerate(dset_pairs):
        # check bls lists aren't empty
        if len(bls1_list[i]) == 0 or len(bls2_list[i]) == 0:
            continue
        # Run OQE
        uvp = ds.pspec(bls1_list[i], bls2_list[i], dset_idxs, pol_pairs, symmetric_taper=symmetric_taper,
                       spw_ranges=spw_ranges, n_dlys=n_dlys, r_params=r_params, rcond=rcond,
                       store_cov=store_cov, store_cov_diag=store_cov_diag, input_data_weight=input_data_weight,
                       exact_norm=exact_norm, sampling=sampling, allow_fft=allow_fft,
                       return_q=return_q, cov_model=cov_model, known_cov=known_cov, suppression_factor=suppression_factor,
                       exclude_flagged_edge_channels=exclude_flagged_edge_channels, standoff=standoff,
                       norm=norm, taper=taper, history=history, verbose=verbose, Nspws=Nspws,
                       filter_extensions=filter_extensions, store_window=store_window)

        # Store output
        psname = '{}_x_{}{}'.format(dset_labels[dset_idxs[0]],
                                    dset_labels[dset_idxs[1]], psname_ext)
        if time_avg:
            if store_cov_diag:
                err_field = cov_model + '_diag'
            else:
                err_field = None
            uvp.average_spectra(time_avg=True, error_field=cov_model + '_diag')

        # write in transactional mode
        if verbose: print("Storing {}".format(psname))
        psc.set_pspec(group=groupname, psname=psname, pspec=uvp,
                      overwrite=overwrite)

    return ds

def get_pspec_run_argparser():
    a = argparse.ArgumentParser(description="argument parser for pspecdata.pspec_run()")

    def list_of_int_tuples(v):
        v = [tuple([int(_x) for _x in x.split()]) for x in v.split(",")]
        return v

    def list_of_str_tuples(v):
        v = [tuple([str(_x) for _x in x.split()]) for x in v.split(",")]
        return v

    def list_of_tuple_tuples(v):
        v = [tuple([int(_x) for _x in x.split()]) for x in v.split(",")]
        v = [(x[:2], x[2:]) for x in v]
        return v

    a.add_argument("dsets", nargs='*', help="List of UVData objects or miriad filepaths.")
    a.add_argument("filename", type=str, help="Output filename of HDF5 container.")
    a.add_argument("--dsets_std", nargs='*', default=None, type=str, help="List of miriad filepaths to visibility standard deviations.")
    a.add_argument("--groupname", default=None, type=str, help="Groupname for the UVPSpec objects in the HDF5 container.")
    a.add_argument("--dset_pairs", default=None, type=list_of_int_tuples, help="List of dset pairings for OQE wrapped in quotes. Ex: '0 0, 1 1' --> [(0, 0), (1, 1), ...]")
    a.add_argument("--dset_labels", default=None, type=str, nargs='*', help="List of string labels for each input dataset.")
    a.add_argument("--spw_ranges", default=None, type=list_of_int_tuples, help="List of spw channel selections wrapped in quotes. Ex: '200 300, 500 650' --> [(200, 300), (500, 650), ...]")
    a.add_argument("--n_dlys", default=None, type=int, nargs='+', help="List of integers specifying number of delays to use per spectral window selection.")
    a.add_argument("--pol_pairs", default=None, type=list_of_str_tuples, help="List of pol-string pairs to use in OQE wrapped in quotes. Ex: 'xx xx, yy yy' --> [('xx', 'xx'), ('yy', 'yy'), ...]")
    a.add_argument("--blpairs", default=None, type=list_of_tuple_tuples, help="List of baseline-pair antenna integers to run OQE on. Ex: '1 2 3 4, 5 6 7 8' --> [((1 2), (3, 4)), ((5, 6), (7, 8)), ...]")
    a.add_argument("--input_data_weight", default='identity', type=str, help="Data weighting for OQE. See PSpecData.pspec for details.")
    a.add_argument("--norm", default='I', type=str, help='M-matrix normalization type for OQE. See PSpecData.pspec for details.')
    a.add_argument("--taper", default='none', type=str, help="Taper function to use in OQE delay transform. See PSpecData.pspec for details.")
    a.add_argument("--beam", default=None, type=str, help="Filepath to UVBeam healpix map of antenna beam.")
    a.add_argument("--cosmo", default=None, nargs='+', type=float, help="List of float values for [Om_L, Om_b, Om_c, H0, Om_M, Om_k].")
    a.add_argument("--rephase_to_dset", default=None, type=int, help="dset integer index to phase all other dsets to. Default is no rephasing.")
    a.add_argument("--trim_dset_lsts", default=False, action='store_true', help="Trim non-overlapping dset LSTs.")
    a.add_argument("--broadcast_dset_flags", default=False, action='store_true', help="Broadcast dataset flags across time according to time_thresh.")
    a.add_argument("--time_thresh", default=0.2, type=float, help="Fractional flagging threshold across time to trigger flag broadcast if broadcast_dset_flags is True")
    a.add_argument("--Jy2mK", default=False, action='store_true', help="Convert datasets from Jy to mK if a beam model is provided.")
    a.add_argument("--exclude_auto_bls", default=False, action='store_true', help='If blpairs is not provided, exclude all baselines paired with itself.')
    a.add_argument("--exclude_cross_bls", default=False, action='store_true', help='If blpairs is not provided, exclude all baselines paired with a different baseline.')
    a.add_argument("--exclude_permutations", default=False, action='store_true', help='If blpairs is not provided, exclude a basline-pair permutations. Ex: if (A, B) exists, exclude (B, A).')
    a.add_argument("--Nblps_per_group", default=None, type=int, help="If blpairs is not provided and group == True, set the number of blpairs in each group.")
    a.add_argument("--bl_len_range", default=(0, 1e10), nargs='+', type=float, help="If blpairs is not provided, limit the baselines used based on their minimum and maximum length in meters.")
    a.add_argument("--bl_deg_range", default=(0, 180), nargs='+', type=float, help="If blpairs is not provided, limit the baseline used based on a min and max angle cut in ENU frame in degrees.")
    a.add_argument("--bl_error_tol", default=1.0, type=float, help="If blpairs is not provided, this is the error tolerance in forming redundant baseline groups in meters.")
    a.add_argument("--store_cov", default=False, action='store_true', help="Compute and store covariance of bandpowers given dsets_std files or empirical covariance.")
    a.add_argument("--store_cov_diag", default=False, action='store_true', help="Compute and store the error bars calculated by QE formalism.")
    a.add_argument("--return_q", default=False, action='store_true', help="Return unnormalized bandpowers given dsets files.")
    a.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output if it exists.")
    a.add_argument("--cov_model", default='empirical', type=str, help="Model for computing covariance, currently supports empirical or dsets")
    a.add_argument("--psname_ext", default='', type=str, help="Extension for pspectra name in PSpecContainer.")
    a.add_argument("--verbose", default=False, action='store_true', help="Report feedback to standard output.")
    a.add_argument("--filter_extensions", default=None, type=list_of_int_tuples, help="List of spw filter extensions wrapped in quotes. Ex:20 20, 40 40' ->> [(20, 20), (40, 40), ...]")
    a.add_argument("--symmetric_taper", default=True, type=bool, help="If True, apply sqrt of taper before foreground filtering and then another sqrt after. If False, apply full taper after foreground Filter. ")
    a.add_argument("--allow_fft", default=False, action='store_true', help="If True, speed computations up with ffts. Requires all spw_Nfreqs = spw_Ndelays and --sampling=True")
    a.add_argument("--sampling", default=False, action='store_true', help="If True, bandpowers are delta functions at k-bin centers rather then piecewise constant.")
    a.add_argument("--interleave_times", default=False, action='store_true', help="interleave even and odd time-steps. Can be used with one or two data sets.")
    a.add_argument("--time_avg", default=False, action='store_true', help='average power spectra in time.')
    a.add_argument("--file_type", default='miriad', help="filetype to load", type=str)
    a.add_argument("--vis_units", default="UNCALIB", help="override vis_units", type=str)
    a.add_argument("--Nspws", default=1, help="number of spw windows to use. Overriden by spw_ranges.", type=int)
    a.add_argument("--exclude_flagged_edge_channels", default=False, action="store_true", help="ignore entirely flagged edge channels. overriden by spw_ranges.")
    a.add_argument("--standoff", default=0.0, help="number of ns to use as a standoff if input_data_weight=='dayenu'.", type=float)
    a.add_argument("--suppression_factor", default=1e-9, help="suppression factor if input_data_weight=='dayenu'.", type=float)
    a.add_argument("--fullband_filter", default=False, action="store_true", help="If True, extend filtering step to include entire data band. Overriden by filter_extensions.")
    a.add_argument("--external_flags", default=None, type=str, nargs="+", help="Optional, specify external flag file to incorporate in data flagging.")
    a.add_argument("--include_autocorrs", default=False, action="store_true", help="Include auto power spectra.")
    a.add_argument("--rcond", default=1e-15, type=float, help="Cutoff for eigenvalues in taking psuedo-inverse for nomralization.")
    a.add_argument("--Jy2mK_avg", default=False, action="store_true", help="Average Jy2mK factorin frequency before applying to data to avoid introducing spurious spectral structure.")
    a.add_argument("--store_window", default=False, action="store_true", help="Store the window function.")
    return a


def validate_blpairs(blpairs, uvd1, uvd2, baseline_tol=1.0, verbose=True):
    """
    Validate baseline pairings in the blpair list are redundant within the
    specified tolerance.

    Parameters
    ----------
    blpairs : list of baseline-pair tuples
        Ex. [((1,2),(1,2)), ((2,3),(2,3))]
        See docstring of PSpecData.pspec() for details on format.

    uvd1, uvd2 : UVData
        UVData instances containing visibility data that first/second bl in
        blpair will draw from

    baseline_tol : float, optional
        Distance tolerance for notion of baseline "redundancy" in meters.
        Default: 1.0.

    verbose : bool, optional
        If True report feedback to stdout. Default: True.
    """
    # ensure uvd1 and uvd2 are UVData objects
    if isinstance(uvd1, UVData) == False:
        raise TypeError("uvd1 must be a UVData instance")
    if isinstance(uvd2, UVData) == False:
        raise TypeError("uvd2 must be a UVData instance")

    # get antenna position dictionary
    ap1, a1 = uvd1.get_ENU_antpos(pick_data_ants=True)
    ap2, a2 = uvd2.get_ENU_antpos(pick_data_ants=True)
    ap1 = dict(zip(a1, ap1))
    ap2 = dict(zip(a2, ap2))

    # ensure shared antenna keys match within tolerance
    shared = sorted(set(ap1.keys()) & set(ap2.keys()))
    for k in shared:
        assert np.linalg.norm(ap1[k] - ap2[k]) <= baseline_tol, \
            "uvd1 and uvd2 don't agree on antenna positions within " \
            "tolerance of {} m".format(baseline_tol)
    ap = ap1
    ap.update(ap2)

    # iterate through baselines and check baselines crossed with each other
    # are within tolerance
    for i, blg in enumerate(blpairs):
        if isinstance(blg, tuple):
            blg = [blg]
        for blp in blg:
            bl1_vec = ap[blp[0][0]] - ap[blp[0][1]]
            bl2_vec = ap[blp[1][0]] - ap[blp[1][1]]
            if np.linalg.norm(bl1_vec - bl2_vec) >= baseline_tol:
                raise_warning("blpair {} exceeds redundancy tolerance of "
                              "{} m".format(blp, baseline_tol), verbose=verbose)


def raise_warning(warning, verbose=True):
    """
    Warning function.
    """
    if verbose:
        print(warning)


def _load_dsets(fnames, bls=None, pols=None, logf=None, verbose=True,
                file_type='miriad', cals=None, cal_flag=True):
    """
    Helper function for loading UVData-compatible datasets in pspec_run.

    Parameters
    ----------
    fnames : list of str, or list of list of str
        Filenames of load. if an element in fnames is a list of str
        load them all in one call
    bls : list of tuples
        Baselines to load. Default is all.
    pols : list of str
        Polarizations to load, default is all.
    logf : file descriptor
        Log file to write to
    verbose : bool
        Report output to logfile.
    file_type : str
        File type of input files.

    Returns
    -------
    list
        List of UVData objects
    """
    ### TODO: data loading for cross-polarization power
    ### spectra is sub-optimal: only dset1 pol1 and dset2 pol2
    ### is needed instead of pol1 & pol2 for dset1 & dset2
    dsets = []
    Ndsets = len(fnames)
    for i, dset in enumerate(fnames):
        utils.log("Reading {} / {} datasets...".format(i+1, Ndsets),
                  f=logf, lvl=1, verbose=verbose)

        # read data
        uvd = UVData()
        if isinstance(dset, (str, np.str)):
            dfiles = glob.glob(dset)
        else:
            dfiles = dset
        uvd.read(dfiles, bls=bls, polarizations=pols,
                 file_type=file_type)
        uvd.extra_keywords['filename'] = json.dumps(dfiles)
        dsets.append(uvd)

    return dsets

def _load_cals(cnames, logf=None, verbose=True):
    """
    Helper function for loading calibration files.

    Parameters
    ----------
    cnames : list of str, or list of list of str
        Calfits filepaths to load. If an element in cnames is a
        list, load it all at once.
    logf : file descriptor
        Log file to write to.
    verbose : bool
        Report feedback to log file.

    Returns
    -------
    list
        List of UVCal objects
    """
    cals = []
    Ncals = len(cnames)
    for i, cfile in enumerate(cnames):
        utils.log("Reading {} / {} calibrations...".format(i+1, Ncals),
                  f=logf, lvl=1, verbose=verbose)

        # read data
        uvc = UVCal()
        if isinstance(cfile, (str, np.str)):
            uvc.read_calfits(glob.glob(cfile))
        else:
            uvc.read_calfits(cfile)
        uvc.extra_keywords['filename'] = json.dumps(cfile)
        cals.append(uvc)

    return cals

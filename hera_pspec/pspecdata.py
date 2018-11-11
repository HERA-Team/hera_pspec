import numpy as np
import aipy
from pyuvdata import UVData
import copy, operator, itertools, sys
from collections import OrderedDict as odict
import hera_cal as hc
from hera_pspec import uvpspec, utils, version, pspecbeam, container
from hera_pspec import uvpspec_utils as uvputils
from pyuvdata import utils as uvutils
import datetime
import time
import argparse
import ast
import glob


class PSpecData(object):

	def __init__(self, dsets=[], wgts=[], dsets_std=None, labels=None, beam=None):
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
			data point in UVData objects in dsets. If specified as a dict, the key names
			will be used to tag each dataset. Default: Empty list.

		wgts : list or dict of UVData objects, optional
			Set of UVData objects containing weights for the input data.
			Default: Empty list.

		labels : list of str, optional
			An ordered list of names/labels for each dataset, if dsets was
			specified as a list. If None, names will not be assigned to the
			datasets. If dsets was specified as a dict, the keys
			of that dict will be used instead of this. Default: None.

		beam : PspecBeam object, optional
			PspecBeam object containing information about the primary beam
			Default: None.
		"""
		self.clear_cache()  # clear matrix cache
		self.dsets = []; self.wgts = []; self.labels = []
		self.dsets_std = []
		self.Nfreqs = None
		self.spw_range = None
		self.spw_Nfreqs = None
		self.spw_Ndlys = None

		# set data weighting to identity by default
		# and taper to none by default
		self.data_weighting = 'identity'
		self.taper = 'none'

		# set dsets_std to None if any are None.
		if not dsets_std is None and None in dsets_std:
			dsets_std = None

		# Store the input UVData objects if specified
		if len(dsets) > 0:
			self.add(dsets, wgts, dsets_std=dsets_std, labels=labels)

		# Store a primary beam
		self.primary_beam = beam

	def add(self, dsets, wgts, labels=None, dsets_std=None):
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
			set to None, the flags of the corresponding

		labels : list of str
			An ordered list of names/labels for each dataset, if dsets was
			specified as a list. If dsets was specified as a dict, the keys
			of that dict will be used instead.

		dsets_std: UVData or list or dict
			Optional UVData object or list of UVData objects containing the
			standard deviations (real and imaginary) of data to add to the
			collection. If dsets is a dict, will assume dsets_std is a dict
			and if dsets is a list, will assume dsets_std is a list.
		"""
		# Check for dicts and unpack into an ordered list if found
		if isinstance(dsets, dict):
			# Disallow labels kwarg if a dict was passed
			if labels is not None:
				raise ValueError("If 'dsets' is a dict, 'labels' cannot be "
								 "specified.")

			if not isinstance(wgts, dict):
				raise TypeError("If 'dsets' is a dict, 'wgts' must also be "
								"a dict")

			if not isinstance(dsets_std, dict):
				if dsets_std is None:
					dsets_std = [None for m in range(len(dsets))]
				else:
					raise TypeError("If 'dsets' is a dict, 'dsets_std' must"
									"also be a dict")
			else:
				_dsets_std = [dsets_std[key] for key in labels]
				dsets_std = _dsets_std

			# Unpack dsets and wgts dicts
			labels = dsets.keys()
			_dsets = [dsets[key] for key in labels]
			_wgts = [wgts[key] for key in labels]
			dsets = _dsets
			wgts = _wgts

		# Convert input args to lists if possible
		if isinstance(dsets, UVData): dsets = [dsets,]
		if isinstance(wgts, UVData): wgts = [wgts,]
		if isinstance(labels, str): labels = [labels,]
		if isinstance(dsets_std, UVData): dsets_std = [dsets_std,]
		if wgts is None: wgts = [wgts,]
		if dsets_std is None: dsets_std = [dsets_std for m in range(len(dsets))]
		if isinstance(dsets, tuple): dsets = list(dsets)
		if isinstance(wgts, tuple): wgts = list(wgts)
		if isinstance(dsets_std, tuple): dsets_std = list(dsets_std)

		# Only allow UVData or lists
		if not isinstance(dsets, list) or not isinstance(wgts, list)\
		or not isinstance(dsets_std, list):
			raise TypeError("dsets, dsets_std, and wgts must be UVData"
							"or lists of UVData")

		# Make sure enough weights were specified
		assert(len(dsets) == len(wgts))
		assert(len(dsets_std) == len(dsets))
		if labels is not None: assert(len(dsets) == len(labels))

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

		# Store labels (if they were set)
		if self.labels is None:
			self.labels = []
		if labels is None:
			labels = ["dset{:d}".format(i) for i in range(len(self.dsets), len(dsets)+len(self.dsets))]
		self.labels += labels

		# Append to list
		self.dsets += dsets
		self.wgts += wgts
		self.dsets_std += dsets_std

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
		self.spw_range = (0, self.Nfreqs)
		self.spw_Nfreqs = self.Nfreqs
		self.spw_Ndlys = self.spw_Nfreqs

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
			raise ValueError("self.wgts does not have same length as self.dsets")

		if len(self.dsets_std) != len(self.dsets):
			raise ValueError("self.dsets_std does not have the same length as "
							 "self.dsets")

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
		lst_diffs = np.array(map(lambda dset: np.unique(self.dsets[0].lst_array) - np.unique(dset.lst_array), self.dsets[1:]))
		if np.max(np.abs(lst_diffs)) > 0.001:
			raise_warning("Warning: taking power spectra between LST bins misaligned by more than 15 seconds",
							verbose=verbose)

		# raise warning if frequencies don't match
		freq_diffs = np.array(map(lambda dset: np.unique(self.dsets[0].freq_array) - np.unique(dset.freq_array), self.dsets[1:]))
		if np.max(np.abs(freq_diffs)) > 0.001e6:
			raise_warning("Warning: taking power spectra between frequency bins misaligned by more than 0.001 MHz",
						  verbose=verbose)

		# Check phase type
		phase_types = []
		for d in self.dsets: phase_types.append(d.phase_type)
		if np.unique(phase_types).size > 1:
			raise ValueError("all datasets must have the same phase type (i.e. 'drift', 'phased', ...)\ncurrent phase types are {}".format(phase_types))

		# Check phase centers if phase type is phased
		if 'phased' in set(phase_types):
			phase_ra = map(lambda d: d.phase_center_ra_degrees, self.dsets)
			phase_dec = map(lambda d: d.phase_center_dec_degrees, self.dsets)
			max_diff_ra = np.max(map(lambda d: np.diff(d), itertools.combinations(phase_ra, 2)))
			max_diff_dec = np.max(map(lambda d: np.diff(d), itertools.combinations(phase_dec, 2)))
			max_diff = np.sqrt(max_diff_ra**2 + max_diff_dec**2)
			if max_diff > 0.15: raise_warning("Warning: maximum phase-center difference between datasets is > 10 arcmin", verbose=verbose)

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
		key = uvutils.get_iterable(key)
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

	def clear_cache(self, keys=None):
		"""
		Clear stored matrix data (or some subset of it).

		Parameters
		----------
		keys : list of tuples, optional
			List of keys to remove from matrix cache. If None, all
			keys will be removed. Default: None.
		"""
		if keys is None:
			self._C, self._I, self._iC, self._Y, self._R = {}, {}, {}, {}, {}
			self._identity_G, self._identity_H, self._identity_Y = {}, {}, {} 
		else:
			for k in keys:
				try: del(self._C[k])
				except(KeyError): pass
				try: del(self._I[k])
				except(KeyError): pass
				try: del(self._iC[k])
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
		elif isinstance(dset, int):
			return dset
		else:
			raise TypeError("dset must be either an int or string")

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
		if isinstance(bl, (int, np.int, np.int32)):
			assert len(key) > 1, "baseline must be fed as a tuple"
			bl = tuple(key[:2])
			key = key[2:]
		else:
			key = key[1:]
		assert isinstance(bl, tuple), "baseline must be fed as a tuple"

		# put pol into bl key if it exists
		if len(key) > 0:
			pol = key[0]
			assert isinstance(pol, (str, int, np.int, np.int32)), "pol must be fed as a str or int"
			bl += (key[0],)

		return dset_idx, bl

	def x(self, key):
		"""
		Get data for a given dataset and baseline, as specified in a standard
		key format.

		Parameters
		----------
		key : tuple
			Tuple containing dataset ID and baseline index. The first element
			of the tuple is the dataset index (or label), and the subsequent
			elements are the baseline ID.

		Returns
		-------
		x : array_like
			Array of data from the requested UVData dataset and baseline.
		"""
		dset, bl = self.parse_blkey(key)
		spw = slice(self.spw_range[0], self.spw_range[1])
		return self.dsets[dset].get_data(bl).T[spw]

	def dx(self, key):
		"""
		Get standard deviation of data for given dataset and baseline as
		pecified in standard key format.

		Parameters
		----------
		key : tuple
			Tuple containing datset ID and baseline index. The first element
			of the tuple is the dataset index (or label), and the subsequent
			elements are the baseline ID.

		Returns
		-------
		dx : array_like
			Array of std data from the requested UVData dataset and baseline.
		"""
		assert isinstance(key,tuple)
		dset,bl = self.parse_blkey(key)
		spw = slice(self.spw_range[0], self.spw_range[1])

		return self.dsets_std[dset].get_data(bl).T[spw]

	def w(self, key):
		"""
		Get weights for a given dataset and baseline, as specified in a
		standard key format.

		Parameters
		----------
		key : tuple
			Tuple containing dataset ID and baseline index. The first element
			of the tuple is the dataset index, and the subsequent elements are
			the baseline ID.

		Returns
		-------
		w : array_like
			Array of weights for the requested UVData dataset and baseline.
		"""
		dset, bl = self.parse_blkey(key)
		spw = slice(self.spw_range[0], self.spw_range[1])

		if self.wgts[dset] is not None:
			return self.wgts[dset].get_data(bl).T[spw]
		else:
			# If weights were not specified, use the flags built in to the
			# UVData dataset object
			wgts = (~self.dsets[dset].get_flags(bl)).astype(float).T[spw]
			return wgts

	def set_C(self, cov):
		"""
		Set the cached covariance matrix to a set of user-provided values.

		Parameters
		----------
		cov : dict
			Dictionary containing new covariance values for given datasets and
			baselines. Keys of the dictionary are tuples, with the first item
			being the ID (index) of the dataset, and subsequent items being the
			baseline indices.
		"""
		self.clear_cache(cov.keys())
		for key in cov: self._C[key] = cov[key]

	def C_model(self, key, model='time_average'):
		"""
		Return a covariance model having specified a key and model type.

		Parameters
		----------
		key : tuple
			Tuple containing indices of dataset and baselines. The first item
			specifies the index (ID) of a dataset in the collection, while
			subsequent indices specify the baseline index, in _key2inds format.

		model : string, optional
			Type of covariance model to calculate, if not cached. options=['time_average']

		Returns
		-------
		C : array-like
			Covariance model for the specified key.
		"""
		# type check
		assert isinstance(key, tuple), "key must be fed as a tuple"
		assert isinstance(model, (str, np.str)), "model must be a string"
		assert model in ['time_average','empirical'], "didn't recognize model {}".format(model)

		# parse key
		dset, bl = self.parse_blkey(key)
		key = (dset,) + (bl,)

		# add model to key
		Ckey = key + (model,)

		# check cache
		if not self._C.has_key(Ckey):
			# calculate covariance model
			if model == 'time_average':
				spw = slice(self.spw_range[0], self.spw_range[1])
				data = self.dsets[dset].get_data(bl).T[spw]
				#(Nspw_freqs, Ntimes)
				data = data.T
				#(Ntimes, Nspw_freqs)
				data_square = np.zeros((data.shape[0], data.shape[1], data.shape[1]), dtype=np.complex128)
				#(Ntimes, Nspw_freqs, Nspw_freqs)
				for time in range(data.shape[0]):
					data_square[time, :, :] = np.einsum('i,j', data[time, :], data[time, :].conj())
				covariance = np.average(data_square, axis=0)
				#(Nspw_freqs, Nspw_freqs)
				data_average = np.average(data, axis=0)
				#(Nspw_freqs)
				data_average_square =  np.einsum('i,j', data_average, data_average.conj())
				#(Nspw_freqs, Nspw_freqs)
				covariance -= data_average_square
				#(Nspw_freqs, Nspw_freqs)
				self.set_C({Ckey: covariance})
			if model == 'empirical':
				self.set_C({Ckey: utils.cov(self.x(key), self.w(key))})
				#(Nspw_freqs, Nspw_freqs)
		return self._C[Ckey]

	def cross_covar_model(self, key1, key2, model='time_average', conj_1=False, conj_2=True):
		"""
		Return a covariance model having specified a key and model type.

		Parameters
		----------
		key1, key2 : tuples
			Tuples containing indices of dataset and baselines. The first item
			specifies the index (ID) of a dataset in the collection, while
			subsequent indices specify the baseline index, in _key2inds format.

		model : string, optional
			Type of covariance model to calculate, if not cached. options=['time_average']

		conj_1 : boolean, optional
			Whether to conjugate first copy of data in covar or not. Default: False

		conj_2 : boolean, optional
			Whether to conjugate second copy of data in covar or not. Default: True

		Returns
		-------
		cross_covar : array-like, spw_Nfreqs x spw_Nfreqs
			Cross covariance model for the specified key.
		"""
		# type check
		assert isinstance(key1, tuple), "key1 must be fed as a tuple"
		assert isinstance(key2, tuple), "key2 must be fed as a tuple"
		assert isinstance(model, (str, np.str)), "model must be a string"
		assert model in ['time_average','empirical'], "didn't recognize model {}".format(model)

		# parse key
		dset, bl = self.parse_blkey(key1)
		key1 = (dset,) + (bl,)
		dset, bl = self.parse_blkey(key2)
		key2 = (dset,) + (bl,)

		if model == 'empirical':
			covar = utils.cov(self.x(key1), self.w(key1),
							  self.x(key2), self.w(key2),
							  conj_1=conj_1, conj_2=conj_2)
		if model == 'time_average':
			x1 = self.x(key1)
			#(Nspw_freqs, Ntimes)
			x2 = self.x(key2)
			#(Nspw_freqs, Ntimes)
			x1 = x1.T
			#(Ntimes, Nspw_freqs)
			x2 = x2.T
			#(Ntimes, Nspw_freqs)
			if conj_1:
				x1 = x1.conj()
			if conj_2:
				x2 = x2.conj()
			x1x2 = np.zeros((x1.shape[0], x1.shape[1], x1.shape[1]), dtype=np.complex128)
			#(Ntimes, Nspw_freqs, Nspw_freqs)
			for time in range(x1.shape[0]):
				x1x2[time, :, :] = np.einsum('i,j', x1[time, :], x2[time, :])
			covar = np.average(x1x2, axis=0)
			#(Nspw_freqs, Nspw_freqs)
			x1_average = np.average(x1, axis=0)
			#(Nspw_freqs)
			x2_average = np.average(x2, axis=0)
			#(Nspw_freqs)
			x1_average_x2_average =  np.einsum('i,j', x1_average, x2_average)
			#(Nspw_freqs, Nspw_freqs)
			covar -= x1_average_x2_average
			#(Nspw_freqs, Nspw_freqs)
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

		if not self._I.has_key(key):
			self._I[key] = np.identity(self.spw_Nfreqs)
		return self._I[key]

	def iC(self, key, model='empirical'):
		"""
		Return the inverse covariance matrix, C^-1.

		Parameters
		----------
		key : tuple
			Tuple containing indices of dataset and baselines. The first item
			specifies the index (ID) of a dataset in the collection, while
			subsequent indices specify the baseline index, in _key2inds format.

		model : string
			Type of covariance model to calculate, if not cached. options=['empirical']

		Returns
		-------
		iC : array_like
			Inverse covariance matrix for specified dataset and baseline.
		"""
		assert isinstance(key, tuple)
		# parse key
		dset, bl = self.parse_blkey(key)
		key = (dset,) + (bl,)

		Ckey = key + (model,)

		# Calculate inverse covariance if not in cache
		if not self._iC.has_key(Ckey):
			C = self.C_model(key, model=model)
			U,S,V = np.linalg.svd(C.conj()) # conj in advance of next step

			# FIXME: Not sure what these are supposed to do
			#if self.lmin is not None: S += self.lmin # ensure invertibility
			#if self.lmode is not None: S += S[self.lmode-1]

			# FIXME: Is series of dot products quicker?
			self.set_iC({Ckey:np.einsum('ij,j,jk', V.T, 1./S, U.T)})
		return self._iC[Ckey]

	def Y(self, key):
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

		Returns
		-------
		Y : array_like
			spw_Nfreqs x spw_Nfreqs diagonal matrix holding AND of flags
			across all times for each freq channel.
		"""
		assert isinstance(key, tuple)
		# parse key
		dset, bl = self.parse_blkey(key)
		key = (dset,) + (bl,)

		if not self._Y.has_key(key):
			self._Y[key] = np.diag(np.max(self.w(key), axis=1))
			if not np.all(np.isclose(self._Y[key], 0.0) + np.isclose(self._Y[key], 1.0)):
				raise NotImplementedError("Non-binary weights not currently implmented")
		return self._Y[key]

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
		for k in d: self._iC[k] = d[k]

	def set_R(self, d):
		"""
		Set the data-weighting matrix for a given dataset and baseline to
		a specified value for later use in q_hat.

		Parameters
		----------
		d : dict
			Dictionary containing data to insert into data-weighting R matrix
			cache. Keys are tuples, following the same format as the input to
			self.R().
		"""
		for k in d: self._R[k] = d[k]

	def R(self, key):
		"""
		Return the data-weighting matrix R, which is a product of
		data covariance matrix (I or C^-1), diagonal flag matrix (Y) and
		diagonal tapering matrix (T):

		R = sqrt(T^t) sqrt(Y^t) K sqrt(Y) sqrt(T)

		where T is a diagonal matrix holding the taper and Y is a diagonal
		matrix holding flag weights. The K matrix comes from either I or iC
		depending on self.data_weighting, T is informed by self.taper and Y
		is taken from self.Y().

		Parameters
		----------
		key : tuple
			Tuple containing indices of dataset and baselines. The first item
			specifies the index (ID) of a dataset in the collection, while
			subsequent indices specify the baseline index, in _key2inds format.
		"""
		assert isinstance(key, tuple)
		# parse key
		dset, bl = self.parse_blkey(key)
		key = (dset,) + (bl,)
		Rkey = key + (self.data_weighting,) + (self.taper,)

		if not self._R.has_key(Rkey):
			# form sqrt(taper) matrix
			if self.taper == 'none':
				sqrtT = np.ones(self.spw_Nfreqs).reshape(1, -1)
			else:
				sqrtT = np.sqrt(aipy.dsp.gen_window(self.spw_Nfreqs, self.taper)).reshape(1, -1)

			# get flag weight vector: straight multiplication of vectors mimics matrix multiplication
			sqrtY = np.sqrt(self.Y(key).diagonal().reshape(1, -1))

			# replace possible nans with zero (when something dips negative in sqrt for some reason)
			sqrtT[np.isnan(sqrtT)] = 0.0
			sqrtY[np.isnan(sqrtY)] = 0.0

			# form R matrix
			if self.data_weighting == 'identity':
				self._R[Rkey] = sqrtT.T * sqrtY.T * self.I(key) * sqrtY * sqrtT

			elif self.data_weighting == 'iC':
				self._R[Rkey] = sqrtT.T * sqrtY.T * self.iC(key) * sqrtY * sqrtT

		return self._R[Rkey]

	def set_weighting(self, data_weighting):
		"""
		Set data weighting type.

		Parameters
		----------
		data_weighting : str
			Type of data weightings. Options=['identity', 'iC']
		"""
		self.data_weighting = data_weighting

	def set_taper(self, taper):
		"""
		Set data tapering type.

		Parameters
		----------
		taper : str
			Type of data tapering. See aipy.dsp.gen_window for options.
		"""
		self.taper = taper

	def set_spw(self, spw_range):
		"""
		Set the spectral window range.

		Parameters
		----------
		spw_range : tuple, contains start and end of spw in channel indices
			used to slice the frequency array
		"""
		assert isinstance(spw_range, tuple), \
			"spw_range must be fed as a len-2 integer tuple"
		assert isinstance(spw_range[0], (int, np.int)), \
			"spw_range must be fed as len-2 integer tuple"
		self.spw_range = spw_range
		self.spw_Nfreqs = spw_range[1] - spw_range[0]

	def set_Ndlys(self, ndlys=None):
		"""
		Set the number of delay bins used.

		Parameters
		----------
		ndlys : integer
			Number of delay bins. Default: None, sets number of delay
			bins equal to the number of frequency channels
		"""

		if ndlys == None:
			self.spw_Ndlys = self.spw_Nfreqs
		else:
			# Check that one is not trying to estimate more delay channels than there are frequencies
			if self.spw_Nfreqs < ndlys:
				raise ValueError("Cannot estimate more delays than there are frequency channels")
			self.spw_Ndlys = ndlys

	def cov_q_hat(self, key1, key2, time_indices=None):
		"""
		Compute the un-normalized covariance matrix for q_hat for a given pair
		of visibility vectors. Returns the following matrix:

		Cov(\hat{q}_a,\hat{q}_b)

		!!!Only supports covariance between same power-spectrum estimates!!!
		(covariance between pair of baselines with the same pair of baselines)
		!!!Assumes that both baselines used in power-spectrum estimate
		!!!have independent noise realizations!!!

		Parameters
		----------
		key1, key2: tuples or lists of tuples
			Tuples containing the indices of the dataset and baselines for the
			two input datavectors. If a list of tuples is provided, the baselines
			in the list will be combined with inverse noise weights.

		time_indices: list of indices of times to include or just a single time.
		default is None -> compute covariance for all times.

		Returns
		-------
		cov_q_hat: matrix with covariances between un-normalized band powers
		"""
		# type check
		if time_indices is None:
			time_indices = [tind for tind in range(self.Ntimes)]
		elif isinstance(time_indices, (int, np.integer)):
			time_indices = [time_indices]
		if not isinstance(time_indices, list):
			raise ValueError("time_indices must be an integer or list of integers.")

		#check time_indices
		for tind in time_indices:
			if not (tind >= 0 and tind <= self.Ntimes):
				raise ValueError("Invalid time index provided.")

		qc = np.zeros((len(time_indices), self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex128)
		R1, R2 = 0.0, 0.0
		n1a, n2a, n1b, n2b=0.0, 0.0, 0.0, 0.0
		# compute noise covariance matrices. Assume diagonal!
		# compute E^alpha and E^beta
		if isinstance(key1, list):
			for _key in key1:
				R1 += self.R(_key)
				n1a += np.real(self.dx(_key))**2.
				n1b += np.imag(self.dx(_key))**2.
		else:
			R1 = self.R(key1)
			n1a = np.real(self.dx(key1))**2.
			n1b = np.imag(self.dx(key1))**2.

		if isinstance(key2, list):
			for _key in key2:
				R2 += self.R(_key)
				n2a += np.real(self.dx(_key))**2.
				n2b += np.imag(self.dx(_key))**2.
		else:
			R2 = self.R(key2)
			n2a = np.real(self.dx(key2)**2.)
			n2b = np.imag(self.dx(key2)**2.)

		N1 = np.zeros((self.spw_Nfreqs, self.spw_Nfreqs), dtype=np.complex128)
		N2 = np.zeros_like(N1)
		Qalphas = np.repeat(np.array([self.get_Q_alt(dly) for dly in range(self.spw_Ndlys)])[np.newaxis, :, :, :], self.spw_Ndlys, axis=0)
		Qbetas = np.repeat(np.array([self.get_Q_alt(dly) for dly in range(self.spw_Ndlys)])[:, np.newaxis, :, :], self.spw_Ndlys, axis=1)
 
		# Q_alpha/Q_beta are N_dlys x N_dlys x N_freq x N_freq
		# taking advantage of broadcast rules!
		# matmul only applies to the last two dimensions
		# and stacks everything else!
		Ealphas = np.matmul(R1.T.conj(), np.matmul(Qalphas, R2))
		Ebetas = np.matmul(R1.T.conj(), np.matmul(Qbetas, R2))
		#E_alpha/E_beta ar N_dlys x N_dlys x N_freq x N_freq
		for indnum, tind in enumerate(time_indices):
			N1[:, :] = np.diag(n1a[:, tind] + n1b[:, tind])
			N2[:, :] = np.diag(n2a[:, tind] + n2b[:, tind])
			# total covariance is sum of real and imaginary covariances
			qc[indnum] = np.trace(np.matmul(Ealphas, np.matmul(N1, np.matmul(Ebetas, N2))), axis1=2, axis2=3)
		return qc/4.

	def q_hat(self, key1, key2, allow_fft=False):
		"""
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

		Returns
		-------
		q_hat : array_like
			Unnormalized bandpowers
		"""
		Rx1, Rx2 = 0.0, 0.0

		# Calculate R x_1
		if isinstance(key1, list):
			for _key in key1: Rx1 += np.dot(self.R(_key), self.x(_key))
		else:
			Rx1 = np.dot(self.R(key1), self.x(key1))

		# Calculate R x_2
		if isinstance(key2, list):
			for _key in key2: Rx2 += np.dot(self.R(_key), self.x(_key))
		else:
			Rx2 = np.dot(self.R(key2), self.x(key2))

		# use FFT if possible and allowed
		if allow_fft and (self.spw_Nfreqs == self.spw_Ndlys):
			_Rx1 = np.fft.fft(Rx1, axis=0)
			_Rx2 = np.fft.fft(Rx2, axis=0)

			return 0.5 * np.fft.fftshift(_Rx1, axes=0).conj() * np.fft.fftshift(_Rx2, axes=0)

		else:
			q = []
			for i in xrange(self.spw_Ndlys):
				Q = self.get_Q_alt(i)
				QRx2 = np.dot(Q, Rx2)
				qi = np.einsum('i...,i...->...', Rx1.conj(), QRx2)
				q.append(qi)
			return 0.5 * np.array(q)

	def get_G(self, key1, key2):
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

		Returns
		-------
		G : array_like, complex
			Fisher matrix, with dimensions (Nfreqs, Nfreqs).
		"""
		if self.spw_Ndlys == None:
			raise ValueError("Number of delay bins should have been set"
							 "by now! Cannot be equal to None")

		G = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
		R1 = self.R(key1)
		R2 = self.R(key2)

		iR1Q, iR2Q = {}, {}
		for ch in xrange(self.spw_Ndlys):
			Q = self.get_Q_alt(ch)
			iR1Q[ch] = np.dot(R1, Q) # R_1 Q
			iR2Q[ch] = np.dot(R2, Q) # R_2 Q

		for i in xrange(self.spw_Ndlys):
			for j in xrange(self.spw_Ndlys):
				# tr(R_2 Q_i R_1 Q_j)
				G[i,j] += np.einsum('ab,ba', iR1Q[i], iR2Q[j])

		# check if all zeros, in which case turn into identity
		if np.count_nonzero(G) == 0:
			G = np.eye(self.spw_Ndlys)

		return G / 2.

	def get_H(self, key1, key2, sampling=False):
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

		Returns
		-------
		H : array_like, complex
			Dimensions (Nfreqs, Nfreqs).
		"""
		if self.spw_Ndlys == None:
			raise ValueError("Number of delay bins should have been set"
							 "by now! Cannot be equal to None")

		H = np.zeros((self.spw_Ndlys, self.spw_Ndlys), dtype=np.complex)
		R1 = self.R(key1)
		R2 = self.R(key2)

		if not sampling:
			sinc_matrix = np.zeros((self.spw_Nfreqs, self.spw_Nfreqs))
			for i in range(self.spw_Nfreqs):
				for j in range(self.spw_Nfreqs):
					sinc_matrix[i,j] = np.float(i - j)
			sinc_matrix = np.sinc(sinc_matrix / np.float(self.spw_Ndlys))

		iR1Q_alt, iR2Q = {}, {}
		for ch in xrange(self.spw_Ndlys):
			Q_alt = self.get_Q_alt(ch)
			iR1Q_alt[ch] = np.dot(R1, Q_alt) # R_1 Q_alt
			Q = Q_alt

			if not sampling:
				Q *= sinc_matrix

			iR2Q[ch] = np.dot(R2, Q) # R_2 Q

		for i in xrange(self.spw_Ndlys): # this loop goes as nchan^4
			for j in xrange(self.spw_Ndlys):
				# tr(R_2 Q_i R_1 Q_j)
				H[i,j] += np.einsum('ab,ba', iR1Q_alt[i], iR2Q[j])

		# check if all zeros, in which case turn into identity
		if np.count_nonzero(H) == 0:
			H = np.eye(self.spw_Ndlys)

		return H / 2.

	def get_unnormed_E(self, key1, key2):
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
		key1, key2 : tuples or lists of tuples
			Tuples containing indices of dataset and baselines for the two 
			input datavectors. If a list of tuples is provided, the baselines 
			in the list will be combined with inverse noise weights.

		Returns
		-------
		E : array_like, complex
			Set of E matrices, with dimensions (Ndlys, Nfreqs, Nfreqs).

		"""
		if self.spw_Ndlys == None:
			raise ValueError("Number of delay bins should have been set"
							 "by now! Cannot be equal to None")

		E_matrices = np.zeros((self.spw_Ndlys, self.spw_Nfreqs, self.spw_Nfreqs),
							   dtype=np.complex)
		R1 = self.R(key1)
		R2 = self.R(key2)
		for dly_idx in range(self.spw_Ndlys):
			QR2 = np.dot(self.get_Q_alt(dly_idx), R2)
			E_matrices[dly_idx] = np.dot(R1, QR2)

		return 0.5 * E_matrices
	
	def get_unnormed_V(self, key1, key2, model='time_average'):
		"""
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

		model : str, default: 'time_average'
			How the covariances of the input data should be estimated.
		
		Returns
		-------
		V : array_like, complex
			Bandpower covariance matrix, with dimensions (Ndlys, Ndlys).
		"""
		# Collect all the relevant pieces
		E_matrices = self.get_unnormed_E(key1, key2)
		C1 = self.C_model(key1, model=model)
		C2 = self.C_model(key2, model=model)
		P21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=False)
		S21 = self.cross_covar_model(key2, key1, model=model, conj_1=True, conj_2=True)

		E21C1 = np.dot(np.transpose(E_matrices.conj(), (0,2,1)), C1)
		E12C2 = np.dot(E_matrices, C2)
		auto_term = np.einsum('aij,bji', E12C2, E21C1)
		E12starS21 = np.dot(E_matrices.conj(), S21)
		E12P21 = np.dot(E_matrices, P21)
		cross_term = np.einsum('aij,bji', E12P21, E12starS21)

		return auto_term + cross_term

	def analytic_variance(self, key1, key2, M, model='time_average'):
		"""
		Calculates the auto-covariance matrix for both the real and imaginary
		parts of bandpowers (i.e., the q vectors and the p vectors). 

		Define
		
		.. math ::
			Real part of q_a = (1/2) (q_a + q_a^dagger)
			Imaginary part of q_a = (1/2) (q_a - q_a^dagger) 
			Real part of p_a = (1/2) (p_a + p_a^dagger)
			Imaginary part of p_a = (1/2) (p_a - p_a^dagger)

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

		The variance of (1/2) (q_a - q_a^dagger) :
		(1/4){ (<q_a q_a> - <q_a><q_a>) - 2(<q_a q_a^dagger> - <q_a><q_a^dagger>) 
		+ (<q_a^dagger q_a^dagger> - <q_a^dagger><q_a^dagger>) }

		The variance of (1/2) (p_a + p_a^dagger):
		(1/4) { M_{ab} M_{ac} (<q_b q_c> - <q_b><q_c>) + 
		M_{ab} M_{ac}^* (<q_b q_c^dagger> - <q_b><q_c^dagger>) + 
		M_{ab}^* M_{ac} (<q_b^dagger q_c> - <q_b^dagger><q_c>) + 
		M_{ab}^* M_{ac}^* (<q_b^dagger q_c^dagger> - <q_b^dagger><q_c^dagger>) }

		The variance of (1/2) (p_a - p_a^dagger):
		(1/4) { M_{ab} M_{ac} (<q_b q_c> - <q_b><q_c>) - 
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

		Parameters
		----------
		key1, key2 : tuples or lists of tuples
			Tuples containing indices of dataset and baselines for the two
			input datavectors. If a list of tuples is provided, the baselines
			in the list will be combined with inverse noise weights.

		model : str, default: 'time_average'
			How the covariances of the input data should be estimated.
		
		Returns
		-------
		V : array_like, complex
			Bandpower variance , with dimensions (Ndlys, ).
		"""
		# Collect all the relevant pieces
		E_matrices = self.get_unnormed_E(key1, key2)
		C11 = self.C_model(key1, model=model)
		C22 = self.C_model(key2, model=model)
		C21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=True)
		C12 = self.cross_covar_model(key1, key2, model=model, conj_1=False, conj_2=True)
		P11 = self.cross_covar_model(key1, key1, model=model, conj_1=False, conj_2=False)
		S11 = self.cross_covar_model(key1, key1, model=model, conj_1=True, conj_2=True)
		P22 = self.cross_covar_model(key2, key2, model=model, conj_1=False, conj_2=False)
		S22 = self.cross_covar_model(key2, key2, model=model, conj_1=True, conj_2=True)
		P21 = self.cross_covar_model(key2, key1, model=model, conj_1=False, conj_2=False)
		S21 = self.cross_covar_model(key2, key1, model=model, conj_1=True, conj_2=True)

		E12C21 = np.dot(E_matrices, C21) 
		E12P22 = np.dot(E_matrices, P22) 
		E21starS11 = np.dot(np.transpose(E_matrices, (0,2,1)), S11)
		E21C11 = np.dot(np.transpose(E_matrices.conj(), (0,2,1)), C11)
		E12C22 = np.dot(E_matrices, C22)
		E12starS21 = np.dot(E_matrices.conj(), S21)
		E12P21 = np.dot(E_matrices, P21)
		E21C12 = np.dot(np.transpose(E_matrices.conj(), (0,2,1)), C12)
		E21P11 = np.dot(np.transpose(E_matrices.conj(), (0,2,1)), P11)
		E12starS22 = np.dot(E_matrices.conj(), S22)	

		q_q = np.einsum('aij,bji', E12P22, E21starS11) + np.einsum('aij,bji', E12C21, E12C21)
		q_qdagger = np.einsum('aij,bji', E12C22, E21C11) + np.einsum('aij,bji', E12P21, E12starS21)
		qdagger_qdagger = np.einsum('aij,bji', E21C12, E21C12) + np.einsum('aij,bji', E21P11, E12starS22)

		var_q_real = (q_q + qdagger_qdagger + 2.*q_qdagger) / 4.
		var_q_imag = (q_q + qdagger_qdagger - 2.*q_qdagger) / 4.
		var_q_real = var_q_real.diagonal()
		var_q_imag = var_q_imag.diagonal()

		var_p_real = ( np.einsum('ab,cd,bd->ac', M, M, q_q) +
			2. * np.einsum('ab,cd,bd->ac', M, M.conj(), q_qdagger) + 
			np.einsum('ab,cd,bd->ac', M.conj(), M.conj(), qdagger_qdagger) )/ 4. 

		var_p_imag = ( np.einsum('ab,cd,bd->ac', M, M, q_q) -
			2. * np.einsum('ab,cd,bd->ac', M, M.conj(), q_qdagger) +
			np.einsum('ab,cd,bd->ac', M.conj(), M.conj(), qdagger_qdagger) )/ 4.

		var_p_real = var_p_real.diagonal()
		var_p_imag = var_p_imag.diagonal()

		return var_q_real, var_q_imag, var_p_real, var_p_imag

	def get_MW(self, G, H, mode='I', band_covar=None):
		"""
		Construct the normalization matrix M and window function matrix W for
		the power spectrum estimator. These are defined through Eqs. 14-16 of
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

		As written, the window functions will not be correclty normalized; it needs
		to be adjusted by the pspec scalar for it to be approximately correctly
		normalized. If the beam is being provided, this will be done in the pspec
		function.

		Parameters
		----------
		G : array_like
			Denominator matrix for the bandpowers, with dimensions (Nfreqs, Nfreqs).

		H : array_like
			Response matrix for the bandpowers, with dimensions (Nfreqs, Nfreqs).

		mode : str, optional
			Definition to use for M. Must be one of the options listed above.
			Default: 'I'.

		band_covar : array_like, optional
			Covariance matrix of the unnormalized bandpowers (i.e., q). Used only
			if requesting the V^-1/2 normalization. Use get_unnormed_V to get the
			covariance to put in here, or provide your own array.
			Default: None

		Returns
		-------
		M : array_like
			Normalization matrix, M. (If G was passed in as a dict, a dict of
			array_like will be returned.)

		W : array_like
			Window function matrix, W. (If G was passed in as a dict, a dict of
			array_like will be returned.)
		"""
		### Next few lines commented out because (while elegant), the situation
		### here where G is a distionary is not supported by the rest of the code.
		# Recursive case, if many G's were specified at once
		# if type(G) is dict:
		#     M,W = {}, {}
		#     for key in G: M[key], W[key] = self.get_MW(G[key], mode=mode)
		#     return M, W

		# Check that mode is supported
		modes = ['H^-1', 'V^-1/2', 'I', 'L^-1']
		assert(mode in modes)

		# Build M matrix according to specified mode
		if mode == 'H^-1':

			try:
				M = np.linalg.inv(H)
			except np.linalg.LinAlgError as err:
				if 'Singular matrix' in str(err):
					M = np.linalg.pinv(H)
					raise_warning("Warning: Window function matrix is singular "
								  "and cannot be inverted, so using "
								  " pseudoinverse instead.")
				else:
					raise np.linalg.LinAlgError("Linear algebra error with H matrix "
												"during MW computation.")

			W = np.dot(M, H)
			W_norm = np.sum(W, axis=1)
			W = (W.T / W_norm).T

		elif mode == 'V^-1/2':
			if np.sum(band_covar) == None:
				raise ValueError("Covariance not supplied for V^-1/2 normalization")
			# First find the eigenvectors and eigenvalues of the unnormalizd covariance
			# Then use it to compute V^-1/2
			eigvals, eigvects = np.linalg.eigh(band_covar)
			if (eigvals <= 0.).any():
				raise_warning("At least one non-positive eigenvalue for the "
							  "unnormed bandpower covariance matrix.")
			V_minus_half = np.dot(eigvects, np.dot(np.diag(1./np.sqrt(eigvals)), eigvects.T))

			W_norm = np.diag(1. / np.sum(np.dot(V_minus_half, H), axis=1))
			M = np.dot(W_norm, V_minus_half)
			W = np.dot(M, H)

		elif mode == 'I':
			# This is not the M matrix as is rigorously defined in the
			# OQE formalism, because the power spectrum scalar is excluded
			# in this matrix normalization (i.e., M doesn't do the full
			# normalization)
			M = np.diag(1. / np.sum(G, axis=1))
			W_norm = np.diag(1. / np.sum(H, axis=1))
			W = np.dot(W_norm, H)
		else:
			raise NotImplementedError("Cholesky decomposition mode not currently supported.")
			# # Cholesky decomposition
			# order = np.arange(G.shape[0]) - np.ceil((G.shape[0]-1.)/2.)
			# order[order < 0] = order[order < 0] - 0.1

			# # Negative integers have larger absolute value so they are sorted
			# # after positive integers.
			# order = (np.abs(order)).argsort()
			# if np.mod(G.shape[0], 2) == 1:
			#     endindex = -2
			# else:
			#     endindex = -1
			# order = np.hstack([order[:5], order[endindex:], order[5:endindex]])
			# iorder = np.argsort(order)

			# G_o = np.take(np.take(G, order, axis=0), order, axis=1)
			# L_o = np.linalg.cholesky(G_o)
			# U,S,V = np.linalg.svd(L_o.conj())
			# M_o = np.dot(np.transpose(V), np.dot(np.diag(1./S), np.transpose(U)))
			# M = np.take(np.take(M_o, iorder, axis=0), iorder, axis=1)

		return M, W

	def get_Q_alt(self, mode, allow_fft=True):
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

		if (self.spw_Ndlys == self.spw_Nfreqs) and (allow_fft == True):
			_m = np.zeros((self.spw_Nfreqs,), dtype=np.complex)
			_m[mode] = 1. # delta function at specific delay mode
			# FFT to transform to frequency space
			m = np.fft.fft(np.fft.ifftshift(_m))
		else:
			if self.spw_Ndlys % 2 == 0:
				start_idx = -self.spw_Ndlys/2
			else:
				start_idx = -(self.spw_Ndlys - 1)/2
			m = (start_idx + mode) * np.arange(self.spw_Nfreqs)
			m = np.exp(-2j * np.pi * m / self.spw_Ndlys)

		Q_alt = np.einsum('i,j', m.conj(), m) # dot it with its conjugate
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
		return np.dot(M, q)

	def cov_p_hat(self, M, q_cov):
		"""
		Covariance estimate between two different band powers p_alpha and p_beta
		given by M_{alpha i} M^*_{beta,j} C_q^{ij} where C_q^{ij} is the q-covariance

		Parameters
		----------
		M : array_like
			Normalization matrix, M.

		q_cov : array_like
			covariance between bandpowers in q_alpha and q_beta
		"""
		p_cov = np.zeros_like(q_cov)
		for tnum in range(len(p_cov)):
			p_cov[tnum] = np.einsum('ab,cd,bd->ac', M, M, q_cov[tnum])
		return p_cov

	def broadcast_dset_flags(self, spw_ranges=None, time_thresh=0.2, unflag=False):
		"""
		For each dataset in self.dset, update the flag_array such that
		the flagging patterns are time-independent for each baseline given
		a selection for spectral windows.

		For each frequency pixel in a selected spw, if the fraction of flagged
		times exceeds time_thresh, then all times are flagged. If it does not,
		the specific integrations which hold flags in the spw are flagged across
		all frequencies in the spw.

		Additionally, one can also unflag the flag_array entirely if desired.

		Note: although technically allowed, this function may give unexpected results
		if multiple spectral windows in spw_ranges have frequency overlap.

		Note: it is generally not recommended to set time_thresh > 0.5, which
		could lead to substantial amounts of data being flagged.

		Parameters
		----------
		spw_ranges : list of tuples
			list of len-2 spectral window tuples, specifying the start (inclusive)
			and stop (exclusive) index of the frequency array for each spw.
			Default is to use the whole band

		time_thresh : float
			Fractional threshold of flagged pixels across time needed to flag all times
			per freq channel. It is not recommend to set this greater than 0.5

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
		assert isinstance(spw_ranges, list), "spw_ranges must be fed as a list of tuples"

		# iterate over datasets
		for dset in self.dsets:
			# iterate over spw ranges
			for spw in spw_ranges:
				self.set_spw(spw)
				# unflag
				if unflag:
					# unflag for all times
					dset.flag_array[:, :, self.spw_range[0]:self.spw_range[1], :] = False
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
						flag_ints = np.max(flags[:, self.spw_range[0]:self.spw_range[1]], axis=1)
						dset.flag_array[bl_inds[flag_ints], :, self.spw_range[0]:self.spw_range[1], i] = True

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
		
	def scalar(self, pol, little_h=True, num_steps=2000, beam=None, taper_override='no_override'):
		"""
		Computes the scalar function to convert a power spectrum estimate
		in "telescope units" to cosmological units, using self.spw_range to set
		spectral window.

		See arxiv:1304.4991 and HERA memo #27 for details.

		This function uses the state of self.taper in constructing scalar.
		See PSpecData.pspec for details.

		Parameters
		----------
		pol: str
				Which polarization to compute the scalar for.
				e.g. 'I', 'Q', 'U', 'V', 'XX', 'YY'...

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

		Returns
		-------
		scalar: float
				[\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
				in h^-3 Mpc^3 or Mpc^3.
		"""
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
										num_steps=num_steps)
		else:
			scalar = beam.compute_pspec_scalar(start, end, len(freqs),
											   pol=pol, taper=self.taper,
											   little_h=little_h,
											   num_steps=num_steps)
		return scalar

	def scalar_delay_adjustment(self, key1=None, key2=None, sampling=False, 
								Gv=None, Hv=None):
		"""
		Computes an adjustment factor for the pspec scalar that is needed
		when the number of delay bins is not equal to the number of
		frequency channels.

		This adjustment is necessary because
		\sum_gamma tr[Q^alt_alpha Q^alt_gamma] = N_freq**2
		is something that is true only when N_freqs = N_dlys.

		In general, the result is still independent of alpha, but is
		no longer given by N_freq**2. (Nor is it just N_dlys**2!)

		This function uses the state of self.taper in constructing adjustment.
		See PSpecData.pspec for details.

		Parameters
		----------
		key1, key2 : tuples or lists of tuples, optional
			Tuples containing indices of dataset and baselines for the two
			input datavectors. If a list of tuples is provided, the baselines
			in the list will be combined with inverse noise weights. If Gv and 
			Hv are specified, these arguments will be ignored. Default: None.

		sampling : boolean, optional
			Whether to sample the power spectrum or to assume integrated
			bands over wide delay bins. Default: False
		
		Gv, Hv : array_like, optional
			If specified, use these arrays instead of calling self.get_G() and 
			self.get_H(). Using precomputed Gv and Hv will speed up this 
			function significantly. Default: None.

		Returns
		-------
		adjustment : float

		"""
		if Gv is None: Gv = self.get_G(key1, key2)
		if Hv is None: Hv = self.get_H(key1, key2, sampling)
		
		# get ratio
		summed_G = np.sum(Gv, axis=1)
		summed_H = np.sum(Hv, axis=1)
		ratio = summed_H.real / summed_G.real

		# fill infs and nans from zeros in summed_G
		ratio[np.isnan(ratio)] = 1.0
		ratio[np.isinf(ratio)] = 1.0

		# get mean ratio
		mean_ratio = np.mean(ratio)
		scatter = np.abs(ratio - mean_ratio)
		if (scatter > 10**-4 * mean_ratio).any():
			raise ValueError("The normalization scalar is band-dependent!")

		adjustment = self.spw_Ndlys / (self.spw_Nfreqs * mean_ratio)

		if self.taper != 'none':
			tapering_fct = aipy.dsp.gen_window(self.spw_Nfreqs, self.taper)
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

		# convert elements to integers if fed as strings
		if isinstance(pol_pair[0], (str, np.str)):
			pol_pair = (uvutils.polstr2num(pol_pair[0]), pol_pair[1])
		if isinstance(pol_pair[1], (str, np.str)):
			pol_pair = (pol_pair[0], uvutils.polstr2num(pol_pair[1]))

		assert isinstance(pol_pair[0], (int, np.integer)), err_msg
		assert isinstance(pol_pair[1], (int, np.integer)), err_msg

		if pol_pair[0] != pol_pair[1]:
			raise NotImplementedError("Only auto/equal polarizations are implement at the moment.")

		dset_ind1 = self.dset_idx(dsets[0])
		dset_ind2 = self.dset_idx(dsets[1])
		dset1 = self.dsets[dset_ind1]  # first UVData object
		dset2 = self.dsets[dset_ind2]  # second UVData object

		valid = True
		if pol_pair[0] not in dset1.polarization_array:
			print "dset {} does not contain data for polarization {}".format(dset_ind1, pol_pair[0])
			valid = False

		if pol_pair[1] not in dset2.polarization_array:
			print "dset {} does not contain data for polarization {}".format(dset_ind2, pol_pair[1])
			valid = False

		return valid

	def pspec(self, bls1, bls2, dsets, pols, n_dlys=None, input_data_weight='identity',
			  norm='I', taper='none', sampling=False, little_h=True, spw_ranges=None,
			  verbose=True, history='', store_cov=False, cov_choice='get_unnormed_V'):
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

		pols : length-2 tuple of strings or integers or list of length-2 tuples of strings or integers
			Contains polarization pairs to use in forming power spectra
			e.g. ('XX','XX') or [('XX','XX'),('XY','YX')] or list of polarization pairs.
			Only auto/equal polarization pairs are implemented at the moment.
			It uses the polarizations of the UVData onjects (specified in dsets)
			by default only if the UVData object consists of equal polarizations.

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
			arguments as aipy.dsp.gen_window(). Default: 'none'.

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

		store_cov : boolean, optional
			If True, calculate an analytic covariance between bandpowers
			given an input visibility noise model, and store the output
			in the UVPSpec object.

		cov_choice : str
			cov_choice in ['get_unnormed_V', 'cov_q_hat'].
			There two independent function in pspecdata to calculate the variance.

		verbose : bool, optional
			If True, print progress, warnings and debugging info to stdout.

		history : str, optional
			history string to attach to UVPSpec object

		Returns
		-------
		uvp : UVPSpec object
			Instance of UVPSpec that holds the output power spectrum data.

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
		# set taper and data weighting
		self.set_taper(taper)
		self.set_weighting(input_data_weight)

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
		assert isinstance(dsets, (list, tuple)), "dsets must be fed as length-2 tuple of integers"
		assert len(dsets) == 2, "len(dsets) must be 2"
		assert isinstance(dsets[0], (int, np.int)) and isinstance(dsets[1], (int, np.int)), "dsets must contain integer indices"
		dset1 = self.dsets[self.dset_idx(dsets[0])]
		dset2 = self.dsets[self.dset_idx(dsets[1])]

		# assert form of bls1 and bls2
		assert isinstance(bls1, list), "bls1 and bls2 must be fed as a list of antpair tuples"
		assert isinstance(bls2, list), "bls1 and bls2 must be fed as a list of antpair tuples"
		assert len(bls1) == len(bls2) and len(bls1) > 0, "length of bls1 must equal length of bls2 and be > 0"

		for i in range(len(bls1)):
			if isinstance(bls1[i], tuple):
				assert isinstance(bls2[i], tuple), "bls1[{}] type must match bls2[{}] type".format(i, i)
			else:
				assert len(bls1[i]) == len(bls2[i]), "len(bls1[{}]) must match len(bls2[{}])".format(i, i)

		# construct list of baseline pairs
		bl_pairs = []
		for i in range(len(bls1)):
			if isinstance(bls1[i], tuple):
				bl_pairs.append( (bls1[i], bls2[i]) )
			elif isinstance(bls1[i], list) and len(bls1[i]) == 1:
				bl_pairs.append( (bls1[i][0], bls2[i][0]) )
			else:
				bl_pairs.append(map(lambda j: (bls1[i][j] , bls2[i][j]), range(len(bls1[i]))))

		# validate bl-pair redundancy
		validate_blpairs(bl_pairs, dset1, dset2, baseline_tol=1.0)

		# configure spectral window selections
		if spw_ranges is None:
			spw_ranges = [(0, self.Nfreqs)]
		else:
			assert np.isclose(map(lambda t: len(t), spw_ranges), 2).all(), "spw_ranges must be fed as a list of length-2 tuples"

		# if using default setting of number of delay bins equal to number of frequency channels
		if n_dlys is None:
			n_dlys = [None for i in range(len(spw_ranges))]
		elif isinstance(n_dlys, (int, np.integer)):
			n_dlys = [n_dlys]

		# if using the whole band in the dataset, then there should just be one n_dly parameter specified
		if spw_ranges is None and n_dlys != None:
			assert len(n_dlys) == 1, "Only one spw, so cannot specify more than one n_dly value"

		# assert that the same number of ndlys has been specified as the number of spws
		assert len(spw_ranges) == len(n_dlys), "Need to specify number of delay bins for each spw"

		# setup polarization selection
		if isinstance(pols, tuple):
			pols = [pols]

		# convert all polarizations to integers if fed as strings
		_pols = []
		for p in pols:
			if isinstance(p[0], (str, np.str)):
				p = (uvutils.polstr2num(p[0]), p[1])
			if isinstance(p[1], (str, np.str)):
				p = (p[0], uvutils.polstr2num(p[1]))
			_pols.append(p)
		pols = _pols

		# initialize empty lists
		data_array = odict()
		data_array_q = odict()
		wgt_array = odict()
		integration_array = odict()
		cov_array = odict()
		cov_array_q = odict()
		var_array_q_real = odict()
		var_array_q_imag = odict()
		var_array_real = odict()
		var_array_imag = odict()
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

		# Loop over spectral windows
		for i in range(len(spw_ranges)):
			
			# set spectral range
			if verbose:
				print( "\nSetting spectral range: {}".format(spw_ranges[i]))
			self.set_spw(spw_ranges[i])

			# set number of delay bins
			self.set_Ndlys(n_dlys[i])
		
			# clear covariance cache
			self.clear_cache()
			
			# setup emtpy data arrays
			spw_data = []
			spw_data_q = []
			spw_wgts = []
			spw_ints = []
			spw_scalar = []
			spw_pol = []
			spw_cov = []
			spw_cov_q = []
			spw_var_q_real = []
			spw_var_q_imag = []
			spw_var_real = []
			spw_var_imag = []
			
			d = self.delays() * 1e-9
			f = dset1.freq_array.flatten()[spw_ranges[i][0]:spw_ranges[i][1]]
			dlys.extend(d)
			dly_spws.extend(np.ones_like(d, np.int16) * i)
			freq_spws.extend(np.ones_like(f, np.int16) * i)
			freqs.extend(f)
			
			# Loop over polarizations
			for j, p in enumerate(pols):
				p_str = tuple(map(lambda _p: uvutils.polnum2str(_p), p))
				if verbose: print( "\nUsing polarization pair: {}".format(p_str))
				
				# validating polarization pair on UVData objects
				valid = self.validate_pol(dsets, tuple(p))
				if not valid:
				   # storing only one polarization as only equal polarization are allowed at the
				   # moment and UVPSpec object also understands one polarization
				   print ("Polarization pair: {} failed the validation test, continuing...".format(p_str))
				   continue
				
				# UVPSpec only takes a single pol currently
				spw_pol.append(p[0])
				pol_data = []
				pol_data_q = []
				pol_wgts = []
				pol_ints = []
				pol_cov = []
				pol_cov_q = []
				pol_var_q_real = []
				pol_var_q_imag = []
				pol_var_real = []
				pol_var_imag = []
				
				# Compute scalar to convert "telescope units" to "cosmo units"
				if self.primary_beam is not None:
					# using zero'th indexed poalrization as cross polarized beam are not yet implemented
					if norm == 'H^-1':
						# If using decorrelation, the H^-1 normalization already deals with the taper,
						# so we need to override the taper when computing the scalar
						scalar = self.scalar(p[0], little_h=True, taper_override='none')
					else:
						scalar = self.scalar(p[0], little_h=True)
				else: 
					raise_warning("Warning: self.primary_beam is not defined, "
								  "so pspectra are not properly normalized",
								  verbose=verbose)
					scalar = 1.0
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
					key1_dof = np.sum(~np.isclose(self.Y(key1).diagonal(), 0.0))
					key2_dof = np.sum(~np.isclose(self.Y(key2).diagonal(), 0.0))
					if key1_dof < self.spw_Ndlys or key2_dof < self.spw_Ndlys:
						if verbose:
							print("WARNING: Number of unflagged chans for key1 and/or key2 < n_dlys\n" \
								  "which may lead to normalization instabilities.")
					
					# Build Fisher matrix
					if input_data_weight == 'identity':
						# in this case, all Gv and Hv differ only by flagging pattern
						# so check if we've already computed this
						# First: get flag weighting matrices given key1 & key2
						Y = np.vstack([self.Y(key1).diagonal(), self.Y(key2).diagonal()])
						
						# Second: check cache for Y
						matches = [np.isclose(Y, y).all() for y in self._identity_Y.values()]
						if True in matches:
							# This Y exists, so pick appropriate G and H and continue
							match = self._identity_Y.keys()[matches.index(True)]
							Gv = self._identity_G[match]
							Hv = self._identity_H[match]
						else:
							# This Y doesn't exist, so compute it
							if verbose: print("  Building G...")
							Gv = self.get_G(key1, key2)
							Hv = self.get_H(key1, key2, sampling=sampling)
							# cache it
							self._identity_Y[(key1, key2)] = Y
							self._identity_G[(key1, key2)] = Gv
							self._identity_H[(key1, key2)] = Hv
					else:
						# for non identity weighting (i.e. iC weighting)
						# Gv and Hv are always different, so compute them
						if verbose: print("  Building G...")
						Gv = self.get_G(key1, key2)
						Hv = self.get_H(key1, key2, sampling=sampling)
					
					# Calculate unnormalized bandpowers
					if verbose: print("  Building q_hat...")
					qv = self.q_hat(key1, key2)
					
					# Normalize power spectrum estimate
					if verbose: print("  Normalizing power spectrum...")
					if norm == 'V^-1/2':
						V_mat = self.get_unnormed_V(key1, key2, model='empirical')
						Mv, Wv = self.get_MW(Gv, Hv, mode=norm, band_covar=V_mat)
					else:
						Mv, Wv = self.get_MW(Gv, Hv, mode=norm)
					pv = self.p_hat(Mv, qv)
					
					# Multiply by scalar
					if self.primary_beam != None:
						if verbose: print("  Computing and multiplying scalar...")
						pv *= scalar
					
					# Wide bin adjustment of scalar, which is only needed for the diagonal norm
					# matrix mode (i.e., norm = 'I')
					if norm == 'I':
						pv *= self.scalar_delay_adjustment(Gv=Gv, Hv=Hv)
					
					# Generate the covariance matrix if error bars provided
					if store_cov:
						if verbose: print(" Building q_hat covariance...")
						if cov_choice == 'get_unnormed_V':
							cov_qv = self.get_unnormed_V(key1, key2, model='time_average')
							cov_qv = np.array([cov_qv for tind in range(self.Ntimes)])

						if cov_choice == 'cov_q_hat':
							cov_qv = self.cov_q_hat(key1, key2)

						cov_pv = self.cov_p_hat(Mv, cov_qv)
						if self.primary_beam != None:
							cov_pv *= \
							(scalar * self.scalar_delay_adjustment(key1, key2,
																   sampling=sampling))**2.
						pol_cov.extend(cov_pv)
						pol_cov_q.extend(cov_qv)

					# Generate the variance for real and imaginary part for bandpowers
					var_q_real, var_q_imag, var_real, var_imag = self.analytic_variance(key1, key2, Mv, model='time_average')
					var_q_real = np.array([var_q_real for tind in range(self.Ntimes)])
					var_q_imag = np.array([var_q_imag for tind in range(self.Ntimes)])
					var_real = np.array([var_real for tind in range(self.Ntimes)])
					var_imag = np.array([var_imag for tind in range(self.Ntimes)])

					if self.primary_beam != None:
							var_real *= \
							(scalar * self.scalar_delay_adjustment(key1, key2,
																   sampling=sampling))**2.
							var_imag *= \
							(scalar * self.scalar_delay_adjustment(key1, key2,
																   sampling=sampling))**2.

					pol_var_q_real.extend(var_q_real)
					pol_var_q_imag.extend(var_q_imag)
					pol_var_real.extend(var_real)
					pol_var_imag.extend(var_imag)	
					
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
					pol_data.extend(pv.T)
					pol_data_q.extend(qv.T)

					# get weights
					wgts1 = self.w(key1).T
					wgts2 = self.w(key2).T

					# get average of nsample across frequency axis, weighted by wgts
					nsamp1 = np.sum(dset1.get_nsamples(bl1 + (p[0],))[:, self.spw_range[0]:self.spw_range[1]] * wgts1, axis=1) \
							 / np.sum(wgts1, axis=1).clip(1, np.inf)
					nsamp2 = np.sum(dset2.get_nsamples(bl2 + (p[1],))[:, self.spw_range[0]:self.spw_range[1]] * wgts2, axis=1) \
							 / np.sum(wgts2, axis=1).clip(1, np.inf)

					# get integ1
					blts1 = dset1.antpair2ind(bl1, ordered=False)
					integ1 = dset1.integration_time[blts1] * nsamp1
					
					# get integ2
					blts2 = dset2.antpair2ind(bl2, ordered=False)
					integ2 = dset2.integration_time[blts2] * nsamp2

					# take inverse average of integ1 and integ2 to get total integration
					# inverse avg is done b/c integ ~ 1/noise_var
					# and due to non-linear operation of V_1 * V_2
					pol_ints.extend(1./np.mean([1./integ1, 1./integ2], axis=0))

					# combined weight is geometric mean
					pol_wgts.extend(np.concatenate([wgts1[:, :, None], wgts2[:, :, None]], axis=2))
	
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
						blp_arr.extend(np.ones_like(inds1, np.int) * uvputils._antnums_to_blpair(blp))

				# insert into data and wgts integrations dictionaries
				spw_data.append(pol_data)
				spw_data_q.append(pol_data_q)		
				spw_wgts.append(pol_wgts)
				spw_ints.append(pol_ints)
				spw_cov.append(pol_cov)
				spw_cov_q.append(pol_cov_q)
				spw_var_q_real.append(pol_var_q_real)
				spw_var_q_imag.append(pol_var_q_imag)
				spw_var_real.append(pol_var_real)
				spw_var_imag.append(pol_var_imag)

			# insert into data and integration dictionaries
			spw_data = np.moveaxis(np.array(spw_data), 0, -1)
			spw_data_q = np.moveaxis(np.array(spw_data_q), 0, -1)
			spw_wgts = np.moveaxis(np.array(spw_wgts), 0, -1)
			spw_ints = np.moveaxis(np.array(spw_ints), 0, -1)
			spw_cov = np.moveaxis(np.array(spw_cov), 0, -1)
			spw_cov_q = np.moveaxis(np.array(spw_cov_q), 0, -1)
			spw_var_q_real = np.moveaxis(np.array(spw_var_q_real), 0, -1)
			spw_var_q_imag = np.moveaxis(np.array(spw_var_q_imag), 0, -1)
			spw_var_real = np.moveaxis(np.array(spw_var_real), 0, -1)
			spw_var_imag = np.moveaxis(np.array(spw_var_imag), 0, -1)
			
			
			data_array[i] = spw_data
			data_array_q[i] = spw_data_q
			cov_array[i] = spw_cov
			cov_array_q[i] = spw_cov_q
			var_array_q_real[i] = spw_var_q_real
			var_array_q_imag[i] = spw_var_q_imag
			var_array_real[i] = spw_var_real
			var_array_imag[i] = spw_var_imag
			wgt_array[i] = spw_wgts
			integration_array[i] = spw_ints
			sclr_arr.append(spw_scalar)

			# raise error if none of pols are consistent witht the UVData objects
			if len(spw_pol)==0:
				raise ValueError("None of the specified polarization pair match that of the UVData objects")

		# fill uvp object
		uvp = uvpspec.UVPSpec()

		# fill meta-data
		uvp.time_1_array = np.array(time1)
		uvp.time_2_array = np.array(time2)
		uvp.time_avg_array = np.mean([uvp.time_1_array, uvp.time_2_array], axis=0)
		uvp.lst_1_array = np.array(lst1)
		uvp.lst_2_array = np.array(lst2)
		uvp.lst_avg_array = np.mean([np.unwrap(uvp.lst_1_array), np.unwrap(uvp.lst_2_array)], axis=0) % (2*np.pi)
		uvp.blpair_array = np.array(blp_arr)
		uvp.Nblpairs = len(np.unique(blp_arr))
		uvp.Ntimes = len(np.unique(time1))
		uvp.Nblpairts = len(time1)
		bls_arr = sorted(set(bls_arr))
		uvp.bl_array = np.array(map(lambda bl: uvp.antnums_to_bl(bl), bls_arr))
		antpos = dict(zip(dset1.antenna_numbers, dset1.antenna_positions))
		uvp.bl_vecs = np.array(map(lambda bl: antpos[bl[0]] - antpos[bl[1]], bls_arr))
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
		uvp.pol_array = np.array(spw_pol, np.int)
		uvp.Npols = len(spw_pol)
		uvp.scalar_array = np.array(sclr_arr)
		uvp.channel_width = dset1.channel_width  # all dsets are validated to agree
		uvp.weighting = input_data_weight
		uvp.vis_units, uvp.norm_units = self.units(little_h=little_h)
		uvp.telescope_location = dset1.telescope_location
		filename1 = getattr(dset1.extra_keywords, 'filename', None)
		filename2 = getattr(dset2.extra_keywords, 'filename', None)
		label1 = self.labels[self.dset_idx(dsets[0])]
		label2 = self.labels[self.dset_idx(dsets[1])]
		uvp.labels = sorted(set([label1, label2]))
		uvp.label_1_array = np.ones((uvp.Nspws, uvp.Nblpairts, uvp.Npols), np.int) \
							* uvp.labels.index(label1)
		uvp.label_2_array = np.ones((uvp.Nspws, uvp.Nblpairts, uvp.Npols), np.int) \
							* uvp.labels.index(label2)
		uvp.labels = np.array(uvp.labels, np.str)
		uvp.history = "UVPSpec written on {} with hera_pspec git hash {}\n{}\n" \
					  "dataset1: filename: {}, label: {}, history:\n{}\n{}\n" \
					  "dataset2: filename: {}, label: {}, history:\n{}\n{}\n" \
					  "".format(datetime.datetime.utcnow(), version.git_hash, '-'*20,
								filename1, label1, dset1.history, '-'*20,
								filename2, label2, dset2.history, '-'*20)
		uvp.taper = taper
		uvp.norm = norm

		if self.primary_beam is not None:
			# attach cosmology
			uvp.cosmo = self.primary_beam.cosmo
			# attach beam info
			uvp.beam_freqs = self.primary_beam.beam_freqs
			uvp.OmegaP, uvp.OmegaPP = self.primary_beam.get_Omegas(uvp.pol_array)
			if hasattr(self.primary_beam, 'filename'):
				uvp.beamfile = self.primary_beam.filename

		# fill data arrays
		uvp.data_array = data_array
		uvp.data_array_q = data_array_q
		if store_cov:
			uvp.cov_array = cov_array
			uvp.cov_array_q = cov_array_q
			uvp.var_array_q_real = var_array_q_real
			uvp.var_array_q_imag = var_array_q_imag
			uvp.var_array_real = var_array_real
			uvp.var_array_imag = var_array_imag

		uvp.integration_array = integration_array
		uvp.wgt_array = wgt_array
		uvp.nsample_array = dict(map(lambda k: (k, np.ones_like(uvp.integration_array[k], np.float)), uvp.integration_array.keys()))

		# run check
		uvp.check()

		return uvp

	def rephase_to_dset(self, dset_index=0, inplace=True):
		"""
		Rephase visibility data in self.dsets to the LST grid of dset[dset_index]
		using hera_cal.utils.lst_rephase. Each integration in all other dsets is
		phased to the center of the corresponding LST bin (by index) in dset[dset_index].

		Will only phase if the dataset's phase type is 'drift'. This is because the rephasing
		algorithm assumes the data is drift-phased when applying phasor term.

		Note that PSpecData.Jy_to_mK() must be run after rephase_to_dset(), if one intends
		to use the former capability at any point.

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
				print "skipping dataset {} b/c it isn't drift phased".format(i)

			# convert UVData to DataContainers. Note this doesn't make
			# a copy of the data
			(data, flgs, antpos, ants, freqs, times, lsts,
			 pols) = hc.io.load_vis(dset, return_meta=True)

			# make bls dictionary
			bls = dict(map(lambda k: (k, antpos[k[0]] - antpos[k[1]]), data.keys()))

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
				polind = pol_list.index(uvutils.polstr2num(k[-1]))
				# insert into dset
				dset.data_array[indices, 0, :, polind] = data[k]

			# set phasing in UVData object to unknown b/c there isn't a single
			# consistent phasing for the entire data set.
			dset.phase_type = 'unknown'

		if inplace is False:
			return dsets

	def Jy_to_mK(self, beam=None):
		"""
		Convert internal datasets from a Jy-scale to mK scale using a primary beam
		model if available. Note that if you intend to rephase_to_dset(), Jy to mK conversion
		must be done after that step.

		Parameters
		----------
		beam : PSpecBeam object
		"""
		# get all unique polarizations of all the datasets
		pols = set(np.ravel([dset.polarization_array for dset in self.dsets]))

		# assign beam
		if beam is None:
			beam = self.primary_beam
		else:
			if self.primary_beam is not None:
				print "Warning: feeding a beam model when self.primary_beam already exists..."

		# Check beam is not None
		assert beam is not None, "Cannot convert Jy --> mK b/c beam object is not defined..."

		# assert type of beam
		assert isinstance(beam, pspecbeam.PSpecBeamBase), "beam model must be a subclass of pspecbeam.PSpecBeamBase"

		# iterate over all pols and get conversion factors
		factors = {}
		for p in pols:
			factors[p] = beam.Jy_to_mK(self.freqs, pol=p)

		# iterate over datasets and apply factor
		for i, dset in enumerate(self.dsets):
			# check dset vis units
			if dset.vis_units.upper() != 'JY':
				print "Cannot convert dset {} Jy -> mK because vis_units = {}".format(i, dset.vis_units)
				continue
			for j, p in enumerate(dset.polarization_array):
				dset.data_array[:, :, :, j] *= factors[p][None, None, :]
			dset.vis_units = 'mK'

	def trim_dset_lsts(self, lst_tol=6):
		"""
		Assuming all datasets in self.dsets are locked to the same LST grid (but
		each may have a constant offset), trim LSTs from each dset that aren't found
		in all other dsets (within some decimal tolerance specified by lst_tol).

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
				print "not all datasets in self.dsets are on the same LST grid, cannot LST trim."
				return

		# get lst array of each dataset and turn into string and add to common_lsts
		lst_arrs = []
		common_lsts = set()
		for i, dset in enumerate(self.dsets):
			lsts = ["{lst:0.{tol}f}".format(lst=l, tol=lst_tol) for l in dset.lst_array]
			lst_arrs.append(lsts)
			if i == 0:
				common_lsts = common_lsts.union(set(lsts))
			else:
				common_lsts = common_lsts.intersection(set(lsts))

		# iterate through dsets and trim off integrations whose lst isn't in common_lsts
		for i, dset in enumerate(self.dsets):
			trim_inds = np.array([l not in common_lsts for l in lst_arrs[i]])
			if np.any(trim_inds):
				self.dsets[i].select(times=dset.time_array[~trim_inds])


def pspec_run(dsets, filename, dsets_std=None, groupname=None, dset_labels=None, dset_pairs=None,
<<<<<<< HEAD
			  psname_ext=None, spw_ranges=None, n_dlys=None, pol_pairs=None, blpairs=None,
			  input_data_weight='identity', norm='I', taper='none',
			  exclude_auto_bls=False, exclude_permutations=True,
			  Nblps_per_group=None, bl_len_range=(0, 1e10), bl_deg_range=(0, 180), bl_error_tol=1.0,
			  beam=None, cosmo=None, rephase_to_dset=None, trim_dset_lsts=False, broadcast_dset_flags=True,
			  time_thresh=0.2, Jy2mK=False, overwrite=True, verbose=True, store_cov=False, history=''):
	"""
	Create a PSpecData object, run OQE delay spectrum estimation and write
	results to a PSpecContainer object.

	Parameters
	----------
	dsets : list
		Contains UVData objects or string filepaths to miriad files

	filename : str
		Output filepath for HDF5 PSpecContainer object

	groupname : str
		Groupname of the subdirectory in the HDF5 container to store the
		UVPSpec objects in. Default is a concatenation the dset_labels.

	dsets_std : list
		Contains UVData objects or string filepaths to miriad files.
		Default is none.

	dset_labels : list
		List of strings to label the input datasets. These labels form
		the psname of each UVPSpec object. Default is "dset0_x_dset1"
		where 0 and 1 are replaced with the dset index in dsets.
		Note: it is not advised to put underscores in the dset label names,
		as some downstream functions assume this to be the case.

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

	exclude_auto_bls : boolean
		If blpairs is None, redundant baseline groups will be formed and
		all cross-multiplies will be constructed. In doing so, if
		exclude_auto_bls is True, eliminate all instances of a bl crossed
		with itself. Default: False

	exclude_permutations : boolean
		If blpairs is None, redundant baseline groups will be formed and
		all cross-multiplies will be constructed. In doing so, if
		exclude_permutations is True, eliminates instances of
		(bl_B, bl_A) if (bl_A, bl_B) also exists. Default: True

	Nblps_per_group : integer
		If blpairs is None, group blpairs into sub-groups of baseline-pairs
		of this size. See utils.calc_blpair_reds() for details. Default: None

	bl_len_range : len-2 float tuple
		A tuple containing the minimum and maximum baseline length to use
		in utils.calc_blpair_reds call. Only used if blpairs is None.

	bl_deg_range : len-2 float tuple
		A tuple containing the min and max baseline angle (ENU frame in degrees)
		to use in utils.calc_blpair_reds. Total range is between 0 and 180 degrees.

	bl_error_tol : float
		Baseline vector error tolerance when constructing redundant groups.

	beam : PSpecBeam object, UVBeam object or string
		Beam model to use in OQE. Can be a PSpecBeam object or a filepath
		to a beamfits healpix map (see UVBeam)

	cosmo : conversions.Cosmo_Conversions object
		A Cosmo_Conversions object to use as the cosmology when normalizing
		the power spectra. Default is a Planck cosmology.
		See conversions.Cosmo_Conversions for details.

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

	store_cov : boolean, optional
		If True, solve for covariance between bandpowers and store in
		output UVPSpec object.

	overwrite : boolean
		If True, overwrite outputs if they exist on disk.

	verbose : boolean
		If True, report feedback to standard output.

	history : str
		String to add to history of each UVPSpec object.

	Returns
	-------
	psc : PSpecContainer object
		A container for the output UVPSpec objects, which themselves contain the
		power spectra and their metadata.

	ds : PSpecData object
		The PSpecData object used for OQE of power spectrum, with cached weighting
		matrices.
	"""
	# type check
	err_msg = "dsets must be fed as a list of dataset string paths or UVData objects."
	assert isinstance(dsets, (list, tuple, np.ndarray)), err_msg

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

	# Construct dataset pairs to operate on
	Ndsets = len(dsets)
	if dset_pairs is None:
		dset_pairs = list(itertools.combinations(range(Ndsets), 2))

	if dset_labels is None:
		dset_labels = ["dset{}".format(i) for i in range(Ndsets)]
	else:
		assert not np.any(['_' in dl for dl in dset_labels]), "cannot accept underscores in input dset_labels: {}".format(dset_labels)

	# load data if fed as filepaths
	if isinstance(dsets[0], (str, np.str)):
		try:
			# load data into UVData objects if fed as list of strings
			t0 = time.time()
			dsets = _load_dsets(dsets, bls=bls, pols=pols, verbose=verbose)
			utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
					  lvl=1, verbose=verbose)
		except ValueError:
			# at least one of the dset loads failed due to no data being present
			utils.log("One of the dset loads failed due to no data overlap given the bls and pols selection", verbose=verbose)
			return None, None

	err_msg = "dsets must be fed as a list of dataset string paths or UVData objects."
	assert np.all([isinstance(d, UVData) for d in dsets]), err_msg

	# check dsets_std input
	if dsets_std is not None:
		err_msg = "input dsets_std must be a list of UVData objects or filepaths to miriad files"
		assert isinstance(dsets_std,(list, tuple, np.ndarray)), err_msg
		assert len(dsets_std) == Ndsets, "len(dsets_std) must equal len(dsets)"

		# if path strings provided, read in UVData objects.
		if isinstance(dsets_std[0], (str, np.str)):
			try:
				# load data into UVData objects if fed as list of strings
				t0 = time.time()
				dsets_std = _load_dsets(dsets_std, bls=bls, pols=pols, verbose=verbose)
				utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
						  lvl=1, verbose=verbose)
			except ValueError:
				# at least one of the dsets_std loads failed due to no data being present
				utils.log("One of the dsets_std loads failed due to no data overlap given the bls and pols selection", verbose=verbose)
				return None, None

		assert np.all([isinstance(d, UVData) for d in dsets]), err_msg

	# configure polarization
	if pol_pairs is None:
		unique_pols = reduce(operator.and_, [set(d.polarization_array) for d in dsets])
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
	ds = PSpecData(dsets=dsets, wgts=[None for d in dsets], labels=dset_labels, dsets_std=dsets_std, beam=beam)

	# Rephase if desired
	if rephase_to_dset is not None:
		ds.rephase_to_dset(rephase_to_dset)

	# trim dset LSTs
	if trim_dset_lsts:
		ds.trim_dset_lsts()

	# broadcast flags
	if broadcast_dset_flags:
		ds.broadcast_dset_flags(time_thresh=time_thresh)

	# perform Jy to mK conversion if desired
	if Jy2mK:
		ds.Jy_to_mK()

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
			 xants2) = utils.calc_blpair_reds(dsets[dsetp[0]], dsets[dsetp[1]],
											  filter_blpairs=True,
											  exclude_auto_bls=exclude_auto_bls,
											  exclude_permutations=exclude_permutations,
											  Nblps_per_group=Nblps_per_group,
											  bl_len_range=bl_len_range,
											  bl_deg_range=bl_deg_range)
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
					and (_bl2 in dset2_bls or _bls2[::-1] in dset2_bls):
					_bls1.append(_bl1)
					_bls2.append(_bl2)

			bls1_list.append(_bls1)
			bls2_list.append(_bls2)

	# Open PSpecContainer to store all output in
	psc = container.PSpecContainer(filename, mode='rw')

	# assign group name
	if groupname is None:
		groupname = '_'.join(dset_labels)
	
	# Loop over dataset combinations
	for i, dset_idxs in enumerate(dset_pairs):
		# check bls lists aren't empty
		if len(bls1_list[i]) == 0 or len(bls2_list[i]) == 0:
			continue

		# Run OQE
		uvp = ds.pspec(bls1_list[i], bls2_list[i], dset_idxs, pol_pairs,
					   spw_ranges=spw_ranges, n_dlys=n_dlys, store_cov=store_cov,
					   input_data_weight=input_data_weight, norm=norm, taper=taper,
					   history=history, verbose=verbose)

		# Store output
		psname = '{}_x_{}{}'.format(dset_labels[dset_idxs[0]],
									dset_labels[dset_idxs[1]], psname_ext)
		psc.set_pspec(group=groupname, psname=psname, pspec=uvp, 
					  overwrite=overwrite)

	return psc, ds
=======
              psname_ext=None, spw_ranges=None, n_dlys=None, pol_pairs=None, blpairs=None,
              input_data_weight='identity', norm='I', taper='none',
              exclude_auto_bls=False, exclude_permutations=True,
              Nblps_per_group=None, bl_len_range=(0, 1e10), bl_deg_range=(0, 180), bl_error_tol=1.0,
              beam=None, cosmo=None, rephase_to_dset=None, trim_dset_lsts=False, broadcast_dset_flags=True,
              time_thresh=0.2, Jy2mK=False, overwrite=True, verbose=True, store_cov=False, history=''):
    """
    Create a PSpecData object, run OQE delay spectrum estimation and write
    results to a PSpecContainer object.

    Parameters
    ----------
    dsets : list
        Contains UVData objects or string filepaths to miriad files

    filename : str
        Output filepath for HDF5 PSpecContainer object

    groupname : str
        Groupname of the subdirectory in the HDF5 container to store the
        UVPSpec objects in. Default is a concatenation the dset_labels.

    dsets_std : list
        Contains UVData objects or string filepaths to miriad files.
        Default is none.

    dset_labels : list
        List of strings to label the input datasets. These labels form
        the psname of each UVPSpec object. Default is "dset0_x_dset1"
        where 0 and 1 are replaced with the dset index in dsets.
        Note: it is not advised to put underscores in the dset label names,
        as some downstream functions assume this to be the case.

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

    exclude_auto_bls : boolean
        If blpairs is None, redundant baseline groups will be formed and
        all cross-multiplies will be constructed. In doing so, if
        exclude_auto_bls is True, eliminate all instances of a bl crossed
        with itself. Default: False

    exclude_permutations : boolean
        If blpairs is None, redundant baseline groups will be formed and
        all cross-multiplies will be constructed. In doing so, if
        exclude_permutations is True, eliminates instances of
        (bl_B, bl_A) if (bl_A, bl_B) also exists. Default: True

    Nblps_per_group : integer
        If blpairs is None, group blpairs into sub-groups of baseline-pairs
        of this size. See utils.calc_blpair_reds() for details. Default: None

    bl_len_range : len-2 float tuple
        A tuple containing the minimum and maximum baseline length to use
        in utils.calc_blpair_reds call. Only used if blpairs is None.

    bl_deg_range : len-2 float tuple
        A tuple containing the min and max baseline angle (ENU frame in degrees)
        to use in utils.calc_blpair_reds. Total range is between 0 and 180 degrees.

    bl_error_tol : float
        Baseline vector error tolerance when constructing redundant groups.

    beam : PSpecBeam object, UVBeam object or string
        Beam model to use in OQE. Can be a PSpecBeam object or a filepath
        to a beamfits healpix map (see UVBeam)

    cosmo : conversions.Cosmo_Conversions object
        A Cosmo_Conversions object to use as the cosmology when normalizing
        the power spectra. Default is a Planck cosmology.
        See conversions.Cosmo_Conversions for details.

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

    store_cov : boolean, optional
        If True, solve for covariance between bandpowers and store in
        output UVPSpec object.

    overwrite : boolean
        If True, overwrite outputs if they exist on disk.

    verbose : boolean
        If True, report feedback to standard output.

    history : str
        String to add to history of each UVPSpec object.

    Returns
    -------
    psc : PSpecContainer object
        A container for the output UVPSpec objects, which themselves contain the
        power spectra and their metadata.

    ds : PSpecData object
        The PSpecData object used for OQE of power spectrum, with cached weighting
        matrices.
    """
    # type check
    err_msg = "dsets must be fed as a list of dataset string paths or UVData objects."
    assert isinstance(dsets, (list, tuple, np.ndarray)), err_msg

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

    # Construct dataset pairs to operate on
    Ndsets = len(dsets)
    if dset_pairs is None:
        dset_pairs = list(itertools.combinations(range(Ndsets), 2))

    if dset_labels is None:
        dset_labels = ["dset{}".format(i) for i in range(Ndsets)]
    else:
        assert not np.any(['_' in dl for dl in dset_labels]), "cannot accept underscores in input dset_labels: {}".format(dset_labels)

    # load data if fed as filepaths
    if isinstance(dsets[0], (str, np.str)):
        try:
            # load data into UVData objects if fed as list of strings
            t0 = time.time()
            dsets = _load_dsets(dsets, bls=bls, pols=pols, verbose=verbose)
            utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
                      lvl=1, verbose=verbose)
        except ValueError:
            # at least one of the dset loads failed due to no data being present
            utils.log("One of the dset loads failed due to no data overlap given the bls and pols selection", verbose=verbose)
            return None, None

    err_msg = "dsets must be fed as a list of dataset string paths or UVData objects."
    assert np.all([isinstance(d, UVData) for d in dsets]), err_msg

    # check dsets_std input
    if dsets_std is not None:
        err_msg = "input dsets_std must be a list of UVData objects or filepaths to miriad files"
        assert isinstance(dsets_std,(list, tuple, np.ndarray)), err_msg
        assert len(dsets_std) == Ndsets, "len(dsets_std) must equal len(dsets)"

        # if path strings provided, read in UVData objects.
        if isinstance(dsets_std[0], (str, np.str)):
            try:
                # load data into UVData objects if fed as list of strings
                t0 = time.time()
                dsets_std = _load_dsets(dsets_std, bls=bls, pols=pols, verbose=verbose)
                utils.log("Loaded data in %1.1f sec." % (time.time() - t0),
                          lvl=1, verbose=verbose)
            except ValueError:
                # at least one of the dsets_std loads failed due to no data being present
                utils.log("One of the dsets_std loads failed due to no data overlap given the bls and pols selection", verbose=verbose)
                return None, None

        assert np.all([isinstance(d, UVData) for d in dsets]), err_msg

    # configure polarization
    if pol_pairs is None:
        unique_pols = reduce(operator.and_, [set(d.polarization_array) for d in dsets])
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
    ds = PSpecData(dsets=dsets, wgts=[None for d in dsets], labels=dset_labels, dsets_std=dsets_std, beam=beam)

    # Rephase if desired
    if rephase_to_dset is not None:
        ds.rephase_to_dset(rephase_to_dset)

    # trim dset LSTs
    if trim_dset_lsts:
        ds.trim_dset_lsts()

    # broadcast flags
    if broadcast_dset_flags:
        ds.broadcast_dset_flags(time_thresh=time_thresh, spw_ranges=spw_ranges)

    # perform Jy to mK conversion if desired
    if Jy2mK:
        ds.Jy_to_mK()

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
             xants2) = utils.calc_blpair_reds(dsets[dsetp[0]], dsets[dsetp[1]],
                                              filter_blpairs=True,
                                              exclude_auto_bls=exclude_auto_bls,
                                              exclude_permutations=exclude_permutations,
                                              Nblps_per_group=Nblps_per_group,
                                              bl_len_range=bl_len_range,
                                              bl_deg_range=bl_deg_range)
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
                    and (_bl2 in dset2_bls or _bls2[::-1] in dset2_bls):
                    _bls1.append(_bl1)
                    _bls2.append(_bl2)

            bls1_list.append(_bls1)
            bls2_list.append(_bls2)

    # Open PSpecContainer to store all output in
    psc = container.PSpecContainer(filename, mode='rw')

    # assign group name
    if groupname is None:
        groupname = '_'.join(dset_labels)
    
    # Loop over dataset combinations
    for i, dset_idxs in enumerate(dset_pairs):
        # check bls lists aren't empty
        if len(bls1_list[i]) == 0 or len(bls2_list[i]) == 0:
            continue

        # Run OQE
        uvp = ds.pspec(bls1_list[i], bls2_list[i], dset_idxs, pol_pairs,
                       spw_ranges=spw_ranges, n_dlys=n_dlys, store_cov=store_cov,
                       input_data_weight=input_data_weight, norm=norm, taper=taper,
                       history=history, verbose=verbose)

        # Store output
        psname = '{}_x_{}{}'.format(dset_labels[dset_idxs[0]],
                                    dset_labels[dset_idxs[1]], psname_ext)
        psc.set_pspec(group=groupname, psname=psname, pspec=uvp, 
                      overwrite=overwrite)

    return psc, ds
>>>>>>> c50139100555fba73fc7973fdf665a1ba578bd41


def get_pspec_run_argparser():
	a = argparse.ArgumentParser(description="argument parser for pspecdata.pspec_run()")

	def list_of_int_tuples(v):
		v = map(lambda x: tuple(map(int, x.split())), v.split(","))
		return v

	def list_of_str_tuples(v):
		v = map(lambda x: tuple(map(str, x.split())), v.split(","))
		return v

	def list_of_tuple_tuples(v):
		v = map(lambda x: tuple(map(int, x.split())), v.split(","))
		v = map(lambda x: (x[:2], x[2:]), v)
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
	a.add_argument("--exclude_permutations", default=False, action='store_true', help='If blpairs is not provided, exclude a basline-pair permutations. Ex: if (A, B) exists, exclude (B, A).')
	a.add_argument("--Nblps_per_group", default=None, type=int, help="If blpairs is not provided and group == True, set the number of blpairs in each group.")
	a.add_argument("--bl_len_range", default=(0, 1e10), nargs='+', type=float, help="If blpairs is not provided, limit the baselines used based on their minimum and maximum length in meters.")
	a.add_argument("--bl_deg_range", default=(0, 180), nargs='+', type=float, help="If blpairs is not provided, limit the baseline used based on a min and max angle cut in ENU frame in degrees.")
	a.add_argument("--bl_error_tol", default=1.0, type=float, help="If blpairs is not provided, this is the error tolerance in forming redundant baseline groups in meters.")
	a.add_argument("--store_cov", default=False, action='store_true', help="Compute and store covariance of bandpowers given dsets_std files.")
	a.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output if it exists.")
	a.add_argument("--psname_ext", default='', type=str, help="Extension for pspectra name in PSpecContainer.")
	a.add_argument("--verbose", default=False, action='store_true', help="Report feedback to standard output.")
	return a


def validate_blpairs(blpairs, uvd1, uvd2, baseline_tol=1.0, verbose=True):
	"""
	Validate baseline pairings in the blpair list are redundant within the
	specified tolerance.

	Parameters
	----------
	blpairs : list of baseline-pair tuples, Ex. [((1,2),(1,2)), ((2,3),(2,3))]
		See docstring of PSpecData.pspec() for details on format.

	uvd1 : UVData instance containing visibility data that first bl in blpair will draw from

	uvd2 : UVData instance containing visibility data that second bl in blpair will draw from

	baseline_tol : float, distance tolerance for notion of baseline "redundancy" in meters

	verbose : bool, if True report feedback to stdout
	"""
	# ensure uvd1 and uvd2 are UVData objects
	if isinstance(uvd1, UVData) == False:
		raise TypeError("uvd1 must be a UVData instance")
	if isinstance(uvd2, UVData) == False:
		raise TypeError("uvd2 must be a UVData instance")

	# get antenna position dictionary
	ap1, a1 = uvd1.get_ENU_antpos(pick_data_ants=True)
	ap2, a2 = uvd1.get_ENU_antpos(pick_data_ants=True)
	ap1 = dict(zip(a1, ap1))
	ap2 = dict(zip(a2, ap2))

	# ensure shared antenna keys match within tolerance
	shared = sorted(set(ap1.keys()) & set(ap2.keys()))
	for k in shared:
		assert np.linalg.norm(ap1[k] - ap2[k]) <= baseline_tol, "uvd1 and uvd2 don't agree on antenna positions within tolerance of {} m".format(baseline_tol)
	ap = ap1
	ap.update(ap2)

	# iterate through baselines and check baselines crossed with each other are within tolerance
	for i, blg in enumerate(blpairs):
		if isinstance(blg, tuple):
			blg = [blg]
		for blp in blg:
			bl1_vec = ap[blp[0][0]] - ap[blp[0][1]]
			bl2_vec = ap[blp[1][0]] - ap[blp[1][1]]
			if np.linalg.norm(bl1_vec - bl2_vec) >= baseline_tol:
				raise_warning("blpair {} exceeds redundancy tolerance of {} m".format(blp, baseline_tol), verbose=verbose)


def raise_warning(warning, verbose=True):
	'''warning function'''
	if verbose:
		print(warning)


def _load_dsets(fnames, bls=None, pols=None, logf=None, verbose=True):
	""" helper function for loading Miriad datasets in pspec_run """
	dsets = []
	Ndsets = len(fnames)
	for i, dset in enumerate(fnames):
		utils.log("Reading {} / {} datasets...".format(i+1, Ndsets), f=logf, lvl=1, verbose=verbose)
		# read data
		uvd = UVData()
		uvd.read_miriad(glob.glob(dset), bls=bls, polarizations=pols)
		dsets.append(uvd)
	return dsets


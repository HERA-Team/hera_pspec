import numpy as np
from collections import OrderedDict as odict
import random
import copy
import warnings
import argparse
from astropy import stats as astats
import os, sys

from . import utils, version, __version__, uvpspec_utils as uvputils
from .uvpspec import _ordered_unique
from .uvwindow import UVWindow


def group_baselines(bls, Ngroups, keep_remainder=False, randomize=False,
                    seed=None):
    """
    Group baselines together into equal-sized sets.

    These groups can be passed into PSpecData.pspec(), where the corresponding
    baselines will be averaged together (grouping reduces the number of
    cross-spectra that need to be computed).

    Parameters
    ----------
    bls : list of tuples
        Set of unique baselines tuples.

    Ngroups : int
        Number of groups to create. The groups will be equal in size, except
        the last group (if there are remainder baselines).

    keep_remainder : bool, optional
        Whether to keep remainder baselines that don't fit exactly into the
        number of specified groups. If True, the remainder baselines are
        appended to the last group in the output list. Otherwise, the remainder
        baselines are discarded. Default: False.

    randomize : bool, optional
        Whether baselines should be added to groups in the order they appear in
        'bls', or if they should be assigned at random. Default: False.

    seed : int, optional
        Random seed to use if randomize=True. If None, no random seed will be
        set. Default: None.

    Returns
    -------
    grouped_bls : list of lists of tuples
        List of grouped baselines.
    """
    Nbls = len(bls) # Total baselines
    n = Nbls // Ngroups # Baselines per group
    rem = Nbls - n*Ngroups

    # Sanity check on number of groups
    if Nbls < Ngroups: raise ValueError("Can't have more groups than baselines.")

    # Make sure only tuples were provided (can't have groups of groups)
    for bl in bls: assert isinstance(bl, tuple)

    # Randomize baseline order if requested
    if randomize:
        if seed is not None: random.seed(seed)
        bls = copy.deepcopy(bls)
        random.shuffle(bls)

    # Assign to groups sequentially
    grouped_bls = [bls[i*n:(i+1)*n] for i in range(Ngroups)]
    if keep_remainder and rem > 0: grouped_bls[-1] += bls[-rem:]
    return grouped_bls


def sample_baselines(bls, seed=None):
    """
    Sample a set of baselines with replacement (to be used to generate a
    bootstrap sample).

    Parameters
    ----------
    bls : list of either tuples or lists of tuples
        Set of unique baselines to be sampled from. If groups of baselines
        (contained in lists) are provided, each group will be treated as a
        single object by the sampler; its members will not be sampled
        separately.

    seed : int, optional
        Random seed to use if randomize=True. If None, no random seed will be
        set. Default: None.

    Returns
    -------
    sampled_bls : list of tuples or lists of tuples
        Bootstrap-sampled set of baselines (will include multiple instances of
        some baselines).
    """
    if seed is not None: random.seed(seed)

    # Sample with replacement; return as many baselines/groups as were input
    return [random.choice(bls) for i in range(len(bls))]


def average_spectra(uvp_in, blpair_groups=None, time_avg=False,
                    blpair_weights=None, error_field=None,
                    error_weights=None, normalize_weights=True,
                    inplace=True, add_to_history=''):
    """
    Average power spectra across the baseline-pair-time axis, weighted by
    each spectrum's integration time or a specified kind of error bars.

    This is an "incoherent" average, in the sense that this averages power
    spectra, rather than visibility data. The 'nsample_array' and
    'integration_array' will be updated to reflect the averaging.

    In the case of averaging across baseline pairs, the resultant averaged
    spectrum is assigned to the zeroth blpair in the group. In the case of
    time averaging, the time and LST arrays are assigned to the mean of the
    averaging window.

    Note that this is designed to be separate from spherical binning in k:
    here we are not connecting k_perp modes to k_para modes. However, if
    blpairs holds groups of iso baseline separation, then this is
    equivalent to cylindrical binning in 3D k-space.

    If you want help constructing baseline-pair groups from baseline
    groups, see self.get_blpair_groups_from_bl_groups.

    Parameters
    ----------
    uvp_in : UVPSpec
        Input power spectrum (to average over).

    blpair_groups : list of baseline-pair groups
        List of list of tuples or integers. All power spectra in a
        baseline-pair group are averaged together. If a baseline-pair
        exists in more than one group, a warning is raised.

        Ex: blpair_groups = [ [((1, 2), (1, 2)), ((2, 3), (2, 3))],
                              [((4, 6), (4, 6))]]
        or blpair_groups = [ [1002001002, 2003002003], [4006004006] ]

    time_avg : bool, optional
        If True, average power spectra across the time axis. Default: False.

    blpair_weights : list of weights (float or int), optional
        Relative weight of each baseline-pair when performing the average. This
        is useful for bootstrapping. This should have the same shape as
        blpair_groups if specified. The weights are automatically normalized
        within each baseline-pair group. Default: None (all baseline pairs have
        unity weights).

    error_field: string or list, optional
        If errorbars have been entered into stats_array, will do a weighted
        sum to shrink the error bars down to the size of the averaged
        data_array. Error_field strings be keys of stats_array. If list,
        does this for every specified key. Every stats_array key that is
        not specified is thrown out of the new averaged object.

    error_weights: string, optional
         error_weights specify which kind of errors we use for weights
         during averaging power spectra.
         The weights are defined as $w_i = 1/ sigma_i^2$,
         where $sigma_i$ is taken from the relevant field of stats_array.
         If `error_weight' is set to None, which means we just use the
         integration time as weights. If error_weights is specified,
         then it also gets appended to error_field as a list.
         Default: None

    normalize_weights: bool, optional
        Whether to normalize the baseline-pair weights so that:
           Sum(blpair_weights) = N_blpairs
        If False, no normalization is applied to the weights. Default: True.

    inplace : bool, optional
        If True, edit data in self, else make a copy and return. Default:
        True.

    add_to_history : str, optional
        Added text to add to file history.

    Notes
    -----
    Currently, every baseline-pair in a blpair group must have the same
    Ntimes, unless time_avg=True. Future versions may support
    baseline-pair averaging of heterogeneous time arrays. This includes
    the scenario of repeated blpairs (e.g. in bootstrapping), which will
    return multiple copies of their time_array.
    """
    if inplace:
        uvp = uvp_in
    else:
        uvp = copy.deepcopy(uvp_in)

    # Copy these, so we don't modify the input lists
    blpair_groups = copy.deepcopy(blpair_groups)
    blpair_weights = copy.deepcopy(blpair_weights)

    # If blpair_groups were fed in, enforce type and structure
    if blpair_groups is not None:

        # Enforce shape of blpair_groups
        assert isinstance(blpair_groups[0], (list, np.ndarray)), \
              "blpair_groups must be fed as a list of baseline-pair lists. " \
              "See docstring."

        # Convert blpair_groups to list of blpair group integers
        if isinstance(blpair_groups[0][0], tuple):
            new_blpair_grps = [[uvp.antnums_to_blpair(blp) for blp in blpg]
                               for blpg in blpair_groups]
            blpair_groups = new_blpair_grps

        # Get all baseline pairs in uvp object (in integer form)
        uvp_blpairs = [uvp.antnums_to_blpair(blp) for blp in uvp.get_blpairs()]
        blvecs_groups = []
        for group in blpair_groups:
            blvecs_groups.append(uvp.get_blpair_blvecs()[uvp_blpairs.index(group[0])])
        # get baseline length for each group of baseline pairs
        # assuming only redundant baselines are paired together
        blpair_lens, _ = utils.get_bl_lens_angs(blvecs_groups, bl_error_tol=1.)

    else:
        # If not, each baseline pair is its own group
        _, idx = np.unique(uvp.blpair_array, return_index=True)
        blpair_groups = [[blp] for blp in uvp.blpair_array[np.sort(idx)]]
        # get baseline length for each group of baseline pairs
        # assuming only redundant baselines are paired together
        blpair_lens = [blv for blv in uvp.get_blpair_seps()[np.sort(idx)]]
        assert blpair_weights is None, "Cannot specify blpair_weights if "\
                                       "blpair_groups is None."

    # Print warning if a blpair appears more than once in all of blpair_groups
    all_blpairs = [item for sublist in blpair_groups for item in sublist]
    if len(set(all_blpairs)) < len(all_blpairs):
        print("Warning: some baseline-pairs are repeated between blpair "\
              "averaging groups.")

    # Create baseline-pair weights list if not specified
    if blpair_weights is None:
        # Assign unity weights to baseline-pair groups that were specified
        blpair_weights = [[1. for item in grp] for grp in blpair_groups]
    else:
        # Check that blpair_weights has the same shape as blpair_groups
        for i, grp in enumerate(blpair_groups):
            try:
                len(blpair_weights[i]) == len(grp)
            except:
                raise IndexError("blpair_weights must have the same shape as "
                                 "blpair_groups")

    # pre-check for error_weights
    if error_weights is None:
        use_error_weights = False
    else:
        if hasattr(uvp, "stats_array"):
            if error_weights not in uvp.stats_array.keys():
                raise KeyError("error_field \"%s\" not found in stats_array keys." % error_weights)
        use_error_weights = True

    # stat_l is a list of supplied error_fields, to sum over.
    if isinstance(error_field, (list, tuple, np.ndarray)):
        stat_l = list(error_field)
    elif isinstance(error_field, str):
        stat_l = [error_field]
    else:
        stat_l = []
    if use_error_weights:
        if error_weights not in stat_l:
            stat_l.append(error_weights)
    for stat in stat_l:
        if hasattr(uvp, "stats_array"):
            if stat not in uvp.stats_array.keys():
                raise KeyError("error_field \"%s\" not found in stats_array keys." % stat)

    if not uvp.exact_windows:
        # For baseline pairs not in blpair_groups, add them as their own group
        extra_blpairs = set(uvp.blpair_array) - set(all_blpairs)
        blpair_groups += [[blp] for blp in extra_blpairs]
        blpair_weights += [[1.,] for blp in extra_blpairs]

    # Create new data arrays
    data_array, wgts_array = odict(), odict()
    ints_array, nsmp_array = odict(), odict()
    stats_array = odict([[stat, odict()] for stat in stat_l])
    # will average covariance array if present
    store_cov = hasattr(uvp, "cov_array_real")
    if store_cov:
        cov_array_real = odict()
        cov_array_imag = odict()

    # same for window function
    store_window = hasattr(uvp, 'window_function_array')
    if store_window:
        window_function_array = odict()
        window_function_kperp, window_function_kpara = odict(), odict()

    # Iterate over spectral windows
    for spw in range(uvp.Nspws):
        spw_data, spw_wgts, spw_ints, spw_nsmp = [], [], [], []
        spw_stats = odict([[stat, []] for stat in stat_l])
        if store_window:
            spw_window_function = []
            spw_wf_kperp_bins, spw_wf_kpara_bins = [], []
        if store_cov:
            spw_cov_real = []
            spw_cov_imag = []
        # Iterate over polarizations
        for i, p in enumerate(uvp.polpair_array):
            pol_data, pol_wgts, pol_ints, pol_nsmp = [], [], [], []
            pol_stats = odict([[stat, []] for stat in stat_l])
            if store_window:
                pol_window_function = []
            if store_cov:
                pol_cov_real = []
                pol_cov_imag = []

            # Iterate over baseline-pair groups
            for j, blpg in enumerate(blpair_groups):
                bpg_data, bpg_wgts, bpg_ints, bpg_nsmp = [], [], [], []
                bpg_stats = odict([[stat, []] for stat in stat_l])
                if store_window:
                    bpg_window_function = []
                if store_cov:
                    bpg_cov_real = []
                    bpg_cov_imag = []
                w_list = []

                # Sum over all weights within this baseline group to get
                # normalization (if weights specified). The normalization is
                # calculated so that Sum (blpair wgts) = no. baselines.
                if blpair_weights is not None:
                    blpg_wgts = np.array(blpair_weights[j])
                    norm = np.sum(blpg_wgts) if normalize_weights else 1.

                    if norm <= 0.:
                        raise ValueError("Sum of baseline-pair weights in "
                                         "group %d is <= 0." % j)
                    blpg_wgts = blpg_wgts * float(blpg_wgts.size) / norm # Apply normalization
                else:
                    blpg_wgts = np.ones(len(blpg))

                # Iterate within a baseline-pair group and get weighted data
                for k, blp in enumerate(blpg):
                    # Get no. samples and construct integration weight
                    nsmp = uvp.get_nsamples((spw, blp, p))[:, None]
                    # shape of nsmp: (Ntimes, 1)
                    data = uvp.get_data((spw, blp, p))
                    # shape of data: (Ntimes, Ndlys)
                    wgts = uvp.get_wgts((spw, blp, p))
                    # shape of wgts: (Ntimes, Nfreqs, 2)
                    ints = uvp.get_integrations((spw, blp, p))[:, None]
                    # shape of ints: (Ntimes, 1)
                    if store_window:
                        window_function = uvp.get_window_function((spw, blp, p))
                        # shape of window_function if approx.: (Ntimes, Ndlys, Ndlys)
                        # shape of window_function if exact: (Ntimes, Ndlys, Nkperp, Nkpara)
                    if store_cov:
                        cov_real = uvp.get_cov((spw, blp, p), component="real")
                        cov_imag = uvp.get_cov((spw, blp, p), component="imag")
                        # shape of cov: (Ntimes, Ndlys, Ndlys)
                    # Get squared statistic
                    errws = {}
                    for stat in stat_l:
                        errws[stat] = uvp.get_stats(stat, (spw, blp, p)).copy()
                        np.square(errws[stat], out=errws[stat], where=np.isfinite(errws[stat]))
                        # shape of errs: (Ntimes, Ndlys)

                    if use_error_weights:
                        # If use_error_weights==True, all arrays are weighted by a specified kind of errors,
                        # including the error_filed in stats_array and cov_array.
                        # For each power spectrum P_i with error_weights sigma_i,
                        # P_avg = \sum{ P_i / (sigma_i)^2 } / \sum{ 1 / (sigma_i)^2 }
                        # while for other variance or covariance terms epsilon_i stored in stats_array and cov_array,
                        # epsilon_avg = \sum{ (epsilon_i / (sigma_i)^4 } / ( \sum{ 1 / (sigma_i)^2 } )^2
                        # For reference: M. Tegmark 1997, The Astrophysical Journal Letters, 480, L87, Table 1, #3
                        # or J. Dillon 2014, Physical Review D, 89, 023002 , Equation 34.
                        stat_val = uvp.get_stats(error_weights, (spw, blp, p)).copy().real #shape (Ntimes, Ndlys)
                        np.square(stat_val, out=stat_val, where=np.isfinite(stat_val))
                        #corrects for potential nan values
                        stat_val = np.nan_to_num(stat_val, copy=False, nan=np.inf, posinf=np.inf)
                        w = np.real(1. / stat_val.clip(1e-40, np.inf))
                        # shape of w: (Ntimes, Ndlys)
                    else:
                        # Otherwise all arrays are averaged in a way weighted by the integration time,
                        # including the error_filed in stats_array and cov_array.
                        # Since P_N ~ Tsys^2 / sqrt{N_incoherent} t_int (see N. Kern, The Astrophysical Journal 888.2 (2020): 70, Equation 7),
                        # we choose w ~ P_N^{-2} ~ (ints * sqrt{nsmp})^2
                        w = (ints * np.sqrt(nsmp))**2
                        # shape of w: (Ntimes, 1)

                    # Take time average if desired
                    if time_avg:
                        wsum = np.sum(w, axis=0).clip(1e-40, np.inf)
                        data = (np.sum(data * w, axis=0) \
                                / wsum)[None]
                        wgts = (np.sum(wgts * w[:, :1, None], axis=0) \
                                / wsum[:1, None])[None]
                        # wgts has a shape of (Ntimes, Nfreqs, 2), while
                        # w has a shape of (Ntimes, Ndlys) or (Ntimes, 1)
                        # To handle with the case  when Nfreqs != Ntimes,
                        # we choose to multiply wgts with w[:,:1,None].
                        ints = (np.sum(ints * w, axis=0) \
                                / wsum)[None]
                        nsmp = np.sum(nsmp, axis=0)[None]
                        if store_window:
                            if uvp.exact_windows:
                                window_function = (np.sum(window_function * w[:, :, None, None], axis=0)\
                                                    / (wsum)[:, None, None])[None]
                            if not uvp.exact_windows:
                                window_function = (np.sum(window_function * w[:, :, None], axis=0) \
                                                   / (wsum)[:, None])[None]
                        if store_cov:
                            cov_real = (np.sum(cov_real * w[:, :, None] * w[:, None, :], axis=0) \
                                   / wsum[:, None] / wsum[None, :])[None]
                            cov_imag = (np.sum(cov_imag * w[:, :, None] * w[:, None, :], axis=0) \
                                   / wsum[:, None] / wsum[None, :])[None]
                        for stat in stat_l:
                            # clip errws to eliminate nan: inf * 0 yields nans
                            weighted_errws = errws[stat].clip(0, 1e40) * w**2
                            errws[stat] = (np.sum(weighted_errws, axis=0) \
                                           / wsum**2)[None]
                            # set near-zero errws to inf, as they should be
                            errws[stat][np.isclose(errws[stat], 0)] = np.inf
                        w = np.sum(w, axis=0)[None]
                        # Above we use the clip method for zero weights. A tolerance
                        # as low as 1e-40 works when using inverse square of noise power
                        # as weights.
                    # Add multiple copies of data for each baseline according
                    # to the weighting/multiplicity;
                    # while multiple copies are only added when bootstrap resampling
                    for m in range(int(blpg_wgts[k])):
                        bpg_data.append(data * w)
                        bpg_wgts.append(wgts * w[:, :1, None])
                        bpg_ints.append(ints * w)
                        bpg_nsmp.append(nsmp)
                        for stat in stat_l:
                            # clip errws for same reason above
                            bpg_stats[stat].append(errws[stat].clip(0, 1e40) * w**2)
                        if store_window:
                            if uvp.exact_windows:
                                bpg_window_function.append(window_function * w[:, :, None, None])
                            else:
                                bpg_window_function.append(window_function * w[:, :, None])
                        if store_cov:
                            bpg_cov_real.append(cov_real * w[:, :, None] * w[:, None, :])
                            bpg_cov_imag.append(cov_imag * w[:, :, None] * w[:, None, :])
                        w_list.append(w)

                # normalize sum: clip to deal with w_list_sum == 0
                w_list_sum = np.sum(w_list, axis=0).clip(1e-40, np.inf)
                bpg_data = np.sum(bpg_data, axis=0) / w_list_sum
                bpg_wgts = np.sum(bpg_wgts, axis=0) / w_list_sum[:,:1, None]
                bpg_nsmp = np.sum(bpg_nsmp, axis=0)
                bpg_ints = np.sum(bpg_ints, axis=0) / w_list_sum
                if store_cov:
                    bpg_cov_real = np.sum(bpg_cov_real, axis=0) / w_list_sum[:, :, None] / w_list_sum[:, None, :]
                    bpg_cov_imag = np.sum(bpg_cov_imag, axis=0) / w_list_sum[:, :, None] / w_list_sum[:, None, :]
                for stat in stat_l:
                    stat_avg = np.sum(bpg_stats[stat], axis=0) / w_list_sum**2
                    # set near-zero stats to inf, as they should be
                    stat_avg[np.isclose(stat_avg, 0)] = np.inf
                    # take sqrt to get back to stat units
                    bpg_stats[stat] = np.sqrt(stat_avg)
                if store_window:
                    if uvp.exact_windows:
                        bpg_window_function = np.sum(bpg_window_function, axis=0) # / w_list_sum[:, :, None, None]
                    else:
                        bpg_window_function = np.sum(bpg_window_function, axis=0) / w_list_sum[:, :, None]
                # Append to lists (polarization)
                pol_data.extend(bpg_data); pol_wgts.extend(bpg_wgts)
                pol_ints.extend(bpg_ints); pol_nsmp.extend(bpg_nsmp)
                for stat in stat_l:
                    pol_stats[stat].extend(bpg_stats[stat])
                if store_window:
                    pol_window_function.extend(bpg_window_function)
                if store_cov:
                    pol_cov_real.extend(bpg_cov_real)
                    pol_cov_imag.extend(bpg_cov_imag)
            # Append to lists (spectral window)
            spw_data.append(pol_data); spw_wgts.append(pol_wgts)
            spw_ints.append(pol_ints); spw_nsmp.append(pol_nsmp)
            for stat in stat_l:
                spw_stats[stat].append(pol_stats[stat])
            if store_window:
                spw_window_function.append(pol_window_function)
                if uvp.exact_windows:
                    spw_wf_kperp_bins.append(uvp.window_function_kperp[spw][:, i])
                    spw_wf_kpara_bins.append(uvp.window_function_kpara[spw][:, i])
            if store_cov:
                spw_cov_real.append(pol_cov_real)
                spw_cov_imag.append(pol_cov_imag)

        # Append to dictionaries
        data_array[spw] = np.moveaxis(spw_data, 0, -1)
        wgts_array[spw] = np.moveaxis(spw_wgts, 0, -1)
        ints_array[spw] = np.moveaxis(spw_ints, 0, -1)[:, 0, :]
        nsmp_array[spw] = np.moveaxis(spw_nsmp, 0, -1)[:, 0, :]
        for stat in stat_l:
            stats_array[stat][spw] = np.moveaxis(spw_stats[stat], 0, -1)
        if store_window:
            window_function_array[spw] = np.moveaxis(spw_window_function, 0, -1)
            if uvp.exact_windows:
                window_function_kperp[spw] = np.moveaxis(spw_wf_kperp_bins, 0, -1)
                window_function_kpara[spw] = np.moveaxis(spw_wf_kpara_bins, 0, -1)
        if store_cov:
            cov_array_real[spw] = np.moveaxis(np.array(spw_cov_real), 0, -1)
            cov_array_imag[spw] = np.moveaxis(np.array(spw_cov_imag), 0, -1)

    # Iterate over blpair groups one more time to assign metadata
    time_1, time_2, time_avg_arr  = [], [], []
    lst_1, lst_2, lst_avg_arr = [], [], []
    blpair_arr, bl_arr = [], []

    for i, blpg in enumerate(blpair_groups):

        # Get blpairts indices for zeroth blpair in this group
        blpairts = uvp.blpair_to_indices(blpg[0])

        # Assign meta-data
        bl_arr.extend(list(uvputils._blpair_to_bls(blpg[0])))
        if time_avg:
            blpair_arr.append(blpg[0])
            time_1.extend([np.mean(uvp.time_1_array[blpairts])])
            time_2.extend([np.mean(uvp.time_2_array[blpairts])])
            time_avg_arr.extend([np.mean(uvp.time_avg_array[blpairts])])
            lst_1.extend([np.mean(np.unwrap(uvp.lst_1_array[blpairts]))%(2*np.pi)])
            lst_2.extend([np.mean(np.unwrap(uvp.lst_2_array[blpairts]))%(2*np.pi)])
            lst_avg_arr.extend([np.mean(np.unwrap(uvp.lst_avg_array[blpairts]))%(2*np.pi)])
        else:
            blpair_arr.extend(np.ones_like(blpairts, int) * blpg[0])
            time_1.extend(uvp.time_1_array[blpairts])
            time_2.extend(uvp.time_2_array[blpairts])
            time_avg_arr.extend(uvp.time_avg_array[blpairts])
            lst_1.extend(uvp.lst_1_array[blpairts])
            lst_2.extend(uvp.lst_2_array[blpairts])
            lst_avg_arr.extend(uvp.lst_avg_array[blpairts])

    # Update arrays
    bl_arr = np.array(sorted(set(bl_arr)))
    bl_vecs = np.array([uvp.bl_vecs[uvp.bl_array.tolist().index(bl)]
                        for bl in bl_arr])

    # Assign arrays and metadata to UVPSpec object
    uvp.Ntimes = len(np.unique(time_avg_arr))
    uvp.Nblpairts = len(time_avg_arr)
    uvp.Nblpairs = len(np.unique(blpair_arr))
    uvp.Nbls = len(bl_arr)

    # Baselines
    uvp.bl_array = bl_arr
    uvp.bl_vecs = bl_vecs
    uvp.blpair_array = np.array(blpair_arr)

    # Times
    uvp.time_1_array = np.array(time_1)
    uvp.time_2_array = np.array(time_2)
    uvp.time_avg_array = np.array(time_avg_arr)

    # LSTs
    uvp.lst_1_array = np.array(lst_1)
    uvp.lst_2_array = np.array(lst_2)
    uvp.lst_avg_array = np.array(lst_avg_arr)

    # Data, weights, and no. samples
    uvp.data_array = data_array
    uvp.integration_array = ints_array
    uvp.wgt_array = wgts_array
    uvp.nsample_array = nsmp_array
    if store_window:
        uvp.window_function_array = window_function_array
        if uvp.exact_windows:
            uvp.window_function_kperp = window_function_kperp
            uvp.window_function_kpara = window_function_kpara
    if store_cov:
        uvp.cov_array_real = cov_array_real
        uvp.cov_array_imag = cov_array_imag
    if len(stat_l) >=1 :
        uvp.stats_array = stats_array
    elif hasattr(uvp, "stats_array"):
        delattr(uvp, "stats_array")

    # Add to history
    uvp.history = "Spectra averaged with hera_pspec [{}]\n{}\n{}\n{}".format(__version__, add_to_history, '-'*40, uvp.history)
    # Validity check
    uvp.check()

    # Return
    if inplace == False:
        return uvp


def spherical_average(uvp_in, kbins, bin_widths, blpair_groups=None, time_avg=False, blpair_weights=None,
                      weight_by_cov=False, error_weights=None,
                      add_to_history='', little_h=True, A={}, run_check=True):
    """
    Perform a spherical average of a UVPSpec, mapping k_perp & k_para onto a |k| grid.
    Use UVPSpec.set_stats_slice to downweight regions of k_perp and k_para grid before averaging.

    Parameters
    ----------
    uvp_in : UVPSpec object
        Input UVPSpec to average

    kbins : array-like
        1D float array of ascending |k| bin centers in [h] Mpc^-1 units
        (h included if little_h is True)

    bin_widths : array-like
        1D float array of kbin widths for each element in kbins

    blpair_groups : list of tuples, optional
        blpair_groups to average if fed (cylindrical binning)

    time_avg : bool, optional
        Time average the power spectra before spherical average if True

    blpair_weights : list, optional
        relative weights of blpairs in blpair averaging (used for bootstrapping)

    weight_by_cov : bool, optional
        If True, weight spherical average by stored bandpower covariance. Supersedes
        error_weights if provided.

    error_weights : str, optional
        Error field to use as weights in averaging. Weight is 1/err^2.
        If not specified, perform a uniform average.

    add_to_history : str, optional
        String to append to object history

    little_h : bool, optional
        If True, kgrid is in h Mpc^-1 units, otherwise just Mpc^-1 units.
        If False, user must ensure adopted h is consistent with uvp_in.cosmo

    A : dict, optional
        Empty dict to populate with A matrix. This is useful for debugging,
        or if you'd like to look at the A matrix used for the average.
        Default is empty and not saved to globals.

    run_check : bool, optional
        If True, run UVPSpec.check() on resultant object

    Returns
    --------
    UVPSpec object
        Spherically averaged UVPSpec object

    Notes
    -----
    1. The full kgrid (magnitude) is represented as kparas in the averaged object, and can be accessed
    via uvp_avg.get_kparas().

    2. If p_cyl = A p_sph, then the binned data, window function, and bandpower covariance are
        p_sph = H.T p_cyl
        C_sph = H.T C_cyl H
        W_sph = H.T W_cyl A
        where H = E.T A [A.T E A]^-1, and E is the bandpower weight matrix.

    3. For speed, it helps to perform cylindrical binning upfront by suppyling blpair_groups.
    """
    # input checks
    if weight_by_cov:
        assert hasattr(uvp_in, 'cov_array_real'), "cannot weight by cov with no cov_array_real"

    if isinstance(bin_widths, (float, int)):
        bin_widths = np.ones_like(kbins) * bin_widths

    # copy input
    uvp = copy.deepcopy(uvp_in) 

    # transform kgrid to little_h units
    if not little_h:
        kbins = kbins / uvp.cosmo.h
        bin_widths = bin_widths / uvp.cosmo.h

    # ensure bins don't overlap
    assert len(kbins) == len(bin_widths)
    kbin_left = kbins - bin_widths / 2
    kbin_right = kbins + bin_widths / 2
    assert np.all(kbin_left[1:] >= kbin_right[:-1] - 1e-6), "kbins must not overlap"

    # perform time and cylindrical averaging upfront if requested
    if not uvp.exact_windows and (blpair_groups is not None or time_avg):
        uvp.average_spectra(blpair_groups=blpair_groups, time_avg=time_avg,
                            blpair_weights=blpair_weights, error_weights=error_weights,
                            inplace=True)

    # initialize blank arrays and dicts
    Nk = len(kbins)
    dlys_array, spw_dlys_array = [], []
    data_array, wgt_array, integration_array, nsample_array = odict(), odict(), odict(), odict()
    store_stats = hasattr(uvp, 'stats_array')
    store_cov = hasattr(uvp, "cov_array_real")
    store_window = hasattr(uvp, 'window_function_array') or uvp.exact_windows
    if store_cov:
        cov_array_real = odict()
        cov_array_imag = odict()
        # spherical averaged cov_array_imag is set to be zero
    if store_stats:
        stats_array = odict([[stat, odict()] for stat in uvp.stats_array.keys()])
    if store_window:
        window_function_array = odict()

    # iterate over spectral windows
    spw_ranges = uvp.get_spw_ranges()
    for spw in uvp.spw_array:
        # setup non delay-based arrays for this spw
        spw_range = spw_ranges[spw]
        wgt_array[spw] = np.zeros((uvp.Ntimes, spw_range[2], 2, uvp.Npols), dtype=np.float64)
        integration_array[spw] = np.zeros((uvp.Ntimes, uvp.Npols), dtype=np.float64)
        nsample_array[spw] = np.zeros((uvp.Ntimes, uvp.Npols), dtype=np.float64)

        # setup arrays with delays in them
        Ndlys = spw_range[3]
        Ndlyblps = Ndlys * uvp.Nblpairs
        data_array[spw] = np.zeros((uvp.Ntimes, Ndlyblps, uvp.Npols), dtype=np.complex128)
        if store_cov:
            cov_array_real[spw] = np.zeros((uvp.Ntimes, Ndlyblps, Ndlyblps, uvp.Npols), dtype=np.float64)
            cov_array_imag[spw] = np.zeros((uvp.Ntimes, Ndlyblps, Ndlyblps, uvp.Npols), dtype=np.float64)
        if store_stats:
            for stat in uvp.stats_array.keys():
                stats_array[stat][spw] = np.zeros((uvp.Ntimes, Ndlyblps, uvp.Npols), dtype=np.complex128)
        if store_window:
            if uvp.exact_windows:
                window_function_array[spw] = np.zeros((uvp.Ntimes, Nk, Nk, uvp.Npols), dtype=np.float64)
            else:
                window_function_array[spw] = np.zeros((uvp.Ntimes, Ndlyblps, Ndlyblps, uvp.Npols), dtype=np.float64)

        # setup the design matrix: P_cyl = A P_sph
        A[spw] = np.zeros((uvp.Ntimes, Ndlyblps, Nk, uvp.Npols), dtype=np.float64)

        # setup weighting matrix: block diagonal for each Ndly x Ndly
        # we can represent the Ndlyblps x Ndlyblps block diagonal matrix as Ndlyblps x Ndlys
        E = np.zeros((uvp.Ntimes, Ndlyblps, Ndlys, uvp.Npols), dtype=np.float64)


        # get kperps for this spw: shape (Nblpairts,)
        kperps = uvp.get_kperps(spw, little_h=True)

        # get kparas for this spw: shape (Ndlys,)
        kparas = uvp.get_kparas(spw, little_h=True)

        # get k to tau mapping for this spw
        avgz = uvp.cosmo.f2z(np.mean(uvp.freq_array[uvp.spw_to_freq_indices(spw)]))
        t2k = uvp.cosmo.tau_to_kpara(avgz, little_h=True)
        taus = kbins / t2k
        dlys_array.extend(taus)

        # store kbins as delay bins
        spw_dlys_array.extend(np.ones_like(taus, dtype=int) * spw)

        # iterate over blpairs
        for b, blp in enumerate(uvp.get_blpairs()):
            # get blpair time indices
            blpt_inds = uvp.blpair_to_indices(blp)

            # get k magnitude of data: (Ndlys,)
            kmags = np.sqrt(kperps[blpt_inds][0]**2 + kparas**2)

            # shape of nsmap: (Ntimes, Npols)
            nsmp = uvp.nsample_array[spw][blpt_inds]

            # shape of wgts: (Ntimes, spw_Nfreqs, 2, Npols)
            wgts = uvp.wgt_array[spw][blpt_inds]

            # shape of ints: (Ntimes, Npols)
            ints = uvp.integration_array[spw][blpt_inds]

            # get Ndlyblps slice
            dstart, dstop = Ndlys * b, Ndlys * (b + 1)
            dslice = slice(dstart, dstop)

            # store data
            data_array[spw][:, dslice] = uvp.data_array[spw][blpt_inds]

            if store_window and not uvp.exact_windows:
                window_function_array[spw][:, dslice, dslice] = uvp.window_function_array[spw][blpt_inds]

            if store_stats:
                for stat in stats_array:
                    stats_array[stat][spw][:, dslice] = uvp.stats_array[stat][spw][blpt_inds]

            if store_cov:
                cov_array_real[spw][:, dslice, dslice] = uvp.cov_array_real[spw][blpt_inds]

            # fill weighting matrix E
            if weight_by_cov:
                # weight by inverse (real) covariance
                for p in range(uvp.Npols):
                    # the covariance is block diagonal assuming no correlations between baseline-pairs
                    E[:, dslice, :, p] = np.linalg.pinv(cov_array_real[spw][:, dslice, dslice, p].real)

            elif error_weights is not None:
                # fill diagonal with by 1/stats_array^2 as weight
                stat_weight = stats_array[error_weights][spw][:, dslice].real.copy()
                np.square(stat_weight, out=stat_weight, where=np.isfinite(stat_weight))
                E[:, range(dstart, dstop), range(0, Ndlys)] = 1 / stat_weight.clip(1e-40, np.inf)

            else:
                E[:, range(dstart, dstop), range(0, Ndlys)] = 1.0
                f = np.isclose(uvp.integration_array[spw][blpt_inds] * uvp.nsample_array[spw][blpt_inds], 0)
                E[:, range(dstart, dstop), range(0, Ndlys)] *= (~f[:, None, :])

            # append to non-dly arrays
            Emean = np.trace(E[:, dslice, :], axis1=1, axis2=2)  # use sum of E across delay as weight
            wgt_array[spw] += wgts * Emean[:, None, None, :]
            integration_array[spw] += ints * Emean
            nsample_array[spw] += nsmp

            # get k_sph -> k_cyl mapping
            for i, kmag in enumerate(kmags):
                kind = (kbin_left < kmag) & (kbin_right >= kmag)

                if not np.any(kind):
                    # skip if not in any kbins
                    continue
                else:
                    # convert kind into an integer for indexing
                    kind = np.where(kind)[0][0]

                # populate A matrix
                A[spw][:, i + Ndlys * b, kind, :] = 1.0

        # normalize metadata sums
        wgt_array[spw] /= np.max(wgt_array[spw], axis=(2, 3), keepdims=True).clip(1e-40, np.inf)
        integration_array[spw] /= np.trace(E.real, axis1=1, axis2=2).clip(1e-40, np.inf)

        # project onto spherically binned space
        # note: matmul (@) is generally as fast or many times faster than einsum here
        # first compute: H = E A [A.T E A]^-1
        # move axes to enable matmul and inv over Ndlyblps and Nk axes
        # Am shape (Npols, Ntimes, Ndlyblps, Nk)
        # Em shape (Npols, Ntimes, Ndlyblps, Ndlys)
        # Ht shape (Npols, Ntimes, Nk, Ndlyblps)
        Am = np.moveaxis(A[spw], -1, 0)
        Em = np.moveaxis(E, -1, 0)
        # Multiply block diagoinal Em @ Am
        # by applying each baseline block in Em
        # to each Ndly x Nk baseline-horizontal block in Am
        EmAm = np.zeros_like(Am)
        for t in range(uvp.Ntimes):
            for p in range(uvp.Npols):
                for b in range(uvp.Nblpairs):
                    blpslice = slice(Ndlys * b, Ndlys * (b + 1))
                    EmAm[p, t, blpslice] = Em[p, t, blpslice] @ Am[p, t, blpslice]
        invAEA = np.linalg.pinv(Am.transpose(0, 1, 3, 2) @ EmAm)
        H = EmAm @ invAEA
        Ht = H.transpose(0, 1, 3, 2)

        # bin data: p_sph = H.T p_cyl
        # dm shape (Npols, Ntimes, Ndlyblps)
        dm = np.moveaxis(data_array[spw], -1, 0)
        dm = (Ht @ dm[:, :, :, None])[:, :, :, 0]
        data_array[spw] = np.moveaxis(dm, 0, -1)

        if store_window and not uvp.exact_windows:
            # bin window function: W_sph = H.T W_cyl A
            # wm shape (Npols, Ntimes, Ndlyblps, Ndlyblps)
            wm = np.moveaxis(window_function_array[spw], -1, 0)
            wm = Ht @ wm @ Am
            window_function_array[spw] = np.moveaxis(wm, 0, -1)

        if store_stats:
            # bin stats: C_sph = H.T C_cyl H
            for stat in stats_array:
                # get squared stat and clip infs b/c linalg doesn't like them
                sq_stat = stats_array[stat][spw].copy()
                np.square(sq_stat, out=sq_stat, where=np.isfinite(sq_stat))
                # einsum is fast enough for this, and is more succinct than matmul
                avg_stat = np.sqrt(np.einsum("ptik,tip,ptik->tkp", H, sq_stat.clip(0, 1e40), H))
                # set zeroed stats to large number
                avg_stat[np.isclose(avg_stat, 0)] = 1e40
                # update stats_array
                stats_array[stat][spw] = avg_stat

        if store_cov:
            # bin covariance: C_sph = H.T C_cyl H
            # cm shape (Npols, Ntimes, Ndlyblps, Ndlyblps)
            cm = np.moveaxis(cov_array_real[spw], -1, 0)
            cm = Ht @ cm.clip(-1e40, 1e40) @ H  # clip infs
            cov_array_real[spw] = np.moveaxis(cm, 0, -1)
            cov_array_imag[spw] = np.zeros_like(cov_array_real[spw])

        if uvp.exact_windows:
            window_function_array[spw] = spherical_wf_from_uvp(uvp, kbins, bin_widths,
                                                               blpair_groups=blpair_groups,
                                                               blpair_weights=blpair_weights,
                                                               time_avg=time_avg,
                                                               error_weights=error_weights,
                                                               spw_array=spw,
                                                               little_h=True,
                                                               verbose=True)[spw]

    # handle data arrays
    uvp.data_array = data_array
    uvp.integration_array = integration_array
    uvp.nsample_array = nsample_array
    uvp.wgt_array = wgt_array
    if store_cov:
        uvp.cov_array_real = cov_array_real
        uvp.cov_array_imag = cov_array_imag
    if store_stats:
        uvp.stats_array = stats_array
    if store_window:
        uvp.window_function_array = window_function_array

    # handle spw metadata
    uvp.Nspwdlys = len(spw_dlys_array)
    uvp.Ndlys = len(np.unique(dlys_array))
    uvp.dly_array = np.asarray(dlys_array)
    uvp.spw_dly_array = np.asarray(spw_dlys_array)

    # handle baseline metadata: use first blpair as representative blpair
    blp = uvp.blpair_array[0]
    blp_inds = uvp.blpair_to_indices(blp)
    uvp.blpair_array = uvp.blpair_array[blp_inds]
    uvp.Nblpairts = uvp.Ntimes
    uvp.Nblpairs = 1
    bl_array = np.unique([uvp.antnums_to_bl(an) for an in uvp.blpair_to_antnums(blp)])
    uvp.bl_vecs = np.asarray([uvp.bl_vecs[np.argmin(uvp.bl_array - bl)] for bl in bl_array])
    uvp.bl_array = bl_array
    uvp.Nbls = len(bl_array)
    uvp.label_1_array = uvp.label_1_array[:, blp_inds]
    uvp.label_2_array = uvp.label_2_array[:, blp_inds]

    # set bl_vecs mag to zero
    # k_mag stored as k_paras for spherically averaged uvp by convention!
    uvp.bl_vecs[:] = 0.0

    # handle other metadata
    uvp.time_avg_array = np.unique(uvp_in.time_avg_array)
    uvp.time_1_array = np.unique(uvp_in.time_1_array)
    uvp.time_2_array = np.unique(uvp_in.time_2_array)
    uvp.lst_avg_array = np.unique(uvp_in.lst_avg_array)
    uvp.lst_1_array = np.unique(uvp_in.lst_1_array)
    uvp.lst_2_array = np.unique(uvp_in.lst_2_array)

    uvp.history = "Spherically averaged with hera_pspec [{}]\n{}\n{}\n{}".format(__version__, add_to_history, '-'*40, uvp.history)

    # validity check
    if run_check:
        uvp.check()

    return uvp

def spherical_wf_from_uvp(uvp_in, kbins, bin_widths,
                          blpair_groups=None, blpair_lens=None, blpair_weights=None,
                          error_weights=None, time_avg=False, spw_array=None,
                          little_h=True, verbose=False):
    
    """
    Obtains exact spherical window functions from an UVPspec object,
    given a set of baseline-pair groups, their associated lengths, and 
    a set of spherical k-bins.

    Parameters
    ----------
    uvp_in : UVPSpec object
        Input UVPSpec to average

    kbins : array-like
        1D float array of ascending |k| bin centers in [h] Mpc^-1 units
        (h included if little_h is True)

    bin_widths : array-like
        1D float array of kbin widths for each element in kbins

    blpair_groups : list of tuples,
        blpair_groups to average if fed (cylindrical binning)

    blpair_weights : list
        relative weights of blpairs in blpair averaging (used for bootstrapping)
    
    blpair_lens : list 
        lengths of blpairs in blpair_groups

    error_weights : str, optional
        Error field to use as weights in averaging. Weight is 1/err^2.
        If not specified, perform a uniform average.

    time_avg : bool, optional
        Time average the power spectra before spherical average if True

    spw_array : list of ints.
        Spectral window indices.

    little_h : bool, optional
        If True, kbins is in h Mpc^-1 units, otherwise just Mpc^-1 units.
        The code ensures adopted h is consistent with uvp_in.cosmo. If not,
        it modifies the unit of kbins.

    verbose : bool, optional
        If True, print progress, warnings and debugging info to stdout.
        If None, value used is the class attribute.

    Returns
    --------
    wf_spherical : array
        Array of spherical window functions.
        Shape (nbinsk, nbinsk).

    """

    # input checks

    if isinstance(bin_widths, (float, int)):
        bin_widths = np.ones_like(kbins) * bin_widths

    # if window functions have been computed without little h
    # it is not possible to re adjust so kbins need to be in Mpc-1
    # and reciprocally
    if little_h != ('h^-3' in uvp_in.norm_units):
        warnings.warn('Changed little_h units to make kbins consistent ' \
                      'with uvp.window_function_array. Might be inconsistent ' \
                      'with the power spectrum units.')
        if little_h:
            kbins *= uvp_in.cosmo.h 
            bin_widths *= uvp_in.cosmo.h 
        else:
            kbins /= uvp_in.cosmo.h
            bin_widths /= uvp_in.cosmo.h
        little_h = 'h^-3' in uvp_in.norm_units

    # ensure bins don't overlap
    assert len(kbins) == len(bin_widths)
    kbin_left = kbins - bin_widths / 2
    kbin_right = kbins + bin_widths / 2
    assert np.all(kbin_left[1:] >= kbin_right[:-1] - 1e-6), "kbins must not overlap"
    Nk = len(kbins)

    # copy input
    uvp = copy.deepcopy(uvp_in) 

    if blpair_groups is None:
        if blpair_lens is not None:
            warnings.warn('blpair_lens given but blpair_groups is None... overriding blpair_lens.')
        blpair_groups, blpair_lens, _ = uvp.get_red_blpairs()
    else:
        # Enforce shape of blpair_groups
        assert isinstance(blpair_groups[0], (list, np.ndarray)), \
                  "blpair_groups must be fed as a list of baseline-pair lists. " \
                  "See docstring."
        if blpair_lens is None:
            # Get all baseline pairs in uvp object (in integer form)
            uvp_blpairs = [uvp.antnums_to_blpair(blp) for blp in uvp.get_blpairs()]
            blvecs_groups = []
            for group in blpair_groups:
                blvecs_groups.append(uvp.get_blpair_blvecs()[uvp_blpairs.index(group[0])])
            # get baseline length for each group of baseline pairs
            # assuming only redundant baselines are paired together
            blpair_lens, _ = utils.get_bl_lens_angs(blvecs_groups, bl_error_tol=1.)
        else:     
            # ensure consistency between inputs
            assert len(blpair_groups)==len(blpair_lens), "Baseline-pair groups" \
                        " are inconsistent with baseline lengths"
    blpair_lens = np.array(blpair_lens)

    # check spw input and create array of spws to loop over
    if spw_array is None:
        # if no spw specified, use attribute
        spw_array = uvp.spw_array
    else:
        spw_array = spw_array if isinstance(spw_array, (list, tuple, np.ndarray)) else [int(spw_array)]
        # check if spw given is in uvp
        assert np.all([spw in uvp.spw_array for spw in spw_array]), \
               "input spw is not in UVPSpec.spw_array."

    assert uvp.exact_windows, "Need to compute exact window functions first."

    if blpair_weights is None:
        # assign weight of one to each baseline length
        blpair_weights = [[1. for item in grp] for grp in blpair_groups]

    # perform redundant cylindrical averaging upfront
    # and apply weights to window functions
    uvp.average_spectra(blpair_groups=blpair_groups,
                        blpair_weights=blpair_weights,
                        error_weights=error_weights,
                        time_avg=time_avg,
                        inplace=True)

    # initialize blank arrays and dicts
    window_function_array = odict()

    # iterate over spectral windows
    for spw in spw_array:

        avg_nu = (uvp.get_spw_ranges(spw)[0][1]+uvp.get_spw_ranges(spw)[0][0])/2

        # construct array giving the k probed by each baseline-tau pair
        kperps = uvp.cosmo.bl_to_kperp(uvp.cosmo.f2z(avg_nu), little_h=little_h) * blpair_lens
        kparas = uvp.cosmo.tau_to_kpara(uvp.cosmo.f2z(avg_nu), little_h=little_h) * uvp.get_dlys(spw)
        kmags = np.sqrt(kperps[:, None]**2 + kparas**2)

        # setup arrays 
        window_function_array[spw] = np.zeros((uvp.Ntimes, Nk, Nk, uvp.Npols), dtype=np.float64)

        # iterate over polarisation
        spw_window_function = []
        for ip, polpair in enumerate(uvp.polpair_array):

            # grids used to compute the window functions
            kperp_bins = uvp.window_function_kperp[spw][:, ip]
            kpara_bins = uvp.window_function_kpara[spw][:, ip]
            ktot = np.sqrt(kperp_bins[:, None]**2 + kpara_bins**2)

            cyl_wf = uvp.window_function_array[spw][..., ip]
            # separate baseline-time axis to iterate over times
            cyl_wf = cyl_wf.reshape((uvp.Ntimes, uvp.Nblpairs, *cyl_wf.shape[1:]))

            # take average for each time
            for it in range(uvp.Ntimes):
                wf_spherical = np.zeros((Nk, Nk))
                for m1 in range(Nk):
                    mask1 = (kbin_left[m1] <= kmags) & (kmags < kbin_right[m1])
                    if np.any(mask1):
                        wf_temp = np.sum(cyl_wf[it, ...]*mask1[:, :, None, None].astype(int), axis=(0, 1))/np.sum(mask1)
                        if np.sum(wf_temp) > 0.: 
                            for m2 in range(Nk):
                                mask2 = (kbin_left[m2] <= ktot) & (ktot < kbin_right[m2])
                                if np.any(mask2): #cannot compute mean if zero elements
                                    wf_spherical[m1, m2] = np.mean(wf_temp[mask2])
                            # normalisation
                            wf_spherical[m1,:] = np.divide(wf_spherical[m1, :], np.sum(wf_spherical[m1, :]),
                                                           where = np.sum(wf_spherical[m1,:]) != 0)
                spw_window_function.append(wf_spherical)

            window_function_array[spw][..., ip] = np.copy(spw_window_function)

    return window_function_array


def fold_spectra(uvp):
    """
    Average bandpowers from matching positive and negative delay bins onto a
    purely positive delay axis. Negative delay bins are still populated, but
    are filled with zero power. This is an in-place operation.

    Will only work if uvp.folded == False, i.e. data is currently unfolded
    across negative and positive delay. Because this averages the data, the
    nsample array is multiplied by a factor of 2.

    WARNING: This operation cannot be undone.

    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object to be folded.
    """
    # assert folded is False
    assert uvp.folded == False, "cannot fold power spectra if uvp.folded == True"
    store_cov = hasattr(uvp, "cov_array_real")
    # Iterate over spw
    for spw in range(uvp.Nspws):

        # get number of dly bins
        Ndlys = len(uvp.get_dlys(spw))

        # This section could be streamlined considerably since there is a lot of
        # code overlap between the even and odd Ndlys cases.

        if Ndlys % 2 == 0:
            # even number of dlys
            left = uvp.data_array[spw][:, 1:Ndlys//2, :][:, ::-1, :]
            right = uvp.data_array[spw][:, Ndlys//2+1:, :]
            uvp.data_array[spw][:, Ndlys//2+1:, :] = np.mean([left, right], axis=0)
            uvp.data_array[spw][:, :Ndlys//2, :] = 0.0
            uvp.nsample_array[spw] *= 2.0
            if hasattr(uvp, 'window_function_array'):
                if uvp.exact_windows:
                    left = uvp.window_function_array[spw][:, 1:Ndlys//2, ...][:, ::-1, ...]
                    right = uvp.window_function_array[spw][:, Ndlys//2+1: , ...]
                    uvp.window_function_array[spw][:, Ndlys//2+1:, ...] = .50*(left+right)
                else:
                    leftleft = uvp.window_function_array[spw][:, 1:Ndlys//2, 1:Ndlys//2, :][:, ::-1, ::-1, :]
                    leftright = uvp.window_function_array[spw][:, 1:Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                    rightleft = uvp.window_function_array[spw][:, Ndlys//2+1: , 1:Ndlys//2, :][:, :, ::-1, :]
                    rightright = uvp.window_function_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                    uvp.window_function_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                                     +leftright\
                                                                                     +rightleft\
                                                                                     +rightright)
                    uvp.window_function_array[spw][:, :, :Ndlys//2, :] = 0.0
                uvp.window_function_array[spw][:, :Ndlys//2, ...] = 0.0
            # fold covariance array if it exists.
            if hasattr(uvp,'cov_array_real'):
                leftleft = uvp.cov_array_real[spw][:, 1:Ndlys//2, 1:Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array_real[spw][:, 1:Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array_real[spw][:, Ndlys//2+1: , 1:Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array_real[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array_real[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array_real[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array_real[spw][:, :, :Ndlys//2, : :] = 0.0

                leftleft = uvp.cov_array_imag[spw][:, 1:Ndlys//2, 1:Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array_imag[spw][:, 1:Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array_imag[spw][:, Ndlys//2+1: , 1:Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array_imag[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array_imag[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array_imag[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array_imag[spw][:, :, :Ndlys//2, : :] = 0.0

            # fold stats array if it exists: sum in inverse quadrature
            if hasattr(uvp, 'stats_array'):
                for stat in uvp.stats_array.keys():
                    left = uvp.stats_array[stat][spw][:, 1:Ndlys//2, :][:, ::-1, :]
                    right = uvp.stats_array[stat][spw][:, Ndlys//2+1:, :]
                    uvp.stats_array[stat][spw][:, Ndlys//2+1:, :] = (np.sum([1./left**2.0, 1./right**2.0], axis=0))**(-0.5)
                    uvp.data_array[spw][:, :Ndlys//2, :] = np.nan

        else:
            # odd number of dlys
            left = uvp.data_array[spw][:, :Ndlys//2, :][:, ::-1, :]
            right = uvp.data_array[spw][:, Ndlys//2+1:, :]
            uvp.data_array[spw][:, Ndlys//2+1:, :] = np.mean([left, right], axis=0)
            uvp.data_array[spw][:, :Ndlys//2, :] = 0.0
            uvp.nsample_array[spw] *= 2.0
            if hasattr(uvp, 'window_function_array'):
                if uvp.exact_windows:
                    left = uvp.window_function_array[spw][:, :Ndlys//2, ...][:, ::-1, ...]
                    right = uvp.window_function_array[spw][:, Ndlys//2+1: , ...]
                    uvp.window_function_array[spw][:, Ndlys//2+1:, ...] = .50*(left+right)
                else:
                    leftleft = uvp.window_function_array[spw][:, :Ndlys//2, :Ndlys//2, :][:, ::-1, ::-1, :]
                    leftright = uvp.window_function_array[spw][:, :Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                    rightleft = uvp.window_function_array[spw][:, Ndlys//2+1: , :Ndlys//2, :][:, :, ::-1, :]
                    rightright = uvp.window_function_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                    uvp.window_function_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                                 +leftright\
                                                                                 +rightleft\
                                                                                 +rightright)
                    uvp.window_function_array[spw][:, :, :Ndlys//2, :] = 0.0
                uvp.window_function_array[spw][:, :Ndlys//2, ...] = 0.0

            # fold covariance array if it exists.
            if hasattr(uvp,'cov_array_real'):
                leftleft = uvp.cov_array_real[spw][:, :Ndlys//2, :Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array_real[spw][:, :Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array_real[spw][:, Ndlys//2+1: , :Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array_real[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array_real[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array_real[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array_real[spw][:, :, :Ndlys//2, : :] = 0.0

                leftleft = uvp.cov_array_imag[spw][:, :Ndlys//2, :Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array_imag[spw][:, :Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array_imag[spw][:, Ndlys//2+1: , :Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array_imag[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array_imag[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array_imag[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array_imag[spw][:, :, :Ndlys//2, : :] = 0.0

            # fold stats array if it exists: sum in inverse quadrature
            if hasattr(uvp, 'stats_array'):
                for stat in uvp.stats_array.keys():
                    left = uvp.stats_array[stat][spw][:, :Ndlys//2, :][:, ::-1, :]
                    right = uvp.stats_array[stat][spw][:, Ndlys//2+1:, :]
                    uvp.stats_array[stat][spw][:, Ndlys//2+1:, :] = (np.sum([1./left**2.0, 1./right**2.0], axis=0))**(-0.5)
                    uvp.data_array[spw][:, :Ndlys//2, :] = np.nan

    uvp.folded = True


def bootstrap_average_blpairs(uvp_list, blpair_groups, time_avg=False,
                              seed=None):
    """
    Generate a bootstrap-sampled average over a set of power spectra. The
    sampling is done over user-defined groups of baseline pairs (with
    replacement).

    Multiple UVPSpec objects can be passed to this function. The bootstrap
    sampling is carried out over all of the available baseline-pairs (in a
    user-specified group) across all UVPSpec objects. (This means that each
    UVPSpec object can contribute a different number of baseline-pairs to the
    average each time.)

    Successive calls to the function will produce different random samples.

    Parameters
    ----------
    uvp_list : list of UVPSpec objects
        List of UVPSpec objects to form a bootstrap sample from.

        The UVPSpec objects can have different numbers of baseline-pairs and
        times, but the averages will only be taken over spectral windows and
        polarizations that match across all UVPSpec objects.

    blpair_groups : list of baseline-pair groups
        List of baseline-pair groups, where each group is a list of tuples or
        integers. The bootstrap sampling and averaging is done over the
        baseline-pairs within each group.

        There is no requirement for each UVPSpec object to contain all
        baseline-pairs within each specified blpair group, as long as they
        exist in at least one of the UVPSpec objects.

    time_avg : bool, optional
        Whether to average over the time axis or not. Default: False.

    seed : int, optional
        Random seed to use when drawing baseline-pairs.

    Returns
    -------
    uvp_avg_list : list of UVPSpec
        List of UVPSpec objects containing averaged power spectra.

        Each object is an averaged version of the corresponding input UVPSpec
        object. Each average is over the baseline-pairs sampled from that
        object only; to return the true bootstrap average, these objects
        should be averaged together.

    blpair_wgts_list : list of lists of integers
        List of weights used for each baseline-pair in each group, from each
        UVPSpec object. This describes how the bootstrap sample was generated.
        Shape: (len(uvp_list), shape(blpair_groups)).
    """
    # Validate input data
    from hera_pspec import UVPSpec
    single_uvp = False
    if isinstance(uvp_list, UVPSpec):
        single_uvp = True
        uvp_list = [uvp_list,]
    assert isinstance(uvp_list, list), \
           "uvp_list must be a list of UVPSpec objects"

    # Check that uvp_list contains UVPSpec objects with the correct dimensions
    assert np.all([isinstance(uvp, UVPSpec) for uvp in uvp_list]), \
           "uvp_list must be a list of UVPSpec objects"

    # Check for same length of time axis if no time averaging will be done
    if not time_avg:
        if np.unique([uvp.Ntimes for uvp in uvp_list]).size > 1:
            raise IndexError("Input UVPSpec objects must have the same number "
                             "of time samples if time_avg is False.")

    # Check that blpair_groups is a list of lists of groups
    assert isinstance(blpair_groups, list), \
        "blpair_groups must be a list of lists of baseline-pair tuples/integers"
    for grp in blpair_groups:
        assert isinstance(grp, list), \
        "blpair_groups must be a list of lists of baseline-pair tuples/integers"

    # Convert blpair tuples into blpair integers if necessary
    if isinstance(blpair_groups[0][0], tuple):
        new_blp_grps = [[uvputils._antnums_to_blpair(blp) for blp in blpg]
                        for blpg in blpair_groups]
        blpair_groups = new_blp_grps

    # Homogenise input UVPSpec objects in terms of available polarizations
    # and spectral windows
    if len(uvp_list) > 1:
        uvp_list = uvpspec_utils.select_common(uvp_list, spws=True,
                                               pols=True, inplace=False)

    # Loop over UVPSpec objects, looking for available blpairs in each
    avail_blpairs = [_ordered_unique(uvp.blpair_array) for uvp in uvp_list]
    all_avail_blpairs = _ordered_unique([blp for blplist in avail_blpairs
                                       for blp in blplist])

    # Check that all requested blpairs exist in at least one UVPSpec object
    all_req_blpairs = _ordered_unique([blp for grp in blpair_groups for blp in grp])
    missing = []
    for blp in all_req_blpairs:
        if blp not in all_avail_blpairs: missing.append(blp)
    if len(missing) > 0:
        raise KeyError("The following baseline-pairs were specified, but do "
                       "not exist in any of the input UVPSpec objects: %s" \
                       % str(missing))

    # Set random seed if specified
    if seed is not None: np.random.seed(seed)

    # Sample with replacement from the full list of available baseline-pairs
    # in each group, and create a blpair_group and blpair_weights entry for
    # each UVPSpec object
    blpair_grps_list = [[] for uvp in uvp_list]
    blpair_wgts_list = [[] for uvp in uvp_list]

    # Loop over each requested blpair group
    for grp in blpair_groups:

        # Find which blpairs in this group are available in each UVPSpec object
        avail = []
        for blp_list in avail_blpairs:
            # np.intersect1d inherently sorts the output array, which can mess up blpair ordering
            indices = np.intersect1d(grp, blp_list, return_indices=True)[1]
            # this keeps inherent ordering of input blpair_groups
            avail.append(np.array(grp)[np.unique(indices)])

        avail_flat = [blp for lst in avail for blp in lst]
        num_avail = len(avail_flat)

        # Draw set of random integers (with replacement) and convert into
        # list of weights for each blpair in each UVPSpec
        draw = np.random.randint(low=0, high=num_avail, size=num_avail)
        wgt = np.array([(draw == i).sum() for i in range(num_avail)])

        # Extract the blpair weights for each UVPSpec
        j = 0
        for i in range(len(uvp_list)):
            n_blps = len(avail[i])
            blpair_grps_list[i].append( list(avail[i]) )
            _wgts = wgt[np.arange(j, j+n_blps)].astype(float) #+ 1e-4
            blpair_wgts_list[i].append( list(_wgts) )
            j += n_blps

    # Loop over UVPSpec objects and calculate averages in each blpair group,
    # using the bootstrap-sampled blpair weights
    uvp_avg = []
    for i, uvp in enumerate(uvp_list):
        _uvp = average_spectra(uvp, blpair_groups=blpair_grps_list[i],
                               blpair_weights=blpair_wgts_list[i],
                               time_avg=time_avg, inplace=False)
        uvp_avg.append(_uvp)

    # Return list of averaged spectra for now
    if single_uvp:
        return uvp_avg[0], blpair_wgts_list[0]
    else:
        return uvp_avg, blpair_wgts_list


def bootstrap_resampled_error(uvp, blpair_groups=None, time_avg=False, Nsamples=1000, seed=None,
                              normal_std=True, robust_std=False, cintervals=None, bl_error_tol=1.0,
                              add_to_history='', verbose=False):
    """
    Given a UVPSpec object, generate bootstrap resamples of its average
    and calculate their spread as an estimate of the errorbar on the
    uniformly averaged data.

    Parameters:
    -----------
    uvp : UVPSpec object
        A UVPSpec object from which to bootstrap resample & average.

    blpair_groups : list
        A list of baseline-pair groups to bootstrap resample over. Default
        behavior is to calculate and use redundant baseline groups.

    time_avg : boolean
        If True, average spectra before bootstrap resampling

    Nsamples : int
        Number of times to perform bootstrap resample in estimating errorbar.

    seed : int
        Random seed to use in bootstrap resampling.

    normal_std : bool
        If True, calculate an error estimate from numpy.std and store as "bs_std"
        in the stats_array of the output UVPSpec object.

    robust_std : bool
        If True, calculate an error estimate from astropy.stats.biweight_midvariance
        and store as "bs_robust_std" in the stats_array of the output UVPSpec object.

    cintervals : list
        A list of confidence interval percentages (0 < cinterval < 100) to calculate
        using numpy.percentile and store in the stats_array of the output UVPSpec
        object as "bs_cinterval_{:05.2f}".format(cinterval).

    bl_error_tol : float
        Redundancy error tolerance of redundant groups if blpair_groups is None.

    add_to_history : str
        String to add to history of output uvp_avg object.

    verbose : bool
        If True, report feedback to stdout.

    Returns:
    --------
    uvp_avg : UVPSpec object
        A uvp holding the uniformaly averaged data in data_array, and the various error
        estimates calculated from the bootstrap resampling.
    """
    from hera_pspec import UVPSpec
    # type check
    assert isinstance(uvp, (UVPSpec, str)), "uvp must be fed as a UVPSpec object or filepath"
    if isinstance(uvp, str):
        _uvp = UVPSpec()
        _uvp.read_hdf5(uvp)
        uvp = _uvp

    # Check for blpair_groups
    if blpair_groups is None:
        blpair_groups, _, _, _ = utils.get_blvec_reds(uvp, bl_error_tol=bl_error_tol)

    # Uniform average
    uvp_avg = average_spectra(uvp, blpair_groups=blpair_groups, time_avg=time_avg,
                              inplace=False)

    # initialize a seed
    if seed is not None: np.random.seed(seed)

    # Iterate over Nsamples and create bootstrap resamples
    uvp_boots = []
    uvp_wgts = []
    for i in range(Nsamples):
        # resample
        boot, wgt = bootstrap_average_blpairs(uvp, blpair_groups=blpair_groups, time_avg=time_avg, seed=None)
        uvp_boots.append(boot)
        uvp_wgts.append(wgt)

    # get all keys in uvp_avg and get data from each uvp_boot
    keys = uvp_avg.get_all_keys()
    uvp_boot_data = odict([(k, np.array([u.get_data(k) for u in uvp_boots]))
                           for k in keys])

    # calculate various error estimates
    if normal_std:
        for k in keys:
            nstd = np.std(uvp_boot_data[k].real, axis=0) \
                 + 1j*np.std(uvp_boot_data[k].imag, axis=0)
            uvp_avg.set_stats("bs_std", k, nstd)

    if robust_std:
        for k in keys:
            rstd = np.sqrt(astats.biweight_midvariance(uvp_boot_data[k].real, axis=0)) \
                    + 1j*np.sqrt(astats.biweight_midvariance(uvp_boot_data[k].imag, axis=0))
            uvp_avg.set_stats("bs_robust_std", k, rstd)

    if cintervals is not None:
        for ci in cintervals:
            ci_tag = "bs_cinterval_{:05.2f}".format(ci)
            for k in keys:
                cint = np.percentile(uvp_boot_data[k].real, ci, axis=0) \
                        + 1j*np.percentile(uvp_boot_data[k].imag, ci, axis=0)
                uvp_avg.set_stats(ci_tag, k, cint)

    # Update history
    uvp_avg.history = "Bootstrap errors estimated w/ hera_pspec [{}], {} samples, {} seed\n{}\n{}\n{}" \
                      "".format(__version__, Nsamples, seed, add_to_history, '-'*40, uvp_avg.history)

    return uvp_avg, uvp_boots, uvp_wgts


def bootstrap_run(filename, spectra=None, blpair_groups=None, time_avg=False, Nsamples=1000, seed=0,
                  normal_std=True, robust_std=True, cintervals=None, keep_samples=False,
                  bl_error_tol=1.0, overwrite=False, add_to_history='', verbose=True, maxiter=1):
    """
    Run bootstrap resampling on a PSpecContainer object to estimate errorbars.
    For each group/spectrum specified in the PSpecContainer, this function produces
        1. uniform average of UVPSpec objects in the group
        2. various error estimates from the bootstrap resamples
       (3.) series of bootstrap resamples of UVPSpec average (optional)

    The output of 1. and 2. are placed in a *_avg spectrum, while the output of 3.
    is placed in *_bs0, *_bs1, *_bs2 etc. objects.

    Note: PSpecContainers should not be opened in SWMR mode for this function.

    Parameters:
    -----------
    filename : str or PSpecContainer object
        PSpecContainer object to run bootstrapping on.

    spectra : list
        A list of power spectra names (with group prefix) to run bootstrapping on.
        Default is all spectra in object. Ex. ["group1/psname1", "group1/psname2", ...]

    blpair_groups : list
        A list of baseline-pair groups to bootstrap over. Default is to solve for and use
        redundant baseline groups. Ex: [ [((1, 2), (2, 3)), ((1, 2), (3, 4))],
                                         [((1, 3), (2, 4)), ((1, 3), (3, 5))],
                                         ...
                                        ]

    time_avg : bool
        If True, perform time-average of power spectra in averaging step.

    Nsamples : int
        The number of samples in bootstrap resampling to generate.

    seed : int
        The random seed to initialize with before drwaing bootstrap samples.

    normal_std : bool
        If True, calculate an error estimate from numpy.std and store as "bs_std"
        in the stats_array of the output UVPSpec object.

    robust_std : bool
        If True, calculate an error estimate from astropy.stats.biweight_midvariance
        and store as "bs_robust_std" in the stats_array of the output UVPSpec object.

    cintervals : list
        A list of confidence interval percentages (0 < cinterval < 100) to calculate
        using numpy.percentile and store in the stats_array of the output UVPSpec
        object as "bs_cinterval_{:05.2f}".format(cinterval).

    keep_samples : bool
        If True, store each bootstrap resample in PSpecContainer object with *_bs# suffix.

    bl_error_tol : float
        If calculating redundant baseline groups, this is the redundancy tolerance in meters.

    overwrite : bool
        If True, overwrite output files if they already exist.

    add_to_history : str
        String to append to history in bootstrap_resample_error() call.

    verbose : bool
        If True, report feedback to stdout.

    maxiter : int, optional, default=1
        Maximum number of attempts to open the PSpecContainer by a single process.
        0.5 sec wait per attempt. Useful in the case of multiprocesses bootstrapping
        different groups of the same container.
    """
    from hera_pspec import uvpspec
    from hera_pspec import PSpecContainer
    # type check
    if isinstance(filename, str):
        # open in transactional mode
        psc = PSpecContainer(filename, mode='rw', keep_open=False, swmr=False, tsleep=0.5, maxiter=maxiter)
    elif isinstance(filename, PSpecContainer):
        psc = filename
        assert not psc.swmr, "PSpecContainer should not be in SWMR mode"
    else:
        raise AssertionError("filename must be a PSpecContainer or filepath to one")

    # get groups in psc
    groups = psc.groups()
    assert len(groups) > 0, "No groups exist in PSpecContainer"

    # get spectra if not fed
    all_spectra = utils.flatten([ [os.path.join(grp, s)
                                   for s in psc.spectra(grp)]
                                   for grp in groups])
    if spectra is None:
        spectra = all_spectra
    else:
        spectra = [spc for spc in spectra if spc in all_spectra]
        assert len(spectra) > 0, "no specified spectra exist in PSpecContainer"

    # iterate over spectra
    for spc_name in spectra:
        # split group and spectra names
        grp, spc = spc_name.split('/')

        # run boostrap_resampled_error
        uvp = psc.get_pspec(grp, spc)
        (uvp_avg, uvp_boots,
         uvp_wgts) = bootstrap_resampled_error(uvp, blpair_groups=blpair_groups,
                                               time_avg=time_avg,
                                               Nsamples=Nsamples, seed=seed,
                                               normal_std=normal_std,
                                               robust_std=robust_std,
                                               cintervals=cintervals,
                                               bl_error_tol=bl_error_tol,
                                               add_to_history=add_to_history,
                                               verbose=verbose)

        # set averaged uvp
        psc.set_pspec(grp, spc+"_avg", uvp_avg, overwrite=overwrite)

        # if keep_samples write uvp_boots
        if keep_samples:
            for i, uvpb in enumerate(uvp_boots):
                psc.set_pspec(grp, spc+"_bs{}".format(i), uvpb, overwrite=overwrite)


def get_bootstrap_run_argparser():
    a = argparse.ArgumentParser(
           description="argument parser for grouping.bootstrap_run()")

    def list_of_lists_of_tuples(s):
        s = [[int(_x) for _x in x.split()] for x in s.split(',')]
        return s

    # Add list of arguments
    a.add_argument("filename", type=str,
                   help="Filename of HDF5 container (PSpecContainer) containing "
                        "input power spectra.")
    a.add_argument("--spectra", default=None, type=str, nargs='+',
                   help="List of power spectra names (with group prefix) to bootstrap over.")
    a.add_argument("--blpair_groups", default=None, type=list_of_lists_of_tuples,
                   help="List of baseline-pair groups (must be space-delimited blpair integers) "
                        "wrapped in quotes to use in resampling. Default is to solve for and use redundant groups (recommended)."
                        "Ex: --blpair_groups '101102103104 102103014015, 101013102104' --> "
                        "[ [((1, 2), (3, 4)), ((2, 3), (4, 5))], [((1, 3), (2, 4))], ...]")
    a.add_argument("--time_avg", default=False, type=bool, help="Perform time-average in averaging step.")
    a.add_argument("--Nsamples", default=100, type=int, help="Number of bootstrap resamples to generate.")
    a.add_argument("--seed", default=0, type=int, help="random seed to initialize bootstrap resampling with.")
    a.add_argument("--normal_std", default=True, type=bool,
                    help="Whether to calculate a 'normal' standard deviation (np.std).")
    a.add_argument("--robust_std", default=False, type=bool,
                    help="Whether to calculate a 'robust' standard deviation (astropy.stats.biweight_midvariance).")
    a.add_argument("--cintervals", default=None, type=float, nargs='+',
                    help="Confidence intervals (precentage from 0 < ci < 100) to calculate.")
    a.add_argument("--keep_samples", default=False, action='store_true',
                    help="If True, store bootstrap resamples in PSpecContainer object with *_bs# extension.")
    a.add_argument("--bl_error_tol", type=float, default=1.0,
                    help="Baseline redudancy tolerance if calculating redundant groups.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite outputs if they exist.")
    a.add_argument("--add_to_history", default='', type=str, help="String to add to history of power spectra.")
    a.add_argument("--verbose", default=False, action='store_true', help="report feedback to stdout.")

    return a

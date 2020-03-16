import numpy as np
from collections import OrderedDict as odict
import random
import copy
import argparse
from astropy import stats as astats
import os

from . import utils, version, uvpspec_utils as uvputils
from .uvpspec import _ordered_unique

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
                    error_weights=None,
                    normalize_weights=True, inplace=True,
                    add_to_history=''):
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
         then it also gets assigned as error_field. And if one specified both,
         then error_weights supercedes as error_field. 
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
    else:
        # If not, each baseline pair is its own group
        blpair_groups = [[blp] for blp in _ordered_unique(uvp.blpair_array)]
        assert blpair_weights is None, "Cannot specify blpair_weights if "\
                                       "blpair_groups is None."
        blpair_weights = [[1.,] for blp in _ordered_unique(uvp.blpair_array)]

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
    elif isinstance(error_field, (str, np.str)):
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

    # For baseline pairs not in blpair_groups, add them as their own group
    extra_blpairs = set(uvp.blpair_array) - set(all_blpairs)
    blpair_groups += [[blp] for blp in extra_blpairs]
    blpair_weights += [[1.,] for blp in extra_blpairs]

    # Create new data arrays
    data_array, wgts_array = odict(), odict()
    ints_array, nsmp_array = odict(), odict()
    stats_array = odict([[stat, odict()] for stat in stat_l])

    # will average covariance array if present
    store_cov = hasattr(uvp, "cov_array")
    if store_cov:
        cov_array = odict()

    # Iterate over spectral windows
    for spw in range(uvp.Nspws):
        spw_data, spw_wgts, spw_ints, spw_nsmp = [], [], [], []
        spw_stats = odict([[stat, []] for stat in stat_l])
        if store_cov:
            spw_cov = []
        
        # Iterate over polarizations
        for i, p in enumerate(uvp.polpair_array):
            pol_data, pol_wgts, pol_ints, pol_nsmp = [], [], [], []
            pol_stats = odict([[stat, []] for stat in stat_l])
            if store_cov:
                pol_cov = []

            # Iterate over baseline-pair groups
            for j, blpg in enumerate(blpair_groups):
                bpg_data, bpg_wgts, bpg_ints, bpg_nsmp = [], [], [], []
                bpg_stats = odict([[stat, []] for stat in stat_l])
                if store_cov:
                    bpg_cov = []
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
                    blpg_wgts *= float(blpg_wgts.size) / norm # Apply normalization
                else:
                    blpg_wgts = np.ones(len(blpg))

                # Iterate within a baseline-pair group and get integration-
                # weighted data
                for k, blp in enumerate(blpg):
                    # Get no. samples and construct integration weight
                    nsmp = uvp.get_nsamples((spw, blp, p))[:, None]
                    # shape of nsmp: (Ntimes, 1)
                    data = uvp.get_data((spw, blp, p))
                    # shape of data: (Ntimes, Ndlys)
                    wgts = uvp.get_wgts((spw, blp, p))
                    # shape of wgts: (Ntimes, Ndlys, 2)
                    ints = uvp.get_integrations((spw, blp, p))[:, None]
                    # shape of ints: (Ntimes, 1)
                    if store_cov:
                        cov = uvp.get_cov((spw, blp, p))
                        # shape of cov: (Ntimes, Ndlys, Ndlys)
                    # Get error bar
                    errws = {}
                    for stat in stat_l:
                        errws[stat] = (uvp.get_stats(stat, (spw, blp, p)))**2
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
                        w = 1./(uvp.get_stats(error_weights, (spw, blp, p)))**2
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
                        data = (np.sum(data * w, axis=0) \
                            / np.sum(w, axis=0).clip(1e-40, np.inf))[None]
                        wgts = (np.sum(wgts * w[:, :, None], axis=0) \
                            / np.sum(w, axis=0).clip(1e-40, np.inf)[:, None])[None]
                        ints = (np.sum(ints * w, axis=0) \
                            / np.sum(w, axis=0).clip(1e-40, np.inf))[None]
                        nsmp = np.sum(nsmp, axis=0)[None]
                        if store_cov:
                            cov = (np.sum(cov * w[:, :, None]**2, axis=0) \
                                / (np.sum(w,axis=0).clip(1e-40, np.inf))[:, None]**2)[None]
                        for stat in stat_l:
                            errws[stat] = (np.sum(errws[stat]*w**2, axis=0) \
                            / (np.sum(w, axis=0).clip(1e-40, np.inf))**2)[None]
                        w = np.sum(w, axis=0)[None]
                        # Here we use clip method for zero weights. A tolerance 
                        # as low as 1e-40 works when using inverse square of noise power 
                        # as weights.  
                    # Add multiple copies of data for each baseline according
                    # to the weighting/multiplicity;
                    # while multiple copies are only added when bootstrap resampling
                    for m in range(int(blpg_wgts[k])):
                        bpg_data.append(data * w)
                        bpg_wgts.append(wgts * w[:, :, None])
                        bpg_ints.append(ints * w)
                        bpg_nsmp.append(nsmp)
                        for stat in stat_l:
                            bpg_stats[stat].append(errws[stat]*w**2)
                        w_list.append(w)
                        if store_cov:
                            bpg_cov.append(cov * w[:, :, None]**2)

                # Average over baseline-pairs
                # Take integration-weighted averages, with clipping to deal
                # with zeros
                # Or take error_weighted averages
                bpg_data = np.sum(bpg_data, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-40, np.inf)
                bpg_wgts = np.sum(bpg_wgts, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-40, np.inf)[:,:, None]
                bpg_nsmp = np.sum(bpg_nsmp, axis=0)
                bpg_ints = np.sum(bpg_ints, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-40, np.inf)
                if store_cov:
                    bpg_cov = np.sum(bpg_cov, axis=0) \
                            / (np.sum(w_list, axis=0).clip(1e-40, np.inf)[:,:,None]**2)
                for stat in stat_l:
                    arr = np.sum(bpg_stats[stat], axis=0) \
                        / (np.sum(w_list, axis=0).clip(1e-40, np.inf)**2)
                    bpg_stats[stat] = np.sqrt(arr)

                # Append to lists (polarization)
                pol_data.extend(bpg_data); pol_wgts.extend(bpg_wgts)
                pol_ints.extend(bpg_ints); pol_nsmp.extend(bpg_nsmp)
                [pol_stats[stat].extend(bpg_stats[stat]) for stat in stat_l]
                if store_cov:
                    pol_cov.extend(bpg_cov)

            # Append to lists (spectral window)
            spw_data.append(pol_data); spw_wgts.append(pol_wgts)
            spw_ints.append(pol_ints); spw_nsmp.append(pol_nsmp)
            [spw_stats[stat].append(pol_stats[stat]) for stat in stat_l]
            if store_cov:
                spw_cov.append(pol_cov)

        # Append to dictionaries
        data_array[spw] = np.moveaxis(spw_data, 0, -1)
        wgts_array[spw] = np.moveaxis(spw_wgts, 0, -1)
        ints_array[spw] = np.moveaxis(spw_ints, 0, -1)[:, 0, :]
        nsmp_array[spw] = np.moveaxis(spw_nsmp, 0, -1)[:, 0, :]
        for stat in stat_l:
            stats_array[stat][spw] = np.moveaxis(spw_stats[stat], 0, -1)
        if store_cov:
            cov_array[spw] = np.moveaxis(np.array(spw_cov), 0, -1)

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
            blpair_arr.extend(np.ones_like(blpairts, np.int) * blpg[0])
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
    if store_cov:
        uvp.cov_array = cov_array
    if len(stat_l) >=1 :
        uvp.stats_array = stats_array
    elif hasattr(uvp, "stats_array"):
        delattr(uvp, "stats_array")

    # Add to history
    uvp.history = "Spectra averaged with hera_pspec [{}]\n{}\n{}\n{}".format(version.git_hash[:15], add_to_history, '-'*40, uvp.history)

    # Validity check
    uvp.check()

    # Return
    if inplace == False:
        return uvp


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

            # fold covariance array if it exists.
            if hasattr(uvp,'cov_array'):
                leftleft = uvp.cov_array[spw][:, 1:Ndlys//2, 1:Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array[spw][:, 1:Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array[spw][:, Ndlys//2+1: , 1:Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array[spw][:, :, :Ndlys//2, : :] = 0.0

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

            # fold covariance array if it exists.
            if hasattr(uvp,'cov_array'):
                leftleft = uvp.cov_array[spw][:, :Ndlys//2, :Ndlys//2, :][:, ::-1, ::-1, :]
                leftright = uvp.cov_array[spw][:, :Ndlys//2, Ndlys//2+1:, :][:, ::-1, :, :]
                rightleft = uvp.cov_array[spw][:, Ndlys//2+1: , :Ndlys//2, :][:, :, ::-1, :]
                rightright = uvp.cov_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :]
                uvp.cov_array[spw][:, Ndlys//2+1:, Ndlys//2+1:, :] = .25*(leftleft\
                                                                         +leftright\
                                                                         +rightleft\
                                                                         +rightright)
                uvp.cov_array[spw][:, :Ndlys//2, :, :] = 0.0
                uvp.cov_array[spw][:, :, :Ndlys//2, : :] = 0.0

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
    assert isinstance(uvp, (UVPSpec, str, np.str)), "uvp must be fed as a UVPSpec object or filepath"
    if isinstance(uvp, (str, np.str)):
        _uvp = UVPSpec()
        _uvp.read_hdf5(uvp)
        uvp = _uvp

    # Check for blpair_groups
    if blpair_groups is None:
        blpair_groups, _, _, _ = utils.get_blvec_reds(uvp, bl_error_tol=bl_error_tol)

    # Uniform average
    uvp_avg = average_spectra(uvp, blpair_groups=blpair_groups, time_avg=time_avg, inplace=False)

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
                      "".format(version.git_hash[:15], Nsamples, seed, add_to_history, '-'*40, uvp_avg.history)

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
    if isinstance(filename, (str, np.str)):
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

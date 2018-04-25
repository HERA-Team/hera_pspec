import numpy as np
from collections import OrderedDict as odict
from hera_pspec import uvpspec_utils as uvputils
import random
import copy

def fold_kparallel(pspec_list):
    """
    Take a power spectrum with bins in positive and negative k, and combine 
    into a power spectrum for absolute |k| only.
    
    Parameters
    ----------
    pspec_list : list of PSpec objects
        List of power spectra, one for each bootstrap. Dimensions: 
        (N_lsts, N_kbins, N_boots).
    
    Returns
    -------
    k_folded : array_like
        Folded k values, |k|.
        
    pspec_folded : array_like
        Folded power spectrum, with only |k_parallel|.
    """
    NotImplementedError()
    # Takes kbins from properties of PSpec.kbins.


def collapse_over_lsts(pspec_list):
    """
    Average a set of bootstrap-sampled power spectra over LSTs.
    
    Parameters
    ----------
    pspec_list : list of PSpec objects
        List of power spectra, one for each bootstrap. Dimensions: 
        (N_lsts, N_kbins, N_boots).
    
    Returns
    -------
    pspec_list : array_like
        List of power spectra averaged over LST, one for each bootstrap. 
        Dimensions: (N_kbins, N_boots).
    """
    # 1. Take a set of power spectra for many bootstrap samples
    # 2. Average over time for each bootstrap sample
    NotImplementedError()


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
    n = Nbls / Ngroups # Baselines per group
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


def average_spectra(uvp_in, blpair_groups=None, time_avg=False, inplace=True):
    """
    Average power spectra across the baseline-pair-time axis, weighted by 
    each spectrum's integration time.
    
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
    groups, see uvp.get_blpair_groups_from_bl_groups.

    Parameters
    ----------
    uvp_in : UVPSpec
        Input UVPSpec file (containing power spectra for a given pair of 
        datasets).
        
    blpair_groups : list of baseline-pair groups
        (i.e. list of lists of tuples or integers)
        All power spectra in a baseline-pair group are averaged together. 
        If a baseline-pair exists in more than one group, a warning is 
        raised.
        Ex: [ [((1, 2), (1, 2)), ((2, 3), (2, 3))], [((4, 6), (4, 6))] ] or
            [ [1002001002, 2003002003], [4006004006] ]

    time_avg : bool, optional
        If True, average power spectra across the time axis. Default: False.

    inplace : bool, optional
        If True, edit data in input UVPSpec object; otherwise, make a copy and 
        return a new UVPSpec object. Default: True.

    Notes
    -----
    Currently, every baseline-pair in a blpair group must have the same 
    Ntimes. Future versions may support baseline-pair averaging of 
    heterogeneous time arrays.
    """
    if inplace:
        uvp = uvp_in
    else:
        uvp = copy.deepcopy(uvp_in)

    # If blpair_groups were fed in, enforce type and structure
    if blpair_groups is not None:
        
        # Enforce shape of blpair_groups
        assert isinstance(blpair_groups[0], list), \
              "blpair_groups must be fed as a list of baseline-pair lists. " \
              "See docstring."

        # Convert blpair_groups to list of blpair group integers
        if isinstance(blpair_groups[0][0], tuple):
            new_blpair_grps = [map(lambda blp: uvp.antnums_to_blpair(blp), blpg) 
                               for blpg in blpair_groups]
            blpair_groups = new_blpair_groups
    else:
        # If not, each baseline pair is its own group
        blpair_groups = map(lambda blp: [blp], np.unique(uvp.blpair_array))

    # Print warning if a blpair appears more than once in all of blpair_groups
    all_blpairs = [item for sublist in blpair_groups for item in sublist]
    if len(set(all_blpairs)) < len(all_blpairs): 
        print "Warning: some baseline-pairs are repeated between blpair "\
              "averaging groups..."

    # For baseline pairs not in blpair_groups, add them as their own group
    extra_blpairs = set(uvp.blpair_array) - set(all_blpairs)
    blpair_groups += map(lambda blp: [blp], extra_blpairs)

    # Create new data arrays
    data_array, wgts_array = odict(), odict()
    ints_array, nsmp_array = odict(), odict()

    # Iterate over spectral windows
    for spw in range(uvp.Nspws):
        spw_data, spw_wgts, spw_ints, spw_nsmp = [], [], [], []

        # Iterate over polarizations
        for i, p in enumerate(uvp.pol_array):
            pol_data, pol_wgts, pol_ints, pol_nsmp = [], [], [], []

            # Iterate over baseline-pair groups
            for j, blpg in enumerate(blpair_groups):
                bpg_data, bpg_wgts, bpg_ints, bpg_nsmp = [], [], [], []
                w_list = []

                # Iterate within a baseline-pair group and get integration-
                # weighted data
                for k, blp in enumerate(blpg):
                    
                    # Get no. samples and construct integration weight
                    nsmp = uvp.get_nsamples(spw, blp, p)[:, None]
                    ints = uvp.get_integrations(spw, blp, p)[:, None]
                    w = (ints * np.sqrt(nsmp))
                    
                    # Apply integration weight to data
                    bpg_data.append(uvp.get_data(spw, blp, p) * w)
                    bpg_wgts.append(uvp.get_wgts(spw, blp, p) * w[:, None])
                    bpg_ints.append(ints * w)
                    bpg_nsmp.append(nsmp)
                    w_list.append(w)

                # Take integration-weighted averages, with clipping to deal 
                # with zeros
                bpg_data = np.sum(bpg_data, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-10, np.inf)
                bpg_wgts = np.sum(bpg_wgts, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-10, np.inf)[:, None]
                bpg_nsmp = np.sum(bpg_nsmp, axis=0)
                bpg_ints = np.sum(bpg_ints, axis=0) \
                         / np.sum(w_list, axis=0).clip(1e-10, np.inf)
                w_list = np.sum(w_list, axis=0)

                # Take time average if desired
                if time_avg:
                    bpg_data = [np.sum(bpg_data * w_list, axis=0) \
                             / np.sum(w_list, axis=0).clip(1e-10, np.inf)]
                    bpg_wgts = [np.sum(bpg_wgts * w_list[:, None], axis=0) \
                             / np.sum(w_list, axis=0).clip(1e-10, np.inf)[:, None]]
                    bpg_nsmp = [np.sum(bpg_nsmp, axis=0)]
                    bpg_ints = [np.sum(bpg_ints * w_list, axis=0) \
                             / np.sum(w_list, axis=0).clip(1e-10, np.inf)]

                # Append to lists
                pol_data.extend(bpg_data); pol_wgts.extend(bpg_wgts)
                pol_ints.extend(bpg_ints); pol_nsmp.extend(bpg_nsmp)

            # Append to lists
            spw_data.append(pol_data); spw_wgts.append(pol_wgts)
            spw_ints.append(pol_ints); spw_nsmp.append(pol_nsmp)

        # Append to dictionaries
        data_array[spw] = np.moveaxis(spw_data, 0, -1)
        wgts_array[spw] = np.moveaxis(spw_wgts, 0, -1)
        ints_array[spw] = np.moveaxis(spw_ints, 0, -1)[:, 0, :]
        nsmp_array[spw] = np.moveaxis(spw_nsmp, 0, -1)[:, 0, :]

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
    bl_vecs = np.array(map(lambda bl: uvp.bl_vecs[uvp.bl_array.tolist().index(bl)], bl_arr))

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

        if Ndlys % 2 == 0:
            # even number of dlys
            left = uvp.data_array[spw][:, 1:Ndlys//2, :][:, ::-1, :]
            right = uvp.data_array[spw][:, Ndlys//2+1:, :]
            uvp.data_array[spw][:, Ndlys//2+1:, :] = np.mean([left, right], axis=0)
            uvp.data_array[spw][:, :Ndlys//2, :] = 0.0
            uvp.nsample_array[spw] *= 2.0

        else:
            # odd number of dlys
            left = uvp.data_array[spw][:, :Ndlys//2, :][:, ::-1, :]
            right = uvp.data_array[spw][:, Ndlys//2+1:, :]   
            uvp.data_array[spw][:, Ndlys//2+1:, :] = np.mean([left, right], axis=0)
            uvp.data_array[spw][:, :Ndlys//2, :] = 0.0
            uvp.nsample_array[spw] *= 2.0

    uvp.folded = True



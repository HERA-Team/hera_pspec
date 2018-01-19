import random

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


def group_baselines(bls, Ngroups, keep_remainder=False, randomize=False):
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
        number of specified groups. If True, a group containing the remainder 
        baselines is appended to the output list. Otherwise, the remainder 
        baselines are discarded. Default: False.
    
    randomize : bool, optional
        Whether baselines should be added to groups in the order they appear in 
        'bls', or if they should be assigned at random. Default: False.
    
    Returns
    -------
    grouped_bls : list of lists of tuples
        List of grouped baselines.
    """
    Nbls = len(bls) # Total baselines
    n = Nbls / Ngroups # Baselines per group
    rem = Nbls % n
    
    # Make sure only tuples were provided (can't have groups of groups)
    for bl in bls: assert isinstance(bl, tuple)
    
    grouped_bls = []
    if randomize:
        # Randomly select members of each group
        for i in range(Ngroups): grouped_bls.append( random.sample(bls, n) )
        if keep_remainder and rem > 0: grouped_bls.append(bls)
    else:
        # Assign to groups sequentially
        for i in range(Ngroups): grouped_bls.append(bls[i*n:(i+1)*n])
        if keep_remainder and rem > 0: grouped_bls.append(bls[-rem:])
    return grouped_bls


def sample_baselines(bls):
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
    
    Returns
    -------
    sampled_bls : list of tuples or lists of tuples
        Bootstrap-sampled set of baselines (will include multiple instances of 
        some baselines).
    """
    # Sample with replacement; return as many baselines/groups as were input
    return [random.choice(bls) for i in range(len(bls))]
    

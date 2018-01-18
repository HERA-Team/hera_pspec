

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
    

def sample_baselines(bls):
    """
    Sample a set of baselines with replacement (to be used to generate a 
    bootstrap sample).
    
    Parameters
    ----------
    bls : list of tuples
        Set of unique baselines to be sampled from.
    
    Returns
    -------
    bl_sample : list of tuples
        Bootstrap-sampled set of baselines (will include multiple instances of 
        some baselines).
    """
    NotImplementedError()

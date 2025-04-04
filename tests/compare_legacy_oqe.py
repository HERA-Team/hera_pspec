#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def run_old_oqe(fname, key1, key2, freqrange):
    """
    Run old OQE algorithm using capo.
    """
    # Legacy functions to load data and estimate power spectrum
    import capo
    legacy_read_files = capo.miriad.read_files
    legacy_group_redundant_bls = capo.red.group_redundant_bls
    legacy_oqe = capo.oqe

    # (1) Read data from file
    s,d,f = legacy_read_files([fname], antstr='all', polstr='xx')
    bls = d.keys()
    #print("Baseline keys:", bls)

    # (1a) Legacy setting to specify whether data are conjugated or not
    # PB: Replace this with made up conj array (all set to False)
    """
    aa = aipy.cal.get_aa('psa6240_v003', np.array([.15]))
    _, conj = legacy_group_redundant_bls(aa.ant_layout) 
        # conj is a dictionary containing whether bl's are conjugated or not
    """
    conj = {bl:False for bl in bls}

    # (1b) Build data and flagging dictionaries
    data_dict = {}; flg_dict = {}
    fmin, fmax = freqrange
    for key in bls:
        # Use only a restricted band of frequencies (e.g. to avoid RFI)
        data_dict[key] = d[key]['xx'][:,fmin:fmax]
        flg_dict[key] = np.logical_not(f[key]['xx'][:,fmin:fmax])

    # (2) Make dataset object
    ds = legacy_oqe.DataSet()
    ds.set_data(dsets=data_dict, conj=conj, wgts=flg_dict)

    # (3) Calculate unweighted power spectrum
    q_I = ds.q_hat(key1, key2, use_cov=False, cov_flagging=False) # unweighted
    F_I = ds.get_F(key1, key2, use_cov=False, cov_flagging=False)
    M_I, W_I = ds.get_MW(F_I, mode='I')
    p_I = ds.p_hat(M_I, q_I)

    # (4) Calculate inverse covariance-weighted power spectrum
    q = ds.q_hat(key1, key2, use_cov=True, cov_flagging=False) # weighted
    F = ds.get_F(key1, key2, use_cov=True, cov_flagging=False)
    M, W = ds.get_MW(F, mode='I')
    p = ds.p_hat(M, q)
    
    return p_I, p


def run_new_oqe(fname, key1, key2, freqrange):
    """
    Run new OQE algorithm using hera_pspec.
    """
    from pyuvdata import UVData
    import hera_pspec as pspec

    # (1) Read data from file
    d1 = UVData()
    d1.read_miriad(fname)
    
    # (1a) Use only a restricted band of frequencies (e.g. to avoid RFI)
    fmin, fmax = freqrange
    d1.select(freq_chans=np.arange(fmin, fmax))
    
    # (1b) Build data and flagging lists
    d = [d1,]
    w = [None for _d in d] # Set weights (None => flags from UVData will be used)
    #print("Baseline keys:", d[0].get_antpairs())

    # (2) Make PSpecData object
    ds = pspec.PSpecData(dsets=d, wgts=w)

    # (3) Calculate unweighted power spectrum
    ds.set_R('identity')
    q_I = ds.q_hat(key1, key2)
    F_I = ds.get_G(key1, key2)
    M_I, W_I = ds.get_MW(F_I, mode='I')
    p_I = ds.p_hat(M_I, q_I)
    
    # (4) Calculate inverse covariance-weighted power spectrum
    ds.set_R('iC')
    q = ds.q_hat(key1, key2)
    F = ds.get_G(key1, key2)
    M, W = ds.get_MW(F, mode='I')
    p = ds.p_hat(M, q)
    
    #pspec, pairs = ds.pspec(bls, input_data_weight='I', norm='I', verbose=True)
    return p_I, p


if __name__ == '__main__':
    
    # Path to datafile
    fname = '../data/zen.2458042.12552.xx.HH.uvXAA'
    
    # Baselines to use
    key1 = (24, 25)
    key2 = (24, 38)
    
    # Frequency channels to include
    freqrange = (28, 52)
    
    # Run old OQE
    pI_old, p_old = run_old_oqe(fname, key1, key2, freqrange)
    print("Old:", p_old.shape)
    
    # Run new OQE
    _key1 = (0,) + key1
    _key2 = (0,) + key2
    pI_new, p_new = run_new_oqe(fname, _key1, _key2, freqrange)
    print("New:", p_new.shape)
    
    # Calculate fractional difference of means (averaged over LST)
    frac_I = np.mean(pI_new, axis=1).real / np.mean(pI_old, axis=1).real - 1.
    frac_iC = np.mean(p_new, axis=1).real / np.mean(p_old, axis=1).real - 1.
    
    # Plot results (averaging over LST bins)
    plt.subplot(221)
    plt.plot(np.mean(pI_old, axis=1).real, 'k-', lw=1.8, label="capo (I)")
    plt.plot(np.mean(pI_new, axis=1).real, 'r--', lw=1.8, label="hera_pspec (I)")
    plt.legend(loc='lower right')
    plt.ylabel(r"$\hat{p}$", fontsize=18)
    
    plt.subplot(222)
    plt.plot(frac_I, 'k-', lw=1.8)
    plt.ylabel(r"$P_{\rm pspec}/P_{\rm capo} - 1$ $(I)$", fontsize=18)
    
    plt.subplot(223)
    plt.plot(np.mean(p_old, axis=1).real, 'k-', lw=1.8, label="capo (iC)")
    plt.plot(np.mean(p_new, axis=1).real, 'r--', lw=1.8, label="hera_pspec (iC)")
    plt.legend(loc='lower right')
    plt.ylabel(r"$\hat{p}$", fontsize=18)
    
    plt.subplot(224)
    plt.plot(frac_iC, 'k-', lw=1.8)
    plt.ylabel(r"$P_{\rm pspec}/P_{\rm capo} - 1$ $(C^{-1})$", fontsize=18)
    
    plt.tight_layout()
    plt.show()
    

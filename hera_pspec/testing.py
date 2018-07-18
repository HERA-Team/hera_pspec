import numpy as np
import os
from pyuvdata import UVData
import hera_pspec as hp
from hera_pspec.data import DATA_PATH

def noise_sim(weighting='identity', bls=None, cosmo=None, beam=None):
    """
    Run a noise simulation on artificial noise data.

    Parameters
    ----------
    weighting : str, optional
        Type of data weighting in PSpecData.pspec() call.

    bls : list, optional
        List of baselines (antenna-pair tuples) to use in pspec calculation
        Default is all 14.6 EW m bls

    cosmo : conversions.Cosmo_Conversions instance, optional
        Cosmology to adopt, default is model in conversions.Cosmo_Conversions()

    beam : pspecbeam.PSpecBeamUV instance, optional
        Beam model to adopt, default is the beam-model in hera_pspec/data/
    """
    # Read file
    dfile = os.path.join(DATA_PATH, "hera_noise_sim.uv")
    uvd = UVData()
    uvd.read_miriad(dfile)
    
    # Configure data
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    uvd1.vis_units = 'mK'
    uvd2.vis_units = 'mK'
    
    pol = uvd1.polarization_array[0]
    nfreq = uvd1.Nfreqs
    
    # Configure beam
    if beam is None:
        beam = hp.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_efield.beamfits"))
    if cosmo is not None:
        beam.cosmo = cosmo
    
    # Create PSpecData
    ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], 
                      labels=['n1', 'n2'], beam=beam)
    # Get baselines
    if bls is None:
        bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
    
    # Calculate power spectra
    uvp = ds.pspec(bls, bls, (0, 1), (pol, pol), input_data_weight=weighting, 
                   spw_ranges=[(0, nfreq),], little_h=True, verbose=False, 
                   store_cov=False,
                   history="noise power spectra from hera_pspec/data/"
                           "hera_noise_sim.uv")
    return ds, uvp


def build_vanilla_uvpspec(beam=None):
    """
    Build an example UVPSpec object, with all necessary metadata.
    """
    uvp = hp.uvpspec.UVPSpec()

    if beam is None:
        beam = hp.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_efield.beamfits"))

    cosmo = beam.cosmo

    Ntimes = 10
    Nfreqs = 50
    Ndlys = 30
    Nspws = 1
    Nspwfreqs = 1 * Nfreqs
    Nspwdlys = 1 * Ndlys

    # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
    blpairs = [1002001002, 2003002003, 1003001003]
    bls = [1002, 2003, 1003]
    Nbls = len(bls)
    Nblpairs = len(blpairs)
    Nblpairts = Nblpairs * Ntimes

    blpair_array = np.tile(blpairs, Ntimes)
    bl_array = np.array(bls)
    bl_vecs = np.array([[  5.33391548e+00,  -1.35907816e+01,  -7.91624188e-09],
                        [ -8.67982998e+00,   4.43554478e+00,  -1.08695203e+01],
                        [ -3.34591450e+00,  -9.15523687e+00,  -1.08695203e+01]])
    time_array = np.repeat(np.linspace(2458042.1, 2458042.2, Ntimes), Nblpairs)
    time_1_array = time_array
    time_2_array = time_array
    lst_array = np.repeat(np.ones(Ntimes, dtype=np.float), Nblpairs)
    lst_1_array = lst_array
    lst_2_array = lst_array
    time_avg_array = time_array
    lst_avg_array = lst_array
    spw_freq_array = np.tile(np.arange(Nspws), Nfreqs)
    spw_dly_array = np.tile(np.arange(Nspws), Ndlys)
    spw_array = np.arange(Nspws)
    freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), Nspws)
    dly_array = np.repeat(hp.utils.get_delays(freq_array, n_dlys=Ndlys), Nspws)
    pol_array = np.array([-5])
    Npols = len(pol_array)
    vis_units = 'unknown'
    norm_units = 'Hz str'
    weighting = 'identity'
    channel_width = np.median(np.diff(freq_array))
    history = 'example'
    taper = "none"
    norm = "I"
    git_hash = hp.version.git_hash
    scalar_array = np.ones((Nspws, Npols), np.float)
    label1 = 'red'
    label2 = 'blue'
    labels = np.array([label1, label2])
    label_1_array = np.ones((Nspws, Nblpairts, Npols), np.int) * 0
    label_2_array = np.ones((Nspws, Nblpairts, Npols), np.int) * 1
    
    OmegaP, OmegaPP = beam.get_Omegas(beam.primary_beam.polarization_array[0])
    beam_freqs = beam.beam_freqs

    telescope_location = np.array([5109325.85521063, 
                                   2005235.09142983, 
                                  -3239928.42475397])

    store_cov = True
    data_array, wgt_array, integration_array = {}, {}, {}
    nsample_array, cov_array = {}, {}
    
    for s in spw_array:
        data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) \
                      * blpair_array[:, None, None] / 1e9
        wgt_array[s] = np.ones((Nblpairts, Ndlys, 2, Npols), dtype=np.float)
        integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
        nsample_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
        
        cov_array[s] = np.moveaxis(
                    np.array([ [np.identity(Ndlys,dtype=np.complex)\
                                for m in range(Nblpairts) ] 
                              for n in range(Npols)]), 0, -1)
    
    params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nspwfreqs', 'Nspws', 
              'Nblpairs', 'Nblpairts', 'Npols', 'Ndlys', 'Nbls', 
              'blpair_array', 'time_1_array','time_2_array', 'lst_1_array', 
              'lst_2_array', 'spw_array',
              'dly_array', 'freq_array', 'pol_array', 'data_array', 'wgt_array',
              'integration_array', 'bl_array', 'bl_vecs', 'telescope_location',
              'vis_units', 'channel_width', 'weighting', 'history', 'taper', 'norm',
              'git_hash', 'nsample_array', 'time_avg_array', 'lst_avg_array',
              'cosmo', 'scalar_array', 'labels', 'norm_units', 'label_1_array',
              'label_2_array', 'store_cov', 'cov_array', 'spw_dly_array', 
              'spw_freq_array']

    if beam is not None:
        params += ['OmegaP', 'OmegaPP', 'beam_freqs']

    # Set all parameters
    for p in params:
        setattr(uvp, p, locals()[p])

    uvp.check()

    return uvp, cosmo


def uvpspec_from_data(data, bl_grps, data_std=None, spw_ranges=None, beam=None, 
                      taper='none', cosmo=None, verbose=False):
    """
    Build an example UVPSpec object from a visibility file and PSpecData.

    Parameters
    ----------
    data : UVData object or str
        This can be a UVData object or a string filepath to a miriad file.

    bl_grps : list
        This is a list of baseline groups (e.g. redundant groups) to form 
        blpairs from.
        Ex: [[(24, 25), (37, 38), ...], [(24, 26), (37, 39), ...], ... ]

    data_std: UVData object or str or None
        Can be UVData object or a string filepath to a miriad file.

    spw_ranges : list
        List of spectral window tuples. See PSpecData.pspec docstring for details.

    beam : PSpecBeamBase subclass or str
        This can be a subclass of PSpecBeamBase of a string filepath to a
        UVBeam healpix map.

    taper : string
        Optional tapering applied to the data before OQE.

    cosmo : Cosmo_Conversions object

    verbose : bool
        if True, report feedback to standard output

    Returns
    -------
    uvp : UVPSpec object
    """
    # Load data
    if isinstance(data, str):
        uvd = UVData()
        uvd.read_miriad(data)
    elif isinstance(data, UVData):
        uvd = data

    if isinstance(data_std, str):
        uvd_std = UVData()
        uvd_std.read_miriad(data_std)
    elif isinstance(data_std, UVData):
        uvd_std = data_std
    else:
        uvd_std = None
    if uvd_std is not None:
        store_cov = True
    else:
        store_cov=False
    
    # Get pol
    pol = uvd.polarization_array[0]

    # Load beam
    if isinstance(beam, str):
        beam = hp.PSpecBeamUV(beam, cosmo=cosmo)
    if beam is not None and cosmo is not None:
        beam.cosmo = cosmo
    
    # Instantiate pspecdata
    ds = hp.PSpecData(dsets=[uvd, uvd], dsets_std=[uvd_std, uvd_std], 
                      wgts=[None, None], labels=['d1', 'd2'], beam=beam)

    # Get blpair groups
    assert isinstance(bl_grps, list), "bl_grps must be a list"
    if not isinstance(bl_grps[0], list): bl_grps = [bl_grps]
    assert np.all([isinstance(blgrp, list) for blgrp in bl_grps]), \
        "bl_grps must be fed as a list of lists"
    assert np.all([isinstance(blgrp[0], tuple) for blgrp in bl_grps]), \
        "bl_grps must be fed as a list of lists of tuples"
    
    bls1, bls2 = [], []
    for blgrp in bl_grps:
        _bls1, _bls2, _ = hp.utils.construct_blpairs(blgrp, 
                                                     exclude_auto_bls=True, 
                                                     exclude_permutations=True)
        bls1.extend(_bls1)
        bls2.extend(_bls2)

    # Run pspec
    uvp = ds.pspec(bls1, bls2, (0, 1), (pol, pol), input_data_weight='identity', 
                   spw_ranges=spw_ranges, taper=taper, verbose=verbose, 
                   store_cov=store_cov)
    return uvp

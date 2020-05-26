#!/usr/bin/env python2
import numpy as np
import copy, operator, itertools
from collections import OrderedDict as odict
from pyuvdata import UVData
from hera_cal.utils import JD2LST
from scipy import stats

from . import uvpspec, pspecdata, conversions, pspecbeam, utils, uvpspec_utils as uvputils


def build_vanilla_uvpspec(beam=None):
    """
    Build an example vanilla UVPSpec object from scratch, with all necessary
    metadata.

    Parameters
    ----------
    beam : PSpecBeamBase subclass
    covariance: if true, compute covariance

    Returns
    -------
    uvp : UVPSpec object
    """
    uvp = uvpspec.UVPSpec()

    Ntimes = 10
    Nfreqs = 50
    Ndlys = 30
    Nspws = 1
    Nspwfreqs = 1 * Nfreqs
    Nspwdlys = 1 * Ndlys

    # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
    blpairs = [101102101102, 102103102103, 101103101103]
    bls = [101102, 102103, 101103]
    Nbls = len(bls)
    Nblpairs = len(blpairs)
    Nblpairts = Nblpairs * Ntimes

    blpair_array = np.tile(blpairs, Ntimes)
    bl_array = np.array(bls)
    bl_vecs = np.array([[  5.33391548e+00,  -1.35907816e+01,  -7.91624188e-09],
                        [ -8.67982998e+00,   4.43554478e+00,  -1.08695203e+01],
                        [ -3.34591450e+00,  -9.15523687e+00,  -1.08695203e+01]])
    time_array = np.linspace(2458042.1, 2458042.2, Ntimes)
    lst_array = JD2LST(time_array, longitude=21.4283)
    time_array = np.repeat(time_array, Nblpairs)
    time_1_array = time_array
    time_2_array = time_array
    lst_array = np.repeat(lst_array, Nblpairs)
    lst_1_array = lst_array
    lst_2_array = lst_array
    time_avg_array = time_array
    lst_avg_array = lst_array
    spw_freq_array = np.tile(np.arange(Nspws), Nfreqs)
    spw_dly_array = np.tile(np.arange(Nspws), Ndlys)
    spw_array = np.arange(Nspws)
    freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False),
                           Nspws)
    dly_array = np.repeat(utils.get_delays(freq_array, n_dlys=Ndlys), Nspws)
    polpair_array = np.array([1515,]) # corresponds to ('xx','xx')
    Npols = len(polpair_array)
    vis_units = 'unknown'
    norm_units = 'Hz str'
    weighting = 'identity'
    channel_width = np.median(np.diff(freq_array))
    history = 'example'
    taper = "none"
    norm = "I"
    git_hash = "random"
    scalar_array = np.ones((Nspws, Npols), np.float)
    label1 = 'red'
    label2 = 'blue'
    r_params = ''
    cov_model = 'dsets'
    labels = np.array([label1, label2])
    label_1_array = np.ones((Nspws, Nblpairts, Npols), np.int) * 0
    label_2_array = np.ones((Nspws, Nblpairts, Npols), np.int) * 1
    if beam is not None:
        pol = beam.primary_beam.polarization_array[0]
        OmegaP, OmegaPP = beam.get_Omegas((pol,pol))
        beam_freqs = beam.beam_freqs

    # HERA coordinates in Karoo Desert, SA
    telescope_location = np.array([5109325.85521063,
                                   2005235.09142983,
                                  -3239928.42475397])

    store_cov = True
    cosmo = conversions.Cosmo_Conversions()

    data_array, wgt_array = {}, {}
    integration_array, nsample_array, cov_array_real, cov_array_imag = {}, {}, {}, {}
    window_function_array = {}
    for s in spw_array:
        data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) \
                      * blpair_array[:, None, None] / 1e9
        wgt_array[s] = np.ones((Nblpairts, Nfreqs, 2, Npols), dtype=np.float)
        # NB: The wgt_array has dimensions Nfreqs rather than Ndlys; it has the
        # dimensions of the input visibilities, not the output delay spectra
        integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
        nsample_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
        window_function_array[s] = np.ones((Nblpairts, Ndlys, Ndlys, Npols), dtype=np.float64)
        cov_array_real[s] = np.moveaxis(np.array([[np.identity(Ndlys,dtype=np.float) \
                                             for m in range(Nblpairts)]
                                             for n in range(Npols)]), 0, -1)
        cov_array_imag[s] = np.moveaxis(np.array([[np.identity(Ndlys,dtype=np.float) \
                                             for m in range(Nblpairts)]
                                             for n in range(Npols)]), 0, -1)

    params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nspwfreqs', 'Nspws',
              'Nblpairs', 'Nblpairts', 'Npols', 'Ndlys', 'Nbls',
              'blpair_array', 'time_1_array', 'time_2_array',
              'lst_1_array', 'lst_2_array', 'spw_array',
              'dly_array', 'freq_array', 'polpair_array', 'data_array',
              'wgt_array', 'r_params', 'window_function_array',
              'integration_array', 'bl_array', 'bl_vecs', 'telescope_location',
              'vis_units', 'channel_width', 'weighting', 'history', 'taper',
              'norm', 'git_hash', 'nsample_array', 'time_avg_array',
              'lst_avg_array', 'cosmo', 'scalar_array', 'labels', 'norm_units',
              'labels', 'label_1_array', 'label_2_array', 'store_cov',
              'cov_array_real', 'cov_array_imag', 'spw_dly_array', 'spw_freq_array', 'cov_model']

    if beam is not None:
        params += ['OmegaP', 'OmegaPP', 'beam_freqs']

    # Set all parameters
    for p in params:
        setattr(uvp, p, locals()[p])

    uvp.check()

    return uvp, cosmo


def uvpspec_from_data(data, bl_grps, data_std=None, spw_ranges=None,
                      data_weighting='identity', beam=None, taper='none', cosmo=None,
                      n_dlys=None, r_params=None, verbose=False, **kwargs):
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

    data_std: UVData object or str, optional
        Can be UVData object or a string filepath to a miriad file.
        Default: None.

    spw_ranges : list, optional
        List of spectral window tuples. See PSpecData.pspec docstring for
        details. Default: None.

    data_weighting : str, optional
        R matrix specification in QE formalism. See PSpecData.pspec for details.
        Default: 'identity'

    beam : PSpecBeamBase subclass or str, optional
        This can be a subclass of PSpecBeamBase of a string filepath to a
        UVBeam healpix map. Default: None.

    taper : str, optional
        Optional tapering applied to the data before OQE. Default: 'none'.

    cosmo : Cosmo_Conversions object
        Cosmology object.

    n_dlys : int, optional
        Number of delay bins to use. Default: None (uses as many delay bins as
        frequency channels).

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
    verbose : bool, optional
        if True, report feedback to standard output. Default: False.

    kwargs : dict, optional
        Additional kwargs to pass to PSpecData.pspec()
    
    Returns
    -------
    uvp : UVPSpec object
    """
    # load data
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
        cov_model = "dsets"
    else:
        store_cov = False
        cov_model = "empirical"

    # get pol
    pol = uvd.polarization_array[0]

    # load beam
    if isinstance(beam, str):
        beam = pspecbeam.PSpecBeamUV(beam, cosmo=cosmo)
    if beam is not None and cosmo is not None:
        beam.cosmo = cosmo

    # instantiate pspecdata
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], dsets_std=[uvd_std, uvd_std],
                             wgts=[None, None], labels=['d1', 'd2'], beam=beam)

    # get blpair groups
    assert isinstance(bl_grps, list), "bl_grps must be a list"
    if not isinstance(bl_grps[0], list): bl_grps = [bl_grps]
    assert np.all([isinstance(blgrp, list) for blgrp in bl_grps]), \
        "bl_grps must be fed as a list of lists"
    assert np.all([isinstance(blgrp[0], tuple) for blgrp in bl_grps]), \
        "bl_grps must be fed as a list of lists of tuples"
    bls1, bls2 = [], []
    for blgrp in bl_grps:
        _bls1, _bls2, _ = utils.construct_blpairs(blgrp, exclude_auto_bls=True,
                                                  exclude_permutations=True)
        bls1.extend(_bls1)
        bls2.extend(_bls2)

    # run pspec
    uvp, uvp_q = ds.pspec(bls1, bls2, (0, 1), (pol, pol), input_data_weight=data_weighting,
                   spw_ranges=spw_ranges, taper=taper, verbose=verbose,
                   store_cov=store_cov, n_dlys=n_dlys, r_params=r_params,
                   cov_model=cov_model, **kwargs)
    return uvp


def noise_sim(data, Tsys, beam=None, Nextend=0, seed=None, inplace=False,
              whiten=False, run_check=True):
    """
    Generate a simulated Gaussian noise (visibility) realization given
    a system temperature Tsys. If a primary beam model is not provided,
    this is in units of Kelvin-steradians

        Trms = Tsys / sqrt(channel_width * integration_time)

    where Trms is divided by an additional sqrt(2) if the polarization
    in data is a pseudo-Stokes polarization. If a primary beam model is
    provided, the output is converted to Jansky.

    Parameters
    ----------
    data : str or UVData object
        A UVData object or path to miriad file.

    Tsys : float
        System temperature in Kelvin.

    beam : str or PSpecBeam object, optional
        A PSpecBeam object or path to beamfits file.

    Nextend : int, optional
        Number of times to extend time axis by default length
        before creating noise sim. Can be used to increase
        number statistics before forming noise realization.

    seed : int, optional
        Seed to set before forming noise realization.

    inplace : bool, optional
        If True, overwrite input data and return None, else
        make a copy and return copy.

    whiten : bool, optional
        If True, clear input data of flags if they exist and
        set all nsamples to 1.

    run_check : bool, optional
        If True, run UVData check before return.

    Returns
    -------
    data : UVData with noise realizations.
    """
    # Read data files
    if isinstance(data, (str, np.str)):
        _data = UVData()
        _data.read_miriad(data)
        data = _data
    elif isinstance(data, UVData):
        if not inplace:
            data = copy.deepcopy(data)
    assert isinstance(data, UVData)

    # whiten input data
    if whiten:
        data.flag_array[:] = False
        data.nsample_array[:] = 1.0

    # Configure beam
    if beam is not None:
        if isinstance(beam, (str, np.str)):
            beam = pspecbeam.PSpecBeamUV(beam)
        assert isinstance(beam, pspecbeam.PSpecBeamBase)

    # Extend times
    Nextend = int(Nextend)
    if Nextend > 0:
        assert data.phase_type == 'drift', "data must be drift phased in order to extend along time axis"
        data = copy.deepcopy(data)
        _data = copy.deepcopy(data)
        dt = np.median(np.diff(np.unique(_data.time_array)))
        dl = np.median(np.diff(np.unique(_data.lst_array)))
        for i in range(Nextend):
            _data.time_array += dt * _data.Ntimes * (i+1)
            _data.lst_array += dl * _data.Ntimes * (i+1)
            _data.lst_array %= 2*np.pi
            data += _data

    # Get Trms
    int_time = data.integration_time
    if not isinstance(int_time, np.ndarray):
        int_time = np.array([int_time])
    Trms = Tsys / np.sqrt(int_time[:, None, None, None] * data.nsample_array * data.channel_width)

    # if a pol is pStokes pol, divide by extra sqrt(2)
    polcorr = np.array([np.sqrt(2) if p in [1, 2, 3, 4] else 1.0 for p in data.polarization_array])
    Trms /= polcorr

    # Get Vrms in Jy using beam
    if beam is not None:
        freqs = np.unique(data.freq_array)[None, None, :, None]
        K_to_Jy = [1e3 / (beam.Jy_to_mK(freqs.squeeze(), pol=p)) for p in data.polarization_array]
        K_to_Jy = np.array(K_to_Jy).T[None, None, :, :]
        K_to_Jy /= np.array([np.sqrt(2) if p in [1, 2, 3, 4] else 1.0 for p in data.polarization_array])
        rms = K_to_Jy * Trms
    else:
        rms = Trms

    # Generate noise
    if seed is not None:
        np.random.seed(seed)
    data.data_array = (stats.norm.rvs(0, 1./np.sqrt(2), size=rms.size).reshape(rms.shape) \
                       + 1j * stats.norm.rvs(0, 1./np.sqrt(2), size=rms.size).reshape(rms.shape) ) * rms
    f = np.isnan(data.data_array) + np.isinf(data.data_array)
    data.data_array[f] = np.nan
    data.flag_array[f] = True

    if run_check:
        data.check()

    if not inplace:
        return data

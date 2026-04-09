#!/usr/bin/env python2
import numpy as np
import copy, operator, itertools
from collections import OrderedDict as odict
from pyuvdata import UVData
from hera_cal.utils import JD2LST
from scipy import stats, interpolate
from astropy import constants

from . import (
    uvpspec,
    pspecdata,
    conversions,
    pspecbeam,
    utils,
    uvpspec_utils as uvputils,
)


def build_vanilla_uvpspec(
    beam: pspecbeam.PSpecBeamBase | None=None,
    Ndlys: int | None = 30,
    equal_time_arrays: bool = True
) -> tuple[uvpspec.UVPSpec, conversions.Cosmo_Conversions]:
    """
    Build an example vanilla UVPSpec object from scratch, with all necessary
    metadata.

    Parameters
    ----------
    beam : PSpecBeamBase subclass
        A beam to use for the UVPSpec object. If None, no beam is used.
    Ndlys : int, optional
        Number of delay bins to use. If None, uses as many delay bins as
        frequency channels. Default is 30, which was the original default, but
        is *different* than the number of frequency channels, which can break 
        window functions.
    equal_time_arrays
        If True, the time_1_array and time_2_array will be equal. If False,
        they will be different. Default is True.
        
    Returns
    -------
    uvp : UVPSpec object
    """
    uvp = uvpspec.UVPSpec()

    Ntimes = 10
    
    uvp.Nfreqs = 50
    uvp.Ndlys = uvp.Nfreqs if Ndlys is None else Ndlys
    uvp.Nspws = 1
    uvp.Nspwfreqs = uvp.Nspws * uvp.Nfreqs
    uvp.Nspwdlys = uvp.Nspws * uvp.Ndlys

    # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
    blpairs = [101102101102, 102103102103, 101103101103]
    bls = [101102, 102103, 101103]
    uvp.Nbls = len(bls)
    uvp.Nblpairs = len(blpairs)
    #uvp.Nblpairts = uvp.Nblpairs * Ntimes

    time_array = np.linspace(2458042.1, 2458042.2, Ntimes)
    uvp.time_1_array = np.tile(time_array, uvp.Nblpairs)
    if equal_time_arrays:
        uvp.time_2_array = uvp.time_1_array.copy()
    else:
        # Make time 2 array in-between time 1 array.
        uvp.time_2_array = uvp.time_1_array + (time_array[1] - time_array[0]) / 2.0
        
    uvp.blpair_array = np.tile(blpairs, Ntimes)
    uvp.bl_array = np.array(bls)
    uvp.bl_vecs = np.array(
        [
            [5.33391548e00, -1.35907816e01, -7.91624188e-09],
            [-8.67982998e00, 4.43554478e00, -1.08695203e01],
            [-3.34591450e00, -9.15523687e00, -1.08695203e01],
        ]
    )
    lst_array = JD2LST(time_array, longitude=21.4283)
    #time_array = np.repeat(time_array, Nblpairs)
    uvp.Ntpairs = len(set([(t1, t2) for t1, t2 in zip(uvp.time_1_array, uvp.time_2_array)]))
    uvp.Nbltpairs = len(set([(blp, t1, t2) for blp, t1, t2 in zip(uvp.blpair_array, uvp.time_1_array, uvp.time_2_array)]))
    
    uvp.lst_1_array = JD2LST(uvp.time_1_array, longitude=21.4283)
    uvp.lst_2_array = JD2LST(uvp.time_2_array, longitude=21.4283)
    uvp.time_avg_array = (uvp.time_1_array + uvp.time_2_array)/2
    uvp.lst_avg_array = JD2LST(uvp.time_avg_array, longitude=21.4283)
    
    uvp.spw_freq_array = np.tile(np.arange(uvp.Nspws), uvp.Nfreqs)
    uvp.spw_dly_array = np.tile(np.arange(uvp.Nspws), uvp.Ndlys)
    uvp.spw_array = np.arange(uvp.Nspws)
    uvp.freq_array = np.linspace(100e6, 105e6, uvp.Nfreqs, endpoint=False)
    uvp.dly_array = utils.get_delays(uvp.freq_array, n_dlys=uvp.Ndlys)
    uvp.polpair_array = np.array([1515])  # corresponds to ('xx','xx')
    uvp.Npols = len(uvp.polpair_array)
    
    uvp.vis_units = "unknown"
    uvp.norm_units = "Hz str"
    uvp.weighting = "identity"
    uvp.channel_width = np.ones(uvp.Nfreqs) * np.median(np.diff(uvp.freq_array))
    uvp.history = "example"
    uvp.taper = "none"
    uvp.norm = "I"
    uvp.git_hash = "random"
    uvp.scalar_array = np.ones((uvp.Nspws, uvp.Npols), float)
    uvp.r_params = ""
    uvp.cov_model = "dsets"
    uvp.exact_windows = False
    
    label1 = "red"
    label2 = "blue"
    uvp.labels = np.array([label1, label2])
    uvp.label_1_array = np.ones((uvp.Nspws, uvp.Nbltpairs, uvp.Npols), int) * 0
    uvp.label_2_array = np.ones((uvp.Nspws, uvp.Nbltpairs, uvp.Npols), int) * 1
    if beam is not None:
        pol = beam.primary_beam.polarization_array[0]
        uvp.OmegaP, uvp.OmegaPP = beam.get_Omegas((pol, pol))
        uvp.beam_freqs = beam.beam_freqs

    # HERA coordinates in Karoo Desert, SA
    uvp.telescope_location = np.array(
        [5109325.85521063, 2005235.09142983, -3239928.42475397]
    )

    uvp.store_cov = True
    cosmo = conversions.Cosmo_Conversions()

    data_array, wgt_array = {}, {}
    integration_array, nsample_array, cov_array_real, cov_array_imag = {}, {}, {}, {}
    window_function_array = {}
    for s in uvp.spw_array:
        data_array[s] = (
            np.ones((uvp.Nbltpairs, uvp.Ndlys, uvp.Npols), dtype=complex)
            * uvp.blpair_array[:, None, None]
            / 1e9
        )
        wgt_array[s] = np.ones((uvp.Nbltpairs, uvp.Nfreqs, 2, uvp.Npols), dtype=float)
        # NB: The wgt_array has dimensions Nfreqs rather than Ndlys; it has the
        # dimensions of the input visibilities, not the output delay spectra
        integration_array[s] = np.ones((uvp.Nbltpairs, uvp.Npols), dtype=float)
        nsample_array[s] = np.ones((uvp.Nbltpairs, uvp.Npols), dtype=float)
        window_function_array[s] = np.ones(
            (uvp.Nbltpairs, uvp.Ndlys, uvp.Ndlys, uvp.Npols), dtype=np.float64
        )
        cov_array_real[s] = np.moveaxis(
            np.array(
                [
                    [np.identity(uvp.Ndlys, dtype=float) for _ in range(uvp.Nbltpairs)]
                    for _ in range(uvp.Npols)
                ]
            ),
            0,
            -1,
        )
        cov_array_imag[s] = np.moveaxis(
            np.array(
                [
                    [np.identity(uvp.Ndlys, dtype=float) for _ in range(uvp.Nbltpairs)]
                    for _ in range(uvp.Npols)
                ]
            ),
            0,
            -1,
        )

    uvp.data_array = data_array
    uvp.wgt_array = wgt_array
    uvp.cov_array_real = cov_array_real
    uvp.cov_array_imag = cov_array_imag
    uvp.integration_array = integration_array
    uvp.nsample_array = nsample_array
    uvp.window_function_array = window_function_array
    uvp.cosmo = cosmo
    
    # From v0.5, this must always be true.
    uvp.Ntimes = uvp.Ntpairs
    
    uvp.check()

    return uvp, cosmo


def uvpspec_from_data(
    data,
    bl_grps,
    data_std=None,
    spw_ranges=None,
    data_weighting="identity",
    beam=None,
    taper="none",
    cosmo=None,
    n_dlys=None,
    r_params=None,
    verbose=False,
    **kwargs
):
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
    ds = pspecdata.PSpecData(
        dsets=[uvd, uvd],
        dsets_std=[uvd_std, uvd_std],
        wgts=[None, None],
        labels=["d1", "d2"],
        beam=beam,
    )

    # get blpair groups
    assert isinstance(bl_grps, list), "bl_grps must be a list"
    if not isinstance(bl_grps[0], list):
        bl_grps = [bl_grps]
    assert np.all(
        [isinstance(blgrp, list) for blgrp in bl_grps]
    ), "bl_grps must be fed as a list of lists"
    assert np.all(
        [isinstance(blgrp[0], tuple) for blgrp in bl_grps]
    ), "bl_grps must be fed as a list of lists of tuples"
    bls1, bls2 = [], []
    for blgrp in bl_grps:
        _bls1, _bls2, _ = utils.construct_blpairs(
            blgrp, exclude_auto_bls=True, exclude_permutations=True
        )
        bls1.extend(_bls1)
        bls2.extend(_bls2)

    # run pspec
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        (pol, pol),
        input_data_weight=data_weighting,
        spw_ranges=spw_ranges,
        taper=taper,
        verbose=verbose,
        store_cov=store_cov,
        n_dlys=n_dlys,
        r_params=r_params,
        cov_model=cov_model,
        **kwargs
    )
    return uvp


def noise_sim(
    data,
    Tsys,
    beam=None,
    Nextend=0,
    seed=None,
    inplace=False,
    whiten=False,
    run_check=True,
):
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
    if isinstance(data, str):
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
        if isinstance(beam, str):
            beam = pspecbeam.PSpecBeamUV(beam)
        assert isinstance(beam, pspecbeam.PSpecBeamBase)

    # Extend times
    Nextend = int(Nextend)
    if Nextend > 0:
        assert (
            data.phase_center_catalog[0]['cat_type'] == "unprojected"
        ), "data must be unprojected in order to extend along time axis"
        data = copy.deepcopy(data)
        _data = copy.deepcopy(data)
        dt = np.median(np.diff(np.unique(_data.time_array)))
        dl = np.median(np.diff(np.unique(_data.lst_array)))
        for i in range(Nextend):
            _data.time_array += dt * _data.Ntimes * (i + 1)
            _data.lst_array += dl * _data.Ntimes * (i + 1)
            _data.lst_array %= 2 * np.pi
            data += _data

    # Get Trms
    int_time = data.integration_time
    if not isinstance(int_time, np.ndarray):
        int_time = np.array([int_time])
    Trms = Tsys / np.sqrt(
        int_time[:, None, None] * data.nsample_array * data.channel_width[None, :, None]
    )

    # if a pol is pStokes pol, divide by extra sqrt(2)
    polcorr = np.array(
        [np.sqrt(2) if p in [1, 2, 3, 4] else 1.0 for p in data.polarization_array]
    )
    Trms /= polcorr

    # Get Vrms in Jy using beam
    if beam is not None:
        freqs = np.unique(data.freq_array)[None, :, None]
        K_to_Jy = [
            1e3 / (beam.Jy_to_mK(freqs.squeeze(), pol=p))
            for p in data.polarization_array
        ]
        K_to_Jy = np.array(K_to_Jy).T[None, :, :]
        K_to_Jy /= np.array(
            [np.sqrt(2) if p in [1, 2, 3, 4] else 1.0 for p in data.polarization_array]
        )
        rms = K_to_Jy * Trms
    else:
        rms = Trms

    # Generate noise
    if seed is not None:
        np.random.seed(seed)
    data.data_array = (
        stats.norm.rvs(0, 1.0 / np.sqrt(2), size=rms.size).reshape(rms.shape)
        + 1j * stats.norm.rvs(0, 1.0 / np.sqrt(2), size=rms.size).reshape(rms.shape)
    ) * rms
    f = np.isnan(data.data_array) + np.isinf(data.data_array)
    data.data_array[f] = np.nan
    data.flag_array[f] = True

    if run_check:
        data.check()

    if not inplace:
        return data


def gauss_cov_fg(cov_amp, cov_length_scale, freqs, Ntimes=100, constant_in_time=True):
    """
    Generate a random foreground signal from a Gaussian covariance

        C = cov_amp * exp(-((f1 - f2) / cov_length_scale)^2)

    Parameters
    ----------
    cov_amp : float
        Covariance amplitude (e.g. in Jy^2)
    cov_length_scale : float
        Length scale of correlation in frequency (in freqs units)
    freqs : ndarray
        Frequency array [Hz]
    Ntimes : int
        Number of time integrations
    constant_in_time : bool
        If True, draw one signal realization for all times
        Else, draw an independent realization for each time

    Returns
    -------
    ndarray
        shape (Ntimes, Nfreqs)
    """
    # generate a random process from a Gaussian covariance
    Nfreqs = len(freqs)

    C = cov_amp * np.exp(
        -(((freqs[:, None] - freqs[None, :]) / cov_length_scale) ** 2)
    )  # a covariance model

    if constant_in_time:
        s = stats.multivariate_normal.rvs(np.zeros(Nfreqs), C, 2).reshape(2, 1, Nfreqs)
        s = s[0] + 1j * s[1]
        s = np.repeat(s, Ntimes, axis=0)

    else:
        s = stats.multivariate_normal.rvs(np.zeros(Nfreqs), C, 2 * Ntimes).reshape(
            2, Ntimes, Nfreqs
        )
        s = s[0] + 1j * s[1]

    return s


def sky_noise_jy_autos(lsts, freqs, autovis, omega_p, integration_time, channel_width=None, Trx=0.0):
    """Make a noise realization for a given auto-visibility level and beam.

    This is a simple replacement for ``hera_sim.noise.sky_noise_jy``.

    Parameters
    ----------
    lsts : array_like
        LSTs at which to compute the sky noise.
    freqs : array_like
        Frequencies at which to compute the sky noise, in Hz.
    autovis : float
        Autocorrelation visibility amplitude, in Jy.
    omega_p : callable or array_like
        A function of frequency giving the integrated beam area.
    integration_time : float, optional
        Integration time in seconds. By default, use the average difference
        between given LSTs.
    channel_width : float, optional
        Channel width in Hz, by default the mean difference between frequencies.
    Trx : float, optional
        Receiver temperature, in K.

    Returns
    -------
    noise : ndarray
        2D array of white noise in LST/freq.
    """
    # Beam solid angle
    if callable(omega_p):
        omega_p = omega_p(freqs)
    assert omega_p.size == freqs.size, "`omega_p` must have the same length as `freqs`"

    if channel_width is None:
        channel_width = np.mean(np.diff(freqs))

    # Calculate Jansky to Kelvin conversion factor
    # The factor of 1e-26 converts from Jy to W/m^2/Hz.
    wavelengths = conversions.units.c / freqs  # meters
    Jy2K = 1e-26 * wavelengths ** 2 / (2 * conversions.units.kb * omega_p)

    # Use autocorrelation vsibility to set noise scale
    Tsky = autovis * Jy2K.reshape(1, -1)
    Tsky += Trx  # add in the receiver temperature

    # Calculate noise visibility sts. dev. in Jy (assuming Tsky is in K)
    vis = Tsky / np.sqrt(integration_time * channel_width) / Jy2K.reshape(1, -1)

    # Make noise realization
    x1 = np.random.normal(scale=1.0 / np.sqrt(2), size=vis.shape)
    x2 = np.random.normal(scale=1.0 / np.sqrt(2), size=vis.shape)
    return (x1 + 1.0j * x2) * vis


def sky_noise_sim(
    data,
    beam,
    cov_amp=1000,
    cov_length_scale=10,
    constant_per_bl=True,
    constant_in_time=True,
    bl_loop_seed=None,
    divide_by_nsamp=False,
):
    """
    Generate a mock simulation of foreground sky + noise.

    Noise component is drawn from the autos in data.
    Sky component is drawn from a Gaussian covariance.
    Each cross correlation is statistically independent.

    Parameters
    ----------
    data : str or UVData object
    beam : str or PSpecBeam object
    cov_amp : float
        Covariance amplitude. See gauss_cov_fg()
    cov_length_scale : float
        Covariance length scale [MHz]. See gauss_cov_fg()
    constant_in_time : bool
        If True, foreground signal is constant in time
    constant_per_bl : bool
        If True, foreground signal is constant across baselines
    bl_loop_seed : int
        random seed to use before starting per-baseline loop
    divide_by_nsamp : bool
        If True, divide noise sim by sqrt(Nsample) in data

    Returns
    -------
    UVData object
    """
    STOKPOLS = ["PI", "PU", "PQ", "PV"]
    AUTOVISPOLS = ["XX", "YY", "EE", "NN"] + STOKPOLS

    if isinstance(data, str):
        uvd = UVData()
        uvd.read(data)
    else:
        uvd = copy.deepcopy(data)
    assert (
        -7 not in uvd.polarization_array and -8 not in uvd.polarization_array
    ), "Does not operate on cross-hand polarizations"

    if isinstance(beam, str):
        beam = pspecbeam.PSpecBeamUV(beam)

    # get metadata
    freqs = uvd.freq_array
    Ntimes = uvd.Ntimes
    lsts = np.unique(uvd.lst_array)
    int_time = np.median(uvd.integration_time)
    pols = uvd.get_pols()

    # get OmegaP from beam
    OmegaP = {}
    for pol in uvd.get_pols():
        # replace pQ, pU, pV with pI
        if pol.upper() in ["PQ", "PU", "PV"]:
            OmegaP[pol] = beam.power_beam_int("pI")
        else:
            OmegaP[pol] = beam.power_beam_int(pol)
        # interpolate to freq_array of data
        OmegaP[pol] = interpolate.interp1d(
            beam.primary_beam.freq_array,
            OmegaP[pol],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(freqs)

    # get baselines
    bls = uvd.get_antpairpols()
    crossbls = [bl for bl in bls if bl[0] != bl[1]]

    # get autos in Kelvin
    autos = {}
    for key in bls:
        if key[0] == key[1] and key[2].upper() in AUTOVISPOLS:
            # handle stokespols
            if key[2].upper() in ["PQ", "PU", "PV"]:
                autos[key] = uvd.get_data(key[:2] + ("pI",)).real
            else:
                autos[key] = uvd.get_data(key).real

    # get signal if constant across bl
    if constant_per_bl:
        # get signal
        sig = gauss_cov_fg(
            cov_amp,
            cov_length_scale * 1e6,
            freqs,
            Ntimes=Ntimes,
            constant_in_time=constant_in_time,
        )

    # iterate over cross correlations
    np.random.seed(bl_loop_seed)
    for bl in crossbls:
        # get time and freq dependent Tsys for this baseline
        Tsys_jy = np.sqrt(autos[(bl[0], bl[0], bl[2])] * autos[(bl[1], bl[1], bl[2])])

        # get raw thermal noise
        n = sky_noise_jy_autos(
            lsts,
            freqs,
            autovis=Tsys_jy,
            omega_p=OmegaP[bl[2]],
            integration_time=int_time,
        )

        # divide by nsamples: set nsample==0 to inf
        if divide_by_nsamp:
            nsamp = uvd.get_nsamples(bl).copy()
            nsamp[np.isclose(nsamp, 0)] = np.inf
            n /= np.sqrt(nsamp)

        # get signal
        if not constant_per_bl:
            sig = gauss_cov_fg(
                cov_amp,
                cov_length_scale * 1e6,
                freqs,
                Ntimes=Ntimes,
                constant_in_time=constant_in_time,
            )

        # fill data
        blt_inds = uvd.antpair2ind(bl[:2])
        pol_ind = pols.index(bl[2])
        uvd.data_array[blt_inds, :, pol_ind] = n + sig

    return uvd

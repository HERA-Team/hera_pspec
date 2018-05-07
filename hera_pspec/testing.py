import os
import numpy as np
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
    # read file
    dfile = os.path.join(DATA_PATH, "hera_noise_sim.uv")
    uvd = UVData()
    uvd.read_miriad(dfile)
    # configure data
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    uvd1.vis_units = 'mK'
    uvd2.vis_units = 'mK'
    # configure beam
    if beam is None:
        beam = hp.PSpecBeamUV(os.path.join(DATA_PATH, "NF_HERA_Beams.beamfits"))
    if cosmo is not None:
        beam.cosmo = cosmo
    # create PSpecData
    ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], labels=['n1', 'n2'], beam=beam)
    # get baselines
    if bls is None:
        bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
    # calculate power spectra
    uvp = ds.pspec(bls, bls, (0, 1), input_data_weight=weighting, little_h=True, 
                   verbose=False, history='noise power spectra from hera_pspec/data/hera_noise_sim.uv')

    return ds, uvp


def build_example_uvpspec(beam=None):
    """
    Build an example UVPSpec object, with all necessary metadata.
    """
    uvp = hp.uvpspec.UVPSpec()

    if beam is None:
        beam = hp.PSpecBeamUV(os.path.join(DATA_PATH, "NF_HERA_Beams.beamfits"))

    cosmo = beam.cosmo

    Ntimes = 10
    Nfreqs = 50
    Ndlys = Nfreqs
    Nspws = 1
    Nspwdlys = Nspws * Nfreqs

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
    spws = np.arange(Nspws)
    spw_array = np.tile(spws, Ndlys)
    freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), Nspws)
    dly_array = np.fft.fftshift(np.repeat(np.fft.fftfreq(Nfreqs, np.median(np.diff(freq_array))), Nspws))
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
    #label2 = 'blue' # Leave commented out to make sure non-named UVPSpecs work!
    OmegaP, OmegaPP = beam.get_Omegas(beam.primary_beam.polarization_array[0])
    beam_freqs = beam.beam_freqs

    telescope_location = np.array([5109325.85521063, 
                                   2005235.09142983, 
                                  -3239928.42475397])

    data_array, wgt_array, integration_array, nsample_array = {}, {}, {}, {}
    for s in spws:
        data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) \
                      * blpair_array[:, None, None] / 1e9
        wgt_array[s] = np.ones((Nblpairts, Ndlys, 2, Npols), dtype=np.float)
        integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)
        nsample_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)

    params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nblpairs', 'Nblpairts', 
              'Npols', 'Ndlys', 'Nbls', 'blpair_array', 'time_1_array', 
              'time_2_array', 'lst_1_array', 'lst_2_array', 'spw_array', 
              'dly_array', 'freq_array', 'pol_array', 'data_array', 'wgt_array',
              'integration_array', 'bl_array', 'bl_vecs', 'telescope_location', 
              'vis_units', 'channel_width', 'weighting', 'history', 'taper', 'norm', 
              'git_hash', 'nsample_array', 'time_avg_array', 'lst_avg_array', 
              'cosmo', 'scalar_array', 'label1', 'norm_units', 'OmegaP', 'OmegaPP', 
              'beam_freqs']
    
    # Set all parameters
    for p in params:
        setattr(uvp, p, locals()[p])
    
    return uvp, cosmo


import hera_pspec as hp
from hera_pspec.data import DATA_PATH
import os
from pyuvdata import UVData
import numpy as np


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

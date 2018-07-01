import nose.tools as nt
from hera_pspec.data import DATA_PATH
from hera_pspec import testing, uvpspec, conversions, pspecbeam, utils
import os
from pyuvdata import UVData
import numpy as np


def test_build_vanilla_uvpspec():
    uvp, cosmo = testing.build_vanilla_uvpspec()    
    nt.assert_true(isinstance(uvp, uvpspec.UVPSpec))
    nt.assert_true(isinstance(cosmo, conversions.Cosmo_Conversions))
    nt.assert_equal(uvp.cosmo, cosmo)

    beam = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits'))
    uvp, cosmo = testing.build_vanilla_uvpspec(beam=beam)
    beam_OP = beam.get_Omegas(uvp.pol_array[0])[0]
    nt.assert_equal(beam_OP.tolist(), uvp.OmegaP.tolist())

def test_uvpspec_from_data():
    fname = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd = UVData()
    uvd.read_miriad(fname)
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)

    uvp = testing.uvpspec_from_data(fname, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beam)
    nt.assert_equal(uvp.Nfreqs, 150)
    nt.assert_equal(np.unique(uvp.blpair_array).tolist(), [37038038039, 37038052053, 37038053054, 38039037038,
                                                            38039052053, 38039053054, 52053037038, 52053038039, 
                                                            52053053054, 53054037038, 53054038039, 53054052053])
    uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beamfile)
    uvp.history = ''
    uvp2.history = ''
    nt.assert_equal(uvp, uvp2)


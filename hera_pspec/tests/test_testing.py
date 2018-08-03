import nose.tools as nt
from hera_pspec.data import DATA_PATH
from hera_pspec import testing, uvpspec, conversions, pspecbeam, utils
import os
from pyuvdata import UVData
import numpy as np
from hera_cal import redcal


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
    # get data
    fname = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    fname_std = os.path.join(DATA_PATH, "zen.even.std.xx.LST.1.28828.uvOCRSA")
    uvd = UVData()
    uvd.read_miriad(fname)
    beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
    beam = pspecbeam.PSpecBeamUV(beamfile)

    # test basic execution
    uvp = testing.uvpspec_from_data(fname, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beam, spw_ranges=[(50, 100)])
    nt.assert_equal(uvp.Nfreqs, 50)
    nt.assert_equal(np.unique(uvp.blpair_array).tolist(), [137138138139, 137138152153, 137138153154, 138139152153, 138139153154, 152153153154])
    uvp2 = testing.uvpspec_from_data(uvd, [(37, 38), (38, 39), (52, 53), (53, 54)], beam=beamfile, spw_ranges=[(50, 100)])
    uvp.history = ''
    uvp2.history = ''
    nt.assert_equal(uvp, uvp2)

    # test multiple bl groups
    antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
    reds = redcal.get_pos_reds(dict(zip(ants, antpos)), low_hi=True)
    uvp = testing.uvpspec_from_data(fname, reds[:3], beam=beam, spw_ranges=[(50, 100)])
    nt.assert_equal(len(set(uvp.bl_array) - set([137138, 137151, 137152, 138139, 138152, 138153, 139153, 139154,
                                                 151152, 151167, 152153, 152167, 152168, 153154, 153168, 153169,
                                                 154169, 167168, 168169])), 0)
    nt.assert_equal(uvp.Nblpairs, 51)

    # test exceptions
    nt.assert_raises(AssertionError, testing.uvpspec_from_data, fname, (37, 38))
    nt.assert_raises(AssertionError, testing.uvpspec_from_data, fname, [([37, 38], [38, 39])])
    nt.assert_raises(AssertionError, testing.uvpspec_from_data, fname, [[[37, 38], [38, 39]]])

    # test std
    uvp = testing.uvpspec_from_data(fname, [(37, 38), (38, 39), (52, 53), (53, 54)],
                                    data_std=fname_std, beam=beam, spw_ranges=[(20,28)])


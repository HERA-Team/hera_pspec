"""Tests for loss functions in hera_pspec."""
import os
from hera_pspec import pspecbeam, testing, loss
from hera_pspec.data import DATA_PATH
import numpy as np

class TestApplyBiasCorrection:
    def test_total_bias_notinplace(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        
        # Get rid of all the stats and covariances, to test if it works still
        del uvp.cov_array_real
        del uvp.cov_array_imag
        #del uvp.stats_array

        uvp2 = loss.apply_bias_correction(
            uvp, total_bias={spw: 2 for spw in uvp.spw_array}, inplace=False
        )
        
        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp2.data_array[spw], 2*uvp.data_array[spw])
        
    def test_total_bias_notinplace_covs(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        
        uvp.stats_array = {"P_N": {spw: 1 for spw in uvp.spw_array}}
        
        uvp2 = loss.apply_bias_correction(
            uvp, total_bias={spw: 2 for spw in uvp.spw_array}, inplace=False
        )
        
        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp2.data_array[spw], 2*uvp.data_array[spw])
            np.testing.assert_allclose(uvp2.cov_array_real[spw], 4*uvp.cov_array_real[spw])
            np.testing.assert_allclose(uvp2.cov_array_imag[spw], 4*uvp.cov_array_imag[spw])
            
            for stat in uvp.stats_array:
                np.testing.assert_allclose(uvp.stats_array[stat][spw]*2, uvp2.stats_array[stat][spw])
            
    def test_data_bias_inplace(self):
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.beam = pspecbeam.PSpecBeamUV(beamfile)
        uvp, cosmo = testing.build_vanilla_uvpspec(beam=self.beam)
        
        data = {spw: dd.copy() for spw, dd in uvp.data_array.items()}
        
        loss.apply_bias_correction(
            uvp, data_bias={spw: 2 for spw in uvp.spw_array}, inplace=True
        )
        
        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp.data_array[spw], data[spw]*2)

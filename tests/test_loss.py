"""Tests for loss functions in hera_pspec."""

import copy

import numpy as np

from hera_pspec import UVPSpec, loss


class TestApplyBiasCorrection:
    def test_total_bias_notinplace(self, mutable_uvp_with_beam: UVPSpec) -> None:
        """Check that total_bias scales the data array when stats/cov arrays are absent."""
        # Get rid of all the stats and covariances, to test if it works still
        uvp = copy.deepcopy(mutable_uvp_with_beam)
        del uvp.cov_array_real
        del uvp.cov_array_imag
        # del uvp.stats_array

        uvp2 = loss.apply_bias_correction(
            uvp, total_bias={spw: 2 for spw in uvp.spw_array}, inplace=False
        )

        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp2.data_array[spw], 2 * uvp.data_array[spw])

    def test_total_bias_notinplace_covs(self, mutable_uvp_with_beam: UVPSpec) -> None:
        """Check that total_bias scales data, cov, and stats arrays consistently (bias and bias**2)."""
        uvp = copy.deepcopy(mutable_uvp_with_beam)
        uvp.stats_array = {"P_N": {spw: 1 for spw in uvp.spw_array}}

        uvp2 = loss.apply_bias_correction(
            uvp, total_bias={spw: 2 for spw in uvp.spw_array}, inplace=False
        )

        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp2.data_array[spw], 2 * uvp.data_array[spw])
            np.testing.assert_allclose(
                uvp2.cov_array_real[spw], 4 * uvp.cov_array_real[spw]
            )
            np.testing.assert_allclose(
                uvp2.cov_array_imag[spw], 4 * uvp.cov_array_imag[spw]
            )

            for stat in uvp.stats_array:
                np.testing.assert_allclose(
                    uvp.stats_array[stat][spw] * 2, uvp2.stats_array[stat][spw]
                )

    def test_data_bias_inplace(self, mutable_uvp_with_beam: UVPSpec) -> None:
        """Check that data_bias scales the data array in place."""
        uvp = copy.deepcopy(mutable_uvp_with_beam)
        data = {spw: dd.copy() for spw, dd in uvp.data_array.items()}

        loss.apply_bias_correction(
            uvp, data_bias={spw: 2 for spw in uvp.spw_array}, inplace=True
        )

        for spw in uvp.spw_array:
            np.testing.assert_allclose(uvp.data_array[spw], data[spw] * 2)

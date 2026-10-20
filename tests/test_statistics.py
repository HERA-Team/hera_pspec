"""Physical and statistical checks of the power spectrum estimators.

The tests in this module check that the estimators give the *right numbers*
(correct amplitudes, unbiased means, calibrated errors), not just that the code
runs. They use synthetic visibilities with a known statistical model, generated
by ``hera_pspec.testing.synthetic_uvdata``. Many integrations of a single
baseline are used as independent realisations of the noise, so a single call to
``PSpecData.pspec`` gives thousands of draws, and the tolerances below are set
from the expected statistical scatter of those draws.
"""

import numpy as np
import pytest

from hera_pspec import PSpecBeamGauss, PSpecData


@pytest.fixture(scope="module")
def white_noise_uvd(synthetic_uvd):
    """White complex Gaussian noise (unit variance) on one baseline.

    32 channels x 2000 integrations, so a pspec() call gives 2000 independent
    spectra.
    """
    return synthetic_uvd(nfreq=32, ntimes=2000, antpairs=[(0, 1)], seed=10)


@pytest.fixture(scope="module")
def gauss_beam():
    return PSpecBeamGauss(0.2, np.linspace(140e6, 170e6, 100))


def mean_white_noise_power(uvd, beam, norm, taper):
    """Mean real power of the auto-spectrum of ``uvd`` over all times and delays."""
    ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam)
    uvp = ds.pspec(
        [(0, 1)],
        [(0, 1)],
        dsets=(0, 1),
        pols=[("xx", "xx")],
        norm=norm,
        taper=taper,
        verbose=False,
    )
    return uvp.data_array[0].real.mean()


class TestWhiteNoiseCalibration:
    @pytest.mark.parametrize("norm", ["I", "H^-1"])
    def test_taper_does_not_change_white_noise_power(
        self, white_noise_uvd, gauss_beam, norm
    ) -> None:
        """A taper must not change the calibrated power of white noise.

        White noise has the same power at every delay, so the taper (which only
        redistributes power between neighbouring delays) must leave the mean
        power unchanged once the spectra are normalised. The normalisation
        accounts for the loss of effective bandwidth due to the taper either in
        the normalisation matrix M or in the beam/cosmology scalar, and it must
        do so exactly once.

        The mean is taken over 2000 x 32 (correlated) power values, so its
        relative scatter is below 1%; we allow 3%.
        """
        untapered = mean_white_noise_power(white_noise_uvd, gauss_beam, norm, "none")
        tapered = mean_white_noise_power(white_noise_uvd, gauss_beam, norm, "bh")
        np.testing.assert_allclose(tapered / untapered, 1.0, atol=0.03)

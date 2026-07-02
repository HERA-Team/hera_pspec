from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyuvdata import UVBeam

from hera_pspec import conversions, pspecbeam
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Frequency range/resolution shared by compute_pspec_scalar calls across all beam types.
LOWER_FREQ = 120.0 * 10**6
UPPER_FREQ = 128.0 * 10**6
NUM_FREQS = 20


@pytest.fixture
def efield_beam() -> pspecbeam.PSpecBeamUV:
    """An efield-type beam with beam_type force-set to 'power' so later validation steps in beam_normalized_response can be reached."""
    beam = pspecbeam.PSpecBeamUV(DATA_PATH / "HERA_NF_efield.beamfits")
    beam.primary_beam.beam_type = "power"
    return beam


class TestPSpecBeamUV:
    def test_init(self) -> None:
        """Check that PSpecBeamUV loads a beamfits file and sets primary_beam/beam_freqs/cosmo."""
        bm = pspecbeam.PSpecBeamUV(DATA_PATH / "HERA_NF_dipole_power.beamfits")
        assert isinstance(bm.primary_beam, UVBeam)
        assert bm.beam_freqs.size > 0
        assert isinstance(bm.cosmo, conversions.Cosmo_Conversions)

    def test_omegas(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check power_beam_int/power_beam_sq_int against precomputed reference values."""
        # Precomputed results were done "by hand" using iPython notebook
        # "Scalar_dev2.ipynb" in tests directory.
        Om_p = beam_nf_pstokes.power_beam_int()
        Om_pp = beam_nf_pstokes.power_beam_sq_int()

        np.testing.assert_almost_equal(Om_p[0], 0.080082680885906782)
        np.testing.assert_almost_equal(Om_p[18], 0.031990943334017245)
        np.testing.assert_almost_equal(Om_p[-1], 0.03100215028171072)

        np.testing.assert_almost_equal(Om_pp[0], 0.036391945229980432)
        np.testing.assert_almost_equal(Om_pp[15], 0.018159280192894631)
        np.testing.assert_almost_equal(Om_pp[-1], 0.014528100116719534)

        assert Om_p.ndim == 1
        assert Om_pp.ndim == 1

    def test_scalar_pI(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check compute_pspec_scalar for pI against a precomputed reference value."""
        scalar = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        assert abs(scalar / 568847837.72586381 - 1.0) <= 1e-4

    def test_scalar_convergence(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check that compute_pspec_scalar converges as num_steps increases."""
        scalar = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        scalar_large_Nsteps = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=10000
        )
        assert abs(scalar / scalar_large_Nsteps - 1.0) <= 1e-5

    def test_custom_cosmology(self) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        beam = pspecbeam.PSpecBeamUV(
            DATA_PATH / "HERA_NF_pstokes_power.beamfits",
            cosmo=conversions.Cosmo_Conversions(),
        )
        assert isinstance(beam.cosmo, conversions.Cosmo_Conversions)

    @pytest.mark.parametrize("pol", ["pQ", "pU", "pV"])
    def test_scalar_other_stokes(
        self, beam_nf_pstokes: pspecbeam.PSpecBeamUV, pol: str
    ) -> None:
        """Check that compute_pspec_scalar does not raise for the other Stokes parameters."""
        beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol=pol, num_steps=2000
        )

    def test_taper(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check compute_pspec_scalar with a taper against a precomputed reference value."""
        scalar = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, num_steps=5000, taper="blackman"
        )
        assert abs(scalar / 1989353792.1765163 - 1.0) <= 1e-8

    def test_jy_to_mk(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check Jy_to_mK output shape/value and that non-float-array inputs raise a TypeError."""
        M = beam_nf_pstokes.Jy_to_mK(np.linspace(100e6, 200e6, 11))
        assert len(M) == 11
        np.testing.assert_almost_equal(M[0], 40.643366654821904)
        M = beam_nf_pstokes.Jy_to_mK(150e6)
        assert isinstance(M, np.ndarray)
        beam_nf_pstokes.Jy_to_mK(np.linspace(90, 210e6, 11))

        with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
            beam_nf_pstokes.Jy_to_mK([1])
        with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
            beam_nf_pstokes.Jy_to_mK(np.array([1]))

    def test_noise_scalar(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check compute_pspec_scalar with noise_scalar=True against a precomputed reference value."""
        sclr = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ,
            UPPER_FREQ,
            NUM_FREQS,
            pol="pI",
            num_steps=2000,
            noise_scalar=True,
        )
        np.testing.assert_almost_equal(sclr, 71.105979715733)

    def test_unrecognised_pol_raises(
        self, beam_nf_pstokes: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a completely unrecognised pol string raises KeyError from pyuvdata."""
        with pytest.raises(KeyError):
            beam_nf_pstokes.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pZ"
            )

    def test_wrong_basis_pol_raises(
        self, beam_nf_pstokes: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a linear-pol string raises ValueError on a Stokes beam."""
        with pytest.raises(
            ValueError, match="Do not have the right polarization information"
        ):
            beam_nf_pstokes.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="XX"
            )

    def test_dipole_beam(self, beam_nf_dipole: pspecbeam.PSpecBeamUV) -> None:
        """Check that dipole (linear-pol) beams compute a scalar for XX but raise for pI."""
        beam_nf_dipole.compute_pspec_scalar(LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="XX")
        with pytest.raises(
            ValueError, match="Do not have the right polarization information"
        ):
            beam_nf_dipole.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI"
            )

    def test_efield_beam(self) -> None:
        """Check that efield beams compute a scalar for XX but raise for pI."""
        beam = pspecbeam.PSpecBeamUV(DATA_PATH / "HERA_NF_efield.beamfits")
        beam.compute_pspec_scalar(LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="XX")
        with pytest.raises(
            ValueError, match="Do not have the right polarization information"
        ):
            beam.compute_pspec_scalar(LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI")

    def test_get_omegas_single_polpair(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check the OmegaP/OmegaPP shape for a single polarization-pair tuple."""
        OP, OPP = beam_nf_dipole.get_Omegas(("xx", "xx"))
        assert OP.shape == (26, 1)
        assert OPP.shape == (26, 1)

    def test_get_omegas_multiple_polpairs(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check the OmegaP/OmegaPP shape for a list of polarization-pair codes."""
        OP, OPP = beam_nf_dipole.get_Omegas([(-5, -5), (-6, -6)])
        assert OP.shape == (26, 2)
        assert OPP.shape == (26, 2)

    def test_get_omegas_raises_on_non_list_input(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a bare (non-list) polpair raises a TypeError."""
        with pytest.raises(TypeError, match="polpairs is not a list"):
            beam_nf_dipole.get_Omegas("xx")

    def test_get_omegas_raises_on_cross_correlation(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a cross-Stokes polpair raises a NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="does not support cross-correlation"
        ):
            beam_nf_dipole.get_Omegas([("pI", "pQ")])

    def test_beam_normalized_response_basic(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check the shape of beam_normalized_response's output for a dipole power beam."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        nside = beam_nf_dipole.primary_beam.nside
        beam_res = pspecbeam.PSpecBeamUV.beam_normalized_response(
            beam_nf_dipole, pol="xx", freq=freq
        )
        assert len(beam_res[1]) == len(freq)
        assert beam_res[0].ndim == 2
        assert np.shape(beam_res[0]) == (len(freq), (12 * nside**2))

    def test_beam_normalized_response_raises_on_wrong_pol(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that an unavailable polarization raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        with pytest.raises(ValueError, match="Do not have the right polarization"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(
                beam_nf_dipole, pol="ll", freq=freq
            )

    def test_beam_normalized_response_raises_on_non_power_beam(self) -> None:
        """Check that a non-power beam_type raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        beam = pspecbeam.PSpecBeamUV(DATA_PATH / "HERA_NF_efield.beamfits")
        beam.primary_beam.beam_type = "voltage"
        with pytest.raises(ValueError, match="beam_type must be power"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol="xx", freq=freq)

    def test_beam_normalized_response_raises_on_wrong_axes(
        self, efield_beam: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that Naxes_vec > 1 raises a ValueError for a power beam."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        efield_beam.primary_beam.Naxes_vec = 2
        with pytest.raises(ValueError, match="Expect scalar for power beam"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(
                efield_beam, pol="xx", freq=freq
            )

    def test_beam_normalized_response_raises_on_non_peak_normalization(
        self, efield_beam: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a non-peak-normalized beam raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        efield_beam.primary_beam._data_normalization.value = "area"
        with pytest.raises(ValueError, match="beam must be peak normalized"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(
                efield_beam, pol="xx", freq=freq
            )

    def test_beam_normalized_response_raises_on_non_healpix_coords(
        self, efield_beam: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that a non-healpix pixel coordinate system raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        efield_beam.primary_beam.pixel_coordinate_system = "cartesian"
        with pytest.raises(ValueError, match="only healpix format supported"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(
                efield_beam, pol="xx", freq=freq
            )


@pytest.fixture(scope="module")
def gauss_beam() -> pspecbeam.PSpecBeamGauss:
    """A Gaussian primary beam (FWHM=0.8), used both as its own test subject and as a reference for PSpecBeamFromArray."""
    return pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))


class TestPSpecBeamGauss:
    def test_omegas(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check power_beam_int/power_beam_sq_int against reference values and that they're frequency-independent."""
        Om_p = gauss_beam.power_beam_int()
        Om_pp = gauss_beam.power_beam_sq_int()
        assert Om_p.ndim == 1
        assert Om_pp.ndim == 1
        assert Om_p[4] == 0.7251776226923511
        assert Om_p[4] == Om_p[0]
        assert Om_pp[4] == 0.36258881134617554
        assert Om_pp[4] == Om_pp[0]

    def test_scalar(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check compute_pspec_scalar against a precomputed reference value."""
        scalar = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        assert abs(scalar / 6392750120.8657961 - 1.0) <= 1e-4

    def test_custom_cosmology(self) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        bgauss = pspecbeam.PSpecBeamGauss(
            0.8,
            np.linspace(115e6, 130e6, 50, endpoint=False),
            cosmo=conversions.Cosmo_Conversions(),
        )
        assert isinstance(bgauss.cosmo, conversions.Cosmo_Conversions)

    def test_scalar_convergence(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that compute_pspec_scalar converges as num_steps increases."""
        scalar = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        scalar_large_Nsteps = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, num_steps=5000
        )
        assert abs(scalar / scalar_large_Nsteps - 1.0) <= 1e-5

    def test_taper(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check compute_pspec_scalar with a taper against a precomputed reference value."""
        scalar = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, num_steps=5000, taper="blackman"
        )
        assert abs(scalar / 22123832163.072491 - 1.0) <= 1e-8


class TestPSpecBeamFromArray:
    def test_construction(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check construction from array or dict OmegaP/OmegaPP, dict-specified pol recognition, and repr."""
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=gauss_beam.beam_freqs
        )
        psbeampol = pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P, "pQ": Om_P},
            OmegaPP={"pI": Om_PP, "pQ": Om_PP},
            beam_freqs=gauss_beam.beam_freqs,
        )
        assert len(str(psbeam)) > 0
        # polarizations are recognized for the dict-specified beam
        scalar = psbeampol.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pQ", num_steps=2000
        )
        assert abs(scalar / 6392750120.8657961 - 1.0) <= 1e-4

    def test_custom_cosmology(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        bm2 = pspecbeam.PSpecBeamFromArray(
            OmegaP=gauss_beam.power_beam_int(),
            OmegaPP=gauss_beam.power_beam_sq_int(),
            beam_freqs=gauss_beam.beam_freqs,
            cosmo=conversions.Cosmo_Conversions(),
        )
        assert isinstance(bm2.cosmo, conversions.Cosmo_Conversions)

    def test_matches_gauss(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that compute_pspec_scalar matches the equivalent Gaussian beam's result."""
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=gauss_beam.power_beam_int(),
            OmegaPP=gauss_beam.power_beam_sq_int(),
            beam_freqs=gauss_beam.beam_freqs,
        )
        scalar = psbeam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        g_scalar = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        np.testing.assert_array_almost_equal(scalar, g_scalar)

    def test_taper(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that compute_pspec_scalar with a taper matches the equivalent Gaussian-beam result."""
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=gauss_beam.power_beam_int(),
            OmegaPP=gauss_beam.power_beam_sq_int(),
            beam_freqs=gauss_beam.beam_freqs,
        )
        scalar = psbeam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, num_steps=5000, taper="blackman"
        )
        assert abs(scalar / 22123832163.072491 - 1.0) <= 1e-8

    @pytest.mark.parametrize(
        "build_kwargs,exc_type,match",
        [
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP=Om_P, OmegaPP={"pI": Om_PP}, beam_freqs=freqs
                ),
                TypeError,
                "must both be either dicts or arrays",
            ),
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP={"pI": Om_P, "pQ": Om_P},
                    OmegaPP={"pI": Om_PP},
                    beam_freqs=freqs,
                ),
                KeyError,
                "must be specified for both",
            ),
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP={"A": Om_P}, OmegaPP={"A": Om_PP}, beam_freqs=freqs
                ),
                KeyError,
                "'a'",
            ),
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP={"pI": Om_P}, OmegaPP={"pI": "string"}, beam_freqs=freqs
                ),
                TypeError,
                "must both be array_like",
            ),
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP={"pI": Om_P}, OmegaPP={"pI": Om_PP[:-2]}, beam_freqs=freqs
                ),
                ValueError,
                "same shape as beam_freqs",
            ),
            (
                lambda Om_P, Om_PP, freqs: dict(
                    OmegaP=Om_P, OmegaPP={"pI": Om_PP[:-2]}, beam_freqs=freqs
                ),
                TypeError,
                "must both be either dicts or arrays",
            ),
        ],
        ids=[
            "array_omegap_dict_omegapp",
            "missing_pol_in_omegapp",
            "unrecognized_pol_key",
            "omegapp_not_array_like",
            "omegapp_wrong_shape",
            "array_omegap_dict_omegapp_wrong_shape",
        ],
    )
    def test_raises_on_invalid_init_args(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        build_kwargs: Callable[[np.ndarray, np.ndarray, np.ndarray], dict],
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Check that malformed OmegaP/OmegaPP combinations raise the documented error type and message."""
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        kwargs = build_kwargs(Om_P, Om_PP, gauss_beam.beam_freqs)
        with pytest.raises(exc_type, match=match):
            pspecbeam.PSpecBeamFromArray(**kwargs)

    @pytest.mark.parametrize(
        "build_kwargs",
        [
            lambda Om_P, Om_PP, freqs: dict(
                OmegaP={"foo": Om_P}, OmegaPP={"pI": Om_PP}, beam_freqs=freqs
            ),
            lambda Om_P, Om_PP, freqs: dict(
                OmegaP={"pI": Om_P}, OmegaPP={"foo": Om_PP}, beam_freqs=freqs
            ),
        ],
        ids=["bad_omegap_key", "bad_omegapp_key"],
    )
    def test_raises_on_unrecognized_pol_in_init(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        build_kwargs: Callable[[np.ndarray, np.ndarray, np.ndarray], dict],
    ) -> None:
        """Check that an unrecognized polarization key in OmegaP/OmegaPP raises."""
        # No match= since pyuvdata's polstr2num raises before pspecbeam's own
        # "Unrecognized polarization" check; exception type/message varies by version.
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        kwargs = build_kwargs(Om_P, Om_PP, gauss_beam.beam_freqs)
        with pytest.raises((KeyError, ValueError)):
            pspecbeam.PSpecBeamFromArray(**kwargs)

    @pytest.mark.parametrize(
        "call",
        [
            lambda psbeam, Om_P, Om_PP: psbeam.add_pol("foo", Om_P, Om_PP),
            lambda psbeam, Om_P, Om_PP: psbeam.power_beam_int("foo"),
            lambda psbeam, Om_P, Om_PP: psbeam.power_beam_sq_int("foo"),
            lambda psbeam, Om_P, Om_PP: psbeam.power_beam_int(pol="blah"),
            lambda psbeam, Om_P, Om_PP: psbeam.power_beam_sq_int(pol="blah"),
            lambda psbeam, Om_P, Om_PP: psbeam.add_pol(
                pol="A", OmegaP=Om_P, OmegaPP=Om_PP
            ),
        ],
        ids=[
            "add_pol_positional_foo",
            "power_beam_int_foo",
            "power_beam_sq_int_foo",
            "power_beam_int_kw_blah",
            "power_beam_sq_int_kw_blah",
            "add_pol_kw_A",
        ],
    )
    def test_raises_on_invalid_pol_method_arg(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        call: Callable[[pspecbeam.PSpecBeamFromArray, np.ndarray, np.ndarray], Any],
    ) -> None:
        """Check that invalid polarization strings/codes passed to method calls raise."""
        # No match= on these: same pyuvdata version sensitivity as the init-arg cases above.
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=gauss_beam.beam_freqs
        )
        with pytest.raises((KeyError, ValueError)):
            call(psbeam, Om_P, Om_PP)


class TestPSpecBeamBase:
    def test_default_construction(self) -> None:
        """Check that the base class can be instantiated with a default cosmology."""
        bm1 = pspecbeam.PSpecBeamBase()
        assert isinstance(bm1.cosmo, conversions.Cosmo_Conversions)

    def test_custom_cosmology(self) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        bm2 = pspecbeam.PSpecBeamBase(cosmo=conversions.Cosmo_Conversions())
        assert isinstance(bm2.cosmo, conversions.Cosmo_Conversions)

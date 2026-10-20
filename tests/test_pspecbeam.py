import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FLRW, LambdaCDM
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AiryBeam, GaussianBeam

from hera_pspec import conversions, pspecbeam
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Frequency range/resolution shared by compute_pspec_scalar calls across all beam types.
LOWER_FREQ = 120.0 * 10**6 * u.Hz
UPPER_FREQ = 128.0 * 10**6 * u.Hz
NUM_FREQS = 20


@pytest.fixture(scope="module")
def beam_nf_pstokes_legacy(
    beam_nf_pstokes: pspecbeam.PSpecBeamUV,
) -> pspecbeam.PSpecBeamUV:
    """beam_nf_pstokes with the v0.x default cosmology, for the v0.x reference scalars."""
    beam = copy.deepcopy(beam_nf_pstokes)
    beam.cosmo = conversions.LEGACY_COSMOLOGY
    return beam


@pytest.fixture
def efield_beam(beam_nf_efield: pspecbeam.PSpecBeamUV) -> pspecbeam.PSpecBeamUV:
    """An efield-type beam with beam_type force-set to 'power' so later validation steps in beam_normalized_response can be reached."""
    beam = copy.deepcopy(beam_nf_efield)
    beam.primary_beam.beam_type = "power"
    return beam


def _leave_unchanged(primary_beam: UVBeam) -> None:
    pass


# A non-default cosmology for the constructor tests.
CUSTOM_COSMO = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7, Ob0=0.05)


class TestPSpecBeamUV:
    def test_init(self, beam_nf_dipole: pspecbeam.PSpecBeamUV) -> None:
        """Check that PSpecBeamUV loads a beamfits file and sets primary_beam/beam_freqs/cosmo."""
        assert isinstance(beam_nf_dipole.primary_beam, UVBeam)
        assert beam_nf_dipole.beam_freqs.size > 0
        assert beam_nf_dipole.cosmo is conversions.DEFAULT_COSMOLOGY

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

    def test_scalar_pI(self, beam_nf_pstokes_legacy: pspecbeam.PSpecBeamUV) -> None:
        """Check compute_pspec_scalar for pI against a precomputed reference value."""
        scalar = beam_nf_pstokes_legacy.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        assert abs(scalar / 568847837.72586381 - 1.0) <= 1e-4

    @pytest.mark.parametrize("pol", ["pQ", "pU", "pV"])
    def test_scalar_other_stokes(
        self, beam_nf_pstokes: pspecbeam.PSpecBeamUV, pol: str
    ) -> None:
        """Check compute_pspec_scalar for the other Stokes parameters."""
        scalar = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol=pol, num_steps=2000
        )
        scalar_pI = beam_nf_pstokes.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        # a positive scalar that uses the requested polarization's beam
        assert np.isfinite(scalar) and scalar > 0
        assert not np.isclose(scalar, scalar_pI, rtol=1e-3)

    def test_jy_to_mk(self, beam_nf_pstokes: pspecbeam.PSpecBeamUV) -> None:
        """Check Jy_to_mK output, extrapolation warnings and that non-float-array inputs raise a TypeError."""
        M = beam_nf_pstokes.Jy_to_mK(np.linspace(100e6, 200e6, 11) * u.Hz)
        assert len(M) == 11
        # k_B from astropy.constants (CODATA 2018); CODATA 2014 gave 40.643366654821904
        np.testing.assert_almost_equal(M[0], 40.643352524643994)
        M = beam_nf_pstokes.Jy_to_mK(150e6 * u.Hz)
        assert isinstance(M, np.ndarray)

        # Frequencies outside the beam's frequency range are extrapolated, with
        # a warning for each side.
        with (
            pytest.warns(UserWarning, match="min freq .* extrapolating"),
            pytest.warns(UserWarning, match="max freq .* extrapolating"),
        ):
            M = beam_nf_pstokes.Jy_to_mK(np.linspace(90e6, 210e6, 13) * u.Hz)
        assert np.all(np.isfinite(M))

        with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
            beam_nf_pstokes.Jy_to_mK([1])
        with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
            beam_nf_pstokes.Jy_to_mK(np.array([1]))

    def test_noise_scalar(self, beam_nf_pstokes_legacy: pspecbeam.PSpecBeamUV) -> None:
        """Check compute_pspec_scalar with noise_scalar=True against a precomputed reference value."""
        sclr = beam_nf_pstokes_legacy.compute_pspec_scalar(
            LOWER_FREQ,
            UPPER_FREQ,
            NUM_FREQS,
            pol="pI",
            num_steps=2000,
            noise_scalar=True,
        )
        np.testing.assert_almost_equal(sclr, 71.105979715733)

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

    @pytest.mark.parametrize("beam_name", ["beam_nf_dipole", "beam_nf_efield"])
    def test_linear_pol_beam(
        self, beam_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Check that linear-pol (dipole power and efield) beams compute a scalar for XX but raise for pI."""
        beam = request.getfixturevalue(beam_name)
        scalar = beam.compute_pspec_scalar(LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="XX")
        assert np.isfinite(scalar) and scalar > 0
        # pol strings are case-insensitive
        assert scalar == beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="xx"
        )
        with pytest.raises(
            ValueError, match="Do not have the right polarization information"
        ):
            beam.compute_pspec_scalar(LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI")

    @pytest.mark.parametrize(
        "polpairs,ncols",
        [(("xx", "xx"), 1), ([(-5, -5), (-6, -6)], 2)],
        ids=["single_polpair_tuple", "list_of_polpair_codes"],
    )
    def test_get_omegas(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV, polpairs, ncols: int
    ) -> None:
        """Check the OmegaP/OmegaPP shape (Nbeam_freqs, Npolpairs) from get_Omegas."""
        OP, OPP = beam_nf_dipole.get_Omegas(polpairs)
        assert OP.shape == (26, ncols)
        assert OPP.shape == (26, ncols)

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
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        nside = beam_nf_dipole.primary_beam.nside
        beam_res = pspecbeam.PSpecBeamUV.beam_normalized_response(
            beam_nf_dipole, pol="xx", freq=freq
        )
        assert len(beam_res[1]) == len(freq)
        assert beam_res[0].ndim == 2
        assert np.shape(beam_res[0]) == (len(freq), (12 * nside**2))

    @pytest.mark.parametrize(
        "beam_name,mutate,pol,match",
        [
            pytest.param(
                "beam_nf_dipole",
                _leave_unchanged,
                "ll",
                "Do not have the right polarization",
                id="unavailable_pol",
            ),
            pytest.param(
                "efield_beam",
                lambda pb: setattr(pb, "beam_type", "voltage"),
                "xx",
                "beam_type must be power",
                id="non_power_beam",
            ),
            pytest.param(
                "efield_beam",
                lambda pb: setattr(pb, "Naxes_vec", 2),
                "xx",
                "Expect scalar for power beam",
                id="vector_power_beam",
            ),
            pytest.param(
                "efield_beam",
                lambda pb: setattr(pb._data_normalization, "value", "area"),
                "xx",
                "beam must be peak normalized",
                id="non_peak_normalized",
            ),
            pytest.param(
                "efield_beam",
                lambda pb: setattr(pb, "pixel_coordinate_system", "cartesian"),
                "xx",
                "only healpix format supported",
                id="non_healpix_coords",
            ),
        ],
    )
    def test_beam_normalized_response_raises(
        self,
        beam_name: str,
        mutate: Callable[[UVBeam], None],
        pol: str,
        match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Check that beams beam_normalized_response cannot handle raise a ValueError."""
        beam = request.getfixturevalue(beam_name)
        mutate(beam.primary_beam)
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        with pytest.raises(ValueError, match=match):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol=pol, freq=freq)


@pytest.fixture(scope="module")
def gauss_beam() -> pspecbeam.PSpecBeamGauss:
    """A Gaussian primary beam (FWHM=0.8), used both as its own test subject and as a reference for PSpecBeamFromArray."""
    return pspecbeam.PSpecBeamGauss(
        0.8 * u.rad, np.linspace(115e6, 130e6, 50, endpoint=False) * u.Hz
    )


@pytest.fixture(scope="module")
def gauss_beam_legacy() -> pspecbeam.PSpecBeamGauss:
    """gauss_beam with the v0.x default cosmology, for the v0.x reference scalars."""
    return pspecbeam.PSpecBeamGauss(
        0.8 * u.rad,
        np.linspace(115e6, 130e6, 50, endpoint=False) * u.Hz,
        cosmo=conversions.LEGACY_COSMOLOGY,
    )


@pytest.fixture(scope="module")
def fromarray_beam_legacy(
    gauss_beam_legacy: pspecbeam.PSpecBeamGauss,
) -> pspecbeam.PSpecBeamFromArray:
    """A PSpecBeamFromArray built from gauss_beam_legacy's beam integrals and cosmology."""
    return pspecbeam.PSpecBeamFromArray(
        OmegaP=gauss_beam_legacy.power_beam_int() * u.sr,
        OmegaPP=gauss_beam_legacy.power_beam_sq_int() * u.sr,
        beam_freqs=gauss_beam_legacy.beam_freqs * u.Hz,
        cosmo=conversions.LEGACY_COSMOLOGY,
    )


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

    def test_scalar(self, gauss_beam_legacy: pspecbeam.PSpecBeamGauss) -> None:
        """Check compute_pspec_scalar against a precomputed reference value."""
        scalar = gauss_beam_legacy.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        assert abs(scalar / 6392750120.8657961 - 1.0) <= 1e-4

    def test_from_gaussian_beam_object(
        self, gauss_beam: pspecbeam.PSpecBeamGauss
    ) -> None:
        """Check that wrapping a GaussianBeam gives the same fwhm/Omegas as the from-fwhm constructor."""
        fwhm = 0.8
        analytic = GaussianBeam(
            sigma=fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))), sigma_type="power"
        )
        bgauss = pspecbeam.PSpecBeamGauss(
            beam=analytic, beam_freqs=gauss_beam.beam_freqs * u.Hz
        )
        assert bgauss.analytic_beam is analytic
        assert isinstance(gauss_beam.analytic_beam, GaussianBeam)
        np.testing.assert_allclose(bgauss.fwhm, gauss_beam.fwhm)
        np.testing.assert_allclose(bgauss.power_beam_int(), gauss_beam.power_beam_int())
        np.testing.assert_allclose(
            bgauss.power_beam_sq_int(), gauss_beam.power_beam_sq_int()
        )

    def test_sigma_type_equivalence(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that power- and efield-specified GaussianBeams of matching width wrap identically."""
        sigma_power = 0.8 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        from_power = pspecbeam.PSpecBeamGauss(
            beam=GaussianBeam(sigma=sigma_power, sigma_type="power"),
            beam_freqs=gauss_beam.beam_freqs * u.Hz,
        )
        from_efield = pspecbeam.PSpecBeamGauss(
            beam=GaussianBeam(sigma=np.sqrt(2.0) * sigma_power, sigma_type="efield"),
            beam_freqs=gauss_beam.beam_freqs * u.Hz,
        )
        np.testing.assert_allclose(from_power.fwhm, from_efield.fwhm)
        np.testing.assert_allclose(
            from_power.power_beam_int(), from_efield.power_beam_int()
        )
        np.testing.assert_allclose(
            from_power.power_beam_sq_int(), from_efield.power_beam_sq_int()
        )

    def test_raises_on_unsupported_analytic_beams(
        self, gauss_beam: pspecbeam.PSpecBeamGauss
    ) -> None:
        """Check clear errors for diameter-specified, chromatic, and non-Gaussian analytic beams."""
        with pytest.raises(ValueError, match="Diameter-specified"):
            pspecbeam.PSpecBeamGauss(
                beam=GaussianBeam(diameter=14.0),
                beam_freqs=gauss_beam.beam_freqs * u.Hz,
            )
        with pytest.raises(NotImplementedError, match="Chromatic"):
            pspecbeam.PSpecBeamGauss(
                beam=GaussianBeam(
                    sigma=0.1, spectral_index=-1.0, reference_frequency=150e6
                ),
                beam_freqs=gauss_beam.beam_freqs * u.Hz,
            )
        with pytest.raises(TypeError, match="must be a pyuvdata GaussianBeam"):
            pspecbeam.PSpecBeamGauss(
                beam=AiryBeam(diameter=14.0), beam_freqs=gauss_beam.beam_freqs * u.Hz
            )

    def test_raises_on_bad_argument_combinations(
        self, gauss_beam: pspecbeam.PSpecBeamGauss
    ) -> None:
        """Check that fwhm and beam are mutually exclusive and beam_freqs is required."""
        analytic = GaussianBeam(sigma=0.1)
        with pytest.raises(ValueError, match="Exactly one of fwhm or beam"):
            pspecbeam.PSpecBeamGauss(
                fwhm=0.8 * u.rad, beam=analytic, beam_freqs=gauss_beam.beam_freqs * u.Hz
            )
        with pytest.raises(ValueError, match="Exactly one of fwhm or beam"):
            pspecbeam.PSpecBeamGauss(beam_freqs=gauss_beam.beam_freqs * u.Hz)
        with pytest.raises(ValueError, match="beam_freqs must be specified"):
            pspecbeam.PSpecBeamGauss(fwhm=0.8 * u.rad)
        with pytest.raises(TypeError, match="pass analytic beams via the beam keyword"):
            pspecbeam.PSpecBeamGauss(analytic, gauss_beam.beam_freqs * u.Hz)

    def test_omegas_match_numerical_integration(self) -> None:
        """Check the closed-form Omegas against healpix-integrated areas of the wrapped GaussianBeam.

        The closed forms are flat-sky results, so agreement is only expected
        to ~0.5% for a narrow beam; they remain the authoritative values.
        """
        fwhm = 0.1
        beam_freqs = np.array([150e6])
        bgauss = pspecbeam.PSpecBeamGauss(fwhm * u.rad, beam_freqs * u.Hz)
        nside = 128
        uvb = bgauss.analytic_beam.to_uvbeam(
            freq_array=beam_freqs,
            beam_type="power",
            pixel_coordinate_system="healpix",
            nside=nside,
        )
        pixel_area = 4.0 * np.pi / (12 * nside**2)
        beam_map = uvb.data_array[0, 0, 0]
        np.testing.assert_allclose(
            beam_map.sum() * pixel_area, bgauss.power_beam_int()[0], rtol=1e-2
        )
        np.testing.assert_allclose(
            (beam_map**2).sum() * pixel_area, bgauss.power_beam_sq_int()[0], rtol=1e-2
        )


class TestPSpecBeamFromArray:
    def test_construction(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check construction from array or dict OmegaP/OmegaPP, dict-specified pol recognition, and repr."""
        Om_P = gauss_beam.power_beam_int() * u.sr
        Om_PP = gauss_beam.power_beam_sq_int() * u.sr
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=gauss_beam.beam_freqs * u.Hz
        )
        psbeampol = pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P, "pQ": Om_P},
            OmegaPP={"pI": Om_PP, "pQ": Om_PP},
            beam_freqs=gauss_beam.beam_freqs * u.Hz,
            # the reference scalar below is from v0.x, with its default cosmology
            cosmo=conversions.LEGACY_COSMOLOGY,
        )
        assert len(str(psbeam)) > 0
        # polarizations are recognized for the dict-specified beam
        scalar = psbeampol.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pQ", num_steps=2000
        )
        assert abs(scalar / 6392750120.8657961 - 1.0) <= 1e-4

    def test_matches_gauss(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check that compute_pspec_scalar matches the equivalent Gaussian beam's result."""
        psbeam = pspecbeam.PSpecBeamFromArray(
            OmegaP=gauss_beam.power_beam_int() * u.sr,
            OmegaPP=gauss_beam.power_beam_sq_int() * u.sr,
            beam_freqs=gauss_beam.beam_freqs * u.Hz,
        )
        scalar = psbeam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        g_scalar = gauss_beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        np.testing.assert_array_almost_equal(scalar, g_scalar)

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
        Om_P = gauss_beam.power_beam_int() * u.sr
        Om_PP = gauss_beam.power_beam_sq_int() * u.sr
        kwargs = build_kwargs(Om_P, Om_PP, gauss_beam.beam_freqs * u.Hz)
        with pytest.raises(exc_type, match=match):
            pspecbeam.PSpecBeamFromArray(**kwargs)

    @pytest.mark.parametrize(
        "call,match",
        [
            (
                lambda Om_P, Om_PP, freqs: pspecbeam.PSpecBeamFromArray(
                    OmegaP={7: Om_P}, OmegaPP={"pI": Om_PP}, beam_freqs=freqs
                ),
                "Unrecognized polarization '7' in OmegaP.",
            ),
            (
                lambda Om_P, Om_PP, freqs: pspecbeam.PSpecBeamFromArray(
                    OmegaP={"pI": Om_P}, OmegaPP={7: Om_PP}, beam_freqs=freqs
                ),
                "Unrecognized polarization '7' in OmegaPP.",
            ),
            (
                lambda Om_P, Om_PP, freqs: pspecbeam.PSpecBeamFromArray(
                    OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=freqs
                ).add_pol("rr", Om_P, Om_PP),
                "Polarization '-1' is not valid.",
            ),
            (
                lambda Om_P, Om_PP, freqs: pspecbeam.PSpecBeamFromArray(
                    OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=freqs
                ).power_beam_int("pQ"),
                "OmegaP not specified for polarization",
            ),
            (
                lambda Om_P, Om_PP, freqs: pspecbeam.PSpecBeamFromArray(
                    OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=freqs
                ).power_beam_sq_int("pQ"),
                "OmegaPP not specified for polarization",
            ),
        ],
        ids=[
            "init_omegap_int_key",
            "init_omegapp_int_key",
            "add_pol_circular_pol",
            "power_beam_int_missing_pol",
            "power_beam_sq_int_missing_pol",
        ],
    )
    def test_raises_on_unavailable_pol(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        call: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
        match: str,
    ) -> None:
        """Check that polarizations that are not allowed, or have no beam
        integrals stored, raise KeyError with an informative message."""
        # Unknown pol *strings* are rejected earlier, by pyuvdata's polstr2num,
        # so these cases use integer codes, a circular pol ("rr", code -1) that
        # the beam does not support, or valid pols that were never added.
        Om_P = gauss_beam.power_beam_int() * u.sr
        Om_PP = gauss_beam.power_beam_sq_int() * u.sr
        with pytest.raises(KeyError, match=match):
            call(Om_P, Om_PP, gauss_beam.beam_freqs * u.Hz)


class TestPSpecBeamBase:
    def test_default_construction(self) -> None:
        """Check that the base class can be instantiated with a default cosmology."""
        bm1 = pspecbeam.PSpecBeamBase()
        assert bm1.cosmo is conversions.DEFAULT_COSMOLOGY


def _beam_from_constructor(name: str, cosmo):
    """Build one of the PSpecBeam classes with the given cosmology."""
    freqs = np.linspace(115e6, 130e6, 50, endpoint=False)
    if name == "base":
        return pspecbeam.PSpecBeamBase(cosmo=cosmo)
    if name == "uv":
        return pspecbeam.PSpecBeamUV(
            DATA_PATH / "HERA_NF_dipole_power.beamfits", cosmo=cosmo
        )
    if name == "gauss":
        return pspecbeam.PSpecBeamGauss(0.8 * u.rad, freqs * u.Hz, cosmo=cosmo)
    gauss = pspecbeam.PSpecBeamGauss(0.8 * u.rad, freqs * u.Hz)
    return pspecbeam.PSpecBeamFromArray(
        OmegaP=gauss.power_beam_int() * u.sr,
        OmegaPP=gauss.power_beam_sq_int() * u.sr,
        beam_freqs=freqs * u.Hz,
        cosmo=cosmo,
    )


BEAM_CONSTRUCTORS = ["base", "uv", "gauss", "from_array"]


class TestBeamCosmology:
    @pytest.mark.parametrize("name", BEAM_CONSTRUCTORS)
    def test_default_is_shared_default_cosmology(self, name: str) -> None:
        """Every beam class defaults to the one DEFAULT_COSMOLOGY object."""
        beam = _beam_from_constructor(name, None)
        assert beam.cosmo is conversions.DEFAULT_COSMOLOGY

    @pytest.mark.parametrize("name", BEAM_CONSTRUCTORS)
    def test_custom_cosmology(self, name: str) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        beam = _beam_from_constructor(name, CUSTOM_COSMO)
        assert beam.cosmo is CUSTOM_COSMO

    @pytest.mark.parametrize("name", BEAM_CONSTRUCTORS)
    def test_cosmo_conversions_deprecated(self, name: str) -> None:
        """A Cosmo_Conversions is converted to astropy with a DeprecationWarning."""
        with pytest.warns(DeprecationWarning):
            old = conversions.Cosmo_Conversions(H0=70.0, Om_L=0.7, Om_b=0.05, Om_c=0.25)
        with pytest.warns(DeprecationWarning, match="astropy.cosmology"):
            beam = _beam_from_constructor(name, old)
        assert isinstance(beam.cosmo, FLRW)
        assert beam.cosmo.is_equivalent(old.to_astropy())

    def test_set_cosmo(self) -> None:
        """Setting beam.cosmo accepts astropy, dicts and None; rejects others."""
        beam = pspecbeam.PSpecBeamBase()
        beam.cosmo = CUSTOM_COSMO
        assert beam.cosmo is CUSTOM_COSMO
        beam.cosmo = conversions.params_from_cosmology(CUSTOM_COSMO)
        assert beam.cosmo.is_equivalent(CUSTOM_COSMO)
        beam.cosmo = None
        assert beam.cosmo is conversions.DEFAULT_COSMOLOGY
        with pytest.raises(TypeError, match="FLRW"):
            beam.cosmo = [0.7, 0.05]
        del beam.cosmo
        assert not hasattr(beam, "cosmo")


class TestAllBeamTypes:
    @pytest.mark.parametrize("beam_name", ["beam_nf_pstokes", "gauss_beam"])
    def test_scalar_convergence(
        self, beam_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Check that compute_pspec_scalar converges as num_steps increases."""
        beam = request.getfixturevalue(beam_name)
        scalar = beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=500
        )
        scalar_large_Nsteps = beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="pI", num_steps=2000
        )
        assert abs(scalar / scalar_large_Nsteps - 1.0) <= 1e-5

    @pytest.mark.parametrize(
        "beam_name,expected",
        [
            ("beam_nf_pstokes_legacy", 1989353792.1765163),
            ("gauss_beam_legacy", 22123832163.072491),
            # the same beam as gauss_beam_legacy, given as arrays of beam integrals
            ("fromarray_beam_legacy", 22123832163.072491),
        ],
        ids=["uv", "gauss", "from_array"],
    )
    def test_taper(
        self, beam_name: str, expected: float, request: pytest.FixtureRequest
    ) -> None:
        """Check compute_pspec_scalar with a taper against a precomputed reference value."""
        beam = request.getfixturevalue(beam_name)
        scalar = beam.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, num_steps=5000, taper="blackman"
        )
        assert abs(scalar / expected - 1.0) <= 1e-8


def test_vectorised_x2y_matches_loop() -> None:
    """The vectorised X2Y gives the per-frequency Cosmo_Conversions values."""
    with pytest.warns(DeprecationWarning):
        old = conversions.Cosmo_Conversions()
    freqs = np.linspace(LOWER_FREQ.to_value(u.Hz), UPPER_FREQ.to_value(u.Hz), 200)
    redshifts = conversions.f2z(freqs)
    for little_h in (True, False):
        length_unit = conversions.length_unit_from_little_h(little_h)
        new = conversions.X2Y(
            redshifts, conversions.LEGACY_COSMOLOGY, length_unit
        ).to_value(length_unit**3 / (u.sr * u.Hz))
        loop = np.array([old.X2Y(z, little_h=little_h) for z in redshifts])
        np.testing.assert_allclose(new, loop, rtol=1e-14)


class TestScalarLengthUnit:
    @pytest.mark.parametrize("exact_norm", [False, True])
    def test_length_unit_matches_little_h(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV, exact_norm: bool
    ) -> None:
        """length_unit=Mpc / Mpc/littleh give the old little_h=False/True values."""
        kwargs = dict(pol="XX", num_steps=500, exact_norm=exact_norm)
        for little_h, length_unit in [(True, u.Mpc / cu.littleh), (False, u.Mpc)]:
            with pytest.warns(DeprecationWarning, match="little_h"):
                old = beam_nf_dipole.compute_pspec_scalar(
                    LOWER_FREQ, UPPER_FREQ, NUM_FREQS, little_h=little_h, **kwargs
                )
            new = beam_nf_dipole.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, length_unit=length_unit, **kwargs
            )
            assert new == old
            assert isinstance(new, float)
        default = beam_nf_dipole.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, **kwargs
        )
        littleh = beam_nf_dipole.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, length_unit=u.Mpc / cu.littleh, **kwargs
        )
        assert default == littleh
        mpc = beam_nf_dipole.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, length_unit="Mpc", **kwargs
        )
        h = beam_nf_dipole.cosmo.h
        np.testing.assert_allclose(mpc, littleh / h**3, rtol=1e-12)

    def test_disagreeing_arguments_raise(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        with (
            pytest.warns(DeprecationWarning, match="little_h"),
            pytest.raises(ValueError, match="disagrees"),
        ):
            beam_nf_dipole.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, little_h=True, length_unit=u.Mpc
            )

    def test_agreeing_arguments_warn(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        with pytest.warns(DeprecationWarning, match="little_h"):
            both = beam_nf_dipole.compute_pspec_scalar(
                LOWER_FREQ,
                UPPER_FREQ,
                NUM_FREQS,
                pol="XX",
                little_h=False,
                length_unit=u.Mpc,
            )
        assert both == beam_nf_dipole.compute_pspec_scalar(
            LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="XX", length_unit=u.Mpc
        )

    @pytest.mark.parametrize("length_unit", [u.s, u.Mpc**3, "littleh"])
    def test_bad_length_unit_raises(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV, length_unit
    ) -> None:
        with pytest.raises(ValueError, match="length_unit must be a length"):
            beam_nf_dipole.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, length_unit=length_unit
            )

    def test_private_function_little_h_deprecated(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        beam_freqs = beam_nf_dipole.beam_freqs
        ratio = beam_nf_dipole.power_beam_sq_int("XX") / (
            beam_nf_dipole.power_beam_int("XX") ** 2
        )
        pspec_freqs = np.linspace(
            LOWER_FREQ.to_value(u.Hz),
            UPPER_FREQ.to_value(u.Hz),
            NUM_FREQS,
            endpoint=False,
        )
        args = (conversions.DEFAULT_COSMOLOGY, beam_freqs, ratio, pspec_freqs)
        with pytest.warns(DeprecationWarning, match="little_h"):
            old = pspecbeam._compute_pspec_scalar(*args, little_h=False)
        assert old == pspecbeam._compute_pspec_scalar(*args, length_unit=u.Mpc)

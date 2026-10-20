import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
from astropy import constants
from astropy.cosmology import FLRW, LambdaCDM
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AiryBeam, GaussianBeam
from scipy import integrate
from uvtools import dspec

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


# Small analytic beams from pyuvdata. For the default sigma_type="efield",
# GaussianBeam's power response is exp(-theta^2 / sigma^2), so its power beam
# integral is close to the flat-sky value pi * sigma^2.
ANALYTIC_BEAM_FREQS = np.linspace(100e6, 200e6, 11)
ANALYTIC_BEAM_SIGMA = 0.3


def analytic_uvbeam(
    beam_type: str = "efield", pixel_coordinate_system: str = "healpix"
) -> UVBeam:
    """Analytic Gaussian UVBeam on an NSIDE=16 HEALPix map or a coarse az/za grid."""
    if pixel_coordinate_system == "healpix":
        grid = dict(nside=16)
    else:
        grid = dict(
            axis1_array=np.linspace(0, 2 * np.pi, 24, endpoint=False),
            axis2_array=np.linspace(0, np.pi / 2, 10),
        )
    return GaussianBeam(sigma=ANALYTIC_BEAM_SIGMA).to_uvbeam(
        freq_array=ANALYTIC_BEAM_FREQS,
        beam_type=beam_type,
        pixel_coordinate_system=pixel_coordinate_system,
        **grid,
    )


@pytest.fixture(scope="module")
def analytic_efield_beam() -> pspecbeam.PSpecBeamUV:
    """PSpecBeamUV built from an analytic efield beam (converted to power on init)."""
    return pspecbeam.PSpecBeamUV(analytic_uvbeam("efield"))


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

    @pytest.mark.parametrize("beam_name", ["beam_nf_dipole", "analytic_efield_beam"])
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
        """Check that get_Omegas gives the xx beam integrals, one column per polpair."""
        OP, OPP = beam_nf_dipole.get_Omegas(polpairs)
        assert OP.shape == (26, ncols)
        assert OPP.shape == (26, ncols)
        # the first polpair of both cases is xx
        np.testing.assert_array_equal(OP[:, 0], beam_nf_dipole.power_beam_int(pol="xx"))
        np.testing.assert_array_equal(
            OPP[:, 0], beam_nf_dipole.power_beam_sq_int(pol="xx")
        )

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

    def test_beam_normalized_response_raises_on_wrong_pol(
        self, beam_nf_dipole: pspecbeam.PSpecBeamUV
    ) -> None:
        """Check that an unavailable polarization raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        with pytest.raises(ValueError, match="Do not have the right polarization"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(
                beam_nf_dipole, pol="ll", freq=freq
            )

    def test_beam_normalized_response_raises_on_non_power_beam(self) -> None:
        """Check that a non-power beam_type raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        # PSpecBeamUV converts efield beams to power beams on init, so an efield
        # primary beam can only appear if it is assigned after construction.
        beam = pspecbeam.PSpecBeamUV(analytic_uvbeam("efield"))
        beam.primary_beam = analytic_uvbeam("efield")
        with pytest.raises(ValueError, match="beam_type must be power"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol="xx", freq=freq)

    def test_beam_normalized_response_raises_on_wrong_axes(self) -> None:
        """Check that Naxes_vec > 1 raises a ValueError for a power beam."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        # A power beam that keeps the basis-vector axis (Naxes_vec = 2).
        uvb = analytic_uvbeam("efield")
        uvb.efield_to_power(keep_basis_vector=True)
        uvb.peak_normalize()
        beam = pspecbeam.PSpecBeamUV(uvb)
        with pytest.raises(ValueError, match="Expect scalar for power beam"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol="xx", freq=freq)

    def test_beam_normalized_response_raises_on_non_peak_normalization(self) -> None:
        """Check that a non-peak-normalized beam raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        # analytic power beams are normalized to physical units
        beam = pspecbeam.PSpecBeamUV(analytic_uvbeam("power"))
        with pytest.raises(ValueError, match="beam must be peak normalized"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol="xx", freq=freq)

    def test_beam_normalized_response_raises_on_non_healpix_coords(self) -> None:
        """Check that a non-healpix pixel coordinate system raises a ValueError."""
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10) * u.Hz
        beam = pspecbeam.PSpecBeamUV(
            analytic_uvbeam("efield", pixel_coordinate_system="az_za")
        )
        with pytest.raises(ValueError, match="only healpix format supported"):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol="xx", freq=freq)

    def test_efield_beam_converted_to_power(
        self, analytic_efield_beam: pspecbeam.PSpecBeamUV
    ) -> None:
        """An efield beam gives the same beam integrals and scalar as the
        equivalent peak-normalized power beam."""
        assert analytic_efield_beam.primary_beam.beam_type == "power"
        assert analytic_efield_beam.primary_beam.data_normalization == "peak"

        power_uvb = analytic_uvbeam("power")
        power_uvb.peak_normalize()
        power_beam = pspecbeam.PSpecBeamUV(power_uvb)
        for pol in ["xx", "yy"]:
            np.testing.assert_allclose(
                analytic_efield_beam.power_beam_int(pol),
                power_beam.power_beam_int(pol),
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                analytic_efield_beam.power_beam_sq_int(pol),
                power_beam.power_beam_sq_int(pol),
                rtol=1e-10,
            )
        np.testing.assert_allclose(
            analytic_efield_beam.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="xx"
            ),
            power_beam.compute_pspec_scalar(
                LOWER_FREQ, UPPER_FREQ, NUM_FREQS, pol="xx"
            ),
            rtol=1e-10,
        )
        # The beam area is close to the flat-sky value; the difference is
        # due to sky curvature and the coarse HEALPix map.
        np.testing.assert_allclose(
            analytic_efield_beam.power_beam_int("xx"),
            np.pi * ANALYTIC_BEAM_SIGMA**2,
            rtol=2e-2,
        )


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


class TestBeamPhysics:
    """Check beam integrals, unit conversions and the power spectrum scalar
    against their analytic definitions."""

    FWHM = 0.3  # FWHM of the Gaussian power beam, in radians
    FREQS = np.linspace(140e6, 170e6, 31)

    @pytest.fixture(scope="class")
    def beam(self) -> pspecbeam.PSpecBeamGauss:
        return pspecbeam.PSpecBeamGauss(self.FWHM * u.rad, self.FREQS * u.Hz)

    def test_gaussian_beam_integrals(self, beam: pspecbeam.PSpecBeamGauss) -> None:
        """PSpecBeamGauss describes a power beam B = exp(-theta^2 / 2 sigma^2)
        whose FWHM is fwhm, i.e. sigma = fwhm / sqrt(8 ln 2), and this is the
        power response of the pyuvdata GaussianBeam it holds. Its integrals over
        the (flat) sky are Omega_p = 2 pi sigma^2 and Omega_pp = pi sigma^2."""
        sigma = self.FWHM / np.sqrt(8 * np.log(2))

        def power_beam(theta):
            return np.exp(-(theta**2) / (2 * sigma**2))

        np.testing.assert_allclose(power_beam(self.FWHM / 2), 0.5)
        theta = np.linspace(0, 3 * self.FWHM, 7)
        analytic = beam.analytic_beam.power_eval(
            az_array=np.zeros_like(theta), za_array=theta, freq_array=self.FREQS[:1]
        )
        # (axis, polarization, frequency, direction); xx
        np.testing.assert_allclose(analytic[0, 0, 0].real, power_beam(theta))
        omega_p = integrate.quad(lambda t: 2 * np.pi * t * power_beam(t), 0, np.inf)
        omega_pp = integrate.quad(
            lambda t: 2 * np.pi * t * power_beam(t) ** 2, 0, np.inf
        )
        np.testing.assert_allclose(beam.power_beam_int(), omega_p[0], rtol=1e-10)
        np.testing.assert_allclose(beam.power_beam_sq_int(), omega_pp[0], rtol=1e-10)

    def test_healpix_gaussian_beam_matches_analytic(self) -> None:
        """A HEALPix map of a Gaussian beam gives the beam integrals, and so the
        pspec scalar, of the analytic Gaussian beam.

        pyuvdata's GaussianBeam(sigma) has an efield response
        exp(-theta^2 / 2 sigma^2) by default, so its power beam is
        exp(-theta^2 / sigma^2), with a FWHM of sqrt(8 ln 2) sigma / sqrt(2). The
        same GaussianBeam, mapped onto HEALPix and wrapped in PSpecBeamUV or
        wrapped directly in PSpecBeamGauss, gives the same beam integrals to
        ~1%, the accuracy of the NSIDE=64 map and of the flat-sky approximation.
        """
        sigma = 0.15
        freqs = np.linspace(100e6, 200e6, 3)
        analytic = GaussianBeam(sigma=sigma)
        uvb = analytic.to_uvbeam(
            freq_array=freqs,
            beam_type="efield",
            pixel_coordinate_system="healpix",
            nside=64,
        )
        healpix_beam = pspecbeam.PSpecBeamUV(uvb)
        gauss = pspecbeam.PSpecBeamGauss(beam=analytic, beam_freqs=freqs * u.Hz)
        np.testing.assert_allclose(
            gauss.fwhm, np.sqrt(8 * np.log(2)) * sigma / np.sqrt(2), rtol=1e-12
        )
        np.testing.assert_allclose(
            healpix_beam.power_beam_int("xx"), gauss.power_beam_int(), rtol=1.5e-2
        )
        np.testing.assert_allclose(
            healpix_beam.power_beam_sq_int("xx"), gauss.power_beam_sq_int(), rtol=1.5e-2
        )
        band = (150e6 * u.Hz, 152e6 * u.Hz, 20)
        kwargs = dict(pol="xx", num_steps=200)
        np.testing.assert_allclose(
            healpix_beam.compute_pspec_scalar(*band, **kwargs),
            gauss.compute_pspec_scalar(*band, **kwargs),
            rtol=1.5e-2,
        )

    def test_jy_to_mk_is_rayleigh_jeans(self, beam: pspecbeam.PSpecBeamGauss) -> None:
        """Jy_to_mK converts a flux density spread over the beam into a
        Rayleigh-Jeans brightness temperature, T = S c^2 / (2 k_B nu^2 Omega_p),
        which scales as nu^-2 for a frequency-independent beam."""
        freqs = np.linspace(145e6, 165e6, 5) * u.Hz
        expected = (
            (
                1
                * u.Jy
                * constants.c**2
                / (2 * constants.k_B * freqs**2 * beam.power_beam_int()[0])
            )
            .to(u.mK)
            .value
        )
        np.testing.assert_allclose(beam.Jy_to_mK(freqs), expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "length_unit", [u.Mpc / cu.littleh, u.Mpc], ids=["littleh", "Mpc"]
    )
    def test_scalar_is_cosmological_volume_over_beam(
        self, beam: pspecbeam.PSpecBeamGauss, length_unit: u.UnitBase
    ) -> None:
        """Without a taper, the scalar is X2Y(z) * B * Omega_p^2 / Omega_pp for a
        narrow band, where X2Y converts sr Hz to comoving volume and B is the
        bandwidth. The bandwidth is N * channel width (the band edges), not the
        (N - 1) channel widths between the first and last channel centres."""
        nfreqs, df, f0 = 20, 100e3 * u.Hz, 150e6 * u.Hz
        scalar = beam.compute_pspec_scalar(
            f0, f0 + nfreqs * df, nfreqs, num_steps=2000, length_unit=length_unit
        )
        z = conversions.f2z(f0 + nfreqs * df / 2)
        x2y = conversions.X2Y(z, beam.cosmo, length_unit)
        omega_p, omega_pp = beam.power_beam_int()[0], beam.power_beam_sq_int()[0]
        expected = x2y * nfreqs * df * omega_p**2 / omega_pp * u.sr
        np.testing.assert_allclose(scalar, expected.to_value(length_unit**3), rtol=1e-3)

    @pytest.mark.parametrize("taper", ["bh", "blackman", "hann"])
    def test_scalar_taper_factor(
        self, beam: pspecbeam.PSpecBeamGauss, taper: str
    ) -> None:
        """A taper w reduces the effective bandwidth B_pp / B by mean(w^2), so the
        scalar increases by 1 / mean(w^2). The taper is interpolated onto the
        integration grid by nearest neighbour, hence the tolerance."""
        nfreqs = 20
        band = (150e6 * u.Hz, 152e6 * u.Hz, nfreqs)
        kwargs = dict(num_steps=2000)
        tapered = beam.compute_pspec_scalar(*band, taper=taper, **kwargs)
        untapered = beam.compute_pspec_scalar(*band, **kwargs)
        window = dspec.gen_window(taper, nfreqs)
        np.testing.assert_allclose(
            tapered / untapered, 1 / np.mean(window**2), rtol=2e-3
        )

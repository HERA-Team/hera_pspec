from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import constants
from astropy import units as u
from pyuvdata import UVBeam
from pyuvdata.analytic_beam import AiryBeam, GaussianBeam
from scipy import integrate
from uvtools import dspec

from hera_pspec import conversions, pspecbeam
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)

# Frequency range/resolution shared by compute_pspec_scalar calls across all beam types.
LOWER_FREQ = 120.0 * 10**6
UPPER_FREQ = 128.0 * 10**6
NUM_FREQS = 20


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


def _efield_primary_beam(request: pytest.FixtureRequest) -> pspecbeam.PSpecBeamUV:
    # PSpecBeamUV converts efield beams to power beams on init, so an efield
    # primary beam can only appear if it is assigned after construction.
    beam = pspecbeam.PSpecBeamUV(analytic_uvbeam("efield"))
    beam.primary_beam = analytic_uvbeam("efield")
    return beam


def _vector_power_beam(request: pytest.FixtureRequest) -> pspecbeam.PSpecBeamUV:
    # A power beam that keeps the basis-vector axis (Naxes_vec = 2).
    uvb = analytic_uvbeam("efield")
    uvb.efield_to_power(keep_basis_vector=True)
    uvb.peak_normalize()
    return pspecbeam.PSpecBeamUV(uvb)


class TestPSpecBeamUV:
    def test_init(self, beam_nf_dipole: pspecbeam.PSpecBeamUV) -> None:
        """Check that PSpecBeamUV loads a beamfits file and sets primary_beam/beam_freqs/cosmo."""
        assert isinstance(beam_nf_dipole.primary_beam, UVBeam)
        assert beam_nf_dipole.beam_freqs.size > 0
        assert isinstance(beam_nf_dipole.cosmo, conversions.Cosmo_Conversions)

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
        M = beam_nf_pstokes.Jy_to_mK(np.linspace(100e6, 200e6, 11))
        assert len(M) == 11
        np.testing.assert_almost_equal(M[0], 40.643366654821904)
        M = beam_nf_pstokes.Jy_to_mK(150e6)
        assert isinstance(M, np.ndarray)

        # Frequencies outside the beam's frequency range are extrapolated, with
        # a warning for each side.
        with (
            pytest.warns(UserWarning, match="min freq .* extrapolating"),
            pytest.warns(UserWarning, match="max freq .* extrapolating"),
        ):
            M = beam_nf_pstokes.Jy_to_mK(np.linspace(90e6, 210e6, 13))
        assert np.all(np.isfinite(M))

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
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        nside = beam_nf_dipole.primary_beam.nside
        beam_res = pspecbeam.PSpecBeamUV.beam_normalized_response(
            beam_nf_dipole, pol="xx", freq=freq
        )
        assert len(beam_res[1]) == len(freq)
        assert beam_res[0].ndim == 2
        assert np.shape(beam_res[0]) == (len(freq), (12 * nside**2))

    @pytest.mark.parametrize(
        "make_beam,pol,match",
        [
            pytest.param(
                lambda request: request.getfixturevalue("beam_nf_dipole"),
                "ll",
                "Do not have the right polarization",
                id="unavailable_pol",
            ),
            pytest.param(
                _efield_primary_beam,
                "xx",
                "beam_type must be power",
                id="non_power_beam",
            ),
            pytest.param(
                _vector_power_beam,
                "xx",
                "Expect scalar for power beam",
                id="vector_power_beam",
            ),
            pytest.param(
                # analytic power beams are normalized to physical units
                lambda request: pspecbeam.PSpecBeamUV(analytic_uvbeam("power")),
                "xx",
                "beam must be peak normalized",
                id="non_peak_normalized",
            ),
            pytest.param(
                lambda request: pspecbeam.PSpecBeamUV(
                    analytic_uvbeam("efield", pixel_coordinate_system="az_za")
                ),
                "xx",
                "only healpix format supported",
                id="non_healpix_coords",
            ),
        ],
    )
    def test_beam_normalized_response_raises(
        self,
        make_beam: Callable[[pytest.FixtureRequest], pspecbeam.PSpecBeamUV],
        pol: str,
        match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Check that beams beam_normalized_response cannot handle raise a ValueError."""
        beam = make_beam(request)
        freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
        with pytest.raises(ValueError, match=match):
            pspecbeam.PSpecBeamUV.beam_normalized_response(beam, pol=pol, freq=freq)

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
    return pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))


@pytest.fixture(scope="module")
def fromarray_beam(
    gauss_beam: pspecbeam.PSpecBeamGauss,
) -> pspecbeam.PSpecBeamFromArray:
    """A PSpecBeamFromArray built from gauss_beam's beam integrals."""
    return pspecbeam.PSpecBeamFromArray(
        OmegaP=gauss_beam.power_beam_int(),
        OmegaPP=gauss_beam.power_beam_sq_int(),
        beam_freqs=gauss_beam.beam_freqs,
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

    def test_scalar(self, gauss_beam: pspecbeam.PSpecBeamGauss) -> None:
        """Check compute_pspec_scalar against a precomputed reference value."""
        scalar = gauss_beam.compute_pspec_scalar(
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
            beam=analytic, beam_freqs=gauss_beam.beam_freqs
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
            beam_freqs=gauss_beam.beam_freqs,
        )
        from_efield = pspecbeam.PSpecBeamGauss(
            beam=GaussianBeam(sigma=np.sqrt(2.0) * sigma_power, sigma_type="efield"),
            beam_freqs=gauss_beam.beam_freqs,
        )
        np.testing.assert_allclose(from_power.fwhm, from_efield.fwhm)
        np.testing.assert_allclose(
            from_power.power_beam_int(), from_efield.power_beam_int()
        )
        np.testing.assert_allclose(
            from_power.power_beam_sq_int(), from_efield.power_beam_sq_int()
        )

    @pytest.mark.parametrize(
        "build_beam,exc_type,match",
        [
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(
                    beam=GaussianBeam(diameter=14.0), beam_freqs=freqs
                ),
                ValueError,
                "Diameter-specified",
                id="diameter_specified_beam",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(
                    beam=GaussianBeam(
                        sigma=0.1, spectral_index=-1.0, reference_frequency=150e6
                    ),
                    beam_freqs=freqs,
                ),
                NotImplementedError,
                "Chromatic",
                id="chromatic_beam",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(
                    beam=AiryBeam(diameter=14.0), beam_freqs=freqs
                ),
                TypeError,
                "must be a pyuvdata GaussianBeam",
                id="non_gaussian_beam",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(
                    fwhm=0.8, beam=GaussianBeam(sigma=0.1), beam_freqs=freqs
                ),
                ValueError,
                "Exactly one of fwhm or beam",
                id="fwhm_and_beam",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(beam_freqs=freqs),
                ValueError,
                "Exactly one of fwhm or beam",
                id="neither_fwhm_nor_beam",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(fwhm=0.8),
                ValueError,
                "beam_freqs must be specified",
                id="no_beam_freqs",
            ),
            pytest.param(
                lambda freqs: pspecbeam.PSpecBeamGauss(GaussianBeam(sigma=0.1), freqs),
                TypeError,
                "pass analytic beams via the beam keyword",
                id="analytic_beam_as_fwhm",
            ),
        ],
    )
    def test_raises_on_invalid_init_args(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        build_beam: Callable[[np.ndarray], pspecbeam.PSpecBeamGauss],
        exc_type: type[Exception],
        match: str,
    ) -> None:
        """Check clear errors for unsupported analytic beams and for invalid
        fwhm/beam/beam_freqs combinations."""
        with pytest.raises(exc_type, match=match):
            build_beam(gauss_beam.beam_freqs)

    def test_omegas_match_numerical_integration(self) -> None:
        """Check the closed-form Omegas against healpix-integrated areas of the wrapped GaussianBeam.

        The closed forms are flat-sky results, so agreement is only expected
        to ~0.5% for a narrow beam; they remain the authoritative values.
        """
        fwhm = 0.1
        beam_freqs = np.array([150e6])
        bgauss = pspecbeam.PSpecBeamGauss(fwhm, beam_freqs)
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

    def test_matches_gauss(
        self,
        gauss_beam: pspecbeam.PSpecBeamGauss,
        fromarray_beam: pspecbeam.PSpecBeamFromArray,
    ) -> None:
        """Check that compute_pspec_scalar matches the equivalent Gaussian beam's result."""
        scalar = fromarray_beam.compute_pspec_scalar(
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
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        kwargs = build_kwargs(Om_P, Om_PP, gauss_beam.beam_freqs)
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
        Om_P = gauss_beam.power_beam_int()
        Om_PP = gauss_beam.power_beam_sq_int()
        with pytest.raises(KeyError, match=match):
            call(Om_P, Om_PP, gauss_beam.beam_freqs)


class TestPSpecBeamBase:
    def test_default_construction(self) -> None:
        """Check that the base class can be instantiated with a default cosmology."""
        bm1 = pspecbeam.PSpecBeamBase()
        assert isinstance(bm1.cosmo, conversions.Cosmo_Conversions)


class TestAllBeamTypes:
    @pytest.mark.parametrize(
        "build_beam",
        [
            lambda request, cosmo: request.getfixturevalue("beam_nf_dipole_wcosmo"),
            lambda request, cosmo: pspecbeam.PSpecBeamGauss(
                0.8, np.linspace(115e6, 130e6, 50, endpoint=False), cosmo=cosmo
            ),
            lambda request, cosmo: pspecbeam.PSpecBeamFromArray(
                OmegaP=request.getfixturevalue("gauss_beam").power_beam_int(),
                OmegaPP=request.getfixturevalue("gauss_beam").power_beam_sq_int(),
                beam_freqs=request.getfixturevalue("gauss_beam").beam_freqs,
                cosmo=cosmo,
            ),
            lambda request, cosmo: pspecbeam.PSpecBeamBase(cosmo=cosmo),
        ],
        ids=["uv", "gauss", "from_array", "base"],
    )
    def test_custom_cosmology(
        self,
        build_beam: Callable[
            [pytest.FixtureRequest, conversions.Cosmo_Conversions],
            pspecbeam.PSpecBeamBase,
        ],
        request: pytest.FixtureRequest,
    ) -> None:
        """Check that a user-defined cosmology can be passed to the constructor."""
        # the UV beam comes from a session fixture built with Cosmo_Conversions()
        beam = build_beam(request, conversions.Cosmo_Conversions())
        assert isinstance(beam.cosmo, conversions.Cosmo_Conversions)

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
            ("beam_nf_pstokes", 1989353792.1765163),
            ("gauss_beam", 22123832163.072491),
            # the same beam as gauss_beam, given as arrays of beam integrals
            ("fromarray_beam", 22123832163.072491),
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


class TestBeamPhysics:
    """Check beam integrals, unit conversions and the power spectrum scalar
    against their analytic definitions."""

    FWHM = 0.3  # FWHM of the Gaussian power beam, in radians
    FREQS = np.linspace(140e6, 170e6, 31)

    @pytest.fixture(scope="class")
    def beam(self) -> pspecbeam.PSpecBeamGauss:
        return pspecbeam.PSpecBeamGauss(self.FWHM, self.FREQS)

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
        gauss = pspecbeam.PSpecBeamGauss(beam=analytic, beam_freqs=freqs)
        np.testing.assert_allclose(
            gauss.fwhm, np.sqrt(8 * np.log(2)) * sigma / np.sqrt(2), rtol=1e-12
        )
        np.testing.assert_allclose(
            healpix_beam.power_beam_int("xx"), gauss.power_beam_int(), rtol=1.5e-2
        )
        np.testing.assert_allclose(
            healpix_beam.power_beam_sq_int("xx"), gauss.power_beam_sq_int(), rtol=1.5e-2
        )
        kwargs = dict(pol="xx", num_steps=200)
        np.testing.assert_allclose(
            healpix_beam.compute_pspec_scalar(150e6, 152e6, 20, **kwargs),
            gauss.compute_pspec_scalar(150e6, 152e6, 20, **kwargs),
            rtol=1.5e-2,
        )

    def test_jy_to_mk_is_rayleigh_jeans(self, beam: pspecbeam.PSpecBeamGauss) -> None:
        """Jy_to_mK converts a flux density spread over the beam into a
        Rayleigh-Jeans brightness temperature, T = S c^2 / (2 k_B nu^2 Omega_p),
        which scales as nu^-2 for a frequency-independent beam."""
        freqs = np.linspace(145e6, 165e6, 5)
        expected = (
            (
                1
                * u.Jy
                * constants.c**2
                / (2 * constants.k_B * (freqs * u.Hz) ** 2 * beam.power_beam_int()[0])
            )
            .to(u.mK)
            .value
        )
        # astropy's physical constants are a more recent CODATA release
        np.testing.assert_allclose(beam.Jy_to_mK(freqs), expected, rtol=1e-5)

    @pytest.mark.parametrize("little_h", [True, False])
    def test_scalar_is_cosmological_volume_over_beam(
        self, beam: pspecbeam.PSpecBeamGauss, little_h: bool
    ) -> None:
        """Without a taper, the scalar is X2Y(z) * B * Omega_p^2 / Omega_pp for a
        narrow band, where X2Y converts sr Hz to comoving volume and B is the
        bandwidth. The bandwidth is N * channel width (the band edges), not the
        (N - 1) channel widths between the first and last channel centres."""
        nfreqs, df, f0 = 20, 100e3, 150e6
        scalar = beam.compute_pspec_scalar(
            f0, f0 + nfreqs * df, nfreqs, num_steps=2000, little_h=little_h
        )
        z = beam.cosmo.f2z(f0 + nfreqs * df / 2)
        omega_p, omega_pp = beam.power_beam_int()[0], beam.power_beam_sq_int()[0]
        expected = (
            beam.cosmo.X2Y(z, little_h=little_h) * nfreqs * df * omega_p**2 / omega_pp
        )
        np.testing.assert_allclose(scalar, expected, rtol=1e-3)

    def test_scalar_little_h(self, beam: pspecbeam.PSpecBeamGauss) -> None:
        """A volume in h^-3 Mpc^3 is h^3 times the same volume in Mpc^3."""
        kwargs = dict(num_steps=500)
        in_mpc = beam.compute_pspec_scalar(150e6, 152e6, 20, little_h=False, **kwargs)
        in_mpc_h = beam.compute_pspec_scalar(150e6, 152e6, 20, little_h=True, **kwargs)
        np.testing.assert_allclose(in_mpc_h / in_mpc, beam.cosmo.h**3, rtol=1e-10)

    @pytest.mark.parametrize("taper", ["bh", "blackman", "hann"])
    def test_scalar_taper_factor(
        self, beam: pspecbeam.PSpecBeamGauss, taper: str
    ) -> None:
        """A taper w reduces the effective bandwidth B_pp / B by mean(w^2), so the
        scalar increases by 1 / mean(w^2). The taper is interpolated onto the
        integration grid by nearest neighbour, hence the tolerance."""
        nfreqs = 20
        kwargs = dict(num_steps=2000)
        tapered = beam.compute_pspec_scalar(150e6, 152e6, nfreqs, taper=taper, **kwargs)
        untapered = beam.compute_pspec_scalar(150e6, 152e6, nfreqs, **kwargs)
        window = dspec.gen_window(taper, nfreqs)
        np.testing.assert_allclose(
            tapered / untapered, 1 / np.mean(window**2), rtol=2e-3
        )

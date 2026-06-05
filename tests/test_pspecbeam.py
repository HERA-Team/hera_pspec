import warnings
from pathlib import Path

import numpy as np
import pytest
from pyuvdata import UVBeam

from hera_pspec import conversions, pspecbeam
from hera_pspec.data import DATA_PATH

DATA_PATH = Path(DATA_PATH)


def test_init():
    beamfile = DATA_PATH / "HERA_NF_dipole_power.beamfits"
    bm = pspecbeam.PSpecBeamUV(beamfile)


def test_beamfiles_load_without_warnings():
    beamfiles = [
        "HERA_NF_pstokes_power.beamfits",
        "HERA_NF_dipole_power.beamfits",
        "HERA_NF_efield.beamfits",
        "isotropic_beam.beamfits",
    ]

    for beamfile in beamfiles:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pspecbeam.PSpecBeamUV(DATA_PATH / beamfile)

        assert not caught, [str(w.message) for w in caught]


def test_UVbeam(beam_nf_dipole):
    # Precomputed results in the following tests were done "by hand" using
    # iPython notebook "Scalar_dev2.ipynb" in tests directory
    pstokes_beamfile = DATA_PATH / "HERA_NF_pstokes_power.beamfits"
    uvb = UVBeam()
    uvb.read_beamfits(pstokes_beamfile)
    beam = pspecbeam.PSpecBeamUV(uvb)
    Om_p = beam.power_beam_int()
    Om_pp = beam.power_beam_sq_int()
    lower_freq = 120.0 * 10**6
    upper_freq = 128.0 * 10**6
    num_freqs = 20

    # check pI polarization
    scalar = beam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=2000
    )

    np.testing.assert_almost_equal(Om_p[0], 0.080082680885906782)
    np.testing.assert_almost_equal(Om_p[18], 0.031990943334017245)
    np.testing.assert_almost_equal(Om_p[-1], 0.03100215028171072)

    np.testing.assert_almost_equal(Om_pp[0], 0.036391945229980432)
    np.testing.assert_almost_equal(Om_pp[15], 0.018159280192894631)
    np.testing.assert_almost_equal(Om_pp[-1], 0.014528100116719534)

    assert abs(scalar / 568847837.72586381 - 1.0) <= 1e-4

    # Check array dimensionality
    assert Om_p.ndim == 1
    assert Om_pp.ndim == 1

    # convergence of integral
    scalar_large_Nsteps = beam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=10000
    )
    assert abs(scalar / scalar_large_Nsteps - 1.0) <= 1e-5

    # Check that user-defined cosmology can be specified
    beam = pspecbeam.PSpecBeamUV(
        pstokes_beamfile, cosmo=conversions.Cosmo_Conversions()
    )

    # Check that errors are not raised for other Stokes parameters
    for pol in ["pQ", "pU", "pV"]:
        scalar = beam.compute_pspec_scalar(
            lower_freq, upper_freq, num_freqs, pol=pol, num_steps=2000
        )

    # test taper execution
    scalar = beam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, num_steps=5000, taper="blackman"
    )
    assert abs(scalar / 1989353792.1765163 - 1.0) <= 1e-8

    # test Jy_to_mK
    M = beam.Jy_to_mK(np.linspace(100e6, 200e6, 11))
    assert len(M) == 11
    np.testing.assert_almost_equal(M[0], 40.643366654821904)
    M = beam.Jy_to_mK(150e6)
    assert isinstance(M, np.ndarray)
    M = beam.Jy_to_mK(np.linspace(90, 210e6, 11))

    # test exception
    with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
        beam.Jy_to_mK([1])
    with pytest.raises(TypeError, match="freqs must be fed as a float ndarray"):
        beam.Jy_to_mK(np.array([1]))

    # test noise scalar
    sclr = beam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=2000, noise_scalar=True
    )
    np.testing.assert_almost_equal(sclr, 71.105979715733)

    # Check that invalid polarizations raise an error.
    # No match= because pyuvdata's polstr2num intercepts the invalid pol string
    # before pspecbeam's own validation, and the exception type/message varies
    # across pyuvdata versions (KeyError in older, ValueError in newer).
    with pytest.raises((KeyError, ValueError)):
        beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol="pZ")
    with pytest.raises((KeyError, ValueError)):
        beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol="XX")

    # check dipole beams work
    scalar = beam_nf_dipole.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="XX"
    )
    with pytest.raises(
        (KeyError, ValueError)
    ):  # see note above about pyuvdata versions
        beam_nf_dipole.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol="pI")

    # check efield beams work
    efield_beamfile = DATA_PATH / "HERA_NF_efield.beamfits"
    beam = pspecbeam.PSpecBeamUV(efield_beamfile)
    scalar = beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol="XX")
    with pytest.raises(
        (KeyError, ValueError)
    ):  # see note above about pyuvdata versions
        beam.compute_pspec_scalar(lower_freq, upper_freq, num_freqs, pol="pI")


def test_Gaussbeam():
    gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))

    Om_p = gauss.power_beam_int()
    Om_pp = gauss.power_beam_sq_int()
    lower_freq = 120.0 * 10**6
    upper_freq = 128.0 * 10**6
    num_freqs = 20
    scalar = gauss.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=2000
    )

    # Check that user-defined cosmology can be specified
    bgauss = pspecbeam.PSpecBeamGauss(
        0.8,
        np.linspace(115e6, 130e6, 50, endpoint=False),
        cosmo=conversions.Cosmo_Conversions(),
    )

    # Check array dimensionality
    assert Om_p.ndim == 1
    assert Om_pp.ndim == 1

    assert Om_p[4] == 0.7251776226923511
    assert Om_p[4] == Om_p[0]

    assert Om_pp[4] == 0.36258881134617554
    assert Om_pp[4] == Om_pp[0]

    assert abs(scalar / 6392750120.8657961 - 1.0) <= 1e-4

    # convergence of integral
    scalar_large_Nsteps = gauss.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, num_steps=5000
    )
    assert abs(scalar / scalar_large_Nsteps - 1.0) <= 1e-5

    # test taper execution
    scalar = gauss.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, num_steps=5000, taper="blackman"
    )
    assert abs(scalar / 22123832163.072491 - 1.0) <= 1e-8


def test_BeamFromArray():
    """
    Test PSpecBeamFromArray
    """
    # Get Gaussian beam to use as a reference
    gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))
    Om_P = gauss.power_beam_int()
    Om_PP = gauss.power_beam_sq_int()
    beam_freqs = gauss.beam_freqs

    # Array specs for tests
    lower_freq = 120.0 * 10**6
    upper_freq = 128.0 * 10**6
    num_freqs = 20

    # Check that PSpecBeamFromArray can be instantiated
    psbeam = pspecbeam.PSpecBeamFromArray(
        OmegaP=Om_P, OmegaPP=Om_PP, beam_freqs=beam_freqs
    )

    psbeampol = pspecbeam.PSpecBeamFromArray(
        OmegaP={"pI": Om_P, "pQ": Om_P},
        OmegaPP={"pI": Om_PP, "pQ": Om_PP},
        beam_freqs=beam_freqs,
    )

    # Check that user-defined cosmology can be specified
    bm2 = pspecbeam.PSpecBeamFromArray(
        OmegaP=Om_P,
        OmegaPP=Om_PP,
        beam_freqs=beam_freqs,
        cosmo=conversions.Cosmo_Conversions(),
    )

    # Compare scalar calculation with Gaussian case
    scalar = psbeam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=2000
    )
    g_scalar = gauss.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pI", num_steps=2000
    )
    np.testing.assert_array_almost_equal(scalar, g_scalar)

    # Check that polarizations are recognized and invalid ones rejected
    scalarp = psbeampol.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, pol="pQ", num_steps=2000
    )

    # Test taper execution (same as Gaussian case)
    scalar = psbeam.compute_pspec_scalar(
        lower_freq, upper_freq, num_freqs, num_steps=5000, taper="blackman"
    )
    assert abs(scalar / 22123832163.072491 - 1.0) <= 1e-8

    # Check that invalid init args raise errors
    with pytest.raises(TypeError, match="must both be either dicts or arrays"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP=Om_P, OmegaPP={"pI": Om_PP}, beam_freqs=beam_freqs
        )
    with pytest.raises(KeyError, match="must be specified for both"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P, "pQ": Om_P},
            OmegaPP={"pI": Om_PP},
            beam_freqs=beam_freqs,
        )
    with pytest.raises(KeyError, match="'a'"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"A": Om_P}, OmegaPP={"A": Om_PP}, beam_freqs=beam_freqs
        )
    with pytest.raises(TypeError, match="must both be array_like"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P}, OmegaPP={"pI": "string"}, beam_freqs=beam_freqs
        )
    with pytest.raises(ValueError, match="same shape as beam_freqs"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P}, OmegaPP={"pI": Om_PP[:-2]}, beam_freqs=beam_freqs
        )
    with pytest.raises(TypeError, match="must both be either dicts or arrays"):
        pspecbeam.PSpecBeamFromArray(
            OmegaP=Om_P, OmegaPP={"pI": Om_PP[:-2]}, beam_freqs=beam_freqs
        )
    # No match= on the next four: pyuvdata's polstr2num raises before pspecbeam's
    # own "Unrecognized polarization" check; exception type/message varies by version.
    with pytest.raises((KeyError, ValueError)):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"foo": Om_P}, OmegaPP={"pI": Om_PP}, beam_freqs=beam_freqs
        )
    with pytest.raises((KeyError, ValueError)):
        pspecbeam.PSpecBeamFromArray(
            OmegaP={"pI": Om_P}, OmegaPP={"foo": Om_PP}, beam_freqs=beam_freqs
        )

    # No match= on the next six: same pyuvdata version sensitivity as above.
    with pytest.raises((KeyError, ValueError)):
        psbeam.add_pol("foo", Om_P, Om_PP)
    with pytest.raises((KeyError, ValueError)):
        psbeam.power_beam_int("foo")
    with pytest.raises((KeyError, ValueError)):
        psbeam.power_beam_sq_int("foo")

    # Check that invalid method args raise errors
    with pytest.raises((KeyError, ValueError)):
        psbeam.power_beam_int(pol="blah")
    with pytest.raises((KeyError, ValueError)):
        psbeam.power_beam_sq_int(pol="blah")
    with pytest.raises((KeyError, ValueError)):
        psbeam.add_pol(pol="A", OmegaP=Om_P, OmegaPP=Om_PP)

    # Check that string works
    assert len(str(psbeam)) > 0


def test_PSpecBeamBase():
    """
    Test that base class can be instantiated.
    """
    bm1 = pspecbeam.PSpecBeamBase()

    # Check that user-defined cosmology can be specified
    bm2 = pspecbeam.PSpecBeamBase(cosmo=conversions.Cosmo_Conversions())


def test_get_Omegas(beam_nf_dipole):
    OP, OPP = beam_nf_dipole.get_Omegas(("xx", "xx"))
    assert OP.shape == (26, 1)
    assert OPP.shape == (26, 1)
    OP, OPP = beam_nf_dipole.get_Omegas([(-5, -5), (-6, -6)])
    assert OP.shape == (26, 2)
    assert OPP.shape == (26, 2)

    with pytest.raises(TypeError, match="polpairs is not a list"):
        beam_nf_dipole.get_Omegas("xx")
    with pytest.raises(NotImplementedError, match="does not support cross-correlation"):
        beam_nf_dipole.get_Omegas([("pI", "pQ")])


def test_beam_normalized_response(beam_nf_dipole):
    freq = np.linspace(130.0 * 1e6, 140.0 * 1e6, 10)
    nside = beam_nf_dipole.primary_beam.nside  # uvbeam object
    beam_res = pspecbeam.PSpecBeamUV.beam_normalized_response(
        beam_nf_dipole, pol="xx", freq=freq
    )

    # tests for dimensions
    assert len(beam_res[1]) == len(freq)
    assert beam_res[0].ndim == 2
    assert np.shape(beam_res[0]) == (len(freq), (12 * nside**2))

    # tests for polarization
    with pytest.raises(ValueError, match="Do not have the right polarization"):
        pspecbeam.PSpecBeamUV.beam_normalized_response(
            beam_nf_dipole, pol="ll", freq=freq
        )

    # test if it is a power beam
    efield_beamfile = DATA_PATH / "HERA_NF_efield.beamfits"
    beam_efield = pspecbeam.PSpecBeamUV(efield_beamfile)
    beam_efield.primary_beam.beam_type = "voltage"
    with pytest.raises(ValueError, match="beam_type must be power"):
        pspecbeam.PSpecBeamUV.beam_normalized_response(beam_efield, pol="xx", freq=freq)

    # test for right axes
    beam_efield.primary_beam.beam_type = "power"
    beam_efield.primary_beam.Naxes_vec = 2
    with pytest.raises(ValueError, match="Expect scalar for power beam"):
        pspecbeam.PSpecBeamUV.beam_normalized_response(beam_efield, pol="xx", freq=freq)

    # test for peak normalization
    beam_efield.primary_beam.Naxes_vec = 1
    beam_efield.primary_beam._data_normalization.value = "area"
    with pytest.raises(ValueError, match="beam must be peak normalized"):
        pspecbeam.PSpecBeamUV.beam_normalized_response(beam_efield, pol="xx", freq=freq)

    # test for the coordinate system
    beam_efield.primary_beam._data_normalization.value = "peak"
    beam_efield.primary_beam.pixel_coordinate_system = "cartesian"
    with pytest.raises(ValueError, match="only healpix format supported"):
        pspecbeam.PSpecBeamUV.beam_normalized_response(beam_efield, pol="xx", freq=freq)

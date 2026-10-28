"""Property-based tests, using hypothesis.

The other test modules check functions on a few hand-picked inputs. The tests
here state a property that must hold for *every* valid input (an identity, a
symmetry, a scaling law, a round trip, ...) and let hypothesis draw many
inputs to try to break it. When a property fails, hypothesis shrinks the input
to a simple failing case and prints it as "Falsifying example: ...".

Inputs are random, so every run tests different cases unless the CI
environment variable is set; then the inputs are derandomized (see the
hypothesis settings profiles in conftest.py). To reproduce a failure, re-run
the test: hypothesis first replays failing examples saved in the local
``.hypothesis/`` database. You can also pass ``--hypothesis-seed=<seed>``
(printed in the pytest header), paste the falsifying example into an
``@example(...)`` decorator, or use the ``@reproduce_failure`` blob that is
printed in CI. ``--hypothesis-show-statistics`` shows how long each test takes.

Hypothesis is a good fit for fast, pure functions and for operations on small
UVPSpec objects. Power spectra of whole files, exact window functions,
bootstrap errors and Monte-Carlo statistics are too slow for it, and are
tested with fixed inputs and seeds elsewhere.
"""

import copy
import functools
import json
import warnings

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pyuvdata.analytic_beam import GaussianBeam
from pyuvdata.utils import POL_NUM2STR_DICT
from uvtools import dspec

from hera_pspec import (
    PSpecBeamGauss,
    PSpecData,
    conversions,
    grouping,
    noise,
    pspecbeam,
    pstokes,
    testing,
    utils,
    uvwindow,
)
from hera_pspec import uvpspec_utils as uvputils

TWO_PI = 2 * np.pi

# Tapers supported by uvtools.dspec.gen_window (and so by PSpecData.set_taper).
TAPERS = ["none", "bh", "bh7", "cs9", "cs11", "blackman", "hann", "tukey"]


def complex_arrays(shape, max_magnitude: float = 1e3):
    return arrays(
        np.complex128,
        shape,
        elements=st.complex_numbers(
            max_magnitude=max_magnitude, allow_nan=False, allow_infinity=False
        ),
    )


@st.composite
def cosmologies(draw, flat: bool = False) -> conversions.Cosmo_Conversions:
    """Cosmologies with Om_M in [0.1, 0.5], H0 in [50, 100] and Om_L in [0, 1],
    so open and closed ones, unless flat=True (Om_L = 1 - Om_M, with the
    curvature either computed or given as exactly zero)."""
    Om_M = draw(st.floats(0.1, 0.5))
    H0 = draw(st.floats(50.0, 100.0))
    if flat:
        Om_k = draw(st.sampled_from([None, 0.0]))
        return conversions.Cosmo_Conversions(Om_L=1 - Om_M, Om_M=Om_M, Om_k=Om_k, H0=H0)
    # astropy's distances divide 0 by 0 for a subnormal Om_L (e.g. 5e-324)
    Om_L = draw(st.one_of(st.just(0.0), st.floats(1e-10, 1.0)))
    return conversions.Cosmo_Conversions(Om_L=Om_L, Om_M=Om_M, H0=H0)


redshifts = st.floats(0.01, 20.0)


class TestCosmology:
    @settings(max_examples=30)
    @given(cosmo=st.one_of(cosmologies(), cosmologies(flat=True)), z=redshifts)
    def test_distances_match_astropy(self, cosmo, z: float) -> None:
        """E(z) and the comoving, transverse comoving and angular diameter
        distances agree with astropy's LambdaCDM for any cosmology and
        redshift, including the sinh/sin branches of the transverse distance
        for open and closed cosmologies."""
        lcdm = cosmo.lcdm
        np.testing.assert_allclose(cosmo.E(z), lcdm.efunc(z), rtol=1e-12)
        for method, astropy_method in [
            ("DC", lcdm.comoving_distance),
            ("DM", lcdm.comoving_transverse_distance),
            ("DA", lcdm.angular_diameter_distance),
        ]:
            np.testing.assert_allclose(
                getattr(cosmo, method)(z, little_h=False),
                astropy_method(z).to_value("Mpc"),
                rtol=1e-8,
            )
        # the angular diameter distance is the transverse comoving distance
        # at the time of emission
        np.testing.assert_allclose(cosmo.DA(z), cosmo.DM(z) / (1 + z), rtol=1e-14)

    @given(cosmo=cosmologies(), z=redshifts, dz=st.floats(1e-3, 5.0))
    def test_comoving_distance_increases_with_redshift(
        self, cosmo, z: float, dz: float
    ) -> None:
        assert cosmo.DC(z + dz) > cosmo.DC(z)

    @given(cosmo=cosmologies(flat=True), z=redshifts)
    def test_flat_transverse_distance_is_comoving_distance(
        self, cosmo, z: float
    ) -> None:
        np.testing.assert_allclose(cosmo.Om_k, 0.0, atol=1e-15)
        np.testing.assert_allclose(cosmo.DM(z), cosmo.DC(z), rtol=1e-10)

    @given(cosmo=cosmologies(), z=redshifts)
    def test_little_h_units(self, cosmo, z: float) -> None:
        """Lengths in h^-1 Mpc are h times lengths in Mpc, so the volume
        conversion X2Y scales as h^3; and the delay to k_para conversion is
        2 pi / dRpara_df in either unit."""
        h = cosmo.h
        for method, power in [("DC", 1), ("DM", 1), ("dRpara_df", 1), ("X2Y", 3)]:
            np.testing.assert_allclose(
                getattr(cosmo, method)(z, little_h=True),
                getattr(cosmo, method)(z, little_h=False) * h**power,
                rtol=1e-10,
            )
        for little_h in [True, False]:
            np.testing.assert_allclose(
                cosmo.tau_to_kpara(z, little_h=little_h)
                * cosmo.dRpara_df(z, little_h=little_h),
                TWO_PI,
                rtol=1e-14,
            )

    @given(z=redshifts, freq=st.floats(50e6, 250e6))
    def test_frequency_redshift_round_trip(self, z: float, freq: float) -> None:
        cosmo = conversions.Cosmo_Conversions()
        np.testing.assert_allclose(cosmo.f2z(cosmo.z2f(z)), z, rtol=1e-12)
        np.testing.assert_allclose(cosmo.z2f(cosmo.f2z(freq)), freq, rtol=1e-14)
        np.testing.assert_allclose(
            cosmo.f2z(cosmo.z2f(z, ghz=True), ghz=True), z, rtol=1e-12
        )

    @given(
        Om_L=st.floats(0.0, 1.0), Om_M=st.floats(0.1, 0.5), H0=st.floats(50.0, 100.0)
    )
    def test_density_parameters_add_up(
        self, Om_L: float, Om_M: float, H0: float
    ) -> None:
        """With Om_M given, the baryon and cold dark matter densities add up to
        it, and the curvature makes the densities add up to one."""
        cosmo = conversions.Cosmo_Conversions(Om_L=Om_L, Om_M=Om_M, H0=H0)
        np.testing.assert_allclose(cosmo.Om_b + cosmo.Om_c, Om_M, rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            cosmo.Om_L + cosmo.Om_M + cosmo.Om_k, 1.0, rtol=1e-14
        )


class TestNoise:
    @given(
        scalar=st.floats(1e6, 1e12),
        Tsys=st.floats(1.0, 1e4),
        t_int=st.floats(0.1, 1e4),
        Ncoherent=st.integers(1, 1000),
        Nincoherent=st.integers(1, 1000),
        factor=st.floats(0.1, 10.0),
        n=st.integers(1, 100),
    )
    def test_noise_power_scaling(
        self,
        scalar: float,
        Tsys: float,
        t_int: float,
        Ncoherent: int,
        Nincoherent: int,
        factor: float,
        n: int,
    ) -> None:
        """The thermal noise power P_N = scalar Tsys^2 / (t_int Ncoherent
        sqrt(Nincoherent)) of Cheng et al. (2018), with Tsys converted to mK.

        The visibility noise variance scales as Tsys^2 / t_int, coherent
        averaging of n visibilities is the same as an n times longer
        integration, and incoherent averaging of n power spectra reduces their
        rms by sqrt(n). The real and imaginary parts of the power each have
        1/sqrt(2) of the rms of its absolute value.
        """
        kwargs = dict(
            scalar=scalar,
            Tsys=Tsys,
            t_int=t_int,
            Ncoherent=Ncoherent,
            Nincoherent=Nincoherent,
        )
        P_N = noise.calc_P_N(**kwargs, component="abs")
        np.testing.assert_allclose(
            P_N,
            scalar * (1e3 * Tsys) ** 2 / (t_int * Ncoherent * np.sqrt(Nincoherent)),
            rtol=1e-12,
        )

        def scaled(**changes):
            return noise.calc_P_N(**{**kwargs, **changes}, component="abs") / P_N

        np.testing.assert_allclose(scaled(Tsys=factor * Tsys), factor**2, rtol=1e-12)
        np.testing.assert_allclose(scaled(t_int=factor * t_int), 1 / factor, rtol=1e-12)
        np.testing.assert_allclose(scaled(scalar=factor * scalar), factor, rtol=1e-12)
        np.testing.assert_allclose(scaled(Ncoherent=n * Ncoherent), 1 / n, rtol=1e-12)
        np.testing.assert_allclose(
            scaled(t_int=n * t_int, Ncoherent=Ncoherent), 1 / n, rtol=1e-12
        )
        np.testing.assert_allclose(
            scaled(Nincoherent=n * Nincoherent), 1 / np.sqrt(n), rtol=1e-12
        )

        real = noise.calc_P_N(**kwargs, component="real")
        np.testing.assert_allclose(noise.calc_P_N(**kwargs, component="imag"), real)
        np.testing.assert_allclose(P_N / real, np.sqrt(2), rtol=1e-12)
        # no incoherent averaging is the same as a single power spectrum
        np.testing.assert_allclose(
            noise.calc_P_N(scalar, Tsys, t_int, Nincoherent=None),
            noise.calc_P_N(scalar, Tsys, t_int, Nincoherent=1),
        )

    @given(
        k=arrays(np.float64, st.integers(1, 20), elements=st.floats(1e-3, 10.0)),
        Tsys=st.floats(1.0, 1e4),
    )
    def test_noise_delta_squared(self, k: np.ndarray, Tsys: float) -> None:
        """Delta^2(k) = k^3 P(k) / (2 pi^2)."""
        pk = noise.calc_P_N(1e9, Tsys, 10.0)
        np.testing.assert_allclose(
            noise.calc_P_N(1e9, Tsys, 10.0, form="DelSq", k=k),
            pk * k**3 / (2 * np.pi**2),
            rtol=1e-12,
        )

    @given(taper=st.sampled_from(TAPERS), n=st.integers(1, 64))
    def test_delay_correlation_matrix(self, taper: str, n: int) -> None:
        """The approximate delay-delay correlation matrix is a correlation
        matrix (symmetric, unit diagonal, |c| <= 1 and positive semi-definite)
        for any taper and number of delays, and the identity without a taper,
        for which the delay modes of white noise are uncorrelated."""
        # Some tapers are identically zero for two channels (they vanish at
        # both ends), and then the correlations are undefined.
        assume(np.any(dspec.gen_window(taper, n) != 0))
        corr = noise.get_approximate_delay_delay_corr_matrix(taper, n)
        assert corr.shape == (n, n)
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-12)
        np.testing.assert_allclose(corr, corr.T, rtol=0, atol=1e-12)
        assert np.all(np.abs(corr) <= 1 + 1e-12)
        assert np.all(np.linalg.eigvalsh(corr) >= -1e-10)
        if taper == "none":
            np.testing.assert_allclose(corr, np.eye(n), rtol=0, atol=1e-12)


class ConstantVolume:
    """A stand-in cosmology whose X2Y (sr Hz -> comoving volume) is the same
    at all frequencies, so that the pspec scalar only depends on the band."""

    X2Y_VALUE = 500.0

    def f2z(self, freq):
        return np.zeros_like(freq)

    def X2Y(self, z, little_h=True):
        return self.X2Y_VALUE


bands = st.tuples(
    st.integers(4, 128),  # number of channels
    st.floats(10e3, 1e6),  # channel width in Hz
    st.floats(100e6, 200e6),  # lowest frequency in Hz
)


def band_freqs(band) -> np.ndarray:
    nfreqs, df, f0 = band
    return f0 + df * np.arange(nfreqs)


class TestBeamScalar:
    BEAM_FREQS = np.linspace(50e6, 400e6, 36)

    @settings(max_examples=20)
    @given(
        band=bands,
        ratio=arrays(np.float64, 36, elements=st.floats(1.0, 100.0)),
        factor=st.floats(0.1, 10.0),
    )
    def test_scalar_is_inversely_proportional_to_omega_ratio(
        self, band, ratio: np.ndarray, factor: float
    ) -> None:
        """The scalar is the inverse of a band integral of
        Omega_pp / Omega_p^2, so scaling that beam ratio by a factor divides
        the scalar by it."""
        cosmo = conversions.Cosmo_Conversions()
        freqs = band_freqs(band)
        kwargs = dict(num_steps=100)
        scalar = pspecbeam._compute_pspec_scalar(
            cosmo, self.BEAM_FREQS, ratio, freqs, **kwargs
        )
        scaled = pspecbeam._compute_pspec_scalar(
            cosmo, self.BEAM_FREQS, factor * ratio, freqs, **kwargs
        )
        np.testing.assert_allclose(scaled, scalar / factor, rtol=1e-12)

    @settings(max_examples=20)
    @given(band=bands, cosmo=cosmologies(), fwhm=st.floats(0.05, 1.0))
    def test_scalar_little_h(self, band, cosmo, fwhm: float) -> None:
        """A volume in h^-3 Mpc^3 is h^3 times the same volume in Mpc^3."""
        beam = PSpecBeamGauss(fwhm, self.BEAM_FREQS, cosmo=cosmo)
        nfreqs, df, f0 = band
        kwargs = dict(num_steps=100)
        in_mpc = beam.compute_pspec_scalar(
            f0, f0 + nfreqs * df, nfreqs, little_h=False, **kwargs
        )
        in_mpc_h = beam.compute_pspec_scalar(
            f0, f0 + nfreqs * df, nfreqs, little_h=True, **kwargs
        )
        np.testing.assert_allclose(in_mpc_h / in_mpc, cosmo.h**3, rtol=1e-10)

    @settings(max_examples=30)
    @given(band=bands, ratio=st.floats(1.0, 100.0), taper=st.sampled_from(TAPERS))
    def test_scalar_bandwidth_and_taper(self, band, ratio: float, taper: str) -> None:
        """For a constant X2Y and beam ratio Omega_pp / Omega_p^2, the scalar is
        X2Y * B / ratio with bandwidth B = N * channel width, and the noise
        scalar is X2Y / ratio. A taper w reduces the effective bandwidth by
        mean(w^2), so it increases the scalar by 1 / mean(w^2). The taper is
        interpolated onto the integration grid by nearest neighbour, hence the
        tolerance."""
        nfreqs, df, f0 = band
        freqs = band_freqs(band)
        args = (ConstantVolume(), self.BEAM_FREQS, np.full(36, ratio), freqs)
        kwargs = dict(num_steps=5000)
        X2Y = ConstantVolume.X2Y_VALUE

        untapered = pspecbeam._compute_pspec_scalar(*args, **kwargs)
        np.testing.assert_allclose(untapered, X2Y * nfreqs * df / ratio, rtol=1e-10)
        noise_scalar = pspecbeam._compute_pspec_scalar(
            *args, noise_scalar=True, **kwargs
        )
        np.testing.assert_allclose(noise_scalar, X2Y / ratio, rtol=1e-10)

        tapered = pspecbeam._compute_pspec_scalar(*args, taper=taper, **kwargs)
        window = dspec.gen_window(taper, nfreqs)
        np.testing.assert_allclose(
            tapered / untapered, 1 / np.mean(window**2), rtol=2e-3
        )

    @settings(max_examples=20)
    @given(fwhm=st.floats(0.01, 1.5), sigma_type=st.sampled_from(["power", "efield"]))
    def test_gaussian_beam_fwhm(self, fwhm: float, sigma_type: str) -> None:
        """fwhm is the full width at half maximum of the power beam of the
        pyuvdata GaussianBeam that PSpecBeamGauss holds. Wrapping a
        GaussianBeam with that power beam, given by the sigma of either its
        power or its E-field beam (which is sqrt(2) wider), gives back fwhm
        and the same beam integrals."""
        beam = PSpecBeamGauss(fwhm, self.BEAM_FREQS)
        power = beam.analytic_beam.power_eval(
            az_array=np.zeros(2),
            za_array=np.array([0.0, fwhm / 2]),
            freq_array=self.BEAM_FREQS[:1],
        )
        # (axis, polarization, frequency, direction); xx
        np.testing.assert_allclose(power[0, 0, 0].real, [1.0, 0.5], rtol=1e-12)

        sigma = fwhm / np.sqrt(8 * np.log(2))  # of the power beam
        if sigma_type == "efield":
            sigma *= np.sqrt(2)
        wrapped = PSpecBeamGauss(
            beam=GaussianBeam(sigma=sigma, sigma_type=sigma_type),
            beam_freqs=self.BEAM_FREQS,
        )
        np.testing.assert_allclose(wrapped.fwhm, fwhm, rtol=1e-12)
        np.testing.assert_allclose(
            wrapped.power_beam_int(), beam.power_beam_int(), rtol=1e-12
        )
        np.testing.assert_allclose(
            wrapped.power_beam_sq_int(), beam.power_beam_sq_int(), rtol=1e-12
        )


@st.composite
def ft_beams(draw) -> uvwindow.FTBeam:
    """Small FTBeams with random data, on increasing frequencies whose
    spacings differ by up to a factor of three."""
    nfreqs = draw(st.integers(2, 8))
    npix = draw(st.integers(1, 4))
    df = draw(st.floats(50e3, 1e6))
    steps = draw(arrays(np.float64, nfreqs - 1, elements=st.floats(0.5, 1.5)))
    freqs = draw(st.floats(100e6, 200e6)) + df * np.concatenate([[0], np.cumsum(steps)])
    data = draw(arrays(np.float64, (nfreqs, npix, npix), elements=st.floats(-1e3, 1e3)))
    pol = draw(st.sampled_from(["pI", "xx", "yy"]))
    return uvwindow.FTBeam(data, pol, freqs, mapsize=draw(st.floats(0.5, 2.0)))


class TestFTBeam:
    @pytest.mark.parametrize(
        "half_plane",
        [
            "upper",
            pytest.param(
                "lower",
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="arccos(x / za) only gives azimuths in [0, pi], so "
                    "(x, y) with y < 0 is mapped to the azimuth of (x, -y), and "
                    "FTBeam.from_beam evaluates the beam at the mirror image of "
                    "every point of the lower half of its map (see #578).",
                ),
            ),
        ],
    )
    @given(
        az=arrays(np.float64, st.integers(1, 10), elements=st.floats(0, np.pi)),
        # x^2 + y^2 underflows for za below ~1e-154 rad, which no beam map
        # resolves; za = 0 is the zenith
        za=st.one_of(st.just(0.0), st.floats(1e-100, np.pi / 2)),
    )
    @example(az=np.array([np.pi / 2]), za=1.0)
    def test_cartesian_to_spherical_round_trip(
        self, half_plane: str, az: np.ndarray, za: float
    ) -> None:
        """_cartesian_to_spherical inverts x = za cos(az), y = za sin(az).
        arccos loses precision for azimuths close to 0 and pi, hence the
        tolerance (1e-8 rad is far below the resolution of any beam map)."""
        if half_plane == "lower":
            az = TWO_PI - az
        x, y = za * np.cos(az), za * np.sin(az)
        az_out, za_out = uvwindow._cartesian_to_spherical(x, y)
        np.testing.assert_allclose(za_out, za, rtol=1e-15)
        np.testing.assert_allclose(za_out * np.cos(az_out), x, rtol=0, atol=1e-15)
        np.testing.assert_allclose(za_out * np.sin(az_out), y, rtol=0, atol=1e-7)

    @given(beam=ft_beams(), data=st.data())
    def test_select_native_freqs(self, beam: uvwindow.FTBeam, data) -> None:
        """Selecting some of the beam's own frequencies, in any order and
        possibly repeated, extracts those channels; selecting all of them
        leaves the beam unchanged. With inplace=False the beam itself is not
        modified."""
        original = copy.deepcopy(beam)
        nfreqs = beam.freq_array.size
        idx = data.draw(st.lists(st.integers(0, nfreqs - 1), min_size=2, max_size=10))
        sub = beam.select_freqs(beam.freq_array[idx], inplace=False)
        assert beam == original
        np.testing.assert_array_equal(sub.freq_array, beam.freq_array[idx])
        np.testing.assert_array_equal(sub.ft_beam, beam.ft_beam[idx])
        assert (sub.pol, sub.mapsize) == (beam.pol, beam.mapsize)

        assert beam.select_freqs(beam.freq_array, inplace=False) == beam
        assert beam.select_freqs(beam.freq_array.copy()) is None
        assert beam == original

    @given(beam=ft_beams(), data=st.data())
    def test_select_freqs_interpolates_linearly(
        self, beam: uvwindow.FTBeam, data
    ) -> None:
        """Between two channels, the FT of the beam is interpolated linearly
        in frequency. So a fraction t of the way from channel i to channel
        i + 1 it is (1 - t) A_i + t A_(i+1), and a beam that is linear in
        frequency is reproduced exactly at any frequency of the band."""
        freqs = beam.freq_array
        nfreqs = freqs.size
        i = np.array(
            data.draw(st.lists(st.integers(0, nfreqs - 2), min_size=2, max_size=10))
        )
        # away from the channels, so that every frequency is interpolated
        t = data.draw(arrays(np.float64, i.size, elements=st.floats(0.01, 0.99)))
        targets = (1 - t) * freqs[i] + t * freqs[i + 1]
        with pytest.warns(UserWarning, match="interpolating"):
            interp = beam.select_freqs(targets, inplace=False)
        np.testing.assert_array_equal(interp.freq_array, targets)
        w = ((targets - freqs[i]) / (freqs[i + 1] - freqs[i]))[:, None, None]
        np.testing.assert_allclose(
            interp.ft_beam,
            (1 - w) * beam.ft_beam[i] + w * beam.ft_beam[i + 1],
            rtol=1e-12,
            atol=1e-12 * np.abs(beam.ft_beam).max(),
        )

        # any frequencies, whether or not they coincide with channels
        targets = np.array(
            data.draw(
                st.lists(st.floats(freqs.min(), freqs.max()), min_size=2, max_size=10)
            )
        )
        A, B = beam.ft_beam[0], beam.ft_beam[-1]
        span = freqs[-1] - freqs[0]
        beam.ft_beam = A + ((freqs - freqs[0]) / span)[:, None, None] * (B - A)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Requested frequencies do not coincide")
            interp = beam.select_freqs(targets, inplace=False)
        # If all frequencies are within 1e-3 channel widths of a channel (the
        # default atol), the channels are extracted instead.
        snap = 1e-3 * np.median(np.diff(freqs)) / span * np.abs(B - A).max()
        np.testing.assert_allclose(
            interp.ft_beam,
            A + ((targets - freqs[0]) / span)[:, None, None] * (B - A),
            rtol=0,
            atol=snap + 1e-10 * (1 + np.abs(beam.ft_beam).max()),
        )

    @given(beam=ft_beams(), data=st.data())
    def test_equality(self, beam: uvwindow.FTBeam, data) -> None:
        """FTBeams are equal if and only if their polarization, map size,
        frequencies and FT beams are; a copy is equal to the original."""
        other = copy.deepcopy(beam)
        assert beam == beam
        assert beam == other and other == beam
        assert beam.__eq__(beam.ft_beam) is NotImplemented

        change = data.draw(st.sampled_from(["pol", "mapsize", "freq", "pixel"]))
        if change == "pol":
            other.pol = data.draw(
                st.sampled_from(["pI", "xx", "yy"]).filter(lambda pol: pol != beam.pol)
            )
        elif change == "mapsize":
            other.mapsize *= 2
        elif change == "freq":
            other.freq_array[data.draw(st.integers(0, beam.freq_array.size - 1))] += 1.0
        else:
            index = tuple(data.draw(st.integers(0, n - 1)) for n in beam.ft_beam.shape)
            other.ft_beam[index] += 1.0
        assert beam != other and other != beam


# Baseline and baseline-pair integers hold each antenna number + 100 in three
# decimal digits, so they can encode antenna numbers 0 to 899.
antnums = st.integers(0, 899)
antpairs = st.tuples(antnums, antnums)
POL_STRINGS = list(POL_NUM2STR_DICT.values())


class TestIntegerEncodings:
    @given(bl1=antpairs, bl2=antpairs)
    def test_blpair_round_trip(self, bl1, bl2) -> None:
        blpair = uvputils._antnums_to_blpair((bl1, bl2))
        assert uvputils._blpair_to_antnums(blpair) == (bl1, bl2)
        assert uvputils._bl_to_antnums(uvputils._antnums_to_bl(bl1)) == bl1
        assert uvputils._blpair_to_bls(blpair) == (
            uvputils._antnums_to_bl(bl1),
            uvputils._antnums_to_bl(bl2),
        )

    @given(a=st.tuples(antpairs, antpairs), b=st.tuples(antpairs, antpairs))
    def test_blpair_order_is_antenna_order(self, a, b) -> None:
        """Sorting baseline-pair integers sorts them by antenna numbers."""
        assert (uvputils._antnums_to_blpair(a) < uvputils._antnums_to_blpair(b)) == (
            a < b
        )

    @given(bl1=antpairs, bl2=antpairs)
    def test_conjugation(self, bl1, bl2) -> None:
        """Conjugating swaps antennas (or baselines), so doing it twice is the
        identity; conjugating both baselines conjugates each in turn."""
        blpair = uvputils._antnums_to_blpair((bl1, bl2))
        bl = uvputils._antnums_to_bl(bl1)
        assert uvputils._conj_bl_int(bl) == uvputils._antnums_to_bl(bl1[::-1])
        assert uvputils._conj_bl_int(uvputils._conj_bl_int(bl)) == bl
        assert uvputils._conj_blpair_int(blpair) == uvputils._antnums_to_blpair(
            (bl2, bl1)
        )
        assert uvputils._conj_blpair_int(uvputils._conj_blpair_int(blpair)) == blpair
        for which in ["first", "second", "both"]:
            conj = uvputils._conj_blpair(blpair, which=which)
            assert uvputils._conj_blpair(conj, which=which) == blpair
        first_then_second = uvputils._conj_blpair(
            uvputils._conj_blpair(blpair, which="first"), which="second"
        )
        assert first_then_second == uvputils._conj_blpair(blpair, which="both")

    @given(pol1=st.sampled_from(POL_STRINGS), pol2=st.sampled_from(POL_STRINGS))
    def test_polpair_round_trip(self, pol1: str, pol2: str) -> None:
        polpair = uvputils.polpair_tuple2int((pol1, pol2))
        assert uvputils.polpair_int2tuple(polpair, pol_strings=True) == (pol1, pol2)
        pol_ints = uvputils.polpair_int2tuple(polpair)
        assert uvputils.polpair_tuple2int(pol_ints) == polpair

    @given(data=st.data())
    def test_r_params_round_trip(self, data) -> None:
        """Compressing r_params (which stores each distinct set of parameters
        once, with the list of baselines using it) loses nothing, and
        compressing the decompressed r_params gives the same parameter sets.
        Empty r_params compress to an empty string."""
        params = data.draw(
            st.lists(
                st.dictionaries(
                    st.sampled_from(
                        ["filter_centers", "filter_half_widths", "filter_factors"]
                    ),
                    st.lists(
                        st.floats(allow_nan=False, allow_infinity=False), max_size=3
                    ),
                ),
                min_size=1,
                max_size=3,
            )
        )
        keys = st.tuples(antnums, antnums, st.sampled_from(["xx", "yy", "pI"]))
        r_params = data.draw(
            st.dictionaries(keys, st.sampled_from(range(len(params))), max_size=6)
        )
        r_params = {key: params[i] for key, i in r_params.items()}
        compressed = uvputils.compress_r_params(r_params)
        if not r_params:
            assert compressed == ""
        assert isinstance(json.loads(compressed or "{}"), dict)
        decompressed = uvputils.decompress_r_params(compressed)
        assert decompressed == r_params
        assert json.loads(uvputils.compress_r_params(decompressed) or "{}") == (
            json.loads(compressed or "{}")
        )


@st.composite
def visibility_arrays(draw, n: int, max_magnitude: float = 1e6):
    """n complex arrays with the same (Nblts, Nfreqs, 1) shape."""
    shape = draw(st.tuples(st.integers(1, 4), st.integers(1, 8), st.just(1)))
    return [draw(complex_arrays(shape, max_magnitude)) for _ in range(n)]


def combine(pol1, pol2, pstokes_str, data, convention="avg"):
    return pstokes._combine_pol_arrays(
        pol1, pol2, pstokes_str, pol_convention=convention, data_list=data
    )[0]


def assert_close_to(actual, expected, *inputs):
    """Equality up to the rounding error of sums of the inputs."""
    scale = max([1.0] + [np.abs(x).max() for x in inputs])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13 * scale)


class TestPseudoStokes:
    @given(
        arrs=visibility_arrays(4),
        a=st.complex_numbers(max_magnitude=10, allow_nan=False, allow_infinity=False),
        pstokes_str=st.sampled_from(["pI", "pQ", "pU", "pV"]),
        convention=st.sampled_from(["avg", "sum"]),
    )
    def test_linear(self, arrs, a: complex, pstokes_str: str, convention) -> None:
        """Pseudo-Stokes visibilities are linear in the input visibilities."""
        d1, d2, e1, e2 = arrs
        pols = {"pI": ("xx", "yy"), "pQ": ("xx", "yy")}.get(pstokes_str, ("xy", "yx"))
        combined = combine(*pols, pstokes_str, [a * d1 + e1, a * d2 + e2], convention)
        expected = a * combine(*pols, pstokes_str, [d1, d2], convention) + combine(
            *pols, pstokes_str, [e1, e2], convention
        )
        assert_close_to(combined, expected, 10 * d1, 10 * d2, e1, e2)

    @given(arrs=visibility_arrays(2))
    def test_recover_linear_polarizations(self, arrs) -> None:
        """With the 'avg' convention, pI = (XX + YY) / 2 and
        pQ = (XX - YY) / 2, so XX = pI + pQ and YY = pI - pQ. The 'sum'
        convention gives twice the 'avg' pseudo-Stokes visibilities."""
        xx, yy = arrs
        pI = combine("xx", "yy", "pI", [xx, yy])
        pQ = combine("xx", "yy", "pQ", [xx, yy])
        assert_close_to(pI + pQ, xx, xx, yy)
        assert_close_to(pI - pQ, yy, xx, yy)
        assert_close_to(combine("xx", "yy", "pI", [xx, yy], convention="sum"), 2 * pI)

    @given(arrs=visibility_arrays(2))
    def test_recover_stokes_u_and_v(self, arrs) -> None:
        """For a sky with Stokes U and V visibilities, XY = U + iV and
        YX = U - iV, and pU, pV recover U and V."""
        U, V = arrs
        xy, yx = U + 1j * V, U - 1j * V
        assert_close_to(combine("xy", "yx", "pU", [xy, yx]), U, U, V)
        assert_close_to(combine("xy", "yx", "pV", [xy, yx]), V, U, V)
        assert_close_to(
            combine("xy", "yx", "pV", [xy, yx], convention="sum"),
            2 * combine("xy", "yx", "pV", [xy, yx]),
        )

    @given(
        nsamples=st.lists(
            st.one_of(st.just(0.0), st.floats(1e-3, 1e3)), min_size=2, max_size=20
        ).flatmap(lambda n1: st.tuples(st.just(n1), st.permutations(n1))),
        pstokes_str=st.sampled_from(["pI", "pQ"]),
    )
    def test_combined_nsamples(self, nsamples, pstokes_str: str) -> None:
        """The pseudo-Stokes nsample n = 4 / (1/n1 + 1/n2) is twice the
        harmonic mean of the input nsamples (see hera_pspec issue #406): it is
        symmetric, 2n for n1 = n2 = n, between 2 min(n1, n2) and
        2 max(n1, n2), and zero if either input has no samples."""
        n1 = np.array(nsamples[0])[None, :, None]
        n2 = np.array(nsamples[1])[None, :, None]

        def combined(a, b):
            return pstokes._combine_pol_arrays(
                "xx", "yy", pstokes_str, nsamples_list=[a, b]
            )[2]

        n = combined(n1, n2)
        np.testing.assert_array_equal(n, combined(n2, n1))
        np.testing.assert_allclose(combined(n1, n1), 2 * n1, rtol=1e-14)
        valid = (n1 > 0) & (n2 > 0)
        np.testing.assert_array_equal(n[~valid], 0)
        lo, hi = np.minimum(n1, n2)[valid], np.maximum(n1, n2)[valid]
        assert np.all(n[valid] >= 2 * lo * (1 - 1e-14))
        assert np.all(n[valid] <= 2 * hi * (1 + 1e-14))


angle_arrays = arrays(np.float64, st.integers(1, 20), elements=st.floats(-20.0, 20.0))


def angle_difference(a, b):
    """The difference of two angles, wrapped to [-pi, pi)."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % TWO_PI - np.pi


class TestUtils:
    @given(
        nfreqs=st.integers(2, 512),
        df=st.floats(1e3, 1e6),
        f0=st.floats(50e6, 250e6),
        data=st.data(),
    )
    def test_get_delays(self, nfreqs: int, df: float, f0: float, data) -> None:
        """The delays of n_dlys bins are sorted, spaced by 1 / (n_dlys df),
        include zero, and are symmetric about zero apart from the single most
        negative delay for even n_dlys, as for numpy's fftfreq."""
        n_dlys = data.draw(st.one_of(st.none(), st.integers(1, nfreqs)))
        freqs = f0 + df * np.arange(nfreqs)
        delays = utils.get_delays(freqs, n_dlys=n_dlys)
        n = nfreqs if n_dlys is None else n_dlys
        assert delays.shape == (n,)
        np.testing.assert_allclose(np.diff(delays), 1 / (n * df), rtol=1e-8)
        assert delays[n // 2] == 0
        symmetric = delays[n % 2 == 0 :]
        np.testing.assert_allclose(symmetric, -symmetric[::-1], rtol=1e-12)
        np.testing.assert_allclose(
            delays, np.fft.fftshift(np.fft.fftfreq(n, d=df)), rtol=1e-8
        )

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="np.angle(z) % (2 pi) rounds to 2 pi for angles just below a "
        "multiple of 2 pi, e.g. for the single angle 2 pi or -1e-96 (see #574).",
    )
    @given(angles=angle_arrays)
    @example(angles=np.array([TWO_PI]))
    def test_circular_average_range(self, angles: np.ndarray) -> None:
        """The circular mean is documented to lie in [0, 2 pi)."""
        avg = utils.circular_average(angles)
        assert 0 <= avg < TWO_PI

    @given(angles=angle_arrays, data=st.data())
    def test_circular_average_wraps(self, angles: np.ndarray, data) -> None:
        """Adding a multiple of 2 pi to an angle does not change the average,
        and rotating all angles rotates the average (when the average is well
        defined, i.e. the angles do not cancel out)."""
        resultant = np.abs(np.mean(np.exp(1j * angles)))
        assume(resultant > 1e-3)
        tol = 1e-12 * (1 + np.abs(angles).max()) / resultant
        avg = utils.circular_average(angles)

        i = data.draw(st.integers(0, len(angles) - 1))
        shifted = angles.copy()
        shifted[i] += TWO_PI * data.draw(st.integers(-3, 3))
        assert abs(angle_difference(utils.circular_average(shifted), avg)) < 10 * tol

        phi = data.draw(st.floats(-10.0, 10.0))
        rotated = utils.circular_average(angles + phi)
        assert abs(angle_difference(rotated, avg + phi)) < 10 * tol

    @given(shape=st.tuples(st.integers(1, 4), st.integers(2, 12)), data=st.data())
    def test_cov(self, shape, data) -> None:
        """With unit weights, cov is the (biased) sample covariance, which is
        Hermitian and positive semi-definite. Rescaling all weights, or
        appending samples with zero weight, does not change any covariance."""
        d = data.draw(complex_arrays(shape, 100.0))
        d2 = data.draw(complex_arrays(shape, 100.0))
        w = data.draw(arrays(np.float64, shape, elements=st.floats(0.1, 10.0)))
        scale = 1.0 + np.abs(d).max() ** 2 + np.abs(d2).max() ** 2
        atol = 1e-12 * scale

        ones = np.ones(shape)
        C = utils.cov(d, ones)
        np.testing.assert_allclose(C, np.cov(d, bias=True), rtol=0, atol=atol)
        np.testing.assert_allclose(C, C.conj().T, rtol=0, atol=atol)
        assert np.all(np.linalg.eigvalsh(C) >= -atol * shape[0])

        # the cross-covariance of d2 and d is the conjugate transpose of the
        # cross-covariance of d and d2
        np.testing.assert_allclose(
            utils.cov(d2, w, d, w), utils.cov(d, w, d2, w).conj().T, rtol=0, atol=atol
        )

        C = utils.cov(d, w, d2, w)
        factor = data.draw(st.floats(0.1, 10.0))
        np.testing.assert_allclose(
            utils.cov(d, factor * w, d2, factor * w), C, rtol=0, atol=atol
        )
        pad = np.zeros((shape[0], 3))
        np.testing.assert_allclose(
            utils.cov(
                np.hstack([d, pad + 1]),
                np.hstack([w, pad]),
                np.hstack([d2, pad - 1]),
                np.hstack([w, pad]),
            ),
            C,
            rtol=0,
            atol=atol,
        )

    @given(data=st.data())
    def test_q_alt_is_a_fourier_mode(self, data) -> None:
        """Q_alt for delay mode a is the outer product conj(m) m^T of the
        Fourier mode m_j = exp(-2 pi i tau_a nu_j), where tau_a is delay a
        of get_delays. So it is Hermitian and rank one with trace N, and
        x^dagger Q_alt y is conj(x~) y~, with x~ and y~ the Fourier transforms
        of x and y at that delay. With one delay per channel the modes are
        those of the FFT, and they add up to N times the identity."""
        n_freqs = data.draw(st.integers(2, 40))
        n_delays = data.draw(st.one_of(st.just(n_freqs), st.integers(1, n_freqs)))
        mode = data.draw(st.integers(0, n_delays - 1))
        allow_fft = data.draw(st.booleans())
        Q = utils.get_Q_alt(mode, n_delays, n_freqs, allow_fft=allow_fft)

        df = 1e5  # any channel width
        nu = df * np.arange(n_freqs)
        tau = utils.get_delays(nu, n_dlys=n_delays)[mode]
        fourier_mode = np.exp(-2j * np.pi * tau * nu)
        np.testing.assert_allclose(
            Q, np.outer(fourier_mode.conj(), fourier_mode), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(Q, Q.conj().T, rtol=0, atol=1e-12)
        assert np.linalg.matrix_rank(Q) == 1
        np.testing.assert_allclose(np.trace(Q), n_freqs, rtol=1e-12)

        x = data.draw(complex_arrays(n_freqs, 10.0))
        y = data.draw(complex_arrays(n_freqs, 10.0))
        np.testing.assert_allclose(
            x.conj() @ Q @ y,
            np.conj(fourier_mode @ x) * (fourier_mode @ y),
            rtol=0,
            atol=1e-10 * n_freqs**2,
        )

        if n_delays == n_freqs:
            np.testing.assert_allclose(
                Q,
                utils.get_Q_alt(mode, n_delays, n_freqs, allow_fft=not allow_fft),
                rtol=0,
                atol=1e-12,
            )
            total = sum(
                utils.get_Q_alt(a, n_delays, n_freqs, allow_fft=allow_fft)
                for a in range(n_delays)
            )
            np.testing.assert_allclose(total, n_freqs * np.eye(n_freqs), atol=1e-10)


baseline_lists = st.lists(
    st.tuples(st.integers(0, 350), st.integers(0, 350)),
    unique=True,
    min_size=1,
    max_size=200,
)


class TestBaselineGroups:
    @given(
        bls=baseline_lists,
        data=st.data(),
        keep_remainder=st.booleans(),
        randomize=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_group_baselines(
        self, bls, data, keep_remainder: bool, randomize: bool, seed: int
    ) -> None:
        """Groups are disjoint and of equal size N // Ngroups. The remainder is
        discarded, or added to the last group if keep_remainder=True; then the
        groups contain every input baseline exactly once. Without
        randomization, the baselines stay in their input order."""
        ngroups = data.draw(st.integers(1, len(bls)))
        original = list(bls)
        groups = grouping.group_baselines(
            bls, ngroups, keep_remainder=keep_remainder, randomize=randomize, seed=seed
        )
        assert bls == original
        assert len(groups) == ngroups
        size = len(bls) // ngroups
        remainder = len(bls) - size * ngroups if keep_remainder else 0
        assert [len(g) for g in groups] == [size] * (ngroups - 1) + [size + remainder]

        grouped = [bl for g in groups for bl in g]
        assert len(set(grouped)) == len(grouped)
        assert set(grouped) <= set(bls)
        if keep_remainder:
            assert set(grouped) == set(bls)
        if not randomize:
            assert grouped == bls[: len(grouped)]
        if randomize:
            again = grouping.group_baselines(
                bls, ngroups, keep_remainder=keep_remainder, randomize=True, seed=seed
            )
            assert again == groups

    @given(bls=baseline_lists, grouped=st.booleans(), seed=st.integers(0, 2**32 - 1))
    def test_sample_baselines(self, bls, grouped: bool, seed: int) -> None:
        """A bootstrap sample draws, with replacement, as many baselines (or
        groups of baselines) as there are in the input."""
        if grouped:
            bls = grouping.group_baselines(bls, max(1, len(bls) // 3))
        sample = grouping.sample_baselines(bls, seed=seed)
        assert len(sample) == len(bls)
        assert all(s in bls for s in sample)
        assert grouping.sample_baselines(bls, seed=seed) == sample


@st.composite
def positive_definite_matrices(draw, n: int):
    """A A^T + eps I for a random n x n matrix A."""
    A = draw(arrays(np.float64, (n, n), elements=st.floats(-1.0, 1.0)))
    eps = draw(st.floats(1e-6, 1.0))
    return A @ A.T + eps * np.eye(n)


def assert_rows_sum_to_one(W):
    """Unit row sums, up to the rounding error of each sum."""
    np.testing.assert_array_less(
        np.abs(W.sum(axis=1) - 1), 1e-10 * (1 + np.abs(W).sum(axis=1))
    )


class TestNormalization:
    ds = PSpecData()

    @given(data=st.data(), n=st.integers(2, 16))
    def test_identity_normalization(self, data, n: int) -> None:
        """For norm='I', M and the window functions W are H and G with their
        rows divided by their sums, so that the rows of W and of M G sum to
        one."""
        G = data.draw(positive_definite_matrices(n))
        H = data.draw(positive_definite_matrices(n))
        M, W = self.ds.get_MW(G, H, mode="I")
        np.testing.assert_array_equal(M, np.diag(np.diag(M)))
        assert_rows_sum_to_one(W)
        assert_rows_sum_to_one(M @ G)

    @given(data=st.data(), n=st.integers(2, 16))
    def test_inverse_normalization(self, data, n: int) -> None:
        """For norm='H^-1', M = H^-1 decorrelates the band powers, so the
        window functions are the identity."""
        G = data.draw(positive_definite_matrices(n))
        H = data.draw(positive_definite_matrices(n))
        # Inverting a nearly singular matrix only gives numerical noise, and
        # the rounding error of an inverse grows with the condition number.
        cond = np.linalg.cond(H)
        assume(cond < 1e8)
        M, W = self.ds.get_MW(G, H, mode="H^-1")
        assert_rows_sum_to_one(W)
        np.testing.assert_allclose(W, np.eye(n), rtol=0, atol=1e-10 * cond)

    @given(data=st.data(), n=st.integers(2, 16))
    def test_inverse_sqrt_normalization(self, data, n: int) -> None:
        """For norm='V^-1/2', M is V^-1/2 with normalized rows, so the
        normalized band powers M q have the diagonal covariance M V M^T, and
        the window functions are W = M H with unit row sums."""
        G = data.draw(positive_definite_matrices(n))
        H = data.draw(positive_definite_matrices(n))
        V = data.draw(positive_definite_matrices(n))
        cond = np.linalg.cond(V)
        assume(cond < 1e8)
        M, W = self.ds.get_MW(G, H, mode="V^-1/2", band_covar=V)
        np.testing.assert_allclose(W, M @ H, rtol=1e-12, atol=1e-14 * np.abs(W).max())
        assert_rows_sum_to_one(W)
        cov = M @ V @ M.T
        d = np.sqrt(np.diag(cov))
        np.testing.assert_allclose(
            cov / np.outer(d, d), np.eye(n), rtol=0, atol=1e-10 * cond
        )


BLPAIRS = [101102101102, 102103102103, 101103101103]


@functools.cache
def _vanilla_uvp(Ndlys: int | None):
    return testing.build_vanilla_uvpspec(Ndlys=Ndlys)[0]


def vanilla_uvp(Ndlys: int | None = None, seed: int | None = None):
    """A copy of a vanilla UVPSpec, with random data if a seed is given."""
    uvp = copy.deepcopy(_vanilla_uvp(Ndlys))
    if seed is not None:
        rng = np.random.default_rng(seed)
        shape = uvp.data_array[0].shape
        uvp.data_array[0] = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return uvp


def random_window_functions(rng, n: int, symmetric: bool = False) -> np.ndarray:
    """Positive (n, n) window functions with unit row sums. If symmetric, the
    response of delay tau to k equals that of -tau to -k, and the most negative
    delay of an even number of delays (which has no positive counterpart) only
    responds to itself."""
    W = rng.random((n, n))
    if symmetric:
        mid = n // 2
        idx = np.arange(int(n % 2 == 0), n)
        mirror = 2 * mid - idx
        W[np.ix_(idx, idx)] += W[np.ix_(mirror, mirror)]
        if n % 2 == 0:
            W[0, :] = 0
            W[:, 0] = 0
            W[0, 0] = 1
    return W / W.sum(axis=1, keepdims=True)


seeds = st.integers(0, 2**32 - 1)


class TestUVPSpecOperations:
    @settings(max_examples=25)
    @given(
        weights=st.lists(
            st.one_of(st.just(0.0), st.floats(0.01, 10.0)), min_size=1, max_size=3
        ),
        blpairs=st.permutations(BLPAIRS),
        seed=seeds,
    )
    def test_weighted_average(self, weights: list, blpairs: list, seed: int) -> None:
        """Baseline-pair weights act as (possibly fractional) multiplicities.

        After normalisation to sum to the number of baseline pairs N, a pair
        with weight m counts as m copies of itself: the average is the
        m-weighted mean, nsamples add up with weights m, and squared errors
        and covariances add up as sum(m * sigma^2) / N^2 (all pairs here have
        the same integration time and nsamples, so no other weights enter).
        """
        assume(sum(weights) > 0)
        blps = blpairs[: len(weights)]
        uvp = vanilla_uvp(seed=seed)
        rng = np.random.default_rng(seed + 1)
        uvp.stats_array = {
            "err": {0: (1 + rng.random(uvp.data_array[0].shape)).astype(complex)}
        }
        uvp.cov_array_real[0] = rng.random(uvp.cov_array_real[0].shape)
        keys = [(0, blp, ("xx", "xx")) for blp in blps]
        m = np.array(weights) * len(weights) / sum(weights)

        avg = grouping.average_spectra(
            uvp,
            blpair_groups=[blps],
            blpair_weights=[weights],
            error_field="err",
            inplace=False,
        )
        key = keys[0]
        np.testing.assert_allclose(
            avg.get_data(key),
            sum(mi * uvp.get_data(k) for mi, k in zip(m, keys)) / m.sum(),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            avg.get_nsamples(key),
            sum(mi * uvp.get_nsamples(k) for mi, k in zip(m, keys)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            avg.get_stats("err", key).real,
            np.sqrt(
                sum(mi * uvp.get_stats("err", k).real ** 2 for mi, k in zip(m, keys))
            )
            / m.sum(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            avg.get_cov(key),
            sum(mi * uvp.get_cov(k) for mi, k in zip(m, keys)) / m.sum() ** 2,
            rtol=1e-12,
        )

    @settings(max_examples=25)
    @given(Ndlys=st.sampled_from([6, 7, 20, 21]), data=st.data(), seed=seeds)
    def test_fold_spectra(self, Ndlys: int, data, seed: int) -> None:
        """Folding averages the spectra at +tau and -tau onto tau > 0.

        A spectrum symmetric in tau folds to its positive half, and an
        antisymmetric one to zero; negative delays are emptied. The folded
        covariance is that of the mean of the +tau and -tau spectra, and
        window functions with the symmetry of a real sky keep unit row sums.
        """
        mid = Ndlys // 2  # index of zero delay
        npos = Ndlys - mid - 1  # number of positive delays
        uvp = vanilla_uvp(Ndlys)
        nrows = uvp.Nbltpairs

        sym = data.draw(complex_arrays(npos + 1))  # at |tau| = 0, 1, ..., npos
        anti = data.draw(complex_arrays(npos))  # at tau = 1, ..., npos
        r_sym = data.draw(complex_arrays(nrows, 10.0))
        r_anti = data.draw(complex_arrays(nrows, 10.0))
        spectra = np.zeros((nrows, Ndlys), dtype=complex)
        spectra[:, mid:] = np.outer(r_sym, sym)
        spectra[:, mid + 1 :] += np.outer(r_anti, anti)
        spectra[:, mid - npos : mid] = (
            np.outer(r_sym, sym[1:]) - np.outer(r_anti, anti)
        )[:, ::-1]
        if Ndlys % 2 == 0:
            spectra[:, 0] = data.draw(complex_arrays(nrows))
        uvp.data_array[0] = spectra[..., None].copy()

        rng = np.random.default_rng(seed)
        B = rng.standard_normal((Ndlys, Ndlys))
        cov = B @ B.T / Ndlys
        uvp.cov_array_real[0][:] = cov[None, ..., None]
        W = random_window_functions(rng, Ndlys, symmetric=True)
        uvp.window_function_array[0][:] = W[None, ..., None]

        grouping.fold_spectra(uvp)
        folded = uvp.data_array[0][..., 0]
        np.testing.assert_array_equal(folded[:, :mid], 0)
        np.testing.assert_array_equal(folded[:, mid], spectra[:, mid])
        assert_close_to(
            folded[:, mid + 1 :], np.outer(r_sym, sym[1:]), 10 * sym, 10 * anti
        )

        pos = np.arange(mid + 1, Ndlys)
        F = np.zeros((npos, Ndlys))
        F[np.arange(npos), pos] = 0.5
        F[np.arange(npos), 2 * mid - pos] = 0.5
        np.testing.assert_allclose(
            uvp.cov_array_real[0][0, mid + 1 :, mid + 1 :, 0],
            F @ cov @ F.T,
            rtol=1e-12,
            atol=1e-14,
        )
        assert_rows_sum_to_one(uvp.window_function_array[0][0, mid + 1 :, :, 0])

    @settings(max_examples=25)
    @given(
        Ndlys=st.sampled_from([20, 21]),
        half_kernel=arrays(np.float64, st.integers(1, 4), elements=st.floats(0.1, 10)),
        centre=st.one_of(st.none(), st.floats(0.1, 10)),
        coeffs=complex_arrays(2, 10.0),
        seed=seeds,
    )
    def test_average_in_delay_bins(
        self, Ndlys: int, half_kernel, centre, coeffs, seed: int
    ) -> None:
        """Binning in delay takes kernel-weighted means over non-overlapping
        bins, keeping the zero delay unbinned.

        A spectrum that is linear in delay is binned to the same linear
        function of the (kernel-weighted) bin delays, bins of independent
        unit-variance spectra have variance sum(kernel^2) / sum(kernel)^2, and
        window functions keep unit row sums.
        """
        kernel = np.concatenate(
            [half_kernel, [] if centre is None else [centre], half_kernel[::-1]]
        )
        uvp = vanilla_uvp(Ndlys)
        dlys = uvp.get_dlys(0)
        uvp.data_array[0][:] = (coeffs[0] + coeffs[1] * dlys / dlys.max())[
            None, :, None
        ]
        W = random_window_functions(np.random.default_rng(seed), Ndlys)
        uvp.window_function_array[0][:] = W[None, ..., None]

        binned = grouping.average_in_delay_bins(uvp, kernel=kernel)
        new_dlys = binned.get_dlys(0)
        zero = np.flatnonzero(new_dlys == 0)
        assert len(zero) == 1
        np.testing.assert_allclose(
            binned.data_array[0][..., 0],
            np.broadcast_to(
                coeffs[0] + coeffs[1] * new_dlys / dlys.max(),
                binned.data_array[0].shape[:2],
            ),
            rtol=1e-12,
            atol=1e-12 * np.abs(coeffs).max(),
        )

        # the vanilla covariances are the identity
        variance = np.sum(kernel**2) / np.sum(kernel) ** 2
        expected = np.full(len(new_dlys), variance)
        expected[zero] = 1.0
        cov = binned.cov_array_real[0][0, ..., 0]
        np.testing.assert_allclose(cov, np.diag(expected), rtol=1e-12, atol=1e-15)
        assert_rows_sum_to_one(binned.window_function_array[0][0, ..., 0])

    @pytest.mark.parametrize(
        "Ndlys",
        [
            21,
            pytest.param(
                20,
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="For an even number of delays, the unpaired most "
                    "negative delay is binned (with its neighbours) whenever the "
                    "kernel length divides Ndlys / 2, giving one more negative "
                    "than positive bin, although the docstring says it is ignored "
                    "(see #575).",
                ),
            ),
        ],
    )
    @settings(max_examples=25)
    @given(
        half_kernel=arrays(np.float64, st.integers(1, 4), elements=st.floats(0.1, 10)),
        centre=st.one_of(st.none(), st.floats(0.1, 10)),
        data=st.data(),
    )
    @example(half_kernel=np.ones(1), centre=None, data=None)
    def test_delay_bins_are_symmetric(
        self, Ndlys: int, half_kernel, centre, data
    ) -> None:
        """With a symmetric kernel, the delay bins are symmetric about zero
        delay, and a spectrum that is symmetric in delay stays symmetric."""
        kernel = np.concatenate(
            [half_kernel, [] if centre is None else [centre], half_kernel[::-1]]
        )
        mid = Ndlys // 2
        npos = Ndlys - mid - 1
        if data is None:
            sym = np.arange(1, npos + 2, dtype=complex)
        else:
            sym = data.draw(complex_arrays(npos + 1))  # at |tau| = 0, 1, ..., npos
        uvp = vanilla_uvp(Ndlys)
        spectrum = np.concatenate([sym[:0:-1], sym])
        if Ndlys % 2 == 0:
            spectrum = np.concatenate([[0], spectrum])
        uvp.data_array[0][:] = spectrum[None, :, None]

        binned = grouping.average_in_delay_bins(uvp, kernel=kernel)
        new_dlys = binned.get_dlys(0)
        np.testing.assert_allclose(new_dlys, -new_dlys[::-1], rtol=1e-12, atol=1e-20)
        new_spectrum = binned.data_array[0][0, :, 0]
        assert_close_to(new_spectrum, new_spectrum[::-1], 10 * sym)

    @settings(max_examples=20)
    @given(
        seeds=st.tuples(seeds, seeds),
        scales=st.tuples(st.floats(0.1, 10.0), st.floats(0.1, 10.0)),
    )
    def test_subtract_uvp(self, seeds, scales) -> None:
        """The difference of two spectra has their difference as data, and,
        for independent spectra, the sum of their covariances; errors add in
        quadrature. A spectrum minus itself is zero."""
        uvps = []
        for seed, scale in zip(seeds, scales):
            uvp = vanilla_uvp(seed=seed)
            uvp.cov_array_real[0] *= scale
            uvp.cov_array_imag[0] *= 2 * scale
            uvp.stats_array = {"err": {0: scale * np.ones_like(uvp.data_array[0])}}
            uvps.append(uvp)
        uvp1, uvp2 = uvps

        diff = uvputils.subtract_uvp(uvp1, uvp2, run_check=False)
        np.testing.assert_allclose(
            diff.data_array[0], uvp1.data_array[0] - uvp2.data_array[0], rtol=1e-14
        )
        for component in ["real", "imag"]:
            np.testing.assert_allclose(
                getattr(diff, f"cov_array_{component}")[0],
                getattr(uvp1, f"cov_array_{component}")[0]
                + getattr(uvp2, f"cov_array_{component}")[0],
                rtol=1e-14,
            )
        np.testing.assert_allclose(
            diff.stats_array["err"][0].real, np.hypot(*scales), rtol=1e-14
        )

        same = uvputils.subtract_uvp(uvp1, uvp1, run_check=False)
        np.testing.assert_array_equal(same.data_array[0], 0)
        np.testing.assert_allclose(
            same.cov_array_real[0], 2 * uvp1.cov_array_real[0], rtol=1e-14
        )

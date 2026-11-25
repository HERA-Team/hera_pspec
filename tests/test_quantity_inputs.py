"""Quantity inputs at the public API boundaries.

Each boundary accepts an astropy Quantity in any equivalent unit, and a plain
number raises an astropy UnitTypeError that names the argument and its unit.
"""

import copy
import re

import astropy.cosmology.units as cu
import h5py
import numpy as np
import pytest
from astropy import units as u

from hera_pspec import noise, pspecbeam, utils, uvwindow
from tests.regression_cases import FT_BEAM_FILE

# frequencies chosen so that MHz -> Hz conversion is exact
FREQS_MHZ = np.arange(140.0, 160.0, 0.5)
FREQS_HZ = FREQS_MHZ * 1e6


class TestQuantityValue:
    def test_quantity_is_converted(self):
        """Check that a Quantity is returned as plain numbers in the requested unit."""
        assert utils.quantity_value(150 * u.MHz, u.Hz, "freq") == 150e6
        np.testing.assert_array_equal(
            utils.quantity_value(FREQS_MHZ * u.MHz, u.Hz, "freqs"), FREQS_HZ
        )

    def test_list_of_quantities_is_stacked(self):
        """Check that lists of Quantities and of (low, high) Quantity pairs are stacked."""
        out = utils.quantity_value([1 * u.MHz, 2000 * u.kHz], u.Hz, "freqs")
        np.testing.assert_array_equal(out, [1e6, 2e6])
        out = utils.quantity_value([(1 * u.MHz, 2 * u.MHz)], u.Hz, "freqs")
        np.testing.assert_array_equal(out, [[1e6, 2e6]])

    def test_none_passes_through(self):
        """Check that None is returned unchanged."""
        assert utils.quantity_value(None, u.Hz, "freqs") is None

    @pytest.mark.parametrize(
        "value,type_name",
        [(3, "int"), (150e6, "float"), (FREQS_HZ, "ndarray"), ([1e6, 2e6], "list")],
        ids=["int", "float", "array", "list"],
    )
    def test_plain_number_raises(self, value, type_name):
        """Check that a plain number raises UnitTypeError naming the argument, unit and type."""
        with pytest.raises(
            u.UnitTypeError,
            match=re.escape(
                "freqs must be an astropy Quantity in units of Hz (e.g. 1 * u.Hz), "
                f"got {type_name}."
            ),
        ):
            utils.quantity_value(value, u.Hz, "freqs")

    def test_plain_number_message_example(self):
        """Check that the error message shows the example Quantity it is given."""
        with pytest.raises(u.UnitTypeError, match=re.escape("(e.g. 100 * u.MHz)")):
            utils.quantity_value(1e8, u.Hz, "freq_range", example="100 * u.MHz")

    def test_wrong_unit_raises(self):
        """Check that a Quantity in the wrong unit, or a mixed list, raises UnitConversionError."""
        with pytest.raises(u.UnitConversionError, match="freqs must be in a unit"):
            utils.quantity_value(3 * u.s, u.Hz, "freqs")
        with pytest.raises(u.UnitConversionError, match="mixed units"):
            utils.quantity_value([1 * u.MHz, 2 * u.s], u.Hz, "freqs")
        with pytest.raises(u.UnitConversionError, match="mixed units"):
            utils.quantity_value([1 * u.MHz, 2e6], u.Hz, "freqs")

    def test_equivalencies(self):
        """Check that equivalencies are used in the conversion."""
        out = utils.quantity_value(2 * u.m, u.Hz, "freq", equivalencies=u.spectral())
        np.testing.assert_allclose(out, 299792458.0 / 2, rtol=1e-15)


@pytest.fixture(scope="module")
def gauss_beam_hz():
    return pspecbeam.PSpecBeamGauss(0.25 * u.rad, np.linspace(100e6, 200e6, 51) * u.Hz)


class TestPSpecBeamGauss:
    def test_scaled_units_match(self, gauss_beam_hz):
        beam = pspecbeam.PSpecBeamGauss(
            np.rad2deg(0.25) * u.deg, np.linspace(100, 200, 51) * u.MHz
        )
        np.testing.assert_allclose(beam.fwhm, gauss_beam_hz.fwhm, rtol=1e-15)
        np.testing.assert_allclose(
            beam.beam_freqs, gauss_beam_hz.beam_freqs, rtol=1e-15
        )
        np.testing.assert_allclose(
            beam.power_beam_int(), gauss_beam_hz.power_beam_int(), rtol=1e-14
        )

    def test_plain_fwhm_raises(self):
        """Check that a plain-number fwhm raises UnitTypeError."""
        with pytest.raises(u.UnitTypeError, match="fwhm must be an astropy Quantity"):
            pspecbeam.PSpecBeamGauss(0.25, FREQS_MHZ * u.MHz)

    def test_wrong_units_raise(self):
        with pytest.raises(u.UnitConversionError, match="fwhm"):
            pspecbeam.PSpecBeamGauss(0.25 * u.m, FREQS_MHZ * u.MHz)
        with pytest.raises(u.UnitConversionError, match="beam_freqs"):
            pspecbeam.PSpecBeamGauss(0.25 * u.rad, FREQS_MHZ * u.s)

    def test_jy_to_mk_with_mhz(self, gauss_beam_hz):
        """The pyuvdata-backed Gaussian beam works with MHz frequencies."""
        expected = gauss_beam_hz.Jy_to_mK(FREQS_HZ * u.Hz)
        np.testing.assert_array_equal(
            gauss_beam_hz.Jy_to_mK(FREQS_MHZ * u.MHz), expected
        )
        with pytest.raises(u.UnitTypeError, match="freqs must be an astropy Quantity"):
            gauss_beam_hz.Jy_to_mK(FREQS_HZ)
        scalar = gauss_beam_hz.Jy_to_mK(150 * u.MHz)
        assert isinstance(scalar, np.ndarray) and scalar.shape == (1,)
        with pytest.raises(u.UnitConversionError, match="freqs"):
            gauss_beam_hz.Jy_to_mK(FREQS_MHZ * u.m)

    def test_analytic_beam_evaluates_with_quantity_input(self):
        beam = pspecbeam.PSpecBeamGauss(10 * u.deg, FREQS_MHZ * u.MHz)
        # the stored frequencies are plain Hz, which pyuvdata needs
        assert not isinstance(beam.beam_freqs, u.Quantity)
        values = beam.analytic_beam.power_eval(
            az_array=np.zeros(1), za_array=np.zeros(1), freq_array=beam.beam_freqs[:1]
        )
        assert np.all(np.isfinite(values))


class TestComputePspecScalar:
    @pytest.mark.parametrize(
        "lower,upper",
        [
            (120e6 * u.Hz, 128e6 * u.Hz),
            (120 * u.MHz, 128 * u.MHz),
            (0.12 * u.GHz, 128000 * u.kHz),
        ],
    )
    def test_units_match(self, beam_nf_pstokes, lower, upper):
        expected = beam_nf_pstokes.compute_pspec_scalar(
            120e6 * u.Hz, 128e6 * u.Hz, 20, num_steps=500
        )
        scalar = beam_nf_pstokes.compute_pspec_scalar(lower, upper, 20, num_steps=500)
        np.testing.assert_allclose(scalar, expected, rtol=1e-14)

    def test_plain_numbers_raise(self, beam_nf_pstokes):
        """Check that plain-number band edges raise UnitTypeError."""
        with pytest.raises(
            u.UnitTypeError, match="lower_freq must be an astropy Quantity"
        ):
            beam_nf_pstokes.compute_pspec_scalar(120e6, 128e6, 20)

    def test_wrong_unit_raises(self, beam_nf_pstokes):
        with pytest.raises(u.UnitConversionError, match="lower_freq"):
            beam_nf_pstokes.compute_pspec_scalar(120 * u.m, 128 * u.MHz, 20)


class TestPSpecBeamFromArray:
    def _arrays(self, gauss_beam_hz):
        return (
            gauss_beam_hz.power_beam_int(),
            gauss_beam_hz.power_beam_sq_int(),
            gauss_beam_hz.beam_freqs,
        )

    def test_scaled_units_match(self, gauss_beam_hz):
        omega_p, omega_pp, freqs = self._arrays(gauss_beam_hz)
        ref = pspecbeam.PSpecBeamFromArray(
            omega_p * u.sr, omega_pp * u.sr, freqs * u.Hz
        )
        beam = pspecbeam.PSpecBeamFromArray(
            {"pI": (omega_p * u.sr).to(u.deg**2)},
            {"pI": (omega_pp * u.sr).to(u.deg**2)},
            (freqs * u.Hz).to(u.MHz),
        )
        np.testing.assert_allclose(
            beam.power_beam_int(), ref.power_beam_int(), rtol=1e-14
        )
        np.testing.assert_allclose(
            beam.power_beam_sq_int(), ref.power_beam_sq_int(), rtol=1e-14
        )
        np.testing.assert_allclose(beam.beam_freqs, ref.beam_freqs, rtol=1e-15)

    def test_plain_numbers_raise(self, gauss_beam_hz):
        """Check that plain-number solid angles raise UnitTypeError, also in add_pol."""
        omega_p, omega_pp, freqs = self._arrays(gauss_beam_hz)
        with pytest.raises(u.UnitTypeError, match="OmegaP must be an astropy Quantity"):
            pspecbeam.PSpecBeamFromArray(omega_p, omega_pp * u.sr, freqs * u.Hz)
        beam = pspecbeam.PSpecBeamFromArray(
            omega_p * u.sr, omega_pp * u.sr, freqs * u.Hz
        )
        with pytest.raises(
            u.UnitTypeError, match="OmegaPP must be an astropy Quantity"
        ):
            beam.add_pol("pQ", omega_p * u.sr, omega_pp)

    def test_add_pol_quantity(self, gauss_beam_hz):
        omega_p, omega_pp, freqs = self._arrays(gauss_beam_hz)
        beam = pspecbeam.PSpecBeamFromArray(
            omega_p * u.sr, omega_pp * u.sr, freqs * u.Hz
        )
        beam.add_pol("pQ", (omega_p * u.sr).to(u.deg**2), omega_pp * u.sr)
        np.testing.assert_allclose(beam.power_beam_int("pQ"), omega_p, rtol=1e-14)

    def test_wrong_unit_raises(self, gauss_beam_hz):
        omega_p, omega_pp, freqs = self._arrays(gauss_beam_hz)
        with pytest.raises(u.UnitConversionError, match="OmegaP"):
            pspecbeam.PSpecBeamFromArray(omega_p * u.m, omega_pp * u.sr, freqs * u.Hz)


class TestBeamNormalizedResponse:
    def test_mhz_matches_hz(self, beam_nf_dipole):
        freqs = np.arange(130.0, 140.0, 1.0)
        res_hz, omega_hz, _ = beam_nf_dipole.beam_normalized_response(
            "xx", freqs * 1e6 * u.Hz
        )
        res_mhz, omega_mhz, _ = beam_nf_dipole.beam_normalized_response(
            "xx", freqs * u.MHz
        )
        np.testing.assert_array_equal(res_mhz, res_hz)
        np.testing.assert_array_equal(omega_mhz, omega_hz)


class TestCalcPN:
    SCALAR = 568847837.72586381

    def test_plain_scalar_quantity_inputs(self):
        expected = noise._calc_P_N(self.SCALAR, 300.0, 10.7, Nincoherent=4)
        out = noise.calc_P_N(self.SCALAR, 300 * u.K, 10.7 * u.s, Nincoherent=4)
        assert not isinstance(out, u.Quantity)
        assert out == expected
        out = noise.calc_P_N(self.SCALAR, 300000 * u.mK, 10700 * u.ms, Nincoherent=4)
        np.testing.assert_allclose(out, expected, rtol=1e-14)

    def test_plain_tsys_raises(self):
        """Check that a plain-number Tsys raises UnitTypeError with a K example."""
        with pytest.raises(
            u.UnitTypeError,
            match=re.escape(
                "Tsys must be an astropy Quantity in units of K (e.g. 300 * u.K)"
            ),
        ):
            noise.calc_P_N(self.SCALAR, 300.0, 10.7 * u.s)

    def test_quantity_scalar_returns_quantity(self):
        scalar_unit = u.Mpc**3 / cu.littleh**3 / (u.Hz * u.sr)
        k = np.linspace(0.05, 2.0, 8)
        pk = noise.calc_P_N(self.SCALAR * scalar_unit, 300 * u.K, 10.7 * u.s)
        assert pk.unit == u.mK**2 * u.Mpc**3 / cu.littleh**3
        assert pk.value == noise._calc_P_N(self.SCALAR, 300.0, 10.7)
        dsq = noise.calc_P_N(
            self.SCALAR * scalar_unit,
            300 * u.K,
            10.7 * u.s,
            form="DelSq",
            k=k * cu.littleh / u.Mpc,
        )
        assert dsq.unit == u.mK**2
        np.testing.assert_array_equal(
            dsq.value, noise._calc_P_N(self.SCALAR, 300.0, 10.7, form="DelSq", k=k)
        )
        # plain k is read in 1 / L, the length unit of the scalar
        dsq_plain_k = noise.calc_P_N(
            self.SCALAR * scalar_unit, 300 * u.K, 10.7 * u.s, form="DelSq", k=k
        )
        np.testing.assert_array_equal(dsq_plain_k.value, dsq.value)

    def test_mpc_scalar(self):
        scalar = self.SCALAR * u.Mpc**3 / (u.Hz * u.sr)
        k = np.linspace(0.05, 2.0, 8)
        pk = noise.calc_P_N(scalar, 300 * u.K, 10.7 * u.s)
        assert pk.unit == u.mK**2 * u.Mpc**3
        dsq = noise.calc_P_N(
            scalar, 300 * u.K, 10.7 * u.s, form="DelSq", k=k * 1e-3 / u.kpc
        )
        np.testing.assert_allclose(
            dsq.value,
            noise._calc_P_N(self.SCALAR, 300.0, 10.7, form="DelSq", k=k),
            rtol=1e-14,
        )
        with pytest.raises(u.UnitConversionError, match="k must be"):
            noise.calc_P_N(
                scalar, 300 * u.K, 10.7 * u.s, form="DelSq", k=k * cu.littleh / u.Mpc
            )

    def test_errors(self):
        with pytest.raises(u.UnitConversionError, match="Tsys"):
            noise.calc_P_N(self.SCALAR, 300 * u.s, 10.7 * u.s)
        with pytest.raises(u.UnitConversionError, match="t_int"):
            noise.calc_P_N(self.SCALAR, 300 * u.K, 10.7 * u.K)
        with pytest.raises(u.UnitConversionError, match="scalar must be"):
            noise.calc_P_N(1.0 * u.s, 300 * u.K, 10.7 * u.s)
        with pytest.raises(TypeError, match="scalar is a plain float"):
            noise.calc_P_N(
                self.SCALAR, 300 * u.K, 10.7 * u.s, form="DelSq", k=[1.0] / u.Mpc
            )


class TestSensitivity:
    @pytest.fixture
    def sense(self, beam_nf_pstokes):
        sense = noise.Sensitivity(beam=copy.deepcopy(beam_nf_pstokes))
        sense.calc_scalar(FREQS_MHZ * u.MHz, "pI", num_steps=500)
        return sense

    def test_calc_scalar_units(self, sense):
        scalar = sense.scalar
        np.testing.assert_array_equal(sense.subband, FREQS_HZ)
        sense.calc_scalar(FREQS_HZ * u.Hz, "pI", num_steps=500)
        assert sense.scalar == scalar
        with pytest.raises(u.UnitConversionError, match="freqs"):
            sense.calc_scalar(FREQS_HZ * u.m, "pI")

    def test_calc_p_n_units(self, sense):
        expected = sense.calc_P_N(300 * u.K, 10.7 * u.s)
        assert not isinstance(expected, u.Quantity)
        np.testing.assert_allclose(
            sense.calc_P_N(0.3 * u.kK, 10700 * u.ms), expected, rtol=1e-14
        )
        with pytest.raises(u.UnitTypeError, match="t_int must be an astropy Quantity"):
            sense.calc_P_N(300 * u.K, 10.7)
        k = np.linspace(0.1, 1.0, 5)
        dsq = sense.calc_P_N(300 * u.K, 10.7 * u.s, form="DelSq", k=k)
        np.testing.assert_array_equal(
            sense.calc_P_N(
                300 * u.K, 10.7 * u.s, form="DelSq", k=k * cu.littleh / u.Mpc
            ),
            dsq,
        )
        with pytest.raises(u.UnitConversionError, match="k must be"):
            sense.calc_P_N(300 * u.K, 10.7 * u.s, form="DelSq", k=k / u.Mpc)


class TestGenerateNoiseSpectra:
    POLPAIR = ("xx", "xx")

    def test_tsys_units(self, vanilla_uvp_with_beam):
        uvp = vanilla_uvp_with_beam
        blp = uvp.get_blpairs()[0]
        blp_int = uvp.antnums_to_blpair(blp)
        expected = uvp.generate_noise_spectra(0, self.POLPAIR, 500 * u.K)[blp_int]
        np.testing.assert_array_equal(
            uvp.generate_noise_spectra(0, self.POLPAIR, 500000 * u.mK)[blp_int],
            expected,
        )
        tsys = {blp_int: 500 * u.K * np.ones(uvp.Ntpairs)}
        np.testing.assert_array_equal(
            uvp.generate_noise_spectra(0, self.POLPAIR, tsys, blpairs=[blp])[blp_int],
            expected,
        )
        with pytest.raises(u.UnitTypeError, match="Tsys must be an astropy Quantity"):
            uvp.generate_noise_spectra(0, self.POLPAIR, 500)
        with pytest.raises(u.UnitTypeError, match="Tsys must be an astropy Quantity"):
            uvp.generate_noise_spectra(
                0, self.POLPAIR, {blp_int: 500.0 * np.ones(uvp.Ntpairs)}, blpairs=[blp]
            )
        with pytest.raises(u.UnitConversionError, match="Tsys"):
            uvp.generate_noise_spectra(0, self.POLPAIR, 500 * u.s)


class TestSpwRangeFromFreqs:
    def test_forms_match(self, vanilla_uvp):
        expected = utils.spw_range_from_freqs(vanilla_uvp, (100.1e6, 100.74e6) * u.Hz)
        assert expected == (1, 7)
        assert (
            utils.spw_range_from_freqs(vanilla_uvp, (100.1 * u.MHz, 100740 * u.kHz))
            == expected
        )
        assert utils.spw_range_from_freqs(
            vanilla_uvp, [(100.1e6, 100.74e6)] * u.Hz
        ) == [expected]
        assert utils.spw_range_from_freqs(
            vanilla_uvp, [(100.1e6 * u.Hz, 100.74e6 * u.Hz)] * 2
        ) == [expected, expected]

    def test_plain_numbers_raise(self, vanilla_uvp):
        """Check that a plain-number frequency range raises UnitTypeError."""
        with pytest.raises(
            u.UnitTypeError,
            match=re.escape(
                "freq_range must be an astropy Quantity in units of Hz "
                "(e.g. (100 * u.MHz, 120 * u.MHz)), got tuple."
            ),
        ):
            utils.spw_range_from_freqs(vanilla_uvp, (100.1e6, 100.74e6))

    def test_errors(self, vanilla_uvp):
        with pytest.raises(u.UnitConversionError, match="freq_range"):
            utils.spw_range_from_freqs(vanilla_uvp, (1, 2) * u.m)
        with pytest.raises(ValueError, match="shape"):
            utils.spw_range_from_freqs(vanilla_uvp, [1, 2, 3] * u.MHz)

    def test_redshifts_need_no_deprecated_path(self, vanilla_uvp):
        # spw_range_from_redshifts passes Quantities on: no warning
        assert utils.spw_range_from_redshifts(vanilla_uvp, (13.1, 13.2)) == (1, 7)


@pytest.fixture(scope="module")
def ft_freqs():
    return uvwindow.FTBeam.get_bandwidth(FT_BEAM_FILE)[5:25]


class TestFTBeam:
    def test_init_units(self, ft_freqs):
        data = np.ones((ft_freqs.size, 5, 5))
        ref = uvwindow.FTBeam(data, "xx", ft_freqs * u.Hz, 1.0 * u.rad)
        beam = uvwindow.FTBeam(
            data, "xx", (ft_freqs * u.Hz).to(u.MHz), np.rad2deg(1.0) * u.deg
        )
        np.testing.assert_allclose(beam.freq_array, ref.freq_array, rtol=1e-15)
        np.testing.assert_allclose(beam.mapsize, ref.mapsize, rtol=1e-15)
        with pytest.raises(
            u.UnitTypeError, match="freq_array must be an astropy Quantity"
        ):
            uvwindow.FTBeam(data, "xx", ft_freqs, 1.0 * u.rad)
        with pytest.raises(u.UnitConversionError, match="mapsize"):
            uvwindow.FTBeam(data, "xx", ft_freqs * u.Hz, 1.0 * u.m)

    def test_from_file_units(self, ft_freqs):
        ref = uvwindow.FTBeam.from_file(FT_BEAM_FILE, freq_array=ft_freqs * u.Hz)
        beam = uvwindow.FTBeam.from_file(
            FT_BEAM_FILE, freq_array=(ft_freqs * u.Hz).to(u.kHz)
        )
        np.testing.assert_allclose(beam.freq_array, ref.freq_array, rtol=1e-15)
        np.testing.assert_array_equal(beam.ft_beam, ref.ft_beam)
        with pytest.raises(u.UnitConversionError, match="freq_array"):
            uvwindow.FTBeam.from_file(FT_BEAM_FILE, freq_array=ft_freqs * u.s)

    def test_select_freqs_units(self, ft_freqs):
        full = uvwindow.FTBeam.from_file(FT_BEAM_FILE, freq_array=ft_freqs * u.Hz)
        ref = full.select_freqs(ft_freqs[2:8] * u.Hz, inplace=False)
        beam = full.select_freqs(ft_freqs[2:8] * u.Hz, atol=1 * u.kHz, inplace=False)
        assert beam == ref
        with pytest.raises(u.UnitTypeError, match="atol must be an astropy Quantity"):
            full.select_freqs(ft_freqs[2:8] * u.Hz, atol=1e3, inplace=False)

    def test_gaussian_units(self, ft_freqs):
        ref = uvwindow.FTBeam.gaussian(
            ft_freqs * u.Hz, 5 * u.deg, "xx", mapsize=1 * u.rad, npix=31
        )
        beam = uvwindow.FTBeam.gaussian(
            (ft_freqs * u.Hz).to(u.MHz), 300 * u.arcmin, "xx", npix=31
        )
        np.testing.assert_allclose(beam.ft_beam, ref.ft_beam, rtol=1e-12)
        with pytest.raises(u.UnitTypeError, match="widths must be an astropy Quantity"):
            uvwindow.FTBeam.gaussian(ft_freqs * u.Hz, 5.0, "xx", npix=31)
        with pytest.raises(u.UnitConversionError, match="widths"):
            uvwindow.FTBeam.gaussian(ft_freqs * u.Hz, 5 * u.m, "xx")


class TestUVWindowLengths:
    @pytest.fixture(scope="class")
    def uvw(self, ft_freqs):
        ftbeam = uvwindow.FTBeam.from_file(FT_BEAM_FILE, freq_array=ft_freqs * u.Hz)
        return uvwindow.UVWindow(ftbeam_obj=ftbeam, taper="blackman-harris")

    def test_k_bins(self, uvw):
        bl_lens = np.array([14.6, 29.2])
        ref = uvw.get_kperp_bins(bl_lens * u.m)
        assert uvw.get_kperp_bins((bl_lens * u.m).to(u.cm)).unit == ref.unit
        np.testing.assert_allclose(
            uvw.get_kperp_bins((bl_lens * u.m).to(u.cm)).value, ref.value, rtol=1e-14
        )
        with pytest.raises(
            u.UnitTypeError, match="bl_lens must be an astropy Quantity"
        ):
            uvw.get_kperp_bins(bl_lens)
        with pytest.raises(u.UnitConversionError, match="bl_lens"):
            uvw.get_kperp_bins(bl_lens * u.Hz)

    def test_cylindrical_and_spherical(self, uvw):
        cyl = uvw.get_cylindrical_wf(14.6 * u.m)
        np.testing.assert_allclose(
            uvw.get_cylindrical_wf(1460 * u.cm), cyl, rtol=1e-12, atol=1e-15
        )
        with pytest.raises(u.UnitTypeError, match="bl_len must be an astropy Quantity"):
            uvw.get_cylindrical_wf(14.6)
        with pytest.raises(u.UnitConversionError, match="bl_len"):
            uvw.get_cylindrical_wf(14.6 * u.s)

        bl_lens = np.array([14.6, 29.2])
        kperp_bins = uvw.get_kperp_bins(bl_lens * u.m)
        kpara_bins = uvw.get_kpara_bins(uvw.freq_array * u.Hz)
        ktot = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2)
        kbins = np.arange(ktot.min(), ktot.max() + 0.1, 0.1) * uvw.kunits
        kwargs = dict(kperp_bins=kperp_bins, kpara_bins=kpara_bins)
        ref = uvw.get_spherical_wf(kbins, bl_lens * u.m, **kwargs)
        np.testing.assert_array_equal(
            uvw.get_spherical_wf(kbins, (bl_lens * u.m).to(u.km), **kwargs), ref
        )
        cyl_wf = np.stack(
            [uvw.get_cylindrical_wf(bl * u.m, kperp_bins, kpara_bins) for bl in bl_lens]
        )
        sph, _ = uvw.cylindrical_to_spherical(cyl_wf, kbins, ktot, bl_lens * u.m)
        np.testing.assert_array_equal(sph, ref)

    def test_run_and_write(self, uvw, tmp_path):
        """Check that run_and_write stores the baseline lengths in m."""
        uvw.run_and_write(tmp_path / "q.h5", bl_lens=[1460] * u.cm)
        with h5py.File(tmp_path / "q.h5") as fq:
            np.testing.assert_allclose(fq["bl_lens"][()], [14.6], rtol=1e-15)

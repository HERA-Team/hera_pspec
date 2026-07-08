import copy
import logging
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import uvtools.dspec as dspec
from astropy import units
from pyuvdata import utils as uvutils
from scipy.interpolate import RegularGridInterpolator, interp1d

from . import conversions, utils
from . import uvpspec_utils as uvputils

logger = logging.getLogger(__name__)

# Cartesian half-width used for the flat-sky beam projection in
# FTBeam.from_beam. The Cartesian grid spans rx in [-_MAX_R, +_MAX_R];
# zenith angle is za = sqrt(x^2 + y^2) in radians, so the corner of the
# grid sits at za = sqrt(2)*_MAX_R = 1 rad (~57 deg), a sensible outer
# bound for the beam-relevant sky.
_MAX_R = 0.7


def _cartesian_to_spherical(x, y):
    """Invert the simple zenith-angle map x = za*cos(az), y = za*sin(az)."""
    za = np.sqrt(x**2 + y**2)
    rho = np.divide(x, za, where=za != 0)
    az = np.arccos(rho)
    return az, za


def _ft_beam_2d(rx, b, npix=500, mapsize=_MAX_R):
    """
    Pad/crop the Cartesian beam to a centered grid and take the 2D FFT.

    `mapsize` controls the FT grid extent. Increasing it improves
    k-resolution at the cost of going beyond the simulation's spatial
    support.
    """
    dr = np.diff(rx).mean()
    ngrid = int(2.0 * mapsize / dr)
    if ngrid % 2 == 0:
        ngrid += 1  # odd, so the FT is centered on zero

    new_map = np.zeros((ngrid, ngrid), dtype=complex)
    if ngrid < npix:
        new_map = b[
            (npix - ngrid) // 2 : -(npix - ngrid) // 2,
            (npix - ngrid) // 2 : -(npix - ngrid) // 2,
        ]
    elif ngrid > npix:
        new_map[
            -(npix - ngrid) // 2 : (npix - ngrid) // 2,
            -(npix - ngrid) // 2 : (npix - ngrid) // 2,
        ] = b
    else:
        new_map = np.copy(b)
        ngrid = rx.size

    Atilde = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(new_map), norm="ortho")
        * (2.0 * mapsize / ngrid) ** 1.0
    )
    Atilde = np.flip(Atilde).real  # decreasing-order axes
    return Atilde


class FTBeam:
    """Class for :class:`FTBeam` objects."""

    def __init__(
        self, data, pol, freq_array, mapsize, verbose=False, x_orientation=None
    ):
        """
        Obtain Fourier transform of beam in sky plane for given frenquencies.

        Initialise an object containing all the information related to the
        Fourier transform of the beam.
            - data contains the array of the Fourier transform, with
                dimensions (Nfreqs, N, N).
            - pol contains the polarisation of the beam used.
            - freq_array gives the frequency coordinates of data.
            - mapsize is the size of the flat map the beam was projected
                onto (in deg).

        Parameters
        ----------
        data : 3D array of floats
            Array containing the real part of the Fourier transform of the
            beam in the sky plane (flat-sky approximation), for each of
            the frequencies in freq_array.
            Has dimensions (Nfreqs,N,N).
        pol : str or int
            Can be pseudo-Stokes or power:
            in str form: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
            in number form: 1, 2, 4, 3, -5, -6, -7, -8
        freq_array : 1D array or list of floats
            Array or list of the frequencies over which the Fourier
            transform of the beam has been computed (in Hz).
        mapsize : float
            Size of the flat map the beam was projected onto (in deg).
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
        x_orientation: str, optional
            Orientation in cardinal direction east or north of X dipole.
            Default keeps polarization in X and Y basis.
            Used to convert polstr to polnum and conversely.
        """
        # initialise basic attributes
        self.verbose = bool(verbose)
        self.x_orientation = x_orientation

        # checks on data input
        data = np.array(data)
        assert data.ndim == 3, "Wrong dimensions for data input"
        assert data.shape[1] == data.shape[2], "Wrong dimensions for data input"
        self.ft_beam = data

        # polarisation
        if isinstance(pol, str):
            assert pol in ["pI", "pQ", "pV", "pU", "xx", "yy", "xy", "yx"], (
                "Wrong polarisation"
            )
        elif isinstance(pol, int):
            assert pol in [1, 2, 4, 3, -5, -6, -7, -8], "Wrong polarisation"
            # convert pol number to str according to AIPS Memo 117.
            pol = uvutils.polnum2str(pol, x_orientation=x_orientation)
        else:
            raise TypeError("Must feed pol as str or int.")
        self.pol = pol

        assert len(freq_array) == data.shape[0], (
            "data must have shape (len(freq_array), N, N)"
        )
        self.freq_array = np.array(freq_array)

        self.mapsize = float(mapsize)

    @classmethod
    def from_beam(
        cls,
        beamfile,
        pol,
        freq_array,
        mapsize=1.0,
        npix=301,
        verbose=False,
        x_orientation=None,
    ):
        """
        Compute Fourier transform of beam in sky plane given a file
        containing beam simulations.

        Given the path to a beam simulation, obtain the Fourier transform
        of the instrument beam in the sky plane for each frequency in
        `freq_array`. Computations correspond to the Fourier transform
        performed in equation 10 of HERA Memo #128: the beam is
        interpolated onto `freq_array`, peak-normalized per frequency,
        projected onto a Cartesian (flat-sky) plane, and 2D-FFT'd per
        frequency slice.

        `freq_array` should be the frequency channels of the data whose
        window functions are being computed (see :meth:`select_freqs`).

        Parameters
        ----------
        beamfile : str
            Path to file containing the beam simulation. Must be readable
            by :meth:`pyuvdata.UVBeam.read_beamfits`.
        pol : str or int
            Polarization, can be pseudo-Stokes or power:
            in str form: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
            in number form: 1, 2, 4, 3, -5, -6, -7, -8.
        freq_array : 1D array or list of floats
            Frequencies (in Hz) to compute the Fourier transform of the
            beam on. Should match the frequency channels of the data.
        mapsize : float, optional
            Half-width of the flat map the beam is projected onto (in
            rad). Increase for better k_perp resolution. Default: 1.0.
        npix : int, optional
            Number of pixels per side used for the Cartesian projection
            of the beam. Preferably odd (even values are decremented).
            Default: 301.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
        x_orientation: str, optional
            Orientation in cardinal direction east or north of X dipole.
            Default keeps polarization in X and Y basis.
            Used to convert polstr to polnum and conversely.
        """
        from pyuvdata import UVBeam

        freq_array = np.asarray(freq_array, dtype=float)
        assert freq_array.ndim == 1 and freq_array.size > 2, (
            "freq_array must be a 1D array with at least three frequencies."
        )

        if isinstance(pol, int):
            polnum = pol
            pol = uvutils.polnum2str(pol, x_orientation=x_orientation)
        else:
            polnum = uvutils.polstr2num(pol, x_orientation=x_orientation)

        logger.info("Reading UVBeam from %s", beamfile)
        beam = UVBeam()
        beam.read_beamfits(str(beamfile))

        if beam.beam_type == "efield":
            if polnum < 0:
                beam.efield_to_power()
            else:
                beam.efield_to_pstokes()
        beam.select(polarizations=[polnum], inplace=True)
        beam.peak_normalize()

        # ensure odd number of pixels so the FT is centered on zero
        npix = int(npix)
        if npix % 2 == 0:
            npix -= 1
            logger.info("Adjusted npix to odd: %d", npix)

        # Cartesian (flat-sky) grid and its sky coordinates
        rx = np.linspace(-_MAX_R, _MAX_R, npix)
        xg, yg = np.meshgrid(rx, rx)
        az, za = _cartesian_to_spherical(xg, yg)

        # frequencies the beam is evaluated at: if a few requested channels
        # fall (slightly) outside the simulation coverage, evaluate the beam
        # at the nearest covered frequency for those channels, but keep the
        # true data frequencies as the frequency coordinates of the FTBeam
        # (the beam evolves slowly with frequency; this is much preferable
        # to shifting or resampling the frequency grid itself, which scales
        # the window functions' k_parallel axis)
        beam_fmin, beam_fmax = beam.freq_array.min(), beam.freq_array.max()
        eval_freqs = freq_array
        if (freq_array.min() < beam_fmin) or (freq_array.max() > beam_fmax):
            n_out = int(np.sum((freq_array < beam_fmin) | (freq_array > beam_fmax)))
            warnings.warn(
                "{} of the {} requested frequencies lie outside the beam "
                "simulation coverage ({:.2f}-{:.2f} MHz; requested "
                "{:.2f}-{:.2f} MHz): for those channels, the beam is "
                "evaluated at the nearest covered frequency.".format(
                    n_out,
                    freq_array.size,
                    beam_fmin / 1e6,
                    beam_fmax / 1e6,
                    freq_array.min() / 1e6,
                    freq_array.max() / 1e6,
                )
            )
            eval_freqs = np.clip(freq_array, beam_fmin, beam_fmax)

        nfreq = freq_array.size
        logger.info(
            "Interpolating beam onto a %dx%d Cartesian grid for %d frequency "
            "channels (%.1f-%.1f MHz)",
            npix,
            npix,
            nfreq,
            freq_array[0] / 1e6,
            freq_array[-1] / 1e6,
        )
        # interpolate the beam at the (az, za) coordinates of the Cartesian
        # grid, for each frequency (works for both az_za and healpix beams)
        interp_data = beam.interp(
            az_array=az.ravel(),
            za_array=za.ravel(),
            freq_array=eval_freqs,
            return_basis_vector=False,
        )[0]
        cart_beam = interp_data[0, 0].real.reshape(nfreq, npix, npix)
        # peak-normalize per frequency
        cart_beam /= np.max(cart_beam, axis=(1, 2))[:, None, None]

        # ngrid is determined by mapsize + the rx spacing
        dr = np.diff(rx).mean()
        ngrid = int(2.0 * mapsize / dr)
        if ngrid % 2 == 0:
            ngrid += 1

        logger.info("Taking 2D FFT per frequency (ngrid=%d)", ngrid)
        data = np.zeros((nfreq, ngrid, ngrid))
        for ifreq in range(nfreq):
            data[ifreq, :, :] = _ft_beam_2d(
                rx, cart_beam[ifreq, :, :], npix=npix, mapsize=mapsize
            ).real

        return cls(
            data=data,
            pol=pol,
            freq_array=freq_array,
            mapsize=mapsize,
            verbose=verbose,
            x_orientation=x_orientation,
        )

    @classmethod
    def from_file(cls, ftfile, spw_range=None, freq_array=None, **kwargs):
        """
        Read Fourier transform of beam in sky plane from file.

        Initialise FTBeam object by reading file containing the
        Fourier transform of the  instrument beam in the sky plane
        for given frequencies.

        Parameters
        ----------
        ftfile : str
            Path to file constraining to the Fourier transform of the
            beam on the sky plane (Eq. 10 in Memo), including the polarisation
            Ex : path/to/file/ft_beam_HERA_dipole_pI.hdf5
            File must be h5 format.
        spw_range : tuple or 2-element array of ints
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA
            bandwidth).
            If None, whole instrument bandwidth is considered.
        freq_array : 1D array or list of floats, optional
            If given, restrict the FTBeam to these frequencies (in Hz)
            via :meth:`select_freqs`. Only the relevant channel range is
            read from disk, which considerably reduces the memory
            footprint for wide-band FT-beam files. Cannot be used
            together with spw_range.
        """
        # check file path input
        ftfile = Path(ftfile)
        if not (ftfile.exists() and ftfile.is_file() and h5py.is_hdf5(ftfile)):
            raise ValueError("Wrong ftfile input. See docstring.")
        if spw_range is not None and freq_array is not None:
            raise ValueError("Cannot feed both spw_range and freq_array.")

        # obtain bandwidth in file to define spectral window
        with h5py.File(ftfile, "r") as f:
            mapsize = f["mapsize"][0]
            bandwidth = np.array(f["freq"][...])
            # polarisation: prefer the attribute written by write_hdf5,
            # fall back to extracting it from the filename
            pol = f.attrs.get("pol", None)

            if freq_array is not None:
                # read only the contiguous channel range covering the
                # requested frequencies (with one channel of padding for
                # interpolation), rather than the whole bandwidth
                freq_array = np.asarray(freq_array, dtype=float)
                i0 = max(np.searchsorted(bandwidth, freq_array.min()) - 2, 0)
                i1 = min(
                    np.searchsorted(bandwidth, freq_array.max()) + 2, bandwidth.size
                )
                ft_beam = np.array(f["FT_beam"][i0:i1, :, :])
                read_freqs = bandwidth[i0:i1]
            elif spw_range is not None:
                # check format
                assert check_spw_range(spw_range, bandwidth), (
                    "Wrong spw range format, see dosctring."
                )
                read_freqs = bandwidth[spw_range[0] : spw_range[-1]]
                ft_beam = np.array(f["FT_beam"][spw_range[0] : spw_range[1], :, :])
            else:
                # if spectral range is not specified, use whole bandwidth
                spw_range = (0, bandwidth.size)
                read_freqs = bandwidth
                ft_beam = np.array(f["FT_beam"])

        if pol is None:
            pol = str(ftfile).split("_")[-1].split(".")[0]
        else:
            pol = str(pol)

        obj = cls(
            data=ft_beam, pol=pol, freq_array=read_freqs, mapsize=mapsize, **kwargs
        )
        if freq_array is not None:
            # extract (or interpolate onto) the exact requested frequencies
            obj.select_freqs(freq_array)
        return obj

    @classmethod
    def gaussian(
        cls,
        freq_array,
        widths,
        pol,
        mapsize=1.0,
        npix=301,
        cosmo=conversions.Cosmo_Conversions(),
        **kwargs,
    ):
        """
        Initiliase a Gaussian beam on a range of frequencies,
        given a polarisation.

        Parameters
        ----------
        freq_array : list or array of floats
            List of frequencies (in Hz) to define the object on.
        widths : list or array of floats
            List of widths, in deg, to use for the Gaussian beam in the sky
            plane. Can be length 1 if the same value is used for all the frequencies.
            Otherwise, must have same length as freq_array
        pol : str or int
            Can be pseudo-Stokes or power:
            in str form: 'pI', 'pQ', 'pV', 'pU', 'xx', 'yy', 'xy', 'yx'
            in number form: 1, 2, 4, 3, -5, -6, -7, -8
        mapsize : float
            Half width of the map the beam is calculated on, in rad. Increase
            if you want better k_perp resolution.
        npix : int
            Number of pixels to grid sky plane, along one direction.
            Preferably an odd number.
        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified.
        """

        # Frequency-related parameters
        freq_array = np.array(freq_array)
        assert np.size(freq_array) > 2, "Must use at least three frequencies."

        # Beam widths per frequency
        if isinstance(widths, (float, int)):
            widths = np.ones_like(freq_array) * widths
        else:
            assert np.shape(widths) == np.shape(freq_array), (
                "There must be as many frequencies as widths."
            )
        # convert to radian
        if np.mean(widths) < 1:
            warnings.warn("Small widths: make sure the input is in degrees.")
        widths = np.array(widths) * np.pi / 180.0
        # convert widths to Fourier space
        FT_widths = 1.0 / np.pi / widths

        # corresponding grid in Fourier space
        FT_x = np.fft.fftfreq(npix) * npix / 2.0 / mapsize
        FT_x = np.fft.fftshift(FT_x)

        # The Fourier transform of the beam is a Gaussian
        ft_beam = np.zeros((freq_array.size, npix, npix))
        for ifreq in range(freq_array.size):
            Gauss = np.exp(
                -1 * (FT_x**2 + FT_x[:, None] ** 2) / (FT_widths[ifreq] ** 2)
            )
            ft_beam[ifreq] = Gauss * (np.pi / 2.0 / mapsize) * widths[ifreq] ** 2

        return cls(
            data=ft_beam, pol=pol, freq_array=freq_array, mapsize=mapsize, **kwargs
        )

    @classmethod
    def get_bandwidth(cls, ftfile):
        """
        Read FT file to extract bandwidth it was computed along.

        Parameters
        ----------
        ftfile : str
            Path to file constraining to the Fourier transform of the
            beam on the sky plane (Eq. 10 in Memo). The input is the
            root name of the file to use, including the polarisation
            Ex : path/to/file/ft_beam_HERA_dipole_pI.hdf5
            File must be h5 format.

        Returns
        ----------
        bandwidth : array of floats
            List of frequencies covered by the instrument, in Hz.
        """
        ftfile = Path(ftfile)
        if not (ftfile.exists() and ftfile.is_file() and h5py.is_hdf5(ftfile)):
            raise ValueError("Wrong ftfile input. See docstring.")

        with h5py.File(ftfile, "r") as f:
            bandwidth = np.array(f["freq"][...])

        assert bandwidth.size > 1, "Error reading file, empty bandwidth."

        return bandwidth

    def write_hdf5(self, filepath, overwrite=False, extra_attrs=None):
        """
        Write the FTBeam object to an HDF5 file readable by :meth:`from_file`.

        Parameters
        ----------
        filepath : str
            Path of the output file. For backwards compatibility with
            :meth:`from_file` on files without a ``pol`` attribute, it is
            recommended (but not required) that the filename ends in
            ``_<pol>.hdf5``, e.g. ``FT_beam_HERA_vivaldi_pI.hdf5``.
        overwrite : bool, optional
            If True, overwrite an existing file. Default: False.
        extra_attrs : dict, optional
            Additional attributes to store in the file (e.g. provenance
            information). Values must be h5py-compatible scalars/strings.
        """
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"{filepath} exists. Use overwrite=True.")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            f.create_dataset("FT_beam", data=self.ft_beam, compression="gzip")
            f.create_dataset("freq", data=self.freq_array)
            f.create_dataset("mapsize", data=np.array([self.mapsize]))
            f.attrs["pol"] = str(self.pol)
            if extra_attrs is not None:
                for key, val in extra_attrs.items():
                    f.attrs[key] = val
        logger.info("Wrote FTBeam to %s", filepath)

    def select_freqs(self, freq_array, atol=None, inplace=True):
        """
        Restrict the FTBeam to a given array of frequencies.

        If every requested frequency coincides with a native channel of
        the FTBeam (within `atol`), the corresponding channels are simply
        extracted. Otherwise, the FT of the beam is linearly interpolated
        along the frequency axis onto `freq_array`. Use this to put the
        FTBeam on the exact frequency channels of the data: the window
        functions' k_parallel axis is built from this object's
        `freq_array`, so a channel-width mismatch scales that axis.

        Parameters
        ----------
        freq_array : 1D array or list of floats
            Frequencies (in Hz) to restrict the FTBeam to. Must be
            within the bandwidth of the FTBeam.
        atol : float, optional
            Absolute tolerance (in Hz) used to decide whether a requested
            frequency coincides with a native channel. Default: 1e-3
            times the native channel width.
        inplace : bool, optional
            If True (default), modify the object in place. Otherwise
            return a new FTBeam object.

        Returns
        -------
        FTBeam object
            Only if inplace is False.
        """
        fq = self if inplace else copy.deepcopy(self)

        freq_array = np.asarray(freq_array, dtype=float)
        assert freq_array.ndim == 1 and freq_array.size > 1, (
            "Must feed 1D array of at least two frequencies."
        )

        native = np.asarray(fq.freq_array, dtype=float)
        delta_native = np.median(np.diff(native))
        if atol is None:
            atol = 1e-3 * abs(delta_native)

        if (freq_array.min() < native.min() - atol) or (
            freq_array.max() > native.max() + atol
        ):
            raise ValueError(
                "Requested frequencies ({:.2f}-{:.2f} MHz) extend beyond the "
                "bandwidth of the FTBeam ({:.2f}-{:.2f} MHz).".format(
                    freq_array.min() / 1e6,
                    freq_array.max() / 1e6,
                    native.min() / 1e6,
                    native.max() / 1e6,
                )
            )

        # nearest native channel for each requested frequency
        nearest = np.abs(native[:, None] - freq_array[None, :]).argmin(axis=0)

        if np.all(np.abs(native[nearest] - freq_array) <= atol):
            # exact match: extract channels
            fq.ft_beam = fq.ft_beam[nearest, :, :]
        else:
            # grid mismatch: interpolate the FT of the beam in frequency
            warnings.warn(
                "Requested frequencies do not coincide with the FTBeam "
                "channels (native width: {:.2f} kHz, requested: {:.2f} kHz): "
                "interpolating the FT of the beam onto the requested "
                "frequencies. Consider computing the FT beam directly on "
                "the frequency channels of the data.".format(
                    delta_native / 1e3, np.median(np.diff(freq_array)) / 1e3
                )
            )
            fq.ft_beam = interp1d(
                native, fq.ft_beam, axis=0, kind="linear", assume_sorted=False
            )(freq_array)

        fq.freq_array = freq_array.copy()

        if not inplace:
            return fq

    def update_spw(self, spw_range):
        """
        Function to extract spectral window from FTBeam defined on whole bandwidth.

        Extract a section of the previously computed Fourier
        transform of the beam.

        Parameters
        ----------
        spw_range : tuple or 2-element array of ints.
            In (start_chan, end_chan). Must be between 0 and 1024 (HERA
            bandwidth).
        """
        # checks on inputs
        assert check_spw_range(spw_range, self.freq_array), (
            "Wrong spw range format, see dosctring."
        )

        # assign new attributes
        self.spw_range = tuple((int(spw_range[0]), int(spw_range[1])))
        self.freq_array = self.freq_array[self.spw_range[0] : self.spw_range[-1]]
        self.ft_beam = self.ft_beam[self.spw_range[0] : self.spw_range[-1], :, :]


class UVWindow:
    """Class for :class:`UVWindow` objects."""

    def __init__(
        self,
        ftbeam_obj,
        taper=None,
        little_h=True,
        cosmo=conversions.Cosmo_Conversions(),
        verbose=False,
    ):
        """
        Class for :class:`UVWindow` objects.

        Provides :meth:`get_spherical_wf` and :meth:`get_cylindrical_wf`
        to obtain accurate window functions for a given set of baselines
        and spectral range.

        Parameters
        ----------
        ftbeam_obj : (list of) FTBeam object(s)
            List of FTBeam objects.
            If a unique object is given, it is expanded in a matching pair of
            FTBeam objects.
            Its bandwidth and polarisation attributes will define the attributes
            of the UVWindow object.
        taper : str
            Type of data tapering applied along bandwidth.
            See :func:`uvtools.dspec.gen_window` for options.
        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: True (h^-1 Mpc).
        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.

        """
        # Summary attributes

        # cosmology
        assert cosmo is not None, (
            "If no preferred cosmology, do not call input parameter"
        )
        self.cosmo = cosmo

        # units
        self.little_h = bool(little_h)
        if self.little_h:
            self.kunits = units.h / units.Mpc
        else:
            self.kunits = units.Mpc ** (-1)

        # verbose
        self.verbose = bool(verbose)

        # taper
        try:
            dspec.gen_window(taper, 1)
        except ValueError:
            raise ValueError("Wrong taper. See uvtools.dspec.gen_window for options.")
        self.taper = taper

        # create list of FTBeam objects for each polarisation channel
        self.ftbeam_obj_pol = (
            list(ftbeam_obj) if np.size(ftbeam_obj) > 1 else [ftbeam_obj, ftbeam_obj]
        )
        assert hasattr(self.ftbeam_obj_pol[0], "mapsize") and hasattr(
            self.ftbeam_obj_pol[1], "mapsize"
        ), "Wrong input given in ftbeam_obj: must be (a list of) FTBeam object(s)"
        # check if elements in list have same properties
        assert len(self.ftbeam_obj_pol[0].freq_array) == len(
            self.ftbeam_obj_pol[1].freq_array
        ), "Spectral ranges of the two FTBeam objects do not match"
        if len(self.ftbeam_obj_pol[0].freq_array) == len(
            self.ftbeam_obj_pol[1].freq_array
        ):
            assert np.all(
                self.ftbeam_obj_pol[0].freq_array == self.ftbeam_obj_pol[1].freq_array
            ), "Spectral ranges of the two FTBeam objects do not match"
        assert self.ftbeam_obj_pol[0].mapsize == self.ftbeam_obj_pol[1].mapsize, (
            "Physical properties of the two FTBeam objects do not match"
        )

        # extract attributes from FTBeam objects
        self.pols = (self.ftbeam_obj_pol[0].pol, self.ftbeam_obj_pol[1].pol)
        self.freq_array = np.copy(self.ftbeam_obj_pol[0].freq_array)
        self.Nfreqs = len(self.freq_array)
        self.dly_array = utils.get_delays(self.freq_array, n_dlys=len(self.freq_array))
        self.avg_nu = np.mean(self.freq_array)
        self.avg_z = self.cosmo.f2z(self.avg_nu)

    @classmethod
    def from_uvpspec(
        cls, uvp, ipol, spw, ftbeam=None, x_orientation=None, verbose=False
    ):
        """
        Method for :class:`UVWindow` objects.

        Initialises UVWindow object from UVPSpec object.

        Parameters
        ----------
        uvp : UVPSPec object
            UVPSpec object containing at least one polpair, one spw_range
        ipol : int
            Choice of polarisation pair (index of pair in uvp.polpair_array).
        spw : int
            Choice of spectral window (must be in uvp.Nspws).
        ftbeam : str or FTBeam object
            Access to the Fourier transform of the beam on the sky plane
            (Eq. 10 in Memo)
            Options are;
                - Load from file. Then input is the root name of the file
                to use, without the polarisation
                Ex : ft_beam_HERA_dipole (+ path)
                - The FTBeam object to use
                - None (default). Computation from beam simulations (slow).
                Not yet implemented.
        x_orientation: str, optional
            Orientation in cardinal direction east or north of X dipole.
            Default keeps polarization in X and Y basis.
            Used to convert pol str to num and convertly.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
        """
        # Summary attributes: initialise attributes from UVPSpec object

        # cosmology
        if not hasattr(uvp, "cosmo"):
            cosmo = conversions.Cosmo_Conversions()
            warnings.warn("uvp has no cosmo attribute. Using fiducial cosmology.")
        else:
            cosmo = uvp.cosmo

        # units
        little_h = "h^-3" in uvp.norm_units

        # spectral window
        assert spw < uvp.Nspws, "Input spw must be smaller or equal to uvp.Nspws"
        freq_array = uvp.freq_array[uvp.spw_to_freq_indices(spw)]

        # polarisation pair
        polpair = uvputils.polpair_int2tuple(uvp.polpair_array[ipol], pol_strings=True)

        # create FTBeam objects
        ftbeam_obj_pol = []
        for ip, pol in enumerate(polpair):
            if (ip > 0) and (polpair[ip] == polpair[0]):
                # do not recompute if two polarisations are identical
                ftbeam_obj_pol.append(ftbeam_obj_pol[0])
            else:
                if ftbeam is None:
                    raise NotImplementedError(
                        "ftbeam=None (computation from beam simulations) is "
                        "not implemented in from_uvpspec because the UVPSpec "
                        "object does not carry the beam simulation file. "
                        "Construct the FTBeam explicitly with "
                        "FTBeam.from_beam(beamfile, pol, freq_array) and "
                        "pass it as the ftbeam argument."
                    )
                elif isinstance(ftbeam, str | Path):
                    ftbeam_obj_pol.append(
                        FTBeam.from_file(
                            f"{ftbeam}_{pol}.hdf5",
                            spw_range=None,
                            verbose=verbose,
                            x_orientation=x_orientation,
                        )
                    )
                elif isinstance(ftbeam, FTBeam):
                    ftbeam_obj_pol.append(copy.deepcopy(ftbeam))
                else:
                    raise TypeError("Check your ftbeam input.")

        # restrict the FTBeam objects to the exact frequency channels of
        # the UVPSpec's spectral window (interpolating, with a warning,
        # if the FTBeam is not defined on the data's frequency grid)
        for ip in range(2):
            if (ip > 0) and (polpair[ip] == polpair[0]):
                continue
            ftbeam_obj_pol[ip].select_freqs(freq_array)

        return cls(
            ftbeam_obj=ftbeam_obj_pol,
            taper=uvp.taper,
            cosmo=cosmo,
            little_h=little_h,
            verbose=verbose,
        )

    def _get_kgrid(self, bl_len, width=0.020):
        """
        Compute the kperp-array the FT of the beam will be interpolated over.

        Must include all the kperp covered by the spectral window for the
        baseline considered.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        width : float
            Distance between central kperp for given bl_len
            and edge of the kgrid.
            Increasing it will slow down the computation.

        Returns
        ----------
        kgrid : array_like
            (kperp_x) grid corresponding to a given baseline.
            One-dimensional.
        kperp_norm : array_like
            Array of kperp vector norms corresponding to kgrid.
            Two-dimensionsal.
            Computed as sqrt(kperp_x**2+kperp_y**2).

        """
        # central kperp for given baseline (i.e. 2pi*b*nu/c/R)
        # divided by sqrt(2) because 2 dimensions
        kp_centre = (
            self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h)
            * bl_len
            / np.sqrt(2.0)
        )
        # spacing of the numerical Fourier grid, in cosmological units
        dk = (
            2.0
            * np.pi
            / (2.0 * self.ftbeam_obj_pol[0].mapsize)
            / self.cosmo.dRperp_dtheta(
                self.cosmo.f2z(self.freq_array.max()), little_h=self.little_h
            )
        )
        assert width > dk, (
            f"Change width to resolve full window function (dk={dk:.2e})."
        )
        # defines kgrid (kperp_x).
        kgrid = np.arange(kp_centre - width, kp_centre + width, step=dk)
        # array of kperp norms.
        kperp_norm = np.sqrt((kgrid**2)[:, None] + kgrid**2)

        return kgrid, kperp_norm

    def _kperp4bl_freq(self, freq, bl_len, ngrid):
        """
        Compute the range of kperp for a given baseline-freq pair.

        It will be assigned to the FT of the beam.

        Parameters
        ----------
        freq : float
            Frequency (in Hz) considered along the spectral window.
        bl_len : float
            Length of the baseline considered (in meters).
        ngrid : int
            Number of pixels in the FT beam array.
            Internal use only. Do not modify.

        Returns
        ----------
        k : array_like
            Array of k_perp values to match to the FT of the beam.

        """
        assert (freq <= self.freq_array.max()) and (freq >= self.freq_array.min()), (
            "Choose frequency within spectral window."
        )
        assert freq / 1e6 >= 1.0, "Frequency must be given in Hz."

        z = self.cosmo.f2z(freq)
        R = self.cosmo.DM(z, little_h=self.little_h)  # Mpc
        # Fourier dual of sky angle theta
        q = (
            np.fft.fftshift(np.fft.fftfreq(ngrid))
            * ngrid
            / (2.0 * self.ftbeam_obj_pol[0].mapsize)
        )
        k = 2.0 * np.pi / R * (freq * bl_len / conversions.units.c / np.sqrt(2.0) - q)
        k = np.flip(k)

        return k

    def _interpolate_ft_beam(self, bl_len, ft_beam):
        """
        Interpolate the FT of the beam on a regular (kperp,kperp) grid.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        ft_beam : array_like
            Array made of the FT of the beam along the spectral window.
            Must have dimensions (Nfreqs, N, N).

        Returns
        ----------
        interp_ft_beam : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        kperp_norm : array_like
            Norm of the kperp vectors throughout the grid.
            Has dimensions (N,N).

        """
        ft_beam = np.array(ft_beam)
        assert ft_beam.ndim == 3, "ft_beam must be dimension 3."
        assert ft_beam.shape[0] == self.Nfreqs, "ft_beam must have shape (Nfreqs,N,N)"
        assert ft_beam.shape[2] == ft_beam.shape[1], (
            "ft_beam must be square in sky plane"
        )

        # regular kperp_x grid the FT of the beam will be interpolated over.
        # kperp_norm is the corresponding total kperp:
        # kperp = sqrt(kperp_x**2 + kperp_y**2)
        kgrid, kperp_norm = self._get_kgrid(bl_len)

        # assign FT of beam to appropriate kperp grid for each frequency
        # and given baseline length
        ngrid = ft_beam.shape[-1]
        interp_ft_beam = np.zeros((kgrid.size, kgrid.size, self.Nfreqs))
        for i in range(self.Nfreqs):
            # kperp values over frequency array for bl_len
            k = self._kperp4bl_freq(self.freq_array[i], bl_len, ngrid=ngrid)
            # interpolate the FT beam over values onto regular kgrid
            A_real = RegularGridInterpolator(
                (k, k), ft_beam[i], bounds_error=False, fill_value=0.0
            )
            interp_ft_beam[:, :, i] = A_real(np.array([kgrid, kgrid]).T)

        return interp_ft_beam, kperp_norm

    def _take_freq_FT(self, interp_ft_beam, delta_nu):
        """
        Take the Fourier transform along frequency of the beam.

        Parameters
        ----------
        interp_ft_beam : array_like
            FT of the beam, interpolated over a regular (kperp_x,kperp_y) grid.
            Has dimensions (Nfreqs, N, N).
        delta_nu : float
            Frequency resolution (Channel width) in Hz
            along the spectral window.

        Returns
        ----------
        fnu : array_like
            Fourier transform of the beam in sky plane and in frequency.
            Has dimensions (Nfreqs, N, N)
        """
        interp_ft_beam = np.array(interp_ft_beam)
        assert interp_ft_beam.ndim == 3, "interp_ft_beam must be dimension 3."
        assert interp_ft_beam.shape[-1] == self.Nfreqs, (
            "interp_ft_beam must have shape (N,N,Nfreqs)"
        )

        # apply taper along frequency direction
        if self.taper is not None:
            tf = dspec.gen_window(self.taper, self.Nfreqs)
            interp_ft_beam = interp_ft_beam * tf[None, None, :]

        # take numerical FT along frequency axis
        # normalise to appropriate units and recentre
        fnu = np.fft.fftshift(
            np.fft.fft(np.fft.fftshift(interp_ft_beam, axes=-1), axis=-1, norm="ortho")
            * delta_nu**0.5,
            axes=-1,
        )

        return fnu

    def _get_wf_for_tau(self, tau, wf_array1, kperp_bins, kpara_bins):
        """
        Get the cylindrical window function for a given delay.

        Performed after binning on the sky plane.

        Parameters
        ----------
        tau : float
            Delay, in secs.
        wf_array1: array_like
            Window function after cylindrical average on kperp plane.
            Dimensions: (nbins_kperp,nfreq).
        kperp_bins : array_like
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning. Must be linearly spaced.
        kpara_bins : array_like
            1D float array of ascending k_parallel bin centers in [h] Mpc^-1
            units.
            Used for cylindrical binning. Must be linearly spaced.

        Returns
        ----------
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is kperp (obtained with :meth:`get_kperp_bins`).
            Axis 1 is kparallel (obtained with :meth:`get_kpara_bins`).
        kpara : array_like
            Values of kpara corresponding to the axis=2 of cyl_wf.
            Note: these values are weighted by their number of counts
            in the cylindrical binning.
        """
        # read kperp bins and find bin_edges
        kperp_bins = np.array(kperp_bins)
        nbins_kperp = kperp_bins.size
        dk_perp = np.diff(kperp_bins).mean()
        kperp_bin_edges = np.arange(
            kperp_bins.min() - dk_perp / 2, kperp_bins.max() + dk_perp, step=dk_perp
        )
        # read kpara bins
        kpara_bins = np.array(kpara_bins)
        nbins_kpara = kpara_bins.size
        dk_para = np.diff(kpara_bins).mean()
        kpara_bin_edges = np.arange(
            kpara_bins.min() - dk_para / 2, kpara_bins.max() + dk_para, step=dk_para
        )

        # get kparallel grid
        # conversion factor to cosmological units
        alpha = self.cosmo.dRpara_df(self.avg_z, little_h=self.little_h, ghz=False)
        # frequency resolution
        delta_nu = np.median(np.diff(self.freq_array))
        # Fourier dual of frequency (unit 1: FT along theta)
        eta = np.fft.fftshift(np.fft.fftfreq(self.Nfreqs), axes=-1) / delta_nu
        # construct array of |kpara| values for given delay tau
        kpar_norm = np.abs(2.0 * np.pi / alpha * (eta + tau))

        # perform binning along k_parallel (vectorized: bin index of each
        # kpar_norm element such that edges[m] <= kpar_norm < edges[m+1])
        idx = np.digitize(kpar_norm, kpara_bin_edges) - 1
        valid = (idx >= 0) & (idx < nbins_kpara)
        counts = np.bincount(idx[valid], minlength=nbins_kpara)

        # indicator matrix: ind[f, m] = 1 if kpar_norm[f] falls in bin m
        ind = np.zeros((kpar_norm.size, nbins_kpara))
        ind[np.nonzero(valid)[0], idx[valid]] = 1.0

        cyl_wf = np.divide(
            wf_array1 @ ind,
            counts[None, :],
            out=np.zeros((nbins_kperp, nbins_kpara)),
            where=counts[None, :] > 0,
        )
        kpara = np.divide(
            np.bincount(idx[valid], weights=kpar_norm[valid], minlength=nbins_kpara),
            counts,
            out=np.zeros(nbins_kpara),
            where=counts > 0,
        )

        return kpara, cyl_wf

    def get_kperp_bins(self, bl_lens):
        """
        Get spherical k_perp bins for a given set of baseline lengths.

        The function makes sure all values probed by bl_lens are included and
        there is no over-sampling.

        Parameters
        ----------
        bl_lens : list
            List of baseline lengths.
            Can be only one value.

        Returns
        ----------
        kperp_bins : array.
            Array of kperp bins to use.
        """
        bl_lens = np.array(bl_lens)
        assert bl_lens.size > 0, "get_kperp_bins() requires array of baseline lengths."

        dk_perp = np.diff(self._get_kgrid(np.min(bl_lens))[1]).mean() * 5
        kperp_max = (
            self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) * np.max(bl_lens)
            + 10.0 * dk_perp
        )
        kperp_bin_edges = np.arange(dk_perp, kperp_max, step=dk_perp)
        kperp_bins = (kperp_bin_edges[1:] + kperp_bin_edges[:-1]) / 2
        nbins_kperp = kperp_bins.size
        if nbins_kperp > 200:
            warnings.warn(
                "get_kperp_bins: Large number of kperp/kpara bins. "
                "Risk of overresolving and slow computing."
            )

        return kperp_bins * self.kunits

    def get_kpara_bins(self, freq_array):
        """
        Get spherical k_para bins for a given spectra window.

        Requires to initialise UVWinndow object
        with :meth:`set_spw_range` and :meth:`set_spw_parameters`, making
        sure all values probed by freq array are included and there is no
        over-sampling

        Parameters
        ----------
        freq_array : array
            List of frequencies you want to compute window functions over.
            In Hz.

        Returns
        ----------
        kpara_bins : array.
            Array of k_parallel bins to use.

        """

        freq_array = np.array(freq_array)
        assert freq_array.size > 1, "Must feed list of frequencies."

        dly_array = utils.get_delays(freq_array, n_dlys=len(freq_array))
        avg_z = self.cosmo.f2z(np.mean(freq_array))

        # define default kperp bins,
        dk_para = (
            self.cosmo.tau_to_kpara(avg_z, little_h=self.little_h)
            * np.diff(dly_array)[0]
        )
        kpara_max = (
            self.cosmo.tau_to_kpara(avg_z, little_h=self.little_h)
            * abs(dly_array).max()
            + 10.0 * dk_para
        )
        kpara_bin_edges = np.arange(dk_para / 2.0, kpara_max, step=dk_para)
        kpara_bins = (kpara_bin_edges[1:] + kpara_bin_edges[:-1]) / 2
        nbins_kpara = kpara_bins.size

        if nbins_kpara > 200:
            warnings.warn(
                "get_kpara_bins: Large number of kperp/kpara bins. "
                "Risk of overresolving and slow computing."
            )

        return kpara_bins * self.kunits

    def get_cylindrical_wf(
        self, bl_len, kperp_bins=None, kpara_bins=None, return_bins=None, verbose=None
    ):
        """
        Get the cylindrical window function for a baseline length.

        Cylindrical wf correspond to in (kperp,kpara) space
        for a given baseline and polarisation, along the spectral window.

        Parameters
        ----------
        bl_len : float
            Length of the baseline considered, in meters.
        kperp_bins : array_like of astropy quantity, with units.
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with :attr:`little_h`.
            If computing for different baselines, make sure to input identical
            arrays.
        kpara_bins : array_like of astropy.quantity, with units.
            1D float array of ascending k_parallel bin centers.
            Used for cylindrical binning.
            Make sure the values are consistent with :attr:`little_h`.
        return_bins : str
            If 'weighted', return bins weighted by the actual modes inside
            of bin.
            If 'unweighted', return bins used to build the histogram.
            If None, does not return anything. Bins can later be retrieved with
            :meth:`get_kperp_bins` and :meth:`get_kpara_bins`.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
            If None, value used is the class attribute.

        Returns
        ----------
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is the array of delays considered (:attr:`dly_array`).
            Axis 1 is kperp (obtained with :meth:`get_kperp_bins`).
            Axis 2 is kparallel (obtained with :meth:`get_kpara_bins`).
        kperp : array_like
            Values of kperp corresponding to the axis=1 of cyl_wf.
            Note: if return_bins='weighted', these values are weighted
            by their number of counts in the cylindrical binning.
        kpara : array_like
            Values of kpara corresponding to the axis=2 of cyl_wf.
            Note: if return_bins='weighted', these values are weighted
            by their number of counts in the cylindrical binning.

        """
        # INITIALISE PARAMETERS

        if verbose is None:
            verbose = self.verbose

        # k-bins for cylindrical binning
        if kperp_bins is None:
            kperp_bins = self.get_kperp_bins([bl_len])
        else:
            self.check_kunits(kperp_bins)
        kperp_bins = np.array(kperp_bins.value)
        if not np.allclose(np.diff(kperp_bins), np.diff(kperp_bins)[0]):
            raise ValueError("get_cylindrical_wf: kperp_bins must be linearly spaced.")
        nbins_kperp = kperp_bins.size
        dk_perp = np.diff(kperp_bins).mean()
        kperp_bin_edges = np.arange(
            kperp_bins.min() - dk_perp / 2, kperp_bins.max() + dk_perp, step=dk_perp
        )
        kperp_centre = (
            self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) * bl_len
        )
        if (kperp_bin_edges.max() < kperp_centre + 3 * dk_perp) or (
            kperp_bin_edges.min() > kperp_centre - 3 * dk_perp
        ):
            warnings.warn(
                "get_cylindrical_wf: The bin centre is not included "
                "in the array of kperp bins given as input."
            )

        if kpara_bins is None:
            kpara_bins = self.get_kpara_bins(self.freq_array)
        else:
            self.check_kunits(kpara_bins)
        kpara_bins = np.array(kpara_bins.value)
        if not np.allclose(np.diff(kpara_bins), np.diff(kpara_bins)[0]):
            raise ValueError("get_cylindrical_wf: kpara_bins must be linearly spaced.")
        nbins_kpara = kpara_bins.size
        dk_para = np.diff(kpara_bins).mean()
        kpara_bin_edges = np.arange(
            kpara_bins.min() - dk_para / 2, kpara_bins.max() + dk_para, step=dk_para
        )
        kpara_centre = (
            self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h)
            * abs(self.dly_array).max()
        )
        if (kpara_bin_edges.max() < kpara_centre + 3 * dk_para) or (
            kpara_bin_edges.min() > kpara_centre - 3 * dk_para
        ):
            warnings.warn(
                "get_cylindrical_wf: The bin centre is not included "
                "in the array of kpara bins given as input."
            )

        # COMPUTE CYLINDRICAL WINDOW FUNCTIONS

        # take fourier transform along frequency for each polarisation channel
        fnu = []
        for ip in range(len(self.pols)):
            ft_beam = np.copy(self.ftbeam_obj_pol[ip].ft_beam)
            # interpolate FT of beam onto regular grid of (kperp_x,kperp_y)
            interp_ft_beam, kperp_norm = self._interpolate_ft_beam(bl_len, ft_beam)
            # frequency resolution
            delta_nu = abs(self.freq_array[-1] - self.freq_array[0]) / self.Nfreqs
            # obtain FT along frequency
            fnu.append(self._take_freq_FT(interp_ft_beam, delta_nu))

        # cylindrical average

        # on sky plane (vectorized: the binning of the (kperp_x, kperp_y)
        # grid onto kperp bins is identical for all frequencies)
        idx = np.digitize(kperp_norm.ravel(), kperp_bin_edges) - 1
        valid = (idx >= 0) & (idx < nbins_kperp)
        counts = np.bincount(idx[valid], minlength=nbins_kperp)

        # real part of the cross-pol product, flattened on the sky plane;
        # real() commutes with the sum performed in the binned average
        prod = (np.conj(fnu[1]) * fnu[0]).real.reshape(-1, self.Nfreqs)
        sums = np.zeros((nbins_kperp, self.Nfreqs))
        np.add.at(sums, idx[valid], prod[valid])
        wf_array1 = np.divide(
            sums,
            counts[:, None],
            out=np.zeros_like(sums),
            where=counts[:, None] > 0,
        )
        kperp = np.divide(
            np.bincount(
                idx[valid], weights=kperp_norm.ravel()[valid], minlength=nbins_kperp
            ),
            counts,
            out=np.zeros(nbins_kperp),
            where=counts > 0,
        )

        # in frequency direction
        cyl_wf = np.zeros((self.Nfreqs, nbins_kperp, nbins_kpara))
        for it, tau in enumerate(self.dly_array[: self.Nfreqs // 2 + 1]):
            kpara, cyl_wf[it, :, :] = self._get_wf_for_tau(
                tau, wf_array1, kperp_bins, kpara_bins
            )
        # fill by symmetry for tau = -tau
        if self.Nfreqs % 2 == 0:
            cyl_wf[self.Nfreqs // 2 + 1 :, :, :] = np.flip(cyl_wf, axis=0)[
                self.Nfreqs // 2 : -1
            ]
        else:
            cyl_wf[self.Nfreqs // 2 + 1 :, :, :] = np.flip(cyl_wf, axis=0)[
                self.Nfreqs // 2 + 1 :
            ]
        # except for tau at 1/4 and 3/4 of the spectral window for symmetry reasons
        _, cyl_wf[self.Nfreqs - self.Nfreqs // 4, :, :] = self._get_wf_for_tau(
            -self.dly_array[self.Nfreqs // 4], wf_array1, kperp_bins, kpara_bins
        )

        # normalisation of window functions
        sum_per_bin = np.sum(cyl_wf, axis=(1, 2))[:, None, None]
        cyl_wf = np.divide(cyl_wf, sum_per_bin, where=sum_per_bin != 0)

        if (return_bins == "unweighted") or return_bins:
            return kperp_bins, kpara_bins, cyl_wf
        elif return_bins == "weighted":
            return kperp, kpara, cyl_wf
        else:
            return cyl_wf

    def cylindrical_to_spherical(self, cyl_wf, kbins, ktot, bl_lens, bl_weights=None):
        """
        Take spherical average of cylindrical window functions.

        Parameters
        ----------
        cyl_wf : array_like
            Window function as a function of (kperp,kpara).
            Axis 0 is the array of baseline lengths
            Axis 1 is the array of delays considered (self.dly_array).
            Axis 2 is kperp.
            Axis 3 is kparallel.
            If only one bl_lens is given, then axis 0 can be omitted.
        kbins : array-like astropy.quantity with units
            1D float array of ascending |k| bin centers in [h] Mpc^-1 units.
            Used for spherical binning. Must be linearly spaced.
        ktot : array_like
            2-dimensional array giving the magnitude of k corresponding to
            kperp and kpara in cyl_wf.
        bl_lens : list.
            List of baseline lengths used to compute cyl_wf.
            Must have same length as same size as cyl_wf.shape[0].
        bl_weights : list of weights (float or int), optional
            Relative weight of each baseline-length when performing
            the average. This should have the same shape as bl_lens.
            Default: None (all baseline pairs have unity weights).

        Returns
        ----------
        wf_spherical : array
            Array of spherical window functions.
            Shape (nbinsk, nbinsk).
        weighted_k : array
            Returns weighted k-modes.
        """
        # check bl_lens and bl_weights are consistent
        bl_lens = (
            bl_lens if isinstance(bl_lens, (list, tuple, np.ndarray)) else [bl_lens]
        )
        bl_lens = np.array(bl_lens)
        if bl_weights is None:
            # assign weight of one to each baseline length
            bl_weights = np.ones(bl_lens.size)
        else:
            bl_weights = np.array(bl_weights)
            assert bl_weights.size == bl_lens.size, (
                "Blpair weights and lengths do not match"
            )

        # if cyl_wf were computed only for one baseline length
        if cyl_wf.ndim == 3:
            assert bl_lens.size == 1, (
                "If only one bl_lens is given,"
                "cyl_wf must be of dimensions (ndlys,nkperp,nkpara)"
            )
            cyl_wf = cyl_wf[None]
        # check shapes are consistent
        assert bl_lens.size == cyl_wf.shape[0]
        assert ktot.shape == cyl_wf.shape[2:], (
            "k magnitude grid does not match (kperp,kpara) grid in cyl_wf"
        )

        # k-bins for spherical binning
        assert len(kbins) > 1, "must feed array of k bins for spherical average"
        self.check_kunits(kbins)  # check k units
        kbins = np.array(kbins.value)
        if not np.allclose(np.diff(kbins), np.diff(kbins)[0]):
            raise ValueError("cylindrical_to_spherical: kbins must be linearly spaced.")
        nbinsk = kbins.size
        dk = np.diff(kbins).mean()
        kbin_edges = np.arange(kbins.min() - dk / 2, kbins.max() + dk, step=dk)

        # construct array giving the k probed by each baseline-tau pair
        kperps = self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) * bl_lens
        kparas = (
            self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h) * self.dly_array
        )
        kmags = np.sqrt(kperps[:, None] ** 2 + kparas**2)

        # take average
        wf_spherical = np.zeros((nbinsk, nbinsk))
        kweights, weighted_k = np.zeros(nbinsk, dtype=int), np.zeros(nbinsk)
        for m1 in range(nbinsk):
            mask2 = (kbin_edges[m1] <= kmags) & (kmags < kbin_edges[m1 + 1])
            if np.any(mask2):
                weighted_k[m1] = np.mean(kmags[mask2])
                mask2 = (
                    mask2.astype(int) * bl_weights[:, None]
                )  # add weights for redundancy
                kweights[m1] = np.sum(mask2)
                wf_temp = np.sum(
                    cyl_wf * mask2[:, :, None, None], axis=(0, 1)
                ) / np.sum(mask2)
                if np.sum(wf_temp) > 0.0:
                    for m in range(nbinsk):
                        mask = (kbin_edges[m] <= ktot) & (ktot < kbin_edges[m + 1])
                        if np.any(mask):  # cannot compute mean if zero elements
                            wf_spherical[m1, m] = np.mean(wf_temp[mask])
                    # normalisation
                    wf_spherical[m1, :] = np.divide(
                        wf_spherical[m1, :],
                        np.sum(wf_spherical[m1, :]),
                        where=np.sum(wf_spherical[m1, :]) != 0,
                    )

        if np.any(kweights == 0.0) and self.verbose:
            warnings.warn(
                "Some spherical bins are empty. "
                "Add baselines or expand spectral window."
            )

        return wf_spherical, weighted_k

    def get_spherical_wf(
        self,
        kbins,
        bl_lens,
        bl_weights=None,
        kperp_bins=None,
        kpara_bins=None,
        return_weighted_k=False,
        verbose=None,
    ):
        """
        Get spherical window functions for a UVWindow object.

        Requires set of baselines, polarisation, spectral range, and
        a set of kbins used for averaging.

        Parameters
        ----------
        kbins : array-like astropy.quantity with units
            1D float array of ascending |k| bin centers in [h] Mpc^-1 units.
            Using for spherical binning.
            Make sure the values are consistent with :attr:`little_h`.
        bl_lens : list.
            List of lengths corresponding to each group
            (can be redundant groups from utils.get_reds).
            Must have same length as bl_weights.
        bl_weights : list, optional.
            List baselines weights. Must have same length as bl_lens.
            If None, a weight of 1 is attributed to
            each bl_len.
        kperp_bins : array-like astropy.quantity with units, optional.
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with :attr:`little_h`.
        kpara_bins : array-like astropy.quantity with units, optional.
            1D float array of ascending k_parallel bin centers.
            Used for cylindrical binning.
            Make sure the values are consistent with :attr:`little_h`.
        return_weighted_k : bool, optional
            Return the weighted k-mode corresponding to each bin.
            Default is False.
        verbose : bool, optional
            If True, print progress, warnings and debugging info to stdout.
            If None, value used is the class attribute.


        Returns
        ----------
        wf_spherical : array
            Array of spherical window functions.
            Shape (nbinsk, nbinsk).
        weighted_k : array
            If return_weighted_k is True.
            Returns weighted k-modes corresponding to each k-bin.
        """
        # INITIALISE PARAMETERS

        if verbose is None:
            verbose = self.verbose

        bl_lens = np.atleast_1d(np.asarray(bl_lens, dtype=float)).reshape(-1)
        nbls = bl_lens.size  # number of redudant groups
        if bl_weights is not None:
            # check consistency of baseline-related inputs
            assert len(bl_weights) == nbls, (
                "bl_weights and bl_lens must have same length"
            )
            bl_weights = np.array(bl_weights)
        else:
            # each baseline length has weight one
            bl_weights = np.ones(nbls)

        # k-bins for cylindrical binning
        if kperp_bins is None or len(kperp_bins) == 0:
            # define default kperp bins, making sure all values probed by
            # bl_lens are included and there is no over-sampling
            kperp_bins = self.get_kperp_bins(bl_lens)
        else:
            self.check_kunits(kperp_bins)
        kperp_bins = np.array(kperp_bins.value)
        if not np.allclose(np.diff(kperp_bins), np.diff(kperp_bins)[0]):
            raise ValueError("get_spherical_wf: kperp_bins must be linearly spaced.")
        nbins_kperp = kperp_bins.size
        dk_perp = np.diff(kperp_bins).mean()
        kperp_bin_edges = np.arange(
            kperp_bins.min() - dk_perp / 2, kperp_bins.max() + dk_perp, step=dk_perp
        )
        # make sure proper kperp values are included in given bins
        # raise warning otherwise
        kperp_max = self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) * np.max(
            bl_lens
        )  # + 10.*dk_perp
        kperp_min = self.cosmo.bl_to_kperp(self.avg_z, little_h=self.little_h) * np.min(
            bl_lens
        )  # + 10.*dk_perp
        if kperp_bin_edges.max() <= kperp_max:
            warnings.warn(
                "get_spherical_wf: Max kperp bin centre not included in binning array"
            )
        if kperp_bin_edges.min() >= kperp_min:
            warnings.warn(
                "get_spherical_wf: Min kperp bin centre not included in binning array"
            )

        if kpara_bins is None or len(kpara_bins) == 0:
            # define default kperp bins, making sure all values probed by freq
            # array are included and there is no over-sampling
            kpara_bins = self.get_kpara_bins(self.freq_array)
        else:
            self.check_kunits(kpara_bins)
        kpara_bins = np.array(kpara_bins.value)
        if not np.allclose(np.diff(kpara_bins), np.diff(kpara_bins)[0]):
            raise ValueError("get_spherical_wf: kpara_bins must be linearly spaced.")
        nbins_kpara = kpara_bins.size
        dk_para = np.diff(kpara_bins).mean()
        kpara_bin_edges = np.arange(
            kpara_bins.min() - dk_para / 2, kpara_bins.max() + dk_para, step=dk_para
        )
        kpara_centre = (
            self.cosmo.tau_to_kpara(self.avg_z, little_h=self.little_h)
            * abs(self.dly_array).max()
        )
        # make sure proper kpara values are included in given bins
        # raise warning otherwise
        if (kpara_bin_edges.max() <= kpara_centre + 3 * dk_para) or (
            kpara_bin_edges.min() >= kpara_centre - 3 * dk_para
        ):
            warnings.warn(
                "get_spherical_wf: The bin centre is not included "
                "in the array of kpara bins given as input."
            )

        # array of |k|=sqrt(kperp**2+kpara**2)
        ktot = np.sqrt(kperp_bins[:, None] ** 2 + kpara_bins**2)

        # k-bins for spherical binning
        self.check_kunits(kbins)  # check k units
        assert kbins.value.size > 1, "must feed array of k bins for spherical average"
        nbinsk = kbins.value.size
        if not np.allclose(np.diff(kbins), np.diff(kbins)[0]):
            raise ValueError("get_spherical_wf: kbins must be linearly spaced.")
        dk = np.diff(kbins.value).mean()
        kbin_edges = np.arange(
            kbins.value.min() - dk / 2, kbins.value.max() + dk, step=dk
        )
        # make sure proper ktot values are included in given bins
        # raise warning otherwise
        if kbin_edges.max() <= ktot.max():
            warnings.warn("Max spherical k probed is not included in bins.")
        if kbin_edges.min() >= ktot.min():
            warnings.warn("Min spherical k probed is not included in bins.")

        # COMPUTE THE WINDOW FUNCTIONS
        # get cylindrical window functions for each baseline length considered
        # as a function of (kperp, kpara)
        cyl_wf = np.zeros((nbls, self.Nfreqs, nbins_kperp, nbins_kpara))
        for ib in range(nbls):
            if verbose:
                sys.stdout.write(f"\rComputing for blg {ib + 1:d} of {nbls:d}...")
            cyl_wf[ib, :, :, :] = self.get_cylindrical_wf(
                bl_len=bl_lens[ib],
                kperp_bins=kperp_bins * self.kunits,
                kpara_bins=kpara_bins * self.kunits,
                verbose=verbose,
            )
        if verbose:
            sys.stdout.write(f"\rComputing for blg {nbls:d} of {nbls:d}... \n")

        # perform spherical binning
        wf_spherical, weighted_k = self.cylindrical_to_spherical(
            cyl_wf, kbins, ktot, bl_lens, bl_weights
        )

        if return_weighted_k:
            return wf_spherical, weighted_k
        else:
            return wf_spherical

    def run_and_write(
        self,
        filepath,
        bl_lens,
        bl_weights=None,
        kperp_bins=None,
        kpara_bins=None,
        clobber=False,
    ):
        """
        Run cylindrical wf and write result to HDF5 file.

        Parameters
        ----------
        filepath : str
            Filepath for output file.
        bl_lens : list, optional.
            List of lengths corresponding to each group
            (can be redundant groups from utils.get_reds).
            Must have same length as bl_weights.
        bl_weights : list, optional.
            List baselines weights. Must have same length as bl_lens.
            If None, a weight of 1 is attributed to
            each bl_len.
        kperp_bins : array-like astropy.quantity with unit, optional.
            1D float array of ascending k_perp bin centers in [h] Mpc^-1 units.
            Used for cylindrical binning,
            Make sure the values are consistent with :attr:`little_h`.
        kpara_bins : array-like astropy.quantity with units, optional.
            1D float array of ascending k_parallel bin centers.
            Used for cylindrical binning.
            Make sure the values are consistent with :attr:`little_h`.
        clobber : bool, optional
            Whether to overwrite output file if it exists. Default: False.
        """
        # Check output
        filepath = Path(filepath)
        if filepath.exists() and clobber is False:
            raise OSError(f"{filepath} exists, not overwriting...")
        elif filepath.exists() and clobber is True:
            print(f"{filepath} exists, overwriting...")
            os.remove(filepath)

        bl_lens = np.atleast_1d(np.asarray(bl_lens, dtype=float)).reshape(-1)
        nbls = bl_lens.size  # number of redudant groups
        if bl_weights is not None:
            # check consistency of baseline-related inputs
            assert len(bl_weights) == nbls, (
                "bl_weights and bl_lens must have same length"
            )
            bl_weights = np.array(bl_weights)
        else:
            # each baseline length has weight one
            bl_weights = np.ones(nbls)

        # k-bins for cylindrical binning
        if kperp_bins is None or len(kperp_bins) == 0:
            # define default kperp bins, making sure all values probed by
            # bl_lens are included and there is no over-sampling
            kperp_bins = self.get_kperp_bins(bl_lens)
        else:
            self.check_kunits(kperp_bins)
        kperp_bins = np.array(kperp_bins.value)
        nbins_kperp = kperp_bins.size

        if kpara_bins is None or len(kpara_bins) == 0:
            # define default kperp bins, making sure all values probed by freq
            # array are included and there is no over-sampling
            kpara_bins = self.get_kpara_bins(self.freq_array)
        else:
            self.check_kunits(kpara_bins)
        kpara_bins = np.array(kpara_bins.value)
        nbins_kpara = kpara_bins.size

        # COMPUTE THE WINDOW FUNCTIONS

        # get cylindrical window functions for each baseline length considered
        # as a function of (kperp, kpara)
        cyl_wf = np.zeros((nbls, self.Nfreqs, nbins_kperp, nbins_kpara))
        for ib in range(nbls):
            cyl_wf[ib, :, :, :] = (
                self.get_cylindrical_wf(
                    bl_lens[ib], kperp_bins * self.kunits, kpara_bins * self.kunits
                )
                * bl_weights[ib]
            )

        # Write file
        with h5py.File(filepath, "w") as f:
            # parameters
            f.attrs["avg_nu"] = self.avg_nu
            f.attrs["avg_z"] = self.avg_z
            f.attrs["nfreqs"] = self.Nfreqs
            f.attrs["little_h"] = self.little_h
            f.attrs["polpair"] = uvputils.polpair_tuple2int(self.pols)
            # cannot write None to file
            if self.taper is None:
                f.attrs["taper"] = "none"
            else:
                f.attrs["taper"] = self.taper
            # arrays
            f.create_dataset(
                "kperp_bins", shape=(nbins_kperp,), data=kperp_bins, dtype=float
            )
            f.create_dataset(
                "kpara_bins", shape=(nbins_kpara,), data=kpara_bins, dtype=float
            )
            f.create_dataset("bl_lens", shape=(nbls,), data=bl_lens, dtype=float)
            f.create_dataset("bl_weights", shape=(nbls,), data=bl_weights, dtype=float)
            f.create_dataset(
                "dly_array", shape=(self.Nfreqs,), data=self.dly_array, dtype=float
            )
            f.create_dataset(
                "cyl_wf",
                shape=(nbls, self.Nfreqs, nbins_kperp, nbins_kpara),
                data=cyl_wf,
                dtype=float,
            )

    def check_kunits(self, karray):
        """
        Check unit consistency between k's throughout code.

        Parameters
        ----------
        karray : array-like
            Array of Fourier modes.
        """
        try:
            karray.unit
        except AttributeError:
            raise AttributeError("Feed k array with units (astropy.units).")
        assert self.kunits.is_equivalent(karray.unit), (
            "k array units not consistent with little_h"
        )


def check_spw_range(spw_range, bandwidth=None):
    """
    Check if spw_range required is in correct format and if
    it is compatible with bandwidth when given,

    Parameters
    ----------
    spw_range : tuple or 2-element array of ints
        In (start_chan, end_chan). Must be between 0 and 1024 (HERA
        bandwidth).
        If None, whole instrument bandwidth is considered.
    bandwidth : 1D array or list of floats
        Array or list of the frequencies over which the Fourier
        transform of the beam has been computed (in Hz).
        If None, does not include bandwidth-related checks.
        Default is None.

    Returns
    ----------
        True if compatible.
        False if not.
    """
    # check format
    if np.size(spw_range) != 2:
        return False
    if spw_range[1] - spw_range[0] <= 0:
        return False
    if min(spw_range) < 0:
        return False
    # bandwidth-related checks
    if bandwidth is not None:
        bandwidth = np.array(bandwidth)
        if max(spw_range) > bandwidth.size:
            return False

    return True

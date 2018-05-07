``PSpecBeam``: Primary beam models
==================================

``PSpecBeam`` objects carry information about the primary beam, such as how the beam solid angle varies with frequency. This information is needed to rescale power spectra into cosmological units, through the computation of a 'beam scalar'.

The main purpose of ``PSpecBeam`` objects is to provide the :class:`~hera_pspec.PSpecData` class with a way of normalizing the power spectra that it produces, using the :meth:`~hera_pspec.pspecbeam.PSpecBeamBase.compute_pspec_scalar` method. To attach a ``PSpecBeam`` object to a ``PSpecData`` object, pass one in when you instantiate the class, e.g.

.. code-block:: python
  
  # Create PSpecBeamUV from a beamfits file
  beam = hp.PSpecBeamUV('HERA_Beams.beamfits')
  
  # Attach beam to PSpecData object
  psd = hp.PSpecData(dsets=[], wgts=[], beam=beam)

Another purpose of ``PSpecBeam`` objects is to convert flux densities to temperature units using the :meth:`~hera_pspec.pspecbeam.PSpecBeamBase.Jy_to_mK` method, e.g.

.. code-block:: python

  # Apply unit conversion factor to UVData
  uvd = UVData()
  uvd.read_miriad(datafile) # Load data (assumed to be in Jy units)
  uvd.data_array *= beam.Jy_to_mK(np.unique(uvd.freq_array))[None, None, :, None]
  # (The brackets [] are needed to match the shape of uvd.data_array)

Note that ``PSpecBeam`` objects have a cosmology attached to them. If you don't specify a cosmology (with the ``cosmo`` keyword argument), they will be instantiated with the default cosmology from `hera_pspec.conversions`.

There are several different types of ``PSpecBeam`` object:

.. contents::
  :local:


``PSpecBeamBase``: Base class for ``PSpecBeam``
-----------------------------------------------

This is the base class for all other ``PSpecBeam`` objects. It provides the generic :meth:`~hera_pspec.PSpecBeamBase.compute_pspec_scalar` and :meth:`~hera_pspec.PSpecBeamBase.Jy_to_mK` methods, but subclasses must provide their own ``power_beam_int`` and ``power_beam_sq_int`` methods.

.. autoclass:: hera_pspec.pspecbeam.PSpecBeamBase
  :members:


``PSpecBeamUV``: Beams from a ``UVBeam`` object
-----------------------------------------------

This class allows you to use any beam that is supported by the ``UVBeam`` class in the ``pyuvdata`` package. These usually contain Healpix-pixelated beams as a function of frequency and polarization.

To create a beam that uses this format, simply create a new ``PSpecBeamUV`` instance with the name of a ``beamfits`` file that is supported by ``UVBeam``, e.g.

.. code-block:: python
  
  beam = hp.PSpecBeamUV('HERA_Beams.beamfits')

.. autoclass:: hera_pspec.pspecbeam.PSpecBeamUV
  :members: __init__, power_beam_int, power_beam_sq_int
  :inherited-members:
  :member-order: bysource
  

``PSpecBeamGauss``: Simple Gaussian beam model
----------------------------------------------

A Gaussian beam type is provided for simple testing purposes. You can specify a beam FWHM that is constant in frequency, for the ``I`` (pseudo-Stokes I) polarization channel only.

For example, to specify a Gaussian beam with a constant FWHM of 0.1 radians, defined over a frequency interval of [100, 200] MHz:

.. code-block:: python
  
  # Each beam is defined over a frequency interval:
  beam_freqs = np.linspace(100e6, 200e6, 200) # in Hz
  
  # Create a new Gaussian beam object with full-width at half-max. of 0.1 radians
  beam_gauss = hp.PSpecBeamGauss(fwhm=0.1, beam_freqs=beam_freqs)

.. autoclass:: hera_pspec.pspecbeam.PSpecBeamGauss
  :members: __init__, power_beam_int, power_beam_sq_int
  


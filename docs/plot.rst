Simple plotting functions
=========================

The ``hera_pspec.plot`` module contains functions for making simple plots of delay power spectra.

The following example plots the power spectra from a ``UVPSpec`` object, averaged over baseline-pairs and times.

.. code-block:: python

  # Load or generate a UVPSpec object containing delay power spectra
  uvp = ...

  # Set which baseline-pairs should be included in the plot.
  # uvp.get_blpairs() returns nested tuples like ((ant1, ant2), (ant3, ant4)).
  blpairs = uvp.get_blpairs()

  # Plot the delay spectrum, averaged over all blpairs and times
  # (for the spectral-window index 0, and auto-polarization 'xx')
  fig = hp.plot.delay_spectrum(uvp, [blpairs], 0, 'xx',
                               average_blpairs=True, average_times=True,
                               delay=False)

  # Setting delay=False plots the power spectrum in cosmological units
  # Use times=uvp.time_avg_array[...] to select specific integrations.
  # By default, static metadata is written to the title and only the
  # varying metadata is written to the legend. Pass title_legend=False
  # to disable the automatic title/legend text entirely.

For a more extensive worked example, see `this example Jupyter notebook <https://github.com/HERA-Team/hera_pspec/blob/master/examples/Plotting_examples.ipynb>`_.

The only plotting function currently available in the ``hera_pspec.plot`` module is ``delay_spectrum()``.

.. autofunction:: hera_pspec.plot.delay_spectrum

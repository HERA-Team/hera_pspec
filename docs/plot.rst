Simple plotting functions
=========================

The ``hera_pspec.plot`` module contains functions for making simple plots of delay power spectra.

The following example plots the power spectra from a ``UVPSpec`` object, averaged over baseline-pairs and times.

.. code-block:: python
  
  # Load or generate a UVPSpec object containing delay power spectra
  uvp = ...
  
  # Set which baseline-pairs should be included in the plot
  blpairs = list(uvp.blpair_array) # This includes all blpairs!
  
  # Plot the delay spectrum, averaged over all blpairs and times
  # (for the spectral window with index=0, and polarization 'xx')
  ax = hp.plot.delay_spectrum(uvp, [blpairs,], spw=0, pol='xx', 
                              average_blpairs=True, average_times=True, 
                              delay=False)
  
  # Setting delay=False plots the power spectrum in cosmological units
  
For a more extensive worked example, see `this example Jupyter notebook <https://github.com/HERA-Team/hera_pspec/blob/master/examples/Plotting_examples.ipynb>`_.

The only plotting function currently available in the `hera_pspec.plot` module is `delay_spectrum()`.

.. automodule:: plot
  :members:


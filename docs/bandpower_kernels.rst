Window Functions and Covariances
================================

``BandpowerWindow`` and ``BandpowerCovariance`` store the window function or
covariance matrix of every (baseline-pair-time, polarization-pair) entry of one
spectral window. Each distinct matrix is stored once, and an integer
``block_index`` of shape ``(Nbltpairs, Npols)`` says which matrix each entry uses,
so matrices shared across times or baselines cost no extra memory or disk.

``UVPSpec`` holds them per spectral window in ``uvp.window_functions``,
``uvp.cov_real`` and ``uvp.cov_imag``. The older ``uvp.window_function_array``,
``uvp.cov_array_real`` and ``uvp.cov_array_imag`` give the same values as
read-only dense arrays. ``UVPSpec.write_hdf5(..., compact=True)`` writes the
kernels to disk instead of dense arrays.

.. autoclass:: hera_pspec.BandpowerKernel
  :members:

.. autoclass:: hera_pspec.BandpowerWindow
  :members:
  :show-inheritance:

.. autoclass:: hera_pspec.BandpowerCovariance
  :members:
  :show-inheritance:

Response matrices and one-step normalization
============================================

:meth:`~hera_pspec.PSpecData.pspec` can normalize power spectra in two ways,
chosen with its ``normalization`` argument:

* ``"two-step"`` (the default): the bandpowers are normalized with the
  dimensionless ``G`` or ``H`` matrices and then multiplied by a scalar that
  converts Hz sr to a cosmological volume. With ``exact_norm=True`` the
  result is labelled ``"two-step-exact"``.
* ``"one-step"``: the normalization matrix comes from a response matrix that
  already includes the beam and the cosmology, so the result is in
  cosmological units directly
  (`#329 <https://github.com/HERA-Team/hera_pspec/issues/329>`_). It needs a
  beam.

The response is ``H_ab = tr[E_a dC/dp_b] / 2``, where ``dC/dp_b`` is the
response of the data covariance to bandpower ``b``
(`#328 <https://github.com/HERA-Team/hera_pspec/issues/328>`_).
:class:`~hera_pspec.DelayResponse` is the built-in model. Any object with a
method ``matrices(freqs, delays, beam, pol, cosmo, length_unit)`` that returns
``dC/dp_b`` as a Quantity of shape ``(Ndlys, Nfreqs, Nfreqs)`` in
``1 / length_unit**3`` can be used instead.

.. code-block:: python

  uvp = ds.pspec(bls1, bls2, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)],
                 normalization="one-step",
                 response=hp.DelayResponse(beam="healpix"))
  uvp.normalization    # 'one-step'
  uvp.response_model   # the response used, as a string

For one-step spectra ``uvp.scalar_array`` holds a reference scalar that was
not applied to the data; :meth:`~hera_pspec.UVPSpec.set_cosmology` uses it to
rescale the spectra. See :doc:`migrating_to_v1` for how one-step results
compare with two-step.

.. autoclass:: hera_pspec.DelayResponse
  :members: kernel, estimator_weight
  :noindex:

.. automethod:: hera_pspec.PSpecData.get_response
  :noindex:

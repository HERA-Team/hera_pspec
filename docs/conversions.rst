Cosmology and units
===================

``hera_pspec`` uses ``astropy`` for units and cosmology. Functions accept
``astropy.units.Quantity`` arguments and getters return Quantities, while
``UVPSpec`` attributes are plain arrays in their declared unit (see
:meth:`~hera_pspec.UVPSpec.get_param_unit`). Cosmological lengths are in
h\ :sup:`-1` Mpc (``u.Mpc / cu.littleh``, with
``astropy.cosmology.units`` imported as ``cu``) or in Mpc, chosen with the
``length_unit`` argument. See :doc:`migrating_to_v1` for how this differs
from v0.x.

Cosmologies are ``astropy.cosmology`` objects. The default is
:data:`hera_pspec.conversions.DEFAULT_COSMOLOGY`.

.. code-block:: python

  import astropy.cosmology.units as cu
  import astropy.units as u

  from hera_pspec import conversions

  cosmo = conversions.DEFAULT_COSMOLOGY
  z = conversions.f2z(150 * u.MHz)
  conversions.X2Y(z, cosmo, u.Mpc / cu.littleh)  # Mpc3 / (Hz sr littleh3)
  conversions.X2Y(z, cosmo, u.Mpc)               # Mpc3 / (Hz sr)

Units of a ``UVPSpec``
----------------------

* :attr:`~hera_pspec.UVPSpec.units`, :attr:`~hera_pspec.UVPSpec.norm_unit`
  and :attr:`~hera_pspec.UVPSpec.vis_unit` give the power spectrum unit and
  its two factors as astropy units. ``norm_units`` and ``vis_units`` hold the
  same as strings.
* :meth:`~hera_pspec.UVPSpec.get_param_unit` gives the declared unit of an
  attribute, e.g. ``Unit("s")`` for ``dly_array``.
* :meth:`~hera_pspec.UVPSpec.window_k_unit` and
  :meth:`~hera_pspec.UVPSpec.get_window_function_kgrid` give the k unit and
  grid of the exact window functions.
* :func:`hera_pspec.uvpspec_utils.parse_pspec_unit` parses ``norm_units`` and
  ``vis_units`` strings, including the legacy strings of v0.x files.

The ``conversions`` module
--------------------------

.. autodata:: hera_pspec.conversions.DEFAULT_COSMOLOGY
  :no-value:

.. autodata:: hera_pspec.conversions.F21
  :no-value:

.. autofunction:: hera_pspec.conversions.f2z

.. autofunction:: hera_pspec.conversions.z2f

.. autofunction:: hera_pspec.conversions.dRperp_dtheta

.. autofunction:: hera_pspec.conversions.dRpara_df

.. autofunction:: hera_pspec.conversions.X2Y

.. autofunction:: hera_pspec.conversions.bl_to_kperp

.. autofunction:: hera_pspec.conversions.tau_to_kpara

.. autofunction:: hera_pspec.conversions.cosmology_from_params

.. autofunction:: hera_pspec.conversions.params_from_cosmology

.. autofunction:: hera_pspec.conversions.as_astropy_cosmology

.. autofunction:: hera_pspec.conversions.resolve_length_unit

.. autofunction:: hera_pspec.conversions.length_unit_from_little_h

.. autoclass:: hera_pspec.conversions.Cosmo_Conversions
  :members:

.. autofunction:: hera_pspec.uvpspec_utils.parse_pspec_unit

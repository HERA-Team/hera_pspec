Migrating to v1.0
=================

hera_pspec 1.0 uses ``astropy`` for units and cosmology. Most v0.x code still
runs. Old arguments (``little_h=``, plain numbers for physical inputs,
``Cosmo_Conversions``) keep working but emit a ``DeprecationWarning`` that says
what to use instead. Python hides these warnings by default; to see them, run
with ``python -W default::DeprecationWarning`` or add
``warnings.simplefilter("default", DeprecationWarning)`` to your script.

Some return types changed outright, with no deprecation period. They are
listed first.

.. contents::
  :local:
  :depth: 1


Breaking changes
----------------

* Getters return ``astropy.units.Quantity`` objects:
  ``UVPSpec.get_dlys`` (s), ``get_kparas`` and ``get_kperps``
  (1 / length unit), ``get_blpair_seps`` (m) and ``get_spw_ranges`` (Hz;
  the returned tuples hold Quantities, so they are no longer hashable).
* ``PSpecData.delays()`` returns seconds as a Quantity. It used to return a
  float array in ns.
* ``PSpecData.scalar()`` and ``UVPSpec.compute_scalar()`` return a Quantity in
  ``length_unit**3 / (Hz sr)``.
* ``UVPSpec.generate_noise_spectra()`` returns Quantities: mK² L³ for
  ``form="Pk"``, mK² for ``form="DelSq"``.
* ``UVPSpec.units`` is an ``astropy.units.Unit``, no longer a string.
  ``norm_units`` and ``vis_units`` are canonical astropy strings such as
  ``"Mpc3 / littleh3"`` and ``"mK"``. ``PSpecData.units()`` returns these
  strings and raises ``ValueError`` for ``vis_units`` it does not know.
* Every ``cosmo`` attribute (``UVPSpec``, the beams, ``UVWindow``,
  ``noise.Sensitivity``) is an ``astropy.cosmology`` object. Methods of the old
  object, such as ``uvp.cosmo.get_params()`` or ``uvp.cosmo.X2Y(z)``, are gone;
  see `Cosmology`_.
* Power spectra normalized in ``Mpc3`` (``length_unit=u.Mpc``) now give k in
  1/Mpc by default: the getters, the ``spherical_average`` k grid, the plots
  and ``utils.uvp_noise_error`` all follow the object's own length unit. In
  v0.x they gave h/Mpc numbers for these objects.
* ``UVPSpec.set_cosmology`` no longer rescales an ``Mpc3`` object by about
  h³. That was a bug.
* ``convert_to_deltasq`` sets the new ``deltasq`` flag and leaves
  ``norm_units`` alone. Calling it twice raises ``ValueError``.
  ``combine_uvpspec`` refuses objects with different ``deltasq`` or
  ``normalization``.
* ``UVPSpec.set_stats`` converts a Quantity to ``uvp.units`` and raises
  ``ValueError`` if it cannot. It used to drop the unit silently.
* ``UVWindow.kunits`` is ``littleh / Mpc``. In v0.x it was ``units.h / units.Mpc``,
  which astropy reads as hour/Mpc. ``UVWindow.little_h`` is read-only, and
  ``UVWindow.check_kunits`` returns the converted Quantity.
* ``UVPSpec.get_param_unit`` returns ``None`` for ``window_function_kperp`` and
  ``window_function_kpara``; use ``UVPSpec.window_k_unit()`` for these.
* A cosmology with an explicit ``Om_k`` that differs from
  ``1 - Om_L - Om_M`` can no longer be used. Astropy derives the curvature
  from ``Om0`` and ``Ode0``, so it cannot represent such a model.
* Physical constants come from ``astropy.constants`` (CODATA 2018, was 2014).
  ``Jy_to_mK`` changes by -3.5e-7 relative, and spectra in mK² by about
  -7e-7.


Deprecations (removal planned for v2.0)
---------------------------------------

* ``little_h=`` everywhere. Use ``length_unit=`` (see `little_h is now
  length_unit`_). Affected: ``PSpecData.pspec``, ``PSpecData.scalar``,
  ``PSpecData.units``, ``UVPSpec.get_kperps``, ``UVPSpec.get_kparas``,
  ``UVPSpec.compute_scalar``, ``UVPSpec.generate_noise_spectra``,
  ``PSpecBeamBase.compute_pspec_scalar``, ``noise.Sensitivity.calc_scalar``,
  ``utils.uvp_noise_error``, ``grouping.spherical_average``,
  ``grouping.spherical_wf_from_uvp``, ``UVWindow(little_h=...)`` (also a
  positional bool in its third slot) and the ``UVWindow.little_h`` attribute.
* ``conversions.Cosmo_Conversions``: creating one, or passing one anywhere a
  cosmology is accepted.
* Plain-number k bins in ``spherical_average`` (``kbins``, ``kbins_theory``)
  and ``spherical_wf_from_uvp`` (``kbin_edges``, ``kbin_edges_theory``).
  ``UVWindow`` k bins in hour/Mpc are read as littleh/Mpc, with a warning.
* Plain numbers for physical inputs (see `Quantity inputs`_).
* ``FTBeam.from_file(spw_range=...)``: pass ``freq_array`` instead.
* ``pspec run --cosmo`` (a ``FutureWarning``, so the CLI shows it).


Units overview
--------------

Quantities appear at the edges of the API only. Functions accept Quantities
and getters return them, but the estimator works on plain floats in fixed
units (Hz, s, m, sr, Mpc), and ``UVPSpec`` attributes are plain arrays in
their declared unit. ``UVPSpec.get_param_unit`` tells you that unit.

The examples on this page share this setup:

.. code-block:: python

  import os

  import astropy.cosmology.units as cu
  import astropy.units as u
  import numpy as np
  from pyuvdata import UVData

  import hera_pspec as hp
  from hera_pspec.data import DATA_PATH

  beam = hp.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))

  uvd1, uvd2 = UVData(), UVData()
  uvd1.read(os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA"))
  uvd2.read(os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA"))
  for uvd in (uvd1, uvd2):
      # pyuvdata objects keep plain Hz; wrap them where hera_pspec needs them
      uvd.data_array *= beam.Jy_to_mK(np.unique(uvd.freq_array) * u.Hz, pol="XX")[
          None, :, None
      ]
      uvd.vis_units = "mK"

  ds = hp.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=beam)
  bls = [(37, 38), (38, 39), (52, 53)]
  uvp = ds.pspec(bls, bls, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)], verbose=False)

Getters give Quantities; the attributes behind them stay plain:

.. code-block:: python

  uvp.get_dlys(0)                   # Quantity in s
  uvp.get_kparas(0)                 # Quantity in littleh / Mpc
  uvp.get_blpair_seps()             # Quantity in m
  uvp.dly_array                     # plain floats ...
  uvp.get_param_unit("dly_array")   # ... in Unit("s")
  uvp.units                         # Unit("Mpc3 mK2 / littleh3")

To get the old plain numbers back, ask for the unit you want:

.. code-block:: python

  dlys_ns = uvp.get_dlys(0).to_value(u.ns)


``little_h`` is now ``length_unit``
-----------------------------------

The ``little_h`` flag is replaced by a ``length_unit`` argument:
``u.Mpc / cu.littleh`` (the default, h⁻¹ Mpc) or ``u.Mpc``. Strings such as
``"Mpc / littleh"`` work too. ``littleh`` is
``astropy.cosmology.units.littleh``.

.. code-block:: python

  # v0.x (still works, with a DeprecationWarning)
  k_para = uvp.get_kparas(0, little_h=True)
  uvp_mpc = ds.pspec(bls, bls, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)],
                     little_h=False, verbose=False)

.. code-block:: python

  # v1.0
  k_para = uvp.get_kparas(0, length_unit=u.Mpc / cu.littleh)
  uvp_mpc = ds.pspec(bls, bls, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)],
                     length_unit=u.Mpc, verbose=False)

``length_unit`` is accepted by ``PSpecData.pspec``, ``scalar`` and ``units``,
``pspecdata.pspec_run``, ``UVPSpec.get_kperps``, ``get_kparas``,
``compute_scalar`` and ``generate_noise_spectra``,
``PSpecBeamBase.compute_pspec_scalar``, ``noise.Sensitivity.calc_scalar``,
``utils.uvp_noise_error`` and ``pspec run --length-unit``.

To convert between littleh and Mpc yourself, use the object's Hubble constant.
Always pass ``H0`` to ``cu.with_H0``: without it, astropy uses Planck18's
H0 = 67.66, not the cosmology of your data.

.. code-block:: python

  k_mpc = uvp.get_kparas(0).to(1 / u.Mpc, cu.with_H0(uvp.cosmo.H0))

Getters default to the object's own length unit: littleh for objects
normalized in ``Mpc3 / littleh3`` (and for ``Hz sr`` objects), Mpc for objects
normalized in ``Mpc3``.

.. code-block:: python

  uvp_mpc.norm_units          # 'Mpc3'
  uvp_mpc.get_kparas(0).unit  # Unit("1 / Mpc"); v0.x gave h/Mpc numbers here


k bins and window functions
---------------------------

``grouping.spherical_average`` and ``grouping.spherical_wf_from_uvp`` take k
bins as Quantities, in littleh/Mpc or 1/Mpc. They are converted to the
object's k unit with its own cosmology. Plain numbers still work, read as
littleh/Mpc, but are deprecated.

.. code-block:: python

  # v0.x
  sph = hp.grouping.spherical_average(uvp, np.arange(0.1, 2.0, 0.2), bin_widths=0.2,
                                      little_h=True, time_avg=True)

.. code-block:: python

  # v1.0
  kbins = np.arange(0.1, 2.0, 0.2) * cu.littleh / u.Mpc
  sph = hp.grouping.spherical_average(uvp, kbins, bin_widths=0.2 * cu.littleh / u.Mpc,
                                      time_avg=True)

``UVWindow`` takes a ``kunits`` argument in place of ``little_h``. In v0.x the
tutorials built k bins with ``units.h / units.Mpc``. ``units.h`` is astropy's
hour, so those bins were in hour/Mpc. They still work, read as littleh/Mpc,
with a ``DeprecationWarning``. Use ``cu.littleh`` instead.

.. code-block:: python

  # v0.x
  from astropy import units
  uvw = hp.UVWindow(ftbeam, little_h=True)
  kbins = np.arange(0.1, 1.0, 0.05) * units.h / units.Mpc

.. code-block:: python

  # v1.0
  ftfile = os.path.join(DATA_PATH, "FT_beam_HERA_dipole_test_xx.hdf5")
  ftbeam = hp.FTBeam.from_file(ftfile, freq_array=hp.FTBeam.get_bandwidth(ftfile)[5:25] * u.Hz)
  uvw = hp.UVWindow(ftbeam, kunits=cu.littleh / u.Mpc)
  kbins = np.arange(0.1, 1.0, 0.05) * cu.littleh / u.Mpc
  sph_wf = uvw.get_spherical_wf(kbins, bl_lens=[14.6, 29.2] * u.m)

For exact window functions stored on a ``UVPSpec``, ``uvp.window_k_unit()``
gives the unit of ``window_function_kperp`` and ``window_function_kpara``
(littleh/Mpc if ``norm_units`` contain littleh, else 1/Mpc), and
``uvp.get_window_function_kgrid(spw)`` returns the grid as Quantities.


Cosmology
---------

Cosmologies are ``astropy.cosmology`` objects. Pass any of them (for example
``astropy.cosmology.Planck18``) wherever a ``cosmo`` argument is taken. The
default is ``hera_pspec.conversions.DEFAULT_COSMOLOGY``: the same model as the
old ``Cosmo_Conversions()``, a ``LambdaCDM`` with H0 = 67.27, Om0 = 0.31353,
Ode0 = 0.6844 and Tcmb0 = 0 (no radiation). It is slightly open
(Ok0 = 0.00207) and it is **not** Planck18.

.. code-block:: python

  # v0.x
  cosmo = hp.conversions.Cosmo_Conversions(H0=70.0)
  z = cosmo.f2z(150e6)
  x2y = cosmo.X2Y(z, little_h=True)
  dc = cosmo.DC(z, little_h=False)

.. code-block:: python

  # v1.0
  cosmo = hp.conversions.DEFAULT_COSMOLOGY.clone(H0=70.0)
  z = hp.conversions.f2z(150 * u.MHz)
  x2y = hp.conversions.X2Y(z, cosmo, u.Mpc / cu.littleh)   # Mpc3 / (Hz sr littleh3)
  dc = cosmo.comoving_distance(z)                           # Mpc

  from astropy.cosmology import Planck18

  beam_planck = hp.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"),
                               cosmo=Planck18)

The 21 cm helpers are module functions of ``hera_pspec.conversions`` that take
a cosmology and a length unit and return Quantities: ``f2z``, ``z2f``,
``dRperp_dtheta``, ``dRpara_df``, ``X2Y``, ``bl_to_kperp`` and
``tau_to_kpara``. For the distances use astropy directly:

=============================  =====================================================
``Cosmo_Conversions`` method   v1.0
=============================  =====================================================
``E(z)``                       ``cosmo.efunc(z)``
``DC(z)``                      ``cosmo.comoving_distance(z)``
``DM(z)``                      ``cosmo.comoving_transverse_distance(z)``
``DA(z)``                      ``cosmo.angular_diameter_distance(z)``
``get_params()``               ``hera_pspec.conversions.params_from_cosmology(cosmo)``
=============================  =====================================================

If you have the old parameter dictionary (for example from a v0.x file),
``conversions.cosmology_from_params`` builds the astropy object.
``conversions.as_astropy_cosmology`` also takes a ``Cosmo_Conversions``
object, with a ``DeprecationWarning``.

.. code-block:: python

  params = {"Om_L": 0.6844, "Om_b": 0.04911, "Om_c": 0.26442, "H0": 67.27}
  cosmo = hp.conversions.cosmology_from_params(params)

Changing the cosmology of a ``UVPSpec`` works as before:
``uvp.set_cosmology(new_cosmo, overwrite=True)``.


Power spectrum units
--------------------

``uvp.units``, ``uvp.norm_unit`` and ``uvp.vis_unit`` are astropy units. The
``norm_units`` and ``vis_units`` attributes are the matching strings.

.. code-block:: python

  uvp.norm_units    # 'Mpc3 / littleh3'  (v0.x: 'h^-3 Mpc^3')
  uvp.vis_units     # 'mK'
  uvp.units         # Unit("Mpc3 mK2 / littleh3")  (v0.x: the string '(mK)^2 h^-3 Mpc^3')

Don't hand legacy strings to ``u.Unit``: astropy reads the ``h`` in
``"h^-3 Mpc^3"`` as hour. ``uvputils.parse_pspec_unit`` maps them correctly.

.. code-block:: python

  hp.uvputils.parse_pspec_unit("h^-3 Mpc^3")   # Unit("Mpc3 / littleh3")

``convert_to_deltasq`` now sets ``deltasq`` and keeps ``norm_units``; ``units``
includes the k³ factor. It can be applied only once.

.. code-block:: python

  dsq = uvp.convert_to_deltasq(inplace=False)
  dsq.deltasq      # True
  dsq.norm_units   # 'Mpc3 / littleh3'
  dsq.units        # Unit("mK2")


Delays and the scalar
---------------------

.. code-block:: python

  # v0.x
  dlys_ns = ds.delays()             # floats in ns
  scalar = ds.scalar(("xx", "xx"))  # float

.. code-block:: python

  # v1.0
  dlys_ns = ds.delays().to_value(u.ns)
  scalar = ds.scalar(("xx", "xx"))  # Quantity in Mpc3 / (Hz sr littleh3)
  scalar.value                      # the v0.x number


Quantity inputs
---------------

Public functions that take a physical number accept a Quantity in any
equivalent unit. A plain number still works and is read in the unit below,
with a ``DeprecationWarning``. A Quantity in the wrong kind of unit raises
``UnitConversionError``.

.. list-table::
  :header-rows: 1

  * - Function
    - Argument
    - Plain numbers read as
  * - ``PSpecBeamGauss``
    - ``fwhm``, ``beam_freqs``
    - rad, Hz
  * - ``PSpecBeamFromArray``, ``PSpecBeamFromArray.add_pol``
    - ``OmegaP``, ``OmegaPP``, ``beam_freqs``
    - sr, Hz
  * - ``PSpecBeamBase.compute_pspec_scalar``
    - ``lower_freq``, ``upper_freq``
    - Hz
  * - ``PSpecBeamBase.Jy_to_mK``
    - ``freqs``
    - Hz
  * - ``PSpecBeamUV.beam_normalized_response``
    - ``freq``
    - Hz
  * - ``noise.calc_P_N``, ``noise.Sensitivity.calc_P_N``
    - ``Tsys``, ``t_int``, ``k``
    - K, s, 1 / length unit
  * - ``noise.Sensitivity.calc_scalar``
    - ``freqs``
    - Hz
  * - ``UVPSpec.generate_noise_spectra``
    - ``Tsys``
    - K
  * - ``utils.spw_range_from_freqs``
    - ``freq_range``
    - Hz
  * - ``FTBeam``, ``FTBeam.from_beam``, ``FTBeam.from_file``, ``FTBeam.select_freqs``
    - ``freq_array``, ``mapsize``, ``atol``
    - Hz, rad, Hz
  * - ``FTBeam.gaussian``
    - ``freq_array``, ``widths``, ``mapsize``
    - Hz, deg, rad
  * - ``UVWindow.get_kperp_bins``, ``get_cylindrical_wf``, ``get_spherical_wf``,
      ``cylindrical_to_spherical``, ``run_and_write``
    - ``bl_lens`` / ``bl_len``
    - m
  * - ``UVWindow.get_kpara_bins``
    - ``freq_array``
    - Hz

.. code-block:: python

  # v0.x
  beam_gauss = hp.PSpecBeamGauss(fwhm=0.1, beam_freqs=np.linspace(100e6, 200e6, 101))
  spws = hp.utils.spw_range_from_freqs(uvd1, freq_range=[(160e6, 165e6), (165e6, 170e6)])
  noise = uvp.generate_noise_spectra(0, ("xx", "xx"), 300)

.. code-block:: python

  # v1.0
  beam_gauss = hp.PSpecBeamGauss(fwhm=0.1 * u.rad, beam_freqs=np.linspace(100, 200, 101) * u.MHz)
  spws = hp.utils.spw_range_from_freqs(uvd1, freq_range=[(160, 165), (165, 170)] * u.MHz)
  noise = uvp.generate_noise_spectra(0, ("xx", "xx"), 300 * u.K)

``pspec_run`` still takes ``bl_len_range`` and ``bl_error_tol`` in metres and
``bl_deg_range`` in degrees as plain numbers. ``pyuvdata`` objects are not
touched: their ``freq_array`` stays in plain Hz.


One-step normalization and response matrices
--------------------------------------------

v1.0 adds a second way to normalize the power spectra. The default is
unchanged.

**Two-step** (the default, as in v0.x). The bandpowers are normalized by a
matrix M built from the dimensionless ``G`` or ``H`` matrices, and then
multiplied by a scalar that converts Hz sr to a cosmological volume. With
``exact_norm=True`` the object is labelled ``"two-step-exact"``.

**One-step** (`#329 <https://github.com/HERA-Team/hera_pspec/issues/329>`_).
M is computed from a response matrix that already includes the beam and the
cosmology, so normalization and unit conversion happen in one step. The
response is ``H_ab = tr[E_a dC/dp_b] / 2``, where ``dC/dp_b`` is the response
of the data covariance to bandpower b
(`#328 <https://github.com/HERA-Team/hera_pspec/issues/328>`_). It needs a
beam.

.. code-block:: python

  uvp_one = ds.pspec(bls, bls, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)],
                     normalization="one-step", verbose=False)
  uvp_one.normalization    # 'one-step'
  uvp_one.response_model   # "DelayResponse(sampling=False, beam='separable', ...)"

The built-in response is ``hera_pspec.DelayResponse``. Its options choose the
bin shape (``sampling``), the beam term (``beam="separable"`` for any beam,
``"healpix"`` for ``PSpecBeamUV``), how X2Y varies across the band (``x2y``)
and the bin width (``bin_width``):

.. code-block:: python

  uvp_hp = ds.pspec(bls, bls, (0, 1), [("xx", "xx")], spw_ranges=[(50, 90)],
                    normalization="one-step",
                    response=hp.DelayResponse(beam="healpix"), verbose=False)

You can also pass your own response: any object with a method
``matrices(freqs, delays, beam, pol, cosmo, length_unit)`` that returns the
``(Ndlys, Nfreqs, Nfreqs)`` array ``dC/dp_b`` as a Quantity in
``1 / length_unit**3`` (``Nfreqs`` counts the channels of the spectral window,
including any filter extensions). ``PSpecData.get_response(key1, key2, response=...)``
returns the resulting H as a float array. ``pspec_run`` and
``pspec run --normalization`` accept the same choice.

What to expect against two-step:

* With ``Ndlys = Nfreqs`` the two agree to a few tenths of a percent. On the
  shipped test data one-step is higher by +0.03% to +0.24%: two-step's scalar
  integrates over a continuous band where one-step sums over channels.
* With ``Ndlys < Nfreqs`` the difference is large (+64% with no taper and
  +3% to +4% with a Blackman-Harris taper at ``Ndlys = Nfreqs / 2``) because
  the two-step result is biased low there.
* With a taper and ``norm="H^-1"`` or ``"V^-1/2"``, two-step comes out too
  high by about 1/⟨T²⟩ (about 4 for Blackman-Harris), because ``scalar()``
  ignores its ``taper_override``. One-step does not have this problem.

``uvp.scalar_array`` means different things for each ``uvp.normalization``:

* ``"two-step"`` and ``"two-step-exact"``: the scalar that was multiplied into
  the power spectra.
* ``"one-step"``: a reference scalar, computed as for two-step but **not**
  applied. ``set_cosmology`` uses it to rescale data, covariances and errors
  when you change the cosmology; the result matches a fresh one-step run in
  the new cosmology to about 1e-5.

Two-step stays the default in v1.0.


Reading old files
-----------------

Files written by v0.x read without changes. Legacy ``norm_units`` strings
(``"h^-3 Mpc^3"``, ``"Hz str"``, ``"... k^3 / (2pi^2)"``) are mapped to the
new strings (and to ``deltasq``), and the cosmology dictionary becomes an
astropy object.

.. code-block:: python

  uvp_old = hp.UVPSpec()
  uvp_old.read_hdf5(os.path.join(DATA_PATH, "test_uvp.h5"))   # written in 2019
  uvp_old.norm_units   # 'Mpc3 / littleh3'
  uvp_old.cosmo        # LambdaCDM(name=None, H0=67.27 km / (Mpc s), Om0=0.31353, ...)

Files written by v1.0 carry a format version (``uvpspec_version = "2"``),
units on each dataset and the cosmology as astropy YAML (``cosmo_yaml``).
They also keep the old ``cosmo`` dictionary, so v0.x can still read them.


The ``pspec run`` command
-------------------------

``pspec run`` has new options for the cosmology, the length unit and the
normalization:

.. code-block:: console

  # a named astropy cosmology (or hera_pspec_default)
  pspec run even.uvh5 odd.uvh5 -o planck18.h5 --beam beam.beamfits --cosmology Planck18

  # parameters: --ode0 gives a LambdaCDM, without it the model is flat
  pspec run even.uvh5 odd.uvh5 -o h70.h5 --beam beam.beamfits --h0 70 --om0 0.3 --ode0 0.7

  # power spectra in Mpc^3 instead of h^-3 Mpc^3, normalized in one step
  pspec run even.uvh5 odd.uvh5 -o mpc.h5 --beam beam.beamfits \
    --length-unit Mpc --normalization one-step

The cosmology options only take effect together with ``--beam``.
``--cosmo`` still works but prints a deprecation warning; use ``--cosmology``
or ``--h0``/``--om0`` instead.

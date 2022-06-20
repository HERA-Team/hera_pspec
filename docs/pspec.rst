Power spectrum calculations
===========================

The ``PSpecData`` class takes a set of ``UVData`` objects containing visibilities and calculates delay power spectra from them. These are output as :class:`~hera_pspec.UVPSpec` objects.

.. contents:: Contents
  :local:


Example delay power spectrum calculation
----------------------------------------

The following example shows how to load ``UVData`` objects into a ``PSpecData`` object, specify a set of baselines and datasets that should be cross-multiplied, specify a set of spectral windows, weights, and tapers, and output a set of delay power spectra into a ``UVPSpec`` object.

.. code-block:: python

  # Load into UVData objects
  uvd1 = UVData(); uvd2 = UVData()
  uvd1.read_miriad(datafile1)
  uvd2.read_miriad(datafile2)

  # Create a new PSpecData object
  ds = hp.PSpecData(dsets=[uvd1, uvd2], beam=beam)

  # bls1 and bls2 are lists of tuples specifying baselines (antenna pairs)
  # Here, we specify three baseline-pairs, i.e. bls1[0] x bls2[0],
  # bls1[1] x bls2[1], and bls1[2] x bls2[2].
  bls1 = [(24,25), (37,38), (38,39)]
  bls2 = [(37,38), (38,39), (24,25)]

  # Calculate cross-spectra of visibilities for baselines bls1[i] x bls2[i],
  # where bls1 are the baselines to take from dataset 0 and bls2 are taken from
  # dataset 1 (and i goes from 0..2). This is done for two spectral windows
  # (freq. channel indexes between 300-400 and 600-721), with unity weights
  # and a Blackman-Harris taper in each spectral window
  uvp = ds.pspec(bls1, bls2, dsets=(0, 1), spw_ranges=[(300, 400), (600, 721)],
                 input_data_weight='identity', norm='I', taper='blackman-harris',
                 verbose=True)

``uvp`` is now a ``UVPSpec`` object containing 2 x 3 x Ntimes delay spectra, where
3 is the number of baseline-pairs (i.e. ``len(bls1) == len(bls2) == 3``), 2 is
the number of spectral windows, and Ntimes is the number of LST bins in the
input ``UVData`` objects. Each delay spectrum has length ``Nfreqs``, i.e. the
number of frequency channels in each spectral window.

To get power spectra from the ``UVPSpec`` object that was returned by ``pspec``:

.. code-block:: python

  # Key specifying desired spectral window, baseline-pair, and polarization pair
  spw = 1
  polpair = ('xx', 'xx')
  blpair =((24, 25), (37, 38))
  key = (spw, blpair, polpair)

  # Get delay bins and power spectrum
  dlys = uvp.get_dlys(spw)
  ps = uvp.get_data(key)


``PSpecData``: Calculate optimal quadratic estimates of delay power spectra
---------------------------------------------------------------------------

The ``PSpecData`` class implements an optimal quadratic estimator for delay power spectra. It takes as its inputs a set of ``UVData`` objects containing visibilities, plus objects containing supporting information such as weights/flags, frequency-frequency covariance matrices, and :any:`pspecbeam`.

Once data have been loaded into a ``PSpecData`` object, the :meth:`~hera_pspec.PSpecData.pspec` method can be used to calculate delay spectra for any combination of datasets, baselines, spectral windows etc. that you specify. Some parts of the calculation (e.g. empirical covariance matrix estimation) are cached within the ``PSpecData`` object to help speed up subsequent calls to :meth:`~hera_pspec.PSpecData.pspec`.

.. note::

  The input datasets should have compatible shapes, i.e. contain the same number of frequency channels and LST bins. The :meth:`~hera_pspec.PSpecData.validate_datasets` method (automatically called by :meth:`~hera_pspec.PSpecData.pspec`) checks for compatibility, and will raise warnings or exceptions if problems are found. You can use the :meth:`pyuvdata.UVData.select` method to select out compatible chunks of ``UVData`` files if needed.

Specifying which spectra to calculate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each call to :meth:`~hera_pspec.PSpecData.pspec` must specify a set of baseline-pairs, a set of datasets, and a set of spectral windows that the power spectrum should be estimated for.

 * **Datasets** correspond to the ``UVData`` objects that were stored inside the ``PSpecData`` object, and are identified either by an index (numbered according to the order that they were added to the ``PSpecData`` object), or label strings (if you specified labels when you added the datasets). A pair of datasets is then specified using the ``dsets`` argument, e.g. ``dsets=(0, 1)`` corresponds to the first and second datasets added to the ``PSpecData`` object. You can specify the same dataset if you want, e.g. ``dsets=(1, 1)``, although you should beware of noise bias in this case.

 * **Baseline pairs** are specified as two lists: ``bls1`` is the list of baselines from the first dataset in the pair specified by the ``dsets`` argument, and ``bls2`` is the list from the second. The baseline pairs are formed by matching each element from the first list with the corresponding element from the second, e.g. ``blpair[i] = bls1[i] x bls2[i]``. A couple of helper functions are provided to construct appropriately paired lists to calculate all of the cross-spectra within a redundant baseline group, for example; see :func:`~hera_pspec.pspecdata.construct_blpairs` and :func:`~hera_pspec.pspecdata.validate_bls`.

 * **Spectral windows** are specified as a list of tuples using the ``spw_ranges`` argument, with each tuple specifying the beginning and end frequency channel of the spectral window. For example, ``spw_ranges=[(220, 320)]`` defines a spectral window from channel 220 to 320, as indexed by the ``UVData`` objects. The larger the spectral window, the longer it will take to calculate the power spectra. Note that

 * **Polarizations** are looped over by default. At the moment, ``pspec()`` will only compute power spectra for matching polarizations from datasets 1 and 2. If the ``UVData`` objects stored inside the ``PSpecData`` object have incompatible polarizations, :meth:`~hera_pspec.PSpecData.validate_datasets` will raise an exception.

.. note::

  If the input datasets are phased slightly differently (e.g. due to offsets in LST bins), you can rephase (fringe-stop) them to help reduce decoherence by using the :meth:`~hera_pspec.PSpecData.rephase_to_dset` method. Note that the :meth:`~hera_pspec.PSpecData.validate_datasets` method automatically checks for discrepancies in how the ``UVData`` objects are phased, and will raise warnings or errors if any problems are found.

The ``PSpecData`` class
^^^^^^^^^^^^^^^^^^^^^^^

The most frequently-used methods from ``PSpecData`` are listed below. See :any:`pspecdata` for a full listing of all methods provided by ``PSpecData``.

.. autoclass:: hera_pspec.PSpecData
  :members: __init__, add, pspec, rephase_to_dset, scalar, delays, units
  :noindex:


``UVPSpec``: Container for power spectra
----------------------------------------

The :meth:`~hera_pspec.PSpecData.pspec` method outputs power spectra as a single ``UVPSpec`` object, which also contains metadata and various methods for accessing the data, input/output etc.

To access the power spectra, use the :meth:`~hera_pspec.UVPSpec.get_data` method, which takes a key of the form: ``(spw, blpair, polpair)``. For example:

.. code-block:: python

  # Key specifying desired spectral window, baseline-pair, and polarization
  spw = 1
  polpair = ('xx', 'xx')
  blpair =((24, 25), (37, 38))
  key = (spw, blpair, polpair)

  # Get delay bins and power spectrum
  dlys = uvp.get_dlys(spw)
  ps = uvp.get_data(key)

A number of methods are provided for returning useful metadata:

 * :meth:`~hera_pspec.UVPSpec.get_integrations`: Get the average integration time (in seconds) for a given delay spectrum.

 * :meth:`~hera_pspec.UVPSpec.get_nsamples`: If the power spectra have been incoherently averaged (i.e. averaged after squaring), this is the effective number of samples in that average.

 * :meth:`~hera_pspec.UVPSpec.get_dlys`: Get the delay for each bin of the delay power spectra (in seconds).

 * :meth:`~hera_pspec.UVPSpec.get_blpair_seps`: Get the average baseline separation for a baseline pair, in the ENU frame, in meters.

Dimensions and indexing of the ``UVPSpec`` data array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The power spectra are stored internally in ``UVPSpec.data_array``, which is a list of three-dimensional numpy arrays, one for each spectral window. Spectral window indices can be retrieved using the :meth:`~hera_pspec.UVPSpec.spw_to_indices` method. Each 3D array has shape ``(Nblpairts, Ndlys, Npols)``.

 * ``Npols`` is the number of polarizations. Available polarizations can be retrieved from the ``UVPSpec.pol_array`` attribute. This dimension can be indexed using the :meth:`~hera_pspec.UVPSpec.pol_to_indices` method.

 * ``Ndlys`` is the number of delays, which is equal to the number of frequency channels within the spectral window. The available frequencies/delays can be retrievd from the ``UVPSpec.freq_array`` and ``UVPSpec.dly_array`` attributes. Alternatively, use the :meth:`~hera_pspec.UVPSpec.get_dlys` method to get the delay values.

 * ``Nblpairts`` is the number of unique combinations of baseline-pairs and times (or equivalently LSTs), i.e. the total number of delay spectra that were calculated for a given polarization and spectral window. Baseline-pairs and times have been collapsed into a single dimension because each baseline-pair can have a different number of time samples.

   You can access slices of the baseline-pair/time dimension using the :meth:`~hera_pspec.UVPSpec.blpair_to_indices` and :meth:`~hera_pspec.UVPSpec.time_to_indices` methods. The baseline-pairs and times contained in the object can be retrieved from the ``UVPSpec.blpair_array`` and ``UVPSpec.time_avg_array`` (or ``UVPSpec.lst_avg_array``) attributes.

.. note::

  The ``UVPSpec.time_avg_array`` attribute is just the average of the times of the input datasets. To access the original times from each dataset, see the ``UVPSpec.time_1_array`` and ``UVPSpec.time_2_array`` attributes (or equivalently ``UVPSpec.lst_1_array`` and ``UVPSpec.lst_2_array``).


Averaging and folding spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, separate delay spectra are produced for every LST bin and polarization available in the input datasets, and for every baseline-pair passed to :meth:`~hera_pspec.PSpecData.pspec`. To (incoherently) average these down into a single delay spectrum, use the :meth:`~hera_pspec.UVPSpec.average_spectra` method. For example, to average over all times and baseline-pairs (i.e. assuming that the ``UVPSpec`` contains spectra for a single redundant baseline group):

.. code-block:: python

  # Build a list of all baseline-pairs in the UVPSpec object
  blpair_group = [sorted(np.unique(uvp.blpair_array))]

  # Average over all baseline pairs and times, and return in a new ``UVPSpec`` object
  uvp_avg = uvp.average_spectra(blpair_groups=blpair_group, time_avg=True, inplace=False)

For ``UVPSpec`` objects containing power spectra from more than one redundant baseline group, use the :meth:`~hera_pspec:UVPSpec.get_blpair_groups_from_bl_groups` method to extract certain groups.

Another useful method is :meth:`~hera_pspec:UVPSpec.fold_spectra`, which averages together :math:`\pm k_\parallel` modes into a single delay spectrum as a function of :math:`|k_\parallel|`.


The ``UVPSpec`` class
^^^^^^^^^^^^^^^^^^^^^

The most relevant methods from ``UVPSpec`` are listed below. See :any:`uvpspec` for a full listing of all methods provided by ``UVPSpec``.

.. autoclass:: hera_pspec.UVPSpec
  :members: __init__, get_data, get_wgts, get_integrations, get_nsamples, get_dlys, get_kperps, get_kparas, get_blpair_seps, select, read_hdf5, write_hdf5, generate_noise_spectra, average_spectra, fold_spectra, get_blpair_groups_from_bl_groups
  :noindex:

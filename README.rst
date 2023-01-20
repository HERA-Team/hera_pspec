**********************************************
``hera_pspec``: HERA delay spectrum estimation
**********************************************

|Run Tests| |codecov| |Documentation|


The ``hera_pspec`` library provides all of the tools and data structures
needed to perform a delay spectrum analysis on interferometric data. The
input data can be in any format supported by ``pyuvdata``, and the
output data are stored in HDF5 containers.

For usage examples and documentation, see
http://hera-pspec.readthedocs.io/en/latest/.

.. inclusion-marker-installation-do-not-remove

Installation
============

For users
---------

The package is installable, along with its dependencies, with PyPi. We
recommend using Anaconda and creating a new conda environment before
installing the package:

::

   $ conda create -n hera_pspec python=3
   $ conda activate hera_pspec
   $ python3 -m pip install hera_pspec

New versions are frequently released on PyPi.

For developers
--------------

If you are developping and/or want to use the latest working version
of ``hera_pspec``, you can directly install from the GitHub repository.

Preferred method of installation for users is simply ``pip install .``
(or ``pip install git+https://github.com/HERA-Team/hera_pspec``). This
will install required dependencies. See below for manual dependency
management.

Dependencies
^^^^^^^^^^^^

If you are using ``conda``, you may wish to install the following
dependencies manually to avoid them being installed automatically by
``pip``:

::

   $ conda install -c conda-forge "numpy>=1.15" "astropy>=2.0" h5py pyuvdata scipy matplotlib pyyaml

Developing
^^^^^^^^^^

If you are developing ``hera_pspec``, it is preferred that you do so in
a fresh ``conda`` environment. The following commands will install all
relevant development packages:

::

   $ git clone https://github.com/HERA-Team/hera_pspec.git
   $ cd hera_pspec
   $ conda create -n hera_pspec python=3
   $ conda activate hera_pspec
   $ conda env update -n hera_pspec -f ci/hera_pspec_tests.yml
   $ pip install -e . 

This will install extra dependencies required for testing/development as
well as the standard ones.

Running Tests
^^^^^^^^^^^^^

Uses the ``pytest`` package to execute test suite. From the source
``hera_pspec`` directory run: ``pytest``.

.. exclusion-marker-installation-do-not-remove

Running ``hera_pspec``
======================

See the documentation for an `overview and
examples <http://hera-pspec.readthedocs.io/en/latest/pspec.html>`__ of
how to run ``hera_pspec``. There are also some example Jupyter
notebooks, including
``examples/PS_estimation_examples.ipynb``
(a brief tutorial on how to create delay spectra), and
``examples/PSpecBeam_tutorial.ipynb`` 
(a brief tutorial on handling beam objects).

.. |Run Tests| image:: https://github.com/HERA-Team/hera_pspec/workflows/Run%20Tests/badge.svg
.. |codecov| image:: https://codecov.io/gh/HERA-Team/hera_pspec/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/HERA-Team/hera_pspec
.. |Documentation| image:: https://readthedocs.org/projects/hera-pspec/badge/?version=latest
   :target: https://readthedocs.org/projects/hera-pspec/badge/?version=latest

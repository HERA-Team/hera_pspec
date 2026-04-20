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

The package is installable from PyPI. ``hera_pspec`` currently supports
Python 3.11, 3.12, and 3.13 (``>=3.11,<3.14``). We recommend installing it
into a fresh virtual environment:

::

   $ python -m venv hera_pspec
   $ source hera_pspec/bin/activate
   $ python -m pip install hera_pspec

New versions are frequently released on PyPi.

For developers
--------------

If you are developing and/or want to use the latest working version
of ``hera_pspec``, you can directly install from the GitHub repository.

Developing
^^^^^^^^^^

The repository already includes a ``uv.lock`` file and dependency groups
for development, documentation, and tests. With ``uv`` installed, the
recommended setup is:

::

   $ git clone https://github.com/HERA-Team/hera_pspec.git
   $ cd hera_pspec
   $ uv sync --all-extras --dev

This installs the package along with the dependencies used in CI for
testing and documentation work.

Running Tests
^^^^^^^^^^^^^

From the source ``hera_pspec`` directory, common development commands are:

::

   $ uv run pytest
   $ uv run pytest -Werror

The repository pytest configuration already includes the standard coverage
options used in CI, so ``uv run pytest`` will also produce coverage output.

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

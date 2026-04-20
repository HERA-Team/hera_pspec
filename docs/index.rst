**************
``hera_pspec``
**************

**The HERA delay spectrum estimation package**

The ``hera_pspec`` library provides all of the tools and data structures needed 
to perform a delay spectrum analysis on interferometric data. The input data 
can be in any format supported by ``pyuvdata``, and the output data are stored in 
HDF5 containers.

You can find the code in the ``hera_pspec`` `GitHub repository <https://github.com/HERA-Team/hera_pspec>`_. It is also available on `PyPi <https://pypi.org/project/hera-pspec/>`_.
A set of `example Jupyter notebooks <https://github.com/HERA-Team/hera_pspec/tree/master/examples>`_ are also available on GitHub.


Installation
------------

The package is installable from PyPI and supports Python 3.11, 3.12, and
3.13.

::

   $ python -m venv hera_pspec
   $ source hera_pspec/bin/activate
   $ python -m pip install hera_pspec

New versions are frequently released on PyPi.
For more installation options, see below.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation   
   pspec
   pspecbeam
   pspecdata
   uvpspec
   container
   plot


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

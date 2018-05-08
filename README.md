# ``hera_pspec``: HERA delay spectrum estimation

[![Build Status](https://travis-ci.org/HERA-Team/hera_pspec.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_pspec)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_pspec/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_pspec?branch=master)
[![Documentation](https://readthedocs.org/projects/hera-pspec/badge/?version=latest)](https://readthedocs.org/projects/hera-pspec/badge/?version=latest)

The ``hera_pspec`` library provides all of the tools and data structures needed to perform a delay spectrum analysis on interferometric data. The input data can be in any format supported by ``pyuvdata``, and the output data are stored in HDF5 containers.

For usage examples and documentation, see http://hera-pspec.readthedocs.io/en/latest/.

## Installation

### Code Dependencies

* numpy >= 1.10
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)
* aipy (```conda install -c conda-forge aipy```)
* scipy >= 0.19
* astropy >= 2.0
* hera_cal (https://github.com/HERA-Team/hera_cal.git)
* pyyaml
* hdf5

For anaconda users, we suggest using conda to install astropy, numpy and scipy.

### Installing hera_pspec
Clone the repo using
`git clone https://github.com/HERA-Team/hera_pspec.git`

Navigate into the directory and run `python setup.py install`.

## Running `hera_pspec`

See the documentation for an [overview and examples](http://hera-pspec.readthedocs.io/en/latest/pspec.html) of how to run `hera_pspec`. There are also some example Jupyter notebooks, including [`examples/PS_estimation_examples.ipynb`](examples/PS_estimation_example.ipynb) (a brief tutorial on how to create delay spectra), and [`examples/PSpecBeam_tutorial.ipynb`](examples/PSpecBeam_tutorial.ipynb) (a brief tutorial on handling beam objects).

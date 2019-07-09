# ``hera_pspec``: HERA delay spectrum estimation

[![Build Status](https://travis-ci.org/HERA-Team/hera_pspec.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_pspec)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_pspec/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_pspec?branch=master)
[![Documentation](https://readthedocs.org/projects/hera-pspec/badge/?version=latest)](https://readthedocs.org/projects/hera-pspec/badge/?version=latest)

The ``hera_pspec`` library provides all of the tools and data structures needed to perform a delay spectrum analysis on interferometric data. The input data can be in any format supported by ``pyuvdata``, and the output data are stored in HDF5 containers.

For usage examples and documentation, see http://hera-pspec.readthedocs.io/en/latest/.

## Installation
Preferred method of installation for users is simply `pip install .`
(or `pip install git+https://github.com/HERA-Team/hera_pspec`). This will install 
required dependencies. See below for manual dependency management.
 
### Dependencies
If you are using `conda`, you may wish to install the following dependencies manually
to avoid them being installed automatically by `pip`::

    $ conda install -c conda-forge "numpy>=1.15" "astropy>=2.0" "aipy>=3.0rc2" h5py pyuvdata scipy matplotlib pyyaml h5py scikit-learn
    
### Developing
If you are developing `hera_pspec`, it is preferred that you do so in a fresh `conda`
environment. The following commands will install all relevant development packages::

    $ git clone https://github.com/HERA-Team/hera_pspec.git
    $ cd hera_pspec
    $ conda create -n hera_pspec python=3
    $ conda activate hera_pspec
    $ conda env update -n hera_pspec -f environment.yml
    $ pip install -e . 

This will install extra dependencies required for testing/development as well as the 
standard ones.

### Running Tests
Uses the `pytest` package to execute test suite.
From the source `hera_qm` directory run: ```pytest``` or ```python -m pytest```.

## Installation


### Code Dependencies

* numpy >= 1.15
* pyuvdata (`pip install pyuvdata` or use https://github.com/HERA-Team/pyuvdata.git)
* aipy (```conda install -c conda-forge aipy```)
* scipy >= 0.19
* matplotlib >= 2.2
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

"""
Tests for pspec power spectrum object.
"""
import nose.tools as nt
import numpy as np
import pspec

def setup_package():
    """
    Construct simple power spectrum and window function data structures.
    """
    # Trivial binning and power spectrum data
    kbins = np.logspace(-2., 0., 20)
    pspec3d = np.ones((kbins.size-1, kbins.size-1, kbins.size-1))
    pspec2d = np.ones((kbins.size-1, kbins.size-1))
    pspec1d = np.ones((kbins.size-1,))


def test_units():
    """
    Make sure that valid units are specified for power spectrum data.
    """
    # Errors should be raised if full set of units is not specified
    nt.assert_raises(KeyError, pspec.pspec, 3)
    nt.assert_raises(KeyError, pspec.pspec, 3, 'Mpc')
    nt.assert_raises(KeyError, pspec.pspec, 3, 'Mpc', 'mK')
    nt.assert_raises(KeyError, pspec.pspec, 3, length_units='Mpc')
    nt.assert_raises(KeyError, pspec.pspec, 3, None, None, None)
    nt.assert_raises(KeyError, pspec.pspec, 3, None, 'mK', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, '', 'mK', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, 'Mpc', '', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, 'Mpc', 'mK', '')
    nt.assert_raises(KeyError, pspec.pspec, 3, ' ', 'mK', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, '  ', 'mK', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, ['Mpc',], 'mK', 'GHz')
    nt.assert_raises(KeyError, pspec.pspec, 3, ('Mpc',), 'mK', 'GHz')
    
    # Make sure valid instantiations of the pspec class work as expected
    nt.assert_true(isinstance(pspec.pspec(3, 'Mpc', 'mK', 'GHz'), pspec.pspec))
    nt.assert_true(isinstance(pspec.pspec(3, 'Mpc/h', 'mK', 'GHz'), pspec.pspec))
    nt.assert_true(isinstance(pspec.pspec(3, 'h^-1 Mpc', 'mK', 'GHz'), pspec.pspec))
    nt.assert_true(isinstance(pspec.pspec(3, 'Mpc', 'mK', 'ghz'), pspec.pspec))
    
    # Check that power spectrum units are correct
    units = pspec.pspec(3, 'Mpc', 'mK', 'GHz')._pspec.units
    nt.assert_true('(mK)^2' in units)
    nt.assert_true('(Mpc)^3' in units)
    
if __name__ == '__main__':
    test_units()

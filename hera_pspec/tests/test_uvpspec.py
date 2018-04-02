import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import UVPSpec

class Test_uvpspec(unittest.TestCase):

    def setUp(self):
        uvp = UVPSpec()

        Ntimes = 10
        Nfreqs = 50
        Ndlys = Nfreqs
        Nspws = 1
        Nspwdlys = Nspws * Nfreqs
        Nbls = 3
        Nblpairs = 3
        Nblpairts = Nblpairs * Ntimes

        # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
        blpairs = [1002001002, 2003002003, 1003001003]

        blpair_array = np.tile(blpairs, Ntimes)
        time_array = np.repeat(np.linspace(2458042.1, 2458042.2, Ntimes), Nblpairs)
        time_1_array = time_array
        time_2_array = time_array
        lst_array = np.repeat(np.ones(Ntimes, dtype=np.float), Nblpairs)
        lst_1_array = lst_array
        lst_2_array = lst_array
        spws = np.arange(Nspws)
        spw_array = np.tile(spws, Ndlys)
        freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), Nspws)
        dly_array = np.repeat(np.fft.fftfreq(Nfreqs, np.median(np.diff(freq_array))), Nspws)
        pol_array = np.array([-5])
        Npols = len(pol_array)

        data_array, flag_array, integration_array = {}, {}, {}
        for s in spws:
            data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) * blpair_array[:, None, None] / 1e9
            flag_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.bool)
            integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)

        params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nblpairs', 'Nblpairts', 'Npols', 'Ndlys',
                  'blpair_array', 'time_1_array', 'time_2_array', 'lst_1_array', 'lst_2_array',
                  'spw_array', 'dly_array', 'freq_array', 'pol_array', 'data_array', 'flag_array',
                  'integration_array']

        for p in params:
            setattr(uvp, p, locals()[p])

        self.uvp = uvp


    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_init(self):
        pass






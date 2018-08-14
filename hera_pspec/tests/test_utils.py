import unittest
import nose.tools as nt
import numpy as np
import os, sys, copy
from hera_pspec.data import DATA_PATH
from hera_pspec import utils, testing
from collections import OrderedDict as odict
from pyuvdata import UVData
from hera_cal import redcal


def test_cov():
    # load another data file
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA"))

    # test basic execution
    d1 = uvd.get_data(24, 25)
    w1 = (~uvd.get_flags(24, 25)).astype(np.float)
    cov = utils.cov(d1, w1)
    nt.assert_equal(cov.shape, (60, 60))
    nt.assert_equal(cov.dtype, np.complex)
    d2 = uvd.get_data(37, 38)
    w2 = (~uvd.get_flags(37, 38)).astype(np.float)
    cov = utils.cov(d1, w2, d2=d2, w2=w2)
    nt.assert_equal(cov.shape, (60, 60))
    nt.assert_equal(cov.dtype, np.complex)
    # test exception
    nt.assert_raises(TypeError, utils.cov, d1, w1*1j)
    nt.assert_raises(TypeError, utils.cov, d1, w1, d2=d2, w2=w2*1j)
    w1 *= -1.0
    nt.assert_raises(ValueError, utils.cov, d1, w1)

def test_load_config():
    """
    Check YAML config file handling.
    """
    fname = os.path.join(DATA_PATH, '_test_utils.yaml')
    cfg = utils.load_config(fname)
    
    # Check that expected keys exist
    assert('data' in cfg.keys())
    assert('pspec' in cfg.keys())
    
    # Check that boolean values are read in correctly
    assert(cfg['pspec']['overwrite'] == True)
    
    # Check that lists are read in as lists
    assert(len(cfg['data']['subdirs']) == 1)
    
    # Check that missing files cause an error
    nt.assert_raises(IOError, utils.load_config, "file_that_doesnt_exist")

    # Check 'None' and list of lists become Nones and list of tuples
    nt.assert_equal(cfg['data']['pairs'], [('xx', 'xx'), ('yy', 'yy')])
    nt.assert_equal(cfg['pspec']['taper'], 'none')
    nt.assert_equal(cfg['pspec']['groupname'], None)
    nt.assert_equal(cfg['pspec']['options']['bar'], [('foo', 'bar')])
    nt.assert_equal(cfg['pspec']['options']['foo'], None)


class Test_Utils(unittest.TestCase):

    def setUp(self):
        # Load data into UVData object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, 
                                          "zen.2458042.17772.xx.HH.uvXA"))
        
        # Create UVPSpec object
        self.uvp, cosmo = testing.build_vanilla_uvpspec()

    def tearDown(self):
        pass

    def runTest(self):
        pass
    
    def test_spw_range_from_freqs(self):
        """
        Test that spectral window ranges are correctly recovered from UVData and 
        UVPSpec files.
        """
        # Check that type errors and bounds errors are raised
        nt.assert_raises(AttributeError, utils.spw_range_from_freqs, np.arange(3), 
                         freq_range=(100e6, 110e6))
        for obj in [self.uvd, self.uvp]:
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(98e6, 110e6)) # lower bound
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(190e6, 202e6)) # upper bound
            nt.assert_raises(ValueError, utils.spw_range_from_freqs, obj, 
                             freq_range=(190e6, 180e6)) # wrong order
            
        # Check that valid frequency ranges are returned
        freq_list = [(100e6, 120e6), (120e6, 140e6), (140e6, 160e6)]
        spw1 = utils.spw_range_from_freqs(self.uvd, freq_range=(110e6, 130e6))
        spw2 = utils.spw_range_from_freqs(self.uvd, freq_range=freq_list)
        spw3 = utils.spw_range_from_freqs(self.uvd, freq_range=(98e6, 120e6), 
                                          bounds_error=False)
        spw4 = utils.spw_range_from_freqs(self.uvd, freq_range=(100e6, 120e6))
        
        # Make sure tuple vs. list arguments were handled correctly
        nt.ok_( isinstance(spw1, tuple) )
        nt.ok_( isinstance(spw2, list) )
        nt.ok_( len(spw2) == len(freq_list) )
        
        # Make sure that bounds_error=False works
        nt.ok_( spw3 == spw4 )
        
        # Make sure that this also works for UVPSpec objects
        spw5 = utils.spw_range_from_freqs(self.uvp, freq_range=(100e6, 104e6))
        nt.ok_( isinstance(spw5, tuple) )
        nt.ok_( spw5[0] is not None )
    
    def test_spw_range_from_redshifts(self):
        """
        Test that spectral window ranges are correctly recovered from UVData and 
        UVPSpec files (when redshift range is specified).
        """
        # Check that type errors and bounds errors are raised
        nt.assert_raises(AttributeError, utils.spw_range_from_redshifts, 
                         np.arange(3), z_range=(9.7, 12.1))
        for obj in [self.uvd, self.uvp]:
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(5., 8.)) # lower bound
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(10., 20.)) # upper bound
            nt.assert_raises(ValueError, utils.spw_range_from_redshifts, obj, 
                             z_range=(11., 10.)) # wrong order
            
        # Check that valid frequency ranges are returned
        z_list = [(6.5, 7.5), (7.5, 8.5), (8.5, 9.5)]
        spw1 = utils.spw_range_from_redshifts(self.uvd, z_range=(7., 8.))
        spw2 = utils.spw_range_from_redshifts(self.uvd, z_range=z_list)
        spw3 = utils.spw_range_from_redshifts(self.uvd, z_range=(12., 14.), 
                                              bounds_error=False)
        spw4 = utils.spw_range_from_redshifts(self.uvd, z_range=(6.2, 7.2))
        
        # Make sure tuple vs. list arguments were handled correctly
        nt.ok_( isinstance(spw1, tuple) )
        nt.ok_( isinstance(spw2, list) )
        nt.ok_( len(spw2) == len(z_list) )
        
        # Make sure that this also works for UVPSpec objects
        spw5 = utils.spw_range_from_redshifts(self.uvp, z_range=(13.1, 13.2))
        nt.ok_( isinstance(spw5, tuple) )
        nt.ok_( spw5[0] is not None )
        

    def test_calc_blpair_reds(self):
        fname = os.path.join(DATA_PATH, 'zen.all.xx.LST.1.06964.uvA')
        uvd = UVData()
        uvd.read_miriad(fname)

        # basic execution
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd, uvd, filter_blpairs=True, exclude_auto_bls=False, exclude_permutations=True)  
        nt.assert_equal(len(bls1), len(bls2), 15)
        nt.assert_equal(blps, zip(bls1, bls2))
        nt.assert_equal(xants1, xants2)
        nt.assert_equal(len(xants1), 42)

        # test xant_flag_thresh
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd, uvd, filter_blpairs=True, exclude_auto_bls=True, exclude_permutations=True,
                                   xant_flag_thresh=0.0)  
        nt.assert_equal(len(bls1), len(bls2), 0)

        # test bl_len_range
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd, uvd, filter_blpairs=True, exclude_auto_bls=False, exclude_permutations=True,
                                   bl_len_range=(0, 15.0))  
        nt.assert_equal(len(bls1), len(bls2), 12)
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd, uvd, filter_blpairs=True, exclude_auto_bls=True, exclude_permutations=True,
                                   bl_len_range=(0, 15.0))  
        nt.assert_equal(len(bls1), len(bls2), 5)
        nt.assert_true(np.all([bls1[i] != bls2[i] for i in range(len(blps))]))

        # test grouping
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd, uvd, filter_blpairs=True, exclude_auto_bls=False, exclude_permutations=True,
                                   Nblps_per_group=2)  
        nt.assert_equal(len(blps), 10)
        nt.assert_true(isinstance(blps[0], list))
        nt.assert_equal(blps[0], [((24, 37), (25, 38)), ((24, 37), (24, 37))])

        # test baseline select on uvd
        uvd2 = copy.deepcopy(uvd)
        uvd2.select(bls=[(24, 25), (37, 38), (24, 39)])
        (bls1, bls2, blps, xants1,
         xants2) = utils.calc_blpair_reds(uvd2, uvd2, filter_blpairs=True, exclude_auto_bls=True, exclude_permutations=True,
                                   bl_len_range=(10.0, 20.0))  
        nt.assert_equal(blps, [((24, 25), (37, 38))])

        # test exceptions
        uvd2 = copy.deepcopy(uvd)
        uvd2.antenna_positions[0] += 2
        nt.assert_raises(AssertionError, utils.calc_blpair_reds, uvd, uvd2)

    def test_get_reds(self):
        fname = os.path.join(DATA_PATH, 'zen.all.xx.LST.1.06964.uvA')
        uvd = UVData()
        uvd.read_miriad(fname, read_data=False)
        antpos, ants = uvd.get_ENU_antpos()
        antpos_d = dict(zip(ants, antpos))

        # test basic execution
        xants = [0, 1, 2]
        r, l, a = utils.get_reds(fname, xants=xants)
        nt.assert_true(np.all([np.all([bl[0] not in xants and bl[1] not in xants for bl in _r]) for _r in r]))
        nt.assert_equal(len(r), len(a), len(l))
        nt.assert_equal(len(r), 104)

        r2, l2, a2 = utils.get_reds(uvd, xants=xants)
        _ = [np.testing.assert_array_equal(_r1, _r2) for _r1, _r2 in zip(r, r2)]

        r2, l2, a2 = utils.get_reds(antpos_d, xants=xants)
        _ = [np.testing.assert_array_equal(_r1, _r2) for _r1, _r2 in zip(r, r2)]

        # restrict
        bl_len_range = (14, 16)
        bl_deg_range = (55, 65)
        r, l, a = utils.get_reds(uvd, bl_len_range=bl_len_range, bl_deg_range=bl_deg_range)
        nt.assert_true(np.all([_l > bl_len_range[0] and _l < bl_len_range[1] for _l in l]))
        nt.assert_true(np.all([_a > bl_deg_range[0] and _a < bl_deg_range[1] for _a in a]))

        # autos
        r, l, a = utils.get_reds(fname, xants=xants, add_autos=True)
        nt.assert_almost_equal(l[0], 0)
        nt.assert_almost_equal(a[0], 0)
        nt.assert_true(len(r), 105)

    def test_config_pspec_blpairs(self):
        # test basic execution
        uv_template = os.path.join(DATA_PATH, "zen.{group}.{pol}.LST.1.28828.uvOCRSA")
        groupings = utils.config_pspec_blpairs(uv_template, [('xx', 'xx')], [('even', 'odd')], verbose=False, exclude_auto_bls=True)
        nt.assert_equal(len(groupings), 1)
        nt.assert_equal(groupings.keys()[0], (('even', 'odd'), ('xx', 'xx')))
        nt.assert_equal(len(groupings.values()[0]), 11833)

        # test multiple, some non-existant pairs
        groupings = utils.config_pspec_blpairs(uv_template, [('xx', 'xx'), ('yy', 'yy')], [('even', 'odd'), ('even', 'odd')], verbose=False, exclude_auto_bls=True)
        nt.assert_equal(len(groupings), 1)
        nt.assert_equal(groupings.keys()[0], (('even', 'odd'), ('xx', 'xx')))

        # test xants
        groupings = utils.config_pspec_blpairs(uv_template, [('xx', 'xx')], [('even', 'odd')], xants=[0, 1, 2], verbose=False, exclude_auto_bls=True)
        nt.assert_equal(len(groupings.values()[0]), 9735)

        # test exceptions
        nt.assert_raises(AssertionError, utils.config_pspec_blpairs, uv_template, [('xx', 'xx'), ('xx', 'xx')], [('even', 'odd')], verbose=False)
        

def test_log():
    """
    Test that log() prints output.
    """
    # print
    utils.log("message")
    utils.log("message", lvl=2)

    # logfile
    logf = open("logf.log", "w")
    utils.log("message", f=logf, verbose=False)
    logf.close()
    with open("logf.log", "r") as f:
        nt.assert_equal(f.readlines()[0], "message")

    # traceback
    logf = open("logf.log", "w")
    try:
        raise NameError
    except NameError:
        utils.log("raised an exception", f=logf, tb=sys.exc_info(), verbose=False)
    logf.close()
    with open("logf.log", "r") as f:
        log = ''.join(f.readlines())
        nt.assert_true("NameError" in log and "raised an exception" in log)
    os.remove("logf.log")


def test_hash():
    """
    Check that MD5 hashing works.
    """
    hsh = utils.hash(np.ones((8,16)))


def test_get_blvec_reds():
    fname = os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA")
    uvd = UVData()
    uvd.read_miriad(fname)
    antpos, ants = uvd.get_ENU_antpos(pick_data_ants=True)
    reds = redcal.get_pos_reds(dict(zip(ants, antpos)), low_hi=True)
    uvp = testing.uvpspec_from_data(fname, reds[:2], spw_ranges=[(10, 40)])

    # test execution w/ dictionary
    blvecs = dict(zip(uvp.bl_array, uvp.get_ENU_bl_vecs()))
    (red_bl_grp, red_bl_len, red_bl_ang,
     red_bl_tag) = utils.get_blvec_reds(blvecs, bl_error_tol=1.0)
    nt.assert_equal(len(red_bl_grp), 2)
    nt.assert_equal(red_bl_tag, ['015_060', '015_120'])

    # test w/ a UVPSpec
    (red_bl_grp, red_bl_len, red_bl_ang,
     red_bl_tag) = utils.get_blvec_reds(uvp, bl_error_tol=1.0)
    nt.assert_equal(len(red_bl_grp), 2)
    nt.assert_equal(red_bl_tag, ['015_060', '015_120'])

    # test w/ zero tolerance: each blpair is its own group
    (red_bl_grp, red_bl_len, red_bl_ang,
     red_bl_tag) = utils.get_blvec_reds(uvp, bl_error_tol=0.0)
    nt.assert_equal(len(red_bl_grp), uvp.Nblpairs)


def test_job_monitor():
    # open empty files
    datafiles = ["./{}".format(i) for i in ['a', 'b', 'c', 'd']]
    for df in datafiles:
        with open(df, 'w') as f:
            pass

    def run_func(i, datafiles=datafiles):
        # open file, perform action, finish
        # if rand_num is above 0.7, fail!
        try:
            rand_num = np.random.rand(1)[0]
            if rand_num > 0.7:
                raise ValueError
            df = datafiles[i]
            with open(df, 'a') as f:
                f.write("Hello World")
        except:
            return 1

        return 0

    # set seed
    np.random.seed(0)
    # run over datafiles
    failures = utils.job_monitor(run_func, range(len(datafiles)), "test", maxiter=1, verbose=False)
    # assert job 1 failed
    np.testing.assert_array_equal(failures, np.array([1]))
    # try with reruns
    np.random.seed(0)
    failures = utils.job_monitor(run_func, range(len(datafiles)), "test", maxiter=10, verbose=False)
    # assert no failures now
    nt.assert_equal(len(failures), 0)

    # remove files
    for df in datafiles:
        os.remove(df)


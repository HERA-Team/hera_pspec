"""
Testing file for scripts in pipelines/ directory
"""
import unittest
import nose.tools as nt
import numpy as np
import os, copy, sys
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData
import subprocess
import hera_pspec as hp
import yaml
import shutil
import fnmatch
import glob

class Test_PreProcess(unittest.TestCase):
    """ Testing class for preprocess_data.py pipeline """

    def setUp(self):
        pass

    def tearDown(self):
        files = ["preproc.yaml", "log.out", "err.out"]
        for f in files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    shutil.rmtree(f)

    def test_preprocess_run(self):
        # load default config
        cf = hp.utils.load_config(os.path.join(DATA_PATH, "../../pipelines/idr2_preprocessing/preprocess_params.yaml"))

        # edit parameters
        cf['analysis']['bl_reformat'] = False
        cf['analysis']['rfi_flag'] = False
        cf['analysis']['timeavg_sub'] = True
        cf['analysis']['time_avg'] = True
        cf['analysis']['form_pstokes'] = True
        cf['analysis']['fg_filt'] = True
        cf['analysis']['multiproc'] = False
        cf['analysis']['maxiter'] = 1
        cf['io']['joinlog'] = True
        cf['io']['verbose'] = False
        cf['io']['work_dir'] = "./"
        cf['io']['out_dir'] = "./"
        cf['io']['logfile'] = "log.out"
        cf['io']['errfile'] = 'err.out'
        cf['io']['overwrite'] = True
        cf['data']['data_template'] = os.path.join(DATA_PATH, cf['data']['data_template'])

        # write new config
        config = "./preproc.yaml"
        with open(config, "w") as f:
            f.write(yaml.dump(cf))

        # clean space
        exts = ['T', 'TF', 'TFP', 'TFD']
        pols = ['xx', 'yy']
        stokes = ['pI', 'pQ']
        files = []
        for e in exts:
            files += glob.glob("zen.all.??.*uvA{}".format(e))
        files += glob.glob("zen.all.??.tavg.uvA")
        for f in files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    shutil.rmtree(f)

        # run preprocess pipe
        out = subprocess.call(["preprocess_data.py", "preproc.yaml"])
        nt.assert_equal(out, 0)

        # check time average (xtalk) subtraction step
        for p in pols:
            # check files exist
            nt.assert_true(os.path.exists("zen.all.{}.tavg.uvA".format(p)))
            nt.assert_true(os.path.exists("zen.all.{}.LST.1.06964.uvAT".format(p)))
            uvd = UVData()
            uvd.read_miriad("zen.all.{}.tavg.uvA".format(p))
            nt.assert_equal(uvd.Ntimes, 1)
            nt.assert_equal(uvd.Nbls, 3)

        # check time averaging (FRF) & pstokes steps
        for p in pols + stokes:
            # check files exist: glob is used b/c starting LST changed slightly
            nt.assert_true(len(glob.glob("zen.all.{}.LST.1.0696?.uvATF".format(p)))==1)

        # check FG filtering step
        for p in pols + stokes:
            # check files exist: glob is used b/c starting LST changed slightly
            nt.assert_true(len(glob.glob("zen.all.{}.LST.1.0696?.uvATFD".format(p)))==1)
            nt.assert_true(len(glob.glob("zen.all.{}.LST.1.0696?.uvATFP".format(p)))==1)

        # clean space
        files = []
        for e in exts:
            files += glob.glob("zen.all.??.*uvA{}".format(e))
        files += glob.glob("zen.all.??.tavg.uvA")
        for f in files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    shutil.rmtree(f)
        os.remove("preproc.yaml")
        os.remove("log.out")



class Test_PspecPipe(unittest.TestCase):
    """ Testing class for pspec_pipe.py pipeline """

    def setUp(self):
        pass

    def tearDown(self):
        files = ["pspec.h5", "pspec.yaml", "log.out", "err.out"]
        for f in files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    shutil.rmtree(f)

    def test_pspec_run(self):
        # load default config
        cf = hp.utils.load_config(os.path.join(DATA_PATH, "../../pipelines/pspec_pipeline/pspec_pipe.yaml"))

        # edit parameters
        cf['analysis']['run_diff'] = False
        cf['analysis']['run_pspec'] = True
        cf['analysis']['run_bootstrap'] = True
        cf['analysis']['multiproc'] = False
        cf['data']['data_template'] = os.path.join(DATA_PATH, cf['data']['data_template'])
        cf['io']['joinlog'] = True
        cf['io']['verbose'] = False
        cf['io']['work_dir'] = "./"
        cf['io']['out_dir'] = "./"
        cf['io']['logfile'] = "log.out"
        cf['io']['errfile'] = 'err.out'
        cf['io']['overwrite'] = True
        cf['algorithm']['pspec']['beam'] = os.path.join(DATA_PATH, cf['algorithm']['pspec']['beam'])

        # write new config
        config = "./pspec.yaml"
        with open(config, "w") as f:
            f.write(yaml.dump(cf))

        # clean space
        if os.path.exists(cf['algorithm']['pspec']['outfname']):
            os.remove(cf['algorithm']['pspec']['outfname'])

        # run preprocess pipe
        out = subprocess.call(["pspec_pipe.py", "pspec.yaml"])
        nt.assert_equal(out, 0)

        # check pspec and bootstrap steps
        nt.assert_true(os.path.exists(cf['algorithm']['pspec']['outfname']))
        psc = hp.PSpecContainer(cf['algorithm']['pspec']['outfname'], 'r')
        groups = psc.groups()
        nt.assert_true(len(groups) > 0)
        for grp in groups:
            spectra = psc.spectra(grp)
            nt.assert_true(len(spectra), 2)
            nt.assert_true("_avg" not in spectra[0])
            nt.assert_true("_avg" in spectra[1])

        # clean space
        if os.path.exists(cf['algorithm']['pspec']['outfname']):
            os.remove(cf['algorithm']['pspec']['outfname'])
        os.remove("pspec.yaml")
        os.remove("log.out")


if __name__ == "__main__":
    unittest.main()

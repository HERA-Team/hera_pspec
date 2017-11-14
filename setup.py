from setuptools import setup
import glob
import os
import sys
import json

setup_args = {
    'name': 'hera_pspec',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_pspec',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'package_dir': {'hera_pspec': 'hera_pspec'},
    'packages': ['hera_pspec'],
    'include_package_data': True,
    'scripts': [],
    'zip_safe': False,
}


if __name__ == '__main__':
    apply(setup, (), setup_args)

from setuptools import setup
import glob
import os.path as path
from os import listdir
import sys
import os
from hera_pspec import version
import os.path as op
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('hera_pspec', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('hera_pspec', 'data')

setup_args = {
    'name': 'hera_pspec',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_pspec',
    'license': 'BSD',
    'description': 'HERA Power Spectrum Estimator Code.',
    'package_dir': {'hera_pspec': 'hera_pspec'},
    'packages': ['hera_pspec'],
    'include_package_data': True,
    'version': version.version,
    'package_data': {'hera_pspec': data_files},
    'zip_safe': False,
}

if __name__ == '__main__':
    apply(setup, (), setup_args)

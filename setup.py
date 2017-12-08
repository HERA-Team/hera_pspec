from setuptools import setup
import sys
import os
from hera_pspec import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_pspec', 'GIT_INFO'), 'w') as outfile:
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
    'name':         'hera_pspec',
    'author':       'HERA Team',
    'url':          'https://github.com/HERA-Team/hera_pspec',
    'license':      'BSD',
    'version':      version.version,
    'description':  'HERA Power Spectrum Estimator Code.',
    'packages':     ['hera_pspec'],
    'package_dir':  {'hera_pspec': 'hera_pspec'},
    'package_data': {'hera_pspec': data_files},
    'include_package_data': True,
    'zip_safe':     False,
}

if __name__ == '__main__':
    apply(setup, (), setup_args)

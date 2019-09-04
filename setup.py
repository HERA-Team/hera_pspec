import json
import os
import sys

from setuptools import setup

sys.path.append("hera_pspec")
import version

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_pspec', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile, default=str)


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


data_files = package_files('hera_pspec', 'data') + package_files('hera_pspec', '../pipelines')

setup_args = {
    'name': 'hera_pspec',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_pspec',
    'license': 'BSD',
    'version': version.version,
    'description': 'HERA Power Spectrum Estimator Code.',
    'packages': ['hera_pspec'],
    'package_dir': {'hera_pspec': 'hera_pspec'},
    'package_data': {'hera_pspec': data_files},
    'install_requires': [
        'numpy>=1.15',
        'scipy',
        'matplotlib>=2.2'
        'pyuvdata',
        'astropy>=2.0',
        'pyyaml',
        'h5py',
        'uvtools @ git+git://github.com/HERA-Team/uvtools',
        'hera_cal @ git+git://github.com/HERA-Team/hera_cal'
    ],
    'include_package_data': True,
    'scripts': ['scripts/pspec_run.py', 'scripts/pspec_red.py',
                'scripts/bootstrap_run.py'],
    'zip_safe': False,
}

if __name__ == '__main__':
    setup(**setup_args)

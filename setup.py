#!/usr/bin/env python3

import setuptools
import os

here = os.path.dirname(os.path.realpath(__file__))
requirementPath = here + '/requirements.txt'
install_requirements = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requirements = f.read().splitlines()


with open('README.md') as f:
    long_description = ''.join(f.readlines())


setuptools.setup(
    name='Granolar',
    version='1.0',
    packages=setuptools.find_packages(exclude=['tests']),
    include_package_data=True,

    description='Example of Python web app with debian packaging (dh_virtualenv & systemd)',
    long_description=long_description,
    author='Ninon Devis, Cyril Lavrat, Emanouil Plitsis, Alice Rixte, Lydia Rodriguez de la Nava',
    author_email='',
    url='https://github.com/fouVReaux/Granolar',

    # All versions are fixed just for case. Once in while try to check for new versions.
    install_requires=install_requirements,

    # Do not use test_require or build_require, because then it's not installed and
    # can be used only by setup.py. We want to use it manually as well.
    # Actually it could be in file like dev-requirements.txt but it's good to have
    # all dependencies close each other.
    extras_require={"dev" : []},

    entry_points={
        'console_scripts': [
            'webapp = webapp.cli:main',
        ],
    },

    classifiers=[
        'Framework :: Flask',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    zip_safe=False,
)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [
    'kubernetes>=9.0.0',
    'dill>=0.2.8.2',
    'deprecation>=2.0.6',
    'Click>=6.0',
    'tabulate>=0.8.5',
    'dill==0.2.8.2',
    'grpcio==1.26.0',
    'pyhelm==2.14.4',
    'appdirs==1.4.3',
    'google-api-python-client==1.7.11',
    'google-auth==1.5.1',
    'google-cloud==0.34.0',
    'google-cloud-container==0.3.0',
    'oauth2client==4.1.2',
    'torchtext==0.5.0',
    'spacy==2.2.3',
    'sklearn==0.0']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', 'pytest-mock', 'deprecation==2.0.6', 'freezegun==0.3.12' ]

setup(
    author="Ralf Grubenmann",
    author_email='ralf.grubenmann@epfl.ch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A public and reproducible collection of reference implementations and benchmark suite for distributed machine learning systems.",
    entry_points={
        'console_scripts': [
            'mlbench=mlbench_core.cli:cli',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mlbench',
    name='mlbench_core',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mlbench/mlbench_core',
    version='2.3.0-dev25',
    zip_safe=False,
)

#!/usr/bin/env python

from setuptools import setup
import setuptools_scm
import toml

setup(
    use_scm_version={'version_scheme': 'final-release',
                     'local_scheme': 'final-release'},
)

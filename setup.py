#!/usr/bin/env python

from setuptools import setup
import setuptools_scm
import toml

setuptools_scm.local_scheme = lambda v: ""

setup(
    use_scm_version=True,
)

#!/usr/bin/env python

from setuptools import setup
import setuptools_scm
import toml

def local_scheme(version):
    return ""

setup(
    use_scm_version={"local_scheme": local_scheme},
)

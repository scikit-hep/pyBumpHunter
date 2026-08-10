#!/usr/bin/env python

"""
Python implementation of the BumpHunter algorithm used by HEP community.

This package provide a BumpHunter class that allows to extract a global p-value
from a data distribution given a reference background distribution.
A 2D extension of the BumpHunter class have been added.
This BumpHunter class also allows to do a signal injection test for the given
background and signal distributions.
Note that BumpHunter2D does NOT support signal injection yet.

Basic usage :

To create a BumpHunter class instance (valid for BumpHunter2D) :
    import pyBumpHunter as BH
    BHtest = BH.BumpHunter1D(...)

To perform a scan using BumpHunter algorithm and compute a global p-value
and significance (valid for BumpHunter2D) :
    BHtest.bump_scan(data,bkg)

To get the results of the last scan performed and do some plots (valid for BumpHunter2D) :
    BHtest.bump_info(data)
    BHtest.plot_tomography(bkg)
    BHtest.plot_bump(data,bkg)
    BHtest.plot_stat()

To perform a signal injection test :
    BHtest.signal_inject(sig,bkg)

To plot the result of the last signal injection performed :
    BHtest.plot_inject()

For more details about the BumpHunter class, please refer to its docstring.

For more details on pyBumpHunter usage in general, please refer to the
pyBumpHunter wiki page :
https://github.com/lovaslin/pyBumpHunter/wiki
"""

from .bumphunter_1dim import BumpHunter1D
from .bumphunter_2dim import BumpHunter2D

# Automatic versioning
from .version import version as __version__

__all__ = ["BumpHunter1D", "BumpHunter2D", "__version__"]

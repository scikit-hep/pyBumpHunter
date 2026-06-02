#!/usr/bin/env python

"""
Python implementation of the BumpHenter algorithm used by HEP community.

This package provides a DataHandler class that manages histograms, as well as the
BumpHunter1D and BumpHunter2D classes that performs bump hunts in 1D and 2D histograms.

This packqge includes several extensions of the algorithm, such as side-band normalization,
signal injection test and multi-channel combination. It also features a automatic fit
procedure of the BumpHunter test statistic distribution.

The plot routines use the current matplotlib.pyplot.Figure instance. They can be used as
the plt.hist function.

Basic usage :

To create an DataHandler class instance :
    import pyBumpHunter as BH
    dh = BH.DataHandler(...)

To create an BumpHunter1D class instance (same for BumpHunter2D) :
    import pyBumpHunter as BH
    bh = BH.BumpHunter1D(...)

To perform a scan using BumpHnuter algorithm and compute a global p-value
and significance (same for BumpHunter2D) :
    bh.bump_scan(dh)

To print the results of the last scan performed and do some plots :
    print(bh.bump_info(dh))
    bh.plot_tomography(dh)  # Only for BumpHunter1D
    bh.plot_bump(dh)
    bh.plot_stat()

To perform a signal injection test :
    bh.singal_inject(dh)

To plot the result of the last signal injection performed :
    bh.plot_inject()

For more details about the BumpHunter class, please refer to its docstring.

For more details on pyBumpHunter usage in general, please refer to the
pyBumpHunter wiki page :
https://github.com/scikit-hep/pyBumpHunter/wiki
"""

from .data_handler import DataHandler
from .bumphunter_1dim import BumpHunter1D
from .bumphunter_2dim import BumpHunter2D
from .functions import bh_stat

# Automatic versioning
from .version import version as __version__

__all__ = ["BumpHunter1D", "BumpHunter2D", "DataHandler", "BHstat", "__version__"]

from .util import deprecated as _deprecated


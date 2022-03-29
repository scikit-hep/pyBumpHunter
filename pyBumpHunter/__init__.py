#!/usr/bin/env python

"""
Python implementation of the BumpHenter algorithm used by HEP community.

This package provide a BumpHunter class that allows to extract a global p-value
from a data distribution given a reference background distribution.
A 2D extension of the BumpHunter class have been added.
This BumpHunter class also allows to do a signal injection test for the given
background and signal distributions.
Note that BumpHunter2D does NOT support signal injection yet.

Basic usage :

To create an BumpHunter class instance (valid of BumpHunter2D) :
    import pyBumpHunter as BH
    BHtest = BH.BumpHunter(...)

To perform a scan using BumpHnuter algorithm and compute a global p-value
and significance (valid of BumpHunter2D) :
    BHtest.BumpScan(data,bkg)

To print the results of the last scan performed and do some plots (valid of BumpHunter2D) :
    BHtest.PrintBumpInfo()
    BHtest.PrintBumpTrue()
    BHtest.GetTomography()
    BHtest.PlotBump()
    BHtest.PlotBHstat()

To perform a signal injection test :
    BHtest.SignalInject(data,sig)

To plot the result of the last signal injection performed :
    BHtest.PlinInject()

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

from .util import deprecated as _deprecated


class BumpHunter(BumpHunter1D):
    @_deprecated("Use BumpHunter1D or BumpHunter2D instead of BumpHunter.")
    def __init__(
        self,
        rang=None,
        mode="excess",
        width_min=1,
        width_max=None,
        width_step=1,
        scan_step=1,
        npe=100,
        bins=60,
        weights=None,
        nworker=4,
        sigma_limit=5,
        str_min=0.5,
        str_step=0.25,
        str_scale="lin",
        signal_exp=None,
        flip_sig=True,
        seed=None,
        use_sideband=None,
        Npe=None,
        Nworker=None,
        useSideBand=None,
    ):
        super().__init__(
            rang,
            mode,
            width_min,
            width_max,
            width_step,
            scan_step,
            Npe,
            bins,
            weights,
            nworker,
            sigma_limit,
            str_min,
            str_step,
            str_scale,
            signal_exp,
            flip_sig,
            seed,
            use_sideband,
            Nworker,
            useSideBand,
        )

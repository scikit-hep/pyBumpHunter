#!/usr/bin/env python

'''
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
'''

from .BumpHunter import BumpHunter
from .BumpHunter2D import BumpHunter2D


# Automatic versioning
from .version import version as __version__



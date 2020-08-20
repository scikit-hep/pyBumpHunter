#!/usr/bin/env python

'''
Python implementation of the BumpHenter algorithm used by HEP community.

This package provide a BumpHunter class that allows to extract a global p-value
from a data distribution given a reference background distribution.
This BumpHunter class also allows to do a signal injection test for the given
background and signal distributions.

Basic usage :

To create an BumpHunter class instance :
    import pyBumpHunter as BH
    BHtest = BH.BumpHunter(...)
    
To perform a scan using BumpHnuter algorithm and compute a global p-value
and significance :
    BHtest.BumpScan(data,bkg)

To print the results of the last scan performed and do some plots :
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

# Automatic versioning
from .version import version as __version__



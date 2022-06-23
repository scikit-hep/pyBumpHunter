# pyBumpHunter

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scikit-hep/pyBumpHunter/master)
[![Test](https://github.com/scikit-hep/pyBumpHunter/workflows/automated_testing/badge.svg)](https://github.com/scikit-hep/pyBumpHunter/actions)
[![PyPI](https://img.shields.io/pypi/v/pyBumpHunter)](https://pypi.org/project/pyBumpHunter/)

This is a python version of the BumpHunter algorithm, see [arXiv:1101.0390, G. Choudalakis](https://arxiv.org/abs/1101.0390), designed to find localized excess (or deficit) of events in a 1D or 2D distribution.

The main BumpHunter function will scan a data distribution using variable-width window sizes and calculate the p-value of data with respect to a given background distribution in each window. The minimum p-value obtained from all windows is the local p-value. To cope with the "look-elsewhere effect" a global p-value is calculated by performing background-only pseudo-experiments.

The BumpHunter algorithm can also perform signal injection tests where more and more signal is injected in toy data until a given signal significance (global) is reached (*signal injection not available in 2D yet*).

### Content

* pyBumpHunter : The pyBumpHunter package
* example : Folder containing a set of example scripts and notebooks
* example/results : Folder containing the outputs of example scripts
* test : Folder containing the testing scripts (based on pytest)
* data/data.root  : Toy data used in the examples and tests
* data/gen_data.C : Code used to generate the toy data with ROOT

### Dependencies

Requires python >= 3.6 py

BumpHunter depends on the following python libraries :

* numpy
* scipy
* matplotlib
* h5py

### [pyBumpHunter wiki](https://github.com/scikit-hep/pyBumpHunter/wiki)

### Examples

The examples provided in example.py and test.ipynb require the [uproot](https://github.com/scikit-hep/uproot) package in
order to read the data from a [ROOT software](https://root.cern.ch/) file.

The data provided in the example consists of three histograms: a steeply falling 'background' distribution in a [0,20] x-axis range, a 'signal' gaussian shape centered on a value of 5.5, and a 'data' distribution sampled from background and signal distributions, with a signal fraction of 0.15%. The data file is produced by running gen_data.C in ROOT.

In order to run the example script, simply type `python3 example.py` in a terminal.

You can also open the example notebook with jupyter or binder.

* Bump hunting:

<p align="center">
<img src="./example/results/1D/bump.png" title="drawing"  width="500">
</p>

* Tomography scan:

<p align="center">
<img src="./example/results/1D/tomography.png" title="drawing"  width="500">
</p>

* Test statistics and global p-value:

<p align="center">
<img src="./example/results/1D/BH_statistics.png" title="drawing"  width="500">
</p>

See the [wiki](https://github.com/scikit-hep/pyBumpHunter/wiki) for a detailed overview of all the features offered by pyBumpHunter.

### To do list

* Run BH on 2D histograms

### Authors and contributors

Louis Vaslin (main developper), Julien Donini

Thanks to Samuel Calvet for his help in cross-checking and validating pyBumpHunter against the (internal) C++ version of BumpHunter developped by the [ATLAS collaboration](https://atlas.cern/).

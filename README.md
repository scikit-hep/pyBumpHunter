# pyBumpHunter

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lovaslin/pyBumpHunter/master)

This is a python version of the BumpHunter algorithm ([arXiv:1101.0390, G. Choudalakis](https://arxiv.org/abs/1101.0390)) designed to find localized excess, or deficit, of events in a 1D distribution.

The main BumpHunter function will scan a data distribution using variable-width window sizes and calculate the p-value of data with respect to a given background distribution in each window. The minimum p-value obtained from all windows is the local p-value. To cope with the "look-elsewhere effect" a global p-value is calculated by performing background-only pseudo-experiments.

The BumpHunter algorithm can perform signal injection tests where more and more signal is injected in toy data until a given signal significance (global) is reached.

Note: the present algorithm has been cross-checked using the (internal) C++ version developped by the [ATLAS collaboration](https://atlas.cern/). It as been made and tested to reproduce as accurately as possible the original version.

### Content

* pyBumpHunter.py : The pyBumpHunter package
* test.py : A little example script that use pyBumpHunter
* test.ipynb : A little example notebook that use pyBumpHunter
* results_py : Folder containing the outputs of test.py
* data.root  : The generated toy data
* gen_data.C : Code used to generate the data with ROOT

### python dependancies

pyBumpHunter depends on the following python libraries :

* numpy
* scipy
* matplotlib

### Examples

The example provided in test.py and test.ipy require [uproot](https://github.com/scikit-hep/uproot) package in order to read the data from a [ROOT software](https://root.cern.ch/) file.

The data provided in the example consists of three histograms: a steeply falling 'background' distribution in a [0,20] x-axis range, a 'signal' gaussian shape centered on a value of 5.5, and a 'data' distribution sampled from background and signal distributions, with a signal fraction of 0.15%.

In order to run the example script, simply type `python3 test.py` in a terminal.

You can also open the example notebook with jupyter or Binder.


![Global p-value](https://raw.githubusercontent.com/lovaslin/pyBumpHunter/master/results_py/BH_statistics.png)



<img src="https://raw.githubusercontent.com/lovaslin/pyBumpHunter/master/results_py/BH_statistics.png" alt="drawing" width="200"/>


### Authors and contributors

Louis Vaslin (main developper), Julien Donini and Samuel Calvet.

### Acknowledgments
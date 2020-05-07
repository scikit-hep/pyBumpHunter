# pyBumpHunter

Simple python version of the BumpHunter algorithm.

This work is based on the official version used by the [ATLAS collaboration](https://atlas.cern/).
It as been made and tested to reproduce as accurately as possible the original version.

### Content

* gen_data.C : Code used to generate the data with root
* data.root  : The generated data
* pyBumpHunter.py : The pyBumpHunter package
* test.py : A little example script that use pyBumpHunter
* test.ipynb : A little example notebook that use pyBumpHunter
* results_py : Folder containing the outputs of test.py

### python dependancies

pyBumpHunter depends on the following python libraries :

* numpy
* scipy
* matplotlib

test.py also use uproot in order to read the data from a root file.

The data have been generated using the [ROOT software](https://root.cern.ch/).

In order to run the example script, simply type `python3 test.py` in a terminal.

You can also open the example notebook with jupyter.

# Test script to be run using pytest
# Here, all the values required to pass te test has been validated using the
# orriginal c++ version of BumpHunter used by ATLAS collaboration.

import os

import uproot as upr
from pathlib import Path
import pytest

import pyBumpHunter as BH

here = os.path.dirname(os.path.realpath(__file__))

"""
# Generate a dataset with numpy
# Migth be use latter
def make_datasets(seed):
    np.random.seed(seed)
    bkg = np.random.exponential(2, size=100000)
    sig = np.random.normal(5.5, 0.35, size=5000)
    data = np.concatenate(
        [np.random.exponential(2, size=100000), np.random.normal(5.5, 0.35, size=150)]
    )
    np.random.shuffle(data)  # to make sure peak is not ordered

    all = [ar[ar < 20] for ar in (data, sig, bkg)]
    return all
"""


# Get the dataset from the ROOT file
# I prefer to keep it for now since the value I am testing are dataset specific
def make_datasets():
    # Get path to data
    path = Path("data/data.root")

    # Open the file
    with upr.open(path) as f:
        # Get the trees
        bkg = f["bkg"].arrays(library="np")["bkg"]
        data = f["data"].arrays(library="np")["data"]
        sig = f["sig"].arrays(library="np")["sig"]

        # Put the data into a DataHandler
        dh = BH.DataHandler(nchan=1, ndim=1)
        dh.set_ref(bkg, bins=[60], rang=[0, 20])
        dh.set_data(data)
        dh.set_sig(sig, signal_exp=150) # Correspond the the real number of signal events generated when making the data

    return dh


@pytest.fixture
def data_sig_bkg1():
    # return make_datasets(seed=534)
    return make_datasets()


@pytest.fixture
def bhunter():
    return BH.BumpHunter1D(
        width_min=2,
        width_max=6,
        width_step=1,
        scan_step=1,
        npe=10000,
        nworker=1,
        seed=666,
    )


# Test if the the bump_scan method runs
def test_scan_run(data_sig_bkg1, bhunter):
    # Get the data
    dh = data_sig_bkg1

    # Run the bump_scan method
    bhunter.bump_scan(dh)

    # Test if the position of the Bump is correct w.r.t. the expected value
    assert bhunter.min_loc_ar[0, 0] == 16  # 16th bin

    # Test if the width of the bump is correct w.r.t. the expected value
    assert bhunter.min_width_ar[0, 0] == 4  # 4 bins

    # Test if the local p-value is correct w.r.t. the expected value (up to 7 digit)
    assert f"{bhunter.min_Pval_ar[0, 0]:.7f}" == "0.0001734"

    # Test if the global p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.global_Pval[0]:.5f}" == "0.01770"

    # Test if the number of tested intervals is correct w.r.t. the expected value
    N = 0
    for r in bhunter.res_ar[0]:
        N += r.size
    assert N == 285

    # Test if the evaluated number of signal event is correct w.r.t. the expected value
    assert bhunter.signal_eval[0] == 208


# Test if the signal_inject method runs
def test_inject_run(bhunter, data_sig_bkg1):
    # Get the data
    dh = data_sig_bkg1

    # Set the injection parametters
    bhunter.sigma_limit = 5
    bhunter.str_min = -1  # if str_scale='log', the real starting value is 10**str_min
    bhunter.str_scale = "log"

    # Run the signal_inject method
    bhunter.signal_inject(dh)

    # Test if the final number of injected signal event is correct w.r.t. the expected value
    assert int(bhunter.signal_min[0]) == 450

    # Test if the final signal ratio is correct w.r.t. the expected value
    assert f"{bhunter.signal_ratio:.2f}" == "3.00"


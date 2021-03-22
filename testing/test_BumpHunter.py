# Test script to be run using pytest
# Here, all the values required to pass te test has been validated using the
# orriginal c++ version of BumpHunter used by ATLAS collaboration.

import os

import numpy as np
import pytest

import pyBumpHunter as BH

# Import the data

here = os.path.dirname(os.path.realpath(__file__))


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


@pytest.fixture
def data_sig_bkg1():
    return make_datasets(seed=534)


@pytest.fixture
def bhunter_noscan():
    return BH.BumpHunter1D(
        rang=[0, 20],
        width_min=2,
        width_max=6,
        width_step=1,
        scan_step=1,
        Npe=10000,
        nworker=1,
        seed=666,
    )


@pytest.fixture
def bhunter(bhunter_noscan, data_sig_bkg1):
    data, _, bkg = data_sig_bkg1
    bhunter_noscan.BumpScan(data, bkg)
    return bhunter_noscan


# Test if the the BumpScan method runs
def test_scan_run(data_sig_bkg1, bhunter_noscan):
    data, _, bkg = data_sig_bkg1
    bhunter_noscan.BumpScan(data, bkg)
    bhunter = bhunter_noscan

    # Test if the position of the Bump is correct w.r.t. the expected value
    assert bhunter.min_loc_ar[0] == 16  # 16th bin

    # Test if the width of the bump is correct w.r.t. the expected value
    assert bhunter.min_width_ar[0] == 4  # 4 bins

    # Test if the local p-value is correct w.r.t. the expected value (up to 7 digit)
    assert "{:.7f}".format(bhunter.min_Pval_ar[0]) == "0.0001734"

    # Test if the global p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.global_Pval:.5f}" == "0.01770"


# Test if the number of tested intervals is correct w.r.t. the expected value
def test_number_interval(bhunter):
    N = 0
    for res in bhunter.res_ar:
        for r in res:
            N += r.size
    assert N == 2811150


# Test if the evaluated number of signal event is correct w.r.t. the expected value
def test_number_signal(bhunter, data_sig_bkg1):
    data, _, bkg = data_sig_bkg1
    Hdata, _ = np.histogram(data, bins=bhunter.bins, range=bhunter.rang)
    Hbkg, _ = np.histogram(
        bkg, bins=bhunter.bins, range=bhunter.rang, weights=bhunter.weights
    )
    # TODO fix: this test never ran as it was named tesl_blabla...
    D = Hdata[min_loc_ar[0] : min_loc_ar[0] + min_width_ar[0]].sum()
    B = Hbkg[min_loc_ar[0] : min_loc_ar[0] + min_width_ar[0]].sum()
    assert int(D - B) == 208


# Test if the SignalInject method runs
def test_inject_run(bhunter, data_sig_bkg1):
    _, sig, bkg = data_sig_bkg1
    bhunter.sigma_limit = 5
    bhunter.str_min = -1  # if str_scale='log', the real starting value is 10**str_min
    bhunter.str_scale = "log"
    bhunter.signal_exp = 150  # Correspond the the real number of signal events generated when making the data

    bhunter.SignalInject(sig, bkg)

    assert int(bhunter.signal_min) == 300
    assert f"{bhunter.signal_ratio:.2f}" == "2.00"

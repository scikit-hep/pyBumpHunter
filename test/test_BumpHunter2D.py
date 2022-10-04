# Test script to be run using pytest
# Here, all the values required to pass te test has been validated using the
# orriginal c++ version of BumpHunter used by ATLAS collaboration.

import pyBumpHunter as BH
import pytest
import numpy as np

# Function to generate a dataset with numpy
def make_datasets(seed):
    np.random.seed(seed)

    # Background (with more stat to have a smoother reference)
    bkg = np.random.exponential(scale=[4, 4], size=(1_000_000, 2))

    # Data
    Nsig = 500
    data = np.empty(shape=(100_000 + Nsig, 2))
    data[:100_000] = np.random.exponential(scale=[4, 4], size=(100_000, 2))
    data[100_000:] = np.random.multivariate_normal(
        mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(Nsig)
    )

    # Signal
    sig = np.random.multivariate_normal(
        mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(10_000)
    )

    # Put everything in a DataHandler
    dh = BH.DataHandler(ndim=2, nchan=1)
    dh.set_ref(
        bkg,
        bins=[[20, 20]],
        rang=[[0, 25], [0, 25]],
        weights = np.full(bkg.shape[0], 0.1) # Rescale bkg to data with event weights
    )
    dh.set_data(data)
    dh.set_sig(sig, signal_exp=Nsig)

    # Return the dataset
    return dh

@pytest.fixture
def data_sig_bkg1():
    return make_datasets(seed=42)

@pytest.fixture
def bhunter():
    return BH.BumpHunter2D(
               width_min=[2, 2],
               width_max=[3, 3],
               width_step=[1, 1],
               scan_step=[1, 1],
               npe=8000,
               nworker=1,
               seed=666
           )

# Test if the bump_scan method runs
def test_scan_run(bhunter, data_sig_bkg1):
    # Get the data
    dh = data_sig_bkg1

    # Run the bump_scan method
    bhunter.bump_scan(dh)

    # Test if the position of the Bump is correct w.r.t. the expected value
    assert bhunter.min_loc_ar[0, :, 0].all() == np.array([3, 5]).all()

    # Test if the width of the bump is correct w.r.t. the expected value
    assert bhunter.min_width_ar[0, :, 0].all() == np.array([3, 3]).all()

    # Test if the local p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.min_Pval_ar[0, 0]:.5g}" == "4.4481e-05"

    # Test if the global p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.global_Pval[0]:.5f}" == "0.02863"
    
    # Test if the number of tested intervals is correct w.r.t. the expected value
    N = 0
    for r in bhunter.res_ar[0]:
        N += r.size
    assert N == 1369

    # Test if the evaluated number of signal event is correct w.r.t. the expected value
    assert int(bhunter.signal_eval[0]) == 163

# Test if the SignalInject method runs
def test_inject_run(bhunter, data_sig_bkg1):
    # Get the data
    dh = data_sig_bkg1

    bhunter.sigma_limit = 5
    bhunter.str_min = -1 # if str_scale='log', the real starting value is 10**str_min
    bhunter.str_scale = 'log'

    bhunter.signal_inject(dh)

    assert f"{bhunter.signal_ratio:.2f}" == "2.00"

    assert int(bhunter.signal_min[0]) == 1000


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
    Nsig = 700
    data = np.empty(shape=(100_000 + Nsig, 2))
    data[:100_000] = np.random.exponential(scale=[4, 4], size=(100_000, 2))
    data[100_000:] = np.random.multivariate_normal(
        mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(Nsig)
    )

    # Signal
    # Not needed yet

    # Return the dataset
    return data, bkg

@pytest.fixture
def data_sig_bkg1():
    return make_datasets(seed=42)

@pytest.fixture
def bhunter():
    return BH.BumpHunter2D(rang=[[0, 25], [0, 25]],
                           width_min=[2, 2],
                           width_max=[3, 3],
                           width_step=[1, 1],
                           scan_step=[1, 1],
                           bins=[20, 20],
                           npe=8000,
                           nworker=1,
                           seed=666,
                           use_sideband=True)

# Test if the bump_scan method runs
def test_scan_run(data_sig_bkg1, bhunter):
    # Get the data
    data, bkg = data_sig_bkg1

    # Run the bump_scan method
    bhunter.bump_scan(data, bkg)

    # Test if the position of the Bump is correct w.r.t. the expected value
    assert bhunter.min_loc_ar[0] == [3, 5]

    # Test if the width of the bump is correct w.r.t. the expected value
    assert bhunter.min_width_ar[0] == [3, 3]

    # Test if the local p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.min_Pval_ar[0]:.5g}" == '3.6572e-07'

    # Test if the global p-value is correct w.r.t. the expected value (up to 5 digit)
    assert f"{bhunter.global_Pval:.5f}" == "0.00063"
    
    # Test if the number of tested intervals is correct w.r.t. the expected value
    N = 0
    for r in bhunter.res_ar:
        N += r.size
    assert N == 1369

    # Test if the evaluated number of signal event is correct w.r.t. the expected value
    assert int(bhunter.signal_eval) == 277


''' 2D signal injection is not implemented yet
# Test if the SignalInject method runs
def test_inject_run():
    BHtest.sigma_limit = 5
    BHtest.str_min = -1 # if str_scale='log', the real starting value is 10**str_min
    BHtest.str_scale = 'log'
    BHtest.signal_exp = 150 # Correspond the the real number of signal events generated when making the data

    BHtest.SignalInject(sig,bkg)

# Test if the final signal strength is correct w.r.t. the expected value (up to 2 digit)
def test_signal_str():
    assert '{0:.2f}'.format(BHtest.signal_ratio) == '2.00'

# Test if the number of injected event is correct w.r.t. the expected value
def test_number_inject():
    assert int(BHtest.signal_min) == 300
'''



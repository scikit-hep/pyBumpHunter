# Test script to be run using pytest
# Here, all the values required to pass te test has been validated using the
# orriginal c++ version of BumpHunter used by ATLAS collaboration.

import pyBumpHunter as BH
import pytest
import uproot as upr
import numpy as np

# Import the data
File = upr.open('data/data.root')
bkg = File['bkg'].arrays(library='np')
data = File['data'].arrays(library='np')
sig = File['sig'].arrays(library='np')

# Create the BumpHunter instance
BHtest = BH.BumpHunter(rang=[0,20],
                       width_min=2,
                       width_max=6,
                       width_step=1,
                       scan_step=1,
                       Npe=10000,
                       Nworker=1,
                       seed=666)

# Test if the the BumpScan method runs
def test_scan_run():
    BHtest.BumpScan(data,bkg)

# Test if the position of the Bump is correct w.r.t. the expected value
def test_bump_pos():
    assert BHtest.min_loc_ar[0] == 16 #16th bin

# Test if the width of the bump is correct w.r.t. the expected value
def test_bump_width():
    assert BHtest.min_width_ar[0] == 4 #4 bins

# Test if the local p-value is correct w.r.t. the expected value (up to 7 digit)
def test_local_pval():
    assert '{0:.7f}'.format(BHtest.min_Pval_ar[0]) == '0.0001734'

# Test if the global p-value is correct w.r.t. the expected value (up to 5 digit)
def test_global_pval():
    assert '{0:.5f}'.format(BHtest.global_Pval) == '0.01770'

# Test if the number of tested intervals is correct w.r.t. the expected value
def test_number_interval():
    N = 0
    for res in BHtest.res_ar:
        for r in res:
            N += r.size
    assert N == 2811150

# Test if the evaluated number of signal event is correct w.r.t. the expected value
def tesl_number_signal():
    Hdata,_ = np.histogram(data,bins=BHtest.bins,range=BHtest.rang)
    Hbkg,_ = np.histogram(bkg,bins=BHtest.bins,range=BHtest.rang,weights=Bhtest.weights)
    D = Hdata[min_loc_ar[0]:min_loc_ar[0]+min_width_ar[0]].sum()
    B = Hbkg[min_loc_ar[0]:min_loc_ar[0]+min_width_ar[0]].sum()
    assert int(D-B) == 208

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




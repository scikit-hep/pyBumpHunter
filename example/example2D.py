# Here we test the BumpHunter2D class.
# This is an extension of the BumpHunter algorithm to 2D histograms.
# We will use 2D histograms ranging between 0 and 25 (both axis) with 20*20 even bins.

import matplotlib
import numpy as np

matplotlib.use("Agg")
from datetime import datetime  # # Used to compute the execution time

import matplotlib.pyplot as plt

import pyBumpHunter as BH

# Generate the background
np.random.seed(42)
bkg = np.random.exponential(scale=[4, 4], size=(100000, 2))

# Generate the data
Nsig = 500
data = np.empty(shape=(100000 + Nsig, 2))
data[:100000] = np.random.exponential(scale=[4, 4], size=(100000, 2))
data[100000:] = np.random.multivariate_normal(
    mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(Nsig)
)

# Generate the signal
sig = np.random.multivariate_normal(
    mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(10000)
)

# Expected position of the bump in the data
Lth = [6.0, 7.0]

# Range of the histograms (used in the scans)
rang = [[0, 25], [0, 25]]

# Plot the 2 distributions (data and background) as 2D histograms
F = plt.figure(figsize=(11, 10))
plt.title("Test distribution (background)")
_, binx, biny, _ = plt.hist2d(
    bkg[:, 0], bkg[:, 1], bins=[20, 20], range=rang, norm=matplotlib.colors.LogNorm()
)
plt.colorbar()
plt.savefig("results/2D/hist_bkg.png", bbox_inches="tight")
plt.close(F)

# The red dashed lines show the true posision of the signal
F = plt.figure(figsize=(11, 10))
plt.title("Test distribution (data)")
plt.hist2d(
    data[:, 0], data[:, 1], bins=[20, 20], range=rang, norm=matplotlib.colors.LogNorm()
)
plt.hlines([5.0, 9.0], binx[0], binx[-1], linestyles="dashed", color="r")
plt.vlines([4.0, 8.0], biny[0], biny[-1], linestyles="dashed", color="r")
plt.colorbar()
plt.savefig("results/2D/hist_data.png", bbox_inches="tight")
plt.close(F)

# Create a BumpHunter class instance
BHtest = BH.BumpHunter2D(
    rang=rang,
    width_min=[2, 2],
    width_max=[3, 3],
    width_step=[1, 1],
    scan_step=[1, 1],
    bins=[20, 20],
    Npe=8000,
    Nworker=1,
    seed=666,
)

# Call the BumpScan method
print("####BumpScan call####")
begin = datetime.now()
BHtest.BumpScan(data, bkg)
end = datetime.now()
print("time={}".format(end - begin))
print("")

# Print bump
BHtest.PrintBumpInfo()
BHtest.PrintBumpTrue(data, bkg)
print(f"   mean (true) = {Lth}")
print("")

# Get and save tomography plot
# BHtest.GetTomography(data,filename='results/tomography.png')


# Get and save bump plot
BHtest.PlotBump(data, bkg, filename="results/2D/bump.png")

# Get and save statistics plot
BHtest.PlotBHstat(show_Pval=True, filename="results/2D/BH_statistics.png")

"""  2D signal injection is not implemeted yet
print('')

# We have to set additionnal parameters specific to the signal injection.
# All the parameters defined previously are kept.
BHtest.sigma_limit = 5
BHtest.str_min = -1 # if str_scale='log', the real starting value is 10**str_min
BHtest.str_scale = 'log'
BHtest.signal_exp = 150 # Correspond the the real number of signal events generated when making the data

print('####SignalInject call####')
begin = datetime.now()
BHtest.SignalInject(sig,bkg,is_hist=False)
end = datetime.now()
print('time={}'.format(end-begin))
print('')


# Get and save the injection plot
BHtest.PlotInject(filename=('results/SignalInject.png','results/SignalInject_log.png'))
"""

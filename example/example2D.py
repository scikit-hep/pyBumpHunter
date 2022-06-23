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
bkg = np.random.exponential(scale=[4, 4], size=(1_000_000, 2)) # Need more stat to have a smoother reference

# Generate the data
Nsig = 700
data = np.empty(shape=(100_000 + Nsig, 2))
data[:100_000] = np.random.exponential(scale=[4, 4], size=(100_000, 2))
data[100_000:] = np.random.multivariate_normal(
    mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(Nsig)
)
sig = np.random.multivariate_normal(
    mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=(10_000)
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
plt.xticks(fontsize="xx-large")
plt.yticks(fontsize="xx-large")
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

# Create a DataHandler class instance and build histograms
dh = BH.DataHandler(ndim=2, nchan=1)
dh.set_ref(
    bkg,
    bins=[[20, 20]],
    rang=rang,
    weights=np.full(bkg.shape[0], 0.1) # Use event weights to normalize the background to data
)
dh.set_data(data)
dh.set_sig(sig, signal_exp=Nsig)  # Set the expected number of signal event here

# Create a BumpHunter2D class instance
hunter = BH.BumpHunter2D(
    width_min=[2, 2],
    width_max=[3, 3],
    width_step=[1, 1],
    scan_step=[1, 1],
    npe=10_000,
    nworker=1,
    seed=666
)

# Call the bump_scan method
print("####bump_scan call####")
begin = datetime.now()
hunter.bump_scan(dh)
end = datetime.now()
print(f"time={end - begin}")
print("")

# Print bump
print(hunter.bump_info(dh))
print(f"   mean (true) = {Lth}")
print("")


# Get and save bump plot
F = plt.figure(figsize=(10, 15))
plt.suptitle("Distribution with bump", size=24)
pl = hunter.plot_bump(dh, fontsize=24)
pl[1].axes.set_xlabel("local significance map", size=24)
plt.savefig("results/2D/bump.png", bbox_inches="tight")
plt.close(F)

# Get and save statistics plot
F = plt.figure(figsize=(12, 8))
plt.title(f"BumpHunter test statistic distribution\nglobal significance={hunter.significance[0]:.3f}$\sigma$", size=24)
hunter.plot_stat()
plt.legend(fontsize=24)
plt.xlabel("test statistic", size=24)
plt.ylabel("pseudo-data count", size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("results/2D/BH_statistics.png", bbox_inches="tight")
plt.close(F)

# We have to set additionnal parameters specific to the signal injection.
# All the parameters defined previously are kept.
hunter.sigma_limit = 5
hunter.str_min = -1  # if str_scale='log', the real starting value is 10**str_min
hunter.str_scale = "log"
hunter.npe_inject = 1000

print("####singal_inject call####")
begin = datetime.now()
hunter.signal_inject(dh, do_pseudo=False) # We don't need to scan bkg-only pseudo-data again
end = datetime.now()
print(f"time={end - begin}")

# Get and save the injection plot
F = plt.figure(figsize=(12,8))
plt.title("Signal injection test", size=24)
hunter.plot_inject()
plt.xlabel("signal strength", size=24)
plt.ylabel("global significance ($\sigma$)", size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("results/2D/SignalInject.png", bbox_inches="tight")

F = plt.figure(figsize=(12,8))
plt.title("Signal injection test", size=24)
hunter.plot_inject(log=True)
plt.xlabel("signal strength", size=24)
plt.ylabel("global significance ($\sigma$)", size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("results/2D/SignalInject_log.png", bbox_inches="tight")
plt.close(F)


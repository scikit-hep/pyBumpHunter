# Here we test pyBumpHunter.
# The result can be compared to what can be obtained with the original C++ version.
# We will use histograms ranging between 0 and 20 with 60 even bins.

import matplotlib

matplotlib.use("Agg")
from datetime import datetime  # # Used to compute the execution time

import matplotlib.pyplot as plt
import uproot  # # Used to read data from a root file

import pyBumpHunter as BH

# Open the file
with uproot.open("../data/data.root") as file:
    # Background
    bkg = file["bkg"].arrays(library="np")["bkg"]

    # Data
    data = file["data"].arrays(library="np")["data"]

    # Signal
    sig = file["sig"].arrays(library="np")["sig"]

# Position of the bump in the data
Lth = 5.5

# Range for the histograms (same that the one used with C++ BumpHunter)
rang = [0, 20]

# Plot the 2 distributions
F = plt.figure(figsize=(12, 8))
plt.title("Test distribution")
plt.hist(
    (bkg, data),
    bins=60,
    histtype="step",
    range=rang,
    label=("bakground", "data"),
    linewidth=2,
)
plt.legend(fontsize='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig("results/1D/hist.png", bbox_inches="tight")
plt.close(F)

# Create an DataHandler class to make histograms
dh = BH.DataHandler(ndim=1, nchan=1)
dh.set_ref(bkg, bins=60, rang=rang)
dh.set_data(data)
dh.set_sig(sig, signal_exp=150)

# Create a BumpHunter1D class instance
hunter = BH.BumpHunter1D(
    width_min=2,
    width_max=6,
    width_step=1,
    scan_step=1,
    npe=10000,
    nworker=1,
    seed=666,
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

# Get and save tomography plot
F = plt.figure(figsize=(12, 8))
plt.title("Tomography plot", size=24)
hunter.plot_tomography(dh)
plt.xlabel("intervals", size=24)
plt.ylabel("local p-value", size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("results/1D/tomography.png", bbox_inches="tight")
plt.close(F)

# Get and save bump plot
F = plt.figure(figsize=(12, 8))
plt.suptitle("Distributions with bump", size=24)
pl = hunter.plot_bump(dh, fontsize=24)
pl[0].legend(fontsize=24)
pl[0].axes.set_ylabel("envent count", size=24)
pl[1].axes.set_xlabel("variable", size=24)
pl[1].axes.set_ylabel("significance", size=24)
plt.savefig("results/1D/bump.png", bbox_inches="tight")
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
plt.savefig("results/1D/BH_statistics.png", bbox_inches="tight")
plt.close(F)

print("")

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
plt.savefig("results/1D/SignalInject.png", bbox_inches="tight")

F = plt.figure(figsize=(12,8))
plt.title("Signal injection test", size=24)
hunter.plot_inject(log=True)
plt.xlabel("signal strength", size=24)
plt.ylabel("global significance ($\sigma$)", size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("results/1D/SignalInject_log.png", bbox_inches="tight")
plt.close(F)


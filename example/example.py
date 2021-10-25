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
plt.legend()
plt.savefig("results/1D/hist.png", bbox_inches="tight")
plt.close(F)

# Create a BumpHunter1D class instance
hunter = BH.BumpHunter1D(
    rang=rang,
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
hunter.bump_scan(data, bkg)
end = datetime.now()
print(f"time={end - begin}")
print("")

# Print bump
print(hunter.bump_info(data))
print(f"   mean (true) = {Lth}")
print("")

# Get and save tomography plot
hunter.plot_tomography(data, filename="results/1D/tomography.png")

# Get and save bump plot
hunter.plot_bump(data, bkg, filename="results/1D/bump.png")

# Get and save statistics plot
hunter.plot_stat(show_Pval=True, filename="results/1D/BH_statistics.png")

print("")

# We have to set additionnal parameters specific to the signal injection.
# All the parameters defined previously are kept.
hunter.sigma_limit = 5
hunter.str_min = -1  # if str_scale='log', the real starting value is 10**str_min
hunter.str_scale = "log"
hunter.signal_exp = 150  # Correspond the the real number of signal events generated when making the data

print("####singal_inject call####")
begin = datetime.now()
hunter.signal_inject(sig, bkg, is_hist=False)
end = datetime.now()
print(f"time={end - begin}")
print("")

# Get and save the injection plot
hunter.plot_inject(filename=("results/1D/SignalInject.png", "results/1D/SignalInject_log.png"))



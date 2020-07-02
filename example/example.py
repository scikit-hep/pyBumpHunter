# Here we test pyBumpHunter.
# The result can be compared to what can be obtained with the original C++ version.
# We will use histograms ranging between 0 and 20 with 60 even bins.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyBumpHunter as BH
from datetime import datetime  ## Used to compute the execution time
import uproot as upr  ## Used to read data from a root file


# Open the file
File = upr.open('data/data.root')
File.items()

# Background
bkg = File['bkg'].arrays(outputtype=np.array)

# Data
data = File['data'].arrays(outputtype=np.array)

# Signal
sig = File['sig'].arrays(outputtype=np.array)

# Position of the bump in the data
Lth = 5.5

# Range for the hitograms (same that the one used with C++ BumpHunter)
rang = [0,20]

# Plot the 2 distributions
F = plt.figure(figsize=(12,8))
plt.title('Test distribution')
plt.hist((bkg,data),bins=60,histtype='step',range=rang,label=('bakground','data'),linewidth=2)
plt.legend()
plt.savefig('results_py/hist.png',bbox_inches='tight')
plt.close(F)


# Create a BumpHunter class instance
BHtest = BH.BumpHunter(rang=rang,
                       width_min=2,
                       width_max=6,
                       width_step=1,
                       scan_step=1,
                       Npe=10000,
                       Nworker=1,
                       seed=666)


# Call the BumpScan method
print('####BmupScan call####')
begin = datetime.now()
BHtest.BumpScan(data,bkg)
end = datetime.now()
print('time={}'.format(end-begin))
print('')

# Print bump
BHtest.PrintBumpInfo()
BHtest.PrintBumpTrue(data,bkg)
print('   mean (true) = {}'.format(Lth))
print('')


# Get and save tomography plot
BHtest.GetTomography(data,filename='results_py/tomography.png')


# Get and save bump plot
BHtest.PlotBump(data,bkg,filename='results_py/bump.png')


# Get and save statistics plot
BHtest.PlotBHstat(show_Pval=True,filename='results_py/BH_statistics.png')

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
BHtest.PlotInject(filename=('results_py/SignalInject.png','results_py/SignalInject_log.png'))


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
File = upr.open('data.root')
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
'''
pval_py = np.fromfile('pval_py.txt',sep='\n')
pval_c = np.fromfile('gpval_c/pval_new.txt',sep='\n')
F = plt.figure(figsize=(12,8))
plt.hist((pval_py,pval_c),bins=40,histtype='step',linewidth=2,label=('python','c++'))
plt.legend(fontsize='large')
plt.xlabel('Global p-value',size='large')
plt.savefig('gpval_all/pval_new.png',bbox_inches='tight')
plt.close(F)
f = open('gpval_all/stat_new.txt','w')
print('stat python : ',file=f)
print(f'   mean={pval_py.mean()}',file=f)
print(f'   std={pval_py.std()}',file=f)
print(f'   min={pval_py.min()}',file=f)
print(f'   max={pval_py.max()}',file=f)
print('',file=f)
print('stat c++ : ',file=f)
print(f'   mean={pval_c.mean()}',file=f)
print(f'   std={pval_c.std()}',file=f)
print(f'   min={pval_c.min()}',file=f)
print(f'   max={pval_c.max()}',file=f)
f.close()
'''
# Call the BumpHunter function
print('####BmupHunter call####')
begin = datetime.now()
BH.BumpHunter(
    data,bkg,Rang=rang,
    Width_min=2,
    Width_max=6,
    Width_step=1,
    Scan_step=1,
    npe=10000,
    NWorker=1,
    Seed=666
)
end = datetime.now()
print('time={}'.format(end-begin))
print('')

# Print bump
BH.PrintBumpInfo()
BH.PrintBumpTrue(data,bkg)
print('   mean (true) = {}'.format(Lth))
print('')


# Get and save tomography plot
BH.GetTomography(data,filename='results_py/tomography1.png')


# Get and save bump plot
BH.PlotBump(data,bkg,filename='results_py/bump1.png')


# Get and save statistics plot
BH.PlotBHstat(show_Pval=True,filename='results_py/BH_statistics1.png')

print('')
# Set injection parrameter and call SignalInject function with keepparam argument
# (so we keep the same BumpHunter parameters, except NPE that I also modify)
BH.sigma_limit = 3
BH.str_min = -1
#BH.str_step = 0.1
BH.str_scale = 'log'
BH.signal_exp = 150

print('####SignalInject call####')
begin = datetime.now()
BH.SignalInject(sig,bkg,is_hist=False,keepparam=True)
end = datetime.now()
print('time={}'.format(end-begin))
print('')

# Print new bump after signal injection (use BH.data_inject to get the generated data)
BH.PrintBumpInfo()
BH.PrintBumpTrue(BH.data_inject,bkg)
print('   mean (true) = {}'.format(Lth))
print('')

Hbkg = np.histogram(bkg,bins=BH.bins,weights=BH.weights)[0]

# Get and save new tomography plot
BH.GetTomography(BH.data_inject,is_hist=True,filename='results_py/tomography2.png')


# Get and save new bump plot
BH.PlotBump(BH.data_inject,Hbkg,is_hist=True,filename='results_py/bump2.png')


# Get and save new statistics plot
BH.PlotBHstat(show_Pval=True,filename='results_py/BH_statistics2.png')

# Get and save the injection plot
BH.PlotInject(filename=('results_py/SignalInject.png','results_py/SignalInject_log.png'))



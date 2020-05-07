########pyBumpHunter########
#
# Python version of the BupHunter algorithm as described in https://arxiv.org/pdf/1101.0390.pdf

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc as G  ## Need G(a,b) for the gamma function
from scipy.stats import norm
import concurrent.futures as thd
from matplotlib import gridspec as grd


# Parameter global variables
rang = None
width_min = 1
width_max = None
width_step = 1
scan_step = 1
Npe = 100
bins = 60
weights = None
Nworker = 4
seed = None
sigma_limit = 5
str_min = 0.5
str_step = 0.25
str_scale='lin'
signal_exp = None
data_inject = []


# Result global variables
global_Pval = 0
significance = 0
res_ar = []
min_Pval_ar = []
min_loc_ar = []
min_width_ar = []
t_ar = []
signal_min = 0
signal_ratio = None


# Function that perform the scan on every pseudo experiment and data (in parrallel threads).
# For each scan, the value of p-value and test statistic t is computed and stored in result array
def BumpHunter(data,bkg,is_hist=False,Rang=None,
               Width_min=1,Width_max=None,Width_step=1,Scan_step=1,
               npe=100,Bins=60,Weights=None,NWorker=4,
               Seed=None,keepparam=False):
    '''    
    Function that perform the full BumpHunter algorithm presented in https://arxiv.org/pdf/1101.0390.pdf
    without sidebands. This includes the generation of pseudo-data, the calculation of the BumpHunter p-value
    associated to data and to all pseudo experiment as well as the calculation of the test satistic t.
    
    The values passed to the arguments and the results are stored in global variables. If ones call the result
    variables before running this function, they will get either a empty list or 0.
    
    Arguments :
        data : Numpy array containing the data distribution. This distribution will be transformed into a binned
               histogram and the algorithm will look for the most significant excess.
        
        bkg : Numpy array containing the background reference distribution. This distribution will be transformed
              into a binned histogram and the algorithm will compare it to data while looking for a bump.
        
        is_hist : Boolean that specify if the given data and background are already in histogram form. If true,
                  data and bkg are histograms and the Bins argument is expected to be a array with the bound of
                  the bins and the Weights argument is expected to be a scalar scale factor or None.
                  Default to False.
        
        Rang : x-axis range of the histograms. Also define the range in which the scan wiil be performed.
        
        Width_min : Minimum value of the scan window width that should be tested. Default to 1.
        
        Width_max : Maximum value of the scan window width that should be tested. Can be either None or a
                    positive integer. if None, the value is set to the total number of bins of the histograms.
                    Default to none.
        
        Width_step : Number of bins by which the scan window width is increased at each step. Default to 1.
        
        Scan_step : Number of bins by which the position of the scan window is shifted at each step. Can be
                    either 'full', 'half' or a positive integer.
                    If 'full', the window will be shifted by a number of bins equal to its width.
                    If 'half', the window will be shifted by a number of bins equal to max(1,width//2).
                    Default to 1.
        
        npe : Number of pseudo-data distributions to be sampled from the reference background distribution.
              Default to 100.
        
        Bins : Define the bins of the histograms. Can be ether a integer of a array-like of floats.
               If integer (N), N bins of equal width will be considered.
               If array-like of float (a), a number of bins equal to a length-1 with the values of a as edges will
               be considered (variable width bins allowed).
               Default to 60.
        
        Weights : Weights for the background distribution. Can be either None or a array-like of float.
                  If array-like of floats, each background events will be acounted by its weights when making
                  histograms. The size of the array-like must be the same than of bkg. The same weights are
                  considered when sampling the pseudo-data.
                  If None, no weights will be considered.
                  Default to 1.
        
        NWorker : Number of thread to be run in parallel when scanning all the histograms (data and pseudo-data).
                  If less or equal to 1, then parallelism will be disabled.
                  Default to 4.
    
        Seed : Seed for the random number generator. Default to None.
        
        keepparam : Boolean specifying if BumpHunter should use the parameters saved in global variables. If True,
                    all other arguments (except data and bkg) are ignored.
                    Default to False.
    
    Result global variables :
        global_Pval : Global p-value obtained from the test statistic distribution.
        
        res_ar : Array of containers containing all the p-value calculated durring the scan of the data (indice=0)
                 and of the pseudo-data (indice>0). For more detail about how the p-values are sorted in the
                 containers, please reffer the the doc of the function scan_hist.
        
        min_Pval_ar : Array containing the minimum p-values obtained for the data (indice=0) and and the pseudo-
                      data (indice>0).
        
        min_loc_ar : Array containing the positions of the windows for which the minimum p-value has been found
                     for the data (indice=0) and pseudo-data (indice>0).
        
        min_width_ar : Array containing the width of the windows for which the minimum p-value has been found for
                       the data (indice=0) and pseudo-data (indice>0).
    '''
    
    # Function that computes the number of iteration for the scan and correct scan_max if not integer
    def get_Nscan(m,M,step):
        '''
        Function that determine the number of step to be computed given a minimum, maximum and step interval.
        The function ensure that this number is integer by adjusting the value of the maximum if necessary.
        
        Arguments :
            m : The value of the minimum (must be integer).
            
            M : The value of the maximum (must be integer).
            
            step : The value of the step interval (must be integer).
        
        Returns : 
            Nscan : The number of iterations to be computed for the scan.
            M : The corrected value of the maximum.
        '''
        
        Nscan = (M-m)/step
        
        if(Nscan-(M-m)//step)>0.0:
            M = np.ceil(Nscan)*step+m
            M = int(M)
            print('ajusting width_max to {}'.format(M))
            Nscan=(M-m)//step
        else:
            Nscan=int(Nscan)
        
        return Nscan+1,M
    
    
    # Function that scan a histogram and compare it to the refernce.
    # Returns a numpy array of python list with all p-values for all windows width and position.
    # This function might be called in parallel threads in order to save time.
    def scan_hist(hist,ref,ih):
        '''
        Function that scan a distribution and compute the p-value associated to every scan window following the
        BumpHunter algorithm. Compute also the significance for the data histogram.
        
        In order to make the function thread friendly, the results are saved through global variables.
        
        Arguments :
            hist : The data histogram (as obtain with the numpy.histogram function).
            
            ref : The reference (background) histogram (as obtain with the numpy.histogram function).
            
            ih : Indice of the distribution to be scanned. ih=0 refers to the data distribution and ih>0 refers to
                 the ih-th pseudo-data distribution.
        
        Results stored in global variables :
            res : Numpy array of python list containing all the p-values of all windows computed durring the
                  scan. The numpy array as dimention (Nwidth), with Nwidth the number of window's width tested.
                  Each python list as dimension (Nstep), with Nstep the number of scan step for a given width
                  (different for every value of width).
                  
            min_Pval : Minimum p_value obtained durring the scan (float).
            
            min_loc : Position of the window corresponding to the minimum p-value (integer).
            
            min_width : Width of the window corresponding to the minimum p-value (integer).
        '''
        # Remove the first/last hist bins if empty ... just to be consistant we c++
        non0 = [iii for iii in range(hist.size) if hist[iii]>0]
        Hinf,Hsup = min(non0),max(non0)+1
        #hist = hist[min(non0):max(non0)+1]
        
        # Create the results array
        res = np.empty(Nwidth,dtype=np.object)
        min_Pval,min_loc = np.empty(Nwidth),np.empty(Nwidth)
        
        # Loop over all the width of the window
        for i in range(Nwidth):
            # Compute the actual width
            w = width_min + i*width_step
            
            # Auto-adjust scan step if specified
            if(scan_step=='full'):
                scan_stepp = w
            elif(scan_step=='half'):
                scan_stepp = max(1,w//2)
            else:
                scan_stepp = scan_step
            
            # Initialize local p-value array for width w
            res[i] = np.empty((Hsup-w+1-Hinf)//scan_stepp)
            
            # Count events in all windows of width w
            #FIXME any better way to do it ?? Without loop ?? FIXME
            count = [[hist[loc*scan_stepp:loc*scan_stepp+w].sum(),ref[loc*scan_stepp:loc*scan_stepp+w].sum()] for loc in range(Hinf,(Hsup-w+1)//scan_stepp)]
            count = np.array(count)
            Nhist,Nref = count[:,0],count[:,1]
            del count
            
            # Calculate all local p-values for for width w
            res[i][Nhist<=Nref] = 1.0
            res[i][Nhist>Nref] = G(Nhist[Nhist>Nref],Nref[Nhist>Nref])
            res[i][(Nhist==0) & (Nref==0)] = 1.0
            res[i][(Nref==0) & (Nhist>0)] = 1.0 # To be consistant with c++ results
            
            # Get the minimum p-value and associated position for width w
            min_Pval[i] = res[i].min()
            min_loc[i] = res[i].argmin()*scan_stepp + Hinf
        
        # Get the minimum p-value and associated windonw among all width
        min_width = width_min + (min_Pval.argmin())*width_step
        min_loc = min_loc[min_Pval.argmin()]
        min_Pval = min_Pval[min_Pval.argmin()]
        
        # Save the results in global variables and return
        res_ar[ih] = res
        min_Pval_ar[ih] = min_Pval
        min_loc_ar[ih] = int(min_loc)
        min_width_ar[ih] = int(min_width)
        return 
    
    
    # Set global parameter variables
    global rang
    global width_min
    global width_max
    global width_step
    global scan_step
    global Npe
    global bins
    global weights
    global Nworker
    global seed
    
    # Check if we must keep the old parameter values or not
    if(keepparam==False):
        rang = Rang
        width_min = Width_min
        width_max = Width_max
        width_step = Width_step
        scan_step = Scan_step
        Npe = npe
        bins = Bins
        weights = Weights
        Nworker = NWorker
        seed = Seed
    
    # Set the seed if required (or reset it if None)
    np.random.seed(seed)
    
    # Set global result variables
    global global_Pval
    global significance
    global res_ar
    global min_Pval_ar
    global min_loc_ar
    global min_width_ar
    global t_ar
    
    
    # Generate the background and data histograms
    if(is_hist==False):
        print('Generating histograms')
        F = plt.figure()
        bkg_hist,Hbin = np.histogram(bkg,bins=bins,weights=weights,range=rang)
        data_hist = np.histogram(data,bins=bins,range=rang)[0]
        plt.close(F)
    else:
        if(weights==None):
            bkg_hist = bkg
        else:
            bkg_hist = bkg * weights
        data_hist = data
        Hbins = Bins
    
    # Generate all the pseudo-data histograms
    pseudo_hist = np.random.poisson(lam=np.tile(bkg_hist,(Npe,1)).transpose(),size=(bkg_hist.size,Npe))
    
    # Set width_max if it is given as None
    if width_max==None:
        width_max = data_hist.size // 2
    
    # Initialize all results containenrs
    min_Pval_ar = np.empty(Npe+1)
    min_loc_ar = np.empty(Npe+1,dtype=int)
    min_width_ar = np.empty(Npe+1,dtype=int)
    res_ar = np.empty(Npe+1,dtype=np.object)
    
    # Auto-adjust the value of width_max and compute Nwidth
    Nwidth,width_max = get_Nscan(width_min,width_max,width_step)
    print('{} values of width will be tested'.format(Nwidth))
    
    # Compute the p-value for data and all pseudo-experiments
    # We must check if we should do it in multiple threads
    print('SCAN')
    if(Nworker>1):
        with thd.ThreadPoolExecutor(max_workers=Nworker) as exe:
            for th in range(Npe+1):
                if(th==0):
                    exe.submit(scan_hist,data_hist,bkg_hist,th)
                else:
                    exe.submit(scan_hist,pseudo_hist[:,th-1],bkg_hist,th)
    else:
        for i in range(Npe+1):
            if(i==0):
                scan_hist(data_hist,bkg_hist,i)
            else:
                scan_hist(pseudo_hist[:,i-1],bkg_hist,i)
    
    # Use the p-value results to compute t
    t_ar = -np.log(min_Pval_ar)
    
    # Compute the global p-value from the t distribution
    tdat = t_ar[0]
    S = t_ar[1:][t_ar[1:]>tdat].size
    global_Pval = S/Npe
    print('Global p-value : {0:1.4f}  ({1} / {2})'.format(global_Pval,S,Npe))
    
    # If global p-value is exactly 0, we might have trouble with the significance
    if(global_Pval<0.0000000000000001):
        significance = norm.ppf(1-0.0000000000000001)
    else:
        significance = norm.ppf(1-global_Pval)
    print('Significance = {0:1.5f}'.format(significance))
    print('')
    
    return


# Function that do the tomography plot for the data
def GetTomography(data,is_hist=False,filename=None):
    '''
    Function that do a tomography plot showing the local p-value for every positions and widths of the scan
    window.
    
    Arguments :
        data : Numpy array containing the raw unbined data.
        
        is_hist : Boolean specifying if data is in histogram form or not. Default to False.
    
        filename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                   but not saved. Default to None.
    '''
    
    # Set global variables
    global res_ar
    global width_min
    global width_step
    global scan_step
    global bins
    global rang
    
    # Same c++ compatibility thing
    non0 = [i for i in range(data.size) if data[i]>0]
    Hinf = min(non0)
    
    # Get real bin bounds
    if(is_hist==False):
        H = np.histogram(data,bins=bins,range=rang)[1]
    else:
        H = bins
    
    res_data = res_ar[0]    
    inter = []
    for i in range(res_data.size):
        w = (H[1]-H[0])*(width_min+i*width_step) # bin_width * Nbins
        
        # Get scan step for width w
        if(scan_step=='half'):
            scan_stepp = max(1,(width_min+i*width_step)//2)
        elif(scan_step=='full'):
            scan_stepp = width_min+i*width_step
        else:
            scan_stepp = scan_step
        
        for j in range(len(res_data[i])):
            loc = H[j*scan_stepp+Hinf]
            inter.append([res_data[i][j],loc,w])
    
    F =plt.figure(figsize=(12,8))
    [plt.plot([i[1],i[1]+i[2]],[i[0],i[0]],'r') for i in inter if i[0]<1.0]
    plt.xlabel('intervals',size='large')
    plt.ylabel('local p-value',size='large')
    plt.yscale('log')
    
    if(filename==None):
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')
        plt.close(F)
    return


# Function that print the local infomation about the most significante bump in data
def PrintBumpInfo():
    '''
    Function that print the local infomation about the most significante bump in data. Information are
    printed to stdout.
    '''
    
    # Set global variables
    global min_loc_ar
    global min_width_ar
    global min_Pval_ar
    global t_ar
    
    # Print stuff
    print('BUMP WINDOW')
    print('   loc = {}'.format(min_loc_ar[0]))
    print('   width = {}'.format(min_width_ar[0]))
    print('   local p-value | t = {} | {}'.format(min_Pval_ar[0],t_ar[0]))
    print('')
    
    return

# Function that print the global infomation about the most significante bump in data
def PrintBumpTrue(data,bkg,is_hist=False):
    '''
    Print the global informations about the most significante bump in data in real scale.
    Information are printed to stdout.
    
    Arguments :
        data : Numpy array containing the raw unbined data.
        
        bkg : Numpy array containing the raw unbined background.
        
        is_hist : Boolean specifying if data and bkg are in histogram form or not. Default to False.
    '''
    
    # Set the global variables
    global bins
    global rang
    global min_loc_ar
    global min_width_ar
    global global_Pval
    
    # Get the data and background in histogram form
    if(is_hist==False):
        H = np.histogram(data,bins=bins,range=rang)
        Hb = np.histogram(bkg,bins=bins,range=rang,weights=weights)[0]
    else:
        H = [data,bins]
        Hb = bkg
    
    # Print informations about the bump itself
    print('BUMP POSITION')
    Bmin = H[1][min_loc_ar[0]]
    Bmax = H[1][min_loc_ar[0]+min_width_ar[0]]
    Bmean = (Bmax+Bmin)/2
    Bwidth = Bmax-Bmin
    D =  H[0][min_loc_ar[0]:min_loc_ar[0]+min_width_ar[0]].sum()
    B = Hb[min_loc_ar[0]:min_loc_ar[0]+min_width_ar[0]].sum()
    
    print('   min : {0:.3f}'.format(Bmin))
    print('   max : {0:.3f}'.format(Bmax))
    print('   mean : {0:.3f}'.format(Bmean))
    print('   width : {0:.3f}'.format(Bwidth))
    print('   number of signal events : {}'.format(D-B))
    print('   global p-value : {0:1.5f}'.format(global_Pval))
    print('   significance = {0:1.5f}'.format(significance))
    print('')
    
    return


# Plot the data and bakground histograms with the bump found by BumpHunter highlighted
def PlotBump(data,bkg,is_hist=False,filename=None):
    '''
    Plot the data and bakground histograms with the bump found by BumpHunter highlighted.
    
    Arguments :
        data : Numpy array containing the raw unbined data.
        
        bkg : Numpy array containing the raw unbined background.
        
        is_hist : Boolean specifying if data and bkg are in histogram form or not. Default to False.
        
        filename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                   but not saved. Default to None.
    '''
    
    # Set the global variables
    global min_loc_ar
    global min_width_ar
    global rang
    global weights
    global bins
    
    # Get the data in histogram form
    if(is_hist==False):
        H = np.histogram(data,bins=bins,range=rang)
    else:
        H = [data,bins]
    
    
    # Get bump min and max
    Bmin = H[1][min_loc_ar[0]]
    Bmax = H[1][min_loc_ar[0]+min_width_ar[0]]
    
    # Get the background in histogram form
    if(is_hist==False):
        Hbkg = np.histogram(bkg,bins=bins,range=rang,weights=weights)[0]
    else:
        if(weights==None):
            Hbkg = bkg
        else:
            Hbkg = bkg * weights
    
    # Calculate significance for each bin
    sig = np.empty(Hbkg.size)
    sig[(H[0]==0) & (Hbkg==0)]=1.0
    sig[H[0]>=Hbkg] = G(H[0][H[0]>=Hbkg],Hbkg[H[0]>=Hbkg])
    sig[H[0]<Hbkg] = 1-G(H[0][H[0]<Hbkg]+1,Hbkg[H[0]<Hbkg])
    sig = norm.ppf(1-sig)
    sig[sig<0] = 0 # If negative, set it to 0
    sig[sig==np.inf] = 0 # Avoid errors
    sig[sig==np.NaN] = 0
    sig[H[0]<Hbkg] = -sig[H[0]<Hbkg]  # Now we can make it signed
    
    # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
    F = plt.figure(figsize=(12,10))
    gs = grd.GridSpec(2, 1, height_ratios=[4, 1])
    
    pl1 = plt.subplot(gs[0])
    plt.title('Distributions with bump')
    
    if(is_hist==False):
        plt.hist(bkg,bins=bins,histtype='step',range=rang,weights=weights,label='background',linewidth=2,color='red')
        plt.errorbar(0.5*(H[1][1:]+H[1][:-1]),H[0],
                     xerr=(H[1][1]-H[1][0])/2,yerr=np.sqrt(H[0]),
                     ls='',color='blue',label='data')
    else:
        plt.hist(bins[:-1],bins=bins,histtype='step',range=rang,weights=Hbkg,label='background',linewidth=2,color='red')
        plt.errorbar(0.5*(H[1][1:]+H[1][:-1]),H[0],
                     xerr=(H[1][1]-H[1][0])/2,yerr=np.sqrt(H[0]),
                     ls='',color='blue',label='data')
    
    plt.plot(np.full(2,Bmin),np.array([0,H[0][min_loc_ar[0]]]),'r--',label=('BUMP'))
    plt.plot(np.full(2,Bmax),np.array([0,H[0][min_loc_ar[0]+min_width_ar[0]]]),'r--')
    plt.legend(fontsize='large')
    plt.yscale('log')
    if rang!=None:
        plt.xlim(rang)
    plt.tight_layout()
    
    plt.subplot(gs[1],sharex=pl1)
    plt.hist(H[1][:-1],bins=H[1],range=rang,weights=sig)
    plt.plot(np.full(2,Bmin),np.array([sig.min(),sig.max()]),'r--',linewidth=2)
    plt.plot(np.full(2,Bmax),np.array([sig.min(),sig.max()]),'r--',linewidth=2)
    plt.yticks(np.arange(np.round(sig.min()),np.round(sig.max())+1,step=1))
    plt.ylabel('significance',size='large')
    
    # Check if the plot should be saved or just displayed
    if(filename==None):
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')
        plt.close(F)
    
    return


# Plot the Bumpunter test statistic distribution with the result for data
def PlotBHstat(show_Pval=False,filename=None):
    '''
    Plot the Bumphunter statistic distribution together with the observed value with the data.
    
    Arguments :
        show_Pval : Boolean specifying if you want the value of global p-value printed on the plot.
                    Default to False.
        
        filename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                   but not saved. Default to None.
    '''
    
    # Set global variables
    global t_ar
    global global_Pval
    
    # Plot the BumpHunter statistic distribution
    F = plt.figure(figsize=(12,8))
    if(show_Pval):
        plt.title('BumpHunter statistics distribution      global p-value = {0:1.4f}'.format(global_Pval))
    else:
        plt.title('BumpHunter statistics distribution')
    H=plt.hist(t_ar[1:],bins=100,histtype='step',linewidth=2,label='pseudo-data')
    plt.plot(np.full(2,t_ar[0]),np.array([0,H[0].max()]),'r--',linewidth=2,label='data')
    plt.legend(fontsize='large')
    plt.xlabel('BumpHunter statistic',size='large')
    plt.yscale('log')
    
    # Check if the plot should be saved or just displayed
    if(filename==None):
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')
        plt.close(F)
    
    return


# Perform signal injection on background and determine the minimum aount of signal required for observation
def SignalInject(sig,bkg,is_hist=False,Rang=None,
                 Width_min=1,Width_max=None,Width_step=1,Scan_step=1,
                 npe=100,Bins=60,Weights=None,NWorker=4,
                 Seed=None,
                 Sigma_limit=5,Str_min=0.5,Str_step=0.25,
                 Str_scale='lin',
                 Signal_exp=None,keepparam=False):
    '''    
    Function that perform a signal injection test in order to determine the minimum signal strength required to
    reach a target significance. This function use the BumpHunter algorithm in order to calculate the reached
    significance for a given signal strength.
    
    This function share most of its parameters with the BumpHunter function.
    
    Arguments :
        sig : Numpy array containing the simulated signal. This distribution will be used to perform the signal
              injection.
        
        bkg : Numpy array containing the expected background. This distribution will be used to build the data in
              which signal will be injected.
        
        is_hist : Boolean that specify if the given signal and background are already in histogram form. If true,
                  data and bkg are histograms and the Bins argument is expected to be a array with the bound of
                  the bins and the Weights argument is expected to be a scalar scale factor or None.
                  Default to False.
        
        Rang : x-axis range of the histograms. Also define the range in which the scan wiil be performed.
        
        Width_min : Minimum value of the scan window width that should be tested. Default to 1.
        
        Width_max : Maximum value of the scan window width that should be tested. Can be either None or a
                    positive integer. if None, the value is set to the total number of bins of the histograms.
                    Default to none.
        
        Width_step : Number of bins by which the scan window width is increased at each step. Default to 1.
        
        Scan_step : Number of bins by which the position of the scan window is shifted at each step. Can be
                    either 'full', 'half' or a positive integer.
                    If 'full', the window will be shifted by a number of bins equal to its width.
                    If 'half', the window will be shifted by a number of bins equal to max(1,width//2).
                    Default to 1.
        
        npe : Number of pseudo-data distributions to be sampled from the reference background distribution.
              Default to 100.
        
        Bins : Define the bins of the histograms. Can be ether a integer of a array-like of floats.
               If integer (N), N bins of equal width will be considered.
               If array-like of float (a), a number of bins equal to a length-1 with the values of a as edges will
               be considered (variable width bins allowed).
               Default to 60.
        
        Weights : Weights for the background distribution. Can be either None or a array-like of float.
                  If array-like of floats, each background events will be acounted by its weights when making
                  histograms. The size of the array-like must be the same than of bkg. The same weights are
                  considered when sampling the pseudo-data.
                  If None, no weights will be considered.
                  Default to 1.
        
        NWorker : Number of thread to be run in parallel when scanning all the histograms (data and pseudo-data).
                  If less or equal to 1, then parallelism will be disabled.
                  Default to 4.
    
        Seed : Seed for the random number generator. Default to None.
        
        Sigma_limit : The minimum significance required after injection. Deault to 5.
        
        Str_min : The minimum number signal stregth to inject in background (first iteration). Default to 0.5.
        
        Str_step : Increase of the signal stregth to be injected in the background at each iteration. Default to 0.25.
        
        str_scale : Specify how the signal strength should vary. If 'log', the the signal strength will vary according to
                    a log scale starting from 10**Str_min. If 'lin', the signal will vary according to a linear scale starting
                    from Str_min with a step of Str_step.
                    Default to 'lin'.
        
        Signal_exp : Expected number of signal used to compute the signal strength. If None, the signal strength is not
                     computed. Default to None.
        
        keepparam : Boolean specifying if BumpHunter should use the parameters saved in global variables. If True,
                    all other arguments (except data and sig) are ignored. If False, all the non specified parameter
                    global variables will be reseted to their default values.
                    Default to False.
    
    Result global variables :        
        signal_ratio : Ratio signal_min/signal_exp (signal strength). If signal_exp is not specified, default to None.
        
        data_inject : Data obtained after injecting signal events in the backgound.
        
        sigma_ar : Numpy array containing the significance values obtained at each step.
    
    All the result global variables of the BumpHunter function will be filled with the results of the scan permormed
    during the last iteration (when sigma_limit is reached).
    '''
    
    # Function that scan a histogram and compare it to the refernce.
    # This is just a local copy of the same function defined inside the BumpHunter function.
    def scan_hist(hist,ref,ih):
        '''
        Function that scan a distribution and compute the p-value associated to every scan window following the
        BumpHunter algorithm. Compute also the significance for the data histogram.
        
        In order to make the function thread friendly, the results are saved through global variables.
        
        Arguments :
            hist : The data histogram (as obtain with the numpy.histogram function).
            
            ref : The reference (background) histogram (as obtain with the numpy.histogram function).
            
            ih : Indice of the distribution to be scanned. ih=0 refers to the data distribution and ih>0 refers to
                 the ih-th pseudo-data distribution.
        
        Results stored in global variables :
            res : Numpy array of python list containing all the p-values of all windows computed durring the
                  scan. The numpy array as dimention (Nwidth), with Nwidth the number of window's width tested.
                  Each python list as dimension (Nstep), with Nstep the number of scan step for a given width
                  (different for every value of width).
                  
            min_Pval : Minimum p_value obtained durring the scan (float).
            
            min_loc : Position of the window corresponding to the minimum p-value (integer).
            
            min_width : Width of the window corresponding to the minimum p-value (integer).
        '''
        # Remove the first/last hist bins if empty ... just to be consistant we c++
        non0 = [iii for iii in range(hist.size) if hist[iii]>0]
        Hinf,Hsup = min(non0),max(non0)+1
        #hist = hist[min(non0):max(non0)+1]
        
        # Create the results array
        res = np.empty(Nwidth,dtype=np.object)
        min_Pval,min_loc = np.empty(Nwidth),np.empty(Nwidth)
        
        # Loop over all the width of the window
        for i in range(Nwidth):
            # Compute the actual width
            w = width_min + i*width_step
            
            # Auto-adjust scan step if specified
            if(scan_step=='full'):
                scan_stepp = w
            elif(scan_step=='half'):
                scan_stepp = max(1,w//2)
            else:
                scan_stepp = scan_step
            
            # Initialize local p-value array for width w
            res[i] = np.empty((Hsup-w+1-Hinf)//scan_stepp)
            
            # Count events in all windows of width w
            #FIXME any better way to do it ?? Without loop ?? FIXME
            count = [[hist[loc*scan_stepp:loc*scan_stepp+w].sum(),ref[loc*scan_stepp:loc*scan_stepp+w].sum()] for loc in range(Hinf,(Hsup-w+1)//scan_stepp)]
            count = np.array(count)
            Nhist,Nref = count[:,0],count[:,1]
            del count
            
            # Calculate all local p-values for for width w
            res[i][Nhist<=Nref] = 1.0
            res[i][Nhist>Nref] = G(Nhist[Nhist>Nref],Nref[Nhist>Nref])
            res[i][(Nhist==0) & (Nref==0)] = 1.0
            res[i][(Nref==0) & (Nhist>0)] = 1.0 # To be consistant with c++ results
            
            # Get the minimum p-value and associated position for width w
            min_Pval[i] = res[i].min()
            min_loc[i] = res[i].argmin()*scan_stepp + Hinf
        
        # Get the minimum p-value and associated windonw among all width
        min_width = width_min + (min_Pval.argmin())*width_step
        min_loc = min_loc[min_Pval.argmin()]
        min_Pval = min_Pval[min_Pval.argmin()]
        
        # Save the results in global variables and return
        res_ar[ih] = res
        min_Pval_ar[ih] = min_Pval
        min_loc_ar[ih] = int(min_loc)
        min_width_ar[ih] = int(min_width)
        return 
    
    # Function that computes the number of iteration for the scan and correct scan_max if not integer
    # Again a local copy
    def get_Nscan(m,M,step):
        '''
        Function that determine the number of step to be computed given a minimum, maximum and step interval.
        The function ensure that this number is integer by adjusting the value of the maximum if necessary.
        
        Arguments :
            m : The value of the minimum (must be integer).
            
            M : The value of the maximum (must be integer).
            
            step : The value of the step interval (must be integer).
        
        Returns : 
            Nscan : The number of iterations to be computed for the scan.
            M : The corrected value of the maximum.
        '''
        
        Nscan = (M-m)/step
        
        if(Nscan-(M-m)//step)>0.0:
            M = np.ceil(Nscan)*step+m
            M = int(M)
            print('ajusting width_max to {}'.format(M))
            Nscan=(M-m)//step
        else:
            Nscan=int(Nscan)
        
        return Nscan+1,M
    
    
    # Set global parameter variables
    global rang
    global width_min
    global width_max
    global width_step
    global scan_step
    global Npe
    global bins
    global weights
    global Nworker
    global seed
    global sigma_limit
    global str_min
    global str_step
    global str_scale
    global signal_exp
    
    # Check if we must keep the old parameter values or not
    if(keepparam==False):
        rang = Rang
        width_min = Width_min
        width_max = Width_max
        width_step = Width_step
        scan_step = Scan_step
        Npe = npe
        bins = Bins
        weights = Weights
        Nworker = NWorker
        seed = Seed
        sigma_limit = Sigma_limit
        str_min = Str_min
        str_min = Str_min
        str_scale = Str_scale
        signal_exp = Signal_exp
        
    # Set the seed if required (or reset it if None)
    np.random.seed(seed)
        
    # Set global result variables
    global global_Pval
    global significance
    global res_ar
    global min_Pval_ar
    global min_loc_ar
    global min_width_ar
    global t_ar
    global signal_min
    global signal_ratio
    global data_inject
    global sigma_ar
    
    # Internal variables
    i = 1
#    inject = 0
    strength = 0
    data = []
    
    # Reset significance and sigma_ar global variable
    significance = 0
    sigma_inf = 0
    sigma_sup = 0
    sigma_ar = []
    
    # Check the expected number of signal event
    if(signal_exp==None):
        if(is_hist==False):
            signal_exp = sig.size
        else:
            signal_exp = sig.sum()
    
    # Turn the background distributions into histogram
    if(is_hist==False):
        bkg_hist,bins = np.histogram(bkg,bins=bins,range=rang,weights=weights)
    
    # Generate pseudo-data by sampling background
    print('Generating background only histograms')
    np.random.seed(seed)
    pseudo_bkg = np.random.poisson(lam=np.tile(bkg_hist,(1000,1)).transpose(),size=(bkg_hist.size,1000))
    
    # Set width_max if it is given as None
    if width_max==None:
        width_max = data_hist.size // 2
    
    # Initialize all results containenrs
    min_Pval_ar = np.empty(1000)
    min_loc_ar = np.empty(1000,dtype=int)
    min_width_ar = np.empty(1000,dtype=int)
    res_ar = np.empty(1000,dtype=np.object)
    
    # Auto-adjust the value of width_max and compute Nwidth
    Nwidth,width_max = get_Nscan(width_min,width_max,width_step)
    print('{} values of width will be tested'.format(Nwidth))
    
    # Compute the p-value for background only pseudo-experiments
    # We must check if we should do it in multiple threads
    print('BACKGROUND ONLY SCAN')
    if(Nworker>1):
        with thd.ThreadPoolExecutor(max_workers=Nworker) as exe:
            for th in range(1000):
                exe.submit(scan_hist,pseudo_bkg[:,th],bkg_hist,th)
    else:
        for th in range(1000):
            scan_hist(pseudo_bkg[:,th],bkg_hist,th)
    
    # Use the p-value results to compute t
    t_ar_bkg = -np.log(min_Pval_ar)
    
    # Save background result separately and free some memory
    min_Pval_ar_bkg = min_Pval_ar
    min_Pval_ar = []
    min_loc_ar_bkg = min_loc_ar
    min_loc_ar = []
    min_width_ar_bkg = min_width_ar
    min_width_ar = []
    res_ar = []
    
    # Main injection loop
    print('STARTING INJECTION')
    while(significance < sigma_limit):
        # Check how we should compute the signal strength to be injected
        if(str_scale=='lin'):
            # Signal strength increase linearly at each step
            if(i==1):
                strength = str_min
            else:
                strength += str_step
            print('   STEP {} : signal strength = {}'.format(i,strength))
            
            # Update signal_min
            signal_min = signal_exp * strength
            i+=1
            
        elif(str_scale=='log'):
            # Signal strength increase to form a logarithmic scale axis
            if(i==1):
                strength = 10**str_min
                str_step = strength
            else:
                strength += str_step
                if(abs(strength-10*str_step)<1e-6):
                    str_step *=10
            print('   STEP {} : signal strength = {}'.format(i,strength))
            
            # Update signal_min
            signal_min = signal_exp * strength
            i+=1
        
        else:
            # If bad str_scale value, print a error mesage and abort
            print("ERROR : Bad str_scale value ! Must be either 'lin' or 'log'")
            return
        
        # Check if the signal is alredy in histogram form or not
        if(is_hist==False):
            sig_hist = np.histogram(sig,bins=bins,range=rang)[0]
            sig_hist = sig_hist * strength * (signal_exp / sig.size)
        else:
            sig_hist = sig
            sig_hist = sig_hist * strength * (signal_exp / sig.sum())
        
        # Inject the signal and do some poissonian fluctuation
        print('Generating background+signal histograms')
        data_hist = bkg_hist + sig_hist
        pseudo_data = np.random.poisson(lam=np.tile(data_hist,(Npe,1)).transpose(),size=(data_hist.size,Npe))
        
        # Initialize all results containenrs
        min_Pval_ar = np.empty(Npe)
        min_loc_ar = np.empty(Npe,dtype=int)
        min_width_ar = np.empty(Npe,dtype=int)
        res_ar = np.empty(Npe,dtype=np.object)
        
        # Compute the p-value for background+signal pseudo-experiments
        # We must check if we should do it in multiple threads
        print('BACKGROUND+SIGNAL SCAN')
        if(Nworker>1):
            with thd.ThreadPoolExecutor(max_workers=Nworker) as exe:
                for th in range(Npe):
                    exe.submit(scan_hist,pseudo_data[:,th],bkg_hist,th)
        else:
            for th in range(Npe):
                scan_hist(pseudo_data[:,th],bkg_hist,th)
        
        # Use the p-value results to compute t
        t_ar = -np.log(min_Pval_ar)
        
        # Compute the global p-value from the t distribution with inf end sup values
        tdat,tinf,tsup = np.median(t_ar),np.quantile(t_ar,0.16),np.quantile(t_ar,0.84)
        S = t_ar_bkg[t_ar_bkg>tdat].size
        Sinf = t_ar_bkg[t_ar_bkg>tinf].size
        Ssup = t_ar_bkg[t_ar_bkg>tsup].size
        global_Pval = S/Npe
        global_inf = Sinf/Npe
        global_sup = Ssup/Npe
        print('Global p-value : {0:1.4f}  ({1} / {2})   {3:1.4f}  ({4})   {5:1.4f}  ({6})'.format(global_Pval,S,Npe,global_inf,Sinf,global_sup,Ssup))
        
        # If global p-value is exactly 0, we might have trouble with the significance
        if(global_Pval<0.0000000000000001):
            significance = norm.ppf(1-0.0000000000000001)
        else:
            significance = norm.ppf(1-global_Pval)
        
        if(global_inf<0.0000000000000001):
            sigma_inf = norm.ppf(1-0.0000000000000001)
        else:
            sigma_inf = norm.ppf(1-global_inf)
        
        if(global_sup<0.0000000000000001):
            sigma_sup = norm.ppf(1-0.0000000000000001)
        else:
            sigma_sup = norm.ppf(1-global_sup)
        print('Significance = {0:1.5f} ({1:1.5f}  {2:1.5f})'.format(significance,sigma_inf,sigma_sup))
        print('')
        
        # Append reached significance to sigma_ar (with sup and inf variations)
        sigma_ar.append([significance,abs(significance-sigma_inf),abs(significance-sigma_sup)])
    
    # End of injection loop
    print('REACHED SIGMA LIMIT')
    print('   Number of signal event injected : {}'.format(signal_min))
    
    # Compute signal strength
    signal_ratio = signal_min/signal_exp
    print('   Signal strength : {0:1.4f}'.format(signal_ratio))
    print('')
    
    # Save the data obtained after last injection in global variable
    data_inject = data_hist
    
    # Append the last step results to the background results
    t_ar = np.append(t_ar_bkg,t_ar)
    min_Pval_ar = np.append(min_Pval_ar_bkg,min_Pval_ar)
    min_loc_ar = np.append(min_loc_ar_bkg,min_loc_ar)
    min_width_ar = np.append(min_width_ar_bkg,min_width_ar)
    
    # Convert the sigma_ar global variable into a numpy array
    sigma_ar = np.array(sigma_ar)
    
    return


# Function to plot the signal injection result
def PlotInject(filename=None):
    '''
    Function that uses the global parameters str_min and str_step as well as the global results sigma_ar to
    generate a plot.
    
    Argument :
        fliename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                   but not saved. Default to None.
    '''
    
    # Get the x-values (signal strength)
    if(str_scale=='lin'):
        sig_str = np.arange(str_min,str_min+str_step*len(sigma_ar),step=str_step)
    else:
        sig_str = np.array([i%10*10**(str_min+i//10) for i in range(len(sigma_ar)+len(sigma_ar)//10+1) if i%10!=0])
    
    # If filename is not None and log scale must check
    if(filename!=None and str_scale=='log'):
        if(type(filename)==type('str')):
            print('WARNING : log plot for signal injection will not be saved !')
            nolog = True
        else:
            nolog = False
    
    # Do the plot
    F = plt.figure(figsize=(12,8))
    plt.title('Significane vs signal strength')
    plt.errorbar(sig_str,sigma_ar[:,0],
                 xerr=0,yerr=[sigma_ar[:,1],sigma_ar[:,2]],
                 linewidth=2,marker='o')
    plt.xlabel('Signal strength',size='large')
    plt.ylabel('Significance',size='large')
    
    if(filename==None):
        plt.show()
    else:
        if(str_scale=='log' and nolog==False):
            plt.savefig(filename[0],bbox_inches='tight')
        else:
            plt.savefig(filename,bbox_inches='tight')
        plt.close(F)
    
    # If log scale, do also a log plot
    if(str_scale=='log'):
        F = plt.figure(figsize=(12,8))
        plt.title('Significane vs signal strength (log scale)')
        plt.errorbar(sig_str,sigma_ar[:,0],
                     xerr=0,yerr=[sigma_ar[:,1],sigma_ar[:,2]],
                     linewidth=2,marker='o')
        plt.xlabel('Signal strength',size='large')
        plt.ylabel('Significance',size='large')
        plt.xscale('log')
        
        if(filename==None):
            plt.show()
        else:
            if(nolog==False):
                plt.savefig(filename[1],bbox_inches='tight')
            plt.close(F)
    
    return




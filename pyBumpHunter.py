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


# Result global variables
global_Pval = 0
res_ar = []
min_Pval_ar = []
min_loc_ar = []
min_width_ar = []
t_ar = []

# Function that perform the scan on every pseudo experiment and data (in parrallel threads).
# For each scan, the value of p-value and test statistic t is computed and stored in result array
def BumpHunter(data,bkg,Rang=None,
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
        
        # Create the results array
        res = np.empty(Nwidth,dtype=np.object)
        min_Pval,min_loc = np.empty(Nwidth),np.empty(Nwidth)
        
        # Loop over all the width of the window
        for i in range(Nwidth):
            # Initialize window position at first bin
            #loc = 0
            
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
            res[i] = np.empty((ref.size-w+1)//scan_stepp)
            
            # Count events in all windows of width w
            #FIXME any better way to do it ?? Without loop ?? FIXME
            count = [[hist[loc*scan_stepp:loc*scan_stepp+w].sum(),ref[loc*scan_stepp:loc*scan_stepp+w].sum()] for loc in range((ref.size-w+1)//scan_stepp)]
            count = np.array(count)
            Nhist,Nref = count[:,0],count[:,1]
            del count
            
            # Calculate all local p-values for for width w
            res[i][(Nhist==0) & (Nref==0)] = 1.0
            res[i][Nref>Nhist] = 1.0
            res[i][Nref<=Nhist] = G(Nhist[Nref<=Nhist],Nref[Nref<=Nhist])
            
            # Get the minimum p-value and associated position for width w
            min_Pval[i] = res[i].min()
            min_loc[i] = res[i].argmin()*scan_stepp
        
        # Get the minimum p-value and associated windonw among all width
        min_width = width_min + min_Pval.argmin()*width_step
        min_loc = min_loc[min_Pval.argmin()]
        min_Pval = min_Pval[min_Pval.argmin()]
        
        # Save the results in global variables and return
        res_ar[ih] = res
        min_Pval_ar[ih] = min_Pval
        min_loc_ar[ih] = int(min_loc)
        min_width_ar[ih] = int(min_width)
        return 
    
    
    # Check if we must keep the old parameter values or not
    if(keepparam==False):
        # Set global parameter variables
        global rang
        rang = Rang
        global width_min
        width_min = Width_min
        global width_max
        width_max = Width_max
        global width_step
        width_step = Width_step
        global scan_step
        scan_step = Scan_step
        global Npe
        Npe = npe
        global bins
        bins = Bins
        global weights
        weights = Weights
        global Nworker
        Nworker = NWorker
        global seed
        seed = Seed
        
        # Set the seed if required
        if(seed!=None):
            np.random.seed(seed)
        
        # Set global result variables
        global global_Pval
        global res_ar
        global min_Pval_ar
        global min_loc_ar
        global min_width_ar
        global t_ar
    
    
    # Generate the background and data histograms
    print('Generating histograms')
    F = plt.figure()
    bkg_hist = np.histogram(bkg,bins=bins,weights=weights,range=rang)
    data_hist = np.histogram(data,bins=bins,range=rang)[0]
    plt.close(F)
    
    # Save histogram information in more usefull way
    Hbin = bkg_hist[1]
    bkg_hist = bkg_hist[0]
    
    # Generate all the pseudo-data histograms
    pseudo_hist = np.random.poisson(lam=np.tile(bkg_hist,(Npe,1)).transpose(),size=(bkg_hist.size,Npe))
    
    # Set width_max if it is given as None
    if width_max==None:
        width_max = data_hist.size
    
    # Initialize all results containenrs
    min_Pval_ar = np.empty(Npe+1)
    min_loc_ar = np.empty(Npe+1,dtype=int)
    min_width_ar = np.empty(Npe+1,dtype=int)
    res_ar = np.empty(Npe+1,dtype=np.object)
    
    # Auto-adjust the value of width_max and compute Nwidth
    Nwidth,width_max = get_Nscan(width_min,width_max,width_step)
    print('{} values of width will be tested'.format(Nwidth))
    
    # Compute the p-value for data and all pseudo-experiment
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
    
    return


def GetTomography(data,filename=None):
    '''
    Function that do a tomography plot showing the local p-value for every positions and widths of the scan
    window.
    
    Arguments :
        data : Numpy array containing the raw unbined data.
    
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
    
    # Get real bin bounds
    F = plt.figure()
    H = plt.hist(data,bins=bins,range=rang)
    H = H[1]
    plt.close(F)
    
    res_data = res_ar[0]    
    inter = []
    for i in range(res_data.size):
        w = H[width_min+i*width_step]-H[0]
        
        # Get scan step for width w
        if(scan_step=='half'):
            scan_stepp = max(1,(width_min+i*width_step)//2)
        elif(scan_step=='full'):
            scan_stepp = width_min+i*width_step
        else:
            scan_stepp = scan_step
        
        for j in range(len(res_data[i])):
            loc = H[j*scan_stepp]
            inter.append([res_data[i][j],loc,w])
    
    plt.figure(figsize=(12,8))
    for i in inter:
        plt.plot([i[1],i[1]+i[2]],[i[0],i[0]],'r')
    plt.xlabel('intervals',size='large')
    plt.ylabel('local p-value',size='large')
    plt.yscale('log')
    
    if(filename==None):
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')
    print('')
    return


# Function that print the infomation about the most significante bump in data
def PrintBumpInfo():
    '''
    Function that print the infomation about the most significante bump in data. Information are
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

def PrintBumpTrue(data,bkg):
    '''
    Print the informations about the most significante bump in data in real scale.
    
    Argument :
        data : Numpy array containing the raw unbined data.
        bkg : Numpy array containing the raw unbined background.
    '''
    
    # Set the global variables
    global bins
    global rang
    global min_loc_ar
    global min_width_ar
    global global_Pval
    
    # Get the data and background in histogram form
    F = plt.figure()
    H = np.histogram(data,bins=bins,range=rang)
    Hb = np.histogram(bkg,bins=bins,range=rang,weights=weights)[0]
    plt.close(F)
    
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
    print('   width : {0:1.3f}'.format(Bwidth))
    print('   Number of signal events : {}'.format(D-B))
    print('   global p-value : {0:1.5f}'.format(global_Pval))
    
    # If global p-value is exactly 0, we might have trouble with the significance
    if(global_Pval<0.0000000000000001):
        print('   significance = {0:1.5f}'.format(norm.ppf(1-0.0000000000000001)))
    else:
        print('   significance = {0:1.5f}'.format(norm.ppf(1-global_Pval)))

    print('')
    
    return


# Plot the data and bakground histograms with the bump found by BumpHunter highlighted
def PlotBump(data,bkg,filename=None):
    '''
    Plot the data and bakground histograms with the bump found by BumpHunter highlighted.
    
    Arguments :
        data : Numpy array containing the raw unbined data.
        
        bkg : Numpy array containing the raw unbined background.
        
        filename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                   but not saved. Default to None.
    '''
    
    # Set the global variables
    global min_loc_ar
    global min_width_ar
    
    # Get the data in histogram form
    F = plt.figure()
    H = np.histogram(data,bins=bins,range=rang)
    plt.close(F)
    
    # Get bump min and max
    Bmin = H[1][min_loc_ar[0]]
    Bmax = H[1][min_loc_ar[0]+min_width_ar[0]]
    
    # Calculate significance for each bin
    Hbkg = np.histogram(bkg,bins=bins,range=rang)[0]
    sig = np.empty(60)
    sig[(H[0]==0) & (Hbkg==0)]=1.0
    sig[H[0]>=Hbkg]=G(H[0][H[0]>=Hbkg],Hbkg[H[0]>=Hbkg])
    sig[H[0]<Hbkg]=1-G(H[0][H[0]<Hbkg]+1,Hbkg[H[0]<Hbkg])

    sig = norm.ppf(1-sig)
    sig[H[0]<Hbkg]=-sig[H[0]<Hbkg]
    sig[sig==np.inf]=0 # Avoid errors
    sig[sig==np.NaN]=0
    
    # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
    F = plt.figure(figsize=(12,10))
    gs = grd.GridSpec(2, 1, height_ratios=[4, 1])
    
    pl1 = plt.subplot(gs[0])
    plt.title('Distributions with bump')
    plt.hist(bkg,bins=bins,histtype='step',range=rang,weights=weights,label='background')
    plt.hist(data,bins=bins,histtype='step',range=rang,label='data')
    plt.plot(np.full(2,Bmin),np.array([0,H[0][min_loc_ar[0]]]),'r--',label=('BUMP'))
    plt.plot(np.full(2,Bmax),np.array([0,H[0][min_loc_ar[0]+min_width_ar[0]]]),'r--')
    plt.legend()
    plt.yscale('log')
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
    H=plt.hist(t_ar[1:],bins=20,histtype='step',label='pseudo-data')
    plt.plot(np.full(2,t_ar[0]),np.array([0,H[0].max()]),'r--',linewidth=2,label=('data'))
    plt.legend()
    plt.xlabel('t',size='large')
    plt.yscale('log')
    
    # Check if the plot should be saved or just displayed
    if(filename==None):
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')
        plt.close(F)
    
    return


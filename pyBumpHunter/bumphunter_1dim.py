#!/usr/bin/env python
"""Python version of the BupHunter algorithm as described in https://arxiv.org/pdf/1101.0390.pdf"""

import concurrent.futures as thd
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec as grd
from scipy.special import gammainc as G  # Need G(a,b) for the gamma function
from scipy.stats import norm

from .util import deprecated, deprecated_arg


class BumpHunter1D:
    """The BumpHunter class is the object providing all the necessary tools to "bump hunt" with ease.

    It comes with a set of methods to perform scans using the BumpHunter algorithm and to manage all
    the parameters and results stored in the inner variables.

    List of inner parameter variables :
        rang :
            x-axis range of the histograms. Also define the range in which the scan will be performed.
            

        mode :
            String specifying if the algorithm must look for a excess or a deficit in the data.
            Can be either 'excess' or 'deficit'.

        width_min : 
            Minimum value of the scan window width that should be tested (in number of bins).

        width_max :
            Maximum value of the scan window width that should be tested (in number of bins).
            Can be either None or a positive integer.
            If None, the value is set to the total number of bins of the histograms divided by 2.

        width_step :
            Number of bins by which the scan window width is increased at each step.

        scan_step :
            Number of bins by which the position of the scan window is shifted at each step.
            Can be either 'full', 'half' or a positive integer.
            If 'full', the window will be shifted by a number of bins equal to its width.
            If 'half', the window will be shifted by a number of bins equal to max(1,width//2).

        npe :
            Number of pseudo-data distributions to be sampled from the reference background distribution.

        bins :
            Define the bins of the histograms. Can be ether a integer of a array-like of floats.
            If integer (N), N bins of equal width will be considered.
            If array-like of float (a), a number of bins equal to a length-1 with the values of a as edges will be considered (variable width bins allowed).

        weights :
            Weights for the background distribution.
            Can be either None or a array-like of float.
            If array-like of floats, each background events will be accounted by its weights when making histograms.
            The size of the array-like must be the same than of bkg.
            If None, no weights will be considered.

        nworker :
            Number of thread to be run in parallel when scanning all the histograms (data and pseudo-data).
            If less or equal to 1, then parallelism will be disabled.

        seed :
            Seed for the random number generator.

        use_sideBand :
            Boolean specifying if side-band normalization should be applied when computing p-values.

        sigma_limit :
            The minimum significance required after injection.

        str_min :
            The minimum number signal stregth to inject in background (first iteration).

        str_step :
            Increase of the signal stregth to be injected in the background at each iteration.

        str_scale :
            Specify how the signal strength should vary.
            If 'log', the signal strength will vary according to a log scale starting from 10**str_min.
            If 'lin', the signal will vary according to a linear scale starting from str_min with a step of str_step.

        signal_exp :
            Expected number of signal used to compute the signal strength.
            If None, the signal strength is not computed.

        flip_sig :
            Boolean specifying if the signal should be fliped when running in deficit mode.
            Ignored in excess mode.

    List of inner results variables :
        global_Pval :
            Global p-value obtained from the test statistic distribution.

        significance :
            Significance corresponding to the globbal p-value from the test statistic distribution.

        res_ar :
            Array-like container containing all the local p-values calculated during the last BumpHnter scan.
            The indice 0 (res_ar[0]) correspond to the sacn of the data and the other indices correspond to the the pseudo-data.
            For each indices, there is a Numpy array of python list containing all the p-values of all windows obtained for a given distribution.
            The numpy array has dimention (Nwidth), with Nwidth the number of window's width tested.
            Each python list as dimension (Nstep), with Nstep the number of scan step for a given width (different for every value of width).


        min_Pval_ar :
            Array containing the minimum p-values obtained for the data (indice=0) and and the pseudo-data (indice>0).

        min_loc_ar : 
            Array containing the positions of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

        min_width_ar :
            Array containing the width of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

        signal_eval :
            Number of signal events evaluated form the last scan.

        signal_min :
            Minimum number of signal events ones must inject in the data in order to reach the required significance (might differ from signal_eval).

        signal_ratio :
            Ratio signal_min/signal_exp (signal strength).

        data_inject :
            Data obtained after injecting signal events in the backgound.

        sigma_ar :
            Numpy array containing the significance values obtained at each step.
    """

    # Initializer method
    @deprecated_arg("useSideBand", "use_sideband")
    @deprecated_arg("Nworker", "nworker")
    @deprecated_arg("Npe", "npe")
    def __init__(
        self,
        rang=None,
        mode: str="excess",
        width_min: int=1,
        width_max=None,
        width_step: int=1,
        scan_step: int=1,
        npe: int=100,
        bins: int=60,
        weights=None,
        nworker: int=4,
        sigma_limit: float=5,
        str_min: float=0.5,
        str_step: float=0.25,
        str_scale: str="lin",
        signal_exp=None,
        flip_sig: bool=True,
        seed=None,
        use_sideband: bool=False,
        Nworker=None,
        useSideBand=None,
        Npe=None,
    ):
        """
        Arguments:
            rang :
                x-axis range of the histograms. Also define the range in which the scan will be performed.
                Can be either None or a array-like of float with shape (2,2).
                If None, the range is set automatically to include all the data given.
                Default to None.
            
            mode :
                String specifying if the algorithm must look for a excess or a deficit in the data.
                Can be either 'excess' or 'deficit'.
                Default to 'excess'.
            
            width_min :
                Minimum value of the scan window width that should be tested (in number of bins).
                Default to 1.
            
            width_max :
                Maximum value of the scan window width that should be tested (in number of bins).
                Can be either None or a positive integer.
                If None, the value is set to the total number of bins of the histograms divided by 2.
                Default to none.
            
            width_step :
                Number of bins by which the scan window width is increased at each step.
                Default to 1.
            
            scan_step :
                Number of bins by which the position of the scan window is shifted at each step.
                Can be either 'full', 'half' or a positive integer.
                If 'full', the window will be shifted by a number of bins equal to its width.
                If 'half', the window will be shifted by a number of bins equal to max(1,width//2).
                Default to 1.
            
            npe :
                Number of pseudo-data distributions to be sampled from the reference background distribution.
                Default to 100.

            bins :
                Define the bins of the histograms. Can be ether a integer of a array-like of floats.
                If integer (N), N bins of equal width will be considered.
                If array-like of float (a), a number of bins equal to a length-1 with the values of a as edges will be considered (variable width bins allowed).
                Default to 60.
    
            weights :
                Weights for the background distribution. Can be either None or a array-like of float.
                If array-like of floats, each background events will be accounted by its weights when making histograms.
                The size of the array-like must be the same than of bkg.
                If None, no weights will be considered.
                Default to None.
    
            nworker : 
                Number of thread to be run in parallel when scanning all the histograms (data and pseudo-data).
                If less or equal to 1, then parallelism will be disabled.
                Default to 4.

            sigma_limit :
                The minimum significance required after injection.
                Deault to 5.

            str_min :
                The minimum number signal stregth to inject in background (first iteration).
                Default to 0.5.
    
            str_step :
                Increase of the signal stregth to be injected in the background at each iteration.
                Default to 0.25.
    
            str_scale :
                Specify how the signal strength should vary.
                If 'log', the signal strength will vary according to a log scale starting from 10**str_min.
                If 'lin', the signal will vary according to a linear scale starting from str_min with a step of str_step.
                Default to 'lin'.
    
            signal_exp :
                Expected number of signal used to compute the signal strength.
                If None, the signal strength is not computed. Default to None.
    
            flip_sig :
                Boolean specifying if the signal should be fliped when running in deficit mode.
                Ignored in excess mode. Default to True.
                
            seed :
                Seed for the random number generator.
                Default to None. 
            
            use_sideband :
                Boolean specifying if the side-band normalization should be applied.
                Default to False.
            
            Npe : *Deprecated*
                Same as npe. This argument is deprecated and will be removed in future versions.
            
            Nworker : *Deprecated*
                Same as Nworker. This argument is deprecated and will be removed in future versions.
            
            useSideBand : *Deprecated*
                Same as useSideBand. This argument is deprecated and will be removed in future versions.
        """
        # legacy deprecation
        if useSideBand is not None:
            use_sideband = useSideBand
        if Nworker is not None:
            nworker = Nworker
        if Npe is not None:
            npe = Npe

        # Initilize all inner parameter variables
        self.rang = rang
        self.mode = mode
        self.width_min = width_min
        self.width_max = width_max
        self.width_step = width_step
        self.scan_step = scan_step
        self.npe = npe
        self.bins = bins
        self.weights = weights
        self.nworker = nworker
        self.sigma_limit = sigma_limit
        self.str_min = str_min
        self.str_step = str_step
        self.str_scale = str_scale
        self.signal_exp = signal_exp
        self.flip_sig = flip_sig
        self.seed = seed
        self.use_sideband = use_sideband

        # Initialize all inner result variables
        self.reset()

    # Private methods

    # Method that performs a scan of a given data histogram and compares it to a reference background histogram.
    # This method is used by the BumpHunter class methods and is not intended to be used directly.
    def _scan_hist(self, hist, ref, w_ar, ih: int):
        """Scan a distribution and compute the p-value associated to every scan window.

        The algorithm follows the BumpHunter algorithm. Compute also the significance for the data histogram.

        Arguments :
            hist :
                The data histogram (as obtain with the numpy.histogram function).

            ref :
                The reference (background) histogram (as obtain with the numpy.histogram function).

            w_ar :
                Array containing all the values of width to be tested.

            ih :
                Indice of the distribution to be scanned.
                ih=0 refers to the data distribution and ih>0 refers to the ih-th pseudo-data distribution.

        Results stored in inner variables :
            res :
                Numpy array of python list containing all the p-values of all windows computed durring the scan.
                The numpy array as dimention (Nwidth), with Nwidth the number of window's width tested.
                Each python list as dimension (Nstep), with Nstep the number of scan step for a given width (different for every value of width).

            min_Pval :
                Minimum p_value obtained durring the scan (float).

            min_loc :
                Position of the window corresponding to the minimum p-value (integer).

            min_width :
                Width of the window corresponding to the minimum p-value (integer).
        """

        # Remove the first/last hist bins if empty ... just to be consistant with c++
        non0 = [iii for iii in range(hist.size) if hist[iii] > 0]
        Hinf, Hsup = min(non0), max(non0) + 1

        # Create the results array
        res = np.empty(w_ar.size, dtype=object)
        min_Pval, min_loc = np.empty(w_ar.size), np.empty(w_ar.size, dtype=int)
        signal_eval = np.empty(w_ar.size)

        if self.use_sideband:
            ref_total = ref[Hinf:Hsup].sum()
            hist_total = hist[Hinf:Hsup].sum()

        # Loop over all the width of the window
        for i, w in enumerate(w_ar):
            # Auto-adjust scan step if specified
            if self.scan_step == "full":
                scan_stepp = w
            elif self.scan_step == "half":
                scan_stepp = max(1, w // 2)
            else:
                scan_stepp = self.scan_step

            # Define possition range
            pos = np.arange(Hinf, Hsup - w + 1, scan_stepp)

            # Check that there is at least one interval to check for width w
            # If not, we must set dummy values in order to avoid crashes
            if pos.size == 0:
                res[i] = np.array([1.0])
                min_Pval[i] = 1.0
                min_loc[i] = 0
                signal_eval[i] = 0
                continue

            # Initialize local p-value array for width w
            res[i] = np.ones(pos.size)

            # Count events in all windows of width w
            # FIXME any better way to do it ?? Without loop ?? FIXME
            Nref = np.array([ref[p : p + w].sum() for p in pos])
            Nhist = np.array([hist[p : p + w].sum() for p in pos])

            if self.use_sideband:
                Nref *= (hist_total - Nhist) / (ref_total - Nref)

            # Calculate all local p-values for for width w
            if self.mode == "excess":
                res[i][(Nhist > Nref) & (Nref > 0)] = G(
                    Nhist[(Nhist > Nref) & (Nref > 0)],
                    Nref[(Nhist > Nref) & (Nref > 0)],
                )
            elif self.mode == "deficit":
                res[i][Nhist < Nref] = 1.0 - G(
                    Nhist[Nhist < Nref] + 1, Nref[Nhist < Nref]
                )

            if self.use_sideband:
                res[i][
                    res[i] < 1e-300
                ] = 1e-300  # prevent issue with very low p-value, sometimes induced by normalisation in the tail

            # Get the minimum p-value and associated position for width w
            min_Pval[i] = res[i].min()
            min_loc[i] = pos[res[i].argmin()]
            signal_eval[i] = Nhist[min_loc[i]] - Nref[min_loc[i]]

        # Get the minimum p-value and associated window among all width
        min_width = w_ar[min_Pval.argmin()]
        min_loc = min_loc[min_Pval.argmin()]

        # Evaluate the number of signal event (for data only)
        if ih == 0:
            self.signal_eval = signal_eval[min_Pval.argmin()]

        min_Pval = min_Pval.min()

        # Save the results in inner variables and return
        self.res_ar[ih] = res
        self.min_Pval_ar[ih] = min_Pval
        self.min_loc_ar[ih] = int(min_loc)
        self.min_width_ar[ih] = int(min_width)

    # Variable management methods

    # Reset method
    def reset(self):
        """
        Reset all the inner result parameter for this BumpHunter instance.
        Use with caution.
        """
        self.global_Pval = 0
        self.significance = 0
        self.res_ar = []
        self.min_Pval_ar = []
        self.min_loc_ar = []
        self.min_width_ar = []
        self.t_ar = []
        self.signal_eval = 0
        self.signal_min = 0
        self.signal_ratio = None
        self.data_inject = []

        return

    @deprecated("Use `reset` instead.")
    def Reset(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    # Export/import parameters/results
    def save_state(self):
        """
        Save the current state (all parameters and results) of a BupHunter instance into a dict variable.

        Ruturns:
            state :
                The dict containing all the parameters and results of this BumpHunter instance.
                The keys of the dict entries correspond the name of their associated parameters/results as defined in the BumpHunter class.
        """
        state = dict()

        # Save parameters
        state["mode"] = self.mode
        state["rang"] = self.rang
        state["bins"] = self.bins
        state["weights"] = self.weights
        state["width_min"] = self.width_min
        state["width_max"] = self.width_max
        state["width_step"] = self.width_step
        state["scan_step"] = self.scan_step
        state["npe"] = self.npe
        state["nworker"] = self.nworker
        state["seed"] = self.seed
        state["sigma_limit"] = self.sigma_limit
        state["str_min"] = self.str_min
        state["str_step"] = self.str_step
        state["str_scale"] = self.str_scale
        state["signal_exp"] = self.signal_exp
        state["sig_flip"] = self.flip_sig
        state["use_sideband"] = self.use_sideband

        # Save results
        state["global_Pval"] = self.global_Pval
        state["significance"] = self.significance
        state["res_ar"] = self.res_ar
        state["min_Pval_ar"] = self.min_Pval_ar
        state["min_loc_ar"] = self.min_loc_ar
        state["min_width_ar"] = self.min_width_ar
        state["t_ar"] = self.t_ar
        state["signal_eval"] = self.signal_eval
        state["signal_min"] = self.signal_min
        state["signal_ratio"] = self.signal_ratio
        state["data_inject"] = self.data_inject

        return state

    @deprecated("Use `save_state` instead.")
    def SaveState(self, *args, **kwargs):
        return self.save_state(*args, **kwargs)

    def load_state(self, state: dict):
        """
        Load all the parameters and results of a previous BumpHunter intance that were saved using the SaveState method.

        Arguments :
            state :
                A dict containing all the parameters/results of a previous BumpHunter instance.
                If a parameter or a result field is missing, it will be set to its default value.
        """

        # Load parameters

        self.mode = state.get("mode", "excess")

        if "rang" in state.keys():
            self.rang = state["rang"]
        else:
            self.rang = None

        if "bins" in state.keys():
            self.bins = state["bins"]
        else:
            self.bins = 60

        if "weights" in state.keys():
            self.rang = state["weights"]
        else:
            self.rang = None

        if "width_min" in state.keys():
            self.width_min = state["width_min"]
        else:
            self.width_min = 2

        if "width_max" in state.keys():
            self.width_max = state["width_max"]
        else:
            self.width_max = None

        if "width_step" in state.keys():
            self.width_step = state["width_step"]
        else:
            self.width_step = 1

        if "scan_step" in state.keys():
            self.scan_step = state["scan_step"]
        else:
            self.scan_step = 1

        if "npe" in state.keys():
            self.npe = state["npe"]
        else:
            self.npe = 100

        if "nworker" in state.keys():
            self.nworker = state["nworker"]
        else:
            self.nworker = 4

        if "seed" in state.keys():
            self.seed = state["seed"]
        else:
            self.seed = None

        if "use_sideband" in state.keys():
            self.use_sideband = state["use_sideband"]
        else:
            self.use_sideband = False

        if "sigma_limit" in state.keys():
            self.sigma_limit = state["sigma_limit"]
        else:
            self.sigma_limit = 5

        if "str_min" in state.keys():
            self.str_min = state["str_min"]
        else:
            self.str_min = 0.5

        if "str_step" in state.keys():
            self.str_step = state["str_step"]
        else:
            self.str_step = 0.25

        if "str_scale" in state.keys():
            self.str_scale = state["str_scale"]
        else:
            self.str_scale = "lin"

        if "signal_exp" in state.keys():
            self.signal_exp = state["signal_exp"]
        else:
            self.signal_exp = None

        if "sig_flip" in state.keys():
            self.sig_flip = state["sig_flip"]
        else:
            self.sig_flip = True

        # Load results
        self.reset()
        if "global_Pval" in state.keys():
            self.global_Pval = state["global_Pval"]
        if "significance" in state.keys():
            self.significance = state["significance"]
        if "res_ar" in state.keys():
            self.res_ar = state["res_ar"]
        if "min_Pval_ar" in state.keys():
            self.min_Pval_ar = state["min_Pval_ar"]
        if "min_loc_ar" in state.keys():
            self.min_loc_ar = state["min_loc_ar"]
        if "min_width_ar" in state.keys():
            self.min_width_ar = state["min_width_ar"]
        if "t_ar" in state.keys():
            self.t_ar = state["t_ar"]
        if "signal_eval" in state.keys():
            self.signal_eval = state["signal_eval"]
        if "signal_min" in state.keys():
            self.signal_min = state["signal_min"]
        if "signal_ratio" in state.keys():
            self.signal_ratio = state["signal_ratio"]
        if "data_inject" in state.keys():
            self.data_inject = state["data_inject"]

        return

    @deprecated("Use `load_state` instead.")
    def LoadState(self, *args, **kwargs):
        return self.load_state(*args, **kwargs)

    ## Scan methods

    # Method that perform the scan on every pseudo experiment and data (in parrallel threads).
    # For each scan, the value of p-value and test statistic t is computed and stored in result array
    def bump_scan(self, data, bkg, is_hist: bool=False, do_pseudo: bool=True):
        """
        Function that perform the full BumpHunter algorithm presented in https://arxiv.org/pdf/1101.0390.pdf without sidebands.
        This includes the generation of pseudo-data, the calculation of the BumpHunter p-value associated to data and to all pseudo experiment as well as the calculation of the test satistic t.

        The results are stored in the inner result variables of this BumpHunter instance.

        Arguments :
            data :
                Numpy array containing the data distribution.
                This distribution will be transformed into a binned histogram and the algorithm will look for the most significant excess.

            bkg :
                Numpy array containing the background reference distribution.
                This distribution will be transformed into a binned histogram and the algorithm will compare it to data while looking for a bump.

            is_hist :
                Boolean that specify if the given data and background are already in histogram form.
                If true, the data and backgrouns are considered as already 'histogramed'.
                Default to False.

            do_pseudo :
                Boolean specifying if pesudo data should be generated.
                If False, then the BumpHunter statistics distribution kept in memory is used to compute the global p-value and significance.
                If there is nothing in memory, the global p-value and significance will not be computed.
                Default to True.

        Result inner variables :
            global_Pval :
                Global p-value obtained from the test statistic distribution.

            res_ar :
                Array of containers containing all the p-value calculated durring the scan of the data (indice=0) and of the pseudo-data (indice>0).
                For more detail about how the p-values are sorted in the containers, please reffer the the doc of the function _scan_hist.

            min_Pval_ar :
                Array containing the minimum p-values obtained for the data (indice=0) and and the pseudo-data (indice>0).

            min_loc_ar :
                Array containing the positions of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

            min_width_ar :
                Array containing the width of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

            signal_eval :
                Number of signal events evaluated form the last scan.
        """

        # Set the seed if required (or reset it if None)
        np.random.seed(self.seed)

        # Generate the background and data histograms
        print("Generating histograms")
        if not is_hist:
            bkg_hist, Hbin = np.histogram(
                bkg, bins=self.bins, weights=self.weights, range=self.rang
            )
            data_hist = np.histogram(data, bins=self.bins, range=self.rang)[0]
        else:
            if self.weights is None:
                bkg_hist = bkg
            else:
                bkg_hist = bkg * self.weights
            data_hist = data
            Hbins = self.bins

        # Generate all the pseudo-data histograms
        if do_pseudo:
            pseudo_hist = np.random.poisson(
                lam=np.tile(bkg_hist, (self.npe, 1)).transpose(),
                size=(bkg_hist.size, self.npe),
            )

        # Set width_max if it is given as None
        if self.width_max is None:
            self.width_max = data_hist.size // 2

        # Initialize all results containenrs
        if do_pseudo:
            self.min_Pval_ar = np.empty(self.npe + 1)
            self.min_loc_ar = np.empty(self.npe + 1, dtype=int)
            self.min_width_ar = np.empty(self.npe + 1, dtype=int)
            self.res_ar = np.empty(self.npe + 1, dtype=object)
        else:
            if self.res_ar == []:
                self.min_Pval_ar = np.empty(1)
                self.min_loc_ar = np.empty(1, dtype=int)
                self.min_width_ar = np.empty(1, dtype=int)
                self.res_ar = np.empty(1, dtype=object)

        # Auto-adjust the value of width_max and do an array of all width
        w_ar = np.arange(self.width_min, self.width_max + 1, self.width_step)
        width_max = w_ar[-1]
        print(f"{w_ar.size} values of width will be tested")

        # Compute the p-value for data and all pseudo-experiments
        # We must check if we should do it in multiple threads
        print("SCAN")
        if do_pseudo:
            if self.nworker > 1:
                with thd.ThreadPoolExecutor(max_workers=self.nworker) as exe:
                    for th in range(self.nnpe + 1):
                        if th == 0:
                            exe.submit(self._scan_hist, data_hist, bkg_hist, w_ar, th)
                        else:
                            exe.submit(
                                self._scan_hist,
                                pseudo_hist[:, th - 1],
                                bkg_hist,
                                w_ar,
                                th,
                            )
            else:
                for i in range(self.npe + 1):
                    if i == 0:
                        self._scan_hist(data_hist, bkg_hist, w_ar, i)
                    else:
                        self._scan_hist(pseudo_hist[:, i - 1], bkg_hist, w_ar, i)
        else:
            self._scan_hist(data_hist, bkg_hist, w_ar, 0)

        # Use the p-value results to compute t
        self.t_ar = -np.log(self.min_Pval_ar)

        # Compute the global p-value from the t distribution
        tdat = self.t_ar[0]
        S = self.t_ar[1:][self.t_ar[1:] > tdat].size
        self.global_Pval = S / self.npe
        print(f"Global p-value : {self.global_Pval:1.4f}  ({S} / {self.npe})")

        # If global p-value is exactly 0, we might have trouble with the significance
        if self.global_Pval < 1e-15:
            self.significance = norm.ppf(1 - 1e-15)
        else:
            self.significance = norm.ppf(1 - self.global_Pval)
        print(f"Significance = {self.significance:1.5f}")
        print("")

        return

    @deprecated("Use `bump_scan` instead.")
    def BumpScacn(self, *args, **kwargs):
        return self.bump_scan(*args, **kwargs)

    # Perform signal injection on background and determine the minimum aount of signal required for observation
    def signal_inject(self, sig, bkg, is_hist: bool=False):
        """
        Function that perform a signal injection test in order to determine the minimum signal strength required to reach a target significance.
        This function use the BumpHunter algorithm in order to calculate the reached significance for a given signal strength.

        This method share most of its parameters with the BumpScan method.

        Arguments :
            sig :
                Numpy array containing the simulated signal. This distribution will be used to perform the signal injection.

            bkg :
                Numpy array containing the expected background. This distribution will be used to build the data in which signal will be injected.

            is_hist :
                Boolean that specify if the given data and background are already in histogram form.
                If true, the data and backgrouns are considered as already 'histogramed'.
                Default to False.

        Result inner variables :
            signal_ratio :
                Ratio signal_min/signal_exp (signal strength).
                If signal_exp is not specified, default to None.

            data_inject :
                Data obtained after injecting signal events in the backgound.

            sigma_ar :
                Numpy array containing the significance values obtained at each step.

        All the result inner variables of the BumpHunter instance will be filled with the results of the scan permormed
        during the last iteration (when sigma_limit is reached).
        """

        # Set the seed if required (or reset it if None)
        np.random.seed(self.seed)

        # Internal variables
        i = 1
        strength = 0
        data = []

        # Reset significance and sigma_ar global variable
        self.significance = 0
        sigma_inf = 0
        sigma_sup = 0
        self.sigma_ar = []

        # Check the expected number of signal event
        if self.signal_exp is None:
            if not is_hist:
                self.signal_exp = sig.size
            else:
                self.signal_exp = sig.sum()

        # Turn the background distributions into histogram
        if not is_hist:
            bkg_hist, bins = np.histogram(
                bkg, bins=self.bins, range=self.rang, weights=self.weights
            )
        else:
            if self.weights is None:
                bkg_hist = bkg
            else:
                bkg_hist = bkg * self.weights
            bins = self.bins

        # Generate pseudo-data by sampling background
        print("Generating background only histograms")
        Nbkg = 1000
        np.random.seed(self.seed)
        pseudo_bkg = np.random.poisson(
            lam=np.tile(bkg_hist, (Nbkg, 1)).transpose(), size=(bkg_hist.size, Nbkg)
        )

        # Set width_max if it is given as None
        if self.width_max is None:
            self.width_max = bkg_hist.size // 2

        # Initialize all results containenrs
        self.min_Pval_ar = np.empty(Nbkg)
        self.min_loc_ar = np.empty(Nbkg, dtype=int)
        self.min_width_ar = np.empty(Nbkg, dtype=int)
        self.res_ar = np.empty(Nbkg, dtype=object)

        # Auto-adjust the value of width_max and do an array of all width
        w_ar = np.arange(self.width_min, self.width_max + 1, self.width_step)
        self.width_max = w_ar[-1]
        print(f"{w_ar.size} values of width will be tested")

        # Compute the p-value for background only pseudo-experiments
        # We must check if we should do it in multiple threads
        print("BACKGROUND ONLY SCAN")
        if self.nworker > 1:
            with thd.ThreadPoolExecutor(max_workers=self.nworker) as exe:
                for th in range(Nbkg):
                    exe.submit(self._scan_hist, pseudo_bkg[:, th], bkg_hist, w_ar, th)
        else:
            for th in range(Nbkg):
                self._scan_hist(pseudo_bkg[:, th], bkg_hist, w_ar, th)

        # Use the p-value results to compute t
        t_ar_bkg = -np.log(self.min_Pval_ar)

        # Save background result separately and free some memory
        min_Pval_ar_bkg = self.min_Pval_ar
        self.min_Pval_ar = []
        min_loc_ar_bkg = self.min_loc_ar
        self.min_loc_ar = []
        min_width_ar_bkg = self.min_width_ar
        self.min_width_ar = []
        self.res_ar = []

        # Main injection loop
        print("STARTING INJECTION")
        while self.significance < self.sigma_limit:
            # Check how we should compute the signal strength to be injected
            if self.str_scale == "lin":
                # Signal strength increase linearly at each step
                if i == 1:
                    strength = self.str_min
                else:
                    strength += self.str_step
                print(f"   STEP {i} : signal strength = {strength}")

                # Update signal_min
                self.signal_min = self.signal_exp * strength
                i += 1

            elif self.str_scale == "log":
                # Signal strength increase to form a logarithmic scale axis
                if i == 1:
                    strength = 10 ** self.str_min
                    self.str_step = strength
                else:
                    strength += self.str_step
                    if abs(strength - 10 * self.str_step) < 1e-6:
                        self.str_step *= 10
                print(f"   STEP {i} : signal strength = {strength}")

                # Update signal_min
                self.signal_min = self.signal_exp * strength
                i += 1

            else:
                # If bad str_scale value, print a error mesage and abort
                print("ERROR : Bad str_scale value ! Must be either 'lin' or 'log'")
                return

            # Check if we inject a deficit
            if self.mode == "deficit":
                self.signal_min = -self.signal_min

            # Check if the signal is alredy in histogram form or not
            if not is_hist:
                sig_hist = np.histogram(sig, bins=self.bins, range=self.rang)[0]
                sig_hist = sig_hist * strength * (self.signal_exp / sig.size)
            else:
                sig_hist = sig
                sig_hist = sig_hist * strength * (self.signal_exp / sig.sum())

            # Check if sig_hist should be fliped in deficit mode
            if self.mode == "deficit":
                if self.flip_sig:
                    sig_hist = -sig_hist

            # Inject the signal and do some poissonian fluctuation
            print("Generating background+signal histograms")
            data_hist = bkg_hist + sig_hist
            pseudo_data = np.random.poisson(
                lam=np.tile(data_hist, (self.npe, 1)).transpose(),
                size=(data_hist.size, self.npe),
            )

            # Initialize all results containenrs
            self.min_Pval_ar = np.empty(self.npe)
            self.min_loc_ar = np.empty(self.npe, dtype=int)
            self.min_width_ar = np.empty(self.npe, dtype=int)
            self.res_ar = np.empty(self.npe, dtype=object)

            # Compute the p-value for background+signal pseudo-experiments
            # We must check if we should do it in multiple threads
            print("BACKGROUND+SIGNAL SCAN")
            if self.nworker > 1:
                with thd.ThreadPoolExecutor(max_workers=self.nworker) as exe:
                    for th in range(self.npe):
                        exe.submit(
                            self._scan_hist, pseudo_data[:, th], bkg_hist, w_ar, th
                        )
            else:
                for th in range(self.npe):
                    self._scan_hist(pseudo_data[:, th], bkg_hist, w_ar, th)

            # Use the p-value results to compute t
            self.t_ar = -np.log(self.min_Pval_ar)

            # Compute the global p-value from the t distribution with inf end sup values
            tdat, tinf, tsup = (
                np.median(self.t_ar),
                np.quantile(self.t_ar, 0.16),
                np.quantile(self.t_ar, 0.84),
            )
            S = t_ar_bkg[t_ar_bkg > tdat].size
            Sinf = t_ar_bkg[t_ar_bkg > tinf].size
            Ssup = t_ar_bkg[t_ar_bkg > tsup].size
            self.global_Pval = S / self.npe
            global_inf = Sinf / self.npe
            global_sup = Ssup / self.npe
            print(
                f"Global p-value : {self.global_Pval:1.4f}  ({S} / {self.npe})   {global_inf:1.4f}  ({Sinf})   {global_sup:1.4f}  ({Ssup})"
            )

            # If global p-value is exactly 0, we might have trouble with the significance
            if self.global_Pval < 1e-15:
                self.significance = norm.ppf(1 - 1e-15)
            else:
                self.significance = norm.ppf(1 - self.global_Pval)

            if global_inf < 1e-15:
                sigma_inf = norm.ppf(1 - 1e-15)
            else:
                sigma_inf = norm.ppf(1 - global_inf)

            if global_sup < 1e-15:
                sigma_sup = norm.ppf(1 - 1e-15)
            else:
                sigma_sup = norm.ppf(1 - global_sup)
            print(
                f"Significance = {self.significance:1.5f} ({sigma_inf:1.5f}  {sigma_sup:1.5f})"
            )
            print("")

            # Append reached significance to sigma_ar (with sup and inf variations)
            self.sigma_ar.append(
                [
                    self.significance,
                    abs(self.significance - sigma_inf),
                    abs(self.significance - sigma_sup),
                ]
            )

        # End of injection loop
        print("REACHED SIGMA LIMIT")
        print(f"   Number of signal event injected : {self.signal_min}")

        # Compute signal strength
        self.signal_ratio = abs(self.signal_min / self.signal_exp)
        print(f"   Signal strength : {self.signal_ratio:1.4f}")
        print("")

        # Save the data obtained after last injection in inner variables
        self.data_inject = data_hist

        # Append the last step results to the background results
        self.t_ar = np.append(t_ar_bkg, self.t_ar)
        self.min_Pval_ar = np.append(min_Pval_ar_bkg, self.min_Pval_ar)
        self.min_loc_ar = np.append(min_loc_ar_bkg, self.min_loc_ar)
        self.min_width_ar = np.append(min_width_ar_bkg, self.min_width_ar)

        # Convert the sigma_ar inner variable into a numpy array
        self.sigma_ar = np.array(self.sigma_ar)

        return

    @deprecated("Use `signal_inject` instead.")
    def SignalInject(self, *args, **kwargs):
        return self.signal_inject(*args, **kwargs)

    ## Display methods

    # Method that do the tomography plot for the data
    def plot_tomography(self, data, is_hist: bool=False, filename=None):
        """
        Function that do a tomography plot showing the local p-value for every positions and widths of the scan
        window.

        Arguments :
            data : Numpy array containing the data.

            is_hist : Boolean specifying if data is in histogram form or not. Default to False.

            filename : Name of the file in which the plot will be saved. If None, the plot will be just shown
                       but not saved. Default to None.
        """

        # Same c++ compatibility thing
        non0 = [i for i in range(data.size) if data[i] > 0]
        Hinf = min(non0)

        # Get real bin bounds
        if not is_hist:
            H = np.histogram(data, bins=self.bins, range=self.rang)[1]
        else:
            H = self.bins

        res_data = self.res_ar[0]
        inter = []
        for i in range(res_data.size):
            w = (H[1] - H[0]) * (
                self.width_min + i * self.width_step
            )  # bin_width * Nbins

            # Get scan step for width w
            if self.scan_step == "half":
                scan_stepp = max(1, (self.width_min + i * self.width_step) // 2)
            elif self.scan_step == "full":
                scan_stepp = self.width_min + i * self.width_step
            else:
                scan_stepp = self.scan_step

            for j in range(len(res_data[i])):
                loc = H[j * scan_stepp + Hinf]
                inter.append([res_data[i][j], loc, w])

        F = plt.figure(figsize=(12, 8))
        [plt.plot([i[1], i[1] + i[2]], [i[0], i[0]], "r") for i in inter if i[0] < 1.0]
        plt.xlabel("intervals", size="large")
        plt.ylabel("local p-value", size="large")
        plt.yscale("log")
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
            plt.close(F)
        return

    @deprecated("Use `plot_tomography` instead.")
    def GetTomography(self, *args, **kwargs):
        return self.plot_tomography(*args, **kwargs)

    # Plot the data and bakground histograms with the bump found by BumpHunter highlighted
    @deprecated_arg("useSideBand", "use_sideband")
    def plot_bump(self, data, bkg, is_hist: bool=False, use_sideband=None, filename=None, useSideBand=None):
        """
        Plot the data and bakground histograms with the bump found by BumpHunter highlighted.

        Arguments :
            data :
                Numpy array containing the data.

            bkg :
                Numpy array containing the background.

            is_hist : 
                Boolean specifying if data and bkg are given in histogram form or not.
                Default to False.

            use_sideband :
                Boolean specifying if side-band normalization should be used to correct the reference background in the plot.
                If None, self.use_sideband is used instead.
                Default to None.

            filename :
                Name of the file in which the plot will be saved.
                If None, the plot will be just shown but not saved.
                Default to None.

            useSideBand : *Deprecated*
                Same as use_sideband. This argument is deprecated and will be removed in a future version.
        """

        # legacy deprecation
        if useSideBand is not None:
            use_sideband = useSideBand

        # Get the data in histogram form
        if not is_hist:
            H = np.histogram(data, bins=self.bins, range=self.rang)
        else:
            H = [data, self.bins]

        # Get bump min and max
        Bmin = H[1][self.min_loc_ar[0]]
        Bmax = H[1][self.min_loc_ar[0] + self.min_width_ar[0]]

        # Get the background in histogram form
        if not is_hist:
            Hbkg = np.histogram(
                bkg, bins=self.bins, range=self.rang, weights=self.weights
            )[0]
        else:
            if self.weights is None:
                Hbkg = bkg
            else:
                Hbkg = bkg * self.weights

        # Chek if we should apply sideband normalization correction
        if use_sideband is None:
            use_sideband = self.use_sideband

        if use_sideband:
            scale = (
                H[0].sum()
                - H[0][
                    self.min_loc_ar[0] : self.min_loc_ar[0] + self.min_width_ar[0]
                ].sum()
            )
            scale = scale / (
                Hbkg.sum()
                - Hbkg[
                    self.min_loc_ar[0] : self.min_loc_ar[0] + self.min_width_ar[0]
                ].sum()
            )
            Hbkg = Hbkg * scale

        # Calculate significance for each bin
        sig = np.ones(Hbkg.size)
        sig[(H[0] > Hbkg) & (Hbkg > 0)] = G(
            H[0][(H[0] > Hbkg) & (Hbkg > 0)], Hbkg[(H[0] > Hbkg) & (Hbkg > 0)]
        )
        sig[H[0] < Hbkg] = 1 - G(H[0][H[0] < Hbkg] + 1, Hbkg[H[0] < Hbkg])
        sig = norm.ppf(1 - sig)
        sig[sig < 0.0] = 0.0  # If negative, set it to 0
        np.nan_to_num(sig, posinf=0, neginf=0, nan=0, copy=False)  # Avoid errors
        sig[H[0] < Hbkg] = -sig[H[0] < Hbkg]  # Now we can make it signed

        # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
        F = plt.figure(figsize=(12, 10))
        gs = grd.GridSpec(2, 1, height_ratios=[4, 1])

        pl1 = plt.subplot(gs[0])
        plt.title("Distributions with bump")

        plt.hist(
            H[1][:-1],
            bins=H[1],
            histtype="step",
            range=self.rang,
            weights=Hbkg,
            label="background",
            linewidth=2,
            color="red",
        )
        plt.errorbar(
            0.5 * (H[1][1:] + H[1][:-1]),
            H[0],
            xerr=(H[1][1] - H[1][0]) / 2,
            yerr=np.sqrt(H[0]),
            ls="",
            color="blue",
            label="data",
        )

        plt.plot(
            np.full(2, Bmin),
            np.array([0, H[0][self.min_loc_ar[0]]]),
            "r--",
            label=("BUMP"),
        )
        plt.plot(
            np.full(2, Bmax),
            np.array([0, H[0][self.min_loc_ar[0] + self.min_width_ar[0] - 1]]),
            "r--",
        )
        plt.legend(fontsize="large")
        plt.yscale("log")
        if self.rang is not None:
            plt.xlim(self.rang)
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")
        plt.tight_layout()

        plt.subplot(gs[1], sharex=pl1)
        plt.hist(H[1][:-1], bins=H[1], range=self.rang, weights=sig)
        plt.plot(np.full(2, Bmin), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
        plt.plot(np.full(2, Bmax), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
        plt.yticks(
            np.arange(np.round(sig.min()), np.round(sig.max()) + 1, step=1),
            fontsize="large",
        )
        plt.ylabel("significance", size="large")
        plt.xticks(fontsize="large")

        # Check if the plot should be saved or just displayed
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
            plt.close(F)

        return

    @deprecated("Use `plot_bump` instead.")
    def PlotBump(self, *args, **kwargs):
        return self.plot_bump(*args, **kwargs)

    # Plot the Bumpunter test statistic distribution with the result for data
    def plot_stat(self, show_Pval: bool=False, filename=None):
        """
        Plot the Bumphunter statistic distribution together with the observed value with the data.

        Arguments :
            show_Pval :
                Boolean specifying if you want the value of global p-value printed on the plot.
                Default to False.

            filename :
                Name of the file in which the plot will be saved.
                If None, the plot will be just shown but not saved.
                Default to None.
        """

        # Plot the BumpHunter statistic distribution
        F = plt.figure(figsize=(12, 8))
        if show_Pval:
            plt.title(
                f"BumpHunter statistics distribution      global p-value = {self.global_Pval:1.4f}"
            )
        else:
            plt.title("BumpHunter statistics distribution")
        H = plt.hist(
            self.t_ar[1:], bins=100, histtype="step", linewidth=2, label="pseudo-data"
        )
        plt.plot(
            np.full(2, self.t_ar[0]),
            np.array([0, H[0].max()]),
            "r--",
            linewidth=2,
            label="data",
        )
        plt.legend(fontsize="large")
        plt.xlabel("BumpHunter statistic", size="large")
        plt.yscale("log")
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")

        # Check if the plot should be saved or just displayed
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
            plt.close(F)

        return

    @deprecated("Use `plot_stat` instead.")
    def PlotBHstat(self, *args, **kwargs):
        return self.plot_stat(*args, **kwargs)

    # Method to plot the signal injection result
    def plot_inject(self, filename=None):
        """
        Function that uses the parameters str_min and str_step as well as the result sigma_ar to generate a plot.

        Argument :
            fliename :
                Name of the file in which the plot will be saved.
                If None, the plot will be just shown but not saved.
                Default to None.
        """

        # Get the x-values (signal strength)
        if self.str_scale == "lin":
            sig_str = np.arange(
                self.str_min,
                self.str_min + self.str_step * len(self.sigma_ar),
                step=self.str_step,
            )
        else:
            sig_str = np.array(
                [
                    i % 10 * 10 ** (self.str_min + i // 10)
                    for i in range(len(self.sigma_ar) + len(self.sigma_ar) // 10 + 1)
                    if i % 10 != 0
                ]
            )

        # If filename is not None and log scale must check
        if filename is not None and self.str_scale == "log":
            if isinstance(filename, str):
                print("WARNING : log plot for signal injection will not be saved !")
                nolog = True
            else:
                nolog = False

        # Do the plot
        F = plt.figure(figsize=(12, 8))
        plt.title("Significance vs signal strength")
        plt.errorbar(
            sig_str,
            self.sigma_ar[:, 0],
            xerr=0,
            yerr=[self.sigma_ar[:, 1], self.sigma_ar[:, 2]],
            linewidth=2,
            marker="o",
        )
        plt.xlabel("Signal strength", size="large")
        plt.ylabel("Significance", size="large")
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")

        if filename is None:
            plt.show()
        else:
            if self.str_scale == "log" and not nolog:
                plt.savefig(filename[0], bbox_inches="tight")
            else:
                plt.savefig(filename, bbox_inches="tight")
            plt.close(F)

        # If log scale, do also a log plot
        if self.str_scale == "log":
            F = plt.figure(figsize=(12, 8))
            plt.title("Significance vs signal strength (log scale)")
            plt.errorbar(
                sig_str,
                self.sigma_ar[:, 0],
                xerr=0,
                yerr=[self.sigma_ar[:, 1], self.sigma_ar[:, 2]],
                linewidth=2,
                marker="o",
            )
            plt.xlabel("Signal strength", size="large")
            plt.ylabel("Significance", size="large")
            plt.xscale("log")
            plt.xticks(fontsize="large")
            plt.yticks(fontsize="large")

            if filename is None:
                plt.show()
            else:
                if not nolog:
                    plt.savefig(filename[1], bbox_inches="tight")
                plt.close(F)

        return

    @deprecated("Use `plot_inject` instead.")
    def PlotInject(self, *args, **kwargs):
        return self.plot_inject(*args, **kwargs)

    # Method that print the local infomation about the most significante bump in data
    def print_bump_info(self):
        """
        Function that print the local infomation about the most significante bump in data.
        Information are printed to stdout.
        """

        # Print stuff
        print("BUMP WINDOW")
        print(f"   loc = {self.min_loc_ar[0]}")
        print(f"   width = {self.min_width_ar[0]}")
        print(
            f"   local p-value | t = {self.min_Pval_ar[0]:.5f} | {self.t_ar[0]:.5f}"
        )
        print("")

        return

    @deprecated("Use `print_bump_info` instead.")
    def PrintBumpInfo(self, *args, **kwargs):
        return self.print_bump_info(*args, **kwargs)

    # Function that print the global infomation about the most significante bump in data
    def print_bump_true(self, data, bkg, is_hist: bool=False):
        """
        Print the global informations about the most significante bump in data in real scale.
        Information are printed to stdout.

        Arguments :
            data :
                Numpy array containing the data.

            bkg :
                Numpy array containing the background.

            is_hist :
                Boolean specifying if data and bkg are given in histogram form or not.
                Default to False.
        """

        # Get the data and background in histogram form
        if not is_hist:
            H = np.histogram(data, bins=self.bins, range=self.rang)
            Hb = np.histogram(
                bkg, bins=self.bins, range=self.rang, weights=self.weights
            )[0]
        else:
            H = [data, self.bins]
            Hb = bkg

        # Print informations about the bump itself
        print("BUMP POSITION")
        Bmin = H[1][self.min_loc_ar[0]]
        Bmax = H[1][self.min_loc_ar[0] + self.min_width_ar[0]]
        Bmean = (Bmax + Bmin) / 2
        Bwidth = Bmax - Bmin

        print(f"   min : {Bmin:.3f}")
        print(f"   max : {Bmax:.3f}")
        print(f"   mean : {Bmean:.3f}")
        print(f"   width : {Bwidth:.3f}")
        print(f"   number of signal events : {self.signal_eval}")
        print(f"   global p-value : {self.global_Pval:1.5f}")
        print(f"   significance = {self.significance:1.5f}")
        print("")

        return

    @deprecated("Use `print_bump_true` instead.")
    def PrintBumpTrue(self, *args, **kwargs):
        return self.print_bump_true(*args, **kwargs)

    # end of BumpHunter class


class BumpHunterInterface(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        """
        Reset all the inner result parameter for this BumpHunter instance.
        Use with caution.
        """
        pass

    @abstractmethod
    def save_state(self):
        """
        Save the current state (all parameters and results) of a BupHunter instance into a dict variable.

        Ruturns:
            state : 
                The dict containing all the parameters and results of this BumpHunter instance.
                The keys of the dict entries correspond the name of their associated parameters/results as defined in the BumpHunter class.d
        """
        pass

    @abstractmethod
    def load_state(self, state):
        """
        Load all the parameters and results of a previous BumpHunter intance that were saved using the SaveState method.

        Arguments :
            state :
                A dict containing all the parameters/results of a previous BumpHunter instance.
                If a parameter or a result field is missing, it will be set to its default value.
        """
        pass

    @abstractmethod
    def bump_scan(self, data, bkg, is_hist, do_pseudo):
        """
        Function that perform the full BumpHunter algorithm presented in https://arxiv.org/pdf/1101.0390.pdf without sidebands.
        This includes the generation of pseudo-data, the calculation of the BumpHunter p-value associated to data and to all pseudo experiment as well as the calculation of the test satistic t.

        The results are stored in the inner result variables of this BumpHunter instance.

        Arguments :
            data : 
                Numpy array containing the data distribution.
                This distribution will be transformed into a binned histogram and the algorithm will look for the most significant excess.

            bkg :
                Numpy array containing the background reference distribution.
                This distribution will be transformed into a binned histogram and the algorithm will compare it to data while looking for a bump.

            is_hist :
                Boolean that specify if the given data and background are already in histogram form.
                If true, the data and backgrouns are considered as already 'histogramed'.
                Default to False.

            do_pseudo :
                Boolean specifying if pesudo data should be generated.
                If False, then the BumpHunter statistics distribution kept in memmory is used to compute the global p-value and significance.
                If there is nothing in memmory, the global p-value and significance will not be computed.
                Default to True.


        Result inner variables :
            global_Pval :
                Global p-value obtained from the test statistic distribution.

            res_ar :
                 Array of containers containing all the p-value calculated durring the scan of the data (indice=0) and of the pseudo-data (indice>0).
                 For more detail about how the p-values are sorted in the containers, please reffer the the doc of the function scan_hist.

            min_Pval_ar :
                Array containing the minimum p-values obtained for the data (indice=0) and and the pseudo-data (indice>0).

            min_loc_ar :
                Array containing the positions of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

            min_width_ar : 
                Array containing the width of the windows for which the minimum p-value has been found for the data (indice=0) and pseudo-data (indice>0).

            signal_eval :
                Number of signal events evaluated form the last scan.
        """
        pass

    @abstractmethod
    def signal_inject(self, sig, bkg, is_hist):
        """
        Function that perform a signal injection test in order to determine the minimum signal strength required to reach a target significance.
        This function use the BumpHunter algorithm in order to calculate the reached significance for a given signal strength.

        This method share most of its parameters with the BumpScan method.

        Arguments :
            sig : 
                Numpy array containing the simulated signal. This distribution will be used to perform the signal injection.

            bkg :
                Numpy array containing the expected background.
                This distribution will be used to build the data in which signal will be injected.

            is_hist :
                Boolean that specify if the given data and background are already in histogram form.
                If true, the data and backgrouns are considered as already 'histogramed'.
                Default to False.

        Result inner variables :
            signal_ratio :
                Ratio signal_min/signal_exp (signal strength).
                If signal_exp is not specified, default to None.

            data_inject :
                Data obtained after injecting signal events in the backgound.

            sigma_ar :
                Numpy array containing the significance values obtained at each step.

        All the result inner variables of the BumpHunter instance will be filled with the results of the scan permormed
        during the last iteration (when sigma_limit is reached).
        """
        pass



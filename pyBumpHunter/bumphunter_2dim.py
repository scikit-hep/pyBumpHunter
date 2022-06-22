#!/usr/bin/env python
"""Extension of the BumpHunter algorithm to 2D distributions"""

import concurrent.futures as thd
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcl
from scipy.special import gammainc as G  # Need G(a,b) for the gamma function
from scipy.stats import norm, chi2
from scipy import optimize as So
from scipy.integrate import quad

from .functions import bh_stat
from pyBumpHunter.bumphunter_1dim import BumpHunterInterface

from .util import deprecated, deprecated_arg


# THE super BumpHunter2D class
class BumpHunter2D(BumpHunterInterface):
    """
    The BumpHunter2D class provide a extension of the BumpHunter algorithm to 2d histograms.
    Currently, only rectangular scan widows are supported.
    Also, note that 2D signal injection is not yet implemented.

    It comes with a 2D version of the methods and parameters of the one-dimentionnal BumpHunter class.


    List of inner parameter variables :
        rang :
            [x,y]-axis ranges of the histograms. Also define the ranges in which the scan will be performed.

        mode :
            String specifying if the algorithm must look for a excess or a deficit in the data.
            Can be either 'excess' or 'deficit'.

        width_min :
            Minimum [x,y] values of the scan window width that should be tested (in number of bins).

        width_max :
            Maximum [x,y] values of the scan window width that should be tested (in number of bins).
            Can be either None or an array-like of 2 positive integers.
            If None, the value is set to the total number of bins of the histograms divided by 2.

        width_step :
            Number of bins by which the scan window width is increased at each step.

        scan_step :
            Number of bins by which the position of the scan window is shifted at each step.
            Can an array-like length 2 of either {'full', 'half'} or positive integers.
            If 'full', the window will be shifted by a number of bins equal to its width.
            If 'half', the window will be shifted by a number of bins equal to max(1,width//2).

        npe :
            Number of pseudo-data distributions to be sampled from the reference background distribution.

        bins :
            Define the bins of the histograms. Can be ether a [integer,integer] or a 2D array-like of floats.
            If integer ([N,M]), N*M bins of equal width will be considered.
            If 2D array-like of float (a), a number of bins equal to (a.shape[0]-1)*(a.shape[1]-1) with the values of a
            as edges will be considered (variable width bins allowed).

        weights :
            Weights for the background distribution. Can be either None or a array-like of float.
            If array-like of floats, each background events will be accounted by its weights when making histograms.
            The size of the array-like must be the same than the number of of bkg events.
            If None, no weights will be considered.

        nworker :
            Number of thread to be run in parallel when scanning all the histograms (data and pseudo-data).
            If less or equal to 1, then parallelism will be disabled.

        seed :
            Seed for the random number generator.

        use_sideband :
            Boolean specifying if side-band normalization should be applied when computing p-values.

        sideband_width :
            Specify the number of bin to be used as side-band during the scan when side-band normalization is activated.
            The side-band will be removed from the scan range, but it will be used for background normalization.
            If None, then all the histograms range will be used for both the scan and normalization.

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

        norm_scale :
            The scale factor computed with side-band normalization.
            If use_sideband is False, norm_scale will be None (not computed)

        signal_min :
            Minimum number of signal events ones must inject in the data in order to reach the required significance.

        signal_ratio :
            Ratio signal_min/signal_exp (signal strength).

        data_inject :
            Data obtained after injecting signal events in the backgound.

        sigma_ar :
            Numpy array containing the significance values obtained at each step of the injection.
    """

    # Initializer method
    def __init__(
        self,
        mode: str = "excess",
        width_min=None,
        width_max=None,
        width_step=[1, 1],
        scan_step=[1, 1],
        npe: int = 50_000,
        nworker: int = 4,
        sigma_limit: float = 5,
        str_min: float = 0.5,
        str_step: float = 0.25,
        str_scale: str = "lin",
        flip_sig: bool = True,
        npe_inject: int = 100,
        seed=None,
        use_sideband: bool = False,
        sideband_width=None,
        check_overlap: bool = False
    ):
        """
        Arguments:
            mode :
                String specifying if the algorithm must look for a excess or a deficit in the data.
                Can be either 'excess' or 'deficit'.
                Default to 'excess'.

            width_min :
                Minimum [x,y] values of the scan window width that should be tested (in number of bins).
                Can be either None or a array-like of 2 positive integers.
                If None, it will be set to [1,1].
                Default to None.

            width_max :
                Maximum [x,y] values of the scan window width that should be tested (in number of bins).
                Can be either None or an array-like of 2 positive integers.
                If None, the value is set to the total number of bins of the histograms divided by 2. Default to none.

            width_step :
                Number of bins by which the scan window width is increased at each step.
                Default to [1,1].

            scan_step :
                Number of bins by which the position of the scan window is shifted at each step.
                Can an array-like length 2 of either 'full', 'half' or positive integers.
                If 'full', the window will be shifted by a number of bins equal to its width.
                If 'half', the window will be shifted by a number of bins equal to max(1,width//2).
                Default to [1,1].

            npe :
                Number of pseudo-data distributions to be sampled from the reference background distribution.
                Default to 100.

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
                If 'log', the signal strength will vary according to a log scale starting from 10**str_min
                If 'lin', the signal will vary according to a linear scale starting from str_min with a step of str_step.
                Default to 'lin'.

            signal_exp :
                Expected number of signal used to compute the signal strength.
                If None, the signal strength is not computed.
                Default to None.

            flip_sig :
                Boolean specifying if the signal should be fliped when running in deficit mode.
                Ignored in excess mode.
                Default to True.

            seed :
                Seed for the random number generator.
                Default to None.

            use_sideband :
                Boolean specifying if the side-band normalization should be applied.
                Default to False.

            sideband_width :
                Specify the number of bin to be used as side-band during the scan when side-band normalization is activated.
                The side-band will be removed from the scan range, but it will be used for background normalization.
                If None, then all the histograms range will be used for both the scan and normalization.
                Default to None.

            check_overlap :
                Boolean specifying if an overlap condition must be applied when combining channels.
                If True, the combined bump edges will be defined by the intersection of all deviation intervals found in individual channels.
                Default to False.
        """

        if width_min is None:
            width_min = [1, 1]

        # Initilize all inner parameter variables
        self.mode = mode
        self.width_min = width_min
        self.width_max = width_max
        self.width_step = width_step
        self.scan_step = scan_step
        self.npe = npe
        self.nworker = nworker
        self.sigma_limit = sigma_limit
        self.str_min = str_min
        self.str_step = str_step
        self.str_scale = str_scale
        self.flip_sig = flip_sig
        self.npe_inject = npe_inject
        self.seed = seed
        self.use_sideband = use_sideband
        self.sideband_width = sideband_width
        self.check_overlap = check_overlap

        # Initialize all inner result variables
        self.reset()

        return

    # Private methods

    # Method that performs a scan of a given data histogram and compares it to a reference background histogram.
    # This method is used by the BumpHunter class methods and is not intended to be used directly.
    def _scan_hist(self, hist, ref, w_ar, ih: int, ch: int):
        """Scan a distribution and compute the p-value associated to every scan window.

        The algorithm follows the BumpHunter algorithm. Compute also the significance for the data histogram.

        Arguments :
            hist :
                The data histogram (as obtain with the numpy.histogram2d function).

            ref :
                The reference (background) histogram (as obtain with the numpy.histogram2d function).

            w_ar :
                Array containing all the values of width to be tested.

            ih :
                Indice of the distribution to be scanned.
                ih=0 refers to the data distribution and ih>0 refers to the ih-th pseudo-data distribution.

            ch :
                Indice of the channel to be scanned.

        Results stored in inner variables :
            res :
                Numpy array of python list containing all the p-values of all windows computed durring the scan.
                The numpy array as dimention (Nwidth), with Nwidth the number of window's width tested.
                Each python list as dimension (Nstep), with Nstep the number of scan step for a given width (different for every value of width).

            min_Pval :
                Minimum p_value obtained durring the scan (float).

            min_loc :
                Position of the window corresponding to the minimum p-value ([integer,integer]).

            min_width :
                Width of the window corresponding to the minimum p-value ([integer,integer]).

            norm_scale :
                The scale factor computed with side-band normalization (float).
                If side-band normalization is not use, norm_scale is set to None.
        """

        # Check for sidebands
        if self.use_sideband:
            Vinf = np.array([0, 0], dtype=int)
            Vsup = np.array([hist.shape[0], hist.shape[1]], dtype=int)
            if self.sideband_width is not None:
                Hinf = Vinf + self.sideband_width
                Hsup = Vsup - self.sideband_width
            else:
                Hinf, Hsup = Vinf, Vsup
        else:
            Hinf = np.array([0, 0], dtype=int)
            Hsup = np.array([hist.shape[0], hist.shape[1]], dtype=int)

        # Create the results array
        res = np.empty(w_ar.shape[0], dtype=object)
        min_Pval = np.empty(w_ar.shape[0])
        min_loc = np.empty((w_ar.shape[0], 2), dtype=int)
        signal_eval = np.empty(w_ar.shape[0])

        # Prepare things for side-band normalization (if needed)
        if self.use_sideband:
            ref_total = ref.sum()
            hist_total = hist.sum()
            min_scale = np.empty(w_ar.shape[0])

        # Loop over all the width of the window
        for i, w in enumerate(w_ar):
            # Auto-adjust scan step if specified
            scan_stepp = [0, 0]
            if self.scan_step[0] == "full":
                scan_stepp[0] = w
            elif self.scan_step[0] == "half":
                scan_stepp[0] = max(1, w // 2)
            else:
                scan_stepp[0] = self.scan_step[0]

            if self.scan_step[1] == "full":
                scan_stepp[1] = w
            elif self.scan_step[1] == "half":
                scan_stepp[1] = max(1, w // 2)
            else:
                scan_stepp[1] = self.scan_step[1]

            # Define possition range
            posx = np.arange(Hinf[0], Hsup[0] - w[0] + 1, scan_stepp[0])
            posy = np.arange(Hinf[1], Hsup[1] - w[1] + 1, scan_stepp[1])
            pos = np.array([[p[0], p[1]] for p in itertools.product(posx, posy)])

            # Check that there is at least one interval to check for width w
            # If not, we must set dummy values in order to avoid crashes
            if pos.size == 0:
                res[i] = np.array([1.0])
                min_Pval[i] = 1.0
                min_loc[i] = np.zeros(2, dtype=int)
                signal_eval[i] = 0
                continue

            # Initialize local p-value array for width w
            res[i] = np.ones(pos.shape[0])

            # Count events in all windows of width w
            # FIXME any better way to do it ?? Without loop ?? FIXME
            Nref = np.array(
                [ref[p[0] : p[0] + w[0], p[1] : p[1] + w[1]].sum() for p in pos],
                dtype=float
            )
            Nhist = np.array(
                [hist[p[0] : p[0] + w[0], p[1] : p[1] + w[1]].sum() for p in pos]
            )

            # Compute and apply side-band normalization scale factor (if needed)
            if self.use_sideband:
                scale = (hist_total - Nhist) / (ref_total - Nref)
                Nref *= scale

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
            signal_eval[i] = Nhist[res[i].argmin()] - Nref[res[i].argmin()]
            if self.use_sideband:
                min_scale[i] = scale[res[i].argmin()]

        # Get the minimum p-value and associated window among all width
        min_width = w_ar[min_Pval.argmin()]
        min_loc = min_loc[min_Pval.argmin()]
        if self.use_sideband:
            min_scale = min_scale[min_Pval.argmin()]

        # Evaluate the number of signal event (for data only)
        if ih == 0 and self.res_ar.ndim == 1:
            # We do a simple scan
            self.signal_eval[ch] = signal_eval[min_Pval.argmin()]
        elif ih < self.npe_inject and self.res_ar.ndim == 2:
            # We do a signal injection test
            self.signal_eval[ih, ch] = signal_eval[min_Pval.argmin()]

        min_Pval = min_Pval.min()

        # Save the other results in inner variables and return
        if ih == 0 and self.res_ar.ndim == 1:
            # Fill results for simple scan
            self.res_ar[ch] = res
            if self.use_sideband:
                self.norm_scale[ch] = min_scale
        elif ih < self.npe_inject and self.res_ar.ndim == 2:
            # Fill results for signal injection
            self.res_ar[ih, ch] = res
            if self.use_sideband:
                self.norm_scale[ch] = min_scale
        self.min_Pval_ar[ih, ch] = min_Pval
        self.min_loc_ar[ih, :, ch] = np.array([int(min_loc[0]), int(min_loc[1])], dtype=int)
        self.min_width_ar[ih, :, ch] = np.array([int(min_width[0]), int(min_width[1])], dtype=int)

    # Method to get the combined bump edges (if there is any)
    def _bump_combined(self, data):
        """
        Mehtod to get the combined bump edges.
        This method is used internaly, but can be safely used by the front-end user.

        Arguments :
            data :
            A DataHandler instance containing the information about the bining.

        Returns :
            comb_bump :
                The combined bump edges returned two lists of 2 floats (one per dimention).
                If there is no combined bump, two None are returned insted.
        """

        # Check if there is anyting to be combined
        if self.min_loc_ar == [] or data.nchan == 1:
            return None, None

        # Get the combined edge of the bump (intersection of each channel bumps)
        leftx = np.array([
            data.bins[ch][0][self.min_loc_ar[0, 0, ch]]
            for ch in range(data.nchan)
        ]).max()
        rightx = np.array([
            data.bins[ch][0][self.min_loc_ar[0, 0, ch] + self.min_width_ar[0, 0, ch]]
            for ch in range(data.nchan)
        ]).min()
        lefty = np.array([
            data.bins[ch][1][self.min_loc_ar[0, 1, ch]]
            for ch in range(data.nchan)
        ]).max()
        righty = np.array([
            data.bins[ch][1][self.min_loc_ar[0, 1, ch] + self.min_width_ar[0, 1, ch]]
            for ch in range(data.nchan)
        ]).min()

        # The combined bump is defined if left < right for both x and y
        if leftx < rightx and lefty < righty:
            return [leftx, rightx], [lefty, righty]

        return None, None

    ## Variable management methods

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
        self.norm_scale = None
        self.signal_min = 0
        self.signal_ratio = None
        self.data_inject = []
        self.sigma_ar = []
        self.fit_param = None
        self.fit_Pval = 0
        self.fit_sigma = 0
        self.comb_Pval = 0
        self.comb_sigma = 0

        return

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
        state["sig_flip"] = self.flip_sig
        state["npe_inject"] = self.npe_inject
        state["use_sideband"] = self.use_sideband
        state["sideband_width"] = self.sideband_width
        state["check_overlap"] = self.check_overlap

        # Save results
        state["global_Pval"] = self.global_Pval
        state["significance"] = self.significance
        state["res_ar"] = self.res_ar
        state["min_Pval_ar"] = self.min_Pval_ar
        state["min_loc_ar"] = self.min_loc_ar
        state["min_width_ar"] = self.min_width_ar
        state["t_ar"] = self.t_ar
        state["signal_eval"] = self.signal_eval
        state["norm_scale"] = self.norm_scale
        state["fit_param"] = self.fit_param
        state["fit_Pval"] = self.fit_Pval
        state["fit_sigma"] = self.fit_sigma
        state["comb_Pval"] = self.comb_Pval
        state["comb_sigma"] = self.comb_sigma
        state["signal_min"] = self.signal_min
        state["signal_ratio"] = self.signal_ratio
        state["data_inject"] = self.data_inject

        return state

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
        self.width_min = state.get("width_min", [1, 1])
        self.width_max = state.get("width_max", None)
        self.width_step = state.get("width_step", [1, 1])
        self.scan_step = state.get("scan_step", [1, 1])
        self.npe = state.get("npe", 50_000)
        self.nworker = state.get("nworker", 4)
        self.seed = state.get("seed", None)
        self.use_sideband = state.get("use_sideband", False)
        self.sideband_width = state.get("sideband_width", None)
        self.check_overlap = state.get("check_overlap", False)
        self.sigma_limit = state.get("sigma_limit", 5)
        self.str_min = state.get("str_min", 0.5)
        self.str_step = state.get("str_step", 0.25)
        self.str_scale = state.get("str_scale", "lin")
        self.sig_flip = state.get("sig_flip", True)
        self.npe_inject = state.get("npe_inject", 100)

        # Load results
        self.reset()
        if "global_Pval" in state:
            self.global_Pval = state["global_Pval"]
        if "significance" in state:
            self.significance = state["significance"]
        if "res_ar" in state:
            self.res_ar = state["res_ar"]
        if "min_Pval_ar" in state:
            self.min_Pval_ar = state["min_Pval_ar"]
        if "min_loc_ar" in state:
            self.min_loc_ar = state["min_loc_ar"]
        if "min_width_ar" in state:
            self.min_width_ar = state["min_width_ar"]
        if "t_ar" in state:
            self.t_ar = state["t_ar"]
        if "signal_eval" in state:
            self.signal_eval = state["signal_eval"]
        if "norm_scale" in state:
            self.norm_scale = state["norm_scale"]
        if "fit_param" in state:
            self.fit_param = state["fit_param"]
        if "fit_Pval" in state:
            self.fit_Pval = state["fit_Pval"]
        if "fit_sigma" in state:
            self.fit_sigma = state["fit_sigma"]
        if "comb_Pval" in state:
            self.comb_Pval = state["comb_Pval"]
        if "comb_sigma" in state:
            self.comb_sigma = state["comb_sigma"]
        if "signal_min" in state:
            self.signal_min = state["signal_min"]
        if "signal_ratio" in state:
            self.signal_ratio = state["signal_ratio"]
        if "data_inject" in state:
            self.data_inject = state["data_inject"]
        return


    ## Scan methods

    # Method that perform the scan on every pseudo experiment and data (in parrallel threads).
    # For each scan, the value of p-value and test statistic t is computed and stored in result array
    def bump_scan(
        self,
        data,
        do_pseudo: bool = True,
        verbose: bool = True
    ):
        """
        Function that perform the full BumpHunter algorithm presented in https://arxiv.org/pdf/1101.0390.pdf without sidebands.
        This includes the generation of pseudo-data, the calculation of the BumpHunter p-value associated to data and to all pseudo experiment as well as the calculation of the test satistic t.

        The results are stored in the inner result variables of this BumpHunter instance.

        Arguments :
            data :
                A DataHandler containing at least a set of reference background and data histograms.
                Ths distributions must be 1D histograms.

            do_pseudo :
                Boolean specifying if pesudo data should be generated.
                If False, then the BumpHunter statistics distribution kept in memory is used to compute the global p-value and significance.
                If there is nothing in memory, the global p-value and significance will not be computed.
                Default to True.

            verbose :
                Boolean specifying if the detailed ouput must be printed.
                Default to True.

        Result inner variables :
            global_Pval :
                Global p-value obtained from the test statistic distribution.

            res_ar :
                Array of containers containing all the p-value calculated durring the scan of the data.
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

        # Check the dimension of the provided histograms
        if data.ndim == 1:
            raise ValueError("ERROR : The provided data contains 1D histograms !\nYou must use a BumpHunter1D instance.")

        # Generate all the pseudo-data histograms
        if do_pseudo:
            if verbose:
                print(f"Generating {self.npe} background-only histograms")

            pseudo_hist = []
            for ch in range(data.nchan):
                pseudo_hist.append(data.ref[ch].flatten())
                pseudo_hist[ch] = np.random.poisson(
                    lam=np.tile(pseudo_hist[ch], (self.npe, 1)).transpose(),
                    size=(pseudo_hist[ch].size, self.npe),
                )
                pseudo_hist[ch] = np.reshape(
                    pseudo_hist[ch], (data.ref[ch].shape[0], data.ref[ch].shape[1], self.npe)
                )

        # Set width_max if it is given as None
        if self.width_max is None:
            self.width_max = [data.hist[0].shape[0] // 2, data.hist[0].shape[1] // 2]

        # Initialize all results containenrs
        if do_pseudo:
            # Initialize/reset all results containenrs
            self.min_Pval_ar = np.empty((self.npe + 1, data.nchan))
            self.min_loc_ar = np.empty((self.npe + 1, 2, data.nchan), dtype=int)
            self.min_width_ar = np.empty((self.npe + 1, 2, data.nchan), dtype=int)
            self.t_ar = np.empty((self.npe + 1, data.nchan))
        else:
            # Check previous results are available
            if not isinstance(self.t_ar, np.ndarray) or self.t_ar.size != self.npe + 1:
                # No pseudo data requested and no previous results available
                self.min_Pval_ar = np.empty((1, data.nchan))
                self.min_loc_ar = np.empty((1, 2, data.nchan), dtype=int)
                self.min_width_ar = np.empty((1, 2, data.nchan), dtype=int)
                self.t_ar = np.empty((1, data.nchan))
        self.res_ar = np.empty((data.nchan), dtype=object)
        self.signal_eval = np.empty((data.nchan))
        self.norm_scale = np.ones((data.nchan))
        self.global_Pval = np.empty((data.nchan))
        self.significance = np.empty((data.nchan))
        self.fit_param = [None for ch in range(data.nchan)]
        self.fit_Pval = np.zeros((data.nchan))
        self.fit_sigma = np.zeros((data.nchan))

        # Auto-adjust the value of width_max and do an array of all width
        wx = np.arange(self.width_min[0], self.width_max[0] + 1, self.width_step[0])
        wy = np.arange(self.width_min[1], self.width_max[1] + 1, self.width_step[1])
        w_ar = np.array([[w[0], w[1]] for w in itertools.product(wx, wy)])
        if verbose:
            print(f"{w_ar.shape[0]} values of width will be tested")

        # Loop over channels
        for ch in range(data.nchan):
            if verbose:
                print(f"SCAN CH{ch}")

            # Scan the data histogram for channel ch
            self._scan_hist(data.hist[ch], data.ref[ch], w_ar, 0, ch)

            # Loop over all pseudo-data histograms to be scanned (if required)
            if do_pseudo:
                # TODO add a condition for multi-core computation that works
                for th in range(self.npe):
                    # Scan channel ch of histogram th
                    self._scan_hist(
                        pseudo_hist[ch][:, :, th],
                        data.ref[ch],
                        w_ar,
                        th + 1,
                        ch
                    )

        # Use the p-value results to compute t
        self.t_ar = -np.log(self.min_Pval_ar)

        # Check if the global p-value should be computed
        if self.t_ar.shape[0] > 1:
            # A mask that tells if a fit was performed for a given channel
            do_fit = np.array([False for ch in range(data.nchan)])

            # Loop over channels
            for ch in range(data.nchan):
                if verbose:
                    print(f"####CH{ch}")

                # Compute the global p-value for channel ch
                tdat = self.t_ar[0, ch]
                S = self.t_ar[1:, ch][self.t_ar[1:, ch] >= tdat].size
                self.global_Pval[ch] = S / self.npe
                if verbose:
                    print(f"Global p-value : {self.global_Pval[ch]:1.5f}  ({S} / {self.npe})")

                # Check global p-value
                if self.global_Pval[ch] == 1:
                    self.significance[ch] = 0
                    if verbose:
                        print(f"Significance = {self.significance[ch]}")
                elif self.global_Pval[ch] == 0:
                    # I this case, we can't compute directly the significance, so we set a limit
                    self.significance[ch] = norm.ppf(1 - (1 / self.npe))
                    if verbose:
                        print(f"Significance > {self.significance[ch]:1.5f} (lower limit)")
                else:
                    self.significance[ch] = norm.ppf(1 - self.global_Pval[ch])
                    if verbose:
                        print(f"Significance = {self.significance[ch]:1.5f}")

                # Check if a fit of the test statistic distribution is needed
                if S < 100:
                    if verbose:
                        print("Fit is required !")

                    # Mark channel in the do_fit mask
                    do_fit[ch] = True

                    # Make histogram of bh_stat distribution
                    Hbh, bbh = np.histogram(self.t_ar[1:, ch][self.t_ar[1:, ch] > 1e-3], bins=50)
                    x = (bbh[:-1] + bbh[1:]) / 2
                    erry = np.sqrt(Hbh)
                    erry[Hbh==0] = 1.5  # Avoid division by 0

                    # Fit the t_ar distribution with scipy
                    param, cov = So.curve_fit(
                        bh_stat,
                        x,
                        Hbh,
                        p0=[0.5, data.hist[ch].size, 42],
                        sigma=erry,
                        absolute_sigma=True
                    )

                    # Save fit results (params)
                    self.fit_param[ch] = {
                        "pM": [param[0], np.sqrt(cov[0,0])],
                        "m": [param[1], np.sqrt(cov[1,1])],
                        "A": [param[2], np.sqrt(cov[2,2])]
                    }

                    # Compute global p-value by integrating the fitted function
                    self.fit_Pval[ch], _ = quad(bh_stat, 1e-10, tdat, args=(param[0], param[1], 1))
                    self.fit_Pval[ch] = 1 - self.fit_Pval[ch]

                    # Compute corresponding significance
                    self.fit_sigma[ch] = norm.ppf(1 - self.fit_Pval[ch])

                    # Print results if required
                    if verbose:
                        print(f"Global p-value (fit) = {self.fit_Pval[ch]:1.5f}")
                        print(f"significnce (fit) = {self.fit_sigma[ch]:.5f}")
                    
            # Combine global p-values (if there are more than one channels)
            if data.nchan > 1:
                # Check if the overlap condition is required
                do_comb = True
                if self.check_overlap:
                    # Get the combined bump edges
                    bump_combx, _ = self._bump_combined(data)

                    # Update do_comb accordingly
                    do_comb = bump_combx is not None

                # Do the combination if needed
                if do_comb:
                    # Get the corect p-values to be combined
                    pval = np.empty((data.nchan))
                    pval[do_fit] = self.fit_Pval[do_fit]
                    pval[~do_fit] = self.global_Pval[~do_fit]

                    # Compute combined test statistic with Fisher method
                    tcomb = -2 * np.sum(np.log(pval))

                    # Compute combined global p-value and significance from tcomb
                    self.comb_Pval = 1 - chi2.cdf(tcomb, df=2 * data.nchan)
                    self.comb_sigma = norm.ppf(1 - self.comb_Pval)

                    # Yet another little print
                    if verbose :
                        print(f"####COMBINED")
                        print(f"Global p-balue = {self.comb_Pval:1.5f}")
                        print(f"Significance = {self.comb_sigma:.5f}")
                else:
                    # The combination is not possible (overlap check failed)
                    self.comb_Pval = 1
                    self.comb_sigma = 0
                    if verbose:
                        print(f"####COMBINED")
                        print(f"Overlap check failed !")
                        print(f"Significance = {self.comb_sigma}")
        else:
            print("No pseudo data found : can't compute global p-value")
        print("")

        return

    # Perform signal injection on background and determine the minimum amount of signal required for observation
    def signal_inject(self, data, do_pseudo: bool = True, verbose: bool = True):
        """
        Function that perform a signal injection test in order to determine the minimum signal strength required to reach a target significance.
        This function use the BumpHunter algorithm in order to calculate the reached significance for a given signal strength.

        This method share most of its parameters with the BumpScan method.

        Arguments :
            data :
                Numpy array containing the simulated signal. This distribution will be used to perform the signal injection.

            do_pseudo :
                Boolean specifying if background-only pseudo-data must be generated.
                If False, the pseudo-data from the previous scan are kept.
                Default to True.

            verbose :
                Boolean specifying if the detailed ouput must be printed.
                Default to True.

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

        # Check the dimension of the provided histograms
        if data.ndim == 1:
            raise ValueError("ERROR : The provided data contains 1D histograms !\nYou must use a BumpHunter1D instance.")

        # Reset significance and p-value global variables
        self.significance = np.empty(data.nchan)
        self.global_Pval = np.empty(data.nchan)
        self.fit_param = [None for ch in range(data.nchan)]
        self.fit_Pval = np.zeros(data.nchan)
        self.fit_sigma = np.zeros(data.nchan)
        self.sigma_ar = []

        # If do_pseudo is False, must check if previous results are avalable
        if not do_pseudo:
            if (not isinstance(self.t_ar, np.ndarray)) or self.t_ar.shape[0] == 1: #TODO check
                print("Warning : pseudo-data are required to performe signal injection.")
                do_pseudo = True

                # Check if last scan was injection
                if self.t_ar.shape[0] == self.npe + 1:
                    # If not, must retrieve bkg only results
                    bkg_loc = self.min_loc_ar[1:]
                    bkg_width = self.min_width_ar[1:]
                    bkg_Pval = self.min_Pvak_ar[1:]
                    bkg_t = self.t_ar[1:]

        # Generate pseudo-data by sampling background
        if do_pseudo:
            if verbose:
                print(f"Generating {self.npe} background-only histograms")

            # loop over channels
            pseudo_hist = []
            for ch in range(data.nchan):
                pseudo_hist.append(data.ref[ch].flatten())
                pseudo_hist[ch] = np.random.poisson(
                    lam=np.tile(pseudo_hist[ch], (self.npe, 1)).transpose(),
                    size=(pseudo_hist[ch].size, self.npe),
                )
                pseudo_hist[ch] = np.reshape(
                    pseudo_hist[ch], (data.ref[ch].shape[0], data.ref[ch].shape[1], self.npe)
                )

        # Set width_max if it is given as None
        if self.width_max is None:
            self.width_max = [data.ref.shape[0] // 2, data.ref.shape[0] // 2]

        # Initialize all results containenrs (keep bkg results if needed)
        if do_pseudo:
            self.min_Pval_ar = np.empty((self.npe + self.npe_inject, data.nchan))
            self.min_loc_ar = np.empty((self.npe + self.npe_inject, 2, data.nchan), dtype=int)
            self.min_width_ar = np.empty((self.npe + self.npe_inject, 2, data.nchan), dtype=int)
            self.t_ar = np.empty((self.npe + self.npe_inject, data.nchan))
        elif self.t_ar.shape[0] == self.npe + 1:
            self.min_Pval_ar = np.empty((self.npe + self.npe_inject, data.nchan))
            self.min_Pval_ar[:npe_inject] = bkg_Pval
            self.min_loc_ar = np.empty((self.npe + self.npe_inject, 2, data.nchan), dtype=int)
            self.min_loc_ar[:npe_inject] = bkg_loc
            self.min_width_ar = np.empty((self.npe + self.npe_inject, 2, data.nchan), dtype=int)
            self.min_width_ar[:npe_inject] = bkg_width
            self.t_ar = np.empty((self.npe + self.npe_inject, data.nchan))
            self.t_ar[:npe_inject] = bkg_t
            del bkg_Pval
            del bkg_width
            del bkg_loc
            del bkg_t
        self.signal_eval = np.empty((self.npe_inject, data.nchan))
        self.res_ar = np.empty((self.npe_inject, data.nchan), dtype=object)

        # Auto-adjust the value of width_max and do an array of all width
        wx = np.arange(self.width_min[0], self.width_max[0] + 1, self.width_step[0])
        wy = np.arange(self.width_min[1], self.width_max[1] + 1, self.width_step[1])
        w_ar = np.array([[w[0], w[1]] for w in itertools.product(wx, wy)])
        if verbose:
            print(f"{w_ar.shape[0]} values of width will be tested")


        # Scan the background-only pseudo-data if required
        if do_pseudo:
            # Loop over channels
            for ch in range(data.nchan):
                if verbose:
                    print(f"BACKGROUND SCAN CH{ch}")

                # TODO add a condition for multi-core computation that works
                for th in range(self.npe):
                    # Scan channel ch of histogram th
                    self._scan_hist(
                        pseudo_hist[ch][:, :, th],
                        data.ref[ch],
                        w_ar,
                        th + self.npe_inject,
                        ch
                    )

        # Use the p-value results to compute t
        self.t_ar[self.npe_inject:] = -np.log(self.min_Pval_ar[self.npe_inject:])

        # Don't need them anymore
        del pseudo_hist

        # Main injection loop
        if verbose:
            print("STARTING INJECTION")
        for i in range(1, 10_001): # Restrict to 10k steps to avoid infinite loop
            # Check how we should compute the signal strength to be injected
            if self.str_scale == "lin":
                # Signal strength increase linearly at each step
                if i == 1:
                    strength = self.str_min
                else:
                    strength += self.str_step
            elif self.str_scale == "log":
                # Signal strength increase to form a logarithmic scale axis
                if i == 1:
                    strength = 10**self.str_min
                    self.str_step = strength
                else:
                    strength += self.str_step
                    if abs(strength - 10 * self.str_step) < 1e-6:
                        self.str_step *= 10

            # Update signal_min
            self.signal_min = np.array(data.signal_exp) * strength
            if verbose:
                print(f"   STEP {i} : signal strength = {strength}")

            # Check if we inject a deficit
            if self.mode == "deficit":
                self.signal_min = -self.signal_min

            # Scale the signal to the current strength
            sig_hist = [
                data.sig[ch] * strength * (data.signal_exp[ch] / data.sig[ch].sum())
                for ch in range(data.nchan)
            ]

            # Check if sig_hist should be fliped in deficit mode
            if self.mode == "deficit":
                if self.flip_sig:
                    sig_hist = [-sig_hist[ch] for ch in range(data.nchan)]

            # Inject the signal and do some poissonian fluctuation for all channels
            if verbose:
                print(f"Generating {self.npe_inject} background+signal histograms")
            pseudo_hist = []
            data_hist = []
            for ch in range(data.nchan):
                data_hist.append(data.ref[ch] + sig_hist[ch])
                pseudo_hist.append(data_hist[ch].flatten())
                pseudo_hist[ch] = np.random.poisson(
                    lam=np.tile(pseudo_hist[ch], (self.npe_inject, 1)).transpose(),
                    size=(pseudo_hist[ch].size, self.npe_inject),
                )
                pseudo_hist[ch] = np.reshape(
                    pseudo_hist[ch], (data.ref[ch].shape[0], data.ref[ch].shape[1], self.npe_inject)
                )

            # Reset fit results
            self.fit_param = [None for ch in range(data.nchan)]
            self.fit_Pval = np.zeros(data.nchan)
            self.fit_sigma = np.zeros(data.nchan)

            # Loop over channels
            for ch in range(data.nchan):
                if verbose:
                    print(f"BACKGROUND+SIGNAL SCAN CH{ch}")

                # TODO add a condition for multi-core computation that works
                for th in range(self.npe_inject):
                    # Scan channel ch of histogram th
                    self._scan_hist(
                        pseudo_hist[ch][:, :, th],
                        data.ref[ch],
                        w_ar,
                        th,
                        ch
                    )

            # Use the p-value results to compute t
            self.t_ar[:self.npe_inject] = -np.log(self.min_Pval_ar[:self.npe_inject])

            # Initialize a few things
            do_fit = np.array([False for ch in range(data.nchan)])
            global_inf = np.zeros(data.nchan)
            global_sup = np.zeros(data.nchan)
            sigma_inf = np.zeros(data.nchan)
            sigma_sup = np.zeros(data.nchan)

            # Loop over channels
            for ch in range(data.nchan):
                if verbose:
                    print(f"####CH{ch}")

                # Compute the global p-value from the t distribution with inf end sup values
                tdat, tinf, tsup = (
                    np.median(self.t_ar[:self.npe_inject, ch]),
                    np.quantile(self.t_ar[:self.npe_inject, ch], 0.25),
                    np.quantile(self.t_ar[:self.npe_inject, ch], 0.75),
                )
                t_ar_bkg = self.t_ar[self.npe_inject:, ch]
                S = t_ar_bkg[t_ar_bkg > tdat].size
                Sinf = t_ar_bkg[t_ar_bkg > tinf].size
                Ssup = t_ar_bkg[t_ar_bkg > tsup].size
                self.global_Pval[ch] = S / self.npe
                global_inf[ch] = Sinf / self.npe
                global_sup[ch] = Ssup / self.npe
                if verbose:
                    print(
                        f"Global p-value : {self.global_Pval[ch]:1.5f}  ({S} / {self.npe})   {global_inf[ch]:1.5f}  ({Sinf})   {global_sup[ch]:1.5f}  ({Ssup})"
                    )

                # If global p-value is exactly 0, we might have trouble with the significance
                if self.global_Pval[ch] <  1 / self.npe:
                    self.significance[ch] = norm.ppf(1 - (1 / self.npe))
                else:
                    self.significance[ch] = norm.ppf(1 - self.global_Pval[ch])

                if global_inf[ch] <  1 / self.npe:
                    sigma_inf[ch] = norm.ppf(1 - (1 / self.npe))
                else:
                    sigma_inf[ch] = norm.ppf(1 - global_inf[ch])

                if global_sup[ch] <  1 / self.npe:
                    sigma_sup[ch] = norm.ppf(1 - (1 / self.npe))
                else:
                    sigma_sup[ch] = norm.ppf(1 - global_sup[ch])
                if verbose:
                    print(
                        f"Significance = {self.significance[ch]:.5g} ({sigma_inf[ch]:.5g}  {sigma_sup[ch]:.5g})"
                    )

                # Check if a fit of the test statistic distribution is needed
                if S < 100 or Sinf < 100 or Ssup < 100:
                    if verbose:
                        print("Fit is required !")

                    # Make histogram of bh_stat distribution
                    Hbh, bbh = np.histogram(
                        self.t_ar[self.npe_inject:, ch][self.t_ar[self.npe_inject:, ch] > 1e-3],
                        bins=50
                    )
                    x = (bbh[:-1] + bbh[1:]) / 2
                    erry = np.sqrt(Hbh)
                    erry[Hbh==0] = 1.5  # Avoid division by 0

                    # Fit the t_ar distribution with scipy
                    param, cov = So.curve_fit(
                        bh_stat,
                        x,
                        Hbh,
                        p0=[0.5, data.hist[ch].size, 42],
                        sigma=erry,
                        absolute_sigma=True
                    )

                    # Save fit results (params)
                    self.fit_param[ch] = {
                        "pM": [param[0], np.sqrt(cov[0,0])],
                        "m": [param[1], np.sqrt(cov[1,1])],
                        "A": [param[2], np.sqrt(cov[2,2])]
                    }

                    # Compute global p-value by integrating the fitted function
                    ptr = [self.global_Pval[ch], self.significance[ch]]
                    if S < 100:
                        self.fit_Pval[ch], _ = quad(bh_stat, 1e-10, tdat, args=(param[0], param[1], 1))
                        self.fit_Pval[ch] = 1 - self.fit_Pval[ch]
                        self.fit_sigma[ch] = norm.ppf(1 - self.fit_Pval[ch])

                        # Mark channel in the do_fit mask
                        do_fit[ch] = True
                        ptr = [self.fit_Pval[ch], self.fit_sigma[ch]]
                    if Sinf < 100:
                        global_inf[ch], _ = quad(bh_stat, 1e-10, tinf, args=(param[0], param[1], 1))
                        global_inf[ch] = 1 - global_inf[ch]
                        sigma_inf[ch] = norm.ppf(1 - global_inf[ch])
                    if Ssup < 100:
                        global_sup[ch], _ = quad(bh_stat, 1e-10, tsup, args=(param[0], param[1], 1))
                        global_sup[ch] = 1 - global_sup[ch]
                        sigma_sup[ch] = norm.ppf(1 - global_sup[ch])

                    # Print results if required
                    if verbose:
                        print(f"Global p-value (fit) = {ptr[0]:1.5f} ({global_inf[ch]:1.5f}  {global_sup[ch]:1.5f})")
                        print(f"significnce (fit) = {ptr[1]:.5g} ({sigma_inf[ch]:.5g}  {sigma_sup[ch]:.5g})")

            # Combine p-values if neeed
            if data.nchan > 1:
                # Get the corect p-values to be combined
                pval = np.empty((data.nchan))
                pval[do_fit] = self.fit_Pval[do_fit]
                pval[~do_fit] = self.global_Pval[~do_fit]

                # Compute combined test statistic with Fisher method
                tcomb = -2 * np.sum(np.log(pval))
                tcinf = -2 * np.sum(np.log(global_inf))
                tcsup = -2 * np.sum(np.log(global_sup))

                # Compute combined global p-value and significance from tcomb
                self.comb_Pval = 1 - chi2.cdf(tcomb, df=2 * data.nchan)
                global_inf = 1 - chi2.cdf(tcinf, df=2 * data.nchan)
                global_sup = 1 - chi2.cdf(tcsup, df=2 * data.nchan)
                self.comb_sigma = norm.ppf(1 - self.comb_Pval)
                sigma_inf = norm.ppf(1 - global_inf)
                sigma_sup = norm.ppf(1 - global_sup)

                # Yet another little print
                if verbose :
                    print(f"####COMBINED")
                    print(f"Global p-value = {self.comb_Pval:1.5f} ({global_inf:1.5f}  {global_sup:1.5f})")
                    print(f"Significance = {self.comb_sigma:.5f} ({sigma_inf:.5g}  {sigma_sup:.5g})")
            else:
                global_inf = global_inf[0]
                global_sup = global_sup[0]
                sigma_inf = sigma_inf[0]
                sigma_sup = sigma_sup[0]

            # Append reached significance to sigma_ar (with sup and inf variations)
            if data.nchan == 1:
                if self.global_Pval > 100/self.npe:
                    self.sigma_ar.append([
                        self.significance[ch],
                        np.abs(self.significance[ch] - sigma_inf),
                        np.abs(self.significance[ch] - sigma_sup),
                    ])
                else:
                    self.sigma_ar.append([
                        self.fit_sigma[ch],
                        np.abs(self.fit_sigma[ch] - sigma_inf),
                        np.abs(self.fit_sigma[ch] - sigma_sup),
                    ])
            else:
                self.sigma_ar.append([
                    self.comb_sigma,
                    np.abs(self.comb_sigma - sigma_inf),
                    np.abs(self.comb_sigma - sigma_sup),
                ])

            # Check if the sigma limit is reached
            if self.sigma_ar[-1][0] >= self.sigma_limit:
                if verbose:
                    print("REACHED SIGMA LIMIT")
                break

            # Check if we reached the end of the loop
            if i == 10_000:
                print("WARNING : Couldn't reach limit after 10_000 steps !'")
                print("WARNING : Stopping injection now")

            # Add spacing to the output if required
            if verbose:
                print("")

        # End of injection loop

        # Don't need them anymore
        del pseudo_hist

        # Compute signal strength
        self.signal_ratio = strength
        if verbose:
            print(f"   Signal strength : {self.signal_ratio:1.4f}")

        # Save the data obtained after last injection in inner variables
        self.data_inject = data_hist

        # Convert the sigma_ar inner variable into a numpy array
        self.sigma_ar = np.array(self.sigma_ar)

        return


    ## Display methods

    # Plot the data and bakground histograms with the bump found by BumpHunter highlighted
    def plot_bump(
        self,
        data,
        use_sideband=None,
        chan: int = 0,
        show_combined: bool = False,
        fontsize='xx-large'
    ):
        """
        Plot the data and bakground histograms with the bump found by BumpHunter highlighted.

        Arguments :
            data :
                A DataHandler object containing the data and reference histograms.

            use_sideband :
                Boolean specifying if side-band normalization should be used to correct the reference background in the plot.
                If None, self.use_sideband is used instead.
                Default to None.

            chan :
                Integer specify the number of the channel to be shown.
                Default to 0 (the first channel).

            show_combined :
                Boolean specifying if the combined bump must be shown.
                If False, the bump area of channel chan is shown instead.
                Ignored if self.check_overlap is False.
                Default to False.
                
            fontsize :
                Specify the font size for the ticks labels.
                Can be either an int or one of matplotlib font size string.
                Default to 'xx-large'.

        Returns :
            pl :
                A list of plt.subplot object conaining the axes used for the plot.
                It can be used to customize the figure (add labels, ...).
        """

        # Get bump min and max
        plot_bump = True
        if show_combined and self.check_overlap:
            Bx, By = self._bump_combined(data)
            if Bx is not None:
                Bminx, Bmaxx = Bx[0], Bx[1]
                Bminy, Bmaxy = By[0], By[1]
            else:
                plot_bump = False
        else:
            Bminx = data.bins[chan][0][self.min_loc_ar[0, 0, chan]]
            Bmaxx = data.bins[chan][0][self.min_loc_ar[0, 0, chan] + self.min_width_ar[0, 0, chan]]
            Bminy = data.bins[chan][1][self.min_loc_ar[0, 1, chan]]
            Bmaxy = data.bins[chan][1][self.min_loc_ar[0, 1, chan] + self.min_width_ar[0, 1, chan]]

        # Chek if we should apply sideband normalization correction
        if use_sideband is None:
            use_sideband = self.use_sideband

        if use_sideband:
            data.ref = data.ref * self.norm_scale[chan]

        # Calculate significance for each bin
        sig = np.ones(data.ref[chan].shape)
        sig[(data.hist[chan] > data.ref[chan]) & (data.ref[chan] > 0)] = G(
            data.hist[chan][(data.hist[chan] > data.ref[chan]) & (data.ref[chan] > 0)],
            data.ref[chan][(data.hist[chan] > data.ref[chan]) & (data.ref[chan] > 0)]
        )
        sig[data.hist[chan] < data.ref[chan]] = 1 - G(
            data.hist[chan][data.hist[chan] < data.ref[chan]] + 1,
            data.ref[chan][data.hist[chan] < data.ref[chan]]
        )
        sig = norm.ppf(1 - sig)
        sig[sig < 0.0] = 0.0  # If negative, set it to 0
        np.nan_to_num(sig, posinf=0, neginf=0, nan=0, copy=False)  # Avoid errors
        sig[data.hist[chan] < data.ref[chan]] = -sig[data.hist[chan] < data.ref[chan]]  # Now we can make it signed

        # Plot the test histograms with the bump found by BumpHunter plus a little significance plot
        # Do the plot in the current figure
        pl1 = plt.subplot(2, 1, 1)
        plt.pcolormesh(
            data.bins[chan][0],
            data.bins[chan][1],
            data.hist[chan].T,
            norm=mcl.LogNorm()
        )
        cb1 = plt.colorbar()
        if plot_bump:
            plt.hlines(
                [Bminy, Bmaxy],
                data.bins[chan][0][0],
                data.bins[chan][0][-1],
                linestyles="dashed",
                color="r",
                lw=2
            )
            plt.vlines(
                [Bminx, Bmaxx],
                data.bins[chan][1][0],
                data.bins[chan][1][-1],
                linestyles="dashed",
                color="r"
            )
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        cb1.ax.tick_params(labelsize=24)
        plt.tight_layout()

        pl2 = plt.subplot(2, 1, 2)
        plt.pcolormesh(
            data.bins[chan][0],
            data.bins[chan][1],
            sig.T
        )
        cb2 = plt.colorbar()
        if plot_bump:
            plt.hlines(
                [Bminy, Bmaxy],
                data.bins[chan][0][0],
                data.bins[chan][0][-1],
                linestyles="dashed",
                color="r",
                lw=2
            )
            plt.vlines(
                [Bminx, Bmaxx],
                data.bins[chan][1][0],
                data.bins[chan][1][-1],
                linestyles="dashed",
                color="r"
            )
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        cb2.ax.tick_params(labelsize=24)

        return [pl1, pl2]

    # Plot the Bumpunter test statistic distribution with the result for data
    def plot_stat(self, chan: int = 0):
        """
        Plot the Bumphunter statistic distribution together with the observed value with the data.

        Arguments :
            chan :
                Integer specify the number of the channel to be shown.
                Default to 0 (the first channel).
        """

        # Check the distributions to plot
        if self.t_ar.shape[0] == self.npe + 1:
            # Plot the bkg-only distribution and one data value
            H = plt.hist(
                self.t_ar[1:, chan],
                bins=50,
                histtype="step",
                lw=2,
            label="pseudo-data"
            )
            plt.vlines(
                self.t_ar[0, chan],
                0,
                H[0].max(),
                colors="r",
                linestyles="dashed",
                lw=2,
                label="data"
            )
        elif self.t_ar.shape[0] == self.npe + self.npe_inject:
            # Plot the bkg-only distribution and the bkg+sig distribution
            plt.hist(
                self.t_ar[self.npe_inject:, chan],
                bins=50,
                histtype="step",
                lw=2,
                label="background only"
            )
            plt.hist(
                self.t_ar[:self.npe_inject, chan],
                bins=50,
                histtype="step",
                lw=2,
                label="background+signal"
            )
        else:
            print("Nothing to plot here !")
            return

        # Check if a fit was done
        if self.fit_param[chan] is not None:
            # Do plot the fit function
            x = np.linspace(self.t_ar[:, chan].min(), self.t_ar[:, chan].max() + 1, 150)
            param = {p:v[0] for p, v in self.fit_param[chan].items()}
            plt.plot(
                x,
                bh_stat(x, **param),
                'g-',
                lw=2,
                label='fit'
            )
        plt.yscale("log")

        return

    # Method to plot the signal injection result
    def plot_inject(self, log: bool = False):
        """
        Function that uses the parameters str_min and str_step as well as the result sigma_ar to generate a plot.

        Argument :
            log :
                Boolean specifying if the plot must be in log scale with respect to x axis.
                Default to False.
        """

       # Get the x-values (signal strength)
        if self.str_scale == "lin":
            sig_str = np.arange(
                self.str_min,
                self.signal_ratio + self.str_step,
                step=self.str_step,
            )
        else:
            sig_str = np.array([
                    i % 10 * 10 ** (self.str_min + i // 10)
                    for i in range(len(self.sigma_ar) + len(self.sigma_ar) // 10 + 1)
                    if i % 10 != 0
                ])

        # Do the plot
        plt.errorbar(
            sig_str,
            self.sigma_ar[:, 0],
            xerr=0,
            yerr=[self.sigma_ar[:, 1], self.sigma_ar[:, 2]],
            marker='o',
            lw=2,
        )
        if log:
            plt.xscale('log')

    # Method to obtained a printable string containing all the results of the last BumpHunter scans
    def bump_info(self, data, is_hist: bool=False):
        """
        Method that return a formated string with all the results of the last performed scan.

        Arguments :
            data :
                A DataHandler instance containing the data and reference histograms.

        Return :
            bstr :
                The formated result string.
        """

        # Check if we have results for simple scan or signal injection
        if self.t_ar.shape[0] == self.npe + 1:
            prt_inject = False
        elif self.t_ar.shape[0] == self.npe + self.npe_inject:
            prt_inject = True

        # Compute bump edges
        if prt_inject:
            # bkg+signal distributions
            Bminx = np.array([
                data.bins[ch][0][self.min_loc_ar[:self.npe_inject, 0, ch]]
                for ch in range(data.nchan)
            ])
            Bmaxx = np.array([
                data.bins[ch][0][self.min_loc_ar[:self.npe_inject, 0, ch] + self.min_width_ar[:self.npe_inject, 0, ch]]
                for ch in range(data.nchan)
            ])
            Bminy = np.array([
                data.bins[ch][1][self.min_loc_ar[:self.npe_inject, 1, ch]]
                for ch in range(data.nchan)
            ])
            Bmaxy = np.array([
                data.bins[ch][1][self.min_loc_ar[:self.npe_inject, 1, ch] + self.min_width_ar[:self.npe_inject, 1, ch]]
                for ch in range(data.nchan)
            ])

            # Bump mean and width for all channels
            Bmeanx = (Bminx + Bmaxx) / 2
            Bwidthx = Bmaxx - Bminx
            Bmeany = (Bminy + Bmaxy) / 2
            Bwidthy = Bmaxy - Bminy

            # Get the combined bumps if needed
            Bmincx = Bminx.max(axis=0)
            Bmaxcx = Bmaxx.min(axis=0)
            Bmeancx = Bmeancx = (Bmaxcx + Bmincx) / 2
            Bwidthcx = Bmaxcx - Bmincx
            Bmincy = Bminy.max(axis=0)
            Bmaxcy = Bmaxy.min(axis=0)
            Bmeancy = Bmeancy = (Bmaxcy + Bmincy) / 2
            Bwidthcy = Bmaxcy - Bmincy
        else:
            # One single data distribution
            Bminx = np.array([
                data.bins[ch][0][self.min_loc_ar[0, 0, ch]]
                for ch in range(data.nchan)
            ])
            Bmaxx = np.array([
                data.bins[ch][0][self.min_loc_ar[0, 0, ch] + self.min_width_ar[0, 0, ch]]
                for ch in range(data.nchan)
            ])
            Bminy = np.array([
                data.bins[ch][1][self.min_loc_ar[0, 1, ch]]
                for ch in range(data.nchan)
            ])
            Bmaxy = np.array([
                data.bins[ch][1][self.min_loc_ar[0, 1, ch] + self.min_width_ar[0, 1, ch]]
                for ch in range(data.nchan)
            ])

            # Bump mean and width for all channels
            Bmeanx = (Bminx + Bmaxx) / 2
            Bwidthx = Bmaxx - Bminx
            Bmeany = (Bminy + Bmaxy) / 2
            Bwidthy = Bmaxy - Bminy

            # Must take the common overlap window id required
            if self.check_overlap:
                Bcombx, Bcomby = self._bump_combined(data)
                if Bcombx is not None:
                    Bmincx = Bcombx[0]
                    Bmaxcx = Bcombx[1]
                    Bmeancx = (Bmaxcx + Bmincx) / 2
                    Bwidthcx = Bmaxcx - Bmincx
                    Bmincy = Bcomby[0]
                    Bmaxcy = Bcomby[1]
                    Bmeancy = (Bmaxcy + Bmincy) / 2
                    Bwidthcy = Bmaxcy - Bmincy

        # Initialise the string
        bstr = ""

        if prt_inject:
            # Append signal stregth
            bstr += "####SIGNAL INJECTION SUMMARY####\n"
            bstr += f"Signal trength : {self.signal_ratio:.3g}\n"

            # Append reached significance
            bstr += "Reached golbal significance : "
            bstr += f"{self.sigma_ar[-1][0]:.5g} - {self.sigma_ar[-1][1]:.5g} + {self.sigma_ar[-1][2]:.5g}\n"

            # Loop over channels
            for ch in range(data.nchan):
                bstr += f"Channel {ch} :\n"

                # Append Number of injected events
                bstr += f"    Number of injected events : {self.signal_min[ch]:.3g}\n"

                # Append bump position (median - 1st qartile + 3rd quartile) along x
                med = [np.median(Bmeanx[ch]), np.median(Bwidthx[ch])]
                q1 = [np.quantile(Bmeanx[ch], 0.25), np.quantile(Bwidthx[ch], 0.25)]
                q3 = [np.quantile(Bmeanx[ch], 0.75), np.quantile(Bwidthx[ch], 0.75)]
                bstr += "    Bump mean x: "
                bstr += f"{med[0]:.3g} - {med[0] - q1[0]:.3g} + {q3[0] - med[0]:.3g}\n"
                bstr += "    Bump width x: "
                bstr += f"{med[1]:.3g} - {med[1] - q1[1]:.3g} + {q3[1] - med[1]:.3g}\n"

                # Append bump position (median - 1st qartile + 3rd quartile) along y
                med = [np.median(Bmeany[ch]), np.median(Bwidthy[ch])]
                q1 = [np.quantile(Bmeany[ch], 0.25), np.quantile(Bwidthy[ch], 0.25)]
                q3 = [np.quantile(Bmeany[ch], 0.75), np.quantile(Bwidthy[ch], 0.75)]
                bstr += "    Bump mean x: "
                bstr += f"{med[0]:.3g} - {med[0] - q1[0]:.3g} + {q3[0] - med[0]:.3g}\n"
                bstr += "    Bump width x: "
                bstr += f"{med[1]:.3g} - {med[1] - q1[1]:.3g} + {q3[1] - med[1]:.3g}\n"

                # Append global p-value and significance of channel ch
                bstr += f"    global p-value : {self.global_Pval[ch]:.5g}\n"
                if self.global_Pval[ch] > 0:
                    bstr += f"    global significance : {self.significance[ch]:.5g}\n"
                else:
                    bstr += f"    global significance > {self.significance[ch]:.5g} (lower limit)\n"

                # Check if fit was done in this channel
                if self.fit_param[ch] is None:
                    bstr += "    No fit for this channel\n"
                else:
                    bstr += "    Fit parameters :\n"
                    for p, v in self.fit_param[ch].items():
                        bstr += f"        {p} = {v[0]:.4g} +- {v[1]:.5g}\n"
                    bstr += f"    Fit p-value : {self.fit_Pval[ch]:.5g}\n"
                    bstr += f"    Fit significance : {self.fit_sigma[ch]:.5g}\n"

            # Append combined bump if required
            if data.nchan > 1:
                bstr += f"Combined :\n"
                bstr += f"    Number of injected events : {self.signal_min.sum():.3g}\n"
                if self.check_overlap:
                    med = [np.median(Bmeancx), np.median(Bwidthcx)]
                    q1 = [np.quantile(Bmeancx, 0.25), np.quantile(Bwidthcx, 0.25)]
                    q3 = [np.quantile(Bmeancx, 0.75), np.quantile(Bwidthcx, 0.75)]
                    bstr += "    Combined bump mean x :"
                    bstr += f"{med[0]:.3g} - {med[0] - q1[0]:.3g} + {q3[0] - med[0]:.3g}\n"
                    bstr += "    Combined bump width x :"
                    bstr += f"{med[1]:.3g} - {med[1] - q1[1]:.3g} + {q3[1] - med[1]:.3g}\n"

                    med = [np.median(Bmeancy), np.median(Bwidthcy)]
                    q1 = [np.quantile(Bmeancy, 0.25), np.quantile(Bwidthcy, 0.25)]
                    q3 = [np.quantile(Bmeancy, 0.75), np.quantile(Bwidthcy, 0.75)]
                    bstr += "    Combined bump mean x :"
                    bstr += f"{med[0]:.3g} - {med[0] - q1[0]:.3g} + {q3[0] - med[0]:.3g}\n"
                    bstr += "    Combined bump width x :"
                    bstr += f"{med[1]:.3g} - {med[1] - q1[1]:.3g} + {q3[1] - med[1]:.3g}\n"
                else:
                    bstr += "    No combned bump edges\n"
        else:
            bstr += "####BUMP SCAN SUMMARY####\n"

            # Loop over channels
            for ch in range(data.nchan):
                bstr += f"Channel {ch} :\n"

                # Append Bump edges
                bstr += f"    Bump x edges : [{Bminx[ch]:.3g}, {Bmaxx[ch]:.3g}]"
                bstr += f" (loc={self.min_loc_ar[0, 0, ch]}, width={self.min_width_ar[0, 0, ch]})\n"
                bstr += f"    Bump mean x | width x : {Bmeanx[ch]:.3g} | {Bwidthx[ch]:.3g}\n"
                bstr += f"    Bump y edges : [{Bminy[ch]:.3g}, {Bmaxy[ch]:.3g}]"
                bstr += f" (loc={self.min_loc_ar[0, 1, ch]}, width={self.min_width_ar[0, 1, ch]})\n"
                bstr += f"    Bump mean y | width y : {Bmeany[ch]:.3g} | {Bwidthy[ch]:.3g}\n"

                # Append evavuated number of signal event
                bstr += f"    Number of signal events : {self.signal_eval[ch]:.3g}\n"

                # Append local p-value, test stat and local significance
                bstr += f"    Local p-value | test statistic : "
                bstr += f"{self.min_Pval_ar[0][ch]:.5g} | {self.t_ar[0][ch]:.4g}\n"
                bstr += f"    Local significance : {norm.ppf(1 - self.min_Pval_ar[0][ch]):.5g}\n"

                # Append global p-value and significance
                bstr += f"    global p-value : {self.global_Pval[ch]:.5g}\n"
                if self.global_Pval[ch] > 0:
                    bstr += f"    global significance : {self.significance[ch]:.5g}\n"
                else:
                    bstr += f"    global significance > {self.significance[ch]:.5g} (lower limit)\n"

                # Check if there was a fit for this channel
                if self.fit_param[ch] is None:
                    bstr += "    No fit for this channel\n"
                else:
                    bstr += "    Fit parameters :\n"
                    for p, v in self.fit_param[ch].items():
                        bstr += f"        {p} = {v[0]:.4g} +- {v[1]:.5g}\n"
                    bstr += f"    Fit p-value : {self.fit_Pval[ch]:.5g}\n"
                    bstr += f"    Fit significance : {self.fit_sigma[ch]:.5g}\n"

            # Check if several channels
            if data.nchan > 1:
                bstr += "Combined :\n"
                bstr += f"    Number of signal events : {self.signal_eval.sum():.3g}\n"
                bstr += f"    global p-value : {self.comb_Pval:.5g}\n"
                bstr += f"    global significance : {self.comb_sigma:.5g}\n"
                if self.check_overlap:
                    if Bcomb is not None:
                        bstr += f"    Combined bump x edges : [{Bmincx:.3g}, {Bmaxcx:.3g}]\n"
                        bstr += f"    Combined bump mean x | width x : {Bmeancx:.3g}, {Bwidthcx:.3g}\n"
                        bstr += f"    Combined bump y edges : [{Bmincy:.3g}, {Bmaxcy:.3g}]\n"
                        bstr += f"    Combined bump mean y | width y : {Bmeancy:.3g}, {Bwidthcy:.3g}\n"
                    else:
                        bstr += "    Overlap check failed (no consistent bump) !"
                else:
                    bstr += "    No combned bump edges\n"

        return bstr

    # end of BumpHunter2D class

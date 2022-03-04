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

        npe_inject :
            Integer specifying the number of background+signal pseudo-experiments to be generated during signal injection test.

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
        npe_inject: int=100,
        seed=None,
        use_sideband: bool=False,
        sideband_width=None,
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
                If you want to consider multiple channels with different binning, you can also give a list of array containing the bin edges of each channels.
                In case bins is a list and multi_chan is true, the algorithm will consider that binning is different for each channel.
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

            npe_inject :
                Number of background+signal pseudo-experiments to be generating during signal injection test.
                Default to 100.

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

            Npe : *Deprecated*
                Same as npe. This argument is deprecated and will be removed in future versions.

            Nworker : *Deprecated*
                Same as nworker. This argument is deprecated and will be removed in future versions.

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
        self.npe_inject = npe_inject
        self.seed = seed
        self.use_sideband = use_sideband
        self.sideband_width = sideband_width

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
                Numpy array of arrays containing all the p-values of all windows computed durring the scan.
                The numpy array as dimention (Nwidth), with Nwidth the number of window's width tested.
                Each array has dimension (Nstep), with Nstep the number of scan step for a given width (different for every value of width).

            min_Pval :
                Minimum p_value obtained durring the scan (float).

            min_loc :
                Position of the window corresponding to the minimum p-value (integer).

            min_width :
                Width of the window corresponding to the minimum p-value (integer).

            norm_scale :
                The scale factor computed with side-band normalization (float).
                If side-band normalization is not use, norm_scale is set to None.
        """

        # Remove the first/last hist bins if empty ... just to be consistant with c++
        non0 = [iii for iii in range(hist.size) if ref[iii] > 0]
        Hinf, Hsup = min(non0), max(non0) + 1

        # Check for sidebands
        if self.use_sideband:
            Vinf, Vsup = Hinf, Hsup
            if self.sideband_width is not None:
                Hinf = Hinf + self.sideband_width
                Hsup = Hsup - self.sideband_width

        # Create the results array
        res = np.empty(w_ar.size, dtype=object)
        min_Pval, min_loc = np.empty(w_ar.size), np.empty(w_ar.size, dtype=int)
        signal_eval = np.empty(w_ar.size)

        # Prepare things for side-band normalization (if needed)
        if self.use_sideband:
            ref_total = ref[Vinf:Vsup].sum()
            hist_total = hist[Vinf:Vsup].sum()
            min_scale = np.empty(w_ar.size)

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
            Nref = np.array([ref[p : p + w].sum() for p in pos], dtype=float)
            Nhist = np.array([hist[p : p + w].sum() for p in pos])

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
        if ih == 0:
            self.signal_eval = signal_eval[min_Pval.argmin()]

        min_Pval = min_Pval.min()

        # Save the other results in inner variables and return
        if ih == 0:
            self.res_ar = res
        self.min_Pval_ar[ih] = min_Pval
        self.min_loc_ar[ih] = int(min_loc)
        self.min_width_ar[ih] = int(min_width)
        if self.use_sideband and ih == 0:
            self.norm_scale = min_scale

    # Extention of the _scan_hist method to multi-channel data.
    def _scan_hist_multi(self, hist, ref, w_ar, ih: int):
        """
        Scan a distribution in multiple channel and compute the p-value associated to every scan window.

        The algorithm follows the BumpHunter algorithm extended to multiple channels.

        Arguments :
            hist :
                The list of data histogram (as obtain with the numpy.histogram function).

            ref :
                The list of reference (background) histogram (as obtain with the numpy.histogram function).

            w_ar :
                Array containing all the values of width to be tested.

            ih :
                Indice of the distribution to be scanned.
                ih=0 refers to the data distribution and ih>0 refers to the ih-th pseudo-data distribution.

        Results stored in inner variables :
            res :
                Numpy array of arrays containing all the p-values of all windows computed durring the scan.
                The numpy array as dimention (Nchan, Nwidth), with Nchan the number of channels and Nwidth the number of window's width tested.
                Each array has dimension (Nstep), with Nstep the number of scan step for a given width (different for every value of width).

            min_Pval :
                Minimum p_value obtained durring the scan (float).

            min_loc :
                Position of the window corresponding to the minimum p-value (integer).

            min_width :
                Width of the window corresponding to the minimum p-value (integer).

            norm_scale :
                The scale factor computed with side-band normalization (float).
                If side-band normalization is not use, norm_scale is set to None.
        """

        # Remove the first/last hist bins if empty ... just to be consistant with c++
        # Different Hinf and Hsup for each channel
        Hinf = []
        Hsup = []
        for ch in range(len(hist)):
            non0 = [iii for iii in range(0,ref[ch].size) if ref[ch][iii] > 0]
            Hinf.append(min(non0))
            Hsup.append(max(non0) + 1)
        Hinf = np.array(Hinf)
        Hsup = np.array(Hsup)

        # Check for sidebands
        if self.use_sideband:
            Vinf, Vsup = Hinf.copy(), Hsup.copy()
            if self.sideband_width is not None:
                Hinf = Hinf + self.sideband_width
                Hsup = Hsup - self.sideband_width


        # Initialize the global results for all channels
        min_loc_all = Hinf
        min_width_all = Hsup
        min_Pval_all = np.full(len(hist), 1.0, dtype=float)
        signal_eval_all = np.full(len(hist), 0.0)

        # Compute the total number of event for sideband normalization
        if self.use_sideband:
            ref_total = []
            hist_total = []
            for ch in range(len(hist)):
                ref_total.append(ref[ch][Vinf[ch]:Vsup[ch]].sum())
                hist_total.append(hist[ch][Vinf[ch]:Vsup[ch]].sum())
            min_scale_all = np.empty(len(hist))

        # Compute scan_stepp for all width
        if self.scan_step == "full":
            scan_stepp = w_ar
        elif self.scan_step == "half":
            scan_stepp = [max(1, w // 2) for w in w_ar]
        else:
            scan_stepp = np.full(w_ar.size, self.scan_step)

        # Compute pos for all width
        pos = []
        for ch in range(len(hist)):
            pos.append([
                np.arange(Hinf[ch], Hsup[ch] - w + 1, scan_stepp[i])
                for i, w in enumerate(w_ar)
            ])

        # Initialize p-value container for all channels, width and pos
        res_all = np.empty((len(hist), w_ar.size), dtype=object)

        # Loop over channels
        for ch in range(len(hist)):
            # Initialize results containers for all width
            min_Pval_current = np.empty(w_ar.size)
            min_loc_current = np.empty(w_ar.size, dtype=int)
            if self.use_sideband:
                min_scale_current = np.empty(w_ar.size)

            # Loop over widths
            for i, w in enumerate(w_ar):
                # Check that there is at least one interval to check for width w
                # If not, we must set dummy values in order to avoid crashes
                if pos[ch][i].size == 0:
                    res_all[ch, i] = np.array([1.0])
                    min_Pval_current[i] = 1.0
                    min_loc_current[i] = Hinf
                    continue

                # Count events in all intervals for channel ch and width w
                Nref = np.array(
                    [ref[ch][p : p + w].sum() for p in pos[ch][i]],
                    dtype=float
                )
                Nhist = np.array(
                    [hist[ch][p : p + w].sum() for p in pos[ch][i]]
                )

                # Compute and apply side-band normalization scale factor (if needed)
                if self.use_sideband:
                    scale = (hist_total[ch] - Nhist) / (ref_total[ch] - Nref)
                    Nref *= scale

                # Initialize a p-value container for this channel and width
                res = np.ones(Nref.size)

                # Compute all local p-values for width w
                if self.mode == "excess":
                    res[(Nhist > Nref) & (Nref > 0)] = G(
                        Nhist[(Nhist > Nref) & (Nref > 0)],
                        Nref[(Nhist > Nref) & (Nref > 0)],
                    )
                elif self.mode == "deficit":
                    res[Nhist < Nref] = 1.0 - G(
                        Nhist[Nhist < Nref] + 1, Nref[Nhist < Nref]
                    )

                # Prevent issue with very low p-value, sometimes induced by normalisation in the tail
                if self.use_sideband:
                    res[res < 1e-300] = 1e-300

                # Save all local p-values for this channel and width
                res_all[ch, i] = res

                # Save/update results for width w
                min_Pval_current[i] = res.min()
                min_loc_current[i] = pos[ch][i][res.argmin()]
                if self.use_sideband:
                    min_scale_current[i] = scale[res.argmin()]

            # Get the best interval for channel ch
            min_loc_current = min_loc_current[min_Pval_current.argmin()]
            min_width_current = w_ar[min_Pval_current.argmin()]
            if self.use_sideband:
                min_scale_current = min_scale_current[min_Pval_current.argmin()]
            min_Pval_current = min_Pval_current.min()

            # Define the combination
            if ch == 0:
                min_Pval_all[ch] = min_Pval_current
                min_loc_all[ch] = min_loc_current
                min_width_all[ch] = min_width_current
                if self.use_sideband:
                    min_scale_all[ch] = min_scale_current
            else:
                # Get the right limit of the bump
                loc_right = min_loc_current + min_width_current
                loc_right_prev = min_loc_all[ch-1] + min_width_all[ch-1]

                # Check for overlap
                if self.bins[ch][loc_right] <= self.bins[ch-1][min_loc_all[ch-1]] \
                or self.bins[ch][min_loc_current] >= self.bins[ch-1][loc_right_prev]:
                    # No overlap, we can break the loop
                    min_Pval_all = np.full(len(ref), 1)
                    min_loc_all = Hinf
                    min_width_all = Hsup
                    signal_eval_all = np.full(len(ref), 0)
                    if self.use_sideband:
                        min_scale_all = None
                    break
                else:
                    # There is an overlap, we can update the global results
                    min_Pval_all[ch] = min_Pval_current

                    # Compute overlap interval (check left bound)
                    if self.bins[ch][min_loc_current] < self.bins[ch-1][min_loc_all[ch-1]]:
                        while self.bins[ch][min_loc_current] < self.bins[ch-1][min_loc_all[ch-1]]:
                            min_loc_current += 1
                        min_loc_current -= min_loc_current - 1
                    # Check right bound
                    if self.bins[ch][loc_right] > self.bins[ch-1][loc_right_prev]:
                        while self.bins[ch][loc_right] > self.bins[ch-1][loc_right_prev]:
                            loc_right -= 1
                        loc_right +=1
                    # Width
                    min_loc_all[ch] = min_loc_current
                    min_width_all[ch] = loc_right - min_loc_all[ch]
                    
                    # Side-band normalization scale
                    if self.use_sideband:
                        min_scale_all[ch] = min_scale_current

        # Use best inverval position and width to compute signal_eval_all
        if ih == 0 and min_Pval_all[-1] < 1:
            signal_eval_all = np.array([
                hist[ch][min_loc_all[ch] : min_loc_all[ch] + min_width_all[ch]].sum() \
                - ref[ch][min_loc_all[ch] : min_loc_all[ch]  + min_width_all[ch]].sum()
                for ch in range(len(hist))
            ])

        # Save the results in inner variables and return
        if ih == 0:
            self.res_ar = res_all
            self.signal_eval = signal_eval_all
        self.min_Pval_ar[ih] = min_Pval_all
        self.min_loc_ar[ih] = np.array(min_loc_all, dtype=int)
        self.min_width_ar[ih] = np.array(min_width_all, dtype=int)
        self.t_ar[ih] = -np.log(min_Pval_all.prod())
        if self.use_sideband and ih == 0:
            self.norm_scale = min_scale_all


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
        state["npe_inject"] = self.npe_inject
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
        state["norm_scale"] = self.norm_scale
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

        if "rang" in state:
            self.rang = state["rang"]
        else:
            self.rang = None

        if "bins" in state:
            self.bins = state["bins"]
        else:
            self.bins = 60

        if "weights" in state:
            self.rang = state["weights"]
        else:
            self.rang = None

        if "width_min" in state:
            self.width_min = state["width_min"]
        else:
            self.width_min = 2

        if "width_max" in state:
            self.width_max = state["width_max"]
        else:
            self.width_max = None

        if "width_step" in state:
            self.width_step = state["width_step"]
        else:
            self.width_step = 1

        if "scan_step" in state:
            self.scan_step = state["scan_step"]
        else:
            self.scan_step = 1

        if "npe" in state:
            self.npe = state["npe"]
        else:
            self.npe = 100

        if "nworker" in state:
            self.nworker = state["nworker"]
        else:
            self.nworker = 4

        if "seed" in state:
            self.seed = state["seed"]
        else:
            self.seed = None

        if "use_sideband" in state:
            self.use_sideband = state["use_sideband"]
        else:
            self.use_sideband = False

        if "sigma_limit" in state:
            self.sigma_limit = state["sigma_limit"]
        else:
            self.sigma_limit = 5

        if "str_min" in state:
            self.str_min = state["str_min"]
        else:
            self.str_min = 0.5

        if "str_step" in state:
            self.str_step = state["str_step"]
        else:
            self.str_step = 0.25

        if "str_scale" in state:
            self.str_scale = state["str_scale"]
        else:
            self.str_scale = "lin"

        if "signal_exp" in state:
            self.signal_exp = state["signal_exp"]
        else:
            self.signal_exp = None

        if "sig_flip" in state:
            self.sig_flip = state["sig_flip"]
        else:
            self.sig_flip = True

        if "npe_inject" in state:
            self.npe_inject = state["npe_inject"]
        else:
            self.npe_inject = 100

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
        if "signal_min" in state:
            self.signal_min = state["signal_min"]
        if "signal_ratio" in state:
            self.signal_ratio = state["signal_ratio"]
        if "data_inject" in state:
            self.data_inject = state["data_inject"]

        return

    @deprecated("Use `load_state` instead.")
    def LoadState(self, *args, **kwargs):
        return self.load_state(*args, **kwargs)

    ## Scan methods

    # Method that perform the scan on every pseudo experiment and data (in parrallel threads).
    # For each scan, the value of p-value and test statistic t is computed and stored in result array
    def bump_scan(self, data, bkg, is_hist: bool=False, do_pseudo: bool=True, multi_chan: bool=False):
        """
        Function that perform the full BumpHunter algorithm presented in https://arxiv.org/pdf/1101.0390.pdf without sidebands.
        This includes the generation of pseudo-data, the calculation of the BumpHunter p-value associated to data and to all pseudo experiment as well as the calculation of the test satistic t.

        The results are stored in the inner result variables of this BumpHunter instance.

        Arguments :
            data :
                The data distribution.
                If there is only one channel, it should be a numpy array containing the data distribution.
                Otherwise, it should be a list of numpy arrays (one per channels).
                The distribution(s) will be transformed into a binned histogram and the algorithm will look for the most significant excess.

            bkg :
                The reference background distribution.
                If there is only one channel, it should be a numpy array containing the reference background distribution.
                Otherwise, it should be a list of numpy arrays (one per channels).
                The distribution(s) will be transformed into a binned histogram and the algorithm will compare it to data while looking for a bump.

            is_hist :
                Boolean that specify if the given data and background are already in histogram form.
                If true, the data and backgrouns are considered as already 'histogramed'.
                Default to False.

            do_pseudo :
                Boolean specifying if pesudo data should be generated.
                If False, then the BumpHunter statistics distribution kept in memory is used to compute the global p-value and significance.
                If there is nothing in memory, the global p-value and significance will not be computed.
                Default to True.

            multi_chan :
                Boolean specifying if there are multiple channels.
                Default to False.

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

        # If we are in multi channel, we must check if bins is given separately for each channel
        if multi_chan:
            if not isinstance(self.bins, list):
                self.bins = [self.bins for ch in range(len(data))]

        # Generate the background and data histograms
        print("Generating histograms")
        if multi_chan:
            data_hist = []
            bkg_hist = []
            bins = []
            for ch in range(len(data)):
                if not is_hist:
                    bkg_hist.append(np.histogram(
                        bkg[ch],
                        bins=self.bins[ch],
                        weights=self.weights,
                        range=self.rang
                    )[0])
                    bins.append(np.histogram_bin_edges(
                        bkg[ch],
                        bins=self.bins[ch],
                        weights=self.weights,
                        range=self.rang
                    ))
                    data_hist.append(
                        np.histogram(data[ch],
                        bins=bins[ch],
                        range=self.rang
                    )[0])
                else:
                    if self.weights is None:
                        bkg_hist.append(bkg[ch])
                    else:
                        bkg_hist.append(bkg[ch] * self.weights[ch])
                    data_hist.append(data[ch])
        else:
            if not is_hist:
                bkg_hist, bins = np.histogram(bkg, bins=self.bins, weights=self.weights, range=self.rang)
                data_hist = np.histogram(data, bins=bins, range=self.rang)[0]
            else:
                if self.weights is None:
                    bkg_hist = bkg
                else:
                    bkg_hist = bkg * self.weights
                data_hist = data

        # If data/bkg is not given as binned histogram, we must set self.bins to bin edges
        if not is_hist:
            self.bins = bins
            del bins

        # Generate all the pseudo-data histograms
        if do_pseudo:
            # Check if we have multiple channels
            if multi_chan:
                # loop over channels
                pseudo_hist = []
                for ch in range(len(data)):
                    pseudo_hist.append(
                            np.random.poisson(
                            lam=np.tile(bkg_hist[ch], (self.npe, 1)).transpose(),
                            size=(bkg_hist[ch].size, self.npe),
                        )
                    )
                # Convert the list into a 3D numpy array
                #pseudo_hist = np.array(pseudo_hist)
                

            else:
                pseudo_hist = np.random.poisson(
                    lam=np.tile(bkg_hist, (self.npe, 1)).transpose(),
                    size=(bkg_hist.size, self.npe),
                )

        # Set width_max if it is given as None
        if self.width_max is None:
            if multi_chan:
                self.width_max = data_hist[0].size // 2
            else:
                self.width_max = data_hist.size // 2

        # Initialize all results containenrs
        if multi_chan:
            if do_pseudo:
                self.min_Pval_ar = np.empty(self.npe + 1, dtype=object)
                self.min_loc_ar = np.empty(self.npe + 1, dtype=object)
                self.min_width_ar = np.empty(self.npe + 1, dtype=object)
                self.t_ar = np.empty(self.npe + 1)
            else:
                if self.min_Pval_ar == []:
                    self.min_Pval_ar = np.empty(1)
                    self.min_loc_ar = np.empty(1, dtype=int)
                    self.min_width_ar = np.empty(1, dtype=int)
                    self.t_ar = np.empty(1)
        else:
            if do_pseudo:
                self.min_Pval_ar = np.empty(self.npe + 1)
                self.min_loc_ar = np.empty(self.npe + 1, dtype=int)
                self.min_width_ar = np.empty(self.npe + 1, dtype=int)
                self.t_ar = np.empty(self.npe + 1)
            else:
                if self.min_Pval_ar == []:
                    self.min_Pval_ar = np.empty(1)
                    self.min_loc_ar = np.empty(1, dtype=int)
                    self.min_width_ar = np.empty(1, dtype=int)
                    self.t_ar = np.empty(1)
        self.res_ar = []

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
                    for th in range(self.npe + 1):
                        if multi_chan:
                            if th == 0:
                                exe.submit(
                                    self._scan_hist_multi,
                                    data_hist,
                                    bkg_hist,
                                    w_ar,
                                    th,
                                )
                            else:
                                pseudo = [
                                    pseudo_hist[ch][:, th-1]
                                    for ch in range(len(data))
                                ]
                                exe.submit(
                                    self._scan_hist_multi,
                                    pseudo,
                                    bkg_hist,
                                    w_ar,
                                    th,
                                )
                        else:
                            if th == 0:
                                exe.submit(
                                    self._scan_hist,
                                    data_hist,
                                    bkg_hist,
                                    w_ar,
                                    th,
                                )
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
                    if multi_chan:
                        if i == 0:
                            self._scan_hist_multi(
                                data_hist,
                                bkg_hist,
                                w_ar,
                                i
                            )
                        else:
                            pseudo = [
                                pseudo_hist[ch][:, i-1]
                                for ch in range(len(data))
                            ]
                            self._scan_hist_multi(
                                pseudo,
                                bkg_hist,
                                w_ar,
                                i
                            )
                    else:
                        if i == 0:
                            self._scan_hist(
                                data_hist,
                                bkg_hist,
                                w_ar,
                                i
                            )
                        else:
                            self._scan_hist(
                                pseudo_hist[:, i - 1],
                                bkg_hist,
                                w_ar,
                                i
                            )
        else:
            if multi_chan:
                self._scan_hist_multi(data_hist, bkg_hist, w_ar, 0)
            else:
                self._scan_hist(data_hist, bkg_hist, w_ar, 0)

        # Use the p-value results to compute t
        if not multi_chan:
            self.t_ar = -np.log(self.min_Pval_ar)

        # Compute the global p-value from the t distribution
        if self.t_ar.size > 1:
            tdat = self.t_ar[0]
            S = self.t_ar[1:][self.t_ar[1:] >= tdat].size
            self.global_Pval = S / self.npe
            print(f"Global p-value : {self.global_Pval:1.4f}  ({S} / {self.npe})")
    
            # Check global p-value
            if self.global_Pval == 1:
                self.significance = 0
                print(f"Significance = {self.significance}")
            elif self.global_Pval == 0:
                # I this case, we can't compute directly the significance, so we set a limit
                self.significance = norm.ppf(1 - (1 / self.npe))
                print(f"Significance > {self.significance:1.5f} (lower limit)")
            else:
                self.significance = norm.ppf(1 - self.global_Pval)
                print(f"Significance = {self.significance:1.5f}")
        else:
            print("No pseudo data found : can't compute global p-value")
        print("")

        return

    @deprecated("Use `bump_scan` instead.")
    def BumpScan(self, *args, **kwargs):
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
        self.global_Pval = 1
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
        np.random.seed(self.seed)
        pseudo_bkg = np.random.poisson(
            lam=np.tile(bkg_hist, (self.npe, 1)).transpose(), size=(bkg_hist.size, self.npe)
        )

        # Set width_max if it is given as None
        if self.width_max is None:
            self.width_max = bkg_hist.size // 2

        # Initialize all results containenrs
        self.min_Pval_ar = np.empty(self.npe)
        self.min_loc_ar = np.empty(self.npe, dtype=int)
        self.min_width_ar = np.empty(self.npe, dtype=int)
        self.res_ar = np.empty(self.npe, dtype=object)

        # Auto-adjust the value of width_max and do an array of all width
        w_ar = np.arange(self.width_min, self.width_max + 1, self.width_step)
        self.width_max = w_ar[-1]
        print(f"{w_ar.size} values of width will be tested")

        # Compute the p-value for background only pseudo-experiments
        # We must check if we should do it in multiple threads
        print("BACKGROUND ONLY SCAN")
        if self.nworker > 1:
            with thd.ThreadPoolExecutor(max_workers=self.nworker) as exe:
                for th in range(self.npe):
                    exe.submit(self._scan_hist, pseudo_bkg[:, th], bkg_hist, w_ar, th)
        else:
            for th in range(self.npe):
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
        while (self.significance < self.sigma_limit)\
        and (self.global_Pval > 1 / self.npe):
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
                lam=np.tile(data_hist, (self.npe_inject, 1)).transpose(),
                size=(data_hist.size, self.npe_inject),
            )

            # Initialize all results containenrs
            self.min_Pval_ar = np.empty(self.npe_inject)
            self.min_loc_ar = np.empty(self.npe_inject, dtype=int)
            self.min_width_ar = np.empty(self.npe_inject, dtype=int)
            self.res_ar = np.empty(self.npe_inject, dtype=object)

            # Compute the p-value for background+signal pseudo-experiments
            # We must check if we should do it in multiple threads
            print("BACKGROUND+SIGNAL SCAN")
            if self.nworker > 1:
                with thd.ThreadPoolExecutor(max_workers=self.nworker) as exe:
                    for th in range(self.npe_inject):
                        exe.submit(
                            self._scan_hist, pseudo_data[:, th], bkg_hist, w_ar, th
                        )
            else:
                for th in range(self.npe_inject):
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
            if self.global_Pval <  1 / self.npe:
                self.significance = norm.ppf(1 - (1 / self.npe))
            else:
                self.significance = norm.ppf(1 - self.global_Pval)

            if global_inf <  1 / self.npe:
                sigma_inf = norm.ppf(1 - (1 / self.npe))
            else:
                sigma_inf = norm.ppf(1 - global_inf)

            if global_sup <  1 / self.npe:
                sigma_sup = norm.ppf(1 - (1 / self.npe))
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
        # Check stop condition
        if self.significance > self.sigma_limit:
            print("REACHED SIGMA LIMIT")
        elif self.global_Pval <= 1 / self.npe:
            print(f"REACHED STAT LIMIT AT {self.significance:.3f} SIGMA")
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
    def plot_tomography(self, bkg, is_hist: bool=False, label: str='' , filename=None, chan: int=0):
        """
        Function that do a tomography plot showing the local p-value for every positions and widths of the scan
        window.

        If there are multiple channels, you can specify which channel to consider in the plot.

        Arguments :
            bkg :
                Numpy array, or list of numpy arrays, containing the reference background.

            is_hist :
                Boolean specifying if data is in histogram form or not. Default to False.

            label :
                Extra label to be added to the plot title given as a string.
                Default to '' (empty string).

            filename :
                Name of the file in which the plot will be saved.
                If None, the plot will be just shown but not saved.
                Default to None.

            chan :
                Specify the number of the channel to be shown (if there are more than one).
                Ignored if there is only one channel.
                Default to 0 (the first channel).
        """

        # Check if there is anything to show.
        if self.res_ar == []:
            print('Nothing to plot here !')
            return

        # Check if we have multiple channels
        if self.res_ar.ndim == 2:
            multi_chan = True
        else:
            multi_chan = False

        # Get the reference histogram
        if multi_chan:
            Hbkg = []
            H = []
            if not is_hist:
                for ch in range(len(bkg)):
                    Hbkg.append(np.histogram(
                        bkg[ch],
                        bins=self.bins[ch],
                        range=self.rang,
                        weights=self.weights
                    )[0])
                    H.append(np.histogram_bin_edges(
                        bkg[ch],
                        bins=self.bins[ch],
                        range=self.rang
                    ))
            else:
                H = self.bins
                if self.weights is None:
                    Hbkg = bkg
                else:
                    Hbkg = [bkg[ch] * self.weights[ch] for ch in range(len(bkg))]
        else:
            if not is_hist:
                Hbkg, H = np.histogram(
                    bkg, bins=self.bins, range=self.rang, weights=self.weights
                )
            else:
                H = self.bins
                if self.weights is None:
                    Hbkg = bkg
                else:
                    Hbkg = bkg * self.weights

        # Remove empty bins at the begining of reference
        Hinf = 0
        if multi_chan:
            non0 = [i for i in range(0,Hbkg[chan].size) if Hbkg[chan][i] > 0]
            Hinf = min(non0)
        else:
            non0 = [i for i in range(Hbkg.size) if Hbkg[i] > 0]
            Hinf = min(non0)

        # Select the required channel (if needed)
        if multi_chan:
            res_data = self.res_ar[chan,:]
            H = H[chan]
        else:
            res_data = self.res_ar

        # Get all width in number of bins
        w_ar = np.arange(self.width_min, self.width_max + 1, self.width_step)

        # Loop over width
        inter = []
        for i in range(res_data.size):
            # Get scan step for width w
            if self.scan_step == "half":
                scan_stepp = max(1, w_ar[i] // 2)
            elif self.scan_step == "full":
                scan_stepp = w_ar[i]
            else:
                scan_stepp = self.scan_step

            # Loop over positions
            for j in range(len(res_data[i])):
                loc = H[j * scan_stepp + Hinf]
                w = H[j * scan_stepp + Hinf + w_ar[i]] - loc
                inter.append([res_data[i][j], loc, w])

        F = plt.figure(figsize=(12, 8))
        plt.title(label, size="xx-large")
        [plt.plot([i[1], i[1] + i[2]], [i[0], i[0]], "r") for i in inter if i[0] < 1.0]
        plt.xlabel("intervals", size="xx-large")
        plt.ylabel("local p-value", size="xx-large")
        plt.yscale("log")
        plt.xticks(fontsize="xx-large")
        plt.yticks(fontsize="xx-large")

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
    def plot_bump(self, data, bkg, is_hist: bool=False, use_sideband=None, label: str='', filename=None, chan: int=0, useSideBand=None):
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

            label :
                Extra label to be added to the plot title given as a string.
                Default to '' (empty string).

            filename :
                Name of the file in which the plot will be saved.
                If None, the plot will be just shown but not saved.
                Default to None.

            chan :
                Specify the number of the channel to be shown (if there are more than one).
                Ignored if there is only one channel.
                Default to 0 (the first channel).

            useSideBand : *Deprecated*
                Same as use_sideband. This argument is deprecated and will be removed in a future version.
        """

        # legacy deprecation
        if useSideBand is not None:
            use_sideband = useSideBand

        # Check if there are multiple channels
        if self.res_ar.ndim == 2:
            multi_chan = True
        else:
            multi_chan = False

        # Get the data in histogram form
        if multi_chan:
            if not is_hist:
                # Take the histogram bin content for the required channel
                H = np.histogram(data[chan], bins=self.bins[chan], range=self.rang)
                    
                H = [H[0], H[1]]
            else:
                H = np.histogram(data[chan], bins=self.bins, range=self.rang)
        else:
            if not is_hist:
                H = np.histogram(data, bins=self.bins, range=self.rang)
            else:
                H = [data, self.bins]

        # Get bump min and max
        if multi_chan:
            Bmin = np.array([
                H[1][self.min_loc_ar[0][ch]]
                for ch in range(len(data))
            ])
            Bmax = np.array([
                H[1][self.min_loc_ar[0][ch] + self.min_width_ar[0][ch]]
                for ch in range(len(data))
            ])
            Bmin = Bmin.max()
            Bmax = Bmax.min()
        else:
            Bmin = H[1][self.min_loc_ar[0]]
            Bmax = H[1][self.min_loc_ar[0] + self.min_width_ar[0]]

        # Get the background in histogram form
        if multi_chan:
            if not is_hist:
                Hbkg = np.histogram(
                    bkg[chan],
                    bins=self.bins[chan],
                    range=self.rang,
                    weights=self.weights
                )[0]
            else:
                if self.weights is None:
                    Hbkg = bkg[chan]
                else:
                    Hbkg = bkg[chan] * self.weights
        else:
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
            # Check if we are in multi-channel
            if multi_chan:
                Hbkg = Hbkg * self.norm_scale[chan]
            else:
                Hbkg = Hbkg * self.norm_scale

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
        plt.title(f"Distributions with bump  {label}", size="xx-large")

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

        plt.vlines(
            [Bmin, Bmax],
            0,
            H[0].max(),
            colors="r",
            linestyles='dashed',
            label="BUMP"
        )
        plt.legend(fontsize="xx-large")
        plt.yscale("log")
        if self.rang is not None:
            plt.xlim(self.rang)
        plt.xticks(fontsize="xx-large")
        plt.yticks(fontsize="xx-large")
        plt.tight_layout()

        plt.subplot(gs[1], sharex=pl1)
        plt.hist(H[1][:-1], bins=H[1], range=self.rang, weights=sig)
        plt.plot(np.full(2, Bmin), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
        plt.plot(np.full(2, Bmax), np.array([sig.min(), sig.max()]), "r--", linewidth=2)
        plt.yticks(
            np.arange(np.round(sig.min()), np.round(sig.max()) + 1, step=1),
            fontsize="xx-large",
        )
        plt.ylabel("significance", size="xx-large")
        plt.xticks(fontsize="xx-large")

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
                f"BumpHunter statistics distribution      global p-value = {self.global_Pval:1.4f}",
                size="xx-large"
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
        plt.legend(fontsize="xx-large")
        plt.xlabel("BumpHunter statistic", size="xx-large")
        plt.yscale("log")
        plt.xticks(fontsize="xx-large")
        plt.yticks(fontsize="xx-large")

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

        # Compute significance saturation masks
        is_sat = self.sigma_ar[:, 2] == 0

        # If filename is not None and log scale must check
        if filename is not None and self.str_scale == "log":
            if isinstance(filename, str):
                print("WARNING : log plot for signal injection will not be saved !")
                nolog = True
            else:
                nolog = False

        # Do the plot
        F = plt.figure(figsize=(12, 8))
        plt.title("Significance vs signal strength", size="xx-large")
        plt.errorbar(
            sig_str,
            self.sigma_ar[:, 0],
            xerr=0,
            yerr=[self.sigma_ar[:, 1], self.sigma_ar[:, 2]],
            marker='o',
            linewidth=2,
            uplims=is_sat
        )
        plt.xlabel("Signal strength", size="xx-large")
        plt.ylabel("Significance", size="xx-large")
        plt.xticks(fontsize="xx-large")
        plt.yticks(fontsize="xx-large")

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
            plt.title("Significance vs signal strength (log scale)", size="xx-large")
            plt.errorbar(
                sig_str,
                self.sigma_ar[:, 0],
                xerr=0,
                yerr=[self.sigma_ar[:, 1], self.sigma_ar[:, 2]],
                marker='o',
                linewidth=2,
                uplims=is_sat
            )
            plt.xlabel("Signal strength", size="xx-large")
            plt.ylabel("Significance", size="xx-large")
            plt.xscale("log")
            plt.xticks(fontsize="xx-large")
            plt.yticks(fontsize="xx-large")

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

    # Method to obtained a printable string containing all the results of the last BumpHunter scans
    def bump_info(self, data, is_hist: bool=False):
        """
        Method that return a formated string with all the results of the last performed scan.

        Arguments :
            data :
                Numpy array containing the data.

        Return :
            bstr :
                The formated result string.
        """

        # Check if we are in multi_chan
        if self.res_ar != [] and self.res_ar.ndim == 2:
            multi_chan = True
        else:
            multi_chan = False
        

        # Get the bin edges
        if not is_hist:
            if multi_chan:
                # Loop over channel
                bins = []
                for ch in range(len(data)):
                    bins.append(np.histogram_bin_edges(
                        data[ch],
                        bins=self.bins[ch],
                        range=self.rang
                    ))
            else:
                bins = np.histogram_bin_edges(data, bins=self.bins, range=self.rang)
        else:
            bins = self.bins

        # Compute bump edges
        if multi_chan:
            Bmin = np.array([
                bins[ch][self.min_loc_ar[0][ch]]
                for ch in range(len(data))
            ])
            Bmax = np.array([
                bins[ch][self.min_loc_ar[0][ch] + self.min_width_ar[0][ch]]
                for ch in range(len(data))
            ])

            # Must take the common overlap window
            Bminc = Bmin.max()
            Bmaxc = Bmax.min()
            Bmean = (Bmaxc + Bminc) / 2
            Bwidth = Bmaxc - Bminc
        else:
            Bmin = bins[self.min_loc_ar[0]]
            Bmax = bins[self.min_loc_ar[0] + self.min_width_ar[0]]
            Bmean = (Bmax + Bmin) / 2
            Bwidth = Bmax - Bmin

        # Initialise the string
        bstr = ''

        # Append local results to the string
        if multi_chan:
            # Append the bump edges of every channels
            bstr += 'Bump edges (per channel):\n'
            for ch in range(len(self.min_Pval_ar[0])):
                bstr += f'    chan {ch+1} -> [{Bmin[ch]:.3g}, {Bmax[ch]:.3g}]'
                bstr += f'  (loc={self.min_loc_ar[0][ch]}, width={self.min_width_ar[0][ch]})\n'

            # Append the combined bump edges, mean and width
            bstr += f'Combined bump edges : [{Bminc:.3g}, {Bmaxc:.3g}]\n'
            bstr += f'Combined bump mean | width : {Bmean:.3g} | {Bwidth:.3g}\n'

            # Append evavuated number of signal event (per channel and total)
            bstr += 'Evaluated number of signal events (per channel):\n'
            for ch in range(len(self.min_Pval_ar[0])):
                bstr += f'    chan {ch+1} -> {self.signal_eval[ch]:.3g}\n'
            bstr += f'    Total -> {self.signal_eval.sum():.3g}\n'

            # Append local information
            bstr += 'Local p-value (per channel):\n'
            for ch in range(len(self.min_Pval_ar[0])):
                bstr += f'    chan {ch+1} -> {self.min_Pval_ar[0][ch]:.5g}\n'
            bstr += f'Local p-value | test statistic (combined) : {self.min_Pval_ar[0].prod():.5g}'
            bstr += f' | {self.t_ar[0]:.5g}\n'
            bstr += f'Local significance (combined) : {norm.ppf(1 - self.min_Pval_ar[0].prod()):.5g}\n'
        else:
            # Append results for only one channel (no 'combined')
            bstr += f'Bump edges : [{Bmin:.3g}, {Bmax:.3g}]'
            bstr += f'  (loc={self.min_loc_ar[0]}, width={self.min_width_ar[0]})\n'
            bstr += f'Bump mean | width : {Bmean:.3g} | {Bwidth:.3g}\n'
            bstr += f'Evaluated number of signal events : {self.signal_eval:.3g}\n'
            bstr += f'Local p-value | test statistic : {self.min_Pval_ar[0]:.5g}'
            bstr += f' | {self.t_ar[0]:.5g}\n'
            bstr += f'Local significance : {norm.ppf(1 - self.min_Pval_ar[0]):.5g}\n'

        # Append global results to the string
        bstr += f'Global p-value : {self.global_Pval:.5g}\n'
        if self.global_Pval == 0:
            bstr += f'Global significance : >{self.significance:.3g}  (lower limit)'
        else:
            bstr += f'Global significance : {self.significance:.3g}'

        return bstr

    # Method that print the local infomation about the most significante bump in data
    @deprecated("Use `bump_info` instead.")
    def print_bump_info(self):
        """
        Function that print the local infomation about the most significante bump in data.
        Information are printed to stdout.
        """

        # Start printing stuff
        print("BUMP WINDOW")
        print(f"   loc = {self.min_loc_ar[0]}")
        print(f"   width = {self.min_width_ar[0]}")

        # Check if there are multiple channels
        if not isinstance(self.min_Pval_ar[0], np.ndarray):
            # Keep printing stuff for one channel
            print(
                f"   local p-value = {self.min_Pval_ar[0]:.5g}"
            )
            print(
                f"   -ln(loc p-value) = {self.t_ar[0]:.5f}"
            )
            print(
                f"   local significance = {norm.ppf(1 - self.min_Pval_ar[0]):.5f}"
            )
        else:
            # Keep printing stuff for multiple channels
            print(
                "   local p-value (per channel) = [",
                end=''
            )
            [
                print(f"{self.min_Pval_ar[0][ch]:.5g}  ",end='')
                for ch in range(len(self.min_Pval_ar[0]))
            ]
            print("]")
            print(
                f"   local p-value (combined) = {self.min_Pval_ar[0].prod():.5g}"
            )
            print(
                f"   -ln(loc p-value) (combined) = {self.t_ar[0]:.5f}"
            )
            print(
                f"   local significance (combined) = {norm.ppf(1 - self.min_Pval_ar[0].prod()):.5f}"
            )

        print("")

        return

    @deprecated("Use `print_bump_info` instead.")
    def PrintBumpInfo(self, *args, **kwargs):
        return self.print_bump_info(*args, **kwargs)

    # Function that print the global infomation about the most significante bump in data
    @deprecated("Use `bump_info` instead.")
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

        # Chek if we have multi-channel
        if self.res_ar != [] and self.res_ar.ndim == 2:
            # We have multiple channels
            multi_chan = True
        else:
            # Only a single channel
            multi_chan = False

        # Get bin edges
        if not is_hist:
            if multi_chan:
                # Loop over channel
                bins = []
                for ch in range(len(data)):
                    bins.append(np.histogram_bin_edges(
                        data[ch],
                        bins=self.bins[ch],
                        range=self.rang
                    ))
            else:
                bins = np.histogram_bin_edges(data, bins=self.bins, range=self.rang)
        else:
            bins = self.bins

        # Compute real bump edges
        if multi_chan:
            Bmin = np.array([
                bins[ch][self.min_loc_ar[0][ch]]
                for ch in range(len(data))
            ])
            Bmax = np.array([
                bins[ch][self.min_loc_ar[0][ch] + self.min_width_ar[0][ch]]
                for ch in range(len(data))
            ])

            # Must take the common overlap window
            Bmin = Bmin.max()
            Bmax = Bmax.min()
        else:
            Bmin = bins[self.min_loc_ar[0]]
            Bmax = bins[self.min_loc_ar[0] + self.min_width_ar[0]]
        Bmean = (Bmax + Bmin) / 2
        Bwidth = Bmax - Bmin

        # Print informations about the bump itself
        print("BUMP POSITION")
        print(f"   min : {Bmin:.3f}")
        print(f"   max : {Bmax:.3f}")
        print(f"   mean : {Bmean:.3f}")
        print(f"   width : {Bwidth:.3f}")
        print(f"   number of signal events : {self.signal_eval}")
        if multi_chan:
            print(f"   global p-value (combined) : {self.global_Pval:1.5f}")
            print(f"   global significance (combined) = {self.significance:1.5f}")
        else:
            print(f"   global p-value : {self.global_Pval:1.5f}")
            print(f"   global significance = {self.significance:1.5f}")
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




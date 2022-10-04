#!/usr/bin/env python

# A DataHandler class for pyBumpHunter that allows to manage histograms in a simple and efficient way.
# Can be also used as an interface to import data from several different formats (numpy, pandas DataFrame, python dict, boost histograms)

import numpy as np
import warnings

class DataHandler():
    """
    The DataHandler class provides a simple a efficient way to manage all the histograms to be used with BumpHunter1D or BumpHunter2D classes.
    It is also an interface to read data from various format.

    List of fields :
        self.dim :
            Number of dimension of the histograms (either 1 or 2)

        self.multi_chan :
            Boolean specifying if we have multiple channels

        self.ref :
            The bin yields of the reference background histogram (numpy array)

        self.hist :
            The bin yields of the data histogram (numpy array)

        self.sig :
            The bin yields of the signal histogram (numpy array)

        self.signal_exp :
            The expected number of signal event (int).
            Can be used to compute signal strength in signal injection test.

        self.bins :
            The bin edges common to all histograms (numpy array)

        self.range :
            The range common to all histograms
    """

    # Initializer
    def __init__(
        self,
        ndim:int = 1 ,
        rang=None,
        bins=None,
        nchan:int = 1
    ):
        """
        Arguments :
            ndim :
               Integer spicifying the number of dimension of the histograms
               Must be either 1 or 2, and default to 1.

            range :
                The range of the histograms.
                Default to None.

            bins :
                Define the binning fo the histograms.
                Can be either None or a list of integers or of array-likes of floats.
                The list must have a length equal to the number of channel.
                If None (default), the binning must be given when setting the first histogram.

            nchan :
                Integer specifying the number of channels
                Default to 1.
        """
        
        # Initialize variables
        self.ref = []
        self.hist = []
        self.sig = []
        self.signal_exp = 0
        self.range = rang
        self.bins = bins
        self.ndim = ndim
        self.nchan = nchan

        return

    # Unified histograming method
    def _make_hist(
        self,
        data,
        what,
        weights,
        datatype,
        is_hist
    ):
        """
        Unified histograming method.
        This method is not intended to be used directly by the front-end user.

        Arguments :
            data :
                The data given as an object which type depends on the datatype argument.

            what :
                String specifying which histogram shoulld be set.
                Can be either 'ref', 'hist' or 'sig'.

            weights :
                The events weights used to make the histogram.
                Must be compatible with the histograming backend.

            datatype :
                String specifying the type of the data.
                Supported types currently are 'np' and 'boost'.

            is_hist :
                Boolean specifying if the data is in binned format.
                Revelant only for datatype 'np'.

        Returns :
            hist_list :
                The list of all channels bin counts.
        """
        # Check if the binning is a list of the correct lenght
        if self.bins is None or isinstance(self.bins, int):
            self.bins = [self.bins for ch in range(self.nchan)]

        # Same check for data
        if not isinstance(data, list):
            data = [data for ch in range(self.nchan)]

        # Same check for the range
        if self.range is None:
            self.range = [None for ch in range(self.nchan)]
        elif isinstance(self.range, list):
            rd = np.array(self.range).ndim
            if rd == self.ndim:
                self.range = [self.range for ch in range(self.nchan)]

        # Same check for the weights
        if not isinstance(weights, list):
            weights = [weights for ch in range(self.nchan)]

        # Initialize the histograms list
        hist_list = []

        # Loop over channels
        for ch in range(self.nchan):
            # Check for datatype
            if datatype == 'boost':
                # Extract and append the histogram bin counts array
                hist, bins = data[ch].to_numpy(flow=False)
                hist_list.append(hist)

                # Check that the number of dimensions match the requirements
                if hist.ndim != self.ndim:
                    raise ValueError(f"ERROR : '{what}' histogram for channel {ch} has {hist.ndim} dimension(s) while {self.ndim} were expected !")

                # Check if self.bins is alredy set for this channel
                if isinstance(self.bins[ch], np.ndarray):
                    # Check that the histogram binning is compatible with the current channel binning
                    check = (bins == self.bins[ch]).sum()
                    if check != bins.size:
                        # There is a problem somewhere
                        raise ValueError(f"ERROR : Bin edges for '{what}' channel {ch} is not compatible with others !")

                # Set the bin adges array # TODO check for 2D bins
                self.bins = bins
            elif datatype == 'np':
                # Check if the data is alreday binned
                if is_hist:
                    # Append histogram
                    hist_list.append(data[ch])
                else:
                    # Check the number of dimensions
                    if self.ndim == 1:
                        # Make histogram for channel ch
                        hist, bins = np.histogram(
                            data[ch],
                            bins=self.bins[ch],
                            weights=weights[ch],
                            range=self.range[ch]
                        )

                        # Append histogram
                        hist_list.append(hist)

                        # Set the binning of channel ch
                        self.bins[ch] = bins
                    else:
                        hist, binx, biny = np.histogram2d(
                            data[ch][:,0],
                            data[ch][:,1],
                            bins=self.bins[ch],
                            weights=weights[ch],
                            range=self.range[ch]
                        )

                        # Append histogram
                        hist_list.append(hist)

                        # Set the binning of channel ch
                        self.bins[ch] = [binx, biny]
            else:
                raise ValueError(f"ERROR : The datatype '{datatype}' is not supported (yet) !")

            # Check if the range value is set correctly
            if self.range[ch] is None:
                if self.ndim == 1:
                    self.range[ch] = [self.bins[ch][0], self.bins[ch][-1]]
                else:
                    self.range[ch] = [
                        [self.bins[ch][0][0], self.bins[ch][0][-1]],
                        [self.bins[ch][1][0], self.bins[ch][1][-1]]
                    ]

        return hist_list

    # Method to set the reference background histogram
    def set_ref(
        self,
        data,
        datatype:str = 'np',
        rang=None,
        bins=None,
        weights=None,
        is_hist:bool = False
    ):
        """
        Method to set the reference background histogram.

        Arguments :
            data :
                The data given as a list of array-like.
                The list must follow have a length equal to the number of channels

            datatype :
                String specifying the data type used to make the histograms.
                Currently supported types are : 'np' and 'boost'
                Default to 'np'.

            rang :
                List of the ranges of hitograms for all channels.
                Ignored if the range was set before.

            bins :
                Define the bin edges.
                Ignored if the bins were set before.

            weights :
                Event weights for the reference background.
                Can be either None, a float or an array-like of floats.
                Default to None.

            is_hist :
                Boolean specirying if the data is given in binned format.
                Default to False and ignored if datatype='boost'.
        """

        # Check if we should ignore some arguments
        if self.range is None:
            self.range = rang
        if self.bins is None:
            self.bins = bins

        # Check if we are not missing the binning
        if self.bins is None:
            if datatype == 'np':
                raise ValueError('ERROR : You must provide a valid bins argument')

        # Reset the reference histogram(s)
        self.ref = []

        # Make reference histogram(s)
        self.ref = self._make_hist(data, "ref", weights, datatype, is_hist)

        return

    # Method to set the data histogram
    def set_data(
        self,
        data,
        datatype:str = 'np',
        rang=None,
        bins=None,
        is_hist:bool = False
    ):
        """
        Method to set the data histogram.

        Arguments :
            data :
                The data given as a list of array-like.
                The list must follow have a length equal to the number of channels

            datatype :
                String specifying the data type used to make the histograms.
                Currently supported types are : 'np' and 'boost'
                Default to 'np'.

            rang :
                The range of the hitogram.
                Ignored if the range was set before.

            bins :
                Define the bin edges.
                Ignored if the bins were set before.

            is_hist :
                Boolean specirying if the data is given in binned format.
                Default to False and ignored if datatype='boost'.
        """

        # Check if we should ignore some arguments
        if self.range is None:
            self.range = rang
        if self.bins is None:
            self.bins = bins

        # Check if we are not missing the binning
        if self.bins is None:
            if datatype == 'np':
                raise ValueError('ERROR : You must provide a valid bins argument')

        # Reset the data histogram(s)
        self.hist = []

        # Make data histogram(s)
        self.hist = self._make_hist(data, "hist", None, datatype, is_hist)

        return

    # Method to set the signal histogram
    def set_sig(
        self,
        data,
        datatype:str = 'np',
        rang=None,
        bins=None,
        signal_exp:int = 100,
        is_hist:bool = False
    ):
        """
        Method to set the signal histogram.

        Arguments :
                The data given as a list of array-like.
                The list must follow have a length equal to the number of channels

            datatype :
                String specifying the data type used to make the histograms.
                Currently supported types are : 'np' and 'boost'
                Default to 'np'.

            rang :
                The range of the hitogram.
                Ignored if the range was set before.

            bins :
                Define the bin edges.
                Ignored if the bins were set before.

            signal_exp :
                Integer or list of integers specifying the expected number of signal events in each channel.
                If an integer is given, the expected number of signal events is assumed to be the same in all channels.
                Default to 100.

            is_hist :
                Boolean specirying if the data is given in binned format.
                Default to False and ignored if datatype='boost'.
        """

        # Check if we should ignore some arguments
        if self.range is None:
            self.range = rang
        if self.bins is None:
            self.bins = bins

        # Check the signal_exp argument
        if not isinstance(signal_exp, list):
            self.signal_exp = [signal_exp for ch in range(self.nchan)]
        elif isinstance(signal_exp, list) and len(signal_exp) == self.nchan:
            self.signal_exp = signal_exp
        else:
             raise ValueError("ERROR : Invalid 'signal_exp' argument (must be a lis of length nchan) !")

        # Check if we are not missing the binning
        if self.bins is None:
            if datatype == 'np':
                raise ValueError("ERROR : You must provide a valid bins argument !")

        # Reset the signal histogram(s)
        self.sig = []

        # Make signal histogram for channel i
        self.sig = self._make_hist(data, "sig", None, datatype, is_hist)

        return

    # Method to set all histograms from a python dict
    def set_from_dict(
        self,
        dct,
        datatype='np',
        rang=None,
        bins=None,
        weights=None,
        signal_exp:int = 100,
        is_hist:bool = False
    ):
        """
        Method to set all histograms from a python dict.
        The histograms, as well as the binning can be recognized from the dict keys.

        Arguments :
            dct :
                The dict dontaining the data to be histogramed.
                The reference must have the key 'ref', the data must have the key 'hist' and the signal must have the key 'sig'.
                You can also provied the binning, weights and signal information  using the keys 'bins' and 'weights', 'signal_exp'.

            datatype :
                String specifying the data type used to make the histograms.
                Currently supported types are : 'np' and 'boost'
                Default to 'np'.

            rang :
                The range of the hitogram.
                Ignored if the range was set before or if given in dct.

            bins :
                Define the bin edges.
                Ignored if the bins were set before or if given in dct.

            weights :
                Event weights for the reference background.
                Can be either None, a float or an array-like of floats.
                Ignored if given in dct.
                Default to None.

            signal_exp :
                Integer or list of integers specifying the expected number of signal events in each channel.
                If an integer is given, the expected number of signal events is assumed to be the same in all channels.
                Default to 100.

            is_hist :
                Boolean specirying if the data is given in binned format.
                Default to False and ignored if datatype='boost'.
        """

        # Check if we should ignore some arguments
        if 'rang' in dct.keys():
            rang = dct['rang']
        if 'bins' in dct.keys():
            bins = dct['bins']
        if 'weights' in dct.keys():
            weights = dct['weights']
        if 'signal_exp' in dct.keys():
            signal_exp = dct['signal_exp']

        # Check if bins and range were alreday set
        if self.range is None:
            self.range = rang
        if self.bins is None:
            self.bins = bins

        # Check the signal_exp argument
        if isinstance(signal_exp, int):
            signal_exp = [signal_exp for ch in range(self.nchan)]
        elif isinstance(signal_exp, list) and len(signal_exp) == self.nchan:
            self.signal_exp = signal_exp
        else:
             raise ValueError("ERROR : Invalid 'signal_exp' argument (must be a list of length nchan) !")

        # Check if reference background is given
        if 'ref' in dct.keys():
            self.set_ref(dct['ref'], datatype, rang, bins, weights, is_hist)

        # Check if data is given
        if 'hist' in dct.keys():
            self.set_data(dct['hist'], datatype, rang, bins, is_hist)

        # Check if signal is given
        if 'sig' in dct.keys():
            self.set_sig(dct['sig'], datatype, rang, bins, signal_exp, is_hist)

        return

    #End of DataHandler class


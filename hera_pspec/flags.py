from __future__ import print_function, division
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyuvdata import UVData
import copy


def uvd_to_array(uvdlist, baseline):
    """
    Reads UVData objects and stores flags and nsamples arrays in a list
    in preparation for stacking
    
    Parameters
    ----------
    uvdlist : list
        a list of UVData objects

    baseline : tuple
        specifying the baseline to look at in the form (ant1, ant2, pol),
        for example (65, 66, 'xx')

    Returns
    -------
    nsamples_list : list
        a list of nsamples arrays from the input files

    flags_list : list
        a list of flags arrays from the input files

    """
    if len(uvdlist) == 0:
        raise ValueError("uvdlist must contain at least 1 UVData object")
    elif not isinstance(uvdlist, list):
        raise TypeError("uvdlist takes list inputs (for 1 UVData object, \
                        add it to a list of length 1)")
    # creating lists of flags and nsamples arrays of input UVData objects
    flags_list = [uvd.get_flags(baseline) for uvd in uvdlist]
    nsamples_list = [uvd.get_nsamples(baseline) for uvd in uvdlist]
    return nsamples_list, flags_list

def stacked_array(array_list):
    """
    Generates a long stacked array for (waterfall plots) from a list of arrays
    
    Parameters
    ----------
    array_list : list
        list of numpy.ndarray objects to be stacked
    
    Returns
    -------
    array_total : numpy.ndarray
        array of all arrays in array_list stacked in list index order
    """
    counter = 0
    if len(array_list) == 0:
        raise ValueError("input array list cannot be empty")
    # looping through all the arrays and stacking them up
    for i in range(len(array_list)):
        array_new = np.zeros(array_list[i].shape)
        if counter == 0:
            array_total = array_list[i]
        elif counter != 0:
            array_new = array_list[i]
            array_total = np.vstack((array_total, array_new))
        counter += 1
    return array_total

def construct_factorizable_mask(uvdlist, spw_ranges=[(0, 1024)], first='col', greedy_threshold=0.3, n_threshold = 1, 
                         retain_flags=True, unflag=False, greedy=True, inplace=False):
    """
    Generates a factorizable mask using a greedy flagging algorithm given a list
    of UVData objects. First, flags are added to the mask based on the number of
    samples available for the pixel. Next, in greedy flagging, based on the
    "first" param, full columns (or rows) exceeding the greedy threshold are 
    flagged, & then any remaining flags have their full rows (or columns) 
    flagged. Unflagging the entire array is also an option.
    
    Parameters
    ----------
    uvdlist : list
        list of UVData objects to operate on

    spw_ranges : list
        list of tuples of the form (min_channel, max_channel) defining which
        spectral window (channel range) to flag - min_channel is inclusive,
        but max_channel is exclusive
    
    first : str
        either 'col' or 'row', defines which axis is flagged first based on
        the greedy_threshold - default is 'col'
        
    greedy_threshold : float
        the flag fraction beyond which a given row or column is flagged in the
        first stage of greedy flagging
        
    n_threshold : int
        the number of samples needed for a pixel to remain unflagged
    
    retain_flags : bool
        if True, then pixels flagged in the file will always remain flagged, even
        if they meet the n_threshold (default is True)
        
    unflag : bool
        if True, the entire mask is unflagged. default is False
        
    greedy : bool
        if True, greedy flagging takes place, & if False, only n_threshold flagging
        is used (resulting mask will not be factorizable). default is True
        
    inplace : bool
        if True, then the input UVData objects' flag arrays are modified, and if
        False, new UVData objects identical to the inputs but with updated flags
        are created and returned
    
    Returns
    -------
    uvdlist_updated : list
        if inplace=False, a new list of UVData objects with updated flags 
    """
    # initialize a list to place output UVData objects in if inplace=False
    uvdlist_updated = []
    
    # iterate over datasets
    for dset in uvdlist:
        if not isinstance(dset, UVData): raise TypeError("uvdlist must be a list of UVData objects")
        if not inplace: uvd_updated_i = copy.deepcopy(dset)
        # iterate over spectral windows
        for spw in spw_ranges:
            if not isinstance(spw, tuple): raise TypeError("spw_ranges must be a list of tuples")
            if unflag:
                #unflag everything if unflag = True
                if inplace:
                    dset.flag_array[:, :, spw[0]:spw[1], :] = False
                    continue
                elif not inplace: 
                    uvd_updated_i.flag_array[:, :, spw[0]:spw[1], :] = False
                    uvdlist_updated.append(uvd_updated_i)
                    continue
            # conduct flagging:
            # iterate over polarizations
            for n in range(dset.Npols):
                # iterate over unique baselines
                ubl = np.unique(dset.baseline_array)
                for bl in ubl:
                    # get baseline-times indices
                    bl_inds = np.where(np.in1d(dset.baseline_array, bl))[0]
                    # create a new array of flags with only those indices
                    flags = dset.flag_array[bl_inds, 0, :, n].copy()
                    nsamples = dset.nsample_array[bl_inds, 0, :, n].copy()
                    Ntimes = int(flags.shape[0])
                    Nfreqs = int(flags.shape[1])
                    narrower_flags_window = flags[:, spw[0]:spw[1]]
                    narrower_nsamples_window = nsamples[:, spw[0]:spw[1]]
                    flags_output = np.zeros(narrower_flags_window.shape)
                    if not (isinstance(greedy_threshold, float) or isinstance(n_threshold, int)):
                        raise TypeError("greedy_threshold must be a float, and n_threshold must be an int")
                    if greedy_threshold >= 1 or greedy_threshold <= 0:
                        raise ValueError("greedy_threshold must be between 0 & 1, exclusive")
                    # if retaining flags, an extra condition is added to the threshold filter
                    if retain_flags:
                        flags_output[(narrower_nsamples_window >= n_threshold) & (narrower_flags_window == False)] = False
                        flags_output[(narrower_nsamples_window < n_threshold) | (narrower_flags_window == True)] = True
                    else:
                        flags_output[(narrower_nsamples_window >= n_threshold)] = False
                        flags_output[(narrower_nsamples_window < n_threshold)] = True
                    # conducting the greedy flagging
                    if greedy:
                        if first != 'col' and first != 'row':
                            raise ValueError("first must be either 'row' or 'col'")
                        if first == 'col':
                            # flagging all columns that exceed the greedy_threshold
                            col_indices = np.where(np.sum(flags_output, axis = 0)/Ntimes > greedy_threshold)
                            flags_output[:, col_indices] = True
                            # flagging all remaining rows
                            remaining_rows = np.where(np.sum(flags_output, axis = 1) > len(list(col_indices[0])))
                            flags_output[remaining_rows, :] = True
                        elif first == 'row':
                            # flagging all rows that exceed the greedy_threshold
                            row_indices = np.where(np.sum(flags_output, axis = 1)/(spw[1]-spw[0]) > greedy_threshold)
                            flags_output[row_indices, :] = True
                            # flagging all remaining columns
                            remaining_cols = np.where(np.sum(flags_output, axis = 0) > len(list(row_indices[0])))
                            flags_output[:, remaining_cols] = True
                    # updating the UVData object's flag_array if inplace, or creating a new object if not
                    if inplace:
                        dset.flag_array[bl_inds, 0, spw[0]:spw[1], n] = flags_output
                    elif not inplace:
                        uvd_updated_i.flag_array[bl_inds, 0, spw[0]:spw[1], n] = flags_output
        if not inplace: uvdlist_updated.append(uvd_updated_i)
    # returning an updated list of UVData objects if not inplace
    if not inplace:
        return uvdlist_updated

def long_waterfall(array_list, title, cmap='gray', starting_lst=[]):
    """    
    Generates a waterfall plot of flags or nsamples with axis sums from an
    input array

    Parameters
    ----------
    array_list : list
        list of arrays to be stacked and displayed
    
    title : str
        title of the plot
    
    cmap : str, optional
        cmap parameter for the waterfall plot (default is 'gray')
        
    starting_lst : list, optional
        list of starting lst to display in the plot
        
    Returns
    -------
    main_waterfall : matplotlib.axes
        Matplotlib Axes instance of the main plot
        
    freq_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across times
        
    time_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across freqs
        
    data : numpy.ndarray
        A copy of the stacked_array output that is being displayed
    """
    # creating the array to be displayed using stacked_array()
    data = stacked_array(array_list)
    # setting up the figure and grid
    fig = plt.figure()
    fig.suptitle(title, fontsize=30, horizontalalignment='center')
    grid = gridspec.GridSpec(ncols=10, nrows=15)
    main_waterfall = fig.add_subplot(grid[0:14, 0:8])
    freq_histogram = fig.add_subplot(grid[14:15, 0:8], sharex=main_waterfall)
    time_histogram = fig.add_subplot(grid[0:14, 8:10], sharey=main_waterfall)
    fig.set_size_inches(20, 80)
    grid.tight_layout(fig)
    counter = data.shape[0] // 60
    # waterfall plot
    main_waterfall.imshow(data, aspect='auto', cmap=cmap, 
                          interpolation='none')
    main_waterfall.set_ylabel('Integration Number')
    main_waterfall.set_yticks(np.arange(0, counter*60 + 1, 30))
    main_waterfall.set_ylim(60*(counter+1), 0)
    #red lines separating files
    for i in range(counter+1):
        main_waterfall.plot(np.arange(data.shape[1]),
                            60*i*np.ones(data.shape[1]), '-r')
    for i in range(len(starting_lst)):
        if not isinstance(starting_lst[i], str):
            raise TypeError("starting_lst must be a list of strings")
    # adding text of filenames
    if len(starting_lst) > 0:
        for i in range(counter):
            short_name = 'first\nintegration LST:\n'+starting_lst[i]
            plt.text(-20, 26 + i*60, short_name, rotation=-90, size='small',
                     horizontalalignment='center')
    main_waterfall.set_xlim(0, 1024)
    # frequency sum plot
    counts_freq = np.sum(data, axis=0)
    max_counts_freq = max(np.amax(counts_freq), data.shape[0])
    normalized_freq = 100 * counts_freq/max_counts_freq
    freq_histogram.set_xticks(np.arange(0, 1024, 50))
    freq_histogram.set_yticks(np.arange(0, 101, 5))
    freq_histogram.set_xlabel('Channel Number (Frequency)')
    freq_histogram.set_ylabel('Occupancy %')
    freq_histogram.grid()
    freq_histogram.plot(np.arange(0, 1024), normalized_freq, 'r-')
    # time sum plot
    counts_times = np.sum(data, axis=1)
    max_counts_times = max(np.amax(counts_times), data.shape[1])
    normalized_times = 100 * counts_times/max_counts_times
    time_histogram.plot(normalized_times, np.arange(data.shape[0]), 'k-',
                        label='all channels')
    time_histogram.set_xticks(np.arange(0, 101, 10))
    time_histogram.set_xlabel('Flag %')
    time_histogram.autoscale(False)
    time_histogram.grid()
    # returning the axes
    return main_waterfall, freq_histogram, time_histogram, data

def flag_channels(uvdlist, spw_ranges, inplace=False):
    """
    Flags a given range of channels entirely for a list of UVData objects
    
    Parameters
    ----------
    uvdlist : list
        list of UVData objects to be flagged
        
    spw_ranges : list
        list of tuples of the form (min_channel, max_channel) defining which
        channels to flag
    
    inplace : bool, optional
        if True, then the input UVData objects' flag arrays are modified, 
        and if False, new UVData objects identical to the inputs but with
        updated flags are created and returned (default is False)
    
    Returns:
    -------
    uvdlist_updated : list
        list of updated UVData objects
    """
    uvdlist_updated = []
    for uvd in uvdlist:
        if not isinstance(uvd, UVData):
            raise TypeError("uvdlist must be a list of UVData objects")
        if not inplace:
            uvd_updated_i = copy.deepcopy(uvd)            
        for spw in spw_ranges:
            if not isinstance(spw, tuple):
                raise TypeError("spw_ranges must be a list of tuples")
            for pol in range(uvd.Npols):
                ubl = np.unique(uvd.baseline_array)
                for bl in ubl:
                    bl_inds = np.where(np.in1d(uvd.baseline_array, bl))[0]
                    fully_flagged = np.ones(uvd.flag_array[bl_inds, 0, spw[0]:spw[1], pol].shape, dtype=bool)
                    if inplace:
                        uvd.flag_array[bl_inds, 0, spw[0]:spw[1], pol] = fully_flagged
                    elif not inplace:
                        uvd_updated_i.flag_array[bl_inds, 0, spw[0]:spw[1], pol] = fully_flagged
        uvdlist_updated.append(uvd_updated_i)
    if not inplace:
        return uvdlist_updated
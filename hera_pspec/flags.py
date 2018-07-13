import numpy as np

mask_generator(nsamples, flags, n_threshold, greedy=False, axis, greedy_threshold, retain_flags=True):
    """
    Generates a greedy flags mask from input flags and nsamples arrays
    
    Parameters
    ----------
    nsamples : numpy.ndarray
        integer array with number of samples available for each frequency channel at a given LST angle
    
    flags : numpy.ndarray
        binary array with 1 representing flagged, 0 representing unflagged
        
    n_threshold : int
        minimum number of samples needed for a point to remain unflagged
        
    greedy : bool
        greedy flagging is used if true (default is False)
        
    axis : int
        which axis to flag first if greedy=True (1 is row-first, 0 is col-first)
        
    greedy_threshold : float
        if greedy=True, the threshold used to flag rows or columns if axis=1 or 0, respectively
        
    retain_flags : bool
        LST-Bin Flags are left flagged even if thresholds are not met (default is True)
    
    Returns
    -------
    mask : numpy.ndarray
	binary array of the new mask where 1 is flagged, 0 is unflagged

"""

    shape = nsamples.shape
    flags_output = np.zeros(shape)

    num_exactly_equal = 0

    # comparing the number of samples to the threshold 
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if nsamples[i, j] < n_threshold:
                flags_output[i, j] = 1
            elif nsamples[i, j] > n_threshold:
                if retain_flags and flags[i, j] == 1:
                    flags_output[i, j] = 1
                else:
                    flags_output[i, j] = 0
            elif nsamples[i, j] == n_threshold:
                if retain_flags and flags[i, j] == 1:
                    flags_output[i, j] = 1
                else:
                    flags_output[i, j] = 0
                    num_exactly_equal += 1

    # the greedy part

    if axis == 0:
        if greedy:
            column_flags_counter = 0
            for j in range(shape[1]):
                if np.sum(flags_output[:, j])/shape[0] > greedy_threshold:
                    flags_output[:, j] = np.ones([shape[0]])
                    column_flags_counter += 1
            for i in range(shape[0]):
                if np.sum(flags_output[i, :]) > column_flags_counter:
                    flags_output[i, :] = np.ones([shape[1]])
    elif axis == 1:
        if greedy:
            row_flags_counter = 0
            for i in range(shape[0]):
                if np.sum(flags_output[i, :])/shape[1] > greedy_threshold:
                    flags_output[i, :] = np.ones([shape[1]])
                    row_flags_counter += 1
            for j in range(shape[1]):
                if np.sum(flags_output[:, j]) > row_flags_counter:
                    flags_output[:, j] = np.ones([shape[0]])

    return flags_output

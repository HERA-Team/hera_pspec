import numpy as np
import pyuvdata
from hera_pspec import conversions
import matplotlib.pyplot as plt
import copy


def delay_spectrum(uvp, blpairs, spw, pol, average_blpairs=False, 
                   average_times=False, fold=False, plot_noise=False, 
                   delay=True, deltasq=False, little_h=True, 
                   legend=False, ax=None):
    """
    Plot a 1D delay spectrum for a group of baselines.
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    blpairs : list of tuples or lists of tuples
        List of baseline-pair tuples, or groups of baseline-pair tuples.
    
    spw, pol : int or str
        Which spectral window and polarization to plot.
    
    average_blpairs : bool, optional
        If True, average over the baseline pairs within each group.
        
    average_times : bool, optional
        If True, average spectra over the time axis. Default: False.
    
    fold : bool, optional
        Whether to fold the power spectrum in |k_parallel|. Default: False.
    
    plot_noise : bool, optional
        Whether to plot noise power spectrum curves or not. Default: False.
    
    delay : bool, optional
        Whether to plot the power spectrum in delay units (ns) or cosmological 
        units (h/Mpc). Default: True.
    
    deltasq : bool, optional
        If True, plot dimensionless power spectra, Delta^2. This is ignored if 
        delay=True. Default: False.
    
    little_h : bool, optional
        If using cosmological units (i.e. delay=False), whether to use h^-1 
        units or not. Default: True.
    
    legend : bool, optional
        Whether to switch on the plot legend. Default: False.
    
    ax : matplotlib.axes, optional
        Use this to pass in an existing Axes object, which the power spectra 
        will be added to. (Warning: Labels and legends will not be altered in 
        this case, even if the existing plot has completely different axis 
        labels etc.) If None, a new Axes object will be created. Default: None.
    
    Returns
    -------
    ax : matplotlib.axes
        Matplotlib Axes instance.
    """
    # Create new Axes if none specified
    new_plot = False
    if ax is None:
        new_plot = True
        ax = plt.subplot()
    
    # Add ungrouped baseline-pairs into a group of their own (expected by the
    # averaging routines) # FIXME: Check if this is an in-place operation
    blpairs_in = blpairs
    blpairs = [] # Must be a list, not an array
    for i, blpgrp in enumerate(blpairs_in):
        if not isinstance(blpgrp, list):
            blpairs.append([blpairs_in[i],])
        else:
            blpairs.append(blpairs_in[i])
    
    # Average over blpairs or times if requested
    blpairs_in = copy.deepcopy(blpairs) # Save input blpair list
    if average_blpairs:
        uvp_plt = uvp.average_spectra(blpair_groups=blpairs, 
                                      time_avg=average_times, inplace=False)
    else:
        uvp_plt = copy.deepcopy(uvp)
        if average_times:
            # Average over times, but not baseline-pairs
            # Unpacks blpairs from their groups into 1D list of 1-element lists
            blpair_list = []
            for blpgrp in blpairs: blpair_list += [[blp,] for blp in blpgrp]
            uvp_plt.average_spectra(blpair_groups=blpair_list, 
                                    time_avg=True, inplace=True)
    
    # Fold the power spectra if requested
    if fold:
        uvp_plt.fold_spectra()
    
    # Convert to Delta^2 units if requested
    if deltasq and not delay:
        uvp_plt.convert_to_deltasq()
    
    # Get x-axis units (delays in ns, or k_parallel in Mpc^-1 or h Mpc^-1)
    if delay:
        dlys = uvp_plt.get_dlys(spw) * 1e9 # ns
        x = dlys
    else:
        k_perp, k_para = uvp_plt.get_kvecs(spw, little_h=little_h)
        x = k_para
    
    # Plot power spectra
    for blgrp in blpairs:
        # Loop over blpairs in group and plot power spectrum for each one
        print "Group:", blgrp
        for blp in blgrp:
            key = (spw, blp, pol)
            print "\tKey:", key
            power = np.abs(np.real(uvp_plt.get_data(key))).T # FIXME: Transpose?
            
            print "\t\t", x.shape, power.shape
            
            ax.plot(x, power, label="%s" % str(key)) # FIXME
            
            # If blpairs were averaged, only the first blpair in the group 
            # exists any more (so skip the rest)
            if average_blpairs: break
    
    # Set log scale
    ax.set_yscale('log')
    
    # Plot noise spectra, if requested
    # FIXME: This takes Tsys as an argument. So should probably be a separate 
    # plotting function...
    """
    if show_noise:
        P_N = uvp_avg.generate_noise_spectra(spw, pol, 400) # FIXME
        P_N = P_N[uvp_avg.antnums_to_blpair(blp)]
    """
    # Add legend
    if legend:
        ax.add_legend(loc='upper left')
    
    # Add labels with units
    if ax.get_xlabel() == "":
        ax.set_xlabel("x")
    
    return ax
    

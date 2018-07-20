import numpy as np
import pyuvdata
from hera_pspec import conversions
import matplotlib
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict as odict


def delay_spectrum(uvp, blpairs, spw, pol, average_blpairs=False, 
                   average_times=False, fold=False, plot_noise=False, 
                   delay=True, deltasq=False, legend=False, ax=None,
                   component='real'):
    """
    Plot a 1D delay spectrum (or spectra) for a group of baselines.
    
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
        Whether to fold the power spectrum in :math:`|k_\parallel|`. 
        Default: False.
    
    plot_noise : bool, optional
        Whether to plot noise power spectrum curves or not. Default: False.
    
    delay : bool, optional
        Whether to plot the power spectrum in delay units (ns) or cosmological 
        units (h/Mpc). Default: True.
    
    deltasq : bool, optional
        If True, plot dimensionless power spectra, Delta^2. This is ignored if 
        delay=True. Default: False.
    
    legend : bool, optional
        Whether to switch on the plot legend. Default: False.
    
    ax : matplotlib.axes, optional
        Use this to pass in an existing Axes object, which the power spectra 
        will be added to. (Warning: Labels and legends will not be altered in 
        this case, even if the existing plot has completely different axis 
        labels etc.) If None, a new Axes object will be created. Default: None.

    component : str
        Component of complex spectra to plot, options=['abs', 'real', 'imag'].
        Default: 'real'. 

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance.
    """
    # Create new Axes if none specified
    new_plot = False
    if ax is None:
        new_plot = True
        fig, ax = plt.subplots(1, 1)
    
    # Add ungrouped baseline-pairs into a group of their own (expected by the
    # averaging routines)
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
            uvp_plt.average_spectra(time_avg=True, inplace=True)
            
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
        k_para = uvp_plt.get_kparas(spw)
        x = k_para
    
    # Plot power spectra
    for blgrp in blpairs:
        # Loop over blpairs in group and plot power spectrum for each one
        for blp in blgrp:
            key = (spw, blp, pol)
            if component == 'real':
                power = np.abs(np.real(uvp_plt.get_data(key))).T
            elif component == 'imag':
                power = np.abs(np.imag(uvp_plt.get_data(key))).T
            elif component == 'abs':
                power = np.abs(uvp_plt.get_data(key)).T

            # flag spectra that have zero integration
            flags = np.isclose(uvp_plt.get_integrations(key), 0.0)
            power[:, flags] = np.nan

            ax.plot(x, power, label="%s" % str(key))
            
            # If blpairs were averaged, only the first blpair in the group 
            # exists any more (so skip the rest)
            if average_blpairs: break
    
    # Set log scale
    ax.set_yscale('log')
    
    # Add legend
    if legend:
        ax.legend(loc='upper left')
    
    # Add labels with units
    if ax.get_xlabel() == "":
        if delay:
            ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=16)
        else:
            ax.set_xlabel("$k_{\parallel}\ h\ Mpc^{-1}$", fontsize=16)
    if ax.get_ylabel() == "":
        # Sanitize power spectrum units 
        psunits = uvp_plt.units
        if "h^-1" in psunits: psunits = psunits.replace("h^-1", "h^{-1}")
        if "h^-3" in psunits: psunits = psunits.replace("h^-3", "h^{-3}")
        if "Mpc" in psunits and "\\rm" not in psunits: 
            psunits = psunits.replace("Mpc", r"{\rm Mpc}")
        if "pi" in psunits and "\\pi" not in psunits: 
            psunits = psunits.replace("pi", r"\pi")
        
        # Power spectrum type
        if deltasq:
            ax.set_ylabel("$\Delta^2$ $[%s]$" % psunits, fontsize=16)
        else:
            ax.set_ylabel("$P(k_\parallel)$ $[%s]$" % psunits, fontsize=16)
    
    # Return Figure: the axis is an attribute of figure
    if new_plot:
        return fig


def delay_waterfall(uvp, blpairs, spw, pol, component='real', average_blpairs=False, 
                    fold=False, delay=True, deltasq=False, log=True, lst_in_hrs=True,
                    vmin=None, vmax=None, cmap='YlGnBu', axes=None, figsize=(14, 6),
                    force_plot=False):
    """
    Plot a 1D delay spectrum waterfall (or spectra) for a group of baselines.
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    blpairs : list of tuples or lists of tuples
        List of baseline-pair tuples, or groups of baseline-pair tuples.
    
    spw, pol : int or str
        Which spectral window and polarization to plot.
    
    component : str
        Component of complex spectra to plot, options=['abs', 'real', 'imag'].
        Default: 'real'.

    average_blpairs : bool, optional
        If True, average over the baseline pairs within each group.
    
    fold : bool, optional
        Whether to fold the power spectrum in :math:`|k_\parallel|`. 
        Default: False.
    
    delay : bool, optional
        Whether to plot the power spectrum in delay units (ns) or cosmological 
        units (h/Mpc). Default: True.
    
    deltasq : bool, optional
        If True, plot dimensionless power spectra, Delta^2. This is ignored if 
        delay=True. Default: False.
    
    log : bool, optional
        Whether to plot the log10 of the data. Default: True.
    
    lst_in_hrs : bool, optional
        If True, LST is plotted in hours, otherwise its plotted in radians.

    vmin, vmax : float, optional
        Clip the color scale of the delay spectrum to these min./max. values. 
        If None, use the natural range of the data. Default: None.
    
    cmap : str, optional
        Matplotlib colormap to use. Default: 'YlGnBu'.
    
    axes : array of matplotlib.axes, optional
        Use this to pass in an existing Axes object or array of axes, which
        the power spectra will be added to. (Warning: Labels and legends will
        not be altered in this case, even if the existing plot has completely different axis 
        labels etc.) If None, a new Axes object will be created. Default: None.
    
    figsize : tuple
        len-2 integer tuple specifying figure size if axes is None

    force_plot : bool
        Certain qualities of an input UVPSpec will raise an exception,
        and this parameter overrides that to continue plotting. One example is
        having more than 20 blpairs in the object.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance if input ax is None.
    """
    # assert component
    assert component in ['real', 'abs', 'imag'], "Can't parse specified component {}".format(component)

    # Add ungrouped baseline-pairs into a group of their own (expected by the
    # averaging routines)
    blpairs_in = blpairs
    blpairs = [] # Must be a list, not an array
    for i, blpgrp in enumerate(blpairs_in):
        if not isinstance(blpgrp, list):
            blpairs.append([blpairs_in[i],])
        else:
            blpairs.append(blpairs_in[i])
    
    # iterate through and make sure they are blpair integers
    _blpairs = []
    for blpgrp in blpairs:
        _blpgrp = []
        for blp in blpgrp:
            if isinstance(blp, tuple):
                blp_int = uvp.antnums_to_blpair(blp)
            else:
                blp_int = blp
            _blpgrp.append(blp_int)
        _blpairs.append(_blpgrp)
    blpairs = _blpairs

    # Average over blpairs or times if requested
    blpairs_in = copy.deepcopy(blpairs) # Save input blpair list
    if average_blpairs:
        uvp_plt = uvp.average_spectra(blpair_groups=blpairs, 
                                      time_avg=False, inplace=False)
    else:
        uvp_plt = copy.deepcopy(uvp)
            
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
        k_para = uvp_plt.get_kparas(spw)
        x = k_para
    
    # Extract power spectra into array
    waterfall = odict()
    for blgrp in blpairs:
        # Loop over blpairs in group and plot power spectrum for each one
        for blp in blgrp:
            # make key
            key = (spw, blp, pol)
            # get power data
            power = uvp_plt.get_data(key, omit_flags=False)
            # set flagged power data to nan
            flags = np.isclose(uvp_plt.get_integrations(key), 0.0)
            power[flags, :] = np.nan
            # get component
            if component == 'abs':
                waterfall[key] = np.abs(power)
            elif component == 'real':
                waterfall[key] = np.real(power)
            elif component == 'imag':
                waterfall[key] = np.imag(power)

            # If blpairs were averaged, only the first blpair in the group 
            # exists any more (so skip the rest)
            if average_blpairs: break
    
    # check for reasonable number of blpairs to plot...
    Nkeys = len(waterfall)
    if Nkeys > 20 and force_plot == False:
        raise ValueError("Nblps > 20 and force_plot == False, quitting plotting routine...")

    # Take logarithm of data if requested
    if log:
        for k in waterfall:
            waterfall[k] = np.log10(np.abs(waterfall[k]))
        logunits = "\log_{10}"
    else:
        logunits = ""
    
    # Create new Axes if none specified
    new_plot = False
    if axes is None:
        new_plot = True
        # figure out how many subplots to make
        Nkeys = len(waterfall)
        Nside = int(np.ceil(np.sqrt(Nkeys)))
        fig, axes = plt.subplots(Nside, Nside, figsize=figsize)
    # Ensure axes is an ndarray
    if isinstance(axes, matplotlib.axes._subplots.Axes):
        axes = np.array([[axes]])
    if isinstance(axes, list):
        axes = np.array(axes)
    # ensure its 2D and get side lengths
    if axes.ndim == 1:
        axes = axes[:, None]
    assert axes.ndim == 2, "input axes must have ndim == 2"
    Nvert, Nhorz = axes.shape

    # get LST range: setting y-ticks is tricky due to LST wrapping...
    y = uvp_plt.lst_avg_array[uvp_plt.key_to_indices(waterfall.keys()[0])[1]]
    if lst_in_hrs:
        lst_units = "Hr"
        y = np.around(y * 24 / (2*np.pi), 2)
    else:
        lst_units = "rad"
        y = np.around(y, 3)
    Ny = len(y)
    if Ny <= 10:
        Ny_thin = 1
    else:
        Ny_thin = int(round(Ny / 10.0))
    Nx = len(x)

    # Sanitize power spectrum units 
    psunits = uvp_plt.units
    if "h^-1" in psunits: psunits = psunits.replace("h^-1", "h^{-1}")
    if "h^-3" in psunits: psunits = psunits.replace("h^-3", "h^{-3}")
    if "Hz" in psunits: psunits = psunits.replace("Hz", r"{\rm Hz}")
    if "str" in psunits: psunits = psunits.replace("str", r"\,{\rm str}")
    if "Mpc" in psunits and "\\rm" not in psunits: 
        psunits = psunits.replace("Mpc", r"{\rm Mpc}")
    if "pi" in psunits and "\\pi" not in psunits: 
        psunits = psunits.replace("pi", r"\pi")
    if "beam normalization not specified" in psunits:
        psunits = psunits.replace("beam normalization not specified", 
                                 r"{\rm unnormed}")

    # iterate over waterfall keys
    keys = waterfall.keys()
    k = 0
    for i in range(Nvert):
        for j in range(Nhorz):
            # set ax
            ax = axes[i, j]

            # turn off subplot if no more plots to make
            if k >= Nkeys:
                ax.axis('off')
                continue

            # get blpair key for this subplot
            key = keys[k]

            # plot waterfall
            cax = ax.matshow(waterfall[key], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, 
                                 extent=[np.min(x), np.max(x), Ny, 0])

            # ax config
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(labelsize=12)
            if ax.get_title() == '':
                ax.set_title("bls: {} x {}".format(*uvp_plt.blpair_to_antnums(key[1])), y=1)

            # set colorbar
            cbar = ax.get_figure().colorbar(cax, ax=ax)
            cbar.ax.tick_params(labelsize=14)

            # configure left-column plots
            if j == 0:
                # set yticks
                ax.set_yticks(np.arange(Ny)[::Ny_thin])
                ax.set_yticklabels(y[::Ny_thin])
                ax.set_ylabel(r"LST [{}]".format(lst_units), fontsize=16)
            else:
                ax.set_yticklabels([])

            # configure bottom-row plots
            if k + Nhorz >= Nkeys:
                if ax.get_xlabel() == "":
                    if delay:
                        ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=16)
                    else:
                        ax.set_xlabel("$k_{\parallel}\ h\ Mpc^{-1}$", fontsize=16)
            else:
                ax.set_xticklabels([])

            k += 1        

    # make suptitle
    if axes[0][0].get_figure()._suptitle is None:
        if deltasq:
            units = "$%s\Delta^2$ $[%s]$" % (logunits, psunits)
        else:
            units = "$%sP(k_\parallel)$ $[%s]$" % (logunits, psunits)

        spwrange = np.around(np.array(uvp_plt.get_spw_ranges()[spw][:2]) / 1e6, 2)
        axes[0][0].get_figure().suptitle("{}\n{} polarization | {} -- {} MHz".format(units, pol, *spwrange), y=1.03, fontsize=14)

    # Return Axes
    if new_plot:
        return fig
    

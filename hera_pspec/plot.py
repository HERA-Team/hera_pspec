import numpy as np
import pyuvdata
from hera_pspec import conversions, uvpspec, utils, grouping
import matplotlib
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict as odict
import astropy.units as u
import astropy.constants as c
from pyuvdata import UVData
from scipy import stats
import uvtools
import astropy.stats as astats

def delay_spectrum(uvp, blpairs, spw, pol, average_blpairs=False, 
                   average_times=False, fold=False, plot_noise=False, 
                   delay=True, deltasq=False, legend=False, ax=None,
                   component='real', lines=True, markers=False, error=None,
                   times=None, logscale=True, force_plot=False,
                   label_type='blpairt', plot_stats=None, **kwargs):
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

    component : str, optional
        Component of complex spectra to plot, options=['abs', 'real', 'imag'].
        Default: 'real'. 

    lines : bool, optional
        If True, plot lines between bandpowers for a given pspectrum.
        Default: True.

    markers : bool, optional
        If True, plot circles at bandpowers. Filled for positive, empty
        for negative. Default: False.

    error : str, optional
        If not None and if error exists in uvp stats_array, plot errors
        on bandpowers. Default: None.

    times : array_like, optional
        Float ndarray containing elements from time_avg_array to plot.

    logscale : bool, optional
        If True, put plot on a log-scale. Else linear scale. Default: True.

    force_plot : bool, optional
        If plotting a large number of spectra (>100), this function will error.
        Set this to True to override this large plot error and force plot.
        Default: False.

    label_type : int, optional
        Line label type in legend, options=['key', 'blpair', 'blpairt'].
            key : Label lines based on (spw, blpair, pol) key.
            blpair : Label lines based on blpair.
            blpairt : Label lines based on blpair-time.

    plot_stats : string, optional
        If not None, plot an entry in uvp.stats_array instead
        of power spectrum in uvp.data_array.

    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.

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

    # Select times if requested
    if times is not None:
        uvp = uvp.select(times=times, inplace=False)

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
      
    # Check plot size
    if uvp_plt.Ntimes * len(blpairs) > 100 and force_plot == False:
        raise ValueError("Trying to plot > 100 spectra... Set force_plot=True to continue.")

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
    
    # Check plot_stats
    if plot_stats is not None:
        assert plot_stats in uvp_plt.stats_array, "specified key {} not found in stats_array".format(plot_stats)

    # Plot power spectra
    for blgrp in blpairs:
        # Loop over blpairs in group and plot power spectrum for each one
        for blp in blgrp:
            # setup key and casting function
            key = (spw, blp, pol)
            if component == 'real':
                cast = np.real
            elif component == 'imag':
                cast = np.imag
            elif component == 'abs':
                cast = np.abs

            # get data array and repeat x array
            if plot_stats is None:
                data = cast(uvp_plt.get_data(key))
            else:
                data = cast(uvp_plt.get_stats(plot_stats, key))

            # flag records that have zero integration
            flags = np.isclose(uvp_plt.get_integrations(key), 0.0)
            data[flags] = np.nan

            # get errs if requessted
            if error is not None and hasattr(uvp_plt, 'stats_array'):
                if error in uvp_plt.stats_array:
                    errs = uvp_plt.get_stats(error, key)
                    errs[flags] = np.nan

            # get times
            blptimes = uvp_plt.time_avg_array[uvp_plt.blpair_to_indices(blp)]

            # iterate over integrations per blp
            for i in range(data.shape[0]):
                # get y data
                y = data[i]
                t = blptimes[i]

                # form label
                if label_type == 'key':
                    label = "{}".format(key)
                elif label_type == 'blpair':
                    label = "{}".format(blp)
                elif label_type == 'blpairt':
                    label = "{}, {:0.5f}".format(blp, t)
                else:
                    raise ValueError("Couldn't undestand label_type {}".format(label_type))

                # plot elements
                cax = None
                if lines:
                    if logscale:
                        _y = np.abs(y)
                    else:
                        _y = y

                    cax, = ax.plot(x, _y, marker='None', label=label, **kwargs)

                if markers:
                    if cax is None:
                        c = None
                    else:
                        c = cax.get_color()
                    if lines:
                        label = None

                    # plot markers
                    if logscale:
                        # plot positive w/ filled circles
                        cax, = ax.plot(x[y >= 0], np.abs(y[y >= 0]), c=c, ls='None', marker='o', 
                                      markerfacecolor=c, markeredgecolor=c, label=label, **kwargs)
                        # plot negative w/ unfilled circles
                        c = cax.get_color()
                        cax, = ax.plot(x[y < 0], np.abs(y[y < 0]), c=c, ls='None', marker='o',
                                       markerfacecolor='None', markeredgecolor=c, **kwargs)
                    else:
                        cax, = ax.plot(x, y, c=c, ls='None', marker='o', 
                                      markerfacecolor=c, markeredgecolor=c, label=label, **kwargs)

                if error is not None and hasattr(uvp_plt, 'stats_array'):
                    if error in uvp_plt.stats_array:
                        if cax is None:
                            c = None
                        else:
                            c = cax.get_color()
                        if logscale:
                            _y = np.abs(y)
                        else:
                            _y = y
                        cax = ax.errorbar(x, _y, fmt='none', ecolor=c, yerr=cast(errs[i]), **kwargs)
                    else:
                        raise KeyError("Error variable '%s' not found in stats_array of UVPSpec object." % error)

            # If blpairs were averaged, only the first blpair in the group 
            # exists any more (so skip the rest)
            if average_blpairs: break
    
    # Set log scale
    if logscale:
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
    if ax.get_ylabel() == "" and plot_stats is None:
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
                    force_plot=False, times=None, title_type='blpair'):
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
        If plotting a large number of blpairs (>20), this routine will quit
        unless force_plot == True.

    times : array_like, optional
        Float ndarray containing elements from time_avg_array to plot.

    title_type : str, optional
        Type of title to put above plot(s). Options = ['blpair', 'blvec']
        blpair : "bls: {bl1} x {bl2}"
        blvec : "bl len {len} m & ang {ang} deg" 

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

    # Select times if requested
    if times is not None:
        uvp = uvp.select(times=times, inplace=False)

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
        raise ValueError("Nblps > 20 and force_plot == False, quitting...")

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
    
    # Ensure its 2D and get side lengths
    if axes.ndim == 1:
        axes = axes[:, None]
    assert axes.ndim == 2, "input axes must have ndim == 2"
    Nvert, Nhorz = axes.shape

    # Get LST range: setting y-ticks is tricky due to LST wrapping...
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

    # get baseline vectors
    blvecs = dict(zip([uvp_plt.bl_to_antnums(bl) for bl in uvp_plt.bl_array], uvp_plt.get_ENU_bl_vecs()))

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

    # Iterate over waterfall keys
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
            blp = uvp_plt.blpair_to_antnums(key[1])

            # plot waterfall
            cax = ax.matshow(waterfall[key], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, 
                                 extent=[np.min(x), np.max(x), Ny, 0])

            # ax config
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(labelsize=12)
            if ax.get_title() == '':
                if title_type == 'blpair':
                    ax.set_title("bls: {} x {}".format(*blp), y=1)
                elif title_type == 'blvec':
                    blv = 0.5 * (blvecs[blp[0]] + blvecs[blp[1]])
                    lens, angs = utils.get_bl_lens_angs([blv], bl_error_tol=1.0)
                    ax.set_title("bl len {len:0.2f} m & {ang:0.0f} deg".format(len=lens[0], ang=angs[0]), y=1)

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
        axes[0][0].get_figure().suptitle("{}\n{} polarization | {} -- {} MHz".format(units, pol, *spwrange), 
                                         y=1.03, fontsize=14)

    # Return Axes
    if new_plot:
        return fig


def delay_wedge(uvp, spw, pol, blpairs=None, times=None, fold=False, delay=True,
                rotate=False, component='real', log10=True, loglog=False,
                red_tol=1.0, center_line=False, horizon_lines=False,
                title=None, ax=None, cmap='viridis', figsize=(8, 6),
                deltasq=False, colorbar=False, cbax=None, vmin=None, vmax=None,
                edgecolor='none', flip_xax=False, flip_yax=False, lw=2, set_bl_tick_major=False,
                set_bl_tick_minor=False, xtick_size=10, xtick_rot=0, ytick_size=10, ytick_rot=0,
                **kwargs):
    """
    Plot a 2D delay spectrum (or spectra) from a UVPSpec object. Note that
    all integrations and redundant baselines are averaged (unless specifying times)
    before plotting.
    
    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object containing delay spectra to plot.

    spw : integer
        Which spectral window to plot.

    pol : int or str
        Polarization integer or string

    blpairs : list of tuples, optional
        List of baseline-pair tuples to use in plotting.

    times : list, optional
        An ndarray or list of times from uvp.time_avg_array to
        select on before plotting. Default: None.

    fold : bool, optional
        Whether to fold the power spectrum in k_parallel. 
        Default: False.

    delay : bool, optional
        Whether to plot the axes in tau (ns). If False, axes will
        be plotted in cosmological units.
        Default: True.

    rotate : bool, optional
        If False, use baseline-type as x-axis and delay as y-axis,
        else use baseline-type as y-axis and delay as x-axis.
        Default: False

    component : str, optional
        Component of complex spectra to plot. Options=['real', 'imag', 'abs']
        Default: 'real'.

    log10 : bool, optional
        If True, take log10 of data before plotting. Default: True

    loglog : bool, optional
        If True, turn x-axis and y-axis into log-log scale. Default: False

    red_tol : float, optional
        Redundancy tolerance when solving for redundant groups in meters.
        Default: 1.0

    center_line : bool, optional
        Whether to plot a dotted line at k_perp = 0.
        Default: False.

    horizon_lines : bool, optional
        Whether to plot dotted lines along the horizon.
        Default: False.

    title : string, optional
        Title for subplot.  Default: None.

    ax : matplotlib.axes, optional
        If not None, use this axes as a subplot for delay wedge.

    cmap : str, optional
        Colormap of wedge plot. Default: 'viridis'

    figsize : len-2 integer tuple, optional
        If ax is None, this is the new figure size.

    deltasq : bool, optional
        Convert to Delta^2 before plotting. This is ignored if delay=True.
        Default: False

    colorbar : bool, optional
        Add a colorbar to the plot. Default: False

    cbax : matplotlib.axes, optional
        Axis object for adding colorbar if True. Default: None

    vmin : float, optional
        Minimum range of colorscale. Default: None

    vmax : float, optional
        Maximum range of colorscale. Default: None

    edgecolor : str, optional
        Edgecolor of bins in pcolormesh. Default: 'none'

    flip_xax : bool, optional
        Flip xaxis if True. Default: False

    flip_yax : bool, optional
        Flip yaxis if True. Default: False

    lw : int, optional
        Line-width of horizon and center lines if plotted. Default: 2.

    set_bl_tick_major : bool, optional
        If True, use the baseline lengths as major ticks, rather than default uniform
        grid.

    set_bl_tick_minor : bool, optional
        If True, use the baseline lengths as minor ticks, which have no labels.

    kwargs : dictionary
        Additional keyword arguments to pass to pcolormesh() call.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance if ax is None.
    """
    # type checking
    uvp = copy.deepcopy(uvp)
    assert isinstance(uvp, uvpspec.UVPSpec), "input uvp must be a UVPSpec object"
    assert isinstance(spw, (int, np.integer))
    assert isinstance(pol, (int, str, np.integer, np.str))

    # check pspec units for little h
    little_h = 'h^-3' in uvp.norm_units

    # Create new ax if none specified
    new_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.get_figure()

    # Select out times if provided
    if times is not None:
        uvp.select(times=times, inplace=True)

    # Average across redundant groups and time
    # this also ensures blpairs are ordered from short_bl --> long_bl
    blp_grps, lens, angs, tags = utils.get_blvec_reds(uvp, bl_error_tol=red_tol, match_bl_lens=True)
    uvp.average_spectra(blpair_groups=blp_grps, time_avg=True, inplace=True)

    # Convert to DeltaSq
    if deltasq and not delay:
        uvp.convert_to_deltasq(inplace=True)

    # Fold array
    if fold:
        uvp.fold_spectra()

    # Format ticks
    if delay:
        x_axis = uvp.get_dlys(spw) * 1e9
        y_axis = uvp.get_blpair_seps()
    else:
        x_axis = uvp.get_kparas(spw, little_h=little_h)
        y_axis = uvp.get_kperps(spw, little_h=little_h)
    if rotate:
        _x_axis = y_axis
        y_axis = x_axis
        x_axis = _x_axis

    # Conigure Units
    psunits = "({})^2\ {}".format(uvp.vis_units, uvp.norm_units)
    if "h^-1" in psunits: psunits = psunits.replace("h^-1", "h^{-1}\ ")
    if "h^-3" in psunits: psunits = psunits.replace("h^-3", "h^{-3}\ ")
    if "Hz" in psunits: psunits = psunits.replace("Hz", r"{\rm Hz}\ ")
    if "str" in psunits: psunits = psunits.replace("str", r"\,{\rm str}\ ")
    if "Mpc" in psunits and "\\rm" not in psunits: 
        psunits = psunits.replace("Mpc", r"{\rm Mpc}")
    if "pi" in psunits and "\\pi" not in psunits: 
        psunits = psunits.replace("pi", r"\pi")
    if "beam normalization not specified" in psunits:
        psunits = psunits.replace("beam normalization not specified", 
                                 r"{\rm unnormed}")

    # get data casting
    if component == 'real':
        cast = np.real
    elif component == 'imag':
        cast = np.imag
    elif component == 'abs':
        cast = np.abs
    else:
        raise ValueError("Did not understand component {}".format(component))

    # get data with shape (Nblpairs, Ndlys)
    data = cast([uvp.get_data((spw, blp, pol)).squeeze() for blp in uvp.get_blpairs()])

    # take log10
    if log10:
        data = np.log10(np.abs(data))

    # loglog
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # rotate
    if rotate:
        data = np.rot90(data[:, ::-1], k=1)

    # Get bin edges
    xdiff = np.diff(x_axis)
    x_edges = np.array([x_axis[0]-xdiff[0]/2.0] + list(x_axis[:-1]+xdiff/2.0) + [x_axis[-1]+xdiff[-1]/2.0])
    ydiff = np.diff(y_axis)
    y_edges = np.array([y_axis[0]-ydiff[0]/2.0] + list(y_axis[:-1]+ydiff/2.0) + [y_axis[-1]+ydiff[-1]/2.0])
    X, Y = np.meshgrid(x_edges, y_edges)

    # plot 
    cax = ax.pcolormesh(X, Y, data, cmap=cmap, edgecolor=edgecolor, lw=0.01,
                        vmin=vmin, vmax=vmax, **kwargs)

    # Configure ticks
    if set_bl_tick_major:
        if rotate:
            ax.set_xticks(map(lambda x: np.around(x, _get_sigfig(x)+2), x_axis))
        else:
            ax.set_yticks(map(lambda x: np.around(x, _get_sigfig(x)+2), y_axis))
    if set_bl_tick_minor:
        if rotate:
            ax.set_xticks(map(lambda x: np.around(x, _get_sigfig(x)+2), x_axis), minor=True)
        else:
            ax.set_yticks(map(lambda x: np.around(x, _get_sigfig(x)+2), y_axis), minor=True)

    # Add colorbar
    if colorbar:
        if cbax is None:
            cbax = ax
        cbar = fig.colorbar(cax, ax=cbax)
        if deltasq:
            p = "\Delta^2"
        else:
            p = "P"
        if delay:
            p = "{}({},\ {})".format(p, r'\tau', r'|\vec{b}|')
        else:
            p = "{}({},\ {})".format(p, r'k_\parallel', r'k_\perp')
        if log10:
            psunits = r"$\log_{{10}}\ {}\ [{}]$".format(p, psunits)
        else:
            psunits = r"${}\ [{}]$".format(p, psunits)
        cbar.set_label(psunits, fontsize=14)

    # Configure tick labels
    if delay:
        xlabel = r"$\tau$ $[{\rm ns}]$"
        ylabel = r"$|\vec{b}|$ $[{\rm m}]$"
    else:
        xlabel = r"$k_{\parallel}\ [h\ \rm Mpc^{-1}]$"
        ylabel = r"$k_{\perp}\ [h\ \rm Mpc^{-1}]$"
    if rotate:
        _xlabel = ylabel
        ylabel = xlabel
        xlabel = _xlabel
    if ax.get_xlabel() == '':
        ax.set_xlabel(xlabel, fontsize=16)
    if ax.get_ylabel() == '':
        ax.set_ylabel(ylabel, fontsize=16)

    # Configure center line
    if center_line:
        if rotate:
            ax.axhline(y=0, color='#000000', ls='--', lw=lw)
        else:
            ax.axvline(x=0, color='#000000', ls='--', lw=lw)

    # Plot horizons
    if horizon_lines:
        # get horizon in ns
        horizons = uvp.get_blpair_seps() / conversions.units.c * 1e9

        # convert to cosmological wave vector
        if not delay:
            # Get average redshift of spw
            avg_z = uvp.cosmo.f2z(np.mean(uvp.freq_array[uvp.spw_to_freq_indices(spw)]))
            horizons *= uvp.cosmo.tau_to_kpara(avg_z, little_h=little_h) / 1e9

        # iterate over bins and plot lines
        if rotate:
            bin_edges = x_edges
        else:
            bin_edges = y_edges
        for i, hor in enumerate(horizons):
            if rotate:
                ax.plot(bin_edges[i:i+2], [hor, hor], color='#ffffff', ls='--', lw=lw)
                if not uvp.folded:
                    ax.plot(bin_edges[i:i+2], [-hor, -hor], color='#ffffff', ls='--', lw=lw)
            else:
                ax.plot([hor, hor], bin_edges[i:i+2], color='#ffffff', ls='--', lw=lw)
                if not uvp.folded:
                    ax.plot([-hor, -hor], bin_edges[i:i+2], color='#ffffff', ls='--', lw=lw)

    # flip axes
    if flip_xax:
        fig.sca(ax)
        fig.gca().invert_xaxis()
    if flip_yax:
        fig.sca(ax)
        fig.gca().invert_yaxis()

    # add title
    if title is not None:
        ax.set_title(title, fontsize=12)

    # Configure tick sizes and rotation
    [tl.set_size(xtick_size) for tl in ax.get_xticklabels()]
    [tl.set_rotation(xtick_rot) for tl in ax.get_xticklabels()]
    [tl.set_size(ytick_size) for tl in ax.get_yticklabels()]
    [tl.set_rotation(ytick_rot) for tl in ax.get_yticklabels()]

    # return figure
    if new_plot:
        return fig


def plot_uvdata_waterfalls(uvd, basename, data='data', plot_mode='log', 
                           vmin=None, vmax=None, recenter=False, format='png',
                           **kwargs):
    """
    Plot waterfalls for all baselines and polarizations within a UVData object, 
    and save to individual files.
    
    Parameters
    ----------
    uvd : UVData object
        Input data object. Waterfalls will be stored for all baselines and 
        polarizations within the object; use uvd.select() to remove unwanted 
        information.

    basename : str
        Base filename for the output plots. This must have two placeholders 
        for baseline ID ('bl') and polarization ('pol'), 
        e.g. basename='plots/uvdata.{pol}.{bl}'.
    
    data : str, optional
        Which data array to plot from the UVData object. Options are: 
            'data', 'flags', 'nsamples'. Default: 'data'.
    
    plot_mode : str, optional
        Plot mode, passed to uvtools.plot.waterfall. Default: 'log'.
    
    vmin, vmax : float, optional
        Min./max. values of the plot colorscale. Default: None (uses the 
        min./max. of the data).
    
    recenter : bool, optional
        Whether to apply recentering (see uvtools.plot.waterfall). 
        Default: False.
    
    format : str, optional
        The file format of the output images. If None, the image format will be 
        deduced from the basename string. Default: 'png'.
    
    **kwargs : dict
        Keyword arguments passed to uvtools.plot.waterfall, which passes them 
        on to matplotlib.imshow.
    """
    assert isinstance(uvd, UVData), "'uvd' must be a UVData object."
    assert data in ['data', 'flags', 'nsamples'], \
            "'%s' not a valid data array; use 'data', 'flags', or 'nsamples'" \
            % data
    
    # Set plot colorscale max/min if specified
    drng = None
    if vmin is not None: 
        assert vmax is not None, "Must also specify vmax if vmin is specified."
        drng = vmax - vmin
    
    # Empty figure
    fig, ax = plt.subplots(1, 1)
    
    # Loop over antenna pairs and pols
    for (ant1, ant2, pol), d in uvd.antpairpol_iter():
        
        # Get chosen data array
        if data == 'data':
            pass
        elif data == 'flags':
            d = uvd.get_flags((ant1, ant2, pol))
        elif data == 'nsamples':
            d = uvd.get_nsamples((ant1, ant2, pol))
        else:
            raise KeyError("Invalid data array type '%s'" % data)
        
        # Make plot
        img = uvtools.plot.waterfall(d, mode=plot_mode, mx=vmax, drng=drng, 
                                     recenter=recenter, **kwargs)
        fig.colorbar(img)
        
        # Save to file
        outfile = basename.format(bl="%d.%d"%(ant1, ant2), pol=pol)
        if format is not None:
            # Make sure format extension is given
            if outfile[-len(format)].lower() != format.lower():
                outfile = "%s.%s" % (outfile, format)
        fig.tight_layout()
        fig.savefig(outfile, format=format)
        fig.clf()

def _get_sigfig(x):
    return -int(np.floor(np.log10(np.abs(x))))

def _round_sigfig(x, up=True):
    sigfigs = get_sigfig(x)
    if up:
        return np.ceil(10**sigfigs * x) / 10**sigfigs
    else:
        return np.floor(10**sigfigs * x) / 10**sigfigs


def plot_uvdata_vis_hist(uvd, axis, index, fit_curve='Gaussian', plot_mode='normal', show_robust=False,
                           **kwargs):
    """
    Plot histograms for the visibilities. 
    
    Parameters
    ----------
    uvd : UVData object
        Input data object. Waterfalls will be stored for all baselines and 
        polarizations within the object; use uvd.select() to remove unwanted 
        information.

    axis : str
        Along which axis to plot the distribution.
        Choices: ['frequencies', 'baseline-times'].
    
    plot_mode : str
        Plot mode. If choose 'normal', just plot histograms of numbers. 
        If choose 'density', plot histograms of normalized density.
        Choices: ['normal', 'density'].
        Default: 'normal'.
    
    index : int 
        The index in the uvd.data_array to draw. 
    
    show_robust : bool
        Whether or not to show the robust mean and std for the histogram.
        Default: False

    fit_curve : str
        The types of fuctions to fit the distribution, only called when plot_mode == 'density'.
        Chocies: ['Gaussian', 'Exponential']
        Default: 'Gaussian'.
    
    **kwargs : 
    """
    assert isinstance(uvd, UVData), "'uvd' must be a UVData object."
    assert axis in ['frequencies', 'baseline-times'], \
            "'%s' not a valid axis; use 'frequencies' or 'baseline-times'" \
            % axis
    assert plot_mode in ['normal', 'density'], \
            "'%s' not a valid axis; use 'normal' or 'density' " \
            % plot_mode
    assert isinstance(index, int), "'index' must be a integer."        
    if axis == 'frequencies':
        assert index >=0 and index < uvd.Nblts, "The index is not valid."
    if axis == 'baseline-times':
        assert index >=0 and index < uvd.Nfreqs, "The index is not valid."
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Choose axis
    if axis == 'frequencies':
        fig.suptitle('Distribution of the visibility at time {}, baseline {} across frequency axis'.format(uvd.time_array[index], uvd.baseline_array[index]), fontsize=14)
        d = uvd.data_array[index][0, :, 0]
    if axis == 'baseline-times':
        fig.suptitle('Distribution of the visibility at frequency {} across baseline-time axis'.format(uvd.freq_array[0,index]), fontsize=14)
        d = uvd.data_array[:, 0, index, 0]
    # Make plots
    ax_list = [axes[0], axes[1]]
    data_list = [d.real,d.imag]
    for (ax, data) in zip(ax_list, data_list):
        # Normal mode
        if plot_mode == 'normal':
            # Whether or not to show robust statiscics
            if show_robust == True:
                n, bins, patches = ax.hist(data, bins=11,  histtype='step', lw=3, 
                    range=((np.mean(data)-4.*np.std(data)), (np.mean(data)+4.*np.std(data)))
                    ,label='real\n(mean:{:.2f}, \nmean(r):{:.2f}, \nstd:{:.2f},\nstd(r):{:.2f})'.format(np.mean(data), astats.biweight_location(data),np.std(data), np.sqrt(astats.biweight_midvariance(data)))
                    ,color = 'blue' )
            else:
                n, bins, patches = ax.hist(data, bins=11,  histtype='step', lw=3, 
                    range=((np.mean(data)-4.*np.std(data)), (np.mean(data)+4.*np.std(data)))
                    ,label='real\n(mean:{:.2f}, \nstd:{:.2f})'.format(np.mean(data),np.std(data))
                    ,color = 'blue' )
            # Only plot error bars from Poission Distribution in normal mode
            bincenters = 0.5*(bins[1:]+bins[:-1])
            menStd     = np.sqrt(n)
            ax.bar(bincenters, n, width=0., ecolor='red', yerr=menStd)
        # Density mode
        if plot_mode == 'density':
            # Whether or not to show robust statiscics
            if show_robust == True:
                n, bins, patches = ax.hist(data, bins=11,  histtype='step', lw=3, 
                    range=((np.mean(data)-4.*np.std(data)), (np.mean(data)+4.*np.std(data)))
                    ,label='real\n(mean:{:.2f}, \nmean(r):{:.2f}, \nstd:{:.2f},\nstd(r):{:.2f})'.format(np.mean(data), astats.biweight_location(data),np.std(data), np.sqrt(astats.biweight_midvariance(data)))
                    ,color = 'blue', density=True)
            else:
                n, bins, patches = ax.hist(data, bins=11,  histtype='step', lw=3, 
                    range=((np.mean(data)-4.*np.std(data)), (np.mean(data)+4.*np.std(data)))
                    ,label='real\n(mean:{:.2f}, \nstd:{:.2f})'.format(np.mean(data),np.std(data))
                    ,color = 'blue', density=True)
            # Choose type of the fit curve
            assert fit_curve in ['Gaussian', 'Exponential'],\
            "'%s' not a valid axis; use 'Gaussian' or 'Exponential'" % fit_curve
            if fit_curve == 'Gaussian':
                y = stats.norm.pdf(bins, np.mean(data), np.std(data))
                ax.plot(bins, y, lw=2, ls='--', label='Gaussian', color='blue')
            if fit_curve == 'Exponential':
                y = stats.laplace.pdf(bins, np.mean(data), np.std(data)/np.sqrt(2.))
                ax.plot(bins, y, lw=2, ls='--', label='Exponential', color='blue')
        
        ax.legend(fontsize=12, loc='lower center',framealpha=0)
        ax.set_yscale('log')
        ax.grid()

def plot_error(uvp, uvp_td, key, time_index, Tsys, extra_error_types=['time_average'], **kwargs):
    """
    Plot different kinds of error bars in one figure. These include 'nn' and 'sn' like
    errors, which are constructed from time-differenced power spectra, uvp_td. Errors
    which rely on pushing an input covariance matrix through the quadratic estimator are assigned 
    in 'extra_error_types', and the first one ranked in 'extra_error_types' will be chosen 
    as the reference, which means we will plot the ratios of other errors with the referred one.    
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    uvp_td : UVPspec from the data difference
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.

    key : tuple
        Baseline-pair key, in the format (spw, ((ant1, ant2),(ant3, ant4)), pol)
    
    time_index : integer

    Tsys : float
        The input system temperature to call in the uvp.generate_noise_spectra().

    extra_error_types: list of strs
        Extra types for error bars.
        Choices:'time_average', 'time_average_min','time_average_max',
        'time_average_diagonal','time_average_mean'
        Default:['time_average'] 

    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.
    """
    assert isinstance(time_index, int), "time_index must be a integer."
    assert time_index >= 0 and time_index < uvp.Ntimes, "time_index is not valid."
    assert len(extra_error_types) >=1, 'extra error types must be more than one.'

    #Get the analytic variance and the color
    analytic_real, analytic_imag, color_list = odict(), odict(), odict() 
    analytic_real[extra_error_types[0]] = np.sqrt(np.abs(np.diag(uvp.get_cov(key,component='real', cov_model=extra_error_types[0])[time_index])))
    analytic_imag[extra_error_types[0]] = np.sqrt(np.abs(np.diag(uvp.get_cov(key,component='imag', cov_model=extra_error_types[0])[time_index])))
    color_list[extra_error_types[0]] = 'green'
    error_flag = 1
    for error_type in extra_error_types[1:]:
        analytic_real[error_type] = np.sqrt(np.abs(np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[time_index])))
        analytic_imag[error_type] = np.sqrt(np.abs(np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[time_index])))
        #Use rgb colors.
        color_list[error_type] = (0.5, abs(np.sin(error_flag)), 0.5)
        error_flag += 1
    
    #The power spectra of time difference data, thought to be error bars in noise dominant regions, 
    pnn_real = np.abs(uvp_td.get_data(key)[time_index].real)/2. 
    pnn_imag = np.abs(uvp_td.get_data(key)[time_index].imag)/2. 
   
    #The products of power spectra of time difference data and power spectra of original data, 
    #thought to be error bars in foreground dominant regions.
    psn_real = np.sqrt(np.abs(uvp.get_data(key)[time_index].real * uvp_td.get_data(key)[time_index].real)) 
    psn_imag = np.sqrt(np.abs(uvp.get_data(key)[time_index].real * uvp_td.get_data(key)[time_index].imag)) 
    
    if not uvp.norm == 'Unnormalized':
        noise = uvp.generate_noise_spectra(0, 'xx', Tsys)[uvp.antnums_to_blpair(key[1])]

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Error bars at time {} on baseline-pair {}".format((np.unique(uvp.time_1_array)[time_index],np.unique(uvp.time_2_array)[time_index]), key[1]), y=0.55, fontsize=12)
    
    dlys = uvp.get_dlys(0) * 1e9
    # The ratio of error bars to the original one
    ax0 = fig.add_axes([0.1, 0.0, 0.4, 0.1])
    ax0.plot(dlys, psn_real / analytic_real[extra_error_types[0]] , c='blue')
    ax0.plot(dlys, pnn_real / analytic_real[extra_error_types[0]], c='orange')
    for error_type in extra_error_types[1:]: 
        ax0.plot(dlys, analytic_real[error_type] / analytic_real[extra_error_types[0]], c=color_list[error_type])
    if not uvp.norm == 'Unnormalized':  
        ax0.plot(dlys, noise[time_index]/analytic_real[extra_error_types[0]], c='k')
    ax0.axhline(y=1, c='g')
    ax0.set_xlabel(r"$\tau$ $[{\rm ns}]$",fontsize=12)
    ax0.semilogy()
    # Plot different kinds of error bars
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
    ax1.plot(dlys, psn_real, label=r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$', c='blue')
    ax1.plot(dlys, pnn_real , label=r'$P_{nn}}$', c='orange')
    for error_type in extra_error_types: 
        ax1.plot(dlys, analytic_real[error_type], label='analytic_'+error_type, c=color_list[error_type])
    if not uvp.norm == 'Unnormalized':
        ax1.plot(dlys, noise[time_index], c='k', label='thermal noise')
    ax1.legend(loc='best', framealpha=0)
    ax1.set_xticklabels('')
    ax1.semilogy()
    # The ratio of error bars to the original one
    ax2 = fig.add_axes([0.5, 0.0, 0.4, 0.1])
    ax2.plot(dlys, psn_imag / analytic_imag[extra_error_types[0]] , c='blue')
    ax2.plot(dlys, pnn_imag / analytic_imag[extra_error_types[0]],  c='orange')
    for error_type in extra_error_types[1:]:
        ax2.plot(dlys, analytic_imag[error_type] / analytic_imag[extra_error_types[0]], c=color_list[error_type])
    if not uvp.norm == 'Unnormalized': 
        ax2.plot(dlys, noise[time_index]/analytic_imag[extra_error_types[0]], c='k')
    ax2.axhline(y=1, c='g')
    ax2.set_xlabel(r"$\tau$ $[{\rm ns}]$",fontsize=12)
    ax2.semilogy()
    ax2.yaxis.tick_right()
    # Plot different kinds of error bars
    ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
    ax3.plot(dlys, psn_imag, label=r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$', c='blue')
    ax3.plot(dlys, pnn_imag , label=r'$P_{nn}}$', c='orange')
    for error_type in extra_error_types: 
        ax3.plot(dlys, analytic_imag[error_type], label='analytic_'+error_type, c=color_list[error_type])
    if not uvp.norm == 'Unnormalized':
        ax3.plot(dlys, noise[time_index], c='k', label='thermal noise')
    ax3.legend(loc='best', framealpha=0)
    ax3.semilogy()
    ax3.yaxis.tick_right()
    ax3.set_xticklabels('')

def plot_zscore_hist(uvp, uvp_td, key, wedge, inside_wedge=True, extra_error_types=['time_average'], fit_curve='Gaussian', plot_mode='normal', show_robust=True, **kwargs):
    """
    Plot the distribution of z-scores on a baseline-pair across time axis and delay axis.
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    uvp_td : UVPspec from the data difference
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.

    key : tuple
        Baseline-pair key, in the format (spw, ((ant1, ant2),(ant3, ant4)), pol)
    
    wedge : float
        The position of the wedge in the delay space. 

    inside_wedge : bool
        The choice to plot the histograms of z-scores inside or outside the wedge.  

    extra_error_types: list of strs
        Extra types for error bars.
        Choices:'time_average', 'time_average_min','time_average_max',
        'time_average_diagonal','time_average_mean'
        Default:['time_average'] 

    plot_mode : str
        Plot mode. If choose 'normal', just plot histograms of numbers. 
        If choose 'density', plot histograms of normalized density.
        Choices: ['normal', 'density'].
        Default: 'normal'.
    
    show_robust : bool
        Whether or not to show the robust mean and std for the histogram.
        Default: False

    fit_curve : str
        The types of fuctions to fit the distribution, only called when plot_mode == 'density'.
        Chocies: ['Gaussian', 'Exponential']
        Default: 'Gaussian'.

    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.
    """
    dlys = uvp.get_dlys(0)*1e9
    assert wedge <= np.max(dlys) and wedge >= np.min(dlys), "The 'wedge' is not valid." 
    assert len(extra_error_types) >=1, 'extra error types must be more than one.'

    # Dictionary of error bars.
    error_type_list_real = odict()
    error_type_list_imag = odict()
    label_list = odict()

    # For the case inside and outside the wedge, plot different kinds of error bars.
    # Inside the wedge: pnn
    # Outside the wedge: psn
    if inside_wedge==True:
        c_td = 'blue'
        pre_td = r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'
        error_type_list_real[pre_td] = np.sqrt(np.abs(uvp.get_data(key)[:uvp_td.Ntimes].real * uvp_td.get_data(key)[:uvp_td.Ntimes].real)).T[abs(dlys)<wedge].reshape(-1)
        error_type_list_imag[pre_td] = np.sqrt(np.abs(uvp.get_data(key)[:uvp_td.Ntimes].real * uvp_td.get_data(key)[:uvp_td.Ntimes].imag)).T[abs(dlys)<wedge].reshape(-1)
        label_list[pre_td] = pre_td

        for error_type in extra_error_types:
            error_type_list_real[error_type] = np.sqrt(np.abs([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[i]) for i in range(uvp_td.Ntimes)])).T[abs(dlys)<wedge].reshape(-1)
            error_type_list_imag[error_type] = np.sqrt(np.abs([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[i]) for i in range(uvp_td.Ntimes)])).T[abs(dlys)<wedge].reshape(-1)
            label_list[error_type] = 'analytic_'+error_type
        
        p_real = uvp.get_data(key)[:uvp_td.Ntimes].real.T[abs(dlys)<wedge].reshape(-1)
        p_real -= np.mean(p_real)
        
        p_imag = uvp.get_data(key)[:uvp_td.Ntimes].imag.T[abs(dlys)<wedge].reshape(-1)
        p_imag -= np.mean(p_imag)
    
    else:
        c_td = 'orange'
        pre_td = r'$P_{nn}$'
        error_type_list_real[pre_td] = (np.abs(uvp_td.get_data(key)[:uvp_td.Ntimes].real)/2.).T[abs(dlys)>wedge].reshape(-1)
        error_type_list_imag[pre_td] = (np.abs(uvp_td.get_data(key)[:uvp_td.Ntimes].imag)/2.).T[abs(dlys)>wedge].reshape(-1)
        label_list[pre_td] = pre_td

        for error_type in extra_error_types:
            error_type_list_real[error_type] = np.sqrt(np.abs([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[i]) for i in range(uvp_td.Ntimes)])).T[abs(dlys)>wedge].reshape(-1)
            error_type_list_imag[error_type] = np.sqrt(np.abs([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[i]) for i in range(uvp_td.Ntimes)])).T[abs(dlys)>wedge].reshape(-1)
            label_list[error_type] = 'analytic_'+error_type

        p_real = uvp.get_data(key)[:uvp_td.Ntimes].real.T[abs(dlys)>wedge].reshape(-1)
        p_real -= np.mean(p_real)
        
        p_imag = uvp.get_data(key)[:uvp_td.Ntimes].imag.T[abs(dlys)>wedge].reshape(-1)
        p_imag -= np.mean(p_imag)
    # Types of error bars and the colors
    error_types = [pre_td,] + extra_error_types 
    colors = [c_td,'green'] + [(0.5, abs(np.sin(error_flag)), 0.5) for error_flag in range(1, len(extra_error_types))]      
    
    nrow = len(error_types)
    height = 4*nrow
    # Make plots
    fig, axes = plt.subplots(nrow,2,figsize=(10, height))

    if inside_wedge==True:
        fig.suptitle("Distribution of z-scores (inside the wedge) on blp {} across time axis and across delay axis".format(key[1]),fontsize=12)
    else:
        fig.suptitle("Distribution of z-scores (outside the wedge) on blp {} across time axis and across delay axis".format(key[1]),fontsize=12)
    
    for j in range(nrow):
        d_list = []
        d_list.append(p_real / error_type_list_real[error_types[j]])
        d_list.append(p_imag / error_type_list_imag[error_types[j]])
        pre_list = ['_real', '_imag']
        for index in range(0,2):  
            ax = axes[j,index]
            d = d_list[index]
            mean = np.mean(d)
            std = np.std(d)
            if plot_mode == 'normal':
                density = False
            if plot_mode == 'density':
                density = True
            # Whether or not to show robust statistics
            if show_robust == True:
                mean_r = astats.biweight_location(d)
                std_r = np.sqrt(astats.biweight_midvariance(d))
                mean_plot = mean_r
                std_plot = std_r
                n, bins, patches = ax.hist(d, bins=11, range=((mean_r-2.*std_r),(mean_r+2.*std_r)), 
                    histtype='step', lw=3, density=density,
                    label=label_list[error_types[j]] + pre_list[index] + '\nmean:{:.2f}\nmean(r):{:.2f}\nstd:{:.2f}\nstd(r):{:.2f}'.format(mean,mean_r,std,std_r), color=colors[j])
            else:
                mean_plot = mean
                std_plot = std
                n, bins, patches = ax.hist(d, bins=11, range=((mean-2.*std),(mean+2.*std)), 
                    histtype='step', lw=3, density=density,
                    label=label_list[error_types[j]] + pre_list[index] + '\nmean:{:.2f}\nstd:{:.2f}'.format(mean,std), color=colors[j])
            # Only plot error bars from Poission distribution in normal mode
            if plot_mode == 'normal':    
                bincenters = 0.5*(bins[1:]+bins[:-1])
                menStd     = np.sqrt(n)
                ax.bar(bincenters, n, width=0., ecolor='c', yerr=menStd)
            # Choose the type of fit curve
            if plot_mode == 'density':  
                assert fit_curve in ['Gaussian', 'Exponential'],\
                "'%s' not a valid axis; use 'Gaussian' or 'Exponential'" % fit_curve
                if fit_curve == 'Gaussian':
                    y = stats.norm.pdf(bins, mean_plot, std_plot)
                    ax.plot(bins, y, lw=2, ls='--', label='Gaussian', color=colors[j])
                if fit_curve == 'Exponential':
                    y = stats.laplace.pdf(bins, mean_plot, std_plot/np.sqrt(2.))
                    ax.plot(bins, y, lw=2, ls='--', label='Exponential', color=colors[j])
            
            ax.legend(loc='upper left', framealpha=0, fontsize=12)

def plot_zscore_blpt_hist(uvp, uvp_td, dly, wedge, spw, pol, blpairs, extra_error_types=['time_average'], fit_curve='Gaussian', plot_mode='normal', show_robust=True, **kwargs):
    """
    Plot the distribution of the z-scores in the same delay bin across blpt axis.
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    uvp_td : UVPspec from the data difference
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.

    dly : float
        The delay time.
    
    wedge : float
        The position of the wedge in the delay space. 

    spw : int
        The index for the spectral window.

    pol : str
        The polarization type.

    blpairs : list
        The list of the baseline-pairs.

    extra_error_types: list of strs
        Extra types for error bars.
        Choices:'time_average', 'time_average_min','time_average_max',
        'time_average_diagonal','time_average_mean'
        Default:['time_average'] 

    plot_mode : str
        Plot mode. If choose 'normal', just plot histograms of numbers. 
        If choose 'density', plot histograms of normalized density.
        Choices: ['normal', 'density'].
        Default: 'normal'.
    
    show_robust : bool
        Whether or not to show the robust mean and std for the histogram.
        Default: False

    fit_curve : str
        The types of fuctions to fit the distribution, only called when plot_mode == 'density'.
        Chocies: ['Gaussian', 'Exponential']
        Default: 'Gaussian'.

    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.
    """
    dlys = uvp.get_dlys(0)*1e9
    assert wedge <= np.max(dlys) and wedge >= np.min(dlys), "The 'wedge' is not valid."
    assert dly <= np.max(dlys) and dly >= np.min(dlys), "The 'dly' is not valid."
    dly_ind = np.argmin(abs(dlys-dly))
    assert isinstance(blpairs, list), "blpairs must a list."
    assert isinstance(blpairs[0], tuple), "blpairs must a list of baseline-pairs."
    assert len(extra_error_types) >=1, 'extra error types must be more than one.'


    key_list = []
    for blpair in blpairs:
        key = (spw, blpair, pol)
        key_list.append(key)
    # Dicts of error bars
    error_type_list_real = odict()
    error_type_list_imag = odict()
    label_list = odict()
    # Plot different kinds of error bars inside and outside the wedge
    if abs(dly)<=abs(wedge):
        c_td = 'blue'
        pre_td = r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'
        label_list[pre_td] = pre_td
        error_type_list_real[pre_td] = np.sqrt(np.abs(np.array([uvp.get_data(key)[:uvp_td.Ntimes,dly_ind].real * uvp_td.get_data(key)[:,dly_ind].real for key in key_list]).reshape(-1)))
        error_type_list_imag[pre_td] = np.sqrt(np.abs(np.array([uvp.get_data(key)[:uvp_td.Ntimes,dly_ind].real * uvp_td.get_data(key)[:,dly_ind].imag for key in key_list]).reshape(-1)))
    else:
        c_td = 'orange'
        pre_td = r'$P_{nn}$'
        label_list[pre_td] = pre_td
        error_type_list_real[pre_td] = np.abs(np.array([uvp_td.get_data(key)[:,dly_ind].real/2. for key in key_list]).reshape(-1))
        error_type_list_imag[pre_td] = np.abs(np.array([uvp_td.get_data(key)[:,dly_ind].imag/2. for key in key_list]).reshape(-1))

    # Get the analytic variance 
    for error_type in extra_error_types:
        error_type_list_real[error_type] = np.sqrt(np.abs(np.array([np.array([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,dly_ind] for key in key_list]).reshape(-1)))
        error_type_list_imag[error_type] = np.sqrt(np.abs(np.array([np.array([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,dly_ind] for key in key_list]).reshape(-1)))
        label_list[error_type] = 'analytic_'+error_type

    p_real = np.array([uvp.get_data(key)[:uvp_td.Ntimes,dly_ind].real for key in key_list]).reshape(-1)
    p_real -= np.mean(p_real)
    p_imag = np.array([uvp.get_data(key)[:uvp_td.Ntimes,dly_ind].imag for key in key_list]).reshape(-1)
    p_imag -= np.mean(p_imag)
    # Error types and colors   
    error_types = [pre_td,] + extra_error_types 
    colors = [c_td,'green'] + [(0.5, abs(np.sin(error_flag)), 0.5) for error_flag in range(1, len(extra_error_types))]      
    nrow = len(error_types)
    height = 4*nrow

    fig, axes = plt.subplots(nrow,2,figsize=(10, height))
    fig.suptitle("Distribution of z-scores at tau {} ns across blpt axis".format(dlys[dly_ind]),fontsize=12)
     
    for j in range(nrow):
        d_list = []
        d_list.append(p_real / error_type_list_real[error_types[j]])
        d_list.append(p_imag / error_type_list_imag[error_types[j]])
        pre_list = ['_real', '_imag']
        for index in range(0,2):  
            ax = axes[j,index]
            d = d_list[index]
            mean = np.mean(d)
            std = np.std(d)
            if plot_mode == 'normal':
                density = False
            if plot_mode == 'density':
                density = True
            # Whether or not to show the robust statistics
            if show_robust == True:
                mean_r = astats.biweight_location(d)
                std_r = np.sqrt(astats.biweight_midvariance(d))
                mean_plot = mean_r
                std_plot = std_r
                n, bins, patches = ax.hist(d, bins=11, range=((mean_r-2.*std_r),(mean_r+2.*std_r)), 
                    histtype='step', lw=3, density=density,
                    label=label_list[error_types[j]]+pre_list[index]+'\nmean:{:.2f}\nmean(r):{:.2f}\nstd:{:.2f}\nstd(r):{:.2f}'.format(mean,mean_r,std,std_r), color=colors[j])
            else:
                mean_plot = mean
                std_plot = std
                n, bins, patches = ax.hist(d, bins=11, range=((mean-2.*std),(mean+2.*std)), 
                    histtype='step', lw=3, density=density,
                    label=label_list[error_types[j]]+pre_list[index]+'\nmean:{:.2f}\nstd:{:.2f}'.format(mean,std), color=colors[j])
            # Only plot error bars from Poission distribution in normal mode
            if plot_mode == 'normal':    
                bincenters = 0.5*(bins[1:]+bins[:-1])
                menStd     = np.sqrt(n)
                ax.bar(bincenters, n, width=0., ecolor='c', yerr=menStd)
            # Choose the type of fit curve
            if plot_mode == 'density':  
                assert fit_curve in ['Gaussian', 'Exponential'],\
                "'%s' not a valid axis; use 'Gaussian' or 'Exponential'" % fit_curve
                if fit_curve == 'Gaussian':
                    y = stats.norm.pdf(bins, mean_plot, std_plot)
                    ax.plot(bins, y, lw=2, ls='--', label='Gaussian', color=colors[j])
                if fit_curve == 'Exponential':
                    y = stats.laplace.pdf(bins, mean_plot, std_plot/np.sqrt(2.))
                    ax.plot(bins, y, lw=2, ls='--', label='Exponential', color=colors[j])
            
            ax.legend(loc='upper left', framealpha=0, fontsize=12)

def plot_error_blpt_avg(mode, uvp, uvp_td, spw, pol, blpairs, Tsys, average_method ='optimal', blpt_weights=None, 
    extra_error_types=['time_average'], **kwargs):
    """
    For different kinds of error bars, just plot their average across baseline-pair-times axis,
    or plot the average ones and the original ones at all available baseline-pairs&times together,
    depending on the input in parameter 'mode'.
    These error bars include 'nn' and 'sn' like errors, which are constructed from time-differenced power spectra, uvp_td.
    We aslo plot the bootstrap error bar based on resampling the average inside the spectra group. 
    Errors which rely on pushing an input covariance matrix through the quadratic estimator are assigned 
    in 'extra_error_types', and the first one ranked in 'extra_error_types' will be chosen 
    as the reference, which means we will plot the ratios of other errors with the referred one.  
    
    Parameters
    ----------
    mode : str
        Options : 'blpt', 'blpt_avg'.
        If mode == 'blpt', plot the average errors and the original errors at all available baseline-pairs&times together. 
        If mode == 'blpt_avg', just plot the average error bars.

    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.
    
    uvp_td : UVPspec from the data difference
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.

    spw : int
        The index for the spectral window.

    pol : str
        The polarization type.

    blpairs : list
        The list of baseline-pairs.
    
    Tsys : float
        The system temperature. 

    average_method : str
        Specify the method to carry on averaging.
        Options : 'simple', 'optimal'.
        For 'optimal' method, please refer to M. Tegmark, ApJ, 480, L87 (1997), arXiv:astro-ph/9611130
        and grouping.average_spectra_with_error.
        For 'simple' method, please refer to grouping.average_spectra. 

    blpt_weights : list of weights (float or int), optional
        Relative weight of each baseline-pair-time when performing the average. This
        is useful for simple average with weights and bootstrapping. This should have the same size with
        the number of blpair-pair-time indices.  
        Default: None.
        
    extra_error_types: list of strs
        Extra types for error bars.
        Choices:'time_average', 'time_average_min','time_average_max',
        'time_average_diagonal','time_average_mean'
        Default:['time_average'] 

    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.
    """
    assert mode == 'blpt' or mode == 'blpt_avg', 'Unvalid choice for modes.'
    assert len(extra_error_types) >=1, 'extra error types must be more than one.'

    # Dicts of average analytic error bars.
    analytic_real_avg_blpt = odict([[error_type, []] for error_type in extra_error_types])
    analytic_imag_avg_blpt = odict([[error_type, []] for error_type in extra_error_types])
    
    # Dicts of colors
    color_list = odict()
    
    # Dicts of labels
    label_list = odict()

    #The power spectra of time difference data, thought to be error bars in noise dominant regions, 
    psn_real_avg_blpt = []
    psn_imag_avg_blpt = []
    
    #The products of power spectra of time difference data and power spectra of original data, 
    #thought to be error bars in foreground dominant regions, 
    pnn_real_avg_blpt = []
    pnn_imag_avg_blpt = []
    
    #noise power spectrum
    noise_avg_blpt = []

    if not uvp.norm == 'Unnormalized':    
        noise = uvp_td.generate_noise_spectra(spw, pol, Tsys, blpairs=blpairs)

    # Get all keys
    key_list = []
    for blpair in blpairs:
        key = (spw, blpair, pol)
        key_list.append(key)

    # Get weights
    if blpt_weights is None:
        # Assign unity weights to baseline-pair groups that were specified
        blpt_weights = np.array([[1. for item in range(uvp_td.Ntimes)] for key in key_list]).reshape(-1)
    else:
        # Check that blpt_weights has the shape as (Nblpairs, Ntimes)
        blpt_weights = abs(np.array(blpt_weights).reshape(-1))
        assert len(blpt_weights) == len(key_list)*uvp_td.Ntimes
    
    # Multiply blpt_weights with the integration time     
    nsmp = np.array([[uvp.get_nsamples(key)[time] for time in range(uvp_td.Ntimes)] for key in key_list]).reshape(-1)
    ints = np.array([[uvp.get_integrations(key)[time] for time in range(uvp_td.Ntimes)] for key in key_list]).reshape(-1)
    w = (ints * np.sqrt(nsmp))
    w = w*blpt_weights
    
    Nspwdlys = len(uvp.get_dlys(spw)) 
    # Get the average error bars across blpt axis
    for i in range(Nspwdlys):
        # Choose the average method to be 'simple'
        if average_method == 'simple':
            for error_type in extra_error_types:
                n = np.array([np.array([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,i] for key in key_list]).reshape(-1)
                analytic_real_avg_blpt[error_type].append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
                n = np.array([np.array([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,i] for key in key_list]).reshape(-1)
                analytic_imag_avg_blpt[error_type].append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
            
            n = np.array([uvp.get_data(key)[:uvp_td.Ntimes,i].real * uvp_td.get_data(key)[:,i].real for key in key_list]).reshape(-1)
            psn_real_avg_blpt.append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
            n = np.array([uvp.get_data(key)[:uvp_td.Ntimes,i].real * uvp_td.get_data(key)[:,i].imag for key in key_list]).reshape(-1)
            psn_imag_avg_blpt.append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
            
            n = np.array([uvp_td.get_data(key)[:,i].real/2. for key in key_list]).reshape(-1)**2
            pnn_real_avg_blpt.append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
            n = np.array([uvp_td.get_data(key)[:,i].imag/2. for key in key_list]).reshape(-1)**2
            pnn_imag_avg_blpt.append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))
            
            if not uvp.norm == 'Unnormalized':
                n = np.array([noise[uvp.antnums_to_blpair(blp)][:, i] for blp in blpairs]).reshape(-1)**2
                noise_avg_blpt.append(np.sum(n*w, axis=0)/np.sum(w, axis=0).clip(1e-10, np.inf))

        # Choose the average method to be 'optimal'
        if average_method == 'optimal':
            p = np.array([uvp.get_data(key)[:uvp_td.Ntimes, i].real for key in key_list]).reshape(-1)
            for error_type in extra_error_types:
                n = np.array([np.array([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,i] for key in key_list]).reshape(-1)
                P, N = grouping.average_spectra_with_error(p,n)
                analytic_real_avg_blpt[error_type].append(N)
                n = np.array([np.array([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)])[:,i] for key in key_list]).reshape(-1)
                P, N = grouping.average_spectra_with_error(p,n)
                analytic_imag_avg_blpt[error_type].append(N)
           
            n = np.array([uvp.get_data(key)[:uvp_td.Ntimes,i].real * uvp_td.get_data(key)[:,i].real for key in key_list]).reshape(-1)
            P, N = grouping.average_spectra_with_error(p,n)
            psn_real_avg_blpt.append(N)
            n = np.array([uvp.get_data(key)[:uvp_td.Ntimes,i].real * uvp_td.get_data(key)[:,i].imag for key in key_list]).reshape(-1)
            P, N = grouping.average_spectra_with_error(p,n)
            psn_imag_avg_blpt.append(N)

            n = np.array([uvp_td.get_data(key)[:,i].real/2. for key in key_list]).reshape(-1)**2
            P, N = grouping.average_spectra_with_error(p,n)
            pnn_real_avg_blpt.append(N)
            n = np.array([uvp_td.get_data(key)[:,i].imag/2. for key in key_list]).reshape(-1)**2
            P, N = grouping.average_spectra_with_error(p,n)
            pnn_imag_avg_blpt.append(N)
            
            if not uvp.norm == 'Unnormalized':
                n = np.array([noise[uvp.antnums_to_blpair(blp)][:, i] for blp in blpairs]).reshape(-1)**2
                P, N = grouping.average_spectra_with_error(p,n)
                noise_avg_blpt.append(N)

    # Get the square roots of the above results
    for error_type in extra_error_types:    
        analytic_real_avg_blpt[error_type] = np.sqrt(np.abs(np.array(analytic_real_avg_blpt[error_type]).reshape(-1)))
        analytic_imag_avg_blpt[error_type] = np.sqrt(np.abs(np.array(analytic_imag_avg_blpt[error_type]).reshape(-1)))
    psn_real_avg_blpt = np.sqrt(np.abs(np.array(psn_real_avg_blpt).reshape(-1)))
    psn_imag_avg_blpt = np.sqrt(np.abs(np.array(psn_imag_avg_blpt).reshape(-1)))
    pnn_real_avg_blpt = np.sqrt(np.abs(np.array(pnn_real_avg_blpt).reshape(-1)))
    pnn_imag_avg_blpt = np.sqrt(np.abs(np.array(pnn_imag_avg_blpt).reshape(-1)))
    if not uvp.norm == 'Unnormalized': 
        noise_avg_blpt = np.sqrt(np.abs(np.array(noise_avg_blpt).reshape(-1)))

    # Generate Bootstrap errors
    boots = []
    spectra = np.array([uvp.get_data(key)[:uvp_td.Ntimes, :] for key in key_list]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)
    for i in range(100):
        select = np.random.choice(np.arange(len(spectra)), len(spectra), replace=True)
        boots.append(np.sum(spectra[select]*w[:, None][select], axis=0)/np.sum(w[select], axis=0).clip(1e-10, np.inf))
    boot_blpt = np.std(np.real(boots), axis=0) + 1j*np.std(np.imag(boots), axis=0)
    
    # Only plot 'thermal noise' when uvp.norm is not 'Unnormalized'
    if not uvp.norm == 'Unnormalized':         
        error_types = [r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$', r'$P_{nn}}$', 'thermal noise'] + extra_error_types 
    else:
        error_types = [r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$', r'$P_{nn}}$'] + extra_error_types     

    # Dicts for avg error bars and error bars at all blpts
    error_avg_list_real = odict()
    error_avg_list_imag = odict()
    error_blpt_list_real = odict()
    error_blpt_list_imag = odict()
    
    # Pack all the data into dicts
    error_avg_list_real[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = psn_real_avg_blpt
    error_avg_list_imag[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = psn_imag_avg_blpt
    if mode == 'blpt':
        error_blpt_list_real[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = np.sqrt(np.abs(np.array([uvp.get_data(key)[:uvp_td.Ntimes,].real * uvp_td.get_data(key).real for key in key_list]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)))
        error_blpt_list_imag[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = np.sqrt(np.abs(np.array([uvp.get_data(key)[:uvp_td.Ntimes,].real * uvp_td.get_data(key).imag for key in key_list]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)))
    color_list[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = 'blue'
    label_list[r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'] = r'$\sqrt{2 {Re}(P_{dd})P_{nn}}$'

    error_avg_list_real[r'$P_{nn}}$'] = pnn_real_avg_blpt
    error_avg_list_imag[r'$P_{nn}}$'] = pnn_imag_avg_blpt
    if mode == 'blpt':
        error_blpt_list_real[r'$P_{nn}}$'] = np.abs(np.array([uvp_td.get_data(key).real/2. for key in key_list]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys))   
        error_blpt_list_imag[r'$P_{nn}}$'] = np.abs(np.array([uvp_td.get_data(key).imag/2. for key in key_list]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys))
    color_list[r'$P_{nn}}$'] = 'orange' 
    label_list[r'$P_{nn}}$'] = r'$P_{nn}}$'

    if not uvp.norm == 'Unnormalized':
        error_avg_list_real['thermal noise'] = noise_avg_blpt
        error_avg_list_imag['thermal noise'] = noise_avg_blpt
        if mode == 'blpt':
            error_blpt_list_real['thermal noise'] = np.array([noise[uvp.antnums_to_blpair(blp)] for blp in blpairs]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)
            error_blpt_list_imag['thermal noise'] = np.array([noise[uvp.antnums_to_blpair(blp)] for blp in blpairs]).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)
        color_list['thermal noise'] = 'k'
        label_list['thermal noise'] = 'thermal noise'

    for error_type in extra_error_types:
        label_list[error_type] = 'analytic_' + error_type
        error_avg_list_real[error_type] = analytic_real_avg_blpt[error_type]
        error_avg_list_imag[error_type] = analytic_imag_avg_blpt[error_type] 
        if mode == 'blpt':
            error_blpt_list_real[error_type] = np.sqrt(np.abs(np.array([np.array([np.diag(uvp.get_cov(key,component='real', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)]) for key in key_list]))).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)
            error_blpt_list_imag[error_type] = np.sqrt(np.abs(np.array([np.array([np.diag(uvp.get_cov(key,component='imag', cov_model=error_type)[time]) for time in range(uvp_td.Ntimes)]) for key in key_list]))).reshape(len(key_list)*uvp_td.Ntimes, Nspwdlys)
    color_list[extra_error_types[0]] = 'green'
    error_flag = 1
    for error_type in extra_error_types[1:]:
        color_list[error_type] = (0.5, abs(np.sin(error_flag)), 0.5)
        error_flag += 1
        
    # Make plots
    if mode == 'blpt_avg':
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Blpt-averaged error bars ", y=0.55, fontsize=12)
        dlys = uvp.get_dlys(0) * 1e9
        # Plot the ratio of error bars to the analytic-original one
        ax0 = fig.add_axes([0.1, 0.0, 0.4, 0.1])
        for error_type in error_types: 
            ax0.plot(dlys, error_avg_list_real[error_type] / error_avg_list_real[extra_error_types[0]] , c=color_list[error_type])
        ax0.plot(dlys, boot_blpt.real/ error_avg_list_real[extra_error_types[0]], c='red')
        ax0.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
        ax0.semilogy()
        # Plot different kinds of error bars
        ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
        for error_type in error_types:
            ax1.plot(dlys, error_avg_list_real[error_type], label=label_list[error_type], c=color_list[error_type])
        ax1.plot(dlys, boot_blpt.real, label='bootstrap', c='red')
        ax1.legend(loc='upper left', framealpha=0)
        ax1.set_xticklabels('')
        ax1.semilogy()
        # Plot the ratio of error bars to the analytic-original one
        ax2 = fig.add_axes([0.5, 0.0, 0.4, 0.1])
        for error_type in error_types: 
            ax2.plot(dlys, error_avg_list_imag[error_type] / error_avg_list_imag[extra_error_types[0]] , c=color_list[error_type])
        ax2.plot(dlys, boot_blpt.imag/ error_avg_list_imag[extra_error_types[0]], c='red')
        ax2.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
        ax2.semilogy()
        ax2.yaxis.tick_right()
        # Plot different kinds of error bars
        ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
        for error_type in error_types:
            ax3.plot(dlys, error_avg_list_imag[error_type], label=label_list[error_type], c=color_list[error_type])
        ax3.plot(dlys, boot_blpt.imag, label='bootstrap', c='red')
        ax3.legend(loc='upper left', framealpha=0)
        ax3.semilogy()
        ax3.yaxis.tick_right()
        ax3.set_xticklabels('')

    if mode == 'blpt':
        nrow = len(error_types)
        height = 4*nrow
        # Make plots
        fig, axes = plt.subplots(nrow,2,figsize=(12, height))
        fig.suptitle("Error bars on different blpts", fontsize=12)
        dlys = uvp.get_dlys(0) * 1e9
        # In each row draw one type of error bars,
        # and left column is for real components and right column is for imaginary components
        for (j,error_type) in zip(range(nrow), error_types):
            ax = axes[j,0]
            for i in range(len(error_blpt_list_real[error_type])):
                ax.plot(dlys, error_blpt_list_real[error_type][i], ls='--')
            ax.plot(dlys, error_avg_list_real[error_type], c=color_list[error_type], label= label_list[error_type] +'(average)')
            ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
            ax.legend(loc='upper left')
            ax.set_yscale('log')

            ax = axes[j,1]
            for i in range(len(error_blpt_list_imag[error_type])):
                ax.plot(dlys, error_blpt_list_imag[error_type][i], ls='--')
            ax.plot(dlys, error_avg_list_imag[error_type], c=color_list[error_type], label= label_list[error_type] +'(average)')
            ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
            ax.legend(loc='upper left')
            ax.set_yscale('log')

def imshow_cov(uvp, key, time_index, error_type, **kwargs):
    """
    Plot the analytic covariance matrix.
    
    Parameters
    ----------
    uvp : UVPspec
        UVPSpec object, containing delay spectra for a set of baseline-pairs, 
        times, polarizations, and spectral windows.

    key : tuple
        Baseline-pair key, in the format (spw, ((ant1, ant2),(ant3, ant4)), pol)
    
    time_index : integer

    error_type: str
    Extra types for error bars.
    Choices:'time_average', 'time_average_min','time_average_max',
    'time_average_diagonal','time_average_mean'
    
    kwargs : dict, optional
        Extra kwargs to pass to _all_ ax.plot calls.
    """
    # Check if the uvp object has attribute cov_array
    assert hasattr(uvp,'cov_array_real'), "No covariance array has been calculated for the input UVPspec object."
    assert isinstance(time_index, int), "time_index must be a integer."
    assert time_index >= 0 and time_index < uvp.Ntimes, "time_index is not valid."
    # Get covariance matrix
    cov_real = np.abs(uvp.get_cov(key, component='real', cov_model=error_type)[time_index])
    cov_imag = np.abs(uvp.get_cov(key, component='imag', cov_model=error_type)[time_index])
    dlys = np.array(uvp.get_dlys(0)*1e9, dtype=np.int)
    # Make plots
    fig = plt.figure(figsize=(6,10))
    fig.suptitle("Bandpower covariance matrix \nat time {} \non baseline-pair {}".format((np.unique(uvp.time_1_array)[time_index],np.unique(uvp.time_2_array)[time_index]), key[1]), 
                 x=0.45,y=1.09, fontsize=12)
    
    ax =fig.add_axes([0.1,0.1,0.7,0.4])
    ax.imshow(cov_imag, origin='lower', cmap='bwr', norm=matplotlib.colors.LogNorm(vmin=cov_imag.min(), vmax=cov_imag.max()))
    ax.set_xticks(np.arange(0, len(dlys),len(dlys)/9))
    ax.set_xticklabels(dlys[0::len(dlys)/9])
    ax.set_yticks(np.arange(0, len(dlys),len(dlys)/9))
    ax.set_yticklabels(dlys[0::len(dlys)/9])
    ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
    ax.set_ylabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
    ax.set_title('Imaginary part', fontsize=12)

    ax = fig.add_axes([0.1,0.6,0.7,0.4])
    pos = ax.imshow(cov_real,origin='lower', cmap='bwr', norm=matplotlib.colors.LogNorm(vmin=cov_real.min(), vmax=cov_real.max()))
    ax.set_title('Real part', fontsize=12)
    ax.set_xticks(np.arange(0, len(dlys),len(dlys)/9))
    ax.set_xticklabels(dlys[0::len(dlys)/9])
    ax.set_yticks(np.arange(0, len(dlys),len(dlys)/9))
    ax.set_yticklabels(dlys[0::len(dlys)/9])
    ax.set_xlabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
    ax.set_ylabel(r"$\tau$ $[{\rm ns}]$", fontsize=12)
    # Show the colorbar
    colorbar_ax = fig.add_axes([0.8, 0.1, 0.02, 0.9])
    fig.colorbar(pos, cax=colorbar_ax)


   
   
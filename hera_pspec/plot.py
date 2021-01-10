import numpy as np
import pyuvdata
import copy
from collections import OrderedDict as odict
import astropy.units as u
import astropy.constants as c
from pyuvdata import UVData
import uvtools

from . import conversions, uvpspec, utils


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
    import matplotlib
    import matplotlib.pyplot as plt

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


def delay_waterfall(uvp, blpairs, spw, pol, component='abs-real', 
                    average_blpairs=False, fold=False, delay=True, 
                    deltasq=False, log=True, lst_in_hrs=True,
                    vmin=None, vmax=None, cmap='YlGnBu', axes=None, 
                    figsize=(14, 6), force_plot=False, times=None, 
                    title_type='blpair', colorbar=True, **kwargs):
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
        Component of complex spectra to plot, options=['abs', 'real', 'imag', 'abs-real', 'abs-imag'].
        abs-real is abs(real(data)), whereas 'real' is real(data)
        Default: 'abs-real'. 

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

    colorbar : bool, optional
        Whether to make a colorbar. Default: True

    kwargs : keyword arguments
        Additional kwargs to pass to ax.matshow()

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance if input ax is None.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # assert component
    assert component in ['real', 'abs', 'imag', 'abs-real', 'abs-imag'], "Can't parse specified component {}".format(component)
    fix_negval = component in ['real', 'imag'] and log

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
                power = np.abs(power)
            elif component == 'real':
                power = np.real(power)
            elif component == 'abs-real':
                power = np.abs(np.real(power))
            elif component == 'imag':
                power = np.imag(power)
            elif component == 'abs-imag':
                power = np.abs(np.real(power))

            # if real or imag and log is True, set negative values to near zero
            # this is done so that one can use cmap.set_under() and cmap.set_bad() separately
            if fix_negval:
                power[power < 0] = np.abs(power).min() * 1e-6 + 1e-10

            # assign to waterfall
            waterfall[key] = power

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
    y = uvp_plt.lst_avg_array[
            uvp_plt.key_to_indices(list(waterfall.keys())[0])[1] ]
    y = np.unwrap(y)
    if y[0] > np.pi:
        # if start is closer to 2pi than 0, lower axis by an octave
        y -= 2 * np.pi
    if lst_in_hrs:
        lst_units = "Hr"
        y *= 24 / (2 * np.pi)
    else:
        lst_units = "rad"

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
    keys = list(waterfall.keys())
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
            cax = ax.matshow(waterfall[key], cmap=cmap, aspect='auto', 
                             vmin=vmin, vmax=vmax, 
                             extent=[x[0], x[-1], y[-1], y[0]], **kwargs)

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
            if colorbar:
                if fix_negval:
                    cb_extend = 'min'
                else:
                    cb_extend = 'neither'
                cbar = ax.get_figure().colorbar(cax, ax=ax, extend=cb_extend)
                cbar.ax.tick_params(labelsize=14)
                if fix_negval:
                    cbar.ax.set_title("$< 0$",y=-0.05, fontsize=16)

            # configure left-column plots
            if j == 0:
                # set yticks
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


def delay_wedge(uvp, spw, pol, blpairs=None, times=None, error_weights=None, fold=False, delay=True,
                rotate=False, component='abs-real', log10=True, loglog=False,
                red_tol=1.0, center_line=False, horizon_lines=False,
                title=None, ax=None, cmap='viridis', figsize=(8, 6),
                deltasq=False, colorbar=False, cbax=None, vmin=None, vmax=None,
                edgecolor='none', flip_xax=False, flip_yax=False, lw=2, 
                set_bl_tick_major=False, set_bl_tick_minor=False, 
                xtick_size=10, xtick_rot=0, ytick_size=10, ytick_rot=0,
                **kwargs):
    """
    Plot a 2D delay spectrum (or spectra) from a UVPSpec object. Note that
    all integrations and redundant baselines are averaged (unless specifying 
    times) before plotting.

    Note: this deepcopies input uvp before averaging.
    
    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object containing delay spectra to plot.

    spw : integer
        Which spectral window to plot.

    pol : int or tuple
        Polarization-pair integer or tuple, e.g. ('pI', 'pI')

    blpairs : list of tuples, optional
        List of baseline-pair tuples to use in plotting.

    times : list, optional
        An ndarray or list of times from uvp.time_avg_array to
        select on before plotting. Default: None.

    error_weights : string, optional
         error_weights specify which kind of errors we use for weights 
         during averaging power spectra.

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
        Component of complex spectra to plot. Options=['real', 'imag', 'abs', 'abs-real', 'abs-imag']
        abs-real is abs(real(data)), whereas 'real' is real(data)
        Default: 'abs-real'.

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
        If True, use the baseline lengths as major ticks, rather than default 
        uniform grid.

    set_bl_tick_minor : bool, optional
        If True, use the baseline lengths as minor ticks, which have no labels.

    kwargs : dictionary
        Additional keyword arguments to pass to pcolormesh() call.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance if ax is None.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # type checking
    uvp = copy.deepcopy(uvp)
    assert isinstance(uvp, uvpspec.UVPSpec), "input uvp must be a UVPSpec object"
    assert isinstance(spw, (int, np.integer))
    assert isinstance(pol, (int, np.integer, tuple))
    fix_negval = component in ['real', 'imag'] and log10

    # check pspec units for little h
    little_h = 'h^-3' in uvp.norm_units

    # Create new ax if none specified
    new_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.get_figure()

    # Select out times and blpairs if provided
    if times is not None:
        uvp.select(blpairs=blpairs, times=times, inplace=True)

    # Average across redundant groups and time
    # this also ensures blpairs are ordered from short_bl --> long_bl
    blp_grps, lens, angs, tags = utils.get_blvec_reds(uvp, bl_error_tol=red_tol, 
                                                      match_bl_lens=True)
    uvp.average_spectra(blpair_groups=blp_grps, time_avg=True, error_weights=error_weights, inplace=True)

    # get blpairs and order by len and enforce bl len ordering anyways
    blpairs, blpair_seps = uvp.get_blpairs(), uvp.get_blpair_seps()
    osort = np.argsort(blpair_seps)
    blpairs, blpair_seps = [blpairs[oi] for oi in osort], blpair_seps[osort]

    # Convert to DeltaSq
    if deltasq and not delay:
        uvp.convert_to_deltasq(inplace=True)

    # Fold array
    if fold:
        uvp.fold_spectra()

    # Format ticks
    if delay:
        x_axis = uvp.get_dlys(spw) * 1e9
        y_axis = blpair_seps
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

    # get data with shape (Nblpairs, Ndlys)
    data = [uvp.get_data((spw, blp, pol)).squeeze() for blp in blpairs]

    # get component
    if component == 'real':
        data = np.real(data)
    elif component == 'abs-real':
        data = np.abs(np.real(data))
    elif component == 'imag':
        data = np.imag(data)
    elif component == 'abs-imag':
        data = np.abs(np.imag(data))
    elif component == 'abs':
        data = np.abs(data)
    else:
        raise ValueError("Did not understand component {}".format(component))

    # if real or imag and log is True, set negative values to near zero
    # this is done so that one can use cmap.set_under() and cmap.set_bad() separately
    if fix_negval:
        data[data < 0] = np.abs(data).min() * 1e-6 + 1e-10

    # take log10
    if log10:
        data = np.log10(data)

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
            ax.set_xticks([np.around(x, _get_sigfig(x)+2) for x in x_axis])
        else:
            ax.set_yticks([np.around(x, _get_sigfig(x)+2) for x in y_axis])
    if set_bl_tick_minor:
        if rotate:
            ax.set_xticks([np.around(x, _get_sigfig(x)+2) for x in x_axis], 
                          minor=True)
        else:
            ax.set_yticks([np.around(x, _get_sigfig(x)+2) for x in y_axis], 
                          minor=True)

    # Add colorbar
    if colorbar:
        if fix_negval:
            cb_extend = 'min'
        else:
            cb_extend = 'neither'
        if cbax is None:
            cbax = ax
        cbar = fig.colorbar(cax, ax=cbax, extend=cb_extend)
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
        if fix_negval:
            cbar.ax.set_title("$< 0$",y=-0.05, fontsize=16)

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
        horizons = blpair_seps / conversions.units.c * 1e9

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
    import matplotlib
    import matplotlib.pyplot as plt

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
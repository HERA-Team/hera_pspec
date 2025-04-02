from .uvpspec import UVPSpec

def apply_bias_correction(
    uvp: UVPSpec, 
    total_bias: dict | None = None, 
    data_bias: dict | None =None,
    inplace: bool = True,
):
    """
    Apply correction factors to power spectra.

    Parameters
    ----------
    uvp
        Object to which the bias correction will be applied.
    total_bias : dict
        bias correction to data and errors, e.g. abscal bias
        keys are spw integers, values are correction scalars
    data_bias : dict
        bias correction only to data, e.g. fringe-rate filtering
        keys are spw integers, values are correction scalars
    inplace : bool
        Whether to apply the bias correction in place.
    
    Returns
    -------
    UVPSpec
        The bias-corrected UVPSpec object.
    """
    if not inplace:
        uvp = uvp.copy()
        
    for spw in uvp.spw_array:
        if total_bias is not None:
            uvp.data_array[spw] *= total_bias[spw]
            if hasattr(uvp, 'cov_array_real'):
                uvp.cov_array_real[spw] *= total_bias[spw]**2
                uvp.cov_array_imag[spw] *= total_bias[spw]**2
            if hasattr(uvp, 'stats_array'):
                for stat in uvp.stats_array:
                    uvp.stats_array[stat][spw] *= total_bias[spw]
                    # TODO: this is right for P_N but not quite right for P_SN (though I'm not sure how much we care)
        if data_bias is not None:
            uvp.data_array[spw] *= data_bias[spw]
    
    return uvp
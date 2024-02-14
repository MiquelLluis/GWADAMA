"""tima.py

Time analysis toolkit.

"""


import numpy as np
import scipy as sp
import scipy.signal
from scipy.interpolate import make_interp_spline as sp_make_interp_spline



def resample(strain: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    """Resample a single strain in time domain.
    
    Resample strain's sampling rate using an interpolation in the time domain
    for upscalling to a constant rate, and then decimate it to the target rate.

    The upscaled sample rate is chosen as the minimum common multiple between
    the next integer value of the maximum sampling rate found in the original
    strain, and the target sample rate.


    PARAMETERS
    ----------
    strain: 2d-array
        NDArray containing the time samples and both polarizations, with shape
        (time, hplus, hcross).
    
    sample_rate: int
        Target sample rate.
    
        
    RETURNS
    -------
    strain_resampled: 2d-array
        NDArray with the same shape as the input strain: (time, hplus, hcross).
    
    sr_up: int
        Upscaled sample rate.
    
    factor_down: int
        Factor at which the signal is decimated after the upscalling.
    
    """
    time, hplus, hcros = strain

    # Upsample:
    sr_max = 1 / np.min(np.diff(time))
    sr_up = int((sr_max // sample_rate + 1) * sample_rate)
    # Intentionally skipping last time point to avoid extrapolation by round-off errors.
    time_up = np.arange(time[0], time[-1], 1/sr_up)
    hplus_up = sp_make_interp_spline(time, hplus, k=2)(time_up)
    hcros_up = sp_make_interp_spline(time, hcros, k=2)(time_up)

    # Downsample:
    factor_down = sr_up // sample_rate
    time_re = time_up[::factor_down]
    hplus_re = sp.signal.decimate(hplus_up, factor_down, ftype='fir')
    hcros_re = sp.signal.decimate(hcros_up, factor_down, ftype='fir')
    
    strain_resampled = np.asarray([time_re, hplus_re, hcros_re])
    
    return strain_resampled, sr_up, factor_down

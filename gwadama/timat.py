"""timat.py

Time analysis toolkit.

"""


import numpy as np
import scipy as sp
import scipy.signal
from scipy.interpolate import make_interp_spline as sp_make_interp_spline



def resample(strain: np.ndarray,
             time: np.ndarray | int,
             sample_rate: int,
             full_output=True) -> tuple[np.ndarray, int, int]:
    """Resample a single strain in time domain.
    
    Resample strain's sampling rate using an interpolation in the time domain
    for upscalling to a constant rate, and then decimate it to the target rate.

    The upscaled sample rate is chosen as the minimum common multiple between
    the next integer value of the maximum sampling rate found in the original
    strain, and the target sample rate.


    PARAMETERS
    ----------
    strain: 1d-array
        Only one strain.
    
    time: 1d-array | int
        Time points. If an Int is given, it is interpreted as the former
        sampling rate, and assumed to be constant.
    
    sample_rate: int
        Target sample rate.
    
    full_output: bool, optional
        If True, also returns the upscaled sampling rate and the factor down.
    
        
    RETURNS
    -------
    strain: 1d-array
        Strain at the new sampling rate.
    
    sr_up: int, optional
        Upscaled sample rate.
    
    factor_down: int, optional
        Factor at which the signal is decimated after the upscalling.
    
    """
    if isinstance(time, np.ndarray):
        sr_max = 1 / np.min(np.diff(time))
    elif isinstance(time, int):
        sr_max = time
    else:
        raise TypeError("'time' type not recognized")

    # Upsample:
    #
    sr_up = int((sr_max // sample_rate + 1) * sample_rate)
    # Intentionally skipping last time point to avoid extrapolation by round-off errors.
    time = np.arange(time[0], time[-1], 1/sr_up)
    strain = sp_make_interp_spline(time, strain, k=2)(time)

    # Downsample:
    #
    factor_down = sr_up // sample_rate
    time = time[::factor_down]
    strain = sp.signal.decimate(strain, factor_down, ftype='fir')
    
    return strain, sr_up, factor_down if full_output else strain

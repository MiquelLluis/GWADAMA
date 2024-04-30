"""tat.py

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
        If True, also returns the new time points, the upscaled sampling rate,
        and the factor down.
    
        
    RETURNS
    -------
    strain: 1d-array
        Strain at the new sampling rate.
    
    time: 1d-array, optional
        New time points.
    
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
    time_up = np.arange(time[0], time[-1], 1/sr_up)
    strain = sp_make_interp_spline(time, strain, k=2)(time_up)

    # Downsample:
    #
    factor_down = sr_up // sample_rate
    time = time[::factor_down]
    strain = sp.signal.decimate(strain, factor_down, ftype='fir')
    
    return strain, time, sr_up, factor_down if full_output else strain


def gen_time_array(t0, t1, sr):
    """Generate a time array with constant sampling rate.
    
    Extension of numpy.arange which takes care of the case when an extra sample
    is produced due to round-off errors. When this happens, the extra sample is
    cut off.

    Parameters
    ----------
    t0, t1: float
        Initial and final times of the array: [t0, t1).
    
    sr: int
        Sample rate.
    
    length: int
        Length of the final time array in samples.
        If due to round-off errors the length of the array is longer, it will
        be adjusted.
    
    Returns
    -------
    times: NDArray
        Time array.
    
    """
    times = np.arange(t0, t1, 1/sr)
    if times[-1] >= t1:
        times = times[:-1]
    
    return times


def pad_time_array(times: np.ndarray, pad: int) -> np.ndarray:
    """Extend a time array on both sides by 'pad' number of samples.
    
    NOTES
    -----
    - Computes again the entire time array.
    - Due to round-off errors some intermediate time values might be slightly
      different.
    
    """
    l = 2*pad + len(times)
    dt = times[1] - times[0]
    t0 = times[0] - pad*dt
    t1 = t0 + (l-1)*dt

    return np.linspace(t0, t1, l)


def shrink_time_array(times: np.ndarray, unpad: int) -> np.ndarray:
    """Shrink a time array on both sides by 'unpad' number of samples.
    
    NOTES
    -----
    - Computes again the entire time array.
    - Due to round-off errors some intermediate time values might be slightly
      different.
    
    """
    l = len(times) - 2*unpad
    dt = times[1] - times[0]
    t0 = times[0] + unpad*dt
    t1 = t0 + (l-1)*dt

    return np.linspace(t0, t1, l)

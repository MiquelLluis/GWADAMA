"""tat.py

Time analysis toolkit.

"""
import numpy as np
from numpy.typing import ArrayLike
import scipy as sp
from scipy.interpolate import make_interp_spline as sp_make_interp_spline



def resample(strain: np.ndarray,
             times: np.ndarray,
             sample_rate: int,
             full_output=True) -> tuple[np.ndarray, np.array, int, int]:
    """Resample a single strain in time domain.
    
    Resample strain's sampling rate using an interpolation in the time domain
    for upscalling to a constant rate, and then decimate it to the target rate.

    The upscaled sample rate is chosen as the minimum common multiple between
    the next integer value of the maximum sampling rate found in the original
    strain, and the target sample rate.


    PARAMETERS
    ----------
    strain: 1d-array
        Input strain signal.
    
    times : 1d-array
        Original time points. Must be a NumPy array.
    
    sample_rate: int
        Target sample rate (Hz). Must be a possitive integer.
    
    full_output: bool, optional
        If True, also returns the new time array, original sample rate,
        and decimation factor.
    
        
    RETURNS
    -------
    strain_resampled : 1d-array
        Resampled strain.

    times_resampled : 1d-array, optional
        Time array at the new sampling rate.

    sr_in : int, optional
        Original (inferred) sample rate, after interpolation if performed.

    up, down : int, optional
        Up and down factors of the resampling.
    
    """
    if not isinstance(times, np.ndarray):
        raise TypeError("'times' must be a NumPy array.")
    if sample_rate <= 0:
        raise ValueError("Target 'sample_rate' must be positive.")

    if not is_arithmetic_progression(times):
        # Interpolate to a uniform time grid at the highest reasonable rate
        sr_interp = int(np.ceil(1 / np.min(np.diff(times))))
        new_length = int((times[-1] - times[0]) * sr_interp) + 1
        times_uniform = np.linspace(times[0], times[-1], new_length, endpoint=True)
        strain = sp_make_interp_spline(times, strain, k=2)(times_uniform)
        times = times_uniform
    else:
        sr_interp = int(round(1 / (times[1] - times[0])))

    # Compute up/down factors
    g = np.gcd(sr_interp, sample_rate)
    up = sample_rate // g
    down = sr_interp // g
    
    strain_resampled = sp.signal.resample_poly(strain, up, down)
    
    if full_output:
        times_resampled = time_array_like(strain_resampled, sr=sample_rate, t0=times[0])
        return strain_resampled, times_resampled, sr_interp, up, down
    return strain_resampled


def gen_time_array(t0, t1, sample_rate, length=None):
    """Generate a time array for a given time range and sampling rate.

    Generate a time array for a given time range and sampling rate with an
    optional consistency check.

    Floating-point precision can cause `(t1 - t0) * sr` to yield an unexpected 
    number of samples, leading to off-by-one errors in :func:`numpy.linspace`.
    If `length` is provided, the function verifies that it matches the expected
    length using `t1` as reference. A mismatch raises will raise `ValueError`,
    ensuring consistency between the expected and actual number of samples.

    If `length` is not provided, the function simply calls
    :func:`numpy.linspace` without performing the safety check.

    Parameters
    ----------
    t0 : float
        Start time (inclusive).
    
    t1 : float
        Final time (exclusive).
    
    sample_rate : float
        Sampling rate.
    
    length : int, optional
        If provided, must match `int((t1 - t0) * sample_rate)` exactly.
        Acts as a safeguard against miscalculations due to floating-point
        precision.

    Returns
    -------
    numpy.ndarray
        A 1D array of evenly spaced time values from `t0` (inclusive) to `t1`
        (exclusive).

    Raises
    ------
    ValueError
        If `length` is provided and does not match the expected number of
        samples.
    
    """
    expected_length = int((t1 - t0) * sample_rate)
    if length is not None and length != expected_length:
        raise ValueError(
            f"Inconsistent input: Expected {expected_length} samples, got {length}."
        )
    
    return np.linspace(t0, t1, expected_length, endpoint=False)


def time_array_like(array, sr=4096, t0=0.0):
    """Generate a time array matching the length of the input array.

    Computes a time array starting from `t0` with evenly spaced values based 
    on the sampling rate `sr`, matching the length of the input 1D array.

    Parameters
    ----------
    array : array_like
        Input 1D array whose length determines the number of time samples.
    
    sr : float, optional
        Sampling rate in Hz. Default is 4096.
    
    t0 : float, optional
        Start time in seconds. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        A 1D array of evenly spaced time values starting at `t0` with spacing
        `1/sr` and the same length as the input array.
    
    """
    n = len(array)
    return np.linspace(t0, t0+n/sr, n, endpoint=False)



def pad_time_array(times: np.ndarray, pad: int | ArrayLike) -> np.ndarray:
    """Extend a time array by 'pad' number of samples.

    Parameters
    ----------
    times: NDArray
        Time array.
    
    pad: int | ArrayLike
        If int, number of samples to add on both sides.
        If ArrayLike, number of samples to add on each side.
    
    Returns
    -------
    NDArray
        Padded time array.
    
    NOTES
    -----
    - Computes again the entire time array.
    - Due to round-off errors some intermediate time values might be slightly
      different.
    
    """
    if isinstance(pad, int):
        pad0, pad1 = pad, pad
    else:
        pad0, pad1 = pad

    length = len(times) + pad0 + pad1
    dt = times[1] - times[0]

    t0 = times[0] - pad0*dt
    t1 = t0 + (length-1)*dt

    return np.linspace(t0, t1, length)


def find_time_origin(times: np.ndarray) -> int:
    """Find the index position of the origin of a time array.
    
    It is just a shortcut for `np.argmin(np.abs(times))`.
    
    Parameters
    ----------
    times : NDArray
        Time array.
    
    Returns
    -------
    _ : int
        Index position of the time origin (0).
    
    """
    return np.argmin(np.abs(times))


def find_merger(h: np.ndarray) -> int:
    """Estimate the index position of the merger in the given strain.

    This function provides a rough estimate of the merger index position by
    locating the maximum of the absolute amplitude of the gravitational wave
    signal in the time domain. It assumes that the merger roughly corresponds
    to this peak, which holds for certain clean or high-SNR simulated CBC
    gravitational waves.

    :warning:
    This function may be replaced in the near future by a more formal estimator
    with a better model, such as a Gaussian fit for binary mergers.
    
    :caution:
    This is a very ad-hoc method and may not be accurate for all datasets,
    especially depending on the sensitivity of the detector. This method
    assumes that the peak amplitude in the time domain corresponds closely to
    the merger, which may not hold for lower-SNR signals or noisy data.

    Parameters
    ----------
    h : np.ndarray
        The gravitational wave strain data.

    Returns
    -------
    int
        The index of the estimated merger position in the strain data.
    
    """
    return np.argmax(np.abs(h))


def planck(N, nleft=0, nright=0):
    """Return a Planck taper window.

    Parameters
    ----------
    N : `int`
        Number of samples in the output window

    nleft : `int`, optional
        Number of samples to taper on the left, should be less than `N/2`

    nright : `int`, optional
        Number of samples to taper on the right, should be less than `N/2`

    Returns
    -------
    w : `ndarray`
        The window, with the maximum value normalized to 1 and at least one
        end tapered smoothly to 0.

    References
    ----------
    Based on :func:`gwpy.signal.window.planck`.
    
    """
    w = np.ones(N, dtype=float)
    if nleft:
        k = np.arange(1, nleft)
        z = nleft * (1/k + 1/(k - nleft))
        w[1:nleft] *= sp.special.expit(-z)  # sigmoid
        w[0] = 0
    if nright:
        k = np.arange(1, nright)
        z = -nright * (1/(k - nright) + 1/k)
        w[-nright:-1] *= sp.special.expit(-z)
        w[-1] = 0
    
    return w


def truncate_transfer(transfer, ncorner=None):
    """Smoothly zero the edges of a frequency domain transfer function

    Parameters
    ----------
    transfer : `numpy.ndarray`
        transfer function to start from, must have at least ten samples

    ncorner : `int`, optional
        number of extra samples to zero off at low frequency, default: `None`

    Returns
    -------
    out : `numpy.ndarray`
        the smoothly truncated transfer function

    References
    ----------
    Based on :func:`gwpy.signal.filter_design.truncate_transfer`.

    """
    nsamp = transfer.size
    ncorner = ncorner or 0
    out = transfer.copy()
    out[:ncorner] = 0
    out[ncorner:] *= planck(nsamp-ncorner, nleft=5, nright=5)
    
    return out


def truncate_impulse(impulse, ntaps, window='hann'):
    """Smoothly truncate a time domain impulse response

    Parameters
    ----------
    impulse : `numpy.ndarray`
        the impulse response to start from

    ntaps : `int`
        number of taps in the final filter

    window : `str`, `numpy.ndarray`, optional
        window function to truncate with, default: ``'hann'``
        see :func:`scipy.signal.get_window` for details on acceptable formats

    Returns
    -------
    out : `numpy.ndarray`
        the smoothly truncated impulse response

    References
    ----------
    Based on :func:`gwpy.signal.filter_design.truncate_impulse`.

    """
    out = impulse.copy()
    trunc_start = ntaps // 2
    trunc_stop = out.size - trunc_start
    window = sp.signal.get_window(window, ntaps)
    out[:trunc_start] *= window[trunc_start:]
    out[trunc_stop:] *= window[:trunc_start]
    out[trunc_start:trunc_stop] = 0
    
    return out


def fir_from_transfer(transfer, ntaps, window='hann', ncorner=None):
    """Design a Type II FIR filter given an arbitrary transfer function

    Parameters
    ----------
    transfer : `numpy.ndarray`
        transfer function to start from, must have at least ten samples

    ntaps : `int`
        number of taps in the final filter, must be an even number

    window : `str`, `numpy.ndarray`, optional
        window function to truncate with, default: ``'hann'``
        see :func:`scipy.signal.get_window` for details on acceptable formats

    ncorner : `int`, optional
        number of extra samples to zero off at low frequency, default: `None`

    Returns
    -------
    out : `numpy.ndarray`
        A time domain FIR filter of length `ntaps`

    References
    ----------
    Based on :func:`gwpy.signal.filter_design.fir_from_transfer`.

    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = np.fft.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = np.roll(impulse, shift=ntaps//2-1)[:ntaps]
    
    return out


def convolve(strain, fir, window='hann'):
    """Convolve a time series.

    Convolve a time series with a FIR filter using the overlap-save method.

    Parameters
    ----------
    strain : numpy.ndarray
        The input time series strain.
    
    fir : numpy.ndarray
        The FIR filter coefficients.
    
    window : str | tuple, optional
        The window function to apply to the boundaries (default: 'hann').

    Returns
    -------
    numpy.ndarray
        The convolved time series, same length as input data.

    Notes
    -----
    This function, based on the implementation of GWpy[1]_, differs in its
    implementation from the `oaconvolve` implementation of SciPy[2]_, which
    uses the overlap-add method:

    - **Algorithm**: Overlap-save discards edge artifacts after convolution,
      whereas overlap-add sums overlapping output segments.
    - **Boundary Windowing**: Explicitly applies a window (e.g., Hann) to the
      first/last ``N/2`` samples (where ``N`` is the filter length) to suppress
      spectral leakage at boundaries. SciPy implementations do not modify input
      boundaries.
    - **Edge Corruption**: Intentionally tolerates edge corruption in a segment
      of length ``N/2`` at both ends, while SciPy's output validity depends on
      the selected mode (`full`/`valid`/`same`).
    - **Use Case Focus**: Optimized for 1D time-series processing with
      boundary-aware windowing. SciPy's `oaconvolve` is more general-purpose,
      focused in computational efficiency for large N-dimensional arrays
      and arbitrary convolution modes.

    References
    ----------
    .. [1] Based on :meth:`gwpy.timeseries.TimeSeries.convolve`.
    .. [2] SciPy's more general implementation :func:`scipy.signal.oaconvolve`.
    
    """
    pad = int(np.ceil(len(fir) / 2))
    nfft = min(8 * len(fir), len(strain))
    
    # Apply window to the boundaries of the input data
    window_arr = sp.signal.get_window(window, len(fir))
    padded_data = strain.copy()
    padded_data[:pad] *= window_arr[:pad]
    padded_data[-pad:] *= window_arr[-pad:]
    
    if nfft >= len(strain) // 2:
        # Perform a single convolution if FFT length is sufficiently large
        conv = sp.signal.fftconvolve(padded_data, fir, mode='same')
    else:
        nstep = nfft - 2 * pad
        conv = np.zeros_like(strain)
        
        # Process the first chunk
        first_chunk = padded_data[:nfft]
        conv[:nfft-pad] = sp.signal.fftconvolve(first_chunk, fir, mode='same')[:nfft-pad]
        
        # Process middle chunks
        for k in range(nfft-pad, len(strain)-nfft+pad, nstep):
            yk = sp.signal.fftconvolve(padded_data[k-pad:k+nstep+pad], fir, mode='same')
            conv[k:k+yk.size-2*pad] = yk[pad:-pad]
        
        # Process the last chunk
        conv[-nfft+pad:] = sp.signal.fftconvolve(
            padded_data[-nfft:],
            fir,
            mode='same'
        )[-nfft+pad:]
    
    return conv


def whiten(strain: np.ndarray,
           *,
           asd: np.ndarray,
           sample_rate: int,
           flength: int,
           window='hann',
           highpass: float = None,
           normed: bool = True) -> np.ndarray:
    """Whiten a single strain signal using a FIR filter.

    Whiten a strain using the input amplitude spectral density 'asd' to
    design the FIR filter.

    This is a standalone implementation of GWpy's whiten method.[1]_

    Parameters
    ----------
    strain : NDArray
        Strain data points in time domain.

    asd : 2d-array
        Amplitude spectral density assumed for the 'set_strain'.
        Its components are:
        - asd[0] = frequency points
        - asd[1] = ASD points
        NOTE: It must have a linear and constant sampling frequency!

    sample_rate : int
        The sampling rate of the strain data.

    flength : int
        Length (in samples) of the time-domain FIR whitening filter.
    
    window : str, np.ndarray, optional
        window function to apply to timeseries prior to FFT,
        default: 'hann'
        see :func:`scipy.signal.get_window` for details on acceptable
        formats.

    highpass : float, optional
        Highpass corner frequency (in Hz) of the FIR whitening filter.

    normed : bool
        If True, normalizes the strains to their maximum absolute amplitude.

    Returns
    -------
    strain_w : NDArray
        Whitened strain (in time domain).

    Notes
    -----
    Due to filter settle-in, a segment of length `0.5*fduration` will be
    corrupted at the beginning and end of the output.
    
    References
    ----------
    .. [1] Based on :meth:`gwpy.timeseries.TimeSeries.whiten`.

    """
    if asd.ndim != 2:
        raise ValueError("'asd' must have 2 dimensions")
    if not is_arithmetic_progression(asd[0]):
        raise ValueError("frequency points in 'asd[0]' must be ascending with constant increment")
    if not isinstance(flength, int):
        raise TypeError("'flength' must be an integer")

    # Constant detrending
    strain_detrended = strain - np.mean(strain)

    asd_freq, asd_vals = asd
    dt = 1 / sample_rate

    freq_target = np.fft.rfftfreq(len(strain), dt)

    # Assumes ASD is defined or extrapolated over full frequency range.
    # No check needed if extrapolated with zeros and signal is bandpass-filtered.
    #--------------------------------------------------------------------------
    # if asd_freq[-1] < freq_target[-1]:
    #     raise ValueError("ASD frequency range is insufficient for the strain data.")
    #--------------------------------------------------------------------------


    # Linear interpolation of ASD if needed, assuming ASD_freq is already
    # covering the strain frequency range.
    asd_interp_function = sp.interpolate.interp1d(
        asd_freq, asd_vals, kind='linear',
        bounds_error=False, fill_value=0
    )
    asd_interpolated = asd_interp_function(freq_target)

    # Compute transfer function
    with np.errstate(divide='ignore'):
        transfer_function = np.reciprocal(asd_interpolated, where=asd_interpolated!=0)
    transfer_function[np.isinf(transfer_function)] = 0  # Handle division by zero

    # Handle highpass
    duration = len(strain) / sample_rate
    delta_f = 1 / duration
    if highpass:
        ncorner = int(highpass / delta_f)
    else:
        ncorner = None

    # Create the causal FIR filter
    fir_filter = fir_from_transfer(transfer_function, ntaps=flength, ncorner=ncorner)

    # Convolve with filter
    strain_whitened = convolve(strain_detrended, fir_filter, window=window)
    strain_whitened *= np.sqrt(2 * dt)  # scaling factor

    # Normalize if needed
    if normed:
        max_abs = np.max(np.abs(strain_whitened))
        if max_abs != 0:
            strain_whitened /= max_abs

    return strain_whitened


def is_arithmetic_progression(arr: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """Check if the array is an arithmetic progression.
    
    Check if the array is a progression with a constant increment, allowing for
    numerical tolerances.

    Parameters
    ----------
    arr : np.ndarray
        Input array to check.
    
    rtol : float, optional
        Relative tolerance for comparing floating-point values.
        Default is 1e-5.
    
    atol : float, optional
        Absolute tolerance for comparing floating-point values.
        Default is 1e-8.

    Returns
    -------
    bool
        True if the array is an arithmetic progression (within a specified
        tolerance at their increments), False otherwise.
    
    """
    if len(arr) <= 1:
        return True
    
    step = arr[1] - arr[0]
    expected_last = arr[0] + step*(len(arr) - 1)
    if not np.isclose(arr[-1], expected_last, rtol=rtol, atol=atol):
        return False
    
    return np.allclose(np.diff(arr), step, rtol=rtol, atol=atol)

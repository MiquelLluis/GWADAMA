"""tat.py

Time analysis toolkit.

"""
from collections.abc import Sequence
from fractions import Fraction
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy as sp
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly



def _build_uniform_grid(t0: float, t1: float, fs: float) -> np.ndarray:
    """Closed interval grid [t0, t1] with step 1/fs."""
    n = int(np.floor((t1 - t0) * fs)) + 1
    return np.linspace(t0, t1, n, endpoint=True)


def resample(
    strain: NDArray,
    times: NDArray,
    fs: int,
    *,
    full_output: bool = True,
    uniform_rtol: float = 1e-6,
    uniform_atol: float = 1e-12,
    percentile: float = 90.0,       # robust cadence from the 90th percentile of instantaneous fs
    f_min_factor: float = 1.25,     # ensure f_u >= f_min_factor * fs to leave FIR headroom
    f_cap_factor: float = 8.0,      # cap f_u <= f_cap_factor * fs to avoid blow-ups
    frac_den_limit: int = 8192      # cap denominator when reducing rational ratios
):
    """
    Resample (possibly irregularly-sampled) 'strain' onto a uniform grid at
    target sampling rate 'fs' (Hz).

    Logic:

    1) Validate input.
    2) If times are approximately uniform:
         - If fs_in == fs (within 0.5 Hz): return a copy aligned to the
           original start.
         - Else: resample_poly directly from fs_in -> fs (anti-aliasing
           included).
       Else (times irregular):
         a) If max instantaneous rate < fs: issue a warning and directly
            interpolate to fs via PCHIP (upsampling).
         b) Otherwise:
              - Pick robust f_u from the 'percentile' of instantaneous rates.
              - Enforce fs <= f_u <= min(max_inst_rate, f_cap_factor*fs), and
                f_u >= f_min_factor*fs.
              - PCHIP-uniformise at f_u, then resample_poly f_u -> fs.
    
    Parameters
    ----------
    strain : numpy.ndarray
        One-dimensional input strain samples, shape (N,).

    times : numpy.ndarray
        One-dimensional, strictly increasing time stamps (seconds), shape (N,).
        No duplicates; must be sorted ascending.

    fs : int
        Target sampling frequency in Hz. Must be a positive integer.

    full_output : bool, default=True
        If True, also return the output time grid, an input-rate estimate,
        and the (up, down) integers used in the final polyphase step.

    uniform_rtol : float, default=1e-6
        Relative tolerance for the uniform-spacing check.

    uniform_atol : float, default=1e-12
        Absolute tolerance for the uniform-spacing check.

    percentile : float, default=90.0
        Percentile (in [0, 100]) of instantaneous sampling rates `1/np.diff(times)`
        used to choose the robust uniformisation rate `f_u` when input is irregular.

    f_min_factor : float, default=1.25
        Lower bound multiplier for `f_u` relative to `fs` (i.e. `f_u ≥ f_min_factor*fs`)
        to leave margin for the FIR anti-alias filter.

    f_cap_factor : float, default=8.0
        Upper bound multiplier for `f_u` relative to `fs` (i.e. `f_u ≤ f_cap_factor*fs`)
        to avoid excessively large intermediate grids. Must satisfy
        `f_cap_factor ≥ f_min_factor`. `f_u` is also capped by the maximum
        instantaneous rate.

    frac_den_limit : int, default=2048
        Maximum allowed denominator when approximating the rate ratio with
        `fractions.Fraction(...).limit_denominator`. Larger values give a closer
        ratio but longer polyphase FIR filters (slower); smaller values shorten
        the filter at the cost of a tiny ratio error.

    Returns
    -------
    y : 1d-array
        Resampled strain at 'fs'.
    
    t : 1d-array (if full_output)
        Time grid at 'fs', starting at times[0].
    
    fs_in : int (if full_output)
        Estimated original sampling frequency (rounded to nearest integer). For
        irregular inputs, this is the robust uniformisation rate if used;
        otherwise a robust estimate from the mean spacing.
    
    up, down : int, int (if full_output)
        Up/down integers used in the final polyphase step. (0, 0) if N/A.
    """
    # --- Basic validations
    if not isinstance(times, np.ndarray):
        raise TypeError("'times' must be a NumPy array.")
    if not isinstance(strain, np.ndarray):
        raise TypeError("'strain' must be a NumPy array.")
    if times.ndim != 1 or strain.ndim != 1 or len(times) != len(strain):
        raise ValueError("'times' and 'strain' must be 1D arrays of equal length.")
    if fs <= 0:
        raise ValueError("Target 'fs' must be a positive integer.")
    if len(times) < 3:
        # Need at least 3 points to sensibly infer cadence / interpolate
        raise ValueError("Need at least 3 samples to resample.")

    # Ensure strictly increasing times (and no duplicates)
    if not np.all(np.diff(times) > 0):
        raise ValueError(
            "'times' must be strictly increasing (no duplicates, sorted "
            "ascending)."
        )

    # --- Uniformity check
    is_uni = is_arithmetic_progression(times, rtol=uniform_rtol, atol=uniform_atol)
    t0, t1 = float(times[0]), float(times[-1])

    # Helper: rational up/down from two (positive) rates
    def _ratio_ud(f_out: float, f_in: float) -> tuple[int, int]:
        r = Fraction(f_out/f_in).limit_denominator(frac_den_limit)
        return r.numerator, r.denominator

    # ---- Case A: approximately uniform input ----
    if is_uni:
        fs_in = float(1 / np.mean(np.diff(times)))  # cast from np.floating to float
        # Sanity check: if input rate < target, warn (we're effectively
        # upsampling)
        if fs_in + 0.5 < fs:
            warnings.warn(
                f"Input sampling rate ({fs_in:.3f} Hz) is below target ({fs} Hz). "
                "Proceeding with upsampling; no anti-alias filtering is needed."
            )
        # If already at the target rate (within ~0.5 Hz), just align to exact
        # integer fs
        if abs(fs_in - fs) < 0.5:
            t = _build_uniform_grid(t0, t1, fs=float(fs))
            y = PchipInterpolator(times, strain, extrapolate=False)(t)
            if full_output:
                return y, t, int(round(fs_in)), 0, 0
            return y

        # Proper rate conversion with polyphase FIR
        up, down = _ratio_ud(fs, fs_in)
        # Align a uniform grid at the original cadence
        t_u = _build_uniform_grid(t0, t1, fs=fs_in)
        # Interpolate once onto that grid
        y_u = PchipInterpolator(times, strain, extrapolate=False)(t_u)
        y = resample_poly(y_u, up, down)
        # Build output time grid (anchored at t0, step 1/fs)
        t = t0 + np.arange(len(y)) / fs
        if full_output:
            return y, t, int(round(fs_in)), up, down
        return y

    # ---- Case B: irregular input ----
    dt = np.diff(times)
    fs_inst = 1.0 / dt
    fs_inst_max = float(np.max(fs_inst))  # maximum instantaneous sampling rate

    # If even the maximum instantaneous rate is below the target, we can only
    # upsample → direct PCHIP to fs
    if fs_inst_max < fs:
        warnings.warn(
            f"Maximum instantaneous sampling rate ({fs_inst_max:.3f} Hz) is "
            f"below target ({fs} Hz). Directly interpolating to the target "
            "grid; anti-alias filtering is unnecessary."
        )
        t = _build_uniform_grid(t0, t1, fs=float(fs))
        y = PchipInterpolator(times, strain, extrapolate=False)(t)
        if full_output:
            # Use a robust estimate of the original cadence for reporting
            fs_in_report = 1 / np.mean(dt)
            return y, t, int(round(fs_in_report)), 0, 0
        return y

    # Otherwise, choose a robust uniformisation rate f_u, e.g. 90th percentile
    # of instantaneous fs. Start from this robust value.
    f_u = float(np.percentile(fs_inst, percentile))

    # Enforce lower bound: keep some headroom for the FIR low-pass (avoid
    # too-low uniformisation)
    f_u = max(f_u, f_min_factor * float(fs))

    # Enforce practical cap relative to the target to avoid huge temporary arrays
    f_u = min(f_u, f_cap_factor * float(fs))

    # Never exceed the actual max instantaneous rate
    f_u = min(f_u, fs_inst_max)

    # Edge case: numerical tie with fs (avoid zero-length filters/degenerate
    # ratios). If the rounding pushed us down, bump slightly
    if f_u < fs:
        f_u = float(fs)

    t_u = _build_uniform_grid(t0, t1, fs=f_u)
    y_u = PchipInterpolator(times, strain, extrapolate=False)(t_u)

    up, down = _ratio_ud(float(fs), f_u)
    y = resample_poly(y_u, up, down)

    # Output time grid at fs
    t = t0 + np.arange(len(y)) / fs

    if full_output:
        return y, t, int(round(f_u)), up, down
    return y


def gen_time_array(
    t0: float,
    t1: float,
    *,
    fs: int,
    length: int|None = None
) -> NDArray[np.floating]:
    """Generate a time array for a given time range and sampling frequency.

    Generate a time array for a given time range and sampling frequency with an
    optional consistency check.

    Floating-point precision can cause `(t1 - t0) * fs` to yield an unexpected 
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
    
    fs : int
        sampling frequency.
    
    length : int, optional
        If provided, must match `int((t1 - t0) * fs)` exactly.
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
    expected_length = int((t1 - t0) * fs)
    if length is not None and length != expected_length:
        raise ValueError(
            f"Inconsistent input: Expected {expected_length} samples, got {length}."
        )
    
    return np.linspace(t0, t1, expected_length, endpoint=False)


def time_array_like(array, fs=4096, t0=0.0):
    """Generate a time array matching the length of the input array.

    Computes a time array starting from `t0` with evenly spaced values based 
    on the sampling frequency `fs`, matching the length of the input 1D array.

    Parameters
    ----------
    array : array_like
        Input 1D array whose length determines the number of time samples.
    
    fs : float, optional
        sampling frequency in Hz. Default is 4096.
    
    t0 : float, optional
        Start time in seconds. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        A 1D array of evenly spaced time values starting at `t0` with spacing
        `1/fs` and the same length as the input array.
    
    """
    n = len(array)
    return np.linspace(t0, t0+n/fs, n, endpoint=False)



def pad_time_array(times: NDArray, pad: int|Sequence[int]) -> np.ndarray:
    """Extend a uniformly sampled time array by 'pad' number of samples.

    Parameters
    ----------
    times: numpy.ndarray
        1D array of time samples. Must be uniformly sampled.
    
    pad: int or Sequence of two ints
        Number of samples to add.
        * If int, the same number of samples is added on both sides.
        * If Sequence of length 2, interpreted as (pad_before, pad_after).
    
    numpy.ndarray
        New time array with the specified padding, using the same time step
        as the input.

    Raises
    ------
    ValueError
        If `times` is not uniformly sampled.
    
    Notes
    -----
    - The function recomputes the entire time array using
      :func:`numpy.linspace`.
    - Due to floating-point round-off, intermediate values may differ
      slightly from the input.

    Examples
    --------
    >>> t = np.array([0.0, 0.1, 0.2])
    >>> pad_time_array(t, 1)
    array([-0.1, 0.0, 0.1, 0.2, 0.3])
    
    """
    if not is_arithmetic_progression(times):
        raise ValueError("Input time array must be uniformly sampled.")

    if isinstance(pad, int):
        pad0, pad1 = pad, pad
    else:
        pad0, pad1 = pad

    length = len(times) + pad0 + pad1
    dt = times[1] - times[0]
    t0 = times[0] - pad0*dt
    t1 = t0 + (length-1)*dt

    return np.linspace(t0, t1, length)


def find_time_origin(times: NDArray) -> int:
    """Return the index of the element closest to zero.

    Finds the position in the time array whose value is nearest to 0.
    
    Parameters
    ----------
    times : NDArray
        Time array.
    
    Returns
    -------
    int
        Index position of the time origin (0).
    
    """
    return int(np.argmin(np.abs(times)))


def find_merger(h: NDArray) -> int:
    """Estimate the index of the merger in a gravitational-wave strain.

    This function provides a rough estimate of the merger time index by finding
    the position of the maximum absolute amplitude in the time-domain strain
    data. It assumes that the merger approximately coincides with this peak,
    which is often valid for certain clean or high‑SNR simulated CBC
    (compact binary coalescence) signals.

    Parameters
    ----------
    h : numpy.ndarray
        1D array containing the gravitational-wave strain.
        If the strain is split into polarisations, combine them in a complex
        array such as: ``h = s_plus - 1j * s_cross``

    Returns
    -------
    int
        Index of the estimated merger position in `h`.

    Warnings
    --------
    This method is **ad hoc** and may give misleading results on some data,
    particularly low-SNR or real detector data where noise fluctuations
    dominate the peak amplitude.

    Notes
    -----
    * Assumes that the peak amplitude in the time domain corresponds to
      the merger, which is a simplification.
    * This function is a **temporary placeholder** and will be replaced in
      future versions by a more formal estimator (for example, a Gaussian
      fit around the expected merger region).

    Examples
    --------
    >>> h = np.array([0.0, 0.2, -0.1, 0.5, 0.3])
    >>> find_merger(h)
    3
    """
    return int(np.argmax(np.abs(h)))


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

    .. [1] McKechan, D.J.A., Robinson, C., and Sathyaprakash, B.S. (April
           2010). "A tapering window for time-domain templates and simulated
           signals in the detection of gravitational waves from coalescing
           compact binaries". Classical and Quantum Gravity 27 (8).
           :doi:`10.1088/0264-9381/27/8/084020`
    
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


def truncate_impulse(impulse, ntaps, window: str|tuple = 'hann'):
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
    # Set up window:
    if isinstance(window, np.ndarray):
        if window.ndim != 1:
            raise ValueError("`window` must be a 1‑D array")
        if window.size != ntaps:
            raise ValueError("`window` length must equal `ntaps`")
    else:
        window = sp.signal.get_window(window, ntaps)

    out = impulse.copy()
    trunc_start = ntaps // 2
    trunc_stop = out.size - trunc_start
    out[:trunc_start] *= window[trunc_start:]
    out[trunc_stop:] *= window[:trunc_start]
    out[trunc_start:trunc_stop] = 0
    
    return out


def fir_from_transfer(transfer, ntaps, window: str|tuple = 'hann', ncorner=None):
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


def convolve(strain, fir, window: str|tuple = 'hann'):
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


def whiten(strain: NDArray,
           *,
           asd: NDArray,
           fs: int,
           flength: int,
           window: str|tuple = 'hann',
           highpass: float|None = None,
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

    fs : int
        The sampling frequency of the strain data.

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
    if asd.ndim != 2 or asd.shape[0] != 2:
        raise ValueError("'asd' must have 2 dimensions")
    if not is_arithmetic_progression(asd[0]):
        raise ValueError("frequency points in 'asd[0]' must be ascending with constant increment")
    if not isinstance(flength, int):
        raise TypeError("'flength' must be an integer")

    # Constant detrending
    strain_detrended = strain - np.mean(strain)

    asd_freq, asd_vals = asd
    dt = 1 / fs

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
    duration = len(strain) / fs
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


def is_arithmetic_progression(arr: NDArray, rtol=1e-5, atol=1e-8) -> bool:
    """Check if the array is an arithmetic progression.
    
    Check if the array is a progression with a constant increment, allowing for
    numerical tolerances.

    Parameters
    ----------
    arr : NDArray
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
    if len(arr) < 2:
        raise ValueError
    
    step = arr[1] - arr[0]
    expected_last = arr[0] + step*(len(arr) - 1)
    if not np.isclose(arr[-1], expected_last, rtol=rtol, atol=atol):
        return False
    
    return np.allclose(np.diff(arr), step, rtol=rtol, atol=atol)

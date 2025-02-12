"""fat.py

Frequency analysis toolkit.

"""
import warnings

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import numpy as np
import scipy as sp



def whiten(strain: np.ndarray,
           *,
           asd: np.ndarray,
           sample_rate: int,
           flength: int,
           highpass: float = None,
           pad: int = 0,
           unpad: int = 0,
           normed: bool = True,
           **kwargs) -> np.ndarray:
    """Whiten a single strain signal.

    Whiten a strain using the input amplitude spectral density 'asd',
    and shrinking signals afterwarwds to 'l_window' to account for the edge
    effects introduced by the windowing.

    Parameters
    ----------
    strain : NDArray
        Strain data points in time domain.
    
    asd : 2d-array
        Amplitude spectral density assumed for the 'set_strain'.
        Its components are:
        - asd[0] = frequency points
        - asd[1] = ASD points
        NOTE: It must has a linear and constant sampling rate!
    
    sample_rate : int
        The thingy that makes things do correctly their thing.
    
    flength : int
        Length (in samples) of the time-domain FIR whitening filter.
        Passed in seconds (`flength/sample_rate`) to GWpy's whiten() function
        as the 'fduration' parameter.
    
    pad : int, optional
        Marging at each side of the strain to add (zero-pad) in order to avoid
        edge effects. The corrupted area at each side is `0.5 * fduration` in
        GWpy's whiten().
        Will be cropped afterwards, thus no samples are added at the end of
        the call to this function.
        If given, 'unpad' will be ignored.
    
    unpad : int, optional
        Marging at each side of the strain to crop.
        Will be ignored if 'pad' is given.
        
    highpass : float, optional
        Highpass corner frequency (in Hz) of the FIR whitening filter.
    
    normed : bool
        If True, normalizes the strains to their maximum absolute amplitude.

    **kwargs:
        Extra arguments passed to gwpy.timeseries.Timeseries.whiten().
    
    Returns
    -------
    strain_w : NDArray
        Whitened strain (in time domain).

    """
    if asd.ndim != 2:
        raise ValueError("'asd' must have 2 dimensions")

    if not isinstance(flength, int):
        raise TypeError("'flength' must be an integer")

    _asd = FrequencySeries(asd[1], frequencies=asd[0])

    if pad > 0:
        strain = np.pad(strain, pad, 'constant', constant_values=0)
        unpad = pad
 
    frame = TimeSeries(strain, sample_rate=sample_rate)
    strain_w = frame.whiten(
        asd=_asd,
        fduration=flength/sample_rate,  # to seconds
        highpass=highpass,
        **kwargs
    ).value  # Convert to numpy array!!!
    
    strain_w = strain_w[unpad:-unpad]
    if normed:
        strain_w /= np.max(np.abs(strain_w))

    return strain_w



# def highpass_filter(signal: np.ndarray,
#                     f_cut: int | float,
#                     f_width: int | float,
#                     sample_rate: int) -> np.ndarray:
#     """Apply a forward-backward digital highpass filter.

#     Apply a forward-backward digital highpass filter to 'signal' CENTERED
#     at frequency 'f_cut' with a transition band of 'f_width'.

#     It enforces the (single) filter to allow only loss of 2 dB at passband
#     (`f_cut + f_width/2` Hz) and a minimum filter of 20 dB at stopband
#     (`f_cut - f_width/2` Hz).
    
#     REFERENCES
#     ----------
#     Order selection: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html
#     Design: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
#     Filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

#     """
#     f_pass = f_cut + f_width/2
#     f_stop = f_cut - f_width/2
#     N, wn = sp.signal.buttord(wp=f_pass, ws=f_stop, gpass=2, gstop=16, fs=self.sample_rate)
#     sos = sp.signal.butter(N, wn, btype='highpass', fs=sample_rate, output='sos')
#     filtered = sp.signal.sosfiltfilt(sos, signal)

#     return filtered



def highpass_filter(signal: np.ndarray,
                    *,
                    f_cut: int | float,
                    f_order: int | float,
                    sample_rate: int) -> np.ndarray:
    """Apply a forward-backward digital highpass filter.

    Apply a forward-backward digital highpass filter to 'signal'
    at frequency 'f_cut' with an order of 'f_order'.
    
    Reference
    ---------
    Design: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    Filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

    """
    sos = sp.signal.butter(f_order, f_cut, btype='highpass', fs=sample_rate, output='sos')
    filtered = sp.signal.sosfiltfilt(sos, signal)

    return filtered


def instant_frequency(signal, *, sample_rate, phase_corrections=None):
    """Computes the instantaneous frequency of a time-domain signal.

    Computes the instantaneous frequency of a time-domain signal using the
    central difference method, with optional phase corrections.
    If negative frequencies are detected, a warning is raised.

    Parameters
    ----------
    signal : ndarray
        The input time-domain signal.
    
    sample_rate : float
        The sampling rate of the signal (in Hz).

    phase_corrections : list of tuples, optional
        A list of phase corrections.
        Each tuple contains: (jump_start, jump_end, correction_factor).
        If None, no phase correction is applied.

    Returns
    -------
    inst_freq : ndarray
        The instantaneous frequency of the signal (in Hz). 
        If negative frequencies are detected, a RuntimeWarning is issued.
    
    """
    # Step 1: Compute the analytic signal using the Hilbert transform
    analytic_signal = sp.signal.hilbert(signal)
    
    # Extract the instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic_signal))
    
    # Step 2: Apply multiple phase corrections if provided
    if phase_corrections is not None:
        # Get the time array corresponding to the signal length
        time = np.arange(len(signal)) / sample_rate
        
        for (jump_start, jump_end, correction_factor) in phase_corrections:
            inst_phase = correct_phase(inst_phase, np.arange(len(signal)) / sample_rate, 
                                       jump_start, jump_end, correction_factor)
    
    # Step 3: Compute the instantaneous frequency by differentiating the phase
    dt = 1.0 / sample_rate
    inst_phase_diff = (inst_phase[2:] - inst_phase[:-2]) / (2 * dt)
    
    # Convert phase difference to frequency
    inst_freq = inst_phase_diff / (2.0 * np.pi)
    
    # Pad the result to match the input length
    inst_freq = np.pad(inst_freq, (1, 1), mode='edge')

    if np.any(inst_freq < 0):
        warnings.warn("Non-physical negative frequencies detected in the array.", RuntimeWarning)
    
    return inst_freq



def correct_phase(phase, time, jump_start, jump_end, correction_factor=1.0):
    """Manually correct a phase jump.

    Fine-tunes the manual phase correction by adjusting the phase after a phase
    jump. The phase after the jump is scaled by the correction factor.

    Parameters
    ----------
    phase : ndarray
        The unwrapped phase of the signal.

    time : ndarray
        The time array corresponding to the signal.

    jump_start : float
        The time where the phase jump starts.

    jump_end : float
        The time where the phase jump ends.

    correction_factor : float, optional
        The factor to scale the phase correction for fine-tuning.
        Default is 1.0.

    Returns
    -------
    corrected_phase : ndarray
        The phase with fine-tuned manual correction applied after the
        specified jump.
    
    """
    corrected_phase = np.copy(phase)

    # Identify the indices corresponding to the jump start and end
    start_idx = np.argmin(np.abs(time - jump_start))
    end_idx = np.argmin(np.abs(time - jump_end))

    # Calculate the phase difference between start and end of the jump
    phase_diff = corrected_phase[end_idx] - corrected_phase[start_idx]

    # Adjust the phase after the jump by scaling the correction factor
    corrected_phase[end_idx:] -= phase_diff * correction_factor

    return corrected_phase


def snr(strain, *, psd, at, window=('tukey',0.5)):
    """Signal to Noise Ratio."""

    # rFFT
    strain = np.asarray(strain)
    ns = len(strain)
    if isinstance(window, tuple):
        window = sp.signal.windows.get_window(window, ns)
    else:
        window = np.asarray(window)
    hh = np.fft.rfft(strain * window)
    ff = np.fft.rfftfreq(ns, d=at)
    af = ff[1]

    # Lowest and highest frequency cut-off taken from the given psd
    f_min, f_max = psd[0][[0,-1]]
    i_min = np.argmin(ff < f_min)
    i_max = np.argmin(ff < f_max)
    if i_max == 0:
        i_max = len(hh)
    hh = hh[i_min:i_max]
    ff = ff[i_min:i_max]

    # SNR
    psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
    sum_ = np.sum(np.abs(hh)**2 / psd_interp)
    snr = np.sqrt(4 * at**2 * af * sum_)

    return snr
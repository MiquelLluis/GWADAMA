"""fat.py

Frequency analysis toolkit.

"""
import warnings

import numpy as np
import scipy as sp


def highpass_filter(signal: np.ndarray,
                    *,
                    f_cut: int | float,
                    f_order: int | float,
                    fs: int) -> np.ndarray:
    """Apply a forward-backward digital highpass filter.

    Apply a forward-backward digital highpass filter to 'signal'
    at frequency 'f_cut' with an order of 'f_order'.
    
    Reference
    ---------
    Design: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    Filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

    """
    sos = sp.signal.butter(f_order, f_cut, btype='highpass', fs=fs, output='sos')
    filtered = sp.signal.sosfiltfilt(sos, signal)

    return filtered


def bandpass_filter(signal: np.ndarray,
                    *,
                    f_low: int | float,
                    f_high: int | float,
                    f_order: int | float,
                    fs: int) -> np.ndarray:
    """Apply a forward-backward digital bandpass filter.

    Apply a forward-backward digital bandpass filter to 'signal'
    between frequencies 'f_low' and 'f_high' with an order of 'f_order'.

    Reference
    ---------
    Design: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    Filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

    """
    sos = sp.signal.butter(f_order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    filtered = sp.signal.sosfiltfilt(sos, signal)

    return filtered


def instant_frequency(signal, *, fs, phase_corrections=None):
    """Computes the instantaneous frequency of a time-domain signal.

    Computes the instantaneous frequency of a time-domain signal using the
    central difference method, with optional phase corrections.
    If negative frequencies are detected, a warning is raised.

    Parameters
    ----------
    signal : ndarray
        The input time-domain signal.
    
    fs : float
        The sampling frequency of the signal (in Hz).

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
        time = np.arange(len(signal)) / fs
        
        for (jump_start, jump_end, correction_factor) in phase_corrections:
            inst_phase = correct_phase(inst_phase, np.arange(len(signal)) / fs, 
                                       jump_start, jump_end, correction_factor)
    
    # Step 3: Compute the instantaneous frequency by differentiating the phase
    dt = 1.0 / fs
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


def snr(strain, *, at, psd=None, window=('tukey', 0.5), fbounds=None):
    """Compute the signal-to-noise ratio (SNR) of a signal segment.

    Compute the SNR of a signal with respect to a background spectrum
    (`psd` provided), or assuming the signal is already whitened
    (`psd=None`).

    Parameters
    ----------
    strain : array_like
        Time-domain signal segment.
    at : float
        Sampling interval (seconds per sample).
    psd : array_like (freq_array, psd_array), optional
        One-sided power spectral density of the background noise.
        If provided, the SNR is computed using this PSD.
        If None, the function assumes the signal is already whitened.
    window : str, tuple, or array_like, optional
        Window to apply before the FFT. Passed to `scipy.signal.get_window`
        if a string/tuple. If an array, it must match the length of `strain`.
    fbounds : array_like (f_min, f_max), optional
        Frequency range (Hz) over which to compute the SNR. If None:
        * with `psd`, uses the min and max of the PSD frequencies,
        * otherwise uses (0, Nyquist).

    Returns
    -------
    snr : float
        The computed signal-to-noise ratio.

    Notes
    -----
    - The function uses a one-sided rFFT and sums over positive frequencies.
    - In whitened space, PSD is assumed to be identity and the formula
      reduces to: ``sqrt(2 * Δt * Δf * Σ |h_w(f)|²)``.
    - In non-whitened space, the formula used is:
      ``sqrt(4 * Δt² * Δf * Σ |h(f)|² / S_n(f))``.

    """
    strain = np.asarray(strain)
    ns = len(strain)

    # Build window
    if isinstance(window, (tuple, str)):
        window = sp.signal.windows.get_window(window, ns)
    else:
        window = np.asarray(window)
        if window.shape != (ns,):
            raise ValueError("Window length must match strain length.")

    # rFFT
    hh = np.fft.rfft(strain * window)
    ff = np.fft.rfftfreq(ns, d=at)
    af = ff[1]  # frequency spacing

    # Frequency range
    if psd is None:
        f_min, f_max = fbounds if fbounds is not None else (0, ff[-1])
    else:
        f_min, f_max = psd[0][[0,-1]]

    i_min = np.searchsorted(ff, f_min, side='left')
    i_max = np.searchsorted(ff, f_max, side='right')
    hh = hh[i_min:i_max]
    ff = ff[i_min:i_max]

    # SNR
    if psd is None:
        # Whitened case
        sum_ = np.sum(np.abs(hh)**2)
        snr = np.sqrt(2 * at * af * sum_)
    else:
        # Non-whitened case
        psd_interp = sp.interpolate.interp1d(*psd, bounds_error=True)(ff)
        sum_ = np.sum(np.abs(hh)**2 / psd_interp)
        snr = np.sqrt(4 * at**2 * af * sum_)

    return snr


def find_power_excess(strain, fs, nperseg=16, noverlap=15, return_time=False):
    """
    Estimate the index of maximum broadband power excess in a 1D signal.

    This function provides a fast and simple estimate of the time region where
    the total broadband power in the input signal is maximised. A spectrogram
    is computed via Welch's method, and the time slice with the highest total
    power (summed over all frequencies) is identified.

    Formally, at each time step `t`, the total power is estimated as:

        P(t) = ∑_f S(f, t)

    where S(f, t) is the estimated power spectral density at frequency `f` and
    time `t`.

    The accuracy of this estimate is limited by the spectrogram parameters
    chosen by the user (`nperseg`, `noverlap`), which determine the temporal
    resolution and frequency leakage.

    Parameters
    ----------
    strain : np.ndarray
        Input signal (1D array).
    fs : int
        sampling frequency in Hz.
    nperseg : int, optional
        Length of each FFT segment (in samples).
    noverlap : int, optional
        Number of overlapping samples between segments.
    return_time : bool, optional
        If True, also return the estimated time (in seconds) corresponding to
        the peak index.

    Returns
    -------
    peak_index : int
        Sample index (in the input array) corresponding to the centre of the
        most power-concentrated region.
    peak_time : float, optional
        Time in seconds of the estimated peak, assuming the first sample occurs
        at t=0. Only returned if `return_time=True`.

    Notes
    -----
    - In real detector data, whitening or equalisation is recommended to avoid
      biasing the result towards dominant background frequencies.
    - The returned index refers to the *centre* of the peak time bin in the
      spectrogram, mapped back to the corresponding sample index of the
      original signal.
    - Spectrogram time resolution is limited by `nperseg` and `noverlap`; high
      accuracy is not guaranteed.
    - This method assumes only one dominant transient is present in the input.
      It is not suitable for detecting or resolving multiple events.
    
    """
    # Compute spectrogram: PSD estimate over time
    f, t, Sxx = sp.signal.spectrogram(
        strain,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum',
        mode='psd',
        window='hann'
    )

    # Total power per time bin (sum over all frequencies)
    power_per_time = Sxx.sum(axis=0)

    # Identify time of maximum total power
    peak_idx = np.argmax(power_per_time)
    peak_time = t[peak_idx]

    # Convert peak time (seconds) to sample index
    peak_index = int(round(peak_time * fs))
    peak_index = np.clip(peak_index, 0, len(strain) - 1)

    if return_time:
        return peak_index, peak_time
    else:
        return peak_index


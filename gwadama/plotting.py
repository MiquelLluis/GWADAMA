"""plotting.py

Custom plotting functions

"""
import warnings

import matplotlib as  mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_spectrogram_with_instantaneous_features(
        strain_array, time_array, fs=2**14, outseg=None, outfreq=None,
        window=sp.signal.windows.tukey(128,0.5), hop=32, mfft=None, vmin=-22,
        spec_interpol='lanczos', if_line_width=2):
    """Plot the spectrogram, instantaneous frequency, and strain's waveform.

    This function generates a multi-panel plot consisting of:
    
    1. A spectrogram of the gravitational wave strain, obtained using
       Short-Time Fourier Transform (STFT), visualizing the frequency evolution
       over time.
    2. The instantaneous frequency of the strain, plotted on top of the
       spectrogram to show the frequency changes in real time.
    3. The raw gravitational wave strain in the time domain, shown above the
       spectrogram for direct comparison.

    Key features of the plot:

    - **Spectrogram**: The frequency content of the gravitational wave signal
      is displayed over time using a color map (`inferno`), with the x-axis
      representing time (in milliseconds) and the y-axis representing
      frequency (in Hz).
    - **Instantaneous Frequency**: Plots the instantaneous frequency of the
      strain over time, highlighting the frequency variations.
    - **Energy Normalization**: The spectrogram uses a logarithmic scale for
      the energy (power spectral density, PSD), normalized by the maximum
      energy value in the signal.
    - **Dynamic Range Control**: The color scale of the spectrogram can be
      adjusted via the `vmin` parameter to emphasize specific energy levels.
    - **Time-Domain Waveform**: A plot of the original strain data in the time
      domain is shown above the spectrogram, providing context for the signal's
      evolution.
    - **Segmentation**: The user can specify the time (`outseg`) and frequency
      (`outfreq`) ranges to focus on specific parts of the data.
    - **Customization**: The plot has a black background, white grid lines,
      and labeled colorbars for clarity.

    Parameters
    ----------
    strain_array : numpy.ndarray
        The time-domain strain data of the gravitational wave signal.
    
    time_array : numpy.ndarray
        Array of time stamps corresponding to the strain data.
    
    fs : int, optional
        The sampling frequency of the data in Hz (default is 2^14, or 16384 Hz).
    
    outseg : tuple, optional
        A tuple specifying the time range (start, end) in seconds for the
        x-axis. If `None`, the entire time range of the input data is used.
    
    outfreq : tuple, optional
        A tuple specifying the frequency range (start, end) in Hz for the
        y-axis. If `None`, the full frequency range (up to Nyquist frequency)
        is used.
    
    window : numpy.ndarray, optional
        The window function applied during STFT computation (default is a
        Tukey window).
    
    hop : int, optional
        The hop size between successive STFT windows (default is 32).
    
    mfft : int, optional
        The number of points in the FFT used for STFT computation (default
        is None).
    
    vmin : float, optional
        The minimum value for the color scale in the spectrogram (default
        is -22). This controls the dynamic range of the color map.

    spec_interpol : str, optional
        Interpolation used for visual representation (default 'Lanczos').
        See `matplotlib.pyplot.imshow` for other options.

    if_line_width : int | float, optional
        The line width of the instantaneous frequency plot (default is 2).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the complete plot.
    
    axs : List[matplotlib.axes.Axes]
        A list of axes objects containing the spectrogram, the colorbar, and
        the time-domain plots.
    
    Sxx : numpy.ndarray
        The computed spectrogram (PSD values) of the input strain data.

    Notes
    -----
    - The spectrogram normalization is performed on the square root of the 
      Power Spectral Density (PSD), converted to a logarithmic scale.
    - The y-axis of the spectrogram uses a kilohertz scale for readability.
    - The time-domain waveform is plotted without axes labels for simplicity.
    - Instantaneous frequency values below zero are masked to avoid displaying 
      non-physical results.
    
    """
    from gwadama.fat import instant_frequency

    # Compute the spectrogram using the ShortTimeFFT class.
    stfft_model = sp.signal.ShortTimeFFT(
        win=window, hop=hop, fs=fs, mfft=mfft,
        fft_mode='onesided', scale_to='psd'
    )
    Sxx = stfft_model.spectrogram(strain_array)
    with np.errstate(divide='ignore'):
        normalized_Sxx = np.log10(np.sqrt(Sxx))
    normalized_Sxx -= np.max(normalized_Sxx)
    t0, t1, f0, f1 = stfft_model.extent(len(strain_array))
    t_origin = time_array[0]
    t0 += t_origin
    t1 += t_origin

    # Create a figure with adequate height
    fig = plt.figure(figsize=(10, 6))

    # Define a grid with 5 rows: top waveform (1), gap (1), spectrogram (3)
    gs = mpl.gridspec.GridSpec(
        nrows=4, ncols=2,
        width_ratios=[60, 1],  # Main plot vs narrow colorbar
        height_ratios=[1, 0.05, 6, 0.05],  # Top waveform, small gap, spectrogram
        hspace=0.05, wspace=0.02
    )

    # Axes
    ax3 = fig.add_subplot(gs[0, 0])  # Top waveform
    ax = fig.add_subplot(gs[2, 0], sharex=ax3)  # Spectrogram
    ax2 = fig.add_subplot(gs[2, 1])  # Colorbar

    # SPECTROGRAM (ax1)
    im = ax.imshow(normalized_Sxx, cmap='inferno', origin='lower', aspect='auto',
              extent=(t0,t1,f0,f1), interpolation=spec_interpol, vmin=vmin)
    # ...and Instant Frequency
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        instant_freq = instant_frequency(strain_array, fs=fs)
    length = len(strain_array)
    t1_instant = t_origin + (length-1)/fs
    instant_time = np.linspace(t_origin, t1_instant, length)
    mask = instant_freq >= 0  # Remove non-physical frequencies
    instant_freq = instant_freq[mask]
    instant_time = instant_time[mask]
    ax.plot(instant_time, instant_freq, 'purple', lw=if_line_width)
    
    # COLORBAR (ax2)
    cbar = fig.colorbar(im, cax=ax2)
    
    # LABELS, LIMITS, ETC
    ax.grid(True, ls='--', alpha=.4)
    # ...limits
    if outseg is None:
        ax.set_xlim(time_array[0], time_array[-1])
    else:
        ax.set_xlim(*outseg)
    if outfreq is None:
        ax.set_ylim(0, fs/2)
    else:
        ax.set_ylim(*outfreq)
    # ...labels.
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [kHz]')
    cbar.set_label('Normalized energy')
    # ...Y ticks to kHz
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))
    # ...X ticks to milliseconds and avoid roundoff errors.
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Set style first, for new ticklabels.
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    # ax.set_xticklabels(np.round(xticks * 1000).astype(int))
    ax.set_xticklabels(np.round(xticks*1000, decimals=1))
    # ...background in black to match the colormap.
    ax.set_facecolor('black')

    # GW IN TIME-DOMAIN ON TOP OF THE SPECTROGRAM (ax3)
    ax3.plot(time_array, strain_array, c='black', lw=1, alpha=1)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_ylim(np.min(strain_array), np.max(strain_array))
    ax3.axis('off')

    fig.subplots_adjust(left=0.08, right=0.91, top=0.96, bottom=0.08)
    
    return fig, [ax, ax2, ax3], Sxx
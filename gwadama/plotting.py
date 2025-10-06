"""plotting.py

Custom plotting functions

"""
import warnings

# from gwpy.timeseries import TimeSeries  # Lazy import
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy as sp
from typing import Literal, TypeAlias


# ---- Typing ----

IntInterval: TypeAlias = tuple[int, int]
FloatInterval: TypeAlias = tuple[float, float]

MaybeIntInterval: TypeAlias = IntInterval | None
MaybeFloatInterval: TypeAlias = FloatInterval | None

# ----

def plot_spectrogram_with_instantaneous_features(
    strain_array,
    time_array,
    fs=2**14,
    outseg=None,
    outfreq=None,
    window=None,
    hop=32,
    mfft=None,
    vmin=None,
    vmax=None,
    spec_log=True,
    spec_norm=True,
    spec_interpol='lanczos',
    if_line_width=2
) -> tuple[Figure, tuple[Axes,Axes,Axes], NDArray]:
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
      representing time (in milliseconds) and the y-axis representing frequency
      (in Hz).
    - **Instantaneous Frequency**: Plots the instantaneous frequency of the
      strain over time, highlighting the frequency variations.
    - **Energy Normalization**: The spectrogram uses a logarithmic scale for
      the energy (power spectral density, PSD), optionally normalized by the
      maximum energy value in the signal.
    - **Dynamic Range Control**: The color scale of the spectrogram can be
      adjusted via the `vmin` parameter to emphasize specific energy levels.
    - **Time-Domain Waveform**: A plot of the original strain data in the time
      domain is shown above the spectrogram, providing context for the signal's
      evolution.
    - **Segmentation**: The user can specify the time (`outseg`) and frequency
      (`outfreq`) ranges to focus on specific parts of the data.
    - **Customization**: The plot has a black background, white grid lines, and
      labeled colorbars for clarity.

    Parameters
    ----------
    strain_array : numpy.ndarray
        The time-domain strain data of the gravitational wave signal.
    
    time_array : numpy.ndarray
        Array of time stamps corresponding to the strain data.
    
    fs : int, optional
        The sampling frequency of the data in Hz (default is 2^14, or 16384
        Hz).
    
    outseg : tuple, optional
        A tuple specifying the time range (start, end) in seconds for the
        x-axis. If `None`, the entire time range of the input data is used.
    
    outfreq : tuple, optional
        A tuple specifying the frequency range (start, end) in Hz for the
        y-axis. If `None`, the full frequency range (up to Nyquist frequency)
        is used.
    
    window : numpy.ndarray, optional
        The window function applied during STFT computation (default is a Tukey
        window).
    
    hop : int, optional
        The hop size between successive STFT windows (default is 32).
    
    mfft : int, optional
        The number of points in the FFT used for STFT computation (default is
        None).
    
    vmin, vmax : float, optional
        The minimum/maximum value for the color scale in the spectrogram. This
        controls the dynamic range of the color map.
    
    spec_log : bool, optional
        If true, represent `np.log10(Sxx)`.

    spec_norm : bool, optional
        If true, normalize the spectrogram to the maximum energy value.

    spec_interpol : str, optional
        Interpolation used for visual representation (default 'Lanczos'). See
        `matplotlib.pyplot.imshow` for other options.

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
    - The y-axis of the spectrogram uses a kilohertz scale for readability.
    - The time-domain waveform is plotted without axes labels for simplicity.
    - Instantaneous frequency values below zero are masked to avoid displaying
      non-physical results.
    
    """
    from gwadama.fat import instant_frequency

    if window is None:
        window = sp.signal.windows.tukey(128,0.5)

    # Compute the spectrogram using the ShortTimeFFT class.
    stfft_model = sp.signal.ShortTimeFFT(
        win=window, hop=hop, fs=fs, mfft=mfft,
        fft_mode='onesided', scale_to='psd'
    )
    Sxx = stfft_model.spectrogram(strain_array)

    # Optional scaling and normalisation.
    if spec_log:
        with np.errstate(divide='ignore', invalid='ignore'):
            _Sxx = np.log10(Sxx)
        finite_mask = np.isfinite(_Sxx)
        if not finite_mask.any():
            raise ValueError("Spectrogram contains no finite values after log10.")
        if spec_norm:
            _Sxx -= np.nanmax(_Sxx)
    else:
        _Sxx = Sxx.astype(float)
        if spec_norm:
            _Sxx = _Sxx / np.nanmax(_Sxx)
    _Sxx = np.ma.masked_invalid(_Sxx)
    
    # If the user didn’t provide limits, derive safe defaults from finite data
    if vmin is None:
        vmin = float(np.nanmin(_Sxx))
    if vmax is None:
        vmax = float(np.nanmax(_Sxx))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
        raise ValueError("Invalid colour limits: ensure finite vmin < vmax.")
    
    # Make the mapping explicit and clipped
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True) # type: ignore

    # Time-frequency grid
    t0, t1, f0, f1 = stfft_model.extent(len(strain_array))
    t_origin = time_array[0]
    t0 += t_origin
    t1 += t_origin


    fig = plt.figure(figsize=(10, 6))
    # Define a grid with 5 rows: top waveform (1), gap (1), spectrogram (3)
    gs = GridSpec(
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
    im = ax.imshow(
        _Sxx, extent=(t0,t1,f0,f1),
        origin='lower', aspect='auto', cmap='inferno',
        interpolation=spec_interpol, norm=norm
    )
    # Background in black to match the colormap.
    ax.set_facecolor('black')
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
    
    # COLOURBAR (ax2)
    cbar = fig.colorbar(
        im, cax=ax2,
        boundaries=np.linspace(vmin, vmax, 256),
        ticks=np.linspace(vmin, vmax, 6),
        extend='both'
    )
    
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
    # ...X ticks to milliseconds
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x*1e3:.1f}")  # seconds → milliseconds
    )
    # ...labels.
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [Hz]')

    match (spec_log, spec_norm):
        case True, True:
            cbar.set_label(r"Norm. $\log_{10}\, \mathrm{PSD}\;[\mathrm{strain}^2/\mathrm{Hz}]$")
        case True, False:
            cbar.set_label(r"$\log_{10}\, \mathrm{PSD}\;[\mathrm{strain}^2/\mathrm{Hz}]$")
        case False, True:
            cbar.set_label(r"Norm. $\mathrm{PSD}\;[\mathrm{strain}^2/\mathrm{Hz}]$")
        case False, False:
            cbar.set_label(r"$\mathrm{PSD}\;[\mathrm{strain}^2/\mathrm{Hz}]$")

    # GW IN TIME-DOMAIN ON TOP OF THE SPECTROGRAM (ax3)
    ax3.plot(time_array, strain_array, c='black', lw=1, alpha=1)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_ylim(np.min(strain_array), np.max(strain_array))
    ax3.axis('off')

    fig.subplots_adjust(left=0.08, right=0.91, top=0.96, bottom=0.08)
    
    return fig, (ax, ax2, ax3), Sxx


def q_transform_with_strain(
    strain: NDArray,
    *,
    times: NDArray,
    fs: int = 2**14,
    outseg: MaybeFloatInterval = None,
    outfreq: MaybeFloatInterval = None,
    # Q-transform exclusive
    qrange: FloatInterval = (4, 64),
    gps: float | None = None,
    search: float = 0.5,
    whiten: bool = False,
    norm: bool | Literal["median", "mean"] = False,
    logf: bool = False,
    # pcolormesh exclusive
    color_norm: str | None = 'log',
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, tuple[Axes,Axes,Axes], NDArray]:
    """Plot the multi-Q transform and strain's time-domain waveform.

    This function generates a multi-panel plot consisting of:
    
    1. A Q-transform spectrogram of the gravitational wave strain, obtained
       using GWpy's :meth:`TimeSeries.q_transform`.
    2. The raw gravitational wave strain in the time domain, shown above the
       spectrogram for direct comparison.

    Parameters
    ----------
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the complete plot.
    
    axs : List[matplotlib.axes.Axes]
        A list of axes objects containing the spectrogram, the colorbar, and
        the time-domain plots.
    
    qspec_gwpy : gwpy.spectrogram.Spectrogram
        The computed Q-transform interpolated spectrogram.
    """
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries(strain, times=times)
    qspec_gwpy = ts.q_transform(
        qrange=qrange,
        frange=outfreq,
        outseg=outseg,
        gps=gps,
        search=search,
        whiten=whiten,
        norm=norm, # type: ignore (GWpy type missing)
        logf=logf
    )
    qspec = qspec_gwpy.value
    # Clip values ≤ 0 before log10; changes propagate to the GWpy Spectrogram
    # via the shared buffer.
    qspec[qspec <= 0] = np.min(qspec[qspec > 0])

    t_c = qspec_gwpy.xindex.value
    f_c = qspec_gwpy.yindex.value
    def centres_to_edges(c: np.ndarray) -> np.ndarray:
        """Compute bin edges from bin centres (1D)."""
        dc = np.diff(c)
        left = c[0] - dc[0]/2
        right = c[-1] + dc[-1]/2
        mids = (c[:-1] + c[1:]) / 2
        return np.concatenate(([left], mids, [right]))
    t_e = centres_to_edges(t_c)      # (nt+1,)
    f_e = centres_to_edges(f_c)      # (nf+1,)
    
    # If the user didn’t provide limits, derive safe defaults from finite data.
    if vmin is None:
        vmin = float(np.nanmin(qspec))
    if vmax is None:
        vmax = float(np.nanmax(qspec))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin >= vmax:
        raise ValueError("Invalid colour limits: ensure finite vmin < vmax.")
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True) # type: ignore

    fig = plt.figure(figsize=(10, 6))
    # Define a grid with 5 rows: top waveform (1), gap (1), spectrogram (3)
    gs = GridSpec(
        nrows=4, ncols=2,
        width_ratios=[60, 1],  # Main plot vs narrow colorbar
        height_ratios=[1, 0.05, 6, 0.05],  # Top waveform, small gap, spectrogram
        hspace=0.05, wspace=0.02
    )
    ax3 = fig.add_subplot(gs[0, 0])  # Top waveform
    ax = fig.add_subplot(gs[2, 0], sharex=ax3)  # Spectrogram
    ax2 = fig.add_subplot(gs[2, 1])  # Colorbar

    # SPECTROGRAM (ax1)
    pm = ax.pcolormesh(
        t_e, f_e, qspec.T,
        shading='auto',
        cmap='inferno',
        norm=color_norm,
        vmax=vmax,
        vmin=vmin
    )
    ax.set_facecolor('black')  # match the colormap minimum.
    
    # COLOURBAR (ax2)
    cbar = fig.colorbar(
        pm, cax=ax2,
        extend='both'
    )
    
    # LABELS, LIMITS, ETC
    ax.grid(True, ls='--', alpha=.5)
    # ...limits
    if outseg is None:
        ax.set_xlim(times[0], times[-1])
    else:
        ax.set_xlim(*outseg)
    if outfreq is None:
        ax.set_ylim(0, fs/2)
    else:
        ax.set_ylim(*outfreq)
    # ...X ticks to milliseconds
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x*1e3:.1f}")  # seconds → milliseconds
    )
    # ...labels.
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [Hz]')
    cbar.set_label(r"Energy")

    # GW IN TIME-DOMAIN ON TOP OF THE SPECTROGRAM (ax3)
    ax3.plot(times, strain, c='black', lw=1, alpha=1)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_ylim(np.min(strain), np.max(strain))
    ax3.axis('off')

    fig.subplots_adjust(left=0.08, right=0.91, top=0.96, bottom=0.08)
    
    return fig, (ax, ax2, ax3), qspec_gwpy
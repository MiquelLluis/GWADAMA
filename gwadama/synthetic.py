from typing import Callable

import numpy as np
from scipy.interpolate import make_interp_spline as sp_make_interp_spline

from . import fat



class NonwhiteGaussianNoise:
    """Simulate non-white gaussian noise.

    ¡¡¡OLD VERSION FROM MASTER'S THESIS!!! (Tweaked)

    I changed several things:
    - Now the PSD argument can be either a function or an array.

    Attributes
    ----------
    noise: NDArray

    duration: float
        Duration of the noise in seconds.
    
    psd: function
        Interpolant of PSD(f).
        If the instance is initialized with a realization of the PSD (an array)
        its original value can be accessed through the attribute '_psd'.
    
    sample_rate: int

    f_nyquist: int
        Nyquist frequency.

    rng: numpy.random.Generator

    """
    def __init__(self, *, duration, psd, sample_rate, rng, freq_cutoff=0):
        """Initialises the noise instance.

        Parameters
        ----------
        duration: float
            Duration of noise to be generated, in seconds. It may change after
            genetarting the noise, depending on its sample frequency.

        psd: function | NDArray
            Power Spectral Density of the non-white part of the noise.
            If NDArray, will be used to create a quadratic spline interpolant,
            assuming shape (2, psd_length):
                psd[0] = frequency samples
                psd[1] = psd samples

        sample_rate: int
            Sample frequency of the signal.

        random_seed: int or 1-d array_like
            Seed for numpy.random.RandomState.
        
        freq_lowcut: int, optional
            Low cut-off frequency to apply when computing noise in frequency space.
        
        rng: numpy.random.Generator
        
        """
        self.duration = duration  # May be corrected after calling _gen_noise()
        self.freq_cutoff = freq_cutoff
        self.sample_rate = sample_rate
        self.freq_nyquist = sample_rate // 2
        self.rng = rng  # Shared with the parent scope.
        self.psd, self._psd = self._setup_psd(psd)
        self._check_initial_parameters()
        
        self._gen_noise()
    
    def __getstate__(self):
        """Avoid error when trying to pickle PSD interpolator."""

        state = self.__dict__.copy()
        del state['psd']  # Delete the encapsulated (unpickable) function
        
        return state
    
    def __setstate__(self, state):
        """Avoid error when trying to pickle PSD interpolator."""

        psd, _ = self._setup_psd(state['_psd'])
        state['psd'] = psd
        self.__dict__.update(state)

    def __getitem__(self, key):
        """Direct slice access to noise array."""
        return self.noise[key]

    def __len__(self):
        """Length of the noise array."""
        return len(self.noise)

    def __repr__(self):
        args = (type(self).__name__, self.duration, self.sample_rate, self.rng.bit_generator.state)

        return "{}(t={}, sample_rate={}, random_state={})".format(*args)

    def _setup_psd(self, psd: np.ndarray | Callable) -> Callable:
        """Return the PSD array AND an interpolating function."""

        if callable(psd):
            psd_fun = psd
            # Compute a realization of the PSD function with 1 bin per
            # integer frequency.
            freqs = np.linspace(0, self.sample_rate//2, self.sample_rate//2)
            psd_array = np.stack([freqs, psd(freqs)])
            i_cut = np.argmin((freqs - self.freq_cutoff) < 0)
            psd_array[1,:i_cut] = 0
        else:
            # Build a spline quadratic interpolant for the input PSD array
            # which ensures to be 0 below the cutoff frequency.
            _psd_interp = sp_make_interp_spline(psd[0], psd[1], k=2)
            def psd_fun(freqs):
                psd = _psd_interp(freqs)
                i_cut = np.argmin((freqs - self.freq_cutoff) < 0)
                psd[:i_cut] = 0
                return psd
            psd_array = np.asarray(psd)
            
        return psd_fun, psd_array

    def inject(self, x, *, snr, snr_lim=None, pos=0, normed=False):
        """Add the simulated noise to the signal 'x'.

        Parameters
        ----------
        x : array
            Signal array. Its length must be lower or equal to
            the length of the noise array.

        snr : int or float, optional
            Signal to Noise Ratio. Defaults to 1.

        snr_lim : tuple, optional
            Limits in 'x' where to calculate the SNR.

        pos : int, optional
            Index position in the noise array where to inject the signal.
            0 by default.

        normed : boolean, optional
            Normalize 'x' to its maximum amplitude after adding the noise.
            False by default.

        Returns
        -------
        noisy : array
            Signal array with noise at the desired SNR.

        scale : float
            Coefficient used to rescale the signal.

        """
        n = len(x)
        if n > len(self.noise):
            raise ValueError("'x' is larger than the noise array")

        if snr_lim is None:
            scale = snr / self.snr(x)
        else:
            scale = snr / self.snr(x[slice(*snr_lim)])

        x_noisy = x * scale + self.noise[pos:pos+n]

        if normed:
            norm = np.max(np.abs(x_noisy))
            x_noisy /= norm
            scale /= norm

        return (x_noisy, scale)

    def rescale(self, x, *, snr):
        """Rescale the signal 'x' to the given snr w.r.t. the PSD.

        Parameters
        ----------
        x : array
            Signal array.

        snr : int or float, optional
            Signal to Noise Ratio. Defaults to 1.

        Returns
        -------
        x_new : float
            Rescaled signal.

        """
        factor = snr / self.snr(x)
        x_new = x * factor

        return (x_new, factor)

    def snr(self, x):
        """Compute the Signal to Noise Ratio.
        
        Due to the legacy code state, I need to compute a sample of the PSD
        first. But in the future I plan to make it so that it can take both
        a PSD estimation function or an array realization like I did here.

        """
        freqs = np.linspace(self.freq_cutoff, self.freq_nyquist, 2*self.sample_rate)
        psd = np.array([freqs, self.psd(freqs)])
        snr = fat.snr(x, psd=psd, at=1/self.sample_rate, window=('tukey',0.1))
        
        return snr

    def _check_initial_parameters(self):
        # Check optional arguments.
        if self.duration is not None:
            if not isinstance(self.duration, (int, float)):
                raise TypeError("'duration' must be an integer or float number")
        elif self.noise is None:
            raise TypeError("either 'duration' or 'noise' must be provided!")
        elif not isinstance(self.noise, (list, tuple, np.ndarray)):
            raise TypeError("'noise' must be an array-like iterable")

        # Check required arguments.
        if self.psd is None:
            raise TypeError("'psd' must be provided")
        if self.sample_rate is None:
            raise TypeError("'sample_rate' must be provided")

    def _gen_noise(self):
        """Generate the noise array."""
        length = int(self.duration * self.sample_rate)
        
        # Positive frequencies + 0
        n = length // 2
        f = np.arange(0, self.freq_nyquist, self.freq_nyquist/n)
        i_cut = np.argmax(f >= self.freq_cutoff)
        
        # Noise components of the positive and zero frequencies in Fourier space
        # weighted by the PSD amplitude and the normal distribution.
        psd = self.psd(f)
        psd[:i_cut] = 0  # Ensure no components are computed under the cutoff frequency.
        nf = np.sqrt(length * self.sample_rate * psd) / 2
        nf = nf*self.rng.normal(size=n) + 1j*nf*self.rng.normal(size=n)
        
        # The final noise array realization
        self.noise = np.fft.irfft(nf, n=length)
        self.duration = len(self.noise) / self.sample_rate  # Actual final duration



def sine_gaussian_waveform(times: np.ndarray,
                           *,
                           t0: float,
                           f0: float,
                           Q: float,
                           hrss: float) -> np.ndarray:
    """Generate a Sine-Gaussian-like waveform.
    
    PARAMETERS
    ----------
    times: NDArray
        Time samples.
    
    t0: float
        Time of the peak.
    
    f0: float
        Central frequency.
    
    Q: float
        Quality factor of the wave.
    
    hrss: float
        Root sum squared amplitude of the wave.
        REF: (2015) Powell J, Trifirò D, Cuoco E, Heng I S and Cavaglià M,
                Class. Quantum Grav. 32 215012.
    
    """
    h0  = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
    env = h0 * np.exp( -(np.pi * f0 / Q * (times-t0)) ** 2)
    arg = 2 * np.pi * f0 * (times - t0)
    
    return env * np.sin(arg)


def gaussian_waveform(times: np.ndarray,
                      *,
                      t0: float,
                      hrss: float,
                      duration: float,
                      amp_threshold: float) -> np.ndarray:
    """Generate a Gaussian-like waveform.
    
    PARAMETERS
    ----------
    times: NDArray
        Time samples.
    
    t0: float
        Time of the peak.
    
    hrss: float
        Root sum squared amplitude of the wave.
        REF: (2015) Powell J, Trifirò D, Cuoco E, Heng I S and Cavaglià M,
                Class. Quantum Grav. 32 215012.
    
    duration: float
        In seconds.
    
    amp_threshold: float
        Fraction w.r.t. the maximum absolute amplitude of the wave where to
        consider the amplitude of the wave zero.
        Here is used to compute the effective duration of the wave.
    
    """
    h0  = (-8*np.log(amp_threshold))**(1/4) * hrss / np.sqrt(duration)
    env = h0 * np.exp(4 * np.log(amp_threshold) * ((times-t0) / duration)**2)

    return env


def ring_down_waveform(times: np.ndarray,
                       *,
                       t0: float,
                       f0: float,
                       Q: float,
                       hrss: float) -> np.ndarray:
    """Generate a Sine-Gaussian-like waveform.

    This waveform has its peak at the beginning. In order to synchronise it
    with other waveforms which have their peak at the center, it is recommended
    to use the 't0' parameter.

    
    PARAMETERS
    ----------
    times: NDArray
        Time samples.
    
    t0: float
        Time of the peak.

    f0: float
        Central frequency.
    
    Q: float
        Quality factor of the wave.
    
    hrss: float
        Root sum squared amplitude of the wave.
        REF: (2015) Powell J, Trifirò D, Cuoco E, Heng I S and Cavaglià M,
                Class. Quantum Grav. 32 215012.

    """
    t0_ = 0
    h0  = np.sqrt(np.sqrt(2) * np.pi * f0 / Q) * hrss
    env = h0 * np.exp(- np.pi / np.sqrt(2) * f0 / Q * (times - t0_))
    arg = 2 * np.pi * f0 * (times - t0_)

    h = env * np.sin(arg)
    sample_rate = int(1/np.median(np.diff(times)))
    pad = int(t0 * sample_rate)
    h = np.pad(h, pad_width=(pad, 0))[:-pad]

    return h

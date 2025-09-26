import bilby
import numpy as np
from numpy.typing import NDArray
import scipy as sp

from . import tat


def project(
    h_plus: NDArray, h_cros: NDArray,
    *,
    parameters: dict,
    fs: int,
    nfft: int,
    detector: str,
    window: str|tuple = ('tukey', 0.04)
) -> NDArray:
    """Project strain modes in a GW detector.
    
    Project the input GW modes in the sky as detected by the specified
    detector using Bilby.
    
    PARAMETERS
    ----------
    h_plus, h_cros : NDArray
        GW polarizations.
    
    parameters : dict
        Sky position and time of the event, as requested by Bilby's method:

        ```
        {
            ra: 0,
            dec: 0,
            geocent_time: 0, # In GPS.
            psi: 0  # Binary polarization angle
        }
        ```

        REF: https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.interferometer.Interferometer.html#bilby.gw.detector.interferometer.Interferometer.get_detector_response
    
    fs : int
        Sampling frequency of the waveform.
    
    nfft : int
        Length of the FFT window.
    
    window : str | tuple
        Passed to :func:`sp.signal.get_window`.
    
    detector : str
        GW detector into which the modes will be projected.
        Must exist in Bilby's InterferometerList().
    
    RETURNS
    -------
    strains_projected: NDArray
        Strain(s) projected. If only one detector specified, it is a 1d-array.
        Otherwise, a 2d-array with shape (3, strain).

    NOTES
    -----
    - Strains are converted to frequency domain in order to project them,
      and hence must be windowed.
    - Before the FFT, a Tukey window is used with `alpha=0.04`, therefore the
      initial length of the strains has to be taken into account since around
      2% of the beginning and the end of the signal will be damped.
        
    """
    ifo = bilby.gw.detector.InterferometerList([detector])[0]

    l_input = len(h_plus)
    assert l_input <= nfft

    # Apply window and pad signal
    pad_l = (nfft - l_input)//2
    pad_r = pad_l + (nfft - l_input)%2
    w = sp.signal.get_window(window, l_input)
    h_plus_padded = np.pad(h_plus*w, (pad_l,pad_r))
    h_cros_padded = np.pad(h_cros*w, (pad_l,pad_r))

    i_merger_pad = tat.find_merger(h_plus_padded - 1j*h_cros_padded)

    # Bilby works in frequencies.
    frequencies = np.fft.rfftfreq(nfft, d=1/fs)
    waveform_polarizations = {
        'plus': np.fft.rfft(h_plus_padded),
        'cross': np.fft.rfft(h_cros_padded)
    }

    # Project the GW
    strains_projected = ifo.get_detector_response(
        waveform_polarizations,
        parameters,
        frequencies=frequencies
    )
    strains_projected = np.fft.irfft(strains_projected)

    # Get back the original length and position of the GW.
    i_merger = tat.find_merger(strains_projected)
    i0 = i_merger - i_merger_pad + pad_l
    i1 = i0 + l_input
    strains_projected = strains_projected[i0:i1]


    return strains_projected

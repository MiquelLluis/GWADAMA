import bilby
import numpy as np
import scipy as sp
import scipy.signal

from clawdia.estimators import find_merger



def project(h_plus: np.ndarray, h_cros: np.ndarray,
            *,
            parameters: dict,
            sf: int,
            nfft: int,
            detector: str) -> np.ndarray:
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
    
    sf : int
        Sample rate of the waveform.
    
    nfft : int
        Length of the FFT window.
    
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

    # Pad signal and apply window (first)
    pad_l = (nfft - l_input)//2
    pad_r = pad_l + (nfft - l_input)%2
    window = sp.signal.windows.tukey(l_input, 0.04)
    h_plus_padded = np.pad(h_plus*window, (pad_l,pad_r))
    h_cros_padded = np.pad(h_cros*window, (pad_l,pad_r))

    i_merger_pad = find_merger(h_plus_padded - 1j*h_cros_padded)

    # Bilby works in frequencies.
    frequencies = np.fft.rfftfreq(nfft, d=1/sf)
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
    i_merger = find_merger(strains_projected)
    i0 = i_merger - i_merger_pad + pad_l
    i1 = i0 + l_input
    strains_projected = strains_projected[i0:i1]


    return strains_projected

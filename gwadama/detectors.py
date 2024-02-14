import bilby
import numpy as np
import scipy as sp
import scipy.signal

from clawdia.estimators import find_merger



def project_et(h_plus: np.ndarray, h_cros: np.ndarray, *, parameters: dict,
               sf: int, nfft: int, detector='all') -> np.ndarray:
    """Project strain modes in ET.
    
    Project the input GW modes in the sky as detected by the 3rd generation
    Einstein Telescope (ET) detector in the proposed configuration.
    
    PARAMETERS
    ----------
    h_plus, h_cros: NDArray
        GW polarizations.
    
    parameters: dict
        Sky position and time of the event, as requested by Bilby's method:
        {
            ra: 0,
            dec: 0,
            geocent_time: 0, # In GPS.
            psi: 0  # Binary polarisation angle
        }
        REF: https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.interferometer.Interferometer.html#bilby.gw.detector.interferometer.Interferometer.get_detector_response
    
    sf: int
        Sample rate of the waveform.
    
    nfft: int
        Length of the FFT window.
    
    detector: str, optional
        If 'all', return the projection in all three detectors of ET.
        Otherwise, specify a single detector of ET: ET1, ET2, ET3.
    
    RETURNS
    -------
    strains_et: NDArray
        Strain(s) projected. If only one detector specified, it is a 1d-array.
        Otherwise, a 2d-array with shape (3, strain).
        
    """
    et_list = bilby.gw.detector.InterferometerList(['ET'])
    if detector != 'all':
        i = {'ET1': 0, 'ET2': 1, 'ET3': 2}[detector]
        et_list = et_list[i]

    l_input = len(h_plus)
    assert l_input <= nfft

    # Pad signal and apply window
    pad_l = (nfft - l_input)//2
    pad_r = pad_l + (nfft - l_input)%2
    window = sp.signal.windows.tukey(nfft, 0.3)
    h_plus_padded = np.pad(h_plus, (pad_l,pad_r)) * window
    h_cros_padded = np.pad(h_cros, (pad_l,pad_r)) * window

    i_merger_pad = find_merger(h_plus_padded - 1j*h_cros_padded)

    # Bilby works in frequencies.
    frequencies = np.fft.rfftfreq(nfft, d=1/sf)
    waveform_polarizations = {
        'plus': np.fft.rfft(h_plus_padded),
        'cross': np.fft.rfft(h_cros_padded)
    }

    # Project the GW in ET.

    if detector == 'all':
        strains_et = np.empty((3, l_input), dtype=float)
        for i, eti in enumerate(et_list):
            strain_eti = eti.get_detector_response(
                waveform_polarizations,
                parameters,
                frequencies=frequencies
            )
            strain_eti = np.fft.irfft(strain_eti)

            # Get back the original length and position of the GW.
            i_merger_et = find_merger(strain_eti)
            i0 = i_merger_et - i_merger_pad + pad_l
            i1 = i0 + l_input
            strains_et[i] = strain_eti[i0:i1]
    else:
        strains_et = et_list.get_detector_response(
            waveform_polarizations,
            parameters,
            frequencies=frequencies
        )
        strains_et = np.fft.irfft(strains_et)

        # Get back the original length and position of the GW.
        i_merger_et = find_merger(strains_et)
        i0 = i_merger_et - i_merger_pad + pad_l
        i1 = i0 + l_input
        strains_et = strains_et[i0:i1]


    return strains_et

"""fat.py

Frequency analysis toolkit.

"""


from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import numpy as np


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
    and shrinking signals afterwarwds to 'l_window' to account for the vignet
    effect introduced by the windowing.

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
        vigneting. The corrupted area at each side is `0.5 * fduration` in
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
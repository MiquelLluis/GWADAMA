"""fat.py

Frequency analysis toolkit.

"""


from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import numpy as np


def whiten(strain, *, asd, margin, sample_rate, fduration, highpass, normed=True, **kwargs):
    """Whiten a single strain signal.

    Whiten a strain using the input amplitude spectral density 'asd',
    and shrinking signals afterwarwds to 'l_window' to account for the vignet
    effect introduced by the windowing.

    Parameters
    ----------
    strain : NDArray
        Strain data points in time domain.
    
    asd : FrequencySeries or 2d-array
        Amplitude spectral density assumed for the 'set_strain'.
        If a 2d-array is given, its components are:
        - asd[0] = frequency points
        - asd[1] = ASD points
        NOTE: It must has a linear and constant sampling rate!
    
    margin : int
        Marging at each side of the strain to crop in order to avoid vigneting
        due to the windowing during the whitening.
        The corrupted area at each side is `0.5*fduration` in GWpy's whiten(),
        which by default is 2 s, thus 1 s at each side.
    
    sample_rate : int

    fduration : int
        Length (in samples) of the time-domain FIR whitening filter.
    
    highpass : int
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
    if not isinstance(asd, FrequencySeries):
        asd = FrequencySeries(asd[1], frequencies=asd[0])
 
    frame = TimeSeries(strain, sample_rate=sample_rate)
    strain_w = frame.whiten(
        asd=asd,
        fduration=fduration/sample_rate,  # to seconds
        highpass=highpass,
        **kwargs
    ).value
    strain_w = strain_w[margin:-margin]

    if normed:
        strain_w /= np.max(np.abs(strain_w))

    return strain_w
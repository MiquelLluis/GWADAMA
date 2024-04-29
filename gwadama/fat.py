"""fat.py

Frequency analysis toolkit.

"""


from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import numpy as np


def whiten(strain, *, asd, sample_rate, **kwargs):
    """Whiten a single strain signal.

    Whiten a strain using the input amplitude spectral density 'asd'.
    It does NOT shrink the strain to remove the vignetting effect due to the
    windowing at the edges.

    Parameters
    ----------
    strain : NDArray
        Strain data points in time domain.

    asd : 2d-array
        Amplitude spectral density assumed for the 'strain'.
        Components:
        - asd[0] = frequency points
        - asd[1] = ASD points
        NOTE: It must has a linear and constant sampling rate! Otherwise GWpy
        will raise an error.
    
    sample_rate : int

    **kwargs :
        Extra arguments passed to gwpy.timeseries.Timeseries.whiten()
    
    Returns
    -------
    strain_w : NDArray
        Whitened data.

    """
    if not isinstance(asd, FrequencySeries):
        asd = FrequencySeries(asd[1], frequencies=asd[0])
    frame = TimeSeries(strain, sample_rate=sample_rate)
    strain_w = frame.whiten(asd=asd, **kwargs).value

    return strain_w


def whiten_batch(set_strain, *, l_window, asd, sf, normed=True, **kwargs):
    """Whiten a batch of strains.

    Whiten a batch of strains using the input amplitude spectral density 'asd',
    and shrinking signals afterwarwds to 'l_window' to account for the vignet
    effect introduced by the windowing.

    NOTE: This does not provide extra parallelization for the whitening, just
    iterates for each strain in 'set_strain'. It does parallelize the cropping,
    although it shouldn't be relevant.

    PARAMETERS
    ----------
    set_strain : 2d-array, shape (n_strains, strain_samples)
        Batch of strains to be whitened.
    
    l_window : int
        Target (final) strain length. Must be `l_window < set_strain.shape[1]`,
        and the cropping window is centered, so that at each side `l_window//2`
        is cropped.

    asd : 2d-array
        Amplitude spectral density assumed for the 'set_strain'.
        Components:
        - asd[0] = frequency points
        - asd[1] = ASD points
        NOTE: It must has a linear and constant sampling rate! Otherwise GWpy
        will raise an error.
    
    normed : bool
        If True, normalizes the strains to their maximum absolute amplitude.

    **kwargs:
        Extra arguments passed to gwpy.timeseries.Timeseries.whiten().

    """
    if not isinstance(asd, FrequencySeries):
        asd = FrequencySeries(asd[1], frequencies=asd[0])
    n_strains = len(set_strain)
    i0 = (set_strain.shape[1] - l_window) // 2
    i1 = i0 + l_window
    
    set_whitened = np.empty((n_strains, l_window))
    for i in range(n_strains):
        strain_w = whiten(set_strain[i], asd=asd, sample_rate=sf, **kwargs)
        set_whitened[i] = strain_w[i0:i1]

    if normed:
        set_whitened /= np.max(np.abs(set_whitened), axis=1, keepdims=True)

    return set_whitened
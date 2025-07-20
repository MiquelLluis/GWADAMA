import pytest
import numpy as np
import pandas as pd

from gwadama.datasets import (
    UnlabeledWaves, InjectedUnlabeledWaves,
    # SyntheticWaves, InjectedSyntheticWaves,
    # CoReWaves, InjectedCoReWaves
)


def generate_test_strains(
        n=9,
        l=4096,
        sample_rate=4096,
        sine_freqs=None,
        gauss_sigmas=None,
        burst_freqs=None,
        burst_sigmas=None,
    ):
    """
    Generate a test strain array with n signals of shape (n, l),
    using explicitly given physical parameters for each wave type.

    """
    if n < 3:
        raise ValueError("n must be at least 3")

    t = np.linspace(0, l / sample_rate, l, endpoint=False)
    center = t[len(t) // 2]

    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2  # leftovers go to burst type

    # Validate input lists
    if sine_freqs is None or len(sine_freqs) != n1:
        raise ValueError(f"sine_freqs must be a list of length {n1}")
    if gauss_sigmas is None or len(gauss_sigmas) != n2:
        raise ValueError(f"gauss_sigmas must be a list of length {n2}")
    if burst_freqs is None or len(burst_freqs) != n3:
        raise ValueError(f"burst_freqs must be a list of length {n3}")
    if burst_sigmas is None or len(burst_sigmas) != n3:
        raise ValueError(f"burst_sigmas must be a list of length {n3}")

    signals = {'sine': {}, 'gauss': {}, 'burst': {}}
    metadata = {
        'sine_freqs': [], 
        'gauss_sigmas': [], 
        'burst_freqs': [], 
        'burst_sigmas': []
    }

    id_ = 0

    # 1. Sine waves
    for freq in sine_freqs:
        s = np.sin(2 * np.pi * freq * t)
        signals['sine'][id_] = s
        metadata['sine_freqs'].append(freq)
        metadata['gauss_sigmas'].append(None)
        metadata['burst_freqs'].append(None)
        metadata['burst_sigmas'].append(None)

    # 2. Gaussian pulses
    for sigma in gauss_sigmas:
        g = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        signals['gauss'][id_] = g
        metadata['sine_freqs'].append(None)
        metadata['gauss_sigmas'].append(sigma)
        metadata['burst_freqs'].append(None)
        metadata['burst_sigmas'].append(None)

    # 3. Sine-Gaussian bursts
    for freq, sigma in zip(burst_freqs, burst_sigmas):
        b = np.sin(2 * np.pi * freq * t) * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        signals['burst'][id_] = b
        metadata['sine_freqs'].append(None)
        metadata['gauss_sigmas'].append(None)
        metadata['burst_freqs'].append(freq)
        metadata['burst_sigmas'].append(sigma)

    return signals, pd.DataFrame(data=metadata)


@pytest.fixture
def raw_strains():
    strains, metadata = generate_test_strains(
        n=9,
        l=4096,
        sample_rate=4096,
        sine_freqs=[20, 30, 40],
        gauss_sigmas=[0.1, 0.12, 0.14],
        burst_freqs=[70, 60, 50],
        burst_sigmas=[0.1, 0.12, 0.14]
    )
    return strains, metadata


@pytest.fixture
def unlabeled_dataset(raw_strains):
    ds = UnlabeledWaves(
        raw_strains[0],
        fs=4096,
        random_seed=1999
    )
    return ds
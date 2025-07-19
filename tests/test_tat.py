import numpy as np
import pytest
from numpy.testing import assert_allclose

from gwadama.tat import resample, gen_time_array

#------------------------------------------------------------------------------
# Fixtures for simple synthetic signals
#------------------------------------------------------------------------------

@pytest.fixture
def simple_sine():
    """A simple sine wave with known frequency and uniform time array.
    
    Since the sine contains the exact number of periods within the duration of
    the signal, NO WINDOWING is needed for FFT operations.

    """
    fs = 100   # Hz
    duration = 1  # seconds
    freq = 5      # Hz
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)

    return signal, t, fs, freq


#------------------------------------------------------------------------------
# Tests for resample()
#------------------------------------------------------------------------------

def test_resample_basic_functionality(simple_sine):
    """Resampling a uniformly sampled sine wave should return a signal with
    the correct length and sampling rate.

    NOTE: Signal's content is not checked, since the actual resample is done
    by SciPy's `resample_poly`.
    
    """
    signal, t, fs_in, _ = simple_sine
    target_fs = 50

    out, t_out, fs_interp, up, down = resample(signal, t, sample_rate=target_fs, full_output=True)

    # Check new sampling rate
    inferred_sr = int(round(1 / (t_out[1] - t_out[0])))
    assert inferred_sr == target_fs
    assert np.allclose(np.diff(t_out), 1/target_fs)

    # Output length should match duration * target_fs
    expected_len = target_fs * (t_out[-1] - t_out[0]) + 1  # approx
    assert len(out) == expected_len

    # Check interpolation factor consistency
    assert fs_interp == fs_in
    assert up * fs_in == down * target_fs
    assert np.gcd(up, down) == 1  # coprimality


def test_resample_raises_on_negative_rate(simple_sine):
    """Target sample rate must be positive."""
    signal, t, _, _ = simple_sine
    with pytest.raises(ValueError):
        resample(signal, t, sample_rate=-10)


def test_resample_raises_on_non_array_times(simple_sine):
    """Times must be a NumPy array."""
    signal, t, _, _ = simple_sine
    with pytest.raises(TypeError):
        resample(signal, list(t), sample_rate=50)


def test_resample_interpolates_nonuniform_times(simple_sine):
    """If times are non-uniform, the function should interpolate first."""
    fs_target = 50
    signal, t, _, _ = simple_sine
    t_jittered = t.copy()
    t_jittered[10] += 0.001  # break uniform spacing slightly

    out, t_out, _, _, _ = resample(signal, t_jittered, sample_rate=fs_target, full_output=True)

    # Check output shape
    assert isinstance(out, np.ndarray)
    assert isinstance(t_out, np.ndarray)
    # Ensure that it still returns a valid signal of expected length
    assert len(out) == int((t_jittered[-1] - t_jittered[0]) * fs_target) + 1
    assert len(t_out) == len(out)


def test_resample_returns_only_signal_when_full_output_false(simple_sine):
    """When full_output=False, only the signal should be returned."""
    signal, t, _, _ = simple_sine
    result = resample(signal, t, sample_rate=50, full_output=False)
    assert isinstance(result, np.ndarray)


def test_resample_preserves_signal_shape(simple_sine):
    """Resampling down and back up should roughly preserve signal shape."""
    signal, t, fs_in, _ = simple_sine
    # Downsample to 50 Hz
    downsampled, t_down, _, _, _ = resample(signal, t, sample_rate=50, full_output=True)
    # Upsample back to 100 Hz
    upsampled, t_up, _, _, _ = resample(downsampled, t_down, sample_rate=fs_in, full_output=True)

    # Compare to original (allowing for some tolerance, and discarding edges)
    assert_allclose(upsampled[15:-15], signal[15:-15], atol=1e-3)
    # Ttimes should be equal
    np.testing.assert_equal(t, t_up)



#------------------------------------------------------------------------------
# Tests for gen_time_array()
#------------------------------------------------------------------------------

def test_gen_time_array_basic():
    """Generated time array should have correct length, spacing, and boundaries."""
    t0 = 0.0
    t1 = 1.0
    fs = 100.0
    ts = 1/fs

    t = gen_time_array(t0, t1, fs)

    # Check length
    expected_len = int((t1 - t0) * fs)
    assert len(t) == expected_len

    # Check start and stop values
    assert t[0] == t0
    # Last value should be < t1 (exclusive)
    assert np.isclose(t[-1], t1-ts)
    # Check uniform spacing
    assert np.allclose(np.diff(t), ts)


def test_gen_time_array_with_length_ok():
    """When length matches expected, output should still be correct."""
    t0 = 0.0
    t1 = 2.0
    fs = 50.0
    ts = 1/fs
    length = int((t1 - t0) * fs)

    t = gen_time_array(t0, t1, fs, length=length)

    assert len(t) == length
    assert t[0] == t0
    assert np.isclose(t[-1], t1-ts)
    assert np.allclose(np.diff(t), ts)


def test_gen_time_array_with_length_mismatch():
    """When length does not match expected, ValueError must be raised."""
    t0 = 0.0
    t1 = 1.0
    fs = 100.0
    wrong_length = 9999  # deliberately incorrect

    with pytest.raises(ValueError) as e:
        gen_time_array(t0, t1, fs, length=wrong_length)
    assert "Expected" in str(e.value)


def test_gen_time_array_non_integer_rate():
    """Function should still work with non-integer sample rates."""
    t0 = 0.0
    t1 = 0.5
    fs = 44.1  # kHz in audio
    t = gen_time_array(t0, t1, fs)

    expected_len = int((t1 - t0) * fs)
    assert len(t) == expected_len

    # Check consistent sampling rate, taking into account possible deviations
    # when `fs` is fractionary and the total length is relatively short.
    fs_out = 1 / (t[1] - t[0])
    fs_expected = expected_len / (t1 - t0)
    assert np.isclose(fs_out, fs_expected), (
        f"Effective fs {fs_out} does not match truncated expectation {fs_expected}"
    )


def test_gen_time_array_zero_duration():
    """If t1 equals t0, output should be empty."""
    t0 = 5.0
    t1 = 5.0
    fs = 100.0
    t = gen_time_array(t0, t1, fs)

    assert isinstance(t, np.ndarray)
    assert t.size == 0

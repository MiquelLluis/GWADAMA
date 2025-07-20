import numpy as np
import pytest
from numpy.testing import assert_allclose

from gwadama.tat import (
    resample, gen_time_array, time_array_like, pad_time_array, find_time_origin,
    find_merger, planck, truncate_transfer
)

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

    out, t_out, fs_interp, up, down = resample(signal, t, fs=target_fs, full_output=True)

    # Check new sampling rate
    inferred_fs = int(round(1 / (t_out[1] - t_out[0])))
    assert inferred_fs == target_fs
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
        resample(signal, t, fs=-10)


def test_resample_raises_on_non_array_times(simple_sine):
    """Times must be a NumPy array."""
    signal, t, _, _ = simple_sine
    with pytest.raises(TypeError):
        resample(signal, list(t), fs=50)


def test_resample_interpolates_nonuniform_times(simple_sine):
    """If times are non-uniform, the function should interpolate first."""
    fs_target = 50
    signal, t, _, _ = simple_sine
    t_jittered = t.copy()
    t_jittered[10] += 0.001  # break uniform spacing slightly

    out, t_out, _, _, _ = resample(signal, t_jittered, fs=fs_target, full_output=True)

    # Check output shape
    assert isinstance(out, np.ndarray)
    assert isinstance(t_out, np.ndarray)
    # Ensure that it still returns a valid signal of expected length
    assert len(out) == int((t_jittered[-1] - t_jittered[0]) * fs_target) + 1
    assert len(t_out) == len(out)


def test_resample_returns_only_signal_when_full_output_false(simple_sine):
    """When full_output=False, only the signal should be returned."""
    signal, t, _, _ = simple_sine
    result = resample(signal, t, fs=50, full_output=False)
    assert isinstance(result, np.ndarray)


def test_resample_preserves_signal_shape(simple_sine):
    """Resampling down and back up should roughly preserve signal shape."""
    signal, t, fs_in, _ = simple_sine
    # Downsample to 50 Hz
    downsampled, t_down, _, _, _ = resample(signal, t, fs=50, full_output=True)
    # Upsample back to 100 Hz
    upsampled, t_up, _, _, _ = resample(downsampled, t_down, fs=fs_in, full_output=True)

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

    t = gen_time_array(t0, t1, fs=fs)

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

    t = gen_time_array(t0, t1, fs=fs, length=length)

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
        gen_time_array(t0, t1, fs=fs, length=wrong_length)
    assert "Expected" in str(e.value)


def test_gen_time_array_non_integer_rate():
    """Function should still work with non-integer sample rates."""
    t0 = 0.0
    t1 = 0.5
    fs = 44.1  # kHz in audio
    t = gen_time_array(t0, t1, fs=fs)

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
    t = gen_time_array(t0, t1, fs=fs)

    assert isinstance(t, np.ndarray)
    assert t.size == 0



#------------------------------------------------------------------------------
# Tests for time_araray_like()
#------------------------------------------------------------------------------

def test_time_array_like_basic(simple_sine):
    """
    Generated time array should match the length of the input, sampling rate,
    and time offset.
    
    """
    fs = 100
    t0 = 123
    times = time_array_like(simple_sine, fs=fs, t0=t0)

    assert len(times) == len(simple_sine)
    assert_allclose(np.diff(times), 1/fs)
    assert times[0] == t0
    # Note: the last point is implicitly checked by the combination of the
    # previous three assertions.


@pytest.mark.parametrize("n", [1, 10, 1000])
def test_time_array_like_lengths(n):
    """Test edge case and different lengths."""

    arr = np.zeros(n)
    times = time_array_like(arr, fs=100.0, t0=5.0)
    
    assert len(times) == n
    if n > 1:
        assert_allclose(np.diff(times), 1/100.0)
    assert times[0] == 5.0


@pytest.mark.parametrize("input_array", [
    [0, 1, 2, 3],
    (0, 1, 2, 3),
    np.arange(4)
])
def test_time_array_like_arraylike(input_array):
    """Test input type flexibility."""

    times = time_array_like(input_array, fs=10, t0=1.0)
    expected = 1.0 + np.arange(len(input_array)) / 10
    
    assert_allclose(times, expected)


def test_time_array_like_empty():
    """Empty imput should return empty array without errors."""

    times = time_array_like([], fs=123.0, t0=7.0)
    
    assert isinstance(times, np.ndarray)
    assert times.size == 0


def test_time_array_like_float_fs():
    """Ensure floating-point `fs` is handled correctly."""

    n = 3
    fs = 2.5
    t0 = 1.0
    times = time_array_like(np.zeros(n), fs=fs, t0=t0)
    expected = t0 + np.arange(n) / fs
    assert_allclose(times, expected)



#------------------------------------------------------------------------------
# Tests for time_araray_like()
#------------------------------------------------------------------------------

def test_pad_time_array_with_int():
    """
    Padded array should add the same number of samples on both sides when pad
    is an int.
    """
    times = np.linspace(0.0, 1.0, 6)  # 0,0.2,0.4,0.6,0.8,1.0
    dt = times[1] - times[0]
    padded = pad_time_array(times, 2)
    
    assert len(padded) == len(times) + 4
    assert_allclose(padded[0], times[0] - 2 * dt)
    assert_allclose(padded[-1], times[-1] + 2 * dt)


def test_pad_time_array_with_tuple():
    """
    Padded array should add different numbers of samples on each side when pad
    is a tuple.
    """
    times = np.linspace(10.0, 10.5, 6)  # spacing 0.1
    padded = pad_time_array(times, (1, 3))
    dt = times[1] - times[0]
    
    assert len(padded) == len(times) + 1 + 3
    assert_allclose(padded[0], times[0] - 1 * dt)
    assert_allclose(padded[-1], times[-1] + 3 * dt)


def test_pad_time_array_zero_pad():
    """
    A null pad should yield a time array identical to the input.
    """
    times = np.linspace(5.0, 5.4, 5)
    padded = pad_time_array(times, 0)
    assert_allclose(padded, times)


def test_pad_time_array_nonzero_start():
    """
    Padded array should correctly account for a non-zero starting time.
    """
    times = np.array([100.0, 100.5, 101.0])  # dt = 0.5
    padded = pad_time_array(times, 1)
    dt = 0.5
    assert_allclose(padded[0], 100.0 - dt)
    assert_allclose(padded[-1], 101.0 + dt)


@pytest.mark.parametrize("pad", [1, (1, 1), (0, 2)])
def test_pad_time_array_spacing_consistency(pad):
    """
    Padded array should preserve the original time step between samples.
    """
    times = np.linspace(0.0, 1.0, 6)  # dt = 0.2
    padded = pad_time_array(times, pad)
    # Check that all differences are equal to the original dt
    dt = times[1] - times[0]
    assert_allclose(np.diff(padded), dt)


def test_pad_time_array_non_uniform_raises():
    """
    Non-uniformly spaced input times should raise a ValueError.
    """
    times = np.array([0.0, 0.1, 0.25])  # not uniform
    with pytest.raises(ValueError, match="uniformly sampled"):
        pad_time_array(times, 1)


def test_pad_time_array_uniform_but_floating_error_tolerance():
    """
    Time arrays with minimal floating-point irregularities within tolerance
    should still be accepted as uniform.
    """
    times = np.array([0.0, 0.1000000001, 0.2000000002])  # close to uniform
    # Should not raise
    padded = pad_time_array(times, 1)
    dt = times[1] - times[0]
    assert_allclose(np.diff(padded), dt)



#------------------------------------------------------------------------------
# Tests for time_araray_like()
#------------------------------------------------------------------------------

def test_find_time_origin_basic():
    """Should return the index of the value closest to zero."""
    times = np.array([-3.0, -0.1, 0.2, 4.0])
    assert find_time_origin(times) == 1  # -0.1 is closest to zero

def test_find_time_origin_exact_zero():
    """Should return the index of an exact zero if present."""
    times = np.array([-1.0, 0.0, 5.0])
    assert find_time_origin(times) == 1  # exact zero at index 1



#------------------------------------------------------------------------------
# Tests for time_araray_like()
#------------------------------------------------------------------------------

def test_find_merger_returns_peak_index():
    """Should return the index of the element with maximum absolute value."""
    h = np.array([-0.1, 0.3, -0.5, 0.4])
    # absolute values: [0.1, 0.3, 0.5, 0.4]
    assert find_merger(h) == 2  # 0.5 at index 2 is the maximum



#------------------------------------------------------------------------------
# Tests for time_araray_like()
#------------------------------------------------------------------------------

def test_planck_basic_properties():
    """The window should have max value 1 and be non-negative."""
    w = planck(128, nleft=10, nright=10)
    assert np.all(w >= 0)
    assert np.isclose(w.max(), 1.0)


@pytest.mark.parametrize("N,nleft,nright", [
    (16, 0, 0),   # no taper
    (16, 4, 0),   # left taper only
    (16, 0, 4),   # right taper only
    (16, 4, 4),   # both tapers
])
def test_planck_shapes_and_length(N, nleft, nright):
    """Output length should match N for all parameter combinations."""
    w = planck(N, nleft, nright)
    
    assert len(w) == N
    
    # Edge behaviour depending on taper
    if nleft > 0:
        assert w[0] == 0
    else:
        assert_allclose(w[0], 1)
    if nright > 0:
        assert w[-1] == 0
    else:
        assert_allclose(w[-1], 1)

    # Flat region (if exists) should be all ones
    if nleft + nright < N:
        assert_allclose(w[nleft:-nright], 1)


def test_planck_left_taper_monotonicity():
    """Left taper should smoothly increase from 0 to 1."""
    N = 16
    nleft = 5
    w = planck(N, nleft, 0)
    left = w[:nleft]
    # Starts at 0
    assert np.isclose(left[0], 0.0)
    # Should be strictly increasing
    assert np.all(np.diff(left) > 0)


def test_planck_right_taper_monotonicity():
    """Right taper should smoothly decrease from 1 to 0."""
    N = 16
    nright = 5
    w = planck(N, 0, nright)
    right = w[-nright:]
    # Ends at 0
    assert np.isclose(right[-1], 0.0)
    # Should be strictly decreasing
    assert np.all(np.diff(right) < 0)


def test_planck_flat_region_equals_one():
    """The middle (untapered) region should equal 1."""
    N = 16
    nleft = 4
    nright = 4
    w = planck(N, nleft, nright)
    middle = w[nleft:N-nright]
    assert np.allclose(middle, 1.0)



#------------------------------------------------------------------------------
# Tests for truncate_transfer()
#------------------------------------------------------------------------------

def test_truncate_transfer_basic():
    """
    Verifies that applying `truncate_transfer` to a flat transfer function:
    * forces the first and last samples to zero (due to the Planck taper),
    * preserves values in the central flat region between the taper edges.
    
    """
    series = np.ones(64)

    # test truncate_transfer
    trunc1 = truncate_transfer(series)
    assert trunc1[0] == 0
    assert trunc1[-1] == 0
    # central region should remain equal (taper does nothing here)
    assert_allclose(trunc1[5:59], series[5:59])


def test_truncate_transfer_with_ncorner():
    """First `ncorner` samples should be forced to zero."""
    series = np.ones(64)
    ncorner = 3
    trunc = truncate_transfer(series, ncorner=ncorner)
    # first ncorner samples zeroed
    assert np.all(trunc[:ncorner] == 0)
    # region after ncorner still follows taper behaviour
    assert trunc[ncorner+1] > 0  # taper rising after corner zeroing
    assert trunc[-1] == 0


@pytest.mark.parametrize("ncorner", [None, 0, 3, 10])
def test_truncate_transfer_preserves_length(ncorner):
    """Output should have the same shape as the input."""
    series = np.ones(128)
    trunc = truncate_transfer(series, ncorner=ncorner)
    assert trunc.shape == series.shape


def test_truncate_transfer_nonnegative_and_bounded():
    """Output should be non-negative and not exceed input amplitude."""
    series = np.ones(64)
    trunc = truncate_transfer(series, ncorner=5)
    assert np.all(trunc >= 0)
    assert np.all(trunc <= 1)


def test_truncate_transfer_none_vs_zero():
    """ncorner=None and ncorner=0 should produce identical results."""
    series = np.ones(64)
    trunc_none = truncate_transfer(series, ncorner=None)
    trunc_zero = truncate_transfer(series, ncorner=0)
    np.testing.assert_allclose(trunc_none, trunc_zero)

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["centroid_moving_average", "fft_signal_clean"]

# def LPF_MAM(x: np.ndarray, times: np.ndarray, tau=12 * 60) -> np.ndarray:
#     """back moving average"""
#     k = np.round(tau / (times[1] - times[0])).astype(int)
#     x_mean = np.zeros(x.shape)
#     N = x.shape[0]
#     for i in range(N):
#         if i == 0:
#             x_mean[i] = x[0]
#         elif i - k < 0:
#             x_mean[i] = x[:i].mean()
#         else:
#             x_mean[i] = x[i - k : i].mean()
#     return x_mean


def centroid_moving_average(x: NDArray[np.float_], times: NDArray[np.int_], tau=12 * 60) -> NDArray[np.float_]:
    k = int(np.round(tau / (times[1] - times[0])))
    x_mean = np.zeros(x.shape)
    n = x.shape[0]

    for i in range(n):
        if i < k:
            x_mean[i] = x[: i + 1].mean()
        elif i >= n - k:
            x_mean[i] = x[i - k + 1 :].mean()
        else:
            x_mean[i] = x[i - k // 2 : i + k // 2 + 1].mean()

    return x_mean


def fft_signal_clean(
    signal: Sequence[float],
    step_interval: int,
    # filter_percent: float = 99.7,
    filter_upper_period: int | None,
    axis: int = -1,
) -> NDArray[np.float_]:
    """
    De-noising signal using Fast Fourier Transformation
    Adapted from: https://www.youtube.com/watch?v=s2K1JfNR7Sc

    All errors are mine.
    """

    # Get signal length
    n = len(signal)

    # Compute the FFT
    sp = np.fft.fft(signal, n, axis=axis)

    freq = np.fft.fftfreq(n, d=step_interval)[:, np.newaxis]
    abs_freq = np.abs(get_sliced(freq, start=1, axis=axis))

    # High-Pass Filter
    if filter_upper_period is not None:
        set_sliced(sp, get_sliced(sp, start=1, axis=axis) * (1 / abs_freq >= filter_upper_period), start=1, axis=axis)

    # # Compute Power Spectrum
    # psd = np.abs(sp) ** 2 / n
    # psd_filter = np.percentile(psd, filter_percent, axis=axis)
    #
    # sp *= psd >= psd_filter

    # Compute inverse FFT
    signal_clean = np.fft.ifft(sp, axis=axis)

    return signal_clean.real


def get_sliced(a, start, end=None, step=1, axis=0):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


def set_sliced(a, v, start, end=None, step=1, axis=0):
    a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)] = v

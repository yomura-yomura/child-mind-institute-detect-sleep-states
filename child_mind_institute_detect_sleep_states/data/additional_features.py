import numpy as np


def LPF_MAM(x: np.ndarray, times: np.ndarray, tau=12 * 60) -> np.ndarray:
    """back moving average"""
    k = np.round(tau / (times[1] - times[0])).astype(int)
    x_mean = np.zeros(x.shape)
    N = x.shape[0]
    for i in range(N):
        if i == 0:
            x_mean[i] = x[0]
        elif i - k < 0:
            x_mean[i] = x[:i].mean()
        else:
            x_mean[i] = x[i - k : i].mean()
    return x_mean


def CMA(x: np.ndarray, times: np.ndarray, tau=12 * 60) -> np.ndarray:
    """centroid moving average"""
    k = int(np.round(tau / (times[1] - times[0])))
    x_mean = np.zeros(x.shape)
    N = x.shape[0]

    for i in range(N):
        if i < k:
            x_mean[i] = x[: i + 1].mean()
        elif i >= N - k:
            x_mean[i] = x[i - k + 1 :].mean()
        else:
            x_mean[i] = x[i - k // 2 : i + k // 2 + 1].mean()

    return x_mean


def fft_signal_clean(
    signal,
    filter_percent=99.7,
):
    """
    Denoising signal using Fast Fourier Transformation
    Adapted from: https://www.youtube.com/watch?v=s2K1JfNR7Sc

    All errors are mine.
    """

    # Get signal length
    n = len(signal)

    # Compute the FFT
    fhat = np.fft.fft(signal, n)

    # Compute Power Spectrum
    PSD = fhat * np.conj(fhat) / n


    PSD_filter = np.percentile(PSD, filter_percent)


    indices = PSD >= PSD_filter

    fhat = fhat * indices

    # Compute inverse FFT
    signal_clean = np.fft.ifft(fhat)

    return signal_clean.real

import numpy as np
from jax import jit
from scipy.signal import convolve

from chromatix.ops.ops import fourier_convolution


def test_fourier_convolution():
    A = np.random.normal(size=(512, 512))
    B = np.random.normal(size=(256, 256))

    result_chromatix = jit(fourier_convolution)(A, B)
    result_scipy = convolve(A, B, method="fft", mode="same")
    assert np.allclose(result_scipy, result_chromatix, atol=1e-3), (
        "Fourier convolution not correct"
    )


def test_fourier_convolution_3d():
    A = np.random.normal(size=(5, 512, 512))
    B = np.random.normal(size=(3, 256, 256))

    result_chromatix = jit(lambda A, B: fourier_convolution(A, B, axes=(0, 1, 2)))(A, B)
    result_scipy = convolve(A, B, method="fft", mode="same")
    assert np.allclose(result_scipy, result_chromatix, atol=1e-3), (
        "Fourier convolution not correct"
    )


def test_no_fast_shape_fourier_convolution():
    A = np.random.normal(size=(513, 513))
    B = np.random.normal(size=(253, 253))

    result_chromatix = jit(
        lambda A, B: fourier_convolution(A, B, fast_fft_shape=False)
    )(A, B)
    result_scipy = convolve(A, B, method="fft", mode="same")
    assert np.allclose(result_scipy, result_chromatix, atol=1e-3), (
        "Fourier convolution not correct"
    )


def test_no_fast_shape_fourier_convolution_3d():
    A = np.random.normal(size=(5, 512, 512))
    B = np.random.normal(size=(3, 256, 256))

    result_chromatix = jit(
        lambda A, B: fourier_convolution(A, B, axes=(0, 1, 2), fast_fft_shape=False)
    )(A, B)
    result_scipy = convolve(A, B, method="fft", mode="same")
    assert np.allclose(result_scipy, result_chromatix, atol=1e-3), (
        "Fourier convolution not correct"
    )

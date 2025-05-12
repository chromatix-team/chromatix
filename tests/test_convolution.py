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

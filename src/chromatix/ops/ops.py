from chex import Array
import jax.numpy as jnp
from functools import partial
from jax import lax
from chromatix.utils import next_order


def fourier_convolution(
    image: Array, kernel: Array, *, axes=(0, 1), fast_fft_shape: bool = True
) -> Array:
    """
    Fourier convolution in 2D over the specified axes of an ``Array``.

    The default axes to perform 2D convolution over are (0, 1), or the first
    two axes of the input.

    This function computes the convolution ``kernel * image`` by employing the
    Fourier convolution theorem. The inputs are padded appropriately to avoid
    circular convolutions.

    By default, the inputs are further padded to the nearest power of 2 that
    is larger than the padded input shape for faster FFT performance. If the
    input shape causes the difference between padded and unpadded to be too
    large (causing either memory or performance issues), this extra padding can
    be disabled.

    Args:
        image: The input to be convolved.
        kernel: The convolution kernel.
        fast_fft_shape: Determines whether inputs should be further padded for
            increased FFT performance. Defaults to ``True``.
    """
    assert axes[1] == (axes[0] + 1), "Axes to convolve over must be contiguous"
    # Get padded shape to prevent circular convolution
    padded_shape = [
        k1 + k2 - 1
        for k1, k2 in zip(
            image.shape[axes[0] : axes[1] + 1], kernel.shape[axes[0] : axes[1] + 1]
        )
    ]
    if fast_fft_shape:
        fast_shape = [next_order(k) for k in padded_shape]
    else:
        fast_shape = padded_shape
    # Save memory with rfft if inputs are not complex
    is_complex = (image.dtype.kind == "c") or (kernel.dtype.kind == "c")
    if is_complex:
        fft = partial(jnp.fft.fft2, s=fast_shape, axes=axes)
        ifft = partial(jnp.fft.ifft2, s=fast_shape, axes=axes)
    else:
        fft = partial(jnp.fft.rfft2, s=fast_shape, axes=axes)
        ifft = partial(jnp.fft.irfft2, s=fast_shape, axes=axes)
    conv = ifft(fft(image) * fft(kernel))
    # Remove padding
    full_padded_shape = list(image.shape)
    for i, a in enumerate(axes):
        full_padded_shape[a] = padded_shape[i]
    conv = conv[tuple([slice(sz) for sz in full_padded_shape])]
    # Remove extra padding if any
    start_idx = [
        (k1 - k2) // 2 if idx in axes else 0
        for idx, (k1, k2) in enumerate(zip(conv.shape, image.shape))
    ]
    stop_idx = [
        k1 + k2 if idx in axes else k2
        for idx, (k1, k2) in enumerate(zip(start_idx, image.shape))
    ]
    conv_image = lax.slice(conv, start_idx, stop_idx)

    return conv_image

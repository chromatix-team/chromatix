from functools import partial

import jax.numpy as jnp
from jax import Array

from chromatix.typing import ArrayLike
from chromatix.utils import next_order


def fourier_convolution(
    image: ArrayLike,
    kernel: ArrayLike,
    *,
    axes: tuple[int, int] = (0, 1),
    fast_fft_shape: bool = True,
    mode: str = "same",
) -> Array:
    """
    Fourier convolution in n dimensions over the specified axes of an ``Array``.

    The convolution dimensions are determined by the `axes` argument. For
    example, the default axes to perform convolution over are (0, 1), or the
    first two axes of the input, which will perform a 2D convolution. The
    ``kernel`` and ``image`` should have the same number of dimensions.

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
        mode: A string that determines whether to crop the result of the
            convolution to the same shape as ``image``. Should be either
            ``"same"`` or ``"full"``. Defaults to ``"same"``.
    """
    for i in range(len(axes) - 1):
        assert axes[i + 1] == (axes[i] + 1), "Axes to convolve over must be contiguous"
    assert (
        image.ndim == kernel.ndim
    ), f"Input ({image.ndim}D) and kernel ({kernel.ndim}D) must have same number of dimensions"
    # Get padded shape to prevent circular convolution
    padded_shape = [
        k1 + k2
        for k1, k2 in zip(
            image.shape[axes[0] : axes[-1] + 1], kernel.shape[axes[0] : axes[-1] + 1]
        )
    ]
    if fast_fft_shape:
        fast_shape = [next_order(k) for k in padded_shape]
    else:
        fast_shape = padded_shape
    # Save memory with rfft if inputs are not complex
    is_complex = (image.dtype.kind == "c") or (kernel.dtype.kind == "c")
    # output_shape = image.shape[axes[0]:axes[-1] + 1] if mode == "same" else fast_shape
    if is_complex:
        fft = partial(jnp.fft.fftn, s=fast_shape, axes=axes)
        ifft = partial(jnp.fft.ifftn, s=fast_shape, axes=axes)
        # ifft = partial(jnp.fft.ifftn, s=output_shape, axes=axes)
    else:
        fft = partial(jnp.fft.rfftn, s=fast_shape, axes=axes)
        ifft = partial(jnp.fft.irfftn, s=fast_shape, axes=axes)
        # ifft = partial(jnp.fft.irfftn, s=output_shape, axes=axes)
    conv = ifft(fft(image) * fft(kernel))
    # Remove padding
    if mode == "same":
        conv = conv[
            tuple(
                [
                    slice((k - 1) // 2, (k - 1) // 2 + i) if idx in axes else slice(i)
                    for idx, (i, k) in enumerate(zip(image.shape, kernel.shape))
                ]
            )
        ]
    return conv

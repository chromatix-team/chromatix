from einops import reduce
from functools import partial
from typing import Tuple
from ..utils import next_order
from jax import lax
from chex import Array
import jax.numpy as jnp


def downsample(data: Array, window_size: Tuple[int, int], reduction="mean") -> Array:
    return reduce(
        data,
        "d (h h_size) (w w_size) c -> d h w c",
        reduction,
        h_size=window_size[0],
        w_size=window_size[1],
    )


def fourier_convolution(
    image: Array, kernel: Array, *, fast_fft_shape: bool = True
) -> Array:
    """Standard fourier convolution over first two axes of an object."""

    # Get padded shape to prevent circular convolution
    padded_shape = [k1 + k2 - 1 for k1, k2 in zip(image.shape[:2], kernel.shape[:2])]
    if fast_fft_shape:
        fast_shape = [next_order(k) for k in padded_shape]
    else:
        fast_shape = padded_shape

    # If real we can do with the real fourier transform
    is_complex = (image.dtype.kind == "c") or (kernel.dtype.kind == "c")
    if is_complex:
        fft = partial(jnp.fft.fft2, s=fast_shape, axes=[0, 1])
        ifft = partial(jnp.fft.ifft2, s=fast_shape, axes=[0, 1])
    else:
        fft = partial(jnp.fft.rfft2, s=fast_shape, axes=[0, 1])
        ifft = partial(jnp.fft.irfft2, s=fast_shape, axes=[0, 1])

    # Transform signals and gettin back to unpadded shape
    conv = ifft(fft(image) * fft(kernel))
    conv = conv[tuple([slice(sz) for sz in [*padded_shape, *image.shape[2:]]])]

    # Returning same mode
    start_idx = [
        (k1 - k2) // 2 if idx < 2 else 0
        for idx, (k1, k2) in enumerate(zip(conv.shape, image.shape))
    ]
    stop_idx = [
        k1 + k2 if idx < 2 else k2
        for idx, (k1, k2) in enumerate(zip(start_idx, image.shape))
    ]
    conv_image = lax.slice(conv, start_idx, stop_idx)

    return conv_image

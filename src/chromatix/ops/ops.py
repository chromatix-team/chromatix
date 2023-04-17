from einops import reduce
from functools import partial
from typing import Callable, Tuple, Union
from ..utils import next_order
from jax import lax
from jax.image import scale_and_translate
from chex import Array
import jax.numpy as jnp


def pooling_downsample(
    data: Array, window_size: Tuple[int, int], reduction: str = "mean"
) -> Array:
    """
    Wrapper for downsampling input of shape `(B... H W C P)` along `(H W)`.

    By default, downsampling is performed as a 2D average pooling. Also
    accepts various reduction functions that will be applied with the given
    ``window_size``, including `'max'`, `'min'`, `'sum'`, `'prod'`, and the
    default `'mean'`.

    Args:
        data: The data to be downsampled of shape `(B... H W C P)`.
        window_size: A tuple of 2 elements defining the window shape (height
            and width) for downsampling along `(H W)`.
        reduction: A string defining the reduction function applied with the
            given ``window_size``.
    """
    return reduce(
        data,
        "... (h h_size) (w w_size) c p -> ... h w c p",
        reduction,
        h_size=window_size[0],
        w_size=window_size[1],
    )


def init_plane_resample(
    out_shape: Tuple[int, ...],
    out_spacing: Union[float, Array],
    resampling_method: str = "linear",
) -> Callable[[Array, float], Array]:
    """
    Returns a function that resamples 2D planes to the specified output shape
    and spacing.

    The returned function is allowed to be jitted because the shape of the
    output will no longer depend on the input of this function.

    Multiple ``resampling_methods`` are supported: either `'pooling'` which
    uses sum pooling (for downsampling only) or any method supported by
    ``jax.image.scale_and_translate`` (`'linear'`, `'cubic'`, `'lanczos3'`,
    or `'lanczos5'`).

    The input may have any number of dimensions after the first two, but
    the returned function assumes that the 2D planes to be downsampled are
    contained in the first two axes. Any other dimensions are treated as batch
    dimensions, i.e. resampling is parallelized across those dimensions. In
    order to add arbitrary batch dimensions before the first two dimensions,
    use ``jax.vmap``.
    """
    assert len(out_shape) == 2, "Shape must be tuple of form (H W)"
    out_spacing = jnp.atleast_1d(out_spacing).squeeze()
    assert (
        out_spacing.size <= 2
    ), "Spacing is either a float or array of shape (2,) for non-square pixels"
    if resampling_method == "pool":

        def op(x: Array, in_spacing: Union[float, Array]) -> Array:
            return reduce(
                x,
                "(h hf) (w wf) ... -> h w ...",
                "sum",
                h=out_shape[0],
                w=out_shape[1],
            )

    else:

        def op(x: Array, in_spacing: Union[float, Array]) -> Array:
            in_spacing = jnp.atleast_1d(in_spacing).squeeze()
            assert (
                in_spacing.size <= 2
            ), "Spacing is either a float or array of shape (2,) for non-square pixels"
            _in_shape, _out_shape = jnp.array(x.shape[:-2]), jnp.array(out_shape)
            scale = in_spacing / out_spacing
            translation = -0.5 * (_in_shape * scale - _out_shape)
            total = x.sum(axis=(0, 1))
            # NOTE(dd): Because scale_and_translate expects shape to have same
            # number of dimensions as input, we have to extend the shape with
            # any channel/ vectorial dimensions here
            extended_shape = out_shape + x.shape[2:]
            x = scale_and_translate(
                x, extended_shape, (0, 1), scale, translation, method=resampling_method
            )
            x = x * (total / x.sum(axis=(0, 1)))
            return x

    return op


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

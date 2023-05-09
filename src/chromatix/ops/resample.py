from einops import reduce
from typing import Callable, Tuple, Union
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

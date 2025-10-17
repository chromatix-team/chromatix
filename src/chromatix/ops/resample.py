import equinox as eqx
import jax.numpy as jnp
import numpy as np
from einops import reduce
from jax.image import scale_and_translate
from jaxtyping import Array, Float, ScalarLike

from chromatix import Resampler


class PoolingPlaneDownsampler(Resampler):
    out_shape: tuple[int, int] = eqx.field(static=True)
    out_spacing: ScalarLike | Float[Array, "2"]

    def __init__(
        self, out_shape: tuple[int, int], out_spacing: ScalarLike | Float[Array, "2"]
    ):
        self.out_shape = out_shape
        self.out_spacing = out_spacing

    def __call__(
        self, resample_input: Float[Array, "h w ..."], in_spacing: Float[Array, "2"]
    ) -> Array:
        return reduce(
            resample_input,
            "(h hf) (w wf) ... -> h w ...",
            "sum",
            h=self.out_shape[0],
            w=self.out_shape[1],
        )


class InterpolatingPlaneResampler(Resampler):
    out_shape: tuple[int, int] = eqx.field(static=True)
    out_spacing: ScalarLike | Float[Array, "2"]
    resampling_method: str = eqx.field(static=True)

    def __init__(
        self,
        out_shape: tuple[int, int],
        out_spacing: ScalarLike | Float[Array, "2"],
        resampling_method: str = "linear",
    ):
        self.out_shape = out_shape
        self.out_spacing = out_spacing
        self.resampling_method = resampling_method

    def __call__(
        self, resample_input: Float[Array, "h w ..."], in_spacing: Float[Array, "2"]
    ) -> Array:
        in_spacing = jnp.atleast_1d(jnp.asarray(in_spacing).squeeze())
        assert in_spacing.size == 2, (
            "Input spacing is an array of shape (2,) representing pixel size in (y x)"
        )
        _in_shape, _out_shape = (
            jnp.asarray(resample_input.shape),
            jnp.asarray(self.out_shape),
        )
        scale = in_spacing / self.out_spacing
        translation = -0.5 * (_in_shape * scale - _out_shape)
        # NOTE(dd): Because scale_and_translate expects shape to have same
        # number of dimensions as input, we have to extend the shape with
        # any channel/ vectorial dimensions here
        # extended_shape = out_shape + x.shape
        resample_output = scale_and_translate(
            resample_input,
            self.out_shape,
            (0, 1),
            scale,
            translation,
            method=self.resampling_method,
        )
        resample_output = resample_output / jnp.prod(scale)
        return resample_output


def init_plane_resample(
    out_shape: tuple[int, ...],
    out_spacing: ScalarLike | Float[Array, "2"],
    resampling_method: str = "linear",
) -> Resampler:
    """
    Returns a function that resamples 2D planes to the specified output shape
    and spacing. These functions are instances of ``Resampler``s in Chromatix.

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
    assert np.atleast_1d(np.asarray(out_spacing).squeeze()).size <= 2, (
        "Spacing is either a float or array of shape (2,) for non-square pixels"
    )
    if resampling_method == "pool":
        return PoolingPlaneDownsampler(out_shape, out_spacing)
    else:
        return InterpolatingPlaneResampler(
            out_shape, out_spacing, resampling_method=resampling_method
        )

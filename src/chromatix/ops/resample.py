import equinox as eqx
import jax.numpy as jnp
import numpy as np
from einops import reduce
from jax.image import scale_and_translate
from jaxtyping import Array, Float, ScalarLike

from chromatix import Resampler

from ._banded_resample import KERNELS, banded_scale_and_translate


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
    banded: bool = eqx.field(static=True)

    def __init__(
        self,
        out_shape: tuple[int, int],
        out_spacing: ScalarLike | Float[Array, "2"],
        resampling_method: str = "linear",
        banded: bool = False,
    ):
        self.out_shape = out_shape
        self.out_spacing = out_spacing
        self.resampling_method = resampling_method
        self.banded = banded

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
        if self.banded:
            # Same interpolation, but gathering only the kernel's nonzero taps
            # instead of contracting a dense (in, out) weight matrix.
            resample_output = banded_scale_and_translate(
                resample_input,
                self.out_shape,
                scale,
                translation,
                method=self.resampling_method,
            )
        else:
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
    banded: bool = False,
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

    Args:
        out_shape: A tuple representing the number of samples (pixels) to
            which the incoming plane should be resampled in the format `(height
            width)`.
        out_spacing: Either a scalar or a 1D array of size 2 (in the format
            `(height width)`) representing the spacing between samples in units
            of distance. A scalar value represents square pixels, which is
            typically what you will want to use.
        resampling_method: A string representing the type of resampling
            method to initialize. Can be either ``"linear"``, ``"cubic"``,
            ``"lanczos3"``, or ``"lanczos5"`` for arbitrary interpolation,
            or ``"pooling"`` for a sum pooling downsampling. Defaults to
            ``"linear"``.
        banded: If ``True``, gather only the interpolation kernel's nonzero
            taps instead of contracting a dense ``(in, out)`` weight matrix.
            This is the same interpolation (results agree to float32
            accumulation noise) but asymptotically cheaper: O(out * taps) per
            axis instead of O(in * out).

            Defaults to ``False``, which preserves the exact previous
            behaviour. Whether it is faster depends on the kernel and the
            resampling ratio. Measured on an NVIDIA L4, forward + backward:

            - ``"linear"``: faster in every case measured, 1.5x (1024 -> 512)
              to 10.0x (4096 -> 4096).
            - ``"lanczos3"``: 1.5-6.0x for mild ratios, but **slower** at heavy
              downsampling (0.86x at 2048 -> 512, 0.84x at 4096 -> 512), where
              the wide forward band gives the banded backward poor arithmetic
              intensity while the dense path becomes a small, well-shaped GEMM.

            So prefer ``True`` for ``"linear"``, or for large planes at mild
            ratios; benchmark before enabling it for the wider kernels with
            aggressive downsampling. Ignored for ``"pool"``.
    Returns:
        A [``Resampler``](core.md#chromatix.core.base.Resampler), which is a
        callable that actually performs the resampling.
    """
    assert len(out_shape) == 2, "Shape must be tuple of form (height width)"
    assert np.atleast_1d(np.asarray(out_spacing).squeeze()).size <= 2, (
        "Spacing is either a float or array of shape (2,) for non-square pixels"
    )
    if resampling_method == "pool":
        return PoolingPlaneDownsampler(out_shape, out_spacing)
    else:
        return InterpolatingPlaneResampler(
            out_shape,
            out_spacing,
            resampling_method=resampling_method,
            banded=banded and resampling_method in KERNELS,
        )

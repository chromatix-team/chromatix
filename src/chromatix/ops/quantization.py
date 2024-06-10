import jax
import jax.numpy as jnp
from jax import Array

from chromatix.typing import ArrayLike

__all__ = ["binarize", "binarize_jvp", "quantize", "quantize_jvp"]


@jax.custom_jvp
def binarize(x: ArrayLike, threshold: float = 0.5) -> Array:
    """
    Binarize each pixel of amplitude mask to be either 0 or 1.

    Optionally takes a ``threshold`` to choose the threshold at which values
    are set to 1. Defaults to 0.5.

    The gradient of ``binarize`` is approximated with the gradient of the
    sigmoid function as in [1].

    [1] Eybposh, M. Hossein, et al. "Optimization of time-multiplexed
    computer-generated holograms with surrogate gradients." Emerging Digital
    Micromirror Device Based Systems and Applications XIV. SPIE, 2022.

    Args:
        x: Input to binarize to 0 or 1.
        threshold: Threshold above which values are set to 1. Defaults to 0.5.
    """
    return (x > threshold) * 1.0


@binarize.defjvp
def binarize_jvp(primals: tuple, tangents: tuple) -> tuple:
    """
    Custom gradient for ``binarize``.

    We approximate the gradient of ``binarize`` with the gradient of the
    sigmoid function.
    """
    sig = 1 / (1 + jnp.exp((-1 * primals[0]) + primals[1]))
    out_tangents = tangents[0] * (1 / 2.0) * (1 - sig**2 - (1 - sig) ** 2)
    return binarize(*primals), out_tangents


@jax.custom_jvp
def quantize(x: Array, bit_depth: float, range: tuple[int, int] | None = None) -> Array:
    """
    Quantize the input ``x`` to the specified ``bit_depth``. Surrogate gradient
    approach [1] is used to adjust the bit depth differentiably.

    [1] Eybposh, M. Hossein, et al. "Optimization of time-multiplexed
    computer-generated holograms with surrogate gradients." Emerging Digital
    Micromirror Device Based Systems and Applications XIV. SPIE, 2022.

    Args:
        x: Input to quantize.
        bit_depth: Number of bits. This parameter does NOT represent the number of
            digitization levels, but the bit depth.
        range: Range to quantize to, provided as ``(minimum, maximum)``. If not
            provided, the range of the values in ``x`` will be used.

    Returns:
        The quantized input.
    """
    if range is None:
        x_min = x.min()
        x_max = (x - x_min).max()
    else:
        x_min, x_max = range
    y = jnp.round(((x - x_min) / x_max) * ((2**bit_depth) - 1)).astype(jnp.float32)
    return (y / ((2**bit_depth) - 1.0)) * x_max + x_min


@quantize.defjvp
def quantize_jvp(primals: tuple, tangents: tuple) -> tuple:
    """
    Custom gradient for ``quantize``.

    We approximate the gradient of ``quantize`` with surrogate gradients, as
    described in [1] (see definition of ``quantize``).
    """
    return quantize(*primals), tangents[0]

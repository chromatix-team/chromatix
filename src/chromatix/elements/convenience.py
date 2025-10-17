import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike

from chromatix import Field

from ..ops import binarize, quantize

__all__ = ["Flip", "ScaleAndBias", "Binarize", "Quantize"]


class Flip(eqx.Module):
    """
    This element flips the incoming ``Field`` upside down.

    This is useful for situations where an upside down image is formed.
    """

    def __call__(self, field: Field) -> Field:
        # TODO: Add in support for both axes?
        return field.replace(u=jnp.flip(field.u, axis=field.spatial_dims[0]))


class ScaleAndBias(eqx.Module):
    """
    This element applies a ``scale`` and ``bias`` to the incoming ``Field``.

    The ``scale`` and ``bias`` can either be scalars or ``Array``s
    broadcastable to the shape of the incoming ``Field``.
    """

    bias: ScalarLike
    scale: ScalarLike

    def __init(self, bias: ScalarLike, scale: ScalarLike):
        self.bias = bias
        self.scale = scale

    def __call__(self, field: Field) -> Field:
        return (field + self.bias) * self.scale


class Binarize(eqx.Module):
    """
    This element binarizes the incoming ``Field``.

    See ``chromatix.ops.quantization.binarize`` for more details.
    """

    threshold: float

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, field: Field) -> Field:
        return field.replace(u=binarize(field.u, self.threshold))


class Quantize(eqx.Module):
    """
    This element quantizes the incoming ``Field`` to the given ``bit_depth``.

    See ``chromatix.ops.quantization.quantize`` for more details.
    """

    bit_depth: int = eqx.field(static=True)
    range: Float[Array, "2"] | tuple[int, int] | None = eqx.field(static=True)

    def __init__(
        self, bit_depth: int, range: Float[Array, "2"] | tuple[int, int] | None = None
    ):
        self.bit_depth = bit_depth
        self.range = range

    def __call__(self, field: Field) -> Field:
        return field.replace(u=quantize(field.u, self.bit_depth, self.range))

import flax.linen as nn
from ..field import Field
from chex import Array
from typing import Optional, Tuple, Union
from ..ops import binarize, quantize

__all__ = ["Flip", "ScaleAndBias", "Binarize", "Quantize"]


class Flip(nn.Module):
    """
    This element flips the incoming ``Field`` upside down.

    This is useful for situations where an upside down image is formed.
    """

    @nn.compact
    def __call__(self, field: Field) -> Field:
        # TODO: Add in support for both axes?
        return field.replace(u=field.u[:, ::-1, :, :])


class ScaleAndBias(nn.Module):
    """
    This element applies a ``scale`` and ``bias`` to the incoming ``Field``.

    The ``scale`` and ``bias`` can either be scalars or ``Array``s
    broadcastable to the shape of the incoming ``Field``.
    """

    bias: Union[float, Array]
    scale: Union[float, Array]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        return (field + self.bias) * self.scale


class Binarize(nn.Module):
    """
    This element binarizes the incoming ``Field``.

    See ``chromatix.ops.quantization.binarize`` for more details.
    """
    threshold: float = 0.5

    @nn.compact
    def __call__(self, field: Field) -> Field:
        return field.replace(u=binarize(field.u, self.threshold))


class Quantize(nn.Module):
    """
    This element quantizes the incoming ``Field`` to the given ``bit_depth``.

    See ``chromatix.ops.quantization.quantize`` for more details.
    """
    bit_depth: int
    range: Optional[Tuple[int, int]] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        return field.replace(u=quantize(field.u, self.bit_depth, self.range))

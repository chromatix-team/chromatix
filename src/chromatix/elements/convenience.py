import flax.linen as nn
from chromatix import Field
from chex import Array
from typing import Union

__all__ = ["Flip", "ScaleAndBias"]


class Flip(nn.Module):
    """
    This elements flips the incoming ``Field`` upside down.

    This is useful for situations where an upside down image is formed.
    """

    @nn.compact
    def __call__(self, field: Field) -> Field:
        # TODO: Add in support for both axes?
        return field.replace(u=field.u[:, ::-1, :, :])


class ScaleAndBias(nn.Module):
    bias: Array

    @nn.compact
    def __call__(self, field: Field, scale: Union[Array, Field]) -> Field:
        # TODO: Not sure why offset is fixed and scale an input?
        return (field + self.bias) * scale

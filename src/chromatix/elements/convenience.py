import jax.numpy as jnp
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
        return field.replace(u=jnp.flip(field.u, axis=field.spatial_dims[0]))


class ScaleAndBias(nn.Module):
    """
    This elements applies a ``scale`` and ``bias`` to the incoming ``Field``.

    The ``scale`` and ``bias`` can either be scalars or ``Array``s
    broadcastable to the shape of the incoming ``Field``.
    """

    bias: Union[float, Array]
    scale: Union[float, Array]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        return (field + self.bias) * self.scale

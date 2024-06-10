from typing import Callable

from chex import PRNGKey
from flax import linen as nn
from jax import Array
from jax.typing import ArrayLike

from chromatix.elements.utils import register
from chromatix.field import Field
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.ops import binarize

__all__ = ["AmplitudeMask"]


class AmplitudeMask(nn.Module):
    """
    Applies an ``amplitude`` mask to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The ``amplitude`` can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        amplitude: The amplitude to be applied. Should have shape `(H W)`.
        is_binary: binarize the amplitude mask if True.
    """

    amplitude: ArrayLike | Callable[[PRNGKey, tuple[int, int]], Array]
    is_binary: bool

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``amplitude`` mask to incoming ``Field``."""
        amplitude = register(self, "amplitude", field.spatial_shape)
        if self.is_binary:
            amplitude = binarize(amplitude)
        return amplitude_change(field, amplitude)

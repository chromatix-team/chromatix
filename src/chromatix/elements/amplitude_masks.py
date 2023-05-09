from typing import Callable, Union, Tuple
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank

from ..field import Field
from ..functional.amplitude_masks import amplitude_change
from ..ops import binarize
from chromatix.elements.utils import register

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

    amplitude: Union[Array, Callable[[PRNGKey, Tuple[int, int]], Array]]
    is_binary: bool

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``amplitude`` mask to incoming ``Field``."""
        amplitude = register(self, "amplitude", field.spatial_shape)
        if self.is_binary:
            amplitude = binarize(amplitude)
        return amplitude_change(field, amplitude)

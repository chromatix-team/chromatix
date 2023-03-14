from typing import Callable, Union, Tuple
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank

from ..field import Field
from ..functional.amplitude_masks import amplitude_change
from ..ops.binarize import binarize
from ..utils import _broadcast_2d_to_spatial

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
        amplitude = (
            self.param("amplitude_pixels", self.amplitude, field.spatial_shape)
            if callable(self.amplitude)
            else self.amplitude
        )

        assert_rank(
            amplitude, 2, custom_message="Amplitude must be array of shape (H W)"
        )
        if self.is_binary:
            amplitude = binarize(amplitude)
        amplitude = _broadcast_2d_to_spatial(amplitude, field.ndim)
        return amplitude_change(field, amplitude)

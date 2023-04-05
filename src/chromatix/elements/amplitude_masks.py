from typing import Callable, Union, Tuple
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank

from ..field import Field
from ..functional.amplitude_masks import amplitude_change
from ..ops import binarize

__all__ = ["AmplitudeMask"]


class AmplitudeMask(nn.Module):
    """
    Applies an ``amplitude`` mask to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The ``amplitude`` can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        amplitude: The amplitude to be applied. Should have shape `[1 H W 1]`.
        is_binary: binarize the amplitude mask if True.
    """

    amplitude: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]
    is_binary: bool

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``amplitude`` mask to incoming ``Field``."""
        amplitude = (
            self.param("amplitude_pixels", self.amplitude, (1, *field.shape[1:3], 1))
            if callable(self.amplitude)
            else self.amplitude
        )

        assert_rank(
            amplitude, 4, custom_message="Amplitude must be array of shape [1 H W 1]"
        )
        if self.is_binary:
            amplitude = binarize(amplitude)
        return amplitude_change(field, amplitude)

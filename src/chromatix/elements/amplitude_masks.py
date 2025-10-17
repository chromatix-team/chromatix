import equinox as eqx
from jaxtyping import Array, Float

from chromatix import Field
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.ops import binarize

__all__ = ["AmplitudeMask"]


class AmplitudeMask(eqx.Module):
    """
    Perturbs ``field`` by ``amplitude``, i.e. ``field * amplitude``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Attributes:
        amplitude: The amplitude mask to apply as a 2D array of amplitude values
            of shape ``(height width)``.
        is_binary: Binarize (make 0 or 1) the amplitude mask if ``True``. Note
            that if this is ``False``, amplitude values do not need to be in
            any range (i.e. they are allowed to be more than 1 which would add
            energy to the ``Field``).
    """

    amplitude: Float[Array, "h w"]
    is_binary: bool = eqx.field(static=True)

    def __init__(self, amplitude: Float[Array, "h w"], is_binary: bool):
        self.amplitude = amplitude
        self.is_binary = is_binary

    def __call__(self, field: Field) -> Field:
        """Applies ``amplitude`` mask to incoming ``Field``."""
        if self.is_binary:
            amplitude = binarize(self.amplitude)
        else:
            amplitude = self.amplitude
        return amplitude_change(field, amplitude)

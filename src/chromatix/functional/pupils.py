from chromatix.typing import ScalarLike

from ..field import Field
from ..utils import l2_sq_norm, linf_norm

__all__ = ["circular_pupil", "square_pupil"]


def circular_pupil(field: Field, w: ScalarLike) -> Field:
    """Applies circular pupil with diameter ``w`` to ``field``."""
    #
    normed_radius = (field.spectrum[..., 0] / field.spectrum) * w / 2
    mask = l2_sq_norm(field.grid(), axis=-1) <= normed_radius**2
    return field * mask[..., None]


def square_pupil(field: Field, w: ScalarLike) -> Field:
    """Applies square pupil with side length ``w`` to ``field``."""
    normed_radius = (field.spectrum[..., 0] / field.spectrum) * w / 2
    mask = linf_norm(field.grid(), axis=-1) <= normed_radius
    return field * mask[..., None]

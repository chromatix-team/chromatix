from ..field import Field
from ..utils import linf_norm, l2_sq_norm

__all__ = ["circular_pupil", "square_pupil"]


def circular_pupil(field: Field, w: float) -> Field:
    """Applies circular pupil with diameter ``w`` to ``field``."""
    mask = (
        l2_sq_norm(field.grid)
        <= ((field.spectrum[..., 0, 0] / field.spectrum) * w / 2) ** 2
    )
    return field * mask


def square_pupil(field: Field, w: float) -> Field:
    """Applies square pupil with side length ``w`` to ``field``."""
    mask = linf_norm(field.grid) <= (field.spectrum[..., 0, 0] / field.spectrum) * w / 2
    return field * mask

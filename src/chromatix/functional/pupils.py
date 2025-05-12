import jax.numpy as jnp
from jax import Array

from ..field import Field
from ..utils import l2_sq_norm, linf_norm

__all__ = ["circular_pupil", "square_pupil", "gaussian_pupil"]


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


def gaussian_pupil(field: Field, w: float, offset: Array | None = None) -> Field:
    """Applies a Gaussian pupil of waist w to the field."""
    grid = field.grid
    if offset is not None:
        grid = grid - offset
    envelope = jnp.exp(-l2_sq_norm(grid) / w**2)
    return field * envelope

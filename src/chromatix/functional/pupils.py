import jax.numpy as jnp

from chromatix.field import Field
from chromatix.typing import ScalarLike
from chromatix.utils import l2_norm, l2_sq_norm, linf_norm

__all__ = ["circular_pupil", "square_pupil", "super_gaussian_pupil", "tukey_pupil"]


def circular_pupil(field: Field, w: ScalarLike) -> Field:
    """Applies circular pupil with diameter ``w`` to ``field``."""
    mask = (
        l2_sq_norm(field.grid)
        <= ((field.spectrum[..., 0, 0] / field.spectrum) * w / 2) ** 2
    )
    return field * mask


def square_pupil(field: Field, w: ScalarLike) -> Field:
    """Applies square pupil with side length ``w`` to ``field``."""
    mask = linf_norm(field.grid) <= (field.spectrum[..., 0, 0] / field.spectrum) * w / 2
    return field * mask


def super_gaussian_pupil(field: Field, w: ScalarLike, n: ScalarLike = 16) -> Field:
    mask = jnp.exp(-((l2_norm(field.grid) / w) ** n))
    return field * mask


def tukey_pupil(field: Field, w: ScalarLike) -> Field:
    alpha = w / linf_norm(field.extent)
    grid = jnp.clip(l2_norm(field.grid / field.extent), 0, 0.5)
    mask = jnp.where(
        grid <= (alpha / 2),
        1,
        0.5 * (1 + jnp.cos((2 * jnp.pi / jnp.array(1 - alpha)) * (grid - alpha / 2))),
    )
    return field * mask

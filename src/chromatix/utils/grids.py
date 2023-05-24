import jax.numpy as jnp
from chex import Array
from einops import rearrange
from typing import  Tuple, Union

def create_grid(shape: Tuple[int, int], spacing: Union[float, Array]) -> Array:
    """
    Args:
        shape: The shape of the grid, described as a tuple of
            integers of the form (H W).
        spacing: The spacing of each pixel in the grid, either a single float
            for square pixels or an array of shape `(2 1)` for non-square
            pixels.
    """
    half_size = jnp.array(shape) / 2
    spacing = jnp.atleast_1d(spacing)
    if spacing.size == 1:
        spacing = jnp.concatenate([spacing, spacing])
    assert spacing.size == 2, "Spacing must be either single float or have shape (2,)"
    spacing = rearrange(spacing, "d -> d 1 1", d=2)
    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    grid = jnp.meshgrid(
        jnp.linspace(-half_size[0], half_size[0] - 1, num=shape[0]) + 0.5,
        jnp.linspace(-half_size[1], half_size[1] - 1, num=shape[1]) + 0.5,
        indexing="ij",
    )
    grid = spacing * jnp.array(grid)
    return grid


def grid_spatial_to_pupil(grid: Array, f: float, NA: float, n: float) -> Array:
    R = f * NA / n  # pupil radius
    return grid / R


def l2_sq_norm(a: Array, axis: Union[int, Tuple[int, ...]] = 0) -> Array:
    """Sum of squares, i.e. `x**2 + y**2`."""
    return jnp.sum(a**2, axis=axis)


def l2_norm(a: Array, axis: Union[int, Tuple[int, ...]] = 0) -> Array:
    """Square root of ``l2_sq_norm``, i.e. `sqrt(x**2 + y**2)`."""
    return jnp.sqrt(jnp.sum(a**2, axis=axis))


def l1_norm(a: Array, axis: Union[int, Tuple[int, ...]] = 0) -> Array:
    """Sum absolute value, i.e. `|x| + |y|`."""
    return jnp.sum(jnp.abs(a), axis=axis)


def linf_norm(a: Array, axis: Union[int, Tuple[int, ...]] = 0) -> Array:
    """Max absolute value, i.e. `max(|x|, |y|)`."""
    return jnp.max(jnp.abs(a), axis=axis)
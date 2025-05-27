from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array
from scipy.ndimage import distance_transform_edt

from chromatix.typing import ArrayLike, ScalarLike

from .shapes import _broadcast_2d_to_spatial


def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def center_pad(u: ArrayLike, padding: Sequence[int], cval: float = 0) -> Array:
    """
    Symmetrically pads ``u`` with lengths specified per axis in ``padding``,
    which should be an iterable of integers and have the same length as
    ``u.ndims``.
    """
    pad = [(n, n) for n in padding]
    return jnp.pad(u, pad, constant_values=cval)


def center_crop(u: Array, crop_length: Sequence[int]) -> Array:
    """
    Symmetrically crops ``u`` with lengths specified per axis in
    ``crop_length``, which should be iterable with same size as ``u.ndims``.
    """
    crop_length = [0 if length is None else length for length in crop_length]
    crop = tuple([slice(n, size - n) for size, n in zip(u.shape, crop_length)])
    return u[crop]


def gaussian_kernel(
    sigma: Sequence[float], truncate: float = 4.0, shape: Sequence[int] | None = None
) -> Array:
    """
    Creates ND Gaussian kernel of given ``sigma``.

    If ``shape`` is not provided, then the shape of the kernel is automatically
    calculated using the given truncation (the same truncation for each
    dimension) and ``sigma``. The number of dimensions is determined by the
    length of ``sigma``, which should be a 1D array.

    If ``shape`` is provided, then ``truncate`` is ignored and the result will
    have the provided ``shape``. The provided ``shape`` must be odd in all
    dimensions to ensure that there is a center pixel.

    Args:
        sigma: A 1D array whose length is the number of dimensions specifying
            the standard deviation of the Gaussian distribution in each
            dimension.
        truncate: If ``shape`` is not provided, then this float is the number
            of standard deviations for which to calculate the Gaussian. This is
            then used to determine the shape of the kernel in each dimension.
        shape: If provided, determines the ``shape`` of the kernel. This will
            cause ``truncate`` to be ignored.

    Returns:
        The ND Gaussian kernel.
    """
    _sigma = np.atleast_1d(np.array(sigma))
    if shape is not None:
        _shape = np.atleast_1d(np.array(shape))
        assert np.all(_shape % 2 != 0), "Shape must be odd in all dimensions"
        radius = ((_shape - 1) / 2).astype(np.int16)
    else:
        radius = (truncate * _sigma + 0.5).astype(np.int16)

    x = jnp.mgrid[tuple(slice(-r, r + 1) for r in radius)]
    phi = jnp.exp(-0.5 * jnp.sum((x.T / _sigma) ** 2, axis=-1))  # type: ignore
    return phi / phi.sum()


def sigmoid_taper(shape: tuple[int, int], width: float, ndim: int = 5) -> Array:
    dist = distance_transform_edt(np.pad(np.ones((shape[0] - 2, shape[1] - 2)), 1))
    taper = 2 * (nn.sigmoid(dist / width) - 0.5)  # type: ignore - it's an array!
    return _broadcast_2d_to_spatial(taper, ndim)


def create_grid(shape: tuple[int, int], spacing: ScalarLike) -> Array:
    """
    Creates a 2D grid of vertical and horizontal coordinates with the specified
    ``shape`` and ``spacing``, with the origin in the center of the grid.

    Args:
        shape: The shape of the grid, described as a tuple of
            integers of the form ``(H W)``.
        spacing: The spacing of each pixel in the grid, either a single float
            for square pixels or an array of shape `(2 1)` for non-square
            pixels.
    Returns:
        The grid as an array of shape ``(2 H W)``.
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


def rotate_grid(grid: Array, rotation: ScalarLike) -> Array:
    """
    Rotates a 2D grid (an array of shape ``(2 H W)``) by ``rotation`` radians.
    Positive rotations are assumed to be in the counter-clockwise direction.
    """
    rotation = jnp.array(
        [
            [jnp.cos(rotation), -jnp.sin(rotation)],
            [jnp.sin(rotation), jnp.cos(rotation)],
        ]
    )
    grid = jnp.einsum("ij, ihw -> jhw", rotation, grid)
    return grid


def grid_spatial_to_pupil(
    grid: Array, f: ScalarLike, NA: ScalarLike, n: ScalarLike
) -> Array:
    R = f * NA / n  # pupil radius
    return grid / R


def l2_sq_norm(a: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Sum of squares, i.e. `x**2 + y**2`."""
    return jnp.sum(a**2, axis=axis)


def l2_norm(a: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Square root of ``l2_sq_norm``, i.e. `sqrt(x**2 + y**2)`."""
    return jnp.sqrt(jnp.sum(a**2, axis=axis))


def l1_norm(a: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Sum absolute value, i.e. `|x| + |y|`."""
    return jnp.sum(jnp.abs(a), axis=axis)


def linf_norm(a: Array, axis: int | tuple[int, ...] = 0) -> Array:
    """Max absolute value, i.e. `max(|x|, |y|)`."""
    return jnp.max(jnp.abs(a), axis=axis)


def matvec(x: Array, y: Array) -> Array:
    """Implements batched matrix - vector multiplication.
    Mostly used in polarization calculations.
    Example [..., N, M] x [...., M] -> [...., N]"""
    return jnp.matmul(x, y[..., None]).squeeze(-1)


def outer(x: Array, y: Array, in_axis: int = -1) -> Array:
    """Calculates batched outer product (Numpy flattens input matrices)
    Includes additional in_axis for which axis to use.
    Output axes will always be last two.
    """
    _x = jnp.moveaxis(x, in_axis, -1)
    _y = jnp.moveaxis(y, in_axis, -1)
    return _x[..., None, :] * _y[..., :, None]

# This is taken from differentiable_tomography and needs to be properly integrated; specially centered or edge pixel coordinates.
import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

__all__ = ["rotate_volume"]


def rotate_volume(volume: Array, angle: float, scale: float) -> Array:
    """Rotates a volume around the y axis (axis 1).
    angle in radians."""
    transform = jnp.matmul(S(scale), Ry(angle, volume.dtype))
    rotated_grid = volume_homogeneous_grid(volume) @ transform.T
    return resample(volume, rotated_grid[..., :3])


def Ry(theta: float, dtype=jnp.float32) -> jnp.ndarray:
    """Generates rotation matrix around y.
    Theta in radians."""
    R = jnp.zeros((4, 4), dtype=dtype)
    sin_t = jnp.sin(theta).astype(dtype)
    cos_t = jnp.cos(theta).astype(dtype)

    R = R.at[1, 1].set(1.0)
    R = R.at[3, 3].set(1.0)  # homogeneous
    R = R.at[0, 0].set(cos_t)
    R = R.at[2, 2].set(cos_t)
    R = R.at[0, 2].set(sin_t)
    R = R.at[2, 0].set(-sin_t)

    return R


def S(scale_factor: float) -> Array:
    """
    Generate a 4x4 homogeneous scaling matrix with uniform scaling.

    Args:
        scale_factor: A single float scaling factor to apply to all dimensions
        dtype: Data type of the matrix

    Returns:
        A 4x4 scaling matrix
    """
    S = jnp.eye(4)
    S = S.at[1, 1].set(1 / scale_factor)  # Scale y
    S = S.at[2, 2].set(1 / scale_factor)  # Scale x
    return S


def volume_homogeneous_grid(volume: Array) -> Array:
    """Given a volume, generates a centred grid of homogeneous coordinates.
    Coordinates are placed along last dimension [z, y, x, 4]"""

    Nz, Ny, Nx = volume.shape
    z = jnp.linspace(-(Nz - 1) / 2, (Nz - 1) / 2, Nz, dtype=volume.dtype)
    y = jnp.linspace(-(Ny - 1) / 2, (Ny - 1) / 2, Ny, dtype=volume.dtype)
    x = jnp.linspace(-(Nx - 1) / 2, (Nx - 1) / 2, Nx, dtype=volume.dtype)
    grid = jnp.stack(jnp.meshgrid(z, y, x, indexing="ij"), axis=-1)
    return jnp.concatenate(
        [grid, jnp.ones((Nz, Ny, Nx, 1), dtype=volume.dtype)], axis=-1
    )


def resample(volume: Array, sample_grid: Array) -> Array:
    """Resample volume on coordinates given by grid.
    Assumes original coordinates were centered, i.e. -N/2 -> N/2"""
    offset = (jnp.array(volume.shape) - 1) / 2
    sample_locations = sample_grid.reshape(-1, 3).T + offset[:, None]
    resampled = map_coordinates(
        volume, list(sample_locations), order=1, mode="constant", cval=0.0
    )
    return resampled.reshape(sample_grid.shape[:3])

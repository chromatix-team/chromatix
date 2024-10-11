from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

eps = jnp.finfo(jnp.float32).eps


def outer(x: ArrayLike, y: ArrayLike, in_axis: int = -1) -> Array:
    """Calculates batched outer product (Numpy flattens input matrices)
    Includes additional in_axis for which axis to use.
    Output axes will always be last two.
    """
    _x = jnp.moveaxis(x, in_axis, -1)
    _y = jnp.moveaxis(y, in_axis, -1)
    return _x[..., None, :] * _y[..., :, None]


def PTFT(k_grid: ArrayLike, km: ArrayLike, eps: float = eps) -> Array:
    """
    Calculates Polarisation Transfer Function Tensor (PTFT) as per eq. 7
    in Multislice computational model for birefringent scattering. Returns
    0 for evanescent waves.
    km: background value
    k_grid: shape [k, z, y, x], with k [kz, ky, kx]
    eps: used for numerical issues"""
    Q = -outer(k_grid / km, k_grid / km, in_axis=0) + jnp.eye(3)
    mask = jnp.greater(jnp.sum(k_grid**2, axis=0), (km + eps) ** 2)
    return jnp.where(mask[..., None, None], 0.0, Q)  # Remove evanescent waves

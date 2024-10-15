from jax import Array
from jax import numpy as jnp
from jax.lax import scan
from jax.typing import ArrayLike

import chromatix.functional as cf
from chromatix.utils.fft import fft, ifft

eps = jnp.finfo(jnp.float32).eps


def outer(x: ArrayLike, y: ArrayLike, in_axis: int = -1) -> Array:
    """Calculates batched outer product (Numpy flattens input matrices)
    Includes additional in_axis for which axis to use.
    Output axes will always be last two.
    """
    _x = jnp.moveaxis(x, in_axis, -1)
    _y = jnp.moveaxis(y, in_axis, -1)
    return _x[..., None, :] * _y[..., :, None]

def matvec(a: ArrayLike, b: ArrayLike) -> Array:
    return jnp.matmul(a, b[..., None]).squeeze(-1)


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



def thick_polarised_sample(field: cf.VectorField, potential: ArrayLike, nm: ArrayLike, dz: ArrayLike) -> cf.VectorField:
    def Q_op(u: Array) -> Array:
        # correct
        """Polarisation transfer operator"""
        return ifft(matvec(Q, fft(u)))
    
    def H_op(u: Array) -> Array:
        # correct
        """Vectorial scattering operator"""
        prefactor = -1j * dz / 2 * jnp.exp(1j * kz * dz) / kz
        prefactor = jnp.where(kz > 0, prefactor, 0)
        return ifft(matvec(Q, prefactor * fft(u)))
    
    def P_op(u: Array) -> Array:
        # correct
        """Vectorial free space operator"""
        # NOTE: Really need to check if we deal correctly with evanescent waves
        prefactor = jnp.where(kz > 0,  jnp.exp(1j * kz * dz), 0)

        # NOTE: Are we not overlapping stuff by not not padding?
        return ifft(matvec(Q, prefactor * fft(u)))

    def propagate_slice(u: Array, potential_slice: Array) -> tuple[Array, None]:
        scatter_field = matvec(potential_slice, Q_op(u))
        return P_op(u) + H_op(scatter_field), None

    # Preliminaries
    # We shift the k_grid so it aligns with unshifted fft output
    k_grid = jnp.fft.ifftshift(field.k_grid, axes=field.spatial_dims)

    # We use z yx ordering, and add the 2pi factor to chromatix kgrrid
    # We chop off evanescent waves
    # NOTE: understand why we need nm here
    km = 2 * jnp.pi * nm / field.spectrum
    k_grid = 2 * jnp.pi * k_grid
    kz = jnp.sqrt(jnp.maximum(0.0, km**2 - jnp.sum(k_grid**2, axis=0))) 
    k_grid =  jnp.concatenate([kz[None, ...], k_grid], axis=0)

    # Calculating PTFT
    Q = PTFT(k_grid, km).squeeze(-3)

    # Running scan over sample
    u, _ = scan(propagate_slice, field.u, potential[..., None, :, :])
    return field.replace(u=u)
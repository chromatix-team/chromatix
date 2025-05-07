import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.lax import scan
from jax.lax import fori_loop
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


def thick_polarised_sample(
    field: cf.VectorField,
    potential: ArrayLike,
    n_background: ArrayLike,
    dz: ArrayLike,
    NA: float = 1.0,
) -> cf.VectorField:
    def Q_op(u: Array) -> Array:
        # correct
        """Polarisation transfer operator"""
        return crop(ifft(matvec(Q, fft(pad(u)))))

    def H_op(u: Array) -> Array:
        # correct
        """Vectorial scattering operator"""
        prefactor = jnp.where(kz > 0, -1j / 2 * jnp.exp(1j * kz * dz) / kz * dz, 0)
        return crop(ifft(matvec(Q, prefactor * fft(pad(u)))))

    def P_op(u: Array) -> Array:
        """Vectorial free space operator"""
        prefactor = jnp.where(kz > 0, jnp.exp(1j * kz * dz), 0)
        return crop(ifft(matvec(Q, prefactor * fft(pad(u)))))

    def propagate_slice(u: Array, potential_slice: Array) -> tuple[Array, Array]:
        scatter_field = matvec(potential_slice, Q_op(u))
        new_field = P_op(u) + H_op(scatter_field)

        return new_field, new_field

    def pad(u):
        return jnp.pad(u, padding)

    def crop(u):
        return u[:, : field.spatial_shape[0], : field.spatial_shape[1]]

    # Padding for circular convolution
    padded_shape = 2 * np.array(field.spatial_shape)
    n_pad = padded_shape - np.array(field.spatial_shape)
    padding = ((0, 0), (0, n_pad[0]), (0, n_pad[1]), (0, 0), (0, 0))

    # Getting k_grid
    k_grid = (
        2
        * jnp.pi
        * jnp.stack(
            jnp.meshgrid(
                jnp.fft.fftfreq(n=padded_shape[0], d=field.dx.squeeze()[1]),
                jnp.fft.fftfreq(n=padded_shape[1], d=field.dx.squeeze()[0]),
                indexing="ij",
            )
        )[:, None, ..., None, None]
    )
    km = 2 * jnp.pi * n_background / field.spectrum
    kz = jnp.sqrt(
        jnp.maximum(0.0, km**2 - jnp.sum(k_grid**2, axis=0))
    )  # chop off evanescent waves
    k_grid = jnp.concatenate([kz[None, ...], k_grid], axis=0)

    # Getting PTFT and band limiting
    Q = (-outer(k_grid / km, k_grid / km, in_axis=0) + jnp.eye(3)).squeeze(-3)
    Q = jnp.where(
        jnp.sum(k_grid[1:, ..., None] ** 2, axis=0) <= NA**2 * km**2, Q, 0
    )  # Limit NA

    # Use lax.fori_loop instead of scan to avoid accumulating intermediates.
    def body_fun(i, u):
        new_field, _ = propagate_slice(u, potential[i])
        return new_field

    num_slices = potential.shape[0]
    u_final = fori_loop(0, num_slices, body_fun, field.u)
    return field.replace(u=u_final)

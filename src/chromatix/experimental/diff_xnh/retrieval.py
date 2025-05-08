import jax
import jax.numpy as jnp
from jaxtyping import Array


def multi_paganin(
    data: Array,
    distances: Array,
    wavelength: float,
    voxelsize: float,
    delta_beta: float,
    eps: float = 1e-7,
) -> tuple[Array, Array]:
    """
    Phase retrieval based on the MultiPaganin method using JAX.

    Parameters
    ----------
    data : ndarray, float32
        Input data for several distances, shape (ndist, n, n)
    distances : ndarray
        Distances in meters, shape (ndist,)
    wavelength : float
        Wavelength in meters
    voxelsize : float
        Voxel size in meters
    delta_beta : float
        Ratio between the real and imaginary parts of the refractive index
    eps : float = 1e-7
        Constant to avoid division by zero

    Returns
    -------
    phase : ndarray
        Recovered phase of the object, shape (ntheta, n, n)
    """

    def update(
        prev: tuple[Array, Array], data_per_dist: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        numerator, denominator = prev
        measurement, dist = data_per_dist

        rad_freq = jnp.fft.fft2(measurement)
        taylor_exp = 1.0 + wavelength * dist * jnp.pi * delta_beta * f_sq_grid

        return (numerator + taylor_exp * rad_freq, denominator + taylor_exp**2), None

    # Setup frequencies
    n = data.shape[-1]
    freqs = jnp.fft.fftfreq(n, d=voxelsize)
    f_sq_grid = jnp.sum(
        jnp.stack(jnp.meshgrid(freqs, freqs, indexing="xy")) ** 2, axis=0
    )

    (numerator, denominator), _ = jax.lax.scan(
        update,
        (
            jnp.zeros((n, n), dtype=jnp.complex64),
            jnp.full((n, n), eps, dtype=jnp.complex64),
        ),
        (data, distances),
    )

    dphi = (
        -1 / 2 * delta_beta * jnp.log(jnp.real(jnp.fft.ifft2(numerator / denominator)))
    )

    return dphi + 1j * (dphi - dphi.min()) / delta_beta

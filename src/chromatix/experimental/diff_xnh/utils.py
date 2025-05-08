import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def shift_matrix(n: int) -> Array:
    """Creates a shift matrix to get a centered Fourier transform."""
    s = 1 - 2 * (jnp.arange(1, n + 1) % 2)  # [1, -1, 1, -1, ...]
    return jnp.outer(s, s)


def make_gaussian_window(ne: int, eps: float = 1e-3) -> tuple[Array, float, int]:
    """Creates a gaussian window for the spectral smoothing."""
    mu = -np.log(eps) / (2 * ne**2)
    m = int(np.ceil(2 * ne / np.pi * np.sqrt(-mu * np.log(eps) + (mu * ne) ** 2 / 4)))

    t = jnp.linspace(-1 / 2, 1 / 2, ne, endpoint=False)
    dx = jnp.stack(jnp.meshgrid(t, t))
    return jnp.exp(mu * ne**2 * jnp.sum(dx**2, axis=0)) * (1 - ne % 4), mu, m


import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def make_gaussian_window(ne: int, eps: float = 1e-3) -> tuple[Array, float, int]:
    """Creates a gaussian window for the spectral smoothing."""
    mu = -np.log(eps) / (2 * ne**2)
    m = int(np.ceil(2 * ne / np.pi * np.sqrt(-mu * np.log(eps) + (mu * ne) ** 2 / 4)))

    t = jnp.linspace(-1 / 2, 1 / 2, ne, endpoint=False)
    dx = jnp.stack(jnp.meshgrid(t, t))
    return jnp.exp(mu * ne**2 * jnp.sum(dx**2, axis=0)) * (1 - ne % 4), mu, m


def gather_kernel(f: Array, theta: float, m: int, mu: float, n: int) -> Array:
    """Gathers the rotated image from the Fourier transform."""
    # Grid indices
    nz = f.shape[0]
    X = jnp.arange(n)

    # Physical coordinates
    x0 = (X - n / 2) / n * jnp.cos(theta)
    y0 = -(X - n / 2) / n * jnp.sin(theta)

    # things for in the loop
    ell1_base = jnp.floor(2 * n * y0) - m
    ell0_base = jnp.floor(2 * n * x0) - m
    coeff0 = jnp.pi / mu
    coeff1 = -(jnp.pi**2) / mu

    def accumulate_kernel(k: int, g_accum: Array) -> Array:
        i0, i1 = jnp.unravel_index(k, (2 * m + 1, 2 * m + 1))

        ell1 = ell1_base + i1
        ell0 = ell0_base + i0

        w0 = ell0 / (2 * n) - x0
        w1 = ell1 / (2 * n) - y0
        w = coeff0 * jnp.exp(coeff1 * (w0**2 + w1**2))

        idx_x = (n + m + ell0).astype(jnp.int32)
        idx_y = (n + m + ell1).astype(jnp.int32)

        f_val = f[:, idx_y, idx_x]
        return g_accum + (w[None, :] * f_val) / n

    return jax.lax.fori_loop(
        0,
        (2 * m + 1) ** 2,
        accumulate_kernel,
        jnp.zeros((nz, n), dtype=f.dtype),
    )


def shift_matrix(n: int) -> Array:
    """Creates a shift matrix to get a centered Fourier transform."""
    s = 1 - 2 * (jnp.arange(1, n + 1) % 2)  # [1, -1, 1, -1, ...]
    return jnp.outer(s, s)


def radon(x: Array, theta: float) -> Array:
    """Calculates the radon transform"""

    # What axes do we rotate around? Seems like the last two,
    n = x.shape[-1]

    # Window and pad
    window, mu, m = make_gaussian_window(n)
    x = x * window
    x = jnp.pad(x, ((0, 0), (int(n // 2), int(n // 2)), (int(n // 2), int(n // 2))))

    # Centered FFT
    s = shift_matrix(x.shape[-1])
    x_hat = s * jnp.fft.fft2(x * s)  # [ne, ne]

    # Pad and gather kernel
    x_hat = jnp.pad(x_hat, ((0, 0), (m, m), (m, m)), mode="wrap")
    sino = gather_kernel(x_hat, theta, m, mu, n)

    # STEP3: ifft 1d
    s = 1 - 2 * (jnp.arange(1, sino.shape[-1] + 1) % 2)
    sino = jnp.fft.ifft(s * sino) * s

    # STEP4: Shift based on the rotation axis
    t = jnp.fft.fftfreq(n)
    w = jnp.exp(-2 * jnp.pi * 1j * t * n)
    sino = jnp.fft.ifft(w * jnp.fft.fft(sino))
    return sino / (4 * n)

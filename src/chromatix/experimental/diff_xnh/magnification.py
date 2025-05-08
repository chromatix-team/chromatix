import jax
import jax.numpy as jnp
from jax import Array
import chromatix.experimental.diff_xnh as diff_xnh

def magnification(x: Array, mag: float | None, n: int | None = None) -> Array:
    """Applies a magnification of factor mag to the input image x through fourier domain, with optional output shape n.
    This essentially does an upscaling and then a USFFT."""
    mag = 1.0 if mag is None else mag
    ne = x.shape[-1]
    if n is None:
        n = ne

    # Centered fourier transform
    s = diff_xnh.utils.shift_matrix(x.shape[-1])
    x_hat = s * jnp.fft.fft2(x * s)  # [ne, ne]

    # Apply gaussian filter
    window, mu, m = diff_xnh.utils.make_gaussian_window(ne)
    x_hat = x_hat * window  # [ne, ne]

    # Pad and fft again
    x_hat = jnp.pad(x_hat, int(ne // 2))  # [2 ne, 2 ne]
    s = diff_xnh.utils.shift_matrix(x_hat.shape[-1])
    x_hat = s * jnp.fft.fft2(x_hat * s)
    x_hat = jnp.pad(x_hat, m, mode="wrap")

    img = gather_mag(x_hat, mag, m, mu, n, ne)
    return img / (4 * ne**2)

def gather_mag(f: Array, mag: float, m: int, mu: float, n: int, ne: int) -> Array:
    """Gathers the magnified image from the Fourier transform."""
    # Grid indices
    Y, X = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")

    # Physical coordinates
    x0 = -(X - n / 2) / (n * mag)
    y0 = -(Y - n / 2) / (n * mag)

    # things for in the loop
    ell1_base = jnp.floor(2 * ne * y0) - m
    ell0_base = jnp.floor(2 * ne * x0) - m
    coeff0 = jnp.pi / mu
    coeff1 = -(jnp.pi**2) / mu

    def accumulate_kernel(k: int, g_accum: Array) -> Array:
        i0, i1 = jnp.unravel_index(k, (2 * m + 1, 2 * m + 1))

        ell1 = ell1_base + i1
        ell0 = ell0_base + i0

        w0 = ell0 / (2 * ne) - x0
        w1 = ell1 / (2 * ne) - y0
        w = coeff0 * jnp.exp(coeff1 * (w0**2 + w1**2))

        idx_x = (ne + m + ell0).astype(jnp.int32)
        idx_y = (ne + m + ell1).astype(jnp.int32)

        f_val = f[idx_y, idx_x]
        return g_accum + (w * f_val) / ne**2

    return jax.lax.fori_loop(
        0, (2 * m + 1) ** 2, accumulate_kernel, jnp.zeros((n, n), dtype=f.dtype)
    )




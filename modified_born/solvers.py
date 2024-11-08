from functools import reduce
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.ndimage import distance_transform_edt
from scipy.signal.windows import tukey
from scipy.special import factorial


def add_bc(
    permittivity: Array,
    width: tuple[float | None, ...],
    spacing: float,
    wavelength: float,
    strength: float | None = None,
    alpha: float | None = None,
    order: int = 4,
) -> tuple[Array, tuple[slice, ...]]:
    """
    width is in wavelengths. Use None for periodic BCs.
    """

    # Figuring out new size
    n_pad = tuple(0 if width_i is None else int(width_i / spacing) for width_i in width)
    roi = tuple(slice(n, n + size) for n, size in zip(n_pad, permittivity.shape))

    # Padding permittivity to new size
    # We repeat the mean value
    padding = [(0, 0) for _ in range(permittivity.ndim)]
    for idx, n in enumerate(n_pad):
        padding[idx] = (n, n)
    permittivity = jnp.pad(permittivity, padding, mode="edge")

    # Gathering constants
    k0 = 2 * jnp.pi / wavelength
    km = k0 * jnp.sqrt(jnp.mean(permittivity))
    match (strength, alpha):
        case (Number(), None):
            alpha = strength * km / 2
        case (None, Number()):
            pass
        case (None, None):
            raise ValueError("Need at least strength or alpha set.")
        case (Number(), Number()):
            raise ValueError("Can only set either strength or alpha, not both.")
        case _:
            raise ValueError("Everything's wrong.")

    # Defining distance from sample
    r = jnp.ones_like(permittivity).at[roi].set(0)
    r = distance_transform_edt(r, sampling=spacing)

    # Making boundary
    ar = alpha * r
    P = reduce(
        lambda P, n: P + (ar**n / factorial(n, exact=True)),
        range(order + 1),
        jnp.zeros_like(ar),
    )

    numerator = alpha**2 * (order - ar + 2 * 1j * km * r) * ar ** (order - 1)
    denominator = P * factorial(order, exact=True)
    boundary = 1 / k0**2 * numerator / denominator

    # Inside the ROI it's 0
    boundary = boundary.at[roi].set(0)

    return permittivity + boundary, roi


def make_source(
    sample_shape: tuple[int, ...],
    spacing: float,
    wavelength: float,
    z_loc: float,
    width: float | None = None,
    alpha: float = 0.5,
) -> Array:
    # We first make a grid
    N_z, N_y, N_x = sample_shape
    z = spacing * (jnp.linspace(0, (N_z - 1), N_z) - N_z / 2)

    # Sinc options
    z_loc = 50
    width = wavelength / 4

    # Adding longitudinal apodisation
    source = jnp.sinc((z + z_loc) / width)
    k0 = 2 * jnp.pi / wavelength
    alpha = 0.5
    width_x = int(150 / spacing)
    width_y = 1
    n_pad = (N_x - width_x) // 2
    mask = (
        tukey(width_y, alpha, sym=False)[:, None]
        * jnp.pad(tukey(width_x, alpha, sym=False), ((n_pad, n_pad)))[None, :]
    )

    source = (
        source[:, None, None, None]
        * mask[None, ..., None]
        * (k0**2 * jnp.array([0, 1, 1]))
    )
    return source


def pad_fourier(x: Array) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    # Pads to fourier friendly shapes (powers of 2), depending
    # on periodic or absorbing BCs
    def n_pad(size) -> tuple[int, tuple[int, int]]:
        new_size = int(2 ** (np.ceil(np.log2(size))))
        return new_size, (0, new_size - size)

    return tuple(zip(*[n_pad(shape) for shape in x.shape]))


def bmatvec(mat: Array, vec: Array) -> Array:
    return jnp.matmul(mat, vec[..., None]).squeeze(-1)


def maxwell_solver(
    permittivity: Array, source: Array, spacing: float, wavelength: float
):
    fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))
    ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))

    def G_fn(k: Array, k0: ArrayLike, alpha: Array) -> Array:
        k_sq = jnp.sum(k**2, axis=-1)[..., None, None]
        k_cross = k[..., :, None] * k[..., None, :] / (alpha * k0**2)
        return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)

    def pad(field: Array) -> Array:
        return jnp.pad(field, (*padding, (0, 0)))

    def crop(field: Array) -> Array:
        return field[: V.shape[0], : V.shape[1], : V.shape[2], :]

    def propagate(G: Array, field: Array) -> Array:
        return crop(ifft(bmatvec(G, fft(pad(field)))))

    def update_fn(args):
        field, history, iteration = args

        # New field
        dE = (
            1j
            / alpha_imag
            * V[..., None]
            * (propagate(G, k0**2 * V[..., None] * field + source) - field)
        )

        # Calculating change
        delta = jnp.mean(jnp.abs(dE) ** 2) / jnp.mean(jnp.abs(field) ** 2)

        return field + dE, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        return (history[iteration - 1] > rtol) & (iteration < max_iter)

    padded_shape, padding = pad_fourier(permittivity)

    alpha_real = (jnp.min(permittivity.real) + jnp.max(permittivity.real)) / 2
    alpha_imag = jnp.max(jnp.abs(permittivity - alpha_real)) / 0.95
    alpha = alpha_real + 1j * alpha_imag

    k0 = 2 * jnp.pi / wavelength
    ks = [2 * jnp.pi * jnp.fft.fftfreq(shape, spacing) for shape in padded_shape]
    k_grid = jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)

    G = G_fn(k_grid, k0, alpha)

    V = permittivity - alpha

    rtol = 1e-6
    max_iter = 1000

    init = update_fn((jnp.zeros_like(source), jnp.zeros(max_iter), 0))
    field, history, iteration = jax.block_until_ready(
        jax.lax.while_loop(cond_fn, update_fn, init)
    )
    history = history[:iteration]
    return field, history

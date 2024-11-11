from functools import reduce
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from jax import Array
from jax.typing import ArrayLike
from scipy.ndimage import distance_transform_edt
from scipy.signal.windows import tukey
from scipy.special import factorial
from typing_extensions import Self

from chromatix import VectorField


class Sample(PyTreeNode):
    """Simple container to hold some sample specific data."""

    permittivity: Array
    roi: tuple[slice, ...]
    spacing: float

    @classmethod
    def init(cls, refractive_index: Array, spacing: float) -> Self:
        roi = tuple(slice(size) for size in refractive_index.shape[:3])
        return cls(refractive_index**2, roi, spacing)

    @property
    def shape(self):
        return self.permittivity.shape


class Source(PyTreeNode):
    """Simple container to hold some source related data."""

    source: Array
    wavelength: float

    @property
    def shape(self):
        return self.source.shape


def add_absorbing_bc(
    sample: Sample,
    wavelength: float,
    width: tuple[float | None, ...],
    strength: float | None = None,
    alpha: float | None = None,
    order: int = 4,
) -> Sample:
    """
    width is in mum. Use None for periodic BCs.
    """
    permittivity = sample.permittivity
    spacing = sample.spacing

    # Figuring out new size and roi
    n_pad = tuple(0 if width_i is None else int(width_i / spacing) for width_i in width)
    roi = tuple(slice(n, n + size) for n, size in zip(n_pad, permittivity.shape))

    # Padding permittivity to new size
    # We repeat the edge value
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
    r = jnp.ones_like(permittivity).at[roi].set(0.0)
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

    return Sample(permittivity + boundary, roi, spacing)


def plane_wave_source(
    field: VectorField,
    sample: Sample,
    z_loc: float = 0.0,
    width: float | None = None,
    apodise: bool = True,
    alpha: float = 0.5,
) -> Source:
    # Getting the longitudinal apodisation
    N_z, N_y, N_x = sample.shape
    z = sample.spacing * jnp.linspace(0, (N_z - 1), N_z)
    if width is None:
        width = field.spectrum.squeeze() / 4
    sinc_pulse = jnp.sinc((z - (sample.roi[0].start * sample.spacing + z_loc)) / width)

    # Padding the field
    # NOTE: We don't support multi wavelength yet.
    n_pad = (np.array([N_y, N_x]) - np.array(field.spatial_shape)) // 2
    padding = ((0, 0), (n_pad[0], n_pad[0]), (n_pad[1], n_pad[1]), (0, 0))
    u = jnp.pad(field.u.squeeze(-2), padding, mode="edge")

    # Making the actual source
    k0 = 2 * jnp.pi / field.spectrum.squeeze()
    source = sinc_pulse[:, None, None, None] * k0**2 * u

    # If necessary, transversal apodising
    if apodise:
        tukey_window = (
            tukey(N_y, alpha, sym=False)[:, None]
            * tukey(N_x, alpha, sym=False)[None, :]
        )
        source = source * tukey_window[None, ..., None]

    return Source(source, field.spectrum.squeeze())


def bmatvec(mat: Array, vec: Array) -> Array:
    return jnp.matmul(mat, vec[..., None]).squeeze(-1)


def bouter(a: Array, b: Array) -> Array:
    return a[..., :, None] * b[..., None, :]


class Results(PyTreeNode):
    field: Array
    roi: tuple[slice, ...]
    history: Array
    n_iter: int


def maxwell_solver(
    sample: Sample,
    source: Source,
    rtol: float = 1e-6,
    max_iter: int = 1000,
    field_init: Array | None = None,
) -> Results:
    # Helper methods
    def fft(x: Array) -> Array:
        return jnp.fft.fftn(x, axes=(0, 1, 2))

    def ifft(x: Array) -> Array:
        return jnp.fft.ifftn(x, axes=(0, 1, 2))

    def pad(field: Array) -> Array:
        return jnp.pad(field, (*padding, (0, 0)))

    def crop(field: Array) -> Array:
        return field[: sample.shape[0], : sample.shape[1], : sample.shape[2], :]

    def calculate_padding(
        shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        # Pads to fourier friendly shapes (powers of 2), depending
        # on periodic or absorbing BCs
        # Returns both padded shape and the padding to apply
        def n_pad(size) -> tuple[int, tuple[int, int]]:
            new_size = int(2 ** (np.ceil(np.log2(size))))
            return new_size, (0, new_size - size)

        return tuple(zip(*[n_pad(shape) for shape in shape]))

    # Physical methods
    def G_fn(k: Array, k0: ArrayLike, alpha: Array) -> Array:
        k_sq = jnp.sum(k**2, axis=-1)[..., None, None]
        k_cross = bouter(k, k) / (alpha * k0**2)
        return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)

    def propagate(G: Array, field: Array) -> Array:
        return crop(ifft(bmatvec(G, fft(pad(field)))))

    # Iteration methods
    def update_fn(args):
        field, history, iteration = args

        # New field
        scattered_field = k0**2 * V * field + source.source
        dE = 1j / alpha_imag * V * (propagate(G, scattered_field) - field)

        # Calculating change
        delta = jnp.mean(jnp.abs(dE) ** 2) / jnp.mean(jnp.abs(field) ** 2)

        return field + dE, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        return (history[iteration - 1] > rtol) & (iteration < max_iter)

    # Calculating background wavenumber and potential
    alpha_real = (sample.permittivity.real.min() + sample.permittivity.real.max()) / 2
    alpha_imag = jnp.max(jnp.abs(sample.permittivity - alpha_real)) / 0.99
    alpha = alpha_real + 1j * alpha_imag
    V = sample.permittivity[..., None] - alpha

    # We pad to a fourier friendly shape
    padded_shape, padding = calculate_padding(sample.shape)
    k0 = 2 * jnp.pi / source.wavelength
    ks = [2 * jnp.pi * jnp.fft.fftfreq(shape, sample.spacing) for shape in padded_shape]
    k_grid = jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)
    G = G_fn(k_grid, k0, alpha)

    # Running the actual calculation
    if field_init is None:
        field_init = jnp.zeros_like(source.source)
    init = update_fn((field_init, jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)

    return Results(field, sample.roi, history, iteration)

from functools import reduce
from numbers import Number

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from jax.lax import stop_gradient
from jax.typing import ArrayLike
from optimistix import FixedPointIteration, fixed_point, max_norm
from scipy.ndimage import distance_transform_edt
from scipy.signal.windows import tukey
from scipy.special import factorial
from typing_extensions import Self
from chromatix import VectorField
from optimistix import ImplicitAdjoint
import lineax as lx
import jaxopt as jop
import jax


class Sample(struct.PyTreeNode):
    """Simple container to hold some sample specific data."""

    permittivity: Array
    spacing: Array
    roi: tuple[slice, ...] = struct.field(pytree_node=False)

    @classmethod
    def init(cls, refractive_index: Array, spacing: Array) -> Self:
        roi = tuple(slice(size) for size in refractive_index.shape[:3])
        return cls(refractive_index**2, spacing, roi)

    @property
    def shape(self):
        return self.permittivity.shape


class Source(struct.PyTreeNode):
    """Simple container to hold some source related data."""

    source: Array
    wavelength: float

    @property
    def shape(self):
        return self.source.shape


def distance_transform(r: Array, spacing: Array):
    # Defining distance from sample
    def fn(r, spacing):
        return distance_transform_edt(r, sampling=spacing)

    out_type = jax.ShapeDtypeStruct(r.shape, r.dtype)
    return jax.pure_callback(fn, out_type, r, spacing, vmap_method="sequential")


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

    # Figuring out new size and roi
    n_pad = tuple(0 if width_i is None else int(width_i) for width_i in width)
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

    # Making boundary
    r = distance_transform(
        jnp.ones(permittivity.shape).at[roi].set(0.0), sample.spacing
    )
    ar = alpha * r
    P = reduce(
        lambda P, n: P + (ar**n / factorial(n, exact=True)),
        range(order + 1),
        jnp.zeros_like(ar),
    )

    numerator = alpha**2 * (order - ar + 2 * 1j * km * r) * ar ** (order - 1)
    denominator = P * factorial(order, exact=True)
    boundary = 1 / k0**2 * numerator / denominator
    boundary = boundary.at[roi].set(0.0)

    # We don't want the gradient through the boundary
    return sample.replace(permittivity=permittivity + stop_gradient(boundary), roi=roi)


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
    z = sample.spacing[0] * jnp.linspace(0, (N_z - 1), N_z)
    if width is None:
        width = field.spectrum.squeeze() / 4
    sinc_pulse = jnp.sinc(
        (z - (sample.roi[0].start * sample.spacing[0] + z_loc)) / width
    )

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


class Results(struct.PyTreeNode):
    field: Array
    rel_error: Array
    n_steps: int
    roi: tuple[slice, ...] = struct.field(pytree_node=False)


def maxwell_solver_optimistix(
    sample: Sample,
    source: Source,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    max_steps: int = 500,
    field_init: Array | None = None,
    pad_fourier: bool = True,
) -> Results:
    # NOTE: Optimistix implicit diff DOES NOT work right now, for some reason.
    # WIll have to dig in later.
    # Physical methods
    def G_fn(k: Array, k0: ArrayLike, alpha: Array) -> Array:
        k_sq = jnp.sum(k**2, axis=-1)[..., None, None]
        k_cross = bouter(k, k) / (alpha * k0**2)
        return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)

    def propagate(G: Array, field: Array) -> Array:
        fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))  # noqa: E731
        ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))  # noqa: E731
        return ifft(bmatvec(G, fft(field)))

    # Iteration methods
    def update_fn(u: Array, args: tuple[Array, Array, Array, Array]) -> Array:
        field = u[0] + 1j * u[1]  # making the complex field
        V, source, G, alpha = args
        scattered_field = k0**2 * V * field + source
        field = field + 1j / alpha.imag * V * (propagate(G, scattered_field) - field)
        return jnp.stack([field.real, field.imag])

    # Calculating background wavenumber and potential
    # We DO NOT want the gradient of alpha, we just want the gradient to flow through V
    alpha_real = (sample.permittivity.real.min() + sample.permittivity.real.max()) / 2
    alpha_imag = jnp.max(jnp.abs(sample.permittivity - alpha_real)) / 0.95
    alpha = stop_gradient(alpha_real + 1j * alpha_imag)
    _V = sample.permittivity[..., None] - alpha

    # Padding source and potential to next power of 2
    # We add 0 at the end to account that for source
    if pad_fourier:
        fourier_size = lambda size: int(2 ** (np.ceil(np.log2(size))))  # noqa: E731
        padding = [fourier_size(size) - size if size > 1 else 0 for size in _V.shape]
        padding = tuple((0, size) for size in padding)
        padded_roi = tuple(
            slice(int(n_pad // 2), int(size + n_pad // 2))
            for size, (_, n_pad) in zip(sample.shape, padding)
        )
        sample_roi = tuple(
            slice(roi_outer.start + roi_inner.start, roi_outer.start + roi_inner.stop)
            for roi_outer, roi_inner in zip(padded_roi, sample.roi)
        )

        pad = lambda x: jnp.pad(x, padding, mode="empty")  # noqa: E731
        _V, _source = pad(_V), pad(source.source)

    else:
        _source = source.source
        sample_roi = sample.roi

    # Making the Greens function
    k0 = 2 * jnp.pi / source.wavelength
    ks = [
        2 * jnp.pi * jnp.fft.fftfreq(shape, spacing)
        for shape, spacing in (_V.shape[:3], sample.spacing)
    ]
    k_grid = jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)
    G = G_fn(k_grid, k0, alpha)

    # Figuring out initial field
    if field_init is None:
        field_init = jnp.zeros((2, *_source.shape))
    else:
        field_init = jnp.stack([field_init.real, field_init.imag])
        field_init = pad(field_init) if pad_fourier else field_init

    # Running solver
    # This is custom norm operating only on the sample ROI
    def norm(x):
        return max_norm(x[:, sample_roi[0], sample_roi[1], sample_roi[2]])

    adjoint = ImplicitAdjoint(linear_solver=lx.NormalCG(atol=1e-3, rtol=1e-3))
    results = fixed_point(
        update_fn,
        solver=FixedPointIteration(rtol=rtol, atol=atol, norm=norm),
        y0=field_init,
        max_steps=max_steps,
        args=(_V, _source, G, alpha),
        throw=False,
        adjoint=adjoint,
    )

    # Decomposed, padded field to complex unpadded
    field = results.value[0] + 1j * results.value[1]
    if pad_fourier:
        field = field[padded_roi]
    return Results(field, results.state.relative_error, results.stats["num_steps"])


def maxwell_solver(
    sample: Sample,
    source: Source,
    tol: float = 1e-3,
    max_steps: int = 500,
    field_init: Array | None = None,
) -> Results:
    # Physical methods
    def G_fn(k: Array, k0: ArrayLike, alpha: Array) -> Array:
        k_sq = jnp.sum(k**2, axis=-1)[..., None, None]
        k_cross = bouter(k, k) / (alpha * k0**2)
        return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)

    def propagate(G: Array, field: Array) -> Array:
        fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))  # noqa: E731
        ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))  # noqa: E731
        return ifft(bmatvec(G, fft(field)))

    # Iteration methods
    def update_fn(u: Array, state: tuple[Array, Array, Array, Array]) -> Array:
        V, source, G, alpha = state
        field = u[0] + 1j * u[1]  # making the complex field
        scattered_field = k0**2 * V * field + source
        field = field + 1j / alpha.imag * V * (propagate(G, scattered_field) - field)
        return jnp.stack([field.real, field.imag])

    # Calculating background wavenumber and potential
    # We DO NOT want the gradient of alpha, we just want the gradient to flow through V
    alpha_real = (sample.permittivity.real.min() + sample.permittivity.real.max()) / 2
    alpha_imag = jnp.max(jnp.abs(sample.permittivity - alpha_real)) / 0.99
    alpha = stop_gradient(alpha_real + 1j * alpha_imag)
    V = sample.permittivity[..., None] - alpha

    # Making the Greens function
    k0 = 2 * jnp.pi / source.wavelength
    ks = [
        2 * jnp.pi * jnp.fft.fftfreq(shape, spacing)
        for shape, spacing in zip(V.shape[:3], sample.spacing)
    ]
    k_grid = jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)
    G = G_fn(k_grid, k0, alpha)

    # Figuring out initial field
    if field_init is None:
        field_init = jnp.zeros((2, *source.shape))
    else:
        field_init = jnp.stack([field_init.real, field_init.imag])

    # JAXopt solver setup and execution
    solver = jop.FixedPointIteration(update_fn, maxiter=max_steps, tol=tol)
    params, state = solver.run(field_init, (V, source.source, G, alpha))

    # Convert result back to complex field
    field = params[0] + 1j * params[1]
    return Results(
        field=field, rel_error=state.error, n_steps=state.iter_num, roi=sample.roi
    )


def thick_sample_exact(
    field: VectorField,
    sample: Sample,
    boundary_width: tuple[int | None],
    alpha: float = 0.35,
    order: int = 4,
    tol: float = 1e-3,
    max_steps: int = 500,
    sinc_width: float | None = None,
    field_init: Array | None = None,
) -> tuple[VectorField, Results]:
    sample = add_absorbing_bc(
        sample, field.spectrum.squeeze(), boundary_width, alpha=alpha, order=order
    )
    source = plane_wave_source(field, sample, width=sinc_width)
    results = maxwell_solver(
        sample, source, tol=tol, max_steps=max_steps, field_init=field_init
    )
    field = field.replace(u=results.field[sample.roi][-1][None, ..., None, :])
    return field, results

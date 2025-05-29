from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from field import AbstractField, EmptyField, Field
from jaxtyping import Array
from spectrum import AbstractSpectrum, Spectrum

from chromatix.utils.fft import fft, ifft


# Some functions
def l2_sq_norm(x):
    return jnp.sum(jnp.abs(x) ** 2, axis=-1)


def phase_change(field: AbstractField, z, n=1.0) -> AbstractField:
    L_sq = field.wavelength * z / n
    phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
    return field * jnp.exp(1j * phase)


def exact_propagate(
    field: AbstractField,
    z: float | Array,
    n: float | Array,
) -> AbstractField:
    propagator = compute_exact_propagator(field, z, n)
    field = kernel_propagate(field, propagator)
    return field


def kernel_propagate(field: AbstractField, propagator: Array) -> AbstractField:
    axes = field.spatial_dims
    u = ifft(fft(field.u, axes=axes) * propagator, axes=axes)
    return field.replace(u=u)


def compute_exact_propagator(
    field: AbstractField,
    z: float | Array,
    n: float | Array,
) -> Array:
    # This is all simplified
    # NOTE: k_grid now returns 2 * pi; old one should've been called fgrid
    km = field.k0 * n
    kernel = 1 - l2_sq_norm(field.k_grid) / km**2
    phase = z * km * jnp.sqrt(jnp.maximum(kernel, 0.0))  # removing evanescent waves
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def point_source(
    shape: tuple[int, int],
    dx: float | Array,
    spectrum: AbstractSpectrum,
    z: float,
    n: float,
    *,
    power: float | None = None,
    amplitude: float | Array | None = None,
    pupil: Callable[[AbstractField], AbstractField] | None = None,
) -> AbstractField:
    # Making empty field, mostly for grid.
    if amplitude is None:
        amplitude = jnp.ones((1,))
    field = EmptyField(amplitude, shape, dx, spectrum)

    # Making actual field
    eps = 1e-8
    L_sq = field.wavelength * z / n + eps
    phase = jnp.pi * l2_sq_norm(field.grid) / L_sq
    u = -1j * amplitude / L_sq * jnp.exp(1j * phase)
    field = field.replace(u=u)

    if pupil is not None:
        field = pupil(field)

    if power is not None:
        field = field * jnp.sqrt(power / field.power)

    return field


# %%
@eqx.filter_jit
def forward(shape):
    spectrum = Spectrum(0.532)
    field = point_source(shape, 0.25, spectrum, 1.0, 1.35)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = forward((512, 512))
print("Scalar")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
@eqx.filter_vmap
def forward(z):
    spectrum = Spectrum(0.532)
    field = point_source((512, 512), 0.25, spectrum, 1.0, 1.35)
    field = phase_change(field, 1.0)
    return exact_propagate(field, z, 1.35)


field = forward(jnp.linspace(10, 100, 10))
print("Vmapped over propagation, scalar")
print(f"field: {field}")
print(f"field shape: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


# %%
@eqx.filter_jit
def forward(u):
    spectrum = Spectrum(0.532)
    spacing = 0.25
    field = Field(u, spacing, spectrum)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = jax.vmap(forward)(jnp.ones((5, 512, 512)))
print("Vmapped scalar")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
def forward(u):
    spectrum = Spectrum([0.1, 0.532, 1.0], [0.2, 0.4, 0.1])
    spacing = 0.25
    field = Field(u, spacing, spectrum)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = forward(jnp.ones((512, 512, 3)))
print("PolyChromatic")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512, 3)))
print("Vmapped PolyChromatic")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
def forward(u):
    spacing = 0.25
    field = Field(u, spacing, Spectrum(0.532))
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = jax.vmap(forward)(jnp.ones((5, 512, 512, 3)))
print("Vmapped MonoChromatic Vector")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
def forward(u):
    spectrum = Spectrum([0.1, 0.532], [0.2, 0.4])
    spacing = 0.1
    field = Field(u, spacing, spectrum)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = jax.vmap(forward)(jnp.ones((5, 512, 512, 2, 3)))
print("Vmapped PolyChromatic Vector")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
@eqx.filter_vmap
def forward(z):
    spectrum = Spectrum([0.1, 0.3, 0.532, 0.7])
    field = point_source(
        (512, 512),
        0.25,
        spectrum,
        1.0,
        1.35,
        amplitude=jnp.ones((3,)),
    )
    field = phase_change(field, 1.0)
    return exact_propagate(field, z, 1.35)


field = forward(jnp.linspace(10, 100, 10))
print("Vmapped over propagation, vector, Polychromatic")
print(f"field: {field}")
print(f"field shape: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
def forward(z):
    spectrum = Spectrum([0.1, 0.3, 0.532, 0.7])
    field = point_source(
        (512, 512),
        -0.25,
        spectrum,
        1.0,
        1.35,
        amplitude=jnp.ones((3,)),
    )
    field = phase_change(field, 1.0)
    return exact_propagate(field, z, 1.35)


try:
    field = forward(10.0)
except:
    print("Failed, as spacing < 0.")


@eqx.filter_jit
def forward(u):
    spectrum = Spectrum([0.1, 0.532], [0.2, 0.4])
    spacing = 0.1
    field = Field(u, spacing, spectrum)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


try:
    field = jax.vmap(forward)(jnp.ones((5, 512, 512, 5, 3)))
except:
    print("Failed, as shapes inconsistent")


"""

# Current implementation
print("Stuff that doesn't work in the current implementation")
import chromatix.functional as cf


@jax.jit
def forward_chromatix(dx):
    u = jnp.ones((1, 512, 512, 1, 1))
    return cf.generic_field(dx, 1.0, 1.0, u, jnp.zeros_like(u))

field = forward_chromatix(0.1)
field.intensity.shape
# %%
field = jax.vmap(forward_chromatix)(jnp.ones((5, 1, 512, 512, 1, 1)))
field.intensity.shape
# %%
field.spectrum
"""

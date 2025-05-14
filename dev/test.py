import equinox as eqx
import jax
import jax.numpy as jnp
from field import (
    AbstractField,
    PolyChromaticScalarField,
    PolyChromaticVectorField,
    ScalarField,
    VectorField,
)
from jaxtyping import Array
from spectrum import Spectrum

from chromatix.utils.fft import fft, ifft


# Testing grid stuff
def l2_sq_norm(x):
    return jnp.sum(jnp.abs(x) ** 2, axis=-1)


def phase_change(field: AbstractField, z, n=1.0) -> AbstractField:
    L_sq = field.wavelength * z / n
    phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
    return field.replace(u=field.u * jnp.exp(1j * phase))  # showing that replace works


def exact_propagate(
    field: AbstractField,
    z: float | Array,
    n: float | Array,
    kykx: Array | tuple[float, float] = (0.0, 0.0),
) -> AbstractField:
    propagator = compute_exact_propagator(field, z, n, kykx)
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
    kykx: Array | tuple[float, float] = (0.0, 0.0),
) -> Array:
    # NOTE: k_grid now returns 2 * pi; old one should've been called fgrid
    km = field.k0 * n
    kernel = jnp.maximum(1 - l2_sq_norm(field.k_grid - jnp.asarray(kykx)) / km**2, 0.0)
    phase = z * km * jnp.sqrt(jnp.maximum(kernel, 0.0))  # removing evanescent waves
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


spectrum = Spectrum(0.532)
spacing = 0.25


# %%
@eqx.filter_jit
def forward(u):
    field = ScalarField(u, spacing, spectrum)
    field = phase_change(field, 1.0)
    return exact_propagate(field, 100.0, 1.35)


field = forward(jnp.ones((512, 512)))
print("Scalar")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512)))
print("Vmapped scalar")
print(f"field: {field.shape}")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


# %%
spectrum = Spectrum([0.1, 0.532, 1.0], [0.2, 0.4, 0.1])


@eqx.filter_jit
def forward(u):
    field = PolyChromaticScalarField(u, spacing, spectrum)
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
    field = VectorField(u, spacing, Spectrum(0.532))
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


spectrum = Spectrum([0.1, 0.532], [0.2, 0.4])


@eqx.filter_jit
def forward(u):
    field = PolyChromaticVectorField(u, spacing, spectrum)
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

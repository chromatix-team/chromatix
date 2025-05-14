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
from spectrum import Spectrum


# Testing grid stuff
def l2_sq_norm(x):
    return jnp.sum(jnp.abs(x) ** 2)


def phase_change(field: AbstractField, z, n=1.0) -> AbstractField:
    L_sq = field.wavelength * z / n
    phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
    return field.replace(u=field.u * jnp.exp(1j * phase))  # showing that replace works


spectrum = Spectrum(0.532)
spacing = 0.25


# %%
@eqx.filter_jit
def forward(u):
    field = ScalarField(u, spacing, spectrum)
    return phase_change(field, 1.0)


field = forward(jnp.ones((512, 512)))
print("Scalar")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512)))
print("Vmapped scalar")
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
    return phase_change(field, 1.0)


field = forward(jnp.ones((512, 512, 3)))
print("PolyChromatic")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512, 3)))
print("Vmapped PolyChromatic")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


@eqx.filter_jit
def forward(u):
    field = VectorField(u, spacing, Spectrum(0.532))
    return phase_change(field, 1.0)


field = jax.vmap(forward)(jnp.ones((5, 512, 512, 3)))
print("Vmapped MonoChromatic Vector")
print(f"Intensity: {field.intensity.shape}")
print(f"Power: {field.power.shape}")
print(f"Grid: {field.grid.shape}")
print(f"k_grid : {field.k_grid.shape}")
print("\n")


spectrum = Spectrum([0.1, 0.532], [0.2, 0.4])


@eqx.filter_jit
def forward(u):
    field = PolyChromaticVectorField(u, spacing, spectrum)
    return phase_change(field, 1.0)


field = jax.vmap(forward)(jnp.ones((5, 512, 512, 2, 3)))
print("Vmapped PolyChromatic Vector")
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

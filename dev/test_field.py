import jax
import jax.numpy as jnp
from field import AbstractField, CoherentScalarField, SpectralCoherentScalarField

# Testing grid stuff
def l2_sq_norm(x):
    return jnp.sum(jnp.abs(x)**2)

def phase_change(field: AbstractField , z, n=1.0) -> AbstractField:
    L_sq = field.spectrum * z / n
    phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
    return field.replace(u=field.u * jnp.exp(1j * phase)) # showing that replace works 

# %%
@jax.jit
def forward(u):
    field = CoherentScalarField(0.25, 0.532, u) 
    return phase_change(field, 1.0)

field = forward(jnp.ones((512, 512)))
print("Scalar")
print(field.intensity.shape)
print(field.power.shape)
print(field.grid.shape)
print(field.k_grid.shape)

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512)))
print("Vmapped scalar")
print(field.intensity.shape)
print(field.power.shape)
print(field.grid.shape)
print(field.k_grid.shape)
# %%
@jax.jit
def forward(u):
    field = SpectralCoherentScalarField(0.25, [0.1, 0.532, 1.0], [0.2, 0.4, 0.1], u) 
    return phase_change(field, 1.0)

field = forward(jnp.ones((512, 512, 3)))
print("Spectral")
print(field.intensity.shape)
print(field.power.shape)
print(field.grid.shape)
print(field.k_grid.shape)

# %%

field = jax.vmap(forward)(jnp.ones((5, 512, 512, 3)))
print("Vmapped Spectral")
print(field.intensity.shape)
print(field.power.shape)
print(field.grid.shape)
print(field.k_grid.shape)







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
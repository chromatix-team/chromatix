import jax
import jax.numpy as jnp
from field import CoherentScalarField, SpectralCoherentScalarField


# %%
@jax.jit
def forward(u):
    return CoherentScalarField(0.25, 0.532, u) 

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
    return SpectralCoherentScalarField(0.25, [0.1, 0.532, 1.0], [0.2, 0.4, 0.1], u) 

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
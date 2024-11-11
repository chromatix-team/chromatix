"""
Author: GJ Both
Date: 11/11/2024
Here we run a small test for running the inverse problem.
"""

# %% Imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from samples import bio_cylinders
from solvers import Sample, thick_sample_exact

import chromatix.functional as cf


# %% Forward pass - generating data
@jax.jit
def generate_data(sample: Sample, kvec: jax.Array) -> Array:
    field = cf.plane_wave(
        (sample.shape[1], sample.shape[2]),
        sample.spacing,
        spectrum=1.0,
        amplitude=jnp.array([0, 1, 1]),
        kykx=kvec,
    )
    field, results = thick_sample_exact(
        field,
        sample,
        boundary_width=(125, None, 125),
    )
    return field.intensity.squeeze(), results


sample = bio_cylinders()
sample = sample.replace(
    permittivity=sample.permittivity[::2, :, ::2], spacing=sample.spacing * 2
)
kvecs = jnp.array([0.0, 0.0])
intensity, results = generate_data(sample, kvecs)

# %%
plt.plot(intensity)

# %%
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Ex")
plt.imshow(jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 2])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Ey")
plt.imshow(jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 1])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

# %%

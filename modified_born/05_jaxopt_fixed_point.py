"""
Author: GJ Both
Date: 08/11/2024
In this notebook we replace all the functions from the previous notebook by a proper implementation.
"""

# %% Imports
import jax.numpy as jnp
import matplotlib.pyplot as plt
from samples import vacuum_cylinders
from solvers import thick_sample_exact
from jax import jit
import chromatix.functional as cf

# Settings
wavelength = 1.0

# %% Sample and incoming field
sample = vacuum_cylinders()
field = cf.plane_wave(
    (sample.shape[1], sample.shape[2]),
    sample.spacing,
    wavelength,
    amplitude=jnp.array([0, 1, 1]),
)
# 250 voxels = 25 wavelengths = 25 mum
field, results = jit(thick_sample_exact, static_argnames=("boundary_width"))(
    field, sample, (250, None, 250)
)

print(f"Converged in {results.n_steps} iterations.")
# %%
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Ex")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[sample.roi][:, 0, :, 2])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Ey")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[sample.roi][:, 0, :, 1])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Ez")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[sample.roi][:, 0, :, 0])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

# %%

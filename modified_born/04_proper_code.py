"""
Author: GJ Both
Date: 08/11/2024
In this notebook we replace all the functions from the previous notebook by a proper implementation.
"""

# %% Imports
import jax.numpy as jnp
import matplotlib.pyplot as plt
from samples import vacuum_cylinders
from solvers import add_absorbing_bc, plane_wave_source, maxwell_solver
import chromatix.functional as cf

# Settings
wavelength = 1.0
width = (25 / wavelength, None, 25 / wavelength)
alpha_boundary = 0.35
order = 4

# %% Sample
sample = vacuum_cylinders()
plt.title("Sample - cylinders in vacuum")
plt.imshow(jnp.rot90(sample.permittivity[:, 0, :]))
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar(label="Permittivity")

# %% Incoming field
field = cf.plane_wave(
    (sample.shape[1], sample.shape[2]),
    sample.spacing,
    wavelength,
    amplitude=jnp.array([0, 1, 1]),
)


# %% Now adding the absorbing boundary conditions
sample = add_absorbing_bc(sample, wavelength, width, alpha=alpha_boundary, order=order)
source = plane_wave_source(field, sample)

# %%
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Ex of source")
plt.imshow(jnp.abs(jnp.rot90(source.source[:, 0, :, 2].real)))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(122)
plt.title("Ey of source")
plt.imshow(jnp.abs(jnp.rot90(source.source[:, 0, :, 1])))
plt.colorbar(fraction=0.046, pad=0.04)


# %%
results = maxwell_solver(sample, source)

# %%
plt.title("Relative change in field")
plt.semilogy(results.history[: results.n_iter])
plt.ylabel("dE")
plt.xlabel("Iteration")

# %%
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Ex")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 2])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Ey")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 1])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Ez")
plt.imshow(
    jnp.rot90(jnp.abs(results.field[results.roi][:, 0, :, 0])), vmin=0.0, vmax=1.2
)
plt.colorbar(fraction=0.046, pad=0.04)

# %%

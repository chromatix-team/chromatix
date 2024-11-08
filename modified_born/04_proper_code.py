"""
Author: GJ Both
Date: 08/11/2024
In this notebook we replace all the functions from the previous notebook by a proper implementation.
"""

# %% Imports
import jax.numpy as jnp
import matplotlib.pyplot as plt
from samples import vacuum_cylinders
from solvers import add_bc, make_source, maxwell_solver

# Settings
spacing = 0.1
wavelength = 1.0
width = (25 / wavelength, None, 25 / wavelength)
alpha_boundary = 0.35
order = 4

# %%
n_sample = vacuum_cylinders()

# %%
plt.title("Sample - cylinders in vacuum")
plt.imshow(jnp.rot90(n_sample[:, 0, :]))
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar(label="Refractive index")

# %% Now adding the absorbing boundary conditions
permittivity, roi = add_bc(
    n_sample**2, width, spacing, wavelength, alpha=alpha_boundary, order=order
)
source = make_source(permittivity.shape, spacing, wavelength, 25)

# %%
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Ex of source")
plt.imshow(jnp.rot90(source[:, 0, :, 2]))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(122)
plt.title("Ey of source")
plt.imshow(jnp.rot90(source[:, 0, :, 1]))
plt.colorbar(fraction=0.046, pad=0.04)


# %%
field, history = maxwell_solver(permittivity, source, spacing, wavelength)

# %%
plt.title("Relative change in field")
plt.semilogy(history)
plt.ylabel("dE")
plt.xlabel("Iteration")

# %%
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Ex")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 2])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Ey")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 1])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 0])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

# %%

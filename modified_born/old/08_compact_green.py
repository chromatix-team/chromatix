# In this script we recreate figure 2 (cylinders in biomaterial)
# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
from samples import Sample, Source, vacuum_cylinders, bio_cylinders
import matplotlib.pyplot as plt
from jax import Array

# %%
wavelength = 1.0
spacing = 0.1
n_sample = vacuum_cylinders()
sample = Sample.init(
    n_sample,
    spacing,
    wavelength,
    boundary_type="pbl",
    boundary_width=(25, None, 25),
    boundary_strength=0.35,
)

source = Source(
    field=jnp.zeros((*n_sample.shape, 3)).at[0].set(jnp.array([0, 1, 1])),
    wavelength=wavelength,
)


# %% Making Greens function
def G_fn(k: Array, k0: Array, alpha: Array) -> Array:
    k_sq = jnp.sum(jnp.abs(k) ** 2, axis=-1)[..., None, None]
    k_cross = k[..., :, None] * k[..., None, :] / (alpha * k0**2)
    return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)


def bmatvec(mat: Array, vec: Array) -> Array:
    return jnp.matmul(mat, vec[..., None]).squeeze(-1)


def propagate(G: Array, field: Array) -> Array:
    fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))
    ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))

    return ifft(bmatvec(G, fft(field)))


def maxwell_solver(source: Source, sample: Sample, rtol=1e-8, max_iter: int = 1000):
    def update_fn(args):
        field, history, iteration = args

        # New field
        dE = 1j / alpha_imag * V * (propagate(Gk, k0**2 * V * field + _source) - field)

        # Calculating change
        delta = jnp.mean(jnp.abs(dE) ** 2) / jnp.mean(jnp.abs(field) ** 2)

        return field + dE, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        return (history[iteration - 1] > rtol) & (iteration < max_iter)

    # Getting real part of alpha
    alpha_real = (jnp.min(sample.permittivity) + jnp.max(sample.permittivity)) / 2
    alpha_imag = jnp.max(jnp.abs(sample.permittivity - alpha_real)) / 0.95
    alpha = alpha_real + 1j * alpha_imag

    # Making greens function and potential
    k0 = 2 * jnp.pi / source.wavelength
    Gk = G_fn(sample.k_grid, k0, alpha)
    V = (sample.permittivity - alpha)[..., None]

    # Setting up source
    _source = jnp.zeros((*sample.shape, 3)).at[*sample.roi, :].set(source.source)

    # Running and postprocessing
    init = update_fn((_source, jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)
    return field[sample.roi], field, {"error": history, "n_iterations": iteration}


# %%
field, field_full, stats = maxwell_solver(source, sample, max_iter=1000)


# %%
plt.semilogy(stats["error"][: stats["n_iterations"]])

# %%
plt.imshow(jnp.rot90(jnp.sum(jnp.abs(field[:, 0, :]) ** 2, axis=-1)))  # %%
plt.colorbar()

# %%
plt.figure(figsize=(15, 10))
plt.subplot(131)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 0]) ** 2), cmap="jet")  # %%
plt.title("E_z")
plt.xlabel("z")
plt.ylabel("x")
# plt.colorbar()

plt.subplot(132)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 1]) ** 2), cmap="jet")  # %%
plt.title("E_y")
plt.xlabel("z")
plt.ylabel("x")
# plt.colorbar()

plt.subplot(133)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 2]) ** 2), cmap="jet")  # %%
plt.title("E_x")
plt.xlabel("z")
plt.ylabel("x")
# plt.colorbar()
# %%
plt.imshow(jnp.log10(jnp.rot90(jnp.abs(field[:, 0, :, 0]) ** 2)))  # %%
# %%
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 0]) ** 2), cmap="jet")  # %%
# %%
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 1]) ** 2), cmap="jet")  # %%
plt.colorbar()
# %%

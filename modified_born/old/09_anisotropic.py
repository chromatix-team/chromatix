# In this script we recreate figure 2 (cylinders in biomaterial)
# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from jaxopt import BFGS, LBFGS, ScipyMinimize
from samples import Sample, Source, vacuum_cylinders

# %%
wavelength = 1.0
spacing = 0.1
n_sample = vacuum_cylinders()
sample = jnp.zeros((*n_sample.shape, 3, 3))
sample = sample.at[..., 0, 0].set(n_sample)
sample = sample.at[..., 1, 1].set(n_sample)
sample = sample.at[..., 2, 2].set(n_sample)

sample = Sample.init(
    sample,
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


# %%
def maxwell_solver(source: Source, sample: Sample, rtol=1e-8, max_iter: int = 1000):
    def update_fn(args):
        field, history, iteration = args

        # New field
        field_prop = propagate(Gk, k0**2 * bmatvec(xi, field) + _source)
        dE = 1j / alpha_imag * bmatvec(xi, (field_prop - field))

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
    xi = sample.permittivity - alpha * jnp.eye(3)

    # Setting up source
    _source = (
        jnp.zeros((*sample.spatial_shape, 3)).at[*sample.roi, :].set(source.source)
    )

    # Running and postprocessing
    init = update_fn((jnp.zeros_like(_source), jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)
    return field[sample.roi], {
        "full_field": field,
        "error": history,
        "n_iterations": iteration,
    }


# %%
field, stats = maxwell_solver(source, sample, max_iter=1000)


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


min_fn = lambda a: jnp.max(jnp.linalg.svdvals(sample.permittivity - jnp.eye(3) * a))
opt = BFGS(min_fn, options={})
approx = (
    jnp.min(jnp.real(sample.permittivity)) + jnp.max(jnp.real(sample.permittivity))
) / 2
# %%
res = opt.run(approx)

# %%

# %%
opt = ScipyMinimize(fun=min_fn, method="Nelder-mead", options={"xtol": 1e-2})
res = opt.run(approx)
# %%
opt = LBFGS(jax.value_and_grad(min_fn), jit=True, value_and_grad=True)
res = opt.run(jnp.array([0.8]))
print(res)
# %%
a = 0.8

fn = lambda y: sample.permittivity - jnp.eye(3) * y
min_fn = lambda x: jnp.max(jnp.linalg.eigvalsh(jnp.conjugate(fn(x)) * fn(x)))
opt = ScipyMinimize(fun=min_fn, method="Nelder-mead")
res = opt.run(0.8)
# %%

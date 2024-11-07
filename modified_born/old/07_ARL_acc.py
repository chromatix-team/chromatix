# In this script we recreate figure 2 (cylinders in biomaterial)
# from the maxwells paper

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from samples import Sample, cylinders, sample_grid

# %% Making sample
cylinder_locs = jnp.array(
    [
        [-44.5, -44.5],
        [44.5, -44.5],
        [44.5, 44.5],
        [20.6, -18.0],
        [-10.4, 18.1],
    ]
)

cylinder_radius = 5.0
n_cylinder = 1.2
n_mat = 1.0
wavelength = 1.0
spacing = 0.1
sim_size = 100
sim_shape = int(sim_size / spacing)

n_sample = cylinders(
    (sim_shape, 1, sim_shape),
    spacing,
    cylinder_locs,
    cylinder_radius,
    n_mat,
    n_cylinder,
    antialiasing=10,
).squeeze((-2, -1))

# %%
plt.imshow(jnp.rot90(n_sample.squeeze(), 1))
plt.ylabel("x")
plt.xlabel("z")
plt.title("Cylinder mask")
plt.colorbar()

# %%
source = ((2 * jnp.pi / wavelength) ** 2 * (1.0 - n_cylinder**2)) * jnp.zeros(
    (*n_sample.shape, 3)
).at[0].set(jnp.array([0, 1, 1]))
sample = Sample.init(
    n_sample,
    source,
    spacing,
    wavelength,
    (25 / wavelength, None, 25 / wavelength),
    boundary_type="arl",
)


# %% Making Greens function
def G_fn(k, k_red_sq) -> Array:
    G = jnp.zeros((3, 3, *k.shape[1:]), dtype=jnp.complex64)

    # Setting diagonal
    k_sq = jnp.sum(jnp.abs(2 * jnp.pi * sample.k_grid) ** 2, axis=0)
    G_diag = 1 / (k_sq - k_red_sq)
    G = G.at[jnp.diag_indices(3)].set(G_diag)

    # Calculating off-diagonal elements
    g_ij = lambda i, j: -k[i] * k[j] / ((k_sq - k_red_sq) * k_red_sq)
    # Adding diagonal
    G = G.at[0, 0].add(g_ij(0, 0))
    G = G.at[1, 1].add(g_ij(1, 1))
    G = G.at[2, 2].add(g_ij(2, 2))

    # Setting upper diagonal
    G = G.at[0, 1].set(g_ij(0, 1))
    G = G.at[0, 2].set(g_ij(0, 2))
    G = G.at[1, 2].set(g_ij(1, 2))

    # Setting lower diagonal, mirror symmetry
    G = G.at[1, 0].set(g_ij(0, 1))
    G = G.at[2, 0].set(g_ij(0, 2))
    G = G.at[2, 1].set(g_ij(1, 2))

    # We move the axes to the back, easier matmul
    return jnp.moveaxis(G, (0, 1), (-2, -1))


def bmatvec(a, b):
    return jnp.matmul(a, b[..., None]).squeeze(-1)


def propagate(field, k, k_red_sq):
    fft = lambda x: jnp.fft.fftn(x, axes=range(sample.permittivity.ndim))
    ifft = lambda x: jnp.fft.ifftn(x, axes=range(sample.permittivity.ndim))

    x = sample_grid(sample.permittivity.shape)
    dk = 0.1
    wiggles = jnp.array(
        [
            [dk / 4, 0, dk / 4],
            [-dk / 4, 0, -dk / 4],
            [-dk / 4, 0, dk / 4],
            [dk / 4, 0, -dk / 4],
        ]
    )
    for wiggle in wiggles:
        kx = jnp.einsum("n, zyxn -> zyx", wiggle, x)
        ramp = jnp.exp(1j * 2 * jnp.pi * kx)
        Gk = G_fn(k + wiggle[:, None, None, None], k_red_sq)
        field += 1 / wiggles.shape[0] * ifft(bmatvec(Gk, fft(field * ramp[..., None])))
    return field


def maxwell_solver(sample: Sample, rtol=1e-8, max_iter: int = 1000):
    def update_fn(args):
        field, history, iteration = args

        # New field
        dE = field - propagate(V * field + source, sample.k_grid, k_red_sq)
        field = field - 1j / epsilon * V * dE

        # Calculating change
        delta = jnp.mean(jnp.abs(dE[sample.roi]) ** 2)
        delta /= jnp.mean(jnp.abs(field[sample.roi]) ** 2)

        return field, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        # return (history[iteration - 1] > rtol) & (iteration < max_iter)
        return iteration < max_iter

    # Unpacking sample
    source, V, epsilon = sample.source, sample.potential(), sample.epsilon

    # Making Greens vector
    k_red_sq = sample.km_sq + 1j * epsilon

    # Running and postprocessing
    init = update_fn((source, jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)
    return field[sample.roi], field, {"error": history, "n_iterations": iteration}


# %%
field, field_full, stats = maxwell_solver(sample, max_iter=500)


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

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
import optax
import chromatix.functional as cf
import numpy as np


# %% Forward pass - generating data
def generate_data(
    sample: Sample, kvec: jax.Array, field_init: Array | None = None
) -> Array:
    field = cf.plane_wave(
        (sample.shape[1], sample.shape[2]),
        sample.spacing[-1],
        spectrum=2.0,
        amplitude=jnp.array([0, 1, 1]),
        kykx=kvec,
    )
    field, results = thick_sample_exact(
        field,
        sample,
        boundary_width=(268, None, 268),
        field_init=field_init,
    )
    return field.intensity.squeeze(), results


sample = bio_cylinders()
# 2D so we can only have 5 vector

kvecs = jnp.pi * jnp.stack([jnp.zeros((10,)), jnp.linspace(-0.2, 0.2, 10)], axis=1)
measurements, results = jax.jit(jax.vmap(generate_data, in_axes=(None, 0)))(
    sample, kvecs
)
print(f"Shape of measurements: {measurements.shape}")
# %%
plt.plot(measurements.T)
plt.title("Intensities from different angle of illuminations.")

# %%
plt.figure(figsize=(15, 5), layout="tight")
plt.subplot(151)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[0][sample.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(152)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[1][sample.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(153)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[2][sample.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(154)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[3][sample.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(155)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[4][sample.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)


# %% Defining loss and update fn
def loss_fn(refractive_index, measurements, kvecs, field_init):
    sample = Sample.init(refractive_index, spacing=jnp.full((3,), 0.4))
    images, results = jax.vmap(generate_data, in_axes=(None, 0, 0))(
        sample, kvecs, field_init
    )
    l = 5e-3
    tv = jnp.sqrt(
        jnp.sum(jnp.diff(n_sample, axis=0) ** 2)
        + jnp.sum(jnp.diff(n_sample, axis=2) ** 2)
    )
    return jnp.mean(jnp.abs(images - measurements)) + l * tv, (
        results.field,
        results.n_steps,
    )


@jax.jit
def update_fn(params, opt_state, measurements, kvecs, field_init):
    (loss, (field_init, n_iter)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, measurements, kvecs, field_init
    )
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, field_init, n_iter


# %% TRAAAAAINING
print("Starting training now.")
# We know it's a biosample so we initialise at 1.3
# n_sample = jnp.ones_like(sample.permittivity)
n_sample = jnp.full((1000, 1, 1000), 1.33)

optimiser = optax.adam(1e-4)
opt_state = optimiser.init(n_sample)
field_init = jnp.zeros((kvecs.shape[0], 1536, 1, 1536, 3), dtype=jnp.complex64)
loss_hist = []
convergence_hist = []

# %%
for idx in range(1000):
    n_sample, opt_state, loss, field_init, n_iter = update_fn(
        n_sample, opt_state, measurements, kvecs, field_init
    )
    loss_hist.append(loss)
    convergence_hist.append(n_iter)

    if idx % 25 == 0:
        print(f"Iteration {idx}: {loss:.2f}, {n_iter} iterations to converge.")
        plot_permittivity(n_sample, sample)
        plot_loss(loss_hist)


# %%
def plot_loss(loss):
    plt.title("Loss")
    plt.semilogy(loss)
    plt.show()


def plot_permittivity(n_learned, sample):
    plt.figure(layout="tight")
    plt.subplot(121)
    plt.imshow(np.rot90(n_learned[:, 0, :] ** 2))
    plt.title("Inferred permittivity")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(122)
    plt.imshow(np.rot90(sample.permittivity[:, 0, :]))
    plt.title("True permittivity")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()


# %%
plot_permittivity(n_sample, sample)
# %%

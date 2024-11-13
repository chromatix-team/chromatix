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
import jax.random as jr


# %% Forward pass - generating data
def generate_data(
    sample: Sample, kvec: jax.Array, field_init: Array | None = None
) -> Array:
    field = cf.plane_wave(
        (sample.shape[1], sample.shape[2]),
        sample.spacing,
        spectrum=2.0,
        amplitude=jnp.array([0, 1, 1]),
        kykx=kvec,
    )
    field, results = thick_sample_exact(
        field,
        sample,
        boundary_width=(125, None, 125),
        max_iter=250,
        rtol=1e-3,
        field_init=field_init,
    )
    return field.intensity.squeeze(), results


sample = bio_cylinders()
sample = sample.replace(
    permittivity=sample.permittivity[::4, :, ::4], spacing=sample.spacing * 4
)

# 2D so we can only have 5 vector

kvecs = jnp.pi * jnp.array(
    [[0.0, -0.25], [0, -0.125], [0.0, 0.0], [0.0, 0.125], [0.0, 0.25]]
)
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
plt.imshow(jnp.rot90(jnp.abs(results.field[0][results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(152)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[1][results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(153)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[2][results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(154)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[3][results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(155)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(results.field[4][results.roi][:, 0, :, 0])))
plt.colorbar(fraction=0.046, pad=0.04)


# %% Defining loss and update fn
def loss_fn(refractive_index, measurements, kvecs, field_init):
    sample = Sample.init(refractive_index, spacing=0.4)
    images, results = jax.vmap(generate_data, in_axes=(None, 0, 0))(
        sample, kvecs, field_init
    )
    return jnp.mean((images - measurements) ** 2) / jnp.mean(measurements) ** 2, (
        results.field,
        results.n_iter,
    )


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
n_sample = 1.3 + 0.01 * jr.normal(jr.key(42), (250, 1, 250))

optimiser = optax.adamw(1e-3)
opt_state = optimiser.init(n_sample)
field_init = jnp.zeros((kvecs.shape[0], 500, 1, 500, 3), dtype=jnp.complex64)
for idx in range(10):
    n_sample, opt_state, loss, field_init, n_iter = update_fn(
        n_sample, opt_state, measurements, kvecs, field_init
    )
    print(f"Iteration {idx}: {loss:.2f}, {n_iter} iterations to converge.")

# %%

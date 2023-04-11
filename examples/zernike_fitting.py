### Train Zernike coefficients

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import numpy as np

from optax import adam
import optax

import matplotlib.pyplot as plt

from chromatix.elements import ObjectivePointSource, FFLens, ZernikeAberrations
from chromatix.ops.noise import shot_noise
from chromatix.utils import trainable

from typing import Callable, Optional, Tuple

key = random.PRNGKey(42)


# %% Set parameters

camera_shape: Tuple[int, int] = (256, 256)
camera_pixel_pitch: float = 0.125
f: float = 100
NA: float = 0.8
n: float = 1.33
wavelength: float = 0.532
wavelength_ratio: float = 1.0
upsample: int = 4
pad: int = 128
crop: int = 450  # for visualization purposes
taper_width: Optional[float] = 5
noise_fn: Callable = shot_noise
shape = tuple(np.array(camera_shape) * upsample + pad)
spacing = upsample * f * wavelength / (n * shape[0] * camera_pixel_pitch)

print(f"Shape of simulation: {shape}")
print(f"Spacing of simulation: {spacing:.2f}")

# Specify "ground truth" parameters for Zernike coefficients
ansi_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
coefficients_truth = jnp.array([2.0, 5.0, 3.0, 0, 1, 0, 1, 0, 1, 0])


# Specify model
class ZernikePSF(nn.Module):
    @nn.compact
    def __call__(self, z):
        field = ObjectivePointSource(
            shape, spacing, wavelength, wavelength_ratio, f, n, NA, power=1e7
        )(z)
        # The only learnable parameters are the Zernike coefficients (since we use the trainable flag)
        field = ZernikeAberrations(
            trainable(jnp.array([0.0] * len(ansi_indices))), f, n, NA, ansi_indices
        )(field)
        field = FFLens(f, n)(field)
        return field


# Initialize model
model = ZernikePSF()
params = model.init(key, z=0)

# Set "ground truth" parameters (which will eventually try to estimate)
params_truth = {
    "params": {"ZernikeAberrations_0": {"zernike_coefficients": coefficients_truth}}
}

# Get the corresponding "measured" PSF using the true coefficients
psf_truth = model.apply(params_truth, z=0)
psf_truth = shot_noise(key, psf_truth.intensity)  # add shot noise

plt.figure(dpi=150)
plt.imshow(psf_truth.squeeze()[crop:-crop, crop:-crop], cmap="afmhot")
plt.colorbar()
plt.title("'Measured' PSF")
plt.axis("off")
plt.show()


# This loss function will be our metric for measuring how close our predicted PSF
# is to the measured PSF. For now we use mean squared error.
def loss_fn(params, data, z):
    psf_estimate = model.apply(params, z=z).intensity.squeeze()
    loss = jnp.mean(jnp.square(psf_estimate - data.squeeze()))
    return loss, {"loss": loss}


grad_fn = jax.jit(jax.grad(loss_fn, has_aux=True))  # jit compiling


# The step function takes care of a single optimization step, including updating our coefficient estimate.
def step_fn(loss_fn, optimizer):
    def step(params, opt_state, *args):
        (_, metrics), grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
            params, *args
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    return step


# Use adam optimizer for learning
optimizer = adam(learning_rate=0.5)
opt_state = optimizer.init(params)
step = jax.jit(
    step_fn(loss_fn, optimizer)
)  # jit compiles and makes everything go brrrrr


# %% Initial guess
params = model.init(key, z=0)  # dummy parameters for init
psf_init = model.apply(params, z=0).intensity.squeeze()

plt.figure(dpi=150)
plt.imshow(psf_init[crop:-crop, crop:-crop], cmap="afmhot")
plt.colorbar()
plt.title("Initial Guess")
plt.axis("off")
plt.show()

# Optimize
max_iterations = 150
print_every = 100
history = []

for iteration in range(max_iterations):
    params, opt_state, metrics = step(params, opt_state, psf_truth, 0)
    history.append(metrics["loss"])
    if iteration % print_every == 0:
        print(iteration, metrics)

# %% Plot results

coefficients_estimated = jnp.abs(
    params["params"]["ZernikeAberrations_0"]["zernike_coefficients"]
)
print("Estimated coefficients:", coefficients_estimated)

coefficients_error = jnp.square(coefficients_estimated - coefficients_truth)
print("Coefficients error:", coefficients_error)

mean_coefficients_error = jnp.mean(
    jnp.square(coefficients_estimated - coefficients_truth)
)
print("Mean coefficients error:", mean_coefficients_error)

# Plots of ground truth, estimates and errors
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fontsize = 20
ax[0].plot(ansi_indices, coefficients_truth, "-o", label="Ground truth")
ax[0].plot(ansi_indices, coefficients_estimated, "-o", label="Estimated")
ax[0].set_xlabel("Zernike index", fontsize=fontsize)
ax[0].set_ylabel("Coefficient", fontsize=fontsize)
ax[0].legend(fontsize=fontsize)
ax[1].semilogy(ansi_indices, coefficients_error, "k-o")
ax[1].set_xlabel("Zernike index", fontsize=fontsize)
ax[1].set_ylabel("Estimation error", fontsize=fontsize)


# PSF plots
psf_estimated = model.apply(params, z=0).intensity.squeeze()

fig, ax = plt.subplots(1, 2, dpi=300)
m = ax[0].imshow(psf_estimated[crop:-crop, crop:-crop], cmap="afmhot")
plt.colorbar(m, fraction=0.046, pad=0.04)
ax[0].title.set_text("Estimated PSF")
m = ax[1].imshow(psf_truth.squeeze()[crop:-crop, crop:-crop], cmap="afmhot")
plt.colorbar(m, fraction=0.046, pad=0.04)
ax[1].title.set_text("True PSF")
for ax in ax:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

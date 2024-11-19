import os
from functools import partial
from timeit import default_timer as timer
import datetime

import chromatix.functional as cx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import optax
from optax import adam, rmsprop, adagrad
from skimage import io

from holoscope.optimization.losses import poisson_loss

key = jax.random.PRNGKey(42)

# read in angled data
all_angles = jnp.load("/nrs/turaga/debd/misc/qpi/k.npy")
# reshape for vmap over the first dim
all_angles = all_angles.reshape((-1, 2))[:, ::-1]
print(f"Using {all_angles.shape[0]} angles")

# read in ground truth
GT_sample_delay = io.imread("/nrs/turaga/debd/misc/qpi/dn_GT.tif")

# thickness of each sample slice in microns
thickness = 1.0

# length of the field in pixels
size = 1200

# pixel spacing
dx = 0.1154

# wavelength in microns
wavelength = 0.523

# power is 1 per pixel
power = (size * size) * dx * dx

# define a zero absorption
absorption = jnp.zeros_like(GT_sample_delay)
# absorption = io.imread("/nrs/turaga/debd/misc/qpi/celegans.tif")

# trainable sample phase, zero initialized
params = jnp.zeros_like(GT_sample_delay)
# params = jax.random.uniform(key, GT_sample_delay.shape)
# params = GT_sample_delay.copy()

# read in the recorded intensity from all angles
target = io.imread("/nrs/turaga/debd/misc/qpi/FOV_01_rawdata_CElegan.tif")[:120]
print(target.max(), target.min())
target = jnp.reshape(target, (-1, 1200, 1200))

# weight of the tv regularizer
tvlamda = 5e-3

# batch settings
batch_size = 1
n_batches = int(jnp.ceil(120 / batch_size))

max_iterations = 800

# optimizer = adam(learning_rate=1e-6)
optimizer = adam(learning_rate=1e-5)

# ------------------------------------------------------------------------
# @partial(jax.pmap, in_axes=(0, None, None))
propagators = jax.vmap(
    partial(
        cx.compute_exact_propagator,
        field=cx.plane_wave(
            shape=(size, size),
            dx=dx,
            spectrum=wavelength,
            spectral_density=1.0,
            power=power,
        ),
        z=thickness,
        n=1.33,
    )
)(kykx=all_angles)
print(propagators.shape)


@partial(jax.vmap, in_axes=(0, 0, None, None))
def generate_intensity_from_angle(kykx, propagator, sample_delay, sample_absorption):
    field = cx.plane_wave(
        shape=(size, size),
        dx=dx,
        spectrum=wavelength,
        spectral_density=1.0,
        power=power,
    )
    field = cx.multislice_thick_sample(
        field,
        sample_absorption,
        sample_delay,
        1.33,
        thickness,
        N_pad=0,
        propagator=propagator,
        kykx=kykx,
    )
    return field.intensity.squeeze()


def loss_fn(
    params,
    angles,
    propagator,
    sample_absorption,
    target,
    lamda,
):
    out = generate_intensity_from_angle(angles, propagator, params, sample_absorption)
    tv = tv3d(params)
    mae = jnp.mean(jnp.abs(out - target))
    mse = jnp.mean((out - target) ** 2)
    # poisson = poisson_loss(out, target)
    # loss = mse + lamda * tv
    loss = mae + lamda * tv
    # loss = poisson + lamda * tv
    return loss, {"loss": loss, "mse": mse, "tv": tv}


def tv3d(params):
    # assuming a (D, H, W) shape
    # nb_pixel = (params.shape[0]) * (params.shape[1]) * (params.shape[2])
    sy = params[:, 1:, :] - params[:, :-1, :]
    sx = params[:, :, 1:] - params[:, :, :-1]
    sz = params[1:, :, :] - params[:-1, :, :]

    tvloss = jnp.sqrt((sx**2).sum() + (sy**2).sum() + (sz**2).sum() + 1e-8)
    return tvloss


def step_fn(loss_fn, optimizer):
    def step(params, angles, propagator, opt_state, sample_absorption, target, lamda):
        (_, metrics), grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
            params, angles, propagator, sample_absorption, target, lamda
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = ((params + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return params, opt_state, metrics

    return step


opt_state = optimizer.init(params)
step = jax.jit(step_fn(loss_fn, optimizer))

# training loop
history = {"loss": [], "mse": [], "tv": []}
step_times = []

total_start = timer()
for epoch in range(max_iterations):
    for i in sample(list(range(n_batches)), n_batches):
        start = datetime.datetime.now()
        params, opt_state, metrics = step(
            params,
            all_angles[batch_size * i : batch_size * (i + 1)],
            propagators[batch_size * i : batch_size * (i + 1)],
            opt_state,
            absorption,
            target[batch_size * i : batch_size * (i + 1)],
            tvlamda,
        )
        params.block_until_ready()
        duration = datetime.datetime.now() - start
        step_times.append(duration.total_seconds())
        history["loss"].append(metrics["loss"])
        history["mse"].append(metrics["mse"])
        history["tv"].append(metrics["tv"])
        print(
            f"{datetime.datetime.now()}: loss = {metrics['loss']}, mse = {metrics['mse']} tv = {metrics['tv']} step time = {step_times[-1]}"
        )
total_end = timer()
print(f"Total time = {total_end - total_start}")

save_path = f"/nrs/turaga/debd/lfm/holoscope-experiments/{os.getcwd().split('/')[-2]}/{os.getcwd().split('/')[-1]}"
os.makedirs(save_path, exist_ok=True)
io.imsave(f"{save_path}/multislice_recon.tif", params)
np.save(f"{save_path}/step_times.npz", np.array(step_times))
np.save(f"{save_path}/metrics.npz", history)


# # gt_angles = generate_intensity_from_angle(all_angles, GT_sample_delay, absorption)
# # io.imsave("multislice_gt_angles.tif", gt_angles)


# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
# ax[0].plot(history["loss"])
# ax[0].set_title("loss")
# ax[1].plot(history["mse"])
# ax[1].set_title("mse")
# ax[2].plot(history["tv"])
# ax[2].set_title("tv")
# plt.savefig("losses.png")

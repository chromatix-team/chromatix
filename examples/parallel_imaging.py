from functools import partial
from time import perf_counter_ns

import jax
import jax.numpy as jnp
import numpy as np

from chromatix.elements import BasicSensor
from chromatix.systems import Microscope, Optical4FSystemPSF

num_devices = 4
num_planes_per_device = 32
num_planes = num_devices * num_planes_per_device
shape = (1536, 1536)  # number of pixels in simulated field
spacing = 0.3  # spacing of pixels for the final PSF, microns
spectrum = 0.532  # microns
f = 100.0  # focal length, microns
n = 1.33  # refractive index of medium
NA = 0.8  # numerical aperture of objective


def init_microscope(phase):
    microscope = Microscope(
        system_psf=Optical4FSystemPSF(shape=shape, spacing=spacing, phase=phase),
        sensor=BasicSensor(
            shape=shape,
            spacing=spacing,
            resampling_method=None,
            reduce_axis=0,
        ),
        f=f,
        n=n,
        NA=NA,
        spectrum=spectrum,
    )
    return microscope


@jax.jit
def compute_image(microscope, volume, z):
    return microscope(volume, z)


volume = jnp.ones((num_planes, *shape))  # fill in your volume here
z = jnp.linspace(-4, 4, num=num_planes)
phase = jnp.zeros(shape)
microscope = init_microscope(phase)
widefield_image = compute_image(microscope, volume, z)
print(f"image has shape: {widefield_image.shape}")

single_gpu_times = []
for i in range(10):
    print(i)
    start = perf_counter_ns()
    _ = compute_image(microscope, volume, z).block_until_ready()
    end = perf_counter_ns()
    single_gpu_times.append((end - start) / 1e6)

print(f"single gpu: {np.mean(single_gpu_times)} +/- {np.std(single_gpu_times)} ms")


@partial(jax.pmap, axis_name="devices")
def init_microscope(phase):
    microscope = Microscope(
        system_psf=Optical4FSystemPSF(shape=shape, spacing=spacing, phase=phase),
        sensor=BasicSensor(
            shape=shape,
            spacing=spacing,
            resampling_method=None,
            reduce_axis=0,
            reduce_parallel_axis_name="devices",
        ),
        f=f,
        n=n,
        NA=NA,
        spectrum=spectrum,
    )
    return microscope


@partial(jax.pmap, axis_name="devices")
def compute_image(microscope, volume, z):
    return microscope(volume, z)


volume = jnp.ones(
    (num_devices, num_planes_per_device, *shape)
)  # fill in your volume here
volume = jax.device_put_sharded(
    [chunk for chunk in volume], jax.devices()[:num_devices]
)
z = jax.device_put_sharded(
    [
        chunk_z
        for chunk_z in jnp.linspace(-4, 4, num=num_planes).reshape(
            num_devices, num_planes_per_device
        )
    ],
    jax.devices()[:num_devices],
)
phase = jnp.zeros((num_devices, *shape))
microscope = init_microscope(phase)
widefield_image = compute_image(microscope, volume, z)
print(f"image has shape: {widefield_image.shape}")
assert jnp.all(widefield_image[0] == widefield_image[1])

pmap_gpu_times = []
for i in range(10):
    print(i)
    start = perf_counter_ns()
    _ = compute_image(microscope, volume, z).block_until_ready()
    end = perf_counter_ns()
    pmap_gpu_times.append((end - start) / 1e6)

print(f"pmap multi gpu: {np.mean(pmap_gpu_times)} +/- {np.std(pmap_gpu_times)} ms")

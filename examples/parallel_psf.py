from chromatix.elements import ObjectivePointSource, PhaseMask, FFLens
from chromatix import OpticalSystem
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from time import perf_counter_ns

num_devices = 4
num_planes_per_device = 32
num_planes = num_devices * num_planes_per_device
shape = (1536, 1536)  # number of pixels in simulated field
spacing = 0.3  # spacing of pixels for the final PSF, microns
spectrum = 0.532  # microns
spectral_density = 1.0
f = 100.0  # focal length, microns
n = 1.33  # refractive index of medium
NA = 0.8  # numerical aperture of objective
optical_model = OpticalSystem(
    [
        ObjectivePointSource(shape, spacing, spectrum, spectral_density, f, n, NA),
        PhaseMask(jnp.ones(shape)[jnp.newaxis, ..., jnp.newaxis]),
        FFLens(f, n),
    ]
)


@jax.jit
def compute_psf(z):
    return optical_model.apply({}, z).intensity


z = jnp.linspace(-4, 4, num=num_planes)
widefield_psf = compute_psf(z)

single_gpu_times = []
for i in range(10):
    print(i)
    start = perf_counter_ns()
    _ = compute_psf(z).block_until_ready()
    end = perf_counter_ns()
    single_gpu_times.append((end - start) / 1e6)

print(f"single gpu: {np.mean(single_gpu_times)} +/- {np.std(single_gpu_times)} ms")


@jax.pmap
def compute_psf(z):
    return optical_model.apply({}, z).intensity


z = jax.device_put_sharded(
    [
        chunk_z
        for chunk_z in jnp.linspace(-4, 4, num=num_planes).reshape(
            num_devices, num_planes_per_device
        )
    ],
    jax.devices()[:num_devices],
)
widefield_psf = compute_psf(z)

pmap_gpu_times = []
for i in range(10):
    print(i)
    start = perf_counter_ns()
    _ = compute_psf(z).block_until_ready()
    end = perf_counter_ns()
    pmap_gpu_times.append((end - start) / 1e6)

print(f"pmap multi gpu: {np.mean(pmap_gpu_times)} +/- {np.std(pmap_gpu_times)} ms")

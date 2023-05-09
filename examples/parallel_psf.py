from chromatix.elements import ObjectivePointSource, PhaseMask, FFLens
from chromatix import OpticalSystem
import jax
import jax.numpy as jnp
import numpy as np
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
z = jnp.linspace(-4, 4, num=num_planes)  # planes to compute PSF at
optical_model = OpticalSystem(
    [
        ObjectivePointSource(shape, spacing, spectrum, spectral_density, f, n, NA),
        PhaseMask(jnp.ones(shape)),
        FFLens(f, n),
    ]
)
variables = optical_model.init(jax.random.PRNGKey(4), z)


@jax.jit
def compute_psf(z):
    return optical_model.apply(variables, z).intensity


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
    return optical_model.apply(variables, z).intensity


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

# Parallelism

Chromatix is designed to be parallelized, both on a single device as well as
across multiple devices (e.g. multiple GPUs).

What this parallelism looks like is heavily application-dependent, so
`chromatix` does not enforce any particular mode of parallelism, but
is written with the potential for multiple kinds of parallelism through
[``jax.vmap``](https://jax.readthedocs.io/en/latest/_autosummary/
jax.vmap.html#jax.vmap), [``jax.pmap``](https://jax.readthedocs.io/
en/latest/_autosummary/jax.pmap.html#jax.pmap), and distributed
[``jax.Array``](https://jax.readthedocs.io/en/latest/notebooks/
Distributed_arrays_and_automatic_parallelization.html?highlight=sharded).

We describe two major categories of parallelization here, and show some
examples of how specific simulations in `chromatix` might be parallelized.

The first category is explicit parallelization, where we directly specify along
which dimensions a computation will be parallelized. This is useful when we
have some knowledge about what parts of a simulation can be run independently,
and can actually be simpler to reason about and program than implicit
parallelization.

The second category is implicit parallelization across multiple devices,
which allows ``jax.jit`` to automatically parallelize a simulation and has the
advantage of allowing for functions to be written as if the computation were
being performed on a single device after sharding the inputs across multiple
devices. However, this style of parallelization can require some experimentation
to choose the correct initial placement of arrays across multiple devices so
that ``jax.jit`` can choose the optimal parallelization.

## Explicit parallelization

A common style of parallelism is across a batch dimension. Chromatix already
allows ``Field`` objects to have arbitrary batch dimensions. This means that on a single device,
any computations across the batch and wavelength
dimensions are already performed in parallel. It is possible to parallelize
additional dimensions using `jax.vmap` on a single device.

But here, we are concerned with parallelization across multiple devices (e.g.
4 NVIDIA A100 GPUs). For example, we can look at the widefield PSF example from
the README (but adjusted so that the workload is large enough to observe some
benefit from parallelization). Here is the single device version:

```python
import jax
import jax.numpy as jnp
import numpy as np

from chromatix.systems import OpticalSystem
from chromatix.elements import FFLens, ObjectivePointSource, PhaseMask

num_devices = 4
num_planes_per_device = 32
num_planes = num_devices * num_planes_per_device
shape = (1536, 1536)  # number of pixels in simulated field
spacing = 0.3  # spacing of pixels for the final PSF, microns
spectrum = 0.532  # microns
f = 100.0  # focal length, microns
n = 1.33  # refractive index of medium
NA = 0.8  # numerical aperture of objective
z = jnp.linspace(-4, 4, num=num_planes)  # planes to compute PSF at
optical_model = OpticalSystem(
    [
        ObjectivePointSource(shape, spacing, spectrum, f, n, NA),
        PhaseMask(jnp.ones(shape)),
        FFLens(f, n),
    ]
)


@jax.jit
def compute_psf(z):
    return optical_model(z).intensity


widefield_psf = compute_psf(z)
```

This takes **25.06ms** to run on an NVIDIA A100 GPU (averaged over 10 runs, not
including the time to JIT compile the function).

We can explicitly parallelize on multiple devicess by running different chunks of
the batch dimension (which in this simulation represents depth) on different
devices, because we know each plane can be simulated independently. We can do this
using ``jax.pmap``:

```python
@jax.pmap
def compute_psf(z):
    return optical_model(z).intensity


# Place our input (z) on multiple devices
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
```

This takes **6.45ms** to run on 4 NVIDIA A100 GPUs (averaged over 10 runs). This
means we have scaled in speed almost linearly with the number of devices, which
makes sense because this computation is totally independent.

Note that we did not have to change anything in the definition of the optical
system in order to obtain this parallelization! We were able to simply
transform the function that computes the intensity and correspondingly change
how we pass the ``z`` values that we want to simulate. The point of the
reshaping is to allow ``jax.pmap`` to map the function ``compute_psf`` over
``z`` values in chunks of 32 across all 4 devices to simulate the same 128 planes
of the PSF.

Let's look at an example where the computation isn't completely independent
across multiple devices: simulating a PSF and then simulating imaging with this
PSF. We can also show how to do this when there are parameters that must be
initialized on each device; in this case, we initialize the phase mask pixels of
the 4f system on each device. First, let's look at the single device version:

```python
from functools import partial

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
        system_psf=Optical4FSystemPSF(
            shape=shape, spacing=spacing, phase=phase
        ),
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
```

Here, we constructed a ``Microscope`` with a 4f system PSF, but this time
we initialized the ``Microscope`` on each device with its own ``phase``
parameter. Note that this ``Microscope`` also accepted a ``BasicSensor`` with
a ``reduce_axis`` argument, which we have specified to sum across the batch
dimension (axis 0) to simulate a camera collecting light from multiple planes.
This computation ran in **172.86ms** on a single NVIDIA A100 GPU (average over
10 runs).

Just like last time, we can parallelize this to multiple devices along the
batch dimension by using ``jax.pmap``:

```python
@partial(jax.pmap, axis_name="devices")
def init_microscope(phase):
    microscope = Microscope(
        system_psf=Optical4FSystemPSF(
            shape=shape, spacing=spacing, phase=phase
        ),
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
volume = jax.device_put_sharded([chunk for chunk in volume], jax.devices()[:num_devices])
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
```

This time, we ended up having to make a change to how we define the optics. That
is because each device first computes a partial image of just the chunk that it
received, which are summed across the batch dimension on each device (because we
specified ``reduce_axis``). So, we need to make sure that we are summing these
partial images together across all the devices. We can tell `jax` to do that by
using ``jax.lax.psum``, which happens internally in the ``BasicSensor``
because we specified ``reduce_parallel_axis_name`` in addition to
``reduce_axis``. Now, each device has a copy of the same final image.

That means that if we look at the shape of ``widefield_image``, we'll see that
it has shape `(4 1536 1536)` because we ran on 4 devices. Each of those 4 2D
images is identical. However, because we computed these images in parallel, this
version ran in **51.71ms** on 4 NVIDIA A100 GPUs (average over 10 runs). Because
of the fact that we have to sum the image across all the devices, you can see
this computation does not scale as well (though is still many times faster than
the single device version).
 
## Implicit parallelization

Using [``jax.Array``](https://jax.readthedocs.io/en/latest/notebooks/
Distributed_arrays_and_automatic_parallelization.html?highlight=sharded), it is
possible to have ``jax.jit`` handle the parallelism across multiple devices.

This requires that you specify how `jax` should split up any input arrays
across multiple devices (this is referred to as "sharding" an array). Then, any
function written as if for a single device will be automatically parallelized
across multiple devices when it is compiled through ``jax.jit`` with a sharded
input array.

For now, we still recommend explicitly parallelizing your simulations for
maximum performance.

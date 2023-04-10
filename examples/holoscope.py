import jax.numpy as jnp
from jax import jit
from jax import random
from jax.tree_util import tree_map

from chromatix.systems import Microscope, Optical4FSystemPSF
from chromatix.utils import trainable
from chromatix.functional.phase_masks import defocused_ramps

holoscope = Microscope(
    system_psf=Optical4FSystemPSF(
        shape=(2560, 2560),
        spacing=0.325,
        phase=trainable(defocused_ramps),
    ),
    sensor_shape=(512, 512),
    sensor_spacing=1.625,
    f=100.0,
    n=1.33,
    NA=0.8,
    spectrum=0.532,
    spectral_density=1.0,
    padding_ratio=0.5,
    taper_width=5.0,
    reduce_axis=0,
    shot_noise_mode="poisson",
    psf_resampling_method="pool",
)

key = random.PRNGKey(42)
z = jnp.linspace(-125, 125, 5)
data = random.normal(key, (z.size, 512, 512, 1, 1))
params = holoscope.init({"params": key, "noise": key}, data, z)
forward = jit(holoscope.apply)

assert jnp.allclose(
    holoscope.apply(params, z, method=holoscope.psf).dx.squeeze(), 0.325
)
print(forward(params, data, z, rngs={"noise": key}).shape)
print(tree_map(lambda x: x.shape, params))

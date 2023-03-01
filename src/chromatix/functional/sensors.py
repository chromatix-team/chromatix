from jax import vmap
from jax.lax import psum
import jax.numpy as jnp
from chex import PRNGKey, Array
from typing import Literal, Optional
from ..ops import fourier_convolution
from ..ops.noise import approximate_shot_noise, shot_noise


def shift_invariant_sensor(
    sample: Array,
    psf: Array,
    noise_key: Optional[PRNGKey] = None,
    shot_noise_mode: Optional[Literal['approximate', 'poisson']] = None,
    reduce_batch: bool = True,
    parallel_axis_name: Optional[str] = None
) -> Array:
    image = vmap(fourier_convolution, in_axes=(0, 0))(sample, psf)
    if reduce_batch:
        image = jnp.sum(image, axis=0)
    if parallel_axis_name is not None:
        image = psum(image, axis_name=parallel_axis_name)
    if noise_key is not None:
        if shot_noise_mode == 'approximate':
            image = approximate_shot_noise(noise_key, image)
        elif shot_noise_mode == 'poisson':
            image = shot_noise(noise_key, image)
    return image

import jax.numpy as jnp
import flax.linen as nn
from jax import vmap
from jax.lax import psum
from typing import Literal, Optional, Tuple
from chex import PRNGKey, Array
from chromatix import Field
from ..ops import init_plane_resample, approximate_shot_noise, shot_noise


class NoisyIntensitySensor(nn.Module):
    shape: Tuple[int, ...]
    spacing: float
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None
    resampling_method: Optional[str] = "linear"
    reduce_axis: Optional[int] = None
    reduce_parallel_axis_name: Optional[str] = None

    def setup(self):
        if self.resampling_method is not None:
            self.resample = init_plane_resample(
                (*self.shape, 1), self.spacing, self.resampling_method
            )

    def __call__(self, field: Field) -> Array:
        if self.shot_noise_mode is not None:
            noise_key = self.make_rng("noise")
        if self.resampling_method is not None:
            image = vmap(self.resample, in_axes=(0, None))(field.intensity, field.dx)
        if self.reduce_axis is not None:
            image = jnp.sum(image, axis=self.reduce_axis)
        if self.reduce_parallel_axis_name is not None:
            image = psum(image, axis_name=self.reduce_parallel_axis_name)
        if self.shot_noise_mode == "approximate":
            image = approximate_shot_noise(noise_key, image)
        elif self.shot_noise_mode == "poisson":
            image = shot_noise(noise_key, image)
        return image

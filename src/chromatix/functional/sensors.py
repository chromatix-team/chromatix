import jax.numpy as jnp
from jax import vmap
from jax.lax import psum
from chex import PRNGKey, Array
from typing import Callable, Optional, Literal

from ..field import Field
from ..ops import approximate_shot_noise, shot_noise

__all__ = ["shot_noise_intensity_sensor"]


def shot_noise_intensity_sensor(
    field: Field,
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None,
    resample: Optional[Callable[[Array, float], Array]] = None,
    reduce_axis: Optional[int] = None,
    reduce_parallel_axis_name: Optional[str] = None,
    noise_key: Optional[PRNGKey] = None,
) -> Field:
    if resample is not None:
        image = vmap(resample, in_axes=(0, None))(
            field.intensity, field.dx[..., 0].squeeze()
        )
    else:
        image = field.intensity
    if reduce_axis is not None:
        image = jnp.sum(image, axis=reduce_axis, keepdims=True)
    if reduce_parallel_axis_name is not None:
        image = psum(image, axis_name=reduce_parallel_axis_name)
    if shot_noise_mode == "approximate":
        image = approximate_shot_noise(noise_key, image)
    elif shot_noise_mode == "poisson":
        image = shot_noise(noise_key, image)
    return image

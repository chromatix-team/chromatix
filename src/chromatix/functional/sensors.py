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
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.

    Attributes:
        shape: The shape in pixels of the sensor. Should be of the form (H W).
        spacing: The pitch of the sensor pixels.
        shot_noise_mode: What type of shot noise simulation to use. Defaults to
            None, in which case no shot noise is simulated.
        resample: If provided, will be called to resample the incoming
            ``Field`` to the given ``shape``.
        reduce_axis: If provided, the result will be summed along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
    """
    if resample is not None:
        for i in range(field.ndim - 4):
            resample = vmap(resample, in_axes=(0, None))
        image = resample(field.intensity, field.dx[..., 0, 0].squeeze())
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

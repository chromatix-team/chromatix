import jax.numpy as jnp
from jax import vmap
from jax.lax import psum
from chex import PRNGKey, Array
from typing import Callable, Optional, Literal, Union
from ..field import Field
from ..ops import approximate_shot_noise, shot_noise

__all__ = ["shot_noise_intensity_sensor"]


def shot_noise_intensity_sensor(
    sensor_input: Union[Field, Array],
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None,
    resample: Optional[Callable[[Array, float], Array]] = None,
    reduce_axis: Optional[int] = None,
    reduce_parallel_axis_name: Optional[str] = None,
    input_spacing: Optional[float] = None,
    noise_key: Optional[PRNGKey] = None,
) -> Array:
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.
    Optionally, can also accept an intensity Array if ``input_spacing`` is
    specified.

    Args:
        sensor_input: Either the incoming ``Field`` or an intensity ``Array`` to
            be sampled by the sensor with shot noise.
        shot_noise_mode: What type of shot noise simulation to use. Defaults to
            None, in which case no shot noise is simulated.
        resample: If provided, will be called to resample the incoming
            ``Field`` to the given ``shape``.
        reduce_axis: If provided, the result will be summed along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
        input_spacing: Only needs to be provided if ``sensor_input`` is an
            intensity ``Array`` and not a ``Field``. If provided, defines the
            spacing of the input to be used for resampling by the sensor.
        noise_key: If provided, will be used to generate the shot noise.
    """
    if isinstance(sensor_input, Field):
        intensity = sensor_input.intensity
        spacing = sensor_input.dx[..., 0, 0].squeeze()
    else:
        assert input_spacing is not None, "Must provide input_spacing for intensity"
        intensity = sensor_input
    if resample is not None:
        for i in range(sensor_input.ndim - 4):
            resample = vmap(resample, in_axes=(0, None))
        image = resample(intensity, spacing)
    else:
        image = intensity
    if reduce_axis is not None:
        image = jnp.sum(image, axis=reduce_axis, keepdims=True)
    if reduce_parallel_axis_name is not None:
        image = psum(image, axis_name=reduce_parallel_axis_name)
    if shot_noise_mode == "approximate":
        image = approximate_shot_noise(noise_key, image)
    elif shot_noise_mode == "poisson":
        image = shot_noise(noise_key, image)
    return image

import jax.numpy as jnp
from jax import vmap
from jax.lax import psum
from chex import PRNGKey, Array
from typing import Callable, Optional, Literal, Union
from ..field import Field
from ..ops import approximate_shot_noise, shot_noise

__all__ = ["basic_shot_noise_sensor"]


def basic_shot_noise_sensor(
    sensor_input: Union[Field, Array],
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None,
    resample_fn: Optional[Callable[[Array, float], Array]] = None,
    reduce_axis: Optional[int] = None,
    reduce_parallel_axis_name: Optional[str] = None,
    input_spacing: Optional[Union[float, Array]] = None,
    noise_key: Optional[PRNGKey] = None,
) -> Array:
    """
    Produces an intensity image from an incoming ``Field`` or intensity
    ``Array`` and simulates shot noise.

    Args:
        sensor_input: Either the incoming ``Field`` or an intensity ``Array`` to
            be sampled by the sensor with shot noise.
        shot_noise_mode: What type of shot noise simulation to use. Can be
            either "approximate", "poisson", or None. Defaults to None, in which
            case no shot noise is simulated (the sensor is perfect).
        resample_fn: If provided, will be called to resample the incoming
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
        # WARNING(dd): @copypaste(Microscope) Assumes that field has same
        # spacing at all wavelengths when calculating intensity!
        input_spacing = sensor_input.dx[..., 0, 0].squeeze()
    else:
        intensity = sensor_input
    if resample_fn is not None:
        assert input_spacing is not None, "Must provide input_spacing for intensity"
    if resample_fn is not None:
        for i in range(sensor_input.ndim - 4):
            resample_fn = vmap(resample_fn, in_axes=(0, None))
        image = resample_fn(intensity, input_spacing)
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

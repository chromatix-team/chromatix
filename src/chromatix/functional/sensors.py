from typing import Literal

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, ScalarLike

from chromatix import Field, Resampler
from chromatix.ops.noise import approximate_shot_noise, shot_noise

__all__ = ["basic_sensor"]


def basic_sensor(
    sensor_input: Field | Array,
    shot_noise_mode: Literal["approximate", "poisson"] | None = None,
    resampler: Resampler | None = None,
    reduce_axis: int | None = None,
    reduce_parallel_axis_name: str | None = None,
    input_spacing: ScalarLike | None = None,
    noise_key: PRNGKey | None = None,
) -> Array:
    """
    Produces an intensity image from an incoming ``Field`` or intensity
    ``Array`` and simulates shot noise. In most cases, it is better and easier
    to use the [BasicSensor](``chromatix.elements.sensors.BasicSensor``) element
    which will handle some of the state/arguments necessary for this function
    (especially for resampling to the pixel size of the sensor).

    !!! warning
        Assumes that the input has the same spacing at all wavelengths!

    Args:
        sensor_input: Either the incoming ``Field`` or an intensity ``Array`` to
            be sampled by the sensor with shot noise.
        shot_noise_mode: What type of shot noise simulation to use. Can be
            either ``"approximate"``, ``"poisson"``, or ``None``. Defaults to
            ``None``, in which case no shot noise is simulated (the sensor is
            perfect).
        resampler: If provided, will be called to resample the
            incoming ``Field`` to the given ``shape``. These are
            instances of ``Resampler``s which can be created using
            [``chromatix.ops.resample.init_plane_resample``](chromatix.ops.resam
            ple.init_plane_resample).
        reduce_axis: If provided, the result will be summed along this
            dimension. Useful for simulating multiple depth planes being summed
            at the sensor plane.
        reduce_parallel_axis_name: If provided, will do a ``jax.lax.psum `` to
            perform a sum across multiple devices along the pmapped axis with
            this name. Useful when simulating multiple planes reaching the
            sensor when those planes are sharded across multiple GPUs/TPUs/etc.
            using ``jax.lax.pmap``. Note that in that case you will likely want
            to use both ``reduce_axis`` and ``reduce_parallel_axis_name``.
        input_spacing: Only needs to be provided if ``sensor_input`` is an
            intensity ``Array`` and not a ``Field``. If provided, defines the
            spacing of the input in units of distance to be used for resampling
            by the sensor.
        noise_key: If provided, will be used to generate the shot noise. If
            ``shot_noise_mode`` is not ``None``, this must be provided.

    Returns:
        The measured intensity of the incoming ``Field`` (potentially summed
        across any depth/batch axis of the incoming ``Field``) at the sensor
        plane.
    """
    if isinstance(sensor_input, Field):
        intensity = sensor_input.intensity
        # WARNING(dd): @copypaste(Microscope) Assumes that field has same
        # spacing at all wavelengths when calculating intensity!
        input_spacing = sensor_input.central_rectangular_dx
    else:
        intensity = sensor_input
    if resampler is not None:
        assert input_spacing is not None, (
            "Must provide input_spacing for intensity array"
        )
    if (resampler is not None) and (input_spacing is not None):
        for i in range(intensity.ndim - 2):
            resampler = jax.vmap(resampler, in_axes=(0, None))
        image = resampler(intensity, jnp.atleast_1d(input_spacing))
    else:
        image = intensity
    if reduce_axis is not None:
        image = jnp.sum(image, axis=reduce_axis)
    if reduce_parallel_axis_name is not None:
        image = jax.lax.psum(image, axis_name=reduce_parallel_axis_name)
    if shot_noise_mode is not None:
        assert noise_key is not None, "Must provide a PRNG key to generate noise"
    if shot_noise_mode == "approximate":
        image = approximate_shot_noise(noise_key, image)
    elif shot_noise_mode == "poisson":
        image = shot_noise(noise_key, image)
    return image

import jax.numpy as jnp
from jax import vmap
import flax.linen as nn
from typing import Optional, Literal, Tuple, Union
from chex import Array
from ..field import Field
from ..ops import init_plane_resample
from ..functional import basic_shot_noise_sensor

__all__ = ["BasicShotNoiseSensor"]


class BasicShotNoiseSensor(nn.Module):
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.
    Optionally, can also accept an intensity Array if ``input_spacing`` is
    specified.

    Attributes:
        shape: The shape in pixels of the sensor. Should be of the form (H W).
        spacing: The pitch of the sensor pixels.
        shot_noise_mode: What type of shot noise simulation to use. Defaults to
            None, in which case no shot noise is simulated.
        resampling_method: What kind of sampling to use when resampling the
            incoming ``Field`` to the pitch of the sensor.
        reduce_axis: If provided, the result will be summed along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
    """

    shape: Tuple[int, int]
    spacing: float
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None
    resampling_method: Optional[str] = "linear"
    reduce_axis: Optional[int] = None
    reduce_parallel_axis_name: Optional[str] = None

    def setup(self):
        if self.resampling_method is not None:
            self.resample_fn = init_plane_resample(
                (*self.shape, 1, 1), self.spacing, self.resampling_method
            )

    def __call__(
        self, sensor_input: Union[Field, Array], input_spacing: Optional[float] = None
    ) -> Array:
        if isinstance(sensor_input, Field):
            # WARNING(dd): @copypaste(Microscope) Assumes that field has same
            # spacing at all wavelengths when calculating intensity, and also
            # that spacing is square!
            input_spacing = sensor_input.dx[..., 0, 0].squeeze()
        input_spacing = jnp.atleast_1d(input_spacing)[0]
        # Only want to resample if the spacing does not match
        if self.resampling_method is not None and input_spacing != self.spacing:
            resample_fn = self.resample_fn
        else:
            resample_fn = None
        if self.shot_noise_mode is not None:
            noise_key = self.make_rng("noise")
        else:
            noise_key = None
        return basic_shot_noise_sensor(
            sensor_input,
            self.shot_noise_mode,
            resample_fn,
            self.reduce_axis,
            self.reduce_parallel_axis_name,
            input_spacing=input_spacing,
            noise_key=noise_key,
        )

    def resample(self, resample_input: Array, input_spacing: float) -> Array:
        if self.resampling_method is not None:
            resample_fn = self.resample_fn
            for i in range(resample_input.ndim - 4):
                resample_fn = vmap(resample_fn, in_axes=(0, None))
            return resample_fn(resample_input, input_spacing)
        else:
            return resample_input

import flax.linen as nn
from typing import Literal, Optional, Tuple
from chex import Array
from ..field import Field
from ..ops import init_plane_resample
from ..functional import shot_noise_intensity_sensor


class ShotNoiseIntensitySensor(nn.Module):
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.

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
            self.resample = init_plane_resample(
                (*self.shape, 1), self.spacing, self.resampling_method
            )

    def __call__(self, field: Field) -> Array:
        if self.resampling_method is not None:
            resample = self.resample
        else:
            resample = None
        if self.shot_noise_mode is not None:
            noise_key = self.make_rng("noise")
        else:
            noise_key = None
        return shot_noise_intensity_sensor(
            field,
            self.shot_noise_mode,
            resample,
            self.reduce_axis,
            self.reduce_parallel_axis_name,
            noise_key=noise_key,
        )

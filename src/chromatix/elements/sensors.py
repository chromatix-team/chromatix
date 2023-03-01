import flax.linen as nn
from typing import Literal, Optional
from chex import PRNGKey, Array
from chromatix import Field
from ..functional.sensors import shift_invariant_sensor


class ShiftInvariantSensor(nn.Module):
    pixel_pitch: float # TODO(dd): Resample to pixel pitch if needed
    shot_noise_mode: Optional[Literal['approximate', 'poisson']] = None
    reduce_batch: bool = True
    parallel_axis_name: Optional[str] = None

    @nn.compact
    def __call__(self, sample: Array, psf: Array) -> Array:
        noise_key = (
            None
            if not self.shot_noise_mode
            else self.make_rng("noise")
        )
        image = shift_invariant_sensor(
            sample,
            psf,
            noise_key,
            self.shot_noise_mode,
            self.reduce_batch,
            self.parallel_axis_name
        )
        return image

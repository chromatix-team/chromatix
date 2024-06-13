from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from jax import Array, vmap

from chromatix.typing import ArrayLike, ScalarLike

from ..field import Field, ScalarField, VectorField
from ..functional import basic_sensor
from ..ops import init_plane_resample

__all__ = ["BasicSensor"]


class BasicSensor(nn.Module):
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.
    Optionally, can also accept an intensity Array if ``input_spacing`` is
    specified.

    Attributes:
        shape: The shape in pixels of the sensor. Should be of the form `(H W)`.
        spacing: The pitch of the sensor pixels.
        shot_noise_mode: What type of shot noise simulation to use. Defaults to
            None, in which case no shot noise is simulated.
        resampling_method: What kind of sampling to use when resampling the
            incoming ``Field`` to the pitch of the sensor. Can be either
            `'pooling'` which uses sum pooling (for downsampling only) or any
            method supported by ``jax.image.scale_and_translate`` (`'linear'`,
            `'cubic'`, `'lanczos3'`, or `'lanczos5'`). If ``None``, then no
            resampling will occur.
        reduce_axis: If provided, the result will be summed along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
    """

    shape: tuple[int, int]
    spacing: ArrayLike
    shot_noise_mode: Literal["approximate", "poisson"] | None = None
    resampling_method: str | None = "linear"
    reduce_axis: int | None = None
    reduce_parallel_axis_name: str | None = None

    def setup(self):
        if self.resampling_method is not None:
            self.resample_fn = init_plane_resample(
                self.shape, self.spacing, self.resampling_method
            )

    def __call__(
        self,
        sensor_input: Array | Field,
        input_spacing: ScalarLike | None = None,
        resample: bool = True,
    ) -> Array:
        """
        Resample the given ``sensor_input`` to the pixels of the sensor and
        potentially reduce the result and add shot noise.

        Args:
            sensor_input: The incoming ``Field`` or intensity ``Array``.
            input_spacing: The spacing of the input, only required if resampling
                is required and the input is an ``Array``.
            resample: Whether to perform resampling or not. Only matters if
                ``resampling_method`` is ``None``. Defaults to ``True``.
        """

        if (input_spacing is not None) and isinstance(
            sensor_input, (ScalarField, VectorField)
        ):
            # WARNING(dd): @copypaste(Microscope) Assumes that field has same
            # spacing at all wavelengths when calculating intensity!
            input_spacing = sensor_input.dx[..., 0, 0].squeeze()
        elif input_spacing is not None:
            input_spacing = jnp.atleast_1d(input_spacing)
        if resample and self.resampling_method is not None:
            resample_fn = self.resample_fn
        else:
            resample_fn = None
        if self.shot_noise_mode is not None:
            noise_key = self.make_rng("noise")
        else:
            noise_key = None
        return basic_sensor(
            sensor_input,
            self.shot_noise_mode,
            resample_fn,
            self.reduce_axis,
            self.reduce_parallel_axis_name,
            input_spacing=input_spacing,
            noise_key=noise_key,
        )

    def resample(self, resample_input: Array, input_spacing: ScalarLike) -> Array:
        """
        Resample the given ``resample_input`` to the pixels of the sensor.

        Args:
            resample_input: The ``Array`` to resample of shape ``(B... H W 1 1)``
            input_spacing: The spacing of the input
        """
        if self.resampling_method is not None:
            resample_fn = self.resample_fn
            for i in range(resample_input.ndim - 4):
                resample_fn = vmap(resample_fn, in_axes=(0, None))
            return resample_fn(resample_input, input_spacing)
        else:
            return resample_input

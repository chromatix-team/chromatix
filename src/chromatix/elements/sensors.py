from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray, ScalarLike

from chromatix import Field, Resampler, Sensor
from chromatix.functional import basic_sensor
from chromatix.ops import init_plane_resample

__all__ = ["BasicSensor"]


class BasicSensor(Sensor):
    """
    Produces an intensity image from an incoming ``Field`` with shot noise.
    Optionally, can also accept an intensity Array if ``input_spacing`` is
    specified.

    Attributes:
        shape: The shape in pixels of the sensor. Should be a tuple of the form
            `height width`.
        spacing: The pitch of the sensor pixels.
        shot_noise_mode: What type of shot noise simulation to use. Defaults to
            ``None``, in which case no shot noise is simulated.
        resampling_method: What kind of sampling to use when resampling the
            incoming ``Field`` to the pitch of the sensor. Can be either
            `'pooling'` which uses sum pooling (for downsampling only) or any
            method supported by ``jax.image.scale_and_translate`` (`'linear'`,
            `'cubic'`, `'lanczos3'`, or `'lanczos5'`). If ``None``, then no
            resampling will occur.
        reduce_axis: If provided, the result will be summed along this
            dimension. Useful for simulating multiple depth planes being summed
            at the sensor plane.
        reduce_parallel_axis_name: If provided, will do a ``jax.lax.psum `` to
            perform a sum across multiple devices along the pmapped axis with
            this name. Useful when simulating multiple planes reaching the
            sensor when those planes are sharded across multiple GPUs/TPUs/etc.
            using ``jax.lax.pmap``.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    spacing: ScalarLike | Float[Array, "2"]
    shot_noise_mode: Literal["approximate", "poisson"] | None = eqx.field(static=True)
    resampling_method: str | None = eqx.field(static=True)
    resampler: Resampler | None = eqx.field(static=True)
    reduce_axis: int | None = eqx.field(static=True)
    reduce_parallel_axis_name: str | None = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: ScalarLike | Float[Array, "2"],
        shot_noise_mode: Literal["approximate", "poisson"] | None = None,
        resampling_method: str | None = "linear",
        reduce_axis: int | None = None,
        reduce_parallel_axis_name: str | None = None,
    ):
        self.shape = shape
        self.spacing = spacing
        self.shot_noise_mode = shot_noise_mode
        self.resampling_method = resampling_method
        if self.resampling_method is not None:
            self.resampler = init_plane_resample(
                self.shape, self.spacing, self.resampling_method
            )
        else:
            self.resampler = None
        self.reduce_axis = reduce_axis
        self.reduce_parallel_axis_name = reduce_parallel_axis_name

    def __call__(
        self,
        sensor_input: Field | Float[Array, "... h w"],
        input_spacing: ScalarLike | None = None,
        resample: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        """
        Resample the given ``sensor_input`` to the pixels of the sensor and
        potentially reduce the result and add shot noise.

        !!! warning
            Assumes that the input has the same spacing at all wavelengths!

        Args:
            sensor_input: The incoming ``Field`` or intensity ``Array`` of shape
                ``(... height width)``.
            input_spacing: The spacing of the input, only required if resampling
                is desired and the input is an ``Array``.
            resample: Whether to perform resampling or not. Only matters if
                ``resampling_method`` is not ``None``. Defaults to ``True``.
            key: If provided, will be used to generate the shot noise. If
                ``shot_noise_mode`` is not ``None``, this must be provided.
        """
        if resample and self.resampling_method is not None:
            resampler = self.resampler
        else:
            resampler = None
        if self.shot_noise_mode is not None:
            key is not None, ("Must provide a PRNGKey if shot_noise_mode is not None")
            _, noise_key = jax.random.split(key)
        else:
            noise_key = None
        return basic_sensor(
            sensor_input,
            self.shot_noise_mode,
            resampler,
            self.reduce_axis,
            self.reduce_parallel_axis_name,
            input_spacing=input_spacing,
            noise_key=noise_key,
        )

    def resample(self, resample_input: Array, input_spacing: ScalarLike) -> Array:
        """
        Resample the given ``resample_input`` to the pixels of the sensor.

        Args:
            resample_input: The ``Array`` to resample of shape `(... height
                width)` (resampling will occur along the last two dimensions,
                with all other dimensions treated as batch dimensions).
            input_spacing: The spacing of the input pixels as a scalar in units
                of distance.
        """
        if self.resampling_method is not None:
            resampler = self.resampler
            for i in range(resample_input.ndim - 2):
                resampler = jax.vmap(resampler, in_axes=(0, None))
            return resampler(resample_input, input_spacing)
        else:
            return resample_input

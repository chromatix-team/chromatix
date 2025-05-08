import abc
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import assert_equal_shape
from chromatix import ScalarField
from einops import rearrange
from jax import Array


from chromatix.experimental.diff_xnh.magnification import magnification
from chromatix.experimental.diff_xnh.radon import radon
from chromatix.experimental.diff_xnh.shift import apply_shift
from chromatix.experimental.diff_xnh.rotate import rotate_volume

radians = Array


class AbstractSample(eqx.Module):
    @abc.abstractmethod
    def project(self, angle: float | None = None) -> Array:
        pass

    @abc.abstractmethod
    def rotate(self, rotation: radians, scale: float | None = None) -> Self:
        pass

    @abc.abstractmethod
    def scale(self, scale: float, n_out: int | None = None) -> Self:
        pass

    @abc.abstractmethod
    def shift(self, scale: Array) -> Self:
        pass

    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass


class Sample(AbstractSample):
    delta: Array  # phase change
    beta: Array  # attenuation
    thickness: Array

    def __init__(self, delta: Array, beta: Array, thickness: float):
        assert_equal_shape([delta, beta])
        self.thickness = thickness
        match beta.ndim:
            case 2:
                self.beta = rearrange(beta, "h w -> 1 h w 1 1")
                self.delta = rearrange(delta, "h w -> 1 h w 1 1")
            case 3:
                self.beta = rearrange(beta, "d h w -> d h w 1 1")
                self.delta = rearrange(delta, "d h w -> d h w 1 1")
            case 5:
                self.beta = beta
                self.delta = delta
            case _:
                raise NotImplementedError()

    def project(self, angle: float | None):
        """Uses Radon transform"""
        if angle is not None:
            # TODO: Radon transform over first and last axes
            # Radon is over last two axis, so we switch y and z.
            data = jnp.squeeze(self.delta - 1j * self.beta, (-2, -1)).swapaxes(0, 1)
            projection = radon(data, angle)[None, ..., None, None]
        else:
            projection = jnp.sum(self.delta - 1j * self.beta, axis=0, keepdims=True)

        return self.thickness * projection

    def rotate(self, rotation: radians, scale: float = 1.0):
        volume = jnp.stack([self.delta, self.beta]).squeeze()
        rotated = jax.vmap(rotate_volume, in_axes=(0, None, None))(
            volume, rotation, scale
        )
        return Sample(*(rotated[..., None, None]), self.thickness)

    def scale(self, scale: float, n_out: int | None = None):
        data = jnp.squeeze(
            (self.delta + 1j * self.beta), (-2, -1)
        )  # get rid of empty axes
        scaled = jax.vmap(lambda volume: magnification(volume, scale, n_out))(
            data
        )
        return Sample(scaled.real, scaled.imag, self.thickness)

    def shift(self, shift: Array):
        data = jnp.squeeze(
            self.delta + 1j * self.beta, (-2, -1)
        )  # get rid of empty axes
        shifted = jax.vmap(apply_shift, in_axes=(0, None))(data, shift)
        return Sample(shifted.real, shifted.imag, self.thickness)

    @property
    def shape(self):
        return self.delta.shape


def empty_sample(size: int, thickness: float = 1.0) -> Sample:
    """Create an empty Sample with delta=beta=0 of same shape as input."""
    empty = jnp.zeros((size, size), dtype=jnp.float32)
    return Sample(empty, empty, thickness)


def thin_sample(
    field: ScalarField, sample: Sample, angle: float, scale: float
) -> ScalarField:
    """See paganin for details."""
    projection = sample.project(angle)
    scaled_projection = magnification(
        projection.squeeze(), scale, field.shape[-3]
    )[None, :, :, None, None]
    return field * jnp.exp(-1j * 2 * jnp.pi / field.spectrum * scaled_projection)


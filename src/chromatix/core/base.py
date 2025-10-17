import abc
from enum import IntEnum
from typing import TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float, ScalarLike

from chromatix.core.spectrum import MonoSpectrum, Spectrum

__all__ = [
    "Monochromatic",
    "Chromatic",
    "Scalar",
    "Vector",
    "Sample",
    "Absorbing",
    "Scattering",
    "Fluorescent",
    "Volume",
    "Sensor",
    "Resampler",
]


_strict_config = eqx.StrictConfig(allow_abstract_name=True)
# NOTE(dd/2025-09-29): This is a placeholder type annotation to prevent circular
# imports. Some of the interfaces defined here will operate on Fields, and it
# is useful to support type annotations for those functions. However, we can't
# import the actual Field class because that would cause a circular import.
Field = TypeVar("Field")


def replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    for key, value in kwargs.items():
        tree = eqx.tree_at(lambda t: getattr(t, key), tree, value)
    return tree


class Monochromatic(eqx.Module, strict=_strict_config):
    spectrum: eqx.AbstractVar[MonoSpectrum]

    @property
    def wavelength(self) -> MonoSpectrum:
        return self.spectrum


class Chromatic(eqx.Module, strict=_strict_config):
    spectrum: eqx.AbstractVar[Spectrum]


class Scalar(eqx.Module, strict=_strict_config):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Spectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        total_intensity = self.spectrum.density * jnp.sum(
            jnp.abs(self.u) ** 2, axis=(self.dims.p, self.dims.y, self.dims.x)
        )
        return area * total_intensity


class Vector(eqx.Module, strict=_strict_config):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Spectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def jones_vector(self) -> Array:
        norm = jnp.linalg.norm(self.u, axis=self.dims.p, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # Set to 1 to avoid division by zero
        return self.u / norm

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        total_intensity = self.spectrum.density * jnp.sum(
            jnp.abs(self.u) ** 2, axis=(self.dims.p, self.dims.y, self.dims.x)
        )
        return area * total_intensity

    @property
    def wavelength(self) -> Array:
        return rearrange(self.spectrum.wavelength, "wv -> wv 1")

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... wv -> ... 1 1 wv")
        return spectral_density * jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)


class Sample(eqx.Module, strict=_strict_config):
    dx: eqx.AbstractVar[ScalarLike | Float[Array, "2"]]
    thickness: eqx.AbstractVar[ScalarLike | Array]

    def _verify_matching_spacing(self, field: Field):
        # NOTE(dd/2025-07-17): A field may have different spacings for each
        # wavelength of its spectrum along with different numbers of dimensions
        # depending on the type of the field, so this subtraction allows us
        # to broadcast all the potential spacings from the field to the single
        # spacing of the sample (just in case). This is why we don't just do a
        # simple equality check.
        assert jnp.all((field.dx - self.dx) == 0), (
            "Incoming field must have same sampling as sample"
        )

    def _verify_scalar_thickness(self):
        assert np.asarray(self.thickness).size == 1, (
            "Thickness must be a scalar (uniform thickness of each plane of the sample)"
        )

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def grid(self) -> Array:
        pass

    @abc.abstractmethod
    def __call__(self, field: Field) -> Field:
        pass


class Absorbing(eqx.Module, strict=_strict_config):
    absorption: eqx.AbstractVar[Array]


class Scattering(eqx.Module, strict=_strict_config):
    dn: eqx.AbstractVar[Array]


class Fluorescent(eqx.Module, strict=_strict_config):
    fluorescence: eqx.AbstractVar[Array]


class Volume(eqx.Module, strict=_strict_config):
    @property
    @abc.abstractmethod
    def num_planes(self) -> int:
        pass


class Sensor(eqx.Module, strict=_strict_config):
    shape: eqx.AbstractVar[tuple[int, ...]]
    spacing: eqx.AbstractVar[Array]

    @abc.abstractmethod
    def __call__(self, sensor_input: Field | Array) -> Array:
        pass

    @abc.abstractmethod
    def resample(self, resample_input: Array, input_spacing: ScalarLike) -> Array:
        pass


class Resampler(eqx.Module, strict=_strict_config):
    out_shape: eqx.AbstractVar[tuple[int, ...]]
    out_spacing: eqx.AbstractVar[Array]

    @abc.abstractmethod
    def __call__(self, resample_input: Array, in_spacing: Array) -> Array:
        pass

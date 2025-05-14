import abc
from enum import IntEnum
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float
from spectrum import MonochromaticSpectrum, PolyChromaticSpectrum


class AbstractField(eqx.Module, strict=True):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]

    # Internal for use
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    @abc.abstractmethod
    def grid(self) -> Array:
        pass

    @property
    @abc.abstractmethod
    def f_grid(self) -> Array:
        pass

    @property
    def k_grid(self) -> Float[Array, "y x d"]:
        return 2 * jnp.pi * self.f_grid

    @property
    def k0(self) -> Array:
        return 2 * jnp.pi / self.wavelength

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return (self.u.shape[self.dims.y], self.u.shape[self.dims.x])

    @property
    def dk(self) -> Array:
        return 1 / (self.dx * jnp.asarray(self.spatial_shape))

    @property
    def surface_area(self) -> Array:
        shape = jnp.array(self.spatial_shape)
        return self.dx * shape

    @property
    def phase(self) -> Array:
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> Array:
        return jnp.abs(self.u)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.u.shape

    @property
    def ndim(self) -> int:
        return self.u.ndim

    @property
    def conj(self) -> Array:
        return self.replace(u=jnp.conj(self.u))

    def replace(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            where_fn = lambda tree: getattr(tree, key)
            result = eqx.tree_at(where_fn, self, value)
        return result


class AbstractMonoChromatic(eqx.Module, strict=True):
    spectrum: eqx.AbstractVar[MonochromaticSpectrum]


class AbstractPolyChromatic(eqx.Module, strict=True):
    spectrum: eqx.AbstractVar[PolyChromaticSpectrum]


class AbstractScalar(eqx.Module, strict=True):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[MonochromaticSpectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        return (
            area
            * self.spectrum.density
            * jnp.sum(jnp.abs(self.u) ** 2, axis=(self.dims.y, self.dims.x))
        )

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... l -> ... 1 1 l")
        return spectral_density * jnp.abs(self.u) ** 2


class AbstractVector(eqx.Module, strict=True):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[MonochromaticSpectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def jones_vector(self) -> Array:
        norm = jnp.linalg.norm(self.u, axis=self.dims.p, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # set to 1 to avoid division by zero
        return self.u / norm

    @property
    def power(self):
        area = jnp.prod(self.dx.squeeze(-2), axis=-1)
        total_intensity = self.spectrum.density * jnp.sum(
            jnp.abs(self.u) ** 2, axis=(self.dims.p, self.dims.y, self.dims.x)
        )
        return area * total_intensity

    @property
    def wavelength(self) -> Array:
        return rearrange(self.spectrum.wavelength, "l -> l 1")

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... l -> ... 1 1 l")
        return spectral_density * jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)

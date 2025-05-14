import abc
from enum import IntEnum
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from spectrum import MonochromaticSpectrum, PolyChromaticSpectrum


class AbstractField(eqx.Module, strict=True):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]

    # Internal for use
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    @abc.abstractmethod
    def intensity(self) -> Array:
        pass

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
    def power(self) -> Array:
        area = jnp.prod(self.dx, axis=-1)
        return area * jnp.sum(self.intensity, axis=(self.dims.y, self.dims.x))

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

    @property
    @abc.abstractmethod
    def wavelength(self) -> Array:
        pass

class AbstractPolyChromatic(eqx.Module, strict=True):
    spectrum: eqx.AbstractVar[PolyChromaticSpectrum]

    @property
    @abc.abstractmethod
    def wavelength(self) -> Array:
        pass

class AbstractScalar(eqx.Module):
    pass


class AbstractVector(eqx.Module, strict=True):
    @property
    @abc.abstractmethod
    def jones_vector(self) -> Array:
        pass


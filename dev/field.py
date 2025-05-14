import abc
from enum import IntEnum
from typing import ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Complex, Float, Real

from spectrum import MonochromaticSpectrum, PolyChromaticSpectrum
from utils import promote_dx, grid, freq_grid

Spacing = float | Real[Array, "1"] | Real[Array, "2"]


# Abstract fields
class AbstractField(eqx.Module):
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


class MonoChromatic(eqx.Module):
    spectrum: eqx.AbstractVar[MonochromaticSpectrum]

    @property
    @abc.abstractmethod
    def wavelength(self) -> Array:
        pass

class PolyChromatic(eqx.Module):
    spectrum: eqx.AbstractVar[PolyChromaticSpectrum]

    @property
    @abc.abstractmethod
    def wavelength(self) -> Array:
        pass

class Scalar(eqx.Module):
    pass


class Vector(eqx.Module):
    @property
    @abc.abstractmethod
    def jones_vector(self) -> Array:
        pass


# Actual field
class ScalarField(AbstractField, MonoChromatic, Scalar):
    u: Complex[Array, "y x"]
    dx: Float[Array, "2"]
    spectrum: MonochromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -2), ("x", -1)])

    def __init__(self, u: Array, dx: Array, spectrum: MonochromaticSpectrum):
        self.dx = promote_dx(dx)
        self.spectrum = spectrum

        # Parsing u
        u = jnp.asarray(u, dtype=jnp.complex64)
        self.u = eqx.error_if(
            u, u.ndim != 2, f"Expected 2-dimensional field, got shape {u.shape}."
        )

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength
    
    @property
    def intensity(self) -> Float[Array, "y x"]:
        return jnp.abs(self.u) ** 2

    @property
    def grid(self) -> Float[Array, "y x d"]:
        return grid(self.spatial_shape, self.dx)

    @property
    def f_grid(self) -> Float[Array, "y x d"]:
        return freq_grid(self.spatial_shape, self.dx)


class PolyChromaticScalarField(AbstractField, PolyChromatic, Scalar):
    u: Complex[Array, "y x l"]
    dx: Float[Array, "#l 2"]
    spectrum: PolyChromatic

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("l", -1)])

    def __init__(self, u: Array, dx: Spacing, spectrum: PolyChromatic):
        self.dx = rearrange(promote_dx(dx), "d -> 1 d")
        self.spectrum = spectrum

        # Parsing u
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == self.wavelength.size, f"Expected last dimension of u to be same as wavelengths."

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength
    
    @property
    def intensity(self):
        spectral_density = rearrange(
            self.spectrum.density, "... l -> ... 1 1 l"
        )
        return spectral_density * jnp.abs(self.u) ** 2

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... l y x d-> ... y x l d")

    @property
    def f_grid(self) -> Array:
        _freq_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_freq_grid, "... l y x d-> ... y x l d")


class VectorField(AbstractField, MonoChromatic, Vector):
    u: Complex[Array, "y x 3"]
    dx: Float[Array, "1 2"]
    spectrum: MonochromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("p", -1)])

    def __init__(self, u, dx, spectrum: MonochromaticSpectrum):
        self.dx = promote_dx(dx)
        self.spectrum = spectrum

        # Parsing u
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == 3, f"Expected last dimension of u to be 3, got {u.shape[-1]}"

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength
    
    @property
    def intensity(self):
        return jnp.sum(jnp.abs(self.u) ** 2, axis=self.dim.p)

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... y x d-> ... y x 1 d") 

    @property
    def f_grid(self) -> Array:
        _f_grid =freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... y x d-> ... y x 1 d")

class PolyChromaticVectorField(AbstractField, PolyChromatic, Vector):
    u: Complex[Array, "y x l 3"]
    dx: Float[Array, "#l 1 2"]
    spectrum: PolyChromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -4), ("x", -3), ("l", -2), ("p", -1)])

    def __init__(self, u, dx, spectrum):
        self.dx = rearrange(promote_dx(dx), "d -> 1 1 d")
        self.spectrum = spectrum

        # Parsing u
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 4, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-2] == self.wavelength.size, f"Expected last dimension of u to be same as wavelengths."
        assert self.u.shape[-1] == 3, f"Expected last dimension of u to be 3, got {u.shape[-1]}"

    @property
    def wavelength(self) -> Array:
        return rearrange(self.spectrum.wavelength, "l -> l 1")
    
    @property
    def intensity(self):
        spectral_density = rearrange(
            self.spectrum.density, "... l -> ... 1 1 l 1"
        )
        return spectral_density * jnp.abs(self.u) ** 2

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... l y x 2-> ... y x l 1 2")

    @property
    def f_grid(self) -> Array:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... y x l 1 2-> ... y x l 1 2")

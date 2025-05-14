from enum import IntEnum
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Complex, Float

from spectrum import MonochromaticSpectrum, PolyChromaticSpectrum
from utils import promote_dx, grid, freq_grid
from base import AbstractField, AbstractMonoChromatic, AbstractPolyChromatic, AbstractScalar, AbstractVector
from custom_types import Spacing

class ScalarField(AbstractField, AbstractMonoChromatic, AbstractScalar, strict=True):
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
    def intensity(self):
        spectral_density = rearrange(
            self.spectrum.density, "... l -> ... 1 1 l"
        )
        return spectral_density * jnp.abs(self.u) ** 2
    
    @property
    def grid(self) -> Float[Array, "y x d"]:
        return grid(self.spatial_shape, self.dx)

    @property
    def f_grid(self) -> Float[Array, "y x d"]:
        return freq_grid(self.spatial_shape, self.dx)


class PolyChromaticScalarField(AbstractField, AbstractPolyChromatic, AbstractScalar, strict=True):
    u: Complex[Array, "y x l"]
    dx: Float[Array, "l 2"]
    spectrum: PolyChromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("l", -1)])

    def __init__(self, u: Array, dx: Spacing, spectrum: PolyChromaticSpectrum):
        self.dx = repeat(promote_dx(dx), "d -> l d", l=spectrum.density.size)
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


class VectorField(AbstractField, AbstractMonoChromatic, AbstractVector, strict=True):
    u: Complex[Array, "y x 3"]
    dx: Float[Array, "1 2"]
    spectrum: MonochromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("p", -1)])

    def __init__(self, u, dx, spectrum: MonochromaticSpectrum):
        self.dx = repeat(promote_dx(dx), "d -> l d", l=spectrum.density.size)
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
        return jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... p y x d-> ... y x p d") 

    @property
    def f_grid(self) -> Array:
        _f_grid =freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... p y x d-> ... y x p d")

class PolyChromaticVectorField(AbstractField, AbstractPolyChromatic, AbstractVector, strict=True):
    u: Complex[Array, "y x l 3"]
    dx: Float[Array, "l 1 2"]
    spectrum: PolyChromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -4), ("x", -3), ("l", -2), ("p", -1)])

    def __init__(self, u, dx, spectrum: PolyChromaticSpectrum):
        self.dx = repeat(promote_dx(dx), "d -> l 1 d", l=spectrum.density.size)
        self.spectrum = spectrum

        # Parsing u
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 4, f"Expected 4-dimensional field, got shape {u.shape}."
        assert self.u.shape[-2] == self.wavelength.size, f"Expected last dimension of u to be same as wavelengths."
        assert self.u.shape[-1] == 3, f"Expected last dimension of u to be 3, got {u.shape[-1]}"

    @property
    def wavelength(self) -> Array:
        return rearrange(self.spectrum.wavelength, "l -> l 1")
    
    @property
    def intensity(self):
        spectral_density = rearrange(
            self.spectrum.density, "... l -> ... 1 1 l"
        )
        return spectral_density * jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)


    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... l p y x d-> ... y x l p d")

    @property
    def f_grid(self) -> Array:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... l p y x d-> ... y x l p d")

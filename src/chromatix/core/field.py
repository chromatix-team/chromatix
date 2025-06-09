from enum import IntEnum
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from chromatix.core.base import (
    Field,
    Chromatic,
    Monochromatic,
    Scalar,
    Vector,
)
from chromatix.core.typing import Spacing
from einops import rearrange
from jaxtyping import Array, Complex, Float
from chromatix.core.spectrum import Spectrum, MonochromaticSpectrum
from utils import freq_grid, grid, promote_dx


def field(u, dx, spectrum) -> Field:
    match (u.ndim, spectrum.size):
        case (2, _):
            field = ScalarField
        case (3, 1):
            field = VectorField
        case (3, _):
            field = PolyChromaticScalarField
        case (4, _):
            field = PolyChromaticVectorField
    return field(u, dx, spectrum)


def empty_field(amplitude: Array, shape, dx, spectrum) -> Field:
    u_empty = jnp.empty((*shape, spectrum.size, amplitude.shape[-1]))
    return field(u_empty.squeeze(), dx, spectrum)


class ScalarField(Field, Monochromatic, Scalar, strict=True):
    u: Complex[Array, "y x"]
    dx: Float[Array, "2"]
    spectrum: MonochromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -2), ("x", -1)])

    def __init__(self, u: Array, dx: Array, spectrum: MonochromaticSpectrum | float):
        self.dx = promote_dx(dx)
        self.spectrum = spectrum
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 2, f"Expected 2-dimensional field, got shape {u.shape}."

    @property
    def grid(self) -> Float[Array, "y x d"]:
        return grid(self.spatial_shape, self.dx)

    @property
    def f_grid(self) -> Float[Array, "y x d"]:
        return freq_grid(self.spatial_shape, self.dx)

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        intensity = jnp.abs(self.u) ** 2
        return area * jnp.sum(intensity, axis=self.spatial_dims)

    @property
    def intensity(self):
        return jnp.abs(self.u) ** 2


class ChromaticScalarField(
    Field, Chromatic, Scalar, strict=True
):
    u: Complex[Array, "y x wv"]
    dx: Float[Array, "#wv 2"]
    spectrum: Spectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("wv", -1)])

    def __init__(self, u: Array, dx: Spacing, spectrum: Spectrum):
        self.dx = rearrange(promote_dx(dx), "d -> 1 d")
        self.spectrum = spectrum
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == self.wavelength.size, (
            "Expected last dimension of u to be same as wavelengths."
        )

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... wv y x d-> ... y x wv d")

    @property
    def f_grid(self) -> Array:
        _freq_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_freq_grid, "... wv y x d-> ... y x wv d")

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        intensity = jnp.abs(self.u) ** 2
        power_density = jnp.sum(intensity, axis=self.spatial_dims)
        return area * self.spectrum.density * power_density

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... wv -> ... 1 1 wv")
        return spectral_density * jnp.abs(self.u) ** 2


class VectorField(Field, Monochromatic, Vector, strict=True):
    u: Complex[Array, "y x 3"]
    dx: Float[Array, "2"]
    spectrum: MonochromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("p", -1)])

    def __init__(self, u, dx, spectrum: MonochromaticSpectrum):
        self.dx = promote_dx(dx)
        self.spectrum = spectrum
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == 3, (
            f"Expected last dimension of u to be 3, got {u.shape[-1]}"
        )

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... y x d-> ... y x 1 d")

    @property
    def f_grid(self) -> Array:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... y x d-> ... y x 1 d")

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        intensity = jnp.abs(self.u) ** 2
        power_density = jnp.sum(intensity, axis=(self.dims.p, *self.spatial_dims))
        return area * power_density

    @property
    def intensity(self):
        return jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)


class ChromaticVectorField(
    Field, Chromatic, Vector, strict=True
):
    u: Complex[Array, "y x wv 3"]
    dx: Float[Array, "#wv 2"]
    spectrum: PolychromaticSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum(
        "dims", [("y", -4), ("x", -3), ("wv", -2), ("p", -1)]
    )

    def __init__(self, u, dx, spectrum: PolychromaticSpectrum):
        self.dx = rearrange(promote_dx(dx), "d -> 1 d")
        self.spectrum = spectrum
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        assert self.u.ndim == 4, f"Expected 4-dimensional field, got shape {u.shape}."
        assert self.u.shape[-2] == self.wavelength.size, (
            "Expected last dimension of u to be same as wavelengths."
        )
        assert self.u.shape[-1] == 3, (
            f"Expected last dimension of u to be 3, got {u.shape[-1]}"
        )

    @property
    def grid(self) -> Array:
        _grid = grid(self.spatial_shape, self.dx)
        return rearrange(_grid, "... wv y x d-> ... y x wv 1 d")

    @property
    def f_grid(self) -> Array:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(_f_grid, "... wv y x d-> ... y x wv 1 d")

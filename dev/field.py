import abc
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Complex, Float, Real


# Abstract fields
class AbstractField(eqx.Module):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Array] # NOTE: Should we make a spectrum type?

    # Internal for use
    dims: eqx.AbstractClassVar[dict[str, int]] # NOTE: Turn this into a enum?

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
    def k_grid(self) -> Array:
        pass

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        return area * jnp.sum(self.intensity, axis=(self.dims["y"], self.dims["x"]))
    
    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.u.shape[self.dims["y"]], self.u.shape[self.dims["x"]]

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
        # TODO: Add replace method?
        return self.replace(u=jnp.conj(self.u))

    # These are standardised grids without empty dimensions;
    # and they should be reshaped in the actual field
    @property
    def _grid(self) -> Array:
     N_y, N_x = self.spatial_shape
     dx = rearrange(self.dx, "... d -> ... 1 1 d")
     grid = jnp.meshgrid(
            jnp.linspace(0, (N_y - 1), N_y) - N_y / 2,
            jnp.linspace(0, (N_x - 1), N_x) - N_x / 2,
            indexing="ij"
        )
     return dx * jnp.stack(grid, axis=-1)
    
    @property
    def _freq_grid(self) -> Array:
        N_y, N_x = self.spatial_shape
        dk = rearrange( 1/ self.dx, "... d -> ... 1 1 d")
        grid = jnp.meshgrid(
            jnp.fft.fftshift(jnp.fft.fftfreq(N_y)),
            jnp.fft.fftshift(jnp.fft.fftfreq(N_x)),
            indexing="ij"
        )
        return dk * jnp.stack(grid, axis=-1)

class Spectral(eqx.Module):
    spectral_density: eqx.AbstractVar[Array] # TODO: is this the right name?

class Scalar(eqx.Module):
    pass

class Vectorial(eqx.Module):

    @property
    @abc.abstractmethod
    def jones_vector(self) -> Array:
        pass


# These two don't do anything yet.
class Coherent(eqx.Module):
    pass

class PartiallyCOherent(eqx.Module):
    pass

# Actual field
class CoherentScalarField(AbstractField, Scalar, Coherent):
    u: Complex[Array, "*b y x"]
    dx: Float[Array, "*b 2"]
    spectrum: Float[Array, "*b 1"]

    dims: ClassVar[dict[str, int]]= {"y": -2, "x": -1}

    def __init__(self, dx: float | Real[Array, "1"] | Real[Array, "2"], spectrum: float, u: Complex[Array, "y x"]):
        # Parsing u
        self.u = jnp.asarray(u)

        # Parsing dx 
        dx = jnp.asarray(dx)
        match dx.size:
            case 1:
                self.dx = jnp.stack([dx, dx])
            case 2:
                self.dx = dx
            case _:
                raise ValueError(f"dx must be of size 1 or 2, got {dx.size}")
        
        # Parsing spectrum
        self.spectrum = jnp.asarray(spectrum)

    # The only functions that need to be implemented are intensity, grid, and k_grid

    @property
    def intensity(self) -> Float[Array, "*b y x"]:
        return jnp.abs(self.u)**2

    @property
    def grid(self) -> Float[Array, "*b y x d"]:
        return self._grid 

    @property
    def k_grid(self) -> Float[Array, "*b y x d"]:
        # TODO: Technically we're missing a factor 2 pi here!
        # This should be called fx
        return self._freq_grid
    


class SpectralCoherentScalarField(AbstractField, Scalar, Spectral, Coherent):
    u: Complex[Array, "*b y x l"]
    dx: Float[Array, "*b #l 2"]
    spectrum: Float[Array, "*b l"]
    spectral_density: Float[Array, "*b l"]

    dims: ClassVar[dict[str, int]]= {"y": -3, "x": -2, "l": -1}

    def __init__(self, dx: float | Real[Array, "1"] | Real[Array, "2"], spectrum: Float[Array, "l"], spectral_density: Float[Array, "l"], u: Complex[Array, "y x l"]):
        # TODO; we need some more shape checking here
        # Parsing u
        self.u = jnp.asarray(u)

        # Parsing dx 
        dx = jnp.asarray(dx)
        match dx.size:
            case 1:
                self.dx = rearrange(jnp.stack([dx, dx]), "c -> 1 c")
            case 2:
                self.dx = rearrange(dx, "c -> 1 c")
            case _:
                raise ValueError(f"dx must be of size 1 or 2, got {dx.size}")
        
        # Parsing spectrum
        self.spectrum = jnp.asarray(spectrum)
        self.spectral_density = jnp.asarray(spectral_density)


    @property
    def intensity(self):
        spectral_density = rearrange(self.spectral_density, "... l -> ... 1 1 l")
        return spectral_density * jnp.abs(self.u)**2

    @property
    def grid(self) -> Array:
        return rearrange(self._grid, "... l y x d-> ... y x l d")

    @property
    def k_grid(self) -> Array:
        # TODO: Technically we're missing a factor 2 pi here!
        # This should be called fx
        return rearrange(self._freq_grid, "... l y x d-> ... y x l d")
        
    
class CoherentVectorField(AbstractField, Vectorial, Coherent):
    pass

class SpectralCoherentVectorField(AbstractField, Vectorial, Spectral, Coherent):
    pass





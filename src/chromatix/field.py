from __future__ import annotations

import jax.numpy as jnp
from chex import Array, assert_rank
from flax import struct
from einops import rearrange
from typing import Union, Optional, Tuple, Any
from numbers import Number

from jax.scipy.ndimage import map_coordinates


class Field(struct.PyTreeNode):
    u: jnp.ndarray  # [B H W C]
    dx: jnp.ndarray
    spectrum: jnp.ndarray
    spectral_density: jnp.ndarray

    @classmethod
    def create(
        cls,
        dx: float,
        spectrum: Union[float, jnp.ndarray],
        spectral_density: Union[float, jnp.ndarray],
        u: Optional[jnp.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Field:
        # Getting everything into right shape
        # We use [B H W C], where B is batch size
        # and C is number of wavelengths.
        field_dx: jnp.ndarray = rearrange(jnp.atleast_1d(dx), "1 -> 1 1 1 1")
        field_spectrum: jnp.ndarray = rearrange(
            jnp.atleast_1d(spectrum), "c -> 1 1 1 c"
        )
        field_spectral_density: jnp.ndarray = rearrange(
            jnp.atleast_1d(spectral_density), "c -> 1 1 1 c"
        )
        field_spectral_density = field_spectral_density / jnp.sum(
            field_spectral_density
        )  # Must sum to 1
        if u is None:
            # NOTE(dd): when jitting this function, shape must be a
            # static argument --- possibly requiring multiple traces
            assert shape is not None, "Must specify shape if u is None"
            field_u: jnp.ndarray = jnp.empty(
                (1, *shape, field_spectrum.size), dtype=jnp.complex64
            )
        else:
            field_u = u
        assert_rank(
            field_u, 4, custom_message="Field must be ndarray of shape [B H W C]"
        )
        field = cls(
            field_u,
            field_dx,
            field_spectrum,
            field_spectral_density,
        )
        return field

    # Grid properties
    @property
    def grid(self) -> jnp.ndarray:
        half_size = jnp.array(self.shape[1:3]) / 2
        # We must use meshgrid instead of mgrid here in order to be jittable
        grid = jnp.meshgrid(
            jnp.linspace(-half_size[0], half_size[0] - 1, num=self.shape[1]) + 0.5,
            jnp.linspace(-half_size[1], half_size[1] - 1, num=self.shape[2]) + 0.5,
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d 1 h w 1")
        return self.dx * grid

    @property
    def l2_sq_grid(self) -> jnp.ndarray:
        return jnp.sum(self.grid ** 2, axis=0)

    @property
    def l2_grid(self) -> jnp.ndarray:
        return jnp.sqrt(jnp.sum(self.grid ** 2, axis=0))

    @property
    def l1_grid(self) -> jnp.ndarray:
        return jnp.sum(jnp.abs(self.grid), axis=0)

    @property
    def linf_grid(self) -> jnp.ndarray:
        return jnp.max(jnp.abs(self.grid), axis=0)

    # Field properties
    @property
    def phase(self) -> jnp.ndarray:
        return jnp.angle(self.u)

    @property
    def intensity(self) -> jnp.ndarray:
        return jnp.sum(
            self.spectral_density * jnp.abs(self.u) ** 2,
            axis=-1,
            keepdims=True,
        )

    @property
    def power(self) -> jnp.ndarray:
        return jnp.sum(self.intensity, axis=(1, 2), keepdims=True) * self.dx ** 2

    @property
    def shape(self) -> Tuple:
        return self.u.shape

    # Math operations
    def __add__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u + other)
        elif isinstance(other, Field):
            return self.replace(u=self.u + other.u)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> Field:
        return self + other

    def __sub__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u - other)
        elif isinstance(other, Field):
            return self.replace(u=self.u - other.u)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> Field:
        return (-1 * self) + other

    def __mul__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u * other)
        elif isinstance(other, Field):
            return self.replace(u=self.u * other.u)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> Field:
        return self * other

    def __truediv__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u / other)
        elif isinstance(other, Field):
            return self.replace(u=self.u / other.u)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> Field:
        return self.replace(u=other / self.u)

    def __floordiv__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u // other)
        elif isinstance(other, Field):
            return self.replace(u=self.u // other.u)
        else:
            return NotImplemented

    def __rfloordiv__(self, other: Any) -> Field:
        return self.replace(u=other // self.u)

    def __mod__(self, other: Union[Number, jnp.ndarray, Field]) -> Field:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u % other)
        elif isinstance(other, Field):
            return self.replace(u=self.u % other.u)
        else:
            return NotImplemented

    def __rmod__(self, other: Any) -> Field:
        return self.replace(u=other % self.u)

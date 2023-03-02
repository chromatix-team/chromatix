from __future__ import annotations

import jax.numpy as jnp
from chex import Array, assert_rank, assert_equal_shape
from flax import struct
from einops import rearrange
from typing import Union, Optional, Tuple, Any
from numbers import Number

from jax.scipy.ndimage import map_coordinates


class Field(struct.PyTreeNode):
    """
    A container that describes the chromatic scalar light field at a 2D plane.

    ``Field`` objects track various attributes of a complex-valued scalar
    field (in addition to the field itself for each wavelength): the spacing
    of the samples along the field, the wavelengths in the spectrum, and the
    density of the wavelengths. This information can be used, for example, to
    calculate the intensity of a field at a plane, appropriately weighted by
    the spectrum. ``Field`` objects also provide various grids for convenience,
    as well as allow elementwise operations with any broadcastable values,
    including scalars, arrays, or other ``Field`` objects. These operations
    include: `+`, `-` (including negation), `*`, `/`, `+=`, `-=`, `*=`, `/=`.

    The shape of a base ``Field`` object is `[B H W C]`, where B is batch,
    H and W are height and width, and C is the channel dimension, which
    we use for different wavelengths in the spectrum of a ``Field``. The
    batch dimension can be used for any purpose, such as different samples,
    depth, or time. If more dimensions are required, we encourage the use of
    ``jax.vmap``, ``jax.pmap``, or a combination of the two. We intend for this
    to be a compromise between not having too many dimensions when they are
    not required, and also not having to litter a program with ``jax.vmap``
    transformations for common simulations in 3D.

    Concretely, this means simulations that are only in 2D and with a single
    wavelength will always have only two distinct singleton dimensions. Any
    simulations taking into account either depth or multiple wavelengths will
    not need to use JAX transformations or loops. Any simulations requiring
    both 3D spatial dimensions and the tracking of the field across time will
    either have to loop through time or use ``jax.vmap``/``jax.pmap``.

    Due to this shape, in order to ensure that attributes of ``Field``
    objects broadcast appropriately, attributes which could be 1D arrays
    are ensured to have extra singleton dimensions. In order to make the
    creation of a ``Field`` object more convenient, we provide the class
    method ``Field.create()`` (detailed below), which accepts scalar or 1D
    array arguments for the various attributes (e.g. if a single wavelength
    is desired, a scalar value can be used, but if multiple wavelengths are
    desired, a 1D array can be used for the value of ``spectrum``). This method
    appropriately reshapes the attributes provided to the correct shapes.

    Attributes:
        u: The scalar field of shape `[B H W C]`.
        dx: The spacing of the samples in ``u`` discretizing a continuous field.
        spectrum: The wavelengths sampled by the field, in any units specified.
        spectral_density: The weights of the wavelengths in the spectrum. Must
            sum to 1.0.
    """

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
        """
        Create a ``Field`` object in a convenient way.

        Creates a ``Field`` object, accepting arguments as scalars or 1D values
        as appropriate. This class function appropriately reshapes the given
        values of attributes to the necessary shapes, allowing a ``Field`` to
        be created with scalar or 1D array values for the spectrum and spectral
        density, as desired.

        Args:
            dx: The spacing of the samples in ``u`` discretizing a continuous field.
            spectrum: The wavelengths sampled by the field, in any units specified.
            spectral_density: The weights of the wavelengths in the spectrum.
                Will be normalized to sum to 1.0.
            u: The scalar field of shape `[B H W C]`. If not given,
                the ``Field`` is allocated with uninitialized values of the
                given ``shape``.
            shape: A tuple defining the shape of only the spatial
                dimensions of the ``Field`` (height and width). Not required
                if ``u`` is provided. If ``u`` is not provided, then ``shape``
                must be provided.
        """
        # Getting everything into right shape
        field_dx: jnp.ndarray = rearrange(jnp.atleast_1d(dx), "c -> 1 1 1 c")
        field_spectrum: jnp.ndarray = rearrange(
            jnp.atleast_1d(spectrum), "c -> 1 1 1 c"
        )
        field_spectral_density: jnp.ndarray = rearrange(
            jnp.atleast_1d(spectral_density), "c -> 1 1 1 c"
        )
        field_spectral_density = field_spectral_density / jnp.sum(
            field_spectral_density
        )  # Must sum to 1
        assert_equal_shape([field_dx, field_spectrum, field_spectral_density])
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
            field_u, 4, custom_message="Field must be ndarray of shape `[B H W C]`"
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
        """
        The grid for each spatial dimension as an array of shape `[2 1 H W 1]`.
        The 2 entries along the first dimension represent the y and x grids,
        respectively. This grid assumes that the center of the ``Field`` is
        the origin and that the elements are sampling from the center, not
        the corner.

        In addition to this actual grid, ``Field`` also provides:
            - ``l2_sq_grid``
            - ``l2_grid``
            - ``l1_grid``
            - ``linf_grid``
        each of which are described below.
        """
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
        """Sum of the squared grid over spatial dimensions, i.e. `x**2 + y**2`."""
        return jnp.sum(self.grid ** 2, axis=0)

    @property
    def l2_grid(self) -> jnp.ndarray:
        """Square root of ``l2_sq_grid``, i.e. `sqrt(x**2 + y**2)`."""
        return jnp.sqrt(jnp.sum(self.grid ** 2, axis=0))

    @property
    def l1_grid(self) -> jnp.ndarray:
        """Sum absolute value over spatial dimensions, i.e. `|x| + |y|`."""
        return jnp.sum(jnp.abs(self.grid), axis=0)

    @property
    def linf_grid(self) -> jnp.ndarray:
        """Max absolute value over spatial dimensions, i.e. `max(|x|, |y|)`."""
        return jnp.max(jnp.abs(self.grid), axis=0)

    # Field properties
    @property
    def phase(self) -> jnp.ndarray:
        """Phase of the complex scalar field, shape `[B H W C]`."""
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> jnp.ndarray:
        """Amplitude of the complex scalar field, shape `[B H W C]`."""
        return jnp.abs(self.u)

    @property
    def intensity(self) -> jnp.ndarray:
        """Intensity of the complex scalar field, shape `[B H W 1]`."""
        return jnp.sum(
            self.spectral_density * jnp.abs(self.u) ** 2,
            axis=-1,
            keepdims=True,
        )

    @property
    def power(self) -> jnp.ndarray:
        """Power of the complex scalar field, shape `[B 1 1 1]`."""
        return jnp.sum(self.intensity, axis=(1, 2), keepdims=True) * self.dx ** 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the complex scalar field."""
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

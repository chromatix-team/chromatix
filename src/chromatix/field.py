from __future__ import annotations
import jax.numpy as jnp
from chex import Array, assert_equal_shape, assert_rank
from flax import struct
from einops import rearrange
from typing import Union, Optional, Tuple, Any
from numbers import Number
from .utils.shapes import (
    _broadcast_1d_to_channels,
    _broadcast_1d_to_grid,
    _broadcast_2d_to_grid,
)


class Field(struct.PyTreeNode):
    """
    A container that describes the chromatic light field at a 2D plane.

    ``Field`` objects track various attributes of a complex-valued field (in
    addition to the field itself for each wavelength): the spacing of the
    samples along the field, the wavelengths in the spectrum, and the density
    of the wavelengths. This information can be used, for example, to calculate
    the intensity of a field at a plane, appropriately weighted by the spectrum.
    ``Field`` objects also provide various grids for convenience, as well
    as allow elementwise operations with any broadcastable values, including
    scalars, arrays, or other ``Field`` objects. These operations include: `+`,
    `-` (including negation), `*`, `/`, `+=`, `-=`, `*=`, `/=`.

    The shape of a ``Field`` object is `(B... H W C [1 | 3])`, where B... is
    an arbitrary number of batch dimensions, H and W are height and width,
    and C is the channel dimension, which we use for different wavelengths in
    the spectrum of a ``Field``. The final dimension has size either 1 for a
    scalar approximation ``ScalarField`` or 3 for the full vectorial case of
    ``VectorField``. Any function in Chromatix that deals with ``Field``s can
    work with either ``ScalarField``s or ``VectorField``s, unless otherwise
    stated.

    The (potentially more than 1) batch dimensions can be used for any purpose,
    such as different samples, depth, or time. Any Chromatix functions that
    produce multiple depths (e.g. propagation to multiple z values) will
    broadcast to the innermost batch dimension. If more dimensions are required,
    we encourage the use of ``jax.vmap``, ``jax.pmap``, or a combination of
    the two. We intend for this to be a compromise between not having too many
    dimensions when they are not required, and also not having to litter a
    program with ``jax.vmap`` transformations for common simulations in 3D or 3D
    over time.

    Due to this shape, in order to ensure that attributes of ``Field``
    objects broadcast appropriately, attributes which could be 1D arrays are
    ensured to have extra singleton dimensions. In order to make the creation
    of ``Field`` objects more convenient, we provide the class methods
    ``ScalarField.create()`` and ``VectorField.create()`` (detailed below),
    which accepts scalar or 1D array arguments for the various attributes
    (e.g. if a single wavelength is desired, a scalar value can be used, but if
    multiple wavelengths are desired, a 1D array can be used for the value of
    ``spectrum``). These methods appropriately reshapes the attributes provided
    to the correct shapes.

    Attributes:
        u: The complex field of shape `(B... H W C [1 | 3])`.
        dx: The spacing of the samples in ``u`` discretizing a continuous
            field. Can either be a 1D array with the same size as the number
            of wavelengths in the spectrum of shape (C), specifying a square
            spacing per wavelength, or a 2D array of shape (2 C) specifying
            the spacing in the y and x directions respectively for non-
            square pixels. A float can also be specified to use the same
            square spacing for all wavelengths. Spacing will be the same per
            wavelength for all entries in a batch.
        spectrum: The wavelengths sampled by the field, in any units specified.
        spectral_density: The weights of the wavelengths in the spectrum. Must
            sum to 1.0.
        spatial_dims: A tuple of two integers specifying the spatial dimensions
            (the `H W` dimensions respectively) within a `Field` that
            potentially has multiple batch dimensions.
    """

    u: Array  # (B... H W C [1 | 3])
    dx: Array  # (2 B... H W C [1 | 3])
    spectrum: Array  # (B... H W C [1 | 3])
    spectral_density: Array  # (B... H W C [1 | 3])

    @property
    def grid(self) -> Array:
        """
        The grid for each spatial dimension as an array of shape `(2 B... H W
        C 1)`. The 2 entries along the first dimension represent the y and x
        grids, respectively. This grid assumes that the center of the ``Field``
        is the origin and that the elements are sampling from the center, not
        the corner.
        """
        # We must use meshgrid instead of mgrid here in order to be jittable
        N_y, N_x = self.spatial_shape
        grid = jnp.meshgrid(
            jnp.linspace(-N_y // 2, N_y // 2 - 1, num=N_y) + 0.5,
            jnp.linspace(-N_x // 2, N_x // 2 - 1, num=N_x) + 0.5,
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d " + ("1 " * (self.ndim - 4)) + "h w 1 1")
        return self.dx * grid

    @property
    def k_grid(self) -> Array:
        N_y, N_x = self.spatial_shape
        grid = jnp.meshgrid(
            jnp.linspace(-N_y // 2, N_y // 2 - 1, num=N_y) + 0.5,
            jnp.linspace(-N_x // 2, N_x // 2 - 1, num=N_x) + 0.5,
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d " + ("1 " * (self.ndim - 4)) + "h w 1 1")
        return self.dk * grid

    @property
    def dk(self) -> Array:
        shape = jnp.array(self.spatial_shape)
        shape = _broadcast_1d_to_grid(shape, self.ndim)
        return 1 / (self.dx * shape)

    @property
    def phase(self) -> Array:
        """Phase of the complex field, shape `(B... H W C [1 | 3])`."""
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> Array:
        """Amplitude of the complex field, shape `(B... H W C [1 | 3])`."""
        return jnp.abs(self.u)

    @property
    def intensity(self) -> Array:
        """Intensity of the complex field, shape `(B... H W 1 1)`."""
        return jnp.sum(
            self.spectral_density * jnp.abs(self.u) ** 2, axis=(-2, -1), keepdims=True
        )

    @property
    def power(self) -> Array:
        """Power of the complex field, shape `(B... 1 1 1)`."""
        area = jnp.prod(self.dx, axis=0, keepdims=False)
        return jnp.sum(self.intensity, axis=(-4, -3), keepdims=True) * area

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the complex field."""
        return self.u.shape

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        """Only the height and width of the complex field."""
        return self.u.shape[self.spatial_dims[0] : self.spatial_dims[1] + 1]

    @property
    def spatial_dims(self) -> Tuple[int, int]:
        """Returns dimensions representing height and width."""
        return (-4, -3)

    @property
    def ndim(self) -> int:
        return self.u.ndim

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


class ScalarField(Field):
    @classmethod
    def create(
        cls,
        dx: Union[float, Array],
        spectrum: Union[float, Array],
        spectral_density: Union[float, Array],
        u: Optional[Array] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Field:
        """
        Create a scalar approximation ``Field`` object in a convenient way.

        This class function appropriately reshapes the given values of
        attributes to the necessary shapes, allowing a ``Field`` to be created
        with scalar or 1D array values for the spectrum and spectral density,
        as desired.

        Args:
            dx: The spacing of the samples in ``u`` discretizing a continuous
                field. Can either be a 1D array with the same size as the
                number of wavelengths in the spectrum of shape (C), specifying
                a square spacing per wavelength, or a 2D array of shape (2 C)
                specifying the spacing in the y and x directions respectively
                for non-square pixels. A float can also be specified to use the
                same square spacing for all wavelengths. Spacing will be the
                same per wavelength for all entries in a batch.
            spectrum: The wavelengths sampled by the field, in any units
                specified. Should be a 1D array containing each wavelength, or
                a float for a single wavelength.
            spectral_density: The weights of the wavelengths in the spectrum.
                Will be scaled to sum to 1.0 over all wavelengths. Should be a
                1D array containing the weight of each wavelength, or a float
                for a single wavelength.
            u: The scalar field of shape `(B... H W C 1)`. If not given,
                the ``Field`` is allocated with uninitialized values of the
                given ``shape`` as `(1 H W C 1)`.
            shape: A tuple defining the shape of only the spatial
                dimensions of the ``Field`` of the form `(H W)`. Not required
                if ``u`` is provided. If ``u`` is not provided, then ``shape``
                must be provided.
        """
        dx: Array = jnp.atleast_1d(dx)
        spectrum: Array = jnp.atleast_1d(spectrum)
        spectral_density: Array = jnp.atleast_1d(spectral_density)
        if u is None:
            assert shape is not None, "Must specify shape if u is None"
            u = jnp.empty((1, *shape, spectrum.size, 1), dtype=jnp.complex64)
        ndim = len(u.shape)
        assert (
            ndim >= 5
        ), "Field must be Array with at least 5 dimensions: (B... H W C 1)."
        assert u.shape[-1] == 1, "Last dimension must be 1 for scalar fields."
        assert_equal_shape([spectrum, spectral_density])
        spectrum = _broadcast_1d_to_channels(spectrum, ndim)
        spectral_density = _broadcast_1d_to_channels(spectral_density, ndim)
        spectral_density = spectral_density / jnp.sum(spectral_density)
        if dx.ndim == 1:
            dx = jnp.stack([dx, dx])
        assert_rank(dx, 2)  # dx should have shape (2, C) here
        dx = _broadcast_2d_to_grid(dx, ndim)
        return cls(u, dx, spectrum, spectral_density)


class VectorField(Field):
    @classmethod
    def create(
        cls,
        dx: Union[float, Array],
        spectrum: Union[float, Array],
        spectral_density: Union[float, Array],
        u: Optional[Array] = None,
        shape: Optional[Tuple[int, int]] = None,
    ) -> Field:
        """
        Create a vectorial ``Field`` object in a convenient way.

        This class function appropriately reshapes the given values of
        attributes to the necessary shapes, allowing a ``Field`` to be created
        with scalar or 1D array values for the spectrum and spectral density,
        as desired.

        Args:
            dx: The spacing of the samples in ``u`` discretizing a continuous
                field. Can either be a 1D array with the same size as the
                number of wavelengths in the spectrum of shape (C), specifying
                a square spacing per wavelength, or a 2D array of shape (2 C)
                specifying the spacing in the y and x directions respectively
                for non-square pixels. A float can also be specified to use the
                same square spacing for all wavelengths. Spacing will be the
                same per wavelength for all entries in a batch.
            spectrum: The wavelengths sampled by the field, in any units
                specified. Should be a 1D array containing each wavelength, or
                a float for a single wavelength.
            spectral_density: The weights of the wavelengths in the spectrum.
                Will be scaled to sum to 1.0 over all wavelengths. Should be a
                1D array containing the weight of each wavelength, or a float
                for a single wavelength.
            u: The vectorial field of shape `(B... H W C 3)`. If not given,
                the ``Field`` is allocated with uninitialized values of the
                given ``shape`` as `(1 H W C 3)`.
            shape: A tuple defining the shape of only the spatial
                dimensions of the ``Field`` of the form `(H W)`. Not required
                if ``u`` is provided. If ``u`` is not provided, then ``shape``
                must be provided.
        """
        dx: Array = jnp.atleast_1d(dx)
        spectrum: Array = jnp.atleast_1d(spectrum)
        spectral_density: Array = jnp.atleast_1d(spectral_density)
        if u is None:
            assert shape is not None, "Must specify shape if u is None"
            u = jnp.empty((1, *shape, spectrum.size, 3), dtype=jnp.complex64)
        ndim = len(u.shape)
        assert (
            ndim >= 5
        ), "Field must be Array with at least 5 dimensions: (B... H W C 3)."
        assert u.shape[-1] == 3, "Last dimension must be 3 for vectorial fields."
        assert_equal_shape([spectrum, spectral_density])
        spectrum = _broadcast_1d_to_channels(spectrum, ndim)
        spectral_density = _broadcast_1d_to_channels(spectral_density, ndim)
        spectral_density = spectral_density / jnp.sum(spectral_density)
        if dx.ndim == 1:
            dx = jnp.stack([dx, dx])
        assert_rank(dx, 2)  # dx should have shape (2, C) here
        dx = _broadcast_2d_to_grid(dx, ndim)
        return cls(u, dx, spectrum, spectral_density)

    @property
    def jones_vector(self) -> Array:
        """Return Jones vector of field."""
        norm = jnp.linalg.norm(self.u, axis=-1, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # set to 1 to avoid division by zero
        return self.u / norm

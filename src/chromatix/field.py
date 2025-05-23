from __future__ import annotations

from numbers import Number
from typing import Any, TypeVar

import jax.numpy as jnp
from chex import assert_equal_shape, assert_rank
from einops import rearrange
from flax import struct
from jax import Array
from typing_extensions import Self

from chromatix.typing import ArrayLike, ScalarLike

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
        u: The complex field of shape ``(B... H W C [1 | 3])``.
        _dx: The spacing of the samples in ``u`` discretizing a continuous
            field. Defined as a 2D array of shape ``(2 C)`` specifying the spacing
            in the y and x directions respectively (can be the same for y and
            x for the common case of square pixels). Spacing is the same per
            wavelength for all entries in a batch. Not intended to be publicly
            accessed, because the shape of this attribute does not dynamically
            adapt to the ``ndim`` of the ``Field``. Instead, use the ``dx``
            property.
        _spectrum: The wavelengths sampled by the field, in any units specified.
            Should be a 1D array. Not intended to be publicly accessed, because
            the shape of this attribute does not dynamically adapt to the
            ``ndim`` of the ``Field``. Instead, use the ``spectrum`` property.
        _spectral_density: The weights of the wavelengths in the spectrum.
            Shouldbe a 1D array of same length as ``_spectrum``. Must sum to
            1.0. Not intended to be publicly accessed, because the shape of
            this attribute does not dynamically adapt to the ``ndim`` of the
            ``Field``. Instead, use the ``spectral_density`` property.
    """

    u: Array  # (B... H W C [1 | 3])
    _dx: Array = struct.field(pytree_node=False)  # (2 B... H W C [1 | 3])
    _spectrum: Array = struct.field(pytree_node=False)  # (B... H W C [1 | 3])
    _spectral_density: Array = struct.field(pytree_node=False)  # (B... H W C [1 | 3])
    _origin: Array = struct.field(pytree_node=False)  # (2 B... H W C [1 | 3])

    @classmethod
    def empty_like(
        cls,
        field: Field,
        dx: ScalarLike | None = None,
        shape: tuple[int, int] | None = None,
        spectrum: ScalarLike | None = None,
        spectral_density: ScalarLike | None = None,
        origin: ArrayLike | None = None,
    ) -> Field:
        """
        Copy over attributes of ``field`` to a new ``Field`` object, with the
        option of changing some attributes.

        Note that this function overwrites the field `u` with a new empty field.
        """
        if dx is None:
            dx = field.dx
        else:
            if dx.ndim == 1:
                dx = jnp.stack([dx, dx])
            assert_rank(dx, 2)  # dx should have shape (2, C) here
        if shape is None:
            shape = field.spatial_shape
        else:
            assert len(shape) == 2
        if spectrum is None:
            spectrum = field.spectrum
        else:
            spectrum = jnp.atleast_1d(spectrum)
        if spectral_density is None:
            spectral_density = field.spectral_density
        else:
            spectral_density = jnp.atleast_1d(spectral_density)
        if origin is None:
            origin = field.origin.squeeze()
        else:
            if isinstance(origin, Number):
                origin = jnp.array([origin, origin])
            origin = jnp.array(origin)
        origin = origin[:, None]
        assert_rank(origin, 2)
        u = jnp.empty(
            (1, *shape, spectrum.size, field.u.shape[-1]), dtype=field.u.dtype
        )
        return cls(u, dx, spectrum, spectral_density, origin)

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
            jnp.linspace(0, (N_y - 1), N_y) - N_y / 2,
            jnp.linspace(0, (N_x - 1), N_x) - N_x / 2,
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d " + ("1 " * (self.ndim - 4)) + "h w 1 1")
        return self.dx * grid + self.origin

    @property
    def k_grid(self) -> Array:
        """
        The frequency grid for each spatial dimension as an array of shape `(2
        B... H W C 1)`. The 2 entries along the first dimension represent the
        y and x grids, respectively. This grid assumes that the center of the
        ``Field`` is the origin and that the elements are sampling from the
        center, not the corner.
        """
        N_y, N_x = self.spatial_shape
        grid = jnp.meshgrid(
            jnp.fft.fftshift(jnp.fft.fftfreq(N_y)),
            jnp.fft.fftshift(jnp.fft.fftfreq(N_x)),
            indexing="ij",
        )
        grid = rearrange(grid, "d h w -> d " + ("1 " * (self.ndim - 4)) + "h w 1 1")
        return grid / self.dx

    @property
    def dx(self) -> Array:
        """
        The spacing of the samples in ``u`` discretizing a continuous field.
        Defined as an array of shape ``(2 1... 1 1 C 1)`` specifying the spacing
        in the y and x directions respectively (can be the same for y and x for
        the common case of square pixels). Spacing is the same per wavelength
        for all entries in a batch.
        """
        return _broadcast_2d_to_grid(self._dx, self.ndim)

    @property
    def origin(self) -> Array:
        """
        The shift of the sampling place, such that it is no longer centered at
        the origin. Defined as an array of shape ``(2 1... 1 1 C 1)``
        specifying the shift in the y and x directions respectively.
        """
        return _broadcast_2d_to_grid(self._origin, self.ndim)

    @property
    def dk(self) -> Array:
        """
        The frequency spacing of the samples in the frequency space of ``u``.
        Defined as an array of shape ``(2 1... 1 1 C 1)`` specifying the spacing
        in the y and x directions respectively (can be the same for y and x for
        the common case of square pixels). Spacing is the same per wavelength
        for all entries in a batch.
        """
        shape = jnp.array(self.spatial_shape)
        shape = _broadcast_1d_to_grid(shape, self.ndim)
        return 1 / (self.dx * shape)

    @property
    def extent(self) -> Array:
        """
        The extent (lengths in height and width per wavelength) of the field
        in units of distance. Defined as an array of shape ``(2 1... 1 1 C 1)``
        specifying the extent in the y and x dimensions respectively.
        """
        shape = jnp.array(self.spatial_shape)
        shape = _broadcast_1d_to_grid(shape, self.ndim)
        return self.dx * shape

    @property
    def spectrum(self) -> Array:
        """
        Wavelengths sampled by the complex field, shape ``(1... 1 1 C 1)``.
        """
        return _broadcast_1d_to_channels(self._spectrum, self.ndim)

    @property
    def spectral_density(self) -> Array:
        """
        Weights of wavelengths sampled by the complex field, shape ``(1... 1 1
        C 1)``.
        """
        return _broadcast_1d_to_channels(self._spectral_density, self.ndim)

    @property
    def phase(self) -> Array:
        """
        Phase of the complex field, shape `(B... H W C [1 | 3])`.
        """
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> Array:
        """
        Amplitude of the complex field, shape `(B... H W C [1 | 3])`. This is
        actually what is called the "magnitude".
        """
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
    def shape(self) -> tuple[int, ...]:
        """Shape of the complex field."""
        return self.u.shape

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Only the height and width of the complex field."""
        return (self.u.shape[self.spatial_dims[0]], self.u.shape[self.spatial_dims[1]])

    @property
    def spatial_dims(self) -> tuple[int, int]:
        """Dimensions representing the height and width of the complex field."""
        return (-4, -3)

    @property
    def ndim(self) -> int:
        """Number of dimensions (the rank) of the complex field."""
        return self.u.ndim

    @property
    def conj(self) -> Self:
        """conjugate of the complex field, as a field of the same shape."""
        return self.replace(u=jnp.conj(self.u))

    @property
    def spatial_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return the spatial limits of the field: (y_min, y_max), (x_min, x_max).
        """
        return (float(self.grid[0].min()), float(self.grid[0].max())), (
            float(self.grid[1].min()),
            float(self.grid[1].max()),
        )

    def __add__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u + other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u + other.u)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> Self:
        return self + other

    def __sub__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u - other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u - other.u)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> Self:
        return (-1 * self) + other

    def __mul__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u * other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u * other.u)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> Self:
        return self * other

    def __matmul__(self, other: ArrayLike) -> Self:
        return self.replace(u=jnp.matmul(self.u, other))

    def __rmatmul__(self, other: ArrayLike) -> Self:
        return self.replace(u=jnp.matmul(other, self.u.squeeze()))

    def __truediv__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u / other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u / other.u)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> Self:
        return self.replace(u=other / self.u)

    def __floordiv__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u // other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u // other.u)
        else:
            return NotImplemented

    def __rfloordiv__(self, other: Any) -> Self:
        return self.replace(u=other // self.u)

    def __mod__(self, other: ScalarLike | ArrayLike | Field) -> Self:
        if isinstance(other, jnp.ndarray) or isinstance(other, Number):
            return self.replace(u=self.u % other)
        elif isinstance(other, (ScalarField, VectorField)):
            return self.replace(u=self.u % other.u)
        else:
            return NotImplemented

    def __rmod__(self, other: Any) -> Self:
        return self.replace(u=other % self.u)

    def __pow__(self, other: Any) -> Self:
        return self.replace(u=self.u**other)

    def __rpow__(self, other: Any) -> Self:
        return self.replace(u=other**self.u)


class ScalarField(Field):
    @classmethod
    def create(
        cls,
        dx: ScalarLike,
        spectrum: ScalarLike,
        spectral_density: ScalarLike,
        u: Array | None = None,
        shape: tuple[int, int] | None = None,
        origin: ArrayLike | None = None,
    ) -> Self:
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
            origin: If provided, defines a shift in the sampling plane such
                that is is no longer centered at the origin. Should be an array
                of shape `[2,]` in the format `[y, x]`.
        """
        # Parsing dx
        _dx = jnp.atleast_1d(dx)
        if _dx.ndim == 1:
            _dx = jnp.stack([_dx, _dx])
        assert_rank(_dx, 2)  # dx should have shape (2, C) here

        # Parsing spectrum and density
        _spectrum = jnp.atleast_1d(spectrum)
        _spectral_density = jnp.atleast_1d(spectral_density)
        assert_equal_shape([_spectrum, _spectral_density])
        _spectral_density = _spectral_density / jnp.sum(_spectral_density)

        # Parsing u
        if u is None:
            assert shape is not None, "Must specify shape if u is None"
            u = jnp.empty((1, *shape, spectrum.size, 1), dtype=jnp.complex64)
        ndim = len(u.shape)
        assert ndim >= 5, (
            "Field must be Array with at least 5 dimensions: (B... H W C 1)."
        )
        assert u.shape[-1] == 1, "Last dimension must be 1 for scalar fields."
        assert_equal_shape([spectrum, spectral_density])
        spectral_density = spectral_density / jnp.sum(spectral_density)
        if dx.ndim == 1:
            dx = jnp.stack([dx, dx])
        assert_rank(dx, 2)  # dx should have shape (2, C) here
        if origin is None:
            origin = jnp.zeros((2, 1))
        elif isinstance(origin, Tuple):
            origin = jnp.array(origin)
        elif isinstance(origin, Number):
            origin = jnp.array([origin, origin])
        if origin.ndim == 1:
            origin = origin[:, None]
        assert_rank(origin, 2)  # shift_yx should have shape (2, C) here
        return cls(u, dx, spectrum, spectral_density, origin)


class VectorField(Field):
    @classmethod
    def create(
        cls,
        dx: ScalarLike,
        spectrum: ScalarLike,
        spectral_density: ScalarLike,
        u: Array | None = None,
        shape: tuple[int, int] | None = None,
        origin: ArrayLike | None = None,
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
            origin: If provided, defines a shift in the sampling plane such
                that is is no longer centered at the origin. Should be an array
                of shape `[2,]` in the format `[y, x]`.
        """

        # Parsing dx
        _dx = jnp.atleast_1d(dx)
        if _dx.ndim == 1:
            _dx = jnp.stack([_dx, _dx])
        assert_rank(_dx, 2)  # dx should have shape (2, C) here

        # Parsing spectrum and density
        _spectrum = jnp.atleast_1d(spectrum)
        _spectral_density = jnp.atleast_1d(spectral_density)
        assert_equal_shape([_spectrum, _spectral_density])
        _spectral_density = _spectral_density / jnp.sum(_spectral_density)

        # Parsing u
        if u is None:
            assert shape is not None, "Must specify shape if u is None"
            u = jnp.empty((1, *shape, spectrum.size, 3), dtype=jnp.complex64)
        ndim = len(u.shape)
        assert ndim >= 5, (
            "Field must be Array with at least 5 dimensions: (B... H W C 3)."
        )
        assert u.shape[-1] == 3, "Last dimension must be 3 for vectorial fields."
        assert_equal_shape([spectrum, spectral_density])
        spectral_density = spectral_density / jnp.sum(spectral_density)
        if dx.ndim == 1:
            dx = jnp.stack([dx, dx])
        assert_rank(dx, 2)  # dx should have shape (2, C) here
        if origin is None:
            origin = jnp.zeros((2, 1))
        assert_rank(origin, 2)  # shift_yx should have shape (2, C) here
        assert origin.shape[0] == 2
        return cls(u, dx, spectrum, spectral_density, origin)

    @property
    def jones_vector(self) -> Array:
        """Return Jones vector of field."""
        norm = jnp.linalg.norm(self.u, axis=-1, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # set to 1 to avoid division by zero
        return self.u / norm


def pad(field: Field, pad_width: int | tuple[int, int], cval: float = 0) -> Field:
    """
    Pad the `field` with zeros in one or two dimensions.
    Args:
        field: The field to pad.
        pad_width: The number of pixels to pad the field with.
        cval: The value to pad the field with (defauls is zero).
    """
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)
    u = jnp.pad(
        field.u,
        [(n, n) for n in (0,) * (field.ndim - 4) + (*pad_width, 0, 0)],
        constant_values=cval,
    )
    return field.replace(u=u)


def crop(field: Field, crop_width: int | tuple[int, int]) -> Field:
    """
    Crop the `field` by removing pixels from the edges.
    Args:
        field: The field to crop.
        crop_width: The number of pixels to remove from the edges.
    """
    if isinstance(crop_width, int):
        crop_width = (crop_width, crop_width)
    crop = [
        slice(n, size - n)
        for size, n in zip(field.shape, (0,) * (field.ndim - 4) + (*crop_width, 0, 0))
    ]
    return field.replace(u=field.u[tuple(crop)])


def shift_grid(field: Field, shift_yx: ScalarLike) -> Field:
    """
    Shift the sampling grid by an arbitrary amount in y and x directions.
    Args:
        shift_yx: The shift in y and x directions. Should be an array of
            shape `[2,]` in the format `[y, x]`.
    """
    if isinstance(shift_yx, Number):
        shift_yx = jnp.array([shift_yx, shift_yx])
    shift_yx = jnp.array(shift_yx)  # Ensure it is an array
    if shift_yx.ndim == 1:
        shift_yx = shift_yx[:, None]
    assert_rank(shift_yx, 2)
    return field.replace(_origin=shift_yx)


def shift_field(field: Field, shiftby: int | tuple[int, int]) -> Field:
    """
    Shift `field` by an integer number of pixels in one or two dimensions,
    while keeping the sampling grid centered at the origin.

    Args:
        field: The field to shift.
        shiftby: The number of pixels to shift the field by.

    See also shift_ft for subpixel shifts.
    """
    if isinstance(shiftby, int):
        shiftby = (shiftby, shiftby)

    crop = [
        (slice(n, dsize) if (n > 0) else slice(0, dsize + n))
        for dsize, n in zip(field.shape, (0,) * (field.ndim - 4) + (*shiftby, 0, 0))
    ]

    pads = [
        ((0, n) if (n > 0) else (-n, 0))
        for n in ((0,) * (field.ndim - 4) + (*shiftby, 0, 0))
    ]
    u = jnp.pad(field.u[tuple(crop)], pads)

    return field.replace(u=u)


def cartesian_to_spherical(field: Field, n: float, NA: float, f: float) -> Array:
    """
    Converts the field to a spherical basis. This is useful for high NA lenses.

    Args:
        field: The incoming ``Field`` in pupil space, in Cartesian coordinates.
        n: Refractive index of the lens.
        NA: NA of the lens.
        f: Focal length of the lens.

    Returns:
        The Field.u in spherical coordinates.
        !!! warning
            Caution: does NOT return a full Field object.
    """
    pupil_radius = f * NA / n
    mask = field.grid[0] ** 2 + field.grid[1] ** 2 <= pupil_radius**2
    sin_theta2 = jnp.sum(field.grid**2, axis=0) * mask / f**2
    cos_theta = jnp.sqrt(1 - sin_theta2)
    sin_theta = jnp.sqrt(sin_theta2)

    phi = jnp.arctan2(field.grid[0], field.grid[1])
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    field_x = field.u[:, :, :, :, 2][..., None]
    field_y = field.u[:, :, :, :, 1][..., None]

    # Source: Eq. (6) of arXiv:2502.03170
    e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x + (
        cos_theta - 1.0
    ) * sin_2phi * field_y
    e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y + (
        cos_theta - 1.0
    ) * sin_2phi * field_x
    e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

    return jnp.stack([e_inf_z, e_inf_y, e_inf_x], axis=-1).squeeze(-2) / 2

import abc
from enum import IntEnum
from typing import ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Complex, Float, ScalarLike

from chromatix.core.base import (
    Chromatic,
    Monochromatic,
    Scalar,
    Vector,
    _strict_config,
    replace,
)
from chromatix.core.spectrum import MonoSpectrum, Spectrum
from chromatix.typing import wv
from chromatix.utils import _broadcast_1d_to_channels, _broadcast_dx_to_grid, l2_sq_norm


def grid(
    shape: tuple[int, int], spacing: Float[Array, "... d"]
) -> Float[Array, "... y x d"]:
    N_y, N_x = shape
    dx = rearrange(spacing, "... d -> ... 1 1 d")
    grid = jnp.meshgrid(
        jnp.linspace(0, (N_y - 1), N_y) - N_y / 2,
        jnp.linspace(0, (N_x - 1), N_x) - N_x / 2,
        indexing="ij",
    )
    return dx * jnp.stack(grid, axis=-1)


def freq_grid(
    shape: tuple[int, int], spacing: Float[Array, "... d"]
) -> Float[Array, "... y x d"]:
    N_y, N_x = shape
    dk = rearrange(1 / spacing, "... d -> ... 1 1 d")
    grid = jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(N_y)),
        jnp.fft.fftshift(jnp.fft.fftfreq(N_x)),
        indexing="ij",
    )
    return dk * jnp.stack(grid, axis=-1)


class Field(eqx.Module, strict=_strict_config):
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

    The shape of a ``Field`` object is determined by the type of `Field` it
    is. In the simplest case, we have `ScalarField`s which are monochromatic
    and scalar descriptions of the electric field. In this case, the complex
    tensor describing the field is a 2D array of shape `(... height width)`
    where the `...` means that potentially zero or more batch dimensions are
    allowed (by default there are no batch dimensions --- but they can be
    useful to describe e.g. a field at multiple depth planes or at different
    time points). For fields that have multiple wavelengths (i.e. a spectrum),
    `ChromaticScalarField`s are used which have the 3D shape `(... height width
    wavelengths)`.

    Any Chromatix functions that produce multiple depths (e.g. propagation
    to multiple z values) will automatically create a batch dimension. If
    more dimensions are required, we encourage the use of ``jax.vmap``,
    ``jax.pmap``, or a combination of the two. We intend for these zero-or-more
    batch dimensions to be a compromise between not having too many dimensions
    when they are not required, and also not having to litter a program with
    ``jax.vmap`` transformations for common simulations in 3D.

    Due to this shape, in order to ensure that attributes of ``Field``
    objects broadcast appropriately, attributes which could be 1D arrays are
    ensured to have extra singleton dimensions. In order to make the creation
    of ``Field`` objects more convenient, we provide the class methods
    ``Field.build``, ``Field.empty``, and ``Field.zeros`` (detailed below),
    which accepts scalar or 1D array arguments for the various attributes
    (e.g. if a single wavelength is desired, a scalar value can be used, but if
    multiple wavelengths are desired, a 1D array can be used for the value of
    ``spectrum``). These methods appropriately reshapes the attributes provided
    to the correct shapes.

    Attributes:
        u: The complex field of shape at least ``(... height width)``.
        dx: The spacing (pixel size) of the samples of the ``Field`` in units
            of distance. In the simplest case, you can pass a scalar value
            to define this spacing which creates a square pixel. Each sample
            (pixel) of the ``Field`` has a height and width. Spacing is also
            allowed to vary with wavelength of the spectrum. To choose a
            different spacing per wavelength (but still square), you can pass a
            1D array of spacings of the same length as the number of wavelengths
            in the spectrum of the ``Field``. To create a non-square spacing,
            you must always pass a 2D array of shape `(wavelengths 2)` where
            the last dimension has length 2 and defines the height and width of
            the spacing in units of distance. To createa a non-square spacing
            you must always include the `wavelengths` dimension even if there
            is only a single wavelength in the spectrum (i.e. a `(1 2)` shaped
            array).
        origin: The origin of the field's coordinate grid in units of distance.
            By default, fields are initialized with an origin of `(0.0, 0.0)`
            such that their center is aligned with the center of the optical
            axis.
        spectrum: The [``Spectrum``](chromatix.core.spectrum.Spectrum) of the
            ``Field`` to be created. This can be specified either as a single
            float value representing a wavelength in units of distance for
            a monochromatic field, a 1D array of wavelengths for a chromatic
            field that has the same intensity in all wavelengths, or a tuple
            of two 1D arrays where the first array represents the wavelengths
            and the second array is a unitless array of weights that define the
            spectral density (the relative intensity of each wavelength in the
            spectrum). This second array of spectral density will automatically
            be normalized to sum to 1.
    """

    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    origin: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Spectrum]

    # Internal for use
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the complex field."""
        return self.u.shape

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """The height and width of the complex field."""
        return (self.u.shape[self.dims.y], self.u.shape[self.dims.x])

    @property
    def spatial_dims(self) -> tuple[int, int]:
        """Axis indices representing the height and width of the complex field."""
        return (self.dims.y, self.dims.x)

    @property
    def ndim(self) -> int:
        """Number of dimensions (the rank) of the complex field."""
        return self.u.ndim

    @property
    def batch_dims(self) -> tuple[int, ...]:
        return tuple(n for n in range(self.ndim + self.spatial_dims[0]))

    @property
    def num_batch_dims(self) -> int:
        return len(self.batch_dims)

    @property
    @abc.abstractmethod
    def grid(self) -> Array:
        """
        The grid for each spatial dimension as an array. The 2 entries along
        the last dimension represent the y and x grids, respectively. This grid
        assumes that the center of the ``Field`` is the origin and that the
        elements are sampling from the center, not the corner.
        """
        pass

    @property
    @abc.abstractmethod
    def f_grid(self) -> Array:
        """
        The frequency grid for each spatial dimension as an array. The 2 entries
        along the last dimension represent the y and x grids, respectively. This
        grid assumes that the center of the ``Field`` is the origin and that the
        elements are sampling from the center, not the corner.
        """
        pass

    @property
    def k_grid(self) -> Array:
        """
        The angular frequency grid for each spatial dimension as an array.
        This is the same as ``f_grid`` but multiplied by ``2 * jnp.pi``.
        The 2 entries along the last dimension represent the y and x grids,
        respectively. This grid assumes that the center of the ``Field`` is
        the origin and that the elements are sampling from the center, not the
        corner.
        """
        return 2 * jnp.pi * self.f_grid

    @property
    def extent(self) -> Array:
        """
        The extent (lengths in height and width per wavelength) of the field
        in units of distance. Defined as an array of shape ``(2 1... 1 1 C 1)``
        specifying the extent in the y and x dimensions respectively.
        """
        return self.dx * jnp.asarray(self.spatial_shape)

    @property
    def spatial_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Return the spatial limits of the field: (y_min, y_max), (x_min, x_max).
        """
        return (float(self.grid[..., 0].min()), float(self.grid[..., 0].max())), (
            float(self.grid[..., 1].min()),
            float(self.grid[..., 1].max()),
        )

    @property
    def df(self) -> Array:
        """
        The frequency spacing of the samples in the frequency space of ``u``.
        Defined as an array of same shape as ``dx`` specifying the spacing in
        the y and x directions respectively (can be the same for y and x for the
        common case of square pixels).
        """
        return 1 / self.extent

    @property
    def wavelength(self) -> Array:
        """
        The wavelength(s) of the field's spectrum as a float or 1D array.
        """
        return self.spectrum.wavelength

    @property
    def central_wavelength(self) -> Array:
        """
        The central wavelength of the spectrum (defined as the first wavelength
        provided to the spectrum because you could construct the complex field
        with multiple wavelengths in any order).
        """
        return self.spectrum.central_wavelength

    @property
    @abc.abstractmethod
    def central_dx(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def central_rectangular_dx(self) -> Array:
        pass

    @property
    def central_extent(self) -> Array:
        """
        The extent (lengths in height and width per wavelength) of the field
        in units of distance.
        """
        return self.central_dx * jnp.asarray(self.spatial_shape)

    @property
    def broadcasted_wavelength(self) -> Array:
        """
        The wavelength(s) of the field's spectrum, reshaped for appropriate
        broadcasting to the appropriate dimension if there are multiple
        wavelengths.
        """
        return _broadcast_1d_to_channels(self.spectrum.wavelength, self.spatial_dims)

    @property
    @abc.abstractmethod
    def power(self) -> Array:
        """Power of the complex field."""
        pass

    @property
    @abc.abstractmethod
    def intensity(self) -> Array:
        """Intensity of the complex field."""
        pass

    @property
    def k(self) -> Array:
        return 2 * jnp.pi / self.wavelength

    @property
    def surface_area(self) -> Array:
        shape = jnp.asarray(self.spatial_shape)
        return self.dx * shape

    @property
    def phase(self) -> Array:
        return jnp.angle(self.u)

    @property
    def amplitude(self) -> Array:
        return jnp.abs(self.u)

    def replace(self, **kwargs) -> Self:
        return replace(self, **kwargs)

    @classmethod
    def empty_like(
        cls,
        field: Self,
        shape: tuple[int, int] | None = None,
        dx: float | Array | None = None,
        origin: Float[Array, "2"] | None = None,
        spectrum: Spectrum | MonoSpectrum | None = None,
    ) -> Self:
        """
        Copy over attributes of ``field`` to a new ``Field`` object, with the
        option of changing some attributes.

        Note that this function creates a field `u` with a new empty field.
        """
        if dx is None:
            dx = field.dx
        else:
            dx = jnp.asarray(dx)
            assert dx.ndim == field.dx.ndim, (
                "New spacing must have same number of dimensions as old spacing"
            )
        if shape is None:
            shape = field.shape
        else:
            assert len(shape) == 2
            _shape = list(field.shape)
            _shape[field.dims.y] = shape[0]
            _shape[field.dims.x] = shape[1]
            shape = _shape
        if spectrum is None:
            spectrum = field.spectrum
        if not isinstance(field.spectrum, MonoSpectrum) and isinstance(
            spectrum, MonoSpectrum
        ):
            raise ValueError("Changing from Spectrum to MonoSpectrum is not allowed")
        if not isinstance(spectrum, MonoSpectrum):
            if isinstance(field.spectrum, MonoSpectrum):
                raise ValueError(
                    "Changing from MonoSpectrum to Spectrum is not allowed"
                )
            shape[field.dims.wv] = spectrum.size
        shape = tuple(shape)
        if origin is None:
            origin = field.origin
        u = jnp.empty_like(field.u, shape=shape)
        return field.replace(u=u, dx=dx, origin=origin, spectrum=spectrum)

    @classmethod
    def build(
        cls,
        u: Array,
        dx: ScalarLike | Array,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    ) -> Self:
        spectrum = Spectrum.build(spectrum)
        match (u.ndim, isinstance(spectrum, MonoSpectrum)):
            case (2, True):
                field = ScalarField
            case (3, True):
                field = VectorField
            case (3, False):
                field = ChromaticScalarField
            case (4, False):
                field = ChromaticVectorField
        return field(u, dx, 0.0, spectrum)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, int],
        dx: ScalarLike | Array,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        scalar: bool = True,
    ) -> Self:
        spectrum = Spectrum.build(spectrum)
        match (isinstance(spectrum, MonoSpectrum), scalar):
            case (True, True):
                u_empty = jnp.empty(shape, dtype=jnp.complex64)
            case (False, True):
                u_empty = jnp.empty((*shape, spectrum.size), dtype=jnp.complex64)
            case (True, False):
                u_empty = jnp.empty((*shape, 3), dtype=jnp.complex64)
            case (False, False):
                u_empty = jnp.empty((*shape, spectrum.size, 3), dtype=jnp.complex64)
        return Field.build(u_empty, dx, spectrum)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        dx: ScalarLike | Array,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        scalar: bool = True,
    ) -> Self:
        spectrum = Spectrum.build(spectrum)
        match (spectrum.size, scalar):
            case (1, True):
                u_zeros = jnp.zeros(shape, dtype=jnp.complex64)
            case (_, True):
                u_zeros = jnp.zeros((*shape, spectrum.size), dtype=jnp.complex64)
            case (1, False):
                u_zeros = jnp.zeros((*shape, 3), dtype=jnp.complex64)
            case (_, False):
                u_zeros = jnp.zeros((*shape, spectrum.size, 3), dtype=jnp.complex64)
        return Field.build(u_zeros, dx, spectrum)

    @property
    def conj(self) -> Self:
        return self.replace(u=jnp.conj(self.u))

    def expand_dims(self) -> Self:
        return self.replace(u=self.u[jnp.newaxis, ...])

    def unsqueeze(self) -> Self:
        return self.expand_dims()

    def squeeze(self) -> Self:
        return self.replace(u=self.u.squeeze(axis=self.batch_dims))

    def __add__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u + other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u + other)

    def __radd__(self, other: float | Array | Self) -> Self:
        return self + other

    def __sub__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u - other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u - other)

    def __rsub__(self, other: float | Array | Self) -> Self:
        return (-1 * self) + other

    def __mul__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u * other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u * other)

    def __rmul__(self, other: float | Array | Self) -> Self:
        return self * other

    def __matmul__(self, other: Array) -> Self:
        return self.replace(u=jnp.matmul(self.u, other))

    def __rmatmul__(self, other: Array) -> Self:
        return self.replace(u=jnp.matmul(other, self.u))

    def __truediv__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u / other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u / other)

    def __rtruediv__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other / self.u)

    def __floordiv__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u // other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u // other)

    def __rfloordiv__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other // self.u)

    def __mod__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u % other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u % other)

    def __rmod__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other % self.u)

    def __pow__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=self.u**other.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=self.u**other)

    def __rpow__(self, other: float | Array | Self) -> Self:
        if isinstance(other, Field):
            # TODO: Make sure Field types are the same
            return self.replace(u=other.u**self.u)
        else:
            # TODO: Make sure shapes match
            return self.replace(u=other**self.u)


def pad(field: Field, pad_width: int | tuple[int, int], cval: float = 0) -> Field:
    """
    Pad the `field` with zeros by the desired amount.
    Args:
        field: The field to pad.
        pad_width: The number of pixels to pad the field with.
        cval: The value to pad the field with (defauls is zero).
    """
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)
    spatial_dims = [field.ndim + d for d in field.spatial_dims]
    pad = [(0, 0) for d in range(field.ndim)]
    pad[spatial_dims[0]] = (pad_width[0], pad_width[0])
    pad[spatial_dims[1]] = (pad_width[1], pad_width[1])
    u = jnp.pad(field.u, pad, constant_values=cval)
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
    spatial_dims = [field.ndim + d for d in field.spatial_dims]
    crop = [slice(0, field.shape[d]) for d in range(field.ndim)]
    crop[spatial_dims[0]] = slice(
        crop_width[0], field.shape[spatial_dims[0]] - crop_width[0]
    )
    crop[spatial_dims[1]] = slice(
        crop_width[1], field.shape[spatial_dims[1]] - crop_width[1]
    )
    u = field.u[tuple(crop)]
    return field.replace(u=u)


def shift_grid(field: Field, shift_yx: ScalarLike | Float[Array, "2"]) -> Field:
    """
    Shift the sampling grid by an arbitrary amount in y and x directions.
    Args:
        shift_yx: The shift in y and x directions. Should be an array of
            shape `[2,]` in the format `[y, x]`.
    """
    shift_yx = jnp.atleast_1d(jnp.asarray(shift_yx))
    if shift_yx.size < 2:
        shift_yx = jnp.array([shift_yx, shift_yx])
    assert shift_yx.size == 2, "Shift must be an array of size (2,) in the order (y x)"
    shift_yx = jnp.array(shift_yx)
    return field.replace(origin=jnp.zeros_like(field.origin) + shift_yx)


def shift_field(field: Field, shift_by: int | tuple[int, int]) -> Field:
    """
    Shift `field` by an integer number of pixels in one or two dimensions,
    while keeping the sampling grid centered at the origin.

    Args:
        field: The field to shift.
        shift_by: The number of pixels to shift the field by.
    """
    if isinstance(shift_by, int):
        shift_by = (shift_by, shift_by)
    spatial_dims = [field.ndim + d for d in field.spatial_dims]
    crop = [slice(0, field.shape[d]) for d in range(field.ndim)]
    pad = [(0, 0) for d in range(field.ndim)]
    for i in range(2):
        limits = (
            (shift_by[i], field.shape[spatial_dims[i]])
            if shift_by[i] > 0
            else (0, field.shape[spatial_dims[i]] + shift_by[i])
        )
        crop[spatial_dims[i]] = slice(*limits)
        pad[spatial_dims[i]] = (
            (0, shift_by[i]) if shift_by[i] > i else (-shift_by[i], 0)
        )
    u = jnp.pad(field.u[tuple(crop)], pad)
    return field.replace(u=u)


class ScalarField(Field, Monochromatic, Scalar, strict=True):
    u: Complex[Array, "y x"]
    dx: Float[Array, "2"]
    origin: Float[Array, "2"]
    spectrum: MonoSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -2), ("x", -1)])

    def __init__(
        self,
        u: Complex[Array, "y x"],
        dx: ScalarLike | Float[Array, "1 2"],
        origin: ScalarLike | Float[Array, "1 2"],
        spectrum: MonoSpectrum | float,
    ):
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        self.dx = rearrange(_broadcast_dx_to_grid(dx, spectrum.size), "1 d -> d", d=2)
        self.origin = rearrange(
            _broadcast_dx_to_grid(origin, spectrum.size), "1 d -> d", d=2
        )
        self.spectrum = spectrum
        assert self.u.ndim == 2, f"Expected 2-dimensional field, got shape {u.shape}."

    @property
    def grid(self) -> Float[Array, "y x d"]:
        return grid(self.spatial_shape, self.dx) + self.origin

    @property
    def f_grid(self) -> Float[Array, "y x d"]:
        return freq_grid(self.spatial_shape, self.dx)

    @property
    def central_dx(self) -> float:
        return self.dx[0]

    @property
    def central_rectangular_dx(self) -> Array:
        return self.dx

    @property
    def power(self) -> Array:
        area = jnp.prod(self.dx, axis=-1)
        power_density = jnp.sum(self.intensity, axis=(-2, -1))
        power_density = rearrange(power_density, "... -> ... 1 1")
        return area * power_density

    @property
    def intensity(self) -> Array:
        return jnp.abs(self.u) ** 2


class ChromaticScalarField(Field, Chromatic, Scalar, strict=True):
    u: Complex[Array, "y x wv"]
    dx: Float[Array, "wv 2"]
    origin: Float[Array, "wv 2"]
    spectrum: Spectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("wv", -1)])

    def __init__(
        self,
        u: Complex[Array, "y x wv"],
        dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
        origin: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
        spectrum: Spectrum,
    ):
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        self.dx = _broadcast_dx_to_grid(dx, spectrum.size)
        self.origin = _broadcast_dx_to_grid(origin, spectrum.size)
        self.spectrum = spectrum
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == self.wavelength.size, (
            "Expected last dimension of u to be same as wavelengths."
        )

    @property
    def grid(self) -> Float[Array, "y x wv d"]:
        _grid = grid(self.spatial_shape, self.dx)
        return (
            rearrange(
                _grid,
                "... wv y x d-> ... y x wv d",
                wv=self.spectrum.size,
                y=self.spatial_shape[0],
                x=self.spatial_shape[1],
                d=2,
            )
            + self.origin
        )

    @property
    def f_grid(self) -> Float[Array, "y x wv d"]:
        _freq_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(
            _freq_grid,
            "... wv y x d-> ... y x wv d",
            wv=self.spectrum.size,
            y=self.spatial_shape[0],
            x=self.spatial_shape[1],
            d=2,
        )

    @property
    def central_dx(self) -> float:
        return self.dx[0, 0]

    @property
    def central_rectangular_dx(self) -> Array:
        return self.dx[0]

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        power_density = jnp.sum(self.intensity, axis=(-2, -1))
        power_density = rearrange(power_density, "... -> ... 1 1 1")
        return area * power_density

    @property
    def intensity(self):
        spectral_density = _broadcast_1d_to_channels(
            self.spectrum.density, self.spatial_dims
        )
        return jnp.sum(spectral_density * jnp.abs(self.u) ** 2, axis=self.dims.wv)


class VectorField(Field, Monochromatic, Vector, strict=True):
    u: Complex[Array, "y x 3"]
    dx: Float[Array, "1 2"]
    origin: Float[Array, "1 2"]
    spectrum: MonoSpectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum("dims", [("y", -3), ("x", -2), ("p", -1)])

    def __init__(
        self,
        u: Complex[Array, "y x 3"],
        dx: ScalarLike | Float[Array, "1 2"],
        origin: ScalarLike | Float[Array, "1 2"],
        spectrum: MonoSpectrum,
    ):
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        self.dx = _broadcast_dx_to_grid(dx, spectrum.size)
        self.origin = _broadcast_dx_to_grid(origin, spectrum.size)
        self.spectrum = spectrum
        assert self.u.ndim == 3, f"Expected 3-dimensional field, got shape {u.shape}."
        assert self.u.shape[-1] == 3, (
            f"Expected last dimension of u to be 3, got {u.shape[-1]}"
        )

    @property
    def grid(self) -> Float[Array, "y x 1 d"]:
        _grid = grid(self.spatial_shape, self.dx)
        return (
            rearrange(
                _grid,
                "... 1 y x d-> ... y x 1 d",
                y=self.spatial_shape[0],
                x=self.spatial_shape[1],
                d=2,
            )
            + self.origin
        )

    @property
    def f_grid(self) -> Float[Array, "y x 1 d"]:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(
            _f_grid,
            "... 1 y x d-> ... y x 1 d",
            y=self.spatial_shape[0],
            x=self.spatial_shape[1],
            d=2,
        )

    @property
    def central_dx(self) -> float:
        return self.dx[0, 0]

    @property
    def central_rectangular_dx(self) -> Array:
        return self.dx[0]

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1).squeeze()
        power_density = jnp.sum(self.intensity, axis=(-2, -1))
        power_density = rearrange(power_density, "... -> ... 1 1 1")
        return area * power_density

    @property
    def intensity(self):
        return jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)


class ChromaticVectorField(Field, Chromatic, Vector, strict=True):
    u: Complex[Array, "y x wv 3"]
    dx: Float[Array, "wv 1 2"]
    origin: Float[Array, "wv 1 2"]
    spectrum: Spectrum

    # Internal
    dims: ClassVar[IntEnum] = IntEnum(
        "dims", [("y", -4), ("x", -3), ("wv", -2), ("p", -1)]
    )

    def __init__(
        self,
        u: Complex[Array, "y x wv 3"],
        dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
        origin: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
        spectrum: Spectrum,
    ):
        self.u = jnp.asarray(u, dtype=jnp.complex64)
        self.dx = rearrange(
            _broadcast_dx_to_grid(dx, spectrum.size), "wv d -> wv 1 d", d=2
        )
        self.origin = rearrange(
            _broadcast_dx_to_grid(origin, spectrum.size), "wv d -> wv 1 d", d=2
        )
        self.spectrum = spectrum
        assert self.u.ndim == 4, f"Expected 4-dimensional field, got shape {u.shape}."
        assert self.u.shape[-2] == self.wavelength.size, (
            "Expected last dimension of u to be same as wavelengths."
        )
        assert self.u.shape[-1] == 3, (
            f"Expected last dimension of u to be 3, got {u.shape[-1]}"
        )

    @property
    def grid(self) -> Float[Array, "y x wv 1 d"]:
        _grid = grid(self.spatial_shape, self.dx)
        return (
            rearrange(
                _grid,
                "... wv 1 y x d-> ... y x wv 1 d",
                wv=self.spectrum.size,
                y=self.spatial_shape[0],
                x=self.spatial_shape[1],
                d=2,
            )
            + self.origin
        )

    @property
    def f_grid(self) -> Float[Array, "y x wv 1 d"]:
        _f_grid = freq_grid(self.spatial_shape, self.dx)
        return rearrange(
            _f_grid,
            "... wv 1 y x d-> ... y x wv 1 d",
            wv=self.spectrum.size,
            y=self.spatial_shape[0],
            x=self.spatial_shape[1],
            d=2,
        )

    @property
    def central_dx(self) -> float:
        return self.dx[0, 0, 0]

    @property
    def central_rectangular_dx(self) -> Array:
        return self.dx[0, 0]

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        power_density = jnp.sum(self.intensity, axis=(-2, -1))
        power_density = rearrange(power_density, "... -> ... 1 1 1 1")
        return area * power_density

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... wv -> ... 1 1 wv 1")
        return jnp.sum(
            spectral_density * jnp.abs(self.u) ** 2, axis=(self.dims.wv, self.dims.p)
        )


def cartesian_to_spherical(
    field: VectorField | ChromaticVectorField, n: float, NA: float, f: float
) -> Array:
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
            Caution, does NOT return a full Field object.
    """
    assert isinstance(field, Vector), "Must be a vectorial Field"
    pupil_radius = f * NA / n
    radius_sq = l2_sq_norm(field.grid)
    mask = radius_sq <= pupil_radius**2
    sin_theta2 = radius_sq * mask / f**2
    cos_theta = jnp.sqrt(1 - sin_theta2)
    sin_theta = jnp.sqrt(sin_theta2)

    phi = jnp.arctan2(field.grid[..., 0], field.grid[..., 1])
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    field_x = field.u[..., 2][..., jnp.newaxis]
    field_y = field.u[..., 1][..., jnp.newaxis]

    # Source: Eq. (6) of arXiv:2502.03170
    e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x + (
        cos_theta - 1.0
    ) * sin_2phi * field_y
    e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y + (
        cos_theta - 1.0
    ) * sin_2phi * field_x
    e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

    return jnp.concatenate([e_inf_z, e_inf_y, e_inf_x], axis=-1) / 2

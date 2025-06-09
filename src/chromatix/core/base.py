import abc
from enum import IntEnum
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float
from spectrum import Spectrum, MonoSpectrum


_strict_config = eqx.StrictConfig(allow_abstract_name=True)


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
        """Only the height and width of the complex field."""
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
    @abc.abstractmethod
    def grid(self) -> Array:
        """
        The grid for each spatial dimension as an array of shape `(2 B... H W
        C 1)`. The 2 entries along the first dimension represent the y and x
        grids, respectively. This grid assumes that the center of the ``Field``
        is the origin and that the elements are sampling from the center, not
        the corner.
        """
        pass

    @property
    @abc.abstractmethod
    def f_grid(self) -> Array:
        """
        The frequency grid for each spatial dimension as an array of shape `(2
        B... H W C 1)`. The 2 entries along the first dimension represent the
        y and x grids, respectively. This grid assumes that the center of the
        ``Field`` is the origin and that the elements are sampling from the
        center, not the corner.
        """
        pass

    @property
    def k_grid(self) -> Float[Array, "y x d"]:
        """
        The angular frequency grid for each spatial dimension as an array
        of shape `(2 B... H W C 1)`. The 2 entries along the first dimension
        represent the y and x grids, respectively. This grid assumes that the
        center of the ``Field`` is the origin and that the elements are sampling
        from the center, not the corner.
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
    def df(self) -> Array:
        """
        The frequency spacing of the samples in the frequency space of ``u``.
        Defined as an array of shape ``(2 1... 1 1 C 1)`` specifying the spacing
        in the y and x directions respectively (can be the same for y and x for
        the common case of square pixels). Spacing is the same per wavelength
        for all entries in a batch.
        """
        return 1 / self.extent

    @property
    @abc.abstractmethod
    def power(self):
        """Power of the complex field, shape `(B... 1 1 1)`."""
        pass

    @property
    @abc.abstractmethod
    def intensity(self):
        """Intensity of the complex field, shape `(B... H W 1 1)`."""
        pass

    @property
    def k(self) -> Array:
        return 2 * jnp.pi / self.wavelength

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

    def replace(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            result = eqx.tree_at(lambda tree: getattr(tree, key), self, value)
        return result

    @property
    def conj(self) -> Array:
        return self.replace(u=jnp.conj(self.u))

    def __add__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u + f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u + arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u + s.u)

    def __radd__(self, other: float | Array | Self) -> Self:
        return self + other

    def __sub__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u - f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u - arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u - s.u)
            case _:
                return NotImplemented

    def __rsub__(self, other: float | Array | Self) -> Self:
        return (-1 * self) + other

    def __mul__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u * f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u * arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u * s.u)
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Array | Self) -> Self:
        return self * other

    def __matmul__(self, other: Array) -> Self:
        return self.replace(u=jnp.matmul(self.u, other))

    def __rmatmul__(self, other: Array) -> Self:
        return self.replace(u=jnp.matmul(other, self.u))

    def __truediv__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u / f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u / arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u / s.u)
            case _:
                return NotImplemented

    def __rtruediv__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other / self.u)

    def __floordiv__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u // f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u // arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u // s.u)
            case _:
                return NotImplemented

    def __rfloordiv__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other // self.u)

    def __mod__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u % f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u % arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u % s.u)
            case _:
                return NotImplemented

    def __rmod__(self, other: float | Array | Self) -> Self:
        return self.replace(u=other % self.u)

    def __pow__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=self.u % f)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=self.u % arr)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=self.u % s.u)
            case _:
                return NotImplemented

    def __rpow__(self, other: float | Array | Self) -> Self:
        match other:
            case float() as f:
                return self.replace(u=f ** self.u)
            case Array() as arr:
                # TODO: Make sure shapes match
                return self.replace(u=arr ** self.u)
            case Self() as s:
                # TODO: Make sure Field types are the same
                return self.replace(u=s.u % self.u)
            case _:
                return NotImplemented


class Monochromatic(eqx.Module, strict=_strict_config):
    spectrum: eqx.AbstractVar[MonoSpectrum]

    @property
    def wavelength(self) -> MonoSpectrum:
        return self.spectrum


class Chromatic(eqx.Module, strict=_strict_config):
    spectrum: eqx.AbstractVar[Spectrum]


class Scalar(eqx.Module, strict=_strict_config):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Spectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def wavelength(self) -> Array:
        return self.spectrum.wavelength


class Vector(eqx.Module, strict=_strict_config):
    u: eqx.AbstractVar[Array]
    dx: eqx.AbstractVar[Array]
    spectrum: eqx.AbstractVar[Spectrum]
    dims: eqx.AbstractClassVar[IntEnum]

    @property
    def jones_vector(self) -> Array:
        norm = jnp.linalg.norm(self.u, axis=self.dims.p, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)  # set to 1 to avoid division by zero
        return self.u / norm

    @property
    def power(self):
        area = jnp.prod(self.dx, axis=-1)
        total_intensity = self.spectrum.density * jnp.sum(
            jnp.abs(self.u) ** 2, axis=(self.dims.p, self.dims.y, self.dims.x)
        )
        return area * total_intensity

    @property
    def wavelength(self) -> Array:
        return rearrange(self.spectrum.wavelength, "wv -> wv 1")

    @property
    def intensity(self):
        spectral_density = rearrange(self.spectrum.density, "... wv -> ... 1 1 wv")
        return spectral_density * jnp.sum(jnp.abs(self.u) ** 2, axis=self.dims.p)

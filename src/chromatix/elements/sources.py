from typing import Callable

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix import Field, ScalarField, Spectrum, VectorField
from chromatix.functional.sources import (
    generic_field,
    objective_point_source,
    plane_wave,
    point_source,
)
from chromatix.typing import wv, z

__all__ = [
    "PointSource",
    "ObjectivePointSource",
    "PlaneWave",
    "GenericField",
]

FieldPupil = Callable[[Field], Field]


class PointSource(eqx.Module):
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created in
            number of samples (pixels).
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
        n: Refractive index.
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For scalar ``Field``s
            this doesnt do anything but scale the field (which will be undone
            if ``power`` is not ``None``), but it is required for vectorial
            ``Field``s to set the polarization.
        offset: The offset of the point source in the plane in units of
            distance. Should be a tuple or an array of shape `(2,)` in the
            order `y x`.
        pupil: A function that applies a pupil to the field if provided.
            Defaults to ``None`` in which case the field is unchanged.
        scalar: Whether the result should be ``ScalarField`` (if ``True``) or
            ``VectorField`` (if ``False``). Defaults to ``True``.
        epsilon: Value added to denominators for numerical stability when z is 0.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    dx: ScalarLike | Float[Array, "2"]
    spectrum: (
        Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]]
    )
    n: ScalarLike
    power: ScalarLike | None
    amplitude: ScalarLike | Float[Array, "3"]
    offset: Float[Array, "2"] | tuple[float, float]
    pupil: FieldPupil | None = eqx.field(static=True)
    scalar: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int],
        dx: ScalarLike | Float[Array, "2"],
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        n: ScalarLike,
        power: ScalarLike | None = 1.0,
        amplitude: ScalarLike | Float[Array, "3"] = 1.0,
        offset: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
        pupil: FieldPupil | None = None,
        scalar: bool = True,
        epsilon: float = float(np.finfo(np.float32).eps),
    ) -> ScalarField | VectorField:
        """
        Generates field due to point source a distance ``z`` away.

        Can also be given ``pupil``.

        Args:
            shape: The shape (height and width) of the ``Field`` to be created
                in number of samples (pixels).
            dx: The spacing (pixel size) of the samples of the ``Field`` in
                units of distance. In the simplest case, you can pass a scalar
                value to define this spacing which creates a square pixel. Each
                sample (pixel) of the ``Field`` has a height and width. Spacing
                is also allowed to vary with wavelength of the spectrum. To
                choose a different spacing per wavelength (but still square),
                you can pass a 1D array of spacings of the same length as the
                number of wavelengths in the spectrum of the ``Field``. To
                create a non-square spacing, you must always pass a 2D array
                of shape `(wavelengths 2)` where the last dimension has length
                2 and defines the height and width of the spacing in units
                of distance. To createa a non-square spacing you must always
                include the `wavelengths` dimension even if there is only a
                single wavelength in the spectrum (i.e. a `(1 2)` shaped array).
            spectrum: The [``Spectrum``](chromatix.core.spectrum.Spectrum)
                of the ``Field`` to be created. This can be specified either
                as a single float value representing a wavelength in units of
                distance for a monochromatic field, a 1D array of wavelengths
                for a chromatic field that has the same intensity in all
                wavelengths, or a tuple of two 1D arrays where the first array
                represents the wavelengths and the second array is a unitless
                array of weights that define the spectral density (the relative
                intensity of each wavelength in the spectrum). This second array
                of spectral density will automatically be normalized to sum
                to 1.
            n: Refractive index.
            power: The total power that the result should be normalized to,
                defaults to 1.0. If ``None``, no normalization occurs.
            amplitude: The amplitude of the electric field. For scalar
                ``Field``s this doesnt do anything but scale the field (which
                will be undone if ``power`` is not ``None``), but it is required
                for vectorial ``Field``s to set the polarization.
            offset: The offset of the point source in the plane in units of
                distance. Should be a tuple or an array of shape `(2,)` in the
                order `y x`.
            pupil: A function that applies a pupil to the field if provided.
                Defaults to ``None`` in which case the field is unchanged.
            scalar: Whether the result should be ``ScalarField`` (if ``True``)
                or ``VectorField`` (if ``False``). Defaults to ``True``.
            epsilon: Value added to denominators for numerical stability when z
                is 0.
        """
        self.shape = shape
        self.dx = dx
        self.spectrum = (spectrum,)
        self.n = n
        self.power = power
        self.amplitude = amplitude
        self.offset = offset
        self.pupil = pupil
        self.scalar = scalar
        self.epsilon = epsilon

    def __call__(self, z: ScalarLike | Float[Array, "z"]) -> ScalarField | VectorField:
        """
        Generates field due to point source a distance ``z`` away.

        Args:
            z: How far away the point source is in units of distance.
        """
        return point_source(
            self.shape,
            self.dx,
            self.spectrum,
            z,
            self.n,
            self.power,
            self.amplitude,
            self.offset,
            self.pupil,
            self.scalar,
            self.epsilon,
        )


class ObjectivePointSource(eqx.Module):
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a thin lens with focal length
    ``f`` and numerical aperture ``NA``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created in
            number of samples (pixels).
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
        f: Focal length of the objective lens in units of distance.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For scalar ``Field``s
            this doesnt do anything but scale the field (which will be undone
            if ``power`` is not ``None``), but it is required for vectorial
            ``Field``s to set the polarization.
        offset: The offset of the point source in the plane in units of
            distance. Should be a tuple or an array of shape `(2,)` in the order
            `y x`.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    dx: ScalarLike | Float[Array, "2"]
    spectrum: (
        Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]]
    )
    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike
    power: ScalarLike | None
    amplitude: ScalarLike
    offset: Float[Array, "2"] | tuple[float, float]
    scalar: bool = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int],
        dx: ScalarLike,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike,
        power: ScalarLike | None = 1.0,
        amplitude: ScalarLike | Float[Array, "3"] = 1.0,
        offset: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
        scalar: bool = True,
    ):
        """
        Generates field due to a point source defocused by an amount ``z`` away
        from the focal plane, just after passing through a thin lens with focal
        length ``f`` and numerical aperture ``NA``.

        Args:
            shape: The shape (height and width) of the ``Field`` to be created
                in number of samples (pixels).
            dx: The spacing (pixel size) of the samples of the ``Field`` in
                units of distance. In the simplest case, you can pass a scalar
                value to define this spacing which creates a square pixel. Each
                sample (pixel) of the ``Field`` has a height and width. Spacing
                is also allowed to vary with wavelength of the spectrum. To
                choose a different spacing per wavelength (but still square),
                you can pass a 1D array of spacings of the same length as the
                number of wavelengths in the spectrum of the ``Field``. To
                create a non-square spacing, you must always pass a 2D array
                of shape `(wavelengths 2)` where the last dimension has length
                2 and defines the height and width of the spacing in units
                of distance. To createa a non-square spacing you must always
                include the `wavelengths` dimension even if there is only a
                single wavelength in the spectrum (i.e. a `(1 2)` shaped array).
            spectrum: The [``Spectrum``](chromatix.core.spectrum.Spectrum)
                of the ``Field`` to be created. This can be specified either
                as a single float value representing a wavelength in units of
                distance for a monochromatic field, a 1D array of wavelengths
                for a chromatic field that has the same intensity in all
                wavelengths, or a tuple of two 1D arrays where the first array
                represents the wavelengths and the second array is a unitless
                array of weights that define the spectral density (the relative
                intensity of each wavelength in the spectrum). This second array
                of spectral density will automatically be normalized to sum
                to 1.
            f: Focal length of the objective lens in units of distance.
            n: Refractive index.
            NA: The numerical aperture of the objective lens.
            power: The total power that the result should be normalized to,
                defaults to 1.0. If ``None``, no normalization occurs.
            amplitude: The amplitude of the electric field. For scalar
                ``Field``s this doesnt do anything but scale the field (which
                will be undone if ``power`` is not ``None``), but it is required
                for vectorial ``Field``s to set the polarization.
            offset: The offset of the point source in the plane in units of
                distance. Should be a tuple or an array of shape `(2,)` in the
                order `y x`.
            scalar: Whether the result should be ``ScalarField`` (if True) or
                ``VectorField`` (if False). Defaults to True.
        """
        self.shape = shape
        self.dx = dx
        self.spectrum = spectrum
        self.f = f
        self.n = n
        self.NA = NA
        self.power = power
        self.amplitude = amplitude
        self.offset = offset
        self.scalar = scalar

    def __call__(self, z: ScalarLike | Float[Array, "z"]) -> ScalarField | VectorField:
        """
        Generates field due to a point source defocused by an amount ``z`` away
        from the focal plane, just after passing through a thin lens with focal
        length ``f`` and numerical aperture ``NA``.

        Args:
            z: How far away the point source is from the focal plane of the lens
                in units of distance.
        """
        return objective_point_source(
            self.shape,
            self.dx,
            self.spectrum,
            z,
            self.f,
            self.n,
            self.NA,
            self.power,
            self.amplitude,
            self.offset,
            self.scalar,
        )


class PlaneWave(eqx.Module):
    """
    Generates plane wave of given ``power``, as ``exp(1j)`` at each location of
    the field. Can also be given ``pupil`` and ``kykx`` vector to control the
    angle of the plane wave. If a ``kykx`` wave vector is provided, the plane
    wave is constructed as ``exp(1j * jnp.sum(kykx * field.grid, axis=-1))``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created in
            number of samples (pixels).
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
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For scalar ``Field``s
            this doesnt do anything but scale the field (which will be undone
            if ``power`` is not ``None``), but it is required for vectorial
            ``Field``s to set the polarization.
        kykx: Defines the orientation of the plane wave. Should be a tuple or an
            array of shape `(2,)` in the format `[ky kx]`. We assume that these
            are wave vectors, i.e. that they have already been multiplied by ``2
            * pi / wavelength``.
        pupil: A function that applies a pupil to the field if provided.
            Defaults to ``None`` in which case the field is unchanged.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    dx: ScalarLike
    spectrum: (
        Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]]
    )
    power: ScalarLike
    amplitude: ScalarLike
    kykx: Float[Array, "2"] | tuple[float, float]
    pupil: FieldPupil | None = eqx.field(static=True)
    scalar: bool = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int],
        dx: ScalarLike,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        power: ScalarLike | None = 1.0,
        amplitude: ScalarLike | Float[Array, "3"] = 1.0,
        kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
        pupil: FieldPupil | None = None,
        scalar: bool = True,
    ):
        """
        Generates plane wave of given ``power``, as ``exp(1j)`` at each location
        of the field. Can also be given ``pupil`` and ``kykx`` vector to control
        the angle of the plane wave. If a ``kykx`` wave vector is provided,
        the plane wave is constructed as ``exp(1j * jnp.sum(kykx * field.grid,
        axis=-1))``.

        Args:
            shape: The shape (height and width) of the ``Field`` to be created
                in number of samples (pixels).
            dx: The spacing (pixel size) of the samples of the ``Field`` in
                units of distance. In the simplest case, you can pass a scalar
                value to define this spacing which creates a square pixel. Each
                sample (pixel) of the ``Field`` has a height and width. Spacing
                is also allowed to vary with wavelength of the spectrum. To
                choose a different spacing per wavelength (but still square),
                you can pass a 1D array of spacings of the same length as the
                number of wavelengths in the spectrum of the ``Field``. To
                create a non-square spacing, you must always pass a 2D array
                of shape `(wavelengths 2)` where the last dimension has length
                2 and defines the height and width of the spacing in units
                of distance. To createa a non-square spacing you must always
                include the `wavelengths` dimension even if there is only a
                single wavelength in the spectrum (i.e. a `(1 2)` shaped array).
            spectrum: The [``Spectrum``](chromatix.core.spectrum.Spectrum)
                of the ``Field`` to be created. This can be specified either
                as a single float value representing a wavelength in units of
                distance for a monochromatic field, a 1D array of wavelengths
                for a chromatic field that has the same intensity in all
                wavelengths, or a tuple of two 1D arrays where the first array
                represents the wavelengths and the second array is a unitless
                array of weights that define the spectral density (the relative
                intensity of each wavelength in the spectrum). This second array
                of spectral density will automatically be normalized to sum
                to 1.
            power: The total power that the result should be normalized to,
                defaults to 1.0. If ``None``, no normalization occurs.
            amplitude: The amplitude of the electric field. For scalar
                ``Field``s this doesnt do anything but scale the field (which
                will be undone if ``power`` is not ``None``), but it is required
                for vectorial ``Field``s to set the polarization.
            kykx: Defines the orientation of the plane wave. Should be a tuple or
                an array of shape `(2,)` in the format `[ky kx]`. We assume
                that these are wave vectors, i.e. that they have already been
                multiplied by ``2 * pi / wavelength``.
            pupil: A function that applies a pupil to the field if provided.
                Defaults to ``None`` in which case the field is unchanged.
            scalar: Whether the result should be ``ScalarField`` (if True) or
                ``VectorField`` (if False). Defaults to True.
        """
        self.shape = shape
        self.dx = dx
        self.spectrum = spectrum
        self.power = power
        self.amplitude = amplitude
        self.kykx = kykx
        self.pupil = pupil
        self.scalar = scalar

    def __call__(self) -> ScalarField | VectorField:
        return plane_wave(
            self.shape,
            self.dx,
            self.spectrum,
            self.power,
            self.amplitude,
            self.kykx,
            self.pupil,
            self.scalar,
        )


class GenericField(eqx.Module):
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.
    You can likely use the appropriate constructor for the type
    of ``Field`` you want rather than this function, or just use
    [``Field.build``](chromatix.core.field.Field).

    Attributes:
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
        amplitude: The amplitude of the field with shape `(... height
            width)` for scalar monochromatic fields, shape `(... height width
            wavelengths)` for scalar fields with multiple wavelengths, shape
            `(height width 3)` for vectorial monochromatic fields, or shape
            `(... height width wavelengths 3)` for vectorial fields with
            multiple wavelengths.
        phase: The phase of the field with shape `(... height width)` for scalar
            monochromatic fields, shape `(... height width wavelengths)` for
            scalar fields with multiple wavelengths, shape `(height width 3)`
            for vectorial monochromatic fields, or shape `(... height width
            wavelengths 3)` for vectorial fields with multiple wavelengths.
        power: The total power (a scalar value) that the result should be
            normalized to, defaults to 1.0. If ``None``, no normalization
            occurs.
        pupil: A function that applies a pupil to the field if provided.
            Defaults to ``None`` in which case the field is unchanged.
        scalar: Whether the result should be ``ScalarField`` (if ``True``) or
            ``VectorField`` (if ``False``). Defaults to ``True``.
    """

    dx: ScalarLike
    spectrum: (
        Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]]
    )
    amplitude: ArrayLike
    phase: ArrayLike
    power: ScalarLike
    pupil: FieldPupil | None = eqx.field(static=True)
    scalar: bool = eqx.field(static=True)

    def __init(
        self,
        dx: ScalarLike,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        amplitude: ArrayLike,
        phase: ArrayLike,
        power: ScalarLike | None = 1.0,
        pupil: FieldPupil | None = None,
        scalar: bool = True,
    ):
        """
        Generates field with arbitrary ``phase`` and ``amplitude``.
        You can likely use the appropriate constructor for the type
        of ``Field`` you want rather than this function, or just use
        [``Field.build``](chromatix.core.field.Field).

        Args:
            dx: The spacing (pixel size) of the samples of the ``Field`` in
                units of distance. In the simplest case, you can pass a scalar
                value to define this spacing which creates a square pixel. Each
                sample (pixel) of the ``Field`` has a height and width. Spacing
                is also allowed to vary with wavelength of the spectrum. To
                choose a different spacing per wavelength (but still square),
                you can pass a 1D array of spacings of the same length as the
                number of wavelengths in the spectrum of the ``Field``. To
                create a non-square spacing, you must always pass a 2D array
                of shape `(wavelengths 2)` where the last dimension has length
                2 and defines the height and width of the spacing in units
                of distance. To createa a non-square spacing you must always
                include the `wavelengths` dimension even if there is only a
                single wavelength in the spectrum (i.e. a `(1 2)` shaped array).
            spectrum: The [``Spectrum``](chromatix.core.spectrum.Spectrum)
                of the ``Field`` to be created. This can be specified either
                as a single float value representing a wavelength in units of
                distance for a monochromatic field, a 1D array of wavelengths
                for a chromatic field that has the same intensity in all
                wavelengths, or a tuple of two 1D arrays where the first array
                represents the wavelengths and the second array is a unitless
                array of weights that define the spectral density (the relative
                intensity of each wavelength in the spectrum). This second array
                of spectral density will automatically be normalized to sum
                to 1.
            amplitude: The amplitude of the field with shape `(... height
                width)` for scalar monochromatic fields, shape `(... height
                width wavelengths)` for scalar fields with multiple wavelengths,
                shape `(height width 3)` for vectorial monochromatic fields, or
                shape `(... height width wavelengths 3)` for vectorial fields
                with multiple wavelengths.
            phase: The phase of the field with shape `(... height width)`
                for scalar monochromatic fields, shape `(... height width
                wavelengths)` for scalar fields with multiple wavelengths, shape
                `(height width 3)` for vectorial monochromatic fields, or shape
                `(... height width wavelengths 3)` for vectorial fields with
                multiple wavelengths.
            power: The total power (a scalar value) that the result should be
                normalized to, defaults to 1.0. If ``None``, no normalization
                occurs.
            pupil: A function that applies a pupil to the field if provided.
                Defaults to ``None`` in which case the field is unchanged.
            scalar: Whether the result should be ``ScalarField`` (if ``True``)
                or ``VectorField`` (if ``False``). Defaults to ``True``.
        """
        self.dx = dx
        self.spectrum = spectrum
        self.amplitude = amplitude
        self.phase = phase
        self.power = power
        self.pupil = pupil
        self.scalar = scalar

    def __call__(self) -> ScalarField | VectorField:
        return generic_field(
            self.dx,
            self.spectrum,
            self.amplitude,
            self.phase,
            self.power,
            self.pupil,
            self.scalar,
        )

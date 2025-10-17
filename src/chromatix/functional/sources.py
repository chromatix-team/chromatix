from typing import Callable

import jax.numpy as jnp
import numpy as np
from chex import assert_axis_dimension, assert_equal_shape
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix import Field, MonoSpectrum, ScalarField, Spectrum, VectorField
from chromatix.functional.pupils import circular_pupil, gaussian_pupil
from chromatix.typing import wv, z
from chromatix.utils import l2_sq_norm
from chromatix.utils.shapes import (
    _broadcast_1d_to_innermost_batch,
)

__all__ = [
    "point_source",
    "objective_point_source",
    "plane_wave",
    "gaussian_plane_wave",
    "generic_field",
]


# We need this alias for typing to pass
FieldPupil = Callable[[Field], Field]


def point_source(
    shape: tuple[int, int],
    dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
    spectrum: Spectrum
    | ScalarLike
    | Float[Array, "wv"]
    | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    z: ScalarLike | Float[Array, "z"],
    n: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike | Float[Array, "3"] = 1.0,
    offset: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
    scalar: bool = True,
    epsilon: float = float(np.finfo(np.float32).eps),
) -> ScalarField | VectorField:
    """
    Generates field due to point source a distance ``z`` away. Can also be given
    ``pupil``.

    !!! warning
        This function is numerically unstable at z = 0, so an epsilon is applied.

    Args:
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
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
            field that has the same intensity in all wavelengths, or a tuple
            of two 1D arrays where the first array represents the wavelengths
            and the second array is a unitless array of weights that define the
            spectral density (the relative intensity of each wavelength in the
            spectrum). This second array of spectral density will automatically
            be normalized to sum to 1.
        z: How far away the point source is in units of distance.
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
    spectrum = Spectrum.build(spectrum)
    field = Field.empty(shape, dx, spectrum, scalar)
    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)
    z = jnp.atleast_1d(jnp.asarray(z)).squeeze()
    if z.size > 1:
        z = _broadcast_1d_to_innermost_batch(z, field.spatial_dims)
        field = field.expand_dims()
    offset = jnp.atleast_1d(jnp.asarray(offset))
    L = jnp.sqrt(field.broadcasted_wavelength * jnp.abs(z) / n)
    L_sq = jnp.sign(z) * jnp.fmax(L**2, epsilon)
    phase = jnp.pi * l2_sq_norm(field.grid - offset) / L_sq
    u = amplitude * -1j / L_sq * jnp.exp(1j * phase)
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def objective_point_source(
    shape: tuple[int, int],
    dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
    spectrum: Spectrum
    | ScalarLike
    | Float[Array, "wv"]
    | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    z: ScalarLike | Float[Array, "z"],
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike | Float[Array, "3"] = 1.0,
    offset: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
    scalar: bool = True,
) -> ScalarField | VectorField:
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a thin lens with focal length
    ``f`` and numerical aperture ``NA``.

    Args:
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
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
            field that has the same intensity in all wavelengths, or a tuple
            of two 1D arrays where the first array represents the wavelengths
            and the second array is a unitless array of weights that define the
            spectral density (the relative intensity of each wavelength in the
            spectrum). This second array of spectral density will automatically
            be normalized to sum to 1.
        z: How far away the point source is from the focal plane of the lens in
            units of distance.
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
    spectrum = Spectrum.build(spectrum)
    field = Field.empty(shape, dx, spectrum, scalar)
    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(jnp.asarray(amplitude))
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)
    z = jnp.atleast_1d(jnp.asarray(z)).squeeze()
    if z.size > 1:
        z = _broadcast_1d_to_innermost_batch(z, field.spatial_dims)
        field = field.expand_dims()
    offset = jnp.atleast_1d(jnp.asarray(offset))
    L = jnp.sqrt(field.broadcasted_wavelength * f / n)
    phase = -jnp.pi * (z / f) * l2_sq_norm(field.grid - offset) / L**2
    u = amplitude * -1j / L**2 * jnp.exp(1j * phase)
    field = field.replace(u=u)
    D = 2 * f * NA / n
    field = circular_pupil(field, D)  # type: ignore
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def plane_wave(
    shape: tuple[int, int],
    dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
    spectrum: Spectrum
    | ScalarLike
    | Float[Array, "wv"]
    | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike | Float[Array, "3"] = 1.0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> ScalarField | VectorField:
    """
    Generates plane wave of given ``power``, as ``exp(1j)`` at each location of
    the field. Can also be given ``pupil`` and ``kykx`` vector to control the
    angle of the plane wave. If a ``kykx`` wave vector is provided, the plane
    wave is constructed as ``exp(1j * jnp.sum(kykx * field.grid, axis=-1))``.

    Args:
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
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
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
    spectrum = Spectrum.build(spectrum)
    field = Field.empty(shape, dx, spectrum, scalar)
    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)
    kykx = jnp.atleast_1d(jnp.asarray(kykx))
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=-1))
    # NOTE(dd/2025-10-02): There's no vectorial dependence on the grid so we
    # need to make sure to match the right shape, hence the multiplication
    # by the ones_like term to ensure we broadcast properly in the case of a
    # vectorial Field.
    field = field.replace(u=u * jnp.ones_like(field.u))
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def gaussian_plane_wave(
    shape: tuple[int, int],
    dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
    spectrum: Spectrum
    | ScalarLike
    | Float[Array, "wv"]
    | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    waist: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike | Float[Array, "3"] = 1.0,
    kykx: ArrayLike | tuple[int, int] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> Field:
    """
    Generates plane wave of given ``power`` with a Gaussian intensity profile
    (as opposed to a totally flat plane wave). Can also be given ``pupil`` and
    ``kykx`` vector to control the angle of the plane wave.

    Args:
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
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
            field that has the same intensity in all wavelengths, or a tuple
            of two 1D arrays where the first array represents the wavelengths
            and the second array is a unitless array of weights that define the
            spectral density (the relative intensity of each wavelength in the
            spectrum). This second array of spectral density will automatically
            be normalized to sum to 1.
        waist: The size of the waist (twice the standard deviation) as a
            scalar in units of distance. This waist defines the width of the 2D
            Gaussian amplitude profile of this beam.
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
    spectrum = Spectrum.build(spectrum)
    field = Field.empty(shape, dx, spectrum, scalar)
    amplitude = jnp.atleast_1d(jnp.asarray(amplitude))
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)
    kykx = jnp.atleast_1d(jnp.asarray(kykx))
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=-1))
    # There's no spectral dependence so we need to manually put in the spectral axis
    # hence the ones_like term.
    field = field.replace(u=u * jnp.ones_like(field.u))
    field = gaussian_pupil(field, waist)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def generic_field(
    dx: ScalarLike | Float[Array, "wv"] | Float[Array, "wv 2"],
    spectrum: Spectrum
    | ScalarLike
    | Float[Array, "wv"]
    | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    amplitude: ArrayLike,
    phase: ArrayLike,
    power: ScalarLike | None = 1.0,
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> Field:
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.
    You can likely use the appropriate constructor for the type
    of ``Field`` you want rather than this function, or just use
    [``Field.build``](field.md#chromatix.core.field.Field.build).

    Args:
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
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
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
    spectrum = Spectrum.build(spectrum)
    assert_equal_shape([amplitude, phase])
    match (scalar, isinstance(spectrum, MonoSpectrum)):
        case (False, False):
            assert amplitude.ndim >= 4, (
                "Amplitude must have at least 4 dimensions: (height width wavelengths 3)"
            )
            assert phase.ndim >= 4, (
                "Phase must have at least 4 dimensions: (height width wavelengths 3)"
            )
            assert_axis_dimension(amplitude, -1, 3)
            assert_axis_dimension(phase, -1, 3)
        case (False, True):
            assert amplitude.ndim >= 3, (
                "Amplitude must have at least 3 dimensions: (height width 3)"
            )
            assert phase.ndim >= 3, (
                "Phase must have at least 3 dimensions: (height width 3)"
            )
            assert_axis_dimension(amplitude, -1, 3)
            assert_axis_dimension(phase, -1, 3)
        case (True, False):
            assert amplitude.ndim >= 3, (
                "Amplitude must have at least 3 dimensions: (height width wavelengths)"
            )
            assert phase.ndim >= 3, (
                "Phase must have at least 3 dimensions: (height width wavelengths)"
            )
        case (True, True):
            assert amplitude.ndim >= 2, (
                "Amplitude must have at least 2 dimensions: (height width)"
            )
            assert phase.ndim >= 2, (
                "Phase must have at least 2 dimensions: (height width)"
            )
    u = jnp.asarray(amplitude) * jnp.exp(1j * jnp.asarray(phase))
    field = Field.build(u, dx, spectrum)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field

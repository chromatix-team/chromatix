import jax.numpy as jnp
from ..field import Field, ScalarField, VectorField
from typing import Optional, Callable, Union, Tuple
from chex import Array, assert_equal_shape, assert_axis_dimension
from .pupils import circular_pupil
from ..utils import l2_sq_norm
from ..utils.shapes import (
    _broadcast_1d_to_innermost_batch,
    _broadcast_1d_to_grid,
    _broadcast_1d_to_polarization,
)

__all__ = [
    "point_source",
    "objective_point_source",
    "plane_wave",
    "generic_field",
]


def point_source(
    shape: Tuple[int, int],
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    spectral_density: Union[float, Array],
    z: float,
    n: float,
    power: float = 1.0,
    amplitude: Union[float, Array] = 1.0,
    pupil: Optional[Callable[[ScalarField], ScalarField]] = None,
    scalar: bool = True,
) -> Field:
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        n: Refractive index.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    L = jnp.sqrt(field.spectrum * z / n)
    phase = jnp.pi * l2_sq_norm(field.grid) / L**2
    u = amplitude * -1j / L**2 * jnp.exp(1j * phase)
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)


def objective_point_source(
    shape: Tuple[int, int],
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    spectral_density: Union[float, Array],
    z: float,
    f: float,
    n: float,
    NA: float,
    power: float = 1.0,
    amplitude: Union[float, Array] = 1.0,
    scalar: bool = True,
) -> Field:
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a lens with focal length ``f``
    and numerical aperture ``NA``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        f: Focal length of the objective lens.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * (z / f) * l2_sq_norm(field.grid) / L**2
    u = amplitude * -1j / L**2 * jnp.exp(1j * phase)
    field = field.replace(u=u)
    D = 2 * f * NA / n
    field = circular_pupil(field, D)
    return field * jnp.sqrt(power / field.power)


def plane_wave(
    shape: Tuple[int, int],
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    spectral_density: Union[float, Array],
    power: float = 1.0,
    amplitude: Union[float, Array] = 1.0,
    kykx: Union[Array, Tuple[float, float]] = (0.0, 0.0),
    pupil: Optional[Callable[[Field], Field]] = None,
    scalar: bool = True,
) -> Field:
    """
    Generates plane wave of given ``power``.

    Can also be given ``pupil`` and ``kykx`` vector to control the angle of the
    plane wave.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to,
            defaults to 1.0
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        kykx: Defines the orientation of the plane wave. Should be an
            array of shape `[2,]` in the format [ky, kx].
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=0))
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)


def generic_field(
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    spectral_density: Union[float, Array],
    amplitude: Array,
    phase: Array,
    power: Optional[float] = 1.0,
    pupil: Optional[Callable[[ScalarField], ScalarField]] = None,
    scalar: bool = True,
) -> Field:
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    Args:
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        amplitude: The amplitude of the field with shape `(B... H W C [1 | 3])`.
        phase: The phase of the field with shape `(B... H W C [1 | 3])`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    assert (
        amplitude.ndim >= 5
    ), "Amplitude must have at least 5 dimensions: (B... H W C [1 | 3])"
    assert (
        phase.ndim >= 5
    ), "Phase must have at least 5 dimensions: (B... H W C [1 | 3])"
    vectorial_dimension = 1 if scalar else 3
    assert_axis_dimension(amplitude, -1, vectorial_dimension)
    assert_axis_dimension(phase, -1, vectorial_dimension)
    assert_equal_shape([amplitude, phase])
    u = amplitude * jnp.exp(1j * phase)
    field = create(dx, spectrum, spectral_density, u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)

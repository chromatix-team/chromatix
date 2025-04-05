from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from chex import Array, assert_axis_dimension, assert_equal_shape

from chromatix.field import Field, ScalarField, VectorField
from chromatix.utils import l2_sq_norm
from chromatix.utils.shapes import (
    _broadcast_1d_to_grid,
    _broadcast_1d_to_innermost_batch,
    _broadcast_1d_to_polarization,
)

from .pupils import circular_pupil

__all__ = [
    "point_source",
    "gaussian_source",
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
    epsilon: float = np.finfo(np.float32).eps,
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
        epsilon: Value added to denominators for numerical stability.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))
    phase = jnp.pi * l2_sq_norm(field.grid) / (L**2 + epsilon)
    u = amplitude * -1j / (L**2 + epsilon) * jnp.exp(1j * phase)
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)


def gaussian_source(
    shape: Tuple[int, int],
    spectrum: Union[float, Array],
    spectral_density: Union[float, Array],
    z: float,
    f: float,
    n: float,
    NA: float,
    power: float = 1.0,
    amplitude: Union[float, Array] = np.array([0.0, 0.0, 1.0]),
    offset: Union[Array, Tuple[float, float]] = (0.0, 0.0),
    scalar: bool = True,
    envelope_waist: float = 1.0,
) -> Field:
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a lens with focal length ``f``
    and numerical aperture ``NA``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
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
        offset: The offset of the point source in the plane. Should be an array
            of shape `[2,]` in the format `[y, x]`.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    D = 2
    fourier_spacing = D / shape[0]
    field = create(fourier_spacing, spectrum, spectral_density, shape=shape)

    mask = field.grid[0] ** 2 + field.grid[1] ** 2 <= 1
    factor = NA / n
    sin_theta2 = factor**2 * jnp.sum(field.grid**2, axis=0) * mask
    cos_theta = jnp.sqrt(1 - sin_theta2)
    sin_theta = jnp.sqrt(sin_theta2)

    phi = jnp.arctan2(field.grid[0], field.grid[1])
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    single_field = field.grid[0] ** 2 + field.grid[1] ** 2 <= 1
    field_x = jnp.complex64(amplitude[2] * single_field)
    field_y = jnp.complex64(amplitude[1] * single_field)

    e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x + (
        cos_theta - 1.0
    ) * sin_2phi * field_y
    e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y + (
        cos_theta - 1.0
    ) * sin_2phi * field_x
    e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

    amplitude = jnp.stack([e_inf_z, e_inf_y, e_inf_x], axis=-1).squeeze(-2) / 2

    z = _broadcast_1d_to_innermost_batch(z, field.ndim)

    offset = _broadcast_1d_to_grid(offset, field.ndim)
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * (z / f) * l2_sq_norm(factor * field.grid - offset) / L**2
    gaussian_envelope = jnp.exp(
        -l2_sq_norm(factor * field.grid - offset) * factor**2 / envelope_waist**2
    )
    u = gaussian_envelope * amplitude * -1j / L**2 * jnp.exp(1j * phase)
    u = jnp.broadcast_to(u, field.shape)
    field = field.replace(u=u)
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
    offset: Union[Array, Tuple[float, float]] = (0.0, 0.0),
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
        offset: The offset of the point source in the plane. Should be an array
            of shape `[2,]` in the format `[y, x]`.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    offset = _broadcast_1d_to_grid(offset, field.ndim)
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * (z / f) * l2_sq_norm(field.grid - offset) / L**2
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
            array of shape `[2,]` in the format `[ky, kx]`.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    field = create(dx, spectrum, spectral_density, shape=shape)
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=0))
    # There's no spectral dependence so we need to manually put in the spectral axis
    # hence the ones_like term.
    field = field.replace(u=u * jnp.ones_like(field.u))

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
    assert amplitude.ndim >= 5, (
        "Amplitude must have at least 5 dimensions: (B... H W C [1 | 3])"
    )
    assert phase.ndim >= 5, (
        "Phase must have at least 5 dimensions: (B... H W C [1 | 3])"
    )
    vectorial_dimension = 1 if scalar else 3
    assert_axis_dimension(amplitude, -1, vectorial_dimension)
    assert_axis_dimension(phase, -1, vectorial_dimension)
    assert_equal_shape([amplitude, phase])
    u = amplitude * jnp.exp(1j * phase)
    field = create(dx, spectrum, spectral_density, u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)

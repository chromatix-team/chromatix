from typing import Callable

import jax.numpy as jnp
import numpy as np
from chex import assert_axis_dimension, assert_equal_shape

from chromatix import Field, ScalarField, VectorField
from chromatix.typing import ArrayLike, ScalarLike
from chromatix.utils import l2_sq_norm
from chromatix.utils.shapes import (
    _broadcast_1d_to_grid,
    _broadcast_1d_to_innermost_batch,
    _broadcast_1d_to_polarization,
)

from .pupils import circular_pupil

__all__ = [
    "point_source",
    "objective_point_source",
    "plane_wave",
    "generic_field",
]


# We need this alias for typing to pass
FieldPupil = Callable[[Field], Field]


def point_source(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    z: ScalarLike,
    n: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike = 1.0,
    pupil: FieldPupil | None = None,
    scalar: bool = True,
    epsilon: float = float(np.finfo(np.float32).eps),
) -> ScalarField | VectorField:
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
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
        epsilon: Value added to denominators for numerical stability.
    """
    create = ScalarField.create if scalar else VectorField.create
    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)

    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    L = jnp.sqrt(field.spectrum * jnp.abs(z) / n)
    L_sq = jnp.sign(z) * jnp.fmax(L**2, epsilon)
    phase = jnp.pi * l2_sq_norm(field.grid) / L_sq
    u = amplitude * -1j / L_sq * jnp.exp(1j * phase)
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def objective_point_source(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    z: ScalarLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike = 1.0,
    offset: ArrayLike | tuple[float, float] = (0.0, 0.0),
    scalar: bool = True,
) -> ScalarField | VectorField:
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
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        offset: The offset of the point source in the plane. Should be an array
            of shape `[2,]` in the format `[y, x]`.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create

    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)

    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    offset = _broadcast_1d_to_grid(offset, field.ndim)
    L = jnp.sqrt(field.spectrum * f / n)
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
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    power: ScalarLike | None = 1.0,
    amplitude: ScalarLike = 1.0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> ScalarField | VectorField:
    """
    Generates plane wave of given ``power``, as ``exp(1j)`` at each location of
    the field.

    Can also be given ``pupil`` and ``kykx`` vector to control the angle of
    the plane wave. If a ``kykx`` wave vector is provided, the plane wave is
    constructed as ``exp(1j * jnp.sum(kykx * field.grid, axis=0))``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        kykx: Defines the orientation of the plane wave. Should be an array of
            shape `[2,]` in the format `[ky, kx]`. We assume that these are wave
            vectors, i.e. that they have already been multiplied by ``2 * pi
            / wavelength``.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create

    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)

    field = create(dx, spectrum, spectral_density, shape=shape)
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=0))
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field


def generic_field(
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    amplitude: ArrayLike,
    phase: ArrayLike,
    power: ScalarLike | None = 1.0,
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> ScalarField | VectorField:
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
        power: The total power that the result should be normalized to, defaults
            to 1.0. If ``None``, no normalization occurs.
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
    u = jnp.array(amplitude) * jnp.exp(1j * phase)
    field = create(dx, spectrum, spectral_density, u=u)
    if pupil is not None:
        field = pupil(field)
    if power is not None:
        field = field * jnp.sqrt(power / field.power)
    return field

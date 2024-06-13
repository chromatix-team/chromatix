from typing import Callable

import jax.numpy as jnp
import numpy as np
from chex import assert_equal_shape
from einops import rearrange

from chromatix import Field, ScalarField, VectorField
from chromatix.typing import ArrayLike, ScalarLike

from ..utils import l2_sq_norm
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
    z: ScalarLike | ArrayLike,
    n: ScalarLike,
    power: ScalarLike = 1.0,
    amplitude: ScalarLike = 1.0,
    pupil: FieldPupil | None = None,
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
    # Parsing inputs
    amplitude = jnp.atleast_1d(amplitude)
    power = jnp.atleast_1d(power)
    z = jnp.atleast_1d(z)
    z = rearrange(z, "b -> b 1 1 1")

    # Create empty field
    match amplitude.shape[-1]:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise NotImplementedError

    L_sq = (field.spectrum * z / n) + epsilon  # epsilon to deal with z=0
    phase = jnp.pi * l2_sq_norm(field.grid()) / L_sq
    field = field.replace(u=amplitude * -1j / L_sq * jnp.exp(1j * phase))
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power())


def objective_point_source(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    z: ScalarLike | ArrayLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike,
    power: ScalarLike = 1.0,
    amplitude: ScalarLike = 1.0,
    offset: ArrayLike | tuple[float, float] = (0.0, 0.0),
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

    # Parsing inputs
    amplitude = jnp.atleast_1d(amplitude)
    power = jnp.array(power)
    z = jnp.atleast_1d(z)
    z = rearrange(z, "b -> b 1 1 1 1")

    # Create empty field
    match amplitude.shape[-1]:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise NotImplementedError

    # Making field
    L_sq = field.spectrum * f / n
    phase = -jnp.pi * (z / f) * l2_sq_norm(field.grid() - offset) / L_sq
    field = field.replace(u=amplitude * -1j / L_sq * jnp.exp(1j * phase))

    # Making pupil
    field = circular_pupil(field, w=2 * f * NA / n)  # type: ignore
    return field * jnp.sqrt(power / field.power())


def plane_wave(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    power: ScalarLike = 1.0,
    amplitude: ScalarLike | ArrayLike = 1.0,
    kxky: ArrayLike | tuple[float, float] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
) -> ScalarField | VectorField:
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
    # Parsing inputs - k needs to be a column vector
    kxky = jnp.array(kxky)[:, None]
    amplitude = jnp.atleast_1d(amplitude)
    power = jnp.atleast_1d(power)

    # Create empty field
    match amplitude.shape[-1]:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise NotImplementedError

    # Add in field
    field = field.replace(u=amplitude * jnp.exp(1j * jnp.dot(field.grid(), kxky)))

    # Apply pupil and set power
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power())


def generic_field(
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    amplitude: ArrayLike,
    phase: ArrayLike,
    power: ScalarLike = 1.0,
    pupil: FieldPupil | None = None,
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
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    # Parsing inputs
    power = jnp.atleast_1d(power)
    amplitude = jnp.array(amplitude)
    phase = jnp.array(phase)
    assert_equal_shape([amplitude, phase])

    match amplitude.ndim:
        case 2:
            amplitude = rearrange(amplitude, "h w -> 1 h w 1 1")
            phase = rearrange(phase, "h w -> 1 h w 1 1")
        case 5:
            pass
        case _:
            raise NotImplementedError

    # Make field
    u = amplitude * jnp.exp(1j * phase)

    # Create empty field
    match amplitude.shape[-1]:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, u=u)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, u=u)
        case _:
            raise NotImplementedError

    # Add pupil
    if pupil is not None:
        field = pupil(field)

    return field * jnp.sqrt(power / field.power())

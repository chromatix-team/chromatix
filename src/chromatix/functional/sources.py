import jax.numpy as jnp
from einops import rearrange
from ..field import Field
from typing import Optional, Callable, Tuple
from chex import Array, assert_rank
from .pupils import circular_pupil
import jax

__all__ = [
    "empty_field",
    "point_source",
    "objective_point_source",
    "plane_wave",
    "generic_field",
]


def empty_field(
    shape: Tuple[int, int], dx: float, spectrum: float, spectral_density: float
) -> Field:
    """Simple wrapper to create empty field."""
    return Field.create(dx, spectrum, spectral_density, shape=shape)


def point_source(
    field: Field,
    z: float,
    n: float,
    power: float = 1.0,
    pupil: Optional[Callable[[Field], Field]] = None,
) -> Field:
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    Args:
        field: The ``Field`` which will be filled with the result of the point
            source (should be empty).
        z: The distance of the point source.
        n: Refractive index.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
    """
    z = rearrange(jnp.atleast_1d(z), "d -> d 1 1 1")

    # Calculating phase and pupil
    L = jnp.sqrt(field.spectrum * z / n)
    phase = jnp.pi * field.l2_sq_grid / L**2
    u = -1j / L**2 * jnp.exp(1j * phase)
    field = field.replace(u=u)

    # Applying pupil
    if pupil is not None:
        field = pupil(field)

    # Normalizing to given power
    return field * jnp.sqrt(power / field.power)


def objective_point_source(
    field: Field, z: float, f: float, n: float, NA: float, power: float = 1.0
) -> Field:
    """
    Generates field due to a point source defocused by an amount ``z`` away
    from the focal plane, just after passing through a lens with focal length
    ``f`` and numerical aperture ``NA``.

    Args:
        field: The ``Field`` which will be filled with the result of the point
            source after an objective lens (should be empty).
        z: The distance of the point source.
        f: Focal length of the objective lens.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
    """
    z = rearrange(jnp.atleast_1d(z), "d -> d 1 1 1")

    # Calculating phase and pupil
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * (z / f) * field.l2_sq_grid / L**2

    # Field
    u = -1j / L**2 * jnp.exp(1j * phase)
    field = field.replace(u=u)

    D = 2 * f * NA / n  # Expression for NA yields width of pupil
    field = circular_pupil(field, D)

    # Normalizing to given power
    return field * jnp.sqrt(power / field.power)


def plane_wave(
    field: Field,
    power: float = 1.0,
    kykx: Array = jnp.zeros(2),
    pupil: Optional[Callable[[Field], Field]] = None,
) -> Field:
    """
    Generates plane wave of given ``power``.

    Can also be given ``pupil`` and ``k`` vector.

    Args:
        field: The ``Field`` which will be filled with the result of the plane
            wave (should be empty).
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        kykx: Defines the orientation of the plane wave. Should be an
            array of shape `[2,]` in the format [ky, kx].
        pupil: If provided, will be called on the field to apply a pupil.
    """
    u = jnp.exp(1j * (jnp.einsum("v, vbhwc->bhwc", kykx, field.grid)))

    field = field.replace(u=u)

    # Applying pupil
    if pupil is not None:
        field = pupil(field)

    # Setting to correct power
    return field * jnp.sqrt(power / field.power)


def generic_field(
    field: Field,
    amplitude: Array,
    phase: Array,
    power: Optional[float] = 1.0,
    pupil: Optional[Callable[[Field], Field]] = None,
) -> Field:
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    Args:
        field: The ``Field`` which will be filled with the result of the
            arbitrary phase perturbation (should be empty).
        amplitude: The amplitude of the field with shape `[B H W C]`.
        phase: The phase of the field with shape `[B H W C]`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
    """
    assert_rank(
        amplitude, 4, custom_message="Amplitude must be array of shape [B, H, W, C]"
    )
    assert_rank(phase, 4, custom_message="Phase must be array of shape [B, H, W, C]")
    field = field.replace(u=amplitude * jnp.exp(1j * phase))

    if pupil is not None:
        field = pupil(field)
    # Setting to correct power
    return field * jnp.sqrt(power / field.power)

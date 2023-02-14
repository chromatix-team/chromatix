import jax.numpy as jnp
from einops import rearrange
from ..field import Field
from typing import Optional, Callable
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
    shape: tuple[int, int], dx: float, spectrum: float, spectral_density: float
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
    """Generates field due to point source a distance z from it. Can take pupil."""
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
    Replace field value with the result of a point source defocused by
    an amount z away from the focal plane, just after passing through
    a lens with focus f and numerical aperture NA.
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
    phase: float = 0.0,
    pupil: Optional[Callable[[Field], Field]] = None,
    k: Optional[Array] = None,
) -> Field:
    """Generates plane wave of given phase and power.
    Can also be given pupil and k vector."""

    # Field values
    if k is None:
        u = jnp.exp(1j * jnp.full(field.shape, phase))
    else:
        u = jnp.exp(1j * 2 * jnp.pi * jnp.dot(k[::-1], jnp.moveaxis(field.grid, 0, -2)))

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
):
    assert_rank(
        amplitude, 4, custom_message="Amplitude must be array of shape [B, H, W, C]"
    )

    assert_rank(phase, 4, custom_message="Phase must be array of shape [B, H, W, C]")
    field = field.replace(u=amplitude * jnp.exp(1j * phase))

    if pupil is not None:
        field = pupil(field)
    # Setting to correct power
    return field * jnp.sqrt(power / field.power)

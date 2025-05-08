from typing import Union

import equinox as eqx
import jax.numpy as jnp
from chromatix import Field, ScalarField
from einops import rearrange
from jax import Array


class Probe(eqx.Module):
    amplitude: Array
    phase: Array

    def __init__(
        self,
        *,
        amplitude: Array | None = None,
        phase: Array | None = None,
        field: Array | None = None,
    ):
        """Either give in phase and amplitude, or complex field"""
        match (amplitude, phase, field):
            case (None, None, _):
                field = rearrange(field, "h w -> 1 h w 1 1")
                self.amplitude = jnp.abs(field)
                self.phase = jnp.angle(field)

            case (_, _, None):
                self.amplitude = rearrange(amplitude, "h w -> 1 h w 1 1")
                self.phase = rearrange(phase, "h w -> 1 h w 1 1")

            case _:
                raise NotImplementedError()


class FlatProbe(eqx.Module):
    """Simple class to hold probe."""

    amplitude: Array
    phase: Array

    def __init__(self, shape: tuple[int, int], amplitude: float = 1.0):
        self.amplitude = jnp.full((1, *shape, 1, 1), amplitude)
        self.phase = jnp.zeros_like(self.amplitude)


# NOTE: this version doesn't normalise power; should be upstreamed to chromatix.
def generic_field(
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    spectral_density: Array,
    amplitude: Array,
    phase: Array,
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
    u = amplitude * jnp.exp(1j * phase)
    field = ScalarField.create(dx, spectrum, spectral_density, u=u)
    return field

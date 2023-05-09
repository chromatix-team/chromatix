import jax.numpy as jnp

from ..field import Field
from chex import Array, assert_equal_rank
from typing import Sequence
from chromatix.utils.shapes import _broadcast_2d_to_spatial
from chromatix.utils.initialisers import zernike, seidel
from chromatix.utils import spectrally_modulate_phase

__all__ = ["phase_change", "wrap_phase"]


def phase_change(field: Field, phase: Array, spectrally_modulate: bool = True) -> Field:
    """
    Perturbs ``field`` by ``phase`` (given in radians).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        phase: The phase to apply.
        spectrally_modulate: sets spectral modulation of phase.
    """
    phase = _broadcast_2d_to_spatial(phase, field.ndim)
    assert_equal_rank((phase, field.u))
    if spectrally_modulate:
        phase = spectrally_modulate_phase(phase, field.spectrum)
    return field * jnp.exp(1j * phase)


def zernike_aberrations(
    field: Field, pupil_radius: float, coefficients: Array, ansi_indices: Sequence[int]
) -> Field:
    phase = zernike(field.grid, pupil_radius, coefficients, ansi_indices)
    return phase_change(field, phase)


def seidel_aberrations(
    field: Field,
    pupil_radius: float,
    coefficients: Array,
    u: float = 0,
    v: float = 0,
) -> Field:
    phase = seidel(field.grid, pupil_radius, field.spectrum, coefficients, u, v)
    return phase_change(field, phase)

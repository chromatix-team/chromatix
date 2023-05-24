import jax.numpy as jnp
from chex import Array, assert_rank
from typing import Tuple
from ..field import Field
from ..utils.shapes import _broadcast_2d_to_spatial

__all__ = ["phase_change", "wrap_phase", "spectrally_modulate_phase"]


def phase_change(field: Field, phase: Array, spectrally_modulate: bool = True) -> Field:
    """
    Perturbs ``field`` by ``phase`` (given in radians).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        phase: The phase to apply.
    """
    phase = _broadcast_2d_to_spatial(phase, field.ndim)
    assert_rank(
        phase,
        field.ndim,
        custom_message="Phase must have same ndim as incoming ``Field``.",
    )
    if spectrally_modulate:
        phase = spectrally_modulate_phase(phase, field)
    return field * jnp.exp(1j * phase)


def wrap_phase(phase: Array, limits: Tuple[float, float] = (-jnp.pi, jnp.pi)) -> Array:
    """
    Wraps values of ``phase`` to the range given by ``limits``.

    Args:
        phase: The phase mask to wrap (in radians).
        limits: A tuple defining the minimum and maximum value that ``phase``
            will be wrapped to.
    """
    phase_min, phase_max = limits
    assert phase_min < phase_max, "Lower limit needs to be smaller than upper limit."
    min_indices = phase < phase_min
    max_indices = phase > phase_max
    phase = phase.at[min_indices].set(
        phase[min_indices]
        + 2 * jnp.pi * (1 + (phase_min - phase[min_indices]) // (2 * jnp.pi))
    )
    phase = phase.at[max_indices].set(
        phase[max_indices]
        - 2 * jnp.pi * (1 + (phase[max_indices] - phase_max) // (2 * jnp.pi))
    )
    return phase


def spectrally_modulate_phase(phase: Array, field: Field) -> Array:
    """Spectrally modulates a given ``phase`` for multiple wavelengths."""
    central_wavelength = field.spectrum[..., 0, 0].squeeze()
    spectral_modulation = central_wavelength / field.spectrum
    return phase * spectral_modulation

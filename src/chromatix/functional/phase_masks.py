import jax
import jax.numpy as jnp
from chex import assert_rank
from jax import Array
from jax.scipy.ndimage import map_coordinates

from chromatix.field import Field, ScalarField, VectorField
from chromatix.ops.quantization import quantize
from chromatix.typing import ArrayLike, ScalarLike
from chromatix.utils.initializers import (
    axicon_phase,
    linear_phase,
    sawtooth_phase,
    sinusoid_phase,
)
from chromatix.utils.shapes import _broadcast_2d_to_spatial

__all__ = [
    "phase_change",
    "interpolated_phase_change",
    "wrap_phase",
    "spectrally_modulate_phase",
    "thin_prism",
    "sawtooth_grating",
    "sinusoid_grating",
    "axicon",
]


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


@jax.custom_jvp
def wrap_phase(
    phase: ArrayLike, limits: ArrayLike | tuple[float, float] = (-jnp.pi, jnp.pi)
) -> Array:
    """
    Wraps values of ``phase`` to the range given by ``limits``.

    Args:
        phase: The phase mask to wrap (in radians).
        limits: A tuple defining the minimum and maximum value that ``phase``
            will be wrapped to.
    """
    phase_min, phase_max = limits
    phase = jnp.where(
        phase < phase_min,
        phase + 2 * jnp.pi * (1 + (phase_min - phase) // (2 * jnp.pi)),
        phase,
    )
    phase = jnp.where(
        phase > phase_max,
        phase - 2 * jnp.pi * (1 + (phase - phase_max) // (2 * jnp.pi)),
        phase,
    )
    return phase


@wrap_phase.defjvp
def wrap_phase_jvp(primals: tuple, tangents: tuple) -> tuple:
    return wrap_phase(*primals), tangents[0]


def spectrally_modulate_phase(phase: Array, field: ScalarField | VectorField) -> Array:
    """Spectrally modulates a given ``phase`` for multiple wavelengths."""
    central_wavelength = field.spectrum[..., 0, 0].squeeze()
    spectral_modulation = central_wavelength / field.spectrum
    return phase * spectral_modulation


def interpolated_phase_change(
    field: Field,
    phase: Array,
    phase_range: ArrayLike | None = None,
    num_bits: int | None = None,
    interpolation_order: int = 0,
) -> Field:
    if phase_range is not None:
        phase = wrap_phase(phase, phase_range)
    if num_bits is not None:
        phase = quantize(phase, num_bits, range=phase_range)
    field_pixel_grid = jnp.meshgrid(
        jnp.linspace(0, phase.shape[0] - 1, num=field.spatial_shape[0]) + 0.5,
        jnp.linspace(0, phase.shape[1] - 1, num=field.spatial_shape[1]) + 0.5,
        indexing="ij",
    )
    phase = map_coordinates(phase, field_pixel_grid, interpolation_order)
    return phase_change(field, phase)


def thin_prism(
    field: Field,
    n_prism: ScalarLike,
    max_thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    phase = linear_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_prism,
        max_thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def sawtooth_grating(
    field: Field,
    n_grating: ScalarLike,
    period: ScalarLike,
    thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    phase = sawtooth_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_grating,
        period,
        thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def sinusoid_grating(
    field: Field,
    n_grating: ScalarLike,
    period: ScalarLike,
    thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    phase = sinusoid_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_grating,
        period,
        thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def axicon(
    field: Field,
    n_axicon: ScalarLike,
    slope_angle: ScalarLike,
    n_medium: ScalarLike = 1.0,
) -> Field:
    phase = axicon_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_axicon,
        slope_angle,
        n_medium,
    )
    field = phase_change(field, phase)
    return field

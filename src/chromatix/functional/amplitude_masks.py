import jax.numpy as jnp
from chex import assert_rank
from jax import Array
from jax.scipy.ndimage import map_coordinates

from chromatix import Field

from ..ops.quantization import binarize, quantize
from ..utils.shapes import _broadcast_2d_to_spatial

__all__ = ["amplitude_change", "interpolated_amplitude_change"]


def amplitude_change(field: Field, amplitude: Array) -> Field:
    """
    Perturbs ``field`` by ``amplitude``.

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude to apply.
    """
    amplitude = _broadcast_2d_to_spatial(amplitude, field.ndim)
    assert_rank(
        amplitude,
        field.ndim,
        custom_message="Amplitude must have same ndim as incoming ``Field``.",
    )
    return field * amplitude.astype(jnp.complex64)


def interpolated_amplitude_change(
    field: Field,
    amplitude: Array,
    binary: bool = False,
    num_bits: int | None = None,
    interpolation_order: int = 0,
) -> Field:
    if binary:
        amplitude = binarize(amplitude)
    if num_bits is not None:
        amplitude = quantize(amplitude, num_bits)
    field_pixel_grid = jnp.meshgrid(
        jnp.linspace(0, amplitude.shape[0] - 1, num=field.spatial_shape[0]) + 0.5,
        jnp.linspace(0, amplitude.shape[1] - 1, num=field.spatial_shape[1]) + 0.5,
        indexing="ij",
    )
    amplitude = map_coordinates(amplitude, field_pixel_grid, interpolation_order)
    return amplitude_change(field, amplitude)

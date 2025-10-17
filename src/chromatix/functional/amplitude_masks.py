import jax.numpy as jnp
from chex import assert_equal_shape
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

from chromatix import Field

from ..ops.quantization import binarize, quantize
from ..utils.shapes import _broadcast_2d_to_spatial

__all__ = ["amplitude_change", "interpolated_amplitude_change"]


def amplitude_change(field: Field, amplitude: Float[Array, "h w"]) -> Field:
    """
    Perturbs ``field`` by ``amplitude``, i.e. ``field * amplitude``.

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude mask to apply as a 2D array of amplitude values
            of shape ``(height width)``.
    """
    if amplitude.ndim != field.ndim:
        amplitude = _broadcast_2d_to_spatial(amplitude, field.spatial_dims)
    assert_equal_shape(
        (amplitude, field.u),
        dims=field.spatial_dims,
        custom_message=(
            "Amplitude must have same height and width as incoming ``Field``."
        ),
    )
    return field * amplitude.astype(jnp.complex64)


def interpolated_amplitude_change(
    field: Field,
    amplitude: Float[Array, "h w"],
    binary: bool = False,
    num_bits: int | None = None,
    interpolation_order: int = 0,
) -> Field:
    """
    Perturbs ``field`` by ``amplitude``, i.e. ``field * amplitude``, but with
    an interpolation of the amplitude mask to the number of samples (pixels)
    in the incoming field. This is useful when the amplitude mask itself has a
    different number of pixels than the field (typically the amplitude mask has
    fewer pixels than the finely-sampled field, e.g. when simulating a digital
    micromirror device). Assumes that the amplitude mask should be interpolated
    to the entire extent of the incoming field, i.e. that the extent of the
    field is exactly that of the amplitude mask.

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude mask to apply as a 2D array of amplitude values
            of shape ``(height width)``.
        binary: Whether the amplitude values should be binarized (0 or 1) or
            not. Defaults to ``False``, in which case the amplitude values are
            not binarized.
        num_bits: An integer value representing the number of bits to which the
            amplitude mask should be quantized, i.e. only `2**num_bits` values
            will be allowed in the amplitude mask. This quantization occurs
            within the full range of values in the provided ``amplitude``.
        interpolation_order: An integer defining the order of the interpolation
            of the amplitude values to the number of pixels of the field. Set to
            `0` by default for nearest-neighbor interpolation, but can also be
            set to 1 for bilinear interpolation. No higher order interpolation
            is supported for now.
    """
    if binary:
        amplitude = binarize(amplitude)
    elif num_bits is not None:
        amplitude = quantize(amplitude, num_bits)
    field_pixel_grid = jnp.meshgrid(
        jnp.linspace(0, amplitude.shape[0] - 1, num=field.spatial_shape[0]) + 0.5,
        jnp.linspace(0, amplitude.shape[1] - 1, num=field.spatial_shape[1]) + 0.5,
        indexing="ij",
    )
    amplitude = map_coordinates(amplitude, field_pixel_grid, interpolation_order)
    return amplitude_change(field, amplitude)

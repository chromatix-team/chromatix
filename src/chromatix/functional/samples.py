from typing import Optional, Union
import jax.numpy as jnp
from chex import Array, assert_equal_shape

from ..field import Field
from .propagation import exact_propagate, kernel_propagate, compute_exact_propagator
from chromatix.utils.shapes import _broadcast_2d_to_spatial
from chromatix.ops.field import pad, crop
from einops import rearrange


def thin_sample(
    field: Field, absorption: Array, dn: Array, thickness: Union[float, Array]
) -> Field:
    """
    Perturbs a ``field`` as if it went through a thin sample object with a given
    ``absorption`` and refractive index change ``dn`` and of a given
    ``thickness`` in micrometres.

    The sample is supposed to follow the thin sample approximation, so the sample
    perturbation is calculated as
    ``exp(1j * 2*pi * (dn + 1j*absorption) * thickness / lambda)``.

    Returns a ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption per micrometre defined as [B H W C] array
        dn: sample refractive index change [B H W C] array
        thickness: thickness at each sample location [B H W C] array
    """
    absorption = _broadcast_2d_to_spatial(absorption, field.ndim)
    dn = _broadcast_2d_to_spatial(dn, field.ndim)
    sample = jnp.exp(
        1j * 2 * jnp.pi * (dn + 1j * absorption) * thickness / field.spectrum
    )
    return field * sample


def multislice_thick_sample(
    field: Field,
    absorption_stack: Array,
    dn_stack: Array,
    n: float,
    thickness_per_slice: float,
    N_pad: int,
    propagator: Optional[Array] = None,
    kykx: Array = jnp.zeros((2,)),
) -> Field:
    """
    Perturbs incoming ``Field`` as if it went through a thick sample. The
    thick sample is modeled as being made of many thin slices each of a given
    thickness. The ``absorption_stack`` and ``dn_stack`` contain the absorbance
    and phase delay of each sample slice.

    A propagator that propagates the field through each slice can be provided.
    By default, a propagtor is calculated inside the function. After passing
    through all slices, the field is propagated backwards to the center of
    the stack.

    Returns a ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption_stack: The sample absorption per micrometre for each slice
            defined as [D H W] array, where D is the total number of slices
        dn_stack: sample refractive index change for each slice [D H W] array.
            Shape should be the same that for ``absorption_stack``.
        thickness_per_slice: thickness of each slice
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a `jax` ``Array``, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            calculate the padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    assert_equal_shape(absorption_stack, dn_stack)
    absorption_stack = rearrange(absorption_stack, "b h w -> b 1 h w 1")
    dn_stack = rearrange(dn_stack, "b h w -> b 1 h w 1")

    # Padding field
    field = pad(field, N_pad)

    # NOTE(ac+dd): Unrolling this loop is much faster than ``jax.scan``-likes.
    if propagator is None:
        propagator = compute_exact_propagator(field, thickness_per_slice, n, kykx)

    for absorption, dn in zip(absorption_stack, dn_stack):
        field = thin_sample(field, absorption, dn, thickness_per_slice)
        field = kernel_propagate(field, propagator)

    # Propagate field backwards to the middle
    # TODO(dd): Allow choosing how far back we propagate here
    half_stack_thickness = thickness_per_slice * absorption_stack.shape[0] / 2
    field = exact_propagate(field, z=-half_stack_thickness, n=n, kykx=kykx, N_pad=0)

    # Cropping output field
    return crop(field, N_pad)

from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, assert_rank
from einops import rearrange

from ..field import Field
from ..ops.fft import fft, fftshift, ifft, ifftshift
from ..utils import center_crop, center_pad
from .propagation import exact_propagate, calculate_exact_kernel


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
    assert_rank(
        absorption, 4, custom_message="Absorption must be array of shape [1 H W 1]"
    )
    assert_rank(
        dn, 4, custom_message="Refractive index must be array of shape [1 H W 1]"
    )

    sample_func = jnp.exp(
        1j * 2 * jnp.pi * (dn + 1j * absorption) * thickness / field.spectrum
    )

    return field * sample_func


def multislice_thick_sample(
    field: Field,
    absorption_stack: Array,
    dn_stack: Array,
    n: float,
    thickness_per_slice: float,
    N_pad: int,
    propagator: Array = None,
    kykx: Array = jnp.zeros((2,)),
    loop_axis: Optional[int] = None,
) -> Field:
    """
    Perturbs a ``field`` as if it went through a thick sample. The thick sample is
    modelled as being made of many thin slices each of a given thickness. The
    ``absorption_stack`` and ``dn_stack`` contain the absorbance and phase delay of each
     sample slice.

    A propagator that propagates the field through each slice can be provided. By
    default, a propagtor is calculated inside the function. After passing through all
    slices, the field is propagated backwards to the center of the stack.

    Returns a ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption_stack: The sample absorption per micrometre for each slice defined as
            [D H W] array, where D is the total number of slices
        dn_stack: sample refractive index change for each slice [D H W] array. Shape
            should be the same that for ``absorption_stack``.
        thickness_per_slice: thickness of each slice
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise a
            ConcretizationError will arise when traced!). Use padding calculator
            utilities from ``chromatix.functional.propagation`` to calculate the
            padding.
        kykx: If provided, defines the orientation of the propagation. Should be an
            array of shape `[2,]` in the format [ky, kx].
    """
    assert (
        absorption_stack.shape == dn_stack.shape
    ), "Absorption stack and phase delay stack should be of the same shape!"

    if propagator is None:
        # Calculating propagator
        propagator = calculate_exact_kernel(
            field.u.shape, field.dx, field.spectrum, thickness_per_slice, n, N_pad, kykx
        )

    def _update_per_slice(i: int, field: Field):
        # Propagate the field
        absorption = (absorption_stack[i])[jnp.newaxis, :, :, jnp.newaxis]
        dn = (dn_stack[i])[jnp.newaxis, :, :, jnp.newaxis]

        field = thin_sample(field, absorption, dn, thickness_per_slice)
        # Propagating field
        u = center_pad(field.u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        u = ifft(fft(u, loop_axis) * propagator, loop_axis)

        # Cropping output field
        u = center_crop(u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        field = field.replace(u=u)
        return field

    # python loop unrolling is the fastest method
    for i in range(absorption_stack.shape[0]):
        field = _update_per_slice(i, field)

    # propagate field backwards to the middle
    half_stack_thikness = thickness_per_slice * absorption_stack.shape[0] / 2

    field = exact_propagate(
        field,
        z=-half_stack_thikness,
        n=n,
        kykx=kykx,
        mode="same",
        N_pad=N_pad,
        cval=0,
    )
    return field

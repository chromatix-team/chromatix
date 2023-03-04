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
    propagator: Array,
    kykx: Array = jnp.zeros((2,)),
    loop_axis: Optional[int] = None,
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
    # Calculating propagator
    # propagator = calculate_exact_kernel(
    #     field.u.shape, field.dx, field.spectrum, thickness_per_slice, n, N_pad, kykx
    # )

    def _update_per_slice(field: Field, slice: Tuple[Array, Array]):
        absorption, dn = slice
        # Propagate the field
        absorption = (absorption)[jnp.newaxis, :, :, jnp.newaxis]
        dn = (dn)[jnp.newaxis, :, :, jnp.newaxis]

        field = thin_sample(field, absorption, dn, thickness_per_slice)
        # Propagating field
        u = center_pad(field.u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        u = ifft(fft(u, loop_axis) * propagator, loop_axis)

        # Cropping output field
        u = center_crop(u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        field = field.replace(u=u)
        return field, None

    # Reshape to have scan axis 0
    scan = jax.checkpoint(
        lambda field, xs: jax.lax.scan(
            jax.checkpoint(_update_per_slice),
            field,
            xs,
            length=absorption_stack.shape[0],
        )[0]
    )
    field = scan(field, (absorption_stack, dn_stack))
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

from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from chex import Array, assert_rank
from einops import rearrange
from ..field import Field


def thin_sample(field: Field, absorption: Array, dn: Array, thickness: Array) -> Field:
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
    assert_rank(
        thickness,
        4,
        custom_message="Thickness must be array of shape [1 H W 1]",
    )

    sample_func = jnp.exp(
        1j * 2 * jnp.pi * (dn + 1j * absorption) * thickness / field.spectrum
    )

    return field * sample_func


def jones_sample(field: Field, absorption: Array, dn: Array) -> Field:
    """
    Perturbs a ``field`` as if it went through a thin sample object with a given
    ``absorption`` and refractive index change ``dn`` and of a given
    ``thickness`` in micrometres using Jones Matrix calculation

    The Jones matrix Suppose that a monochromatic plane wave of light is travelling
    in the positive z-direction, with angular frequency ω and wave vector k = (0,0,k),
    where k = 2pi/wavelength. We ignore the incoming field in z direction.

    The sample is supposed to follow the thin sample approximation, so the sample
    perturbation is calculated for each component in Jones Matrix
    ``exp(1j * 2*pi * (dn + 1j*absorption) * thickness / lambda)``.

    Returns a ``Field`` containing x y component with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption per micrometre defined as [B 2 2 H W C] array
        The
        dn: sample refractive index change [B 2 2 H W C] array
        thickness: optional, default is 1.
            thickness at each sample location [B 2 2 H W C] array
    """

    assert_rank(
        absorption, 6, custom_message="Absorption must be array of shape [1 2 2 H W 1]"
    )
    assert_rank(
        dn, 6, custom_message="Refractive index must be array of shape [1 2 2 H W 1]"
    )

    # Thickness is the same for four elements in Jones Matrix
    sample_jones = jnp.exp(1j * 2 * jnp.pi * (dn + 1j * absorption) / field.spectrum)
    sample_jones = sample_jones[::-1, ::-1]

    u = jnp.einsum(
        "ijklmn, ijlmn -> ijlmn", sample_jones, field.u[:, 1:3, :, :, :]
    )  # the field is in y-x order
    # assume the light travel in z direction, therefore, Ez = 0
    u = jnp.concatenate((jnp.zeros((1, 1, u.shape[-3], u.shape[-2], 1)), u), axis=1)

    return field.replace(u=u)

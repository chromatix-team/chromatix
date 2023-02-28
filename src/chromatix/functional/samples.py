from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from chex import Array, assert_rank

from ..field import Field


def thin_sample(field: Field, absorption: Array, dn: Array, thickness: Array) -> Field:
    """
    Perturbs a ``field`` as if it went through a thin sample object with a given
    ``absorption`` and refractive index change ``dn`` and of a given
    ``thickness`` in micrometres.

    The sample is supposed to follow the thin sample approximation, so the sample
    perturbation is calculated as
    ``absorption * exp(1j * 2*pi * dn * thickness / lambda)``.

    Returns a ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption defined as [B H W C] array
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

    sample_func = absorption * jnp.exp(
        1j * 2 * jnp.pi * dn * thickness / field.spectrum
    )

    return field * sample_func

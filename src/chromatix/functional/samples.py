from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from chex import Array, assert_rank

from ..field import Field
from ..ops.fft import optical_fft
from .pupils import circular_pupil


def thin_sample(
    field: Field, absorption: Array, dn: Array, optical_thickness: Array
) -> Field:
    """
    Perturbs a ``field`` as if it went through a thin sample object with a given
    ``absorption`` and refractive index change ``dn`` and of a given 
    ``optical_thickness``.

    The sample is supposed to follow the thin sample approximation, so the sample
    perturbation is calculated as 
    ``absorption * exp{1j * dn * optical_thickness / lambda}``.

    Returns a ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption
        dn: sample refractive index change times 2 pi
        optical_thickness: optical thickness at each sample location
    """
    assert_rank(
        absorption, 4, custom_message="Absorption must be array of shape [1 H W 1]"
    )
    assert_rank(
        dn, 4, custom_message="Refractive index must be array of shape [1 H W 1]"
    )
    assert_rank(
        optical_thickness,
        4,
        custom_message="Thickness must be array of shape [1 H W 1]",
    )

    sample_func = absorption * jnp.exp(1j * dn * optical_thickness / field.spectrum)

    return field * sample_func

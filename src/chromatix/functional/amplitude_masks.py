import jax.numpy as jnp
from chex import Array, assert_rank

from ..field import Field

__all__ = [
    "amplitude_change"
]

# Field function
def amplitude_change(field: Field, amplitude: Array) -> Field:
    """
    Perturbs ``field`` by ``amplitude`` (given in binary numbers).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude to apply.
    """
    assert_rank(amplitude, 4, custom_message="Phase must be array of shape [1 H W 1]")
    return field * amplitude.astype(jnp.complex64)



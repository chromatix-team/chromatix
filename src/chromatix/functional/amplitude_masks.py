import jax.numpy as jnp
from chex import Array, assert_rank
from ..field import Field

__all__ = ["amplitude_change"]


def amplitude_change(field: Field, amplitude: Array) -> Field:
    """
    Perturbs ``field`` by ``amplitude`` (given in binary numbers).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude to apply.
    """
    assert_rank(
        amplitude,
        field.ndim,
        custom_message="Amplitude must have same number of dimensions as incoming ``Field``.",
    )
    return field * amplitude.astype(jnp.complex64)

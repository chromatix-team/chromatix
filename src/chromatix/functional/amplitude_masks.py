import jax.numpy as jnp
from chex import Array
from chromatix.utils.shapes import _broadcast_2d_to_spatial
from ..field import Field

__all__ = ["amplitude_change"]


# Field function
def amplitude_change(field: Field, amplitude: Array) -> Field:
    """
    Perturbs ``field`` by ``amplitude`` (given in binary numbers).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude to apply.
    """
    amplitude = _broadcast_2d_to_spatial(amplitude, field.ndim)
    return field * amplitude.astype(jnp.complex64)
